"""Qiskit engine transpiler implementation.

This module provides QiskitTranspiler for converting Qamomile QKernels
into Qiskit QuantumCircuits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, cast

if TYPE_CHECKING:
    import qamomile.observable as qm_o
    from qiskit import QuantumCircuit

from qamomile.circuit.transpiler.circuit_ir import (
    CircuitEngineEmitPass,
    CompilationPolicy,
)
from qamomile.circuit.transpiler.executable import (
    ParameterMetadata,
    QuantumExecutor,
)
from qamomile.circuit.transpiler.execution_capability import ExecutionCapabilities
from qamomile.circuit.transpiler.execution_handle import (
    ExecutionHandle,
    ExecutionReference,
)
from qamomile.circuit.transpiler.execution_request import EstimateRequest, SampleRequest
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.circuit.transpiler.passes.separate import SegmentationPass
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.qiskit.execution import QiskitExecutionOptions
from qamomile.qiskit.materializer import QiskitMaterializer
from qamomile.qiskit.runtime import _RuntimeExecutor


class QiskitExecutor(QuantumExecutor["QuantumCircuit"]):
    """Execute Qiskit circuits locally or on a selected IBM Quantum backend.

    With no backend, use AerSimulator or BasicSimulator. Named backends are
    resolved through QiskitRuntimeService using the supplied credentials or
    the SDK's saved account. IBMBackend objects select Runtime automatically.
    Credentials are passed to the SDK without saving an account to disk.

    Args:
        backend (Any): Qiskit backend object, IBM backend name, or ``None``
            for the default local simulator.
        estimator (Any): Optional local expectation estimator. Defaults to
            ``None`` for StatevectorEstimator; unavailable with Runtime.
        api_key (str | None): IBM API key for a named backend. Must be supplied
            with ``instance_crn``. Defaults to the SDK's saved account.
        instance_crn (str | None): IBM instance CRN paired with ``api_key``.
            Defaults to the SDK's configured instance.
        mode (Any): Caller-owned Runtime Session or Batch paired with a backend
            object. A backend object can also select Runtime local testing mode.
            Defaults to None; unavailable with a backend name.
        options (QiskitExecutionOptions | None): Qamomile-owned Runtime settings.
            Defaults to None; cannot be combined with ``sampler_options`` or
            ``estimator_options``.
        sampler_options (Any): Runtime sampler options. Defaults to None.
        estimator_options (Any): Runtime estimator options. Defaults to None.
        pass_manager (Any): Runtime target pass manager. Defaults to None.
        service (Any): Existing Runtime service for named backend lookup or
            job restoration. Cannot be combined with explicit credentials.

    Raises:
        TypeError: If ``options`` is not a QiskitExecutionOptions instance.
        ValueError: If credentials are incomplete, arguments conflict, or
            Runtime options are supplied for local execution.
        ImportError: If IBM execution is requested without its SDK extra.
        QiskitBackendNotFoundError: If the named backend cannot be found for
            the selected account and instance.
        Exception: If SDK authentication, lookup, or setup fails.

    Example:
        executor = QiskitExecutor()  # Uses AerSimulator when available
        counts = executor.execute(circuit, shots=1000)
        executor = QiskitExecutor(
            backend="your_backend_name",
            api_key=api_key,
            instance_crn=instance_crn,
        )
        job = executable.sample(executor, shots=1024)
    """

    def __init__(
        self,
        backend: Any = None,
        estimator: Any = None,
        *,
        api_key: str | None = None,
        instance_crn: str | None = None,
        mode: Any = None,
        options: QiskitExecutionOptions | None = None,
        sampler_options: Any = None,
        estimator_options: Any = None,
        pass_manager: Any = None,
        service: Any = None,
    ) -> None:
        """Select local execution or authenticate a named IBM backend.

        Args:
            backend (Any): Qiskit backend object or IBM device name. Defaults
                to a local simulator when None.
            estimator (Any): Local expectation estimator, or None for defaults.
            api_key (str | None): IBM API key, paired with ``instance_crn``
                for a named backend. Defaults to None for the saved account.
            instance_crn (str | None): IBM instance CRN paired with ``api_key``.
                Defaults to None for the SDK's configured instance.
            mode (Any): Runtime Session, Batch, or local testing backend paired
                with a backend object. Defaults to None; not used with a name.
            options (QiskitExecutionOptions | None): Qamomile-owned Runtime
                settings. Defaults to None; cannot be combined with direct
                sampler or estimator options.
            sampler_options (Any): Runtime sampler options, or None.
            estimator_options (Any): Runtime estimator options, or None.
            pass_manager (Any): Runtime hardware compilation pass manager,
                or None for the preset at optimization level one.
            service (Any): Existing Runtime service for lookup or restoration,
                or None. Cannot be combined with explicit credentials.

        Raises:
            TypeError: If ``options`` is not a QiskitExecutionOptions instance.
            ValueError: If credentials are incomplete, arguments conflict, or
                Runtime options are supplied for local execution.
            ImportError: If IBM execution is requested without its SDK extra.
            QiskitBackendNotFoundError: If the named backend cannot be found
                for the selected account and instance.
            Exception: If SDK authentication, lookup, or setup fails.
        """
        if options is not None:
            if not isinstance(options, QiskitExecutionOptions):
                raise TypeError("options must be a QiskitExecutionOptions instance")
            if sampler_options is not None or estimator_options is not None:
                raise ValueError(
                    "options cannot be combined with sampler_options or estimator_options"
                )
            sampler_options = options.sampler_kwargs()
            estimator_options = options.estimator_kwargs()
        _validate_runtime_credentials(backend, api_key, instance_crn, service)
        named_backend = isinstance(backend, str)
        if named_backend:
            if mode is not None:
                raise ValueError("Use a backend object with a Runtime mode")
            if estimator is not None:
                raise ValueError("Use estimator_options with an IBM Runtime backend")
            backend, service = _resolve_ibm_backend(
                backend, api_key, instance_crn, service
            )
        runtime_requested = (
            named_backend or mode is not None or _is_ibm_backend(backend)
        )
        runtime_options = (sampler_options, estimator_options, pass_manager, service)
        if not runtime_requested and any(
            option is not None for option in runtime_options
        ):
            raise ValueError(
                "Runtime options require an IBM backend or an explicit mode"
            )
        if mode is not None and backend is None:
            raise ValueError("A Runtime mode requires an explicit backend")
        if runtime_requested and estimator is not None:
            raise ValueError("Use estimator_options with an IBM Runtime backend")
        self.backend = backend
        self._estimator = estimator
        self._runtime = (
            _RuntimeExecutor(
                backend,
                mode=mode,
                sampler_options=sampler_options,
                estimator_options=estimator_options,
                pass_manager=pass_manager,
                service=service,
            )
            if runtime_requested
            else None
        )

        if self.backend is None:
            try:
                from qiskit_aer import AerSimulator

                self.backend = AerSimulator()
            except ImportError:
                try:
                    from qiskit.providers.basic_provider import BasicSimulator

                    self.backend = BasicSimulator()
                except ImportError:
                    pass

    @property
    def capabilities(self) -> ExecutionCapabilities:
        """Describe execution features of the selected local or IBM backend.

        Returns:
            ExecutionCapabilities: Selected adapter's lifecycle and accuracy
                support, preserving local executor defaults.
        """
        if self._runtime is not None:
            return self._runtime.capabilities
        return super().capabilities

    def submit_sample(
        self, request: SampleRequest[QuantumCircuit]
    ) -> ExecutionHandle[dict[str, int]]:
        """Submit samples through the selected execution adapter.

        Args:
            request (SampleRequest[QuantumCircuit]): Circuit, bindings, and shots.

        Returns:
            ExecutionHandle[dict[str, int]]: Immediate local result or lazy IBM
                job handle.

        Raises:
            Exception: If request validation, compilation, or submission fails.
        """
        if self._runtime is not None:
            return self._runtime.submit_sample(request)
        return super().submit_sample(request)

    def submit_estimate(
        self, request: EstimateRequest[QuantumCircuit]
    ) -> ExecutionHandle[float]:
        """Submit an expectation through the selected execution adapter.

        Args:
            request (EstimateRequest[QuantumCircuit]): Circuit, observable,
                and optional accuracy policy.

        Returns:
            ExecutionHandle[float]: Immediate local result or lazy IBM job.

        Raises:
            NotImplementedError: If the selected adapter rejects the accuracy.
            Exception: If validation, compilation, or submission fails.
        """
        if self._runtime is not None:
            return self._runtime.submit_estimate(request)
        return super().submit_estimate(request)

    def restore(self, reference: ExecutionReference) -> ExecutionHandle[Any]:
        """Reconnect to an IBM job using the selected backend's service.

        Args:
            reference (ExecutionReference): Previously saved execution reference.

        Returns:
            ExecutionHandle[Any]: Restored IBM sample or expectation handle.

        Raises:
            NotImplementedError: If the selected backend has no restoration.
            ValueError: If the reference targets another provider or backend.
            Exception: If the SDK cannot retrieve the job.
        """
        if self._runtime is not None:
            return self._runtime.restore(reference)
        return super().restore(reference)

    def execute(self, circuit: "QuantumCircuit", shots: int) -> dict[str, int]:
        """Execute circuit and return bitstring counts.

        Args:
            circuit (QuantumCircuit): Qiskit circuit to execute.
            shots (int): Number of measurement shots.

        Returns:
            dict[str, int]: Native dictionary mapping bitstrings to counts,
                without SDK-specific result metadata. A circuit without quantum
                or classical bits returns ``{"": shots}``.

        Raises:
            RuntimeError: If no Qiskit backend is available for execution, or
                if Aer would still receive an empty-parameter multiplexer after
                the workaround decomposition.
            Exception: If IBM compilation, submission, or execution fails.
        """
        if self._runtime is not None:
            return self._runtime.execute(circuit, shots)
        if circuit.num_qubits == 0 and circuit.num_clbits == 0:
            return {"": shots}

        from qiskit import transpile

        if self.backend is None:
            raise RuntimeError("No backend available for execution")

        circuit_with_meas = self._ensure_measurements(circuit)
        transpiled = transpile(circuit_with_meas, self.backend)
        if type(self.backend).__module__.startswith("qiskit_aer."):
            transpiled = _decompose_empty_parameter_multiplexers(
                transpiled, self.backend
            )
        job = self.backend.run(transpiled, shots=shots)
        return dict(job.result().get_counts())

    def bind_parameters(
        self,
        circuit: "QuantumCircuit",
        bindings: dict[str, Any],
        parameter_metadata: ParameterMetadata,
    ) -> "QuantumCircuit":
        """Bind parameter values to the Qiskit circuit.

        Args:
            circuit (QuantumCircuit): Parameterized circuit.
            bindings (dict[str, Any]): Flattened runtime parameter values.
            parameter_metadata (ParameterMetadata): Backend parameter mapping.

        Returns:
            QuantumCircuit: New circuit with parameters bound.

        Raises:
            ValueError: If required Runtime values are missing.
            Exception: If Qiskit rejects a parameter assignment.
        """
        if self._runtime is not None:
            return self._runtime.bind_parameters(circuit, bindings, parameter_metadata)
        qiskit_bindings = {}
        for param_info in parameter_metadata.parameters:
            if param_info.name in bindings:
                qiskit_bindings[param_info.engine_param] = bindings[param_info.name]

        return circuit.assign_parameters(qiskit_bindings)

    def estimate(
        self,
        circuit: "QuantumCircuit",
        hamiltonian: "qm_o.Hamiltonian",
        params: Sequence[float] | None = None,
    ) -> float:
        """Estimate the expectation value of a Hamiltonian.

        Args:
            circuit (QuantumCircuit): State preparation ansatz.
            hamiltonian (qm_o.Hamiltonian): Observable to measure.
            params (Sequence[float] | None): Optional values in Qiskit parameter
                order. Defaults to None for an already bound circuit.

        Returns:
            float: Estimated expectation value.

        Raises:
            Exception: If estimator setup, compilation, or execution fails.
        """
        if self._runtime is not None:
            return self._runtime.estimate(circuit, hamiltonian, params)
        if self._estimator is None:
            # Create default Qiskit Estimator
            try:
                from qiskit.primitives import StatevectorEstimator

                self._estimator = StatevectorEstimator()
            except ImportError:
                try:
                    # Fallback for older Qiskit versions
                    from qiskit.primitives import (
                        Estimator,  # type: ignore[attr-defined]
                    )

                    self._estimator = Estimator()
                except ImportError:
                    from qiskit_aer.primitives import Estimator

                    self._estimator = Estimator()

        # Convert Hamiltonian to SparsePauliOp
        from qamomile.qiskit.observable import hamiltonian_to_sparse_pauli_op

        sparse_pauli_op = hamiltonian_to_sparse_pauli_op(hamiltonian)

        # Run estimation
        if params is not None:
            param_values = list(params)
        else:
            param_values = []

        # Check if this is V1 or V2 interface
        # V2 interface (new): run([(circuit, observable, params)])
        # V1 interface (old): run(circuits, observables, parameter_values)
        estimator_run = cast(Any, self._estimator).run
        try:
            # Try V2 interface first
            job = estimator_run([(circuit, sparse_pauli_op, param_values)])
            result = job.result()
            return float(result[0].data.evs)
        except (TypeError, AttributeError):
            # Fall back to V1 interface
            job = estimator_run(
                [circuit],
                [sparse_pauli_op],
                [param_values] if param_values else None,
            )
            result = job.result()
            return float(result.values[0])

    def _ensure_measurements(self, circuit: "QuantumCircuit") -> "QuantumCircuit":
        """Add measurements to a copy when there are no classical bits.

        Args:
            circuit (QuantumCircuit): Local circuit to sample.

        Returns:
            QuantumCircuit: Original circuit or a copy measuring all qubits.
        """
        if circuit.num_clbits > 0:
            return circuit

        circuit_copy = circuit.copy()
        circuit_copy.measure_all()
        return circuit_copy


def _validate_runtime_credentials(
    backend: Any,
    api_key: str | None,
    instance_crn: str | None,
    service: Any,
) -> None:
    """Validate named backend and credential combinations before SDK access.

    Args:
        backend (Any): Local backend, IBM backend name, or None.
        api_key (str | None): Explicit IBM API key, or None.
        instance_crn (str | None): Explicit instance CRN, or None.
        service (Any): Caller-configured Runtime service, or None.

    Raises:
        ValueError: If the name or credentials are blank, the credential pair
            is incomplete, or credentials conflict with other configuration.
    """
    if isinstance(backend, str) and not backend.strip():
        raise ValueError("IBM backend name must not be empty")
    for name, value in (("api_key", api_key), ("instance_crn", instance_crn)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be a non-empty string")
    if (api_key is None) != (instance_crn is None):
        raise ValueError("api_key and instance_crn must be supplied together")
    if api_key is not None:
        if not isinstance(backend, str):
            raise ValueError("Explicit IBM credentials require a backend name")
        if service is not None:
            raise ValueError("Use either explicit IBM credentials or a service")


def _resolve_ibm_backend(
    name: str,
    api_key: str | None,
    instance_crn: str | None,
    service: Any,
) -> tuple[Any, Any]:
    """Resolve a backend through the account and instance scoped SDK lookup.

    Args:
        name (str): IBM backend name selected by the caller.
        api_key (str | None): IBM API key, or None for saved credentials.
        instance_crn (str | None): Instance CRN, or None for SDK configuration.
        service (Any): Existing Runtime service, or None to create one.

    Returns:
        tuple[Any, Any]: Authorized IBM backend and its Runtime service.

    Raises:
        ImportError: If the optional Runtime SDK is unavailable.
        QiskitBackendNotFoundError: If the selected account and instance have
            no matching backend, or the SDK cannot discover the backend list.
        Exception: If SDK authentication or backend discovery fails.
    """
    if service is None:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
        except ImportError as error:
            raise ImportError(
                'IBM execution requires pip install "qamomile[qiskit]"'
            ) from error
        kwargs: dict[str, Any] = {"channel": "ibm_quantum_platform"}
        if api_key is not None:
            kwargs["token"] = api_key
            kwargs["instance"] = instance_crn
        service = QiskitRuntimeService(**kwargs)
    lookup = {} if instance_crn is None else {"instance": instance_crn}
    return service.backend(name, **lookup), service


def _is_ibm_backend(backend: Any) -> bool:
    """Recognize an IBM Runtime backend without requiring the optional SDK.

    Args:
        backend (Any): Backend object to classify, or None.

    Returns:
        bool: Whether the object is an IBMBackend instance.
    """
    if backend is None:
        return False
    try:
        from qiskit_ibm_runtime import IBMBackend
    except ImportError:
        return False
    return isinstance(backend, IBMBackend)


class QiskitTranspiler(Transpiler["QuantumCircuit"]):
    """Qiskit engine transpiler.

    Converts Qamomile QKernels into Qiskit QuantumCircuits.

    Args:
        use_native_composite (bool): Whether to prefer native Qiskit library
            realizations for semantic composites such as QFT/IQFT. Defaults
            to ``True``.
        use_native_pauli_evolution (bool): Whether to prefer
            ``PauliEvolutionGate`` over gate gadgets. Defaults to ``True``.

    Example:
        from qamomile.qiskit import QiskitTranspiler
        import qamomile as qm

        @qm.qkernel
        def bell_state(q0: qm.Qubit, q1: qm.Qubit) -> tuple[qm.Bit, qm.Bit]:
            q0 = qm.h(q0)
            q0, q1 = qm.cx(q0, q1)
            return qm.measure(q0), qm.measure(q1)

        transpiler = QiskitTranspiler()
        circuit = transpiler.to_circuit(bell_state)
        print(circuit.draw())
    """

    def __init__(
        self,
        use_native_composite: bool = True,
        use_native_pauli_evolution: bool = True,
    ) -> None:
        """Initialize the Qiskit transpiler.

        Args:
            use_native_composite (bool): Whether to prefer engine-native
                realizations of semantic composites such as QFT, state
                preparation, arithmetic, and multi-controlled X. Defaults to
                ``True``.
            use_native_pauli_evolution (bool): Whether to prefer native Pauli
                evolution over gate gadgets. Defaults to ``True``.
        """
        self._use_native_composite = use_native_composite
        self._use_native_pauli_evolution = use_native_pauli_evolution

    def _create_segmentation_pass(self) -> SegmentationPass:
        """Create the host-orchestrated circuit segmentation pass.

        Returns:
            SegmentationPass: Standard single-quantum-segment planner.
        """
        return SegmentationPass()

    def _create_emit_pass(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> EmitPass["QuantumCircuit"]:
        """Create the capability-driven Qiskit materialization pipeline.

        Args:
            bindings (dict[str, Any] | None): Compile-time bindings. Defaults
                to ``None``.
            parameters (list[str] | None): Runtime parameter names. Defaults
                to ``None``.

        Returns:
            EmitPass[QuantumCircuit]: Circuit lowering, legalization, and
                Qiskit materialization pass.
        """
        return CircuitEngineEmitPass(
            QiskitMaterializer(),
            bindings,
            parameters,
            policy=CompilationPolicy(
                prefer_native_semantic_ops=self._use_native_composite,
                prefer_native_pauli_evolution=self._use_native_pauli_evolution,
            ),
        )

    def executor(  # type: ignore[override]
        self,
        backend: Any = None,
        *,
        estimator: Any = None,
        api_key: str | None = None,
        instance_crn: str | None = None,
        mode: Any = None,
        options: QiskitExecutionOptions | None = None,
        sampler_options: Any = None,
        estimator_options: Any = None,
        pass_manager: Any = None,
        service: Any = None,
    ) -> QiskitExecutor:
        """Create a local or IBM Quantum executor with the same execution API.

        Args:
            backend (Any): Qiskit backend object or IBM backend name. Defaults
                to a local simulator when None.
            estimator (Any): Optional local expectation estimator.
            api_key (str | None): IBM API key for a named backend. Must be
                paired with ``instance_crn``; defaults to saved credentials.
            instance_crn (str | None): Instance CRN paired with ``api_key``.
                Defaults to the SDK's configured instance.
            mode (Any): Caller-owned Runtime Session, Batch, or local testing
                backend paired with a backend object. Defaults to None;
                unavailable with a backend name.
            options (QiskitExecutionOptions | None): Qamomile-owned Runtime
                settings. Defaults to None; cannot be combined with direct
                sampler or estimator options.
            sampler_options (Any): Runtime sampler options, or None.
            estimator_options (Any): Runtime estimator options, or None.
            pass_manager (Any): Runtime hardware compilation pass manager,
                or None for the backend preset.
            service (Any): Existing Runtime service for lookup or restoration,
                or None. Cannot be combined with explicit credentials.

        Returns:
            QiskitExecutor: Executor configured for the selected execution target.

        Raises:
            TypeError: If ``options`` is not a QiskitExecutionOptions instance.
            ValueError: If credentials are incomplete or arguments conflict.
            ImportError: If IBM execution is requested without its SDK extra.
            QiskitBackendNotFoundError: If the selected account and instance
                have no matching backend.
            Exception: If SDK authentication or executor setup fails.

        Example:
            executor = transpiler.executor(
                backend="your_backend_name",
                api_key=api_key,
                instance_crn=instance_crn,
            )
            job = executable.sample(executor, shots=1024)
        """
        return QiskitExecutor(
            backend,
            estimator,
            api_key=api_key,
            instance_crn=instance_crn,
            mode=mode,
            options=options,
            sampler_options=sampler_options,
            estimator_options=estimator_options,
            pass_manager=pass_manager,
            service=service,
        )


def _contains_empty_parameter_multiplexer(circuit: "QuantumCircuit") -> bool:
    """Return whether a circuit contains Aer's unsafe multiplexer form.

    Args:
        circuit (QuantumCircuit): Qiskit circuit to inspect, including any
            nested control-flow blocks reachable through ``ControlFlowOp``.

    Returns:
        bool: True when the circuit contains a ``multiplexer`` instruction
            whose parameter list is empty, otherwise False.
    """
    from qiskit.circuit import ControlFlowOp

    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name == "multiplexer" and not operation.params:
            return True
        if isinstance(operation, ControlFlowOp) and any(
            _contains_empty_parameter_multiplexer(block) for block in operation.blocks
        ):
            return True
    return False


def _decompose_empty_parameter_multiplexers(
    circuit: "QuantumCircuit", backend: Any
) -> "QuantumCircuit":
    """Decompose Aer's unsafe empty-parameter multiplexers before execution.

    Args:
        circuit (QuantumCircuit): Transpiled Qiskit circuit to sanitize.
        backend (Any): Qiskit backend used for the follow-up transpilation.

    Returns:
        QuantumCircuit: The original circuit when no unsafe multiplexer is
            present, otherwise a re-transpiled circuit with matching
            multiplexers decomposed through nested control-flow blocks.

    Raises:
        RuntimeError: If an empty-parameter multiplexer remains after the
            decomposition pass and re-transpilation.
    """
    if not _contains_empty_parameter_multiplexer(circuit):
        return circuit

    from qiskit import transpile
    from qiskit.circuit import ControlFlowOp

    decomposed = transpile(
        circuit.decompose(gates_to_decompose=["multiplexer", ControlFlowOp], reps=4),
        backend,
    )
    if _contains_empty_parameter_multiplexer(decomposed):
        raise RuntimeError(
            "Aer execution would receive an empty-parameter multiplexer that "
            "can crash native assembly."
        )
    return decomposed
