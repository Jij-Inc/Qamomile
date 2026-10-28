"""Execute Qiskit circuits on IBM Quantum through Runtime V2 primitives."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

from qamomile.circuit.transpiler.execution_capability import ExecutionCapabilities
from qamomile.circuit.transpiler.execution_handle import (
    CompletedExecutionHandle,
    ExecutionHandle,
    ExecutionReference,
)
from qamomile.circuit.transpiler.execution_request import (
    CircuitInvocation,
    EstimateRequest,
    SampleRequest,
    TargetPrecision,
)
from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata
from qamomile.circuit.transpiler.quantum_executor import QuantumExecutor
from qamomile.observable.hamiltonian import HERMITIAN_IMAG_ATOL
from qamomile.qiskit.observable import hamiltonian_to_sparse_pauli_op
from qamomile.qiskit.runtime_execution import RuntimeExecutionHandle

if TYPE_CHECKING:
    import qamomile.observable as qm_o
    from qiskit import QuantumCircuit


class _RuntimeExecutor(QuantumExecutor["QuantumCircuit"]):
    """Implement Runtime execution behind the public Qiskit executor.

    Circuits are compiled to the backend ISA before submission, unrolling
    static for-loops when the target does not support them. Runtime
    parameters are bound before that compilation. Session and Batch lifetimes
    remain under the caller's control. The public ``QiskitExecutor`` resolves
    credentials and named backends before constructing this adapter.

    Args:
        backend (Any): Qiskit BackendV2 defining the target ISA and device.
        mode (Any): Runtime Session or Batch for this backend. Defaults to
            job mode on ``backend``.
        sampler_options (Any): Runtime SamplerOptions or option dictionary.
            Defaults to the SDK settings.
        estimator_options (Any): Runtime EstimatorOptions or option dictionary.
            Defaults to the SDK settings.
        pass_manager (Any): Qiskit pass manager targeting ``backend``. Defaults
            to the backend preset at optimization level one.
        service (Any): Runtime service used by ``restore``. Defaults to
            ``backend.service`` when available.

    Raises:
        ImportError: If the optional Runtime dependency is unavailable.
        ValueError: If no backend is supplied or the mode targets a different
            backend.
        Exception: If the SDK rejects the options or execution mode.
    """

    def __init__(
        self,
        backend: Any,
        *,
        mode: Any = None,
        sampler_options: Any = None,
        estimator_options: Any = None,
        pass_manager: Any = None,
        service: Any = None,
    ) -> None:
        """Configure Runtime primitives and hardware compilation.

        Args:
            backend (Any): Explicit Qiskit BackendV2 execution target.
            mode (Any): Caller-owned Runtime Session or Batch, or ``None``.
            sampler_options (Any): Sampler options, or ``None`` for defaults.
            estimator_options (Any): Estimator options, or ``None`` for defaults.
            pass_manager (Any): Target-specific pass manager, or ``None``.
            service (Any): Service for job restoration, or ``None``.

        Raises:
            ImportError: If the optional Runtime dependency is unavailable.
            ValueError: If no backend is supplied or the mode targets a
                different backend.
            Exception: If the SDK rejects the options or execution mode.
        """
        if backend is None:
            raise ValueError("Runtime execution requires an explicit backend")
        try:
            from qiskit_ibm_runtime import EstimatorV2, SamplerV2
        except ImportError as error:
            raise ImportError(
                'IBM execution requires pip install "qamomile[qiskit]"'
            ) from error
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        self.backend = backend
        self._service = (
            service if service is not None else getattr(backend, "service", None)
        )
        self._sampler = SamplerV2(
            mode=backend if mode is None else mode, options=sampler_options
        )
        self._estimator = EstimatorV2(
            mode=backend if mode is None else mode, options=estimator_options
        )
        if mode is not None and self._sampler.backend().name != backend.name:
            raise ValueError("Runtime mode and compilation backend must match")
        self._pass_manager = (
            pass_manager
            if pass_manager is not None
            else generate_preset_pass_manager(backend=backend, optimization_level=1)
        )

    @property
    def capabilities(self) -> ExecutionCapabilities:
        """Describe Runtime submission and accuracy support.

        Returns:
            ExecutionCapabilities: Asynchronous sample and estimate support,
                cancellation, and restoration when a service is available.
        """
        return ExecutionCapabilities(
            supports_async_sampling=True,
            supports_async_estimation=True,
            supports_estimation=True,
            supports_cancellation=True,
            supports_restoration=self._service is not None,
            estimation_accuracy=frozenset({TargetPrecision}),
        )

    def bind_parameters(
        self,
        circuit: QuantumCircuit,
        bindings: dict[str, Any],
        parameter_metadata: ParameterMetadata,
    ) -> QuantumCircuit:
        """Bind Qamomile runtime values without mutating the source circuit.

        Args:
            circuit (QuantumCircuit): Parameterized Qiskit circuit.
            bindings (dict[str, Any]): Flattened runtime values.
            parameter_metadata (ParameterMetadata): Engine parameter mapping.

        Returns:
            QuantumCircuit: A copy with runtime values assigned.

        Raises:
            ValueError: If required runtime values are missing.
            Exception: If Qiskit rejects a parameter assignment.
        """
        missing = {
            info.name for info in parameter_metadata.parameters
        } - bindings.keys()
        if missing:
            raise ValueError(f"Missing runtime parameter values: {sorted(missing)}")
        values = {
            info.engine_param: bindings[info.name]
            for info in parameter_metadata.parameters
        }
        return circuit.assign_parameters(values)

    def execute(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
        """Submit sampling and wait for its counts.

        Args:
            circuit (QuantumCircuit): Bound circuit to execute.
            shots (int): Positive measurement shot count.

        Returns:
            dict[str, int]: Counts ordered from highest to lowest classical bit.

        Raises:
            ValueError: If the request or classical registers are invalid.
            Exception: If compilation, submission, or execution fails.
        """
        invocation = CircuitInvocation(circuit, {}, ParameterMetadata())
        return self.submit_sample(SampleRequest(invocation, shots)).result()

    def submit_sample(
        self, request: SampleRequest[QuantumCircuit]
    ) -> ExecutionHandle[dict[str, int]]:
        """Submit one sampler job without waiting for remote execution.

        Args:
            request (SampleRequest[QuantumCircuit]): Circuit, bindings, and shots.

        Returns:
            ExecutionHandle[dict[str, int]]: Remote handle, or a completed
                empty count for a circuit with no bits.

        Raises:
            ValueError: If parameters remain unbound or a classical bit has
                no register, which Runtime cannot return.
            Exception: If compilation or Runtime submission fails.
        """
        circuit = self.bind_invocation(request.invocation)
        if circuit.num_qubits == 0 and circuit.num_clbits == 0:
            return CompletedExecutionHandle({"": request.shots})
        if circuit.num_clbits == 0:
            circuit = circuit.copy()
            circuit.measure_all()
        isa = self._compile(circuit)
        bit_map = _classical_bit_map(isa)
        native = self._sampler.run([isa], shots=request.shots)
        reference = self._reference(
            native, {"kind": "sample", "bits": json.dumps(bit_map)}
        )
        return RuntimeExecutionHandle(
            native, partial(_sample_counts, bit_map=bit_map), reference
        )

    def estimate(
        self,
        circuit: QuantumCircuit,
        hamiltonian: qm_o.Hamiltonian,
        params: Sequence[float] | None = None,
    ) -> float:
        """Submit an expectation estimate and wait for its value.

        Args:
            circuit (QuantumCircuit): State preparation circuit.
            hamiltonian (qm_o.Hamiltonian): Observable in logical qubit order.
            params (Sequence[float] | None): Values in Qiskit parameter order,
                or ``None`` when the circuit is already bound.

        Returns:
            float: Estimated expectation value.

        Raises:
            ValueError: If the observable or parameter values are invalid.
            Exception: If compilation, submission, or execution fails.
        """
        if params is not None:
            circuit = circuit.assign_parameters(list(params))
        invocation = CircuitInvocation(circuit, {}, ParameterMetadata())
        return self.submit_estimate(EstimateRequest(invocation, hamiltonian)).result()

    def submit_estimate(
        self, request: EstimateRequest[QuantumCircuit]
    ) -> ExecutionHandle[float]:
        """Submit an estimator job with its observable mapped to hardware.

        Args:
            request (EstimateRequest[QuantumCircuit]): Circuit, Hamiltonian,
                and optional absolute target precision.

        Returns:
            ExecutionHandle[float]: Remote expectation handle, or a completed
                value for a constant Hamiltonian.

        Raises:
            NotImplementedError: If accuracy is not TargetPrecision or None.
            ValueError: If parameters remain unbound or the observable refers
                to qubits outside the circuit or has a coefficient whose
                imaginary part exceeds the Hermiticity tolerance.
            Exception: If compilation or Runtime submission fails.
        """
        if request.accuracy is not None and not isinstance(
            request.accuracy, TargetPrecision
        ):
            raise NotImplementedError(
                "Runtime estimation supports TargetPrecision only"
            )
        circuit = self.bind_invocation(request.invocation)
        if request.hamiltonian.num_qubits > circuit.num_qubits:
            raise ValueError("Hamiltonian acts on qubits outside the circuit")
        constant = complex(request.hamiltonian.constant)
        if abs(constant.imag) > HERMITIAN_IMAG_ATOL or any(
            abs(complex(coefficient).imag) > HERMITIAN_IMAG_ATOL
            for coefficient in request.hamiltonian.terms.values()
        ):
            raise ValueError(
                "Runtime expectation values require a Hermitian Hamiltonian"
            )
        if not any(
            complex(coefficient).real
            for coefficient in request.hamiltonian.terms.values()
        ):
            return CompletedExecutionHandle(float(constant.real))
        circuit = circuit.copy()
        circuit.remove_final_measurements()
        observable = hamiltonian_to_sparse_pauli_op(request.hamiltonian)
        # Runtime requires exactly real coefficients, including tolerated roundoff.
        observable.coeffs = observable.coeffs.real
        observable = observable.apply_layout(None, num_qubits=circuit.num_qubits)
        isa = self._compile(circuit)
        observable = observable.apply_layout(isa.layout, num_qubits=isa.num_qubits)
        kwargs = (
            {"precision": request.accuracy.precision}
            if isinstance(request.accuracy, TargetPrecision)
            else {}
        )
        native = self._estimator.run([(isa, observable)], **kwargs)
        return RuntimeExecutionHandle(
            native, _expectation_value, self._reference(native, {"kind": "estimate"})
        )

    def _compile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Compile logical circuits to the target ISA without changing inputs.

        Args:
            circuit (QuantumCircuit): Bound circuit in logical qubit order.

        Returns:
            QuantumCircuit: Compiled hardware circuit with layout metadata.

        Raises:
            ValueError: If runtime parameters remain after compilation.
            Exception: If the SDK cannot lower the circuit to the target ISA.
        """
        from qiskit.transpiler.passes import UnrollForLoops

        if "for_loop" not in self.backend.operation_names:
            circuit = UnrollForLoops()(circuit)
        isa = self._pass_manager.run(circuit)
        if isa.num_parameters:
            raise ValueError("Runtime execution requires all parameters to be bound")
        return isa

    def _reference(
        self, job: Any, context: Mapping[str, str]
    ) -> ExecutionReference | None:
        """Describe a submitted job without credentials or SDK objects.

        Args:
            job (Any): Submitted primitive job.
            context (Mapping[str, str]): Result decoding metadata.

        Returns:
            ExecutionReference | None: Job ID, backend name, and decoding
                context, or ``None`` without a restoration service.

        Raises:
            Exception: If the SDK cannot provide a valid job identifier.
        """
        if self._service is None:
            return None
        return ExecutionReference(
            provider="qiskit-runtime",
            job_ids=(job.job_id(),),
            target=self.backend.name,
            context=context,
        )

    def restore(self, reference: ExecutionReference) -> ExecutionHandle[Any]:
        """Restore a previously submitted job through the configured service.

        Args:
            reference (ExecutionReference): Reference from this executor type.

        Returns:
            ExecutionHandle[Any]: Restored sampling or expectation handle.

        Raises:
            ValueError: If the provider, target, or result context is invalid.
            NotImplementedError: If no restoration service is configured.
            Exception: If the service cannot retrieve the job.
        """
        if (
            reference.provider != "qiskit-runtime"
            or len(reference.job_ids) != 1
            or reference.target != self.backend.name
        ):
            raise ValueError(
                "Reference does not identify a job on this Runtime backend"
            )
        kind = reference.context.get("kind")
        if kind == "sample":
            try:
                bit_map = json.loads(reference.context["bits"])
            except (KeyError, ValueError) as error:
                raise ValueError("Invalid Runtime sampling reference") from error
            if (
                not isinstance(bit_map, list)
                or not bit_map
                or not all(
                    isinstance(entry, list)
                    and len(entry) == 2
                    and isinstance(entry[0], str)
                    and bool(entry[0])
                    and type(entry[1]) is int
                    and entry[1] >= 0
                    for entry in bit_map
                )
            ):
                raise ValueError("Invalid Runtime classical bit mapping")
            decoder = partial(_sample_counts, bit_map=bit_map)
        elif kind == "estimate":
            decoder = _expectation_value
        else:
            raise ValueError("Unknown Runtime result kind")
        if self._service is None:
            raise NotImplementedError("Runtime restoration requires a service")
        native = self._service.job(reference.job_ids[0])
        return RuntimeExecutionHandle(native, decoder, reference)


def _classical_bit_map(circuit: QuantumCircuit) -> list[tuple[str, int]]:
    """Map each classical bit to its Runtime register and register index.

    Args:
        circuit (QuantumCircuit): Compiled circuit with classical registers.

    Returns:
        list[tuple[str, int]]: Register locations in increasing clbit order.

    Raises:
        ValueError: If a classical bit does not belong to a named register.
    """
    bits = []
    for bit in circuit.clbits:
        registers = circuit.find_bit(bit).registers
        if not registers:
            raise ValueError("Runtime sampling requires named classical registers")
        register, index = registers[0]
        bits.append((register.name, index))
    return bits


def _sample_counts(result: Any, *, bit_map: Sequence[Sequence[Any]]) -> dict[str, int]:
    """Decode joint shot data in Qamomile classical bit order.

    Args:
        result (Any): Runtime primitive result containing one sampler PUB.
        bit_map (Sequence[Sequence[Any]]): Register name and index per clbit.

    Returns:
        dict[str, int]: Joint counts with the highest classical bit leftmost.

    Raises:
        ValueError: If the result has an unexpected shape or register width.
        TypeError: If the sampler did not return classified bit data.
    """
    from qiskit.primitives.containers import BitArray

    if len(result) != 1:
        raise ValueError("Expected one Runtime sampling result")
    registers = {}
    for name, index in bit_map:
        data = getattr(result[0].data, name)
        if not isinstance(data, BitArray):
            raise TypeError("Runtime sampling requires classified BitArray results")
        if data.shape or index >= data.num_bits:
            raise ValueError("Unexpected Runtime register shape or width")
        if name not in registers:
            registers[name] = data.get_bitstrings()
    shot_counts = {len(bits) for bits in registers.values()}
    if len(shot_counts) != 1:
        raise ValueError("Runtime registers have inconsistent shot counts")
    shots = shot_counts.pop()
    return dict(
        Counter(
            "".join(
                registers[name][shot][-1 - index] for name, index in reversed(bit_map)
            )
            for shot in range(shots)
        )
    )


def _expectation_value(result: Any) -> float:
    """Decode one scalar expectation from a Runtime estimator result.

    Args:
        result (Any): Primitive result containing one expectation PUB.

    Returns:
        float: Scalar expectation value.

    Raises:
        ValueError: If the result contains multiple PUBs or expectation values.
    """
    if len(result) != 1:
        raise ValueError("Expected one Runtime estimation result")
    return float(result[0].data.evs)
