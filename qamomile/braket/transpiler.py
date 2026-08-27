"""Transpile and execute Qamomile programs with Amazon Braket."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from numbers import Integral
from typing import TYPE_CHECKING, Any, cast

from qamomile.braket.execution import BraketExecutionHandle, BraketExecutionOptions
from qamomile.braket.materializer import BraketMaterializer
from qamomile.circuit.transpiler.circuit_ir import (
    CircuitBackendEmitPass,
    CompilationPolicy,
)
from qamomile.circuit.transpiler.executable import ParameterMetadata, QuantumExecutor
from qamomile.circuit.transpiler.execution_capability import ExecutionCapabilities
from qamomile.circuit.transpiler.execution_handle import (
    CompletedExecutionHandle,
    ExecutionHandle,
    ExecutionReference,
)
from qamomile.circuit.transpiler.execution_request import (
    CircuitInvocation,
    EstimateRequest,
    Exact,
    SampleRequest,
    ShotBased,
    TargetPrecision,
)
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.circuit.transpiler.passes.separate import SegmentationPass
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.observable.hamiltonian import (
    HERMITIAN_IMAG_ATOL,
    PAULI_TERM_ZERO_ATOL,
)

if TYPE_CHECKING:
    import qamomile.observable as qm_o
    from braket.circuits import Circuit  # type: ignore[import-not-found]


class BraketExecutor(QuantumExecutor["Circuit"]):
    """Submit Braket circuits to a local simulator or injected AWS device.

    The default device is created lazily, so importing ``qamomile.braket``
    remains safe when the optional SDK dependency is absent.

    Args:
        device (Any): Braket device exposing ``run`` and optionally
            ``run_batch``. Defaults to ``LocalSimulator``.
        estimation_shots (int): Shots used for expectation values. Zero uses
            exact state-vector expectation on compatible devices. Defaults to
            zero.
        options (BraketExecutionOptions | None): Structured task and batch
            submission policy. Defaults to SDK behavior with no batch retry.
        run_kwargs (Mapping[str, Any] | None): Extra keyword arguments passed
            to every device task. This compatibility argument is deprecated in
            favor of ``options``. Defaults to none.
    """

    def __init__(
        self,
        device: Any = None,
        *,
        estimation_shots: int = 0,
        options: BraketExecutionOptions | None = None,
        run_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the Braket executor.

        Args:
            device (Any): Braket device or compatible test double. Defaults
                to a lazily created local simulator.
            estimation_shots (int): Non-negative expectation task shots.
                Defaults to zero for exact local estimation.
            options (BraketExecutionOptions | None): Structured execution
                options. Defaults to SDK behavior with retry disabled.
            run_kwargs (Mapping[str, Any] | None): Extra task options copied
                for each run. Kept for compatibility; cannot be combined with
                ``options``. Defaults to none.

        Raises:
            ValueError: If ``estimation_shots`` is negative or both option
                forms are supplied.
        """
        if (
            isinstance(estimation_shots, bool)
            or not isinstance(cast(object, estimation_shots), Integral)
            or estimation_shots < 0
        ):
            raise ValueError("estimation_shots must be a non-negative integer")
        if options is not None and run_kwargs is not None:
            raise ValueError("options and run_kwargs cannot be used together")
        self._device = device
        self._estimation_shots = int(estimation_shots)
        self._options = options or _legacy_execution_options(run_kwargs)

    @property
    def device(self) -> Any:
        """Return the configured device, creating a local simulator lazily.

        Returns:
            Any: Braket device used for subsequent tasks.

        Raises:
            ImportError: If the optional Amazon Braket SDK is unavailable.
        """
        if self._device is None:
            try:
                from braket.devices import (  # type: ignore[import-not-found]
                    LocalSimulator,
                )
            except ImportError as error:
                raise ImportError(
                    "Amazon Braket support requires amazon-braket-sdk. "
                    "Install Qamomile with the 'braket' extra."
                ) from error
            self._device = LocalSimulator()
        return self._device

    @property
    def capabilities(self) -> ExecutionCapabilities:
        """Describe lifecycle and estimation features for the target device.

        Returns:
            ExecutionCapabilities: Target-aware Braket execution features.
        """
        device = self.device
        policies: set[type[Exact] | type[ShotBased] | type[TargetPrecision]] = {
            ShotBased
        }
        if not _is_qpu(device):
            policies.add(Exact)
        return ExecutionCapabilities(
            supports_async_sampling=True,
            supports_async_estimation=True,
            supports_estimation=True,
            supports_cancellation=True,
            supports_restoration=getattr(device, "aws_session", None) is not None,
            supports_native_batch=hasattr(device, "run_batch"),
            supports_native_parameter_inputs=True,
            estimation_accuracy=frozenset(policies),
        )

    def execute(self, circuit: "Circuit", shots: int) -> dict[str, int]:
        """Sample a Braket circuit and return Qamomile-ordered counts.

        Args:
            circuit (Circuit): Bound Braket state-preparation circuit.
            shots (int): Number of measurement shots.

        Returns:
            dict[str, int]: Counts with the highest qubit index on the left.
                A zero-qubit circuit returns ``{"": shots}``.

        Raises:
            RuntimeError: If the Braket result omits measurement counts.
        """
        request = SampleRequest(
            CircuitInvocation(circuit, {}, ParameterMetadata()),
            shots,
        )
        return self.submit_sample(request).result()

    def submit_sample(
        self,
        request: SampleRequest["Circuit"],
    ) -> ExecutionHandle[dict[str, int]]:
        """Submit a native Braket sampling task without waiting for results.

        Args:
            request (SampleRequest[Circuit]): Circuit, native parameter inputs,
                and shot count.

        Returns:
            ExecutionHandle[dict[str, int]]: Lazy Braket execution handle.
        """
        circuit = request.invocation.circuit
        if circuit.qubit_count == 0:
            return CompletedExecutionHandle({"": request.shots})
        kwargs = self._options.task_kwargs()
        inputs = self._invocation_inputs(request.invocation)
        if inputs:
            kwargs["inputs"] = inputs
        task = self.device.run(circuit, shots=request.shots, **kwargs)
        reference = _execution_reference(
            self.device,
            (task,),
            context={"kind": "sample", "width": str(circuit.qubit_count)},
        )
        return BraketExecutionHandle(
            tasks=(task,),
            result_loader=lambda: (task.result(),),
            decoder=lambda results: _decode_sampling_result(
                results[0], circuit.qubit_count
            ),
            reference=reference,
            native=task,
            poll_interval_seconds=self._handle_poll_interval,
            default_timeout_seconds=self._options.poll_timeout_seconds,
        )

    def submit_estimate(
        self,
        request: EstimateRequest["Circuit"],
    ) -> ExecutionHandle[float]:
        """Submit Braket expectation tasks without retrieving their results.

        Args:
            request (EstimateRequest[Circuit]): Circuit, observable, inputs,
                and accuracy policy.

        Returns:
            ExecutionHandle[float]: Lazy exact or shot-based result handle.

        Raises:
            ValueError: If the Hamiltonian is non-Hermitian, target precision
                is unsupported, or exact execution targets a QPU.
        """
        shots = self._estimation_shots
        if isinstance(request.accuracy, Exact):
            shots = 0
        elif isinstance(request.accuracy, ShotBased):
            shots = request.accuracy.shots
        elif isinstance(request.accuracy, TargetPrecision):
            raise ValueError(
                "Amazon Braket gate-model tasks do not support target-precision "
                "expectation requests; use ShotBased or Exact"
            )

        constant, terms = _validated_hamiltonian_terms(request.hamiltonian)
        if not terms:
            return CompletedExecutionHandle(constant)
        if shots == 0 and _is_qpu(self.device):
            raise ValueError(
                "Exact Braket expectation execution is unavailable on QPUs"
            )

        import qamomile.observable as qm_o
        from braket.circuits import ResultType  # type: ignore[import-not-found]
        from qamomile.braket.observable import hamiltonian_to_braket_observable

        result_type_api = cast(Any, ResultType)
        inputs = self._invocation_inputs(request.invocation)
        coefficients = tuple(
            float(complex(coefficient).real) for _, coefficient in terms
        )
        context = {
            "kind": "estimate",
            "constant": repr(constant),
            "coefficients": json.dumps(coefficients),
        }

        def add_expectation(circuit: "Circuit", operators: Any) -> "Circuit":
            """Copy a circuit and attach one unscaled expectation result type.

            Args:
                circuit (Circuit): State-preparation circuit.
                operators (Any): Qamomile Pauli term mapping.

            Returns:
                Circuit: Copied circuit with one Braket result type.
            """
            term = qm_o.Hamiltonian(num_qubits=request.hamiltonian.num_qubits)
            term.add_term(operators, 1.0)
            observable = hamiltonian_to_braket_observable(term)
            return circuit.copy().add_result_type(
                result_type_api.Expectation(observable)
            )

        if shots == 0:
            task_circuit = request.invocation.circuit.copy()
            for operators, _ in terms:
                term = qm_o.Hamiltonian(num_qubits=request.hamiltonian.num_qubits)
                term.add_term(operators, 1.0)
                task_circuit.add_result_type(
                    result_type_api.Expectation(hamiltonian_to_braket_observable(term))
                )
            kwargs = self._options.task_kwargs()
            if inputs:
                kwargs["inputs"] = inputs
            task = self.device.run(task_circuit, shots=0, **kwargs)
            context["layout"] = "single_task"
            reference = _execution_reference(self.device, (task,), context=context)
            return BraketExecutionHandle(
                tasks=(task,),
                result_loader=lambda: (task.result(),),
                decoder=lambda results: (
                    constant
                    + sum(
                        coefficient * value
                        for coefficient, value in zip(
                            coefficients,
                            _extract_expectations(results[0]),
                            strict=True,
                        )
                    )
                ),
                reference=reference,
                native=task,
                poll_interval_seconds=self._handle_poll_interval,
                default_timeout_seconds=self._options.poll_timeout_seconds,
            )

        task_circuits = tuple(
            add_expectation(request.invocation.circuit, operators)
            for operators, _ in terms
        )
        context["layout"] = "term_tasks"
        if hasattr(self.device, "run_batch"):
            kwargs = self._options.batch_kwargs()
            if inputs:
                kwargs["inputs"] = [dict(inputs) for _ in task_circuits]
            batch = self.device.run_batch(task_circuits, shots=shots, **kwargs)
            tasks = tuple(getattr(batch, "tasks", ()))

            def load_batch_results() -> Sequence[Any]:
                """Retrieve batch results under the explicit retry policy.

                Returns:
                    Sequence[Any]: Raw results in task submission order.
                """
                if self._options.batch_max_retries:
                    return batch.results(
                        fail_unsuccessful=True,
                        max_retries=self._options.batch_max_retries,
                    )
                if tasks:
                    return tuple(task.result() for task in tasks)
                return batch.results()

            reference = _execution_reference(
                self.device,
                tasks,
                context=context,
                group_id=str(getattr(batch, "id", "")) or None,
            )
            return BraketExecutionHandle(
                tasks=tasks,
                result_loader=load_batch_results,
                decoder=lambda results: (
                    constant
                    + sum(
                        coefficient * _extract_expectation(result)
                        for coefficient, result in zip(
                            coefficients, results, strict=True
                        )
                    )
                ),
                reference=reference,
                reference_factory=(
                    lambda: _execution_reference(
                        self.device,
                        tuple(getattr(batch, "tasks", ())),
                        context=context,
                        group_id=str(getattr(batch, "id", "")) or None,
                    )
                ),
                native=batch,
                poll_interval_seconds=self._handle_poll_interval,
                default_timeout_seconds=self._options.poll_timeout_seconds,
                allow_unsuccessful_loader=bool(self._options.batch_max_retries),
            )

        kwargs = self._options.task_kwargs()
        if inputs:
            kwargs["inputs"] = inputs
        tasks = tuple(
            self.device.run(task_circuit, shots=shots, **kwargs)
            for task_circuit in task_circuits
        )
        reference = _execution_reference(self.device, tasks, context=context)
        return BraketExecutionHandle(
            tasks=tasks,
            result_loader=lambda: tuple(task.result() for task in tasks),
            decoder=lambda results: (
                constant
                + sum(
                    coefficient * _extract_expectation(result)
                    for coefficient, result in zip(coefficients, results, strict=True)
                )
            ),
            reference=reference,
            native=tasks,
            poll_interval_seconds=self._handle_poll_interval,
            default_timeout_seconds=self._options.poll_timeout_seconds,
        )

    def restore(
        self,
        reference: ExecutionReference,
    ) -> ExecutionHandle[Any]:
        """Restore AWS quantum tasks from their serializable references.

        Args:
            reference (ExecutionReference): Reference returned by a prior
                Braket execution handle.

        Returns:
            ExecutionHandle[Any]: Restored sampling or expectation handle.

        Raises:
            ValueError: If the reference provider or decoding context is
                invalid.
            TypeError: If the configured device has no AWS session.
        """
        if reference.provider != "amazon_braket":
            raise ValueError(
                f"Cannot restore {reference.provider!r} with BraketExecutor"
            )
        if not reference.job_ids:
            raise ValueError("Braket execution reference has no task ARNs")
        aws_session = getattr(self.device, "aws_session", None)
        if aws_session is None:
            raise TypeError("Restoring Braket tasks requires an AWS device session")

        from braket.aws import AwsQuantumTask  # type: ignore[import-not-found]

        task_kwargs: dict[str, Any] = {"aws_session": aws_session}
        if self._options.poll_timeout_seconds is not None:
            task_kwargs["poll_timeout_seconds"] = self._options.poll_timeout_seconds
        if self._options.poll_interval_seconds is not None:
            task_kwargs["poll_interval_seconds"] = self._options.poll_interval_seconds
        tasks = tuple(
            AwsQuantumTask(task_id, **task_kwargs) for task_id in reference.job_ids
        )
        decoder = _reference_decoder(reference)
        native: object = tasks[0] if len(tasks) == 1 else tasks
        return BraketExecutionHandle(
            tasks=tasks,
            result_loader=lambda: tuple(task.result() for task in tasks),
            decoder=decoder,
            reference=reference,
            native=native,
            poll_interval_seconds=self._handle_poll_interval,
            default_timeout_seconds=self._options.poll_timeout_seconds,
        )

    @property
    def _handle_poll_interval(self) -> float:
        """Return the positive local polling interval for execution handles.

        Returns:
            float: Configured interval or one second.
        """
        return self._options.poll_interval_seconds or 1.0

    @staticmethod
    def _invocation_inputs(
        invocation: CircuitInvocation["Circuit"],
    ) -> dict[str, float]:
        """Convert Qamomile bindings to Braket native input names.

        Args:
            invocation (CircuitInvocation[Circuit]): Circuit invocation ABI.

        Returns:
            dict[str, float]: Braket free-parameter input mapping.

        Raises:
            ValueError: If a required Qamomile binding is absent.
        """
        values: dict[str, float] = {}
        for parameter in invocation.parameter_metadata.parameters:
            if parameter.name not in invocation.bindings:
                raise ValueError(f"Missing binding for parameter {parameter.name!r}")
            values[str(parameter.backend_param)] = float(
                invocation.bindings[parameter.name]
            )
        return values

    def bind_parameters(
        self,
        circuit: "Circuit",
        bindings: dict[str, Any],
        parameter_metadata: ParameterMetadata,
    ) -> "Circuit":
        """Bind Qamomile runtime parameters into a Braket circuit.

        Args:
            circuit (Circuit): Parameterized Braket circuit.
            bindings (dict[str, Any]): Values keyed by flattened Qamomile
                parameter name.
            parameter_metadata (ParameterMetadata): Compiled parameter ABI.

        Returns:
            Circuit: New circuit with all required parameters bound.

        Raises:
            ValueError: If a required binding is absent.
        """
        values: dict[str, float] = {}
        for parameter in parameter_metadata.parameters:
            if parameter.name not in bindings:
                raise ValueError(f"Missing binding for parameter {parameter.name!r}")
            values[str(parameter.backend_param)] = float(bindings[parameter.name])
        return circuit.make_bound_circuit(cast(dict[str, Any], values), strict=False)

    def estimate(
        self,
        circuit: "Circuit",
        hamiltonian: "qm_o.Hamiltonian",
        params: Sequence[float] | None = None,
    ) -> float:
        """Estimate a Qamomile Hamiltonian expectation value.

        Exact estimation submits one Braket task containing one result type
        per Pauli term. Shot-based estimation submits one task per term so
        non-commuting terms remain valid on devices with sampled result types.

        Args:
            circuit (Circuit): Bound Braket state-preparation circuit.
            hamiltonian (qm_o.Hamiltonian): Hamiltonian to evaluate.
            params (Sequence[float] | None): Positional parameter values for
                direct executor use. Qamomile normally binds before calling
                this method. Defaults to none.

        Returns:
            float: Real expectation value including the constant term.

        Raises:
            ValueError: If ``params`` do not match Braket's parameter order or
                if the result has a non-negligible imaginary component.
            RuntimeError: If a Braket task omits an expectation result.
        """
        if params is not None:
            names = sorted(str(parameter) for parameter in circuit.parameters)
            if len(names) != len(params):
                raise ValueError(
                    f"Expected {len(names)} Braket parameters, received {len(params)}"
                )
            circuit = circuit.make_bound_circuit(
                cast(dict[str, Any], dict(zip(names, params, strict=True))),
                strict=True,
            )
        request = EstimateRequest(
            CircuitInvocation(circuit, {}, ParameterMetadata()),
            hamiltonian,
        )
        return self.submit_estimate(request).result()


class BraketTranspiler(Transpiler["Circuit"]):
    """Transpile Qamomile quantum kernels to Amazon Braket circuits."""

    def _create_segmentation_pass(self) -> SegmentationPass:
        """Create the default host-orchestrated segmentation pass.

        Returns:
            SegmentationPass: Standard quantum/classical planner.
        """
        return SegmentationPass()

    def _create_emit_pass(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> EmitPass["Circuit"]:
        """Create the Braket legalization and materialization pass.

        Args:
            bindings (dict[str, Any] | None): Compile-time argument values.
                Defaults to none.
            parameters (list[str] | None): Runtime parameter names. Defaults
                to none.

        Returns:
            EmitPass[Circuit]: Capability-driven Braket emit pass.
        """
        return CircuitBackendEmitPass(
            BraketMaterializer(),
            bindings,
            parameters,
            policy=CompilationPolicy(
                prefer_native_semantic_ops=False,
                prefer_native_pauli_evolution=False,
            ),
        )

    def executor(  # type: ignore[override]
        self,
        device: Any = None,
        *,
        estimation_shots: int = 0,
        options: BraketExecutionOptions | None = None,
        run_kwargs: Mapping[str, Any] | None = None,
    ) -> BraketExecutor:
        """Create a Braket executor.

        Args:
            device (Any): Braket local simulator, AWS device, or compatible
                test double. Defaults to a local simulator.
            estimation_shots (int): Shots for expectation tasks. Defaults to
                zero for exact estimation.
            options (BraketExecutionOptions | None): Structured submission,
                polling, concurrency, and retry policy. Defaults to SDK
                behavior with retry disabled.
            run_kwargs (Mapping[str, Any] | None): Extra device task options.
                Compatibility form; cannot be combined with ``options``.
                Defaults to none.

        Returns:
            BraketExecutor: Configured task-backed executor.

        Raises:
            ValueError: If executor options are invalid.
        """
        return BraketExecutor(
            device,
            estimation_shots=estimation_shots,
            options=options,
            run_kwargs=run_kwargs,
        )


def _extract_expectation(result: Any) -> float:
    """Extract and validate the first Braket expectation result.

    Args:
        result (Any): Braket gate-model task result.

    Returns:
        float: Real expectation value.

    Raises:
        RuntimeError: If no result value is available.
        ValueError: If the expectation has a non-negligible imaginary part.
    """
    return sum(_extract_expectations(result))


def _extract_expectations(result: Any) -> list[float]:
    """Extract and validate every Braket expectation result.

    Args:
        result (Any): Braket gate-model task result.

    Returns:
        list[float]: Real expectation values in result-type order.

    Raises:
        RuntimeError: If no result value is available.
        ValueError: If an expectation has a non-negligible imaginary part.
    """
    values = getattr(result, "values", None)
    if not values:
        raise RuntimeError("Braket task returned no expectation result")
    extracted = []
    for raw_value in values:
        value = complex(raw_value)
        if abs(value.imag) > 1e-10:
            raise ValueError(f"Braket returned a complex expectation value: {value}")
        extracted.append(float(value.real))
    return extracted


def _decode_sampling_result(result: Any, width: int) -> dict[str, int]:
    """Normalize a Braket measurement result to Qamomile bit ordering.

    Args:
        result (Any): Braket gate-model task result.
        width (int): Full emitted circuit width.

    Returns:
        dict[str, int]: Counts with the highest qubit index on the left.

    Raises:
        RuntimeError: If measurement counts are absent.
    """
    raw_counts = result.measurement_counts
    if raw_counts is None:
        raise RuntimeError("Braket sampling task returned no measurement counts")
    measured_qubits = tuple(int(qubit) for qubit in result.measured_qubits)
    counts: dict[str, int] = {}
    for raw_bits, count in raw_counts.items():
        bits = ["0"] * width
        for qubit, value in zip(measured_qubits, raw_bits, strict=True):
            bits[qubit] = value
        bitstring = "".join(reversed(bits))
        counts[bitstring] = counts.get(bitstring, 0) + int(count)
    return counts


def _validated_hamiltonian_terms(
    hamiltonian: "qm_o.Hamiltonian",
) -> tuple[float, list[tuple[Any, complex | float]]]:
    """Validate Hermiticity and discard numerically zero Pauli terms.

    Args:
        hamiltonian (qm_o.Hamiltonian): Qamomile Hamiltonian.

    Returns:
        tuple[float, list[tuple[Any, complex | float]]]: Real constant and
            non-zero Pauli terms.

    Raises:
        ValueError: If any Hamiltonian coefficient has a significant
            imaginary component.
    """
    constant = complex(hamiltonian.constant)
    if abs(constant.imag) > HERMITIAN_IMAG_ATOL:
        raise ValueError("Braket expectation values require a Hermitian Hamiltonian")
    terms = [
        (operators, coefficient)
        for operators, coefficient in hamiltonian.terms.items()
        if abs(coefficient) > PAULI_TERM_ZERO_ATOL
    ]
    if any(
        abs(complex(coefficient).imag) > HERMITIAN_IMAG_ATOL for _, coefficient in terms
    ):
        raise ValueError("Braket expectation values require a Hermitian Hamiltonian")
    return float(constant.real), terms


def _execution_reference(
    device: Any,
    tasks: Sequence[Any],
    *,
    context: Mapping[str, str],
    group_id: str | None = None,
) -> ExecutionReference | None:
    """Build a restorable reference for AWS tasks only.

    Args:
        device (Any): Braket device that accepted the tasks.
        tasks (Sequence[Any]): Submitted native tasks.
        context (Mapping[str, str]): Secret-free decoding context.
        group_id (str | None): Optional batch identifier. Defaults to none.

    Returns:
        ExecutionReference | None: AWS reference, or ``None`` for local tasks.
    """
    task_ids = tuple(str(getattr(task, "id", "")) for task in tasks)
    if not task_ids or any(
        not task_id.startswith("arn:") or ":braket:" not in task_id
        for task_id in task_ids
    ):
        return None
    target_value = getattr(device, "arn", None) or getattr(device, "name", None)
    target = None if target_value is None else str(target_value)
    return ExecutionReference(
        provider="amazon_braket",
        job_ids=task_ids,
        target=target,
        group_id=group_id,
        context=dict(context),
    )


def _reference_decoder(
    reference: ExecutionReference,
) -> Callable[[Sequence[Any]], Any]:
    """Reconstruct the result decoder stored in an execution reference.

    Args:
        reference (ExecutionReference): Restored Braket task reference.

    Returns:
        Callable[[Sequence[Any]], Any]: Decoder accepting raw task results.

    Raises:
        ValueError: If the stored decoding context is malformed or unknown.
    """
    kind = reference.context.get("kind")
    if kind == "sample":
        try:
            width = int(reference.context["width"])
        except (KeyError, ValueError) as error:
            raise ValueError("Invalid Braket sampling reference context") from error
        return lambda results: _decode_sampling_result(results[0], width)
    if kind != "estimate":
        raise ValueError(f"Unknown Braket execution reference kind: {kind!r}")
    try:
        constant = float(reference.context["constant"])
        coefficients = tuple(
            float(value) for value in json.loads(reference.context["coefficients"])
        )
        layout = reference.context["layout"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("Invalid Braket expectation reference context") from error
    if layout == "single_task":
        return lambda results: (
            constant
            + sum(
                coefficient * value
                for coefficient, value in zip(
                    coefficients,
                    _extract_expectations(results[0]),
                    strict=True,
                )
            )
        )
    if layout == "term_tasks":
        return lambda results: (
            constant
            + sum(
                coefficient * _extract_expectation(result)
                for coefficient, result in zip(coefficients, results, strict=True)
            )
        )
    raise ValueError(f"Unknown Braket expectation layout: {layout!r}")


def _is_qpu(device: Any) -> bool:
    """Return whether a Braket device advertises itself as a QPU.

    Args:
        device (Any): Braket device or compatible test double.

    Returns:
        bool: Whether the normalized device type is ``QPU``.
    """
    device_type = getattr(device, "type", None)
    value = getattr(device_type, "value", device_type)
    return str(value).upper().rsplit(".", 1)[-1] == "QPU"


def _legacy_execution_options(
    run_kwargs: Mapping[str, Any] | None,
) -> BraketExecutionOptions:
    """Translate the legacy flat option mapping to structured options.

    Args:
        run_kwargs (Mapping[str, Any] | None): Legacy task keyword arguments.

    Returns:
        BraketExecutionOptions: Structured compatibility options.

    Raises:
        ValueError: If the legacy mapping contains invalid structured values.
    """
    values = dict(run_kwargs or {})
    shared = {
        key: values.pop(key, None)
        for key in (
            "s3_destination_folder",
            "reservation_arn",
            "max_parallel",
            "poll_timeout_seconds",
            "poll_interval_seconds",
        )
    }
    return BraketExecutionOptions(
        s3_destination_folder=shared["s3_destination_folder"],
        reservation_arn=shared["reservation_arn"],
        max_parallel=shared["max_parallel"],
        poll_timeout_seconds=shared["poll_timeout_seconds"],
        poll_interval_seconds=shared["poll_interval_seconds"],
        task_options=values,
        batch_options=values,
    )
