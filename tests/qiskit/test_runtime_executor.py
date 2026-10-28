"""Exercise Runtime submission and result conversion using real local primitives."""

from __future__ import annotations

import json
import math
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("qiskit")

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister  # noqa: E402
from qiskit.circuit import Parameter  # noqa: E402
from qiskit.primitives import StatevectorEstimator, StatevectorSampler  # noqa: E402
from qiskit.providers.exceptions import QiskitBackendNotFoundError  # noqa: E402
from qiskit.providers.fake_provider import GenericBackendV2  # noqa: E402
from qiskit.transpiler.preset_passmanagers import (  # noqa: E402
    generate_preset_pass_manager,
)

import qamomile.circuit as qmc  # noqa: E402
import qamomile.observable as qm_o  # noqa: E402
from qamomile.circuit.transpiler.execution_handle import (  # noqa: E402
    ExecutionReference,
)
from qamomile.circuit.transpiler.execution_request import (  # noqa: E402
    CircuitInvocation,
    EstimateRequest,
    Exact,
    SampleRequest,
    ShotBased,
    TargetPrecision,
)
from qamomile.circuit.transpiler.execution_snapshot import (  # noqa: E402
    ExecutionSnapshotKind,
)
from qamomile.circuit.transpiler.job import JobSnapshot, SampleJob  # noqa: E402
from qamomile.circuit.transpiler.parameter_binding import (  # noqa: E402
    ParameterInfo,
    ParameterMetadata,
)
from qamomile.qiskit import (  # noqa: E402
    QiskitExecutionOptions,
    QiskitExecutor,
    QiskitTranspiler,
)


class _RecordingJob:
    """Record result retrieval while executing through a Qiskit primitive job.

    Args:
        native (Any): Local Qiskit primitive job to wrap.
    """

    def __init__(self, native: Any) -> None:
        """Initialize the primitive job and its retrieval counter.

        Args:
            native (Any): Local Qiskit primitive job to wrap.
        """
        self.native = native
        self.result_calls = 0

    def job_id(self) -> str:
        """Read the wrapped job's identifier.

        Returns:
            str: Native primitive job identifier.
        """
        return self.native.job_id()

    def status(self) -> Any:
        """Read the wrapped job's execution status.

        Returns:
            Any: Native primitive job status.
        """
        return self.native.status()

    def result(self, timeout: float | None = None) -> Any:
        """Record retrieval and wait for the native primitive result.

        Args:
            timeout (float | None): Unused compatibility argument. Defaults
                to None; the wrapped local job controls its wait.

        Returns:
            Any: Native primitive result.

        Raises:
            Exception: If the native primitive job fails.
        """
        self.result_calls += 1
        return self.native.result()

    def cancel(self) -> None:
        """Request native primitive cancellation.

        Raises:
            Exception: If the native job rejects cancellation.
        """
        self.native.cancel()


@pytest.fixture
def runtime_sdk(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace only the cloud transport with recording statevector primitives.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture that restores the SDK import.

    Returns:
        SimpleNamespace: Recorded primitives, jobs, and account interactions.
    """
    state = SimpleNamespace(
        samplers=[],
        estimators=[],
        jobs={},
        available_backends=[],
        service_calls=[],
        lookup_calls=[],
        service_error=None,
        lookup_error=None,
    )

    class IBMBackend(GenericBackendV2):
        """Provide an identifiable IBM backend with a local simulation target.

        Args:
            name (str): Device name. Defaults to "ibm_available".
        """

        def __init__(self, name: str = "ibm_available") -> None:
            """Create a noiseless two-qubit device.

            Args:
                name (str): Device name. Defaults to "ibm_available".
            """
            super().__init__(num_qubits=2, noise_info=False, seed=23)
            self.name = name

    class Service:
        """Record account access and provide fixture-owned backends and jobs.

        Args:
            **kwargs (Any): SDK account configuration to record.

        Raises:
            Exception: If the fixture configures an authentication failure.
        """

        def __init__(self, **kwargs: Any) -> None:
            """Record SDK account construction.

            Args:
                **kwargs (Any): Account options supplied by the executor.

            Raises:
                Exception: If the fixture configures an authentication failure.
            """
            state.service_calls.append(kwargs)
            if state.service_error is not None:
                raise state.service_error

        def backend(self, name: str, **kwargs: Any) -> IBMBackend:
            """Resolve a fixture backend and attach this service.

            Args:
                name (str): Requested device name.
                **kwargs (Any): Additional backend lookup options to record.

            Returns:
                IBMBackend: Matching fixture backend.

            Raises:
                QiskitBackendNotFoundError: If no fixture backend matches.
                Exception: If the fixture configures a backend lookup failure.
            """
            state.lookup_calls.append({"name": name, **kwargs})
            if state.lookup_error is not None:
                raise state.lookup_error
            for target in state.available_backends:
                if target.name == name:
                    target.service = self
                    return target
            raise QiskitBackendNotFoundError("No backend matches the criteria")

        def job(self, job_id: str) -> _RecordingJob:
            """Retrieve a previously recorded primitive job.

            Args:
                job_id (str): Native job identifier.

            Returns:
                _RecordingJob: Previously submitted job.

            Raises:
                KeyError: If the identifier does not belong to a fixture job.
            """
            return state.jobs[job_id]

        @staticmethod
        def save_account(**kwargs: Any) -> None:
            """Fail if executor construction attempts to persist credentials.

            Args:
                **kwargs (Any): Account options that would have been saved.

            Raises:
                pytest.fail.Exception: Always, because account writes are forbidden.
            """
            pytest.fail("Executor construction must not persist account credentials")

    state.IBMBackend = IBMBackend
    state.Service = Service

    class Sampler:
        """Record Runtime sampler calls and delegate to a seeded local sampler.

        Args:
            mode (Any): Execution backend or caller-owned mode. Defaults to None.
            options (Any): Runtime sampler options to record. Defaults to None.
        """

        def __init__(self, mode: Any = None, options: Any = None) -> None:
            """Create a deterministic local sampler transport.

            Args:
                mode (Any): Backend or caller-owned mode. Defaults to None.
                options (Any): Runtime sampler options. Defaults to None.
            """
            self.mode = mode
            self.options = options
            self.calls: list[tuple[Any, dict[str, Any]]] = []
            self.primitive = StatevectorSampler(seed=19)
            state.samplers.append(self)

        def backend(self) -> Any:
            """Resolve the backend associated with the execution mode.

            Returns:
                Any: Mode's backend or the backend supplied directly.
            """
            return getattr(self.mode, "target_backend", self.mode)

        def run(self, pubs: Any, **kwargs: Any) -> _RecordingJob:
            """Record a sampling request and submit it locally.

            Args:
                pubs (Any): Sampler primitive input publications.
                **kwargs (Any): Per-request sampler options.

            Returns:
                _RecordingJob: Recorded local sampling job.

            Raises:
                Exception: If the local sampler rejects the request.
            """
            self.calls.append((pubs, kwargs))
            job = _RecordingJob(self.primitive.run(pubs, **kwargs))
            state.jobs[job.job_id()] = job
            return job

    class Estimator:
        """Record Runtime estimator calls and delegate to a local estimator.

        Args:
            mode (Any): Execution backend or caller-owned mode. Defaults to None.
            options (Any): Runtime estimator options to record. Defaults to None.
        """

        def __init__(self, mode: Any = None, options: Any = None) -> None:
            """Create a seeded local estimator transport.

            Args:
                mode (Any): Backend or caller-owned mode. Defaults to None.
                options (Any): Runtime estimator options. Defaults to None.
            """
            self.mode = mode
            self.options = options
            self.calls: list[tuple[Any, dict[str, Any]]] = []
            self.primitive = StatevectorEstimator(seed=19)
            state.estimators.append(self)

        def run(self, pubs: Any, **kwargs: Any) -> _RecordingJob:
            """Record an expectation request and submit it locally.

            Args:
                pubs (Any): Estimator primitive input publications.
                **kwargs (Any): Per-request estimator options.

            Returns:
                _RecordingJob: Recorded local expectation job.

            Raises:
                Exception: If the local estimator rejects the request.
            """
            self.calls.append((pubs, kwargs))
            job = _RecordingJob(self.primitive.run(pubs, **kwargs))
            state.jobs[job.job_id()] = job
            return job

    sdk = ModuleType("qiskit_ibm_runtime")
    sdk.IBMBackend = IBMBackend
    sdk.QiskitRuntimeService = Service
    sdk.SamplerV2 = Sampler
    sdk.EstimatorV2 = Estimator
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", sdk)
    return state


@pytest.fixture
def backend() -> GenericBackendV2:
    """Provide a hardware-shaped target without credentials or remote jobs.

    Returns:
        GenericBackendV2: Seeded five-qubit compilation target.
    """
    return GenericBackendV2(num_qubits=5, seed=23)


@pytest.mark.parametrize(
    "credentials", [{}, {"api_key": "test-key", "instance_crn": "test-crn"}]
)
def test_named_backend_rejects_an_independently_authenticated_mode(
    runtime_sdk: SimpleNamespace, credentials: dict[str, str]
) -> None:
    """A preconfigured mode cannot override the named account's execution context."""
    with pytest.raises(ValueError, match="backend object"):
        QiskitExecutor("ibm_available", mode=object(), **credentials)

    assert runtime_sdk.service_calls == []
    assert runtime_sdk.lookup_calls == []
    assert runtime_sdk.samplers == []


def _invocation(circuit: QuantumCircuit) -> CircuitInvocation[QuantumCircuit]:
    """Wrap a parameter-free circuit in a backend-neutral invocation.

    Args:
        circuit (QuantumCircuit): Parameter-free circuit to execute.

    Returns:
        CircuitInvocation[QuantumCircuit]: Invocation with empty bindings.
    """
    return CircuitInvocation(circuit, {}, ParameterMetadata())


@qmc.qkernel
def _parameterized_sample(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Prepare and measure a parameterized two-qubit entangled state.

    Ry(theta) prepares amplitudes cos(theta/2) and sin(theta/2), then CX
    correlates both measurements. Angles zero and pi produce 00 and 11.

    Args:
        theta (qmc.Float): Ry rotation angle in radians.

    Returns:
        qmc.Vector[qmc.Bit]: Two correlated computational-basis outcomes.
    """
    qubits = qmc.qubit_array(2, "qubits")
    qubits[0] = qmc.ry(qubits[0], theta)
    qubits[0], qubits[1] = qmc.cx(qubits[0], qubits[1])
    return qmc.measure(qubits)


@qmc.qkernel
def _parameterized_expectation(
    theta: qmc.Float, observable: qmc.Observable
) -> qmc.Float:
    """Evaluate an observable after a runtime parameter rotation.

    Args:
        theta (qmc.Float): Ry rotation angle in radians.
        observable (qmc.Observable): Observable to evaluate after preparation.

    Returns:
        qmc.Float: Expectation value on the prepared two-qubit state;
            Z(0) has expectation cos(theta).
    """
    qubits = qmc.qubit_array(2, "qubits")
    qubits[0] = qmc.ry(qubits[0], theta)
    return qmc.expval(qubits, observable)


@qmc.qkernel
def _product_state_expectation(
    width: qmc.UInt,
    angles: qmc.Vector[qmc.Float],
    observable: qmc.Observable,
) -> qmc.Float:
    """Evaluate an observable on independent Ry-rotated qubits.

    Args:
        width (qmc.UInt): Number of prepared qubits.
        angles (qmc.Vector[qmc.Float]): One Ry angle in radians per qubit.
        observable (qmc.Observable): Observable evaluated on the product state.

    Returns:
        qmc.Float: Expectation value on the prepared product state.
    """
    qubits = qmc.qubit_array(width, "qubits")
    for index in qmc.range(width):
        qubits[index] = qmc.ry(qubits[index], angles[index])
    return qmc.expval(qubits, observable)


@qmc.qkernel
def _paired_expectations(
    first_observable: qmc.Observable,
    second_observable: qmc.Observable,
) -> tuple[qmc.Float, qmc.Float]:
    """Evaluate two observables on separately prepared excited states.

    Args:
        first_observable (qmc.Observable): First observable to evaluate.
        second_observable (qmc.Observable): Second observable to evaluate.

    Returns:
        tuple[qmc.Float, qmc.Float]: First expectation (qmc.Float) and
            second expectation (qmc.Float), in declaration order.
    """
    first_qubits = qmc.qubit_array(1, "first")
    first_qubits[0] = qmc.x(first_qubits[0])
    second_qubits = qmc.qubit_array(1, "second")
    second_qubits[0] = qmc.x(second_qubits[0])
    first = qmc.expval(first_qubits, first_observable)
    second = qmc.expval(second_qubits, second_observable)
    return first, second


@qmc.qkernel
def _empty_sample() -> qmc.Vector[qmc.Bit]:
    """Return the empty measurement vector without a Runtime submission.

    Returns:
        qmc.Vector[qmc.Bit]: Empty computational-basis outcome.
    """
    return qmc.measure(qmc.qubit_array(0, "empty"))


@qmc.qkernel
def _loop_induction_sample() -> qmc.Bit:
    """Apply a static loop induction angle without exposing a public parameter.

    Returns:
        qmc.Bit: Deterministic one outcome; Rz only changes the excited state's phase.
    """
    qubit = qmc.qubit("qubit")
    qubit = qmc.x(qubit)
    for index in qmc.range(3):
        qubit = qmc.rz(qubit, index)
    return qmc.measure(qubit)


@qmc.qkernel
def _loop_repeated_x() -> qmc.Bit:
    """Apply three X gates in a static native loop.

    Returns:
        qmc.Bit: Deterministic one outcome because an odd number of X gates flips zero.
    """
    qubit = qmc.qubit("qubit")
    for index in qmc.range(3):
        qubit = qmc.x(qubit)
    return qmc.measure(qubit)


@qmc.qkernel
def _loop_induction_expectation(observable: qmc.Observable) -> qmc.Float:
    """Accumulate Rx induction angles zero, one, and two before estimation.

    Args:
        observable (qmc.Observable): Observable evaluated after the Rx rotations.

    Returns:
        qmc.Float: Expectation after Rx(3); Z has expectation cos(3).
    """
    qubits = qmc.qubit_array(1, "qubits")
    for index in qmc.range(3):
        qubits[0] = qmc.rx(qubits[0], index)
    return qmc.expval(qubits, observable)


def test_named_backend_signs_in_and_executes_public_sample_and_estimate_jobs(
    runtime_sdk: SimpleNamespace,
) -> None:
    """A name and credentials select IBM Runtime for both public execution APIs."""
    target = runtime_sdk.IBMBackend()
    runtime_sdk.available_backends = [target]
    executor = QiskitExecutor(
        "ibm_available", api_key="test-api-key", instance_crn="crn:test-instance"
    )
    transpiler = QiskitTranspiler()
    sample_program = transpiler.transpile(_parameterized_sample, parameters=["theta"])
    estimate_program = transpiler.transpile(
        _parameterized_expectation,
        bindings={"observable": qm_o.Z(0)},
        parameters=["theta"],
    )

    sampled = sample_program.sample(executor, shots=17, bindings={"theta": math.pi})
    estimated = estimate_program.run(executor, bindings={"theta": math.pi})

    assert runtime_sdk.service_calls == [
        {
            "channel": "ibm_quantum_platform",
            "token": "test-api-key",
            "instance": "crn:test-instance",
        }
    ]
    assert runtime_sdk.lookup_calls == [
        {"name": "ibm_available", "instance": "crn:test-instance"}
    ]
    assert executor.backend is target
    assert runtime_sdk.samplers[0].mode is target
    assert runtime_sdk.estimators[0].mode is target
    assert all(job.result_calls == 0 for job in runtime_sdk.jobs.values())
    assert sampled.result().results == [((1, 1), 17)]
    assert sampled.result().shots == 17
    assert estimated.result() == pytest.approx(-1.0, abs=1e-10, rel=1e-10)


def test_named_backend_can_use_a_previously_saved_account(
    runtime_sdk: SimpleNamespace,
) -> None:
    """A backend name alone lets the IBM SDK resolve its saved account."""
    target = runtime_sdk.IBMBackend()
    runtime_sdk.available_backends = [target]

    executor = QiskitExecutor("ibm_available")

    assert executor.backend is target
    assert runtime_sdk.service_calls == [{"channel": "ibm_quantum_platform"}]
    assert runtime_sdk.lookup_calls == [{"name": "ibm_available"}]


def test_named_backend_can_use_an_explicit_service(
    runtime_sdk: SimpleNamespace,
) -> None:
    """A caller-owned service performs name resolution without another sign-in."""
    target = runtime_sdk.IBMBackend()
    runtime_sdk.available_backends = [target]
    service = runtime_sdk.Service()
    runtime_sdk.service_calls.clear()

    executor = QiskitExecutor("ibm_available", service=service)

    assert executor.backend is target
    assert runtime_sdk.service_calls == []
    assert runtime_sdk.lookup_calls == [{"name": "ibm_available"}]
    assert executor.capabilities.supports_restoration


def test_ibm_backend_objects_automatically_select_runtime(
    runtime_sdk: SimpleNamespace,
) -> None:
    """An authenticated IBM backend object needs no separate executor or mode."""
    target = runtime_sdk.IBMBackend()
    circuit = QuantumCircuit(1)
    circuit.x(0)

    executor = QiskitExecutor(target)

    assert executor.execute(circuit, 13) == {"1": 13}
    assert executor.estimate(circuit, qm_o.Z(0)) == pytest.approx(
        -1.0, abs=1e-10, rel=1e-10
    )
    assert runtime_sdk.samplers[0].mode is target
    assert runtime_sdk.estimators[0].mode is target
    assert runtime_sdk.service_calls == []


@pytest.mark.parametrize("available", [False, True])
def test_inaccessible_named_backend_fails_before_creating_primitives(
    runtime_sdk: SimpleNamespace, available: bool
) -> None:
    """An unavailable name propagates the SDK error without simulator fallback."""
    if available:
        runtime_sdk.available_backends = [runtime_sdk.IBMBackend("ibm_other")]

    with pytest.raises(QiskitBackendNotFoundError):
        QiskitExecutor(
            "ibm_unavailable",
            api_key="test-api-key",
            instance_crn="crn:test-instance",
        )

    assert runtime_sdk.lookup_calls == [
        {"name": "ibm_unavailable", "instance": "crn:test-instance"}
    ]
    assert runtime_sdk.samplers == []
    assert runtime_sdk.estimators == []
    assert runtime_sdk.jobs == {}


@pytest.mark.parametrize("stage", ["service_error", "lookup_error"])
def test_sdk_authentication_and_lookup_errors_propagate_without_fallback(
    runtime_sdk: SimpleNamespace, stage: str
) -> None:
    """SDK authentication and access errors keep their identity and cause."""
    failure = RuntimeError("IBM authentication or access failed")
    setattr(runtime_sdk, stage, failure)

    with pytest.raises(RuntimeError) as caught:
        QiskitExecutor(
            "ibm_available", api_key="test-api-key", instance_crn="crn:test-instance"
        )

    assert caught.value is failure
    assert runtime_sdk.samplers == []
    assert runtime_sdk.estimators == []


@pytest.mark.parametrize(
    "credentials",
    [
        {"api_key": "test-api-key"},
        {"instance_crn": "crn:test-instance"},
        {"api_key": "", "instance_crn": "crn:test-instance"},
        {"api_key": "   ", "instance_crn": "crn:test-instance"},
        {"api_key": "test-api-key", "instance_crn": ""},
        {"api_key": "test-api-key", "instance_crn": "   "},
    ],
)
def test_incomplete_or_blank_credentials_fail_before_sign_in(
    runtime_sdk: SimpleNamespace, credentials: dict[str, str]
) -> None:
    """Explicit credentials require a nonblank API key and instance together."""
    with pytest.raises(ValueError):
        QiskitExecutor("ibm_available", **credentials)

    assert runtime_sdk.service_calls == []
    assert runtime_sdk.lookup_calls == []
    assert runtime_sdk.samplers == []


@pytest.mark.parametrize("use_backend_object", [False, True])
def test_credentials_require_a_named_backend(
    runtime_sdk: SimpleNamespace,
    backend: GenericBackendV2,
    use_backend_object: bool,
) -> None:
    """Credentials cannot silently select local execution or replace a backend."""
    with pytest.raises(ValueError):
        QiskitExecutor(
            backend if use_backend_object else None,
            api_key="test-api-key",
            instance_crn="crn:test-instance",
        )

    assert runtime_sdk.service_calls == []
    assert runtime_sdk.samplers == []


def test_credentials_cannot_override_an_explicit_service(
    runtime_sdk: SimpleNamespace,
) -> None:
    """Explicit credentials and a caller-owned service cannot compete for access."""
    with pytest.raises(ValueError):
        QiskitExecutor(
            "ibm_available",
            api_key="test-api-key",
            instance_crn="crn:test-instance",
            service=SimpleNamespace(),
        )

    assert runtime_sdk.service_calls == []
    assert runtime_sdk.lookup_calls == []


@pytest.mark.parametrize(
    "option", ["sampler_options", "estimator_options", "pass_manager", "service"]
)
@pytest.mark.parametrize("use_backend_object", [False, True])
def test_runtime_options_are_rejected_for_ordinary_local_execution(
    runtime_sdk: SimpleNamespace,
    backend: GenericBackendV2,
    option: str,
    use_backend_object: bool,
) -> None:
    """Runtime-only configuration never disappears into the legacy local path."""
    value = {} if option.endswith("options") else SimpleNamespace()

    with pytest.raises(ValueError):
        QiskitExecutor(backend if use_backend_object else None, **{option: value})

    assert runtime_sdk.service_calls == []
    assert runtime_sdk.samplers == []


def test_runtime_backend_rejects_a_local_estimator_override(
    runtime_sdk: SimpleNamespace,
) -> None:
    """A hardware executor cannot evaluate requested observables locally by mistake."""
    with pytest.raises(ValueError):
        QiskitExecutor(runtime_sdk.IBMBackend(), estimator=StatevectorEstimator())

    assert runtime_sdk.samplers == []
    assert runtime_sdk.estimators == []


@pytest.mark.parametrize("backend_kind", ["default", "basic", "generic"])
def test_existing_local_construction_preserves_sampling_and_estimation(
    runtime_sdk: SimpleNamespace, backend_kind: str
) -> None:
    """Existing constructors retain local sample and expectation behavior."""
    if backend_kind == "default":
        executor = QiskitExecutor()
    else:
        from qiskit.providers.basic_provider import BasicSimulator

        target = (
            BasicSimulator()
            if backend_kind == "basic"
            else GenericBackendV2(num_qubits=2, noise_info=False, seed=23)
        )
        executor = QiskitExecutor(target, StatevectorEstimator())
        assert executor.backend is target
    circuit = QuantumCircuit(1)
    circuit.x(0)

    assert executor.execute(circuit, 13) == {"1": 13}
    assert executor.estimate(circuit, qm_o.Z(0)) == pytest.approx(
        -1.0, abs=1e-10, rel=1e-10
    )
    assert executor.submit_sample(SampleRequest(_invocation(circuit), 17)).result() == {
        "1": 17
    }
    assert executor.submit_estimate(
        EstimateRequest(_invocation(circuit), qm_o.Z(0))
    ).result() == pytest.approx(-1.0, abs=1e-10, rel=1e-10)
    assert runtime_sdk.service_calls == []
    assert runtime_sdk.samplers == []
    assert runtime_sdk.estimators == []


def test_local_execution_does_not_require_ibm_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ordinary simulator API remains usable when Runtime is unavailable."""
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", None)
    circuit = QuantumCircuit(1)
    circuit.x(0)
    executor = QiskitExecutor()

    assert executor.execute(circuit, 7) == {"1": 7}
    assert executor.estimate(circuit, qm_o.Z(0)) == pytest.approx(
        -1.0, abs=1e-10, rel=1e-10
    )


def test_api_key_is_not_retained_in_executor_or_restoration_metadata(
    runtime_sdk: SimpleNamespace,
) -> None:
    """Credentials reach the SDK without becoming executor or saved-job metadata."""
    secret = "test-api-key-that-must-not-be-persisted"
    runtime_sdk.available_backends = [runtime_sdk.IBMBackend()]
    executor = QiskitExecutor(
        "ibm_available", api_key=secret, instance_crn="crn:test-instance"
    )
    executable = QiskitTranspiler().transpile(
        _parameterized_sample, parameters=["theta"]
    )
    job = executable.sample(executor, shots=7, bindings={"theta": math.pi})

    assert secret not in repr(executor)
    assert secret not in repr(vars(executor))
    assert secret not in json.dumps(job.snapshot().to_dict())
    assert secret not in json.dumps(
        [reference.to_dict() for reference in job.references()]
    )
    assert executor.capabilities.supports_restoration


@pytest.mark.parametrize("explicit_service", [False, True])
def test_transpiler_factory_forwards_runtime_configuration(
    runtime_sdk: SimpleNamespace, explicit_service: bool
) -> None:
    """The existing factory exposes credentials, service, and Runtime options."""
    target = runtime_sdk.IBMBackend()
    runtime_sdk.available_backends = [target]
    compiled_circuits = []
    pass_manager = generate_preset_pass_manager(backend=target, optimization_level=1)

    def compile_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
        """Record the circuit passed to the custom hardware pass manager.

        Args:
            circuit (QuantumCircuit): Circuit to compile for the fixture backend.

        Returns:
            QuantumCircuit: Circuit lowered to the fixture backend's ISA.

        Raises:
            Exception: If the Qiskit pass manager cannot compile the circuit.
        """
        compiled_circuits.append(circuit)
        return pass_manager.run(circuit)

    credentials = (
        {"service": runtime_sdk.Service()}
        if explicit_service
        else {"api_key": "test-api-key", "instance_crn": "crn:test-instance"}
    )
    executor = QiskitTranspiler().executor(
        "ibm_available",
        sampler_options={"default_shots": 32},
        estimator_options={"resilience_level": 1},
        pass_manager=SimpleNamespace(run=compile_circuit),
        **credentials,
    )

    assert isinstance(executor, QiskitExecutor)
    assert executor.execute(QuantumCircuit(1), 5) == {"0": 5}
    assert len(compiled_circuits) == 1
    assert runtime_sdk.samplers[0].mode is target
    assert runtime_sdk.estimators[0].mode is target
    assert runtime_sdk.samplers[0].options == {"default_shots": 32}
    assert runtime_sdk.estimators[0].options == {"resilience_level": 1}


@pytest.mark.parametrize("factory", [QiskitExecutor, QiskitTranspiler().executor])
def test_named_backend_forwards_qamomile_execution_options(
    runtime_sdk: SimpleNamespace, factory: Any
) -> None:
    """Named authentication forwards shared and primitive-specific Qamomile options."""
    target = runtime_sdk.IBMBackend()
    runtime_sdk.available_backends = [target]
    options = QiskitExecutionOptions(
        max_execution_time=300,
        resilience_level=1,
        sampler_options={"default_shots": 32},
        estimator_options={"default_precision": 0.1},
    )

    executor = factory(
        "ibm_available",
        api_key="test-api-key",
        instance_crn="crn:test-instance",
        options=options,
    )

    assert executor.backend is target
    assert runtime_sdk.service_calls == [
        {
            "channel": "ibm_quantum_platform",
            "token": "test-api-key",
            "instance": "crn:test-instance",
        }
    ]
    assert runtime_sdk.lookup_calls == [
        {"name": "ibm_available", "instance": "crn:test-instance"}
    ]
    assert runtime_sdk.samplers[0].options == {
        "max_execution_time": 300,
        "default_shots": 32,
    }
    assert runtime_sdk.estimators[0].options == {
        "max_execution_time": 300,
        "resilience_level": 1,
        "default_precision": 0.1,
    }
    assert runtime_sdk.jobs == {}


@pytest.mark.parametrize("factory", [QiskitExecutor, QiskitTranspiler().executor])
@pytest.mark.parametrize("legacy", ["sampler_options", "estimator_options"])
def test_qamomile_options_conflict_with_legacy_options_before_login(
    runtime_sdk: SimpleNamespace, factory: Any, legacy: str
) -> None:
    """Even an empty legacy mapping cannot silently override the options object."""
    with pytest.raises(ValueError, match="options"):
        factory(
            "ibm_available",
            api_key="test-api-key",
            instance_crn="crn:test-instance",
            options=QiskitExecutionOptions(),
            **{legacy: {}},
        )

    assert runtime_sdk.service_calls == []
    assert runtime_sdk.lookup_calls == []
    assert runtime_sdk.samplers == []
    assert runtime_sdk.estimators == []


@pytest.mark.parametrize("factory", [QiskitExecutor, QiskitTranspiler().executor])
@pytest.mark.parametrize("invalid_options", [{}, True, "runtime"])
def test_wrong_qamomile_options_type_fails_before_login(
    runtime_sdk: SimpleNamespace, factory: Any, invalid_options: Any
) -> None:
    """The unified options argument requires the Qamomile configuration type."""
    with pytest.raises(TypeError, match="QiskitExecutionOptions"):
        factory(
            "ibm_available",
            api_key="test-api-key",
            instance_crn="crn:test-instance",
            options=invalid_options,
        )

    assert runtime_sdk.service_calls == []
    assert runtime_sdk.lookup_calls == []
    assert runtime_sdk.samplers == []
    assert runtime_sdk.estimators == []


@pytest.mark.parametrize("factory", [QiskitExecutor, QiskitTranspiler().executor])
@pytest.mark.parametrize("use_backend_object", [False, True])
def test_qamomile_runtime_options_are_rejected_for_local_execution(
    runtime_sdk: SimpleNamespace,
    backend: GenericBackendV2,
    factory: Any,
    use_backend_object: bool,
) -> None:
    """An explicit Runtime options object is never ignored by a local executor."""
    with pytest.raises(ValueError, match="Runtime options"):
        factory(
            backend if use_backend_object else None, options=QiskitExecutionOptions()
        )

    assert runtime_sdk.service_calls == []
    assert runtime_sdk.samplers == []
    assert runtime_sdk.estimators == []


def test_reused_qamomile_options_are_isolated_between_executors(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2
) -> None:
    """An SDK's nested option mutation cannot affect a later executor or its source."""
    options = QiskitExecutionOptions(
        sampler_options={"environment": {"job_tags": ["sample"]}},
        estimator_options={"environment": {"job_tags": ["estimate"]}},
    )
    QiskitExecutor(backend, mode=backend, options=options)
    runtime_sdk.samplers[0].options["environment"]["job_tags"].append("changed")
    runtime_sdk.estimators[0].options["environment"]["job_tags"].append("changed")

    QiskitTranspiler().executor(backend, mode=backend, options=options)

    assert runtime_sdk.samplers[1].options == {"environment": {"job_tags": ["sample"]}}
    assert runtime_sdk.estimators[1].options == {
        "environment": {"job_tags": ["estimate"]}
    }
    assert options.sampler_kwargs() == {"environment": {"job_tags": ["sample"]}}
    assert options.estimator_kwargs() == {"environment": {"job_tags": ["estimate"]}}


@pytest.mark.parametrize("factory", [QiskitExecutor, QiskitTranspiler().executor])
def test_constructor_routes_backend_mode_and_primitive_options(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2, factory: Any
) -> None:
    """A caller-owned Session or Batch and each options dictionary reach Runtime."""
    mode = SimpleNamespace(target_backend=backend)
    sampler_options = {"default_shots": 128}
    estimator_options = {"resilience_level": 1}

    factory(
        backend,
        mode=mode,
        sampler_options=sampler_options,
        estimator_options=estimator_options,
    )

    assert runtime_sdk.samplers[0].mode is mode
    assert runtime_sdk.estimators[0].mode is mode
    assert runtime_sdk.samplers[0].options == sampler_options
    assert runtime_sdk.estimators[0].options == estimator_options


def test_mode_for_a_different_backend_is_rejected(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2
) -> None:
    """Runtime mode cannot execute circuits compiled for a different target."""
    mode = SimpleNamespace(target_backend=SimpleNamespace(name="other_backend"))

    with pytest.raises(ValueError, match="backend must match"):
        QiskitExecutor(backend, mode=mode)


@pytest.mark.parametrize("named_backend", [False, True])
def test_missing_runtime_dependency_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
    backend: GenericBackendV2,
    named_backend: bool,
) -> None:
    """A missing Runtime package points to the unified Qiskit installation."""
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", None)

    with pytest.raises(ImportError, match="qamomile\\[qiskit\\]"):
        if named_backend:
            QiskitExecutor("ibm_available")
        else:
            QiskitExecutor(backend, mode=backend)


def test_sample_submission_binds_before_compilation_and_is_lazy(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2
) -> None:
    """A bound ISA circuit is submitted without fetching the Runtime result."""
    theta = Parameter("theta")
    circuit = QuantumCircuit(2)
    circuit.ry(theta, 0)
    metadata = ParameterMetadata(
        parameters=[ParameterInfo("theta", "theta", None, theta)]
    )
    executor = QiskitExecutor(backend, mode=backend)

    handle = executor.submit_sample(
        SampleRequest(CircuitInvocation(circuit, {"theta": math.pi}, metadata), 17)
    )

    assert runtime_sdk.samplers[0].mode is backend
    pubs, kwargs = runtime_sdk.samplers[0].calls[0]
    submitted = pubs[0]
    assert submitted.num_parameters == 0
    assert submitted.num_clbits == 2
    assert submitted.num_qubits == backend.num_qubits
    assert set(submitted.count_ops()) <= set(backend.operation_names) | {"barrier"}
    assert kwargs == {"shots": 17}
    assert handle.native.result_calls == 0
    assert circuit.num_parameters == 1
    assert circuit.num_clbits == 0
    assert handle.result() == {"01": 17}
    assert handle.native.result_calls == 1


def test_sampling_preserves_existing_classical_register_order(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2
) -> None:
    """Multiple named measurement registers decode to global classical order."""
    qubits = QuantumRegister(3, "qubits")
    left = ClassicalRegister(2, "left")
    right = ClassicalRegister(1, "right")
    circuit = QuantumCircuit(qubits, left, right)
    circuit.x(0)
    circuit.x(2)
    circuit.measure([0, 1, 2], [0, 1, 2])

    counts = QiskitExecutor(backend, mode=backend).execute(circuit, 13)

    submitted = runtime_sdk.samplers[0].calls[0][0][0]
    assert [(register.name, len(register)) for register in submitted.cregs] == [
        ("left", 2),
        ("right", 1),
    ]
    assert counts == {"101": 13}


def test_sampling_preserves_correlations_across_classical_registers(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2
) -> None:
    """Bell outcomes stay correlated when each qubit uses a separate register."""
    left = ClassicalRegister(1, "left")
    right = ClassicalRegister(1, "right")
    circuit = QuantumCircuit(QuantumRegister(2), left, right)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure(0, left[0])
    circuit.measure(1, right[0])

    counts = QiskitExecutor(backend, mode=backend).execute(circuit, 128)

    assert set(counts) == {"00", "11"}
    assert sum(counts.values()) == 128


def test_sampling_uses_global_clbit_order_when_register_order_differs(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2
) -> None:
    """Runtime register bitstrings are reordered by the circuit's clbit list."""
    left = ClassicalRegister(2, "left")
    right = ClassicalRegister(1, "right")
    circuit = QuantumCircuit(3)
    circuit.add_bits([left[1], right[0], left[0]])
    circuit.add_register(left, right)
    circuit.x(0)
    circuit.measure([0, 1, 2], [0, 1, 2])

    counts = QiskitExecutor(backend, mode=backend).execute(circuit, 13)

    assert counts == {"001": 13}


def test_empty_sampling_returns_without_submitting_a_job(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2
) -> None:
    """An empty circuit has one deterministic outcome and needs no Runtime job."""
    handle = QiskitExecutor(backend, mode=backend).submit_sample(
        SampleRequest(_invocation(QuantumCircuit(0)), 17)
    )

    assert handle.result() == {"": 17}
    assert runtime_sdk.samplers[0].calls == []


@pytest.mark.parametrize("kind", ["sample", "estimate"])
def test_local_jobs_without_service_do_not_expose_restoration_references(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2, kind: str
) -> None:
    """Local primitive jobs cannot advertise references that need a cloud service."""
    executor = QiskitExecutor(backend, mode=backend)
    invocation = _invocation(QuantumCircuit(1))
    if kind == "sample":
        handle = executor.submit_sample(SampleRequest(invocation, 3))
    else:
        handle = executor.submit_estimate(EstimateRequest(invocation, qm_o.Z(0)))

    assert handle.references() == ()
    assert not executor.capabilities.supports_restoration


def test_estimation_pads_observable_and_applies_nontrivial_physical_layout(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2
) -> None:
    """Observable support follows a logical qubit moved to physical qubit three."""
    circuit = QuantumCircuit(3)
    circuit.x(0)
    circuit.measure_all()
    hamiltonian = 2.0 * qm_o.Z(0)
    hamiltonian.constant = 0.25
    pass_manager = generate_preset_pass_manager(
        backend=backend, initial_layout=[3, 1, 4], optimization_level=1
    )
    executor = QiskitExecutor(backend, mode=backend, pass_manager=pass_manager)

    handle = executor.submit_estimate(
        EstimateRequest(_invocation(circuit), hamiltonian)
    )

    pubs, kwargs = runtime_sdk.estimators[0].calls[0]
    submitted, observable = pubs[0]
    assert submitted.num_clbits == 0
    assert "measure" not in submitted.count_ops()
    assert observable.num_qubits == submitted.num_qubits == backend.num_qubits
    assert dict(observable.to_list()) == {"IZIII": 2.0, "IIIII": 0.25}
    assert kwargs.get("precision") is None
    assert handle.native.result_calls == 0
    # X prepares |1>, so 2<Z> + 0.25 = -2 + 0.25 = -1.75.
    assert handle.result() == pytest.approx(-1.75, abs=1e-10, rel=1e-10)
    assert "measure" in circuit.count_ops()


def test_estimation_accepts_a_constant_on_a_wider_circuit(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2
) -> None:
    """A constant observable returns its value for a wider preparation circuit."""
    hamiltonian = qm_o.Hamiltonian()
    hamiltonian.constant = 2.5

    result = QiskitExecutor(backend, mode=backend).estimate(
        QuantumCircuit(3), hamiltonian
    )

    assert result == pytest.approx(2.5, abs=1e-10, rel=1e-10)


@pytest.mark.parametrize("width, seed", [(1, 7), (2, 19), (3, 101)])
def test_randomized_runtime_parameters_and_observables_preserve_logical_layout(
    runtime_sdk: SimpleNamespace,
    backend: GenericBackendV2,
    width: int,
    seed: int,
) -> None:
    """Seeded angles and mixed Pauli coefficients match a product-state oracle.

    For Ry(theta)|0>, X and Z expectations are sin(theta) and cos(theta).
    Independence makes each Zi Zj expectation the product of its local cosines.
    """
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-2.0 * math.pi, 2.0 * math.pi, width)
    x_coefficients = rng.uniform(-2.0, 2.0, width)
    z_coefficients = rng.uniform(-2.0, 2.0, width)
    constant = float(rng.uniform(-1.0, 1.0))
    observable = qm_o.Hamiltonian.identity(constant)
    expected = constant
    for index in range(width):
        observable += float(x_coefficients[index]) * qm_o.X(index)
        observable += float(z_coefficients[index]) * qm_o.Z(index)
        expected += x_coefficients[index] * math.sin(angles[index])
        expected += z_coefficients[index] * math.cos(angles[index])
    if width > 1:
        interaction = float(rng.uniform(-2.0, 2.0))
        observable += interaction * qm_o.Z(0) * qm_o.Z(width - 1)
        expected += interaction * math.cos(angles[0]) * math.cos(angles[-1])
    layout = [int(index) for index in rng.permutation(backend.num_qubits)[:width]]
    pass_manager = generate_preset_pass_manager(
        backend=backend, initial_layout=layout, optimization_level=1
    )
    executable = QiskitTranspiler().transpile(
        _product_state_expectation,
        bindings={"width": width, "observable": observable},
        parameters=["angles"],
    )

    result = executable.run(
        QiskitExecutor(backend, mode=backend, pass_manager=pass_manager),
        bindings={"angles": angles.tolist()},
    ).result()

    np.testing.assert_allclose(result, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("width", [0, 3])
def test_real_constant_observable_returns_without_submitting_a_job(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2, width: int
) -> None:
    """A real-valued complex identity coefficient needs no hardware execution."""
    hamiltonian = qm_o.Hamiltonian.identity(1.0 + 0.0j)

    result = QiskitExecutor(backend, mode=backend).estimate(
        QuantumCircuit(width), hamiltonian
    )

    assert result == pytest.approx(1.0, abs=1e-10, rel=1e-10)
    assert runtime_sdk.estimators[0].calls == []


@pytest.mark.parametrize("width", [0, 3])
def test_nonhermitian_constant_fails_before_submitting_a_job(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2, width: int
) -> None:
    """An imaginary identity coefficient is not silently cast to a real estimate."""
    hamiltonian = qm_o.Hamiltonian.identity(1.0 + 1.0j)
    executor = QiskitExecutor(backend, mode=backend)

    with pytest.raises(ValueError):
        executor.estimate(QuantumCircuit(width), hamiltonian)

    assert runtime_sdk.estimators[0].calls == []


@pytest.mark.parametrize(
    "hamiltonian",
    [
        1j * qm_o.X(0),
        -1j * qm_o.Y(0),
        (1.0 + 1e-9j) * qm_o.Z(0),
        qm_o.X(0) + 1j * qm_o.Z(1),
    ],
)
def test_nonhermitian_pauli_terms_fail_before_submitting_a_job(
    runtime_sdk: SimpleNamespace,
    backend: GenericBackendV2,
    hamiltonian: qm_o.Hamiltonian,
) -> None:
    """Every Pauli coefficient must be real before Runtime accepts a request."""
    executor = QiskitExecutor(backend, mode=backend)

    with pytest.raises(ValueError, match="Hermitian Hamiltonian"):
        executor.estimate(QuantumCircuit(2), hamiltonian)

    assert runtime_sdk.estimators[0].calls == []


@pytest.mark.parametrize("imaginary_residue", [0.0, 1e-10, -1e-10])
def test_hermitian_pauli_terms_with_numerical_residue_execute(
    runtime_sdk: SimpleNamespace,
    backend: GenericBackendV2,
    imaginary_residue: float,
) -> None:
    """Real-valued complex coefficients and tolerated roundoff retain the energy."""
    circuit = QuantumCircuit(1)
    circuit.h(0)
    hamiltonian = (2.0 + imaginary_residue * 1j) * qm_o.X(0)
    hamiltonian += (0.5 - imaginary_residue * 1j) * qm_o.Z(0)
    hamiltonian.constant = 0.25 + imaginary_residue * 1j
    original_terms = dict(hamiltonian.terms)

    result = QiskitExecutor(backend, mode=backend).estimate(circuit, hamiltonian)

    assert result == pytest.approx(2.25, abs=1e-10, rel=1e-10)
    assert len(runtime_sdk.estimators[0].calls) == 1
    assert hamiltonian.terms == original_terms
    assert hamiltonian.constant == 0.25 + imaginary_residue * 1j


@pytest.mark.parametrize("imaginary_residue", [1e-10, -1e-10])
@pytest.mark.parametrize("constant", [0.0, 0.25])
def test_imaginary_residue_only_terms_return_without_submitting_a_job(
    runtime_sdk: SimpleNamespace,
    backend: GenericBackendV2,
    imaginary_residue: float,
    constant: float,
) -> None:
    """Discarding tolerated imaginary terms leaves a locally evaluated constant."""
    hamiltonian = imaginary_residue * 1j * qm_o.X(0)
    hamiltonian.constant = constant + imaginary_residue * 1j

    result = QiskitExecutor(backend, mode=backend).estimate(
        QuantumCircuit(1), hamiltonian
    )

    assert result == constant
    assert runtime_sdk.estimators[0].calls == []


def test_target_precision_is_forwarded_to_estimator(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2
) -> None:
    """Per-request target precision reaches the Runtime V2 estimator unchanged."""
    executor = QiskitExecutor(backend, mode=backend)

    executor.submit_estimate(
        EstimateRequest(
            _invocation(QuantumCircuit(1)), qm_o.Z(0), TargetPrecision(0.07)
        )
    )

    assert runtime_sdk.estimators[0].calls[0][1] == {"precision": 0.07}


@pytest.mark.parametrize("accuracy", [Exact(), ShotBased(100)])
def test_unsupported_estimation_policies_fail_before_submission(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2, accuracy: Any
) -> None:
    """Runtime does not silently reinterpret exact values or shot counts."""
    executor = QiskitExecutor(backend, mode=backend)

    with pytest.raises(NotImplementedError):
        executor.submit_estimate(
            EstimateRequest(_invocation(QuantumCircuit(1)), qm_o.Z(0), accuracy)
        )

    assert runtime_sdk.estimators[0].calls == []


def test_capabilities_describe_runtime_features(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2
) -> None:
    """Runtime advertises its async lifecycle and supported precision policy."""
    capabilities = QiskitExecutor(backend, mode=backend).capabilities

    assert capabilities.supports_async_sampling
    assert capabilities.supports_async_estimation
    assert capabilities.supports_estimation
    assert capabilities.supports_cancellation
    assert capabilities.estimation_accuracy == frozenset({TargetPrecision})
    assert not capabilities.supports_native_batch
    assert not capabilities.supports_native_parameter_inputs


def test_composite_sampling_preserves_order_without_fetching_early(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2
) -> None:
    """Inherited batching submits all child requests and retains their order."""
    zero = QuantumCircuit(1)
    one = QuantumCircuit(1)
    one.x(0)
    executor = QiskitExecutor(backend, mode=backend)

    handle = executor.submit_samples(
        [SampleRequest(_invocation(zero), 7), SampleRequest(_invocation(one), 11)]
    )

    assert len(runtime_sdk.samplers[0].calls) == 2
    assert all(job.result_calls == 0 for job in runtime_sdk.jobs.values())
    assert handle.result() == ({"0": 7}, {"1": 11})


@pytest.mark.parametrize("theta, expected", [(0.0, (0, 0)), (math.pi, (1, 1))])
def test_parameterized_qkernel_sampling_executes_end_to_end(
    runtime_sdk: SimpleNamespace,
    backend: GenericBackendV2,
    theta: float,
    expected: tuple[int, int],
) -> None:
    """Public sample jobs bind runtime angles and reconstruct typed vector results."""
    executable = QiskitTranspiler().transpile(
        _parameterized_sample, parameters=["theta"]
    )

    job = executable.sample(
        QiskitExecutor(backend, mode=backend), shots=19, bindings={"theta": theta}
    )

    assert all(job.result_calls == 0 for job in runtime_sdk.jobs.values())
    assert job.result().results == [(expected, 19)]
    assert job.result().shots == 19


@pytest.mark.parametrize("theta, expected", [(0.0, 1.0), (math.pi, -1.0)])
def test_parameterized_qkernel_expectation_executes_end_to_end(
    runtime_sdk: SimpleNamespace,
    backend: GenericBackendV2,
    theta: float,
    expected: float,
) -> None:
    """Public run jobs bind parameters and execute through the Runtime estimator."""
    executable = QiskitTranspiler().transpile(
        _parameterized_expectation,
        bindings={"observable": qm_o.Z(0)},
        parameters=["theta"],
    )

    job = executable.run(
        QiskitExecutor(backend, mode=backend), bindings={"theta": theta}
    )

    assert all(job.result_calls == 0 for job in runtime_sdk.jobs.values())
    assert job.result() == pytest.approx(expected, abs=1e-10, rel=1e-10)


@pytest.mark.parametrize("kernel", [_loop_repeated_x, _loop_induction_sample])
def test_static_sampling_loops_lower_to_hardware_isa_without_mutating_source(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2, kernel: Any
) -> None:
    """Hardware targets without loop support receive bound, unrolled ISA gates."""
    executable = QiskitTranspiler().transpile(kernel)
    circuit = executable.quantum_circuit
    original = circuit.copy()
    assert "for_loop" in circuit.count_ops()
    assert executable.parameter_names == []

    result = executable.sample(QiskitExecutor(backend, mode=backend), shots=11).result()

    submitted = runtime_sdk.samplers[0].calls[0][0][0]
    assert "for_loop" not in submitted.count_ops()
    assert submitted.num_parameters == 0
    assert result.results == [(1, 11)]
    assert result.shots == 11
    assert circuit == original


def test_static_estimation_loop_binds_induction_angle_before_hardware_compilation(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2
) -> None:
    """A native loop's induction parameter yields the analytical cos(3) value."""
    executable = QiskitTranspiler().transpile(
        _loop_induction_expectation, bindings={"observable": qm_o.Z(0)}
    )
    circuit = executable.quantum_circuit
    original = circuit.copy()
    assert "for_loop" in circuit.count_ops()
    assert executable.parameter_names == []

    result = executable.run(QiskitExecutor(backend, mode=backend)).result()

    submitted = runtime_sdk.estimators[0].calls[0][0][0][0]
    assert "for_loop" not in submitted.count_ops()
    assert submitted.num_parameters == 0
    assert result == pytest.approx(math.cos(3.0), abs=1e-10, rel=1e-10)
    assert circuit == original


def test_restore_roundtrip_uses_explicit_service_without_resubmission(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2
) -> None:
    """A JSON-safe reference restores the original result through its service."""
    circuit = QuantumCircuit(2)
    circuit.x(0)
    restored_ids = []

    def retrieve(job_id: str) -> _RecordingJob:
        """Record restoration and retrieve the original sampling job.

        Args:
            job_id (str): Previously submitted native job identifier.

        Returns:
            _RecordingJob: Original local primitive job.

        Raises:
            KeyError: If the job identifier was not submitted by this fixture.
        """
        restored_ids.append(job_id)
        return runtime_sdk.jobs[job_id]

    executor = QiskitExecutor(
        backend, mode=backend, service=SimpleNamespace(job=retrieve)
    )
    original = executor.submit_sample(SampleRequest(_invocation(circuit), 5))
    reference = original.references()[0]
    reference = ExecutionReference.from_dict(
        json.loads(json.dumps(reference.to_dict()))
    )

    restored = executor.restore(reference)

    assert reference.provider == "qiskit-runtime"
    assert restored_ids == list(reference.job_ids)
    assert len(runtime_sdk.samplers[0].calls) == 1
    assert restored.native.result_calls == 0
    assert restored.result() == {"01": 5}


def test_restore_infers_service_from_backend(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2
) -> None:
    """An IBM backend's configured service can restore expectation jobs."""
    restored_ids = []

    def retrieve(job_id: str) -> _RecordingJob:
        """Record restoration and retrieve the original expectation job.

        Args:
            job_id (str): Previously submitted native job identifier.

        Returns:
            _RecordingJob: Original local primitive job.

        Raises:
            KeyError: If the job identifier was not submitted by this fixture.
        """
        restored_ids.append(job_id)
        return runtime_sdk.jobs[job_id]

    backend.service = SimpleNamespace(job=retrieve)
    executor = QiskitExecutor(backend, mode=backend)
    original = executor.submit_estimate(
        EstimateRequest(_invocation(QuantumCircuit(1)), qm_o.Z(0))
    )

    restored = executor.restore(original.references()[0])

    assert executor.capabilities.supports_restoration
    assert restored_ids == [original.native.job_id()]
    assert len(runtime_sdk.estimators[0].calls) == 1
    assert restored.result() == pytest.approx(1.0, abs=1e-10, rel=1e-10)


def test_public_sample_snapshot_restores_typed_results_without_resubmission(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2
) -> None:
    """A JSON snapshot reconstructs the public parameterized sample result ABI."""
    restored_ids = []

    def retrieve(job_id: str) -> _RecordingJob:
        """Record restoration and retrieve the job behind a public snapshot.

        Args:
            job_id (str): Previously submitted native job identifier.

        Returns:
            _RecordingJob: Original local primitive job.

        Raises:
            KeyError: If the job identifier was not submitted by this fixture.
        """
        restored_ids.append(job_id)
        return runtime_sdk.jobs[job_id]

    executor = QiskitExecutor(
        backend, mode=backend, service=SimpleNamespace(job=retrieve)
    )
    executable = QiskitTranspiler().transpile(
        _parameterized_sample, parameters=["theta"]
    )
    original = executable.sample(executor, shots=13, bindings={"theta": math.pi})
    snapshot = JobSnapshot.from_dict(
        json.loads(json.dumps(original.snapshot().to_dict()))
    )

    restored = executable.restore(executor, snapshot, bindings={"theta": math.pi})

    assert isinstance(restored, SampleJob)
    assert restored_ids == [original.native.job_id()]
    assert len(runtime_sdk.samplers[0].calls) == 1
    assert original.native.result_calls == 0
    assert restored.result().results == [((1, 1), 13)]
    assert restored.result().shots == 13


@pytest.mark.parametrize("reverse_order", [False, True])
def test_public_expectation_group_snapshot_restores_child_boundaries(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2, reverse_order: bool
) -> None:
    """One reference per child restores the ordered tuple without resubmission.

    Both states are |1>, so Z and 2Z have expectations -1 and -2.
    Reversing their order verifies that restoration retains child positions.
    """
    observables = (qm_o.Z(0), 2 * qm_o.Z(0))
    expected = (-1.0, -2.0)
    if reverse_order:
        observables = observables[::-1]
        expected = expected[::-1]
    restored_ids = []

    def retrieve(job_id: str) -> _RecordingJob:
        """Record restoration of an individual expectation job.

        Args:
            job_id (str): Previously submitted native job identifier.

        Returns:
            _RecordingJob: Original local primitive job.

        Raises:
            KeyError: If the identifier does not belong to a fixture job.
        """
        restored_ids.append(job_id)
        return runtime_sdk.jobs[job_id]

    executor = QiskitExecutor(
        backend, mode=backend, service=SimpleNamespace(job=retrieve)
    )
    executable = QiskitTranspiler().transpile(
        _paired_expectations,
        bindings={
            "first_observable": observables[0],
            "second_observable": observables[1],
        },
    )
    original = executable.run(executor)
    snapshot = JobSnapshot.from_dict(
        json.loads(json.dumps(original.snapshot().to_dict()))
    )

    restored = executable.restore(executor, snapshot)

    assert len(snapshot.executions) == 2
    assert tuple(reference.job_ids for reference in snapshot.executions) == tuple(
        (native.job_id(),) for native in original.native
    )
    assert restored_ids == [native.job_id() for native in original.native]
    assert len(runtime_sdk.estimators[0].calls) == 2
    assert runtime_sdk.samplers[0].calls == []
    assert all(native.result_calls == 0 for native in runtime_sdk.jobs.values())
    assert type(restored) is type(original)
    np.testing.assert_allclose(restored.result(), expected, atol=1e-12, rtol=1e-12)
    assert len(runtime_sdk.estimators[0].calls) == 2
    assert all(native.result_calls == 1 for native in runtime_sdk.jobs.values())


@pytest.mark.parametrize(
    "observables, expected, remote_count",
    [
        ((qm_o.Z(0), qm_o.Hamiltonian.identity(0.25)), (-1.0, 0.25), 1),
        ((qm_o.Hamiltonian.identity(0.25), qm_o.Z(0)), (0.25, -1.0), 1),
        (
            (qm_o.Hamiltonian.identity(0.25), qm_o.Hamiltonian.identity(-1.5)),
            (0.25, -1.5),
            0,
        ),
    ],
    ids=["remote-then-local", "local-then-remote", "all-local"],
)
def test_local_expectation_group_snapshot_restores_values_and_positions(
    runtime_sdk: SimpleNamespace,
    backend: GenericBackendV2,
    observables: tuple[qm_o.Hamiltonian, qm_o.Hamiltonian],
    expected: tuple[float, float],
    remote_count: int,
) -> None:
    """JSON snapshots preserve local constants alongside deferred remote values.

    The identity has expectation 0.25 in every state, and Z on |1> gives -1.
    Snapshotting and restoration do not fetch results or submit replacements.
    """
    executable = QiskitTranspiler().transpile(
        _paired_expectations,
        bindings={
            "first_observable": observables[0],
            "second_observable": observables[1],
        },
    )
    executor = QiskitExecutor(backend, mode=backend, service=runtime_sdk.Service())

    original = executable.run(executor)
    snapshot = JobSnapshot.from_dict(
        json.loads(json.dumps(original.snapshot().to_dict()))
    )
    restored = executable.restore(executor, snapshot)

    assert len(snapshot.executions) == remote_count
    assert len(runtime_sdk.estimators[0].calls) == remote_count
    assert runtime_sdk.samplers[0].calls == []
    assert all(native.result_calls == 0 for native in runtime_sdk.jobs.values())
    assert type(restored) is type(original)
    result = restored.result()
    assert type(result) is tuple
    assert all(type(value) is float for value in result)
    np.testing.assert_allclose(result, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(original.result(), result, atol=1e-12, rtol=1e-12)
    assert len(runtime_sdk.estimators[0].calls) == remote_count


@pytest.mark.parametrize("constant", [0.0, 0.25, -2.5])
def test_local_scalar_expectation_snapshot_restores_without_provider_jobs(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2, constant: float
) -> None:
    """A scalar constant restores its public float type and original bindings."""
    executable = QiskitTranspiler().transpile(
        _parameterized_expectation,
        bindings={"observable": qm_o.Hamiltonian.identity(constant)},
        parameters=["theta"],
    )
    executor = QiskitExecutor(backend, mode=backend)
    bindings = {"theta": math.pi}

    original = executable.run(executor, bindings=bindings)
    snapshot_data = json.loads(json.dumps(original.snapshot().to_dict()))
    restored = executable.restore(
        executor, JobSnapshot.from_dict(snapshot_data), bindings=bindings
    )

    assert snapshot_data["executions"] == []
    assert "theta" not in json.dumps(snapshot_data)
    assert not executor.capabilities.supports_restoration
    assert type(restored) is type(original)
    assert type(restored.result()) is float
    np.testing.assert_allclose(
        [restored.result(), original.result()],
        [constant, constant],
        atol=1e-12,
        rtol=1e-12,
    )
    assert runtime_sdk.jobs == {}
    assert runtime_sdk.estimators[0].calls == []
    assert runtime_sdk.samplers[0].calls == []


@pytest.mark.parametrize("kind", ["sample", "run"])
def test_empty_public_execution_snapshot_restores_without_provider_jobs(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2, kind: str
) -> None:
    """Empty raw counts restore through the public sample and run conversions."""
    executable = QiskitTranspiler().transpile(_empty_sample)
    executor = QiskitExecutor(backend, mode=backend, service=runtime_sdk.Service())
    original = (
        executable.sample(executor, shots=17)
        if kind == "sample"
        else executable.run(executor)
    )
    snapshot = JobSnapshot.from_dict(
        json.loads(json.dumps(original.snapshot().to_dict()))
    )

    restored = executable.restore(executor, snapshot)

    assert snapshot.executions == ()
    assert type(restored) is type(original)
    if kind == "sample":
        assert isinstance(restored, SampleJob)
        assert restored.result().results == original.result().results == [((), 17)]
        assert restored.result().shots == original.result().shots == 17
    else:
        assert type(restored.result()) is tuple
        assert restored.result() == original.result() == ()
    assert runtime_sdk.jobs == {}
    assert runtime_sdk.estimators[0].calls == []
    assert runtime_sdk.samplers[0].calls == []


@pytest.mark.parametrize(
    "change",
    [
        {"provider": "other-provider"},
        {"target": "other-backend"},
        {"target": None},
        {"job_ids": ("first-job", "second-job")},
        {"context": {}},
        {"context": {"kind": "unknown"}},
        {"context": {"kind": "sample"}},
        {"context": {"kind": "sample", "bits": "not JSON"}},
        {"context": {"kind": "sample", "bits": "{}"}},
        {"context": {"kind": "sample", "bits": "[]"}},
        {"context": {"kind": "sample", "bits": '["meas", 0]'}},
        {"context": {"kind": "sample", "bits": '[["", 0]]'}},
        {"context": {"kind": "sample", "bits": '[["meas", -1]]'}},
        {"context": {"kind": "sample", "bits": '[["meas", true]]'}},
        {"context": {"kind": "sample", "bits": '[["meas", 0.5]]'}},
    ],
)
def test_restore_rejects_invalid_references_before_contacting_service(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2, change: dict[str, Any]
) -> None:
    """Malformed provider, target, kind, and decoder metadata never contact IBM."""

    def forbidden_retrieval(job_id: str) -> None:
        """Fail if malformed metadata reaches provider job lookup.

        Args:
            job_id (str): Job identifier that must not be retrieved.

        Raises:
            pytest.fail.Exception: Always, because invalid references are rejected locally.
        """
        pytest.fail("Invalid restoration must not retrieve a provider job")

    executor = QiskitExecutor(
        backend, mode=backend, service=SimpleNamespace(job=forbidden_retrieval)
    )
    fields = {
        "provider": "qiskit-runtime",
        "job_ids": ("saved-job",),
        "target": backend.name,
        "context": {"kind": "sample", "bits": '[["meas", 0]]'},
    }
    fields.update(change)

    with pytest.raises(ValueError):
        executor.restore(ExecutionReference(**fields))


def test_restore_without_service_is_explicitly_unsupported(
    runtime_sdk: SimpleNamespace, backend: GenericBackendV2
) -> None:
    """Local execution targets require an explicit Runtime restoration service."""
    executor = QiskitExecutor(backend, mode=backend)
    reference = ExecutionReference(
        provider="qiskit-runtime",
        job_ids=("saved-job",),
        target=backend.name,
        context={"kind": "estimate"},
    )

    assert not executor.capabilities.supports_restoration
    with pytest.raises(NotImplementedError, match="requires a service"):
        executor.restore(reference)


@pytest.mark.parametrize("theta, expected", [(0.0, 0), (math.pi, 1)])
def test_real_runtime_local_mode_restores_public_jobs_without_service(
    theta: float, expected: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real Runtime sample and nonconstant estimates restore cached local values."""
    pytest.importorskip("qiskit_ibm_runtime")
    local_backend = GenericBackendV2(num_qubits=2, noise_info=False, seed=23)
    executor = QiskitExecutor(local_backend, mode=local_backend)
    transpiler = QiskitTranspiler()
    sample_program = transpiler.transpile(_parameterized_sample, parameters=["theta"])
    estimate_program = transpiler.transpile(
        _parameterized_expectation,
        bindings={"observable": qm_o.Z(0)},
        parameters=["theta"],
    )

    bindings = {"theta": theta}
    sample_job = sample_program.sample(executor, shots=32, bindings=bindings)
    estimate_job = estimate_program.run(
        executor, bindings=bindings, estimation=TargetPrecision(0.125)
    )
    assert not executor.capabilities.supports_restoration
    for job in (sample_job, estimate_job):
        assert job.references() == ()
        with pytest.raises(ValueError, match=r"call result\(\) successfully"):
            job.snapshot()

    sample_result = sample_job.result()
    expectation = estimate_job.result()

    forbidden_calls = []
    for target, methods in (
        (executor, ("submit_sample", "submit_estimates", "restore")),
        (sample_job.native, ("result", "status")),
        (estimate_job.native[0], ("result", "status")),
    ):
        for method in methods:
            forbidden = Mock(
                side_effect=AssertionError("Cached snapshots need no provider calls")
            )
            monkeypatch.setattr(target, method, forbidden)
            forbidden_calls.append(forbidden)

    snapshots = [
        JobSnapshot.from_dict(
            json.loads(json.dumps(job.snapshot().to_dict(), allow_nan=False))
        )
        for job in (sample_job, estimate_job)
    ]
    assert all(snapshot.executions == () for snapshot in snapshots)
    assert snapshots[0].execution.kind is ExecutionSnapshotKind.LOCAL
    assert snapshots[1].execution.children[0].kind is ExecutionSnapshotKind.LOCAL
    restored_sample = sample_program.restore(executor, snapshots[0], bindings)
    restored_estimate = estimate_program.restore(executor, snapshots[1], bindings)

    assert sample_result.results == [((expected, expected), 32)]
    assert sample_result.shots == 32
    assert expectation == pytest.approx(1.0 - 2.0 * expected, abs=1e-10, rel=1e-10)
    assert type(restored_sample) is type(sample_job)
    assert type(restored_estimate) is type(estimate_job)
    assert restored_sample.result() == sample_result
    assert type(restored_estimate.result()) is float
    assert restored_estimate.result() == pytest.approx(
        expectation, abs=1e-10, rel=1e-10
    )
    for forbidden in forbidden_calls:
        forbidden.assert_not_called()


def test_real_runtime_local_mode_accepts_qamomile_execution_options() -> None:
    """The real SDK accepts Qamomile options for public sampling and estimation.

    A pi rotation prepares |11>, whose first-qubit Z expectation is -1.
    Runtime local mode ignores resilience settings but validates their schema.
    """
    pytest.importorskip("qiskit_ibm_runtime")
    backend = GenericBackendV2(num_qubits=2, noise_info=False, seed=23)
    executor = QiskitTranspiler().executor(
        backend,
        mode=backend,
        options=QiskitExecutionOptions(
            max_execution_time=300,
            resilience_level=0,
            sampler_options={"default_shots": 16},
            estimator_options={"default_precision": 0.125},
        ),
    )
    transpiler = QiskitTranspiler()
    sample_program = transpiler.transpile(_parameterized_sample, parameters=["theta"])
    estimate_program = transpiler.transpile(
        _parameterized_expectation,
        bindings={"observable": qm_o.Z(0)},
        parameters=["theta"],
    )

    sampled = sample_program.sample(
        executor, shots=16, bindings={"theta": math.pi}
    ).result()
    with pytest.warns(UserWarning, match="resilience_level.*no effect"):
        estimated = estimate_program.run(executor, bindings={"theta": math.pi}).result()

    assert sampled.results == [((1, 1), 16)]
    assert sampled.shots == 16
    np.testing.assert_allclose(estimated, -1.0, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("kernel", [_loop_repeated_x, _loop_induction_sample])
def test_real_runtime_local_mode_executes_static_sampling_loops(kernel: Any) -> None:
    """Real Runtime local mode accepts native qkernel loops after ISA lowering."""
    pytest.importorskip("qiskit_ibm_runtime")
    backend = GenericBackendV2(num_qubits=2, noise_info=False, seed=23)
    executable = QiskitTranspiler().transpile(kernel)
    assert "for_loop" in executable.quantum_circuit.count_ops()

    result = executable.sample(QiskitExecutor(backend, mode=backend), shots=16).result()

    assert result.results == [(1, 16)]
    assert result.shots == 16


def test_real_runtime_local_mode_estimates_static_loop_induction_angles() -> None:
    """Real Runtime estimation preserves a fixed loop's accumulated Rx angle."""
    pytest.importorskip("qiskit_ibm_runtime")
    backend = GenericBackendV2(num_qubits=2, noise_info=False, seed=23)
    executable = QiskitTranspiler().transpile(
        _loop_induction_expectation, bindings={"observable": qm_o.Z(0)}
    )
    assert "for_loop" in executable.quantum_circuit.count_ops()

    result = executable.run(
        QiskitExecutor(backend, mode=backend), estimation=TargetPrecision(0.025)
    ).result()

    assert result == pytest.approx(math.cos(3.0), abs=0.03, rel=1e-10)
