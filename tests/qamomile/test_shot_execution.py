"""Closed quantum frontiers preserve conditional populations and correlations."""

from dataclasses import replace

import pytest

import qamomile as qm
from qamomile._native import compiler as native
from qamomile.effects import Requirement

SOURCE = """
import qamomile as qm
@qm.qkernel
def terminal(value: qm.Param[qm.Real]) -> qm.Shot[qm.Real]:
    a = qm.measure(qm.prepare(False))
    if a:
        b = 1.0 / value
    else:
        b = 2.0
    return b
@qm.qkernel
def terminal_return(value: qm.Param[qm.Real]) -> qm.Shot[qm.Real]:
    a = qm.measure(qm.prepare(False))
    if a:
        return 1.0 / value
    else:
        return 2.0
@qm.qkernel
def discarded(check: qm.Const[qm.Bool], value: qm.Param[qm.Real]) -> qm.Shot[qm.Bool]:
    a = qm.measure(qm.prepare(check))
    if a:
        f = 1.0 / value
    else:
        f = 2.0
    return a
@qm.qkernel
def independent() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bool]]:
    a = qm.measure(qm.h(qm.prepare(False)))
    b = qm.prepare(False)
    if a:
        b = qm.x(b)
    return (a, qm.measure(b))
@qm.qkernel
def fresh() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bool]]:
    a = qm.measure(qm.h(qm.prepare(False)))
    b = qm.measure(qm.h(qm.prepare(False)))
    return (a, b)
@qm.qkernel
def angle() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bool]]:
    a = qm.measure(qm.h(qm.prepare(False)))
    f = 3.141592653589793 if a else 0.0
    b = qm.ry(qm.prepare(False), f)
    return (a, qm.measure(b))
@qm.qkernel
def prepare_bit() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bool]]:
    a = qm.measure(qm.h(qm.prepare(False)))
    return (a, qm.measure(qm.prepare(a)))
@qm.qkernel
def scaled(scale: qm.Param[qm.Real]) -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bool]]:
    a = qm.measure(qm.h(qm.prepare(False)))
    f = 3.141592653589793 * scale if a else 0.0
    return (a, qm.measure(qm.ry(qm.prepare(False), f)))
@qm.qkernel
def checked(value: qm.Param[qm.Real]) -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Real], qm.Shot[qm.Bool]]:
    a = qm.measure(qm.prepare(True))
    if a:
        f = 1.0 / value
    else:
        f = 2.0
    b = qm.prepare(a)
    return (a, f, qm.measure(b))
@qm.qkernel
def guarded_sqrt(value: qm.Param[qm.Real]) -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Real], qm.Shot[qm.Bool]]:
    a = qm.measure(qm.prepare(False))
    if a:
        f = qm.sqrt(value)
    else:
        f = 2.0
    return (a, f, qm.measure(qm.prepare(a)))
@qm.qkernel
def skipped() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bool]]:
    a = qm.measure(qm.prepare(False))
    if a:
        b = qm.measure(qm.prepare(True))
    else:
        b = False
    return (a, b)
@qm.qkernel
def branched_measurements() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bool]]:
    a = qm.measure(qm.h(qm.prepare(False)))
    if a:
        b = qm.measure(qm.prepare(True))
    else:
        b = qm.measure(qm.prepare(False))
    return (a, b)
@qm.qkernel
def completion_only() -> qm.Shot[qm.Bool]:
    a = qm.measure(qm.h(qm.prepare(False)))
    if a:
        b = qm.measure(qm.prepare(True))
    else:
        q = qm.x(qm.prepare(False))
        qm.discard(q)
        b = False
    return a
@qm.qkernel
def live_angle() -> qm.Shot[qm.Bool]:
    b = qm.prepare(False)
    a = qm.measure(qm.h(qm.prepare(False)))
    f = 3.141592653589793 if a else 0.0
    return qm.measure(qm.ry(b, f))
@qm.qkernel
def live_arithmetic() -> qm.Shot[qm.Bool]:
    b = qm.prepare(False)
    a = qm.measure(qm.h(qm.prepare(False)))
    f = 1.0 + (3.0 if a else 0.0)
    return qm.measure(qm.ry(b, f))
@qm.qkernel
def future_join() -> qm.Shot[qm.Bits[2]]:
    survivor = qm.h(qm.qubits(1))
    a = qm.measure(qm.h(qm.prepare(False)))
    b = qm.qubits(1)
    if a:
        b = qm.x(b)
    joined = qm.concat(survivor, b)
    joined = qm.cx(joined, 0, 1)
    return qm.measure(joined)
@qm.qkernel
def early_preparation() -> qm.Shot[qm.Bool]:
    b = qm.prepare(False)
    a = qm.measure(qm.h(qm.prepare(False)))
    if a:
        b = qm.x(b)
    return qm.measure(b)
"""


@pytest.fixture
def qiskit_available():
    """Skip SDK witnesses when the optional simulator is not installed."""
    pytest.importorskip("qiskit_aer")


def program(name):
    """Build a minimal source witness at the actual frontend boundary."""
    return qm.build_source(SOURCE).specialize(name)


def test_native_cohorts_preserve_exact_population_and_interface():
    """Weighted native cohorts request 700/300 shots without any simulation."""
    p = program("independent")
    state = native.ShotExecutionHandle(p.to_bytes(), {}, 1000, 7)
    first = state.next_request()
    assert first.shots == 1000
    assert first.circuit.width == 1
    assert first.circuit.measurement_count == 1
    with pytest.raises(ValueError, match="outstanding request"):
        state.next_request()
    with pytest.raises(ValueError, match="shot count"):
        state.accept([([0], 699), ([1], 300)])
    state.accept([([0], 700), ([1], 300)])
    requests = []
    while (request := state.next_request()) is not None:
        requests.append(request.shots)
        has_x = any(op.opcode == "x" for op in request.circuit.instructions)
        assert all(op.opcode != "if" for op in request.circuit.instructions)
        state.accept([([int(has_x)], request.shots)])
    assert sorted(requests) == [300, 700]
    assert dict((tuple(row), n) for row, n in state.finish()) == {
        (0, 0): 700,
        (1, 1): 300,
    }
    assert state.physical_shots == 2000
    assert state.submissions == 3


@pytest.mark.parametrize("name", ["future_join", "early_preparation"])
@pytest.mark.usefixtures("qiskit_available")
def test_live_quantum_frontier_prevents_host_fallback(name):
    """A future join and a previously prepared survivor both require feedback."""
    from qamomile.qiskit import Qiskit

    target = Qiskit(requirements=Qiskit().requirements - {Requirement.DYNAMIC_BRANCH})
    with pytest.raises(qm.ArtifactError, match="Q7015.*dynamic_branch"):
        qm.compile(program(name), to=target)


@pytest.mark.usefixtures("qiskit_available")
@pytest.mark.parametrize("name", ["terminal", "terminal_return"])
def test_terminal_classical_if_preserves_lazy_evaluation(name):
    """Classical-only Shot control does not require a dynamic quantum branch."""
    from qamomile.circuit import lower_to_circuit
    from qamomile.qiskit import Aer, Qiskit

    p = program(name)
    target = Qiskit(requirements=Qiskit().requirements - {Requirement.DYNAMIC_BRANCH})
    artifact = qm.compile(p, to=target)
    assert Requirement.DYNAMIC_BRANCH not in artifact.execution_contract.requirements
    assert [op.opcode for op in lower_to_circuit(p).body.instructions] == ["measure"]
    assert qm.execute(
        qm.compile(p, to=Aer().target), on=Aer(), parameters={"value": 0.0}, shots=8
    ).values == {2.0: 8}


@pytest.mark.parametrize(
    "name", ["independent", "angle", "prepare_bit", "branched_measurements"]
)
@pytest.mark.usefixtures("qiskit_available")
def test_qiskit_executes_conditional_shot_segments(name):
    """Dependent fresh states preserve the measured value's joint pairing."""
    from qamomile.qiskit import Aer

    endpoint = Aer()
    artifact = qm.compile(program(name), to=endpoint.target)
    assert artifact.format == "qamomile.shot-execution.v1"
    result = qm.execute(artifact, on=endpoint, shots=128, seed=7)
    assert set(result.values) == {(False, False), (True, True)}
    assert result.shots == 128


@pytest.mark.usefixtures("qiskit_available")
def test_independent_fresh_measurements_are_not_reused():
    """Equal quantum preparations still provide separate stochastic samples."""
    from qamomile.qiskit import Aer

    result = qm.run(program("fresh"), on=Aer(), shots=128, seed=7).result
    assert len(result.values) == 4


@pytest.mark.usefixtures("qiskit_available")
def test_shot_plan_persistence_rechecks_source_and_cuts():
    """Restored recipes retain conditional semantics and reject a forged ABI."""
    from qamomile.persistence import deserialize, serialize
    from qamomile.qiskit import Aer

    endpoint = Aer()
    artifact = deserialize(
        serialize(qm.compile(program("independent"), to=endpoint.target))
    )
    assert qm.execute(artifact, on=endpoint, shots=32, seed=7).shots == 32
    state = native.ShotExecutionHandle(program("independent").to_bytes(), {}, 8, 0)
    request = state.next_request()
    native.validate_shot_segment_interface(
        program("independent").to_bytes(), request.interface, b"", request.result_abi
    )
    with pytest.raises(ValueError, match="ABI"):
        native.validate_shot_segment_interface(
            program("independent").to_bytes(),
            request.interface,
            b"",
            artifact.result_abi._to_bytes(),
        )
    with pytest.raises(qm.ExecutionError, match="source"):
        qm.execute(
            replace(
                artifact, payload=replace(artifact.payload, program=program("fresh"))
            ),
            on=endpoint,
        )


@pytest.mark.usefixtures("qiskit_available")
def test_launch_parameters_and_guarded_numeric_continuations():
    """Launch binding and untaken domain failures stay native and lazy."""
    from qamomile.qiskit import Aer

    endpoint = Aer()
    scaled = qm.compile(program("scaled"), to=endpoint.target)
    assert set(
        qm.execute(
            scaled, on=endpoint, parameters={"scale": 1.0}, shots=64, seed=7
        ).values
    ) == {(False, False), (True, True)}
    guarded = qm.compile(program("guarded_sqrt"), to=endpoint.target)
    assert qm.execute(
        guarded, on=endpoint, parameters={"value": -1.0}, shots=8
    ).values == {(False, 2.0, False): 8}


def test_classical_failure_prevents_the_next_quantum_request():
    """A source check fails before preparing the downstream quantum segment."""
    state = native.ShotExecutionHandle(
        program("checked").to_bytes(), {"value": 0.0}, 8, 7
    )
    first = state.next_request()
    assert first.shots == 8
    state.accept([([1], 8)])
    with pytest.raises(ValueError, match="division by zero"):
        state.next_request()
    assert state.submissions == 1
    assert state.physical_shots == 8


@pytest.mark.usefixtures("qiskit_available")
def test_skipped_quantum_branch_consumes_no_physical_shots():
    """Inactive measurement slots are internal placeholders, not submissions."""
    from qamomile.qiskit import Aer

    record = qm.run(program("skipped"), on=Aer(), shots=8)
    assert record.result.values == {(False, False): 8}
    assert dict(record.result.execution_statistics.entries) == {
        "logical_shots": 8,
        "physical_segment_shots": 8,
        "submissions": 1,
    }


@pytest.mark.usefixtures("qiskit_available")
def test_finite_live_angle_choice_uses_dynamic_circuit():
    """A live quantum survivor can use a finite measured angle selector."""
    from qamomile.qiskit import Aer

    endpoint = Aer()
    artifact = qm.compile(program("live_angle"), to=endpoint.target)
    assert artifact.format != "qamomile.shot-execution.v1"
    assert Requirement.DYNAMIC_BRANCH in artifact.execution_contract.requirements
    assert Requirement.DYNAMIC_ROTATION not in artifact.execution_contract.requirements
    assert set(qm.execute(artifact, on=endpoint, shots=64, seed=7).values) == {
        False,
        True,
    }
    with pytest.raises(qm.ArtifactError, match="dynamic_rotation"):
        qm.compile(program("live_arithmetic"), to=endpoint.target)


@pytest.mark.usefixtures("qiskit_available")
def test_terminal_checks_are_not_eliminated_with_unused_results():
    """An evaluated expression can fail even when its result is not returned."""
    from qamomile.qiskit import Aer

    endpoint = Aer()
    module = qm.build_source(SOURCE)
    safe = qm.compile(module.specialize("discarded", check=False), to=endpoint.target)
    assert qm.execute(safe, on=endpoint, parameters={"value": 0.0}, shots=8).values == {
        False: 8
    }
    invalid = qm.compile(module.specialize("discarded", check=True), to=endpoint.target)
    with pytest.raises(qm.ExecutionError, match="division by zero"):
        qm.execute(invalid, on=endpoint, parameters={"value": 0.0}, shots=8)


def test_native_population_budget_rejects_before_row_expansion():
    """A wide distribution is rejected atomically and cannot complete a plan."""
    state = native.ShotExecutionHandle(program("independent").to_bytes(), {}, 8, 7)
    with pytest.raises(ValueError, match="incomplete"):
        state.finish()
    state.next_request()
    with pytest.raises(ValueError, match="measurement row"):
        state.accept([([2], 8)])
    assert state.submissions == 0
    state.accept([([0], 8)])
    state.next_request()
    state.accept([([0], 8)])
    assert state.next_request() is None
    assert sum(n for _, n in state.finish()) == 8
    excessive = native.ShotExecutionHandle(
        program("independent").to_bytes(), {}, 1_000_001, 0
    )
    with pytest.raises(ValueError, match="budget.*before submission"):
        excessive.next_request()
    assert excessive.submissions == 0


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("cirq", marks=pytest.mark.cirq),
        pytest.param("quri_parts", marks=pytest.mark.quri_parts),
        pytest.param("braket", marks=pytest.mark.braket),
        pytest.param("cudaq", marks=pytest.mark.cudaq),
        pytest.param("quration", marks=pytest.mark.quration),
    ],
)
def test_closed_plan_uses_existing_backend_transport(backend):
    """The same native cohort plan runs through each capable local endpoint."""
    if backend == "cirq":
        pytest.importorskip("cirq")
        from qamomile.cirq import CirqSimulator

        endpoint = CirqSimulator()
    elif backend == "quri_parts":
        pytest.importorskip("qulacs")
        from qamomile.quri_parts import Qulacs

        endpoint = Qulacs()
    elif backend == "braket":
        pytest.importorskip("braket")
        from qamomile.braket import BraketLocal

        endpoint = BraketLocal()
    elif backend == "cudaq":
        pytest.importorskip("cudaq")
        from qamomile.cudaq import CudaqLocal

        endpoint = CudaqLocal()
    else:
        pytest.importorskip("pyqret")
        from qamomile.quration import QurationLocal

        endpoint = QurationLocal()
    source = program("independent")
    record = qm.run(source, on=endpoint, shots=64)
    assert set(record.result.values) == {(False, False), (True, True)}
    assert record.result.shots == 64
    assert (
        dict(record.result.execution_statistics.entries)["physical_segment_shots"]
        == 128
    )
    branched = qm.run(program("branched_measurements"), on=endpoint, shots=64)
    assert set(branched.result.values) == {(False, False), (True, True)}
    assert branched.result.shots == 64
    reference = qm.run(source, on=qm.Reference(), shots=64, seed=7)
    assert set(reference.result.values) == set(record.result.values)


@pytest.mark.usefixtures("qiskit_available")
def test_timeout_retains_child_and_seeded_occurrences_are_distinct():
    """Retrieval retries neither resubmit completed work nor restart a stream."""
    from qamomile.compilation import JobStatus
    from qamomile.qiskit import Aer

    class Delayed:
        def __init__(self, inner):
            self.inner = inner
            self.waited = False

        @property
        def job_id(self):
            return self.inner.job_id

        @property
        def status(self):
            return JobStatus.RUNNING if not self.waited else self.inner.status

        def result(self, timeout=None):
            if not self.waited:
                self.waited = True
                raise TimeoutError("first retrieval expires")
            return self.inner.result(timeout)

        def cancel(self):
            self.inner.cancel()

        def snapshot(self):
            return self.inner.snapshot()

    class Counting:
        def __init__(self):
            self.delegate = Aer()
            self.seeds = []

        @property
        def name(self):
            return self.delegate.name

        @property
        def target(self):
            return self.delegate.target

        @property
        def capabilities(self):
            return self.delegate.capabilities

        def launch(self, artifact, invocation):
            self.seeds.append(invocation.seed)
            child = self.delegate.launch(artifact, invocation)
            return Delayed(child) if len(self.seeds) == 1 else child

    endpoint = Counting()
    artifact = qm.compile(program("independent"), to=endpoint.target)
    job = qm.submit(artifact, on=endpoint, shots=64)
    assert len(endpoint.seeds) == 1
    with pytest.raises(qm.ExecutionError, match="in-flight"):
        job.snapshot()
    with pytest.raises(TimeoutError):
        job.result()
    assert len(endpoint.seeds) == 1
    result = job.result()
    assert result.shots == 64
    assert len(endpoint.seeds) == len(set(endpoint.seeds)) == 3
    assert endpoint.seeds[0] == 0
    repeated = qm.execute(artifact, on=endpoint.delegate, shots=64)
    assert repeated.counts == result.counts
    completed = qm.ExecutionSnapshot.from_bytes(job.snapshot().to_bytes())
    assert completed.result.execution_statistics == result.execution_statistics
    job.cancel()
    assert job.result() == result
    other = Counting()
    cancelled = qm.submit(artifact, on=other, shots=8)
    cancelled.cancel()
    assert cancelled.status == JobStatus.CANCELLED
    with pytest.raises(qm.ExecutionError, match="cancelled"):
        cancelled.result()
    assert len(other.seeds) == 1


@pytest.mark.usefixtures("qiskit_available")
def test_closed_plan_discharges_mid_circuit_device_requirement():
    """Each closed child has final measurements on a capability-limited target."""
    from qamomile.qiskit import Qiskit

    target = Qiskit(
        requirements=Qiskit().requirements
        - {Requirement.DYNAMIC_BRANCH, Requirement.MID_CIRCUIT_MEASUREMENT}
    )
    artifact = qm.compile(program("independent"), to=target)
    assert artifact.format == "qamomile.shot-execution.v1"
    assert (
        Requirement.MID_CIRCUIT_MEASUREMENT
        not in artifact.execution_contract.requirements
    )


@pytest.mark.hugr
def test_rich_target_retains_the_whole_correlated_program():
    """Compatible HUGR control uses one whole program with the same joint result."""
    pytest.importorskip("hugr")
    from qamomile.hugr import Selene

    source = program("independent")
    endpoint = Selene(timeout=30.0)
    artifact = qm.compile(source, to=endpoint.target)
    assert artifact.format != "qamomile.shot-execution.v1"
    result = qm.execute(artifact, on=endpoint, shots=64, seed=7)
    assert set(result.values) == {(False, False), (True, True)}
    assert result.shots == 64


@pytest.mark.usefixtures("qiskit_available")
def test_completion_only_quantum_branch_stays_fused():
    """Quantum work without readout uses a supported whole-circuit route."""
    from qamomile.qiskit import Aer

    endpoint = Aer()
    artifact = qm.compile(program("completion_only"), to=endpoint.target)
    assert artifact.format != "qamomile.shot-execution.v1"
    assert set(qm.execute(artifact, on=endpoint, shots=64, seed=7).values) == {
        False,
        True,
    }
