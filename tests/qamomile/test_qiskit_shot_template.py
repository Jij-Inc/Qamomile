"""Verify fixed-template Qiskit execution without per-shot compilation."""

import math
from types import SimpleNamespace

import pytest

import qamomile as qm

_SOURCE = """
import qamomile as qm

@qm.qkernel
def sample(offset: qm.Param[qm.Real]) -> tuple[qm.Shot[qm.Bit], qm.Shot[qm.Bool]]:
    index = qm.get_current_shot()
    angle = float(index % 2) * 3.141592653589793 + offset
    measured = qm.measure(qm.ry(qm.prepare(False), angle))
    return measured, index % 2 == 1
"""


def _forbid_compilation(monkeypatch):
    """Make any launch-time source compilation or SDK emission fail visibly."""
    import qamomile.compilation as compilation
    import qamomile.qiskit._integration as integration

    def forbidden(*args, **kwargs):
        raise AssertionError("execution must reuse its compiled SDK template")

    monkeypatch.setattr(compilation, "compile", forbidden)
    monkeypatch.setattr(integration, "_emit_circuit", forbidden)
    monkeypatch.setattr(integration, "_emit_for_target", forbidden)


@pytest.mark.parametrize("endpoint_name", ["aer", "backend"])
def test_fixed_template_reuses_emission_across_invocations(
    endpoint_name, monkeypatch
) -> None:
    """Native slot rows vary while one retained circuit serves repeated launches."""
    import qiskit
    from qiskit.providers.basic_provider import BasicSimulator

    from qamomile.qiskit import Aer, QiskitBackend
    from qamomile.qiskit._payload import QiskitExecutable

    endpoint = Aer() if endpoint_name == "aer" else QiskitBackend(BasicSimulator())
    artifact = qm.compile(
        qm.build_source(_SOURCE).specialize("sample"), to=endpoint.target
    )
    payload = artifact.execution_payload(QiskitExecutable)
    assert payload.shot_template is not None
    assert payload.hidden_parameters
    _forbid_compilation(monkeypatch)
    preparations = []
    transpile = qiskit.transpile

    def prepare(*args, **kwargs):
        preparations.append(1)
        return transpile(*args, **kwargs)

    monkeypatch.setattr(qiskit, "transpile", prepare)
    assert qm.execute(
        artifact, on=endpoint, parameters={"offset": 0.0}, shots=3
    ).counts == {
        (0, 0): 2,
        (1, 1): 1,
    }
    assert qm.execute(
        artifact, on=endpoint, parameters={"offset": 3.141592653589793}, shots=2
    ).counts == {(1, 0): 1, (0, 1): 1}
    assert len(preparations) == (1 if endpoint_name == "aer" else 0)


@pytest.mark.parametrize("quantum", [False, True])
def test_runtime_fixed_template_uses_one_parameter_row_pub(
    quantum: bool, monkeypatch
) -> None:
    """One shaped PUB preserves logical coordinates, including zero hidden slots."""
    from qiskit.primitives import StatevectorSampler
    from qiskit.providers.basic_provider import BasicSimulator

    import qamomile.qiskit.runtime as runtime
    from qamomile.qiskit import IBMRuntime

    submissions = []
    native_jobs = {}

    class Sampler:
        """Keep provider transport local while exercising real SDK row shapes."""

        def __init__(self, *, mode, options):
            self.mode = mode

        def backend(self):
            return self.mode

        def run(self, pubs, *, shots):
            submissions.append((pubs, shots))
            result = StatevectorSampler(seed=0).run(pubs, shots=shots).result()
            job = SimpleNamespace(
                job_id=lambda: "template-job",
                result=lambda: result,
                status=lambda: "DONE",
            )
            native_jobs["template-job"] = job
            return job

    monkeypatch.setattr(
        runtime, "_runtime_module", lambda: SimpleNamespace(SamplerV2=Sampler)
    )
    endpoint = IBMRuntime(
        BasicSimulator(),
        service=SimpleNamespace(job=lambda identity: native_jobs[identity]),
    )
    source = (
        _SOURCE
        if quantum
        else """
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bool]:
    qm.discard(qm.h(qm.prepare(False)))
    return qm.get_current_shot() == 0
"""
    )
    artifact = qm.compile(
        qm.build_source(source).specialize("sample"), to=endpoint.target
    )
    _forbid_compilation(monkeypatch)
    parameters = {"offset": 0.0} if quantum else {}
    job = qm.submit(artifact, on=endpoint, parameters=parameters, shots=3)
    assert len(submissions) == 1
    pubs, shots = submissions[0]
    assert shots == 1
    assert len(pubs) == 1
    assert pubs[0][1].shape == (3,)
    if not quantum:
        assert pubs[0][1].num_parameters == 0
    snapshot = qm.ExecutionSnapshot.from_bytes(job.snapshot().to_bytes())
    restored = endpoint.restore(snapshot, artifact=artifact, parameters=parameters)
    expected = {(0, 0): 2, (1, 1): 1} if quantum else {(1,): 1, (0,): 2}
    assert restored.result().counts == expected
    assert job.result().counts == expected
    assert len(submissions) == 1


def test_fixed_template_pure_context_needs_no_sdk_submission(monkeypatch) -> None:
    """Zero-qubit coordinates are evaluated entirely by the native owner."""
    from qiskit.providers.basic_provider import BasicSimulator

    from qamomile.qiskit import QiskitBackend

    backend = BasicSimulator()
    endpoint = QiskitBackend(backend)
    artifact = qm.compile(
        qm.build_source("""
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bool]:
    return qm.get_current_shot() == 0
""").specialize("sample"),
        to=endpoint.target,
    )
    _forbid_compilation(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("zero-qubit context must not submit an SDK job")

    monkeypatch.setattr(backend, "run", forbidden)
    assert qm.execute(artifact, on=endpoint, shots=3).counts == {(1,): 1, (0,): 2}


def test_fixed_template_checks_batch_limit_before_copying(monkeypatch) -> None:
    """Unsupported batch sizes fail before SDK circuit copies are allocated."""
    from qiskit.providers.basic_provider import BasicSimulator

    import qamomile.qiskit._shot_template as transport
    from qamomile.qiskit import QiskitBackend

    backend = BasicSimulator()
    endpoint = QiskitBackend(backend)
    artifact = qm.compile(
        qm.build_source(_SOURCE).specialize("sample"), to=endpoint.target
    )
    monkeypatch.setattr(type(backend), "max_circuits", property(lambda self: 2))

    def forbidden(*args, **kwargs):
        raise AssertionError("batch capacity must be checked before copying")

    monkeypatch.setattr(transport, "template_circuits", forbidden)
    with pytest.raises(qm.ExecutionError, match="backend limit is 2"):
        qm.execute(artifact, on=endpoint, parameters={"offset": 0.0}, shots=3)


def test_fixed_template_persistence_retains_native_plan(monkeypatch) -> None:
    """Portable export restores hidden SDK identities and the native result plan."""
    from qamomile.persistence import deserialize, serialize
    from qamomile.qiskit import Aer
    from qamomile.qiskit._payload import QiskitExecutable

    endpoint = Aer()
    original = qm.compile(
        qm.build_source(_SOURCE).specialize("sample"), to=endpoint.target
    )
    restored = deserialize(serialize(original))
    owner = restored.execution_payload(QiskitExecutable)
    assert owner.shot_template is not None
    assert owner.hidden_parameters
    _forbid_compilation(monkeypatch)
    assert qm.execute(
        restored, on=endpoint, parameters={"offset": 0.0}, shots=3
    ).counts == {(0, 0): 2, (1, 1): 1}


@pytest.mark.parametrize("parameter_table", [False, True])
def test_fixed_template_indexes_angle_tables_without_reemission(
    parameter_table: bool, monkeypatch
) -> None:
    """Literal and launch-bound table selection become native numeric slot rows."""
    from qamomile.qiskit import Aer
    from qamomile.qiskit._payload import QiskitExecutable

    signature = "angles: qm.Param[qm.Vector[qm.Real, 2]]" if parameter_table else ""
    local = "" if parameter_table else "    angles = [0.0, 3.141592653589793]\n"
    source = (
        "import qamomile as qm\n@qm.qkernel\n"
        f"def sample({signature}) -> qm.Shot[qm.Bit]:\n"
        f"{local}"
        "    angle = angles[qm.get_current_shot() % 2]\n"
        "    return qm.measure(qm.ry(qm.prepare(False), angle))\n"
    )
    endpoint = Aer()
    artifact = qm.compile(
        qm.build_source(source).specialize("sample"), to=endpoint.target
    )
    assert artifact.execution_payload(QiskitExecutable).shot_template is not None
    _forbid_compilation(monkeypatch)
    parameters = {"angles": (0.0, 3.141592653589793)} if parameter_table else {}
    assert qm.execute(artifact, on=endpoint, parameters=parameters, shots=3).counts == {
        (0,): 2,
        (1,): 1,
    }


def test_fixed_template_persistence_preserves_checked_unused_results(
    monkeypatch,
) -> None:
    """Unused checked arithmetic retains its valid runtime guard after restoration."""
    from qamomile.persistence import deserialize, serialize
    from qamomile.qiskit import Aer

    endpoint = Aer()
    original = qm.compile(
        qm.build_source("""
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bool]:
    index = qm.get_current_shot()
    ignored = 1.0 / float(index + 1)
    return index == 0
""").specialize("sample"),
        to=endpoint.target,
    )
    restored = deserialize(serialize(original))
    _forbid_compilation(monkeypatch)
    for artifact in (original, restored):
        assert qm.execute(artifact, on=endpoint, shots=2).counts == {(1,): 1, (0,): 1}


def test_fixed_template_rejects_unused_division_by_zero_before_submission(
    monkeypatch,
) -> None:
    """Native guards run even when a contextual arithmetic result is unused."""
    from qiskit.providers.basic_provider import BasicSimulator

    from qamomile.qiskit import QiskitBackend

    backend = BasicSimulator()
    endpoint = QiskitBackend(backend)
    artifact = qm.compile(
        qm.build_source("""
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bool]:
    index = qm.get_current_shot()
    ignored = 1.0 / float(index)
    qm.discard(qm.h(qm.prepare(False)))
    return index == 0
""").specialize("sample"),
        to=endpoint.target,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("native arithmetic guards must run before submission")

    monkeypatch.setattr(backend, "run", forbidden)
    with pytest.raises(qm.ExecutionError, match="zero"):
        qm.execute(artifact, on=endpoint, shots=2)


@pytest.mark.parametrize(
    ("success", "experiment_success", "shots"),
    [(False, False, 0), (True, False, 1), (True, True, 0), (True, True, 2)],
)
def test_fixed_template_rejects_failed_unmeasured_backend_job(
    success: bool, experiment_success: bool, shots: int, monkeypatch
) -> None:
    """A real failed SDK Result cannot become an empty successful logical sample."""
    from qiskit.providers.basic_provider import BasicSimulator
    from qiskit.result import Result

    from qamomile.qiskit import QiskitBackend

    backend = BasicSimulator()
    endpoint = QiskitBackend(backend)
    artifact = qm.compile(
        qm.build_source("""
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bool]:
    qm.discard(qm.h(qm.prepare(False)))
    return qm.get_current_shot() == 0
""").specialize("sample"),
        to=endpoint.target,
    )
    failed = Result.from_dict(
        {
            "backend_name": "local-failure-witness",
            "backend_version": "1",
            "job_id": "failed-job",
            "success": success,
            "results": [{"success": experiment_success, "shots": shots, "data": {}}],
        }
    )
    monkeypatch.setattr(
        backend,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            result=lambda timeout=None: failed,
            job_id=lambda: "failed-job",
            status=lambda: "DONE",
        ),
    )
    with pytest.raises(qm.ExecutionError, match="did not succeed|one shot"):
        qm.execute(artifact, on=endpoint, shots=1)


def test_fixed_template_preserves_context_in_quantum_helper(monkeypatch) -> None:
    """Context-dependent helper angles remain hidden bindings in one SDK circuit."""
    from qamomile.qiskit import Aer
    from qamomile.qiskit._payload import QiskitExecutable

    endpoint = Aer()
    artifact = qm.compile(
        qm.build_source("""
import qamomile as qm

@qm.qkernel
def rotate(state: qm.Q[qm.Bool]) -> qm.Q[qm.Bool]:
    angle = float(qm.get_current_shot() % 2) * 3.141592653589793
    return qm.ry(state, angle)

@qm.qkernel
def sample() -> qm.Shot[qm.Bit]:
    return qm.measure(rotate(qm.prepare(False)))
""").specialize("sample"),
        to=endpoint.target,
    )
    assert artifact.execution_payload(QiskitExecutable).shot_template is not None
    _forbid_compilation(monkeypatch)
    assert qm.execute(artifact, on=endpoint, shots=3).counts == {(0,): 2, (1,): 1}


def test_nested_loop_bindings_are_independent_across_invocations_and_shots() -> None:
    """Each shot receives its own rotation inside nested loops of a shared template."""
    from qiskit.quantum_info import Statevector
    from qiskit.transpiler.passes import UnrollForLoops

    from qamomile.qiskit._shot_template import template_circuits, template_inputs

    quarter_pi = math.pi / 4
    program = qm.build_source(f"""
import qamomile as qm

@qm.qkernel
def rotate(state: qm.Q[qm.Bool], offset: qm.Param[qm.Real]) -> qm.Q[qm.Bool]:
    angle = float(qm.get_current_shot()) * {quarter_pi}
    return qm.ry(qm.ry(state, angle), offset)

@qm.qkernel
def sample(offset: qm.Param[qm.Real]) -> qm.Shot[qm.Bit]:
    state = qm.prepare(False)
    for i in range(2):
        for j in range(2):
            state = rotate(state, offset)
    return qm.measure(state)
""").specialize("sample")
    artifact = qm.compile(program, to=qm.qiskit.Aer().target)
    template = qm.qiskit.load_circuit(artifact)
    [outer_loop] = template.get_instructions("for_loop")
    assert outer_loop.operation.blocks[0].count_ops()["for_loop"] == 1
    for offset, expected in [(0.0, [0, 1]), (quarter_pi, [1, 0])]:
        invocation = qm.Invocation.create(parameters={"offset": offset}, shots=2)
        prepared, hidden, rows = template_inputs(artifact, invocation, template)
        circuits = template_circuits(prepared, hidden, rows)
        probabilities = []
        for circuit in circuits:
            rotations = UnrollForLoops()(
                circuit.remove_final_measurements(inplace=False)
            )
            probabilities.append(
                Statevector.from_instruction(rotations).probabilities()[1]
            )
        assert probabilities == pytest.approx(expected)


def test_fixed_template_checks_only_taken_scalar_branches(monkeypatch) -> None:
    """An unused division in an untaken branch is never evaluated eagerly."""
    from qiskit_aer import AerSimulator

    from qamomile.qiskit import Aer
    from qamomile.qiskit._payload import QiskitExecutable

    endpoint = Aer()
    artifact = qm.compile(
        qm.build_source("""
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bool]:
    index = qm.get_current_shot()
    if index != 0:
        ignored = 1.0 / float(index)
    return index == 0
""").specialize("sample"),
        to=endpoint.target,
    )
    assert artifact.execution_payload(QiskitExecutable).shot_template is not None
    _forbid_compilation(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("scalar contextual branches must need no SDK submission")

    monkeypatch.setattr(AerSimulator, "run", forbidden)
    assert qm.execute(artifact, on=endpoint, shots=3).counts == {(1,): 1, (0,): 2}


@pytest.mark.parametrize(
    "rows",
    [
        [(1, [1], 1), (0, [0], 1)],
        [(0, [0], 1)],
        [(0, [0], 1), (0, [1], 1)],
    ],
    ids=["reordered", "missing", "duplicate"],
)
def test_native_template_rejects_invalid_logical_coordinates(rows) -> None:
    """Native postprocessing checks coordinate coverage before aggregating samples."""
    from qamomile.qiskit import Aer
    from qamomile.qiskit._payload import QiskitExecutable

    artifact = qm.compile(
        qm.build_source(_SOURCE).specialize("sample"), to=Aer().target
    )
    owner = artifact.execution_payload(QiskitExecutable).shot_template
    assert owner is not None
    with pytest.raises(ValueError, match="Q4452"):
        owner.postprocess(rows, {"offset": 0.0}, 2, artifact.result_abi._to_bytes())


@pytest.mark.parametrize("restore", [False, True])
def test_zero_slot_template_preparation_retains_only_cardinality(restore) -> None:
    """Real preparation retains ten million empty rows without Python row objects."""
    import tracemalloc

    from qamomile import _qamomile_native as native
    from qamomile.persistence import deserialize, serialize
    from qamomile.qiskit import Aer
    from qamomile.qiskit._shot_template import template_inputs

    artifact = qm.compile(
        qm.build_source("""
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bool]:
    index = qm.get_current_shot()
    qm.discard(qm.prepare(False))
    return True
""").specialize("sample"),
        to=Aer().target,
    )
    if restore:
        artifact = deserialize(serialize(artifact))
    invocation = qm.Invocation.create(shots=10_000_000, seed=0)
    tracemalloc.start()
    try:
        _, hidden, rows = template_inputs(artifact, invocation)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert not hidden
    assert isinstance(rows, native.NativeShotLaunchRows)
    assert len(rows) == 10_000_000
    assert rows.slot_count == 0
    assert peak < 1_000_000
    assert rows[0] == rows[-1] == rows[9_999_999] == []
    for index in (10_000_000, -10_000_001):
        with pytest.raises(IndexError, match="shot row index"):
            rows[index]
    with pytest.raises(TypeError):
        rows[:1]
    with pytest.raises(AttributeError):
        rows.slot_count = 1
    with pytest.raises(TypeError):
        native.NativeShotLaunchRows()


@pytest.mark.parametrize("restore", [False, True])
def test_zero_slot_template_preparation_checks_later_indices(restore) -> None:
    """Compact rows cannot bypass an unused arithmetic guard at a later index."""
    from qamomile.persistence import deserialize, serialize
    from qamomile.qiskit import Aer
    from qamomile.qiskit._payload import QiskitExecutable
    from qamomile.qiskit._shot_template import template_inputs

    artifact = qm.compile(
        qm.build_source("""
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bool]:
    index = qm.get_current_shot()
    ignored = 1.0 / float(1 - index)
    qm.discard(qm.h(qm.prepare(False)))
    return index == 0
""").specialize("sample"),
        to=Aer().target,
    )
    if restore:
        artifact = deserialize(serialize(artifact))
    owner = artifact.execution_payload(QiskitExecutable).shot_template
    assert owner is not None and owner.slot_count == 0
    _, _, rows = template_inputs(artifact, qm.Invocation.create(shots=1, seed=0))
    assert list(rows) == [[]]
    with pytest.raises(qm.ExecutionError, match="zero"):
        template_inputs(artifact, qm.Invocation.create(shots=2, seed=0))


def test_native_template_rows_preserve_slot_and_index_order() -> None:
    """Flat native cells expose independent ordered rows and negative indices."""
    from qamomile.qiskit import Aer
    from qamomile.qiskit._payload import QiskitExecutable

    artifact = qm.compile(
        qm.build_source("""
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bit]:
    index = qm.get_current_shot()
    state = qm.ry(qm.prepare(False), float(index))
    return qm.measure(qm.rx(state, float(index + 10)))
""").specialize("sample"),
        to=Aer().target,
    )
    owner = artifact.execution_payload(QiskitExecutable).shot_template
    assert owner is not None
    rows = owner.bind_rows({}, 3)
    assert rows.slot_count == 2
    assert list(rows) == [[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]]
    assert rows[-3] == rows[0]
    assert rows[-1] == rows[2]
    copy = rows[0]
    copy[0] = 99.0
    assert rows[0] == [0.0, 10.0]
