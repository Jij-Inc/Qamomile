"""Verify bounded single-job indexed transport without million-shot simulation."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

import qamomile as qm

_ANGLES = """
import qamomile as qm

@qm.qkernel
def sample() -> tuple[qm.Shot[qm.Bit], qm.Shot[qm.Bool]]:
    index = qm.get_current_shot()
    angle = float(index % 2) * 3.141592653589793
    measured = qm.measure(qm.ry(qm.prepare(False), angle))
    return measured, index == 0
"""

_MEMORY = """
import qamomile as qm

@qm.qkernel
def sample() -> tuple[qm.Shot[qm.Bit], qm.Shot[qm.Bool]]:
    measured = qm.measure(qm.h(qm.prepare(False)))
    return measured, qm.get_current_shot() == 0
"""


def _compile(source, endpoint):
    """Compile one small indexed source for the selected endpoint."""
    return qm.compile(qm.build_source(source).specialize("sample"), to=endpoint.target)


@pytest.mark.parametrize("route", ["parameters", "memory", "removed_parameters"])
def test_aer_template_submits_one_circuit_without_per_shot_copies(
    route, monkeypatch
) -> None:
    """Retained and removed hidden slots both preserve one ordered SDK job."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    import qamomile.qiskit._shot_template as transport
    from qamomile.qiskit import Aer
    from qamomile.qiskit._payload import QiskitExecutable

    endpoint = Aer()
    source = _MEMORY if route == "memory" else _ANGLES
    if route == "removed_parameters":
        source = _ANGLES.replace("qm.ry(", "qm.rz(")
    artifact = _compile(source, endpoint)
    owner = artifact.execution_payload(QiskitExecutable)
    if route == "removed_parameters":
        assert owner.hidden_parameters
        prepared = QuantumCircuit(1, 1)
        prepared.measure(0, 0)
        monkeypatch.setattr(owner, "aer_circuit", lambda *args: prepared.copy())

    def forbidden(*args, **kwargs):
        raise AssertionError("Aer must retain one circuit instead of copying each row")

    monkeypatch.setattr(transport, "template_circuits", forbidden)
    run = AerSimulator.run
    submissions = []

    def record(backend, circuit, **options):
        assert isinstance(circuit, QuantumCircuit)
        assert backend.options.max_job_size is None
        assert backend.options.max_shot_size is None
        submissions.append(options)
        return run(backend, circuit, **options)

    monkeypatch.setattr(AerSimulator, "run", record)
    result = qm.execute(artifact, on=endpoint, shots=3, seed=17)
    assert sum(result.counts.values()) == 3
    assert len(submissions) == 1
    options = submissions[0]
    if route == "parameters":
        assert options["shots"] == 1
        assert options["runtime_parameter_bind_enable"] is True
        assert len(options["parameter_binds"]) == 1
        assert list(options["parameter_binds"][0].values()) == [
            [0.0, 3.141592653589793, 0.0]
        ]
        assert result.counts == {(0, 1): 1, (1, 0): 1, (0, 0): 1}
    else:
        assert options["shots"] == 3
        assert options["memory"] is True
        assert "parameter_binds" not in options
        if route == "removed_parameters":
            assert result.counts == {(0, 1): 1, (0, 0): 2}


def test_aer_parameter_free_memory_preserves_logical_order(monkeypatch) -> None:
    """Ordered readouts retain correlations that a histogram cannot recover."""
    from qiskit.result import Result
    from qiskit_aer import AerSimulator

    from qamomile.qiskit import Aer

    endpoint = Aer()
    artifact = _compile(_MEMORY, endpoint)
    native = Result.from_dict(
        {
            "backend_name": "ordered-witness",
            "backend_version": "1",
            "job_id": "one-job",
            "success": True,
            "results": [
                {
                    "success": True,
                    "shots": 3,
                    "header": {"memory_slots": 1, "creg_sizes": [["c", 1]]},
                    "data": {"memory": ["0x1", "0x0", "0x0"]},
                }
            ],
        }
    )

    def no_histogram(*args, **kwargs):
        raise AssertionError("logical coordinates must not come from a histogram")

    monkeypatch.setattr(native, "get_counts", no_histogram)
    monkeypatch.setattr(
        AerSimulator,
        "run",
        lambda *args, **kwargs: SimpleNamespace(result=lambda: native),
    )
    assert qm.execute(artifact, on=endpoint, shots=3).counts == {
        (1, 1): 1,
        (0, 0): 2,
    }


@pytest.mark.parametrize("source", [_ANGLES, _MEMORY])
def test_aer_template_seed_artifact_and_snapshot_round_trip(source) -> None:
    """Both one-circuit transports preserve seeded reuse and persisted results."""
    from qamomile.persistence import deserialize, serialize
    from qamomile.qiskit import Aer

    endpoint = Aer()
    artifact = deserialize(serialize(_compile(source, endpoint)))
    job = qm.submit(artifact, on=endpoint, shots=16, seed=41)
    result = job.result()
    assert qm.execute(artifact, on=endpoint, shots=16, seed=41).counts == result.counts
    restored = qm.ExecutionSnapshot.from_bytes(job.snapshot().to_bytes()).result
    assert restored == result
    assert restored.result_abi == artifact.result_abi


def test_aer_parameter_rows_do_not_repeat_a_single_random_sample() -> None:
    """Repeated parameter values retain distinct seeded draws within each group."""
    from qamomile.qiskit import Aer

    source = _ANGLES.replace(
        "angle = float(index % 2) * 3.141592653589793",
        "angle = 1.5707963267948966 + float(index % 2) * 3.141592653589793",
    ).replace("return measured, index == 0", "return measured, index % 2 == 1")
    endpoint = Aer()
    artifact = _compile(source, endpoint)
    first = qm.execute(artifact, on=endpoint, shots=64, seed=41)
    repeated = qm.execute(artifact, on=endpoint, shots=64, seed=41)
    assert first.counts == repeated.counts
    assert all(
        first.counts.get((measurement, group), 0) > 0
        for measurement in (0, 1)
        for group in (0, 1)
    )
    assert sum(first.counts.values()) == 64


@pytest.mark.parametrize("parameterized", [False, True])
def test_aer_template_unmeasured_quantum_work_still_runs(parameterized, monkeypatch):
    """Missing measurements never suppress the real quantum SDK execution."""
    from qiskit_aer import AerSimulator

    from qamomile.qiskit import Aer

    source = _ANGLES.replace(
        "measured = qm.measure(qm.ry(qm.prepare(False), angle))",
        "qm.discard(qm.ry(qm.prepare(False), angle))\n    measured = False",
    )
    if not parameterized:
        source = _MEMORY.replace(
            "measured = qm.measure(qm.h(qm.prepare(False)))",
            "qm.discard(qm.h(qm.prepare(False)))\n    measured = False",
        )
    source = source.replace("qm.Shot[qm.Bit]", "qm.Shot[qm.Bool]")
    endpoint = Aer()
    artifact = _compile(source, endpoint)
    submissions = []
    run = AerSimulator.run

    def record(backend, circuit, **options):
        submissions.append(options)
        return run(backend, circuit, **options)

    monkeypatch.setattr(AerSimulator, "run", record)
    assert qm.execute(artifact, on=endpoint, shots=3).counts == {
        (0, 1): 1,
        (0, 0): 2,
    }
    assert len(submissions) == 1


@pytest.mark.parametrize("parameterized", [False, True])
@pytest.mark.parametrize(
    "invalid", ["failed", "failed_row", "missing", "extra", "shots"]
)
def test_aer_template_checks_result_coverage_before_empty_data(
    parameterized, invalid
) -> None:
    """Even unmeasured results require complete successful SDK execution."""
    from qiskit import QuantumCircuit

    from qamomile.qiskit._shot_template import aer_template_counts

    rows = [
        SimpleNamespace(success=True, shots=1 if parameterized else 3)
        for _ in range(3 if parameterized else 1)
    ]
    result = SimpleNamespace(success=True, results=rows)
    if invalid == "failed":
        result.success = False
    elif invalid == "failed_row":
        rows[0].success = False
    elif invalid == "missing":
        rows.pop()
    elif invalid == "extra":
        rows.append(rows[0])
    else:
        rows[0].shots += 1
    with pytest.raises(qm.ExecutionError, match="Q8233"):
        aer_template_counts(result, QuantumCircuit(1), 3, parameterized=parameterized)


@pytest.mark.parametrize("memory", [None, "000", ["0"], ["0"] * 4, [[0]] * 3])
def test_aer_template_rejects_invalid_ordered_memory(memory) -> None:
    """A successful histogram is insufficient when ordered samples are absent."""
    from qiskit import QuantumCircuit

    from qamomile.qiskit._shot_template import aer_template_counts

    result = SimpleNamespace(
        success=True,
        results=[SimpleNamespace(success=True, shots=3)],
        get_memory=lambda index: memory,
    )
    with pytest.raises(qm.ExecutionError, match="Q8233"):
        aer_template_counts(result, QuantumCircuit(1, 1), 3, parameterized=False)


@pytest.mark.parametrize("sample", ["01", "x", ""])
def test_aer_template_rejects_wrong_memory_width(sample, monkeypatch) -> None:
    """Ordered memory must still match the prepared SDK classical-bit width."""
    from qiskit_aer import AerSimulator

    from qamomile.qiskit import Aer

    endpoint = Aer()
    artifact = _compile(_MEMORY, endpoint)
    native = SimpleNamespace(
        success=True,
        results=[SimpleNamespace(success=True, shots=3)],
        get_memory=lambda index: [sample] * 3,
    )
    monkeypatch.setattr(
        AerSimulator,
        "run",
        lambda *args, **kwargs: SimpleNamespace(result=lambda: native),
    )
    with pytest.raises(qm.ExecutionError, match="Q8233.*width"):
        qm.execute(artifact, on=endpoint, shots=3)


def test_aer_template_checks_single_circuit_capacity(monkeypatch) -> None:
    """Parameter rows do not consume a backend's separate circuit limit."""
    from qiskit_aer import AerSimulator

    from qamomile.qiskit import Aer

    endpoint = Aer()
    artifact = _compile(_ANGLES, endpoint)
    monkeypatch.setattr(AerSimulator, "max_circuits", property(lambda self: 1))
    assert sum(qm.execute(artifact, on=endpoint, shots=3).counts.values()) == 3


def test_aer_parameter_free_template_checks_shot_capacity(monkeypatch) -> None:
    """An unsupported shared-circuit shot request fails before SDK submission."""
    from qiskit_aer import AerSimulator

    from qamomile.qiskit import Aer

    endpoint = Aer()
    artifact = _compile(_MEMORY, endpoint)
    configuration = AerSimulator.configuration

    def limited(backend):
        config = configuration(backend)
        config.max_shots = 2
        return config

    def forbidden(*args, **kwargs):
        raise AssertionError("shot capacity must fail before submission")

    monkeypatch.setattr(AerSimulator, "configuration", limited)
    monkeypatch.setattr(AerSimulator, "run", forbidden)
    with pytest.raises(qm.ExecutionError, match="Q8234.*backend limit is 2"):
        qm.execute(artifact, on=endpoint, shots=3)


@pytest.mark.parametrize("phase", ["prepare", "submit", "result"])
def test_aer_template_normalizes_sdk_failures(phase, monkeypatch) -> None:
    """Preparation, submission, and worker failures retain their SDK cause."""
    from qiskit_aer import AerSimulator

    from qamomile.qiskit import Aer
    from qamomile.qiskit._payload import QiskitExecutable

    endpoint = Aer()
    artifact = _compile(_MEMORY, endpoint)
    failure = ValueError("SDK failure witness")

    def fail(*args, **kwargs):
        raise failure

    if phase == "prepare":
        monkeypatch.setattr(
            artifact.execution_payload(QiskitExecutable), "aer_circuit", fail
        )
    else:
        monkeypatch.setattr(
            AerSimulator,
            "run",
            fail
            if phase == "submit"
            else lambda *args, **kwargs: SimpleNamespace(result=fail),
        )
    with pytest.raises(qm.ExecutionError, match="Q8233") as raised:
        qm.execute(artifact, on=endpoint, shots=3)
    assert raised.value.__cause__ is failure


def test_generic_template_preparation_budget_precedes_copies(monkeypatch) -> None:
    """An oversized SDK batch fails before copying circuits or submitting a job."""
    from qiskit.providers.basic_provider import BasicSimulator

    import qamomile.qiskit._indexed as indexed
    import qamomile.qiskit._shot_template as transport
    from qamomile.qiskit import QiskitBackend

    endpoint = QiskitBackend(BasicSimulator())
    artifact = _compile(_ANGLES, endpoint)
    monkeypatch.setattr(indexed, "_MAX_INDEXED_PREPARATION_WORK", 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("preparation must fail before copying or submission")

    monkeypatch.setattr(transport, "template_circuits", forbidden)
    monkeypatch.setattr(endpoint.backend, "run", forbidden)
    with pytest.raises(qm.ExecutionError, match="Q8234.*work budget"):
        qm.execute(artifact, on=endpoint, shots=3)


def test_indexed_guard_checks_million_copy_shape_without_qpy_or_copies(
    monkeypatch,
) -> None:
    """Shape checks accept tiny million-row transports without allocating them."""
    from qamomile.qiskit import Aer
    from qamomile.qiskit._indexed import IndexedMaterializationGuard
    from qamomile.qiskit._payload import QiskitExecutable

    artifact = _compile(_MEMORY, Aer())
    owner = artifact.execution_payload(QiskitExecutable)

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must inspect retained SDK facts directly")

    monkeypatch.setattr(owner, "to_bytes", forbidden)
    monkeypatch.setattr(owner, "circuit", forbidden)
    IndexedMaterializationGuard().add(artifact, copies=1_000_000)


def test_indexed_guard_bounds_metadata_before_retaining_next_child(monkeypatch):
    """Actual ABI/metadata bytes are charged per artifact, not per circuit copy."""
    import qamomile.qiskit._indexed as indexed
    from qamomile.ir import FrozenMapping
    from qamomile.qiskit import Aer
    from qamomile.qiskit._indexed import IndexedMaterializationGuard

    artifact = _compile(_MEMORY, Aer())
    artifact = replace(artifact, metadata=FrozenMapping.from_mapping({"x": b"a" * 100}))
    monkeypatch.setattr(indexed, "_MAX_INDEXED_METADATA_BYTES", 250)
    guard = IndexedMaterializationGuard()
    guard.add(artifact, copies=1_000)
    guard.add(artifact)
    with pytest.raises(qm.ExecutionError, match="Q8234.*metadata content"):
        guard.add(replace(artifact))
