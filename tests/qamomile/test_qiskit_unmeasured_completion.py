"""Exercise compact completion through real jobs and native result evaluation."""

import tracemalloc
from types import SimpleNamespace

import pytest

import qamomile as qm

_SOURCE = """
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bool]:
    index = qm.get_current_shot()
    qm.discard(qm.prepare(False))
    return True
"""


def _completion_peak(action, shots):
    """Require actual native completion without one retained Python object per shot."""
    tracemalloc.start()
    try:
        result = action()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.counts == {(1,): shots}
    assert peak < 1_000_000


@pytest.mark.parametrize("restore", [False, True])
def test_runtime_unmeasured_completion_keeps_cardinality(restore, monkeypatch):
    """Real preparation, job completion and restored jobs keep empty rows compact."""
    from qiskit.providers.basic_provider import BasicSimulator

    import qamomile.qiskit.runtime as runtime
    from qamomile.persistence import deserialize, serialize
    from qamomile.qiskit import IBMRuntime

    shots = 100_000
    native_jobs = {}

    class Sampler:
        """Certify the submitted PUB shape without simulating quantum states."""

        def __init__(self, *, mode, options):
            self.mode = mode

        def backend(self):
            return self.mode

        def run(self, pubs, *, shots):
            assert len(pubs) == 1 and shots == 1
            assert pubs[0][1].num_parameters == 0
            pub = SimpleNamespace(
                data=SimpleNamespace(shape=pubs[0][1].shape), metadata={"shots": 1}
            )
            job = SimpleNamespace(
                job_id=lambda: "compact-completion",
                result=lambda: [pub],
                status=lambda: "DONE",
            )
            native_jobs["compact-completion"] = job
            return job

    monkeypatch.setattr(
        runtime, "_runtime_module", lambda: SimpleNamespace(SamplerV2=Sampler)
    )
    endpoint = IBMRuntime(
        BasicSimulator(), service=SimpleNamespace(job=lambda key: native_jobs[key])
    )
    artifact = qm.compile(
        qm.build_source(_SOURCE).specialize("sample"), to=endpoint.target
    )
    if restore:
        artifact = deserialize(serialize(artifact))
    tracemalloc.start()
    try:
        job = qm.submit(artifact, on=endpoint, shots=shots)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1_000_000
    snapshot = qm.ExecutionSnapshot.from_bytes(job.snapshot().to_bytes())
    restored_job = endpoint.restore(snapshot, artifact=artifact)
    _completion_peak(job.result, shots)
    _completion_peak(restored_job.result, shots)


def test_aer_unmeasured_completion_keeps_cardinality(monkeypatch):
    """Aer validates one provider experiment and forwards no empty measurement rows."""
    from qiskit_aer import AerSimulator

    from qamomile.qiskit import Aer
    from qamomile.qiskit._payload import QiskitExecutable

    endpoint = Aer()
    artifact = qm.compile(
        qm.build_source(_SOURCE).specialize("sample"), to=endpoint.target
    )
    # Warm the real SDK preparation cache outside the completion allocation check.
    artifact.execution_payload(QiskitExecutable).aer_circuit(
        AerSimulator(),
        endpoint.method,
        endpoint.optimization_level,
        endpoint.seed_transpiler,
    )
    shots = 100_000

    def run(self, circuit, **options):
        assert not circuit.clbits and options["shots"] == shots
        result = SimpleNamespace(
            success=True, results=[SimpleNamespace(success=True, shots=shots)]
        )
        return SimpleNamespace(result=lambda: result)

    monkeypatch.setattr(AerSimulator, "run", run)
    _completion_peak(lambda: qm.execute(artifact, on=endpoint, shots=shots), shots)


def test_backend_unmeasured_completion_keeps_cardinality(monkeypatch):
    """Ordinary backend completion does not duplicate already submitted circuits."""
    from qiskit.providers.basic_provider import BasicSimulator

    from qamomile.qiskit import QiskitBackend

    backend = BasicSimulator()
    endpoint = QiskitBackend(backend)
    artifact = qm.compile(
        qm.build_source(_SOURCE).specialize("sample"), to=endpoint.target
    )
    shots = 4096

    def run(circuits, *, shots):
        assert shots == 1
        result = SimpleNamespace(
            success=True,
            results=tuple(SimpleNamespace(success=True, shots=1) for _ in circuits),
        )
        return SimpleNamespace(
            result=lambda timeout=None: result,
            job_id=lambda: "compact-backend",
            status=lambda: "DONE",
        )

    monkeypatch.setattr(backend, "run", run)
    job = qm.submit(artifact, on=endpoint, shots=shots)
    _completion_peak(job.result, shots)


@pytest.mark.parametrize("endpoint_name", ["aer", "backend", "runtime"])
@pytest.mark.parametrize("restore", [False, True])
def test_result_only_completion_preserves_indices(endpoint_name, restore, monkeypatch):
    """Every result-only endpoint uses compact coverage and native index semantics."""
    from qiskit.providers.basic_provider import BasicSimulator
    from qiskit_aer import AerSimulator

    from qamomile.persistence import deserialize, serialize
    from qamomile.qiskit import Aer, IBMRuntime, QiskitBackend

    backend = BasicSimulator()
    endpoint = {
        "aer": lambda: Aer(),
        "backend": lambda: QiskitBackend(backend),
        "runtime": lambda: IBMRuntime(backend),
    }[endpoint_name]()
    source = _SOURCE.replace("    qm.discard(qm.prepare(False))\n", "").replace(
        "    return True", "    return index % 3 == 1"
    )
    artifact = qm.compile(
        qm.build_source(source).specialize("sample"), to=endpoint.target
    )
    if restore:
        artifact = deserialize(serialize(artifact))

    def forbidden(*args, **kwargs):
        raise AssertionError("result-only completion must not submit to a provider")

    monkeypatch.setattr(backend, "run", forbidden)
    monkeypatch.setattr(AerSimulator, "run", forbidden)
    assert qm.execute(artifact, on=endpoint, shots=7).counts == {(0,): 5, (1,): 2}
    assert qm.execute(artifact, on=endpoint, shots=2).counts == {(0,): 1, (1,): 1}


@pytest.mark.parametrize("restore", [False, True])
def test_compact_native_completion_rejects_missing_measurements(restore):
    """An SDK zero-width result cannot hide measurements required by the native plan."""
    from qamomile.persistence import deserialize, serialize
    from qamomile.qiskit import Aer
    from qamomile.qiskit._shot_template import MeasurementFreeShots, template_result

    source = _SOURCE.replace(
        "    qm.discard(qm.prepare(False))",
        "    measured = qm.measure(qm.prepare(False))",
    ).replace("    return True", "    return measured and index == 0")
    artifact = qm.compile(qm.build_source(source).specialize("sample"), to=Aer().target)
    if restore:
        artifact = deserialize(serialize(artifact))
    with pytest.raises(qm.ExecutionError, match="Q4452.*measurement-free"):
        template_result(
            artifact,
            qm.Invocation.create(shots=2, seed=0),
            MeasurementFreeShots(2),
            0,
            backend="missing-measurements",
        )


@pytest.mark.parametrize("restore", [False, True])
def test_compact_native_completion_evaluates_late_runtime_checks(restore):
    """Compact input keeps unused numeric guards at their original logical index."""
    from qamomile.persistence import deserialize, serialize
    from qamomile.qiskit import Aer
    from qamomile.qiskit._payload import QiskitExecutable

    source = _SOURCE.replace(
        "    return True", "    unused = 1.0 / float(2 - index)\n    return index == 0"
    )
    artifact = qm.compile(qm.build_source(source).specialize("sample"), to=Aer().target)
    if restore:
        artifact = deserialize(serialize(artifact))
    owner = artifact.execution_payload(QiskitExecutable).shot_template
    assert owner is not None
    assert dict(
        owner.postprocess_unmeasured({}, 2, artifact.result_abi._to_bytes())
    ) == {
        b"\x00": 1,
        b"\x01": 1,
    }
    with pytest.raises(ValueError, match="zero"):
        owner.postprocess_unmeasured({}, 3, artifact.result_abi._to_bytes())


@pytest.mark.parametrize(
    ("shape", "metadata", "pub_count"),
    [
        ((2,), {"shots": 1}, 0),
        ((2,), {"shots": 1}, 2),
        ((), {"shots": 1}, 1),
        (None, {"shots": 1}, 1),
        ((1,), {"shots": 1}, 1),
        ((2,), {}, 1),
        ((2,), None, 1),
        ((2,), [("shots", 1)], 1),
        ((2,), {"shots": True}, 1),
        ((2,), {"shots": 0}, 1),
        ((2,), {"shots": 2}, 1),
    ],
)
def test_compact_runtime_completion_requires_provider_coverage(
    shape, metadata, pub_count
):
    """A compact carrier requires complete PUB shape and exact integer shot metadata."""
    from qamomile.qiskit.runtime import _runtime_template_counts

    pub = SimpleNamespace(data=SimpleNamespace(shape=shape), metadata=metadata)
    with pytest.raises((ValueError, qm.ExecutionError), match="count|one shot"):
        _runtime_template_counts([pub] * pub_count, (), 2)


@pytest.mark.parametrize("parameterized", [False, True])
@pytest.mark.parametrize(
    ("success", "row_success", "row_shots", "row_count"),
    [(False, True, 1, 1), (True, False, 1, 1), (True, True, 0, 1), (True, True, 1, 0)],
)
def test_compact_aer_completion_requires_provider_coverage(
    parameterized, success, row_success, row_shots, row_count
):
    """An empty classical register cannot turn failed or missing experiments into shots."""
    from qamomile.qiskit._shot_template import aer_template_counts

    result = SimpleNamespace(
        success=success,
        results=[SimpleNamespace(success=row_success, shots=row_shots)] * row_count,
    )
    with pytest.raises(qm.ExecutionError, match="succeed|count|shot"):
        aer_template_counts(
            result, SimpleNamespace(clbits=[]), 1, parameterized=parameterized
        )
