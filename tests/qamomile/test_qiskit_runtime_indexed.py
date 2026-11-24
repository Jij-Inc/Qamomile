"""Bound Runtime current-shot transport and persistent decoding metadata."""

import json
from hashlib import sha256
from types import SimpleNamespace

import pytest

import qamomile as qm


def _pending_runtime(monkeypatch):
    """Capture submissions and restores without executing quantum circuits."""
    from qiskit.providers.basic_provider import BasicSimulator

    import qamomile.qiskit.runtime as runtime
    from qamomile.qiskit import IBMRuntime
    from qamomile.qiskit._integration import _CAPABILITIES

    submissions = []
    restores = []
    native = SimpleNamespace(job_id=lambda: "pending-indexed", status=lambda: "QUEUED")

    class Sampler:
        """Retain one provider job without requesting simulator results."""

        def __init__(self, *, mode, options):
            self.mode = mode

        def backend(self):
            return self.mode

        def run(self, pubs, *, shots):
            submissions.append((pubs, shots))
            return native

    def restore(identity):
        """Return the original provider job and record its exact identity."""
        restores.append(identity)
        return native

    monkeypatch.setattr(
        runtime, "_runtime_module", lambda: SimpleNamespace(SamplerV2=Sampler)
    )
    endpoint = IBMRuntime(
        BasicSimulator(),
        service=SimpleNamespace(job=restore),
        requirements=_CAPABILITIES,
    )
    return endpoint, submissions, restores


def test_runtime_bit_map_digest_preserves_canonical_order() -> None:
    """Incremental hashing agrees with compact JSON, including empty registers."""
    from qamomile.qiskit.runtime import _classical_bit_maps_digest

    bit_maps = ((("résultat", 1), ("meas", 0)), (), (("meas", 2),))
    expected = sha256(json.dumps(bit_maps, separators=(",", ":")).encode()).hexdigest()
    assert _classical_bit_maps_digest(bit_maps) == expected
    assert _classical_bit_maps_digest(tuple(reversed(bit_maps))) != expected


def test_runtime_circuit_limit_precedes_child_sdk_copying(monkeypatch) -> None:
    """A declared single-job circuit limit fails before loading child copies."""
    import qamomile.qiskit._integration as integration

    endpoint, submissions, restores = _pending_runtime(monkeypatch)
    artifact = qm.compile(
        qm.build_source("""
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bit]:
    return qm.measure(qm.prepare(False))
""").specialize("sample"),
        to=endpoint.target,
    )
    monkeypatch.setattr(
        type(endpoint.backend), "max_circuits", property(lambda self: 1)
    )

    def forbidden(*args, **kwargs):
        """Keep over-capacity children entirely before SDK copying."""
        raise AssertionError("backend capacity must be checked before SDK copying")

    monkeypatch.setattr(integration, "load_circuit", forbidden)
    children = ((artifact, qm.Invocation.create(shots=1, seed=None)),) * 2
    with pytest.raises(qm.ExecutionError, match="backend limit is 1"):
        endpoint._launch_indexed(
            artifact, qm.Invocation.create(shots=2, seed=None), children
        )
    guard = endpoint._indexed_materialization_guard()
    guard.add(artifact)
    with pytest.raises(qm.ExecutionError, match="backend limit is 1"):
        guard.add(artifact)
    assert not submissions
    assert not restores


def test_runtime_indexed_snapshot_retains_constant_size_context(monkeypatch) -> None:
    """A large repeated transport stays restorable within native snapshot budgets."""
    import qamomile.compilation as compilation
    import qamomile.qiskit.runtime as runtime
    from qamomile.qiskit._integration import load_circuit

    endpoint, submissions, restores = _pending_runtime(monkeypatch)
    program = qm.build_source("""
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bit]:
    return qm.measure(qm.prepare(False))
""").specialize("sample")
    artifact = qm.compile(program, to=endpoint.target)
    circuit = load_circuit(artifact)
    count = 100_000
    children = ((artifact, qm.Invocation.create(shots=1, seed=None)),) * count
    circuits = (circuit,) * count
    monkeypatch.setattr(runtime, "indexed_circuits", lambda *args, **kwargs: circuits)
    monkeypatch.setattr(compilation, "_current_shot_children", lambda *args: children)
    job = endpoint._launch_indexed(
        artifact, qm.Invocation.create(shots=count, seed=None), children
    )
    assert len(submissions) == 1
    assert len(submissions[0][0]) == count
    assert submissions[0][1] == 1
    assert job.status is qm.JobStatus.SUBMITTED

    snapshot_bytes = job.snapshot().to_bytes()
    assert len(snapshot_bytes) < 1024
    snapshot = qm.ExecutionSnapshot.from_bytes(snapshot_bytes)
    reference = snapshot.references()[0]
    context = dict(reference.context_values)
    assert len(context["bits_digest"]) == 64
    assert "bits" not in context
    restored = endpoint.restore(snapshot, artifact=artifact)
    assert restored.snapshot().to_bytes() == snapshot_bytes
    assert restores == [job.job_id]
    assert len(submissions) == 1

    for digest in (None, "", "0" * 64, "A" * 64):
        altered = dict(context)
        if digest is None:
            altered.pop("bits_digest")
        else:
            altered["bits_digest"] = digest
        corrupted = qm.ExecutionReference.create(
            provider=reference.provider,
            job_ids=reference.job_ids,
            target=reference.target,
            context=altered,
        )
        with pytest.raises(ValueError, match="bit maps disagree"):
            endpoint.restore(corrupted, artifact=artifact)
    assert restores == [job.job_id]
    assert len(submissions) == 1


def test_runtime_restores_existing_indexed_references(monkeypatch) -> None:
    """Saved external jobs keep their original canonical decoding references."""
    import qamomile.compilation as compilation
    import qamomile.qiskit.runtime as runtime
    from qamomile.qiskit._integration import load_circuit

    endpoint, submissions, restores = _pending_runtime(monkeypatch)
    artifact = qm.compile(
        qm.build_source("""
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bit]:
    return qm.measure(qm.prepare(False))
""").specialize("sample"),
        to=endpoint.target,
    )
    circuit = load_circuit(artifact)
    children = ((artifact, qm.Invocation.create(shots=1, seed=None)),) * 2
    circuits = (circuit, circuit)
    monkeypatch.setattr(runtime, "indexed_circuits", lambda *args, **kwargs: circuits)
    monkeypatch.setattr(compilation, "_current_shot_children", lambda *args: children)
    job = endpoint._launch_indexed(
        artifact, qm.Invocation.create(shots=2, seed=None), children
    )
    reference = job.snapshot().references()[0]
    context = dict(reference.context_values)
    assert "bits_digest" in context
    assert "bits" not in context
    bits = json.dumps(
        tuple(runtime._classical_bit_map(item) for item in circuits),
        separators=(",", ":"),
    )
    legacy_context = {
        key: value for key, value in context.items() if key != "bits_digest"
    }
    legacy_context["bits"] = bits
    legacy_reference = qm.ExecutionReference.create(
        provider=reference.provider,
        job_ids=reference.job_ids,
        target=reference.target,
        context=legacy_context,
    )
    snapshot_bytes = qm.ExecutionSnapshot.remote(legacy_reference).to_bytes()
    snapshot = qm.ExecutionSnapshot.from_bytes(snapshot_bytes)
    restored = endpoint.restore(snapshot, artifact=artifact)
    assert restored.snapshot().to_bytes() == snapshot_bytes
    assert restores == [job.job_id]
    assert len(submissions) == 1

    for altered in (
        {**context, "bits": bits},
        {**legacy_context, "bits": "[]"},
        {**legacy_context, "bits": bits + " "},
        {key: value for key, value in legacy_context.items() if key != "bits"},
    ):
        corrupted = qm.ExecutionReference.create(
            provider=reference.provider,
            job_ids=reference.job_ids,
            target=reference.target,
            context=altered,
        )
        with pytest.raises(ValueError, match="bit maps disagree"):
            endpoint.restore(corrupted, artifact=artifact)
    assert restores == [job.job_id]
    assert len(submissions) == 1


def test_runtime_ten_million_shots_use_one_parameter_row_submission(
    monkeypatch,
) -> None:
    """Ten million checked logical rows need one PUB and no simulator execution."""
    from qiskit.primitives.containers.bindings_array import BindingsArray

    from qamomile.qiskit._payload import QiskitExecutable

    endpoint, submissions, restores = _pending_runtime(monkeypatch)
    program = qm.build_source("""
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bool]:
    index = qm.get_current_shot()
    state = qm.prepare(False)
    qm.discard(state)
    return True
""").specialize("sample")
    artifact = qm.compile(program, to=endpoint.target)
    payload = artifact.execution_payload(QiskitExecutable)
    assert payload.shot_template is not None
    payload.shot_template.check_invocation(10_000_000)
    with pytest.raises(ValueError, match="work budget"):
        payload.shot_template.check_invocation(10_000_001)
    circuit = payload.circuit()
    assert circuit.num_qubits == 1
    assert not circuit.parameters

    job = endpoint._launch_shot_template(
        artifact, qm.Invocation.create(shots=10_000_000, seed=None)
    )
    assert len(submissions) == 1
    pubs, shots = submissions[0]
    assert shots == 1
    assert len(pubs) == 1
    submitted_circuit, bindings = pubs[0]
    assert submitted_circuit == circuit
    assert isinstance(bindings, BindingsArray)
    assert bindings.shape == (10_000_000,)
    assert bindings.num_parameters == 0
    assert job.status is qm.JobStatus.SUBMITTED
    assert len(job.snapshot().to_bytes()) < 1024
    assert not restores
