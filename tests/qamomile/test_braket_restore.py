"""Restore saved Braket tasks through SDK stand-ins without AWS requests."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

import qamomile as qm
from qamomile.braket import BraketDevice

pytestmark = pytest.mark.braket
pytest.importorskip("braket")

SOURCE = """
import qamomile as qm
@qm.qkernel
def sample(flip: qm.Param[qm.Bool]) -> tuple[qm.Shot[qm.Bits[3]], qm.Shot[qm.Bool]]:
    bits = qm.measure(qm.x(qm.qubits(3), target=0))
    return bits, bool(bits[0]) != flip
"""


class Task:
    """Completed AWS task exposing result decoding and cancellation."""

    id = "task-existing"
    cancelled = False

    def state(self):
        return "COMPLETED"

    def result(self):
        return SimpleNamespace(measurement_counts={"100": 4})

    def cancel(self):
        self.cancelled = True


def create_saved_task(monkeypatch):
    """Capture a submitted task and intercept every SDK restore call."""
    task = Task()
    launches = []
    restores = []

    def run(program, **options):
        launches.append((program, options))
        return task

    device = SimpleNamespace(
        name="stand-in",
        arn="device-existing",
        properties="fixed",
        aws_session=object(),
        run=run,
    )
    endpoint = BraketDevice(device, poll_timeout_seconds=30, poll_interval_seconds=0.01)
    artifact = qm.compile(
        qm.build_source(SOURCE).specialize("sample"), to=endpoint.target
    )
    job = qm.submit(artifact, on=endpoint, parameters={"flip": False}, shots=4)
    snapshot = qm.ExecutionSnapshot.from_bytes(job.snapshot().to_bytes())

    def restore(identity, **options):
        restores.append((identity, options))
        return task

    import braket.aws

    monkeypatch.setattr(braket.aws, "AwsQuantumTask", restore)
    return SimpleNamespace(
        task=task,
        endpoint=endpoint,
        artifact=artifact,
        snapshot=snapshot,
        launches=launches,
        restores=restores,
    )


@pytest.fixture
def saved_task(monkeypatch):
    """Reuse the local SDK stand-in for checked task restoration."""
    return create_saved_task(monkeypatch)


def test_restore_saved_reference_preserves_typed_result_and_existing_task(saved_task):
    """Roundtripped references reuse the provider ID and original result decoding."""
    s = saved_task
    restored = s.endpoint.restore(
        s.snapshot, artifact=s.artifact, parameters={"flip": False}
    )
    assert restored.job_id == s.task.id
    assert restored.status is qm.JobStatus.COMPLETED
    assert restored.result().values == {((1, 0, 0), True): 4}
    assert len(s.launches) == 1
    assert s.restores == [
        (
            s.task.id,
            {
                "aws_session": s.endpoint.device.aws_session,
                "poll_timeout_seconds": 30,
                "poll_interval_seconds": 0.01,
            },
        )
    ]
    assert restored.snapshot() == s.snapshot
    restored.cancel()
    assert s.task.cancelled
    direct = s.endpoint.restore(
        s.snapshot.reference, artifact=s.artifact, parameters={"flip": False}
    )
    assert direct.result().values == restored.result().values


def test_restore_completed_local_result_avoids_sdk(saved_task):
    """Completed typed snapshots can be restored without a provider lookup."""
    s = saved_task
    job = s.endpoint.restore(
        s.snapshot, artifact=s.artifact, parameters={"flip": False}
    )
    cached = qm.ExecutionSnapshot.completed(job.result())
    calls = len(s.restores)
    assert (
        s.endpoint.restore(cached, artifact=s.artifact).result().values
        == job.result().values
    )
    assert len(s.restores) == calls


@pytest.mark.parametrize(
    "field,value",
    [("provider", "other"), ("target", "other"), ("job_ids", ("first", "second"))],
)
def test_restore_rejects_wrong_provider_or_target_before_sdk(saved_task, field, value):
    """Invalid routing cannot trigger an SDK request."""
    s = saved_task
    reference = replace(s.snapshot.reference, **{field: value})
    with pytest.raises(ValueError, match="sampling task"):
        s.endpoint.restore(reference, artifact=s.artifact, parameters={"flip": False})
    assert not s.restores


@pytest.mark.parametrize(
    "key,value",
    [
        ("kind", "expectation"),
        ("compilation_identity", "wrong"),
        ("parameter_identity", "wrong"),
        ("shots", "0"),
        ("shots", "invalid"),
    ],
)
def test_restore_rejects_changed_decoding_context_before_sdk(saved_task, key, value):
    """Task context is checked before provider access or result decoding."""
    s = saved_task
    context = dict(s.snapshot.reference.context)
    context[key] = value
    reference = replace(s.snapshot.reference, context=tuple(sorted(context.items())))
    with pytest.raises(ValueError):
        s.endpoint.restore(reference, artifact=s.artifact, parameters={"flip": False})
    assert not s.restores


def test_restore_rejects_changed_parameters_and_artifact(saved_task):
    """Runtime postprocessing cannot silently use different launch arguments."""
    s = saved_task
    with pytest.raises(ValueError, match="parameters do not match"):
        s.endpoint.restore(s.snapshot, artifact=s.artifact, parameters={"flip": True})
    with pytest.raises(qm.ExecutionError):
        s.endpoint.restore(s.snapshot, artifact=s.artifact)
    other = qm.compile(
        qm.build_source(SOURCE.replace("target=0", "target=1")).specialize("sample"),
        to=s.endpoint.target,
    )
    with pytest.raises(ValueError, match="artifact does not match"):
        s.endpoint.restore(s.snapshot, artifact=other, parameters={"flip": False})
    assert not s.restores


def test_restore_rejects_legacy_reference_without_artifact_identity(saved_task):
    """Old task-ID-only references lack enough context for checked typed decoding."""
    s = saved_task
    reference = replace(s.snapshot.reference, context=(("kind", "sample"),))
    with pytest.raises(ValueError, match="artifact does not match"):
        s.endpoint.restore(reference, artifact=s.artifact, parameters={"flip": False})
    assert not s.restores


def test_restored_task_rejects_incorrect_shot_total(saved_task):
    """Provider results cannot pass as the originally requested shot count."""
    s = saved_task
    context = dict(s.snapshot.reference.context)
    context["shots"] = "5"
    reference = replace(s.snapshot.reference, context=tuple(sorted(context.items())))
    restored = s.endpoint.restore(
        reference, artifact=s.artifact, parameters={"flip": False}
    )
    with pytest.raises(qm.ExecutionError, match="returned 4 shots; expected 5"):
        restored.result()
