"""Tests for canonical local and remote execution snapshots."""

from __future__ import annotations

from dataclasses import replace

import pytest

import qamomile as qm
import qamomile.compilation as compilation_module
from qamomile.compilation import CompletedJob, ResultABI


def test_remote_snapshot_round_trips_deterministically() -> None:
    """Remote identifiers and secret-free context survive canonical storage."""

    reference = qm.ExecutionReference.create(
        provider="provider",
        job_ids=("job-1", "job-2"),
        target="device",
        group_id="batch",
        context={"shots": "16", "width": "3"},
    )
    snapshot = qm.ExecutionSnapshot.remote(reference)

    payload = snapshot.to_bytes()
    restored = qm.ExecutionSnapshot.from_bytes(payload)

    assert restored == snapshot
    assert restored.to_bytes() == payload
    assert restored.references() == (reference,)
    assert "credential" not in reference.context_values


def test_composite_snapshot_restores_a_typed_completed_result() -> None:
    """Composite trees retain remote order and the sealed ResultABI."""

    result_abi = ResultABI(qm.Shot[qm.UInt[3]], (2, 0, 1), 3)
    result = replace(
        qm.SampleResult.from_counts(
            backend="reference",
            counts={(1, 0, 1): 3, (0, 1, 0): 1},
        ),
        result_abi=result_abi,
    )
    reference = qm.ExecutionReference.create(
        provider="provider",
        job_ids=("pending",),
    )
    snapshot = qm.ExecutionSnapshot.composite(
        (
            qm.ExecutionSnapshot.remote(reference),
            qm.ExecutionSnapshot.completed(result),
        )
    )

    restored = qm.ExecutionSnapshot.from_bytes(snapshot.to_bytes())
    completed = restored.children[1].result

    assert restored.references() == (reference,)
    assert isinstance(completed, qm.SampleResult)
    assert completed.counts == result.counts
    assert completed.result_abi == result_abi
    assert completed.most_likely_value == result.most_likely_value == 5


def test_completed_job_snapshot_uses_the_same_portable_result_contract() -> None:
    """Local and remote jobs expose one snapshot-shaped persistence UX."""

    result = qm.ValueResult("reference", {"answer": (4, 2)})
    job = CompletedJob("completed", result)

    assert job.status is qm.JobStatus.COMPLETED
    assert qm.ExecutionSnapshot.from_bytes(job.snapshot().to_bytes()).result == result
    assert job.cancel() is None


def test_snapshot_loading_fails_closed_without_native_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python never substitutes a fallback for Rust-owned persistence checks."""

    snapshot = qm.ExecutionSnapshot.remote(
        qm.ExecutionReference.create(provider="provider", job_ids=("job",))
    )
    payload = snapshot.to_bytes()
    monkeypatch.setattr(
        compilation_module.native,
        "inspect_execution_snapshot",
        None,
    )

    with pytest.raises(qm.SerializationError, match="native execution snapshot"):
        qm.ExecutionSnapshot.from_bytes(payload)


def test_snapshot_corruption_is_rejected_before_python_decoding() -> None:
    """The integrity digest rejects modified provider state."""

    payload = bytearray(
        qm.ExecutionSnapshot.remote(
            qm.ExecutionReference.create(provider="provider", job_ids=("job",))
        ).to_bytes()
    )
    payload[-1] ^= 1

    with pytest.raises(qm.SerializationError, match="Q8406"):
        qm.ExecutionSnapshot.from_bytes(bytes(payload))


def test_execution_reference_requires_sorted_unique_context() -> None:
    """Direct construction preserves the deterministic reference invariant."""

    with pytest.raises(ValueError, match="unique and sorted"):
        qm.ExecutionReference(
            "provider",
            ("job",),
            context=(("z", "1"), ("a", "2")),
        )
