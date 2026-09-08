"""Tests for backend-neutral execution handles and request policies."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from qamomile.circuit.transpiler.execution_capability import ExecutionCapabilities
from qamomile.circuit.transpiler.execution_handle import (
    CompositeExecutionHandle,
    ExecutionHandle,
    ExecutionReference,
    JobStatus,
)
from qamomile.circuit.transpiler.execution_request import ShotBased, TargetPrecision
from qamomile.circuit.transpiler.job import JobKind, JobSnapshot, SampleJob
from qamomile.circuit.transpiler.quantum_executor import QuantumExecutor


class _RecordingHandle(ExecutionHandle[dict[str, int]]):
    """Expose controllable state for public-job delegation tests."""

    def __init__(self, status: JobStatus = JobStatus.QUEUED) -> None:
        """Initialize a recording handle.

        Args:
            status (JobStatus): Initial normalized status.
        """
        self.current_status = status
        self.result_calls = 0
        self.cancel_calls = 0

    def result(self, timeout: float | None = None) -> dict[str, int]:
        """Return deterministic counts and record retrieval.

        Args:
            timeout (float | None): Ignored test timeout.

        Returns:
            dict[str, int]: Deterministic counts.
        """
        self.result_calls += 1
        self.current_status = JobStatus.COMPLETED
        return {"1": 3}

    def status(self) -> JobStatus:
        """Return the controllable status.

        Returns:
            JobStatus: Current test status.
        """
        return self.current_status

    def raw_status(self) -> object:
        """Return a provider-like status string.

        Returns:
            object: Test status string.
        """
        return self.current_status.name

    def cancel(self) -> None:
        """Record cancellation and transition to cancelled."""
        self.cancel_calls += 1
        self.current_status = JobStatus.CANCELLED

    def references(self) -> tuple[ExecutionReference, ...]:
        """Return one stable test reference.

        Returns:
            tuple[ExecutionReference, ...]: Test reference.
        """
        return (ExecutionReference("test", ("job-1",)),)


class _SynchronousEstimateExecutor(QuantumExecutor[str]):
    """Provide synchronous sampling and estimation for capability tests."""

    def execute(self, circuit: str, shots: int) -> dict[str, int]:
        """Return deterministic counts.

        Args:
            circuit (str): Ignored circuit value.
            shots (int): Count returned for the zero state.

        Returns:
            dict[str, int]: Deterministic counts.
        """
        return {"0": shots}

    def estimate(
        self,
        circuit: str,
        hamiltonian: Any,
        params: Any = None,
    ) -> float:
        """Return a deterministic expectation value.

        Args:
            circuit (Any): Ignored circuit value.
            hamiltonian (Any): Ignored observable value.
            params (Any): Ignored parameter values.

        Returns:
            float: Constant expectation value.
        """
        return 0.0


def test_sample_job_preserves_lazy_handle_lifecycle() -> None:
    """Constructing a public job does not retrieve a provider result."""
    handle = _RecordingHandle()

    job = SampleJob(handle, lambda counts: [(1, counts["1"])], shots=3)

    assert handle.result_calls == 0
    assert job.status() is JobStatus.QUEUED
    assert job.raw_status() == "QUEUED"
    assert job.references()[0].job_ids == ("job-1",)
    assert job.result().results == [(1, 3)]
    assert handle.result_calls == 1
    assert job.result().results == [(1, 3)]
    assert handle.result_calls == 1


def test_sample_job_delegates_cancel() -> None:
    """Public cancellation reaches the backend handle."""
    handle = _RecordingHandle(JobStatus.RUNNING)
    job = SampleJob(handle, lambda counts: [], shots=1)

    job.cancel()

    assert handle.cancel_calls == 1
    assert job.status() is JobStatus.CANCELLED


def test_sample_job_supports_async_result() -> None:
    """Public jobs expose a non-blocking async compatibility path."""
    handle = _RecordingHandle()
    job = SampleJob(handle, lambda counts: [(1, counts["1"])], shots=3)

    result = asyncio.run(job.result_async())

    assert result.results == [(1, 3)]
    assert handle.result_calls == 1


def test_composite_status_retains_partial_terminal_outcome() -> None:
    """Mixed success and failure is not collapsed into generic failure."""
    completed = _RecordingHandle(JobStatus.COMPLETED)
    failed = _RecordingHandle(JobStatus.FAILED)

    handle = CompositeExecutionHandle((completed, failed))

    assert handle.status() is JobStatus.PARTIAL


def test_composite_zero_timeout_returns_ready_results() -> None:
    """A zero wait budget still permits immediately available child results."""
    completed = _RecordingHandle(JobStatus.COMPLETED)

    handle = CompositeExecutionHandle((completed,))

    assert handle.result(timeout=0) == ({"1": 3},)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ShotBased(0), "positive"),
        (lambda: ShotBased(True), "positive"),
        (lambda: TargetPrecision(0.0), "positive"),
        (lambda: TargetPrecision(float("nan")), "positive"),
    ],
)
def test_accuracy_policies_reject_non_positive_values(factory, message: str) -> None:
    """Invalid accuracy requests fail before provider submission."""
    with pytest.raises(ValueError, match=message):
        factory()


def test_original_job_status_values_remain_stable() -> None:
    """Adding cloud states preserves serialization of existing statuses."""
    assert JobStatus.PENDING.value == 1
    assert JobStatus.RUNNING.value == 2
    assert JobStatus.COMPLETED.value == 3
    assert JobStatus.FAILED.value == 4


def test_execution_reference_rejects_missing_identifiers() -> None:
    """Restorable references always contain a provider and at least one ID."""
    with pytest.raises(ValueError, match="provider"):
        ExecutionReference("", ("job",))
    with pytest.raises(ValueError, match="job_ids"):
        ExecutionReference("provider", ())


def test_default_capabilities_preserve_synchronous_executor_compatibility() -> None:
    """Existing executors gain conservative lifecycle feature declarations."""
    capabilities = _SynchronousEstimateExecutor().capabilities

    assert capabilities.supports_estimation
    assert not capabilities.supports_async_sampling
    assert not capabilities.supports_async_estimation
    assert not capabilities.supports_restoration


def test_capabilities_reject_accuracy_without_estimation() -> None:
    """Accuracy policies cannot be advertised without estimation support."""
    with pytest.raises(ValueError, match="supports_estimation"):
        ExecutionCapabilities(estimation_accuracy=frozenset({ShotBased}))


def test_job_snapshot_validates_operation_specific_shots() -> None:
    """Sample and run snapshots retain distinct shot-count contracts."""
    reference = ExecutionReference("test", ("job-1",))

    assert JobSnapshot(JobKind.SAMPLE, (reference,), 10).shots == 10
    with pytest.raises(ValueError, match="positive"):
        JobSnapshot(JobKind.SAMPLE, (reference,))
    with pytest.raises(ValueError, match="must not contain shots"):
        JobSnapshot(JobKind.RUN, (reference,), 1)


def test_job_snapshot_round_trips_through_json() -> None:
    """Typed restoration metadata can cross process boundaries as JSON."""
    snapshot = JobSnapshot(
        JobKind.SAMPLE,
        (
            ExecutionReference(
                "provider",
                ("job-1", "job-2"),
                target="device",
                group_id="batch",
                context={"kind": "sample"},
            ),
        ),
        shots=100,
    )

    payload = json.loads(json.dumps(snapshot.to_dict()))

    assert JobSnapshot.from_dict(payload) == snapshot
