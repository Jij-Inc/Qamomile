"""Tests for engine-neutral execution handles and request policies."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from qamomile.circuit.transpiler.execution_capability import ExecutionCapabilities
from qamomile.circuit.transpiler.execution_handle import (
    CompletedExecutionHandle,
    CompositeExecutionHandle,
    ExecutionHandle,
    ExecutionReference,
    JobStatus,
    MappedExecutionHandle,
)
from qamomile.circuit.transpiler.execution_request import ShotBased, TargetPrecision
from qamomile.circuit.transpiler.job import JobKind, JobSnapshot, RunJob, SampleJob
from qamomile.circuit.transpiler.quantum_executor import QuantumExecutor


class _RecordingHandle(ExecutionHandle[dict[str, int]]):
    """Expose controllable state for public-job delegation tests.

    Args:
        status (JobStatus): Initial normalized status. Defaults to queued.
        references (tuple[ExecutionReference, ...] | None): Restoration
            references, or None for one default test reference.
        status_error (Exception | None): Optional provider status failure.
        cancel_error (Exception | None): Optional provider cancellation failure.
    """

    def __init__(
        self,
        status: JobStatus = JobStatus.QUEUED,
        references: tuple[ExecutionReference, ...] | None = None,
        *,
        status_error: Exception | None = None,
        cancel_error: Exception | None = None,
    ) -> None:
        """Initialize a recording handle.

        Args:
            status (JobStatus): Initial normalized status.
            references (tuple[ExecutionReference, ...] | None): Restoration
                references, or None for one default test reference.
            status_error (Exception | None): Optional provider status failure.
            cancel_error (Exception | None): Optional provider cancellation failure.
        """
        self.current_status = status
        self.result_calls = 0
        self.cancel_calls = 0
        self.status_calls = 0
        self.status_error = status_error
        self.cancel_error = cancel_error
        self._references = (
            (ExecutionReference("test", ("job-1",)),)
            if references is None
            else references
        )

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

        Raises:
            Exception: If a provider status failure was configured.
        """
        self.status_calls += 1
        if self.status_error is not None:
            raise self.status_error
        return self.current_status

    def raw_status(self) -> object:
        """Return a provider-like status string.

        Returns:
            object: Test status string.
        """
        return self.current_status.name

    def cancel(self) -> None:
        """Record cancellation and transition to cancelled when successful.

        Raises:
            Exception: If a provider cancellation failure was configured.
        """
        self.cancel_calls += 1
        if self.cancel_error is not None:
            raise self.cancel_error
        self.current_status = JobStatus.CANCELLED

    def references(self) -> tuple[ExecutionReference, ...]:
        """Return the configured stable test references.

        Returns:
            tuple[ExecutionReference, ...]: Configured test references.
        """
        return self._references


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
    """Public cancellation reaches the engine handle."""
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


def test_composite_cancel_visits_unfinished_children_without_results() -> None:
    """Cancellation skips known terminal states without retrieving child results."""
    terminal = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
    children = [_RecordingHandle(status) for status in JobStatus]

    CompositeExecutionHandle(children).cancel()

    for initial, child in zip(JobStatus, children, strict=True):
        assert child.cancel_calls == (0 if initial in terminal else 1)
        assert child.status_calls == 1
        assert child.result_calls == 0
        if initial in terminal:
            assert child.current_status is initial
        else:
            assert child.current_status is JobStatus.CANCELLED


@pytest.mark.parametrize("statuses", [(), (JobStatus.COMPLETED, JobStatus.CANCELLED)])
def test_composite_cancel_empty_or_terminal_children_is_a_noop(statuses) -> None:
    """Empty and completed groups require no provider cancellation or results."""
    children = [_RecordingHandle(status) for status in statuses]

    assert CompositeExecutionHandle(children).cancel() is None

    assert all(child.cancel_calls == child.result_calls == 0 for child in children)


@pytest.mark.parametrize("failure_count", [1, 2])
def test_composite_cancel_reports_failures_after_trying_every_child(
    failure_count,
) -> None:
    """One or more cancellation failures cannot prevent later provider requests."""
    failures = (RuntimeError("first cancellation failed"), OSError("second failed"))[
        :failure_count
    ]
    children = [
        _RecordingHandle(JobStatus.RUNNING, cancel_error=failures[0]),
        _RecordingHandle(JobStatus.QUEUED),
        _RecordingHandle(
            JobStatus.RUNNING, cancel_error=failures[1] if failure_count == 2 else None
        ),
        _RecordingHandle(JobStatus.PENDING),
    ]

    with pytest.raises(ExceptionGroup, match="Failed to cancel") as caught:
        CompositeExecutionHandle(children).cancel()

    assert caught.value.exceptions == failures
    assert all(error.__traceback__ is not None for error in caught.value.exceptions)
    assert all(child.cancel_calls == child.status_calls == 1 for child in children)
    assert all(child.result_calls == 0 for child in children)
    assert children[-1].current_status is JobStatus.CANCELLED


@pytest.mark.parametrize("cancel_fails", [False, True])
def test_composite_cancel_attempts_children_whose_status_is_unavailable(
    cancel_fails,
) -> None:
    """A status error is retained while both that child and later jobs are cancelled."""
    status_failure = ConnectionError("status unavailable")
    cancel_failure = RuntimeError("cancel unavailable") if cancel_fails else None
    unknown = _RecordingHandle(
        JobStatus.RUNNING, status_error=status_failure, cancel_error=cancel_failure
    )
    later = _RecordingHandle(JobStatus.RUNNING)

    with pytest.raises(ExceptionGroup) as caught:
        CompositeExecutionHandle((unknown, later)).cancel()

    expected = (status_failure, cancel_failure) if cancel_fails else (status_failure,)
    assert caught.value.exceptions == expected
    assert all(error.__traceback__ is not None for error in caught.value.exceptions)
    assert unknown.status_calls == unknown.cancel_calls == 1
    assert later.status_calls == later.cancel_calls == 1
    assert unknown.result_calls == later.result_calls == 0
    assert later.current_status is JobStatus.CANCELLED


def test_nested_composite_cancel_preserves_failures_and_reaches_siblings() -> None:
    """A failed nested cancellation retains its exception group and visits siblings."""
    failure = RuntimeError("nested cancellation failed")
    failed = _RecordingHandle(cancel_error=failure)
    inner_sibling = _RecordingHandle()
    outer_sibling = _RecordingHandle()
    inner = CompositeExecutionHandle((failed, inner_sibling))

    with pytest.raises(ExceptionGroup) as caught:
        CompositeExecutionHandle((inner, outer_sibling)).cancel()

    assert len(caught.value.exceptions) == 1
    nested = caught.value.exceptions[0]
    assert isinstance(nested, ExceptionGroup)
    assert nested.exceptions == (failure,)
    assert failure.__traceback__ is not None
    assert all(
        child.cancel_calls == 1 and child.result_calls == 0
        for child in (failed, inner_sibling, outer_sibling)
    )
    assert (
        inner_sibling.current_status
        is outer_sibling.current_status
        is JobStatus.CANCELLED
    )


@pytest.mark.parametrize("method", ["status", "cancel"])
def test_composite_cancel_does_not_group_process_interruptions(
    monkeypatch, method
) -> None:
    """Process interruptions propagate immediately instead of becoming provider failures."""
    interrupted = _RecordingHandle()
    later = _RecordingHandle()
    failure = KeyboardInterrupt("interrupted")
    monkeypatch.setattr(interrupted, method, Mock(side_effect=failure))

    with pytest.raises(KeyboardInterrupt) as caught:
        CompositeExecutionHandle((interrupted, later)).cancel()

    assert caught.value is failure
    assert later.cancel_calls == later.status_calls == later.result_calls == 0


@pytest.mark.parametrize(
    "reference_counts",
    [
        (),
        (0,),
        (1,),
        (2,),
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 0, 1),
        (1, 1),
        (2, 1),
        (1, 2),
        (2, 2),
    ],
)
def test_composite_snapshots_preserve_local_children_and_reject_opaque_groups(
    reference_counts: tuple[int, ...],
) -> None:
    """Snapshots preserve child boundaries or reject ambiguous restoration."""
    children: list[ExecutionHandle[dict[str, int]]] = []
    all_references = []
    for child_index, count in enumerate(reference_counts):
        references = tuple(
            ExecutionReference("test", (f"job-{child_index}-{reference_index}",))
            for reference_index in range(count)
        )
        all_references.extend(references)
        children.append(
            _RecordingHandle(references=references)
            if references
            else CompletedExecutionHandle({"0": 3})
        )
    handle = CompositeExecutionHandle(children)
    job = RunJob.from_handle(handle)
    expected = (
        tuple(all_references) if all(count == 1 for count in reference_counts) else ()
    )

    assert handle.references() == expected
    assert all(
        child.result_calls == 0
        for child in children
        if isinstance(child, _RecordingHandle)
    )
    if all(count <= 1 for count in reference_counts):
        snapshot = job.snapshot()
        assert snapshot.kind is JobKind.RUN
        assert snapshot.executions == tuple(all_references)
        assert snapshot.shots is None
        assert JobSnapshot.from_dict(json.loads(json.dumps(snapshot.to_dict()))) == (
            snapshot
        )
        restored = snapshot.execution.restore(lambda reference: _RecordingHandle())
        assert all(
            child.result_calls == 0
            for child in children
            if isinstance(child, _RecordingHandle)
        )
        assert restored.result() == tuple(
            {"1": 3} if count else {"0": 3} for count in reference_counts
        )
    else:
        with pytest.raises(ValueError, match="multiple references"):
            job.snapshot()

    assert job.result() == tuple(
        {"1": 3} if count else {"0": 3} for count in reference_counts
    )
    assert handle.references() == expected


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


def test_nested_snapshot_restores_boundaries_without_waiting() -> None:
    """Nested groups and a multiple-ID provider leaf keep distinct result slots."""
    remote = _RecordingHandle(
        references=(ExecutionReference("test", ("first", "second")),)
    )
    handle = CompositeExecutionHandle(
        [
            CompletedExecutionHandle(0.25),
            CompositeExecutionHandle([remote, CompletedExecutionHandle((True, [3]))]),
            CompletedExecutionHandle({"": 4}),
        ]
    )
    snapshot = JobSnapshot.from_dict(
        json.loads(json.dumps(RunJob.from_handle(handle).snapshot().to_dict()))
    )
    restored_remote = _RecordingHandle(references=remote.references())
    retrieve = Mock(return_value=restored_remote)
    restored = snapshot.execution.restore(retrieve)

    retrieve.assert_called_once_with(remote.references()[0])
    assert remote.result_calls == remote.status_calls == 0
    assert restored_remote.result_calls == restored_remote.status_calls == 0
    result = restored.result()
    original = handle.result()
    np.testing.assert_allclose(
        [result[0], original[0]], [0.25, 0.25], atol=1e-12, rtol=1e-12
    )
    assert result[1:] == original[1:] == (({"1": 3}, (True, [3])), {"": 4})
    assert type(result) is tuple
    assert type(result[0]) is float
    assert type(result[1]) is tuple
    assert type(result[1][1][0]) is bool
    assert type(result[1][1][1]) is list


def test_snapshot_refuses_to_drop_an_arbitrary_mapping() -> None:
    """A callable requires an owner that can reconstruct the same transformation."""
    source = _RecordingHandle()
    handle = MappedExecutionHandle(source, lambda counts: 4.0)
    with pytest.raises(ValueError, match="arbitrary mapped execution"):
        RunJob.from_handle(handle).snapshot()
    assert source.result_calls == 0
    np.testing.assert_allclose(handle.result(), 4.0, atol=1e-12, rtol=1e-12)


def test_unknown_execution_cannot_snapshot_by_waiting_for_a_result() -> None:
    """A completed provider state alone does not authorize reading its result."""
    handle = _RecordingHandle(status=JobStatus.COMPLETED, references=())
    with pytest.raises(ValueError, match="restorable reference"):
        RunJob.from_handle(handle).snapshot()
    assert handle.result_calls == handle.status_calls == 0


@pytest.mark.parametrize("version", [0, 3, True, "2", None])
def test_job_snapshot_rejects_unknown_versions(version) -> None:
    """Unsupported schemas cannot fall back to a lossy legacy interpretation."""
    data = RunJob.from_handle(CompletedExecutionHandle(0.25)).snapshot().to_dict()
    data["version"] = version
    with pytest.raises(ValueError, match="version"):
        JobSnapshot.from_dict(data)


def test_job_snapshot_rejects_tree_reference_disagreement() -> None:
    """Dropping a tree or its remote-reference inventory is detected."""
    data = RunJob.from_handle(_RecordingHandle()).snapshot().to_dict()
    data["executions"] = []
    with pytest.raises(ValueError, match="disagree"):
        JobSnapshot.from_dict(data)
    data.pop("execution")
    with pytest.raises(KeyError, match="execution"):
        JobSnapshot.from_dict(data)


def test_job_snapshot_requires_version_for_structured_execution() -> None:
    """A tree without its schema version is not interpreted as a legacy job."""
    data = RunJob.from_handle(_RecordingHandle()).snapshot().to_dict()
    data.pop("version")
    with pytest.raises(ValueError, match="unknown fields"):
        JobSnapshot.from_dict(data)


def test_structured_snapshot_rejects_unknown_reference_inventory_fields() -> None:
    """Both copies of a version 2 provider reference reject unrecognized data."""
    data = RunJob.from_handle(_RecordingHandle()).snapshot().to_dict()
    data["executions"][0]["garbage"] = ["would be lost"]
    with pytest.raises(ValueError, match="unknown reference fields"):
        JobSnapshot.from_dict(data)
