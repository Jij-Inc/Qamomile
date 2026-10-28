"""Verify lazy Runtime primitive job lifecycle and local wait semantics."""

from __future__ import annotations

import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any

import pytest

from qamomile.circuit.transpiler.errors import ExecutionError
from qamomile.circuit.transpiler.execution_handle import (
    CompositeExecutionHandle,
    ExecutionReference,
    JobStatus,
)
from qamomile.circuit.transpiler.execution_snapshot import ExecutionSnapshotKind
from qamomile.qiskit.runtime_execution import RuntimeExecutionHandle


class _Job:
    """Provide a local primitive job without a result timeout argument.

    Args:
        value (Any): Result or exception returned on retrieval. Defaults to 3.
        status (object): Native job status or status error. Defaults to "DONE".
    """

    def __init__(self, value: Any = 3, status: object = "DONE") -> None:
        """Initialize retrieval counters and synchronization events.

        Args:
            value (Any): Result or exception. Defaults to 3.
            status (object): Native status or status error. Defaults to "DONE".
        """
        self.value = value
        self.state = status
        self.result_calls = 0
        self.cancel_calls = 0
        self.ready = threading.Event()
        self.ready.set()
        self.started = threading.Event()

    def result(self) -> Any:
        """Wait for test release and retrieve the configured result.

        Returns:
            Any: Configured result value.

        Raises:
            AssertionError: If the test does not release retrieval in time.
            Exception: If the configured result is an exception.
        """
        self.result_calls += 1
        self.started.set()
        assert self.ready.wait(5), "Test failed to release native result retrieval"
        if isinstance(self.value, Exception):
            raise self.value
        return self.value

    def status(self) -> object:
        """Retrieve the configured native status.

        Returns:
            object: Native status value.

        Raises:
            Exception: If the configured status is an exception.
        """
        if isinstance(self.state, Exception):
            raise self.state
        return self.state

    def cancel(self) -> None:
        """Count the cancellation request and mark the job cancelled."""
        self.cancel_calls += 1
        self.state = "CANCELLED"


def test_result_is_lazy_cached_and_shared_between_waiters() -> None:
    """Concurrent callers share one native retrieval and one conversion."""
    job = _Job()
    job.ready.clear()
    decode_calls = []

    def decode(value: int) -> int:
        """Record conversion and double the input value.

        Args:
            value (int): Native result to convert.

        Returns:
            int: Twice the native result.
        """
        decode_calls.append(value)
        return value * 2

    handle = RuntimeExecutionHandle(job, decode)
    assert job.result_calls == 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(handle.result)
        second = pool.submit(handle.result)
        assert job.started.wait(1)
        assert job.result_calls == 1
        job.ready.set()
        assert first.result(timeout=2) == second.result(timeout=2) == 6
    assert handle.result(timeout=0) == 6
    assert decode_calls == [3]
    assert handle.status() is JobStatus.COMPLETED


def test_local_timeout_does_not_cancel_or_restart_retrieval() -> None:
    """A timed-out local wait can later collect the same provider result."""
    job = _Job(status="RUNNING")
    job.ready.clear()
    handle = RuntimeExecutionHandle(job, int)
    try:
        with pytest.raises(TimeoutError):
            handle.result(timeout=0.01)
        assert job.started.wait(1)
        assert handle.status() is JobStatus.RUNNING
        assert job.cancel_calls == 0
        with pytest.raises(TimeoutError):
            handle.result(timeout=0)
        assert job.result_calls == 1
    finally:
        job.ready.set()
    assert handle.result(timeout=1) == 3
    assert job.result_calls == 1


@pytest.mark.parametrize("timeout", [-1, True, float("inf"), float("nan")])
def test_invalid_timeout_fails_before_native_retrieval(timeout: float) -> None:
    """Invalid wait budgets never start a native result request."""
    job = _Job()
    with pytest.raises(ValueError, match="timeout"):
        RuntimeExecutionHandle(job, int).result(timeout=timeout)
    assert job.result_calls == 0


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("INITIALIZING", JobStatus.PENDING),
        ("VALIDATING", JobStatus.PENDING),
        ("QUEUED", JobStatus.QUEUED),
        ("RUNNING", JobStatus.RUNNING),
        ("CANCELLING", JobStatus.CANCELLING),
        ("CANCELLED", JobStatus.CANCELLED),
        ("DONE", JobStatus.COMPLETED),
        ("ERROR", JobStatus.FAILED),
        ("UNRECOGNIZED", JobStatus.UNKNOWN),
        (
            Enum("LocalStatus", {"DONE": "job has successfully run"}).DONE,
            JobStatus.COMPLETED,
        ),
    ],
)
def test_status_normalizes_strings_and_qiskit_enums(
    status: object, expected: JobStatus
) -> None:
    """Both cloud Runtime and local primitive statuses use common states."""
    handle = RuntimeExecutionHandle(_Job(status=status), int)
    assert handle.status() is expected
    assert handle.raw_status() is status


def test_failure_is_normalized_and_cached() -> None:
    """Provider failures keep their cause and cannot resubmit or retrieve again."""
    error = RuntimeError("execution failed")
    job = _Job(error, status="ERROR")
    handle = RuntimeExecutionHandle(job, int)
    with pytest.raises(ExecutionError) as first:
        handle.result()
    with pytest.raises(ExecutionError) as second:
        handle.result()
    assert first.value is second.value
    assert first.value.__cause__ is error
    assert handle.status() is JobStatus.FAILED
    assert job.result_calls == 1


def test_decoder_error_is_cached_and_overrides_done_status() -> None:
    """Malformed result data is a failure even when the provider job is done."""
    job = _Job("not-an-integer")
    handle = RuntimeExecutionHandle(job, int)
    with pytest.raises(ExecutionError, match="decoding") as first:
        handle.result()
    with pytest.raises(ExecutionError) as second:
        handle.result()
    assert first.value is second.value
    assert handle.status() is JobStatus.FAILED
    assert job.result_calls == 1


def test_status_error_is_not_reported_as_completion() -> None:
    """A network failure during status lookup remains an explicit failure."""
    handle = RuntimeExecutionHandle(_Job(status=RuntimeError("unreachable")), int)
    with pytest.raises(ExecutionError, match="status"):
        handle.status()


def test_cancel_delegates_only_for_unfinished_job() -> None:
    """A cancellation request does not claim success before provider status."""
    job = _Job(status="RUNNING")
    handle = RuntimeExecutionHandle(job, int)
    handle.cancel()
    handle.cancel()
    assert job.cancel_calls == 1
    assert handle.status() is JobStatus.CANCELLED


def test_cancelled_result_keeps_cancelled_status() -> None:
    """Cancelled results raise an execution error without losing their state."""
    handle = RuntimeExecutionHandle(
        _Job(RuntimeError("cancelled"), status="CANCELLED"), int
    )
    with pytest.raises(ExecutionError):
        handle.result()
    assert handle.status() is JobStatus.CANCELLED


def test_cancel_after_result_transport_failure_uses_native_status() -> None:
    """A failed result request cannot prevent cancellation of a running job."""
    job = _Job(RuntimeError("result connection dropped"), status="RUNNING")
    handle = RuntimeExecutionHandle(job, int)
    with pytest.raises(ExecutionError):
        handle.result()
    assert handle.status() is JobStatus.RUNNING
    handle.cancel()
    assert job.cancel_calls == 1
    assert handle.status() is JobStatus.CANCELLED


@pytest.mark.parametrize(
    ("native_status", "expected_status"),
    [
        ("INITIALIZING", JobStatus.PENDING),
        ("QUEUED", JobStatus.QUEUED),
        ("RUNNING", JobStatus.RUNNING),
        ("UNRECOGNIZED", JobStatus.UNKNOWN),
    ],
)
def test_composite_cancels_active_job_after_result_transport_failure(
    native_status: str, expected_status: JobStatus
) -> None:
    """A failed result download must not hide an active child from cancellation."""
    failure = RuntimeError("result connection dropped")
    job = _Job(failure, status=native_status)
    child = RuntimeExecutionHandle(job, int)
    composite = CompositeExecutionHandle([child])

    with pytest.raises(ExecutionError) as caught:
        composite.result()

    assert caught.value.__cause__ is failure
    assert child.status() is expected_status
    assert composite.status() is expected_status
    composite.cancel()
    assert job.cancel_calls == 1
    assert child.status() is JobStatus.CANCELLED
    assert composite.status() is JobStatus.CANCELLED
    assert job.result_calls == 1


def test_reference_metadata_and_native_are_exposed() -> None:
    """Remote references and metrics remain available before result retrieval."""
    reference = ExecutionReference("qiskit_runtime", ("job-123",))
    job = _Job()
    job.metrics = lambda: {"usage": {"quantum_seconds": 2}}
    handle = RuntimeExecutionHandle(job, int, reference)
    assert handle.references() == (reference,)
    assert handle.native is job
    assert handle.metadata() == {"metrics": {"usage": {"quantum_seconds": 2}}}
    assert RuntimeExecutionHandle(_Job(), int).references() == ()
    assert RuntimeExecutionHandle(_Job(), int).metadata() == {}
    assert job.result_calls == 0


def test_snapshot_without_cached_result_never_calls_provider() -> None:
    """Snapshotting never starts retrieval or consults provider completion state."""
    job = _Job(status=AssertionError("Snapshot must not query status"))
    handle = RuntimeExecutionHandle(job, int)

    with pytest.raises(ValueError, match=r"call result\(\) successfully"):
        handle.snapshot()

    assert job.result_calls == 0


def test_snapshot_does_not_wait_for_an_active_result_retrieval() -> None:
    """An unfinished retrieval cannot block snapshotting or start a second fetch."""
    job = _Job(status="RUNNING")
    job.ready.clear()
    handle = RuntimeExecutionHandle(job, int)
    try:
        with pytest.raises(TimeoutError):
            handle.result(timeout=0)
        assert job.started.wait(1)
        with pytest.raises(ValueError, match=r"call result\(\) successfully"):
            handle.snapshot()
        assert job.result_calls == 1
    finally:
        job.ready.set()

    assert handle.result(timeout=1) == 3
    snapshot = handle.snapshot()
    assert snapshot.kind is ExecutionSnapshotKind.LOCAL
    assert snapshot.value == 3
    assert job.result_calls == 1


@pytest.mark.parametrize("value", [RuntimeError("retrieval failed"), "invalid"])
def test_snapshot_rejects_cached_retrieval_or_decoding_failures(value: Any) -> None:
    """A failed result is never serialized as a successful local execution."""
    job = _Job(value)
    handle = RuntimeExecutionHandle(job, int)
    with pytest.raises(ExecutionError):
        handle.result()

    with pytest.raises(ValueError, match=r"call result\(\) successfully"):
        handle.snapshot()

    assert job.result_calls == 1


@pytest.mark.parametrize("value", [3, RuntimeError("retrieval failed")])
def test_snapshot_preserves_remote_reference_after_result_retrieval(
    value: Any,
) -> None:
    """Remote jobs stay retrievable by reference after either retrieval outcome."""
    reference = ExecutionReference("qiskit-runtime", ("job-123",))
    job = _Job(value)
    handle = RuntimeExecutionHandle(job, int, reference)
    initial = handle.snapshot()
    assert job.result_calls == 0
    if isinstance(value, Exception):
        with pytest.raises(ExecutionError):
            handle.result()
    else:
        assert handle.result() == value

    assert handle.snapshot() == initial
    assert initial.kind is ExecutionSnapshotKind.REMOTE
    assert initial.reference == reference
    assert job.result_calls == 1


def test_provider_wait_timeout_is_retrievable_but_execution_timeout_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only result-wait expiration permits retrying a native result request."""
    exceptions = types.ModuleType("qiskit_ibm_runtime.exceptions")

    class RuntimeJobTimeoutError(Exception):
        """Represent a retryable expiration of a provider result wait.

        Example:
            Correct: retrieve a completed value with ``_Job(3).result()``.
            Incorrect: ``_Job(RuntimeJobTimeoutError()).result()`` raises
            this error to signal that the result is not yet available.
        """

        pass

    class RuntimeJobMaxTimeoutError(RuntimeJobTimeoutError):
        """Represent a terminal expiration of the provider execution limit.

        Example:
            Correct: retrieve a completed value with ``_Job(3).result()``.
            Incorrect: ``_Job(RuntimeJobMaxTimeoutError()).result()`` raises
            this error to signal that execution cannot be resumed.
        """

        pass

    exceptions.RuntimeJobTimeoutError = RuntimeJobTimeoutError
    exceptions.RuntimeJobMaxTimeoutError = RuntimeJobMaxTimeoutError
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime.exceptions", exceptions)
    job = _Job(RuntimeJobTimeoutError("wait expired"), status="RUNNING")
    handle = RuntimeExecutionHandle(job, int)
    with pytest.raises(TimeoutError):
        handle.result()
    assert handle.status() is JobStatus.RUNNING
    job.value = 4
    assert handle.result() == 4
    assert job.result_calls == 2

    expired = _Job(RuntimeJobMaxTimeoutError("QPU execution time limit"), "ERROR")
    failed = RuntimeExecutionHandle(expired, int)
    with pytest.raises(ExecutionError) as first:
        failed.result()
    with pytest.raises(ExecutionError) as second:
        failed.result()
    assert first.value is second.value
    assert failed.status() is JobStatus.FAILED
    assert expired.result_calls == 1
