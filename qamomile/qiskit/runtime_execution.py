"""Adapt Qiskit Runtime primitive jobs to Qamomile execution handles."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from typing import Any, Generic, TypeVar

from qamomile.circuit.transpiler.errors import ExecutionError
from qamomile.circuit.transpiler.execution_handle import (
    ExecutionHandle,
    ExecutionReference,
    JobStatus,
)
from qamomile.circuit.transpiler.execution_snapshot import (
    ExecutionSnapshot,
    ExecutionSnapshotKind,
)

ResultT = TypeVar("ResultT")

_RUNTIME_STATUSES: Mapping[str, JobStatus] = {
    "INITIALIZING": JobStatus.PENDING,
    "VALIDATING": JobStatus.PENDING,
    "QUEUED": JobStatus.QUEUED,
    "RUNNING": JobStatus.RUNNING,
    "CANCELLING": JobStatus.CANCELLING,
    "CANCELLED": JobStatus.CANCELLED,
    "DONE": JobStatus.COMPLETED,
    "ERROR": JobStatus.FAILED,
}


class RuntimeExecutionHandle(ExecutionHandle[ResultT], Generic[ResultT]):
    """Expose one Runtime primitive job without blocking its submission.

    Result retrieval runs once in a daemon thread so local wait limits also
    work with injected Qiskit primitive jobs whose ``result()`` method has no
    timeout argument. Expiring a wait neither cancels nor resubmits the job.

    Args:
        job (Any): Native Runtime or compatible local primitive job.
        decoder (Callable[[Any], ResultT]): Convert the primitive result into
            a backend-neutral value.
        reference (ExecutionReference | None): Secret-free restoration
            reference. Defaults to none without a restoration service.
    """

    def __init__(
        self,
        job: Any,
        decoder: Callable[[Any], ResultT],
        reference: ExecutionReference | None = None,
    ) -> None:
        """Initialize lazy retrieval and provider lifecycle delegation.

        Args:
            job (Any): Native primitive job.
            decoder (Callable[[Any], ResultT]): Result conversion function.
            reference (ExecutionReference | None): Serializable restoration
                reference. Defaults to none.
        """
        self._job = job
        self._decoder = decoder
        self._reference = reference
        self._future: Future[ResultT] | None = None
        self._retrieval_lock = threading.Lock()

    def result(self, timeout: float | None = None) -> ResultT:
        """Wait for the decoded primitive result with a local wait limit.

        Successful results and execution failures are cached. A provider-side
        result-wait timeout permits another retrieval attempt, while an
        expired local wait leaves the existing retrieval running.

        Args:
            timeout (float | None): Maximum local wait in seconds, including
                zero for an immediate check. ``None`` waits indefinitely.

        Returns:
            ResultT: Cached or newly decoded primitive result.

        Raises:
            ValueError: If the timeout is negative, boolean, or non-finite.
            TimeoutError: If the local or provider result wait expires.
            ExecutionError: If execution or result decoding fails.
        """
        if timeout is not None and (
            isinstance(timeout, bool) or not math.isfinite(timeout) or timeout < 0
        ):
            raise ValueError("timeout must be a non-negative finite number or None")
        with self._retrieval_lock:
            future = self._future
            if future is None or (
                future.done() and isinstance(future.exception(), TimeoutError)
            ):
                future = Future()
                self._future = future
                threading.Thread(
                    target=self._retrieve,
                    args=(future,),
                    daemon=True,
                    name="qamomile-qiskit-result",
                ).start()
        try:
            return future.result(timeout=timeout)
        except TimeoutError as error:
            raise TimeoutError("Qiskit Runtime execution result timed out") from error

    def _retrieve(self, future: Future[ResultT]) -> None:
        """Retrieve and decode one result for every concurrent waiter.

        Args:
            future (Future[ResultT]): Shared completion future to resolve.
        """
        try:
            raw_result = self._job.result()
        except Exception as error:
            if _is_result_wait_timeout(error):
                failure: Exception = TimeoutError(
                    "Qiskit Runtime provider result wait timed out"
                )
            else:
                failure = ExecutionError(f"Qiskit Runtime execution failed: {error}")
            failure.__cause__ = error
            future.set_exception(failure)
            return
        try:
            result = self._decoder(raw_result)
        except Exception as error:
            failure = ExecutionError(f"Qiskit Runtime result decoding failed: {error}")
            failure.__cause__ = error
            future.set_exception(failure)
            return
        future.set_result(result)

    def status(self) -> JobStatus:
        """Return the normalized provider or cached result status.

        Returns:
            JobStatus: Current provider-independent execution state.

        Raises:
            ExecutionError: If the native job status cannot be retrieved.
        """
        future = self._future
        if future is not None and future.done():
            error = future.exception()
            if error is None:
                return JobStatus.COMPLETED
            if not isinstance(error, TimeoutError):
                provider_status = _map_runtime_status(self.raw_status())
                # A result transport failure does not stop the provider job.
                # Preserve active states so composite handles can cancel it.
                if provider_status is JobStatus.COMPLETED:
                    return JobStatus.FAILED
                return provider_status
        return _map_runtime_status(self.raw_status())

    def raw_status(self) -> object:
        """Read the native string or Qiskit status enum.

        Returns:
            object: Unmodified provider status.

        Raises:
            ExecutionError: If the native status query fails.
        """
        try:
            return self._job.status()
        except Exception as error:
            raise ExecutionError(
                f"Cannot retrieve Qiskit Runtime job status: {error}"
            ) from error

    def cancel(self) -> None:
        """Request cancellation when the primitive job is unfinished.

        Raises:
            ExecutionError: If the status query or cancellation request fails.
        """
        if _map_runtime_status(self.raw_status()) in {
            JobStatus.COMPLETED,
            JobStatus.CANCELLED,
            JobStatus.FAILED,
        }:
            return
        try:
            self._job.cancel()
        except Exception as error:
            raise ExecutionError(
                f"Cannot cancel Qiskit Runtime job: {error}"
            ) from error

    def references(self) -> tuple[ExecutionReference, ...]:
        """Return the job's secret-free restoration reference when available.

        Returns:
            tuple[ExecutionReference, ...]: One remote reference or an empty
                tuple when no restoration service was configured.
        """
        return () if self._reference is None else (self._reference,)

    def snapshot(self) -> ExecutionSnapshot:
        """Capture a Runtime reference or an already decoded result.

        Restorable jobs retain their provider reference after result retrieval.
        Without a reference, result retrieval must have completed successfully
        before capturing its cached value. This method never starts retrieval,
        queries the provider, or waits for another result caller.

        Returns:
            ExecutionSnapshot: One remote reference or a detached local value.

        Raises:
            TypeError: If a cached result contains unsupported objects.
            ValueError: If no reference or successful cached result exists, or
                the cached value is nonfinite, cyclic, or nested too deeply.
        """
        if self._reference is not None:
            return ExecutionSnapshot(
                ExecutionSnapshotKind.REMOTE, reference=self._reference
            )
        future = self._future
        if (
            future is None
            or not future.done()
            or future.cancelled()
            or future.exception() is not None
        ):
            raise ValueError(
                "This Qiskit Runtime execution has no restorable reference or "
                "cached result; call result() successfully before snapshot()"
            )
        return ExecutionSnapshot(ExecutionSnapshotKind.LOCAL, value=future.result())

    def metadata(self) -> Mapping[str, Any]:
        """Return native Runtime job metrics when supported.

        Returns:
            Mapping[str, Any]: Provider metrics or an empty mapping when the
                local job does not expose metrics.

        Raises:
            ExecutionError: If native metric retrieval fails.
        """
        metrics = getattr(self._job, "metrics", None)
        if metrics is None:
            return {}
        try:
            return {"metrics": metrics()}
        except Exception as error:
            raise ExecutionError(
                f"Cannot retrieve Qiskit Runtime job metrics: {error}"
            ) from error

    @property
    def native(self) -> object:
        """Return the wrapped primitive job.

        Returns:
            object: Provider-native Runtime or local primitive job.
        """
        return self._job


def _map_runtime_status(status: object) -> JobStatus:
    """Normalize IBM Runtime strings and local Qiskit status enums.

    Args:
        status (object): Native status string or enum with a ``name`` field.

    Returns:
        JobStatus: Corresponding common status, or unknown for new values.
    """
    name = str(getattr(status, "name", status)).upper()
    return _RUNTIME_STATUSES.get(name, JobStatus.UNKNOWN)


def _is_result_wait_timeout(error: Exception) -> bool:
    """Distinguish local result timeouts from provider execution time limits.

    Args:
        error (Exception): Native result retrieval exception.

    Returns:
        bool: Whether retrieval can be retried without submitting another job.
    """
    try:
        from qiskit_ibm_runtime.exceptions import (
            RuntimeJobMaxTimeoutError,
            RuntimeJobTimeoutError,
        )
    except ImportError:
        return isinstance(error, TimeoutError)
    return not isinstance(error, RuntimeJobMaxTimeoutError) and isinstance(
        error, (TimeoutError, RuntimeJobTimeoutError)
    )


__all__ = ["RuntimeExecutionHandle"]
