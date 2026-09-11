"""Adapt Amazon Braket task lifecycles to Qamomile execution handles."""

from __future__ import annotations

import dataclasses
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from numbers import Integral
from typing import Any, Generic, TypeVar, cast

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


@dataclasses.dataclass(frozen=True)
class BraketExecutionOptions:
    """Configure Braket task and batch submission without flat kwargs.

    Args:
        s3_destination_folder (tuple[str, str] | None): S3 bucket and prefix
            for AWS task results. Defaults to the SDK configuration.
        reservation_arn (str | None): Direct reservation ARN. Defaults to
            ``None``.
        max_parallel (int | None): Maximum AWS batch concurrency. Defaults to
            the SDK configuration.
        poll_timeout_seconds (float | None): Provider result polling timeout and
            default local result-wait limit. Defaults to the SDK configuration
            with no local limit.
        poll_interval_seconds (float | None): Provider status polling interval.
            Defaults to the SDK configuration.
        batch_max_retries (int): Maximum explicit Braket batch resubmissions.
            Defaults to zero to prevent implicit additional QPU cost.
        task_options (Mapping[str, Any]): Additional ``device.run`` options.
        batch_options (Mapping[str, Any]): Additional ``device.run_batch``
            options.

    Raises:
        ValueError: If options contain executor-owned argument names or an
            invalid numeric value.
    """

    s3_destination_folder: tuple[str, str] | None = None
    reservation_arn: str | None = None
    max_parallel: int | None = None
    poll_timeout_seconds: float | None = None
    poll_interval_seconds: float | None = None
    batch_max_retries: int = 0
    task_options: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    batch_options: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy and validate provider option mappings.

        Raises:
            ValueError: If a reserved argument or invalid numeric value is
                present.
        """
        reserved = {
            "inputs",
            "max_parallel",
            "poll_interval_seconds",
            "poll_timeout_seconds",
            "reservation_arn",
            "s3_destination_folder",
            "shots",
        }
        task_options = dict(self.task_options)
        batch_options = dict(self.batch_options)
        invalid = reserved & (task_options.keys() | batch_options.keys())
        if invalid:
            raise ValueError(
                "Braket execution options contain executor-owned keys: "
                f"{sorted(invalid)}"
            )
        if self.s3_destination_folder is not None and (
            not isinstance(self.s3_destination_folder, tuple)
            or len(self.s3_destination_folder) != 2
            or not isinstance(self.s3_destination_folder[0], str)
            or not self.s3_destination_folder[0]
            or not isinstance(self.s3_destination_folder[1], str)
        ):
            raise ValueError("s3_destination_folder must contain a bucket and prefix")
        if self.reservation_arn is not None and (
            not isinstance(self.reservation_arn, str) or not self.reservation_arn
        ):
            raise ValueError("reservation_arn must not be empty")
        if self.max_parallel is not None and (
            isinstance(self.max_parallel, bool)
            or not isinstance(cast(object, self.max_parallel), Integral)
            or self.max_parallel <= 0
        ):
            raise ValueError("max_parallel must be a positive integer")
        if (
            isinstance(self.batch_max_retries, bool)
            or not isinstance(cast(object, self.batch_max_retries), Integral)
            or self.batch_max_retries < 0
        ):
            raise ValueError("batch_max_retries must be a non-negative integer")
        for name, value in (
            ("poll_timeout_seconds", self.poll_timeout_seconds),
            ("poll_interval_seconds", self.poll_interval_seconds),
        ):
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number")
        if self.max_parallel is not None:
            object.__setattr__(self, "max_parallel", int(self.max_parallel))
        object.__setattr__(self, "batch_max_retries", int(self.batch_max_retries))
        if self.poll_timeout_seconds is not None:
            object.__setattr__(
                self, "poll_timeout_seconds", float(self.poll_timeout_seconds)
            )
        if self.poll_interval_seconds is not None:
            object.__setattr__(
                self, "poll_interval_seconds", float(self.poll_interval_seconds)
            )
        object.__setattr__(self, "task_options", task_options)
        object.__setattr__(self, "batch_options", batch_options)

    def task_kwargs(self) -> dict[str, Any]:
        """Build keyword arguments for one Braket task.

        Returns:
            dict[str, Any]: Validated ``device.run`` keyword arguments.
        """
        values = dict(self.task_options)
        self._add_shared_options(values)
        return values

    def batch_kwargs(self) -> dict[str, Any]:
        """Build keyword arguments for a Braket task batch.

        Returns:
            dict[str, Any]: Validated ``device.run_batch`` keyword arguments.
        """
        values = dict(self.batch_options)
        self._add_shared_options(values)
        if self.max_parallel is not None:
            values["max_parallel"] = self.max_parallel
        return values

    def _add_shared_options(self, values: dict[str, Any]) -> None:
        """Add non-null options accepted by tasks and batches.

        Args:
            values (dict[str, Any]): Option mapping to update in place.
        """
        optional = {
            "s3_destination_folder": self.s3_destination_folder,
            "reservation_arn": self.reservation_arn,
            "poll_timeout_seconds": self.poll_timeout_seconds,
            "poll_interval_seconds": self.poll_interval_seconds,
        }
        values.update(
            {key: value for key, value in optional.items() if value is not None}
        )


class BraketExecutionHandle(ExecutionHandle[ResultT], Generic[ResultT]):
    """Wrap one Braket task or a task batch without blocking submission.

    Args:
        tasks (Sequence[Any]): Provider quantum tasks when individually
            addressable.
        result_loader (Callable[[], Sequence[Any]]): Blocking raw-result
            loader that does not perform implicit retries unless configured.
        decoder (Callable[[Sequence[Any]], ResultT]): Engine result decoder.
        reference (ExecutionReference | None): Serializable AWS reference.
        reference_factory (Callable[[], ExecutionReference | None] | None):
            Dynamic reference builder for batches that explicitly resubmit
            failed tasks. Defaults to none.
        native (object): Native Braket task or batch object.
        poll_interval_seconds (float): Local status polling interval.
        default_timeout_seconds (float | None): Default local result timeout.
            Defaults to no timeout.
        allow_unsuccessful_loader (bool): Whether the result loader owns
            explicit recovery from failed child tasks. Defaults to false.
    """

    def __init__(
        self,
        *,
        tasks: Sequence[Any],
        result_loader: Callable[[], Sequence[Any]],
        decoder: Callable[[Sequence[Any]], ResultT],
        reference: ExecutionReference | None,
        reference_factory: Callable[[], ExecutionReference | None] | None = None,
        native: object,
        poll_interval_seconds: float = 1.0,
        default_timeout_seconds: float | None = None,
        allow_unsuccessful_loader: bool = False,
    ) -> None:
        """Initialize a Braket-backed execution handle.

        Args:
            tasks (Sequence[Any]): Individually addressable Braket tasks.
            result_loader (Callable[[], Sequence[Any]]): Blocking loader.
            decoder (Callable[[Sequence[Any]], ResultT]): Result decoder.
            reference (ExecutionReference | None): Serializable AWS reference.
            reference_factory (Callable[[], ExecutionReference | None] | None):
                Dynamic reference builder. Defaults to none.
            native (object): Native task or batch.
            poll_interval_seconds (float): Positive local polling interval.
            default_timeout_seconds (float | None): Positive default local
                result timeout. Defaults to no timeout.
            allow_unsuccessful_loader (bool): Whether failed children may be
                handled by the explicit result loader. Defaults to false.

        Raises:
            ValueError: If the polling interval or default timeout is invalid.
        """
        if (
            isinstance(poll_interval_seconds, bool)
            or not math.isfinite(poll_interval_seconds)
            or poll_interval_seconds <= 0
        ):
            raise ValueError("poll_interval_seconds must be a positive finite number")
        if default_timeout_seconds is not None and (
            isinstance(default_timeout_seconds, bool)
            or not math.isfinite(default_timeout_seconds)
            or default_timeout_seconds <= 0
        ):
            raise ValueError("default_timeout_seconds must be a positive finite number")
        self._tasks = tuple(tasks)
        self._result_loader = result_loader
        self._decoder = decoder
        self._reference = reference
        self._reference_factory = reference_factory
        self._native = native
        self._poll_interval_seconds = float(poll_interval_seconds)
        self._default_timeout_seconds = (
            None if default_timeout_seconds is None else float(default_timeout_seconds)
        )
        self._allow_unsuccessful_loader = allow_unsuccessful_loader
        self._result: ResultT | None = None
        self._has_result = False
        self._failed = False
        self._error: Exception | None = None
        self._result_lock = threading.Lock()

    def result(self, timeout: float | None = None) -> ResultT:
        """Wait for and decode all Braket task results.

        Args:
            timeout (float | None): Maximum local status-wait time in seconds.
                ``None`` uses the configured Braket polling timeout when one
                exists, otherwise waits indefinitely. Expiration does not
                cancel remote tasks.

        Returns:
            ResultT: Decoded engine-neutral result.

        Raises:
            TimeoutError: If ``timeout`` expires before terminal state.
            ValueError: If ``timeout`` is negative or not finite.
            ExecutionError: If a task fails, is cancelled, or omits a result.
        """
        effective_timeout = (
            self._default_timeout_seconds if timeout is None else timeout
        )
        if effective_timeout is not None and (
            isinstance(effective_timeout, bool)
            or not math.isfinite(effective_timeout)
            or effective_timeout < 0
        ):
            raise ValueError("timeout must be a non-negative finite number or None")
        started = time.monotonic()
        if effective_timeout is None:
            self._result_lock.acquire()
            remaining = None
        else:
            acquired = self._result_lock.acquire(timeout=float(effective_timeout))
            if not acquired:
                raise TimeoutError("Amazon Braket execution result timed out")
            remaining = max(
                0.0,
                float(effective_timeout) - (time.monotonic() - started),
            )
        try:
            return self._result_under_lock(remaining)
        finally:
            self._result_lock.release()

    def _result_under_lock(self, timeout: float | None) -> ResultT:
        """Retrieve one result while serializing concurrent callers.

        Args:
            timeout (float | None): Remaining local wait budget in seconds.

        Returns:
            ResultT: Cached or newly decoded execution result.

        Raises:
            TimeoutError: If the remaining wait budget expires.
            ExecutionError: If a provider task is unsuccessful.
            Exception: If native retrieval or result decoding fails.
        """
        if self._has_result:
            return self._result  # type: ignore[return-value]
        if self._error is not None:
            raise self._error
        self._wait_for_terminal_state(timeout)
        unsuccessful = [
            task
            for task in self._tasks
            if _map_braket_status(str(task.state()))
            in {JobStatus.FAILED, JobStatus.CANCELLED}
        ]
        if unsuccessful and not self._allow_unsuccessful_loader:
            details = ", ".join(
                f"{getattr(task, 'id', '<unknown>')}={task.state()}"
                for task in unsuccessful
            )
            self._failed = True
            self._error = ExecutionError(
                f"Amazon Braket task did not complete: {details}"
            )
            raise self._error
        raw_results = tuple(self._result_loader())
        if any(result is None for result in raw_results):
            raise ExecutionError("Amazon Braket returned an empty task result")
        self._result = self._decoder(raw_results)
        if self._reference_factory is not None:
            self._reference = self._reference_factory()
        self._has_result = True
        return self._result

    def status(self) -> JobStatus:
        """Return the aggregate Braket task status.

        Returns:
            JobStatus: Provider-independent aggregate status.
        """
        if self._failed:
            return JobStatus.FAILED
        if self._has_result:
            return JobStatus.COMPLETED
        if not self._tasks:
            return JobStatus.COMPLETED
        return _aggregate_braket_status(
            tuple(_map_braket_status(str(task.state())) for task in self._tasks)
        )

    def raw_status(self) -> object:
        """Return raw task states in stable order.

        Returns:
            object: One state string or a tuple of state strings.
        """
        states = tuple(str(task.state()) for task in self._tasks)
        if len(states) == 1:
            return states[0]
        return states

    def cancel(self) -> None:
        """Request best-effort cancellation of every unfinished task."""
        for task in self._tasks:
            if _map_braket_status(str(task.state())) not in {
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            }:
                task.cancel()

    def references(self) -> tuple[ExecutionReference, ...]:
        """Return the logical Braket execution reference.

        Returns:
            tuple[ExecutionReference, ...]: Empty for local tasks, otherwise
                one reference containing every task ARN.
        """
        reference = (
            self._reference_factory()
            if self._reference_factory is not None
            else self._reference
        )
        return () if reference is None else (reference,)

    def snapshot(self) -> ExecutionSnapshot:
        """Capture an AWS reference or an already retrieved local result.

        AWS tasks retain their provider references after result retrieval.
        Local tasks require a successful ``result()`` call first; this method
        never retrieves results or waits for another result caller.

        Returns:
            ExecutionSnapshot: One remote reference or a detached local value.

        Raises:
            TypeError: If a cached local result contains unsupported objects.
            ValueError: If no reference or cached result exists, or the cached
                value is nonfinite, cyclic, or nested too deeply.
        """
        references = self.references()
        if references:
            return ExecutionSnapshot(
                ExecutionSnapshotKind.REMOTE, reference=references[0]
            )
        if not self._has_result:
            raise ValueError(
                "This Braket execution has no restorable reference or cached "
                "result; call result() successfully before snapshot()"
            )
        return ExecutionSnapshot(ExecutionSnapshotKind.LOCAL, value=self._result)

    def metadata(self) -> Mapping[str, Any]:
        """Return cached-or-provider metadata for every task.

        Returns:
            Mapping[str, Any]: Child task metadata in submission order.
        """
        values = []
        for task in self._tasks:
            metadata = getattr(task, "metadata", None)
            if metadata is None:
                values.append({})
                continue
            try:
                values.append(metadata(use_cached_value=True))
            except TypeError:
                values.append(metadata())
        return {"tasks": tuple(values)}

    @property
    def native(self) -> object | None:
        """Return the native Braket task or batch.

        Returns:
            object | None: Native Braket execution object.
        """
        return self._native

    def _wait_for_terminal_state(self, timeout: float | None) -> None:
        """Poll task states until all tasks become terminal.

        Args:
            timeout (float | None): Maximum local wait in seconds.

        Raises:
            TimeoutError: If tasks remain non-terminal after ``timeout``.
        """
        if not self._tasks:
            return
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            statuses = tuple(
                _map_braket_status(str(task.state())) for task in self._tasks
            )
            if all(
                status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
                for status in statuses
            ):
                return
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Amazon Braket execution result timed out")
                time.sleep(min(self._poll_interval_seconds, remaining))
            else:
                time.sleep(self._poll_interval_seconds)


def _map_braket_status(status: str) -> JobStatus:
    """Map one Braket task state to the common status model.

    Args:
        status (str): Braket task state.

    Returns:
        JobStatus: Provider-independent status.
    """
    return {
        "CREATED": JobStatus.PENDING,
        "QUEUED": JobStatus.QUEUED,
        "RUNNING": JobStatus.RUNNING,
        "CANCELLING": JobStatus.CANCELLING,
        "COMPLETED": JobStatus.COMPLETED,
        "FAILED": JobStatus.FAILED,
        "CANCELLED": JobStatus.CANCELLED,
    }.get(status.upper(), JobStatus.UNKNOWN)


def _aggregate_braket_status(statuses: tuple[JobStatus, ...]) -> JobStatus:
    """Aggregate Braket child states while retaining partial outcomes.

    Args:
        statuses (tuple[JobStatus, ...]): Normalized child statuses.

    Returns:
        JobStatus: Aggregate status.
    """
    if not statuses or all(status is JobStatus.COMPLETED for status in statuses):
        return JobStatus.COMPLETED
    for active in (
        JobStatus.RUNNING,
        JobStatus.CANCELLING,
        JobStatus.QUEUED,
        JobStatus.PENDING,
    ):
        if active in statuses:
            return active
    terminal = {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    }
    if all(status in terminal for status in statuses):
        if all(status is JobStatus.FAILED for status in statuses):
            return JobStatus.FAILED
        if all(status is JobStatus.CANCELLED for status in statuses):
            return JobStatus.CANCELLED
        return JobStatus.PARTIAL
    return JobStatus.UNKNOWN


__all__ = ["BraketExecutionHandle", "BraketExecutionOptions"]
