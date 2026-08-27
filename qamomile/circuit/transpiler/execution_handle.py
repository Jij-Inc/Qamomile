"""Represent local and remote quantum execution lifecycles."""

from __future__ import annotations

import asyncio
import dataclasses
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import Any, Generic, TypeVar

ResultT = TypeVar("ResultT")
MappedT = TypeVar("MappedT")


class JobStatus(Enum):
    """Describe a provider-independent execution state.

    The numeric values of the original four states remain stable for
    serialization compatibility.
    """

    PENDING = 1
    RUNNING = 2
    COMPLETED = 3
    FAILED = 4
    QUEUED = 5
    CANCELLING = 6
    PARTIAL = 7
    CANCELLED = 8
    UNKNOWN = 9


@dataclasses.dataclass(frozen=True)
class ExecutionReference:
    """Store secret-free identifiers needed to restore remote execution.

    Args:
        provider (str): Stable provider or adapter name.
        job_ids (tuple[str, ...]): One or more provider job identifiers.
        target (str | None): Provider target or device identifier. Defaults to
            ``None``.
        group_id (str | None): Session, batch, program, or parent identifier.
            Defaults to ``None``.
        context (Mapping[str, str]): Additional non-secret identifiers needed
            to restore the job. Defaults to an empty mapping.

    Raises:
        ValueError: If the provider name or any job identifier is empty.
        TypeError: If identifiers or context have incompatible types.
    """

    provider: str
    job_ids: tuple[str, ...]
    target: str | None = None
    group_id: str | None = None
    context: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identifiers and detach mutable context mappings.

        Raises:
            ValueError: If the provider or task identifiers are empty.
            TypeError: If identifiers or context have incompatible types.
        """
        if not isinstance(self.provider, str):
            raise TypeError("ExecutionReference.provider must be a string")
        if not self.provider:
            raise ValueError("ExecutionReference.provider must not be empty")
        if not isinstance(self.job_ids, (list, tuple)) or not all(
            isinstance(job_id, str) for job_id in self.job_ids
        ):
            raise TypeError("ExecutionReference.job_ids must be a string sequence")
        if not self.job_ids or any(not job_id for job_id in self.job_ids):
            raise ValueError("ExecutionReference.job_ids must not be empty")
        if self.target is not None and not isinstance(self.target, str):
            raise TypeError("ExecutionReference.target must be a string or None")
        if self.group_id is not None and not isinstance(self.group_id, str):
            raise TypeError("ExecutionReference.group_id must be a string or None")
        if not isinstance(self.context, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.context.items()
        ):
            raise TypeError("ExecutionReference.context must map strings to strings")
        object.__setattr__(self, "job_ids", tuple(self.job_ids))
        object.__setattr__(self, "context", dict(self.context))

    def to_dict(self) -> dict[str, Any]:
        """Convert the provider reference to JSON-compatible data.

        Returns:
            dict[str, Any]: Provider identifiers and decoding context without
                credentials or SDK objects.
        """
        return {
            "provider": self.provider,
            "job_ids": list(self.job_ids),
            "target": self.target,
            "group_id": self.group_id,
            "context": dict(self.context),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExecutionReference:
        """Reconstruct a provider reference from JSON-compatible data.

        Args:
            data (Mapping[str, Any]): Mapping produced by :meth:`to_dict`.

        Returns:
            ExecutionReference: Validated provider execution reference.

        Raises:
            KeyError: If a required provider or job identifier field is absent.
            TypeError: If a field has an incompatible type.
            ValueError: If provider or job identifiers are empty.
        """
        provider = data["provider"]
        job_ids = data["job_ids"]
        target = data.get("target")
        group_id = data.get("group_id")
        context = data.get("context", {})
        if not isinstance(provider, str):
            raise TypeError("ExecutionReference.provider must be a string")
        if not isinstance(job_ids, (list, tuple)) or not all(
            isinstance(job_id, str) for job_id in job_ids
        ):
            raise TypeError("ExecutionReference.job_ids must be a string sequence")
        if target is not None and not isinstance(target, str):
            raise TypeError("ExecutionReference.target must be a string or None")
        if group_id is not None and not isinstance(group_id, str):
            raise TypeError("ExecutionReference.group_id must be a string or None")
        if not isinstance(context, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in context.items()
        ):
            raise TypeError("ExecutionReference.context must map strings to strings")
        return cls(
            provider=provider,
            job_ids=tuple(job_ids),
            target=target,
            group_id=group_id,
            context=dict(context),
        )


class ExecutionHandle(ABC, Generic[ResultT]):
    """Expose a backend execution without forcing immediate result retrieval."""

    @abstractmethod
    def result(self, timeout: float | None = None) -> ResultT:
        """Wait for and return the backend-neutral raw result.

        Args:
            timeout (float | None): Maximum local wait in seconds. ``None``
                delegates the wait policy to the provider.

        Returns:
            ResultT: Raw result normalized by the backend executor.

        Raises:
            TimeoutError: If the local wait expires before completion.
        """
        raise NotImplementedError

    async def result_async(self, timeout: float | None = None) -> ResultT:
        """Wait asynchronously for the backend-neutral raw result.

        Args:
            timeout (float | None): Maximum local wait in seconds. Defaults to
                provider behavior when ``None``.

        Returns:
            ResultT: Raw result normalized by the backend executor.

        Raises:
            TimeoutError: If the local wait expires before completion.
        """
        return await asyncio.to_thread(self.result, timeout)

    @abstractmethod
    def status(self) -> JobStatus:
        """Return the current provider-independent execution status.

        Returns:
            JobStatus: Current normalized status.
        """
        raise NotImplementedError

    def raw_status(self) -> object:
        """Return provider-specific status information.

        Returns:
            object: Provider status value, or the normalized status when no
                richer value exists.
        """
        return self.status()

    def cancel(self) -> None:
        """Request best-effort cancellation.

        Cancellation is intentionally not reported as a boolean because
        providers may accept a request after execution has already started.
        Call :meth:`status` to observe the eventual state.
        """

    def references(self) -> tuple[ExecutionReference, ...]:
        """Return serializable remote execution references.

        Returns:
            tuple[ExecutionReference, ...]: Secret-free provider references.
        """
        return ()

    def metadata(self) -> Mapping[str, Any]:
        """Return optional provider execution metadata.

        Returns:
            Mapping[str, Any]: Provider metadata such as timestamps or usage.
        """
        return {}

    @property
    def native(self) -> object | None:
        """Return the wrapped provider-native task when available.

        Returns:
            object | None: Native task, batch, or future.
        """
        return None


class CompletedExecutionHandle(ExecutionHandle[ResultT]):
    """Wrap an already available result for synchronous executors.

    Args:
        value (ResultT): Completed execution value.
    """

    def __init__(self, value: ResultT) -> None:
        """Initialize an immediately completed execution.

        Args:
            value (ResultT): Completed execution value.
        """
        self._value = value

    def result(self, timeout: float | None = None) -> ResultT:
        """Return the completed value without waiting.

        Args:
            timeout (float | None): Ignored compatibility timeout.

        Returns:
            ResultT: Stored execution value.
        """
        return self._value

    def status(self) -> JobStatus:
        """Return the completed status.

        Returns:
            JobStatus: Always :attr:`JobStatus.COMPLETED`.
        """
        return JobStatus.COMPLETED


class MappedExecutionHandle(ExecutionHandle[MappedT], Generic[ResultT, MappedT]):
    """Lazily transform another execution handle's result.

    Args:
        source (ExecutionHandle[ResultT]): Underlying execution handle.
        transform (Callable[[ResultT], MappedT]): Result transformation.
    """

    def __init__(
        self,
        source: ExecutionHandle[ResultT],
        transform: Callable[[ResultT], MappedT],
    ) -> None:
        """Initialize a lazy mapped execution.

        Args:
            source (ExecutionHandle[ResultT]): Underlying execution handle.
            transform (Callable[[ResultT], MappedT]): Result transformation.
        """
        self._source = source
        self._transform = transform
        self._has_result = False
        self._result: MappedT | None = None

    def result(self, timeout: float | None = None) -> MappedT:
        """Retrieve and transform the source result once.

        Args:
            timeout (float | None): Maximum local wait in seconds.

        Returns:
            MappedT: Cached transformed result.

        Raises:
            Exception: Any source or transformation failure.
        """
        if self._has_result:
            return self._result  # type: ignore[return-value]
        self._result = self._transform(self._source.result(timeout))
        self._has_result = True
        return self._result

    async def result_async(self, timeout: float | None = None) -> MappedT:
        """Retrieve and transform the source result asynchronously.

        Args:
            timeout (float | None): Maximum local wait in seconds.

        Returns:
            MappedT: Cached transformed result.

        Raises:
            Exception: Any source or transformation failure.
        """
        if self._has_result:
            return self._result  # type: ignore[return-value]
        self._result = self._transform(await self._source.result_async(timeout))
        self._has_result = True
        return self._result

    def status(self) -> JobStatus:
        """Return the source execution status.

        Returns:
            JobStatus: Current mapped execution status.
        """
        return self._source.status()

    def raw_status(self) -> object:
        """Return the source provider status.

        Returns:
            object: Provider-specific source status.
        """
        return self._source.raw_status()

    def cancel(self) -> None:
        """Forward a cancellation request to the source execution."""
        self._source.cancel()

    def references(self) -> tuple[ExecutionReference, ...]:
        """Return source execution references.

        Returns:
            tuple[ExecutionReference, ...]: Source references.
        """
        return self._source.references()

    def metadata(self) -> Mapping[str, Any]:
        """Return source execution metadata.

        Returns:
            Mapping[str, Any]: Source metadata.
        """
        return self._source.metadata()

    @property
    def native(self) -> object | None:
        """Return the source provider-native task.

        Returns:
            object | None: Source native task.
        """
        return self._source.native


class CompositeExecutionHandle(ExecutionHandle[tuple[ResultT, ...]]):
    """Aggregate several independently submitted executions.

    Args:
        handles (Sequence[ExecutionHandle[ResultT]]): Child executions in
            stable result order.
    """

    def __init__(self, handles: Sequence[ExecutionHandle[ResultT]]) -> None:
        """Initialize an ordered execution aggregate.

        Args:
            handles (Sequence[ExecutionHandle[ResultT]]): Child executions.
        """
        self._handles = tuple(handles)
        self._result: tuple[ResultT, ...] | None = None

    def result(self, timeout: float | None = None) -> tuple[ResultT, ...]:
        """Return all child results in submission order.

        Args:
            timeout (float | None): Total local wait budget in seconds.

        Returns:
            tuple[ResultT, ...]: Ordered child results.

        Raises:
            TimeoutError: If the total wait budget expires.
            Exception: Any child execution failure.
        """
        if self._result is not None:
            return self._result
        deadline = None if timeout is None else time.monotonic() + timeout
        values = []
        for handle in self._handles:
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            values.append(handle.result(remaining))
        self._result = tuple(values)
        return self._result

    async def result_async(self, timeout: float | None = None) -> tuple[ResultT, ...]:
        """Return all child results asynchronously.

        Args:
            timeout (float | None): Total local wait budget in seconds.

        Returns:
            tuple[ResultT, ...]: Ordered child results.

        Raises:
            TimeoutError: If the total wait budget expires.
            Exception: Any child execution failure.
        """
        if self._result is not None:
            return self._result
        pending = asyncio.gather(
            *(handle.result_async(timeout) for handle in self._handles)
        )
        try:
            values = await asyncio.wait_for(pending, timeout=timeout)
        except asyncio.TimeoutError as error:
            raise TimeoutError("Composite execution result timed out") from error
        self._result = tuple(values)
        return self._result

    def status(self) -> JobStatus:
        """Aggregate child statuses without hiding partial completion.

        Returns:
            JobStatus: Aggregate execution status.
        """
        statuses = tuple(handle.status() for handle in self._handles)
        if not statuses or all(status is JobStatus.COMPLETED for status in statuses):
            return JobStatus.COMPLETED
        if any(status is JobStatus.RUNNING for status in statuses):
            return JobStatus.RUNNING
        if any(status is JobStatus.CANCELLING for status in statuses):
            return JobStatus.CANCELLING
        if any(status is JobStatus.QUEUED for status in statuses):
            return JobStatus.QUEUED
        if any(status is JobStatus.PENDING for status in statuses):
            return JobStatus.PENDING
        terminal = {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.PARTIAL,
        }
        if all(status in terminal for status in statuses):
            if all(status is JobStatus.FAILED for status in statuses):
                return JobStatus.FAILED
            if all(status is JobStatus.CANCELLED for status in statuses):
                return JobStatus.CANCELLED
            return JobStatus.PARTIAL
        return JobStatus.UNKNOWN

    def raw_status(self) -> object:
        """Return every child provider status.

        Returns:
            object: Tuple of child raw statuses.
        """
        return tuple(handle.raw_status() for handle in self._handles)

    def cancel(self) -> None:
        """Request cancellation of every unfinished child execution."""
        for handle in self._handles:
            if handle.status() not in {
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            }:
                handle.cancel()

    def references(self) -> tuple[ExecutionReference, ...]:
        """Flatten child execution references.

        Returns:
            tuple[ExecutionReference, ...]: Ordered child references.
        """
        return tuple(
            reference for handle in self._handles for reference in handle.references()
        )

    def metadata(self) -> Mapping[str, Any]:
        """Return metadata grouped by child index.

        Returns:
            Mapping[str, Any]: Child metadata sequence.
        """
        return {
            "children": tuple(handle.metadata() for handle in self._handles),
        }

    @property
    def native(self) -> object | None:
        """Return every child provider-native task.

        Returns:
            object | None: Tuple of native child tasks.
        """
        return tuple(handle.native for handle in self._handles)


__all__ = [
    "CompletedExecutionHandle",
    "CompositeExecutionHandle",
    "ExecutionHandle",
    "ExecutionReference",
    "JobStatus",
    "MappedExecutionHandle",
]
