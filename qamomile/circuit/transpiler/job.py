"""Job classes for quantum execution results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Generic, TypeVar, cast

import numpy as np

from qamomile.circuit.transpiler.execution_handle import (
    CompletedExecutionHandle,
    ExecutionHandle,
    ExecutionReference,
    JobStatus,
)

T = TypeVar("T")


class JobKind(StrEnum):
    """Identify the public operation needed to reconstruct a typed job."""

    SAMPLE = "sample"
    RUN = "run"


@dataclass(frozen=True)
class JobSnapshot:
    """Store the public operation and provider references for restoration.

    Runtime bindings are intentionally excluded. They can contain arbitrary
    application data, so callers supply them again to
    :meth:`ExecutableProgram.restore` instead of persisting them implicitly.

    Args:
        kind (JobKind): Public operation that created the job.
        executions (tuple[ExecutionReference, ...]): Ordered provider
            references needed to reconstruct the raw execution.
        shots (int | None): Sampling shot count. Required for sample jobs and
            absent for run jobs.

    Raises:
        ValueError: If references are empty or the shot count is inconsistent
            with the operation kind.
        TypeError: If operation kind, references, or shots have incompatible
            types.
    """

    kind: JobKind
    executions: tuple[ExecutionReference, ...]
    shots: int | None = None

    def __post_init__(self) -> None:
        """Validate the operation-specific restoration metadata.

        Raises:
            ValueError: If references are empty, sample shots are not positive,
                or a run snapshot contains a shot count.
            TypeError: If operation kind, references, or shots have
                incompatible types.
        """
        if not isinstance(self.kind, JobKind):
            raise TypeError("JobSnapshot.kind must be a JobKind")
        if not isinstance(self.executions, (list, tuple)) or not all(
            isinstance(reference, ExecutionReference) for reference in self.executions
        ):
            raise TypeError(
                "JobSnapshot.executions must contain ExecutionReference values"
            )
        if not self.executions:
            raise ValueError("JobSnapshot.executions must not be empty")
        if self.kind is JobKind.SAMPLE:
            if self.shots is not None and not isinstance(self.shots, int):
                raise TypeError("Sample JobSnapshot.shots must be an integer")
            if self.shots is None or isinstance(self.shots, bool) or self.shots <= 0:
                raise ValueError("Sample JobSnapshot.shots must be positive")
            object.__setattr__(self, "shots", int(self.shots))
        elif self.shots is not None:
            raise ValueError("Run JobSnapshot must not contain shots")
        object.__setattr__(self, "executions", tuple(self.executions))

    def to_dict(self) -> dict[str, Any]:
        """Convert the snapshot to JSON-compatible data.

        Returns:
            dict[str, Any]: Operation metadata and serialized provider
                references without runtime bindings.
        """
        return {
            "kind": self.kind.value,
            "executions": [reference.to_dict() for reference in self.executions],
            "shots": self.shots,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JobSnapshot:
        """Reconstruct a validated snapshot from JSON-compatible data.

        Args:
            data (Mapping[str, Any]): Mapping produced by :meth:`to_dict`.

        Returns:
            JobSnapshot: Validated typed-job restoration snapshot.

        Raises:
            KeyError: If operation kind or execution references are absent.
            TypeError: If execution references are not a sequence of mappings.
            ValueError: If operation metadata or references are invalid.
        """
        raw_executions = data["executions"]
        if not isinstance(raw_executions, (list, tuple)) or not all(
            isinstance(reference, Mapping) for reference in raw_executions
        ):
            raise TypeError("JobSnapshot.executions must be a mapping sequence")
        return cls(
            kind=JobKind(data["kind"]),
            executions=tuple(
                ExecutionReference.from_dict(reference) for reference in raw_executions
            ),
            shots=data.get("shots"),
        )


def _typed_result_key(value: Any) -> Hashable:
    """Build an exact hashable key for one public sample value.

    Args:
        value (Any): Converted public result value.

    Returns:
        Hashable: Structure-, type-, dtype-, and shape-preserving key.
            Unknown unhashable objects use identity because Qamomile's public
            result contract does not define equality for arbitrary objects.
    """
    if isinstance(value, tuple):
        return ("tuple", tuple(_typed_result_key(element) for element in value))
    if isinstance(value, list):
        return ("list", tuple(_typed_result_key(element) for element in value))
    if isinstance(value, dict):
        return (
            "dict",
            frozenset(
                (_typed_result_key(key), _typed_result_key(entry_value))
                for key, entry_value in value.items()
            ),
        )
    if isinstance(value, np.ndarray):
        return (
            "ndarray",
            value.dtype.str,
            tuple(value.shape),
            tuple(_typed_result_key(element) for element in value.reshape(-1).tolist()),
        )
    if isinstance(value, np.generic):
        return ("np_scalar", value.dtype.str, _typed_result_key(value.item()))
    if isinstance(value, Hashable):
        return ("scalar", type(value), value)
    return ("identity", id(value))


def _aggregate_typed_results(
    results: list[tuple[T, int]],
) -> list[tuple[T, int]]:
    """Combine counts whose converted public result values are equal.

    Backend raw bitstrings can differ only on qubits that are not part of the
    program output. After result conversion those rows represent the same
    public value and must appear as one ``SampleResult`` entry.

    Args:
        results (list[tuple[T, int]]): Converted result values and counts.

    Returns:
        list[tuple[T, int]]: Stable first-seen values with duplicate counts
            summed.
    """
    aggregated: list[tuple[T, int]] = []
    positions: dict[Hashable, int] = {}
    for value, count in results:
        key = _typed_result_key(value)
        position = positions.get(key)
        if position is None:
            positions[key] = len(aggregated)
            aggregated.append((value, count))
            continue
        existing, existing_count = aggregated[position]
        aggregated[position] = (existing, existing_count + count)
    return aggregated


class Job(ABC, Generic[T]):
    """Abstract base class for quantum execution jobs.

    A Job represents a quantum execution that can be awaited for results.
    """

    def __init__(
        self,
        handle: ExecutionHandle[Any],
        kind: JobKind,
        shots: int | None = None,
    ) -> None:
        """Initialize a public job around an execution handle.

        Args:
            handle (ExecutionHandle[Any]): Raw or mapped backend execution.
            kind (JobKind): Public operation represented by the job.
            shots (int | None): Sampling shot count. Defaults to ``None`` for
                run jobs.
        """
        self._handle = handle
        self._kind = kind
        self._snapshot_shots = shots

    @abstractmethod
    def result(self, timeout: float | None = None) -> T:
        """Wait for and return the result.

        Blocks until the job completes.

        Args:
            timeout (float | None): Maximum local wait in seconds. ``None``
                uses provider behavior.

        Returns:
            T: Execution result with the appropriate public type.

        Raises:
            ExecutionError: If the job failed.
        """
        raise NotImplementedError

    async def result_async(self, timeout: float | None = None) -> T:
        """Wait asynchronously for and return the public result.

        Args:
            timeout (float | None): Maximum local wait in seconds. Defaults to
                provider behavior when ``None``.

        Returns:
            T: Execution result with the appropriate public type.
        """
        import asyncio

        return await asyncio.to_thread(self.result, timeout)

    def status(self) -> JobStatus:
        """Return the current job status.

        Returns:
            JobStatus: Current normalized status.
        """
        return self._handle.status()

    def raw_status(self) -> object:
        """Return provider-specific status information.

        Returns:
            object: Provider status or aggregate status values.
        """
        return self._handle.raw_status()

    def cancel(self) -> None:
        """Request best-effort cancellation of the underlying execution."""
        self._handle.cancel()

    def references(self) -> tuple[ExecutionReference, ...]:
        """Return secret-free references for later job restoration.

        Returns:
            tuple[ExecutionReference, ...]: Provider execution references.
        """
        return self._handle.references()

    def snapshot(self) -> JobSnapshot:
        """Capture secret-free information needed for typed restoration.

        Returns:
            JobSnapshot: Public operation metadata and provider references.

        Raises:
            ValueError: If the underlying execution exposes no restorable
                references.
        """
        references = self.references()
        if not references:
            raise ValueError("This job does not expose restorable references")
        # Bindings stay caller-owned because they may contain application data
        # unrelated to the provider job identity.
        return JobSnapshot(self._kind, references, self._snapshot_shots)

    def metadata(self) -> Mapping[str, Any]:
        """Return provider execution metadata.

        Returns:
            Mapping[str, Any]: Provider-specific metadata.
        """
        return self._handle.metadata()

    @property
    def native(self) -> object | None:
        """Return the wrapped provider-native task when available.

        Returns:
            object | None: Native task, batch, or future.
        """
        return self._handle.native


@dataclass
class SampleResult(Generic[T]):
    """Result of a sample() execution.

    Contains results as a list of (value, count) tuples.

    Example:
        result.results  # [(0.25, 500), (0.75, 500)]
    """

    results: list[tuple[T, int]]
    """List of (value, count) tuples."""

    shots: int
    """Total number of shots executed."""

    def most_common(self, n: int = 1) -> list[tuple[T, int]]:
        """Return the n most common results.

        Args:
            n: Number of results to return.

        Returns:
            List of (result, count) tuples sorted by count descending.
        """
        sorted_items = sorted(self.results, key=lambda x: x[1], reverse=True)
        return sorted_items[:n]

    def probabilities(self) -> list[tuple[T, float]]:
        """Return probability distribution over results.

        Returns:
            List of (value, probability) tuples.
        """
        return [(v, c / self.shots) for v, c in self.results]


class SampleJob(Job[SampleResult[T]], Generic[T]):
    """Job for sampling execution (multiple shots).

    Returns a SampleResult containing counts for each unique result.
    """

    def __init__(
        self,
        raw_counts: dict[str, int] | ExecutionHandle[dict[str, int]],
        result_converter: Callable[[dict[str, int]], list[tuple[T, int]]],
        shots: int,
    ) -> None:
        """Initialize sample job.

        Args:
            raw_counts (dict[str, int] | ExecutionHandle[dict[str, int]]):
                Counts or a deferred counts execution.
            result_converter (Callable[[dict[str, int]], list[tuple[T, int]]]):
                Function converting raw counts to typed values.
            shots (int): Number of requested shots.
        """
        handle = (
            raw_counts
            if isinstance(raw_counts, ExecutionHandle)
            else CompletedExecutionHandle(raw_counts)
        )
        super().__init__(handle, JobKind.SAMPLE, shots)
        self._result_converter = result_converter
        self._shots = shots
        self._result: SampleResult[T] | None = None

    def _convert(self, raw_counts: dict[str, int]) -> SampleResult[T]:
        """Convert raw counts and cache the public sample result.

        Args:
            raw_counts (dict[str, int]): Backend-normalized bitstring counts.

        Returns:
            SampleResult[T]: Aggregated typed sample result.
        """
        if self._result is None:
            typed_results = _aggregate_typed_results(self._result_converter(raw_counts))
            self._result = SampleResult(results=typed_results, shots=self._shots)
        return self._result

    def result(self, timeout: float | None = None) -> SampleResult[T]:
        """Wait for and return the typed sample result.

        Args:
            timeout (float | None): Maximum local wait in seconds.

        Returns:
            SampleResult[T]: Aggregated typed result.
        """
        if self._result is not None:
            return self._result
        return self._convert(self._handle.result(timeout))

    async def result_async(self, timeout: float | None = None) -> SampleResult[T]:
        """Wait asynchronously for the typed sample result.

        Args:
            timeout (float | None): Maximum local wait in seconds.

        Returns:
            SampleResult[T]: Aggregated typed result.
        """
        if self._result is not None:
            return self._result
        return self._convert(await self._handle.result_async(timeout))


class RunJob(Job[T], Generic[T]):
    """Job for single execution.

    Returns a single result value matching the kernel's return type.
    """

    def __init__(
        self,
        raw_counts: dict[str, int] | ExecutionHandle[dict[str, int]] | None,
        result_converter: Callable[[str], T] | None,
        *,
        value_handle: ExecutionHandle[T] | None = None,
    ) -> None:
        """Initialize run job.

        Args:
            raw_counts (dict[str, int] | ExecutionHandle[dict[str, int]] | None):
                Counts or deferred counts. May be ``None`` with
                ``value_handle``.
            result_converter (Callable[[str], T] | None): Function converting
                one bitstring. May be ``None`` with ``value_handle``.
            value_handle (ExecutionHandle[T] | None): Handle already producing
                the final public value. Defaults to ``None``.

        Raises:
            ValueError: If neither a valid counts source nor ``value_handle``
                is supplied.
        """
        if value_handle is not None:
            handle: ExecutionHandle[Any] = value_handle
        elif raw_counts is not None and result_converter is not None:
            handle = (
                raw_counts
                if isinstance(raw_counts, ExecutionHandle)
                else CompletedExecutionHandle(raw_counts)
            )
        else:
            raise ValueError(
                "RunJob requires counts and a converter, or a value_handle"
            )
        super().__init__(handle, JobKind.RUN)
        self._result_converter = result_converter
        self._value_handle = value_handle
        self._result: T | None = None

    @classmethod
    def from_handle(cls, handle: ExecutionHandle[T]) -> RunJob[T]:
        """Create a run job whose handle already returns the public value.

        Args:
            handle (ExecutionHandle[T]): Final-value execution handle.

        Returns:
            RunJob[T]: Public run job delegating to ``handle``.
        """
        return cls(None, None, value_handle=handle)

    def _convert_counts(self, raw_counts: dict[str, int]) -> T:
        """Convert the first sampled bitstring to the public run value.

        Args:
            raw_counts (dict[str, int]): Single-shot backend counts.

        Returns:
            T: Public kernel return value.

        Raises:
            RuntimeError: If the backend returned no counts.
        """
        if not raw_counts:
            raise RuntimeError("No results from execution")
        converter = cast(Callable[[str], T], self._result_converter)
        return converter(next(iter(raw_counts)))

    def result(self, timeout: float | None = None) -> T:
        """Wait for and return the single public result.

        Args:
            timeout (float | None): Maximum local wait in seconds.

        Returns:
            T: Public kernel return value.
        """
        if self._result is not None:
            return self._result
        raw_result = self._handle.result(timeout)
        if self._value_handle is not None:
            self._result = cast(T, raw_result)
        else:
            self._result = self._convert_counts(cast(dict[str, int], raw_result))
        return self._result

    async def result_async(self, timeout: float | None = None) -> T:
        """Wait asynchronously for the single public result.

        Args:
            timeout (float | None): Maximum local wait in seconds.

        Returns:
            T: Public kernel return value.
        """
        if self._result is not None:
            return self._result
        raw_result = await self._handle.result_async(timeout)
        if self._value_handle is not None:
            self._result = cast(T, raw_result)
        else:
            self._result = self._convert_counts(cast(dict[str, int], raw_result))
        return self._result


class ExpvalJob(Job[float]):
    """Job for expectation value computation.

    Returns a single float representing <psi|H|psi>.
    """

    def __init__(self, exp_val: float | ExecutionHandle[float]) -> None:
        """Initialize expval job.

        Args:
            exp_val (float | ExecutionHandle[float]): Completed value or
                deferred expectation execution.
        """
        handle = (
            exp_val
            if isinstance(exp_val, ExecutionHandle)
            else CompletedExecutionHandle(float(exp_val))
        )
        super().__init__(handle, JobKind.RUN)
        self._exp_val: float | None = None

    def result(self, timeout: float | None = None) -> float:
        """Wait for and return the expectation value.

        Args:
            timeout (float | None): Maximum local wait in seconds.

        Returns:
            float: Expectation value.
        """
        if self._exp_val is None:
            self._exp_val = float(self._handle.result(timeout))
        return self._exp_val

    async def result_async(self, timeout: float | None = None) -> float:
        """Wait asynchronously for and return the expectation value.

        Args:
            timeout (float | None): Maximum local wait in seconds.

        Returns:
            float: Expectation value.
        """
        if self._exp_val is None:
            self._exp_val = float(await self._handle.result_async(timeout))
        return self._exp_val


__all__ = [
    "ExpvalJob",
    "Job",
    "JobKind",
    "JobSnapshot",
    "JobStatus",
    "RunJob",
    "SampleJob",
    "SampleResult",
]
