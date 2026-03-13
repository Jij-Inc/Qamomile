"""Job classes for quantum execution results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class JobStatus(Enum):
    """Status of a quantum job."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


class Job(ABC, Generic[T]):
    """Abstract base class for quantum execution jobs.

    A Job represents a quantum execution that can be awaited for results.
    """

    @abstractmethod
    def result(self) -> T:
        """Wait for and return the result.

        Blocks until the job completes.

        Returns:
            The execution result with the appropriate type.

        Raises:
            ExecutionError: If the job failed.
        """
        pass

    @abstractmethod
    def status(self) -> JobStatus:
        """Return the current job status."""
        pass


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
        raw_counts: dict[str, int],
        result_converter: Callable[[dict[str, int]], list[tuple[T, int]]],
        shots: int,
    ):
        """Initialize sample job.

        Args:
            raw_counts: Bitstring counts from executor (e.g., {"00": 512, "11": 512})
            result_converter: Function to convert raw counts to typed results
            shots: Number of shots executed
        """
        self._raw_counts = raw_counts
        self._result_converter = result_converter
        self._shots = shots
        self._status = JobStatus.COMPLETED
        self._result: SampleResult[T] | None = None

    def result(self) -> SampleResult[T]:
        """Return the sample result."""
        if self._result is not None:
            return self._result

        # Convert to typed results (returns list[tuple[T, int]])
        typed_results = self._result_converter(self._raw_counts)

        self._result = SampleResult(results=typed_results, shots=self._shots)
        return self._result

    def status(self) -> JobStatus:
        """Return job status."""
        return self._status


class RunJob(Job[T], Generic[T]):
    """Job for single execution.

    Returns a single result value matching the kernel's return type.
    """

    def __init__(
        self,
        raw_counts: dict[str, int],
        result_converter: Callable[[str], T],
    ):
        """Initialize run job.

        Args:
            raw_counts: Bitstring counts from executor (should have single entry)
            result_converter: Function to convert bitstring to typed result
        """
        self._raw_counts = raw_counts
        self._result_converter = result_converter
        self._status = JobStatus.COMPLETED
        self._result: T | None = None

    def result(self) -> T:
        """Return the single result."""
        if self._result is not None:
            return self._result

        # Get the single result (should be only one with count=1)
        if self._raw_counts:
            bitstring = next(iter(self._raw_counts.keys()))
            self._result = self._result_converter(bitstring)
        else:
            raise RuntimeError("No results from execution")

        return self._result

    def status(self) -> JobStatus:
        """Return job status."""
        return self._status


class ExpvalJob(Job[float]):
    """Job for expectation value computation.

    Returns a single float representing <psi|H|psi>.
    """

    def __init__(self, exp_val: float):
        """Initialize expval job.

        Args:
            exp_val: The computed expectation value
        """
        self._exp_val = exp_val
        self._status = JobStatus.COMPLETED

    def result(self) -> float:
        """Return the expectation value."""
        return self._exp_val

    def status(self) -> JobStatus:
        """Return job status."""
        return self._status
