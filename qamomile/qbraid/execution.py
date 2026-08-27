"""Adapt qBraid quantum jobs to Qamomile execution handles."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Any

from qamomile.circuit.transpiler.execution_handle import (
    ExecutionHandle,
    ExecutionReference,
    JobStatus,
)


class QBraidExecutionHandle(ExecutionHandle[dict[str, int]]):
    """Expose one qBraid sampling job through the shared lifecycle API.

    Args:
        job (Any): qBraid ``QuantumJob`` or compatible object.
        decoder (Callable[[Any], dict[str, int]]): Function converting the
            native result to normalized bitstring counts.
        target (str | None): qBraid device identifier. Defaults to ``None``.
        timeout (float | None): Default provider wait timeout in seconds.
            Defaults to ``None``.
        poll_interval (float): Provider polling interval in seconds.

    Raises:
        ValueError: If ``poll_interval`` is not positive.
    """

    def __init__(
        self,
        job: Any,
        decoder: Callable[[Any], dict[str, int]],
        *,
        target: str | None,
        timeout: float | None,
        poll_interval: float,
    ) -> None:
        """Initialize a lazy qBraid execution handle.

        Args:
            job (Any): Native qBraid job.
            decoder (Callable[[Any], dict[str, int]]): Native result decoder.
            target (str | None): qBraid device identifier.
            timeout (float | None): Default provider wait timeout in seconds.
            poll_interval (float): Positive provider polling interval.

        Raises:
            ValueError: If ``poll_interval`` is not positive.
        """
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self._job = job
        self._decoder = decoder
        self._target = target
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._result: dict[str, int] | None = None
        self._lock = threading.Lock()

    def result(self, timeout: float | None = None) -> dict[str, int]:
        """Wait for and decode the qBraid result once.

        Args:
            timeout (float | None): Maximum wait in seconds. ``None`` uses the
                executor-configured timeout.

        Returns:
            dict[str, int]: Normalized big-endian bitstring counts.

        Raises:
            TimeoutError: If the provider does not complete within the timeout.
            Exception: Any qBraid wait, retrieval, or decoding failure.
        """
        if self._result is not None:
            return dict(self._result)
        with self._lock:
            if self._result is None:
                wait_timeout = self._timeout if timeout is None else timeout
                self._job.wait_for_final_state(
                    timeout=wait_timeout,
                    poll_interval=self._poll_interval,
                )
                self._result = self._decoder(self._job.result())
        return dict(self._result)

    def status(self) -> JobStatus:
        """Return the normalized qBraid job status.

        Returns:
            JobStatus: Current provider-independent state.
        """
        return _map_qbraid_status(self.raw_status())

    def raw_status(self) -> object:
        """Return the native qBraid status value.

        Returns:
            object: Provider status enum or string.
        """
        status = getattr(self._job, "status", None)
        return status() if callable(status) else status

    def cancel(self) -> None:
        """Request best-effort cancellation from qBraid."""
        cancel = getattr(self._job, "cancel", None)
        if callable(cancel):
            cancel()

    def references(self) -> tuple[ExecutionReference, ...]:
        """Return the qBraid job identifier when the SDK exposes one.

        Returns:
            tuple[ExecutionReference, ...]: One qBraid reference, or an empty
                tuple for jobs without a stable identifier.
        """
        value = getattr(self._job, "id", None)
        job_id = value() if callable(value) else value
        if not isinstance(job_id, str) or not job_id:
            return ()
        return (
            ExecutionReference(
                provider="qbraid",
                job_ids=(job_id,),
                target=self._target,
                context={"kind": "sample"},
            ),
        )

    def metadata(self) -> Mapping[str, Any]:
        """Return native qBraid job metadata when available.

        Returns:
            Mapping[str, Any]: Provider metadata, or an empty mapping.
        """
        metadata = getattr(self._job, "metadata", None)
        value = metadata() if callable(metadata) else metadata
        return value if isinstance(value, Mapping) else {}

    @property
    def native(self) -> object | None:
        """Return the wrapped qBraid job.

        Returns:
            object | None: Native qBraid job.
        """
        return self._job


def _map_qbraid_status(value: object) -> JobStatus:
    """Map qBraid and provider-specific status values to Qamomile states.

    Args:
        value (object): Native enum, string, or ``None`` status.

    Returns:
        JobStatus: Provider-independent execution state.
    """
    name = getattr(value, "name", value)
    normalized = str(name).upper() if name is not None else ""
    aliases = {
        "CREATED": JobStatus.PENDING,
        "INITIALIZING": JobStatus.PENDING,
        "PENDING": JobStatus.PENDING,
        "SUBMITTED": JobStatus.PENDING,
        "QUEUED": JobStatus.QUEUED,
        "WAITING": JobStatus.QUEUED,
        "RUNNING": JobStatus.RUNNING,
        "CANCELLING": JobStatus.CANCELLING,
        "CANCELED": JobStatus.CANCELLED,
        "CANCELLED": JobStatus.CANCELLED,
        "COMPLETED": JobStatus.COMPLETED,
        "DONE": JobStatus.COMPLETED,
        "SUCCESS": JobStatus.COMPLETED,
        "ERROR": JobStatus.FAILED,
        "FAILED": JobStatus.FAILED,
    }
    return aliases.get(normalized, JobStatus.UNKNOWN)


__all__ = ["QBraidExecutionHandle"]
