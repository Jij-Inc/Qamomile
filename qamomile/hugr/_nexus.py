"""Submit instrumented HUGR programs to Quantinuum Nexus."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import math
import threading
import time
from collections.abc import Mapping
from enum import StrEnum
from numbers import Integral
from typing import Any
from uuid import uuid4

from qamomile.circuit.transpiler import (
    ExecutionError,
    ExecutionHandle,
    ExecutionReference,
    JobStatus,
)

_NEXUS_STATUS_MAP: Mapping[str, JobStatus] = {
    "SUBMITTED": JobStatus.PENDING,
    "RETRYING": JobStatus.PENDING,
    "QUEUED": JobStatus.QUEUED,
    "RUNNING": JobStatus.RUNNING,
    "COMPLETED": JobStatus.COMPLETED,
    "CANCELLING": JobStatus.CANCELLING,
    "CANCELLED": JobStatus.CANCELLED,
    "ERROR": JobStatus.FAILED,
    "TERMINATED": JobStatus.FAILED,
    "DEPLETED": JobStatus.FAILED,
}


class NexusRegion(StrEnum):
    """Select a supported Nexus execution region."""

    US = "us"
    SG = "sg"


@dataclasses.dataclass(frozen=True)
class NexusExecutionOptions:
    """Configure HUGR execution on a Helios target through Nexus.

    Args:
        project (Any | None): Native ``qnexus`` project reference. ``None``
            uses the active Nexus project.
        backend_config (Any | None): Native ``qnexus.HeliosConfig`` with
            optional emulator or compiler settings. When supplied, its
            ``system_name`` takes precedence over ``system_name`` below.
        system_name (str): Helios device name, defaulting to ``Helios-1``.
        name (str): Program and job name prefix. Each submission receives a
            unique suffix, including separate expectation measurement jobs.
        max_cost (float | None): Optional maximum HQC cost for each submitted
            program. This is not an aggregate expectation-estimation budget.
        n_qubits (int | None): Optional maximum qubit count passed to Nexus.
        credential_name (str | None): Name of a credential already in Nexus.
        user_group (str | None): Nexus user group for scheduling.
        target_region (str | NexusRegion | None): Execution region, ``us`` or ``sg``.
        poll_interval_seconds (float): Positive local status polling interval.
        timeout_seconds (float | None): Default local status-wait deadline.
            ``None`` waits indefinitely. Timing out never cancels a job.

    Raises:
        ValueError: If a numeric option, name, region, or config is invalid.
    """

    project: Any | None = None
    backend_config: Any | None = None
    system_name: str = "Helios-1"
    name: str = "qamomile-hugr"
    max_cost: float | None = None
    n_qubits: int | None = None
    credential_name: str | None = None
    user_group: str | None = None
    target_region: str | NexusRegion | None = None
    poll_interval_seconds: float = 1.0
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        """Validate settings before any provider operation.

        Raises:
            ValueError: If a numeric option, name, region, or config is invalid.
        """
        for name in ("system_name", "name"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        _validate_seconds(self.poll_interval_seconds, "poll_interval_seconds", False)
        if self.timeout_seconds is not None:
            _validate_seconds(self.timeout_seconds, "timeout_seconds", True)
        if self.max_cost is not None:
            _validate_seconds(self.max_cost, "max_cost", True)
        if self.n_qubits is not None:
            _validate_positive_integer(self.n_qubits, "n_qubits")
            object.__setattr__(self, "n_qubits", int(self.n_qubits))
        if self.target_region is not None:
            try:
                object.__setattr__(
                    self, "target_region", NexusRegion(self.target_region)
                )
            except ValueError as error:
                raise ValueError("target_region must be 'us', 'sg', or None") from error
        if self.backend_config is not None and (
            getattr(self.backend_config, "type", None) != "HeliosConfig"
            or not getattr(self.backend_config, "system_name", None)
        ):
            raise ValueError("backend_config must be a qnexus.HeliosConfig")


def _validate_seconds(value: float, name: str, allow_zero: bool) -> None:
    """Validate a finite numeric setting.

    Args:
        value (float): Numeric value to validate.
        name (str): Argument name used in the error.
        allow_zero (bool): Whether zero is accepted.

    Raises:
        ValueError: If the value is nonnumeric, nonfinite, or out of range.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (value < 0 if allow_zero else value <= 0)
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} finite number")


def _validate_positive_integer(value: Any, name: str) -> None:
    """Validate a positive count without accepting booleans.

    Args:
        value (Any): Count to validate.
        name (str): Argument name used in the error.

    Raises:
        ValueError: If the count is not a positive integer.
    """
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _status_name(status: Any) -> str:
    """Read the enum value from a Nexus status record.

    Args:
        status (Any): Native ``qnexus.models.job_status.JobStatus`` record.

    Returns:
        str: Provider status name, preserving unknown future names.
    """
    value = getattr(status, "status", status)
    return str(getattr(value, "value", value))


def _map_status(status: Any) -> JobStatus:
    """Map the documented Nexus states to shared execution states.

    Args:
        status (Any): Native Nexus status record.

    Returns:
        JobStatus: Normalized state, or ``UNKNOWN`` for a future provider state.
    """
    return _NEXUS_STATUS_MAP.get(_status_name(status), JobStatus.UNKNOWN)


def _decode_results(raw: Any, shots: int) -> list[dict[str, Any]]:
    """Preserve typed tagged outputs without overwriting repeated records.

    Args:
        raw (Any): Native HUGR ``QsysResult`` returned by Nexus.
        shots (int): Required number of completed shots.

    Returns:
        list[dict[str, Any]]: One tag-to-value mapping per shot.

    Raises:
        ExecutionError: If results have an incompatible format, repeated tags,
            or a shot count different from the submitted request.
    """
    native_shots = getattr(raw, "results", None)
    if not isinstance(native_shots, list):
        raise ExecutionError("Nexus did not return a HUGR QsysResult")
    if len(native_shots) != shots:
        raise ExecutionError(
            f"Nexus returned {len(native_shots)} shots; expected {shots}"
        )
    records = []
    for shot in native_shots:
        if not callable(getattr(shot, "collate_tags", None)):
            raise ExecutionError("Nexus returned an invalid HUGR shot record")
        collated = shot.collate_tags()
        record = {}
        for tag, values in collated.items():
            if not isinstance(tag, str) or len(values) != 1:
                raise ExecutionError(
                    f"Nexus output tag {tag!r} must have exactly one value per shot"
                )
            if tag.startswith("USER:"):
                parts = tag.split(":", 2)
                if len(parts) != 3 or parts[1] not in ("BOOL", "FLOAT", "UINT"):
                    raise ExecutionError(
                        f"Nexus returned an unsupported output tag: {tag}"
                    )
                tag = parts[2]
            if tag in record:
                raise ExecutionError(f"Nexus output tag {tag!r} is ambiguous")
            record[tag] = values[0]
        records.append(record)
    return records


class NexusExecutionHandle(ExecutionHandle[list[dict[str, Any]]]):
    """Adapt one Nexus execute job without resubmitting unsuccessful work.

    Args:
        client (Any): Imported ``qnexus`` module or a compatible test client.
        native (Any): Native Nexus execute-job reference.
        reference (ExecutionReference): Secret-free job restoration reference.
        options (NexusExecutionOptions): Local polling and timeout policy.
    """

    def __init__(
        self,
        client: Any,
        native: Any,
        reference: ExecutionReference,
        options: NexusExecutionOptions,
    ) -> None:
        """Store a submitted or restored execution.

        Args:
            client (Any): Nexus SDK client module.
            native (Any): Native execute-job reference.
            reference (ExecutionReference): Validated reference including shots.
            options (NexusExecutionOptions): Local waiting policy.
        """
        self._client = client
        self._native = native
        self._reference = reference
        self._options = options
        self._result: list[dict[str, Any]] | None = None
        self._result_error: ExecutionError | None = None
        self._lock = threading.Lock()
        self._last_status: Any = getattr(native, "last_status_detail", None)

    def result(self, timeout: float | None = None) -> list[dict[str, Any]]:
        """Wait for complete results, leaving timed-out jobs active remotely.

        Args:
            timeout (float | None): Non-negative local wait limit in seconds.
                ``None`` uses the options default. Network request duration is
                controlled separately by the Nexus SDK's HTTP configuration.

        Returns:
            list[dict[str, Any]]: Typed output records in shot order.

        Raises:
            ValueError: If the timeout is invalid.
            TimeoutError: If the lock or status wait exceeds the deadline.
            ExecutionError: If the remote job fails, is cancelled, or returns
                malformed or incomplete results.
            Exception: If the Nexus SDK cannot retrieve status or results.
        """
        effective = self._options.timeout_seconds if timeout is None else timeout
        if effective is not None:
            _validate_seconds(effective, "timeout", True)
        deadline = None if effective is None else time.monotonic() + effective
        acquired = (
            self._lock.acquire()
            if effective is None
            else self._lock.acquire(timeout=float(effective))
        )
        if not acquired:
            raise TimeoutError("Nexus execution result timed out")
        try:
            if self._result is not None:
                return self._result
            if self._result_error is not None:
                raise self._result_error
            while True:
                status = self.status()
                if status == JobStatus.COMPLETED:
                    break
                if status in (JobStatus.FAILED, JobStatus.CANCELLED):
                    self._result_error = ExecutionError(
                        "Nexus job did not complete: " + _status_name(self._last_status)
                    )
                    raise self._result_error
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("Nexus execution result timed out")
                time.sleep(
                    self._options.poll_interval_seconds
                    if remaining is None
                    else min(self._options.poll_interval_seconds, remaining)
                )
            refs = self._client.jobs.results(self._native, allow_incomplete=False)
            if len(refs) != 1:
                self._result_error = ExecutionError(
                    "Nexus must return exactly one result for one submitted program"
                )
                raise self._result_error
            raw = refs[0].download_result()
            try:
                self._result = _decode_results(
                    raw, int(self._reference.context["shots"])
                )
            except ExecutionError as exc:
                self._result_error = exc
                raise
            return self._result
        finally:
            self._lock.release()

    def status(self) -> JobStatus:
        """Return the latest normalized provider status.

        Returns:
            JobStatus: Current provider state or a cached terminal result state.

        Raises:
            Exception: If the Nexus SDK cannot retrieve the job status.
        """
        if self._result_error is not None:
            if _map_status(self._last_status) == JobStatus.CANCELLED:
                return JobStatus.CANCELLED
            return JobStatus.FAILED
        if self._result is not None:
            return JobStatus.COMPLETED
        return _map_status(self.raw_status())

    def raw_status(self) -> object:
        """Fetch provider status details without changing the remote job.

        Returns:
            object: Native Nexus status record with queue, time, and cost data.

        Raises:
            Exception: If the Nexus SDK cannot retrieve the job status.
        """
        self._last_status = self._client.jobs.status(self._native)
        return self._last_status

    def cancel(self) -> None:
        """Request cancellation and let later status calls observe the outcome.

        Raises:
            Exception: If the Nexus SDK cannot request job cancellation.
        """
        self._client.jobs.cancel(self._native)

    def references(self) -> tuple[ExecutionReference, ...]:
        """Return the identifiers required to restore this job.

        Returns:
            tuple[ExecutionReference, ...]: One secret-free Nexus reference.
        """
        return (self._reference,)

    def metadata(self) -> Mapping[str, Any]:
        """Return non-secret metadata from the most recently observed status.

        Returns:
            Mapping[str, Any]: Provider, job, target, shots, queue, and cost data.
        """
        metadata: dict[str, Any] = {
            "provider": self._reference.provider,
            "job_id": self._reference.job_ids[0],
            "target": self._reference.target,
            "shots": int(self._reference.context["shots"]),
        }
        for key in ("queue_position", "cost"):
            value = getattr(self._last_status, key, None)
            if value is not None:
                metadata[key] = value
        return metadata

    @property
    def native(self) -> object:
        """Expose the native Nexus execute-job reference.

        Returns:
            object: Submitted or restored ``ExecuteJobRef``.
        """
        return self._native


class NexusTransport:
    """Submit closed HUGR entry points through the optional Nexus SDK.

    The caller supplies an execution wrapper with runtime values connected to
    the unchanged parameterized body and with explicit output records. Nexus
    0.49 has no separate HUGR entry-point argument field.

    Args:
        options (NexusExecutionOptions | None): Submission and polling settings.
        client (Any | None): Optional SDK-compatible client for contract tests.

    Raises:
        ImportError: If the optional Nexus SDK is unavailable.
    """

    def __init__(
        self,
        options: NexusExecutionOptions | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        """Load the SDK without authenticating or submitting any remote work.

        Args:
            options (NexusExecutionOptions | None): Submission configuration.
            client (Any | None): Injected SDK-compatible client for testing.

        Raises:
            ImportError: If ``qnexus`` is unavailable.
        """
        self.options = options or NexusExecutionOptions()
        if client is None:
            try:
                client = importlib.import_module("qnexus")
            except ImportError as exc:
                raise ImportError(
                    "Helios execution requires qnexus; install qamomile[hugr]"
                ) from exc
        self._client = client

    def submit(self, package: Any, shots: int) -> NexusExecutionHandle:
        """Upload one HUGR package and submit exactly one execution request.

        Args:
            package (Any): Instrumented HUGR package with a closed entry point.
            shots (int): Positive number of shots to execute.

        Returns:
            NexusExecutionHandle: Handle returned without waiting for results.

        Raises:
            ValueError: If the number of shots is invalid.
            Exception: If Nexus rejects upload, compilation, or submission.
        """
        _validate_positive_integer(shots, "shots")
        fingerprint = hashlib.sha256(package.to_bytes()).hexdigest()
        options = self.options
        config = options.backend_config
        if config is None:
            config = self._client.HeliosConfig(system_name=options.system_name)
        name = f"{options.name}-{uuid4().hex}"
        program = self._client.hugr.upload(
            hugr_package=package, name=name, project=options.project
        )
        kwargs = {
            key: value
            for key, value in {
                "max_cost": options.max_cost,
                "n_qubits": options.n_qubits,
                "credential_name": options.credential_name,
                "user_group": options.user_group,
                "target_region": options.target_region,
            }.items()
            if value is not None
        }
        native = self._client.start_execute_job(
            programs=[program],
            n_shots=[int(shots)],
            backend_config=config,
            name=name,
            project=options.project,
            valid_check=True,
            **kwargs,
        )
        reference = ExecutionReference(
            provider="quantinuum-nexus",
            job_ids=(str(native.id),),
            target=config.system_name,
            context={"shots": str(shots), "package_sha256": fingerprint},
        )
        return NexusExecutionHandle(self._client, native, reference, options)

    def retrieve(self, reference: ExecutionReference) -> NexusExecutionHandle:
        """Restore an execution without uploading or rerunning its program.

        Args:
            reference (ExecutionReference): Reference returned by this transport.

        Returns:
            NexusExecutionHandle: Restored handle for the existing remote job.

        Raises:
            ValueError: If the reference is incompatible or identifies a
                non-execution job.
            Exception: If Nexus cannot retrieve the existing job.
        """
        if reference.provider != "quantinuum-nexus" or len(reference.job_ids) != 1:
            raise ValueError("Expected one quantinuum-nexus execution reference")
        try:
            shots = int(reference.context["shots"])
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "Nexus reference must contain a valid shots count"
            ) from exc
        _validate_positive_integer(shots, "reference shots")
        native = self._client.jobs.get(id=reference.job_ids[0])
        kind = getattr(native, "job_type", None)
        if getattr(kind, "value", kind) != "execute":
            raise ValueError("Nexus reference does not identify an execute job")
        return NexusExecutionHandle(self._client, native, reference, self.options)
