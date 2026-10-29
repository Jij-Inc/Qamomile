"""Submit instrumented HUGR programs to Helios or H2 through Nexus."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import math
import re
import threading
import time
from collections.abc import Mapping
from enum import StrEnum
from numbers import Integral
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from qamomile.circuit.transpiler import (
    ExecutionError,
    ExecutionHandle,
    ExecutionReference,
    JobStatus,
)
from qamomile.hugr._qir import _H2AbortError, to_h2_qir
from qamomile.hugr._qir_results import decode_qir_results, parse_qir_output_manifest
from qamomile.hugr._shot_range import (
    SHOT_SUCCESS_TAG,
    normalize_h2_shots,
    prepare_h2_shot_range,
)

if TYPE_CHECKING:
    from hugr.package import Package

_H2_SHOT_ORIGIN = 1
_H2_TARGET = re.compile(r"H2-(?:[1-9][0-9]*(?:E|SC)?|Emulator)")
_SHA256 = re.compile(r"[0-9a-f]{64}")

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
class _NexusDestination:
    """Keep device-specific execution conventions inside the transport."""

    target: str
    program_format: Literal["hugr", "qir"]
    shot_origin: int


@dataclasses.dataclass(frozen=True)
class NexusExecutionOptions:
    """Configure execution on a Helios or H2 target through Nexus.

    Args:
        project (Any | None): Nexus project name or native ``qnexus`` project
            reference. A name is looked up or created when submitting a job.
            ``None`` uses the active Nexus project, or looks up or creates
            ``qamomile`` if no project is active. Authentication is automatic
            on the first submission or restoration; an initial login may
            require browser approval.
        backend_config (Any | None): Native ``qnexus.HeliosConfig`` or H2
            ``qnexus.QuantinuumConfig`` with optional device settings. Its
            ``system_name`` or ``device_name`` overrides ``system_name`` below.
        system_name (str): Device name, defaulting to ``Helios-1``. Helios
            receives HUGR; H2 receives automatically converted QIR. H2 requires
            Python 3.11 or 3.12 with the ``hugr`` extra and Bit public outputs.
            Every H2 shot read uses a zero-based logical index, normalized from
            the internal device origin before QIR conversion. Helios uses an
            internal zero-based origin policy and preserves shot reads without
            rewriting the upload; its device origin remains unverified.
        name (str): Program and job name prefix. Each submission receives a
            unique suffix, including separate expectation measurement jobs.
        max_cost (float | None): Optional maximum HQC cost for each submitted
            program. Required when submitting to Helios; saved jobs can be
            restored without it. A deprecated native config limit is used
            when omitted, and conflicting limits are rejected. This is not
            an aggregate expectation-estimation budget.
        n_qubits (int | None): Optional maximum qubit count passed to Nexus.
        credential_name (str | None): Name of a credential already in Nexus.
        user_group (str | None): Nexus user group for scheduling.
        target_region (str | NexusRegion | None): Execution region, ``us`` or ``sg``.
        poll_interval_seconds (float): Positive local status polling interval.
        timeout_seconds (float | None): Default local status-wait deadline.
            ``None`` waits indefinitely. Timing out never cancels a job.

    Raises:
        ValueError: If a numeric option, project name, region, or config is invalid.
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
        if isinstance(self.project, str) and not self.project.strip():
            raise ValueError("project name must be a non-empty string")
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
        _destination(self)


def _destination(options: NexusExecutionOptions) -> _NexusDestination:
    """Resolve the actual device and submission format before provider calls.

    Args:
        options (NexusExecutionOptions): Native config or device-name selection.

    Returns:
        _NexusDestination: Target, program format, and internal shot origin.

    Raises:
        ValueError: If a device or native config family is unsupported.
    """
    config = options.backend_config
    kind = getattr(config, "type", None)
    if config is None:
        name = options.system_name
    elif kind == "HeliosConfig":
        name = getattr(config, "system_name", None)
    elif kind == "QuantinuumConfig":
        name = getattr(config, "device_name", None)
    else:
        raise ValueError(
            "backend_config must be a qnexus.HeliosConfig or H2 QuantinuumConfig"
        )
    if isinstance(name, str):
        if re.fullmatch(r"Helios-[A-Za-z0-9][A-Za-z0-9-]*", name):
            if config is None or kind == "HeliosConfig":
                return _NexusDestination(name, "hugr", 0)
        if _H2_TARGET.fullmatch(name):
            if config is None or kind == "QuantinuumConfig":
                return _NexusDestination(name, "qir", _H2_SHOT_ORIGIN)
    raise ValueError("Nexus execution requires a matching Helios or H2 device config")


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


def _qir_success_tag(
    context: Mapping[str, str], outputs: Mapping[str, tuple[str, int]]
) -> str | None:
    """Read the required success output from a saved execution reference.

    Args:
        context (Mapping[str, str]): Saved execution settings, including shots
            and the origin used when the submitted program was prepared.
        outputs (Mapping[str, tuple[str, int]]): Validated QIR output manifest.

    Returns:
        str | None: Private Boolean tag to check on every shot and omit from
        public results, or ``None`` for an unguarded program.

    Raises:
        ValueError: If the success output or origin is missing, malformed,
            inconsistent, or outside the unsigned 64-bit shot range.
    """
    success_tag = context.get("shot_success_tag")
    shot_origin = context.get("shot_origin")
    if success_tag is None:
        if shot_origin is not None or SHOT_SUCCESS_TAG in outputs:
            raise ValueError("QIR shot marker requires a mandatory success output")
        return None
    if (
        not isinstance(success_tag, str)
        or not success_tag
        or not isinstance(shot_origin, str)
        or re.fullmatch(r"0|[1-9][0-9]*", shot_origin) is None
        or outputs.get(success_tag) != ("bool", 1)
        or (SHOT_SUCCESS_TAG in outputs and success_tag != SHOT_SUCCESS_TAG)
    ):
        raise ValueError("Invalid H2 shot guard contract")
    if int(shot_origin) + int(context["shots"]) >= 2**64:
        raise ValueError("H2 shot guard range must fit unsigned 64-bit integers")
    return success_tag


def _prepared_context(context: Mapping[str, str]) -> dict[str, str]:
    """Validate and detach the recovery context for exact H2 bitcode.

    Args:
        context (Mapping[str, str]): Output and original package metadata.
            Unknown string fields are retained for compatible extensions.

    Returns:
        dict[str, str]: Independently owned, validated context.

    Raises:
        ValueError: If required metadata, counts, hashes, or guards are invalid.
    """
    if not isinstance(context, Mapping) or not all(
        isinstance(key, str) and key and isinstance(value, str)
        for key, value in context.items()
    ):
        raise ValueError("Prepared H2 context must map non-empty strings to strings")
    context = dict(context)
    if (
        re.fullmatch(r"[1-9][0-9]*", context.get("shots", "")) is None
        or context.get("program_format") != "qir"
        or _SHA256.fullmatch(context.get("package_sha256", "")) is None
    ):
        raise ValueError(
            "Prepared H2 context requires valid shots, format, and package hash"
        )
    _validate_positive_integer(int(context["shots"]), "prepared shots")
    outputs = parse_qir_output_manifest(context.get("qir_outputs", ""))
    success_tag = _qir_success_tag(context, outputs)
    h2_hash = context.get("h2_package_sha256")
    if (success_tag is None and h2_hash is not None) or (
        success_tag is not None
        and (h2_hash is None or _SHA256.fullmatch(h2_hash) is None)
    ):
        raise ValueError("Prepared H2 guard requires its normalized package hash")
    return context


def _prepared_checksum(metadata: Mapping[str, Any]) -> str:
    """Checksum canonical metadata including the separate QIR digest.

    Args:
        metadata (Mapping[str, Any]): Version, QIR digest, and validated context.

    Returns:
        str: SHA-256 checksum for accidental corruption detection.
    """
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclasses.dataclass(frozen=True)
class PreparedH2Submission:
    """Own exact H2 QIR bytes and their read-only recovery context.

    This record contains no client, credentials, destination, or job name. Save
    the QIR as binary and :meth:`to_dict` metadata as JSON before submission.
    Checksums detect accidental corruption and do not establish authenticity.

    Args:
        qir (bytes): Final LLVM bitcode sent unchanged to Nexus.
        context (Mapping[str, str]): Matching output, shot, original package,
            and optional guard metadata. The record copies this mapping.

    Raises:
        ValueError: If bitcode magic or recovery metadata is invalid.
    """

    qir: bytes
    context: Mapping[str, str]

    def __post_init__(self) -> None:
        """Validate the payload and make its detached context read-only.

        Raises:
            ValueError: If bitcode magic or recovery metadata is invalid.
        """
        if not isinstance(self.qir, bytes) or not self.qir.startswith(b"BC\xc0\xde"):
            raise ValueError("Prepared H2 QIR must be LLVM bitcode bytes")
        object.__setattr__(
            self, "context", MappingProxyType(_prepared_context(self.context))
        )

    @property
    def shots(self) -> int:
        """Return the fixed repetition count.

        Returns:
            int: Positive count saved in the recovery context.
        """
        return int(self.context["shots"])

    @property
    def qir_sha256(self) -> str:
        """Return the digest of the exact submitted bytes.

        Returns:
            str: Lowercase hexadecimal SHA-256 digest.
        """
        return hashlib.sha256(self.qir).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return versioned JSON-compatible metadata without the QIR bytes.

        Returns:
            dict[str, Any]: Detached metadata with QIR and combined checksums.
        """
        metadata = {
            "version": 1,
            "qir_sha256": self.qir_sha256,
            "context": dict(self.context),
        }
        return {**metadata, "prepared_sha256": _prepared_checksum(metadata)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, qir: bytes) -> PreparedH2Submission:
        """Load saved metadata and its separately stored QIR bytes.

        Args:
            data (Mapping[str, Any]): Exact version-one metadata from ``to_dict``.
            qir (bytes): Separately saved binary LLVM bitcode.

        Returns:
            PreparedH2Submission: Validated, independently owned payload.

        Raises:
            ValueError: If the schema, metadata, bytes, or checksums are invalid.
        """
        if not isinstance(data, Mapping) or set(data) != {
            "version",
            "qir_sha256",
            "context",
            "prepared_sha256",
        }:
            raise ValueError("Invalid prepared H2 metadata schema")
        if type(data["version"]) is not int or data["version"] != 1:
            raise ValueError("Unsupported prepared H2 metadata version")
        for name in ("qir_sha256", "prepared_sha256"):
            if not isinstance(data[name], str) or _SHA256.fullmatch(data[name]) is None:
                raise ValueError(f"Invalid prepared H2 {name}")
        prepared = cls(qir=qir, context=data["context"])
        if prepared.qir_sha256 != data["qir_sha256"]:
            raise ValueError("Prepared H2 QIR checksum mismatch")
        if prepared.to_dict()["prepared_sha256"] != data["prepared_sha256"]:
            raise ValueError("Prepared H2 metadata checksum mismatch")
        return prepared

    def reference(self, *, job_id: str, target: str) -> ExecutionReference:
        """Attach an independently discovered H2 job ID to saved context.

        Args:
            job_id (str): Non-empty Nexus execute-job identifier found by caller.
            target (str): Actual supported H2 device used for that submission.

        Returns:
            ExecutionReference: Existing tagged-result restoration reference.

        Raises:
            ValueError: If the job ID or H2 target is invalid.
        """
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("job_id must be a non-empty string")
        if not isinstance(target, str) or _H2_TARGET.fullmatch(target) is None:
            raise ValueError("Prepared H2 references require a supported H2 target")
        return ExecutionReference(
            provider="quantinuum-nexus",
            job_ids=(job_id,),
            target=target,
            context=self.context,
        )


def prepare_h2_submission(package: Package, shots: int = 1024) -> PreparedH2Submission:
    """Prepare final H2 bitcode and recovery metadata without contacting Nexus.

    Shot normalization and fallback table proofs leave the original package
    unchanged. The result can be saved before creating any executor or client.

    Args:
        package (Package): Bound HUGR package with closed, instrumented ``main``.
        shots (int): Positive repetition count, default ``1024``.

    Returns:
        PreparedH2Submission: Final bytes and matching tagged-result context.

    Raises:
        ValueError: If the repetition count is invalid.
        ImportError: If H2 conversion dependencies are unavailable.
        ExecutionError: If normalization, conversion, or output contracts fail.
    """
    _validate_positive_integer(shots, "shots")
    fingerprint = hashlib.sha256(package.to_bytes()).hexdigest()
    h2_package, guarded = normalize_h2_shots(package, shots, _H2_SHOT_ORIGIN)
    try:
        bitcode, outputs = to_h2_qir(h2_package)
    except _H2AbortError:
        h2_package, guarded = prepare_h2_shot_range(package, shots, _H2_SHOT_ORIGIN)
        if not guarded:
            raise
        bitcode, outputs = to_h2_qir(h2_package)
    context = {
        "shots": str(shots),
        "package_sha256": fingerprint,
        "program_format": "qir",
        "qir_outputs": json.dumps(outputs, sort_keys=True, separators=(",", ":")),
    }
    if guarded:
        context.update(
            shot_success_tag=SHOT_SUCCESS_TAG,
            shot_origin=str(_H2_SHOT_ORIGIN),
            h2_package_sha256=hashlib.sha256(h2_package.to_bytes()).hexdigest(),
        )
    try:
        return PreparedH2Submission(bitcode, context)
    except ValueError as exc:
        raise ExecutionError("Invalid H2 shot guard output contract") from exc


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


def _raw_output_tag(tag: str, values: list[Any]) -> str | None:
    """Classify a RAW record before interpreting its user-provided name."""
    parts = tag.split(":", 2)
    if parts[0] in ("EXIT", "_EXCEPTION", "_STDERR", "_STDOUT"):
        raise ExecutionError(
            f"Nexus HUGR shot terminated or contains execution error context: {tag}"
        )
    if len(parts) == 3 and parts[0] == "USER":
        if parts[1] in ("BOOL", "INT", "UINT", "FLOAT") and parts[2]:
            return parts[2]
    if parts[0] == "METRICS":
        if (
            len(parts) == 3
            and parts[1] in ("BOOL", "INT", "FLOAT")
            and parts[2]
            and all(type(value) in (bool, int, float) for value in values)
        ):
            return None
        raise ExecutionError(f"Nexus returned a malformed HUGR metric: {tag}")
    raise ExecutionError(f"Nexus returned an unsupported output tag: {tag}")


def _decode_results(
    raw: Any, shots: int, *, raw_format: bool = False
) -> list[dict[str, Any]]:
    """Preserve typed tagged outputs without overwriting repeated records.

    Args:
        raw (Any): Native HUGR ``QsysResult`` returned by Nexus.
        shots (int): Required number of completed shots.
        raw_format (bool): Whether the result was downloaded with RAW version 4.
            RAW records retain their namespace and type prefixes. The default
            treats every tag as a literal user name from DEFAULT version 3.

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
            if not isinstance(tag, str):
                raise ExecutionError(
                    f"Nexus returned an invalid HUGR output tag: {tag!r}"
                )
            if raw_format:
                tag = _raw_output_tag(tag, values)
                if tag is None:
                    continue
            if len(values) != 1:
                raise ExecutionError(
                    f"Nexus output tag {tag!r} must have exactly one value per shot"
                )
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
            is_qir = self._reference.context.get("program_format") == "qir"
            if is_qir:
                raw = refs[0].download_result()
            else:
                from qnexus.models.references import ResultVersions

                raw = refs[0].download_result(version=ResultVersions.RAW)
            try:
                shots = int(self._reference.context["shots"])
                if is_qir:
                    outputs = parse_qir_output_manifest(
                        self._reference.context["qir_outputs"]
                    )
                    success_tag = _qir_success_tag(self._reference.context, outputs)
                    self._result = (
                        decode_qir_results(
                            raw, shots, outputs, required_success_tag=success_tag
                        )
                        if success_tag is not None
                        else decode_qir_results(raw, shots, outputs)
                    )
                else:
                    self._result = _decode_results(raw, shots, raw_format=True)
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
    """Submit closed HUGR entry points to Helios or convert them for H2.

    The caller supplies an execution wrapper with runtime values connected to
    the unchanged parameterized body and with explicit output records. Nexus
    HUGR submission has no separate entry-point argument field. This transport owns
    lazy authentication and project resolution without changing the SDK's
    active project. Local validation precedes authentication and remote calls.

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
                    "Nexus execution requires qnexus; install qamomile[hugr]"
                ) from exc
        self._client = client
        self._session_lock = threading.Lock()
        self._authenticated = False
        self._named_project: Any | None = None

    def _authenticate(self) -> None:
        """Reuse saved Nexus credentials or start login once per transport.

        Failed authentication leaves the transport available for a later retry.
        The execution target region is independent of the SDK's login region.

        Raises:
            Exception: If Nexus authentication fails.
        """
        with self._session_lock:
            if not self._authenticated:
                self._client.login()
                self._authenticated = True

    def _submission_project(self) -> Any:
        """Resolve a project for upload without changing the SDK active project.

        Explicit references are preserved, and an active project is read on
        every submission. Named or default projects are resolved once after
        successful authentication.

        Returns:
            Any: Native Nexus project reference for upload and execution.

        Raises:
            Exception: If authentication or project lookup or creation fails.
        """
        self._authenticate()
        project = self.options.project
        if project is None:
            project = self._client.context.get_active_project(project_required=False)
        if project is not None and not isinstance(project, str):
            return project
        with self._session_lock:
            if self._named_project is None:
                self._named_project = self._client.projects.get_or_create(
                    name=project or "qamomile"
                )
            return self._named_project

    def _submission_configuration(
        self, destination: _NexusDestination
    ) -> tuple[Any, dict[str, Any]]:
        """Resolve native configuration and validate per-program cost settings.

        Args:
            destination (_NexusDestination): Actual device and upload format.

        Returns:
            tuple[Any, dict[str, Any]]: Native config and optional SDK settings.

        Raises:
            ValueError: If cost limits are invalid, conflicting, or absent for Helios.
        """
        options = self.options
        config = options.backend_config
        native_cost = getattr(config, "max_cost", None)
        if native_cost is not None:
            _validate_seconds(native_cost, "backend_config.max_cost", True)
        if (
            options.max_cost is not None
            and native_cost is not None
            and options.max_cost != native_cost
        ):
            raise ValueError(
                "max_cost conflicts with backend_config.max_cost; "
                "specify one limit or use equal values"
            )
        max_cost = options.max_cost if options.max_cost is not None else native_cost
        if destination.program_format == "hugr" and max_cost is None:
            raise ValueError("Helios submission requires an explicit max_cost")
        if config is None:
            config = (
                self._client.HeliosConfig(system_name=destination.target)
                if destination.program_format == "hugr"
                else self._client.QuantinuumConfig(device_name=destination.target)
            )
        settings = {
            key: value
            for key, value in {
                "max_cost": max_cost,
                "n_qubits": options.n_qubits,
                "credential_name": options.credential_name,
                "user_group": options.user_group,
                "target_region": options.target_region,
            }.items()
            if value is not None
        }
        return config, settings

    def _start_execution(
        self,
        *,
        program: Any,
        project: Any,
        config: Any,
        settings: Mapping[str, Any],
        name: str,
        target: str,
        context: Mapping[str, str],
    ) -> NexusExecutionHandle:
        """Start one uploaded program and retain its exact restoration context.

        Args:
            program (Any): Uploaded native program reference.
            project (Any): Resolved Nexus project reference.
            config (Any): Validated native device configuration.
            settings (Mapping[str, Any]): Optional per-program SDK settings.
            name (str): Exact program and execution name.
            target (str): Actual submitted device.
            context (Mapping[str, str]): Matching shot and result metadata.

        Returns:
            NexusExecutionHandle: Handle for the single execution request.

        Raises:
            Exception: If the Nexus SDK cannot start execution.
        """
        native = self._client.start_execute_job(
            programs=[program],
            n_shots=[int(context["shots"])],
            backend_config=config,
            name=name,
            project=project,
            valid_check=True,
            **settings,
        )
        reference = ExecutionReference(
            provider="quantinuum-nexus",
            job_ids=(str(native.id),),
            target=target,
            context=context,
        )
        return NexusExecutionHandle(self._client, native, reference, self.options)

    def submit(self, package: Package, shots: int) -> NexusExecutionHandle:
        """Prepare the selected device format and submit one execution request.

        Args:
            package (Package): Instrumented HUGR package with a closed entry point.
            shots (int): Positive number of shots to execute.

        Returns:
            NexusExecutionHandle: Handle returned without waiting for results.

        Raises:
            ValueError: If shots, destination, or cost settings are invalid.
            ImportError: If H2 conversion dependencies are unavailable.
            ExecutionError: If conversion fails or a syntax checker cannot
                verify the required shot-success guard.
            Exception: If authentication, project resolution, upload, or submission fails.
        """
        _validate_positive_integer(shots, "shots")
        destination = _destination(self.options)
        config, settings = self._submission_configuration(destination)
        name = f"{self.options.name}-{uuid4().hex}"
        if destination.program_format == "qir":
            return self._send_prepared(
                prepare_h2_submission(package, shots),
                name=name,
                destination=destination,
                config=config,
                settings=settings,
            )
        context = {
            "shots": str(shots),
            "package_sha256": hashlib.sha256(package.to_bytes()).hexdigest(),
        }
        project = self._submission_project()
        program = self._client.hugr.upload(
            hugr_package=package, name=name, project=project
        )
        return self._start_execution(
            program=program,
            project=project,
            config=config,
            settings=settings,
            name=name,
            target=destination.target,
            context=context,
        )

    def submit_prepared(
        self, prepared: PreparedH2Submission, *, name: str
    ) -> NexusExecutionHandle:
        """Submit saved H2 bytes unchanged with an exact caller-owned name.

        Args:
            prepared (PreparedH2Submission): Validated bytes and fixed shot context.
            name (str): Complete non-empty program and execution name. No suffix
                is added, so callers can use it to match an uncertain submission.

        Returns:
            NexusExecutionHandle: Handle for the single Nexus execution request.

        Raises:
            TypeError: If prepared is not a PreparedH2Submission.
            ValueError: If name, device, shot origin, or cost settings are invalid.
            ExecutionError: If a syntax checker cannot verify the required guard.
            Exception: If authentication, project resolution, upload, or submission fails.
        """
        if not isinstance(prepared, PreparedH2Submission):
            raise TypeError("prepared must be a PreparedH2Submission")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        destination = _destination(self.options)
        if destination.program_format != "qir":
            raise ValueError("Prepared H2 submissions require an H2 destination")
        config, settings = self._submission_configuration(destination)
        return self._send_prepared(
            prepared,
            name=name,
            destination=destination,
            config=config,
            settings=settings,
        )

    def _send_prepared(
        self,
        prepared: PreparedH2Submission,
        *,
        name: str,
        destination: _NexusDestination,
        config: Any,
        settings: Mapping[str, Any],
    ) -> NexusExecutionHandle:
        """Check device compatibility and send the prepared bytes once.

        Args:
            prepared (PreparedH2Submission): Validated final bytes and context.
            name (str): Exact program and execution name.
            destination (_NexusDestination): Resolved H2 device and shot origin.
            config (Any): Validated native device configuration.
            settings (Mapping[str, Any]): Validated optional SDK settings.

        Returns:
            NexusExecutionHandle: Handle for the single execution request.

        Raises:
            ValueError: If the saved shot origin differs from the destination.
            ExecutionError: If a syntax checker cannot verify the required guard.
            Exception: If authentication, project resolution, upload, or submission fails.
        """
        origin = prepared.context.get("shot_origin")
        if origin is not None and origin != str(destination.shot_origin):
            raise ValueError(
                "Prepared H2 shot origin is incompatible with the destination"
            )
        if prepared.context.get(
            "shot_success_tag"
        ) is not None and destination.target.endswith("SC"):
            raise ExecutionError(
                "H2 syntax checkers return synthetic zero results and cannot "
                "verify shot-index guards; use an H2 emulator or hardware"
            )
        project = self._submission_project()
        program = self._client.qir.upload(qir=prepared.qir, name=name, project=project)
        return self._start_execution(
            program=program,
            project=project,
            config=config,
            settings=settings,
            name=name,
            target=destination.target,
            context=prepared.context,
        )

    def retrieve(self, reference: ExecutionReference) -> NexusExecutionHandle:
        """Restore an execution without uploading or rerunning its program.

        Args:
            reference (ExecutionReference): Reference returned by this transport.

        Returns:
            NexusExecutionHandle: Restored handle for the existing remote job.

        Raises:
            ValueError: If the reference is incompatible or identifies a
                non-execution job.
            Exception: If authentication or retrieval of the existing job fails.
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
        program_format = reference.context.get("program_format", "hugr")
        if program_format not in ("hugr", "qir"):
            raise ValueError("Nexus reference has an unsupported program format")
        if program_format == "qir":
            try:
                outputs = parse_qir_output_manifest(reference.context["qir_outputs"])
                _qir_success_tag(reference.context, outputs)
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    "Nexus reference must contain valid QIR outputs"
                ) from exc
        elif any(
            name in reference.context for name in ("shot_success_tag", "shot_origin")
        ):
            raise ValueError("H2 shot guard contract requires QIR results")
        self._authenticate()
        native = self._client.jobs.get(id=reference.job_ids[0])
        kind = getattr(native, "job_type", None)
        if getattr(kind, "value", kind) != "execute":
            raise ValueError("Nexus reference does not identify an execute job")
        return NexusExecutionHandle(self._client, native, reference, self.options)
