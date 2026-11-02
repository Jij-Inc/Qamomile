"""Quantinuum Nexus endpoint for direct HUGR execution on Helios."""

from __future__ import annotations

import importlib
import math
import threading
import uuid
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from qamomile.compilation import (
    CapabilitySnapshot,
    CompiledProgram,
    CompletedJob,
    ExecutionReference,
    ExecutionSnapshot,
    ExecutionSnapshotKind,
    Invocation,
    Job,
    JobStatus,
    SampleResult,
)
from qamomile.errors import ExecutionError

from ._execution import load_package
from ._profile import CAPABILITIES, FORMAT
from ._target import Hugr

_STATUS_MAP: Mapping[str, JobStatus] = {
    "SUBMITTED": JobStatus.SUBMITTED,
    "RETRYING": JobStatus.SUBMITTED,
    "QUEUED": JobStatus.SUBMITTED,
    "RUNNING": JobStatus.RUNNING,
    "CANCELLING": JobStatus.RUNNING,
    "CANCELLED": JobStatus.CANCELLED,
    "COMPLETED": JobStatus.COMPLETED,
    "ERROR": JobStatus.FAILED,
    "TERMINATED": JobStatus.FAILED,
    "DEPLETED": JobStatus.FAILED,
}


@dataclass(frozen=True, slots=True)
class HeliosOptions:
    """Configure one Nexus Helios execution boundary.

    Args:
        project: Optional native Nexus project reference.
        backend_config: Optional native ``HeliosConfig``.
        system_name: Helios device or emulator name.
        name: Human-readable upload and execution name prefix.
        max_cost: Optional maximum HQC cost for the submission.
        n_qubits: Optional maximum qubit count supplied to Nexus.
        credential_name: Name of a credential already stored by Nexus.
        user_group: Optional scheduling group.
        target_region: Optional ``"us"`` or ``"sg"`` execution region.
        timeout: Default nonnegative local result wait in seconds.

    Raises:
        ValueError: If an option is empty, nonfinite, or out of range.
    """

    project: Any | None = field(default=None, repr=False, compare=False)
    backend_config: Any | None = field(default=None, repr=False, compare=False)
    system_name: str = "Helios-1"
    name: str = "qamomile-hugr"
    max_cost: float | None = None
    n_qubits: int | None = None
    credential_name: str | None = field(default=None, repr=False, compare=False)
    user_group: str | None = None
    target_region: str | None = None
    timeout: float | None = None

    def __post_init__(self) -> None:
        """Validate local policy without contacting Nexus.

        Raises:
            ValueError: If any configured value is invalid.
        """

        if not self.system_name or not self.name:
            raise ValueError("system_name and name must be nonempty")
        if self.max_cost is not None:
            _validate_nonnegative(self.max_cost, "max_cost")
        if self.n_qubits is not None and (
            isinstance(self.n_qubits, bool)
            or not isinstance(self.n_qubits, int)
            or self.n_qubits <= 0
        ):
            raise ValueError("n_qubits must be a positive integer")
        if self.target_region not in {None, "us", "sg"}:
            raise ValueError("target_region must be 'us', 'sg', or None")
        if self.timeout is not None:
            _validate_nonnegative(self.timeout, "timeout")
        if self.backend_config is not None and (
            getattr(self.backend_config, "type", None) != "HeliosConfig"
            or not getattr(self.backend_config, "system_name", None)
        ):
            raise ValueError("backend_config must be a qnexus HeliosConfig")


@dataclass(frozen=True, slots=True)
class Helios:
    """Submit result-instrumented HUGR artifacts to Helios through Nexus.

    Args:
        options: Nexus submission and local waiting policy.
        client: Optional qnexus-compatible client for controlled environments.

    Raises:
        ImportError: If qnexus is unavailable and no client is provided.
    """

    options: HeliosOptions = HeliosOptions()
    client: Any | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Load qnexus without authenticating or submitting work.

        Raises:
            ImportError: If the optional qnexus dependency is unavailable.
        """

        if self.client is None:
            try:
                client = importlib.import_module("qnexus")
            except ImportError as error:
                raise ImportError(
                    'Helios execution requires pip install "qamomile[hugr]"'
                ) from error
            object.__setattr__(self, "client", client)

    @property
    def name(self) -> str:
        """Return the target-scoped endpoint identity."""

        return f"hugr.helios.{self._system_name}"

    @property
    def target(self) -> Hugr:
        """Return the HUGR target with ordered QSystem result records."""

        return Hugr(record_results=True)

    @property
    def capabilities(self) -> CapabilitySnapshot:
        """Return the direct HUGR execution capability snapshot."""

        return CapabilitySnapshot(FORMAT, CAPABILITIES)

    @property
    def _system_name(self) -> str:
        """Return the configured native or shorthand Helios system name."""

        config = self.options.backend_config
        return str(getattr(config, "system_name", self.options.system_name))

    def launch(self, artifact: CompiledProgram, invocation: Invocation) -> Job:
        """Upload and asynchronously submit one closed HUGR artifact.

        Args:
            artifact: Result-instrumented HUGR package for :attr:`target`.
            invocation: Rust-normalized shots and portable seed policy.

        Returns:
            Job: Remote Nexus job with cancellation and snapshot support.

        Raises:
            ExecutionError: If the artifact or invocation is incompatible.
        """

        if artifact.format != FORMAT:
            raise ExecutionError(
                f"Q8640: Helios accepts {FORMAT!r}, got {artifact.format!r}"
            )
        metadata = dict(artifact.metadata.entries)
        if metadata.get("record_results") is not True:
            raise ExecutionError(
                "Q8641: Helios requires a HUGR target with result recording"
            )
        if invocation.parameters.entries:
            raise ExecutionError("Q8642: Helios received unresolved parameters")
        if invocation.seed != 0:
            raise ExecutionError(
                "Q8643: Helios does not provide a portable seed contract; use seed=0"
            )
        client = self.client
        assert client is not None
        config = self.options.backend_config
        if config is None:
            config_type = getattr(client, "HeliosConfig", None)
            if not callable(config_type):
                config_type = getattr(
                    getattr(client, "models", None), "HeliosConfig", None
                )
            if not callable(config_type):
                raise TypeError("qnexus must expose HeliosConfig")
            config = config_type(system_name=self.options.system_name)
        name = f"{self.options.name}-{uuid.uuid4().hex}"
        package = load_package(artifact)
        uploaded = client.hugr.upload(
            hugr_package=package,
            name=name,
            project=self.options.project,
        )
        optional = {
            key: value
            for key, value in {
                "credential_name": self.options.credential_name,
                "max_cost": self.options.max_cost,
                "n_qubits": self.options.n_qubits,
                "target_region": self.options.target_region,
                "user_group": self.options.user_group,
            }.items()
            if value is not None
        }
        native = client.start_execute_job(
            programs=[uploaded],
            n_shots=[invocation.shots],
            backend_config=config,
            name=name,
            project=self.options.project,
            valid_check=True,
            **optional,
        )
        reference = ExecutionReference.create(
            provider="quantinuum-nexus",
            job_ids=(str(native.id),),
            target=str(config.system_name),
            context={
                "compilation_identity": artifact.execution_identity,
                "kind": "sample",
                "shots": str(invocation.shots),
            },
        )
        return _HeliosJob(client, native, artifact, self.options, reference)

    def restore(
        self,
        snapshot: ExecutionSnapshot | ExecutionReference,
        *,
        artifact: CompiledProgram,
    ) -> Job:
        """Restore one existing Nexus job without upload or resubmission.

        Args:
            snapshot: Remote snapshot or direct Nexus reference.
            artifact: Exact HUGR artifact used for the original submission.

        Returns:
            Job: Handle to the existing Nexus execution.

        Raises:
            ValueError: If the reference and artifact do not match this endpoint.
            TypeError: If Nexus returns a non-execution job.
        """

        if (
            artifact.format != FORMAT
            or artifact.target_fingerprint != self.target.fingerprint
        ):
            raise ValueError("artifact does not match this Helios target")
        if isinstance(snapshot, ExecutionSnapshot) and (
            snapshot.kind is ExecutionSnapshotKind.LOCAL
        ):
            result = snapshot.result
            if (
                not isinstance(result, SampleResult)
                or result.result_abi != artifact.result_abi
                or result.backend != self.name
            ):
                raise ValueError("local snapshot does not match this Helios artifact")
            return CompletedJob(f"{self.name}.restored", result)
        reference = _snapshot_reference(snapshot)
        if (
            reference.provider != "quantinuum-nexus"
            or len(reference.job_ids) != 1
            or reference.target != self._system_name
            or reference.context_values.get("kind") != "sample"
        ):
            raise ValueError("snapshot does not identify this Helios endpoint")
        digest = artifact.execution_identity
        if reference.context_values.get("compilation_identity") != digest:
            raise ValueError("artifact does not match the snapshotted Helios job")
        try:
            shots = int(reference.context_values["shots"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Helios snapshot context is invalid") from error
        if shots <= 0:
            raise ValueError("Helios snapshot shots must be positive")
        client = self.client
        assert client is not None
        native = client.jobs.get(id=reference.job_ids[0])
        job_type = getattr(getattr(native, "job_type", None), "value", None)
        if job_type not in {None, "execute"}:
            raise TypeError("Nexus reference does not identify an execute job")
        return _HeliosJob(client, native, artifact, self.options, reference)


@dataclass(frozen=True, slots=True)
class _HeliosJob:
    """Translate one Nexus execute job into the common Job ABI."""

    client: Any = field(repr=False, compare=False)
    native: Any = field(repr=False, compare=False)
    artifact: CompiledProgram = field(repr=False)
    options: HeliosOptions = field(repr=False)
    reference: ExecutionReference
    _result: SampleResult | None = field(
        init=False, default=None, repr=False, compare=False
    )
    _lock: threading.Lock = field(
        init=False, default_factory=threading.Lock, repr=False, compare=False
    )

    @property
    def job_id(self) -> str:
        """Return the Nexus execute-job identity."""

        return self.reference.job_ids[0]

    @property
    def status(self) -> JobStatus:
        """Fetch and normalize the current Nexus status."""

        if self._result is not None:
            return JobStatus.COMPLETED
        try:
            return _nexus_status(self.client.jobs.status(self.native))
        except Exception as error:
            raise ExecutionError(f"Q8644: Nexus status failed: {error}") from error

    def result(self, timeout: float | None = None) -> SampleResult:
        """Wait for and decode one QSystem result through the sealed ABI.

        Args:
            timeout: Nonnegative local wait in seconds, defaulting to options.

        Returns:
            SampleResult: Logical source-ordered counts.

        Raises:
            ValueError: If the timeout is invalid.
            TimeoutError: If the local Nexus wait expires.
            ExecutionError: If execution or result decoding fails.
        """

        effective = self.options.timeout if timeout is None else timeout
        if effective is not None:
            _validate_nonnegative(effective, "timeout")
        with self._lock:
            if self._result is not None:
                return self._result
            try:
                waited = self.client.jobs.wait_for(self.native, timeout=effective)
            except Exception as error:
                if (
                    "timeout" in type(error).__name__.lower()
                    or "timed out" in str(error).lower()
                ):
                    raise TimeoutError("Helios execution result timed out") from error
                raise ExecutionError(f"Q8645: Nexus wait failed: {error}") from error
            status = _nexus_status(waited)
            if status is not JobStatus.COMPLETED:
                raise ExecutionError(f"Q8645: Helios job ended as {status.value}")
            try:
                refs = self.client.jobs.results(self.native, allow_incomplete=False)
                if len(refs) != 1:
                    raise ValueError("Nexus returned an unexpected result count")
                raw = refs[0].download_result()
                shots = int(self.reference.context_values["shots"])
                counts = _decode_qsystem_counts(
                    raw,
                    shots=shots,
                    width=self.artifact.result_abi.width,
                )
                result = SampleResult.from_counts(
                    backend=f"hugr.helios.{self.reference.target}",
                    counts=counts,
                    expected_width=self.artifact.result_abi.width,
                )
                result = replace(result, result_abi=self.artifact.result_abi)
                object.__setattr__(self, "_result", result)
                return result
            except ExecutionError:
                raise
            except Exception as error:
                raise ExecutionError(f"Q8646: invalid Nexus result: {error}") from error

    def cancel(self) -> None:
        """Request cancellation unless the remote job is already terminal."""

        if self.status in {
            JobStatus.CANCELLED,
            JobStatus.COMPLETED,
            JobStatus.FAILED,
        }:
            return
        try:
            self.client.jobs.cancel(self.native)
        except Exception as error:
            raise ExecutionError(
                f"Q8647: Nexus cancellation failed: {error}"
            ) from error

    def snapshot(self) -> ExecutionSnapshot:
        """Capture a cached local result or a secret-free Nexus reference."""

        with self._lock:
            if self._result is not None:
                return ExecutionSnapshot.completed(self._result)
        return ExecutionSnapshot.remote(self.reference)

    def metadata(self) -> Mapping[str, object]:
        """Return detached, non-secret execution metadata."""

        return {
            "job_id": self.job_id,
            "provider": self.reference.provider,
            "shots": int(self.reference.context_values["shots"]),
            "target": self.reference.target,
        }


def _validate_nonnegative(value: object, name: str) -> None:
    """Validate one finite nonnegative timeout or budget."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative finite number")


def _nexus_status(value: object) -> JobStatus:
    """Map one qnexus status record into the portable lifecycle."""

    status = getattr(value, "status", value)
    name = str(getattr(status, "value", getattr(status, "name", status))).upper()
    return _STATUS_MAP.get(name, JobStatus.FAILED)


def _decode_qsystem_counts(
    raw: object,
    *,
    shots: int,
    width: int,
) -> dict[tuple[int, ...], int]:
    """Decode ordered ``qamomile`` result tags into source-ordered counts."""

    native_shots = getattr(raw, "results", None)
    if not isinstance(native_shots, list) or len(native_shots) != shots:
        raise ExecutionError(
            f"Q8646: Nexus returned {len(native_shots) if isinstance(native_shots, list) else 'invalid'} shots; expected {shots}"
        )
    counts: Counter[tuple[int, ...]] = Counter()
    for shot in native_shots:
        collate = getattr(shot, "collate_tags", None)
        if not callable(collate):
            raise ExecutionError("Q8646: Nexus returned an invalid QSystem shot")
        values = collate().get("qamomile")
        if not isinstance(values, list) or len(values) != width:
            raise ExecutionError("Q8646: Nexus result width disagrees with ResultABI")
        bits = tuple(int(value) for value in values)
        if any(bit not in {0, 1} for bit in bits):
            raise ExecutionError("Q8646: Nexus result contains a non-bit value")
        counts[bits] += 1
    return dict(counts)


def _snapshot_reference(
    snapshot: ExecutionSnapshot | ExecutionReference,
) -> ExecutionReference:
    """Extract exactly one remote reference from restoration input."""

    if isinstance(snapshot, ExecutionReference):
        return snapshot
    if not isinstance(snapshot, ExecutionSnapshot):
        raise TypeError("restore expects an ExecutionSnapshot or ExecutionReference")
    if snapshot.kind is not ExecutionSnapshotKind.REMOTE or snapshot.reference is None:
        raise ValueError("restore expects exactly one remote execution snapshot")
    return snapshot.reference


__all__ = ["Helios", "HeliosOptions"]
