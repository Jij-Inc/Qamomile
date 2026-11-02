"""Implement the qBraid endpoint for Qamomile Qiskit artifacts.

QCore and the Qiskit target remain the semantic and artifact authorities. This
package owns only live qBraid device resolution, job submission, lifecycle
mapping, and provider-result normalization. SDK objects and credentials never
enter a persistent Qamomile artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from operator import index
from types import MappingProxyType
from typing import Any

from qamomile.compilation import (
    CapabilitySnapshot,
    CompiledProgram,
    ExecutionReference,
    ExecutionSnapshot,
    Invocation,
    Job,
    JobStatus,
    SampleResult,
)
from qamomile.effects import Requirement
from qamomile.errors import ExecutionError
from qamomile.ir import FrozenMapping
from qamomile.qiskit import (
    FORMAT,
    Qiskit,
    bind_circuit,
    decode_counts,
    load_circuit,
)

_CAPABILITIES = frozenset(
    {
        Requirement.CLASSICAL_COMPUTE,
        Requirement.COHERENT_CONTROL,
        Requirement.DISCARD,
        Requirement.MEASUREMENT,
        Requirement.PARAMETERIZED_EXECUTION,
        Requirement.QUANTUM,
        Requirement.ROTATION,
    }
)


@dataclass(frozen=True, slots=True)
class QBraidDevice:
    """Submit Qiskit QPY artifacts to one live qBraid device.

    The endpoint deliberately advertises only static sampling. Device-specific
    dynamic-control or reset support must not broaden the Qamomile source
    language through an implicit provider conversion.

    Args:
        device: qBraid device exposing ``run(program, shots=...)``.
        poll_interval_seconds: Positive interval passed to qBraid job waits.
        wait_timeout_seconds: Optional positive default wait timeout.
        run_options: Provider options forwarded to ``device.run``. ``shots``
            is reserved by Qamomile and cannot appear here.

    Raises:
        TypeError: If the device or run options do not expose the required
            interface.
        ValueError: If polling settings are invalid or ``shots`` appears in
            run options.
    """

    device: Any = field(repr=False, compare=False)
    poll_interval_seconds: float = 5.0
    wait_timeout_seconds: float | None = None
    run_options: Mapping[str, object] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate and freeze the live provider boundary."""

        if not callable(getattr(self.device, "run", None)):
            raise TypeError("device must expose run(program, ...)")
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if self.wait_timeout_seconds is not None and self.wait_timeout_seconds <= 0:
            raise ValueError("wait_timeout_seconds must be positive or None")
        if not isinstance(self.run_options, Mapping):
            raise TypeError("run_options must be a mapping")
        options = dict(self.run_options)
        if "shots" in options:
            raise ValueError("run_options cannot contain the reserved key 'shots'")
        object.__setattr__(self, "run_options", MappingProxyType(options))

    @classmethod
    def from_device_id(
        cls,
        device_id: str,
        *,
        provider: object | None = None,
        poll_interval_seconds: float = 5.0,
        wait_timeout_seconds: float | None = None,
        run_options: Mapping[str, object] | None = None,
    ) -> QBraidDevice:
        """Resolve a qBraid device without retaining provider credentials.

        Args:
            device_id: Nonempty qBraid device identifier.
            provider: Optional provider exposing ``get_device``. When omitted,
                the installed SDK's default ``QbraidProvider`` is used.
            poll_interval_seconds: Positive interval passed to job waits.
            wait_timeout_seconds: Optional positive default wait timeout.
            run_options: Provider options forwarded at submission.

        Returns:
            QBraidDevice: Endpoint containing only the resolved live device.

        Raises:
            ValueError: If ``device_id`` is empty.
            TypeError: If the provider does not expose ``get_device``.
            ImportError: If the optional qBraid SDK is unavailable.
        """

        if not isinstance(device_id, str) or not device_id:
            raise ValueError("device_id must be a nonempty string")
        if provider is None:
            try:
                from qbraid import QbraidProvider  # type: ignore[import-not-found]
            except (ImportError, ModuleNotFoundError) as error:
                raise ImportError(
                    "qBraid execution requires the optional dependencies; "
                    "install 'qamomile[qbraid]'"
                ) from error
            provider = QbraidProvider()
        get_device = getattr(provider, "get_device", None)
        if not callable(get_device):
            raise TypeError("provider must expose get_device(device_id)")
        return cls(
            get_device(device_id),
            poll_interval_seconds=poll_interval_seconds,
            wait_timeout_seconds=wait_timeout_seconds,
            run_options=run_options or {},
        )

    @property
    def name(self) -> str:
        """Return a provider-scoped endpoint identity."""

        return f"qbraid.device.{_device_name(self.device)}"

    @property
    def target(self) -> Qiskit:
        """Return the generic QPY target accepted by qBraid conversion."""

        return Qiskit()

    @property
    def capabilities(self) -> CapabilitySnapshot:
        """Return the conservative static-sampling capability snapshot."""

        return CapabilitySnapshot(FORMAT, _CAPABILITIES)

    def launch(self, artifact: CompiledProgram, invocation: Invocation) -> Job:
        """Bind and submit one inspected QPY artifact to qBraid."""

        if artifact.format != FORMAT:
            raise ExecutionError(
                f"Q8801: qBraid accepts {FORMAT!r}, got {artifact.format!r}"
            )
        if artifact.result_abi.width == 0:
            raise ExecutionError("Q8802: qBraid execution requires sampling")
        if invocation.seed != 0:
            raise ExecutionError(
                "Q8803: qBraid devices do not provide one portable seed "
                "contract; use seed=0"
            )
        circuit = bind_circuit(load_circuit(artifact), artifact, invocation)
        native = self.device.run(
            circuit,
            shots=invocation.shots,
            **dict(self.run_options),
        )
        return _QBraidJob(
            native=native,
            circuit=circuit,
            artifact=artifact,
            parameters=invocation.parameters,
            backend=self.name,
            poll_interval_seconds=self.poll_interval_seconds,
            wait_timeout_seconds=self.wait_timeout_seconds,
        )


@dataclass(frozen=True, slots=True)
class _QBraidJob:
    """Translate one qBraid job into the portable Qamomile lifecycle."""

    native: Any = field(repr=False)
    circuit: Any = field(repr=False)
    artifact: CompiledProgram = field(repr=False)
    parameters: FrozenMapping = field(repr=False)
    backend: str
    poll_interval_seconds: float
    wait_timeout_seconds: float | None

    @property
    def job_id(self) -> str:
        """Return the provider job identity."""

        identity = getattr(self.native, "id", None)
        value = identity() if callable(identity) else identity
        if value is None:
            raise ExecutionError("Q8804: qBraid job has no job identity")
        return str(value)

    @property
    def status(self) -> JobStatus:
        """Map qBraid's provider status to the portable lifecycle."""

        status = getattr(self.native, "status", None)
        value = status() if callable(status) else status
        name = str(getattr(value, "name", value)).upper()
        if name in {"CREATED", "INITIALIZING", "PENDING", "QUEUED", "SUBMITTED"}:
            return JobStatus.SUBMITTED
        if name in {"RUNNING", "CANCELLING"}:
            return JobStatus.RUNNING
        if name in {"COMPLETED", "DONE", "SUCCESS"}:
            return JobStatus.COMPLETED
        if name in {"CANCELED", "CANCELLED"}:
            return JobStatus.CANCELLED
        return JobStatus.FAILED

    def result(self, timeout: float | None = None) -> SampleResult:
        """Wait for and decode qBraid counts through the Qiskit ABI."""

        wait = getattr(self.native, "wait_for_final_state", None)
        if not callable(wait):
            raise ExecutionError("Q8805: qBraid job has no wait operation")
        wait(
            timeout=self.wait_timeout_seconds if timeout is None else timeout,
            poll_interval=self.poll_interval_seconds,
        )
        result_method = getattr(self.native, "result", None)
        if not callable(result_method):
            raise ExecutionError("Q8806: qBraid job has no result() method")
        native_result = result_method()
        get_counts = getattr(getattr(native_result, "data", None), "get_counts", None)
        if not callable(get_counts):
            raise ExecutionError("Q8807: qBraid result has no measurement counts")
        native_counts = get_counts()
        if not isinstance(native_counts, Mapping):
            raise ExecutionError("Q8807: qBraid measurement counts are not a mapping")
        try:
            normalized: dict[object, int] = {
                str(bitstring).replace(" ", ""): index(count)
                for bitstring, count in native_counts.items()
            }
            logical_counts = decode_counts(
                self.artifact,
                normalized,
                dict(self.parameters.entries),
                len(self.circuit.clbits),
            )
            return SampleResult.from_counts(
                backend=self.backend,
                counts=logical_counts,
                expected_width=self.artifact.result_abi.width,
            )
        except (ExecutionError, TypeError, ValueError) as error:
            raise ExecutionError("Q8808: qBraid returned malformed counts") from error

    def cancel(self) -> None:
        """Request best-effort cancellation from the qBraid provider job."""

        cancel = getattr(self.native, "cancel", None)
        if not callable(cancel):
            raise ExecutionError("Q8809: qBraid job does not support cancellation")
        cancel()

    def snapshot(self) -> ExecutionSnapshot:
        """Capture the qBraid job ID without retaining an SDK object.

        Returns:
            Secret-free remote execution reference.
        """

        return ExecutionSnapshot.remote(
            ExecutionReference.create(
                provider="qbraid",
                job_ids=(self.job_id,),
                target=self.backend,
                context={"kind": "sample"},
            )
        )


def _device_name(device: Any) -> str:
    """Return one stable printable qBraid device identity."""

    value = getattr(device, "id", None)
    value = value() if callable(value) else value
    if value is None:
        profile = getattr(device, "profile", None)
        value = getattr(profile, "device_id", None)
    if value is None:
        value = type(device).__name__
    return str(value).replace(" ", "-")


QBraidDevice.__module__ = __package__


__all__ = ["QBraidDevice"]
