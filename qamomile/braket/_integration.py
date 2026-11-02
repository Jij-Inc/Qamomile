"""Implement Amazon Braket OpenQASM realizations and execution endpoints."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from importlib.metadata import version
from typing import Any

from qamomile.circuit import (
    RESULT_POSTPROCESSING_FORMAT,
    RESULT_POSTPROCESSING_METADATA,
    CircuitProgram,
    circuit_result_postprocessing,
    lower_circuit_target,
    postprocess_circuit_counts,
    scalar_parameter_bindings,
)
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
    LoweredProgram,
    RealizationCapability,
    SampleResult,
)
from qamomile.effects import Requirement
from qamomile.errors import ExecutionError
from qamomile.ir import FrozenMapping, Program
from qamomile.local_execution import launch_local_sample
from qamomile.lowering import RealizationOptions

from ._emission import emit_openqasm as _emit_openqasm

FORMAT = "braket.openqasm3.v1"
_CAPABILITIES = frozenset(
    {
        Requirement.CLASSICAL_COMPUTE,
        Requirement.COHERENT_CONTROL,
        Requirement.PARAMETERIZED_EXECUTION,
        Requirement.QUANTUM,
        Requirement.MEASUREMENT,
        Requirement.MID_CIRCUIT_MEASUREMENT,
        Requirement.DYNAMIC_BRANCH,
        Requirement.RESET,
        Requirement.DISCARD,
        Requirement.ROTATION,
    }
)
_DEVICE_CAPABILITIES = frozenset(
    {
        Requirement.CLASSICAL_COMPUTE,
        Requirement.COHERENT_CONTROL,
        Requirement.PARAMETERIZED_EXECUTION,
        Requirement.QUANTUM,
        Requirement.MEASUREMENT,
        Requirement.DISCARD,
        Requirement.ROTATION,
    }
)
_REALIZATIONS = frozenset({RealizationCapability.RZZ})


@dataclass(frozen=True, slots=True)
class Braket:
    """Compile circuit-shaped QCore to retained SDK OpenQASM 3 programs.

    Args:
        requirements (frozenset[Requirement]): Supported execution requirements.
        device_snapshot (str | None): Optional immutable device identity.
        realizations (RealizationOptions): Exact operation-realization overrides.
        use_native_controls (bool): Retain controlled gates for compatible endpoints.

    Raises:
        TypeError: If requirements are not capability values.
        ValueError: If a provided device snapshot is empty.
    """

    requirements: frozenset[Requirement] = _CAPABILITIES
    device_snapshot: str | None = None
    realizations: RealizationOptions = RealizationOptions()
    use_native_controls: bool = field(default=True, kw_only=True)

    def __post_init__(self) -> None:
        """Validate an optional managed-device snapshot identity."""

        requirements = frozenset(self.requirements)
        if any(not isinstance(item, Requirement) for item in requirements):
            raise TypeError("requirements must contain Requirement values")
        object.__setattr__(self, "requirements", requirements)
        if self.device_snapshot is not None and not self.device_snapshot:
            raise ValueError("device_snapshot must be nonempty when provided")

    @property
    def name(self) -> str:
        """Return the stable artifact format identifier."""

        return FORMAT

    @property
    def capabilities(self) -> CapabilitySnapshot:
        """Return the supported OpenQASM execution contract."""

        return CapabilitySnapshot(
            FORMAT, self.requirements, self._realizations, self.realizations
        )

    @property
    def _realizations(self) -> frozenset[RealizationCapability]:
        """Return controlled-gate retention allowed by this endpoint profile."""
        return _REALIZATIONS | (
            frozenset(
                {RealizationCapability.MCX, RealizationCapability.CONTROLLED_SWAP}
            )
            if self.use_native_controls
            else frozenset()
        )

    @property
    def supports_circuit_result_postprocessing(self) -> bool:
        """Return whether execution evaluates retained result trees."""

        return True

    @property
    def fingerprint(self) -> str:
        """Return the versioned OpenQASM emission identity."""

        policy = json.dumps(
            {
                "emitter": "braket-openqasm3-v3",
                "result_postprocessing": RESULT_POSTPROCESSING_FORMAT,
                "sdk": version("amazon-braket-sdk"),
                "requirements": sorted(item.value for item in self.requirements),
                "realization_capabilities": sorted(
                    item.value for item in self._realizations
                ),
                "realization_options": self.realizations.overrides,
                "device_snapshot": self.device_snapshot,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return "braket-openqasm3-" + hashlib.sha256(policy).hexdigest()[:20]

    def lower(self, program: Program) -> LoweredProgram:
        """Retain an SDK OpenQASM Program with ABI facts."""

        return lower_circuit_target(program, self)

    def lower_circuit(self, logical: CircuitProgram) -> LoweredProgram:
        """Retain one verified circuit as an SDK OpenQASM Program."""

        source = _emit_openqasm(logical)
        from ._payload import BraketExecutable

        payload = BraketExecutable(source)
        return LoweredProgram(
            payload=payload,
            parameter_abi=logical.parameter_abi,
            result_abi=logical.result_abi,
            stages=(*logical.realization_stages, "openqasm3-emission"),
            circuit_qubits=logical.width,
            circuit_gates=logical.operation_count,
            emitted_instructions=logical.operation_count,
            metadata=FrozenMapping.from_mapping(
                {
                    "braket_sdk_version": version("amazon-braket-sdk"),
                    **(
                        {RESULT_POSTPROCESSING_METADATA: plan}
                        if (plan := circuit_result_postprocessing(logical)) is not None
                        else {}
                    ),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class BraketLocal:
    """Execute OpenQASM artifacts in this process with Braket LocalSimulator.

    Args:
        backend (str): Local simulator implementation name.
        realizations (RealizationOptions): Exact choices used when compiling source.
    """

    backend: str = "default"
    realizations: RealizationOptions = RealizationOptions()

    @property
    def name(self) -> str:
        """Return the endpoint identity."""

        return f"braket.local.{self.backend}"

    @property
    def target(self) -> Braket:
        """Return the compiler target accepted by this endpoint."""

        return Braket(realizations=self.realizations)

    @property
    def capabilities(self) -> CapabilitySnapshot:
        """Return this local simulator's immutable capability snapshot."""

        return CapabilitySnapshot(FORMAT, _CAPABILITIES)

    def launch(
        self,
        artifact: CompiledProgram,
        invocation: Invocation,
    ) -> Job:
        """Execute an already emitted OpenQASM artifact in this process.

        Args:
            artifact (CompiledProgram): Sealed OpenQASM artifact.
            invocation (Invocation): Validated parameters, shots, and seed.

        Returns:
            Job: Completed local simulation job.

        Raises:
            ExecutionError: If the artifact format or simulation is invalid.
        """

        return launch_local_sample(
            artifact,
            invocation,
            endpoint=self.name,
            artifact_format=FORMAT,
            job_prefix="braket-local",
            execute=lambda: _execute_local(artifact, invocation, backend=self.backend),
        )


@dataclass(frozen=True, slots=True)
class BraketDevice:
    """Submit portable OpenQASM to one live Amazon Braket device.

    The SDK device and task remain endpoint-local. Artifacts contain only
    OpenQASM, logical ABIs, and the immutable snapshot used for capability
    negotiation; they never serialize AWS sessions or credentials.

    Args:
        device: An ``AwsDevice``-compatible object exposing ``run``.
        requirements: Explicit capabilities accepted by the device. The
            default deliberately excludes dynamic control and reset.
        device_snapshot: Optional immutable properties/calibration identity.
        poll_timeout_seconds: Native task polling timeout supplied at launch.
        poll_interval_seconds: Native and portable status polling interval.

    Raises:
        TypeError: If ``device`` does not expose ``run``.
        ValueError: If a polling value is not positive.
    """

    device: Any = field(repr=False, compare=False)
    requirements: frozenset[Requirement] = _DEVICE_CAPABILITIES
    device_snapshot: str | None = None
    poll_timeout_seconds: float = 432_000.0
    poll_interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        """Validate the thin live-device boundary."""

        if not callable(getattr(self.device, "run", None)):
            raise TypeError("device must expose run(program, ...)")
        requirements = frozenset(self.requirements)
        if any(not isinstance(item, Requirement) for item in requirements):
            raise TypeError("requirements must contain Requirement values")
        object.__setattr__(self, "requirements", requirements)
        if self.poll_timeout_seconds <= 0 or self.poll_interval_seconds <= 0:
            raise ValueError("polling durations must be positive")

    @property
    def name(self) -> str:
        """Return a stable endpoint identity for the Braket device."""

        return f"braket.device.{_device_name(self.device)}"

    @property
    def target(self) -> Braket:
        """Return the OpenQASM target sealed to this device snapshot."""

        return Braket(
            requirements=self.requirements,
            device_snapshot=self.device_snapshot or _device_snapshot(self.device),
            use_native_controls=False,
        )

    @property
    def capabilities(self) -> CapabilitySnapshot:
        """Return the endpoint's explicit capability snapshot."""

        return CapabilitySnapshot(
            FORMAT,
            self.requirements,
        )

    def launch(self, artifact: CompiledProgram, invocation: Invocation) -> Job:
        """Create one native Braket task from an inspected OpenQASM artifact."""

        if artifact.format != FORMAT:
            raise ExecutionError(
                f"Q8415: Braket device accepts {FORMAT!r}, got {artifact.format!r}"
            )
        if artifact.result_abi.width == 0:
            raise ExecutionError("Q8416: Braket device execution requires sampling")
        if invocation.seed != 0:
            raise ExecutionError(
                "Q8417: managed Braket devices do not provide a portable seed "
                "contract; use seed=0"
            )
        inputs = {
            f"p{index}": value
            for index, (_, value) in enumerate(
                scalar_parameter_bindings(
                    artifact.parameter_abi,
                    invocation.values(),
                )
            )
        }
        task = self.device.run(
            _load_program(artifact),
            shots=invocation.shots,
            inputs=inputs,
            poll_timeout_seconds=self.poll_timeout_seconds,
            poll_interval_seconds=self.poll_interval_seconds,
        )
        return _BraketJob(
            native=task,
            artifact=artifact,
            parameters=invocation.parameters,
            shots=invocation.shots,
            backend=self.name,
            poll_interval_seconds=self.poll_interval_seconds,
        )

    def restore(
        self,
        snapshot: ExecutionSnapshot | ExecutionReference,
        *,
        artifact: CompiledProgram,
        parameters: Mapping[str, object] | None = None,
    ) -> Job:
        """Attach to a snapshotted AWS task using its original decoding context.

        Args:
            snapshot (ExecutionSnapshot | ExecutionReference): Sampling reference
                captured by this endpoint, or a completed local sample snapshot.
            artifact (CompiledProgram): Exact artifact used for the original task.
            parameters (Mapping[str, object] | None): Original runtime arguments.

        Returns:
            Job: Existing task handle, with no new task submission.

        Raises:
            ValueError: If the endpoint, artifact, arguments, or reference context
                disagree, including older references without decoding metadata.
            ExecutionError: If runtime arguments violate the artifact ABI.
            ImportError: If the optional Braket SDK is unavailable.
        """

        if (
            artifact.format != FORMAT
            or artifact.target_fingerprint != self.target.fingerprint
        ):
            raise ValueError("artifact does not match this Braket target")
        if (
            isinstance(snapshot, ExecutionSnapshot)
            and snapshot.kind is ExecutionSnapshotKind.LOCAL
        ):
            result = snapshot.result
            if (
                not isinstance(result, SampleResult)
                or result.result_abi != artifact.result_abi
                or result.backend != self.name
            ):
                raise ValueError("local snapshot does not match this Braket artifact")
            return CompletedJob(f"{self.name}.restored", result)
        reference = (
            snapshot.reference if isinstance(snapshot, ExecutionSnapshot) else snapshot
        )
        if (
            not isinstance(reference, ExecutionReference)
            or reference.provider != "amazon-braket"
            or len(reference.job_ids) != 1
            or reference.target != self.name
            or reference.context_values.get("kind") != "sample"
        ):
            raise ValueError(
                "snapshot does not identify a sampling task on this Braket device"
            )
        context = reference.context_values
        if context.get("compilation_identity") != artifact.execution_identity:
            raise ValueError("artifact does not match the snapshotted Braket task")
        try:
            shots = int(context["shots"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Braket snapshot shots are invalid") from error
        if shots <= 0 or artifact.result_abi.width == 0:
            raise ValueError("Braket restoration requires positive sampling shots")
        bound = artifact.parameter_abi.bind(parameters or {})
        invocation = Invocation(FrozenMapping.from_mapping(dict(bound)), shots, 0)
        if context.get("parameter_identity") != _braket_parameter_identity(
            invocation.parameters
        ):
            raise ValueError("parameters do not match the snapshotted Braket task")
        from braket.aws import AwsQuantumTask

        native = AwsQuantumTask(
            reference.job_ids[0],
            aws_session=getattr(self.device, "aws_session", None),
            poll_timeout_seconds=self.poll_timeout_seconds,
            poll_interval_seconds=self.poll_interval_seconds,
        )
        return _BraketJob(
            native,
            artifact,
            invocation.parameters,
            shots,
            self.name,
            self.poll_interval_seconds,
        )


@dataclass(frozen=True, slots=True)
class _BraketJob:
    """Translate an AWS quantum task into the portable Job lifecycle."""

    native: Any = field(repr=False)
    artifact: CompiledProgram = field(repr=False)
    parameters: FrozenMapping = field(repr=False)
    shots: int
    backend: str
    poll_interval_seconds: float

    @property
    def job_id(self) -> str:
        """Return the AWS quantum task identity."""

        identity = getattr(self.native, "id", None)
        value = identity() if callable(identity) else identity
        if value is None:
            raise ExecutionError("Q8418: Braket task has no task identity")
        return str(value)

    @property
    def status(self) -> JobStatus:
        """Map the AWS task state to the portable lifecycle."""

        state = getattr(self.native, "state", None)
        value = state() if callable(state) else state
        name = str(value).upper()
        if name in {"CREATED", "QUEUED"}:
            return JobStatus.SUBMITTED
        if name == "RUNNING":
            return JobStatus.RUNNING
        if name == "COMPLETED":
            return JobStatus.COMPLETED
        if name == "CANCELLED":
            return JobStatus.CANCELLED
        return JobStatus.FAILED

    def result(self, timeout: float | None = None) -> SampleResult:
        """Wait for the task and decode q0-first counts with the sealed result ABI.

        Args:
            timeout (float | None): Optional portable status polling deadline.

        Returns:
            SampleResult: Typed logical counts with the original invocation shots.

        Raises:
            TimeoutError: If the polling deadline expires.
            ExecutionError: If result methods, counts, or shot totals are invalid.
        """

        if timeout is not None:
            deadline = time.monotonic() + timeout
            while self.status in {JobStatus.SUBMITTED, JobStatus.RUNNING}:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Braket task {self.job_id} did not complete")
                time.sleep(min(self.poll_interval_seconds, remaining))
        result_method = getattr(self.native, "result", None)
        if not callable(result_method):
            raise ExecutionError("Q8419: Braket task has no result() method")
        native_result = result_method()
        native_counts = getattr(native_result, "measurement_counts", None)
        if not isinstance(native_counts, Mapping):
            raise ExecutionError("Q8420: Braket task returned no measurement counts")
        logical_counts = _decode_braket_counts(
            self.artifact,
            native_counts,
            dict(self.parameters.entries),
        )
        result = SampleResult.from_counts(
            backend=self.backend,
            counts=logical_counts,
            expected_width=self.artifact.result_abi.width,
        )
        if result.shots != self.shots:
            raise ExecutionError(
                f"Q8422: Braket returned {result.shots} shots; expected {self.shots}"
            )
        return replace(result, result_abi=self.artifact.result_abi)

    def cancel(self) -> None:
        """Request best-effort cancellation from the AWS quantum task."""

        cancel = getattr(self.native, "cancel", None)
        if not callable(cancel):
            raise ExecutionError("Q8421: Braket task does not support cancellation")
        cancel()

    def snapshot(self) -> ExecutionSnapshot:
        """Capture the task ARN and checked decoding context without SDK objects.

        The artifact and launch-argument identities prevent accidental reuse of
        a different result ABI or different runtime postprocessing parameters.

        Returns:
            Secret-free remote execution reference.
        """

        return ExecutionSnapshot.remote(
            ExecutionReference.create(
                provider="amazon-braket",
                job_ids=(self.job_id,),
                target=self.backend,
                context={
                    "kind": "sample",
                    "compilation_identity": self.artifact.execution_identity,
                    "parameter_identity": _braket_parameter_identity(self.parameters),
                    "shots": str(self.shots),
                },
            )
        )


def _braket_parameter_identity(parameters: FrozenMapping) -> str:
    """Digest canonical launch arguments without persisting their values.

    Args:
        parameters (FrozenMapping): ABI-bound original runtime arguments.

    Returns:
        str: SHA-256 identity of their native-sealed portable encoding.
    """

    return hashlib.sha256(ExecutionSnapshot.local(parameters).to_bytes()).hexdigest()


def _device_name(device: Any) -> str:
    """Return one stable printable Braket device identity."""

    value = getattr(device, "name", None) or getattr(device, "arn", None)
    if value is None:
        value = type(device).__name__
    return str(value).replace(" ", "-")


def _device_snapshot(device: Any) -> str:
    """Derive a deterministic identity from a device ARN and properties."""

    payload = json.dumps(
        {
            "arn": str(getattr(device, "arn", "")),
            "name": _device_name(device),
            "properties": str(getattr(device, "properties", None)),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_program(artifact: CompiledProgram) -> Any:
    """Copy the retained OpenQASM program for one independent SDK launch.

    Args:
        artifact (CompiledProgram): Braket executable or explicitly restored artifact.

    Returns:
        Any: SDK OpenQASM Program with private launch inputs.
    """
    from ._payload import BraketExecutable

    if artifact.format != FORMAT:
        raise ExecutionError(f"Q8404: expected {FORMAT!r}, got {artifact.format!r}")
    return artifact.execution_payload(BraketExecutable).program()


def load_openqasm(artifact: CompiledProgram) -> str:
    """Read the retained Braket OpenQASM source, importing saved data once.

    Args:
        artifact (CompiledProgram): Compiled or explicitly imported Braket artifact.

    Returns:
        str: SDK OpenQASM program source.

    Raises:
        ExecutionError: If the format or saved UTF-8 encoding is invalid.
    """

    if artifact.format != FORMAT:
        raise ExecutionError(f"Q8403: expected {FORMAT!r}, got {artifact.format!r}")
    return _load_program(artifact).source


def _execute_local(
    artifact: CompiledProgram,
    invocation: Invocation,
    *,
    backend: str,
) -> SampleResult:
    """Execute OpenQASM with Braket LocalSimulator in this process.

    Args:
        artifact (CompiledProgram): Sealed OpenQASM artifact and result ABI.
        invocation (Invocation): Validated parameters, shots, and seed.
        backend (str): Braket local simulator implementation name.

    Returns:
        SampleResult: Source-ordered sample counts.

    Raises:
        ExecutionError: If the seed, artifact, or result is incompatible.
    """

    from braket.devices import LocalSimulator

    inputs = {
        f"p{index}": value
        for index, (_, value) in enumerate(
            scalar_parameter_bindings(
                artifact.parameter_abi,
                invocation.values(),
            )
        )
    }
    device: Any = LocalSimulator(backend=backend)
    program = _load_program(artifact)
    if invocation.seed != 0:
        raise ExecutionError(
            "Q8414: this Braket LocalSimulator profile does not provide a "
            "reproducible seed contract; use seed=0"
        )
    task = device.run(program, shots=invocation.shots, inputs=inputs)
    native_result: Any = task.result()
    native = native_result.measurement_counts
    logical_counts = _decode_braket_counts(
        artifact,
        native,
        invocation.values(),
    )
    return SampleResult.from_counts(
        backend=f"braket.local.{backend}",
        counts=logical_counts,
        expected_width=artifact.result_abi.width,
    )


def _decode_braket_counts(
    artifact: CompiledProgram,
    native_counts: Mapping[object, int],
    parameters: Mapping[str, object],
) -> dict[tuple[int, ...], int]:
    """Decode q0-first counts and evaluate an optional result tree."""

    plan = dict(artifact.metadata.entries).get(RESULT_POSTPROCESSING_METADATA)
    if plan is None:
        logical: dict[tuple[int, ...], int] = {}
        for bitstring, count in native_counts.items():
            bits = artifact.result_abi.decode_q0_first(str(bitstring))
            logical[bits] = logical.get(bits, 0) + int(count)
        return logical
    if not isinstance(plan, bytes):
        raise ExecutionError("Q8421: invalid Braket result postprocessing metadata")
    raw: dict[tuple[int, ...], int] = {}
    for bitstring, count in native_counts.items():
        bits = tuple(int(bit) for bit in str(bitstring).replace(" ", ""))
        raw[bits] = raw.get(bits, 0) + int(count)
    return postprocess_circuit_counts(plan, raw, parameters, artifact.result_abi)


for _public_symbol in (Braket, BraketDevice, BraketLocal, load_openqasm):
    _public_symbol.__module__ = __package__
del _public_symbol


__all__ = ["Braket", "BraketDevice", "BraketLocal", "FORMAT", "load_openqasm"]
