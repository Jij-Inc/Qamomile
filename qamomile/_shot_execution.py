"""Transport Rust-owned correlated shot plans through existing endpoints.

Dependence, placement, scalar evaluation, histogram transitions, and transport
ABIs are native compiler facts. This module only emits and submits checked
children; it has no Python implementation of those semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from threading import Event, RLock
from time import monotonic
from typing import Any
from uuid import uuid4

from ._compiler_optimization import OPTIMIZATION_METADATA_KEY, _restore_optimization
from ._execution_payload import ExecutablePayload
from ._native import compiler as native
from .compilation import (
    CircuitResultPostprocessingTarget,
    CircuitTarget,
    CompiledProgram,
    ExecutionContract,
    ExecutionEndpoint,
    ExecutionSnapshot,
    Invocation,
    Job,
    JobStatus,
    LoweredProgram,
    ParameterABI,
    ResultABI,
    SampleResult,
    _ValidatedJob,
    compilation_report,
)
from .effects import Requirement
from .errors import ArtifactError, ExecutionError, QamomileError
from .ir import FrozenMapping, Program
from .proto import qcore_pb2 as wire
from .resources.report import ResourceStudy
from .serialization import decode_type

FORMAT = "qamomile.shot-execution.v1"
INTERFACE_KEY = "qamomile.shot-segment-interface"


@dataclass(frozen=True, slots=True)
class ShotExecutable(ExecutablePayload):
    """Retain a verified source recipe for native conditional shot execution.

    Args:
        program (Program): Immutable source authority.
        target_format (str): Child SDK format expected from the endpoint.
    """

    program: Program
    target_format: str

    def to_bytes(self) -> bytes:
        """Seal the canonical source recipe through Rust.

        Returns:
            bytes: Portable recipe with compiler-reconstructed cut proofs.

        Raises:
            ArtifactError: If native persistence is missing or rejects the recipe.
        """
        seal = getattr(native, "seal_shot_execution_recipe", None)
        if not callable(seal):
            raise ArtifactError("Q4470: native shot recipe persistence is unavailable")
        try:
            return bytes(seal(self.target_format, self.program.to_bytes()))
        except (TypeError, ValueError) as error:
            raise ArtifactError(str(error)) from error

    @classmethod
    def from_bytes(cls, payload: bytes) -> ShotExecutable:
        """Restore a source recipe after native structural and semantic checks.

        Args:
            payload (bytes): Canonical plan recipe.

        Returns:
            ShotExecutable: Immutable verified recipe.

        Raises:
            ArtifactError: If persistence is unavailable or the recipe is invalid.
        """
        inspect = getattr(native, "inspect_shot_execution_recipe", None)
        if not callable(inspect):
            raise ArtifactError("Q4470: native shot recipe persistence is unavailable")
        try:
            target, source = inspect(payload)
            return cls(Program.from_bytes(bytes(source)), target)
        except (TypeError, ValueError) as error:
            raise ArtifactError(str(error)) from error


def compile_shot_program(
    program: Program,
    target: CircuitTarget,
    required: frozenset[Requirement],
    study: ResourceStudy,
    planning_stage: str,
) -> CompiledProgram:
    """Adopt the Rust-selected segmented route without enumerating outcomes.

    Args:
        program (Program): Verified original source.
        target (CircuitTarget): Exact target profile for all children.
        required (frozenset[Requirement]): Native residual requirements.
        study (ResourceStudy): Original logical resource evidence.
        planning_stage (str): Native planning identity.

    Returns:
        CompiledProgram: Reusable source-bound execution plan.
    """
    return CompiledProgram(
        payload=ShotExecutable(program, target.name),
        target_fingerprint=target.fingerprint,
        parameter_abi=ParameterABI.from_program(program),
        result_abi=ResultABI.from_program(program),
        report=compilation_report(
            program,
            target=target.name,
            stages=(planning_stage, "rust-closed-shot-execution.v1"),
            assumptions=(
                "logical resources describe one source trajectory; conditional physical work is recorded at execution",
            ),
            resources=study,
        ),
        execution_contract=ExecutionContract(FORMAT, required),
        metadata=FrozenMapping.from_mapping({"target_format": target.name}),
    )


@dataclass(slots=True)
class ShotJob:
    """Coordinate checked conditional children without resubmitting outcomes.

    Args:
        artifact (CompiledProgram): Native-selected reusable recipe.
        endpoint (ExecutionEndpoint): Target-matched child submission endpoint.
        invocation (Invocation): Validated outer launch values and shot demand.
    """

    artifact: CompiledProgram
    endpoint: ExecutionEndpoint
    invocation: Invocation
    _state: Any = field(init=False, repr=False)
    _child: Job | None = field(default=None, init=False, repr=False)
    _result: SampleResult | None = field(default=None, init=False, repr=False)
    _error: BaseException | None = field(default=None, init=False, repr=False)
    _cancelled: Event = field(default_factory=Event, init=False, repr=False)
    _lock: Any = field(default_factory=RLock, init=False, repr=False)
    _identity: str = field(default_factory=lambda: str(uuid4()), init=False)
    _submission_uncertain: bool = field(default=False, init=False, repr=False)
    _program: Program = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Create native invocation state before any provider submission.

        Raises:
            ExecutionError: If native planning or the endpoint profile is invalid.
        """
        from .circuit import _native_launch_value, _realization_labels

        handle = getattr(native, "ShotExecutionHandle", None)
        if not callable(handle):
            raise ExecutionError("Q4470: native shot execution is unavailable")
        target = self.endpoint.target
        if not isinstance(target, CircuitTarget):
            raise ExecutionError("Q4470: shot execution requires a circuit endpoint")
        if self.artifact.format != FORMAT:
            raise ExecutionError(
                "Q4470: shot recipe has an incompatible artifact format"
            )
        owner = self.artifact.execution_payload(ShotExecutable)
        self._program = owner.program
        receipt = dict(self.artifact.metadata.entries).get(OPTIMIZATION_METADATA_KEY)
        if receipt is not None:
            source = self.artifact.report.resources._semantic_source()
            if not isinstance(receipt, bytes) or not isinstance(source, Program):
                raise ExecutionError("Q7802: invalid optimization recipe")
            self._program = _restore_optimization(source, self._program, receipt)
        if (
            self._program._native_program().compilation_source_digest()
            != self.artifact.source_digest
        ):
            raise ExecutionError("Q4470: shot recipe disagrees with its source")
        try:
            self._state = handle(
                self._program._native_program(),
                {
                    name: _native_launch_value(value)
                    for name, value in self.invocation.values().items()
                },
                self.invocation.shots,
                self.invocation.seed,
                _realization_labels(target.capabilities.realization_capabilities),
                target.capabilities.realizations.overrides,
                self.endpoint.capabilities.seeded_execution,
            )
        except (TypeError, ValueError) as error:
            raise ExecutionError(str(error)) from error
        try:
            request = self._state.next_request()
            if request is not None:
                self._child = self._launch_child(request)
        except TimeoutError as error:
            raise ExecutionError(
                "Q4470: initial provider submission outcome is uncertain; resolve the original job before retrying"
            ) from error
        except (QamomileError, TypeError, ValueError, RuntimeError) as error:
            raise ExecutionError(
                f"Q4470: initial shot submission failed: {error}"
            ) from error

    @property
    def job_id(self) -> str:
        """Return the source-bound outer execution identity."""
        return f"shot:{self._identity}"

    @property
    def status(self) -> JobStatus:
        """Return completion, failure, cancellation, or child progress."""
        if self._result is not None:
            return JobStatus.COMPLETED
        if self._cancelled.is_set():
            return JobStatus.CANCELLED
        if self._error is not None:
            return JobStatus.FAILED
        return JobStatus.RUNNING if self._child is not None else JobStatus.SUBMITTED

    @property
    def statistics(self) -> FrozenMapping:
        """Return native physical demand independently of logical shot counts."""
        return FrozenMapping.from_mapping(
            {
                "logical_shots": self.invocation.shots,
                "physical_segment_shots": self._state.physical_shots,
                "submissions": self._state.submissions,
            }
        )

    def result(self, timeout: float | None = None) -> SampleResult:
        """Execute pending populations and return their native joint result.

        Args:
            timeout (float | None): Total wait budget for this retrieval.

        Returns:
            SampleResult: Exactly the outer logical shot population.

        Raises:
            ExecutionError: If a child, native transition, or result fails.
            TimeoutError: If the wait budget expires; the child is retained.
        """
        deadline = None if timeout is None else monotonic() + timeout
        with self._lock:
            if self._cancelled.is_set():
                raise ExecutionError("Q4470: shot execution was cancelled")
            if self._error is not None:
                raise ExecutionError(
                    "Q4470: shot execution previously failed"
                ) from self._error
            if self._result is not None:
                return self._result
            try:
                while True:
                    if self._cancelled.is_set():
                        raise ExecutionError("Q4470: shot execution was cancelled")
                    remaining = None if deadline is None else deadline - monotonic()
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError("shot execution retrieval timed out")
                    if self._child is None:
                        request = self._state.next_request()
                        if request is None:
                            break
                        self._child = self._launch_child(request)
                    remaining = None if deadline is None else deadline - monotonic()
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError("shot execution retrieval timed out")
                    result = self._child.result(remaining)
                    if not isinstance(result, SampleResult):
                        raise ExecutionError(
                            "Q4470: raw quantum segment returned a value"
                        )
                    self._state.accept(
                        [(bytes(row), count) for row, count in result._counts]
                    )
                    self._child = None
                counts = {tuple(row): count for row, count in self._state.finish()}
                self._result = replace(
                    SampleResult.from_counts(
                        backend=self.endpoint.target.name,
                        counts=counts,
                        expected_width=self.artifact.result_abi.width,
                    ),
                    result_abi=self.artifact.result_abi,
                    execution_statistics=self.statistics,
                )
                return self._result
            except TimeoutError:
                if self._submission_uncertain:
                    self._error = ExecutionError(
                        "Q4470: provider submission outcome is uncertain; resolve the original job before continuing"
                    )
                    raise self._error
                raise
            except (QamomileError, TypeError, ValueError, RuntimeError) as error:
                self._error = error
                raise ExecutionError(
                    f"Q4470: conditional shot execution failed: {error}"
                ) from error

    def _launch_child(self, request: Any) -> Job:
        """Emit and submit one native raw-measurement child.

        Args:
            request (Any): Checked native circuit, ABI, population and stream.

        Returns:
            Job: Retained child enforcing the raw transport result ABI.

        Raises:
            ExecutionError: If emission or the endpoint changes native facts.
        """
        from .circuit import _circuit_from_native_projection

        encoded = wire.ResultABIProto.FromString(request.result_abi)
        abi = ResultABI(
            decode_type(encoded.type), tuple(encoded.source_order), encoded.native_width
        )
        circuit = _circuit_from_native_projection(
            self._program,
            request.circuit,
            semantic_id=self._program.semantic_digest,
            parameter_abi=self.artifact.parameter_abi,
            result_abi=abi,
        )
        target = self.endpoint.target
        if not isinstance(target, CircuitTarget):
            raise ExecutionError("Q4470: endpoint lost its circuit profile")
        if circuit.requires_result_postprocessing and not (
            isinstance(target, CircuitResultPostprocessingTarget)
            and target.supports_circuit_result_postprocessing is True
        ):
            raise ExecutionError(
                "Q4470: target cannot normalize inactive measurement slots"
            )
        lowered = target.lower_circuit(circuit)
        if (
            not isinstance(lowered, LoweredProgram)
            or lowered.result_abi != abi
            or lowered.parameter_abi != self.artifact.parameter_abi
        ):
            raise ExecutionError(
                "Q4470: segment emitter changed its native transport ABI"
            )
        metadata = dict(lowered.metadata.entries)
        if INTERFACE_KEY in metadata:
            raise ExecutionError("Q4470: target metadata shadows the native interface")
        metadata[INTERFACE_KEY] = bytes(request.interface)
        optimization = dict(self.artifact.metadata.entries).get(
            OPTIMIZATION_METADATA_KEY
        )
        if optimization is not None:
            metadata[OPTIMIZATION_METADATA_KEY] = optimization
        child = CompiledProgram(
            payload=lowered.payload,
            target_fingerprint=self.artifact.target_fingerprint,
            parameter_abi=lowered.parameter_abi,
            result_abi=lowered.result_abi,
            report=self.artifact.report,
            execution_contract=ExecutionContract(
                target.name, self.artifact.execution_contract.requirements
            ),
            metadata=FrozenMapping.from_mapping(metadata),
        )
        self.endpoint.capabilities.require(child.execution_contract)
        invocation = Invocation(self.invocation.parameters, request.shots, request.seed)
        if self._cancelled.is_set():
            raise ExecutionError("Q4470: shot execution was cancelled")
        self._submission_uncertain = True
        job = self.endpoint.launch(child, invocation)
        if not isinstance(job, Job):
            raise ExecutionError("Q8128: segment endpoint did not return a Job")
        self._submission_uncertain = False
        if self._cancelled.is_set():
            job.cancel()
            raise ExecutionError("Q4470: shot execution was cancelled")
        return _ValidatedJob(job, child, request.shots)

    def cancel(self) -> None:
        """Cancel the active child and prohibit further submissions."""
        if self._result is not None:
            return
        self._cancelled.set()
        if self._child is not None:
            self._child.cancel()

    def snapshot(self) -> ExecutionSnapshot:
        """Export completion; reject unsupported in-flight persistence explicitly.

        Returns:
            ExecutionSnapshot: Portable completed logical result.

        Raises:
            ExecutionError: If conditional execution is still in progress.
        """
        if self._result is None:
            raise ExecutionError(
                "Q4470: in-flight shot continuation snapshots are not yet supported"
            )
        return ExecutionSnapshot.completed(self._result)
