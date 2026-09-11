"""Expose typed jobs for direct HUGR program graphs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from functools import partial
from types import TracebackType
from typing import Any

from qamomile.circuit.transpiler import (
    CompilationMetadata,
    CompiledProgram,
    CompletedExecutionHandle,
    CompositeExecutionHandle,
    EstimationAccuracy,
    ExecutionHandle,
    ExecutionReference,
    ExecutionSnapshot,
    ExecutionSnapshotKind,
    ExpvalJob,
    Job,
    JobKind,
    JobSnapshot,
    MappedExecutionHandle,
    ProgramABI,
    RunJob,
    SampleResult,
    aggregate_typed_results,
    flatten_user_bindings,
)
from qamomile.hugr._expval import (
    ShotExpectationPlan as _ShotExpectationPlan,
    ShotExpectationWorkflow as _ShotExpectationWorkflow,
)
from qamomile.hugr._runtime import (
    PreparedExecution as _PreparedExecution,
    _runtime_input_names,
    prepare_execution as _prepare_execution,
    resolve_runtime_bindings,
)
from qamomile.hugr.execution import (
    HugrExecutor,
    _resolve_estimation_shots,
    _validate_shots,
)


def _decode_rows(
    rows: list[dict[str, Any]], prepared: _PreparedExecution, shots: int
) -> list[Any]:
    """Decode exactly the requested number of tagged shots.

    Args:
        rows (list[dict[str, Any]]): Provider output rows.
        prepared (_PreparedExecution): Output schema and submitted wrapper.
        shots (int): Expected row count.

    Returns:
        list[Any]: Typed values in shot order.

    Raises:
        ValueError: If output tags or values disagree with the ABI.
        RuntimeError: If the number of shots disagrees with the request.
    """
    if len(rows) != shots:
        raise RuntimeError(f"HUGR returned {len(rows)} shots; expected {shots}")
    return [prepared.decode_shot(row) for row in rows]


def _sample_result(values: list[Any], shots: int) -> SampleResult[Any]:
    """Aggregate typed shots in first-observed order.

    Args:
        values (list[Any]): Decoded per-shot values.
        shots (int): Requested shot count.

    Returns:
        SampleResult[Any]: Unique public values with counts.
    """
    return SampleResult(
        results=aggregate_typed_results((value, 1) for value in values), shots=shots
    )


def _single_result(values: list[Any]) -> Any:
    """Extract the result of an already validated single-shot request.

    Args:
        values (list[Any]): One decoded value.

    Returns:
        Any: Public return value.
    """
    return values[0]


def _cancel_after_submission_error(
    handle: ExecutionHandle[Any],
    error_type: type[BaseException] | None,
    error: BaseException | None,
    traceback: TracebackType | None,
) -> None:
    """Cancel a partial submission while preserving the original failure.

    Args:
        handle (ExecutionHandle[Any]): Previously submitted provider execution.
        error_type (type[BaseException] | None): Active exception type from ExitStack.
        error (BaseException | None): Active submission failure, when present.
        traceback (TracebackType | None): Original traceback supplied by ExitStack.

    Raises:
        Exception: If cancellation fails without an active submission error.
    """
    try:
        handle.cancel()
    except Exception as cleanup_error:
        # Provider SDKs expose different exception families. Suppress cleanup
        # failures only while preserving an already active submission error.
        if error is None:
            raise
        error.add_note(
            "Cancelling a previously submitted HUGR term also failed: "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )


def _identity(value: Any) -> Any:
    """Preserve a result while a mapped handle annotates its references.

    Args:
        value (Any): Scalar or structured public result.

    Returns:
        Any: Unmodified public result.
    """
    return value


def _snapshot_with_context(
    snapshot: ExecutionSnapshot, context: Mapping[str, str]
) -> ExecutionSnapshot:
    """Annotate HUGR execution leaves while preserving their term boundaries.

    Args:
        snapshot (ExecutionSnapshot): Tagged local or remote execution tree.
        context (Mapping[str, str]): Non-secret restoration fingerprints.

    Returns:
        ExecutionSnapshot: Tree carrying the additional restoration context.

    Raises:
        ValueError: If a local leaf lacks the HUGR tagged-row envelope.
    """
    if snapshot.kind is ExecutionSnapshotKind.COMPOSITE:
        return replace(
            snapshot,
            children=tuple(
                _snapshot_with_context(child, context) for child in snapshot.children
            ),
        )
    if snapshot.kind is ExecutionSnapshotKind.REMOTE:
        reference = snapshot.reference
        assert reference is not None
        return replace(
            snapshot,
            reference=replace(reference, context={**reference.context, **context}),
        )
    if (
        not isinstance(snapshot.value, dict)
        or snapshot.value.get("hugr_kind") != "tagged_rows"
    ):
        raise ValueError("A HUGR expectation snapshot requires tagged execution rows")
    return replace(snapshot, value={**snapshot.value, **context})


def _bindings_fingerprint(bindings: Mapping[str, Any]) -> str:
    """Hash normalized bindings without retaining their values in a snapshot.

    Args:
        bindings (Mapping[str, Any]): Validated, owned runtime parameter values.

    Returns:
        str: Stable digest of canonical flattened runtime parameters.

    Raises:
        TypeError: If a normalized value cannot be safely serialized.
        ValueError: If a normalized value is not finite.
    """
    data = ExecutionSnapshot(
        ExecutionSnapshotKind.LOCAL, value=flatten_user_bindings(dict(bindings))
    ).to_dict()
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


class _ConstantExpectationHandle(CompletedExecutionHandle[Any]):
    """Retain a local expectation and its reconstruction identity.

    Args:
        value (Any): Already evaluated public result.
        fingerprint (str): Complete expectation recipe identity.
        bindings_fingerprint (str): Original normalized runtime binding digest.
    """

    def __init__(self, value: Any, fingerprint: str, bindings_fingerprint: str) -> None:
        """Store an evaluated expectation with its non-secret identity.

        Args:
            value (Any): Completed public expectation result.
            fingerprint (str): Complete expectation recipe identity.
            bindings_fingerprint (str): Normalized runtime binding digest.
        """
        super().__init__(value)
        self._fingerprint = fingerprint
        self._bindings_fingerprint = bindings_fingerprint

    def snapshot(self) -> ExecutionSnapshot:
        """Serialize a local value with its program and binding identity.

        Returns:
            ExecutionSnapshot: Validated local expectation envelope.

        Raises:
            TypeError: If the public result has an unsupported value type.
            ValueError: If the public result contains non-finite values.
        """
        return ExecutionSnapshot(
            ExecutionSnapshotKind.LOCAL,
            value={
                "hugr_kind": "constant_expectation",
                "value": self._value,
                "expectation_sha256": self._fingerprint,
                "bindings_sha256": self._bindings_fingerprint,
            },
        )


class _DecodedExecutionHandle(MappedExecutionHandle[list[dict[str, Any]], list[Any]]):
    """Decode tagged results and retain their public shape contract in references.

    Args:
        source (ExecutionHandle[list[dict[str, Any]]]): Tagged provider results.
        prepared (_PreparedExecution): Submission schema.
        shots (int): Expected number of shots.
    """

    def __init__(
        self,
        source: ExecutionHandle[list[dict[str, Any]]],
        prepared: _PreparedExecution,
        shots: int,
    ) -> None:
        """Initialize decoding with the public ABI fingerprint.

        Args:
            source (ExecutionHandle[list[dict[str, Any]]]): Tagged provider handle.
            prepared (_PreparedExecution): Submission schema.
            shots (int): Expected number of shots.
        """
        super().__init__(source, partial(_decode_rows, prepared=prepared, shots=shots))
        self._abi_sha256 = prepared.abi_sha256
        self._package_sha256 = hashlib.sha256(prepared.package.to_bytes()).hexdigest()
        self._shots = shots

    def references(self) -> tuple[ExecutionReference, ...]:
        """Attach shape-aware ABI identity to the provider references.

        Returns:
            tuple[ExecutionReference, ...]: References retaining typed result shape.
        """
        return tuple(
            replace(
                reference, context={**reference.context, "abi_sha256": self._abi_sha256}
            )
            for reference in super().references()
        )

    def snapshot(self) -> ExecutionSnapshot:
        """Save raw tagged rows or an annotated remote reference without waiting.

        Returns:
            ExecutionSnapshot: Leaf retaining the public ABI and bound program.

        Raises:
            ValueError: If the source does not represent one HUGR execution.
            TypeError: If a local row contains an unsupported value type.
        """
        snapshot = self._source.snapshot()
        if snapshot.kind is ExecutionSnapshotKind.REMOTE:
            reference = snapshot.reference
            assert reference is not None
            return replace(
                snapshot,
                reference=replace(
                    reference,
                    context={**reference.context, "abi_sha256": self._abi_sha256},
                ),
            )
        if snapshot.kind is not ExecutionSnapshotKind.LOCAL:
            raise ValueError("A HUGR tagged execution must be one snapshot leaf")
        return ExecutionSnapshot(
            ExecutionSnapshotKind.LOCAL,
            value={
                "hugr_kind": "tagged_rows",
                "rows": snapshot.value,
                "abi_sha256": self._abi_sha256,
                "package_sha256": self._package_sha256,
                "shots": str(self._shots),
            },
        )


class _HugrSampleJob(Job[SampleResult[Any]]):
    """Expose shared sample results for structured program-graph outputs.

    Args:
        handle (ExecutionHandle[list[Any]]): Decoded per-shot values.
        shots (int): Number of requested samples.
    """

    def __init__(self, handle: ExecutionHandle[list[Any]], shots: int) -> None:
        """Initialize typed aggregation over a deferred handle.

        Args:
            handle (ExecutionHandle[list[Any]]): Decoded shot handle.
            shots (int): Number of requested samples.
        """
        super().__init__(
            MappedExecutionHandle(
                handle, partial(_sample_result, shots=shots), snapshot_source=True
            ),
            JobKind.SAMPLE,
            shots,
        )

    def result(self, timeout: float | None = None) -> SampleResult[Any]:
        """Wait for and return aggregated samples.

        Args:
            timeout (float | None): Maximum local wait in seconds.

        Returns:
            SampleResult[Any]: Public values and counts.

        Raises:
            Exception: If provider execution or ABI decoding fails.
        """
        return self._handle.result(timeout)


class _ExpectationExecutionHandle(MappedExecutionHandle[Any, Any]):
    """Attach the estimation recipe identity to remote restoration references.

    Args:
        handle (ExecutionHandle[Any]): Aggregated expectation execution.
        fingerprint (str): Identity of coefficients and output reconstruction.
    """

    def __init__(self, handle: ExecutionHandle[Any], fingerprint: str) -> None:
        """Initialize identity-preserving expectation reference metadata.

        Args:
            handle (ExecutionHandle[Any]): Aggregated expectation execution.
            fingerprint (str): Estimation recipe digest.
        """
        super().__init__(handle, _identity)
        self._fingerprint = fingerprint

    def references(self) -> tuple[ExecutionReference, ...]:
        """Return source references with the expectation recipe identity.

        Returns:
            tuple[ExecutionReference, ...]: References safe for typed restoration.
        """
        return tuple(
            replace(
                reference,
                context={**reference.context, "expectation_sha256": self._fingerprint},
            )
            for reference in super().references()
        )

    def snapshot(self) -> ExecutionSnapshot:
        """Attach the expectation recipe identity to every saved term.

        Returns:
            ExecutionSnapshot: Ordered term tree with reconstruction context.

        Raises:
            ValueError: If a child lacks supported HUGR restoration context.
            TypeError: If a local result cannot be safely serialized.
        """
        return _snapshot_with_context(
            self._source.snapshot(), {"expectation_sha256": self._fingerprint}
        )


class HugrExecutable:
    """Execute a HUGR artifact through a destination-independent typed API.

    ``HugrTranspiler.transpile`` also prepares shot-based expectation programs.
    Existing ``CompiledProgram[Package]`` values can be wrapped directly.

    Args:
        compiled (CompiledProgram[Any]): HUGR artifact and public ABI.
    """

    def __init__(self, compiled: CompiledProgram[Any]) -> None:
        """Own a compiled artifact and its public ABI.

        Args:
            compiled (CompiledProgram[Any]): Package, ABI and provenance.
        """
        self._compiled = replace(
            compiled, artifact=deepcopy(compiled.artifact), abi=deepcopy(compiled.abi)
        )
        self._expectation: _ShotExpectationPlan | _ShotExpectationWorkflow | None = None
        self._terms: tuple[CompiledProgram[Any], ...] = ()

    @classmethod
    def _for_expectation(
        cls,
        compiled: CompiledProgram[Any],
        expectation: _ShotExpectationPlan | _ShotExpectationWorkflow,
        terms: tuple[CompiledProgram[Any], ...],
    ) -> HugrExecutable:
        """Construct an executable for an internal Pauli measurement plan.

        Args:
            compiled (CompiledProgram[Any]): Public expectation ABI and metadata.
            expectation (_ShotExpectationPlan | _ShotExpectationWorkflow):
                Internal measurement and classical reconstruction recipe.
            terms (tuple[CompiledProgram[Any], ...]): Compiled measurement packages.

        Returns:
            HugrExecutable: Executable with native shot aggregation.
        """
        executable = cls(compiled)
        executable._expectation = expectation
        executable._terms = tuple(
            replace(term, artifact=deepcopy(term.artifact), abi=deepcopy(term.abi))
            for term in terms
        )
        return executable

    @property
    def artifact(self) -> Any:
        """Return the native package retained by the executable.

        Returns:
            Any: HUGR package, or tuple of packages for an expectation program.
        """
        return self._compiled.artifact

    @property
    def abi(self) -> ProgramABI:
        """Return a defensive copy of the public input and output contract.

        Returns:
            ProgramABI: Original qkernel ABI.
        """
        return deepcopy(self._compiled.abi)

    @property
    def metadata(self) -> CompilationMetadata:
        """Return target compilation provenance.

        Returns:
            CompilationMetadata: HUGR target and pipeline metadata.
        """
        return self._compiled.metadata

    @property
    def parameter_names(self) -> list[str]:
        """Return runtime argument names in native input-port order.

        HUGR retains whole argument ports. Array parameters therefore appear
        by their root names even when execution uses indexed binding keys.

        Returns:
            list[str]: Independent list of public runtime argument names.

        Raises:
            ValueError: If runtime input metadata disagrees with the ABI.
        """
        return list(_runtime_input_names(self._compiled))

    @property
    def has_parameters(self) -> bool:
        """Report whether execution requires runtime arguments.

        Returns:
            bool: Whether the native program has runtime input ports.

        Raises:
            ValueError: If runtime input metadata disagrees with the ABI.
        """
        return bool(self.parameter_names)

    def sample(
        self,
        executor: HugrExecutor,
        shots: int = 1024,
        bindings: Mapping[str, Any] | None = None,
    ) -> Job[SampleResult[Any]]:
        """Sample the typed program with runtime parameter values.

        Args:
            executor (HugrExecutor): Selene or Helios destination.
            shots (int): Positive number of repetitions, default ``1024``.
            bindings (Mapping[str, Any] | None): Runtime values, keyed by
                whole argument names or shared indexed parameter names.

        Returns:
            Job[SampleResult[Any]]: Deferred structured sample counts.

        Raises:
            ValueError: If shots, bindings, or operation are invalid.
            Exception: If compilation or provider submission fails.
        """
        _validate_shots(shots)
        if self._expectation is not None:
            raise ValueError("Expectation programs use run(..., estimation=...)")
        prepared = _prepare_execution(self._compiled, bindings)
        handle = _DecodedExecutionHandle(
            executor.submit(prepared.package, shots), prepared, shots
        )
        return _HugrSampleJob(handle, shots)

    def run(
        self,
        executor: HugrExecutor,
        bindings: Mapping[str, Any] | None = None,
        *,
        shots: int | None = None,
        estimation: EstimationAccuracy | None = None,
    ) -> RunJob[Any] | ExpvalJob:
        """Execute one ordinary shot or compute a program's expectations.

        For an expectation, ``ShotBased`` specifies measurements per
        nonidentity Pauli term; total device shots scale with that term count.

        Args:
            executor (HugrExecutor): Selene or Helios destination.
            bindings (Mapping[str, Any] | None): Whole or indexed runtime
                parameter values.
            shots (int | None): Legacy positive shots per expectation term.
                Mutually exclusive with estimation; both omitted uses 1024.
            estimation (EstimationAccuracy | None): Shared accuracy policy.
                Only ShotBased is supported for expectations. Ordinary
                programs ignore estimation and execute exactly one shot.

        Returns:
            RunJob[Any] | ExpvalJob: Typed single return or estimated expectation.

        Raises:
            ValueError: If runtime values or shots are invalid.
            TypeError: If an expectation accuracy policy is unrecognized.
            NotImplementedError: If exact or target-precision estimation is requested.
            Exception: If target compilation or provider submission fails.
        """
        if self._expectation is None:
            if shots is not None:
                _validate_shots(shots)
            prepared = _prepare_execution(self._compiled, bindings)
            decoded = _DecodedExecutionHandle(
                executor.submit(prepared.package, 1), prepared, 1
            )
            return RunJob.from_handle(
                MappedExecutionHandle(decoded, _single_result, snapshot_source=True)
            )
        shots = _resolve_estimation_shots(estimation, shots)
        runtime = resolve_runtime_bindings(self._compiled, bindings)
        reconstruct = partial(self._aggregate_expectation, bindings=runtime)
        prepared_terms = [_prepare_execution(term, runtime) for term in self._terms]
        if not prepared_terms:
            return self._expectation_job(
                _ConstantExpectationHandle(
                    reconstruct([]),
                    self._expectation_fingerprint(),
                    _bindings_fingerprint(runtime),
                )
            )
        fingerprint = self._expectation_fingerprint()
        handles = []
        with ExitStack() as cleanup:
            for prepared in prepared_terms:
                raw = executor.submit(prepared.package, shots)
                cleanup.push(partial(_cancel_after_submission_error, raw))
                handles.append(_DecodedExecutionHandle(raw, prepared, shots))
            cleanup.pop_all()
        aggregate = CompositeExecutionHandle(handles)
        return self._expectation_job(
            _ExpectationExecutionHandle(
                MappedExecutionHandle(aggregate, reconstruct, snapshot_source=True),
                fingerprint,
            )
        )

    def restore(
        self,
        executor: HugrExecutor,
        snapshot: JobSnapshot,
        bindings: Mapping[str, Any] | None = None,
    ) -> Job[Any]:
        """Restore a typed local or remote job using the same compiled program.

        Args:
            executor (HugrExecutor): Destination used to retrieve remote leaves.
            snapshot (JobSnapshot): Saved operation and ordered execution tree.
            bindings (Mapping[str, Any] | None): Original runtime parameter values.

        Returns:
            Job[Any]: Restored sample, single-run, or expectation job.

        Raises:
            ValueError: If the execution layout, identity, or operation is invalid.
            Exception: If runtime validation or provider retrieval fails.
        """
        snapshot = JobSnapshot.from_dict(snapshot.to_dict())
        compiled_terms = (
            self._terms if self._expectation is not None else (self._compiled,)
        )
        if self._expectation is not None and snapshot.kind is not JobKind.RUN:
            raise ValueError("An expectation snapshot must describe run()")
        runtime = resolve_runtime_bindings(self._compiled, bindings)
        if self._expectation is not None and not compiled_terms:
            return self._restore_constant_expectation(snapshot, runtime)
        if snapshot.execution is None:
            nodes = tuple(
                ExecutionSnapshot(ExecutionSnapshotKind.REMOTE, reference=reference)
                for reference in snapshot.executions
            )
        elif self._expectation is not None:
            if snapshot.execution.kind is not ExecutionSnapshotKind.COMPOSITE:
                raise ValueError(
                    "A HUGR expectation snapshot must retain its term group"
                )
            nodes = snapshot.execution.children
        else:
            nodes = (snapshot.execution,)
        if len(nodes) != len(compiled_terms) or any(
            node.kind not in {ExecutionSnapshotKind.LOCAL, ExecutionSnapshotKind.REMOTE}
            for node in nodes
        ):
            raise ValueError(
                "Snapshot execution layout does not match this HUGR program"
            )
        prepared_terms = [_prepare_execution(term, runtime) for term in compiled_terms]
        decoded = []
        for prepared, node in zip(prepared_terms, nodes, strict=True):
            context: Mapping[str, Any]
            if node.kind is ExecutionSnapshotKind.REMOTE:
                assert node.reference is not None
                context = node.reference.context
            else:
                context = node.value
                if (
                    not isinstance(context, dict)
                    or context.get("hugr_kind") != "tagged_rows"
                ):
                    raise ValueError(
                        "Invalid HUGR local snapshot: expected tagged rows"
                    )
                expected_fields = {
                    "hugr_kind",
                    "rows",
                    "abi_sha256",
                    "package_sha256",
                    "shots",
                }
                if self._expectation is not None:
                    expected_fields.add("expectation_sha256")
                if context.keys() != expected_fields:
                    raise ValueError("Invalid HUGR local snapshot context fields")
            if (
                self._expectation is not None
                and context.get("expectation_sha256") != self._expectation_fingerprint()
            ):
                raise ValueError(
                    "Snapshot expectation recipe does not match this HUGR program"
                )
            if context.get("abi_sha256") != prepared.abi_sha256:
                raise ValueError("Snapshot public ABI does not match this HUGR program")
            fingerprint = hashlib.sha256(prepared.package.to_bytes()).hexdigest()
            if context.get("package_sha256") != fingerprint:
                raise ValueError(
                    "Snapshot does not match this HUGR program and runtime bindings"
                )
            try:
                raw_shots = context["shots"]
                if not isinstance(raw_shots, str):
                    raise ValueError("Snapshot shots must be encoded as a string")
                shots = int(raw_shots)
                _validate_shots(shots)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Snapshot shots must be a positive integer") from error
            if self._expectation is None:
                expected_shots = (
                    snapshot.shots if snapshot.kind is JobKind.SAMPLE else 1
                )
                if shots != expected_shots:
                    raise ValueError("Snapshot shot count disagrees with its operation")
            if node.kind is ExecutionSnapshotKind.LOCAL:
                rows = node.value.get("rows")
                if not isinstance(rows, list) or any(
                    not isinstance(row, dict) for row in rows
                ):
                    raise ValueError(
                        "Invalid HUGR local snapshot: expected tagged rows"
                    )
                _decode_rows(rows, prepared, shots)
            decoded.append((node, prepared, shots))
        handles = [
            _DecodedExecutionHandle(
                executor.retrieve(node.reference)
                if node.kind is ExecutionSnapshotKind.REMOTE
                and node.reference is not None
                else CompletedExecutionHandle(node.value["rows"]),
                prepared,
                shots,
            )
            for node, prepared, shots in decoded
        ]
        if self._expectation is not None:
            return self._expectation_job(
                _ExpectationExecutionHandle(
                    MappedExecutionHandle(
                        CompositeExecutionHandle(handles),
                        partial(self._aggregate_expectation, bindings=runtime),
                        snapshot_source=True,
                    ),
                    self._expectation_fingerprint(),
                )
            )
        if snapshot.kind is JobKind.SAMPLE:
            assert snapshot.shots is not None
            return _HugrSampleJob(handles[0], snapshot.shots)
        return RunJob.from_handle(
            MappedExecutionHandle(handles[0], _single_result, snapshot_source=True)
        )

    def _restore_constant_expectation(
        self, snapshot: JobSnapshot, bindings: Mapping[str, Any]
    ) -> RunJob[Any] | ExpvalJob:
        """Validate and restore an expectation requiring no provider execution.

        Args:
            snapshot (JobSnapshot): Saved local expectation envelope.
            bindings (Mapping[str, Any]): Validated original runtime parameters.

        Returns:
            RunJob[Any] | ExpvalJob: Completed typed expectation job.

        Raises:
            ValueError: If the recipe, bindings, or stored value is incompatible.
            TypeError: If a stored value cannot be safely serialized.
        """
        node = snapshot.execution
        if (
            node is None
            or node.kind is not ExecutionSnapshotKind.LOCAL
            or not isinstance(node.value, dict)
        ):
            raise ValueError("A constant HUGR expectation requires a local snapshot")
        value = node.value
        if value.get("hugr_kind") != "constant_expectation":
            raise ValueError("Invalid constant HUGR expectation snapshot")
        fingerprint = self._expectation_fingerprint()
        if value.get("expectation_sha256") != fingerprint:
            raise ValueError(
                "Snapshot expectation recipe does not match this HUGR program"
            )
        bindings_fingerprint = _bindings_fingerprint(bindings)
        if value.get("bindings_sha256") != bindings_fingerprint:
            raise ValueError(
                "Snapshot does not match this HUGR program and runtime bindings"
            )
        handle = _ConstantExpectationHandle(
            self._aggregate_expectation([], bindings), fingerprint, bindings_fingerprint
        )
        if node.to_dict() != handle.snapshot().to_dict():
            raise ValueError("Constant HUGR snapshot value does not match its recipe")
        return self._expectation_job(handle)

    def _aggregate_expectation(
        self, rows: list[Any], bindings: Mapping[str, Any]
    ) -> Any:
        """Reconstruct the original public result from measured Pauli batches.

        Args:
            rows (list[Any]): Measurement batches in submission order.
            bindings (Mapping[str, Any]): Owned normalized runtime arguments.

        Returns:
            Any: Pure expectation or classically processed public result.

        Raises:
            ValueError: If provider measurement batches are malformed.
            ExecutionError: If classical output computation fails.
        """
        plan = self._expectation
        assert plan is not None
        if isinstance(plan, _ShotExpectationWorkflow):
            return plan.aggregate(rows, bindings)
        return plan.aggregate(rows)

    def _expectation_job(self, handle: ExecutionHandle[Any]) -> RunJob[Any] | ExpvalJob:
        """Preserve scalar compatibility while returning typed workflow results.

        Args:
            handle (ExecutionHandle[Any]): Deferred public result computation.

        Returns:
            RunJob[Any] | ExpvalJob: Structured workflow or pure scalar job.
        """
        if isinstance(self._expectation, _ShotExpectationWorkflow):
            return RunJob.from_handle(handle)
        return ExpvalJob(handle)

    def _expectation_fingerprint(self) -> str:
        """Identify the complete estimator recipe independently of native shots.

        Returns:
            str: Stable digest of estimates and public result computation.
        """
        plan = self._expectation
        assert plan is not None
        recipe = (
            plan.recipe()
            if isinstance(plan, _ShotExpectationWorkflow)
            else (
                plan.coefficients,
                plan.parity_indices,
                plan.constant,
                plan.num_qubits,
            )
        )
        return hashlib.sha256(json.dumps(recipe, allow_nan=False).encode()).hexdigest()
