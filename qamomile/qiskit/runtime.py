"""IBM Runtime sampling and provider validation for Qiskit artifacts."""

from __future__ import annotations

import importlib
import inspect
import json
import math
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Any, cast

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
    ValidationResult,
)
from qamomile.effects import Requirement
from qamomile.errors import ExecutionError
from qamomile.ir import FrozenMapping
from qamomile.local_execution import sample_job_id

from ._indexed import (
    IndexedChildren,
    IndexedMaterializationGuard,
    indexed_circuits,
    indexed_result,
    require_batch_capacity,
)
from ._integration import (
    FORMAT,
    Qiskit,
    _backend_name,
    _backend_requirements,
    bind_circuit,
    decode_counts,
    load_circuit,
)
from ._shot_template import MeasurementFreeShots

_STATUS_MAP: Mapping[str, JobStatus] = {
    "INITIALIZING": JobStatus.SUBMITTED,
    "VALIDATING": JobStatus.SUBMITTED,
    "QUEUED": JobStatus.SUBMITTED,
    "RUNNING": JobStatus.RUNNING,
    "CANCELLING": JobStatus.RUNNING,
    "CANCELLED": JobStatus.CANCELLED,
    "DONE": JobStatus.COMPLETED,
    "ERROR": JobStatus.FAILED,
}


@dataclass(frozen=True, slots=True)
class IBMRuntime:
    """Execute routed QPY artifacts with IBM Runtime Sampler V2.

    Args:
        backend (Any): Explicit IBM BackendV2 used for routing and execution.
        mode (Any | None): Optional caller-owned Runtime Session or Batch.
        sampler_options (Any | None): Optional native Runtime sampler options.
        service (Any | None): Runtime service used for restoration. Defaults to
            ``backend.service`` when available.
        requirements (frozenset[Requirement] | None): Explicit capabilities.
            ``None`` derives a conservative set from the backend target.
        backend_snapshot (str | None): Optional immutable backend identity.
        optimization_level (int): Qiskit transpiler optimization level.
        seed_transpiler (int): Deterministic transpiler seed.
        scheduling_method (str | None): Optional Qiskit scheduling method.
        dry_run (bool): Request IBM cloud validation and usage estimates instead
            of measurements. Requires Runtime 0.51 or later and job or Batch
            mode. Defaults to False.

    Raises:
        ImportError: If the optional IBM Runtime SDK is unavailable.
        TypeError: If the backend, dry-run flag, or SDK surface is incompatible.
        ValueError: If Runtime mode and compilation backend differ, or dry-run
            requests local or Session execution, or a cloud mock backend is
            selected without explicit dry-run.
    """

    backend: Any = field(repr=False, compare=False)
    mode: Any | None = field(default=None, repr=False, compare=False)
    sampler_options: Any | None = field(default=None, repr=False, compare=False)
    service: Any | None = field(default=None, repr=False, compare=False)
    requirements: frozenset[Requirement] | None = None
    backend_snapshot: str | None = None
    optimization_level: int = 1
    seed_transpiler: int = 7
    scheduling_method: str | None = None
    dry_run: bool = field(default=False, kw_only=True)
    _sampler: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Load Runtime lazily and validate the paired target configuration.

        Raises:
            ImportError: If ``qiskit-ibm-runtime`` is not installed.
            TypeError: If the backend, dry-run flag, or SDK surface is incompatible.
            ValueError: If Runtime mode selects a different backend or dry-run
                requests local or Session execution, or a cloud mock backend
                is selected without explicit dry-run.
        """

        if self.backend is None:
            raise ValueError("IBM Runtime requires an explicit backend")
        if type(self.dry_run) is not bool:
            raise TypeError("dry_run must be a bool")
        runtime = _runtime_module()
        sampler_type = getattr(runtime, "SamplerV2", None)
        if not callable(sampler_type):
            raise TypeError("qiskit_ibm_runtime must expose SamplerV2")
        backend_type = getattr(runtime, "IBMBackend", None)
        cloud_backend = isinstance(backend_type, type) and isinstance(
            self.backend, backend_type
        )
        if (
            cloud_backend
            and getattr(self.backend, "is_mock", False) is True
            and not self.dry_run
        ):
            raise ValueError("IBM cloud mock backends require explicit dry_run=True")
        if self.dry_run:
            if not cloud_backend:
                raise ValueError("IBM Runtime dry-run requires an IBM cloud backend")
            _require_dry_run_mode(runtime, self.mode)
            if not _accepts_keyword(getattr(sampler_type, "run", None), "dry_run"):
                raise TypeError(
                    "IBM Runtime dry-run requires qiskit-ibm-runtime>=0.51 "
                    "with SamplerV2.run(dry_run=...)"
                )
        if self.requirements is not None:
            requirements = frozenset(self.requirements)
            if any(not isinstance(item, Requirement) for item in requirements):
                raise TypeError("requirements must contain Requirement values")
            object.__setattr__(self, "requirements", requirements)
        sampler = sampler_type(
            mode=self.backend if self.mode is None else self.mode,
            options=self.sampler_options,
        )
        if self.dry_run:
            _require_dry_run_mode(runtime, getattr(sampler, "mode", None))
        selected_backend = getattr(sampler, "backend", None)
        selected = selected_backend() if callable(selected_backend) else None
        if selected is not None and _backend_name(selected) != _backend_name(
            self.backend
        ):
            raise ValueError("Runtime mode and compilation backend must match")
        object.__setattr__(self, "_sampler", sampler)
        if self.service is None:
            object.__setattr__(self, "service", getattr(self.backend, "service", None))
        _ = self.target

    @property
    def name(self) -> str:
        """Return the provider and backend scoped endpoint name."""

        return f"qiskit.runtime.{_backend_name(self.backend)}"

    @property
    def validation_only(self) -> bool:
        """Select provider validation reports for explicitly requested dry runs.

        Returns:
            bool: Whether jobs return ValidationResult instead of measurements.
        """
        return self.dry_run

    def _submit_sampler(self, pubs: Sequence[Any], *, shots: int) -> Any:
        """Submit once with the endpoint's explicit provider execution policy.

        Args:
            pubs (Sequence[Any]): Prepared Runtime primitive inputs.
            shots (int): Positive provider observations per input.

        Returns:
            Any: The existing SDK job returned by the single submission.

        Raises:
            TypeError: If the SDK cannot expose the current Runtime context.
            ValueError: If an unsupported Session became active before launch.
        """
        if self.dry_run:
            mode = getattr(self._sampler, "mode", None)
            _require_dry_run_mode(
                _runtime_module(),
                _runtime_context_mode() if mode is None else mode,
            )
            return self._sampler.run(pubs, shots=shots, dry_run=True)
        return self._sampler.run(pubs, shots=shots)

    @property
    def target(self) -> Qiskit:
        """Return the exact routed-QPY target accepted by this endpoint."""

        return Qiskit(
            optimization_level=self.optimization_level,
            seed_transpiler=self.seed_transpiler,
            scheduling_method=self.scheduling_method,
            backend=self.backend,
            backend_snapshot=self.backend_snapshot,
            requirements=self._requirements,
        )

    @property
    def capabilities(self) -> CapabilitySnapshot:
        """Return the endpoint's explicit capability snapshot."""

        return CapabilitySnapshot(FORMAT, self._requirements)

    @property
    def _requirements(self) -> frozenset[Requirement]:
        """Return explicit capabilities or conservative backend inference."""

        if self.requirements is not None:
            return self.requirements
        return _backend_requirements(self.backend)

    def _indexed_materialization_guard(self) -> IndexedMaterializationGuard:
        """Bound retained child circuits before native Runtime submission.

        Returns:
            IndexedMaterializationGuard: Shared transport guard for this backend.
        """
        return IndexedMaterializationGuard(self.backend)

    def validate_invocation(self, invocation: Invocation) -> None:
        """Reject unsupported seed policy before any batch member is submitted.

        Args:
            invocation (Invocation): Canonically bound public execution request.

        Raises:
            ExecutionError: If an explicit portable seed is requested.
        """
        if invocation.seed is not None:
            raise ExecutionError(
                "Q8224: IBM Runtime does not provide a portable seed contract; use seed=None"
            )

    def launch(self, artifact: CompiledProgram, invocation: Invocation) -> Job:
        """Submit one routed and fully bound SDK circuit asynchronously.

        Args:
            artifact (CompiledProgram): Live circuit compiled for :attr:`target`.
            invocation (Invocation): Rust-normalized parameters and shot policy.

        Returns:
            Job: Nonblocking Runtime job with cancellation and snapshots.

        Raises:
            ExecutionError: If the request is not portable to Runtime sampling,
                or dry-run contains no quantum cloud workload.
        """

        if artifact.format != FORMAT:
            raise ExecutionError(
                f"Q8222: IBM Runtime accepts {FORMAT!r}, got {artifact.format!r}"
            )
        if artifact.result_abi.width == 0:
            raise ExecutionError("Q8223: IBM Runtime endpoint requires sampling")
        self.validate_invocation(invocation)
        circuit = bind_circuit(load_circuit(artifact), artifact, invocation)
        bit_map = _classical_bit_map(circuit)
        if self.dry_run and not circuit.num_qubits:
            raise ExecutionError("Q8232: dry-run requires a quantum cloud workload")
        native = self._submit_sampler([circuit], shots=invocation.shots)
        reference = ExecutionReference.create(
            provider="qiskit-runtime",
            job_ids=(_job_id(native),),
            target=_backend_name(self.backend),
            context={
                "compilation_identity": artifact.execution_identity,
                "bits": json.dumps(bit_map, separators=(",", ":")),
                "kind": "sample",
                "shots": str(invocation.shots),
                **({"dry_run": "true"} if self.dry_run else {}),
            },
        )
        return _IBMRuntimeJob(
            native,
            circuit,
            artifact,
            invocation.parameters,
            invocation.shots,
            self.name,
            bit_map,
            reference,
            dry_run=self.dry_run,
        )

    def restore(
        self,
        snapshot: ExecutionSnapshot | ExecutionReference,
        *,
        artifact: CompiledProgram,
        parameters: Mapping[str, object] | None = None,
    ) -> Job:
        """Restore one existing Runtime job without rerouting or resubmitting.

        Args:
            snapshot (ExecutionSnapshot | ExecutionReference): Saved provider job.
            artifact (CompiledProgram): Compiled program with the original
                execution identity.
            parameters (Mapping[str, object] | None): Original launch parameters.

        Returns:
            Job: Handle for the existing provider job.

        Raises:
            NotImplementedError: If no restoration service is configured.
            ValueError: If the reference does not match this endpoint, artifact,
                or explicitly selected dry-run policy.
            ExecutionError: If launch parameters do not satisfy the artifact ABI.
        """

        if artifact.target_fingerprint != self.target.fingerprint:
            raise ValueError("artifact does not match this Runtime target")
        if isinstance(snapshot, ExecutionSnapshot) and (
            snapshot.kind is ExecutionSnapshotKind.LOCAL
        ):
            result = snapshot.result
            if isinstance(result, ValidationResult):
                metadata = result.metadata()
                if (
                    not self.dry_run
                    or metadata.get("dry_run") is not True
                    or result.backend != self.name
                    or metadata.get("compilation_identity")
                    != artifact.execution_identity
                ):
                    raise ValueError(
                        "local dry-run snapshot does not match this Runtime endpoint"
                    )
                return CompletedJob(f"{self.name}.restored", result)
            if (
                self.dry_run
                or not isinstance(result, SampleResult)
                or result.result_abi != artifact.result_abi
                or result.backend != self.name
            ):
                raise ValueError("local snapshot does not match this Runtime artifact")
            return CompletedJob(f"{self.name}.restored", result)
        reference = _snapshot_reference(snapshot)
        context = reference.context_values
        if (
            context.get("dry_run") not in {None, "true"}
            or (context.get("dry_run") == "true") != self.dry_run
        ):
            raise ValueError("Runtime snapshot dry-run mode does not match endpoint")
        if (
            reference.provider != "qiskit-runtime"
            or len(reference.job_ids) != 1
            or reference.target != _backend_name(self.backend)
            or context.get("kind")
            not in {"sample", "indexed_sample", "shot_template_sample"}
        ):
            raise ValueError(
                "snapshot does not identify a sampling job on this Runtime backend"
            )
        digest = artifact.execution_identity
        if context.get("compilation_identity") != digest:
            raise ValueError("artifact does not match the snapshotted Runtime job")
        try:
            shots = int(context["shots"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Runtime snapshot context is invalid") from error
        if shots <= 0:
            raise ValueError("Runtime snapshot shots must be positive")
        bound = artifact.parameter_abi.bind(parameters or {})
        invocation = Invocation(FrozenMapping.from_mapping(dict(bound)), shots, 0)
        children: IndexedChildren
        circuits: tuple[Any, ...]
        bit_maps: tuple[tuple[tuple[str, int], ...], ...]
        if context["kind"] == "shot_template_sample":
            from ._shot_template import template_inputs

            circuit, _, _ = template_inputs(artifact, invocation)
            bit_map = _classical_bit_map(circuit)
            if context.get("bits") != json.dumps(bit_map, separators=(",", ":")):
                raise ValueError("Runtime snapshot bit map disagrees with artifact")
            children = ()
            circuits = ()
            bit_maps = ()
        elif context["kind"] == "indexed_sample":
            from qamomile.compilation import _current_shot_children

            children = _current_shot_children(artifact, self, invocation)
            circuits = indexed_circuits(children, invocation, backend=self.backend)
            bit_maps = tuple(_classical_bit_map(circuit) for circuit in circuits)
            if ("bits_digest" in context) == ("bits" in context):
                raise ValueError("Runtime snapshot bit maps disagree with artifact")
            bits_digest = context.get("bits_digest")
            if bits_digest is None:
                # Saved references to existing external jobs cannot be rewritten.
                # Read their canonical JSON while new references retain only its digest.
                bits_digest = sha256(context["bits"].encode("utf-8")).hexdigest()
            if bits_digest != _classical_bit_maps_digest(bit_maps):
                raise ValueError("Runtime snapshot bit maps disagree with artifact")
            circuit = circuits[0]
            bit_map = bit_maps[0]
        else:
            children = ()
            circuits = ()
            bit_maps = ()
            try:
                bit_map = _decode_classical_bit_map(context["bits"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Runtime snapshot context is invalid") from error
            circuit = bind_circuit(load_circuit(artifact), artifact, invocation)
            if _classical_bit_map(circuit) != bit_map:
                raise ValueError("Runtime snapshot bit map disagrees with artifact")
        if self.service is None:
            raise NotImplementedError("IBM Runtime restoration requires a service")
        restore = getattr(self.service, "job", None)
        if not callable(restore):
            raise TypeError("Runtime service must expose job(job_id)")
        native = restore(reference.job_ids[0])
        return _IBMRuntimeJob(
            native,
            circuit,
            artifact,
            invocation.parameters,
            shots,
            self.name,
            bit_map,
            reference,
            children=children,
            circuits=circuits,
            bit_maps=bit_maps,
            shot_template=context["kind"] == "shot_template_sample",
            dry_run=self.dry_run,
        )

    def _launch_shot_template(
        self, artifact: CompiledProgram, invocation: Invocation
    ) -> Job:
        """Submit one retained SDK circuit with explicit native parameter rows.

        Args:
            artifact (CompiledProgram): Artifact retaining a native fixed template.
            invocation (Invocation): Validated public parameters and shot policy.

        Returns:
            Job: One asynchronous parameter-row PUB or completed native samples.

        Raises:
            ExecutionError: If the request has a nonportable seed policy or
                dry-run contains no quantum cloud workload.
        """
        from qiskit.primitives.containers.bindings_array import BindingsArray

        from ._shot_template import template_inputs, template_result

        self.validate_invocation(invocation)
        circuit, hidden, rows = template_inputs(artifact, invocation)
        bit_map = _classical_bit_map(circuit)
        if not circuit.num_qubits:
            if self.dry_run:
                raise ExecutionError("Q8232: dry-run requires a quantum cloud workload")
            return CompletedJob(
                sample_job_id("runtime-shot-template", artifact, invocation),
                template_result(
                    artifact,
                    invocation,
                    MeasurementFreeShots(invocation.shots),
                    len(circuit.clbits),
                    backend=self.name,
                ),
            )
        require_batch_capacity(self.backend, 1)
        retained = tuple(
            (index, parameter)
            for index, parameter in enumerate(hidden)
            if parameter in circuit.parameters
        )
        bindings = BindingsArray(
            {
                tuple(parameter for _, parameter in retained): [
                    [row[index] for index, _ in retained] for row in rows
                ]
            }
            if retained
            else {},
            shape=(invocation.shots,),
        )
        native = self._submit_sampler([(circuit, bindings)], shots=1)
        reference = ExecutionReference.create(
            provider="qiskit-runtime",
            job_ids=(_job_id(native),),
            target=_backend_name(self.backend),
            context={
                "compilation_identity": artifact.execution_identity,
                "bits": json.dumps(bit_map, separators=(",", ":")),
                "kind": "shot_template_sample",
                "shots": str(invocation.shots),
                **({"dry_run": "true"} if self.dry_run else {}),
            },
        )
        return _IBMRuntimeJob(
            native,
            circuit,
            artifact,
            invocation.parameters,
            invocation.shots,
            self.name,
            bit_map,
            reference,
            shot_template=True,
            dry_run=self.dry_run,
        )

    def _launch_indexed(
        self,
        artifact: CompiledProgram,
        invocation: Invocation,
        children: IndexedChildren,
    ) -> Job:
        """Submit ordered logical children as one Runtime one-shot PUB batch.

        Args:
            artifact (CompiledProgram): Original target-bound invocation recipe.
            invocation (Invocation): Original logical invocation.
            children (IndexedChildren): Ordered native-materialized child artifacts.

        Returns:
            Job: One asynchronous provider job preserving each child's context.

        Raises:
            ExecutionError: If the request has a nonportable seed policy or
                dry-run contains no quantum cloud workload.
        """
        self.validate_invocation(invocation)
        circuits = indexed_circuits(children, invocation, backend=self.backend)
        bit_maps = tuple(_classical_bit_map(circuit) for circuit in circuits)
        bits_digest = _classical_bit_maps_digest(bit_maps)
        active = [circuit for circuit in circuits if circuit.num_qubits]
        require_batch_capacity(self.backend, len(active))
        if not active:
            if self.dry_run:
                raise ExecutionError("Q8232: dry-run requires a quantum cloud workload")
            return CompletedJob(
                sample_job_id("runtime-indexed", artifact, invocation),
                indexed_result(
                    artifact,
                    children,
                    circuits,
                    tuple({"": 1} for _ in circuits),
                    backend=self.name,
                ),
            )
        native = self._submit_sampler(active, shots=1)
        reference = ExecutionReference.create(
            provider="qiskit-runtime",
            job_ids=(_job_id(native),),
            target=_backend_name(self.backend),
            context={
                "compilation_identity": artifact.execution_identity,
                "bits_digest": bits_digest,
                "kind": "indexed_sample",
                "shots": str(invocation.shots),
                **({"dry_run": "true"} if self.dry_run else {}),
            },
        )
        return _IBMRuntimeJob(
            native,
            circuits[0],
            artifact,
            invocation.parameters,
            invocation.shots,
            self.name,
            bit_maps[0],
            reference,
            children=children,
            circuits=circuits,
            bit_maps=bit_maps,
            dry_run=self.dry_run,
        )


@dataclass(frozen=True, slots=True)
class _IBMRuntimeJob:
    """Adapt one Runtime primitive job to Qamomile's portable Job contract."""

    native: Any = field(repr=False, compare=False)
    circuit: Any = field(repr=False, compare=False)
    artifact: CompiledProgram = field(repr=False)
    parameters: FrozenMapping = field(repr=False)
    shots: int
    backend: str
    bit_map: tuple[tuple[str, int], ...]
    reference: ExecutionReference
    children: IndexedChildren = field(default=(), repr=False)
    circuits: tuple[Any, ...] = field(default=(), repr=False)
    bit_maps: tuple[tuple[tuple[str, int], ...], ...] = field(default=(), repr=False)
    shot_template: bool = field(default=False, repr=False)
    dry_run: bool = False
    _future: Future[SampleResult | ValidationResult] | None = field(
        init=False, default=None, repr=False, compare=False
    )
    _lock: threading.Lock = field(
        init=False, default_factory=threading.Lock, repr=False, compare=False
    )

    @property
    def job_id(self) -> str:
        """Return the provider-scoped job identity."""

        return self.reference.job_ids[0]

    @property
    def status(self) -> JobStatus:
        """Return the normalized provider or cached retrieval status."""

        future = self._future
        if future is not None and future.done():
            if future.exception() is None:
                result = future.result()
                if isinstance(result, ValidationResult):
                    return result.status
                return JobStatus.COMPLETED
            provider = _runtime_status(self.native)
            return JobStatus.FAILED if provider is JobStatus.COMPLETED else provider
        return _runtime_status(self.native)

    def result(self, timeout: float | None = None) -> SampleResult | ValidationResult:
        """Retrieve measurements or a dry-run report once with a local timeout.

        Args:
            timeout (float | None): Nonnegative local wait limit. Expiration
                leaves the remote job and shared retrieval thread running.

        Returns:
            SampleResult | ValidationResult: Decoded measurements or provider
                validation, without interpreting randomized dry-run results.

        Raises:
            ValueError: If ``timeout`` is invalid.
            TimeoutError: If the local wait expires.
            ExecutionError: If provider retrieval or decoding fails.
        """

        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("timeout must be a non-negative finite number or None")
        with self._lock:
            future = self._future
            if future is None:
                future = Future()
                object.__setattr__(self, "_future", future)
                threading.Thread(
                    target=self._retrieve,
                    args=(future,),
                    daemon=True,
                    name="qamomile-ibm-runtime-result",
                ).start()
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as error:
            raise TimeoutError("IBM Runtime execution result timed out") from error

    def _retrieve(self, future: Future[SampleResult | ValidationResult]) -> None:
        """Resolve the shared result future exactly once.

        Args:
            future (Future[SampleResult | ValidationResult]): Shared retrieval
                future used by all concurrent result callers.
        """

        try:
            if self.dry_run:
                future.set_result(self._retrieve_validation())
                return
            raw = self.native.result()
            if self.shot_template:
                from ._shot_template import template_result

                counts = _runtime_template_counts(raw, self.bit_map, self.shots)
                future.set_result(
                    template_result(
                        self.artifact,
                        Invocation(self.parameters, self.shots, 0),
                        counts,
                        len(self.circuit.clbits),
                        backend=self.backend,
                    )
                )
                return
            if self.children:
                active_counts = iter(
                    _runtime_indexed_counts(
                        raw,
                        bit_maps=tuple(
                            bit_map
                            for circuit, bit_map in zip(self.circuits, self.bit_maps)
                            if circuit.num_qubits
                        ),
                    )
                )
                counts = tuple(
                    next(active_counts) if circuit.num_qubits else {"": 1}
                    for circuit in self.circuits
                )
                future.set_result(
                    indexed_result(
                        self.artifact,
                        self.children,
                        self.circuits,
                        counts,
                        backend=self.backend,
                    )
                )
                return
            native_counts = _runtime_counts(raw, bit_map=self.bit_map)
            parameters = Invocation(self.parameters, self.shots, 0).values()
            logical_counts = decode_counts(
                self.artifact,
                cast(Mapping[object, int], native_counts),
                parameters,
                len(self.circuit.clbits),
            )
            result = SampleResult.from_counts(
                backend=self.backend,
                counts=logical_counts,
                expected_width=self.artifact.result_abi.width,
            )
            if result.shots != self.shots:
                raise ExecutionError(
                    f"Q8230: Runtime returned {result.shots} shots; expected {self.shots}"
                )
            future.set_result(replace(result, result_abi=self.artifact.result_abi))
        except Exception as error:
            if isinstance(error, ExecutionError):
                future.set_exception(error)
            else:
                future.set_exception(
                    ExecutionError(f"Q8227: IBM Runtime result failed: {error}")
                )

    def _retrieve_validation(self) -> ValidationResult:
        """Wait for provider validation without decoding randomized measurements.

        Returns:
            ValidationResult: Terminal validation state and portable usage data.

        Raises:
            ExecutionError: If retrieval fails or the provider state is invalid.
        """
        failure: Exception | None = None
        try:
            self.native.result()
        except Exception as error:
            runtime = _runtime_module()
            terminal_errors = tuple(
                error_type
                for name in (
                    "RuntimeJobFailureError",
                    "RuntimeInvalidStateError",
                    "RuntimeJobMaxTimeoutError",
                )
                if isinstance(error_type := getattr(runtime, name, None), type)
            )
            if not isinstance(error, terminal_errors):
                raise
            failure = error
        metadata = dict(self.metadata())
        status = _STATUS_MAP.get(str(metadata["provider_status"]))
        if status not in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            raise ExecutionError(
                "Q8233: Runtime validation returned a nonterminal status"
            )
        if failure is not None:
            if status is JobStatus.COMPLETED:
                raise failure
            metadata["error_message"] = str(failure)
        return ValidationResult.from_metadata(
            backend=self.backend, status=status, metadata=metadata
        )

    def cancel(self) -> None:
        """Request best-effort cancellation for an unfinished Runtime job.

        Raises:
            ExecutionError: If the native job cannot be cancelled.
        """

        if self.status in {
            JobStatus.COMPLETED,
            JobStatus.CANCELLED,
            JobStatus.FAILED,
        }:
            return
        cancel = getattr(self.native, "cancel", None)
        if not callable(cancel):
            raise ExecutionError("Q8226: Runtime job does not support cancellation")
        try:
            cancel()
        except Exception as error:
            raise ExecutionError(
                f"Q8226: Runtime cancellation failed: {error}"
            ) from error

    def snapshot(self) -> ExecutionSnapshot:
        """Capture a cached local result or a secret-free remote reference."""

        future = self._future
        if future is not None and future.done() and future.exception() is None:
            return ExecutionSnapshot.completed(future.result())
        return ExecutionSnapshot.remote(self.reference)

    def metadata(self) -> Mapping[str, object]:
        """Return provider metrics and explicitly labelled dry-run usage.

        Returns:
            Mapping[str, object]: Provider job identity and metrics. Dry runs
                include usage estimates in seconds and optional resource units;
                unavailable or pending estimates remain None.

        Raises:
            ExecutionError: If provider metadata cannot be read or is malformed.
        """

        metrics = getattr(self.native, "metrics", None)
        try:
            provider_metrics = metrics() if callable(metrics) else None
        except Exception as error:
            raise ExecutionError(f"Q8231: Runtime metrics failed: {error}") from error
        metadata: dict[str, object] = {
            "backend": self.backend,
            "job_id": self.job_id,
            "shots": self.shots,
            **({"metrics": provider_metrics} if provider_metrics is not None else {}),
        }
        if self.dry_run:
            metadata.update(
                dry_run=True,
                compilation_identity=self.artifact.execution_identity,
                provider_status=_runtime_status_name(self.native),
                usage=_runtime_usage(self.native, provider_metrics),
            )
        return metadata


def _accepts_keyword(function: Any, name: str) -> bool:
    """Inspect an explicit SDK keyword without invoking provider operations.

    Args:
        function (Any): Callable SDK surface to inspect.
        name (str): Required explicitly declared keyword argument.

    Returns:
        bool: Whether the callable declares the argument as a keyword.
    """
    try:
        parameter = inspect.signature(function).parameters.get(name)
    except (TypeError, ValueError):
        return False
    return parameter is not None and parameter.kind in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }


def _require_dry_run_mode(runtime: Any, mode: Any) -> None:
    """Require job or Batch mode, including the primitive's resolved mode.

    Args:
        runtime (Any): Imported IBM Runtime SDK module.
        mode (Any): Explicit or context-resolved primitive execution mode.

    Raises:
        ValueError: If a Session or another unsupported mode is selected.
    """
    batch_type = getattr(runtime, "Batch", None)
    if mode is None or (isinstance(batch_type, type) and isinstance(mode, batch_type)):
        return
    raise ValueError("IBM Runtime dry-run supports job or Batch mode, not Session mode")


def _runtime_context_mode() -> Any:
    """Read the SDK context that a new primitive may resolve at submission.

    Returns:
        Any: Current SDK Session or Batch, or None for job mode.

    Raises:
        TypeError: If this SDK cannot expose the current execution context.
    """
    try:
        context = importlib.import_module("qiskit_ibm_runtime.utils.default_session")
        current = context.get_cm_session
    except (ImportError, AttributeError) as error:
        raise TypeError(
            "IBM Runtime dry-run requires SDK execution-context inspection"
        ) from error
    if not callable(current):
        raise TypeError("IBM Runtime execution context must expose get_cm_session()")
    return current()


def _runtime_usage(native: Any, metrics: object) -> Mapping[str, float | None]:
    """Read finalized dry-run usage without converting pending usage to zero.

    Args:
        native (Any): IBM Runtime job exposing usage().
        metrics (object): Provider metrics containing usage calculation status.

    Returns:
        Mapping[str, float | None]: Estimated quantum seconds and resource units,
            each None until the provider certifies completed usage calculation.

    Raises:
        ExecutionError: If usage metadata is malformed or retrieval fails.
    """
    values: dict[str, float | None] = {
        "estimated_quantum_seconds": None,
        "estimated_resource_units": None,
    }
    if metrics is not None and not isinstance(metrics, Mapping):
        raise ExecutionError("Q8231: Runtime metrics must be a mapping")
    reported = metrics.get("usage") if isinstance(metrics, Mapping) else None
    if reported is None:
        return values
    if not isinstance(reported, Mapping):
        raise ExecutionError("Q8231: Runtime usage metrics must be a mapping")
    if reported.get("status") != "complete":
        return values
    usage = getattr(native, "usage", None)
    if not callable(usage):
        raise ExecutionError("Q8231: Runtime dry-run job must expose usage()")
    try:
        values["estimated_quantum_seconds"] = _usage_number(usage())
        if _accepts_keyword(usage, "resource_units"):
            values["estimated_resource_units"] = _usage_number(
                usage(resource_units=True)
            )
    except Exception as error:
        raise ExecutionError(f"Q8231: Runtime usage failed: {error}") from error
    return values


def _usage_number(value: object) -> float | None:
    """Validate one optional nonnegative finite provider usage measurement.

    Args:
        value (object): Reported usage amount or None when unavailable.

    Returns:
        float | None: Checked usage amount in the requested unit.

    Raises:
        ValueError: If the provider amount is not a finite nonnegative number.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Runtime usage must be a nonnegative finite number or None")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("Runtime usage must be a nonnegative finite number or None")
    return result


def _runtime_module() -> Any:
    """Import Runtime with an actionable optional-dependency hint."""

    try:
        return importlib.import_module("qiskit_ibm_runtime")
    except ImportError as error:
        raise ImportError(
            'IBM Runtime execution requires pip install "qamomile[qiskit]"'
        ) from error


def _job_id(native: Any) -> str:
    """Read and validate one provider job identity."""

    identity = getattr(native, "job_id", None)
    value = identity() if callable(identity) else identity
    if value is None or not str(value):
        raise ExecutionError("Q8225: Runtime job has no job identity")
    return str(value)


def _runtime_status(native: Any) -> JobStatus:
    """Map the documented Runtime lifecycle into portable states.

    Args:
        native (Any): Provider job exposing status().

    Returns:
        JobStatus: Normalized lifecycle state, with unknown states failing closed.

    Raises:
        ExecutionError: If provider status retrieval fails.
    """

    return _STATUS_MAP.get(_runtime_status_name(native), JobStatus.FAILED)


def _runtime_status_name(native: Any) -> str:
    """Read the provider's lifecycle name without treating unknowns as success.

    Args:
        native (Any): Runtime job exposing status().

    Returns:
        str: Uppercase provider status name.

    Raises:
        ExecutionError: If the status method is absent or retrieval fails.
    """

    status = getattr(native, "status", None)
    if not callable(status):
        raise ExecutionError("Q8228: Runtime job has no status() method")
    try:
        value = status()
    except Exception as error:
        raise ExecutionError(f"Q8228: Runtime status failed: {error}") from error
    return str(getattr(value, "name", value)).upper()


def _classical_bit_map(circuit: Any) -> tuple[tuple[str, int], ...]:
    """Map increasing Qiskit clbits to named Runtime register locations."""

    locations: list[tuple[str, int]] = []
    for bit in circuit.clbits:
        registers = circuit.find_bit(bit).registers
        if not registers:
            raise ExecutionError(
                "Q8229: Runtime sampling requires named classical registers"
            )
        register, index = registers[0]
        locations.append((str(register.name), int(index)))
    return tuple(locations)


def _classical_bit_maps_digest(
    bit_maps: Sequence[Sequence[tuple[str, int]]],
) -> str:
    """Hash ordered register maps without allocating their complete JSON text.

    Args:
        bit_maps (Sequence[Sequence[tuple[str, int]]]): Runtime register locations
            in logical-index order, including empty maps for unmeasured circuits.

    Returns:
        str: SHA-256 digest of the canonical compact JSON register maps.
    """
    digest = sha256()
    for chunk in json.JSONEncoder(separators=(",", ":")).iterencode(bit_maps):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def _decode_classical_bit_map(payload: str) -> tuple[tuple[str, int], ...]:
    """Validate a snapshot's JSON classical-bit map."""

    raw = json.loads(payload)
    if (
        not isinstance(raw, list)
        or not raw
        or any(
            not isinstance(entry, list)
            or len(entry) != 2
            or type(entry[0]) is not str
            or not entry[0]
            or type(entry[1]) is not int
            or entry[1] < 0
            for entry in raw
        )
    ):
        raise ValueError("invalid Runtime classical-bit map")
    return tuple((entry[0], entry[1]) for entry in raw)


def _runtime_counts(
    result: Any,
    *,
    bit_map: Sequence[Sequence[Any]],
) -> dict[str, int]:
    """Join classified Runtime register rows in Qiskit big-endian order."""

    if len(result) != 1:
        raise ValueError("expected one Runtime sampling result")
    return _runtime_pub_counts(result[0], bit_map=bit_map)


def _runtime_pub_counts(
    result: Any,
    *,
    bit_map: Sequence[Sequence[Any]],
    location: int | None = None,
    rows: int | None = None,
) -> dict[str, int]:
    """Decode one classified PUB position without combining parameter rows.

    Args:
        result (Any): One Runtime sampling PUB result.
        bit_map (Sequence): Ordered register names and bit positions.
        location (int | None): Parameter-row position, or a scalar PUB.
        rows (int | None): Expected parameter-row count for a shaped PUB.

    Returns:
        dict[str, int]: Counts in Qiskit's big-endian register order.

    Raises:
        TypeError: If register data is not classified BitArray data.
        ValueError: If register shapes, widths, or shot counts disagree.
    """
    from qiskit.primitives.containers import BitArray

    registers: dict[str, list[str]] = {}
    for name, index in bit_map:
        data = getattr(result.data, name)
        if not isinstance(data, BitArray):
            raise TypeError("Runtime sampling requires classified BitArray results")
        expected_shape = () if rows is None else (rows,)
        if data.shape != expected_shape or index >= data.num_bits:
            raise ValueError("unexpected Runtime register shape or width")
        if name not in registers:
            registers[name] = data.get_bitstrings(location)
    if not registers and location is not None:
        return {"": 1}
    shot_counts = {len(rows) for rows in registers.values()}
    if len(shot_counts) != 1:
        raise ValueError("Runtime registers have inconsistent shot counts")
    shots = shot_counts.pop()
    return dict(
        Counter(
            "".join(
                registers[name][shot][-1 - index] for name, index in reversed(bit_map)
            )
            for shot in range(shots)
        )
    )


def _runtime_template_counts(
    result: Any, bit_map: Sequence[Sequence[Any]], shots: int
) -> tuple[dict[str, int], ...] | MeasurementFreeShots:
    """Decode a fixed-template PUB without expanding unmeasured coordinates.

    Args:
        result (Any): Runtime result containing one parameter-row PUB.
        bit_map (Sequence[Sequence[Any]]): Template classical register map.
        shots (int): Required logical parameter-row cardinality.

    Returns:
        tuple[dict[str, int], ...] | MeasurementFreeShots: Ordered measurements
        or compact validated coverage for an unmeasured PUB.

    Raises:
        ValueError: If the PUB count or measured row shape is invalid.
        ExecutionError: If unmeasured coverage lacks one completed shot per row.
    """
    if bit_map:
        return _runtime_indexed_counts(
            result, bit_maps=(bit_map,) * shots, parameter_rows=True
        )
    if len(result) != 1:
        raise ValueError("Runtime indexed result count disagrees with submission")
    _require_unmeasured_pub(result[0], (shots,))
    return MeasurementFreeShots(shots)


def _require_unmeasured_pub(pub: Any, expected_shape: tuple[int, ...]) -> None:
    """Check Runtime evidence of one completed shot per unmeasured position.

    Args:
        pub (Any): Provider PUB result with data shape and shot metadata.
        expected_shape (tuple[int, ...]): Submitted parameter-row shape.

    Raises:
        ExecutionError: If shape or completed shot count disagrees.
    """
    metadata = getattr(pub, "metadata", None)
    completed_shots = metadata.get("shots") if isinstance(metadata, Mapping) else None
    if (
        getattr(getattr(pub, "data", None), "shape", None) != expected_shape
        or type(completed_shots) is not int
        or completed_shots != 1
    ):
        raise ExecutionError("Q8233: Runtime unmeasured PUB positions require one shot")


def _runtime_indexed_counts(
    result: Any,
    *,
    bit_maps: Sequence[Sequence[Sequence[Any]]],
    parameter_rows: bool = False,
) -> tuple[dict[str, int], ...]:
    """Keep each logical index separate while decoding a one-shot batch.

    Args:
        result (Any): Runtime results in submission order.
        bit_maps (Sequence): One classical register map per logical index.
        parameter_rows (bool): Whether one PUB carries all parameter rows.

    Returns:
        tuple[dict[str, int], ...]: One one-shot count map per logical index.

    Raises:
        ValueError: If PUB count, register shape, or shot cardinality differs.
        ExecutionError: If an unmeasured PUB does not certify one completed
            shot at each submitted coordinate.
    """
    count = len(bit_maps)
    if len(result) != (1 if parameter_rows else count):
        raise ValueError("Runtime indexed result count disagrees with submission")
    decoded = []
    for index, bit_map in enumerate(bit_maps):
        pub = result[0] if parameter_rows else result[index]
        if bit_map:
            counts = _runtime_pub_counts(
                pub,
                bit_map=bit_map,
                location=index if parameter_rows else None,
                rows=count if parameter_rows else None,
            )
        else:
            _require_unmeasured_pub(pub, (count,) if parameter_rows else ())
            counts = {"": 1}
        decoded.append(counts)
    if any(sum(row.values()) != 1 for row in decoded):
        raise ValueError("Runtime indexed PUB positions must contain one shot")
    return tuple(decoded)


def _snapshot_reference(
    snapshot: ExecutionSnapshot | ExecutionReference,
) -> ExecutionReference:
    """Extract exactly one remote reference from public snapshot input."""

    if isinstance(snapshot, ExecutionReference):
        return snapshot
    if not isinstance(snapshot, ExecutionSnapshot):
        raise TypeError("restore expects an ExecutionSnapshot or ExecutionReference")
    if snapshot.kind is not ExecutionSnapshotKind.REMOTE or snapshot.reference is None:
        raise ValueError("restore expects exactly one remote execution snapshot")
    return snapshot.reference


__all__ = ["IBMRuntime"]
