"""IBM Runtime sampling endpoint for sealed Qiskit QPY artifacts."""

from __future__ import annotations

import importlib
import json
import math
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field, replace
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
)
from qamomile.effects import Requirement
from qamomile.errors import ExecutionError
from qamomile.ir import FrozenMapping

from ._integration import (
    FORMAT,
    Qiskit,
    _backend_name,
    _backend_requirements,
    bind_circuit,
    decode_counts,
    load_circuit,
)

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
        backend: Explicit IBM BackendV2 used for routing and execution.
        mode: Optional caller-owned Runtime Session or Batch.
        sampler_options: Optional native Runtime sampler options.
        service: Runtime service used for restoration. Defaults to
            ``backend.service`` when available.
        requirements: Explicit endpoint capabilities. ``None`` derives a
            conservative set from the backend target.
        backend_snapshot: Optional immutable backend identity.
        optimization_level: Qiskit transpiler optimization level.
        seed_transpiler: Deterministic transpiler seed.
        scheduling_method: Optional Qiskit scheduling method.

    Raises:
        ImportError: If the optional IBM Runtime SDK is unavailable.
        TypeError: If the backend or SDK surface is incompatible.
        ValueError: If Runtime mode and compilation backend differ.
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
    _sampler: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Load Runtime lazily and validate the paired target configuration.

        Raises:
            ImportError: If ``qiskit-ibm-runtime`` is not installed.
            TypeError: If the backend or SDK surface is incompatible.
            ValueError: If the Runtime mode selects a different backend.
        """

        if self.backend is None:
            raise ValueError("IBM Runtime requires an explicit backend")
        sampler_type = getattr(_runtime_module(), "SamplerV2", None)
        if not callable(sampler_type):
            raise TypeError("qiskit_ibm_runtime must expose SamplerV2")
        if self.requirements is not None:
            requirements = frozenset(self.requirements)
            if any(not isinstance(item, Requirement) for item in requirements):
                raise TypeError("requirements must contain Requirement values")
            object.__setattr__(self, "requirements", requirements)
        sampler = sampler_type(
            mode=self.backend if self.mode is None else self.mode,
            options=self.sampler_options,
        )
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

    def launch(self, artifact: CompiledProgram, invocation: Invocation) -> Job:
        """Submit one routed and fully bound SDK circuit asynchronously.

        Args:
            artifact (CompiledProgram): Live circuit compiled for :attr:`target`.
            invocation: Rust-normalized parameter values and shot policy.

        Returns:
            Job: Nonblocking Runtime job with cancellation and snapshots.

        Raises:
            ExecutionError: If the request is not portable to Runtime sampling.
        """

        if artifact.format != FORMAT:
            raise ExecutionError(
                f"Q8222: IBM Runtime accepts {FORMAT!r}, got {artifact.format!r}"
            )
        if artifact.result_abi.width == 0:
            raise ExecutionError("Q8223: IBM Runtime endpoint requires sampling")
        if invocation.seed != 0:
            raise ExecutionError(
                "Q8224: IBM Runtime does not provide a portable seed contract; "
                "use seed=0"
            )
        circuit = bind_circuit(load_circuit(artifact), artifact, invocation)
        bit_map = _classical_bit_map(circuit)
        native = self._sampler.run([circuit], shots=invocation.shots)
        reference = ExecutionReference.create(
            provider="qiskit-runtime",
            job_ids=(_job_id(native),),
            target=_backend_name(self.backend),
            context={
                "compilation_identity": artifact.execution_identity,
                "bits": json.dumps(bit_map, separators=(",", ":")),
                "kind": "sample",
                "shots": str(invocation.shots),
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
            snapshot: Remote snapshot or reference returned by this endpoint.
            artifact: Compiled program with the same execution identity as the original job.
            parameters: Original launch parameters required by the artifact ABI.

        Returns:
            Job: Handle for the existing provider job.

        Raises:
            NotImplementedError: If no restoration service is configured.
            ValueError: If the reference does not match this endpoint or artifact.
            ExecutionError: If launch parameters do not satisfy the artifact ABI.
        """

        if (
            artifact.format != FORMAT
            or artifact.target_fingerprint != self.target.fingerprint
        ):
            raise ValueError("artifact does not match this Runtime target")
        if isinstance(snapshot, ExecutionSnapshot) and (
            snapshot.kind is ExecutionSnapshotKind.LOCAL
        ):
            result = snapshot.result
            if (
                not isinstance(result, SampleResult)
                or result.result_abi != artifact.result_abi
                or result.backend != self.name
            ):
                raise ValueError("local snapshot does not match this Runtime artifact")
            return CompletedJob(f"{self.name}.restored", result)
        reference = _snapshot_reference(snapshot)
        context = reference.context_values
        if (
            reference.provider != "qiskit-runtime"
            or len(reference.job_ids) != 1
            or reference.target != _backend_name(self.backend)
            or context.get("kind") != "sample"
        ):
            raise ValueError(
                "snapshot does not identify a sampling job on this Runtime backend"
            )
        digest = artifact.execution_identity
        if context.get("compilation_identity") != digest:
            raise ValueError("artifact does not match the snapshotted Runtime job")
        try:
            shots = int(context["shots"])
            bit_map = _decode_classical_bit_map(context["bits"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Runtime snapshot context is invalid") from error
        if shots <= 0:
            raise ValueError("Runtime snapshot shots must be positive")
        bound = artifact.parameter_abi.bind(parameters or {})
        invocation = Invocation(FrozenMapping.from_mapping(dict(bound)), shots, 0)
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
    _future: Future[SampleResult] | None = field(
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
                return JobStatus.COMPLETED
            provider = _runtime_status(self.native)
            return JobStatus.FAILED if provider is JobStatus.COMPLETED else provider
        return _runtime_status(self.native)

    def result(self, timeout: float | None = None) -> SampleResult:
        """Retrieve and decode the Runtime result once with a local timeout.

        Args:
            timeout: Nonnegative local wait limit. Expiration leaves the remote
                job and shared retrieval thread running.

        Returns:
            SampleResult: Counts decoded through the sealed ResultABI.

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

    def _retrieve(self, future: Future[SampleResult]) -> None:
        """Resolve the shared result future exactly once.

        Args:
            future: Shared future used by all concurrent result callers.
        """

        try:
            raw = self.native.result()
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
        """Return non-secret provider metrics when available."""

        metrics = getattr(self.native, "metrics", None)
        try:
            provider_metrics = metrics() if callable(metrics) else None
        except Exception as error:
            raise ExecutionError(f"Q8231: Runtime metrics failed: {error}") from error
        return {
            "backend": self.backend,
            "job_id": self.job_id,
            "shots": self.shots,
            **({"metrics": provider_metrics} if provider_metrics is not None else {}),
        }


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
    """Map the documented Runtime lifecycle into portable states."""

    status = getattr(native, "status", None)
    if not callable(status):
        raise ExecutionError("Q8228: Runtime job has no status() method")
    try:
        value = status()
    except Exception as error:
        raise ExecutionError(f"Q8228: Runtime status failed: {error}") from error
    name = str(getattr(value, "name", value)).upper()
    return _STATUS_MAP.get(name, JobStatus.FAILED)


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

    from qiskit.primitives.containers import BitArray

    if len(result) != 1:
        raise ValueError("expected one Runtime sampling result")
    registers: dict[str, list[str]] = {}
    for name, index in bit_map:
        data = getattr(result[0].data, name)
        if not isinstance(data, BitArray):
            raise TypeError("Runtime sampling requires classified BitArray results")
        if data.shape or index >= data.num_bits:
            raise ValueError("unexpected Runtime register shape or width")
        if name not in registers:
            registers[name] = data.get_bitstrings()
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
