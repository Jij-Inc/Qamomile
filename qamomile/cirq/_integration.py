"""Implement Cirq targets, simulators, and Google execution endpoints.

The Rust compiler remains the authority for source semantics and ABI facts.
This module owns only Cirq emission, device routing, sampling, and native result
normalization.  Optional Cirq SDK objects never enter persistent Qamomile data.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import uuid
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field, replace
from enum import StrEnum
from importlib.metadata import version
from types import MappingProxyType
from typing import Any, cast

from qamomile.circuit import (
    RESULT_POSTPROCESSING_FORMAT,
    RESULT_POSTPROCESSING_METADATA,
    CircuitBlock,
    CircuitInstruction,
    CircuitProgram,
    CircuitValue,
    ClassicalExpr,
    ParameterValue,
    circuit_result_postprocessing,
    lower_circuit_target,
    postprocess_circuit_counts,
    scalar_parameter_bindings,
    scalar_parameters,
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
from qamomile.errors import CapabilityError, ExecutionError
from qamomile.ir import FrozenMapping, Program
from qamomile.local_execution import launch_local_sample
from qamomile.lowering import RealizationOptions

FORMAT = "cirq.json.v1"
_CAPABILITIES = frozenset(
    {
        Requirement.CLASSICAL_COMPUTE,
        Requirement.COHERENT_CONTROL,
        Requirement.PARAMETERIZED_EXECUTION,
        Requirement.QUANTUM,
        Requirement.MEASUREMENT,
        Requirement.RESET,
        Requirement.DISCARD,
        Requirement.ROTATION,
    }
)
_DEVICE_CAPABILITIES = _CAPABILITIES - {Requirement.RESET}
_REALIZATIONS = frozenset(
    {
        RealizationCapability.RZZ,
        RealizationCapability.MCX,
        RealizationCapability.CONTROLLED_SWAP,
    }
)


class GoogleQVMNoise(StrEnum):
    """Select one noise profile supplied by the Google virtual engine."""

    NONE = "none"
    CALIBRATED = "calibrated"


@dataclass(frozen=True, slots=True)
class DepolarizingNoise:
    """Configure uniform single-qubit depolarizing noise.

    Args:
        probability (float): Total non-identity Pauli error probability in the
            inclusive range from zero to one.

    Raises:
        TypeError: If probability is not a real scalar.
        ValueError: If probability is outside the inclusive unit interval.
    """

    probability: float

    def __post_init__(self) -> None:
        """Validate and normalize the channel probability.

        Raises:
            TypeError: If probability is not a real scalar.
            ValueError: If probability is outside the inclusive unit interval.
        """

        if isinstance(self.probability, bool) or not isinstance(
            self.probability, (int, float)
        ):
            raise TypeError("probability must be a real number")
        probability = float(self.probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between zero and one")
        object.__setattr__(self, "probability", probability)


@dataclass(frozen=True, slots=True)
class Cirq:
    """Compile circuit-shaped QCore into deterministic Cirq JSON.

    Args:
        device (object | None): Optional Cirq device used for placement,
            target-gateset compilation, and validation.
        qubit_mapping (Mapping[int, object] | None): Optional complete logical
            index to physical-qubit mapping. An empty mapping requests
            automatic placement when the device exposes a connectivity graph.
        device_snapshot (str | None): Optional immutable device/calibration
            identity used in the target fingerprint.
        requirements (frozenset[Requirement]): Explicit target capability
            contract. Google device endpoints use a conservative static subset.
        realizations (RealizationOptions): Exact operation-realization overrides.

    Raises:
        TypeError: If the mapping or requirements are malformed.
        ValueError: If a mapping is incomplete, duplicated, or outside the
            selected device.
    """

    device: Any | None = field(default=None, repr=False, compare=False)
    qubit_mapping: Mapping[int, Any] | None = field(
        default=None, repr=False, compare=False
    )
    device_snapshot: str | None = None
    requirements: frozenset[Requirement] = _CAPABILITIES
    realizations: RealizationOptions = RealizationOptions()

    def __post_init__(self) -> None:
        """Freeze and validate the target configuration.

        Raises:
            TypeError: If the mapping or requirements are malformed.
            ValueError: If mapping keys, values, or the snapshot are invalid.
        """

        requirements = frozenset(self.requirements)
        if any(not isinstance(item, Requirement) for item in requirements):
            raise TypeError("requirements must contain Requirement values")
        object.__setattr__(self, "requirements", requirements)
        if self.device_snapshot is not None and not self.device_snapshot:
            raise ValueError("device_snapshot must be nonempty when provided")
        mapping = dict(self.qubit_mapping or {})
        invalid = [
            index
            for index in mapping
            if isinstance(index, bool) or not isinstance(index, int) or index < 0
        ]
        if invalid:
            raise ValueError("qubit_mapping keys must be non-negative integers")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("qubit_mapping must assign each physical qubit once")
        available = getattr(getattr(self.device, "metadata", None), "qubit_set", None)
        if available is not None and any(
            qubit not in available for qubit in mapping.values()
        ):
            raise ValueError("qubit_mapping names a qubit outside the selected device")
        object.__setattr__(self, "qubit_mapping", MappingProxyType(mapping))

    @property
    def name(self) -> str:
        """Return the stable artifact format identifier.

        Returns:
            str: Cirq JSON artifact format.
        """

        return FORMAT

    @property
    def capabilities(self) -> CapabilitySnapshot:
        """Return the immutable Cirq target capability contract.

        Returns:
            CapabilitySnapshot: Supported requirements and realizations.
        """

        return CapabilitySnapshot(
            FORMAT, self.requirements, _REALIZATIONS, self.realizations
        )

    @property
    def fingerprint(self) -> str:
        """Return the versioned Cirq emission and device identity.

        Returns:
            str: Stable target fingerprint.
        """

        mapping = self.qubit_mapping or {}
        policy = json.dumps(
            {
                "emitter": "cirq-json-v1",
                "cirq_version": version("cirq-core"),
                "result_postprocessing": RESULT_POSTPROCESSING_FORMAT,
                "device": self.device_snapshot or _device_snapshot(self.device),
                "mapping": sorted(
                    (index, str(qubit)) for index, qubit in mapping.items()
                ),
                "requirements": sorted(item.value for item in self.requirements),
                "realization_capabilities": sorted(
                    item.value for item in _REALIZATIONS
                ),
                "realization_options": self.realizations.overrides,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return "cirq-" + hashlib.sha256(policy).hexdigest()[:20]

    @property
    def supports_circuit_result_postprocessing(self) -> bool:
        """Return whether execution evaluates retained result expressions.

        Returns:
            bool: Always true for Cirq artifacts.
        """

        return True

    def lower(self, program: Program) -> LoweredProgram:
        """Project verified QCore and emit one Cirq artifact.

        Args:
            program (Program): Closed verified Program.

        Returns:
            LoweredProgram: Unsealed Cirq artifact and Rust-derived ABI facts.

        Raises:
            CapabilityError: If the program requires unsupported control or
                Cirq cannot emit or route an instruction.
        """

        return lower_circuit_target(program, self)

    def lower_circuit(self, logical: CircuitProgram) -> LoweredProgram:
        """Retain one verified static circuit as a FrozenCircuit.

        Args:
            logical (CircuitProgram): Rust-projected target-neutral circuit.

        Returns:
            LoweredProgram: Frozen SDK circuit plus immutable result metadata.

        Raises:
            CapabilityError: If runtime control remains or device compilation
                cannot preserve the circuit.
            ValueError: If the explicit mapping does not cover the circuit.
        """

        _validate_block(logical.body)
        circuit = _emit_circuit(logical)
        stages = ["cirq-emission"]
        if self.device is not None:
            circuit = _prepare_for_device(
                circuit,
                self.device,
                dict(self.qubit_mapping or {}),
            )
            stages.append("cirq-routing")
        from ._payload import CirqExecutable

        payload = CirqExecutable(circuit)
        return LoweredProgram(
            payload=payload,
            parameter_abi=logical.parameter_abi,
            result_abi=logical.result_abi,
            stages=(*logical.realization_stages, *stages),
            circuit_qubits=logical.width,
            circuit_gates=logical.operation_count,
            emitted_instructions=sum(1 for _ in circuit.all_operations()),
            metadata=FrozenMapping.from_mapping(
                {
                    "cirq_version": version("cirq-core"),
                    "measurement_count": logical.measurement_count,
                    **(
                        {RESULT_POSTPROCESSING_METADATA: plan}
                        if (plan := circuit_result_postprocessing(logical)) is not None
                        else {}
                    ),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CirqSimulator:
    """Execute Cirq JSON artifacts in the calling process.

    Args:
        noise (DepolarizingNoise | None): Optional uniform depolarizing channel.
            No noise uses Cirq's state-vector simulator; noise selects its
            density-matrix simulator.
        realizations (RealizationOptions): Exact choices used when compiling source.
    """

    noise: DepolarizingNoise | None = None
    realizations: RealizationOptions = RealizationOptions()

    def __post_init__(self) -> None:
        """Validate the portable simulator configuration.

        Raises:
            TypeError: If noise is not a supported portable noise model.
        """

        if self.noise is not None and not isinstance(self.noise, DepolarizingNoise):
            raise TypeError("noise must be DepolarizingNoise or None")

    @property
    def name(self) -> str:
        """Return the endpoint identity.

        Returns:
            str: Stable simulator endpoint name.
        """

        return "cirq.density-matrix" if self.noise is not None else "cirq.simulator"

    @property
    def target(self) -> Cirq:
        """Return the exact compiler target accepted by this endpoint.

        Returns:
            Cirq: Default local Cirq target.
        """

        return Cirq(realizations=self.realizations)

    @property
    def capabilities(self) -> CapabilitySnapshot:
        """Return the local simulator capability snapshot.

        Returns:
            CapabilitySnapshot: Supported artifact format and requirements.
        """

        return CapabilitySnapshot(FORMAT, _CAPABILITIES, seeded_execution=True)

    def launch(self, artifact: CompiledProgram, invocation: Invocation) -> Job:
        """Launch one Cirq simulation in this process.

        Args:
            artifact (CompiledProgram): Sealed Cirq JSON artifact.
            invocation (Invocation): Rust-normalized parameters, shots, and seed.

        Returns:
            Job: Completed local simulation job.

        Raises:
            ExecutionError: If the artifact is incompatible with this endpoint.
        """

        probability = None if self.noise is None else self.noise.probability
        return launch_local_sample(
            artifact,
            invocation,
            endpoint=self.name,
            artifact_format=FORMAT,
            job_prefix="cirq",
            execute=lambda: _execute_local(
                artifact,
                invocation,
                depolarizing_probability=probability,
            ),
        )


@dataclass(frozen=True, slots=True)
class CirqSampler:
    """Execute routed Cirq artifacts through one live sampler.

    Args:
        sampler (object): Cirq sampler exposing ``run``.
        device (object | None): Optional device used during target routing and
            validation.
        qubit_mapping (Mapping[int, object] | None): Optional logical-to-
            physical mapping. Omit it to request automatic device placement.
        device_snapshot (str | None): Optional immutable calibration identity.
        endpoint_name (str): Stable provider-facing endpoint label.

    Raises:
        TypeError: If sampler does not expose ``run``.
    """

    sampler: Any = field(repr=False, compare=False)
    device: Any | None = field(default=None, repr=False, compare=False)
    qubit_mapping: Mapping[int, Any] | None = field(
        default=None, repr=False, compare=False
    )
    device_snapshot: str | None = None
    endpoint_name: str = "cirq.sampler"

    def __post_init__(self) -> None:
        """Validate the live sampler and target configuration.

        Raises:
            TypeError: If sampler lacks a callable ``run`` method.
            ValueError: If target configuration is malformed.
        """

        if not callable(getattr(self.sampler, "run", None)):
            raise TypeError("sampler must expose run(circuit, ...)")
        _ = self.target

    @classmethod
    def from_google_qvm(
        cls,
        *,
        processor_id: str = "willow_pink",
        noise: GoogleQVMNoise | DepolarizingNoise | str = GoogleQVMNoise.NONE,
        qubit_mapping: Mapping[int, Any] | None = None,
    ) -> CirqProcessor:
        """Create a sampler backed by the local Google virtual engine.

        Args:
            processor_id (str): Bundled virtual processor record.
            noise (GoogleQVMNoise | DepolarizingNoise | str): Virtual-engine
                noise profile.
            qubit_mapping (Mapping[int, object] | None): Optional explicit
                physical placement.

        Returns:
            CirqProcessor: Device-routed Google QVM endpoint.

        Raises:
            ImportError: If ``cirq-google`` is unavailable.
            ValueError: If processor or noise selection is invalid.
        """

        return google_qvm(
            processor_id=processor_id,
            noise=noise,
            qubit_mapping=qubit_mapping,
        )

    @classmethod
    def from_google_engine(
        cls,
        project_id: str,
        processor_id: str,
        *,
        qubit_mapping: Mapping[int, Any] | None = None,
        run_name: str = "",
        snapshot_id: str = "",
        device_config_name: str = "",
    ) -> CirqProcessor:
        """Create a sampler backed by Google Quantum Engine.

        Args:
            project_id (str): Google Cloud project with Quantum Engine access.
            processor_id (str): Accessible processor identifier.
            qubit_mapping (Mapping[int, object] | None): Optional explicit
                physical placement.
            run_name (str): Optional processor automation run.
            snapshot_id (str): Optional immutable processor snapshot.
            device_config_name (str): Optional processor configuration.

        Returns:
            CirqProcessor: Routed Google Engine endpoint.

        Raises:
            ImportError: If ``cirq-google`` is unavailable.
            ValueError: If execution configuration is inconsistent.
        """

        return google_engine(
            project_id,
            processor_id,
            qubit_mapping=qubit_mapping,
            run_name=run_name,
            snapshot_id=snapshot_id,
            device_config_name=device_config_name,
        )

    @property
    def name(self) -> str:
        """Return the configured endpoint identity.

        Returns:
            str: Stable endpoint label.
        """

        return self.endpoint_name

    @property
    def target(self) -> Cirq:
        """Return the device-aware Cirq compiler target.

        Returns:
            Cirq: Target paired with this live sampler.
        """

        return Cirq(
            device=self.device,
            qubit_mapping=self.qubit_mapping,
            device_snapshot=self.device_snapshot,
            requirements=_DEVICE_CAPABILITIES,
        )

    @property
    def capabilities(self) -> CapabilitySnapshot:
        """Return the conservative live-sampler capability contract.

        Returns:
            CapabilitySnapshot: Static sampling capabilities.
        """

        return CapabilitySnapshot(FORMAT, _DEVICE_CAPABILITIES)

    def launch(self, artifact: CompiledProgram, invocation: Invocation) -> Job:
        """Bind and synchronously sample one routed Cirq artifact.

        Args:
            artifact (CompiledProgram): Sealed routed Cirq JSON artifact.
            invocation (Invocation): Rust-normalized launch values.

        Returns:
            Job: Completed portable execution job.

        Raises:
            ExecutionError: If the artifact, seed, sampler, or result is invalid.
        """

        if invocation.seed != 0:
            raise ExecutionError(
                "Q8831: live Cirq samplers do not provide a portable seed contract; "
                "use seed=0"
            )
        result = _execute_with_sampler(
            artifact,
            invocation,
            sampler=self.sampler,
            backend=self.name,
        )
        return CompletedJob(f"cirq-{uuid.uuid4().hex}", result)


@dataclass(frozen=True, slots=True)
class CirqProcessor:
    """Submit Cirq artifacts to one local or remote Google processor.

    Args:
        processor (object): Cirq Google processor exposing ``run_sweep``.
        sampler (object): Processor sampler retaining run/snapshot options.
        device (object): Device used during target routing and validation.
        qubit_mapping (Mapping[int, object] | None): Optional explicit physical
            placement; omission enables automatic routing.
        device_snapshot (str | None): Optional immutable calibration identity.
        endpoint_name (str): Stable provider-facing endpoint label.
        run_name (str): Optional processor automation run.
        snapshot_id (str): Optional immutable processor snapshot.
        device_config_name (str): Optional processor configuration.

    Raises:
        TypeError: If processor or sampler lacks its required SDK methods.
        ValueError: If the target configuration is malformed.
    """

    processor: Any = field(repr=False, compare=False)
    sampler: Any = field(repr=False, compare=False)
    device: Any = field(repr=False, compare=False)
    qubit_mapping: Mapping[int, Any] | None = field(
        default=None, repr=False, compare=False
    )
    device_snapshot: str | None = None
    endpoint_name: str = "cirq.google-processor"
    run_name: str = ""
    snapshot_id: str = ""
    device_config_name: str = ""

    def __post_init__(self) -> None:
        """Validate processor submission and target surfaces.

        Raises:
            TypeError: If required processor or sampler methods are absent.
            ValueError: If the target configuration is malformed.
        """

        if not callable(getattr(self.processor, "run_sweep", None)):
            raise TypeError("processor must expose run_sweep(program, ...)")
        if not callable(getattr(self.sampler, "run", None)):
            raise TypeError("sampler must expose run(circuit, ...)")
        _ = self.target

    @property
    def name(self) -> str:
        """Return the configured processor endpoint identity.

        Returns:
            str: Stable endpoint label.
        """

        return self.endpoint_name

    @property
    def target(self) -> Cirq:
        """Return the device-routed Cirq compiler target.

        Returns:
            Cirq: Target paired with this processor.
        """

        return Cirq(
            device=self.device,
            qubit_mapping=self.qubit_mapping,
            device_snapshot=self.device_snapshot,
            requirements=_DEVICE_CAPABILITIES,
        )

    @property
    def capabilities(self) -> CapabilitySnapshot:
        """Return the conservative processor capability contract.

        Returns:
            CapabilitySnapshot: Static Cirq sampling capabilities.
        """

        return CapabilitySnapshot(FORMAT, _DEVICE_CAPABILITIES)

    def launch(self, artifact: CompiledProgram, invocation: Invocation) -> Job:
        """Submit one routed artifact without blocking for results.

        Args:
            artifact (CompiledProgram): Sealed routed Cirq artifact.
            invocation (Invocation): Rust-normalized launch values.

        Returns:
            Job: Restorable Google processor job.

        Raises:
            ExecutionError: If format, seed, binding, or submission is invalid.
        """

        if artifact.format != FORMAT:
            raise ExecutionError(
                f"Q8830: Google processor accepts {FORMAT!r}, got {artifact.format!r}"
            )
        if invocation.seed != 0:
            raise ExecutionError(
                "Q8831: Google processors do not provide a portable seed contract; "
                "use seed=0"
            )
        circuit = bind_circuit(load_circuit(artifact), artifact, invocation)
        try:
            native = self.processor.run_sweep(
                program=circuit,
                params=None,
                repetitions=invocation.shots,
                run_name=self.run_name,
                snapshot_id=self.snapshot_id,
                device_config_name=self.device_config_name,
            )
        except Exception as error:
            raise ExecutionError(
                f"Q8832: Cirq processor submission failed: {error}"
            ) from error
        reference = ExecutionReference.create(
            provider="cirq-google",
            job_ids=(_native_job_id(native),),
            target=_processor_id(self.processor),
            group_id=_native_program_id(native),
            context={
                "compilation_identity": artifact.execution_identity,
                "kind": "sample",
                "shots": str(invocation.shots),
            },
        )
        return _CirqProcessorJob(
            native=native,
            artifact=artifact,
            parameters=invocation.parameters,
            shots=invocation.shots,
            backend=self.name,
            reference=reference,
        )

    def restore(
        self,
        snapshot: ExecutionSnapshot | ExecutionReference,
        *,
        artifact: CompiledProgram,
        parameters: Mapping[str, object] | None = None,
    ) -> Job:
        """Restore an existing Google processor job without resubmission.

        Args:
            snapshot (ExecutionSnapshot | ExecutionReference): Remote snapshot
                or reference returned by this endpoint.
            artifact (CompiledProgram): Exact artifact used for submission.
            parameters (Mapping[str, object] | None): Original launch values.

        Returns:
            Job: Handle for the existing provider job.

        Raises:
            ValueError: If snapshot, artifact, or endpoint identities disagree.
            TypeError: If the processor cannot expose its owning engine.
            ExecutionError: If launch parameters violate the artifact ABI.
        """

        if (
            artifact.format != FORMAT
            or artifact.target_fingerprint != self.target.fingerprint
        ):
            raise ValueError("artifact does not match this Cirq processor target")
        if isinstance(snapshot, ExecutionSnapshot) and (
            snapshot.kind is ExecutionSnapshotKind.LOCAL
        ):
            result = snapshot.result
            if (
                not isinstance(result, SampleResult)
                or result.result_abi != artifact.result_abi
                or result.backend != self.name
            ):
                raise ValueError("local snapshot does not match this Cirq artifact")
            return CompletedJob(f"{self.name}.restored", result)
        if isinstance(snapshot, ExecutionSnapshot):
            if (
                snapshot.kind is not ExecutionSnapshotKind.REMOTE
                or snapshot.reference is None
            ):
                raise ValueError("snapshot is not one remote Cirq processor job")
            reference = snapshot.reference
        elif isinstance(snapshot, ExecutionReference):
            reference = snapshot
        else:
            raise TypeError("restore expects ExecutionSnapshot or ExecutionReference")
        context = reference.context_values
        if (
            reference.provider != "cirq-google"
            or len(reference.job_ids) != 1
            or reference.target != _processor_id(self.processor)
            or reference.group_id is None
            or context.get("kind") != "sample"
        ):
            raise ValueError("snapshot does not identify this Cirq processor")
        if context.get("compilation_identity") != artifact.execution_identity:
            raise ValueError("artifact does not match the snapshotted Cirq job")
        try:
            shots = int(context["shots"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Cirq snapshot context is invalid") from error
        if shots <= 0:
            raise ValueError("Cirq snapshot shots must be positive")
        bound = artifact.parameter_abi.bind(parameters or {})
        engine_method = getattr(self.processor, "engine", None)
        if not callable(engine_method):
            raise TypeError("processor must expose engine() for restoration")
        try:
            native = (
                engine_method()
                .get_program(reference.group_id)
                .get_job(reference.job_ids[0])
            )
        except Exception as error:
            raise ExecutionError(
                f"Q8833: Cirq processor restoration failed: {error}"
            ) from error
        return _CirqProcessorJob(
            native=native,
            artifact=artifact,
            parameters=FrozenMapping.from_mapping(dict(bound)),
            shots=shots,
            backend=self.name,
            reference=reference,
        )


@dataclass(frozen=True, slots=True)
class _CirqProcessorJob:
    """Adapt one Cirq Google job to Qamomile's portable Job contract.

    Args:
        native (object): Provider-native job.
        artifact (CompiledProgram): Exact submitted artifact.
        parameters (FrozenMapping): Canonical original launch parameters.
        shots (int): Requested repetitions.
        backend (str): Stable endpoint identity.
        reference (ExecutionReference): Secret-free restoration metadata.
    """

    native: Any = field(repr=False, compare=False)
    artifact: CompiledProgram = field(repr=False)
    parameters: FrozenMapping = field(repr=False)
    shots: int
    backend: str
    reference: ExecutionReference
    _future: Future[SampleResult] | None = field(
        init=False, default=None, repr=False, compare=False
    )
    _lock: threading.Lock = field(
        init=False, default_factory=threading.Lock, repr=False, compare=False
    )

    @property
    def job_id(self) -> str:
        """Return the provider-scoped job identity.

        Returns:
            str: Native Cirq Google job ID.
        """

        return self.reference.job_ids[0]

    @property
    def status(self) -> JobStatus:
        """Map the Cirq Google lifecycle into portable states.

        Returns:
            JobStatus: Provider-independent job state.
        """

        future = self._future
        if future is not None and future.done():
            return (
                JobStatus.COMPLETED if future.exception() is None else JobStatus.FAILED
            )
        status = getattr(self.native, "execution_status", None)
        value = status() if callable(status) else status
        name = str(getattr(value, "name", value)).upper()
        if name in {"READY", "STATE_UNSPECIFIED"}:
            return JobStatus.SUBMITTED
        if name in {"RUNNING", "CANCELLING"}:
            return JobStatus.RUNNING
        if name == "SUCCESS":
            return JobStatus.COMPLETED
        if name == "CANCELLED":
            return JobStatus.CANCELLED
        return JobStatus.FAILED

    def result(self, timeout: float | None = None) -> SampleResult:
        """Retrieve and decode processor results exactly once.

        Args:
            timeout (float | None): Optional nonnegative local wait limit.

        Returns:
            SampleResult: Source-ordered typed counts.

        Raises:
            ValueError: If timeout is invalid.
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
                    name="qamomile-cirq-google-result",
                ).start()
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as error:
            raise TimeoutError("Cirq processor result timed out") from error

    def _retrieve(self, future: Future[SampleResult]) -> None:
        """Resolve the shared provider result future once.

        Args:
            future (Future[SampleResult]): Future shared by result callers.
        """

        try:
            results = self.native.results()
            if len(results) != 1:
                raise ExecutionError("Q8834: Cirq processor returned an invalid sweep")
            invocation = Invocation(self.parameters, self.shots, 0)
            result = _decode_cirq_result(
                results[0],
                self.artifact,
                invocation,
                backend=self.backend,
            )
            future.set_result(replace(result, result_abi=self.artifact.result_abi))
        except Exception as error:
            future.set_exception(
                error
                if isinstance(error, ExecutionError)
                else ExecutionError(f"Q8834: Cirq processor result failed: {error}")
            )

    def cancel(self) -> None:
        """Request best-effort cancellation from Cirq Google.

        Raises:
            ExecutionError: If the native job cannot be cancelled.
        """

        if self.status in {JobStatus.COMPLETED, JobStatus.CANCELLED, JobStatus.FAILED}:
            return
        cancel = getattr(self.native, "cancel", None)
        if not callable(cancel):
            raise ExecutionError("Q8835: Cirq processor job cannot be cancelled")
        try:
            cancel()
        except Exception as error:
            raise ExecutionError(
                f"Q8835: Cirq processor cancellation failed: {error}"
            ) from error

    def snapshot(self) -> ExecutionSnapshot:
        """Capture a completed result or secret-free provider reference.

        Returns:
            ExecutionSnapshot: Local result or remote Cirq reference.
        """

        future = self._future
        if future is not None and future.done() and future.exception() is None:
            return ExecutionSnapshot.completed(future.result())
        return ExecutionSnapshot.remote(self.reference)


def google_qvm(
    *,
    processor_id: str = "willow_pink",
    noise: GoogleQVMNoise | DepolarizingNoise | str = GoogleQVMNoise.NONE,
    qubit_mapping: Mapping[int, Any] | None = None,
) -> CirqProcessor:
    """Create a local Google Quantum Virtual Machine endpoint.

    Args:
        processor_id (str): Bundled virtual processor record.
        noise (GoogleQVMNoise | DepolarizingNoise | str): Noiseless, calibrated,
            or uniform depolarizing model.
        qubit_mapping (Mapping[int, object] | None): Optional explicit physical
            placement; omission enables automatic routing.

    Returns:
        CirqProcessor: Device-routed local Google QVM endpoint.

    Raises:
        ImportError: If ``cirq-google`` is unavailable.
        ValueError: If processor or noise selection is invalid.
    """

    try:
        import cirq_google
    except ImportError as error:
        raise ImportError(
            "Google QVM requires cirq-google; install 'qamomile[cirq]'."
        ) from error
    selected: GoogleQVMNoise | DepolarizingNoise
    if isinstance(noise, DepolarizingNoise):
        selected = noise
    else:
        try:
            selected = GoogleQVMNoise(noise)
        except ValueError as error:
            raise ValueError(f"unknown Google QVM noise profile {noise!r}") from error
    try:
        if selected is GoogleQVMNoise.NONE:
            engine = cirq_google.engine.create_noiseless_virtual_engine_from_latest_templates()
        elif selected is GoogleQVMNoise.CALIBRATED:
            engine = cirq_google.engine.create_default_noisy_quantum_virtual_machine(
                processor_id
            )
        else:
            record = cirq_google.SimulatedProcessorWithLocalDeviceRecord(
                processor_id,
                noise_strength=selected.probability,
            )
            engine = cirq_google.engine.SimulatedLocalEngine([record.get_processor()])
        processor = engine.get_processor(processor_id)
    except (KeyError, ValueError) as error:
        available = cirq_google.engine.list_virtual_processors()
        raise ValueError(
            f"unable to create Google QVM processor {processor_id!r}; "
            f"available processors: {available}"
        ) from error
    return CirqProcessor(
        processor=processor,
        sampler=processor.get_sampler(),
        device=processor.get_device(),
        qubit_mapping=qubit_mapping,
        endpoint_name=f"cirq.google-qvm.{processor_id}",
    )


def google_engine(
    project_id: str,
    processor_id: str,
    *,
    qubit_mapping: Mapping[int, Any] | None = None,
    run_name: str = "",
    snapshot_id: str = "",
    device_config_name: str = "",
) -> CirqProcessor:
    """Create a Google Quantum Engine sampler endpoint.

    Args:
        project_id (str): Google Cloud project with Quantum Engine access.
        processor_id (str): Accessible quantum processor identifier.
        qubit_mapping (Mapping[int, object] | None): Optional explicit physical
            placement; omission enables automatic routing.
        run_name (str): Optional processor automation run.
        snapshot_id (str): Optional immutable processor snapshot.
        device_config_name (str): Optional processor configuration.

    Returns:
        CirqProcessor: Routed live Google Engine endpoint.

    Raises:
        ImportError: If ``cirq-google`` is unavailable.
        ValueError: If run and snapshot configuration is inconsistent.
    """

    if run_name and snapshot_id:
        raise ValueError("run_name and snapshot_id are mutually exclusive")
    if bool(run_name or snapshot_id) != bool(device_config_name):
        raise ValueError(
            "device_config_name must accompany exactly one run_name or snapshot_id"
        )
    try:
        import cirq_google
    except ImportError as error:
        raise ImportError(
            "Google Engine requires cirq-google; install 'qamomile[cirq]'."
        ) from error
    engine = cirq_google.Engine(project_id=project_id)
    processor = engine.get_processor(processor_id)
    sampler = cirq_google.ProcessorSampler(
        processor=processor,
        run_name=run_name,
        snapshot_id=snapshot_id,
        device_config_name=device_config_name,
    )
    return CirqProcessor(
        processor=processor,
        sampler=sampler,
        device=processor.get_device(),
        qubit_mapping=qubit_mapping,
        device_snapshot=snapshot_id or run_name or None,
        endpoint_name=f"cirq.google-engine.{processor_id}",
        run_name=run_name,
        snapshot_id=snapshot_id,
        device_config_name=device_config_name,
    )


def load_circuit(artifact: CompiledProgram) -> Any:
    """Load one validated Cirq circuit from a sealed artifact.

    Args:
        artifact (CompiledProgram): CompiledProgram whose format must be Cirq JSON.

    Returns:
        object: Fresh ``cirq.Circuit`` instance.

    Raises:
        ExecutionError: If the format or JSON payload is invalid.
    """

    if artifact.format != FORMAT:
        raise ExecutionError(f"Q8810: expected {FORMAT!r}, got {artifact.format!r}")
    from ._payload import CirqExecutable

    return artifact.execution_payload(CirqExecutable).circuit.unfreeze()


def bind_circuit(
    circuit: Any,
    artifact: CompiledProgram,
    invocation: Invocation,
) -> Any:
    """Bind Rust-normalized aggregate parameters into a Cirq circuit.

    Args:
        circuit (object): Loaded Cirq circuit.
        artifact (CompiledProgram): CompiledProgram carrying the authoritative ABI.
        invocation (Invocation): Canonical launch values.

    Returns:
        object: Fully resolved Cirq circuit.

    Raises:
        ExecutionError: If a retained symbol is missing after binding.
    """

    import cirq

    flattened = scalar_parameter_bindings(
        artifact.parameter_abi,
        invocation.values(),
    )
    resolver = {f"p{index}": value for index, (_, value) in enumerate(flattened)}
    resolved = cirq.resolve_parameters(circuit, cast(Any, resolver), recursive=True)
    missing = sorted(str(item) for item in cirq.parameter_symbols(resolved))
    if missing:
        raise ExecutionError(f"Q8813: missing Cirq parameters: {', '.join(missing)}")
    return resolved


def _validate_block(block: CircuitBlock) -> None:
    """Reject structured runtime control unavailable to Cirq sampling.

    Args:
        block (CircuitBlock): Logical instruction block to inspect.

    Raises:
        CapabilityError: If a condition or structured region remains.
    """

    for instruction in block.instructions:
        if instruction.condition is not None or instruction.regions:
            raise CapabilityError("Q8801: Cirq target requires static control")


def _emit_circuit(program: CircuitProgram) -> Any:
    """Materialize target-neutral instructions as a Cirq circuit.

    Args:
        program (CircuitProgram): Verified logical circuit.

    Returns:
        object: Parameterized ``cirq.Circuit``.

    Raises:
        CapabilityError: If a scalar or instruction cannot be represented.
    """

    import cirq
    import sympy

    qubits = tuple(cirq.LineQubit.range(program.width))
    parameters = {
        leaf.value: sympy.Symbol(f"p{index}")
        for index, leaf in enumerate(scalar_parameters(program.parameter_abi))
    }
    circuit = cirq.Circuit()
    _emit_block(circuit, program.body, qubits, parameters)
    present = circuit.all_qubits()
    circuit.append(cirq.I(qubit) for qubit in qubits if qubit not in present)
    return circuit


def _emit_block(
    circuit: Any,
    block: CircuitBlock,
    qubits: tuple[Any, ...],
    parameters: dict[ParameterValue, Any],
) -> None:
    """Append one logical instruction block to a Cirq circuit.

    Args:
        circuit (object): Destination Cirq circuit.
        block (CircuitBlock): Source-ordered instructions.
        qubits (tuple[object, ...]): Logical line qubits.
        parameters (dict[ParameterValue, object]): Stable symbol cache.

    Raises:
        CapabilityError: If an instruction cannot be represented.
    """

    for instruction in block.instructions:
        _emit_instruction(circuit, instruction, qubits, parameters)


def _emit_instruction(
    circuit: Any,
    instruction: CircuitInstruction,
    qubits: tuple[Any, ...],
    parameters: dict[ParameterValue, Any],
) -> None:
    """Append one logical instruction with exact coherent controls.

    Args:
        circuit (object): Destination Cirq circuit.
        instruction (CircuitInstruction): Logical operation.
        qubits (tuple[object, ...]): Logical line qubits.
        parameters (dict[ParameterValue, object]): Stable symbol cache.

    Raises:
        CapabilityError: If the instruction or its operands are unsupported.
    """

    import cirq

    op = instruction.opcode
    targets = tuple(qubits[index] for index in instruction.qubits)
    controls = [qubits[index] for index in instruction.controls]
    values = tuple(_emit_value(item, parameters) for item in instruction.parameters)
    if op == "measure":
        if (
            controls
            or len(targets) != 1
            or len(values) != 1
            or not isinstance(values[0], int)
        ):
            raise CapabilityError("Q8802: malformed Cirq measurement")
        circuit.append(cirq.measure(targets[0], key=f"m{values[0]}"))
        return
    if op == "reset":
        if controls or len(targets) != 1:
            raise CapabilityError("Q8803: Cirq cannot coherently control reset")
        circuit.append(cirq.reset(targets[0]))
        return
    if op == "global_phase":
        angle = values[0]
        if not controls:
            return
        phase = cirq.ZPowGate(exponent=_turns(angle)).on(controls[-1])
        outer_controls = controls[:-1]
        circuit.append(
            phase.controlled_by(*outer_controls) if outer_controls else phase
        )
        return
    fixed = {
        "h": cirq.H,
        "x": cirq.X,
        "y": cirq.Y,
        "z": cirq.Z,
        "s": cirq.S,
        "sdg": cirq.S**-1,
        "t": cirq.T,
        "tdg": cirq.T**-1,
        "cx": cirq.CNOT,
        "cz": cirq.CZ,
        "ccx": cirq.CCX,
        "swap": cirq.SWAP,
    }
    gate = fixed.get(op)
    if gate is None:
        if op == "p":
            gate = cirq.ZPowGate(exponent=_turns(values[0]))
        elif op == "rx":
            gate = cirq.rx(values[0])
        elif op == "ry":
            gate = cirq.ry(values[0])
        elif op == "rz":
            gate = cirq.rz(values[0])
        elif op == "cp":
            gate = cirq.CZPowGate(exponent=_turns(values[0]))
        elif op == "rzz":
            gate = cirq.ZZPowGate(
                exponent=_turns(values[0]),
                global_shift=-0.5,
            )
        else:
            raise CapabilityError(f"Q8804: Cirq cannot emit instruction {op!r}")
    operation = gate.on(*targets)
    if controls:
        operation = operation.controlled_by(*controls)
    circuit.append(operation)


def _turns(angle: Any) -> Any:
    """Convert radians to Cirq half turns without symbolic numeric residue.

    Args:
        angle (object): Concrete numeric or Sympy-compatible angle in radians.

    Returns:
        object: Floating-point turns for concrete input, otherwise a symbolic
        expression divided by pi.
    """

    if isinstance(angle, (int, float)) and not isinstance(angle, bool):
        return float(angle) / math.pi
    import sympy

    return angle / sympy.pi


def _emit_value(value: CircuitValue, parameters: dict[ParameterValue, Any]) -> Any:
    """Lower one arithmetic circuit value to a Sympy expression.

    Args:
        value (CircuitValue): Rust-projected scalar expression.
        parameters (dict[ParameterValue, object]): Stable symbol cache.

    Returns:
        object: Concrete scalar or Sympy expression accepted by Cirq.

    Raises:
        CapabilityError: If the scalar expression is unsupported.
    """

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, ParameterValue):
        return parameters[value]
    if isinstance(value, ClassicalExpr):
        operands = tuple(_emit_value(item, parameters) for item in value.operands)
        op = value.opcode.removeprefix("classical.")
        if op == "neg":
            return -operands[0]
        if op == "add":
            return operands[0] + operands[1]
        if op == "sub":
            return operands[0] - operands[1]
        if op == "mul":
            return operands[0] * operands[1]
        if op in {"div", "truediv"}:
            return operands[0] / operands[1]
        if op == "pow":
            return operands[0] ** operands[1]
    raise CapabilityError(f"Q8806: unsupported Cirq scalar expression {value!r}")


def _prepare_for_device(circuit: Any, device: Any, mapping: dict[int, Any]) -> Any:
    """Place, route, compile, and validate a circuit for one Cirq device.

    Args:
        circuit (object): Logical LineQubit circuit.
        device (object): Cirq device with metadata and validation.
        mapping (dict[int, object]): Optional explicit physical placement.

    Returns:
        object: Device-valid routed Cirq circuit.

    Raises:
        ValueError: If placement does not cover the circuit or routing fails.
        CapabilityError: If target-gateset compilation or validation fails.
    """

    import cirq

    logical = sorted(circuit.all_qubits())
    if mapping:
        missing = [int(qubit.x) for qubit in logical if int(qubit.x) not in mapping]
        if missing:
            raise ValueError(f"qubit_mapping is missing logical indices {missing!r}")
        circuit = circuit.transform_qubits(lambda qubit: mapping[int(qubit.x)])
    target_gateset = next(
        iter(
            getattr(
                getattr(device, "metadata", None), "compilation_target_gatesets", ()
            )
        ),
        None,
    )
    if target_gateset is not None:
        try:
            circuit = cirq.optimize_for_target_gateset(circuit, gateset=target_gateset)
        except (TypeError, ValueError) as error:
            raise CapabilityError(
                f"Q8820: Cirq target-gateset optimization failed: {error}"
            ) from error
    if not mapping:
        graph = getattr(getattr(device, "metadata", None), "nx_graph", None)
        if graph is None:
            raise ValueError(
                "automatic Cirq placement requires device.metadata.nx_graph"
            )
        try:
            routed, _, _ = cirq.RouteCQC(graph).route_circuit(
                circuit,
                initial_mapper=cirq.LineInitialMapper(graph),
            )
            circuit = cirq.Circuit(routed)
        except ValueError as error:
            raise ValueError(f"automatic Cirq routing failed: {error}") from error
        if target_gateset is not None:
            circuit = cirq.optimize_for_target_gateset(circuit, gateset=target_gateset)
    try:
        device.validate_circuit(circuit)
    except ValueError as error:
        raise CapabilityError(
            f"Q8821: Cirq device validation failed: {error}"
        ) from error
    return circuit


def _device_snapshot(device: Any | None) -> str | None:
    """Derive a deterministic structural identity for a Cirq device.

    Args:
        device (object | None): Optional device object.

    Returns:
        str | None: Structural SHA-256 digest or None without a device.
    """

    if device is None:
        return None
    metadata = getattr(device, "metadata", None)
    qubits = sorted(str(item) for item in getattr(metadata, "qubit_set", ()))
    graph = getattr(metadata, "nx_graph", None)
    edges = (
        sorted(tuple(sorted((str(left), str(right)))) for left, right in graph.edges)
        if graph is not None
        else []
    )
    payload = json.dumps(
        {"type": type(device).__qualname__, "qubits": qubits, "edges": edges},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _execute_local(
    artifact: CompiledProgram,
    invocation: Invocation,
    *,
    depolarizing_probability: float | None,
) -> SampleResult:
    """Execute one Cirq artifact in the calling process.

    Args:
        artifact (CompiledProgram): Sealed Cirq artifact.
        invocation (Invocation): Canonical execution values.
        depolarizing_probability (float | None): Optional channel probability.

    Returns:
        SampleResult: Typed source-order sample counts.

    Raises:
        ExecutionError: If simulation or result normalization fails.
    """

    import cirq

    circuit = bind_circuit(load_circuit(artifact), artifact, invocation)
    if depolarizing_probability is None:
        sampler = cirq.Simulator(seed=_cirq_seed(invocation.seed))
        backend = "cirq.simulator"
    else:
        noise = DepolarizingNoise(depolarizing_probability)
        sampler = cirq.DensityMatrixSimulator(
            noise=cirq.depolarize(noise.probability),
            seed=_cirq_seed(invocation.seed),
        )
        backend = "cirq.density-matrix"
    return _sample_bound_circuit(
        circuit,
        artifact,
        invocation,
        sampler=sampler,
        backend=backend,
    )


def _execute_with_sampler(
    artifact: CompiledProgram,
    invocation: Invocation,
    *,
    sampler: Any,
    backend: str,
) -> SampleResult:
    """Load, bind, and execute a Cirq artifact through a live sampler.

    Args:
        artifact (CompiledProgram): Sealed Cirq artifact.
        invocation (Invocation): Canonical execution values.
        sampler (object): Cirq sampler exposing ``run``.
        backend (str): Result backend identity.

    Returns:
        SampleResult: Typed source-order sample counts.

    Raises:
        ExecutionError: If loading, sampling, or decoding fails.
    """

    circuit = bind_circuit(load_circuit(artifact), artifact, invocation)
    return _sample_bound_circuit(
        circuit,
        artifact,
        invocation,
        sampler=sampler,
        backend=backend,
    )


def _sample_bound_circuit(
    circuit: Any,
    artifact: CompiledProgram,
    invocation: Invocation,
    *,
    sampler: Any,
    backend: str,
) -> SampleResult:
    """Sample a bound Cirq circuit and decode measurement keys once.

    Args:
        circuit (object): Fully bound Cirq circuit.
        artifact (CompiledProgram): CompiledProgram carrying result ABI and metadata.
        invocation (Invocation): Canonical shots and parameters.
        sampler (object): Cirq sampler exposing ``run``.
        backend (str): Result backend identity.

    Returns:
        SampleResult: Typed source-order sample counts.

    Raises:
        ExecutionError: If Cirq rejects execution or returns malformed rows.
    """

    try:
        result = sampler.run(circuit, repetitions=invocation.shots)
    except Exception as error:
        raise ExecutionError(f"Q8814: Cirq sampling failed: {error}") from error
    return _decode_cirq_result(result, artifact, invocation, backend=backend)


def _decode_cirq_result(
    result: Any,
    artifact: CompiledProgram,
    invocation: Invocation,
    *,
    backend: str,
) -> SampleResult:
    """Decode one Cirq result through the sealed source ResultABI.

    Args:
        result (object): Cirq result exposing keyed measurement arrays.
        artifact (CompiledProgram): CompiledProgram carrying result ABI and metadata.
        invocation (Invocation): Canonical shots and parameter values.
        backend (str): Stable result backend identity.

    Returns:
        SampleResult: Typed source-order sample counts.

    Raises:
        ExecutionError: If metadata or measurement rows are malformed.
    """

    raw_measurement_count = dict(artifact.metadata.entries).get("measurement_count")
    if not isinstance(raw_measurement_count, int) or raw_measurement_count < 0:
        raise ExecutionError("Q8815: invalid Cirq measurement metadata")
    rows: Counter[tuple[int, ...]] = Counter()
    for shot in range(invocation.shots):
        bits: list[int] = []
        for index in range(raw_measurement_count):
            raw = result.measurements.get(f"m{index}")
            if raw is None or raw.shape != (invocation.shots, 1):
                raise ExecutionError("Q8816: Cirq result omitted a measurement key")
            bits.append(int(raw[shot, 0]))
        rows[tuple(bits)] += 1
    plan = dict(artifact.metadata.entries).get(RESULT_POSTPROCESSING_METADATA)
    if plan is None:
        projected_counts: Counter[tuple[int, ...]] = Counter()
        try:
            for raw_bits, count in rows.items():
                projected = tuple(
                    raw_bits[index] for index in artifact.result_abi.source_order
                )
                projected_counts[projected] += count
        except IndexError as error:
            raise ExecutionError(
                "Q8817: result ABI names an absent measurement"
            ) from error
        logical = dict(projected_counts)
    else:
        if not isinstance(plan, bytes):
            raise ExecutionError("Q8818: invalid Cirq result metadata")
        logical = postprocess_circuit_counts(
            plan,
            rows,
            invocation.values(),
            artifact.result_abi,
        )
    return SampleResult.from_counts(
        backend=backend,
        counts=logical,
        expected_width=artifact.result_abi.width,
    )


def _native_job_id(native: Any) -> str:
    """Read one nonempty Cirq Google job identity.

    Args:
        native (object): Provider-native job.

    Returns:
        str: Stable provider job ID.

    Raises:
        ExecutionError: If the job exposes no usable identity.
    """

    identity = getattr(native, "id", None)
    value = identity() if callable(identity) else identity
    if value is None or not str(value):
        raise ExecutionError("Q8836: Cirq processor job has no identity")
    return str(value)


def _native_program_id(native: Any) -> str:
    """Read the owning Cirq Google program identity.

    Args:
        native (object): Provider-native job.

    Returns:
        str: Stable owning-program ID.

    Raises:
        ExecutionError: If the owning program exposes no usable identity.
    """

    program_method = getattr(native, "program", None)
    if not callable(program_method):
        raise ExecutionError("Q8837: Cirq processor job has no owning program")
    program = program_method()
    identity = getattr(program, "id", None)
    identity = identity() if callable(identity) else identity
    if identity is None or not str(identity):
        raise ExecutionError("Q8837: Cirq processor program has no identity")
    return str(identity)


def _processor_id(processor: Any) -> str:
    """Read one stable Cirq Google processor identifier.

    Args:
        processor (object): Local or remote Cirq Google processor.

    Returns:
        str: Nonempty processor identifier.

    Raises:
        ExecutionError: If no processor identity is available.
    """

    identity = getattr(processor, "processor_id", None)
    identity = identity() if callable(identity) else identity
    if identity is None or not str(identity):
        raise ExecutionError("Q8838: Cirq processor has no identity")
    return str(identity)


def _cirq_seed(seed: int) -> int:
    """Fold an unsigned 64-bit seed into NumPy's legacy 32-bit domain.

    Args:
        seed (int): Canonical unsigned invocation seed.

    Returns:
        int: Deterministic unsigned 32-bit seed.
    """

    return (seed ^ (seed >> 32)) & ((1 << 32) - 1)


__all__ = [
    "Cirq",
    "CirqProcessor",
    "CirqSampler",
    "CirqSimulator",
    "DepolarizingNoise",
    "FORMAT",
    "GoogleQVMNoise",
    "bind_circuit",
    "google_engine",
    "google_qvm",
    "load_circuit",
]
