"""Implement Qiskit circuit emission and execution endpoints for Qamomile.

Rust owns compiler semantics and ABI facts. This adapter retains SDK circuits
in process-local executable owners and exports QPY only for explicit persistence.
Binding and endpoint preparation preserve the unbound circuit template.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any

from qamomile._measurement_plan import _MeasurementGroupPlan
from qamomile.circuit import (
    RESULT_POSTPROCESSING_FORMAT,
    RESULT_POSTPROCESSING_METADATA,
    CircuitProgram,
    circuit_result_postprocessing,
    lower_circuit_target,
    lower_to_circuit,
    lower_to_circuit_at,
    postprocess_circuit_counts,
    scalar_parameter_bindings,
)
from qamomile.compilation import (
    CapabilitySnapshot,
    CompiledProgram,
    CouplingEdge,
    EdgeProperty,
    ExecutionReference,
    ExecutionSnapshot,
    Invocation,
    Job,
    JobStatus,
    LoweredProgram,
    ParameterABI,
    QubitProperty,
    RealizationCapability,
    SampleResult,
    TargetCalibration,
    TargetProfile,
    TargetTopology,
)
from qamomile.effects import Requirement
from qamomile.errors import ExecutionError
from qamomile.expectation import ExpectationResult
from qamomile.ir import FrozenMapping, Program
from qamomile.local_execution import launch_local_sample
from qamomile.lowering import RealizationOptions
from qamomile.observable import Hamiltonian
from qamomile.source import ProgramFamily, QKernel, _close_program

from ._emission import emit_circuit as _emit_circuit

FORMAT = "qiskit.qpy.v1"
_CAPABILITIES = frozenset(
    {
        Requirement.CLASSICAL_COMPUTE,
        Requirement.COHERENT_CONTROL,
        Requirement.PARAMETERIZED_EXECUTION,
        Requirement.QUANTUM,
        Requirement.MEASUREMENT,
        Requirement.MID_CIRCUIT_MEASUREMENT,
        Requirement.DYNAMIC_BRANCH,
        Requirement.DYNAMIC_LOOP,
        Requirement.RESET,
        Requirement.DISCARD,
        Requirement.ROTATION,
    }
)
_AER_CAPABILITIES = _CAPABILITIES - {Requirement.DYNAMIC_LOOP}
_REALIZATIONS = frozenset(
    {
        RealizationCapability.NAMED_CALL,
        RealizationCapability.NATIVE_FOR,
        RealizationCapability.PAULI_EVOLUTION,
        RealizationCapability.RZZ,
        RealizationCapability.ADD_INTO,
        RealizationCapability.MCX,
        RealizationCapability.CONTROLLED_SWAP,
        RealizationCapability.COHERENT_REGIONS,
        RealizationCapability.PAULI_MEASUREMENT,
        RealizationCapability.INDEXED_FOR,
    }
)


@dataclass(frozen=True, slots=True)
class Qiskit:
    """Compile circuit-shaped QCore to a retained Qiskit circuit.

    Args:
        basis_gates (tuple[str, ...]): Optional SDK target gate set.
        optimization_level (int): SDK optimization level from zero to three.
        seed_transpiler (int): Deterministic SDK transpilation seed.
        use_classical_stores (bool): Enable verified top-level Boolean stores;
            defaults to False to preserve target compatibility.
        use_native_pauli_evolution (bool): Advertise native Pauli composites.
        realizations (RealizationOptions): Exact operation-realization overrides.
        scheduling_method (str | None): Optional live-backend scheduling policy.
        backend (Any | None): Live SDK backend, excluded from persistent data.
        backend_snapshot (str | None): Immutable backend identity override.
        requirements (frozenset[Requirement]): Supported execution requirements.
        profile (TargetProfile | None): Explicit operator-supplied device facts.
            When omitted, a live backend with a coupling map derives one
            through :func:`profile_from_backend`; otherwise none is exposed.
            The profile digest enters the fingerprint whenever a profile exists.

    Raises:
        TypeError: If capability or native-Pauli flags have invalid types.
        ValueError: If target configuration is inconsistent.
    """

    basis_gates: tuple[str, ...] = ()
    optimization_level: int = 1
    seed_transpiler: int = 7
    use_native_pauli_evolution: bool = True
    use_classical_stores: bool = field(default=False, kw_only=True)
    realizations: RealizationOptions = RealizationOptions()
    scheduling_method: str | None = None
    backend: Any | None = field(default=None, repr=False, compare=False)
    backend_snapshot: str | None = None
    requirements: frozenset[Requirement] = _CAPABILITIES
    profile: TargetProfile | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Reject an ambiguous or unidentifiable target configuration."""

        basis_gates = tuple(self.basis_gates)
        requirements = frozenset(self.requirements)
        if any(not isinstance(item, Requirement) for item in requirements):
            raise TypeError("requirements must contain Requirement values")
        object.__setattr__(self, "basis_gates", basis_gates)
        object.__setattr__(self, "requirements", requirements)
        if self.backend is not None and self.basis_gates:
            raise ValueError("backend and basis_gates are mutually exclusive")
        if self.backend_snapshot is not None and not self.backend_snapshot:
            raise ValueError("backend_snapshot must be nonempty when provided")
        if not 0 <= self.optimization_level <= 3:
            raise ValueError("optimization_level must be between zero and three")
        if not isinstance(self.use_native_pauli_evolution, bool):
            raise TypeError("use_native_pauli_evolution must be a bool")
        if not isinstance(self.use_classical_stores, bool):
            raise TypeError("use_classical_stores must be a bool")
        if self.scheduling_method == "":
            raise ValueError("scheduling_method must be nonempty when provided")
        if self.scheduling_method is not None and self.backend is None:
            raise ValueError("scheduling_method requires a live backend target")
        if self.profile is not None:
            if not isinstance(self.profile, TargetProfile):
                raise TypeError("profile must be a TargetProfile or None")
            if self.profile.format != FORMAT:
                raise ValueError(f"profile must describe the {FORMAT!r} format")

    @property
    def name(self) -> str:
        """Return the stable artifact format identifier."""

        return FORMAT

    @property
    def _profile(self) -> TargetProfile | None:
        """Return the explicit profile, else one derived from a live backend."""

        if self.profile is not None:
            return self.profile
        return profile_from_backend(self.backend)

    @property
    def fingerprint(self) -> str:
        """Return a stable fingerprint for the immutable target configuration.

        The policy folds in the target profile digest whenever a profile
        exists, so a changed coupling map or calibration changes the identity
        even when the structural backend snapshot would not.
        """

        profile = self._profile
        policy = json.dumps(
            {
                "emitter": "qiskit-retained-operations-v3",
                **({"target_profile": profile.digest} if profile is not None else {}),
                "result_postprocessing": RESULT_POSTPROCESSING_FORMAT,
                "basis_gates": self.basis_gates,
                "optimization_level": self.optimization_level,
                "seed_transpiler": self.seed_transpiler,
                "use_native_pauli_evolution": self.use_native_pauli_evolution,
                "realization_options": self.realizations.overrides,
                "scheduling_method": self.scheduling_method,
                "backend_snapshot": self.backend_snapshot
                or _backend_snapshot(self.backend),
                "requirements": sorted(item.value for item in self.requirements),
                "realization_capabilities": sorted(
                    item.value for item in self._realizations
                ),
                "qiskit_version": version("qiskit"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return "qiskit-" + hashlib.sha256(policy).hexdigest()[:20]

    @property
    def capabilities(self) -> CapabilitySnapshot:
        """Return capabilities shared by QPY emission and Aer execution."""

        return CapabilitySnapshot(
            FORMAT,
            self.requirements,
            self._realizations,
            self.realizations,
            profile=self._profile,
        )

    @property
    def _realizations(self) -> frozenset[RealizationCapability]:
        """Return realization choices enabled by this target configuration."""

        capabilities = _REALIZATIONS
        if not self.use_native_pauli_evolution:
            capabilities -= {RealizationCapability.PAULI_EVOLUTION}
        if self.use_classical_stores:
            capabilities |= {RealizationCapability.BOOLEAN_STORE}
        return capabilities

    @property
    def supports_circuit_result_postprocessing(self) -> bool:
        """Return whether artifact execution evaluates retained result trees."""

        return True

    def lower(self, program: Program) -> LoweredProgram:
        """Retain one emitted SDK circuit with its ABI facts."""

        return lower_circuit_target(program, self)

    def lower_circuit(self, logical: CircuitProgram) -> LoweredProgram:
        """Retain one verified circuit without QPY serialization."""

        circuit, stages = _emit_for_target(logical, self)
        from ._payload import QiskitExecutable

        payload = QiskitExecutable(circuit)
        return LoweredProgram(
            payload=payload,
            parameter_abi=logical.parameter_abi,
            result_abi=logical.result_abi,
            stages=(*logical.realization_stages, *stages),
            circuit_qubits=logical.width,
            circuit_gates=logical.operation_count,
            emitted_instructions=sum(circuit.count_ops().values()),
            metadata=FrozenMapping.from_mapping(
                {
                    "qiskit_version": version("qiskit"),
                    **(
                        {RESULT_POSTPROCESSING_METADATA: plan}
                        if (plan := circuit_result_postprocessing(logical)) is not None
                        else {}
                    ),
                }
            ),
        )


def _emit_for_target(
    logical: CircuitProgram, target: Qiskit
) -> tuple[Any, tuple[str, ...]]:
    """Share SDK emission and target preparation between inspection and execution.

    Args:
        logical (CircuitProgram): Verified target-neutral circuit projection.
        target (Qiskit): SDK basis, routing and realization configuration.

    Returns:
        tuple: SDK circuit and emission stage names, without an execution artifact.
    """
    from qiskit import transpile

    circuit = _emit_circuit(logical)
    stages = ["qiskit-emission"]
    if target.backend is not None:
        circuit = transpile(
            circuit,
            backend=target.backend,
            optimization_level=target.optimization_level,
            seed_transpiler=target.seed_transpiler,
            scheduling_method=target.scheduling_method,
        )
        stages.append("qiskit-routing")
        if target.scheduling_method is not None:
            stages.append("qiskit-scheduling")
    elif target.basis_gates:
        circuit = transpile(
            circuit,
            basis_gates=list(target.basis_gates),
            optimization_level=target.optimization_level,
            seed_transpiler=target.seed_transpiler,
        )
        stages.append("qiskit-transpile")
    return circuit, tuple(stages)


@dataclass(frozen=True, slots=True)
class Aer:
    """Execute retained Qiskit circuits on an Aer simulator.

    Args:
        method (str): Aer simulation method, defaulting to automatic selection.
        use_classical_stores (bool): Opt into verified top-level Boolean stores.
        optimization_level (int): Target SDK optimization level.
        seed_transpiler (int): Target SDK transpilation seed.
        realizations (RealizationOptions): Exact choices used when compiling source.
    """

    method: str = "automatic"
    use_classical_stores: bool = field(default=False, kw_only=True)
    optimization_level: int = 1
    seed_transpiler: int = 7
    realizations: RealizationOptions = RealizationOptions()

    @property
    def name(self) -> str:
        """Return the endpoint identity."""

        return f"qiskit.aer.{self.method}"

    @property
    def target(self) -> Qiskit:
        """Return the exact compiler target accepted by this endpoint."""

        return Qiskit(
            use_classical_stores=self.use_classical_stores,
            optimization_level=self.optimization_level,
            seed_transpiler=self.seed_transpiler,
            realizations=self.realizations,
        )

    @property
    def capabilities(self) -> CapabilitySnapshot:
        """Return this Aer endpoint's immutable compatibility snapshot."""

        return CapabilitySnapshot(FORMAT, _AER_CAPABILITIES, seeded_execution=True)

    def _prepare_estimation(
        self,
        source: Program,
        observable: Hamiltonian,
        plan: tuple[_MeasurementGroupPlan, ...],
        parameters: Mapping[str, object],
    ) -> Callable[[], ExpectationResult] | None:
        """Select a compatible SDK estimator before any shots are consumed.

        Args:
            source (Program): Verified state-preparation component.
            observable (Hamiltonian): Validated Hermitian observable.
            plan (tuple[_MeasurementGroupPlan, ...]): Native measurement allocation.
            parameters (Mapping[str, object]): Normalized runtime bindings.

        Returns:
            Callable[[], ExpectationResult] | None: Deferred SDK execution or decline.
        """
        from ._estimation import prepare_estimation

        return prepare_estimation(self, source, observable, plan, parameters)

    def launch(
        self,
        artifact: CompiledProgram,
        invocation: Invocation,
    ) -> Job:
        """Run an already compiled QPY artifact in this process.

        Args:
            artifact (CompiledProgram): Sealed QPY artifact.
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
            job_prefix="aer",
            execute=lambda: _execute_local(
                artifact,
                invocation,
                method=self.method,
                optimization_level=self.optimization_level,
                seed_transpiler=self.seed_transpiler,
            ),
        )


@dataclass(frozen=True, slots=True)
class QiskitBackend:
    """Execute QPY on one live Qiskit backend through the common Job ABI.

    The live SDK object never enters a QCore image or artifact. Its immutable
    target snapshot is used only while routing and fingerprinting the explicit
    compiler target returned by :attr:`target`.

    Args:
        backend: A Qiskit backend exposing ``run`` and target metadata.
        requirements: Capabilities the endpoint operator is willing to claim.
            When omitted, a conservative set is inferred from target operation
            names; dynamic control is never guessed.
        backend_snapshot: Optional operator-supplied immutable calibration or
            target identity. When omitted, a deterministic structural snapshot
            is derived from the backend target.
        optimization_level: Qiskit transpiler optimization level.
        seed_transpiler: Deterministic transpiler seed.
        scheduling_method: Optional Qiskit ``"alap"`` or ``"asap"`` method.
        profile: Explicit device facts for the routing target; omitted, the
            target derives them from the backend when it has a coupling map.

    Raises:
        TypeError: If ``backend`` does not expose the required SDK surface.
    """

    backend: Any = field(repr=False, compare=False)
    requirements: frozenset[Requirement] | None = None
    backend_snapshot: str | None = None
    optimization_level: int = 1
    seed_transpiler: int = 7
    scheduling_method: str | None = None
    profile: TargetProfile | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Validate the live boundary without importing provider packages."""

        if not callable(getattr(self.backend, "run", None)):
            raise TypeError("backend must expose run(circuit, ...)")
        if self.requirements is not None:
            requirements = frozenset(self.requirements)
            if any(not isinstance(item, Requirement) for item in requirements):
                raise TypeError("requirements must contain Requirement values")
            object.__setattr__(self, "requirements", requirements)
        _ = self.target

    @property
    def name(self) -> str:
        """Return an endpoint identity derived from the live backend."""

        return f"qiskit.backend.{_backend_name(self.backend)}"

    @property
    def target(self) -> Qiskit:
        """Return the routing target paired with this exact backend snapshot."""

        return Qiskit(
            optimization_level=self.optimization_level,
            seed_transpiler=self.seed_transpiler,
            scheduling_method=self.scheduling_method,
            backend=self.backend,
            backend_snapshot=self.backend_snapshot,
            requirements=self._requirements,
            profile=self.profile,
        )

    @property
    def capabilities(self) -> CapabilitySnapshot:
        """Return the endpoint's explicit conservative capability claim."""

        return CapabilitySnapshot(
            FORMAT,
            self._requirements,
            profile=self.target._profile,
        )

    @property
    def _requirements(self) -> frozenset[Requirement]:
        """Return explicit capabilities or infer only structural basics."""

        if self.requirements is not None:
            return self.requirements
        return _backend_requirements(self.backend)

    def launch(self, artifact: CompiledProgram, invocation: Invocation) -> Job:
        """Bind one retained routed circuit and create the provider-native job."""

        if artifact.format != FORMAT:
            raise ExecutionError(
                f"Q8214: Qiskit backend accepts {FORMAT!r}, got {artifact.format!r}"
            )
        if artifact.result_abi.width == 0:
            raise ExecutionError("Q8215: Qiskit backend execution requires sampling")
        if invocation.seed != 0:
            raise ExecutionError(
                "Q8216: live Qiskit backends do not provide a portable seed contract; "
                "use seed=0"
            )
        circuit = load_circuit(artifact)
        circuit = bind_circuit(circuit, artifact, invocation)
        native = self.backend.run(circuit, shots=invocation.shots)
        return _QiskitJob(
            native=native,
            circuit=circuit,
            artifact=artifact,
            parameters=invocation.parameters,
            shots=invocation.shots,
            backend=self.name,
        )


@dataclass(frozen=True, slots=True)
class _QiskitJob:
    """Translate one provider-native Qiskit job into the portable Job ABI."""

    native: Any = field(repr=False)
    circuit: Any = field(repr=False)
    artifact: CompiledProgram = field(repr=False)
    parameters: FrozenMapping = field(repr=False)
    shots: int
    backend: str

    @property
    def job_id(self) -> str:
        """Return the provider-scoped native job identity."""

        identity = getattr(self.native, "job_id", None)
        value = identity() if callable(identity) else identity
        if value is None:
            raise ExecutionError("Q8217: Qiskit job has no job identity")
        return str(value)

    @property
    def status(self) -> JobStatus:
        """Map Qiskit's status enum to the portable lifecycle."""

        status = getattr(self.native, "status", None)
        value = status() if callable(status) else status
        name = str(getattr(value, "name", value)).upper()
        if name in {"INITIALIZING", "QUEUED", "VALIDATING"}:
            return JobStatus.SUBMITTED
        if name == "RUNNING":
            return JobStatus.RUNNING
        if name == "DONE":
            return JobStatus.COMPLETED
        if name == "CANCELLED":
            return JobStatus.CANCELLED
        return JobStatus.FAILED

    def result(self, timeout: float | None = None) -> SampleResult:
        """Wait for native counts and decode them through the sealed ABI."""

        result_method = getattr(self.native, "result", None)
        if not callable(result_method):
            raise ExecutionError("Q8218: Qiskit job has no result() method")
        native_result = result_method(timeout=timeout)
        try:
            native_counts = native_result.get_counts(self.circuit)
        except Exception as error:
            raise ExecutionError(
                "Q8219: Qiskit job returned no sampling counts"
            ) from error
        logical_counts = decode_counts(
            self.artifact,
            native_counts,
            dict(self.parameters.entries),
            len(self.circuit.clbits),
        )
        return SampleResult.from_counts(
            backend=self.backend,
            counts=logical_counts,
            expected_width=self.artifact.result_abi.width,
        )

    def cancel(self) -> None:
        """Request best-effort cancellation from the native provider job."""

        cancel = getattr(self.native, "cancel", None)
        if not callable(cancel):
            raise ExecutionError("Q8221: Qiskit job does not support cancellation")
        cancel()

    def snapshot(self) -> ExecutionSnapshot:
        """Capture the provider job identity without retaining SDK objects.

        Returns:
            Secret-free remote execution reference.
        """

        return ExecutionSnapshot.remote(
            ExecutionReference.create(
                provider="qiskit-backend",
                job_ids=(self.job_id,),
                target=self.backend,
                context={"kind": "sample"},
            )
        )


def _execute_local(
    artifact: CompiledProgram,
    invocation: Invocation,
    *,
    method: str,
    optimization_level: int,
    seed_transpiler: int,
) -> SampleResult:
    """Execute the retained Aer-compatible template with fresh bound inputs.

    Args:
        artifact (CompiledProgram): Sealed QPY artifact and result ABI.
        invocation (Invocation): Validated parameters, shots, and seed.
        method (str): Aer simulator method.
        optimization_level (int): Endpoint synthesis optimization level.
        seed_transpiler (int): Reproducible synthesis seed.

    Returns:
        SampleResult: Source-ordered sample counts.

    Raises:
        ExecutionError: If artifact loading or result decoding fails.
    """

    if artifact.format != FORMAT:
        raise ExecutionError(f"Q8201: Aer accepts {FORMAT!r}, got {artifact.format!r}")
    from qiskit_aer import AerSimulator

    from ._payload import QiskitExecutable

    backend = AerSimulator(
        method=method,
        max_parallel_threads=1,
        max_parallel_experiments=1,
        max_parallel_shots=1,
    )
    circuit = artifact.execution_payload(QiskitExecutable).aer_circuit(
        backend, method, optimization_level, seed_transpiler
    )
    circuit = bind_circuit(circuit, artifact, invocation)
    native = backend.run(
        circuit,
        shots=invocation.shots,
        seed_simulator=_aer_seed(invocation.seed),
    ).result()
    native_counts = native.get_counts(circuit)
    logical_counts = decode_counts(
        artifact,
        native_counts,
        invocation.values(),
        len(circuit.clbits),
    )
    return SampleResult.from_counts(
        backend=f"qiskit.aer.{method}",
        counts=logical_counts,
        expected_width=artifact.result_abi.width,
    )


def _aer_seed(seed: int) -> int:
    """Project one unsigned seed to Aer's signed 64-bit Python boundary."""

    return seed if seed < 1 << 63 else seed - (1 << 64)


def decode_counts(
    artifact: CompiledProgram,
    native_counts: Mapping[object, int],
    parameters: Mapping[str, object],
    native_width: int,
) -> dict[tuple[int, ...], int]:
    """Decode Qiskit register order and evaluate an optional result tree.

    Args:
        artifact: Sealed QPY artifact carrying the logical result ABI.
        native_counts: Counts keyed by Qiskit's integer or bit-string format.
        parameters: Canonical source parameters for derived result evaluation.
        native_width: Number of classical bits in the emitted Qiskit circuit.

    Returns:
        Counts keyed in Qamomile source-result order.

    Raises:
        ExecutionError: If result postprocessing metadata is malformed.
        ValueError: If a native count key is not a binary measurement row.
    """

    plan = dict(artifact.metadata.entries).get(RESULT_POSTPROCESSING_METADATA)
    if plan is None:
        logical: dict[tuple[int, ...], int] = {}
        for raw_key, count in native_counts.items():
            bitstring = (
                format(raw_key, f"0{artifact.result_abi.native_width}b")
                if isinstance(raw_key, int)
                else str(raw_key)
            )
            bits = artifact.result_abi.decode_big_endian(bitstring)
            logical[bits] = logical.get(bits, 0) + int(count)
        return logical
    if not isinstance(plan, bytes):
        raise ExecutionError("Q8220: invalid Qiskit result postprocessing metadata")
    raw: dict[tuple[int, ...], int] = {}
    for raw_key, count in native_counts.items():
        bitstring = (
            format(raw_key, f"0{native_width}b")
            if isinstance(raw_key, int)
            else str(raw_key).replace(" ", "")
        )
        bits = tuple(int(bit) for bit in reversed(bitstring))
        raw[bits] = raw.get(bits, 0) + int(count)
    return postprocess_circuit_counts(plan, raw, parameters, artifact.result_abi)


def _free_circuit_parameters(circuit: Any) -> set[Any]:
    """Collect parameters not bound by an enclosing SDK loop scope.

    Args:
        circuit (Any): SDK circuit whose operations may include control flow.

    Returns:
        set[Any]: Parameters requiring launch bindings. A loop index used outside
        its declaring body remains free and therefore fails launch validation.
    """
    from qiskit.circuit import ControlFlowOp, ForLoopOp

    free = set(getattr(circuit.global_phase, "parameters", ()))
    for row in circuit.data:
        operation = row.operation
        if isinstance(operation, ControlFlowOp):
            for block in operation.blocks:
                nested = _free_circuit_parameters(block)
                if isinstance(operation, ForLoopOp):
                    nested.discard(operation.params[1])
                free.update(nested)
        else:
            for value in operation.params:
                free.update(getattr(value, "parameters", ()))
    return free


def bind_circuit(
    circuit: Any,
    artifact: CompiledProgram,
    invocation: Invocation,
) -> Any:
    """Bind the shared aggregate ABI projection into one Qiskit circuit.

    Args:
        circuit (Any): Qiskit circuit loaded from the sealed artifact.
        artifact (CompiledProgram): CompiledProgram carrying the authoritative parameter ABI.
        invocation (Invocation): Rust-normalized invocation values to bind.

    Returns:
        Any: Circuit with launch parameters bound and loop indices scoped.

    Raises:
        ExecutionError: If the artifact retains a parameter not supplied by the
            canonical invocation projection.
    """

    return _bind_circuit_parameters(
        circuit, artifact.parameter_abi, invocation.values()
    )


def _bind_circuit_parameters(
    circuit: Any, parameter_abi: ParameterABI, parameters: Mapping[str, object]
) -> Any:
    """Bind Rust-normalized scalar leaves into an SDK circuit.

    Args:
        circuit (Any): Unbound SDK circuit.
        parameter_abi (ParameterABI): Retained classical parameter layout.
        parameters (Mapping[str, object]): Normalized bindings for that layout.

    Returns:
        Any: A bound copy, preserving the symbolic template.

    Raises:
        ExecutionError: If the circuit retains an unbound parameter.
    """
    flattened = scalar_parameter_bindings(
        parameter_abi,
        parameters,
    )
    native_names = {parameter.name for parameter in _free_circuit_parameters(circuit)}
    bindings = {
        f"p{index}": value
        for index, (_, value) in enumerate(flattened)
        if f"p{index}" in native_names
    }
    if bindings:
        circuit = circuit.assign_parameters(bindings, inplace=False)
    if unbound := _free_circuit_parameters(circuit):
        missing = ", ".join(sorted(parameter.name for parameter in unbound))
        raise ExecutionError(f"Q8203: missing Qiskit parameters: {missing}")
    return circuit


def _backend_name(backend: Any) -> str:
    """Return one stable printable Qiskit backend name."""

    raw = getattr(backend, "name", type(backend).__name__)
    value = raw() if callable(raw) else raw
    return str(value).replace(" ", "-")


def _backend_snapshot(backend: Any | None) -> str | None:
    """Derive a deterministic structural identity for one live target."""

    if backend is None:
        return None
    target = getattr(backend, "target", None)
    operations = sorted(str(item) for item in getattr(target, "operation_names", ()))
    payload = json.dumps(
        {
            "name": _backend_name(backend),
            "num_qubits": getattr(target, "num_qubits", None),
            "operations": operations,
            "target": str(target),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def profile_from_backend(backend: Any) -> TargetProfile | None:
    """Derive an immutable target profile from a live Qiskit backend.

    The topology facet lists the target's directed coupling edges, collapsing
    pairs native in both directions into one symmetric edge, and the native
    one- and two-qubit operation names. The calibration facet records finite
    ``t1``, ``t2``, and ``frequency`` qubit properties in seconds and hertz and
    per-operation ``error`` and ``duration`` instruction properties keyed as
    ``<operation>.<quantity>``. No credential or network call is made; the
    backend object itself never enters the profile.

    Args:
        backend (Any): Backend exposing a Qiskit ``target``.

    Returns:
        TargetProfile | None: Validated profile, or None when the target has no
        qubit count or coupling map (for example an all-to-all simulator).

    Raises:
        ArtifactError: If the derived facts fail native validation.
    """

    target = getattr(backend, "target", None)
    if target is None:
        return None
    num_qubits = getattr(target, "num_qubits", None)
    build_coupling_map = getattr(target, "build_coupling_map", None)
    coupling_map = build_coupling_map() if callable(build_coupling_map) else None
    if (
        isinstance(num_qubits, bool)
        or not isinstance(num_qubits, int)
        or num_qubits <= 0
        or coupling_map is None
    ):
        return None
    directed = {
        (int(control), int(qubit)) for control, qubit in coupling_map.get_edges()
    }
    edges = []
    for control, qubit in sorted(directed):
        if (qubit, control) in directed:
            if control < qubit:
                edges.append(CouplingEdge(control, qubit, True))
        else:
            edges.append(CouplingEdge(control, qubit))
    operations: dict[str, int] = {}
    for name in getattr(target, "operation_names", ()):
        try:
            operation = target.operation_from_name(name)
        except (AttributeError, KeyError, TypeError):
            continue
        arity = getattr(operation, "num_qubits", None)
        if arity in (1, 2):
            operations[str(name)] = arity
    qubit_rows: list[QubitProperty] = []
    edge_rows: list[EdgeProperty] = []
    qubit_properties = getattr(target, "qubit_properties", None) or ()
    for index, properties in enumerate(tuple(qubit_properties)[:num_qubits]):
        for quantity, unit in (("t1", "s"), ("t2", "s"), ("frequency", "Hz")):
            value = _finite(getattr(properties, quantity, None))
            if value is not None:
                qubit_rows.append(QubitProperty(index, quantity, value, unit))
    for name, arity in sorted(operations.items()):
        try:
            table: Mapping[tuple[int, ...] | None, Any] | None = target[name]
        except (KeyError, TypeError):
            continue
        for qargs, properties in (table or {}).items():
            if qargs is None or properties is None or len(qargs) != arity:
                continue
            indices = tuple(int(qubit) for qubit in qargs)
            for quantity, unit in (("error", "probability"), ("duration", "s")):
                value = _finite(getattr(properties, quantity, None))
                if value is None:
                    continue
                label = f"{name}.{quantity}"
                if arity == 1 and indices[0] < num_qubits:
                    qubit_rows.append(QubitProperty(indices[0], label, value, unit))
                elif arity == 2 and indices in directed:
                    edge_rows.append(
                        EdgeProperty(indices[0], indices[1], label, value, unit)
                    )
    calibration = None
    if qubit_rows or edge_rows:
        calibration = TargetCalibration(
            tuple(qubit_rows), tuple(edge_rows), "qiskit.target"
        )
    return TargetProfile(
        FORMAT,
        TargetTopology(
            num_qubits,
            tuple(edges),
            tuple(sorted(name for name, arity in operations.items() if arity == 1)),
            tuple(sorted(name for name, arity in operations.items() if arity == 2)),
        ),
        calibration,
        backend_name=_backend_name(backend),
        provenance="qiskit.target",
    )


def _finite(value: object) -> float | None:
    """Return one finite numeric SDK property as float, else None."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _backend_requirements(backend: Any) -> frozenset[Requirement]:
    """Infer only capabilities represented by standard Qiskit target ops."""

    target = getattr(backend, "target", None)
    operations = {str(item).lower() for item in getattr(target, "operation_names", ())}
    requirements = {
        Requirement.CLASSICAL_COMPUTE,
        Requirement.COHERENT_CONTROL,
        Requirement.DISCARD,
        Requirement.MEASUREMENT,
        Requirement.PARAMETERIZED_EXECUTION,
        Requirement.QUANTUM,
        Requirement.ROTATION,
    }
    if "reset" in operations:
        requirements.add(Requirement.RESET)
    if "if_else" in operations:
        requirements.update(
            {
                Requirement.DYNAMIC_BRANCH,
                Requirement.MID_CIRCUIT_MEASUREMENT,
            }
        )
    if "while_loop" in operations:
        requirements.update(
            {
                Requirement.DYNAMIC_LOOP,
                Requirement.MID_CIRCUIT_MEASUREMENT,
            }
        )
    return frozenset(requirements)


def to_circuit(
    source: Program | ProgramFamily | QKernel,
    *,
    target: Qiskit | None = None,
    parameters: Mapping[str, object] | None = None,
    **bindings: object,
) -> Any:
    """Inspect a Qamomile component as an SDK circuit without making an executable.

    Quantum-returning preparations are valid here. This performs the same Rust
    circuit projection and SDK emission used by compilation, but creates no
    CompiledProgram, result execution contract, or resource report. No measurement
    is added and no device is executed. SDK matrix/statevector tools impose their
    own supported-instruction restrictions on the resulting circuit.

    Args:
        source (Program | ProgramFamily | QKernel): Component to inspect.
        target (Qiskit | None): Emission configuration; defaults to Qiskit().
        parameters (Mapping[str, object] | None): Complete runtime bindings for
            a concrete circuit, or None to retain supported symbolic parameters.
        **bindings (object): Const bindings used to specialize the source.

    Returns:
        Any: Independent Qiskit QuantumCircuit for display or SDK analysis.

    Raises:
        TypeError: If target or source has an invalid Python shape.
        CapabilityError: If the source cannot be represented as a circuit.
        ExecutionError: If parameter binding fails.
    """
    if target is None:
        target = Qiskit()
    if not isinstance(target, Qiskit):
        raise TypeError("target must be a Qiskit profile")
    program = _close_program(source, bindings)
    capabilities = target.capabilities
    if parameters is None:
        logical = lower_to_circuit(
            program,
            realization_capabilities=capabilities.realization_capabilities,
            realizations=capabilities.realizations,
        )
        return _emit_for_target(logical, target)[0]
    normalized = dict(ParameterABI.from_program(program).bind(parameters))
    logical = lower_to_circuit_at(
        program,
        normalized,
        realization_capabilities=capabilities.realization_capabilities,
        realizations=capabilities.realizations,
    )
    retained = {
        slot.name: normalized[slot.name] for slot in logical.parameter_abi.slots
    }
    return _bind_circuit_parameters(
        _emit_for_target(logical, target)[0], logical.parameter_abi, retained
    )


def load_circuit(artifact: CompiledProgram) -> Any:
    """Copy the retained Qiskit template, without restoring saved data."""

    if artifact.format != FORMAT:
        raise ExecutionError(f"Q8204: expected {FORMAT!r}, got {artifact.format!r}")
    from ._payload import QiskitExecutable

    return artifact.execution_payload(QiskitExecutable).circuit()


from .runtime import IBMRuntime as IBMRuntime  # noqa: E402

for _public_symbol in (
    Qiskit,
    Aer,
    QiskitBackend,
    bind_circuit,
    decode_counts,
    load_circuit,
    to_circuit,
):
    _public_symbol.__module__ = __package__
del _public_symbol

__all__ = [
    "Aer",
    "FORMAT",
    "IBMRuntime",
    "Qiskit",
    "QiskitBackend",
    "bind_circuit",
    "decode_counts",
    "load_circuit",
    "profile_from_backend",
    "to_circuit",
]
