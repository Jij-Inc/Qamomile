"""Integrate native Quration IR with the ordinary artifact lifecycle."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, replace
from importlib.metadata import version
from typing import Any

from qamomile.circuit import (
    RESULT_POSTPROCESSING_FORMAT,
    RESULT_POSTPROCESSING_METADATA,
    CircuitProgram,
    circuit_result_postprocessing,
    lower_circuit_target,
    postprocess_circuit_counts,
)
from qamomile.compilation import (
    CapabilitySnapshot,
    CompiledProgram,
    Invocation,
    Job,
    LoweredProgram,
    SampleResult,
)
from qamomile.effects import Requirement
from qamomile.errors import ExecutionError
from qamomile.ir import FrozenMapping, Program
from qamomile.local_execution import launch_local_sample
from qamomile.lowering import RealizationOptions
from qamomile.persistence import export_payload

from ._emission import emit_module

FORMAT = "quration.ir-json.v1"
_CAPABILITIES = frozenset(
    {
        Requirement.CLASSICAL_COMPUTE,
        Requirement.COHERENT_CONTROL,
        Requirement.QUANTUM,
        Requirement.MEASUREMENT,
        Requirement.MID_CIRCUIT_MEASUREMENT,
        Requirement.ROTATION,
    }
)


@dataclass(frozen=True, slots=True)
class Quration:
    """Compile concrete QCore circuits to retained native PyQret objects.

    Args:
        rotation_precision (float): Per-rotation FTQC synthesis tolerance in
            (0, 1). This is not a whole-program error bound. Local simulation
            uses unsynthesized rotations.
        realizations (RealizationOptions): Exact compiler realization choices.

    Raises:
        ValueError: If precision is nonfinite or outside (0, 1).
    """

    rotation_precision: float = 1e-10
    realizations: RealizationOptions = RealizationOptions()

    def __post_init__(self) -> None:
        """Validate the explicit target synthesis tolerance."""
        if (
            type(self.rotation_precision) not in (int, float)
            or not math.isfinite(self.rotation_precision)
            or not 0 < self.rotation_precision < 1
        ):
            raise ValueError(
                "rotation_precision must be finite and between zero and one"
            )

    @property
    def name(self) -> str:
        """Return the native module artifact format.

        Returns:
            str: Stable Quration JSON format identifier.
        """
        return FORMAT

    @property
    def capabilities(self) -> CapabilitySnapshot:
        """Return the static circuit capability contract.

        Returns:
            CapabilitySnapshot: Supported requirements and exact realization options.
        """
        return CapabilitySnapshot(FORMAT, _CAPABILITIES, realizations=self.realizations)

    @property
    def fingerprint(self) -> str:
        """Bind emission to the SDK, tolerance and realization configuration.

        Returns:
            str: Versioned hash used by the artifact execution contract.
        """
        identity = f"quration-v1:{version('pyqret')}:{RESULT_POSTPROCESSING_FORMAT}:{self.rotation_precision!r}:{self.realizations.overrides!r}"
        return "quration-" + hashlib.sha256(identity.encode()).hexdigest()[:20]

    @property
    def supports_circuit_result_postprocessing(self) -> bool:
        """Opt in to the existing Rust-owned sampled-result evaluator.

        Returns:
            bool: True for the shared sampled-result plan.
        """
        return True

    def lower(self, program: Program) -> LoweredProgram:
        """Project a verified program through the shared Rust circuit path.

        Args:
            program (Program): Specialized, verified entry program.

        Returns:
            LoweredProgram: Native circuit owner with unchanged logical ABIs.

        Raises:
            CapabilityError: If the source cannot be emitted by this target.
        """
        return lower_circuit_target(program, self)

    def lower_circuit(self, logical: CircuitProgram) -> LoweredProgram:
        """Emit the already projected static circuit through PyQret.

        Args:
            logical (CircuitProgram): Checked projection from the Rust compiler.

        Returns:
            LoweredProgram: Retained SDK objects and native-derived ABI facts.

        Raises:
            CapabilityError: If gates or launch parameters are unsupported.
        """
        payload, emitted = emit_module(logical, self.rotation_precision)
        metadata: dict[str, Any] = {
            "pyqret_version": version("pyqret"),
            "quration_measurement_count": logical.measurement_count,
            "quration_width": logical.width,
        }
        plan = circuit_result_postprocessing(logical)
        if plan is not None:
            metadata[RESULT_POSTPROCESSING_METADATA] = plan
        return LoweredProgram(
            payload=payload,
            parameter_abi=logical.parameter_abi,
            result_abi=logical.result_abi,
            stages=(*logical.realization_stages, "quration-native-json-emission"),
            circuit_qubits=logical.width,
            circuit_gates=logical.operation_count,
            emitted_instructions=emitted,
            metadata=FrozenMapping.from_mapping(metadata),
        )


def load_circuit(artifact: CompiledProgram) -> Any:
    """Expose the retained context-owned SDK circuit from a Quration artifact.

    Treat this borrowed SDK view as read-only. ``compile_resources`` creates an
    independent module for the external compiler's mutating machine-IR pass.

    Args:
        artifact (CompiledProgram): Concrete native Quration module artifact.

    Returns:
        Any: PyQret Circuit retaining its context and module owners.

    Raises:
        ExecutionError: If the format or native module is invalid. Families
            and launch materializers must be resolved before SDK inspection.
        ImportError: If PyQret is unavailable.
    """
    from ._payload import QurationExecutable

    if artifact.format != FORMAT:
        raise ExecutionError(f"Q8905: expected {FORMAT!r}, got {artifact.format!r}")
    return artifact.execution_payload(QurationExecutable).circuit


@dataclass(frozen=True, slots=True)
class QurationLocal:
    """Execute Quration artifacts in the PyQret full-quantum simulator.

    Args:
        max_qubits (int): Statevector allocation limit, default 20.
        rotation_precision (float): Compilation tolerance for source inputs.
        realizations (RealizationOptions): Exact compiler realization choices.

    Raises:
        ValueError: If the qubit limit or rotation tolerance is invalid.
    """

    max_qubits: int = 20
    rotation_precision: float = 1e-10
    realizations: RealizationOptions = RealizationOptions()

    def __post_init__(self) -> None:
        """Validate simulation bounds before any native allocation."""
        if type(self.max_qubits) is not int or not 1 <= self.max_qubits <= 24:
            raise ValueError("max_qubits must be between 1 and 24")
        Quration(self.rotation_precision, self.realizations)

    @property
    def name(self) -> str:
        """Return the local simulator provenance name.

        Returns:
            str: Stable local endpoint identifier.
        """
        return "quration.local"

    @property
    def target(self) -> Quration:
        """Return the matching concrete circuit compiler target.

        Returns:
            Quration: Target with this endpoint's compilation settings.
        """
        return Quration(self.rotation_precision, self.realizations)

    @property
    def capabilities(self) -> CapabilitySnapshot:
        """Return requirements accepted by the local simulator.

        Returns:
            CapabilitySnapshot: The endpoint's supported execution requirements.
        """
        return replace(self.target.capabilities, seeded_execution=True)

    def launch(self, artifact: CompiledProgram, invocation: Invocation) -> Job:
        """Sample a sealed circuit with bounded native state allocation.

        Args:
            artifact (CompiledProgram): Concrete native circuit artifact.
            invocation (Invocation): Validated shots, seed and parameter ABI.

        Returns:
            Job: Completed local sampling job.

        Raises:
            ExecutionError: If the artifact, bounds or SDK results are invalid.
        """
        return launch_local_sample(
            artifact,
            invocation,
            endpoint=self.name,
            artifact_format=FORMAT,
            job_prefix="quration-local",
            execute=lambda: _sample(artifact, invocation, self.max_qubits),
        )


def _sample(
    artifact: CompiledProgram, invocation: Invocation, max_qubits: int
) -> SampleResult:
    """Normalize chronological SDK measurements before the shared result codec.

    Args:
        artifact (CompiledProgram): Concrete circuit and compiler-derived metadata.
        invocation (Invocation): Validated execution arguments.
        max_qubits (int): Statevector allocation limit.

    Returns:
        SampleResult: Source-ordered logical counts.

    Raises:
        ExecutionError: If metadata, allocation limits or SDK output are invalid.
    """
    from pyqret.runtime import Simulator, SimulatorConfig

    metadata = dict(artifact.metadata.entries)
    width = metadata.get("quration_width")
    measured = metadata.get("quration_measurement_count")
    if type(width) is not int or type(measured) is not int or width < 0 or measured < 0:
        raise ExecutionError("Q8907: invalid Quration layout metadata")
    if width > max_qubits:
        raise ExecutionError(
            f"Q8908: Quration simulation requires {width} qubits; limit is {max_qubits}"
        )
    from ._payload import QurationExecutable

    circuit = artifact.execution_payload(QurationExecutable).circuit
    raw: Counter[tuple[int, ...]] = Counter()
    for shot in range(invocation.shots):
        config = SimulatorConfig(seed=(invocation.seed + shot) % (1 << 64))
        result = Simulator(config, circuit).run()
        bits = result.get("c", [])
        if len(bits) != measured or any(type(bit) is not bool for bit in bits):
            raise ExecutionError("Q8909: malformed PyQret measurement row")
        raw[tuple(int(bit) for bit in bits)] += 1
    plan = metadata.get(RESULT_POSTPROCESSING_METADATA)
    if plan is not None:
        if not isinstance(plan, bytes):
            raise ExecutionError("Q8910: malformed Quration result plan")
        counts = postprocess_circuit_counts(
            plan, raw, invocation.values(), artifact.result_abi
        )
    else:
        projected: Counter[tuple[int, ...]] = Counter()
        for bits, count in raw.items():
            projected[
                tuple(bits[index] for index in artifact.result_abi.source_order)
            ] += count
        counts = dict(projected)
    return SampleResult.from_counts(
        backend="quration.local",
        counts=counts,
        expected_width=artifact.result_abi.width,
    )


@dataclass(frozen=True, slots=True)
class QurationResourceResult:
    """Retain process-local SDK results from FTQC machine compilation.

    Attributes:
        circuit (Any): Fresh SDK circuit, including its native ownership graph.
        compiler (Any): SDK compiler retaining the selected machine options.
        compile_result (Any): Native pass execution report.
        compile_info (Any): Machine-specific Quration resource information.

    These SDK objects are not a portable ResourceStudy, and their estimates do
    not replace Qamomile's logical resource facts or claim a certified bound.
    """

    circuit: Any
    compiler: Any
    compile_result: Any
    compile_info: Any


def compile_resources(artifact: CompiledProgram, option: Any) -> QurationResourceResult:
    """Compile a concrete artifact for an explicitly configured FTQC machine.

    Args:
        artifact (CompiledProgram): Native Quration artifact from qm.compile.
        option (Any): PyQret CompileOption with an SC_LS_FIXED_V0 topology.

    Returns:
        QurationResourceResult: SDK circuit, compiler and resource information.

    Raises:
        ExecutionError: If the artifact is incompatible or SDK compilation fails.
        TypeError: If option is not a PyQret CompileOption.
        ValueError: If no FTQC machine configuration is supplied.
    """
    from pyqret.backend import CompileOption, Compiler

    if not isinstance(option, CompileOption):
        raise TypeError("option must be a pyqret.backend.CompileOption")
    try:
        option.sc_ls_fixed_v0_option
    except RuntimeError as error:
        raise ValueError(
            "compile_resources requires an FTQC machine configuration"
        ) from error
    from ._payload import QurationExecutable

    if artifact.format != FORMAT:
        raise ExecutionError(f"Q8905: expected {FORMAT!r}, got {artifact.format!r}")
    # The external compiler attaches machine IR to its input. Give that boundary
    # an independent module so repeated simulation retains the original circuit.
    circuit = QurationExecutable.from_bytes(export_payload(artifact)).circuit
    try:
        compiler = Compiler(option)
        result = compiler.compile(circuit)
        info = compiler.get_compile_info()
    except RuntimeError as error:
        raise ExecutionError(
            f"Q8911: Quration resource compilation failed: {error}"
        ) from error
    return QurationResourceResult(circuit, compiler, result, info)


for _public_symbol in (
    Quration,
    QurationLocal,
    QurationResourceResult,
    compile_resources,
    load_circuit,
):
    _public_symbol.__module__ = __package__
del _public_symbol

__all__ = [
    "FORMAT",
    "Quration",
    "QurationLocal",
    "QurationResourceResult",
    "compile_resources",
    "load_circuit",
]
