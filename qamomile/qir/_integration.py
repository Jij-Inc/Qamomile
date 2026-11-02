"""Integrate QIR export with Qamomile compilation and optional persistence.

This standalone target consumes a native circuit without advertising the
CircuitTarget protocol: host-dispatched families and segmented shot plans
require an execution endpoint and cannot be exported as one QIR module.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.metadata import version

from qamomile.circuit import lower_to_circuit
from qamomile.compilation import (
    CapabilitySnapshot,
    CompiledProgram,
    LoweredProgram,
    _target_optimization_stages,
)
from qamomile.effects import Requirement
from qamomile.errors import ExecutionError
from qamomile.ir import FrozenMapping, Program
from qamomile.lowering import RealizationOptions

from ._emission import OUTPUT_LABELING_SCHEMA, emit_module
from ._payload import QIRExecutable

FORMAT = "qir.bitcode.v1"
_CAPABILITIES = frozenset(
    {
        Requirement.CLASSICAL_COMPUTE,
        Requirement.COHERENT_CONTROL,
        Requirement.QUANTUM,
        Requirement.MEASUREMENT,
        Requirement.ROTATION,
    }
)


@dataclass(frozen=True, slots=True)
class QIR:
    """Compile static execution entries to QIR 1.0 Base Profile bitcode.

    Specialize all Const inputs before compilation. Runtime parameters, reset,
    discard, feedback, gates after measurement, and computed classical results
    are unsupported. The emitted QIS contains Clifford, T, Pauli rotations,
    CNOT, Toffoli and SWAP calls; an external consumer must support those used
    by its module. Overall global phase is omitted at this classical-output
    boundary; controlled relative phases are preserved.

    Args:
        realizations (RealizationOptions): Exact native realization choices.
    """

    realizations: RealizationOptions = RealizationOptions()

    @property
    def name(self) -> str:
        """Return the LLVM bitcode artifact format.

        Returns:
            str: Stable QIR format identifier.
        """
        return FORMAT

    @property
    def capabilities(self) -> CapabilitySnapshot:
        """Return the static Base Profile capability contract.

        Returns:
            CapabilitySnapshot: Requirements and native realization choices.
        """
        return CapabilitySnapshot(
            FORMAT,
            _CAPABILITIES,
            realizations=self.realizations,
            circuit_optimization=True,
        )

    @property
    def fingerprint(self) -> str:
        """Identify the emitter, SDK version and realization settings.

        Returns:
            str: Stable hash of the compilation configuration.
        """
        identity = f"qir-v1:base_profile:{version('pyqir')}:{OUTPUT_LABELING_SCHEMA}:{self.realizations.overrides!r}"
        return "qir-" + hashlib.sha256(identity.encode()).hexdigest()[:20]

    def lower(self, program: Program) -> LoweredProgram:
        """Emit the native circuit projection as one standalone QIR entry.

        Args:
            program (Program): Specialized, verified execution entry.

        Returns:
            LoweredProgram: Bitcode with unchanged native result and parameter ABIs.

        Raises:
            CapabilityError: If the circuit exceeds the Base Profile or budget.
            ImportError: If PyQIR is unavailable.
        """
        logical = lower_to_circuit(program, realizations=self.realizations)
        payload, emitted = emit_module(logical)
        return LoweredProgram(
            payload=payload,
            parameter_abi=logical.parameter_abi,
            result_abi=logical.result_abi,
            stages=(
                "qcore-restored",
                *_target_optimization_stages(program),
                "circuit-projection",
                *logical.realization_stages,
                "qir-base-profile-emission",
            ),
            circuit_qubits=logical.width,
            circuit_gates=logical.operation_count,
            emitted_instructions=emitted,
            metadata=FrozenMapping.from_mapping(
                {
                    "pyqir_version": version("pyqir"),
                    "qir_version": "1.0",
                    "qir_profile": "base_profile",
                    "qir_entrypoint": "main",
                    "qir_output_labeling_schema": OUTPUT_LABELING_SCHEMA,
                }
            ),
        )


def load_bitcode(artifact: CompiledProgram) -> bytes:
    """Return standalone LLVM bitcode for a compiled QIR entry.

    Args:
        artifact (CompiledProgram): QIR artifact produced by qm.compile.

    Returns:
        bytes: LLVM bitcode suitable for writing to a .bc file.

    Raises:
        ExecutionError: If the artifact is not a concrete QIR module.
    """
    if artifact.format != FORMAT:
        raise ExecutionError(f"Q9006: expected {FORMAT!r}, got {artifact.format!r}")
    return artifact.execution_payload(QIRExecutable).bitcode


def load_ir(artifact: CompiledProgram) -> str:
    """Return human-readable LLVM IR for a compiled QIR entry.

    Args:
        artifact (CompiledProgram): QIR artifact produced by qm.compile.

    Returns:
        str: LLVM IR suitable for writing to a .ll file.

    Raises:
        ExecutionError: If the artifact is not a valid concrete QIR module.
        ImportError: If PyQIR is unavailable.
    """
    load_bitcode(artifact)
    return str(artifact.execution_payload(QIRExecutable).module())


for _public_symbol in (QIR, load_bitcode, load_ir):
    _public_symbol.__module__ = __package__
del _public_symbol

__all__ = ["FORMAT", "QIR", "load_bitcode", "load_ir"]
