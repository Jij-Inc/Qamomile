"""CompositeGate operation for representing complex multi-gate operations."""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING, Any

from qamomile.circuit.ir.types.primitives import BlockType, QubitType

from .operation import Operation, OperationKind, ParamHint, Signature

if TYPE_CHECKING:
    from qamomile.circuit.ir.block_value import BlockValue
    from qamomile.circuit.ir.value import Value


class CompositeGateType(enum.Enum):
    """Registry of known composite gate types."""

    QPE = "qpe"  # Quantum Phase Estimation
    QFT = "qft"  # Quantum Fourier Transform
    IQFT = "iqft"  # Inverse QFT
    CUSTOM = "custom"  # User-defined composite gate


@dataclasses.dataclass
class ResourceMetadata:
    """Resource estimation metadata for composite gates.

    Attributes:
        query_complexity: Number of oracle/unitary queries
        t_gate_count: Estimated T-gate count
        ancilla_qubits: Number of ancilla qubits required
        custom_metadata: Additional user-defined metadata
    """

    query_complexity: int | None = None
    t_gate_count: int | None = None
    ancilla_qubits: int = 0
    custom_metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class CompositeGateOperation(Operation):
    """Represents a composite gate (QPE, QFT, etc.) as a single operation.

    CompositeGate allows representing complex multi-gate operations as a single
    atomic operation in the IR. This enables:
    - Resource estimation without full implementation
    - Backend-native conversion (e.g., Qiskit's QPE)
    - User-defined complex gates

    The operands structure depends on has_implementation:
    - If has_implementation=True:
        - operands[0]: BlockValue (the implementation)
        - operands[1:1+num_control_qubits]: Control qubits (if any)
        - operands[1+num_control_qubits:1+num_control_qubits+num_target_qubits]: Target qubits
        - operands[1+num_control_qubits+num_target_qubits:]: Parameters
    - If has_implementation=False (stub):
        - operands[0:num_control_qubits]: Control qubits (if any)
        - operands[num_control_qubits:num_control_qubits+num_target_qubits]: Target qubits
        - operands[num_control_qubits+num_target_qubits:]: Parameters

    The results structure:
    - results[0:num_control_qubits]: Control qubits (returned)
    - results[num_control_qubits:]: Target qubits (returned)

    Attributes:
        gate_type: The type of composite gate (QPE, QFT, CUSTOM, etc.)
        num_control_qubits: Number of control qubits
        num_target_qubits: Number of target qubits
        custom_name: Name for CUSTOM gate types (used for identification)
        resource_metadata: Optional resource estimation metadata
        has_implementation: Whether this operation has an implementation BlockValue
        composite_gate_instance: Optional reference to the CompositeGate class instance
            that created this operation (for accessing _decompose() at emit time)
    """

    gate_type: CompositeGateType = CompositeGateType.CUSTOM
    num_control_qubits: int = 0
    num_target_qubits: int = 0
    custom_name: str = ""
    resource_metadata: ResourceMetadata | None = None
    has_implementation: bool = True
    composite_gate_instance: Any = None  # Reference to CompositeGate instance

    @property
    def implementation(self) -> "BlockValue | None":
        """Get the implementation BlockValue, if any."""
        if not self.has_implementation or not self.operands:
            return None
        from qamomile.circuit.ir.block_value import BlockValue

        impl = self.operands[0]
        if isinstance(impl, BlockValue):
            return impl
        return None

    @property
    def control_qubits(self) -> list["Value"]:
        """Get the control qubit operands."""
        start = 1 if self.has_implementation else 0
        return list(self.operands[start : start + self.num_control_qubits])

    @property
    def target_qubits(self) -> list["Value"]:
        """Get the target qubit operands."""
        start = (1 if self.has_implementation else 0) + self.num_control_qubits
        end = start + self.num_target_qubits
        return list(self.operands[start:end])

    @property
    def parameters(self) -> list["Value"]:
        """Get the parameter operands (angles, etc.)."""
        start = (
            (1 if self.has_implementation else 0)
            + self.num_control_qubits
            + self.num_target_qubits
        )
        return list(self.operands[start:])

    @property
    def name(self) -> str:
        """Human-readable name of this composite gate."""
        if self.gate_type == CompositeGateType.CUSTOM:
            return self.custom_name or "custom"
        return self.gate_type.value

    @property
    def signature(self) -> Signature:
        """Return the operation signature."""
        operand_hints: list[ParamHint | None] = []

        if self.has_implementation:
            operand_hints.append(ParamHint("implementation", BlockType()))

        for i in range(self.num_control_qubits):
            operand_hints.append(ParamHint(f"control_{i}", QubitType()))

        for i in range(self.num_target_qubits):
            operand_hints.append(ParamHint(f"target_{i}", QubitType()))

        # Parameters - use their actual types
        for i, param in enumerate(self.parameters):
            operand_hints.append(ParamHint(f"param_{i}", param.type))

        result_hints = [
            ParamHint(f"control_out_{i}", QubitType())
            for i in range(self.num_control_qubits)
        ] + [
            ParamHint(f"target_out_{i}", QubitType())
            for i in range(self.num_target_qubits)
        ]

        return Signature(operands=operand_hints, results=result_hints)

    @property
    def operation_kind(self) -> OperationKind:
        """Return the operation kind (always QUANTUM)."""
        return OperationKind.QUANTUM
