"""Cast operation for type conversions over the same quantum resources."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .operation import Operation, OperationKind, ParamHint, Signature

if TYPE_CHECKING:
    from qamomile.circuit.ir.types import ValueType


@dataclass
class CastOperation(Operation):
    """Type cast operation for creating aliases over the same quantum resources.

    This operation does NOT allocate new qubits. It creates a new Value
    that references the same underlying quantum resources with a different type.

    Use cases:
    - Vector[Qubit] -> QFixed (after QPE, for phase measurement)
    - Vector[Qubit] -> QUInt (for quantum arithmetic)
    - QUInt -> QFixed (reinterpret bits with different encoding)
    - QFixed -> QUInt (reinterpret bits with different encoding)

    Attributes:
        source_type: The type being cast from
        target_type: The type being cast to
        qubit_mapping: List of source qubit UUIDs that the target references

    operands: [source_value] - The value being cast
    results: [cast_result] - The new value with target type (same physical qubits)
    """

    source_type: "ValueType | None" = None
    target_type: "ValueType | None" = None
    qubit_mapping: list[str] = field(default_factory=list)

    @property
    def signature(self) -> Signature:
        """Return the type signature of this cast operation."""
        return Signature(
            operands=[
                ParamHint(name="source", type=self.source_type)
                if self.source_type
                else None
            ],
            results=[ParamHint(name="cast_result", type=self.target_type)]
            if self.target_type
            else [],
        )

    @property
    def operation_kind(self) -> OperationKind:
        """Cast stays in the same segment as its source (QUANTUM for quantum types)."""
        return OperationKind.QUANTUM

    @property
    def num_qubits(self) -> int:
        """Number of qubits involved in the cast."""
        return len(self.qubit_mapping)
