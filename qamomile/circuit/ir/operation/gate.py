from __future__ import annotations

import dataclasses
import enum
import typing

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.types import QFixedType
from qamomile.circuit.ir.types.primitives import (
    BitType,
    FloatType,
    QubitType,
)
from qamomile.circuit.ir.value import Value, ValueBase

from .operation import Operation, OperationKind, ParamHint, Signature


class GateOperationType(enum.Enum):
    H = enum.auto()
    X = enum.auto()
    Y = enum.auto()
    Z = enum.auto()
    T = enum.auto()
    TDG = enum.auto()  # T† = P(-π/4)
    S = enum.auto()
    SDG = enum.auto()  # S† = P(-π/2)
    P = enum.auto()  # Phase gate: P(θ)|1⟩ = e^{iθ}|1⟩
    RX = enum.auto()
    RY = enum.auto()
    RZ = enum.auto()
    CX = enum.auto()
    CZ = enum.auto()
    SWAP = enum.auto()
    CP = enum.auto()
    RZZ = enum.auto()
    TOFFOLI = enum.auto()


_ROTATION_GATES = frozenset(
    {
        GateOperationType.RX,
        GateOperationType.RY,
        GateOperationType.RZ,
        GateOperationType.P,
        GateOperationType.CP,
        GateOperationType.RZZ,
    }
)


@dataclasses.dataclass
class GateOperation(Operation):
    """Quantum gate operation.

    For rotation gates (RX, RY, RZ, P, CP, RZZ), the angle parameter is
    stored as the **last element** of ``operands``.  Use the ``theta``
    property for typed read access and the ``rotation`` / ``fixed`` factory
    class-methods for type-safe construction.
    """

    gate_type: GateOperationType | None = None

    def __post_init__(self):
        if not self.gate_type:
            raise ValueError("gate_type must be specified for GateOperation.")

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------
    @classmethod
    def fixed(
        cls,
        gate_type: GateOperationType,
        qubits: list[Value],
        results: list[Value],
    ) -> "GateOperation":
        """Create a fixed gate (H, X, CX, SWAP, …) with no angle parameter."""
        return cls(gate_type=gate_type, operands=list(qubits), results=list(results))

    @classmethod
    def rotation(
        cls,
        gate_type: GateOperationType,
        qubits: list[Value],
        theta: Value,
        results: list[Value],
    ) -> "GateOperation":
        """Create a rotation gate (RX, RY, RZ, P, CP, RZZ) with an angle."""
        return cls(
            gate_type=gate_type,
            operands=[*qubits, theta],
            results=list(results),
        )

    # ------------------------------------------------------------------
    # Typed accessors
    # ------------------------------------------------------------------
    @property
    def theta(self) -> Value | None:
        """Angle parameter for rotation gates, or ``None`` for fixed gates."""
        if self.gate_type in _ROTATION_GATES and self.operands:
            return self.operands[-1]
        return None

    @property
    def qubit_operands(self) -> list[Value]:
        """Qubit operands (excluding the theta parameter if present)."""
        if self.theta is not None:
            return self.operands[:-1]
        return list(self.operands)

    @property
    def signature(self) -> Signature:
        qubit_ops = self.qubit_operands
        hints: list[ParamHint | None] = [
            ParamHint(name=f"qubit_{i}", type=QubitType())
            for i in range(len(qubit_ops))
        ]
        if self.theta is not None:
            hints.append(ParamHint(name="theta", type=FloatType()))
        return Signature(
            operands=hints,
            results=[
                ParamHint(name=f"qubit_{i}", type=QubitType())
                for i in range(len(qubit_ops))
            ],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.QUANTUM


@dataclasses.dataclass
class MeasureOperation(Operation):
    @property
    def signature(self) -> Signature:
        return Signature(
            operands=[ParamHint(name="qubit", type=QubitType())],
            results=[ParamHint(name="bit", type=BitType())],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.HYBRID


@dataclasses.dataclass
class ControlledUOperation(Operation):
    """Base class for controlled-U operations.

    Three concrete subclasses handle distinct operand layouts:

    - ``ConcreteControlledU``: Fixed ``num_controls: int``, individual qubit
      operands.
    - ``SymbolicControlledU``: Symbolic ``num_controls: Value``, vector-based
      control operands.
    - ``IndexSpecControlledU``: Single vector with explicit index lists
      selecting which elements are controls/targets.

    All ``isinstance(op, ControlledUOperation)`` checks match every subclass.

    Attributes:
        power: Number of times to apply U (default 1). For QPE this is
            ``2**k``. Symbolic ``Value`` is resolved during constant folding.
        block: The unitary ``Block`` to apply conditionally.
    """

    power: int | Value = 1
    block: Block | None = None

    @property
    def has_index_spec(self) -> bool:
        """Whether target/control positions are specified via index lists."""
        return False

    @property
    def is_symbolic_num_controls(self) -> bool:
        """Whether num_controls is symbolic (Value) rather than concrete."""
        return False

    @property
    def control_operands(self) -> list[Value]:
        """Get the control qubit values."""
        raise NotImplementedError  # pragma: no cover

    @property
    def target_operands(self) -> list[Value]:
        """Get the target qubit values (arguments to U)."""
        raise NotImplementedError  # pragma: no cover

    @property
    def param_operands(self) -> list[Value]:
        """Get parameter operands (non-qubit, non-block)."""
        raise NotImplementedError  # pragma: no cover

    @property
    def signature(self) -> Signature:
        raise NotImplementedError  # pragma: no cover

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.QUANTUM

    def all_input_values(self) -> list[ValueBase]:
        values = super().all_input_values()
        if isinstance(self.power, Value):
            values.append(self.power)
        return values

    def _replace_power(
        self, result: "ControlledUOperation", mapping: dict[str, ValueBase]
    ) -> tuple["ControlledUOperation", bool]:
        """Shared helper: replace power if it's a mapped Value."""
        if isinstance(result.power, Value) and result.power.uuid in mapping:
            mapped = mapping[result.power.uuid]
            if isinstance(mapped, Value):
                return dataclasses.replace(result, power=mapped), True
        return result, False

    def replace_values(self, mapping: dict[str, ValueBase]) -> Operation:
        result = super().replace_values(mapping)
        assert isinstance(result, ControlledUOperation)
        result, _ = self._replace_power(result, mapping)
        return result


@dataclasses.dataclass
class ConcreteControlledU(ControlledUOperation):
    """Controlled-U with concrete (int) number of controls.

    Operand layout: ``[ctrl_0, ..., ctrl_n, tgt_0, ..., tgt_m, params...]``
    Result layout:  ``[ctrl_0', ..., ctrl_n', tgt_0', ..., tgt_m']``
    """

    num_controls: int = 1

    @property
    def control_operands(self) -> list[Value]:
        return self.operands[: self.num_controls]

    @property
    def target_operands(self) -> list[Value]:
        return self.operands[self.num_controls :]

    @property
    def param_operands(self) -> list[Value]:
        return [
            op for op in self.operands[self.num_controls :] if op.type.is_classical()
        ]

    @property
    def signature(self) -> Signature:
        nc = self.num_controls
        return Signature(
            operands=[
                *[ParamHint(name=f"control_{i}", type=QubitType()) for i in range(nc)],
                *[
                    ParamHint(name=f"arg_{i}", type=op.type)
                    for i, op in enumerate(self.operands[nc:])
                ],
            ],
            results=[
                *[ParamHint(name=f"control_{i}", type=QubitType()) for i in range(nc)],
                *[
                    ParamHint(name=f"target_{i}", type=r.type)
                    for i, r in enumerate(self.results[nc:])
                ],
            ],
        )


@dataclasses.dataclass
class SymbolicControlledU(ControlledUOperation):
    """Controlled-U with symbolic (Value) number of controls.

    Operand layout: ``[ctrl_vector, tgt_0, ..., tgt_m, params...]``
    Result layout:  ``[ctrl_vector', tgt_0', ..., tgt_m']``
    """

    num_controls: Value = dataclasses.field(
        default_factory=lambda: Value(type=FloatType(), name="_placeholder")
    )  # type: ignore[assignment]

    @property
    def is_symbolic_num_controls(self) -> bool:
        return True

    @property
    def control_operands(self) -> list[Value]:
        return [self.operands[0]]

    @property
    def target_operands(self) -> list[Value]:
        return self.operands[1:]

    @property
    def param_operands(self) -> list[Value]:
        return [op for op in self.operands[1:] if op.type.is_classical()]

    @property
    def signature(self) -> Signature:
        raise NotImplementedError("Cannot compute signature for SymbolicControlledU.")

    def all_input_values(self) -> list[ValueBase]:
        values = super().all_input_values()
        values.append(self.num_controls)
        return values

    def replace_values(self, mapping: dict[str, ValueBase]) -> Operation:
        result = super().replace_values(mapping)
        assert isinstance(result, SymbolicControlledU)
        if result.num_controls.uuid in mapping:
            mapped = mapping[result.num_controls.uuid]
            if isinstance(mapped, Value):
                return dataclasses.replace(result, num_controls=mapped)
        return result


@dataclasses.dataclass
class IndexSpecControlledU(ControlledUOperation):
    """Controlled-U with explicit target/control index specification.

    A single vector covers both controls and targets; the partition is
    determined by ``target_indices`` or ``controlled_indices``.

    Operand layout: ``[vector, params...]``
    Result layout:  ``[vector']``
    """

    num_controls: int | Value = 1
    target_indices: list[Value] | None = None
    controlled_indices: list[Value] | None = None

    @property
    def has_index_spec(self) -> bool:
        return True

    @property
    def is_symbolic_num_controls(self) -> bool:
        return isinstance(self.num_controls, Value)

    @property
    def control_operands(self) -> list[Value]:
        return [self.operands[0]]

    @property
    def target_operands(self) -> list[Value]:
        return []

    @property
    def param_operands(self) -> list[Value]:
        return [op for op in self.operands[1:] if op.type.is_classical()]

    @property
    def signature(self) -> Signature:
        raise NotImplementedError("Cannot compute signature for IndexSpecControlledU.")

    def all_input_values(self) -> list[ValueBase]:
        values = super().all_input_values()
        if isinstance(self.num_controls, Value):
            values.append(self.num_controls)
        if self.target_indices:
            values.extend(self.target_indices)
        if self.controlled_indices:
            values.extend(self.controlled_indices)
        return values

    def replace_values(self, mapping: dict[str, ValueBase]) -> Operation:
        result = super().replace_values(mapping)
        assert isinstance(result, IndexSpecControlledU)
        changed = False
        new_nc: int | Value = result.num_controls
        new_ti = result.target_indices
        new_ci = result.controlled_indices
        if (
            isinstance(result.num_controls, Value)
            and result.num_controls.uuid in mapping
        ):
            mapped = mapping[result.num_controls.uuid]
            if isinstance(mapped, Value):
                new_nc = mapped
                changed = True
        if result.target_indices is not None:
            new_ti = [
                typing.cast(Value, mapping.get(v.uuid, v))
                for v in result.target_indices
            ]
            if new_ti != result.target_indices:
                changed = True
        if result.controlled_indices is not None:
            new_ci = [
                typing.cast(Value, mapping.get(v.uuid, v))
                for v in result.controlled_indices
            ]
            if new_ci != result.controlled_indices:
                changed = True
        if changed:
            return dataclasses.replace(
                result,
                num_controls=new_nc,
                target_indices=new_ti,
                controlled_indices=new_ci,
            )
        return result


@dataclasses.dataclass
class MeasureVectorOperation(Operation):
    """Measure a vector of qubits.

    Takes a Vector[Qubit] (ArrayValue) and produces a Vector[Bit] (ArrayValue).
    This operation measures all qubits in the vector as a single operation.

    operands: [ArrayValue of qubits]
    results: [ArrayValue of bits]
    """

    @property
    def signature(self) -> Signature:
        return Signature(
            operands=[ParamHint(name="qubits", type=QubitType())],
            results=[ParamHint(name="bits", type=BitType())],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.HYBRID


@dataclasses.dataclass
class MeasureQFixedOperation(Operation):
    """Measure a quantum fixed-point number.

    This operation measures all qubits in a QFixed register and produces
    a Float result. During transpilation, this is lowered to individual
    MeasureOperations plus a DecodeQFixedOperation.

    Attributes:
        num_bits: Total number of bits in the fixed-point representation.
        int_bits: Number of integer bits (0 for pure fractional like QPE phase).

    operands: [QFixed value (contains qubit_values in params)]
    results: [Float value]

    Encoding:
        For QPE phase (int_bits=0):
            float_value = 0.b0b1b2... = b0*0.5 + b1*0.25 + b2*0.125 + ...
    """

    num_bits: int = 0
    int_bits: int = 0  # For QPE phase, this is 0 (all bits are fractional)

    @property
    def signature(self) -> Signature:
        return Signature(
            operands=[ParamHint(name="qfixed", type=QFixedType())],
            results=[ParamHint(name="float_out", type=FloatType())],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.HYBRID  # Quantum measurement + classical decode
