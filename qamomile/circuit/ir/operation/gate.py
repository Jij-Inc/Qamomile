from __future__ import annotations

import dataclasses
import enum
import typing

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import CallableRef
from qamomile.circuit.ir.types import QFixedType
from qamomile.circuit.ir.types.primitives import (
    BitType,
    FloatType,
    QubitType,
    UIntType,
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

    Two concrete subclasses handle distinct operand layouts:

    - ``ConcreteControlledU``: Fixed ``num_controls: int``, individual qubit
      operands.
    - ``SymbolicControlledU``: Symbolic ``num_controls: Value``, vector-based
      control operands; optional ``control_indices`` selects a subset of
      the control vector to act as controls (the rest pass through).

    All ``isinstance(op, ControlledUOperation)`` checks match every subclass.

    Attributes:
        power: Number of times to apply U (default 1). For QPE this is
            ``2**k``. Symbolic ``Value`` is resolved during constant folding.
        block: The unitary ``Block`` to apply conditionally.
        num_controls: Number of control qubits. The base declaration exists
            only so generic code typed against ``ControlledUOperation`` can
            access ``op.num_controls`` and have a typed contract to read from
            (``int | Value``). The default ``1`` is a dataclass slot
            reservation — ``ControlledUOperation`` is never instantiated
            directly (concrete subclasses are the only producers; see the
            pattern-match dispatch in the symbolic ``ResourceEstimator``).
            Every concrete subclass redeclares ``num_controls`` with the
            correct narrow type and the default it actually wants
            (``ConcreteControlledU``: ``int = 1`` matches the single-control
            shape; ``SymbolicControlledU``: a ``UIntType`` ``Value`` placeholder
            via ``default_factory``).
        callable_ref: Stable identity of the controlled callable.
        callable_attrs: Serializer-friendly attrs copied from the controlled
            callable definition.
    """

    power: int | Value = 1
    block: Block | None = None
    num_controls: int | Value = 1
    callable_ref: CallableRef | None = None
    callable_attrs: dict[str, typing.Any] = dataclasses.field(default_factory=dict)

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

    Operand layout: ``[ctrl_arg_0, ..., ctrl_arg_{k-1}, tgt_0, ..., tgt_m, params...]``
    Result layout:  ``[ctrl_arg_0', ..., ctrl_arg_{k-1}', tgt_0', ..., tgt_m']``

    The number of control arguments ``k`` is recorded in
    ``num_control_args``; the default ``k = 1`` corresponds to the
    historical single-pool form (``operands[0]`` is a
    ``Vector[Qubit]`` / ``VectorView`` whose length equals
    ``num_controls``, or whose ``control_indices``-selected subset
    does).  When ``k > 1`` the control prefix is a heterogeneous
    sequence of scalar ``Qubit`` values and ``ArrayValue``s whose
    *total* qubit count is ``num_controls``; the emit pass walks them
    in order to recover the per-physical-qubit control set.

    When ``control_indices`` is ``None`` the entire control prefix
    is used as active controls (one-arg form: ``len(ctrl_vector) ==
    num_controls``; multi-arg form: the qubit-count sum of the
    prefix args equals ``num_controls``).  When non-``None``, the
    listed indices select exactly ``num_controls`` slots from a
    single-arg pool to act as controls; combining
    ``control_indices`` with the multi-arg control prefix is
    rejected at frontend time.

    Each ``control_indices`` entry is stored as a ``Value`` of
    ``UIntType`` regardless of whether the frontend passed an
    ``int`` literal or a ``UInt`` handle, so all downstream
    value-substitution passes see a uniform shape.
    """

    # The default exists only so the dataclass field ordering works for
    # subclasses; every production call site passes ``num_controls=`` and
    # the IR contract is for it to be a ``UIntType`` ``Value``.  The
    # default uses ``name=""`` (the anonymous-marker convention from
    # ``Value`` -- see :class:`qamomile.circuit.ir.value.Value`'s docstring)
    # so two short-form ``SymbolicControlledU(...)`` constructions never
    # collide in the resolver's name-keyed lookup branch; ``type=UIntType()``
    # keeps the type tag honest for any code that switches on
    # ``num_controls.type`` (e.g. ``_fold_value_list`` materialises the
    # folded constant with the *original* type, so a wrong-type default
    # would propagate downstream as a ``FloatType`` UInt).
    num_controls: Value = dataclasses.field(
        default_factory=lambda: Value(type=UIntType(), name="")
    )  # type: ignore[assignment]
    control_indices: tuple[Value, ...] | None = None
    # Number of operand slots that hold the control prefix.  Default
    # ``1`` preserves the single-pool layout used by serialised v1
    # payloads and by every call site that pre-dates the
    # multi-control-arg extension.
    num_control_args: int = 1

    @property
    def is_symbolic_num_controls(self) -> bool:
        return True

    @property
    def control_operands(self) -> list[Value]:
        return list(self.operands[: self.num_control_args])

    @property
    def target_operands(self) -> list[Value]:
        return list(self.operands[self.num_control_args :])

    @property
    def param_operands(self) -> list[Value]:
        return [
            op
            for op in self.operands[self.num_control_args :]
            if op.type.is_classical()
        ]

    @property
    def signature(self) -> Signature:
        raise NotImplementedError("Cannot compute signature for SymbolicControlledU.")

    def all_input_values(self) -> list[ValueBase]:
        values = super().all_input_values()
        values.append(self.num_controls)
        if self.control_indices is not None:
            values.extend(self.control_indices)
        return values

    def replace_values(self, mapping: dict[str, ValueBase]) -> Operation:
        result = super().replace_values(mapping)
        assert isinstance(result, SymbolicControlledU)
        changed = False
        new_nc = result.num_controls
        new_ci = result.control_indices
        if result.num_controls.uuid in mapping:
            mapped = mapping[result.num_controls.uuid]
            if isinstance(mapped, Value):
                new_nc = mapped
                changed = True
        if result.control_indices is not None:
            substituted = tuple(
                typing.cast(Value, mapping.get(v.uuid, v))
                for v in result.control_indices
            )
            if substituted != result.control_indices:
                new_ci = substituted
                changed = True
        if changed:
            return dataclasses.replace(
                result, num_controls=new_nc, control_indices=new_ci
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
