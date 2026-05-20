from __future__ import annotations

import abc
import copy
import dataclasses
import typing
import uuid

from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    CompOp,
    CompOpKind,
    CondOp,
    CondOpKind,
    NotOp,
)
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.errors import QubitConsumedError
from qamomile.circuit.transpiler.passes.eval_utils import evaluate_binop_values

if typing.TYPE_CHECKING:
    from .array import ArrayBase
    from .primitives import UInt


def _emit_binop(
    lhs: Value, rhs: Value, result: "Handle | Value", kind: BinOpKind
) -> None:
    """Emit a BinOp to the current tracer, or fold eagerly when both
    operands are compile-time constants.

    When both ``lhs`` and ``rhs`` carry a concrete constant
    (``Value.is_constant()``), the operation is evaluated immediately
    and the result is materialised as a constant ``Value`` rather than
    a BinOp in the trace.  This eliminates trivial residual ops like
    ``lo + 0`` (where ``lo`` is constant) that would otherwise leave
    symbolic shapes in downstream ``ArrayValue.shape`` and break
    stdlib helpers (e.g. :func:`qamomile.circuit.stdlib.qft._get_size`)
    which require concrete sizes at trace time.

    The eager-fold path requires a mutable :class:`Handle` for
    ``result`` so the constant Value can be installed in place; passing
    a raw :class:`Value` keeps the legacy emit-only behaviour for the
    rare callers that don't construct a handle.

    Args:
        lhs: Left operand Value.
        rhs: Right operand Value.
        result: Either the result Handle (whose ``value`` is mutated
            on fold) or a raw Value (emit-only legacy path).
        kind: The :class:`BinOpKind` to apply.
    """
    if isinstance(result, Handle) and lhs.is_constant() and rhs.is_constant():
        lhs_v = lhs.get_const()
        rhs_v = rhs.get_const()
        if lhs_v is not None and rhs_v is not None:
            folded = evaluate_binop_values(kind, lhs_v, rhs_v)
            if folded is not None:
                result.value = result.value.with_const(folded)
                if hasattr(result, "init_value"):
                    try:
                        result.init_value = type(result.init_value)(folded)
                    except (TypeError, ValueError):
                        result.init_value = folded
                return

    result_value = result.value if isinstance(result, Handle) else result
    binop = BinOp(
        operands=[lhs, rhs],
        results=[result_value],
        kind=kind,
    )
    tracer = get_current_tracer()
    tracer.add_operation(binop)


def _emit_compop(lhs: Value, rhs: Value, result: Value, kind: CompOpKind) -> None:
    """Emit a CompOp to the current tracer."""
    compop = CompOp(
        operands=[lhs, rhs],
        results=[result],
        kind=kind,
    )
    tracer = get_current_tracer()
    tracer.add_operation(compop)


def _emit_condop(lhs: Value, rhs: Value, result: Value, kind: CondOpKind) -> None:
    """Emit a CondOp (logical AND/OR) to the current tracer.

    Used by ``Bit.__and__`` / ``__or__`` to lift Python's ``&`` / ``|``
    on Bit handles into the IR. Distinct from ``BinOp`` because the
    semantics is short-circuit-style boolean logic, not bitwise integer
    arithmetic.

    Args:
        lhs: Left operand Value.
        rhs: Right operand Value.
        result: Result Value (``BitType``) the op writes into.
        kind: ``CondOpKind.AND`` or ``CondOpKind.OR``.

    Returns:
        None.
    """
    condop = CondOp(
        operands=[lhs, rhs],
        results=[result],
        kind=kind,
    )
    tracer = get_current_tracer()
    tracer.add_operation(condop)


def _emit_notop(operand: Value, result: Value) -> None:
    """Emit a NotOp (logical negation) to the current tracer.

    Used by ``Bit.__invert__`` to lift Python's ``~`` on a Bit handle
    into the IR.

    Args:
        operand: Operand Value to negate.
        result: Result Value (``BitType``) the op writes into.

    Returns:
        None.
    """
    notop = NotOp(operands=[operand], results=[result])
    tracer = get_current_tracer()
    tracer.add_operation(notop)


@dataclasses.dataclass
class Handle(abc.ABC):
    value: Value
    parent: "ArrayBase | None" = None
    indices: tuple["UInt", ...] = ()
    name: str | None = None
    id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    _consumed: bool = False
    _consumed_by: str | None = None

    def _should_enforce_linear(self) -> bool:
        """Check if this handle type requires linear enforcement.

        Only quantum types (Qubit) require affine type enforcement.
        Classical values (Float, UInt, Bit) can be used multiple times.
        """
        return self.value.type.is_quantum()

    def consume(self, operation_name: str = "unknown") -> typing.Self:
        """Mark this handle as consumed and return a fresh handle.

        Args:
            operation_name: Name of the operation consuming this handle,
                           used for error messages.

        Returns:
            A new handle pointing to the same underlying value.

        Raises:
            QubitConsumedError: If this handle was already consumed
                               and is a quantum type.
        """
        if self._consumed and self._should_enforce_linear():
            display_name = self.name or f"qubit_{self.id[:8]}"
            raise QubitConsumedError(
                f"Qubit '{display_name}' was already consumed by '{self._consumed_by}' "
                f"and cannot be used again in '{operation_name}'.\n\n"
                f"Affine type rule: Each qubit handle can only be used once. "
                f"After a gate operation, reassign the result to use the new handle.\n\n"
                f"Fix:\n"
                f"  q = qm.h(q)  # Reassign to capture the new handle\n"
                f"  q = qm.x(q)  # Use the reassigned handle",
                handle_name=display_name,
                operation_name=operation_name,
                first_use_location=self._consumed_by,
            )
        self._consumed = True
        self._consumed_by = operation_name

        # Use type(self) to preserve the actual subclass type (Qubit, UInt, etc.)
        cls = type(self)
        new_handle = object.__new__(cls)
        new_handle.value = self.value
        new_handle.parent = self.parent
        new_handle.indices = copy.copy(self.indices)
        new_handle.name = self.name
        new_handle.id = self.id
        new_handle._consumed = False
        new_handle._consumed_by = None
        self._copy_subclass_state_to(new_handle)
        return new_handle

    def _copy_subclass_state_to(self, new_handle: "Handle") -> None:
        """Hook for subclasses to copy additional state during consume()."""
        pass


class ArithmeticMixin:
    """Mixin providing arithmetic operations for numeric Handle types.

    Requires:
        - value: Value attribute
        - _make_result(): Method to create result Handle of same type
        - _coerce(): Method to convert Python literals to Handle
    """

    value: Value  # Declare for type checking

    def _coerce(self, other) -> "Handle":
        """Convert int/float to Handle if needed (to be implemented in subclass)."""
        raise NotImplementedError("_coerce must be implemented in subclass.")

    def _make_result(self) -> "Handle":
        """Create a result Handle for an operation (to be implemented in subclass)."""
        raise NotImplementedError("_make_result must be implemented in subclass.")

    def _make_float_result(self) -> "Handle":
        """Create a Float result for division operations (to be implemented in subclass)."""
        raise NotImplementedError("_make_float_result must be implemented in subclass.")

    def __add__(self, other):
        other = self._coerce(other)
        result = self._make_result()
        _emit_binop(self.value, other.value, result, BinOpKind.ADD)
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self._coerce(other)
        result = self._make_result()
        _emit_binop(self.value, other.value, result, BinOpKind.SUB)
        return result

    def __rsub__(self, other):
        other = self._coerce(other)
        result = self._make_result()
        _emit_binop(other.value, self.value, result, BinOpKind.SUB)
        return result

    def __mul__(self, other):
        other = self._coerce(other)
        result = self._make_result()
        _emit_binop(self.value, other.value, result, BinOpKind.MUL)
        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = self._coerce(other)
        result = self._make_float_result()  # Division always returns Float
        _emit_binop(self.value, other.value, result, BinOpKind.DIV)
        return result

    def __rtruediv__(self, other):
        other = self._coerce(other)
        result = self._make_float_result()
        _emit_binop(other.value, self.value, result, BinOpKind.DIV)
        return result
