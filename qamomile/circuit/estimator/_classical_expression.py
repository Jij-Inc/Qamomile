"""Apply scalar IR operations to exact symbolic expressions."""

from __future__ import annotations

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOpKind,
    CompOpKind,
    CondOpKind,
    UnaryMathOpKind,
)
from qamomile.circuit.ir.types.primitives import BitType, FloatType, UIntType
from qamomile.circuit.ir.value import Value

from ._utils import BINOP_TO_SYMPY, UNARY_MATH_TO_SYMPY


def _parameter_symbol(value: Value, name: str) -> sp.Symbol:
    """Create a symbol matching an IR parameter's scalar domain.

    Args:
        value (Value): Parameter value whose IR type defines assumptions.
        name (str): Public parameter name used for the symbol.

    Returns:
        sp.Symbol: A nonnegative integer for UInt/Bit, a real symbol for
            Float, or an unconstrained symbol for other value types.
    """
    if isinstance(value.type, FloatType):
        return sp.Symbol(name, real=True)
    if isinstance(value.type, (BitType, UIntType)):
        # Zero is a valid UInt/Bit value. Assuming strict positivity lets SymPy
        # erase ``value == 0`` branches and zero-trip width guards before a
        # later substitution can recover them.
        return sp.Symbol(name, integer=True, nonnegative=True)
    return sp.Symbol(name)


def _fallback_symbol(value: Value) -> sp.Symbol:
    """Create the identity-qualified symbol for one unresolved IR value.

    Args:
        value (Value): Unresolved scalar IR value.

    Returns:
        sp.Symbol: Typed private symbol whose spelling includes the value UUID.
    """
    fallback_name = f"{value.name}_{value.uuid}"
    if isinstance(value.type, FloatType):
        return sp.Symbol(fallback_name, real=True)
    if isinstance(value.type, (BitType, UIntType)):
        return sp.Symbol(fallback_name, integer=True, nonnegative=True)
    return sp.Symbol(fallback_name)


_COMPOP_MAP = {
    CompOpKind.EQ: sp.Eq,
    CompOpKind.NEQ: sp.Ne,
    CompOpKind.LT: sp.Lt,
    CompOpKind.LE: sp.Le,
    CompOpKind.GT: sp.Gt,
    CompOpKind.GE: sp.Ge,
}


def _as_boolean(expression: sp.Basic) -> Boolean:
    """Convert a numeric or predicate expression to logical truthiness.

    Qamomile predicates follow Python scalar truthiness for compile-time
    numeric values. Symbolically, that means a non-Boolean expression is true
    exactly when it is nonzero.

    Args:
        expression (sp.Basic): Numeric or Boolean SymPy expression.

    Returns:
        Boolean: Boolean expression with the same truthiness.
    """
    if isinstance(expression, Boolean):
        return expression
    return sp.Ne(expression, 0)


def _apply_binop(kind: BinOpKind, left: sp.Expr, right: sp.Expr) -> sp.Expr:
    """Apply binary arithmetic.

    Args:
        kind (BinOpKind): The arithmetic operation kind.
        left (sp.Expr): Left operand.
        right (sp.Expr): Right operand.

    Returns:
        sp.Expr: Result of applying the operation.

    Raises:
        ValueError: If *kind* is not in ``BINOP_TO_SYMPY``.
    """
    fn = BINOP_TO_SYMPY.get(kind)
    if fn is None:
        raise ValueError(f"Unknown BinOpKind: {kind}")
    return fn(left, right)


def _apply_unary_math(
    kind: UnaryMathOpKind,
    operand: sp.Expr,
) -> sp.Expr:
    """Apply one exact symbolic unary mathematical operation.

    Args:
        kind (UnaryMathOpKind): Mathematical operation kind.
        operand (sp.Expr): Symbolic numeric operand.

    Returns:
        sp.Expr: Exact SymPy expression.

    Raises:
        ValueError: If ``kind`` has no symbolic implementation.
    """
    fn = UNARY_MATH_TO_SYMPY.get(kind)
    if fn is None:
        raise ValueError(f"Unknown UnaryMathOpKind: {kind}")
    return fn(operand)


def _apply_compop(kind: CompOpKind, left: sp.Expr, right: sp.Expr) -> sp.Expr:
    """Apply comparison operation.

    Args:
        kind (CompOpKind): The comparison operation kind.
        left (sp.Expr): Left operand.
        right (sp.Expr): Right operand.

    Returns:
        sp.Expr: SymPy relational expression (e.g. ``sp.Eq``, ``sp.Lt``).

    Raises:
        ValueError: If *kind* is not in ``_COMPOP_MAP``.
    """
    fn = _COMPOP_MAP.get(kind)
    if fn is None:
        raise ValueError(f"Unknown CompOpKind: {kind}")
    return fn(left, right)  # type: ignore[return-value]


def _apply_condop(
    kind: CondOpKind,
    left: sp.Basic,
    right: sp.Basic,
) -> Boolean:
    """Apply a symbolic logical AND or OR operation.

    Args:
        kind (CondOpKind): Logical operation kind.
        left (sp.Basic): Left numeric or Boolean operand.
        right (sp.Basic): Right numeric or Boolean operand.

    Returns:
        Boolean: SymPy Boolean expression.

    Raises:
        ValueError: If ``kind`` has no symbolic implementation.
    """
    match kind:
        case CondOpKind.AND:
            return sp.And(_as_boolean(left), _as_boolean(right))
        case CondOpKind.OR:
            return sp.Or(_as_boolean(left), _as_boolean(right))
        case _:
            raise ValueError(f"Unknown CondOpKind: {kind}")
