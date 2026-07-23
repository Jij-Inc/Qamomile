"""Build abstract unary mathematical expressions for qkernel tracing."""

from __future__ import annotations

import math

from qamomile.circuit.frontend.handle.primitives import Float, UInt
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.arithmetic_operations import (
    UnaryMathOp,
    UnaryMathOpKind,
)
from qamomile.circuit.ir.types.primitives import FloatType, UIntType
from qamomile.circuit.ir.value import Value


def _coerce_numeric(value: UInt | Float | int | float) -> UInt | Float:
    """Coerce a Python numeric literal to a classical handle.

    Args:
        value (UInt | Float | int | float): Numeric input to normalize.

    Returns:
        UInt | Float: Existing handle or a constant handle for the literal.

    Raises:
        TypeError: If ``value`` is a boolean or unsupported numeric type.
        ValueError: If an integer literal is negative.
    """
    if isinstance(value, (UInt, Float)):
        return value
    if isinstance(value, bool):
        raise TypeError("Boolean values are not valid mathematical inputs.")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("Unsigned mathematical input must be non-negative.")
        return UInt(
            value=Value(type=UIntType(), name="math_uint").with_const(value),
            init_value=value,
        )
    if isinstance(value, float):
        return Float(
            value=Value(type=FloatType(), name="math_float").with_const(value),
            init_value=value,
        )
    raise TypeError(
        "Mathematical inputs must be UInt, Float, int, or float; "
        f"got {type(value).__name__}."
    )


def log2(value: UInt | Float | int | float) -> Float:
    """Compute a base-two logarithm as an abstract real expression.

    Concrete values fold immediately. Unresolved qkernel values emit one
    abstract ``LOG2`` operation so structural expressions such as
    ``ceil(log2(n))`` remain visible until compile-time bindings are applied.

    Args:
        value (UInt | Float | int | float): Strictly positive input.

    Returns:
        Float: Base-two logarithm of ``value``.

    Raises:
        TypeError: If ``value`` is not a supported numeric input.
        ValueError: If a concrete input is non-finite or not strictly positive.
    """
    operand = _coerce_numeric(value)
    if operand.value.is_constant():
        concrete = operand.value.get_const()
        assert concrete is not None
        is_non_finite_float = isinstance(concrete, float) and not math.isfinite(
            concrete
        )
        if is_non_finite_float or concrete <= 0:
            raise ValueError("log2 input must be finite and strictly positive.")
        result = math.log2(concrete)
        return Float(
            value=Value(type=FloatType(), name="log2").with_const(result),
            init_value=result,
        )

    result = Float(value=Value(type=FloatType(), name="log2"))
    get_current_tracer().add_operation(
        UnaryMathOp(
            operands=[operand.value],
            results=[result.value],
            kind=UnaryMathOpKind.LOG2,
        )
    )
    return result


def ceil(value: Float | UInt | int | float) -> UInt:
    """Round a non-negative numeric expression upward to an integer.

    ``UInt`` is Qamomile's structural integer type, so negative results are
    outside this operation's domain. Concrete values fold immediately;
    unresolved values emit one abstract ``CEIL`` operation.

    Args:
        value (Float | UInt | int | float): Non-negative numeric expression.

    Returns:
        UInt: Least integer greater than or equal to ``value``.

    Raises:
        TypeError: If ``value`` is not a supported numeric input.
        ValueError: If a concrete input is non-finite or rounds to a negative
            integer.
    """
    operand = _coerce_numeric(value)
    if isinstance(operand, UInt):
        return operand
    if operand.value.is_constant():
        concrete = operand.value.get_const()
        assert concrete is not None
        if isinstance(concrete, float) and not math.isfinite(concrete):
            raise ValueError("ceil input must be finite.")
        result = math.ceil(concrete)
        if result < 0:
            raise ValueError("ceil result must be non-negative for UInt.")
        return UInt(
            value=Value(type=UIntType(), name="ceil").with_const(result),
            init_value=result,
        )

    result = UInt(value=Value(type=UIntType(), name="ceil"))
    get_current_tracer().add_operation(
        UnaryMathOp(
            operands=[operand.value],
            results=[result.value],
            kind=UnaryMathOpKind.CEIL,
        )
    )
    return result
