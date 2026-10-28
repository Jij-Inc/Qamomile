"""Shared utility functions for resource estimation."""

from __future__ import annotations

import sympy as sp

from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOpKind,
    UnaryMathOpKind,
)


def _smart_floordiv(lhs: sp.Expr, r: sp.Expr) -> sp.Expr:
    """Smart FLOORDIV: avoids sp.floor() when quotient is obviously integer.

    Enables cleaner symbolic expressions like ``2**m / 2**i = 2**(m-i)``.

    Args:
        lhs (sp.Expr): Dividend.
        r (sp.Expr): Divisor.

    Returns:
        sp.Expr: ``lhs // r`` as a simplified SymPy expression.
            Uses exact division when the quotient is obviously integral
            (Integer, Symbol, or Pow with non-negative exponent);
            falls back to ``sp.floor(lhs / r)`` otherwise.
    """
    quotient = sp.simplify(lhs / r)
    if isinstance(quotient, (sp.Integer, sp.Symbol)):
        return quotient
    if isinstance(quotient, sp.Pow):
        _, exp = quotient.as_base_exp()
        if exp.is_nonnegative is True:
            return quotient
    return sp.floor(lhs / r)


BINOP_TO_SYMPY = {
    BinOpKind.ADD: lambda lhs, r: lhs + r,
    BinOpKind.SUB: lambda lhs, r: lhs - r,
    BinOpKind.MUL: lambda lhs, r: lhs * r,
    BinOpKind.DIV: lambda lhs, r: lhs / r,
    BinOpKind.FLOORDIV: _smart_floordiv,
    BinOpKind.MOD: lambda lhs, r: sp.Mod(lhs, r),
    BinOpKind.POW: lambda lhs, r: lhs**r,
    BinOpKind.MIN: lambda lhs, r: sp.Min(lhs, r),
}


UNARY_MATH_TO_SYMPY = {
    UnaryMathOpKind.LOG2: lambda value: sp.log(value, 2),
    UnaryMathOpKind.CEIL: sp.ceiling,
}
