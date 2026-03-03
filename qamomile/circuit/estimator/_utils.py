"""Shared utility functions for resource estimation."""

from __future__ import annotations

import sympy as sp

from qamomile.circuit.ir.operation.arithmetic_operations import BinOpKind

BINOP_TO_SYMPY = {
    BinOpKind.ADD: lambda l, r: l + r,
    BinOpKind.SUB: lambda l, r: l - r,
    BinOpKind.MUL: lambda l, r: l * r,
    BinOpKind.DIV: lambda l, r: l / r,
    BinOpKind.FLOORDIV: lambda l, r: sp.floor(l / r),
    BinOpKind.POW: lambda l, r: l**r,
}


def _strip_nonneg_max(expr: sp.Expr) -> sp.Expr:
    """Canonicalize Max(0, x) -> x in resource estimation expressions.

    Resource estimates (qubits, gates, depths) are non-negative by physical
    construction, so Max(0, expr) is a redundant artifact introduced by
    sp.Max operations. This normalization aligns all paths to the same
    canonical form.
    """
    if not isinstance(expr, sp.Expr):
        return expr
    # Bottom-up: first process sub-expressions, then this node
    if expr.args:
        new_args = [_strip_nonneg_max(a) for a in expr.args]
        expr = expr.func(*new_args)
    # Match Max(0, x) or Max(x, 0)
    if isinstance(expr, sp.Max) and len(expr.args) == 2:
        a, b = expr.args
        if a == 0 or a == sp.Integer(0):
            return b
        if b == 0 or b == sp.Integer(0):
            return a
    return expr
