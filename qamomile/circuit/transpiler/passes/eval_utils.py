"""Shared arithmetic evaluation utilities for transpiler passes."""

from __future__ import annotations

from qamomile.circuit.ir.operation.arithmetic_operations import BinOpKind


def evaluate_binop_values(
    kind: BinOpKind | None,
    left: float | int,
    right: float | int,
) -> float | int | None:
    """Evaluate a binary arithmetic operation on two concrete values.

    Returns the result, or ``None`` on division by zero, unknown kind,
    or arithmetic error.
    """
    if kind is None:
        return None
    try:
        match kind:
            case BinOpKind.ADD:
                return left + right
            case BinOpKind.SUB:
                return left - right
            case BinOpKind.MUL:
                return left * right
            case BinOpKind.DIV:
                return left / right if right != 0 else None
            case BinOpKind.FLOORDIV:
                return left // right if right != 0 else None
            case BinOpKind.POW:
                return left**right
            case _:
                return None
    except (TypeError, ValueError, OverflowError):
        return None
