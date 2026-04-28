"""Shared evaluation utilities for transpiler passes.

This module is the **single source of truth** for kind-dispatched
classical-op evaluation. Three transpiler passes need to evaluate the
same family of ops (``CompOp``, ``CondOp``, ``NotOp``, ``BinOp``):

- ``compile_time_if_lowering`` — folds ops at compile time when operands
  resolve to constants.
- ``classical_executor`` — evaluates ops at runtime over a results dict.
- ``cast_binop_emission`` (emit pass) — folds ops at emit time for
  inlining loop variables and parameter bindings into ``if`` conditions.

Each pass differs in how it *resolves operands* (concrete_values map,
results dict, bindings dict + resolver). They all share the same
*kind → Python operation* dispatch. By centralizing the dispatch here,
adding a new ``CompOpKind`` requires changing only this file.

The functions below take fully resolved Python operands and return the
computed value (or ``None`` if the op cannot be evaluated, e.g. division
by zero, unknown kind, type mismatch). They never touch bindings or
resolvers — those responsibilities stay with the caller.
"""

from __future__ import annotations

from typing import Any

from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOpKind,
    CompOpKind,
    CondOpKind,
)


def evaluate_binop_values(
    kind: BinOpKind | None,
    left: float | int,
    right: float | int,
) -> float | int | None:
    """Evaluate a binary arithmetic operation on two concrete values.

    Args:
        kind: The ``BinOpKind`` to apply.
        left: Left operand (numeric).
        right: Right operand (numeric).

    Returns:
        The result, or ``None`` on division by zero, unknown kind, or
        arithmetic error.
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


def evaluate_compop_values(
    kind: CompOpKind | None,
    left: Any,
    right: Any,
) -> bool | None:
    """Evaluate a comparison operation on two concrete values.

    Args:
        kind: The ``CompOpKind`` to apply.
        left: Left operand.
        right: Right operand.

    Returns:
        The boolean result, or ``None`` on unknown kind or type
        incompatibility.
    """
    if kind is None:
        return None
    try:
        match kind:
            case CompOpKind.EQ:
                return left == right
            case CompOpKind.NEQ:
                return left != right
            case CompOpKind.LT:
                return left < right
            case CompOpKind.LE:
                return left <= right
            case CompOpKind.GT:
                return left > right
            case CompOpKind.GE:
                return left >= right
            case _:
                return None
    except (TypeError, ValueError):
        return None


def evaluate_condop_values(
    kind: CondOpKind | None,
    left: Any,
    right: Any,
) -> bool | None:
    """Evaluate a logical and/or operation on two concrete values.

    Args:
        kind: The ``CondOpKind`` to apply.
        left: Left operand.
        right: Right operand.

    Returns:
        The boolean result, or ``None`` on unknown kind.
    """
    if kind is None:
        return None
    try:
        match kind:
            case CondOpKind.AND:
                return bool(left and right)
            case CondOpKind.OR:
                return bool(left or right)
            case _:
                return None
    except (TypeError, ValueError):
        return None


def evaluate_notop_value(operand: Any) -> bool | None:
    """Evaluate a logical-not operation on a concrete value.

    Args:
        operand: The operand to negate.

    Returns:
        The boolean negation, or ``None`` on type error.
    """
    try:
        return not bool(operand)
    except (TypeError, ValueError):
        return None
