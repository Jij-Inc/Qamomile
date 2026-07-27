"""Shared evaluation utilities for transpiler passes.

This module is the **single source of truth** for kind-dispatched
classical-op evaluation. Three transpiler passes need to evaluate the
same family of ops (``CompOp``, ``CondOp``, ``NotOp``, ``BinOp``):

- ``compile_time_if_lowering`` — folds ops at compile time when operands
  resolve to constants.
- ``classical_executor`` — evaluates ops at runtime over a results dict.
- ``cast_binop_emission`` (emit pass) — folds ops at emit time for
  inlining loop variables and parameter bindings into ``if`` conditions.

Compile-time constant folding additionally uses the same interface for
``UnaryMathOp``.

The low-level ``evaluate_*_values`` functions take fully resolved Python
operands and return the computed value (or ``None`` if the op cannot be
evaluated, e.g. division by zero, unknown kind, type mismatch). They
never touch bindings or resolvers — those responsibilities stay with the
caller.

The high-level ``fold_classical_op`` adds a uniform parameter-respecting
policy on top: callers supply a resolver callable and a ``FoldPolicy``,
and the function returns either the folded scalar or ``None``. This is
the layer that prevents the silent miscompilation class of bugs (Issue
#354 B-series): the parameter-array-element guard is now structurally
inside this function rather than re-implemented per call site.
"""

from __future__ import annotations

import enum
import math
from typing import Any, Callable

from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    CompOp,
    CompOpKind,
    CondOp,
    CondOpKind,
    NotOp,
    RuntimeOpKind,
    UnaryMathOp,
    UnaryMathOpKind,
)


class FoldPolicy(enum.Enum):
    """Policy controlling how ``fold_classical_op`` treats parameters.

    ``COMPILE_TIME`` is for passes that have no notion of runtime
    parameters (e.g. ``compile_time_if_lowering``); every operand the
    resolver returns is treated as a real value to fold.

    ``EMIT_RESPECT_PARAMS`` is for emit-time passes where some Values
    may correspond to runtime backend parameters whose concrete values
    are placeholders or absent. Operands whose Value or
    ``parent_array.name`` is in the active ``parameters`` set are
    treated as symbolic and the fold returns ``None`` rather than
    producing an incorrect concrete result.
    """

    COMPILE_TIME = "compile_time"
    EMIT_RESPECT_PARAMS = "emit_respect_params"


def _is_runtime_parameter_operand(operand: Any, parameters: set[str]) -> bool:
    """Return True when ``operand`` traces to a runtime parameter.

    Two cases are recognised:
    1. ``operand`` is an array element (``arr[idx]``) and ``arr`` is in
       ``parameters``.
    2. ``operand`` itself is a parameter Value whose name is in
       ``parameters``.

    Args:
        operand: The IR operand to inspect (typically a ``Value``).
        parameters: The set of names declared as runtime parameters.

    Returns:
        True if folding this operand would consume a placeholder rather
        than a true compile-time value.
    """
    parent_array = getattr(operand, "parent_array", None)
    if parent_array is not None and getattr(parent_array, "name", None) in parameters:
        return True
    is_parameter = getattr(operand, "is_parameter", None)
    if callable(is_parameter) and is_parameter():
        param_name_fn = getattr(operand, "parameter_name", None)
        if callable(param_name_fn):
            pname = param_name_fn()
            if pname and pname in parameters:
                return True
    return False


def fold_classical_op(
    op: "BinOp | CompOp | CondOp | NotOp | UnaryMathOp",
    operand_resolver: Callable[[Any], Any],
    parameters: set[str],
    policy: FoldPolicy,
) -> Any | None:
    """Fold a classical op to a concrete value, respecting the given policy.

    The caller supplies an ``operand_resolver`` callable that knows how
    to look up a Value in the caller's context (a ``concrete_values``
    map, an emit ``bindings`` dict + ``ValueResolver``, or any other
    source). This function handles:

    1. The parameter guard (skips folding when an operand is a runtime
       parameter or parameter-array element under ``EMIT_RESPECT_PARAMS``).
    2. Strict scalar-only typing (rejects backend ``Expr`` objects and
       other non-numeric values that would spuriously fold to ``True``).
    3. Kind dispatch via the underlying ``evaluate_*_values`` primitives.

    Args:
        op: The classical op to fold. Must be one of ``BinOp``,
            ``CompOp``, ``CondOp``, ``NotOp``, or ``UnaryMathOp``.
        operand_resolver: Callable mapping each operand Value to a
            resolved Python scalar (or ``None`` when unresolvable).
        parameters: Set of runtime parameter names. Used only when
            ``policy`` is ``EMIT_RESPECT_PARAMS``.
        policy: Folding policy. See ``FoldPolicy`` docstring.

    Returns:
        The folded value (numeric for arithmetic and unary math operations,
        bool for predicates), or ``None`` when any operand is symbolic,
        missing, or a runtime parameter under the policy.
    """
    operands = op.operands

    if policy is FoldPolicy.EMIT_RESPECT_PARAMS:
        for opnd in operands:
            if _is_runtime_parameter_operand(opnd, parameters):
                return None

    resolved = [operand_resolver(opnd) for opnd in operands]
    if any(r is None for r in resolved):
        return None

    if not all(isinstance(r, (bool, int, float)) for r in resolved):
        return None

    if isinstance(op, BinOp):
        if len(resolved) != 2:
            return None
        return evaluate_binop_values(op.kind, resolved[0], resolved[1])
    if isinstance(op, CompOp):
        if len(resolved) != 2:
            return None
        return evaluate_compop_values(op.kind, resolved[0], resolved[1])
    if isinstance(op, CondOp):
        if len(resolved) != 2:
            return None
        return evaluate_condop_values(op.kind, resolved[0], resolved[1])
    if isinstance(op, NotOp):
        if len(resolved) != 1:
            return None
        return evaluate_notop_value(resolved[0])
    if isinstance(op, UnaryMathOp):
        if len(resolved) != 1:
            return None
        return evaluate_unary_math_value(op.kind, resolved[0])

    # All op kinds in the signature are exhausted above; this guard
    # only matters if a caller bypasses the type system at runtime.
    return None  # type: ignore[unreachable]


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
            case BinOpKind.MOD:
                return left % right if right != 0 else None
            case BinOpKind.POW:
                return left**right
            case BinOpKind.MIN:
                return left if left <= right else right
            case _:
                return None  # type: ignore[unreachable]
    except (TypeError, ValueError, OverflowError):
        return None


def evaluate_unary_math_value(
    kind: UnaryMathOpKind | None,
    operand: float | int,
) -> float | int | None:
    """Evaluate one unary mathematical operation on a concrete value.

    Args:
        kind (UnaryMathOpKind | None): Mathematical operation to apply.
        operand (float | int): Concrete numeric operand.

    Returns:
        float | int | None: Computed value, or ``None`` for an invalid domain
            or missing operation kind.
    """
    if kind is None:
        return None
    try:
        if isinstance(operand, float) and not math.isfinite(operand):
            return None
        if kind is UnaryMathOpKind.LOG2:
            return math.log2(operand) if operand > 0 else None
        assert kind is UnaryMathOpKind.CEIL
        result = math.ceil(operand)
        return result if result >= 0 else None
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
                return None  # type: ignore[unreachable]
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
                return None  # type: ignore[unreachable]
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


_RUNTIME_TO_BINOP_KIND = {
    RuntimeOpKind.ADD: BinOpKind.ADD,
    RuntimeOpKind.SUB: BinOpKind.SUB,
    RuntimeOpKind.MUL: BinOpKind.MUL,
    RuntimeOpKind.DIV: BinOpKind.DIV,
    RuntimeOpKind.FLOORDIV: BinOpKind.FLOORDIV,
    RuntimeOpKind.MOD: BinOpKind.MOD,
    RuntimeOpKind.POW: BinOpKind.POW,
}
_RUNTIME_TO_COMPOP_KIND = {
    RuntimeOpKind.EQ: CompOpKind.EQ,
    RuntimeOpKind.NEQ: CompOpKind.NEQ,
    RuntimeOpKind.LT: CompOpKind.LT,
    RuntimeOpKind.LE: CompOpKind.LE,
    RuntimeOpKind.GT: CompOpKind.GT,
    RuntimeOpKind.GE: CompOpKind.GE,
}
_RUNTIME_TO_CONDOP_KIND = {
    RuntimeOpKind.AND: CondOpKind.AND,
    RuntimeOpKind.OR: CondOpKind.OR,
}


def evaluate_runtime_op_values(
    kind: RuntimeOpKind | None,
    operands: list[Any],
) -> Any | None:
    """Evaluate a lowered runtime-classical expression on concrete operands.

    Delegates to the same family evaluators used by ``BinOp``, ``CompOp``,
    ``CondOp``, and ``NotOp`` so lowering an operation changes only when it is
    evaluated, not its scalar semantics.

    Args:
        kind (RuntimeOpKind | None): Unified lowered operation kind.
        operands (list[Any]): Concrete operands in IR order. ``NOT`` requires
            one operand; every other kind requires two.

    Returns:
        Any | None: Evaluated numeric or boolean result, or ``None`` when the
            kind/arity is invalid or the underlying operation fails.
    """
    if kind is RuntimeOpKind.NOT:
        if len(operands) != 1:
            return None
        return evaluate_notop_value(operands[0])
    if kind is None or len(operands) != 2:
        return None

    binop_kind = _RUNTIME_TO_BINOP_KIND.get(kind)
    if binop_kind is not None:
        return evaluate_binop_values(binop_kind, operands[0], operands[1])
    compop_kind = _RUNTIME_TO_COMPOP_KIND.get(kind)
    if compop_kind is not None:
        return evaluate_compop_values(compop_kind, operands[0], operands[1])
    condop_kind = _RUNTIME_TO_CONDOP_KIND.get(kind)
    if condop_kind is not None:
        return evaluate_condop_values(condop_kind, operands[0], operands[1])
    return None
