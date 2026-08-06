"""Normalize and substitute symbolic resource scalars."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any, cast

import sympy as sp
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._constants import _ZERO
from qamomile.circuit.estimator._resource_base import (
    _SYMPY_SIMPLIFICATION_ERRORS,
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_conditions import _boolean_condition
from qamomile.circuit.estimator._resource_sums import (
    _evaluate_constant_piecewise_sum,
    _has_large_concrete_sum,
    _large_sum_expression_is_structurally_negative,
    _large_sum_expression_is_structurally_nonnegative,
)


def _expr(value: ResourceExpr | int | float) -> ResourceExpr:
    """Convert a Python scalar to a SymPy expression.

    Args:
        value (ResourceExpr | int | float): Value to convert.

    Returns:
        ResourceExpr: SymPy expression.
    """
    if isinstance(value, sp.Basic):
        return cast(ResourceExpr, value)
    if isinstance(value, int):
        return sp.Integer(value)
    return cast(ResourceExpr, sp.Float(value))


def _substitute_resource_expr(
    expression: sp.Expr,
    substitutions: Mapping[sp.Symbol, sp.Expr],
) -> sp.Expr:
    """Substitute a resource expression and enforce its concrete lower bound.

    Symbolic simplification and later substitution are separate phases, so a
    boundary substitution can expose a negative concrete expression even when
    the symbolic estimate was retained in an unspecialized form. Resource
    metrics cannot be negative, so concrete negative results are clamped when
    their sign can be resolved safely.  A retained huge finite Sum is never
    evaluated merely to discover its sign; an explicitly negative wrapper
    that cannot be proved is instead kept behind an unevaluated ``Max(0, ...)``
    clamp.

    Args:
        expression (sp.Expr): Resource expression to rewrite.
        substitutions (Mapping[sp.Symbol, sp.Expr]): Simultaneous symbol
            replacements.

    Returns:
        sp.Expr: Substituted expression, or zero for a concrete negative
            resource count.
    """
    substituted = cast(
        sp.Expr,
        _substitute_basic_lazily(_expr(expression), substitutions),
    )
    if _has_large_concrete_sum(substituted):
        # A retained finite Sum is already an exact expression. Asking SymPy
        # to ``doit`` it can enter Euler--Maclaurin evaluation whose cost grows
        # with an arbitrarily large user-supplied loop bound.
        if _large_sum_expression_is_structurally_negative(substituted):
            return _ZERO
        if _large_sum_expression_is_structurally_nonnegative(substituted):
            return substituted
        return cast(
            sp.Expr,
            sp.Max(_ZERO, substituted, evaluate=False),
        )
    resolved = cast(sp.Expr, substituted.doit())
    if not resolved.free_symbols <= substituted.free_symbols:
        resolved = substituted
    if resolved.is_number and resolved.is_negative is True:
        return _ZERO
    return resolved


def _substitute_basic_lazily(
    expression: sp.Basic,
    substitutions: Mapping[sp.Symbol, sp.Expr],
) -> sp.Basic:
    """Apply simultaneous substitutions without visiting inactive branches.

    SymPy's ordinary substitution eagerly rebuilds every ``Piecewise`` value
    before deciding its conditions. A concrete resource input can therefore
    evaluate a large ``Sum`` in a branch that is immediately discarded. This
    walker resolves conditions first and stops at the first definitely active
    branch while preserving exact symbol-identity replacement everywhere
    else.

    Args:
        expression (sp.Basic): Symbolic expression or predicate to rewrite.
        substitutions (Mapping[sp.Symbol, sp.Expr]): Simultaneous free-symbol
            replacements.

    Returns:
        sp.Basic: Rewritten expression with unreachable branches untouched.
    """
    if not substitutions or not expression.free_symbols.intersection(substitutions):
        return expression
    replacement = substitutions.get(cast(sp.Symbol, expression))
    if replacement is not None and isinstance(expression, sp.Symbol):
        return replacement
    if isinstance(expression, sp.Piecewise):
        branches: list[tuple[sp.Expr, Boolean | bool]] = []
        for branch in expression.args:
            value, condition = branch.args
            rewritten_condition = _boolean_condition(
                _substitute_basic_lazily(
                    cast(sp.Basic, condition),
                    substitutions,
                )
            )
            if rewritten_condition is sp.false:
                continue
            rewritten_value = cast(
                sp.Expr,
                _substitute_basic_lazily(
                    cast(sp.Basic, value),
                    substitutions,
                ),
            )
            if rewritten_condition is sp.true:
                if not branches:
                    return rewritten_value
                branches.append((rewritten_value, True))
                break
            branches.append((rewritten_value, rewritten_condition))
        if not branches:
            return sp.nan
        return sp.Piecewise(*branches)
    rewritten_args = tuple(
        _substitute_basic_lazily(cast(sp.Basic, argument), substitutions)
        for argument in expression.args
    )
    if rewritten_args == expression.args:
        return expression
    if isinstance(expression, sp.Sum):
        rewritten_sum = cast(sp.Sum, expression.func(*rewritten_args))
        evaluated = _evaluate_constant_piecewise_sum(rewritten_sum)
        return rewritten_sum if evaluated is None else evaluated
    if isinstance(
        expression,
        (
            sp.Max,
            sp.Min,
            Relational,
            sp.floor,
            sp.ceiling,
            sp.Mod,
            sp.Abs,
            sp.Pow,
            sp.Add,
            sp.Mul,
        ),
    ) and any(_has_large_concrete_sum(argument) for argument in rewritten_args):
        # Max/Min try to compare their arguments during construction. For an
        # unevaluated huge finite Sum that comparison invokes numerical
        # summation even though retaining the exact symbolic node is enough.
        constructor = cast(Any, expression.func)
        return cast(sp.Basic, constructor(*rewritten_args, evaluate=False))
    return cast(sp.Basic, expression.func(*rewritten_args))


@lru_cache(maxsize=4096)
def _safe_simplify(expression: ResourceExpr) -> ResourceExpr:
    """Simplify an expression without releasing internal bound symbols.

    SymPy can incorrectly move a bound ``Sum`` index into a Piecewise
    condition while simplifying some nested loop expressions. Any newly free
    symbol would be exposed as a fake qkernel parameter, so such a rewrite is
    rejected in favor of the original equivalent expression.

    Args:
        expression (ResourceExpr): Resource expression to simplify.

    Returns:
        ResourceExpr: Simplified expression when symbol provenance is
        preserved, otherwise the original expression when SymPy cannot
        simplify it safely.
    """
    normalized = _expr(expression)
    if _has_large_concrete_sum(normalized):
        return normalized
    try:
        simplified = cast(ResourceExpr, sp.simplify(normalized))
    except _SYMPY_SIMPLIFICATION_ERRORS:
        return normalized
    if simplified.free_symbols <= normalized.free_symbols:
        return simplified
    return normalized


def _safe_constraint_substitute(
    expression: ResourceExpr,
    substitutions: Mapping[sp.Symbol, sp.Expr],
) -> ResourceExpr:
    """Substitute constraint variables without clamping invalid values.

    Args:
        expression (ResourceExpr): Constraint or range expression.
        substitutions (Mapping[sp.Symbol, sp.Expr]): Bound loop values.

    Returns:
        ResourceExpr: Safely evaluated substituted expression.
    """
    substituted = cast(
        ResourceExpr,
        _substitute_basic_lazily(_expr(expression), substitutions),
    )
    if _has_large_concrete_sum(substituted):
        return substituted
    evaluated = cast(ResourceExpr, substituted.doit())
    if not evaluated.free_symbols <= substituted.free_symbols:
        return substituted
    return _safe_simplify(evaluated)


def _resource_expr(value: sp.Basic) -> ResourceExpr:
    """Narrow a SymPy scalar expression to the resource expression type.

    SymPy annotates relational ``Piecewise`` results as ``Basic`` even though
    they participate in the same scalar arithmetic as ``Expr`` throughout the
    estimator.

    Args:
        value (sp.Basic): SymPy scalar expression to narrow.

    Returns:
        ResourceExpr: Expression accepted by resource result records.

    Raises:
        TypeError: If ``value`` is not a scalar SymPy expression.
    """
    if not isinstance(value, sp.Expr):
        raise TypeError(
            f"Expected a scalar SymPy expression, got {type(value).__name__}"
        )
    return value


def _add_maps(
    left: Mapping[str, ResourceExpr],
    right: Mapping[str, ResourceExpr],
) -> dict[str, ResourceExpr]:
    """Add two expression dictionaries key-wise.

    Args:
        left (Mapping[str, ResourceExpr]): Left mapping.
        right (Mapping[str, ResourceExpr]): Right mapping.

    Returns:
        dict[str, ResourceExpr]: Merged mapping.
    """
    merged = dict(left)
    for name, value in right.items():
        merged[name] = merged.get(name, _ZERO) + value
    return merged
