"""Prove and maximize structural resource-expression bounds."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import cast

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._constants import _ONE, _ZERO
from qamomile.circuit.estimator._resource_base import ResourceExpr
from qamomile.circuit.estimator._resource_conditions import (
    _boolean_condition,
    _ConditionIndicator,
    _linear_condition_boundaries,
    _piecewise,
    _RangeAtLeastTwo,
    _resource_activity_condition,
)
from qamomile.circuit.estimator._resource_scalars import _safe_simplify
from qamomile.circuit.estimator._resource_sums import _sum_expr


@lru_cache(maxsize=4096)
def _is_structurally_nonnegative(expression: ResourceExpr) -> bool:
    """Prove nonnegativity from resource-expression constructors alone.

    This deliberately avoids general symbolic simplification. Resource width
    expressions frequently contain nested ``Min``, ``Max``, and ``Piecewise``
    nodes; recognizing their explicit nonnegative branches is both cheaper and
    more reliable than asking SymPy to derive the same invariant globally. An
    additive expression with exactly one such node also distributes its affine
    remainder into that node's values, which proves common-offset bounds
    without expanding combinations of independent extrema.

    Args:
        expression (ResourceExpr): Expression whose sign should be inspected.

    Returns:
        bool: Whether the expression structure proves a nonnegative value.
    """
    if expression == _ZERO:
        return True
    if isinstance(expression, sp.Number):
        return expression.is_nonnegative is True
    if isinstance(expression, sp.Symbol):
        return expression.is_nonnegative is True
    if getattr(type(expression), "is_nonnegative", None) is True:
        return True
    if isinstance(expression, sp.Max):
        return any(
            _is_structurally_nonnegative(cast(ResourceExpr, argument))
            for argument in expression.args
        )
    if isinstance(expression, sp.Min):
        return all(
            _is_structurally_nonnegative(cast(ResourceExpr, argument))
            for argument in expression.args
        )
    if isinstance(expression, sp.Piecewise):
        return all(
            _is_structurally_nonnegative(cast(ResourceExpr, pair.args[0]))
            for pair in expression.args
        )
    if isinstance(expression, sp.Add):
        if all(
            _is_structurally_nonnegative(cast(ResourceExpr, argument))
            for argument in expression.args
        ):
            return True
        extrema = tuple(
            argument
            for argument in expression.args
            if isinstance(argument, (sp.Max, sp.Min, sp.Piecewise))
        )
        if len(extrema) != 1:
            return False
        extremum = extrema[0]
        remainder = cast(
            ResourceExpr,
            sp.Add(
                *(argument for argument in expression.args if argument is not extremum)
            ),
        )
        if isinstance(extremum, sp.Max):
            return any(
                _is_structurally_nonnegative(cast(ResourceExpr, argument + remainder))
                for argument in extremum.args
            )
        values = (
            tuple(pair.args[0] for pair in extremum.args)
            if isinstance(extremum, sp.Piecewise)
            else extremum.args
        )
        return all(
            _is_structurally_nonnegative(cast(ResourceExpr, value + remainder))
            for value in values
        )
    if isinstance(expression, sp.Mul):
        return all(
            _is_structurally_nonnegative(cast(ResourceExpr, argument))
            for argument in expression.args
        )
    if isinstance(expression, sp.Pow):
        base, exponent = expression.args
        return _is_structurally_nonnegative(cast(ResourceExpr, base)) or (
            exponent.is_integer is True and exponent.is_even is True
        )
    return False


@lru_cache(maxsize=4096)
def _is_structurally_less_equal(
    left: ResourceExpr,
    right: ResourceExpr,
) -> bool:
    """Prove one resource expression does not exceed another structurally.

    Args:
        left (ResourceExpr): Candidate lower expression.
        right (ResourceExpr): Candidate upper expression.

    Returns:
        bool: Whether every structural branch proves ``left <= right``.
    """
    if left == right:
        return True
    left_coefficient, left_remainder = left.as_coeff_Mul()
    right_coefficient, right_remainder = right.as_coeff_Mul()
    if left_remainder == right_remainder and _is_structurally_nonnegative(
        cast(ResourceExpr, right_coefficient - left_coefficient)
    ):
        return True
    if isinstance(left, _ConditionIndicator):
        return _is_structurally_less_equal(_ONE, right)
    if isinstance(left, sp.Mul) and any(
        isinstance(argument, _ConditionIndicator) for argument in left.args
    ):
        unguarded = cast(
            ResourceExpr,
            sp.Mul(
                *(
                    argument
                    for argument in left.args
                    if not isinstance(argument, _ConditionIndicator)
                )
            ),
        )
        if _is_structurally_nonnegative(unguarded) and _is_structurally_less_equal(
            unguarded, right
        ):
            return True
    if isinstance(left, sp.Piecewise):
        return all(
            _is_structurally_less_equal(
                cast(ResourceExpr, pair.args[0]),
                right,
            )
            for pair in left.args
        )
    if isinstance(right, sp.Piecewise):
        return all(
            _is_structurally_less_equal(
                left,
                cast(ResourceExpr, pair.args[0]),
            )
            for pair in right.args
        )
    if isinstance(left, sp.Max):
        return all(
            _is_structurally_less_equal(cast(ResourceExpr, argument), right)
            for argument in left.args
        )
    if isinstance(left, sp.Min):
        return any(
            _is_structurally_less_equal(cast(ResourceExpr, argument), right)
            for argument in left.args
        )
    if isinstance(right, sp.Max):
        return any(
            _is_structurally_less_equal(left, cast(ResourceExpr, argument))
            for argument in right.args
        )
    if isinstance(right, sp.Min):
        return all(
            _is_structurally_less_equal(left, cast(ResourceExpr, argument))
            for argument in right.args
        )
    return _is_structurally_nonnegative(right - left)


def _guarded_resource_extension(
    base: ResourceExpr,
    candidate: ResourceExpr,
) -> ResourceExpr | None:
    """Recover an exact maximum from one activity-gated extension.

    Scheduling can represent an optional completion as
    ``(base + extension) * indicator(extension > 0)``. Its maximum with the
    nonnegative ``base`` is exactly ``base + extension``: while inactive the
    extension is zero, and while active the guarded candidate dominates.

    Args:
        base (ResourceExpr): Completion retained when the extension is absent.
        candidate (ResourceExpr): Potential activity-gated completion.

    Returns:
        ResourceExpr | None: Unguarded dominating completion when the pattern
            is proven structurally, otherwise ``None``.
    """
    if not isinstance(candidate, sp.Mul) or not _is_structurally_nonnegative(base):
        return None
    indicators = tuple(
        argument
        for argument in candidate.args
        if isinstance(argument, _ConditionIndicator)
    )
    if len(indicators) != 1:
        return None
    unguarded = cast(
        ResourceExpr,
        sp.Mul(
            *(
                argument
                for argument in candidate.args
                if not isinstance(argument, _ConditionIndicator)
            )
        ),
    )
    if not _is_structurally_less_equal(base, unguarded):
        return None
    extension = cast(ResourceExpr, unguarded - base)
    if not _is_structurally_nonnegative(extension):
        return None
    if indicators[0].args[0] != _resource_activity_condition(extension):
        return None
    return unguarded


def _resource_max(left: ResourceExpr, right: ResourceExpr) -> ResourceExpr:
    """Return a compact maximum using structural resource bounds.

    Args:
        left (ResourceExpr): First candidate.
        right (ResourceExpr): Second candidate.

    Returns:
        ResourceExpr: Dominating expression when proven, otherwise ``Max``.
    """
    if (extension := _guarded_resource_extension(left, right)) is not None:
        return extension
    if (extension := _guarded_resource_extension(right, left)) is not None:
        return extension
    if _is_structurally_less_equal(left, right):
        return right
    if _is_structurally_less_equal(right, left):
        return left
    candidates = (
        *(left.args if isinstance(left, sp.Max) else (left,)),
        *(right.args if isinstance(right, sp.Max) else (right,)),
    )
    unique_candidates = tuple(dict.fromkeys(candidates))
    if len(unique_candidates) == 1:
        return cast(ResourceExpr, unique_candidates[0])
    return cast(ResourceExpr, sp.Max(*unique_candidates, evaluate=False))


def _resource_max_many(expressions: Sequence[ResourceExpr]) -> ResourceExpr:
    """Fold resource maxima without triggering eager general simplification.

    The dependency scheduler repeatedly compares expressions already known to
    be nonnegative and monotone. Folding through :func:`_resource_max` proves
    those common dominance relations structurally before building an
    unevaluated ``Max``. This avoids SymPy's general pairwise relation proofs,
    whose cost is quadratic in the number and size of candidates.

    Args:
        expressions (Sequence[ResourceExpr]): Candidate resource expressions.

    Returns:
        ResourceExpr: Structural maximum, or zero for an empty sequence.
    """
    maximum = _ZERO
    for expression in expressions:
        maximum = _resource_max(maximum, expression)
    return maximum


def _maximum_expr_over_range(
    expr: ResourceExpr,
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> tuple[ResourceExpr, Boolean]:
    """Maximize one nonnegative resource expression over a loop range.

    Constant, affine, finite, and provably monotone expressions are reduced
    exactly. When SymPy cannot prove a maximum, summing the nonnegative body
    expression gives a conservative loop-symbol-free upper bound.

    Args:
        expr (ResourceExpr): Per-iteration resource expression.
        loop_symbol (sp.Symbol): Loop variable symbol.
        start (ResourceExpr): First loop value.
        step (ResourceExpr): Loop step.
        iterations (ResourceExpr): Number of executed iterations.

    Returns:
        tuple[ResourceExpr, Boolean]: Maximum expression and the condition
        under which its conservative fallback may overestimate.
    """
    condition = sp.Gt(iterations, _ZERO)
    if loop_symbol not in expr.free_symbols:
        return _piecewise(expr, _ZERO, condition), sp.false

    if expr.has(sp.Piecewise) and isinstance(expr, (sp.Min, sp.Max)):
        normalized = _safe_simplify(expr)
        if normalized != expr:
            return _maximum_expr_over_range(
                normalized,
                loop_symbol,
                start,
                step,
                iterations,
            )

    if not isinstance(expr, sp.Piecewise) and expr.has(sp.Piecewise):
        folded = cast(ResourceExpr, sp.piecewise_fold(expr))
        if folded != expr:
            return _maximum_expr_over_range(
                folded,
                loop_symbol,
                start,
                step,
                iterations,
            )

    if isinstance(iterations, sp.Integer) and 0 <= int(iterations) <= 64:
        count = int(iterations)
        if count == 0:
            return _ZERO, sp.false
        values = [
            cast(ResourceExpr, expr.subs(loop_symbol, start + step * index))
            for index in range(count)
        ]
        return cast(ResourceExpr, sp.Max(*values)), sp.false

    if isinstance(expr, sp.Max):
        argument_maxima = [
            _maximum_expr_over_range(
                cast(ResourceExpr, argument),
                loop_symbol,
                start,
                step,
                iterations,
            )
            for argument in expr.args
        ]
        return (
            cast(ResourceExpr, sp.Max(*(maximum for maximum, _ in argument_maxima))),
            _boolean_condition(sp.Or(*(guard for _, guard in argument_maxima))),
        )

    if isinstance(expr, sp.Piecewise):
        return _maximum_piecewise_over_range(
            expr,
            loop_symbol,
            start,
            step,
            iterations,
        )

    index = sp.Dummy("k", integer=True, nonnegative=True)
    transformed = cast(
        sp.Expr,
        expr.subs(loop_symbol, start + step * index),
    )
    first = cast(ResourceExpr, transformed.subs(index, _ZERO))
    last = cast(ResourceExpr, transformed.subs(index, iterations - _ONE))
    try:
        polynomial = sp.Poly(transformed, index)
    except sp.PolynomialError:
        polynomial = None
    if polynomial is not None and polynomial.degree() <= 1:
        return _piecewise(sp.Max(first, last), _ZERO, condition), sp.false

    return _conservative_maximum_sum_bound(
        expr,
        loop_symbol,
        start,
        step,
        iterations,
    )


def _maximum_piecewise_over_range(
    expression: sp.Piecewise,
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> tuple[ResourceExpr, Boolean]:
    """Maximize an endpoint-bounded Piecewise expression over a loop.

    Piecewise resource formulas commonly encode special control arities, such
    as zero ancillas for one or two controls and an affine fallback above that.
    Maximizing every branch over the full loop ignores those predicates and can
    select an unreachable value. This routine instead evaluates the complete
    Piecewise expression at loop endpoints and every linear predicate boundary.

    Args:
        expression (sp.Piecewise): Per-iteration resource expression.
        loop_symbol (sp.Symbol): Loop variable symbol.
        start (ResourceExpr): First loop value.
        step (ResourceExpr): Loop step.
        iterations (ResourceExpr): Number of executed iterations.

    Returns:
        tuple[ResourceExpr, Boolean]: Predicate-aware maximum and the
        condition under which its conservative fallback may overestimate.
    """
    index = sp.Dummy("piecewise_index", integer=True, nonnegative=True)
    folded = cast(sp.Piecewise, sp.piecewise_fold(expression))
    transformed = cast(
        sp.Piecewise,
        folded.subs(loop_symbol, start + step * index),
    )
    boundaries: set[ResourceExpr] = set()
    conditions_supported = True
    branches_endpoint_bounded = True
    for branch_expression, condition in cast(
        tuple[tuple[sp.Expr, sp.Basic], ...],
        transformed.args,
    ):
        branch_boundaries, supported = _linear_condition_boundaries(
            condition,
            index,
        )
        boundaries.update(branch_boundaries)
        conditions_supported = conditions_supported and supported
        branches_endpoint_bounded = branches_endpoint_bounded and (
            _endpoint_bounded_expression(branch_expression, index)
            or _condition_restricts_to_finite_points(condition, index)
        )

    if conditions_supported and branches_endpoint_bounded:
        candidates: set[ResourceExpr] = {_ZERO, iterations - _ONE}
        for boundary in boundaries:
            floor = cast(ResourceExpr, sp.floor(boundary))
            ceiling = cast(ResourceExpr, sp.ceiling(boundary))
            candidates.update(
                {
                    boundary,
                    floor - _ONE,
                    floor,
                    ceiling,
                    ceiling + _ONE,
                }
            )
        values = [
            _guarded_range_candidate(transformed, index, candidate, iterations)
            for candidate in candidates
        ]
        return cast(ResourceExpr, sp.Max(*values)), sp.false

    return _conservative_maximum_sum_bound(
        cast(ResourceExpr, folded),
        loop_symbol,
        start,
        step,
        iterations,
    )


def _conservative_maximum_sum_bound(
    expression: ResourceExpr,
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> tuple[ResourceExpr, Boolean]:
    """Bound a loop maximum while preserving its first-iteration baseline.

    Summing the complete nonnegative expression is safe but unnecessarily
    multiplies every loop-invariant baseline.  Instead, retain the first
    iteration once and sum only positive excess above it.  Besides producing a
    tighter bound, this keeps an inactive conditional contribution at zero
    after later parameter substitution.

    Args:
        expression (ResourceExpr): Per-iteration nonnegative resource value.
        loop_symbol (sp.Symbol): Loop variable symbol.
        start (ResourceExpr): First loop value.
        step (ResourceExpr): Loop step.
        iterations (ResourceExpr): Number of executed iterations.

    Returns:
        tuple[ResourceExpr, Boolean]: Conservative maximum and the condition
        under which positive excess occurs in multiple iterations and makes
        the bound potentially inexact.
    """
    first = cast(ResourceExpr, expression.subs(loop_symbol, start))
    baseline = _resource_max(_ZERO, first)
    excess = _resource_max(
        _ZERO,
        cast(ResourceExpr, expression - baseline),
    )
    if excess.has(sp.Piecewise):
        try:
            excess = cast(ResourceExpr, sp.piecewise_fold(excess))
        except (RecursionError, TypeError, ValueError):
            pass
    upper_bound = cast(
        ResourceExpr,
        baseline
        + _sum_expr(
            excess,
            loop_symbol,
            start,
            step,
            iterations,
        ),
    )
    repeated_excess = _RangeAtLeastTwo(
        sp.Lambda(loop_symbol, _resource_activity_condition(excess)),
        start,
        step,
        iterations,
    )
    conservative_when = _boolean_condition(sp.Ne(repeated_excess, _ZERO))
    return (
        _piecewise(upper_bound, _ZERO, sp.Gt(iterations, _ZERO)),
        conservative_when,
    )


def _endpoint_bounded_expression(expression: sp.Expr, index: sp.Symbol) -> bool:
    """Return whether interval endpoints determine an expression's maximum.

    Args:
        expression (sp.Expr): One Piecewise branch expression.
        index (sp.Symbol): Integer loop-position symbol.

    Returns:
        bool: Whether the expression is affine, constant, or a maximum of such
        expressions.
    """
    if index not in expression.free_symbols:
        return True
    if isinstance(expression, sp.Max):
        return all(
            _endpoint_bounded_expression(cast(sp.Expr, argument), index)
            for argument in expression.args
        )
    try:
        return sp.Poly(expression, index).degree() <= 1
    except sp.PolynomialError:
        return False


def _condition_restricts_to_finite_points(
    condition: sp.Basic,
    index: sp.Symbol,
) -> bool:
    """Return whether a predicate admits only finitely many loop positions.

    Args:
        condition (sp.Basic): Piecewise branch predicate.
        index (sp.Symbol): Integer loop-position symbol.

    Returns:
        bool: Whether linear equalities restrict the branch to finite points.
    """
    if isinstance(condition, sp.Equality):
        boundaries, supported = _linear_condition_boundaries(condition, index)
        return supported and bool(boundaries)
    if isinstance(condition, sp.logic.boolalg.Or):
        return all(
            _condition_restricts_to_finite_points(
                cast(sp.Basic, argument),
                index,
            )
            for argument in condition.args
        )
    if isinstance(condition, sp.logic.boolalg.And):
        return any(
            _condition_restricts_to_finite_points(
                cast(sp.Basic, argument),
                index,
            )
            for argument in condition.args
        )
    return False


def _guarded_range_candidate(
    expression: sp.Expr,
    index: sp.Symbol,
    candidate: ResourceExpr,
    iterations: ResourceExpr,
) -> ResourceExpr:
    """Evaluate one candidate only when it is a valid loop position.

    Args:
        expression (sp.Expr): Loop-indexed Piecewise expression.
        index (sp.Symbol): Integer loop-position symbol.
        candidate (ResourceExpr): Candidate index to evaluate.
        iterations (ResourceExpr): Number of loop iterations.

    Returns:
        ResourceExpr: Candidate resource value, or zero when out of range.
    """
    condition: Boolean = sp.And(
        sp.Gt(iterations, _ZERO),
        sp.Ge(candidate, _ZERO),
        sp.Lt(candidate, iterations),
    )
    if candidate.is_integer is not True:
        condition = sp.And(
            condition,
            cast(Boolean, sp.Eq(candidate, sp.floor(candidate))),
        )
    value = cast(ResourceExpr, expression.subs(index, candidate))
    return _piecewise(value, _ZERO, condition)
