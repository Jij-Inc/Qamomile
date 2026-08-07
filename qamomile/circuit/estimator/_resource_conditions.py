"""Normalize Boolean and activity conditions for resource expressions."""

from __future__ import annotations

import math
from typing import Any, cast

import sympy as sp
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._constants import _ONE, _ZERO
from qamomile.circuit.estimator._resource_base import (
    ResourceExpr,
    _is_concrete_integer,
)

_RANGE_ANY_REPLAY_LIMIT = 1024


def _boolean_condition(condition: sp.Basic) -> Boolean:
    """Normalize a symbolic branch predicate to a SymPy Boolean.

    Args:
        condition (sp.Basic): Boolean or numeric branch expression.

    Returns:
        Boolean: Boolean predicate with numeric truth represented as nonzero.
    """
    branches = cast(tuple[Any, ...], condition.args)
    if isinstance(condition, sp.Piecewise) and branches[-1][1] is sp.true:
        remaining: Boolean = sp.true
        active_terms: list[Boolean] = []
        for value, raw_guard in branches:
            guard = _boolean_condition(cast(sp.Basic, raw_guard))
            active_terms.append(
                cast(
                    Boolean,
                    sp.And(
                        remaining,
                        guard,
                        _boolean_condition(cast(sp.Basic, value)),
                    ),
                )
            )
            remaining = cast(Boolean, sp.And(remaining, sp.Not(guard)))
        return cast(Boolean, sp.Or(*active_terms))
    if (
        condition in (sp.true, sp.false)
        or isinstance(condition, sp.logic.boolalg.BooleanFunction)
        or getattr(condition, "is_Relational", False)
    ):
        return cast(Boolean, condition)
    return cast(Boolean, sp.Ne(condition, 0))


def _unresolved_condition_guard(condition: sp.Basic) -> Boolean:
    """Keep metadata active only while a branch predicate remains symbolic.

    A symbolic condition indicator is intentionally compared with both of its
    concrete values. SymPy cannot resolve either comparison until parameter
    substitution chooses zero or one, at which point the conjunction becomes
    false and guarded branch-union metadata is pruned.

    Args:
        condition (sp.Basic): Branch predicate whose resolution state guards
            conservative metadata.

    Returns:
        Boolean: Potentially active guard while ``condition`` is symbolic, or
            false after it resolves to either Boolean value.
    """
    indicator = _ConditionIndicator(_boolean_condition(condition))
    return cast(
        Boolean,
        sp.And(
            sp.Ne(indicator, _ZERO),
            sp.Ne(indicator, _ONE),
        ),
    )


class _ConditionIndicator(sp.Function):
    """Encode a Boolean predicate as a binder-safe zero-or-one expression.

    This node stays opaque to SymPy's ``piecewise_fold`` while symbolic.
    A nested ``Piecewise`` under ``Sum`` can otherwise release the sum's bound
    variable into a public resource expression.
    """

    nargs = 1
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, condition: sp.Basic) -> sp.Integer | None:
        """Reduce a concrete Boolean predicate to zero or one.

        Args:
            condition (sp.Basic): Boolean predicate to encode.

        Returns:
            sp.Integer | None: One for true, zero for false, or ``None`` while
            the predicate remains symbolic.
        """
        if condition is sp.true:
            return _ONE
        if condition is sp.false:
            return _ZERO
        return None


class _PhaseIdentity(sp.Function):
    """Classify a phase as identity or nonidentity after substitution.

    Args:
        phase (sp.Expr): Phase angle in radians.
    """

    nargs = 1

    @classmethod
    def eval(cls, phase: sp.Expr) -> sp.Integer | None:
        """Evaluate a concrete phase modulo one full turn.

        Args:
            phase (sp.Expr): Phase angle in radians.

        Returns:
            sp.Integer | None: One for an identity phase, zero for a concrete
            nonidentity phase, or ``None`` to retain a symbolic application.
        """
        normalized = cast(sp.Expr, sp.sympify(phase))
        if not normalized.is_number:
            return None
        if normalized.is_real is False:
            return _ZERO
        turns = sp.simplify(normalized / (2 * sp.pi))
        if turns.is_integer is True:
            return _ONE
        try:
            numeric_value = float(sp.N(normalized))
        except (TypeError, ValueError):
            return _ZERO
        if not math.isfinite(numeric_value):
            return _ZERO

        if numeric_value == 0.0:
            # A nonzero exact value can underflow when converted to ``float``.
            # Only the exact zero recognized above is an identity phase.
            return _ZERO
        remainder = math.fmod(numeric_value, math.tau)
        return sp.Integer(int(remainder == 0.0))


class _RangeAny(sp.Function):
    """Represent whether any value in a finite integer range satisfies a guard.

    The predicate is stored in a one-argument ``Lambda`` so its induction
    variable remains bound across substitution and ``Piecewise`` rewriting.
    Small concrete ranges are replayed directly. Affine relational predicates
    over larger concrete ranges are decided from their finitely many truth
    boundaries. Nonlinear or symbolic large ranges retain this expression
    instead of asking SymPy to perform expensive general set construction.
    """

    nargs = 4
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(
        cls,
        predicate: sp.Basic,
        start: sp.Expr,
        step: sp.Expr,
        iterations: sp.Expr,
    ) -> sp.Integer | None:
        """Resolve a finite existential guard over a concrete range.

        Args:
            predicate (sp.Basic): One-argument Boolean Lambda.
            start (sp.Expr): First Python-range value.
            step (sp.Expr): Python-range step.
            iterations (sp.Expr): Number of executed iterations.

        Returns:
            sp.Integer | None: One when any position satisfies the predicate,
            zero when none do, or ``None`` when symbolic or unresolved.
        """
        if not isinstance(predicate, sp.Lambda) or len(predicate.variables) != 1:
            return None
        if iterations.is_zero is True:
            return _ZERO
        loop_symbol = predicate.variables[0]
        condition = _boolean_condition(cast(sp.Basic, predicate.expr))
        if loop_symbol not in condition.free_symbols:
            if condition is sp.true:
                if iterations.is_positive is True:
                    return _ONE
                return None
            if condition is sp.false:
                return _ZERO
            return None
        if not (
            start.is_integer is True
            and start.is_number
            and step.is_integer is True
            and step.is_number
            and iterations.is_integer is True
            and iterations.is_number
        ):
            return None
        count = int(iterations)
        if count <= 0:
            return _ZERO
        if condition.free_symbols - {loop_symbol}:
            return None
        if count > _RANGE_ANY_REPLAY_LIMIT:
            return _resolve_large_affine_range_any(
                condition,
                loop_symbol,
                start,
                step,
                count,
            )
        unresolved = False
        for offset in range(count):
            transformed = _boolean_condition(
                cast(
                    sp.Basic,
                    condition.subs(loop_symbol, start + step * offset),
                )
            )
            if transformed is sp.true:
                return _ONE
            if transformed is not sp.false:
                unresolved = True
        return None if unresolved else _ZERO


class _RangeAtLeastTwo(sp.Function):
    """Represent whether a finite integer range activates a guard twice.

    The predicate is stored in a one-argument ``Lambda`` so its induction
    variable remains bound during substitution. Concrete ranges up to the
    shared replay limit are evaluated directly; larger or symbolic ranges
    retain this bounded expression rather than constructing a relational over
    an unevaluated ``Sum``.
    """

    nargs = 4
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(
        cls,
        predicate: sp.Basic,
        start: sp.Expr,
        step: sp.Expr,
        iterations: sp.Expr,
    ) -> sp.Integer | None:
        """Resolve whether at least two concrete positions satisfy a guard.

        Args:
            predicate (sp.Basic): One-argument Boolean Lambda.
            start (sp.Expr): First Python-range value.
            step (sp.Expr): Python-range step.
            iterations (sp.Expr): Number of executed iterations.

        Returns:
            sp.Integer | None: One when at least two positions satisfy the
                predicate, zero when fewer than two do, or ``None`` when the
                bounded check must remain symbolic.
        """
        if not isinstance(predicate, sp.Lambda) or len(predicate.variables) != 1:
            return None
        if not (
            iterations.is_integer is True
            and iterations.is_number
            and start.is_integer is True
            and start.is_number
            and step.is_integer is True
            and step.is_number
        ):
            return None
        count = int(iterations)
        if count <= 1:
            return _ZERO
        loop_symbol = predicate.variables[0]
        condition = _boolean_condition(cast(sp.Basic, predicate.expr))
        if loop_symbol not in condition.free_symbols:
            if condition is sp.true:
                return _ONE
            if condition is sp.false:
                return _ZERO
            return None
        if condition.free_symbols - {loop_symbol}:
            return None
        if count > _RANGE_ANY_REPLAY_LIMIT:
            return None
        matches = 0
        unresolved = False
        for offset in range(count):
            transformed = _boolean_condition(
                cast(
                    sp.Basic,
                    condition.subs(loop_symbol, start + step * offset),
                )
            )
            if transformed is sp.true:
                matches += 1
                if matches >= 2:
                    return _ONE
            elif transformed is not sp.false:
                unresolved = True
        return None if unresolved else _ZERO


def _resolve_large_affine_range_any(
    condition: Boolean,
    loop_symbol: sp.Symbol,
    start: sp.Expr,
    step: sp.Expr,
    count: int,
) -> sp.Integer | None:
    """Decide an affine Boolean guard from its finite truth boundaries.

    After mapping the Python-range value to a zero-based position, a Boolean
    combination of affine relational atoms can change truth only at an atom's
    root. Testing both endpoints, every integral root, and the adjacent integer
    positions therefore decides existence independently of the range length.

    Args:
        condition (Boolean): Guard whose only free symbol is ``loop_symbol``.
        loop_symbol (sp.Symbol): Symbol representing the Python-range value.
        start (sp.Expr): Concrete first range value.
        step (sp.Expr): Concrete nonzero range step.
        count (int): Positive number of range positions.

    Returns:
        sp.Integer | None: One when a position satisfies the guard, zero when
        none do, or ``None`` when the predicate is not an affine relational
        Boolean formula or a candidate cannot be resolved exactly.
    """
    position = sp.Dummy("range_position", integer=True, nonnegative=True)
    transformed = _boolean_condition(
        cast(
            sp.Basic,
            condition.subs(loop_symbol, start + step * position),
        )
    )
    boundaries, supported = _linear_condition_boundaries(
        transformed,
        position,
    )
    if not supported:
        return None

    candidates = {0, count - 1}
    for boundary in boundaries:
        if boundary.is_number is not True or boundary.is_finite is not True:
            return None
        floor = cast(sp.Expr, sp.floor(boundary))
        ceiling = cast(sp.Expr, sp.ceiling(boundary))
        for candidate in (
            floor - _ONE,
            floor,
            ceiling,
            ceiling + _ONE,
        ):
            if candidate.is_number is not True or not _is_concrete_integer(candidate):
                return None
            candidates.add(int(candidate))

    unresolved = False
    for candidate in candidates:
        if not 0 <= candidate < count:
            continue
        resolved = _boolean_condition(
            cast(
                sp.Basic,
                transformed.subs(position, sp.Integer(candidate)),
            )
        )
        if resolved is sp.true:
            return _ONE
        if resolved is not sp.false:
            unresolved = True
    return None if unresolved else _ZERO


def _resource_activity_condition(expression: ResourceExpr) -> Boolean:
    """Return a conservative nonzero-resource guard with bound hygiene.

    SymPy may simplify ``Sum(..., (k, ...)) > 0`` into a predicate that leaks
    the bound ``k`` outside the sum. Each finite sum is replaced by a
    zero-or-one nonempty-range proxy before testing the enclosing nonnegative
    expression. This retains surrounding multiplication and branch structure
    without exposing bound symbols. A nonempty all-zero integrand can overstate
    activity, but never creates an optimistic depth.

    Args:
        expression (ResourceExpr): Nonnegative resource expression.

    Returns:
        Boolean: True, false, or a bound-safe positivity predicate.
    """
    if expression == _ZERO or expression.is_zero is True:
        return sp.false
    if expression.is_positive is True:
        return sp.true
    if isinstance(expression, sp.Piecewise):
        remaining: Boolean = sp.true
        active_branches: list[Boolean] = []
        for branch in expression.args:
            value, condition = branch.args
            branch_condition = _boolean_condition(cast(sp.Basic, condition))
            effective_condition = _and_conditions(remaining, branch_condition)
            branch_activity = _resource_activity_condition(cast(ResourceExpr, value))
            active_branches.append(
                _and_conditions(effective_condition, branch_activity)
            )
            remaining = _and_conditions(
                remaining,
                cast(Boolean, sp.Not(branch_condition)),
            )
            if remaining is sp.false:
                break
        return cast(Boolean, sp.Or(*active_branches))
    replacements: dict[sp.Sum, ResourceExpr] = {}
    for summation in expression.atoms(sp.Sum):
        range_conditions: list[Boolean] = []
        for limit in summation.limits:
            if len(limit) != 3:
                range_conditions = [sp.true]
                break
            _symbol, lower, upper = limit
            range_conditions.append(cast(Boolean, sp.Ge(upper, lower)))
        nonempty = sp.And(*range_conditions) if range_conditions else sp.true
        replacements[summation] = _piecewise(_ONE, _ZERO, nonempty)
    proxy = cast(ResourceExpr, expression.xreplace(replacements))
    return _boolean_condition(sp.Gt(proxy, _ZERO))


def _and_conditions(left: sp.Basic, right: sp.Basic) -> Boolean:
    """Conjoin two symbolic activation predicates with trivial folding.

    Args:
        left (sp.Basic): Existing activation predicate.
        right (sp.Basic): Additional activation predicate.

    Returns:
        Boolean: Simplified conjunction.
    """
    left_condition = _boolean_condition(left)
    right_condition = _boolean_condition(right)
    if left_condition is sp.false or right_condition is sp.false:
        return sp.false
    if left_condition is sp.true:
        return right_condition
    if right_condition is sp.true:
        return left_condition
    return sp.And(left_condition, right_condition)


def _rewrite_condition(condition: sp.Basic, fn: Any) -> Boolean:
    """Rewrite and normalize one symbolic activation predicate.

    Args:
        condition (sp.Basic): Predicate to rewrite.
        fn (Any): Symbolic-expression rewrite function.

    Returns:
        Boolean: Rewritten Boolean predicate.
    """
    rewritten = cast(sp.Basic, fn(condition))
    return _boolean_condition(rewritten)


def _piecewise(
    true_value: ResourceExpr,
    false_value: ResourceExpr,
    condition: sp.Basic,
) -> ResourceExpr:
    """Select one resource expression with a symbolic Boolean.

    Args:
        true_value (ResourceExpr): Value when ``condition`` is true.
        false_value (ResourceExpr): Value when ``condition`` is false.
        condition (sp.Basic): SymPy Boolean predicate.

    Returns:
        ResourceExpr: Simplified piecewise expression.
    """
    return cast(
        ResourceExpr,
        sp.Piecewise((true_value, cast(Any, condition)), (false_value, True)),
    )


def _linear_condition_boundaries(
    condition: sp.Basic,
    index: sp.Symbol,
) -> tuple[set[ResourceExpr], bool]:
    """Extract roots of linear relational atoms in a branch predicate.

    Args:
        condition (sp.Basic): Piecewise branch condition.
        index (sp.Symbol): Integer loop-position symbol.

    Returns:
        tuple[set[ResourceExpr], bool]: Predicate boundaries and whether every
        index-dependent atom was supported.
    """
    if condition in (sp.true, sp.false):
        return set(), True
    if isinstance(
        condition,
        (sp.logic.boolalg.And, sp.logic.boolalg.Or, sp.logic.boolalg.Not),
    ):
        boundaries: set[ResourceExpr] = set()
        supported = True
        for argument in condition.args:
            nested, nested_supported = _linear_condition_boundaries(
                cast(sp.Basic, argument),
                index,
            )
            boundaries.update(nested)
            supported = supported and nested_supported
        return boundaries, supported
    if not isinstance(condition, Relational):
        return set(), index not in condition.free_symbols
    difference = cast(sp.Expr, condition.lhs) - cast(sp.Expr, condition.rhs)
    if index not in difference.free_symbols:
        return set(), True
    try:
        polynomial = sp.Poly(difference, index)
    except sp.PolynomialError:
        return set(), False
    if polynomial.degree() != 1:
        return set(), False
    slope, intercept = polynomial.all_coeffs()
    return {cast(ResourceExpr, -intercept / slope)}, True


def _activation_over_range(
    condition: sp.Basic,
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> sp.Basic:
    """Return whether a guarded fact is active in any loop iteration.

    Args:
        condition (sp.Basic): Per-iteration activation condition.
        loop_symbol (sp.Symbol): Loop variable symbol.
        start (ResourceExpr): First loop value.
        step (ResourceExpr): Loop step.
        iterations (ResourceExpr): Number of executed iterations.

    Returns:
        sp.Basic: Condition that is true exactly when at least one reachable
            iteration activates the fact.
    """
    active = _boolean_condition(condition)
    nonempty = sp.Gt(iterations, _ZERO)
    if loop_symbol not in active.free_symbols:
        return _boolean_condition(sp.And(nonempty, active))
    active_iterations = _RangeAny(
        sp.Lambda(loop_symbol, active),
        start,
        step,
        iterations,
    )
    return _boolean_condition(sp.Ne(active_iterations, _ZERO))
