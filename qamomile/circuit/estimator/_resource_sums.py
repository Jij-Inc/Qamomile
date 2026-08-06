"""Analyze and evaluate symbolic resource sums."""

from __future__ import annotations

from typing import cast

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._constants import _ONE, _ZERO
from qamomile.circuit.estimator._resource_base import (
    _SUM_EAGER_EVALUATION_LIMIT,
    ResourceExpr,
    _is_concrete_integer,
)
from qamomile.circuit.estimator._resource_conditions import (
    _boolean_condition,
    _linear_condition_boundaries,
)


def _large_sum_expression_is_structurally_negative(expression: sp.Expr) -> bool:
    """Prove a simple negative multiple of positive concrete sums.

    Querying SymPy's generic sign properties on an unsupported huge ``Sum``
    can itself invoke numerical summation. This deliberately narrow proof
    preserves the resource clamping contract for the common ``-Sum(...)``
    shape without evaluating arbitrary wrappers.

    Args:
        expression (sp.Expr): Large-sum expression to inspect.

    Returns:
        bool: Whether the expression is a negative coefficient times factors
            that are structurally nonnegative and include a positive finite
            sum.
    """
    coefficient, remainder = expression.as_coeff_Mul()
    if coefficient.is_negative is not True:
        return False
    factors = remainder.args if isinstance(remainder, sp.Mul) else (remainder,)
    has_positive_sum = False
    for factor in factors:
        if isinstance(factor, sp.Sum):
            if factor.function.is_nonnegative is not True:
                return False
            for limit in factor.limits:
                if len(limit) != 3:
                    return False
                _symbol, lower, upper = limit
                count = cast(sp.Expr, upper - lower + _ONE)
                if count.is_positive is not True:
                    return False
            has_positive_sum = True
        else:
            # Asking a wrapper around the same Sum for generic sign metadata
            # can trigger the expensive summation that this proof exists to
            # avoid. Only direct Sum factors are intentionally recognized.
            return False
    return has_positive_sum


def _large_sum_expression_is_structurally_nonnegative(expression: sp.Expr) -> bool:
    """Prove nonnegativity without evaluating a retained finite sum.

    Each finite sum is replaced by a fresh nonnegative proxy only when its
    summand is already known nonnegative and every limit has nonnegative
    cardinality. The surrounding arithmetic can then use inexpensive SymPy
    sign inference without invoking finite-sum evaluation. Failure to prove a
    sign is intentional: the caller retains the resource lower-bound contract
    with an unevaluated ``Max(0, expression)``.

    Args:
        expression (sp.Expr): Large-sum expression to inspect.

    Returns:
        bool: Whether the complete expression is structurally nonnegative.
    """
    if isinstance(expression, sp.Max):
        for argument in expression.args:
            if not _has_large_concrete_sum(argument):
                if argument.is_nonnegative is True:
                    return True
                continue
            if _large_sum_expression_is_structurally_nonnegative(
                cast(sp.Expr, argument)
            ):
                return True
        return False

    replacements: dict[sp.Sum, sp.Dummy] = {}
    for index, summation in enumerate(expression.atoms(sp.Sum)):
        if not _finite_sum_is_structurally_nonnegative(summation):
            return False
        replacements[summation] = sp.Dummy(
            f"resource_sum_{index}",
            nonnegative=True,
        )
    if not replacements:
        return expression.is_nonnegative is True
    proxy = cast(sp.Expr, expression.xreplace(replacements))
    return proxy.is_nonnegative is True


def _finite_sum_is_structurally_nonnegative(summation: sp.Sum) -> bool:
    """Prove one finite sum nonnegative from its rectangular limits.

    In addition to SymPy's direct sign metadata, this recognizes affine
    summands whose minimum lies at a known endpoint of each finite range. It
    deliberately declines nonlinear or sign-ambiguous cases instead of
    invoking a general optimizer.

    Args:
        summation (sp.Sum): Finite sum to inspect without evaluating.

    Returns:
        bool: Whether every summand in the declared finite ranges is proven
            nonnegative and each range has nonnegative cardinality.
    """
    minimum = cast(sp.Expr, summation.function)
    for limit in summation.limits:
        if len(limit) != 3:
            return False
        symbol, lower, upper = limit
        count = cast(sp.Expr, upper - lower + _ONE)
        if count.is_nonnegative is not True:
            return False
        if minimum.is_nonnegative is True:
            continue
        try:
            polynomial = sp.Poly(minimum, symbol)
        except sp.PolynomialError:
            return False
        if polynomial.degree() > 1:
            return False
        coefficient = cast(sp.Expr, sp.diff(minimum, symbol))
        if symbol in coefficient.free_symbols:
            return False
        if coefficient.is_nonnegative is True:
            endpoint = lower
        elif coefficient.is_nonpositive is True:
            endpoint = upper
        else:
            return False
        minimum = cast(sp.Expr, minimum.subs(symbol, endpoint))
    return minimum.is_nonnegative is True


def _has_large_concrete_sum(expression: sp.Basic) -> bool:
    """Return whether an expression contains an expensive concrete sum.

    Args:
        expression (sp.Basic): Expression whose finite sums should be checked.

    Returns:
        bool: Whether any concrete sum dimension, or their known combined
            work, exceeds the eager evaluation budget.
    """
    for summation in expression.atoms(sp.Sum):
        concrete_work = _ONE
        for limit in summation.limits:
            if len(limit) != 3:
                # An indefinite or malformed dimension says nothing about a
                # later finite dimension, which can still exceed the eager
                # evaluation budget on its own.
                continue
            _symbol, lower, upper = limit
            count = cast(sp.Expr, upper - lower + _ONE)
            if not (count.is_number and _is_concrete_integer(count)):
                continue
            if count.is_zero is True:
                concrete_work = _ZERO
                continue
            if count.is_negative is True:
                # SymPy uses Karr's reversed-limit convention rather than an
                # empty range, so a large negative cardinality can still be an
                # expensive symbolic evaluation.
                count = cast(sp.Expr, sp.Abs(count))
            concrete_work = cast(sp.Expr, concrete_work * count)
            if (
                sp.Gt(
                    concrete_work,
                    _SUM_EAGER_EVALUATION_LIMIT,
                )
                is sp.true
            ):
                return True
    return False


def _evaluate_constant_piecewise_sum(
    summation: sp.Sum,
) -> ResourceExpr | None:
    """Evaluate a concrete constant-valued Piecewise sum by set cardinality.

    A branch such as ``Piecewise((1, k > 0), (0, True))`` has a constant
    value on each integer region. Counting those regions is independent of the
    range magnitude, unlike SymPy's generic finite-sum evaluator. Unsupported
    predicates or index-dependent branch values deliberately retain ``Sum``.

    Args:
        summation (sp.Sum): Candidate one-dimensional finite sum.

    Returns:
        ResourceExpr | None: Exact closed form when every active integer region
            has a supported cardinality, otherwise ``None``.
    """
    limits = tuple(tuple(limit) for limit in summation.limits)
    if len(limits) != 1:
        return None
    limit = limits[0]
    if len(limit) != 3:
        return None
    symbol, lower, upper = limit
    count = cast(sp.Expr, upper - lower + _ONE)
    if not (
        isinstance(symbol, sp.Symbol)
        and lower.is_number
        and upper.is_number
        and _is_concrete_integer(lower)
        and _is_concrete_integer(upper)
        and count.is_positive is True
        and isinstance(summation.function, sp.Piecewise)
    ):
        return None

    domain = sp.Range(lower, upper + _ONE)
    prior_conditions: list[Boolean] = []
    terms: list[ResourceExpr] = []
    has_terminal_branch = False
    for pair in summation.function.args:
        value, raw_condition = pair.args
        if symbol in value.free_symbols:
            return None
        condition = _boolean_condition(cast(sp.Basic, raw_condition))
        effective = cast(
            Boolean,
            sp.And(
                condition,
                *(sp.Not(previous) for previous in prior_conditions),
            ),
        )
        if effective is sp.false:
            cardinality = _ZERO
        elif effective is sp.true:
            cardinality = count
        else:
            if effective.free_symbols - {symbol}:
                return None
            _boundaries, supported = _linear_condition_boundaries(
                effective,
                symbol,
            )
            if not supported:
                return None
            try:
                active_values = sp.Intersection(domain, effective.as_set())
            except (
                ArithmeticError,
                AttributeError,
                NotImplementedError,
                RecursionError,
                TypeError,
                ValueError,
            ):
                return None
            cardinality = _finite_integer_set_cardinality(active_values)
            if cardinality is None:
                return None
        terms.append(cast(ResourceExpr, value * cardinality))
        prior_conditions.append(condition)
        if condition is sp.true:
            has_terminal_branch = True
            break
    return cast(ResourceExpr, sp.Add(*terms)) if has_terminal_branch else None


def _finite_integer_set_cardinality(values: sp.Set) -> ResourceExpr | None:
    """Return the exact cardinality of a normalized finite integer set.

    Args:
        values (sp.Set): Set produced by intersecting a concrete ``Range``
            with a Boolean predicate.

    Returns:
        ResourceExpr | None: Exact finite cardinality, or ``None`` when the set
            representation does not provide a cheap disjoint decomposition.
    """
    if values is sp.S.EmptySet or values.is_empty is True:
        return _ZERO
    if isinstance(values, sp.FiniteSet):
        return sp.Integer(len(values))
    if isinstance(values, sp.Range):
        return cast(ResourceExpr, values.size)
    if isinstance(values, sp.Union):
        subsets = tuple(cast(sp.Set, subset) for subset in values.args)
        for index, left in enumerate(subsets):
            if any(
                sp.Intersection(left, right).is_empty is not True
                for right in subsets[index + 1 :]
            ):
                return None
        cardinalities = tuple(
            _finite_integer_set_cardinality(subset) for subset in subsets
        )
        if any(cardinality is None for cardinality in cardinalities):
            return None
        return cast(
            ResourceExpr,
            sp.Add(*(cast(ResourceExpr, value) for value in cardinalities)),
        )
    return None


def _sum_expr(
    expr: ResourceExpr,
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> ResourceExpr:
    """Sum an expression over Python ``range`` semantics.

    Args:
        expr (ResourceExpr): Expression to sum.
        loop_symbol (sp.Symbol): Loop variable symbol.
        start (ResourceExpr): Start bound.
        step (ResourceExpr): Step value.
        iterations (ResourceExpr): Number of iterations.

    Returns:
        ResourceExpr: Summed expression.
    """
    if expr == _ZERO or iterations == _ZERO:
        return _ZERO
    if loop_symbol not in expr.free_symbols:
        return cast(ResourceExpr, expr * iterations)
    k = sp.Dummy("k", integer=True, nonnegative=True)
    transformed = expr.subs(loop_symbol, start + step * k)
    transformed = _simplify_sum_range_guards(
        transformed,
        symbol=k,
        lower=_ZERO,
        upper=iterations - 1,
    )
    summation = cast(
        sp.Sum,
        sp.Sum(transformed, (k, 0, iterations - 1)),
    )
    piecewise_total = _evaluate_constant_piecewise_sum(summation)
    if piecewise_total is not None:
        return piecewise_total
    if _has_large_concrete_sum(summation):
        return cast(ResourceExpr, summation)
    evaluated = cast(ResourceExpr, summation.doit())
    if not evaluated.free_symbols <= summation.free_symbols:
        return cast(ResourceExpr, summation)
    return cast(ResourceExpr, evaluated)


def _simplify_sum_range_guards(
    expression: sp.Expr,
    *,
    symbol: sp.Symbol,
    lower: sp.Expr,
    upper: sp.Expr,
) -> sp.Expr:
    """Remove nonnegative affine guards proven by a finite sum range.

    Nested ``qmc.range`` loops produce exact trip counts such as
    ``Max(0, n - 1 - k)``. SymPy does not use the enclosing summation bound
    ``0 <= k <= n - 1`` when simplifying that guard, so triangular loops stay
    as unevaluated sums. This helper removes only guards whose affine argument
    is provably nonnegative at the endpoint where it reaches its minimum.

    Args:
        expression (sp.Expr): Summand to simplify.
        symbol (sp.Symbol): Summation index.
        lower (sp.Expr): Inclusive lower index bound.
        upper (sp.Expr): Inclusive upper index bound.

    Returns:
        sp.Expr: Expression with range-proven ``Max(0, affine)`` guards removed.
    """
    replacements: dict[sp.Basic, sp.Basic] = {}
    for node in sp.preorder_traversal(expression):
        if (
            not isinstance(node, sp.Max)
            or len(node.args) != 2
            or _ZERO not in node.args
        ):
            continue
        guarded = node.args[0] if node.args[1] == _ZERO else node.args[1]
        try:
            polynomial = sp.Poly(guarded, symbol)
        except sp.PolynomialError:
            continue
        if polynomial.degree() > 1:
            continue
        slope = sp.diff(guarded, symbol)
        if slope.is_nonnegative:
            minimum = guarded.subs(symbol, lower)
        elif slope.is_nonpositive:
            minimum = guarded.subs(symbol, upper)
        else:
            continue
        if sp.simplify(minimum).is_nonnegative:
            replacements[node] = guarded
    return cast(sp.Expr, expression.xreplace(replacements))
