"""Build and simplify bounded symbolic resource expressions."""

from __future__ import annotations

import numbers
from typing import cast

import sympy as sp

from qamomile.circuit.estimator._constants import (
    _ONE,
    _ZERO,
)
from qamomile.circuit.estimator._resource_base import (
    ResourceExpr,
    _is_concrete_integer,
)
from qamomile.circuit.estimator._resource_expressions import (
    _boolean_condition,
    _ConditionIndicator,
    _expr,
    _has_large_concrete_sum,
    _safe_simplify,
)

# Global SymPy simplification becomes superlinear on nested branch extrema;
# resource composition has already normalized larger expressions structurally.
_PUBLIC_RESOURCE_SIMPLIFY_NODE_LIMIT = 32


class _CappedRangeSum(sp.Function):
    """Represent a loop-local batching sum with its induction variable bound.

    The control batching policy only needs to distinguish zero, one, and at
    least two units of work. Keeping the summand inside a ``Lambda`` prevents
    SymPy from releasing a ``Sum`` dummy into public parameters when a
    surrounding ``Piecewise`` is simplified.
    """

    nargs = 4
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(
        cls,
        summand: sp.Basic,
        start: sp.Expr,
        step: sp.Expr,
        iterations: sp.Expr,
    ) -> sp.Expr | None:
        """Evaluate a capped sum once its loop range is concrete.

        Args:
            summand (sp.Basic): One-argument Lambda for per-iteration work.
            start (sp.Expr): First Python-range value.
            step (sp.Expr): Python-range step.
            iterations (sp.Expr): Number of executed iterations.

        Returns:
            sp.Expr | None: Integer in ``[0, 2]`` when decidable, otherwise
            ``None`` to retain the bound symbolic node.
        """
        if not isinstance(summand, sp.Lambda) or len(summand.variables) != 1:
            return None
        if iterations.is_zero is True:
            return _ZERO
        loop_symbol = summand.variables[0]
        expression = cast(sp.Expr, summand.expr)
        if loop_symbol not in expression.free_symbols and expression.is_number:
            return cast(sp.Expr, sp.Min(2, expression * iterations))
        if not all(
            value.is_number and _is_concrete_integer(value)
            for value in (start, step, iterations)
        ):
            return None
        count = int(iterations)
        if count <= 0:
            return _ZERO
        offset = sp.Dummy("batch_offset", integer=True, nonnegative=True)
        transformed = expression.subs(
            loop_symbol,
            start + step * offset,
        )
        capped = _capped_nonnegative_integer_sum(
            transformed,
            offset,
            count,
        )
        if capped is not None:
            return sp.Integer(capped)
        expanded = expression.replace(
            lambda node: isinstance(node, _ConditionIndicator),
            lambda node: sp.Piecewise(
                (_ONE, cast(sp.Basic, node.args[0])),
                (_ZERO, True),
            ),
        )
        transformed = expanded.subs(
            loop_symbol,
            start + step * offset,
        )
        evaluated = cast(
            sp.Expr,
            sp.Sum(transformed, (offset, 0, count - 1)).doit(),
        )
        if evaluated.is_integer is True and evaluated.is_number:
            return sp.Integer(min(2, max(0, int(evaluated))))
        return None


def _simplify_public_resource_expression(
    expression: ResourceExpr,
) -> ResourceExpr:
    """Simplify a public metric unless it contains branching structure.

    ``_CappedRangeSum`` deliberately keeps a loop induction variable inside a
    ``Lambda``, while ``_ConditionIndicator`` keeps a Boolean predicate opaque
    to ``Piecewise`` rewriting. Resource expressions also contain nested
    ``Min``, ``Max``, and ``Piecewise`` nodes that are already structurally
    normalized while they are composed. SymPy's global simplifier spends
    substantial time exploring those nodes, usually returning an expression
    of the same shape and occasionally attempting to release a bound variable.
    Concrete substitution evaluates them directly, so retaining the symbolic
    form is both faster and safer.

    Args:
        expression (ResourceExpr): Public resource expression to normalize.

    Returns:
        ResourceExpr: Safely simplified expression, or the original
            structurally normalized expression when global simplification is
            unsafe or disproportionately expensive.
    """
    normalized = _expr(expression)
    if _has_large_concrete_sum(normalized):
        return normalized
    has_branching_structure = False
    for node_count, node in enumerate(
        sp.preorder_traversal(normalized),
        start=1,
    ):
        if isinstance(node, (_CappedRangeSum, _ConditionIndicator)):
            return normalized
        if isinstance(node, (sp.Max, sp.Min, sp.Piecewise)):
            has_branching_structure = True
        if (
            has_branching_structure
            and node_count > _PUBLIC_RESOURCE_SIMPLIFY_NODE_LIMIT
        ):
            return normalized
    return _safe_simplify(normalized)


def _capped_nonnegative_integer_sum(
    expression: sp.Expr,
    symbol: sp.Symbol,
    count: int,
) -> int | None:
    """Return a nonnegative integer range sum capped at two.

    Args:
        expression (sp.Expr): Per-position nonnegative integer expression.
        symbol (sp.Symbol): Zero-based range-position symbol.
        count (int): Number of positions.

    Returns:
        int | None: Exact capped sum, or ``None`` when the expression grammar
        cannot be decided without expanding the range.
    """
    if count <= 0 or expression == _ZERO:
        return 0
    if symbol not in expression.free_symbols and expression.is_number:
        return min(2, max(0, int(expression) * count))
    if isinstance(expression, _ConditionIndicator):
        return _capped_integer_condition_count(
            cast(sp.Basic, expression.args[0]),
            symbol,
            count,
        )
    coefficient, remainder = expression.as_coeff_Mul()
    if (
        coefficient.is_integer is True
        and coefficient.is_nonnegative is True
        and isinstance(remainder, _ConditionIndicator)
    ):
        active_count = _capped_integer_condition_count(
            cast(sp.Basic, remainder.args[0]),
            symbol,
            count,
        )
        if active_count is None:
            return None
        return min(2, int(coefficient) * active_count)
    if isinstance(expression, sp.Add):
        total = 0
        for term in expression.args:
            term_sum = _capped_nonnegative_integer_sum(
                cast(sp.Expr, term),
                symbol,
                count,
            )
            if term_sum is None:
                return None
            total = min(2, total + term_sum)
            if total == 2:
                return 2
        return total
    if isinstance(expression, sp.Min) and _expr(2) in expression.args:
        remaining = [arg for arg in expression.args if arg != _expr(2)]
        if len(remaining) == 1:
            return _capped_nonnegative_integer_sum(
                cast(sp.Expr, remaining[0]),
                symbol,
                count,
            )
    return None


def _capped_integer_condition_count(
    condition: sp.Basic,
    symbol: sp.Symbol,
    count: int,
) -> int | None:
    """Count satisfying integer range positions, capped at two.

    Args:
        condition (sp.Basic): Boolean condition over ``symbol``.
        symbol (sp.Symbol): Zero-based integer position.
        count (int): Exclusive upper bound.

    Returns:
        int | None: Exact capped cardinality, or ``None`` for an unsupported
        symbolic set.
    """
    normalized = _boolean_condition(condition)
    if normalized is sp.false or count <= 0:
        return 0
    if normalized is sp.true:
        return min(2, count)
    if normalized.free_symbols - {symbol}:
        return None
    try:
        satisfying = sp.Intersection(
            normalized.as_set(),
            sp.Range(0, count),
        )
    except (
        ArithmeticError,
        AttributeError,
        NotImplementedError,
        RecursionError,
        TypeError,
        ValueError,
    ):
        return None
    return _capped_integer_set_cardinality(satisfying)


def _capped_integer_set_cardinality(values: sp.Set) -> int | None:
    """Return an integer set's cardinality capped at two.

    Args:
        values (sp.Set): Integer-valued set.

    Returns:
        int | None: Exact capped cardinality, or ``None`` when unavailable.
    """
    if values is sp.S.EmptySet or values.is_empty is True:
        return 0
    if isinstance(values, sp.FiniteSet):
        return min(2, len(values))
    if isinstance(values, sp.Range):
        size = values.size
        if size.is_integer is True and size.is_number:
            return min(2, max(0, int(size)))
        return None
    if isinstance(values, sp.Union):
        total = 0
        for subset in values.args:
            subset_count = _capped_integer_set_cardinality(cast(sp.Set, subset))
            if subset_count is None:
                return None
            total = min(2, total + subset_count)
            if total == 2:
                return 2
        return total
    return None


def _normalize_resource_scalar(
    value: object,
    *,
    label: str,
    allow_symbolic: bool,
    allow_bool: bool,
) -> sp.Expr:
    """Normalize a public resource scalar and reject invalid number domains.

    Args:
        value (object): Candidate Python or SymPy scalar.
        label (str): User-facing parameter label for diagnostics.
        allow_symbolic (bool): Whether a non-numeric SymPy expression may
            remain in the estimate.
        allow_bool (bool): Whether a Boolean may be normalized to zero or one.

    Returns:
        sp.Expr: Normalized finite-real scalar or permitted symbolic
            expression.

    Raises:
        TypeError: If the value is Boolean when disallowed, is a string, or is
            not a supported numeric/SymPy scalar.
        ValueError: If a concrete number is non-real or non-finite, or a
            symbolic expression is provably non-real or non-finite.
    """
    expected = (
        "a numeric scalar or explicit SymPy expression"
        if allow_symbolic
        else "a concrete numeric scalar"
    )
    if isinstance(value, bool):
        if allow_bool:
            return sp.Integer(int(value))
        raise TypeError(f"{label} requires {expected}, got bool ({value!r}).")
    if isinstance(value, (str, bytes)) or not (
        isinstance(value, numbers.Complex) or isinstance(value, sp.Expr)
    ):
        raise TypeError(
            f"{label} requires {expected}, got {type(value).__name__} ({value!r})."
        )
    scalar = value.item() if hasattr(value, "item") else value
    try:
        normalized = sp.sympify(scalar)
    except (TypeError, ValueError, sp.SympifyError) as error:
        raise TypeError(
            f"{label} requires {expected}, got {type(value).__name__} ({value!r})."
        ) from error
    if not isinstance(normalized, sp.Expr):
        raise TypeError(
            f"{label} requires {expected}, got {type(value).__name__} ({value!r})."
        )
    if normalized.is_number is not True:
        if not allow_symbolic:
            raise TypeError(
                f"{label} requires {expected}, got {type(value).__name__} ({value!r})."
            )
        if normalized.is_real is False or normalized.is_finite is False:
            raise ValueError(f"{label} must be finite and real, got {value!r}.")
        return cast(sp.Expr, normalized)
    if normalized.is_real is not True or normalized.is_finite is not True:
        raise ValueError(f"{label} must be finite and real, got {value!r}.")
    return cast(sp.Expr, normalized)


def _canonicalize_concrete_integer(value: sp.Expr) -> sp.Expr:
    """Return an exact SymPy integer for an integer-valued concrete number.

    Python and NumPy integer-valued floats are accepted for integer resource
    parameters. Canonicalizing them at the public boundary prevents SymPy's
    undecided ``Float.is_integer`` property from retaining internal loop nodes
    after all user parameters have been substituted.

    Args:
        value (sp.Expr): Concrete or symbolic resource scalar.

    Returns:
        sp.Expr: Exact integer for a concrete integral value, otherwise the
            original expression.
    """
    if value.is_number and _is_concrete_integer(value):
        return sp.Integer(int(value))
    return value
