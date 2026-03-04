"""Loop execution strategies for resource estimation.

Provides:
  - Concrete range resolution (loop bounds → int)
  - Parametric interpolation (polynomial + exponential-linear model selection)

This is the ONLY module allowed to perform interpolation or sampling.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

import sympy as sp

from ._resolver import ExprResolver, UnresolvedValueError

T = TypeVar("T")

# Minimum number of valid sample points for interpolation.
_MIN_SAMPLES = 6
# Maximum sample n value.
_MAX_SAMPLE_N = 20
# Maximum recursion depth for nested interpolation.
_MAX_INTERP_DEPTH = 3


# ------------------------------------------------------------------ #
#  Concrete range resolution                                          #
# ------------------------------------------------------------------ #


def try_resolve_range(
    resolver: ExprResolver,
    start: Any,
    stop: Any,
    step: Any,
) -> tuple[int, int, int] | None:
    """Try to resolve loop bounds to concrete integers.

    Returns ``(start, stop, step)`` as ints, or ``None`` if any bound is
    symbolic.
    """
    try:
        s = resolver.resolve_concrete(start)
        e = resolver.resolve_concrete(stop)
        st = resolver.resolve_concrete(step)
        return (s, e, st)
    except UnresolvedValueError:
        return None


def concrete_range(start: int, stop: int, step: int) -> list[int]:
    """Python ``range()`` semantics — handles positive and negative step."""
    if step == 0:
        raise ValueError("Loop step cannot be zero")
    return list(range(start, stop, step))


# ------------------------------------------------------------------ #
#  Finding parametric symbols in bounds                               #
# ------------------------------------------------------------------ #


def find_parametric_symbols(
    start: sp.Expr,
    stop: sp.Expr,
    step: sp.Expr,
) -> set[sp.Symbol]:
    """Return free symbols appearing in loop bounds."""
    syms: set[sp.Symbol] = set()
    for expr in (start, stop, step):
        if isinstance(expr, sp.Expr):
            syms |= expr.free_symbols
    return syms


# ------------------------------------------------------------------ #
#  Sampling infrastructure                                            #
# ------------------------------------------------------------------ #


def collect_sample_points(
    start: sp.Expr,
    stop: sp.Expr,
    step: sp.Expr,
    param_sym: sp.Symbol,
    sample_fn: Callable[[int], sp.Expr | int | float],
    min_samples: int = _MIN_SAMPLES,
    max_n: int = _MAX_SAMPLE_N,
) -> list[tuple[int, sp.Expr]]:
    """Collect ``(n_value, metric_value)`` sample points.

    For each ``n`` in ``range(2, max_n+1)``:
      1. Substitute ``param_sym = n`` into bounds.
      2. If the resulting range has zero iterations, skip.
      3. Otherwise, call ``sample_fn(n)`` and record the result.

    Stops after collecting at least *min_samples* valid points.
    """
    points: list[tuple[int, sp.Expr]] = []
    for n_val in range(2, max_n + 1):
        subs = {param_sym: n_val}
        try:
            s = int(start.subs(subs)) if isinstance(start, sp.Expr) else int(start)
            e = int(stop.subs(subs)) if isinstance(stop, sp.Expr) else int(stop)
            st = int(step.subs(subs)) if isinstance(step, sp.Expr) else int(step)
        except (TypeError, ValueError):
            continue
        if st == 0 or len(range(s, e, st)) == 0:
            continue
        result = sample_fn(n_val)
        if isinstance(result, (int, float)):
            result = sp.Integer(result) if isinstance(result, int) else sp.Float(result)
        points.append((n_val, result))
        if len(points) >= min_samples:
            break
    return points


# ------------------------------------------------------------------ #
#  Model selection (deterministic — NOT fallback chain)               #
# ------------------------------------------------------------------ #


def interpolate_scalar(
    sample_points: list[tuple[int, sp.Expr]],
    symbol: sp.Symbol,
) -> sp.Expr:
    """Fit a symbolic expression to ``(n, value)`` sample points.

    Model selection strategy (deterministic):
      1. Split into train (first K−4) and holdout (last 4) points.
      2. Fit polynomial AND exponential+linear (``a·2^n + b·n + c``)
         simultaneously.
      3. Verify each candidate against ALL holdout points using exact
         ``sp.simplify(lhs − rhs) == 0``.
      4. Select:
         - Exactly one verifies → use it.
         - Both verify → use polynomial (Occam's razor).
         - Neither verifies → use polynomial on all points (best effort).
    """
    if not sample_points:
        return sp.Integer(0)

    if len(sample_points) == 1:
        _, v = sample_points[0]
        return v if isinstance(v, sp.Expr) else sp.Integer(int(v))

    # Sort by n value
    pts = sorted(sample_points, key=lambda p: p[0])

    # Constant check
    first_val = pts[0][1]
    if all(sp.simplify(v - first_val) == 0 for _, v in pts):
        return first_val

    # Split: train = first K−holdout, holdout = last
    holdout_count = _holdout_size(len(pts))
    if holdout_count > 0:
        train = pts[: len(pts) - holdout_count]
        holdout = pts[len(pts) - holdout_count :]
    else:
        train, holdout = pts, []

    # Fit both models simultaneously
    poly = _fit_polynomial(train, symbol)
    poly_ok = _verify_holdout(poly, holdout, symbol)

    exp_lin = _fit_exp_linear(train, symbol)
    exp_ok = _verify_holdout(exp_lin, holdout, symbol) if exp_lin is not None else False

    # Model selection
    if poly_ok and exp_ok:
        return poly  # Occam's razor
    if poly_ok:
        return poly
    if exp_ok and exp_lin is not None:
        return exp_lin

    # Neither verified on holdout — use polynomial on ALL points
    return _fit_polynomial(pts, symbol)


def interpolate_fields(
    field_samples: dict[str, list[tuple[int, sp.Expr]]],
    symbol: sp.Symbol,
) -> dict[str, sp.Expr]:
    """Interpolate each field independently.

    Args:
        field_samples: ``{field_name: [(n, value), ...]}``
        symbol: The parametric symbol.

    Returns:
        ``{field_name: symbolic_expr}``
    """
    return {
        name: interpolate_scalar(pts, symbol) for name, pts in field_samples.items()
    }


# ------------------------------------------------------------------ #
#  Symbolic loop iteration count                                      #
# ------------------------------------------------------------------ #


def symbolic_iterations(
    start: sp.Expr,
    stop: sp.Expr,
    step: sp.Expr,
) -> sp.Expr:
    """Compute the number of iterations of ``range(start, stop, step)``.

    Uses the discrete formula ``Max(0, ceiling((stop - start) / step))``
    which exactly matches Python ``range()`` semantics for any integer
    step (positive, negative, or symbolic).

    Raises:
        ValueError: If *step* is a concrete zero.
    """
    if hasattr(step, "is_zero") and step.is_zero:
        raise ValueError("Loop step cannot be zero")
    return sp.Max(0, sp.ceiling((stop - start) / step))


# ------------------------------------------------------------------ #
#  Internal helpers                                                   #
# ------------------------------------------------------------------ #


def _holdout_size(n: int) -> int:
    """Determine holdout set size given *n* total points."""
    if n >= 8:
        return 4
    if n >= 5:
        return min(4, n - 3)
    return 0


def _fit_polynomial(
    pts: list[tuple[int, sp.Expr]],
    symbol: sp.Symbol,
) -> sp.Expr:
    """Lagrange interpolation through sample points."""
    data = [(sp.Integer(n), v) for n, v in pts]
    return sp.interpolate(data, symbol)


def _fit_exp_linear(
    pts: list[tuple[int, sp.Expr]],
    symbol: sp.Symbol,
) -> sp.Expr | None:
    """Fit ``a·2^n + b·n + c`` using first 3 points.

    Returns the expression if coefficients are rational and the model
    verifies on ALL training points.  Returns ``None`` otherwise.
    """
    if len(pts) < 3:
        return None

    a, b, c = sp.symbols("_a _b _c")
    n1, v1 = pts[0]
    n2, v2 = pts[1]
    n3, v3 = pts[2]

    eqs = [
        a * sp.Integer(2) ** n1 + b * n1 + c - v1,
        a * sp.Integer(2) ** n2 + b * n2 + c - v2,
        a * sp.Integer(2) ** n3 + b * n3 + c - v3,
    ]

    sol = sp.solve(eqs, [a, b, c])
    if not sol:
        return None

    a_val = sol.get(a)
    b_val = sol.get(b)
    c_val = sol.get(c)
    if a_val is None or b_val is None or c_val is None:
        return None

    # Require rational coefficients and nonzero exponential part
    if not all(isinstance(v, sp.Rational) for v in [a_val, b_val, c_val]):
        return None
    if a_val == 0:
        return None

    expr = a_val * sp.Integer(2) ** symbol + b_val * symbol + c_val

    # Verify on ALL training points
    for n_val, expected in pts:
        if sp.simplify(expr.subs(symbol, n_val) - expected) != 0:
            return None

    return expr


def _verify_holdout(
    expr: sp.Expr,
    holdout: list[tuple[int, sp.Expr]],
    symbol: sp.Symbol,
) -> bool:
    """Verify *expr* matches all holdout points exactly."""
    if not holdout:
        return True
    for n_val, expected in holdout:
        if sp.simplify(expr.subs(symbol, n_val) - expected) != 0:
            return False
    return True
