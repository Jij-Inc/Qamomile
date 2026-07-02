"""Share SymPy conversion and validation helpers across resource estimation.

Every module in ``qamomile.resource_estimation`` converts user-facing numeric
or symbolic inputs to SymPy expressions and validates their signs the same
way. Keeping the helpers in one package-private module guarantees identical
error behavior across all estimator entry points.
"""

from __future__ import annotations

import sympy as sp

_SympyLike = sp.Expr | int | float
_CoefficientLike = _SympyLike | complex


def _as_expr(value: _CoefficientLike, name: str) -> sp.Expr:
    """Convert a numeric or symbolic value to a SymPy expression.

    Args:
        value (sp.Expr | int | float | complex): Value to convert.
        name (str): Field name used in error messages.

    Returns:
        sp.Expr: Converted SymPy expression.

    Raises:
        TypeError: If ``value`` cannot be sympified.
    """
    try:
        return sp.sympify(value)
    except (TypeError, sp.SympifyError) as exc:
        raise TypeError(f"{name} must be a numeric or SymPy expression.") from exc


def _validate_positive(expr: sp.Expr, name: str) -> None:
    """Validate that an expression is positive when decidable.

    Args:
        expr (sp.Expr): Expression to validate.
        name (str): Field name used in error messages.

    Raises:
        ValueError: If SymPy can prove that ``expr`` is not positive.
    """
    if expr.is_positive is False:
        raise ValueError(f"{name} must be positive.")


def _validate_nonnegative(expr: sp.Expr, name: str) -> None:
    """Validate that an expression is nonnegative when decidable.

    Args:
        expr (sp.Expr): Expression to validate.
        name (str): Field name used in error messages.

    Raises:
        ValueError: If SymPy can prove that ``expr`` is negative.
    """
    if expr.is_nonnegative is False:
        raise ValueError(f"{name} must be nonnegative.")
