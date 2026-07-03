"""Share SymPy conversion and validation helpers across resource estimation.

Every module in ``qamomile.resource_estimation`` converts user-facing numeric
or symbolic inputs to SymPy expressions and validates their signs the same
way. Keeping the helpers in one package-private module guarantees identical
error behavior across all estimator entry points.
"""

from __future__ import annotations

import sympy as sp

SympyLike = sp.Expr | int | float
_SympyLike = SympyLike
_CoefficientLike = SympyLike | complex


def as_expr(value: _CoefficientLike, name: str) -> sp.Expr:
    """Convert a numeric or symbolic value to a SymPy expression.

    This is the public coercion helper algorithm packages should use when
    implementing workload hooks such as ``_own_resource_values``.

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


_as_expr = as_expr


def _convert_fields(instance: object, names: tuple[str, ...]) -> None:
    """Sympify the named dataclass fields in place on a frozen instance.

    Call from ``__post_init__`` so numeric user inputs are converted to
    SymPy expressions exactly once at construction instead of on every
    property access.

    Args:
        instance (object): Frozen dataclass instance under construction.
        names (tuple[str, ...]): Field names to convert.

    Raises:
        TypeError: If a named field cannot be sympified.
    """
    for name in names:
        object.__setattr__(instance, name, _as_expr(getattr(instance, name), name))


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
