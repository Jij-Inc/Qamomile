"""Provide exact symbolic loop algebra for resource estimation."""

from __future__ import annotations

import sympy as sp


def symbolic_iterations(
    start: sp.Expr,
    stop: sp.Expr,
    step: sp.Expr,
) -> sp.Expr:
    """Compute the exact iteration count of a symbolic Python range.

    Args:
        start (sp.Expr): Inclusive symbolic range start.
        stop (sp.Expr): Exclusive symbolic range stop.
        step (sp.Expr): Nonzero symbolic range step.

    Returns:
        sp.Expr: ``Max(0, ceiling((stop - start) / step))``.

    Raises:
        ValueError: If ``step`` is concretely zero.
    """
    if step.is_zero:
        raise ValueError("Loop step cannot be zero")
    return sp.Max(0, sp.ceiling((stop - start) / step))
