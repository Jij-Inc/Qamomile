"""Recognize affine resource expressions without symbolic expansion."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import cast

import sympy as sp

from qamomile.circuit.estimator._resource_base import ResourceExpr

_DOMAIN_AFFINE_NODE_LIMIT = 512


@dataclasses.dataclass(frozen=True)
class _AffineDomainExpression:
    """Store one affine expression as a constant and scalar coefficients.

    Args:
        constant (sp.Expr): Symbol-free constant term.
        coefficients (tuple[tuple[sp.Symbol, sp.Expr], ...]): Deterministically
            ordered symbol coefficients. Every coefficient is symbol-free.
    """

    constant: sp.Expr
    coefficients: tuple[tuple[sp.Symbol, sp.Expr], ...]

    def as_expression(self) -> ResourceExpr:
        """Reconstruct the affine expression without distributing products.

        Returns:
            ResourceExpr: Exact expression represented by this record.
        """
        terms = (
            self.constant,
            *(coefficient * symbol for symbol, coefficient in self.coefficients),
        )
        return cast(ResourceExpr, sp.Add(*terms))


def _extract_affine_domain_expression(
    expression: sp.Expr,
    *,
    node_limit: int | None = None,
) -> _AffineDomainExpression | None:
    """Extract an affine form through a bounded structural traversal.

    The extractor never calls polynomial conversion or symbolic expansion.
    Multiplication is accepted only when at most one factor contains public
    symbols, and symbolic powers are accepted only for exponents zero or one.

    Args:
        expression (sp.Expr): Candidate arithmetic expression.
        node_limit (int | None): Maximum number of syntax nodes to inspect.
            Defaults to the current module limit.

    Returns:
        _AffineDomainExpression | None: Exact affine form, or ``None`` when the
        expression is nonlinear, unsupported, or exceeds the node budget.
    """
    active_limit = _DOMAIN_AFFINE_NODE_LIMIT if node_limit is None else node_limit
    if active_limit < 1 or not isinstance(expression, sp.Expr):
        return None
    for index, _node in enumerate(sp.preorder_traversal(expression), start=1):
        if index > active_limit:
            return None
    return _extract_affine_node(expression)


def _extract_affine_node(expression: sp.Expr) -> _AffineDomainExpression | None:
    """Extract one already-budgeted affine syntax node recursively.

    Args:
        expression (sp.Expr): Current arithmetic syntax node.

    Returns:
        _AffineDomainExpression | None: Exact affine form, or ``None`` when the
        current structure cannot be proven affine.
    """
    if not expression.free_symbols:
        return _AffineDomainExpression(expression, ())
    if isinstance(expression, sp.Dummy):
        return None
    if isinstance(expression, sp.Symbol):
        return _AffineDomainExpression(sp.S.Zero, ((expression, sp.S.One),))
    if isinstance(expression, sp.Add):
        parts = tuple(
            _extract_affine_node(cast(sp.Expr, argument))
            for argument in expression.args
        )
        if any(part is None for part in parts):
            return None
        return _add_affine_forms(cast(tuple[_AffineDomainExpression, ...], parts))
    if isinstance(expression, sp.Mul):
        constant = sp.S.One
        symbolic: _AffineDomainExpression | None = None
        for argument in expression.args:
            part = _extract_affine_node(cast(sp.Expr, argument))
            if part is None:
                return None
            if not part.coefficients:
                constant *= part.constant
                continue
            if symbolic is not None:
                return None
            symbolic = part
        if symbolic is None:
            return _AffineDomainExpression(constant, ())
        return _scale_affine_form(symbolic, constant)
    if isinstance(expression, sp.Pow):
        base = cast(sp.Expr, expression.base)
        exponent = cast(sp.Expr, expression.exp)
        if exponent.free_symbols:
            return None
        if exponent == 0:
            return _AffineDomainExpression(sp.S.One, ())
        if exponent == 1:
            return _extract_affine_node(base)
        return None
    return None


def _add_affine_forms(
    forms: Iterable[_AffineDomainExpression],
) -> _AffineDomainExpression:
    """Add exact affine forms without expanding any symbolic products.

    Args:
        forms (Iterable[_AffineDomainExpression]): Forms to combine.

    Returns:
        _AffineDomainExpression: Exact combined affine form.
    """
    constant = sp.S.Zero
    coefficients: dict[sp.Symbol, sp.Expr] = {}
    for form in forms:
        constant += form.constant
        for symbol, coefficient in form.coefficients:
            coefficients[symbol] = coefficients.get(symbol, sp.S.Zero) + coefficient
    return _AffineDomainExpression(
        constant,
        tuple(
            (symbol, coefficient)
            for symbol, coefficient in sorted(
                coefficients.items(),
                key=lambda item: sp.default_sort_key(item[0]),
            )
            if not _constant_is_zero(coefficient)
        ),
    )


def _scale_affine_form(
    form: _AffineDomainExpression,
    factor: sp.Expr,
) -> _AffineDomainExpression:
    """Multiply an affine form by one symbol-free factor.

    Args:
        form (_AffineDomainExpression): Form to scale.
        factor (sp.Expr): Symbol-free scalar factor.

    Returns:
        _AffineDomainExpression: Exact scaled form.
    """
    return _AffineDomainExpression(
        factor * form.constant,
        tuple(
            (symbol, factor * coefficient)
            for symbol, coefficient in form.coefficients
            if not _constant_is_zero(factor * coefficient)
        ),
    )


def _constant_is_zero(expression: sp.Expr) -> bool:
    """Return whether a symbol-free coefficient is provably zero.

    Args:
        expression (sp.Expr): Symbol-free arithmetic expression.

    Returns:
        bool: Whether exact local facts prove zero.
    """
    return expression == 0 or expression.is_zero is True


__all__: list[str] = []
