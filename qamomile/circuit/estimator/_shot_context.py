"""Keep device context values separate from measurement readiness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import sympy as sp


@dataclass(slots=True)
class _ShotContext:
    """Own one repetition's private shot identity and bounded expressions.

    These symbols describe execution values, not public inputs or scheduler
    dependency tokens. One instance belongs to one interpretation run.

    Args:
        current (sp.Symbol): Private nonnegative integer context value. A
            fresh identity is created by default.
        derived (set[sp.Symbol]): Context-derived private values. Defaults to
            an empty set.
        remainders (dict[sp.Expr, sp.Symbol]): Run-local memoized remainder
            identities. Defaults to an empty mapping.
        used (bool): Whether this run has encountered a context read.
            Defaults to false.
        used_uint_arithmetic (bool): Whether this run evaluated context-derived
            UInt arithmetic under mathematical integer semantics. Defaults to
            false.
    """

    current: sp.Symbol = field(
        default_factory=lambda: sp.Dummy("current_shot", integer=True, nonnegative=True)
    )
    derived: set[sp.Symbol] = field(default_factory=set)
    remainders: dict[sp.Expr, sp.Symbol] = field(default_factory=dict)
    used: bool = False
    used_uint_arithmetic: bool = False

    def contains(self, expression: sp.Basic) -> bool:
        """Return whether an expression uses this repetition's context.

        Args:
            expression (sp.Basic): Resolved expression to inspect.

        Returns:
            bool: Whether a private context symbol remains.
        """
        return bool(expression.free_symbols & (self.derived | {self.current}))

    def normalize_remainders(
        self,
        expression: sp.Expr,
        domains: dict[sp.Symbol, tuple[sp.Expr, ...] | None],
        *,
        limit: int,
    ) -> sp.Expr:
        """Replace bounded context remainders by finite private alternatives.

        Args:
            expression (sp.Expr): Resolved scalar expression.
            domains (dict[sp.Symbol, tuple[sp.Expr, ...] | None]): Run-owned
                finite runtime domains to extend.
            limit (int): Maximum number of alternatives per remainder.

        Returns:
            sp.Expr: Expression with supported remainders made finite.
        """
        replacements: dict[sp.Expr, sp.Symbol] = {}
        for remainder in expression.atoms(sp.Mod):
            dividend, divisor = remainder.args
            if (
                not self.contains(dividend)
                or not isinstance(divisor, sp.Integer)
                or not 0 < divisor <= limit
                or dividend.is_integer is not True
            ):
                continue
            symbol = self.remainders.get(remainder)
            if symbol is None:
                symbol = sp.Dummy("shot_remainder", integer=True, nonnegative=True)
                self.remainders[remainder] = symbol
                self.derived.add(symbol)
                domains[symbol] = tuple(sp.Integer(i) for i in range(int(divisor)))
            replacements[remainder] = symbol
        return cast(sp.Expr, expression.xreplace(replacements))
