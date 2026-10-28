"""Serialize symbolic resource expressions for user-facing payloads."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import sympy as sp


class SymbolRegistry:
    """Assign deterministic, identity-preserving public symbol names.

    SymPy permits distinct ``Symbol`` and ``Dummy`` objects to share the same
    printed name. Public resource payloads cannot use that spelling for both
    identities without changing expressions such as ``Ne(left, right)`` into
    ``False``. The registry therefore retains the first spelling and assigns
    ``__2``, ``__3``, and so on to later same-name identities.

    Args:
        symbols (Iterable[sp.Symbol]): Symbols in deterministic encounter
            order. Equal SymPy symbols are treated as one identity.
        preferred_names (Mapping[sp.Symbol, str] | None): Previously published
            aliases to retain for symbols that remain present. Defaults to
            ``None``.
    """

    def __init__(
        self,
        symbols: Iterable[sp.Symbol],
        preferred_names: Mapping[sp.Symbol, str] | None = None,
    ):
        """Build a registry from symbols in deterministic encounter order.

        Args:
            symbols (Iterable[sp.Symbol]): Symbols to assign public names.
            preferred_names (Mapping[sp.Symbol, str] | None): Existing aliases
                to preserve when possible. Defaults to ``None``.
        """
        ordered: list[sp.Symbol] = []
        seen: set[sp.Symbol] = set()
        for symbol in symbols:
            if symbol in seen:
                continue
            seen.add(symbol)
            ordered.append(symbol)

        preferred = preferred_names or {}
        protected = {preferred[symbol] for symbol in ordered if symbol in preferred}
        reserved = {_symbol_base_name(symbol) for symbol in ordered}
        used: set[str] = set()
        names: dict[sp.Symbol, str] = {}
        for symbol in ordered:
            base = _symbol_base_name(symbol)
            previous = preferred.get(symbol)
            if previous is not None and previous not in used:
                public_name = previous
            elif base not in used and base not in protected:
                public_name = base
            else:
                suffix = 2
                public_name = f"{base}__{suffix}"
                while (
                    public_name in used
                    or public_name in reserved
                    or public_name in protected
                ):
                    suffix += 1
                    public_name = f"{base}__{suffix}"
            used.add(public_name)
            names[symbol] = public_name
        self._names = names

    @classmethod
    def from_expressions(
        cls,
        expressions: Iterable[sp.Basic | int | float],
        preferred_names: Mapping[sp.Symbol, str] | None = None,
    ) -> SymbolRegistry:
        """Build a registry in symbolic-expression traversal order.

        Args:
            expressions (Iterable[sp.Basic | int | float]): Expressions whose
                symbols share one public namespace.
            preferred_names (Mapping[sp.Symbol, str] | None): Previously
                published aliases to preserve. Defaults to ``None``.

        Returns:
            SymbolRegistry: Registry covering every encountered symbol.
        """
        return cls(_symbols_in_expressions(expressions), preferred_names)

    def aliases(self) -> dict[sp.Symbol, str]:
        """Return a copy of every identity-to-public-name assignment.

        Returns:
            dict[sp.Symbol, str]: Registry assignments in encounter order.
        """
        return dict(self._names)

    def name(self, symbol: sp.Symbol) -> str:
        """Return the unique public name assigned to one symbol.

        Args:
            symbol (sp.Symbol): Registered SymPy symbol.

        Returns:
            str: Stable public name for the symbol identity.

        Raises:
            KeyError: If the symbol was not included when the registry was
                created.
        """
        try:
            return self._names[symbol]
        except KeyError as exc:
            raise KeyError(
                f"Symbol {symbol!r} is not present in this serialization registry."
            ) from exc

    def normalize(self, expression: sp.Basic | int | float) -> sp.Basic:
        """Replace registered symbols with uniquely named public symbols.

        Args:
            expression (sp.Basic | int | float): Symbolic or numeric resource
                expression to normalize.

        Returns:
            sp.Basic: Semantically equivalent expression using ordinary public
                symbols in place of identity-only dummies.

        Raises:
            KeyError: If the expression contains a symbol absent from this
                registry.
        """
        sympy_expression = sp.sympify(expression)
        replacements = {
            symbol: sp.Symbol(self.name(symbol), **symbol.assumptions0)
            for symbol in sympy_expression.atoms(sp.Symbol)
        }
        return sympy_expression.xreplace(replacements)

    def stringify(self, expression: sp.Basic | int | float) -> str:
        """Render an expression using this registry's public names.

        Args:
            expression (sp.Basic | int | float): Symbolic or numeric resource
                expression to render.

        Returns:
            str: SymPy-compatible expression text with unique symbol names.

        Raises:
            KeyError: If the expression contains a symbol absent from this
                registry.
        """
        return str(self.normalize(expression))


def _symbol_base_name(symbol: sp.Symbol) -> str:
    """Return a symbol's declared name without SymPy's dummy prefix.

    Args:
        symbol (sp.Symbol): Symbol to name.

    Returns:
        str: Declared symbol name.
    """
    return symbol.name


def _symbols_in_expressions(
    expressions: Iterable[sp.Basic | int | float],
) -> Iterable[sp.Symbol]:
    """Yield free symbols before bound symbols in deterministic order.

    Public resource parameters are free symbols. Giving them priority keeps a
    bound ``Sum`` index from claiming the natural spelling of a qkernel input
    that appears later in the summation bounds.

    Args:
        expressions (Iterable[sp.Basic | int | float]): Expressions to walk.

    Returns:
        Iterable[sp.Symbol]: Lazily generated symbols with free identities
            first, followed by every bound identity.
    """
    normalized = tuple(sp.sympify(expression) for expression in expressions)
    free_symbols = set().union(
        *(expression.free_symbols for expression in normalized),
    )
    for expression in normalized:
        for node in sp.preorder_traversal(expression):
            if isinstance(node, sp.Symbol) and node in free_symbols:
                yield node
    for expression in normalized:
        for node in sp.preorder_traversal(sp.sympify(expression)):
            if isinstance(node, sp.Symbol):
                yield node


def normalize_expression(
    expression: sp.Basic | int | float,
    registry: SymbolRegistry | None = None,
) -> sp.Basic:
    """Replace identity-only SymPy dummies with user-facing symbols.

    SymPy prints a ``Dummy("index")`` as ``_index`` even though its declared
    name is ``index``. Resource-estimation parameters and quantified ranges
    use the declared name. Same-name identities receive unique suffixed names
    so normalization cannot collapse a condition or combine two parameters.

    Args:
        expression (sp.Basic | int | float): Symbolic or numeric resource
            expression to normalize.
        registry (SymbolRegistry | None): Shared identity registry for a larger
            payload. Defaults to a registry local to ``expression``.

    Returns:
        sp.Basic: Equivalent expression containing ordinary same-name symbols
        in place of dummies.
    """
    active_registry = registry or SymbolRegistry.from_expressions((expression,))
    return active_registry.normalize(expression)


def stringify_expression(
    expression: sp.Basic | int | float,
    registry: SymbolRegistry | None = None,
) -> str:
    """Render a symbolic expression with stable external symbol names.

    Args:
        expression (sp.Basic | int | float): Symbolic or numeric resource
            expression to render.
        registry (SymbolRegistry | None): Shared identity registry for a larger
            payload. Defaults to a registry local to ``expression``.

    Returns:
        str: SymPy-compatible expression text without dummy-printing prefixes.
    """
    active_registry = registry or SymbolRegistry.from_expressions((expression,))
    return active_registry.stringify(expression)
