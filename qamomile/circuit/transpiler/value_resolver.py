"""Unified value resolution for transpiler passes.

Several transpiler passes need to resolve an IR ``Value`` to a concrete
Python value (int, float, bool).  Each pass previously carried its own
copy of the resolution logic with slightly different lookup orders and
naming conventions.

``ValueResolver`` consolidates those into a single class with a
deterministic resolution order:

1. **Context map** — caller-supplied ``UUID → concrete value`` dict
   (e.g. ``folded_values``, ``concrete_values``).
2. **Constant** — ``value.is_constant() → value.get_const()``.
3. **Bindings by parameter name** — ``value.is_parameter() → bindings[param_name]``.
4. **Bindings by value name** — ``value.name → bindings[name]``.
5. Returns ``None`` if none of the above match.
"""

from __future__ import annotations

from typing import Any


class ValueResolver:
    """Resolves IR Values to concrete Python values.

    Parameters
    ----------
    context:
        UUID-keyed map of already-resolved values.  The *values* may be
        either raw Python scalars or ``Value`` objects; if a ``Value`` is
        found its ``get_const()`` is extracted automatically.
    bindings:
        Name-keyed parameter bindings supplied by the user at transpile
        time.
    """

    __slots__ = ("_context", "_bindings")

    def __init__(
        self,
        context: dict[str, Any] | None = None,
        bindings: dict[str, Any] | None = None,
    ):
        self._context = context or {}
        self._bindings = bindings or {}

    def resolve(self, value: Any) -> Any | None:
        """Resolve *value* to a concrete Python value, or ``None``.

        If *value* is not a Value-like object (no ``uuid`` attribute) it
        is returned as-is — the caller already has a concrete value.
        """
        if not hasattr(value, "uuid"):
            return value  # Already concrete

        # 1. Context map (folded values, concrete values)
        if value.uuid in self._context:
            ctx_val = self._context[value.uuid]
            # The context may store Value objects (constant_fold) or raw
            # scalars (compile_time_if_lowering).  Unwrap Value → const.
            if hasattr(ctx_val, "get_const"):
                return ctx_val.get_const()
            return ctx_val

        # 2. Constant
        if hasattr(value, "is_constant") and value.is_constant():
            return value.get_const()

        # 3. Bindings by parameter name
        if hasattr(value, "is_parameter") and value.is_parameter():
            param_name = value.parameter_name()
            if param_name and param_name in self._bindings:
                return self._bindings[param_name]

        # 4. Bindings by value name
        if hasattr(value, "name") and value.name and value.name in self._bindings:
            return self._bindings[value.name]

        return None
