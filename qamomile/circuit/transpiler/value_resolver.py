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
3. **Compile-time array element** — ``arr[i]`` or ``arr[a:b][i]``
   resolves from the root array's ``const_array`` metadata or binding.
4. **Bindings by parameter name** — ``value.is_parameter() → bindings[param_name]``.
5. **Bindings by value name** — ``value.name → bindings[name]``.
6. Returns ``None`` if none of the above match.
"""

from __future__ import annotations

import numbers
from typing import Any

import numpy as np

from qamomile.circuit.ir.value import ArrayValue, Value


class ValueResolver:
    """Resolve IR Values to concrete Python values.

    Args:
        context (dict[str, Any] | None): UUID-keyed map of already
            resolved values. The values may be either raw Python scalars
            or ``Value`` objects; if a ``Value`` is found its
            ``get_const()`` is extracted automatically.
        bindings (dict[str, Any] | None): Name-keyed parameter
            bindings supplied by the user at transpile time.
    """

    __slots__ = ("_context", "_bindings")

    def __init__(
        self,
        context: dict[str, Any] | None = None,
        bindings: dict[str, Any] | None = None,
    ):
        """Create a resolver with optional context and bindings.

        Args:
            context (dict[str, Any] | None): UUID-keyed map of
                already-resolved values. Defaults to None.
            bindings (dict[str, Any] | None): Name-keyed parameter
                bindings supplied by the user. Defaults to None.
        """
        self._context = context or {}
        self._bindings = bindings or {}

    def resolve(self, value: Any) -> Any | None:
        """Resolve a Value-like object to a concrete Python value.

        If *value* is not a Value-like object (no ``uuid`` attribute) it
        is returned as-is — the caller already has a concrete value.

        Args:
            value (Any): The Value-like object or already concrete value
                to resolve.

        Returns:
            Any | None: The resolved concrete value, the original
                concrete value for non-Value inputs, or ``None`` when no
                resolution rule applies.
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

        # 3. Compile-time array element
        array_element = self._resolve_array_element(value)
        if array_element is not None:
            return array_element

        # 4. Bindings by parameter name
        if hasattr(value, "is_parameter") and value.is_parameter():
            param_name = value.parameter_name()
            if param_name and param_name in self._bindings:
                return self._bindings[param_name]

        # 5. Bindings by value name
        if hasattr(value, "name") and value.name and value.name in self._bindings:
            return self._bindings[value.name]

        return None

    def _resolve_array_element(self, value: Any) -> Any | None:
        """Resolve an array-element Value from compile-time array data.

        Args:
            value (Any): The Value-like object to inspect. It may be an
                element Value with ``parent_array`` and
                ``element_indices`` metadata.

        Returns:
            Any | None: The indexed element when every index resolves and
                the parent array has compile-time data. Returns ``None``
                when the value is not an array element, the parent has no
                compile-time data, an index is symbolic or resolves to a
                negative value (Python-style wrapping is refused), or
                indexing fails.
        """
        parent_array = getattr(value, "parent_array", None)
        element_indices = getattr(value, "element_indices", None)
        # Only array-element Values have both an ArrayValue parent and
        # element indices. Plain scalar Values should fall through to the
        # later parameter/name lookup rules instead of being treated as
        # failed array accesses.
        if not isinstance(parent_array, ArrayValue) or not element_indices:
            return None

        root_array, element_indices = self._resolve_root_array_indices(
            parent_array, element_indices
        )
        if root_array is None:
            return None

        # Prefer const_array metadata because bound Vector inputs created
        # by the frontend keep their compile-time payload on the root
        # ArrayValue itself. Slice views intentionally walk back to the
        # root before this lookup so view-local indices address the same
        # compile-time container the user originally bound.
        container = root_array.get_const_array()
        if container is None:
            # Fall back to explicit bindings for callers that resolve an
            # element Value against a separate binding map instead of an
            # ArrayValue carrying const_array metadata.
            parent_name = getattr(root_array, "name", None)
            parent_uuid = getattr(root_array, "uuid", None)
            if parent_name in self._bindings:
                container = self._bindings[parent_name]
            elif parent_uuid in self._bindings:
                container = self._bindings[parent_uuid]
        if container is None:
            # No compile-time source exists, so the element must remain
            # symbolic. Returning None lets the caller preserve that state
            # rather than inventing a placeholder value.
            return None

        for index_value in element_indices:
            # Every index must be concrete before we can safely index the
            # Python container. Symbolic indices intentionally stop
            # resolution here to avoid silently selecting the wrong element.
            index = self.resolve(index_value)
            if index is None:
                return None
            try:
                concrete_index = int(index)
            except (TypeError, ValueError):
                return None
            if concrete_index < 0:
                # Negative indices must not reach Python container
                # indexing, where they would silently wrap to the wrong
                # element. Python-style negative indexing is unsupported.
                return None
            try:
                container = container[concrete_index]
            except (IndexError, KeyError, TypeError):
                # Shape/type mismatches mean this Value is not resolvable
                # from the available compile-time data. Unlike the
                # emit-time resolver (which raises ``EmitError`` on a
                # concrete out-of-range index into compile-time data),
                # this resolver stays permissive on purpose: it feeds
                # constant folding and compile-time-if lowering, where
                # "unresolved" legitimately means "keep symbolic" —
                # folding visits branches that compile-time-if lowering
                # may later discard, so a hard error here could reject
                # dead code. Out-of-range accesses that survive to emit
                # are rejected there, before parameter creation.
                return None
        return _normalize_bound_scalar(container)

    def _resolve_root_array_indices(
        self,
        parent_array: ArrayValue,
        element_indices: tuple[Any, ...],
    ) -> tuple[ArrayValue | None, tuple[Any, ...]]:
        """Map view-local element indices onto the root array.

        Args:
            parent_array (ArrayValue): Array that directly owns the
                element Value. It may be a sliced ``ArrayValue`` backing
                a ``VectorView``.
            element_indices (tuple[Any, ...]): Element indices local to
                ``parent_array``.

        Returns:
            tuple[ArrayValue | None, tuple[Any, ...]]: The root array and
                root-local indices when every slice bound can be resolved,
                or ``(None, ())`` when the slice chain is symbolic, a
                resolved index is negative, a resolved bound violates the
                frontend contract (non-negative start, positive step), or
                the chain is otherwise not resolvable at compile time.
        """
        root_array = parent_array
        root_indices = element_indices

        # VectorView is represented as a sliced ArrayValue, not a distinct
        # Value type. Compose each slice frame into the leading element
        # index so view[i] resolves against the root container as
        # root[start + step * i].
        while root_array.slice_of is not None:
            if not root_indices:
                return None, ()

            start = self.resolve(root_array.slice_start)
            step = self.resolve(root_array.slice_step)
            index = self.resolve(root_indices[0])
            if start is None or step is None or index is None:
                # Symbolic view bounds or symbolic indices must stay
                # unresolved; guessing a root slot would silently pick
                # the wrong compile-time value.
                return None, ()
            try:
                start_int = int(start)
                step_int = int(step)
                index_int = int(index)
            except (TypeError, ValueError):
                return None, ()
            if index_int < 0 or start_int < 0 or step_int <= 0:
                # A negative local index would compose to a wrong root
                # slot (e.g. ``view[-1]`` for ``view = a[1:3]`` maps to
                # ``a[0]`` instead of ``a[2]``), and resolved bounds must
                # satisfy the frontend contract (non-negative start,
                # positive step). Stay unresolved instead of guessing.
                return None, ()
            root_index = start_int + step_int * index_int

            root_indices = (
                _constant_index_like(root_indices[0], root_index),
                *root_indices[1:],
            )
            root_array = root_array.slice_of

        return root_array, root_indices


def _normalize_bound_scalar(value: Any) -> Any:
    """Normalize NumPy-style scalar values to Python primitives.

    Args:
        value (Any): The value resolved from a compile-time container.

    Returns:
        Any: A Python ``bool``, ``int``, or ``float`` for numeric scalar
            inputs, or the original value for non-numeric objects.
    """
    # bool is an Integral subclass, so preserve it before the integer
    # branch. This keeps Bit values from becoming 0/1 integers.
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, np.generic):
        # NumPy scalar arrays return np.generic instances on indexing.
        # Convert only scalar NumPy values; arbitrary object-array
        # payloads should pass through unchanged unless item() itself
        # unwraps a NumPy scalar into a Python primitive.
        item = value.item()
        if isinstance(item, bool):
            return item
        if isinstance(item, numbers.Integral):
            return int(item)
        if isinstance(item, numbers.Real):
            return float(item)
        return item
    return value


def _constant_index_like(index_value: Any, index: int) -> Any:
    """Create a constant index Value when possible.

    Args:
        index_value (Any): Original index object from an element access.
            Value-like inputs provide the UInt type needed to keep the
            rewritten index compatible with later ``resolve()`` calls.
        index (int): Concrete root-array index computed from a slice
            affine map.

    Returns:
        Any: A constant Value with the original type when ``index_value``
            is Value-like, otherwise the raw Python integer.
    """
    if hasattr(index_value, "type"):
        return Value(
            type=index_value.type,
            name=getattr(index_value, "name", "idx"),
        ).with_const(index)
    return index
