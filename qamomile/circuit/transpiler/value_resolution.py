"""Shared helpers for resolving classical values from runtime/emit bindings."""

from __future__ import annotations

import numbers
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from qamomile.circuit.ir.value import Value


class BindingLookup:
    """Lookup helper that searches multiple binding sources in priority order."""

    def __init__(self, *sources: Mapping[str, Any] | None):
        self._sources = tuple(source for source in sources if source is not None)

    def has(self, key: str) -> bool:
        return any(key in source for source in self._sources)

    def get(self, key: str) -> Any:
        for source in self._sources:
            if key in source:
                return source[key]
        raise KeyError(key)


def is_concrete_real_number(value: Any) -> bool:
    """Return True for concrete builtin/numpy real scalars."""

    return isinstance(value, numbers.Real)


def resolve_classical_value(
    value: "Value",
    bindings: BindingLookup,
    *,
    allow_parameter_name: bool = False,
) -> Any | None:
    """Resolve a classical value from bindings.

    Lookup order:
        1. uuid
        2. parameter name (optional)
        3. value name
        4. array element via parent array + element indices
        5. constant value
    """

    if bindings.has(value.uuid):
        return bindings.get(value.uuid)

    if allow_parameter_name:
        param_name = value.parameter_name()
        if param_name is not None and bindings.has(param_name):
            return bindings.get(param_name)

    if bindings.has(value.name):
        return bindings.get(value.name)

    if value.parent_array is not None and value.element_indices:
        return resolve_array_element_value(value, bindings)

    if value.is_constant():
        return value.get_const()

    if value.params and "const" in value.params:
        return value.params["const"]

    return None


def resolve_array_element_value(
    value: "Value",
    bindings: BindingLookup,
) -> int | float | None:
    """Resolve an array element using concrete indices from bindings/results."""

    if value.parent_array is None or not value.element_indices:
        return None

    array_key = value.parent_array.name
    if not bindings.has(array_key):
        array_param_name = value.parent_array.parameter_name()
        if array_param_name is None or not bindings.has(array_param_name):
            return None
        array_key = array_param_name

    array_data = bindings.get(array_key)

    indices = []
    for idx in value.element_indices:
        idx_val = resolve_int_value(idx, bindings)
        if idx_val is None:
            return None
        indices.append(idx_val)

    try:
        result = array_data
        for idx in indices:
            result = result[idx]
        if is_concrete_real_number(result):
            return result
        return None
    except (IndexError, TypeError, KeyError):
        return None


def resolve_int_value(
    value: Any,
    bindings: BindingLookup,
) -> int | None:
    """Resolve a concrete integer from bindings without silent fallback."""

    from qamomile.circuit.ir.value import Value

    if is_concrete_real_number(value):
        return int(value)

    if isinstance(value, Value):
        if value.is_constant():
            return int(value.get_const())

        if value.is_parameter():
            param_name = value.parameter_name()
            if param_name and bindings.has(param_name):
                bound_val = bindings.get(param_name)
                if is_concrete_real_number(bound_val):
                    return int(bound_val)
                return None

        if bindings.has(value.uuid):
            bound_val = bindings.get(value.uuid)
            if is_concrete_real_number(bound_val):
                return int(bound_val)
            return None

        if bindings.has(value.name):
            bound_val = bindings.get(value.name)
            if is_concrete_real_number(bound_val):
                return int(bound_val)
            return None

        if value.parent_array is not None and value.element_indices:
            resolved = resolve_array_element_value(value, bindings)
            if is_concrete_real_number(resolved):
                return int(resolved)
            return None

    return None
