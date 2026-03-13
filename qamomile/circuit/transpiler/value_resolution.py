"""Shared helpers for resolving classical values from runtime/emit bindings."""

from __future__ import annotations

import numbers
import re
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


_DIMENSION_NAME_RE = re.compile(r"^(.+)_dim(\d+)$")


def is_concrete_real_number(value: Any) -> bool:
    """Return True for concrete builtin/numpy real scalars."""

    return isinstance(value, numbers.Real)


def _lookup_bound_array(
    array_name: str,
    parameter_name: str | None,
    bindings: BindingLookup,
) -> Any | None:
    """Look up bound array-like data by array name or parameter name."""
    if bindings.has(array_name):
        return bindings.get(array_name)
    if parameter_name is not None and bindings.has(parameter_name):
        return bindings.get(parameter_name)
    return None


def _resolve_bound_array_dimension(array_data: Any, dim_index: int) -> int | None:
    """Resolve one dimension of a bound array-like object."""
    if hasattr(array_data, "shape"):
        shape = array_data.shape
        if dim_index < len(shape):
            dim_value = shape[dim_index]
            if is_concrete_real_number(dim_value):
                return int(dim_value)

    if dim_index == 0 and hasattr(array_data, "__len__"):
        return len(array_data)

    return None


def resolve_array_dimension_value(
    value: "Value",
    bindings: BindingLookup,
) -> int | None:
    """Resolve synthetic array-dimension values like ``hi_dim0``.

    These values can appear either as:
    - shape-derived values that still point at ``parent_array`` before inlining
    - renamed symbolic values such as ``hi_dim0`` after inlining/substitution
    """
    # Pre-inline shape access: keep following the parent array when available.
    if value.parent_array is not None and not value.element_indices:
        match = _DIMENSION_NAME_RE.match(value.name)
        dim_index = int(match.group(2)) if match else 0
        array_data = _lookup_bound_array(
            value.parent_array.name,
            value.parent_array.parameter_name(),
            bindings,
        )
        if array_data is not None:
            resolved = _resolve_bound_array_dimension(array_data, dim_index)
            if resolved is not None:
                return resolved

    # Post-inline shape access: values become standalone names like ``hi_dim0``.
    match = _DIMENSION_NAME_RE.match(value.name)
    if match is None:
        return None

    array_name = match.group(1)
    dim_index = int(match.group(2))
    array_data = _lookup_bound_array(array_name, value.parameter_name(), bindings)
    if array_data is None:
        return None

    return _resolve_bound_array_dimension(array_data, dim_index)


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
        4. synthetic array dimension value (e.g. hi_dim0)
        5. array element via parent array + element indices
        6. constant value
    """

    if bindings.has(value.uuid):
        return bindings.get(value.uuid)

    if allow_parameter_name:
        param_name = value.parameter_name()
        if param_name is not None and bindings.has(param_name):
            return bindings.get(param_name)

    if bindings.has(value.name):
        return bindings.get(value.name)

    resolved_dim = resolve_array_dimension_value(value, bindings)
    if resolved_dim is not None:
        return resolved_dim

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

        resolved_dim = resolve_array_dimension_value(value, bindings)
        if resolved_dim is not None:
            return resolved_dim

        if value.parent_array is not None and value.element_indices:
            resolved = resolve_array_element_value(value, bindings)
            if is_concrete_real_number(resolved):
                return int(resolved)
            return None

    return None
