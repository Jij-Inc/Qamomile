"""Build input helpers for ``QKernel`` tracing."""

from __future__ import annotations

import inspect
import numbers
from typing import Any

import numpy as np

from qamomile.circuit.frontend.func_to_block import (
    create_dummy_input,
    is_array_type,
    is_dict_type,
    is_tuple_type,
)
from qamomile.circuit.frontend.handle import Observable
from qamomile.circuit.frontend.handle.array import Vector
from qamomile.circuit.frontend.handle.containers import Dict
from qamomile.circuit.frontend.handle.primitives import Bit, Float, Handle, Qubit, UInt
from qamomile.circuit.frontend.qkernel_utils import get_array_element_type
from qamomile.circuit.frontend.static_binding import (
    is_static_binding_annotation,
    validate_static_binding,
)
from qamomile.circuit.ir.types import BitType, FloatType, ObservableType, UIntType
from qamomile.circuit.ir.value import ArrayValue, DictValue, Value


def is_parameterizable_type(param_type: Any) -> bool:
    """Return whether an annotation can stay as a runtime parameter.

    Args:
        param_type (Any): Frontend type annotation to inspect.

    Returns:
        bool: ``True`` when the type can be represented by backend runtime
        parameters.
    """
    if param_type in (float, Float, int, UInt):
        return True
    if is_array_type(param_type):
        element_type = get_array_element_type(param_type)
        return element_type in (float, Float, int, UInt)
    return False


def auto_detect_parameters(
    signature: inspect.Signature,
    input_types: dict[str, type],
    kwargs: dict[str, Any],
) -> list[str]:
    """Detect unbound classical arguments that should be runtime parameters.

    Args:
        signature (inspect.Signature): Python signature of the qkernel.
        input_types (dict[str, type]): Resolved frontend annotations keyed by
            parameter name.
        kwargs (dict[str, Any]): Compile-time bindings supplied to
            ``QKernel.build``.

    Returns:
        list[str]: Parameter names that should remain symbolic.
    """
    detected: list[str] = []
    for name, param in signature.parameters.items():
        param_type = input_types.get(name, param.annotation)

        if param_type is Qubit:
            continue
        if is_array_type(param_type) and get_array_element_type(param_type) is Qubit:
            continue
        if name in kwargs:
            continue
        if param.default is not inspect.Parameter.empty:
            continue
        if is_parameterizable_type(param_type):
            detected.append(name)

    return detected


def validate_parameters(
    input_types: dict[str, type],
    parameters: list[str],
) -> None:
    """Validate the explicit runtime parameter list.

    Args:
        input_types (dict[str, type]): Resolved qkernel input annotations.
        parameters (list[str]): Requested runtime parameter names.

    Raises:
        ValueError: If a requested name is not a qkernel parameter.
        TypeError: If a requested parameter type cannot stay symbolic.
    """
    for name in parameters:
        if name not in input_types:
            raise ValueError(f"Unknown parameter: '{name}'")

        param_type = input_types[name]
        if is_static_binding_annotation(param_type):
            raise TypeError(
                f"Parameter '{name}' has static binding type {param_type}; "
                "static bindings must be supplied at compile time and cannot "
                "be runtime parameters"
            )
        if is_dict_type(param_type):
            args = getattr(param_type, "__args__", None)
            if not args or len(args) < 2:
                raise TypeError(
                    f"Parameter '{name}' must be annotated as "
                    f"Dict[K, Float] with explicit key and value "
                    f"types to be kept as a runtime parameter; got "
                    f"{param_type}"
                )
            value_type = args[1]
            if value_type not in (float, Float):
                raise TypeError(
                    f"Parameter '{name}' is a Dict with value type "
                    f"{value_type}; only Dict[K, Float] can be kept "
                    f"as a runtime parameter"
                )
            continue

        if not is_parameterizable_type(param_type):
            raise TypeError(
                f"Parameter '{name}' has type {param_type}, "
                f"but only float, int, UInt, their arrays, and "
                f"Dict[K, Float] can be parameters"
            )


def validate_kwargs(
    signature: inspect.Signature,
    input_types: dict[str, type],
    parameters: list[str],
    kwargs: dict[str, Any],
) -> None:
    """Validate compile-time bindings for ``QKernel.build``.

    Args:
        signature (inspect.Signature): Python signature of the qkernel.
        input_types (dict[str, type]): Resolved frontend annotations keyed by
            parameter name.
        parameters (list[str]): Runtime parameter names.
        kwargs (dict[str, Any]): Compile-time bindings.

    Raises:
        ValueError: If an unknown argument is supplied, or if a required
            non-parameter classical argument is missing.
        TypeError: If a static binding has a default value or a supplied
            object does not match its registered annotation.
    """
    known_names = set(signature.parameters.keys())
    unknown = set(kwargs.keys()) - known_names
    if unknown:
        names = ", ".join(f"'{n}'" for n in sorted(unknown))
        raise ValueError(
            f"Unknown argument(s) {names} provided. "
            f"Known arguments are: {sorted(known_names)}"
        )

    for name, param in signature.parameters.items():
        param_type = input_types.get(name, param.annotation)
        if is_static_binding_annotation(param_type):
            if param.default is not inspect.Parameter.empty:
                raise TypeError(
                    f"Static binding parameter {name!r} cannot have a "
                    "default value; provide it through bindings when building "
                    "or transpiling the qkernel."
                )
            if name in parameters:
                raise TypeError(
                    f"Static binding argument {name!r} must be supplied at "
                    "compile time and cannot be a runtime parameter."
                )
            if name not in kwargs:
                raise ValueError(
                    f"Static binding argument {name!r} must be provided "
                    "through bindings."
                )
            validate_static_binding(param_type, name, kwargs[name])
            continue

        if name in parameters:
            continue

        if param_type is Qubit:
            continue
        if is_array_type(param_type):
            element_type = get_array_element_type(param_type)
            if element_type is Qubit:
                continue
            if element_type is Observable:
                continue
        if param_type is Observable:
            continue
        if is_dict_type(param_type) or is_tuple_type(param_type):
            continue
        if name not in kwargs and param.default is inspect.Parameter.empty:
            raise ValueError(
                f"Argument '{name}' must be provided or have a default value "
                f"(not a parameter, Qubit, or Observable type)"
            )


def create_parameter_input(param_type: Any, name: str) -> Handle:
    """Create a symbolic frontend handle for a runtime parameter.

    Args:
        param_type (Any): Frontend type annotation.
        name (str): QKernel parameter name.

    Returns:
        Handle: Symbolic handle carrying runtime parameter metadata.

    Raises:
        TypeError: If ``param_type`` cannot be represented symbolically.
    """
    if param_type in (float, Float):
        value = Value(type=FloatType(), name=name).with_parameter(name)
        return Float(value=value)

    if param_type is Observable:
        value = Value(type=ObservableType(), name=name).with_parameter(name)
        return Observable(value=value)

    if param_type in (int, UInt):
        value = Value(type=UIntType(), name=name).with_parameter(name)
        return UInt(value=value)

    if is_array_type(param_type):
        element_type = get_array_element_type(param_type)
        if element_type not in (Float, float, UInt, int, Observable):
            raise TypeError(
                f"Array parameter must have Float, UInt, or Observable "
                f"element type, got {element_type}"
            )
        if element_type is Observable and (
            getattr(param_type, "__origin__", param_type) is not Vector
        ):
            raise TypeError(f"Only Vector[Observable] is supported; got {param_type}")
        return create_dummy_input(param_type, name, emit_init=False)

    if is_dict_type(param_type):
        dict_value = DictValue(name=name, entries=()).with_parameter(name)
        dict_handle = Dict(value=dict_value, _entries=[], _runtime_parameter=True)
        if hasattr(param_type, "__args__") and param_type.__args__:
            dict_handle._key_type = param_type.__args__[0]
            if len(param_type.__args__) >= 2:
                dict_handle._value_type = param_type.__args__[1]
        return dict_handle

    raise TypeError(f"Cannot create parameter for type {param_type}")


def _array_binding_payload(
    param_type: Any,
    name: str,
    value: Any,
) -> tuple[Any, tuple[int, ...], Any]:
    """Convert an array binding into IR element type, shape, and payload.

    Args:
        param_type (Any): Array frontend annotation.
        name (str): QKernel parameter name.
        value (Any): Python binding value.

    Returns:
        tuple[Any, tuple[int, ...], Any]: IR element type, array shape, and
        serializer-friendly constant payload.

    Raises:
        TypeError: If the array element type or rank is unsupported.
    """
    element_type = get_array_element_type(param_type)
    if element_type in (Float, float):
        arr = np.asarray(value)
        return FloatType(), arr.shape, arr.tolist()
    if element_type in (UInt, int):
        arr = np.asarray(value, dtype=object)
        normalized = [
            _coerce_uint_binding(name, item, index=index)
            for index, item in enumerate(arr.flat)
        ]
        payload = np.asarray(normalized, dtype=object).reshape(arr.shape).tolist()
        return UIntType(), arr.shape, payload
    if element_type is Observable:
        if getattr(param_type, "__origin__", param_type) is not Vector:
            raise TypeError(
                f"Only Vector[Observable] bindings are supported; got {param_type}"
            )
        if isinstance(value, np.ndarray) and value.ndim != 1:
            raise TypeError(
                f"Vector[Observable] binding '{name}' must be 1-D; "
                f"got ndarray with shape {value.shape}."
            )
        items = list(value)
        for i, item in enumerate(items):
            if isinstance(item, (list, tuple, np.ndarray)):
                raise TypeError(
                    f"Vector[Observable] binding '{name}' must be a flat "
                    f"sequence of Hamiltonians; element {i} is "
                    f"{type(item).__name__}."
                )
        return ObservableType(), (len(items),), items
    raise TypeError(f"Unsupported element type for array binding: {element_type}")


def create_bound_input(param_type: Any, name: str, value: Any) -> Handle:
    """Create a frontend handle for a compile-time-bound value.

    Args:
        param_type (Any): Frontend type annotation.
        name (str): QKernel parameter name.
        value (Any): Concrete compile-time binding.

    Returns:
        Handle: Frontend handle carrying constant or runtime metadata.

    Raises:
        TypeError: If ``param_type`` cannot be bound from ``value``.
    """
    if param_type in (float, Float):
        return Float(
            value=Value(type=FloatType(), name=name).with_const(float(value)),
            init_value=float(value),
        )

    if param_type in (bool, Bit):
        return Bit(
            value=Value(type=BitType(), name=name).with_const(bool(value)),
            init_value=bool(value),
        )

    if param_type in (int, UInt):
        normalized = _coerce_uint_binding(name, value)
        return UInt(
            value=Value(type=UIntType(), name=name).with_const(normalized),
            init_value=normalized,
        )

    if is_array_type(param_type):
        ir_element_type, shape, const_data = _array_binding_payload(
            param_type,
            name,
            value,
        )
        shape_values = tuple(
            Value(type=UIntType(), name=f"dim_{i}").with_const(dim)
            for i, dim in enumerate(shape)
        )
        array_value = ArrayValue(
            type=ir_element_type,
            name=name,
            shape=shape_values,
        ).with_array_runtime_metadata(const_array=const_data)

        actual_class = getattr(param_type, "__origin__", param_type)
        instance = object.__new__(actual_class)
        instance.value = array_value
        instance._shape = shape
        instance._borrowed_indices = {}
        instance.parent = None
        instance.indices = ()
        instance.name = name
        instance.id = str(id(instance))
        instance._consumed = False
        instance.element_type = get_array_element_type(param_type)
        return instance

    if is_dict_type(param_type):
        dict_value = (
            DictValue(name=name, entries=())
            .with_parameter(name)
            .with_dict_runtime_metadata(value)
        )
        dict_handle = Dict(value=dict_value, _entries=[])
        if hasattr(param_type, "__args__") and param_type.__args__:
            dict_handle._key_type = param_type.__args__[0]
            if len(param_type.__args__) >= 2:
                dict_handle._value_type = param_type.__args__[1]
        return dict_handle

    raise TypeError(f"Cannot create bound value for type {param_type}")


def _coerce_uint_binding(name: str, value: Any, *, index: int | None = None) -> int:
    """Validate and normalize one compile-time UInt binding.

    Args:
        name (str): QKernel parameter name used in diagnostics.
        value (Any): Candidate scalar binding.
        index (int | None): Optional flattened array index. Defaults to
            ``None`` for a scalar binding.

    Returns:
        int: Non-negative integral binding.

    Raises:
        TypeError: If ``value`` is boolean or non-integral.
        ValueError: If ``value`` is negative.
    """
    location = f" at flattened index {index}" if index is not None else ""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(
            f"UInt binding '{name}'{location} must be an integer, got "
            f"{type(value).__name__} ({value!r})."
        )
    normalized = int(value)
    if normalized < 0:
        raise ValueError(
            f"UInt binding '{name}'{location} must be non-negative, got {normalized}."
        )
    return normalized
