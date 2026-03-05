"""Type validation for QKernel argument and return value checking.

This module provides validation functions that ensure Handle types
passed to QKernel.__call__ match the function's type annotations.
"""

from __future__ import annotations

import typing

from qamomile.circuit.frontend.func_to_block import (
    is_array_type,
    is_dict_type,
    is_tuple_type,
    handle_type_map,
)
from qamomile.circuit.frontend.handle.handle import Handle
from qamomile.circuit.frontend.handle.array import ArrayBase
from qamomile.circuit.frontend.handle.primitives import Float, UInt, Bit
from qamomile.circuit.frontend.handle.containers import Tuple, Dict
from qamomile.circuit.ir.value import ArrayValue, Value, TupleValue, DictValue


_PRIMITIVE_TO_HANDLE: dict[type, type] = {
    float: Float,
    int: UInt,
    bool: Bit,
}


def _normalize_primitive(annotation: typing.Any) -> typing.Any:
    """Normalize a Python primitive type annotation to its Handle class.

    Maps ``float`` to ``Float``, ``int`` to ``UInt``, ``bool`` to ``Bit``.
    Non-primitive annotations are returned unchanged.
    """
    return _PRIMITIVE_TO_HANDLE.get(annotation, annotation)


def _format_annotation(annotation: typing.Any) -> str:
    """Format a type annotation for error messages.

    Recursively formats nested types so that e.g.
    ``Tuple[Vector[Qubit], UInt]`` is displayed in full rather than
    ``Tuple[Vector, UInt]``.
    """
    if is_array_type(annotation):
        cls = getattr(annotation, "__origin__", annotation).__name__
        if hasattr(annotation, "__args__") and annotation.__args__:
            elem = _format_annotation(annotation.__args__[0])
            return f"{cls}[{elem}]"
        return cls

    if is_tuple_type(annotation):
        if hasattr(annotation, "__args__") and annotation.__args__:
            elems = ", ".join(_format_annotation(a) for a in annotation.__args__)
            return f"Tuple[{elems}]"
        return "Tuple"

    if is_dict_type(annotation):
        if hasattr(annotation, "__args__") and len(annotation.__args__) == 2:
            key = _format_annotation(annotation.__args__[0])
            val = _format_annotation(annotation.__args__[1])
            return f"Dict[{key}, {val}]"
        return "Dict"

    return annotation.__name__


def _format_handle(handle: Handle) -> str:
    """Format a Handle instance for error messages."""
    if isinstance(handle, ArrayBase):
        cls = type(handle).__name__
        if hasattr(handle, "element_type") and handle.element_type is not None:
            return f"{cls}[{handle.element_type.__name__}]"
        return cls

    if isinstance(handle, Tuple):
        if handle._elements:
            elems = ", ".join(type(e).__name__ for e in handle._elements)
            return f"Tuple[{elems}]"
        return "Tuple"

    if isinstance(handle, Dict):
        # Try _entries first
        if handle._entries:
            types = _infer_homogeneous_types(
                (type(k).__name__, type(v).__name__) for k, v in handle._entries
            )
            if types:
                return f"Dict[{types[0]}, {types[1]}]"
            return "Dict"
        # Fallback: infer from bound_data
        bound_data = getattr(handle.value, "params", {}).get("bound_data")
        if isinstance(bound_data, dict) and bound_data:
            types = _infer_types_from_bound_data(bound_data)
            if types:
                return f"Dict[{types[0]}, {types[1]}]"
        return "Dict"

    return type(handle).__name__


def _infer_homogeneous_types(
    pairs: typing.Iterable[tuple[str, str]],
) -> tuple[str, str] | None:
    """Return ``(key_type, val_type)`` if all pairs are the same, else ``None``."""
    key_type: str | None = None
    val_type: str | None = None
    for k, v in pairs:
        if key_type is None:
            key_type, val_type = k, v
        elif k != key_type or v != val_type:
            return None
    if key_type is None:
        return None
    return (key_type, val_type)


def _infer_types_from_bound_data(
    data: dict[typing.Any, typing.Any],
) -> tuple[str, str] | None:
    """Infer key/value Handle type names from bound Python data.

    Returns ``(key_name, val_name)`` if all entries share the same types,
    or ``None`` for heterogeneous or unrecognizable data.
    """
    _PYTHON_TO_HANDLE_NAME: dict[type, str] = {
        int: "UInt",
        float: "Float",
        bool: "Bit",
        tuple: "Tuple",
    }
    key_name: str | None = None
    val_name: str | None = None
    for k, v in data.items():
        k_name = _PYTHON_TO_HANDLE_NAME.get(type(k))
        v_name = _PYTHON_TO_HANDLE_NAME.get(type(v))
        if k_name is None or v_name is None:
            return None
        # For tuple keys, try to detail the element types
        if isinstance(k, tuple) and k:
            elem_types = set(type(e) for e in k)
            if len(elem_types) == 1:
                inner = _PYTHON_TO_HANDLE_NAME.get(elem_types.pop())
                if inner:
                    k_name = f"Tuple[{', '.join(inner for _ in k)}]"
                else:
                    return None
            else:
                return None
        if key_name is None:
            key_name, val_name = k_name, v_name
        elif k_name != key_name or v_name != val_name:
            return None
    if key_name is None:
        return None
    return (key_name, val_name)


def _annotation_kind(annotation: typing.Any) -> str:
    """Return the structural kind of a type annotation.

    Returns one of ``"array"``, ``"tuple"``, ``"dict"``, or ``"scalar"``.
    """
    if is_array_type(annotation):
        return "array"
    if is_tuple_type(annotation):
        return "tuple"
    if is_dict_type(annotation):
        return "dict"
    return "scalar"


def _handle_kind(handle: Handle) -> str:
    """Return the structural kind of a Handle instance.

    Returns one of ``"array"``, ``"tuple"``, ``"dict"``, or ``"scalar"``.
    """
    if isinstance(handle, ArrayBase):
        return "array"
    if isinstance(handle, Tuple):
        return "tuple"
    if isinstance(handle, Dict):
        return "dict"
    return "scalar"


def validate_argument_type(
    annotation: typing.Any, handle: Handle, arg_name: str
) -> None:
    """Validate that a handle's type matches the expected annotation.

    Args:
        annotation: The expected type annotation from the function signature.
            May be a Handle subclass (``Qubit``, ``Float``, etc.), a Python
            primitive (``float``, ``int``, ``bool``), or a generic alias
            (``Vector[Qubit]``, ``Dict[Tuple[UInt, UInt], Float]``).
        handle: The actual Handle instance passed at call time.
        arg_name: The parameter name, used in error messages.

    Raises:
        TypeError: If the structural kind (scalar/array/tuple/dict) or, for
            arrays and scalars, the element/class type does not match.

    Notes:
        For Tuple and Dict types, only the structural *kind* is validated.
        Element types within Tuple/Dict are not checked because they are not
        available during tracing.
    """
    ann_kind = _annotation_kind(annotation)
    hdl_kind = _handle_kind(handle)

    # Kind mismatch check
    if ann_kind != hdl_kind:
        raise TypeError(
            f"Argument '{arg_name}': expected {_format_annotation(annotation)}, "
            f"got {_format_handle(handle)}."
        )

    # Element type checks within the same kind
    if ann_kind == "array":
        # Array class mismatch (Vector vs Matrix vs Tensor)
        expected_class = getattr(annotation, "__origin__", annotation).__name__
        actual_class = type(handle).__name__
        if expected_class != actual_class:
            raise TypeError(
                f"Argument '{arg_name}': expected {_format_annotation(annotation)}, "
                f"got {_format_handle(handle)}."
            )
        # Element type mismatch
        if hasattr(annotation, "__args__") and annotation.__args__:
            if annotation.__args__[0] is not handle.element_type:
                raise TypeError(
                    f"Argument '{arg_name}': expected {_format_annotation(annotation)}, "
                    f"got {_format_handle(handle)}."
                )
    elif ann_kind == "scalar":
        # Scalar: normalize primitive annotations (float->Float, int->UInt, bool->Bit)
        normalized = _normalize_primitive(annotation)
        if type(handle) is not normalized:
            raise TypeError(
                f"Argument '{arg_name}': expected {_format_annotation(annotation)}, "
                f"got {_format_handle(handle)}."
            )
    # Tuple/Dict: kind match only (element types not available at tracing time)


def validate_return_type(annotation: typing.Any, val: typing.Any, index: int) -> None:
    """Validate that a return value matches the expected annotation.

    Checks both the structural match (array vs scalar vs tuple vs dict) and,
    for scalar/array values, the IR element type.

    Args:
        annotation: The expected return type annotation.
        val: The actual IR value (``Value``, ``ArrayValue``, ``TupleValue``,
            or ``DictValue``) produced by the kernel call.
        index: Zero-based index of this return value, used in error messages.

    Raises:
        TypeError: If the structural kind or IR element type does not match.
    """
    ann_is_array = is_array_type(annotation)
    ann_is_tuple = is_tuple_type(annotation)
    ann_is_dict = is_dict_type(annotation)

    # Structural mismatch checks
    if ann_is_array and not isinstance(val, ArrayValue):
        raise TypeError(
            f"Return value {index}: expected {_format_annotation(annotation)}, "
            f"got scalar value."
        )
    if (
        not ann_is_array
        and not ann_is_tuple
        and not ann_is_dict
        and isinstance(val, ArrayValue)
    ):
        raise TypeError(
            f"Return value {index}: expected {_format_annotation(annotation)}, "
            f"got array value."
        )
    if ann_is_tuple and not isinstance(val, TupleValue):
        raise TypeError(
            f"Return value {index}: expected {_format_annotation(annotation)}, "
            f"got non-tuple value."
        )
    if ann_is_dict and not isinstance(val, DictValue):
        raise TypeError(
            f"Return value {index}: expected {_format_annotation(annotation)}, "
            f"got non-dict value."
        )

    # IR type comparison (Value/ArrayValue only)
    if isinstance(val, (Value, ArrayValue)):
        expected_ir_type = handle_type_map(annotation)
        if val.type != expected_ir_type:
            raise TypeError(
                f"Return value {index}: expected element type "
                f"{expected_ir_type.label()}, got {val.type.label()}."
            )
