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
from qamomile.circuit.frontend.handle.containers import Tuple, Dict
from qamomile.circuit.ir.value import ArrayValue, Value, TupleValue, DictValue


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
        return "Tuple"

    if isinstance(handle, Dict):
        return "Dict"

    return type(handle).__name__


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
    """Validate that handle type matches annotation.

    Raises TypeError if the handle's structural type (scalar/array/tuple/dict)
    or element type doesn't match the expected annotation.
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
        # Scalar: direct class comparison
        if type(handle) is not annotation:
            raise TypeError(
                f"Argument '{arg_name}': expected {_format_annotation(annotation)}, "
                f"got {_format_handle(handle)}."
            )
    # Tuple/Dict: kind match only (element types not available at tracing time)


def validate_return_type(annotation: typing.Any, val: typing.Any, index: int) -> None:
    """Validate that a return value matches the expected annotation.

    Checks both structural match (array vs scalar) and IR type match.
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
