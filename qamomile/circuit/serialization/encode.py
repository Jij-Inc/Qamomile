"""Encode one unbound qkernel into the semantic protobuf graph model."""

from __future__ import annotations

import inspect
import typing
from typing import Any

from qamomile.circuit.frontend.func_to_block import (
    handle_type_map,
    is_array_type,
    is_dict_type,
    is_tuple_type,
)
from qamomile.circuit.frontend.handle import (
    Bit,
    Dict,
    Float,
    Matrix,
    Observable,
    QFixed,
    Qubit,
    Tensor,
    Tuple,
    UInt,
    Vector,
)
from qamomile.circuit.frontend.qkernel_callable import (
    qkernel_callable_attrs,
    qkernel_callable_ref,
)
from qamomile.circuit.frontend.qkernel_like import QKernelLike
from qamomile.circuit.ir.block import BlockKind
from qamomile.circuit.ir.operation.callable import CallableDef, CallPolicy
from qamomile.circuit.ir.parameter import ParamKind
from qamomile.circuit.ir.serialize.encode import (
    _encode_block,
    _encode_callable_def,
    _encode_payload,
    _encode_value_type,
    _EncodeContext,
)
from qamomile.circuit.ir.types import ValueType
from qamomile.circuit.ir.value import ValueBase

from .canonical import canonicalize_graph
from .schema import QAMOMILE_VERSION
from .validation import validate_qkernel_ir


def to_dict(kernel: QKernelLike) -> dict[str, Any]:
    """Encode an unbound qkernel into an internal graph envelope.

    Args:
        kernel (QKernelLike): QKernel-like frontend object whose static body is
            preserved.

    Returns:
        dict[str, Any]: Internal graph record consumed by the protobuf bridge.

    Raises:
        TypeError: If ``kernel`` is not qkernel-like or contains an unsupported
            frontend type or process-local emitter.
        ValueError: If the body is specialized, non-hierarchical, or its
            interface disagrees with the signature.
    """
    _validate_kernel_surface(kernel)
    block = kernel.block
    if block.kind is not BlockKind.HIERARCHICAL:
        raise ValueError(
            "Only an unbound HIERARCHICAL qkernel body can be serialized; "
            f"got {block.kind}"
        )
    if block.parameters:
        raise ValueError(
            "Cannot serialize a qkernel body with selected runtime parameters; "
            "serialize the undecorated qkernel object before transpilation"
        )
    if any(
        slot.kind is ParamKind.COMPILE_TIME_BOUND or slot.bound_value is not None
        for slot in block.param_slots
    ):
        raise ValueError(
            "Cannot serialize compile-time bindings; serialize the unbound "
            "qkernel and provide bindings after deserialization"
        )
    if block.label_args != list(kernel.signature.parameters):
        raise ValueError("qkernel signature does not match Block.label_args")
    if len(kernel.output_types) != len(block.output_values):
        raise ValueError("qkernel return annotations do not match Block outputs")
    validate_qkernel_ir(block)

    ctx = _EncodeContext()
    parameters = []
    slots = {slot.name: slot for slot in block.param_slots}
    for (name, parameter), formal in zip(
        kernel.signature.parameters.items(),
        block.input_values,
        strict=True,
    ):
        annotation = kernel.input_types[name]
        parameters.append(
            {
                "name": name,
                "type": _encode_kernel_type(annotation, formal),
                "kind": parameter.kind.name,
                "has_default": parameter.default is not inspect.Parameter.empty,
                "default": (
                    _encode_payload(parameter.default)
                    if parameter.default is not inspect.Parameter.empty
                    else None
                ),
                "differentiable": (
                    slots[name].differentiable if name in slots else False
                ),
            }
        )
    results = [
        _encode_kernel_type(annotation, value)
        for annotation, value in zip(
            kernel.output_types,
            block.output_values,
            strict=True,
        )
    ]
    artifact = {
        "$type": "QKernel",
        "name": kernel.name,
        "parameters": parameters,
        "results": results,
        "return_annotation": _encode_frontend_annotation(
            _resolve_return_annotation(kernel)
        ),
        "body": _encode_block(block, ctx),
        "callable_definition": _encode_callable_def(
            CallableDef(
                ref=qkernel_callable_ref(kernel),
                implementations=list(getattr(kernel, "_callable_implementations", ())),
                default_policy=getattr(
                    kernel,
                    "_callable_policy",
                    CallPolicy.INLINE,
                ),
                attrs=qkernel_callable_attrs(kernel),
            ),
            ctx,
        ),
    }
    envelope = {
        "qamomile_version": QAMOMILE_VERSION,
        "artifact": artifact,
        "value_table": ctx.value_table_dicts,
        "callable_table": ctx.encode_callable_table(),
    }
    return canonicalize_graph(envelope)


def _validate_kernel_surface(kernel: QKernelLike) -> None:
    """Validate the structural qkernel interface required by serialization.

    Args:
        kernel (QKernelLike): Candidate qkernel-like object.

    Raises:
        TypeError: If a required qkernel attribute has the wrong type.
    """
    if not isinstance(getattr(kernel, "name", None), str):
        raise TypeError("serialize() expected a qkernel-like object with a string name")
    if not isinstance(getattr(kernel, "signature", None), inspect.Signature):
        raise TypeError("serialize() expected a qkernel-like object with a signature")
    if not isinstance(getattr(kernel, "input_types", None), dict):
        raise TypeError("serialize() expected a qkernel-like object with input_types")
    if not isinstance(getattr(kernel, "output_types", None), list):
        raise TypeError("serialize() expected a qkernel-like object with output_types")


def _encode_kernel_type(annotation: Any, value: ValueBase) -> dict[str, Any]:
    """Encode a frontend annotation as an IR type plus array rank.

    Args:
        annotation (Any): Resolved frontend qkernel annotation.
        value (ValueBase): Corresponding formal or result IR value.

    Returns:
        dict[str, Any]: Closed kernel type descriptor.

    Raises:
        TypeError: If the annotation cannot be represented by static IR.
    """
    ndim = _array_ndim(annotation)
    try:
        value_type = handle_type_map(annotation)
    except TypeError:
        if annotation is QFixed and hasattr(value, "type"):
            value_type = value.type
        else:
            raise
    if not isinstance(value_type, ValueType):
        raise TypeError(f"Unsupported qkernel type annotation {annotation!r}")
    return {
        "value_type": _encode_value_type(value_type),
        "ndim": ndim,
        "annotation": _encode_frontend_annotation(annotation),
    }


def _resolve_return_annotation(kernel: QKernelLike) -> Any:
    """Resolve one qkernel return annotation without losing its type family.

    Args:
        kernel (QKernelLike): QKernel-like object whose resolved return
            annotation is required.

    Returns:
        Any: Resolved scalar, container, array, or Python tuple annotation.

    Raises:
        TypeError: If a deferred annotation cannot be resolved exactly.
        ValueError: If the qkernel signature has no return annotation.
    """
    annotation = kernel.signature.return_annotation
    if annotation is inspect.Signature.empty:
        raise ValueError("qkernel signature has no return annotation")
    if not isinstance(annotation, str):
        return annotation

    raw_func = getattr(kernel, "raw_func", None)
    if not callable(raw_func):
        raise TypeError(
            f"Cannot resolve deferred return annotation {annotation!r} "
            "without the original qkernel function"
        )
    try:
        hints = typing.get_type_hints(
            raw_func,
            globalns=getattr(raw_func, "__globals__", {}),
            localns=None,
        )
    except (NameError, TypeError) as exc:
        raise TypeError(
            f"Cannot resolve deferred return annotation {annotation!r}"
        ) from exc
    resolved = hints.get("return")
    if resolved is None:
        raise TypeError(f"Cannot resolve deferred return annotation {annotation!r}")
    return resolved


def _encode_frontend_annotation(annotation: Any) -> dict[str, Any]:
    """Encode a supported frontend annotation without IR normalization.

    Args:
        annotation (Any): Resolved qkernel input or return annotation.

    Returns:
        dict[str, Any]: Recursive frontend annotation descriptor.

    Raises:
        TypeError: If the annotation is outside the qkernel type surface.
    """
    scalar_kinds = {
        UInt: "QAMOMILE_UINT",
        int: "PYTHON_INT",
        Float: "QAMOMILE_FLOAT",
        float: "PYTHON_FLOAT",
        Bit: "QAMOMILE_BIT",
        bool: "PYTHON_BOOL",
        Qubit: "QAMOMILE_QUBIT",
        QFixed: "QAMOMILE_QFIXED",
        Observable: "QAMOMILE_OBSERVABLE",
    }
    if annotation in scalar_kinds:
        return {"kind": scalar_kinds[annotation], "arguments": []}

    origin = getattr(annotation, "__origin__", annotation)
    arguments = list(getattr(annotation, "__args__", ()))
    if is_array_type(annotation):
        array_kinds = {
            Vector: "QAMOMILE_VECTOR",
            Matrix: "QAMOMILE_MATRIX",
            Tensor: "QAMOMILE_TENSOR",
        }
        kind = array_kinds.get(origin)
        if kind is None or len(arguments) != 1:
            raise TypeError(f"Unsupported qkernel array annotation {annotation!r}")
        return {
            "kind": kind,
            "arguments": [_encode_frontend_annotation(arguments[0])],
        }
    if is_tuple_type(annotation):
        if origin is not Tuple or not arguments:
            raise TypeError(f"Unsupported qkernel tuple annotation {annotation!r}")
        return {
            "kind": "QAMOMILE_TUPLE",
            "arguments": [_encode_frontend_annotation(item) for item in arguments],
        }
    if is_dict_type(annotation):
        if origin is not Dict or len(arguments) != 2:
            raise TypeError(f"Unsupported qkernel dict annotation {annotation!r}")
        return {
            "kind": "QAMOMILE_DICT",
            "arguments": [_encode_frontend_annotation(item) for item in arguments],
        }
    if origin is tuple:
        if not arguments:
            raise TypeError(f"Unsupported Python tuple annotation {annotation!r}")
        return {
            "kind": "PYTHON_TUPLE",
            "arguments": [_encode_frontend_annotation(item) for item in arguments],
        }
    raise TypeError(f"Unsupported frontend annotation {annotation!r}")


def _array_ndim(annotation: Any) -> int:
    """Return the declared array rank of a frontend annotation.

    Args:
        annotation (Any): Resolved frontend annotation.

    Returns:
        int: Zero for scalar/container values, otherwise one through three.

    Raises:
        TypeError: If an unknown array wrapper is used.
    """
    if not is_array_type(annotation):
        return 0
    origin = getattr(annotation, "__origin__", annotation)
    ranks = {Vector: 1, Matrix: 2, Tensor: 3}
    if origin not in ranks:
        raise TypeError(f"Unsupported qkernel array annotation {annotation!r}")
    return ranks[origin]
