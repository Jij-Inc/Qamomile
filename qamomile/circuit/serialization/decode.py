"""Decode one static qkernel from the semantic protobuf graph model."""

from __future__ import annotations

import dataclasses
import inspect
import types
from typing import Any, cast

from qamomile.circuit.frontend.func_to_block import build_param_slots, handle_type_map
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
from qamomile.circuit.frontend.static_binding import (
    get_static_binding_by_type_key,
    validate_static_binding_slot,
)
from qamomile.circuit.ir.serialize.decode import (
    _decode_block,
    _decode_callable_def,
    _decode_payload,
    _decode_value_type,
    _DecodeContext,
)
from qamomile.circuit.ir.types import DictType, QFixedType, ValueType
from qamomile.circuit.ir.value import ArrayValue, ValueLike

from .kernel import SerializedQKernel
from .schema import QAMOMILE_VERSION
from .validation import validate_qkernel_ir

_PARAMETER_KINDS = {
    kind.name: kind
    for kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.VAR_KEYWORD,
    )
}
_FRONTEND_SCALAR_TYPES: dict[str, Any] = {
    "QAMOMILE_UINT": UInt,
    "PYTHON_INT": int,
    "QAMOMILE_FLOAT": Float,
    "PYTHON_FLOAT": float,
    "QAMOMILE_BIT": Bit,
    "PYTHON_BOOL": bool,
    "QAMOMILE_QUBIT": Qubit,
    "QAMOMILE_QFIXED": QFixed,
    "QAMOMILE_OBSERVABLE": Observable,
}


def from_dict(envelope: dict[str, Any]) -> SerializedQKernel:
    """Reconstruct a static qkernel from an internal graph envelope.

    Args:
        envelope (dict[str, Any]): Dictionary produced by :func:`to_dict`.

    Returns:
        SerializedQKernel: QKernel-like object accepted by normal transpilers.

    Raises:
        ValueError: If the version, graph, interface, or references are
            malformed.
    """
    if not isinstance(envelope, dict):
        raise ValueError(
            f"from_dict() expected a dict envelope, got {type(envelope).__name__}"
        )
    version = envelope.get("qamomile_version")
    if version != QAMOMILE_VERSION:
        raise ValueError(
            f"qamomile_version mismatch: payload reports {version!r}, this "
            f"loader supports {QAMOMILE_VERSION!r}. Cross-version migration "
            "is not provided."
        )
    artifact = envelope.get("artifact")
    value_table = envelope.get("value_table")
    callable_table = envelope.get("callable_table")
    if not isinstance(artifact, dict) or artifact.get("$type") != "QKernel":
        raise ValueError("envelope is missing its QKernel artifact")
    if not isinstance(value_table, list) or not isinstance(callable_table, list):
        raise ValueError("envelope is missing value or callable tables")

    ctx = _DecodeContext(value_table, callable_table)
    ctx.populate_definitions()
    raw_body = artifact.get("body")
    if not isinstance(raw_body, dict):
        raise ValueError("QKernel artifact is missing its body Block")
    body = _decode_block(raw_body, ctx)
    raw_callable_definition = artifact.get("callable_definition")
    if not isinstance(raw_callable_definition, dict):
        raise ValueError("QKernel artifact is missing its callable definition")
    callable_definition = _decode_callable_def(raw_callable_definition, ctx)
    validate_qkernel_ir(body)

    raw_parameters = artifact.get("parameters")
    raw_results = artifact.get("results")
    if not isinstance(raw_parameters, list) or not isinstance(raw_results, list):
        raise ValueError("QKernel interface fields must be lists")
    if len(raw_parameters) != len(body.input_values) + len(body.static_bindings):
        raise ValueError(
            "QKernel parameter count does not match its ordinary inputs and "
            "static bindings"
        )
    if len(raw_results) != len(body.output_values):
        raise ValueError("QKernel result count does not match its body output count")

    parameters: list[inspect.Parameter] = []
    input_types: dict[str, Any] = {}
    differentiable: dict[str, bool] = {}
    formals = dict(zip(body.label_args, body.input_values, strict=True))
    static_slots = {slot.name: slot for slot in body.static_bindings}
    ordinary_names: list[str] = []
    static_names: list[str] = []
    for index, item in enumerate(raw_parameters):
        if not isinstance(item, dict):
            raise ValueError("QKernel parameter entries must be dictionaries")
        name = item.get("name")
        kind_name = item.get("kind")
        if not isinstance(name, str) or not isinstance(kind_name, str):
            raise ValueError("QKernel parameter name and kind must be strings")
        kind = _PARAMETER_KINDS.get(kind_name)
        if kind is None:
            raise ValueError(f"unknown qkernel parameter kind {kind_name!r}")
        raw_type = item.get("type")
        static_binding_type = item.get("static_binding_type")
        has_value_type = isinstance(raw_type, dict)
        has_static_type = isinstance(static_binding_type, str)
        if has_value_type == has_static_type:
            raise ValueError(
                f"QKernel parameter {name!r} requires exactly one ordinary "
                "or static binding type"
            )
        if has_static_type:
            assert isinstance(static_binding_type, str)
            if item.get("has_default") is True:
                raise ValueError(
                    f"static binding parameter {name!r} cannot have a default"
                )
            if item.get("differentiable") is True:
                raise ValueError(
                    f"static binding parameter {name!r} cannot be differentiable"
                )
            try:
                spec = get_static_binding_by_type_key(static_binding_type)
            except KeyError as exc:
                raise ValueError(
                    f"unknown static binding type {static_binding_type!r}"
                ) from exc
            slot = static_slots.get(name)
            if slot is None:
                raise ValueError(
                    f"static binding parameter {name!r} does not match its Block slot"
                )
            validate_static_binding_slot(spec, slot)
            annotation = spec.annotation
            static_names.append(name)
        else:
            formal = formals.get(name)
            if formal is None:
                raise ValueError(
                    f"ordinary qkernel parameter {name!r} has no Block input"
                )
            annotation = _decode_kernel_type_for_value(
                raw_type,
                formal,
                ctx,
                f"parameter {name!r} at index {index}",
            )
            ordinary_names.append(name)
        default = (
            _decode_payload(item.get("default"))
            if item.get("has_default") is True
            else inspect.Parameter.empty
        )
        parameters.append(
            inspect.Parameter(
                name=name,
                kind=kind,
                default=default,
                annotation=annotation,
            )
        )
        if name in input_types:
            raise ValueError(f"duplicate qkernel parameter {name!r}")
        input_types[name] = annotation
        differentiable[name] = item.get("differentiable") is True

    if ordinary_names != body.label_args:
        raise ValueError("QKernel ordinary parameter order disagrees with Block inputs")
    if static_names != list(static_slots):
        raise ValueError("QKernel static parameter order disagrees with Block slots")

    output_types = [
        _decode_kernel_type_for_value(
            item,
            formal,
            ctx,
            f"result at index {index}",
        )
        for index, (item, formal) in enumerate(
            zip(raw_results, body.output_values, strict=True)
        )
    ]
    raw_return_annotation = artifact.get("return_annotation")
    if not isinstance(raw_return_annotation, dict):
        raise ValueError("QKernel artifact is missing its return annotation")
    return_annotation = _decode_frontend_annotation(raw_return_annotation)
    if _return_output_types(return_annotation) != output_types:
        raise ValueError(
            "QKernel return annotation does not match its result descriptors"
        )
    name = artifact.get("name")
    if not isinstance(name, str):
        raise ValueError("QKernel name must be a string")
    signature = inspect.Signature(
        parameters,
        return_annotation=return_annotation,
    )
    body.param_slots = tuple(
        dataclasses.replace(
            slot,
            differentiable=differentiable.get(slot.name, False),
        )
        for slot in build_param_slots(
            signature=signature,
            input_types=input_types,
            bind_defaults=False,
        )
    )
    return SerializedQKernel(
        name=name,
        signature=signature,
        input_types=input_types,
        output_types=output_types,
        _block=body,
        _callable_definition=callable_definition,
    )


def _decode_kernel_type_record(
    value: Any,
    ctx: _DecodeContext,
) -> tuple[Any, ValueType, int]:
    """Decode an IR type plus rank into its semantic components.

    Args:
        value (Any): Internal kernel type record.
        ctx (_DecodeContext): Active graph decode context.

    Returns:
        tuple[Any, ValueType, int]: Frontend annotation, decoded IR value type,
            and array rank.

    Raises:
        ValueError: If the type descriptor or rank is invalid.
        TypeError: If the IR type has no qkernel frontend annotation.
    """
    if not isinstance(value, dict):
        raise ValueError("KernelType must be a dictionary")
    raw_value_type = value.get("value_type")
    ndim = value.get("ndim")
    raw_annotation = value.get("annotation")
    if (
        not isinstance(raw_value_type, dict)
        or not isinstance(ndim, int)
        or not isinstance(raw_annotation, dict)
    ):
        raise ValueError(
            "KernelType requires a ValueType, integer ndim, and annotation"
        )
    value_type = _decode_value_type(raw_value_type, ctx)
    annotation = _decode_frontend_annotation(raw_annotation)
    annotation_ndim = _frontend_annotation_ndim(annotation)
    try:
        annotation_value_type = handle_type_map(annotation)
    except TypeError as exc:
        if annotation is QFixed and isinstance(value_type, QFixedType):
            annotation_value_type = value_type
        else:
            raise ValueError(
                f"frontend annotation {annotation!r} has no matching IR type"
            ) from exc
    if annotation_value_type != value_type or annotation_ndim != ndim:
        raise ValueError(
            "KernelType annotation disagrees with its IR type or rank: "
            f"annotation={annotation!r}, value_type={value_type!r}, ndim={ndim}"
        )
    return annotation, value_type, ndim


def _decode_kernel_type_for_value(
    value: Any,
    body_value: ValueLike,
    ctx: _DecodeContext,
    label: str,
) -> Any:
    """Decode and validate one interface type against its body value.

    Args:
        value (Any): Internal kernel type record.
        body_value (ValueLike): Corresponding typed Block ABI value.
        ctx (_DecodeContext): Active graph decode context.
        label (str): Human-readable interface position for diagnostics.

    Returns:
        Any: Reconstructed frontend annotation.

    Raises:
        ValueError: If the descriptor type or rank differs from the Block ABI.
        TypeError: If the IR type has no qkernel frontend annotation.
    """
    try:
        annotation, value_type, ndim = _decode_kernel_type_record(value, ctx)
    except ValueError as exc:
        raise ValueError(f"QKernel {label} type does not match its annotation") from exc
    body_ndim = len(body_value.shape) if isinstance(body_value, ArrayValue) else 0
    if not _is_body_type_compatible(value_type, body_value.type) or ndim != body_ndim:
        raise ValueError(
            f"QKernel {label} type does not match its body value: "
            f"interface=({value_type!r}, ndim={ndim}), "
            f"body=({body_value.type!r}, ndim={body_ndim})"
        )
    return annotation


def _is_body_type_compatible(
    interface_type: ValueType,
    body_type: ValueType,
) -> bool:
    """Check whether an interface type agrees with a Block ABI type.

    Unbound ``DictValue`` formals currently carry ``DictType(None, None)`` in
    the Block even though their qkernel annotation retains concrete key and
    value types. Those absent Block details are not contradictions; every type
    component that the Block does specify must still match exactly.

    Args:
        interface_type (ValueType): Type declared by the qkernel interface.
        body_type (ValueType): Type attached to the corresponding Block value.

    Returns:
        bool: Whether all type information present in the Block agrees.
    """
    if isinstance(body_type, DictType):
        return (
            isinstance(interface_type, DictType)
            and (
                body_type.key_type is None
                or interface_type.key_type == body_type.key_type
            )
            and (
                body_type.value_type is None
                or interface_type.value_type == body_type.value_type
            )
        )
    return interface_type == body_type


def _decode_frontend_annotation(value: Any) -> Any:
    """Decode one recursive frontend annotation descriptor.

    Args:
        value (Any): Annotation kind and nested argument records.

    Returns:
        Any: Reconstructed Python or Qamomile frontend annotation.

    Raises:
        ValueError: If the kind, arity, or nested record is malformed.
    """
    if not isinstance(value, dict):
        raise ValueError("frontend annotation must be a dictionary")
    kind = value.get("kind")
    raw_arguments = value.get("arguments")
    if not isinstance(kind, str) or not isinstance(raw_arguments, list):
        raise ValueError("frontend annotation requires kind and arguments")
    arguments = [_decode_frontend_annotation(item) for item in raw_arguments]

    scalar = _FRONTEND_SCALAR_TYPES.get(kind)
    if scalar is not None:
        if arguments:
            raise ValueError(f"scalar frontend annotation {kind} takes no arguments")
        return scalar
    if kind == "QAMOMILE_VECTOR":
        _require_annotation_arity(kind, arguments, 1)
        return cast(Any, Vector)[arguments[0]]
    if kind == "QAMOMILE_MATRIX":
        _require_annotation_arity(kind, arguments, 1)
        return cast(Any, Matrix)[arguments[0]]
    if kind == "QAMOMILE_TENSOR":
        _require_annotation_arity(kind, arguments, 1)
        return cast(Any, Tensor)[arguments[0]]
    if kind == "QAMOMILE_TUPLE":
        if not arguments:
            raise ValueError("QAMOMILE_TUPLE requires at least one argument")
        return cast(Any, Tuple)[tuple(arguments)]
    if kind == "QAMOMILE_DICT":
        _require_annotation_arity(kind, arguments, 2)
        return cast(Any, Dict)[arguments[0], arguments[1]]
    if kind == "PYTHON_TUPLE":
        if not arguments:
            raise ValueError("PYTHON_TUPLE requires at least one argument")
        return types.GenericAlias(tuple, tuple(arguments))
    raise ValueError(f"unknown frontend annotation kind {kind!r}")


def _require_annotation_arity(
    kind: str,
    arguments: list[Any],
    expected: int,
) -> None:
    """Require an exact frontend annotation argument count.

    Args:
        kind (str): Annotation kind used in diagnostics.
        arguments (list[Any]): Decoded nested annotations.
        expected (int): Required argument count.

    Raises:
        ValueError: If the annotation has a different argument count.
    """
    if len(arguments) != expected:
        raise ValueError(
            f"frontend annotation {kind} requires {expected} arguments, "
            f"got {len(arguments)}"
        )


def _frontend_annotation_ndim(annotation: Any) -> int:
    """Return the array rank expressed by a frontend annotation.

    Args:
        annotation (Any): Reconstructed frontend annotation.

    Returns:
        int: Zero for non-arrays, otherwise one through three.

    Raises:
        ValueError: If an unknown array wrapper reaches validation.
    """
    origin = getattr(annotation, "__origin__", annotation)
    ranks = {Vector: 1, Matrix: 2, Tensor: 3}
    if origin in ranks:
        return ranks[origin]
    if origin in {UInt, int, Float, float, Bit, bool, Qubit, QFixed, Observable}:
        return 0
    if origin in {Tuple, Dict}:
        return 0
    raise ValueError(f"unsupported frontend annotation {annotation!r}")


def _return_output_types(return_annotation: Any) -> list[Any]:
    """Flatten a Python tuple return annotation like qkernel construction.

    Args:
        return_annotation (Any): Reconstructed qkernel return annotation.

    Returns:
        list[Any]: Per-result frontend annotations.
    """
    if getattr(return_annotation, "__origin__", None) is tuple:
        return list(getattr(return_annotation, "__args__", ()))
    return [return_annotation]
