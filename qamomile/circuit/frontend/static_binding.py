"""Register and trace compile-time object bindings for qkernels."""

from __future__ import annotations

import copy
import dataclasses
import inspect
import types
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, cast

from qamomile.circuit.frontend.handle import Float, Handle, Qubit, UInt, Vector
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation.callable import (
    CallableBodyRef,
    CallableDef,
    CallableRef,
    CallPolicy,
    InvokeOperation,
    signature_from_values,
)
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.static_binding import StaticBindingField, StaticBindingSlot
from qamomile.circuit.ir.types import FloatType, QubitType, UIntType, ValueType
from qamomile.circuit.ir.value import ArrayValue, Value, ValueLike, ValueMetadata


@dataclasses.dataclass(frozen=True)
class StaticBindingFieldSpec:
    """Describe one scalar field exposed by a static-binding proxy.

    Args:
        handle_type (type[Handle]): Frontend scalar handle returned while
            tracing an unbound qkernel.
        getter (Callable[[Any], int | float]): Extractor used when a concrete
            object is bound.
    """

    handle_type: type[Handle]
    getter: Callable[[Any], int | float]


@dataclasses.dataclass(frozen=True)
class StaticBindingMemberSpec:
    """Describe one deferred qkernel-valued member of a static binding.

    Args:
        input_types (Mapping[str, Any]): Ordered frontend input annotations.
        output_types (tuple[Any, ...]): Ordered frontend result annotations.
        return_annotation (Any): Complete Python return annotation.
        getter (Callable[[Any], Any]): Extractor returning the concrete
            qkernel-like member.
        qubit_width_fields (Mapping[str, str]): Input-name to scalar field-name
            mapping used to specialize quantum vector widths.
    """

    input_types: Mapping[str, Any]
    output_types: tuple[Any, ...]
    return_annotation: Any
    getter: Callable[[Any], Any]
    qubit_width_fields: Mapping[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class StaticBindingSpec:
    """Register the closed qkernel surface of one compile-time object type.

    Args:
        annotation (type[Any]): Public qkernel parameter annotation.
        type_key (str): Stable serialization key.
        fields (Mapping[str, StaticBindingFieldSpec]): Scalar projections
            available while tracing.
        members (Mapping[str, StaticBindingMemberSpec]): Deferred callable
            members available while tracing.
    """

    annotation: type[Any]
    type_key: str
    fields: Mapping[str, StaticBindingFieldSpec]
    members: Mapping[str, StaticBindingMemberSpec]


_SPECS_BY_ANNOTATION: dict[type[Any], StaticBindingSpec] = {}
_SPECS_BY_TYPE_KEY: dict[str, StaticBindingSpec] = {}


def register_static_binding(spec: StaticBindingSpec) -> None:
    """Register one closed compile-time object adapter.

    Args:
        spec (StaticBindingSpec): Adapter contract to register.

    Raises:
        TypeError: If the annotation or type key has the wrong type, or if a
            field getter, field handle, or deferred member ABI is unsupported.
        ValueError: If the annotation or stable type key is already registered,
            or if the contract is empty or malformed.
    """
    if not isinstance(spec.annotation, type):
        raise TypeError("static binding annotation must be a concrete type")
    if not isinstance(spec.type_key, str):
        raise TypeError("static binding type_key must be a string")
    if not spec.type_key:
        raise ValueError("static binding type_key must be non-empty")
    if spec.annotation in _SPECS_BY_ANNOTATION:
        raise ValueError(
            f"static binding annotation {spec.annotation!r} is already registered"
        )
    if spec.type_key in _SPECS_BY_TYPE_KEY:
        raise ValueError(f"static binding type key {spec.type_key!r} is registered")
    if not spec.fields and not spec.members:
        raise ValueError("static binding adapters must expose a field or member")
    if any(not isinstance(name, str) for name in (*spec.fields, *spec.members)):
        raise TypeError("static binding field and member names must be strings")
    if any(not name for name in (*spec.fields, *spec.members)):
        raise ValueError("static binding field and member names must be non-empty")
    overlap = set(spec.fields) & set(spec.members)
    if overlap:
        raise ValueError(
            f"static binding field and member names overlap: {sorted(overlap)!r}"
        )
    fields = dict(spec.fields)
    for field in fields.values():
        if not callable(field.getter):
            raise TypeError("static binding field getter must be callable")
        _field_value_type(field)
    members: dict[str, StaticBindingMemberSpec] = {}
    for name, member in spec.members.items():
        frozen_member = dataclasses.replace(
            member,
            input_types=MappingProxyType(dict(member.input_types)),
            output_types=tuple(member.output_types),
            qubit_width_fields=MappingProxyType(dict(member.qubit_width_fields)),
        )
        _validate_static_member_spec(name, frozen_member, fields)
        members[name] = frozen_member
    registered_spec = dataclasses.replace(
        spec,
        fields=MappingProxyType(fields),
        members=MappingProxyType(members),
    )
    _SPECS_BY_ANNOTATION[spec.annotation] = registered_spec
    _SPECS_BY_TYPE_KEY[spec.type_key] = registered_spec


def _validate_static_member_spec(
    name: str,
    member: StaticBindingMemberSpec,
    fields: Mapping[str, StaticBindingFieldSpec],
) -> None:
    """Validate and close one registered deferred-member contract.

    Args:
        name (str): Registered member name.
        member (StaticBindingMemberSpec): Member contract to validate.
        fields (Mapping[str, StaticBindingFieldSpec]): Available scalar fields.

    Raises:
        TypeError: If the getter or vector ABI is unsupported.
        ValueError: If width metadata is incomplete or refers to an invalid
            scalar field.
    """
    if not callable(member.getter):
        raise TypeError(f"static binding member {name!r} getter must be callable")
    input_names = list(member.input_types)
    if not input_names or any(
        annotation != Vector[Qubit] for annotation in member.input_types.values()
    ):
        raise TypeError(
            f"static binding member {name!r} inputs must be quantum vectors"
        )
    if list(member.output_types) != [Vector[Qubit]] * len(input_names):
        raise TypeError(
            f"static binding member {name!r} outputs must mirror its inputs"
        )
    expected_return = (
        member.output_types[0]
        if len(member.output_types) == 1
        else types.GenericAlias(tuple, tuple(member.output_types))
    )
    if member.return_annotation != expected_return:
        raise TypeError(
            f"static binding member {name!r} return annotation must be "
            f"{expected_return!r}"
        )
    if set(member.qubit_width_fields) != set(input_names):
        raise ValueError(
            f"static binding member {name!r} must declare one width field "
            "for every input"
        )
    for input_name, field_name in member.qubit_width_fields.items():
        field = fields.get(field_name)
        if field is None:
            raise ValueError(
                f"static binding member {name!r} input {input_name!r} refers "
                f"to unknown width field {field_name!r}"
            )
        if field.handle_type is not UInt:
            raise TypeError(
                f"static binding member {name!r} width field {field_name!r} "
                "must use UInt"
            )


def get_static_binding_by_annotation(
    annotation: Any,
) -> StaticBindingSpec | None:
    """Return the adapter registered for a qkernel annotation.

    Args:
        annotation (Any): Resolved qkernel parameter annotation.

    Returns:
        StaticBindingSpec | None: Registered adapter, or ``None`` for an
        ordinary qkernel argument.
    """
    return _SPECS_BY_ANNOTATION.get(annotation)


def get_static_binding_by_type_key(type_key: str) -> StaticBindingSpec:
    """Return the adapter registered under a stable serialization key.

    Args:
        type_key (str): Stable type key from serialized IR.

    Returns:
        StaticBindingSpec: Matching registered adapter.

    Raises:
        KeyError: If the installed Qamomile distribution does not know the key.
    """
    return _SPECS_BY_TYPE_KEY[type_key]


def is_static_binding_annotation(annotation: Any) -> bool:
    """Return whether an annotation denotes a registered static binding.

    Args:
        annotation (Any): Resolved qkernel parameter annotation.

    Returns:
        bool: Whether the annotation is registered.
    """
    return get_static_binding_by_annotation(annotation) is not None


def validate_static_binding(annotation: Any, name: str, value: Any) -> Any:
    """Validate one concrete compile-time object binding.

    Args:
        annotation (Any): Registered qkernel parameter annotation.
        name (str): Parameter name used in diagnostics.
        value (Any): Candidate binding value.

    Returns:
        Any: The validated binding value.

    Raises:
        TypeError: If the annotation is not registered or the value has the
            wrong concrete type.
    """
    spec = get_static_binding_by_annotation(annotation)
    if spec is None:
        raise TypeError(f"{annotation!r} is not a registered static binding type")
    if not isinstance(value, spec.annotation):
        raise TypeError(
            f"Static binding {name!r} must be {spec.annotation.__name__}, got "
            f"{type(value).__name__}."
        )
    return value


def validate_static_binding_argument(
    annotation: Any,
    name: str,
    value: Any,
) -> Any:
    """Validate a concrete binding or caller-owned symbolic binding proxy.

    Args:
        annotation (Any): Registered qkernel parameter annotation.
        name (str): Callee parameter name used as the binding-slot identity.
        value (Any): Concrete registered object or symbolic binding proxy.

    Returns:
        Any: The validated concrete object or unchanged symbolic proxy.

    Raises:
        TypeError: If a concrete object has the wrong type, or a symbolic
            proxy does not preserve the callee parameter's slot name and type
            key.
    """
    if not isinstance(value, StaticBindingProxy):
        return validate_static_binding(annotation, name, value)

    expected_spec = get_static_binding_by_annotation(annotation)
    if expected_spec is None:
        raise TypeError(f"{annotation!r} is not a registered static binding type")
    slot = value.slot
    if slot.name != name:
        raise TypeError(
            f"Symbolic static binding argument {name!r} must come from the "
            f"same-named slot, got {slot.name!r}."
        )
    if slot.type_key != expected_spec.type_key:
        raise TypeError(
            f"Symbolic static binding argument {name!r} must have type key "
            f"{expected_spec.type_key!r}, got {slot.type_key!r}."
        )
    return value


def validate_static_binding_slot(
    spec: StaticBindingSpec,
    slot: StaticBindingSlot,
) -> None:
    """Validate a serialized IR slot against its installed adapter.

    Args:
        spec (StaticBindingSpec): Installed adapter contract.
        slot (StaticBindingSlot): IR manifest entry to validate.

    Raises:
        ValueError: If the type key, projected field names, or field types do
            not exactly match the registered adapter.
    """
    if slot.type_key != spec.type_key:
        raise ValueError(
            f"static binding {slot.name!r} has type key {slot.type_key!r}, "
            f"expected {spec.type_key!r}"
        )
    if [field.name for field in slot.fields] != list(spec.fields):
        raise ValueError(
            f"static binding {slot.name!r} field order does not match "
            f"registered type {spec.type_key!r}"
        )
    fields = {field.name: field for field in slot.fields}
    for name, field_spec in spec.fields.items():
        field_value = fields[name].value
        if (
            type(field_value) is not Value
            or field_value.parent_array is not None
            or field_value.element_indices
            or field_value.metadata != ValueMetadata()
        ):
            raise ValueError(
                f"static binding {slot.name!r} field {name!r} must be a "
                "standalone scalar Value"
            )
        expected_type = _field_value_type(field_spec)
        if field_value.type != expected_type:
            raise ValueError(
                f"static binding {slot.name!r} field {name!r} has type "
                f"{field_value.type!r}, expected {expected_type!r}"
            )


def without_static_bindings(
    input_types: Mapping[str, Any],
    bindings: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Remove compile-time object bindings already consumed by qkernel build.

    Args:
        input_types (Mapping[str, Any]): QKernel input annotations by name.
        bindings (Mapping[str, Any] | None): User-provided compile-time values.

    Returns:
        dict[str, Any]: Ordinary scalar, array, and structural bindings only.
    """
    if bindings is None:
        return {}
    return {
        name: value
        for name, value in bindings.items()
        if not is_static_binding_annotation(input_types.get(name))
    }


class StaticBindingProxy:
    """Expose a registered static object surface during unbound tracing.

    Args:
        spec (StaticBindingSpec): Registered object contract.
        name (str): QKernel parameter name identifying the binding slot.
    """

    def __init__(self, spec: StaticBindingSpec, name: str) -> None:
        """Create symbolic fields and deferred callable members.

        Args:
            spec (StaticBindingSpec): Registered object contract.
            name (str): QKernel parameter name identifying the binding slot.
        """
        self._spec = spec
        self._name = name
        self._field_handles = {
            field_name: _field_handle(field_name, field_spec, slot_name=name)
            for field_name, field_spec in spec.fields.items()
        }
        self._member_kernels = {
            member_name: _StaticBindingMemberKernel(
                slot_name=name,
                type_key=spec.type_key,
                member_name=member_name,
                spec=member_spec,
            )
            for member_name, member_spec in spec.members.items()
        }

    @property
    def slot(self) -> StaticBindingSlot:
        """Return the IR manifest entry owned by this proxy.

        Returns:
            StaticBindingSlot: Typed slot with every registered scalar field.
        """
        return StaticBindingSlot(
            name=self._name,
            type_key=self._spec.type_key,
            fields=tuple(
                StaticBindingField(name=name, value=cast(Value, handle.value))
                for name, handle in self._field_handles.items()
            ),
        )

    def __getattr__(self, name: str) -> Any:
        """Resolve one registered scalar field or callable member.

        Args:
            name (str): Attribute requested by the traced qkernel body.

        Returns:
            Any: Symbolic scalar handle or deferred qkernel-like member.

        Raises:
            AttributeError: If the adapter does not expose ``name``.
        """
        field = self._field_handles.get(name)
        if field is not None:
            return field
        member = self._member_kernels.get(name)
        if member is not None:
            return member
        raise AttributeError(
            f"{self._spec.annotation.__name__} static binding has no "
            f"registered attribute {name!r}"
        )


def create_static_binding_proxy(annotation: Any, name: str) -> StaticBindingProxy:
    """Create an unbound tracing proxy for a registered annotation.

    Args:
        annotation (Any): Registered qkernel parameter annotation.
        name (str): QKernel parameter name identifying the slot.

    Returns:
        StaticBindingProxy: Closed symbolic adapter surface.

    Raises:
        TypeError: If ``annotation`` is not registered.
    """
    spec = get_static_binding_by_annotation(annotation)
    if spec is None:
        raise TypeError(f"{annotation!r} is not a registered static binding type")
    return StaticBindingProxy(spec, name)


def materialize_static_field(
    spec: StaticBindingSpec,
    binding: Any,
    field_name: str,
) -> int | float:
    """Extract and validate one registered scalar field.

    Args:
        spec (StaticBindingSpec): Registered object contract.
        binding (Any): Validated concrete object.
        field_name (str): Registered field name.

    Returns:
        int | float: Scalar value suitable for IR constant metadata.

    Raises:
        KeyError: If ``field_name`` is not registered.
        TypeError: If the extracted value does not match its handle type.
        ValueError: If a ``UInt`` field is negative.
    """
    field = spec.fields[field_name]
    value = field.getter(binding)
    if field.handle_type is UInt:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"static UInt field {field_name!r} must be an int")
        if value < 0:
            raise ValueError(f"static UInt field {field_name!r} must be non-negative")
        return value
    if field.handle_type is Float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"static Float field {field_name!r} must be real")
        return float(value)
    raise TypeError(f"unsupported static field handle {field.handle_type!r}")


def materialize_static_member(
    spec: StaticBindingSpec,
    binding: Any,
    member_name: str,
) -> tuple[Any, StaticBindingMemberSpec]:
    """Extract one registered qkernel-like member.

    Args:
        spec (StaticBindingSpec): Registered object contract.
        binding (Any): Validated concrete object.
        member_name (str): Registered member name.

    Returns:
        tuple[Any, StaticBindingMemberSpec]: Concrete member and its adapter
        contract.

    Raises:
        KeyError: If ``member_name`` is not registered.
        TypeError: If the getter does not return a qkernel-like object.
    """
    member_spec = spec.members[member_name]
    member = member_spec.getter(binding)
    if not all(
        hasattr(member, attribute)
        for attribute in ("name", "signature", "input_types", "output_types", "block")
    ):
        raise TypeError(
            f"static member {member_name!r} did not produce a qkernel-like object"
        )
    signature = member.signature
    if not isinstance(member.name, str) or not isinstance(signature, inspect.Signature):
        raise TypeError(
            f"static member {member_name!r} did not produce a qkernel-like object"
        )
    if list(signature.parameters) != list(member_spec.input_types):
        raise TypeError(
            f"static member {member_name!r} parameter order disagrees with its "
            "registered adapter"
        )
    if any(
        parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
        or parameter.default is not inspect.Parameter.empty
        for parameter in signature.parameters.values()
    ):
        raise TypeError(
            f"static member {member_name!r} parameters must be required "
            "positional-or-keyword arguments"
        )
    if signature.return_annotation is inspect.Signature.empty:
        raise TypeError(f"static member {member_name!r} requires a return annotation")
    if dict(member.input_types) != dict(member_spec.input_types) or list(
        member.output_types
    ) != list(member_spec.output_types):
        raise TypeError(
            f"static member {member_name!r} frontend ABI disagrees with its "
            "registered adapter"
        )
    if not isinstance(member.block, Block):
        raise TypeError(f"static member {member_name!r} has no IR Block body")
    return member, member_spec


def _field_value_type(spec: StaticBindingFieldSpec) -> ValueType:
    """Return the IR scalar type for a registered field.

    Args:
        spec (StaticBindingFieldSpec): Registered field contract.

    Returns:
        ValueType: Matching scalar IR type.

    Raises:
        TypeError: If the adapter uses an unsupported frontend handle.
    """
    if spec.handle_type is UInt:
        return UIntType()
    if spec.handle_type is Float:
        return FloatType()
    raise TypeError(
        "static binding fields currently support only UInt and Float, got "
        f"{spec.handle_type!r}"
    )


def _field_handle(
    name: str,
    spec: StaticBindingFieldSpec,
    *,
    slot_name: str,
) -> Handle:
    """Create one symbolic scalar handle projected from a binding slot.

    Args:
        name (str): Registered field name.
        spec (StaticBindingFieldSpec): Registered field contract.
        slot_name (str): Owning qkernel parameter name.

    Returns:
        Handle: Symbolic ``UInt`` or ``Float`` handle.
    """
    value = Value(type=_field_value_type(spec), name=f"{slot_name}.{name}")
    if spec.handle_type is UInt:
        return UInt(value=value)
    return Float(value=value)


class _StaticBindingMemberKernel:
    """QKernel-like placeholder whose body contains a deferred member call.

    Args:
        slot_name (str): Owning qkernel parameter name.
        type_key (str): Stable static-binding adapter key.
        member_name (str): Registered callable member name.
        spec (StaticBindingMemberSpec): Callable signature contract.
    """

    def __init__(
        self,
        *,
        slot_name: str,
        type_key: str,
        member_name: str,
        spec: StaticBindingMemberSpec,
    ) -> None:
        """Build a bodyless member reference behind a normal qkernel ABI.

        Args:
            slot_name (str): Owning qkernel parameter name.
            type_key (str): Stable static-binding adapter key.
            member_name (str): Registered callable member name.
            spec (StaticBindingMemberSpec): Callable signature contract.
        """
        self.name = f"{slot_name}.{member_name}"
        self.signature = inspect.Signature(
            [
                inspect.Parameter(
                    name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=annotation,
                )
                for name, annotation in spec.input_types.items()
            ],
            return_annotation=spec.return_annotation,
        )
        self.input_types = dict(spec.input_types)
        self.output_types = list(spec.output_types)
        self._block = _deferred_member_block(
            slot_name=slot_name,
            type_key=type_key,
            member_name=member_name,
            spec=spec,
        )
        self._block_building = False
        self._specializing = True
        self._callable_kind = "qkernel"
        self._callable_name = self.name
        self._callable_namespace = "qamomile.static_binding.proxy"
        self._callable_policy = CallPolicy.INLINE
        self._callable_implementations: tuple[Any, ...] = ()
        self._callable_semantic_arguments: dict[str, Any] = {}

    @property
    def block(self) -> Block:
        """Return the hierarchical deferred-member wrapper body.

        Returns:
            Block: Static wrapper body.
        """
        return self._block

    def build(
        self,
        parameters: list[str] | None = None,
        **kwargs: Any,
    ) -> Block:
        """Return an owned placeholder body without accepting specialization.

        Args:
            parameters (list[str] | None): Unsupported runtime parameter names.
            **kwargs (Any): Unsupported compile-time bindings.

        Returns:
            Block: Deep-copied placeholder body.

        Raises:
            TypeError: If specialization arguments are supplied.
        """
        if parameters or kwargs:
            raise TypeError("static binding member placeholders cannot be specialized")
        return copy.deepcopy(self._block)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the deferred member inside an active trace.

        Args:
            *args (Any): Positional frontend handles.
            **kwargs (Any): Keyword frontend handles.

        Returns:
            Any: Frontend result handles matching the registered member ABI.
        """
        from qamomile.circuit.frontend.qkernel_invocation import invoke_qkernel

        return invoke_qkernel(self, *args, **kwargs)


def _deferred_member_block(
    *,
    slot_name: str,
    type_key: str,
    member_name: str,
    spec: StaticBindingMemberSpec,
) -> Block:
    """Build a wrapper block containing one body-ref invocation.

    Args:
        slot_name (str): Owning qkernel parameter name.
        type_key (str): Stable static-binding adapter key.
        member_name (str): Registered callable member name.
        spec (StaticBindingMemberSpec): Callable signature contract.

    Returns:
        Block: Hierarchical wrapper with no fallback implementation body.

    Raises:
        TypeError: If the member ABI is not a quantum-vector pass-through.
    """
    inputs: list[ValueLike] = []
    for name, annotation in spec.input_types.items():
        if annotation != Vector[Qubit]:
            raise TypeError(
                "static callable members currently require Vector[Qubit] "
                f"inputs, got {annotation!r} for {name!r}"
            )
        shape = Value(type=UIntType(), name=f"{name}_dim0")
        inputs.append(
            ArrayValue(type=QubitType(), name=name, shape=(shape,)).with_parameter(name)
        )
    if list(spec.output_types) != [Vector[Qubit]] * len(inputs):
        raise TypeError(
            "static callable members must return their quantum-vector inputs"
        )
    results = [cast(ValueLike, value.next_version()) for value in inputs]
    ref = CallableRef(
        namespace="qamomile.static_binding",
        name=f"{type_key}.{member_name}",
    )
    body_ref = CallableBodyRef(
        ref=ref,
        kind="static_binding",
        attrs={"slot": slot_name, "type_key": type_key, "member": member_name},
    )
    attrs = {
        "kind": "static_binding",
        "default_policy": CallPolicy.PRESERVE_BOX.name,
        "slot": slot_name,
        "type_key": type_key,
        "member": member_name,
    }
    definition = CallableDef(
        ref=ref,
        signature=signature_from_values(
            inputs,
            results,
            operand_names=list(spec.input_types),
        ),
        body_ref=body_ref,
        default_policy=CallPolicy.PRESERVE_BOX,
        attrs=attrs,
    )
    invoke = InvokeOperation(
        operands=inputs,
        results=results,
        target=ref,
        attrs=attrs,
        definition=definition,
    )
    return Block(
        name=f"{slot_name}.{member_name}",
        label_args=list(spec.input_types),
        input_values=inputs,
        output_values=results,
        output_names=list(spec.input_types),
        operations=[
            invoke,
            ReturnOperation(operands=cast(list[Value], results), results=[]),
        ],
        kind=BlockKind.HIERARCHICAL,
    )


__all__ = [
    "StaticBindingFieldSpec",
    "StaticBindingMemberSpec",
    "StaticBindingProxy",
    "StaticBindingSpec",
    "create_static_binding_proxy",
    "get_static_binding_by_annotation",
    "get_static_binding_by_type_key",
    "is_static_binding_annotation",
    "materialize_static_field",
    "materialize_static_member",
    "register_static_binding",
    "validate_static_binding",
    "validate_static_binding_argument",
    "validate_static_binding_slot",
    "without_static_bindings",
]
