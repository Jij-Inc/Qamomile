"""Static qkernel implementation reconstructed from serialized IR."""

from __future__ import annotations

import copy
import dataclasses
import inspect
from typing import Any, cast

from qamomile.circuit.frontend.func_to_block import build_param_slots
from qamomile.circuit.frontend.param_validation import (
    validate_bindings_parameters_disjoint,
)
from qamomile.circuit.frontend.qkernel_inputs import (
    auto_detect_parameters,
    create_bound_input,
    validate_kwargs,
    validate_parameters,
)
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableRef,
    CallPolicy,
    CompositeGateType,
)
from qamomile.circuit.ir.operation.control_flow import HasNestedOps
from qamomile.circuit.ir.operation.operation import CInitOperation
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
    ValueLike,
)
from qamomile.circuit.ir.value_mapping import ValueSubstitutor


@dataclasses.dataclass
class SerializedQKernel:
    """A qkernel whose Python-independent semantic IR was deserialized.

    The object implements the compiler-facing qkernel protocol. Its body is a
    static, unbound hierarchical block, so :meth:`build` specializes existing
    IR values instead of executing the original Python function again.

    Attributes:
        name (str): User-facing qkernel name.
        signature (inspect.Signature): Reconstructed Python call signature.
        input_types (dict[str, Any]): Frontend input annotations by name.
        output_types (list[Any]): Frontend output annotations.
    """

    name: str
    signature: inspect.Signature
    input_types: dict[str, Any]
    output_types: list[Any]
    _block: Block
    _callable_definition: CallableDef
    _callable_kind: str = dataclasses.field(init=False, repr=False)
    _callable_name: str = dataclasses.field(init=False, repr=False)
    _callable_namespace: str | None = dataclasses.field(init=False, repr=False)
    _callable_policy: CallPolicy = dataclasses.field(init=False, repr=False)
    _callable_gate_type: CompositeGateType = dataclasses.field(
        init=False,
        repr=False,
    )
    _callable_implementations: tuple[Any, ...] = dataclasses.field(
        init=False,
        repr=False,
    )
    _callable_semantic_arguments: dict[str, Any] = dataclasses.field(
        init=False,
        repr=False,
    )
    _callable_ref_override: CallableRef = dataclasses.field(
        init=False,
        repr=False,
    )
    _callable_attrs_override: dict[str, Any] = dataclasses.field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate and own the static qkernel body.

        Raises:
            ValueError: If the body, interface, or root callable definition is
                inconsistent with a static unbound qkernel.
        """
        if self._block.kind is not BlockKind.HIERARCHICAL:
            raise ValueError(
                "SerializedQKernel requires a HIERARCHICAL static body, got "
                f"{self._block.kind}"
            )
        signature_names = list(self.signature.parameters)
        if self._block.label_args != signature_names:
            raise ValueError(
                "serialized qkernel signature does not match Block.label_args"
            )
        if set(self.input_types) != set(signature_names):
            raise ValueError(
                "serialized qkernel input types do not match its signature"
            )
        if self._block.parameters:
            raise ValueError("SerializedQKernel requires an unbound body")

        definition = self._callable_definition
        if (
            definition.signature is not None
            or definition.body is not None
            or definition.body_ref is not None
        ):
            raise ValueError(
                "root callable definition must not duplicate the qkernel "
                "signature or body"
            )
        kind = definition.attrs.get("kind")
        if kind not in {"qkernel", "composite"}:
            raise ValueError(f"unknown root callable kind {kind!r}")
        origin_qualified = definition.attrs.get("origin_qualified")
        if not isinstance(origin_qualified, bool):
            raise ValueError("root callable origin_qualified must be a bool")
        attr_policy = definition.attrs.get("default_policy")
        if attr_policy != definition.default_policy.name:
            raise ValueError(
                "root callable attrs and definition disagree on default_policy"
            )
        raw_gate_type = definition.attrs.get("gate_type", "CUSTOM")
        if not isinstance(raw_gate_type, str):
            raise ValueError("root callable gate_type must be a string")
        try:
            gate_type = CompositeGateType[raw_gate_type]
        except KeyError as exc:
            raise ValueError(
                f"unknown root callable gate_type {raw_gate_type!r}"
            ) from exc
        semantic_arguments = definition.attrs.get("semantic_arguments", {})
        if not isinstance(semantic_arguments, dict):
            raise ValueError("root callable semantic_arguments must be a dict")
        if kind == "composite":
            custom_name = definition.attrs.get("custom_name")
            if custom_name != definition.ref.name:
                raise ValueError(
                    "root composite custom_name must match its callable reference"
                )

        self._block, self._callable_definition = copy.deepcopy(
            (self._block, self._callable_definition)
        )
        definition = self._callable_definition
        self._callable_kind = kind
        self._callable_name = definition.ref.name
        self._callable_namespace = (
            None if origin_qualified else definition.ref.namespace
        )
        self._callable_policy = definition.default_policy
        self._callable_gate_type = gate_type
        self._callable_implementations = tuple(definition.implementations)
        self._callable_semantic_arguments = copy.deepcopy(semantic_arguments)
        self._callable_ref_override = definition.ref
        self._callable_attrs_override = dict(definition.attrs)
        # Invocation specialization normally re-executes the Python function.
        # Keeping this guard true selects the preserved semantic block instead.
        self._block_building = False
        self._specializing = True

    @property
    def block(self) -> Block:
        """Return the cached unbound hierarchical body.

        Returns:
            Block: Static qkernel body.
        """
        return self._block

    def build(
        self,
        parameters: list[str] | None = None,
        **kwargs: Any,
    ) -> Block:
        """Specialize the preserved IR for bindings and runtime parameters.

        This method mirrors the public qkernel argument contract while avoiding
        Python re-tracing. Compile-time values replace the corresponding formal
        IR values; runtime parameters remain symbolic.

        Args:
            parameters (list[str] | None): Names retained as backend runtime
                parameters. ``None`` auto-detects unbound parameterizable
                arguments.
            **kwargs (Any): Compile-time bindings keyed by qkernel argument.

        Returns:
            Block: Specialized traced-stage body accepted by the normal
                Qamomile compiler pipeline.

        Raises:
            TypeError: If a requested parameter or binding type is unsupported.
            ValueError: If arguments are unknown, missing, or present in both
                ``parameters`` and ``kwargs``.
        """
        validate_bindings_parameters_disjoint(kwargs, parameters)
        selected_parameters = (
            auto_detect_parameters(self.signature, self.input_types, kwargs)
            if parameters is None
            else list(parameters)
        )
        validate_parameters(self.input_types, selected_parameters)
        validate_kwargs(
            self.signature,
            self.input_types,
            selected_parameters,
            kwargs,
        )

        block = copy.deepcopy(self._block)
        replacements: dict[str, ValueBase] = {}
        runtime_values: dict[str, Value] = {}
        for name, formal in zip(block.label_args, block.input_values, strict=True):
            if name in selected_parameters:
                runtime_values[name] = cast(Value, formal)
                continue

            parameter = self.signature.parameters[name]
            if name in kwargs:
                concrete = kwargs[name]
            elif parameter.default is not inspect.Parameter.empty:
                concrete = parameter.default
            else:
                continue
            bound = create_bound_input(self.input_types[name], name, concrete)
            _collect_formal_replacements(
                cast(ValueBase, formal),
                cast(ValueBase, bound.value),
                replacements,
            )

        if replacements:
            block = _replace_formal_values(block, replacements)

        block.parameters = {
            name: cast(Value, _replace_value(value, replacements))
            for name, value in runtime_values.items()
        }
        block.param_slots = build_param_slots(
            signature=self.signature,
            input_types=self.input_types,
            parameters=selected_parameters,
            kwargs=kwargs,
            bind_defaults=True,
        )
        block.kind = BlockKind.TRACED
        return block

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the deserialized qkernel inside an active trace.

        Args:
            *args (Any): Positional frontend handles or promoted literals.
            **kwargs (Any): Keyword frontend handles or promoted literals.

        Returns:
            Any: Frontend handle result matching ``output_types``.
        """
        from qamomile.circuit.frontend.qkernel_invocation import invoke_qkernel

        return invoke_qkernel(self, *args, **kwargs)


def _replace_value(
    value: ValueLike,
    replacements: dict[str, ValueBase],
) -> ValueLike:
    """Replace one structural value through a substitution mapping.

    Args:
        value (ValueLike): Value to rewrite.
        replacements (dict[str, ValueBase]): UUID-keyed replacements.

    Returns:
        ValueLike: Rewritten value.
    """
    return cast(ValueLike, ValueSubstitutor(replacements).substitute_value(value))


def _collect_formal_replacements(
    formal: ValueBase,
    concrete: ValueBase,
    replacements: dict[str, ValueBase],
) -> None:
    """Map a formal value and its structural children to concrete values.

    Array shape dimensions are independent IR Values and may be referenced
    directly by loop bounds or allocation sizes. Mapping only the ArrayValue
    therefore leaves those uses symbolic. Tuple and dictionary children follow
    the same recursive identity rule.

    Args:
        formal (ValueBase): Static qkernel formal value.
        concrete (ValueBase): Concrete frontend value for the binding.
        replacements (dict[str, ValueBase]): Mapping updated in place.

    Returns:
        None: ``replacements`` is mutated.
    """
    replacements[formal.uuid] = concrete
    if isinstance(formal, ArrayValue) and isinstance(concrete, ArrayValue):
        for formal_dim, concrete_dim in zip(
            formal.shape,
            concrete.shape,
            strict=False,
        ):
            _collect_formal_replacements(formal_dim, concrete_dim, replacements)
        return
    if isinstance(formal, TupleValue) and isinstance(concrete, TupleValue):
        for formal_item, concrete_item in zip(
            formal.elements,
            concrete.elements,
            strict=False,
        ):
            _collect_formal_replacements(formal_item, concrete_item, replacements)
        return
    if isinstance(formal, DictValue) and isinstance(concrete, DictValue):
        for (formal_key, formal_value), (concrete_key, concrete_value) in zip(
            formal.entries,
            concrete.entries,
            strict=False,
        ):
            _collect_formal_replacements(formal_key, concrete_key, replacements)
            _collect_formal_replacements(formal_value, concrete_value, replacements)


def _replace_operations(
    operations: list[Operation],
    substitutor: ValueSubstitutor,
    bound_formal_uuids: set[str],
) -> list[Operation]:
    """Replace formal values recursively in an operation region.

    Args:
        operations (list[Operation]): Operations in the current region.
        substitutor (ValueSubstitutor): Shared value substitutor.
        bound_formal_uuids (set[str]): Original formal UUIDs bound to concrete
            values.

    Returns:
        list[Operation]: Rewritten operations without obsolete classical input
            initialization nodes.
    """
    rewritten: list[Operation] = []
    for operation in operations:
        if isinstance(operation, CInitOperation) and any(
            result.uuid in bound_formal_uuids for result in operation.results
        ):
            continue
        new_operation = substitutor.substitute_operation(operation)
        if isinstance(new_operation, HasNestedOps):
            nested = [
                _replace_operations(items, substitutor, bound_formal_uuids)
                for items in new_operation.nested_op_lists()
            ]
            new_operation = new_operation.rebuild_nested(nested)
        rewritten.append(new_operation)
    return rewritten


def _replace_formal_values(
    block: Block,
    replacements: dict[str, ValueBase],
) -> Block:
    """Return a block whose compile-time formals are concrete values.

    Args:
        block (Block): Owned static body to specialize.
        replacements (dict[str, ValueBase]): Formal UUID replacements.

    Returns:
        Block: Specialized block with all entrypoint references updated.
    """
    substitutor = ValueSubstitutor(replacements)
    return dataclasses.replace(
        block,
        input_values=[
            cast(ValueLike, substitutor.substitute_value(value))
            for value in block.input_values
        ],
        output_values=[
            cast(ValueLike, substitutor.substitute_value(value))
            for value in block.output_values
        ],
        operations=_replace_operations(
            block.operations,
            substitutor,
            set(replacements),
        ),
    )
