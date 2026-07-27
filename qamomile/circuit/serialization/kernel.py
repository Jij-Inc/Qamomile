"""Static qkernel implementation reconstructed from serialized IR."""

from __future__ import annotations

import copy
import dataclasses
import inspect
from typing import Any, cast

from qamomile.circuit.frontend.func_to_block import build_param_slots, is_array_type
from qamomile.circuit.frontend.handle import Observable
from qamomile.circuit.frontend.param_validation import (
    validate_bindings_parameters_disjoint,
)
from qamomile.circuit.frontend.qkernel_build import build_specialized_block
from qamomile.circuit.frontend.qkernel_callable import qkernel_callable_def
from qamomile.circuit.frontend.qkernel_inputs import (
    auto_detect_parameters,
    create_bound_input,
    validate_kwargs,
    validate_parameters,
)
from qamomile.circuit.frontend.qkernel_utils import get_array_element_type
from qamomile.circuit.frontend.static_binding import (
    StaticBindingMemberSpec,
    StaticBindingSpec,
    get_static_binding_by_annotation,
    get_static_binding_by_type_key,
    materialize_static_field,
    materialize_static_member,
    validate_static_binding,
    validate_static_binding_slot,
)
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.effect import KernelEffect
from qamomile.circuit.ir.operation import ExpvalOp, Operation
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableImplementation,
    CallableRef,
    CallPolicy,
    CallTransform,
    CompositeGateType,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.control_flow import HasNestedOps
from qamomile.circuit.ir.operation.gate import ControlledUOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import CInitOperation
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
    ValueLike,
)
from qamomile.circuit.ir.value_mapping import ValueSubstitutor


@dataclasses.dataclass(frozen=True)
class _StaticBindingContext:
    """Hold one validated binding and its scalar-field snapshot.

    Args:
        spec (StaticBindingSpec): Installed adapter contract.
        binding (Any): Concrete object supplied through ``bindings``.
        fields (dict[str, int | float]): Scalar fields extracted exactly once.
    """

    spec: StaticBindingSpec
    binding: Any
    fields: dict[str, int | float]


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
        static_specs = {
            name: get_static_binding_by_annotation(self.input_types.get(name))
            for name in signature_names
        }
        ordinary_names = [name for name, spec in static_specs.items() if spec is None]
        if self._block.label_args != ordinary_names:
            raise ValueError(
                "serialized qkernel ordinary signature does not match Block.label_args"
            )
        if set(self.input_types) != set(signature_names):
            raise ValueError(
                "serialized qkernel input types do not match its signature"
            )
        static_slots = {slot.name: slot for slot in self._block.static_bindings}
        expected_static_names = [
            name for name, spec in static_specs.items() if spec is not None
        ]
        if list(static_slots) != expected_static_names:
            raise ValueError(
                "serialized qkernel static signature order does not match "
                "Block.static_bindings"
            )
        for name in expected_static_names:
            parameter = self.signature.parameters[name]
            if parameter.default is not inspect.Parameter.empty:
                raise ValueError(
                    f"static binding parameter {name!r} cannot have a default"
                )
            spec = static_specs[name]
            assert spec is not None
            slot = static_slots[name]
            validate_static_binding_slot(spec, slot)
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

    @property
    def effects(self) -> KernelEffect:
        """Return cached semantic effects of the preserved body.

        Returns:
            KernelEffect: Aggregated non-unitary effects.
        """
        return self._block.effects

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
        static_contexts, replacements = _materialize_static_binding_fields(
            block,
            self.input_types,
            kwargs,
        )
        runtime_values: dict[str, ValueLike] = {}
        for name, formal in zip(block.label_args, block.input_values, strict=True):
            param_type = self.input_types[name]
            is_symbolic_observable = param_type is Observable or (
                is_array_type(param_type)
                and get_array_element_type(param_type) is Observable
                and name not in kwargs
            )
            if name in selected_parameters or is_symbolic_observable:
                runtime_values[name] = formal
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

        _validate_materialized_expval_parent_indices(block)
        _StaticBindingResolver(static_contexts).resolve(block)
        block.static_bindings = ()

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


def _materialize_static_binding_fields(
    block: Block,
    input_types: dict[str, Any],
    bindings: dict[str, Any],
) -> tuple[dict[str, _StaticBindingContext], dict[str, ValueBase]]:
    """Resolve static-binding objects and their scalar field projections.

    Args:
        block (Block): Owned hierarchical block containing static slots.
        input_types (dict[str, Any]): QKernel annotations keyed by argument.
        bindings (dict[str, Any]): Compile-time bindings supplied to build.

    Returns:
        tuple[dict[str, _StaticBindingContext], dict[str, ValueBase]]:
            Validated slot contexts and UUID-keyed constant replacements.

    Raises:
        KeyError: If an installed distribution does not register a serialized
            static-binding type key or field.
        TypeError: If a concrete binding or extracted field has the wrong
            type.
        ValueError: If the serialized slot disagrees with its qkernel
            annotation or a required binding is absent.
    """
    contexts: dict[str, _StaticBindingContext] = {}
    replacements: dict[str, ValueBase] = {}
    fields_by_logical_id: dict[str, int | float] = {}
    for slot in block.static_bindings:
        annotation = input_types.get(slot.name)
        annotation_spec = get_static_binding_by_annotation(annotation)
        if annotation_spec is None:
            raise ValueError(
                f"static binding slot {slot.name!r} has no registered qkernel "
                "annotation"
            )
        type_key_spec = get_static_binding_by_type_key(slot.type_key)
        if type_key_spec is not annotation_spec:
            raise ValueError(
                f"static binding slot {slot.name!r} type key {slot.type_key!r} "
                "does not match its qkernel annotation"
            )
        validate_static_binding_slot(annotation_spec, slot)
        if slot.name not in bindings:
            raise ValueError(
                f"Static binding argument {slot.name!r} must be provided "
                "through bindings."
            )
        binding = validate_static_binding(annotation, slot.name, bindings[slot.name])
        field_values: dict[str, int | float] = {}
        for field in slot.fields:
            concrete = materialize_static_field(
                annotation_spec,
                binding,
                field.name,
            )
            field_values[field.name] = concrete
            replacements[field.value.uuid] = field.value.with_const(concrete)
            fields_by_logical_id[field.value.logical_id] = concrete
        contexts[slot.name] = _StaticBindingContext(
            spec=annotation_spec,
            binding=binding,
            fields=field_values,
        )
    for value in _collect_reachable_values(block):
        concrete = fields_by_logical_id.get(value.logical_id)
        if concrete is not None and type(value) is Value:
            replacements[value.uuid] = value.with_const(concrete)
    return contexts, replacements


def _collect_reachable_values(block: Block) -> list[ValueBase]:
    """Collect every value reachable through a hierarchical block graph.

    Args:
        block (Block): Root block whose values and owned blocks to traverse.

    Returns:
        list[ValueBase]: Values in first-visit order, deduplicated by UUID.
    """
    values: list[ValueBase] = []
    seen_values: set[str] = set()
    seen_blocks: set[int] = set()

    def visit_value(value: ValueBase) -> None:
        """Visit one structural value graph.

        Args:
            value (ValueBase): Root value to visit.
        """
        if value.uuid in seen_values:
            return
        seen_values.add(value.uuid)
        values.append(value)
        if isinstance(value, TupleValue):
            for element in value.elements:
                visit_value(element)
        elif isinstance(value, DictValue):
            for key, entry_value in value.entries:
                visit_value(key)
                visit_value(entry_value)
        elif isinstance(value, ArrayValue):
            for dimension in value.shape:
                visit_value(dimension)
            for dependency in (
                value.slice_of,
                value.slice_start,
                value.slice_step,
            ):
                if dependency is not None:
                    visit_value(dependency)
        elif isinstance(value, Value):
            if value.parent_array is not None:
                visit_value(value.parent_array)
            for index in value.element_indices:
                visit_value(index)

    def visit_operation(operation: Operation) -> None:
        """Visit one operation and all lexical or owned children.

        Args:
            operation (Operation): Operation graph node to visit.
        """
        for value in operation.all_input_values():
            visit_value(value)
        for result in operation.results:
            visit_value(result)
        if isinstance(operation, HasNestedOps):
            for region in operation.nested_regions():
                for child in region.operations:
                    visit_operation(child)
        if isinstance(operation, InvokeOperation):
            definition = operation.definition
            if definition is not None:
                if definition.body is not None:
                    visit_block(definition.body)
                for implementation in definition.implementations:
                    if implementation.body is not None:
                        visit_block(implementation.body)
        elif isinstance(operation, ControlledUOperation):
            if operation.block is not None:
                visit_block(operation.block)
        elif isinstance(operation, InverseBlockOperation):
            if operation.source_block is not None:
                visit_block(operation.source_block)
            if operation.implementation_block is not None:
                visit_block(operation.implementation_block)
        elif isinstance(operation, SelectOperation):
            for case in operation.case_blocks:
                visit_block(case)

    def visit_block(current: Block) -> None:
        """Visit one block once, including self-referential definitions.

        Args:
            current (Block): Block graph node to visit.
        """
        if id(current) in seen_blocks:
            return
        seen_blocks.add(id(current))
        for value in current.input_values:
            visit_value(value)
        for slot in current.static_bindings:
            for field in slot.fields:
                visit_value(field.value)
        for value in current.parameters.values():
            visit_value(value)
        for operation in current.operations:
            visit_operation(operation)
        for value in current.output_values:
            visit_value(value)

    visit_block(block)
    return values


def _validate_materialized_expval_parent_indices(block: Block) -> None:
    """Reject tuple-expval parent indices outside newly concrete shapes.

    Static field substitution can make an array shape concrete only after a
    serialized qkernel is bound. Deserialization therefore validates the
    carrier structure immediately and this second check closes its deferred
    index bounds before target compilation.

    Args:
        block (Block): Static-binding-specialized hierarchical block.

    Raises:
        ValueError: If a tuple-expval parent index is outside a concrete array
            shape after materialization.
    """
    values = {value.uuid: value for value in _collect_reachable_values(block)}
    seen_blocks: set[int] = set()

    def visit_operation(operation: Operation) -> None:
        """Validate one operation and recurse through every owned block.

        Args:
            operation (Operation): Operation graph node to inspect.
        """
        if isinstance(operation, ExpvalOp):
            carrier = operation.operands[0]
            if isinstance(carrier, ArrayValue):
                runtime = carrier.metadata.array_runtime
                if runtime is not None:
                    for parent_uuid, parent_index in zip(
                        runtime.element_parent_uuids,
                        runtime.element_parent_indices,
                        strict=True,
                    ):
                        if not parent_uuid:
                            continue
                        parent = values.get(parent_uuid)
                        if not isinstance(parent, ArrayValue):
                            continue
                        size = 1
                        for dimension in parent.shape:
                            concrete_dimension = dimension.get_const()
                            if type(concrete_dimension) is not int:
                                break
                            size *= concrete_dimension
                        else:
                            if parent_index >= size:
                                raise ValueError(
                                    "tuple expval parent index "
                                    f"{parent_index} is outside array width {size}"
                                )
        if isinstance(operation, HasNestedOps):
            for region in operation.nested_regions():
                for child in region.operations:
                    visit_operation(child)
        if isinstance(operation, InvokeOperation):
            definition = operation.definition
            if definition is not None:
                if definition.body is not None:
                    visit_block(definition.body)
                for implementation in definition.implementations:
                    if implementation.body is not None:
                        visit_block(implementation.body)
        elif isinstance(operation, ControlledUOperation):
            if operation.block is not None:
                visit_block(operation.block)
        elif isinstance(operation, InverseBlockOperation):
            if operation.source_block is not None:
                visit_block(operation.source_block)
            if operation.implementation_block is not None:
                visit_block(operation.implementation_block)
        elif isinstance(operation, SelectOperation):
            for case in operation.case_blocks:
                visit_block(case)

    def visit_block(current: Block) -> None:
        """Validate one block once while preserving shared graph identity.

        Args:
            current (Block): Block graph node to inspect.
        """
        if id(current) in seen_blocks:
            return
        seen_blocks.add(id(current))
        for operation in current.operations:
            visit_operation(operation)

    visit_block(block)


class _StaticBindingResolver:
    """Replace deferred static-member calls by bound qkernel definitions.

    Args:
        contexts (dict[str, _StaticBindingContext]): Validated concrete
            bindings keyed by serialized slot name.
    """

    def __init__(self, contexts: dict[str, _StaticBindingContext]) -> None:
        """Initialize a resolver and its specialization caches.

        Args:
            contexts (dict[str, _StaticBindingContext]): Validated concrete
                bindings keyed by serialized slot name.
        """
        self._contexts = contexts
        self._definitions: dict[tuple[str, str], CallableDef] = {}
        self._members: dict[
            tuple[str, str],
            tuple[Any, StaticBindingMemberSpec, dict[str, int]],
        ] = {}
        self._seen_blocks: set[int] = set()

    def resolve(self, block: Block) -> None:
        """Resolve static members and materialize final call-site widths.

        Args:
            block (Block): Root block to rewrite in place.

        Raises:
            KeyError: If a marker names an unknown registered member or width
                field.
            TypeError: If a materialized member or width field is malformed.
            ValueError: If a marker is inconsistent with its static slot or
                if an owned block declares an unknown static slot.
        """
        if self._contexts:
            # Validate deferred member widths before resolution removes their
            # static-binding markers. This same call-site traversal also
            # finalizes inverse widths already present in the payload graph.
            self._validate_block_call_widths(block, {}, {})
            self._resolve_block(block)
        else:
            # Value substitution can leave owned-block formals symbolic even
            # when their enclosing call operands are concrete. Without static
            # members, this is the single traversal that propagates those
            # call-site widths into inverse metadata.
            self._validate_block_call_widths(block, {}, {})

    def _validate_block_call_widths(
        self,
        block: Block,
        widths: dict[str, int],
        active_blocks: dict[int, dict[str, int]],
    ) -> None:
        """Validate calls and materialize reachable inverse target widths.

        Args:
            block (Block): Block whose operation graph should be inspected.
            widths (dict[str, int]): Concrete widths keyed by array logical ID.
            active_blocks (dict[int, dict[str, int]]): Input-vector widths for
                blocks active on the current recursion path.

        Raises:
            ValueError: If a concrete member-call width mismatch is found or
                a recursive edge changes the active block's input widths.
        """
        block_id = id(block)
        input_widths = {
            value.logical_id: widths[value.logical_id]
            for value in block.input_values
            if isinstance(value, ArrayValue) and value.logical_id in widths
        }
        previous_widths = active_blocks.get(block_id)
        if previous_widths is not None:
            if previous_widths != input_widths:
                raise ValueError(
                    "Recursive static binding calls cannot change quantum "
                    "vector widths."
                )
            return
        active_blocks[block_id] = input_widths
        try:
            for operation in block.operations:
                self._validate_operation_call_widths(
                    operation,
                    widths,
                    active_blocks,
                )
        finally:
            del active_blocks[block_id]

    def _validate_operation_call_widths(
        self,
        operation: Operation,
        widths: dict[str, int],
        active_blocks: dict[int, dict[str, int]],
    ) -> None:
        """Validate deferred calls and materialize inverse call-site widths.

        Args:
            operation (Operation): Operation and owned blocks to inspect.
            widths (dict[str, int]): Concrete widths keyed by array logical ID.
            active_blocks (dict[int, dict[str, int]]): Input-vector widths for
                blocks active on the current recursion path.

        Raises:
            ValueError: If a concrete member-call width is invalid or an
                owned block's call ABI is malformed.
        """
        if isinstance(operation, HasNestedOps):
            for region in operation.nested_regions():
                for item in region.operations:
                    self._validate_operation_call_widths(
                        item,
                        widths,
                        active_blocks,
                    )

        if isinstance(operation, InvokeOperation):
            if operation.body_ref is not None and (
                operation.body_ref.kind == "static_binding"
            ):
                self._validate_static_invoke_widths(operation, widths)
            definition = operation.definition
            if definition is None:
                return
            actuals_without_controls = list(
                operation.operands[operation.num_control_qubits :]
            )
            if definition.body is not None:
                self._validate_owned_block_call_widths(
                    definition.body,
                    actuals_without_controls,
                    list(operation.operands),
                    widths,
                    active_blocks,
                )
            for implementation in definition.implementations:
                if implementation.body is not None:
                    self._validate_owned_block_call_widths(
                        implementation.body,
                        actuals_without_controls,
                        list(operation.operands),
                        widths,
                        active_blocks,
                    )
            return

        if isinstance(operation, ControlledUOperation):
            if operation.block is not None:
                actuals_without_controls = list(
                    operation.operands[len(operation.control_operands) :]
                )
                self._validate_owned_block_call_widths(
                    operation.block,
                    actuals_without_controls,
                    list(operation.operands),
                    widths,
                    active_blocks,
                )
            return

        if isinstance(operation, InverseBlockOperation):
            _materialize_inverse_target_width(
                operation,
                [
                    self._concrete_array_width(target, widths)
                    if isinstance(target, ArrayValue)
                    else 1
                    for target in operation.target_qubits
                ],
            )
            actuals = [*operation.target_qubits, *operation.parameters]
            for owned_block in (
                operation.source_block,
                operation.implementation_block,
            ):
                if owned_block is not None:
                    self._validate_owned_block_call_widths(
                        owned_block,
                        actuals,
                        list(operation.operands),
                        widths,
                        active_blocks,
                    )
            return

        if isinstance(operation, SelectOperation):
            actuals = [*operation.target_operands, *operation.param_operands]
            for case_block in operation.case_blocks:
                self._validate_owned_block_call_widths(
                    case_block,
                    actuals,
                    actuals,
                    widths,
                    active_blocks,
                )

    def _validate_owned_block_call_widths(
        self,
        block: Block,
        ordinary_actuals: list[Value],
        complete_actuals: list[Value],
        parent_widths: dict[str, int],
        active_blocks: dict[int, dict[str, int]],
    ) -> None:
        """Bind call-site widths to an owned block and validate its body.

        Transform-specific implementation blocks may include control operands
        in their ABI while ordinary direct and inverse bodies omit them. The
        block arity selects the matching actual-operand view.

        Args:
            block (Block): Owned callable, control, inverse, or SELECT block.
            ordinary_actuals (list[Value]): Actuals with outer controls
                removed.
            complete_actuals (list[Value]): Complete operation operands.
            parent_widths (dict[str, int]): Caller width environment.
            active_blocks (dict[int, dict[str, int]]): Input-vector widths for
                blocks active on the recursion path.

        """
        if len(block.input_values) == len(ordinary_actuals):
            actuals = ordinary_actuals
        elif len(block.input_values) == len(complete_actuals):
            actuals = complete_actuals
        else:
            raise ValueError(
                f"owned block {block.name!r} expects {len(block.input_values)} "
                "inputs, but its call site provides "
                f"{len(ordinary_actuals)} ordinary or "
                f"{len(complete_actuals)} complete operands"
            )
        child_widths = dict(parent_widths)
        for formal, actual in zip(block.input_values, actuals, strict=True):
            if not isinstance(formal, ArrayValue):
                continue
            width = self._concrete_array_width(actual, parent_widths)
            if width is not None:
                child_widths[formal.logical_id] = width
        self._validate_block_call_widths(block, child_widths, active_blocks)

    @staticmethod
    def _concrete_array_width(
        value: ValueLike,
        widths: dict[str, int],
    ) -> int | None:
        """Return a call-site array width when it is compile-time concrete.

        Args:
            value (ValueLike): Actual array value to inspect.
            widths (dict[str, int]): Widths inherited from enclosing calls.

        Returns:
            int | None: Product of concrete shape dimensions, an inherited
            formal width, or ``None`` when the shape remains symbolic.
        """
        if not isinstance(value, ArrayValue):
            return None
        inherited = widths.get(value.logical_id)
        if inherited is not None:
            return inherited
        return _concrete_quantum_width(value)

    def _validate_static_invoke_widths(
        self,
        operation: InvokeOperation,
        widths: dict[str, int],
    ) -> None:
        """Validate one marker call against its descriptor-declared widths.

        A mismatching call is first traced through the concrete member at the
        caller's widths. This deliberately reuses the member's public width
        validator, preserving producer-specific diagnostics such as the Pauli
        LCU ``signal`` and ``system`` messages. A generic error is used only
        when a registered member fails to enforce its declared contract.

        Args:
            operation (InvokeOperation): Deferred static-member invocation.
            widths (dict[str, int]): Concrete widths inherited from outer
                callable/control/SELECT boundaries.

        Raises:
            TypeError: If the marker call does not use vector operands.
            ValueError: If an operand count or concrete width is invalid.
        """
        slot_name, _, member_name = self._static_marker(operation)
        member, member_spec, expected_sizes = self._member_contract(
            slot_name,
            member_name,
        )
        input_names = list(member_spec.input_types)
        targets = operation.target_qubits
        if operation.parameters:
            raise ValueError(
                f"static member {member_name!r} cannot have classical operands"
            )
        if len(targets) != len(input_names):
            raise ValueError(
                f"static member {member_name!r} expects {len(input_names)} "
                f"target operands, got {len(targets)}"
            )
        target_results = operation.results[operation.num_control_qubits :]
        if len(target_results) != len(member_spec.output_types) or any(
            not isinstance(result, ArrayValue) for result in target_results
        ):
            raise ValueError(
                f"static member {member_name!r} results do not match its "
                "registered quantum-vector ABI"
            )
        call_sizes: dict[str, int] = {}
        for input_name, target in zip(input_names, targets, strict=True):
            if not isinstance(target, ArrayValue):
                raise TypeError(
                    f"static member {member_name!r} input {input_name!r} "
                    "must be a quantum vector"
                )
            actual = self._concrete_array_width(target, widths)
            if actual is not None:
                call_sizes[input_name] = actual

        mismatches = [
            (input_name, expected_sizes[input_name], actual)
            for input_name, actual in call_sizes.items()
            if input_name in expected_sizes and actual != expected_sizes[input_name]
        ]
        if not mismatches:
            return

        specialization_sizes = dict(expected_sizes)
        specialization_sizes.update(call_sizes)
        build_specialized_block(
            member,
            parameters=[],
            bindings={},
            qubit_sizes=specialization_sizes,
        )
        input_name, expected, actual = mismatches[0]
        raise ValueError(
            f"static member {member_name!r} requires {expected} qubits for "
            f"{input_name!r}, got {actual}"
        )

    def _resolve_block(self, block: Block) -> None:
        """Resolve one owned block exactly once.

        Args:
            block (Block): Owned block to inspect and rewrite.

        Raises:
            ValueError: If the block declares an unavailable static slot.
        """
        block_id = id(block)
        if block_id in self._seen_blocks:
            return
        self._seen_blocks.add(block_id)
        unknown_slots = {
            slot.name
            for slot in block.static_bindings
            if slot.name not in self._contexts
        }
        if unknown_slots:
            raise ValueError(
                "owned block refers to unavailable static binding slots: "
                f"{sorted(unknown_slots)!r}"
            )
        block.operations = [
            self._resolve_operation(operation) for operation in block.operations
        ]

    def _resolve_operation(self, operation: Operation) -> Operation:
        """Resolve deferred members in one operation and every owned region.

        Args:
            operation (Operation): Operation to rewrite.

        Returns:
            Operation: Rewritten operation, including rebuilt control-flow
            regions when present.
        """
        if isinstance(operation, HasNestedOps):
            operation = operation.rebuild_regions(
                tuple(
                    dataclasses.replace(
                        region,
                        operations=tuple(
                            self._resolve_operation(item) for item in region.operations
                        ),
                    )
                    for region in operation.nested_regions()
                )
            )

        if isinstance(operation, InvokeOperation):
            if operation.body_ref is not None and (
                operation.body_ref.kind == "static_binding"
            ):
                self._resolve_static_invoke(operation)
            if operation.definition is not None:
                self._resolve_definition(operation.definition)
        elif isinstance(operation, ControlledUOperation):
            if operation.block is not None:
                self._resolve_block(operation.block)
        elif isinstance(operation, InverseBlockOperation):
            if operation.source_block is not None:
                self._resolve_block(operation.source_block)
            if operation.implementation_block is not None:
                self._resolve_block(operation.implementation_block)
        elif isinstance(operation, SelectOperation):
            for case_block in operation.case_blocks:
                self._resolve_block(case_block)
        return operation

    def _resolve_definition(self, definition: CallableDef) -> None:
        """Resolve blocks owned by one callable definition.

        Args:
            definition (CallableDef): Definition whose bodies are reachable.

        Raises:
            ValueError: If a static body reference appears outside its marker
                invocation.
        """
        if definition.body_ref is not None and (
            definition.body_ref.kind == "static_binding"
        ):
            raise ValueError(
                "static binding body references must be attached to a "
                "deferred member invocation"
            )
        if definition.body is not None:
            self._resolve_block(definition.body)
        for implementation in definition.implementations:
            self._resolve_implementation(implementation)

    def _resolve_implementation(
        self,
        implementation: CallableImplementation,
    ) -> None:
        """Resolve one callable implementation body.

        Args:
            implementation (CallableImplementation): Implementation to visit.

        Raises:
            ValueError: If an implementation directly carries a static body
                reference instead of a marker invocation.
        """
        if implementation.body_ref is not None and (
            implementation.body_ref.kind == "static_binding"
        ):
            raise ValueError(
                "static binding implementation references require an "
                "InvokeOperation marker"
            )
        if implementation.body is not None:
            self._resolve_block(implementation.body)

    def _static_marker(
        self,
        operation: InvokeOperation,
    ) -> tuple[str, str, str]:
        """Validate and decode one deferred static-member marker.

        Args:
            operation (InvokeOperation): Invocation carrying the marker.

        Returns:
            tuple[str, str, str]: Slot name, registered type key, and member
            name.

        Raises:
            ValueError: If the invocation definition, references, attributes,
                or registered slot identity are inconsistent.
        """
        definition = operation.definition
        body_ref = operation.body_ref
        if definition is None or body_ref is None:
            raise ValueError("static binding invocation is missing its definition")
        if body_ref.kind != "static_binding":
            raise ValueError("invocation does not carry a static binding marker")
        if definition.body_ref != body_ref:
            raise ValueError(
                "static binding marker must be the definition's default body reference"
            )
        if definition.body is not None:
            raise ValueError("static binding invocation cannot also own a body")
        if body_ref.ref != definition.ref or operation.target != definition.ref:
            raise ValueError("static binding invocation has inconsistent references")

        marker = body_ref.attrs
        if set(marker) != {"slot", "type_key", "member"}:
            raise ValueError(
                "static binding marker requires exactly slot, type_key, and member"
            )
        if not all(isinstance(value, str) and value for value in marker.values()):
            raise ValueError("static binding marker attributes must be strings")
        slot_name = cast(str, marker["slot"])
        type_key = cast(str, marker["type_key"])
        member_name = cast(str, marker["member"])
        context = self._contexts.get(slot_name)
        if context is None:
            raise ValueError(f"unknown static binding slot {slot_name!r}")
        spec = context.spec
        if spec.type_key != type_key:
            raise ValueError(
                f"static binding marker type {type_key!r} does not match "
                f"slot {slot_name!r}"
            )
        for owner, attrs in (
            ("invocation", operation.attrs),
            ("definition", definition.attrs),
        ):
            if attrs.get("kind") != "static_binding":
                raise ValueError(
                    f"static binding {owner} kind must be 'static_binding'"
                )
            for key, expected in marker.items():
                if attrs.get(key) != expected:
                    raise ValueError(
                        f"static binding {owner} attribute {key!r} disagrees "
                        "with its body reference"
                    )
        if definition.default_policy is not CallPolicy.PRESERVE_BOX:
            raise ValueError("static binding definitions must use PRESERVE_BOX policy")
        return slot_name, type_key, member_name

    def _member_contract(
        self,
        slot_name: str,
        member_name: str,
    ) -> tuple[Any, StaticBindingMemberSpec, dict[str, int]]:
        """Return a concrete member and its descriptor-declared widths.

        Args:
            slot_name (str): Owning static-binding slot.
            member_name (str): Registered callable member name.

        Returns:
            tuple[Any, StaticBindingMemberSpec, dict[str, int]]: Concrete
            qkernel-like member, registered ABI, and expected vector widths.

        Raises:
            KeyError: If the member or a declared width field is unknown.
            TypeError: If a declared qubit width is not an integer.
            ValueError: If a width is declared for an unknown member input.
        """
        key = (slot_name, member_name)
        cached = self._members.get(key)
        if cached is not None:
            return cached
        context = self._contexts[slot_name]
        spec = context.spec
        member, member_spec = materialize_static_member(
            spec,
            context.binding,
            member_name,
        )
        expected_sizes: dict[str, int] = {}
        for input_name, field_name in member_spec.qubit_width_fields.items():
            if input_name not in member_spec.input_types:
                raise ValueError(
                    f"static member {member_name!r} declares width for "
                    f"unknown input {input_name!r}"
                )
            width = context.fields[field_name]
            if type(width) is not int:
                raise TypeError(
                    f"static member width field {field_name!r} must be an int"
                )
            if width <= 0:
                raise ValueError(
                    f"static member width field {field_name!r} must be positive"
                )
            expected_sizes[input_name] = width
        contract = (member, member_spec, expected_sizes)
        self._members[key] = contract
        return contract

    def _resolve_static_invoke(self, operation: InvokeOperation) -> None:
        """Replace one deferred member invocation by its concrete definition.

        Args:
            operation (InvokeOperation): Static marker invocation to replace.

        Raises:
            KeyError: If the marker names an unknown member.
            ValueError: If marker metadata disagrees with its slot contract.
        """
        slot_name, _, member_name = self._static_marker(operation)
        concrete_definition = self._definition_for(
            slot_name,
            member_name,
            require_inverse=operation.transform is CallTransform.INVERSE,
        )
        resolved_attrs = dict(concrete_definition.attrs)
        for key in (
            "num_control_qubits",
            "num_target_qubits",
            "control_value",
            "strategy_name",
        ):
            if key in operation.attrs:
                resolved_attrs[key] = operation.attrs[key]
        operation.target = concrete_definition.ref
        operation.attrs = resolved_attrs
        operation.definition = concrete_definition

    def _definition_for(
        self,
        slot_name: str,
        member_name: str,
        *,
        require_inverse: bool,
    ) -> CallableDef:
        """Build or reuse a concrete callable definition for one member.

        Args:
            slot_name (str): Owning static-binding slot.
            member_name (str): Registered callable member name.
            require_inverse (bool): Whether an inverse implementation must be
                materialized for this call site.

        Returns:
            CallableDef: Concrete specialized callable definition.

        Raises:
            KeyError: If the member or a declared width field is unknown.
            TypeError: If a declared qubit width is not an integer.
        """
        key = (slot_name, member_name)
        definition = self._definitions.get(key)
        if definition is None:
            member, _, expected_sizes = self._member_contract(
                slot_name,
                member_name,
            )
            concrete_block = build_specialized_block(
                member,
                parameters=[],
                bindings={},
                qubit_sizes=expected_sizes,
            )
            definition = qkernel_callable_def(member, concrete_block)
            self._definitions[key] = definition

        if (
            require_inverse
            and definition.implementation_for(transform=CallTransform.INVERSE) is None
        ):
            from qamomile.circuit.frontend.operation.inverse import _BlockInverter

            if definition.body is None:
                raise ValueError(
                    f"static member {member_name!r} has no direct body to invert"
                )
            definition.implementations.append(
                CallableImplementation(
                    transform=CallTransform.INVERSE,
                    body=_BlockInverter().invert_block(definition.body),
                )
            )
        return definition


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


def _concrete_quantum_width(value: ValueBase) -> int | None:
    """Return the scalar qubit width of one concrete quantum value.

    Args:
        value (ValueBase): Scalar or array quantum value to inspect.

    Returns:
        int | None: Scalar qubit width, or ``None`` when an array dimension
            remains symbolic.
    """
    if not isinstance(value, ArrayValue):
        return 1
    if not value.shape:
        return None
    width = 1
    for dimension in value.shape:
        concrete = dimension.get_const()
        if type(concrete) is not int:
            return None
        width *= concrete
    return width


def _materialize_inverse_target_width(
    operation: InverseBlockOperation,
    target_widths: list[int | None],
) -> None:
    """Refresh inverse scalar-width metadata when every target is concrete.

    Args:
        operation (InverseBlockOperation): Inverse operation to update in
            place.
        target_widths (list[int | None]): Scalar widths resolved from the
            operation values or their enclosing call-site environment.
    """
    if target_widths and all(width is not None for width in target_widths):
        operation.num_target_qubits = sum(cast(int, width) for width in target_widths)


def _replace_operations(
    operations: list[Operation],
    substitutor: ValueSubstitutor,
    bound_formal_uuids: set[str],
    replacements: dict[str, ValueBase],
    block_cache: dict[int, Block],
) -> list[Operation]:
    """Replace formal values recursively in an operation region.

    Args:
        operations (list[Operation]): Operations in the current region.
        substitutor (ValueSubstitutor): Shared value substitutor.
        bound_formal_uuids (set[str]): Original formal UUIDs bound to concrete
            values.
        replacements (dict[str, ValueBase]): Global UUID substitutions used by
            lexically captured owned blocks.
        block_cache (dict[int, Block]): Original and rewritten block identities
            mapped to their rewritten block.

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
            nested = tuple(
                dataclasses.replace(
                    region,
                    operations=tuple(
                        _replace_operations(
                            list(region.operations),
                            substitutor,
                            bound_formal_uuids,
                            replacements,
                            block_cache,
                        )
                    ),
                )
                for region in new_operation.nested_regions()
            )
            new_operation = new_operation.rebuild_regions(nested)
        _replace_owned_blocks(new_operation, replacements, block_cache)
        rewritten.append(new_operation)
    return rewritten


def _replace_owned_blocks(
    operation: Operation,
    replacements: dict[str, ValueBase],
    block_cache: dict[int, Block],
) -> None:
    """Rewrite lexical binding captures in blocks owned by an operation.

    Args:
        operation (Operation): Rewritten operation whose owned blocks should
            be updated in place.
        replacements (dict[str, ValueBase]): Global UUID substitutions.
        block_cache (dict[int, Block]): Cycle-safe rewritten block cache.
    """
    if isinstance(operation, InvokeOperation):
        definition = operation.definition
        if definition is None:
            return
        if definition.body is not None:
            definition.body = _replace_formal_values(
                definition.body,
                replacements,
                block_cache,
            )
        for implementation in definition.implementations:
            if implementation.body is not None:
                implementation.body = _replace_formal_values(
                    implementation.body,
                    replacements,
                    block_cache,
                )
        return
    if isinstance(operation, ControlledUOperation):
        if operation.block is not None:
            operation.block = _replace_formal_values(
                operation.block,
                replacements,
                block_cache,
            )
        return
    if isinstance(operation, InverseBlockOperation):
        if operation.source_block is not None:
            operation.source_block = _replace_formal_values(
                operation.source_block,
                replacements,
                block_cache,
            )
        if operation.implementation_block is not None:
            operation.implementation_block = _replace_formal_values(
                operation.implementation_block,
                replacements,
                block_cache,
            )
        return
    if isinstance(operation, SelectOperation):
        operation.case_blocks = [
            _replace_formal_values(case, replacements, block_cache)
            for case in operation.case_blocks
        ]


def _replace_formal_values(
    block: Block,
    replacements: dict[str, ValueBase],
    block_cache: dict[int, Block] | None = None,
) -> Block:
    """Return a block whose compile-time formals are concrete values.

    Args:
        block (Block): Owned static body to specialize.
        replacements (dict[str, ValueBase]): Formal UUID replacements.
        block_cache (dict[int, Block] | None): Cycle-safe rewritten block
            cache. Defaults to a new cache for the root call.

    Returns:
        Block: Specialized block with all entrypoint references updated.
    """
    if block_cache is None:
        block_cache = {}
    cached = block_cache.get(id(block))
    if cached is not None:
        return cached

    substitutor = ValueSubstitutor(replacements)
    rewritten = dataclasses.replace(
        block,
        input_values=[
            cast(ValueLike, substitutor.substitute_value(value))
            for value in block.input_values
        ],
        output_values=[
            cast(ValueLike, substitutor.substitute_value(value))
            for value in block.output_values
        ],
        parameters={
            name: cast(Value, substitutor.substitute_value(value))
            for name, value in block.parameters.items()
        },
        operations=[],
    )
    block_cache[id(block)] = rewritten
    block_cache[id(rewritten)] = rewritten
    rewritten.operations = _replace_operations(
        block.operations,
        substitutor,
        set(replacements),
        replacements,
        block_cache,
    )
    return rewritten
