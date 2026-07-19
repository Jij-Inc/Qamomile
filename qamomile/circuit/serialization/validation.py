"""Semantic validation for static qkernel IR at the serialization boundary."""

from __future__ import annotations

import dataclasses
from typing import Iterable, cast

from qamomile._utils import is_plain_int
from qamomile.circuit.frontend.handle import Qubit, Vector
from qamomile.circuit.frontend.static_binding import (
    StaticBindingMemberSpec,
    StaticBindingSpec,
    get_static_binding_by_type_key,
    validate_static_binding_slot,
)
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import (
    ExpvalOp,
    ForItemsOperation,
    GateOperation,
    GateOperationType,
    GlobalPhaseOperation,
    InverseBlockOperation,
    InvokeOperation,
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
    ProjectOperation,
    ResetOperation,
    ReturnOperation,
    SelectOperation,
)
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    CompOp,
    CompOpKind,
    CondOp,
    NotOp,
    RuntimeClassicalExpr,
    RuntimeOpKind,
)
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableRef,
    CallPolicy,
    CallTransform,
)
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.classical_ops import (
    DecodeQFixedOperation,
    DictGetItemOperation,
    ReturnQuantumArrayElementOperation,
    StoreArrayElementOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    HasNestedOps,
    IfOperation,
    RegionArg,
    WhileOperation,
)
from qamomile.circuit.ir.operation.control_value import normalize_control_value
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    SymbolicControlledU,
)
from qamomile.circuit.ir.operation.operation import (
    CInitOperation,
    Operation,
    ParamHint,
    QInitOperation,
)
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.slice_array import (
    ReleaseSliceViewOperation,
    SliceArrayOperation,
)
from qamomile.circuit.ir.static_binding import StaticBindingSlot
from qamomile.circuit.ir.types.hamiltonian import ObservableType
from qamomile.circuit.ir.types.primitives import (
    BitType,
    FloatType,
    QubitType,
    UIntType,
    ValueType,
)
from qamomile.circuit.ir.types.q_register import QFixedType
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
    ValueLike,
    collect_value_like_uuids,
    resolve_root_qubit_address,
)
from qamomile.circuit.transpiler.block_parameter_binding import pair_block_operands

_SINGLE_QUBIT_GATES = frozenset(
    {
        GateOperationType.H,
        GateOperationType.X,
        GateOperationType.Y,
        GateOperationType.Z,
        GateOperationType.T,
        GateOperationType.TDG,
        GateOperationType.S,
        GateOperationType.SDG,
    }
)
_SINGLE_QUBIT_ROTATIONS = frozenset(
    {
        GateOperationType.P,
        GateOperationType.RX,
        GateOperationType.RY,
        GateOperationType.RZ,
    }
)
_TWO_QUBIT_GATES = frozenset(
    {GateOperationType.CX, GateOperationType.CZ, GateOperationType.SWAP}
)
_TWO_QUBIT_ROTATIONS = frozenset({GateOperationType.CP, GateOperationType.RZZ})


@dataclasses.dataclass
class _ValidationState:
    """Mutable state shared while validating a complete callable graph."""

    seen_blocks: set[int] = dataclasses.field(default_factory=set)
    seen_definitions: set[int] = dataclasses.field(default_factory=set)
    static_bindings: dict[
        str,
        tuple[StaticBindingSlot, StaticBindingSpec],
    ] = dataclasses.field(default_factory=dict)
    static_field_uuids: set[str] = dataclasses.field(default_factory=set)
    static_field_logical_ids: dict[str, str] = dataclasses.field(default_factory=dict)
    root_block_id: int | None = None


def validate_qkernel_ir(block: Block) -> None:
    """Validate every operation and nested region reachable from a qkernel.

    Args:
        block (Block): Root unbound hierarchical qkernel block.

    Raises:
        ValueError: If an operation violates its arity, type, SSA, callable,
            or region contract.
    """
    state = _ValidationState()
    _register_root_static_bindings(block, state)
    _validate_block(block, state, "qkernel body")


def _register_root_static_bindings(
    block: Block,
    state: _ValidationState,
) -> None:
    """Validate and register the root qkernel's static-binding manifest.

    Args:
        block (Block): Root qkernel block.
        state (_ValidationState): Shared graph-validation state to populate.

    Raises:
        ValueError: If a slot uses an unknown adapter or disagrees with the
            installed adapter contract.
    """
    state.root_block_id = id(block)
    for slot in block.static_bindings:
        try:
            spec = get_static_binding_by_type_key(slot.type_key)
        except KeyError as exc:
            raise ValueError(f"unknown static binding type {slot.type_key!r}") from exc
        validate_static_binding_slot(spec, slot)
        state.static_bindings[slot.name] = (slot, spec)
        for field in slot.fields:
            previous = state.static_field_logical_ids.get(field.value.logical_id)
            if previous is not None and previous != field.value.uuid:
                raise ValueError(
                    "static binding fields must have unique logical identities"
                )
            state.static_field_uuids.add(field.value.uuid)
            state.static_field_logical_ids[field.value.logical_id] = field.value.uuid


def _validate_block(block: Block, state: _ValidationState, location: str) -> None:
    """Validate one block and every graph edge reachable from it.

    Args:
        block (Block): Block to validate.
        state (_ValidationState): Shared graph-validation state.
        location (str): Human-readable graph location.

    Raises:
        ValueError: If the block contains malformed values or operations.
    """
    if id(block) in state.seen_blocks:
        return
    state.seen_blocks.add(id(block))
    if id(block) != state.root_block_id and block.static_bindings:
        raise ValueError(
            f"{location} cannot declare static bindings; only the root "
            "qkernel owns binding slots"
        )
    if id(block) != state.root_block_id:
        nested_formal_uuids = {
            uuid
            for value in block.input_values
            for uuid in collect_value_like_uuids(value)
        }
        aliases = nested_formal_uuids & state.static_field_uuids
        if aliases:
            raise ValueError(
                f"{location} aliases root static binding fields as owned inputs: "
                f"{sorted(aliases)!r}"
            )
    _require_unique_values(block.input_values, f"{location} inputs")
    _require_unique_values(block.output_values, f"{location} outputs")
    producers = {
        field.value.uuid: (
            f"{location} static binding {slot.name!r} field {field.name!r}"
        )
        for slot in block.static_bindings
        for field in slot.fields
    }

    for index, operation in enumerate(block.operations):
        op_location = f"{location} operation {index} ({type(operation).__name__})"
        _validate_operation(operation, state, producers, op_location)
    if state.static_bindings:
        _validate_value_producer_completeness(block, state, location)


def _validate_value_producer_completeness(
    block: Block,
    state: _ValidationState,
    location: str,
) -> None:
    """Require every symbolic value reference to have a local producer.

    The check is intentionally scope-aware but order-independent. It rejects
    dangling references and values borrowed from another owned block while
    leaving the existing operation validator responsible for SSA ordering and
    region semantics.

    Args:
        block (Block): Block whose local value graph should be checked.
        state (_ValidationState): Root static-field identities.
        location (str): Human-readable block location.

    Raises:
        ValueError: If a non-constant value is referenced without a producer
            in this block's scope.
    """
    allowed_aliases = _static_scalar_pass_through_aliases(block, state)
    available: set[str] = set()
    available_values: dict[str, ValueBase] = {}

    def register(value: ValueBase) -> None:
        """Register one locally produced structural value graph.

        Args:
            value (ValueBase): Produced value or formal input.
        """
        for item in _iter_value_graph(value):
            available.add(item.uuid)
            available_values[item.uuid] = item

    def register_local(
        values: Iterable[ValueBase],
        role: str,
        *,
        allow_static_pass_through: bool = False,
    ) -> None:
        """Register producer identities disjoint from static binding fields.

        Args:
            values (Iterable[ValueBase]): Values defined by a local producer.
            role (str): Producer role used in diagnostics.
            allow_static_pass_through (bool): Whether a structurally verified
                direct-call result may retain a static field's logical ID.
                Defaults to ``False``.

        Raises:
            ValueError: If a local producer aliases a static field by UUID or
                logical identity.
        """
        for value in values:
            expected_uuid = state.static_field_logical_ids.get(value.logical_id)
            if value.uuid in state.static_field_uuids or (
                expected_uuid is not None
                and (not allow_static_pass_through or value.uuid not in allowed_aliases)
            ):
                raise ValueError(f"{location} {role} aliases a static binding field")
            available.add(value.uuid)
            available_values[value.uuid] = value

    for value in block.input_values:
        register_local(_iter_value_graph(value), "input")
    for value in block.parameters.values():
        register_local(_iter_value_graph(value), "parameter")
    for slot, _ in state.static_bindings.values():
        for field in slot.fields:
            register(field.value)

    operations = list(_iter_operations(block.operations))
    for operation in operations:
        for result in operation.results:
            register_local(
                _iter_result_producers(result),
                "operation result",
                allow_static_pass_through=True,
            )
        if isinstance(operation, ForOperation) and operation.loop_var_value is not None:
            register_local(
                _iter_value_graph(operation.loop_var_value),
                "loop variable",
            )
        if isinstance(operation, ForItemsOperation):
            for key_value in operation.key_var_values or ():
                register_local(_iter_value_graph(key_value), "item key")
            if operation.value_var_value is not None:
                register_local(
                    _iter_value_graph(operation.value_var_value),
                    "item value",
                )
        for region_arg in getattr(operation, "region_args", ()):
            register_local(
                _iter_value_graph(region_arg.block_arg),
                "region block argument",
            )

    inline_expval_operations = _inline_expval_operations(
        block,
        available_values,
    )
    referenced: list[ValueBase] = []
    invalid_inline_carrier_uuids: set[str] = set()
    for operation in operations:
        for input_index, value in enumerate(operation.all_input_values()):
            if isinstance(operation, ExpvalOp) and input_index == 0:
                if id(operation) in inline_expval_operations:
                    continue
                if _is_inline_expval_carrier_candidate(value):
                    invalid_inline_carrier_uuids.add(value.uuid)
            referenced.append(value)
        for result in operation.results:
            referenced.extend(_result_dependency_values(result))
    referenced.extend(block.output_values)
    producer_conflicts = sorted(
        {
            value.uuid
            for root in referenced
            for value in _iter_value_graph(root)
            if value.uuid in available_values
            and not _same_value_identity(value, available_values[value.uuid])
        }
    )
    if producer_conflicts:
        raise ValueError(
            f"{location} references UUIDs with conflicting value identities: "
            f"{producer_conflicts!r}"
        )
    logical_aliases = sorted(
        {
            value.uuid
            for root in referenced
            for value in _iter_value_graph(root)
            if value.logical_id in state.static_field_logical_ids
            and value.uuid != state.static_field_logical_ids[value.logical_id]
            and value.uuid not in allowed_aliases
        }
    )
    if logical_aliases:
        raise ValueError(
            f"{location} aliases static binding logical identities with new "
            f"UUIDs: {logical_aliases!r}"
        )
    dangling = sorted(
        invalid_inline_carrier_uuids
        | {
            value.uuid
            for root in referenced
            for value in _iter_value_graph(root)
            if value.uuid not in available
            and not value.is_constant()
            and not (
                isinstance(value, Value)
                and not isinstance(value, ArrayValue)
                and value.parent_array is not None
            )
        }
    )
    if dangling:
        raise ValueError(
            f"{location} references values without a local producer: {dangling!r}"
        )


def _is_inline_expval_carrier_candidate(value: ValueBase) -> bool:
    """Return whether a value uses the frontend's tuple-carrier envelope.

    Args:
        value (ValueBase): Expval quantum operand to classify.

    Returns:
        bool: Whether the operand is a zero-shape quantum array with runtime
            element metadata and therefore must pass the dedicated lexical
            carrier validation instead of falling back to an ordinary value.
    """
    return (
        isinstance(value, ArrayValue)
        and value.type == QubitType()
        and not value.shape
        and value.metadata.array_runtime is not None
    )


def _same_value_identity(left: ValueBase, right: ValueBase) -> bool:
    """Compare identity-defining fields on two same-UUID values.

    Args:
        left (ValueBase): Referenced value.
        right (ValueBase): Locally produced value with the same UUID.

    Returns:
        bool: Whether kind, type, logical identity, and SSA version agree.
    """
    return (
        type(left) is type(right)
        and left.type == right.type
        and left.logical_id == right.logical_id
        and getattr(left, "version", None) == getattr(right, "version", None)
    )


def _inline_expval_operations(
    block: Block,
    all_produced_values: dict[str, ValueBase],
) -> set[int]:
    """Identify expval uses with a valid inline tuple carrier.

    The frontend represents ``expval((q0, q1), observable)`` by a synthetic
    zero-shape ``ArrayValue`` whose metadata records the real qubit producers.
    The carrier is intentionally not an SSA operation result. This walk keeps
    the exception local to that exact expval operand and tracks lexical
    producer order, so the carrier cannot escape to another operation or refer
    to a later or branch-local qubit producer.

    Args:
        block (Block): Block whose lexical operation tree to inspect.
        all_produced_values (dict[str, ValueBase]): Every value produced in the
            block, used to detect future or conflicting element UUIDs.

    Returns:
        set[int]: Object identities of expval operations whose first operand
            is a structurally valid inline carrier.
    """
    visible: dict[str, ValueBase] = {}

    def register_visible(values: Iterable[ValueBase]) -> None:
        """Add one structural producer graph to the lexical environment.

        Args:
            values (Iterable[ValueBase]): Values visible from the next
                operation onward.
        """
        for value in values:
            visible[value.uuid] = value

    for value in (*block.input_values, *block.parameters.values()):
        register_visible(_iter_value_graph(value))
    for slot in block.static_bindings:
        for field in slot.fields:
            register_visible(_iter_value_graph(field.value))

    valid: set[int] = set()
    used_carrier_uuids: set[str] = set()

    def walk(
        operations: Iterable[Operation],
        scope: dict[str, ValueBase],
    ) -> None:
        """Validate inline carriers along one lexical control-flow path.

        Args:
            operations (Iterable[Operation]): Operations in producer order.
            scope (dict[str, ValueBase]): Values dominating the current path.
        """
        for operation in operations:
            if isinstance(operation, ExpvalOp):
                carrier = operation.operands[0]
                if (
                    carrier.uuid not in scope
                    and carrier.uuid not in all_produced_values
                    and carrier.uuid not in used_carrier_uuids
                    and _is_valid_inline_expval_carrier(
                        carrier,
                        scope,
                        all_produced_values,
                    )
                ):
                    valid.add(id(operation))
                    used_carrier_uuids.add(carrier.uuid)

            if isinstance(operation, HasNestedOps):
                nested_scope = dict(scope)
                if (
                    isinstance(operation, ForOperation)
                    and operation.loop_var_value is not None
                ):
                    for value in _iter_value_graph(operation.loop_var_value):
                        nested_scope[value.uuid] = value
                if isinstance(operation, ForItemsOperation):
                    for key_value in operation.key_var_values or ():
                        for value in _iter_value_graph(key_value):
                            nested_scope[value.uuid] = value
                    if operation.value_var_value is not None:
                        for value in _iter_value_graph(operation.value_var_value):
                            nested_scope[value.uuid] = value
                for region_arg in getattr(operation, "region_args", ()):
                    for value in _iter_value_graph(region_arg.block_arg):
                        nested_scope[value.uuid] = value
                for nested in operation.nested_op_lists():
                    walk(nested, dict(nested_scope))

            for result in operation.results:
                for value in _iter_result_producers(result):
                    scope[value.uuid] = value

    walk(block.operations, visible)
    return valid


def _is_valid_inline_expval_carrier(
    carrier: ValueBase,
    visible: dict[str, ValueBase],
    all_produced_values: dict[str, ValueBase],
) -> bool:
    """Return whether one unproduced value is an exact tuple-expval carrier.

    Args:
        carrier (ValueBase): Candidate first operand of an ``ExpvalOp``.
        visible (dict[str, ValueBase]): Producers dominating the expval use.
        all_produced_values (dict[str, ValueBase]): Producers from the complete
            block, including later and branch-local values.

    Returns:
        bool: Whether the carrier has the frontend's exact structural form and
            every element resolves to one distinct visible logical qubit.
    """
    if (
        not isinstance(carrier, ArrayValue)
        or carrier.type != QubitType()
        or carrier.shape
        or carrier.parent_array is not None
        or carrier.element_indices
        or carrier.slice_of is not None
        or carrier.slice_start is not None
        or carrier.slice_step is not None
    ):
        return False
    metadata = carrier.metadata
    runtime = metadata.array_runtime
    if (
        runtime is None
        or runtime.const_array is not None
        or metadata.scalar is not None
        or metadata.cast is not None
        or metadata.qfixed is not None
        or metadata.dict_runtime is not None
    ):
        return False
    element_count = len(runtime.element_uuids)
    if (
        element_count == 0
        or len(runtime.element_logical_ids) != element_count
        or len(runtime.element_parent_uuids) != element_count
        or len(runtime.element_parent_indices) != element_count
        or len(set(runtime.element_uuids)) != element_count
        or len(set(runtime.element_logical_ids)) != element_count
        or carrier.uuid in runtime.element_uuids
        or carrier.logical_id in runtime.element_logical_ids
    ):
        return False

    produced_logical_ids = {value.logical_id for value in all_produced_values.values()}
    if carrier.logical_id in produced_logical_ids:
        return False
    logical_addresses: list[tuple[str, int]] = []
    for element_uuid, element_logical_id, parent_uuid, parent_index in zip(
        runtime.element_uuids,
        runtime.element_logical_ids,
        runtime.element_parent_uuids,
        runtime.element_parent_indices,
        strict=True,
    ):
        if parent_uuid:
            parent = visible.get(parent_uuid)
            if (
                not isinstance(parent, ArrayValue)
                or parent.type != QubitType()
                or parent.slice_of is not None
                or parent_index < 0
                or not _array_index_may_be_in_bounds(parent, parent_index)
            ):
                return False
            known_element = all_produced_values.get(element_uuid)
            if known_element is not None:
                if (
                    element_uuid not in visible
                    or type(known_element) is not Value
                    or known_element.type != QubitType()
                    or known_element.logical_id != element_logical_id
                    or resolve_root_qubit_address(known_element)
                    != (parent_uuid, parent_index)
                ):
                    return False
            elif element_logical_id in produced_logical_ids:
                return False
            logical_address = (parent.logical_id, parent_index)
        else:
            element = visible.get(element_uuid)
            if (
                type(element) is not Value
                or element.type != QubitType()
                or element.logical_id != element_logical_id
                or element.parent_array is not None
                or element.element_indices
                or parent_index != -1
            ):
                return False
            logical_address = (element.logical_id, -1)
        logical_addresses.append(logical_address)
    return len(set(logical_addresses)) == element_count


def _array_index_may_be_in_bounds(array: ArrayValue, index: int) -> bool:
    """Check a concrete array bound while deferring symbolic dimensions.

    Args:
        array (ArrayValue): Root quantum array carrying shape metadata.
        index (int): Flat element index recorded by tuple-expval lowering.

    Returns:
        bool: ``False`` for a provable out-of-bounds index, otherwise ``True``.
    """
    size = 1
    for dimension in array.shape:
        concrete_dimension = dimension.get_const()
        if type(concrete_dimension) is not int:
            return True
        size *= concrete_dimension
    return index < size


def _static_scalar_pass_through_aliases(
    block: Block,
    state: _ValidationState,
) -> set[str]:
    """Identify validated call results derived from static scalar fields.

    Ordinary qkernel calls advance the SSA version of a pass-through input.
    A static scalar therefore acquires a new UUID when a helper returns it,
    while intentionally retaining the field's logical identity. Only direct,
    top-level scalar formal pass-throughs are accepted in producer order;
    arbitrary same-logical-ID values and region-local aliases remain invalid
    at the serialization trust boundary.

    Args:
        block (Block): Block whose lexical invocation results to inspect.
        state (_ValidationState): Registered root static-field identities.

    Returns:
        set[str]: Original static-field UUIDs plus structurally verified
            pass-through result UUIDs.
    """
    aliases = set(state.static_field_uuids)
    for operation in block.operations:
        if (
            not isinstance(operation, InvokeOperation)
            or operation.transform is not CallTransform.DIRECT
            or operation.num_control_qubits
            or operation.definition is None
            or operation.definition.body is None
            or operation.definition.body_ref is not None
            or operation.definition.implementations
            or operation.definition.opaque_cost is not None
        ):
            continue
        body = operation.definition.body
        if len(body.input_values) != len(operation.operands) or len(
            body.output_values
        ) != len(operation.results):
            continue
        formal_indices = {
            formal.uuid: index for index, formal in enumerate(body.input_values)
        }
        for output, result in zip(
            body.output_values,
            operation.results,
            strict=True,
        ):
            if type(output) is not Value or type(result) is not Value:
                continue
            formal_index = formal_indices.get(output.uuid)
            if formal_index is None:
                continue
            formal = body.input_values[formal_index]
            actual = operation.operands[formal_index]
            if (
                type(formal) is not Value
                or type(actual) is not Value
                or actual.uuid not in aliases
                or formal.type != output.type
                or output.type != actual.type
                or result.logical_id != actual.logical_id
                or result.type != actual.type
            ):
                continue
            aliases.add(result.uuid)
    return aliases


def _iter_operations(operations: Iterable[Operation]) -> Iterable[Operation]:
    """Yield operations and their lexical control-flow regions.

    Args:
        operations (Iterable[Operation]): Operations in one block scope.

    Yields:
        Operation: Each operation reachable without crossing into an owned
            callable block.
    """
    for operation in operations:
        yield operation
        if isinstance(operation, HasNestedOps):
            for nested in operation.nested_op_lists():
                yield from _iter_operations(nested)


def _iter_value_graph(value: ValueBase) -> Iterable[ValueBase]:
    """Yield one structural value graph without revisiting UUIDs.

    Args:
        value (ValueBase): Root value or dependency to traverse.

    Yields:
        ValueBase: Root and recursively referenced structural values.
    """
    pending = [value]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current.uuid in seen:
            continue
        seen.add(current.uuid)
        yield current
        if isinstance(current, TupleValue):
            pending.extend(current.elements)
        elif isinstance(current, DictValue):
            for key, entry_value in current.entries:
                pending.extend((key, entry_value))
        elif isinstance(current, ArrayValue):
            pending.extend(current.shape)
            for dependency in (
                current.slice_of,
                current.slice_start,
                current.slice_step,
            ):
                if dependency is not None:
                    pending.append(dependency)
        elif isinstance(current, Value):
            if current.parent_array is not None:
                pending.append(current.parent_array)
            pending.extend(current.element_indices)


def _iter_result_producers(value: ValueLike) -> Iterable[ValueBase]:
    """Yield identities defined by one operation result.

    Array shapes and element indices are dependencies. A scalar element result
    also produces its next-version parent array, while tuple and dictionary
    elements are produced as part of their container.

    Args:
        value (ValueLike): Operation result value.

    Yields:
        ValueBase: Values introduced by the result.
    """
    yield value
    if isinstance(value, Value) and value.parent_array is not None:
        yield value.parent_array
    elif isinstance(value, TupleValue):
        for element in value.elements:
            yield from _iter_result_producers(element)
    elif isinstance(value, DictValue):
        for key, entry_value in value.entries:
            yield from _iter_result_producers(key)
            yield from _iter_result_producers(entry_value)


def _result_dependency_values(value: ValueLike) -> list[ValueBase]:
    """Return structural references consumed by an operation result.

    Args:
        value (ValueLike): Operation result value.

    Returns:
        list[ValueBase]: Shape, slice, parent, and index dependencies.
    """
    dependencies: list[ValueBase] = []
    if isinstance(value, ArrayValue):
        dependencies.extend(value.shape)
        for dependency in (value.slice_of, value.slice_start, value.slice_step):
            if dependency is not None:
                dependencies.append(dependency)
    elif isinstance(value, Value):
        if value.parent_array is not None:
            dependencies.append(value.parent_array)
        dependencies.extend(value.element_indices)
    elif isinstance(value, TupleValue):
        for element in value.elements:
            dependencies.extend(_result_dependency_values(element))
    elif isinstance(value, DictValue):
        for key, entry_value in value.entries:
            dependencies.extend(_result_dependency_values(key))
            dependencies.extend(_result_dependency_values(entry_value))
    return dependencies


def _validate_operation(
    operation: Operation,
    state: _ValidationState,
    producers: dict[str, str],
    location: str,
) -> None:
    """Validate one operation before following its nested graph edges.

    Args:
        operation (Operation): Operation to validate.
        state (_ValidationState): Shared graph-validation state.
        producers (dict[str, str]): SSA producers in the owning block scope.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If the operation is structurally or semantically invalid.
    """
    if not all(isinstance(value, ValueBase) for value in operation.operands):
        raise ValueError(f"{location} has a non-Value operand")
    if not all(isinstance(value, ValueBase) for value in operation.results):
        raise ValueError(f"{location} has a non-Value result")
    _require_unique_values(operation.results, f"{location} results")
    operand_uuids = {value.uuid for value in operation.operands}
    for result in operation.results:
        if result.uuid in state.static_field_uuids:
            raise ValueError(
                f"{location} reuses a root static binding field as an SSA result"
            )
        if result.uuid in operand_uuids:
            raise ValueError(f"{location} reuses an operand UUID as an SSA result")
        previous = producers.get(result.uuid)
        if previous is not None:
            raise ValueError(
                f"{location} produces UUID {result.uuid!r} already produced by "
                f"{previous}"
            )
        producers[result.uuid] = location

    _validate_operation_contract(operation, location)
    _validate_quantum_operand_uniqueness(operation, location)
    if isinstance(operation, InvokeOperation):
        _validate_static_binding_invoke(operation, state, location)

    if isinstance(operation, HasNestedOps):
        for region_index, operations in enumerate(operation.nested_op_lists()):
            for child_index, child in enumerate(operations):
                _validate_operation(
                    child,
                    state,
                    producers,
                    f"{location} region {region_index} operation {child_index} "
                    f"({type(child).__name__})",
                )
    if isinstance(operation, InvokeOperation) and operation.definition is not None:
        _validate_definition(operation.definition, state, location)
    if isinstance(operation, (ConcreteControlledU, SymbolicControlledU)):
        if operation.block is not None:
            _validate_block(operation.block, state, f"{location} unitary block")
    if isinstance(operation, InverseBlockOperation):
        if operation.source_block is not None:
            _validate_block(operation.source_block, state, f"{location} source block")
        if operation.implementation_block is not None:
            _validate_block(
                operation.implementation_block,
                state,
                f"{location} implementation block",
            )
    if isinstance(operation, SelectOperation):
        for case_index, case_block in enumerate(operation.case_blocks):
            _validate_block(
                case_block,
                state,
                f"{location} case block {case_index}",
            )


def _validate_definition(
    definition: CallableDef,
    state: _ValidationState,
    location: str,
) -> None:
    """Validate semantic blocks owned by one callable definition.

    Args:
        definition (CallableDef): Callable definition to inspect.
        state (_ValidationState): Shared graph-validation state.
        location (str): Location of the invocation that references it.

    Raises:
        ValueError: If a callable body or implementation is malformed.
    """
    if id(definition) in state.seen_definitions:
        return
    state.seen_definitions.add(id(definition))
    if definition.body is not None:
        _validate_block(definition.body, state, f"{location} callable body")
    for index, implementation in enumerate(definition.implementations):
        if implementation.body is not None:
            _validate_block(
                implementation.body,
                state,
                f"{location} callable implementation {index}",
            )


def _validate_operation_contract(operation: Operation, location: str) -> None:
    """Dispatch operation-specific arity, type, and region checks.

    Args:
        operation (Operation): Operation to inspect.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If a concrete operation contract is violated.
    """
    if isinstance(operation, GateOperation):
        _validate_gate(operation, location)
    elif isinstance(operation, MeasureOperation):
        _require_arity(operation, 1, 1, location)
        _require_types(operation.operands, [QubitType()], location, "operand")
        _require_types(operation.results, [BitType()], location, "result")
    elif isinstance(operation, ProjectOperation):
        _require_arity(operation, 1, 2, location)
        _require_types(operation.operands, [QubitType()], location, "operand")
        _require_types(
            operation.results,
            [QubitType(), BitType()],
            location,
            "result",
        )
    elif isinstance(operation, ResetOperation):
        _require_arity(operation, 1, 1, location)
        _require_types(operation.operands, [QubitType()], location, "operand")
        _require_types(operation.results, [QubitType()], location, "result")
    elif isinstance(operation, MeasureVectorOperation):
        _require_arity(operation, 1, 1, location)
        _require_array_type(operation.operands[0], QubitType(), location)
        _require_array_type(operation.results[0], BitType(), location)
    elif isinstance(operation, MeasureQFixedOperation):
        _require_arity(operation, 1, 1, location)
        _validate_fixed_point_layout(
            operation.num_bits,
            operation.int_bits,
            location,
        )
        _require_types(operation.operands, [QFixedType()], location, "operand")
        _require_types(operation.results, [FloatType()], location, "result")
    elif isinstance(operation, DecodeQFixedOperation):
        _require_arity(operation, 1, 1, location)
        _validate_fixed_point_layout(
            operation.num_bits,
            operation.int_bits,
            location,
        )
        _require_array_type(operation.operands[0], BitType(), location)
        _require_types(operation.results, [FloatType()], location, "result")
    elif isinstance(operation, StoreArrayElementOperation):
        _validate_array_store(operation, location)
    elif isinstance(operation, ReturnQuantumArrayElementOperation):
        _validate_quantum_array_return(operation, location)
    elif isinstance(operation, DictGetItemOperation):
        _validate_dict_get(operation, location)
    elif isinstance(operation, CastOperation):
        _require_arity(operation, 1, 1, location)
        if operation.source_type is None or operation.target_type is None:
            raise ValueError(f"{location} requires source_type and target_type")
        _require_types(
            operation.operands,
            [operation.source_type],
            location,
            "operand",
        )
        _require_types(
            operation.results,
            [operation.target_type],
            location,
            "result",
        )
    elif isinstance(operation, (QInitOperation, CInitOperation)):
        _require_arity(operation, 0, 1, location)
        if (
            isinstance(operation, QInitOperation)
            and not operation.results[0].type.is_quantum()
        ):
            raise ValueError(f"{location} must initialize a quantum value")
        if (
            isinstance(operation, CInitOperation)
            and operation.results[0].type.is_quantum()
        ):
            raise ValueError(f"{location} cannot initialize a quantum value")
    elif isinstance(operation, SliceArrayOperation):
        _require_arity(operation, 3, 1, location)
        if not isinstance(operation.operands[0], ArrayValue) or not isinstance(
            operation.results[0], ArrayValue
        ):
            raise ValueError(f"{location} must slice an ArrayValue into an ArrayValue")
        _require_types(
            operation.operands[1:], [UIntType(), UIntType()], location, "operand"
        )
    elif isinstance(operation, ReleaseSliceViewOperation):
        _require_arity(operation, 1, 0, location)
        if not isinstance(operation.operands[0], ArrayValue):
            raise ValueError(f"{location} must release an ArrayValue")
    elif isinstance(operation, ReturnOperation):
        _require_result_count(operation, 0, location)
    elif isinstance(operation, GlobalPhaseOperation):
        _require_arity(operation, 1, 0, location)
        _require_scalar_type(
            operation.operands[0],
            FloatType(),
            f"{location} phase operand",
        )
    elif isinstance(operation, ExpvalOp):
        _require_arity(operation, 2, 1, location)
        if not operation.operands[0].type.is_quantum():
            raise ValueError(f"{location} first operand must be quantum")
        _require_types(operation.operands[1:], [ObservableType()], location, "operand")
        _require_types(operation.results, [FloatType()], location, "result")
    elif isinstance(operation, PauliEvolveOp):
        _require_arity(operation, 3, 1, location)
        if not operation.operands[0].type.is_quantum():
            raise ValueError(f"{location} first operand must be quantum")
        _require_types(
            operation.operands[1:2],
            [ObservableType()],
            location,
            "operand",
        )
        _require_scalar_type(
            operation.operands[2],
            FloatType(),
            f"{location} angle operand",
        )
        if operation.results[0].type != operation.operands[0].type:
            raise ValueError(f"{location} result type must match its quantum input")
    elif isinstance(operation, BinOp):
        _validate_binop(operation, location)
    elif isinstance(operation, CompOp):
        _validate_comparison(operation, location)
    elif isinstance(operation, CondOp):
        _validate_logical(operation, location)
    elif isinstance(operation, NotOp):
        _require_arity(operation, 1, 1, location)
        _require_types(operation.operands, [BitType()], location, "operand")
        _require_types(operation.results, [BitType()], location, "result")
    elif isinstance(operation, RuntimeClassicalExpr):
        _validate_runtime_expression(operation, location)
    elif isinstance(operation, ForOperation):
        _require_operand_count(operation, 3, location)
        _require_types(
            operation.operands,
            [UIntType(), UIntType(), UIntType()],
            location,
            "operand",
        )
        if operation.loop_var_value is None:
            raise ValueError(f"{location} requires a loop_var_value")
        _require_value_type(operation.loop_var_value, UIntType(), location)
        _validate_region_args(operation.region_args, operation, location)
    elif isinstance(operation, ForItemsOperation):
        _require_operand_count(operation, 1, location)
        if not isinstance(operation.operands[0], DictValue):
            raise ValueError(f"{location} iterable operand must be a DictValue")
        if operation.key_var_values is None or operation.value_var_value is None:
            raise ValueError(f"{location} requires key and value region identities")
        _validate_region_args(operation.region_args, operation, location)
    elif isinstance(operation, WhileOperation):
        if len(operation.operands) not in {1, 2}:
            raise ValueError(f"{location} requires one or two condition operands")
        _require_types(
            operation.operands,
            [BitType()] * len(operation.operands),
            location,
            "operand",
        )
        _validate_region_args(operation.region_args, operation, location)
    elif isinstance(operation, IfOperation):
        _validate_if(operation, location)
    elif isinstance(operation, ConcreteControlledU):
        _validate_concrete_controlled(operation, location)
    elif isinstance(operation, SymbolicControlledU):
        _validate_symbolic_controlled(operation, location)
    elif isinstance(operation, InvokeOperation):
        _validate_invoke(operation, location)
    elif isinstance(operation, InverseBlockOperation):
        _validate_inverse_block(operation, location)
    elif isinstance(operation, SelectOperation):
        _validate_select(operation, location)
    elif isinstance(operation, GlobalPhaseOperation):
        _require_arity(operation, 1, 0, location)
        _require_types(operation.operands, [FloatType()], location, "operand")
    else:
        raise ValueError(f"{location} has unsupported operation type")


def _validate_gate(operation: GateOperation, location: str) -> None:
    """Validate a primitive gate's fixed qubit and angle layout.

    Args:
        operation (GateOperation): Gate to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If the gate type or operand/result layout is invalid.
    """
    gate_type = operation.gate_type
    if gate_type in _SINGLE_QUBIT_GATES:
        qubits, has_angle = 1, False
    elif gate_type in _SINGLE_QUBIT_ROTATIONS:
        qubits, has_angle = 1, True
    elif gate_type in _TWO_QUBIT_GATES:
        qubits, has_angle = 2, False
    elif gate_type in _TWO_QUBIT_ROTATIONS:
        qubits, has_angle = 2, True
    elif gate_type is GateOperationType.TOFFOLI:
        qubits, has_angle = 3, False
    else:
        raise ValueError(f"{location} has an unknown gate type {gate_type!r}")
    _require_arity(operation, qubits + int(has_angle), qubits, location)
    _require_types(
        operation.operands[:qubits],
        [QubitType()] * qubits,
        location,
        "operand",
    )
    if has_angle:
        _require_scalar_type(
            operation.operands[-1],
            FloatType(),
            f"{location} angle operand",
        )
    _require_types(
        operation.results,
        [QubitType()] * qubits,
        location,
        "result",
    )


def _validate_fixed_point_layout(
    num_bits: int,
    int_bits: int,
    location: str,
) -> None:
    """Validate a measured fixed-point bit layout.

    Args:
        num_bits (int): Total number of encoded bits.
        int_bits (int): Number of integer bits.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If either width is invalid or internally inconsistent.
    """
    if num_bits < 1:
        raise ValueError(f"{location} num_bits must be positive")
    if int_bits < 0 or int_bits > num_bits:
        raise ValueError(f"{location} int_bits must be between 0 and num_bits")


def _validate_binop(operation: BinOp, location: str) -> None:
    """Validate a scalar numeric arithmetic operation.

    Args:
        operation (BinOp): Arithmetic operation to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If arity or numeric type promotion is invalid.
    """
    _require_arity(operation, 2, 1, location)
    operand_types = [value.type for value in operation.operands]
    if not all(
        isinstance(value_type, (UIntType, FloatType)) for value_type in operand_types
    ):
        raise ValueError(f"{location} operands must be UIntType or FloatType")

    if operation.kind in {BinOpKind.FLOORDIV, BinOpKind.MOD}:
        if not all(isinstance(value_type, UIntType) for value_type in operand_types):
            raise ValueError(f"{location} requires UIntType operands")
        expected_result = UIntType()
    elif operation.kind is BinOpKind.DIV:
        expected_result = FloatType()
    else:
        result_type = operation.results[0].type
        if not isinstance(result_type, (UIntType, FloatType)):
            raise ValueError(f"{location} result must be UIntType or FloatType")
        expected_result = result_type
    _require_types(operation.results, [expected_result], location, "result")


def _supports_comparison_operands(
    operand_types: Iterable[ValueType],
    *,
    equality: bool,
) -> bool:
    """Return whether scalar types support one comparison family.

    Args:
        operand_types (Iterable[ValueType]): Ordered comparison operand types.
        equality (bool): Whether equality and inequality semantics apply.

    Returns:
        bool: ``True`` for numeric pairs, or for Bit/UInt pairs when the
            operation is equality or inequality.
    """
    types = tuple(operand_types)
    numeric_comparison = all(
        isinstance(value_type, (UIntType, FloatType)) for value_type in types
    )
    bit_equality = equality and all(
        isinstance(value_type, (BitType, UIntType)) for value_type in types
    )
    return numeric_comparison or bit_equality


_COMPARISON_OPERANDS_ERROR = (
    "operands must be numeric scalars (each UIntType or FloatType); equality and "
    "inequality additionally allow each operand to be BitType or UIntType"
)


def _validate_comparison(operation: CompOp, location: str) -> None:
    """Validate a scalar comparison.

    Args:
        operation (CompOp): Comparison operation to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If the operand types do not support the comparison kind or
            the result is not a bit.
    """
    _require_arity(operation, 2, 1, location)
    operand_types = [value.type for value in operation.operands]
    if not _supports_comparison_operands(
        operand_types,
        equality=operation.kind in {CompOpKind.EQ, CompOpKind.NEQ},
    ):
        raise ValueError(f"{location} {_COMPARISON_OPERANDS_ERROR}")
    _require_types(operation.results, [BitType()], location, "result")


def _validate_logical(operation: CondOp, location: str) -> None:
    """Validate a binary bitwise logical operation.

    Args:
        operation (CondOp): Logical operation to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If operands or result are not bits.
    """
    _require_arity(operation, 2, 1, location)
    _require_types(operation.operands, [BitType(), BitType()], location, "operand")
    _require_types(operation.results, [BitType()], location, "result")


def _validate_runtime_expression(
    operation: RuntimeClassicalExpr,
    location: str,
) -> None:
    """Validate one lowered runtime classical expression.

    Args:
        operation (RuntimeClassicalExpr): Runtime expression to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If the expression family, arity, or scalar types disagree.
    """
    if operation.kind is RuntimeOpKind.NOT:
        _require_arity(operation, 1, 1, location)
        _require_types(operation.operands, [BitType()], location, "operand")
        _require_types(operation.results, [BitType()], location, "result")
        return
    if operation.kind is RuntimeOpKind.SELECT:
        _require_arity(operation, 3, 1, location)
        _require_value_type(operation.operands[0], BitType(), location)
        branch_type = operation.operands[1].type
        if not isinstance(branch_type, (BitType, UIntType, FloatType)):
            raise ValueError(f"{location} selected values must be classical scalars")
        _require_types(
            operation.operands[1:],
            [branch_type, branch_type],
            location,
            "selected operand",
        )
        _require_types(operation.results, [branch_type], location, "result")
        return
    if operation.kind in {
        RuntimeOpKind.EQ,
        RuntimeOpKind.NEQ,
        RuntimeOpKind.LT,
        RuntimeOpKind.LE,
        RuntimeOpKind.GT,
        RuntimeOpKind.GE,
    }:
        _require_arity(operation, 2, 1, location)
        operand_types = [value.type for value in operation.operands]
        if not _supports_comparison_operands(
            operand_types,
            equality=operation.kind in {RuntimeOpKind.EQ, RuntimeOpKind.NEQ},
        ):
            raise ValueError(f"{location} {_COMPARISON_OPERANDS_ERROR}")
        _require_types(operation.results, [BitType()], location, "result")
        return
    if operation.kind in {RuntimeOpKind.AND, RuntimeOpKind.OR}:
        _require_arity(operation, 2, 1, location)
        _require_types(
            operation.operands,
            [BitType(), BitType()],
            location,
            "operand",
        )
        _require_types(operation.results, [BitType()], location, "result")
        return

    _require_arity(operation, 2, 1, location)
    operand_types = [value.type for value in operation.operands]
    if not all(
        isinstance(value_type, (UIntType, FloatType)) for value_type in operand_types
    ):
        raise ValueError(f"{location} operands must be UIntType or FloatType")
    if operation.kind in {RuntimeOpKind.FLOORDIV, RuntimeOpKind.MOD}:
        if not all(isinstance(value_type, UIntType) for value_type in operand_types):
            raise ValueError(f"{location} requires UIntType operands")
        expected_result = UIntType()
    elif operation.kind is RuntimeOpKind.DIV:
        expected_result = FloatType()
    else:
        result_type = operation.results[0].type
        if not isinstance(result_type, (UIntType, FloatType)):
            raise ValueError(f"{location} result must be UIntType or FloatType")
        expected_result = result_type
    _require_types(operation.results, [expected_result], location, "result")


def _validate_array_store(
    operation: StoreArrayElementOperation,
    location: str,
) -> None:
    """Validate one classical array-element SSA rewrite.

    Args:
        operation (StoreArrayElementOperation): Store to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If array, index, stored-value, or result contracts differ.
    """
    if len(operation.operands) < 3:
        raise ValueError(f"{location} requires an array, value, and index")
    _require_result_count(operation, 1, location)
    source = operation.operands[0]
    result = operation.results[0]
    if not isinstance(source, ArrayValue) or not isinstance(result, ArrayValue):
        raise ValueError(f"{location} source and result must be ArrayValue instances")
    if source.type != operation.operands[1].type or result.type != source.type:
        raise ValueError(f"{location} array element and result types disagree")
    _require_types(
        operation.operands[2:],
        [UIntType()] * (len(operation.operands) - 2),
        location,
        "index",
    )


def _validate_quantum_array_return(
    operation: ReturnQuantumArrayElementOperation,
    location: str,
) -> None:
    """Validate one deferred quantum array borrow return.

    Args:
        operation (ReturnQuantumArrayElementOperation): Return to validate.
        location (str): Human-readable graph location.

    Raises:
        ValueError: If the array, qubit, index arity, or result contract is
            malformed.
    """
    if len(operation.operands) != 4:
        raise ValueError(
            f"{location} requires an array, qubit, target index, and source index"
        )
    _require_result_count(operation, 0, location)
    source = operation.operands[0]
    if not isinstance(source, ArrayValue) or not source.type.is_quantum():
        raise ValueError(f"{location} first operand must be a quantum ArrayValue")
    if operation.returned_value.type != source.type:
        raise ValueError(f"{location} returned qubit type must match its array")
    _require_types(
        [*operation.target_indices, *operation.source_indices],
        [UIntType()] * (2 * operation.index_arity),
        location,
        "index",
    )


def _validate_dict_get(operation: DictGetItemOperation, location: str) -> None:
    """Validate a dictionary lookup's declared key arity.

    Args:
        operation (DictGetItemOperation): Lookup to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If its dictionary, key, or result layout is invalid.
    """
    if operation.key_arity < 1:
        raise ValueError(f"{location} key_arity must be positive")
    _require_arity(operation, 1 + operation.key_arity, 1, location)
    if not isinstance(operation.operands[0], DictValue):
        raise ValueError(f"{location} first operand must be a DictValue")


def _validate_if(operation: IfOperation, location: str) -> None:
    """Validate a conditional's condition and merge edges.

    Args:
        operation (IfOperation): Conditional to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If the condition or any branch merge is inconsistent.
    """
    _require_operand_count(operation, 1, location)
    _require_types(operation.operands, [BitType()], location, "operand")
    try:
        merges = list(operation.iter_merges())
    except RuntimeError as exc:
        raise ValueError(f"{location} has inconsistent branch merges") from exc
    for merge in merges:
        if not (merge.true_value.type == merge.false_value.type == merge.result.type):
            raise ValueError(f"{location} branch merge types disagree")


def _validate_region_args(
    region_args: tuple[RegionArg, ...],
    operation: Operation,
    location: str,
) -> None:
    """Validate explicit loop-carried SSA region arguments.

    Args:
        region_args (tuple[RegionArg, ...]): Region records to validate.
        operation (Operation): Owning loop operation.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If result ownership, types, or identities are inconsistent.
    """
    if len(region_args) != len(operation.results):
        raise ValueError(f"{location} region-arg count must match result count")
    result_uuids = {value.uuid for value in operation.results}
    block_arg_uuids: set[str] = set()
    for region_arg in region_args:
        values = (
            region_arg.init,
            region_arg.block_arg,
            region_arg.yielded,
            region_arg.result,
        )
        if not all(value.type == region_arg.init.type for value in values):
            raise ValueError(f"{location} region-arg types disagree")
        if region_arg.result.uuid not in result_uuids:
            raise ValueError(f"{location} region result is not owned by the loop")
        if region_arg.block_arg.uuid in block_arg_uuids:
            raise ValueError(f"{location} repeats a region block-argument UUID")
        block_arg_uuids.add(region_arg.block_arg.uuid)


def _validate_concrete_controlled(
    operation: ConcreteControlledU,
    location: str,
) -> None:
    """Validate a concrete controlled-unitary layout.

    Args:
        operation (ConcreteControlledU): Operation to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If controls or quantum results do not match operands.
    """
    if operation.num_controls < 1 or operation.num_controls > len(operation.operands):
        raise ValueError(f"{location} has an invalid num_controls")
    if not all(
        operand.type.is_quantum()
        for operand in operation.operands[: operation.num_controls]
    ):
        raise ValueError(f"{location} controls must be quantum values")
    _validate_control_activation(
        operation.control_value,
        operation.num_controls,
        location,
    )
    _validate_controlled_results(operation, location)


def _validate_control_activation(
    control_value: object,
    num_controls: int,
    location: str,
) -> None:
    """Validate a canonical coherent-control activation value.

    Args:
        control_value (object): Candidate LSB-first activation integer or null.
        num_controls (int): Concrete width of the control register.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If the value is malformed, does not fit the control width,
            or uses the non-canonical explicit all-ones representation.
    """
    if num_controls == 0:
        if control_value is not None:
            raise ValueError(f"{location} control_value requires a control qubit")
        return
    try:
        normalized = normalize_control_value(
            cast("int | None", control_value),
            num_controls,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} has an invalid control_value") from exc
    if normalized != control_value:
        raise ValueError(f"{location} has a non-canonical control_value")


def _validate_symbolic_controlled(
    operation: SymbolicControlledU,
    location: str,
) -> None:
    """Validate a symbolic controlled-unitary layout.

    Args:
        operation (SymbolicControlledU): Operation to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If symbolic controls, indices, or results are invalid.
    """
    _require_value_type(operation.num_controls, UIntType(), location)
    if operation.num_control_args < 1 or operation.num_control_args > len(
        operation.operands
    ):
        raise ValueError(f"{location} has an invalid num_control_args")
    if not all(
        operand.type.is_quantum()
        for operand in operation.operands[: operation.num_control_args]
    ):
        raise ValueError(f"{location} control arguments must be quantum values")
    if operation.control_indices is not None:
        _require_types(
            operation.control_indices,
            [UIntType()] * len(operation.control_indices),
            location,
            "control index",
        )
    _validate_controlled_results(operation, location)


def _validate_controlled_results(
    operation: ConcreteControlledU | SymbolicControlledU,
    location: str,
) -> None:
    """Require one quantum result for every quantum controlled operand.

    Args:
        operation (ConcreteControlledU | SymbolicControlledU): Controlled op.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If the result count or result types are invalid.
    """
    quantum_operands = [
        value for value in operation.operands if value.type.is_quantum()
    ]
    if len(operation.results) != len(quantum_operands):
        raise ValueError(f"{location} results must mirror quantum operands")
    if not all(result.type.is_quantum() for result in operation.results):
        raise ValueError(f"{location} results must be quantum values")


def _validate_invoke(operation: InvokeOperation, location: str) -> None:
    """Validate a callable invocation against its explicit signature.

    Args:
        operation (InvokeOperation): Invocation to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If the target definition or its signature disagrees.
    """
    if operation.definition is None:
        raise ValueError(f"{location} requires a callable definition")
    if operation.definition.ref != operation.target:
        raise ValueError(f"{location} target disagrees with its definition")
    operation_kind = operation.attrs.get("kind")
    definition_kind = operation.definition.attrs.get("kind")
    if operation_kind != definition_kind:
        raise ValueError(f"{location} kind disagrees with its definition")
    control_count = 0
    if operation.transform is CallTransform.CONTROLLED:
        raw_control_count = operation.attrs.get("num_control_qubits")
        if (
            not isinstance(raw_control_count, int)
            or isinstance(raw_control_count, bool)
            or raw_control_count < 1
        ):
            raise ValueError(
                f"{location} controlled transform requires a positive integer "
                "num_control_qubits"
            )
        control_count = raw_control_count
        if control_count > len(operation.operands) or control_count > len(
            operation.results
        ):
            raise ValueError(f"{location} has fewer values than declared controls")
        if not all(
            value.type.is_quantum()
            for value in operation.operands[:control_count]
            + operation.results[:control_count]
        ):
            raise ValueError(f"{location} control inputs and results must be quantum")
        if (
            "control_value" in operation.attrs
            and operation.attrs["control_value"] is None
        ):
            raise ValueError(f"{location} has a non-canonical null control_value")
        _validate_control_activation(
            operation.attrs.get("control_value"),
            control_count,
            location,
        )
    elif "control_value" in operation.attrs:
        raise ValueError(f"{location} has control_value on a non-controlled invocation")
    signature = operation.definition.signature
    if signature is None:
        return
    signature_includes_controls = operation_kind == "oracle"
    signature_offset = 0 if signature_includes_controls else control_count
    operands = operation.operands[signature_offset:]
    results = operation.results[signature_offset:]
    if len(signature.operands) != len(operands) or len(signature.results) != len(
        results
    ):
        raise ValueError(f"{location} arity disagrees with its callable signature")
    for index, (value, hint) in enumerate(
        zip(operands, signature.operands, strict=True)
    ):
        if hint is not None and value.type != hint.type:
            raise ValueError(
                f"{location} operand {index} type disagrees with signature"
            )
    _require_types(
        results,
        [hint.type for hint in signature.results],
        location,
        "result",
    )


def _validate_static_binding_invoke(
    operation: InvokeOperation,
    state: _ValidationState,
    location: str,
) -> None:
    """Validate one deferred static-binding member invocation.

    Static member references are executable capabilities supplied only after
    deserialization. The serialized marker must therefore identify a known
    root slot and a closed, registered member ABI before any concrete binding
    is accepted.

    Args:
        operation (InvokeOperation): Invocation to inspect.
        state (_ValidationState): Root static-binding registry and graph state.
        location (str): Human-readable graph location.

    Raises:
        ValueError: If marker metadata, policy, identity, or callable ABI is
            malformed or names an unregistered slot or member.
    """
    definition = operation.definition
    if definition is None:
        return
    body_ref = definition.body_ref
    static_ref_present = any(
        ref is not None and ref.kind == "static_binding"
        for ref in (
            body_ref,
            *(implementation.body_ref for implementation in definition.implementations),
        )
    )
    static_attrs_present = (
        operation.attrs.get("kind") == "static_binding"
        or definition.attrs.get("kind") == "static_binding"
    )
    if not static_ref_present and not static_attrs_present:
        return
    if body_ref is None or body_ref.kind != "static_binding":
        raise ValueError(
            f"{location} static binding marker must use the definition body_ref"
        )
    if operation.body_ref != body_ref:
        raise ValueError(
            f"{location} static binding marker selects an inconsistent body_ref"
        )
    if definition.body is not None or definition.implementations:
        raise ValueError(
            f"{location} static binding marker cannot embed implementations"
        )
    if definition.opaque_cost is not None:
        raise ValueError(f"{location} static binding marker cannot set opaque_cost")

    marker = body_ref.attrs
    if set(marker) != {"slot", "type_key", "member"} or not all(
        isinstance(value, str) and value for value in marker.values()
    ):
        raise ValueError(
            f"{location} static binding body_ref requires exactly non-empty "
            "slot, type_key, and member strings"
        )
    slot_name = cast(str, marker["slot"])
    type_key = cast(str, marker["type_key"])
    member_name = cast(str, marker["member"])
    context = state.static_bindings.get(slot_name)
    if context is None:
        raise ValueError(
            f"{location} refers to unknown static binding slot {slot_name!r}"
        )
    slot, spec = context
    if slot.type_key != type_key or spec.type_key != type_key:
        raise ValueError(
            f"{location} static binding marker type {type_key!r} disagrees "
            f"with slot {slot_name!r}"
        )
    member_spec = spec.members.get(member_name)
    if member_spec is None:
        raise ValueError(
            f"{location} refers to unknown static binding member {member_name!r}"
        )

    expected_ref = CallableRef(
        namespace="qamomile.static_binding",
        name=f"{type_key}.{member_name}",
    )
    if (
        operation.target != expected_ref
        or definition.ref != expected_ref
        or body_ref.ref != expected_ref
    ):
        raise ValueError(
            f"{location} static binding marker has an inconsistent callable reference"
        )
    base_attrs = {
        "kind": "static_binding",
        "default_policy": CallPolicy.PRESERVE_BOX.name,
        **marker,
    }
    if definition.attrs != base_attrs:
        raise ValueError(
            f"{location} static binding definition attributes are non-canonical"
        )
    allowed_operation_attrs = {
        *base_attrs,
        "num_control_qubits",
        "num_target_qubits",
        "control_value",
        "strategy_name",
    }
    if any(operation.attrs.get(key) != value for key, value in base_attrs.items()) or (
        set(operation.attrs) - allowed_operation_attrs
    ):
        raise ValueError(
            f"{location} static binding invocation attributes are non-canonical"
        )
    if definition.default_policy is not CallPolicy.PRESERVE_BOX:
        raise ValueError(f"{location} static binding definition must use PRESERVE_BOX")

    _validate_static_binding_signature(
        definition,
        member_spec,
        operation,
        location,
    )


def _validate_static_binding_signature(
    definition: CallableDef,
    member_spec: StaticBindingMemberSpec,
    operation: InvokeOperation,
    location: str,
) -> None:
    """Validate a static marker's registered vector pass-through ABI.

    Args:
        definition (CallableDef): Deferred callable definition.
        member_spec (StaticBindingMemberSpec): Installed member contract.
        operation (InvokeOperation): Invocation carrying concrete IR values.
        location (str): Human-readable graph location.

    Raises:
        ValueError: If the registered or serialized ABI is not the exact
            quantum-vector pass-through form supported by static members.
    """
    input_names = list(member_spec.input_types)
    if any(
        annotation != Vector[Qubit] for annotation in member_spec.input_types.values()
    ):
        raise ValueError(
            f"{location} static binding member inputs must be quantum vectors"
        )
    if list(member_spec.output_types) != [Vector[Qubit]] * len(input_names):
        raise ValueError(
            f"{location} static binding member outputs must mirror its inputs"
        )
    signature = definition.signature
    if signature is None:
        raise ValueError(f"{location} static binding definition requires a signature")
    if any(hint is None for hint in (*signature.operands, *signature.results)):
        raise ValueError(
            f"{location} static binding signature requires explicit type hints"
        )
    operand_hints = cast(list[ParamHint], signature.operands)
    result_hints = cast(list[ParamHint], signature.results)
    if [hint.name for hint in operand_hints] != input_names or any(
        hint.type != QubitType() for hint in operand_hints
    ):
        raise ValueError(
            f"{location} static binding operand signature disagrees with its adapter"
        )
    expected_result_names = [
        f"result_{index}" for index in range(len(member_spec.output_types))
    ]
    if [hint.name for hint in result_hints] != expected_result_names or any(
        hint.type != QubitType() for hint in result_hints
    ):
        raise ValueError(
            f"{location} static binding result signature disagrees with its adapter"
        )
    if operation.parameters:
        raise ValueError(
            f"{location} static binding member cannot have classical operands"
        )
    targets = operation.target_qubits
    target_results = operation.results[operation.num_control_qubits :]
    if len(targets) != len(input_names) or len(target_results) != len(
        member_spec.output_types
    ):
        raise ValueError(
            f"{location} static binding invocation arity disagrees with its adapter"
        )
    if not all(
        isinstance(value, ArrayValue) and value.type == QubitType()
        for value in (*targets, *target_results)
    ):
        raise ValueError(
            f"{location} static binding operands and results must be quantum vectors"
        )
    if any(
        operand.logical_id != result.logical_id
        for operand, result in zip(targets, target_results, strict=True)
    ):
        raise ValueError(
            f"{location} static binding results must preserve logical vector order"
        )


def _validate_inverse_block(
    operation: InverseBlockOperation,
    location: str,
) -> None:
    """Validate required semantic bodies of an inverse operation.

    Args:
        operation (InverseBlockOperation): Inverse operation to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If neither source nor implementation body is available or
            a control operand is a quantum array instead of a scalar qubit.
    """
    if operation.source_block is None and operation.implementation_block is None:
        raise ValueError(f"{location} requires a source or implementation block")
    if any(
        isinstance(value, ArrayValue)
        for value in operation.operands[: operation.num_control_qubits]
    ):
        raise ValueError(f"{location} control operands must be scalar qubits")
    _validate_control_activation(
        operation.control_value,
        operation.num_control_qubits,
        location,
    )


def _validate_select(operation: SelectOperation, location: str) -> None:
    """Validate a SELECT operation and every case interface.

    Args:
        operation (SelectOperation): Multiplexer operation to validate.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If the index width, argument grouping, quantum result
            layout, or case block interfaces are inconsistent.
    """
    width = operation.num_index_qubits
    num_index_args = operation.num_index_args
    if num_index_args < 1 or num_index_args >= len(operation.operands):
        raise ValueError(f"{location} has an invalid num_index_args")

    if is_plain_int(width):
        concrete_width = cast(int, width)
        if concrete_width < 1 or num_index_args != concrete_width:
            raise ValueError(f"{location} has an invalid num_index_qubits")
        minimum_width = (len(operation.case_blocks) - 1).bit_length()
        if len(operation.case_blocks) < 2 or concrete_width < minimum_width:
            raise ValueError(f"{location} has an invalid number of case blocks")
    else:
        if not isinstance(width, Value) or isinstance(width, ArrayValue):
            raise ValueError(f"{location} has an invalid num_index_qubits")
        _require_value_type(width, UIntType(), location)
        if len(operation.case_blocks) < 2:
            raise ValueError(f"{location} has an invalid number of case blocks")

    index_operands = operation.operands[:num_index_args]
    if not all(value.type.is_quantum() for value in index_operands):
        raise ValueError(f"{location} index arguments must be quantum values")
    _require_types(
        index_operands,
        [QubitType()] * num_index_args,
        location,
        "index argument",
    )
    if is_plain_int(width):
        if any(isinstance(value, ArrayValue) for value in index_operands):
            raise ValueError(
                f"{location} concrete index operands must be scalar qubits"
            )

    target_operands = [
        value
        for value in operation.operands[num_index_args:]
        if value.type.is_quantum()
    ]
    if not target_operands:
        raise ValueError(f"{location} requires at least one quantum target")
    quantum_operands = [*index_operands, *target_operands]
    if len(operation.results) != len(quantum_operands):
        raise ValueError(f"{location} results must mirror quantum operands")
    if any(
        isinstance(operand, ArrayValue) != isinstance(result, ArrayValue)
        for operand, result in zip(
            quantum_operands,
            operation.results,
            strict=True,
        )
    ):
        raise ValueError(f"{location} results must preserve quantum argument grouping")
    _require_types(
        operation.results,
        [value.type for value in quantum_operands],
        location,
        "result",
    )

    case_inputs = operation.operands[num_index_args:]
    case_outputs = target_operands
    for case_index, case_block in enumerate(operation.case_blocks):
        case_location = f"{location} case block {case_index}"
        if len(case_block.input_values) != len(case_inputs):
            raise ValueError(f"{case_location} input arity disagrees with SELECT")
        case_input_pairs = pair_block_operands(case_block, case_inputs)
        if len(case_input_pairs) != len(case_inputs):
            raise ValueError(f"{case_location} input categories disagree with SELECT")
        _require_types(
            [formal for formal, _ in case_input_pairs],
            [actual.type for _, actual in case_input_pairs],
            case_location,
            "input",
        )
        _require_types(
            case_block.output_values,
            [value.type for value in case_outputs],
            case_location,
            "output",
        )


def _validate_quantum_operand_uniqueness(
    operation: Operation,
    location: str,
) -> None:
    """Reject use of one quantum SSA value in multiple operand positions.

    Args:
        operation (Operation): Operation to inspect.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If a quantum operand UUID occurs more than once.
    """
    quantum_uuids = [
        value.uuid for value in operation.operands if value.type.is_quantum()
    ]
    if len(quantum_uuids) != len(set(quantum_uuids)):
        raise ValueError(f"{location} repeats a quantum operand UUID")


def _require_arity(
    operation: Operation,
    operands: int,
    results: int,
    location: str,
) -> None:
    """Require exact operation operand and result counts.

    Args:
        operation (Operation): Operation to inspect.
        operands (int): Required operand count.
        results (int): Required result count.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If either count differs.
    """
    _require_operand_count(operation, operands, location)
    _require_result_count(operation, results, location)


def _require_operand_count(
    operation: Operation,
    expected: int,
    location: str,
) -> None:
    """Require an exact operation operand count.

    Args:
        operation (Operation): Operation to inspect.
        expected (int): Required operand count.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If the count differs.
    """
    if len(operation.operands) != expected:
        raise ValueError(
            f"{location} requires {expected} operands, got {len(operation.operands)}"
        )


def _require_result_count(
    operation: Operation,
    expected: int,
    location: str,
) -> None:
    """Require an exact operation result count.

    Args:
        operation (Operation): Operation to inspect.
        expected (int): Required result count.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If the count differs.
    """
    if len(operation.results) != expected:
        raise ValueError(
            f"{location} requires {expected} results, got {len(operation.results)}"
        )


def _require_types(
    values: Iterable[ValueBase],
    expected: Iterable[ValueType],
    location: str,
    role: str,
) -> None:
    """Require positional values to have exact IR types.

    Args:
        values (Iterable[ValueBase]): Values to inspect.
        expected (Iterable[ValueType]): Expected IR type objects.
        location (str): Human-readable operation location.
        role (str): Diagnostic role such as ``operand`` or ``result``.

    Raises:
        ValueError: If a positional type differs.
    """
    for index, (value, expected_type) in enumerate(zip(values, expected, strict=True)):
        if value.type != expected_type:
            raise ValueError(
                f"{location} {role} {index} has type {value.type.label()}, "
                f"expected {expected_type.label()}"
            )


def _require_value_type(
    value: ValueBase,
    expected: ValueType,
    location: str,
) -> None:
    """Require one value to have an exact IR type.

    Args:
        value (ValueBase): Value to inspect.
        expected (ValueType): Expected IR type object.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If the value's type differs.
    """
    if value.type != expected:
        raise ValueError(
            f"{location} value has type {value.type.label()}, "
            f"expected {expected.label()}"
        )


def _require_scalar_type(
    value: ValueBase,
    expected: ValueType,
    location: str,
) -> None:
    """Require a scalar Value with a specific IR type.

    Args:
        value (ValueBase): Value to inspect.
        expected (ValueType): Expected IR type object.
        location (str): Human-readable value location.

    Raises:
        ValueError: If the value is not a matching scalar.
    """
    if (
        not isinstance(value, Value)
        or isinstance(value, ArrayValue)
        or value.type != expected
    ):
        raise ValueError(f"{location} requires a scalar {expected.label()}")


def _require_array_type(
    value: ValueBase,
    expected: ValueType,
    location: str,
) -> None:
    """Require an ArrayValue with a specific element type.

    Args:
        value (ValueBase): Value to inspect.
        expected (ValueType): Expected element IR type.
        location (str): Human-readable operation location.

    Raises:
        ValueError: If the value is not a matching array.
    """
    if not isinstance(value, ArrayValue) or value.type != expected:
        raise ValueError(f"{location} requires an ArrayValue[{expected.label()}]")


def _require_unique_values(values: Iterable[ValueBase], location: str) -> None:
    """Require a value sequence to contain no duplicate UUIDs.

    Args:
        values (Iterable[ValueBase]): Values to inspect.
        location (str): Human-readable sequence location.

    Raises:
        ValueError: If a UUID occurs more than once.
    """
    uuids = [value.uuid for value in values]
    if len(uuids) != len(set(uuids)):
        raise ValueError(f"{location} contain duplicate UUIDs")
