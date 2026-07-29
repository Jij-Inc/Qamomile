"""Dependency scheduling and logical-width liveness analysis.

The helpers in this module map IR values to physical wire identities, schedule
resource summaries along those dependencies, and track allocation liveness.
They consume metric records without owning interpretation or decomposition
policy.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import sympy as sp

from qamomile.circuit.estimator._metrics import (
    _ONE,
    _ZERO,
    DepthResources,
    EstimateQuality,
    ResourceAssumption,
    ResourceExpr,
    WidthResources,
    _and_conditions,
    _boolean_condition,
    _conditional_depth,
    _expr,
    _is_concrete_integer,
    _is_structurally_nonnegative,
    _maximum_expr_over_range,
    _piecewise,
    _resource_activity_condition,
    _resource_expr,
    _resource_max,
    _resource_max_many,
    _safe_simplify,
    _symbol_display_name,
)
from qamomile.circuit.estimator._resolver import (
    ExprResolver,
    input_shape_dimension_aliases,
)
from qamomile.circuit.ir._resource_contract import quantum_operand_widths
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import CallTransform, InvokeOperation
from qamomile.circuit.ir.operation.classical_ops import (
    ReturnQuantumArrayElementOperation,
    StoreArrayElementOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    GateOperation,
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
    ProjectOperation,
    ResetOperation,
)
from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import Operation, QInitOperation
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.types.primitives import QubitType
from qamomile.circuit.ir.types.q_register import QFixedType, QUIntType
from qamomile.circuit.ir.value import (
    ArrayValue,
    Value,
    ValueBase,
    split_indexed_identifier,
)
from qamomile.circuit.transpiler.block_parameter_binding import pair_block_operands

if TYPE_CHECKING:
    from qamomile.circuit.estimator.resource_estimator import ResourceEstimate
    from qamomile.circuit.frontend.qkernel import QKernel

WireKey = tuple[str, int | None]


def _estimate_has_nonzero_depth(estimate: ResourceEstimate) -> bool:
    """Return whether any depth metric is structurally nonzero.

    Args:
        estimate (ResourceEstimate): Estimate whose depth should be inspected.

    Returns:
        bool: Whether at least one depth field is not the exact zero
            expression.
    """
    return any(
        getattr(estimate.depth, field.name) != _ZERO
        for field in dataclasses.fields(DepthResources)
    )


def _merge_dependency_keys(
    left: ResourceEstimate,
    right: ResourceEstimate,
) -> frozenset[WireKey] | None:
    """Merge proven dependency masks without erasing unknown footprints.

    ``None`` means that the enclosing operation must supply a conservative
    footprint, while an empty set means that the estimate has proven no
    caller-visible blocking wires. A structurally zero estimate is therefore
    the neutral element even when its default mask is ``None``.

    Args:
        left (ResourceEstimate): Left composition operand.
        right (ResourceEstimate): Right composition operand.

    Returns:
        frozenset[WireKey] | None: Union of known masks, or ``None`` when a
            nonzero operand still has an unknown footprint.
    """

    def normalized(estimate: ResourceEstimate) -> frozenset[WireKey] | None:
        """Treat a zero-depth unknown mask as the empty dependency set.

        Args:
            estimate (ResourceEstimate): Estimate whose mask is normalized.

        Returns:
            frozenset[WireKey] | None: Explicit dependency mask or ``None``.
        """
        if estimate._dependency_keys is not None:
            return estimate._dependency_keys
        if not _estimate_has_nonzero_depth(estimate):
            return frozenset()
        return None

    left_keys = normalized(left)
    right_keys = normalized(right)
    if left_keys is None or right_keys is None:
        return None
    return frozenset((*left_keys, *right_keys))


class _LocalBlock:
    """Provide a minimal operation container for nested resource scopes.

    Args:
        operations (list[Operation]): Operations visible in the nested scope.
    """

    __slots__ = ("operations",)

    def __init__(self, operations: list[Operation]) -> None:
        """Initialize a local operation container.

        Args:
            operations (list[Operation]): Operations visible in the nested
                scope.
        """
        self.operations = operations


def _specialize_dependency_expression(
    expression: ResourceExpr,
    scalar_values: Mapping[str, sp.Expr] | None,
    used_names: set[str] | None = None,
) -> ResourceExpr:
    """Apply supplied scalar inputs only for physical dependency resolution.

    Resource formulas remain symbolic elsewhere. Dependency scheduling may
    nevertheless use a supplied array/control index to distinguish physical
    wires without expanding a problem-sized loop or circuit.

    Args:
        expression (ResourceExpr): Resolved symbolic index or view expression.
        scalar_values (Mapping[str, sp.Expr] | None): Supplied numeric values
            keyed by qkernel argument name.
        used_names (set[str] | None): Optional set updated with names that
            participated in dependency resolution.

    Returns:
        ResourceExpr: Expression specialized by matching supplied values.
    """
    if not scalar_values or not expression.free_symbols:
        return expression
    substitutions: dict[sp.Symbol, sp.Expr] = {}
    for symbol in expression.free_symbols:
        if isinstance(symbol, sp.Dummy):
            continue
        typed_symbol = cast(sp.Symbol, symbol)
        name = _symbol_display_name(typed_symbol)
        if name not in scalar_values:
            continue
        substitutions[typed_symbol] = scalar_values[name]
        if used_names is not None:
            used_names.add(name)
    if not substitutions:
        return expression
    return cast(
        ResourceExpr,
        expression.subs(substitutions, simultaneous=True).doit(),
    )


def _quantum_element_index_expression(
    value: Value,
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> ResourceExpr | None:
    """Resolve an array element to its root-array index expression.

    Args:
        value (Value): Quantum scalar that may carry array-element ancestry.
        resolver (ExprResolver): Resolver for symbolic indices and view bounds.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            values used only to resolve physical indices. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with dependency input
            names. Defaults to ``None``.

    Returns:
        ResourceExpr | None: Root-array index expression, or ``None`` when the
            value is not a one-dimensional array element.
    """
    if value.parent_array is None or len(value.element_indices) != 1:
        return None
    resolved = _resolve_root_array_index_expression(
        value.parent_array,
        resolver.resolve(value.element_indices[0]),
        resolver,
        scalar_values,
        used_names,
    )
    return None if resolved is None else resolved[1]


def _resolve_root_array_index_expression(
    array: ArrayValue,
    local_index: ResourceExpr,
    resolver: ExprResolver,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> tuple[ArrayValue, ResourceExpr] | None:
    """Compose one array index through a validated symbolic view chain.

    This is the resolver-aware counterpart of
    :func:`resolve_root_array_index`. Each concrete affine component is
    checked at the hop where it appears so malformed raw IR cannot compose an
    invalid negative stride into an apparently valid root index.

    Args:
        array (ArrayValue): Array whose local index is being resolved.
        local_index (ResourceExpr): Index relative to ``array``.
        resolver (ExprResolver): Resolver for symbolic view bounds.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        tuple[ArrayValue, ResourceExpr] | None: Root array and composed index,
        or ``None`` when a component is missing, unresolved for a concrete
        address, or violates the nonnegative-start/positive-step contract.
    """

    def valid_concrete_component(
        expression: ResourceExpr,
        *,
        positive: bool,
    ) -> bool:
        """Validate one concrete index, start, or stride expression.

        Args:
            expression (ResourceExpr): Component to inspect.
            positive (bool): Require strict positivity instead of
                nonnegativity.

        Returns:
            bool: ``False`` only for a concrete malformed component.
        """
        if not expression.is_number:
            return True
        if not _is_concrete_integer(expression):
            return False
        return bool(expression > 0) if positive else bool(expression >= 0)

    index = _specialize_dependency_expression(
        local_index,
        scalar_values,
        used_names,
    )
    if not valid_concrete_component(index, positive=False):
        return None
    current = array
    while current.slice_of is not None:
        if current.slice_start is None or current.slice_step is None:
            return None
        start = _specialize_dependency_expression(
            resolver.resolve(current.slice_start),
            scalar_values,
            used_names,
        )
        step = _specialize_dependency_expression(
            resolver.resolve(current.slice_step),
            scalar_values,
            used_names,
        )
        if not valid_concrete_component(
            start,
            positive=False,
        ) or not valid_concrete_component(step, positive=True):
            return None
        index = cast(ResourceExpr, start + step * index)
        current = current.slice_of
    return current, index


def _quantum_element_wire_index(
    value: Value,
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> int | None:
    """Resolve an array element to its concrete root-array scalar index.

    Args:
        value (Value): Quantum scalar that may carry array-element ancestry.
        resolver (ExprResolver): Resolver for symbolic indices and view bounds.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            values used only to resolve physical indices. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with dependency input
            names. Defaults to ``None``.

    Returns:
        int | None: Nonnegative scalar index in the root array, or ``None``
            when the element or any view mapping remains unresolved.
    """
    index = _quantum_element_index_expression(
        value,
        resolver,
        scalar_values=scalar_values,
        used_names=used_names,
    )
    if (
        index is None
        or not index.is_number
        or not _is_concrete_integer(index)
        or index < 0
    ):
        return None
    return int(index)


def _quantum_value_wire_keys(
    value: Value,
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> set[tuple[str, int | None]]:
    """Return dependency keys for one quantum value.

    A scalar array element uses its root allocation and physical scalar index.
    A whole register, view, or unresolved element uses an owner-wide key whose
    ``None`` index aliases every scalar key for that allocation. Independent
    scalar qubits use their own logical identity with an owner-wide key.

    Args:
        value (Value): Quantum scalar, array, or array view.
        resolver (ExprResolver): Resolver for element and view indices.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        set[tuple[str, int | None]]: Root-owner and optional scalar-index keys.
    """
    carrier_keys = _cast_carrier_wire_keys(value)
    if carrier_keys is not None:
        return carrier_keys
    owner = _quantum_allocation_owner(value)
    if isinstance(value, ArrayValue):
        if value.slice_of is None:
            return {(owner, None)}
        size = _specialize_dependency_expression(
            _qubit_value_size(value, resolver),
            scalar_values,
            used_names,
        )
        if size.is_number and _is_concrete_integer(size) and 0 <= size <= 4096:
            return {
                _array_wire_key_at_index(
                    value,
                    index,
                    resolver,
                    scalar_values=scalar_values,
                    used_names=used_names,
                )
                for index in range(int(size))
            }
        return {(owner, None)}
    if value.parent_array is None:
        return {(owner, None)}
    return {
        (
            owner,
            _quantum_element_wire_index(
                value,
                resolver,
                scalar_values=scalar_values,
                used_names=used_names,
            ),
        )
    }


def _array_wire_key_at_index(
    array: ArrayValue,
    index: int,
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> WireKey:
    """Map one concrete array slot through any caller-side view chain.

    Args:
        array (ArrayValue): Actual array or view supplied at a call boundary.
        index (int): Concrete element index relative to ``array``.
        resolver (ExprResolver): Resolver for view starts and strides.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        WireKey: Root-owner scalar key, or an owner-wide key when a view
            offset cannot be resolved safely.
    """
    owner = _quantum_allocation_owner(array)
    resolved = _resolve_root_array_index_expression(
        array,
        sp.Integer(index),
        resolver,
        scalar_values,
        used_names,
    )
    if resolved is None:
        return owner, None
    resolved_index = resolved[1]
    if (
        not resolved_index.is_number
        or not _is_concrete_integer(resolved_index)
        or resolved_index < 0
    ):
        return owner, None
    return owner, int(resolved_index)


def _map_value_dependency_keys(
    source: Value,
    actual: Value,
    dependency_keys: frozenset[WireKey],
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> set[WireKey]:
    """Map dependency keys owned by one body value onto a caller value.

    Args:
        source (Value): Body input or output whose owner appears in the mask.
        actual (Value): Caller-side operand or result corresponding to source.
        dependency_keys (frozenset[WireKey]): Body-scoped touched-wire mask.
        resolver (ExprResolver): Caller resolver for arrays and views.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        set[WireKey]: Caller-scoped dependency keys for this correspondence.
    """
    source_owner = _quantum_allocation_owner(source)
    mapped: set[WireKey] = set()
    for owner, index in dependency_keys:
        if owner != source_owner:
            continue
        if index is None or not isinstance(actual, ArrayValue):
            mapped.update(
                _quantum_value_wire_keys(
                    actual,
                    resolver,
                    scalar_values=scalar_values,
                    used_names=used_names,
                )
            )
        else:
            mapped.add(
                _array_wire_key_at_index(
                    actual,
                    index,
                    resolver,
                    scalar_values=scalar_values,
                    used_names=used_names,
                )
            )
    return mapped


def _map_body_dependency_keys(
    block: Block,
    body_estimate: ResourceEstimate,
    actual_operands: Sequence[ValueBase],
    caller_results: Sequence[ValueBase],
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> frozenset[WireKey] | None:
    """Translate a body's touched inputs and outputs to caller wire keys.

    Callee-local scratch allocations deliberately disappear at the boundary;
    their depth remains part of the call duration but cannot block unrelated
    caller wires. Returned allocations are mapped through ``output_values`` so
    a caller gate cannot start before the producing body work completes.

    Args:
        block (Block): Evaluated callable implementation.
        body_estimate (ResourceEstimate): Body-scoped estimate carrying its
            touched-wire mask.
        actual_operands (Sequence[ValueBase]): Caller operands aligned by the
            shared block-pairing convention.
        caller_results (Sequence[ValueBase]): Caller results aligned with the
            block outputs.
        resolver (ExprResolver): Caller-side value resolver.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        frozenset[WireKey] | None: Caller-scoped mask, or ``None`` when the
            body did not provide an authoritative mask.
    """
    dependency_keys = body_estimate._dependency_keys
    if dependency_keys is None:
        return None
    mapped: set[WireKey] = set()
    for formal, actual in pair_block_operands(block, actual_operands):
        if (
            isinstance(formal, Value)
            and isinstance(actual, Value)
            and formal.type.is_quantum()
            and actual.type.is_quantum()
        ):
            mapped.update(
                _map_value_dependency_keys(
                    formal,
                    actual,
                    dependency_keys,
                    resolver,
                    scalar_values=scalar_values,
                    used_names=used_names,
                )
            )
    if len(block.output_values) == len(caller_results):
        for output, result in zip(
            block.output_values,
            caller_results,
            strict=True,
        ):
            if (
                isinstance(output, Value)
                and isinstance(result, Value)
                and output.type.is_quantum()
                and result.type.is_quantum()
            ):
                mapped.update(
                    _map_value_dependency_keys(
                        output,
                        result,
                        dependency_keys,
                        resolver,
                        scalar_values=scalar_values,
                        used_names=used_names,
                    )
                )
    return frozenset(mapped)


def _wire_keys_for_values(
    values: Sequence[Value],
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> set[WireKey]:
    """Collect caller dependency keys for quantum values.

    Args:
        values (Sequence[Value]): Values whose physical keys are needed.
        resolver (ExprResolver): Resolver for array elements and views.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        set[WireKey]: Union of all quantum value footprints.
    """
    keys: set[WireKey] = set()
    for value in values:
        if value.type.is_quantum():
            keys.update(
                _quantum_value_wire_keys(
                    value,
                    resolver,
                    scalar_values=scalar_values,
                    used_names=used_names,
                )
            )
    return keys


def _controlled_u_control_wire_keys(
    operation: ControlledUOperation,
    resolved_indices: Sequence[ResourceExpr],
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> set[WireKey]:
    """Return only the active control-pool wires of a controlled call.

    Args:
        operation (ControlledUOperation): Controlled operation whose control
            prefix may be a selected array pool.
        resolved_indices (Sequence[ResourceExpr]): Resolved ``control_indices``
            values, empty when the whole prefix is active.
        resolver (ExprResolver): Caller-side value resolver.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        set[WireKey]: Selected physical control keys. Unresolved selections
            conservatively use the whole pool owner.
    """
    controls = operation.control_operands
    if not resolved_indices:
        return _wire_keys_for_values(
            controls,
            resolver,
            scalar_values=scalar_values,
            used_names=used_names,
        )
    if len(controls) != 1 or not isinstance(controls[0], ArrayValue):
        return _wire_keys_for_values(
            controls,
            resolver,
            scalar_values=scalar_values,
            used_names=used_names,
        )
    pool = controls[0]
    keys: set[WireKey] = set()
    for index in resolved_indices:
        if index.is_number and _is_concrete_integer(index) and index >= 0:
            keys.add(
                _array_wire_key_at_index(
                    pool,
                    int(index),
                    resolver,
                    scalar_values=scalar_values,
                    used_names=used_names,
                )
            )
        else:
            keys.add((_quantum_allocation_owner(pool), None))
    return keys


def _with_aggregate_boundary_depth_metadata(
    estimate: ResourceEstimate,
    dependency_estimate: ResourceEstimate,
    *,
    source: str,
    boundary: str,
    active_when: sp.Basic | None = None,
) -> ResourceEstimate:
    """Mark a multi-wire aggregate boundary as a conservative upper bound.

    One scalar duration is attached to every touched outer wire. This can
    over-serialize a later gate when independent touched wires finish at
    different internal layers.

    Args:
        estimate (ResourceEstimate): Outer-scoped estimate to classify.
        dependency_estimate (ResourceEstimate): Estimate whose touched-wire
            mask determines whether aggregation is conservative.
        source (str): Operation or callable name for the assumption.
        boundary (str): Human-readable boundary kind, such as ``"call"`` or
            ``"control-flow"``.
        active_when (sp.Basic | None): Optional boundary-activity condition.
            Defaults to whether the aggregate depth is positive.

    Returns:
        ResourceEstimate: Estimate with conditional quality metadata.
    """
    keys = dependency_estimate._dependency_keys
    if (
        keys is None
        or len(keys) <= 1
        or not _estimate_has_nonzero_depth(dependency_estimate)
    ):
        return estimate
    assumption = ResourceAssumption(
        f"{boundary} boundary depth conservatively applies one aggregate "
        "latency to multiple touched wires",
        source=source,
    )
    condition = (
        _resource_activity_condition(estimate.depth.depth)
        if active_when is None
        else active_when
    )
    return estimate._with_metadata(
        assumptions=(assumption,),
        quality=EstimateQuality.UPPER_BOUND,
        active_when=condition,
    )


def _with_body_boundary_depth_metadata(
    estimate: ResourceEstimate,
    body_estimate: ResourceEstimate,
    *,
    source: str,
    zero_controls: ResourceExpr | int = 0,
) -> ResourceEstimate:
    """Classify aggregate call depth and open-control exit latency.

    Args:
        estimate (ResourceEstimate): Caller-scoped body estimate.
        body_estimate (ResourceEstimate): Original body-scoped estimate.
        source (str): Callable name for the modeling assumption.
        zero_controls (ResourceExpr | int): Open-control X brackets surrounding
            the body. Defaults to zero.

    Returns:
        ResourceEstimate: Estimate with conservative boundary metadata.
    """
    keys = body_estimate._dependency_keys
    if keys is None or not _estimate_has_nonzero_depth(body_estimate):
        return estimate
    estimate = _with_aggregate_boundary_depth_metadata(
        estimate,
        body_estimate,
        source=source,
        boundary="call",
    )
    bracket_condition = _and_conditions(
        sp.Gt(_expr(zero_controls), _ZERO),
        _resource_activity_condition(estimate.depth.depth),
    )
    if keys and bracket_condition is not sp.false:
        assumption = ResourceAssumption(
            "open-control brackets may finish on a different layer than the "
            "controlled body targets",
            source=source,
        )
        estimate = estimate._with_metadata(
            assumptions=(assumption,),
            quality=EstimateQuality.UPPER_BOUND,
            active_when=bracket_condition,
        )
    return estimate


def _quantum_wire_keys(
    operation: Operation,
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> tuple[set[tuple[str, int | None]], set[tuple[str, int | None]]]:
    """Collect quantum logical wires read and written by one operation.

    Args:
        operation (Operation): Operation whose dependency footprint is needed.
        resolver (ExprResolver): Resolver for array element and view indices.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        tuple[set[tuple[str, int | None]], set[tuple[str, int | None]]]:
            Physical owner/index keys read and written. Nested control flow
            conservatively treats every touched wire as both read and written
            at the enclosing boundary.
    """
    reads: set[tuple[str, int | None]] = set()
    for value in operation.all_input_values():
        if isinstance(value, Value) and value.type.is_quantum():
            reads.update(
                _quantum_value_wire_keys(
                    value,
                    resolver,
                    scalar_values=scalar_values,
                    used_names=used_names,
                )
            )
    writes: set[tuple[str, int | None]] = set()
    for value in operation.results:
        if value.type.is_quantum():
            writes.update(
                _quantum_value_wire_keys(
                    value,
                    resolver,
                    scalar_values=scalar_values,
                    used_names=used_names,
                )
            )
    if isinstance(operation, HasNestedOps):
        nested_keys: set[tuple[str, int | None]] = set()
        for body in operation.nested_op_lists():
            for child in body:
                child_reads, child_writes = _quantum_wire_keys(
                    child,
                    resolver,
                    scalar_values=scalar_values,
                    used_names=used_names,
                )
                nested_keys |= child_reads | child_writes
        reads |= nested_keys
        writes |= nested_keys
    return reads, writes


def _operation_has_unresolved_quantum_index(
    operation: Operation,
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> bool:
    """Return whether a quantum element aliases its whole owner conservatively.

    Args:
        operation (Operation): Operation whose quantum values are inspected.
        resolver (ExprResolver): Resolver for element and view expressions.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        bool: Whether any scalar quantum element remains physically unresolved.
    """
    values = [*operation.all_input_values(), *operation.results]
    for value in values:
        if (
            isinstance(value, ArrayValue)
            and value.type.is_quantum()
            and value.slice_of is not None
        ):
            keys = _quantum_value_wire_keys(
                value,
                resolver,
                scalar_values=scalar_values,
                used_names=used_names,
            )
            if (_quantum_allocation_owner(value), None) in keys:
                return True
            continue
        if (
            not isinstance(value, Value)
            or not value.type.is_quantum()
            or value.parent_array is None
            or _quantum_element_wire_index(
                value,
                resolver,
                scalar_values=scalar_values,
                used_names=used_names,
            )
            is not None
        ):
            continue
        expressions = [
            _specialize_dependency_expression(
                resolver.resolve(index),
                scalar_values,
                used_names,
            )
            for index in value.element_indices
        ]
        current = value.parent_array
        while current.slice_of is not None:
            if current.slice_start is not None:
                expressions.append(
                    _specialize_dependency_expression(
                        resolver.resolve(current.slice_start),
                        scalar_values,
                        used_names,
                    )
                )
            if current.slice_step is not None:
                expressions.append(
                    _specialize_dependency_expression(
                        resolver.resolve(current.slice_step),
                        scalar_values,
                        used_names,
                    )
                )
            current = current.slice_of
        free_symbols = {
            symbol for expression in expressions for symbol in expression.free_symbols
        }
        if free_symbols and all(
            isinstance(symbol, sp.Dummy) for symbol in free_symbols
        ):
            # A ForOperation owns this symbolic index. Its loop-level
            # disjointness proof decides whether sequential depth is exact.
            continue
        return True
    return False


def _uses_measurement_derived_classical_input(
    operation: Operation,
    measurement_derived: set[str],
) -> bool:
    """Return whether an operation consumes measurement-derived classical data.

    Args:
        operation (Operation): Operation whose inputs should be inspected.
        measurement_derived (set[str]): UUIDs tainted by measurement results.

    Returns:
        bool: Whether a non-quantum input carries measurement provenance.
    """
    return any(
        isinstance(value, ValueBase)
        and not value.type.is_quantum()
        and value.uuid in measurement_derived
        for value in operation.all_input_values()
    )


def _operation_depth_is_dependency_schedulable(
    operation: Operation,
    measurement_derived: set[str],
) -> bool:
    """Return whether wire dependencies fully describe an operation's depth.

    Body-backed unitary calls and compile-time control flow can be scheduled by
    their quantum footprints just like primitive gates. Runtime feed-forward
    and while loops retain sequential composition because their classical
    dependencies are not represented by quantum logical IDs.

    Args:
        operation (Operation): Operation to classify.
        measurement_derived (set[str]): UUIDs tainted by measurement results.

    Returns:
        bool: Whether dependency scheduling is exact for this operation.
    """
    if _uses_measurement_derived_classical_input(operation, measurement_derived):
        return False
    if isinstance(
        operation,
        (
            GateOperation,
            QInitOperation,
            MeasureOperation,
            MeasureVectorOperation,
            MeasureQFixedOperation,
            ProjectOperation,
            ResetOperation,
            GlobalPhaseOperation,
        ),
    ):
        return True
    if isinstance(operation, InvokeOperation):
        return operation.effects.is_unitary
    if isinstance(
        operation,
        (ControlledUOperation, InverseBlockOperation, SelectOperation),
    ):
        return True
    if isinstance(operation, WhileOperation):
        return False
    if isinstance(operation, IfOperation):
        if operation.condition.uuid in measurement_derived:
            return False
        return all(
            _operation_depth_is_dependency_schedulable(child, measurement_derived)
            for body in operation.nested_op_lists()
            for child in body
        )
    if isinstance(operation, (ForOperation, ForItemsOperation)):
        return all(
            _operation_depth_is_dependency_schedulable(child, measurement_derived)
            for body in operation.nested_op_lists()
            for child in body
        )
    return False


def _disjoint_concrete_loop_depth(
    operation: ForOperation,
    resolver: ExprResolver,
    body_depth: DepthResources,
    *,
    start: ResourceExpr,
    stop: ResourceExpr,
    step: ResourceExpr,
    loop_symbol: sp.Symbol,
    clean_ancillas: ResourceExpr,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> DepthResources | None:
    """Parallelize concrete loop iterations with disjoint quantum footprints.

    This optimization is deliberately bounded: it proves pairwise-disjoint
    physical owner/index keys for at most 4096 concrete iterations. Whole
    registers, unresolved views, shared clean ancillas, or any overlapping
    element conservatively retain sequential loop depth.

    Args:
        operation (ForOperation): Loop whose independent body was summarized.
        resolver (ExprResolver): Resolver for the enclosing scope.
        body_depth (DepthResources): One symbolic body-iteration depth.
        start (ResourceExpr): Inclusive Python-range start.
        stop (ResourceExpr): Exclusive Python-range stop.
        step (ResourceExpr): Python-range step.
        loop_symbol (sp.Symbol): Internal body loop-variable symbol.
        clean_ancillas (ResourceExpr): Shared fallback ancilla demand.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        DepthResources | None: Parallel critical-path depth when disjointness
            is proven, otherwise ``None``.
    """
    bounds = tuple(
        _specialize_dependency_expression(
            bound,
            scalar_values,
            used_names,
        )
        for bound in (start, stop, step)
    )
    if not all(value.is_number and _is_concrete_integer(value) for value in bounds):
        return None
    concrete_start, concrete_stop, concrete_step = (int(value) for value in bounds)
    if concrete_step == 0 or clean_ancillas != _ZERO:
        return None
    iterations = range(concrete_start, concrete_stop, concrete_step)
    if len(iterations) > 4096:
        return None

    seen: set[tuple[str, int | None]] = set()
    fields = tuple(field.name for field in dataclasses.fields(DepthResources))
    peaks: dict[str, ResourceExpr] = {field: _ZERO for field in fields}
    for loop_value in iterations:
        value = sp.Integer(loop_value)
        if operation.loop_var_value is None:
            return None
        child = resolver.child_scope(
            inner_block=_LocalBlock(operation.operations),
            extra_context={operation.loop_var_value.uuid: value},
            extra_loop_vars={operation.loop_var: value},
        )
        footprint: set[tuple[str, int | None]] = set()
        for body_operation in operation.operations:
            reads, writes = _quantum_wire_keys(
                body_operation,
                child,
                scalar_values=scalar_values,
                used_names=used_names,
            )
            footprint |= reads | writes
        if not footprint and any(
            getattr(body_depth, field) != _ZERO for field in fields
        ):
            return None
        if _wire_footprints_overlap(seen, footprint):
            return None
        seen |= footprint
        for field in fields:
            expression = cast(sp.Expr, getattr(body_depth, field))
            iteration_depth = _safe_simplify(
                cast(sp.Expr, expression.subs(loop_symbol, value).doit())
            )
            peaks[field] = sp.Max(peaks[field], iteration_depth)
    return DepthResources(**peaks)


def _wire_footprints_overlap(
    left: set[tuple[str, int | None]],
    right: set[tuple[str, int | None]],
) -> bool:
    """Return whether two physical owner/index footprints may alias.

    Args:
        left (set[tuple[str, int | None]]): Existing physical wire keys.
        right (set[tuple[str, int | None]]): Candidate physical wire keys.

    Returns:
        bool: Whether any owner-wide or matching scalar key overlaps.
    """
    return any(
        left_owner == right_owner
        and (left_index is None or right_index is None or left_index == right_index)
        for left_owner, left_index in left
        for right_owner, right_index in right
    )


def _loop_body_has_symbolic_quantum_index(
    operation: ForOperation,
    resolver: ExprResolver,
    loop_symbol: sp.Symbol,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> bool:
    """Return whether a loop body addresses an array through its loop value.

    Args:
        operation (ForOperation): Loop whose body values are inspected.
        resolver (ExprResolver): Loop-body resolver.
        loop_symbol (sp.Symbol): Symbol representing the current iteration.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        bool: Whether any quantum element index depends on ``loop_symbol``.
    """
    return any(
        loop_symbol in index.free_symbols
        for body_operation in operation.operations
        for value in (*body_operation.all_input_values(), *body_operation.results)
        if isinstance(value, Value)
        and value.type.is_quantum()
        and value.parent_array is not None
        and (
            index := _quantum_element_index_expression(
                value,
                resolver,
                scalar_values=scalar_values,
                used_names=used_names,
            )
        )
        is not None
    )


def _symbolic_disjoint_loop_depth(
    operation: ForOperation,
    resolver: ExprResolver,
    body_depth: DepthResources,
    *,
    loop_symbol: sp.Symbol,
    iterations: ResourceExpr,
    clean_ancillas: ResourceExpr,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> DepthResources | None:
    """Prove parallel loop depth for one injective affine element footprint.

    For each root allocation, every depth-carrying body access must use the
    same affine index ``a * i + b`` with a provably nonzero ``a``. Distinct
    Python-range iterations then touch distinct physical slots, so iteration
    gate layers can run in parallel while additive gate counts still scale by
    the trip count.

    Args:
        operation (ForOperation): Loop whose footprint should be proven.
        resolver (ExprResolver): Resolver scoped to the loop body.
        body_depth (DepthResources): One iteration's dependency depth.
        loop_symbol (sp.Symbol): Symbol representing the loop value.
        iterations (ResourceExpr): Python-range iteration count.
        clean_ancillas (ResourceExpr): Shared decomposition ancilla demand.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        DepthResources | None: One-iteration depth guarded by a nonempty range,
            or ``None`` when injectivity cannot be proven.
    """
    if clean_ancillas != _ZERO or any(
        loop_symbol in cast(ResourceExpr, getattr(body_depth, field.name)).free_symbols
        for field in dataclasses.fields(DepthResources)
    ):
        return None
    ignored_operations = (
        QInitOperation,
        GlobalPhaseOperation,
        StoreArrayElementOperation,
        ReturnQuantumArrayElementOperation,
    )
    indices_by_owner: dict[str, list[ResourceExpr]] = {}
    for body_operation in operation.operations:
        if isinstance(body_operation, ignored_operations):
            continue
        if isinstance(body_operation, HasNestedOps):
            return None
        quantum_values = [
            value
            for value in (
                *body_operation.all_input_values(),
                *body_operation.results,
            )
            if isinstance(value, Value) and value.type.is_quantum()
        ]
        for value in quantum_values:
            if isinstance(value, ArrayValue) or value.parent_array is None:
                return None
            index = _quantum_element_index_expression(
                value,
                resolver,
                scalar_values=scalar_values,
                used_names=used_names,
            )
            if index is None:
                return None
            indices_by_owner.setdefault(
                _quantum_allocation_owner(value),
                [],
            ).append(index)
    if not indices_by_owner and any(
        getattr(body_depth, field.name) != _ZERO
        for field in dataclasses.fields(DepthResources)
    ):
        return None
    for indices in indices_by_owner.values():
        representative = indices[0]
        if any(_safe_simplify(index - representative) != _ZERO for index in indices):
            return None
        slope = _safe_simplify(cast(ResourceExpr, sp.diff(representative, loop_symbol)))
        if loop_symbol in slope.free_symbols or slope.is_zero is not False:
            return None
    return _conditional_depth(
        body_depth,
        DepthResources.zero(),
        sp.Gt(iterations, _ZERO),
    )


def _dependency_depth(
    scheduled: Sequence[tuple[Operation, ResourceEstimate]],
    resolver: ExprResolver,
    *,
    measurement_derived: set[str] | None = None,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> DepthResources:
    """Schedule operation summaries by wire dependencies and hybrid barriers.

    Args:
        scheduled (Sequence[tuple[Operation, ResourceEstimate]]): Operations in
            program order paired with their internally computed summaries.
        resolver (ExprResolver): Resolver for array element and view indices.
        measurement_derived (set[str] | None): Classical values transitively
            derived from runtime quantum observations. Operations that consume
            them, plus other unschedulable hybrid/control operations, form
            global ordering barriers. Defaults to ``None``.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        DepthResources: Critical-path depth for every tracked gate family.
    """
    fields = tuple(field.name for field in dataclasses.fields(DepthResources))
    availability: dict[
        str,
        dict[tuple[str, int | None], ResourceExpr],
    ] = {field: {} for field in fields}
    barrier_availability: dict[str, ResourceExpr] = {field: _ZERO for field in fields}
    peaks: dict[str, ResourceExpr] = {field: _ZERO for field in fields}
    for operation, estimate in scheduled:
        if not _estimate_has_nonzero_depth(estimate):
            continue
        if estimate._dependency_keys is None:
            reads, writes = _quantum_wire_keys(
                operation,
                resolver,
                scalar_values=scalar_values,
                used_names=used_names,
            )
        else:
            reads = set(estimate._dependency_keys)
            writes = set(estimate._dependency_keys)
        if estimate.width.clean_ancilla_qubits != _ZERO:
            # Width reports one reusable clean-ancilla pool (a maximum across
            # sequential operations). Treat that pool as a shared dependency
            # wire so depth never assumes two independent decompositions use
            # the same ancillas simultaneously.
            shared_pool = ("$resource_clean_ancilla_pool", None)
            reads.add(shared_pool)
            writes.add(shared_pool)
        if estimate.width.dirty_ancilla_qubits != _ZERO:
            shared_dirty_pool = ("$resource_dirty_ancilla_pool", None)
            reads.add(shared_dirty_pool)
            writes.add(shared_dirty_pool)
        touched = reads | writes
        operation_active = _resource_activity_condition(estimate.depth.depth)
        if operation_active is sp.false:
            continue
        schedulable = _operation_depth_is_dependency_schedulable(
            operation,
            measurement_derived or set(),
        )
        for field in fields:
            wire_depth = availability[field]
            dependencies: list[ResourceExpr] = []
            for owner, index in reads:
                if index is None:
                    dependencies.extend(
                        depth
                        for (
                            candidate_owner,
                            _candidate_index,
                        ), depth in wire_depth.items()
                        if candidate_owner == owner
                    )
                else:
                    dependencies.extend(
                        (
                            wire_depth.get((owner, None), _ZERO),
                            wire_depth.get((owner, index), _ZERO),
                        )
                    )
            if schedulable:
                dependencies.append(barrier_availability[field])
                start = _resource_max_many(dependencies)
            else:
                start = _resource_max(
                    peaks[field],
                    barrier_availability[field],
                )
            duration = cast(ResourceExpr, getattr(estimate.depth, field))
            finish = start + duration
            peaks[field] = _resource_max(peaks[field], finish)
            if not schedulable:
                previous_barrier = barrier_availability[field]
                barrier_availability[field] = (
                    finish
                    if operation_active is sp.true
                    else _piecewise(finish, previous_barrier, operation_active)
                )
            for key in touched:
                previous = wire_depth.get(key, _ZERO)
                wire_depth[key] = (
                    finish
                    if operation_active is sp.true
                    else _piecewise(finish, previous, operation_active)
                )
    return DepthResources(**peaks)


def _block_input_allocations(
    block: Block,
    resolver: ExprResolver,
) -> dict[str, ResourceExpr]:
    """Return caller-owned quantum widths keyed by logical wire identity.

    Traced qkernel blocks contain declaration-like ``QInitOperation`` nodes
    for their formal quantum inputs. Seeding those logical IDs lets liveness
    distinguish the declarations from true body allocations.

    Args:
        block (Block): Block whose formal quantum inputs should be seeded.
        resolver (ExprResolver): Resolver for symbolic array dimensions.

    Returns:
        dict[str, ResourceExpr]: Formal quantum widths by logical ID.
    """
    return {
        value.logical_id: _qubit_value_size(value, resolver)
        for value in block.input_values
        if isinstance(value, Value) and value.type.is_quantum()
    }


def _root_callable_resource_attrs(
    kernel: "QKernel[Any, Any] | Block | Sequence[Operation]",
) -> Mapping[str, Any]:
    """Return resource metadata preserved on a root qkernel-like object.

    A plain :class:`Block` intentionally has no callable identity or attrs.
    Serialized qkernels and stdlib descriptors preserve their definition attrs
    on ``_callable_attrs_override``, which lets root estimation enforce the
    same contract as nested invocation operations without importing frontend
    helpers into the estimator core.

    Args:
        kernel (QKernel[Any, Any] | Block | Sequence[Operation]): Estimator
            input before it is coerced to IR.

    Returns:
        Mapping[str, Any]: Preserved callable attrs, or an empty mapping.
    """
    attrs = getattr(kernel, "_callable_attrs_override", None)
    return attrs if isinstance(attrs, Mapping) else {}


def _root_callable_shape_inputs(
    attrs: Mapping[str, Any],
    block_or_ops: Block | Sequence[Operation],
    explicit_inputs: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, int]:
    """Infer one-dimensional root quantum-port widths from exact metadata.

    A callable resource contract describes total scalar widths. That value
    uniquely determines the shape of a one-dimensional quantum vector, but it
    cannot safely choose dimensions for a higher-rank array. Explicit port or
    dimension inputs always take precedence so a conflicting value reaches
    the normal contract validator and produces a useful error.

    Args:
        attrs (Mapping[str, Any]): Root callable definition attributes.
        block_or_ops (Block | Sequence[Operation]): Coerced estimator input.
        explicit_inputs (Mapping[str, Any]): User-supplied specialization
            inputs before build/estimation partitioning.
        source (str): Callable name used in malformed-contract diagnostics.

    Returns:
        dict[str, int]: Inferred one-dimensional quantum-port widths keyed by
            their public argument names.

    Raises:
        ValueError: If present resource metadata is malformed.
    """
    if not isinstance(block_or_ops, Block):
        return {}
    quantum_ports = [
        (name, value)
        for name, value in zip(
            block_or_ops.label_args,
            block_or_ops.input_values,
        )
        if isinstance(value, Value) and value.type.is_quantum()
    ]
    shape_aliases = input_shape_dimension_aliases(block_or_ops)
    inferred: dict[str, int] = {}
    for entry in quantum_operand_widths(attrs, source=source):
        if entry.index >= len(quantum_ports):
            # The constraint builder reports the complete call-shape
            # diagnostic before inference runs.
            continue
        name, value = quantum_ports[entry.index]
        if not isinstance(value, ArrayValue) or len(value.shape) != 1:
            continue
        if value.shape[0].is_constant():
            continue
        dimension_name = shape_aliases.get(value.shape[0].uuid)
        if name in explicit_inputs or (
            dimension_name is not None and dimension_name in explicit_inputs
        ):
            continue
        inferred[name] = entry.width
    return inferred


def _captured_quantum_allocations(
    operations: Sequence[Operation],
    resolver: ExprResolver,
    allocation_owners_by_uuid: Mapping[str, str] | None = None,
) -> dict[str, ResourceExpr]:
    """Return outer quantum allocations captured by a nested operation list.

    Branch liveness must start with captured wires live so measuring or
    replacing them can release capacity before a branch-local allocation. A
    value is captured when it is read by the nested list but is not produced by
    any operation in that same list.

    Args:
        operations (Sequence[Operation]): Nested operations to inspect.
        resolver (ExprResolver): Resolver for symbolic array dimensions.
        allocation_owners_by_uuid (Mapping[str, str] | None): Optional
            enclosing QInit UUID to logical-owner map for synthetic tuple
            carriers. Defaults to ``None``.

    Returns:
        dict[str, ResourceExpr]: Captured root allocation widths by owner.
    """
    produced = {
        result.uuid
        for operation in operations
        for result in operation.results
        if isinstance(result, Value)
    }
    captured: dict[str, ResourceExpr] = {}
    for operation in operations:
        for value in operation.all_input_values():
            if (
                not isinstance(value, Value)
                or not value.type.is_quantum()
                or value.uuid in produced
            ):
                continue
            runtime_sizes = _runtime_carrier_owner_sizes(
                value,
                allocation_owners_by_uuid or {},
            )
            if runtime_sizes is not None:
                for owner, size in runtime_sizes.items():
                    captured[owner] = captured.get(owner, _ZERO) + size
                continue
            owner = _quantum_allocation_owner(value)
            captured[owner] = _quantum_owner_capacity(value, resolver)
    return captured


def _branch_owner_sizes(
    true_sizes: Mapping[str, ResourceExpr],
    false_sizes: Mapping[str, ResourceExpr],
    *,
    condition: sp.Basic,
    runtime_condition: bool,
) -> dict[str, ResourceExpr]:
    """Combine captured owner widths across two conditional branches.

    Args:
        true_sizes (Mapping[str, ResourceExpr]): True-branch captured widths.
        false_sizes (Mapping[str, ResourceExpr]): False-branch captured widths.
        condition (sp.Basic): Compile-time branch predicate.
        runtime_condition (bool): Whether the branch is selected at runtime.

    Returns:
        dict[str, ResourceExpr]: Captured width for each possible owner.
    """
    return {
        owner: (
            sp.Max(true_sizes.get(owner, _ZERO), false_sizes.get(owner, _ZERO))
            if runtime_condition
            else _piecewise(
                true_sizes.get(owner, _ZERO),
                false_sizes.get(owner, _ZERO),
                condition,
            )
        )
        for owner in true_sizes.keys() | false_sizes.keys()
    }


def _with_operation_output_summary(
    estimate: ResourceEstimate,
    operation: ForOperation | WhileOperation | ForItemsOperation,
    resolver: ExprResolver,
    *,
    active_when: sp.Basic,
    allocation_owners_by_uuid: Mapping[str, str] | None = None,
) -> ResourceEstimate:
    """Attach authoritative live quantum results to a nested estimate.

    Args:
        estimate (ResourceEstimate): Nested operation estimate.
        operation (ForOperation | WhileOperation | ForItemsOperation):
            Enclosing control-flow operation.
        resolver (ExprResolver): Resolver after publishing carried results.
        active_when (sp.Basic): Condition under which the body executes at
            least once.
        allocation_owners_by_uuid (Mapping[str, str] | None): Optional
            enclosing QInit UUID to logical-owner map for synthetic tuple
            carriers. Defaults to ``None``.

    Returns:
        ResourceEstimate: Estimate whose output summary may be empty when every
        captured quantum input was consumed.
    """
    captured = _captured_quantum_allocations(
        operation.operations,
        resolver,
        allocation_owners_by_uuid,
    )
    body_consumed = _definitely_consumed_captured_allocations(
        operation.operations,
        captured,
        resolver,
        allocation_owners_by_uuid,
    )
    condition = _boolean_condition(active_when)
    consumed = {
        owner: _piecewise(size, _ZERO, condition)
        for owner, size in body_consumed.items()
    }
    return dataclasses.replace(
        estimate,
        _output_sizes=_quantum_result_owner_sizes(
            operation.results,
            resolver,
            allocation_owners_by_uuid,
        ),
        _input_sizes=consumed,
        _has_output_summary=True,
    )


def _definitely_consumed_captured_allocations(
    operations: Sequence[Operation],
    captured: Mapping[str, ResourceExpr],
    resolver: ExprResolver,
    allocation_owners_by_uuid: Mapping[str, str] | None = None,
) -> dict[str, ResourceExpr]:
    """Find captured owners whose final full-width action consumes them.

    Partial element operations are deliberately ignored: without tracking the
    exact final element set, retaining the owner is safer than releasing live
    siblings. Full-width gates keep an owner live, while a later full-width
    measurement or replacement consumes it.

    Args:
        operations (Sequence[Operation]): Nested body operations in order.
        captured (Mapping[str, ResourceExpr]): Captured owner capacities.
        resolver (ExprResolver): Resolver for symbolic operand widths.
        allocation_owners_by_uuid (Mapping[str, str] | None): Optional
            enclosing QInit UUID to logical-owner map for synthetic tuple
            carriers. Defaults to ``None``.

    Returns:
        dict[str, ResourceExpr]: Owners proven fully consumed by the body.
    """
    consumed = {owner: False for owner in captured}
    for operation in operations:
        if isinstance(operation, HasNestedOps):
            continue
        input_sizes = _quantum_result_owner_sizes(
            [
                value
                for value in operation.all_input_values()
                if isinstance(value, Value) and value.type.is_quantum()
            ],
            resolver,
            allocation_owners_by_uuid,
        )
        result_sizes = _quantum_result_owner_sizes(
            [result for result in operation.results if result.type.is_quantum()],
            resolver,
            allocation_owners_by_uuid,
        )
        for owner, capacity in captured.items():
            touched = input_sizes.get(owner, _ZERO)
            if _safe_simplify(touched - capacity) != _ZERO:
                continue
            returned = result_sizes.get(owner, _ZERO)
            consumed[owner] = _safe_simplify(returned - capacity) != _ZERO
    return {
        owner: captured[owner] for owner, is_consumed in consumed.items() if is_consumed
    }


def _without_input_allocation_sites(
    sites: Mapping[str, ResourceExpr],
    operations: Sequence[Operation],
    initial_allocations: Mapping[str, ResourceExpr],
) -> dict[str, ResourceExpr]:
    """Remove formal-input QInit declarations from allocation-site metadata.

    Args:
        sites (Mapping[str, ResourceExpr]): QInit sites collected while
            evaluating the operations.
        operations (Sequence[Operation]): Operations in the evaluated scope.
        initial_allocations (Mapping[str, ResourceExpr]): Caller-owned formal
            quantum wires keyed by logical ID.

    Returns:
        dict[str, ResourceExpr]: Allocation sites containing only body-owned
        qubits.
    """
    formal_site_ids = {
        operation.results[0].uuid
        for operation in operations
        if isinstance(operation, QInitOperation)
        and operation.results
        and operation.results[0].logical_id in initial_allocations
    }
    return {site: size for site, size in sites.items() if site not in formal_site_ids}


def _quantum_root_array(value: Value) -> ArrayValue | None:
    """Return the root array allocation aliased by a quantum value.

    Args:
        value (Value): Quantum scalar, array, or array view.

    Returns:
        ArrayValue | None: Root array, or ``None`` for an independent scalar.
    """
    if isinstance(value, ArrayValue):
        array = value
    else:
        array = value.parent_array
    if array is None:
        return None
    while array.slice_of is not None:
        array = array.slice_of
    return array


def _cast_carrier_wire_keys(value: Value) -> set[WireKey] | None:
    """Return physical dependency keys recorded by quantum cast metadata.

    Concrete vector-to-register casts retain root-space logical identifiers
    such as ``"<root>_3"`` for every carrier. Symbolic casts cannot enumerate
    carriers, so their source logical ID remains an owner-wide conservative
    alias.

    Args:
        value (Value): Candidate cast result.

    Returns:
        set[WireKey] | None: Carrier keys, an owner-wide fallback, or ``None``
            when ``value`` is not a cast result.
    """
    if not value.is_cast_result():
        return None
    logical_ids = value.get_cast_qubit_logical_ids() or ()
    if logical_ids:
        keys: set[WireKey] = set()
        for logical_id in logical_ids:
            indexed = split_indexed_identifier(logical_id)
            if indexed is None:
                keys.add((logical_id, None))
                continue
            owner, index = indexed
            keys.add((owner, int(index)))
        return keys
    source = value.get_cast_source_logical_id()
    return {(source, None)} if source is not None else {(value.logical_id, None)}


def _quantum_allocation_owner(value: Value) -> str:
    """Return the root logical allocation identity for a quantum value.

    Array elements and sliced arrays have their own SSA/logical identities but
    alias storage owned by the root array allocation. Treating those views as
    fresh returned qubits inflates liveness by one per element operation.

    Args:
        value (Value): Quantum scalar, array, or array view.

    Returns:
        str: Root array logical ID for a view/element, otherwise the value's
        own logical ID.
    """
    carrier_keys = _cast_carrier_wire_keys(value)
    if carrier_keys:
        owners = {owner for owner, _index in carrier_keys}
        if len(owners) == 1:
            return next(iter(owners))
    array = _quantum_root_array(value)
    if array is None:
        return value.logical_id
    return array.logical_id


def _runtime_carrier_owner_sizes(
    value: Value,
    allocation_owners_by_uuid: Mapping[str, str],
) -> dict[str, ResourceExpr] | None:
    """Resolve tuple-style synthetic carriers to physical allocation owners.

    Args:
        value (Value): Candidate synthetic quantum array carrier.
        allocation_owners_by_uuid (Mapping[str, str]): QInit-result UUIDs
            mapped to root logical allocation IDs.

    Returns:
        dict[str, ResourceExpr] | None: Per-owner carrier counts, or ``None``
            when ``value`` has no element-identity runtime metadata.
    """
    runtime = value.metadata.array_runtime
    if runtime is None or not runtime.element_uuids or not runtime.element_logical_ids:
        return None
    parent_addresses = value.get_element_parent_addresses()
    owners: dict[str, ResourceExpr] = {}
    for index, logical_id in enumerate(runtime.element_logical_ids):
        address = parent_addresses[index] if index < len(parent_addresses) else None
        owner = (
            allocation_owners_by_uuid.get(address[0], logical_id)
            if address is not None
            else logical_id
        )
        owners[owner] = owners.get(owner, _ZERO) + _ONE
    return owners


def _quantum_result_owner_sizes(
    results: Sequence[Value],
    resolver: ExprResolver,
    allocation_owners_by_uuid: Mapping[str, str] | None = None,
) -> dict[str, ResourceExpr]:
    """Aggregate live result width by root allocation identity.

    A nested qkernel can return several scalar elements from one freshly
    allocated array. Those elements share one owner but collectively keep
    more than one qubit live. Their returned widths are summed and capped by
    the root allocation size so duplicate or overlapping views remain
    conservative without exceeding the underlying allocation.

    Args:
        results (Sequence[Value]): Quantum and classical operation results.
        resolver (ExprResolver): Resolver for symbolic result dimensions.
        allocation_owners_by_uuid (Mapping[str, str] | None): Optional
            QInit-result UUID to logical-owner map for synthetic tuple
            carriers. Defaults to ``None``.

    Returns:
        dict[str, ResourceExpr]: Live returned width by allocation owner.
    """
    returned: dict[str, ResourceExpr] = {}
    capacities: dict[str, ResourceExpr] = {}
    for result in results:
        if not result.type.is_quantum():
            continue
        runtime_sizes = _runtime_carrier_owner_sizes(
            result,
            allocation_owners_by_uuid or {},
        )
        if runtime_sizes is not None:
            for owner, size in runtime_sizes.items():
                returned[owner] = returned.get(owner, _ZERO) + size
                capacities[owner] = returned[owner]
            continue
        owner = _quantum_allocation_owner(result)
        returned[owner] = returned.get(owner, _ZERO) + _qubit_value_size(
            result,
            resolver,
        )
        capacities[owner] = _quantum_owner_capacity(result, resolver)
    return {owner: sp.Min(size, capacities[owner]) for owner, size in returned.items()}


def _destructive_input_owner_sizes(
    operation: Operation,
    resolver: ExprResolver,
    allocation_owners_by_uuid: Mapping[str, str],
) -> dict[str, ResourceExpr]:
    """Resolve physical owners consumed by measurement-like operations.

    Tuple-form expectation values use a synthetic ``ArrayValue`` whose runtime
    metadata retains each original element and, for borrowed array elements,
    its root-allocation UUID. Liveness needs that metadata because the
    synthetic carrier itself is not an allocation owner.

    Args:
        operation (Operation): Destructive quantum observation.
        resolver (ExprResolver): Resolver for ordinary quantum operand widths.
        allocation_owners_by_uuid (Mapping[str, str]): QInit-result UUIDs
            mapped to their root logical allocation IDs.

    Returns:
        dict[str, ResourceExpr]: Consumed qubit count by live allocation owner.
    """
    quantum_inputs = [
        value
        for value in operation.all_input_values()
        if isinstance(value, Value) and value.type.is_quantum()
    ]
    return _quantum_result_owner_sizes(
        quantum_inputs,
        resolver,
        allocation_owners_by_uuid,
    )


def _quantum_owner_capacity(
    value: Value,
    resolver: ExprResolver,
) -> ResourceExpr:
    """Return the complete allocation width for a quantum value's owner.

    Args:
        value (Value): Quantum scalar, array, or array view.
        resolver (ExprResolver): Resolver for symbolic dimensions.

    Returns:
        ResourceExpr: Root array width or one independent scalar width.
    """
    root = _quantum_root_array(value)
    return (
        _qubit_value_size(root, resolver)
        if root is not None
        else _qubit_value_size(value, resolver)
    )


def _invoke_quantum_output_sizes(
    operation: InvokeOperation,
    body: Block,
    child_resolver: ExprResolver,
    caller_resolver: ExprResolver,
    *,
    body_implements_transform: bool,
) -> tuple[dict[str, ResourceExpr], bool]:
    """Map body-derived quantum output widths onto caller result owners.

    A callee branch may return arrays whose branches have different symbolic
    widths even though the caller-side IR result retains one representative
    static shape. The callee resolver contains the merged shape binding, so
    invocation liveness must carry that size across the call boundary.

    Args:
        operation (InvokeOperation): Caller-side invocation.
        body (Block): Selected implementation body.
        child_resolver (ExprResolver): Resolver after evaluating the body.
        caller_resolver (ExprResolver): Resolver for caller-side controls.
        body_implements_transform (bool): Whether the selected body explicitly
            includes transform-specific control inputs and outputs.

    Returns:
        tuple[dict[str, ResourceExpr], bool]: Output width by caller allocation
        owner and whether the positional mapping was complete.
    """
    sources: list[tuple[ValueBase, ExprResolver]] = []
    if (
        operation.transform is CallTransform.CONTROLLED
        and not body_implements_transform
    ):
        control_count = operation.num_control_qubits
        sources.extend(
            (operand, caller_resolver) for operand in operation.operands[:control_count]
        )
    sources.extend((output, child_resolver) for output in body.output_values)
    if len(sources) != len(operation.results):
        return {}, False

    output_sizes: dict[str, ResourceExpr] = {}
    capacities: dict[str, ResourceExpr] = {}
    for result, (source, source_resolver) in zip(
        operation.results,
        sources,
        strict=True,
    ):
        if (
            not isinstance(result, Value)
            or not isinstance(source, Value)
            or not result.type.is_quantum()
        ):
            continue
        owner = _quantum_allocation_owner(result)
        output_sizes[owner] = output_sizes.get(owner, _ZERO) + _qubit_value_size(
            source,
            source_resolver,
        )
        capacities[owner] = _quantum_owner_capacity(result, caller_resolver)
    return {
        owner: sp.Min(size, capacities[owner]) for owner, size in output_sizes.items()
    }, True


def _liveness_width(
    scheduled: Sequence[tuple[Operation, ResourceEstimate]],
    initial_allocations: Mapping[str, ResourceExpr],
    resolver: ExprResolver,
    *,
    allocation_owners_by_uuid: Mapping[str, str] | None = None,
) -> WidthResources:
    """Compute peak width from affine allocation and consumption lifetimes.

    Args:
        scheduled (Sequence[tuple[Operation, ResourceEstimate]]): Operations and
            their nested width summaries in program order.
        initial_allocations (Mapping[str, ResourceExpr]): Live input wire sizes
            keyed by logical ID.
        resolver (ExprResolver): Resolver for symbolic result dimensions.
        allocation_owners_by_uuid (Mapping[str, str] | None): Optional known
            QInit UUID to logical-owner map inherited from enclosing scopes.
            Defaults to ``None``.

    Returns:
        WidthResources: Total body allocations and liveness-aware peak width.
    """
    live = dict(initial_allocations)
    baseline = sum(live.values(), _ZERO)
    current = baseline
    peak = current
    allocated = _ZERO
    clean = _ZERO
    dirty = _ZERO
    resolved_allocation_owners = dict(allocation_owners_by_uuid or {})
    for operation, estimate in scheduled:
        clean = _resource_max(clean, estimate.width.clean_ancilla_qubits)
        dirty = _resource_max(dirty, estimate.width.dirty_ancilla_qubits)
        if isinstance(operation, QInitOperation):
            result = operation.results[0]
            resolved_allocation_owners[result.uuid] = result.logical_id
            if result.logical_id in live:
                # Formal quantum inputs are represented by QInit declarations
                # in traced blocks but remain caller-owned wires.
                continue
            amount = estimate.width.allocated_qubits
            allocated += amount
            live[result.logical_id] = amount
            current += amount
            peak = sp.Max(peak, current)
        else:
            allocated += estimate.width.allocated_qubits
            peak = sp.Max(peak, current + estimate.width.peak_qubits)
        if isinstance(
            operation,
            (
                MeasureOperation,
                MeasureVectorOperation,
                MeasureQFixedOperation,
                ExpvalOp,
            ),
        ):
            for owner, measured in _destructive_input_owner_sizes(
                operation,
                resolver,
                resolved_allocation_owners,
            ).items():
                previous = live.get(owner, _ZERO)
                remaining = sp.Max(_ZERO, previous - measured)
                current += remaining - previous
                if remaining == _ZERO:
                    live.pop(owner, None)
                else:
                    live[owner] = remaining
        if not isinstance(operation, QInitOperation):
            input_values = [
                value
                for value in operation.all_input_values()
                if isinstance(value, Value) and value.type.is_quantum()
            ]
            result_values = [
                result
                for result in operation.results
                if isinstance(result, Value) and result.type.is_quantum()
            ]
            input_sizes = _quantum_result_owner_sizes(input_values, resolver)
            if estimate._input_sizes:
                input_sizes = estimate._input_sizes
            result_sizes = (
                estimate._output_sizes
                if estimate._has_output_summary
                else _quantum_result_owner_sizes(result_values, resolver)
            )
            if isinstance(operation, InvokeOperation):
                capacities = {
                    _quantum_allocation_owner(value): _quantum_owner_capacity(
                        value,
                        resolver,
                    )
                    for value in (*input_values, *result_values)
                }
                for owner in input_sizes.keys() | result_sizes.keys():
                    previous = live.get(owner, _ZERO)
                    consumed = input_sizes.get(owner, _ZERO)
                    returned = result_sizes.get(owner, _ZERO)
                    if estimate._has_output_summary and consumed == returned:
                        # Affine calls that return the same aggregate owner
                        # width preserve liveness exactly. Avoid expanding the
                        # equivalent ``Max(0, previous-consumed)+returned``;
                        # SymPy cannot generally cancel it through Min/Piecewise
                        # array-width expressions.
                        updated = previous
                    else:
                        remaining = sp.Max(
                            _ZERO,
                            previous - consumed,
                        )
                        updated = (
                            remaining + returned
                            if estimate._has_output_summary
                            else sp.Min(capacities[owner], remaining + returned)
                        )
                    current += updated - previous
                    if updated == _ZERO:
                        live.pop(owner, None)
                    else:
                        live[owner] = updated
            elif estimate._has_output_summary:
                for owner in input_sizes.keys() | result_sizes.keys():
                    previous = live.get(owner, _ZERO)
                    consumed = input_sizes.get(owner, _ZERO)
                    returned = result_sizes.get(owner, _ZERO)
                    if consumed == returned:
                        continue
                    remaining = sp.Max(
                        _ZERO,
                        previous - consumed,
                    )
                    updated = remaining + returned
                    current += updated - previous
                    if updated == _ZERO:
                        live.pop(owner, None)
                    else:
                        live[owner] = updated
            else:
                for owner, amount in result_sizes.items():
                    if owner in live or owner in input_sizes:
                        continue
                    live[owner] = amount
                    current += amount
            peak = sp.Max(peak, current)
    relative_peak = sp.Max(_ZERO, peak - baseline)
    static_width = allocated + clean + dirty
    if _is_structurally_nonnegative(relative_peak - static_width):
        # Static circuit width is a hard upper bound. If liveness arithmetic
        # already reached that bound, retain the compact exact expression
        # instead of exposing an equivalent nest of Max/Piecewise nodes.
        relative_peak = static_width
    else:
        relative_peak = sp.Min(relative_peak, static_width)
    return WidthResources(
        allocated_qubits=allocated,
        clean_ancilla_qubits=clean,
        dirty_ancilla_qubits=dirty,
        peak_qubits=relative_peak,
    )


def _merge_allocation_sites(
    left: Mapping[str, ResourceExpr],
    right: Mapping[str, ResourceExpr],
) -> dict[str, ResourceExpr]:
    """Merge static QInit sites without counting one identity twice.

    A QInit operation in a loop body is emitted once and reset before each
    replayed iteration. The same result UUID therefore denotes one allocation
    site even when concrete interpretation visits it repeatedly. If a symbolic
    site size differs between visits, the largest size is retained safely.

    Args:
        left (Mapping[str, ResourceExpr]): Sites collected so far.
        right (Mapping[str, ResourceExpr]): Sites from another estimate.

    Returns:
        dict[str, ResourceExpr]: Union keyed by QInit result UUID.
    """
    merged = dict(left)
    for site, size in right.items():
        previous = merged.get(site)
        merged[site] = size if previous is None else sp.Max(previous, size)
    return merged


def _namespace_allocation_sites(
    estimate: ResourceEstimate,
    operation: Operation,
) -> ResourceEstimate:
    """Qualify nested QInit identities by their static call site.

    Callable implementation Blocks are shared recipes: two distinct Invoke,
    ControlledU, or Inverse operations may traverse the SAME inner QInit UUID.
    The emitter clones/allocates those call sites independently, while repeated
    evaluation of ONE operation in a concrete loop must still reuse its site.
    Prefixing with the stable in-memory operation identity provides exactly
    that scope for one estimation run; the map is internal and excluded from
    equality and serialization, so process-local identities never leak.

    Args:
        estimate (ResourceEstimate): Nested body estimate to qualify.
        operation (Operation): Static call-site operation owning the body.

    Returns:
        ResourceEstimate: Estimate with call-site-qualified allocation keys.
    """
    if not estimate._allocation_sites:
        return estimate
    namespace = f"{type(operation).__name__}:{id(operation)}"
    return dataclasses.replace(
        estimate,
        _allocation_sites={
            f"{namespace}/{site}": size
            for site, size in estimate._allocation_sites.items()
        },
    )


def _activate_allocation_sites(
    sites: Mapping[str, ResourceExpr],
    condition: sp.Basic,
) -> dict[str, ResourceExpr]:
    """Guard allocation-site sizes by whether a repeated body executes.

    Args:
        sites (Mapping[str, ResourceExpr]): Static QInit sites in the body.
        condition (sp.Basic): Boolean expression that is true when the body
            executes at least once.

    Returns:
        dict[str, ResourceExpr]: Site sizes that become zero on a zero-trip
        path.
    """
    return {
        site: _resource_expr(sp.Piecewise((size, condition), (_ZERO, True)))
        for site, size in sites.items()
    }


def _allocation_site_total(
    sites: Mapping[str, ResourceExpr],
) -> ResourceExpr:
    """Sum the sizes of distinct static QInit identities.

    Args:
        sites (Mapping[str, ResourceExpr]): QInit result UUIDs and sizes.

    Returns:
        ResourceExpr: Total qubits allocated by the distinct sites.
    """
    return sum(sites.values(), _ZERO)


def _anonymous_allocation_width(
    width: WidthResources,
    sites: Mapping[str, ResourceExpr],
) -> ResourceExpr:
    """Return allocated width not attributable to explicit QInit sites.

    Opaque cost models may report allocated qubits without exposing an IR
    QInit identity. Their contribution stays reusable across iterations and
    is tracked separately from the identity-aware site union.

    Args:
        width (WidthResources): Width summary containing total allocations.
        sites (Mapping[str, ResourceExpr]): Explicit QInit sites represented in
            that summary.

    Returns:
        ResourceExpr: Nonnegative anonymous allocation contribution.
    """
    return sp.Max(
        _ZERO,
        sp.simplify(width.allocated_qubits - _allocation_site_total(sites)),
    )


def _width_with_identity_aware_allocations(
    width: WidthResources,
    sites: Mapping[str, ResourceExpr],
    *,
    anonymous_allocated: ResourceExpr | None = None,
) -> WidthResources:
    """Replace only allocated width with a static-site-aware total.

    Peak width and ancilla fields retain their liveness/reuse semantics. Only
    ``allocated_qubits`` distinguishes repeated visits to one QInit identity
    from visits to different identities.

    Args:
        width (WidthResources): Reusable width summary to preserve otherwise.
        sites (Mapping[str, ResourceExpr]): Distinct explicit QInit sites.
        anonymous_allocated (ResourceExpr | None): Reusable allocation amount
            without explicit site identities. Defaults to the residual derived
            from ``width``.

    Returns:
        WidthResources: Width with identity-aware ``allocated_qubits``.
    """
    residual = (
        _anonymous_allocation_width(width, sites)
        if anonymous_allocated is None
        else anonymous_allocated
    )
    return dataclasses.replace(
        width,
        allocated_qubits=sp.simplify(_allocation_site_total(sites) + residual),
    )


def _branch_width_with_static_allocations(
    branch_width: WidthResources,
    left_width: WidthResources,
    left_sites: Mapping[str, ResourceExpr],
    right_width: WidthResources,
    right_sites: Mapping[str, ResourceExpr],
    merged_sites: Mapping[str, ResourceExpr],
) -> WidthResources:
    """Combine branch liveness with the union of static allocation sites.

    Runtime branches are mutually exclusive, so peak live width remains a
    maximum or Piecewise expression. Static circuit allocation is different:
    emitters reserve distinct QInit sites from both branches. Anonymous opaque
    allocations have no identity that proves reuse, so their branch residuals
    are conservatively added as well.

    Args:
        branch_width (WidthResources): Liveness-aware maximum or conditional
            branch width.
        left_width (WidthResources): True/left branch width.
        left_sites (Mapping[str, ResourceExpr]): Explicit allocation sites in
            the true/left branch.
        right_width (WidthResources): False/right branch width.
        right_sites (Mapping[str, ResourceExpr]): Explicit allocation sites in
            the false/right branch.
        merged_sites (Mapping[str, ResourceExpr]): Union of explicit sites from
            both branches.

    Returns:
        WidthResources: Branch width with conservative static allocations.
    """
    anonymous_allocated = _anonymous_allocation_width(
        left_width,
        left_sites,
    ) + _anonymous_allocation_width(
        right_width,
        right_sites,
    )
    return _width_with_identity_aware_allocations(
        branch_width,
        merged_sites,
        anonymous_allocated=anonymous_allocated,
    )


def _maximum_width_over_range(
    width: WidthResources,
    sites: Mapping[str, ResourceExpr],
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> tuple[WidthResources, dict[str, ResourceExpr], bool]:
    """Maximize reusable width across a symbolic loop range.

    Additive resources such as gates and depth are summed across loop
    iterations, but qubits can be reused. Explicit allocation sites are
    maximized individually because a static emitted circuit reserves every
    distinct site, even when different sizes occur on different iterations.

    Args:
        width (WidthResources): Width used by one symbolic loop iteration.
        sites (Mapping[str, ResourceExpr]): Explicit QInit sites in the body.
        loop_symbol (sp.Symbol): Loop variable symbol.
        start (ResourceExpr): First loop value.
        step (ResourceExpr): Loop step.
        iterations (ResourceExpr): Number of executed iterations.

    Returns:
        tuple[WidthResources, dict[str, ResourceExpr], bool]: Maximum reusable
        width, maximized allocation sites, and whether every maximum is exact.
    """
    maximized_sites: dict[str, ResourceExpr] = {}
    exact = True
    for site, size in sites.items():
        maximum, maximum_is_exact = _maximum_expr_over_range(
            size,
            loop_symbol,
            start,
            step,
            iterations,
        )
        maximized_sites[site] = maximum
        exact = exact and maximum_is_exact

    anonymous = _anonymous_allocation_width(width, sites)
    maximum_anonymous, anonymous_is_exact = _maximum_expr_over_range(
        anonymous,
        loop_symbol,
        start,
        step,
        iterations,
    )
    exact = exact and anonymous_is_exact

    maxima: dict[str, ResourceExpr] = {}
    for field in dataclasses.fields(WidthResources):
        if field.name == "allocated_qubits":
            continue
        maximum, maximum_is_exact = _maximum_expr_over_range(
            getattr(width, field.name),
            loop_symbol,
            start,
            step,
            iterations,
        )
        maxima[field.name] = maximum
        exact = exact and maximum_is_exact
    maximized_width = WidthResources(
        allocated_qubits=_ZERO,
        **maxima,
    )
    return (
        _width_with_identity_aware_allocations(
            maximized_width,
            maximized_sites,
            anonymous_allocated=maximum_anonymous,
        ),
        maximized_sites,
        exact,
    )


def _qubit_value_size(value: Value, resolver: ExprResolver) -> ResourceExpr:
    """Return the logical width represented by one quantum value.

    Args:
        value (Value): Scalar or array quantum value.
        resolver (ExprResolver): Resolver for symbolic array dimensions.

    Returns:
        ResourceExpr: Number of represented qubits.
    """
    if isinstance(value, ArrayValue) and isinstance(value.type, QubitType):
        runtime = value.metadata.array_runtime
        if runtime is not None and runtime.element_uuids and not value.shape:
            return sp.Integer(len(runtime.element_uuids))
        count: ResourceExpr = _ONE
        for dim in value.shape:
            count *= resolver.resolve(dim)
        return count
    if isinstance(value.type, QubitType):
        return _ONE
    if isinstance(value.type, QUIntType):
        return _quantum_register_component_size(value.type.width, resolver)
    if isinstance(value.type, QFixedType):
        return _quantum_register_component_size(
            value.type.integer_bits,
            resolver,
        ) + _quantum_register_component_size(
            value.type.fractional_bits,
            resolver,
        )
    return _ZERO


def _quantum_register_component_size(
    component: int | Value,
    resolver: ExprResolver,
) -> ResourceExpr:
    """Resolve one integer component of a quantum-register width.

    Args:
        component (int | Value): Concrete bit count or symbolic UInt value.
        resolver (ExprResolver): Resolver for symbolic register widths.

    Returns:
        ResourceExpr: Resolved nonnegative width component expression.
    """
    if isinstance(component, Value):
        return resolver.resolve(component)
    return sp.Integer(component)


def _count_qinit(operation: QInitOperation, resolver: ExprResolver) -> ResourceExpr:
    """Count qubits allocated by a qinit operation.

    Args:
        operation (QInitOperation): Qubit initialization operation.
        resolver (ExprResolver): Resolver for symbolic array shapes.

    Returns:
        ResourceExpr: Allocated-qubit count.
    """
    result = operation.results[0]
    if isinstance(result, ArrayValue) and isinstance(result.type, QubitType):
        count: ResourceExpr = _ONE
        for dim in result.shape:
            count *= resolver.resolve(dim)
        return count
    if isinstance(result.type, QubitType):
        return _ONE
    return _ZERO
