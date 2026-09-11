"""Map quantum liveness across nested operation boundaries."""

from __future__ import annotations

import dataclasses
from collections import ChainMap
from collections.abc import Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING, cast

import sympy as sp

from qamomile.circuit.estimator._constants import _ONE, _ZERO
from qamomile.circuit.estimator._dependency_footprints import _quantum_value_wire_keys
from qamomile.circuit.estimator._dependency_indices import _normalize_wire_index
from qamomile.circuit.estimator._quantum_values import (
    _quantum_allocation_owner,
    _quantum_root_array,
    _qubit_value_size,
)
from qamomile.circuit.estimator._resolver import ExprResolver
from qamomile.circuit.estimator._resource_base import (
    ResourceExpr,
    _is_concrete_integer,
)
from qamomile.circuit.estimator._resource_expressions import (
    _boolean_condition,
    _piecewise,
    _resource_max,
    _safe_simplify,
)
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    WhileOperation,
)
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.operation.gate import (
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureQIntOperation,
    MeasureVectorOperation,
)
from qamomile.circuit.ir.operation.operation import Operation, QInitOperation
from qamomile.circuit.ir.value import Value, ValueBase
from qamomile.circuit.transpiler.block_parameter_binding import pair_block_operands

if TYPE_CHECKING:
    from qamomile.circuit.estimator._estimate import ResourceEstimate


_DESTRUCTIVE_LOOP_OBSERVATION_TYPES = (
    MeasureOperation,
    MeasureVectorOperation,
    MeasureQFixedOperation,
    MeasureQIntOperation,
    ExpvalOp,
)


def _loop_body_has_destructive_observation(
    operations: Sequence[Operation],
) -> bool:
    """Return whether a loop body contains a tracked destructive observation.

    Operation-owned nested control-flow regions are inspected recursively.
    Callable definitions remain opaque because loop captured-consumption
    analysis does not expand ``InvokeOperation`` bodies.

    Args:
        operations (Sequence[Operation]): Loop-body operations to inspect.

    Returns:
        bool: Whether a direct or nested tracked observation is present.
    """
    for operation in operations:
        if isinstance(operation, _DESTRUCTIVE_LOOP_OBSERVATION_TYPES):
            return True
        if isinstance(operation, HasNestedOps) and any(
            _loop_body_has_destructive_observation(nested)
            for nested in operation.nested_op_lists()
        ):
            return True
    return False


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


def _captured_quantum_allocations(
    operations: Sequence[Operation],
    resolver: ExprResolver,
    allocation_owners_by_uuid: Mapping[str, str] | None = None,
) -> dict[str, ResourceExpr]:
    """Return outer quantum allocations captured by a nested operation list.

    Branch liveness must start with captured wires live so measuring or
    replacing them can release capacity before a branch-local allocation. A
    value is captured when it is read by the nested list but is not produced by
    any operation in that same list. Array elements and slice views carry fresh
    SSA UUIDs, so a value whose root owner comes from a body-local QInit is also
    excluded explicitly.

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
    local_allocation_owners_by_uuid = {
        result.uuid: _quantum_allocation_owner(result)
        for operation in operations
        if isinstance(operation, QInitOperation)
        for result in operation.results
        if isinstance(result, Value) and result.type.is_quantum()
    }
    local_allocation_owners = frozenset(local_allocation_owners_by_uuid.values())
    resolved_allocation_owners: Mapping[str, str] = ChainMap(
        local_allocation_owners_by_uuid,
        cast(MutableMapping[str, str], allocation_owners_by_uuid or {}),
    )
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
                resolved_allocation_owners,
            )
            if runtime_sizes is not None:
                for owner, size in runtime_sizes.items():
                    if owner in local_allocation_owners:
                        continue
                    captured[owner] = captured.get(owner, _ZERO) + size
                continue
            owner = _quantum_allocation_owner(value)
            if owner in local_allocation_owners:
                continue
            captured[owner] = _quantum_owner_capacity(value, resolver)
    return captured


def _with_operation_output_summary(
    estimate: ResourceEstimate,
    operation: ForOperation | WhileOperation | ForItemsOperation,
    resolver: ExprResolver,
    *,
    active_when: sp.Basic,
    body_output_sizes: Mapping[str, ResourceExpr] | None = None,
    additional_consumed_allocations: Mapping[str, ResourceExpr] | None = None,
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
        body_output_sizes (Mapping[str, ResourceExpr] | None): Authoritative
            retained live-owner summary across modeled body executions.
            Loop evaluators may provide owner-wise maxima when a final-state
            release cannot be proven. Defaults to ``None``.
        additional_consumed_allocations (Mapping[str, ResourceExpr] | None):
            Captured owners proven consumed by the union of concrete loop
            iterations, in addition to whole-owner body operations. Defaults
            to ``None``.
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
    body_consumed.update(additional_consumed_allocations or {})
    condition = _boolean_condition(active_when)
    consumed = {
        owner: _piecewise(size, _ZERO, condition)
        for owner, size in body_consumed.items()
    }
    output_sizes = _quantum_result_owner_sizes(
        operation.results,
        resolver,
        allocation_owners_by_uuid,
    )
    residual = sum(
        (
            size
            for owner, size in (body_output_sizes or {}).items()
            if owner not in captured
        ),
        _ZERO,
    )
    guarded_residual = _piecewise(residual, _ZERO, condition)
    if guarded_residual != _ZERO:
        output_sizes[f"{type(operation).__name__}:{id(operation)}/live"] = (
            guarded_residual
        )
    return dataclasses.replace(
        estimate,
        _output_sizes=output_sizes,
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


def _loop_captured_observation_consumption(
    operations: Sequence[Operation],
    captured: Mapping[str, ResourceExpr],
    resolvers: Sequence[ExprResolver],
    *,
    definite_iterations: bool,
    allocation_owners_by_uuid: Mapping[str, str] | None = None,
) -> tuple[dict[str, ResourceExpr], bool]:
    """Project destructive loop observations onto captured allocation slots.

    Every resolver represents one possible loop iteration. When the complete
    iteration set is concrete, unconditional observations are unioned by
    physical owner/index and an owner is released only if every scalar slot is
    covered. Observations nested under control flow are never treated as
    definite because they may not execute on every runtime path. For symbolic
    or truncated iteration sets, the helper reports uncertainty instead of
    claiming an exact final liveness state.

    Args:
        operations (Sequence[Operation]): Loop-body operations.
        captured (Mapping[str, ResourceExpr]): Live captured owner capacities.
        resolvers (Sequence[ExprResolver]): Per-iteration resolvers, or one
            symbolic probe resolver when the iteration set is unresolved.
        definite_iterations (bool): Whether ``resolvers`` enumerates every
            executed iteration exactly.
        allocation_owners_by_uuid (Mapping[str, str] | None): Optional QInit
            UUID to physical owner mapping. Defaults to ``None``.

    Returns:
        tuple[dict[str, ResourceExpr], bool]: Captured owners proven fully
            consumed, and whether any observed captured owner remains
            conservatively live.
    """
    if not resolvers or not captured:
        return {}, False
    owner_map = allocation_owners_by_uuid or {}
    fully_consumed: set[str] = set()
    covered_indices: dict[str, set[int]] = {}
    uncertain_owners: set[str] = set()

    def record_observation(
        operation: Operation,
        resolver: ExprResolver,
        *,
        unconditional: bool,
    ) -> None:
        """Record one destructive operation for one iteration resolver.

        Args:
            operation (Operation): Destructive observation operation.
            resolver (ExprResolver): Resolver for one iteration.
            unconditional (bool): Whether the operation is outside nested
                control flow and therefore executes on every represented path.
        """
        quantum_inputs = [
            value
            for value in operation.all_input_values()
            if isinstance(value, Value) and value.type.is_quantum()
        ]
        for value in quantum_inputs:
            sizes = _quantum_result_owner_sizes([value], resolver, owner_map)
            keys = _quantum_value_wire_keys(
                value,
                resolver,
                allocation_owners_by_uuid=owner_map,
            )
            for owner, size in sizes.items():
                capacity = captured.get(owner)
                if capacity is None:
                    continue
                if not definite_iterations or not unconditional:
                    uncertain_owners.add(owner)
                    continue
                if _safe_simplify(size - capacity) == _ZERO:
                    fully_consumed.add(owner)
                    continue
                concrete_keys = 0
                for key_owner, index in keys:
                    if key_owner != owner:
                        continue
                    if not isinstance(index, (int, sp.Expr)):
                        uncertain_owners.add(owner)
                        continue
                    normalized = _normalize_wire_index(index)
                    if isinstance(normalized, int):
                        covered_indices.setdefault(owner, set()).add(normalized)
                        concrete_keys += 1
                    elif isinstance(normalized, sp.Integer):
                        covered_indices.setdefault(owner, set()).add(int(normalized))
                        concrete_keys += 1
                if (
                    not size.is_number
                    or not _is_concrete_integer(size)
                    or concrete_keys < int(size)
                ):
                    uncertain_owners.add(owner)

    def visit(
        body: Sequence[Operation],
        resolver: ExprResolver,
        *,
        unconditional: bool,
    ) -> None:
        """Visit observations while retaining control-path certainty.

        Args:
            body (Sequence[Operation]): Operations to inspect.
            resolver (ExprResolver): Resolver for one iteration.
            unconditional (bool): Whether every operation in ``body`` is on
                the unconditional loop path.
        """
        for operation in body:
            if isinstance(operation, _DESTRUCTIVE_LOOP_OBSERVATION_TYPES):
                record_observation(
                    operation,
                    resolver,
                    unconditional=unconditional,
                )
            if isinstance(operation, HasNestedOps):
                for nested in operation.nested_op_lists():
                    visit(nested, resolver, unconditional=False)

    for iteration_resolver in resolvers:
        visit(operations, iteration_resolver, unconditional=True)

    for owner, indices in covered_indices.items():
        capacity = captured[owner]
        if (
            capacity.is_number
            and _is_concrete_integer(capacity)
            and int(capacity) >= 0
            and indices.issuperset(range(int(capacity)))
        ):
            fully_consumed.add(owner)
    consumed = {
        owner: (
            captured[owner] if owner in fully_consumed else sp.Integer(len(indices))
        )
        for owner, indices in covered_indices.items()
        if owner in fully_consumed or indices
    }
    consumed.update(
        {owner: captured[owner] for owner in fully_consumed if owner not in consumed}
    )
    uncertain_owners.difference_update(fully_consumed)
    return consumed, bool(uncertain_owners)


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
        parent_uuid = (
            runtime.element_parent_uuids[index]
            if index < len(runtime.element_parent_uuids)
            else ""
        )
        owner = allocation_owners_by_uuid.get(
            address[0] if address is not None else parent_uuid,
            logical_id,
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
        owner = (allocation_owners_by_uuid or {}).get(
            result.uuid,
            _quantum_allocation_owner(result),
        )
        returned[owner] = returned.get(owner, _ZERO) + _qubit_value_size(
            result,
            resolver,
        )
        capacities[owner] = _quantum_owner_capacity(result, resolver)
    return {owner: sp.Min(size, capacities[owner]) for owner, size in returned.items()}


def _quantum_owner_capacities(
    values: Sequence[ValueBase],
    resolver: ExprResolver,
    allocation_owners_by_uuid: Mapping[str, str] | None = None,
) -> dict[str, ResourceExpr]:
    """Resolve complete root-allocation widths touched by quantum values.

    A scalar array element keeps its complete root allocation live even though
    the element itself has width one. Conditional merge inputs use this helper
    because branch bodies can be structurally empty while their merge values
    still retain an enclosing allocation.

    Args:
        values (Sequence[ValueBase]): Candidate quantum values.
        resolver (ExprResolver): Resolver for symbolic root dimensions.
        allocation_owners_by_uuid (Mapping[str, str] | None): Optional QInit
            result UUID to logical-owner map. Defaults to ``None``.

    Returns:
        dict[str, ResourceExpr]: Complete capacity by root allocation owner.
    """
    capacities: dict[str, ResourceExpr] = {}
    owner_map = allocation_owners_by_uuid or {}
    for value in values:
        if not isinstance(value, Value) or not value.type.is_quantum():
            continue
        owner = owner_map.get(value.uuid, _quantum_allocation_owner(value))
        capacities[owner] = _resource_max(
            capacities.get(owner, _ZERO),
            _quantum_owner_capacity(value, resolver),
        )
    return capacities


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
    body_external_control_qubits: int,
    body_final_live: Mapping[str, ResourceExpr],
    actual_operands: Sequence[ValueBase],
    allocation_owners_by_uuid: Mapping[str, str] | None = None,
) -> tuple[dict[str, ResourceExpr], dict[str, ResourceExpr], bool]:
    """Map body liveness onto caller input and output allocation owners.

    A callee branch may return arrays whose branches have different symbolic
    widths even though the caller-side IR result retains one representative
    static shape. The callee resolver contains the merged shape binding, so
    invocation liveness must carry that size across the call boundary. The
    body summary also identifies allocations that remain live without being
    returned. Such residual workspace is retained under a call-local owner;
    allocations consumed inside the body are absent from the summary and are
    therefore not resurrected.

    Args:
        operation (InvokeOperation): Caller-side invocation.
        body (Block): Selected implementation body.
        child_resolver (ExprResolver): Resolver after evaluating the body.
        caller_resolver (ExprResolver): Resolver for caller-side controls.
        body_external_control_qubits (int): Number of leading caller controls
            implemented outside the selected body's input/output contract.
        body_final_live (Mapping[str, ResourceExpr]): Authoritative live widths
            by callee allocation owner after body evaluation.
        actual_operands (Sequence[ValueBase]): Caller operands aligned to the
            selected body after wrapper-only controls are removed.
        allocation_owners_by_uuid (Mapping[str, str] | None): Optional known
            QInit UUID to logical-owner map. Defaults to ``None``.

    Returns:
        tuple[dict[str, ResourceExpr], dict[str, ResourceExpr], bool]: Input
        widths, output widths, and whether positional output mapping was
        complete.
    """
    owner_map = allocation_owners_by_uuid or {}
    input_sizes = _quantum_result_owner_sizes(
        [value for value in operation.operands if value.type.is_quantum()],
        caller_resolver,
        owner_map,
    )
    sources: list[tuple[ValueBase, ExprResolver, bool]] = []
    if body_external_control_qubits:
        sources.extend(
            (operand, caller_resolver, False)
            for operand in operation.operands[:body_external_control_qubits]
        )
    sources.extend((output, child_resolver, True) for output in body.output_values)
    if len(sources) != len(operation.results):
        return input_sizes, {}, False

    output_sizes: dict[str, ResourceExpr] = {}
    capacities: dict[str, ResourceExpr] = {}
    returned_by_body_owner: dict[str, ResourceExpr] = {}
    for result, (source, source_resolver, comes_from_body) in zip(
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
        owner = owner_map.get(result.uuid, _quantum_allocation_owner(result))
        source_size = _qubit_value_size(source, source_resolver)
        output_sizes[owner] = output_sizes.get(owner, _ZERO) + _qubit_value_size(
            source,
            source_resolver,
        )
        capacities[owner] = _quantum_owner_capacity(result, caller_resolver)
        if comes_from_body:
            source_owner = owner_map.get(
                source.uuid,
                _quantum_allocation_owner(source),
            )
            returned_by_body_owner[source_owner] = (
                returned_by_body_owner.get(source_owner, _ZERO) + source_size
            )
    output_sizes = {
        owner: sp.Min(size, capacities[owner]) for owner, size in output_sizes.items()
    }

    formal_to_actual_owner: dict[str, str] = {}
    for formal, actual in pair_block_operands(body, actual_operands):
        if (
            not isinstance(formal, Value)
            or not isinstance(actual, Value)
            or not formal.type.is_quantum()
            or not actual.type.is_quantum()
        ):
            continue
        formal_owner = owner_map.get(
            formal.uuid,
            _quantum_allocation_owner(formal),
        )
        actual_owner = owner_map.get(
            actual.uuid,
            _quantum_allocation_owner(actual),
        )
        formal_to_actual_owner[formal_owner] = actual_owner

    namespace = f"{type(operation).__name__}:{id(operation)}/live"
    for body_owner, live_size in body_final_live.items():
        residual = sp.Max(
            _ZERO,
            live_size - returned_by_body_owner.get(body_owner, _ZERO),
        )
        if residual == _ZERO:
            continue
        caller_owner = formal_to_actual_owner.get(
            body_owner,
            f"{namespace}/{body_owner}",
        )
        output_sizes[caller_owner] = output_sizes.get(caller_owner, _ZERO) + residual
    return input_sizes, output_sizes, True
