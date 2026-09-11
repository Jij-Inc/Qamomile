"""Engine-independent dataflow utilities for semantic Qamomile IR."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureQIntOperation,
    MeasureVectorOperation,
    ProjectOperation,
)
from qamomile.circuit.ir.types.primitives import BitType
from qamomile.circuit.ir.value import ArrayValue, Value, ValueBase


def walk_operations(operations: Sequence[Operation]) -> Iterable[Operation]:
    """Yield operations in preorder across every nested control-flow region.

    Args:
        operations (Sequence[Operation]): Top-level semantic operations.

    Returns:
        Iterable[Operation]: Preorder traversal including nested operations.
    """
    for operation in operations:
        yield operation
        if isinstance(operation, HasNestedOps):
            for region in operation.nested_regions():
                yield from walk_operations(region.operations)


def _seed_structural_edges(
    graph: dict[str, set[str]],
    value: object,
) -> None:
    """Add array-element and slice ancestry edges for one IR value.

    Args:
        graph (dict[str, set[str]]): Dependency graph to update.
        value (object): Candidate value carrying array ancestry metadata.
    """
    if not isinstance(value, ValueBase):
        return
    parent = getattr(value, "parent_array", None)
    if parent is None:
        return
    graph.setdefault(value.uuid, set()).add(parent.uuid)
    current = parent
    while getattr(current, "slice_of", None) is not None:
        graph.setdefault(current.uuid, set()).add(current.slice_of.uuid)
        current = current.slice_of


def build_dependency_graph(operations: Sequence[Operation]) -> dict[str, set[str]]:
    """Build result-to-input dependency edges for semantic operations.

    The graph includes nested control flow, branch merges, loop-carried region
    arguments, array-element ancestry, and slice ancestry. These are the
    shared semantics used by measurement provenance, kernel effects, and the
    compiler's classical lowering passes.

    Args:
        operations (Sequence[Operation]): Top-level semantic operations.

    Returns:
        dict[str, set[str]]: Result UUIDs mapped to the UUIDs they depend on.
    """
    graph: dict[str, set[str]] = {}
    for operation in walk_operations(operations):
        operand_uuids = {
            value.uuid for value in operation.operands if isinstance(value, ValueBase)
        }
        for result in operation.results:
            graph.setdefault(result.uuid, set()).update(operand_uuids)
        for value in operation.operands:
            _seed_structural_edges(graph, value)

        if isinstance(operation, IfOperation):
            condition = operation.operands[0] if operation.operands else None
            condition_uuid = (
                condition.uuid if isinstance(condition, ValueBase) else None
            )
            for merge in operation.iter_merges():
                dependencies = graph.setdefault(merge.result.uuid, set())
                if condition_uuid is not None:
                    dependencies.add(condition_uuid)
                dependencies.add(merge.true_value.uuid)
                dependencies.add(merge.false_value.uuid)
                _seed_structural_edges(graph, merge.true_value)
                _seed_structural_edges(graph, merge.false_value)

        if isinstance(operation, (ForOperation, ForItemsOperation, WhileOperation)):
            for region_arg in operation.region_args:
                dependencies = {
                    region_arg.init.uuid,
                    region_arg.yielded.uuid,
                }
                graph.setdefault(region_arg.block_arg.uuid, set()).update(dependencies)
                graph.setdefault(region_arg.result.uuid, set()).update(dependencies)
                for value in (
                    region_arg.init,
                    region_arg.block_arg,
                    region_arg.yielded,
                    region_arg.result,
                ):
                    _seed_structural_edges(graph, value)
    return graph


def find_measurement_results(operations: Sequence[Operation]) -> set[str]:
    """Return UUIDs directly produced from quantum measurement.

    Args:
        operations (Sequence[Operation]): Top-level semantic operations.

    Returns:
        set[str]: Direct scalar, vector, quantum-integer, fixed-point, and
            projection results.
    """
    results: set[str] = set()
    for operation in walk_operations(operations):
        if isinstance(
            operation,
            (
                MeasureOperation,
                MeasureVectorOperation,
                MeasureQFixedOperation,
                MeasureQIntOperation,
            ),
        ):
            results.update(result.uuid for result in operation.results)
        elif isinstance(operation, ProjectOperation):
            results.add(operation.results[1].uuid)
    return results


def find_measurement_derived_values(
    dependency_graph: dict[str, set[str]],
    measurement_uuids: set[str],
) -> set[str]:
    """Propagate measurement provenance forward through a dependency graph.

    Args:
        dependency_graph (dict[str, set[str]]): Result UUIDs mapped to their
            dependency UUIDs.
        measurement_uuids (set[str]): Direct measurement-result UUIDs.

    Returns:
        set[str]: Direct and transitively measurement-derived UUIDs.
    """
    dependents: dict[str, list[str]] = {}
    for uuid, dependencies in dependency_graph.items():
        for dependency in dependencies:
            dependents.setdefault(dependency, []).append(uuid)

    derived: set[str] = set()
    worklist = list(measurement_uuids)
    while worklist:
        current = worklist.pop()
        if current in derived:
            continue
        derived.add(current)
        for dependent in dependents.get(current, ()):
            if dependent not in derived:
                worklist.append(dependent)
    return derived


def find_loop_carried_condition_reads(
    loop_operation: ForOperation | ForItemsOperation | WhileOperation,
    *,
    condition_values: Sequence[ValueBase] | None = None,
    selected_aliases: Mapping[str, str] | None = None,
) -> set[tuple[str, str]]:
    """Find legacy loop rebinds whose entry value controls a nested branch.

    A legacy ``LoopCarriedRebind`` does not provide runtime storage between
    iterations.  If its entry value transitively feeds an ``IfOperation``
    condition, pruning that first-iteration branch must therefore not erase
    the evidence that a later iteration would read the updated value.

    Args:
        loop_operation (ForOperation | ForItemsOperation | WhileOperation):
            Loop whose legacy rebind records are inspected.
        condition_values (Sequence[ValueBase] | None): Optional branch
            conditions from a reachability-aware caller. When omitted, every
            nested ``IfOperation`` condition in the loop body is considered.
        selected_aliases (Mapping[str, str] | None): Optional merge-result to
            selected-source aliases established by branch specialization.

    Returns:
        set[tuple[str, str]]: ``(before_uuid, after_uuid)`` pairs for rebinds
            whose entry value transitively influences a considered condition.
    """
    body_operations = tuple(
        operation
        for region in loop_operation.nested_regions()
        for operation in region.operations
    )
    considered_conditions = (
        tuple(condition_values)
        if condition_values is not None
        else tuple(
            operation.condition
            for operation in walk_operations(body_operations)
            if isinstance(operation, IfOperation)
        )
    )

    return find_loop_carried_value_reads(
        loop_operation,
        considered_conditions,
        selected_aliases=selected_aliases,
    )


def find_loop_carried_value_reads(
    loop_operation: ForOperation | ForItemsOperation | WhileOperation,
    values: Sequence[ValueBase],
    *,
    selected_aliases: Mapping[str, str] | None = None,
) -> set[tuple[str, str]]:
    """Find unsupported scalar Bit carries read by selected body values.

    Args:
        loop_operation (ForOperation | ForItemsOperation | WhileOperation):
            Loop whose legacy rebind records are inspected.
        values (Sequence[ValueBase]): Reached operation inputs whose transitive
            dependencies are checked.
        selected_aliases (Mapping[str, str] | None): Optional merge-result to
            selected-source aliases. An alias replaces the merge's ordinary
            dependency edges because the other branch is unreachable.

    Returns:
        set[tuple[str, str]]: ``(before_uuid, after_uuid)`` pairs for legacy
            scalar Bit rebinds read by at least one selected value.
    """
    records = loop_operation.loop_carried_rebinds
    if not records or not values:
        return set()
    body_operations = tuple(
        operation
        for region in loop_operation.nested_regions()
        for operation in region.operations
    )
    dependency_graph = build_dependency_graph(body_operations)
    aliases = selected_aliases or {}

    def canonical(uuid: str) -> str:
        """Follow selected merge aliases without crossing a cycle.

        Args:
            uuid (str): Starting value UUID.

        Returns:
            str: First UUID without a selected alias.
        """
        visited: set[str] = set()
        while uuid in aliases and uuid not in visited:
            visited.add(uuid)
            uuid = aliases[uuid]
        return uuid

    def depends_on(value_uuid: str, source_uuid: str) -> bool:
        """Return whether one value transitively depends on another.

        Args:
            value_uuid (str): UUID whose dependency chain is traversed.
            source_uuid (str): UUID sought in that dependency chain.

        Returns:
            bool: Whether ``source_uuid`` is reachable from ``value_uuid``.
        """
        pending = [value_uuid]
        visited: set[str] = set()
        while pending:
            current = pending.pop()
            aliased = canonical(current)
            if aliased != current:
                pending.append(aliased)
                continue
            if current == source_uuid:
                return True
            if current in visited:
                continue
            visited.add(current)
            pending.extend(dependency_graph.get(current, ()))
        return False

    return {
        (record.before.uuid, record.after.uuid)
        for record in records
        if isinstance(record.before, Value)
        and not isinstance(record.before, ArrayValue)
        and isinstance(record.after, Value)
        and not isinstance(record.after, ArrayValue)
        and isinstance(record.before.type, BitType)
        and isinstance(record.after.type, BitType)
        and canonical(record.after.uuid) != record.before.uuid
        if any(depends_on(value.uuid, record.before.uuid) for value in values)
    }


def find_loop_carried_condition_uuids(
    operations: Sequence[Operation],
) -> set[str]:
    """Find branch conditions that read legacy loop-carried scalar Bits.

    Compile-time specialization normally removes a branch whose first traced
    condition is a constant. That is unsafe while recursively unrolling a body
    when the same condition reads a legacy loop-carried Bit: later iterations
    observe the refreshed value, so the final loop-state validator still needs
    the branch as dependency evidence.

    Args:
        operations (Sequence[Operation]): Semantic operation tree to inspect.

    Returns:
        set[str]: UUIDs of conditions that transitively depend on a legacy
            scalar Bit entry value.
    """
    preserved: set[str] = set()
    for operation in walk_operations(operations):
        if not isinstance(
            operation,
            (ForOperation, ForItemsOperation, WhileOperation),
        ):
            continue
        body_operations = tuple(
            nested
            for region in operation.nested_regions()
            for nested in region.operations
        )
        for nested in walk_operations(body_operations):
            if not isinstance(nested, IfOperation):
                continue
            if find_loop_carried_value_reads(operation, [nested.condition]):
                preserved.add(nested.condition.uuid)
    return preserved


def has_legacy_scalar_bit_rebinds(operations: Sequence[Operation]) -> bool:
    """Return whether an operation tree contains legacy scalar Bit state.

    Args:
        operations (Sequence[Operation]): Top-level semantic operations.

    Returns:
        bool: Whether a loop carries a scalar ``Bit`` through legacy rebind
            metadata rather than an explicit region argument.
    """
    return any(
        isinstance(operation, (ForOperation, ForItemsOperation, WhileOperation))
        and any(
            isinstance(record.before, Value)
            and not isinstance(record.before, ArrayValue)
            and isinstance(record.after, Value)
            and not isinstance(record.after, ArrayValue)
            and isinstance(record.before.type, BitType)
            and isinstance(record.after.type, BitType)
            for record in operation.loop_carried_rebinds
        )
        for operation in walk_operations(operations)
    )
