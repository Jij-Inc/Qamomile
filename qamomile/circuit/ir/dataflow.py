"""Backend-independent dataflow utilities for semantic Qamomile IR."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

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
    MeasureVectorOperation,
    ProjectOperation,
)
from qamomile.circuit.ir.value import ValueBase


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
        set[str]: Direct scalar, vector, fixed-point, and projection results.
    """
    results: set[str] = set()
    for operation in walk_operations(operations):
        if isinstance(
            operation,
            (MeasureOperation, MeasureVectorOperation, MeasureQFixedOperation),
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
