"""Track guarded classical and measurement provenance for estimation."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import cast

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._classical_facts import _ResolvedClassicalFact
from qamomile.circuit.estimator._estimate import ResourceEstimate
from qamomile.circuit.estimator._measurement_provenance import (
    _merge_measurement_taint_conditions,
)
from qamomile.circuit.estimator._resolver import ExprResolver
from qamomile.circuit.estimator._resource_expressions import _boolean_condition
from qamomile.circuit.ir.dataflow import (
    build_dependency_graph,
    find_measurement_derived_values,
    walk_operations,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    LoopCarriedRebind,
    WhileOperation,
)
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.operation.gate import (
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
    ProjectOperation,
)
from qamomile.circuit.ir.operation.operation import Operation
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
    ValueLike,
    collect_value_like_uuids,
)

_OBSERVATION_SOURCE_PREFIX = "$observation"

_LOOP_ARRAY_SOURCE_PREFIX = "$loop-array"


def _operation_input_taint_condition(
    operation: Operation,
    taint_conditions: Mapping[str, Boolean],
) -> Boolean:
    """Return when an operation consumes observation-derived classical data.

    Args:
        operation (Operation): Operation whose inputs should be inspected.
        taint_conditions (Mapping[str, Boolean]): UUID-keyed observation
            provenance in the current evaluation scope.

    Returns:
        Boolean: Union of active provenance conditions on classical inputs.
    """
    conditions = [
        _value_taint_condition(value, taint_conditions)
        for value in operation.all_input_values()
        if isinstance(value, ValueBase) and not value.type.is_quantum()
    ]
    return _boolean_condition(sp.Or(*conditions)) if conditions else sp.false


def _value_taint_condition(
    value: ValueBase,
    taint_conditions: Mapping[str, Boolean],
) -> Boolean:
    """Return guarded observation provenance through value ancestry.

    Args:
        value (ValueBase): Classical scalar, container, or array element.
        taint_conditions (Mapping[str, Boolean]): UUID-keyed provenance map.

    Returns:
        Boolean: Union of provenance on the value and structural ancestors.
    """
    conditions = [
        taint_conditions[uuid]
        for uuid in collect_value_like_uuids(cast(ValueLike, value))
        if uuid in taint_conditions
    ]
    return _boolean_condition(sp.Or(*conditions)) if conditions else sp.false


def _merge_classical_source_conditions(
    target: dict[str, Boolean],
    additions: Mapping[str, sp.Basic],
) -> None:
    """Merge guarded observation-source conditions into one mapping.

    Args:
        target (dict[str, Boolean]): Mutable destination keyed by source token.
        additions (Mapping[str, sp.Basic]): Source guards to union into the
            destination.
    """
    for source, raw_condition in additions.items():
        condition = _boolean_condition(raw_condition)
        merged = _boolean_condition(sp.Or(target.get(source, sp.false), condition))
        if merged is sp.false:
            target.pop(source, None)
        else:
            target[source] = merged


def _resolved_classical_source_conditions(
    values: Iterable[ValueBase],
    resolver: ExprResolver,
    *,
    ignored_uuids: frozenset[str] = frozenset(),
) -> dict[str, Boolean]:
    """Resolve semantic classical dependencies without broadening elements.

    A classical array element is resolved as one fact. Its parent array is not
    traversed again, because doing so would reintroduce a coarse whole-array
    dependency after the persistent array state proved the selected element
    clean. Quantum values contribute only classical address metadata.

    Args:
        values (Iterable[ValueBase]): Values whose semantic dependencies are
            collected.
        resolver (ExprResolver): Resolver carrying classical facts.
        ignored_uuids (frozenset[str]): Structural inputs that are not consumed
            by the operation. Defaults to an empty set.

    Returns:
        dict[str, Boolean]: Classical source tokens and activation guards.
    """
    source_conditions: dict[str, Boolean] = {}

    def visit(value: ValueBase) -> None:
        """Collect dependencies from one aggregate or scalar value.

        Args:
            value (ValueBase): Value to inspect.
        """
        if isinstance(value, TupleValue):
            for element in value.elements:
                visit(element)
            return
        if isinstance(value, DictValue):
            for key, entry_value in value.entries:
                visit(key)
                visit(entry_value)
            return
        if not isinstance(value, Value) or value.uuid in ignored_uuids:
            return
        if value.type.is_quantum():
            metadata: list[Value] = []
            if isinstance(value, ArrayValue):
                metadata.extend(value.shape)
                if value.slice_start is not None:
                    metadata.append(value.slice_start)
                if value.slice_step is not None:
                    metadata.append(value.slice_step)
            else:
                metadata.extend(value.element_indices)
            for item in metadata:
                _merge_classical_source_conditions(
                    source_conditions,
                    resolver.resolve_classical_fact(item).dependencies,
                )
            return
        if isinstance(value, ArrayValue):
            _merge_classical_source_conditions(
                source_conditions,
                resolver.array_state_dependencies(value),
            )
            for item in (
                *value.shape,
                *((value.slice_start,) if value.slice_start is not None else ()),
                *((value.slice_step,) if value.slice_step is not None else ()),
            ):
                _merge_classical_source_conditions(
                    source_conditions,
                    resolver.resolve_classical_fact(item).dependencies,
                )
            return
        _merge_classical_source_conditions(
            source_conditions,
            resolver.resolve_classical_fact(value).dependencies,
        )

    for root in values:
        visit(root)
    return source_conditions


def _resolved_classical_facts(
    values: Iterable[ValueBase],
    resolver: ExprResolver,
) -> tuple[_ResolvedClassicalFact, ...]:
    """Resolve semantic scalar or aggregate facts without broad ancestry.

    Args:
        values (Iterable[ValueBase]): Classical values read by one operation.
        resolver (ExprResolver): Resolver carrying value and array state.

    Returns:
        tuple[_ResolvedClassicalFact, ...]: Facts associated with the supplied
            semantic inputs.
    """
    facts: list[_ResolvedClassicalFact] = []

    def visit(value: ValueBase) -> None:
        """Append facts represented by one structured value.

        Args:
            value (ValueBase): Value to inspect.
        """
        if isinstance(value, TupleValue):
            for element in value.elements:
                visit(element)
            return
        if isinstance(value, DictValue):
            for key, entry_value in value.entries:
                visit(key)
                visit(entry_value)
            return
        if isinstance(value, Value) and not value.type.is_quantum():
            facts.append(resolver.resolve_classical_fact(value))

    for root in values:
        visit(root)
    return tuple(facts)


def _classical_fact_runtime_condition(
    fact: _ResolvedClassicalFact,
) -> Boolean:
    """Return when a resolved classical fact depends on an observation.

    Args:
        fact (_ResolvedClassicalFact): Resolved value and guarded source tokens.

    Returns:
        Boolean: Union of all active source guards, or false when independent.
    """
    dependencies = {
        source: guard
        for source, guard in fact.dependencies.items()
        if source.startswith(_OBSERVATION_SOURCE_PREFIX)
    }
    return (
        _boolean_condition(sp.Or(*dependencies.values())) if dependencies else sp.false
    )


def _classical_fact_uncertainty_condition(
    fact: _ResolvedClassicalFact,
) -> Boolean:
    """Return when a fact depends on a conservative loop-array fallback.

    Loop-array source tokens are scheduler readiness edges, not runtime
    observations. Keeping this condition separate prevents an unresolved
    symbolic loop state from being mistaken for measurement feed-forward.

    Args:
        fact (_ResolvedClassicalFact): Resolved value and guarded source tokens.

    Returns:
        Boolean: Union of active loop-array fallback guards, or false.
    """
    dependencies = {
        source: guard
        for source, guard in fact.dependencies.items()
        if source.startswith(_LOOP_ARRAY_SOURCE_PREFIX)
    }
    return (
        _boolean_condition(sp.Or(*dependencies.values())) if dependencies else sp.false
    )


def _operation_classical_dependency_inputs(
    operation: Operation,
) -> tuple[ValueBase, ...]:
    """Return values semantically read by one scheduling boundary.

    Structured ``if`` and loop regions require guarded traversal and are
    handled by ``_scheduling_classical_input_sources``. Other operations retain
    their ordinary IR input list.

    Args:
        operation (Operation): Operation whose semantic reads are requested.

    Returns:
        tuple[ValueBase, ...]: Values whose classical readiness can delay the
            operation.
    """
    return tuple(operation.all_input_values())


def _operation_classical_dependency_outputs(
    operation: Operation,
) -> tuple[ValueBase, ...]:
    """Return classical boundary values whose source tokens become ready.

    Loop-carried array rewrites are represented by ``LoopCarriedRebind``
    records rather than ordinary operation results. Their exposed ``after``
    values must still publish accumulated observation readiness at the loop
    boundary so later feed-forward work cannot start at loop entry.

    Args:
        operation (Operation): Operation whose boundary outputs are requested.

    Returns:
        tuple[ValueBase, ...]: Ordinary results plus loop-rebound values.
    """
    outputs: list[ValueBase] = list(operation.results)
    if isinstance(operation, (ForOperation, ForItemsOperation, WhileOperation)):
        outputs.extend(rebind.after for rebind in _loop_array_state_rebinds(operation))
        outputs.extend(
            rebind.after
            for rebind in operation.loop_carried_rebinds
            if not isinstance(rebind.after, ArrayValue)
        )
    return tuple(outputs)


def _loop_array_state_rebinds(
    operation: ForOperation | ForItemsOperation | WhileOperation,
) -> tuple[LoopCarriedRebind, ...]:
    """Return explicit and inferred classical-array loop state boundaries.

    A direct element Store can expose its produced array SSA value after a loop
    without creating a frontend ``LoopCarriedRebind`` record. The estimator
    infers that boundary from each logical lineage's unproduced entry operand
    and last produced result so detached iteration scopes retain the same
    semantics as the shared traced graph.

    Args:
        operation (ForOperation | ForItemsOperation | WhileOperation): Loop
            whose classical-array state boundaries are requested.

    Returns:
        tuple[LoopCarriedRebind, ...]: Explicit records followed by inferred
        classical-array records in body order.
    """
    explicit = tuple(
        rebind
        for rebind in operation.loop_carried_rebinds
        if isinstance(rebind.before, ArrayValue)
        and isinstance(rebind.after, ArrayValue)
        and not rebind.before.type.is_quantum()
        and not rebind.after.type.is_quantum()
    )
    covered_lineages = {
        cast(ArrayValue, rebind.before).logical_id for rebind in explicit
    }
    nested = tuple(walk_operations(operation.operations))
    produced = {
        result.uuid
        for nested_operation in nested
        for result in nested_operation.results
        if isinstance(result, ArrayValue)
    }
    entries: dict[str, ArrayValue] = {}
    exits: dict[str, ArrayValue] = {}
    for nested_operation in nested:
        for operand in nested_operation.operands:
            if (
                isinstance(operand, ArrayValue)
                and not operand.type.is_quantum()
                and operand.uuid not in produced
            ):
                entries.setdefault(operand.logical_id, operand)
        for result in nested_operation.results:
            if isinstance(result, ArrayValue) and not result.type.is_quantum():
                exits[result.logical_id] = result
    inferred = tuple(
        LoopCarriedRebind(
            var_name=entry.name,
            before=entry,
            after=exits[logical_id],
        )
        for logical_id, entry in entries.items()
        if logical_id in exits and logical_id not in covered_lineages
    )
    return (*explicit, *inferred)


def _structural_value_ancestry(value: ValueBase) -> tuple[ValueBase, ...]:
    """Return one value and recursively embedded structural values.

    Args:
        value (ValueBase): Scalar, array, tuple, or dictionary IR value.

    Returns:
        tuple[ValueBase, ...]: Identity-deduplicated ancestry in visit order.
    """
    ordered: list[ValueBase] = []
    visited: set[str] = set()

    def visit(current: ValueBase) -> None:
        """Visit one structural value.

        Args:
            current (ValueBase): Value whose embedded ancestry is traversed.
        """
        if current.uuid in visited:
            return
        visited.add(current.uuid)
        ordered.append(current)
        if isinstance(current, TupleValue):
            for element in current.elements:
                visit(element)
        elif isinstance(current, DictValue):
            for key, entry_value in current.entries:
                visit(key)
                visit(entry_value)
        elif isinstance(current, ArrayValue):
            for dimension in current.shape:
                visit(dimension)
            if current.slice_of is not None:
                visit(current.slice_of)
            if current.slice_start is not None:
                visit(current.slice_start)
            if current.slice_step is not None:
                visit(current.slice_step)
        elif isinstance(current, Value):
            if current.parent_array is not None:
                visit(current.parent_array)
            for index in current.element_indices:
                visit(index)

    visit(value)
    return tuple(ordered)


def _direct_observation_result_uuids(operation: Operation) -> tuple[str, ...]:
    """Return result UUIDs directly produced by a runtime observation.

    Args:
        operation (Operation): Operation whose direct results should be
            classified.

    Returns:
        tuple[str, ...]: Direct measurement or expectation-value result UUIDs.
    """
    if isinstance(
        operation,
        (MeasureOperation, MeasureVectorOperation, MeasureQFixedOperation, ExpvalOp),
    ):
        return tuple(result.uuid for result in operation.results)
    if isinstance(operation, ProjectOperation) and len(operation.results) > 1:
        return (operation.results[1].uuid,)
    return ()


def _direct_observation_source_conditions(
    operation: Operation,
    resolver: ExprResolver,
) -> dict[str, Boolean]:
    """Return source tokens created by one direct observation operation.

    Array-state projection deliberately avoids treating a whole array as one
    element dependency. A vector observation is the exception at its producer
    boundary: its aggregate result fact is where the scheduler first learns
    that the source token becomes ready. Element reads remain precise after
    this publication.

    Args:
        operation (Operation): Operation whose direct observation results are
            inspected.
        resolver (ExprResolver): Resolver containing the newly published
            result facts.

    Returns:
        dict[str, Boolean]: Newly created source tokens and activation guards.
    """
    direct_results = frozenset(_direct_observation_result_uuids(operation))
    source_conditions: dict[str, Boolean] = {}
    for result in operation.results:
        if not isinstance(result, Value) or result.uuid not in direct_results:
            continue
        _merge_classical_source_conditions(
            source_conditions,
            resolver.resolve_classical_fact(result).dependencies,
        )
    return source_conditions


def _propagate_operation_measurement_taint(
    operation: Operation,
    estimate: ResourceEstimate,
    inherited: Mapping[str, Boolean],
) -> dict[str, Boolean]:
    """Propagate guarded observation provenance across one operation.

    Structured operations publish branch- and loop-aware result conditions in
    their estimate. Ordinary operations retain the existing conservative
    all-input-to-all-result dataflow rule without flattening nested region
    edges into the enclosing scope.

    Args:
        operation (Operation): Evaluated operation.
        estimate (ResourceEstimate): Operation estimate carrying any nested
            result provenance.
        inherited (Mapping[str, Boolean]): Provenance before the operation.

    Returns:
        dict[str, Boolean]: Updated provenance for the current scope.
    """
    updated = _merge_measurement_taint_conditions(
        inherited,
        estimate._measurement_taint_conditions,
    )
    if not isinstance(operation, HasNestedOps):
        input_condition = _operation_input_taint_condition(operation, updated)
        if input_condition is not sp.false:
            updated = _merge_measurement_taint_conditions(
                updated,
                {result.uuid: input_condition for result in operation.results},
            )
    direct = _direct_observation_result_uuids(operation)
    if direct:
        updated = _merge_measurement_taint_conditions(
            updated,
            {uuid: sp.true for uuid in direct},
        )
    return updated


@dataclasses.dataclass(frozen=True)
class _LoopMayTaint:
    """Summarize path-insensitive observation provenance for a loop.

    Args:
        at_iteration (dict[str, Boolean]): Carried block arguments that may be
            observation-derived in at least one loop iteration.
        final (dict[str, Boolean]): Carried block arguments whose corresponding
            loop results may be observation-derived after the loop.
    """

    at_iteration: dict[str, Boolean]
    final: dict[str, Boolean]


def _loop_may_taint(
    operation: ForOperation | ForItemsOperation,
    initial: Mapping[str, Boolean],
    probe_estimate: ResourceEstimate,
) -> _LoopMayTaint:
    """Compute a two-point may-be-runtime fixed point for a symbolic loop.

    The analysis intentionally discards path and iteration correlations.  A
    source that can reach a carried value in any iteration marks that value as
    runtime-derived for resource decisions.  This is the requested
    conservative policy and avoids constructing a symbolic Boolean recurrence
    whose exactness is not useful once runtime branches are combined by their
    field-wise maxima.

    Args:
        operation (ForOperation | ForItemsOperation): Symbolic region loop.
        initial (Mapping[str, Boolean]): Observation provenance on entry.
        probe_estimate (ResourceEstimate): One body probe carrying selected
            callable observation results and ordinary dataflow propagation.

    Returns:
        _LoopMayTaint: May-provenance at an arbitrary iteration and loop exit.
    """
    graph = build_dependency_graph(operation.operations)
    for arg in operation.region_args:
        # The next iteration's block argument receives this iteration's yield.
        graph.setdefault(arg.block_arg.uuid, set()).add(arg.yielded.uuid)
    seeds = {uuid for uuid, condition in initial.items() if condition is not sp.false}
    seeds.update(
        uuid
        for uuid, condition in probe_estimate._measurement_taint_conditions.items()
        if condition is not sp.false
    )
    # A yielded array element can depend on an observation through its index
    # or view metadata without having a producer edge of its own. Seed that
    # synthetic leaf explicitly so the loop backedge cannot turn it into an
    # apparently clean public carry.
    seeds.update(
        arg.yielded.uuid
        for arg in operation.region_args
        if _value_taint_condition(
            arg.yielded,
            probe_estimate._measurement_taint_conditions,
        )
        is not sp.false
    )
    derived = find_measurement_derived_values(graph, seeds)
    at_iteration = {
        arg.block_arg.uuid: sp.true
        for arg in operation.region_args
        if arg.block_arg.uuid in derived
        or initial.get(arg.block_arg.uuid, sp.false) is not sp.false
    }
    final = {
        arg.block_arg.uuid: sp.true
        for arg in operation.region_args
        if initial.get(arg.block_arg.uuid, sp.false) is not sp.false
        or arg.yielded.uuid in derived
    }
    return _LoopMayTaint(at_iteration=at_iteration, final=final)
