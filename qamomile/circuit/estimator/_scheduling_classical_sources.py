"""Resolve guarded classical inputs at scheduling boundaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TypeAlias, cast

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._classical_provenance import (
    _merge_classical_source_conditions,
    _operation_classical_dependency_inputs,
    _resolved_classical_source_conditions,
)
from qamomile.circuit.estimator._constants import _ZERO
from qamomile.circuit.estimator._loop_executor import symbolic_iterations
from qamomile.circuit.estimator._resolver import ExprResolver
from qamomile.circuit.estimator._resource_base import ResourceExpr
from qamomile.circuit.estimator._resource_expressions import (
    _activation_over_range,
    _and_conditions,
    _boolean_condition,
)
from qamomile.circuit.estimator._scopes import (
    build_for_items_scope,
    build_for_loop_scope,
    build_if_scopes,
    build_while_scope,
    resolve_for_items_cardinality,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.operation import Operation
from qamomile.circuit.ir.value import ValueBase

_RangeBounds: TypeAlias = tuple[
    sp.Symbol,
    ResourceExpr,
    ResourceExpr,
    ResourceExpr,
]
_EnclosingRangeDomains: TypeAlias = tuple[_RangeBounds, ...]
_WhileOperationKey: TypeAlias = tuple[tuple[tuple[int, int], ...], int]
_WhileTripCountNames: TypeAlias = Mapping[
    _WhileOperationKey,
    tuple[WhileOperation, str],
]


def _project_over_ranges(
    condition: Boolean,
    ranges: _EnclosingRangeDomains,
) -> Boolean:
    """Project a per-iteration guard to the enclosing boundary.

    Args:
        condition (Boolean): Guard expressed in nested loop scopes.
        ranges (_EnclosingRangeDomains): Enclosing range domains from outer
            to inner.

    Returns:
        Boolean: Existential guard with every range induction symbol bound
        inside a finite-domain predicate.
    """
    projected: sp.Basic = condition
    for loop_symbol, start, step, iterations in reversed(ranges):
        projected = _activation_over_range(
            projected,
            loop_symbol,
            start,
            step,
            iterations,
        )
    return _boolean_condition(projected)


def _add_values(
    collected: dict[str, Boolean],
    values: Iterable[ValueBase],
    scope: ExprResolver,
    active_when: Boolean,
    ranges: _EnclosingRangeDomains,
) -> None:
    """Add guarded source tokens from semantic input values.

    Args:
        collected (dict[str, Boolean]): Run-local accumulator to update.
        values (Iterable[ValueBase]): Values read by the region.
        scope (ExprResolver): Resolver for the values' region.
        active_when (Boolean): Guard under which the read executes.
        ranges (_EnclosingRangeDomains): Enclosing finite range domains.
    """
    if active_when is sp.false:
        return
    additions = _resolved_classical_source_conditions(values, scope)
    _merge_classical_source_conditions(
        collected,
        {
            source: _project_over_ranges(
                _and_conditions(active_when, guard),
                ranges,
            )
            for source, guard in additions.items()
        },
    )


def _branch_conditions(
    predicate: Boolean,
    active_when: Boolean,
    unprojectable_symbols: frozenset[sp.Symbol],
) -> tuple[Boolean, Boolean]:
    """Return guarded true and false activity for one nested branch.

    Args:
        predicate (Boolean): Resolved branch predicate.
        active_when (Boolean): Activity inherited from parent regions.
        unprojectable_symbols (frozenset[sp.Symbol]): Loop-local symbols whose
            finite values are unavailable at this boundary, such as dictionary
            keys.

    Returns:
        tuple[Boolean, Boolean]: True- and false-region guards. When an
        unprojectable local symbol participates, both retain the parent guard
        as a safe envelope.
    """
    if predicate.free_symbols & unprojectable_symbols:
        return active_when, active_when
    return (
        _and_conditions(active_when, predicate),
        _and_conditions(
            active_when,
            cast(Boolean, sp.Not(predicate)),
        ),
    )


def _visit_operations(
    operations: Iterable[Operation],
    scope: ExprResolver,
    active_when: Boolean,
    ranges: _EnclosingRangeDomains,
    unprojectable_symbols: frozenset[sp.Symbol],
    collected: dict[str, Boolean],
    while_trip_count_names: _WhileTripCountNames,
) -> None:
    """Visit one structured region in program order.

    Args:
        operations (Iterable[Operation]): Region operations to visit.
        scope (ExprResolver): Resolver for the region.
        active_when (Boolean): Guard under which the region executes.
        ranges (_EnclosingRangeDomains): Enclosing finite range domains.
        unprojectable_symbols (frozenset[sp.Symbol]): Loop-local symbols
            without an enumerable range.
        collected (dict[str, Boolean]): Run-local source accumulator to update.
        while_trip_count_names (_WhileTripCountNames): Read-only names reserved
            for while-loop trip counts in the current interpretation run.
    """
    for nested in operations:
        _visit(
            nested,
            scope,
            active_when,
            ranges,
            unprojectable_symbols,
            collected,
            while_trip_count_names,
        )


def _visit(
    current: Operation,
    scope: ExprResolver,
    active_when: Boolean,
    ranges: _EnclosingRangeDomains,
    unprojectable_symbols: frozenset[sp.Symbol],
    collected: dict[str, Boolean],
    while_trip_count_names: _WhileTripCountNames,
) -> None:
    """Collect source reads from one atomic or structured operation.

    Args:
        current (Operation): Operation to inspect.
        scope (ExprResolver): Resolver for the operation's region.
        active_when (Boolean): Guard under which the operation executes.
        ranges (_EnclosingRangeDomains): Enclosing finite range domains.
        unprojectable_symbols (frozenset[sp.Symbol]): Loop-local symbols
            without an enumerable range.
        collected (dict[str, Boolean]): Run-local source accumulator to update.
        while_trip_count_names (_WhileTripCountNames): Read-only names reserved
            for while-loop trip counts in the current interpretation run.
    """
    if active_when is sp.false:
        return
    if isinstance(current, IfOperation):
        _add_values(collected, (current.condition,), scope, active_when, ranges)
        predicate = _boolean_condition(
            scope.resolve_classical_fact(current.condition).value
        )
        true_active, false_active = _branch_conditions(
            predicate,
            active_when,
            unprojectable_symbols,
        )
        true_child, false_child = build_if_scopes(current, scope)
        _visit_operations(
            current.true_operations,
            true_child,
            true_active,
            ranges,
            unprojectable_symbols,
            collected,
            while_trip_count_names,
        )
        _visit_operations(
            current.false_operations,
            false_child,
            false_active,
            ranges,
            unprojectable_symbols,
            collected,
            while_trip_count_names,
        )
        return
    if isinstance(current, ForOperation) and len(current.operands) >= 3:
        _add_values(collected, current.operands[:3], scope, active_when, ranges)
        child, start, stop, step, loop_symbol = build_for_loop_scope(current, scope)
        iterations = symbolic_iterations(start, stop, step)
        _visit_operations(
            current.operations,
            child,
            active_when,
            (*ranges, (loop_symbol, start, step, iterations)),
            unprojectable_symbols,
            collected,
            while_trip_count_names,
        )
        return
    if isinstance(current, ForItemsOperation) and current.operands:
        loop_active = _and_conditions(
            active_when,
            cast(Boolean, sp.Gt(resolve_for_items_cardinality(current), _ZERO)),
        )
        # Dictionary values are not structural bounds. Guard their readiness
        # by nonempty iteration before visiting the body.
        _add_values(collected, current.operands, scope, loop_active, ranges)
        child = build_for_items_scope(current, scope)
        local_values = (
            *(current.key_var_values or ()),
            *((current.value_var_value,) if current.value_var_value else ()),
        )
        local_symbols = {
            symbol
            for value in local_values
            for symbol in child.resolve(value).free_symbols
            if isinstance(symbol, sp.Symbol)
        }
        _visit_operations(
            current.operations,
            child,
            loop_active,
            ranges,
            unprojectable_symbols | frozenset(local_symbols),
            collected,
            while_trip_count_names,
        )
        return
    if isinstance(current, WhileOperation):
        if current.operands:
            _add_values(collected, current.operands[:1], scope, active_when, ranges)
        operation_key = (scope.structural_scope, id(current))
        cached_name = while_trip_count_names.get(operation_key)
        trip_count_name = (
            cached_name[1]
            if cached_name is not None and cached_name[0] is current
            else "|while|"
        )
        child, trip_count = build_while_scope(
            current,
            scope,
            trip_count_name=trip_count_name,
        )
        _visit_operations(
            current.operations,
            child,
            _and_conditions(
                active_when,
                cast(Boolean, sp.Gt(trip_count, _ZERO)),
            ),
            ranges,
            unprojectable_symbols,
            collected,
            while_trip_count_names,
        )
        return
    _add_values(
        collected,
        _operation_classical_dependency_inputs(current),
        scope,
        active_when,
        ranges,
    )


def _scheduling_classical_input_sources(
    operation: Operation,
    resolver: ExprResolver,
    while_trip_count_names: _WhileTripCountNames,
) -> dict[str, Boolean]:
    """Resolve guarded classical reads for one scheduling boundary.

    A structured operation contributes only the reads of regions that can
    execute. Compile-time branch predicates therefore guard reads inside their
    respective regions instead of turning every captured source into an
    unconditional dependency. Range-local predicates are projected over the
    finite iteration domain, so a read is retained exactly when at least one
    reachable iteration can execute it.

    Args:
        operation (Operation): Operation whose scheduling reads are needed.
        resolver (ExprResolver): Resolver for the enclosing scope.
        while_trip_count_names (_WhileTripCountNames): Read-only names reserved
            for while-loop trip counts in the current interpretation run.

    Returns:
        dict[str, Boolean]: Classical source tokens and their execution guards
        at the operation boundary.
    """
    collected: dict[str, Boolean] = {}
    _visit(
        operation,
        resolver,
        sp.true,
        (),
        frozenset(),
        collected,
        while_trip_count_names,
    )
    return collected
