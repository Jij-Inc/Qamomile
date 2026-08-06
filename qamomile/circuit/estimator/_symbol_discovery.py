"""Discover symbolic expressions used by resource estimates and serialization."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import sympy as sp

from qamomile.circuit.estimator._resource_base import (
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_types import (
    ResourceTraceNode,
)
from qamomile.circuit.estimator._serialization import SymbolRegistry

if TYPE_CHECKING:
    from qamomile.circuit.estimator._estimate import ResourceEstimate


def _free_symbols(estimate: ResourceEstimate) -> set[sp.Symbol]:
    """Collect all free symbols from an estimate.

    Args:
        estimate (ResourceEstimate): Estimate to inspect.

    Returns:
        set[sp.Symbol]: Free symbols used by metrics, constraints, metadata,
            or scheduler state.
    """
    symbols: set[sp.Symbol] = set()
    for expr in _all_exprs(estimate):
        symbols.update(cast(set[sp.Symbol], sp.sympify(expr).free_symbols))
    for constraint in estimate._constraints:
        constraint_symbols = set(sp.sympify(constraint.expression).free_symbols)
        constraint_symbols.update(sp.sympify(constraint.active_when).free_symbols)
        if constraint.expected is not None:
            constraint_symbols.update(sp.sympify(constraint.expected).free_symbols)
        bound_symbols = {loop_range.symbol for loop_range in constraint.ranges}
        for loop_range in constraint.ranges:
            constraint_symbols.update(sp.sympify(loop_range.start).free_symbols)
            constraint_symbols.update(sp.sympify(loop_range.step).free_symbols)
            constraint_symbols.update(sp.sympify(loop_range.iterations).free_symbols)
        symbols.update(cast(set[sp.Symbol], constraint_symbols - bound_symbols))
    for fact in (
        *(estimate._guarded_assumptions or ()),
        *(estimate._guarded_derivations or ()),
        *(estimate._guarded_qualities or ()),
        *(estimate._guarded_approximations or ()),
    ):
        symbols.update(cast(set[sp.Symbol], sp.sympify(fact.active_when).free_symbols))
    symbols.update(
        cast(
            set[sp.Symbol],
            sp.sympify(estimate._global_barrier_condition).free_symbols,
        )
    )
    return symbols


def _trace_guard_free_symbols(
    trace: ResourceTraceNode | None,
) -> set[sp.Symbol]:
    """Collect symbols used only by explanation-node activity guards.

    Trace-only symbols do not become public substitution parameters, but an
    internal runtime outcome or unresolved carry must still be rejected if it
    would escape through :meth:`ResourceEstimate.explain` or serialization.

    Args:
        trace (ResourceTraceNode | None): Optional explanation tree root.

    Returns:
        set[sp.Symbol]: Free symbols used by trace activity guards.
    """
    symbols: set[sp.Symbol] = set()
    pending = [trace] if trace is not None else []
    while pending:
        node = pending.pop()
        symbols.update(
            cast(
                set[sp.Symbol],
                sp.sympify(node.active_when).free_symbols,
            )
        )
        pending.extend(node.children)
    return symbols


def _serialization_expressions(
    estimate: ResourceEstimate,
) -> list[sp.Basic | int | float]:
    """Return expressions sharing the estimate's public symbol namespace.

    Metric expressions, structural requirements, quantified range variables,
    guarded metadata, and explanation guards must use one identity registry.
    Otherwise two symbols that collide only across separate fields could
    receive the same public name and make the combined payload ambiguous.

    Args:
        estimate (ResourceEstimate): Estimate to serialize.

    Returns:
        list[sp.Basic | int | float]: Expressions in deterministic payload
            encounter order.
    """
    expressions: list[sp.Basic | int | float] = list(_all_exprs(estimate))
    for constraint in estimate._constraints:
        expressions.append(constraint.expression)
        expressions.append(constraint.active_when)
        if constraint.expected is not None:
            expressions.append(constraint.expected)
        for loop_range in constraint.ranges:
            expressions.extend(
                (
                    loop_range.symbol,
                    loop_range.start,
                    loop_range.step,
                    loop_range.iterations,
                )
            )
    expressions.extend(
        fact.active_when
        for fact in (
            *(estimate._guarded_assumptions or ()),
            *(estimate._guarded_derivations or ()),
            *(estimate._guarded_qualities or ()),
            *(estimate._guarded_approximations or ()),
        )
    )
    expressions.append(estimate._global_barrier_condition)
    pending_trace_nodes = [estimate.trace] if estimate.trace is not None else []
    while pending_trace_nodes:
        trace_node = pending_trace_nodes.pop()
        expressions.append(trace_node.active_when)
        pending_trace_nodes.extend(reversed(trace_node.children))
    return expressions


def _serialization_registry(estimate: ResourceEstimate) -> SymbolRegistry:
    """Build the shared public symbol registry for an estimate.

    Args:
        estimate (ResourceEstimate): Estimate whose symbols need names.

    Returns:
        SymbolRegistry: Deterministic identity-to-public-name mapping.
    """
    return SymbolRegistry.from_expressions(
        _serialization_expressions(estimate),
        estimate._symbol_aliases,
    )


def _all_exprs(estimate: ResourceEstimate) -> list[ResourceExpr]:
    """Return every symbolic expression in an estimate.

    Args:
        estimate (ResourceEstimate): Estimate to inspect.

    Returns:
        list[ResourceExpr]: Metric expressions.
    """
    return [
        estimate.width.input_qubits,
        estimate.width.allocated_qubits,
        estimate.width.clean_ancilla_qubits,
        estimate.width.dirty_ancilla_qubits,
        estimate.width.peak_qubits,
        estimate.gates.total,
        estimate.gates.single_qubit,
        estimate.gates.two_qubit,
        estimate.gates.multi_qubit,
        estimate.gates.clifford,
        estimate.gates.rotation,
        estimate.gates.t,
        estimate.gates.toffoli,
        estimate.gates.non_clifford,
        estimate.measurements.total,
        estimate.resets.total,
        estimate.depth.depth,
        estimate.depth.clifford_depth,
        estimate.depth.rotation_depth,
        estimate.depth.t_depth,
        estimate.depth.toffoli_depth,
        estimate.depth.non_clifford_depth,
        estimate.depth.measurement_depth,
        estimate.depth.gate_depth,
        estimate.depth.reset_depth,
        *estimate.calls.calls_by_name.values(),
        *estimate.calls.queries_by_name.values(),
        *estimate._output_sizes.values(),
        *estimate._input_sizes.values(),
    ]


def _collect_parameters(
    estimate: ResourceEstimate,
    registry: SymbolRegistry | None = None,
) -> dict[str, sp.Symbol]:
    """Collect symbolic parameters from an estimate.

    Args:
        estimate (ResourceEstimate): Estimate to inspect.
        registry (SymbolRegistry | None): Shared estimate registry. Defaults to
            a newly derived registry.

    Returns:
        dict[str, sp.Symbol]: Symbol map keyed by unique public name.
    """
    active_registry = registry or _serialization_registry(estimate)
    return {
        active_registry.name(symbol): symbol
        for symbol in sorted(
            _free_symbols(estimate),
            key=active_registry.name,
        )
    }
