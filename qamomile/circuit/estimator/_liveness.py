"""Compute live-owner state and peak logical width."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._call_liveness import (
    _destructive_input_owner_sizes,
    _quantum_owner_capacity,
    _quantum_result_owner_sizes,
)
from qamomile.circuit.estimator._constants import _ONE, _ZERO
from qamomile.circuit.estimator._quantum_values import _quantum_allocation_owner
from qamomile.circuit.estimator._resolver import ExprResolver
from qamomile.circuit.estimator._resource_base import (
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_expressions import (
    _boolean_condition,
    _is_structurally_nonnegative,
    _maximum_expr_over_range,
    _piecewise,
    _resource_max,
    _resource_max_many,
    _safe_simplify,
)
from qamomile.circuit.estimator._resource_types import WidthResources
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.operation.gate import (
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureQIntOperation,
    MeasureVectorOperation,
)
from qamomile.circuit.ir.operation.operation import Operation, QInitOperation
from qamomile.circuit.ir.value import Value

if TYPE_CHECKING:
    from qamomile.circuit.estimator._estimate import ResourceEstimate


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


@dataclasses.dataclass(frozen=True)
class _LivenessSummary:
    """Carry liveness width and the owners still live after a body.

    Args:
        width (WidthResources): Liveness-aware width of the evaluated body.
        final_live_by_owner (dict[str, ResourceExpr]): Remaining live qubits
            keyed by root allocation owner at the body boundary.
    """

    width: WidthResources
    final_live_by_owner: dict[str, ResourceExpr]


def _liveness_width(
    scheduled: Sequence[tuple[Operation, ResourceEstimate]],
    initial_allocations: Mapping[str, ResourceExpr],
    resolver: ExprResolver,
    *,
    allocation_owners_by_uuid: Mapping[str, str] | None = None,
) -> _LivenessSummary:
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
        _LivenessSummary: Width and authoritative final live-owner state.
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
                MeasureQIntOperation,
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
            if estimate._has_output_summary:
                # Boundary summaries describe both sides of the liveness
                # transfer. An empty input map is authoritative and means the
                # operation creates returned owners from no caller-owned
                # quantum input; falling back to syntactic merge/call inputs
                # would consume those freshly returned owners immediately.
                input_sizes = estimate._input_sizes
            elif estimate._input_sizes:
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
                        fallback_capacity = sp.Max(
                            previous,
                            consumed,
                            returned,
                        )
                        updated = (
                            remaining + returned
                            if estimate._has_output_summary
                            else sp.Min(
                                capacities.get(owner, fallback_capacity),
                                remaining + returned,
                            )
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
    return _LivenessSummary(
        width=WidthResources(
            allocated_qubits=allocated,
            clean_ancilla_qubits=clean,
            dirty_ancilla_qubits=dirty,
            peak_qubits=relative_peak,
        ),
        final_live_by_owner=dict(live),
    )


def _maximum_live_owner_sizes(
    summaries: Sequence[Mapping[str, ResourceExpr]],
) -> tuple[dict[str, ResourceExpr], Boolean]:
    """Retain the largest observed live size for every loop-local owner.

    A later concrete iteration may report a smaller size for the same static
    allocation site. Without an explicit inter-iteration release proof, using
    only that final report could silently discard qubits that were live after
    an earlier iteration. Taking the owner-wise maximum is order-independent
    and conservative.

    Args:
        summaries (Sequence[Mapping[str, ResourceExpr]]): Live-owner summaries
            produced after each concrete iteration.

    Returns:
        tuple[dict[str, ResourceExpr], Boolean]: Owner-wise maximum sizes and
        the condition under which the maximum retains anything not present in
        the final summary.
    """
    if not summaries:
        return {}, sp.false
    owners = sorted({owner for summary in summaries for owner in summary})
    maxima = {
        owner: _resource_max_many([summary.get(owner, _ZERO) for summary in summaries])
        for owner in owners
    }
    final = summaries[-1]
    retains_prior_when = _boolean_condition(
        sp.Or(
            *(
                sp.Gt(maximum, final.get(owner, _ZERO))
                for owner, maximum in maxima.items()
            )
        )
    )
    return maxima, retains_prior_when


def _maximum_live_owner_sizes_over_range(
    sizes: Mapping[str, ResourceExpr],
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> tuple[dict[str, ResourceExpr], Boolean, Boolean]:
    """Maximize live loop-local owners across a symbolic iteration range.

    Args:
        sizes (Mapping[str, ResourceExpr]): Per-iteration live-owner sizes.
        loop_symbol (sp.Symbol): Loop variable symbol.
        start (ResourceExpr): First loop value.
        step (ResourceExpr): Loop step.
        iterations (ResourceExpr): Number of executed iterations.

    Returns:
        tuple[dict[str, ResourceExpr], Boolean, Boolean]: Owner-wise maximum
        sizes, the condition under which a symbolic maximum may overestimate,
        and the condition under which a maximum retains owner width from
        before the final iteration.
    """
    maxima: dict[str, ResourceExpr] = {}
    conservative_guards: list[Boolean] = []
    retention_guards: list[Boolean] = []
    for owner in sorted(sizes):
        size = sizes[owner]
        maximum, conservative_when = _maximum_expr_over_range(
            size,
            loop_symbol,
            start,
            step,
            iterations,
        )
        maxima[owner] = maximum
        conservative_guards.append(conservative_when)
        final = _piecewise(
            cast(
                ResourceExpr,
                size.subs(loop_symbol, start + (iterations - _ONE) * step),
            ),
            _ZERO,
            sp.Gt(iterations, _ZERO),
        )
        maximum_matches_final = _safe_simplify(maximum - final) == _ZERO
        if (
            not maximum_matches_final
            and conservative_when is sp.false
            and loop_symbol in size.free_symbols
        ):
            index = sp.Dummy("live_owner_index", integer=True, nonnegative=True)
            transformed = cast(
                sp.Expr,
                size.subs(loop_symbol, start + step * index),
            )
            try:
                polynomial = sp.Poly(transformed, index)
            except sp.PolynomialError:
                polynomial = None
            maximum_matches_final = (
                polynomial is not None
                and polynomial.degree() <= 1
                and _is_structurally_nonnegative(polynomial.coeff_monomial(index))
            )
        retention_guards.append(
            sp.false
            if maximum_matches_final
            else _boolean_condition(sp.Gt(maximum, final))
        )
    return (
        maxima,
        _boolean_condition(sp.Or(*conservative_guards)),
        _boolean_condition(sp.Or(*retention_guards)),
    )
