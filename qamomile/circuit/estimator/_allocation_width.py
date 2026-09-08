"""Compose static allocation identities and reusable width."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._constants import _ZERO
from qamomile.circuit.estimator._resource_base import ResourceExpr
from qamomile.circuit.estimator._resource_expressions import (
    _boolean_condition,
    _maximum_expr_over_range,
    _resource_expr,
)
from qamomile.circuit.estimator._resource_types import WidthResources
from qamomile.circuit.ir.operation.operation import Operation, QInitOperation

if TYPE_CHECKING:
    from qamomile.circuit.estimator._estimate import ResourceEstimate


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
) -> tuple[WidthResources, dict[str, ResourceExpr], Boolean]:
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
        tuple[WidthResources, dict[str, ResourceExpr], Boolean]: Maximum
        reusable width, maximized allocation sites, and the condition under
        which at least one conservative maximum may overestimate.
    """
    maximized_sites: dict[str, ResourceExpr] = {}
    conservative_guards: list[Boolean] = []
    for site, size in sites.items():
        maximum, conservative_when = _maximum_expr_over_range(
            size,
            loop_symbol,
            start,
            step,
            iterations,
        )
        maximized_sites[site] = maximum
        conservative_guards.append(conservative_when)

    anonymous = _anonymous_allocation_width(width, sites)
    maximum_anonymous, anonymous_conservative_when = _maximum_expr_over_range(
        anonymous,
        loop_symbol,
        start,
        step,
        iterations,
    )
    conservative_guards.append(anonymous_conservative_when)

    maxima: dict[str, ResourceExpr] = {}
    for field in dataclasses.fields(WidthResources):
        if field.name == "allocated_qubits":
            continue
        maximum, conservative_when = _maximum_expr_over_range(
            getattr(width, field.name),
            loop_symbol,
            start,
            step,
            iterations,
        )
        maxima[field.name] = maximum
        conservative_guards.append(conservative_when)
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
        _boolean_condition(sp.Or(*conservative_guards)),
    )
