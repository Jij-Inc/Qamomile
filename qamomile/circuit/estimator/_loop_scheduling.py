"""Prove and project loop-specific dependency scheduling."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import cast

import sympy as sp

from qamomile.circuit.estimator._constants import _ZERO
from qamomile.circuit.estimator._dependency_footprints import _quantum_wire_keys
from qamomile.circuit.estimator._dependency_indices import (
    _MAX_EXACT_LOOP_DISJOINTNESS_EXPANSION,
    _MAX_EXACT_LOOP_WIRE_EXPANSION,
    _WIRE_RANGE_OFFSET,
    WireKey,
    _normalize_wire_index,
    _OwnerWireIndices,
    _specialize_dependency_expression,
    _wire_index_relation,
    _WireRangeIndex,
    _WireRelation,
)
from qamomile.circuit.estimator._quantum_values import (
    _quantum_allocation_owner,
    _quantum_element_index_expression,
)
from qamomile.circuit.estimator._resolver import ExprResolver
from qamomile.circuit.estimator._resource_algebra import _conditional_depth
from qamomile.circuit.estimator._resource_base import (
    ResourceExpr,
    _is_concrete_integer,
)
from qamomile.circuit.estimator._resource_expressions import (
    _resource_max,
    _safe_simplify,
)
from qamomile.circuit.estimator._resource_types import DepthResources
from qamomile.circuit.estimator._scheduling import (
    _expressions_proven_equal_without_simplify,
)
from qamomile.circuit.estimator._scopes import _LocalBlock
from qamomile.circuit.ir.operation.classical_ops import (
    ReturnQuantumArrayElementOperation,
    StoreArrayElementOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    HasNestedOps,
    IfOperation,
)
from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.value import ArrayValue, Value


def _disjoint_concrete_loop_depth(
    operation: ForOperation,
    resolver: ExprResolver,
    body_depth: DepthResources,
    *,
    body_dependency_keys: frozenset[WireKey] | None = None,
    start: ResourceExpr,
    stop: ResourceExpr,
    step: ResourceExpr,
    loop_symbol: sp.Symbol,
    allocated_qubits: ResourceExpr,
    clean_ancillas: ResourceExpr,
    dirty_ancillas: ResourceExpr,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> DepthResources | None:
    """Parallelize concrete loop iterations with disjoint quantum footprints.

    This optimization is deliberately bounded: it proves pairwise-disjoint
    physical owner/index keys only for a small concrete loop. Whole registers,
    unresolved views, shared clean ancillas, large loops, or any overlapping
    element conservatively retain sequential loop depth.

    Args:
        operation (ForOperation): Loop whose independent body was summarized.
        resolver (ExprResolver): Resolver for the enclosing scope.
        body_depth (DepthResources): One symbolic body-iteration depth.
        body_dependency_keys (frozenset[WireKey] | None): Evaluated body
            footprint, including nested operation summaries when available.
            Defaults to ``None``.
        start (ResourceExpr): Inclusive Python-range start.
        stop (ResourceExpr): Exclusive Python-range stop.
        step (ResourceExpr): Python-range step.
        loop_symbol (sp.Symbol): Internal body loop-variable symbol.
        allocated_qubits (ResourceExpr): Body-local allocation demand reused
            between sequential iterations.
        clean_ancillas (ResourceExpr): Shared fallback ancilla demand.
        dirty_ancillas (ResourceExpr): Shared dirty-ancilla demand.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        DepthResources | None: Parallel critical-path depth when disjointness
            is proven, otherwise ``None``.
    """
    iterations = _bounded_concrete_loop_values(
        start,
        stop,
        step,
        limit=_MAX_EXACT_LOOP_DISJOINTNESS_EXPANSION,
        scalar_values=scalar_values,
        used_names=used_names,
    )
    if iterations is None or any(
        demand != _ZERO for demand in (allocated_qubits, clean_ancillas, dirty_ancillas)
    ):
        return None

    seen: dict[str, _OwnerWireIndices] = {}
    fields = tuple(field.name for field in dataclasses.fields(DepthResources))
    peaks: dict[str, ResourceExpr] = {field: _ZERO for field in fields}
    depth_expressions = {
        field: cast(sp.Expr, getattr(body_depth, field)) for field in fields
    }
    varying_depths = {
        field: expression
        for field, expression in depth_expressions.items()
        if loop_symbol in expression.free_symbols
    }
    if len(iterations) > 0:
        for field, expression in depth_expressions.items():
            if field in varying_depths:
                continue
            invariant_depth = _safe_simplify(cast(sp.Expr, expression.doit()))
            peaks[field] = sp.Max(_ZERO, invariant_depth)
    for loop_value in iterations:
        value = sp.Integer(loop_value)
        if operation.loop_var_value is None:
            return None
        child = resolver.child_scope(
            inner_block=_LocalBlock(operation.operations),
            extra_context={operation.loop_var_value.uuid: value},
            extra_loop_vars={operation.loop_var: value},
        )
        footprint: set[WireKey] = set()
        if body_dependency_keys is not None:
            for key in body_dependency_keys:
                footprint.update(
                    _specialize_loop_dependency_key(
                        key,
                        loop_symbol,
                        value,
                    )
                )
        else:
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
        if not _record_disjoint_wire_footprint(seen, footprint):
            return None
        for field, expression in varying_depths.items():
            iteration_depth = _safe_simplify(
                cast(sp.Expr, expression.subs(loop_symbol, value).doit())
            )
            peaks[field] = sp.Max(peaks[field], iteration_depth)
    return DepthResources(**peaks)


def _bounded_concrete_loop_values(
    start: ResourceExpr,
    stop: ResourceExpr,
    step: ResourceExpr,
    *,
    limit: int = _MAX_EXACT_LOOP_WIRE_EXPANSION,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> range | None:
    """Return a safely bounded concrete Python range for wire analysis.

    Args:
        start (ResourceExpr): Inclusive Python-range start.
        stop (ResourceExpr): Exclusive Python-range stop.
        step (ResourceExpr): Python-range step.
        limit (int): Maximum iteration count to enumerate. Defaults to the
            compact dependency-metadata expansion budget.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        range | None: Concrete range within the exact-expansion budget, or
            ``None`` when a bound is unresolved, malformed, or too large.
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
    if concrete_step == 0:
        return None
    iterations = range(concrete_start, concrete_stop, concrete_step)
    if len(iterations[: limit + 1]) > limit:
        return None
    return iterations


def _concrete_loop_dependency_completion(
    completion: Mapping[WireKey, ResourceExpr] | None,
    loop_symbol: sp.Symbol,
    *,
    start: ResourceExpr,
    stop: ResourceExpr,
    step: ResourceExpr,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> dict[WireKey, ResourceExpr] | None:
    """Project exact local wire completion through a small concrete loop.

    A disjoint loop runs every iteration from the same logical start layer.
    Its caller-visible completion therefore comes from the corresponding
    iteration-local completion, not from the sequential sum used before the
    disjoint-depth proof.

    Args:
        completion (Mapping[WireKey, ResourceExpr] | None): One-iteration
            body completion map.
        loop_symbol (sp.Symbol): Internal loop-variable symbol.
        start (ResourceExpr): Inclusive Python-range start.
        stop (ResourceExpr): Exclusive Python-range stop.
        step (ResourceExpr): Python-range step.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        dict[WireKey, ResourceExpr] | None: Exact projected completion map, or
            ``None`` when exact bounded projection is unavailable.
    """
    if completion is None:
        return None
    iterations = _bounded_concrete_loop_values(
        start,
        stop,
        step,
        scalar_values=scalar_values,
        used_names=used_names,
    )
    if iterations is None:
        return None
    projected: dict[WireKey, ResourceExpr] = {}
    for loop_value in iterations:
        value = sp.Integer(loop_value)
        for key, depth in completion.items():
            local_depth = _safe_simplify(
                cast(
                    ResourceExpr,
                    depth.subs(
                        loop_symbol,
                        value,
                        simultaneous=True,
                    ).doit(),
                )
            )
            for specialized_key in _specialize_loop_dependency_key(
                key,
                loop_symbol,
                value,
            ):
                projected[specialized_key] = _resource_max(
                    projected.get(specialized_key, _ZERO),
                    local_depth,
                )
    return projected


def _uniform_parallel_loop_dependency_completion(
    completion: Mapping[WireKey, ResourceExpr] | None,
    *,
    body_depth: ResourceExpr,
    projected_keys: frozenset[WireKey] | None,
    parallel_depth: ResourceExpr,
    loop_symbol: sp.Symbol,
) -> dict[WireKey, ResourceExpr] | None:
    """Recover compact completion for an invariant uniform loop body.

    This path avoids enumerating a large or symbolic disjoint loop. It is
    exact only when every body-visible wire finishes at the body's aggregate
    depth and that depth does not vary with the loop induction value.

    Args:
        completion (Mapping[WireKey, ResourceExpr] | None): One-iteration
            body completion map.
        body_depth (ResourceExpr): One-iteration aggregate depth.
        projected_keys (frozenset[WireKey] | None): Caller-visible loop
            footprint after range projection.
        parallel_depth (ResourceExpr): Aggregate depth after the disjoint-loop
            proof.
        loop_symbol (sp.Symbol): Internal loop-variable symbol.

    Returns:
        dict[WireKey, ResourceExpr] | None: Compact exact completion map, or
            ``None`` when uniformity cannot be proven.
    """
    if completion is None or projected_keys is None:
        return None
    if loop_symbol in body_depth.free_symbols:
        return None
    if any(
        not _expressions_proven_equal_without_simplify(depth, body_depth)
        for depth in completion.values()
    ):
        return None
    return {key: parallel_depth for key in projected_keys}


def _specialize_loop_dependency_key(
    key: WireKey,
    loop_symbol: sp.Symbol,
    loop_value: sp.Integer,
) -> set[WireKey]:
    """Specialize and, when finite, expand one loop-body dependency key.

    Args:
        key (WireKey): Body-scoped dependency address.
        loop_symbol (sp.Symbol): Loop-local symbol being bound.
        loop_value (sp.Integer): Concrete value for one loop iteration.

    Returns:
        set[WireKey]: Specialized scalar or symbolic-range addresses.
    """
    owner, index = key
    if isinstance(index, sp.Expr):
        return {
            (
                owner,
                _normalize_wire_index(
                    cast(
                        ResourceExpr,
                        index.subs(
                            loop_symbol,
                            loop_value,
                            simultaneous=True,
                        ),
                    )
                ),
            )
        }
    if not isinstance(index, _WireRangeIndex):
        return {key}
    specialized = index.mapped(
        lambda expression: expression.subs(
            loop_symbol,
            loop_value,
            simultaneous=True,
        )
    )
    iterations = specialized.iterations
    if (
        not iterations.is_number
        or not _is_concrete_integer(iterations)
        or not 0 <= iterations <= _MAX_EXACT_LOOP_WIRE_EXPANSION
    ):
        return {(owner, specialized)}
    return {
        (
            owner,
            _normalize_wire_index(
                cast(
                    ResourceExpr,
                    specialized.index_at_offset.subs(
                        _WIRE_RANGE_OFFSET,
                        offset,
                        simultaneous=True,
                    ),
                )
            ),
        )
        for offset in range(int(iterations))
    }


def _record_disjoint_wire_footprint(
    seen: dict[str, _OwnerWireIndices],
    footprint: set[WireKey],
) -> bool:
    """Record one footprint only when it is disjoint from all prior wires.

    An owner-wide ``None`` index aliases every scalar index for the same
    allocation owner. Scalar indices are retained in per-owner sets and
    compared with the same symbolic relation used by the dependency scheduler.

    Args:
        seen (dict[str, _OwnerWireIndices]): Previously recorded addresses
            indexed by allocation owner.
        footprint (set[WireKey]): Candidate physical wire keys.

    Returns:
        bool: Whether the candidate was disjoint and has been recorded.
    """
    for owner, index in footprint:
        owner_indices = seen.get(owner)
        if owner_indices is None:
            continue
        if any(
            _wire_index_relation(index, previous) is not _WireRelation.DISJOINT
            for previous in owner_indices.candidates(index)
        ):
            return False
    for owner, index in footprint:
        seen.setdefault(owner, _OwnerWireIndices()).add(index)
    return True


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


def _dependency_keys_depend_on_symbol(
    keys: frozenset[WireKey] | None,
    symbol: sp.Symbol,
) -> bool:
    """Return whether an evaluated footprint still depends on a local symbol.

    Args:
        keys (frozenset[WireKey] | None): Evaluated dependency footprint, or
            ``None`` when no precise footprint is available.
        symbol (sp.Symbol): Loop-local symbol to find.

    Returns:
        bool: Whether a scalar address or symbolic range uses ``symbol``.
    """
    if keys is None:
        return False
    for _owner, index in keys:
        if isinstance(index, _WireRangeIndex):
            if symbol in (
                index.index_at_offset.free_symbols | index.iterations.free_symbols
            ):
                return True
        elif isinstance(index, sp.Expr) and symbol in index.free_symbols:
            return True
    return False


def _symbolic_disjoint_loop_depth(
    operation: ForOperation,
    resolver: ExprResolver,
    body_depth: DepthResources,
    *,
    loop_symbol: sp.Symbol,
    iterations: ResourceExpr,
    allocated_qubits: ResourceExpr,
    clean_ancillas: ResourceExpr,
    dirty_ancillas: ResourceExpr,
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
        allocated_qubits (ResourceExpr): Body-local allocation demand reused
            between sequential iterations.
        clean_ancillas (ResourceExpr): Shared decomposition ancilla demand.
        dirty_ancillas (ResourceExpr): Shared dirty-ancilla demand.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            dependency values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        DepthResources | None: One-iteration depth guarded by a nonempty range,
            or ``None`` when injectivity cannot be proven.
    """
    if any(
        demand != _ZERO for demand in (allocated_qubits, clean_ancillas, dirty_ancillas)
    ) or any(
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
    pending_operations = list(reversed(operation.operations))
    while pending_operations:
        body_operation = pending_operations.pop()
        if isinstance(body_operation, ignored_operations):
            continue
        if isinstance(body_operation, HasNestedOps):
            if not isinstance(body_operation, IfOperation):
                return None
            for region in reversed(body_operation.nested_regions()):
                pending_operations.extend(reversed(region.operations))
            # The branch's own merge values only alias the physical values
            # already inspected in its regions. Including those aggregate
            # ArrayValues would turn an otherwise injective ``register[i]``
            # body into an owner-wide footprint and defeat the proof.
            continue
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
        representative = _normalize_wire_index(indices[0])
        if any(
            _wire_index_relation(
                _normalize_wire_index(index),
                representative,
            )
            is not _WireRelation.DEFINITE_OVERLAP
            for index in indices
        ):
            return None
        if not isinstance(representative, sp.Expr):
            return None
        slope = _safe_simplify(cast(ResourceExpr, sp.diff(representative, loop_symbol)))
        if loop_symbol in slope.free_symbols or slope.is_zero is not False:
            return None
    return _conditional_depth(
        body_depth,
        DepthResources.zero(),
        sp.Gt(iterations, _ZERO),
    )
