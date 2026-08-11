"""Prove and project loop-specific dependency scheduling."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import cast

import sympy as sp

from qamomile.circuit.estimator._constants import _ONE, _ZERO
from qamomile.circuit.estimator._dependency_footprints import _quantum_wire_keys
from qamomile.circuit.estimator._dependency_indices import (
    _MAX_EXACT_LOOP_DISJOINTNESS_EXPANSION,
    _MAX_EXACT_LOOP_WIRE_EXPANSION,
    _WIRE_RANGE_OFFSET,
    WireKey,
    _normalize_wire_index,
    _OwnerWireIndices,
    _specialize_dependency_expression,
    _symbolic_wire_range_index,
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
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp
from qamomile.circuit.ir.operation.classical_ops import (
    ReturnQuantumArrayElementOperation,
    StoreArrayElementOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    HasNestedOps,
    IfOperation,
)
from qamomile.circuit.ir.operation.gate import GateOperation
from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.value import ArrayValue, Value


def _scalar_quantum_address(
    value: Value,
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> WireKey | None:
    """Resolve one scalar quantum value to its root physical address.

    Args:
        value (Value): Scalar quantum input or result value.
        resolver (ExprResolver): Resolver scoped to the value's operations.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        WireKey | None: Root allocation owner and normalized scalar index, or
        ``None`` when the value is not a resolvable array element.
    """
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
    return _quantum_allocation_owner(value), _normalize_wire_index(index)


def _single_two_qubit_gate_addresses(
    operations: list[object],
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> tuple[GateOperation, tuple[WireKey, WireKey]] | None:
    """Return the physical addresses of one self-consistent two-qubit gate.

    Args:
        operations (list[object]): Candidate operation list.
        resolver (ExprResolver): Resolver scoped to ``operations``.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        tuple[GateOperation, tuple[WireKey, WireKey]] | None: Gate and input
        addresses when the list contains exactly one two-qubit gate whose
        results preserve those addresses, otherwise ``None``.
    """
    if len(operations) != 1 or not isinstance(operations[0], GateOperation):
        return None
    gate = operations[0]
    if len(gate.qubit_operands) != 2 or len(gate.results) != 2:
        return None
    input_addresses = tuple(
        _scalar_quantum_address(
            value,
            resolver,
            scalar_values=scalar_values,
            used_names=used_names,
        )
        for value in gate.qubit_operands
    )
    result_addresses = tuple(
        _scalar_quantum_address(
            value,
            resolver,
            scalar_values=scalar_values,
            used_names=used_names,
        )
        for value in gate.results
    )
    if any(address is None for address in (*input_addresses, *result_addresses)):
        return None
    inputs = cast(tuple[WireKey, WireKey], input_addresses)
    results = cast(tuple[WireKey, WireKey], result_addresses)
    if any(
        input_owner != result_owner
        or _wire_index_relation(input_index, result_index)
        is not _WireRelation.DEFINITE_OVERLAP
        for (input_owner, input_index), (result_owner, result_index) in zip(
            inputs,
            results,
            strict=True,
        )
    ):
        return None
    return gate, inputs


def _depth_is_one_gate_per_active_field(depth: DepthResources) -> bool:
    """Return whether every active depth field represents one gate layer.

    Args:
        depth (DepthResources): Candidate one-iteration depth profile.

    Returns:
        bool: Whether every field is structurally zero or one and at least one
        field is active.
    """
    values = tuple(
        cast(ResourceExpr, getattr(depth, field.name))
        for field in dataclasses.fields(DepthResources)
    )
    return any(value != _ZERO for value in values) and all(
        value in (_ZERO, _ONE) for value in values
    )


def _symbolic_shared_anchor_loop_entry_keys(
    operation: ForOperation,
    resolver: ExprResolver,
    body_depth: DepthResources,
    *,
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
    allocated_qubits: ResourceExpr,
    clean_ancillas: ResourceExpr,
    dirty_ancillas: ResourceExpr,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> frozenset[WireKey] | None:
    """Prove exact sequential depth for a one-gate shared-anchor loop.

    Every iteration must apply one two-qubit gate, preserve both physical
    addresses, and reuse one loop-invariant scalar wire. That shared wire
    totally orders all active iterations, so the ordinary sequential depth is
    exact even though the second address remains symbolic.

    Args:
        operation (ForOperation): Candidate flat range loop.
        resolver (ExprResolver): Resolver scoped to the loop body.
        body_depth (DepthResources): One-iteration depth profile.
        loop_symbol (sp.Symbol): Internal induction symbol.
        start (ResourceExpr): Inclusive range start.
        step (ResourceExpr): Python-range step.
        iterations (ResourceExpr): Number of active iterations.
        allocated_qubits (ResourceExpr): Body-local allocation demand.
        clean_ancillas (ResourceExpr): Body clean-ancilla demand.
        dirty_ancillas (ResourceExpr): Body dirty-ancilla demand.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        frozenset[WireKey] | None: Caller-independent input addresses whose
        readiness must be synchronized, or ``None`` when the proof fails.
    """
    if (
        len(operation.operands) not in (2, 3)
        or operation.region_args
        or operation.loop_carried_rebinds
        or any(
            demand != _ZERO
            for demand in (allocated_qubits, clean_ancillas, dirty_ancillas)
        )
        or not _depth_is_one_gate_per_active_field(body_depth)
    ):
        return None
    resolved = _single_two_qubit_gate_addresses(
        cast(list[object], operation.operations),
        resolver,
        scalar_values=scalar_values,
        used_names=used_names,
    )
    if resolved is None:
        return None
    _gate, addresses = resolved
    candidates: list[tuple[WireKey, WireKey]] = []
    for anchor, moving in (addresses, tuple(reversed(addresses))):
        anchor_index = anchor[1]
        moving_index = moving[1]
        if (
            isinstance(moving_index, sp.Expr)
            and not isinstance(anchor_index, _WireRangeIndex)
            and anchor_index is not None
            and (
                not isinstance(anchor_index, sp.Expr)
                or loop_symbol not in anchor_index.free_symbols
            )
            and loop_symbol in moving_index.free_symbols
        ):
            candidates.append((anchor, moving))
    if len(candidates) != 1:
        return None
    anchor, (moving_owner, moving_index) = candidates[0]
    assert isinstance(moving_index, sp.Expr)
    return frozenset(
        {
            anchor,
            (
                moving_owner,
                _symbolic_wire_range_index(
                    moving_index,
                    loop_symbol,
                    start=start,
                    step=step,
                    iterations=iterations,
                ),
            ),
        }
    )


def _symbolic_triangular_pair_loop_depth(
    operation: ForOperation,
    resolver: ExprResolver,
    body_depth: DepthResources,
    *,
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    stop: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
    allocated_qubits: ResourceExpr,
    clean_ancillas: ResourceExpr,
    dirty_ancillas: ResourceExpr,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> tuple[DepthResources, frozenset[WireKey]] | None:
    """Prove the source-order ASAP depth of a triangular all-pairs loop.

    The accepted shape is ``for i in range(s, t): for j in range(i + 1, t)``
    with one invariant two-qubit gate on ``q[f(i)]`` and ``q[f(j)]``. For
    ``K`` outer iterations, the per-wire source dependencies place pair
    ``(i, j)`` at relative layer ``i + j`` and the critical path has
    ``Max(0, 2*K - 3)`` layers.

    Args:
        operation (ForOperation): Candidate outer range loop.
        resolver (ExprResolver): Resolver scoped to the outer body.
        body_depth (DepthResources): One outer iteration's depth profile.
        loop_symbol (sp.Symbol): Internal outer induction symbol.
        start (ResourceExpr): Inclusive outer start.
        stop (ResourceExpr): Exclusive outer stop.
        step (ResourceExpr): Outer step.
        iterations (ResourceExpr): Exact outer trip count.
        allocated_qubits (ResourceExpr): Body-local allocation demand.
        clean_ancillas (ResourceExpr): Body clean-ancilla demand.
        dirty_ancillas (ResourceExpr): Body dirty-ancilla demand.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used input
            names. Defaults to ``None``.

    Returns:
        tuple[DepthResources, frozenset[WireKey]] | None: Exact aggregate depth
        and synchronized-entry address range, or ``None`` when any proof
        premise fails.
    """
    if (
        len(operation.operands) not in (2, 3)
        or operation.region_args
        or operation.loop_carried_rebinds
        or _safe_simplify(cast(ResourceExpr, step - _ONE)) != _ZERO
        or any(
            demand != _ZERO
            for demand in (allocated_qubits, clean_ancillas, dirty_ancillas)
        )
    ):
        return None
    nested = [item for item in operation.operations if isinstance(item, ForOperation)]
    if len(nested) != 1 or any(
        item is not nested[0] and not isinstance(item, BinOp)
        for item in operation.operations
    ):
        return None
    inner = nested[0]
    if inner.loop_var_value is None or inner.region_args or inner.loop_carried_rebinds:
        return None
    if len(inner.operands) not in (2, 3):
        return None
    inner_start = resolver.resolve(inner.operands[0])
    inner_stop = resolver.resolve(inner.operands[1])
    inner_step = (
        resolver.resolve(inner.operands[2]) if len(inner.operands) >= 3 else _ONE
    )
    if any(
        _safe_simplify(cast(ResourceExpr, difference)) != _ZERO
        for difference in (
            inner_start - (loop_symbol + _ONE),
            inner_stop - stop,
            inner_step - _ONE,
        )
    ):
        return None
    inner_symbol = sp.Dummy(
        "triangular_inner",
        integer=True,
        nonnegative=True,
    )
    inner_resolver = resolver.child_scope(
        inner_block=_LocalBlock(inner.operations),
        extra_context={inner.loop_var_value.uuid: inner_symbol},
        extra_loop_vars={inner.loop_var: inner_symbol},
    )
    resolved = _single_two_qubit_gate_addresses(
        cast(list[object], inner.operations),
        inner_resolver,
        scalar_values=scalar_values,
        used_names=used_names,
    )
    if resolved is None:
        return None
    _gate, addresses = resolved
    if addresses[0][0] != addresses[1][0]:
        return None
    oriented: tuple[sp.Expr, sp.Expr] | None = None
    for outer_address, inner_address in (addresses, tuple(reversed(addresses))):
        outer_index = outer_address[1]
        inner_index = inner_address[1]
        if not isinstance(outer_index, sp.Expr) or not isinstance(inner_index, sp.Expr):
            continue
        if (
            loop_symbol not in outer_index.free_symbols
            or inner_symbol in outer_index.free_symbols
            or inner_symbol not in inner_index.free_symbols
            or loop_symbol in inner_index.free_symbols
        ):
            continue
        mapped_outer = cast(
            ResourceExpr,
            outer_index.subs(loop_symbol, inner_symbol, simultaneous=True),
        )
        if _safe_simplify(cast(ResourceExpr, mapped_outer - inner_index)) != _ZERO:
            continue
        slope = _safe_simplify(cast(ResourceExpr, sp.diff(outer_index, loop_symbol)))
        if loop_symbol in slope.free_symbols or slope.is_zero is not False:
            continue
        oriented = outer_index, inner_index
        break
    if oriented is None:
        return None
    outer_index, _inner_index = oriented
    inner_iterations = cast(
        ResourceExpr,
        sp.Max(_ZERO, inner_stop - inner_start),
    )
    depth_fields: dict[str, ResourceExpr] = {}
    active_fields = 0
    for field in dataclasses.fields(DepthResources):
        value = cast(ResourceExpr, getattr(body_depth, field.name))
        if value == _ZERO:
            depth_fields[field.name] = _ZERO
            continue
        if _safe_simplify(cast(ResourceExpr, value - inner_iterations)) != _ZERO:
            return None
        active_fields += 1
        depth_fields[field.name] = cast(
            ResourceExpr,
            sp.Max(_ZERO, 2 * iterations - 3),
        )
    if active_fields == 0:
        return None
    entry_key: WireKey = (
        addresses[0][0],
        _symbolic_wire_range_index(
            outer_index,
            loop_symbol,
            start=start,
            step=step,
            iterations=iterations,
        ),
    )
    return DepthResources(**depth_fields), frozenset((entry_key,))


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
