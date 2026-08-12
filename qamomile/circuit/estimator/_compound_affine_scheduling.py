"""Prove exact schedules for adjacent compound affine loop regions."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import cast

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._affine_loop_scheduling import (
    _analyze_accesses,
    _depth_is_one_gate_per_active_field,
    _LoopAxis,
    _project_accesses,
    _scalar_wire_access,
    _WireAccess,
)
from qamomile.circuit.estimator._constants import _ONE, _ZERO
from qamomile.circuit.estimator._dependency_indices import (
    WireKey,
    _specialize_dependency_expression,
)
from qamomile.circuit.estimator._gate_classification import (
    _classify_uncontrolled_gate,
    _serial_depth_from_gate_resources,
)
from qamomile.circuit.estimator._loop_executor import symbolic_iterations
from qamomile.circuit.estimator._resolver import ExprResolver, UnresolvedValueError
from qamomile.circuit.estimator._resource_base import ResourceExpr
from qamomile.circuit.estimator._resource_expressions import (
    _boolean_condition,
    _safe_simplify,
)
from qamomile.circuit.estimator._resource_types import DepthResources
from qamomile.circuit.estimator._scopes import build_for_loop_scope
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp
from qamomile.circuit.ir.operation.control_flow import ForOperation
from qamomile.circuit.ir.operation.gate import GateOperation
from qamomile.circuit.ir.operation.operation import Operation


@dataclasses.dataclass(frozen=True)
class _AffineTraversal:
    """Describe one canonical physical-wire traversal.

    Args:
        owner (str): Unique root allocation owner.
        start (ResourceExpr): Physical index at ordinal zero.
        step (ResourceExpr): Physical index stride per ordinal.
        iterations (ResourceExpr): Number of traversal elements.
        access (_WireAccess): Original outer-loop access used for projection.
        axis (_LoopAxis): Original Python-range axis.
    """

    owner: str
    start: ResourceExpr
    step: ResourceExpr
    iterations: ResourceExpr
    access: _WireAccess
    axis: _LoopAxis


@dataclasses.dataclass(frozen=True)
class _PairSweep:
    """Describe one exact triangular pair sweep and its diagonal gate.

    Args:
        traversal (_AffineTraversal): Canonical wire order.
        depth (DepthResources): Exact compound-core depth.
    """

    traversal: _AffineTraversal
    depth: DepthResources


@dataclasses.dataclass(frozen=True)
class _MirrorLayer:
    """Describe one exact reversal-pair layer.

    Args:
        traversal (_AffineTraversal): Canonical wire order shared with a sweep.
        profile (DepthResources): One mirror primitive's depth profile.
    """

    traversal: _AffineTraversal
    profile: DepthResources


@dataclasses.dataclass(frozen=True)
class _CompoundAffineSchedule:
    """Carry an exact schedule for two adjacent affine loop operations.

    Args:
        first_index (int): Inclusive index of the first grouped operation.
        stop_index (int): Exclusive index after the second grouped loop.
        component_indices (tuple[int, int]): Exact loop-operation indices whose
            private scheduling metadata is consumed by the proof.
        depth (DepthResources): Exact depth of the grouped operations.
        coverage (frozenset[WireKey]): Proven complete quantum footprint.
        active_when (Boolean): Guard under which multiple gate layers require
            synchronized external entry wires.
    """

    first_index: int
    stop_index: int
    component_indices: tuple[int, int]
    depth: DepthResources
    coverage: frozenset[WireKey]
    active_when: Boolean


def _loop_axis(
    operation: ForOperation,
    resolver: ExprResolver,
    *,
    scalar_values: Mapping[str, sp.Expr] | None,
    used_names: set[str] | None,
) -> tuple[ExprResolver, _LoopAxis] | None:
    """Resolve one ordinary range loop into an exact symbolic axis.

    Args:
        operation (ForOperation): Candidate loop operation.
        resolver (ExprResolver): Resolver for the enclosing region.
        scalar_values (Mapping[str, sp.Expr] | None): Optional concrete scalar
            values used consistently for bounds and accesses.
        used_names (set[str] | None): Optional set updated with used names.

    Returns:
        tuple[ExprResolver, _LoopAxis] | None: Body resolver and exact axis, or
        ``None`` when the loop shape or bounds cannot be resolved safely.
    """
    if (
        len(operation.operands) not in (2, 3)
        or operation.region_args
        or operation.loop_carried_rebinds
        or operation.loop_var_value is None
    ):
        return None
    try:
        child, start, stop, step, symbol = build_for_loop_scope(operation, resolver)
        start, stop, step = (
            _specialize_dependency_expression(bound, scalar_values, used_names)
            for bound in (start, stop, step)
        )
        iterations = symbolic_iterations(start, stop, step)
    except (UnresolvedValueError, ValueError, TypeError):
        return None
    if step.is_zero is not False:
        return None
    return child, _LoopAxis(
        symbol=symbol,
        start=start,
        stop=stop,
        step=step,
        iterations=cast(ResourceExpr, iterations),
    )


def _gate_depth_profile(gate: GateOperation) -> DepthResources | None:
    """Return a strict zero-or-one depth profile for one primitive gate.

    Args:
        gate (GateOperation): Candidate uncontrolled primitive gate.

    Returns:
        DepthResources | None: Classified unit-layer profile, or ``None`` if
        any active depth field is not exactly one.
    """
    if gate.gate_type is None:
        return None
    profile = _serial_depth_from_gate_resources(
        _classify_uncontrolled_gate(gate.gate_type.name.lower())
    )
    return profile if _depth_is_one_gate_per_active_field(profile) else None


def _expressions_equal(left: ResourceExpr, right: ResourceExpr) -> bool:
    """Return whether two resource expressions simplify to the same value.

    Args:
        left (ResourceExpr): Left expression.
        right (ResourceExpr): Right expression.

    Returns:
        bool: Whether the difference is proven to be zero.
    """
    return _safe_simplify(cast(ResourceExpr, left - right)) == _ZERO


def _access_expression_at_ordinals(
    access: _WireAccess,
    outer: _LoopAxis,
    outer_ordinal: sp.Symbol,
    *,
    inner: _LoopAxis | None = None,
    inner_ordinal: sp.Symbol | None = None,
) -> ResourceExpr | None:
    """Rewrite one physical index through canonical loop ordinals.

    Args:
        access (_WireAccess): Scalar physical access to rewrite.
        outer (_LoopAxis): Outer Python-range axis.
        outer_ordinal (sp.Symbol): Canonical outer ordinal.
        inner (_LoopAxis | None): Optional inner Python-range axis. Defaults to
            ``None``.
        inner_ordinal (sp.Symbol | None): Optional canonical inner ordinal.
            Defaults to ``None``.

    Returns:
        ResourceExpr | None: Binder-free ordinal expression, or ``None`` when
        the access is not a scalar array element.
    """
    if access.index is None:
        return None
    expression = access.index
    if inner is not None:
        if inner_ordinal is None:
            return None
        expression = cast(
            ResourceExpr,
            expression.subs(
                inner.symbol,
                inner.start + inner.step * inner_ordinal,
                simultaneous=True,
            ),
        )
    expression = cast(
        ResourceExpr,
        expression.subs(
            outer.symbol,
            outer.start + outer.step * outer_ordinal,
            simultaneous=True,
        ),
    )
    return _safe_simplify(expression)


def _traversal_from_outer_access(
    access: _WireAccess,
    axis: _LoopAxis,
) -> _AffineTraversal | None:
    """Normalize one outer-loop access to a physical ordinal traversal.

    Args:
        access (_WireAccess): Candidate outer-index access.
        axis (_LoopAxis): Outer range axis.

    Returns:
        _AffineTraversal | None: Exact nonconstant traversal, or ``None``.
    """
    ordinal = sp.Dummy("compound_ordinal", integer=True, nonnegative=True)
    expression = _access_expression_at_ordinals(access, axis, ordinal)
    if expression is None:
        return None
    try:
        step = _safe_simplify(cast(ResourceExpr, sp.diff(expression, ordinal)))
    except (TypeError, ValueError):
        return None
    start = _safe_simplify(
        cast(ResourceExpr, expression.subs(ordinal, _ZERO, simultaneous=True))
    )
    if (
        ordinal in step.free_symbols
        or ordinal in start.free_symbols
        or step.is_zero is not False
        or not _expressions_equal(expression, start + step * ordinal)
    ):
        return None
    return _AffineTraversal(
        owner=access.owner,
        start=start,
        step=step,
        iterations=axis.iterations,
        access=access,
        axis=axis,
    )


def _address_preserving_one_qubit_access(
    gate: GateOperation,
    resolver: ExprResolver,
    axis: _LoopAxis,
    *,
    owner_aliases: Mapping[str, frozenset[str]] | None,
    scalar_values: Mapping[str, sp.Expr] | None,
    used_names: set[str] | None,
) -> _WireAccess | None:
    """Return the preserved physical address of one unary primitive gate.

    Args:
        gate (GateOperation): Candidate one-qubit gate.
        resolver (ExprResolver): Resolver scoped to the gate.
        axis (_LoopAxis): Enclosing loop axis.
        owner_aliases (Mapping[str, frozenset[str]] | None): Conditional owner
            aliases.
        scalar_values (Mapping[str, sp.Expr] | None): Optional concrete scalar
            values.
        used_names (set[str] | None): Optional set updated with used names.

    Returns:
        _WireAccess | None: Preserved scalar address, or ``None``.
    """
    if len(gate.qubit_operands) != 1 or len(gate.results) != 1:
        return None
    before = _scalar_wire_access(
        gate.qubit_operands[0],
        resolver,
        symbols=(axis.symbol,),
        owner_aliases=owner_aliases,
        scalar_values=scalar_values,
        used_names=used_names,
    )
    after = _scalar_wire_access(
        gate.results[0],
        resolver,
        symbols=(axis.symbol,),
        owner_aliases=owner_aliases,
        scalar_values=scalar_values,
        used_names=used_names,
    )
    if (
        before is None
        or after is None
        or before.owner != after.owner
        or before.index is None
        or after.index is None
        or not _expressions_equal(before.index, after.index)
    ):
        return None
    return before


def _core_depth(
    iterations: ResourceExpr,
    diagonal: DepthResources,
    pair: DepthResources,
) -> DepthResources:
    """Build the exact field-wise depth of a diagonal triangular sweep.

    Args:
        iterations (ResourceExpr): Number of traversed wires.
        diagonal (DepthResources): One diagonal primitive profile.
        pair (DepthResources): One pair primitive profile.

    Returns:
        DepthResources: Exact critical-path depth for every depth field.
    """
    values: dict[str, ResourceExpr] = {}
    for field in dataclasses.fields(DepthResources):
        diagonal_active = getattr(diagonal, field.name) == _ONE
        pair_active = getattr(pair, field.name) == _ONE
        if diagonal_active and pair_active:
            value = sp.Max(_ZERO, 2 * iterations - _ONE)
        elif diagonal_active:
            value = iterations
        elif pair_active:
            value = sp.Max(_ZERO, 2 * iterations - 3)
        else:
            value = _ZERO
        values[field.name] = cast(ResourceExpr, value)
    return DepthResources(**values)


def _prove_pair_sweep(
    operation: ForOperation,
    resolver: ExprResolver,
    *,
    owner_aliases: Mapping[str, frozenset[str]] | None,
    scalar_values: Mapping[str, sp.Expr] | None,
    used_names: set[str] | None,
) -> tuple[_PairSweep, bool] | None:
    """Prove one suffix- or prefix-triangular pair sweep.

    Args:
        operation (ForOperation): Candidate outer sweep loop.
        resolver (ExprResolver): Enclosing region resolver.
        owner_aliases (Mapping[str, frozenset[str]] | None): Conditional owner
            aliases.
        scalar_values (Mapping[str, sp.Expr] | None): Optional concrete scalar
            values.
        used_names (set[str] | None): Optional set updated with used names.

    Returns:
        tuple[_PairSweep, bool] | None: Proven sweep and whether its diagonal
        gate precedes the pair loop, or ``None`` on any unsupported shape.
    """
    resolved = _loop_axis(
        operation,
        resolver,
        scalar_values=scalar_values,
        used_names=used_names,
    )
    if resolved is None:
        return None
    outer_resolver, outer = resolved
    if not _expressions_equal(abs(outer.step), _ONE):
        return None
    quantum_operations = [
        item
        for item in operation.operations
        if isinstance(item, (GateOperation, ForOperation))
    ]
    if len(quantum_operations) != 2 or not all(
        isinstance(item, (BinOp, GateOperation, ForOperation))
        for item in operation.operations
    ):
        return None
    first_operation, second_operation = quantum_operations
    if isinstance(first_operation, GateOperation) and isinstance(
        second_operation, ForOperation
    ):
        diagonal_before = True
        diagonal_gate = first_operation
        inner_operation = second_operation
    elif isinstance(first_operation, ForOperation) and isinstance(
        second_operation, GateOperation
    ):
        diagonal_before = False
        inner_operation = first_operation
        diagonal_gate = second_operation
    else:
        return None
    inner_resolved = _loop_axis(
        inner_operation,
        outer_resolver,
        scalar_values=scalar_values,
        used_names=used_names,
    )
    if inner_resolved is None or not all(
        isinstance(item, (BinOp, GateOperation)) for item in inner_operation.operations
    ):
        return None
    inner_resolver, inner = inner_resolved
    summary = _analyze_accesses(
        cast(Sequence[Operation], inner_operation.operations),
        inner_resolver,
        symbols=(outer.symbol, inner.symbol),
        owner_aliases=owner_aliases,
        scalar_values=scalar_values,
        used_names=used_names,
    )
    gate_inputs = None if summary is None else summary.gate_inputs
    if gate_inputs is None or gate_inputs[0].owner != gate_inputs[1].owner:
        return None
    diagonal_access = _address_preserving_one_qubit_access(
        diagonal_gate,
        outer_resolver,
        outer,
        owner_aliases=owner_aliases,
        scalar_values=scalar_values,
        used_names=used_names,
    )
    diagonal_profile = _gate_depth_profile(diagonal_gate)
    pair_gates = [
        item for item in inner_operation.operations if isinstance(item, GateOperation)
    ]
    pair_profile = _gate_depth_profile(pair_gates[0]) if len(pair_gates) == 1 else None
    if diagonal_access is None or diagonal_profile is None or pair_profile is None:
        return None
    outer_ordinal = sp.Dummy("compound_outer", integer=True, nonnegative=True)
    inner_ordinal = sp.Dummy("compound_inner", integer=True, nonnegative=True)
    traversal = _traversal_from_outer_access(diagonal_access, outer)
    if traversal is None:
        return None
    normalized_inner_iterations = _safe_simplify(
        cast(
            ResourceExpr,
            inner.iterations.subs(
                outer.symbol,
                outer.start + outer.step * outer_ordinal,
                simultaneous=True,
            ),
        )
    )
    expected_inner_iterations = symbolic_iterations(
        _ZERO,
        (
            traversal.iterations - outer_ordinal - _ONE
            if diagonal_before
            else outer_ordinal
        ),
        _ONE,
    )
    if not _expressions_equal(
        normalized_inner_iterations,
        cast(ResourceExpr, expected_inner_iterations),
    ):
        return None
    expected_outer = traversal.start + traversal.step * outer_ordinal
    expected_inner = (
        traversal.start + traversal.step * (outer_ordinal + inner_ordinal + _ONE)
        if diagonal_before
        else traversal.start + traversal.step * inner_ordinal
    )
    oriented = False
    for anchor, partner in (gate_inputs, tuple(reversed(gate_inputs))):
        if anchor.owner != traversal.owner or partner.owner != traversal.owner:
            continue
        anchor_expression = _access_expression_at_ordinals(
            anchor,
            outer,
            outer_ordinal,
            inner=inner,
            inner_ordinal=inner_ordinal,
        )
        partner_expression = _access_expression_at_ordinals(
            partner,
            outer,
            outer_ordinal,
            inner=inner,
            inner_ordinal=inner_ordinal,
        )
        if (
            anchor_expression is not None
            and partner_expression is not None
            and _expressions_equal(anchor_expression, expected_outer)
            and _expressions_equal(partner_expression, expected_inner)
        ):
            oriented = True
            break
    diagonal_expression = _access_expression_at_ordinals(
        diagonal_access,
        outer,
        outer_ordinal,
    )
    if (
        not oriented
        or diagonal_expression is None
        or not _expressions_equal(diagonal_expression, expected_outer)
    ):
        return None
    return (
        _PairSweep(
            traversal=traversal,
            depth=_core_depth(
                outer.iterations,
                diagonal_profile,
                pair_profile,
            ),
        ),
        diagonal_before,
    )


def _traversals_equal(left: _AffineTraversal, right: _AffineTraversal) -> bool:
    """Return whether two traversals enumerate the same ordered wire series.

    Args:
        left (_AffineTraversal): Left traversal.
        right (_AffineTraversal): Right traversal.

    Returns:
        bool: Whether owner, start, stride, and trip count all match exactly.
    """
    return (
        left.owner == right.owner
        and _expressions_equal(left.start, right.start)
        and _expressions_equal(left.step, right.step)
        and _expressions_equal(left.iterations, right.iterations)
    )


def _prove_mirror_layer(
    operation: ForOperation,
    resolver: ExprResolver,
    traversal: _AffineTraversal,
    *,
    owner_aliases: Mapping[str, frozenset[str]] | None,
    scalar_values: Mapping[str, sp.Expr] | None,
    used_names: set[str] | None,
) -> _MirrorLayer | None:
    """Prove one disjoint layer pairing opposite traversal endpoints.

    Args:
        operation (ForOperation): Candidate mirror loop.
        resolver (ExprResolver): Enclosing region resolver.
        traversal (_AffineTraversal): Pair-sweep traversal to reverse.
        owner_aliases (Mapping[str, frozenset[str]] | None): Conditional owner
            aliases.
        scalar_values (Mapping[str, sp.Expr] | None): Optional concrete scalar
            values.
        used_names (set[str] | None): Optional set updated with used names.

    Returns:
        _MirrorLayer | None: Exact mirror-layer proof, or ``None``.
    """
    resolved = _loop_axis(
        operation,
        resolver,
        scalar_values=scalar_values,
        used_names=used_names,
    )
    if resolved is None:
        return None
    body_resolver, axis = resolved
    if not all(
        isinstance(item, (BinOp, GateOperation)) for item in operation.operations
    ):
        return None
    summary = _analyze_accesses(
        cast(Sequence[Operation], operation.operations),
        body_resolver,
        symbols=(axis.symbol,),
        owner_aliases=owner_aliases,
        scalar_values=scalar_values,
        used_names=used_names,
    )
    gate_inputs = None if summary is None else summary.gate_inputs
    gates = [item for item in operation.operations if isinstance(item, GateOperation)]
    profile = _gate_depth_profile(gates[0]) if len(gates) == 1 else None
    if gate_inputs is None or profile is None:
        return None
    if not _expressions_equal(
        axis.iterations,
        cast(ResourceExpr, sp.floor(traversal.iterations / 2)),
    ):
        return None
    ordinal = sp.Dummy("compound_mirror", integer=True, nonnegative=True)
    expressions = [
        _access_expression_at_ordinals(access, axis, ordinal) for access in gate_inputs
    ]
    expected = (
        traversal.start + traversal.step * ordinal,
        traversal.start + traversal.step * (traversal.iterations - _ONE - ordinal),
    )
    if any(expression is None for expression in expressions):
        return None
    typed_expressions = cast(list[ResourceExpr], expressions)
    matches = any(
        _expressions_equal(typed_expressions[0], candidate[0])
        and _expressions_equal(typed_expressions[1], candidate[1])
        for candidate in (expected, tuple(reversed(expected)))
    )
    if not matches or any(access.owner != traversal.owner for access in gate_inputs):
        return None
    mirror_traversal = _AffineTraversal(
        owner=traversal.owner,
        start=traversal.start,
        step=traversal.step,
        iterations=traversal.iterations,
        access=traversal.access,
        axis=traversal.axis,
    )
    if not _traversals_equal(traversal, mirror_traversal):
        return None
    return _MirrorLayer(traversal=mirror_traversal, profile=profile)


def _compound_depth(
    sweep: _PairSweep,
    mirror: _MirrorLayer,
) -> DepthResources:
    """Add an exact mirror layer to an exact triangular sweep depth.

    Args:
        sweep (_PairSweep): Exact core sweep.
        mirror (_MirrorLayer): Exact endpoint-pair layer.

    Returns:
        DepthResources: Exact field-wise compound depth.
    """
    mirror_active = _boolean_condition(sp.Gt(sweep.traversal.iterations, _ONE))
    values: dict[str, ResourceExpr] = {}
    for field in dataclasses.fields(DepthResources):
        mirror_value = getattr(mirror.profile, field.name)
        contribution = cast(
            ResourceExpr,
            sp.Piecewise((mirror_value, mirror_active), (_ZERO, True)),
        )
        values[field.name] = _safe_simplify(
            cast(ResourceExpr, getattr(sweep.depth, field.name) + contribution)
        )
    return DepthResources(**values)


def _compound_affine_region_schedules(
    operations: Sequence[Operation],
    resolver: ExprResolver,
    *,
    owner_aliases: Mapping[str, frozenset[str]] | None = None,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> tuple[_CompoundAffineSchedule, ...]:
    """Find nonoverlapping exact pair-transform compounds in one region.

    The accepted shape is a triangular pair sweep with one diagonal primitive
    adjacent, up to zero-depth ``BinOp`` bound preparation, to a mirror-pair
    layer over the same canonical affine wire traversal. The proof is entirely
    structural and does not inspect callable names.

    Args:
        operations (Sequence[Operation]): Region operations in program order.
        resolver (ExprResolver): Resolver for the region.
        owner_aliases (Mapping[str, frozenset[str]] | None): Conditional owner
            aliases. Defaults to ``None``.
        scalar_values (Mapping[str, sp.Expr] | None): Optional concrete scalar
            values. Defaults to ``None``.
        used_names (set[str] | None): Optional set updated with used names.
            Defaults to ``None``.

    Returns:
        tuple[_CompoundAffineSchedule, ...]: Proven nonoverlapping schedules in
        program order. Unsupported regions return an empty tuple.
    """
    loop_indices = [
        index
        for index, operation in enumerate(operations)
        if isinstance(operation, ForOperation)
    ]
    schedules: list[_CompoundAffineSchedule] = []
    consumed_until = 0
    for first, second in zip(loop_indices, loop_indices[1:]):
        if first < consumed_until or any(
            not isinstance(operation, BinOp)
            for operation in operations[first + 1 : second]
        ):
            continue
        first_operation = cast(ForOperation, operations[first])
        second_operation = cast(ForOperation, operations[second])
        first_sweep = _prove_pair_sweep(
            first_operation,
            resolver,
            owner_aliases=owner_aliases,
            scalar_values=scalar_values,
            used_names=used_names,
        )
        second_sweep = _prove_pair_sweep(
            second_operation,
            resolver,
            owner_aliases=owner_aliases,
            scalar_values=scalar_values,
            used_names=used_names,
        )
        sweep: _PairSweep
        mirror: _MirrorLayer | None
        if first_sweep is not None and first_sweep[1]:
            sweep = first_sweep[0]
            mirror = _prove_mirror_layer(
                second_operation,
                resolver,
                sweep.traversal,
                owner_aliases=owner_aliases,
                scalar_values=scalar_values,
                used_names=used_names,
            )
        elif second_sweep is not None and not second_sweep[1]:
            sweep = second_sweep[0]
            mirror = _prove_mirror_layer(
                first_operation,
                resolver,
                sweep.traversal,
                owner_aliases=owner_aliases,
                scalar_values=scalar_values,
                used_names=used_names,
            )
        else:
            continue
        if mirror is None or not _traversals_equal(sweep.traversal, mirror.traversal):
            continue
        coverage = _project_accesses(
            (sweep.traversal.access,),
            sweep.traversal.axis,
        )
        schedules.append(
            _CompoundAffineSchedule(
                first_index=first,
                stop_index=second + 1,
                component_indices=(first, second),
                depth=_compound_depth(sweep, mirror),
                coverage=coverage,
                active_when=_boolean_condition(sp.Gt(sweep.traversal.iterations, _ONE)),
            )
        )
        consumed_until = second + 1
    return tuple(schedules)
