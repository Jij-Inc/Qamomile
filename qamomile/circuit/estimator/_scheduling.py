"""Schedule resource depth from per-operation dependency metadata."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._constants import _ONE, _ZERO
from qamomile.circuit.estimator._dependency_footprints import (
    _is_classical_dependency_key,
    _WireFootprint,
)
from qamomile.circuit.estimator._dependency_indices import (
    WireIndex,
    WireKey,
    _OwnerWireIndices,
    _specialize_dependency_expression,
    _wire_index_relation,
    _WireRelation,
)
from qamomile.circuit.estimator._resource_base import (
    EstimateQuality,
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_expressions import (
    _and_conditions,
    _boolean_condition,
    _ConditionIndicator,
    _expr,
    _resource_activity_condition,
    _resource_max,
    _resource_max_many,
)
from qamomile.circuit.estimator._resource_types import (
    DepthResources,
    ResourceAssumption,
)
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    GateOperation,
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
    ProjectOperation,
    ResetOperation,
)
from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import Operation, QInitOperation
from qamomile.circuit.ir.operation.select import SelectOperation

if TYPE_CHECKING:
    from qamomile.circuit.estimator._estimate import ResourceEstimate


def _estimate_has_nonzero_depth(estimate: ResourceEstimate) -> bool:
    """Return whether any depth metric is structurally nonzero.

    Args:
        estimate (ResourceEstimate): Estimate whose depth should be inspected.

    Returns:
        bool: Whether at least one depth field is not the exact zero
            expression.
    """
    return any(
        getattr(estimate.depth, field.name) != _ZERO
        for field in dataclasses.fields(DepthResources)
    )


def _estimate_depth_activity_condition(estimate: ResourceEstimate) -> Boolean:
    """Return when at least one declared depth field is active.

    Aggregate opaque costs may provide category depths without also providing
    ``depth``. Such a declaration still participates in dependency barriers
    for every category, including categories where its own duration is zero.

    Args:
        estimate (ResourceEstimate): Estimate whose depth fields define
            operation activity.

    Returns:
        Boolean: Symbolic condition under which any depth field is nonzero.
    """
    active: Boolean = sp.false
    for field in dataclasses.fields(DepthResources):
        field_active = _resource_activity_condition(
            cast(ResourceExpr, getattr(estimate.depth, field.name))
        )
        if field_active is sp.true:
            return sp.true
        if field_active is not sp.false:
            active = cast(Boolean, sp.Or(active, field_active))
    return active


def _scheduled_depth_activity_conditions(
    scheduled: Sequence[tuple[Operation, ResourceEstimate]],
) -> tuple[Boolean, ...]:
    """Compute depth-activity guards once for a scheduled operation list.

    Args:
        scheduled (Sequence[tuple[Operation, ResourceEstimate]]): Operations
            paired with their resource summaries in program order.

    Returns:
        tuple[Boolean, ...]: Activity guards aligned with ``scheduled``.
    """
    return tuple(
        _estimate_depth_activity_condition(estimate)
        for _operation, estimate in scheduled
    )


def _merge_dependency_keys(
    left: ResourceEstimate,
    right: ResourceEstimate,
) -> frozenset[WireKey] | None:
    """Merge proven dependency masks without erasing unknown footprints.

    ``None`` means that the enclosing operation must supply a conservative
    footprint, while an empty set means that the estimate has proven no
    caller-visible blocking wires. A structurally zero estimate is therefore
    the neutral element even when its default mask is ``None``.

    Args:
        left (ResourceEstimate): Left composition operand.
        right (ResourceEstimate): Right composition operand.

    Returns:
        frozenset[WireKey] | None: Union of known masks, or ``None`` when a
            nonzero operand still has an unknown footprint.
    """

    def normalized(estimate: ResourceEstimate) -> frozenset[WireKey] | None:
        """Treat a zero-depth unknown mask as the empty dependency set.

        Args:
            estimate (ResourceEstimate): Estimate whose mask is normalized.

        Returns:
            frozenset[WireKey] | None: Explicit dependency mask or ``None``.
        """
        if estimate._dependency_keys is not None:
            return estimate._dependency_keys
        if not _estimate_has_nonzero_depth(estimate):
            return frozenset()
        return None

    left_keys = normalized(left)
    right_keys = normalized(right)
    if left_keys is None or right_keys is None:
        return None
    return frozenset((*left_keys, *right_keys))


def _with_body_boundary_depth_metadata(
    estimate: ResourceEstimate,
    body_estimate: ResourceEstimate,
    *,
    source: str,
    zero_controls: ResourceExpr | int = 0,
    scalar_broadcast: ResourceExpr | int = 1,
) -> ResourceEstimate:
    """Classify conservative completion at a callable boundary.

    Args:
        estimate (ResourceEstimate): Caller-scoped body estimate.
        body_estimate (ResourceEstimate): Original body-scoped estimate.
        source (str): Callable name for the modeling assumption.
        zero_controls (ResourceExpr | int): Open-control X brackets surrounding
            the body. Defaults to zero.
        scalar_broadcast (ResourceExpr | int): Number of actual target
            elements receiving one scalar body. Defaults to one.

    Returns:
        ResourceEstimate: Estimate with conservative boundary metadata.
    """
    keys = body_estimate._dependency_keys
    if keys is None or not _estimate_has_nonzero_depth(body_estimate):
        return estimate
    activity_condition = _estimate_depth_activity_condition(estimate)
    bracket_condition = _and_conditions(
        sp.Gt(_expr(zero_controls), _ZERO),
        activity_condition,
    )
    if keys and bracket_condition is not sp.false:
        assumption = ResourceAssumption(
            "open-control brackets may finish on a different layer than the "
            "controlled body targets",
            source=source,
        )
        estimate = estimate._with_metadata(
            assumptions=(assumption,),
            quality=EstimateQuality.CONSERVATIVE,
            active_when=bracket_condition,
        )
    broadcast_condition = _and_conditions(
        sp.Gt(_expr(scalar_broadcast), _ONE),
        activity_condition,
    )
    if broadcast_condition is not sp.false:
        assumption = ResourceAssumption(
            "scalar-to-vector broadcast uses aggregate call latency for each "
            "target element's completion",
            source=source,
        )
        estimate = estimate._with_metadata(
            assumptions=(assumption,),
            quality=EstimateQuality.CONSERVATIVE,
            active_when=broadcast_condition,
        )
    return estimate


def _operation_depth_is_dependency_schedulable(
    operation: Operation,
) -> bool:
    """Return whether wire dependencies fully describe an operation's depth.

    Every supported aggregate carries its conditional global-ordering need in
    ``ResourceEstimate._global_barrier_condition``. This classifier therefore
    handles only operation kinds whose structure cannot be represented by the
    ordinary quantum footprint at all; it must not inspect nested bodies and
    thereby erase their branch activation conditions.

    Args:
        operation (Operation): Operation to classify.
    Returns:
        bool: Whether dependency scheduling is exact for this operation.
    """
    if isinstance(
        operation,
        (
            GateOperation,
            QInitOperation,
            MeasureOperation,
            MeasureVectorOperation,
            MeasureQFixedOperation,
            ProjectOperation,
            ResetOperation,
            GlobalPhaseOperation,
        ),
    ):
        return True
    if isinstance(operation, InvokeOperation):
        # The selected body or opaque model is scheduled on the invocation's
        # actual operands. ``operation.effects`` is a conservative union over
        # every backend/strategy implementation and must not serialize an
        # unrelated unitary selection.
        return True
    if isinstance(
        operation,
        (ControlledUOperation, InverseBlockOperation, SelectOperation),
    ):
        return True
    if isinstance(operation, (IfOperation, ForOperation, ForItemsOperation)):
        return True
    if isinstance(operation, WhileOperation):
        return True
    if not isinstance(operation, HasNestedOps):
        # Every supported atomic operation either has an explicit quantum
        # footprint or zero duration. Unknown atomic operations are rejected
        # by the interpreter before scheduling reaches this classifier.
        return True
    return False


def _operation_has_uniform_intrinsic_completion(
    operation: Operation,
    estimate: ResourceEstimate,
    dependency_keys: frozenset[WireKey],
    *,
    surrounding_controls: ResourceExpr,
) -> bool:
    """Return whether one atomic operation finishes all operands uniformly.

    Aggregate operations such as Pauli evolution and opaque calls may contain
    different internal exit layers even when they do not expose a dependency
    summary. A lowered multi-layer gate may likewise finish its controls,
    targets, or hidden ancillas on different layers. Only operations whose
    one-step resource semantics proves one uniform layer are accepted here.

    Args:
        operation (Operation): Operation whose intrinsic completion contract
            should be classified.
        estimate (ResourceEstimate): Resource summary produced for the
            operation.
        dependency_keys (frozenset[WireKey]): Caller-visible wires touched by
            the operation.
        surrounding_controls (ResourceExpr): Number of coherent controls
            surrounding the operation.

    Returns:
        bool: Whether all operand completions equal every aggregate depth-field
            peak for this atomic operation.
    """
    if not isinstance(
        operation,
        (
            GateOperation,
            QInitOperation,
            MeasureOperation,
            MeasureVectorOperation,
            MeasureQFixedOperation,
            ProjectOperation,
            ResetOperation,
            GlobalPhaseOperation,
            ExpvalOp,
        ),
    ):
        return False
    if any(
        demand != _ZERO
        for demand in (
            estimate.width.allocated_qubits,
            estimate.width.clean_ancilla_qubits,
            estimate.width.dirty_ancilla_qubits,
        )
    ):
        return False
    if (
        surrounding_controls == _ZERO
        and len(dependency_keys) <= 1
        or surrounding_controls == _ONE
        and not dependency_keys
    ):
        return True
    return _expressions_proven_equal_without_simplify(estimate.depth.depth, _ONE)


def _expressions_proven_equal_without_simplify(
    left: ResourceExpr,
    right: ResourceExpr,
) -> bool:
    """Prove equality without sending the expression to general simplify.

    This helper is used only to prove completion uniformity. Declining an
    expensive proof makes the estimate conservative but cannot undercount its
    gate or depth resources.

    Args:
        left (ResourceExpr): First resource expression.
        right (ResourceExpr): Second resource expression.

    Returns:
        bool: Whether structural comparison or SymPy assumptions prove
            equality.
    """
    if left == right:
        return True
    difference = cast(ResourceExpr, left - right)
    if difference == _ZERO or difference.is_zero is True:
        return True
    return False


def _aggregate_completion_overlap_condition(
    scheduled: Sequence[tuple[Operation, ResourceEstimate]],
    wire_footprints: Sequence[_WireFootprint | None],
    *,
    activity_conditions: Sequence[Boolean] | None = None,
) -> Boolean:
    """Return when an aggregate completion can delay a later operation.

    A nonuniform operation is harmless when it is the last user of its wires:
    its aggregate critical-path depth is still exact. The resource estimate
    becomes conservative only when that scalar latency is reused as a
    caller-visible wire completion for a subsequent operation.

    Args:
        scheduled (Sequence[tuple[Operation, ResourceEstimate]]): Operations
            paired with their resource summaries in program order.
        wire_footprints (Sequence[_WireFootprint | None]): Read/write
            footprints aligned with ``scheduled``.
        activity_conditions (Sequence[Boolean] | None): Optional precomputed
            depth-activity guards aligned with ``scheduled``. Defaults to
            computing them once for this call.

    Returns:
        Boolean: Guard under which a nonuniform aggregate may over-serialize
            a later wire dependency.

    Raises:
        AssertionError: If the operation, footprint, and activity sequences
            differ in length.
    """
    if activity_conditions is None:
        activity_conditions = _scheduled_depth_activity_conditions(scheduled)
    if not (len(scheduled) == len(wire_footprints) == len(activity_conditions)):
        raise AssertionError(
            "Scheduled operations, wire footprints, and activity conditions "
            "must have equal lengths."
        )
    uncertain_indices: dict[str, _OwnerWireIndices] = {}
    uncertain_activity: dict[WireKey, Boolean] = {}
    overlap_conditions: set[Boolean] = set()
    for (_operation, estimate), footprint, active in zip(
        scheduled,
        wire_footprints,
        activity_conditions,
        strict=True,
    ):
        if not _estimate_has_nonzero_depth(estimate):
            continue
        if footprint is None:
            raise AssertionError(
                "A nonzero-depth scheduled operation requires a wire footprint."
            )
        reads, writes = map(set, footprint)
        for owner, index in reads:
            owner_indices = uncertain_indices.get(owner)
            if owner_indices is None:
                continue
            for candidate in owner_indices.candidates(index):
                if _wire_index_relation(index, candidate) is _WireRelation.DISJOINT:
                    continue
                condition = _and_conditions(
                    uncertain_activity[(owner, candidate)],
                    active,
                )
                if condition is not sp.false:
                    overlap_conditions.add(condition)
        if estimate._dependency_completion_uniform is True:
            continue
        for key in reads | writes:
            owner, index = key
            uncertain_indices.setdefault(owner, _OwnerWireIndices()).add(index)
            previous = uncertain_activity.get(key, sp.false)
            uncertain_activity[key] = cast(Boolean, sp.Or(previous, active))
    return cast(
        Boolean,
        sp.Or(*overlap_conditions) if overlap_conditions else sp.false,
    )


def _conditional_completion(
    active: ResourceExpr,
    inactive: ResourceExpr,
    condition: Boolean,
) -> ResourceExpr:
    """Select a completion depth without expanding a Boolean to ``ITE``.

    Args:
        active (ResourceExpr): Completion when the operation executes.
        inactive (ResourceExpr): Previous completion when it does not.
        condition (Boolean): Symbolic operation-activity predicate.

    Returns:
        ResourceExpr: Binder-safe conditional completion expression.
    """
    return cast(
        ResourceExpr,
        inactive + _ConditionIndicator(condition) * (active - inactive),
    )


def _completion_after_conditional_duration(
    finish: ResourceExpr,
    start: ResourceExpr,
    inactive: ResourceExpr,
    active_when: Boolean,
) -> ResourceExpr:
    """Keep one scheduled completion compact across an inactive duration.

    When ``start`` already equals the inactive completion, ``finish`` differs
    only by the operation duration. That duration is zero whenever its
    activity guard is false, so wrapping the same condition in an additional
    :class:`_ConditionIndicator` is redundant.

    Args:
        finish (ResourceExpr): Completion after adding the operation duration.
        start (ResourceExpr): Dependency start selected for the operation.
        inactive (ResourceExpr): Completion retained when the duration is zero.
        active_when (Boolean): Guard under which the duration may be nonzero.

    Returns:
        ResourceExpr: Exact completion with no redundant activity indicator
            when structural equality proves it unnecessary.
    """
    if active_when is sp.true or _expressions_proven_equal_without_simplify(
        start,
        inactive,
    ):
        return finish
    return _conditional_completion(finish, inactive, active_when)


def _dependency_depth(
    scheduled: Sequence[tuple[Operation, ResourceEstimate]],
    wire_footprints: Sequence[_WireFootprint | None],
    *,
    activity_conditions: Sequence[Boolean] | None = None,
    read_conditions: Sequence[Mapping[WireKey, Boolean]] | None = None,
    scalar_values: Mapping[str, sp.Expr] | None = None,
    used_names: set[str] | None = None,
) -> tuple[DepthResources, dict[WireKey, ResourceExpr], Boolean, bool]:
    """Schedule operation summaries by wire dependencies and hybrid barriers.

    Args:
        scheduled (Sequence[tuple[Operation, ResourceEstimate]]): Operations in
            program order paired with their internally computed summaries.
        wire_footprints (Sequence[_WireFootprint | None]): Precomputed read and
            write keys aligned with ``scheduled``. Zero-depth operations use
            ``None``.
        activity_conditions (Sequence[Boolean] | None): Optional precomputed
            depth-activity guards aligned with ``scheduled``. Defaults to
            computing them once for this call.
        read_conditions (Sequence[Mapping[WireKey, Boolean]] | None): Optional
            per-operation guards for individual dependency reads. Missing keys
            are unconditional. Defaults to unconditional reads.
        scalar_values (Mapping[str, sp.Expr] | None): Optional supplied scalar
            values used only to prove completion uniformity. Defaults to
            ``None``.
        used_names (set[str] | None): Optional set updated with supplied names
            used by the uniformity proof. Defaults to ``None``.

    Returns:
        tuple[DepthResources, dict[WireKey, ResourceExpr], Boolean, bool]:
            Critical-path depths, caller-visible completion time of each
            touched wire for the total-depth field, and the condition under
            which a possible symbolic alias affected scheduling, followed by
            whether every touched wire is proven to complete at every
            aggregate depth-field peak.

    Raises:
        AssertionError: If a footprint is missing, or the operation,
            footprint, and activity sequences are not aligned.
    """
    if activity_conditions is None:
        activity_conditions = _scheduled_depth_activity_conditions(scheduled)
    if read_conditions is None:
        read_conditions = tuple({} for _ in scheduled)
    if not (
        len(scheduled)
        == len(wire_footprints)
        == len(activity_conditions)
        == len(read_conditions)
    ):
        raise AssertionError(
            "Scheduled operations, wire footprints, activity conditions, and "
            "read conditions must have equal lengths."
        )
    fields = tuple(field.name for field in dataclasses.fields(DepthResources))
    availability: dict[
        str,
        dict[str, dict[WireIndex, ResourceExpr]],
    ] = {field: {} for field in fields}
    indices_by_owner: dict[str, _OwnerWireIndices] = {}
    barrier_availability: dict[str, ResourceExpr] = {field: _ZERO for field in fields}
    peaks: dict[str, ResourceExpr] = {field: _ZERO for field in fields}
    possible_alias_conditions: set[Boolean] = set()
    completion_is_uniform = True
    for (operation, estimate), footprint, operation_active, operation_reads in zip(
        scheduled,
        wire_footprints,
        activity_conditions,
        read_conditions,
        strict=True,
    ):
        structurally_schedulable = _operation_depth_is_dependency_schedulable(operation)
        barrier_condition = (
            estimate._global_barrier_condition if structurally_schedulable else sp.true
        )
        footprint_reads = set(footprint[0]) if footprint is not None else set()
        footprint_writes = set(footprint[1]) if footprint is not None else set()
        if (
            not _estimate_has_nonzero_depth(estimate)
            and barrier_condition is sp.false
            and not footprint_reads
            and not footprint_writes
        ):
            continue
        if footprint is None and _estimate_has_nonzero_depth(estimate):
            raise AssertionError(
                "A nonzero-depth scheduled operation requires a wire footprint."
            )
        reads = footprint_reads
        writes = footprint_writes
        if estimate.width.clean_ancilla_qubits != _ZERO:
            # Width reports one reusable clean-ancilla pool (a maximum across
            # sequential operations). Treat that pool as a shared dependency
            # wire so depth never assumes two independent decompositions use
            # the same ancillas simultaneously.
            shared_pool = ("$resource_clean_ancilla_pool", None)
            reads.add(shared_pool)
            writes.add(shared_pool)
        if estimate.width.dirty_ancilla_qubits != _ZERO:
            shared_dirty_pool = ("$resource_dirty_ancilla_pool", None)
            reads.add(shared_dirty_pool)
            writes.add(shared_dirty_pool)
        if (
            _estimate_has_nonzero_depth(estimate)
            and estimate.width.allocated_qubits != _ZERO
        ):
            anonymous_workspace = (
                f"$resource_anonymous_workspace:{id(operation)}",
                None,
            )
            reads.add(anonymous_workspace)
            writes.add(anonymous_workspace)
        # Quantum resources are occupied for the whole operation even when an
        # operation consumes them without returning a quantum result (for
        # example measurement).  Classical SSA tokens are immutable and may
        # fan out, so reading one must not delay another independent consumer.
        occupied = {
            key for key in reads | writes if not _is_classical_dependency_key(key)
        }
        occupied.update(key for key in writes if _is_classical_dependency_key(key))
        if (
            estimate._dependency_keys is not None
            and estimate._dependency_completion_uniform is not True
        ):
            completion_is_uniform = False
        scheduling_active = _boolean_condition(
            sp.Or(operation_active, barrier_condition)
        )
        if scheduling_active is sp.false:
            continue
        for field in fields:
            duration = cast(ResourceExpr, getattr(estimate.depth, field))
            owner_depths = availability[field]
            definite_dependencies: list[ResourceExpr] = []
            possible_dependencies: list[ResourceExpr] = []
            for owner, index in reads:
                read_condition = operation_reads.get((owner, index), sp.true)
                if read_condition is sp.false:
                    continue
                owner_indices = indices_by_owner.get(owner)
                if owner_indices is None:
                    continue
                wire_depths = owner_depths.get(owner, {})
                for candidate_index in owner_indices.candidates(index):
                    depth = wire_depths.get(candidate_index)
                    if depth is None:
                        continue
                    if read_condition is not sp.true:
                        depth = _conditional_completion(
                            depth,
                            _ZERO,
                            read_condition,
                        )
                    relation = _wire_index_relation(index, candidate_index)
                    if relation is _WireRelation.DISJOINT:
                        continue
                    if relation is _WireRelation.DEFINITE_OVERLAP:
                        definite_dependencies.append(depth)
                    else:
                        possible_dependencies.append(depth)
            if structurally_schedulable:
                baseline_start = _resource_max_many(
                    [*definite_dependencies, barrier_availability[field]]
                )
                dependency_start = _resource_max_many(
                    [baseline_start, *possible_dependencies]
                )
                barrier_start = _resource_max(
                    peaks[field],
                    barrier_availability[field],
                )
                start = _conditional_completion(
                    barrier_start,
                    dependency_start,
                    barrier_condition,
                )
                for dependency_depth in possible_dependencies:
                    condition = _and_conditions(
                        _and_conditions(
                            operation_active,
                            cast(Boolean, sp.Not(barrier_condition)),
                        ),
                        _and_conditions(
                            _resource_activity_condition(dependency_depth),
                            sp.Gt(
                                dependency_depth,
                                baseline_start,
                            ),
                        ),
                    )
                    if condition is not sp.false:
                        possible_alias_conditions.add(condition)
            else:
                start = _resource_max(
                    peaks[field],
                    barrier_availability[field],
                )
            finish = start + duration
            peaks[field] = _resource_max(peaks[field], finish)
            if barrier_condition is not sp.false:
                previous_barrier = barrier_availability[field]
                barrier_availability[field] = _completion_after_conditional_duration(
                    finish,
                    start,
                    previous_barrier,
                    barrier_condition,
                )
            for owner, index in occupied:
                wire_depth = owner_depths.setdefault(owner, {})
                previous = wire_depth.get(index, _ZERO)
                wire_depth[index] = _completion_after_conditional_duration(
                    finish,
                    start,
                    previous,
                    scheduling_active,
                )
        for owner, index in occupied:
            indices_by_owner.setdefault(owner, _OwnerWireIndices()).add(index)
    completion_keys = {
        (owner, index)
        for owner, owner_depths in availability["depth"].items()
        for index in owner_depths
    }
    if completion_is_uniform:
        for field in fields:
            peak = _specialize_dependency_expression(
                peaks[field],
                scalar_values,
                used_names,
            )
            if any(
                not _expressions_proven_equal_without_simplify(
                    _specialize_dependency_expression(
                        cast(
                            ResourceExpr,
                            availability[field].get(owner, {}).get(index, _ZERO),
                        ),
                        scalar_values,
                        used_names,
                    ),
                    peak,
                )
                for owner, index in completion_keys
            ):
                completion_is_uniform = False
                break
    return (
        DepthResources(**peaks),
        {
            (owner, index): depth
            for owner, owner_depths in availability["depth"].items()
            for index, depth in owner_depths.items()
        },
        cast(
            Boolean,
            sp.Or(*possible_alias_conditions)
            if possible_alias_conditions
            else sp.false,
        ),
        completion_is_uniform,
    )
