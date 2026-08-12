"""Profile regions and dispatch IR operations for resource estimation."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import sympy as sp
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._allocation_width import (
    _without_input_allocation_sites,
)
from qamomile.circuit.estimator._classical_provenance import (
    _classical_fact_runtime_condition,
    _direct_observation_source_conditions,
    _merge_classical_source_conditions,
    _operation_classical_dependency_outputs,
    _propagate_operation_measurement_taint,
    _resolved_classical_source_conditions,
)
from qamomile.circuit.estimator._compound_affine_scheduling import (
    _compound_affine_region_schedules,
    _CompoundAffineSchedule,
)
from qamomile.circuit.estimator._constants import (
    _ONE,
    _ZERO,
)
from qamomile.circuit.estimator._constraints import (
    _operation_array_constraints,
    _quantum_operand_width_constraints,
)
from qamomile.circuit.estimator._dependency_footprints import (
    _classical_dependency_footprint,
    _classical_dependency_key,
    _expand_dependency_owner_aliases,
    _is_classical_dependency_key,
    _operation_has_unresolved_quantum_index,
    _quantum_wire_keys,
    _WireFootprint,
)
from qamomile.circuit.estimator._dependency_indices import WireKey
from qamomile.circuit.estimator._dependency_metadata import (
    _merge_synchronized_entry_conditions,
    _normalized_dependency_completion,
)
from qamomile.circuit.estimator._dependency_synchronization import (
    _merge_synchronized_entry_certificates,
    _SynchronizedEntryCertificate,
)
from qamomile.circuit.estimator._estimate import ResourceEstimate
from qamomile.circuit.estimator._estimate_validation import (
    _estimate_activity,
    _with_constraints,
)
from qamomile.circuit.estimator._interpreter_control_batching import (
    _ControlBatchingInterpreter,
)
from qamomile.circuit.estimator._interpreter_dataflow import (
    _require_uncontrolled_operation,
    _ResourceInlineBoundaryOperation,
)
from qamomile.circuit.estimator._liveness import (
    _liveness_width,
)
from qamomile.circuit.estimator._product_formula import (
    _apply_product_formula_contract,
    _require_concrete_product_formula_structure,
)
from qamomile.circuit.estimator._resolver import (
    ExprResolver,
)
from qamomile.circuit.estimator._resource_base import (
    ApproximationStatus,
    EstimateDerivation,
    EstimateQuality,
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_constraints import (
    _ResourceConstraint,
)
from qamomile.circuit.estimator._resource_expressions import (
    _boolean_condition,
    _expr,
    _resource_activity_condition,
    _resource_max,
)
from qamomile.circuit.estimator._resource_types import (
    ResourceAssumption,
)
from qamomile.circuit.estimator._scheduling import (
    _aggregate_completion_overlap_condition,
    _dependency_depth,
    _estimate_has_nonzero_depth,
    _merge_dependency_keys,
    _operation_has_uniform_intrinsic_completion,
    _scheduled_depth_activity_conditions,
    _synchronized_entry_scan,
)
from qamomile.circuit.estimator._scheduling_classical_sources import (
    _scheduling_classical_input_sources,
)
from qamomile.circuit.ir.operation.arithmetic_operations import (
    UnaryMathOp,
    UnaryMathOpKind,
)
from qamomile.circuit.ir.operation.callable import (
    InvokeOperation,
)
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.classical_ops import (
    ReturnQuantumArrayElementOperation,
    StoreArrayElementOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.control_work import (
    ControlWorkKind,
    classify_control_work,
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
from qamomile.circuit.ir.operation.operation import (
    Operation,
    OperationKind,
    QInitOperation,
)
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.operation.slice_array import (
    ReleaseSliceViewOperation,
    SliceArrayOperation,
)
from qamomile.circuit.ir.types.primitives import (
    FloatType,
    UIntType,
)

_CONSUMABLE_COMPOUND_SCHEDULING_ASSUMPTIONS = frozenset(
    {
        (
            "for",
            "symbolic loop depth is sequential because disjoint iteration "
            "footprints could not be proven",
        ),
        (
            "dependency scheduler",
            "symbolic quantum indices may alias and are scheduled conservatively",
        ),
        (
            "dependency scheduler",
            "aggregate latency may over-serialize a later wire dependency and "
            "therefore overestimate depth",
        ),
        (
            "dependency scheduler",
            "aggregate loop depth assumes synchronized input wires, but prior "
            "overlapping work may desynchronize them",
        ),
        (
            "ForOperation",
            "unresolved quantum index may alias any scalar of its allocation",
        ),
        (
            "GateOperation",
            "unresolved quantum index may alias any scalar of its allocation",
        ),
    }
)


def _compound_component_metadata_is_consumable(
    estimate: ResourceEstimate,
) -> bool:
    """Return whether a structural proof may replace scheduling-only facts.

    Args:
        estimate (ResourceEstimate): Candidate loop component estimate.

    Returns:
        bool: Whether every non-exact fact is one of the scheduling limitations
        independently superseded by the compound affine proof.
    """
    guarded_assumptions = estimate._guarded_assumptions or ()
    guarded_qualities = estimate._guarded_qualities or ()
    return (
        estimate.derivation is EstimateDerivation.STRUCTURAL
        and estimate.approximation is ApproximationStatus.EXACT
        and estimate._global_barrier_condition is sp.false
        and not (estimate._guarded_derivations or ())
        and not (estimate._guarded_approximations or ())
        and all(
            (fact.assumption.source, fact.assumption.message)
            in _CONSUMABLE_COMPOUND_SCHEDULING_ASSUMPTIONS
            for fact in guarded_assumptions
        )
        and all(
            fact.quality is EstimateQuality.CONSERVATIVE for fact in guarded_qualities
        )
        and all(
            any(
                quality.active_when == assumption.active_when
                for assumption in guarded_assumptions
            )
            for quality in guarded_qualities
        )
    )


def _without_consumed_compound_scheduling_facts(
    estimate: ResourceEstimate,
) -> ResourceEstimate:
    """Remove only scheduling facts independently discharged by a proof.

    Args:
        estimate (ResourceEstimate): Proven compound component estimate.

    Returns:
        ResourceEstimate: Copy retaining public counts, trace, constraints, and
        non-scheduling provenance while clearing consumed scheduling facts.
    """
    return dataclasses.replace(
        estimate,
        assumptions=(),
        quality=EstimateQuality.EXACT,
        _guarded_assumptions=(),
        _guarded_qualities=(),
        _dependency_synchronized_entry_conditions={},
        _dependency_synchronized_entry_certificates=(),
        _global_barrier_condition=sp.false,
    )


def _compound_schedule_is_applicable(
    schedule: _CompoundAffineSchedule,
    scheduled: list[tuple[Operation, ResourceEstimate]],
    footprints: list[_WireFootprint | None],
    classical_dependency_conditions: list[Boolean],
    classical_read_conditions: list[dict[WireKey, Boolean]],
) -> bool:
    """Validate metadata premises for one structural compound schedule.

    Args:
        schedule (_CompoundAffineSchedule): Structural affine proof.
        scheduled (list[tuple[Operation, ResourceEstimate]]): Normalized
            operation estimates in program order.
        footprints (list[_WireFootprint | None]): Aligned read/write footprints.
        classical_dependency_conditions (list[Boolean]): Aligned classical
            dependency activity guards.
        classical_read_conditions (list[dict[WireKey, Boolean]]): Aligned
            guarded classical reads.

    Returns:
        bool: Whether the proof can replace exactly its two loop components
        without dropping any external classical or non-scheduling metadata.
    """
    component_indices = frozenset(schedule.component_indices)
    for index in range(schedule.first_index, schedule.stop_index):
        _operation, estimate = scheduled[index]
        footprint = footprints[index]
        if (
            classical_dependency_conditions[index] is not sp.false
            or classical_read_conditions[index]
            or (
                footprint is not None
                and any(
                    _is_classical_dependency_key(key)
                    for key in (*footprint[0], *footprint[1])
                )
            )
        ):
            return False
        if index in component_indices:
            if not _compound_component_metadata_is_consumable(estimate):
                return False
            continue
        if _estimate_has_nonzero_depth(estimate) or footprint not in (
            None,
            (frozenset(), frozenset()),
        ):
            return False
    return True


def _compound_schedule_estimate(
    schedule: _CompoundAffineSchedule,
    scheduled: list[tuple[Operation, ResourceEstimate]],
) -> ResourceEstimate:
    """Build one scheduler-only estimate for a proven compound region.

    Args:
        schedule (_CompoundAffineSchedule): Exact structural schedule proof.
        scheduled (list[tuple[Operation, ResourceEstimate]]): Original aligned
            operation estimates.

    Returns:
        ResourceEstimate: Sequential public resources with proof-derived depth
        and exact dependency metadata.
    """
    component_indices = frozenset(schedule.component_indices)
    span = [
        (
            _without_consumed_compound_scheduling_facts(estimate)
            if index in component_indices
            else estimate
        )
        for index, (_operation, estimate) in enumerate(
            scheduled[schedule.first_index : schedule.stop_index],
            start=schedule.first_index,
        )
    ]
    estimate = ResourceEstimate.seq_all(span)
    certificates = (
        (
            _SynchronizedEntryCertificate(
                coverage=schedule.coverage,
                frontier=schedule.frontier,
                active_when=schedule.active_when,
            ),
        )
        if schedule.coverage and schedule.active_when is not sp.false
        else ()
    )
    return dataclasses.replace(
        estimate,
        depth=schedule.depth,
        _dependency_keys=schedule.coverage,
        _dependency_reads=schedule.coverage,
        _dependency_writes=schedule.coverage,
        _dependency_completion={key: schedule.depth.depth for key in schedule.coverage},
        _dependency_completion_uniform=schedule.completion_uniform,
        _dependency_synchronized_entry_conditions={},
        _dependency_synchronized_entry_certificates=certificates,
        _global_barrier_condition=sp.false,
    )


class _RegionAnalysisInterpreter(_ControlBatchingInterpreter):
    """Add observation summaries, region scheduling, and dispatch."""

    def eval_operations(
        self,
        operations: list[Operation],
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
        initial_allocations: Mapping[str, ResourceExpr] | None = None,
        allow_control_batching: bool = True,
    ) -> ResourceEstimate:
        """Evaluate a list of operations sequentially.

        Args:
            operations (list[Operation]): Operations to evaluate.
            resolver (ExprResolver): Value resolver for this scope.
            controls (ResourceExpr | int): Surrounding control count. Defaults
                to zero.
            initial_allocations (Mapping[str, ResourceExpr] | None): Live
                quantum inputs keyed by logical wire ID. Defaults to ``None``.
            allow_control_batching (bool): Whether this body boundary may
                choose a shared control ladder. Recursive call boundaries keep
                their own independent choice. Defaults to ``True``.

        Returns:
            ResourceEstimate: Sequential composition of operation resources.
        """
        self._reserve_while_trip_count_names(operations, resolver)
        control_count = self._apply_condition_values(
            _expr(controls),
            record_usage=False,
        )
        batch_condition = (
            self._controlled_body_batch_condition(
                operations,
                resolver,
                control_count,
            )
            if allow_control_batching
            else sp.false
        )
        if batch_condition is not sp.false:
            body = self.eval_operations(
                operations,
                resolver,
                controls=_ONE,
                initial_allocations=initial_allocations,
            )
            active = _resource_activity_condition(_estimate_activity(body))
            shared = self._with_shared_control_ladder(
                body,
                control_count,
            ).conditional(body, active)
            if batch_condition is sp.true:
                return shared
            direct = self.eval_operations(
                operations,
                resolver,
                controls=control_count,
                initial_allocations=initial_allocations,
                allow_control_batching=False,
            )
            batch_dependency_keys = _merge_dependency_keys(shared, direct)
            shared = dataclasses.replace(
                shared,
                _dependency_keys=batch_dependency_keys,
            )
            direct = dataclasses.replace(
                direct,
                _dependency_keys=batch_dependency_keys,
            )
            return shared.conditional(direct, batch_condition)

        previous_taint = self._run_state.measurement_taint_conditions
        self._run_state.measurement_taint_conditions = dict(previous_taint)
        try:
            scheduled: list[tuple[Operation, ResourceEstimate]] = []
            scheduled_classical_sources: list[
                tuple[dict[str, Boolean], dict[str, Boolean]]
            ] = []
            known_classical_sources: set[str] = set()
            seen_array_constraints: set[_ResourceConstraint] = set()
            for operation in operations:
                structural_input_sources = _resolved_classical_source_conditions(
                    operation.all_input_values(),
                    resolver,
                )
                known_classical_sources.update(structural_input_sources)
                classical_input_sources = _scheduling_classical_input_sources(
                    operation,
                    resolver,
                    self._run_state.while_trip_count_names,
                )
                operation_estimate = self.eval_operation(
                    operation,
                    resolver,
                    controls=control_count,
                )
                if isinstance(operation, StoreArrayElementOperation):
                    resolver.record_array_store(operation)
                self._run_state.measurement_taint_conditions = (
                    _propagate_operation_measurement_taint(
                        operation,
                        operation_estimate,
                        self._run_state.measurement_taint_conditions,
                    )
                )
                self._record_runtime_value_symbols(operation, resolver)
                self._publish_operation_classical_facts(
                    operation,
                    resolver,
                    classical_input_sources,
                )
                classical_output_sources = _resolved_classical_source_conditions(
                    _operation_classical_dependency_outputs(operation),
                    resolver,
                )
                _merge_classical_source_conditions(
                    classical_output_sources,
                    _direct_observation_source_conditions(operation, resolver),
                )
                published_classical_sources = {
                    source: guard
                    for source, guard in classical_output_sources.items()
                    if source not in known_classical_sources
                }
                known_classical_sources.update(classical_output_sources)
                known_classical_sources.update(published_classical_sources)
                if _estimate_has_nonzero_depth(
                    operation_estimate
                ) and _operation_has_unresolved_quantum_index(
                    operation,
                    resolver,
                    scalar_values=self._run_state.condition_values,
                    used_names=self._run_state.branch_condition_names,
                ):
                    assumption = ResourceAssumption(
                        "unresolved quantum index may alias any scalar of its "
                        "allocation",
                        source=type(operation).__name__,
                    )
                    operation_estimate = operation_estimate._with_metadata(
                        assumptions=(assumption,),
                        quality=EstimateQuality.CONSERVATIVE,
                    )
                if not self.config.trace:
                    # Every operation estimate is freshly produced for this
                    # traversal. Dropping its optional explanation before the
                    # sequential fold avoids constructing a deep tree that the
                    # public estimator would discard at the end anyway.
                    operation_estimate.trace = None
                discovered_constraints = (
                    ()
                    if isinstance(operation, HasNestedOps)
                    else _operation_array_constraints(
                        operation,
                        resolver,
                        active_when=self._run_state.constraint_scope_condition,
                        proven_cache=self._run_state.array_constraint_proven,
                    )
                )
                array_constraints = tuple(
                    constraint
                    for constraint in discovered_constraints
                    if constraint not in seen_array_constraints
                    and constraint not in operation_estimate._constraints
                )
                seen_array_constraints.update(array_constraints)
                operation_estimate = _with_constraints(
                    operation_estimate,
                    *array_constraints,
                )
                operation_estimate = dataclasses.replace(
                    operation_estimate,
                    control_decomposition=self.config.control_decomposition,
                )
                scheduled.append((operation, operation_estimate))
                scheduled_classical_sources.append(
                    (classical_input_sources, published_classical_sources)
                )
            dependency_keys: set[WireKey] = set()
            wire_footprints: list[_WireFootprint | None] = []
            classical_dependency_conditions: list[Boolean] = []
            classical_read_conditions: list[dict[WireKey, Boolean]] = []
            scheduled_with_dependencies: list[tuple[Operation, ResourceEstimate]] = []
            for (
                operation,
                operation_estimate,
            ), (
                classical_input_sources,
                classical_output_sources,
            ) in zip(
                scheduled,
                scheduled_classical_sources,
                strict=True,
            ):
                classical_reads, classical_writes, classical_active = (
                    _classical_dependency_footprint(
                        classical_input_sources,
                        classical_output_sources,
                    )
                )
                classical_dependency_conditions.append(classical_active)
                classical_read_conditions.append(
                    {
                        _classical_dependency_key(source): condition
                        for source, raw_condition in classical_input_sources.items()
                        if (condition := _boolean_condition(raw_condition))
                        is not sp.false
                    }
                )
                if isinstance(operation, (IfOperation, WhileOperation)):
                    condition_value = (
                        operation.condition
                        if isinstance(operation, IfOperation)
                        else operation.operands[0]
                    )
                    condition_runtime = _classical_fact_runtime_condition(
                        resolver.resolve_classical_fact(condition_value),
                    )
                    if condition_runtime is not sp.false and not classical_reads:
                        raise AssertionError(
                            "A runtime control-flow operation must read its "
                            "observation dependency token."
                        )
                if (
                    not _estimate_has_nonzero_depth(operation_estimate)
                    and not classical_reads
                    and not classical_writes
                ):
                    wire_footprints.append(None)
                    scheduled_with_dependencies.append(
                        (
                            operation,
                            dataclasses.replace(
                                operation_estimate,
                                _dependency_keys=frozenset(),
                                _dependency_completion={},
                                _dependency_completion_uniform=True,
                            ),
                        )
                    )
                    continue
                if operation_estimate._dependency_keys is not None:
                    footprint_keys = frozenset(
                        _expand_dependency_owner_aliases(
                            set(operation_estimate._dependency_keys),
                            self._run_state.dependency_owner_aliases,
                        )
                    )
                    reads = set(footprint_keys)
                    writes = set(footprint_keys)
                    completion_uniform = (
                        operation_estimate._dependency_completion_uniform
                    )
                else:
                    reads, writes = _quantum_wire_keys(
                        operation,
                        resolver,
                        scalar_values=self._run_state.condition_values,
                        used_names=self._run_state.branch_condition_names,
                        owner_aliases=self._run_state.dependency_owner_aliases,
                        allocation_owners_by_uuid=self._run_state.allocation_owners_by_uuid,
                    )
                    footprint_keys = frozenset(reads | writes)
                    completion_uniform = _operation_has_uniform_intrinsic_completion(
                        operation,
                        operation_estimate,
                        footprint_keys,
                        surrounding_controls=_expr(control_count),
                    )
                synchronized_entry_conditions: dict[WireKey, Boolean] = {}
                for (
                    key,
                    condition,
                ) in (
                    operation_estimate._dependency_synchronized_entry_conditions.items()
                ):
                    expanded_keys = _expand_dependency_owner_aliases(
                        {key},
                        self._run_state.dependency_owner_aliases,
                    )
                    synchronized_entry_conditions = (
                        _merge_synchronized_entry_conditions(
                            synchronized_entry_conditions,
                            {expanded_key: condition for expanded_key in expanded_keys},
                        )
                    )
                wire_footprints.append(
                    (
                        frozenset((*reads, *classical_reads)),
                        frozenset((*writes, *classical_writes)),
                    )
                )
                dependency_keys.update(footprint_keys)
                operation_completion = _normalized_dependency_completion(
                    operation_estimate
                )
                if (
                    operation_completion is not None
                    and self._run_state.dependency_owner_aliases
                ):
                    expanded_completion: dict[WireKey, ResourceExpr] = {}
                    for key, completion in operation_completion.items():
                        for expanded_key in _expand_dependency_owner_aliases(
                            {key},
                            self._run_state.dependency_owner_aliases,
                        ):
                            expanded_completion[expanded_key] = _resource_max(
                                expanded_completion.get(expanded_key, _ZERO),
                                completion,
                            )
                    operation_completion = expanded_completion
                if operation_completion is None:
                    operation_completion = {
                        key: operation_estimate.depth.depth for key in footprint_keys
                    }
                operation_estimate = dataclasses.replace(
                    operation_estimate,
                    _dependency_keys=footprint_keys,
                    _dependency_completion=operation_completion,
                    _dependency_completion_uniform=completion_uniform,
                    _dependency_synchronized_entry_conditions=(
                        synchronized_entry_conditions
                    ),
                )
                scheduled_with_dependencies.append((operation, operation_estimate))
            scheduled = scheduled_with_dependencies
            public_scheduled = list(scheduled)
            scheduler_scheduled = list(scheduled)
            scheduler_wire_footprints = list(wire_footprints)
            scheduler_classical_conditions = list(classical_dependency_conditions)
            scheduler_read_conditions = list(classical_read_conditions)
            if _expr(control_count) == _ZERO:
                structural_schedules = _compound_affine_region_schedules(
                    operations,
                    resolver,
                    owner_aliases=self._run_state.dependency_owner_aliases,
                    scalar_values=self._run_state.condition_values,
                    used_names=self._run_state.branch_condition_names,
                )
                accepted_schedules = tuple(
                    schedule
                    for schedule in structural_schedules
                    if _compound_schedule_is_applicable(
                        schedule,
                        scheduled,
                        wire_footprints,
                        classical_dependency_conditions,
                        classical_read_conditions,
                    )
                )
                consumed_components = {
                    index
                    for schedule in accepted_schedules
                    for index in schedule.component_indices
                }
                public_scheduled = [
                    (
                        operation,
                        (
                            _without_consumed_compound_scheduling_facts(
                                operation_estimate
                            )
                            if index in consumed_components
                            else operation_estimate
                        ),
                    )
                    for index, (operation, operation_estimate) in enumerate(scheduled)
                ]
                schedules_by_start = {
                    schedule.first_index: schedule for schedule in accepted_schedules
                }
                scheduler_scheduled = []
                scheduler_wire_footprints = []
                scheduler_classical_conditions = []
                scheduler_read_conditions = []
                index = 0
                while index < len(scheduled):
                    schedule = schedules_by_start.get(index)
                    if schedule is None:
                        scheduler_scheduled.append(scheduled[index])
                        scheduler_wire_footprints.append(wire_footprints[index])
                        scheduler_classical_conditions.append(
                            classical_dependency_conditions[index]
                        )
                        scheduler_read_conditions.append(
                            classical_read_conditions[index]
                        )
                        index += 1
                        continue
                    compound = _compound_schedule_estimate(schedule, scheduled)
                    scheduler_scheduled.append((scheduled[index][0], compound))
                    scheduler_wire_footprints.append(
                        (schedule.coverage, schedule.coverage)
                    )
                    scheduler_classical_conditions.append(sp.false)
                    scheduler_read_conditions.append({})
                    index = schedule.stop_index
            estimate = ResourceEstimate.seq_all(
                operation_estimate for _, operation_estimate in public_scheduled
            )
            dependency_keys = {
                key
                for _operation, operation_estimate in scheduler_scheduled
                for key in (operation_estimate._dependency_keys or frozenset())
            }
            depth_footprints = scheduler_wire_footprints
            if _expr(controls) != _ZERO:
                control_carrier = ("$resource_control_carrier", None)
                depth_footprints = [
                    (
                        (
                            frozenset((*footprint[0], control_carrier)),
                            frozenset((*footprint[1], control_carrier)),
                        )
                        if footprint is not None
                        else None
                    )
                    for footprint in scheduler_wire_footprints
                ]
            depth_activity_conditions = _scheduled_depth_activity_conditions(
                scheduler_scheduled
            )
            depth_activity_conditions = tuple(
                _boolean_condition(sp.Or(depth_active, classical_active))
                for depth_active, classical_active in zip(
                    depth_activity_conditions,
                    scheduler_classical_conditions,
                    strict=True,
                )
            )
            (
                scheduled_depth,
                scheduled_completion,
                possible_alias_active,
                completion_is_uniform,
            ) = _dependency_depth(
                scheduler_scheduled,
                depth_footprints,
                activity_conditions=depth_activity_conditions,
                read_conditions=scheduler_read_conditions,
                scalar_values=self._run_state.condition_values,
                used_names=self._run_state.branch_condition_names,
            )
            aggregate_completion_active = _aggregate_completion_overlap_condition(
                scheduler_scheduled,
                depth_footprints,
                activity_conditions=depth_activity_conditions,
            )
            synchronized_entry = _synchronized_entry_scan(
                scheduler_scheduled,
                depth_footprints,
                activity_conditions=depth_activity_conditions,
            )
            liveness = _liveness_width(
                scheduled,
                initial_allocations or {},
                resolver,
                allocation_owners_by_uuid=self._run_state.allocation_owners_by_uuid,
            )
            body_owned_allocation_owners = {
                operation.results[0].logical_id
                for operation in operations
                if isinstance(operation, QInitOperation)
                and operation.results
                and operation.results[0].logical_id not in (initial_allocations or {})
            }
            caller_entry_certificates = []
            for certificate in synchronized_entry.residual_certificates:
                coverage = frozenset(
                    key
                    for key in certificate.coverage
                    if key[0] not in body_owned_allocation_owners
                )
                if not coverage:
                    continue
                caller_entry_certificates.append(
                    dataclasses.replace(
                        certificate,
                        coverage=coverage,
                        frontier=frozenset(
                            key
                            for key in certificate.frontier
                            if key[0] not in body_owned_allocation_owners
                        ),
                    )
                )
            caller_entry_conditions: dict[WireKey, Boolean] = {}
            for _operation, operation_estimate in scheduler_scheduled:
                caller_entry_conditions = _merge_synchronized_entry_conditions(
                    caller_entry_conditions,
                    {
                        key: condition
                        for key, condition in (
                            operation_estimate._dependency_synchronized_entry_conditions.items()
                        )
                        if key[0] not in body_owned_allocation_owners
                    },
                )
            result = dataclasses.replace(
                estimate,
                depth=scheduled_depth,
                width=liveness.width,
                _allocation_sites=_without_input_allocation_sites(
                    estimate._allocation_sites,
                    operations,
                    initial_allocations or {},
                ),
                _dependency_keys=frozenset(dependency_keys),
                _dependency_reads=frozenset(
                    key
                    for (_operation, operation_estimate), footprint in zip(
                        scheduler_scheduled,
                        scheduler_wire_footprints,
                        strict=True,
                    )
                    if footprint is not None
                    and _estimate_has_nonzero_depth(operation_estimate)
                    for key in footprint[0]
                ),
                _dependency_writes=frozenset(
                    key
                    for (_operation, operation_estimate), footprint in zip(
                        scheduler_scheduled,
                        scheduler_wire_footprints,
                        strict=True,
                    )
                    if footprint is not None
                    and _estimate_has_nonzero_depth(operation_estimate)
                    for key in footprint[1]
                ),
                _dependency_completion={
                    key: completion
                    for key, completion in scheduled_completion.items()
                    if key in dependency_keys
                },
                _dependency_completion_uniform=completion_is_uniform,
                _dependency_synchronized_entry_conditions=caller_entry_conditions,
                _dependency_synchronized_entry_certificates=(
                    _merge_synchronized_entry_certificates(caller_entry_certificates)
                ),
                _output_sizes=liveness.final_live_by_owner,
                _input_sizes=dict(initial_allocations or {}),
                _has_output_summary=True,
                _measurement_taint_conditions=dict(
                    self._run_state.measurement_taint_conditions
                ),
            )
            if possible_alias_active is not sp.false:
                assumption = ResourceAssumption(
                    "symbolic quantum indices may alias and are scheduled "
                    "conservatively",
                    source="dependency scheduler",
                )
                result = result._with_metadata(
                    assumptions=(assumption,),
                    quality=EstimateQuality.CONSERVATIVE,
                    active_when=possible_alias_active,
                )
            if aggregate_completion_active is not sp.false:
                assumption = ResourceAssumption(
                    "aggregate latency may over-serialize a later wire dependency "
                    "and therefore overestimate depth",
                    source="dependency scheduler",
                )
                result = result._with_metadata(
                    assumptions=(assumption,),
                    quality=EstimateQuality.CONSERVATIVE,
                    active_when=aggregate_completion_active,
                )
            if synchronized_entry.violation is not sp.false:
                assumption = ResourceAssumption(
                    "aggregate loop depth assumes synchronized input wires, "
                    "but prior overlapping work may desynchronize them",
                    source="dependency scheduler",
                )
                result = result._with_metadata(
                    assumptions=(assumption,),
                    quality=EstimateQuality.CONSERVATIVE,
                    active_when=synchronized_entry.violation,
                )
            return result
        finally:
            self._run_state.measurement_taint_conditions = previous_taint

    def eval_operation(
        self,
        operation: Operation,
        resolver: ExprResolver,
        *,
        controls: ResourceExpr | int = 0,
    ) -> ResourceEstimate:
        """Evaluate one operation.

        Args:
            operation (Operation): Operation to evaluate.
            resolver (ExprResolver): Resolver for operation operands.
            controls (ResourceExpr | int): Surrounding control count. Defaults
                to zero.

        Returns:
            ResourceEstimate: Operation resource estimate.

        Raises:
            ValueError: If an estimator-only call-site width contract is
                malformed or violated.
            NotImplementedError: If the operation kind is not supported by
                resource estimation.
        """
        match operation:
            case _ResourceInlineBoundaryOperation():
                self._bind_resource_inline_array_states(operation, resolver)
                _require_concrete_product_formula_structure(
                    operation.callable_attrs,
                    operation.resource_operands,
                    bindings={
                        **self.bindings,
                        **self._run_state.condition_values,
                    },
                    resolver=resolver,
                    specialize=lambda expression: self._apply_condition_values(
                        expression,
                        record_usage=False,
                    ),
                    source=operation.source,
                )
                boundary = _with_constraints(
                    ResourceEstimate.zero(),
                    *_quantum_operand_width_constraints(
                        operation.callable_attrs,
                        operation.constraint_operands,
                        resolver,
                        source=operation.source,
                    ),
                )
                return _apply_product_formula_contract(
                    boundary,
                    operation.callable_attrs,
                    operation.resource_operands,
                    resolver,
                    bindings=self.bindings,
                    specialize=lambda expression: self._apply_condition_values(
                        expression,
                        record_usage=False,
                    ),
                    source=operation.source,
                )
            case GateOperation():
                return self.eval_gate(operation, controls=controls)
            case QInitOperation():
                return self.eval_qinit(operation, resolver)
            case (
                MeasureOperation() | MeasureVectorOperation() | MeasureQFixedOperation()
            ):
                _require_uncontrolled_operation(operation, controls)
                return self.eval_measure(operation, resolver)
            case ExpvalOp():
                _require_uncontrolled_operation(operation, controls)
                return self.eval_expval(operation, resolver)
            case ProjectOperation():
                _require_uncontrolled_operation(operation, controls)
                return self.eval_project(operation)
            case ResetOperation():
                _require_uncontrolled_operation(operation, controls)
                return self.eval_reset(operation)
            case ForOperation():
                return self.eval_for(operation, resolver, controls=controls)
            case WhileOperation():
                return self.eval_while(operation, resolver, controls=controls)
            case IfOperation():
                return self.eval_if(operation, resolver, controls=controls)
            case ForItemsOperation():
                return self.eval_for_items(operation, resolver, controls=controls)
            case InvokeOperation():
                return self.eval_invoke(operation, resolver, controls=controls)
            case SelectOperation():
                return self.eval_select(operation, resolver, controls=controls)
            case GlobalPhaseOperation():
                return self.eval_global_phase(
                    operation,
                    resolver,
                    controls=controls,
                )
            case ControlledUOperation():
                return self.eval_controlled_u(operation, resolver, controls=controls)
            case InverseBlockOperation():
                return self.eval_inverse_block(operation, resolver, controls=controls)
            case PauliEvolveOp():
                return self.eval_pauli_evolve(
                    operation,
                    resolver,
                    controls=controls,
                )
            case UnaryMathOp():
                if classify_control_work(operation) is ControlWorkKind.UNSUPPORTED:
                    _require_uncontrolled_operation(operation, controls)
                return self.eval_unary_math(operation, resolver)
            case CastOperation() | ReturnQuantumArrayElementOperation():
                return ResourceEstimate.zero()
            case StoreArrayElementOperation():
                # The enclosing sequential interpreter publishes the updated
                # array state after this zero-resource semantic operation.
                return ResourceEstimate.zero()
            case SliceArrayOperation() | ReleaseSliceViewOperation():
                # Estimation runs before the transpiler strips slice-lifetime
                # markers. They are valid zero-work structure here even though
                # either marker reaching the later emit walker is an error.
                return ResourceEstimate.zero()
            case HasNestedOps():
                raise NotImplementedError(
                    "Resource estimation does not support nested operation "
                    f"{type(operation).__name__}."
                )
            case _:
                if classify_control_work(operation) is ControlWorkKind.UNSUPPORTED:
                    _require_uncontrolled_operation(operation, controls)
                if operation.operation_kind is OperationKind.CLASSICAL:
                    return ResourceEstimate.zero()
                raise NotImplementedError(
                    "Resource estimation does not support non-classical "
                    f"operation {type(operation).__name__} "
                    f"({operation.operation_kind.value}); refusing to report "
                    "it as an exact zero-cost operation."
                )

    def eval_unary_math(
        self,
        operation: UnaryMathOp,
        resolver: ExprResolver,
    ) -> ResourceEstimate:
        """Evaluate structural requirements of one unary math expression.

        Classical arithmetic has no quantum gate cost, but unary math domains
        must remain valid after resource-input substitution. Retaining those
        domains here produces source-level diagnostics before downstream width
        expressions turn into infinities or invalid unsigned sizes.

        Args:
            operation (UnaryMathOp): Unary mathematical IR operation.
            resolver (ExprResolver): Resolver for the mathematical operand.

        Returns:
            ResourceEstimate: Zero quantum cost with any retained structural
                input-domain requirements.

        Raises:
            ValueError: If the operation types are malformed or a concrete
                input violates the unary operation's numeric domain.
            NotImplementedError: If the unary operation kind is unknown.
        """
        kind = operation.kind
        if kind is None:
            raise ValueError("Cannot estimate unary math operation without a kind.")
        expected_output_type = FloatType if kind is UnaryMathOpKind.LOG2 else UIntType
        if not isinstance(
            operation.input.type, (UIntType, FloatType)
        ) or not isinstance(
            operation.output.type,
            expected_output_type,
        ):
            raise ValueError(
                f"Cannot estimate malformed {kind.name} operation: "
                "expected UInt or Float input and "
                f"{expected_output_type.__name__} output."
            )
        match kind:
            case UnaryMathOpKind.LOG2:
                is_uint = isinstance(operation.input.type, UIntType)
                constraint = _ResourceConstraint(
                    expression=resolver.resolve(operation.input),
                    minimum=1 if is_uint else 0,
                    label="log2 input",
                    integer=is_uint,
                    minimum_inclusive=is_uint,
                    finite=not is_uint,
                )
                return _with_constraints(
                    ResourceEstimate.zero("log2"),
                    constraint,
                )
            case UnaryMathOpKind.CEIL:
                if isinstance(operation.input.type, UIntType):
                    return ResourceEstimate.zero("ceil")
                constraint = _ResourceConstraint(
                    expression=resolver.resolve(operation.input),
                    minimum=-1,
                    label="ceil input",
                    integer=False,
                    minimum_inclusive=False,
                    finite=True,
                )
                return _with_constraints(
                    ResourceEstimate.zero("ceil"),
                    constraint,
                )
            case _:
                raise NotImplementedError(
                    "Resource estimation does not support unary math kind "
                    f"{operation.kind!r}."
                )
