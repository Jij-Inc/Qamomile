"""Compose complete resource estimates across execution alternatives."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import sympy as sp

from qamomile.circuit.estimator._allocation_width import (
    _activate_allocation_sites,
    _anonymous_allocation_width,
    _branch_width_with_static_allocations,
    _merge_allocation_sites,
    _width_with_identity_aware_allocations,
)
from qamomile.circuit.estimator._dependency_metadata import (
    _conditional_dependency_completion,
    _max_dependency_completion,
    _merge_dependency_accesses,
    _merge_synchronized_entry_conditions,
    _seq_dependency_completion,
)
from qamomile.circuit.estimator._dependency_synchronization import (
    _guard_synchronized_entry_certificates,
    _merge_synchronized_entry_certificates,
)
from qamomile.circuit.estimator._estimate_provenance import (
    _merge_estimate_provenance,
    _merge_symbol_aliases,
    _with_estimate_metadata,
)
from qamomile.circuit.estimator._measurement_provenance import (
    _conditional_measurement_taint_conditions,
    _merge_measurement_taint_conditions,
)
from qamomile.circuit.estimator._resource_algebra import (
    _add_calls,
    _add_depth,
    _add_gates,
    _add_measurements,
    _add_resets,
    _conditional_calls,
    _conditional_depth,
    _conditional_gates,
    _conditional_measurements,
    _conditional_resets,
    _conditional_trace,
    _conditional_width,
    _max_calls,
    _max_depth,
    _max_gates,
    _max_measurements,
    _max_resets,
    _max_width,
    _merge_trace,
    _parallel_width,
    _seq_width,
)
from qamomile.circuit.estimator._resource_base import (
    EstimateQuality,
    _combine_approximation,
    _combine_derivation,
    _combine_quality,
)
from qamomile.circuit.estimator._resource_expressions import (
    _boolean_condition,
    _piecewise,
    _unresolved_condition_guard,
)
from qamomile.circuit.estimator._resource_types import (
    ResourceAssumption,
    _GuardedQuality,
)
from qamomile.circuit.estimator._scheduling import _merge_dependency_keys

if TYPE_CHECKING:
    from qamomile.circuit.estimator._estimate import ResourceEstimate


def _compose_sequential(
    left: ResourceEstimate,
    right: ResourceEstimate,
) -> ResourceEstimate:
    """Compose one estimate after another.

    Args:
        left (ResourceEstimate): Estimate executed first.
        right (ResourceEstimate): Estimate executed second.

    Returns:
        ResourceEstimate: Sequential composition.
    """
    control_decomposition = _merge_estimate_provenance(left, right)
    return dataclasses.replace(
        left.zero(),
        width=_seq_width(left.width, right.width),
        gates=_add_gates(left.gates, right.gates),
        depth=_add_depth(left.depth, right.depth),
        calls=_add_calls(left.calls, right.calls),
        measurements=_add_measurements(left.measurements, right.measurements),
        resets=_add_resets(left.resets, right.resets),
        assumptions=(*left.assumptions, *right.assumptions),
        trace=_merge_trace("seq", left.trace, right.trace),
        derivation=_combine_derivation(left.derivation, right.derivation),
        quality=_combine_quality(left.quality, right.quality),
        approximation=_combine_approximation(
            left.approximation,
            right.approximation,
        ),
        control_decomposition=control_decomposition,
        _allocation_sites=_merge_allocation_sites(
            left._allocation_sites,
            right._allocation_sites,
        ),
        _constraints=(*left._constraints, *right._constraints),
        _dependency_keys=_merge_dependency_keys(left, right),
        _dependency_reads=_merge_dependency_accesses(
            left,
            right,
            writes=False,
        ),
        _dependency_writes=_merge_dependency_accesses(
            left,
            right,
            writes=True,
        ),
        _dependency_completion=_seq_dependency_completion(left, right),
        _dependency_synchronized_entry_conditions=(
            _merge_synchronized_entry_conditions(
                left._dependency_synchronized_entry_conditions,
                right._dependency_synchronized_entry_conditions,
            )
        ),
        _dependency_synchronized_entry_certificates=(
            _merge_synchronized_entry_certificates(
                left._dependency_synchronized_entry_certificates,
                right._dependency_synchronized_entry_certificates,
            )
        ),
        _global_barrier_condition=sp.Or(
            left._global_barrier_condition,
            right._global_barrier_condition,
        ),
        _measurement_taint_conditions=_merge_measurement_taint_conditions(
            left._measurement_taint_conditions,
            right._measurement_taint_conditions,
        ),
        _guarded_assumptions=(
            *(left._guarded_assumptions or ()),
            *(right._guarded_assumptions or ()),
        ),
        _guarded_derivations=(
            *(left._guarded_derivations or ()),
            *(right._guarded_derivations or ()),
        ),
        _guarded_qualities=(
            *(left._guarded_qualities or ()),
            *(right._guarded_qualities or ()),
        ),
        _guarded_approximations=(
            *(left._guarded_approximations or ()),
            *(right._guarded_approximations or ()),
        ),
        _symbol_aliases=_merge_symbol_aliases(left, right),
    )


def _compose_parallel(
    left: ResourceEstimate,
    right: ResourceEstimate,
) -> ResourceEstimate:
    """Compose two estimates that run concurrently.

    Args:
        left (ResourceEstimate): First concurrent estimate.
        right (ResourceEstimate): Second concurrent estimate.

    Returns:
        ResourceEstimate: Parallel composition.
    """
    control_decomposition = _merge_estimate_provenance(left, right)
    return dataclasses.replace(
        left.zero(),
        width=_parallel_width(left.width, right.width),
        gates=_add_gates(left.gates, right.gates),
        depth=_max_depth(left.depth, right.depth),
        calls=_add_calls(left.calls, right.calls),
        measurements=_add_measurements(left.measurements, right.measurements),
        resets=_add_resets(left.resets, right.resets),
        assumptions=(*left.assumptions, *right.assumptions),
        trace=_merge_trace("parallel", left.trace, right.trace),
        derivation=_combine_derivation(left.derivation, right.derivation),
        quality=_combine_quality(left.quality, right.quality),
        approximation=_combine_approximation(
            left.approximation,
            right.approximation,
        ),
        control_decomposition=control_decomposition,
        _allocation_sites=_merge_allocation_sites(
            left._allocation_sites,
            right._allocation_sites,
        ),
        _constraints=(*left._constraints, *right._constraints),
        _dependency_keys=_merge_dependency_keys(left, right),
        _dependency_reads=_merge_dependency_accesses(
            left,
            right,
            writes=False,
        ),
        _dependency_writes=_merge_dependency_accesses(
            left,
            right,
            writes=True,
        ),
        _dependency_completion=_max_dependency_completion(left, right),
        _dependency_synchronized_entry_conditions=(
            _merge_synchronized_entry_conditions(
                left._dependency_synchronized_entry_conditions,
                right._dependency_synchronized_entry_conditions,
            )
        ),
        _dependency_synchronized_entry_certificates=(
            _merge_synchronized_entry_certificates(
                left._dependency_synchronized_entry_certificates,
                right._dependency_synchronized_entry_certificates,
            )
        ),
        _global_barrier_condition=sp.Or(
            left._global_barrier_condition,
            right._global_barrier_condition,
        ),
        _measurement_taint_conditions=_merge_measurement_taint_conditions(
            left._measurement_taint_conditions,
            right._measurement_taint_conditions,
        ),
        _guarded_assumptions=(
            *(left._guarded_assumptions or ()),
            *(right._guarded_assumptions or ()),
        ),
        _guarded_derivations=(
            *(left._guarded_derivations or ()),
            *(right._guarded_derivations or ()),
        ),
        _guarded_qualities=(
            *(left._guarded_qualities or ()),
            *(right._guarded_qualities or ()),
        ),
        _guarded_approximations=(
            *(left._guarded_approximations or ()),
            *(right._guarded_approximations or ()),
        ),
        _symbol_aliases=_merge_symbol_aliases(left, right),
    )


def _compose_choice(
    left: ResourceEstimate,
    right: ResourceEstimate,
) -> ResourceEstimate:
    """Take a conservative element-wise choice between two estimates.

    Args:
        left (ResourceEstimate): First possible estimate.
        right (ResourceEstimate): Second possible estimate.

    Returns:
        ResourceEstimate: Conservative choice composition.
    """
    control_decomposition = _merge_estimate_provenance(left, right)
    allocation_sites = _merge_allocation_sites(
        left._allocation_sites,
        right._allocation_sites,
    )
    return dataclasses.replace(
        left.zero(),
        width=_branch_width_with_static_allocations(
            _max_width(left.width, right.width),
            left.width,
            left._allocation_sites,
            right.width,
            right._allocation_sites,
            allocation_sites,
        ),
        gates=_max_gates(left.gates, right.gates),
        depth=_max_depth(left.depth, right.depth),
        calls=_max_calls(left.calls, right.calls),
        measurements=_max_measurements(left.measurements, right.measurements),
        resets=_max_resets(left.resets, right.resets),
        assumptions=(*left.assumptions, *right.assumptions),
        trace=_merge_trace("choice", left.trace, right.trace),
        derivation=_combine_derivation(left.derivation, right.derivation),
        quality=_combine_quality(
            EstimateQuality.CONSERVATIVE,
            _combine_quality(left.quality, right.quality),
        ),
        approximation=_combine_approximation(
            left.approximation,
            right.approximation,
        ),
        control_decomposition=control_decomposition,
        _allocation_sites=allocation_sites,
        _constraints=(*left._constraints, *right._constraints),
        _dependency_keys=_merge_dependency_keys(left, right),
        _dependency_reads=_merge_dependency_accesses(
            left,
            right,
            writes=False,
        ),
        _dependency_writes=_merge_dependency_accesses(
            left,
            right,
            writes=True,
        ),
        _dependency_completion=_max_dependency_completion(left, right),
        _dependency_synchronized_entry_conditions=(
            _merge_synchronized_entry_conditions(
                left._dependency_synchronized_entry_conditions,
                right._dependency_synchronized_entry_conditions,
            )
        ),
        _dependency_synchronized_entry_certificates=(
            _merge_synchronized_entry_certificates(
                left._dependency_synchronized_entry_certificates,
                right._dependency_synchronized_entry_certificates,
            )
        ),
        _global_barrier_condition=sp.Or(
            left._global_barrier_condition,
            right._global_barrier_condition,
        ),
        _measurement_taint_conditions=_merge_measurement_taint_conditions(
            left._measurement_taint_conditions,
            right._measurement_taint_conditions,
        ),
        _guarded_assumptions=(
            *(left._guarded_assumptions or ()),
            *(right._guarded_assumptions or ()),
        ),
        _guarded_derivations=(
            *(left._guarded_derivations or ()),
            *(right._guarded_derivations or ()),
        ),
        _guarded_qualities=(
            *(left._guarded_qualities or ()),
            *(right._guarded_qualities or ()),
            _GuardedQuality(sp.true, EstimateQuality.CONSERVATIVE),
        ),
        _guarded_approximations=(
            *(left._guarded_approximations or ()),
            *(right._guarded_approximations or ()),
        ),
        _symbol_aliases=_merge_symbol_aliases(left, right),
    )


def _compose_conditional(
    when_true: ResourceEstimate,
    when_false: ResourceEstimate,
    condition: sp.Basic,
) -> ResourceEstimate:
    """Select between two estimates with a symbolic condition.

    Args:
        when_true (ResourceEstimate): Estimate selected when true.
        when_false (ResourceEstimate): Estimate selected when false.
        condition (sp.Basic): Symbolic selection predicate.

    Returns:
        ResourceEstimate: Field-wise conditional estimate.
    """
    predicate = _boolean_condition(condition)
    if predicate is sp.true:
        return when_true
    if predicate is sp.false:
        return when_false
    control_decomposition = _merge_estimate_provenance(
        when_true,
        when_false,
    )
    allocation_sites = _merge_allocation_sites(
        _activate_allocation_sites(when_true._allocation_sites, predicate),
        _activate_allocation_sites(
            when_false._allocation_sites,
            sp.Not(predicate),
        ),
    )
    conditional_width = _conditional_width(
        when_true.width,
        when_false.width,
        predicate,
    )
    anonymous_allocated = _piecewise(
        _anonymous_allocation_width(
            when_true.width,
            when_true._allocation_sites,
        ),
        _anonymous_allocation_width(
            when_false.width,
            when_false._allocation_sites,
        ),
        predicate,
    )
    dependency_keys = _merge_dependency_keys(when_true, when_false)
    estimate = dataclasses.replace(
        when_true.zero(),
        width=_width_with_identity_aware_allocations(
            conditional_width,
            allocation_sites,
            anonymous_allocated=anonymous_allocated,
        ),
        gates=_conditional_gates(when_true.gates, when_false.gates, predicate),
        depth=_conditional_depth(when_true.depth, when_false.depth, predicate),
        calls=_conditional_calls(when_true.calls, when_false.calls, predicate),
        measurements=_conditional_measurements(
            when_true.measurements,
            when_false.measurements,
            predicate,
        ),
        resets=_conditional_resets(
            when_true.resets,
            when_false.resets,
            predicate,
        ),
        assumptions=(*when_true.assumptions, *when_false.assumptions),
        trace=_conditional_trace(predicate, when_true.trace, when_false.trace),
        derivation=_combine_derivation(
            when_true.derivation,
            when_false.derivation,
        ),
        quality=_combine_quality(when_true.quality, when_false.quality),
        approximation=_combine_approximation(
            when_true.approximation,
            when_false.approximation,
        ),
        control_decomposition=control_decomposition,
        _allocation_sites=allocation_sites,
        _constraints=(
            *(constraint.when(predicate) for constraint in when_true._constraints),
            *(
                constraint.when(sp.Not(predicate))
                for constraint in when_false._constraints
            ),
        ),
        _dependency_keys=dependency_keys,
        _dependency_reads=_merge_dependency_accesses(
            when_true,
            when_false,
            writes=False,
        ),
        _dependency_writes=_merge_dependency_accesses(
            when_true,
            when_false,
            writes=True,
        ),
        _dependency_completion=_conditional_dependency_completion(
            when_true,
            when_false,
            predicate,
        ),
        _dependency_synchronized_entry_conditions=(
            _merge_synchronized_entry_conditions(
                {
                    key: sp.And(predicate, active)
                    for key, active in (
                        when_true._dependency_synchronized_entry_conditions.items()
                    )
                },
                {
                    key: sp.And(sp.Not(predicate), active)
                    for key, active in (
                        when_false._dependency_synchronized_entry_conditions.items()
                    )
                },
            )
        ),
        _dependency_synchronized_entry_certificates=(
            _merge_synchronized_entry_certificates(
                _guard_synchronized_entry_certificates(
                    when_true._dependency_synchronized_entry_certificates,
                    predicate,
                ),
                _guard_synchronized_entry_certificates(
                    when_false._dependency_synchronized_entry_certificates,
                    sp.Not(predicate),
                ),
            )
        ),
        _global_barrier_condition=sp.Or(
            sp.And(predicate, when_true._global_barrier_condition),
            sp.And(sp.Not(predicate), when_false._global_barrier_condition),
        ),
        _measurement_taint_conditions=(
            _conditional_measurement_taint_conditions(
                when_true._measurement_taint_conditions,
                when_false._measurement_taint_conditions,
                predicate,
            )
        ),
        _guarded_assumptions=(
            *(fact.when(predicate) for fact in (when_true._guarded_assumptions or ())),
            *(
                fact.when(sp.Not(predicate))
                for fact in (when_false._guarded_assumptions or ())
            ),
        ),
        _guarded_derivations=(
            *(fact.when(predicate) for fact in (when_true._guarded_derivations or ())),
            *(
                fact.when(sp.Not(predicate))
                for fact in (when_false._guarded_derivations or ())
            ),
        ),
        _guarded_qualities=(
            *(fact.when(predicate) for fact in (when_true._guarded_qualities or ())),
            *(
                fact.when(sp.Not(predicate))
                for fact in (when_false._guarded_qualities or ())
            ),
        ),
        _guarded_approximations=(
            *(
                fact.when(predicate)
                for fact in (when_true._guarded_approximations or ())
            ),
            *(
                fact.when(sp.Not(predicate))
                for fact in (when_false._guarded_approximations or ())
            ),
        ),
        _symbol_aliases=_merge_symbol_aliases(when_true, when_false),
    )
    masks_differ = when_true._dependency_keys != when_false._dependency_keys
    if masks_differ and (dependency_keys is None or len(dependency_keys) > 1):
        assumption = ResourceAssumption(
            "symbolic branch depth uses the union of branch-specific wire dependencies",
            source=str(predicate),
        )
        estimate = _with_estimate_metadata(
            estimate,
            assumptions=(assumption,),
            quality=EstimateQuality.CONSERVATIVE,
            active_when=_unresolved_condition_guard(predicate),
        )
    return estimate


class _SequentialEstimateComposer:
    """Compose a stream of estimates with logarithmic intermediate storage."""

    def __init__(self, empty: ResourceEstimate) -> None:
        """Initialize an empty binary-counter reduction.

        Args:
            empty (ResourceEstimate): Exact zero returned when no estimate is
                appended.
        """
        self._levels: list[ResourceEstimate | None] = []
        self._empty = empty

    def append(self, estimate: ResourceEstimate) -> None:
        """Append one estimate after all previously supplied estimates.

        Args:
            estimate (ResourceEstimate): Next estimate in execution order.

        Raises:
            ValueError: If composition encounters incompatible
                control-decomposition provenance.
        """
        carry = estimate
        level = 0
        while level < len(self._levels) and self._levels[level] is not None:
            earlier = self._levels[level]
            assert earlier is not None
            carry = earlier.seq(carry)
            self._levels[level] = None
            level += 1
        if level == len(self._levels):
            self._levels.append(carry)
        else:
            self._levels[level] = carry

    def finish(self) -> ResourceEstimate:
        """Return the order-preserving composition accumulated so far.

        Returns:
            ResourceEstimate: Sequential composition, or exact zero when no
                estimate was appended.

        Raises:
            ValueError: If composition encounters incompatible
                control-decomposition provenance.
        """
        result: ResourceEstimate | None = None
        for estimate in reversed(self._levels):
            if estimate is None:
                continue
            result = estimate if result is None else result.seq(estimate)
        return result if result is not None else self._empty
