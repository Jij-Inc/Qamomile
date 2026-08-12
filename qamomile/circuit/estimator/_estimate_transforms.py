"""Apply repeat, coherent-control, and inverse estimate transforms."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, cast

import sympy as sp

from qamomile.circuit.estimator._aggregate_control_profile import (
    _aggregate_arity_profile_reason,
    _aggregate_arity_projection_constraints,
    _aggregate_resource_profile_constraints,
    _aggregate_zero_gate_residual_condition,
    _simplify_aggregate_metadata_guard,
    _unprojected_aggregate_control_assumption,
)
from qamomile.circuit.estimator._aggregate_control_projection import (
    _project_abstract_aggregate_controlled_cost,
    _project_clean_ancilla_aggregate_controlled_cost,
)
from qamomile.circuit.estimator._allocation_width import (
    _activate_allocation_sites,
)
from qamomile.circuit.estimator._constants import _ONE, _ZERO
from qamomile.circuit.estimator._dependency_exactness import (
    _has_single_serializing_quantum_dependency,
)
from qamomile.circuit.estimator._dependency_metadata import (
    _synchronized_entry_activity_condition,
)
from qamomile.circuit.estimator._dependency_synchronization import (
    _clear_synchronized_entry_frontiers,
    _guard_synchronized_entry_certificates,
)
from qamomile.circuit.estimator._estimate_provenance import (
    _estimate_has_control_sensitive_resources,
    _with_estimate_metadata,
)
from qamomile.circuit.estimator._estimate_validation import (
    _estimate_activity,
    _require_unitary_resource_estimate,
)
from qamomile.circuit.estimator._measurement_provenance import (
    _guard_measurement_taint_conditions,
)
from qamomile.circuit.estimator._resource_algebra import (
    _conditional_width,
    _scale_calls,
    _scale_depth,
    _scale_gates,
    _scale_measurements,
    _scale_resets,
    _wrap_trace,
)
from qamomile.circuit.estimator._resource_base import (
    ApproximationStatus,
    ControlDecomposition,
    EstimateDerivation,
    EstimateQuality,
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_constraints import _ResourceConstraint
from qamomile.circuit.estimator._resource_expressions import (
    _boolean_condition,
    _ConditionIndicator,
    _expr,
    _resource_activity_condition,
    _safe_simplify,
)
from qamomile.circuit.estimator._resource_types import (
    ResourceAssumption,
    WidthResources,
)
from qamomile.circuit.estimator._scheduling import (
    _estimate_depth_activity_condition,
)

if TYPE_CHECKING:
    from qamomile.circuit.estimator._estimate import ResourceEstimate


def _repeat_estimate(
    estimate: ResourceEstimate,
    factor: ResourceExpr | int,
    *,
    conservative_nonuniform: bool = False,
    nonuniform_source: str = "repeat",
) -> ResourceEstimate:
    """Repeat an estimate while reusing its width.

    Args:
        estimate (ResourceEstimate): Estimate to repeat.
        factor (ResourceExpr | int): Iteration or power factor.
        conservative_nonuniform (bool): Whether to disclose that scalar depth
            scaling may over-serialize a nonuniform per-wire completion map.
        nonuniform_source (str): Source label for that disclosure.

    Returns:
        ResourceEstimate: Repeated estimate.

    Raises:
        ValueError: If a concrete factor is negative or non-integral.
    """
    factor_expr = _expr(factor)
    factor_constraint = _ResourceConstraint(
        expression=factor_expr,
        minimum=0,
        label="Resource repetition factor",
    )
    factor_constraint.validate()
    active_width = _conditional_width(
        estimate.width,
        WidthResources.zero(),
        sp.Gt(factor_expr, _ZERO),
    )
    active_sites = _activate_allocation_sites(
        estimate._allocation_sites,
        sp.Gt(factor_expr, _ZERO),
    )
    active_when = sp.Gt(factor_expr, _ZERO)
    repeated = dataclasses.replace(
        estimate.zero(),
        width=active_width,
        gates=_scale_gates(estimate.gates, factor_expr),
        depth=_scale_depth(estimate.depth, factor_expr),
        calls=_scale_calls(estimate.calls, factor_expr),
        measurements=_scale_measurements(estimate.measurements, factor_expr),
        resets=_scale_resets(estimate.resets, factor_expr),
        assumptions=(),
        trace=_wrap_trace(
            f"repeat({factor_expr})",
            estimate.trace.when(active_when) if estimate.trace is not None else None,
        ),
        derivation=EstimateDerivation.STRUCTURAL,
        quality=EstimateQuality.EXACT,
        approximation=ApproximationStatus.EXACT,
        control_decomposition=estimate.control_decomposition,
        _allocation_sites=active_sites,
        _constraints=tuple(
            constraint.when(active_when) for constraint in estimate._constraints
        )
        + ((factor_constraint,) if not factor_expr.is_number else ()),
        _dependency_keys=(
            frozenset() if factor_expr == _ZERO else estimate._dependency_keys
        ),
        _dependency_reads=(
            frozenset() if factor_expr == _ZERO else estimate._dependency_reads
        ),
        _dependency_writes=(
            frozenset() if factor_expr == _ZERO else estimate._dependency_writes
        ),
        _dependency_completion=(
            {
                key: cast(
                    ResourceExpr,
                    _ConditionIndicator(
                        sp.And(
                            active_when,
                            _resource_activity_condition(_expr(completion)),
                        )
                    )
                    * ((factor_expr - _ONE) * estimate.depth.depth + completion),
                )
                for key, completion in estimate._dependency_completion.items()
            }
            if estimate._dependency_completion is not None
            else None
        ),
        _dependency_completion_uniform=estimate._dependency_completion_uniform,
        _dependency_synchronized_entry_conditions=(
            {
                key: _boolean_condition(sp.And(active_when, condition))
                for key, condition in (
                    estimate._dependency_synchronized_entry_conditions.items()
                )
                if _boolean_condition(sp.And(active_when, condition)) is not sp.false
            }
            if factor_expr != _ZERO
            else {}
        ),
        _dependency_synchronized_entry_certificates=(
            _clear_synchronized_entry_frontiers(
                _guard_synchronized_entry_certificates(
                    estimate._dependency_synchronized_entry_certificates,
                    active_when,
                )
            )
            if factor_expr != _ZERO
            else ()
        ),
        _global_barrier_condition=sp.And(
            estimate._global_barrier_condition,
            active_when,
        ),
        _measurement_taint_conditions=_guard_measurement_taint_conditions(
            estimate._measurement_taint_conditions,
            active_when,
        ),
        _guarded_assumptions=tuple(
            fact.when(active_when) for fact in (estimate._guarded_assumptions or ())
        ),
        _guarded_derivations=tuple(
            fact.when(active_when) for fact in (estimate._guarded_derivations or ())
        ),
        _guarded_qualities=tuple(
            fact.when(active_when) for fact in (estimate._guarded_qualities or ())
        ),
        _guarded_approximations=tuple(
            fact.when(active_when) for fact in (estimate._guarded_approximations or ())
        ),
        _symbol_aliases=estimate._symbol_aliases,
    )
    if (
        factor_expr.is_positive is not True
        and estimate._dependency_keys is not None
        and len(estimate._dependency_keys) > 1
        and factor_expr != _ZERO
    ):
        assumption = ResourceAssumption(
            "a possibly zero repetition uses a conservative shared "
            "wire-dependency summary",
            source=str(factor_expr),
        )
        repeated = _with_estimate_metadata(
            repeated,
            assumptions=(assumption,),
            quality=EstimateQuality.CONSERVATIVE,
            active_when=sp.And(
                active_when,
                _estimate_depth_activity_condition(estimate),
            ),
        )
    if conservative_nonuniform:
        repeated = _with_nonuniform_repetition_bound(
            repeated,
            estimate,
            active_when=sp.And(
                sp.Gt(factor_expr, _ONE),
                _estimate_depth_activity_condition(estimate),
            ),
            source=nonuniform_source,
        )
    synchronized_entry_active = _synchronized_entry_activity_condition(estimate)
    if (
        not conservative_nonuniform
        and synchronized_entry_active is not sp.false
        and estimate._dependency_completion_uniform is not True
    ):
        assumption = ResourceAssumption(
            "repeated aggregate depth may over-serialize a later invocation "
            "when the previous invocation leaves input wires at different layers",
            source="repeat",
        )
        repeated = _with_estimate_metadata(
            repeated,
            assumptions=(assumption,),
            quality=EstimateQuality.CONSERVATIVE,
            active_when=sp.And(
                sp.Gt(factor_expr, _ONE),
                synchronized_entry_active,
            ),
        )
    return repeated


def _with_nonuniform_repetition_bound(
    repeated: ResourceEstimate,
    estimate: ResourceEstimate,
    *,
    active_when: sp.Basic,
    source: str,
) -> ResourceEstimate:
    """Disclose scalar depth scaling over nonuniform wire completions.

    Args:
        repeated (ResourceEstimate): Already scaled or summed estimate.
        estimate (ResourceEstimate): One-iteration estimate whose dependency
            completion profile was scaled.
        active_when (sp.Basic): Condition under which at least two active
            iterations contribute.
        source (str): Assumption source label.

    Returns:
        ResourceEstimate: Estimate marked conservative exactly when the scalar
            depth may exceed a per-wire composition.
    """
    if (
        estimate._dependency_keys is None
        or estimate._dependency_completion_uniform is True
        or _has_single_serializing_quantum_dependency(estimate)
    ):
        return repeated
    assumption = ResourceAssumption(
        "repeated aggregate depth is a conservative scalar sum because "
        "per-wire completion layers cannot be composed exactly",
        source=source,
    )
    return _with_estimate_metadata(
        repeated,
        assumptions=(assumption,),
        quality=EstimateQuality.CONSERVATIVE,
        active_when=active_when,
    )


def _control_estimate(
    estimate: ResourceEstimate,
    num_controls: ResourceExpr | int,
) -> ResourceEstimate:
    """Project an aggregate estimate through coherent controls.

    Args:
        estimate (ResourceEstimate): Aggregate estimate to control.
        num_controls (ResourceExpr | int): Number of active controls.

    Returns:
        ResourceEstimate: Controlled aggregate estimate.

    Raises:
        ValueError: If a concrete control count or projected gate count is
            negative or non-integral, or if the estimate contains measurement
            or reset resources.
    """
    controls = _expr(num_controls)
    control_constraint = _ResourceConstraint(
        expression=controls,
        minimum=0,
        label="Aggregate controlled resource count",
        unit="control qubit",
    )
    control_constraint.validate()
    if controls == _ZERO:
        return estimate
    _require_unitary_resource_estimate(
        estimate,
        transform="coherently control",
    )
    profile_constraints = (
        *_aggregate_resource_profile_constraints(estimate, controls),
        *_aggregate_arity_projection_constraints(
            estimate,
            controls,
            model_label="Controlled",
        ),
    )
    if _safe_simplify(
        _estimate_activity(estimate)
    ) == _ZERO and not _estimate_has_control_sensitive_resources(estimate):
        if controls.is_number:
            return estimate
        return dataclasses.replace(
            estimate,
            _constraints=(
                *estimate._constraints,
                *profile_constraints,
                control_constraint,
            ),
        )
    if not _estimate_has_control_sensitive_resources(estimate):
        projected = None
        reason = _aggregate_arity_profile_reason(estimate) or (
            "the aggregate has no control-sensitive resource profile"
        )
    elif estimate.control_decomposition is ControlDecomposition.ABSTRACT:
        projected, reason = _project_abstract_aggregate_controlled_cost(
            estimate,
            controls,
        )
    else:
        projected, reason = _project_clean_ancilla_aggregate_controlled_cost(
            estimate,
            controls,
        )
    if projected is not None:
        projected = dataclasses.replace(
            projected,
            _constraints=(*projected._constraints, *profile_constraints),
        )
        if not controls.is_number:
            projected = estimate.conditional(
                projected,
                sp.Eq(controls, _ZERO),
            )
            projected = dataclasses.replace(
                projected,
                _constraints=(*projected._constraints, control_constraint),
                _output_sizes=estimate._output_sizes,
                _input_sizes=estimate._input_sizes,
                _has_output_summary=estimate._has_output_summary,
                _symbol_aliases=estimate._symbol_aliases,
            )
        return projected
    assumption = _unprojected_aggregate_control_assumption(
        reason,
        controls,
    )
    controlled = dataclasses.replace(
        estimate,
        trace=_wrap_trace(f"controlled({controls})", estimate.trace),
        parameters={},
        _constraints=(
            *estimate._constraints,
            *profile_constraints,
            *((control_constraint,) if not controls.is_number else ()),
        ),
    )
    return _with_estimate_metadata(
        controlled,
        assumptions=(assumption,),
        derivation=EstimateDerivation.MODELED,
        quality=EstimateQuality.UNKNOWN,
        active_when=_simplify_aggregate_metadata_guard(
            sp.And(
                sp.Gt(controls, _ZERO),
                sp.Or(
                    sp.Gt(estimate.gates.total, _ZERO),
                    _aggregate_zero_gate_residual_condition(estimate),
                ),
            )
        ),
    )


def _invert_estimate(estimate: ResourceEstimate) -> ResourceEstimate:
    """Apply an inverse transform to an aggregate estimate.

    Args:
        estimate (ResourceEstimate): Estimate to invert.

    Returns:
        ResourceEstimate: Estimate with identical logical resources.

    Raises:
        ValueError: If the estimate contains measurement or reset resources.
    """
    _require_unitary_resource_estimate(
        estimate,
        transform="invert",
    )
    return dataclasses.replace(
        estimate,
        trace=_wrap_trace("inverse", estimate.trace),
        _dependency_completion=None,
        _dependency_completion_uniform=None,
        _dependency_synchronized_entry_certificates=(
            _clear_synchronized_entry_frontiers(
                estimate._dependency_synchronized_entry_certificates
            )
        ),
    )
