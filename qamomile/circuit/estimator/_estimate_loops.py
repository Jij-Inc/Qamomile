"""Aggregate resource estimates over symbolic Python-range loops."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, cast

import sympy as sp

from qamomile.circuit.estimator._allocation_width import (
    _maximum_width_over_range,
)
from qamomile.circuit.estimator._constants import _ZERO
from qamomile.circuit.estimator._dependency_metadata import (
    _project_dependency_metadata_over_symbol,
)
from qamomile.circuit.estimator._estimate_provenance import _with_estimate_metadata
from qamomile.circuit.estimator._estimate_transforms import _repeat_estimate
from qamomile.circuit.estimator._estimate_validation import _with_constraints
from qamomile.circuit.estimator._loop_executor import symbolic_iterations
from qamomile.circuit.estimator._resource_algebra import _wrap_trace
from qamomile.circuit.estimator._resource_base import EstimateQuality, ResourceExpr
from qamomile.circuit.estimator._resource_constraints import _ResourceConstraint
from qamomile.circuit.estimator._resource_expressions import (
    _activation_over_range,
    _boolean_condition,
    _ConditionIndicator,
    _resource_activity_condition,
    _sum_expr,
)
from qamomile.circuit.estimator._resource_loop_reductions import (
    _sum_calls,
    _sum_depth,
    _sum_gates,
    _sum_measurements,
    _sum_resets,
)
from qamomile.circuit.estimator._resource_types import ResourceAssumption
from qamomile.circuit.estimator._symbol_discovery import _free_symbols

if TYPE_CHECKING:
    from qamomile.circuit.estimator._estimate import ResourceEstimate


def _sum_estimate_over_range(
    estimate: ResourceEstimate,
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    stop: ResourceExpr,
    step: ResourceExpr,
    *,
    dependency_start: ResourceExpr,
    dependency_stop: ResourceExpr,
    dependency_step: ResourceExpr,
) -> ResourceEstimate:
    """Sum resources while specializing caller-visible wire projection.

    Args:
        estimate (ResourceEstimate): Per-iteration estimate.
        loop_symbol (sp.Symbol): Symbol used for the loop variable.
        start (ResourceExpr): Inclusive resource-expression start bound.
        stop (ResourceExpr): Exclusive resource-expression stop bound.
        step (ResourceExpr): Resource-expression loop step.
        dependency_start (ResourceExpr): Start bound specialized only for
            caller-visible wire projection.
        dependency_stop (ResourceExpr): Stop bound specialized only for
            caller-visible wire projection.
        dependency_step (ResourceExpr): Step specialized only for
            caller-visible wire projection.

    Returns:
        ResourceEstimate: Estimate with additive metrics summed over the loop
            and width kept reusable.
    """
    step_constraint = _ResourceConstraint(
        expression=sp.Abs(step),
        minimum=1,
        label="Loop step magnitude",
    )
    step_constraint.validate()
    iterations = symbolic_iterations(start, stop, step)
    projected_iterations = symbolic_iterations(
        dependency_start,
        dependency_stop,
        dependency_step,
    )
    if loop_symbol not in _free_symbols(estimate):
        return _project_dependency_metadata_over_symbol(
            _with_constraints(
                _repeat_estimate(estimate, iterations),
                step_constraint,
            ),
            loop_symbol,
            start=dependency_start,
            step=dependency_step,
            iterations=projected_iterations,
        )
    width, allocation_sites, width_conservative_when = _maximum_width_over_range(
        estimate.width,
        estimate._allocation_sites,
        loop_symbol,
        start,
        step,
        iterations,
    )
    summed = dataclasses.replace(
        estimate.zero(),
        width=width,
        gates=_sum_gates(estimate.gates, loop_symbol, start, step, iterations),
        depth=_sum_depth(estimate.depth, loop_symbol, start, step, iterations),
        calls=_sum_calls(estimate.calls, loop_symbol, start, step, iterations),
        measurements=_sum_measurements(
            estimate.measurements,
            loop_symbol,
            start,
            step,
            iterations,
        ),
        resets=_sum_resets(
            estimate.resets,
            loop_symbol,
            start,
            step,
            iterations,
        ),
        trace=_wrap_trace(
            f"sum({loop_symbol}={start}..{stop})",
            estimate.trace.when(sp.Gt(iterations, _ZERO))
            if estimate.trace is not None
            else None,
        ),
        control_decomposition=estimate.control_decomposition,
        _allocation_sites=allocation_sites,
        _constraints=(
            step_constraint,
            *(
                constraint.bound_over(
                    loop_symbol,
                    start,
                    step,
                    iterations,
                ).when(sp.Gt(iterations, _ZERO))
                for constraint in estimate._constraints
            ),
        ),
        _dependency_keys=estimate._dependency_keys,
        _dependency_reads=estimate._dependency_reads,
        _dependency_writes=estimate._dependency_writes,
        _dependency_completion=(
            {
                key: cast(
                    ResourceExpr,
                    _ConditionIndicator(
                        _activation_over_range(
                            _resource_activity_condition(completion),
                            loop_symbol,
                            start,
                            step,
                            iterations,
                        )
                    )
                    * _sum_expr(
                        estimate.depth.depth,
                        loop_symbol,
                        start,
                        step,
                        iterations,
                    ),
                )
                for key, completion in estimate._dependency_completion.items()
            }
            if estimate._dependency_completion is not None
            else None
        ),
        _global_barrier_condition=_boolean_condition(
            _activation_over_range(
                estimate._global_barrier_condition,
                loop_symbol,
                start,
                step,
                iterations,
            )
        ),
        _measurement_taint_conditions={
            uuid: _boolean_condition(
                _activation_over_range(
                    condition,
                    loop_symbol,
                    start,
                    step,
                    iterations,
                )
            )
            for uuid, condition in estimate._measurement_taint_conditions.items()
        },
        _guarded_assumptions=tuple(
            dataclasses.replace(
                fact,
                active_when=_activation_over_range(
                    fact.active_when,
                    loop_symbol,
                    start,
                    step,
                    iterations,
                ),
            )
            for fact in (estimate._guarded_assumptions or ())
        ),
        _guarded_derivations=tuple(
            dataclasses.replace(
                fact,
                active_when=_activation_over_range(
                    fact.active_when,
                    loop_symbol,
                    start,
                    step,
                    iterations,
                ),
            )
            for fact in (estimate._guarded_derivations or ())
        ),
        _guarded_qualities=tuple(
            dataclasses.replace(
                fact,
                active_when=_activation_over_range(
                    fact.active_when,
                    loop_symbol,
                    start,
                    step,
                    iterations,
                ),
            )
            for fact in (estimate._guarded_qualities or ())
        ),
        _guarded_approximations=tuple(
            dataclasses.replace(
                fact,
                active_when=_activation_over_range(
                    fact.active_when,
                    loop_symbol,
                    start,
                    step,
                    iterations,
                ),
            )
            for fact in (estimate._guarded_approximations or ())
        ),
        _symbol_aliases=estimate._symbol_aliases,
    )
    if width_conservative_when is not sp.false:
        summed = _with_estimate_metadata(
            summed,
            assumptions=(
                ResourceAssumption(
                    "symbolic loop width uses a conservative excess-sum "
                    "bound because its maximum could not be proven",
                    source=str(loop_symbol),
                ),
            ),
            quality=EstimateQuality.CONSERVATIVE,
            active_when=width_conservative_when,
        )
    return _project_dependency_metadata_over_symbol(
        summed,
        loop_symbol,
        start=dependency_start,
        step=dependency_step,
        iterations=projected_iterations,
    )
