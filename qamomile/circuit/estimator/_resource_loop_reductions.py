"""Reduce additive resource fields over symbolic Python-range loops."""

from __future__ import annotations

import sympy as sp

from qamomile.circuit.estimator._resource_base import ResourceExpr
from qamomile.circuit.estimator._resource_expressions import _sum_expr
from qamomile.circuit.estimator._resource_types import (
    CallResources,
    DepthResources,
    GateResources,
    MeasurementResources,
    ResetResources,
)


def _sum_gates(
    gates: GateResources,
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> GateResources:
    """Sum gate resources over a loop.

    Args:
        gates (GateResources): Gate resources.
        loop_symbol (sp.Symbol): Loop variable symbol.
        start (ResourceExpr): Start bound.
        step (ResourceExpr): Step value.
        iterations (ResourceExpr): Number of iterations.

    Returns:
        GateResources: Summed gate resources.
    """
    return GateResources(
        total=_sum_expr(gates.total, loop_symbol, start, step, iterations),
        single_qubit=_sum_expr(
            gates.single_qubit, loop_symbol, start, step, iterations
        ),
        two_qubit=_sum_expr(gates.two_qubit, loop_symbol, start, step, iterations),
        multi_qubit=_sum_expr(gates.multi_qubit, loop_symbol, start, step, iterations),
        clifford=_sum_expr(gates.clifford, loop_symbol, start, step, iterations),
        rotation=_sum_expr(gates.rotation, loop_symbol, start, step, iterations),
        t=_sum_expr(gates.t, loop_symbol, start, step, iterations),
        toffoli=_sum_expr(gates.toffoli, loop_symbol, start, step, iterations),
        non_clifford=_sum_expr(
            gates.non_clifford,
            loop_symbol,
            start,
            step,
            iterations,
        ),
    )


def _sum_depth(
    depth: DepthResources,
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> DepthResources:
    """Sum depth resources over a loop.

    Args:
        depth (DepthResources): Depth resources.
        loop_symbol (sp.Symbol): Loop variable symbol.
        start (ResourceExpr): Start bound.
        step (ResourceExpr): Step value.
        iterations (ResourceExpr): Number of iterations.

    Returns:
        DepthResources: Summed depth resources.
    """
    return DepthResources(
        depth=_sum_expr(depth.depth, loop_symbol, start, step, iterations),
        clifford_depth=_sum_expr(
            depth.clifford_depth,
            loop_symbol,
            start,
            step,
            iterations,
        ),
        rotation_depth=_sum_expr(
            depth.rotation_depth,
            loop_symbol,
            start,
            step,
            iterations,
        ),
        t_depth=_sum_expr(depth.t_depth, loop_symbol, start, step, iterations),
        toffoli_depth=_sum_expr(
            depth.toffoli_depth,
            loop_symbol,
            start,
            step,
            iterations,
        ),
        non_clifford_depth=_sum_expr(
            depth.non_clifford_depth,
            loop_symbol,
            start,
            step,
            iterations,
        ),
        measurement_depth=_sum_expr(
            depth.measurement_depth,
            loop_symbol,
            start,
            step,
            iterations,
        ),
        gate_depth=_sum_expr(
            depth.gate_depth,
            loop_symbol,
            start,
            step,
            iterations,
        ),
        reset_depth=_sum_expr(
            depth.reset_depth,
            loop_symbol,
            start,
            step,
            iterations,
        ),
    )


def _sum_measurements(
    measurements: MeasurementResources,
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> MeasurementResources:
    """Sum measurement resources over a loop.

    Args:
        measurements (MeasurementResources): Measurement resources.
        loop_symbol (sp.Symbol): Loop variable symbol.
        start (ResourceExpr): Start bound.
        step (ResourceExpr): Step value.
        iterations (ResourceExpr): Number of iterations.

    Returns:
        MeasurementResources: Summed measurement resources.
    """
    return MeasurementResources(
        total=_sum_expr(
            measurements.total,
            loop_symbol,
            start,
            step,
            iterations,
        )
    )


def _sum_resets(
    resets: ResetResources,
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> ResetResources:
    """Sum reset resources over a loop.

    Args:
        resets (ResetResources): Reset resources.
        loop_symbol (sp.Symbol): Loop variable symbol.
        start (ResourceExpr): Start bound.
        step (ResourceExpr): Step value.
        iterations (ResourceExpr): Number of iterations.

    Returns:
        ResetResources: Summed reset resources.
    """
    return ResetResources(
        total=_sum_expr(
            resets.total,
            loop_symbol,
            start,
            step,
            iterations,
        )
    )


def _sum_calls(
    calls: CallResources,
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
) -> CallResources:
    """Sum call resources over a loop.

    Args:
        calls (CallResources): Call resources.
        loop_symbol (sp.Symbol): Loop variable symbol.
        start (ResourceExpr): Start bound.
        step (ResourceExpr): Step value.
        iterations (ResourceExpr): Number of iterations.

    Returns:
        CallResources: Summed call resources.
    """
    return CallResources(
        calls_by_name={
            name: _sum_expr(value, loop_symbol, start, step, iterations)
            for name, value in calls.calls_by_name.items()
        },
        queries_by_name={
            name: _sum_expr(value, loop_symbol, start, step, iterations)
            for name, value in calls.queries_by_name.items()
        },
    )
