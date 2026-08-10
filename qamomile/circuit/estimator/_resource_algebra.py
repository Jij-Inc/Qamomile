"""Compose resource fields conditionally, serially, and over ranges."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import cast

import sympy as sp

from qamomile.circuit.estimator._constants import _ONE, _ZERO
from qamomile.circuit.estimator._resource_base import (
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_expressions import (
    _add_maps,
    _piecewise,
    _resource_max,
)
from qamomile.circuit.estimator._resource_types import (
    CallResources,
    DepthResources,
    GateResources,
    MeasurementResources,
    ResetResources,
    ResourceTraceNode,
    WidthResources,
)


def _conditional_width(
    true_value: WidthResources,
    false_value: WidthResources,
    condition: sp.Basic,
) -> WidthResources:
    """Select width resources with a symbolic condition.

    Args:
        true_value (WidthResources): True-branch width.
        false_value (WidthResources): False-branch width.
        condition (sp.Basic): SymPy Boolean predicate.

    Returns:
        WidthResources: Field-wise piecewise width.
    """
    return WidthResources(
        **{
            field.name: _piecewise(
                getattr(true_value, field.name),
                getattr(false_value, field.name),
                condition,
            )
            for field in dataclasses.fields(WidthResources)
        }
    )


def _conditional_gates(
    true_value: GateResources,
    false_value: GateResources,
    condition: sp.Basic,
) -> GateResources:
    """Select gate resources with a symbolic condition.

    Args:
        true_value (GateResources): True-branch gates.
        false_value (GateResources): False-branch gates.
        condition (sp.Basic): SymPy Boolean predicate.

    Returns:
        GateResources: Field-wise piecewise gate resources.
    """
    return GateResources(
        **{
            field.name: _piecewise(
                getattr(true_value, field.name),
                getattr(false_value, field.name),
                condition,
            )
            for field in dataclasses.fields(GateResources)
        }
    )


def _conditional_measurements(
    true_value: MeasurementResources,
    false_value: MeasurementResources,
    condition: sp.Basic,
) -> MeasurementResources:
    """Select measurement resources with a symbolic condition.

    Args:
        true_value (MeasurementResources): True-branch measurements.
        false_value (MeasurementResources): False-branch measurements.
        condition (sp.Basic): SymPy Boolean predicate.

    Returns:
        MeasurementResources: Field-wise piecewise measurement resources.
    """
    return MeasurementResources(
        total=_piecewise(
            true_value.total,
            false_value.total,
            condition,
        )
    )


def _conditional_resets(
    true_value: ResetResources,
    false_value: ResetResources,
    condition: sp.Basic,
) -> ResetResources:
    """Select reset resources with a symbolic condition.

    Args:
        true_value (ResetResources): True-branch resets.
        false_value (ResetResources): False-branch resets.
        condition (sp.Basic): SymPy Boolean predicate.

    Returns:
        ResetResources: Field-wise piecewise reset resources.
    """
    return ResetResources(
        total=_piecewise(
            true_value.total,
            false_value.total,
            condition,
        )
    )


def _conditional_depth(
    true_value: DepthResources,
    false_value: DepthResources,
    condition: sp.Basic,
) -> DepthResources:
    """Select depth resources with a symbolic condition.

    Args:
        true_value (DepthResources): True-branch depth.
        false_value (DepthResources): False-branch depth.
        condition (sp.Basic): SymPy Boolean predicate.

    Returns:
        DepthResources: Field-wise piecewise depth resources.
    """
    return DepthResources(
        **{
            field.name: _piecewise(
                getattr(true_value, field.name),
                getattr(false_value, field.name),
                condition,
            )
            for field in dataclasses.fields(DepthResources)
        }
    )


def _conditional_calls(
    true_value: CallResources,
    false_value: CallResources,
    condition: sp.Basic,
) -> CallResources:
    """Select callable counts with a symbolic condition.

    Args:
        true_value (CallResources): True-branch calls.
        false_value (CallResources): False-branch calls.
        condition (sp.Basic): SymPy Boolean predicate.

    Returns:
        CallResources: Key-wise piecewise callable counts.
    """

    def select_maps(
        true_map: Mapping[str, ResourceExpr],
        false_map: Mapping[str, ResourceExpr],
    ) -> dict[str, ResourceExpr]:
        """Select two call-count mappings key by key.

        Args:
            true_map (Mapping[str, ResourceExpr]): True-branch mapping.
            false_map (Mapping[str, ResourceExpr]): False-branch mapping.

        Returns:
            dict[str, ResourceExpr]: Piecewise mapping over the union of keys.
        """
        return {
            name: _piecewise(
                true_map.get(name, _ZERO),
                false_map.get(name, _ZERO),
                condition,
            )
            for name in sorted(set(true_map) | set(false_map))
        }

    return CallResources(
        calls_by_name=select_maps(
            true_value.calls_by_name,
            false_value.calls_by_name,
        ),
        queries_by_name=select_maps(
            true_value.queries_by_name,
            false_value.queries_by_name,
        ),
    )


def _max_maps(
    left: Mapping[str, ResourceExpr],
    right: Mapping[str, ResourceExpr],
) -> dict[str, ResourceExpr]:
    """Take key-wise maxima of two expression dictionaries.

    Args:
        left (Mapping[str, ResourceExpr]): Left mapping.
        right (Mapping[str, ResourceExpr]): Right mapping.

    Returns:
        dict[str, ResourceExpr]: Merged mapping.
    """
    merged: dict[str, ResourceExpr] = {}
    for key in sorted(set(left) | set(right)):
        merged[key] = sp.Max(left.get(key, _ZERO), right.get(key, _ZERO))
    return merged


def _add_gates(left: GateResources, right: GateResources) -> GateResources:
    """Add gate resources.

    Args:
        left (GateResources): Left resources.
        right (GateResources): Right resources.

    Returns:
        GateResources: Sum.
    """
    return GateResources(
        total=left.total + right.total,
        single_qubit=left.single_qubit + right.single_qubit,
        two_qubit=left.two_qubit + right.two_qubit,
        multi_qubit=left.multi_qubit + right.multi_qubit,
        clifford=left.clifford + right.clifford,
        rotation=left.rotation + right.rotation,
        t=left.t + right.t,
        toffoli=left.toffoli + right.toffoli,
        non_clifford=left.non_clifford + right.non_clifford,
    )


def _max_gates(left: GateResources, right: GateResources) -> GateResources:
    """Take element-wise maxima of gate resources.

    Args:
        left (GateResources): Left resources.
        right (GateResources): Right resources.

    Returns:
        GateResources: Element-wise maximum.
    """
    return GateResources(
        total=sp.Max(left.total, right.total),
        single_qubit=sp.Max(left.single_qubit, right.single_qubit),
        two_qubit=sp.Max(left.two_qubit, right.two_qubit),
        multi_qubit=sp.Max(left.multi_qubit, right.multi_qubit),
        clifford=sp.Max(left.clifford, right.clifford),
        rotation=sp.Max(left.rotation, right.rotation),
        t=sp.Max(left.t, right.t),
        toffoli=sp.Max(left.toffoli, right.toffoli),
        non_clifford=sp.Max(left.non_clifford, right.non_clifford),
    )


def _scale_gates(gates: GateResources, factor: ResourceExpr) -> GateResources:
    """Scale gate resources.

    Args:
        gates (GateResources): Gate resources.
        factor (ResourceExpr): Multiplicative factor.

    Returns:
        GateResources: Scaled resources.
    """
    return GateResources(
        total=gates.total * factor,
        single_qubit=gates.single_qubit * factor,
        two_qubit=gates.two_qubit * factor,
        multi_qubit=gates.multi_qubit * factor,
        clifford=gates.clifford * factor,
        rotation=gates.rotation * factor,
        t=gates.t * factor,
        toffoli=gates.toffoli * factor,
        non_clifford=gates.non_clifford * factor,
    )


def _add_measurements(
    left: MeasurementResources,
    right: MeasurementResources,
) -> MeasurementResources:
    """Add measurement resources.

    Args:
        left (MeasurementResources): Left resources.
        right (MeasurementResources): Right resources.

    Returns:
        MeasurementResources: Sum.
    """
    return MeasurementResources(total=left.total + right.total)


def _max_measurements(
    left: MeasurementResources,
    right: MeasurementResources,
) -> MeasurementResources:
    """Take element-wise maxima of measurement resources.

    Args:
        left (MeasurementResources): Left resources.
        right (MeasurementResources): Right resources.

    Returns:
        MeasurementResources: Element-wise maximum.
    """
    return MeasurementResources(total=sp.Max(left.total, right.total))


def _scale_measurements(
    measurements: MeasurementResources,
    factor: ResourceExpr,
) -> MeasurementResources:
    """Scale measurement resources.

    Args:
        measurements (MeasurementResources): Measurement resources.
        factor (ResourceExpr): Multiplicative factor.

    Returns:
        MeasurementResources: Scaled resources.
    """
    return MeasurementResources(total=measurements.total * factor)


def _add_resets(
    left: ResetResources,
    right: ResetResources,
) -> ResetResources:
    """Add reset resources.

    Args:
        left (ResetResources): Left resources.
        right (ResetResources): Right resources.

    Returns:
        ResetResources: Sum.
    """
    return ResetResources(total=left.total + right.total)


def _max_resets(
    left: ResetResources,
    right: ResetResources,
) -> ResetResources:
    """Take element-wise maxima of reset resources.

    Args:
        left (ResetResources): Left resources.
        right (ResetResources): Right resources.

    Returns:
        ResetResources: Element-wise maximum.
    """
    return ResetResources(total=sp.Max(left.total, right.total))


def _scale_resets(
    resets: ResetResources,
    factor: ResourceExpr,
) -> ResetResources:
    """Scale reset resources.

    Args:
        resets (ResetResources): Reset resources.
        factor (ResourceExpr): Multiplicative factor.

    Returns:
        ResetResources: Scaled resources.
    """
    return ResetResources(total=resets.total * factor)


def _add_depth(left: DepthResources, right: DepthResources) -> DepthResources:
    """Add depth resources.

    Args:
        left (DepthResources): Left resources.
        right (DepthResources): Right resources.

    Returns:
        DepthResources: Sum.
    """
    return DepthResources(
        depth=left.depth + right.depth,
        clifford_depth=left.clifford_depth + right.clifford_depth,
        rotation_depth=left.rotation_depth + right.rotation_depth,
        t_depth=left.t_depth + right.t_depth,
        toffoli_depth=left.toffoli_depth + right.toffoli_depth,
        non_clifford_depth=left.non_clifford_depth + right.non_clifford_depth,
        measurement_depth=left.measurement_depth + right.measurement_depth,
        gate_depth=left.gate_depth + right.gate_depth,
        reset_depth=left.reset_depth + right.reset_depth,
    )


def _max_depth(left: DepthResources, right: DepthResources) -> DepthResources:
    """Take element-wise maxima of depth resources.

    Args:
        left (DepthResources): Left resources.
        right (DepthResources): Right resources.

    Returns:
        DepthResources: Element-wise maximum.
    """
    return DepthResources(
        depth=sp.Max(left.depth, right.depth),
        clifford_depth=sp.Max(left.clifford_depth, right.clifford_depth),
        rotation_depth=sp.Max(left.rotation_depth, right.rotation_depth),
        t_depth=sp.Max(left.t_depth, right.t_depth),
        toffoli_depth=sp.Max(left.toffoli_depth, right.toffoli_depth),
        non_clifford_depth=sp.Max(left.non_clifford_depth, right.non_clifford_depth),
        measurement_depth=sp.Max(left.measurement_depth, right.measurement_depth),
        gate_depth=sp.Max(left.gate_depth, right.gate_depth),
        reset_depth=sp.Max(left.reset_depth, right.reset_depth),
    )


def _scale_depth(depth: DepthResources, factor: ResourceExpr) -> DepthResources:
    """Scale depth resources.

    Args:
        depth (DepthResources): Depth resources.
        factor (ResourceExpr): Multiplicative factor.

    Returns:
        DepthResources: Scaled resources.
    """
    return DepthResources(
        depth=depth.depth * factor,
        clifford_depth=depth.clifford_depth * factor,
        rotation_depth=depth.rotation_depth * factor,
        t_depth=depth.t_depth * factor,
        toffoli_depth=depth.toffoli_depth * factor,
        non_clifford_depth=depth.non_clifford_depth * factor,
        measurement_depth=depth.measurement_depth * factor,
        gate_depth=depth.gate_depth * factor,
        reset_depth=depth.reset_depth * factor,
    )


def _add_calls(left: CallResources, right: CallResources) -> CallResources:
    """Add call resources.

    Args:
        left (CallResources): Left resources.
        right (CallResources): Right resources.

    Returns:
        CallResources: Sum.
    """
    return CallResources(
        calls_by_name=_add_maps(left.calls_by_name, right.calls_by_name),
        queries_by_name=_add_maps(left.queries_by_name, right.queries_by_name),
    )


def _max_calls(left: CallResources, right: CallResources) -> CallResources:
    """Take key-wise maxima of call resources.

    Args:
        left (CallResources): Left resources.
        right (CallResources): Right resources.

    Returns:
        CallResources: Element-wise maximum.
    """
    return CallResources(
        calls_by_name=_max_maps(left.calls_by_name, right.calls_by_name),
        queries_by_name=_max_maps(left.queries_by_name, right.queries_by_name),
    )


def _scale_calls(calls: CallResources, factor: ResourceExpr) -> CallResources:
    """Scale call resources.

    Args:
        calls (CallResources): Call resources.
        factor (ResourceExpr): Multiplicative factor.

    Returns:
        CallResources: Scaled resources.
    """
    return CallResources(
        calls_by_name={
            name: value * factor for name, value in calls.calls_by_name.items()
        },
        queries_by_name={
            name: value * factor for name, value in calls.queries_by_name.items()
        },
    )


def _seq_width(left: WidthResources, right: WidthResources) -> WidthResources:
    """Compose width resources sequentially.

    Args:
        left (WidthResources): Left resources.
        right (WidthResources): Right resources.

    Returns:
        WidthResources: Sequential width estimate.
    """
    allocated = left.allocated_qubits + right.allocated_qubits
    return WidthResources(
        input_qubits=sp.Max(left.input_qubits, right.input_qubits),
        allocated_qubits=allocated,
        clean_ancilla_qubits=_resource_max(
            left.clean_ancilla_qubits,
            right.clean_ancilla_qubits,
        ),
        dirty_ancilla_qubits=_resource_max(
            left.dirty_ancilla_qubits,
            right.dirty_ancilla_qubits,
        ),
        peak_qubits=sp.Max(left.peak_qubits, left.allocated_qubits + right.peak_qubits),
    )


def _parallel_width(left: WidthResources, right: WidthResources) -> WidthResources:
    """Compose width resources in parallel.

    Args:
        left (WidthResources): Left resources.
        right (WidthResources): Right resources.

    Returns:
        WidthResources: Parallel width estimate.
    """
    return WidthResources(
        input_qubits=left.input_qubits + right.input_qubits,
        allocated_qubits=left.allocated_qubits + right.allocated_qubits,
        clean_ancilla_qubits=left.clean_ancilla_qubits + right.clean_ancilla_qubits,
        dirty_ancilla_qubits=left.dirty_ancilla_qubits + right.dirty_ancilla_qubits,
        peak_qubits=left.peak_qubits + right.peak_qubits,
    )


def _max_width(left: WidthResources, right: WidthResources) -> WidthResources:
    """Take element-wise maxima of width resources.

    Args:
        left (WidthResources): Left resources.
        right (WidthResources): Right resources.

    Returns:
        WidthResources: Element-wise maximum.
    """
    return WidthResources(
        input_qubits=sp.Max(left.input_qubits, right.input_qubits),
        allocated_qubits=sp.Max(left.allocated_qubits, right.allocated_qubits),
        clean_ancilla_qubits=_resource_max(
            left.clean_ancilla_qubits,
            right.clean_ancilla_qubits,
        ),
        dirty_ancilla_qubits=_resource_max(
            left.dirty_ancilla_qubits,
            right.dirty_ancilla_qubits,
        ),
        peak_qubits=sp.Max(left.peak_qubits, right.peak_qubits),
    )


def _active_gate_family_depth(
    value: ResourceExpr,
    *,
    active: ResourceExpr,
) -> ResourceExpr:
    """Return one active layer when a gate-family count is positive.

    Args:
        value (ResourceExpr): Possibly symbolic gate-family count.
        active (ResourceExpr): Overall primitive activation indicator.

    Returns:
        ResourceExpr: Overall activation guarded by a positive family count.
    """
    if value == _ZERO:
        return _ZERO
    if value == _ONE:
        return active
    return _piecewise(active, _ZERO, sp.Gt(value, _ZERO))


def _depth_from_gate_resources(gates: GateResources) -> DepthResources:
    """Create a primitive depth estimate from gate resources.

    Args:
        gates (GateResources): Primitive gate resources.

    Returns:
        DepthResources: One-layer depth categorized by gate type.
    """
    active = cast(
        ResourceExpr,
        sp.Piecewise((_ONE, sp.Gt(gates.total, 0)), (_ZERO, True)),
    )
    return DepthResources(
        depth=active,
        clifford_depth=_active_gate_family_depth(gates.clifford, active=active),
        rotation_depth=_active_gate_family_depth(gates.rotation, active=active),
        t_depth=_active_gate_family_depth(gates.t, active=active),
        toffoli_depth=_active_gate_family_depth(gates.toffoli, active=active),
        non_clifford_depth=_active_gate_family_depth(
            gates.non_clifford,
            active=active,
        ),
        gate_depth=active,
    )


def _merge_trace(
    name: str,
    left: ResourceTraceNode | None,
    right: ResourceTraceNode | None,
) -> ResourceTraceNode | None:
    """Merge two trace nodes under a parent.

    Args:
        name (str): Parent node name.
        left (ResourceTraceNode | None): Left child.
        right (ResourceTraceNode | None): Right child.

    Returns:
        ResourceTraceNode | None: Parent node or ``None`` when both children
        are absent.
    """
    children = tuple(child for child in (left, right) if child is not None)
    if not children:
        return None
    if len(children) == 1 and name == "seq":
        return children[0]
    return ResourceTraceNode(name=name, source_kind="algebra", children=children)


def _conditional_trace(
    condition: sp.Basic,
    true_trace: ResourceTraceNode | None,
    false_trace: ResourceTraceNode | None,
) -> ResourceTraceNode | None:
    """Merge two trace branches with structured activation guards.

    Args:
        condition (sp.Basic): Predicate selecting ``true_trace``.
        true_trace (ResourceTraceNode | None): True-branch trace.
        false_trace (ResourceTraceNode | None): False-branch trace.

    Returns:
        ResourceTraceNode | None: Conditional trace, or ``None`` when neither
            branch carries a trace.
    """
    guarded_true = true_trace.when(condition) if true_trace is not None else None
    guarded_false = (
        false_trace.when(sp.Not(condition)) if false_trace is not None else None
    )
    return _merge_trace(
        f"if[{condition}]",
        guarded_true,
        guarded_false,
    )


def _wrap_trace(
    name: str,
    child: ResourceTraceNode | None,
    *,
    source_kind: str = "algebra",
    strategy: str | None = None,
) -> ResourceTraceNode:
    """Wrap an optional child trace with a parent node.

    Args:
        name (str): Parent node name.
        child (ResourceTraceNode | None): Optional child.
        source_kind (str): Parent source kind. Defaults to ``"algebra"``.
        strategy (str | None): Selected strategy. Defaults to ``None``.

    Returns:
        ResourceTraceNode: Parent trace node.
    """
    children = (child,) if child is not None else ()
    return ResourceTraceNode(
        name=name,
        source_kind=source_kind,
        strategy=strategy,
        children=children,
    )
