"""Model critical-path depth for Clifford+T decompositions."""

from __future__ import annotations

import dataclasses

import sympy as sp

from qamomile.circuit.estimator._clifford_t_decomposition import (
    _classify_clifford_t_gate,
    _classify_uncontrolled_clifford_t_gate,
    _clifford_t_mcx_toffoli_count,
)
from qamomile.circuit.estimator._constants import (
    _ONE,
    _ZERO,
)
from qamomile.circuit.estimator._gate_classification import (
    _serial_depth_from_gate_resources,
)
from qamomile.circuit.estimator._resource_algebra import (
    _conditional_depth,
)
from qamomile.circuit.estimator._resource_base import (
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_expressions import (
    _resource_expr,
)
from qamomile.circuit.estimator._resource_types import (
    DepthResources,
    GateResources,
)
from qamomile.circuit.ir.operation.gate import (
    GateOperation,
)
from qamomile.circuit.transpiler.passes.emit_support.clean_ancilla_toffoli import (
    clean_ancilla_toffoli_ladder_or_empty,
)


def _named_clifford_t_depth(
    name: str,
    surrounding_controls: ResourceExpr,
    gates: GateResources,
    precision: float,
) -> DepthResources:
    """Return canonical depth for an estimator-introduced Clifford+T gate.

    Args:
        name (str): Lowercase gate name.
        surrounding_controls (ResourceExpr): Additional coherent controls.
        gates (GateResources): Already-classified aggregate gate counts.
        precision (float): Rotation-synthesis precision used to classify
            ``gates``.

    Returns:
        DepthResources: Canonical critical-path depth where available,
        otherwise a conservative serial depth.
    """
    inherent_controls = {"x": 0, "cx": 1, "toffoli": 2}
    if name in inherent_controls:
        return _clifford_t_gate_depth_for_mcx(
            surrounding_controls + inherent_controls[name]
        )
    if name == "swap":
        if surrounding_controls == _ZERO:
            return DepthResources(
                depth=sp.Integer(3),
                clifford_depth=sp.Integer(3),
                gate_depth=sp.Integer(3),
            )
        middle = _clifford_t_gate_depth_for_mcx(surrounding_controls + _ONE)
        return dataclasses.replace(
            middle,
            depth=middle.depth + 2,
            clifford_depth=middle.clifford_depth + 2,
            gate_depth=middle.gate_depth + 2,
        )
    if name in {"z", "y"}:
        if surrounding_controls == _ZERO:
            return _serial_depth_from_gate_resources(gates)
        middle = _clifford_t_gate_depth_for_mcx(surrounding_controls)
        controlled = dataclasses.replace(
            middle,
            depth=middle.depth + 2,
            clifford_depth=middle.clifford_depth + 2,
            gate_depth=middle.gate_depth + 2,
        )
        if surrounding_controls.is_number:
            return controlled
        return _conditional_depth(
            DepthResources(
                depth=_ONE,
                clifford_depth=_ONE,
                gate_depth=_ONE,
            ),
            controlled,
            sp.Eq(surrounding_controls, _ZERO),
        )
    if name == "cz":
        middle = _clifford_t_gate_depth_for_mcx(surrounding_controls + _ONE)
        return dataclasses.replace(
            middle,
            depth=middle.depth + 2,
            clifford_depth=middle.clifford_depth + 2,
            gate_depth=middle.gate_depth + 2,
        )
    if name in {"p", "cp"}:
        effective_controls = surrounding_controls + (_ONE if name == "cp" else _ZERO)
        ladder_steps = clean_ancilla_toffoli_ladder_or_empty(
            effective_controls
        ).total_toffolis
        rotation_t = _classify_uncontrolled_clifford_t_gate("p", precision).t
        controlled = _clifford_t_cp_depth(
            ladder_steps,
            rotation_t,
        )
        if name == "cp":
            return controlled
        uncontrolled = _serial_depth_from_gate_resources(
            _classify_uncontrolled_clifford_t_gate("p", precision)
        )
        if surrounding_controls == _ZERO:
            return uncontrolled
        if surrounding_controls.is_number:
            return controlled
        return _conditional_depth(
            uncontrolled,
            controlled,
            sp.Eq(surrounding_controls, _ZERO),
        )
    if name in {"s", "sdg"}:
        if surrounding_controls == _ZERO:
            return _serial_depth_from_gate_resources(gates)
        ladder_steps = clean_ancilla_toffoli_ladder_or_empty(
            surrounding_controls
        ).total_toffolis
        controlled = DepthResources(
            depth=15 * ladder_steps + 4,
            clifford_depth=8 * ladder_steps + 2,
            t_depth=3 * ladder_steps + 2,
            non_clifford_depth=3 * ladder_steps + 2,
            gate_depth=15 * ladder_steps + 4,
        )
        if surrounding_controls.is_number:
            return controlled
        return _conditional_depth(
            DepthResources(
                depth=_ONE,
                clifford_depth=_ONE,
                gate_depth=_ONE,
            ),
            controlled,
            sp.Eq(surrounding_controls, _ZERO),
        )
    if name in {"t", "tdg"}:
        ladder_steps = clean_ancilla_toffoli_ladder_or_empty(
            surrounding_controls + _ONE
        ).total_toffolis
        return DepthResources(
            depth=15 * ladder_steps + 1,
            clifford_depth=8 * ladder_steps,
            t_depth=3 * ladder_steps + 1,
            non_clifford_depth=3 * ladder_steps + 1,
            gate_depth=15 * ladder_steps + 1,
        )
    return _serial_depth_from_gate_resources(gates)


def _clifford_t_cp_depth(
    ladder_steps: ResourceExpr,
    rotation_t: ResourceExpr,
) -> DepthResources:
    """Return depth for a Toffoli ladder around one synthesized CP gate.

    A CP decomposition uses three axial rotations and two CX gates. The two
    same-sign rotations can occupy one parallel layer, so the central CP has
    T-depth ``2 * rotation_t`` rather than ``3 * rotation_t``.

    Args:
        ladder_steps (ResourceExpr): Number of serial Toffoli gates used to
            compute and uncompute the surrounding control conjunction.
        rotation_t (ResourceExpr): T count and T-depth of one synthesized
            axial rotation.

    Returns:
        DepthResources: Aggregate and category depth of the complete
            controlled-phase recipe.
    """
    central_depth = 2 * rotation_t + 2
    central_t_depth = 2 * rotation_t
    return DepthResources(
        depth=15 * ladder_steps + central_depth,
        clifford_depth=8 * ladder_steps + 2,
        t_depth=3 * ladder_steps + central_t_depth,
        non_clifford_depth=3 * ladder_steps + central_t_depth,
        gate_depth=15 * ladder_steps + central_depth,
    )


def _clifford_t_gate_depth(
    operation: GateOperation,
    surrounding_controls: ResourceExpr,
    precision: float,
) -> DepthResources:
    """Return depth for the selected canonical Clifford+T decomposition.

    Args:
        operation (GateOperation): Logical primitive being lowered.
        surrounding_controls (ResourceExpr): Additional enclosing controls.
        precision (float): Rotation-synthesis precision.

    Returns:
        DepthResources: Conservative decomposition critical path.
    """
    name = operation.gate_type.name.lower() if operation.gate_type else "unknown"
    if name == "ccx":
        name = "toffoli"
    gates = _classify_clifford_t_gate(name, surrounding_controls, precision)
    return _named_clifford_t_depth(
        name,
        surrounding_controls,
        gates,
        precision,
    )


def _clifford_t_gate_depth_for_mcx(
    controls: ResourceExpr,
) -> DepthResources:
    """Return the Toffoli-ladder depth for a multi-controlled X.

    Args:
        controls (ResourceExpr): Number of controls.

    Returns:
        DepthResources: Canonical ladder depth.
    """
    toffolis = _clifford_t_mcx_toffoli_count(controls)
    return DepthResources(
        depth=_resource_expr(
            sp.Piecewise(
                (_ONE, controls <= 1),
                (sp.Integer(15), sp.Eq(controls, 2)),
                (15 * toffolis + _ONE, True),
            )
        ),
        clifford_depth=_resource_expr(
            sp.Piecewise(
                (_ONE, controls <= 1),
                (sp.Integer(8), sp.Eq(controls, 2)),
                (8 * toffolis + _ONE, True),
            )
        ),
        t_depth=_resource_expr(
            sp.Piecewise(
                (_ZERO, controls <= 1),
                (3 * toffolis, True),
            )
        ),
        non_clifford_depth=_resource_expr(
            sp.Piecewise(
                (_ZERO, controls <= 1),
                (3 * toffolis, True),
            )
        ),
        gate_depth=_resource_expr(
            sp.Piecewise(
                (_ONE, controls <= 1),
                (sp.Integer(15), sp.Eq(controls, 2)),
                (15 * toffolis + _ONE, True),
            )
        ),
    )
