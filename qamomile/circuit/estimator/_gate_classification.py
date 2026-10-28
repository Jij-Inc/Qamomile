"""Classify logical gate resources and serial gate depths."""

from __future__ import annotations

from typing import cast

import sympy as sp

from qamomile.circuit.estimator._constants import (
    _ONE,
    _ZERO,
)
from qamomile.circuit.estimator._gate_catalog import (
    _CLIFFORD_GATES,
    _CONTROLLED_CLIFFORD_GATES,
    _GATE_BASE_QUBITS,
    _MULTI_QUBIT_GATES,
    _ROTATION_GATES,
    _SINGLE_QUBIT_GATES,
    _T_GATES,
    _TWO_QUBIT_GATES,
)
from qamomile.circuit.estimator._resource_base import (
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_types import (
    DepthResources,
    GateResources,
)


def _serial_depth_from_gate_resources(gates: GateResources) -> DepthResources:
    """Return a conservative serial depth for a logical gate sequence.

    Args:
        gates (GateResources): Aggregate resources for a sequence whose
            decomposition steps share controls or targets.

    Returns:
        DepthResources: Gate-family counts interpreted as serial depths.
    """
    return DepthResources(
        depth=gates.total,
        clifford_depth=gates.clifford,
        rotation_depth=gates.rotation,
        t_depth=gates.t,
        toffoli_depth=gates.toffoli,
        non_clifford_depth=gates.non_clifford,
        gate_depth=gates.total,
    )


def _classify_uncontrolled_gate(gate_name: str) -> GateResources:
    """Classify an uncontrolled primitive gate.

    Args:
        gate_name (str): Lowercase gate name.

    Returns:
        GateResources: Resource contribution of the gate.
    """
    clifford = _ONE if gate_name in _CLIFFORD_GATES else _ZERO
    rotation = _ONE if gate_name in _ROTATION_GATES else _ZERO
    t_count = _ONE if gate_name in _T_GATES else _ZERO
    multi = _ONE if gate_name in _MULTI_QUBIT_GATES else _ZERO
    return GateResources(
        total=_ONE,
        single_qubit=_ONE if gate_name in _SINGLE_QUBIT_GATES else _ZERO,
        two_qubit=_ONE if gate_name in _TWO_QUBIT_GATES else _ZERO,
        multi_qubit=multi,
        clifford=clifford,
        rotation=rotation,
        t=t_count,
        toffoli=_ONE if gate_name == "toffoli" else _ZERO,
        non_clifford=_ONE - clifford,
    )


def _select_zero_control_resource(
    uncontrolled: ResourceExpr,
    controlled: ResourceExpr,
    num_controls: ResourceExpr,
) -> ResourceExpr:
    """Select an uncontrolled field when a symbolic control count is zero.

    Args:
        uncontrolled (ResourceExpr): Resource field for the base primitive.
        controlled (ResourceExpr): Resource field for positive control counts.
        num_controls (ResourceExpr): Possibly symbolic control count.

    Returns:
        ResourceExpr: Field that specializes consistently for zero controls.
    """
    return cast(
        ResourceExpr,
        sp.Piecewise(
            (uncontrolled, sp.Eq(num_controls, _ZERO)),
            (controlled, True),
        ),
    )


def _classify_controlled_gate(
    gate_name: str,
    num_controls: ResourceExpr,
) -> GateResources:
    """Classify a controlled primitive gate.

    Args:
        gate_name (str): Lowercase base gate name.
        num_controls (ResourceExpr): Number of active controls.

    Returns:
        GateResources: Resource contribution of the controlled primitive.
    """
    if gate_name in _SINGLE_QUBIT_GATES:
        base_qubits = 1
    elif gate_name in _TWO_QUBIT_GATES:
        base_qubits = 2
    else:
        base_qubits = _GATE_BASE_QUBITS.get(gate_name, 1)
    total_qubits = num_controls + base_qubits
    two = cast(
        ResourceExpr, sp.Piecewise((_ONE, sp.Eq(total_qubits, 2)), (_ZERO, True))
    )
    multi = cast(ResourceExpr, sp.Piecewise((_ONE, total_qubits > 2), (_ZERO, True)))
    if gate_name in _CONTROLLED_CLIFFORD_GATES:
        clifford = cast(
            ResourceExpr,
            sp.Piecewise((_ONE, sp.Eq(num_controls, 1)), (_ZERO, True)),
        )
    else:
        clifford = _ZERO
    rotation = _ONE if gate_name in _ROTATION_GATES else _ZERO
    inherent_controls = {"x": 0, "cx": 1, "toffoli": 2}
    effective_x_controls = num_controls + inherent_controls.get(gate_name, 0)
    is_x_family = gate_name in inherent_controls
    toffoli = (
        cast(
            ResourceExpr,
            sp.Piecewise(
                (_ONE, sp.Eq(effective_x_controls, 2)),
                (_ZERO, True),
            ),
        )
        if is_x_family
        else _ZERO
    )
    controlled = GateResources(
        total=_ONE,
        single_qubit=_ZERO,
        two_qubit=two,
        multi_qubit=multi,
        clifford=clifford,
        rotation=rotation,
        t=_ONE if gate_name in _T_GATES else _ZERO,
        toffoli=toffoli,
        non_clifford=_ONE - clifford,
    )
    uncontrolled = _classify_uncontrolled_gate(gate_name)
    return GateResources(
        total=_select_zero_control_resource(
            uncontrolled.total,
            controlled.total,
            num_controls,
        ),
        single_qubit=_select_zero_control_resource(
            uncontrolled.single_qubit,
            controlled.single_qubit,
            num_controls,
        ),
        two_qubit=_select_zero_control_resource(
            uncontrolled.two_qubit,
            controlled.two_qubit,
            num_controls,
        ),
        multi_qubit=_select_zero_control_resource(
            uncontrolled.multi_qubit,
            controlled.multi_qubit,
            num_controls,
        ),
        clifford=_select_zero_control_resource(
            uncontrolled.clifford,
            controlled.clifford,
            num_controls,
        ),
        rotation=_select_zero_control_resource(
            uncontrolled.rotation,
            controlled.rotation,
            num_controls,
        ),
        t=_select_zero_control_resource(
            uncontrolled.t,
            controlled.t,
            num_controls,
        ),
        toffoli=_select_zero_control_resource(
            uncontrolled.toffoli,
            controlled.toffoli,
            num_controls,
        ),
        non_clifford=_select_zero_control_resource(
            uncontrolled.non_clifford,
            controlled.non_clifford,
            num_controls,
        ),
    )
