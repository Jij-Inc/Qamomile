"""Model Clifford+T gate counts and clean-ancilla requirements."""

from __future__ import annotations

import math
from typing import cast

import sympy as sp

from qamomile.circuit.estimator._constants import (
    _ONE,
    _ZERO,
)
from qamomile.circuit.estimator._gate_catalog import (
    _ROTATION_GATES,
)
from qamomile.circuit.estimator._gate_classification import (
    _classify_uncontrolled_gate,
)
from qamomile.circuit.estimator._resource_algebra import (
    _add_gates,
    _conditional_gates,
    _scale_gates,
)
from qamomile.circuit.estimator._resource_base import (
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_expressions import (
    _boolean_condition,
    _piecewise,
    _resource_expr,
)
from qamomile.circuit.estimator._resource_types import (
    GateResources,
)
from qamomile.circuit.ir.operation.gate import (
    GateOperation,
)
from qamomile.circuit.transpiler.passes.emit_support.clean_ancilla_toffoli import (
    clean_ancilla_toffoli_ladder,
    clean_ancilla_toffoli_ladder_or_empty,
)

_PHASE_CLASS_CODES = {
    None: 0,
    "z": 1,
    "s": 2,
    "t": 3,
    "sdg": 4,
    "tdg": 5,
    "p": -1,
}


class _CanonicalPhaseClass(sp.Function):
    """Classify a phase after numeric substitution without erasing small phases."""

    nargs = 1

    @classmethod
    def eval(cls, phase: sp.Expr) -> sp.Integer | None:
        """Evaluate a concrete phase to its canonical gate-class code.

        Args:
            phase (sp.Expr): Phase angle in radians.

        Returns:
            sp.Integer | None: Canonical class for a numeric phase, or
            ``None`` to retain a symbolic function application.
        """
        normalized = cast(sp.Expr, sp.sympify(phase))
        if not normalized.is_number:
            return None
        gate_name = _canonical_phase_gate_name(normalized)
        return sp.Integer(_PHASE_CLASS_CODES[gate_name])


def _canonical_phase_gate_name(phase: sp.Expr) -> str | None:
    """Return the simplest fixed phase gate for a resolved angle.

    Numeric multiples of pi/4 are recognized modulo 2*pi. Other numeric and
    symbolic phases retain the arbitrary phase gate ``p``.

    Args:
        phase (sp.Expr): Resolved real phase angle in radians.

    Returns:
        str | None: ``None`` for an identity phase, a fixed gate name for a
        recognized Clifford+T phase, or ``"p"`` otherwise.
    """
    if not phase.is_number or phase.is_real is False:
        return "p"
    pi_quarters = sp.simplify(4 * phase / sp.pi)
    if pi_quarters.is_integer is True and pi_quarters.is_number:
        exact_class = int(pi_quarters) % 8
        return {
            0: None,
            1: "t",
            2: "s",
            4: "z",
            6: "sdg",
            7: "tdg",
        }.get(exact_class, "p")

    numeric_value = float(sp.N(phase))
    if not math.isfinite(numeric_value):
        return "p"
    value = math.fmod(numeric_value, math.tau)
    if value < 0:
        value += math.tau
    # Match only a literal floating-point representation of the identity.
    # A fixed absolute tolerance here would silently erase small but nonzero
    # phases and could undercount below the requested synthesis precision.
    if math.isclose(value, 0.0, rel_tol=0.0, abs_tol=0.0):
        return None
    fixed = (
        (math.pi / 4, "t"),
        (math.pi / 2, "s"),
        (math.pi, "z"),
        (3 * math.pi / 2, "sdg"),
        (7 * math.pi / 4, "tdg"),
    )
    for angle, gate_name in fixed:
        floating_error = 4 * max(math.ulp(value), math.ulp(angle))
        if math.isclose(value, angle, rel_tol=0.0, abs_tol=floating_error):
            return gate_name
    return "p"


def _named_clifford_t_clean_ancillas(
    name: str,
    surrounding_controls: ResourceExpr,
) -> ResourceExpr:
    """Return Clifford+T ancillas for an estimator-introduced named gate.

    Args:
        name (str): Lowercase gate name.
        surrounding_controls (ResourceExpr): Additional coherent controls.

    Returns:
        ResourceExpr: Peak clean-ancilla demand.
    """
    return _clifford_t_clean_ancillas_for_name(name, surrounding_controls)


def _clifford_t_clean_ancillas_for_name(
    name: str,
    surrounding_controls: ResourceExpr,
) -> ResourceExpr:
    """Return clean ancillas for one canonical Clifford+T lowering.

    Args:
        name (str): Lowercase logical gate name.
        surrounding_controls (ResourceExpr): Additional coherent controls.

    Returns:
        ResourceExpr: Reusable clean-ancilla demand.
    """
    inherent_controls = {"x": 0, "cx": 1, "toffoli": 2}
    if name in inherent_controls:
        controls = surrounding_controls + inherent_controls[name]
        return _clifford_t_mcx_clean_ancillas(controls)
    if name in {"z", "y"}:
        return _clifford_t_mcx_clean_ancillas(surrounding_controls)
    if name == "cz":
        return _clifford_t_mcx_clean_ancillas(surrounding_controls + _ONE)
    if name == "swap":
        return _clifford_t_mcx_clean_ancillas(surrounding_controls + _ONE)
    if name == "p":
        return clean_ancilla_toffoli_ladder_or_empty(
            surrounding_controls
        ).clean_ancillas
    if name == "cp":
        recipe = clean_ancilla_toffoli_ladder_or_empty(surrounding_controls + _ONE)
        return _piecewise(
            _ZERO,
            recipe.clean_ancillas,
            sp.Eq(surrounding_controls, _ZERO),
        )
    if name in {"s", "sdg"}:
        return clean_ancilla_toffoli_ladder_or_empty(
            surrounding_controls
        ).clean_ancillas
    if name in {"t", "tdg"}:
        return clean_ancilla_toffoli_ladder_or_empty(
            surrounding_controls + _ONE
        ).clean_ancillas
    return clean_ancilla_toffoli_ladder_or_empty(surrounding_controls).clean_ancillas


def _clifford_t_mcx_clean_ancillas(controls: ResourceExpr) -> ResourceExpr:
    """Return workspace for the clean-ancilla MCX recipe.

    Args:
        controls (ResourceExpr): Number of controls on the X target.

    Returns:
        ResourceExpr: Clean ancillas required by the logical control recipe
            after lowering its Toffoli gates to Clifford+T.
    """
    recipe = clean_ancilla_toffoli_ladder(controls)
    return _resource_expr(
        sp.Piecewise(
            (_ZERO, controls <= 2),
            (recipe.clean_ancillas, True),
        )
    )


def _clifford_t_mcx_toffoli_count(controls: ResourceExpr) -> ResourceExpr:
    """Return the Toffoli count for the clean-ancilla MCX recipe.

    Args:
        controls (ResourceExpr): Number of controls on the X target.

    Returns:
        ResourceExpr: Toffoli gates in the selected logical control recipe.
    """
    recipe = clean_ancilla_toffoli_ladder(controls)
    return _resource_expr(
        sp.Piecewise(
            (_ZERO, controls <= 1),
            (_ONE, sp.Eq(controls, 2)),
            (recipe.total_toffolis, True),
        )
    )


def _clifford_t_conservative_condition(
    gate_name: str,
    num_controls: ResourceExpr,
) -> sp.Basic:
    """Return when a Clifford+T gate uses a conservative exact decomposition.

    Args:
        gate_name (str): Lowercase logical gate name.
        num_controls (ResourceExpr): Number of surrounding coherent controls.

    Returns:
        sp.Basic: Boolean activation condition for conservative provenance.
    """
    if gate_name in _ROTATION_GATES:
        return sp.false
    inherent_controls = {"x": 0, "cx": 1, "toffoli": 2}
    if gate_name in inherent_controls:
        return _boolean_condition(sp.Gt(num_controls + inherent_controls[gate_name], 2))
    mcx_wrapper_controls = {"z": 0, "y": 0, "cz": 1, "swap": 1}
    if gate_name in mcx_wrapper_controls:
        return _boolean_condition(
            sp.Gt(num_controls + mcx_wrapper_controls[gate_name], 2)
        )
    if gate_name in {"s", "sdg"}:
        return _boolean_condition(sp.Gt(num_controls, _ONE))
    exact_controlled = {
        "t",
        "tdg",
    }
    if gate_name in exact_controlled:
        return sp.false
    return _boolean_condition(sp.Gt(num_controls, _ZERO))


def _classify_clifford_t_gate(
    gate_name: str,
    num_controls: ResourceExpr,
    precision: float,
) -> GateResources:
    """Lower one logical primitive to aggregate Clifford+T resources.

    Exact canonical decompositions are used for X-family, Pauli, SWAP, and
    fixed phases. Arbitrary uncontrolled axial rotations use the
    Ross-Selinger asymptotic cost model
    ``ceil(3 log2(1 / precision))`` T gates. This scalar formula is not a
    field-wise upper bound for a concrete synthesized sequence. A controlled primitive is
    rejected unless this estimator defines an explicit Clifford+T lowering.

    Args:
        gate_name (str): Lowercase logical gate name.
        num_controls (ResourceExpr): Number of surrounding controls.
        precision (float): Rotation-synthesis precision.

    Returns:
        GateResources: Clifford+T aggregate counts.
    """
    inherent_controls = {"x": 0, "cx": 1, "toffoli": 2}
    if gate_name in inherent_controls:
        return _multi_controlled_x_clifford_t(
            num_controls + inherent_controls[gate_name]
        )
    uncontrolled = _classify_uncontrolled_clifford_t_gate(gate_name, precision)
    if num_controls.is_number and num_controls.is_integer and int(num_controls) == 0:
        return uncontrolled
    controlled = _classify_controlled_clifford_t_gate(
        gate_name,
        num_controls,
        precision,
    )
    if num_controls.is_number and num_controls.is_integer:
        return controlled
    return _conditional_gates(
        uncontrolled,
        controlled,
        sp.Eq(num_controls, _ZERO),
    )


def _classify_uncontrolled_clifford_t_gate(
    gate_name: str,
    precision: float,
) -> GateResources:
    """Lower an uncontrolled logical primitive to Clifford+T resources.

    Args:
        gate_name (str): Lowercase logical gate name.
        precision (float): Rotation-synthesis precision.

    Returns:
        GateResources: Aggregate resources for the canonical decomposition.
    """
    if gate_name == "swap":
        return GateResources(
            total=sp.Integer(3),
            two_qubit=sp.Integer(3),
            clifford=sp.Integer(3),
        )
    if gate_name == "cz":
        # Use the same H-CX-H canonical Clifford lowering as controlled Z.
        return GateResources(
            total=sp.Integer(3),
            single_qubit=sp.Integer(2),
            two_qubit=sp.Integer(1),
            clifford=sp.Integer(3),
        )
    rotation_t = sp.Integer(math.ceil(3 * math.log2(1 / precision)))
    rotation_multiplicity = 0
    extra_single_clifford = 0
    extra_two_clifford = 0
    if gate_name in {"rz", "p"}:
        rotation_multiplicity = 1
    elif gate_name == "rx":
        rotation_multiplicity = 1
        extra_single_clifford = 2
    elif gate_name == "ry":
        rotation_multiplicity = 1
        extra_single_clifford = 4
    elif gate_name == "cp":
        rotation_multiplicity = 3
        extra_two_clifford = 2
    elif gate_name == "rzz":
        rotation_multiplicity = 1
        extra_two_clifford = 2
    if rotation_multiplicity:
        t_count = rotation_multiplicity * rotation_t
        extra_clifford = extra_single_clifford + extra_two_clifford
        return GateResources(
            total=t_count + extra_clifford,
            single_qubit=t_count + extra_single_clifford,
            two_qubit=sp.Integer(extra_two_clifford),
            clifford=sp.Integer(extra_clifford),
            t=t_count,
            non_clifford=t_count,
        )
    return _classify_uncontrolled_gate(gate_name)


def _classify_controlled_clifford_t_gate(
    gate_name: str,
    num_controls: ResourceExpr,
    precision: float,
) -> GateResources:
    """Return the defined Clifford+T lowering for a controlled primitive.

    Multi-control lowering uses the clean-ancilla Toffoli model: clean ancillas
    compute a reusable conjunction, one singly controlled primitive is
    applied, and the conjunction is uncomputed. Specialized Pauli, SWAP, and
    phase paths avoid treating an arbitrary control arity as one opaque gate.

    Args:
        gate_name (str): Lowercase logical gate name.
        num_controls (ResourceExpr): Positive surrounding-control count.
        precision (float): Rotation-synthesis precision.

    Returns:
        GateResources: Aggregate decomposition resources.

    Raises:
        ValueError: If no explicit controlled Clifford+T lowering is defined.
    """
    if gate_name == "swap":
        middle = _multi_controlled_x_clifford_t(num_controls + _ONE)
        return _add_gates(
            middle,
            GateResources(
                total=sp.Integer(2),
                two_qubit=sp.Integer(2),
                clifford=sp.Integer(2),
            ),
        )
    if gate_name in {"z", "y"}:
        middle = _multi_controlled_x_clifford_t(num_controls)
        return _add_gates(
            middle,
            GateResources(
                total=sp.Integer(2),
                single_qubit=sp.Integer(2),
                clifford=sp.Integer(2),
            ),
        )
    if gate_name == "cz":
        middle = _multi_controlled_x_clifford_t(num_controls + _ONE)
        return _add_gates(
            middle,
            GateResources(
                total=sp.Integer(2),
                single_qubit=sp.Integer(2),
                clifford=sp.Integer(2),
            ),
        )
    if gate_name in {"p", "cp"}:
        effective_controls = num_controls + (_ONE if gate_name == "cp" else _ZERO)
        ladder_steps = clean_ancilla_toffoli_ladder_or_empty(
            effective_controls
        ).total_toffolis
        ladder = _scale_gates(
            _multi_controlled_x_clifford_t(sp.Integer(2)),
            ladder_steps,
        )
        central = _classify_uncontrolled_clifford_t_gate("cp", precision)
        return _add_gates(ladder, central)
    if gate_name in {"s", "sdg"}:
        # CS = (T x T) - CX - Tdg(target) - CX. The first two T gates
        # share a layer, so this exact phase-polynomial circuit uses three
        # T gates at T-depth two and needs no clean carrier.
        effective_ladder_steps = clean_ancilla_toffoli_ladder_or_empty(
            num_controls
        ).total_toffolis
        ladder = _scale_gates(
            _multi_controlled_x_clifford_t(sp.Integer(2)),
            effective_ladder_steps,
        )
        central = GateResources(
            total=sp.Integer(5),
            single_qubit=sp.Integer(3),
            two_qubit=sp.Integer(2),
            clifford=sp.Integer(2),
            t=sp.Integer(3),
            non_clifford=sp.Integer(3),
        )
        return _add_gates(ladder, central)
    if gate_name in {"t", "tdg"}:
        # Compute the conjunction of every control and the target, phase one
        # clean carrier exactly, then uncompute it. This avoids assuming a
        # controlled-T primitive is itself a Clifford+T gate.
        ladder = _scale_gates(
            _multi_controlled_x_clifford_t(sp.Integer(2)),
            clean_ancilla_toffoli_ladder_or_empty(num_controls + _ONE).total_toffolis,
        )
        return _add_gates(ladder, _classify_uncontrolled_gate(gate_name))
    raise ValueError(
        "Clifford+T lowering is not defined for controlled gate "
        f"'{gate_name}'. Use the logical basis or provide an explicit cost "
        "model."
    )


def _multi_controlled_x_clifford_t(
    num_controls: ResourceExpr,
) -> GateResources:
    """Lower a multi-controlled X using a clean-ancilla Toffoli ladder.

    Args:
        num_controls (ResourceExpr): Number of controls on the X target.

    Returns:
        GateResources: Aggregate Clifford+T counts for the selected logical
            clean-ancilla recipe.
    """
    toffolis = _clifford_t_mcx_toffoli_count(num_controls)
    return GateResources(
        total=_resource_expr(
            sp.Piecewise(
                (_ONE, num_controls <= 1),
                (sp.Integer(15), sp.Eq(num_controls, 2)),
                (15 * toffolis + _ONE, True),
            )
        ),
        single_qubit=_resource_expr(
            sp.Piecewise(
                (_ONE, sp.Eq(num_controls, 0)),
                (_ZERO, sp.Eq(num_controls, 1)),
                (9 * toffolis, sp.Eq(num_controls, 2)),
                (9 * toffolis, True),
            )
        ),
        two_qubit=_resource_expr(
            sp.Piecewise(
                (_ZERO, sp.Eq(num_controls, 0)),
                (_ONE, sp.Eq(num_controls, 1)),
                (6 * toffolis, sp.Eq(num_controls, 2)),
                (6 * toffolis + _ONE, True),
            )
        ),
        clifford=_resource_expr(
            sp.Piecewise(
                (_ONE, num_controls <= 1),
                (8 * toffolis, sp.Eq(num_controls, 2)),
                (8 * toffolis + _ONE, True),
            )
        ),
        t=_resource_expr(
            sp.Piecewise(
                (_ZERO, num_controls <= 1),
                (7 * toffolis, True),
            )
        ),
        non_clifford=_resource_expr(
            sp.Piecewise(
                (_ZERO, num_controls <= 1),
                (7 * toffolis, True),
            )
        ),
    )


def _clifford_t_clean_ancillas(
    operation: GateOperation,
    surrounding_controls: ResourceExpr,
) -> ResourceExpr:
    """Return clean ancillas required by the selected basis decomposition.

    Args:
        operation (GateOperation): Logical primitive being lowered.
        surrounding_controls (ResourceExpr): Additional enclosing controls.

    Returns:
        ResourceExpr: Peak clean-ancilla requirement.
    """
    name = operation.gate_type.name.lower() if operation.gate_type else "unknown"
    if name == "ccx":
        name = "toffoli"
    return _clifford_t_clean_ancillas_for_name(name, surrounding_controls)
