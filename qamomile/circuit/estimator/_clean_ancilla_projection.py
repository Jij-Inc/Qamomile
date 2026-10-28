"""Project named gates through the clean-ancilla Toffoli model."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import sympy as sp

from qamomile.circuit.estimator._resource_algebra import (
    _add_gates,
    _scale_gates,
)
from qamomile.circuit.estimator._resource_base import (
    EstimateQuality,
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_types import (
    GateResources,
    ResourceAssumption,
    ResourceTraceNode,
    WidthResources,
)
from qamomile.circuit.ir.operation.gate import (
    GateOperation,
    GateOperationType,
)
from qamomile.circuit.transpiler.passes.emit_support.clean_ancilla_toffoli import (
    clean_ancilla_toffoli_ladder,
)

if TYPE_CHECKING:
    from qamomile.circuit.estimator._estimate import ResourceEstimate

from qamomile.circuit.estimator._constants import (
    _ONE,
    _ZERO,
)
from qamomile.circuit.estimator._gate_catalog import (
    _CLEAN_ANCILLA_EXPLICIT_MULTI_TARGET_GATE_TYPES,
    _GATE_OPERATION_ARITY,
)
from qamomile.circuit.estimator._gate_classification import (
    _classify_controlled_gate,
    _classify_uncontrolled_gate,
    _serial_depth_from_gate_resources,
)


def _clean_ancilla_sequence_estimate(
    empty: ResourceEstimate,
    name: str,
    gates: GateResources,
    *,
    clean_ancillas: ResourceExpr = _ZERO,
) -> ResourceEstimate:
    """Build one clean-ancilla Toffoli decomposition estimate.

    Args:
        empty (ResourceEstimate): Exact-zero estimate used as the construction
            seed.
        name (str): Human-readable gate or decomposition name.
        gates (GateResources): Aggregate logical gate resources.
        clean_ancillas (ResourceExpr): Reusable clean-ancilla demand.
            Defaults to zero.

    Returns:
        ResourceEstimate: Serial gate/depth and reusable-width estimate.
    """
    width = WidthResources(
        clean_ancilla_qubits=clean_ancillas,
        peak_qubits=clean_ancillas,
    )
    return dataclasses.replace(
        empty.zero(),
        width=width,
        gates=gates,
        depth=_serial_depth_from_gate_resources(gates),
        trace=ResourceTraceNode(
            name=name,
            source_kind="clean_ancilla_toffoli",
            summary=f"gates={gates.total}, clean_ancillas={clean_ancillas}",
        ),
    )


def _clean_ancilla_primitive_estimate(
    empty: ResourceEstimate,
    name: str,
) -> ResourceEstimate:
    """Return one uncontrolled logical primitive estimate.

    Args:
        empty (ResourceEstimate): Exact-zero estimate used as the construction
            seed.
        name (str): Lowercase primitive gate name.

    Returns:
        ResourceEstimate: One-gate logical estimate.
    """
    return _clean_ancilla_sequence_estimate(
        empty,
        name,
        _classify_uncontrolled_gate(name),
    )


def _clean_ancilla_single_control_estimate(
    empty: ResourceEstimate,
    name: str,
) -> ResourceEstimate:
    """Return one singly controlled logical primitive estimate.

    The selected recipe represents fixed S/T-family controls with a controlled
    phase gate.

    Args:
        empty (ResourceEstimate): Exact-zero estimate used as the construction
            seed.
        name (str): Lowercase base gate name.

    Returns:
        ResourceEstimate: Singly controlled logical estimate.
    """
    controlled_name = "p" if name in {"s", "sdg", "t", "tdg"} else name
    return _clean_ancilla_sequence_estimate(
        empty,
        f"c-{name}",
        _classify_controlled_gate(controlled_name, _ONE),
    )


def _clean_ancilla_generic_multi_control_estimate(
    empty: ResourceEstimate,
    name: str,
    controls: ResourceExpr,
) -> ResourceEstimate:
    """Lower a generic gate with the clean-ancilla Toffoli recipe.

    The recipe computes the AND of ``n`` controls with ``n - 1`` clean
    ancillas, applies one singly controlled primitive, then uncomputes the
    ladder. Other decompositions or whole-body ladder sharing may be cheaper,
    so the result is an upper bound.

    Args:
        empty (ResourceEstimate): Exact-zero estimate used as the construction
            seed.
        name (str): Lowercase single-target base gate name.
        controls (ResourceExpr): Number of controls, interpreted on the branch
            where it is at least two.

    Returns:
        ResourceEstimate: Clean-ancilla Toffoli estimate.
    """
    recipe = clean_ancilla_toffoli_ladder(controls)
    ladder = _scale_gates(
        _classify_uncontrolled_gate("toffoli"),
        recipe.total_toffolis,
    )
    central = _clean_ancilla_single_control_estimate(empty, name).gates
    estimate = _clean_ancilla_sequence_estimate(
        empty,
        f"mc-{name}",
        _add_gates(ladder, central),
        clean_ancillas=recipe.clean_ancillas,
    )
    reason = ResourceAssumption(
        "clean-ancilla multi-control decomposition is an upper bound because "
        "whole-body ladder sharing or another decomposition may use fewer "
        "resources",
        source=f"mc-{name}",
    )
    return estimate._with_metadata(
        assumptions=(reason,),
        quality=EstimateQuality.CONSERVATIVE,
    )


def _select_control_count_estimate(
    controls: ResourceExpr,
    *,
    zero: ResourceEstimate,
    one: ResourceEstimate,
    two: ResourceEstimate,
    many: ResourceEstimate,
) -> ResourceEstimate:
    """Select a decomposition by a concrete or symbolic control count.

    Args:
        controls (ResourceExpr): Nonnegative control-count expression.
        zero (ResourceEstimate): Uncontrolled estimate.
        one (ResourceEstimate): Single-control estimate.
        two (ResourceEstimate): Two-control estimate.
        many (ResourceEstimate): Estimate for three or more controls.

    Returns:
        ResourceEstimate: Concrete branch or field-wise symbolic piecewise
        estimate.

    Raises:
        ValueError: If ``controls`` is a concrete negative integer.
    """
    if controls.is_number and controls.is_integer:
        value = int(controls)
        if value < 0:
            raise ValueError(f"control count must be nonnegative, got {value}.")
        if value == 0:
            return zero
        if value == 1:
            return one
        if value == 2:
            return two
        return many
    estimate = many
    estimate = two.conditional(estimate, sp.Eq(controls, 2))
    estimate = one.conditional(estimate, sp.Eq(controls, 1))
    return zero.conditional(estimate, sp.Eq(controls, 0))


def _clean_ancilla_single_target_estimate(
    empty: ResourceEstimate,
    name: str,
    controls: ResourceExpr,
) -> ResourceEstimate:
    """Estimate a controlled primitive with a clean-ancilla Toffoli ladder.

    Args:
        empty (ResourceEstimate): Exact-zero estimate used as the construction
            seed.
        name (str): Lowercase base gate name.
        controls (ResourceExpr): Surrounding control count.

    Returns:
        ResourceEstimate: Logical decomposition estimate.
    """
    zero = _clean_ancilla_primitive_estimate(empty, name)
    one = _clean_ancilla_single_control_estimate(empty, name)
    generic = _clean_ancilla_generic_multi_control_estimate(empty, name, controls)
    return _select_control_count_estimate(
        controls,
        zero=zero,
        one=one,
        two=generic,
        many=generic,
    )


def _clean_ancilla_x_estimate(
    empty: ResourceEstimate,
    controls: ResourceExpr,
) -> ResourceEstimate:
    """Estimate an X gate with the shared multi-control fast paths.

    Args:
        empty (ResourceEstimate): Exact-zero estimate used as the construction
            seed.
        controls (ResourceExpr): Number of controls on the X target.

    Returns:
        ResourceEstimate: Clean-ancilla X-family estimate.
    """
    return _select_control_count_estimate(
        controls,
        zero=_clean_ancilla_primitive_estimate(empty, "x"),
        one=_clean_ancilla_single_control_estimate(empty, "x"),
        two=_clean_ancilla_primitive_estimate(empty, "toffoli"),
        many=_clean_ancilla_generic_multi_control_estimate(empty, "x", controls),
    )


def _clean_ancilla_z_estimate(
    empty: ResourceEstimate,
    controls: ResourceExpr,
) -> ResourceEstimate:
    """Estimate a Z gate with the shared two-control fast path.

    Args:
        empty (ResourceEstimate): Exact-zero estimate used as the construction
            seed.
        controls (ResourceExpr): Number of controls on the Z target.

    Returns:
        ResourceEstimate: Clean-ancilla Z-family estimate.
    """
    two = (
        _clean_ancilla_primitive_estimate(empty, "h")
        .seq(_clean_ancilla_primitive_estimate(empty, "toffoli"))
        .seq(_clean_ancilla_primitive_estimate(empty, "h"))
    )
    return _select_control_count_estimate(
        controls,
        zero=_clean_ancilla_primitive_estimate(empty, "z"),
        one=_clean_ancilla_single_control_estimate(empty, "z"),
        two=two,
        many=_clean_ancilla_generic_multi_control_estimate(empty, "z", controls),
    )


def _clean_ancilla_y_estimate(
    empty: ResourceEstimate,
    controls: ResourceExpr,
) -> ResourceEstimate:
    """Estimate a Y gate with the exact two-control conjugation.

    Args:
        empty (ResourceEstimate): Exact-zero estimate used as the construction
            seed.
        controls (ResourceExpr): Number of controls on the Y target.

    Returns:
        ResourceEstimate: Clean-ancilla Y-family estimate.
    """
    two = (
        _clean_ancilla_primitive_estimate(empty, "sdg")
        .seq(_clean_ancilla_primitive_estimate(empty, "toffoli"))
        .seq(_clean_ancilla_primitive_estimate(empty, "s"))
    )
    return _select_control_count_estimate(
        controls,
        zero=_clean_ancilla_primitive_estimate(empty, "y"),
        one=_clean_ancilla_single_control_estimate(empty, "y"),
        two=two,
        many=_clean_ancilla_generic_multi_control_estimate(empty, "y", controls),
    )


def _clean_ancilla_controlled_branch(
    controls: ResourceExpr,
    uncontrolled: ResourceEstimate,
    controlled: ResourceEstimate,
) -> ResourceEstimate:
    """Select an uncontrolled estimate only when the control count is zero.

    Args:
        controls (ResourceExpr): Surrounding control count.
        uncontrolled (ResourceEstimate): Zero-control estimate.
        controlled (ResourceEstimate): Positive-control estimate.

    Returns:
        ResourceEstimate: Selected or symbolic piecewise estimate.
    """
    if controls.is_number and controls.is_integer:
        return uncontrolled if int(controls) == 0 else controlled
    return uncontrolled.conditional(controlled, sp.Eq(controls, 0))


def _estimate_clean_ancilla_named_gate(
    empty: ResourceEstimate,
    name: str,
    controls: ResourceExpr,
) -> ResourceEstimate:
    """Estimate one named gate after clean-ancilla controlled lowering.

    Args:
        empty (ResourceEstimate): Exact-zero estimate used as the construction
            seed.
        name (str): Lowercase Qamomile gate name.
        controls (ResourceExpr): Additional surrounding controls.

    Returns:
        ResourceEstimate: Logical decomposition estimate.
    """
    if name == "ccx":
        name = "toffoli"
    if name == "x":
        return _clean_ancilla_x_estimate(empty, controls)
    if name == "z":
        return _clean_ancilla_z_estimate(empty, controls)
    if name == "y":
        return _clean_ancilla_y_estimate(empty, controls)
    if name == "cx":
        return _clean_ancilla_controlled_branch(
            controls,
            _clean_ancilla_primitive_estimate(empty, "cx"),
            _clean_ancilla_x_estimate(empty, controls + _ONE),
        )
    if name == "cz":
        return _clean_ancilla_controlled_branch(
            controls,
            _clean_ancilla_primitive_estimate(empty, "cz"),
            _clean_ancilla_z_estimate(empty, controls + _ONE),
        )
    if name == "cp":
        return _clean_ancilla_controlled_branch(
            controls,
            _clean_ancilla_primitive_estimate(empty, "cp"),
            _clean_ancilla_single_target_estimate(empty, "p", controls + _ONE),
        )
    if name == "toffoli":
        return _clean_ancilla_controlled_branch(
            controls,
            _clean_ancilla_primitive_estimate(empty, "toffoli"),
            _clean_ancilla_x_estimate(empty, controls + 2),
        )
    if name == "swap":
        middle = _clean_ancilla_x_estimate(empty, controls + _ONE)
        controlled = (
            _clean_ancilla_primitive_estimate(empty, "cx")
            .seq(middle)
            .seq(_clean_ancilla_primitive_estimate(empty, "cx"))
        )
        return _clean_ancilla_controlled_branch(
            controls,
            _clean_ancilla_primitive_estimate(empty, "swap"),
            controlled,
        )
    if name == "rzz":
        controlled = (
            _clean_ancilla_primitive_estimate(empty, "cx")
            .seq(_clean_ancilla_single_target_estimate(empty, "rz", controls))
            .seq(_clean_ancilla_primitive_estimate(empty, "cx"))
        )
        return _clean_ancilla_controlled_branch(
            controls,
            _clean_ancilla_primitive_estimate(empty, "rzz"),
            controlled,
        )
    return _clean_ancilla_single_target_estimate(empty, name, controls)


def _estimate_clean_ancilla_gate(
    empty: ResourceEstimate,
    operation: GateOperation,
    controls: ResourceExpr,
) -> ResourceEstimate:
    """Estimate a primitive through the clean-ancilla Toffoli model.

    Args:
        empty (ResourceEstimate): Exact-zero estimate used as the construction
            seed.
        operation (GateOperation): Primitive gate operation.
        controls (ResourceExpr): Surrounding control count.

    Returns:
        ResourceEstimate: Logical decomposition estimate.

    Raises:
        NotImplementedError: If the operation has no registered IR arity or a
            multi-target gate lacks an explicit clean-ancilla lowering.
    """
    gate_type = operation.gate_type
    if (
        not isinstance(gate_type, GateOperationType)
        or gate_type not in _GATE_OPERATION_ARITY
    ):
        raise NotImplementedError(
            "Clean-ancilla Toffoli control decomposition is not defined for "
            f"IR gate {gate_type!r}."
        )
    arity = _GATE_OPERATION_ARITY[gate_type]
    if arity > 1 and gate_type not in _CLEAN_ANCILLA_EXPLICIT_MULTI_TARGET_GATE_TYPES:
        raise NotImplementedError(
            "Clean-ancilla Toffoli control decomposition requires an explicit "
            "multi-target "
            f"lowering for IR gate {gate_type.name}."
        )
    name = gate_type.name.lower()
    return _estimate_clean_ancilla_named_gate(empty, name, controls)
