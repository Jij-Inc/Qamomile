"""Route logical gate estimation to the selected resource model."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import sympy as sp

import qamomile.observable as qm_o
from qamomile.circuit.estimator._clean_ancilla_projection import (
    _estimate_clean_ancilla_named_gate,
)
from qamomile.circuit.estimator._clifford_t_decomposition import (
    _classify_clifford_t_gate,
    _clifford_t_conservative_condition,
    _named_clifford_t_clean_ancillas,
)
from qamomile.circuit.estimator._clifford_t_depth import _named_clifford_t_depth
from qamomile.circuit.estimator._config import (
    _DEFAULT_GATE_BASIS,
    _DEFAULT_ROTATION_SYNTHESIS_PRECISION,
)
from qamomile.circuit.estimator._constants import (
    _ZERO,
)
from qamomile.circuit.estimator._estimate import (
    ResourceEstimate,
)
from qamomile.circuit.estimator._gate_catalog import (
    _ROTATION_GATES,
)
from qamomile.circuit.estimator._gate_classification import (
    _classify_controlled_gate,
    _classify_uncontrolled_gate,
)
from qamomile.circuit.estimator._resource_algebra import (
    _add_depth,
    _scale_depth,
)
from qamomile.circuit.estimator._resource_base import (
    ApproximationStatus,
    ControlDecomposition,
    EstimateQuality,
    GateBasis,
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_expressions import (
    _expr,
)
from qamomile.circuit.estimator._resource_types import (
    DepthResources,
    GateResources,
    ResourceTraceNode,
    WidthResources,
)
from qamomile.circuit.ir.operation.gate import (
    GateOperation,
)


def _sympify_resource_value(value: Any, fallback_name: str) -> sp.Expr:
    """Convert a bound loop item to a scalar resource expression.

    Args:
        value (Any): Bound dictionary key or value.
        fallback_name (str): Symbol name used for a non-scalar value.

    Returns:
        sp.Expr: SymPy scalar, or a symbolic placeholder for structural data.
    """
    try:
        expression = sp.sympify(value)
    except (TypeError, ValueError, sp.SympifyError):
        return sp.Symbol(fallback_name)
    if isinstance(expression, sp.Expr):
        return expression
    return sp.Symbol(fallback_name)


def _estimate_named_gate_in_basis(
    name: str,
    controls: ResourceExpr,
    *,
    basis: GateBasis,
    control_decomposition: ControlDecomposition,
    precision: float,
) -> ResourceEstimate:
    """Estimate a named primitive in one public gate model.

    This helper is used for decomposition gates introduced by the estimator
    itself, such as control-value brackets and Pauli-gadget basis changes.

    Args:
        name (str): Lowercase gate name.
        controls (ResourceExpr): Number of additional coherent controls.
        basis (GateBasis): Requested output basis.
        control_decomposition (ControlDecomposition): Requested coherent-control
            representation.
        precision (float): Rotation-synthesis precision for ``CLIFFORD_T``.

    Returns:
        ResourceEstimate: Gate, depth, and decomposition-ancilla resources.

    Raises:
        ValueError: If the Clifford+T basis is asked to preserve a controlled
            primitive abstractly or lacks a lowering for the named gate.
    """
    if (
        basis is GateBasis.LOGICAL
        and control_decomposition is ControlDecomposition.CLEAN_ANCILLA_TOFFOLI
    ):
        return _estimate_clean_ancilla_named_gate(
            ResourceEstimate.zero(),
            name,
            controls,
        )

    normalized_name = "toffoli" if name == "ccx" else name
    if basis is GateBasis.CLIFFORD_T:
        if control_decomposition is ControlDecomposition.ABSTRACT and controls != _ZERO:
            raise ValueError(
                "Clifford+T estimation cannot preserve a controlled primitive "
                "as abstract. Select the clean-ancilla Toffoli control "
                "decomposition."
            )
        gates = _classify_clifford_t_gate(
            normalized_name,
            controls,
            precision,
        )
        clean_ancillas = _named_clifford_t_clean_ancillas(
            normalized_name,
            controls,
        )
        estimate = ResourceEstimate(
            width=WidthResources(
                clean_ancilla_qubits=clean_ancillas,
                peak_qubits=clean_ancillas,
            ),
            gates=gates,
            depth=_named_clifford_t_depth(
                normalized_name,
                controls,
                gates,
                precision,
            ),
            trace=ResourceTraceNode(
                normalized_name,
                "clifford_t_decomposition",
                summary=f"gates={gates.total}",
            ),
        )
        estimate = dataclasses.replace(
            estimate,
            basis=basis,
            control_decomposition=control_decomposition,
            precision=precision,
        )
        conservative_when = _clifford_t_conservative_condition(
            normalized_name,
            controls,
        )
        if conservative_when is not sp.false:
            estimate = estimate._with_metadata(
                quality=EstimateQuality.CONSERVATIVE,
                active_when=conservative_when,
            )
        if normalized_name in _ROTATION_GATES:
            estimate = estimate._with_metadata(
                quality=EstimateQuality.UNKNOWN,
                approximation=ApproximationStatus.APPROXIMATE,
            )
        return estimate

    gates = (
        _classify_uncontrolled_gate(normalized_name)
        if controls == _ZERO
        else _classify_controlled_gate(normalized_name, controls)
    )
    return dataclasses.replace(
        ResourceEstimate.primitive(normalized_name, gates),
        basis=basis,
        control_decomposition=control_decomposition,
        precision=None,
    )


def _classify_gate(
    operation: GateOperation,
    *,
    num_controls: ResourceExpr | int = 0,
    basis: GateBasis = _DEFAULT_GATE_BASIS,
    precision: float = _DEFAULT_ROTATION_SYNTHESIS_PRECISION,
) -> GateResources:
    """Classify one primitive gate into logical gate resources.

    Args:
        operation (GateOperation): Primitive gate operation.
        num_controls (ResourceExpr | int): Surrounding controls. Defaults to
            zero.
        basis (GateBasis): Gate basis to report. Defaults to ``LOGICAL``.
        precision (float): Rotation-synthesis precision in ``CLIFFORD_T``
            basis. Defaults to ``1e-10``.

    Returns:
        GateResources: Resource contribution of the primitive gate.
    """
    gate_name = operation.gate_type.name.lower() if operation.gate_type else "unknown"
    if gate_name == "ccx":
        gate_name = "toffoli"
    if basis is GateBasis.CLIFFORD_T:
        return _classify_clifford_t_gate(
            gate_name,
            _expr(num_controls),
            precision,
        )
    if _expr(num_controls) == 0:
        return _classify_uncontrolled_gate(gate_name)
    return _classify_controlled_gate(gate_name, _expr(num_controls))


def _gate_has_rotation(operation: GateOperation) -> bool:
    """Return whether a primitive gate carries an arbitrary rotation.

    Args:
        operation (GateOperation): Primitive gate operation.

    Returns:
        bool: Whether the gate requires approximate Clifford+T synthesis.
    """
    name = operation.gate_type.name.lower() if operation.gate_type else "unknown"
    return name in _ROTATION_GATES


def _pauli_terms_share_local_basis(
    terms: Sequence[Sequence[Any]],
) -> bool:
    """Return whether every qubit uses at most one Pauli basis across terms.

    Terms drawn from one local tensor-product basis commute pairwise. This
    linear fast path covers diagonal Ising/QUBO Hamiltonians without scanning
    every term pair; mixed-basis commuting sets fall back to the exact
    pairwise anticommutation test.

    Args:
        terms (Sequence[Sequence[Any]]): Active non-identity Pauli strings.

    Returns:
        bool: Whether one consistent Pauli basis exists at every qubit index.
    """
    basis_by_qubit: dict[int, Any] = {}
    for term in terms:
        for operator in term:
            previous = basis_by_qubit.setdefault(operator.index, operator.pauli)
            if previous != operator.pauli:
                return False
    return True


def _classify_pauli_evolve_depth(
    operators: Sequence[Any],
    *,
    basis_h_layer: DepthResources,
    basis_s_layer: DepthResources,
    parity_layer: DepthResources,
    rotation: DepthResources,
) -> DepthResources:
    """Estimate the exact logical depth of one Pauli-gadget term.

    Basis changes acting on distinct qubits share a layer. The forward and
    inverse parity ladders remain sequential because neighboring CNOTs overlap,
    and the axial rotation follows the forward ladder.

    Args:
        operators (Sequence[Any]): Non-identity Pauli operators in the term.
        basis_h_layer (DepthResources): Depth of one Hadamard layer.
        basis_s_layer (DepthResources): Depth of one phase-gate layer.
        parity_layer (DepthResources): Depth of one parity-ladder CNOT.
        rotation (DepthResources): Depth of the axial rotation, including any
            surrounding controls.

    Returns:
        DepthResources: Exact logical depth of the Pauli-gadget term.
    """
    has_h_basis_change = any(
        operator.pauli in (qm_o.Pauli.X, qm_o.Pauli.Y) for operator in operators
    )
    has_s_basis_change = any(operator.pauli == qm_o.Pauli.Y for operator in operators)
    depth = rotation
    if has_h_basis_change:
        depth = _add_depth(depth, _scale_depth(basis_h_layer, sp.Integer(2)))
    if has_s_basis_change:
        depth = _add_depth(depth, _scale_depth(basis_s_layer, sp.Integer(2)))
    parity_layers = sp.Integer(2 * max(0, len(operators) - 1))
    return _add_depth(depth, _scale_depth(parity_layer, parity_layers))
