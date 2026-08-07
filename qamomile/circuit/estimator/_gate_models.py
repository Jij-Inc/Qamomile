"""Estimate gates in Qamomile's logical resource model."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from typing import Any

import sympy as sp

import qamomile.observable as qm_o
from qamomile.circuit.estimator._clean_ancilla_projection import (
    _estimate_clean_ancilla_named_gate,
)
from qamomile.circuit.estimator._constants import (
    _ZERO,
)
from qamomile.circuit.estimator._estimate import (
    ResourceEstimate,
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
    ControlDecomposition,
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_expressions import (
    _expr,
)
from qamomile.circuit.estimator._resource_types import (
    DepthResources,
    GateResources,
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


def _estimate_named_gate(
    name: str,
    controls: ResourceExpr,
    *,
    control_decomposition: ControlDecomposition,
) -> ResourceEstimate:
    """Estimate a named primitive under one logical control model.

    This helper is used for decomposition gates introduced by the estimator
    itself, such as control-value brackets and Pauli-gadget basis changes. Its
    gate-family metrics describe the resulting logical circuit after applying
    ``control_decomposition``.

    Args:
        name (str): Lowercase gate name.
        controls (ResourceExpr): Number of additional coherent controls.
        control_decomposition (ControlDecomposition): Requested coherent-control
            representation.

    Returns:
        ResourceEstimate: Gate, depth, and decomposition-ancilla resources.
    """
    if control_decomposition is ControlDecomposition.CLEAN_ANCILLA_TOFFOLI:
        return _estimate_clean_ancilla_named_gate(
            ResourceEstimate.zero(),
            name,
            controls,
        )

    normalized_name = "toffoli" if name == "ccx" else name
    gates = (
        _classify_uncontrolled_gate(normalized_name)
        if controls == _ZERO
        else _classify_controlled_gate(normalized_name, controls)
    )
    return dataclasses.replace(
        ResourceEstimate.primitive(normalized_name, gates),
        control_decomposition=control_decomposition,
    )


def _classify_gate(
    operation: GateOperation,
    *,
    num_controls: ResourceExpr | int = 0,
) -> GateResources:
    """Classify one primitive gate into logical gate resources.

    Args:
        operation (GateOperation): Primitive gate operation.
        num_controls (ResourceExpr | int): Surrounding controls. Defaults to
            zero.

    Returns:
        GateResources: Resource contribution of the primitive gate.
    """
    gate_name = operation.gate_type.name.lower() if operation.gate_type else "unknown"
    if gate_name == "ccx":
        gate_name = "toffoli"
    if _expr(num_controls) == 0:
        return _classify_uncontrolled_gate(gate_name)
    return _classify_controlled_gate(gate_name, _expr(num_controls))


def _pauli_terms_share_local_basis(
    terms: Iterable[Sequence[Any]],
) -> bool:
    """Return whether every qubit uses at most one Pauli basis across terms.

    Terms drawn from one local tensor-product basis commute pairwise. This
    linear fast path covers diagonal Ising/QUBO Hamiltonians without scanning
    every term pair. Callers that need an exact mixed-basis answer must use a
    fuller test, such as component-Hamiltonian commutators that retain
    cancellation between Pauli-pair contributions.

    Args:
        terms (Iterable[Sequence[Any]]): Active non-identity Pauli strings.

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
