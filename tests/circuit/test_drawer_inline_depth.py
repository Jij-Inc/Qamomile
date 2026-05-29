"""Regression tests for deep inline drawing qubit routing."""

from typing import Any

import matplotlib

matplotlib.use("Agg")

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.algorithm.qaoa import qaoa_state
from qamomile.circuit.ir.operation.gate import GateOperationType
from qamomile.circuit.visualization.analyzer import CircuitAnalyzer
from qamomile.circuit.visualization.style import DEFAULT_STYLE
from qamomile.circuit.visualization.visual_ir import (
    VFoldedBlock,
    VGate,
    VInlineBlock,
    VisualCircuit,
    VisualNode,
    VUnfoldedSequence,
)


@qmc.qkernel
def _indexed_cx(
    q: qmc.Vector[qmc.Qubit],
    pairs: qmc.Matrix[qmc.UInt],
) -> qmc.Vector[qmc.Qubit]:
    """Apply CX to the first pair stored in a bound matrix.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register to update.
        pairs (qmc.Matrix[qmc.UInt]): Matrix whose first row stores
            the control and target indices.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register.
    """
    q[pairs[0, 0]], q[pairs[0, 1]] = qmc.cx(q[pairs[0, 0]], q[pairs[0, 1]])
    return q


@qmc.qkernel
def _forward_indexed_cx(
    q: qmc.Vector[qmc.Qubit],
    pairs: qmc.Matrix[qmc.UInt],
) -> qmc.Vector[qmc.Qubit]:
    """Forward bound matrix data through one helper before applying CX.

    Args:
        q (qmc.Vector[qmc.Qubit]): Qubit register to update.
        pairs (qmc.Matrix[qmc.UInt]): Matrix containing CX operand indices.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register.
    """
    q = _indexed_cx(q, pairs)
    return q


@qmc.qkernel
def _indexed_cx_state(
    pairs: qmc.Matrix[qmc.UInt],
) -> qmc.Vector[qmc.Qubit]:
    """Create three qubits and apply a nested matrix-indexed CX.

    Args:
        pairs (qmc.Matrix[qmc.UInt]): Matrix containing CX operand indices.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated qubit register.
    """
    q = qmc.qubit_array(3, "q")
    q = _forward_indexed_cx(q, pairs)
    return q


def _build_visual_circuit(
    kernel: Any,
    *,
    analyzer_kwargs: dict[str, object],
    kernel_kwargs: dict[str, object] | None = None,
) -> VisualCircuit:
    """Build a VisualCircuit directly from a qkernel for assertions.

    Args:
        kernel (Any): QKernel-like object that can build a visualization graph.
        analyzer_kwargs (dict): Keyword arguments passed to CircuitAnalyzer.
        kernel_kwargs (dict | None): Concrete draw-time bindings for the qkernel.

    Returns:
        VisualCircuit: Visual IR for assertion inspection.
    """
    graph = kernel._build_graph_for_visualization(**(kernel_kwargs or {}))
    analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, **analyzer_kwargs)
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
    return analyzer.build_visual_ir(graph, qubit_map, qubit_names, num_qubits)


def _walk_gates(nodes: list[VisualNode]) -> list[VGate]:
    """Collect VGate nodes recursively from inline and unfolded children.

    Args:
        nodes (list[VisualNode]): Visual nodes to traverse.

    Returns:
        list[VGate]: Gate nodes found in traversal order.
    """
    gates: list[VGate] = []
    for node in nodes:
        if isinstance(node, VGate):
            gates.append(node)
        elif isinstance(node, VInlineBlock):
            gates.extend(_walk_gates(node.children))
        elif isinstance(node, VUnfoldedSequence):
            for iteration in node.iterations:
                gates.extend(_walk_gates(iteration))
        elif isinstance(node, VFoldedBlock):
            continue
    return gates


@pytest.mark.parametrize("inline_depth", [2, 3, 5])
def test_nested_inline_matrix_indices_route_multi_qubit_gate(inline_depth):
    """Deep inline drawing must preserve matrix-indexed CX operands."""
    vc = _build_visual_circuit(
        _indexed_cx_state,
        kernel_kwargs={"pairs": [[1, 2]]},
        analyzer_kwargs={
            "inline": True,
            "fold_loops": False,
            "inline_depth": inline_depth,
        },
    )

    cx_indices = [
        gate.qubit_indices
        for gate in _walk_gates(vc.children)
        if gate.gate_type == GateOperationType.CX
    ]
    assert cx_indices == [[1, 2]]


@pytest.mark.parametrize("inline_depth", [3, 5])
def test_qaoa_state_deep_inline_routes_rzz_from_bound_dict(inline_depth):
    """Deep inline drawing must preserve QAOA dict-indexed RZZ operands."""
    vc = _build_visual_circuit(
        qaoa_state,
        kernel_kwargs={
            "p": 1,
            "quad": {(1, 2): 1.0},
            "linear": {},
            "n": 3,
            "gammas": [1.0],
            "betas": [1.0],
        },
        analyzer_kwargs={
            "inline": True,
            "fold_loops": False,
            "inline_depth": inline_depth,
        },
    )

    rzz_indices = [
        gate.qubit_indices
        for gate in _walk_gates(vc.children)
        if gate.gate_type == GateOperationType.RZZ
    ]
    assert rzz_indices == [[1, 2]]
