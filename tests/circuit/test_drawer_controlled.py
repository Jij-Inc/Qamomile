"""Regression tests for dedicated rendering of controlled built-in gates."""

import matplotlib

matplotlib.use("Agg")

from typing import Any

import matplotlib.patches as mpatches
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.gate import GateOperationType
from qamomile.circuit.visualization.analyzer import CircuitAnalyzer
from qamomile.circuit.visualization.drawer import MatplotlibDrawer
from qamomile.circuit.visualization.style import DEFAULT_STYLE
from qamomile.circuit.visualization.visual_ir import VGate, VGateKind


@qmc.qkernel
def _x_gate(q: qmc.Qubit) -> qmc.Qubit:
    """Apply X as a wrapped single-gate qkernel.

    Args:
        q (qmc.Qubit): Input qubit.

    Returns:
        qmc.Qubit: Output qubit after X.
    """
    return qmc.x(q)


@qmc.qkernel
def _z_gate(q: qmc.Qubit) -> qmc.Qubit:
    """Apply Z as a wrapped single-gate qkernel.

    Args:
        q (qmc.Qubit): Input qubit.

    Returns:
        qmc.Qubit: Output qubit after Z.
    """
    return qmc.z(q)


@qmc.qkernel
def _cx_gate(c: qmc.Qubit, t: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Apply CX as a wrapped single-gate qkernel.

    Args:
        c (qmc.Qubit): Control qubit.
        t (qmc.Qubit): Target qubit.

    Returns:
        tuple[qmc.Qubit, qmc.Qubit]: Output control and target qubits.
    """
    return qmc.cx(c, t)


@qmc.qkernel
def _cz_gate(c: qmc.Qubit, t: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Apply CZ as a wrapped single-gate qkernel.

    Args:
        c (qmc.Qubit): Control qubit.
        t (qmc.Qubit): Target qubit.

    Returns:
        tuple[qmc.Qubit, qmc.Qubit]: Output control and target qubits.
    """
    return qmc.cz(c, t)


@qmc.qkernel
def _swap_gate(a: qmc.Qubit, b: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Apply SWAP as a wrapped single-gate qkernel.

    Args:
        a (qmc.Qubit): First qubit.
        b (qmc.Qubit): Second qubit.

    Returns:
        tuple[qmc.Qubit, qmc.Qubit]: Output qubits after SWAP.
    """
    return qmc.swap(a, b)


@qmc.qkernel
def _ccx_gate(
    c0: qmc.Qubit, c1: qmc.Qubit, t: qmc.Qubit
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    """Apply Toffoli as a wrapped single-gate qkernel.

    Args:
        c0 (qmc.Qubit): First control qubit.
        c1 (qmc.Qubit): Second control qubit.
        t (qmc.Qubit): Target qubit.

    Returns:
        tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]: Output control and target qubits.
    """
    return qmc.ccx(c0, c1, t)


@qmc.qkernel
def _controlled_x_kernel() -> tuple[qmc.Qubit, qmc.Qubit]:
    """Apply controlled X so the drawer can use the CX symbol."""
    q = qmc.qubit_array(2, "q")
    cx = qmc.control(_x_gate)
    return cx(q[0], q[1])


@qmc.qkernel
def _controlled_z_kernel() -> tuple[qmc.Qubit, qmc.Qubit]:
    """Apply controlled Z so the drawer can use the CZ-dot symbol."""
    q = qmc.qubit_array(2, "q")
    cz = qmc.control(_z_gate)
    return cz(q[0], q[1])


@qmc.qkernel
def _controlled_cx_kernel() -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    """Apply controlled CX so the drawer can use the Toffoli-style symbol."""
    q = qmc.qubit_array(3, "q")
    ccx = qmc.control(_cx_gate)
    return ccx(q[0], q[1], q[2])


@qmc.qkernel
def _controlled_cz_kernel() -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    """Apply controlled CZ so the drawer can use the CCZ-style symbol."""
    q = qmc.qubit_array(3, "q")
    ccz = qmc.control(_cz_gate)
    return ccz(q[0], q[1], q[2])


@qmc.qkernel
def _controlled_swap_kernel() -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    """Apply controlled SWAP so the drawer can use the Fredkin symbol."""
    q = qmc.qubit_array(3, "q")
    cswap = qmc.control(_swap_gate)
    return cswap(q[0], q[1], q[2])


@qmc.qkernel
def _controlled_builtin_swap_kernel() -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    """Apply controlled built-in SWAP so the synthesized wrapper is covered."""
    q = qmc.qubit_array(3, "q")
    cswap = qmc.control(qmc.swap)
    return cswap(q[0], q[1], q[2])


@qmc.qkernel
def _controlled_ccx_kernel() -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    """Apply controlled Toffoli so the drawer can use a multi-control X symbol."""
    q = qmc.qubit_array(4, "q")
    cccx = qmc.control(_ccx_gate)
    return cccx(q[0], q[1], q[2], q[3])


@qmc.qkernel
def _powered_controlled_swap_kernel() -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    """Keep powered controlled SWAP as a box so the exponent is visible."""
    q = qmc.qubit_array(3, "q")
    cswap = qmc.control(_swap_gate)
    return cswap(q[0], q[1], q[2], power=2)


def _controlled_u_nodes(kernel: Any) -> list[VGate]:
    """Build controlled-U visual nodes from a kernel.

    Args:
        kernel (Any): QKernel whose visual IR should be inspected.

    Returns:
        list[VGate]: Controlled-U visual nodes found in the top-level block.
    """
    graph = kernel._build_graph_for_visualization()
    analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE)
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
    vc = analyzer.build_visual_ir(graph, qubit_map, qubit_names, num_qubits)
    return [
        node
        for node in vc.children
        if isinstance(node, VGate) and node.kind == VGateKind.CONTROLLED_U_BOX
    ]


@pytest.mark.parametrize(
    ("kernel", "gate_type", "qubit_indices"),
    [
        pytest.param(_controlled_x_kernel, GateOperationType.X, [0, 1], id="x"),
        pytest.param(_controlled_z_kernel, GateOperationType.Z, [0, 1], id="z"),
        pytest.param(_controlled_cx_kernel, GateOperationType.CX, [0, 1, 2], id="cx"),
        pytest.param(_controlled_cz_kernel, GateOperationType.CZ, [0, 1, 2], id="cz"),
        pytest.param(
            _controlled_swap_kernel,
            GateOperationType.SWAP,
            [0, 1, 2],
            id="swap",
        ),
        pytest.param(
            _controlled_builtin_swap_kernel,
            GateOperationType.SWAP,
            [0, 1, 2],
            id="builtin-swap",
        ),
        pytest.param(
            _controlled_ccx_kernel,
            GateOperationType.TOFFOLI,
            [0, 1, 2, 3],
            id="toffoli",
        ),
    ],
)
def test_controlled_single_gate_nodes_remember_dedicated_gate_type(
    kernel, gate_type, qubit_indices
):
    """Controlled wrappers around dedicated gates carry their gate type."""
    nodes = _controlled_u_nodes(kernel)
    assert len(nodes) == 1
    assert nodes[0].gate_type == gate_type
    assert nodes[0].control_count == 1
    assert nodes[0].qubit_indices == qubit_indices
    assert nodes[0].estimated_width == DEFAULT_STYLE.gate_width


def test_powered_controlled_swap_stays_generic_box():
    """Powered controlled SWAP remains boxed because the power matters."""
    nodes = _controlled_u_nodes(_powered_controlled_swap_kernel)
    assert len(nodes) == 1
    assert nodes[0].gate_type is None
    assert nodes[0].estimated_width > DEFAULT_STYLE.gate_width


def test_controlled_swap_draws_symbols_instead_of_target_box():
    """Controlled SWAP renders as a control dot plus SWAP markers."""
    fig = MatplotlibDrawer.draw_kernel(_controlled_swap_kernel)
    ax = fig.axes[0]

    rounded_boxes = [
        patch for patch in ax.patches if isinstance(patch, mpatches.FancyBboxPatch)
    ]
    control_dots = [patch for patch in ax.patches if isinstance(patch, mpatches.Circle)]

    assert rounded_boxes == []
    assert len(control_dots) == 1
