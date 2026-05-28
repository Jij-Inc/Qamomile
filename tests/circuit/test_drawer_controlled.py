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


class TestInlineControlledLineLayering:
    """The vertical connection line drawn for an inline-expanded
    ``ControlledUOperation`` must stop at the target box's outer border.

    A multi-control MCX whose target qubit happens to lie *between*
    two control qubits would otherwise show the connection line
    bleeding through both the target gate's rounded box and the
    surrounding inline-block green border.  Two changes guard the
    desired appearance together:

    - ``PORDER_LINE`` sits *below* ``PORDER_GATE``, so any in-band
      remnant line is at least painted under the gate's body.
    - The renderer emits *two* line segments when the target box
      sits between controls above *and* below, so the segment that
      would otherwise cross the box is never drawn at all.

    Each test below pins one of these guarantees.
    """

    def test_porder_constants_put_connection_lines_below_gates(self):
        """The ``PORDER_*`` constants encode the required draw order."""
        from qamomile.circuit.visualization.types import (
            PORDER_GATE,
            PORDER_LINE,
            PORDER_TEXT,
            PORDER_WIRE,
        )

        assert PORDER_WIRE < PORDER_LINE < PORDER_GATE < PORDER_TEXT, (
            PORDER_WIRE,
            PORDER_LINE,
            PORDER_GATE,
            PORDER_TEXT,
        )

    def test_inline_controlled_u_line_zorder_below_gate_patches(self):
        """The control vertical line is drawn under every gate patch.

        Renders the user's multi-arg controlled-X ladder with
        ``inline=True`` and inspects each ``Line2D`` Matplotlib added.
        Every line whose zorder lies in the connection-line band must
        be strictly below the minimum zorder of any drawn gate patch.
        """

        @qmc.qkernel
        def _shift_helper(
            q: qmc.Vector[qmc.Qubit], control_index: qmc.UInt
        ) -> qmc.Vector[qmc.Qubit]:
            for target_index in qmc.range(3):
                mcx = qmc.control(qmc.x, num_controls=target_index + 1)
                (
                    q[control_index : control_index + 1],
                    q[0:target_index],
                    q[target_index],
                ) = mcx(
                    q[control_index : control_index + 1],
                    q[0:target_index],
                    q[target_index],
                )
            return q

        @qmc.qkernel
        def _shift_main() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(5, "q")
            q = _shift_helper(q, control_index=4)
            return qmc.measure(q)

        from qamomile.circuit.visualization.types import PORDER_GATE

        fig = _shift_main.draw(fold_loops=False, inline=True)
        ax = fig.axes[0]

        # Every gate patch (FancyBboxPatch for boxed gates, Circle for
        # control dots / target glyphs) draws at PORDER_GATE.  Any
        # connection line emitted by the inline-block path must use a
        # strictly smaller zorder.
        gate_patch_zs = [
            patch.get_zorder()
            for patch in ax.patches
            if isinstance(patch, (mpatches.FancyBboxPatch, mpatches.Circle))
        ]
        assert gate_patch_zs, "expected at least one gate patch in the figure"
        min_gate_z = min(gate_patch_zs)
        assert min_gate_z >= PORDER_GATE, (min_gate_z, PORDER_GATE)

        connection_line_zs = [
            line.get_zorder()
            for line in ax.lines
            # Skip the long horizontal qubit wires (PORDER_WIRE) and
            # text underlines; restrict to vertical connection lines by
            # checking x-equality at both endpoints.
            if (
                len(line.get_xdata()) == 2
                and float(line.get_xdata()[0]) == float(line.get_xdata()[1])
            )
        ]
        # At least three vertical lines (one per MCX iteration) must
        # exist; every one of them must sit under the gate patches.
        assert len(connection_line_zs) >= 3, connection_line_zs
        assert all(z < min_gate_z for z in connection_line_zs), (
            connection_line_zs,
            min_gate_z,
        )

    def test_target_box_sandwiched_between_controls_emits_two_segments(self):
        """A target box with controls on both sides splits the line in two.

        Iteration 2 of the helper kernel has ``mcx(q[3:4], q[0:1], q[1])``:
        controls live at q[0] (above the target) and q[3] (below it),
        and the target box sits at q[1].  The renderer must emit two
        vertical line segments -- one from the upper control down to
        the box's top edge, one from the box's bottom edge down to the
        lower control -- instead of a single segment that crosses the
        box.  The combined set of vertical lines in the figure must
        contain at least one pair whose y-spans are disjoint with a
        gap matching the target box's height.
        """

        @qmc.qkernel
        def _shift_helper(
            q: qmc.Vector[qmc.Qubit], control_index: qmc.UInt
        ) -> qmc.Vector[qmc.Qubit]:
            for target_index in qmc.range(3):
                mcx = qmc.control(qmc.x, num_controls=target_index + 1)
                (
                    q[control_index : control_index + 1],
                    q[0:target_index],
                    q[target_index],
                ) = mcx(
                    q[control_index : control_index + 1],
                    q[0:target_index],
                    q[target_index],
                )
            return q

        @qmc.qkernel
        def _shift_main() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(5, "q")
            q = _shift_helper(q, control_index=3)
            return qmc.measure(q)

        fig = _shift_main.draw(fold_loops=False, inline=True)
        ax = fig.axes[0]

        # Collect (x, y_lo, y_hi) for every vertical Line2D.
        verticals: list[tuple[float, float, float]] = []
        for line in ax.lines:
            xs = line.get_xdata()
            ys = line.get_ydata()
            if len(xs) != 2 or float(xs[0]) != float(xs[1]):
                continue
            verticals.append((float(xs[0]), float(min(ys)), float(max(ys))))

        # Group verticals by approximate x and find one column with
        # two segments separated by a gap.  Iteration 2 is the
        # crossing case (target at q[1], controls at q[0] and q[3]).
        from collections import defaultdict

        by_x: dict[float, list[tuple[float, float]]] = defaultdict(list)
        for x, lo, hi in verticals:
            by_x[round(x, 2)].append((lo, hi))

        found_split = False
        for x_key, segs in by_x.items():
            if len(segs) < 2:
                continue
            segs_sorted = sorted(segs)
            for lo1, hi1 in segs_sorted:
                for lo2, hi2 in segs_sorted:
                    if (lo2 - hi1) > 0.05:
                        found_split = True
                        break
                if found_split:
                    break
            if found_split:
                break
        assert found_split, (
            "expected at least one inline-MCX column to draw the control "
            "line as two separated segments around the target box"
        )


class TestPhantomWireSuppression:
    """A symbolic-bound slice inside a sub-kernel loop must alias the
    parent's existing wires rather than allocating fresh ones.

    Calling ``apply_controlled_shift_1_plus_on_q(q, ...)`` twice with
    ``inline=True`` used to allocate phantom wires q5 / q6 below the
    real ``q[0]..q[4]`` because ``build_qubit_map``'s ForOperation
    handler never pre-evaluated body BinOps; the slice's
    ``_uint_min(0, target_index)`` clamp could not fold and
    ``_resolve_view_chain_to_root`` returned ``None``, falling through
    to ``map_block_results``'s fresh-wire branch.  The fix is a single
    ``_evaluate_loop_body_intermediates`` call inside the iterations
    loop; this regression keeps it from going missing.
    """

    def test_double_inline_call_does_not_grow_qubit_count(self):
        """Two side-by-side calls keep ``num_qubits == 5``."""
        from qamomile.circuit.visualization.analyzer import CircuitAnalyzer
        from qamomile.circuit.visualization.style import DEFAULT_STYLE

        @qmc.qkernel
        def _shift_helper(
            q: qmc.Vector[qmc.Qubit], control_index: qmc.UInt
        ) -> qmc.Vector[qmc.Qubit]:
            for target_index in qmc.range(3):
                mcx = qmc.control(qmc.x, num_controls=target_index + 1)
                (
                    q[control_index : control_index + 1],
                    q[0:target_index],
                    q[target_index],
                ) = mcx(
                    q[control_index : control_index + 1],
                    q[0:target_index],
                    q[target_index],
                )
            return q

        @qmc.qkernel
        def _shift_main() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(5, "q")
            q = _shift_helper(q, control_index=3)
            q = _shift_helper(q, control_index=4)
            return qmc.measure(q)

        graph = _shift_main._build_graph_for_visualization()
        analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=True, fold_loops=False)
        _, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
        assert num_qubits == 5, (num_qubits, qubit_names)
        assert sorted(qubit_names.values()) == [
            "q[0]",
            "q[1]",
            "q[2]",
            "q[3]",
            "q[4]",
        ], qubit_names
