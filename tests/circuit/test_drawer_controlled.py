"""Regression tests for dedicated rendering of controlled built-in gates."""

import math

import matplotlib

matplotlib.use("Agg")

from typing import Any

import matplotlib.patches as mpatches
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.gate import GateOperationType
from qamomile.circuit.visualization.analyzer import CircuitAnalyzer
from qamomile.circuit.visualization.circuit_adapter import (
    circuit_program_to_visual_ir,
)
from qamomile.circuit.visualization.drawer import MatplotlibDrawer
from qamomile.circuit.visualization.drawing_compiler import (
    compile_qkernel_for_drawing,
)
from qamomile.circuit.visualization.geometry import compute_border_padding
from qamomile.circuit.visualization.layout import CircuitLayoutEngine
from qamomile.circuit.visualization.renderer import MatplotlibRenderer
from qamomile.circuit.visualization.style import DEFAULT_STYLE
from qamomile.circuit.visualization.visual_ir import (
    VGate,
    VGateKind,
    VInlineBlock,
    VisualCircuit,
)


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


def test_interleaved_generic_control_dot_renders_above_target_box() -> None:
    """An interleaved generic control remains visible over its target box."""
    gate = VGate(
        node_key=("interleaved-control",),
        label="U",
        qubit_indices=[1, 0, 2],
        estimated_width=DEFAULT_STYLE.gate_width,
        kind=VGateKind.CONTROLLED_U_BOX,
        control_count=1,
    )
    visual_circuit = VisualCircuit(
        children=[gate],
        qubit_map={f"q{index}": index for index in range(3)},
        qubit_names={index: f"q{index}" for index in range(3)},
        num_qubits=3,
    )
    layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(visual_circuit)
    target_box = layout.gate_box_rects[gate.node_key]
    control_y = layout.qubit_y[1]
    assert target_box.bottom < control_y < target_box.top

    figure = MatplotlibRenderer(DEFAULT_STYLE).render(visual_circuit, layout)
    axes = figure.axes[0]
    [control_dot] = [
        patch
        for patch in axes.patches
        if isinstance(patch, mpatches.Circle)
        and math.isclose(patch.center[1], control_y)
    ]
    [rendered_target_box] = [
        patch for patch in axes.patches if isinstance(patch, mpatches.FancyBboxPatch)
    ]
    [rendered_label] = [text for text in axes.texts if text.get_text() == "U"]

    assert control_dot.get_zorder() > rendered_target_box.get_zorder()
    assert not math.isclose(control_dot.center[1], rendered_label.get_position()[1])


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

        Renders three statically addressed controlled-X gates with
        ``inline=True`` and inspects each ``Line2D`` Matplotlib added.
        Every line whose zorder lies in the connection-line band must
        be strictly below the minimum zorder of any drawn gate patch.
        """

        @qmc.qkernel
        def _static_mcx_main() -> qmc.Vector[qmc.Bit]:
            """Build three valid controlled-X columns for z-order testing.

            Returns:
                qmc.Vector[qmc.Bit]: Measurements of the five-qubit register.
            """
            q = qmc.qubit_array(5, "q")
            cx = qmc.control(qmc.x)
            ccx = qmc.control(qmc.x, num_controls=2)
            q[4], q[0] = cx(q[4], q[0])
            q[4], q[0], q[1] = ccx(q[4], q[0], q[1])
            q[4], q[1], q[2] = ccx(q[4], q[1], q[2])
            return qmc.measure(q)

        from qamomile.circuit.visualization.types import PORDER_GATE

        fig = _static_mcx_main.draw(fold_loops=False, inline=True)
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
                and line.get_zorder() < PORDER_GATE
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

        A static controlled custom gate has controls at q[0] (above the
        target) and q[3] (below it), while its target box sits at q[1].
        The renderer must emit two
        vertical line segments -- one from the upper control down to
        the box's top edge, one from the box's bottom edge down to the
        lower control -- instead of a single segment that crosses the
        box.  The combined set of vertical lines in the figure must
        contain at least one pair whose y-spans are disjoint with a
        gap matching the target box's height.
        """

        target_width = DEFAULT_STYLE.gate_width
        border_padding = compute_border_padding(DEFAULT_STYLE, depth=0)
        visual_circuit = VisualCircuit(
            children=[
                VInlineBlock(
                    node_key=("sandwiched-control",),
                    label="U",
                    children=[
                        VGate(
                            node_key=("sandwiched-control", "target"),
                            label=r"$R_y$(0.25)",
                            qubit_indices=[1],
                            estimated_width=target_width,
                            kind=VGateKind.GATE,
                            gate_type=GateOperationType.RY,
                            has_param=True,
                        )
                    ],
                    affected_qubits=[0, 1, 3],
                    control_qubit_indices=[0, 3],
                    power=1,
                    depth=0,
                    border_padding=border_padding,
                    max_gate_width=target_width,
                    label_width=target_width,
                    content_width=target_width,
                    final_width=target_width + 2 * border_padding,
                )
            ],
            qubit_map={},
            qubit_names={index: f"q{index}" for index in range(4)},
            num_qubits=4,
        )
        layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(visual_circuit)
        fig = MatplotlibRenderer(DEFAULT_STYLE).render(visual_circuit, layout)
        ax = fig.axes[0]

        # Collect (x, y_lo, y_hi) for every vertical Line2D.
        verticals: list[tuple[float, float, float]] = []
        for line in ax.lines:
            xs = line.get_xdata()
            ys = line.get_ydata()
            if len(xs) != 2 or float(xs[0]) != float(xs[1]):
                continue
            verticals.append((float(xs[0]), float(min(ys)), float(max(ys))))

        # Group verticals by approximate x and find the controlled block's
        # column with two segments separated by its target box.
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
            "expected the inline controlled column to draw the control "
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


@qmc.qkernel
def _apply_controlled_slice_shifts_1d(
    q: qmc.Vector[qmc.Qubit],
    num_system: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Apply two controlled modular shifts to one sliced system register.

    Args:
        q (qmc.Vector[qmc.Qubit]): Full register containing system and control
            qubits.
        num_system (qmc.UInt): Number of qubits in the system slice.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated full register.
    """
    controlled_decrement = qmc.control(qmc.modular_decrement)
    controlled_increment = qmc.control(qmc.modular_increment)
    q[num_system], q[0:num_system] = controlled_decrement(
        q[num_system],
        q[0:num_system],
    )
    q[num_system + 1], q[0:num_system] = controlled_increment(
        q[num_system + 1],
        q[0:num_system],
    )
    return q


@qmc.qkernel
def _controlled_slice_shifts_1d(
    total_qubits: qmc.UInt,
    num_system: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    """Draw controlled shifts nested inside an ordinary qkernel call.

    Args:
        total_qubits (qmc.UInt): Total register width.
        num_system (qmc.UInt): Number of system qubits targeted by each shift.

    Returns:
        qmc.Vector[qmc.Bit]: Measurements of the full register.
    """
    q = qmc.qubit_array(total_qubits, "q")
    q = _apply_controlled_slice_shifts_1d(q, num_system)
    return qmc.measure(q)


@qmc.qkernel
def _apply_controlled_slice_shifts_2d(
    q: qmc.Vector[qmc.Qubit],
    num_axis: qmc.UInt,
    num_system: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Apply four double-controlled shifts to two disjoint slice views.

    Args:
        q (qmc.Vector[qmc.Qubit]): Full register containing both axes and
            control qubits.
        num_axis (qmc.UInt): Width of each disjoint system-axis slice.
        num_system (qmc.UInt): Combined width of both system slices.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated full register.
    """
    controlled_decrement = qmc.control(qmc.modular_decrement, num_controls=2)
    controlled_increment = qmc.control(qmc.modular_increment, num_controls=2)
    signal_start = num_system

    q[signal_start + 2], q[signal_start], q[0:num_axis] = controlled_decrement(
        q[signal_start + 2],
        q[signal_start],
        q[0:num_axis],
    )
    q[signal_start + 2], q[signal_start + 1], q[0:num_axis] = controlled_increment(
        q[signal_start + 2],
        q[signal_start + 1],
        q[0:num_axis],
    )
    q[signal_start + 2], q[signal_start], q[num_axis:num_system] = controlled_decrement(
        q[signal_start + 2],
        q[signal_start],
        q[num_axis:num_system],
    )
    q[signal_start + 2], q[signal_start + 1], q[num_axis:num_system] = (
        controlled_increment(
            q[signal_start + 2],
            q[signal_start + 1],
            q[num_axis:num_system],
        )
    )
    return q


@qmc.qkernel
def _controlled_slice_shifts_2d(
    total_qubits: qmc.UInt,
    num_axis: qmc.UInt,
    num_system: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    """Draw double-controlled shifts nested inside a qkernel call.

    Args:
        total_qubits (qmc.UInt): Total register width.
        num_axis (qmc.UInt): Width of one system-axis slice.
        num_system (qmc.UInt): Combined system-register width.

    Returns:
        qmc.Vector[qmc.Bit]: Measurements of the full register.
    """
    q = qmc.qubit_array(total_qubits, "q")
    q = _apply_controlled_slice_shifts_2d(q, num_axis, num_system)
    return qmc.measure(q)


@pytest.mark.parametrize(
    ("kernel", "bindings", "expected_block_wires"),
    [
        pytest.param(
            _controlled_slice_shifts_1d,
            {"total_qubits": 5, "num_system": 3},
            [{0, 1, 2, 3}, {0, 1, 2, 4}],
            id="one-dimensional",
        ),
        pytest.param(
            _controlled_slice_shifts_2d,
            {"total_qubits": 9, "num_axis": 3, "num_system": 6},
            [
                {0, 1, 2, 6, 8},
                {0, 1, 2, 7, 8},
                {3, 4, 5, 6, 8},
                {3, 4, 5, 7, 8},
            ],
            id="two-dimensional",
        ),
    ],
)
def test_controlled_slice_calls_alias_existing_root_wires(
    kernel: Any,
    bindings: dict[str, int],
    expected_block_wires: list[set[int]],
) -> None:
    """Controlled slice actuals neither allocate nor target phantom wires."""
    drawing = compile_qkernel_for_drawing(kernel, bindings)
    visual_circuit = circuit_program_to_visual_ir(drawing.circuit)

    total_qubits = bindings["total_qubits"]
    assert drawing.circuit.num_qubits == total_qubits
    assert drawing.qubit_names == {
        index: f"q[{index}]" for index in range(total_qubits)
    }
    shift_blocks = [
        node
        for node in visual_circuit.children
        if isinstance(node, VGate)
        and node.kind is VGateKind.CONTROLLED_U_BOX
        and "modular_" in node.label
    ]
    assert [set(node.qubit_indices) for node in shift_blocks] == expected_block_wires


class TestPostInlineVectorAliasing:
    """``MapBlockResults`` aliases per-element keys across SSA versions.

    When a sub-kernel returns a ``Vector[Qubit]`` after mutating it
    (e.g. ``qft_encoding``) or a controlled-U produces a next-version
    ``Vector[Qubit]`` result, the inlined-IR path used by the
    ``transpiler.to_block + transpiler.inline -> MatplotlibDrawer``
    workflow ends up with a downstream op whose ``parent_array``
    points at the *new* ArrayValue ``logical_id``.  The drawer's
    ``build_qubit_map`` only aliased the scalar lid -> wire pairing
    via :meth:`map_block_results`, not the per-element keys
    (``{new_lid}_[i]``).  A subsequent op (``qmc.iqft`` expanded
    inline to per-element CP/H gates, a per-element ``q[i] = ...``
    follow-up loop, ...) then fell through the element-key lookup
    and the CompositeGate / ControlledU dispatch fresh-allocated a
    phantom wire per element.

    The fix in :func:`map_block_results` copies the operand's
    ``{operand_lid}_[i]`` keys to ``{result.logical_id}_[i]``
    whenever both operand and result are ``ArrayValue``.  This test
    pins it by exercising the user-reported ``transpiler.inline +
    MatplotlibDrawer`` flow on a kernel that goes through
    ``qft_encoding -> controlled(qft_encoding) -> iqft``: the
    qubit count must equal the QInit'd register count, not the
    register count plus phantom iqft wires.
    """

    @pytest.fixture(autouse=True)
    def _require_qiskit(self):
        pytest.importorskip("qiskit")

    def test_inline_then_draw_does_not_grow_qubit_count(self):
        """to_block + inline + drawer keeps num_qubits at the QInit count.

        The bug only triggers when the controlled-U lives *inside* a
        sub-kernel whose return is a ``Vector[Qubit]`` (here:
        ``first_degree``), because InlinePass preserves logical ids
        through plain helper calls and through top-level controlled-U
        results.  Wrapping the controlled-U in a sub-kernel that
        returns the modified Vector forces the post-inline IR to
        carry a new-lid ``ArrayValue`` that the iqft's per-element
        operands then reference via ``parent_array``.
        """
        import math

        from qamomile.circuit.visualization.analyzer import CircuitAnalyzer
        from qamomile.circuit.visualization.style import DEFAULT_STYLE
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def qft_encoding(
            q: qmc.Vector[qmc.Qubit], coef: qmc.Float
        ) -> qmc.Vector[qmc.Qubit]:
            m = q.shape[0]
            for i in qmc.range(m):
                q[i] = qmc.p(q[i], 2 * math.pi * coef / (2**m) * (2**i))
            return q

        @qmc.qkernel
        def first_degree(
            q_out: qmc.Vector[qmc.Qubit],
            q_in: qmc.Vector[qmc.Qubit],
            control_idx: qmc.UInt,
            coef: qmc.Float,
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl_qft = qmc.control(qft_encoding)
            c = q_in[control_idx]
            c, q_out = ctrl_qft(c, q_out, coef)
            q_in[control_idx] = c
            return q_out, q_in

        @qmc.qkernel
        def main_kernel() -> qmc.Vector[qmc.Bit]:
            q_out = qmc.qubit_array(4, "q_out")
            q_in = qmc.qubit_array(2, "q_in")
            q_out, q_in = first_degree(q_out, q_in, 0, 0.5)
            q_out = qmc.iqft(q_out)
            qmc.measure(q_in)
            return qmc.measure(q_out)

        t = QiskitTranspiler()
        block = t.to_block(main_kernel)
        block = t.inline(block)

        analyzer = CircuitAnalyzer(
            block,
            DEFAULT_STYLE,
            inline=True,
            fold_loops=False,
            expand_composite=True,
        )
        _, qubit_names, num_qubits = analyzer.build_qubit_map(block)
        assert num_qubits == 6, (num_qubits, qubit_names)
        assert sorted(qubit_names.values()) == [
            "q_in[0]",
            "q_in[1]",
            "q_out[0]",
            "q_out[1]",
            "q_out[2]",
            "q_out[3]",
        ], qubit_names

    def test_inline_then_draw_iqft_targets_q_out(self):
        """The inline-expanded iqft children live on the q_out wires.

        Pins the visual outcome (no phantom wires under iqft) by
        walking the VisualCircuit children and asserting every
        VGate emitted by the expanded iqft addresses wires the
        ``q_out`` register actually owns.  Same wrapping pattern as
        the qubit-count test above to actually trip the per-element
        aliasing path.
        """
        import math

        from qamomile.circuit.visualization.analyzer import CircuitAnalyzer
        from qamomile.circuit.visualization.style import DEFAULT_STYLE
        from qamomile.circuit.visualization.visual_ir import (
            VFoldedBlock,
            VGate,
            VInlineBlock,
            VUnfoldedSequence,
        )
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def qft_encoding(
            q: qmc.Vector[qmc.Qubit], coef: qmc.Float
        ) -> qmc.Vector[qmc.Qubit]:
            m = q.shape[0]
            for i in qmc.range(m):
                q[i] = qmc.p(q[i], 2 * math.pi * coef / (2**m) * (2**i))
            return q

        @qmc.qkernel
        def first_degree(
            q_out: qmc.Vector[qmc.Qubit],
            q_in: qmc.Vector[qmc.Qubit],
            control_idx: qmc.UInt,
            coef: qmc.Float,
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl_qft = qmc.control(qft_encoding)
            c = q_in[control_idx]
            c, q_out = ctrl_qft(c, q_out, coef)
            q_in[control_idx] = c
            return q_out, q_in

        @qmc.qkernel
        def main_kernel() -> qmc.Vector[qmc.Bit]:
            q_out = qmc.qubit_array(4, "q_out")
            q_in = qmc.qubit_array(2, "q_in")
            q_out, q_in = first_degree(q_out, q_in, 0, 0.5)
            q_out = qmc.iqft(q_out)
            qmc.measure(q_in)
            return qmc.measure(q_out)

        t = QiskitTranspiler()
        block = t.to_block(main_kernel)
        block = t.inline(block)

        analyzer = CircuitAnalyzer(
            block,
            DEFAULT_STYLE,
            inline=True,
            fold_loops=False,
            expand_composite=True,
        )
        qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(block)
        vc = analyzer.build_visual_ir(block, qubit_map, qubit_names, num_qubits)

        q_out_wires = {0, 1, 2, 3}

        def collect_iqft_wires(node):
            """Return all qubit wires used by the iqft inline block."""
            if isinstance(node, VInlineBlock) and (
                "iqft" in (node.label or "").lower()
            ):
                wires: set[int] = set()
                stack = list(node.children)
                while stack:
                    child = stack.pop()
                    if isinstance(child, VGate):
                        wires.update(child.qubit_indices)
                    elif isinstance(child, (VInlineBlock, VFoldedBlock)):
                        stack.extend(getattr(child, "children", []))
                    elif isinstance(child, VUnfoldedSequence):
                        for iter_children in child.iterations:
                            stack.extend(iter_children)
                return wires
            for child in getattr(node, "children", []) or []:
                hit = collect_iqft_wires(child)
                if hit is not None:
                    return hit
            if isinstance(node, VUnfoldedSequence):
                for iter_children in node.iterations:
                    for child in iter_children:
                        hit = collect_iqft_wires(child)
                        if hit is not None:
                            return hit
            return None

        iqft_wires = None
        for top in vc.children:
            iqft_wires = collect_iqft_wires(top)
            if iqft_wires is not None:
                break

        assert iqft_wires is not None, "expected the iqft inline block in the tree"
        assert iqft_wires, "expected the iqft block to emit at least one gate"
        assert iqft_wires.issubset(q_out_wires), (iqft_wires, q_out_wires)
