"""Regression tests for renderer-independent circuit layout geometry."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from copy import deepcopy
from dataclasses import replace
from typing import Any

import matplotlib.patches as mpatches
import pytest
from matplotlib import rc_context
from matplotlib.figure import Figure

import qamomile.circuit as qmc
from qamomile.circuit.frontend.handle import Qubit
from qamomile.circuit.ir.operation.gate import GateOperationType
from qamomile.circuit.visualization.analyzer import CircuitAnalyzer
from qamomile.circuit.visualization.circuit_adapter import (
    circuit_program_to_visual_ir,
)
from qamomile.circuit.visualization.drawer import _prepare_graph_for_visualization
from qamomile.circuit.visualization.drawing_compiler import (
    compile_qkernel_for_drawing,
)
from qamomile.circuit.visualization.layout import CircuitLayoutEngine
from qamomile.circuit.visualization.renderer import MatplotlibRenderer
from qamomile.circuit.visualization.style import DEFAULT_STYLE
from qamomile.circuit.visualization.text_metrics import measure_text
from qamomile.circuit.visualization.types import HorizontalSpan, LayoutResult, Rect
from qamomile.circuit.visualization.visual_ir import (
    VFoldedBlock,
    VFoldedKind,
    VGate,
    VGateKind,
    VInlineBlock,
    VisualCircuit,
    VisualNode,
    VUnfoldedKind,
    VUnfoldedSequence,
)

_GEOMETRY_TOLERANCE = 1e-9


@qmc.qkernel
def while_in_if_geometry() -> qmc.Bit:
    """Build the nested IF/WHILE circuit that exposed escaping borders.

    Returns:
        qmc.Bit: Outer measurement result.
    """
    q0 = qmc.qubit("q0")
    q0 = qmc.x(q0)
    b0 = qmc.measure(q0)
    if b0:
        q1 = qmc.qubit("q1")
        q1 = qmc.h(q1)
        b1 = qmc.measure(q1)
        while b1:
            q2 = qmc.qubit("q2")
            q2 = qmc.h(q2)
            b1 = qmc.measure(q2)
    return b0


@qmc.qkernel
def nested_if_geometry(q0: Qubit, q1: Qubit, q2: Qubit) -> Qubit:
    """Build two nested IFs whose leaf branches share one target wire.

    Args:
        q0 (Qubit): Outer condition qubit.
        q1 (Qubit): Inner condition qubit.
        q2 (Qubit): Shared branch target.

    Returns:
        Qubit: Updated target qubit.
    """
    b0 = qmc.measure(q0)
    if b0:
        b1 = qmc.measure(q1)
        if b1:
            q2 = qmc.x(q2)
        else:
            q2 = qmc.h(q2)
    else:
        q2 = qmc.z(q2)
    return q2


@qmc.qkernel
def long_if_followed_on_other_wire() -> Qubit:
    """Build an asymmetric IF followed by a gate on a different wire.

    Returns:
        Qubit: Qubit updated after the IF.
    """
    q0 = qmc.qubit("q0")
    q1 = qmc.qubit("q1")
    q2 = qmc.qubit("q2")
    bit = qmc.measure(q0)
    if bit:
        q1 = qmc.x(q1)
        q1 = qmc.h(q1)
        q1 = qmc.x(q1)
    else:
        q1 = qmc.h(q1)
    q2 = qmc.z(q2)
    return q2


@qmc.qkernel
def sparse_folded_if_followed_on_middle_wire() -> Qubit:
    """Build a folded IF on q0/q2 followed by a gate on enclosed q1.

    Returns:
        Qubit: Updated middle qubit.
    """
    q0 = qmc.qubit("q0")
    q1 = qmc.qubit("q1")
    q2 = qmc.qubit("q2")
    condition_qubit = qmc.qubit("condition")
    bit = qmc.measure(condition_qubit)
    if bit:
        q0 = qmc.x(q0)
        q2 = qmc.h(q2)
    else:
        q0 = qmc.h(q0)
        q2 = qmc.x(q2)
    q1 = qmc.z(q1)
    return q1


@qmc.qkernel
def if_with_empty_true_branch() -> Qubit:
    """Build an IF whose explicit true branch contains only ``pass``.

    Returns:
        Qubit: Qubit updated by the false branch.
    """
    q0 = qmc.qubit("q0")
    q1 = qmc.qubit("q1")
    bit = qmc.measure(q0)
    if bit:
        pass
    else:
        q1 = qmc.x(q1)
    return q1


@qmc.qkernel
def conditional_helper(q0: Qubit, q1: Qubit) -> Qubit:
    """Apply a measurement-backed IF inside an inlineable helper.

    Args:
        q0 (Qubit): Condition qubit to measure.
        q1 (Qubit): Branch target qubit.

    Returns:
        Qubit: Updated target qubit.
    """
    bit = qmc.measure(q0)
    if bit:
        q1 = qmc.x(q1)
    else:
        q1 = qmc.h(q1)
    return q1


@qmc.qkernel
def inline_conditional_geometry() -> Qubit:
    """Call a helper whose inline border must contain its nested IF.

    Returns:
        Qubit: Updated target returned by the helper.
    """
    q0 = qmc.qubit("q0")
    q1 = qmc.qubit("q1")
    return conditional_helper(q0, q1)


@qmc.qkernel
def symbolic_nested_if_geometry(
    flag0: qmc.UInt,
    flag1: qmc.UInt,
) -> tuple[Qubit, Qubit]:
    """Build nested symbolic IFs between gates on a separate wire.

    Args:
        flag0 (qmc.UInt): Outer compile-time condition value.
        flag1 (qmc.UInt): Inner compile-time condition value.

    Returns:
        tuple[Qubit, Qubit]: Updated independent and conditional qubits.
    """
    q0 = qmc.qubit("q0")
    q1 = qmc.qubit("q1")
    q0 = qmc.h(q0)
    if flag0 == 1:
        if flag1 == 1:
            q1 = qmc.x(q1)
    q0 = qmc.z(q0)
    return q0, q1


@qmc.qkernel
def nested_conditional_helper(target: Qubit) -> Qubit:
    """Apply a conditional gate using a helper-local measurement.

    Args:
        target (Qubit): Target qubit updated by the true branch.

    Returns:
        Qubit: Updated target qubit.
    """
    condition = qmc.qubit("condition")
    bit = qmc.measure(condition)
    if bit:
        target = qmc.x(target)
    return target


@qmc.qkernel
def inline_between_gates_geometry() -> tuple[Qubit, Qubit]:
    """Place a nested conditional helper between unrelated gates.

    Returns:
        tuple[Qubit, Qubit]: Updated independent and helper target qubits.
    """
    q0 = qmc.qubit("q0")
    q1 = qmc.qubit("q1")
    q0 = qmc.h(q0)
    q1 = nested_conditional_helper(q1)
    q0 = qmc.z(q0)
    return q0, q1


@qmc.qkernel
def fresh_loop_helper(target: Qubit) -> Qubit:
    """Create and measure a fresh qubit inside an unrolled helper loop.

    Args:
        target (Qubit): Unchanged helper target used to anchor the call.

    Returns:
        Qubit: Original target qubit.
    """
    for _ in qmc.range(1):
        fresh = qmc.qubit("fresh")
        fresh = qmc.x(fresh)
        qmc.measure(fresh)
    return target


@qmc.qkernel
def inline_fresh_loop_geometry() -> tuple[Qubit, Qubit]:
    """Call a helper whose unrolled loop introduces a fresh wire.

    Returns:
        tuple[Qubit, Qubit]: Independent and helper target qubits.
    """
    q0 = qmc.qubit("q0")
    q1 = qmc.qubit("q1")
    q0 = qmc.h(q0)
    q1 = fresh_loop_helper(q1)
    return q0, q1


@qmc.qkernel
def long_gate_label_geometry(
    WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW: qmc.Float,
) -> Qubit:
    """Build a parametric gate whose label contains wide glyphs.

    Args:
        WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW (qmc.Float): Runtime rotation angle
            with a deliberately wide display name.

    Returns:
        Qubit: Updated gate target.
    """
    q0 = qmc.qubit("q0")
    q0 = qmc.rx(q0, WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW)
    return qmc.h(q0)


@qmc.qkernel
def long_if_header_geometry(
    WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW: qmc.UInt,
) -> Qubit:
    """Build a symbolic IF whose header contains wide glyphs.

    Args:
        WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW (qmc.UInt): Runtime condition value
            with a deliberately wide display name.

    Returns:
        Qubit: Conditional target qubit.
    """
    q0 = qmc.qubit("q0")
    if WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW == 1:
        q0 = qmc.x(q0)
    return q0


@qmc.qkernel
def long_wire_label_geometry() -> Qubit:
    """Build a circuit with a deliberately wide qubit name.

    Returns:
        Qubit: Named qubit after one gate.
    """
    q0 = qmc.qubit("WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW")
    return qmc.h(q0)


@qmc.qkernel
def multiline_wire_label_geometry() -> Qubit:
    """Build a circuit with a tall multiline qubit name.

    Returns:
        Qubit: Named qubit after one gate.
    """
    q0 = qmc.qubit("wire\nwire\nwire\nwire\nwire\nwire\nwire\nwire\nwire\nwire\nwire")
    return qmc.h(q0)


def _walk(nodes: Iterable[VisualNode]) -> Iterator[VisualNode]:
    """Yield every visual node reachable from the supplied roots.

    Args:
        nodes (Iterable[VisualNode]): Root nodes to traverse.

    Yields:
        VisualNode: Each node in pre-order.
    """
    for node in nodes:
        yield node
        if isinstance(node, VInlineBlock):
            yield from _walk(node.children)
        elif isinstance(node, VUnfoldedSequence):
            for iteration in node.iterations:
                yield from _walk(iteration)


def _visual_circuit(
    kernel: Any,
    *,
    inline: bool = False,
    fold_loops: bool = True,
    fold_ifs: bool = False,
    fold_whiles: bool = False,
) -> VisualCircuit:
    """Build the visual IR needed by the pure layout engine.

    Args:
        kernel (Any): QKernel-like object to trace.
        inline (bool): Whether to expand call-block contents. Defaults to False.
        fold_loops (bool): Whether to fold finite FOR loops. Defaults to True.
        fold_ifs (bool): Whether to fold IF nodes. Defaults to False.
        fold_whiles (bool): Whether to fold WHILE nodes. Defaults to False.

    Returns:
        VisualCircuit: Exact compiler-backed circuit without renderer involvement.
    """
    drawing = compile_qkernel_for_drawing(kernel)
    return circuit_program_to_visual_ir(
        drawing.circuit,
        trace=drawing.trace,
        style=DEFAULT_STYLE,
        qubit_names=drawing.qubit_names,
        output_names=drawing.output_names,
        expectation_value_qubits=drawing.expectation_value_qubits,
        expand_calls=inline,
        fold_loops=fold_loops,
        fold_ifs=fold_ifs,
        fold_whiles=fold_whiles,
        inline_depth=None,
    )


def _compute_layout(
    kernel: Any,
    *,
    inline: bool = False,
    fold_loops: bool = True,
    fold_ifs: bool = False,
    fold_whiles: bool = False,
) -> tuple[VisualCircuit, LayoutResult]:
    """Return visual IR and its renderer-independent geometry.

    Args:
        kernel (Any): QKernel-like object to lay out.
        inline (bool): Whether to expand call-block contents. Defaults to False.
        fold_loops (bool): Whether to fold finite FOR loops. Defaults to True.
        fold_ifs (bool): Whether to fold IF nodes. Defaults to False.
        fold_whiles (bool): Whether to fold WHILE nodes. Defaults to False.

    Returns:
        tuple[VisualCircuit, LayoutResult]: Visual IR and computed geometry.
    """
    visual_circuit = _visual_circuit(
        kernel,
        inline=inline,
        fold_loops=fold_loops,
        fold_ifs=fold_ifs,
        fold_whiles=fold_whiles,
    )
    layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(visual_circuit)
    return visual_circuit, layout


def _legacy_symbolic_condition_layout(
    kernel: Any,
) -> tuple[VisualCircuit, LayoutResult]:
    """Build synthetic non-measurement IF geometry for layout-only tests.

    Exact circuit lowering intentionally rejects runtime quantum conditions
    that do not originate from a measurement. These two geometry tests need a
    symbolic header only, so they use the compatibility analyzer as a fixture
    factory without exercising the public drawing path.

    Args:
        kernel (Any): Synthetic qkernel with an unbound classical IF condition.

    Returns:
        tuple[VisualCircuit, LayoutResult]: Fixture Visual IR and pure layout.
    """
    block = _prepare_graph_for_visualization(kernel._build_graph_for_visualization())
    analyzer = CircuitAnalyzer(block, DEFAULT_STYLE, fold_ifs=False)
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(block)
    visual = analyzer.build_visual_ir(block, qubit_map, qubit_names, num_qubits)
    return visual, CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(visual)


def _assert_contains_with_margin(outer: Rect, inner: Rect) -> None:
    """Assert that ``outer`` contains ``inner`` with positive margin on all sides.

    Args:
        outer (Rect): Parent rectangle.
        inner (Rect): Nested rectangle.
    """
    horizontal_margin = DEFAULT_STYLE.box_padding_x
    vertical_margin = DEFAULT_STYLE.box_padding_y
    assert outer.left <= inner.left - horizontal_margin + _GEOMETRY_TOLERANCE
    assert outer.right >= inner.right + horizontal_margin - _GEOMETRY_TOLERANCE
    assert outer.bottom <= inner.bottom - vertical_margin + _GEOMETRY_TOLERANCE
    assert outer.top >= inner.top + vertical_margin - _GEOMETRY_TOLERANCE


def _top_level_gate(vc: VisualCircuit, gate_type: GateOperationType) -> VGate:
    """Return the unique top-level gate of the requested type.

    Args:
        vc (VisualCircuit): Visual circuit to inspect.
        gate_type (GateOperationType): Gate type to find.

    Returns:
        VGate: Matching top-level gate.

    Raises:
        AssertionError: If the requested gate is not unique at the top level.
    """
    matches = [
        node
        for node in vc.children
        if isinstance(node, VGate) and node.gate_type == gate_type
    ]
    assert len(matches) == 1
    return matches[0]


def _rects_overlap(first: Rect, second: Rect) -> bool:
    """Return whether two rectangles have positive-area intersection.

    Args:
        first (Rect): First rectangle.
        second (Rect): Second rectangle.

    Returns:
        bool: True when the rectangles overlap in both dimensions.
    """
    return (
        first.left < second.right - _GEOMETRY_TOLERANCE
        and first.right > second.left + _GEOMETRY_TOLERANCE
        and first.bottom < second.top - _GEOMETRY_TOLERANCE
        and first.top > second.bottom + _GEOMETRY_TOLERANCE
    )


def _rendered_text_bounds(fig: Figure, label: str) -> Rect:
    """Return one rendered text artist's bounds in layout coordinates.

    Args:
        fig (Figure): Rendered circuit figure.
        label (str): Exact text value identifying the artist.

    Returns:
        Rect: Artist bounds transformed from pixels into data coordinates.

    Raises:
        AssertionError: If the requested text artist is not unique.
    """
    fig.canvas.draw()
    ax = fig._qm_ax  # type: ignore[attr-defined]
    matches = [text for text in ax.texts if text.get_text() == label]
    assert len(matches) == 1
    bounds = matches[0].get_window_extent(fig.canvas.get_renderer())
    data_bounds = bounds.transformed(ax.transData.inverted())
    return Rect(data_bounds.x0, data_bounds.y0, data_bounds.x1, data_bounds.y1)


class TestNestedControlFlowGeometry:
    """Nested control-flow rectangles remain inside their parent regions."""

    def test_outer_if_true_box_contains_inner_while_on_all_sides(self):
        """The IF true region encloses the nested WHILE with visible margins."""
        vc, layout = _compute_layout(while_in_if_geometry)
        outer_if = next(
            node
            for node in vc.children
            if isinstance(node, VUnfoldedSequence) and node.kind == VUnfoldedKind.IF
        )
        inner_while = next(
            node
            for node in _walk(outer_if.iterations[0])
            if isinstance(node, VUnfoldedSequence) and node.kind == VUnfoldedKind.WHILE
        )

        outer_true_rect = layout.control_flow_layouts[outer_if.node_key].boxes[0].rect
        inner_rect = layout.control_flow_layouts[inner_while.node_key].outer_rect
        _assert_contains_with_margin(outer_true_rect, inner_rect)

    def test_outer_if_true_box_contains_every_nested_if_branch(self):
        """The outer true region encloses both branches of its nested IF."""
        vc, layout = _compute_layout(nested_if_geometry)
        outer_if = next(
            node
            for node in vc.children
            if isinstance(node, VUnfoldedSequence) and node.kind == VUnfoldedKind.IF
        )
        inner_if = next(
            node
            for node in _walk(outer_if.iterations[0])
            if isinstance(node, VUnfoldedSequence) and node.kind == VUnfoldedKind.IF
        )

        outer_true_rect = layout.control_flow_layouts[outer_if.node_key].boxes[0].rect
        inner_layout = layout.control_flow_layouts[inner_if.node_key]
        assert len(inner_layout.boxes) == 2
        for inner_box in inner_layout.boxes:
            _assert_contains_with_margin(outer_true_rect, inner_box.rect)

    def test_inline_block_contains_and_exports_nested_if_layout(self):
        """An inline border contains its IF and retains the IF placement map."""
        vc, layout = _compute_layout(inline_conditional_geometry, inline=True)
        inline_block = next(
            node for node in vc.children if isinstance(node, VInlineBlock)
        )
        nested_if = next(
            node
            for node in _walk(inline_block.children)
            if isinstance(node, VUnfoldedSequence) and node.kind == VUnfoldedKind.IF
        )

        inline_rect = layout.inline_block_layouts[inline_block.node_key].inner_rect
        nested_rect = layout.control_flow_layouts[nested_if.node_key].outer_rect
        _assert_contains_with_margin(inline_rect, nested_rect)


class TestSequentialControlFlowGeometry:
    """Control-flow extents reserve horizontal space for later operations."""

    def test_control_flow_barrier_never_rewinds_a_longer_wire_frontier(self):
        """A short IF cannot move a previously occupied wire edge leftward."""
        wide_gate = VGate(
            node_key=("wide",),
            label="wide",
            qubit_indices=[0],
            estimated_width=10.0,
            kind=VGateKind.GATE,
            gate_type=GateOperationType.X,
        )
        branch_gate = VGate(
            node_key=("if", "true", "x"),
            label="X",
            qubit_indices=[1],
            estimated_width=DEFAULT_STYLE.gate_width,
            kind=VGateKind.GATE,
            gate_type=GateOperationType.X,
        )
        if_node = VUnfoldedSequence(
            node_key=("if",),
            iterations=[[branch_gate]],
            affected_qubits=[1],
            kind=VUnfoldedKind.IF,
            condition_label="if flag:",
            condition_label_width=1.2,
            branch_label_widths=[1.2],
        )
        following_gate = VGate(
            node_key=("following",),
            label="Z",
            qubit_indices=[0],
            estimated_width=DEFAULT_STYLE.gate_width,
            kind=VGateKind.GATE,
            gate_type=GateOperationType.Z,
        )
        vc = VisualCircuit(
            children=[wide_gate, if_node, following_gate],
            qubit_map={"q0": 0, "q1": 1},
            qubit_names={0: "q0", 1: "q1"},
            num_qubits=2,
        )

        layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(vc)
        following_left = layout.node_spans[following_gate.node_key].left

        assert (
            following_left
            >= layout.node_spans[wide_gate.node_key].right
            + DEFAULT_STYLE.gate_gap
            - _GEOMETRY_TOLERANCE
        )

    def test_gate_on_other_wire_does_not_intersect_asymmetric_if(self):
        """A packed post-IF gate remains outside the IF's vertical footprint."""
        vc, layout = _compute_layout(long_if_followed_on_other_wire)
        outer_if = next(
            node
            for node in vc.children
            if isinstance(node, VUnfoldedSequence) and node.kind == VUnfoldedKind.IF
        )
        following_gate = _top_level_gate(vc, GateOperationType.Z)

        assert outer_if.affected_qubits == [1]
        assert following_gate.qubit_indices == [2]
        assert not _rects_overlap(
            layout.control_flow_layouts[outer_if.node_key].outer_rect,
            layout.node_rects[following_gate.node_key],
        )

    def test_sparse_folded_box_reserves_its_intermediate_wire(self):
        """A q0/q2 folded IF cannot overlap the following gate on q1."""
        vc, layout = _compute_layout(
            sparse_folded_if_followed_on_middle_wire,
            fold_ifs=True,
        )
        folded_if = next(
            node
            for node in _walk(vc.children)
            if isinstance(node, VFoldedBlock) and node.kind == VFoldedKind.IF
        )
        following_gate = _top_level_gate(vc, GateOperationType.Z)

        assert folded_if.affected_qubits == [0, 2]
        assert following_gate.qubit_indices == [1]
        folded_right = layout.folded_block_layouts[folded_if.node_key].rect.right
        following_left = layout.node_spans[following_gate.node_key].left
        assert (
            following_left
            >= folded_right + DEFAULT_STYLE.gate_gap - _GEOMETRY_TOLERANCE
        )


class TestContainerBarrierGeometry:
    """Container borders remain disjoint from unrelated sequential gates."""

    def test_symbolic_nested_if_does_not_cover_neighbor_wire_gates(self):
        """A nested symbolic IF starts after H and ends before Z on q0."""
        vc, layout = _legacy_symbolic_condition_layout(symbolic_nested_if_geometry)
        outer_if = next(
            node
            for node in vc.children
            if isinstance(node, VUnfoldedSequence) and node.kind == VUnfoldedKind.IF
        )
        container = layout.control_flow_layouts[outer_if.node_key].outer_rect

        for gate_type in (GateOperationType.H, GateOperationType.Z):
            gate = _top_level_gate(vc, gate_type)
            assert not _rects_overlap(container, layout.node_rects[gate.node_key])

    def test_nested_inline_border_clears_gates_and_unrelated_wire(self):
        """An inline conditional clears adjacent q0 gates and the q0 wire."""
        vc, layout = _compute_layout(inline_between_gates_geometry, inline=True)
        inline_block = next(
            node for node in vc.children if isinstance(node, VInlineBlock)
        )
        container = layout.inline_block_layouts[inline_block.node_key].outer_rect

        for gate_type in (GateOperationType.H, GateOperationType.Z):
            gate = _top_level_gate(vc, gate_type)
            assert not _rects_overlap(container, layout.node_rects[gate.node_key])
        assert not container.bottom < layout.qubit_y[0] < container.top

    def test_inline_border_contains_fresh_wire_from_unrolled_loop(self):
        """Inline padding applies to descendant wires absent from call results."""
        vc, layout = _compute_layout(
            inline_fresh_loop_geometry,
            inline=True,
            fold_loops=False,
        )
        inline_block = next(
            node for node in vc.children if isinstance(node, VInlineBlock)
        )
        fresh_x = next(
            node
            for node in _walk(inline_block.children)
            if isinstance(node, VGate) and node.gate_type == GateOperationType.X
        )
        container = layout.inline_block_layouts[inline_block.node_key].inner_rect

        _assert_contains_with_margin(container, layout.node_rects[fresh_x.node_key])

    def test_controlled_inline_clears_prior_gate_on_control_wire(self):
        """A controlled inline block reserves its external control wire."""
        style = DEFAULT_STYLE
        prior = VGate(
            node_key=("prior",),
            label="wide",
            qubit_indices=[0],
            estimated_width=3.0,
            kind=VGateKind.GATE,
            gate_type=GateOperationType.H,
        )
        child = VGate(
            node_key=("inline", "child"),
            label="X",
            qubit_indices=[1],
            estimated_width=style.gate_width,
            kind=VGateKind.GATE,
            gate_type=GateOperationType.X,
        )
        inline = VInlineBlock(
            node_key=("inline",),
            label="controlled_helper",
            children=[child],
            affected_qubits=[0, 1],
            control_qubit_indices=[0],
            power=1,
            depth=0,
            border_padding=style.border_padding_base,
            max_gate_width=style.gate_width,
            label_width=2.0,
            content_width=style.gate_width,
            final_width=2.0,
        )
        vc = VisualCircuit(
            children=[prior, inline],
            qubit_map={"control": 0, "target": 1},
            qubit_names={0: "control", 1: "target"},
            num_qubits=2,
        )
        layout = CircuitLayoutEngine(style).compute_layout(vc)

        assert (
            layout.node_spans[inline.node_key].left
            >= layout.node_spans[prior.node_key].right
            + style.gate_gap
            - _GEOMETRY_TOLERANCE
        )

    def test_independent_control_flow_does_not_serialize_every_wire(self):
        """Disjoint IF boxes remain compact instead of forming a global chain."""
        num_qubits = 32
        nodes: list[VisualNode] = []
        for qubit in range(num_qubits):
            child = VGate(
                node_key=("if", qubit, "child"),
                label="X",
                qubit_indices=[qubit],
                estimated_width=DEFAULT_STYLE.gate_width,
                kind=VGateKind.GATE,
                gate_type=GateOperationType.X,
            )
            nodes.append(
                VUnfoldedSequence(
                    node_key=("if", qubit),
                    iterations=[[child]],
                    affected_qubits=[qubit],
                    kind=VUnfoldedKind.IF,
                    condition_label=f"if flag{qubit}:",
                    condition_label_width=2.0,
                    branch_label_widths=[2.0],
                )
            )
        vc = VisualCircuit(
            children=nodes,
            qubit_map={f"q{qubit}": qubit for qubit in range(num_qubits)},
            qubit_names={qubit: f"q{qubit}" for qubit in range(num_qubits)},
            num_qubits=num_qubits,
        )
        layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(vc)

        assert layout.viewport.width < 10.0

    def test_condition_connector_routes_around_intermediate_wire_gate(self):
        """An orthogonal IF connector clears gates on crossed wire rows."""
        measurement = VGate(
            node_key=("measure",),
            label="M",
            qubit_indices=[0],
            estimated_width=DEFAULT_STYLE.gate_width,
            kind=VGateKind.MEASURE,
        )
        intermediate = VGate(
            node_key=("intermediate",),
            label="wide",
            qubit_indices=[1],
            estimated_width=4.0,
            kind=VGateKind.GATE,
            gate_type=GateOperationType.RX,
        )
        child = VGate(
            node_key=("if", "child"),
            label="H",
            qubit_indices=[3],
            estimated_width=DEFAULT_STYLE.gate_width,
            kind=VGateKind.GATE,
            gate_type=GateOperationType.H,
        )
        if_node = VUnfoldedSequence(
            node_key=("if",),
            iterations=[[child]],
            affected_qubits=[3],
            kind=VUnfoldedKind.IF,
            condition_label="if q0_measured:",
            condition_label_width=2.5,
            branch_label_widths=[2.5],
            condition_measure_node_key=measurement.node_key,
            condition_measure_qubit_indices=[0],
        )
        vc = VisualCircuit(
            children=[measurement, intermediate, if_node],
            qubit_map={f"q{qubit}": qubit for qubit in range(4)},
            qubit_names={qubit: f"q{qubit}" for qubit in range(4)},
            num_qubits=4,
        )
        layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(vc)
        segments = layout.control_flow_layouts[if_node.node_key].connector_segments
        intermediate_rect = layout.node_rects[intermediate.node_key]

        assert len(segments) == 2
        horizontal, vertical = segments
        assert horizontal.start_y == horizontal.end_y == layout.qubit_y[0]
        assert vertical.start_x == vertical.end_x
        assert (
            vertical.start_x
            >= intermediate_rect.right + DEFAULT_STYLE.gate_gap - _GEOMETRY_TOLERANCE
        )

    def test_vector_measurement_connector_keeps_every_source_wire(self) -> None:
        """A multi-bit condition routes from every exact vector source."""
        measurement = VGate(
            node_key=("vector-measure",),
            label="M",
            qubit_indices=[0, 1],
            estimated_width=DEFAULT_STYLE.gate_width,
            kind=VGateKind.MEASURE_VECTOR,
        )
        child = VGate(
            node_key=("if", "child"),
            label="X",
            qubit_indices=[2],
            estimated_width=DEFAULT_STYLE.gate_width,
            kind=VGateKind.GATE,
            gate_type=GateOperationType.X,
        )
        if_node = VUnfoldedSequence(
            node_key=("if",),
            iterations=[[child]],
            affected_qubits=[2],
            kind=VUnfoldedKind.IF,
            condition_label="if c[0] and c[1]:",
            condition_label_width=2.5,
            branch_label_widths=[2.5],
            condition_measure_node_key=measurement.node_key,
            condition_measure_qubit_indices=[0, 1],
        )
        circuit = VisualCircuit(
            children=[measurement, if_node],
            qubit_map={f"q{qubit}": qubit for qubit in range(3)},
            qubit_names={qubit: f"q{qubit}" for qubit in range(3)},
            num_qubits=3,
        )

        layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(circuit)
        segments = layout.control_flow_layouts[if_node.node_key].connector_segments
        source_x = layout.node_spans[measurement.node_key].right
        source_points = {(segment.start_x, segment.start_y) for segment in segments}

        assert (source_x, layout.qubit_y[0]) in source_points
        assert (source_x, layout.qubit_y[1]) in source_points


class TestExactPrimitiveGeometry:
    """Layout includes wrapper, marker, and rounded-connector extents."""

    def test_powered_controlled_wrapper_matches_rendered_rectangle(self):
        """The renderer consumes the exact target wrapper resolved by layout."""
        inner_width = 2.0
        power = 3
        gate = VGate(
            node_key=("controlled",),
            label="$R_x$(angle=0.25)",
            qubit_indices=[0, 1],
            estimated_width=inner_width + 2 * DEFAULT_STYLE.power_wrapper_margin,
            kind=VGateKind.CONTROLLED_U_BOX,
            gate_type=GateOperationType.RX,
            box_width=inner_width,
            control_count=1,
            power=power,
        )
        vc = VisualCircuit(
            children=[gate],
            qubit_map={"control": 0, "target": 1},
            qubit_names={0: "control", 1: "target"},
            num_qubits=2,
        )
        layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(vc)
        placement = layout.powered_gate_layouts[gate.node_key]
        fig = MatplotlibRenderer(DEFAULT_STYLE).render(vc, layout)
        ax = fig._qm_ax  # type: ignore[attr-defined]
        wrappers = [
            patch
            for patch in ax.patches
            if type(patch) is mpatches.Rectangle and patch.get_linestyle() == "--"
        ]

        assert len(wrappers) == 1
        wrapper = wrappers[0]
        actual = Rect(
            wrapper.get_x(),
            wrapper.get_y(),
            wrapper.get_x() + wrapper.get_width(),
            wrapper.get_y() + wrapper.get_height(),
        )
        assert math.isclose(
            actual.left,
            placement.wrapper_rect.left,
            abs_tol=_GEOMETRY_TOLERANCE,
        )
        assert math.isclose(
            actual.bottom,
            placement.wrapper_rect.bottom,
            abs_tol=_GEOMETRY_TOLERANCE,
        )
        assert math.isclose(
            actual.right,
            placement.wrapper_rect.right,
            abs_tol=_GEOMETRY_TOLERANCE,
        )
        assert math.isclose(
            actual.top,
            placement.wrapper_rect.top,
            abs_tol=_GEOMETRY_TOLERANCE,
        )

    def test_folded_markers_are_inside_reserved_node_span(self):
        """Participation-marker radii cannot consume the next gate gap."""
        style = replace(DEFAULT_STYLE, gate_gap=0.01)
        folded = VFoldedBlock(
            node_key=("folded",),
            header_label="if flag:",
            body_lines=[],
            affected_qubits=[0],
            folded_width=style.gate_width,
            kind=VFoldedKind.IF,
            affected_qubits_precise=True,
        )
        following = VGate(
            node_key=("following",),
            label="H",
            qubit_indices=[0],
            estimated_width=style.gate_width,
            kind=VGateKind.GATE,
            gate_type=GateOperationType.H,
        )
        vc = VisualCircuit(
            children=[folded, following],
            qubit_map={"q0": 0},
            qubit_names={0: "q0"},
            num_qubits=1,
        )
        layout = CircuitLayoutEngine(style).compute_layout(vc)

        assert math.isclose(
            layout.node_spans[folded.node_key].right,
            layout.folded_block_layouts[folded.node_key].rect.right
            + style.folded_marker_radius,
            abs_tol=_GEOMETRY_TOLERANCE,
        )
        assert (
            layout.node_spans[following.node_key].left
            >= layout.node_rects[folded.node_key].right
            + style.gate_gap
            - _GEOMETRY_TOLERANCE
        )

        fig = MatplotlibRenderer(style).render(vc, layout)
        ax = fig._qm_ax  # type: ignore[attr-defined]
        markers = [
            patch
            for patch in ax.patches
            if isinstance(patch, mpatches.Circle)
            and math.isclose(
                patch.get_radius(),
                style.folded_marker_radius,
                abs_tol=_GEOMETRY_TOLERANCE,
            )
        ]
        expected_x = {
            layout.folded_block_layouts[folded.node_key].rect.left,
            layout.folded_block_layouts[folded.node_key].rect.right,
        }
        assert len(markers) == 2
        assert {marker.center[0] for marker in markers} == expected_x
        assert all(
            math.isclose(
                marker.center[1],
                layout.qubit_y[0],
                abs_tol=_GEOMETRY_TOLERANCE,
            )
            for marker in markers
        )


class TestRenderedTextGeometry:
    """Font-aware layout contains wide glyphs in boxes and the viewport."""

    def test_wide_parametric_gate_text_stays_inside_its_box(self):
        """A long W-heavy rotation label cannot collide with the next gate."""
        vc, layout = _compute_layout(long_gate_label_geometry)
        gate = _top_level_gate(vc, GateOperationType.RX)
        following = _top_level_gate(vc, GateOperationType.H)
        fig = MatplotlibRenderer(DEFAULT_STYLE).render(vc, layout)
        text_rect = _rendered_text_bounds(fig, gate.label)
        gate_rect = layout.node_rects[gate.node_key]

        assert gate_rect.left <= text_rect.left
        assert gate_rect.right >= text_rect.right
        assert (
            layout.node_spans[following.node_key].left
            >= gate_rect.right + DEFAULT_STYLE.gate_gap - _GEOMETRY_TOLERANCE
        )

    def test_wide_symbolic_if_header_stays_inside_branch_box(self):
        """A long W-heavy symbolic condition remains inside its IF header."""
        vc, layout = _legacy_symbolic_condition_layout(long_if_header_geometry)
        if_node = next(
            node
            for node in vc.children
            if isinstance(node, VUnfoldedSequence) and node.kind == VUnfoldedKind.IF
        )
        branch = layout.control_flow_layouts[if_node.node_key].boxes[0]
        fig = MatplotlibRenderer(DEFAULT_STYLE).render(vc, layout)
        text_rect = _rendered_text_bounds(fig, branch.label)

        assert branch.rect.left <= text_rect.left
        assert branch.rect.right >= text_rect.right

    def test_wide_wire_label_stays_inside_viewport(self):
        """A long W-heavy qubit name remains visible on the physical canvas."""
        vc, layout = _compute_layout(long_wire_label_geometry)
        label = vc.qubit_names[0]
        fig = MatplotlibRenderer(DEFAULT_STYLE).render(vc, layout)
        text_rect = _rendered_text_bounds(fig, label)

        assert layout.viewport.left <= text_rect.left
        assert layout.viewport.right >= text_rect.right

    def test_font_family_change_invalidates_text_measurement(self):
        """Changing matplotlib defaults cannot reuse stale label widths."""
        with rc_context({"font.family": "monospace"}):
            _compute_layout(long_gate_label_geometry)

        with rc_context({"font.family": "sans-serif"}):
            vc, layout = _compute_layout(long_gate_label_geometry)
            gate = _top_level_gate(vc, GateOperationType.RX)
            fig = MatplotlibRenderer(DEFAULT_STYLE).render(vc, layout)
            text_rect = _rendered_text_bounds(fig, gate.label)

        gate_rect = layout.node_rects[gate.node_key]
        assert gate_rect.left <= text_rect.left
        assert gate_rect.right >= text_rect.right

    def test_text_metrics_follow_parse_math_and_ignore_default_weight(self):
        """Renderer and layout share math parsing and regular-weight rules."""
        with rc_context({"text.parse_math": False, "font.weight": "bold"}):
            vc, layout = _compute_layout(long_gate_label_geometry)
            gate = _top_level_gate(vc, GateOperationType.RX)
            fig = MatplotlibRenderer(DEFAULT_STYLE).render(vc, layout)
            text_rect = _rendered_text_bounds(fig, gate.label)

        gate_rect = layout.node_rects[gate.node_key]
        assert gate_rect.left <= text_rect.left
        assert gate_rect.right >= text_rect.right

    def test_multiline_wire_label_stays_inside_viewport(self):
        """Every line of a tall qubit name remains on the physical canvas."""
        vc, layout = _compute_layout(multiline_wire_label_geometry)
        label = vc.qubit_names[0]
        fig = MatplotlibRenderer(DEFAULT_STYLE).render(vc, layout)
        text_rect = _rendered_text_bounds(fig, label)

        assert layout.viewport.bottom <= text_rect.bottom
        assert layout.viewport.top >= text_rect.top

    def test_large_custom_font_expands_gate_box_and_viewport(self):
        """A large configured font enlarges its gate instead of escaping it."""
        style = replace(DEFAULT_STYLE, font_size=100)
        gate = VGate(
            node_key=("large-font",),
            label="H",
            qubit_indices=[0],
            estimated_width=2.0,
            kind=VGateKind.GATE,
            gate_type=GateOperationType.H,
        )
        vc = VisualCircuit(
            children=[gate],
            qubit_map={"q0": 0},
            qubit_names={0: "q0"},
            num_qubits=1,
        )
        layout = CircuitLayoutEngine(style).compute_layout(vc)
        box_rect = layout.gate_box_rects[gate.node_key]
        fig = MatplotlibRenderer(style).render(vc, layout)
        text_rect = _rendered_text_bounds(fig, gate.label)

        assert box_rect.height > style.gate_height
        assert box_rect.left <= text_rect.left
        assert box_rect.right >= text_rect.right
        assert box_rect.bottom <= text_rect.bottom
        assert box_rect.top >= text_rect.top
        assert layout.viewport.bottom <= text_rect.bottom
        assert layout.viewport.top >= text_rect.top

    def test_multiline_math_wire_label_uses_renderer_line_spacing(self):
        """Mathtext line spacing is measured by the same Agg renderer."""
        label = "\n".join([r"$\sum_{i=0}^{99}$"] * 20)
        gate = VGate(
            node_key=("wide",),
            label="H",
            qubit_indices=[0],
            estimated_width=6.0,
            kind=VGateKind.GATE,
            gate_type=GateOperationType.H,
        )
        vc = VisualCircuit(
            children=[gate],
            qubit_map={"q0": 0},
            qubit_names={0: label},
            num_qubits=1,
        )
        layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(vc)
        fig = MatplotlibRenderer(DEFAULT_STYLE).render(vc, layout)
        text_rect = _rendered_text_bounds(fig, label)

        assert layout.viewport.bottom <= text_rect.bottom
        assert layout.viewport.top >= text_rect.top

    def test_large_subfont_expands_folded_summary_box(self):
        """A folded multiline label remains inside a large-subfont box."""
        style = replace(DEFAULT_STYLE, subfont_size=40)
        folded = VFoldedBlock(
            node_key=("folded-large-font",),
            header_label="if flag:",
            body_lines=["first", "second", "third"],
            affected_qubits=[0],
            folded_width=10.0,
            kind=VFoldedKind.IF,
        )
        vc = VisualCircuit(
            children=[folded],
            qubit_map={"q0": 0},
            qubit_names={0: "q0"},
            num_qubits=1,
        )
        layout = CircuitLayoutEngine(style).compute_layout(vc)
        box_rect = layout.folded_block_layouts[folded.node_key].rect
        fig = MatplotlibRenderer(style).render(vc, layout)
        label = "\n".join([folded.header_label, *folded.body_lines])
        text_rect = _rendered_text_bounds(fig, label)

        assert box_rect.bottom <= text_rect.bottom
        assert box_rect.top >= text_rect.top


class TestControlFlowBoundaryGeometry:
    """Control-flow boxes and connectors expose complete boundary geometry."""

    def test_empty_branch_has_nonzero_rectangular_bounds(self):
        """An explicit empty branch still owns a visible nonzero rectangle."""
        vc, layout = _compute_layout(if_with_empty_true_branch)
        if_node = next(
            node
            for node in vc.children
            if isinstance(node, VUnfoldedSequence) and node.kind == VUnfoldedKind.IF
        )
        true_rect = layout.control_flow_layouts[if_node.node_key].boxes[0].rect

        assert true_rect.right > true_rect.left
        assert true_rect.top > true_rect.bottom

    def test_nested_connectors_join_measurement_right_to_box_left(self):
        """Nested IF and WHILE connectors terminate exactly on both bounds."""
        vc, layout = _compute_layout(while_in_if_geometry)
        fig = MatplotlibRenderer(DEFAULT_STYLE).render(vc, layout)
        ax = fig._qm_ax  # type: ignore[attr-defined]
        control_nodes = [
            node
            for node in _walk(vc.children)
            if isinstance(node, VUnfoldedSequence)
            and node.kind in (VUnfoldedKind.IF, VUnfoldedKind.WHILE)
        ]
        assert len(control_nodes) == 2

        for node in control_nodes:
            assert node.condition_measure_node_key is not None
            control_layout = layout.control_flow_layouts[node.node_key]
            assert control_layout.connector_segments
            first_segment = control_layout.connector_segments[0]
            last_segment = control_layout.connector_segments[-1]
            measurement_span = layout.node_spans[node.condition_measure_node_key]
            assert math.isclose(
                first_segment.start_x,
                measurement_span.right,
                abs_tol=_GEOMETRY_TOLERANCE,
            )
            assert math.isclose(
                last_segment.end_x,
                control_layout.outer_rect.left,
                abs_tol=_GEOMETRY_TOLERANCE,
            )
            corner_radius = min(
                DEFAULT_STYLE.gate_corner_radius,
                control_layout.outer_rect.width / 2,
                control_layout.outer_rect.height / 2,
            )
            assert (
                control_layout.outer_rect.bottom + corner_radius
                <= last_segment.end_y
                <= control_layout.outer_rect.top - corner_radius
            )
            expected_x = [
                first_segment.start_x,
                *(segment.end_x for segment in control_layout.connector_segments),
            ]
            expected_y = [
                first_segment.start_y,
                *(segment.end_y for segment in control_layout.connector_segments),
            ]
            assert any(
                len(line.get_xdata()) == len(expected_x)
                and all(
                    math.isclose(actual, expected, abs_tol=_GEOMETRY_TOLERANCE)
                    for actual, expected in zip(
                        line.get_xdata(), expected_x, strict=True
                    )
                )
                and all(
                    math.isclose(actual, expected, abs_tol=_GEOMETRY_TOLERANCE)
                    for actual, expected in zip(
                        line.get_ydata(), expected_y, strict=True
                    )
                )
                for line in ax.lines
            )

    def test_renderer_does_not_mutate_authoritative_layout(self):
        """Rendering consumes final geometry without changing LayoutResult."""
        vc, layout = _compute_layout(while_in_if_geometry)
        before = deepcopy(layout)

        MatplotlibRenderer(DEFAULT_STYLE).render(vc, layout)

        assert layout == before

    def test_renderer_uses_layout_wire_spans_and_viewport(self):
        """Wire artists and axes limits exactly consume LayoutResult geometry."""
        vc, layout = _compute_layout(while_in_if_geometry)
        fig = MatplotlibRenderer(DEFAULT_STYLE).render(vc, layout)
        ax = fig._qm_ax  # type: ignore[attr-defined]

        assert ax.get_xlim() == (layout.viewport.left, layout.viewport.right)
        assert ax.get_ylim() == (layout.viewport.bottom, layout.viewport.top)
        for qubit, span in layout.wire_spans.items():
            wire_y = layout.qubit_y[qubit]
            assert any(
                all(
                    math.isclose(actual, expected, abs_tol=_GEOMETRY_TOLERANCE)
                    for actual, expected in zip(
                        [*line.get_xdata(), *line.get_ydata()],
                        [span.left, span.right, wire_y, wire_y],
                        strict=True,
                    )
                )
                for line in ax.lines
            )

    def test_viewport_contains_every_final_node_rectangle(self):
        """The final viewport encloses every finalized node rectangle."""
        _, layout = _compute_layout(inline_conditional_geometry, inline=True)
        viewport = layout.viewport

        for rect in layout.node_rects.values():
            assert viewport.left <= rect.left
            assert viewport.right >= rect.right
            assert viewport.bottom <= rect.bottom
            assert viewport.top >= rect.top


class TestGeometryPrimitiveValidation:
    """Geometry value objects reject invalid extents and measurements."""

    def test_reversed_span_and_rect_edges_are_rejected(self):
        """Immutable geometry cannot represent reversed coordinate ranges."""
        with pytest.raises(ValueError, match="right edge"):
            HorizontalSpan(1.0, 0.0)
        with pytest.raises(ValueError, match="top edge"):
            Rect(0.0, 1.0, 1.0, 0.0)

    def test_negative_padding_is_rejected(self):
        """Span and rectangle expansion require non-negative padding."""
        with pytest.raises(ValueError, match="non-negative"):
            HorizontalSpan(0.0, 1.0).expanded(-0.1)
        with pytest.raises(ValueError, match="non-negative"):
            Rect(0.0, 0.0, 1.0, 1.0).expanded(-0.1)

    def test_invalid_text_metric_arguments_are_rejected(self):
        """Text measurement validates font size and fallback width."""
        with pytest.raises(ValueError, match="Font size"):
            measure_text("label", font_size=0)
        with pytest.raises(ValueError, match="Fallback"):
            measure_text("label", font_size=13, fallback_char_width=-0.1)


class TestGlobalPhaseGeometry:
    """Global phase annotations participate in authoritative geometry."""

    @staticmethod
    def _visual_circuit() -> VisualCircuit:
        """Build a one-gate circuit carrying a symbolic global phase.

        Returns:
            VisualCircuit: Circuit with a nonzero phase annotation.
        """
        gate = VGate(
            node_key=("global-phase-gate",),
            label="H",
            qubit_indices=[0],
            estimated_width=DEFAULT_STYLE.gate_width,
            kind=VGateKind.GATE,
            gate_type=GateOperationType.H,
        )
        return VisualCircuit(
            children=[gate],
            qubit_map={"q0": 0},
            qubit_names={0: "q0"},
            num_qubits=1,
            global_phase="phi / 2",
        )

    def test_global_phase_layout_is_outside_content_and_inside_viewport(self) -> None:
        """The phase label owns bounds above all circuit geometry."""
        visual_circuit = self._visual_circuit()
        layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(visual_circuit)

        assert layout.global_phase_layout is not None
        phase = layout.global_phase_layout
        assert phase.label == "global phase: phi / 2"
        assert phase.rect.bottom > max(rect.top for rect in layout.node_rects.values())
        assert layout.viewport.left <= phase.rect.left
        assert layout.viewport.right >= phase.rect.right
        assert layout.viewport.top >= phase.rect.top

    def test_renderer_draws_global_phase_with_layout_owned_bounds(self) -> None:
        """The rendered phase text stays inside its authoritative rectangle."""
        visual_circuit = self._visual_circuit()
        layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(visual_circuit)
        figure = MatplotlibRenderer(DEFAULT_STYLE).render(visual_circuit, layout)

        assert layout.global_phase_layout is not None
        phase = layout.global_phase_layout
        rendered_bounds = _rendered_text_bounds(figure, phase.label)
        assert phase.rect.left - _GEOMETRY_TOLERANCE <= rendered_bounds.left
        assert phase.rect.right + _GEOMETRY_TOLERANCE >= rendered_bounds.right
        assert phase.rect.bottom - _GEOMETRY_TOLERANCE <= rendered_bounds.bottom
        assert phase.rect.top + _GEOMETRY_TOLERANCE >= rendered_bounds.top

    def test_zero_qubit_global_phase_is_rendered_with_empty_circuit_label(self) -> None:
        """A phase-only circuit keeps both its phase and empty-state label."""
        visual_circuit = VisualCircuit(
            children=[],
            qubit_map={},
            qubit_names={},
            num_qubits=0,
            global_phase="pi / 4",
        )
        layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(visual_circuit)
        figure = MatplotlibRenderer(DEFAULT_STYLE).render(visual_circuit, layout)

        assert layout.global_phase_layout is not None
        phase = layout.global_phase_layout
        assert phase.label == "global phase: pi / 4"
        assert layout.viewport.left <= phase.rect.left
        assert layout.viewport.right >= phase.rect.right
        assert layout.viewport.top >= phase.rect.top
        assert {text.get_text() for text in figure.axes[0].texts} == {
            "Empty circuit",
            phase.label,
        }
        assert figure.axes[0].get_xlim() == (
            layout.viewport.left,
            layout.viewport.right,
        )


class TestOutOfRangeIndexGeometry:
    """Layout and renderer agree on visibility-filtered gate classification.

    The analyzer has historically emitted phantom wire indices outside the
    displayed range (see ``TestSliceSubKernelArgumentDraw`` in
    ``test_drawer_slice.py``), which is why the renderer filters indices
    through ``_visible_qubits`` before dispatching. Layout must classify and
    split gates on the same filtered basis: classifying on the unfiltered
    indices skips the box rect the renderer then demands via
    ``_require_box_rect``, turning a degraded drawing into a crash.
    """

    def _render(self, gate: VGate, num_qubits: int) -> Figure:
        """Lay out and render a single-gate circuit.

        Args:
            gate (VGate): Gate node under test.
            num_qubits (int): Number of displayed wires.

        Returns:
            Figure: Rendered matplotlib figure.
        """
        vc = VisualCircuit(
            children=[gate],
            qubit_map={f"q{i}": i for i in range(num_qubits)},
            qubit_names={i: f"q{i}" for i in range(num_qubits)},
            num_qubits=num_qubits,
        )
        layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(vc)
        return MatplotlibRenderer(DEFAULT_STYLE).render(vc, layout)

    def test_native_multi_gate_with_out_of_range_index_renders(self):
        """A CX whose second index is out of range degrades to a box."""
        gate = VGate(
            node_key=("g1",),
            label="$X$",
            qubit_indices=[0, 5],
            estimated_width=DEFAULT_STYLE.gate_width,
            kind=VGateKind.GATE,
            gate_type=GateOperationType.CX,
        )
        fig = self._render(gate, num_qubits=2)
        assert isinstance(fig, Figure)

    def test_controlled_u_with_out_of_range_control_renders(self):
        """A controlled-U whose control index is out of range keeps its target box."""
        gate = VGate(
            node_key=("g2",),
            label="$U$",
            qubit_indices=[5, 0],
            control_count=1,
            estimated_width=DEFAULT_STYLE.gate_width,
            kind=VGateKind.CONTROLLED_U_BOX,
        )
        fig = self._render(gate, num_qubits=1)
        assert isinstance(fig, Figure)

    def test_out_of_range_control_keeps_target_box_on_target_wire(self):
        """The stored target box anchors to the real target wire, not a control."""
        gate = VGate(
            node_key=("g3",),
            label="$U$",
            qubit_indices=[5, 0],
            control_count=1,
            estimated_width=DEFAULT_STYLE.gate_width,
            kind=VGateKind.CONTROLLED_U_BOX,
        )
        vc = VisualCircuit(
            children=[gate],
            qubit_map={"q0": 0},
            qubit_names={0: "q0"},
            num_qubits=1,
        )
        layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(vc)
        # The split must happen on the unfiltered indices (controls first),
        # so the surviving target wire 0 still owns a box rect.
        assert ("g3",) in layout.gate_box_rects
        box = layout.gate_box_rects[("g3",)]
        assert box.bottom <= layout.qubit_y[0] <= box.top


def test_sparse_collapsed_call_marks_only_exact_operand_wires() -> None:
    """A spanning call box exposes ports only on its sparse operands."""
    gate = VGate(
        node_key=("sparse-call",),
        label="PAIR",
        qubit_indices=[0, 2],
        estimated_width=DEFAULT_STYLE.gate_width,
        kind=VGateKind.COMPOSITE_BOX,
    )
    visual = VisualCircuit(
        children=[gate],
        qubit_map={f"q{index}": index for index in range(3)},
        qubit_names={index: f"q{index}" for index in range(3)},
        num_qubits=3,
    )
    layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(visual)
    figure = MatplotlibRenderer(DEFAULT_STYLE).render(visual, layout)
    axis = figure._qm_ax  # type: ignore[attr-defined]
    ports = [patch for patch in axis.patches if isinstance(patch, mpatches.Wedge)]

    assert len(ports) == 4
    port_y = [port.center[1] for port in ports]
    assert port_y.count(layout.qubit_y[0]) == 2
    assert port_y.count(layout.qubit_y[2]) == 2
    assert layout.qubit_y[1] not in port_y
    box = layout.gate_box_rects[gate.node_key]
    assert all(
        box.left <= port.center[0] <= box.right
        and box.bottom <= port.center[1] <= box.top
        for port in ports
    )
