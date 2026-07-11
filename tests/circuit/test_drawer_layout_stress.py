"""Stress-test mixed nested circuit drawing geometry."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest
from matplotlib.figure import Figure

import qamomile.circuit as qmc
from qamomile.circuit.visualization.analyzer import CircuitAnalyzer
from qamomile.circuit.visualization.drawer import _prepare_graph_for_visualization
from qamomile.circuit.visualization.layout import CircuitLayoutEngine
from qamomile.circuit.visualization.renderer import MatplotlibRenderer
from qamomile.circuit.visualization.style import DEFAULT_STYLE
from qamomile.circuit.visualization.types import LayoutResult, LineSegment, Rect
from qamomile.circuit.visualization.visual_ir import (
    GateOperationType,
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

_GEOMETRY_TOLERANCE = 1e-7


@qmc.qkernel
def _stress_mixed_leaf(target: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
    """Apply nested FOR, IF, and WHILE operations to one target.

    Args:
        target (qmc.Qubit): Target qubit updated by every control-flow layer.
        angle (qmc.Float): Rotation angle used by the finite FOR body.

    Returns:
        qmc.Qubit: Target qubit after the nested control-flow operations.
    """
    condition = qmc.qubit("condition")
    bit = qmc.measure(condition)
    if bit:
        for _ in qmc.range(2):
            target = qmc.rx(target, angle)
        loop_condition = qmc.qubit("loop_condition")
        loop_bit = qmc.measure(loop_condition)
        while loop_bit:
            fresh = qmc.qubit("fresh")
            target = qmc.h(target)
            loop_bit = qmc.measure(fresh)
    else:
        target = qmc.x(target)
    return target


@qmc.qkernel
def _stress_mixed_vector_helper(
    q: qmc.Vector[qmc.Qubit],
    coeffs: qmc.Dict[qmc.UInt, qmc.Float],
) -> qmc.Vector[qmc.Qubit]:
    """Combine FOR_ITEMS and nested inline calls over a register.

    Args:
        q (qmc.Vector[qmc.Qubit]): Register updated by the mixed helper.
        coeffs (qmc.Dict[qmc.UInt, qmc.Float]): Per-wire RZ angles iterated by
            the FOR_ITEMS operation.

    Returns:
        qmc.Vector[qmc.Qubit]: Register after dictionary and finite-loop work.
    """
    for index, coefficient in coeffs.items():
        q[index] = qmc.rz(q[index], coefficient)
    for index in qmc.range(2):
        q[index + 1] = _stress_mixed_leaf(q[index + 1], angle=0.25)
    return q


@qmc.qkernel
def _stress_mixed_root(
    coeffs: qmc.Dict[qmc.UInt, qmc.Float],
) -> qmc.Vector[qmc.Qubit]:
    """Place a deeply mixed inline region between unrelated gates.

    Args:
        coeffs (qmc.Dict[qmc.UInt, qmc.Float]): Bound values materialized by
            the nested FOR_ITEMS operation.

    Returns:
        qmc.Vector[qmc.Qubit]: Five-qubit register after the mixed helper.
    """
    q = qmc.qubit_array(5, "q")
    q[0] = qmc.h(q[0])
    q = _stress_mixed_vector_helper(q, coeffs)
    q[0] = qmc.z(q[0])
    q[4] = qmc.x(q[4])
    return q


@qmc.qkernel
def _stress_swap_pair(
    left: qmc.Qubit,
    right: qmc.Qubit,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Swap two target qubits inside a controlled inline wrapper.

    Args:
        left (qmc.Qubit): First SWAP target.
        right (qmc.Qubit): Second SWAP target.

    Returns:
        tuple[qmc.Qubit, qmc.Qubit]: Swapped target qubits.
    """
    return qmc.swap(left, right)


@qmc.qkernel
def _stress_powered_controlled_root() -> qmc.Vector[qmc.Qubit]:
    """Place a sparse powered controlled inline block between gates.

    Returns:
        qmc.Vector[qmc.Qubit]: Five-qubit register after the controlled SWAP.
    """
    q = qmc.qubit_array(5, "q")
    q[1] = qmc.h(q[1])
    controlled_swap = qmc.control(_stress_swap_pair)
    q[0], q[2], q[4] = controlled_swap(q[0], q[2], q[4], power=3)
    q[1] = qmc.z(q[1])
    q[3] = qmc.x(q[3])
    return q


def _connector_obstacle_circuit() -> tuple[
    VisualCircuit,
    VGate,
    tuple[VGate, VGate],
    VUnfoldedSequence,
]:
    """Build a condition connector with gates obstructing its direct route.

    The first obstacle follows the condition measurement on the same source
    wire. The second occupies an intermediate wire and overlaps the connector's
    horizontal range, forcing the layout to choose a clear orthogonal lane.

    Returns:
        tuple[VisualCircuit, VGate, tuple[VGate, VGate], VUnfoldedSequence]:
            Visual circuit, source measurement, obstacle gates, and target IF.
    """
    measurement = VGate(
        node_key=("connector", "measure"),
        label="M",
        qubit_indices=[0],
        estimated_width=DEFAULT_STYLE.gate_width,
        kind=VGateKind.MEASURE,
        terminates_wire=False,
    )
    intermediate_obstacle = VGate(
        node_key=("connector", "middle-obstacle"),
        label="$R_x$",
        qubit_indices=[1],
        estimated_width=2.0,
        kind=VGateKind.GATE,
        gate_type=GateOperationType.RX,
    )
    source_wire_obstacle = VGate(
        node_key=("connector", "source-obstacle"),
        label="$H$",
        qubit_indices=[0],
        estimated_width=DEFAULT_STYLE.gate_width,
        kind=VGateKind.GATE,
        gate_type=GateOperationType.H,
    )
    branch_gate = VGate(
        node_key=("connector", "if", "child"),
        label="$X$",
        qubit_indices=[2],
        estimated_width=DEFAULT_STYLE.gate_width,
        kind=VGateKind.GATE,
        gate_type=GateOperationType.X,
    )
    conditional = VUnfoldedSequence(
        node_key=("connector", "if"),
        iterations=[[branch_gate]],
        affected_qubits=[2],
        kind=VUnfoldedKind.IF,
        condition_label="if q0_measured:",
        condition_label_width=2.5,
        branch_label_widths=[2.5],
        condition_measure_node_key=measurement.node_key,
        condition_measure_qubit_indices=[0],
    )
    circuit = VisualCircuit(
        children=[
            measurement,
            intermediate_obstacle,
            source_wire_obstacle,
            conditional,
        ],
        qubit_map={f"q{index}": index for index in range(3)},
        qubit_names={index: f"q{index}" for index in range(3)},
        num_qubits=3,
    )
    return (
        circuit,
        measurement,
        (intermediate_obstacle, source_wire_obstacle),
        conditional,
    )


def _walk(nodes: Iterable[VisualNode]) -> Iterator[VisualNode]:
    """Yield every visual node in pre-order.

    Args:
        nodes (Iterable[VisualNode]): Root nodes to traverse.

    Yields:
        VisualNode: Each reachable visual node in pre-order.
    """
    for node in nodes:
        yield node
        if isinstance(node, VInlineBlock):
            yield from _walk(node.children)
        elif isinstance(node, VUnfoldedSequence):
            for iteration in node.iterations:
                yield from _walk(iteration)


def _build_visual_circuit(
    kernel: Any,
    *,
    inline: bool,
    fold_loops: bool,
    **bindings: Any,
) -> VisualCircuit:
    """Analyze one qkernel into the Visual IR used by layout.

    Args:
        kernel (Any): QKernel-like object to trace.
        inline (bool): Whether to expand call and controlled blocks.
        fold_loops (bool): Whether to collapse FOR and FOR_ITEMS operations.
        **bindings (Any): Concrete visualization-time kernel arguments.

    Returns:
        VisualCircuit: Visual tree ready for layout and rendering.
    """
    block = _prepare_graph_for_visualization(
        kernel._build_graph_for_visualization(**bindings)
    )
    analyzer = CircuitAnalyzer(
        block,
        DEFAULT_STYLE,
        inline=inline,
        fold_loops=fold_loops,
        fold_ifs=False,
        fold_whiles=False,
        expand_composite=False,
        inline_depth=None,
    )
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(block)
    return analyzer.build_visual_ir(block, qubit_map, qubit_names, num_qubits)


def _rect_contains(outer: Rect, inner: Rect) -> bool:
    """Return whether one rectangle contains another within tolerance.

    Args:
        outer (Rect): Candidate containing rectangle.
        inner (Rect): Rectangle expected to remain inside ``outer``.

    Returns:
        bool: True when every inner edge is inside the corresponding outer
            edge within the geometry tolerance.
    """
    return (
        outer.left <= inner.left + _GEOMETRY_TOLERANCE
        and outer.right >= inner.right - _GEOMETRY_TOLERANCE
        and outer.bottom <= inner.bottom + _GEOMETRY_TOLERANCE
        and outer.top >= inner.top - _GEOMETRY_TOLERANCE
    )


def _rects_overlap(first: Rect, second: Rect) -> bool:
    """Return whether two rectangles have positive-area intersection.

    Args:
        first (Rect): First rectangle.
        second (Rect): Second rectangle.

    Returns:
        bool: True when both horizontal and vertical interiors intersect.
    """
    return (
        first.left < second.right - _GEOMETRY_TOLERANCE
        and first.right > second.left + _GEOMETRY_TOLERANCE
        and first.bottom < second.top - _GEOMETRY_TOLERANCE
        and first.top > second.bottom + _GEOMETRY_TOLERANCE
    )


def _segment_crosses_rect_interior(segment: LineSegment, rect: Rect) -> bool:
    """Return whether an orthogonal segment crosses a rectangle's interior.

    Args:
        segment (LineSegment): Horizontal or vertical connector segment.
        rect (Rect): Obstacle rectangle whose boundary may be touched but whose
            interior must remain clear.

    Returns:
        bool: True when the segment intersects the obstacle interior.

    Raises:
        ValueError: If the connector segment is not orthogonal.
    """
    if math.isclose(
        segment.start_x,
        segment.end_x,
        abs_tol=_GEOMETRY_TOLERANCE,
    ):
        low, high = sorted((segment.start_y, segment.end_y))
        return (
            rect.left + _GEOMETRY_TOLERANCE
            < segment.start_x
            < rect.right - _GEOMETRY_TOLERANCE
            and low < rect.top - _GEOMETRY_TOLERANCE
            and high > rect.bottom + _GEOMETRY_TOLERANCE
        )
    if math.isclose(
        segment.start_y,
        segment.end_y,
        abs_tol=_GEOMETRY_TOLERANCE,
    ):
        low, high = sorted((segment.start_x, segment.end_x))
        return (
            rect.bottom + _GEOMETRY_TOLERANCE
            < segment.start_y
            < rect.top - _GEOMETRY_TOLERANCE
            and low < rect.right - _GEOMETRY_TOLERANCE
            and high > rect.left + _GEOMETRY_TOLERANCE
        )
    raise ValueError("Connector segments must be horizontal or vertical")


def _drawable_rect(node: VisualNode, layout: LayoutResult) -> Rect | None:
    """Return concrete node ink without connector bounding-box overapproximation.

    Args:
        node (VisualNode): Visual node whose drawn body is required.
        layout (LayoutResult): Final typed geometry maps.

    Returns:
        Rect | None: Concrete body bounds, or None for transparent sequences
            and nodes without visible geometry.
    """
    if isinstance(node, VInlineBlock):
        return layout.inline_block_layouts[node.node_key].outer_rect
    if isinstance(node, VFoldedBlock):
        rect = layout.folded_block_layouts[node.node_key].rect
        if not node.affected_qubits_precise:
            return rect
        radius = DEFAULT_STYLE.folded_marker_radius
        return Rect(rect.left - radius, rect.bottom, rect.right + radius, rect.top)
    if isinstance(node, VUnfoldedSequence):
        if node.kind in (VUnfoldedKind.IF, VUnfoldedKind.WHILE):
            return layout.control_flow_layouts[node.node_key].outer_rect
        return None
    return layout.node_rects.get(node.node_key)


def _assert_parent_containment(
    nodes: list[VisualNode],
    layout: LayoutResult,
) -> None:
    """Assert that every inline or branch region contains its direct children.

    Args:
        nodes (list[VisualNode]): Visual nodes at the current tree level.
        layout (LayoutResult): Authoritative finalized geometry.

    Raises:
        AssertionError: If a direct child escapes its owning region.
    """
    for node in nodes:
        if isinstance(node, VInlineBlock):
            inner = layout.inline_block_layouts[node.node_key].inner_rect
            for child in node.children:
                child_rect = layout.node_rects.get(child.node_key)
                if child_rect is not None:
                    assert _rect_contains(inner, child_rect), (
                        node.node_key,
                        child.node_key,
                        inner,
                        child_rect,
                    )
            _assert_parent_containment(node.children, layout)
            continue

        if not isinstance(node, VUnfoldedSequence):
            continue
        if node.kind in (VUnfoldedKind.IF, VUnfoldedKind.WHILE):
            boxes = layout.control_flow_layouts[node.node_key].boxes
            assert len(boxes) == len(node.iterations)
            for box, children in zip(boxes, node.iterations, strict=True):
                for child in children:
                    child_rect = layout.node_rects.get(child.node_key)
                    if child_rect is not None:
                        assert _rect_contains(box.rect, child_rect), (
                            node.node_key,
                            child.node_key,
                            box.rect,
                            child_rect,
                        )
        for iteration in node.iterations:
            _assert_parent_containment(iteration, layout)


def _drawable_nodes_with_ancestors(
    nodes: list[VisualNode],
) -> list[tuple[VisualNode, frozenset[tuple]]]:
    """Collect concrete drawing nodes with their ancestor keys.

    Transparent unrolled FOR and FOR_ITEMS sequences are omitted because their
    rectangles are only unions of descendant ink and are not rendered.

    Args:
        nodes (list[VisualNode]): Root visual nodes to collect.

    Returns:
        list[tuple[VisualNode, frozenset[tuple]]]: Drawable nodes paired with
            the node keys of every visual ancestor.
    """
    entries: list[tuple[VisualNode, frozenset[tuple]]] = []

    def collect(current: list[VisualNode], ancestors: frozenset[tuple]) -> None:
        """Collect one subtree into the outer result.

        Args:
            current (list[VisualNode]): Nodes at the current tree level.
            ancestors (frozenset[tuple]): Ancestor keys above ``current``.
        """
        for node in current:
            next_ancestors = ancestors | {node.node_key}
            is_transparent_loop = isinstance(
                node, VUnfoldedSequence
            ) and node.kind not in (VUnfoldedKind.IF, VUnfoldedKind.WHILE)
            if not is_transparent_loop:
                entries.append((node, ancestors))
            if isinstance(node, VInlineBlock):
                collect(node.children, next_ancestors)
            elif isinstance(node, VUnfoldedSequence):
                for iteration in node.iterations:
                    collect(iteration, next_ancestors)

    collect(nodes, frozenset())
    return entries


def _assert_non_ancestor_rectangles_do_not_overlap(
    vc: VisualCircuit,
    layout: LayoutResult,
) -> None:
    """Assert that unrelated rendered node rectangles remain disjoint.

    Args:
        vc (VisualCircuit): Visual tree whose ancestry determines exclusions.
        layout (LayoutResult): Authoritative finalized geometry.

    Raises:
        AssertionError: If two non-ancestor drawing nodes overlap.
    """
    entries = _drawable_nodes_with_ancestors(vc.children)
    for index, (first, first_ancestors) in enumerate(entries):
        first_rect = _drawable_rect(first, layout)
        if first_rect is None:
            continue
        for second, second_ancestors in entries[index + 1 :]:
            if first.node_key in second_ancestors or second.node_key in first_ancestors:
                continue
            second_rect = _drawable_rect(second, layout)
            if second_rect is None:
                continue
            assert not _rects_overlap(first_rect, second_rect), (
                first.node_key,
                second.node_key,
                first_rect,
                second_rect,
            )


def _assert_viewport_contains_rendered_artists(
    fig: Figure,
    layout: LayoutResult,
) -> None:
    """Assert that every rendered patch, text, and line stays in the viewport.

    Args:
        fig (Figure): Fully rendered Matplotlib circuit figure.
        layout (LayoutResult): Layout containing the viewport under test.

    Raises:
        AssertionError: If an artist or line endpoint escapes the viewport.
    """
    fig.canvas.draw()
    ax = fig._qm_ax  # type: ignore[attr-defined]
    renderer = fig.canvas.get_renderer()

    for artist in [*ax.patches, *ax.texts]:
        bounds = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
        artist_rect = Rect(bounds.x0, bounds.y0, bounds.x1, bounds.y1)
        assert _rect_contains(layout.viewport, artist_rect), (
            type(artist).__name__,
            artist_rect,
            layout.viewport,
        )

    for line in ax.lines:
        x_values = [float(value) for value in line.get_xdata()]
        y_values = [float(value) for value in line.get_ydata()]
        if not x_values or not y_values:
            continue
        line_rect = Rect(
            min(x_values),
            min(y_values),
            max(x_values),
            max(y_values),
        )
        assert _rect_contains(layout.viewport, line_rect), (
            line_rect,
            layout.viewport,
        )


def _assert_layout_render_invariants(
    vc: VisualCircuit,
) -> tuple[LayoutResult, Figure]:
    """Compute, render, and verify the shared mixed-layout invariants.

    Args:
        vc (VisualCircuit): Visual tree to lay out and render.

    Returns:
        tuple[LayoutResult, Figure]: Verified geometry and rendered figure.

    Raises:
        AssertionError: If containment, overlap, viewport, or immutability
            invariants fail.
    """
    layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(vc)
    _assert_parent_containment(vc.children, layout)
    _assert_non_ancestor_rectangles_do_not_overlap(vc, layout)
    assert all(
        _rect_contains(layout.viewport, rect) for rect in layout.node_rects.values()
    )

    before_render = deepcopy(layout)
    fig = MatplotlibRenderer(DEFAULT_STYLE).render(vc, layout)
    assert layout == before_render
    _assert_viewport_contains_rendered_artists(fig, layout)
    return layout, fig


class TestMixedNestedLayoutStress:
    """Exercise representative mixed nesting found by seeded stress analysis."""

    def test_unfolded_inline_for_items_if_while_and_fresh_wires(self):
        """Deep mixed expansion preserves all geometry invariants."""
        vc = _build_visual_circuit(
            _stress_mixed_root,
            inline=True,
            fold_loops=False,
            coeffs={1: 0.2, 3: -0.4},
        )
        nodes = list(_walk(vc.children))
        sequence_kinds = {
            node.kind for node in nodes if isinstance(node, VUnfoldedSequence)
        }

        assert vc.num_qubits > 5
        assert sum(isinstance(node, VInlineBlock) for node in nodes) >= 3
        assert {
            VUnfoldedKind.FOR,
            VUnfoldedKind.FOR_ITEMS,
            VUnfoldedKind.IF,
            VUnfoldedKind.WHILE,
        } <= sequence_kinds
        _assert_layout_render_invariants(vc)

    def test_folded_for_and_for_items_inside_inline_region(self):
        """Folded loop summaries remain contained beside unrelated gates."""
        vc = _build_visual_circuit(
            _stress_mixed_root,
            inline=True,
            fold_loops=True,
            coeffs={1: 0.2, 3: -0.4},
        )
        nodes = list(_walk(vc.children))
        folded_kinds = {node.kind for node in nodes if isinstance(node, VFoldedBlock)}

        assert {VFoldedKind.FOR, VFoldedKind.FOR_ITEMS} <= folded_kinds
        _assert_layout_render_invariants(vc)

    @pytest.mark.parametrize("inline", [False, True])
    def test_sparse_powered_controlled_block_between_neighbor_gates(self, inline):
        """Powered controlled geometry clears crossed wires in both views."""
        vc = _build_visual_circuit(
            _stress_powered_controlled_root,
            inline=inline,
            fold_loops=False,
        )
        nodes = list(_walk(vc.children))

        assert any(getattr(node, "power", 1) == 3 for node in nodes)
        layout, _ = _assert_layout_render_invariants(vc)
        if inline:
            powered_inline = next(
                node
                for node in nodes
                if isinstance(node, VInlineBlock) and node.power == 3
            )
            placement = layout.inline_block_layouts[powered_inline.node_key]
            assert placement.outer_rect.width > placement.inner_rect.width
            assert placement.outer_rect.height > placement.inner_rect.height

    def test_condition_connector_avoids_same_wire_and_intermediate_obstacles(self):
        """Orthogonal connector lanes never cross intervening gate interiors."""
        vc, measurement, obstacles, conditional = _connector_obstacle_circuit()
        layout, _ = _assert_layout_render_invariants(vc)
        connector = layout.control_flow_layouts[conditional.node_key].connector_segments

        assert len(connector) >= 3
        assert connector[0].start_x == layout.node_spans[measurement.node_key].right
        for obstacle in obstacles:
            obstacle_rect = layout.node_rects[obstacle.node_key]
            assert all(
                not _segment_crosses_rect_interior(segment, obstacle_rect)
                for segment in connector
            ), (obstacle.node_key, obstacle_rect, connector)

    @pytest.mark.parametrize("missing_kind", ["inline", "control_flow"])
    def test_renderer_fails_fast_when_required_geometry_is_missing(
        self,
        missing_kind,
    ):
        """Renderer rejects incomplete container geometry instead of skipping it."""
        vc = _build_visual_circuit(
            _stress_mixed_root,
            inline=True,
            fold_loops=False,
            coeffs={1: 0.2, 3: -0.4},
        )
        layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(vc)

        if missing_kind == "inline":
            node = next(
                node for node in _walk(vc.children) if isinstance(node, VInlineBlock)
            )
            incomplete = replace(
                layout,
                inline_block_layouts={
                    key: value
                    for key, value in layout.inline_block_layouts.items()
                    if key != node.node_key
                },
            )
            expected_message = "Missing inline-block geometry"
        else:
            node = next(
                node
                for node in _walk(vc.children)
                if isinstance(node, VUnfoldedSequence)
                and node.kind in (VUnfoldedKind.IF, VUnfoldedKind.WHILE)
            )
            incomplete = replace(
                layout,
                control_flow_layouts={
                    key: value
                    for key, value in layout.control_flow_layouts.items()
                    if key != node.node_key
                },
            )
            expected_message = "Missing control-flow geometry"

        with pytest.raises(ValueError, match=expected_message):
            MatplotlibRenderer(DEFAULT_STYLE).render(vc, incomplete)
