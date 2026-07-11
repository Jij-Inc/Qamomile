"""Compute complete circuit drawing geometry from the Visual IR.

The layout engine owns every coordinate used by a renderer. Horizontal
placement records one authoritative full span per visual node. Once wire y
coordinates are known, a bottom-up finalization pass turns those spans into
rectangles, connector segments, wire bounds, and a viewport.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geometry import compute_border_padding
from .style import CircuitStyle
from .text_metrics import measure_text, measure_text_width
from .types import (
    ControlFlowBoxLayout,
    ControlFlowLayout,
    FoldedBlockLayout,
    HorizontalSpan,
    InlineBlockLayout,
    LayoutResult,
    LayoutState,
    LineSegment,
    PoweredGateLayout,
    Rect,
)
from .visual_ir import (
    GateOperationType,
    VFoldedBlock,
    VGate,
    VGateKind,
    VInlineBlock,
    VisualCircuit,
    VisualNode,
    VSkip,
    VUnfoldedKind,
    VUnfoldedSequence,
)

_MAX_VERTICAL_LAYOUT_PASSES = 16
_GEOMETRY_TOLERANCE = 1e-9


@dataclass
class _FinalizationState:
    """Collect typed output maps during bottom-up geometry finalization.

    Args:
        node_rects (dict[tuple, Rect]): Final rectangle for each drawing node.
        gate_box_rects (dict[tuple, Rect]): Exact bounds of text-bearing gate
            boxes.
        control_flow_layouts (dict[tuple, ControlFlowLayout]): Final unfolded
            IF and WHILE geometry.
        folded_block_layouts (dict[tuple, FoldedBlockLayout]): Final folded
            control-flow geometry.
        inline_block_layouts (dict[tuple, InlineBlockLayout]): Final expanded
            block geometry.
        powered_gate_layouts (dict[tuple, PoweredGateLayout]): Final powered
            controlled-gate geometry.
    """

    node_rects: dict[tuple, Rect] = field(default_factory=dict)
    gate_box_rects: dict[tuple, Rect] = field(default_factory=dict)
    control_flow_layouts: dict[tuple, ControlFlowLayout] = field(default_factory=dict)
    folded_block_layouts: dict[tuple, FoldedBlockLayout] = field(default_factory=dict)
    inline_block_layouts: dict[tuple, InlineBlockLayout] = field(default_factory=dict)
    powered_gate_layouts: dict[tuple, PoweredGateLayout] = field(default_factory=dict)


class CircuitLayoutEngine:
    """Compute renderer-independent coordinates for a visual circuit.

    Args:
        style (CircuitStyle): Style constants that determine geometry.
    """

    def __init__(self, style: CircuitStyle) -> None:
        """Initialize a layout engine.

        Args:
            style (CircuitStyle): Style constants that determine geometry.
        """
        self.style = style

    @staticmethod
    def _wire_range(qubits: list[int]) -> list[int]:
        """Return every wire crossed by a node's affected qubits.

        Args:
            qubits (list[int]): Directly affected qubit indices.

        Returns:
            list[int]: Inclusive contiguous range from the minimum to maximum
                affected index, or an empty list when no qubits are affected.
        """
        if not qubits:
            return []
        return list(range(min(qubits), max(qubits) + 1))

    @staticmethod
    def _union_spans(spans: list[HorizontalSpan]) -> HorizontalSpan | None:
        """Return the union of a possibly empty span list.

        Args:
            spans (list[HorizontalSpan]): Horizontal spans to combine.

        Returns:
            HorizontalSpan | None: Combined span, or None for an empty list.
        """
        if not spans:
            return None
        result = spans[0]
        for span in spans[1:]:
            result = result.union(span)
        return result

    @staticmethod
    def _union_rects(rects: list[Rect]) -> Rect | None:
        """Return the union of a possibly empty rectangle list.

        Args:
            rects (list[Rect]): Rectangles to combine.

        Returns:
            Rect | None: Combined rectangle, or None for an empty list.
        """
        if not rects:
            return None
        result = rects[0]
        for rect in rects[1:]:
            result = result.union(rect)
        return result

    def _spans_for_nodes(
        self, nodes: list[VisualNode], state: LayoutState
    ) -> HorizontalSpan | None:
        """Return the union of already placed direct child spans.

        Container node spans already include all descendants, so direct-child
        union is sufficient and avoids prefix-based key heuristics.

        Args:
            nodes (list[VisualNode]): Positioned direct child nodes.
            state (LayoutState): Current placement state.

        Returns:
            HorizontalSpan | None: Union of child spans, or None when the child
                list contains no space-bearing node.
        """
        spans = [
            state.node_spans[node.node_key]
            for node in nodes
            if node.node_key in state.node_spans
            and state.node_spans[node.node_key].width > 0
        ]
        return self._union_spans(spans)

    def _rects_for_nodes(
        self, nodes: list[VisualNode], node_rects: dict[tuple, Rect]
    ) -> Rect | None:
        """Return the union of finalized direct child rectangles.

        Args:
            nodes (list[VisualNode]): Finalized direct child nodes.
            node_rects (dict[tuple, Rect]): Final rectangles keyed by node key.

        Returns:
            Rect | None: Union of child rectangles, or None when no child draws
                concrete geometry.
        """
        rects = [
            node_rects[node.node_key] for node in nodes if node.node_key in node_rects
        ]
        return self._union_rects(rects)

    def _frontier(self, state: LayoutState, wires: list[int]) -> float:
        """Return the furthest occupied right edge among wires.

        Args:
            state (LayoutState): Current placement state.
            wires (list[int]): Wires whose frontiers should be inspected.

        Returns:
            float: Maximum right edge, or the initial wire position when the
                wire list is empty.
        """
        if not wires:
            return self.style.initial_wire_position
        return max(
            state.qubit_right_edges.get(q, self.style.initial_wire_position)
            for q in wires
        )

    def _synchronize_frontier(
        self, state: LayoutState, wires: list[int], right_edge: float
    ) -> None:
        """Advance a contiguous wire range to at least one right edge.

        Args:
            state (LayoutState): Current placement state.
            wires (list[int]): Wires occupied by the node.
            right_edge (float): Minimum new frontier. A wire already farther
                right is never rewound.
        """
        for qubit in wires:
            current_edge = state.qubit_right_edges.get(
                qubit, self.style.initial_wire_position
            )
            synchronized_edge = max(current_edge, right_edge)
            state.qubit_right_edges[qubit] = synchronized_edge
            state.qubit_columns[qubit] = synchronized_edge + self.style.gate_gap

    def _next_center(
        self, state: LayoutState, wires: list[int], full_width: float
    ) -> float:
        """Compute the earliest legal center for a full-width node.

        Args:
            state (LayoutState): Current placement state.
            wires (list[int]): Contiguous occupied wire range.
            full_width (float): Complete horizontal drawing width.

        Returns:
            float: Earliest center that clears all wire frontiers and preserves
                the historical first-column baseline.
        """
        half_width = full_width / 2
        frontier_center = (
            self._frontier(state, wires) + self.style.gate_gap + half_width
        )
        if wires:
            column_center = max(state.qubit_columns[q] for q in wires)
        else:
            column_center = state.column
        return max(frontier_center, column_center)

    def _record_span(
        self,
        node_key: tuple,
        span: HorizontalSpan,
        state: LayoutState,
        *,
        center: float | None = None,
    ) -> None:
        """Record a node span and update global compatibility bounds.

        Args:
            node_key (tuple): Stable Visual IR node key.
            span (HorizontalSpan): Authoritative full drawing extent.
            state (LayoutState): Current placement state.
            center (float | None): Compatibility center override. Defaults to
                the span midpoint.
        """
        state.node_spans[node_key] = span
        state.positions[node_key] = span.center if center is None else center
        state.actual_width = max(state.actual_width, span.right)
        state.column = max(state.column, span.right)
        if state.first_gate_x is None and span.width > 0:
            state.first_gate_x = span.center
            state.first_gate_half_width = span.width / 2

    def _condition_measure_right_edge(
        self, node_key: tuple | None, state: LayoutState
    ) -> float | None:
        """Return the full right edge of a condition measurement.

        Args:
            node_key (tuple | None): Measurement node key, if one was resolved.
            state (LayoutState): Current placement state.

        Returns:
            float | None: Measurement's authoritative right edge, or None when
                the measurement has not been placed.
        """
        if node_key is None:
            return None
        span = state.node_spans.get(node_key)
        return None if span is None else span.right

    def _place_vgate(self, node: VGate, state: LayoutState) -> None:
        """Place one gate-like leaf using its complete visual width.

        Args:
            node (VGate): Gate, measurement, block box, or expectation node.
            state (LayoutState): Current placement state.
        """
        wires = self._wire_range(node.qubit_indices)
        full_width = max(0.0, node.estimated_width)
        center = self._next_center(state, wires, full_width)
        span = HorizontalSpan(center - full_width / 2, center + full_width / 2)

        self._record_span(node.node_key, span, state, center=center)
        state.gate_widths[node.node_key] = full_width
        if node.box_width is not None:
            state.block_widths[node.node_key] = node.box_width
        self._synchronize_frontier(state, wires, span.right)

        if (
            node.kind in (VGateKind.MEASURE, VGateKind.MEASURE_VECTOR)
            and node.terminates_wire
        ) or node.kind == VGateKind.EXPVAL:
            for qubit in node.qubit_indices:
                state.qubit_end_positions[qubit] = span.right

    def _place_vinline_block(
        self, node: VInlineBlock, state: LayoutState, depth: int
    ) -> None:
        """Place children and the complete border of an expanded block.

        Args:
            node (VInlineBlock): Expanded inline block to position.
            state (LayoutState): Current placement state.
            depth (int): Current visual nesting depth.
        """
        semantic_qubits = self._semantic_qubits(
            node,
            len(state.qubit_right_edges),
        )
        placement_qubits = sorted(
            set(semantic_qubits) | set(node.control_qubit_indices)
        )
        wires = self._wire_range(placement_qubits)
        minimum_outer_width = max(node.final_width, self.style.gate_width)
        provisional_center = self._next_center(state, wires, minimum_outer_width)
        outer_left = provisional_center - minimum_outer_width / 2
        wrapper_margin = self.style.power_wrapper_margin if node.power > 1 else 0.0
        inner_left = outer_left + wrapper_margin
        child_padding = max(
            node.border_padding + self.style.gate_text_padding,
            self.style.box_padding_x,
        )

        content_left = inner_left + child_padding
        self._synchronize_frontier(state, wires, content_left - self.style.gate_gap)
        state.max_depth = max(state.max_depth, depth + 1)
        self._place_visual_nodes(node.children, state, depth + 1)

        child_span = self._spans_for_nodes(node.children, state)
        content_right = content_left if child_span is None else child_span.right
        inner_right = max(
            content_right + child_padding,
            inner_left + node.label_width,
        )
        outer_right = inner_right + wrapper_margin
        outer_right = max(outer_right, outer_left + minimum_outer_width)
        if node.power > 1:
            pow_label_width = (
                measure_text_width(
                    f"pow={node.power}",
                    font_size=self.style.subfont_size,
                    fallback_char_width=self.style.char_width_base,
                )
                + 2 * self.style.label_horizontal_padding
            )
            outer_right = max(outer_right, outer_left + pow_label_width)

        inner_span = HorizontalSpan(inner_left, inner_right)
        outer_span = HorizontalSpan(outer_left, outer_right)
        state.inline_inner_spans[node.node_key] = inner_span
        self._record_span(node.node_key, outer_span, state)
        self._synchronize_frontier(state, wires, outer_span.right)
        state.inlined_op_keys.add(node.node_key)
        state.block_widths[node.node_key] = outer_span.width

        descendant_centers = [
            state.positions[child.node_key]
            for child in node.children
            if child.node_key in state.positions
        ]
        if descendant_centers:
            actual_start = min(descendant_centers)
            actual_end = max(descendant_centers)
        else:
            actual_start = actual_end = inner_span.center
        state.block_ranges.append(
            {
                "name": node.label,
                "start_x": actual_start,
                "end_x": actual_end,
                "qubit_indices": list(node.affected_qubits),
                "control_qubit_indices": list(node.control_qubit_indices),
                "power": node.power,
                "depth": depth,
                "max_gate_width": node.max_gate_width,
            }
        )

    def _place_vfolded_block(self, node: VFoldedBlock, state: LayoutState) -> None:
        """Place a folded control-flow summary as one full-width box.

        Args:
            node (VFoldedBlock): Folded summary node to position.
            state (LayoutState): Current placement state.
        """
        placement_qubits = sorted(
            set(node.affected_qubits) | set(node.condition_measure_qubit_indices)
        )
        wires = self._wire_range(placement_qubits)
        box_width = max(node.folded_width, self.style.gate_width)
        marker_radius = (
            self.style.folded_marker_radius if node.affected_qubits_precise else 0.0
        )
        width = box_width + 2 * marker_radius
        center = self._next_center(state, wires, width)
        condition_right = self._condition_measure_right_edge(
            node.condition_measure_node_key, state
        )
        if condition_right is not None:
            center = max(center, condition_right + self.style.gate_gap + width / 2)
        span = HorizontalSpan(center - width / 2, center + width / 2)
        self._record_span(node.node_key, span, state, center=center)
        self._synchronize_frontier(state, wires, span.right)
        state.block_widths[node.node_key] = box_width
        state.folded_block_extents[node.node_key] = {
            "affected_qubits": list(node.affected_qubits),
            "num_text_lines": 1 + len(node.body_lines),
        }

    def _branch_labels(self, node: VUnfoldedSequence) -> list[str]:
        """Return display labels aligned with control-flow branches.

        Args:
            node (VUnfoldedSequence): Unfolded IF or WHILE node.

        Returns:
            list[str]: One header label per branch.
        """
        if node.kind == VUnfoldedKind.WHILE:
            first = node.condition_label or "while cond:"
        else:
            first = node.condition_label or "if cond:"
        labels = [first]
        for index in range(1, len(node.iterations)):
            labels.append("else:" if index == 1 else "")
        return labels

    def _place_boxed_control_flow(
        self, node: VUnfoldedSequence, state: LayoutState, depth: int
    ) -> None:
        """Place unfolded IF or WHILE branches in authoritative boxes.

        Args:
            node (VUnfoldedSequence): Unfolded IF or WHILE node.
            state (LayoutState): Current placement state.
            depth (int): Current visual nesting depth.
        """
        affected = list(node.affected_qubits)
        if not affected:
            affected = list(node.condition_measure_qubit_indices)
        semantic_qubits = self._semantic_qubits(
            node,
            len(state.qubit_right_edges),
        )
        placement_qubits = sorted(
            set(semantic_qubits or affected) | set(node.condition_measure_qubit_indices)
        )
        wires = self._wire_range(placement_qubits)
        padding = max(
            self.style.box_padding_x,
            compute_border_padding(self.style, depth=0),
        )

        first_label_width = (
            node.branch_label_widths[0]
            if node.branch_label_widths
            else node.condition_label_width
        )
        first_min_width = max(first_label_width, self.style.gate_width)
        left_from_frontier = self._frontier(state, wires) + self.style.gate_gap
        if wires:
            left_from_column = (
                max(state.qubit_columns[q] for q in wires) - first_min_width / 2
            )
        else:
            left_from_column = state.column
        cursor = max(left_from_frontier, left_from_column)
        condition_right = self._condition_measure_right_edge(
            node.condition_measure_node_key, state
        )
        if condition_right is not None:
            cursor = max(
                cursor,
                condition_right + max(self.style.gate_gap, padding),
            )

        branch_spans: list[HorizontalSpan] = []
        for index, children in enumerate(node.iterations):
            label_width = (
                node.branch_label_widths[index]
                if index < len(node.branch_label_widths)
                else 0.0
            )
            minimum_width = max(label_width, self.style.gate_width)
            branch_left = cursor
            content_left = branch_left + padding
            self._synchronize_frontier(state, wires, content_left - self.style.gate_gap)
            self._place_visual_nodes(children, state, depth + 1)
            child_span = self._spans_for_nodes(children, state)
            branch_right = branch_left + minimum_width
            if child_span is not None:
                branch_right = max(branch_right, child_span.right + padding)
            branch_span = HorizontalSpan(branch_left, branch_right)
            branch_spans.append(branch_span)
            self._synchronize_frontier(state, wires, branch_right)
            cursor = branch_right

        if not branch_spans:
            branch_spans.append(HorizontalSpan(cursor, cursor + first_min_width))
        node_span = self._union_spans(branch_spans)
        assert node_span is not None
        state.control_flow_branch_spans[node.node_key] = tuple(branch_spans)
        self._record_span(
            node.node_key,
            node_span,
            state,
            center=branch_spans[0].center,
        )
        state.block_widths[node.node_key] = branch_spans[0].width
        self._synchronize_frontier(state, wires, node_span.right)

    def _place_unboxed_sequence(
        self, node: VUnfoldedSequence, state: LayoutState, depth: int
    ) -> None:
        """Place unrolled FOR or FOR_ITEMS iterations without a box.

        Args:
            node (VUnfoldedSequence): Unboxed unrolled sequence.
            state (LayoutState): Current placement state.
            depth (int): Current visual nesting depth.
        """
        iteration_spans: list[HorizontalSpan] = []
        for children in node.iterations:
            self._place_visual_nodes(children, state, depth + 1)
            child_span = self._spans_for_nodes(children, state)
            if child_span is not None:
                iteration_spans.append(child_span)
        span = self._union_spans(iteration_spans)
        if span is None:
            wires = self._wire_range(node.affected_qubits)
            anchor = self._frontier(state, wires)
            span = HorizontalSpan(anchor, anchor)
        self._record_span(node.node_key, span, state)

    def _place_vunfolded_sequence(
        self, node: VUnfoldedSequence, state: LayoutState, depth: int
    ) -> None:
        """Place an unfolded control-flow sequence.

        Args:
            node (VUnfoldedSequence): Unfolded loop or branch sequence.
            state (LayoutState): Current placement state.
            depth (int): Current visual nesting depth.
        """
        if node.kind in (VUnfoldedKind.IF, VUnfoldedKind.WHILE):
            self._place_boxed_control_flow(node, state, depth)
        else:
            self._place_unboxed_sequence(node, state, depth)

    def _place_visual_nodes(
        self,
        nodes: list[VisualNode],
        state: LayoutState,
        depth: int = 0,
    ) -> None:
        """Place Visual IR nodes recursively by concrete node type.

        Args:
            nodes (list[VisualNode]): Visual nodes to position in order.
            state (LayoutState): Current placement state.
            depth (int): Current nesting depth. Defaults to zero.
        """
        state.max_depth = max(state.max_depth, depth)
        for node in nodes:
            if isinstance(node, VSkip):
                anchor = state.column
                if node.node_key:
                    self._record_span(
                        node.node_key, HorizontalSpan(anchor, anchor), state
                    )
                continue
            if isinstance(node, VGate):
                self._place_vgate(node, state)
                continue
            if isinstance(node, VInlineBlock):
                self._place_vinline_block(node, state, depth)
                continue
            if isinstance(node, VFoldedBlock):
                self._place_vfolded_block(node, state)
                continue
            if isinstance(node, VUnfoldedSequence):
                self._place_vunfolded_sequence(node, state, depth)

    def _label_height(
        self,
        text: str,
        *,
        font_weight: str = "normal",
    ) -> float:
        """Return a font-aware block header height.

        Args:
            text (str): Header text rendered inside the reserved band.
            font_weight (str): Matplotlib font weight. Defaults to
                ``"normal"``.

        Returns:
            float: Header height including vertical label padding.
        """
        measured = measure_text(
            text,
            font_size=self.style.subfont_size,
            font_weight=font_weight,
            fallback_char_width=self.style.char_width_base,
        ).height
        return (
            max(
                self.style.qubit_y_label_height,
                measured,
            )
            + 2 * self.style.label_padding
        )

    def _power_label_height(self, power: int) -> float:
        """Return a font-aware power-wrapper header height.

        Args:
            power (int): Power value displayed in the wrapper header.

        Returns:
            float: Height reserved for the ``pow=N`` label.
        """
        return self._label_height(f"pow={power}")

    def _gate_draws_text_box(self, node: VGate) -> bool:
        """Return whether a gate node renders one text-bearing rectangle.

        Args:
            node (VGate): Gate-like visual node.

        Returns:
            bool: True for generic gate, block, controlled-unitary, and
                expectation boxes; False for measurements and native symbols.
        """
        if node.kind == VGateKind.GATE:
            native_multi = {
                GateOperationType.CX,
                GateOperationType.CZ,
                GateOperationType.TOFFOLI,
                GateOperationType.SWAP,
            }
            return len(node.qubit_indices) == 1 or node.gate_type not in native_multi
        if node.kind == VGateKind.CONTROLLED_U_BOX:
            target_count = len(node.qubit_indices) - node.control_count
            if node.gate_type == GateOperationType.SWAP:
                return target_count != 2
            native_controlled = {
                GateOperationType.X,
                GateOperationType.CX,
                GateOperationType.TOFFOLI,
                GateOperationType.Z,
                GateOperationType.CZ,
            }
            return target_count == 0 or node.gate_type not in native_controlled
        return node.kind in {
            VGateKind.BLOCK_BOX,
            VGateKind.COMPOSITE_BOX,
            VGateKind.EXPVAL,
        }

    def _gate_box_height(self, node: VGate) -> float:
        """Return a text-bearing gate box height.

        Args:
            node (VGate): Gate-like visual node.

        Returns:
            float: At least ``style.gate_height``, enlarged for the configured
                font when this node draws text.
        """
        if not self._gate_draws_text_box(node):
            return self.style.gate_height
        text_height = measure_text(
            node.label,
            font_size=self.style.font_size,
            fallback_char_width=self.style.char_width_gate,
        ).height
        return max(
            self.style.gate_height,
            text_height + 2 * self.style.label_padding,
        )

    def _folded_text_height(self, node: VFoldedBlock) -> float:
        """Return the text height required by a folded summary.

        Args:
            node (VFoldedBlock): Folded control-flow summary.

        Returns:
            float: Multiline text height including folded-box padding.
        """
        combined = "\n".join([node.header_label, *node.body_lines])
        regular_height = measure_text(
            combined,
            font_size=self.style.subfont_size,
            fallback_char_width=self.style.char_width_gate,
        ).height
        monospace_height = measure_text(
            combined,
            font_size=self.style.subfont_size,
            font_family="monospace",
            fallback_char_width=self.style.char_width_monospace,
        ).height
        bold_header_height = measure_text(
            node.header_label,
            font_size=self.style.subfont_size,
            font_weight="bold",
            fallback_char_width=self.style.char_width_bold,
        ).height
        return (
            max(regular_height, monospace_height, bold_header_height)
            + 2 * self.style.folded_box_text_v_padding
        )

    @staticmethod
    def _orthogonal_segments(
        points: list[tuple[float, float]],
    ) -> tuple[LineSegment, ...]:
        """Convert orthogonal path points into minimal line segments.

        Args:
            points (list[tuple[float, float]]): Ordered path vertices. Every
                consecutive pair must share either x or y.

        Returns:
            tuple[LineSegment, ...]: Nonzero segments with redundant collinear
                vertices removed.

        Raises:
            ValueError: If two consecutive vertices form a diagonal segment.
        """
        deduplicated: list[tuple[float, float]] = []
        for point in points:
            if deduplicated and all(
                math.isclose(current, new, abs_tol=_GEOMETRY_TOLERANCE)
                for current, new in zip(deduplicated[-1], point, strict=True)
            ):
                continue
            if deduplicated:
                previous_x, previous_y = deduplicated[-1]
                if not (
                    math.isclose(previous_x, point[0], abs_tol=_GEOMETRY_TOLERANCE)
                    or math.isclose(
                        previous_y,
                        point[1],
                        abs_tol=_GEOMETRY_TOLERANCE,
                    )
                ):
                    raise ValueError("Connector paths must be orthogonal")
            deduplicated.append(point)

        compressed: list[tuple[float, float]] = []
        for point in deduplicated:
            if len(compressed) >= 2:
                before = compressed[-2]
                previous = compressed[-1]
                same_x = math.isclose(
                    before[0], previous[0], abs_tol=_GEOMETRY_TOLERANCE
                ) and math.isclose(previous[0], point[0], abs_tol=_GEOMETRY_TOLERANCE)
                same_y = math.isclose(
                    before[1], previous[1], abs_tol=_GEOMETRY_TOLERANCE
                ) and math.isclose(previous[1], point[1], abs_tol=_GEOMETRY_TOLERANCE)
                if same_x or same_y:
                    compressed[-1] = point
                    continue
            compressed.append(point)

        return tuple(
            LineSegment(start_x, start_y, end_x, end_y)
            for (start_x, start_y), (end_x, end_y) in zip(
                compressed,
                compressed[1:],
                strict=False,
            )
        )

    @staticmethod
    def _segment_crosses_rect(segment: LineSegment, rect: Rect) -> bool:
        """Return whether an orthogonal segment crosses a rectangle interior.

        Args:
            segment (LineSegment): Horizontal or vertical segment to inspect.
            rect (Rect): Obstacle rectangle whose boundary may be touched.

        Returns:
            bool: True when the segment enters the rectangle interior.

        Raises:
            ValueError: If ``segment`` is diagonal.
        """
        if math.isclose(
            segment.start_y,
            segment.end_y,
            abs_tol=_GEOMETRY_TOLERANCE,
        ):
            left = min(segment.start_x, segment.end_x)
            right = max(segment.start_x, segment.end_x)
            return (
                rect.bottom + _GEOMETRY_TOLERANCE
                < segment.start_y
                < rect.top - _GEOMETRY_TOLERANCE
                and left < rect.right - _GEOMETRY_TOLERANCE
                and right > rect.left + _GEOMETRY_TOLERANCE
            )
        if math.isclose(
            segment.start_x,
            segment.end_x,
            abs_tol=_GEOMETRY_TOLERANCE,
        ):
            bottom = min(segment.start_y, segment.end_y)
            top = max(segment.start_y, segment.end_y)
            return (
                rect.left + _GEOMETRY_TOLERANCE
                < segment.start_x
                < rect.right - _GEOMETRY_TOLERANCE
                and bottom < rect.top - _GEOMETRY_TOLERANCE
                and top > rect.bottom + _GEOMETRY_TOLERANCE
            )
        raise ValueError("Connector segments must be orthogonal")

    def _route_condition_connector(
        self,
        source: tuple[float, float],
        target: tuple[float, float],
        obstacles: list[Rect],
    ) -> tuple[LineSegment, ...]:
        """Find a short orthogonal path that clears all obstacle rectangles.

        The fast path is the usual horizontal-then-vertical connector. When
        that path is blocked, candidate routes use a lane above or below the
        obstacle set and free vertical corridors at obstacle edges.

        Args:
            source (tuple[float, float]): Measurement output port.
            target (tuple[float, float]): Control-flow box input port.
            obstacles (list[Rect]): Rectangles to avoid with wire clearance.

        Returns:
            tuple[LineSegment, ...]: Minimal clear path in deterministic order.

        Raises:
            RuntimeError: If no clear orthogonal route can be found.
        """
        start_x, source_y = source
        target_x, target_y = target
        inflated = [rect.expanded(self.style.qubit_clearance) for rect in obstacles]

        def is_clear(segments: tuple[LineSegment, ...]) -> bool:
            """Return whether every segment clears every inflated obstacle.

            Args:
                segments (tuple[LineSegment, ...]): Candidate route segments.

            Returns:
                bool: True when no segment crosses an obstacle interior.
            """
            return not any(
                self._segment_crosses_rect(segment, obstacle)
                for segment in segments
                for obstacle in inflated
            )

        direct = self._orthogonal_segments([source, (target_x, source_y), target])
        if is_clear(direct):
            return direct

        lanes = (
            max(rect.top for rect in inflated),
            min(rect.bottom for rect in inflated),
        )

        outer_candidates = [
            self._orthogonal_segments(
                [source, (start_x, lane_y), (target_x, lane_y), target]
            )
            for lane_y in lanes
        ]
        clear_outer = [segments for segments in outer_candidates if is_clear(segments)]
        if clear_outer:
            return min(
                clear_outer,
                key=lambda segments: (
                    sum(
                        abs(segment.end_x - segment.start_x)
                        + abs(segment.end_y - segment.start_y)
                        for segment in segments
                    ),
                    len(segments),
                ),
            )

        corridor_x = {start_x, target_x}
        for rect in inflated:
            if start_x <= rect.left <= target_x:
                corridor_x.add(rect.left)
            if start_x <= rect.right <= target_x:
                corridor_x.add(rect.right)

        candidates: list[tuple[float, int, tuple[LineSegment, ...]]] = []
        for lane_y in lanes:
            for entry_x in sorted(corridor_x):
                for exit_x in sorted(corridor_x):
                    segments = self._orthogonal_segments(
                        [
                            source,
                            (entry_x, source_y),
                            (entry_x, lane_y),
                            (exit_x, lane_y),
                            (exit_x, target_y),
                            target,
                        ]
                    )
                    if not is_clear(segments):
                        continue
                    length = sum(
                        abs(segment.end_x - segment.start_x)
                        + abs(segment.end_y - segment.start_y)
                        for segment in segments
                    )
                    candidates.append((length, len(segments), segments))

        if not candidates:
            raise RuntimeError(
                "Unable to route a measurement connector around circuit geometry"
            )
        return min(candidates, key=lambda candidate: candidate[:2])[2]

    def _condition_connector_segments(
        self,
        node: VFoldedBlock | VUnfoldedSequence,
        target: Rect,
        state: LayoutState,
        qubit_y: list[float],
        obstacles: dict[tuple, Rect],
    ) -> tuple[LineSegment, ...]:
        """Route a measurement connector around already placed geometry.

        Args:
            node (VFoldedBlock | VUnfoldedSequence): Conditional visual node.
            target (Rect): Final target box rectangle.
            state (LayoutState): Placement state containing source spans.
            qubit_y (list[float]): Final qubit wire coordinates.
            obstacles (dict[tuple, Rect]): Finalized node rectangles keyed by
                visual-node identity. The source measurement is excluded from
                collision checks.

        Returns:
            tuple[LineSegment, ...]: Horizontal and optional vertical segments,
                or an empty tuple when the source is absent or ambiguous.

        Raises:
            RuntimeError: If geometry blocks every candidate orthogonal route.
        """
        measure_key = node.condition_measure_node_key
        if measure_key is None or measure_key not in state.node_spans:
            return ()
        if len(node.condition_measure_qubit_indices) != 1:
            return ()
        qubit = node.condition_measure_qubit_indices[0]
        if qubit < 0 or qubit >= len(qubit_y):
            return ()
        start_x = state.node_spans[measure_key].right
        if target.left <= start_x:
            return ()
        source_y = qubit_y[qubit]
        corner_radius = min(
            self.style.gate_corner_radius,
            target.width / 2,
            target.height / 2,
        )
        target_y = min(
            max(source_y, target.bottom + corner_radius),
            target.top - corner_radius,
        )

        relevant_obstacles = [
            rect
            for obstacle_key, rect in obstacles.items()
            if obstacle_key != measure_key
            and rect.width > _GEOMETRY_TOLERANCE
            and rect.height > _GEOMETRY_TOLERANCE
            and rect.right + self.style.qubit_clearance > start_x + _GEOMETRY_TOLERANCE
            and rect.left - self.style.qubit_clearance
            < target.left - _GEOMETRY_TOLERANCE
            and not (
                rect.left < start_x < rect.right and rect.bottom < source_y < rect.top
            )
        ]
        return self._route_condition_connector(
            (start_x, source_y),
            (target.left, target_y),
            relevant_obstacles,
        )

    def _finalize_vgate(
        self,
        node: VGate,
        state: LayoutState,
        qubit_y: list[float],
        finalization: _FinalizationState,
    ) -> None:
        """Create the final rectangle for one gate-like node.

        Args:
            node (VGate): Positioned gate-like node.
            state (LayoutState): Placement state containing its full span.
            qubit_y (list[float]): Final qubit wire coordinates.
            finalization (_FinalizationState): Shared typed output maps.
        """
        span = state.node_spans[node.node_key]
        valid_qubits = [q for q in node.qubit_indices if 0 <= q < len(qubit_y)]
        if not valid_qubits:
            finalization.node_rects[node.node_key] = Rect(
                span.left, 0.0, span.right, 0.0
            )
            return

        half_height = self._gate_box_height(node) / 2
        y_values = [qubit_y[q] for q in valid_qubits]
        bottom = min(y_values) - half_height
        top = max(y_values) + half_height

        box_rect: Rect | None = None
        if self._gate_draws_text_box(node):
            box_qubits = valid_qubits
            if node.kind == VGateKind.CONTROLLED_U_BOX:
                box_qubits = valid_qubits[node.control_count :]
            if box_qubits:
                box_y = [qubit_y[q] for q in box_qubits]
                box_width = node.box_width or span.width
                box_rect = Rect(
                    span.center - box_width / 2,
                    min(box_y) - half_height,
                    span.center + box_width / 2,
                    max(box_y) + half_height,
                )
                finalization.gate_box_rects[node.node_key] = box_rect

        if node.kind == VGateKind.CONTROLLED_U_BOX and node.power > 1:
            target_qubits = valid_qubits[node.control_count :]
            if target_qubits and box_rect is not None:
                target_rect = box_rect
                wrapper_rect = Rect(
                    span.left,
                    target_rect.bottom - self.style.power_wrapper_margin,
                    span.right,
                    target_rect.top
                    + self.style.power_wrapper_margin
                    + self._power_label_height(node.power),
                )
                finalization.powered_gate_layouts[node.node_key] = PoweredGateLayout(
                    target_rect,
                    wrapper_rect,
                )
                bottom = min(
                    bottom,
                    wrapper_rect.bottom,
                )
                top = max(
                    top,
                    wrapper_rect.top,
                )
        finalization.node_rects[node.node_key] = Rect(
            span.left, bottom, span.right, top
        )

    def _inline_control_segments(
        self,
        node: VInlineBlock,
        target: Rect,
        qubit_y: list[float],
    ) -> tuple[LineSegment, ...]:
        """Resolve controlled-inline connector segments outside its target box.

        Args:
            node (VInlineBlock): Expanded inline block with optional controls.
            target (Rect): Outermost target block rectangle.
            qubit_y (list[float]): Final qubit wire coordinates.

        Returns:
            tuple[LineSegment, ...]: Zero, one, or two vertical line segments.
        """
        control_y = [
            qubit_y[q] for q in node.control_qubit_indices if 0 <= q < len(qubit_y)
        ]
        if not control_y:
            return ()
        x = target.center_x
        below = [y for y in control_y if y < target.bottom]
        above = [y for y in control_y if y > target.top]
        segments: list[LineSegment] = []
        if below:
            segments.append(LineSegment(x, min(below), x, target.bottom))
        if above:
            segments.append(LineSegment(x, target.top, x, max(above)))
        if not segments:
            segments.append(LineSegment(x, min(control_y), x, max(control_y)))
        return tuple(segments)

    def _finalize_vinline_block(
        self,
        node: VInlineBlock,
        state: LayoutState,
        qubit_y: list[float],
        finalization: _FinalizationState,
    ) -> None:
        """Create bottom-up inner and outer rectangles for an inline block.

        Args:
            node (VInlineBlock): Positioned expanded block.
            state (LayoutState): Placement state containing horizontal spans.
            qubit_y (list[float]): Final qubit wire coordinates.
            finalization (_FinalizationState): Shared typed output maps.
        """
        self._finalize_visual_nodes(
            node.children,
            state,
            qubit_y,
            finalization,
        )
        child_rect = self._rects_for_nodes(node.children, finalization.node_rects)
        control_set = set(node.control_qubit_indices)
        target_qubits = [q for q in node.affected_qubits if q not in control_set]
        if not target_qubits:
            target_qubits = list(node.affected_qubits)
        target_y = [qubit_y[q] for q in target_qubits if 0 <= q < len(qubit_y)]
        if target_y:
            content_bottom = min(target_y) - self.style.gate_height / 2
            content_top = max(target_y) + self.style.gate_height / 2
        elif child_rect is not None:
            content_bottom = child_rect.bottom
            content_top = child_rect.top
        else:
            content_bottom = -self.style.gate_height / 2
            content_top = self.style.gate_height / 2
        if child_rect is not None:
            content_bottom = min(content_bottom, child_rect.bottom)
            content_top = max(content_top, child_rect.top)

        vertical_padding = max(node.border_padding, self.style.box_padding_y)
        inner_span = state.inline_inner_spans[node.node_key]
        inner_rect = Rect(
            inner_span.left,
            content_bottom - vertical_padding,
            inner_span.right,
            content_top + vertical_padding + self._label_height(node.label),
        )
        if node.power > 1:
            margin = self.style.power_wrapper_margin
            outer_span = state.node_spans[node.node_key]
            outer_rect = Rect(
                outer_span.left,
                inner_rect.bottom - margin,
                outer_span.right,
                inner_rect.top + margin + self._power_label_height(node.power),
            )
        else:
            outer_rect = inner_rect
        control_segments = self._inline_control_segments(node, outer_rect, qubit_y)
        layout = InlineBlockLayout(inner_rect, outer_rect, control_segments)
        finalization.inline_block_layouts[node.node_key] = layout
        finalization.node_rects[node.node_key] = outer_rect

    def _finalize_vfolded_block(
        self,
        node: VFoldedBlock,
        state: LayoutState,
        qubit_y: list[float],
        finalization: _FinalizationState,
    ) -> None:
        """Create a folded summary rectangle and optional connector.

        Args:
            node (VFoldedBlock): Positioned folded summary node.
            state (LayoutState): Placement state containing its full span.
            qubit_y (list[float]): Final qubit wire coordinates.
            finalization (_FinalizationState): Shared typed output maps.
        """
        full_span = state.node_spans[node.node_key]
        marker_radius = (
            self.style.folded_marker_radius if node.affected_qubits_precise else 0.0
        )
        box_span = HorizontalSpan(
            full_span.left + marker_radius,
            full_span.right - marker_radius,
        )
        y_values = [qubit_y[q] for q in node.affected_qubits if 0 <= q < len(qubit_y)]
        if y_values:
            center_y = (min(y_values) + max(y_values)) / 2
            qubit_height = max(y_values) - min(y_values) + self.style.gate_height
        else:
            center_y = 0.0
            qubit_height = self.style.gate_height
        text_height = max(
            (1 + len(node.body_lines)) * self.style.line_height
            + 2 * self.style.folded_box_text_v_padding,
            self._folded_text_height(node),
        )
        height = max(qubit_height, text_height)
        rect = Rect(
            box_span.left,
            center_y - height / 2,
            box_span.right,
            center_y + height / 2,
        )
        connector_segments = self._condition_connector_segments(
            node,
            rect,
            state,
            qubit_y,
            finalization.node_rects,
        )
        finalization.folded_block_layouts[node.node_key] = FoldedBlockLayout(
            rect,
            connector_segments,
        )
        marker_y = [qubit_y[q] for q in node.affected_qubits if 0 <= q < len(qubit_y)]
        if marker_radius and marker_y:
            finalization.node_rects[node.node_key] = Rect(
                full_span.left,
                min(rect.bottom, min(marker_y) - marker_radius),
                full_span.right,
                max(rect.top, max(marker_y) + marker_radius),
            )
        else:
            finalization.node_rects[node.node_key] = rect

    def _finalize_boxed_control_flow(
        self,
        node: VUnfoldedSequence,
        state: LayoutState,
        qubit_y: list[float],
        finalization: _FinalizationState,
    ) -> None:
        """Create vertically aligned branch rectangles bottom-up.

        Args:
            node (VUnfoldedSequence): Positioned unfolded IF or WHILE.
            state (LayoutState): Placement state containing branch spans.
            qubit_y (list[float]): Final qubit wire coordinates.
            finalization (_FinalizationState): Shared typed output maps.
        """
        for children in node.iterations:
            self._finalize_visual_nodes(
                children,
                state,
                qubit_y,
                finalization,
            )
        child_rects = [
            rect
            for children in node.iterations
            for child in children
            if (rect := finalization.node_rects.get(child.node_key)) is not None
        ]
        child_union = self._union_rects(child_rects)
        affected = list(node.affected_qubits) or list(
            node.condition_measure_qubit_indices
        )
        y_values = [qubit_y[q] for q in affected if 0 <= q < len(qubit_y)]
        if y_values:
            base_bottom = min(y_values) - self.style.gate_height / 2
            base_top = max(y_values) + self.style.gate_height / 2
        elif child_union is not None:
            base_bottom = child_union.bottom
            base_top = child_union.top
        else:
            base_bottom = -self.style.gate_height / 2
            base_top = self.style.gate_height / 2

        vertical_padding = max(
            self.style.box_padding_y,
            compute_border_padding(self.style, depth=0),
        )
        labels = self._branch_labels(node)
        header_height = max(
            (
                self._label_height(label, font_weight="bold")
                for label in labels
                if label
            ),
            default=self._label_height("if", font_weight="bold"),
        )
        bottom = base_bottom - vertical_padding
        top = base_top + vertical_padding + header_height
        if child_union is not None:
            bottom = min(bottom, child_union.bottom - vertical_padding)
            top = max(
                top,
                child_union.top + vertical_padding + header_height,
            )

        horizontal_spans = state.control_flow_branch_spans[node.node_key]
        boxes = tuple(
            ControlFlowBoxLayout(
                index,
                labels[index] if index < len(labels) else "",
                Rect(span.left, bottom, span.right, top),
            )
            for index, span in enumerate(horizontal_spans)
        )
        outer_rect = boxes[0].rect
        for box in boxes[1:]:
            outer_rect = outer_rect.union(box.rect)
        connector_segments = self._condition_connector_segments(
            node,
            outer_rect,
            state,
            qubit_y,
            finalization.node_rects,
        )
        finalization.control_flow_layouts[node.node_key] = ControlFlowLayout(
            boxes,
            outer_rect,
            connector_segments,
        )
        finalization.node_rects[node.node_key] = outer_rect

    def _finalize_unboxed_sequence(
        self,
        node: VUnfoldedSequence,
        state: LayoutState,
        qubit_y: list[float],
        finalization: _FinalizationState,
    ) -> None:
        """Finalize an unboxed loop as the union of its child rectangles.

        Args:
            node (VUnfoldedSequence): Positioned unboxed loop sequence.
            state (LayoutState): Placement state containing its full span.
            qubit_y (list[float]): Final qubit wire coordinates.
            finalization (_FinalizationState): Shared typed output maps.
        """
        for children in node.iterations:
            self._finalize_visual_nodes(
                children,
                state,
                qubit_y,
                finalization,
            )
        child_rect = self._union_rects(
            [
                rect
                for children in node.iterations
                for child in children
                if (rect := finalization.node_rects.get(child.node_key)) is not None
            ]
        )
        if child_rect is not None:
            finalization.node_rects[node.node_key] = child_rect

    def _finalize_visual_nodes(
        self,
        nodes: list[VisualNode],
        state: LayoutState,
        qubit_y: list[float],
        finalization: _FinalizationState,
    ) -> None:
        """Finalize node rectangles and specialized layouts bottom-up.

        Args:
            nodes (list[VisualNode]): Positioned visual nodes to finalize.
            state (LayoutState): Placement state containing horizontal spans.
            qubit_y (list[float]): Final qubit wire coordinates.
            finalization (_FinalizationState): Shared typed output maps.
        """
        for node in nodes:
            if isinstance(node, VSkip):
                continue
            if isinstance(node, VGate):
                self._finalize_vgate(node, state, qubit_y, finalization)
                continue
            if isinstance(node, VInlineBlock):
                self._finalize_vinline_block(node, state, qubit_y, finalization)
                continue
            if isinstance(node, VFoldedBlock):
                self._finalize_vfolded_block(node, state, qubit_y, finalization)
                continue
            if node.kind in (VUnfoldedKind.IF, VUnfoldedKind.WHILE):
                self._finalize_boxed_control_flow(
                    node,
                    state,
                    qubit_y,
                    finalization,
                )
            else:
                self._finalize_unboxed_sequence(
                    node,
                    state,
                    qubit_y,
                    finalization,
                )

    def _wire_geometry(
        self,
        vc: VisualCircuit,
        state: LayoutState,
        qubit_y: list[float],
    ) -> tuple[dict[int, HorizontalSpan], Rect]:
        """Resolve every wire span and their combined bounds.

        Args:
            vc (VisualCircuit): Visual circuit carrying wire count and names.
            state (LayoutState): Completed placement state.
            qubit_y (list[float]): Final qubit wire coordinates.

        Returns:
            tuple[dict[int, HorizontalSpan], Rect]: Per-wire spans and their
                combined bounding rectangle.
        """
        if vc.num_qubits == 0:
            return {}, Rect(0.0, 0.0, 0.0, 0.0)
        visible_spans = [span for span in state.node_spans.values() if span.width > 0]
        if visible_spans:
            content_left = min(span.left for span in visible_spans)
            content_right = max(span.right for span in visible_spans)
        else:
            content_left = self.style.initial_wire_position
            content_right = self.style.initial_wire_position
        wire_start = content_left - self.style.wire_extension
        default_end = max(content_right, state.actual_width) + self.style.wire_extension
        wire_spans = {
            qubit: HorizontalSpan(
                wire_start,
                max(wire_start, state.qubit_end_positions.get(qubit, default_end)),
            )
            for qubit in range(vc.num_qubits)
        }
        wire_bounds = Rect(
            wire_start,
            min(qubit_y),
            max(span.right for span in wire_spans.values()),
            max(qubit_y),
        )
        return wire_spans, wire_bounds

    def _compute_viewport(
        self,
        vc: VisualCircuit,
        wire_bounds: Rect,
        qubit_y: list[float],
        finalization: _FinalizationState,
    ) -> Rect:
        """Compute a viewport containing every layout-owned primitive.

        Args:
            vc (VisualCircuit): Visual circuit carrying qubit labels.
            wire_bounds (Rect): Combined wire rectangle.
            qubit_y (list[float]): Final qubit wire coordinates.
            finalization (_FinalizationState): Shared typed output maps.

        Returns:
            Rect: Padded viewport containing all geometry and qubit labels.
        """
        if vc.num_qubits == 0:
            return Rect(0.0, 0.0, 4.0, 2.0)
        geometry = wire_bounds
        for rect in finalization.node_rects.values():
            geometry = geometry.union(rect)
        segments = [
            segment
            for layout in finalization.control_flow_layouts.values()
            for segment in layout.connector_segments
        ]
        segments.extend(
            segment
            for layout in finalization.folded_block_layouts.values()
            for segment in layout.connector_segments
        )
        segments.extend(
            segment
            for layout in finalization.inline_block_layouts.values()
            for segment in layout.control_segments
        )
        for segment in segments:
            geometry = geometry.union(
                Rect(
                    min(segment.start_x, segment.end_x),
                    min(segment.start_y, segment.end_y),
                    max(segment.start_x, segment.end_x),
                    max(segment.start_y, segment.end_y),
                )
            )

        label_right = wire_bounds.left - 0.2
        for qubit in range(vc.num_qubits):
            metrics = measure_text(
                vc.qubit_names.get(qubit, f"q{qubit}"),
                font_size=self.style.font_size,
                fallback_char_width=self.style.char_width_gate,
            )
            label_y = qubit_y[qubit]
            geometry = geometry.union(
                Rect(
                    label_right - metrics.width,
                    label_y - metrics.height / 2,
                    label_right,
                    label_y + metrics.height / 2,
                )
            )
        left_margin, right_margin, top_margin, bottom_margin = self.style.margin
        return Rect(
            geometry.left - left_margin,
            geometry.bottom - bottom_margin,
            geometry.right + right_margin,
            geometry.top + top_margin,
        )

    def compute_layout(self, vc: VisualCircuit) -> LayoutResult:
        """Compute complete geometry for a visual circuit.

        Args:
            vc (VisualCircuit): Pre-resolved Visual IR tree and qubit metadata.

        Returns:
            LayoutResult: Authoritative spans, rectangles, connectors, wires,
                viewport, and compatibility fields for existing callers.

        Raises:
            RuntimeError: If a condition connector cannot be routed or nested
                vertical geometry does not converge within the pass limit.
        """
        state = LayoutState()
        for qubit in range(vc.num_qubits):
            state.qubit_right_edges[qubit] = self.style.initial_wire_position
            _ = state.qubit_columns[qubit]
        self._place_visual_nodes(vc.children, state)

        qubit_y, max_above, max_below, finalization = self._resolve_vertical_geometry(
            vc, state
        )
        wire_spans, wire_bounds = self._wire_geometry(vc, state, qubit_y)
        viewport = self._compute_viewport(
            vc,
            wire_bounds,
            qubit_y,
            finalization,
        )
        first_x = state.first_gate_x if state.first_gate_x is not None else 1.0
        content_right = max(
            (span.right for span in state.node_spans.values()), default=1.0
        )

        return LayoutResult(
            width=max(state.column, content_right),
            positions=state.positions,
            block_ranges=state.block_ranges,
            max_depth=state.max_depth,
            block_widths=state.block_widths,
            actual_width=content_right,
            first_gate_x=first_x,
            first_gate_half_width=state.first_gate_half_width,
            qubit_y=qubit_y,
            qubit_end_positions=state.qubit_end_positions,
            inlined_op_keys=state.inlined_op_keys,
            gate_widths=state.gate_widths,
            folded_block_extents=state.folded_block_extents,
            max_above=max_above,
            max_below=max_below,
            node_spans=state.node_spans,
            node_rects=finalization.node_rects,
            gate_box_rects=finalization.gate_box_rects,
            control_flow_layouts=finalization.control_flow_layouts,
            folded_block_layouts=finalization.folded_block_layouts,
            inline_block_layouts=finalization.inline_block_layouts,
            powered_gate_layouts=finalization.powered_gate_layouts,
            viewport=viewport,
            wire_bounds=wire_bounds,
            wire_spans=wire_spans,
        )

    def _initial_qubit_y_positions(self, num_qubits: int) -> list[float]:
        """Return uniformly spaced provisional wire coordinates.

        Args:
            num_qubits (int): Total number of qubit wires.

        Returns:
            list[float]: Descending wire coordinates with the final wire at
                zero.
        """
        return [
            (num_qubits - 1 - qubit) * self.style.qubit_base_spacing
            for qubit in range(num_qubits)
        ]

    def _semantic_qubits(self, node: VisualNode, num_qubits: int) -> list[int]:
        """Return wires semantically owned by one node and its descendants.

        Args:
            node (VisualNode): Visual node whose rectangle needs anchors.
            num_qubits (int): Total wire count used to discard stale indices.

        Returns:
            list[int]: Sorted unique semantic wire indices.
        """
        qubits: set[int] = set()
        if isinstance(node, VGate):
            qubits.update(node.qubit_indices)
        elif isinstance(node, VFoldedBlock):
            qubits.update(node.affected_qubits)
        elif isinstance(node, VInlineBlock):
            controls = set(node.control_qubit_indices)
            qubits.update(q for q in node.affected_qubits if q not in controls)
            for child in node.children:
                qubits.update(self._semantic_qubits(child, num_qubits))
        elif isinstance(node, VUnfoldedSequence):
            qubits.update(node.affected_qubits)
            for iteration in node.iterations:
                for child in iteration:
                    qubits.update(self._semantic_qubits(child, num_qubits))
        return sorted(q for q in qubits if 0 <= q < num_qubits)

    def _geometry_extents(
        self,
        nodes: list[VisualNode],
        finalization: _FinalizationState,
        qubit_y: list[float],
    ) -> tuple[dict[int, float], dict[int, float]]:
        """Derive per-wire vertical reservations from finalized rectangles.

        Args:
            nodes (list[VisualNode]): Root nodes whose geometry is finalized.
            finalization (_FinalizationState): Final rectangle maps.
            qubit_y (list[float]): Provisional wire coordinates.

        Returns:
            tuple[dict[int, float], dict[int, float]]: Maximum space needed
                above and below each semantic boundary wire.
        """
        num_qubits = len(qubit_y)
        max_above = {qubit: 0.0 for qubit in range(num_qubits)}
        max_below = {qubit: 0.0 for qubit in range(num_qubits)}

        def visit(current: list[VisualNode]) -> None:
            """Accumulate extents for a visual subtree.

            Args:
                current (list[VisualNode]): Nodes at the current tree level.
            """
            for node in current:
                if isinstance(node, VInlineBlock):
                    visit(node.children)
                elif isinstance(node, VUnfoldedSequence):
                    for iteration in node.iterations:
                        visit(iteration)

                if isinstance(node, VUnfoldedSequence) and node.kind not in (
                    VUnfoldedKind.IF,
                    VUnfoldedKind.WHILE,
                ):
                    continue
                rect = finalization.node_rects.get(node.node_key)
                anchors = self._semantic_qubits(node, num_qubits)
                if rect is None or not anchors:
                    continue
                top_qubit = anchors[0]
                bottom_qubit = anchors[-1]
                max_above[top_qubit] = max(
                    max_above[top_qubit],
                    rect.top - qubit_y[top_qubit],
                )
                max_below[bottom_qubit] = max(
                    max_below[bottom_qubit],
                    qubit_y[bottom_qubit] - rect.bottom,
                )

        visit(nodes)
        return max_above, max_below

    def _positions_from_extents(
        self,
        num_qubits: int,
        max_above: dict[int, float],
        max_below: dict[int, float],
    ) -> list[float]:
        """Resolve wire coordinates from adjacent rectangle reservations.

        Args:
            num_qubits (int): Total number of qubit wires.
            max_above (dict[int, float]): Required space above boundary wires.
            max_below (dict[int, float]): Required space below boundary wires.

        Returns:
            list[float]: Descending collision-free wire coordinates.
        """
        if num_qubits == 0:
            return []
        spacings = [
            max(
                self.style.qubit_base_spacing,
                max_below[index] + max_above[index + 1] + self.style.qubit_clearance,
            )
            for index in range(num_qubits - 1)
        ]
        cumulative = [0.0]
        for spacing in spacings:
            cumulative.append(cumulative[-1] + spacing)
        maximum = cumulative[-1]
        return [maximum - value for value in cumulative]

    def _resolve_vertical_geometry(
        self,
        vc: VisualCircuit,
        state: LayoutState,
    ) -> tuple[
        list[float],
        dict[int, float],
        dict[int, float],
        _FinalizationState,
    ]:
        """Iterate final rectangles and wire spacing to a stable geometry.

        Rectangles are the source of truth: each pass finalizes the full visual
        tree, derives the space those rectangles require around their semantic
        wires, and recomputes adjacent wire gaps. Nested headers therefore
        reserve their actual footprint instead of relying on depth heuristics.

        Args:
            vc (VisualCircuit): Visual tree and total wire count.
            state (LayoutState): Completed horizontal placement state.

        Returns:
            tuple[list[float], dict[int, float], dict[int, float],
            _FinalizationState]: Stable wire coordinates, vertical extents,
                and the matching finalized geometry maps.

        Raises:
            RuntimeError: If the rectangle-derived wire spacing does not
                converge within the layout pass limit.
        """
        qubit_y = self._initial_qubit_y_positions(vc.num_qubits)
        max_above = {qubit: 0.0 for qubit in range(vc.num_qubits)}
        max_below = {qubit: 0.0 for qubit in range(vc.num_qubits)}

        for _ in range(_MAX_VERTICAL_LAYOUT_PASSES):
            finalization = _FinalizationState()
            self._finalize_visual_nodes(vc.children, state, qubit_y, finalization)
            max_above, max_below = self._geometry_extents(
                vc.children,
                finalization,
                qubit_y,
            )
            resolved = self._positions_from_extents(
                vc.num_qubits,
                max_above,
                max_below,
            )
            if all(
                math.isclose(current, new, abs_tol=_GEOMETRY_TOLERANCE)
                for current, new in zip(qubit_y, resolved, strict=True)
            ):
                return qubit_y, max_above, max_below, finalization
            qubit_y = resolved

        raise RuntimeError(
            "Vertical circuit layout did not converge within "
            f"{_MAX_VERTICAL_LAYOUT_PASSES} passes"
        )
