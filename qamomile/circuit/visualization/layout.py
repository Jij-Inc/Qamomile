"""Circuit layout engine: coordinate computation from Visual IR.

This module provides CircuitLayoutEngine, which assigns x/y coordinates
to pre-resolved Visual IR nodes. It has no matplotlib dependency.
"""

from __future__ import annotations

import math
from collections import defaultdict

from .geometry import compute_block_box_bounds
from .style import CircuitStyle
from .types import (
    LayoutResult,
    LayoutState,
)
from .visual_ir import (
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


class CircuitLayoutEngine:
    """Computes layout coordinates for circuit visualization.

    Takes a VisualCircuit (pre-resolved Visual IR tree) and assigns
    x/y coordinates to each node. No Measure Phase is needed since
    widths are already computed in the Visual IR nodes.

    Has no matplotlib dependency.
    """

    def __init__(self, style: CircuitStyle):
        self.style = style

    # ------------------------------------------------------------------
    # Place Phase: assign coordinates from pre-resolved Visual IR nodes
    # ------------------------------------------------------------------

    def _place_vgate(self, node: VGate, state: LayoutState) -> None:
        """Place a VGate node (gate, measure, block-box, expval)."""
        affected_qubits = list(node.qubit_indices)

        # Find the earliest column where all affected qubits are free
        if affected_qubits:
            min_column = max(state.qubit_columns[q] for q in affected_qubits)
        else:
            min_column = state.column

        op_half_width = node.estimated_width / 2
        columns_needed = max(1, math.ceil(node.estimated_width))

        # Store block_widths for block-box mode operations
        if (
            node.kind
            in (
                VGateKind.BLOCK_BOX,
                VGateKind.COMPOSITE_BOX,
                VGateKind.CONTROLLED_U_BOX,
                VGateKind.EXPVAL,
            )
            and node.box_width is not None
        ):
            state.block_widths[node.node_key] = node.box_width

        # For multi-qubit gates, also check intermediate qubits
        if len(affected_qubits) > 1:
            span_qubits = list(range(min(affected_qubits), max(affected_qubits) + 1))
        else:
            span_qubits = list(affected_qubits)

        # Overlap prevention
        for q in span_qubits:
            if q in state.qubit_right_edges:
                required_center = (
                    state.qubit_right_edges[q] + self.style.gate_gap + op_half_width
                )
                min_column = max(min_column, required_center)

        # Place operation
        state.positions[node.node_key] = min_column
        state.gate_widths[node.node_key] = node.estimated_width
        state.column = max(state.column, min_column + columns_needed)

        # Track first gate position
        if state.first_gate_x is None:
            state.first_gate_x = min_column
            state.first_gate_half_width = op_half_width

        # Update qubit columns and right edges
        right_edge = min_column + op_half_width
        for q in span_qubits:
            state.qubit_columns[q] = right_edge + self.style.gate_gap
            state.qubit_right_edges[q] = right_edge

        # Record measurement positions for wire termination
        if node.kind == VGateKind.MEASURE:
            for q in affected_qubits:
                state.qubit_end_positions[q] = min_column + op_half_width
        elif node.kind == VGateKind.MEASURE_VECTOR:
            for q in affected_qubits:
                state.qubit_end_positions[q] = min_column + op_half_width
        elif node.kind == VGateKind.EXPVAL:
            for q in affected_qubits:
                state.qubit_end_positions[q] = min_column + op_half_width

        # Update actual_width
        op_actual_width = min_column + op_half_width + 0.5
        state.actual_width = max(state.actual_width, op_actual_width)

    def _get_first_child_half_width(self, children: list[VisualNode]) -> float:
        """Get the half-width of the first non-skip child for border alignment."""
        for child in children:
            if isinstance(child, VSkip):
                continue
            if isinstance(child, VGate):
                return child.estimated_width / 2
            if isinstance(child, VInlineBlock):
                return child.final_width / 2
            if isinstance(child, VFoldedBlock):
                return child.folded_width / 2
            if isinstance(child, VUnfoldedSequence):
                # Get first child of first iteration
                if child.iterations:
                    for sub in child.iterations[0]:
                        if isinstance(sub, VSkip):
                            continue
                        if isinstance(sub, VGate):
                            return sub.estimated_width / 2
                        if isinstance(sub, VInlineBlock):
                            return sub.final_width / 2
                        if isinstance(sub, VFoldedBlock):
                            return sub.folded_width / 2
                        break
            break
        return self.style.gate_width / 2  # fallback

    def _place_vinline_block(
        self, node: VInlineBlock, state: LayoutState, depth: int
    ) -> None:
        """Place a VInlineBlock node (inlined CallBlock/ControlledU/CompositeGate)."""
        affected_qubits = node.affected_qubits

        # Align all affected qubits to the maximum right edge
        if affected_qubits:
            max_edge = max(state.qubit_right_edges.get(q, 0.0) for q in affected_qubits)
            for q in affected_qubits:
                state.qubit_right_edges[q] = max_edge
                state.qubit_columns[q] = max_edge + self.style.gate_gap

        # Advance for border extent (left padding)
        border_padding = node.border_padding
        max_gate_width = node.max_gate_width
        border_extent = (
            max_gate_width / 2 + border_padding + self.style.gate_text_padding
        )

        first_child_half = self._get_first_child_half_width(node.children)

        min_border_advance = max(
            0.0,
            border_padding + self.style.gate_text_padding,
        )
        advance = max(min_border_advance, border_extent - first_child_half)
        for q in affected_qubits:
            if q in state.qubit_right_edges:
                state.qubit_right_edges[q] += advance
                state.qubit_columns[q] = (
                    state.qubit_right_edges[q] + self.style.gate_gap
                )

        # Center content when label is wider
        if node.final_width > node.content_width:
            center_offset = (node.final_width - node.content_width) / 2
            for q in affected_qubits:
                state.qubit_right_edges[q] += center_offset
                state.qubit_columns[q] = (
                    state.qubit_right_edges[q] + self.style.gate_gap
                )

        state.max_depth = max(state.max_depth, depth + 1)

        # Recursively place children (already resolved -- no analyzer needed)
        child_scope = node.node_key
        self._place_visual_nodes(node.children, state, depth + 1, child_scope)

        # Compute block range from placed children
        block_op_columns = []
        key_len = len(node.node_key)
        for pos_key, pos_val in state.positions.items():
            if (
                len(pos_key) > key_len
                and pos_key[:key_len] == node.node_key
                and pos_val > 0
            ):
                block_op_columns.append(pos_val)

        if block_op_columns:
            actual_start = min(block_op_columns)
            actual_end = max(block_op_columns)
        else:
            if affected_qubits:
                actual_start = min(
                    state.qubit_columns.get(q, 0) for q in affected_qubits
                )
                actual_end = actual_start
            else:
                actual_start = state.column
                actual_end = state.column

        # Expand actual_start/actual_end for children with explicit box widths
        for pos_key, pos_val in state.positions.items():
            if (
                len(pos_key) > key_len
                and pos_key[:key_len] == node.node_key
                and pos_val > 0
            ):
                bw = state.block_widths.get(pos_key)
                if bw is not None and bw > max_gate_width:
                    extra = (bw - max_gate_width) / 2
                    actual_start = min(actual_start, pos_val - extra)
                    actual_end = max(actual_end, pos_val + extra)

        if affected_qubits:
            state.block_ranges.append(
                {
                    "name": node.label,
                    "start_x": actual_start,
                    "end_x": actual_end,
                    "qubit_indices": affected_qubits,
                    "control_qubit_indices": node.control_qubit_indices,
                    "power": node.power,
                    "depth": depth,
                    "max_gate_width": max_gate_width,
                }
            )

        # Expand first_gate_half_width to cover border extent
        if state.first_gate_x is not None and affected_qubits:
            border_left = (
                actual_start
                - max_gate_width / 2
                - border_padding
                - self.style.gate_text_padding
            )
            current_left = state.first_gate_x - state.first_gate_half_width
            if border_left < current_left:
                state.first_gate_half_width = state.first_gate_x - border_left

        # Compute border right edge using shared helper
        _, border_right_edge = compute_block_box_bounds(
            self.style,
            node.label,
            actual_start,
            actual_end,
            depth,
            max_gate_width,
            node.power,
        )

        for q in affected_qubits:
            state.qubit_right_edges[q] = border_right_edge
            state.qubit_columns[q] = border_right_edge + self.style.gate_gap

    def _place_vfolded_block(self, node: VFoldedBlock, state: LayoutState) -> None:
        """Place a VFoldedBlock node (folded For/While/ForItems/If)."""
        affected_qubits = node.affected_qubits

        if affected_qubits:
            start_column = max(state.qubit_columns[q] for q in affected_qubits)
        elif state.qubit_columns:
            start_column = max(state.qubit_columns.values())
        else:
            start_column = 0

        loop_width = node.folded_width
        op_half_width = loop_width / 2
        gap = self.style.gate_gap

        for q in affected_qubits:
            if q in state.qubit_right_edges:
                required_center = state.qubit_right_edges[q] + gap + op_half_width
                start_column = max(start_column, required_center)

        state.positions[node.node_key] = start_column
        state.block_widths[node.node_key] = loop_width

        if state.first_gate_x is None:
            state.first_gate_x = start_column
            state.first_gate_half_width = op_half_width

        for q in affected_qubits:
            state.qubit_columns[q] = start_column + op_half_width + gap
            state.qubit_right_edges[q] = start_column + op_half_width

        state.column = max(state.column, start_column + 1)
        state.actual_width = max(state.actual_width, start_column + op_half_width + 0.5)

        state.folded_block_extents[node.node_key] = {
            "affected_qubits": affected_qubits,
            "num_text_lines": 1 + len(node.body_lines),
        }

    def _place_vunfolded_sequence(
        self, node: VUnfoldedSequence, state: LayoutState, depth: int
    ) -> None:
        """Place a VUnfoldedSequence node (unfolded For/ForItems/If)."""
        affected_qubits = node.affected_qubits

        for i, iteration_children in enumerate(node.iterations):
            # For If: i=0 is "true", i=1 is "false"
            # For loops: i is iteration index
            if node.kind == VUnfoldedKind.IF:  # Future use: not yet dispatched
                iter_key = (*node.node_key, "true" if i == 0 else "false")
            else:
                iter_key = (*node.node_key, i)

            # Align affected qubits only for IF branches (mutually exclusive
            # alternatives).  For FOR/FOR_ITEMS, skip synchronization so gates
            # on independent qubits pack at the same x-position.
            if affected_qubits and node.kind == VUnfoldedKind.IF:
                max_edge = max(
                    state.qubit_right_edges.get(q, 0.0) for q in affected_qubits
                )
                for q in affected_qubits:
                    state.qubit_right_edges[q] = max_edge
                    state.qubit_columns[q] = max_edge + self.style.gate_gap

            self._place_visual_nodes(iteration_children, state, depth + 1, iter_key)

    def _place_visual_nodes(
        self,
        nodes: list[VisualNode],
        state: LayoutState,
        depth: int = 0,
        scope_path: tuple = (),
    ) -> None:
        """Place Visual IR nodes, dispatching by type."""
        state.max_depth = max(state.max_depth, depth)

        for node in nodes:
            if isinstance(node, VSkip):
                # QInit: register position at 0
                if node.node_key:
                    state.positions[node.node_key] = 0
                continue

            if isinstance(node, VGate):
                self._place_vgate(node, state)
                continue

            if isinstance(node, VInlineBlock):
                state.inlined_op_keys.add(node.node_key)
                self._place_vinline_block(node, state, depth)
                continue

            if isinstance(node, VFoldedBlock):
                self._place_vfolded_block(node, state)
                continue

            if isinstance(node, VUnfoldedSequence):
                self._place_vunfolded_sequence(node, state, depth)
                continue

    def compute_layout(self, vc: VisualCircuit) -> LayoutResult:
        """Compute layout from a VisualCircuit.

        Uses the Visual IR tree which carries all pre-resolved information
        (labels, qubit indices, widths). No Measure Phase is needed since
        widths are already computed in the Visual IR nodes.

        Args:
            vc: VisualCircuit containing pre-resolved Visual IR nodes.

        Returns:
            LayoutResult with all computed positions and sizing.
        """
        state = LayoutState()
        for q_idx in set(vc.qubit_map.values()):
            state.qubit_right_edges[q_idx] = self.style.initial_wire_position
            _ = state.qubit_columns[q_idx]

        self._place_visual_nodes(vc.children, state)

        state.actual_width = max(state.actual_width, state.column)

        qubit_y, max_above, max_below = self._compute_qubit_y_positions(
            vc.num_qubits, state.block_ranges
        )

        return LayoutResult(
            width=state.column,
            positions=state.positions,
            block_ranges=state.block_ranges,
            max_depth=state.max_depth,
            block_widths=state.block_widths,
            gate_widths=state.gate_widths,
            actual_width=state.actual_width,
            first_gate_x=state.first_gate_x if state.first_gate_x is not None else 1.0,
            first_gate_half_width=state.first_gate_half_width,
            qubit_y=qubit_y,
            qubit_end_positions=state.qubit_end_positions,
            inlined_op_keys=state.inlined_op_keys,
            folded_block_extents=state.folded_block_extents,
            max_above=max_above,
            max_below=max_below,
        )

    def _compute_qubit_y_positions(
        self, num_qubits: int, block_ranges: list[dict]
    ) -> list[float]:
        """Compute y-positions with variable spacing based on block borders.

        Each qubit's vertical extent is calculated from all block borders
        covering it. Multi-qubit borders contribute label height only at
        the topmost qubit.

        Args:
            num_qubits: Total number of qubits.
            block_ranges: List of block range dicts from layout phase, each
                containing 'qubit_indices', 'depth', 'start_x', 'end_x'.

        Returns:
            Tuple of (y_positions, max_above, max_below).
        """
        if not block_ranges:
            max_above: dict[int, float] = {}
            max_below: dict[int, float] = {}
            if num_qubits <= 1:
                return [0.0] * num_qubits, max_above, max_below
            return (
                [float(num_qubits - 1 - q) for q in range(num_qubits)],
                max_above,
                max_below,
            )

        gate_half = self.style.gate_height / 2
        base_padding = 0.4
        label_height = self.style.qubit_y_label_height
        label_step = label_height + self.style.label_step_gap
        overlap_step = label_height + self.style.overlap_step_gap
        clearance = self.style.qubit_clearance
        base_spacing = self.style.qubit_base_spacing

        # Compute max_depth for inverted label offset
        max_depth = max((b["depth"] for b in block_ranges), default=0)

        # For each qubit, track max extent above and below its wire
        max_above = {q: 0.0 for q in range(num_qubits)}
        max_below = {q: 0.0 for q in range(num_qubits)}

        # Count overlapping borders per (topmost_qubit, position)
        overlap_counts = defaultdict(lambda: defaultdict(int))
        for br in block_ranges:
            top_q = min(br["qubit_indices"])
            pos_key = (br["start_x"], br["end_x"])
            overlap_counts[top_q][pos_key] += 1

        qubit_max_overlaps = {}
        for q in range(num_qubits):
            if q in overlap_counts:
                qubit_max_overlaps[q] = max(overlap_counts[q].values())
            else:
                qubit_max_overlaps[q] = 0

        for br in block_ranges:
            qubits = sorted(br["qubit_indices"])
            depth = br["depth"]
            padding = max(
                self.style.min_block_padding,
                base_padding - depth * self.style.qubit_clearance,
            )

            top_q = qubits[0]  # smallest index = topmost (highest y)
            bottom_q = qubits[-1]  # largest index = bottommost (lowest y)

            # Extra vertical space for power wrapper box
            power_extra_above = 0.0
            power_extra_below = 0.0
            if br.get("power", 1) > 1:
                power_extra_above = label_height + self.style.power_wrapper_margin * 2
                power_extra_below = self.style.power_wrapper_margin

            # Below the bottommost qubit
            extent_below = gate_half + padding + power_extra_below
            max_below[bottom_q] = max(max_below[bottom_q], extent_below)

            # Above the topmost qubit (includes label + offsets)
            depth_offset = (max_depth - depth) * label_step
            overlap_offset = max(0, qubit_max_overlaps.get(top_q, 0) - 1) * overlap_step
            extent_above = (
                gate_half
                + padding
                + label_height
                + depth_offset
                + overlap_offset
                + power_extra_above
            )
            max_above[top_q] = max(max_above[top_q], extent_above)

            # Middle/non-top/non-bottom qubits: just gate+padding
            for q in qubits:
                if q != top_q:
                    max_above[q] = max(max_above[q], gate_half + padding)
                if q != bottom_q:
                    max_below[q] = max(max_below[q], gate_half + padding)

        if num_qubits <= 1:
            return [0.0] * num_qubits, max_above, max_below

        # Compute spacings between adjacent pairs
        spacings = []
        for i in range(num_qubits - 1):
            spacing = max(base_spacing, max_below[i] + max_above[i + 1] + clearance)
            spacings.append(spacing)

        # Build cumulative y-positions (qubit 0 at top)
        y_positions = [0.0] * num_qubits
        cumulative = 0.0
        for i in range(1, num_qubits):
            cumulative += spacings[i - 1]
            y_positions[i] = cumulative

        # Flip: qubit 0 at top (highest y), qubit N-1 at bottom (y~0)
        max_val = y_positions[-1]
        for i in range(num_qubits):
            y_positions[i] = max_val - y_positions[i]

        return y_positions, max_above, max_below
