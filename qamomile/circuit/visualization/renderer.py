"""Matplotlib-based circuit rendering.

This module provides MatplotlibRenderer, which handles all matplotlib
drawing operations for circuit visualization.
"""

from __future__ import annotations

import io
import math
from collections import defaultdict
from typing import TYPE_CHECKING

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from .geometry import (
    compute_block_box_bounds,
    compute_border_padding,
    compute_nested_block_box_bounds,
)
from .style import CircuitStyle
from .types import (
    PORDER_GATE,
    PORDER_LINE,
    PORDER_TEXT,
    PORDER_WIRE,
    LayoutResult,
)
from .visual_ir import (
    GateOperationType,
    VFoldedBlock,
    VFoldedKind,
    VGate,
    VGateKind,
    VInlineBlock,
    VSkip,
    VUnfoldedSequence,
    VisualCircuit,
    VisualNode,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.transforms import Bbox


class MatplotlibRenderer:
    """Renders circuit diagrams using matplotlib.

    Takes pre-computed layout coordinates and draws the circuit
    using matplotlib primitives.
    """

    def __init__(self, style: CircuitStyle):
        self.style = style
        # These are set during render()
        self.layout: LayoutResult | None = None
        self.qubit_y: list[float] = []
        self.qubit_end_positions: dict[int, float] = {}
        self.qubit_names: dict[int, str] = {}
        self._inlined_op_keys: set[tuple] = set()

    def render(self, vc: VisualCircuit, layout: LayoutResult) -> Figure:
        """Render the circuit from a VisualCircuit.

        Uses the Visual IR tree which carries all pre-resolved information.

        Args:
            vc: VisualCircuit containing pre-resolved Visual IR nodes.
            layout: Pre-computed layout result.

        Returns:
            matplotlib Figure.
        """
        self.layout = layout
        self.qubit_y = layout.qubit_y
        self.qubit_end_positions = layout.qubit_end_positions
        self.qubit_names = vc.qubit_names
        self._inlined_op_keys = layout.inlined_op_keys
        self._output_names = vc.output_names

        fig = self._create_figure(vc.num_qubits)
        self._draw_wires(fig, vc.num_qubits, vc.qubit_map)
        self._draw_operations(fig, vc)
        self._add_jupyter_display_support(fig)
        return fig

    def _draw_operations(self, fig: Figure, vc: VisualCircuit) -> None:
        """Draw all operations from Visual IR nodes.

        Dispatches on VisualNode types instead of IR Operation types,
        eliminating all Analyzer dependencies during drawing.

        Args:
            fig: matplotlib Figure.
            vc: VisualCircuit containing pre-resolved Visual IR nodes.
        """
        positions = self.layout.positions
        block_ranges = self.layout.block_ranges
        block_widths = self.layout.block_widths
        gate_widths = self.layout.gate_widths

        # First, draw block borders (behind gates)
        max_depth = max((b["depth"] for b in block_ranges), default=0)

        # Pre-expand outer block boundaries to contain inner blocks
        for inner in sorted(block_ranges, key=lambda b: -b["depth"]):
            for outer in block_ranges:
                if outer["depth"] < inner["depth"]:
                    inner_qubits = set(inner["qubit_indices"])
                    outer_qubits = set(outer["qubit_indices"])
                    if inner_qubits.issubset(outer_qubits):
                        if (
                            inner["end_x"] < outer["start_x"]
                            or inner["start_x"] > outer["end_x"]
                        ):
                            continue
                        inner_mgw = inner.get("max_gate_width", self.style.gate_width)
                        inner_box_left, inner_box_right = compute_block_box_bounds(
                            self.style,
                            inner.get("name", "block"),
                            inner["start_x"],
                            inner["end_x"],
                            inner["depth"],
                            inner_mgw,
                            inner.get("power", 1),
                        )
                        outer_mgw = outer.get("max_gate_width", self.style.gate_width)
                        outer_padding = compute_border_padding(
                            self.style, outer["depth"]
                        )
                        outer_gtp = self.style.gate_text_padding
                        outer_box_left = (
                            outer["start_x"] - outer_mgw / 2 - outer_padding - outer_gtp
                        )
                        _, outer_box_right = compute_block_box_bounds(
                            self.style,
                            outer.get("name", "block"),
                            outer["start_x"],
                            outer["end_x"],
                            outer["depth"],
                            outer_mgw,
                            outer.get("power", 1),
                        )
                        margin = 0.1
                        if inner_box_left - margin < outer_box_left:
                            outer["start_x"] = (
                                inner_box_left
                                - margin
                                + outer_mgw / 2
                                + outer_padding
                                + outer_gtp
                            )
                        if inner_box_right + margin > outer_box_right:
                            outer["end_x"] = (
                                inner_box_right
                                + margin
                                - outer_mgw / 2
                                - outer_padding
                                - outer_gtp
                            )

        # Group blocks by topmost qubit for overlap calculation
        qubit_blocks: dict[int, list[dict]] = defaultdict(list)
        for block_info in block_ranges:
            top_qubit = min(block_info["qubit_indices"])
            qubit_blocks[top_qubit].append(block_info)

        for qubit, blocks in qubit_blocks.items():
            blocks.sort(key=lambda b: b["start_x"])
            for i, block_info in enumerate(blocks):
                overlap_count = 0
                block_x_start = block_info["start_x"]
                block_x_end = block_info["end_x"]
                for j in range(i):
                    prev_block = blocks[j]
                    prev_x_start = prev_block["start_x"]
                    prev_x_end = prev_block["end_x"]
                    if block_x_start < prev_x_end + 0.5:
                        if (
                            prev_x_start <= block_x_start and prev_x_end >= block_x_end
                        ) or (
                            block_x_start <= prev_x_start and block_x_end >= prev_x_end
                        ):
                            continue
                        overlap_count += 1

                ctrl_indices = block_info.get("control_qubit_indices", [])
                if ctrl_indices:
                    target_only = [
                        q for q in block_info["qubit_indices"] if q not in ctrl_indices
                    ]
                    border_qubits = (
                        target_only if target_only else block_info["qubit_indices"]
                    )
                else:
                    border_qubits = block_info["qubit_indices"]

                box_bottom, box_top = self._draw_inlined_block_border(
                    fig._qm_ax,
                    name=block_info["name"],
                    start_x=block_info["start_x"],
                    end_x=block_info["end_x"],
                    qubit_indices=border_qubits,
                    depth=block_info["depth"],
                    max_gate_width=block_info.get("max_gate_width"),
                    max_depth=max_depth,
                    overlap_index=overlap_count,
                    is_controlled=bool(ctrl_indices),
                    power=block_info.get("power", 1),
                )

                # Draw control dots and vertical line for ControlledUOperation
                if ctrl_indices:
                    ax = fig._qm_ax
                    mgw = block_info.get("max_gate_width", self.style.gate_width)
                    bl, br = compute_block_box_bounds(
                        self.style,
                        block_info["name"],
                        block_info["start_x"],
                        block_info["end_x"],
                        block_info["depth"],
                        mgw,
                        block_info.get("power", 1),
                    )
                    block_center_x = (bl + br) / 2
                    ctrl_y_list = [self.qubit_y[q] for q in ctrl_indices]

                    border_endpoints: list[float] = []
                    min_ctrl = min(ctrl_y_list)
                    max_ctrl = max(ctrl_y_list)
                    if max_ctrl > box_top:
                        border_endpoints.append(box_top)
                    if min_ctrl < box_bottom:
                        border_endpoints.append(box_bottom)
                    all_y_endpoints = ctrl_y_list + border_endpoints
                    line_min_y = min(all_y_endpoints)
                    line_max_y = max(all_y_endpoints)
                    ax.add_line(
                        mlines.Line2D(
                            [block_center_x, block_center_x],
                            [line_min_y, line_max_y],
                            color=self.style.wire_color,
                            linewidth=1.5,
                            zorder=PORDER_LINE,
                        )
                    )

                    for ctrl_idx in ctrl_indices:
                        ctrl_y = self.qubit_y[ctrl_idx]
                        self._draw_control_dot(ax, block_center_x, ctrl_y)

        # Draw operations from Visual IR tree
        self._draw_visual_nodes(fig, vc.children, positions, block_widths, gate_widths)

    def _draw_inlined_block_border(
        self,
        ax: Axes,
        name: str,
        start_x: float,
        end_x: float,
        qubit_indices: list[int],
        depth: int = 0,
        max_gate_width: float | None = None,
        max_depth: int = 0,
        overlap_index: int = 0,
        is_controlled: bool = False,
        power: int = 1,
    ) -> tuple[float, float]:
        """Draw dashed border around inlined block operations.

        When power > 1, draws nested boxes: an outer wrapper with "pow=N"
        and an inner box with the block name.
        """
        if max_gate_width is None:
            max_gate_width = self.style.gate_width

        y_coords = [self.qubit_y[q] for q in qubit_indices]
        min_y = min(y_coords)
        max_y = max(y_coords)

        text_height = self._calculate_text_height(ax, name, self.style.subfont_size)
        padding = compute_border_padding(self.style, depth)

        lp = self.style.label_padding
        label_height = text_height + 2 * lp

        depth_label_offset = (max_depth - depth) * (
            label_height + self.style.label_step_gap
        )
        overlap_label_offset = overlap_index * (
            label_height + self.style.overlap_step_gap
        )
        label_vertical_offset = depth_label_offset + overlap_label_offset

        (inner_left, inner_right), (outer_left, outer_right) = (
            compute_nested_block_box_bounds(
                self.style, name, start_x, end_x, depth, max_gate_width, power
            )
        )

        inner_bottom = min_y - self.style.gate_height / 2 - padding
        inner_top = (
            max_y
            + self.style.gate_height / 2
            + padding
            + label_height
            + label_vertical_offset
        )

        block_zorder = PORDER_WIRE + 0.5 + depth * 0.1
        linestyle = "-" if is_controlled else "--"
        facecolor = self.style.background_color if is_controlled else "none"

        if power > 1:
            # Outer wrapper box for "pow=N"
            margin = self.style.power_wrapper_margin
            pow_text_height = self._calculate_text_height(
                ax, f"pow={power}", self.style.subfont_size
            )
            pow_label_height = pow_text_height + 2 * lp

            outer_bottom = inner_bottom - margin
            outer_top = inner_top + pow_label_height + margin

            outer_zorder = block_zorder - 0.01
            outer_rect = mpatches.Rectangle(
                (outer_left, outer_bottom),
                outer_right - outer_left,
                outer_top - outer_bottom,
                facecolor=facecolor,
                edgecolor=self.style.block_border_color,
                linewidth=1.5,
                linestyle=linestyle,
                zorder=outer_zorder,
            )
            ax.add_patch(outer_rect)

            # "pow=N" label on outer box (right-aligned)
            pow_right = outer_right - self.style.label_horizontal_padding
            ax.text(
                pow_right,
                outer_top - lp,
                f"pow={power}",
                ha="right",
                va="top",
                fontsize=self.style.subfont_size,
                color=self.style.block_border_color,
                zorder=PORDER_TEXT,
            )

        # Inner box
        inner_rect = mpatches.Rectangle(
            (inner_left, inner_bottom),
            inner_right - inner_left,
            inner_top - inner_bottom,
            facecolor=facecolor,
            edgecolor=self.style.block_border_color,
            linewidth=1.5,
            linestyle=linestyle,
            zorder=block_zorder,
        )
        ax.add_patch(inner_rect)

        if is_controlled:
            wire_zorder = block_zorder + 0.05
            for q in qubit_indices:
                wire_y = self.qubit_y[q]
                ax.add_line(
                    mlines.Line2D(
                        [inner_left, inner_right],
                        [wire_y, wire_y],
                        color=self.style.wire_color,
                        linewidth=1.0,
                        zorder=wire_zorder,
                    )
                )

        # Block name on inner box label bar
        inner_label_left = inner_left + self.style.label_horizontal_padding
        ax.text(
            inner_label_left,
            inner_top - lp,
            name,
            ha="left",
            va="top",
            fontsize=self.style.subfont_size,
            color=self.style.block_border_color,
            zorder=PORDER_TEXT,
        )

        # Return outermost bounds for control dot logic
        if power > 1:
            return outer_bottom, outer_top
        return inner_bottom, inner_top

    def _draw_visual_nodes(
        self,
        fig: Figure,
        nodes: list[VisualNode],
        positions: dict[tuple, float],
        block_widths: dict[tuple, float],
        gate_widths: dict[tuple, float],
    ) -> None:
        """Recursively draw Visual IR nodes."""
        for node in nodes:
            if isinstance(node, VSkip):
                continue

            if isinstance(node, VGate):
                if node.node_key in positions:
                    self._draw_vgate(
                        fig,
                        node,
                        positions[node.node_key],
                        block_widths.get(node.node_key),
                        gate_widths.get(node.node_key),
                    )
                continue

            if isinstance(node, VInlineBlock):
                # Children are drawn recursively
                self._draw_visual_nodes(
                    fig, node.children, positions, block_widths, gate_widths
                )
                continue

            if isinstance(node, VFoldedBlock):
                if node.node_key in positions:
                    self._draw_vfolded_block(
                        fig,
                        node,
                        positions[node.node_key],
                        block_widths.get(node.node_key),
                    )
                continue

            if isinstance(node, VUnfoldedSequence):
                for iteration_children in node.iterations:
                    self._draw_visual_nodes(
                        fig,
                        iteration_children,
                        positions,
                        block_widths,
                        gate_widths,
                    )
                continue

    def _draw_vgate(
        self,
        fig: Figure,
        node: VGate,
        x_pos: float,
        block_width: float | None = None,
        layout_width: float | None = None,
    ) -> None:
        """Draw a VGate node using pre-resolved fields."""
        ax = fig._qm_ax
        qubit_indices = node.qubit_indices

        if not qubit_indices:
            return

        y_coords = [self.qubit_y[q] for q in qubit_indices]

        if node.kind == VGateKind.GATE:
            width = layout_width or node.estimated_width
            if len(qubit_indices) == 1:
                self._draw_vgate_single(ax, node, x_pos, y_coords[0], width)
            else:
                self._draw_vgate_multi(ax, node, x_pos, y_coords, width)

        elif node.kind == VGateKind.MEASURE:
            for y in y_coords:
                self._draw_measurement_box(ax, x_pos, y)

        elif node.kind == VGateKind.MEASURE_VECTOR:
            for y in y_coords:
                self._draw_measurement_box(ax, x_pos, y)

        elif node.kind == VGateKind.BLOCK_BOX:
            self._draw_vblock_box(ax, node, x_pos, block_width, is_call_block=True)

        elif node.kind == VGateKind.COMPOSITE_BOX:
            self._draw_vblock_box(ax, node, x_pos, block_width, is_call_block=False)

        elif node.kind == VGateKind.CONTROLLED_U_BOX:
            self._draw_vcontrolled_u_box(ax, node, x_pos, block_width)

        elif node.kind == VGateKind.EXPVAL:
            self._draw_vexpval(ax, node, x_pos, block_width)

    def _draw_vgate_single(
        self, ax: Axes, node: VGate, x: float, y: float, width: float
    ) -> None:
        """Draw a single-qubit gate from VGate."""
        height = self.style.gate_height

        rect = mpatches.FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle=mpatches.BoxStyle.Round(
                pad=0, rounding_size=self.style.gate_corner_radius
            ),
            facecolor=self.style.gate_face_color,
            edgecolor=self.style.wire_color,
            linewidth=1,
            zorder=PORDER_GATE,
        )
        ax.add_patch(rect)

        ax.text(
            x,
            y,
            node.label,
            ha="center",
            va="center",
            color=self.style.gate_text_color,
            fontsize=self.style.font_size,
            zorder=PORDER_TEXT,
        )

    def _draw_vgate_multi(
        self, ax: Axes, node: VGate, x: float, y_coords: list[float], width: float
    ) -> None:
        """Draw a multi-qubit gate from VGate."""
        min_y = min(y_coords)
        max_y = max(y_coords)

        # Draw vertical connection line
        line = mlines.Line2D(
            [x, x],
            [min_y, max_y],
            color=self.style.connection_line_color,
            linewidth=2,
            zorder=PORDER_GATE - 1,
        )
        ax.add_line(line)

        # Dispatch on gate_type for special rendering
        if node.gate_type == GateOperationType.CX:
            self._draw_control_dot(ax, x, y_coords[0])
            self._draw_target_x(ax, x, y_coords[1])
        elif node.gate_type == GateOperationType.CZ:
            for y in y_coords:
                self._draw_control_dot(ax, x, y)
        elif node.gate_type == GateOperationType.SWAP:
            for y in y_coords:
                self._draw_swap_x(ax, x, y)
        else:
            # Generic multi-qubit gate box
            height = max_y - min_y + self.style.gate_height
            y_center = (min_y + max_y) / 2

            rect = mpatches.FancyBboxPatch(
                (x - width / 2, min_y - self.style.gate_height / 2),
                width,
                height,
                boxstyle=mpatches.BoxStyle.Round(
                    pad=0, rounding_size=self.style.gate_corner_radius
                ),
                facecolor=self.style.gate_face_color,
                edgecolor=self.style.wire_color,
                linewidth=1,
                zorder=PORDER_GATE,
            )
            ax.add_patch(rect)

            ax.text(
                x,
                y_center,
                node.label,
                ha="center",
                va="center",
                color=self.style.gate_text_color,
                fontsize=self.style.font_size,
                zorder=PORDER_TEXT,
            )

            # Semicircle connection dots
            half_w = width / 2
            dot_radius = 0.05
            for y in y_coords:
                left_dot = mpatches.Wedge(
                    (x - half_w, y),
                    dot_radius,
                    theta1=90,
                    theta2=270,
                    facecolor=self.style.gate_text_color,
                    edgecolor=self.style.gate_text_color,
                    zorder=PORDER_GATE + 1,
                )
                ax.add_patch(left_dot)
                right_dot = mpatches.Wedge(
                    (x + half_w, y),
                    dot_radius,
                    theta1=270,
                    theta2=90,
                    facecolor=self.style.gate_text_color,
                    edgecolor=self.style.gate_text_color,
                    zorder=PORDER_GATE + 1,
                )
                ax.add_patch(right_dot)

    def _draw_vblock_box(
        self,
        ax: Axes,
        node: VGate,
        x_pos: float,
        block_width: float | None,
        is_call_block: bool,
    ) -> None:
        """Draw a block-box (CallBlock or CompositeGate) from VGate."""
        qubit_indices = node.qubit_indices
        if not qubit_indices:
            return

        y_coords = [self.qubit_y[q] for q in qubit_indices]
        min_y = min(y_coords)
        max_y = max(y_coords)

        width = block_width or self.style.gate_width
        height = max_y - min_y + self.style.gate_height
        y_center = (min_y + max_y) / 2

        if is_call_block:
            face_color = self.style.block_face_color
            edge_color = self.style.block_box_edge_color
            text_color = self.style.block_text_color
            linestyle = "--"
        else:
            face_color = self.style.gate_face_color
            edge_color = self.style.wire_color
            text_color = self.style.gate_text_color
            linestyle = "-"

        rect = mpatches.FancyBboxPatch(
            (x_pos - width / 2, min_y - self.style.gate_height / 2),
            width,
            height,
            boxstyle=mpatches.BoxStyle.Round(
                pad=0, rounding_size=self.style.gate_corner_radius
            ),
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=1.5,
            linestyle=linestyle,
            zorder=PORDER_GATE,
        )
        ax.add_patch(rect)

        ax.text(
            x_pos,
            y_center,
            node.label,
            ha="center",
            va="center",
            color=text_color,
            fontsize=self.style.font_size,
            zorder=PORDER_TEXT,
        )

    def _draw_vcontrolled_u_box(
        self,
        ax: Axes,
        node: VGate,
        x_pos: float,
        block_width: float | None,
    ) -> None:
        """Draw a ControlledU box from VGate with control dots."""
        qubit_indices = node.qubit_indices
        if not qubit_indices:
            return

        # Split control and target indices
        control_indices = qubit_indices[: node.control_count]
        target_indices = qubit_indices[node.control_count :]

        control_y = [self.qubit_y[q] for q in control_indices]
        target_y = [self.qubit_y[q] for q in target_indices]

        all_y = control_y + target_y
        if not all_y:
            return

        # Draw vertical connection line
        line = mlines.Line2D(
            [x_pos, x_pos],
            [min(all_y), max(all_y)],
            color=self.style.connection_line_color,
            linewidth=2,
            zorder=PORDER_GATE - 1,
        )
        ax.add_line(line)

        # Draw control dots
        for y in control_y:
            self._draw_control_dot(ax, x_pos, y)

        # Draw target box
        if target_y:
            width = block_width or self.style.gate_width
            min_target_y = min(target_y)
            max_target_y = max(target_y)
            height = max_target_y - min_target_y + self.style.gate_height
            y_center = (min_target_y + max_target_y) / 2

            rect = mpatches.FancyBboxPatch(
                (x_pos - width / 2, min_target_y - self.style.gate_height / 2),
                width,
                height,
                boxstyle=mpatches.BoxStyle.Round(
                    pad=0, rounding_size=self.style.gate_corner_radius
                ),
                facecolor=self.style.gate_face_color,
                edgecolor=self.style.wire_color,
                linewidth=1,
                zorder=PORDER_GATE,
            )
            ax.add_patch(rect)

            ax.text(
                x_pos,
                y_center,
                node.label,
                ha="center",
                va="center",
                color=self.style.gate_text_color,
                fontsize=self.style.font_size,
                zorder=PORDER_TEXT,
            )

    def _draw_vexpval(
        self,
        ax: Axes,
        node: VGate,
        x_pos: float,
        block_width: float | None,
    ) -> None:
        """Draw an expectation value operation from VGate."""
        qubit_indices = node.qubit_indices
        if not qubit_indices:
            return

        y_coords = [self.qubit_y[q] for q in qubit_indices]
        min_y = min(y_coords)
        max_y = max(y_coords)

        width = block_width or self.style.gate_width
        height = max_y - min_y + self.style.gate_height
        y_center = (min_y + max_y) / 2

        rect = mpatches.FancyBboxPatch(
            (x_pos - width / 2, min_y - self.style.gate_height / 2),
            width,
            height,
            boxstyle=mpatches.BoxStyle.Round(
                pad=0, rounding_size=self.style.gate_corner_radius
            ),
            facecolor=self.style.expval_face_color,
            edgecolor=self.style.expval_edge_color,
            linewidth=1.5,
            zorder=PORDER_GATE,
        )
        ax.add_patch(rect)

        ax.text(
            x_pos,
            y_center,
            node.label,
            ha="center",
            va="center",
            color=self.style.expval_text_color,
            fontsize=self.style.font_size,
            zorder=PORDER_TEXT,
        )

    def _draw_vfolded_block(
        self,
        fig: Figure,
        node: VFoldedBlock,
        x_pos: float,
        block_width: float | None,
    ) -> None:
        """Draw a folded control-flow block from VFoldedBlock."""
        affected_qubits = node.affected_qubits
        if not affected_qubits:
            return

        ax = fig._qm_ax
        y_coords = [self.qubit_y[q] for q in affected_qubits]
        min_y = min(y_coords)
        max_y = max(y_coords)

        header = node.header_label
        body_lines = node.body_lines
        width = block_width or node.folded_width

        # Select colors based on kind
        if node.kind == VFoldedKind.FOR:
            face_color = self.style.for_loop_face_color
            edge_color = self.style.for_loop_edge_color
            text_color = self.style.for_loop_text_color
            linestyle = "-"
        elif node.kind == VFoldedKind.WHILE:
            face_color = self.style.while_loop_face_color
            edge_color = self.style.while_loop_edge_color
            text_color = self.style.while_loop_text_color
            linestyle = "--"
        elif node.kind == VFoldedKind.FOR_ITEMS:
            face_color = self.style.for_loop_face_color
            edge_color = self.style.for_loop_edge_color
            text_color = self.style.for_loop_text_color
            linestyle = "-"
        elif node.kind == VFoldedKind.IF:
            face_color = self.style.if_face_color
            edge_color = self.style.if_edge_color
            text_color = self.style.if_text_color
            linestyle = "-."
        else:
            face_color = self.style.for_loop_face_color
            edge_color = self.style.for_loop_edge_color
            text_color = self.style.for_loop_text_color
            linestyle = "-"

        # Calculate box dimensions
        num_text_lines = 1 + len(body_lines)
        min_text_height = (
            num_text_lines * self.style.line_height
            + 2 * self.style.folded_box_text_v_padding
        )
        qubit_span_height = max_y - min_y + self.style.gate_height
        height = max(qubit_span_height, min_text_height)
        y_center = (min_y + max_y) / 2
        box_top = y_center + height / 2
        box_bottom = y_center - height / 2

        rect = mpatches.FancyBboxPatch(
            (x_pos - width / 2, box_bottom),
            width,
            height,
            boxstyle=mpatches.BoxStyle.Round(
                pad=0, rounding_size=self.style.gate_corner_radius
            ),
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=1.5,
            linestyle=linestyle,
            zorder=PORDER_GATE,
        )
        ax.add_patch(rect)

        # For ForOperation: header at top + body text below
        if node.kind in (VFoldedKind.FOR, VFoldedKind.FOR_ITEMS) and body_lines:
            header_y = box_top - self.style.label_vertical_offset - 0.15
            header_text = ax.text(
                x_pos,
                header_y,
                header,
                ha="center",
                va="top",
                color=text_color,
                fontsize=self.style.subfont_size,
                fontweight="bold",
                zorder=PORDER_TEXT,
                clip_on=True,
            )
            header_text.set_clip_path(rect)

            body_text = "\n".join(body_lines)
            body_center_y = (box_bottom + header_y) / 2
            body_text_obj = ax.text(
                x_pos,
                body_center_y,
                body_text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=self.style.subfont_size,
                family="monospace",
                zorder=PORDER_TEXT,
                clip_on=True,
            )
            body_text_obj.set_clip_path(rect)
        else:
            # While/ForItems/If or For without body: combined label
            combined = header
            if body_lines:
                combined += "\n" + "\n".join(body_lines)
            ax.text(
                x_pos,
                y_center,
                combined,
                ha="center",
                va="center",
                color=text_color,
                fontsize=self.style.subfont_size,
                zorder=PORDER_TEXT,
            )

    def _add_jupyter_display_support(self, fig: Figure) -> None:
        """Add Jupyter display support to the figure.

        This adds _repr_png_() method to enable automatic display in Jupyter notebooks.

        Args:
            fig: matplotlib Figure to enhance.
        """

        def _repr_png_() -> bytes:
            """Return PNG representation for Jupyter display."""
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            return buf.read()

        # Attach the method to the figure instance
        fig._repr_png_ = _repr_png_

    def _measure_text_bbox(self, ax: Axes, text: str, fontsize: int) -> Bbox:
        """Measure rendered text bbox in display pixels.

        Args:
            ax: matplotlib Axes used for rendering.
            text: Text string to measure.
            fontsize: Font size in points.

        Returns:
            matplotlib Bbox of the text in display coordinates.
        """
        if ax.figure.canvas is None or not hasattr(ax.figure.canvas, "get_renderer"):
            FigureCanvasAgg(ax.figure)

        renderer = ax.figure.canvas.get_renderer()
        font_props = FontProperties(size=fontsize)
        temp_text = ax.text(
            0, 0, text, fontsize=fontsize, fontproperties=font_props, visible=False
        )
        bbox = temp_text.get_window_extent(renderer=renderer)
        temp_text.remove()
        return bbox

    def _calculate_text_width(self, ax: Axes, text: str, fontsize: int) -> float:
        """Calculate actual rendered text width in data coordinates.

        Args:
            ax: matplotlib Axes used for rendering.
            text: Text string to measure.
            fontsize: Font size in points.

        Returns:
            Text width in data coordinate units.
        """
        bbox = self._measure_text_bbox(ax, text, fontsize)

        xlim = ax.get_xlim()
        ax_bbox = ax.get_position()
        ax_width_inches = ax_bbox.width * ax.figure.get_figwidth()
        data_range = xlim[1] - xlim[0]
        pixels_per_data_unit = ax.figure.dpi * ax_width_inches / data_range

        if pixels_per_data_unit > 0:
            return bbox.width / pixels_per_data_unit
        return len(text) * self.style.fallback_char_width

    def _calculate_text_height(self, ax: Axes, text: str, fontsize: int) -> float:
        """Calculate actual rendered text height in data coordinates.

        Args:
            ax: matplotlib Axes used for rendering.
            text: Text string to measure.
            fontsize: Font size in points.

        Returns:
            Text height in data coordinate units.
        """
        bbox = self._measure_text_bbox(ax, text, fontsize)

        ylim = ax.get_ylim()
        ax_bbox = ax.get_position()
        ax_height_inches = ax_bbox.height * ax.figure.get_figheight()
        data_range = ylim[1] - ylim[0]
        pixels_per_data_unit = ax.figure.dpi * ax_height_inches / data_range

        if pixels_per_data_unit > 0:
            return bbox.height / pixels_per_data_unit
        return self.style.fallback_text_height

    def _create_figure(self, num_qubits: int) -> Figure:
        """Create matplotlib figure with appropriate size.

        Args:
            num_qubits: Number of qubits.

        Returns:
            matplotlib Figure.
        """
        if num_qubits == 0:
            fig = Figure(figsize=(4, 2))
            FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Empty circuit",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            fig._qm_ax = ax
            return fig

        width = self.layout.width
        block_ranges = self.layout.block_ranges

        # Use Figure directly to avoid pyplot auto-display in Jupyter
        fig = Figure(figsize=(6, 2))  # Temporary size, will be updated
        # Attach canvas for Jupyter display
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        # Dynamic vertical margins based on actual block border extents
        base_margin = 0.6  # minimum: gate half-height + comfortable padding
        clearance = 0.2  # extra breathing room beyond the border edge

        if self.layout.max_above:
            y_margin_top = max(self.layout.max_above.get(0, 0), base_margin) + clearance
        else:
            y_margin_top = base_margin

        if self.layout.max_below:
            y_margin_bottom = (
                max(self.layout.max_below.get(num_qubits - 1, 0), base_margin)
                + clearance
            )
        else:
            y_margin_bottom = base_margin

        y_top = self.qubit_y[0] + y_margin_top
        y_bottom = self.qubit_y[num_qubits - 1] - y_margin_bottom

        # Dynamic horizontal limits based on block border extents
        # Use first gate position and wire extension for symmetric left margin
        first_gate_x = self.layout.first_gate_x
        wire_ext = self.style.wire_extension
        # Left limit: wire extends wire_extension past first gate, plus label space
        first_gate_hw = self.layout.first_gate_half_width
        x_left = first_gate_x - first_gate_hw - wire_ext - 0.7
        x_right = max(width + 0.5, self.layout.actual_width + wire_ext + 0.2)

        if block_ranges:
            for br in block_ranges:
                block_name = br.get("name", "block")
                block_power = br.get("power", 1)
                mgw = br.get("max_gate_width", self.style.gate_width)
                border_left, border_right = compute_block_box_bounds(
                    self.style,
                    block_name,
                    br["start_x"],
                    br["end_x"],
                    br["depth"],
                    mgw,
                    block_power,
                )

                x_right = max(x_right, border_right + 0.3)
                x_left = min(x_left, border_left - 0.3)

        # Account for folded ForOperation boxes in figure sizing
        positions = self.layout.positions
        block_widths_dict = self.layout.block_widths
        for key, bw in block_widths_dict.items():
            if key in positions:
                box_left = positions[key] - bw / 2
                box_right = positions[key] + bw / 2
                x_left = min(x_left, box_left - 0.3)
                x_right = max(x_right, box_right + 0.3)

        # Account for folded block vertical extents in figure sizing
        for key, info in self.layout.folded_block_extents.items():
            if key not in positions:
                continue
            qubits = info["affected_qubits"]
            if not qubits:
                continue
            y_coords = [self.qubit_y[q] for q in qubits]
            min_qy, max_qy = min(y_coords), max(y_coords)
            y_center_fb = (min_qy + max_qy) / 2
            num_lines = info["num_text_lines"]
            min_text_height = (
                num_lines * self.style.line_height
                + 2 * self.style.folded_box_text_v_padding
            )
            qubit_span = max_qy - min_qy + self.style.gate_height
            box_height = max(qubit_span, min_text_height)
            box_top_fb = y_center_fb + box_height / 2
            box_bottom_fb = y_center_fb - box_height / 2
            y_top = max(y_top, box_top_fb + 0.3)
            y_bottom = min(y_bottom, box_bottom_fb - 0.3)

        # Extend x_right if output_names are present (for right-side labels)
        if self._output_names:
            max_label_len = max(len(name) for name in self._output_names)
            x_right += max_label_len * 0.12 + 0.5

        # Set axis limits
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(y_bottom, y_top)

        # Calculate figure size from axis limits
        x_range = x_right - x_left
        y_range = y_top - y_bottom
        fig_width = max(6, x_range * 1.0)
        fig_height = max(2, y_range * 0.8)

        # Update figure size
        fig.set_size_inches(fig_width, fig_height)
        ax.axis("off")
        ax.set_aspect("equal")

        # Store ax in figure for later access
        fig._qm_ax = ax

        return fig

    def _draw_wires(
        self,
        fig: Figure,
        num_qubits: int,
        qubit_map: dict[str, int],
    ) -> None:
        """Draw qubit wires and labels.

        Args:
            fig: matplotlib Figure.
            num_qubits: Number of qubits.
            qubit_map: Qubit to wire index mapping.
        """
        ax = fig._qm_ax
        # Compute wire start/end symmetrically from gate edges
        actual_width = self.layout.actual_width
        first_gate_x = self.layout.first_gate_x
        wire_ext = self.style.wire_extension
        first_gate_hw = self.layout.first_gate_half_width
        wire_start = first_gate_x - first_gate_hw - wire_ext
        default_wire_end = actual_width + wire_ext

        # Draw horizontal wires for each qubit
        for i in range(num_qubits):
            y = self.qubit_y[i]
            # Use measurement position if available, otherwise default
            wire_end = self.qubit_end_positions.get(i, default_wire_end)
            # Measurement-terminated wires use "butt" capstyle to prevent
            # the half-linewidth overhang past the measurement box edge.
            capstyle = "butt" if i in self.qubit_end_positions else "projecting"
            line = mlines.Line2D(
                [wire_start, wire_end],
                [y, y],
                color=self.style.wire_color,
                linewidth=1,
                zorder=PORDER_WIRE,
                solid_capstyle=capstyle,
            )
            ax.add_line(line)

            # Draw left qubit label
            label = self.qubit_names.get(i, f"q{i}")
            ax.text(
                wire_start - 0.2,
                y,
                label,
                ha="right",
                va="center",
                fontsize=self.style.font_size,
                zorder=PORDER_TEXT,
            )

        # Output labels on the right side are intentionally not drawn
        # to keep the circuit diagram cleaner

    def _draw_control_dot(self, ax: Axes, x: float, y: float) -> None:
        """Draw a control dot for controlled gates.

        Args:
            ax: matplotlib Axes.
            x: X coordinate of the dot center.
            y: Y coordinate of the dot center.
        """
        circle = mpatches.Circle(
            (x, y),
            radius=0.1,
            facecolor=self.style.wire_color,
            edgecolor=self.style.wire_color,
            zorder=PORDER_GATE,
        )
        ax.add_patch(circle)

    def _draw_target_x(self, ax: Axes, x: float, y: float) -> None:
        """Draw a target X (plus sign in circle) for CNOT.

        Args:
            ax: matplotlib Axes.
            x: X coordinate of the target center.
            y: Y coordinate of the target center.
        """
        # Outer circle
        circle = mpatches.Circle(
            (x, y),
            radius=0.25,
            facecolor="none",
            edgecolor=self.style.wire_color,
            linewidth=2,
            zorder=PORDER_GATE,
        )
        ax.add_patch(circle)

        # Plus sign
        line1 = mlines.Line2D(
            [x, x],
            [y - 0.25, y + 0.25],
            color=self.style.wire_color,
            linewidth=2,
            zorder=PORDER_GATE,
        )
        line2 = mlines.Line2D(
            [x - 0.25, x + 0.25],
            [y, y],
            color=self.style.wire_color,
            linewidth=2,
            zorder=PORDER_GATE,
        )
        ax.add_line(line1)
        ax.add_line(line2)

    def _draw_swap_x(self, ax: Axes, x: float, y: float) -> None:
        """Draw an X marker for SWAP gate.

        Args:
            ax: matplotlib Axes.
            x: X coordinate of the marker center.
            y: Y coordinate of the marker center.
        """
        size = 0.15
        line1 = mlines.Line2D(
            [x - size, x + size],
            [y - size, y + size],
            color=self.style.wire_color,
            linewidth=2,
            zorder=PORDER_GATE,
        )
        line2 = mlines.Line2D(
            [x - size, x + size],
            [y + size, y - size],
            color=self.style.wire_color,
            linewidth=2,
            zorder=PORDER_GATE,
        )
        ax.add_line(line1)
        ax.add_line(line2)

    def _draw_meter_symbol(
        self, ax: Axes, x_pos: float, y: float, width: float, height: float
    ) -> None:
        """Draw a meter symbol (half-circle arc + needle) for measurement.

        Args:
            ax: matplotlib Axes.
            x_pos: X center position.
            y: Y center position.
            width: Width of the containing box.
            height: Height of the containing box.
        """
        # Calculate arc radius based on box size
        arc_radius = min(width, height) * 0.25
        arc_center_y = y - arc_radius * 0.3

        # Draw half-circle arc (bottom half)
        arc = mpatches.Arc(
            (x_pos, arc_center_y),
            arc_radius * 2,
            arc_radius * 2,
            angle=0,
            theta1=0,
            theta2=180,
            color=self.style.measure_symbol_color,
            linewidth=1.5,
            zorder=PORDER_TEXT,
        )
        ax.add_patch(arc)

        # Draw needle (line pointing up-right from arc center, extending beyond arc)
        needle_angle = math.radians(50)
        needle_length = arc_radius * 1.3
        needle_end_x = x_pos + needle_length * math.cos(needle_angle)
        needle_end_y = arc_center_y + needle_length * math.sin(needle_angle)
        ax.plot(
            [x_pos, needle_end_x],
            [arc_center_y, needle_end_y],
            color=self.style.measure_symbol_color,
            linewidth=1.5,
            solid_capstyle="round",
            zorder=PORDER_TEXT,
        )

    def _draw_measurement_box(self, ax: Axes, x_pos: float, y: float) -> None:
        """Draw a single measurement box with meter symbol at the given position.

        Args:
            ax: matplotlib Axes.
            x_pos: X coordinate of the box center.
            y: Y coordinate of the box center.
        """
        width = self.style.gate_width
        height = self.style.gate_height

        rect = mpatches.FancyBboxPatch(
            (x_pos - width / 2, y - height / 2),
            width,
            height,
            boxstyle=mpatches.BoxStyle.Round(
                pad=0, rounding_size=self.style.gate_corner_radius
            ),
            facecolor=self.style.measure_face_color,
            edgecolor=self.style.wire_color,
            linewidth=1,
            zorder=PORDER_GATE,
        )
        ax.add_patch(rect)
        self._draw_meter_symbol(ax, x_pos, y, width, height)
