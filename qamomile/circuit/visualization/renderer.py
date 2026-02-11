"""Matplotlib-based circuit rendering.

This module provides MatplotlibRenderer, which handles all matplotlib
drawing operations for circuit visualization.
"""

from __future__ import annotations

import io
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from qamomile.circuit.ir.block_value import BlockValue
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.composite_gate import CompositeGateOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    GateOperation,
    GateOperationType,
    MeasureOperation,
    MeasureVectorOperation,
)
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.value import Value

from .analyzer import CircuitAnalyzer
from .style import CircuitStyle
from .types import (
    PORDER_GATE,
    PORDER_LINE,
    PORDER_TEXT,
    PORDER_WIRE,
    LayoutResult,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.transforms import Bbox

    from qamomile.circuit.ir.graph import Graph


class MatplotlibRenderer:
    """Renders circuit diagrams using matplotlib.

    Takes pre-computed layout coordinates and draws the circuit
    using matplotlib primitives.
    """

    def __init__(self, analyzer: CircuitAnalyzer, style: CircuitStyle):
        self.analyzer = analyzer
        self.style = style
        # These are set during render()
        self.layout: LayoutResult | None = None
        self.qubit_y: list[float] = []
        self.qubit_end_positions: dict[int, float] = {}
        self.qubit_names: dict[int, str] = {}
        self._inlined_op_keys: set[tuple] = set()

    def render(
        self,
        graph: Graph,
        qubit_map: dict[str, int],
        qubit_names: dict[int, str],
        num_qubits: int,
        layout: LayoutResult,
    ) -> Figure:
        """Render the circuit as a matplotlib Figure.

        Args:
            graph: IR computation graph.
            qubit_map: Mapping from logical_id to wire index.
            qubit_names: Mapping from wire index to display name.
            num_qubits: Total number of qubit wires.
            layout: Pre-computed layout result.

        Returns:
            matplotlib Figure.
        """
        self.layout = layout
        self.qubit_y = layout.qubit_y
        self.qubit_end_positions = layout.qubit_end_positions
        self.qubit_names = qubit_names
        self._inlined_op_keys = layout.inlined_op_keys

        fig = self._create_figure(num_qubits)
        self._draw_wires(fig, num_qubits, qubit_map)
        self._draw_operations(fig, graph, qubit_map)
        self._add_jupyter_display_support(fig)
        return fig

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
                depth = br["depth"]
                padding = self.analyzer._compute_border_padding(depth)
                mgw = br.get("max_gate_width", self.style.gate_width)
                gate_border_right = br["end_x"] + mgw / 2 + padding
                border_left = br["start_x"] - mgw / 2 - padding

                # Account for title width (same logic as _draw_inlined_block_border)
                block_name = br.get("name", "block")
                title_char_width = self.style.char_width_base
                label_left = border_left + self.style.label_horizontal_padding
                title_text_width = len(block_name) * title_char_width
                label_right = (
                    label_left + title_text_width + self.style.label_horizontal_padding
                )
                border_right = max(gate_border_right, label_right)

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

        # Extend x_right if output_names are present (for right-side labels)
        if self.analyzer.graph.output_names:
            max_label_len = max(len(name) for name in self.analyzer.graph.output_names)
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
            line = mlines.Line2D(
                [wire_start, wire_end],
                [y, y],
                color=self.style.wire_color,
                linewidth=1,
                zorder=PORDER_WIRE,
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

    def _draw_inline_block_ops(
        self,
        fig: Figure,
        block_value: BlockValue,
        actual_inputs: list[Value],
        op_key: tuple,
        logical_id_remap: dict[str, str],
        param_values: dict,
        draw_ops_fn: Callable[..., None],
        qubit_map: dict[str, int] | None = None,
    ) -> None:
        """Recursively draw inlined block operations.

        Shared logic for CallBlock, CompositeGate, and ControlledU inline drawing.

        Args:
            fig: matplotlib Figure.
            block_value: BlockValue containing the operations to draw.
            actual_inputs: Actual input Values passed to the block.
            op_key: Unique tuple key identifying this operation.
            logical_id_remap: Mapping from dummy logical_ids to actual logical_ids.
            param_values: Parameter values for resolving expressions.
            draw_ops_fn: Callback to recursively draw child operations.
            qubit_map: Qubit to wire index mapping for resolving array elements.
        """
        new_logical_id_remap, child_param_values = (
            self.analyzer._build_block_value_mappings(
                block_value,
                actual_inputs,
                logical_id_remap,
                param_values,
                qubit_map=qubit_map,
            )
        )
        self.analyzer._evaluate_loop_body_intermediates(
            block_value.operations, child_param_values
        )
        draw_ops_fn(
            block_value.operations,
            new_logical_id_remap,
            scope_path=op_key,
            param_values=child_param_values,
        )

    def _draw_call_block_or_inline(
        self,
        fig: Figure,
        op: CallBlockOperation,
        op_key: tuple,
        qubit_map: dict[str, int],
        positions: dict[tuple, float],
        block_widths: dict[tuple, float],
        logical_id_remap: dict[str, str],
        param_values: dict,
        draw_ops_fn: Callable[..., None],
    ) -> None:
        """Draw a CallBlockOperation: inline or as box.

        Args:
            fig: matplotlib Figure.
            op: CallBlockOperation to draw.
            op_key: Unique tuple key identifying this operation.
            qubit_map: Mapping from logical_id to qubit wire index.
            positions: Layout positions mapping op_key to x coordinate.
            block_widths: Layout widths mapping op_key to box width.
            logical_id_remap: Mapping from dummy logical_ids to actual logical_ids.
            param_values: Parameter values for resolving expressions.
            draw_ops_fn: Callback to recursively draw child operations.
        """
        if op_key in self._inlined_op_keys:
            block_value = op.operands[0]
            # The IR guarantees that operands[0] of CallBlockOperation is always a BlockValue,
            # so this assertion should always pass.
            # If it fails, there is a bug in the IR construction.
            assertion_message = (
                "[FOR DEVELOPER] CallBlockOperation.operands[0] must be a BlockValue. "
                "If this assertion fails, there is a bug in the IR construction."
            )
            assert isinstance(block_value, BlockValue), assertion_message
            self._draw_inline_block_ops(
                fig,
                block_value,
                op.operands[1:],
                op_key,
                logical_id_remap,
                param_values,
                draw_ops_fn,
                qubit_map=qubit_map,
            )
        elif op_key in positions:
            self._draw_call_block(
                fig,
                op,
                qubit_map,
                positions[op_key],
                block_widths.get(op_key),
                logical_id_remap=logical_id_remap,
                param_values=param_values,
            )

    def _draw_composite_or_inline(
        self,
        fig: Figure,
        op: CompositeGateOperation,
        op_key: tuple,
        qubit_map: dict[str, int],
        positions: dict[tuple, float],
        block_widths: dict[tuple, float],
        logical_id_remap: dict[str, str],
        param_values: dict,
        draw_ops_fn: Callable[..., None],
    ) -> None:
        """Draw a CompositeGateOperation: inline or as box.

        Args:
            fig: matplotlib Figure.
            op: CompositeGateOperation to draw.
            op_key: Unique tuple key identifying this operation.
            qubit_map: Mapping from logical_id to qubit wire index.
            positions: Layout positions mapping op_key to x coordinate.
            block_widths: Layout widths mapping op_key to box width.
            logical_id_remap: Mapping from dummy logical_ids to actual logical_ids.
            param_values: Parameter values for resolving expressions.
            draw_ops_fn: Callback to recursively draw child operations.
        """
        if op_key in self._inlined_op_keys:
            block_value = op.implementation
            # The IR guarantees that implementation returns a BlockValue when has_implementation is True,
            # so this assertion should always pass.
            # If it fails, there is a bug in the IR construction.
            assertion_message = (
                "[FOR DEVELOPER] CompositeGateOperation.implementation must return a BlockValue "
                "when has_implementation is True. "
                "If this assertion fails, there is a bug in the IR construction."
            )
            assert isinstance(block_value, BlockValue), assertion_message
            self._draw_inline_block_ops(
                fig,
                block_value,
                list(op.operands[1:]),
                op_key,
                logical_id_remap,
                param_values,
                draw_ops_fn,
                qubit_map=qubit_map,
            )
        elif op_key in positions:
            self._draw_composite_gate(
                fig,
                op,
                qubit_map,
                positions[op_key],
                block_widths.get(op_key),
                logical_id_remap,
                param_values,
            )

    def _draw_controlled_u_or_inline(
        self,
        fig: Figure,
        op: ControlledUOperation,
        op_key: tuple,
        qubit_map: dict[str, int],
        positions: dict[tuple, float],
        block_widths: dict[tuple, float],
        logical_id_remap: dict[str, str],
        param_values: dict,
        draw_ops_fn: Callable[..., None],
    ) -> None:
        """Draw a ControlledUOperation: inline or as box.

        Args:
            fig: matplotlib Figure.
            op: ControlledUOperation to draw.
            op_key: Unique tuple key identifying this operation.
            qubit_map: Mapping from logical_id to qubit wire index.
            positions: Layout positions mapping op_key to x coordinate.
            block_widths: Layout widths mapping op_key to box width.
            logical_id_remap: Mapping from dummy logical_ids to actual logical_ids.
            param_values: Parameter values for resolving expressions.
            draw_ops_fn: Callback to recursively draw child operations.
        """
        if op_key in self._inlined_op_keys:
            block_value = op.block
            # The IR guarantees that ControlledUOperation.block is always a BlockValue,
            # so this assertion should always pass.
            # If it fails, there is a bug in the IR construction.
            assertion_message = (
                "[FOR DEVELOPER] ControlledUOperation.block must be a BlockValue. "
                "If this assertion fails, there is a bug in the IR construction."
            )
            assert isinstance(block_value, BlockValue), assertion_message
            self._draw_inline_block_ops(
                fig,
                block_value,
                list(op.target_operands),
                op_key,
                logical_id_remap,
                param_values,
                draw_ops_fn,
                qubit_map=qubit_map,
            )
        elif op_key in positions:
            self._draw_controlled_u(
                fig,
                op,
                qubit_map,
                positions[op_key],
                block_widths.get(op_key),
                logical_id_remap,
                param_values,
            )

    def _draw_for_op(
        self,
        fig: Figure,
        op: ForOperation,
        op_key: tuple,
        qubit_map: dict[str, int],
        positions: dict[tuple, float],
        block_widths: dict[tuple, float],
        logical_id_remap: dict[str, str],
        param_values: dict,
        draw_ops_fn: Callable[..., None],
    ) -> None:
        """Handle drawing a ForOperation (fold or expand).

        Args:
            fig: matplotlib Figure.
            op: ForOperation to draw.
            op_key: Unique tuple key identifying this operation.
            qubit_map: Mapping from logical_id to qubit wire index.
            positions: Layout positions mapping op_key to x coordinate.
            block_widths: Layout widths mapping op_key to box width.
            logical_id_remap: Mapping from dummy logical_ids to actual logical_ids.
            param_values: Parameter values for resolving loop range.
            draw_ops_fn: Callback to recursively draw child operations.
        """
        start_val, stop_val_raw, step_val = self.analyzer._evaluate_loop_range(
            op, param_values
        )

        if self.analyzer._is_zero_iteration_loop(start_val, stop_val_raw, step_val):
            return

        if self.analyzer.fold_loops:
            if op_key in positions:
                op_width = block_widths.get(op_key)
                self._draw_for_loop_box(
                    fig,
                    op,
                    qubit_map,
                    positions[op_key],
                    op_width,
                    param_values=param_values,
                    logical_id_remap=logical_id_remap,
                )
        else:
            # Can't expand if stop is symbolic
            if stop_val_raw is None:
                if op_key in positions:
                    op_width = block_widths.get(op_key)
                    self._draw_for_loop_box(
                        fig,
                        op,
                        qubit_map,
                        positions[op_key],
                        op_width,
                        param_values=param_values,
                        logical_id_remap=logical_id_remap,
                    )
                return

            num_iterations = self.analyzer._compute_loop_iterations(
                start_val, stop_val_raw, step_val
            )
            for iteration in range(num_iterations):
                iter_value = start_val + iteration * step_val
                child_param_values = dict(param_values)
                child_param_values[f"_loop_{op.loop_var}"] = iter_value

                self.analyzer._evaluate_loop_body_intermediates(
                    op.operations, child_param_values
                )

                draw_ops_fn(
                    op.operations,
                    logical_id_remap,
                    scope_path=(*op_key, iteration),
                    param_values=child_param_values,
                )

    def _draw_operations(
        self,
        fig: Figure,
        graph: Graph,
        qubit_map: dict[str, int],
    ) -> None:
        """Draw all operations.

        Args:
            fig: matplotlib Figure.
            graph: Computation graph.
            qubit_map: Qubit to wire index mapping.
        """
        positions = self.layout.positions
        block_ranges = self.layout.block_ranges
        block_widths = self.layout.block_widths

        # First, draw block borders (behind gates)
        # Compute max_depth for inverted label offset
        max_depth = max((b["depth"] for b in block_ranges), default=0)

        # Pre-expand outer block boundaries to contain inner blocks
        # Sort by depth descending (process inner blocks first)
        for inner in sorted(block_ranges, key=lambda b: -b["depth"]):
            for outer in block_ranges:
                if outer["depth"] < inner["depth"]:
                    inner_qubits = set(inner["qubit_indices"])
                    outer_qubits = set(outer["qubit_indices"])
                    if inner_qubits.issubset(outer_qubits):
                        # Skip if no horizontal overlap (raw gate positions)
                        if (
                            inner["end_x"] < outer["start_x"]
                            or inner["start_x"] > outer["end_x"]
                        ):
                            continue
                        # Compare rendered box edges, not raw gate positions
                        inner_mgw = inner.get("max_gate_width", self.style.gate_width)
                        inner_box_left, inner_box_right = (
                            self.analyzer._compute_block_box_bounds(
                                inner.get("name", "block"),
                                inner["start_x"],
                                inner["end_x"],
                                inner["depth"],
                                inner_mgw,
                            )
                        )
                        outer_mgw = outer.get("max_gate_width", self.style.gate_width)
                        outer_padding = self.analyzer._compute_border_padding(
                            outer["depth"]
                        )
                        outer_box_left = (
                            outer["start_x"] - outer_mgw / 2 - outer_padding
                        )
                        _, outer_box_right = self.analyzer._compute_block_box_bounds(
                            outer.get("name", "block"),
                            outer["start_x"],
                            outer["end_x"],
                            outer["depth"],
                            outer_mgw,
                        )
                        margin = 0.1
                        if inner_box_left - margin < outer_box_left:
                            outer["start_x"] = (
                                inner_box_left - margin + outer_mgw / 2 + outer_padding
                            )
                        if inner_box_right + margin > outer_box_right:
                            outer["end_x"] = (
                                inner_box_right + margin - outer_mgw / 2 - outer_padding
                            )

        # Group blocks by topmost qubit, then calculate overlap_index based on horizontal overlap
        qubit_blocks = defaultdict(list)
        for block_info in block_ranges:
            top_qubit = min(block_info["qubit_indices"])
            qubit_blocks[top_qubit].append(block_info)

        # For each qubit group, calculate overlap_index for horizontally overlapping blocks
        for qubit, blocks in qubit_blocks.items():
            # Sort blocks by start_x
            blocks.sort(key=lambda b: b["start_x"])

            for i, block_info in enumerate(blocks):
                # Count how many previous blocks overlap horizontally with this block
                overlap_count = 0
                block_x_start = block_info["start_x"]
                block_x_end = block_info["end_x"]

                for j in range(i):
                    prev_block = blocks[j]
                    prev_x_start = prev_block["start_x"]
                    prev_x_end = prev_block["end_x"]

                    # Check if this block starts before the previous block ends (horizontal overlap)
                    # Add some margin (0.5) to consider blocks as overlapping if they're very close
                    if block_x_start < prev_x_end + 0.5:
                        # Skip parent-child (containment) relationships
                        if (
                            prev_x_start <= block_x_start and prev_x_end >= block_x_end
                        ) or (
                            block_x_start <= prev_x_start and block_x_end >= prev_x_end
                        ):
                            continue
                        overlap_count += 1

                # For ControlledU, draw border around target qubits only
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
                )

                # Draw control dots and vertical line for ControlledUOperation
                if ctrl_indices:
                    ax = fig._qm_ax
                    # Center control line at horizontal center of the border
                    mgw = block_info.get("max_gate_width", self.style.gate_width)
                    bl, br = self.analyzer._compute_block_box_bounds(
                        block_info["name"],
                        block_info["start_x"],
                        block_info["end_x"],
                        block_info["depth"],
                        mgw,
                    )
                    block_center_x = (bl + br) / 2
                    ctrl_y_list = [self.qubit_y[q] for q in ctrl_indices]

                    # Vertical line connects control dots to border edge
                    all_y_endpoints = ctrl_y_list + [box_bottom, box_top]
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

                    # Draw control dots
                    for ctrl_idx in ctrl_indices:
                        ctrl_y = self.qubit_y[ctrl_idx]
                        self._draw_control_dot(ax, block_center_x, ctrl_y)

        def draw_ops(
            ops: list[Operation],
            logical_id_remap: dict[str, str]
            | None = None,  # Dummy logical_id → Actual logical_id
            scope_path: tuple = (),
            param_values: dict[int, float | int | str] | None = None,
        ) -> None:
            """Recursively draw operations, dispatching each to the appropriate draw method.

            Args:
                ops: List of operations to draw.
                logical_id_remap: Mapping from formal parameter logical_ids to actual logical_ids.
                scope_path: Hierarchical tuple key identifying the current scope.
                param_values: Resolved parameter values for expression evaluation.
            """
            if logical_id_remap is None:
                logical_id_remap = {}
            if param_values is None:
                param_values = {}

            for op in ops:
                op_key = (*scope_path, id(op))

                if isinstance(op, (QInitOperation, CastOperation)):
                    continue

                if isinstance(op, ExpvalOp):
                    if op_key in positions:
                        self._draw_expval_op(
                            fig,
                            op,
                            qubit_map,
                            positions[op_key],
                            block_widths.get(op_key),
                            logical_id_remap,
                            param_values,
                        )
                    continue

                if isinstance(op, WhileOperation):
                    self._draw_while_op(
                        fig,
                        op,
                        op_key,
                        qubit_map,
                        positions,
                        block_widths,
                        logical_id_remap,
                        param_values,
                        draw_ops,
                    )
                    continue

                if isinstance(op, IfOperation):
                    self._draw_if_op(
                        fig,
                        op,
                        op_key,
                        qubit_map,
                        positions,
                        block_widths,
                        logical_id_remap,
                        param_values,
                        draw_ops,
                    )
                    continue

                if isinstance(op, ForItemsOperation):
                    self._draw_for_items_op(
                        fig,
                        op,
                        op_key,
                        qubit_map,
                        positions,
                        block_widths,
                        logical_id_remap,
                        param_values,
                        draw_ops,
                    )
                    continue

                if isinstance(op, ForOperation):
                    self._draw_for_op(
                        fig,
                        op,
                        op_key,
                        qubit_map,
                        positions,
                        block_widths,
                        logical_id_remap,
                        param_values,
                        draw_ops,
                    )
                    continue

                if isinstance(op, GateOperation):
                    if op_key in positions:
                        self._draw_gate(
                            fig,
                            op,
                            qubit_map,
                            positions[op_key],
                            logical_id_remap,
                            param_values,
                        )
                    continue

                if isinstance(op, CallBlockOperation):
                    self._draw_call_block_or_inline(
                        fig,
                        op,
                        op_key,
                        qubit_map,
                        positions,
                        block_widths,
                        logical_id_remap,
                        param_values,
                        draw_ops,
                    )
                    continue

                if isinstance(op, CompositeGateOperation):
                    self._draw_composite_or_inline(
                        fig,
                        op,
                        op_key,
                        qubit_map,
                        positions,
                        block_widths,
                        logical_id_remap,
                        param_values,
                        draw_ops,
                    )
                    continue

                if isinstance(op, ControlledUOperation):
                    self._draw_controlled_u_or_inline(
                        fig,
                        op,
                        op_key,
                        qubit_map,
                        positions,
                        block_widths,
                        logical_id_remap,
                        param_values,
                        draw_ops,
                    )
                    continue

                if isinstance(op, MeasureOperation):
                    if op_key in positions:
                        self._draw_measurement(
                            fig, op, qubit_map, positions[op_key], logical_id_remap
                        )
                    continue

                if isinstance(op, MeasureVectorOperation):
                    if op_key in positions:
                        self._draw_measurement_vector(
                            fig, op, qubit_map, positions[op_key], logical_id_remap
                        )

        draw_ops(graph.operations)

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
    ) -> tuple[float, float]:
        """Draw dashed border around inlined block operations.

        Args:
            ax: matplotlib Axes.
            name: Block name for label.
            start_x: Start x coordinate (column).
            end_x: End x coordinate (column).
            qubit_indices: List of qubit indices affected by the block.
            depth: Nesting depth for nested blocks.
            max_gate_width: Maximum gate width in the block (for parametric gates).
            max_depth: Maximum nesting depth across all blocks (for label offset).
            overlap_index: Index for blocks at the same horizontal position (to avoid label overlap).

        Returns:
            Tuple of (box_bottom, box_top) y-coordinates of the drawn border.
        """
        # Use provided max_gate_width or default
        if max_gate_width is None:
            max_gate_width = self.style.gate_width

        # Convert qubit indices to y coordinates using variable spacing
        y_coords = [self.qubit_y[q] for q in qubit_indices]
        min_y = min(y_coords)
        max_y = max(y_coords)

        # Calculate text height for label spacing
        text_height = self._calculate_text_height(ax, name, self.style.subfont_size)

        # Depth-based padding to separate nested block boundaries
        padding = self.analyzer._compute_border_padding(depth)

        lp = self.style.label_padding
        label_height = text_height + 2 * lp

        # Vertical offset for labels based on depth and overlap index
        depth_label_offset = (max_depth - depth) * (
            label_height + self.style.label_step_gap
        )
        overlap_label_offset = overlap_index * (
            label_height + self.style.overlap_step_gap
        )
        label_vertical_offset = depth_label_offset + overlap_label_offset

        # Calculate box boundaries using shared helper (right-only label expansion)
        box_left, box_right = self.analyzer._compute_block_box_bounds(
            name, start_x, end_x, depth, max_gate_width
        )
        label_left = box_left + self.style.label_horizontal_padding

        # Calculate vertical box boundaries (always include label height)
        box_bottom = min_y - self.style.gate_height / 2 - padding
        box_top = (
            max_y
            + self.style.gate_height / 2
            + padding
            + label_height
            + label_vertical_offset
        )

        # Draw border rectangle
        # Use depth-adjusted zorder so inner (higher depth) blocks appear on top
        block_zorder = PORDER_WIRE + 0.5 + depth * 0.1
        if is_controlled:
            # ControlledU: white fill + solid border
            rect = mpatches.Rectangle(
                (box_left, box_bottom),
                box_right - box_left,
                box_top - box_bottom,
                facecolor=self.style.background_color,
                edgecolor=self.style.block_border_color,
                linewidth=1.5,
                linestyle="-",
                zorder=block_zorder,
            )
            ax.add_patch(rect)
            # Redraw wires inside the border at higher z-order
            wire_zorder = block_zorder + 0.05
            for q in qubit_indices:
                wire_y = self.qubit_y[q]
                ax.add_line(
                    mlines.Line2D(
                        [box_left, box_right],
                        [wire_y, wire_y],
                        color=self.style.wire_color,
                        linewidth=1.0,
                        zorder=wire_zorder,
                    )
                )
        else:
            # Regular block: transparent fill + dashed border
            rect = mpatches.Rectangle(
                (box_left, box_bottom),
                box_right - box_left,
                box_top - box_bottom,
                facecolor="none",
                edgecolor=self.style.block_border_color,
                linewidth=1.5,
                linestyle="--",
                zorder=block_zorder,
            )
            ax.add_patch(rect)

        # Draw block name label (above the box, with vertical offset for nested blocks)
        ax.text(
            label_left,
            box_top - lp,
            name,
            ha="left",
            va="top",
            fontsize=self.style.subfont_size,
            color=self.style.block_border_color,
            zorder=PORDER_TEXT,
        )

        return box_bottom, box_top

    def _draw_gate(
        self,
        fig: Figure,
        op: GateOperation,
        qubit_map: dict[str, int],
        x_pos: int,
        logical_id_remap: dict[str, str] | None = None,
        param_values: dict[int, float | int | str] | None = None,
    ) -> None:
        """Draw a gate operation.

        Args:
            fig: matplotlib Figure.
            op: Gate operation.
            qubit_map: Qubit to wire index mapping.
            x_pos: X position for the gate.
            logical_id_remap: Mapping from dummy logical_ids to actual logical_ids.
            param_values: Mapping from logical_ids to resolved parameter values.
        """
        if logical_id_remap is None:
            logical_id_remap = {}
        if param_values is None:
            param_values = {}

        ax = fig._qm_ax

        # Get affected qubits (map dummy IDs to actual IDs)
        qubit_indices = []
        for operand in op.operands:
            idx = self.analyzer._resolve_operand_to_qubit_index(
                operand, qubit_map, logical_id_remap, param_values
            )
            if idx is not None:
                qubit_indices.append(idx)

        if not qubit_indices:
            return

        # Convert to y coordinates using variable spacing
        y_coords = [self.qubit_y[q] for q in qubit_indices]

        if len(qubit_indices) == 1:
            # Single-qubit gate
            self._draw_single_qubit_gate(ax, op, x_pos, y_coords[0], param_values)
        else:
            # Multi-qubit gate
            self._draw_multi_qubit_gate(ax, op, x_pos, y_coords, param_values)

    def _compute_gate_draw_width(
        self,
        ax: Axes,
        op: GateOperation,
        label: str,
        has_param: bool,
        param_values: dict | None,
    ) -> float:
        """Compute the draw width for a gate, handling parametric gates.

        Args:
            ax: matplotlib Axes used for text measurement.
            op: GateOperation whose width to compute.
            label: Pre-computed gate label string.
            has_param: Whether the gate has parameters (affects width).
            param_values: Parameter values for resolving gate labels.

        Returns:
            Gate draw width in data coordinate units.
        """
        if has_param:
            text_width = self._calculate_text_width(ax, label, self.style.font_size)
            calculated_width = text_width + 2 * self.style.text_padding
            estimated_width = self.analyzer._estimate_gate_width(op, param_values)
            return max(estimated_width, calculated_width)
        return self.style.gate_width

    def _draw_single_qubit_gate(
        self,
        ax: Axes,
        op: GateOperation,
        x: float,
        y: float,
        param_values: dict | None = None,
    ) -> None:
        """Draw a single-qubit gate box.

        Args:
            ax: matplotlib Axes.
            op: Gate operation.
            x: X coordinate.
            y: Y coordinate.
            param_values: Mapping from logical_ids to resolved parameter values.
        """
        label, has_param = self.analyzer._get_gate_label(op, param_values)
        font_size = self.style.font_size
        width = self._compute_gate_draw_width(ax, op, label, has_param, param_values)

        height = self.style.gate_height

        # Draw gate box
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

        # Draw label
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            color=self.style.gate_text_color,
            fontsize=font_size,
            zorder=PORDER_TEXT,
        )

    def _draw_multi_qubit_gate(
        self,
        ax: Axes,
        op: GateOperation,
        x: float,
        y_coords: list[float],
        param_values: dict | None = None,
    ) -> None:
        """Draw a multi-qubit gate.

        Args:
            ax: matplotlib Axes.
            op: Gate operation.
            x: X coordinate.
            y_coords: Y coordinates of involved qubits.
            param_values: Mapping from logical_ids to resolved parameter values.
        """
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

        # Draw control and target markers based on gate type
        label, has_param = self.analyzer._get_gate_label(op, param_values)
        font_size = self.style.font_size
        width = self._compute_gate_draw_width(ax, op, label, has_param, param_values)

        if op.gate_type == GateOperationType.CX:
            # CNOT: control dot on first qubit, target circle on second
            self._draw_control_dot(ax, x, y_coords[0])
            self._draw_target_x(ax, x, y_coords[1])
        elif op.gate_type == GateOperationType.CZ:
            # CZ: control dots on both qubits
            for y in y_coords:
                self._draw_control_dot(ax, x, y)
        elif op.gate_type == GateOperationType.SWAP:
            # SWAP: X markers on both qubits
            for y in y_coords:
                self._draw_swap_x(ax, x, y)
        else:
            # Generic multi-qubit gate: draw a single unified box spanning all qubits
            # Calculate box dimensions to span all qubits
            height = max_y - min_y + self.style.gate_height
            y_center = (min_y + max_y) / 2

            # Draw unified gate box
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

            # Draw label centered in the box
            ax.text(
                x,
                y_center,
                label,
                ha="center",
                va="center",
                color=self.style.gate_text_color,
                fontsize=font_size,
                zorder=PORDER_TEXT,
            )

            # Draw semicircle connection dots at left and right edges of the box
            half_w = width / 2
            dot_radius = 0.05
            for y in y_coords:
                # Left semicircle (facing left)
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
                # Right semicircle (facing right)
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

    def _draw_call_block(
        self,
        fig: Figure,
        op: CallBlockOperation,
        qubit_map: dict[str, int],
        x_pos: float,
        block_width: float | None = None,
        logical_id_remap: dict[str, str] | None = None,
        param_values: dict | None = None,
    ) -> None:
        """Draw a CallBlockOperation as a box.

        Args:
            fig: matplotlib Figure.
            op: CallBlock operation.
            qubit_map: Qubit to wire index mapping.
            x_pos: X position.
            block_width: Pre-calculated block width from layout phase.
            logical_id_remap: Mapping from dummy logical_ids to actual logical_ids.
            param_values: Parameter values for resolving loop variables.
        """
        if logical_id_remap is None:
            logical_id_remap = {}
        if param_values is None:
            param_values = {}

        ax = fig._qm_ax

        # Get affected qubits (skip first operand which is BlockValue)
        qubit_indices = []
        for operand in op.operands[1:]:  # Skip BlockValue
            qubit_indices.extend(
                self.analyzer._resolve_operand_to_qubit_indices(
                    operand, qubit_map, logical_id_remap, param_values
                )
            )

        if not qubit_indices:
            return

        # Convert to y coordinates using variable spacing
        y_coords = [self.qubit_y[q] for q in qubit_indices]
        min_y = min(y_coords)
        max_y = max(y_coords)

        # Get block label with parameters
        label = self.analyzer._get_block_label(op, qubit_map)

        # Use pre-calculated width from layout phase if available
        # This ensures consistency between layout and rendering
        if block_width is not None:
            width = block_width
        else:
            width = self.analyzer._estimate_label_box_width(label)

        # Draw box spanning all qubits
        height = max_y - min_y + self.style.gate_height
        y_center = (min_y + max_y) / 2

        rect = mpatches.FancyBboxPatch(
            (x_pos - width / 2, min_y - self.style.gate_height / 2),
            width,
            height,
            boxstyle=mpatches.BoxStyle.Round(
                pad=0, rounding_size=self.style.gate_corner_radius
            ),
            facecolor=self.style.block_face_color,
            edgecolor=self.style.block_box_edge_color,
            linewidth=1.5,
            linestyle="--",  # Dashed line for blocks
            zorder=PORDER_GATE,
        )
        ax.add_patch(rect)

        # Draw label
        ax.text(
            x_pos,
            y_center,
            label,
            ha="center",
            va="center",
            color=self.style.block_text_color,
            fontsize=self.style.font_size,
            zorder=PORDER_TEXT,
        )

    def _draw_composite_gate(
        self,
        fig: Figure,
        op: CompositeGateOperation,
        qubit_map: dict[str, int],
        x_pos: float,
        block_width: float | None = None,
        logical_id_remap: dict[str, str] | None = None,
        param_values: dict | None = None,
    ) -> None:
        """Draw a CompositeGateOperation as a labeled box.

        Args:
            fig: matplotlib Figure.
            op: CompositeGateOperation to draw.
            qubit_map: Mapping from logical_id to qubit wire index.
            x_pos: X position for the box center.
            block_width: Pre-calculated box width, or None for default.
            logical_id_remap: Mapping from dummy logical_ids to actual logical_ids.
            param_values: Parameter values for resolving loop variables.
        """
        if logical_id_remap is None:
            logical_id_remap = {}
        if param_values is None:
            param_values = {}
        ax = fig._qm_ax

        # Collect qubit indices from control + target qubits
        qubit_indices = []
        for qval in op.control_qubits + op.target_qubits:
            idx = self.analyzer._resolve_operand_to_qubit_index(
                qval, qubit_map, logical_id_remap, param_values
            )
            if idx is not None:
                qubit_indices.append(idx)

        if not qubit_indices:
            return

        y_coords = [self.qubit_y[q] for q in qubit_indices]
        min_y = min(y_coords)
        max_y = max(y_coords)

        label = op.name.upper()
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
            facecolor=self.style.gate_face_color,
            edgecolor=self.style.wire_color,
            linewidth=1.5,
            zorder=PORDER_GATE,
        )
        ax.add_patch(rect)

        ax.text(
            x_pos,
            y_center,
            label,
            ha="center",
            va="center",
            color=self.style.gate_text_color,
            fontsize=self.style.font_size,
            zorder=PORDER_TEXT,
        )

    def _draw_controlled_u(
        self,
        fig: Figure,
        op: ControlledUOperation,
        qubit_map: dict[str, int],
        x_pos: float,
        block_width: float | None = None,
        logical_id_remap: dict[str, str] | None = None,
        param_values: dict | None = None,
    ) -> None:
        """Draw a ControlledUOperation with control dots and target box.

        Args:
            fig: matplotlib Figure.
            op: ControlledUOperation to draw.
            qubit_map: Mapping from logical_id to qubit wire index.
            x_pos: X position for the gate center.
            block_width: Pre-calculated box width, or None for default.
            logical_id_remap: Mapping from dummy logical_ids to actual logical_ids.
            param_values: Parameter values for resolving loop variables.
        """
        if logical_id_remap is None:
            logical_id_remap = {}
        if param_values is None:
            param_values = {}
        ax = fig._qm_ax

        # Resolve control qubit y-coordinates
        control_y = []
        for qval in op.control_operands:
            idx = self.analyzer._resolve_operand_to_qubit_index(
                qval, qubit_map, logical_id_remap, param_values
            )
            if idx is not None:
                control_y.append(self.qubit_y[idx])

        # Resolve target qubit y-coordinates
        target_y = []
        for qval in op.target_operands:
            idx = self.analyzer._resolve_operand_to_qubit_index(
                qval, qubit_map, logical_id_remap, param_values
            )
            if idx is not None:
                target_y.append(self.qubit_y[idx])

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
            block_val = op.block
            u_name = getattr(block_val, "name", "U") or "U"
            power_val = (
                op.power
                if isinstance(op.power, int)
                else getattr(op.power, "init_value", op.power)
            )
            label = (
                f"{u_name}^{power_val}"
                if isinstance(power_val, int) and power_val > 1
                else u_name
            )
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
                label,
                ha="center",
                va="center",
                color=self.style.gate_text_color,
                fontsize=self.style.font_size,
                zorder=PORDER_TEXT,
            )

    def _draw_meter_symbol(
        self, ax: Axes, x_pos: float, y: float, width: float, height: float
    ) -> None:
        """Draw a meter symbol (half-circle arc + arrow) for measurement.

        Args:
            ax: matplotlib Axes.
            x_pos: X center position.
            y: Y center position.
            width: Width of the containing box.
            height: Height of the containing box.
        """
        # Calculate arc radius based on box size
        arc_radius = min(width, height) * 0.25

        # Draw half-circle arc (bottom half)
        arc = mpatches.Arc(
            (x_pos, y - arc_radius * 0.3),
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

        # Draw arrow (needle pointing up-right from arc center)
        ax.annotate(
            "",
            xy=(x_pos + arc_radius * 1.0, y + arc_radius * 0.8),
            xytext=(x_pos, y - arc_radius * 0.3),
            arrowprops=dict(
                arrowstyle="->",
                color=self.style.measure_symbol_color,
                lw=1.5,
            ),
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

    def _draw_measurement(
        self,
        fig: Figure,
        op: MeasureOperation,
        qubit_map: dict[str, int],
        x_pos: float,
        logical_id_remap: dict[str, str] | None = None,
    ) -> None:
        """Draw a MeasureOperation as an M box with meter symbol.

        Args:
            fig: matplotlib Figure.
            op: MeasureOperation to draw.
            qubit_map: Mapping from logical_id to qubit wire index.
            x_pos: X position for the measurement box center.
            logical_id_remap: Mapping from dummy logical_ids to actual logical_ids.
        """
        if not op.operands:
            return

        qubit_idx = self.analyzer._resolve_operand_to_qubit_index(
            op.operands[0], qubit_map, logical_id_remap or {}
        )
        if qubit_idx is not None:
            self._draw_measurement_box(fig._qm_ax, x_pos, self.qubit_y[qubit_idx])

    def _draw_measurement_vector(
        self,
        fig: Figure,
        op: MeasureVectorOperation,
        qubit_map: dict[str, int],
        x_pos: float,
        logical_id_remap: dict[str, str] | None = None,
    ) -> None:
        """Draw a MeasureVectorOperation as M boxes on all measured qubits.

        Args:
            fig: matplotlib Figure.
            op: MeasureVectorOperation to draw.
            qubit_map: Mapping from logical_id to qubit wire index.
            x_pos: X position for the measurement box centers.
            logical_id_remap: Mapping from dummy logical_ids to actual logical_ids.
        """
        if not op.operands:
            return

        qubit_indices = self.analyzer._resolve_operand_to_qubit_indices(
            op.operands[0], qubit_map, logical_id_remap or {}
        )
        ax = fig._qm_ax
        for qubit_idx in qubit_indices:
            self._draw_measurement_box(ax, x_pos, self.qubit_y[qubit_idx])

    def _draw_for_loop_box(
        self,
        fig: Figure,
        op: ForOperation,
        qubit_map: dict[str, int],
        x_pos: float,
        block_width: float | None = None,
        param_values: dict | None = None,
        logical_id_remap: dict[str, str] | None = None,
    ) -> None:
        """Draw a ForOperation as a box (folded mode).

        Args:
            fig: matplotlib Figure.
            op: ForOperation.
            qubit_map: Qubit to wire index mapping.
            x_pos: X position.
            block_width: Pre-calculated block width from layout phase.
            param_values: Parameter values for resolving symbolic expressions.
            logical_id_remap: Mapping from dummy logical_ids to actual logical_ids.
        """
        if param_values is None:
            param_values = {}

        ax = fig._qm_ax

        # Get affected qubits
        affected_qubits = self.analyzer._analyze_loop_affected_qubits(
            op, qubit_map, logical_id_remap, param_values
        )

        if not affected_qubits:
            return

        # Convert to y coordinates using variable spacing
        y_coords = [self.qubit_y[q] for q in affected_qubits]
        min_y = min(y_coords)
        max_y = max(y_coords)

        # Get loop range for label
        range_str = self.analyzer._format_range_str(op, set(), param_values)
        label = f"for {op.loop_var} in {range_str}"

        # Collect operation expressions from loop body
        expressions = []
        for body_op in op.operations:
            expr = self.analyzer._format_operation_as_expression(
                body_op,
                {op.loop_var},
                param_values=param_values,
                body_operations=op.operations,
            )
            if expr:
                expressions.append(expr)

        # Use pre-calculated width or default
        if block_width is not None:
            width = block_width
        else:
            width = self.style.folded_loop_width

        # Draw box spanning all qubits, ensuring enough room for text
        num_text_lines = 1 + len(expressions)  # header + body lines
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
            facecolor=self.style.for_loop_face_color,
            edgecolor=self.style.for_loop_edge_color,
            linewidth=1.5,
            linestyle="-",
            zorder=PORDER_GATE,
        )
        ax.add_patch(rect)

        # Draw header at top of box
        header_y = box_top - self.style.label_vertical_offset - 0.15
        header_text = ax.text(
            x_pos,
            header_y,
            label,
            ha="center",
            va="top",
            color=self.style.for_loop_text_color,
            fontsize=self.style.subfont_size,
            fontweight="bold",
            zorder=PORDER_TEXT,
            clip_on=True,
        )
        header_text.set_clip_path(rect)

        # Draw operation text centered in remaining space
        if expressions:
            body_text = "\n".join(expressions)
            body_center_y = (box_bottom + header_y) / 2
            body_text_obj = ax.text(
                x_pos,
                body_center_y,
                body_text,
                ha="center",
                va="center",
                color=self.style.for_loop_text_color,
                fontsize=self.style.subfont_size,
                family="monospace",
                zorder=PORDER_TEXT,
                clip_on=True,
            )
            body_text_obj.set_clip_path(rect)

    def _draw_expval_op(
        self,
        fig: Figure,
        op: ExpvalOp,
        qubit_map: dict[str, int],
        x_pos: float,
        block_width: float | None = None,
        logical_id_remap: dict[str, str] | None = None,
        param_values: dict | None = None,
    ) -> None:
        """Draw an ExpvalOp as a box spanning affected qubits."""
        if logical_id_remap is None:
            logical_id_remap = {}
        if param_values is None:
            param_values = {}
        ax = fig._qm_ax

        # Resolve qubit indices from first operand (qubit register)
        qubit_indices = self.analyzer._resolve_operand_to_qubit_indices(
            op.operands[0], qubit_map, logical_id_remap, param_values
        )
        if not qubit_indices:
            return

        y_coords = [self.qubit_y[q] for q in qubit_indices]
        min_y = min(y_coords)
        max_y = max(y_coords)

        label = "<H>"
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
            label,
            ha="center",
            va="center",
            color=self.style.expval_text_color,
            fontsize=self.style.font_size,
            zorder=PORDER_TEXT,
        )

    def _draw_while_op(
        self,
        fig: Figure,
        op: WhileOperation,
        op_key: tuple,
        qubit_map: dict[str, int],
        positions: dict[tuple, float],
        block_widths: dict[tuple, float],
        logical_id_remap: dict[str, str],
        param_values: dict,
        draw_ops_fn: Callable[..., None],
    ) -> None:
        """Draw a WhileOperation (always folded box)."""
        if op_key in positions:
            self._draw_while_loop_box(
                fig,
                op,
                qubit_map,
                positions[op_key],
                block_widths.get(op_key),
                param_values=param_values,
                logical_id_remap=logical_id_remap,
            )

    def _draw_while_loop_box(
        self,
        fig: Figure,
        op: WhileOperation,
        qubit_map: dict[str, int],
        x_pos: float,
        block_width: float | None = None,
        param_values: dict | None = None,
        logical_id_remap: dict[str, str] | None = None,
    ) -> None:
        """Draw a WhileOperation as a folded box."""
        if param_values is None:
            param_values = {}
        ax = fig._qm_ax

        affected_qubits = self.analyzer._analyze_loop_affected_qubits(
            op, qubit_map, logical_id_remap, param_values
        )
        if not affected_qubits:
            return

        y_coords = [self.qubit_y[q] for q in affected_qubits]
        min_y = min(y_coords)
        max_y = max(y_coords)

        # Build label
        label = "while cond:"
        expressions = []
        for body_op in op.operations:
            expr = self.analyzer._format_operation_as_expression(
                body_op,
                set(),
                body_operations=op.operations,
            )
            if expr:
                expressions.append(expr)
        if expressions:
            expr_lines = expressions[:3]
            if len(expressions) > 3:
                expr_lines.append("...")
            label += "\n" + "\n".join(expr_lines)

        width = block_width if block_width is not None else self.style.folded_loop_width
        num_text_lines = label.count("\n") + 1
        min_text_height = (
            num_text_lines * self.style.line_height
            + 2 * self.style.folded_box_text_v_padding
        )
        qubit_span_height = max_y - min_y + self.style.gate_height
        height = max(qubit_span_height, min_text_height)
        y_center = (min_y + max_y) / 2

        rect = mpatches.FancyBboxPatch(
            (x_pos - width / 2, y_center - height / 2),
            width,
            height,
            boxstyle=mpatches.BoxStyle.Round(
                pad=0, rounding_size=self.style.gate_corner_radius
            ),
            facecolor=self.style.while_loop_face_color,
            edgecolor=self.style.while_loop_edge_color,
            linewidth=1.5,
            linestyle="--",
            zorder=PORDER_GATE,
        )
        ax.add_patch(rect)

        ax.text(
            x_pos,
            y_center,
            label,
            ha="center",
            va="center",
            color=self.style.while_loop_text_color,
            fontsize=self.style.subfont_size,
            zorder=PORDER_TEXT,
        )

    def _draw_if_op(
        self,
        fig: Figure,
        op: IfOperation,
        op_key: tuple,
        qubit_map: dict[str, int],
        positions: dict[tuple, float],
        block_widths: dict[tuple, float],
        logical_id_remap: dict[str, str],
        param_values: dict,
        draw_ops_fn: Callable[..., None],
    ) -> None:
        """Draw an IfOperation (fold or unfold)."""
        if self.analyzer.fold_loops:
            # Folded: draw as box
            if op_key in positions:
                self._draw_if_box(
                    fig,
                    op,
                    qubit_map,
                    positions[op_key],
                    block_widths.get(op_key),
                    param_values=param_values,
                    logical_id_remap=logical_id_remap,
                )
        else:
            # Unfolded: draw both branches
            # True branch
            draw_ops_fn(
                op.true_operations,
                logical_id_remap,
                scope_path=(*op_key, "true"),
                param_values=param_values,
            )
            # False branch
            if op.false_operations:
                draw_ops_fn(
                    op.false_operations,
                    logical_id_remap,
                    scope_path=(*op_key, "false"),
                    param_values=param_values,
                )

    def _draw_if_box(
        self,
        fig: Figure,
        op: IfOperation,
        qubit_map: dict[str, int],
        x_pos: float,
        block_width: float | None = None,
        param_values: dict | None = None,
        logical_id_remap: dict[str, str] | None = None,
    ) -> None:
        """Draw an IfOperation as a folded box."""
        if param_values is None:
            param_values = {}
        ax = fig._qm_ax

        # Get affected qubits
        affected_qubits = self.analyzer._collect_if_affected_qubits(
            op, qubit_map, logical_id_remap, param_values
        )

        if not affected_qubits:
            return

        y_coords = [self.qubit_y[q] for q in affected_qubits]
        min_y = min(y_coords)
        max_y = max(y_coords)

        # Build condition label
        cond = op.condition
        cond_name = getattr(cond, "name", None) or "cond"
        label = f"if {cond_name}:"

        # Add body summary
        expressions = []
        for body_op in op.true_operations:
            expr = self.analyzer._format_operation_as_expression(
                body_op,
                set(),
                body_operations=op.true_operations,
            )
            if expr:
                expressions.append(expr)
        if expressions:
            expr_lines = expressions[:2]
            if len(expressions) > 2:
                expr_lines.append("...")
            label += "\n" + "\n".join(expr_lines)
        if op.false_operations:
            label += "\nelse: ..."

        width = block_width if block_width is not None else self.style.folded_loop_width
        num_text_lines = label.count("\n") + 1
        min_text_height = (
            num_text_lines * self.style.line_height
            + 2 * self.style.folded_box_text_v_padding
        )
        qubit_span_height = max_y - min_y + self.style.gate_height
        height = max(qubit_span_height, min_text_height)
        y_center = (min_y + max_y) / 2

        rect = mpatches.FancyBboxPatch(
            (x_pos - width / 2, y_center - height / 2),
            width,
            height,
            boxstyle=mpatches.BoxStyle.Round(
                pad=0, rounding_size=self.style.gate_corner_radius
            ),
            facecolor=self.style.if_face_color,
            edgecolor=self.style.if_edge_color,
            linewidth=1.5,
            linestyle="-.",
            zorder=PORDER_GATE,
        )
        ax.add_patch(rect)

        ax.text(
            x_pos,
            y_center,
            label,
            ha="center",
            va="center",
            color=self.style.if_text_color,
            fontsize=self.style.subfont_size,
            zorder=PORDER_TEXT,
        )

    def _draw_for_items_op(
        self,
        fig: Figure,
        op: ForItemsOperation,
        op_key: tuple,
        qubit_map: dict[str, int],
        positions: dict[tuple, float],
        block_widths: dict[tuple, float],
        logical_id_remap: dict[str, str],
        param_values: dict,
        draw_ops_fn: Callable[..., None],
    ) -> None:
        """Draw a ForItemsOperation (fold or expand)."""
        dict_value = op.operands[0] if op.operands else None
        materialized = (
            self.analyzer._materialize_dict_entries(dict_value) if dict_value else None
        )

        if self.analyzer.fold_loops or materialized is None:
            # Folded: draw as box
            if op_key in positions:
                self._draw_for_items_box(
                    fig,
                    op,
                    qubit_map,
                    positions[op_key],
                    block_widths.get(op_key),
                    param_values=param_values,
                    logical_id_remap=logical_id_remap,
                )
        else:
            # Unfolded: iterate materialized entries
            for iteration, (entry_key, entry_value) in enumerate(materialized):
                child_param_values = dict(param_values)
                # Set key variables — handle both IR Values and raw Python
                if hasattr(entry_key, "elements"):
                    for key_var, elem in zip(op.key_vars, entry_key.elements):
                        val = elem.get_const() if hasattr(elem, "get_const") else None
                        if val is not None:
                            child_param_values[f"_loop_{key_var}"] = val
                elif isinstance(entry_key, tuple):
                    for key_var, elem in zip(op.key_vars, entry_key):
                        child_param_values[f"_loop_{key_var}"] = elem
                else:
                    if op.key_vars:
                        if hasattr(entry_key, "get_const"):
                            val = entry_key.get_const()
                        else:
                            val = entry_key
                        if val is not None:
                            child_param_values[f"_loop_{op.key_vars[0]}"] = val
                # Set value variable
                if hasattr(entry_value, "get_const"):
                    val = entry_value.get_const()
                else:
                    val = entry_value
                if val is not None:
                    child_param_values[f"_loop_{op.value_var}"] = val

                self.analyzer._evaluate_loop_body_intermediates(
                    op.operations, child_param_values
                )

                draw_ops_fn(
                    op.operations,
                    logical_id_remap,
                    scope_path=(*op_key, iteration),
                    param_values=child_param_values,
                )

    def _draw_for_items_box(
        self,
        fig: Figure,
        op: ForItemsOperation,
        qubit_map: dict[str, int],
        x_pos: float,
        block_width: float | None = None,
        param_values: dict | None = None,
        logical_id_remap: dict[str, str] | None = None,
    ) -> None:
        """Draw a ForItemsOperation as a folded box."""
        if param_values is None:
            param_values = {}
        ax = fig._qm_ax

        affected_qubits = self.analyzer._analyze_loop_affected_qubits(
            op, qubit_map, logical_id_remap, param_values
        )
        if not affected_qubits:
            return

        y_coords = [self.qubit_y[q] for q in affected_qubits]
        min_y = min(y_coords)
        max_y = max(y_coords)

        # Build label
        key_str = ", ".join(op.key_vars) if op.key_vars else "k"
        if len(op.key_vars) > 1:
            key_str = f"({key_str})"
        dict_value = op.operands[0] if op.operands else None
        dict_name = getattr(dict_value, "name", "dict") if dict_value else "dict"
        label = f"for {key_str}, {op.value_var} in {dict_name}"

        # Add body summary
        expressions = []
        for body_op in op.operations:
            expr = self.analyzer._format_operation_as_expression(
                body_op,
                set(op.key_vars),
                body_operations=op.operations,
            )
            if expr:
                expressions.append(expr)
        if expressions:
            expr_lines = expressions[:3]
            if len(expressions) > 3:
                expr_lines.append("...")
            label += "\n" + "\n".join(expr_lines)

        width = block_width if block_width is not None else self.style.folded_loop_width
        num_text_lines = label.count("\n") + 1
        min_text_height = (
            num_text_lines * self.style.line_height
            + 2 * self.style.folded_box_text_v_padding
        )
        qubit_span_height = max_y - min_y + self.style.gate_height
        height = max(qubit_span_height, min_text_height)
        y_center = (min_y + max_y) / 2

        rect = mpatches.FancyBboxPatch(
            (x_pos - width / 2, y_center - height / 2),
            width,
            height,
            boxstyle=mpatches.BoxStyle.Round(
                pad=0, rounding_size=self.style.gate_corner_radius
            ),
            facecolor=self.style.for_items_face_color,
            edgecolor=self.style.for_items_edge_color,
            linewidth=1.5,
            linestyle="-",
            zorder=PORDER_GATE,
        )
        ax.add_patch(rect)

        ax.text(
            x_pos,
            y_center,
            label,
            ha="center",
            va="center",
            color=self.style.for_items_text_color,
            fontsize=self.style.subfont_size,
            zorder=PORDER_TEXT,
        )
