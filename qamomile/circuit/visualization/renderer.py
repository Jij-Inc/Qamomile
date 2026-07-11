"""Matplotlib-based circuit rendering.

This module provides MatplotlibRenderer, which handles all matplotlib
drawing operations for circuit visualization.
"""

from __future__ import annotations

import io
import math
from typing import TYPE_CHECKING

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .style import CircuitStyle
from .types import (
    PORDER_GATE,
    PORDER_LINE,
    PORDER_TEXT,
    PORDER_WIRE,
    ControlFlowLayout,
    FoldedBlockLayout,
    InlineBlockLayout,
    LayoutResult,
    LineSegment,
    PoweredGateLayout,
    Rect,
)
from .visual_ir import (
    GateOperationType,
    VFoldedBlock,
    VFoldedKind,
    VGate,
    VGateKind,
    VInlineBlock,
    VisualCircuit,
    VisualNode,
    VSkip,
    VUnfoldedKind,
    VUnfoldedSequence,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class MatplotlibRenderer:
    """Render circuit diagrams using matplotlib.

    Takes pre-computed layout coordinates and draws the circuit
    using matplotlib primitives.

    Args:
        style (CircuitStyle): Visual style used for patches, text, and figure
            sizing.
    """

    def __init__(self, style: CircuitStyle) -> None:
        """Initialize a renderer with immutable style settings.

        Args:
            style (CircuitStyle): Visual style used for patches, text, and
                figure sizing.
        """
        self.style = style
        # These are set during render()
        self.layout: LayoutResult | None = None
        self.qubit_y: list[float] = []
        self.qubit_end_positions: dict[int, float] = {}
        self.qubit_names: dict[int, str] = {}

    def render(self, vc: VisualCircuit, layout: LayoutResult) -> Figure:
        """Render the circuit from a VisualCircuit.

        Uses the Visual IR tree which carries all pre-resolved information.

        Args:
            vc (VisualCircuit): Circuit containing pre-resolved visual nodes.
            layout (LayoutResult): Pre-computed authoritative geometry.

        Returns:
            Figure: Configured matplotlib figure.
        """
        self.layout = layout
        self.qubit_y = layout.qubit_y
        self.qubit_end_positions = layout.qubit_end_positions
        self.qubit_names = vc.qubit_names

        fig = self._create_figure(vc.num_qubits)
        self._draw_wires(fig, vc.num_qubits)
        self._draw_operations(fig, vc)
        self._add_jupyter_display_support(fig)
        return fig

    def _draw_operations(self, fig: Figure, vc: VisualCircuit) -> None:
        """Draw all operations from Visual IR nodes.

        Dispatches on VisualNode types instead of IR Operation types,
        eliminating all Analyzer dependencies during drawing.

        Args:
            fig (Figure): Target matplotlib figure.
            vc (VisualCircuit): Visual circuit containing pre-resolved nodes.

        Returns:
            None

        Raises:
            ValueError: If any non-empty visual node lacks layout-owned
                geometry required for rendering.
        """
        assert self.layout is not None
        self._draw_visual_nodes(fig, vc.children)

    def _draw_inline_block(
        self,
        ax: Axes,
        node: VInlineBlock,
        placement: InlineBlockLayout,
    ) -> None:
        """Draw an inline block from layout-owned rectangles.

        Args:
            ax (Axes): Axes to draw on.
            node (VInlineBlock): Visual block carrying label and style metadata.
            placement (InlineBlockLayout): Final block geometry produced by the
                layout engine.

        Returns:
            None
        """
        inner = placement.inner_rect
        outer = placement.outer_rect
        is_controlled = bool(node.control_qubit_indices)
        block_zorder = PORDER_WIRE + 0.5 + node.depth * 0.1
        linestyle = "-" if is_controlled else "--"
        facecolor = self.style.background_color if is_controlled else "none"

        if node.power > 1:
            outer_patch = mpatches.Rectangle(
                (outer.left, outer.bottom),
                outer.right - outer.left,
                outer.top - outer.bottom,
                facecolor=facecolor,
                edgecolor=self.style.block_border_color,
                linewidth=1.5,
                linestyle=linestyle,
                zorder=block_zorder - 0.01,
            )
            ax.add_patch(outer_patch)
            ax.text(
                outer.right - self.style.label_horizontal_padding,
                outer.top - self.style.label_padding,
                f"pow={node.power}",
                ha="right",
                va="top",
                fontsize=self.style.subfont_size,
                fontweight="normal",
                color=self.style.block_border_color,
                zorder=PORDER_TEXT,
            )

        inner_patch = mpatches.Rectangle(
            (inner.left, inner.bottom),
            inner.right - inner.left,
            inner.top - inner.bottom,
            facecolor=facecolor,
            edgecolor=self.style.block_border_color,
            linewidth=1.5,
            linestyle=linestyle,
            zorder=block_zorder,
        )
        ax.add_patch(inner_patch)

        if is_controlled:
            target_qubits = [
                qubit
                for qubit in node.affected_qubits
                if qubit not in node.control_qubit_indices
            ]
            for qubit in target_qubits or node.affected_qubits:
                wire_y = self.qubit_y[qubit]
                ax.add_line(
                    mlines.Line2D(
                        [inner.left, inner.right],
                        [wire_y, wire_y],
                        color=self.style.wire_color,
                        linewidth=1.0,
                        zorder=block_zorder + 0.05,
                    )
                )

        ax.text(
            inner.left + self.style.label_horizontal_padding,
            inner.top - self.style.label_padding,
            node.label,
            ha="left",
            va="top",
            fontsize=self.style.subfont_size,
            fontweight="normal",
            color=self.style.block_border_color,
            zorder=PORDER_TEXT,
        )

        for segment in placement.control_segments:
            self._draw_line_segment(
                ax,
                segment,
                color=self.style.wire_color,
                linewidth=1.5,
                zorder=PORDER_LINE,
            )

        if node.control_qubit_indices:
            control_x = (outer.left + outer.right) / 2
            for qubit in node.control_qubit_indices:
                self._draw_control_dot(ax, control_x, self.qubit_y[qubit])

    def _draw_visual_nodes(
        self,
        fig: Figure,
        nodes: list[VisualNode],
    ) -> None:
        """Recursively draw nodes using layout-owned geometry.

        Args:
            fig (Figure): Target matplotlib figure.
            nodes (list[VisualNode]): Visual nodes to draw in traversal order.

        Returns:
            None

        Raises:
            ValueError: If a non-empty visual node lacks required layout-owned
                geometry.
        """
        assert self.layout is not None
        ax = fig._qm_ax  # type: ignore[attr-defined]

        for node in nodes:
            if isinstance(node, VSkip):
                continue

            if isinstance(node, VGate):
                if node.node_key not in self.layout.positions:
                    raise ValueError(
                        f"Missing gate position for visual node {node.node_key!r}"
                    )
                self._draw_vgate(
                    fig,
                    node,
                    self.layout.positions[node.node_key],
                    self.layout.gate_box_rects.get(node.node_key),
                    self.layout.powered_gate_layouts.get(node.node_key),
                )
                continue

            if isinstance(node, VInlineBlock):
                placement = self.layout.inline_block_layouts.get(node.node_key)
                if placement is None:
                    raise ValueError(
                        f"Missing inline-block geometry for {node.node_key!r}"
                    )
                self._draw_inline_block(ax, node, placement)
                self._draw_visual_nodes(fig, node.children)
                continue

            if isinstance(node, VFoldedBlock):
                placement = self.layout.folded_block_layouts.get(node.node_key)
                if placement is None:
                    raise ValueError(
                        f"Missing folded-block geometry for {node.node_key!r}"
                    )
                self._draw_vfolded_block(fig, node, placement)
                continue

            if isinstance(node, VUnfoldedSequence):
                if node.kind in (VUnfoldedKind.IF, VUnfoldedKind.WHILE):
                    placement = self.layout.control_flow_layouts.get(node.node_key)
                    if placement is None:
                        raise ValueError(
                            f"Missing control-flow geometry for {node.node_key!r}"
                        )
                    self._draw_control_flow(ax, node, placement)
                for iteration_children in node.iterations:
                    self._draw_visual_nodes(fig, iteration_children)
                continue

    def _draw_control_flow(
        self,
        ax: Axes,
        node: VUnfoldedSequence,
        placement: ControlFlowLayout,
    ) -> None:
        """Draw an unfolded IF or WHILE from final layout geometry.

        Args:
            ax (Axes): Axes to draw on.
            node (VUnfoldedSequence): Conditional node carrying its visual kind.
            placement (ControlFlowLayout): Final branch and connector geometry.

        Returns:
            None
        """
        edge_color, text_color, linestyle = self._control_flow_box_style(node.kind)
        if placement.connector_segments:
            self._draw_connector_segments(
                ax,
                placement.connector_segments,
                color=edge_color,
                linewidth=1.5,
                zorder=PORDER_LINE,
            )

        for box in placement.boxes:
            self._draw_control_flow_box(
                ax,
                box.rect,
                box.label,
                edge_color=edge_color,
                text_color=text_color,
                linestyle=linestyle,
            )

    def _control_flow_box_style(
        self, kind: VUnfoldedKind | VFoldedKind
    ) -> tuple[str, str, str]:
        """Resolve the (edge, text, linestyle) triple for a control-flow box.

        A ``while`` box uses the while-loop palette and a dashed border; every
        other boxed control-flow node (``if``/``else``) uses the if palette and
        a dash-dot border. This keeps the unfolded ``while`` visually distinct
        from an ``if`` while sharing the same layout machinery.

        Args:
            kind (VUnfoldedKind | VFoldedKind): The control-flow node kind.

        Returns:
            tuple[str, str, str]: ``(edge_color, text_color, linestyle)`` for
                the box border and its header label.
        """
        if kind in (VUnfoldedKind.WHILE, VFoldedKind.WHILE):
            return (
                self.style.while_loop_edge_color,
                self.style.while_loop_text_color,
                "--",
            )
        return self.style.if_edge_color, self.style.if_text_color, "-."

    def _draw_line_segment(
        self,
        ax: Axes,
        segment: LineSegment,
        *,
        color: str,
        linewidth: float,
        zorder: float,
    ) -> None:
        """Draw a line segment whose endpoints were fixed by layout.

        Args:
            ax (Axes): Axes to draw on.
            segment (LineSegment): Final line endpoints.
            color (str): Matplotlib line color.
            linewidth (float): Line width in points.
            zorder (float): Matplotlib drawing order.

        Returns:
            None
        """
        ax.add_line(
            mlines.Line2D(
                [segment.start_x, segment.end_x],
                [segment.start_y, segment.end_y],
                color=color,
                linewidth=linewidth,
                linestyle="-",
                solid_capstyle="butt",
                zorder=zorder,
            )
        )

    def _draw_connector_segments(
        self,
        ax: Axes,
        segments: tuple[LineSegment, ...],
        *,
        color: str,
        linewidth: float,
        zorder: float,
    ) -> None:
        """Draw contiguous layout segments as one polyline artist.

        Args:
            ax (Axes): Axes to draw on.
            segments (tuple[LineSegment, ...]): Ordered contiguous segments.
            color (str): Matplotlib line color.
            linewidth (float): Line width in points.
            zorder (float): Matplotlib drawing order.
        """
        if not segments:
            return
        x_values = [segments[0].start_x, *(segment.end_x for segment in segments)]
        y_values = [segments[0].start_y, *(segment.end_y for segment in segments)]
        ax.add_line(
            mlines.Line2D(
                x_values,
                y_values,
                color=color,
                linewidth=linewidth,
                linestyle="-",
                solid_capstyle="butt",
                solid_joinstyle="miter",
                zorder=zorder,
            )
        )

    def _draw_control_flow_box(
        self,
        ax: Axes,
        rect: Rect,
        label: str,
        *,
        edge_color: str,
        text_color: str,
        linestyle: str,
    ) -> None:
        """Draw one conditional branch from a final rectangle.

        Args:
            ax (Axes): Axes to draw on.
            rect (Rect): Final branch rectangle.
            label (str): Header label drawn at the top-left.
            edge_color (str): Branch border color.
            text_color (str): Header text color.
            linestyle (str): Matplotlib border style.

        Returns:
            None
        """
        patch = mpatches.FancyBboxPatch(
            (rect.left, rect.bottom),
            rect.right - rect.left,
            rect.top - rect.bottom,
            boxstyle=mpatches.BoxStyle.Round(
                pad=0, rounding_size=self.style.gate_corner_radius
            ),
            facecolor="none",
            edgecolor=edge_color,
            linewidth=1.5,
            linestyle=linestyle,
            zorder=PORDER_WIRE + 0.5,
        )
        ax.add_patch(patch)

        if label:
            ax.text(
                rect.left + self.style.label_horizontal_padding,
                rect.top - self.style.label_padding,
                label,
                ha="left",
                va="top",
                fontsize=self.style.subfont_size,
                color=text_color,
                fontweight="bold",
                zorder=PORDER_TEXT,
            )

    @staticmethod
    def _require_box_rect(
        node: VGate,
        box_rect: Rect | None,
        *,
        geometry_name: str,
    ) -> Rect:
        """Return required layout-owned box geometry for a gate node.

        Args:
            node (VGate): Gate-like node whose geometry is required.
            box_rect (Rect | None): Rectangle supplied by the layout engine.
            geometry_name (str): Human-readable geometry role used in errors.

        Returns:
            Rect: The supplied layout-owned rectangle.

        Raises:
            ValueError: If ``box_rect`` is None.
        """
        if box_rect is None:
            raise ValueError(
                f"Missing {geometry_name} geometry for visual node {node.node_key!r}"
            )
        return box_rect

    def _draw_vgate(
        self,
        fig: Figure,
        node: VGate,
        x_pos: float,
        box_rect: Rect | None = None,
        powered_layout: PoweredGateLayout | None = None,
    ) -> None:
        """Draw a gate-like node using pre-resolved fields and placement.

        Args:
            fig (Figure): Target matplotlib figure.
            node (VGate): Visual gate node to draw.
            x_pos (float): Center x-coordinate assigned by layout.
            box_rect (Rect | None): Exact text-bearing box bounds. Defaults to
                None for nodes rendered as native symbols.
            powered_layout (PoweredGateLayout | None): Target and wrapper
                rectangles for a powered controlled gate. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If layout omitted geometry required by the gate's
                rendering strategy.
        """
        ax = fig._qm_ax  # type: ignore[attr-defined]
        qubit_indices = node.qubit_indices

        if not qubit_indices:
            return

        y_coords = [self.qubit_y[q] for q in qubit_indices]

        if node.kind == VGateKind.GATE:
            if len(qubit_indices) == 1:
                self._draw_vgate_single(
                    ax,
                    node,
                    x_pos,
                    y_coords[0],
                    box_rect,
                )
            else:
                self._draw_vgate_multi(
                    ax,
                    node,
                    x_pos,
                    y_coords,
                    box_rect,
                )

        elif node.kind in (VGateKind.MEASURE, VGateKind.MEASURE_VECTOR):
            for y in y_coords:
                self._draw_measurement_box(ax, x_pos, y)

        elif node.kind == VGateKind.BLOCK_BOX:
            self._draw_vblock_box(
                ax,
                node,
                x_pos,
                box_rect,
                is_call_block=True,
            )

        elif node.kind == VGateKind.COMPOSITE_BOX:
            self._draw_vblock_box(
                ax,
                node,
                x_pos,
                box_rect,
                is_call_block=False,
            )

        elif node.kind == VGateKind.CONTROLLED_U_BOX:
            self._draw_vcontrolled_u_box(
                ax,
                node,
                x_pos,
                box_rect,
                powered_layout,
            )

        elif node.kind == VGateKind.EXPVAL:
            self._draw_vexpval(ax, node, x_pos, box_rect)

    def _draw_vgate_single(
        self,
        ax: Axes,
        node: VGate,
        x: float,
        y: float,
        box_rect: Rect | None,
    ) -> None:
        """Draw a single-qubit gate from a visual node.

        Args:
            ax (Axes): Axes to draw on.
            node (VGate): Gate node carrying label and style metadata.
            x (float): Gate center x-coordinate.
            y (float): Gate center y-coordinate.
            box_rect (Rect | None): Layout-owned box bounds.

        Raises:
            ValueError: If layout omitted the required gate-box rectangle.
        """
        bounds = self._require_box_rect(
            node,
            box_rect,
            geometry_name="single-gate box",
        )

        rect = mpatches.FancyBboxPatch(
            (bounds.left, bounds.bottom),
            bounds.width,
            bounds.height,
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
            fontweight="normal",
            zorder=PORDER_TEXT,
        )

    def _draw_vgate_multi(
        self,
        ax: Axes,
        node: VGate,
        x: float,
        y_coords: list[float],
        box_rect: Rect | None,
    ) -> None:
        """Draw a multi-qubit gate and its connection symbols.

        Args:
            ax (Axes): Axes to draw on.
            node (VGate): Gate node carrying type and label metadata.
            x (float): Shared center x-coordinate.
            y_coords (list[float]): Operand wire y-coordinates in gate order.
            box_rect (Rect | None): Layout-owned generic box bounds, or None
                for a native multi-qubit symbol.

        Raises:
            ValueError: If layout omitted a required generic gate-box
                rectangle.
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

        # Dispatch on gate_type for special rendering
        if node.gate_type == GateOperationType.CX:
            self._draw_control_dot(ax, x, y_coords[0])
            self._draw_target_x(ax, x, y_coords[1])
        elif node.gate_type == GateOperationType.CZ:
            for y in y_coords:
                self._draw_control_dot(ax, x, y)
        elif node.gate_type == GateOperationType.TOFFOLI:
            # Operands: [control1, control2, target]. Render in the
            # same style as CX — control dots on the two control
            # wires, target-X on the target wire.
            for y in y_coords[:-1]:
                self._draw_control_dot(ax, x, y)
            self._draw_target_x(ax, x, y_coords[-1])
        elif node.gate_type == GateOperationType.SWAP:
            for y in y_coords:
                self._draw_swap_x(ax, x, y)
        else:
            # Generic multi-qubit gate box
            bounds = self._require_box_rect(
                node,
                box_rect,
                geometry_name="multi-gate box",
            )

            rect = mpatches.FancyBboxPatch(
                (bounds.left, bounds.bottom),
                bounds.width,
                bounds.height,
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
                bounds.center_y,
                node.label,
                ha="center",
                va="center",
                color=self.style.gate_text_color,
                fontsize=self.style.font_size,
                fontweight="normal",
                zorder=PORDER_TEXT,
            )

            # Semicircle connection dots
            dot_radius = 0.05
            for y in y_coords:
                left_dot = mpatches.Wedge(
                    (bounds.left + dot_radius, y),
                    dot_radius,
                    theta1=90,
                    theta2=270,
                    facecolor=self.style.gate_text_color,
                    edgecolor=self.style.gate_text_color,
                    zorder=PORDER_GATE + 1,
                )
                ax.add_patch(left_dot)
                right_dot = mpatches.Wedge(
                    (bounds.right - dot_radius, y),
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
        box_rect: Rect | None,
        is_call_block: bool,
    ) -> None:
        """Draw a collapsed call-block or composite-gate box.

        Args:
            ax (Axes): Axes to draw on.
            node (VGate): Block-like visual node.
            x_pos (float): Box center x-coordinate.
            box_rect (Rect | None): Layout-owned box bounds.
            is_call_block (bool): Whether to use call-block colors and border.

        Raises:
            ValueError: If layout omitted the required block-box rectangle.
        """
        qubit_indices = node.qubit_indices
        if not qubit_indices:
            return

        bounds = self._require_box_rect(
            node,
            box_rect,
            geometry_name="block box",
        )

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
            (bounds.left, bounds.bottom),
            bounds.width,
            bounds.height,
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
            bounds.center_y,
            node.label,
            ha="center",
            va="center",
            color=text_color,
            fontsize=self.style.font_size,
            fontweight="normal",
            zorder=PORDER_TEXT,
        )

    def _draw_vcontrolled_u_box(
        self,
        ax: Axes,
        node: VGate,
        x_pos: float,
        box_rect: Rect | None,
        powered_layout: PoweredGateLayout | None,
    ) -> None:
        """Draw a controlled-U box with controls and an optional power wrapper.

        Args:
            ax (Axes): Axes to draw on.
            node (VGate): Controlled-U visual node.
            x_pos (float): Center x-coordinate assigned by layout.
            box_rect (Rect | None): Layout-owned target box bounds, or None
                when a native controlled symbol is rendered.
            powered_layout (PoweredGateLayout | None): Layout-owned target and
                wrapper rectangles, or None for an unpowered gate.

        Returns:
            None

        Raises:
            ValueError: If layout omitted required generic or powered gate
                geometry.
        """
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

        if self._draw_vcontrolled_u_special_gate(
            ax, node.gate_type, x_pos, control_y, target_y
        ):
            return

        # Draw control dots
        for y in control_y:
            self._draw_control_dot(ax, x_pos, y)

        # Draw target box
        if target_y:
            if box_rect is not None:
                target_bounds = box_rect
            elif powered_layout is not None:
                target_bounds = powered_layout.target_rect
            else:
                target_bounds = self._require_box_rect(
                    node,
                    box_rect,
                    geometry_name="controlled target box",
                )
            y_center = target_bounds.center_y

            rect = mpatches.FancyBboxPatch(
                (target_bounds.left, target_bounds.bottom),
                target_bounds.width,
                target_bounds.height,
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
                fontweight="normal",
                zorder=PORDER_TEXT,
            )

            # When the wrapped unitary is raised to a power, draw an
            # outer dashed wrapper that hugs the *target* box (not the
            # controls), with a ``pow=N`` annotation in the top-right
            # corner of that wrapper.  The vertical line and control
            # dots stay outside, exactly the way an inline-expanded
            # controlled-U (``VInlineBlock``) renders the same op, so
            # the collapsed and inline views look like the same shape
            # at two different zoom levels.
            if node.power > 1:
                if powered_layout is None:
                    raise ValueError(
                        "Powered controlled gates require layout-owned geometry"
                    )
                lp = self.style.label_padding
                pow_text = f"pow={node.power}"
                wrapper = powered_layout.wrapper_rect

                # ``facecolor=background_color`` (rather than ``"none"``)
                # so the wrapper *visually* stays transparent but
                # actually occludes whatever sits behind it.  The
                # multi-qubit connection line drawn by the controlled-U
                # box path (``PORDER_LINE`` / ``PORDER_GATE - 1``)
                # spans the full control <-> target range and would
                # otherwise show through inside the wrapper's extra
                # top label band where the ``pow=N`` annotation lives,
                # contradicting the wrapper's intent of containing
                # the powered gate.  The matching inline-block border
                # in ``_draw_inline_block`` already uses the
                # same ``background_color`` trick.
                outer_rect = mpatches.Rectangle(
                    (wrapper.left, wrapper.bottom),
                    wrapper.width,
                    wrapper.height,
                    facecolor=self.style.background_color,
                    edgecolor=self.style.block_border_color,
                    linewidth=1.5,
                    linestyle="--",
                    zorder=PORDER_GATE - 0.5,
                )
                ax.add_patch(outer_rect)

                ax.text(
                    wrapper.right - self.style.label_horizontal_padding,
                    wrapper.top - lp,
                    pow_text,
                    ha="right",
                    va="top",
                    fontsize=self.style.subfont_size,
                    fontweight="normal",
                    color=self.style.block_border_color,
                    zorder=PORDER_TEXT,
                )

    def _draw_vcontrolled_u_special_gate(
        self,
        ax: Axes,
        gate_type: GateOperationType | None,
        x_pos: float,
        control_y: list[float],
        target_y: list[float],
    ) -> bool:
        """Draw a controlled built-in gate without falling back to a box.

        Args:
            ax (Axes): Matplotlib Axes to draw on.
            gate_type (GateOperationType | None): Built-in gate type wrapped
                by the `ControlledUOperation`.
            x_pos (float): X coordinate of the operation.
            control_y (list[float]): Y coordinates of explicit control wires.
            target_y (list[float]): Y coordinates of wrapped-gate operands.

        Returns:
            bool: True when a dedicated symbol was drawn; False when the
                caller should draw the generic controlled-U box.
        """
        if gate_type is None or not target_y:
            return False

        if gate_type == GateOperationType.SWAP:
            if len(target_y) != 2:
                return False
            for y in control_y:
                self._draw_control_dot(ax, x_pos, y)
            for y in target_y:
                self._draw_swap_x(ax, x_pos, y)
            return True

        if gate_type in {
            GateOperationType.X,
            GateOperationType.CX,
            GateOperationType.TOFFOLI,
        }:
            for y in control_y + target_y[:-1]:
                self._draw_control_dot(ax, x_pos, y)
            self._draw_target_x(ax, x_pos, target_y[-1])
            return True

        if gate_type in {GateOperationType.Z, GateOperationType.CZ}:
            for y in control_y + target_y:
                self._draw_control_dot(ax, x_pos, y)
            return True

        return False

    def _draw_vexpval(
        self,
        ax: Axes,
        node: VGate,
        x_pos: float,
        box_rect: Rect | None,
    ) -> None:
        """Draw an expectation-value box over its operand wires.

        Args:
            ax (Axes): Axes to draw on.
            node (VGate): Expectation-value visual node.
            x_pos (float): Box center x-coordinate.
            box_rect (Rect | None): Layout-owned box bounds.

        Raises:
            ValueError: If layout omitted the required expectation-box
                rectangle.
        """
        qubit_indices = node.qubit_indices
        if not qubit_indices:
            return

        bounds = self._require_box_rect(
            node,
            box_rect,
            geometry_name="expectation box",
        )

        rect = mpatches.FancyBboxPatch(
            (bounds.left, bounds.bottom),
            bounds.width,
            bounds.height,
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
            bounds.center_y,
            node.label,
            ha="center",
            va="center",
            color=self.style.expval_text_color,
            fontsize=self.style.font_size,
            fontweight="normal",
            zorder=PORDER_TEXT,
        )

    def _draw_vfolded_block(
        self,
        fig: Figure,
        node: VFoldedBlock,
        placement: FoldedBlockLayout,
    ) -> None:
        """Draw a folded control-flow block from final layout geometry.

        Args:
            fig (Figure): Target matplotlib figure carrying ``_qm_ax``.
            node (VFoldedBlock): Folded block node to render.
            placement (FoldedBlockLayout): Final box and connector geometry.

        Returns:
            None
        """
        affected_qubits = node.affected_qubits
        if not affected_qubits:
            return

        ax = fig._qm_ax  # type: ignore[attr-defined]
        header = node.header_label
        body_lines = node.body_lines
        bounds = placement.rect
        x_pos = (bounds.left + bounds.right) / 2
        y_center = (bounds.bottom + bounds.top) / 2
        width = bounds.right - bounds.left
        height = bounds.top - bounds.bottom

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
            face_color = self.style.for_items_face_color
            edge_color = self.style.for_items_edge_color
            text_color = self.style.for_items_text_color
            linestyle = "-"
        elif node.kind == VFoldedKind.IF:
            face_color = self.style.if_face_color
            edge_color = self.style.if_edge_color
            text_color = self.style.if_text_color
            linestyle = "-."
        else:
            face_color = self.style.for_loop_face_color  # type: ignore[unreachable]
            edge_color = self.style.for_loop_edge_color
            text_color = self.style.for_loop_text_color
            linestyle = "-"

        if placement.connector_segments:
            self._draw_connector_segments(
                ax,
                placement.connector_segments,
                color=edge_color,
                linewidth=1.5,
                zorder=PORDER_LINE,
            )

        rect = mpatches.FancyBboxPatch(
            (bounds.left, bounds.bottom),
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
            header_y = bounds.top - self.style.label_vertical_offset - 0.15
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
            body_center_y = (bounds.bottom + header_y) / 2
            body_text_obj = ax.text(
                x_pos,
                body_center_y,
                body_text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=self.style.subfont_size,
                family="monospace",
                fontweight="normal",
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
                fontweight="normal",
                zorder=PORDER_TEXT,
            )

        # Participation markers: when the analyzer precisely determined
        # which wires the loop touches, draw a small control-style dot
        # centered on each affected wire at the box's left and right
        # edges so viewers can tell which wires participate vs. which
        # are passthrough. Skip when the analyzer fell back to
        # conservative analysis — the affected set may over-
        # approximate and dots would misleadingly imply certainty.
        #
        # Intentional zorder: ``PORDER_GATE - 1`` places each dot
        # *behind* the folded-block patch (``PORDER_GATE``), so the
        # opaque box fill hides the half that sits inside the box and
        # only the outer half protrudes. This mirrors the way a CX
        # control dot appears to sit on a wire (with the wire partly
        # hidden behind the dot). Raising the zorder would reveal the
        # full circle stuck on each edge — not the intended look.
        if node.affected_qubits_precise:
            for q in affected_qubits:
                y = self.qubit_y[q]
                for x in (bounds.left, bounds.right):
                    circle = mpatches.Circle(
                        (x, y),
                        radius=self.style.folded_marker_radius,
                        facecolor=self.style.wire_color,
                        edgecolor=self.style.wire_color,
                        zorder=PORDER_GATE - 1,
                    )
                    ax.add_patch(circle)

    def _add_jupyter_display_support(self, fig: Figure) -> None:
        """Add Jupyter display support to the figure.

        Adds ``_repr_png_()`` so notebooks can display the returned figure.

        Args:
            fig (Figure): Matplotlib figure to enhance.

        Returns:
            None
        """

        def _repr_png_() -> bytes:
            """Return a PNG representation for Jupyter display.

            Returns:
                bytes: Encoded PNG data for ``fig``.
            """
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            return buf.read()

        # Attach the method to the figure instance
        fig._repr_png_ = _repr_png_  # type: ignore[attr-defined]

    def _create_figure(self, num_qubits: int) -> Figure:
        """Create matplotlib figure with appropriate size.

        Args:
            num_qubits (int): Number of qubit wires.

        Returns:
            Figure: Configured matplotlib figure.
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
            fig._qm_ax = ax  # type: ignore[attr-defined]
            return fig

        assert self.layout is not None
        viewport = self.layout.viewport

        x_range = max(viewport.width, 1e-9)
        y_range = max(viewport.height, 1e-9)
        coordinate_scale = max(
            1.0,
            self.style.figure_scale_factor,
            self.style.figure_min_width / x_range,
            self.style.figure_min_height / y_range,
        )

        # Fill the physical canvas with the axes and use one common scale for
        # x and y. Text metrics measured in inches are then conservative data
        # widths, independent of subplot margins and circuit aspect ratio.
        fig = Figure(figsize=(x_range * coordinate_scale, y_range * coordinate_scale))
        FigureCanvasAgg(fig)
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.set_xlim(viewport.left, viewport.right)
        ax.set_ylim(viewport.bottom, viewport.top)
        ax.axis("off")
        ax.set_aspect("equal", adjustable="box")

        # Store ax in figure for later access
        fig._qm_ax = ax  # type: ignore[attr-defined]

        return fig

    def _draw_wires(
        self,
        fig: Figure,
        num_qubits: int,
    ) -> None:
        """Draw qubit wires and labels.

        Args:
            fig (Figure): Target matplotlib figure.
            num_qubits (int): Number of qubit wires.

        Returns:
            None
        """
        ax = fig._qm_ax  # type: ignore[attr-defined]
        assert self.layout is not None

        for i in range(num_qubits):
            y = self.qubit_y[i]
            wire_span = self.layout.wire_spans[i]
            # Measurement-terminated wires use "butt" capstyle to prevent
            # the half-linewidth overhang past the measurement box edge.
            capstyle: str = "butt" if i in self.qubit_end_positions else "projecting"
            line = mlines.Line2D(
                [wire_span.left, wire_span.right],
                [y, y],
                color=self.style.wire_color,
                linewidth=1,
                zorder=PORDER_WIRE,
                solid_capstyle=capstyle,  # type: ignore[arg-type]
            )
            ax.add_line(line)

            # Draw left qubit label
            label = self.qubit_names.get(i, f"q{i}")
            ax.text(
                wire_span.left - 0.2,
                y,
                label,
                ha="right",
                va="center",
                fontsize=self.style.font_size,
                fontweight="normal",
                zorder=PORDER_TEXT,
            )

        # Output labels on the right side are intentionally not drawn
        # to keep the circuit diagram cleaner

    def _draw_control_dot(self, ax: Axes, x: float, y: float) -> None:
        """Draw a control dot for controlled gates.

        Args:
            ax (Axes): Axes to draw on.
            x (float): X coordinate of the dot center.
            y (float): Y coordinate of the dot center.
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
            ax (Axes): Axes to draw on.
            x (float): X coordinate of the target center.
            y (float): Y coordinate of the target center.
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
            ax (Axes): Axes to draw on.
            x (float): X coordinate of the marker center.
            y (float): Y coordinate of the marker center.
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
            ax (Axes): Axes to draw on.
            x_pos (float): X center position.
            y (float): Y center position.
            width (float): Width of the containing box.
            height (float): Height of the containing box.
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
            ax (Axes): Axes to draw on.
            x_pos (float): X coordinate of the box center.
            y (float): Y coordinate of the box center.
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
