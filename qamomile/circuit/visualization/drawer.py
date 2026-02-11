"""Matplotlib-based circuit visualization.

This module provides Qiskit-style static circuit visualization
using matplotlib, focusing on clarity and simplicity.
"""

from __future__ import annotations

import io
import math
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.transforms import Bbox

    from qamomile.circuit.ir.graph import Graph

from qamomile.circuit.ir.block_value import BlockValue
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
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
from qamomile.circuit.ir.types.primitives import QubitType
from qamomile.circuit.ir.value import ArrayValue, Value

from .style import DEFAULT_STYLE, CircuitStyle

# Z-order for drawing priority (inspired by Qiskit)
PORDER_WIRE = 1  # Wire lines (lowest priority)
PORDER_GATE = 10  # Gate boxes
PORDER_LINE = 11  # Connection lines for multi-qubit gates
PORDER_TEXT = 13  # Text labels (highest priority)

# Known TeX symbol names (Greek letters)
_TEX_SYMBOLS = {
    # Lowercase Greek
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "varepsilon",
    "zeta",
    "eta",
    "theta",
    "vartheta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "pi",
    "rho",
    "varrho",
    "sigma",
    "varsigma",
    "tau",
    "upsilon",
    "phi",
    "varphi",
    "chi",
    "psi",
    "omega",
    # Uppercase Greek
    "Gamma",
    "Delta",
    "Theta",
    "Lambda",
    "Xi",
    "Pi",
    "Sigma",
    "Upsilon",
    "Phi",
    "Psi",
    "Omega",
    # Common math symbols
    "hbar",
    "ell",
    "nabla",
    "partial",
    "infty",
}


def _default_qubit_columns() -> defaultdict[int, int]:
    """Default factory: every qubit starts at column 1."""
    return defaultdict(lambda: 1)


@dataclass
class LayoutState:
    """Mutable state shared across layout handler methods."""

    positions: dict[tuple, int] = field(default_factory=dict)
    block_ranges: list[dict] = field(default_factory=list)
    block_widths: dict[tuple, float] = field(default_factory=dict)
    column: int = 1
    max_depth: int = 0
    actual_width: float = 1.0
    first_gate_x: float | None = None
    first_gate_half_width: float = 0.0
    qubit_columns: dict[int, int] = field(default_factory=_default_qubit_columns)
    qubit_right_edges: dict[int, float] = field(default_factory=dict)


@dataclass
class GateMeasure:
    """Width measurement for a gate/measurement/block-box operation."""

    estimated_width: float
    half_width: float
    columns_needed: int
    is_block_box: bool = False
    box_width: float | None = None


@dataclass
class BlockMeasure:
    """Width measurement for an inlined block."""

    label: str
    label_width: float
    content_width: float
    final_width: float
    max_gate_width: float
    border_padding: float
    children: list[MeasureNode]
    affected_qubits: list[int]
    depth: int
    control_qubit_indices: list[int] = field(default_factory=list)


@dataclass
class LoopMeasure:
    """Width measurement for a ForOperation."""

    fold: bool
    affected_qubits: list[int]
    folded_width: float | None = None
    iteration_children: list[list[MeasureNode]] | None = None
    iteration_widths: list[float] | None = None
    num_iterations: int = 0
    iter_param_values: list[dict] | None = None


@dataclass
class SkipMeasure:
    """Marker for zero-space operations."""

    pass


@dataclass
class IfMeasure:
    """Width measurement for an IfOperation."""

    fold: bool
    affected_qubits: list[int]
    condition_label: str
    folded_width: float | None = None
    true_children: list["MeasureNode"] | None = None
    false_children: list["MeasureNode"] | None = None
    true_width: float = 0.0
    false_width: float = 0.0


MeasureNode = Union[GateMeasure, BlockMeasure, LoopMeasure, SkipMeasure, IfMeasure]


class MatplotlibDrawer:
    """Matplotlib-based circuit drawer with Qiskit-style layout.

    This drawer produces static matplotlib figures showing quantum circuits.
    It supports two modes:
    - Block mode (inline=False): Shows CallBlockOperation as boxes
    - Inline mode (inline=True): Expands CallBlockOperation contents
    """

    def __init__(self, graph: Graph, style: CircuitStyle | None = None):
        """Initialize the drawer.

        Args:
            graph: Computation graph to visualize.
            style: Visual style configuration. Uses DEFAULT_STYLE if None.
        """
        self.graph = graph
        self.style = style or DEFAULT_STYLE
        self.inline = False
        self.fold_loops = True
        self.expand_composite = False
        self.inline_depth: int | None = None
        self._inlined_op_keys: set[tuple] = set()

        self.qubit_y: list[float] = []
        self.qubit_end_positions: dict[int, float] = {}
        self.qubit_names: dict[int, str] = {}
        self._num_qubits: int = 0
        self._max_above: dict[int, float] = {}
        self._max_below: dict[int, float] = {}

    def _should_inline_at_depth(self, depth: int) -> bool:
        """Whether to inline CallBlock/ControlledU at this nesting depth."""
        return self.inline and (self.inline_depth is None or depth < self.inline_depth)

    def draw(
        self,
        inline: bool = False,
        fold_loops: bool = True,
        expand_composite: bool = False,
        inline_depth: int | None = None,
    ) -> Figure:
        """Generate a matplotlib Figure of the circuit.

        Args:
            inline: If True, expand CallBlockOperation. If False, show as boxes.
            fold_loops: If True (default), display ForOperation as blocks instead of unrolling.
                       If False, expand loops and show all iterations.
            expand_composite: If True, expand CompositeGateOperation (QFT, QPE, etc.).
                            If False (default), show as boxes. Independent of inline.
            inline_depth: Maximum nesting depth for inline expansion. None means
                         unlimited (default). 0 means no inlining, 1 means top-level
                         only, etc. Only affects CallBlock/ControlledU, not CompositeGate.

        Returns:
            Figure object.
        """
        self.inline = inline
        self.fold_loops = fold_loops
        self.expand_composite = expand_composite
        self.inline_depth = inline_depth
        self._inlined_op_keys = set()
        graph = self.graph

        # Build qubit mapping
        qubit_map = self._build_qubit_map(graph)
        # Use the total number of wires tracked during map building
        num_qubits = self._num_qubits

        # Initialize qubit end positions (for wire termination at measurements)
        self.qubit_end_positions: dict[int, float] = {}

        # Layout operations
        layout = self._layout_operations(graph, qubit_map)

        # Compute per-qubit y-positions (variable spacing for inline blocks)
        block_ranges = layout.get("block_ranges", [])
        self.qubit_y = self._compute_qubit_y_positions(num_qubits, block_ranges)

        # Create figure
        fig = self._create_figure(num_qubits, layout)

        # Draw wires
        self._draw_wires(fig, num_qubits, layout, qubit_map)

        # Draw operations
        self._draw_operations(fig, graph, qubit_map, layout)

        # Add Jupyter display support
        self._add_jupyter_display_support(fig)

        return fig

    def _build_qubit_map(self, graph: Graph) -> dict[str, int]:
        """Build mapping from qubit logical_id to wire indices.

        In SSA form, each operation creates new Values via next_version(),
        which preserves logical_id. This means all versions of a qubit share
        the same logical_id, so we only need logical_id-based tracking.

        Args:
            graph: Computation graph.

        Returns:
            Dictionary mapping logical_id (str) to wire index (0-based).
        """
        qubit_map: dict[str, int] = {}
        qubit_names: dict[int, str] = {}
        next_idx = 0

        def map_block_results(
            operands: list[Value],
            results: list[Value],
            logical_id_remap: dict[str, str],
            param_values: dict | None = None,
        ) -> None:
            """Map block output logical_ids to the same wire indices as their inputs.

            For each (operand, result) pair that is a qubit, resolves the operand's
            logical_id via logical_id_remap, looks up its wire index in qubit_map,
            and assigns the same index to the result's logical_id. If the operand
            is not yet registered, both are assigned a new wire index.

            Args:
                operands: Input values passed to the block (actual arguments).
                results: Output values returned from the block.
                logical_id_remap: Mapping from formal-parameter logical_ids to
                    actual-argument logical_ids.
                param_values: Parameter values for resolving symbolic indices.
            """
            nonlocal next_idx
            for operand, result in zip(operands, results, strict=True):
                if not isinstance(result.type, QubitType):
                    continue
                lid = self._resolve_array_element_lid(
                    operand, qubit_map, logical_id_remap, param_values
                )
                qubit_idx = qubit_map.get(lid)
                if qubit_idx is not None:
                    qubit_map[result.logical_id] = qubit_idx
                else:
                    # Guard: symbolic array element whose parent is known
                    # → don't create a new wire; let layout resolve dynamically
                    if (
                        hasattr(operand, "parent_array")
                        and operand.parent_array is not None
                        and hasattr(operand, "element_indices")
                        and operand.element_indices
                        and not operand.element_indices[0].is_constant()
                    ):
                        parent_lid = logical_id_remap.get(
                            operand.parent_array.logical_id,
                            operand.parent_array.logical_id,
                        )
                        if parent_lid in qubit_map:
                            continue

                    qubit_map[operand.logical_id] = next_idx
                    qubit_map[result.logical_id] = next_idx
                    next_idx += 1

        def build_chains(
            ops: list[Operation],
            logical_id_remap: dict[str, str] | None = None,
            depth: int = 0,
            param_values: dict | None = None,
        ) -> None:
            """Register qubit logical_ids by walking operations recursively.

            For QInitOperation, registers new qubits (scalar or array elements).
            For CallBlockOperation (inline=True), builds a logical_id_remap from
            block formal parameters to actual arguments, then recurses into the
            block body. For CastOperation, propagates the source qubit's wire
            index to the cast result.

            GateOperation and similar are no-ops because next_version() preserves
            logical_id.

            Args:
                ops: List of operations to process.
                logical_id_remap: Mapping from block formal-parameter logical_ids
                    to actual-argument logical_ids. Only non-empty when recursing
                    into inlined CallBlockOperations.
                depth: Current nesting depth for inline_depth checking.
                param_values: Parameter values for resolving symbolic indices.
            """
            nonlocal next_idx
            if logical_id_remap is None:
                logical_id_remap = {}
            if param_values is None:
                param_values = {}

            for op in ops:
                if isinstance(op, QInitOperation):
                    qubits = op.results[0]

                    # If the qubits is an array, we need to register each element separately.
                    if isinstance(qubits, ArrayValue):
                        # If this assertion fails without adding a new quantum array type, there are some bugs.
                        # If you added a new quantum array type, please also update the visualization logic here.
                        assertion_message = f"[FOR DEVELOPER] QInitOperation produced non-qubit array type: {qubits.type}"
                        assert isinstance(qubits.type, QubitType), assertion_message

                        if len(qubits.shape) == 1:  # 1D array (Vector)
                            size_value = qubits.shape[0]
                            if not size_value.is_constant():
                                raise ValueError(
                                    f"Cannot visualize circuit: qubit array "
                                    f"'{qubits.name}' has symbolic size. "
                                    f"Please provide concrete values for all "
                                    f"size parameters when calling draw()."
                                )

                            array_size = size_value.get_const()

                            # As we have already verified if it is constant, this assertion must be passed.
                            # If it fails, there is a bug in the code leading to this point.
                            assertion_message = "[FOR DEVELOPER] The given qubit must have a known integer size as a constant here, however, apparently not, which means there is a bug in the code leading to this point."
                            assert array_size is not None, assertion_message

                            # In QKernel._create_bound_input, the size of qubit arrays is always converted into integer by int function.
                            # If this assertion fails without adding a new quantum array type, there are some bugs.
                            assertion_message = f"[FOR DEVELOPER] Qubit array '{qubits.name}' has non-integer size: {array_size}"
                            assert isinstance(array_size, int), assertion_message

                            for i in range(array_size):
                                element_key = f"{qubits.logical_id}_[{i}]"
                                if element_key not in qubit_map:
                                    qubit_map[element_key] = next_idx
                                    qubit_names[next_idx] = f"{qubits.name}[{i}]"
                                    next_idx += 1
                            # Store base index for array
                            qubit_map[qubits.logical_id] = qubit_map.get(
                                f"{qubits.logical_id}_[0]", next_idx
                            )
                            continue
                        # TODO: 2D, 3D arrays if needed.
                        else:
                            raise NotImplementedError(
                                f"Cannot visualize circuit: qubit array "
                                f"'{qubits.name}' has unsupported rank "
                                f"{len(qubits.shape)}."
                            )
                    # If the qubits is actually just a qubit, then register the qubit.
                    # This else branch relied on the specification that QInitOperation is always producing qubits.
                    else:
                        if qubits.logical_id not in qubit_map:
                            qubit_map[qubits.logical_id] = next_idx
                            qubit_names[next_idx] = qubits.name
                            next_idx += 1

                elif isinstance(op, CallBlockOperation):
                    if self._should_inline_at_depth(depth):
                        block_value = op.operands[0]
                        # The IR guarantees that operands[0] of CallBlockOperation is always a BlockValue,
                        # so this assertion should always pass.
                        # If it fails, there is a bug in the IR construction.
                        assertion_message = (
                            "[FOR DEVELOPER] CallBlockOperation.operands[0] must be a BlockValue. "
                            "If this assertion fails, there is a bug in the IR construction."
                        )
                        assert isinstance(block_value, BlockValue), assertion_message
                        new_remap = dict(logical_id_remap)
                        actual_inputs = op.operands[1:]

                        for dummy_input, actual_input in zip(
                            block_value.input_values, actual_inputs, strict=True
                        ):
                            if not isinstance(dummy_input.type, QubitType):
                                continue
                            actual_lid = self._resolve_array_element_lid(
                                actual_input,
                                qubit_map,
                                logical_id_remap,
                                param_values,
                            )
                            new_remap[dummy_input.logical_id] = actual_lid

                            # Ensure actual_input is registered
                            if actual_lid not in qubit_map:
                                qubit_map[actual_lid] = next_idx
                                next_idx += 1

                        build_chains(
                            block_value.operations,
                            new_remap,
                            depth + 1,
                            param_values,
                        )

                    qubit_operands = [
                        v for v in op.operands[1:] if isinstance(v.type, QubitType)
                    ]
                    qubit_results = [
                        v for v in op.results if isinstance(v.type, QubitType)
                    ]
                    map_block_results(
                        qubit_operands, qubit_results, logical_id_remap, param_values
                    )

                elif isinstance(op, CastOperation):
                    assert op.operands and op.results, (
                        "[FOR DEVELOPER] CastOperation must have operands and results. "
                        "If this assertion fails, there is a bug in the IR construction."
                    )
                    source = op.operands[0]
                    result = op.results[0]
                    source_lid = logical_id_remap.get(
                        source.logical_id, source.logical_id
                    )
                    qubit_idx = qubit_map.get(source_lid)
                    if qubit_idx is not None:
                        qubit_map[result.logical_id] = qubit_idx

                elif isinstance(
                    op, ControlledUOperation
                ) and self._should_inline_at_depth(depth):
                    block_value = op.block
                    if isinstance(block_value, BlockValue):
                        new_remap = dict(logical_id_remap)
                        for dummy_input, actual_input in zip(
                            block_value.input_values, op.target_operands
                        ):
                            if not isinstance(dummy_input.type, QubitType):
                                continue
                            actual_lid = self._resolve_array_element_lid(
                                actual_input,
                                qubit_map,
                                logical_id_remap,
                                param_values,
                            )
                            new_remap[dummy_input.logical_id] = actual_lid
                            if actual_lid not in qubit_map:
                                qubit_map[actual_lid] = next_idx
                                next_idx += 1
                        build_chains(
                            block_value.operations,
                            new_remap,
                            depth + 1,
                            param_values,
                        )
                    qubit_operands = [
                        v
                        for v in list(op.control_operands) + list(op.target_operands)
                        if isinstance(v.type, QubitType)
                    ]
                    qubit_results = [
                        v for v in op.results if isinstance(v.type, QubitType)
                    ]
                    map_block_results(
                        qubit_operands, qubit_results, logical_id_remap, param_values
                    )

                elif (
                    isinstance(op, CompositeGateOperation)
                    and self.expand_composite
                    and op.has_implementation
                ):
                    block_value = op.implementation
                    if isinstance(block_value, BlockValue):
                        new_remap = dict(logical_id_remap)
                        for dummy_input, actual_input in zip(
                            block_value.input_values, op.target_qubits
                        ):
                            if not isinstance(dummy_input.type, QubitType):
                                continue
                            actual_lid = self._resolve_array_element_lid(
                                actual_input,
                                qubit_map,
                                logical_id_remap,
                                param_values,
                            )
                            new_remap[dummy_input.logical_id] = actual_lid
                            if actual_lid not in qubit_map:
                                qubit_map[actual_lid] = next_idx
                                next_idx += 1
                        build_chains(
                            block_value.operations,
                            new_remap,
                            depth + 1,
                            param_values,
                        )
                    qubit_operands = [
                        v
                        for v in list(op.control_qubits) + list(op.target_qubits)
                        if isinstance(v.type, QubitType)
                    ]
                    qubit_results = [
                        v for v in op.results if isinstance(v.type, QubitType)
                    ]
                    map_block_results(
                        qubit_operands, qubit_results, logical_id_remap, param_values
                    )

                elif isinstance(op, ForOperation):
                    start, stop, step = self._evaluate_loop_range(op, param_values)
                    if stop is not None and (not self.fold_loops or self.inline):
                        for iter_value in range(start, stop, step):
                            child_pv = dict(param_values)
                            child_pv[f"_loop_{op.loop_var}"] = iter_value
                            build_chains(
                                op.operations,
                                logical_id_remap,
                                depth + 1,
                                child_pv,
                            )
                    else:
                        build_chains(
                            op.operations,
                            logical_id_remap,
                            depth + 1,
                            param_values,
                        )

                elif isinstance(op, WhileOperation):
                    build_chains(
                        op.operations, logical_id_remap, depth + 1, param_values
                    )

                elif isinstance(op, IfOperation):
                    build_chains(
                        op.true_operations,
                        logical_id_remap,
                        depth + 1,
                        param_values,
                    )
                    build_chains(
                        op.false_operations,
                        logical_id_remap,
                        depth + 1,
                        param_values,
                    )

                elif isinstance(op, ForItemsOperation):
                    dict_value = op.operands[0] if op.operands else None
                    materialized = (
                        self._materialize_dict_entries(dict_value)
                        if dict_value
                        else None
                    )
                    if materialized is not None and not self.fold_loops:
                        for entry_key, entry_value in materialized:
                            child_pv = dict(param_values)
                            if isinstance(entry_key, tuple):
                                for kv, ek in zip(op.key_vars, entry_key):
                                    child_pv[f"_loop_{kv}"] = ek
                            elif op.key_vars:
                                child_pv[f"_loop_{op.key_vars[0]}"] = entry_key
                            child_pv[f"_loop_{op.value_var}"] = entry_value
                            build_chains(
                                op.operations,
                                logical_id_remap,
                                depth + 1,
                                child_pv,
                            )
                    else:
                        build_chains(
                            op.operations,
                            logical_id_remap,
                            depth + 1,
                            param_values,
                        )

                # GateOperation, non-expanded CompositeGateOperation:
                # No-op — next_version() preserves logical_id

        build_chains(graph.operations, depth=0, param_values={})
        self.qubit_names = qubit_names
        self._num_qubits = next_idx
        return qubit_map

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

    @staticmethod
    def _is_zero_iteration_loop(
        start_val: int, stop_val_raw: int | None, step_val: int
    ) -> bool:
        """Check if a loop has zero iterations.

        Returns True only when stop is concrete and the range is empty.

        Args:
            start_val: Loop start value.
            stop_val_raw: Loop stop value, or None if symbolic.
            step_val: Loop step value.

        Returns:
            True if the loop would produce zero iterations.
        """
        if stop_val_raw is None:
            return False
        if step_val > 0 and start_val >= stop_val_raw:
            return True
        if step_val < 0 and start_val <= stop_val_raw:
            return True
        return False

    def _compute_border_padding(self, depth: int) -> float:
        """Compute border padding for a given nesting depth.

        Args:
            depth: Nesting depth of the block.

        Returns:
            Border padding value, clamped to min_block_padding.
        """
        return max(
            self.style.min_block_padding,
            self.style.border_padding_base
            - depth * self.style.border_padding_depth_factor,
        )

    def _compute_block_box_bounds(
        self,
        name: str,
        start_x: float,
        end_x: float,
        depth: int,
        max_gate_width: float,
    ) -> tuple[float, float]:
        """Compute (box_left, box_right) for an inlined block border.

        Label expansion is right-only: box_left is always gate-based,
        box_right expands rightward if the label text needs more space.
        """
        padding = self._compute_border_padding(depth)
        box_left = start_x - max_gate_width / 2 - padding
        gate_box_right = end_x + max_gate_width / 2 + padding
        title_text_width = len(name) * self.style.char_width_base
        label_right = (
            box_left
            + self.style.label_horizontal_padding
            + title_text_width
            + self.style.label_horizontal_padding
        )
        box_right = max(gate_box_right, label_right)
        return box_left, box_right

    def _analyze_loop_affected_qubits(
        self,
        op: ForOperation,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str] | None = None,
        param_values: dict | None = None,
    ) -> list[int]:
        """Analyze which qubits are affected by a ForOperation.

        Args:
            op: ForOperation to analyze.
            qubit_map: Mapping from logical_id to qubit index.
            logical_id_remap: Mapping from dummy logical_ids to actual logical_ids.
            param_values: Parameter values for resolving loop range and indices.

        Returns:
            List of affected qubit indices.
        """
        if logical_id_remap is None:
            logical_id_remap = {}
        if param_values is None:
            param_values = {}

        # Try precise iteration-based analysis when loop range is known (ForOperation only)
        if isinstance(op, ForOperation):
            start_val, stop_val_raw, step_val = self._evaluate_loop_range(
                op, param_values
            )
            if (
                stop_val_raw is not None
                and step_val != 0
                and not self._is_zero_iteration_loop(start_val, stop_val_raw, step_val)
            ):
                num_iters = len(range(int(start_val), int(stop_val_raw), int(step_val)))
                if 0 < num_iters <= 100:
                    precise_affected: set[int] = set()
                    all_resolved = True
                    for iter_val in range(
                        int(start_val), int(stop_val_raw), int(step_val)
                    ):
                        iter_params = dict(param_values)
                        iter_params[f"_loop_{op.loop_var}"] = iter_val
                        self._evaluate_loop_body_intermediates(
                            op.operations, iter_params
                        )
                        for inner_op in op.operations:
                            operands: list = []
                            if isinstance(inner_op, GateOperation):
                                operands = list(inner_op.operands)
                            elif isinstance(inner_op, CallBlockOperation):
                                operands = list(inner_op.operands[1:])
                            elif isinstance(inner_op, ControlledUOperation):
                                operands = list(inner_op.operands[1:])
                            elif isinstance(inner_op, CompositeGateOperation):
                                operands = list(inner_op.control_qubits) + list(
                                    inner_op.target_qubits
                                )
                            elif isinstance(inner_op, MeasureOperation):
                                operands = list(inner_op.operands[:1])
                            for operand in operands:
                                idx = self._resolve_operand_to_qubit_index(
                                    operand,
                                    qubit_map,
                                    logical_id_remap,
                                    iter_params,
                                )
                                if idx is not None:
                                    precise_affected.add(idx)
                                else:
                                    all_resolved = False
                    if all_resolved and precise_affected:
                        return list(precise_affected)

        # Fallback to conservative analysis
        affected = set()

        def get_qubit_index_or_expand(operand: Value) -> int | None:
            """Get qubit index, expanding symbolic array indices into `affected`."""
            is_array_element = (
                hasattr(operand, "parent_array") and operand.parent_array is not None
            )
            if is_array_element:
                parent_lid = operand.parent_array.logical_id
                parent_lid = logical_id_remap.get(parent_lid, parent_lid)
                if parent_lid in qubit_map:
                    # As the first level if-condition checked this is an array element,
                    # this assertion must be passed.
                    assertion_message = "[FOR DEVELOPER] Operand must have element_indices for array element access."
                    assert operand.element_indices, assertion_message

                    idx_value = operand.element_indices[0]
                    if not idx_value.is_constant():
                        base_idx = qubit_map[parent_lid]
                        size = None
                        if (
                            hasattr(operand.parent_array, "shape")
                            and operand.parent_array.shape
                        ):
                            size_val = operand.parent_array.shape[0]
                            if size_val.is_constant():
                                size = size_val.get_const()
                        # Fallback: count qubit_map entries with matching pattern
                        if size is None or not isinstance(size, int):
                            count = 0
                            while f"{parent_lid}_[{count}]" in qubit_map:
                                count += 1
                            if count > 0:
                                for i in range(count):
                                    affected.add(base_idx + i)
                        else:
                            for i in range(size):
                                affected.add(base_idx + i)
                        return None  # Already added all

            return self._resolve_operand_to_qubit_index(
                operand, qubit_map, logical_id_remap
            )

        def collect_from_ops(ops: list[Operation]) -> None:
            """Recursively collect qubit indices from all operations into `affected` set."""
            for inner_op in ops:
                if isinstance(inner_op, GateOperation):
                    for operand in inner_op.operands:
                        idx = get_qubit_index_or_expand(operand)
                        if idx is not None:
                            affected.add(idx)
                elif isinstance(inner_op, CallBlockOperation):
                    for operand in inner_op.operands[1:]:  # Skip BlockValue
                        idx = get_qubit_index_or_expand(operand)
                        if idx is not None:
                            affected.add(idx)
                elif isinstance(inner_op, ControlledUOperation):
                    for operand in inner_op.operands[1:]:  # Skip BlockValue
                        idx = get_qubit_index_or_expand(operand)
                        if idx is not None:
                            affected.add(idx)
                elif isinstance(inner_op, CompositeGateOperation):
                    for operand in list(inner_op.control_qubits) + list(
                        inner_op.target_qubits
                    ):
                        idx = get_qubit_index_or_expand(operand)
                        if idx is not None:
                            affected.add(idx)
                elif isinstance(inner_op, MeasureOperation):
                    if inner_op.operands:
                        idx = get_qubit_index_or_expand(inner_op.operands[0])
                        if idx is not None:
                            affected.add(idx)
                elif isinstance(inner_op, MeasureVectorOperation):
                    if inner_op.operands:
                        indices = self._resolve_operand_to_qubit_indices(
                            inner_op.operands[0], qubit_map, logical_id_remap
                        )
                        affected.update(indices)
                elif isinstance(inner_op, ForOperation):
                    collect_from_ops(inner_op.operations)
                elif isinstance(inner_op, WhileOperation):
                    collect_from_ops(inner_op.operations)
                elif isinstance(inner_op, IfOperation):
                    collect_from_ops(inner_op.true_operations)
                    collect_from_ops(inner_op.false_operations)
                elif isinstance(inner_op, ForItemsOperation):
                    collect_from_ops(inner_op.operations)

        collect_from_ops(op.operations)
        return list(affected)

    def _evaluate_value(
        self,
        value: Value,
        param_values: dict,
        operations: list[Operation] | None = None,
    ) -> int | float | None:
        """Evaluate a Value to a concrete number using param_values and BinOp resolution.

        Handles constants, named parameters, and binary arithmetic expressions.

        Args:
            value: IR Value to evaluate.
            param_values: Mapping from logical_ids or parameter names to concrete values.
            operations: Operation list to search for defining BinOps.
                Falls back to self.graph.operations if None.

        Returns:
            Concrete numeric value, or None if unresolvable.
        """
        # 1. Direct constant
        if value.is_constant():
            return value.get_const()

        # 2. Named parameter
        if value.is_parameter():
            name = value.parameter_name()
            if name and name in param_values:
                v = param_values[name]
                if isinstance(v, (int, float)):
                    return v

        # 3. Check param_values by logical_id
        vid = value.logical_id
        if vid in param_values:
            v = param_values[vid]
            if isinstance(v, (int, float)):
                return v

        # 3.5. Loop variable name-based lookup
        if value.name:
            loop_key = f"_loop_{value.name}"
            if loop_key in param_values:
                v = param_values[loop_key]
                if isinstance(v, (int, float)):
                    return v

        # 3.6. Array element access (e.g. edges[idx, 0])
        if hasattr(value, "parent_array") and value.parent_array is not None:
            parent = value.parent_array
            const_array = None
            if hasattr(parent, "params") and "const_array" in parent.params:
                const_array = parent.params["const_array"]
            if const_array is None:
                const_array = param_values.get(f"_array_data_{parent.logical_id}")
            if (
                const_array is not None
                and hasattr(value, "element_indices")
                and value.element_indices
            ):
                indices = []
                for idx_val in value.element_indices:
                    resolved_idx = self._evaluate_value(
                        idx_val, param_values, operations
                    )
                    if resolved_idx is None:
                        break
                    indices.append(int(resolved_idx))
                else:
                    try:
                        result = const_array
                        for idx in indices:
                            result = result[idx]
                        return result
                    except (IndexError, TypeError):
                        pass

        # 4. Look for defining BinOp in operations
        if operations is None:
            operations = getattr(self, "graph", None)
            operations = operations.operations if operations else []

        for op in operations:
            if isinstance(op, BinOp) and op.results and id(op.results[0]) == id(value):
                lhs_val = self._evaluate_value(op.lhs, param_values, operations)
                rhs_val = self._evaluate_value(op.rhs, param_values, operations)
                if lhs_val is not None and rhs_val is not None:
                    if op.kind == BinOpKind.ADD:
                        return lhs_val + rhs_val
                    elif op.kind == BinOpKind.SUB:
                        return lhs_val - rhs_val
                    elif op.kind == BinOpKind.MUL:
                        return lhs_val * rhs_val
                    elif op.kind == BinOpKind.FLOORDIV:
                        return lhs_val // rhs_val if rhs_val != 0 else None
                    elif op.kind == BinOpKind.DIV:
                        return lhs_val / rhs_val if rhs_val != 0 else None
                    elif op.kind == BinOpKind.POW:
                        result = lhs_val**rhs_val
                        if isinstance(rhs_val, (int, float)) and rhs_val < 0:
                            return result
                        return int(result)
                return None

        return None

    @staticmethod
    def _safe_int_str(val: int | float | None) -> str:
        """Convert a numeric value to an integer string, handling non-integer floats.

        Args:
            val: Numeric value to convert, or None.

        Returns:
            Integer string representation, or "?" if val is None.
        """
        if val is None:
            return "?"
        try:
            return str(int(val))
        except (ValueError, OverflowError):
            return str(val)

    def _format_range_str(
        self, op: ForOperation, loop_vars: set[str], param_values: dict
    ) -> str:
        """Format range string for ForOperation with algebraic fallback.

        Tries concrete resolution first. If stop value is unresolvable,
        falls back to algebraic expression using _format_value_as_expression.
        """
        start_val, stop_val_raw, step_val = self._evaluate_loop_range(op, param_values)
        start_str = self._safe_int_str(start_val)
        step_str = self._safe_int_str(step_val)

        if stop_val_raw is not None:
            stop_str = self._safe_int_str(stop_val_raw)
        elif len(op.operands) > 1:
            stop_str = self._format_value_as_expression(op.operands[1], loop_vars)
        else:
            stop_str = "?"

        if start_val == 0 and step_val == 1:
            return f"qm.range({stop_str})"
        elif step_val == 1:
            return f"qm.range({start_str}, {stop_str})"
        else:
            return f"qm.range({start_str}, {stop_str}, {step_str})"

    def _format_binop_operand(self, value: Value, param_values: dict) -> str | None:
        """Format a BinOp operand as a symbolic string.

        Returns a human-readable string for the operand, or None if
        unresolvable. Numeric values are formatted as numbers, symbolic
        parameters use their name (with Greek letter mapping).
        """
        # Check param_values first (may contain number or symbolic string)
        if value.logical_id in param_values:
            pv = param_values[value.logical_id]
            if isinstance(pv, (int, float)):
                if isinstance(pv, float) and pv == int(pv):
                    return str(int(pv))
                return str(pv)
            if isinstance(pv, str):
                return pv
        # Constant
        c = value.get_const()
        if c is not None:
            if isinstance(c, float) and c == int(c):
                return str(int(c))
            return str(c)
        # Named parameter
        if value.is_parameter():
            name = value.parameter_name() or value.name
            if name:
                return name
        # Fallback: name
        if hasattr(value, "name") and value.name:
            return value.name
        return None

    def _build_symbolic_binop(self, binop: BinOp, param_values: dict) -> str | None:
        """Build a symbolic string for a BinOp expression.

        Returns expressions like "2*gamma", "theta+1", "gamma" (when multiplied by 1).
        Returns None if either operand is unresolvable.
        """
        lhs_str = self._format_binop_operand(binop.lhs, param_values)
        rhs_str = self._format_binop_operand(binop.rhs, param_values)
        if lhs_str is None or rhs_str is None:
            return None

        # Simplify common cases
        if binop.kind == BinOpKind.MUL:
            if lhs_str == "1":
                return rhs_str
            if rhs_str == "1":
                return lhs_str
            if lhs_str == "0" or rhs_str == "0":
                return "0"
            # Prefer coefficient form: "2*gamma" rather than "gamma*2"
            return f"{lhs_str}*{rhs_str}"
        elif binop.kind == BinOpKind.ADD:
            if lhs_str == "0":
                return rhs_str
            if rhs_str == "0":
                return lhs_str
            return f"{lhs_str}+{rhs_str}"
        elif binop.kind == BinOpKind.SUB:
            if rhs_str == "0":
                return lhs_str
            return f"{lhs_str}-{rhs_str}"
        elif binop.kind == BinOpKind.DIV:
            if rhs_str == "1":
                return lhs_str
            return f"{lhs_str}/{rhs_str}"
        else:
            op_sym = {
                BinOpKind.FLOORDIV: "//",
                BinOpKind.POW: "**",
            }.get(binop.kind, "?")
            return f"{lhs_str}{op_sym}{rhs_str}"

    def _evaluate_loop_body_intermediates(
        self,
        operations: list[Operation],
        param_values: dict,
    ) -> None:
        """Pre-evaluate intermediate BinOp results in loop body.

        Scans loop body operations for BinOp (e.g., j = i + 1) and stores
        evaluated results by value ID in param_values. This allows subsequent
        index resolution to find the concrete values.

        When a BinOp cannot be fully resolved numerically (e.g., one operand
        is a symbolic parameter), a symbolic string expression is stored
        instead (e.g., "2*gamma", "theta+1").

        Args:
            operations: List of operations in the loop body.
            param_values: Mutable mapping updated in-place with resolved
                intermediate values keyed by logical_id.
        """
        for op in operations:
            if isinstance(op, BinOp) and op.results:
                result_value = op.results[0]
                resolved = self._evaluate_value(result_value, param_values, operations)
                if resolved is not None:
                    param_values[result_value.logical_id] = resolved
                else:
                    # Try building a symbolic string expression
                    symbolic = self._build_symbolic_binop(op, param_values)
                    if symbolic is not None:
                        param_values[result_value.logical_id] = symbolic

        # Also resolve array element access intermediates (e.g. i = edges[idx, 0])
        for op in operations:
            if not isinstance(
                op,
                (
                    GateOperation,
                    CallBlockOperation,
                    ControlledUOperation,
                    CompositeGateOperation,
                ),
            ):
                continue
            operand_list = (
                op.operands if isinstance(op, GateOperation) else op.operands[1:]
            )
            for operand in operand_list:
                if hasattr(operand, "element_indices") and operand.element_indices:
                    for idx_val in operand.element_indices:
                        if (
                            hasattr(idx_val, "parent_array")
                            and idx_val.parent_array is not None
                        ):
                            resolved = self._evaluate_value(
                                idx_val, param_values, operations
                            )
                            if resolved is not None:
                                param_values[idx_val.logical_id] = resolved

    def _resolve_array_element_lid(
        self,
        value: Value,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict | None = None,
    ) -> str:
        """Resolve a Value's logical_id to a canonical qubit_map key.

        For array element Values whose logical_id is a random UUID (created by
        Vector.__getitem__), construct the canonical key format
        ``f"{parent_lid}_[{idx}]"`` and register an alias if found.
        When param_values is provided, symbolic indices (e.g. loop variables)
        can be resolved via _evaluate_value.
        """
        lid = logical_id_remap.get(value.logical_id, value.logical_id)
        if lid in qubit_map:
            return lid
        if hasattr(value, "parent_array") and value.parent_array is not None:
            parent_lid = logical_id_remap.get(
                value.parent_array.logical_id, value.parent_array.logical_id
            )
            if hasattr(value, "element_indices") and value.element_indices:
                idx_value = value.element_indices[0]
                if idx_value.is_constant():
                    idx = idx_value.get_const()
                    if idx is not None:
                        element_key = f"{parent_lid}_[{int(idx)}]"
                        if element_key in qubit_map:
                            qubit_map[lid] = qubit_map[element_key]
                            return element_key
                elif param_values:
                    idx = self._evaluate_value(idx_value, param_values)
                    if idx is not None:
                        element_key = f"{parent_lid}_[{int(idx)}]"
                        if element_key in qubit_map:
                            # Don't cache — symbolic index may resolve
                            # differently across loop iterations
                            return element_key
        return lid

    def _resolve_operand_to_qubit_index(
        self,
        operand: Value,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str] | None = None,
        param_values: dict | None = None,
    ) -> int | None:
        """Resolve a single operand to its qubit wire index.

        Handles logical_id_remap lookup, direct qubit_map, parent_array + element_indices,
        and BinOp evaluation for computed indices.

        Args:
            operand: IR Value representing a qubit operand.
            qubit_map: Mapping from logical_id to qubit wire index.
            logical_id_remap: Mapping from dummy logical_ids to actual logical_ids.
            param_values: Parameter values for evaluating computed indices.

        Returns:
            Qubit wire index or None if unresolvable.
        """
        if logical_id_remap is None:
            logical_id_remap = {}
        if param_values is None:
            param_values = {}

        resolved_lid = logical_id_remap.get(operand.logical_id, operand.logical_id)

        # Parent array element access — check FIRST for array elements
        # so that param_values can resolve symbolic indices correctly
        if hasattr(operand, "parent_array") and operand.parent_array is not None:
            parent_lid = operand.parent_array.logical_id
            parent_lid = logical_id_remap.get(parent_lid, parent_lid)
            if parent_lid in qubit_map:
                if hasattr(operand, "element_indices") and operand.element_indices:
                    idx_value = operand.element_indices[0]
                    idx = None
                    if idx_value.is_constant():
                        idx = idx_value.get_const()
                    else:
                        idx = self._evaluate_value(idx_value, param_values)
                    if idx is not None:
                        return qubit_map[parent_lid] + int(idx)
                # Fallback to base index
                return qubit_map[parent_lid]

        # Direct lookup (non-array-element operands)
        if resolved_lid in qubit_map:
            return qubit_map[resolved_lid]

        return None

    def _resolve_operand_to_qubit_indices(
        self,
        operand: Value,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str] | None = None,
        param_values: dict | None = None,
    ) -> list[int]:
        """Resolve an operand to qubit wire indices, expanding ArrayValue.

        For ArrayValue operands with known size, returns all element indices.
        For scalar operands, wraps the single index in a list.

        Args:
            operand: IR Value representing a qubit operand.
            qubit_map: Mapping from logical_id to qubit wire index.
            logical_id_remap: Mapping from dummy logical_ids to actual logical_ids.
            param_values: Parameter values for evaluating computed indices.

        Returns:
            List of qubit wire indices (may be empty if unresolvable).
        """
        if logical_id_remap is None:
            logical_id_remap = {}

        resolved_lid = logical_id_remap.get(operand.logical_id, operand.logical_id)

        # Expand ArrayValue to all individual qubits
        if (
            isinstance(operand, ArrayValue)
            and hasattr(operand, "shape")
            and operand.shape
            and resolved_lid in qubit_map
        ):
            base_idx = qubit_map[resolved_lid]
            size_value = operand.shape[0]
            if size_value.is_constant():
                array_size = size_value.get_const()
                if array_size is not None and isinstance(array_size, int):
                    return [base_idx + ai for ai in range(array_size)]

        # Handle synthetic ArrayValue from expval() tuple input
        if (
            isinstance(operand, ArrayValue)
            and hasattr(operand, "params")
            and "qubit_values" in operand.params
        ):
            indices = []
            for qv in operand.params["qubit_values"]:
                idx = self._resolve_operand_to_qubit_index(
                    qv, qubit_map, logical_id_remap, param_values
                )
                if idx is not None:
                    indices.append(idx)
            if indices:
                return indices

        # Single operand resolution
        idx = self._resolve_operand_to_qubit_index(
            operand, qubit_map, logical_id_remap, param_values
        )
        if idx is not None:
            return [idx]
        return []

    def _evaluate_loop_range(
        self, op: ForOperation, param_values: dict
    ) -> tuple[int, int | None, int]:
        """Evaluate ForOperation start/stop/step to concrete values.

        Args:
            op: ForOperation whose range to evaluate.
            param_values: Parameter values for resolving symbolic operands.

        Returns:
            (start_val, stop_val_or_None, step_val) where stop_val is None
            if unresolvable (symbolic).

        Raises:
            ValueError: If step evaluates to zero.
        """
        start_val = (
            self._evaluate_value(op.operands[0], param_values) if op.operands else None
        )
        stop_val = (
            self._evaluate_value(op.operands[1], param_values)
            if len(op.operands) > 1
            else None
        )
        step_val = (
            self._evaluate_value(op.operands[2], param_values)
            if len(op.operands) > 2
            else None
        )

        if start_val is None:
            start_val = 0
        if step_val is None:
            step_val = 1
        if step_val == 0:
            raise ValueError("ForOperation step must not be zero")

        return (
            int(start_val),
            stop_val if stop_val is None else int(stop_val),
            int(step_val),
        )

    def _compute_loop_iterations(
        self, start_val: int, stop_val: int, step_val: int
    ) -> int:
        """Compute number of loop iterations from resolved range values.

        Args:
            start_val: Loop start value.
            stop_val: Loop stop value (must be concrete).
            step_val: Loop step value (non-zero).

        Returns:
            Number of iterations.
        """
        if step_val > 0:
            return (stop_val - start_val + step_val - 1) // step_val
        return (start_val - stop_val - step_val - 1) // (-step_val)

    def _build_block_value_mappings(
        self,
        block_value,
        actual_inputs: list[Value],
        logical_id_remap: dict[str, str],
        param_values: dict,
        qubit_map: dict[str, int] | None = None,
    ) -> tuple[dict[str, str], dict]:
        """Build logical_id_remap and child_param_values for a BlockValue.

        Maps dummy block input logical_ids to actual input logical_ids,
        building the mappings needed for recursive processing.

        Args:
            block_value: BlockValue whose input mappings to build.
            actual_inputs: Actual input Values passed to the block.
            logical_id_remap: Current logical_id remapping (copied, not mutated).
            param_values: Current parameter values (copied, not mutated).
            qubit_map: Qubit wire map for resolving array element indices.

        Returns:
            (new_logical_id_remap, child_param_values)
        """
        new_logical_id_remap = dict(logical_id_remap)

        for dummy_input, actual_input in zip(block_value.input_values, actual_inputs):
            # For Qubit array elements, resolve through _resolve_array_element_lid
            # to get the canonical parent_[idx] key instead of the raw UUID
            if (
                qubit_map is not None
                and hasattr(actual_input, "parent_array")
                and actual_input.parent_array is not None
                and isinstance(actual_input.type, QubitType)
            ):
                resolved_lid = self._resolve_array_element_lid(
                    actual_input, qubit_map, logical_id_remap, param_values
                )
                new_logical_id_remap[dummy_input.logical_id] = resolved_lid
            else:
                actual_lid = logical_id_remap.get(
                    actual_input.logical_id, actual_input.logical_id
                )
                new_logical_id_remap[dummy_input.logical_id] = actual_lid

        child_param_values = dict(param_values)
        for dummy_input, actual_input in zip(block_value.input_values, actual_inputs):
            # The IR guarantees that operands are always Value instances, so this assertion should always pass.
            # If it fails, there is a bug in the IR construction.
            assertion_message = (
                "[FOR DEVELOPER] Operation.operands elements must be Value instances. "
                "If this assertion fails, there is a bug in the IR construction."
            )
            assert isinstance(actual_input, Value), assertion_message
            const = actual_input.get_const()
            if const is not None:
                child_param_values[dummy_input.logical_id] = const
            elif actual_input.logical_id in param_values:
                child_param_values[dummy_input.logical_id] = param_values[
                    actual_input.logical_id
                ]
            elif actual_input.is_parameter():
                child_param_values[dummy_input.logical_id] = (
                    actual_input.parameter_name() or actual_input.name
                )

        # Propagate ArrayValue shape dimensions and const_array data
        for dummy_input, actual_input in zip(block_value.input_values, actual_inputs):
            if isinstance(actual_input, ArrayValue) and isinstance(
                dummy_input, ArrayValue
            ):
                if dummy_input.shape and actual_input.shape:
                    for dummy_dim, actual_dim in zip(
                        dummy_input.shape, actual_input.shape
                    ):
                        const = actual_dim.get_const()
                        if const is not None:
                            child_param_values[dummy_dim.logical_id] = const
                        elif actual_dim.logical_id in param_values:
                            child_param_values[dummy_dim.logical_id] = param_values[
                                actual_dim.logical_id
                            ]
                if (
                    hasattr(actual_input, "params")
                    and "const_array" in actual_input.params
                ):
                    child_param_values[f"_array_data_{dummy_input.logical_id}"] = (
                        actual_input.params["const_array"]
                    )

        return new_logical_id_remap, child_param_values

    def _estimate_gate_width(
        self, op: GateOperation, param_values: dict | None = None
    ) -> float:
        """Estimate gate box width from label text without matplotlib axes.

        Uses character-based estimation suitable for layout phase
        (before figure creation). Also serves as floor for drawing phase.

        Args:
            op: GateOperation whose width to estimate.
            param_values: Parameter values for resolving gate labels.

        Returns:
            Estimated gate width in data coordinate units.
        """
        label, has_param = self._get_gate_label(op, param_values)
        if not has_param:
            return self.style.gate_width

        # Strip TeX $ delimiters (not rendered)
        visual = label.replace("$", "")
        # TeX commands like \theta render as ~1 wide character
        visual = re.sub(r"\\[a-zA-Z]+", "X", visual)
        effective_len = len(visual)
        # Use style settings for width calculation
        char_width = self.style.char_width_gate  # 0.14 default, configurable
        text_width = effective_len * char_width
        padding = self.style.text_padding  # match drawing phase
        return max(self.style.gate_width, text_width + 2 * padding)

    def _estimate_label_box_width(self, label: str) -> float:
        """Estimate box width for a text label (blocks, composite gates, controlled-U).

        Strips TeX formatting for visual length estimation.

        Args:
            label: Text label to estimate width for.

        Returns:
            Estimated box width in data coordinate units.
        """
        visual_label = label.replace("$", "")
        visual_label = re.sub(r"\\[a-zA-Z]+", "X", visual_label)
        text_width = len(visual_label) * self.style.char_width_gate
        return max(text_width + 2 * self.style.text_padding, self.style.gate_width)

    @staticmethod
    def _get_child_op_lists(op: Operation) -> list[list[Operation]]:
        """Return child operation lists for control-flow operations."""
        if isinstance(op, (ForOperation, WhileOperation, ForItemsOperation)):
            return [op.operations]
        if isinstance(op, IfOperation):
            return [op.true_operations, op.false_operations]
        return []

    def _max_block_gate_width(
        self, operations: list[Operation], param_values: dict | None = None
    ) -> float:
        """Find the maximum estimated gate width among GateOperations in a block.

        Args:
            operations: List of operations to scan.
            param_values: Parameter values for gate width estimation.

        Returns:
            Maximum gate width, at least style.gate_width.
        """
        max_width = self.style.gate_width
        for op in operations:
            if isinstance(op, GateOperation):
                max_width = max(max_width, self._estimate_gate_width(op, param_values))
            else:
                for child_ops in self._get_child_op_lists(op):
                    max_width = max(
                        max_width,
                        self._max_block_gate_width(child_ops, param_values),
                    )
        return max_width

    # ------------------------------------------------------------------
    # Measure Phase: compute widths without assigning coordinates
    # ------------------------------------------------------------------

    def _measure_generic_operation(
        self,
        op: Operation,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> GateMeasure:
        """Measure width of a generic operation (gate, measurement, block box)."""
        if isinstance(op, GateOperation):
            estimated_w = self._estimate_gate_width(op, param_values)
            return GateMeasure(
                estimated_width=estimated_w,
                half_width=estimated_w / 2,
                columns_needed=max(1, math.ceil(estimated_w)),
            )
        elif isinstance(op, CallBlockOperation):
            label = self._get_block_label(op, qubit_map)
            box_width = self._estimate_label_box_width(label)
            return GateMeasure(
                estimated_width=box_width,
                half_width=box_width / 2,
                columns_needed=max(1, int(box_width) + 1),
                is_block_box=True,
                box_width=box_width,
            )
        elif isinstance(op, CompositeGateOperation):
            box_width = self._estimate_label_box_width(op.name.upper())
            return GateMeasure(
                estimated_width=box_width,
                half_width=box_width / 2,
                columns_needed=max(1, int(box_width) + 1),
                is_block_box=True,
                box_width=box_width,
            )
        elif isinstance(op, ControlledUOperation):
            u_name = getattr(op.block, "name", "U") or "U"
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
            box_width = self._estimate_label_box_width(label)
            return GateMeasure(
                estimated_width=box_width,
                half_width=box_width / 2,
                columns_needed=max(1, int(box_width) + 1),
                is_block_box=True,
                box_width=box_width,
            )
        elif isinstance(op, (MeasureOperation, MeasureVectorOperation)):
            w = self.style.gate_width
            return GateMeasure(
                estimated_width=w,
                half_width=w / 2,
                columns_needed=1,
            )
        elif isinstance(op, ExpvalOp):
            label = "<H>"
            box_width = self._estimate_label_box_width(label)
            return GateMeasure(
                estimated_width=box_width,
                half_width=box_width / 2,
                columns_needed=max(1, int(box_width) + 1),
                is_block_box=True,
                box_width=box_width,
            )
        else:
            raise TypeError(
                f"Unsupported operation type for _measure_generic_operation: {type(op).__name__}"
            )

    def _sum_children_widths(self, children: list[MeasureNode]) -> float:
        """Sum widths of MeasureNode children with inter-element gaps.

        Handles GateMeasure, BlockMeasure, LoopMeasure (folded and unfolded),
        and skips SkipMeasure. Does NOT add border extent.
        """
        gap = self.style.gate_gap
        total = 0.0
        count = 0

        for child in children:
            if isinstance(child, SkipMeasure):
                continue
            elif isinstance(child, GateMeasure):
                total += child.estimated_width
            elif isinstance(child, BlockMeasure):
                total += child.final_width
            elif isinstance(child, LoopMeasure):
                if child.fold:
                    total += child.folded_width or 0.0
                else:
                    total += sum(child.iteration_widths or [])
                    if child.num_iterations > 1:
                        total += gap * (child.num_iterations - 1)
            elif isinstance(child, IfMeasure):
                if child.fold:
                    total += child.folded_width or 0.0
                else:
                    total += child.true_width + child.false_width + gap
            count += 1

        if count > 1:
            total += gap * (count - 1)

        return total

    def _compute_content_width(
        self,
        children: list[MeasureNode],
        max_gate_width: float,
        depth: int,
    ) -> float:
        """Compute the total content width of a list of measure nodes.

        This is the width that the content occupies inside a block, including
        inter-gate gaps and border extent on both sides.
        """
        border_padding = self._compute_border_padding(depth)
        border_extent = max_gate_width / 2 + border_padding

        total = self._sum_children_widths(children)

        # Add border extent on both sides
        total += 2 * border_extent

        return total

    def _measure_inline_block(
        self,
        op: CallBlockOperation | ControlledUOperation | CompositeGateOperation,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
    ) -> BlockMeasure:
        """Measure width of an inlined block operation."""
        # Extract block_value, affected_qubits, and actual_inputs based on op type
        if isinstance(op, CallBlockOperation):
            block_value = op.operands[0]
            assert isinstance(block_value, BlockValue), (
                "[FOR DEVELOPER] CallBlockOperation.operands[0] must be a BlockValue. "
                "If this assertion fails, there is a bug in the IR construction."
            )
            affected_qubits: list[int] = []
            for operand in op.operands[1:]:
                affected_qubits.extend(
                    self._resolve_operand_to_qubit_indices(
                        operand, qubit_map, logical_id_remap, param_values
                    )
                )
            actual_inputs = op.operands[1:]
            block_name = block_value.name or "block"
        elif isinstance(op, ControlledUOperation):
            block_value = op.block
            assert isinstance(block_value, BlockValue), (
                "[FOR DEVELOPER] ControlledUOperation.block must be a BlockValue. "
                "If this assertion fails, there is a bug in the IR construction."
            )
            control_qubit_indices: list[int] = []
            for operand in op.control_operands:
                idx = self._resolve_operand_to_qubit_index(
                    operand, qubit_map, logical_id_remap, param_values
                )
                if idx is not None:
                    control_qubit_indices.append(idx)
            affected_qubits = list(control_qubit_indices)
            for operand in op.target_operands:
                idx = self._resolve_operand_to_qubit_index(
                    operand, qubit_map, logical_id_remap, param_values
                )
                if idx is not None:
                    affected_qubits.append(idx)
            actual_inputs = list(op.target_operands)
            u_name = getattr(block_value, "name", "U") or "U"
            block_name = u_name
        elif isinstance(op, CompositeGateOperation):
            block_value = op.implementation
            assert isinstance(block_value, BlockValue), (
                "[FOR DEVELOPER] CompositeGateOperation.implementation must be a BlockValue. "
                "If this assertion fails, there is a bug in the IR construction."
            )
            affected_qubits = []
            for operand in list(op.control_qubits) + list(op.target_qubits):
                idx = self._resolve_operand_to_qubit_index(
                    operand, qubit_map, logical_id_remap, param_values
                )
                if idx is not None:
                    affected_qubits.append(idx)
            actual_inputs = list(op.target_qubits)
            block_name = op.name
        else:
            raise TypeError(
                f"Unsupported operation type for inline block: {type(op).__name__}"
            )

        new_logical_id_remap, child_param_values = self._build_block_value_mappings(
            block_value,
            actual_inputs,
            logical_id_remap,
            param_values,
            qubit_map=qubit_map,
        )

        # Include qubits created inside the block body via QInitOperation
        for body_op in block_value.operations:
            if isinstance(body_op, QInitOperation):
                result_val = body_op.results[0]
                lid = result_val.logical_id
                if lid in qubit_map:
                    idx = qubit_map[lid]
                    if idx not in affected_qubits:
                        affected_qubits.append(idx)

        max_gate_width = self._max_block_gate_width(
            block_value.operations, child_param_values
        )

        children = self._measure_operations(
            block_value.operations,
            qubit_map,
            new_logical_id_remap,
            child_param_values,
            depth + 1,
        )

        border_padding = self._compute_border_padding(depth)
        content_width = self._compute_content_width(children, max_gate_width, depth)

        # Label width using the same estimation as _estimate_label_box_width
        label_width = self._estimate_label_box_width(block_name)

        final_width = max(label_width, content_width)

        ctrl_indices = (
            control_qubit_indices if isinstance(op, ControlledUOperation) else []
        )
        return BlockMeasure(
            label=block_name,
            label_width=label_width,
            content_width=content_width,
            final_width=final_width,
            max_gate_width=max_gate_width,
            border_padding=border_padding,
            children=children,
            affected_qubits=affected_qubits,
            depth=depth,
            control_qubit_indices=ctrl_indices,
        )

    def _estimate_folded_loop_width(
        self, op: ForOperation, param_values: dict
    ) -> float:
        """Estimate box width needed for folded ForOperation text."""
        range_str = self._format_range_str(op, set(), param_values)
        header = f"for {op.loop_var} in {range_str}"

        body_lines: list[str] = []
        for body_op in op.operations:
            expr = self._format_operation_as_expression(
                body_op,
                {op.loop_var},
                param_values=param_values,
                body_operations=op.operations,
            )
            if expr:
                body_lines.extend(expr.split("\n"))

        scale = self.style.subfont_size / self.style.font_size
        header_width = len(header) * self.style.char_width_base * scale
        max_body_chars = max((len(line) for line in body_lines), default=0)
        body_width = max_body_chars * self.style.char_width_monospace * scale
        text_width = max(header_width, body_width)
        return max(
            self.style.folded_loop_width, text_width + 2 * self.style.text_padding
        )

    def _measure_for_operation(
        self,
        op: ForOperation,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
    ) -> LoopMeasure | SkipMeasure:
        """Measure width of a ForOperation."""
        start_val, stop_val_raw, step_val = self._evaluate_loop_range(op, param_values)

        if self._is_zero_iteration_loop(start_val, stop_val_raw, step_val):
            return SkipMeasure()

        affected_qubits = self._analyze_loop_affected_qubits(
            op, qubit_map, logical_id_remap, param_values
        )

        if self.fold_loops:
            return LoopMeasure(
                fold=True,
                affected_qubits=affected_qubits,
                folded_width=self._estimate_folded_loop_width(op, param_values),
            )

        # Cannot unfold if stop is symbolic — fall back to folded box
        if stop_val_raw is None:
            return LoopMeasure(
                fold=True,
                affected_qubits=affected_qubits,
                folded_width=self._estimate_folded_loop_width(op, param_values),
            )

        # Unfolded: measure each iteration
        num_iterations = self._compute_loop_iterations(
            start_val, stop_val_raw, step_val
        )
        iteration_children: list[list[MeasureNode]] = []
        iteration_widths: list[float] = []
        iter_param_values_list: list[dict] = []

        for iteration in range(num_iterations):
            iter_value = start_val + iteration * step_val
            child_param_values = dict(param_values)
            child_param_values[f"_loop_{op.loop_var}"] = iter_value

            self._evaluate_loop_body_intermediates(op.operations, child_param_values)

            children = self._measure_operations(
                op.operations,
                qubit_map,
                logical_id_remap,
                child_param_values,
                depth + 1,
            )

            # Content width for this iteration (without border — loops don't have borders per iteration)
            iter_width = self._sum_children_widths(children)

            iteration_children.append(children)
            iteration_widths.append(iter_width)
            iter_param_values_list.append(child_param_values)

        return LoopMeasure(
            fold=False,
            affected_qubits=affected_qubits,
            iteration_children=iteration_children,
            iteration_widths=iteration_widths,
            num_iterations=num_iterations,
            iter_param_values=iter_param_values_list,
        )

    def _measure_while_operation(
        self,
        op: WhileOperation,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
    ) -> LoopMeasure:
        """Measure width of a WhileOperation (always folded)."""
        affected_qubits = self._analyze_loop_affected_qubits(
            op, qubit_map, logical_id_remap, param_values
        )
        label = "while cond:"
        expressions = []
        for body_op in op.operations:
            expr = self._format_operation_as_expression(
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
        label_width = (
            len(label) * self.style.char_width_monospace + 2 * self.style.text_padding
        )
        width = max(label_width, self.style.folded_loop_width)
        return LoopMeasure(
            fold=True,
            affected_qubits=affected_qubits,
            folded_width=width,
        )

    def _measure_if_operation(
        self,
        op: IfOperation,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
    ) -> IfMeasure:
        """Measure width of an IfOperation (fold or unfold)."""
        # Collect affected qubits from both branches
        affected: set[int] = set()

        def collect_qubits(ops: list[Operation]) -> None:
            """Recursively collect qubit indices from operations into `affected` set."""
            for inner_op in ops:
                operands: list = []
                if isinstance(inner_op, GateOperation):
                    operands = list(inner_op.operands)
                elif isinstance(inner_op, CallBlockOperation):
                    operands = list(inner_op.operands[1:])
                elif isinstance(inner_op, ControlledUOperation):
                    operands = list(inner_op.operands[1:])
                elif isinstance(inner_op, CompositeGateOperation):
                    operands = list(inner_op.control_qubits) + list(
                        inner_op.target_qubits
                    )
                elif isinstance(inner_op, MeasureOperation):
                    operands = list(inner_op.operands[:1])
                elif isinstance(inner_op, ForOperation):
                    collect_qubits(inner_op.operations)
                    continue
                elif isinstance(inner_op, WhileOperation):
                    collect_qubits(inner_op.operations)
                    continue
                elif isinstance(inner_op, IfOperation):
                    collect_qubits(inner_op.true_operations)
                    collect_qubits(inner_op.false_operations)
                    continue
                for operand in operands:
                    idx = self._resolve_operand_to_qubit_index(
                        operand, qubit_map, logical_id_remap, param_values
                    )
                    if idx is not None:
                        affected.add(idx)

        collect_qubits(op.true_operations)
        collect_qubits(op.false_operations)
        affected_qubits = list(affected)

        # Build condition label
        cond = op.condition
        cond_name = getattr(cond, "name", None) or "cond"
        condition_label = f"if {cond_name}:"

        if self.fold_loops:
            label = condition_label
            label_width = (
                len(label) * self.style.char_width_monospace
                + 2 * self.style.text_padding
            )
            width = max(label_width, self.style.folded_loop_width)
            return IfMeasure(
                fold=True,
                affected_qubits=affected_qubits,
                condition_label=condition_label,
                folded_width=width,
            )

        # Unfolded: measure both branches
        true_children = self._measure_operations(
            op.true_operations, qubit_map, logical_id_remap, param_values, depth + 1
        )
        true_width = self._sum_children_widths(true_children)
        false_children = self._measure_operations(
            op.false_operations, qubit_map, logical_id_remap, param_values, depth + 1
        )
        false_width = self._sum_children_widths(false_children)

        return IfMeasure(
            fold=False,
            affected_qubits=affected_qubits,
            condition_label=condition_label,
            true_children=true_children,
            false_children=false_children,
            true_width=true_width,
            false_width=false_width,
        )

    @staticmethod
    def _materialize_dict_entries(
        dict_value: Value,
    ) -> list[tuple] | None:
        """Extract entries from DictValue, including bound_data in params.

        DictValue.entries is typically empty when data is bound at build
        time; the actual data lives in params["bound_data"]. This helper
        materializes those entries into a plain list of (key, value) tuples
        where keys and values are raw Python objects (not IR Values).

        Returns None if the dict is truly unbound (no data available).
        """
        # Try IR-level entries first
        if hasattr(dict_value, "entries") and dict_value.entries:
            return dict_value.entries

        # Try params["bound_data"]
        if hasattr(dict_value, "params") and "bound_data" in dict_value.params:
            bound = dict_value.params["bound_data"]
            if isinstance(bound, dict):
                return list(bound.items())

        return None

    def _measure_for_items_operation(
        self,
        op: ForItemsOperation,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
    ) -> LoopMeasure | SkipMeasure:
        """Measure width of a ForItemsOperation (fold or unfold)."""
        affected_qubits = self._analyze_loop_affected_qubits(
            op, qubit_map, logical_id_remap, param_values
        )

        # Access DictValue from operands
        dict_value = op.operands[0] if op.operands else None

        # Build label for folded mode
        key_str = ", ".join(op.key_vars) if op.key_vars else "k"
        if len(op.key_vars) > 1:
            key_str = f"({key_str})"
        dict_name = getattr(dict_value, "name", "dict") if dict_value else "dict"
        label = f"for {key_str}, {op.value_var} in {dict_name}"
        label_width = (
            len(label) * self.style.char_width_monospace + 2 * self.style.text_padding
        )
        folded_width = max(label_width, self.style.folded_loop_width)

        # Materialize entries from DictValue (handles bound_data in params)
        materialized = (
            self._materialize_dict_entries(dict_value) if dict_value else None
        )

        if self.fold_loops or materialized is None:
            return LoopMeasure(
                fold=True,
                affected_qubits=affected_qubits,
                folded_width=folded_width,
            )

        # Unfolded: iterate materialized entries
        entries = materialized
        num_iterations = len(entries)
        if num_iterations == 0:
            return SkipMeasure()

        iteration_children: list[list[MeasureNode]] = []
        iteration_widths: list[float] = []
        iter_param_values_list: list[dict] = []

        for entry_key, entry_value in entries:
            child_param_values = dict(param_values)
            # Set key variables — handle both IR Values and raw Python objects
            if hasattr(entry_key, "elements"):
                # TupleValue key (IR)
                for key_var, elem in zip(op.key_vars, entry_key.elements):
                    val = elem.get_const() if hasattr(elem, "get_const") else None
                    if val is not None:
                        child_param_values[f"_loop_{key_var}"] = val
            elif isinstance(entry_key, tuple):
                # Raw Python tuple key (from bound_data)
                for key_var, elem in zip(op.key_vars, entry_key):
                    child_param_values[f"_loop_{key_var}"] = elem
            else:
                # Single key
                if op.key_vars:
                    if hasattr(entry_key, "get_const"):
                        val = entry_key.get_const()
                    else:
                        val = entry_key
                    if val is not None:
                        child_param_values[f"_loop_{op.key_vars[0]}"] = val
            # Set value variable — handle both IR Value and raw Python
            if hasattr(entry_value, "get_const"):
                val = entry_value.get_const()
            else:
                val = entry_value
            if val is not None:
                child_param_values[f"_loop_{op.value_var}"] = val

            children = self._measure_operations(
                op.operations,
                qubit_map,
                logical_id_remap,
                child_param_values,
                depth + 1,
            )
            iter_width = self._sum_children_widths(children)
            iteration_children.append(children)
            iteration_widths.append(iter_width)
            iter_param_values_list.append(child_param_values)

        return LoopMeasure(
            fold=False,
            affected_qubits=affected_qubits,
            iteration_children=iteration_children,
            iteration_widths=iteration_widths,
            num_iterations=num_iterations,
            iter_param_values=iter_param_values_list,
        )

    def _measure_operations(
        self,
        ops: list[Operation],
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str] | None = None,
        param_values: dict | None = None,
        depth: int = 0,
    ) -> list[MeasureNode]:
        """Measure Phase: compute widths for all operations without placing them."""
        if logical_id_remap is None:
            logical_id_remap = {}
        if param_values is None:
            param_values = {}

        result: list[MeasureNode] = []

        for op in ops:
            if isinstance(op, QInitOperation):
                result.append(SkipMeasure())
                continue

            if isinstance(op, ExpvalOp):
                result.append(
                    self._measure_generic_operation(
                        op, qubit_map, logical_id_remap, param_values
                    )
                )
                continue

            if isinstance(op, WhileOperation):
                result.append(
                    self._measure_while_operation(
                        op, qubit_map, logical_id_remap, param_values, depth
                    )
                )
                continue

            if isinstance(op, IfOperation):
                result.append(
                    self._measure_if_operation(
                        op, qubit_map, logical_id_remap, param_values, depth
                    )
                )
                continue

            if isinstance(op, ForItemsOperation):
                result.append(
                    self._measure_for_items_operation(
                        op, qubit_map, logical_id_remap, param_values, depth
                    )
                )
                continue

            if isinstance(op, CastOperation):
                result.append(SkipMeasure())
                continue

            if isinstance(op, ForOperation):
                result.append(
                    self._measure_for_operation(
                        op, qubit_map, logical_id_remap, param_values, depth
                    )
                )
                continue

            if isinstance(op, CallBlockOperation) and self._should_inline_at_depth(
                depth
            ):
                result.append(
                    self._measure_inline_block(
                        op, qubit_map, logical_id_remap, param_values, depth
                    )
                )
                continue

            if isinstance(op, ControlledUOperation) and self._should_inline_at_depth(
                depth
            ):
                result.append(
                    self._measure_inline_block(
                        op, qubit_map, logical_id_remap, param_values, depth
                    )
                )
                continue

            if (
                isinstance(op, CompositeGateOperation)
                and self.expand_composite
                and op.has_implementation
            ):
                result.append(
                    self._measure_inline_block(
                        op, qubit_map, logical_id_remap, param_values, depth
                    )
                )
                continue

            if isinstance(
                op,
                (
                    GateOperation,
                    CallBlockOperation,
                    MeasureOperation,
                    MeasureVectorOperation,
                    CompositeGateOperation,
                    ControlledUOperation,
                ),
            ):
                result.append(
                    self._measure_generic_operation(
                        op, qubit_map, logical_id_remap, param_values
                    )
                )

        return result

    # ------------------------------------------------------------------
    # Place Phase: assign coordinates using pre-computed widths
    # ------------------------------------------------------------------

    def _place_generic_operation(
        self,
        measure: GateMeasure,
        state: LayoutState,
        op: Operation,
        op_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> None:
        """Place a generic operation using pre-computed width from Measure Phase."""
        # Find which qubits this operation touches
        affected_qubits: list[int] = []
        for i, operand in enumerate(op.operands):
            if isinstance(op, CallBlockOperation) and i == 0:
                continue
            if isinstance(op, ControlledUOperation) and i == 0:
                continue
            if (
                isinstance(op, CompositeGateOperation)
                and op.has_implementation
                and i == 0
            ):
                continue
            affected_qubits.extend(
                self._resolve_operand_to_qubit_indices(
                    operand, qubit_map, logical_id_remap, param_values
                )
            )

        # Find the earliest column where all affected qubits are free
        if affected_qubits:
            min_column = max(state.qubit_columns[q] for q in affected_qubits)
        else:
            min_column = state.column

        op_half_width = measure.half_width
        columns_needed = measure.columns_needed

        # Store block_widths for block-box mode operations
        if measure.is_block_box and measure.box_width is not None:
            state.block_widths[op_key] = measure.box_width

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
        state.positions[op_key] = min_column
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
        if isinstance(op, MeasureOperation):
            for q in affected_qubits:
                self.qubit_end_positions[q] = min_column + op_half_width
        elif isinstance(op, MeasureVectorOperation):
            measure_qubits = list(affected_qubits)
            operand = op.operands[0] if op.operands else None
            if operand is not None and isinstance(operand, ArrayValue):
                parent_lid = operand.logical_id
                if parent_lid in qubit_map:
                    base_idx = qubit_map[parent_lid]
                    if len(operand.shape) == 1:
                        size_value = operand.shape[0]
                        if size_value.is_constant():
                            array_size = size_value.get_const()
                            if array_size is not None and isinstance(array_size, int):
                                measure_qubits = list(
                                    range(base_idx, base_idx + array_size)
                                )
            for q in measure_qubits:
                self.qubit_end_positions[q] = min_column + op_half_width

        # Update actual_width
        op_actual_width = min_column + op_half_width + 0.5
        state.actual_width = max(state.actual_width, op_actual_width)

    def _place_inline_block(
        self,
        measure: BlockMeasure,
        state: LayoutState,
        op: CallBlockOperation | ControlledUOperation | CompositeGateOperation,
        op_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
        place_fn: Callable[..., None],
    ) -> None:
        """Place an inlined block using pre-computed widths."""
        affected_qubits = measure.affected_qubits

        # Fix Issue 002: Align all affected qubits to the maximum right edge
        if affected_qubits:
            max_edge = max(state.qubit_right_edges.get(q, 0.0) for q in affected_qubits)
            for q in affected_qubits:
                state.qubit_right_edges[q] = max_edge
                state.qubit_columns[q] = max_edge + self.style.gate_gap

        # Advance for border extent (left padding)
        border_padding = measure.border_padding
        max_gate_width = measure.max_gate_width
        border_extent = max_gate_width / 2 + border_padding

        # Compute first child's half-width for accurate left offset
        first_child_half = self.style.gate_width / 2  # fallback
        if measure.children:
            fc = measure.children[0]
            if isinstance(fc, GateMeasure):
                first_child_half = fc.half_width
            elif isinstance(fc, BlockMeasure):
                first_child_half = fc.final_width / 2
            elif (
                isinstance(fc, LoopMeasure) and fc.fold and fc.folded_width is not None
            ):
                first_child_half = fc.folded_width / 2
            elif isinstance(fc, IfMeasure) and fc.fold and fc.folded_width is not None:
                first_child_half = fc.folded_width / 2

        for q in affected_qubits:
            if q in state.qubit_right_edges:
                state.qubit_right_edges[q] += border_extent - first_child_half
                state.qubit_columns[q] = (
                    state.qubit_right_edges[q] + self.style.gate_gap
                )

        # Fix Issue 001: Center content when label is wider
        center_offset = 0.0
        if measure.final_width > measure.content_width:
            center_offset = (measure.final_width - measure.content_width) / 2
            for q in affected_qubits:
                state.qubit_right_edges[q] += center_offset
                state.qubit_columns[q] = (
                    state.qubit_right_edges[q] + self.style.gate_gap
                )

        state.max_depth = max(state.max_depth, depth + 1)

        # Build block mappings for recursive placement
        if isinstance(op, CallBlockOperation):
            block_value = op.operands[0]
            assert isinstance(block_value, BlockValue), (
                "[FOR DEVELOPER] CallBlockOperation.operands[0] must be a BlockValue. "
                "If this assertion fails, there is a bug in the IR construction."
            )
            actual_inputs = op.operands[1:]
        elif isinstance(op, ControlledUOperation):
            block_value = op.block
            assert isinstance(block_value, BlockValue), (
                "[FOR DEVELOPER] ControlledUOperation.block must be a BlockValue. "
                "If this assertion fails, there is a bug in the IR construction."
            )
            actual_inputs = list(op.target_operands)
        elif isinstance(op, CompositeGateOperation):
            block_value = op.implementation
            assert isinstance(block_value, BlockValue), (
                "[FOR DEVELOPER] CompositeGateOperation.implementation must be a BlockValue. "
                "If this assertion fails, there is a bug in the IR construction."
            )
            actual_inputs = list(op.target_qubits)
        else:
            raise TypeError(
                f"Unsupported operation type for inline block placement: {type(op).__name__}"
            )

        new_logical_id_remap, child_param_values = self._build_block_value_mappings(
            block_value,
            actual_inputs,
            logical_id_remap,
            param_values,
            qubit_map=qubit_map,
        )

        # Recursively place block content
        place_fn(
            block_value.operations,
            measure.children,
            qubit_map,
            state,
            new_logical_id_remap,
            child_param_values,
            depth + 1,
            op_key,
        )

        # Compute block range from placed children
        block_op_columns = []
        op_key_len = len(op_key)
        for pos_key, pos_val in state.positions.items():
            if (
                len(pos_key) > op_key_len
                and pos_key[:op_key_len] == op_key
                and pos_val > 0
            ):
                block_op_columns.append(pos_val)

        if block_op_columns:
            actual_start = min(block_op_columns)
            actual_end = max(block_op_columns)
        else:
            # Empty block: use current qubit positions
            if affected_qubits:
                actual_start = min(
                    state.qubit_columns.get(q, 0) for q in affected_qubits
                )
                actual_end = actual_start
            else:
                actual_start = state.column
                actual_end = state.column

        # Expand actual_start/actual_end for children with explicit box widths
        # (e.g., folded loop boxes that are wider than max_gate_width)
        for pos_key, pos_val in state.positions.items():
            if (
                len(pos_key) > op_key_len
                and pos_key[:op_key_len] == op_key
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
                    "name": measure.label,
                    "start_x": actual_start,
                    "end_x": actual_end,
                    "qubit_indices": affected_qubits,
                    "control_qubit_indices": measure.control_qubit_indices,
                    "depth": depth,
                    "max_gate_width": max_gate_width,
                }
            )

        # Expand first_gate_half_width to cover border extent
        if state.first_gate_x is not None and affected_qubits:
            border_left = actual_start - max_gate_width / 2 - border_padding
            current_left = state.first_gate_x - state.first_gate_half_width
            if border_left < current_left:
                state.first_gate_half_width = state.first_gate_x - border_left

        # Compute border right edge using shared helper
        _, border_right_edge = self._compute_block_box_bounds(
            measure.label, actual_start, actual_end, depth, max_gate_width
        )

        for q in affected_qubits:
            state.qubit_right_edges[q] = border_right_edge
            state.qubit_columns[q] = border_right_edge + self.style.gate_gap

    def _place_for_operation(
        self,
        measure: LoopMeasure,
        state: LayoutState,
        op: ForOperation,
        op_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
        place_fn: Callable[..., None],
    ) -> None:
        """Place a ForOperation using pre-computed widths."""
        affected_qubits = measure.affected_qubits

        if measure.fold:
            # Folded mode: same as current logic
            if affected_qubits:
                start_column = max(state.qubit_columns[q] for q in affected_qubits)
            elif state.qubit_columns:
                start_column = max(state.qubit_columns.values())
            else:
                start_column = 0

            loop_width = measure.folded_width or self.style.folded_loop_width
            op_half_width = loop_width / 2
            gap = self.style.gate_gap

            for q in affected_qubits:
                if q in state.qubit_right_edges:
                    required_center = state.qubit_right_edges[q] + gap + op_half_width
                    start_column = max(start_column, required_center)

            state.positions[op_key] = start_column
            state.block_widths[op_key] = loop_width

            if state.first_gate_x is None:
                state.first_gate_x = start_column
                state.first_gate_half_width = op_half_width

            for q in affected_qubits:
                state.qubit_columns[q] = start_column + op_half_width + gap
                state.qubit_right_edges[q] = start_column + op_half_width

            state.column = max(state.column, start_column + 1)
            state.actual_width = max(
                state.actual_width, start_column + op_half_width + 0.5
            )
        else:
            # Unfolded mode: place each iteration with uniform width
            assert measure.iteration_children is not None, (
                "[FOR DEVELOPER] LoopMeasure.iteration_children must be populated in unfolded mode. "
                "The _measure_for_operation method did not set this field."
            )
            assert measure.iter_param_values is not None, (
                "[FOR DEVELOPER] LoopMeasure.iter_param_values must be populated in unfolded mode. "
                "The _measure_for_operation method did not set this field."
            )

            for iteration in range(measure.num_iterations):
                iter_children = measure.iteration_children[iteration]
                child_param_values = measure.iter_param_values[iteration]

                # Place iteration operations
                place_fn(
                    op.operations,
                    iter_children,
                    qubit_map,
                    state,
                    logical_id_remap,
                    child_param_values,
                    depth + 1,
                    (*op_key, iteration),
                )

    def _place_if_operation(
        self,
        measure: IfMeasure,
        state: LayoutState,
        op: IfOperation,
        op_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
        place_fn: Callable[..., None],
    ) -> None:
        """Place an IfOperation using pre-computed widths."""
        affected_qubits = measure.affected_qubits

        if measure.fold:
            # Folded mode: same pattern as ForOperation folded
            if affected_qubits:
                start_column = max(state.qubit_columns[q] for q in affected_qubits)
            elif state.qubit_columns:
                start_column = max(state.qubit_columns.values())
            else:
                start_column = 0

            loop_width = measure.folded_width or self.style.folded_loop_width
            op_half_width = loop_width / 2
            gap = self.style.gate_gap

            for q in affected_qubits:
                if q in state.qubit_right_edges:
                    required_center = state.qubit_right_edges[q] + gap + op_half_width
                    start_column = max(start_column, required_center)

            state.positions[op_key] = start_column
            state.block_widths[op_key] = loop_width

            if state.first_gate_x is None:
                state.first_gate_x = start_column
                state.first_gate_half_width = op_half_width

            for q in affected_qubits:
                state.qubit_columns[q] = start_column + op_half_width + gap
                state.qubit_right_edges[q] = start_column + op_half_width

            state.column = max(state.column, start_column + 1)
            state.actual_width = max(
                state.actual_width, start_column + op_half_width + 0.5
            )
        else:
            # Unfolded: place true branch, then false branch
            assert measure.true_children is not None, (
                "[FOR DEVELOPER] IfMeasure.true_children must be populated in unfolded mode. "
                "The _measure_if_operation method did not set this field."
            )
            assert measure.false_children is not None, (
                "[FOR DEVELOPER] IfMeasure.false_children must be populated in unfolded mode. "
                "The _measure_if_operation method did not set this field."
            )

            # Place true branch
            if affected_qubits:
                max_edge = max(
                    state.qubit_right_edges.get(q, 0.0) for q in affected_qubits
                )
                for q in affected_qubits:
                    state.qubit_right_edges[q] = max_edge
                    state.qubit_columns[q] = max_edge + self.style.gate_gap

            place_fn(
                op.true_operations,
                measure.true_children,
                qubit_map,
                state,
                logical_id_remap,
                param_values,
                depth + 1,
                (*op_key, "true"),
            )

            # Place false branch
            if op.false_operations:
                if affected_qubits:
                    max_edge = max(
                        state.qubit_right_edges.get(q, 0.0) for q in affected_qubits
                    )
                    for q in affected_qubits:
                        state.qubit_right_edges[q] = max_edge
                        state.qubit_columns[q] = max_edge + self.style.gate_gap

                place_fn(
                    op.false_operations,
                    measure.false_children,
                    qubit_map,
                    state,
                    logical_id_remap,
                    param_values,
                    depth + 1,
                    (*op_key, "false"),
                )

    @staticmethod
    def _assert_measure_place(
        measure_node: MeasureNode,
        expected_type: type,
        op_name: str,
    ) -> None:
        """Assert measure_node matches expected type in Place Phase."""
        assert isinstance(measure_node, expected_type), (
            f"[FOR DEVELOPER] Measure-Place mismatch: expected {expected_type.__name__} "
            f"for {op_name}, got {type(measure_node).__name__}. "
            "The _measure_operations and _place_operations dispatchers are out of sync."
        )

    def _place_operations(
        self,
        ops: list[Operation],
        measure_nodes: list[MeasureNode],
        qubit_map: dict[str, int],
        state: LayoutState,
        logical_id_remap: dict[str, str] | None = None,
        param_values: dict | None = None,
        depth: int = 0,
        scope_path: tuple = (),
    ) -> None:
        """Place Phase: assign coordinates using pre-computed widths."""
        if logical_id_remap is None:
            logical_id_remap = {}
        if param_values is None:
            param_values = {}

        state.max_depth = max(state.max_depth, depth)

        measure_iter = iter(measure_nodes)

        for op in ops:
            op_key = (*scope_path, id(op))

            if isinstance(op, QInitOperation):
                state.positions[op_key] = 0
                next(measure_iter)  # consume SkipMeasure
                continue

            if isinstance(op, ExpvalOp):
                measure_node = next(measure_iter)
                self._assert_measure_place(measure_node, GateMeasure, "ExpvalOp")
                self._place_generic_operation(
                    measure_node,
                    state,
                    op,
                    op_key,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                )
                continue

            if isinstance(op, WhileOperation):
                measure_node = next(measure_iter)
                self._assert_measure_place(measure_node, LoopMeasure, "WhileOperation")
                self._place_for_operation(
                    measure_node,
                    state,
                    op,
                    op_key,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                    depth,
                    self._place_operations,
                )
                continue

            if isinstance(op, IfOperation):
                measure_node = next(measure_iter)
                self._assert_measure_place(measure_node, IfMeasure, "IfOperation")
                self._place_if_operation(
                    measure_node,
                    state,
                    op,
                    op_key,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                    depth,
                    self._place_operations,
                )
                continue

            if isinstance(op, ForItemsOperation):
                measure_node = next(measure_iter)
                if isinstance(measure_node, SkipMeasure):
                    continue
                self._assert_measure_place(
                    measure_node, LoopMeasure, "ForItemsOperation"
                )
                self._place_for_operation(
                    measure_node,
                    state,
                    op,
                    op_key,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                    depth,
                    self._place_operations,
                )
                continue

            if isinstance(op, CastOperation):
                next(measure_iter)  # consume SkipMeasure
                continue

            if isinstance(op, ForOperation):
                measure_node = next(measure_iter)
                if isinstance(measure_node, SkipMeasure):
                    continue  # zero-iteration loop
                self._assert_measure_place(measure_node, LoopMeasure, "ForOperation")
                self._place_for_operation(
                    measure_node,
                    state,
                    op,
                    op_key,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                    depth,
                    self._place_operations,
                )
                continue

            if isinstance(op, CallBlockOperation) and self._should_inline_at_depth(
                depth
            ):
                self._inlined_op_keys.add(op_key)
                measure_node = next(measure_iter)
                self._assert_measure_place(
                    measure_node, BlockMeasure, "inlined CallBlockOperation"
                )
                self._place_inline_block(
                    measure_node,
                    state,
                    op,
                    op_key,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                    depth,
                    self._place_operations,
                )
                continue

            if isinstance(op, ControlledUOperation) and self._should_inline_at_depth(
                depth
            ):
                self._inlined_op_keys.add(op_key)
                measure_node = next(measure_iter)
                self._assert_measure_place(
                    measure_node, BlockMeasure, "inlined ControlledUOperation"
                )
                self._place_inline_block(
                    measure_node,
                    state,
                    op,
                    op_key,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                    depth,
                    self._place_operations,
                )
                continue

            if (
                isinstance(op, CompositeGateOperation)
                and self.expand_composite
                and op.has_implementation
            ):
                self._inlined_op_keys.add(op_key)
                measure_node = next(measure_iter)
                self._assert_measure_place(
                    measure_node, BlockMeasure, "expanded CompositeGateOperation"
                )
                self._place_inline_block(
                    measure_node,
                    state,
                    op,
                    op_key,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                    depth,
                    self._place_operations,
                )
                continue

            if isinstance(
                op,
                (
                    GateOperation,
                    CallBlockOperation,
                    MeasureOperation,
                    MeasureVectorOperation,
                    CompositeGateOperation,
                    ControlledUOperation,
                ),
            ):
                measure_node = next(measure_iter)
                self._assert_measure_place(measure_node, GateMeasure, type(op).__name__)
                self._place_generic_operation(
                    measure_node,
                    state,
                    op,
                    op_key,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                )

    def _layout_operations(
        self, graph: Graph, qubit_map: dict[str, int]
    ) -> dict[str, int | dict]:
        """Calculate layout positions for operations using 2-phase approach.

        Phase 1 (Measure): Recursively compute widths without assigning coordinates.
        Phase 2 (Place): Assign coordinates using pre-computed widths.

        Args:
            graph: IR Graph containing operations to lay out.
            qubit_map: Mapping from logical_id to qubit wire index.

        Returns:
            Layout dictionary with:
            - 'width': Total circuit width in coordinate units
            - 'positions': dict mapping operation key to x positions
            - 'block_ranges': list[dict] block boundary information (when inline=True)
            - 'max_depth': Maximum nesting depth of blocks
            - 'block_widths': dict mapping operation key to box widths
            - 'actual_width': Actual circuit width considering box widths
        """
        # Phase 1: Measure — compute widths
        measure_nodes = self._measure_operations(graph.operations, qubit_map)

        # Phase 2: Place — assign coordinates
        state = LayoutState()
        for q_idx in set(qubit_map.values()):
            state.qubit_right_edges[q_idx] = self.style.initial_wire_position
            _ = state.qubit_columns[q_idx]

        self._place_operations(graph.operations, measure_nodes, qubit_map, state)

        state.actual_width = max(state.actual_width, state.column)
        return {
            "width": state.column,
            "positions": state.positions,
            "block_ranges": state.block_ranges,
            "max_depth": state.max_depth,
            "block_widths": state.block_widths,
            "actual_width": state.actual_width,
            "first_gate_x": state.first_gate_x
            if state.first_gate_x is not None
            else 1.0,
            "first_gate_half_width": state.first_gate_half_width,
        }

    def _create_figure(self, num_qubits: int, layout: dict) -> Figure:
        """Create matplotlib figure with appropriate size.

        Args:
            num_qubits: Number of qubits.
            layout: Layout dictionary.

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

        width = layout["width"]
        block_ranges = layout.get("block_ranges", [])

        # Use Figure directly to avoid pyplot auto-display in Jupyter
        fig = Figure(figsize=(6, 2))  # Temporary size, will be updated
        # Attach canvas for Jupyter display
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        # Dynamic vertical margins based on actual block border extents
        base_margin = 0.6  # minimum: gate half-height + comfortable padding
        clearance = 0.2  # extra breathing room beyond the border edge

        if self._max_above:
            y_margin_top = max(self._max_above.get(0, 0), base_margin) + clearance
        else:
            y_margin_top = base_margin

        if self._max_below:
            y_margin_bottom = (
                max(self._max_below.get(num_qubits - 1, 0), base_margin) + clearance
            )
        else:
            y_margin_bottom = base_margin

        y_top = self.qubit_y[0] + y_margin_top
        y_bottom = self.qubit_y[num_qubits - 1] - y_margin_bottom

        # Dynamic horizontal limits based on block border extents
        # Use first gate position and wire extension for symmetric left margin
        first_gate_x = layout.get("first_gate_x", 1.0)
        wire_ext = self.style.wire_extension
        # Left limit: wire extends wire_extension past first gate, plus label space
        first_gate_hw = layout.get("first_gate_half_width", self.style.gate_width / 2)
        x_left = first_gate_x - first_gate_hw - wire_ext - 0.7
        x_right = max(width + 0.5, layout.get("actual_width", width) + wire_ext + 0.2)

        if block_ranges:
            for br in block_ranges:
                depth = br["depth"]
                padding = self._compute_border_padding(depth)
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
        positions = layout.get("positions", {})
        block_widths_dict = layout.get("block_widths", {})
        for key, bw in block_widths_dict.items():
            if key in positions:
                box_left = positions[key] - bw / 2
                box_right = positions[key] + bw / 2
                x_left = min(x_left, box_left - 0.3)
                x_right = max(x_right, box_right + 0.3)

        # Extend x_right if output_names are present (for right-side labels)
        if self.graph.output_names:
            max_label_len = max(len(name) for name in self.graph.output_names)
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
        layout: dict,
        qubit_map: dict[str, int],
    ) -> None:
        """Draw qubit wires and labels.

        Args:
            fig: matplotlib Figure.
            num_qubits: Number of qubits.
            layout: Layout dictionary.
            qubit_map: Qubit to wire index mapping.
        """
        ax = fig._qm_ax
        # Compute wire start/end symmetrically from gate edges
        actual_width = layout.get("actual_width", layout["width"])
        first_gate_x = layout.get("first_gate_x", 1.0)
        wire_ext = self.style.wire_extension
        first_gate_hw = layout.get("first_gate_half_width", self.style.gate_width / 2)
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
        """
        new_logical_id_remap, child_param_values = self._build_block_value_mappings(
            block_value, actual_inputs, logical_id_remap, param_values
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
        start_val, stop_val_raw, step_val = self._evaluate_loop_range(op, param_values)

        if self._is_zero_iteration_loop(start_val, stop_val_raw, step_val):
            return

        if self.fold_loops:
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

            num_iterations = self._compute_loop_iterations(
                start_val, stop_val_raw, step_val
            )
            for iteration in range(num_iterations):
                iter_value = start_val + iteration * step_val
                child_param_values = dict(param_values)
                child_param_values[f"_loop_{op.loop_var}"] = iter_value

                self._evaluate_loop_body_intermediates(
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
        layout: dict,
    ) -> None:
        """Draw all operations.

        Args:
            fig: matplotlib Figure.
            graph: Computation graph.
            qubit_map: Qubit to wire index mapping.
            layout: Layout dictionary.
        """
        positions = layout["positions"]
        block_ranges = layout.get("block_ranges", [])
        block_widths = layout.get("block_widths", {})

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
                            self._compute_block_box_bounds(
                                inner.get("name", "block"),
                                inner["start_x"],
                                inner["end_x"],
                                inner["depth"],
                                inner_mgw,
                            )
                        )
                        outer_mgw = outer.get("max_gate_width", self.style.gate_width)
                        outer_padding = self._compute_border_padding(outer["depth"])
                        outer_box_left = (
                            outer["start_x"] - outer_mgw / 2 - outer_padding
                        )
                        _, outer_box_right = self._compute_block_box_bounds(
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
                    bl, br = self._compute_block_box_bounds(
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

    def _compute_qubit_y_positions(
        self,
        num_qubits: int,
        block_ranges: list[dict],
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
            List indexed by qubit index. Qubit 0 is at highest y (top).
        """
        if not block_ranges:
            self._max_above = {}
            self._max_below = {}
            if num_qubits <= 1:
                return [0.0] * num_qubits
            return [float(num_qubits - 1 - q) for q in range(num_qubits)]

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

            # Below the bottommost qubit
            extent_below = gate_half + padding
            max_below[bottom_q] = max(max_below[bottom_q], extent_below)

            # Above the topmost qubit (includes label + offsets)
            depth_offset = (max_depth - depth) * label_step
            overlap_offset = max(0, qubit_max_overlaps.get(top_q, 0) - 1) * overlap_step
            extent_above = (
                gate_half + padding + label_height + depth_offset + overlap_offset
            )
            max_above[top_q] = max(max_above[top_q], extent_above)

            # Middle/non-top/non-bottom qubits: just gate+padding
            for q in qubits:
                if q != top_q:
                    max_above[q] = max(max_above[q], gate_half + padding)
                if q != bottom_q:
                    max_below[q] = max(max_below[q], gate_half + padding)

        # Store extents for use by _create_figure
        self._max_above = max_above
        self._max_below = max_below

        if num_qubits <= 1:
            return [0.0] * num_qubits

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

        # Flip: qubit 0 at top (highest y), qubit N-1 at bottom (y≈0)
        max_val = y_positions[-1]
        for i in range(num_qubits):
            y_positions[i] = max_val - y_positions[i]

        return y_positions

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
        padding = self._compute_border_padding(depth)

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
        box_left, box_right = self._compute_block_box_bounds(
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
            idx = self._resolve_operand_to_qubit_index(
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
            estimated_width = self._estimate_gate_width(op, param_values)
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
        label, has_param = self._get_gate_label(op, param_values)
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
        label, has_param = self._get_gate_label(op, param_values)
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
                self._resolve_operand_to_qubit_indices(
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
        label = self._get_block_label(op, qubit_map)

        # Use pre-calculated width from layout phase if available
        # This ensures consistency between layout and rendering
        if block_width is not None:
            width = block_width
        else:
            width = self._estimate_label_box_width(label)

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
            idx = self._resolve_operand_to_qubit_index(
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
            idx = self._resolve_operand_to_qubit_index(
                qval, qubit_map, logical_id_remap, param_values
            )
            if idx is not None:
                control_y.append(self.qubit_y[idx])

        # Resolve target qubit y-coordinates
        target_y = []
        for qval in op.target_operands:
            idx = self._resolve_operand_to_qubit_index(
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

        qubit_idx = self._resolve_operand_to_qubit_index(
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

        qubit_indices = self._resolve_operand_to_qubit_indices(
            op.operands[0], qubit_map, logical_id_remap or {}
        )
        ax = fig._qm_ax
        for qubit_idx in qubit_indices:
            self._draw_measurement_box(ax, x_pos, self.qubit_y[qubit_idx])

    def _get_gate_params_for_expression(self, op: GateOperation) -> str | None:
        """Get gate parameters as a string for expression format.

        Args:
            op: GateOperation.

        Returns:
            Parameter string (e.g., "0.5") or None if no parameters.
        """
        # Parameterized gates have additional operands beyond qubits
        # Check for common parameterized gates
        param_gates = {
            GateOperationType.RX,
            GateOperationType.RY,
            GateOperationType.RZ,
            GateOperationType.RZZ,
            GateOperationType.P,
            GateOperationType.CP,
        }

        if op.gate_type not in param_gates:
            return None

        # Check theta attribute first (rotation gates store angle here)
        if hasattr(op, "theta") and op.theta is not None:
            theta = op.theta
            if isinstance(theta, (int, float)):
                return self._format_parameter(theta)
            elif isinstance(theta, Value):
                const = theta.get_const()
                if const is not None:
                    return self._format_parameter(const)
                elif theta.is_parameter():
                    param_name = theta.parameter_name() or "θ"
                    return self._format_symbolic_param(param_name)

        # Find parameter values (non-qubit operands)
        params = []
        for operand in op.operands:
            if hasattr(operand, "get_const"):
                const = operand.get_const()
                if const is not None and isinstance(const, (int, float)):
                    params.append(self._format_parameter(const))
                elif hasattr(operand, "is_parameter") and operand.is_parameter():
                    param_name = operand.parameter_name() or "θ"
                    params.append(self._format_symbolic_param(param_name))

        if params:
            return ", ".join(params)
        return None

    def _format_operation_as_expression(
        self,
        op: Operation,
        loop_vars: set[str],
        indent: int = 0,
        max_depth: int = 2,
        param_values: dict | None = None,
        body_operations: list | None = None,
    ) -> str | None:
        """Convert an operation to expression format string.

        Args:
            op: Operation to format.
            loop_vars: Set of loop variable names in scope (e.g., {"i", "j"}).
            indent: Indentation level (2 spaces per level).
            max_depth: Maximum nesting depth for recursive formatting.
            param_values: Parameter values for resolving symbolic expressions.

        Returns:
            Expression string (e.g., "q[i],q[j] = cx(q[i],q[j])") or None.
        """
        prefix = "  " * indent

        if isinstance(op, GateOperation):
            gate_name = op.gate_type.name.lower()

            if not op.operands:
                return None

            # Collect all qubit operand strings
            qubit_strs = []
            for operand in op.operands:
                if (
                    hasattr(operand, "parent_array")
                    and operand.parent_array is not None
                ):
                    array_name = operand.parent_array.name or "qubits"
                    idx_str = self._resolve_index_expression(
                        operand, loop_vars, body_operations
                    )
                    qubit_strs.append(f"{array_name}[{idx_str}]")
                elif hasattr(operand, "name") and operand.name:
                    qubit_strs.append(operand.name)

            if not qubit_strs:
                return None

            params = self._get_gate_params_for_expression(op)
            result_str = ",".join(qubit_strs)
            args_str = ",".join(qubit_strs)
            if params:
                args_str += f", {params}"
            return f"{prefix}{result_str} = {gate_name}({args_str})"

        elif isinstance(op, (MeasureOperation, MeasureVectorOperation)):
            if not op.operands:
                return None

            operand = op.operands[0]
            if hasattr(operand, "parent_array") and operand.parent_array is not None:
                array_name = operand.parent_array.name or "qubits"
                idx_str = self._resolve_index_expression(
                    operand, loop_vars, body_operations
                )
                return f"{prefix}measure({array_name}[{idx_str}])"
            elif hasattr(operand, "name") and operand.name:
                return f"{prefix}measure({operand.name})"
            return f"{prefix}measure(...)"

        elif isinstance(op, CallBlockOperation):
            block_value = op.operands[0]
            assertion_message = (
                "[FOR DEVELOPER] CallBlockOperation.operands[0] must be a BlockValue. "
                "If this assertion fails, there is a bug in the IR construction."
            )
            assert isinstance(block_value, BlockValue), assertion_message
            block_name = block_value.name or "block"

            if len(op.operands) > 1:
                operand = op.operands[1]
                if (
                    hasattr(operand, "parent_array")
                    and operand.parent_array is not None
                ):
                    array_name = operand.parent_array.name or "qubits"
                    idx_str = self._resolve_index_expression(
                        operand, loop_vars, body_operations
                    )
                    return f"{prefix}{array_name}[{idx_str}] = {block_name}({array_name}[{idx_str}])"
            return f"{prefix}{block_name}(...)"

        elif isinstance(op, ForOperation):
            if max_depth <= 0:
                return f"{prefix}..."

            range_str = self._format_range_str(op, loop_vars, param_values or {})
            header = f"{prefix}for {op.loop_var} in {range_str}:"
            inner_vars = loop_vars | {op.loop_var}

            lines = [header]
            for body_op in op.operations:
                expr = self._format_operation_as_expression(
                    body_op,
                    inner_vars,
                    indent + 1,
                    max_depth - 1,
                    param_values,
                    body_operations=op.operations,
                )
                if expr:
                    lines.append(expr)
            return "\n".join(lines)

        elif isinstance(op, WhileOperation):
            return f"{prefix}while ...:"

        elif isinstance(op, IfOperation):
            return f"{prefix}if ...:"

        elif isinstance(op, ForItemsOperation):
            key_str = ", ".join(op.key_vars) if op.key_vars else "k"
            return f"{prefix}for {key_str}, {op.value_var} in ...:"

        return None

    def _resolve_index_expression(
        self,
        operand: Value,
        loop_vars: set[str],
        operations: list | None = None,
    ) -> str:
        """Resolve an operand's element index to a human-readable string.

        Args:
            operand: A Value that may be an array element.
            loop_vars: Set of loop variable names in scope.
            operations: Operations list to search for BinOp definitions.

        Returns:
            Index expression string (e.g., "i", "i+1", "1").
        """
        if not hasattr(operand, "element_indices") or not operand.element_indices:
            return next(iter(loop_vars)) if loop_vars else "?"

        idx_value = operand.element_indices[0]

        # Constant index (e.g., j = 1; qubits[j])
        const = idx_value.get_const()
        if const is not None:
            return (
                str(int(const))
                if isinstance(const, float) and const == int(const)
                else str(const)
            )

        # Try to find the defining BinOp operation for this value
        return self._format_value_as_expression(idx_value, loop_vars, operations)

    def _format_value_as_expression(
        self,
        value: Value,
        loop_vars: set[str],
        operations: list | None = None,
    ) -> str:
        """Format a Value as a human-readable expression string.

        Recursively resolves BinOp chains to produce expressions like "i+1", "2*i".

        Args:
            value: IR Value to format.
            loop_vars: Set of loop variable names in scope.
            operations: Operations list to search for BinOp definitions.
                If None, falls back to self.graph.operations.

        Returns:
            Human-readable expression string.
        """
        # Constant
        if value.is_constant():
            c = value.get_const()
            if c is not None:
                return str(int(c)) if isinstance(c, float) and c == int(c) else str(c)

        # Loop variable
        if value.is_parameter():
            name = value.parameter_name()
            if name:
                return name

        # Check if this value is a loop variable by name
        if hasattr(value, "name") and value.name in loop_vars:
            return value.name

        # Search for the defining BinOp in the provided or graph-level operations
        if operations is None:
            graph = getattr(self, "graph", None)
            operations = graph.operations if graph else []
        for op in operations:
            if isinstance(op, BinOp) and op.results and id(op.results[0]) == id(value):
                lhs_str = self._format_value_as_expression(
                    op.lhs, loop_vars, operations
                )
                rhs_str = self._format_value_as_expression(
                    op.rhs, loop_vars, operations
                )
                op_symbol = {
                    BinOpKind.ADD: "+",
                    BinOpKind.SUB: "-",
                    BinOpKind.MUL: "*",
                    BinOpKind.FLOORDIV: "//",
                    BinOpKind.POW: "**",
                }.get(op.kind, "?")
                return f"{lhs_str}{op_symbol}{rhs_str}"

        return next(iter(loop_vars)) if loop_vars else "?"

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
        affected_qubits = self._analyze_loop_affected_qubits(
            op, qubit_map, logical_id_remap, param_values
        )

        if not affected_qubits:
            return

        # Convert to y coordinates using variable spacing
        y_coords = [self.qubit_y[q] for q in affected_qubits]
        min_y = min(y_coords)
        max_y = max(y_coords)

        # Get loop range for label
        range_str = self._format_range_str(op, set(), param_values)
        label = f"for {op.loop_var} in {range_str}"

        # Collect operation expressions from loop body
        expressions = []
        for body_op in op.operations:
            expr = self._format_operation_as_expression(
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
        ax.text(
            x_pos,
            header_y,
            label,
            ha="center",
            va="top",
            color=self.style.for_loop_text_color,
            fontsize=self.style.subfont_size,
            fontweight="bold",
            zorder=PORDER_TEXT,
        )

        # Draw operation text centered in remaining space
        if expressions:
            body_text = "\n".join(expressions)
            body_center_y = (box_bottom + header_y) / 2
            ax.text(
                x_pos,
                body_center_y,
                body_text,
                ha="center",
                va="center",
                color=self.style.for_loop_text_color,
                fontsize=self.style.subfont_size,
                family="monospace",
                zorder=PORDER_TEXT,
            )

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
        qubit_indices = self._resolve_operand_to_qubit_indices(
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

        affected_qubits = self._analyze_loop_affected_qubits(
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
            expr = self._format_operation_as_expression(
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
        if self.fold_loops:
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
        affected: set[int] = set()

        def collect_qubits(ops: list[Operation]) -> None:
            """Recursively collect qubit indices from operations into `affected` set."""
            for inner_op in ops:
                operands: list = []
                if isinstance(inner_op, GateOperation):
                    operands = list(inner_op.operands)
                elif isinstance(inner_op, CallBlockOperation):
                    operands = list(inner_op.operands[1:])
                elif isinstance(inner_op, ControlledUOperation):
                    operands = list(inner_op.operands[1:])
                elif isinstance(inner_op, CompositeGateOperation):
                    operands = list(inner_op.control_qubits) + list(
                        inner_op.target_qubits
                    )
                elif isinstance(inner_op, MeasureOperation):
                    operands = list(inner_op.operands[:1])
                elif isinstance(inner_op, ForOperation):
                    collect_qubits(inner_op.operations)
                    continue
                elif isinstance(inner_op, WhileOperation):
                    collect_qubits(inner_op.operations)
                    continue
                elif isinstance(inner_op, IfOperation):
                    collect_qubits(inner_op.true_operations)
                    collect_qubits(inner_op.false_operations)
                    continue
                for operand in operands:
                    idx = self._resolve_operand_to_qubit_index(
                        operand, qubit_map, logical_id_remap or {}, param_values
                    )
                    if idx is not None:
                        affected.add(idx)

        collect_qubits(op.true_operations)
        collect_qubits(op.false_operations)
        affected_qubits = list(affected)

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
            expr = self._format_operation_as_expression(
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
            self._materialize_dict_entries(dict_value) if dict_value else None
        )

        if self.fold_loops or materialized is None:
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

                self._evaluate_loop_body_intermediates(
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

        affected_qubits = self._analyze_loop_affected_qubits(
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
            expr = self._format_operation_as_expression(
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

    def _format_parameter(self, value: float | int) -> str:
        """Format a parameter value to fit in gate box.

        Args:
            value: Numeric parameter value.

        Returns:
            Formatted string:
            - Very small/large numbers: scientific notation (1e-3, 2e5)
            - Normal numbers: 1-2 decimal places
            - Max length: ~5 characters
        """
        if value == 0:
            return "0"

        abs_val = abs(value)

        # Very large or very small: use scientific notation
        if abs_val >= 1000 or abs_val < 0.01:
            return f"{value:.0e}"  # e.g., "1e3", "-2e-4"

        # Normal range: truncate decimals
        if abs_val >= 10:
            return f"{value:.1f}"  # e.g., "12.3"
        return f"{value:.2f}"  # e.g., "0.79"

    def _format_symbolic_param(self, name: str) -> str:
        """Format symbolic parameter name for display with TeX notation.

        All symbolic parameters are wrapped in $...$ for consistent math rendering.

        Args:
            name: Parameter name.

        Returns:
            TeX-formatted string:
            - Contains backslash (e.g., r"\theta"): use as-is (already TeX command)
            - Contains underscore (e.g., "theta_2"): make subscript → $theta_{2}$
            - Known TeX symbol (e.g., "theta"): convert to TeX command → $\theta$
            - Otherwise: wrap in $...$ → $name$
        """
        if "\\" in name:
            # Already a TeX command like \theta, \phi
            return f"${name}$"
        elif "_" in name:
            # Subscript notation like theta_2 → $\theta_{2}$
            parts = name.split("_", 1)
            prefix = f"\\{parts[0]}" if parts[0] in _TEX_SYMBOLS else parts[0]
            return f"${prefix}_{{{parts[1]}}}$"
        else:
            # Known TeX symbol → $\theta$, unknown → $name$
            if name in _TEX_SYMBOLS:
                return f"$\\{name}$"
            return f"${name}$"

    def _get_block_label(
        self, op: CallBlockOperation, qubit_map: dict[str, int]
    ) -> str:
        """Get display label for a CallBlockOperation, including parameters.

        Args:
            op: CallBlockOperation to label.
            qubit_map: Mapping from logical_id to qubit wire index (used to
                distinguish qubit args from parameter args).

        Returns:
            Display label string, e.g. "block(0.5)" or "my_kernel".
        """
        block_value = op.operands[0]
        # The IR guarantees that operands[0] of CallBlockOperation is always a BlockValue,
        # so this assertion should always pass.
        # If it fails, there is a bug in the IR construction.
        assertion_message = (
            "[FOR DEVELOPER] CallBlockOperation.operands[0] must be a BlockValue. "
            "If this assertion fails, there is a bug in the IR construction."
        )
        assert isinstance(block_value, BlockValue), assertion_message

        label = block_value.name or "block"

        # Collect non-qubit parameter values
        params = []
        for arg_name, actual_input in zip(block_value.label_args, op.operands[1:]):
            if not (
                hasattr(actual_input, "logical_id")
                and actual_input.logical_id in qubit_map
            ):
                # The IR guarantees that operands are always Value instances,
                # so this assertion should always pass.
                # If it fails, there is a bug in the IR construction.
                assertion_message = (
                    "[FOR DEVELOPER] Operation.operands elements must be Value instances. "
                    "If this assertion fails, there is a bug in the IR construction."
                )
                assert isinstance(actual_input, Value), assertion_message
                const = actual_input.get_const()
                if const is not None:
                    params.append(self._format_parameter(const))
                elif actual_input.is_parameter():
                    param_name = actual_input.parameter_name() or arg_name
                    params.append(self._format_symbolic_param(param_name))
                else:
                    params.append(arg_name)

        if params:
            label = f"{label}({', '.join(params)})"
        return label

    def _get_gate_label(
        self, op: GateOperation, param_values: dict | None = None
    ) -> tuple[str, bool]:
        """Get display label for a gate.

        Args:
            op: Gate operation.
            param_values: Mapping from logical_ids to resolved parameter values,
                used for inline block expansion.

        Returns:
            Tuple of (label_string, has_parameter).
        """
        # TeX-style gate names
        tex_labels = {
            # Parametric gates
            GateOperationType.RX: r"$R_x$",
            GateOperationType.RY: r"$R_y$",
            GateOperationType.RZ: r"$R_z$",
            GateOperationType.RZZ: r"$R_{zz}$",
            GateOperationType.CP: r"$CP$",
            GateOperationType.P: r"$P$",
            # Non-parametric gates
            GateOperationType.H: r"$H$",
            GateOperationType.X: r"$X$",
            GateOperationType.Y: r"$Y$",
            GateOperationType.Z: r"$Z$",
            GateOperationType.T: r"$T$",
            GateOperationType.S: r"$S$",
            GateOperationType.CX: r"$CX$",
            GateOperationType.CZ: r"$CZ$",
            GateOperationType.SWAP: r"$SWAP$",
            GateOperationType.TOFFOLI: r"$CCX$",
        }

        base_label = tex_labels.get(op.gate_type, str(op.gate_type))

        # Add parameter display if the gate has a theta parameter
        if op.theta is not None:
            if isinstance(op.theta, (int, float)):
                param_str = self._format_parameter(op.theta)
            elif isinstance(op.theta, Value):
                # Check if the Value has a bound constant
                const_val = op.theta.get_const()
                if const_val is not None:
                    # Use the bound constant value
                    param_str = self._format_parameter(const_val)
                elif op.theta.is_parameter():
                    # Check if resolved through param_values (inline block expansion)
                    if param_values and op.theta.logical_id in param_values:
                        resolved = param_values[op.theta.logical_id]
                        if isinstance(resolved, (int, float)):
                            param_str = self._format_parameter(resolved)
                        else:
                            param_str = self._format_symbolic_param(str(resolved))
                    else:
                        # Try evaluating (handles array element access like phis[i])
                        evaluated = self._evaluate_value(op.theta, param_values or {})
                        if evaluated is not None and isinstance(
                            evaluated, (int, float)
                        ):
                            param_str = self._format_parameter(evaluated)
                        else:
                            # Symbolic parameter with parameter_name
                            name = op.theta.parameter_name() or op.theta.name
                            param_str = self._format_symbolic_param(name)
                else:
                    # Generic Value — try evaluating (handles array elements, BinOps)
                    evaluated = self._evaluate_value(op.theta, param_values or {})
                    if evaluated is not None and isinstance(evaluated, (int, float)):
                        param_str = self._format_parameter(evaluated)
                    elif (
                        param_values
                        and op.theta.logical_id in param_values
                        and isinstance(param_values[op.theta.logical_id], str)
                    ):
                        param_str = self._format_symbolic_param(
                            param_values[op.theta.logical_id]
                        )
                    else:
                        param_str = self._format_symbolic_param(op.theta.name or "?")
            else:
                # Unknown type, convert to string
                param_str = str(op.theta)

            return f"{base_label}({param_str})", True

        return base_label, False
