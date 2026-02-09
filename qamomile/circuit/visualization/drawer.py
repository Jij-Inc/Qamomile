"""Matplotlib-based circuit visualization.

This module provides Qiskit-style static circuit visualization
using matplotlib, focusing on clarity and simplicity.
"""

from __future__ import annotations

import io
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

if TYPE_CHECKING:
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
    qubit_columns: dict[int, int] = field(
        default_factory=lambda: defaultdict(lambda: 1)
    )
    qubit_right_edges: dict[int, float] = field(default_factory=dict)


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

    def draw(self, inline: bool = False, fold_loops: bool = True) -> Figure:
        """Generate a matplotlib Figure of the circuit.

        Args:
            inline: If True, expand CallBlockOperation. If False, show as boxes.
            fold_loops: If True (default), display ForOperation as blocks instead of unrolling.
                       If False, expand loops and show all iterations.

        Returns:
            Figure object.
        """
        self.inline = inline
        self.fold_loops = fold_loops
        graph = self.graph

        # Build qubit mapping
        qubit_map = self._build_qubit_map(graph)
        # Use the total number of wires tracked during map building
        num_qubits = self._num_qubits

        if num_qubits == 0:
            # Empty circuit
            fig = self._create_empty_figure()
            self._add_jupyter_display_support(fig)
            return fig

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

    def _add_jupyter_display_support(self, fig: Figure) -> None:
        """Add Jupyter display support to the figure.

        This adds _repr_png_() method to enable automatic display in Jupyter notebooks.

        Args:
            fig: matplotlib Figure to enhance.
        """

        def _repr_png_():
            """Return PNG representation for Jupyter display."""
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            return buf.read()

        # Attach the method to the figure instance
        fig._repr_png_ = _repr_png_

    def _measure_text_bbox(self, ax, text: str, fontsize: int):
        """Measure rendered text bbox in display pixels.

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

    def _calculate_text_width(self, ax, text: str, fontsize: int) -> float:
        """Calculate actual rendered text width in data coordinates."""
        bbox = self._measure_text_bbox(ax, text, fontsize)

        xlim = ax.get_xlim()
        ax_bbox = ax.get_position()
        ax_width_inches = ax_bbox.width * ax.figure.get_figwidth()
        data_range = xlim[1] - xlim[0]
        pixels_per_data_unit = ax.figure.dpi * ax_width_inches / data_range

        if pixels_per_data_unit > 0:
            return bbox.width / pixels_per_data_unit
        return len(text) * self.style.fallback_char_width

    def _calculate_text_height(self, ax, text: str, fontsize: int) -> float:
        """Calculate actual rendered text height in data coordinates."""
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

    def _analyze_loop_affected_qubits(
        self,
        op: ForOperation,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str] | None = None,
    ) -> list[int]:
        """Analyze which qubits are affected by a ForOperation.

        Args:
            op: ForOperation to analyze.
            qubit_map: Mapping from logical_id to qubit index.
            logical_id_remap: Mapping from dummy logical_ids to actual logical_ids.

        Returns:
            List of affected qubit indices.
        """
        if logical_id_remap is None:
            logical_id_remap = {}

        affected = set()

        def get_qubit_index_or_expand(operand) -> int | None:
            """Get qubit index, expanding symbolic array indices into `affected`."""
            if hasattr(operand, "parent_array") and operand.parent_array is not None:
                parent_lid = operand.parent_array.logical_id
                parent_lid = logical_id_remap.get(parent_lid, parent_lid)
                if parent_lid in qubit_map:
                    if hasattr(operand, "element_indices") and operand.element_indices:
                        idx_value = operand.element_indices[0]
                        if not idx_value.is_constant():
                            base_idx = qubit_map[parent_lid]
                            if (
                                hasattr(operand.parent_array, "shape")
                                and operand.parent_array.shape
                            ):
                                size_val = operand.parent_array.shape[0]
                                if size_val.is_constant():
                                    size = size_val.get_const()
                                    if size is not None and isinstance(size, int):
                                        for i in range(size):
                                            affected.add(base_idx + i)
                            return None  # Already added all

            return self._resolve_operand_to_qubit_index(
                operand, qubit_map, logical_id_remap
            )

        def collect_from_ops(ops: list[Operation]) -> None:
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
                elif isinstance(inner_op, ForOperation):
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
                    elif op.kind == BinOpKind.POW:
                        result = lhs_val**rhs_val
                        if isinstance(rhs_val, (int, float)) and rhs_val < 0:
                            return result
                        return int(result)
                return None

        return None

    @staticmethod
    def _safe_int_str(val: int | float | None) -> str:
        """Convert a numeric value to an integer string, handling non-integer floats."""
        if val is None:
            return "?"
        try:
            return str(int(val))
        except (ValueError, OverflowError):
            return str(val)

    def _evaluate_loop_body_intermediates(
        self,
        operations: list[Operation],
        param_values: dict,
    ) -> None:
        """Pre-evaluate intermediate BinOp results in loop body.

        Scans loop body operations for BinOp (e.g., j = i + 1) and stores
        evaluated results by value ID in param_values. This allows subsequent
        index resolution to find the concrete values.
        """
        for op in operations:
            if isinstance(op, BinOp) and op.results:
                result_value = op.results[0]
                resolved = self._evaluate_value(result_value, param_values, operations)
                if resolved is not None:
                    param_values[result_value.logical_id] = resolved

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

        Returns:
            Qubit wire index or None if unresolvable.
        """
        if logical_id_remap is None:
            logical_id_remap = {}
        if param_values is None:
            param_values = {}

        resolved_lid = logical_id_remap.get(operand.logical_id, operand.logical_id)

        # Direct lookup
        if resolved_lid in qubit_map:
            return qubit_map[resolved_lid]

        # Parent array element access
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

        Returns:
            (start_val, stop_val_or_None, step_val) where stop_val is None
            if unresolvable (symbolic).
        """
        start_val = (
            self._evaluate_value(op.operands[0], param_values) if op.operands else 0
        )
        stop_val = (
            self._evaluate_value(op.operands[1], param_values)
            if len(op.operands) > 1
            else 0
        )
        step_val = (
            self._evaluate_value(op.operands[2], param_values)
            if len(op.operands) > 2
            else 1
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
        """Compute number of loop iterations from resolved range values."""
        if step_val > 0:
            return (stop_val - start_val + step_val - 1) // step_val
        return (start_val - stop_val - step_val - 1) // (-step_val)

    def _build_block_value_mappings(
        self,
        block_value,
        actual_inputs: list,
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> tuple[dict[str, str], dict]:
        """Build logical_id_remap and child_param_values for a BlockValue.

        Maps dummy block input logical_ids to actual input logical_ids,
        building the mappings needed for recursive processing.

        Returns:
            (new_logical_id_remap, child_param_values)
        """
        new_logical_id_remap = dict(logical_id_remap)

        for dummy_input, actual_input in zip(block_value.input_values, actual_inputs):
            actual_lid = logical_id_remap.get(
                actual_input.logical_id, actual_input.logical_id
            )
            new_logical_id_remap[dummy_input.logical_id] = actual_lid

        child_param_values = dict(param_values)
        for dummy_input, actual_input in zip(block_value.input_values, actual_inputs):
            if isinstance(actual_input, Value):
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

        return new_logical_id_remap, child_param_values

    def _estimate_gate_width(
        self, op: GateOperation, param_values: dict | None = None
    ) -> float:
        """Estimate gate box width from label text without matplotlib axes.

        Uses character-based estimation suitable for layout phase
        (before figure creation). Also serves as floor for drawing phase.
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
        """
        visual_label = label.replace("$", "")
        visual_label = re.sub(r"\\[a-zA-Z]+", "X", visual_label)
        text_width = len(visual_label) * self.style.char_width_gate
        return max(text_width + 2 * self.style.text_padding, self.style.gate_width)

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

        def resolve_qubit_idx(operand, logical_id_remap: dict[str, str]) -> int | None:
            """Resolve operand to qubit index via logical_id."""
            # Array element access
            if hasattr(operand, "parent_array") and operand.parent_array is not None:
                if len(operand.element_indices) == 1:
                    idx_value = operand.element_indices[0]
                    if idx_value.is_constant():
                        idx = idx_value.get_const()
                        if idx is not None and isinstance(idx, int):
                            parent_lid = operand.parent_array.logical_id
                            parent_lid = logical_id_remap.get(parent_lid, parent_lid)
                            element_key = f"{parent_lid}_[{idx}]"
                            if element_key in qubit_map:
                                return qubit_map[element_key]

            lid = logical_id_remap.get(operand.logical_id, operand.logical_id)
            return qubit_map.get(lid)

        def map_block_results(
            operands, results, logical_id_remap: dict[str, str]
        ) -> None:
            """Map block output results to same qubits as corresponding inputs."""
            nonlocal next_idx
            for operand, result in zip(operands, results):
                if not isinstance(result.type, QubitType):
                    continue
                lid = logical_id_remap.get(operand.logical_id, operand.logical_id)
                qubit_idx = qubit_map.get(lid)
                if qubit_idx is not None:
                    qubit_map[result.logical_id] = qubit_idx
                else:
                    qubit_map[operand.logical_id] = next_idx
                    qubit_map[result.logical_id] = next_idx
                    next_idx += 1

        def build_chains(
            ops: list[Operation],
            logical_id_remap: dict[str, str] | None = None,
        ) -> None:
            nonlocal next_idx
            if logical_id_remap is None:
                logical_id_remap = {}

            for op in ops:
                if isinstance(op, QInitOperation):
                    qubit = op.results[0]

                    # Check if this is an ArrayValue (qubit array)
                    if isinstance(qubit, ArrayValue):
                        if isinstance(qubit.type, QubitType):
                            if len(qubit.shape) == 1:  # 1D array (Vector)
                                size_value = qubit.shape[0]
                                if size_value.is_constant():
                                    array_size = size_value.get_const()
                                    if array_size is not None and isinstance(
                                        array_size, int
                                    ):
                                        for i in range(array_size):
                                            element_key = f"{qubit.logical_id}_[{i}]"
                                            if element_key not in qubit_map:
                                                qubit_map[element_key] = next_idx
                                                qubit_names[next_idx] = (
                                                    f"{qubit.name}[{i}]"
                                                )
                                                next_idx += 1
                                        # Store base index for array
                                        qubit_map[qubit.logical_id] = qubit_map.get(
                                            f"{qubit.logical_id}_[0]", next_idx
                                        )
                                        continue
                                else:
                                    raise ValueError(
                                        f"Cannot visualize circuit: qubit array "
                                        f"'{qubit.name}' has symbolic size. "
                                        f"Please provide concrete values for all "
                                        f"size parameters when calling draw()."
                                    )

                    # Scalar qubit
                    if qubit.logical_id not in qubit_map:
                        qubit_map[qubit.logical_id] = next_idx
                        qubit_names[next_idx] = qubit.name
                        next_idx += 1

                elif isinstance(op, CallBlockOperation):
                    if self.inline:
                        block_value = op.operands[0]
                        if isinstance(block_value, BlockValue):
                            new_remap = dict(logical_id_remap)
                            actual_inputs = op.operands[1:]

                            for dummy_input, actual_input in zip(
                                block_value.input_values, actual_inputs
                            ):
                                if not isinstance(dummy_input.type, QubitType):
                                    continue
                                actual_lid = logical_id_remap.get(
                                    actual_input.logical_id, actual_input.logical_id
                                )
                                new_remap[dummy_input.logical_id] = actual_lid

                                # Ensure actual_input is registered
                                if actual_lid not in qubit_map:
                                    qubit_map[actual_lid] = next_idx
                                    next_idx += 1

                            build_chains(block_value.operations, new_remap)
                            map_block_results(
                                op.operands[1:], op.results, logical_id_remap
                            )
                    else:
                        map_block_results(op.operands[1:], op.results, logical_id_remap)

                elif isinstance(op, CastOperation):
                    if op.operands and op.results:
                        source = op.operands[0]
                        result = op.results[0]
                        source_lid = logical_id_remap.get(
                            source.logical_id, source.logical_id
                        )
                        qubit_idx = qubit_map.get(source_lid)
                        if qubit_idx is not None:
                            qubit_map[result.logical_id] = qubit_idx

                # GateOperation, ControlledUOperation, CompositeGateOperation:
                # No-op — next_version() preserves logical_id

        build_chains(graph.operations)
        self.qubit_names = qubit_names
        self._num_qubits = next_idx
        return qubit_map

    def _layout_for_operation(
        self,
        state: LayoutState,
        op: ForOperation,
        op_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
        process_operations_fn,
    ) -> bool:
        """Handle ForOperation layout: fold as box or expand iterations.

        Returns True if the loop was zero-iteration (should be skipped).
        """
        start_val, stop_val_raw, step_val = self._evaluate_loop_range(op, param_values)

        if self._is_zero_iteration_loop(start_val, stop_val_raw, step_val):
            return True

        stop_val = stop_val_raw if stop_val_raw is not None else 0

        if self.fold_loops:
            # Draw as a box (folded)
            affected_qubits = self._analyze_loop_affected_qubits(
                op, qubit_map, logical_id_remap
            )

            if affected_qubits:
                start_column = max(state.qubit_columns[q] for q in affected_qubits)
            elif state.qubit_columns:
                start_column = max(state.qubit_columns.values())
            else:
                start_column = 0

            # Use fixed width for folded loop box
            loop_width = self.style.folded_loop_width
            op_half_width = loop_width / 2
            gap = self.style.gate_gap

            # Overlap prevention
            for q in affected_qubits:
                if q in state.qubit_right_edges:
                    required_center = state.qubit_right_edges[q] + gap + op_half_width
                    start_column = max(start_column, required_center)

            state.positions[op_key] = start_column
            state.block_widths[op_key] = loop_width

            # Update qubit columns and right edges
            for q in affected_qubits:
                state.qubit_columns[q] = start_column + op_half_width + gap
                state.qubit_right_edges[q] = start_column + op_half_width

            state.column = max(state.column, start_column + 1)
            state.actual_width = max(
                state.actual_width, start_column + op_half_width + 0.5
            )
        else:
            # Expand loop: process each iteration
            num_iterations = self._compute_loop_iterations(
                start_val, stop_val, step_val
            )
            for iteration in range(num_iterations):
                iter_value = start_val + iteration * step_val
                child_param_values = dict(param_values)
                child_param_values[f"_loop_{op.loop_var}"] = iter_value

                self._evaluate_loop_body_intermediates(
                    op.operations, child_param_values
                )

                process_operations_fn(
                    op.operations,
                    logical_id_remap,
                    depth + 1,
                    scope_path=(*op_key, iteration),
                    param_values=child_param_values,
                )
        return False

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
        return max_width

    def _prepare_inline_block(
        self,
        state: LayoutState,
        affected_qubits: list[int],
        depth: int,
    ) -> dict[int, float]:
        """Common pre-processing for inline block layout.

        Advances qubit_right_edges for border extent and returns qubit_start_columns.
        """
        max_gate_width = self.style.gate_width
        border_padding = self._compute_border_padding(depth)
        border_extent = max_gate_width / 2 + border_padding
        for q in affected_qubits:
            if q in state.qubit_right_edges:
                state.qubit_right_edges[q] += border_extent - self.style.gate_width / 2
        return {q: state.qubit_columns[q] for q in affected_qubits}

    def _layout_inline_call_block(
        self,
        state: LayoutState,
        op: CallBlockOperation,
        op_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
        process_operations_fn,
    ) -> None:
        """Layout a CallBlockOperation in inline mode."""
        block_value = op.operands[0]
        if not isinstance(block_value, BlockValue):
            return

        affected_qubits = []
        for operand in op.operands[1:]:
            affected_qubits.extend(
                self._resolve_operand_to_qubit_indices(
                    operand, qubit_map, logical_id_remap, param_values
                )
            )

        qubit_start_columns = self._prepare_inline_block(state, affected_qubits, depth)

        actual_inputs = op.operands[1:]
        new_logical_id_remap, child_param_values = self._build_block_value_mappings(
            block_value, actual_inputs, logical_id_remap, param_values
        )

        max_gate_width = self._max_block_gate_width(
            block_value.operations, child_param_values
        )

        process_operations_fn(
            block_value.operations,
            new_logical_id_remap,
            depth + 1,
            scope_path=op_key,
            param_values=child_param_values,
        )

        self._finalize_inline_block_layout(
            state,
            op_key,
            block_value.name or "block",
            affected_qubits,
            qubit_start_columns,
            max_gate_width,
            depth,
        )

    def _layout_inline_controlled_u(
        self,
        state: LayoutState,
        op: ControlledUOperation,
        op_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
        process_operations_fn,
    ) -> None:
        """Layout a ControlledUOperation in inline mode."""
        block_value = op.block
        if not isinstance(block_value, BlockValue):
            return

        affected_qubits = []
        for operand in list(op.control_operands) + list(op.target_operands):
            idx = self._resolve_operand_to_qubit_index(
                operand, qubit_map, logical_id_remap, param_values
            )
            if idx is not None:
                affected_qubits.append(idx)

        qubit_start_columns = self._prepare_inline_block(state, affected_qubits, depth)

        actual_inputs = list(op.target_operands)
        new_logical_id_remap, child_param_values = self._build_block_value_mappings(
            block_value, actual_inputs, logical_id_remap, param_values
        )

        max_gate_width = self._max_block_gate_width(
            block_value.operations, child_param_values
        )

        process_operations_fn(
            block_value.operations,
            new_logical_id_remap,
            depth + 1,
            scope_path=op_key,
            param_values=child_param_values,
        )

        u_name = getattr(block_value, "name", "U") or "U"
        self._finalize_inline_block_layout(
            state,
            op_key,
            u_name,
            affected_qubits,
            qubit_start_columns,
            max_gate_width,
            depth,
        )

    def _layout_inline_composite_gate(
        self,
        state: LayoutState,
        op: CompositeGateOperation,
        op_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
        process_operations_fn,
    ) -> None:
        """Layout a CompositeGateOperation in inline mode."""
        block_value = op.implementation
        if not isinstance(block_value, BlockValue):
            return

        affected_qubits = []
        for operand in list(op.control_qubits) + list(op.target_qubits):
            idx = self._resolve_operand_to_qubit_index(
                operand, qubit_map, logical_id_remap, param_values
            )
            if idx is not None:
                affected_qubits.append(idx)

        qubit_start_columns = self._prepare_inline_block(state, affected_qubits, depth)

        actual_inputs = list(op.operands[1:])
        new_logical_id_remap, child_param_values = self._build_block_value_mappings(
            block_value, actual_inputs, logical_id_remap, param_values
        )

        max_gate_width = self._max_block_gate_width(
            block_value.operations, child_param_values
        )

        process_operations_fn(
            block_value.operations,
            new_logical_id_remap,
            depth + 1,
            scope_path=op_key,
            param_values=child_param_values,
        )

        self._finalize_inline_block_layout(
            state,
            op_key,
            op.name,
            affected_qubits,
            qubit_start_columns,
            max_gate_width,
            depth,
        )

    def _finalize_inline_block_layout(
        self,
        state: LayoutState,
        op_key: tuple,
        block_name: str,
        affected_qubits: list[int],
        qubit_start_columns: dict[int, float],
        max_gate_width: float,
        depth: int,
    ) -> None:
        """Common post-processing for inline block layout: compute block range and update edges."""
        qubit_end_columns = {q: state.qubit_columns[q] for q in affected_qubits}
        if affected_qubits:
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
                actual_start = min(qubit_start_columns.values())
                actual_end = max(qubit_end_columns.values()) - 1

            state.block_ranges.append(
                {
                    "name": block_name,
                    "start_x": actual_start,
                    "end_x": actual_end,
                    "qubit_indices": affected_qubits,
                    "depth": depth,
                    "max_gate_width": max_gate_width,
                }
            )

            border_padding = self._compute_border_padding(depth)
            gate_border_right = actual_end + max_gate_width / 2 + border_padding
            box_left = actual_start - max_gate_width / 2 - border_padding
            title_text_width = len(block_name) * self.style.char_width_base
            label_right = (
                box_left
                + self.style.label_horizontal_padding
                + title_text_width
                + self.style.label_horizontal_padding
            )
            border_right_edge = max(gate_border_right, label_right)

            for q in affected_qubits:
                state.qubit_right_edges[q] = border_right_edge
                state.qubit_columns[q] = border_right_edge + self.style.gate_gap

    def _layout_generic_operation(
        self,
        state: LayoutState,
        op: Operation,
        op_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> None:
        """Layout a generic operation (gate, measurement, block box)."""
        # Find which qubits this operation touches
        affected_qubits = []
        for i, operand in enumerate(op.operands):
            # Skip BlockValue for CallBlockOperation, ControlledUOperation, and CompositeGateOperation
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

        # Calculate required columns and operation half-width
        columns_needed = 1
        op_half_width = self.style.gate_width / 2  # default

        if isinstance(op, GateOperation):
            estimated_w = self._estimate_gate_width(op, param_values)
            columns_needed = max(1, math.ceil(estimated_w))
            op_half_width = estimated_w / 2
        elif isinstance(op, CallBlockOperation):
            label = self._get_block_label(op, qubit_map)
            box_width = self._estimate_label_box_width(label)
            state.block_widths[op_key] = box_width
            columns_needed = max(1, int(box_width) + 1)
            op_half_width = box_width / 2
        elif isinstance(op, CompositeGateOperation):
            box_width = self._estimate_label_box_width(op.name.upper())
            state.block_widths[op_key] = box_width
            columns_needed = max(1, int(box_width) + 1)
            op_half_width = box_width / 2
        elif isinstance(op, ControlledUOperation):
            u_name = getattr(op.block, "name", "U") or "U"
            label = f"{u_name}^{op.power}" if op.power > 1 else u_name
            box_width = self._estimate_label_box_width(label)
            state.block_widths[op_key] = box_width
            columns_needed = max(1, int(box_width) + 1)
            op_half_width = box_width / 2

        # For multi-qubit gates, also check and update intermediate qubits
        if len(affected_qubits) > 1:
            span_qubits = list(range(min(affected_qubits), max(affected_qubits) + 1))
        else:
            span_qubits = list(affected_qubits)

        # Overlap prevention: ensure this box's left edge clears previous box's right edge
        for q in span_qubits:
            if q in state.qubit_right_edges:
                required_center = (
                    state.qubit_right_edges[q] + self.style.gate_gap + op_half_width
                )
                min_column = max(min_column, required_center)

        # Place operation
        state.positions[op_key] = min_column
        state.column = max(state.column, min_column + columns_needed)

        # Track first gate position and half-width for wire symmetry
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

    def _layout_operations(
        self, graph: Graph, qubit_map: dict[str, int]
    ) -> dict[str, int | dict]:
        """Calculate layout positions for operations.

        Returns:
            Layout dictionary with:
            - 'width': Total circuit width in coordinate units
            - 'positions': dict[int, int] mapping operation ID to x positions
            - 'block_ranges': list[dict] block boundary information (when inline=True)
            - 'max_depth': Maximum nesting depth of blocks
            - 'block_widths': dict[int, float] mapping operation ID to box widths (for CallBlockOperation)
            - 'actual_width': Actual circuit width considering box widths
        """
        state = LayoutState()

        # Initialize right edges to initial_wire_position so that the first gate
        # is placed far enough right to avoid overlapping wire labels.
        for q_idx in set(qubit_map.values()):
            state.qubit_right_edges[q_idx] = self.style.initial_wire_position
            # Pre-initialize qubit_columns to ensure entries exist
            _ = state.qubit_columns[q_idx]

        def process_operations(
            ops: list[Operation],
            logical_id_remap: dict[str, str]
            | None = None,  # Dummy logical_id → Actual logical_id
            depth: int = 0,  # Nesting depth for blocks
            scope_path: tuple = (),
            param_values: dict[int, float | int | str] | None = None,
        ) -> None:
            if logical_id_remap is None:
                logical_id_remap = {}
            if param_values is None:
                param_values = {}

            state.max_depth = max(state.max_depth, depth)

            for op in ops:
                op_key = (*scope_path, id(op))

                if isinstance(op, QInitOperation):
                    # Initialization doesn't take space
                    state.positions[op_key] = 0
                    continue

                if isinstance(op, ExpvalOp):
                    raise NotImplementedError(
                        f"Visualization of ExpvalOp is not yet supported. "
                        f"The circuit contains an expectation value calculation: {op}"
                    )

                if isinstance(op, (WhileOperation, IfOperation, ForItemsOperation)):
                    raise NotImplementedError(
                        f"Visualization of {type(op).__name__} is not yet supported."
                    )

                if isinstance(op, CastOperation):
                    # Passthrough: no visual space needed
                    continue

                if isinstance(op, ForOperation):
                    self._layout_for_operation(
                        state,
                        op,
                        op_key,
                        qubit_map,
                        logical_id_remap,
                        param_values,
                        depth,
                        process_operations,
                    )
                    continue

                if isinstance(op, CallBlockOperation) and self.inline:
                    self._layout_inline_call_block(
                        state,
                        op,
                        op_key,
                        qubit_map,
                        logical_id_remap,
                        param_values,
                        depth,
                        process_operations,
                    )
                    continue

                if isinstance(op, ControlledUOperation) and self.inline:
                    self._layout_inline_controlled_u(
                        state,
                        op,
                        op_key,
                        qubit_map,
                        logical_id_remap,
                        param_values,
                        depth,
                        process_operations,
                    )
                    continue

                if (
                    isinstance(op, CompositeGateOperation)
                    and self.inline
                    and op.has_implementation
                ):
                    self._layout_inline_composite_gate(
                        state,
                        op,
                        op_key,
                        qubit_map,
                        logical_id_remap,
                        param_values,
                        depth,
                        process_operations,
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
                    self._layout_generic_operation(
                        state, op, op_key, qubit_map, logical_id_remap, param_values
                    )

        process_operations(graph.operations)
        # Ensure actual_width is at least as large as column
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

    def _create_empty_figure(self) -> Figure:
        """Create an empty figure for circuits with no qubits."""
        # Use Figure directly to avoid pyplot auto-display in Jupyter
        fig = Figure(figsize=(4, 2))
        # Attach canvas for Jupyter display
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

        # Store ax in figure for later access
        fig._qm_ax = ax

        return fig

    def _create_figure(self, num_qubits: int, layout: dict) -> Figure:
        """Create matplotlib figure with appropriate size.

        Args:
            num_qubits: Number of qubits.
            layout: Layout dictionary.

        Returns:
            matplotlib Figure.
        """
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

        if hasattr(self, "_max_above") and self._max_above:
            y_margin_top = max(self._max_above.get(0, 0), base_margin) + clearance
        else:
            y_margin_top = base_margin

        if hasattr(self, "_max_below") and self._max_below:
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

    def _build_output_label_map(self, qubit_map: dict[str, int]) -> dict[int, str]:
        """Build mapping from wire index to output variable names.

        Args:
            qubit_map: Mapping from logical_id to wire index.

        Returns:
            Dictionary mapping wire index to output variable name.
        """
        result = {}
        for i, value in enumerate(self.graph.output_values):
            if i >= len(self.graph.output_names):
                break
            wire = qubit_map.get(value.logical_id)
            if wire is not None:
                result[wire] = self.graph.output_names[i]
        return result

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
        actual_inputs: list,
        op_key: tuple,
        logical_id_remap: dict[str, str],
        param_values: dict,
        draw_ops_fn,
    ) -> None:
        """Recursively draw inlined block operations.

        Shared logic for CallBlock, CompositeGate, and ControlledU inline drawing.
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
        positions: dict,
        block_widths: dict,
        logical_id_remap: dict[str, str],
        param_values: dict,
        draw_ops_fn,
    ) -> None:
        """Draw a CallBlockOperation: inline or as box."""
        if self.inline:
            block_value = op.operands[0]
            if isinstance(block_value, BlockValue):
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
            )

    def _draw_composite_or_inline(
        self,
        fig: Figure,
        op: CompositeGateOperation,
        op_key: tuple,
        qubit_map: dict[str, int],
        positions: dict,
        block_widths: dict,
        logical_id_remap: dict[str, str],
        param_values: dict,
        draw_ops_fn,
    ) -> None:
        """Draw a CompositeGateOperation: inline or as box."""
        if self.inline and op.has_implementation:
            block_value = op.implementation
            if isinstance(block_value, BlockValue):
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
            )

    def _draw_controlled_u_or_inline(
        self,
        fig: Figure,
        op: ControlledUOperation,
        op_key: tuple,
        qubit_map: dict[str, int],
        positions: dict,
        block_widths: dict,
        logical_id_remap: dict[str, str],
        param_values: dict,
        draw_ops_fn,
    ) -> None:
        """Draw a ControlledUOperation: inline or as box."""
        if self.inline:
            block_value = op.block
            if isinstance(block_value, BlockValue):
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
            )

    def _draw_for_op(
        self,
        fig: Figure,
        op: ForOperation,
        op_key: tuple,
        qubit_map: dict[str, int],
        positions: dict,
        block_widths: dict,
        logical_id_remap: dict[str, str],
        param_values: dict,
        draw_ops_fn,
    ) -> None:
        """Handle drawing a ForOperation (fold or expand)."""
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
                        margin = 0.3
                        if inner["start_x"] - margin < outer["start_x"]:
                            outer["start_x"] = inner["start_x"] - margin
                        if inner["end_x"] + margin > outer["end_x"]:
                            outer["end_x"] = inner["end_x"] + margin

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

                self._draw_inlined_block_border(
                    fig._qm_ax,
                    name=block_info["name"],
                    start_x=block_info["start_x"],
                    end_x=block_info["end_x"],
                    qubit_indices=block_info["qubit_indices"],
                    depth=block_info["depth"],
                    max_gate_width=block_info.get("max_gate_width"),
                    max_depth=max_depth,
                    overlap_index=overlap_count,
                )

        def draw_ops(
            ops: list[Operation],
            logical_id_remap: dict[str, str]
            | None = None,  # Dummy logical_id → Actual logical_id
            scope_path: tuple = (),
            param_values: dict[int, float | int | str] | None = None,
        ) -> None:
            if logical_id_remap is None:
                logical_id_remap = {}
            if param_values is None:
                param_values = {}

            for op in ops:
                op_key = (*scope_path, id(op))

                if isinstance(op, (QInitOperation, CastOperation)):
                    continue

                if isinstance(op, ExpvalOp):
                    raise NotImplementedError(
                        f"Visualization of ExpvalOp is not yet supported. "
                        f"The circuit contains an expectation value calculation: {op}"
                    )

                if isinstance(op, (WhileOperation, IfOperation, ForItemsOperation)):
                    raise NotImplementedError(
                        f"Visualization of {type(op).__name__} is not yet supported."
                    )

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
        ax,
        name: str,
        start_x: float,
        end_x: float,
        qubit_indices: list[int],
        depth: int = 0,
        max_gate_width: float | None = None,
        max_depth: int = 0,
        overlap_index: int = 0,
    ) -> None:
        """Draw dashed border around inlined block operations.

        Args:
            ax: matplotlib Axes.
            name: Block name for label.
            start_x: Start x coordinate (column).
            end_x: End x coordinate (column).
            qubit_indices: List of qubit indices affected by the block.
            depth: Nesting depth for nested blocks.
            max_gate_width: Maximum gate width in the block (for parametric gates).
            overlap_index: Index for blocks at the same horizontal position (to avoid label overlap).
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

        # Calculate initial box boundaries based on gates
        gate_box_left = start_x - max_gate_width / 2 - padding
        gate_box_right = end_x + max_gate_width / 2 + padding

        # Calculate label position
        label_left = gate_box_left + self.style.label_horizontal_padding

        # Extend box_right if title text is wider than gate-based width
        title_text_width = len(name) * self.style.char_width_base

        # Expand box symmetrically if title needs more space
        gate_width_span = gate_box_right - gate_box_left
        required_width = max(
            gate_width_span, title_text_width + 2 * self.style.text_padding
        )
        if required_width > gate_width_span:
            extra = (required_width - gate_width_span) / 2
            box_left = gate_box_left - extra
            box_right = gate_box_right + extra
        else:
            box_left = gate_box_left
            box_right = gate_box_right

        # Calculate vertical box boundaries (always include label height)
        box_bottom = min_y - self.style.gate_height / 2 - padding
        box_top = (
            max_y
            + self.style.gate_height / 2
            + padding
            + label_height
            + label_vertical_offset
        )

        # Draw dashed rectangle
        # Use depth-adjusted zorder so inner (higher depth) blocks appear on top
        block_zorder = PORDER_WIRE + 0.5 + depth * 0.1
        rect = mpatches.Rectangle(
            (box_left, box_bottom),
            box_right - box_left,
            box_top - box_bottom,
            facecolor="none",  # No fill
            edgecolor=self.style.block_border_color,
            linewidth=1.5,
            linestyle="--",  # Dashed line
            zorder=block_zorder,  # Above wire, below gates; inner blocks on top
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
        ax,
        op: GateOperation,
        label: str,
        has_param: bool,
        param_values: dict | None,
    ) -> float:
        """Compute the draw width for a gate, handling parametric gates."""
        if has_param:
            text_width = self._calculate_text_width(ax, label, self.style.font_size)
            calculated_width = text_width + 2 * self.style.text_padding
            estimated_width = self._estimate_gate_width(op, param_values)
            return max(estimated_width, calculated_width)
        return self.style.gate_width

    def _draw_single_qubit_gate(
        self,
        ax,
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
        ax,
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

    def _draw_control_dot(self, ax, x: float, y: float) -> None:
        """Draw a control dot for controlled gates."""
        circle = mpatches.Circle(
            (x, y),
            radius=0.1,
            facecolor=self.style.wire_color,
            edgecolor=self.style.wire_color,
            zorder=PORDER_GATE,
        )
        ax.add_patch(circle)

    def _draw_target_x(self, ax, x: float, y: float) -> None:
        """Draw a target X (plus sign in circle) for CNOT."""
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

    def _draw_swap_x(self, ax, x: float, y: float) -> None:
        """Draw an X marker for SWAP gate."""
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
    ) -> None:
        """Draw a CallBlockOperation as a box.

        Args:
            fig: matplotlib Figure.
            op: CallBlock operation.
            qubit_map: Qubit to wire index mapping.
            x_pos: X position.
            block_width: Pre-calculated block width from layout phase.
            logical_id_remap: Mapping from dummy logical_ids to actual logical_ids.
        """
        if logical_id_remap is None:
            logical_id_remap = {}

        ax = fig._qm_ax

        # Get affected qubits (skip first operand which is BlockValue)
        qubit_indices = []
        for operand in op.operands[1:]:  # Skip BlockValue
            qubit_indices.extend(
                self._resolve_operand_to_qubit_indices(
                    operand, qubit_map, logical_id_remap
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
    ) -> None:
        """Draw a CompositeGateOperation as a labeled box."""
        if logical_id_remap is None:
            logical_id_remap = {}
        ax = fig._qm_ax

        # Collect qubit indices from control + target qubits
        qubit_indices = []
        for qval in op.control_qubits + op.target_qubits:
            idx = self._resolve_operand_to_qubit_index(
                qval, qubit_map, logical_id_remap
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
    ) -> None:
        """Draw a ControlledUOperation with control dots and target box."""
        if logical_id_remap is None:
            logical_id_remap = {}
        ax = fig._qm_ax

        # Resolve control qubit y-coordinates
        control_y = []
        for qval in op.control_operands:
            idx = self._resolve_operand_to_qubit_index(
                qval, qubit_map, logical_id_remap
            )
            if idx is not None:
                control_y.append(self.qubit_y[idx])

        # Resolve target qubit y-coordinates
        target_y = []
        for qval in op.target_operands:
            idx = self._resolve_operand_to_qubit_index(
                qval, qubit_map, logical_id_remap
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
            label = f"{u_name}^{op.power}" if op.power > 1 else u_name
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
        self, ax, x_pos: float, y: float, width: float, height: float
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

    def _draw_measurement_box(self, ax, x_pos: float, y: float) -> None:
        """Draw a single measurement box with meter symbol at the given position."""
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
        """Draw a MeasureOperation as an M box with meter symbol."""
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
        """Draw a MeasureVectorOperation as M boxes on all measured qubits."""
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
        self, op: Operation, loop_var: str
    ) -> str | None:
        """Convert an operation to expression format string.

        Args:
            op: Operation to format.
            loop_var: Loop variable name (e.g., "i").

        Returns:
            Expression string (e.g., "qubits[i] = h(qubits[i])") or None.
        """
        if isinstance(op, GateOperation):
            gate_name = op.gate_type.name.lower()

            if not op.operands:
                return None

            # Get the first operand (target qubit for most gates)
            operand = op.operands[0]
            array_name = None

            # Check if operand is an array element
            if hasattr(operand, "parent_array") and operand.parent_array is not None:
                array_name = operand.parent_array.name or "qubits"

            if array_name is None:
                return None

            # Resolve the actual index expression
            idx_str = self._resolve_index_expression(operand, loop_var)

            # Get parameters if any
            params = self._get_gate_params_for_expression(op)
            if params:
                return f"{array_name}[{idx_str}] = {gate_name}({array_name}[{idx_str}], {params})"
            return f"{array_name}[{idx_str}] = {gate_name}({array_name}[{idx_str}])"

        elif isinstance(op, (MeasureOperation, MeasureVectorOperation)):
            # Format measurement operation
            if not op.operands:
                return None

            operand = op.operands[0]
            array_name = None

            if hasattr(operand, "parent_array") and operand.parent_array is not None:
                array_name = operand.parent_array.name or "qubits"
            elif hasattr(operand, "name"):
                array_name = operand.name or "qubits"

            if array_name:
                idx_str = self._resolve_index_expression(operand, loop_var)
                return f"measure({array_name}[{idx_str}])"
            return "measure(...)"

        elif isinstance(op, CallBlockOperation):
            block_value = op.operands[0]
            if isinstance(block_value, BlockValue):
                block_name = block_value.name or "block"

                # Get qubit operand
                if len(op.operands) > 1:
                    operand = op.operands[1]
                    array_name = None
                    if (
                        hasattr(operand, "parent_array")
                        and operand.parent_array is not None
                    ):
                        array_name = operand.parent_array.name or "qubits"

                    if array_name:
                        idx_str = self._resolve_index_expression(operand, loop_var)
                        return f"{array_name}[{idx_str}] = {block_name}({array_name}[{idx_str}])"
                return f"{block_name}(...)"

        return None

    def _resolve_index_expression(self, operand, loop_var: str) -> str:
        """Resolve an operand's element index to a human-readable string.

        Args:
            operand: A Value that may be an array element.
            loop_var: Loop variable name (e.g., "i").

        Returns:
            Index expression string (e.g., "i", "i+1", "1").
        """
        if not hasattr(operand, "element_indices") or not operand.element_indices:
            return loop_var

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
        return self._format_value_as_expression(idx_value, loop_var)

    def _format_value_as_expression(self, value: Value, loop_var: str) -> str:
        """Format a Value as a human-readable expression string.

        Recursively resolves BinOp chains to produce expressions like "i+1", "2*i".
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
        if hasattr(value, "name") and value.name == loop_var:
            return loop_var

        # Search for the defining BinOp in the graph
        graph = getattr(self, "graph", None)
        operations = graph.operations if graph else []
        for op in operations:
            if isinstance(op, BinOp) and op.results and id(op.results[0]) == id(value):
                lhs_str = self._format_value_as_expression(op.lhs, loop_var)
                rhs_str = self._format_value_as_expression(op.rhs, loop_var)
                op_symbol = {
                    BinOpKind.ADD: "+",
                    BinOpKind.SUB: "-",
                    BinOpKind.MUL: "*",
                    BinOpKind.FLOORDIV: "//",
                    BinOpKind.POW: "**",
                }.get(op.kind, "?")
                return f"{lhs_str}{op_symbol}{rhs_str}"

        return loop_var

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
            op, qubit_map, logical_id_remap
        )

        if not affected_qubits:
            return

        # Convert to y coordinates using variable spacing
        y_coords = [self.qubit_y[q] for q in affected_qubits]
        min_y = min(y_coords)
        max_y = max(y_coords)

        # Get loop range for label
        start_val, stop_val_raw, step_val = self._evaluate_loop_range(op, param_values)

        start_str = self._safe_int_str(start_val)
        stop_str = self._safe_int_str(stop_val_raw)
        step_str = self._safe_int_str(step_val)

        if start_val == 0 and step_val == 1:
            range_str = f"qm.range({stop_str})"
        elif step_val == 1:
            range_str = f"qm.range({start_str}, {stop_str})"
        else:
            range_str = f"qm.range({start_str}, {stop_str}, {step_str})"
        label = f"for {op.loop_var} in {range_str}"

        # Collect operation expressions from loop body for detailed label
        expressions = []
        for body_op in op.operations:
            expr = self._format_operation_as_expression(body_op, op.loop_var)
            if expr:
                expressions.append(expr)

        if expressions:
            # Show up to 3 expression lines
            expr_lines = expressions[:3]
            if len(expressions) > 3:
                expr_lines.append("...")
            label += "\n" + "\n".join(expr_lines)

        # Use pre-calculated width or default
        if block_width is not None:
            width = block_width
        else:
            width = self.style.folded_loop_width

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
            facecolor=self.style.for_loop_face_color,
            edgecolor=self.style.for_loop_edge_color,
            linewidth=1.5,
            linestyle="-",  # Solid line for for-loop boxes
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
            color=self.style.for_loop_text_color,
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

    def _get_block_label(self, op: CallBlockOperation, qubit_map: dict) -> str:
        """Get display label for a CallBlockOperation, including parameters."""
        block_value = op.operands[0]
        if not isinstance(block_value, BlockValue):
            return "block"

        label = block_value.name or "block"

        # Collect non-qubit parameter values
        params = []
        for arg_name, actual_input in zip(block_value.label_args, op.operands[1:]):
            if not (
                hasattr(actual_input, "logical_id")
                and actual_input.logical_id in qubit_map
            ):
                if isinstance(actual_input, Value):
                    const = actual_input.get_const()
                    if const is not None:
                        params.append(self._format_parameter(const))
                    elif actual_input.is_parameter():
                        param_name = actual_input.parameter_name() or arg_name
                        params.append(self._format_symbolic_param(param_name))
                    else:
                        params.append(arg_name)
                elif isinstance(actual_input, (int, float)):
                    params.append(self._format_parameter(actual_input))
                else:
                    params.append(str(actual_input))

        if params:
            label = f"{label}({', '.join(params)})"
        return label

    def _get_gate_label(
        self, op: GateOperation, param_values: dict | None = None
    ) -> tuple[str, bool]:
        """Get display label for a gate.

        Args:
            op: Gate operation.

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
                        # Symbolic parameter with parameter_name
                        name = op.theta.parameter_name() or op.theta.name
                        param_str = self._format_symbolic_param(name)
                else:
                    # Generic Value (use its name)
                    param_str = self._format_symbolic_param(op.theta.name)
            else:
                # Unknown type, convert to string
                param_str = str(op.theta)

            return f"{base_label}({param_str})", True

        return base_label, False
