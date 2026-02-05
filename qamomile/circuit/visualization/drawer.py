"""Matplotlib-based circuit visualization.

This module provides Qiskit-style static circuit visualization
using matplotlib, focusing on clarity and simplicity.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.figure
    from qamomile.circuit.ir.graph import Graph

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.operation.control_flow import ForOperation
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.operation.gate import GateOperation, GateOperationType
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.value import Value, ArrayValue

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


class MatplotlibDrawer:
    """Matplotlib-based circuit drawer with Qiskit-style layout.

    This drawer produces static matplotlib figures showing quantum circuits.
    It supports two modes:
    - Block mode (inline=False): Shows CallBlockOperation as boxes
    - Inline mode (inline=True): Expands CallBlockOperation contents
    """

    def __init__(
        self, graph: Graph, style: CircuitStyle | None = None, fold_loops: bool = True
    ):
        """Initialize the drawer.

        Args:
            graph: Computation graph to visualize.
            style: Visual style configuration. Uses DEFAULT_STYLE if None.
            fold_loops: If True (default), display ForOperation as blocks instead of unrolling.
                       If False, expand loops and show all iterations.
        """
        self.graph = graph
        self.style = style or DEFAULT_STYLE
        self.inline = False
        self.fold_loops = fold_loops

    def draw(
        self, inline: bool = False, fold_loops: bool | None = None
    ) -> matplotlib.figure.Figure:
        """Generate a matplotlib Figure of the circuit.

        Args:
            inline: If True, expand CallBlockOperation. If False, show as boxes.
            fold_loops: If provided, override the instance fold_loops setting.

        Returns:
            matplotlib.figure.Figure object.
        """
        self.inline = inline
        if fold_loops is not None:
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

    def _add_jupyter_display_support(self, fig: matplotlib.figure.Figure) -> None:
        """Add Jupyter display support to the figure.

        This adds _repr_png_() method to enable automatic display in Jupyter notebooks.

        Args:
            fig: matplotlib Figure to enhance.
        """
        import io

        def _repr_png_():
            """Return PNG representation for Jupyter display."""
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            return buf.read()

        # Attach the method to the figure instance
        fig._repr_png_ = _repr_png_

    def _calculate_text_width(self, ax, text: str, fontsize: int) -> float:
        """Calculate actual rendered text width in data coordinates.

        Args:
            ax: matplotlib Axes.
            text: Text string to measure.
            fontsize: Font size.

        Returns:
            Width in data coordinates.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.font_manager import FontProperties

        # Ensure figure has a canvas for rendering
        if ax.figure.canvas is None or not hasattr(ax.figure.canvas, "get_renderer"):
            FigureCanvasAgg(ax.figure)

        renderer = ax.figure.canvas.get_renderer()

        # Get font properties
        font_props = FontProperties(size=fontsize)

        # Create temporary text to get bbox in display coordinates
        temp_text = ax.text(
            0, 0, text, fontsize=fontsize, fontproperties=font_props, visible=False
        )
        bbox = temp_text.get_window_extent(renderer=renderer)
        temp_text.remove()

        # Convert pixels to data coordinates using the current x-axis scale
        # Get the width of 1 data unit in pixels
        xlim = ax.get_xlim()
        fig_width_inches = ax.figure.get_figwidth()
        # Approximate: bbox width in points, convert to data units
        # Use a more reliable conversion: points to data units via DPI and figure size
        bbox_width_points = bbox.width
        # Get axes position in figure coordinates
        ax_bbox = ax.get_position()
        ax_width_inches = ax_bbox.width * fig_width_inches
        data_range = xlim[1] - xlim[0]
        # pixels per inch * inches per data unit
        dpi = ax.figure.dpi
        pixels_per_inch = dpi
        inches_per_data_unit = ax_width_inches / data_range
        pixels_per_data_unit = pixels_per_inch * inches_per_data_unit

        if pixels_per_data_unit > 0:
            data_width = bbox_width_points / pixels_per_data_unit
        else:
            # Fallback to character-based estimate
            data_width = len(text) * 0.15

        return data_width

    def _calculate_text_height(self, ax, text: str, fontsize: int) -> float:
        """Calculate actual rendered text height in data coordinates.

        Args:
            ax: matplotlib Axes.
            text: Text string to measure.
            fontsize: Font size.

        Returns:
            Height in data coordinates.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.font_manager import FontProperties

        # Ensure figure has a canvas for rendering
        if ax.figure.canvas is None or not hasattr(ax.figure.canvas, "get_renderer"):
            FigureCanvasAgg(ax.figure)

        renderer = ax.figure.canvas.get_renderer()

        # Get font properties
        font_props = FontProperties(size=fontsize)

        # Create temporary text to get bbox in display coordinates
        temp_text = ax.text(
            0, 0, text, fontsize=fontsize, fontproperties=font_props, visible=False
        )
        bbox = temp_text.get_window_extent(renderer=renderer)
        temp_text.remove()

        # Convert pixels to data coordinates using the current y-axis scale
        # Get the height of 1 data unit in pixels
        ylim = ax.get_ylim()
        fig_height_inches = ax.figure.get_figheight()
        # Get axes position in figure coordinates
        ax_bbox = ax.get_position()
        ax_height_inches = ax_bbox.height * fig_height_inches
        data_range = ylim[1] - ylim[0]
        # pixels per inch * inches per data unit
        dpi = ax.figure.dpi
        pixels_per_inch = dpi
        inches_per_data_unit = ax_height_inches / data_range
        pixels_per_data_unit = pixels_per_inch * inches_per_data_unit

        if pixels_per_data_unit > 0:
            data_height = bbox.height / pixels_per_data_unit
        else:
            # Fallback to a reasonable estimate
            data_height = 0.15

        return data_height

    def _trace_value_chain(
        self, start_value: Value, operations: list[Operation]
    ) -> set[int]:
        """Trace all values in the SSA chain starting from a given value.

        This finds all intermediate values created from the start_value through
        gate operations within a block. For multi-qubit gates, only the
        corresponding result (same index) is added to the chain.

        Args:
            start_value: The starting value (typically a block input).
            operations: List of operations in the block.

        Returns:
            Set of value IDs in the chain.
        """
        chain = {id(start_value)}
        changed = True

        while changed:
            changed = False
            for op in operations:
                if isinstance(op, GateOperation):
                    # For each operand in the chain, add the corresponding result
                    for i, operand in enumerate(op.operands):
                        if id(operand) in chain and i < len(op.results):
                            result = op.results[i]
                            if id(result) not in chain:
                                chain.add(id(result))
                                changed = True

        return chain

    def _analyze_loop_affected_qubits(
        self,
        op: ForOperation,
        qubit_map: dict[int, int],
        value_id_map: dict[int, int] | None = None,
    ) -> list[int]:
        """Analyze which qubits are affected by a ForOperation.

        Args:
            op: ForOperation to analyze.
            qubit_map: Mapping from Value ID to qubit index.
            value_id_map: Mapping from dummy value IDs to actual value IDs.

        Returns:
            List of affected qubit indices.
        """
        if value_id_map is None:
            value_id_map = {}

        affected = set()

        def get_qubit_index(operand) -> int | None:
            """Get qubit index for an operand, handling array elements."""
            operand_id = value_id_map.get(id(operand), id(operand))

            # Direct lookup
            if operand_id in qubit_map:
                return qubit_map[operand_id]

            # Check if this is an array element - use parent array's qubit index
            if hasattr(operand, "parent_array") and operand.parent_array is not None:
                parent_id = id(operand.parent_array)
                if parent_id in qubit_map:
                    # For array elements, we need to determine which wire(s) are affected
                    # If the index is constant, use the specific wire
                    # If the index is symbolic (loop variable), all wires in the array are affected
                    if hasattr(operand, "element_indices") and operand.element_indices:
                        idx_value = operand.element_indices[0]
                        if idx_value.is_constant():
                            idx = idx_value.get_const()
                            if idx is not None and isinstance(idx, int):
                                # Specific element - add that wire
                                base_idx = qubit_map[parent_id]
                                return base_idx + idx
                        else:
                            # Symbolic index - conservatively mark all array qubits
                            base_idx = qubit_map[parent_id]
                            # We need array size - check parent's shape
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
                    return qubit_map[parent_id]

            return None

        def collect_from_ops(ops: list[Operation]) -> None:
            for inner_op in ops:
                if isinstance(inner_op, GateOperation):
                    for operand in inner_op.operands:
                        idx = get_qubit_index(operand)
                        if idx is not None:
                            affected.add(idx)
                elif isinstance(inner_op, CallBlockOperation):
                    for operand in inner_op.operands[1:]:  # Skip BlockValue
                        idx = get_qubit_index(operand)
                        if idx is not None:
                            affected.add(idx)
                elif isinstance(inner_op, ForOperation):
                    # Recursively collect from nested loops
                    collect_from_ops(inner_op.operations)

        collect_from_ops(op.operations)
        return list(affected)

    def _estimate_gate_width(
        self, op: GateOperation, param_values: dict | None = None
    ) -> float:
        """Estimate gate box width from label text without matplotlib axes.

        Uses character-based estimation suitable for layout phase
        (before figure creation). Also serves as floor for drawing phase.
        """
        import re

        label, has_param = self._get_gate_label(op, param_values)
        if not has_param:
            return self.style.gate_width

        # Strip TeX $ delimiters (not rendered)
        visual = label.replace("$", "")
        # TeX commands like \theta render as ~1 wide character
        visual = re.sub(r"\\[a-zA-Z]+", "X", visual)
        effective_len = len(visual)
        char_width = 0.14
        text_width = effective_len * char_width
        padding = 0.25
        return max(self.style.gate_width, text_width + 2 * padding)

    def _build_qubit_map(self, graph: Graph) -> dict[str, int]:
        """Build mapping from qubit Value IDs to wire indices.

        In SSA form, each operation creates new Values, but they represent
        the same qubit. We need to track these SSA chains and map all values
        in a chain to the same wire index.

        Uses logical_id for more robust tracking across inline blocks.

        Args:
            graph: Computation graph.

        Returns:
            Dictionary mapping Value ID to wire index (0-based).
        """
        # Track both logical_id and value_id mappings
        logical_id_to_qubit: dict[str, int] = {}  # logical_id -> Qubit index
        value_to_qubit: dict[int, int] = {}  # Value ID -> Qubit index
        qubit_names: dict[int, str] = {}  # Qubit index -> Display name
        next_idx = 0

        def build_chains(
            ops: list[Operation],
            value_id_map: dict[int, int]
            | None = None,  # Dummy value ID → Actual value ID
        ) -> None:
            nonlocal next_idx
            if value_id_map is None:
                value_id_map = {}

            for op in ops:
                if isinstance(op, QInitOperation):
                    # Allocate new qubit and start a chain
                    qubit = op.results[0]
                    qubit_id = value_id_map.get(id(qubit), id(qubit))

                    # Check if this is an ArrayValue (qubit array)
                    if isinstance(qubit, ArrayValue):
                        from qamomile.circuit.ir.types.primitives import QubitType

                        if isinstance(qubit.type, QubitType):
                            # Try to expand the array if size is known
                            if len(qubit.shape) == 1:  # 1D array (Vector)
                                size_value = qubit.shape[0]
                                # Check if size is constant
                                if size_value.is_constant():
                                    array_size = size_value.get_const()
                                    if array_size is not None and isinstance(
                                        array_size, int
                                    ):
                                        # Create individual qubit wires for each array element
                                        for i in range(array_size):
                                            element_key = f"{qubit.logical_id}_[{i}]"
                                            if element_key not in logical_id_to_qubit:
                                                logical_id_to_qubit[element_key] = (
                                                    next_idx
                                                )
                                                qubit_names[next_idx] = (
                                                    f"{qubit.name}[{i}]"
                                                )
                                                next_idx += 1
                                        # Also register the array itself
                                        value_to_qubit[qubit_id] = (
                                            logical_id_to_qubit.get(
                                                f"{qubit.logical_id}_[0]", next_idx
                                            )
                                        )
                                        continue

                    # Use logical_id for consistent tracking
                    if hasattr(qubit, "logical_id") and qubit.logical_id:
                        if qubit.logical_id not in logical_id_to_qubit:
                            logical_id_to_qubit[qubit.logical_id] = next_idx
                            qubit_names[next_idx] = qubit.name
                            next_idx += 1
                        value_to_qubit[qubit_id] = logical_id_to_qubit[qubit.logical_id]
                    else:
                        # Fallback to value ID
                        if qubit_id not in value_to_qubit:
                            value_to_qubit[qubit_id] = next_idx
                            qubit_names[next_idx] = qubit.name
                            next_idx += 1

                elif isinstance(op, GateOperation):
                    # Assign results to the same qubit as their operands
                    # For multi-qubit gates, operand[i] -> result[i]
                    for i, (operand, result) in enumerate(zip(op.operands, op.results)):
                        operand_id = value_id_map.get(id(operand), id(operand))
                        result_id = value_id_map.get(id(result), id(result))

                        qubit_idx = None
                        # When inside an inlined block (value_id_map non-empty and operand was
                        # remapped), skip logical_id lookup on the shared dummy object — its
                        # logical_id belongs to the BlockValue, not to the actual qubit
                        is_remapped = value_id_map and id(operand) in value_id_map

                        # Check if operand is an array element
                        if (
                            hasattr(operand, "parent_array")
                            and operand.parent_array is not None
                        ):
                            # Get the array element index
                            if len(operand.element_indices) == 1:
                                idx_value = operand.element_indices[0]
                                if idx_value.is_constant():
                                    idx = idx_value.get_const()
                                    if idx is not None and isinstance(idx, int):
                                        # Build the element key
                                        element_key = (
                                            f"{operand.parent_array.logical_id}_[{idx}]"
                                        )
                                        qubit_idx = logical_id_to_qubit.get(element_key)

                        if qubit_idx is None and not is_remapped:
                            if hasattr(operand, "logical_id") and operand.logical_id:
                                qubit_idx = logical_id_to_qubit.get(operand.logical_id)
                        if qubit_idx is None and operand_id in value_to_qubit:
                            qubit_idx = value_to_qubit[operand_id]

                        if qubit_idx is not None:
                            value_to_qubit[result_id] = qubit_idx
                            # Record name if not already set
                            if qubit_idx not in qubit_names:
                                qubit_names[qubit_idx] = operand.name
                            # Only write logical_id for non-remapped (real) values
                            if not is_remapped:
                                if hasattr(result, "logical_id") and result.logical_id:
                                    logical_id_to_qubit[result.logical_id] = qubit_idx
                                # Also track array elements
                                if (
                                    hasattr(result, "parent_array")
                                    and result.parent_array is not None
                                ):
                                    if len(result.element_indices) == 1:
                                        idx_value = result.element_indices[0]
                                        if idx_value.is_constant():
                                            idx = idx_value.get_const()
                                            if idx is not None and isinstance(idx, int):
                                                element_key = f"{result.parent_array.logical_id}_[{idx}]"
                                                logical_id_to_qubit[element_key] = (
                                                    qubit_idx
                                                )

                elif isinstance(op, CallBlockOperation):
                    if self.inline:
                        # If inlining, process block operations recursively
                        from qamomile.circuit.ir.block_value import BlockValue

                        block_value = op.operands[0]
                        if isinstance(block_value, BlockValue):
                            # Create mapping from dummy values to actual values
                            new_value_id_map = dict(value_id_map)
                            actual_inputs = op.operands[1:]  # Actual arguments

                            # Map all values in each qubit chain to the corresponding actual qubit
                            for i, (dummy_input, actual_input) in enumerate(
                                zip(block_value.input_values, actual_inputs)
                            ):
                                # Skip non-Qubit types (e.g., Float, Int)
                                from qamomile.circuit.ir.types.primitives import (
                                    QubitType,
                                )

                                if not isinstance(dummy_input.type, QubitType):
                                    continue

                                actual_id = value_id_map.get(
                                    id(actual_input), id(actual_input)
                                )

                                # Map the dummy input itself
                                new_value_id_map[id(dummy_input)] = actual_id

                                # Trace all values in the SSA chain for this qubit
                                chain_ids = self._trace_value_chain(
                                    dummy_input, block_value.operations
                                )
                                for chain_id in chain_ids:
                                    new_value_id_map[chain_id] = actual_id

                                # Ensure actual_input is registered in value_to_qubit
                                if actual_id not in value_to_qubit:
                                    # Try to use logical_id first
                                    qubit_idx = None
                                    if (
                                        hasattr(actual_input, "logical_id")
                                        and actual_input.logical_id
                                    ):
                                        qubit_idx = logical_id_to_qubit.get(
                                            actual_input.logical_id
                                        )

                                    # Try to find existing mapping from original value ID
                                    if qubit_idx is None:
                                        original_id = id(actual_input)
                                        if original_id in value_to_qubit:
                                            qubit_idx = value_to_qubit[original_id]

                                    # Only allocate new qubit index if not found anywhere
                                    if qubit_idx is None:
                                        # This should ideally not happen - indicates a mapping issue
                                        qubit_idx = next_idx
                                        next_idx += 1

                                    value_to_qubit[actual_id] = qubit_idx
                                    if (
                                        hasattr(actual_input, "logical_id")
                                        and actual_input.logical_id
                                    ):
                                        logical_id_to_qubit[actual_input.logical_id] = (
                                            qubit_idx
                                        )

                            build_chains(block_value.operations, new_value_id_map)

                            # Also map block results to the same qubit as inputs (needed for subsequent operations)
                            for i, (operand, result) in enumerate(
                                zip(op.operands[1:], op.results)
                            ):
                                # Skip non-Qubit types (e.g., Float, Int)
                                from qamomile.circuit.ir.types.primitives import (
                                    QubitType,
                                )

                                if not isinstance(result.type, QubitType):
                                    continue

                                operand_id = value_id_map.get(id(operand), id(operand))
                                result_id = value_id_map.get(id(result), id(result))

                                # Try to use logical_id first
                                qubit_idx = None
                                if (
                                    hasattr(operand, "logical_id")
                                    and operand.logical_id
                                ):
                                    qubit_idx = logical_id_to_qubit.get(
                                        operand.logical_id
                                    )
                                if qubit_idx is None and operand_id in value_to_qubit:
                                    qubit_idx = value_to_qubit[operand_id]

                                if qubit_idx is not None:
                                    value_to_qubit[result_id] = qubit_idx
                                    if (
                                        hasattr(result, "logical_id")
                                        and result.logical_id
                                    ):
                                        logical_id_to_qubit[result.logical_id] = (
                                            qubit_idx
                                        )
                                else:
                                    # If operand is not in value_to_qubit, allocate a new qubit
                                    value_to_qubit[operand_id] = next_idx
                                    value_to_qubit[result_id] = next_idx
                                    if (
                                        hasattr(result, "logical_id")
                                        and result.logical_id
                                    ):
                                        logical_id_to_qubit[result.logical_id] = (
                                            next_idx
                                        )
                                    next_idx += 1
                    else:
                        # Map block results to the same qubit as inputs
                        for i, (operand, result) in enumerate(
                            zip(op.operands[1:], op.results)
                        ):
                            # Skip non-Qubit types (e.g., Float, Int)
                            from qamomile.circuit.ir.types.primitives import QubitType

                            if not isinstance(result.type, QubitType):
                                continue

                            operand_id = value_id_map.get(id(operand), id(operand))
                            result_id = value_id_map.get(id(result), id(result))

                            # Try to use logical_id first
                            qubit_idx = None
                            if hasattr(operand, "logical_id") and operand.logical_id:
                                qubit_idx = logical_id_to_qubit.get(operand.logical_id)
                            if qubit_idx is None and operand_id in value_to_qubit:
                                qubit_idx = value_to_qubit[operand_id]

                            if qubit_idx is not None:
                                value_to_qubit[result_id] = qubit_idx
                                if hasattr(result, "logical_id") and result.logical_id:
                                    logical_id_to_qubit[result.logical_id] = qubit_idx
                            else:
                                # If operand is not in value_to_qubit, allocate a new qubit
                                value_to_qubit[operand_id] = next_idx
                                value_to_qubit[result_id] = next_idx
                                if hasattr(result, "logical_id") and result.logical_id:
                                    logical_id_to_qubit[result.logical_id] = next_idx
                                next_idx += 1

        build_chains(graph.operations)
        self.qubit_names = qubit_names
        self._num_qubits = next_idx
        return value_to_qubit

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
        positions: dict[tuple, int] = {}
        block_ranges: list[dict] = []  # Track inline block ranges
        block_widths: dict[tuple, float] = {}  # Track CallBlockOperation box widths
        column = 1  # Start at 1 to add margin between labels and gates
        max_depth = 0  # Track maximum nesting depth
        actual_width = 1.0  # Track actual width considering box sizes

        # Track which column each qubit is at (also start at 1)
        qubit_columns: dict[int, int] = defaultdict(lambda: 1)
        qubit_right_edges: dict[
            int, float
        ] = {}  # Track right edge of last box per qubit

        # Initialize right edges to 0.0 for layout calculation (logical coordinates)
        # The initial_wire_position offset is applied during the drawing phase
        for q_idx in set(qubit_map.values()):
            qubit_right_edges[q_idx] = 0.0
            # Pre-initialize qubit_columns to ensure entries exist
            _ = qubit_columns[q_idx]

        def process_operations(
            ops: list[Operation],
            value_id_map: dict[int, int]
            | None = None,  # Dummy value ID → Actual value ID
            depth: int = 0,  # Nesting depth for blocks
            scope_path: tuple = (),
            param_values: dict[int, float | int | str] | None = None,
        ) -> None:
            nonlocal column, max_depth, actual_width
            if value_id_map is None:
                value_id_map = {}
            if param_values is None:
                param_values = {}

            max_depth = max(max_depth, depth)

            for op in ops:
                op_key = (*scope_path, id(op))

                if isinstance(op, QInitOperation):
                    # Initialization doesn't take space
                    positions[op_key] = 0
                    continue

                if isinstance(op, ExpvalOp):
                    raise NotImplementedError(
                        f"Visualization of ExpvalOp is not yet supported. "
                        f"The circuit contains an expectation value calculation: {op}"
                    )

                if isinstance(op, ForOperation):
                    # Handle ForOperation: either fold as box or expand iterations
                    start_val = op.operands[0].get_const() if op.operands else 0
                    stop_val_raw = (
                        op.operands[1].get_const() if len(op.operands) > 1 else 0
                    )
                    step_val = (
                        op.operands[2].get_const() if len(op.operands) > 2 else 1
                    )

                    # Handle None values for start and step
                    if start_val is None:
                        start_val = 0
                    if step_val is None:
                        step_val = 1

                    # Skip zero-iteration loops only if stop is concrete
                    # If stop is symbolic (None), we can't determine iteration count
                    stop_val = stop_val_raw if stop_val_raw is not None else 0
                    if stop_val_raw is not None:
                        if step_val > 0 and start_val >= stop_val:
                            continue
                        if step_val < 0 and start_val <= stop_val:
                            continue

                    if self.fold_loops:
                        # Draw as a box (folded)
                        affected_qubits = self._analyze_loop_affected_qubits(
                            op, qubit_map, value_id_map
                        )

                        if affected_qubits:
                            start_column = max(qubit_columns[q] for q in affected_qubits)
                        elif qubit_columns:
                            start_column = max(qubit_columns.values())
                        else:
                            start_column = 0

                        # Use fixed width for folded loop box
                        loop_width = self.style.folded_loop_width
                        op_half_width = loop_width / 2
                        gap = self.style.gate_gap

                        # Overlap prevention
                        for q in affected_qubits:
                            if q in qubit_right_edges:
                                required_center = (
                                    qubit_right_edges[q] + gap + op_half_width
                                )
                                start_column = max(start_column, required_center)

                        positions[op_key] = start_column
                        block_widths[op_key] = loop_width

                        # Update qubit columns and right edges
                        for q in affected_qubits:
                            qubit_columns[q] = start_column + op_half_width + gap
                            qubit_right_edges[q] = start_column + op_half_width

                        column = max(column, start_column + 1)
                        actual_width = max(actual_width, start_column + op_half_width + 0.5)
                    else:
                        # Expand loop: process each iteration
                        num_iterations = (
                            (stop_val - start_val + step_val - 1) // step_val
                            if step_val > 0
                            else (start_val - stop_val - step_val - 1) // (-step_val)
                        )
                        for iteration in range(num_iterations):
                            iter_value = start_val + iteration * step_val
                            # Create param_values for this iteration
                            child_param_values = dict(param_values)
                            # Map loop variable to current iteration value
                            # The loop_var name is stored in ForOperation
                            child_param_values[f"_loop_{op.loop_var}"] = iter_value

                            # Process body operations for this iteration
                            process_operations(
                                op.operations,
                                value_id_map,
                                depth + 1,
                                scope_path=(*op_key, iteration),
                                param_values=child_param_values,
                            )
                    continue

                if isinstance(op, CallBlockOperation) and self.inline:
                    # If inlining, process block operations recursively
                    from qamomile.circuit.ir.block_value import BlockValue

                    block_value = op.operands[0]
                    if isinstance(block_value, BlockValue):
                        # Get affected qubits
                        affected_qubits = []
                        for operand in op.operands[1:]:  # Skip BlockValue
                            operand_id = value_id_map.get(id(operand), id(operand))
                            if operand_id in qubit_map:
                                affected_qubits.append(qubit_map[operand_id])

                        # Track maximum gate width in this block
                        max_gate_width = self.style.gate_width

                        # Account for the left border extent:
                        # The border extends (max_gate_width/2 + border_padding) left
                        # of the first gate's center. Advance qubit_right_edges so the
                        # overlap check inside the block pushes the first gate far enough
                        # right to clear the border.
                        border_padding = max(0.1, 0.3 - depth * 0.1)
                        border_extent = max_gate_width / 2 + border_padding
                        for q in affected_qubits:
                            if q in qubit_right_edges:
                                qubit_right_edges[q] += (
                                    border_extent - self.style.gate_width / 2
                                )

                        # Track the starting column for each affected qubit
                        qubit_start_columns = {}
                        for q in affected_qubits:
                            qubit_start_columns[q] = qubit_columns[q]

                        # Create mapping from dummy values to actual values
                        new_value_id_map = dict(value_id_map)
                        actual_inputs = op.operands[1:]  # Actual arguments

                        # Map all values in each qubit chain to the corresponding actual qubit
                        for i, (dummy_input, actual_input) in enumerate(
                            zip(block_value.input_values, actual_inputs)
                        ):
                            actual_id = value_id_map.get(
                                id(actual_input), id(actual_input)
                            )

                            # Map the dummy input itself
                            new_value_id_map[id(dummy_input)] = actual_id

                            # Also map actual_input's original ID if different
                            if id(actual_input) != actual_id:
                                new_value_id_map[id(actual_input)] = actual_id

                            # Trace all values in the SSA chain for this qubit
                            chain_ids = self._trace_value_chain(
                                dummy_input, block_value.operations
                            )
                            for chain_id in chain_ids:
                                new_value_id_map[chain_id] = actual_id

                        # Build child param_values for parameter resolution
                        child_param_values = dict(param_values)
                        for dummy_input, actual_input in zip(
                            block_value.input_values, actual_inputs
                        ):
                            if isinstance(actual_input, Value):
                                const = actual_input.get_const()
                                if const is not None:
                                    child_param_values[id(dummy_input)] = const
                                elif id(actual_input) in param_values:
                                    child_param_values[id(dummy_input)] = param_values[
                                        id(actual_input)
                                    ]
                                elif actual_input.is_parameter():
                                    child_param_values[id(dummy_input)] = (
                                        actual_input.parameter_name()
                                        or actual_input.name
                                    )

                        # Calculate max gate width in block
                        for block_op in block_value.operations:
                            if isinstance(block_op, GateOperation):
                                gate_width = self._estimate_gate_width(
                                    block_op, child_param_values
                                )
                                max_gate_width = max(max_gate_width, gate_width)

                        # Process block operations recursively
                        process_operations(
                            block_value.operations,
                            new_value_id_map,
                            depth + 1,
                            scope_path=op_key,
                            param_values=child_param_values,
                        )

                        # After processing, determine the column range used by this block
                        # by checking how far each affected qubit advanced
                        qubit_end_columns = {}
                        for q in affected_qubits:
                            qubit_end_columns[q] = qubit_columns[q]

                        # The block spans from the minimum start column to the maximum end column
                        if affected_qubits:
                            # Calculate operation columns in the block first
                            # Scan all descendant operations (including nested blocks) by prefix matching
                            block_op_columns = []
                            op_key_len = len(op_key)
                            for pos_key, pos_val in positions.items():
                                if (
                                    len(pos_key) > op_key_len
                                    and pos_key[:op_key_len] == op_key
                                    and pos_val
                                    > 0  # Skip QInitOperations at position 0
                                ):
                                    block_op_columns.append(pos_val)

                            # Use operation positions for actual_start (consistent with actual_end)
                            if block_op_columns:
                                actual_start = min(block_op_columns)
                                actual_end = max(block_op_columns)
                            else:
                                # Fallback if no operations found
                                actual_start = min(qubit_start_columns.values())
                                actual_end = max(qubit_end_columns.values()) - 1

                            block_ranges.append(
                                {
                                    "name": block_value.name or "block",
                                    "start_x": actual_start,
                                    "end_x": actual_end,
                                    "qubit_indices": affected_qubits,
                                    "depth": depth,
                                    "max_gate_width": max_gate_width,
                                }
                            )

                        # Set qubit_right_edges to the border's right edge so
                        # the overlap check spaces the next gate from the border,
                        # not from the last inner gate.
                        border_padding = max(0.1, 0.3 - depth * 0.1)
                        gate_border_right = (
                            actual_end + max_gate_width / 2 + border_padding
                        )

                        # Also account for title width
                        block_name = block_value.name or "block"
                        title_char_width = 0.12
                        box_left = actual_start - max_gate_width / 2 - border_padding
                        label_left = box_left + 0.1
                        title_text_width = len(block_name) * title_char_width
                        label_right = label_left + title_text_width + 0.1
                        border_right_edge = max(gate_border_right, label_right)

                        gap = 0.3
                        for q in affected_qubits:
                            qubit_right_edges[q] = border_right_edge
                            qubit_columns[q] = border_right_edge + gap
                    continue

                if isinstance(op, (GateOperation, CallBlockOperation)):
                    # Find which qubits this operation touches
                    affected_qubits = []
                    for i, operand in enumerate(op.operands):
                        # Skip BlockValue for CallBlockOperation
                        if isinstance(op, CallBlockOperation) and i == 0:
                            continue
                        operand_id = value_id_map.get(id(operand), id(operand))
                        if operand_id in qubit_map:
                            affected_qubits.append(qubit_map[operand_id])
                        # Handle array element access via parent_array
                        elif (
                            hasattr(operand, "parent_array")
                            and operand.parent_array is not None
                        ):
                            parent_id = id(operand.parent_array)
                            if parent_id in qubit_map:
                                # Get the element index
                                if (
                                    hasattr(operand, "element_indices")
                                    and operand.element_indices
                                ):
                                    idx_value = operand.element_indices[0]
                                    if idx_value.is_constant():
                                        idx = idx_value.get_const()
                                        if idx is not None and isinstance(idx, int):
                                            base_idx = qubit_map[parent_id]
                                            affected_qubits.append(base_idx + idx)
                                            continue
                                # Fallback to base index
                                affected_qubits.append(qubit_map[parent_id])

                    # Find the earliest column where all affected qubits are free
                    if affected_qubits:
                        min_column = max(qubit_columns[q] for q in affected_qubits)
                    else:
                        min_column = column

                    # Calculate required columns and operation half-width
                    columns_needed = 1
                    op_half_width = self.style.gate_width / 2  # default

                    if isinstance(op, GateOperation):
                        import math

                        estimated_w = self._estimate_gate_width(op, param_values)
                        columns_needed = max(1, math.ceil(estimated_w))
                        op_half_width = estimated_w / 2
                    elif isinstance(op, CallBlockOperation):
                        # Calculate box width for CallBlockOperation (same logic as _draw_call_block)
                        label = self._get_block_label(op, qubit_map)

                        # Character-based width calculation (conservative estimate)
                        # Use same char_width as gate estimation for consistency
                        char_width = 0.14
                        # Strip TeX for visual length (e.g., $\theta$ renders as 1 char)
                        import re

                        visual_label = label.replace("$", "")
                        visual_label = re.sub(r"\\[a-zA-Z]+", "X", visual_label)
                        text_width = len(visual_label) * char_width
                        text_padding = 0.25
                        box_width = max(
                            text_width + 2 * text_padding, self.style.gate_width
                        )

                        # Store box width for later use in drawing
                        block_widths[op_key] = box_width

                        # Convert width to columns (round up to ensure enough space)
                        columns_needed = max(1, int(box_width) + 1)
                        op_half_width = box_width / 2

                    # Overlap prevention: ensure this box's left edge clears previous box's right edge
                    gap = 0.3
                    for q in affected_qubits:
                        if q in qubit_right_edges:
                            required_center = qubit_right_edges[q] + gap + op_half_width
                            min_column = max(min_column, required_center)

                    # Place operation
                    positions[op_key] = min_column
                    column = max(column, min_column + columns_needed)

                    # Update qubit columns — use actual right edge + gap for tight spacing.
                    # The overlap prevention check (line 748-753) already ensures geometrically
                    # correct placement by accounting for both the current and next operation's
                    # half-widths. Using the integer columns_needed here over-advances the cursor,
                    # creating excess gaps after wide operations.
                    for q in affected_qubits:
                        qubit_columns[q] = min_column + op_half_width + gap

                    # Track right edge for future overlap checks
                    for q in affected_qubits:
                        qubit_right_edges[q] = min_column + op_half_width

                    # Update actual_width (simplified — use op_half_width directly)
                    op_actual_width = min_column + op_half_width + 0.5
                    actual_width = max(actual_width, op_actual_width)

        process_operations(graph.operations)
        # Ensure actual_width is at least as large as column
        actual_width = max(actual_width, column)
        return {
            "width": column,
            "positions": positions,
            "block_ranges": block_ranges,
            "max_depth": max_depth,
            "block_widths": block_widths,
            "actual_width": actual_width,
        }

    def _create_empty_figure(self) -> matplotlib.figure.Figure:
        """Create an empty figure for circuits with no qubits."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

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

    def _create_figure(self, num_qubits: int, layout: dict) -> matplotlib.figure.Figure:
        """Create matplotlib figure with appropriate size.

        Args:
            num_qubits: Number of qubits.
            layout: Layout dictionary.

        Returns:
            matplotlib Figure.
        """
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

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
        x_left = -1
        x_right = max(width + 1, layout.get("actual_width", width) + 0.5)

        if block_ranges:
            for br in block_ranges:
                depth = br["depth"]
                padding = max(0.1, 0.3 - depth * 0.1)  # Updated formula
                mgw = br.get("max_gate_width", self.style.gate_width)
                gate_border_right = br["end_x"] + mgw / 2 + padding
                border_left = br["start_x"] - mgw / 2 - padding

                # Account for title width (same logic as _draw_inlined_block_border)
                block_name = br.get("name", "block")
                title_char_width = 0.12
                label_left = border_left + 0.1
                title_text_width = len(block_name) * title_char_width
                label_right = label_left + title_text_width + 0.1
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

    def _build_output_label_map(self, qubit_map: dict[int, int]) -> dict[int, str]:
        """Build mapping from wire index to output variable names.

        Args:
            qubit_map: Mapping from Value ID to wire index.

        Returns:
            Dictionary mapping wire index to output variable name.
        """
        result = {}
        for i, value in enumerate(self.graph.output_values):
            if i >= len(self.graph.output_names):
                break
            wire = qubit_map.get(id(value))
            if wire is not None:
                result[wire] = self.graph.output_names[i]
        return result

    def _draw_wires(
        self,
        fig: matplotlib.figure.Figure,
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
        import matplotlib.lines as mlines

        ax = fig._qm_ax
        # Use dynamic x_right from axis limits for wire end
        wire_end = ax.get_xlim()[1] - 0.2  # leave small margin before axis edge

        # Draw horizontal wires for each qubit
        for i in range(num_qubits):
            y = self.qubit_y[i]
            line = mlines.Line2D(
                [0, wire_end],
                [y, y],
                color=self.style.wire_color,
                linewidth=1,
                zorder=PORDER_WIRE,
            )
            ax.add_line(line)

            # Draw left qubit label
            label = self.qubit_names.get(i, f"q{i}")
            ax.text(
                -0.5,
                y,
                label,
                ha="right",
                va="center",
                fontsize=self.style.font_size,
                zorder=PORDER_TEXT,
            )

        # Draw right-side output labels
        output_labels = self._build_output_label_map(qubit_map)
        if output_labels:
            for i in range(num_qubits):
                if i in output_labels:
                    ax.text(
                        wire_end + 0.3,
                        self.qubit_y[i],
                        output_labels[i],
                        ha="left",
                        va="center",
                        fontsize=self.style.font_size,
                        zorder=PORDER_TEXT,
                    )

    def _draw_operations(
        self,
        fig: matplotlib.figure.Figure,
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
            value_id_map: dict[int, int]
            | None = None,  # Dummy value ID → Actual value ID
            scope_path: tuple = (),
            param_values: dict[int, float | int | str] | None = None,
        ) -> None:
            if value_id_map is None:
                value_id_map = {}
            if param_values is None:
                param_values = {}

            for op in ops:
                op_key = (*scope_path, id(op))

                if isinstance(op, QInitOperation):
                    # Skip initialization (no visual representation)
                    continue
                elif isinstance(op, ExpvalOp):
                    raise NotImplementedError(
                        f"Visualization of ExpvalOp is not yet supported. "
                        f"The circuit contains an expectation value calculation: {op}"
                    )
                elif isinstance(op, ForOperation):
                    # Handle ForOperation based on fold_loops setting
                    start_val = op.operands[0].get_const() if op.operands else 0
                    stop_val_raw = (
                        op.operands[1].get_const() if len(op.operands) > 1 else 0
                    )
                    step_val = (
                        op.operands[2].get_const() if len(op.operands) > 2 else 1
                    )

                    # Handle None values for start and step
                    if start_val is None:
                        start_val = 0
                    if step_val is None:
                        step_val = 1

                    # Skip zero-iteration loops only if stop is concrete
                    stop_val = stop_val_raw if stop_val_raw is not None else 0
                    if stop_val_raw is not None:
                        if step_val > 0 and start_val >= stop_val:
                            continue
                        if step_val < 0 and start_val <= stop_val:
                            continue

                    if self.fold_loops:
                        # Draw as a box
                        if op_key in positions:
                            op_width = block_widths.get(op_key)
                            self._draw_for_loop_box(
                                fig, op, qubit_map, positions[op_key], op_width
                            )
                    else:
                        # Expand loop: draw each iteration's operations
                        # Can't expand if stop is symbolic
                        if stop_val_raw is None:
                            # Skip expansion for symbolic loops, treat as folded
                            if op_key in positions:
                                op_width = block_widths.get(op_key)
                                self._draw_for_loop_box(
                                    fig, op, qubit_map, positions[op_key], op_width
                                )
                            continue

                        num_iterations = (
                            (stop_val - start_val + step_val - 1) // step_val
                            if step_val > 0
                            else (start_val - stop_val - step_val - 1) // (-step_val)
                        )
                        for iteration in range(num_iterations):
                            iter_value = start_val + iteration * step_val
                            # Create param_values for this iteration
                            child_param_values = dict(param_values)
                            child_param_values[f"_loop_{op.loop_var}"] = iter_value

                            # Draw body operations for this iteration
                            draw_ops(
                                op.operations,
                                value_id_map,
                                scope_path=(*op_key, iteration),
                                param_values=child_param_values,
                            )
                elif isinstance(op, GateOperation):
                    self._draw_gate(
                        fig,
                        op,
                        qubit_map,
                        positions[op_key],
                        value_id_map,
                        param_values,
                    )
                elif isinstance(op, CallBlockOperation):
                    if self.inline:
                        # Recursively draw block operations
                        from qamomile.circuit.ir.block_value import BlockValue

                        block_value = op.operands[0]
                        if isinstance(block_value, BlockValue):
                            # Create mapping from dummy values to actual values
                            new_value_id_map = dict(value_id_map)
                            actual_inputs = op.operands[1:]  # Actual arguments

                            # Map all values in each qubit chain to the corresponding actual qubit
                            for i, (dummy_input, actual_input) in enumerate(
                                zip(block_value.input_values, actual_inputs)
                            ):
                                actual_id = value_id_map.get(
                                    id(actual_input), id(actual_input)
                                )

                                # Map the dummy input itself
                                new_value_id_map[id(dummy_input)] = actual_id

                                # Also map actual_input's original ID if different
                                if id(actual_input) != actual_id:
                                    new_value_id_map[id(actual_input)] = actual_id

                                # Trace all values in the SSA chain for this qubit
                                chain_ids = self._trace_value_chain(
                                    dummy_input, block_value.operations
                                )
                                for chain_id in chain_ids:
                                    new_value_id_map[chain_id] = actual_id

                            # Build child param_values
                            child_param_values = dict(param_values)
                            for dummy_input, actual_input in zip(
                                block_value.input_values, actual_inputs
                            ):
                                if isinstance(actual_input, Value):
                                    const = actual_input.get_const()
                                    if const is not None:
                                        child_param_values[id(dummy_input)] = const
                                    elif id(actual_input) in param_values:
                                        child_param_values[id(dummy_input)] = (
                                            param_values[id(actual_input)]
                                        )
                                    elif actual_input.is_parameter():
                                        child_param_values[id(dummy_input)] = (
                                            actual_input.parameter_name()
                                            or actual_input.name
                                        )

                            draw_ops(
                                block_value.operations,
                                new_value_id_map,
                                scope_path=op_key,
                                param_values=child_param_values,
                            )
                    else:
                        # Draw as box
                        op_width = block_widths.get(op_key)
                        self._draw_call_block(
                            fig, op, qubit_map, positions[op_key], op_width
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

        gate_half = self.style.gate_height / 2  # 0.325
        base_padding = 0.4
        label_height = 0.25  # text_height + 2*label_padding
        label_step = label_height + 0.1  # 0.35 per depth level
        overlap_step = label_height + 0.15  # 0.40 per extra overlap
        clearance = 0.15
        base_spacing = 1.0

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
            padding = max(0.1, base_padding - depth * 0.15)

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
        import matplotlib.patches as mpatches

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
        # Outer blocks (depth=0) have maximum padding, inner blocks have less
        base_padding = 0.3  # Base padding for outermost blocks
        depth_reduction = 0.1  # Padding reduction per depth level
        padding = max(0.1, base_padding - depth * depth_reduction)

        label_padding = 0.05  # Extra space above and below text
        label_height = text_height + 2 * label_padding

        # Vertical offset for labels based on depth and overlap index
        # depth: for nested blocks at different depths (inverted: outer blocks get MORE offset)
        # overlap_index: for blocks at the same depth and horizontal position
        depth_label_offset = (max_depth - depth) * (label_height + 0.1)
        overlap_label_offset = overlap_index * (label_height + 0.15)
        label_vertical_offset = depth_label_offset + overlap_label_offset

        # Calculate initial box boundaries based on gates
        gate_box_left = start_x - max_gate_width / 2 - padding
        gate_box_right = end_x + max_gate_width / 2 + padding

        # Calculate label position
        label_left = gate_box_left + 0.1  # Label starts with small offset from box

        # Extend box_right if title text is wider than gate-based width
        title_char_width = (
            0.12  # subfont_size(8pt) estimate, accounts for equal-aspect scaling
        )
        title_text_width = len(name) * title_char_width
        label_right = label_left + title_text_width + 0.1  # 0.1 right margin
        box_left = gate_box_left
        box_right = max(gate_box_right, label_right)

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
        rect = mpatches.Rectangle(
            (box_left, box_bottom),
            box_right - box_left,
            box_top - box_bottom,
            facecolor="none",  # No fill
            edgecolor=self.style.block_border_color,
            linewidth=1.5,
            linestyle="--",  # Dashed line
            zorder=PORDER_WIRE + 0.5,  # Above wire, below gates
        )
        ax.add_patch(rect)

        # Draw block name label (above the box, with vertical offset for nested blocks)
        ax.text(
            label_left,
            box_top - label_padding,
            name,
            ha="left",
            va="top",
            fontsize=self.style.subfont_size,
            color=self.style.block_border_color,
            zorder=PORDER_TEXT,
        )

    def _draw_gate(
        self,
        fig: matplotlib.figure.Figure,
        op: GateOperation,
        qubit_map: dict[str, int],
        x_pos: int,
        value_id_map: dict[int, int] | None = None,
        param_values: dict[int, float | int | str] | None = None,
    ) -> None:
        """Draw a gate operation.

        Args:
            fig: matplotlib Figure.
            op: Gate operation.
            qubit_map: Qubit to wire index mapping.
            x_pos: X position for the gate.
            value_id_map: Mapping from dummy value IDs to actual value IDs.
            param_values: Mapping from dummy value IDs to resolved parameter values.
        """
        if value_id_map is None:
            value_id_map = {}
        if param_values is None:
            param_values = {}

        ax = fig._qm_ax

        # Get affected qubits (map dummy IDs to actual IDs)
        qubit_indices = []
        for operand in op.operands:
            operand_id = value_id_map.get(id(operand), id(operand))
            qubit_idx = None

            # Try direct value ID mapping
            if operand_id in qubit_map:
                qubit_idx = qubit_map[operand_id]
            # Try parent array for array element access
            elif hasattr(operand, "parent_array") and operand.parent_array is not None:
                parent_id = id(operand.parent_array)
                if parent_id in qubit_map:
                    # Get the element index
                    if hasattr(operand, "element_indices") and operand.element_indices:
                        idx_value = operand.element_indices[0]
                        if idx_value.is_constant():
                            idx = idx_value.get_const()
                            if idx is not None and isinstance(idx, int):
                                base_idx = qubit_map[parent_id]
                                qubit_idx = base_idx + idx
                    if qubit_idx is None:
                        # Fallback to base index
                        qubit_idx = qubit_map[parent_id]
            # Fallback: try using logical_id if available
            elif hasattr(operand, "logical_id") and operand.logical_id:
                # Search for logical_id in the qubit_map by checking all mapped values
                # This is a last resort fallback
                for vid, qidx in qubit_map.items():
                    # We need to find if any mapped value has this logical_id
                    # This is not ideal but provides a safety net
                    pass  # Skip fallback search for now to avoid performance issues

            if qubit_idx is not None:
                qubit_indices.append(qubit_idx)

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
            param_values: Mapping from dummy value IDs to resolved parameter values.
        """
        import matplotlib.patches as mpatches

        # Gate label
        label, has_param = self._get_gate_label(op, param_values)

        # Use unified font size for all gates
        font_size = self.style.font_size

        # Calculate width dynamically for parametric gates
        if has_param:
            text_width = self._calculate_text_width(ax, label, font_size)
            padding = 0.25
            calculated_width = text_width + 2 * padding
            estimated_width = self._estimate_gate_width(op, param_values)
            width = max(estimated_width, calculated_width)
        else:
            width = self.style.gate_width

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
            param_values: Mapping from dummy value IDs to resolved parameter values.
        """
        import matplotlib.lines as mlines

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

        # Use unified font size for all gates
        font_size = self.style.font_size

        # Calculate width dynamically for parametric gates
        if has_param:
            text_width = self._calculate_text_width(ax, label, font_size)
            padding = 0.25
            calculated_width = text_width + 2 * padding
            estimated_width = self._estimate_gate_width(op, param_values)
            width = max(estimated_width, calculated_width)
        else:
            width = self.style.gate_width

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
            # Generic multi-qubit gate: draw boxes on all qubits
            import matplotlib.patches as mpatches

            for y in y_coords:
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

    def _draw_control_dot(self, ax, x: float, y: float) -> None:
        """Draw a control dot for controlled gates."""
        import matplotlib.patches as mpatches

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
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines

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
        import matplotlib.lines as mlines

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
        fig: matplotlib.figure.Figure,
        op: CallBlockOperation,
        qubit_map: dict[str, int],
        x_pos: int,
        block_width: float | None = None,
    ) -> None:
        """Draw a CallBlockOperation as a box.

        Args:
            fig: matplotlib Figure.
            op: CallBlock operation.
            qubit_map: Qubit to wire index mapping.
            x_pos: X position.
            block_width: Pre-calculated block width from layout phase.
        """
        import matplotlib.patches as mpatches

        ax = fig._qm_ax

        # Get affected qubits (skip first operand which is BlockValue)
        qubit_indices = []
        for operand in op.operands[1:]:  # Skip BlockValue
            operand_id = id(operand)
            if operand_id in qubit_map:
                qubit_indices.append(qubit_map[operand_id])

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
            # Fallback: use character-based estimate (same as layout phase)
            char_width = 0.14
            # Strip TeX for visual length (e.g., $\theta$ renders as 1 char)
            import re

            visual_label = label.replace("$", "")
            visual_label = re.sub(r"\\[a-zA-Z]+", "X", visual_label)
            text_width = len(visual_label) * char_width
            text_padding = 0.25
            width = max(text_width + 2 * text_padding, self.style.gate_width)

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

    def _draw_for_loop_box(
        self,
        fig: matplotlib.figure.Figure,
        op: ForOperation,
        qubit_map: dict[int, int],
        x_pos: float,
        block_width: float | None = None,
    ) -> None:
        """Draw a ForOperation as a box (folded mode).

        Args:
            fig: matplotlib Figure.
            op: ForOperation.
            qubit_map: Qubit to wire index mapping.
            x_pos: X position.
            block_width: Pre-calculated block width from layout phase.
        """
        import matplotlib.patches as mpatches

        ax = fig._qm_ax

        # Get affected qubits
        affected_qubits = self._analyze_loop_affected_qubits(op, qubit_map)

        if not affected_qubits:
            return

        # Convert to y coordinates using variable spacing
        y_coords = [self.qubit_y[q] for q in affected_qubits]
        min_y = min(y_coords)
        max_y = max(y_coords)

        # Get loop range for label
        start_val = op.operands[0].get_const() if op.operands else 0
        stop_val = op.operands[1].get_const() if len(op.operands) > 1 else 0
        step_val = op.operands[2].get_const() if len(op.operands) > 2 else 1

        if start_val is None:
            start_val = "?"
        if stop_val is None:
            stop_val = "?"
        if step_val is None or step_val == 1:
            label = f"for {op.loop_var} in [{start_val}..{stop_val})"
        else:
            label = f"for {op.loop_var} in [{start_val}..{stop_val}):{step_val}"

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
        else:
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
        from qamomile.circuit.ir.block_value import BlockValue

        block_value = op.operands[0]
        if not isinstance(block_value, BlockValue):
            return "block"

        label = block_value.name or "block"

        # Collect non-qubit parameter values
        params = []
        for arg_name, actual_input in zip(block_value.label_args, op.operands[1:]):
            if id(actual_input) not in qubit_map:
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
                    if param_values and id(op.theta) in param_values:
                        resolved = param_values[id(op.theta)]
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
