"""Circuit layout engine: measure-then-place coordinate computation.

This module provides CircuitLayoutEngine, which handles the 2-phase
layout computation for circuit visualization. It has no matplotlib dependency.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING

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
    MeasureOperation,
    MeasureVectorOperation,
)
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.value import ArrayValue

from .analyzer import CircuitAnalyzer
from .style import CircuitStyle
from .types import (
    BlockMeasure,
    GateMeasure,
    IfMeasure,
    LayoutResult,
    LayoutState,
    LoopMeasure,
    MeasureNode,
    SkipMeasure,
)


if TYPE_CHECKING:
    from qamomile.circuit.ir.graph import Graph


class CircuitLayoutEngine:
    """Computes layout coordinates for circuit visualization.

    Uses a 2-phase approach:
    1. Measure Phase: Recursively compute widths without coordinates.
    2. Place Phase: Assign coordinates using pre-computed widths.

    Has no matplotlib dependency.
    """

    def __init__(self, analyzer: CircuitAnalyzer, style: CircuitStyle):
        self.analyzer = analyzer
        self.style = style

    def _measure_generic_operation(
        self,
        op: Operation,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> GateMeasure:
        """Measure width of a generic operation (gate, measurement, block box)."""
        if isinstance(op, GateOperation):
            estimated_w = self.analyzer._estimate_gate_width(op, param_values)
            return GateMeasure(
                estimated_width=estimated_w,
                half_width=estimated_w / 2,
                columns_needed=max(1, math.ceil(estimated_w)),
            )
        elif isinstance(op, CallBlockOperation):
            label = self.analyzer._get_block_label(op, qubit_map)
            box_width = self.analyzer._estimate_label_box_width(label)
            return GateMeasure(
                estimated_width=box_width,
                half_width=box_width / 2,
                columns_needed=max(1, int(box_width) + 1),
                is_block_box=True,
                box_width=box_width,
            )
        elif isinstance(op, CompositeGateOperation):
            box_width = self.analyzer._estimate_label_box_width(op.name.upper())
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
            box_width = self.analyzer._estimate_label_box_width(label)
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
            box_width = self.analyzer._estimate_label_box_width(label)
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
        border_padding = self.analyzer._compute_border_padding(depth)
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
                    self.analyzer._resolve_operand_to_qubit_indices(
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
                idx = self.analyzer._resolve_operand_to_qubit_index(
                    operand, qubit_map, logical_id_remap, param_values
                )
                if idx is not None:
                    control_qubit_indices.append(idx)
            affected_qubits = list(control_qubit_indices)
            for operand in op.target_operands:
                idx = self.analyzer._resolve_operand_to_qubit_index(
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
                idx = self.analyzer._resolve_operand_to_qubit_index(
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

        new_logical_id_remap, child_param_values = (
            self.analyzer._build_block_value_mappings(
                block_value,
                actual_inputs,
                logical_id_remap,
                param_values,
                qubit_map=qubit_map,
            )
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

        max_gate_width = self.analyzer._max_block_gate_width(
            block_value.operations, child_param_values
        )

        children = self._measure_operations(
            block_value.operations,
            qubit_map,
            new_logical_id_remap,
            child_param_values,
            depth + 1,
        )

        border_padding = self.analyzer._compute_border_padding(depth)
        content_width = self._compute_content_width(children, max_gate_width, depth)

        # Label width using the same estimation as _estimate_label_box_width
        label_width = self.analyzer._estimate_label_box_width(block_name)

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
        range_str = self.analyzer._format_range_str(op, set(), param_values)
        header = f"for {op.loop_var} in {range_str}"

        body_lines: list[str] = []
        for body_op in op.operations:
            expr = self.analyzer._format_operation_as_expression(
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
        start_val, stop_val_raw, step_val = self.analyzer._evaluate_loop_range(
            op, param_values
        )

        if self.analyzer._is_zero_iteration_loop(start_val, stop_val_raw, step_val):
            return SkipMeasure()

        affected_qubits = self.analyzer._analyze_loop_affected_qubits(
            op, qubit_map, logical_id_remap, param_values
        )

        if self.analyzer.fold_loops:
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
        num_iterations = self.analyzer._compute_loop_iterations(
            start_val, stop_val_raw, step_val
        )
        iteration_children: list[list[MeasureNode]] = []
        iteration_widths: list[float] = []
        iter_param_values_list: list[dict] = []

        for iteration in range(num_iterations):
            iter_value = start_val + iteration * step_val
            child_param_values = dict(param_values)
            child_param_values[f"_loop_{op.loop_var}"] = iter_value

            self.analyzer._evaluate_loop_body_intermediates(
                op.operations, child_param_values
            )

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
        affected_qubits = self.analyzer._analyze_loop_affected_qubits(
            op, qubit_map, logical_id_remap, param_values
        )
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
                    idx = self.analyzer._resolve_operand_to_qubit_index(
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

        if self.analyzer.fold_loops:
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

    def _measure_for_items_operation(
        self,
        op: ForItemsOperation,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
    ) -> LoopMeasure | SkipMeasure:
        """Measure width of a ForItemsOperation (fold or unfold)."""
        affected_qubits = self.analyzer._analyze_loop_affected_qubits(
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
            self.analyzer._materialize_dict_entries(dict_value) if dict_value else None
        )

        if self.analyzer.fold_loops or materialized is None:
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

            if isinstance(
                op, CallBlockOperation
            ) and self.analyzer._should_inline_at_depth(depth):
                result.append(
                    self._measure_inline_block(
                        op, qubit_map, logical_id_remap, param_values, depth
                    )
                )
                continue

            if isinstance(
                op, ControlledUOperation
            ) and self.analyzer._should_inline_at_depth(depth):
                result.append(
                    self._measure_inline_block(
                        op, qubit_map, logical_id_remap, param_values, depth
                    )
                )
                continue

            if (
                isinstance(op, CompositeGateOperation)
                and self.analyzer.expand_composite
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
                self.analyzer._resolve_operand_to_qubit_indices(
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
                state.qubit_end_positions[q] = min_column + op_half_width
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
                state.qubit_end_positions[q] = min_column + op_half_width

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

        new_logical_id_remap, child_param_values = (
            self.analyzer._build_block_value_mappings(
                block_value,
                actual_inputs,
                logical_id_remap,
                param_values,
                qubit_map=qubit_map,
            )
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
        _, border_right_edge = self.analyzer._compute_block_box_bounds(
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

            if isinstance(
                op, CallBlockOperation
            ) and self.analyzer._should_inline_at_depth(depth):
                state.inlined_op_keys.add(op_key)
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

            if isinstance(
                op, ControlledUOperation
            ) and self.analyzer._should_inline_at_depth(depth):
                state.inlined_op_keys.add(op_key)
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
                and self.analyzer.expand_composite
                and op.has_implementation
            ):
                state.inlined_op_keys.add(op_key)
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

    def compute_layout(
        self, graph: Graph, qubit_map: dict[str, int], num_qubits: int
    ) -> LayoutResult:
        """Calculate layout positions for operations using 2-phase approach.

        Phase 1 (Measure): Recursively compute widths without assigning coordinates.
        Phase 2 (Place): Assign coordinates using pre-computed widths.

        Args:
            graph: IR Graph containing operations to lay out.
            qubit_map: Mapping from logical_id to qubit wire index.
            num_qubits: Total number of qubit wires.

        Returns:
            LayoutResult with all computed positions and sizing.
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

        # Phase 3: Compute qubit Y positions
        qubit_y, max_above, max_below = self._compute_qubit_y_positions(
            num_qubits, state.block_ranges
        )

        return LayoutResult(
            width=state.column,
            positions=state.positions,
            block_ranges=state.block_ranges,
            max_depth=state.max_depth,
            block_widths=state.block_widths,
            actual_width=state.actual_width,
            first_gate_x=state.first_gate_x if state.first_gate_x is not None else 1.0,
            first_gate_half_width=state.first_gate_half_width,
            qubit_y=qubit_y,
            qubit_end_positions=state.qubit_end_positions,
            inlined_op_keys=state.inlined_op_keys,
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

        # Flip: qubit 0 at top (highest y), qubit N-1 at bottom (y≈0)
        max_val = y_positions[-1]
        for i in range(num_qubits):
            y_positions[i] = max_val - y_positions[i]

        return y_positions, max_above, max_below
