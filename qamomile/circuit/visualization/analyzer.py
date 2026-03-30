"""Circuit analysis: IR inspection, value resolution, and label generation.

This module provides CircuitAnalyzer, which handles all IR-level analysis
for the circuit visualization pipeline. It has no matplotlib dependency.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

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

from .geometry import compute_border_padding
from .style import CircuitStyle
from .types import _TEX_SYMBOLS
from .visual_ir import (
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
    from qamomile.circuit.ir.graph import Graph


class CircuitAnalyzer:
    """Analyzes IR graphs for circuit visualization.

    Handles qubit mapping, value resolution, label generation,
    and width estimation. Has no matplotlib dependency.
    """

    def __init__(
        self,
        graph: Graph,
        style: CircuitStyle,
        inline: bool = False,
        fold_loops: bool = True,
        expand_composite: bool = False,
        inline_depth: int | None = None,
    ):
        self.graph = graph
        self.style = style
        self.inline = inline
        self.fold_loops = fold_loops
        self.expand_composite = expand_composite
        self.inline_depth = inline_depth

    def _should_inline_at_depth(self, depth: int) -> bool:
        """Whether to inline CallBlock/ControlledU at this nesting depth."""
        return self.inline and (self.inline_depth is None or depth < self.inline_depth)

    def build_qubit_map(
        self, graph: Graph
    ) -> tuple[dict[str, int], dict[int, str], int]:
        """Build mapping from qubit logical_id to wire indices.

        In SSA form, each operation creates new Values via next_version(),
        which preserves logical_id. This means all versions of a qubit share
        the same logical_id, so we only need logical_id-based tracking.

        Args:
            graph: Computation graph.

        Returns:
            Tuple of (qubit_map, qubit_names, num_qubits).
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
            for operand, result in zip(operands, results):
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
                            qubit_map[result.logical_id] = qubit_map[parent_lid]
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
                            array_size = None
                            if size_value.is_constant():
                                array_size = size_value.get_const()
                            elif param_values:
                                # Try resolving from param_values (e.g. inlined block)
                                pv = param_values.get(size_value.logical_id)
                                if pv is not None and isinstance(pv, (int, float)):
                                    array_size = int(pv)
                            if array_size is None:
                                raise ValueError(
                                    f"Cannot visualize circuit: qubit array "
                                    f"'{qubits.name}' has symbolic size. "
                                    f"Please provide concrete values for all "
                                    f"size parameters when calling draw()."
                                )

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

                        # Build child_param_values: propagate non-qubit actual values
                        child_param_values = dict(param_values) if param_values else {}
                        for dummy_input, actual_input in zip(
                            block_value.input_values, actual_inputs, strict=True
                        ):
                            if isinstance(dummy_input.type, QubitType):
                                continue
                            c = actual_input.get_const()
                            if c is not None:
                                child_param_values[dummy_input.logical_id] = c
                            elif (
                                param_values and actual_input.logical_id in param_values
                            ):
                                child_param_values[dummy_input.logical_id] = (
                                    param_values[actual_input.logical_id]
                                )
                        # Propagate ArrayValue shape dimensions
                        for dummy_input, actual_input in zip(
                            block_value.input_values, actual_inputs, strict=True
                        ):
                            if isinstance(actual_input, ArrayValue) and isinstance(
                                dummy_input, ArrayValue
                            ):
                                if dummy_input.shape and actual_input.shape:
                                    for dummy_dim, actual_dim in zip(
                                        dummy_input.shape, actual_input.shape
                                    ):
                                        const = actual_dim.get_const()
                                        if const is not None:
                                            child_param_values[dummy_dim.logical_id] = (
                                                const
                                            )
                                        elif (
                                            param_values
                                            and actual_dim.logical_id in param_values
                                        ):
                                            child_param_values[dummy_dim.logical_id] = (
                                                param_values[actual_dim.logical_id]
                                            )

                        build_chains(
                            block_value.operations,
                            new_remap,
                            depth + 1,
                            child_param_values,
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

                    # Handle fresh-return qubits (results with no corresponding operand)
                    for fresh_result in qubit_results[len(qubit_operands) :]:
                        if fresh_result.logical_id in qubit_map:
                            continue
                        if isinstance(fresh_result, ArrayValue):
                            # Resolve array shape using block param mapping
                            block_value = op.operands[0]
                            fresh_pv = dict(param_values) if param_values else {}
                            for dummy, actual in zip(
                                block_value.input_values, op.operands[1:]
                            ):
                                c = actual.get_const()
                                if c is not None:
                                    fresh_pv[dummy.logical_id] = c
                                elif param_values and actual.logical_id in param_values:
                                    fresh_pv[dummy.logical_id] = param_values[
                                        actual.logical_id
                                    ]
                            if fresh_result.shape:
                                dim_val = fresh_result.shape[0]
                                size = None
                                c = dim_val.get_const()
                                if c is not None:
                                    size = int(c)
                                elif dim_val.logical_id in fresh_pv:
                                    pv = fresh_pv[dim_val.logical_id]
                                    if isinstance(pv, (int, float)):
                                        size = int(pv)
                                if size is not None:
                                    for i in range(size):
                                        ek = f"{fresh_result.logical_id}_[{i}]"
                                        if ek not in qubit_map:
                                            qubit_map[ek] = next_idx
                                            name = fresh_result.name or "q"
                                            qubit_names[next_idx] = f"{name}[{i}]"
                                            next_idx += 1
                                    qubit_map[fresh_result.logical_id] = qubit_map.get(
                                        f"{fresh_result.logical_id}_[0]",
                                        next_idx,
                                    )
                                else:
                                    raise ValueError(
                                        f"Cannot visualize circuit: fresh-return "
                                        f"qubit array '{fresh_result.name}' has "
                                        f"symbolic size. Please provide concrete "
                                        f"values for all size parameters."
                                    )
                        else:
                            qubit_map[fresh_result.logical_id] = next_idx
                            qubit_names[next_idx] = fresh_result.name or f"q{next_idx}"
                            next_idx += 1

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
                    else:
                        resolved = self._resolve_array_element_lid(
                            source, qubit_map, logical_id_remap, param_values
                        )
                        resolved_idx = qubit_map.get(resolved)
                        if resolved_idx is not None:
                            qubit_map[result.logical_id] = resolved_idx

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
                        # Build child_param_values for non-qubit inputs
                        child_param_values = dict(param_values) if param_values else {}
                        for dummy_input, actual_input in zip(
                            block_value.input_values, op.target_operands
                        ):
                            if isinstance(dummy_input.type, QubitType):
                                continue
                            c = actual_input.get_const()
                            if c is not None:
                                child_param_values[dummy_input.logical_id] = c
                            elif (
                                param_values and actual_input.logical_id in param_values
                            ):
                                child_param_values[dummy_input.logical_id] = (
                                    param_values[actual_input.logical_id]
                                )
                        build_chains(
                            block_value.operations,
                            new_remap,
                            depth + 1,
                            child_param_values,
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
                        # Build child_param_values for non-qubit inputs
                        child_param_values = dict(param_values) if param_values else {}
                        for dummy_input, actual_input in zip(
                            block_value.input_values, op.target_qubits
                        ):
                            if isinstance(dummy_input.type, QubitType):
                                continue
                            c = actual_input.get_const()
                            if c is not None:
                                child_param_values[dummy_input.logical_id] = c
                            elif (
                                param_values and actual_input.logical_id in param_values
                            ):
                                child_param_values[dummy_input.logical_id] = (
                                    param_values[actual_input.logical_id]
                                )
                        build_chains(
                            block_value.operations,
                            new_remap,
                            depth + 1,
                            child_param_values,
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
                        if dict_value is not None
                        else None
                    )
                    if materialized is not None and (
                        not self.fold_loops or self.inline
                    ):
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
        return qubit_map, qubit_names, next_idx

    # ------------------------------------------------------------------
    # Visual IR construction
    # ------------------------------------------------------------------

    def build_visual_ir(
        self,
        graph: Graph,
        qubit_map: dict[str, int],
        qubit_names: dict[int, str],
        num_qubits: int,
    ) -> VisualCircuit:
        """Build a Visual IR tree from the IR graph.

        Walks all operations, resolving labels, qubit indices, and widths
        into pre-computed VisualNode dataclasses. The resulting VisualCircuit
        can be consumed by Layout and Renderer without any Analyzer access.

        Args:
            graph: IR computation graph.
            qubit_map: Mapping from logical_id to wire index.
            qubit_names: Mapping from wire index to display name.
            num_qubits: Total number of qubit wires.

        Returns:
            VisualCircuit containing the VisualNode tree.
        """
        children = self._build_visual_nodes(
            graph.operations, qubit_map, {}, {}, depth=0, scope_path=()
        )
        return VisualCircuit(
            children=children,
            qubit_map=qubit_map,
            qubit_names=qubit_names,
            num_qubits=num_qubits,
            output_names=getattr(graph, "output_names", None) or [],
        )

    def _build_visual_nodes(
        self,
        ops: list[Operation],
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
        scope_path: tuple,
    ) -> list[VisualNode]:
        """Recursively build VisualNode list from IR operations.

        This is the core dispatch method that replaces both Layout's
        _measure_operations and Renderer's draw_ops dispatch.
        """
        result: list[VisualNode] = []

        for op in ops:
            node_key = (*scope_path, id(op))

            if isinstance(op, QInitOperation):
                result.append(VSkip(node_key=node_key))
                continue

            if isinstance(op, CastOperation):
                result.append(VSkip(node_key=node_key))
                continue

            if isinstance(op, BinOp):
                continue  # No visual representation

            if isinstance(op, ExpvalOp):
                node = self._build_vexpval(
                    op, node_key, qubit_map, logical_id_remap, param_values
                )
                result.append(node)
                continue

            if isinstance(op, WhileOperation):
                raise NotImplementedError(
                    "Circuit visualization does not yet support WhileOperation. "
                    "This feature will be added in a future release."
                )

            if isinstance(op, IfOperation):
                raise NotImplementedError(
                    "Circuit visualization does not yet support IfOperation. "
                    "This feature will be added in a future release."
                )

            if isinstance(op, ForItemsOperation):
                node = self._build_vfor_items(
                    op,
                    node_key,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                    depth,
                    scope_path,
                )
                result.append(node)
                continue

            if isinstance(op, ForOperation):
                node = self._build_vfor(
                    op,
                    node_key,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                    depth,
                    scope_path,
                )
                result.append(node)
                continue

            if isinstance(op, CallBlockOperation) and self._should_inline_at_depth(
                depth
            ):
                node = self._build_vinline_block(
                    op,
                    node_key,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                    depth,
                    scope_path,
                )
                result.append(node)
                continue

            if isinstance(op, ControlledUOperation) and self._should_inline_at_depth(
                depth
            ):
                node = self._build_vinline_block(
                    op,
                    node_key,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                    depth,
                    scope_path,
                )
                result.append(node)
                continue

            if (
                isinstance(op, CompositeGateOperation)
                and self.expand_composite
                and op.has_implementation
            ):
                node = self._build_vinline_block(
                    op,
                    node_key,
                    qubit_map,
                    logical_id_remap,
                    param_values,
                    depth,
                    scope_path,
                )
                result.append(node)
                continue

            # Generic: GateOperation, CallBlock/ControlledU/CompositeGate (box mode),
            # MeasureOperation, MeasureVectorOperation
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
                node = self._build_vgate(
                    op, node_key, qubit_map, logical_id_remap, param_values
                )
                result.append(node)

        return result

    def _build_vgate(
        self,
        op: Operation,
        node_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> VGate:
        """Build a VGate node for gates, measurements, and block boxes."""
        if isinstance(op, GateOperation):
            label, has_param = self._get_gate_label(op, param_values)
            estimated_width = self._estimate_gate_width(op, param_values)
            qubit_indices: list[int] = []
            for operand in op.operands:
                idx = self._resolve_operand_to_qubit_index(
                    operand, qubit_map, logical_id_remap, param_values
                )
                if idx is not None:
                    qubit_indices.append(idx)
            return VGate(
                node_key=node_key,
                label=label,
                qubit_indices=qubit_indices,
                estimated_width=estimated_width,
                kind=VGateKind.GATE,
                gate_type=op.gate_type,
                has_param=has_param,
            )

        if isinstance(op, MeasureOperation):
            gate_width = self.style.gate_width
            qubit_indices = []
            if op.operands:
                idx = self._resolve_operand_to_qubit_index(
                    op.operands[0], qubit_map, logical_id_remap, param_values
                )
                if idx is not None:
                    qubit_indices.append(idx)
            return VGate(
                node_key=node_key,
                label="M",
                qubit_indices=qubit_indices,
                estimated_width=gate_width,
                kind=VGateKind.MEASURE,
            )

        if isinstance(op, MeasureVectorOperation):
            gate_width = self.style.gate_width
            qubit_indices = []
            if op.operands:
                qubit_indices = self._resolve_operand_to_qubit_indices(
                    op.operands[0], qubit_map, logical_id_remap, param_values
                )
            return VGate(
                node_key=node_key,
                label="M",
                qubit_indices=qubit_indices,
                estimated_width=gate_width,
                kind=VGateKind.MEASURE_VECTOR,
            )

        if isinstance(op, CallBlockOperation):
            label = self._get_block_label(op, qubit_map, param_values=param_values)
            box_width = self._estimate_block_label_box_width(label)
            qubit_indices = []
            for operand in op.operands[1:]:  # Skip BlockValue
                qubit_indices.extend(
                    self._resolve_operand_to_qubit_indices(
                        operand, qubit_map, logical_id_remap, param_values
                    )
                )
            # Fresh-return pattern: if no qubit operands, resolve from results
            if not qubit_indices:
                for result_val in op.results:
                    if isinstance(result_val.type, QubitType):
                        qubit_indices.extend(
                            self._resolve_operand_to_qubit_indices(
                                result_val, qubit_map, logical_id_remap, param_values
                            )
                        )
            return VGate(
                node_key=node_key,
                label=label,
                qubit_indices=qubit_indices,
                estimated_width=box_width,
                kind=VGateKind.BLOCK_BOX,
                box_width=box_width,
            )

        if isinstance(op, CompositeGateOperation):
            label = op.name.upper()
            box_width = self._estimate_label_box_width(label)
            qubit_indices = []
            for qval in list(op.control_qubits) + list(op.target_qubits):
                idx = self._resolve_operand_to_qubit_index(
                    qval, qubit_map, logical_id_remap, param_values
                )
                if idx is not None:
                    qubit_indices.append(idx)
            return VGate(
                node_key=node_key,
                label=label,
                qubit_indices=qubit_indices,
                estimated_width=box_width,
                kind=VGateKind.COMPOSITE_BOX,
                box_width=box_width,
            )

        if isinstance(op, ControlledUOperation):
            u_name = getattr(op.block, "name", "U") or "U"
            power_val = self._resolve_controlled_u_power(op, param_values)
            label = f"{u_name}^{power_val}" if power_val > 1 else u_name
            box_width = self._estimate_label_box_width(label)
            # Control qubits first, then target qubits
            control_indices: list[int] = []
            for qval in op.control_operands:
                idx = self._resolve_operand_to_qubit_index(
                    qval, qubit_map, logical_id_remap, param_values
                )
                if idx is not None:
                    control_indices.append(idx)
            target_indices: list[int] = []
            for qval in op.target_operands:
                idx = self._resolve_operand_to_qubit_index(
                    qval, qubit_map, logical_id_remap, param_values
                )
                if idx is not None:
                    target_indices.append(idx)
            return VGate(
                node_key=node_key,
                label=label,
                qubit_indices=control_indices + target_indices,
                estimated_width=box_width,
                kind=VGateKind.CONTROLLED_U_BOX,
                box_width=box_width,
                control_count=len(control_indices),
            )

        raise TypeError(
            f"Unsupported operation type for _build_vgate: {type(op).__name__}"
        )

    def _build_vexpval(
        self,
        op: ExpvalOp,
        node_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> VGate:
        """Build a VGate node for an ExpvalOp."""
        label = "<H>"
        box_width = self._estimate_label_box_width(label)
        qubit_indices: list[int] = []
        if op.operands:
            qubit_indices = self._resolve_operand_to_qubit_indices(
                op.operands[0], qubit_map, logical_id_remap, param_values
            )
        return VGate(
            node_key=node_key,
            label=label,
            qubit_indices=qubit_indices,
            estimated_width=box_width,
            kind=VGateKind.EXPVAL,
            box_width=box_width,
        )

    def _build_vinline_block(
        self,
        op: CallBlockOperation | ControlledUOperation | CompositeGateOperation,
        node_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
        scope_path: tuple,
    ) -> VInlineBlock:
        """Build a VInlineBlock node for an inlined block operation."""
        # Extract block_value, affected_qubits, and actual_inputs based on op type
        if isinstance(op, CallBlockOperation):
            block_value = op.operands[0]
            assert isinstance(block_value, BlockValue)
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
            assert isinstance(block_value, BlockValue)
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
            assert isinstance(block_value, BlockValue)
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
            raise TypeError(f"Unsupported inline block type: {type(op).__name__}")

        new_logical_id_remap, child_param_values = self._build_block_value_mappings(
            block_value,
            actual_inputs,
            logical_id_remap,
            param_values,
            qubit_map=qubit_map,
        )

        self._evaluate_loop_body_intermediates(
            block_value.operations, child_param_values
        )

        # Include qubits created inside the block body via QInitOperation
        for body_op in block_value.operations:
            if isinstance(body_op, QInitOperation):
                result_val = body_op.results[0]
                resolved = self._resolve_operand_to_qubit_indices(
                    result_val, qubit_map, logical_id_remap, param_values
                )
                for idx in resolved:
                    if idx not in affected_qubits:
                        affected_qubits.append(idx)

        # Fresh-return fallback: resolve from results (same as box mode)
        if not affected_qubits and isinstance(op, CallBlockOperation):
            for result_val in op.results:
                if isinstance(result_val.type, QubitType):
                    affected_qubits.extend(
                        self._resolve_operand_to_qubit_indices(
                            result_val, qubit_map, logical_id_remap, param_values
                        )
                    )

        max_gate_width = self._max_block_gate_width(
            block_value.operations, child_param_values
        )

        children = self._build_visual_nodes(
            block_value.operations,
            qubit_map,
            new_logical_id_remap,
            child_param_values,
            depth + 1,
            node_key,
        )

        # Update max_gate_width from children
        children_max = self._max_visual_gate_width(children)
        max_gate_width = max(max_gate_width, children_max)

        border_padding = compute_border_padding(self.style, depth)
        content_width = self._compute_visual_content_width(
            children, max_gate_width, depth
        )
        power = (
            self._resolve_controlled_u_power(op, param_values)
            if isinstance(op, ControlledUOperation)
            else 1
        )
        # Inner box label width (block name only, no power text)
        label_width = self._estimate_label_box_width(block_name)
        if power > 1:
            # Account for outer wrapper box: margin on each side + pow label width
            margin = self.style.power_wrapper_margin
            pow_label_width = self._estimate_label_box_width(f"pow={power}")
            final_width = max(
                label_width + 2 * margin,
                pow_label_width + 2 * margin,
                content_width + 2 * margin,
            )
        else:
            final_width = max(label_width, content_width)

        ctrl_indices = (
            control_qubit_indices if isinstance(op, ControlledUOperation) else []
        )
        return VInlineBlock(
            node_key=node_key,
            label=block_name,
            children=children,
            affected_qubits=affected_qubits,
            control_qubit_indices=ctrl_indices,
            power=power,
            depth=depth,
            border_padding=border_padding,
            max_gate_width=max_gate_width,
            label_width=label_width,
            content_width=content_width,
            final_width=final_width,
        )

    def _build_vfor(
        self,
        op: ForOperation,
        node_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
        scope_path: tuple,
    ) -> VFoldedBlock | VUnfoldedSequence | VSkip:
        """Build a Visual IR node for a ForOperation."""
        start_val, stop_val_raw, step_val = self._evaluate_loop_range(op, param_values)

        if self._is_zero_iteration_loop(start_val, stop_val_raw, step_val):
            return VSkip(node_key=node_key)

        affected_qubits = self._analyze_loop_affected_qubits(
            op, qubit_map, logical_id_remap, param_values
        )

        # Folded mode: fold_loops=True or symbolic stop
        if self.fold_loops or stop_val_raw is None:
            header, body_lines, folded_width = self._compute_folded_for_info(
                op, param_values
            )
            return VFoldedBlock(
                node_key=node_key,
                header_label=header,
                body_lines=body_lines,
                affected_qubits=affected_qubits,
                folded_width=folded_width,
                kind=VFoldedKind.FOR,
            )

        # Unfolded: measure each iteration
        num_iterations = self._compute_loop_iterations(
            start_val, stop_val_raw, step_val
        )
        iterations: list[list[VisualNode]] = []
        iteration_widths: list[float] = []

        for iteration in range(num_iterations):
            iter_value = start_val + iteration * step_val
            child_param_values = dict(param_values)
            child_param_values[f"_loop_{op.loop_var}"] = iter_value

            self._evaluate_loop_body_intermediates(op.operations, child_param_values)

            children = self._build_visual_nodes(
                op.operations,
                qubit_map,
                logical_id_remap,
                child_param_values,
                depth + 1,
                (*node_key, iteration),
            )
            iter_width = self._sum_visual_widths(children)
            iterations.append(children)
            iteration_widths.append(iter_width)

        return VUnfoldedSequence(
            node_key=node_key,
            iterations=iterations,
            affected_qubits=affected_qubits,
            kind=VUnfoldedKind.FOR,
            iteration_widths=iteration_widths,
        )

    def _build_vwhile(
        self,
        op: WhileOperation,
        node_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> VFoldedBlock:
        """Build a VFoldedBlock for a WhileOperation (always folded).

        Note: This method is implemented for future use but currently not called,
        because ``_build_visual_nodes`` raises ``NotImplementedError`` for
        ``WhileOperation``.  It will be connected once While-loop visualization
        is fully enabled.
        """
        affected_qubits = self._analyze_loop_affected_qubits(
            op, qubit_map, logical_id_remap, param_values
        )
        header = "while cond:"
        local_param_values = dict(param_values)
        self._evaluate_loop_body_intermediates(op.operations, local_param_values)
        body_lines: list[str] = []
        for body_op in op.operations:
            expr = self._format_operation_as_expression(
                body_op,
                set(),
                body_operations=op.operations,
                param_values=local_param_values,
            )
            if expr:
                body_lines.extend(expr.split("\n"))
        # Truncate to 3 lines
        if len(body_lines) > 3:
            body_lines = body_lines[:3] + ["..."]

        folded_width = self._compute_folded_text_width(header, body_lines)

        return VFoldedBlock(
            node_key=node_key,
            header_label=header,
            body_lines=body_lines,
            affected_qubits=affected_qubits,
            folded_width=folded_width,
            kind=VFoldedKind.WHILE,
        )

    def _build_vif(
        self,
        op: IfOperation,
        node_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
        scope_path: tuple,
    ) -> VFoldedBlock | VUnfoldedSequence:
        """Build a Visual IR node for an IfOperation.

        Note: This method is implemented for future use but currently not called,
        because ``_build_visual_nodes`` raises ``NotImplementedError`` for
        ``IfOperation``.  It will be connected once If-operation visualization
        is fully enabled.
        """
        affected_qubits = self._collect_if_affected_qubits(
            op, qubit_map, logical_id_remap, param_values
        )

        cond = op.condition
        cond_name = getattr(cond, "name", None) or "cond"
        condition_label = f"if {cond_name}:"

        if self.fold_loops:
            local_param_values = dict(param_values)
            self._evaluate_loop_body_intermediates(
                op.true_operations, local_param_values
            )
            body_lines: list[str] = []
            for body_op in op.true_operations:
                expr = self._format_operation_as_expression(
                    body_op,
                    set(),
                    body_operations=op.true_operations,
                    param_values=local_param_values,
                )
                if expr:
                    body_lines.extend(expr.split("\n"))
            if len(body_lines) > 3:
                body_lines = body_lines[:3] + ["..."]

            folded_width = self._compute_folded_text_width(condition_label, body_lines)

            return VFoldedBlock(
                node_key=node_key,
                header_label=condition_label,
                body_lines=body_lines,
                affected_qubits=affected_qubits,
                folded_width=folded_width,
                kind=VFoldedKind.IF,
            )

        # Unfolded: build both branches
        self._evaluate_loop_body_intermediates(op.true_operations, param_values)
        true_children = self._build_visual_nodes(
            op.true_operations,
            qubit_map,
            logical_id_remap,
            param_values,
            depth + 1,
            (*node_key, "true"),
        )
        true_width = self._sum_visual_widths(true_children)

        self._evaluate_loop_body_intermediates(op.false_operations, param_values)
        false_children = self._build_visual_nodes(
            op.false_operations,
            qubit_map,
            logical_id_remap,
            param_values,
            depth + 1,
            (*node_key, "false"),
        )
        false_width = self._sum_visual_widths(false_children)

        iterations = [true_children]
        iteration_widths = [true_width]
        if false_children:
            iterations.append(false_children)
            iteration_widths.append(false_width)

        return VUnfoldedSequence(
            node_key=node_key,
            iterations=iterations,
            affected_qubits=affected_qubits,
            kind=VUnfoldedKind.IF,
            iteration_widths=iteration_widths,
            condition_label=condition_label,
        )

    def _build_vfor_items(
        self,
        op: ForItemsOperation,
        node_key: tuple,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        depth: int,
        scope_path: tuple,
    ) -> VFoldedBlock | VUnfoldedSequence | VSkip:
        """Build a Visual IR node for a ForItemsOperation."""
        affected_qubits = self._analyze_loop_affected_qubits(
            op, qubit_map, logical_id_remap, param_values
        )

        # Build label
        key_str = ", ".join(op.key_vars) if op.key_vars else "k"
        if len(op.key_vars) > 1:
            key_str = f"({key_str})"
        dict_value = op.operands[0] if op.operands else None
        dict_name = (
            getattr(dict_value, "name", "dict") if dict_value is not None else "dict"
        )
        header = f"for {key_str}, {op.value_var} in {dict_name}"

        # Materialize entries
        materialized = (
            self._materialize_dict_entries(dict_value)
            if dict_value is not None
            else None
        )

        if self.fold_loops or materialized is None:
            loop_vars = set(op.key_vars) | {op.value_var}
            body_lines: list[str] = []
            for body_op in op.operations:
                expr = self._format_operation_as_expression(
                    body_op,
                    loop_vars,
                    param_values=param_values,
                    body_operations=op.operations,
                )
                if expr:
                    body_lines.extend(expr.split("\n"))

            folded_width = self._compute_folded_text_width(header, body_lines)
            return VFoldedBlock(
                node_key=node_key,
                header_label=header,
                body_lines=body_lines,
                affected_qubits=affected_qubits,
                folded_width=folded_width,
                kind=VFoldedKind.FOR_ITEMS,
            )

        entries = materialized
        if not entries:
            return VSkip(node_key=node_key)

        # Unfolded: iterate materialized entries
        iterations: list[list[VisualNode]] = []
        iteration_widths: list[float] = []

        for entry_key, entry_value in entries:
            child_param_values = dict(param_values)
            # Set key variables
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

            self._evaluate_loop_body_intermediates(op.operations, child_param_values)

            children = self._build_visual_nodes(
                op.operations,
                qubit_map,
                logical_id_remap,
                child_param_values,
                depth + 1,
                (*node_key, len(iterations)),
            )
            iter_width = self._sum_visual_widths(children)
            iterations.append(children)
            iteration_widths.append(iter_width)

        return VUnfoldedSequence(
            node_key=node_key,
            iterations=iterations,
            affected_qubits=affected_qubits,
            kind=VUnfoldedKind.FOR_ITEMS,
            iteration_widths=iteration_widths,
        )

    def _compute_folded_for_info(
        self, op: ForOperation, param_values: dict
    ) -> tuple[str, list[str], float]:
        """Compute header, body lines, and folded width for a ForOperation."""
        range_str = self._format_range_str(op, set(), param_values)
        header = f"for {op.loop_var} in {range_str}"

        # Evaluate intermediate BinOp results so that nested expressions
        # (e.g., a = omega + 1; b = a / 2) resolve to symbolic names
        # instead of exposing internal temporaries like "float_tmp".
        local_param_values = dict(param_values)
        self._evaluate_loop_body_intermediates(op.operations, local_param_values)

        body_lines: list[str] = []
        for body_op in op.operations:
            expr = self._format_operation_as_expression(
                body_op,
                {op.loop_var},
                param_values=local_param_values,
                body_operations=op.operations,
            )
            if expr:
                body_lines.extend(expr.split("\n"))

        folded_width = self._compute_folded_text_width(header, body_lines)
        return header, body_lines, folded_width

    def _compute_folded_text_width(self, header: str, body_lines: list[str]) -> float:
        """Compute the width needed for a folded block's text content."""
        scale = self.style.subfont_size / self.style.font_size
        header_width = len(header) * self.style.char_width_bold * scale
        max_body_chars = max((len(line) for line in body_lines), default=0)
        body_width = max_body_chars * self.style.char_width_monospace * scale
        text_width = max(header_width, body_width)
        margin = 0.15  # Extra margin for folded blocks
        return max(
            self.style.folded_loop_width,
            text_width + 2 * self.style.text_padding + margin,
        )

    def _sum_visual_widths(self, children: list[VisualNode]) -> float:
        """Sum widths of VisualNode children with inter-element gaps."""
        gap = self.style.gate_gap
        total = 0.0
        count = 0

        for child in children:
            if isinstance(child, VSkip):
                continue
            elif isinstance(child, VGate):
                total += child.estimated_width
            elif isinstance(child, VInlineBlock):
                total += child.final_width
            elif isinstance(child, VFoldedBlock):
                total += child.folded_width
            elif isinstance(child, VUnfoldedSequence):
                total += sum(child.iteration_widths)
                n_iters = len(child.iterations)
                if n_iters > 1:
                    total += gap * (n_iters - 1)
            count += 1

        if count > 1:
            total += gap * (count - 1)

        return total

    def _max_visual_gate_width(self, children: list[VisualNode]) -> float:
        """Find the maximum VGate.estimated_width in the VisualNode tree."""
        max_w = 0.0
        for child in children:
            if isinstance(child, VGate):
                max_w = max(max_w, child.estimated_width)
            elif isinstance(child, VInlineBlock):
                max_w = max(max_w, self._max_visual_gate_width(child.children))
            elif isinstance(child, VUnfoldedSequence):
                for iter_children in child.iterations:
                    max_w = max(max_w, self._max_visual_gate_width(iter_children))
        return max_w

    def _compute_visual_content_width(
        self,
        children: list[VisualNode],
        max_gate_width: float,
        depth: int,
    ) -> float:
        """Compute the total content width of VisualNode children inside a block."""
        border_padding = compute_border_padding(self.style, depth)
        border_extent = (
            max_gate_width / 2 + border_padding + self.style.gate_text_padding
        )
        total = self._sum_visual_widths(children)
        total += 2 * border_extent
        return total

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

    def _collect_if_affected_qubits(
        self,
        op: IfOperation,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str] | None = None,
        param_values: dict | None = None,
    ) -> list[int]:
        """Collect qubit indices affected by an IfOperation.

        Recursively walks both true and false branches to collect all
        qubit indices that are operands of any operation.

        Note: This method is implemented for future use but currently not called,
        because it is only invoked by ``_build_vif``, which itself is not yet
        connected to the dispatcher.

        Args:
            op: IfOperation to analyze.
            qubit_map: Mapping from logical_id to qubit wire index.
            logical_id_remap: Mapping from dummy logical_ids to actual logical_ids.
            param_values: Parameter values for resolving expressions.

        Returns:
            List of affected qubit indices.
        """
        if logical_id_remap is None:
            logical_id_remap = {}
        if param_values is None:
            param_values = {}

        affected: set[int] = set()

        def collect_qubits(ops: list[Operation]) -> None:
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
                elif isinstance(inner_op, MeasureVectorOperation):
                    if inner_op.operands:
                        indices = self._resolve_operand_to_qubit_indices(
                            inner_op.operands[0], qubit_map, logical_id_remap
                        )
                        affected.update(indices)
                    continue
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
                elif isinstance(inner_op, ForItemsOperation):
                    collect_qubits(inner_op.operations)
                    continue
                for operand in operands:
                    idx = self._resolve_operand_to_qubit_index(
                        operand, qubit_map, logical_id_remap, param_values
                    )
                    if idx is not None:
                        affected.add(idx)

        collect_qubits(op.true_operations)
        collect_qubits(op.false_operations)
        return list(affected)

    def _resolve_controlled_u_power(
        self,
        op: ControlledUOperation,
        param_values: dict,
    ) -> int:
        """Resolve ControlledUOperation.power to a concrete integer.

        Handles int literals, Value expressions (e.g., 2**k from QPE),
        and init_value fallback.

        Args:
            op: ControlledUOperation whose power to resolve.
            param_values: Parameter values for evaluating Value expressions.

        Returns:
            Resolved power as int (minimum 1).
        """
        if isinstance(op.power, int):
            return op.power
        if isinstance(op.power, Value):
            evaluated = self._evaluate_value(op.power, param_values)
            if isinstance(evaluated, (int, float)):
                return int(evaluated)
            init = getattr(op.power, "init_value", None)
            if isinstance(init, int):
                return init
        return 1

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
                param_value = param_values[name]
                if isinstance(param_value, (int, float)):
                    return param_value

        # 3. Check param_values by logical_id
        vid = value.logical_id
        if vid in param_values:
            param_value = param_values[vid]
            if isinstance(param_value, (int, float)):
                return param_value

        # 3.5. Loop variable name-based lookup
        if value.name:
            loop_key = f"_loop_{value.name}"
            if loop_key in param_values:
                param_value = param_values[loop_key]
                if isinstance(param_value, (int, float)):
                    return param_value

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

    def _resolve_symbolic_array_name(
        self, value: Value, param_values: dict
    ) -> str | None:
        """Try to resolve array element indices for symbolic display.

        For array elements like phis[i] where the array is symbolic but
        indices can be resolved (e.g., loop variable i=0), produces
        "phis[0]" instead of "phis[i]".

        Args:
            value: IR Value that may be an array element.
            param_values: Mapping from logical_ids/names to concrete values.

        Returns:
            Resolved name string (e.g., "phis[0]"), or None if not applicable.
        """
        if not (
            hasattr(value, "parent_array")
            and value.parent_array is not None
            and hasattr(value, "element_indices")
            and value.element_indices
        ):
            return None

        parent_lid = value.parent_array.logical_id
        if parent_lid in param_values and isinstance(param_values[parent_lid], str):
            parent_name = param_values[parent_lid]
        else:
            parent_name = value.parent_array.name or "params"
        resolved_indices = []
        for idx_val in value.element_indices:
            resolved = self._evaluate_value(idx_val, param_values)
            if resolved is not None:
                resolved_indices.append(
                    str(int(resolved))
                    if isinstance(resolved, float) and resolved == int(resolved)
                    else str(resolved)
                )
            else:
                return None

        return f"{parent_name}[{','.join(resolved_indices)}]"

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
        const = value.get_const()
        if const is not None:
            if isinstance(const, float) and const == int(const):
                return str(int(const))
            return str(const)
        # Named parameter
        if value.is_parameter():
            name = value.parameter_name() or value.name
            if name:
                return name
        # Array element (e.g., weights[e])
        if hasattr(value, "parent_array") and value.parent_array is not None:
            array_name = value.parent_array.name or "arr"
            if hasattr(value, "element_indices") and value.element_indices:
                idx_parts = []
                for idx_val in value.element_indices:
                    idx = self._format_binop_operand(idx_val, param_values) or "?"
                    idx_parts.append(idx)
                return f"{array_name}[{','.join(idx_parts)}]"

        # Fallback: name
        if hasattr(value, "name") and value.name:
            return value.name
        return None

    def _resolve_binop_as_symbolic(
        self,
        value: Value,
        param_values: dict,
        operations: list[Operation] | None = None,
    ) -> str | None:
        """Find the defining BinOp for a value and build a symbolic expression string."""
        # Search provided operations first (e.g., loop body)
        if operations is not None:
            for op in operations:
                if (
                    isinstance(op, BinOp)
                    and op.results
                    and id(op.results[0]) == id(value)
                ):
                    return self._build_symbolic_binop(op, param_values)
        # Fallback: search top-level graph
        graph = getattr(self, "graph", None)
        if graph is None:
            return None
        for op in graph.operations:
            if isinstance(op, BinOp) and op.results and id(op.results[0]) == id(value):
                return self._build_symbolic_binop(op, param_values)
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
        # For symbolic array elements, skip cached UUID lookup —
        # the cached qubit_map entry may be stale from a different loop
        # iteration (e.g., map_block_results registers result.lid which
        # shares operand.lid via SSA).
        is_symbolic_array_element = (
            hasattr(value, "parent_array")
            and value.parent_array is not None
            and hasattr(value, "element_indices")
            and value.element_indices
            and not value.element_indices[0].is_constant()
        )
        if lid in qubit_map and not is_symbolic_array_element:
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
            array_size = None
            if size_value.is_constant():
                c = size_value.get_const()
                if c is not None and isinstance(c, int):
                    array_size = c
            elif param_values:
                ev = self._evaluate_value(size_value, param_values)
                if ev is not None and isinstance(ev, (int, float)):
                    array_size = int(ev)
            if array_size is not None:
                return [base_idx + ai for ai in range(array_size)]
            # Last resort: scan qubit_map for element keys
            element_indices = []
            i = 0
            while f"{resolved_lid}_[{i}]" in qubit_map:
                element_indices.append(qubit_map[f"{resolved_lid}_[{i}]"])
                i += 1
            if element_indices:
                return element_indices

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
                pv = param_values[actual_input.logical_id]
                if isinstance(pv, (int, float)):
                    # Numeric value: store directly
                    child_param_values[dummy_input.logical_id] = pv
                elif (
                    hasattr(actual_input, "parent_array")
                    and actual_input.parent_array is not None
                    and not isinstance(actual_input.type, QubitType)
                ):
                    # Non-qubit array element with non-numeric value:
                    # try resolving the array index before storing the raw string
                    evaluated = self._evaluate_value(actual_input, param_values)
                    if evaluated is not None and isinstance(evaluated, (int, float)):
                        child_param_values[dummy_input.logical_id] = evaluated
                    else:
                        resolved = self._resolve_symbolic_array_name(
                            actual_input, param_values
                        )
                        child_param_values[dummy_input.logical_id] = (
                            resolved if resolved is not None else pv
                        )
                else:
                    child_param_values[dummy_input.logical_id] = pv
            elif (
                hasattr(actual_input, "parent_array")
                and actual_input.parent_array is not None
                and not isinstance(actual_input.type, QubitType)
            ):
                # Non-qubit array element not in param_values:
                # try evaluating or resolving symbolic name
                evaluated = self._evaluate_value(actual_input, param_values)
                if evaluated is not None and isinstance(evaluated, (int, float)):
                    child_param_values[dummy_input.logical_id] = evaluated
                else:
                    resolved = self._resolve_symbolic_array_name(
                        actual_input, param_values
                    )
                    if resolved is not None:
                        child_param_values[dummy_input.logical_id] = resolved
                    elif actual_input.is_parameter():
                        child_param_values[dummy_input.logical_id] = (
                            actual_input.parameter_name() or actual_input.name
                        )
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

    def _estimate_block_label_box_width(self, label: str) -> float:
        """Estimate box width for a CallBlock label.

        Uses ``char_width_block`` (wider than ``char_width_gate``) because
        block labels often contain longer text with parenthesized parameters
        (e.g., ``mixer(omegas[0])``).

        Args:
            label: Text label to estimate width for.

        Returns:
            Estimated box width in data coordinate units.
        """
        visual_label = label.replace("$", "")
        visual_label = re.sub(r"\\[a-zA-Z]+", "X", visual_label)
        text_width = len(visual_label) * self.style.char_width_block
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

    def _get_gate_params_for_expression(
        self,
        op: GateOperation,
        loop_vars: set[str] | None = None,
        body_operations: list | None = None,
        param_values: dict | None = None,
    ) -> str | None:
        """Get gate parameters as a string for expression format.

        Args:
            op: GateOperation.
            loop_vars: Set of loop variable names in scope (e.g., {"i", "j"}).
            body_operations: Operations list for resolving index expressions.
            param_values: Parameter values for resolving symbolic expressions.

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
                elif hasattr(theta, "parent_array") and theta.parent_array is not None:
                    array_name = theta.parent_array.name or "params"
                    idx_str = self._resolve_index_expression(
                        theta, loop_vars or set(), body_operations
                    )
                    return self._format_symbolic_param(f"{array_name}[{idx_str}]")
                elif theta.is_parameter():
                    param_name = theta.parameter_name() or "θ"
                    return self._format_symbolic_param(param_name)
                else:
                    # BinOp result (e.g., gamma * wij)
                    symbolic = self._resolve_binop_as_symbolic(
                        theta, param_values or {}, body_operations
                    )
                    if symbolic is not None:
                        return self._format_symbolic_expression(symbolic)
                    if hasattr(theta, "name") and theta.name:
                        return self._format_symbolic_param(theta.name)

        return None

    def _format_param_for_expression(
        self,
        operand: Value,
        loop_vars: set[str],
        param_values: dict | None = None,
        body_operations: list | None = None,
    ) -> str | None:
        """Format a non-qubit parameter operand for folded body expressions.

        Applies numeric evaluation and TeX symbolic formatting consistently
        for CallBlockOperation, ControlledUOperation, and CompositeGateOperation
        expression branches.

        Args:
            operand: Non-qubit Value to format.
            loop_vars: Loop variable names in scope.
            param_values: Parameter values for evaluation.
            body_operations: Operations list for BinOp resolution.

        Returns:
            Formatted string (possibly TeX), or None if unresolvable.
        """
        # Try numeric evaluation first
        if param_values is not None:
            evaluated = self._evaluate_value(operand, param_values, body_operations)
            if evaluated is not None and isinstance(evaluated, (int, float)):
                return self._format_parameter(evaluated)

        # Array element: resolve index and apply TeX formatting
        if hasattr(operand, "parent_array") and operand.parent_array is not None:
            arr = operand.parent_array.name or "params"
            idx = self._resolve_index_expression(operand, loop_vars, body_operations)
            return self._format_symbolic_param(f"{arr}[{idx}]")

        # Named parameter: apply TeX formatting
        if hasattr(operand, "name") and operand.name:
            return self._format_symbolic_param(operand.name)

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

            params = self._get_gate_params_for_expression(
                op, loop_vars, body_operations, param_values=param_values
            )
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

            qubit_parts: list[str] = []
            param_parts: list[str] = []
            for operand in op.operands[1:]:
                if isinstance(operand.type, QubitType):
                    s = None
                    if (
                        hasattr(operand, "parent_array")
                        and operand.parent_array is not None
                    ):
                        arr = operand.parent_array.name or "qubits"
                        idx = self._resolve_index_expression(
                            operand, loop_vars, body_operations
                        )
                        s = f"{arr}[{idx}]"
                    elif hasattr(operand, "name") and operand.name:
                        s = operand.name
                    if s is not None:
                        qubit_parts.append(s)
                else:
                    s = self._format_param_for_expression(
                        operand, loop_vars, param_values, body_operations
                    )
                    if s is not None:
                        param_parts.append(s)

            if not qubit_parts:
                return f"{prefix}{block_name}(...)"
            result_str = ",".join(qubit_parts)
            args_str = ",".join(qubit_parts + param_parts)
            return f"{prefix}{result_str} = {block_name}({args_str})"

        elif isinstance(op, ControlledUOperation):
            block_value = op.block
            block_name = (
                block_value.name if isinstance(block_value, BlockValue) else "U"
            ) or "U"

            def _qubit_str(v: Value) -> str | None:
                if hasattr(v, "parent_array") and v.parent_array is not None:
                    arr = v.parent_array.name or "qubits"
                    idx = self._resolve_index_expression(v, loop_vars, body_operations)
                    return f"{arr}[{idx}]"
                if hasattr(v, "name") and v.name:
                    return v.name
                return None

            all_operands = list(op.control_operands) + list(op.target_operands)
            qubit_parts = [
                s
                for v in all_operands
                if isinstance(v.type, QubitType) and (s := _qubit_str(v)) is not None
            ]
            param_parts = [
                s
                for v in all_operands
                if not isinstance(v.type, QubitType)
                and (
                    s := self._format_param_for_expression(
                        v, loop_vars, param_values, body_operations
                    )
                )
                is not None
            ]
            if not qubit_parts:
                return f"{prefix}{block_name}(...)"
            result_str = ",".join(qubit_parts)
            args_str = ",".join(qubit_parts + param_parts)
            return f"{prefix}{result_str} = {block_name}({args_str})"

        elif isinstance(op, CompositeGateOperation):
            block_name = op.name or "composite"
            qubit_parts: list[str] = []
            param_parts: list[str] = []
            for qval in list(op.control_qubits) + list(op.target_qubits):
                s = None
                if hasattr(qval, "parent_array") and qval.parent_array is not None:
                    arr = qval.parent_array.name or "qubits"
                    idx = self._resolve_index_expression(
                        qval, loop_vars, body_operations
                    )
                    s = f"{arr}[{idx}]"
                elif hasattr(qval, "name") and qval.name:
                    s = qval.name
                if s is not None:
                    qubit_parts.append(s)
            for pval in op.parameters:
                s = self._format_param_for_expression(
                    pval, loop_vars, param_values, body_operations
                )
                if s is not None:
                    param_parts.append(s)
            if not qubit_parts:
                return f"{prefix}{block_name}(...)"
            result_str = ",".join(qubit_parts)
            args_str = ",".join(qubit_parts + param_parts)
            return f"{prefix}{result_str} = {block_name}({args_str})"

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

        # Array element access (e.g., edges[_e, 0])
        if hasattr(value, "parent_array") and value.parent_array is not None:
            array_name = value.parent_array.name or "arr"
            if hasattr(value, "element_indices") and value.element_indices:
                idx_parts = []
                for idx in value.element_indices:
                    idx_parts.append(
                        self._format_value_as_expression(idx, loop_vars, operations)
                    )
                return f"{array_name}[{','.join(idx_parts)}]"
            return array_name

        # Named value fallback (e.g., edges_dim0 for unbound array shapes)
        if hasattr(value, "name") and value.name:
            return value.name

        return next(iter(loop_vars)) if loop_vars else "?"

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
        if isinstance(value, int):
            if value == 0:
                return "0"
        elif math.isclose(value, 0.0, abs_tol=1e-15):
            return "0"

        abs_val = abs(value)

        # Very large or very small: use scientific notation
        if abs_val >= 1000 or abs_val < 0.01:
            return f"{value:.0e}"  # e.g., "1e3", "-2e-4"

        # Normal range: truncate decimals
        if abs_val >= 10:
            return f"{value:.1f}"  # e.g., "12.3"
        return f"{value:.2f}"  # e.g., "0.79"

    @staticmethod
    def _extract_greek_prefix(name: str) -> tuple[str, str] | None:
        """Find the longest Greek letter prefix in *name*.

        Returns ``(tex_symbol, remainder)`` or ``None``.
        Only matches when there IS a remaining suffix (not exact match).
        """
        best: tuple[str, str] | None = None
        for symbol in _TEX_SYMBOLS:
            if name.startswith(symbol) and len(symbol) < len(name):
                if best is None or len(symbol) > len(best[0]):
                    best = (symbol, name[len(symbol) :])
        return best

    def _format_symbolic_param(self, name: str) -> str:
        """Format symbolic parameter name for display.

        Greek-letter names get TeX rendering ($\\theta$).
        Names with a Greek prefix get the prefix converted (phis → ${\\phi}s$).
        Non-Greek names with underscores/brackets get subscript notation.
        Other names are returned as plain text.

        Args:
            name: Parameter name.

        Returns:
            Formatted string suitable for matplotlib text rendering.
        """
        if "\\" in name:
            # Already a TeX command like \theta, \phi
            return f"${name}$"

        # Handle array subscript notation (e.g., "phi[0]", "phis[i]")
        bracket_pos = name.find("[")
        if bracket_pos > 0:
            base = name[:bracket_pos]
            rest = name[bracket_pos:]  # e.g., "[i]" or "[0]"
            if base in _TEX_SYMBOLS:
                return f"$\\{base}{rest}$"
            prefix = self._extract_greek_prefix(base)
            if prefix:
                symbol, suffix = prefix
                return f"${{\\{symbol}}}{suffix}{rest}$"
            return f"$\\mathrm{{{base}}}{rest}$"

        # Handle underscore notation (e.g., "theta_2", "x_2", "phis_0")
        if "_" in name:
            parts = name.split("_")
            # Build nested subscripts: beta_a_b_c → \beta_{a_{b_{c}}}
            subscript = parts[-1]
            for i in range(len(parts) - 2, 0, -1):
                subscript = f"{parts[i]}_{{{subscript}}}"

            base = parts[0]
            if base in _TEX_SYMBOLS:
                return f"$\\{base}_{{{subscript}}}$"
            prefix = self._extract_greek_prefix(base)
            if prefix:
                symbol, suffix = prefix
                return f"${{\\{symbol}}}{suffix}_{{{subscript}}}$"
            return f"$\\mathrm{{{base}}}_{{{subscript}}}$"

        # Simple name
        if name in _TEX_SYMBOLS:
            return f"$\\{name}$"

        # Simple name with Greek prefix (e.g., "phis" → ${\phi}s$)
        prefix = self._extract_greek_prefix(name)
        if prefix:
            symbol, suffix = prefix
            return f"${{\\{symbol}}}{suffix}$"

        return name  # plain text, e.g., "params"

    def _format_symbolic_expression(self, expr: str) -> str:
        """Format a symbolic expression (e.g., 'theta/2') as a TeX math string.

        Splits on arithmetic operators and converts each operand individually,
        applying Greek letter TeX conversion where appropriate.
        """
        parts = re.split(r"([+\-*/])", expr)
        tex_parts: list[str] = []
        for part in parts:
            stripped = part.strip()
            if stripped in "+-*/":
                tex_parts.append(stripped)
            elif stripped:
                if stripped in _TEX_SYMBOLS:
                    tex_parts.append(f"\\{stripped}")
                else:
                    tex_parts.append(stripped)
        return "$" + "".join(tex_parts) + "$"

    def _get_block_label(
        self,
        op: CallBlockOperation,
        qubit_map: dict[str, int],
        param_values: dict | None = None,
    ) -> str:
        """Get display label for a CallBlockOperation, including parameters.

        Non-qubit arguments are resolved in order:
        1. Constant value → numeric display
        2. ``_evaluate_value`` → numeric via param_values / BinOp
        3. ``_resolve_symbolic_array_name`` → e.g. ``omegas[0]``
        4. ``is_parameter()`` → symbolic name
        5. Raw ``arg_name`` fallback

        Args:
            op: CallBlockOperation to label.
            qubit_map: Mapping from logical_id to qubit wire index (used to
                distinguish qubit args from parameter args).
            param_values: Mapping from logical_ids to resolved parameter values,
                used to resolve array indices and bound values.

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
                else:
                    # Try numeric evaluation through param_values
                    evaluated = self._evaluate_value(actual_input, param_values or {})
                    if evaluated is not None and isinstance(evaluated, (int, float)):
                        params.append(self._format_parameter(evaluated))
                    else:
                        # Try resolved array name (e.g., omegas[0])
                        resolved_name = self._resolve_symbolic_array_name(
                            actual_input, param_values or {}
                        )
                        if resolved_name is not None:
                            params.append(self._format_symbolic_param(resolved_name))
                        elif actual_input.is_parameter():
                            param_name = actual_input.parameter_name() or arg_name
                            params.append(self._format_symbolic_param(param_name))
                        else:
                            params.append(self._format_symbolic_param(arg_name))

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
                            # Try resolving array indices (e.g., phis[i] → phis[0])
                            resolved_name = self._resolve_symbolic_array_name(
                                op.theta, param_values or {}
                            )
                            if resolved_name is not None:
                                param_str = self._format_symbolic_param(resolved_name)
                            else:
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
                        resolved_name = self._resolve_symbolic_array_name(
                            op.theta, param_values or {}
                        )
                        if resolved_name is not None:
                            param_str = self._format_symbolic_param(resolved_name)
                        else:
                            symbolic = self._resolve_binop_as_symbolic(
                                op.theta, param_values or {}
                            )
                            if symbolic is not None:
                                param_str = self._format_symbolic_expression(symbolic)
                            else:
                                param_str = self._format_symbolic_param(
                                    op.theta.name or "?"
                                )
            else:
                # Unknown type, convert to string
                param_str = str(op.theta)

            return f"{base_label}({param_str})", True

        return base_label, False
