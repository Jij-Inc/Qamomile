"""Circuit analysis: IR inspection, value resolution, and label generation.

This module provides CircuitAnalyzer, which handles all IR-level analysis
for the circuit visualization pipeline. It has no matplotlib dependency.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from qamomile.circuit.ir.block import Block
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
    MeasureQFixedOperation,
    MeasureVectorOperation,
)
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
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
    from qamomile.circuit.ir.block import Block


_INTERNAL_TMP_NAMES: frozenset[str] = frozenset({"uint_tmp", "float_tmp", "bit_tmp"})


class CircuitAnalyzer:
    """Analyzes IR blocks for circuit visualization.

    Handles qubit mapping, value resolution, label generation,
    and width estimation. Has no matplotlib dependency.
    """

    @staticmethod
    def _is_internal_temp_name(name: str | None) -> bool:
        """Return True if `name` is an IR-internal placeholder.

        These default names (``uint_tmp`` / ``float_tmp`` / ``bit_tmp``
        from `qamomile.circuit.frontend.handle.primitives`) label
        anonymous Values produced by BinOp and handle construction.
        They are implementation detail and must never reach user-facing
        rendered labels.

        Args:
            name: Candidate display string.

        Returns:
            True if `name` equals one of the reserved placeholders.
        """
        return name in _INTERNAL_TMP_NAMES

    def __init__(
        self,
        graph: "Block",
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
        self, graph: "Block"
    ) -> tuple[dict[str, int], dict[int, str], int]:
        """Build mapping from qubit logical_id to wire indices.

        In SSA form, each operation creates new Values via next_version(),
        which preserves logical_id. This means all versions of a qubit share
        the same logical_id, so we only need logical_id-based tracking.

        Args:
            graph: Computation block.

        Returns:
            Tuple of (qubit_map, qubit_names, num_qubits).
        """
        qubit_map: dict[str, int] = {}
        qubit_names: dict[int, str] = {}
        next_idx = 0
        # Per-element wire indices for callee parameters whose actual
        # argument was a slice view of a root register.  A slice view
        # owns a non-contiguous subset of the root's wires, so the
        # callee's parameter cannot be registered as a single fresh
        # wire — see ``_compute_slice_view_wires`` for details.  Each
        # entry maps a callee parameter's ``logical_id`` to the list
        # of root-space wire indices it aliases, in element order.
        slice_view_wires: dict[str, list[int]] = {}

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
                        block_value = op.block
                        assert isinstance(block_value, Block)
                        new_remap = dict(logical_id_remap)
                        actual_inputs = op.operands

                        for dummy_input, actual_input in zip(
                            block_value.input_values, actual_inputs, strict=True
                        ):
                            if not isinstance(dummy_input.type, QubitType):
                                continue

                            # Slice-view actual argument special case:
                            # pre-populate the callee parameter's
                            # per-element entries so ``v[i]`` inside the
                            # callee resolves to the right root-space
                            # wires (non-contiguous in general).  See
                            # :meth:`_compute_slice_view_wires`.
                            slice_wires = self._compute_slice_view_wires(
                                actual_input,
                                qubit_map,
                                logical_id_remap,
                                param_values,
                                slice_view_wires,
                            )
                            if slice_wires is not None:
                                # Populate per-element entries under
                                # both the callee parameter's lid (for
                                # ``build_qubit_map``'s own recursion
                                # into the inlined body, which keeps
                                # the lid unremapped) and the actual
                                # argument's lid (for the visual IR
                                # build, which remaps the parameter to
                                # the argument's lid through
                                # ``_build_block_value_mappings``).
                                # Either lookup path then finds the
                                # right root-space wire without
                                # falling back to fresh-wire allocation
                                # in ``map_block_results``.
                                actual_input_lid = logical_id_remap.get(
                                    actual_input.logical_id,
                                    actual_input.logical_id,
                                )
                                for i, wire_idx in enumerate(slice_wires):
                                    qubit_map[f"{dummy_input.logical_id}_[{i}]"] = (
                                        wire_idx
                                    )
                                    qubit_map[f"{actual_input_lid}_[{i}]"] = wire_idx
                                if slice_wires:
                                    qubit_map[dummy_input.logical_id] = slice_wires[0]
                                    qubit_map[actual_input_lid] = slice_wires[0]
                                slice_view_wires[dummy_input.logical_id] = list(
                                    slice_wires
                                )
                                slice_view_wires[actual_input_lid] = list(slice_wires)
                                # The parameter aliases existing root
                                # wires — no fresh wire allocation, no
                                # logical-id remap (each ``v[i]`` lookup
                                # builds its own canonical key against
                                # ``dummy_input.logical_id``).
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
                        v for v in op.operands if isinstance(v.type, QubitType)
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
                            block_value = op.block
                            assert isinstance(block_value, Block)
                            fresh_pv = dict(param_values) if param_values else {}
                            for dummy, actual in zip(
                                block_value.input_values, op.operands
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
                    if isinstance(block_value, Block):
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
                    if isinstance(block_value, Block):
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
        graph: "Block",
        qubit_map: dict[str, int],
        qubit_names: dict[int, str],
        num_qubits: int,
    ) -> VisualCircuit:
        """Build a Visual IR tree from the IR block.

        Walks all operations, resolving labels, qubit indices, and widths
        into pre-computed VisualNode dataclasses. The resulting VisualCircuit
        can be consumed by Layout and Renderer without any Analyzer access.

        Args:
            graph: IR computation block.
            qubit_map: Mapping from logical_id to wire index.
            qubit_names: Mapping from wire index to display name.
            num_qubits: Total number of qubit wires.

        Returns:
            VisualCircuit containing the VisualNode tree.
        """
        # Pre-evaluate top-level intermediate BinOps so that CallBlock
        # arguments derived from them (e.g. final_base = reps * 2 * n)
        # resolve to numeric or symbolic strings instead of leaking the
        # IR-internal placeholder name "uint_tmp" downstream.
        top_param_values: dict = {}
        self._evaluate_loop_body_intermediates(graph.operations, top_param_values)
        children = self._build_visual_nodes(
            graph.operations, qubit_map, {}, top_param_values, depth=0, scope_path=()
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
            # MeasureOperation, MeasureVectorOperation, MeasureQFixedOperation
            if isinstance(
                op,
                (
                    GateOperation,
                    CallBlockOperation,
                    MeasureOperation,
                    MeasureVectorOperation,
                    MeasureQFixedOperation,
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
                indices = self._resolve_operand_to_qubit_indices(
                    operand, qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    qubit_indices.extend(indices)
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
                indices = self._resolve_operand_to_qubit_indices(
                    op.operands[0], qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    qubit_indices.extend(indices)
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
                indices = self._resolve_operand_to_qubit_indices(
                    op.operands[0], qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    qubit_indices = indices
            return VGate(
                node_key=node_key,
                label="M",
                qubit_indices=qubit_indices,
                estimated_width=gate_width,
                kind=VGateKind.MEASURE_VECTOR,
            )

        if isinstance(op, MeasureQFixedOperation):
            # ``MeasureQFixedOperation`` is the HYBRID measurement that
            # ``plan`` later splits into ``MeasureVectorOperation +
            # DecodeQFixedOperation``.  The visualizer works on the
            # pre-plan IR so it sees the unsplit form and must resolve
            # the carrier qubits itself.  The operand is the QFixed
            # ``Value`` produced by the upstream ``CastOperation``,
            # which attaches a ``CastMetadata`` whose
            # ``qubit_logical_ids`` field enumerates the source qubits
            # in carrier order.  Each entry is of the form
            # ``f"{root_logical_id}_{idx}"`` (without brackets);
            # ``QInitOperation`` registers root qubits in
            # ``qubit_map`` under ``f"{root_logical_id}_[{idx}]"``
            # (with brackets).  We bridge the two encodings here so
            # the carrier wires resolve precisely — including the
            # non-contiguous indices a slice-source cast produces
            # (e.g. ``cast(q[1::2], QFixed)`` covers root ``{1, 3}``).
            qubit_indices = []
            if op.operands:
                qubit_indices = self._resolve_qfixed_carrier_indices(
                    op.operands[0], qubit_map, logical_id_remap
                )
            return VGate(
                node_key=node_key,
                label="M",
                qubit_indices=qubit_indices,
                estimated_width=self.style.gate_width,
                kind=VGateKind.MEASURE_VECTOR,
            )

        if isinstance(op, CallBlockOperation):
            label = self._get_block_label(op, qubit_map, param_values=param_values)
            box_width = self._estimate_block_label_box_width(label)
            qubit_indices = []
            for operand in op.operands:
                indices = self._resolve_operand_to_qubit_indices(
                    operand, qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    qubit_indices.extend(indices)
            # Fresh-return pattern: if no qubit operands, resolve from results
            if not qubit_indices:
                for result_val in op.results:
                    if isinstance(result_val.type, QubitType):
                        indices = self._resolve_operand_to_qubit_indices(
                            result_val, qubit_map, logical_id_remap, param_values
                        )
                        if indices is not None:
                            qubit_indices.extend(indices)
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
            box_width = self._estimate_block_label_box_width(label)
            qubit_indices = []
            for qval in list(op.control_qubits) + list(op.target_qubits):
                indices = self._resolve_operand_to_qubit_indices(
                    qval, qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    qubit_indices.extend(indices)
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
            controlled_gate_type = self._controlled_u_single_gate_type(op, power_val)
            box_width = (
                self.style.gate_width
                if controlled_gate_type is not None
                else self._estimate_label_box_width(label)
            )
            # Control qubits first, then target qubits
            control_indices: list[int] = []
            for qval in op.control_operands:
                indices = self._resolve_operand_to_qubit_indices(
                    qval, qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    control_indices.extend(indices)
            target_indices: list[int] = []
            for qval in op.target_operands:
                indices = self._resolve_operand_to_qubit_indices(
                    qval, qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    target_indices.extend(indices)
            return VGate(
                node_key=node_key,
                label=label,
                qubit_indices=control_indices + target_indices,
                estimated_width=box_width,
                kind=VGateKind.CONTROLLED_U_BOX,
                gate_type=controlled_gate_type,
                box_width=box_width,
                control_count=len(control_indices),
            )

        raise TypeError(
            f"Unsupported operation type for _build_vgate: {type(op).__name__}"
        )

    def _controlled_u_single_gate_type(
        self, op: ControlledUOperation, power: int
    ) -> GateOperationType | None:
        """Return the wrapped gate type when controlled-U is a single gate.

        Args:
            op (ControlledUOperation): Controlled operation whose wrapped
                block may contain one concrete `GateOperation`.
            power (int): Resolved controlled-U power. Values greater than one
                are left in box form so the exponent remains visible.

        Returns:
            GateOperationType | None: Wrapped gate type when the controlled
                block is exactly one gate plus an optional return operation;
                otherwise None.
        """
        if power != 1:
            return None
        block = op.block
        if block is None:
            return None
        body_ops = [
            body_op
            for body_op in block.operations
            if not isinstance(body_op, ReturnOperation)
        ]
        if len(body_ops) != 1 or not isinstance(body_ops[0], GateOperation):
            return None
        gate_type = body_ops[0].gate_type
        if gate_type not in {
            GateOperationType.X,
            GateOperationType.Z,
            GateOperationType.CX,
            GateOperationType.CZ,
            GateOperationType.SWAP,
            GateOperationType.TOFFOLI,
        }:
            return None
        return gate_type

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
            indices = self._resolve_operand_to_qubit_indices(
                op.operands[0], qubit_map, logical_id_remap, param_values
            )
            if indices is not None:
                qubit_indices = indices
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
            block_value = op.block
            assert isinstance(block_value, Block)
            affected_qubits: list[int] = []
            for operand in op.operands:
                indices = self._resolve_operand_to_qubit_indices(
                    operand, qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    affected_qubits.extend(indices)
            actual_inputs = op.operands
            block_name = block_value.name or "block"
        elif isinstance(op, ControlledUOperation):
            block_value = op.block
            assert isinstance(block_value, Block)
            control_qubit_indices: list[int] = []
            for operand in op.control_operands:
                indices = self._resolve_operand_to_qubit_indices(
                    operand, qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    control_qubit_indices.extend(indices)
            affected_qubits = list(control_qubit_indices)
            for operand in op.target_operands:
                indices = self._resolve_operand_to_qubit_indices(
                    operand, qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    affected_qubits.extend(indices)
            actual_inputs = list(op.target_operands)
            u_name = getattr(block_value, "name", "U") or "U"
            block_name = u_name
        elif isinstance(op, CompositeGateOperation):
            block_value = op.implementation
            assert isinstance(block_value, Block)
            affected_qubits = []
            for operand in list(op.control_qubits) + list(op.target_qubits):
                indices = self._resolve_operand_to_qubit_indices(
                    operand, qubit_map, logical_id_remap, param_values
                )
                if indices is not None:
                    affected_qubits.extend(indices)
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
                if resolved is not None:
                    for idx in resolved:
                        if idx not in affected_qubits:
                            affected_qubits.append(idx)

        # Fresh-return fallback: resolve from results (same as box mode)
        if not affected_qubits and isinstance(op, CallBlockOperation):
            for result_val in op.results:
                if isinstance(result_val.type, QubitType):
                    indices = self._resolve_operand_to_qubit_indices(
                        result_val, qubit_map, logical_id_remap, param_values
                    )
                    if indices is not None:
                        affected_qubits.extend(indices)

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

        affected_qubits, affected_qubits_precise = self._analyze_loop_affected_qubits(
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
                affected_qubits_precise=affected_qubits_precise,
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
            affected_qubits_precise=affected_qubits_precise,
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
        affected_qubits, affected_qubits_precise = self._analyze_loop_affected_qubits(
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
            affected_qubits_precise=affected_qubits_precise,
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
        affected_qubits, affected_qubits_precise = self._collect_if_affected_qubits(
            op, qubit_map, logical_id_remap, param_values
        )

        cond = op.condition
        cond_name = getattr(cond, "name", None)
        if cond_name is None or self._is_internal_temp_name(cond_name):
            cond_name = "cond"
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
                affected_qubits_precise=affected_qubits_precise,
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
            affected_qubits_precise=affected_qubits_precise,
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
        affected_qubits, affected_qubits_precise = self._analyze_loop_affected_qubits(
            op, qubit_map, logical_id_remap, param_values
        )

        # Build label
        key_str = ", ".join(op.key_vars) if op.key_vars else "k"
        if len(op.key_vars) > 1:
            key_str = f"({key_str})"
        dict_value = op.operands[0] if op.operands else None
        if dict_value is None:
            dict_name = "dict"
        else:
            dict_name = getattr(dict_value, "name", None)
            if dict_name is None or self._is_internal_temp_name(dict_name):
                dict_name = "dict"
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
                affected_qubits_precise=affected_qubits_precise,
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
            affected_qubits_precise=affected_qubits_precise,
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

    @staticmethod
    def _qubit_bearing_operands(op: Operation) -> list[Value] | None:
        """Return the operands of `op` that may reference qubits.

        Used by the loop- and if-affect analyzers to pick out operands
        to feed into ``_resolve_operand_to_affected_qubits``. Returns
        None for control-flow operations, which the caller must handle
        via recursion rather than operand resolution.

        Args:
            op: IR operation to inspect.

        Returns:
            A list of operand Values to resolve, or None when the op is
            a control-flow construct (For/While/If/ForItems) that the
            caller handles separately.
        """
        if isinstance(op, (GateOperation, CallBlockOperation, ControlledUOperation)):
            return list(op.operands)
        if isinstance(op, CompositeGateOperation):
            return list(op.control_qubits) + list(op.target_qubits)
        if isinstance(op, (MeasureOperation, MeasureVectorOperation)):
            return list(op.operands[:1])
        return None

    def _analyze_loop_affected_qubits(
        self,
        op: ForOperation | WhileOperation | ForItemsOperation,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str] | None = None,
        param_values: dict | None = None,
    ) -> tuple[list[int], bool]:
        """Analyze which qubits are affected by a loop operation.

        Args:
            op: Loop operation (ForOperation, WhileOperation, or ForItemsOperation).
            qubit_map: Mapping from logical_id to qubit index.
            logical_id_remap: Mapping from dummy logical_ids to actual logical_ids.
            param_values: Parameter values for resolving loop range and indices.

        Returns:
            Tuple ``(indices, is_precise)`` where ``indices`` is the list
            of affected qubit wire indices and ``is_precise`` is True
            when every operand was resolved during the precise
            iteration walk (including any nested control-flow
            recursion). False when the conservative fallback was used,
            meaning the result may over-approximate.
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
                            if isinstance(
                                inner_op,
                                (ForOperation, WhileOperation, ForItemsOperation),
                            ):
                                nested, nested_precise = (
                                    self._analyze_loop_affected_qubits(
                                        inner_op,
                                        qubit_map,
                                        logical_id_remap,
                                        iter_params,
                                    )
                                )
                                precise_affected.update(nested)
                                if not nested_precise:
                                    all_resolved = False
                                continue
                            if isinstance(inner_op, IfOperation):
                                nested, nested_precise = (
                                    self._collect_if_affected_qubits(
                                        inner_op,
                                        qubit_map,
                                        logical_id_remap,
                                        iter_params,
                                    )
                                )
                                precise_affected.update(nested)
                                if not nested_precise:
                                    all_resolved = False
                                continue
                            operands = self._qubit_bearing_operands(inner_op)
                            if operands is None:
                                # Unknown op type — degrade to the recursive
                                # fallback to avoid silently dropping qubits.
                                all_resolved = False
                                continue
                            for operand in operands:
                                indices = self._resolve_operand_to_affected_qubits(
                                    operand,
                                    qubit_map,
                                    logical_id_remap,
                                    iter_params,
                                )
                                if indices is None:
                                    if isinstance(operand.type, QubitType):
                                        all_resolved = False
                                else:
                                    precise_affected.update(indices)
                    if all_resolved and precise_affected:
                        return list(precise_affected), True

        # Fallback to conservative analysis
        affected: set[int] = set()

        def add_affected(operand: Value) -> None:
            """Resolve `operand` and merge its wire indices into `affected`."""
            indices = self._resolve_operand_to_affected_qubits(
                operand, qubit_map, logical_id_remap
            )
            if indices is not None:
                affected.update(indices)

        def collect_from_ops(ops: list[Operation]) -> None:
            """Recursively collect qubit indices from all operations into `affected` set."""
            for inner_op in ops:
                if isinstance(
                    inner_op,
                    (ForOperation, WhileOperation, ForItemsOperation),
                ):
                    collect_from_ops(inner_op.operations)
                    continue
                if isinstance(inner_op, IfOperation):
                    collect_from_ops(inner_op.true_operations)
                    collect_from_ops(inner_op.false_operations)
                    continue
                operands = self._qubit_bearing_operands(inner_op)
                if operands is None:
                    continue
                for operand in operands:
                    add_affected(operand)

        collect_from_ops(op.operations)
        return list(affected), False

    def _collect_if_affected_qubits(
        self,
        op: IfOperation,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str] | None = None,
        param_values: dict | None = None,
    ) -> tuple[list[int], bool]:
        """Collect qubit indices affected by an IfOperation.

        Recursively walks both true and false branches to collect all
        qubit indices that are operands of any operation.

        Args:
            op: IfOperation to analyze.
            qubit_map: Mapping from logical_id to qubit wire index.
            logical_id_remap: Mapping from dummy logical_ids to actual logical_ids.
            param_values: Parameter values for resolving expressions.

        Returns:
            Tuple ``(indices, is_precise)``. ``is_precise`` is True
            when every operand — including those inside nested
            control flow — resolved cleanly.
        """
        if logical_id_remap is None:
            logical_id_remap = {}
        if param_values is None:
            param_values = {}

        affected: set[int] = set()
        is_precise = True

        def collect_qubits(ops: list[Operation]) -> None:
            nonlocal is_precise
            for inner_op in ops:
                if isinstance(
                    inner_op,
                    (ForOperation, WhileOperation, ForItemsOperation),
                ):
                    collect_qubits(inner_op.operations)
                    continue
                if isinstance(inner_op, IfOperation):
                    collect_qubits(inner_op.true_operations)
                    collect_qubits(inner_op.false_operations)
                    continue
                operands = self._qubit_bearing_operands(inner_op)
                if operands is None:
                    continue
                for operand in operands:
                    indices = self._resolve_operand_to_affected_qubits(
                        operand, qubit_map, logical_id_remap, param_values
                    )
                    if indices is None:
                        if isinstance(operand.type, QubitType):
                            is_precise = False
                    else:
                        affected.update(indices)

        collect_qubits(op.true_operations)
        collect_qubits(op.false_operations)
        return list(affected), is_precise

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
            const_array = parent.get_const_array()
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
                    elif op.kind == BinOpKind.MIN:
                        return lhs_val if lhs_val <= rhs_val else rhs_val
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

        Resolution order: ``param_values[logical_id]`` → ``_loop_<name>``
        fallback for ForOperation/ForItemsOperation loop variables →
        constant → array element access (``arr[i,j]``, recursing on each
        index so loop-bound indices substitute) → named parameter →
        display-name fallback (rejecting IR-internal placeholders).

        The array-element branch is checked *before* ``is_parameter()``
        because an element access on a parameter array reports
        ``is_parameter()`` True and ``parameter_name() == "gammas[layer]"``
        — i.e. the parameter name is the pre-formatted string with the
        unsubstituted loop variable baked in. Returning that verbatim
        would leak ``layer`` into the rendered expression even though
        ForOperation unrolling has already provided the concrete index
        via ``_loop_layer``. Trying the recursive array-element formatter
        first lets the index resolution see those concrete values
        (``gammas[0]``, ``gammas[1]``).

        Args:
            value (Value): IR Value used as a BinOp operand (lhs or rhs).
            param_values (dict): Mapping from ``logical_id`` (or ``_loop_<var>``
                / parameter name) to a resolved entry — either a numeric value
                or a previously-built symbolic string. Numeric values are
                formatted as integers when float-equal to an integer
                (``2.0`` → ``"2"``); otherwise as ``str(value)``.

        Returns:
            str | None: Human-readable string for the operand, or ``None``
                when the operand cannot be resolved by any of the rules
                above (caller treats ``None`` as "give up on this BinOp").
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
        # Loop variable: ForOperation/ForItemsOperation unrolling writes
        # `_loop_<var>` entries keyed by the variable *name* rather than the
        # operand's logical_id, so loop-bound operands like ``Jij`` are not
        # found by the logical_id lookup above. Mirror the same fallback
        # ``_evaluate_value`` already uses so concrete entry values (e.g.
        # ``0.5``) substitute into rendered angle expressions instead of
        # leaking the loop-variable name (``Jij*gamma``).
        if value.name:
            loop_key = f"_loop_{value.name}"
            if loop_key in param_values:
                pv = param_values[loop_key]
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
        # Array element (e.g., weights[e]). Checked before is_parameter()
        # so unrolled loop indices substitute (`gammas[layer]` →
        # `gammas[0]`) instead of leaking via the pre-formatted
        # parameter_name string `value.parameter_name()` returns for
        # element accesses on parameter arrays.
        if (
            hasattr(value, "parent_array")
            and value.parent_array is not None
            and hasattr(value, "element_indices")
            and value.element_indices
        ):
            array_name = value.parent_array.name or "arr"
            idx_parts = []
            for idx_val in value.element_indices:
                idx = self._format_binop_operand(idx_val, param_values) or "?"
                idx_parts.append(idx)
            return f"{array_name}[{','.join(idx_parts)}]"
        # Named parameter
        if value.is_parameter():
            name = value.parameter_name() or value.name
            if name and not self._is_internal_temp_name(name):
                return name

        # Fallback: name (refuse IR-internal placeholders so that
        # "uint_tmp"/"float_tmp"/"bit_tmp" never leak into rendered
        # expressions).
        if (
            hasattr(value, "name")
            and value.name
            and not self._is_internal_temp_name(value.name)
        ):
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
        elif binop.kind == BinOpKind.MIN:
            return f"min({lhs_str},{rhs_str})"
        else:
            _extra_ops: dict[BinOpKind, str] = {
                BinOpKind.FLOORDIV: "//",
                BinOpKind.POW: "**",
            }
            op_sym = _extra_ops.get(binop.kind, "?") if binop.kind is not None else "?"
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
            operand_list = op.operands
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

        For *view* elements (``value.parent_array.slice_of is not None``)
        the ``parent_array`` is the view's own ``ArrayValue`` and is **not**
        registered in ``qubit_map`` — only the root register is.  In that
        case the chain is walked via :meth:`_resolve_view_chain_to_root`
        so the view-local ``element_indices[0]`` is composed with the
        chain's affine map to a root-space index, and the canonical key
        is built against the root register.  Without this, inlined
        sub-kernels called with ``view[i]`` would allocate a fresh wire
        instead of mapping to the root register's wire (mirrors the same
        composition done by :meth:`_resolve_parent_array_element`).
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

        # Whole slice view (``ArrayValue`` with ``slice_of``, no
        # ``parent_array``): callers like ``map_block_results`` need a
        # canonical key that resolves to an existing root-space wire.
        # Without this branch the lid falls through and the caller
        # allocates a phantom wire for a value that actually aliases
        # root slots (visible as ghost wires past ``num_qubits`` in
        # rendered circuits).  Alias ``lid`` to the root's first
        # slice-covered element's wire and return the *array* lid
        # itself (NOT the element key) — callers (e.g. inline-block
        # remap consumers, ControlledU / CompositeGate paths) keep
        # using the returned string as a logical-id key for further
        # ``f"{lid}_[{i}]"`` element-key construction; returning an
        # element key here would yield malformed keys like
        # ``f"q.lid_[0]_[{i}]"``.
        if (
            isinstance(value, ArrayValue)
            and getattr(value, "slice_of", None) is not None
        ):
            resolved = self._resolve_view_chain_to_root(value)
            if resolved is not None:
                root_av, start, _step = resolved
                root_lid = logical_id_remap.get(root_av.logical_id, root_av.logical_id)
                root_elem_key = f"{root_lid}_[{start}]"
                if root_elem_key in qubit_map:
                    qubit_map[lid] = qubit_map[root_elem_key]
                    return lid
                if root_lid in qubit_map:
                    qubit_map[lid] = qubit_map[root_lid] + start
                    return lid

        if hasattr(value, "parent_array") and value.parent_array is not None:
            parent_array = value.parent_array
            # When the parent is a slice view, compose back to the root
            # register so the canonical key references the wire that
            # actually lives in ``qubit_map``.  The view itself was
            # never registered as a wire owner.
            chain_start = 0
            chain_step = 1
            chain_failed = False
            if getattr(parent_array, "slice_of", None) is not None:
                resolved = self._resolve_view_chain_to_root(parent_array)
                if resolved is None:
                    # Symbolic ``slice_start`` / ``slice_step`` along the
                    # chain — fall back to the original ``parent_lid``
                    # path; the caller's broader resolution may still
                    # find a match through a different alias.
                    chain_failed = True
                else:
                    root_av, chain_start, chain_step = resolved
                    parent_array = root_av

            if chain_failed:
                parent_lid = logical_id_remap.get(
                    value.parent_array.logical_id, value.parent_array.logical_id
                )
            else:
                parent_lid = logical_id_remap.get(
                    parent_array.logical_id, parent_array.logical_id
                )

            if hasattr(value, "element_indices") and value.element_indices:
                idx_value = value.element_indices[0]
                if idx_value.is_constant():
                    idx = idx_value.get_const()
                    if idx is not None:
                        root_idx = chain_start + chain_step * int(idx)
                        element_key = f"{parent_lid}_[{root_idx}]"
                        if element_key in qubit_map:
                            qubit_map[lid] = qubit_map[element_key]
                            return element_key
                elif param_values:
                    idx = self._evaluate_value(idx_value, param_values)
                    if idx is not None:
                        root_idx = chain_start + chain_step * int(idx)
                        element_key = f"{parent_lid}_[{root_idx}]"
                        if element_key in qubit_map:
                            # Don't cache — symbolic index may resolve
                            # differently across loop iterations
                            return element_key
        return lid

    def _resolve_array_size(
        self,
        array_value: Value,
        resolved_lid: str,
        qubit_map: dict[str, int],
        param_values: dict,
    ) -> int | None:
        """Resolve the length of an `ArrayValue`.

        Tries `shape[0].get_const()`, then `_evaluate_value` against
        `param_values`, and finally a scan of `qubit_map` keys
        matching `"{resolved_lid}_[{i}]"`.

        Args:
            array_value: The array whose length to resolve. Expected
                to expose a `shape` attribute.
            resolved_lid: Logical id used to index `qubit_map` for the
                element-key scan fallback.
            qubit_map: Mapping from logical_id to qubit wire index.
                Used only for the scan fallback.
            param_values: Parameter values for evaluating a symbolic
                shape.

        Returns:
            The array length as an int, or None if no strategy
            succeeds.
        """
        if hasattr(array_value, "shape") and array_value.shape:
            size_value = array_value.shape[0]
            if size_value.is_constant():
                c = size_value.get_const()
                if isinstance(c, int):
                    return c
            else:
                ev = self._evaluate_value(size_value, param_values)
                if isinstance(ev, (int, float)):
                    return int(ev)
        count = 0
        while f"{resolved_lid}_[{count}]" in qubit_map:
            count += 1
        if count > 0:
            return count
        return None

    def _resolve_qfixed_carrier_indices(
        self,
        operand: Value,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
    ) -> list[int]:
        """Resolve a QFixed measurement operand to its carrier wire indices.

        The operand is the QFixed ``Value`` produced by ``CastOperation``;
        its ``metadata.cast.qubit_logical_ids`` carries one entry per
        carrier qubit in measurement order, formatted as
        ``f"{root_logical_id}_{idx}"``.  ``QInitOperation`` registers
        each root element in ``qubit_map`` as ``f"{root_logical_id}_[{idx}]"``.
        This helper bridges the two encodings: it parses each cast
        carrier string, applies ``logical_id_remap`` to the root prefix
        (in case the cast lives in an inlined block), reformats with
        brackets, and looks up the result in ``qubit_map``.

        Args:
            operand (Value): The QFixed-typed operand of a
                ``MeasureQFixedOperation`` (i.e. the result of a prior
                ``CastOperation``).
            qubit_map (dict[str, int]): Mapping from logical_id-derived
                keys to qubit wire indices.
            logical_id_remap (dict[str, str]): Mapping from formal-
                parameter logical_ids to actual-argument logical_ids,
                used when the cast occurs inside an inlined block.

        Returns:
            list[int]: The wire indices the QFixed measurement targets,
                in carrier (most-significant-bit-first) order.  Empty
                when the operand carries no cast metadata or when no
                carrier entry resolves to a known wire.
        """
        cast_meta = getattr(operand.metadata, "cast", None)
        if cast_meta is None or not cast_meta.qubit_logical_ids:
            return []
        indices: list[int] = []
        for carrier_key in cast_meta.qubit_logical_ids:
            # Carrier key format: ``f"{root_logical_id}_{idx}"`` (no
            # brackets).  Split on the *last* ``_`` to recover the root
            # prefix and idx; the root logical_id itself may contain
            # underscores (UUIDs use them), so split from the right.
            try:
                root_lid, idx_str = carrier_key.rsplit("_", 1)
                idx_int = int(idx_str)
            except (ValueError, AttributeError):
                continue
            remapped_lid = logical_id_remap.get(root_lid, root_lid)
            bracket_key = f"{remapped_lid}_[{idx_int}]"
            wire = qubit_map.get(bracket_key)
            if wire is not None:
                indices.append(wire)
        return indices

    def _resolve_view_chain_to_root(
        self,
        array_value: "ArrayValue",
    ) -> tuple["ArrayValue", int, int] | None:
        """Walk the ``slice_of`` chain to the root array, composing the affine map.

        For a sliced ``ArrayValue`` (i.e. one with ``slice_of is not None``),
        the element-index in operand metadata is *view-local*: ``view[i]``
        on ``q[0::2]`` records ``element_indices[0] = i`` against
        ``parent_array = view_av``, not against ``q``.  Translating to a
        physical wire index requires walking the ``slice_of`` chain to
        the root and composing the start / step pairs of every hop.

        At each hop ``cur = slice_of(child)`` the parent-space index of
        ``child[i]`` is ``cur.slice_start + cur.slice_step * i``.
        Composing through a chain accumulates as
        ``(start, step) -> (cur.slice_start + cur.slice_step * start,
        cur.slice_step * step)`` until ``cur.slice_of is None`` (root).

        Args:
            array_value (ArrayValue): A possibly-sliced ArrayValue.  When
                ``slice_of is None`` the function returns the identity
                transform ``(array_value, 0, 1)``.

        Returns:
            tuple[ArrayValue, int, int] | None: ``(root_av, start, step)``
                so that the root-space index of a view-local ``idx`` is
                ``start + step * idx``.  ``None`` when any
                ``slice_start`` / ``slice_step`` along the chain is
                symbolic (cannot compose the affine map at draw time).
        """
        cur = array_value
        start = 0
        step = 1
        while getattr(cur, "slice_of", None) is not None:
            slice_start = cur.slice_start
            slice_step = cur.slice_step
            if slice_start is None or not slice_start.is_constant():
                return None
            if slice_step is None or not slice_step.is_constant():
                return None
            s = slice_start.get_const()
            st = slice_step.get_const()
            if s is None or st is None:
                return None
            try:
                s_int = int(s)
                st_int = int(st)
            except (TypeError, ValueError):
                return None
            start = s_int + st_int * start
            step = st_int * step
            cur = cur.slice_of
        return cur, start, step

    def _compute_slice_view_wires(
        self,
        actual_input: Value,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
        slice_view_wires: dict[str, list[int]],
    ) -> list[int] | None:
        """Per-element wires for a slice-view sub-kernel argument.

        When a ``@qkernel`` is called with a slice view (or with a
        parameter that was itself slice-aliased one call up), the
        callee's qubit parameter must alias the *same* root-register
        wires the caller's slice covers — not a fresh contiguous wire
        block.  This routine produces the wire indices to record under
        the callee parameter's ``logical_id`` so the renderer's qubit
        lookup ends up at the right physical wires.

        Two cases are handled:

        - ``actual_input`` is itself an ``ArrayValue`` with a
          ``slice_of`` chain (the caller wrote ``q[a:b:c]``): walk the
          chain to the root, then map each element of the view to the
          root's wire via ``start + step * i``.  If the root is itself
          slice-aliased (a previously-pre-populated parameter), the
          chain's start/step composes with that root's alias wires.
        - ``actual_input`` resolves (via ``logical_id_remap``) to a
          ``logical_id`` already registered in ``slice_view_wires``:
          the alias is forwarded one call deeper.

        Args:
            actual_input (Value): The IR value being passed into the
                ``CallBlockOperation``.
            qubit_map (dict[str, int]): Current wire mapping.  Used to
                look up the root's per-element entries.
            logical_id_remap (dict[str, str]): Logical-id remap from
                enclosing inlined blocks.
            param_values (dict): Parameter values, used for resolving a
                symbolic view shape.
            slice_view_wires (dict[str, list[int]]): Existing
                parameter-to-wires aliases from outer enclosing calls.

        Returns:
            list[int] | None: One wire index per element of the
                callee's parameter when ``actual_input`` resolves to a
                slice view, ``None`` otherwise (the caller falls back
                to the default fresh-wire registration path).
        """
        if not isinstance(actual_input, ArrayValue):
            return None

        actual_lid_resolved = logical_id_remap.get(
            actual_input.logical_id, actual_input.logical_id
        )

        # Case A: actual_input is itself a slice view (``slice_of`` set).
        if getattr(actual_input, "slice_of", None) is not None:
            resolved = self._resolve_view_chain_to_root(actual_input)
            if resolved is None:
                return None
            root_av, start, step = resolved
            root_lid = logical_id_remap.get(root_av.logical_id, root_av.logical_id)
            size = self._resolve_array_size(
                actual_input, actual_lid_resolved, qubit_map, param_values
            )
            if size is None:
                return None
            root_alias = slice_view_wires.get(root_lid)
            wires: list[int] = []
            for i in range(size):
                root_idx = start + step * i
                if root_alias is not None:
                    if 0 <= root_idx < len(root_alias):
                        wires.append(root_alias[root_idx])
                    else:
                        return None
                else:
                    root_elem_key = f"{root_lid}_[{root_idx}]"
                    if root_elem_key in qubit_map:
                        wires.append(qubit_map[root_elem_key])
                    elif root_lid in qubit_map:
                        wires.append(qubit_map[root_lid] + root_idx)
                    else:
                        return None
            return wires

        # Case B: actual_input is a previously slice-aliased parameter
        # being forwarded to a deeper call.
        if actual_lid_resolved in slice_view_wires:
            return list(slice_view_wires[actual_lid_resolved])

        return None

    def _resolve_parent_array_element(
        self,
        operand: Value,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> int | None:
        """Resolve a `q[i]` operand to its wire index when `i` is concrete.

        Handles both direct ``q[i]`` and ``view[i]`` where ``view`` is a
        slice of ``q``: when ``parent_array.slice_of`` is non-None, the
        chain is walked via :meth:`_resolve_view_chain_to_root` so that
        view-local ``i`` is composed with the chain's affine map to a
        root-space index.

        Args:
            operand: IR Value expected to have `parent_array` and
                `element_indices`. Operands without a `parent_array`
                return None.
            qubit_map: Mapping from logical_id to qubit wire index.
            logical_id_remap: Remap from dummy logical_ids to actual
                logical_ids.
            param_values: Parameter values for evaluating a symbolic
                `element_indices[0]`.

        Returns:
            The wire index for the resolved element.  When the parent
            is a slice view, the view-local index is composed with the
            ``slice_of`` chain back to the root register.  ``None`` when
            the parent / root is not in ``qubit_map``,
            ``element_indices`` is absent, or any index / slice
            metadata in the chain is symbolic and cannot be evaluated.
        """
        if not (hasattr(operand, "parent_array") and operand.parent_array is not None):
            return None
        parent_array = operand.parent_array
        if not (hasattr(operand, "element_indices") and operand.element_indices):
            return None
        idx_value = operand.element_indices[0]
        if idx_value.is_constant():
            idx = idx_value.get_const()
        else:
            idx = self._evaluate_value(idx_value, param_values)
        if idx is None:
            return None
        idx_int = int(idx)
        # Slice view → walk to root, compose affine map.  Mirror the
        # direct-element branch's ``element_key`` preference: when the
        # root has per-element entries (an ordinary register, or a
        # callee parameter slice-aliased to non-contiguous root wires
        # by ``build_qubit_map``'s sub-kernel handling), the formula
        # ``base + start + step * idx`` is only correct for contiguous
        # wires.  Look up the canonical ``f"{root_lid}_[{root_idx}]"``
        # key first and fall back to the formula only when no
        # per-element entry exists.
        if getattr(parent_array, "slice_of", None) is not None:
            resolved = self._resolve_view_chain_to_root(parent_array)
            if resolved is None:
                return None
            root_av, start, step = resolved
            root_lid = logical_id_remap.get(root_av.logical_id, root_av.logical_id)
            root_idx = start + step * idx_int
            root_elem_key = f"{root_lid}_[{root_idx}]"
            if root_elem_key in qubit_map:
                return qubit_map[root_elem_key]
            if root_lid not in qubit_map:
                return None
            return qubit_map[root_lid] + root_idx
        # Direct array element.  Prefer the canonical per-element key
        # (``f"{parent_lid}_[{idx}]"``) which is populated both by
        # ``QInitOperation`` for ordinary registers and by the slice-
        # view sub-kernel-argument handling in ``build_qubit_map``.
        # The formula fallback (``base + idx``) only fires for arrays
        # without per-element entries, where wires are contiguous by
        # construction.
        parent_lid = logical_id_remap.get(
            parent_array.logical_id, parent_array.logical_id
        )
        element_key = f"{parent_lid}_[{idx_int}]"
        if element_key in qubit_map:
            return qubit_map[element_key]
        if parent_lid not in qubit_map:
            return None
        return qubit_map[parent_lid] + idx_int

    def _resolve_non_element_operand(
        self,
        operand: Value,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str],
        param_values: dict,
    ) -> list[int] | None:
        """Resolve a non-element operand to qubit wire indices.

        Dispatches the three non-`parent_array` operand shapes: a
        whole `ArrayValue` with `shape`, a synthetic `ArrayValue`
        from `expval()` tuple input, and a direct `logical_id`
        lookup.

        Args:
            operand: IR Value that is not a `parent_array` element
                access.
            qubit_map: Mapping from logical_id to qubit wire index.
            logical_id_remap: Remap from dummy logical_ids to actual
                logical_ids.
            param_values: Parameter values for evaluating symbolic
                shapes.

        Returns:
            None if the operand is unresolvable (logical_id not in
            `qubit_map`, or shape symbolic and unevaluable). An
            empty list if the operand is resolved but touches zero
            qubits (e.g. concrete `shape[0] == 0`). Otherwise, a
            list of wire indices.
        """
        resolved_lid = logical_id_remap.get(operand.logical_id, operand.logical_id)

        # Slice view operand → walk the ``slice_of`` chain to the root
        # register and translate each element to a root-space wire via
        # ``start + step * i``.  Without this, the broadcast gate /
        # call-block / measure paths would treat a slice view as a
        # contiguous fresh-wire block (the formula ``base + k``), which
        # is correct only for an ordinary register.
        if (
            isinstance(operand, ArrayValue)
            and getattr(operand, "slice_of", None) is not None
        ):
            chain = self._resolve_view_chain_to_root(operand)
            if chain is not None:
                root_av, start, step = chain
                root_lid = logical_id_remap.get(root_av.logical_id, root_av.logical_id)
                size = self._resolve_array_size(
                    operand, resolved_lid, qubit_map, param_values
                )
                if size is not None:
                    # Loop-with-``else``: the ``else`` block runs only
                    # when the loop completes without a break.  A
                    # successful resolution (including an empty slice
                    # where ``size == 0``) returns the accumulated
                    # ``wires`` deterministically — matching the
                    # function contract that documents ``[]`` as the
                    # "zero qubits" outcome.  A missing-key break
                    # falls through to the other resolution branches.
                    wires: list[int] = []
                    for i in range(size):
                        root_idx = start + step * i
                        root_elem_key = f"{root_lid}_[{root_idx}]"
                        if root_elem_key in qubit_map:
                            wires.append(qubit_map[root_elem_key])
                        elif root_lid in qubit_map:
                            wires.append(qubit_map[root_lid] + root_idx)
                        else:
                            break
                    else:
                        return wires

        if (
            isinstance(operand, ArrayValue)
            and hasattr(operand, "shape")
            and operand.shape
            and resolved_lid in qubit_map
        ):
            size = self._resolve_array_size(
                operand, resolved_lid, qubit_map, param_values
            )
            if size is not None:
                # Prefer per-element keys (``f"{resolved_lid}_[{k}]"``)
                # when present.  ``QInitOperation`` populates them for
                # ordinary registers and ``build_qubit_map``'s slice-
                # view sub-kernel-argument handling populates them
                # with non-contiguous wires when a slice view is
                # passed as a helper qkernel argument.  The ``base +
                # k`` formula is only correct for genuinely contiguous
                # arrays where every wire is consecutive, so use the
                # element-key lookup first and fall back to the
                # formula only when no per-element entry exists.
                element_keyed_wires: list[int] = []
                for k in range(size):
                    elem_key = f"{resolved_lid}_[{k}]"
                    if elem_key in qubit_map:
                        element_keyed_wires.append(qubit_map[elem_key])
                    else:
                        element_keyed_wires = []
                        break
                if element_keyed_wires:
                    return element_keyed_wires
                base_idx = qubit_map[resolved_lid]
                return [base_idx + k for k in range(size)]
            return None

        if isinstance(operand, ArrayValue) and operand.get_element_uuids():
            return [
                qubit_map[uuid]
                for uuid in operand.get_element_uuids()
                if uuid in qubit_map
            ]

        if resolved_lid in qubit_map:
            return [qubit_map[resolved_lid]]

        return None

    def _resolve_operand_to_qubit_indices(
        self,
        operand: Value,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str] | None = None,
        param_values: dict | None = None,
    ) -> list[int] | None:
        """Resolve an operand to qubit wire indices for single-gate placement.

        Entry point for gate-building callers. When `operand` is
        `q[i]` with a symbolic `i` that cannot be evaluated, the
        result collapses to the parent array's base wire index —
        a single-qubit approximation that keeps the gate on one wire
        visually. Loop-affect analysis that needs to expand that
        case to the full array should call
        `_resolve_operand_to_affected_qubits` instead.

        Args:
            operand: IR Value representing a qubit or qubit-array
                operand.
            qubit_map: Mapping from logical_id to qubit wire index.
            logical_id_remap: Optional remap from dummy logical_ids
                (used during CallBlock inlining) to actual
                logical_ids. Defaults to None (empty remap).
            param_values: Optional parameter values for evaluating
                computed indices and shapes. Defaults to None.

        Returns:
            None if the operand is unresolvable. An empty list if
            the operand resolves to zero qubits. Otherwise, a list
            of wire indices.
        """
        if logical_id_remap is None:
            logical_id_remap = {}
        if param_values is None:
            param_values = {}

        idx = self._resolve_parent_array_element(
            operand, qubit_map, logical_id_remap, param_values
        )
        if idx is not None:
            return [idx]

        if hasattr(operand, "parent_array") and operand.parent_array is not None:
            parent_lid = logical_id_remap.get(
                operand.parent_array.logical_id, operand.parent_array.logical_id
            )
            if parent_lid in qubit_map:
                return [qubit_map[parent_lid]]

        return self._resolve_non_element_operand(
            operand, qubit_map, logical_id_remap, param_values
        )

    def _resolve_operand_to_affected_qubits(
        self,
        operand: Value,
        qubit_map: dict[str, int],
        logical_id_remap: dict[str, str] | None = None,
        param_values: dict | None = None,
    ) -> list[int] | None:
        """Resolve an operand to all wires it may touch across iterations.

        Entry point for loop-affect analysis. When `operand` is
        `q[i]` with a symbolic `i` that cannot be evaluated, the
        result expands to every element of the parent array — the
        correct semantics for "across all iterations this loop
        touches every element of `q`". If the parent array's size
        cannot be resolved either, returns None so callers can
        propagate `all_resolved = False`.

        Args:
            operand: IR Value representing a qubit or qubit-array
                operand.
            qubit_map: Mapping from logical_id to qubit wire index.
            logical_id_remap: Optional remap from dummy logical_ids
                to actual logical_ids. Defaults to None.
            param_values: Optional parameter values for evaluating
                computed indices and shapes. Defaults to None.

        Returns:
            None if the operand is unresolvable. An empty list if
            the operand resolves to zero qubits. Otherwise, a list
            of wire indices.
        """
        if logical_id_remap is None:
            logical_id_remap = {}
        if param_values is None:
            param_values = {}

        idx = self._resolve_parent_array_element(
            operand, qubit_map, logical_id_remap, param_values
        )
        if idx is not None:
            return [idx]

        if hasattr(operand, "parent_array") and operand.parent_array is not None:
            parent_lid = logical_id_remap.get(
                operand.parent_array.logical_id, operand.parent_array.logical_id
            )
            if parent_lid in qubit_map:
                base_idx = qubit_map[parent_lid]
                size = self._resolve_array_size(
                    operand.parent_array, parent_lid, qubit_map, param_values
                )
                if size is not None:
                    return [base_idx + k for k in range(size)]
                return None

        return self._resolve_non_element_operand(
            operand, qubit_map, logical_id_remap, param_values
        )

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
        """Build logical_id_remap and child_param_values for a nested Block.

        Maps dummy block input logical_ids to actual input logical_ids,
        building the mappings needed for recursive processing.

        Args:
            block_value: Block whose input mappings to build.
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
            else:
                # actual_input is a BinOp result (or similar unresolved
                # non-parameter Value). Try numeric evaluation via
                # graph.operations first — after top-level
                # pre-evaluation in build_visual_ir this usually
                # succeeds.
                evaluated = self._evaluate_value(actual_input, param_values)
                if evaluated is not None and isinstance(evaluated, (int, float)):
                    child_param_values[dummy_input.logical_id] = evaluated
                else:
                    # Fall back to a symbolic expression so downstream
                    # renders e.g. "reps*2*n" instead of "uint_tmp".
                    # `_format_value_as_expression` is called with empty
                    # `loop_vars` and `operations=None` here, so it may
                    # legitimately return its unresolved sentinel "?"
                    # for values that a richer downstream context could
                    # still resolve. Treat "?" the same as a temp-name
                    # miss and forward the logical_id so the downstream
                    # renderer gets a second chance with its own context.
                    expr = self._format_value_as_expression(actual_input, set(), None)
                    if expr and expr != "?" and not self._is_internal_temp_name(expr):
                        child_param_values[dummy_input.logical_id] = expr
                    else:
                        # Last resort: forward the logical_id so
                        # downstream lookups can still chase the
                        # defining BinOp via logical_id_remap.
                        new_logical_id_remap[dummy_input.logical_id] = (
                            actual_input.logical_id
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
                const_array = actual_input.get_const_array()
                if const_array is not None:
                    child_param_values[f"_array_data_{dummy_input.logical_id}"] = (
                        const_array
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
    ) -> Sequence[tuple] | None:
        """Extract entries from a DictValue from either IR or runtime metadata.

        A ``DictValue`` carries entries in one of two complementary
        encodings depending on how it was constructed:

        * **IR-level**: ``DictValue.entries`` is a tuple of
          ``(TupleValue | Value, Value)`` pairs — IR Values, not
          plain Python objects. Used when the dict was built inside a
          kernel (e.g. literal ``{i: x[i]}`` against symbolic
          parameters).
        * **Runtime metadata**: ``DictValue.metadata.dict_runtime.bound_data``
          holds a tuple of ``(raw_key, raw_value)`` Python pairs. Used
          when the dict was bound at ``transpile`` / ``draw`` time via
          ``bindings={"d": {0: 1.0, ...}}``.

        The runtime-metadata branch distinguishes "bound but empty" by
        checking ``metadata.dict_runtime is not None`` directly (so a
        dict bound to ``{}`` returns ``[]`` and renders as ``VSkip``).
        The IR-entries branch only fires when ``DictValue.entries`` is
        truthy: an empty IR ``entries`` tuple is treated as the
        "unbound parameter placeholder" case (every kernel-parameter
        ``Dict`` is constructed with ``entries=()``) and falls through
        to ``None``, which the caller renders as a folded box. There
        is currently no way for a kernel to declare a literal empty
        ``DictValue`` whose contents are knowable to be empty without
        also setting ``dict_runtime`` — if that ever becomes possible
        the IR-entries branch needs the same "is the entries field
        explicitly empty?" check the runtime branch already does.

        Callers (`_build_vfor_items`, `build_qubit_map`'s ForItems
        branch) handle the union shape by sniffing each pair —
        ``hasattr(entry_key, "elements")`` for ``TupleValue``,
        ``hasattr(entry_key, "get_const")`` for scalar IR Values,
        otherwise treating the pair as raw Python.

        Args:
            dict_value (Value): The DictValue (or compatible Value)
                whose entries should be materialized. Should carry
                IR-level ``entries`` or runtime ``dict_runtime``
                metadata; otherwise treated as truly unbound.

        Returns:
            Sequence[tuple] | None: A sequence of ``(key, value)``
                pairs whose element types depend on the encoding above
                (IR Values for the IR-level branch, raw Python objects
                for the runtime branch). Returns ``[]`` for a dict
                that is bound to an empty mapping so callers can
                render zero iterations as a ``VSkip`` rather than a
                folded box. Returns ``None`` only when the dict is
                truly unbound — no IR-level entries AND no runtime
                metadata.
        """
        # IR-level entries: a tuple of (TupleValue | Value, Value)
        # pairs. Returned as-is so the caller can resolve each operand
        # against `param_values` and pull symbolic / constant info as
        # appropriate.
        if hasattr(dict_value, "entries") and dict_value.entries:
            return dict_value.entries

        # Runtime metadata: a tuple of (raw_key, raw_value) Python
        # pairs. Distinguish "bound but empty" from "never bound" — a
        # truthy check on get_bound_data_items() would conflate the
        # two (both return the empty tuple), which made ForItems over
        # an empty bound Dict render as a folded box even when
        # fold_loops=False.
        if (
            hasattr(dict_value, "metadata")
            and dict_value.metadata.dict_runtime is not None
        ):
            return list(dict_value.get_bound_data_items())

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
                    if symbolic is not None and not self._is_internal_temp_name(
                        symbolic
                    ):
                        return self._format_symbolic_expression(symbolic)
                    if (
                        hasattr(theta, "name")
                        and theta.name
                        and not self._is_internal_temp_name(theta.name)
                    ):
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

        # Defining BinOp in scope — expand recursively before consulting
        # pre-computed symbolic strings. Keeps kernel parameters
        # symbolic (renders "r*2*n" rather than "r*2*2" because
        # _format_value_as_expression does not consult param_values for
        # numeric folding).
        if self._operand_is_binop_result(operand, body_operations):
            expr = self._format_value_as_expression(operand, loop_vars, body_operations)
            if expr and not self._is_internal_temp_name(expr):
                return self._format_symbolic_param(expr)

        # Symbolic string from pre-evaluation (e.g., _build_symbolic_binop
        # storing "reps*2*n" keyed by logical_id). Reached only when the
        # operand lacks a visible defining BinOp — typical for top-level
        # intermediates that were pre-evaluated during build_visual_ir.
        if param_values is not None and operand.logical_id in param_values:
            pv = param_values[operand.logical_id]
            if isinstance(pv, str) and not self._is_internal_temp_name(pv):
                return self._format_symbolic_param(pv)

        # Array element: resolve index and apply TeX formatting
        if hasattr(operand, "parent_array") and operand.parent_array is not None:
            arr = operand.parent_array.name or "params"
            idx = self._resolve_index_expression(operand, loop_vars, body_operations)
            return self._format_symbolic_param(f"{arr}[{idx}]")

        # Named parameter: apply TeX formatting. Skip IR-internal
        # placeholders to avoid leaking "uint_tmp" etc.
        if (
            hasattr(operand, "name")
            and operand.name
            and not self._is_internal_temp_name(operand.name)
        ):
            return self._format_symbolic_param(operand.name)

        return None

    def _operand_is_binop_result(
        self,
        operand: Value,
        operations: list | None,
    ) -> bool:
        """Return True if `operand` is the result Value of a BinOp in scope.

        Args:
            operand: IR Value to test.
            operations: Operations list to search. Falls back to
                ``self.graph.operations`` when None.

        Returns:
            True if a matching BinOp is found, else False.
        """
        ops = operations
        if ops is None:
            graph = getattr(self, "graph", None)
            ops = graph.operations if graph else []
        for op in ops:
            if (
                isinstance(op, BinOp)
                and op.results
                and id(op.results[0]) == id(operand)
            ):
                return True
        return False

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
            if op.gate_type is None:
                return None
            gate_name = op.gate_type.name.lower()

            if not op.operands:
                return None

            # Collect qubit operand strings. Rotation gates (rx/ry/rz
            # etc.) store their angle as the last operand via
            # GateOperation.rotation(); exclude it here by filtering on
            # QubitType so the result LHS and argument list show only
            # qubit operands.
            qubit_strs = []
            for operand in op.operands:
                if not isinstance(operand.type, QubitType):
                    continue
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
            block_value = op.block
            assert isinstance(block_value, Block)
            block_name = block_value.name or "block"

            qubit_parts: list[str] = []
            param_parts: list[str] = []
            for operand in op.operands:
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
                block_value.name if isinstance(block_value, Block) else "U"
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
                if op.kind == BinOpKind.MIN:
                    return f"min({lhs_str},{rhs_str})"
                _binop_symbols: dict[BinOpKind, str] = {
                    BinOpKind.ADD: "+",
                    BinOpKind.SUB: "-",
                    BinOpKind.MUL: "*",
                    BinOpKind.FLOORDIV: "//",
                    BinOpKind.POW: "**",
                }
                op_symbol = (
                    _binop_symbols.get(op.kind, "?") if op.kind is not None else "?"
                )
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

        # Named value fallback (e.g., edges_dim0 for unbound array
        # shapes). Refuse IR-internal placeholders to avoid leaking
        # "uint_tmp" etc. to rendered labels.
        if (
            hasattr(value, "name")
            and value.name
            and not self._is_internal_temp_name(value.name)
        ):
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
        block_value = op.block
        assert isinstance(block_value, Block)

        label = block_value.name or "block"

        # Collect non-qubit parameter values
        params = []
        for arg_name, actual_input in zip(block_value.label_args, op.operands):
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
                        pv_str = (
                            param_values.get(actual_input.logical_id)
                            if param_values
                            else None
                        )
                        if resolved_name is not None:
                            params.append(self._format_symbolic_param(resolved_name))
                        elif self._operand_is_binop_result(actual_input, None):
                            # BinOp result: expand recursively (e.g. reps*2*n)
                            # rather than falling through to the block's
                            # formal parameter name.
                            expr = self._format_value_as_expression(
                                actual_input, set(), None
                            )
                            if expr and not self._is_internal_temp_name(expr):
                                params.append(self._format_symbolic_param(expr))
                            else:
                                params.append(self._format_symbolic_param(arg_name))
                        elif isinstance(
                            pv_str, str
                        ) and not self._is_internal_temp_name(pv_str):
                            # Pre-computed symbolic string from top-level
                            # BinOp pre-evaluation in build_visual_ir.
                            params.append(self._format_symbolic_param(pv_str))
                        elif actual_input.is_parameter():
                            param_name = actual_input.parameter_name() or arg_name
                            if self._is_internal_temp_name(param_name):
                                param_name = arg_name
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
            GateOperationType.TDG: r"$T^{\dagger}$",
            GateOperationType.S: r"$S$",
            GateOperationType.SDG: r"$S^{\dagger}$",
            GateOperationType.CX: r"$CX$",
            GateOperationType.CZ: r"$CZ$",
            GateOperationType.SWAP: r"$SWAP$",
            GateOperationType.TOFFOLI: r"$CCX$",
        }

        base_label = (
            tex_labels.get(op.gate_type, str(op.gate_type))
            if op.gate_type is not None
            else "?"
        )

        # Add parameter display if the gate has a theta parameter
        if op.theta is not None:
            theta = op.theta
            if isinstance(theta, (int, float)):
                param_str = self._format_parameter(theta)
            elif isinstance(theta, Value):
                # Check if the Value has a bound constant
                const_val = theta.get_const()
                if const_val is not None:
                    # Use the bound constant value
                    param_str = self._format_parameter(const_val)
                elif theta.is_parameter():
                    # Check if resolved through param_values (inline block expansion)
                    if param_values and theta.logical_id in param_values:
                        resolved = param_values[theta.logical_id]
                        if isinstance(resolved, (int, float)):
                            param_str = self._format_parameter(resolved)
                        else:
                            param_str = self._format_symbolic_param(str(resolved))
                    else:
                        # Try evaluating (handles array element access like phis[i])
                        evaluated = self._evaluate_value(theta, param_values or {})
                        if evaluated is not None and isinstance(
                            evaluated, (int, float)
                        ):
                            param_str = self._format_parameter(evaluated)
                        else:
                            # Try resolving array indices (e.g., phis[i] → phis[0])
                            resolved_name = self._resolve_symbolic_array_name(
                                theta, param_values or {}
                            )
                            if resolved_name is not None:
                                param_str = self._format_symbolic_param(resolved_name)
                            else:
                                name = theta.parameter_name() or theta.name
                                if self._is_internal_temp_name(name):
                                    name = "?"
                                param_str = self._format_symbolic_param(name or "?")
                else:
                    # Generic Value — try evaluating (handles array elements, BinOps)
                    evaluated = self._evaluate_value(theta, param_values or {})
                    if evaluated is not None and isinstance(evaluated, (int, float)):
                        param_str = self._format_parameter(evaluated)
                    elif (
                        param_values
                        and theta.logical_id in param_values
                        and isinstance(param_values[theta.logical_id], str)
                        and not self._is_internal_temp_name(
                            param_values[theta.logical_id]
                        )
                    ):
                        param_str = self._format_symbolic_param(
                            param_values[theta.logical_id]
                        )
                    else:
                        resolved_name = self._resolve_symbolic_array_name(
                            theta, param_values or {}
                        )
                        if resolved_name is not None:
                            param_str = self._format_symbolic_param(resolved_name)
                        else:
                            symbolic = self._resolve_binop_as_symbolic(
                                theta, param_values or {}
                            )
                            if symbolic is not None and not self._is_internal_temp_name(
                                symbolic
                            ):
                                param_str = self._format_symbolic_expression(symbolic)
                            else:
                                fallback = theta.name
                                if fallback is None or self._is_internal_temp_name(
                                    fallback
                                ):
                                    fallback = "?"
                                param_str = self._format_symbolic_param(fallback)
            else:
                # Unknown type, convert to string (defensive fallback)
                param_str = str(theta)  # type: ignore[unreachable]

            return f"{base_label}({param_str})", True

        return base_label, False
