"""Separate pass: Split a block into quantum and classical segments."""

from __future__ import annotations

import dataclasses

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.operation import OperationKind
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.operation.gate import (
    MeasureQFixedOperation,
    MeasureVectorOperation,
)
from qamomile.circuit.ir.operation.classical_ops import DecodeQFixedOperation
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.value import Value, ArrayValue
from qamomile.circuit.ir.types.primitives import BitType, QubitType, UIntType
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.passes.control_flow_visitor import ControlFlowVisitor
from qamomile.circuit.transpiler.segments import (
    ClassicalSegment,
    ExpvalSegment,
    HybridBoundary,
    QuantumSegment,
    Segment,
    SeparatedProgram,
)


class SeparatePass(Pass[Block, SeparatedProgram]):
    """Separate a block into quantum and classical segments.

    This pass:
    1. Materializes return operations (syncs output_values from ReturnOperation)
    2. Splits the operation list into alternating quantum and classical segments

    Input: Block (typically ANALYZED or LINEAR)
    Output: SeparatedProgram with segments and boundaries
    """

    @property
    def name(self) -> str:
        return "separate"

    def run(self, input: Block) -> SeparatedProgram:
        """Separate the block into segments."""
        # Phase 1: Materialize return operations
        block = self._materialize_return(input)

        # Phase 2: Lower high-level operations (MeasureQFixedOperation)
        block = self._lower_operations(block)

        # Phase 3: Separate into segments
        return self._separate_segments(block)

    # =========================================================================
    # Phase 1: Materialize return
    # =========================================================================

    def _materialize_return(self, block: Block) -> Block:
        """Synchronize output_values from ReturnOperation.

        Uses ReturnOperation.operands as the source of truth for return values,
        ensuring consistency between the operation stream and block metadata.
        """
        # Find ReturnOperation in operations (expected at the end)
        return_op = None
        for op in reversed(block.operations):
            if isinstance(op, ReturnOperation):
                return_op = op
                break

        if return_op is not None:
            # Use ReturnOperation.operands as source of truth
            return dataclasses.replace(
                block,
                output_values=list(return_op.operands),
            )

        # No ReturnOperation found, keep existing output_values
        return block

    # =========================================================================
    # Phase 2: Lower high-level operations
    # =========================================================================

    def _lower_operations(self, block: Block) -> Block:
        """Lower high-level operations like MeasureQFixedOperation.

        MeasureQFixedOperation is lowered to:
        1. Individual MeasureOperations for each qubit (HYBRID → QUANTUM segment)
        2. DecodeQFixedOperation to convert bits to float (CLASSICAL segment)
        """
        lowered_ops: list[Operation] = []

        for op in block.operations:
            if isinstance(op, MeasureQFixedOperation):
                lowered_ops.extend(self._lower_measure_qfixed(op))
            else:
                lowered_ops.append(op)

        return dataclasses.replace(block, operations=lowered_ops)

    def _lower_measure_qfixed(
        self,
        op: MeasureQFixedOperation,
    ) -> list[Operation]:
        """Lower MeasureQFixedOperation to MeasureVectorOperation + decode.

        Returns:
            List of operations: [MeasureVectorOperation, DecodeQFixedOperation]
        """
        qfixed = op.operands[0]

        # Prefer cast metadata for qubit UUIDs (tracks the actual physical qubits)
        # Fall back to qubit_values for backward compatibility
        qubit_uuids = qfixed.get_cast_qubit_uuids() or qfixed.params.get(
            "qubit_values", []
        )
        num_bits = op.num_bits or len(qubit_uuids)
        int_bits = op.int_bits

        # Create size value for array shape
        size_value = Value(
            type=UIntType(),
            name="qfixed_size",
            params={"const": num_bits},
        )

        # Create ArrayValue for qubits (store element UUIDs/logical_ids for emit pass)
        # Also store cast metadata if available for proper resource tracking
        array_params = {"element_uuids": qubit_uuids}
        if qfixed.get_cast_source_uuid():
            array_params["cast_source_uuid"] = qfixed.get_cast_source_uuid()
        if qfixed.get_cast_source_logical_id():
            array_params["cast_source_logical_id"] = qfixed.get_cast_source_logical_id()
        if qfixed.get_cast_qubit_logical_ids():
            array_params["element_logical_ids"] = qfixed.get_cast_qubit_logical_ids()

        qubits_array = ArrayValue(
            type=QubitType(),
            name="qfixed_qubits",
            shape=(size_value,),
            params=array_params,
        )

        # Create ArrayValue for bits (result)
        bits_array = ArrayValue(
            type=BitType(),
            name="qfixed_bits",
            shape=(size_value,),
        )

        # Create single MeasureVectorOperation
        measure_vec_op = MeasureVectorOperation(
            operands=[qubits_array],
            results=[bits_array],
        )

        # Create DecodeQFixedOperation with bits ArrayValue
        decode_op = DecodeQFixedOperation(
            num_bits=num_bits,
            int_bits=int_bits,
            operands=[bits_array],
            results=list(op.results),  # Original Float result
        )

        return [measure_vec_op, decode_op]

    # =========================================================================
    # Phase 3: Separate into segments
    # =========================================================================

    def _separate_segments(self, block: Block) -> SeparatedProgram:
        """Separate the block into quantum and classical segments."""
        segments: list[Segment] = []
        boundaries: list[HybridBoundary] = []

        current_ops: list[Operation] = []
        current_kind: OperationKind | None = None

        for op in block.operations:
            # Skip ReturnOperation - it's a terminal operation handled separately
            if isinstance(op, ReturnOperation):
                continue

            op_kind = self._effective_kind(op)

            # Handle ExpvalOp specially - it creates its own segment
            if isinstance(op, ExpvalOp):
                # Flush current quantum segment first
                if current_ops:
                    segment = self._create_segment(current_kind, current_ops)
                    segments.append(segment)
                    current_ops = []

                # Create ExpvalSegment
                expval_segment = ExpvalSegment(
                    operations=[op],
                    hamiltonian_value=op.hamiltonian,
                    qubits_value=op.qubits,
                    result_ref=op.output.uuid,
                )
                segments.append(expval_segment)

                # Reset state - next operations start fresh
                current_kind = None
                continue

            if op_kind == OperationKind.HYBRID:
                # HYBRID operations (measurements) need special handling:
                # - They take quantum input and produce classical output
                # - The measurement itself belongs to the quantum segment
                # - We accumulate consecutive measurements in the same segment

                # If we're in classical mode, flush and start quantum
                if current_kind == OperationKind.CLASSICAL and current_ops:
                    segment = self._create_segment(current_kind, current_ops)
                    segments.append(segment)
                    current_ops = []

                # If no current kind, start as quantum
                if current_kind is None:
                    current_kind = OperationKind.QUANTUM

                # Add the measurement to the current (quantum) segment
                # Note: we stay in QUANTUM mode to accumulate consecutive measurements
                if current_kind != OperationKind.QUANTUM:
                    current_kind = OperationKind.QUANTUM
                current_ops.append(op)

                # Create boundary for tracking quantum-classical transition
                boundary = HybridBoundary(
                    operation=op,
                    source_segment_index=len(segments),  # Current segment being built
                    target_segment_index=len(segments) + 1,  # Next segment after flush
                    value_ref=op.results[0].uuid if op.results else "",
                )
                boundaries.append(boundary)
                continue

            if current_kind is None:
                current_kind = op_kind

            if op_kind != current_kind and op_kind in (
                OperationKind.QUANTUM,
                OperationKind.CLASSICAL,
            ):
                # Context switch - flush current segment
                if current_ops:
                    segment = self._create_segment(current_kind, current_ops)
                    segments.append(segment)
                    current_ops = []
                current_kind = op_kind

            current_ops.append(op)

        # Flush final segment
        if current_ops:
            segment = self._create_segment(current_kind, current_ops)
            segments.append(segment)

        # Compute input/output refs for each segment
        self._compute_segment_io(segments, block)

        return SeparatedProgram(
            segments=segments,
            boundaries=boundaries,
            parameters=block.parameters,
            output_refs=[v.uuid for v in block.output_values],
        )

    def _effective_kind(self, op: Operation) -> OperationKind:
        """Determine the effective kind of an operation.

        For control flow, examine the inner operations.
        """
        kind = op.operation_kind

        if kind != OperationKind.CONTROL:
            return kind

        # Control flow - determine by inner operations
        if isinstance(op, ForOperation):
            inner_kinds = {self._effective_kind(inner) for inner in op.operations}
        elif isinstance(op, IfOperation):
            inner_kinds = {self._effective_kind(inner) for inner in op.true_operations}
            inner_kinds.update(
                self._effective_kind(inner) for inner in op.false_operations
            )
        elif isinstance(op, WhileOperation):
            inner_kinds = {self._effective_kind(inner) for inner in op.operations}
        else:
            # Other control - treat as classical for now
            return OperationKind.CLASSICAL

        # Remove CONTROL from inner kinds (nested control flow)
        inner_kinds.discard(OperationKind.CONTROL)

        if len(inner_kinds) == 0:
            return OperationKind.CLASSICAL  # Empty control flow
        elif len(inner_kinds) == 1:
            return inner_kinds.pop()
        else:
            # Mixed operations in control flow
            # If any QUANTUM or HYBRID operation exists, treat as QUANTUM
            # because measurements belong in the quantum circuit
            # EmitPass will handle backend-specific behavior
            if (
                OperationKind.QUANTUM in inner_kinds
                or OperationKind.HYBRID in inner_kinds
            ):
                return OperationKind.QUANTUM
            return OperationKind.CLASSICAL

    def _create_segment(
        self,
        kind: OperationKind | None,
        operations: list[Operation],
    ) -> Segment:
        """Create a segment of the appropriate type."""
        if kind == OperationKind.QUANTUM:
            return QuantumSegment(operations=operations)
        else:
            return ClassicalSegment(operations=operations)

    def _compute_segment_io(
        self,
        segments: list[Segment],
        block: Block,
    ) -> None:
        """Compute input/output refs for each segment.

        A segment reads values defined outside it (inputs)
        and writes values used outside it or returned (outputs).
        """
        # Track which segment defines each value
        value_definitions: dict[str, int] = {}  # uuid -> segment index

        # Initialize with block inputs
        for v in block.input_values:
            value_definitions[v.uuid] = -1  # -1 = external input

        # Initialize with parameters
        for v in block.parameters.values():
            value_definitions[v.uuid] = -1  # -1 = external input

        for i, segment in enumerate(segments):
            segment_inputs: set[str] = set()
            segment_outputs: set[str] = set()

            self._collect_segment_io(
                segment.operations,
                i,
                value_definitions,
                segment_inputs,
                segment_outputs,
            )

            segment.input_refs = list(segment_inputs)
            segment.output_refs = list(segment_outputs)

    def _collect_segment_io(
        self,
        operations: list[Operation],
        segment_index: int,
        value_definitions: dict[str, int],
        segment_inputs: set[str],
        segment_outputs: set[str],
    ) -> None:
        """Recursively collect input/output refs from operations."""

        class SegmentIOCollector(ControlFlowVisitor):
            def visit_operation(self, op: Operation) -> None:
                # Operands not defined in this segment are inputs
                for operand in op.operands:
                    if isinstance(operand, Value):
                        if (
                            operand.uuid not in value_definitions
                            or value_definitions[operand.uuid] != segment_index
                        ):
                            segment_inputs.add(operand.uuid)

                # Results are defined by this segment
                for result in op.results:
                    value_definitions[result.uuid] = segment_index
                    segment_outputs.add(result.uuid)

        collector = SegmentIOCollector()
        collector.visit_operations(operations)
