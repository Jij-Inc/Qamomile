"""Separate pass: Split a block into quantum and classical segments."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.operation import OperationKind
from qamomile.circuit.ir.operation.control_flow import ForOperation, IfOperation, WhileOperation
from qamomile.circuit.ir.operation.gate import MeasureOperation
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.circuit.transpiler.segments import (
    ClassicalSegment,
    HybridBoundary,
    QuantumSegment,
    Segment,
    SeparatedProgram,
)


class SeparatePass(Pass[Block, SeparatedProgram]):
    """Separate a block into quantum and classical segments.

    This pass splits the operation list into alternating segments
    of quantum and classical operations. Hybrid operations (measurements)
    become boundaries between segments.

    Control flow (For, If) is handled by examining the operation
    kind of operations inside - if all quantum, goes to quantum segment;
    if all classical, goes to classical segment; if mixed, treated as classical.

    Input: Block with BlockKind.ANALYZED
    Output: SeparatedProgram with segments and boundaries
    """

    @property
    def name(self) -> str:
        return "separate"

    def run(self, input: Block) -> SeparatedProgram:
        """Separate the block into segments."""
        if input.kind != BlockKind.ANALYZED:
            raise ValidationError(
                f"SeparatePass expects ANALYZED block, got {input.kind}"
            )

        segments: list[Segment] = []
        boundaries: list[HybridBoundary] = []

        current_ops: list[Operation] = []
        current_kind: OperationKind | None = None

        for op in input.operations:
            op_kind = self._effective_kind(op)

            if op_kind == OperationKind.HYBRID:
                # Flush current segment
                if current_ops:
                    segment = self._create_segment(current_kind, current_ops)
                    segments.append(segment)
                    current_ops = []

                # Create boundary for measurement
                boundary = HybridBoundary(
                    operation=op,
                    source_segment_index=len(segments) - 1 if segments else -1,
                    target_segment_index=len(segments),  # Next segment
                    value_ref=op.results[0].uuid if op.results else "",
                )
                boundaries.append(boundary)

                # Measurement produces classical output, so start new classical segment
                current_kind = OperationKind.CLASSICAL
                continue

            if current_kind is None:
                current_kind = op_kind

            if op_kind != current_kind and op_kind in (OperationKind.QUANTUM, OperationKind.CLASSICAL):
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
        self._compute_segment_io(segments, input)

        return SeparatedProgram(
            segments=segments,
            boundaries=boundaries,
            parameters=input.parameters,
            output_refs=[v.uuid for v in input.output_values],
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
            # Mixed quantum/classical in control flow
            # This is complex - for now, treat as classical (conservative)
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
        for op in operations:
            # Operands not defined in this segment are inputs
            for operand in op.operands:
                if isinstance(operand, Value):
                    if operand.uuid not in value_definitions or \
                       value_definitions[operand.uuid] != segment_index:
                        segment_inputs.add(operand.uuid)

            # Results are defined by this segment
            for result in op.results:
                value_definitions[result.uuid] = segment_index
                segment_outputs.add(result.uuid)

            # Recurse into control flow
            if isinstance(op, ForOperation):
                self._collect_segment_io(
                    op.operations,
                    segment_index,
                    value_definitions,
                    segment_inputs,
                    segment_outputs,
                )
            elif isinstance(op, IfOperation):
                self._collect_segment_io(
                    op.true_operations,
                    segment_index,
                    value_definitions,
                    segment_inputs,
                    segment_outputs,
                )
                self._collect_segment_io(
                    op.false_operations,
                    segment_index,
                    value_definitions,
                    segment_inputs,
                    segment_outputs,
                )
            elif isinstance(op, WhileOperation):
                self._collect_segment_io(
                    op.operations,
                    segment_index,
                    value_definitions,
                    segment_inputs,
                    segment_outputs,
                )
