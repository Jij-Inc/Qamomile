"""Inline pass: Inline CallBlockOperations to create a linear block."""

from __future__ import annotations

import copy
import dataclasses
import uuid
from typing import TYPE_CHECKING

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.operation.control_flow import ForOperation, IfOperation, WhileOperation
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.errors import InliningError

if TYPE_CHECKING:
    from qamomile.circuit.ir.block_value import BlockValue


class InlinePass(Pass[Block, Block]):
    """Inline all CallBlockOperations to create a linear block.

    This pass recursively inlines function calls while preserving
    control flow structures (For, If, While).

    Input: Block with BlockKind.HIERARCHICAL (may contain CallBlockOperations)
    Output: Block with BlockKind.LINEAR (no CallBlockOperations)
    """

    @property
    def name(self) -> str:
        return "inline"

    def run(self, input: Block) -> Block:
        """Inline all CallBlockOperations."""
        if input.kind != BlockKind.HIERARCHICAL:
            # Already inlined, return as-is
            return input

        # Build value substitution map for inlining
        value_map: dict[str, Value] = {}

        serialized_ops = self._serialize_operations(
            input.operations,
            value_map,
        )

        # Map output values through the value_map
        output_values = [
            value_map.get(v.uuid, v) for v in input.output_values
        ]

        return Block(
            name=input.name,
            label_args=input.label_args,
            input_values=input.input_values,
            output_values=output_values,
            operations=serialized_ops,
            kind=BlockKind.LINEAR,
            parameters=input.parameters,
        )

    def _serialize_operations(
        self,
        operations: list[Operation],
        value_map: dict[str, Value],
    ) -> list[Operation]:
        """Recursively serialize a list of operations."""
        result: list[Operation] = []

        for op in operations:
            if isinstance(op, CallBlockOperation):
                # Inline the called block
                inlined = self._inline_call(op, value_map)
                result.extend(inlined)

            elif isinstance(op, ForOperation):
                # Recurse into loop body, preserve the For structure
                serialized_body = self._serialize_operations(
                    op.operations, value_map
                )
                new_op = dataclasses.replace(op, operations=serialized_body)
                # Apply value substitutions to operands
                new_op = self._substitute_values(new_op, value_map)
                result.append(new_op)

            elif isinstance(op, IfOperation):
                # Recurse into both branches, preserve the If structure
                serialized_true = self._serialize_operations(
                    op.true_operations, value_map
                )
                serialized_false = self._serialize_operations(
                    op.false_operations, value_map
                )
                new_op = dataclasses.replace(
                    op,
                    true_operations=serialized_true,
                    false_operations=serialized_false,
                )
                # Apply value substitutions to operands and results
                new_op = self._substitute_values(new_op, value_map)
                result.append(new_op)

            elif isinstance(op, WhileOperation):
                # Recurse into loop body, preserve the While structure
                serialized_body = self._serialize_operations(
                    op.operations, value_map
                )
                new_op = dataclasses.replace(op, operations=serialized_body)
                new_op = self._substitute_values(new_op, value_map)
                result.append(new_op)

            else:
                # Regular operation - apply value substitutions
                substituted = self._substitute_values(op, value_map)
                result.append(substituted)

        return result

    def _inline_call(
        self,
        call_op: CallBlockOperation,
        value_map: dict[str, Value],
    ) -> list[Operation]:
        """Inline a CallBlockOperation.

        Creates a fresh value_map for the called block's scope,
        mapping block inputs to call arguments.
        """
        from qamomile.circuit.ir.block_value import BlockValue

        block: BlockValue = call_op.operands[0]  # type: ignore
        call_args = call_op.operands[1:]  # Arguments passed to the call

        # Map block's input values to call's argument values
        local_map = value_map.copy()
        for block_input, call_arg in zip(block.input_values, call_args):
            # Apply value_map to call_arg first
            resolved_arg = value_map.get(call_arg.uuid, call_arg)
            local_map[block_input.uuid] = resolved_arg

        # Create a UUID mapping for fresh copies of internal values
        uuid_remap: dict[str, str] = {}

        # Clone operations with fresh UUIDs
        cloned_ops = self._clone_operations_with_fresh_uuids(
            block.operations, uuid_remap
        )

        # Update local_map to use the remapped UUIDs for block inputs
        remapped_local_map: dict[str, Value] = {}
        for old_uuid, value in local_map.items():
            new_uuid = uuid_remap.get(old_uuid, old_uuid)
            remapped_local_map[new_uuid] = value

        # Recursively serialize the cloned operations
        inlined = self._serialize_operations(cloned_ops, remapped_local_map)

        # Map block's return values to call's result values
        for block_return, call_result in zip(block.return_values, call_op.results):
            remapped_uuid = uuid_remap.get(block_return.uuid, block_return.uuid)
            if remapped_uuid in remapped_local_map:
                # The return value was mapped during inlining
                value_map[call_result.uuid] = remapped_local_map[remapped_uuid]
            else:
                # Find the value in the inlined operations
                value_map[call_result.uuid] = call_result

        return inlined

    def _clone_operations_with_fresh_uuids(
        self,
        operations: list[Operation],
        uuid_remap: dict[str, str],
    ) -> list[Operation]:
        """Clone operations and create fresh UUIDs for all values."""
        result: list[Operation] = []

        for op in operations:
            cloned_op = self._clone_operation(op, uuid_remap)
            result.append(cloned_op)

        return result

    def _clone_operation(
        self,
        op: Operation,
        uuid_remap: dict[str, str],
    ) -> Operation:
        """Clone an operation with fresh UUIDs."""
        # Clone operands
        new_operands = []
        for v in op.operands:
            if isinstance(v, Value):
                new_operands.append(self._clone_value(v, uuid_remap))
            else:
                # BlockValue in CallBlockOperation - don't clone
                new_operands.append(v)

        # Clone results
        new_results = [self._clone_value(v, uuid_remap) for v in op.results]

        # Handle control flow operations
        if isinstance(op, ForOperation):
            cloned_body = self._clone_operations_with_fresh_uuids(
                op.operations, uuid_remap
            )
            return dataclasses.replace(
                op,
                operands=new_operands,
                results=new_results,
                operations=cloned_body,
            )
        elif isinstance(op, IfOperation):
            cloned_true = self._clone_operations_with_fresh_uuids(
                op.true_operations, uuid_remap
            )
            cloned_false = self._clone_operations_with_fresh_uuids(
                op.false_operations, uuid_remap
            )
            return dataclasses.replace(
                op,
                operands=new_operands,
                results=new_results,
                true_operations=cloned_true,
                false_operations=cloned_false,
            )
        elif isinstance(op, WhileOperation):
            cloned_body = self._clone_operations_with_fresh_uuids(
                op.operations, uuid_remap
            )
            return dataclasses.replace(
                op,
                operands=new_operands,
                results=new_results,
                operations=cloned_body,
            )
        else:
            return dataclasses.replace(
                op,
                operands=new_operands,
                results=new_results,
            )

    def _clone_value(
        self,
        value: Value,
        uuid_remap: dict[str, str],
    ) -> Value:
        """Clone a value with a fresh UUID."""
        old_uuid = value.uuid
        if old_uuid not in uuid_remap:
            uuid_remap[old_uuid] = str(uuid.uuid4())

        new_uuid = uuid_remap[old_uuid]
        return dataclasses.replace(value, uuid=new_uuid)

    def _substitute_values(
        self,
        op: Operation,
        value_map: dict[str, Value],
    ) -> Operation:
        """Substitute values in an operation using the value map."""
        new_operands = []
        for v in op.operands:
            if isinstance(v, Value):
                new_operands.append(value_map.get(v.uuid, v))
            else:
                new_operands.append(v)

        new_results = [
            value_map.get(v.uuid, v) for v in op.results
        ]

        return dataclasses.replace(
            op,
            operands=new_operands,
            results=new_results,
        )
