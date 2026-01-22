"""Inline pass: Inline CallBlockOperations to create a linear block."""

from __future__ import annotations

import dataclasses

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.block_value import BlockValue
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
    CompositeGateType,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.passes.value_mapping import (
    UUIDRemapper,
    ValueSubstitutor,
)


def find_return_operation(operations: list[Operation]) -> ReturnOperation | None:
    """Find the ReturnOperation in a list of operations (expected at the end)."""
    for op in reversed(operations):
        if isinstance(op, ReturnOperation):
            return op
    return None


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
        output_values = [value_map.get(v.uuid, v) for v in input.output_values]

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
            if isinstance(op, ReturnOperation):
                # Skip ReturnOperation during inlining - the caller handles
                # return value mapping via block.return_values / call_op.results
                continue

            elif isinstance(op, CallBlockOperation):
                # Inline the called block
                inlined = self._inline_call(op, value_map)
                result.extend(inlined)

            elif isinstance(op, ForOperation):
                # Recurse into loop body, preserve the For structure
                serialized_body = self._serialize_operations(op.operations, value_map)
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
                serialized_body = self._serialize_operations(op.operations, value_map)
                new_op = dataclasses.replace(op, operations=serialized_body)
                new_op = self._substitute_values(new_op, value_map)
                result.append(new_op)

            elif isinstance(op, CompositeGateOperation):
                # Handle CompositeGateOperation
                if self._should_inline_composite(op):
                    # Inline the implementation if available
                    if op.has_implementation and op.implementation is not None:
                        inlined = self._inline_composite(op, value_map)
                        result.extend(inlined)
                    else:
                        # Stub operation - keep as-is with value substitutions
                        substituted = self._substitute_values(op, value_map)
                        result.append(substituted)
                else:
                    # Keep as atomic operation (for native backend support)
                    substituted = self._substitute_values(op, value_map)
                    result.append(substituted)

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
        block: BlockValue = call_op.operands[0]  # type: ignore
        call_args = call_op.operands[1:]  # Arguments passed to the call

        # Map block's input values to call's argument values
        local_map = value_map.copy()
        for block_input, call_arg in zip(block.input_values, call_args):
            # Apply value_map to call_arg first
            resolved_arg = value_map.get(call_arg.uuid, call_arg)
            local_map[block_input.uuid] = resolved_arg

            # If both are ArrayValues, also map shape dimensions
            # This ensures symbolic dimensions (e.g., qubits_dim0) are resolved
            # to concrete values from the caller's array
            if isinstance(resolved_arg, ArrayValue) and isinstance(
                block_input, ArrayValue
            ):
                for block_dim, arg_dim in zip(block_input.shape, resolved_arg.shape):
                    local_map[block_dim.uuid] = arg_dim

        # Clone operations with fresh UUIDs using UUIDRemapper
        remapper = UUIDRemapper()
        cloned_ops = remapper.clone_operations(block.operations)
        uuid_remap = remapper.uuid_remap

        # Update local_map to use the remapped UUIDs for block inputs
        remapped_local_map: dict[str, Value] = {}
        for old_uuid, value in local_map.items():
            new_uuid = uuid_remap.get(old_uuid, old_uuid)
            remapped_local_map[new_uuid] = value

        # Recursively serialize the cloned operations
        inlined = self._serialize_operations(cloned_ops, remapped_local_map)

        # Get return values from ReturnOperation (source of truth)
        return_op = find_return_operation(cloned_ops)
        return_values = list(return_op.operands) if return_op else block.return_values

        # Map block's return values to call's result values
        for block_return, call_result in zip(return_values, call_op.results):
            remapped_uuid = uuid_remap.get(block_return.uuid, block_return.uuid)
            if remapped_uuid in remapped_local_map:
                # The return value was mapped during inlining
                value_map[call_result.uuid] = remapped_local_map[remapped_uuid]
            else:
                # Find the value in the inlined operations
                value_map[call_result.uuid] = call_result

        return inlined

    def _substitute_values(
        self,
        op: Operation,
        value_map: dict[str, Value],
    ) -> Operation:
        """Substitute values in an operation using the value map."""
        substitutor = ValueSubstitutor(value_map)
        return substitutor.substitute_operation(op)

    def _should_inline_composite(self, op: CompositeGateOperation) -> bool:
        """Determine if a composite gate should be inlined.

        Known composite gate types (QPE, QFT, IQFT) are NOT inlined because:
        - Their BlockValue represents the unitary reference, not the full circuit
        - They are emitted natively by the EmitPass

        Custom composite gates WITH implementations are inlined.
        """
        # Known composite gates should NOT be inlined
        # They are handled natively by the EmitPass
        if op.gate_type in (
            CompositeGateType.QPE,
            CompositeGateType.QFT,
            CompositeGateType.IQFT,
        ):
            return False

        # Custom gates with implementations should be inlined
        return op.has_implementation and op.implementation is not None

    def _inline_composite(
        self,
        op: CompositeGateOperation,
        value_map: dict[str, Value],
    ) -> list[Operation]:
        """Inline a CompositeGateOperation's implementation.

        Similar to _inline_call but handles CompositeGate operand structure.

        Note: With the new uuid/logical_id design, each Value has a unique uuid,
        so we can use the standard _serialize_operations method instead of
        the versioned variant.
        """
        impl = op.implementation
        if impl is None:
            return []

        # Get the qubit arguments from the operation
        # For CompositeGate: control qubits + target qubits
        qubit_args = op.control_qubits + op.target_qubits

        # Clone operations with fresh UUIDs using UUIDRemapper
        remapper = UUIDRemapper()
        cloned_ops = remapper.clone_operations(impl.operations)
        uuid_remap = remapper.uuid_remap

        # Map block's input values to operation's qubit arguments
        # Since uuid is now unique per Value, we can use simple uuid mapping
        local_map: dict[str, Value] = {}

        for block_input, qubit_arg in zip(impl.input_values, qubit_args):
            resolved_arg = value_map.get(qubit_arg.uuid, qubit_arg)

            # Get the cloned version of the input value
            cloned_input = remapper.clone_value(block_input)

            # Map the cloned input's uuid to the resolved argument
            local_map[cloned_input.uuid] = resolved_arg

        # Recursively serialize the cloned operations using standard method
        inlined = self._serialize_operations(cloned_ops, local_map)

        # Get return values from ReturnOperation (source of truth)
        return_op = find_return_operation(cloned_ops)
        return_values = list(return_op.operands) if return_op else []

        # Clone return values if not already cloned
        if not return_values:
            return_values = [remapper.clone_value(v) for v in impl.return_values]

        # Map block's return values to operation's result values
        for block_return, op_result in zip(return_values, op.results):
            if block_return.uuid in local_map:
                value_map[op_result.uuid] = local_map[block_return.uuid]
            else:
                value_map[op_result.uuid] = op_result

        return inlined
