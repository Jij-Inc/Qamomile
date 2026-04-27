"""Inline pass: Inline CallBlockOperations to create an affine block."""

from __future__ import annotations

import dataclasses
from typing import cast

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
    CompositeGateType,
)
from qamomile.circuit.ir.operation.control_flow import (
    HasNestedOps,
    IfOperation,
)
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.value import ArrayValue, Value, ValueBase
from qamomile.circuit.transpiler.errors import InliningError
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


def _has_any_call_block(operations: list[Operation]) -> bool:
    """True if any op (or nested op) is a CallBlockOperation."""
    for op in operations:
        if isinstance(op, CallBlockOperation):
            return True
        if isinstance(op, IfOperation):
            if _has_any_call_block(op.true_operations) or _has_any_call_block(
                op.false_operations
            ):
                return True
        elif isinstance(op, HasNestedOps):
            for body in op.nested_op_lists():
                if _has_any_call_block(body):
                    return True
    return False


def count_call_blocks(operations: list[Operation]) -> int:
    """Count CallBlockOperations, including those nested inside IfOps and
    HasNestedOps.  Used by the unroll loop as the primary termination
    signal (count==0 means the block is fully inlined)."""
    count = 0
    for op in operations:
        if isinstance(op, CallBlockOperation) and op.block is not None:
            count += 1
        if isinstance(op, IfOperation):
            count += count_call_blocks(op.true_operations)
            count += count_call_blocks(op.false_operations)
        elif isinstance(op, HasNestedOps):
            for body in op.nested_op_lists():
                count += count_call_blocks(body)
    return count


class InlinePass(Pass[Block, Block]):
    """Inline all CallBlockOperations to create an affine block.

    This pass recursively inlines function calls while preserving
    control flow structures (For, If, While).

    Input: Block with BlockKind.HIERARCHICAL (may contain CallBlockOperations)
    Output: Block with BlockKind.AFFINE (no CallBlockOperations)
    """

    @property
    def name(self) -> str:
        return "inline"

    def run(self, input: Block) -> Block:
        """Inline all CallBlockOperations.

        Self-recursive CallBlockOperations (ones whose ``.block`` is the
        block currently being expanded) are unrolled **one level per
        call**: the inner self-call is substituted but left intact so
        that the outer fixed-point loop (inline ↔ partial_eval in
        ``Transpiler.transpile``) can fold the base-case ``if`` between
        iterations.  The output kind is ``AFFINE`` when no
        CallBlockOperations remain, otherwise ``HIERARCHICAL``.
        """
        if input.kind not in (BlockKind.HIERARCHICAL, BlockKind.TRACED):
            # Already inlined, return as-is
            return input

        # Build value substitution map for inlining
        value_map: dict[str, Value] = {}

        serialized_ops = self._serialize_operations(
            input.operations,
            value_map,
            visiting_blocks=set(),
        )

        # Map output values through the value_map
        output_values = [value_map.get(v.uuid, v) for v in input.output_values]

        out_kind = (
            BlockKind.HIERARCHICAL
            if _has_any_call_block(serialized_ops)
            else BlockKind.AFFINE
        )

        return Block(
            name=input.name,
            label_args=input.label_args,
            input_values=input.input_values,
            output_values=output_values,
            operations=serialized_ops,
            kind=out_kind,
            parameters=input.parameters,
        )

    def _serialize_operations(
        self,
        operations: list[Operation],
        value_map: dict[str, Value],
        visiting_blocks: set[int],
    ) -> list[Operation]:
        """Recursively serialize a list of operations."""
        result: list[Operation] = []

        for op in operations:
            if isinstance(op, ReturnOperation):
                # Skip ReturnOperation during inlining - the caller handles
                # return value mapping via block.output_values / call_op.results
                continue

            elif isinstance(op, CallBlockOperation):
                if op.block is not None and id(op.block) in visiting_blocks:
                    # Self-recursive cycle.  Keep the call as-is (with
                    # value substitutions) so the outer fixed-point loop
                    # can unroll one more layer after partial_eval folds
                    # the base-case condition.
                    substituted = self._substitute_values(op, value_map)
                    result.append(substituted)
                else:
                    inlined = self._inline_call(op, value_map, visiting_blocks)
                    result.extend(inlined)

            elif isinstance(op, IfOperation):
                # Recurse into both branches, preserve the If structure
                serialized_true = self._serialize_operations(
                    op.true_operations, value_map, visiting_blocks
                )
                serialized_false = self._serialize_operations(
                    op.false_operations, value_map, visiting_blocks
                )
                new_op = dataclasses.replace(
                    op,
                    true_operations=serialized_true,
                    false_operations=serialized_false,
                )
                # Apply value substitutions to operands and results
                new_op = self._substitute_values(new_op, value_map)
                result.append(new_op)

            elif isinstance(op, HasNestedOps):
                # Generic recursion for For/ForItems/While: recurse into
                # nested bodies, rebuild, then substitute values.
                new_lists = [
                    self._serialize_operations(body, value_map, visiting_blocks)
                    for body in op.nested_op_lists()
                ]
                new_op = op.rebuild_nested(new_lists)
                new_op = self._substitute_values(new_op, value_map)
                result.append(new_op)

            elif isinstance(op, CompositeGateOperation):
                # Handle CompositeGateOperation
                if self._should_inline_composite(op):
                    # Inline the implementation if available
                    if op.has_implementation and op.implementation is not None:
                        inlined = self._inline_composite(op, value_map, visiting_blocks)
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
        visiting_blocks: set[int],
    ) -> list[Operation]:
        """Inline a CallBlockOperation.

        Creates a fresh value_map for the called block's scope,
        mapping block inputs to call arguments.
        """
        if call_op.block is None:
            raise InliningError("CallBlockOperation.block must be set")
        block = call_op.block
        call_args = call_op.operands  # Arguments passed to the call

        # Map block's input values to call's argument values
        local_map = value_map.copy()
        # Use ValueSubstitutor to resolve call_args. Beyond a direct UUID
        # lookup, this also rewrites parent_array references on array-element
        # values so a callee that takes an individual ``q[i]`` from a Vector[Qubit]
        # parameter sees an element value whose ``parent_array`` points at the
        # caller's concrete array, not at a transient cloned ArrayValue. Without
        # this, the resource allocator at emit time fails to find the parent
        # array's QInitOperation entry in qubit_map.
        #
        # IMPORTANT: ``substitute_value`` returns ``ValueBase`` (covering
        # ``DictValue`` / ``TupleValue`` / ``ArrayValue``, not only the
        # narrower ``Value``). An earlier version narrowed to ``Value`` and
        # silently dropped non-Value substitutions, which broke kernel calls
        # passing a ``DictValue`` parameter (e.g. ``ising_cost(quad, ...)``
        # in QAOA): the callee's ``quad`` parameter ended up bound to the
        # cloned ``quad_param_callee`` instead of the caller's actual
        # ``quad_``, and ``emit_for_items`` could not find the dict in
        # bindings. Always trust the substituted result.
        arg_substitutor = ValueSubstitutor(local_map, transitive=True)
        for block_input, call_arg in zip(block.input_values, call_args):
            substituted_arg = arg_substitutor.substitute_value(call_arg)
            resolved_arg = cast(Value, substituted_arg)
            local_map[block_input.uuid] = resolved_arg

            # If both are ArrayValues, also map shape dimensions
            # This ensures symbolic dimensions (e.g., qubits_dim0) are resolved
            # to concrete values from the caller's array. Chase through
            # value_map so multi-level mappings (cloned_dim -> outer_dim ->
            # const) collapse to the terminal value.
            if isinstance(resolved_arg, ArrayValue) and isinstance(
                block_input, ArrayValue
            ):
                for block_dim, arg_dim in zip(block_input.shape, resolved_arg.shape):
                    resolved_dim = value_map.get(arg_dim.uuid, arg_dim)
                    # Multi-level chase (guard against cycles).
                    seen = {arg_dim.uuid}
                    while (
                        isinstance(resolved_dim, Value)
                        and resolved_dim.uuid in value_map
                        and resolved_dim.uuid not in seen
                    ):
                        seen.add(resolved_dim.uuid)
                        resolved_dim = value_map[resolved_dim.uuid]
                    local_map[block_dim.uuid] = resolved_dim
                    # Also propagate to outer ``value_map``. The callee
                    # parameter's shape dim UUID can leak into outer Values
                    # whenever frontend tracing inherits a Vector's shape from
                    # a call result — most commonly ``qmc.measure(data)`` where
                    # ``data`` is a helper return whose shape carries the
                    # helper parameter's symbolic dim. Without this propagation
                    # the outer op's result Vector shape stays as the inner
                    # ``data_dim0`` Value, the resource allocator's
                    # ``_resolve_size`` returns ``None``, and clbit allocation
                    # silently drops the measurement entirely.
                    value_map[block_dim.uuid] = resolved_dim

        # Clone operations with fresh UUIDs using UUIDRemapper
        remapper = UUIDRemapper()
        cloned_ops = remapper.clone_operations(block.operations)
        uuid_remap = remapper.uuid_remap

        # Build remapped_local_map with cloned UUIDs
        remapped_local_map: dict[str, Value] = {}
        for old_uuid, value in local_map.items():
            new_uuid = uuid_remap.get(old_uuid, old_uuid)
            remapped_local_map[new_uuid] = value

        # CRITICAL FIX: For any value that was cloned, ensure the cloned
        # version also maps back to the resolved argument. This handles the case
        # where operations reference a cloned parent_array.
        for old_uuid, new_uuid in uuid_remap.items():
            if old_uuid in local_map and new_uuid not in remapped_local_map:
                remapped_local_map[new_uuid] = local_map[old_uuid]

        # Recursively serialize the cloned operations, marking this block
        # as currently-being-expanded so self-recursive calls inside it
        # are left intact for the outer unroll loop.
        inner_visiting = visiting_blocks | {id(block)}
        inlined = self._serialize_operations(
            cloned_ops, remapped_local_map, inner_visiting
        )

        # Get return values from ReturnOperation (source of truth)
        return_op = find_return_operation(cloned_ops)
        return_values = list(return_op.operands) if return_op else block.output_values

        # Map block's return values to call's result values.
        # Always apply ValueSubstitutor so that newly-created ArrayValues
        # (e.g. from pauli_evolve) have their shape dimensions resolved
        # to the caller's concrete values.
        sub = ValueSubstitutor(
            {
                k: v for k, v in remapped_local_map.items()
            },  # copy as dict[str, ValueBase]
        )
        for block_return, call_result in zip(return_values, call_op.results):
            remapped_uuid = uuid_remap.get(block_return.uuid, block_return.uuid)
            if remapped_uuid in remapped_local_map:
                # The return value was mapped during inlining (modified input)
                resolved = remapped_local_map[remapped_uuid]
                value_map[call_result.uuid] = resolved
            else:
                # The return value is a newly created value (not a modified input).
                # Substitute to resolve shape dims, parent_array, etc.
                substituted = sub.substitute_value(block_return)
                if isinstance(substituted, Value):
                    value_map[call_result.uuid] = substituted
                    resolved = substituted
                else:
                    value_map[call_result.uuid] = call_result
                    resolved = call_result

            # Propagate the call_result's shape dim UUIDs to the outer
            # value_map. The frontend creates a fresh ``ArrayValue`` for
            # each ``CallBlockOperation.result`` with a fresh shape dim
            # Value (derived from the callee's return type annotation,
            # not the actual return Value's shape). Without this loop,
            # any outer op whose result inherits the call_result's shape
            # at frontend trace time (e.g., ``qmc.measure(call_result)``)
            # will keep referencing the unresolved fresh shape dim
            # forever, and downstream resource allocation drops the op.
            if (
                isinstance(call_result, ArrayValue)
                and isinstance(resolved, ArrayValue)
                and call_result.shape
                and resolved.shape
            ):
                for cr_dim, resolved_dim in zip(call_result.shape, resolved.shape):
                    if cr_dim.uuid != resolved_dim.uuid:
                        value_map[cr_dim.uuid] = resolved_dim

        return inlined

    def _substitute_values(
        self,
        op: Operation,
        value_map: dict[str, Value],
    ) -> Operation:
        """Substitute values in an operation using the value map."""
        substitutor = ValueSubstitutor(
            {k: v for k, v in value_map.items()},  # copy as dict[str, ValueBase]
        )
        return substitutor.substitute_operation(op)

    def _should_inline_composite(self, op: CompositeGateOperation) -> bool:
        """Determine if a composite gate should be inlined.

        Known composite gate types (QPE, QFT, IQFT) are NOT inlined because:
        - Their Block represents the unitary reference, not the full circuit
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
        visiting_blocks: set[int] | None = None,
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
        inlined = self._serialize_operations(
            cloned_ops, local_map, visiting_blocks or set()
        )

        # Get return values from ReturnOperation (source of truth)
        return_op = find_return_operation(cloned_ops)
        return_values: list[ValueBase] = list(return_op.operands) if return_op else []

        # Clone return values if not already cloned
        if not return_values:
            return_values = [remapper.clone_value(v) for v in impl.output_values]

        # Map block's return values to operation's result values
        for block_return, op_result in zip(return_values, op.results):
            if block_return.uuid in local_map:
                value_map[op_result.uuid] = local_map[block_return.uuid]
            else:
                value_map[op_result.uuid] = op_result

        return inlined
