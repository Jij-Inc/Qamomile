"""Inline pass: resolve inline callable invocations to create an affine block."""

from __future__ import annotations

import dataclasses
from typing import cast

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.callable import (
    CallPolicy,
    CallTransform,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.control_flow import HasNestedOps
from qamomile.circuit.ir.operation.gate import ControlledUOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.errors import QubitConsumedError
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


def _invoke_inline_body(op: InvokeOperation) -> Block | None:
    """Return the body of an inlineable InvokeOperation.

    Args:
        op (InvokeOperation): Invocation to inspect.

    Returns:
        Block | None: Body to inline, or ``None`` when the invocation should
            stay boxed.
    """
    if op.transform is not CallTransform.DIRECT:
        return None
    if op.default_policy is not CallPolicy.INLINE:
        return None
    body = op.effective_body()
    if isinstance(body, Block):
        return body
    return None


def _has_any_inline_call(operations: list[Operation]) -> bool:
    """Return whether any operation (or nested operation) is a call.

    Recurses into the nested blocks of ``InverseBlockOperation`` and
    ``ControlledUOperation`` and into ``HasNestedOps``
    bodies, so a call hidden inside a control-flow body or an
    operation-owned block is still detected. ``InlinePass`` uses this to
    decide whether its output block is ``AFFINE`` (no calls remain) or
    stays ``HIERARCHICAL``.

    Args:
        operations (list[Operation]): Operations to scan.

    Returns:
        bool: ``True`` if at least one inlineable call is reachable
            from *operations*, otherwise ``False``.
    """
    for op in operations:
        if isinstance(op, InvokeOperation) and _invoke_inline_body(op) is not None:
            return True
        if isinstance(op, InverseBlockOperation):
            for block in (op.source_block, op.implementation_block):
                if block is not None and _has_any_inline_call(block.operations):
                    return True
        if isinstance(op, ControlledUOperation):
            if op.block is not None and _has_any_inline_call(op.block.operations):
                return True
        if isinstance(op, HasNestedOps):
            for body in op.nested_op_lists():
                if _has_any_inline_call(body):
                    return True
    return False


def count_inline_invokes(operations: list[Operation]) -> int:
    """Count all inlineable calls reachable from an operation list.

    Recurses into ``InverseBlockOperation`` / ``ControlledUOperation``
    nested blocks and into ``HasNestedOps`` bodies, so calls hidden inside
    control flow or operation-owned blocks are
    counted. ``unroll_recursion`` uses this as the primary termination
    signal (``count == 0`` means the block is fully inlined).

    Args:
        operations (list[Operation]): Operations to scan.

    Returns:
        int: Total number of inlineable calls reachable from *operations*.
    """
    count = 0
    for op in operations:
        if isinstance(op, InvokeOperation) and _invoke_inline_body(op) is not None:
            count += 1
        if isinstance(op, InverseBlockOperation):
            for block in (op.source_block, op.implementation_block):
                if block is not None:
                    count += count_inline_invokes(block.operations)
        if isinstance(op, ControlledUOperation):
            if op.block is not None:
                count += count_inline_invokes(op.block.operations)
        if isinstance(op, HasNestedOps):
            for body in op.nested_op_lists():
                count += count_inline_invokes(body)
    return count


def count_unrollable_inline_invokes(operations: list[Operation]) -> int:
    """Count inlineable calls the inline/partial-eval loop can still resolve.

    This mirrors :func:`count_inline_invokes` but **does not** descend into
    a ``ControlledUOperation.block`` or an ``InverseBlockOperation``'s
    nested blocks. A call still inside one of those operation-owned blocks
    after a full ``inline`` pass is a self-recursive call that inline's
    cycle guard could not unroll — it stops after one layer and does not
    re-enter the operation-owned block — so no later ``unroll_recursion``
    iteration can resolve it. Folding compile-time ``if``s there (which
    ``CompileTimeIfLoweringPass`` does do for a ``ControlledUOperation``'s
    block) never removes the trapped call itself. Such a call is therefore
    *not* unrollable. Calls at the top level or inside ``For`` / ``If`` /
    ``While`` bodies are unrollable and are counted.

    The unroll loop uses this to tell two failure modes apart: a non-zero
    :func:`count_inline_invokes` with a zero ``count_unrollable_inline_invokes``
    means every residual call is trapped inside a controlled / inverted
    block (i.e. a recursive ``@qkernel`` was passed to ``qmc.control`` /
    ``qmc.inverse``), as opposed to a genuinely non-terminating top-level
    recursion.

    Args:
        operations (list[Operation]): Operations to scan.

    Returns:
        int: Number of inlineable calls reachable without entering a
            ``ControlledUOperation.block`` or ``InverseBlockOperation``
            nested block.
    """
    count = 0
    for op in operations:
        if isinstance(op, InvokeOperation) and _invoke_inline_body(op) is not None:
            count += 1
        if isinstance(op, HasNestedOps):
            for body in op.nested_op_lists():
                count += count_unrollable_inline_invokes(body)
    return count


class InlinePass(Pass[Block, Block]):
    """Inline qkernel callables to create an affine block.

    This pass recursively inlines function calls while preserving
    control flow structures (For, If, While).

    Input: Block with BlockKind.HIERARCHICAL (may contain callable calls)
    Output: Block with BlockKind.AFFINE (no inline callable calls)
    """

    @property
    def name(self) -> str:
        return "inline"

    def run(self, input: Block) -> Block:
        """Inline all inline-policy callable invocations.

        Self-recursive qkernel invocations are unrolled one level per call:
        the inner self-call is substituted but left intact so the outer
        fixed-point loop can fold the base-case ``if`` between iterations.
        The output kind is ``AFFINE`` when no inlineable calls remain,
        otherwise ``HIERARCHICAL``.
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
            if _has_any_inline_call(serialized_ops)
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
            param_slots=input.param_slots,
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

            elif (
                isinstance(op, InvokeOperation)
                and (body := _invoke_inline_body(op)) is not None
            ):
                if id(body) in visiting_blocks:
                    substituted = self._substitute_values(op, value_map)
                    result.append(substituted)
                else:
                    inlined = self._inline_invoke(op, body, value_map, visiting_blocks)
                    result.extend(inlined)

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

            elif isinstance(op, InverseBlockOperation):
                source_block = self._inline_nested_block(
                    op.source_block,
                    visiting_blocks,
                )
                implementation_block = self._inline_nested_block(
                    op.implementation_block,
                    visiting_blocks,
                )
                new_op = dataclasses.replace(
                    op,
                    source_block=source_block,
                    implementation_block=implementation_block,
                )
                substituted = self._substitute_values(new_op, value_map)
                result.append(substituted)

            elif isinstance(op, ControlledUOperation):
                # ControlledUOperation carries its unitary as a nested
                # ``block`` field (like InverseBlockOperation's source /
                # implementation blocks), not as HasNestedOps children, so
                # the generic recursion above never reaches it. Inline the
                # calls inside that block here. Without this, a pass-through
                # wrapper kernel (whose body is a single inline invocation
                # forwarding to a leaf gate) keeps the unexpanded call in
                # ``block``; at emit time ``blockvalue_to_gate`` cannot turn
                # a residual inline call into a gate, so the wrapped unitary
                # collapses to the identity and the controlled gate is
                # silently dropped.
                new_op = dataclasses.replace(
                    op,
                    block=self._inline_nested_block(op.block, visiting_blocks),
                )
                substituted = self._substitute_values(new_op, value_map)
                result.append(substituted)

            else:
                # Regular operation - apply value substitutions
                substituted = self._substitute_values(op, value_map)
                result.append(substituted)

        return result

    def _inline_block_call(
        self,
        *,
        block: Block,
        call_operands: list[Value],
        call_results: list[Value],
        value_map: dict[str, Value],
        visiting_blocks: set[int],
    ) -> list[Operation]:
        """Inline a callable block into the caller operation stream.

        Creates a fresh value_map for the called block's scope,
        mapping block inputs to call arguments.

        Args:
            block (Block): Callable body to inline.
            call_operands (list[Value]): Actual argument values passed by the
                call site.
            call_results (list[Value]): Result Values produced by the call
                site.
            value_map (dict[str, Value]): Caller-scope value substitutions
                accumulated so far; updated in place with the mappings for
                this call's results.
            visiting_blocks (set[int]): ids of blocks currently being
                expanded, used to leave self-recursive calls intact.

        Returns:
            list[Operation]: The callee's operations, cloned and rewritten
                into the caller's value scope.

        Raises:
            QubitConsumedError: If two of the call's quantum operands
                resolve to the same value — inlining would silently alias
                two formal registers onto the same qubits, and this pass
                dissolves the call before ``affine_validate`` could see the
                duplicate.  Frontend-traced kernels reject this earlier in
                ``QKernel.__call__``; this guard covers hand-built or
                deserialized IR.
        """
        call_args = call_operands  # Arguments passed to the call

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
        # Same-uuid duplicates among the resolved quantum operands mean the
        # call binds one register to two parameters.  Distinct SSA versions
        # of one wire are legitimate IR values, so this deliberately keys on
        # ``uuid`` (not ``logical_id``); wire-level linearity is enforced by
        # the frontend.
        seen_quantum_args: dict[str, str] = {}
        for arg_index, (block_input, call_arg) in enumerate(
            zip(block.input_values, call_args)
        ):
            substituted_arg = arg_substitutor.substitute_value(call_arg)
            resolved_arg = cast(Value, substituted_arg)
            if isinstance(resolved_arg, Value) and resolved_arg.type.is_quantum():
                label = (
                    block.label_args[arg_index]
                    if arg_index < len(block.label_args)
                    else f"argument {arg_index}"
                )
                first_label = seen_quantum_args.get(resolved_arg.uuid)
                if first_label is not None:
                    raise QubitConsumedError(
                        f"Call into block '{block.name}' binds the same "
                        f"quantum value ('{resolved_arg.name}') to parameters "
                        f"'{first_label}' and '{label}'.\n\n"
                        f"Affine type rule: Each qubit register can be passed "
                        f"to a kernel call at most once — binding one register "
                        f"to two parameters would alias both onto the same "
                        f"physical qubits.",
                        handle_name=resolved_arg.name,
                        operation_name=f"inline[{block.name}]",
                    )
                seen_quantum_args[resolved_arg.uuid] = label
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
        for block_return, call_result in zip(return_values, call_results):
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
            # each call result with a fresh shape dim
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

    def _inline_nested_block(
        self,
        block: Block | None,
        visiting_blocks: set[int],
    ) -> Block | None:
        """Inline calls stored inside an operation-owned nested block.

        Args:
            block (Block | None): Nested block to inline. ``None`` is
                returned unchanged.
            visiting_blocks (set[int]): Blocks currently being expanded by
                the enclosing inline traversal.

        Returns:
            Block | None: Nested block with its internal calls inlined as far
                as the current recursion context allows.
        """
        if block is None:
            return None

        value_map: dict[str, Value] = {}
        serialized_ops = self._serialize_operations(
            block.operations,
            value_map,
            visiting_blocks | {id(block)},
        )
        output_values = [value_map.get(v.uuid, v) for v in block.output_values]
        out_kind = (
            BlockKind.HIERARCHICAL
            if _has_any_inline_call(serialized_ops)
            else BlockKind.AFFINE
        )
        return dataclasses.replace(
            block,
            output_values=output_values,
            operations=serialized_ops,
            kind=out_kind,
        )

    def _inline_invoke(
        self,
        op: InvokeOperation,
        body: Block,
        value_map: dict[str, Value],
        visiting_blocks: set[int],
    ) -> list[Operation]:
        """Inline an InvokeOperation through its callable body.

        Args:
            op (InvokeOperation): Invocation to inline.
            body (Block): Body resolved from ``op.definition``.
            value_map (dict[str, Value]): Caller-scope value substitutions.
            visiting_blocks (set[int]): Blocks currently being expanded.

        Returns:
            list[Operation]: Inlined operations.
        """
        return self._inline_block_call(
            block=body,
            call_operands=op.operands,
            call_results=op.results,
            value_map=value_map,
            visiting_blocks=visiting_blocks,
        )

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
