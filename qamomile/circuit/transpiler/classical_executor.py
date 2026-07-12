"""Classical segment executor for Python-based classical operations."""

from __future__ import annotations

from typing import Any

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    CompOp,
    CondOp,
    NotOp,
    RuntimeClassicalExpr,
)
from qamomile.circuit.ir.operation.classical_ops import (
    DecodeQFixedOperation,
    DictGetItemOperation,
    StoreArrayElementOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    IfOperation,
    RegionArg,
    WhileOperation,
    validate_region_args,
)
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    array_static_length,
)
from qamomile.circuit.transpiler.errors import ExecutionError
from qamomile.circuit.transpiler.execution_context import ExecutionContext
from qamomile.circuit.transpiler.segments import ClassicalSegment


class ClassicalExecutor:
    """Executes classical segments in Python."""

    def execute(
        self,
        segment: ClassicalSegment,
        context: ExecutionContext,
    ) -> dict[str, Any]:
        """Execute classical operations and return outputs.

        Interprets the operations list directly using Python.
        """
        results: dict[str, Any] = {}
        self._execute_operations(segment.operations, context, results, {})
        return results

    def _execute_operations(
        self,
        operations: list[Operation],
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> None:
        for op in operations:
            self._execute_operation(op, context, results, scoped_locals)

    def _execute_operation(
        self,
        op: Operation,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> None:
        """Execute a single operation."""
        if isinstance(op, BinOp):
            self._execute_binop(op, context, results, scoped_locals)
        elif isinstance(op, CompOp):
            self._execute_compop(op, context, results, scoped_locals)
        elif isinstance(op, NotOp):
            self._execute_notop(op, context, results, scoped_locals)
        elif isinstance(op, CondOp):
            self._execute_condop(op, context, results, scoped_locals)
        elif isinstance(op, RuntimeClassicalExpr):
            self._execute_runtime_classical_expr(op, context, results, scoped_locals)
        elif isinstance(op, DecodeQFixedOperation):
            self._execute_decode_qfixed(op, context, results, scoped_locals)
        elif isinstance(op, DictGetItemOperation):
            self._execute_dict_getitem(op, context, results, scoped_locals)
        elif isinstance(op, StoreArrayElementOperation):
            self._execute_store_array_element(op, context, results, scoped_locals)
        elif isinstance(op, ForOperation):
            self._execute_for(op, context, results, scoped_locals)
        elif isinstance(op, ForItemsOperation):
            self._execute_for_items(op, context, results, scoped_locals)
        elif isinstance(op, IfOperation):
            self._execute_if(op, context, results, scoped_locals)
        elif isinstance(op, WhileOperation):
            self._execute_while(op, context, results, scoped_locals)
        elif isinstance(op, HasNestedOps):
            raise ExecutionError(
                f"Unhandled control flow in classical executor: {type(op).__name__}"
            )
        else:
            raise ExecutionError(
                f"Unsupported classical operation: {type(op).__name__}"
            )

    def _execute_binop(
        self,
        op: BinOp,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> None:
        """Execute binary operation.

        Kind dispatch is delegated to ``eval_utils.evaluate_binop_values`` so
        the same op semantics are used by compile-time folding, runtime
        execution, and emit-time inlining.
        """
        from qamomile.circuit.transpiler.passes.eval_utils import (
            evaluate_binop_values,
        )

        lhs = self._get_value(op.operands[0], context, results, scoped_locals)
        rhs = self._get_value(op.operands[1], context, results, scoped_locals)
        result_value = evaluate_binop_values(op.kind, lhs, rhs)
        if result_value is None:
            raise ExecutionError(f"BinOp evaluation failed: kind={op.kind}")
        if op.results:
            results[op.results[0].uuid] = result_value

    def _execute_compop(
        self,
        op: CompOp,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> None:
        """Execute comparison operation."""
        from qamomile.circuit.transpiler.passes.eval_utils import (
            evaluate_compop_values,
        )

        lhs = self._get_value(op.operands[0], context, results, scoped_locals)
        rhs = self._get_value(op.operands[1], context, results, scoped_locals)
        result_value = evaluate_compop_values(op.kind, lhs, rhs)
        if result_value is None:
            raise ExecutionError(f"CompOp evaluation failed: kind={op.kind}")
        if op.results:
            results[op.results[0].uuid] = result_value

    def _execute_notop(
        self,
        op: NotOp,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> None:
        """Execute not operation."""
        from qamomile.circuit.transpiler.passes.eval_utils import (
            evaluate_notop_value,
        )

        operand = self._get_value(op.operands[0], context, results, scoped_locals)
        result_value = evaluate_notop_value(operand)
        if result_value is None:
            raise ExecutionError("NotOp evaluation failed")
        if op.results:
            results[op.results[0].uuid] = result_value

    def _execute_condop(
        self,
        op: CondOp,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> None:
        """Execute conditional operation (AND, OR)."""
        from qamomile.circuit.transpiler.passes.eval_utils import (
            evaluate_condop_values,
        )

        lhs = self._get_value(op.operands[0], context, results, scoped_locals)
        rhs = self._get_value(op.operands[1], context, results, scoped_locals)
        result_value = evaluate_condop_values(op.kind, lhs, rhs)
        if result_value is None:
            raise ExecutionError(f"CondOp evaluation failed: kind={op.kind}")
        if op.results:
            results[op.results[0].uuid] = result_value

    def _execute_runtime_classical_expr(
        self,
        op: RuntimeClassicalExpr,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> None:
        """Execute a lowered measurement-derived classical expression.

        ``ClassicalLoweringPass`` preserves every operand/result ``Value``
        while replacing the original arithmetic/predicate operation with this
        unified IR node. Resolving operands through :meth:`_get_value` therefore
        preserves UUID-first identity and loop RegionArg state exactly as for
        the original operation families.

        Args:
            op (RuntimeClassicalExpr): Lowered expression to evaluate.
            context (ExecutionContext): Runtime bindings and measurements.
            results (dict[str, Any]): Current segment and loop results by UUID.
            scoped_locals (dict[str, Any]): Legacy loop display-name scope.

        Raises:
            ExecutionError: If the operation has no single result, has invalid
                arity/kind, or cannot be evaluated (for example division by
                zero).
        """
        from qamomile.circuit.transpiler.passes.eval_utils import (
            evaluate_runtime_op_values,
        )

        if len(op.results) != 1:
            raise ExecutionError(
                "RuntimeClassicalExpr requires exactly one result; "
                f"got {len(op.results)}."
            )
        operands = [
            self._get_value(operand, context, results, scoped_locals)
            for operand in op.operands
        ]
        result_value = evaluate_runtime_op_values(op.kind, operands)
        if result_value is None:
            raise ExecutionError(
                "RuntimeClassicalExpr evaluation failed: "
                f"kind={op.kind}, operand_count={len(operands)}."
            )
        results[op.results[0].uuid] = result_value

    def _execute_decode_qfixed(
        self,
        op: DecodeQFixedOperation,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> None:
        """Decode measured bits to float.

        After QPE with inverse QFT, bits are ordered [LSB, ..., MSB].
        The decoding formula accounts for this:
            float_value = sum(bit[i] * 2^(int_bits - n + i))

        For QPE phase (int_bits=0, n=4):
            weights: [0.0625, 0.125, 0.25, 0.5] for bits[0] to bits[3]

        Example:
            bits = [1, 0, 1] with int_bits=0 and n=3
            After QPE: bits[0]=LSB, bits[2]=MSB
            -> value = bits[0]*0.125 + bits[1]*0.25 + bits[2]*0.5
                     = 1*0.125 + 0*0.25 + 1*0.5 = 0.625
        """
        from qamomile.circuit.ir.value import ArrayValue

        # Get bit values from operands
        # Handle both ArrayValue (new) and individual bits (legacy)
        bits: list[Any] = []
        if len(op.operands) == 1 and isinstance(op.operands[0], ArrayValue):
            # New format: single ArrayValue containing all bits
            bits_array = op.operands[0]
            n = op.num_bits
            for i in range(n):
                # Bits are stored in context with indexed UUID format
                bit_uuid = f"{bits_array.uuid}_{i}"
                if context.has(bit_uuid):
                    bits.append(context.get(bit_uuid))
                elif bit_uuid in results:
                    bits.append(results[bit_uuid])
                else:
                    raise ExecutionError(
                        f"Bit {i} not found for ArrayValue {bits_array.name}"
                    )
        else:
            # Legacy format: individual bit operands
            bits = [
                self._get_value(b, context, results, scoped_locals) for b in op.operands
            ]

        # Decode: bits -> fixed-point float
        # After QPE with inverse QFT, bits are in order [LSB, ..., MSB]
        # So bits[n-1] is MSB (weight 0.5), bits[0] is LSB
        n = len(bits)
        value = 0.0
        for i, bit in enumerate(bits):
            # Position: int_bits - n + i
            # For int_bits=0 and n=4: positions are -4, -3, -2, -1
            # -> weights: 0.0625, 0.125, 0.25, 0.5
            position = op.int_bits - n + i
            value += int(bit) * (2.0**position)

        if op.results:
            results[op.results[0].uuid] = value

    def _execute_store_array_element(
        self,
        op: StoreArrayElementOperation,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> None:
        """Execute a classical array element store.

        Reads the current contents of the source array, replaces the
        addressed element with the stored value, and records the updated
        container under the result value's UUID.

        Inside a loop body each store op executes once per iteration
        while the IR carries a single ``source -> result`` version pair.
        To keep iterations cumulative — including bodies with multiple
        stores to the same logical array — the running contents are
        shared across all stores to that array via a namespaced
        ``results`` key derived from the array's ``logical_id`` (stable
        across SSA versions). Each store bases its update on that shared
        snapshot when present (falling back to the source array on the
        very first store) and writes the updated contents back to both
        the shared key and its own ``result.uuid``. Stores that read an
        element of the same logical array they write (which would see
        stale pre-loop contents in this loop-carried mode) are rejected
        earlier, at compile time, by ``AnalyzePass``.

        Args:
            op (StoreArrayElementOperation): The store to execute.
            context (ExecutionContext): Execution context holding
                measurements and bindings.
            results (dict[str, Any]): Mutable results map; the updated
                container is recorded under ``op.results[0].uuid`` and
                under the array's shared running-state key.
            scoped_locals (dict[str, Any]): Loop-scoped variables.

        Raises:
            ExecutionError: If the source array contents cannot be
                resolved or the index is out of range.
        """
        result_value = op.results[0]

        # The store's array and its result share one logical_id across SSA
        # versions; the prefix keeps this running-state key from colliding
        # with value-uuid / name keys in ``results``.
        state_key = f"__store_array_state__:{result_value.logical_id}"

        if state_key in results:
            # A previous store to the same logical array (earlier in this
            # iteration or in a previous one): continue from its contents.
            base = results[state_key]
        else:
            base = self._resolve_store_base(op.array, context, results, scoped_locals)

        index_values = op.index_values
        if len(index_values) != 1:
            raise ExecutionError(
                f"StoreArrayElementOperation supports 1-D arrays only; "
                f"got {len(index_values)} indices."
            )
        index = int(self._get_value(index_values[0], context, results, scoped_locals))

        elements = list(base)
        if not 0 <= index < len(elements):
            raise ExecutionError(
                f"Store index {index} is out of range for array "
                f"'{op.array.name}' of length {len(elements)}."
            )
        elements[index] = self._get_value(
            op.stored_value, context, results, scoped_locals
        )
        updated = tuple(elements)
        # The shared key chains subsequent stores to the same logical array;
        # the per-version uuid entry keeps downstream reads of this SSA
        # version and block-output resolution via output_refs working.
        results[state_key] = updated
        results[result_value.uuid] = updated

    def _materialize_loop_store_defaults(
        self,
        operations: list[Operation],
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> None:
        """Materialize zero-iteration defaults for loop-carried array stores.

        Store operations inside a traced loop produce SSA result values
        even when the loop executes zero times at runtime.  Python
        semantics leave the array unchanged in that case, so each store
        result in the loop body must resolve to the current pre-loop
        contents until an executed iteration overwrites it.

        Args:
            operations (list[Operation]): Loop body operations to scan,
                including nested control-flow bodies.
            context (ExecutionContext): Execution context holding
                measurements and bindings.
            results (dict[str, Any]): Mutable results map seeded with
                default store outputs.
            scoped_locals (dict[str, Any]): Loop-scoped variables.

        Raises:
            ExecutionError: If a store's source array cannot be resolved.
        """
        for op in operations:
            if isinstance(op, StoreArrayElementOperation):
                self._materialize_store_default(op, context, results, scoped_locals)
            elif isinstance(op, HasNestedOps):
                for body in op.nested_op_lists():
                    self._materialize_loop_store_defaults(
                        body, context, results, scoped_locals
                    )

    def _materialize_store_default(
        self,
        op: StoreArrayElementOperation,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> None:
        """Record a store result as the current source-array contents.

        Args:
            op (StoreArrayElementOperation): Store whose result should
                default to the pre-loop array state.
            context (ExecutionContext): Execution context holding
                measurements and bindings.
            results (dict[str, Any]): Mutable results map updated under
                both the store result UUID and the logical running-state key.
            scoped_locals (dict[str, Any]): Loop-scoped variables.

        Raises:
            ExecutionError: If the source array cannot be resolved.
        """
        result_value = op.results[0]
        state_key = f"__store_array_state__:{result_value.logical_id}"
        base = (
            results[state_key]
            if state_key in results
            else self._resolve_store_base(op.array, context, results, scoped_locals)
        )
        results[state_key] = base
        results[result_value.uuid] = base

    def _resolve_store_base(
        self,
        array_value: ArrayValue,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> Any:
        """Resolve the full contents of a store's source array.

        Tries the whole-container lookup used for reads
        (:meth:`_get_array_data`) first, then falls back to assembling
        per-element composite carrier keys (``"<uuid>_<index>"``) — the
        format under which measurement results are loaded into the
        execution context.

        Args:
            array_value (ArrayValue): The store's source array operand.
            context (ExecutionContext): Execution context holding
                measurements and bindings.
            results (dict[str, Any]): Results map of the current segment.
            scoped_locals (dict[str, Any]): Loop-scoped variables.

        Returns:
            Any: A sequence with the array's current contents.

        Raises:
            ExecutionError: If neither a whole container nor per-element
                keys can be resolved.
        """
        container = self._get_array_data(array_value, context, results, scoped_locals)
        if container is not None:
            return container

        elements: list[Any] = []
        i = 0
        while context.has(f"{array_value.uuid}_{i}"):
            elements.append(context.get(f"{array_value.uuid}_{i}"))
            i += 1
        if elements:
            return tuple(elements)

        raise ExecutionError(
            f"Array contents for '{array_value.name}' could not be "
            f"resolved for element store."
        )

    def _execute_for(
        self,
        op: ForOperation,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> None:
        """Execute a classical for loop."""
        start = (
            int(self._get_value(op.operands[0], context, results, scoped_locals))
            if len(op.operands) > 0
            else 0
        )
        stop = (
            int(self._get_value(op.operands[1], context, results, scoped_locals))
            if len(op.operands) > 1
            else 0
        )
        step = (
            int(self._get_value(op.operands[2], context, results, scoped_locals))
            if len(op.operands) > 2
            else 1
        )

        if step == 0:
            raise ExecutionError("ForOperation step must not be zero")

        self._materialize_loop_store_defaults(
            op.operations, context, results, scoped_locals
        )
        self._validated_region_args(op)
        if op.loop_var_value is None:
            raise ExecutionError(
                f"ForOperation '{op.loop_var or '<unnamed>'}' has no "
                "loop_var_value; the IR must be rebuilt with the current "
                "frontend."
            )
        self._seed_region_args(op, context, results, scoped_locals)
        for loop_value in range(start, stop, step):
            loop_scope = scoped_locals.copy()
            results[op.loop_var_value.uuid] = loop_value
            self._execute_operations(op.operations, context, results, loop_scope)
            self._advance_region_args(op, context, results, loop_scope)
        self._publish_region_results(op, results)

    def _validated_region_args(
        self,
        op: ForOperation | ForItemsOperation | WhileOperation,
    ) -> tuple[RegionArg, ...]:
        """Validate and return a loop operation's region arguments.

        Args:
            op (ForOperation | ForItemsOperation | WhileOperation): The
                loop operation to validate.

        Returns:
            tuple[RegionArg, ...]: The validated region arguments.

        Raises:
            ExecutionError: If result counts or UUIDs do not align,
                block/result UUIDs are duplicated, or one argument's
                values have different types.
        """
        try:
            return validate_region_args(op)
        except ValueError as error:
            raise ExecutionError(str(error)) from error

    def _seed_region_args(
        self,
        op: ForOperation | ForItemsOperation | WhileOperation,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> None:
        """Bind each region argument's block argument to its init value.

        Runs once before iteration 0 so the body's loop-carried reads
        (which reference ``block_arg``) observe the pre-loop value on
        the first pass.

        Args:
            op (ForOperation | ForItemsOperation | WhileOperation): The
                loop being executed.
            context (ExecutionContext): Execution context.
            results (dict[str, Any]): Intermediate results by UUID.
            scoped_locals (dict[str, Any]): Loop-scoped variables.
        """
        for arg in self._validated_region_args(op):
            results[arg.block_arg.uuid] = self._get_value(
                arg.init, context, results, scoped_locals
            )

    def _advance_region_args(
        self,
        op: ForOperation | ForItemsOperation | WhileOperation,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> None:
        """Carry each region argument's yielded value into the next iteration.

        Runs after each body execution: the block argument is rebound to
        the value the body just yielded, exactly the MLIR
        ``iter_args`` / ``yield`` step.

        Args:
            op (ForOperation | ForItemsOperation | WhileOperation): The
                loop being executed.
            context (ExecutionContext): Execution context.
            results (dict[str, Any]): Intermediate results by UUID.
            scoped_locals (dict[str, Any]): Loop-scoped variables.
        """
        # Two-phase: resolve every yielded value against the CURRENT
        # iteration's state before rebinding any block argument, so
        # simultaneous carries (``a, b = b, a``) read the pre-advance
        # values instead of a partially updated mix.
        advanced = {
            arg.block_arg.uuid: self._get_value(
                arg.yielded, context, results, scoped_locals
            )
            for arg in self._validated_region_args(op)
        }
        results.update(advanced)

    def _publish_region_results(
        self,
        op: ForOperation | ForItemsOperation | WhileOperation,
        results: dict[str, Any],
    ) -> None:
        """Expose each region argument's final carried value as the loop result.

        Runs once after the last iteration (or immediately for a
        zero-trip loop, in which case the result is the init value the
        seed step stored).

        Args:
            op (ForOperation | ForItemsOperation | WhileOperation): The
                executed loop.
            results (dict[str, Any]): Intermediate results by UUID.
        """
        for arg in self._validated_region_args(op):
            results[arg.result.uuid] = results[arg.block_arg.uuid]

    def _execute_for_items(
        self,
        op: ForItemsOperation,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> None:
        """Execute a classical dict iteration."""
        if not op.operands:
            raise ExecutionError("ForItemsOperation requires an iterable operand")
        if op.key_var_values is None or len(op.key_var_values) != len(op.key_vars):
            raise ExecutionError(
                "ForItemsOperation key identities are missing or inconsistent; "
                "the IR must be rebuilt with the current frontend."
            )
        if op.value_var_value is None:
            raise ExecutionError(
                "ForItemsOperation value identity is missing; the IR must be "
                "rebuilt with the current frontend."
            )
        if op.key_is_vector:
            if (
                not op.key_var_values
                or not isinstance(op.key_var_values[0], ArrayValue)
                or not op.key_var_values[0].shape
            ):
                raise ExecutionError(
                    "ForItemsOperation vector-key shape identity is missing; "
                    "the IR must be rebuilt with the current frontend."
                )
        self._validated_region_args(op)

        iterable = self._get_iterable(op.operands[0], context, results, scoped_locals)

        self._materialize_loop_store_defaults(
            op.operations, context, results, scoped_locals
        )
        self._seed_region_args(op, context, results, scoped_locals)
        for key, value in iterable:
            loop_scope = scoped_locals.copy()
            self._bind_for_items_key(loop_scope, results, op, key)
            loop_scope[op.value_var] = value
            results[op.value_var_value.uuid] = value
            self._execute_operations(op.operations, context, results, loop_scope)
            self._advance_region_args(op, context, results, loop_scope)
        self._publish_region_results(op, results)

    def _execute_if(
        self,
        op: IfOperation,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> None:
        """Execute a classical if/else.

        Runs the taken branch's operations, then resolves every merged
        output to its selected branch source.

        Args:
            op (IfOperation): The if-else to execute.
            context (ExecutionContext): Execution context holding
                measurements and bindings.
            results (dict[str, Any]): Mutable results map; merged outputs
                are recorded under their result UUIDs.
            scoped_locals (dict[str, Any]): Loop-scoped variables.
        """
        condition = bool(self._get_value(op.condition, context, results, scoped_locals))
        branch_scope = scoped_locals.copy()
        branch_ops = op.true_operations if condition else op.false_operations
        self._execute_operations(branch_ops, context, results, branch_scope)

        for merge in op.iter_merges():
            selected = merge.select(condition)
            results[merge.result.uuid] = self._get_value(
                selected, context, results, branch_scope
            )

    def _execute_while(
        self,
        op: WhileOperation,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> None:
        """Execute a classical while loop."""
        if not op.operands:
            raise ExecutionError("WhileOperation requires a condition operand")

        condition_value = op.operands[0]
        next_condition = op.operands[1] if len(op.operands) > 1 else condition_value

        self._validated_region_args(op)
        self._seed_region_args(op, context, results, scoped_locals)
        while bool(self._get_value(condition_value, context, results, scoped_locals)):
            loop_scope = scoped_locals.copy()
            self._execute_operations(op.operations, context, results, loop_scope)
            self._advance_region_args(op, context, results, loop_scope)
            condition_value = next_condition
        self._publish_region_results(op, results)

    def _execute_dict_getitem(
        self,
        op: DictGetItemOperation,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> None:
        """Execute a dict subscript lookup (``d[key]``).

        Resolves the key components to concrete values, resolves the
        dict entries via :meth:`_get_iterable`, and stores the looked-up
        value under the result UUID.

        Args:
            op (DictGetItemOperation): The lookup op being executed.
            context (ExecutionContext): Execution context.
            results (dict[str, Any]): Intermediate results by UUID.
            scoped_locals (dict[str, Any]): Loop-scoped variables.

        Raises:
            ExecutionError: If the key is not found in the dict.
        """
        resolved_key = [
            int(self._get_value(kv, context, results, scoped_locals))
            for kv in op.operands[1:]
        ]
        entries = self._get_iterable(op.operands[0], context, results, scoped_locals)
        lookup_key: Any = tuple(resolved_key) if op.key_arity > 1 else resolved_key[0]
        for entry_key, entry_value in entries:
            if isinstance(entry_key, (tuple, list)):
                entry_key = tuple(entry_key)
            if entry_key == lookup_key:
                results[op.results[0].uuid] = entry_value
                return
        raise ExecutionError(
            f"Key {lookup_key!r} not found in dict "
            f"'{getattr(op.operands[0], 'name', '?')}'"
        )

    def _get_value(
        self,
        value: Value,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> Any:
        """Get the concrete value from context or results.

        Args:
            value (Value): Scalar, container, or structural element to resolve.
            context (ExecutionContext): Runtime bindings and measurements.
            results (dict[str, Any]): Current segment results by UUID.
            scoped_locals (dict[str, Any]): Legacy loop display-name scope.

        Returns:
            Any: Resolved runtime or compile-time value.

        Raises:
            ExecutionError: If the value cannot be resolved from the
                execution state — with a slice-view-specific diagnostic
                when the unresolved value is a sliced ``ArrayValue``
                (most commonly a view merged from different slices
                across if/else branches, which has no materialized
                contents).
        """
        if value.uuid in results:
            return results[value.uuid]
        if context.has(value.uuid):
            return context.get(value.uuid)
        if isinstance(value, ArrayValue):
            const_array = value.get_const_array()
            if const_array is not None:
                return const_array
            if array_static_length(value) == 0:
                return ()
        if isinstance(value, DictValue) and value.metadata.dict_runtime is not None:
            return value.get_bound_data()
        if value.is_constant():
            return value.get_const()
        if value.is_parameter():
            param_name = value.parameter_name()
            if param_name and context.has(param_name):
                return context.get(param_name)
        if value.is_array_element():
            return self._get_array_element_value(value, context, results, scoped_locals)
        # Display-name lookups are a legacy compatibility bridge. Keep them
        # last so a user parameter named ``uint_const`` (or like an internal
        # array/dict temporary) cannot shadow typed IR identity/metadata.
        if value.name in scoped_locals:
            return scoped_locals[value.name]
        if value.name and context.has(value.name):
            return context.get(value.name)
        if isinstance(value, ArrayValue) and value.slice_of is not None:
            raise ExecutionError(
                f"Array view '{value.name}' could not be resolved in the "
                f"classical segment. A common cause is merging different "
                f"slices across if/else branches — such a merged view has "
                f"no materialized contents. Slice identically in both "
                f"branches or read the root array instead."
            )
        raise ExecutionError(f"Value {value.name} not found in context or results")

    def _get_array_element_value(
        self,
        value: Value,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> Any:
        """Resolve an array element from its parent container.

        Args:
            value (Value): Element value carrying parent/index metadata.
            context (ExecutionContext): Runtime bindings and measurements.
            results (dict[str, Any]): Current segment results by UUID.
            scoped_locals (dict[str, Any]): Legacy loop display-name scope.

        Returns:
            Any: Resolved element contents.

        Raises:
            ExecutionError: If indices/slice bounds or array contents cannot
                be resolved.
        """
        parent = value.parent_array
        if parent is None:
            raise ExecutionError(f"Value {value.name} is not an array element")

        indices = tuple(
            int(self._get_value(idx, context, results, scoped_locals))
            for idx in value.element_indices
        )
        location = self._resolve_array_element_location(
            parent,
            indices,
            context,
            results,
            scoped_locals,
        )
        if location is None:
            raise ExecutionError(
                f"Array element {parent.name}{indices} has unresolved slice bounds"
            )
        root, root_indices = location
        # A view can inherit the root parameter's metadata. Resolve and index
        # the root container, never the view-local index against that root
        # payload (``values[1:][0]`` must read root slot 1, not slot 0).
        container = self._get_array_data_by_identity(root, context, results)

        if container is not None:
            return self._index_array_container(container, root_indices)

        if len(root_indices) == 1:
            # Measurement results are loaded into the context under
            # per-element composite carrier keys ("<root_uuid>_<index>").
            # Compose slice views back onto the root so a view-local
            # index addresses the right physical slot.
            composite_key = f"{root.uuid}_{root_indices[0]}"
            if context.has(composite_key):
                return context.get(composite_key)

        # Some control-flow merges and host-side transforms materialize a view
        # under the view's own UUID instead of its root. Preserve that explicit
        # payload as the final identity-based fallback, indexed in view-local
        # coordinates.
        if parent.uuid != root.uuid:
            parent_container = self._get_array_data_by_identity(
                parent, context, results
            )
            if parent_container is not None:
                return self._index_array_container(parent_container, indices)

        if len(root_indices) == 1:
            indexed_key = f"{root.name}[{root_indices[0]}]"
            if root.name and context.has(indexed_key):
                return context.get(indexed_key)

        # Programmatic/legacy IR may have only display-name bindings. This is
        # deliberately after root UUID/metadata/parameter and physical
        # composite-key resolution.
        container = self._get_array_data(root, context, results, scoped_locals)
        if container is not None:
            return self._index_array_container(container, root_indices)
        if parent.uuid != root.uuid:
            legacy_parent_container = self._get_array_data(
                parent, context, results, scoped_locals
            )
            if legacy_parent_container is not None:
                return self._index_array_container(legacy_parent_container, indices)

        raise ExecutionError(
            f"Array element {parent.name}{indices} could not be resolved"
        )

    @staticmethod
    def _index_array_container(container: Any, indices: tuple[int, ...]) -> Any:
        """Index a one- or multi-dimensional classical container.

        Args:
            container (Any): Runtime array-like payload.
            indices (tuple[int, ...]): One-dimensional scalar index or a
                multi-dimensional index tuple.

        Returns:
            Any: Element stored at ``indices``.

        Raises:
            IndexError: If an index is outside the container bounds.
            KeyError: If a mapping-like container lacks the indexed key.
            TypeError: If the container does not support the requested index
                form.
        """
        if len(indices) == 1:
            return container[indices[0]]
        return container[indices]

    def _resolve_array_element_location(
        self,
        parent: ArrayValue,
        indices: tuple[int, ...],
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> tuple[ArrayValue, tuple[int, ...]] | None:
        """Compose an element's indices through its slice ancestry.

        Args:
            parent (ArrayValue): Immediate parent array or sliced view.
            indices (tuple[int, ...]): Indices local to ``parent``.
            context (ExecutionContext): Runtime bindings and measurements.
            results (dict[str, Any]): Current segment results by UUID.
            scoped_locals (dict[str, Any]): Legacy loop display-name scope.

        Returns:
            tuple[ArrayValue, tuple[int, ...]] | None: Root array and root-local
                indices, or None when a slice frame is unresolved/invalid.

        Raises:
            ExecutionError: If a symbolic slice bound cannot be resolved from
                the current execution state.
        """
        if not indices:
            return parent, ()
        if any(index < 0 for index in indices):
            return None
        current = parent
        leading_index = indices[0]
        while current.slice_of is not None:
            if current.slice_start is None or current.slice_step is None:
                return None
            start = int(
                self._get_value(
                    current.slice_start,
                    context,
                    results,
                    scoped_locals,
                )
            )
            step = int(
                self._get_value(
                    current.slice_step,
                    context,
                    results,
                    scoped_locals,
                )
            )
            if start < 0 or step <= 0:
                return None
            leading_index = start + step * leading_index
            current = current.slice_of
        return current, (leading_index, *indices[1:])

    def _get_array_data(
        self,
        array_value: ArrayValue,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> Any:
        """Resolve an array-like container from execution state.

        Args:
            array_value (ArrayValue): Array identity to resolve.
            context (ExecutionContext): Runtime bindings by UUID/provenance.
            results (dict[str, Any]): Current segment results by UUID.
            scoped_locals (dict[str, Any]): Legacy loop display-name scope.

        Returns:
            Any: Resolved container, or None when absent.
        """
        resolved = self._get_array_data_by_identity(array_value, context, results)
        if resolved is not None:
            return resolved
        # Legacy display-name fallback comes last.
        if array_value.name in scoped_locals:
            return scoped_locals[array_value.name]
        if array_value.name and context.has(array_value.name):
            return context.get(array_value.name)
        return None

    @staticmethod
    def _get_array_data_by_identity(
        array_value: ArrayValue,
        context: ExecutionContext,
        results: dict[str, Any],
    ) -> Any:
        """Resolve an array without consulting its display name.

        Args:
            array_value (ArrayValue): Array identity to resolve.
            context (ExecutionContext): Runtime bindings by UUID/provenance.
            results (dict[str, Any]): Current segment results by UUID.

        Returns:
            Any: UUID-, metadata-, or parameter-resolved container, or None.
        """
        if array_value.uuid in results:
            return results[array_value.uuid]
        if context.has(array_value.uuid):
            return context.get(array_value.uuid)
        const_array = array_value.get_const_array()
        if const_array is not None:
            return const_array
        if array_static_length(array_value) == 0:
            return ()
        if array_value.is_parameter():
            param_name = array_value.parameter_name()
            if param_name and context.has(param_name):
                return context.get(param_name)
        return None

    def _get_iterable(
        self,
        iterable_value: Any,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> list[tuple[Any, Any]]:
        """Resolve a DictValue to concrete key/value pairs.

        Args:
            iterable_value (Any): Candidate dictionary IR value.
            context (ExecutionContext): Runtime bindings and measurements.
            results (dict[str, Any]): Intermediate results by UUID.
            scoped_locals (dict[str, Any]): Loop-scoped values by display name.

        Returns:
            list[tuple[Any, Any]]: Concrete entries in iteration order,
                including an empty list for an explicitly bound empty dict.

        Raises:
            ExecutionError: If no concrete dictionary source can be resolved.
        """
        if isinstance(iterable_value, DictValue):
            if context.has(iterable_value.uuid):
                return list(context.get(iterable_value.uuid).items())
            if iterable_value.metadata.dict_runtime is not None:
                return list(iterable_value.get_bound_data_items())
            if iterable_value.is_parameter():
                parameter_name = iterable_value.parameter_name()
                if parameter_name and context.has(parameter_name):
                    return list(context.get(parameter_name).items())
            if iterable_value.entries:
                return [
                    (
                        self._resolve_structured_value(
                            key, context, results, scoped_locals
                        ),
                        self._get_value(val, context, results, scoped_locals),
                    )
                    for key, val in iterable_value.entries
                ]
            # Legacy display-name fallback comes after typed entries and
            # parameter provenance.
            if iterable_value.name in scoped_locals:
                return list(scoped_locals[iterable_value.name].items())
            if iterable_value.name and context.has(iterable_value.name):
                return list(context.get(iterable_value.name).items())

        raise ExecutionError("ForItemsOperation iterable could not be resolved")

    def _resolve_structured_value(
        self,
        value: TupleValue | Value,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> Any:
        """Resolve a tuple or scalar value to a Python object."""
        if isinstance(value, TupleValue):
            return tuple(
                self._get_value(elem, context, results, scoped_locals)
                for elem in value.elements
            )
        return self._get_value(value, context, results, scoped_locals)

    def _bind_for_items_key(
        self,
        loop_scope: dict[str, Any],
        results: dict[str, Any],
        op: ForItemsOperation,
        key: Any,
    ) -> None:
        """Bind for-items key variables by stable UUID and display name.

        Args:
            loop_scope (dict[str, Any]): Per-iteration display-name scope.
            results (dict[str, Any]): UUID-keyed execution results.
            op (ForItemsOperation): The for-items operation whose key
                identities were validated by :meth:`_execute_for_items`.
            key (Any): The current concrete dict key.

        Raises:
            ExecutionError: If a destructured key is not a tuple/list or
                has the wrong arity.
        """
        assert op.key_var_values is not None
        if len(op.key_vars) == 1:
            loop_scope[op.key_vars[0]] = key
            results[op.key_var_values[0].uuid] = key
            if op.key_is_vector:
                key_value = op.key_var_values[0]
                assert isinstance(key_value, ArrayValue) and key_value.shape
                results[key_value.shape[0].uuid] = len(key)
            return

        if not isinstance(key, (tuple, list)):
            raise ExecutionError(
                f"ForItemsOperation expected tuple/list key for {op.key_vars}, got {key!r}"
            )
        if len(key) != len(op.key_vars):
            raise ExecutionError(
                f"ForItemsOperation key arity mismatch: expected {len(op.key_vars)}, "
                f"got {len(key)}"
            )

        for name, identity, element in zip(
            op.key_vars, op.key_var_values, key, strict=True
        ):
            loop_scope[name] = element
            results[identity.uuid] = element
