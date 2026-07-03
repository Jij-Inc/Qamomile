"""Classical segment executor for Python-based classical operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    CompOp,
    CompOpKind,
    CondOp,
    CondOpKind,
    NotOp,
    PhiOp,
    RuntimeClassicalExpr,
    RuntimeOpKind,
)
from qamomile.circuit.ir.operation.classical_ops import (
    DecodeQFixedOperation,
    StoreArrayElementOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    resolve_root_array_index,
)
from qamomile.circuit.transpiler.errors import ExecutionError
from qamomile.circuit.transpiler.execution_context import ExecutionContext
from qamomile.circuit.transpiler.segments import ClassicalSegment

# ``RuntimeOpKind`` → per-family kind tables for host-side evaluation of
# ``RuntimeClassicalExpr``. Inverse of the forward maps used by
# ``ClassicalLoweringPass`` (see ``arithmetic_operations``); dispatching
# through the same ``eval_utils`` helpers as the ``BinOp`` / ``CompOp`` /
# ``CondOp`` executors above keeps runtime-expression semantics identical
# to compile-time folding.
_RUNTIME_TO_BINOP_KIND: dict[RuntimeOpKind, BinOpKind] = {
    RuntimeOpKind.ADD: BinOpKind.ADD,
    RuntimeOpKind.SUB: BinOpKind.SUB,
    RuntimeOpKind.MUL: BinOpKind.MUL,
    RuntimeOpKind.DIV: BinOpKind.DIV,
    RuntimeOpKind.FLOORDIV: BinOpKind.FLOORDIV,
    RuntimeOpKind.MOD: BinOpKind.MOD,
    RuntimeOpKind.POW: BinOpKind.POW,
}
_RUNTIME_TO_COMPOP_KIND: dict[RuntimeOpKind, CompOpKind] = {
    RuntimeOpKind.EQ: CompOpKind.EQ,
    RuntimeOpKind.NEQ: CompOpKind.NEQ,
    RuntimeOpKind.LT: CompOpKind.LT,
    RuntimeOpKind.LE: CompOpKind.LE,
    RuntimeOpKind.GT: CompOpKind.GT,
    RuntimeOpKind.GE: CompOpKind.GE,
}
_RUNTIME_TO_CONDOP_KIND: dict[RuntimeOpKind, CondOpKind] = {
    RuntimeOpKind.AND: CondOpKind.AND,
    RuntimeOpKind.OR: CondOpKind.OR,
}


def resolve_runtime_array_location(
    array: ArrayValue,
    indices: tuple[int, ...],
    resolve_int: Callable[[Value], int | None],
) -> tuple[ArrayValue, tuple[int, ...]] | None:
    """Resolve local array indices through runtime-bound slice views.

    Args:
        array (ArrayValue): Array whose local indices should be resolved. May
            be a root array or a ``slice_of`` view chain.
        indices (tuple[int, ...]): Concrete indices in ``array``'s local
            coordinate space.
        resolve_int (Callable[[Value], int | None]): Callback used to
            evaluate ``slice_start`` /
            ``slice_step`` values against the current runtime state.

    Returns:
        tuple[ArrayValue, tuple[int, ...]] | None: The root array and indices
            in root coordinates, or ``None`` when a slice bound is unresolved
            or violates the frontend slice contract.
    """
    if len(indices) != 1:
        return array, indices

    idx = indices[0]
    if idx < 0:
        return None

    current = array
    while current.slice_of is not None:
        if current.slice_start is None or current.slice_step is None:
            return None
        start = resolve_int(current.slice_start)
        step = resolve_int(current.slice_step)
        if start is None or step is None or start < 0 or step <= 0:
            return None
        idx = start + step * idx
        current = current.slice_of
    return current, (idx,)


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
            self._execute_runtime_expr(op, context, results, scoped_locals)
        elif isinstance(op, PhiOp):
            self._execute_phi(op, context, results, scoped_locals)
        elif isinstance(op, DecodeQFixedOperation):
            self._execute_decode_qfixed(op, context, results, scoped_locals)
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

    def _execute_runtime_expr(
        self,
        op: RuntimeClassicalExpr,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> None:
        """Evaluate a measurement-derived runtime expression host-side.

        ``SegmentationPass`` routes a ``RuntimeClassicalExpr`` into a
        classical segment when its result is a block output or feeds
        host-side post-processing (in-circuit consumers instead keep the
        op in the quantum segment, where backend emit lowers it). The
        unified ``RuntimeOpKind`` is mapped back to its per-family kind
        and delegated to the shared ``eval_utils`` helpers so evaluation
        semantics match compile-time folding exactly. Boolean results are
        coerced to ``int`` so Bit-typed outputs surface as ``0`` / ``1``,
        consistent with directly measured bits.

        Args:
            op (RuntimeClassicalExpr): The runtime expression to evaluate.
                Binary kinds read ``operands[0]`` / ``operands[1]``; the
                unary NOT kind reads ``operands[0]`` only.
            context (ExecutionContext): Execution context holding measured
                bit values and bound parameters.
            results (dict[str, Any]): Segment-local results keyed by value
                UUID; the expression result is written here.
            scoped_locals (dict[str, Any]): Loop-scoped local variables.

        Raises:
            ExecutionError: If ``op.kind`` is not a recognized
                ``RuntimeOpKind`` or evaluation fails (e.g. division by
                zero, operand type mismatch).
        """
        from qamomile.circuit.transpiler.passes.eval_utils import (
            evaluate_binop_values,
            evaluate_compop_values,
            evaluate_condop_values,
            evaluate_notop_value,
        )

        kind = op.kind
        result_value: float | int | bool | None
        if kind is RuntimeOpKind.NOT:
            operand = self._get_value(op.operands[0], context, results, scoped_locals)
            result_value = evaluate_notop_value(operand)
        else:
            lhs = self._get_value(op.operands[0], context, results, scoped_locals)
            rhs = self._get_value(op.operands[1], context, results, scoped_locals)
            if kind in _RUNTIME_TO_COMPOP_KIND:
                result_value = evaluate_compop_values(
                    _RUNTIME_TO_COMPOP_KIND[kind], lhs, rhs
                )
            elif kind in _RUNTIME_TO_CONDOP_KIND:
                result_value = evaluate_condop_values(
                    _RUNTIME_TO_CONDOP_KIND[kind], lhs, rhs
                )
            elif kind in _RUNTIME_TO_BINOP_KIND:
                result_value = evaluate_binop_values(
                    _RUNTIME_TO_BINOP_KIND[kind], lhs, rhs
                )
            else:
                raise ExecutionError(
                    f"RuntimeClassicalExpr has unsupported kind: {kind}"
                )
        if result_value is None:
            raise ExecutionError(
                f"RuntimeClassicalExpr evaluation failed (division by zero or "
                f"operand type mismatch): kind={kind}"
            )
        if isinstance(result_value, bool):
            result_value = int(result_value)
        if op.results:
            results[op.results[0].uuid] = result_value

    def _execute_phi(
        self,
        op: PhiOp,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> None:
        """Execute SSA phi merge after a conditional branch."""
        condition = bool(self._get_value(op.condition, context, results, scoped_locals))
        selected = op.true_value if condition else op.false_value
        merged = self._get_value(selected, context, results, scoped_locals)
        results[op.output.uuid] = merged

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
        for loop_value in range(start, stop, step):
            loop_scope = scoped_locals.copy()
            loop_scope[op.loop_var] = loop_value
            self._execute_operations(op.operations, context, results, loop_scope)

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

        iterable = self._get_iterable(op.operands[0], context, results, scoped_locals)

        self._materialize_loop_store_defaults(
            op.operations, context, results, scoped_locals
        )
        for key, value in iterable:
            loop_scope = scoped_locals.copy()
            self._bind_for_items_key(loop_scope, op, key)
            loop_scope[op.value_var] = value
            self._execute_operations(op.operations, context, results, loop_scope)

    def _execute_if(
        self,
        op: IfOperation,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> None:
        """Execute a classical if/else."""
        condition = bool(self._get_value(op.condition, context, results, scoped_locals))
        branch_scope = scoped_locals.copy()
        branch_ops = op.true_operations if condition else op.false_operations
        self._execute_operations(branch_ops, context, results, branch_scope)

        for phi_op in op.phi_ops:
            self._execute_phi(phi_op, context, results, branch_scope)

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

        while bool(self._get_value(condition_value, context, results, scoped_locals)):
            loop_scope = scoped_locals.copy()
            self._execute_operations(op.operations, context, results, loop_scope)
            condition_value = next_condition

    def _get_value(
        self,
        value: Value,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> Any:
        """Get the concrete value from context or results."""
        if value.uuid in results:
            return results[value.uuid]
        if context.has(value.uuid):
            return context.get(value.uuid)
        if value.name in scoped_locals:
            return scoped_locals[value.name]
        if context.has(value.name):
            return context.get(value.name)
        if value.is_array_element():
            return self._get_array_element_value(value, context, results, scoped_locals)
        # Check if it's a constant
        if value.is_constant():
            return value.get_const()
        if value.is_parameter():
            param_name = value.parameter_name()
            if param_name and context.has(param_name):
                return context.get(param_name)
        raise ExecutionError(f"Value {value.name} not found in context or results")

    def _get_array_element_value(
        self,
        value: Value,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> Any:
        """Resolve an array element from its parent container.

        Resolution order: the parent container as a whole (results /
        context / locals / constant / parameter), then per-element carrier
        keys — the ``{array_uuid}_{index}`` composite key under which the
        orchestrator stores measured bits (see
        ``ProgramOrchestrator._load_measurements``), then the legacy
        ``{array_name}[{index}]`` context key.

        Args:
            value (Value): The array-element value to resolve. Must carry
                ``parent_array`` metadata.
            context (ExecutionContext): Execution context holding measured
                bits and bound parameters.
            results (dict[str, Any]): Segment-local results keyed by value
                UUID.
            scoped_locals (dict[str, Any]): Loop-scoped local variables.

        Returns:
            Any: The concrete element value.

        Raises:
            ExecutionError: If ``value`` has no parent array or the element
                cannot be resolved through any lookup path.
        """
        parent = value.parent_array
        if parent is None:
            raise ExecutionError(f"Value {value.name} is not an array element")

        indices = tuple(
            int(self._get_value(idx, context, results, scoped_locals))
            for idx in value.element_indices
        )
        container = self._get_array_data(parent, context, results, scoped_locals)

        if container is not None:
            if len(indices) == 1:
                return container[indices[0]]
            return container[indices]

        if len(indices) == 1:
            composite_key = f"{parent.uuid}_{indices[0]}"
            if composite_key in results:
                return results[composite_key]
            if context.has(composite_key):
                return context.get(composite_key)
            indexed_key = f"{parent.name}[{indices[0]}]"
            if context.has(indexed_key):
                return context.get(indexed_key)
            # Measurement results are loaded into the context under
            # per-element composite carrier keys ("<root_uuid>_<index>").
            # Compose slice views back onto the root so a view-local
            # index addresses the right physical slot.
            resolved = resolve_root_array_index(parent, indices[0])
            if resolved is not None:
                root_array, root_index = resolved
                composite_key = f"{root_array.uuid}_{root_index}"
                if context.has(composite_key):
                    return context.get(composite_key)

        resolved_location = resolve_runtime_array_location(
            parent,
            indices,
            lambda v: self._get_optional_int_value(
                v,
                context,
                results,
                scoped_locals,
            ),
        )
        if resolved_location is not None:
            root, root_indices = resolved_location
            root_container = self._get_array_data(root, context, results, scoped_locals)
            if root_container is not None:
                if len(root_indices) == 1:
                    return root_container[root_indices[0]]
                return root_container[root_indices]

            if len(root_indices) == 1:
                root_key = f"{root.uuid}_{root_indices[0]}"
                if root_key in results:
                    return results[root_key]
                if context.has(root_key):
                    return context.get(root_key)
                root_indexed_key = f"{root.name}[{root_indices[0]}]"
                if context.has(root_indexed_key):
                    return context.get(root_indexed_key)

        raise ExecutionError(
            f"Array element {parent.name}{indices} could not be resolved"
        )

    def _get_optional_int_value(
        self,
        value: Value,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> int | None:
        """Resolve ``value`` to ``int`` when available.

        Args:
            value (Value): Scalar value to resolve.
            context (ExecutionContext): Execution context holding measured
                values and bindings.
            results (dict[str, Any]): Segment-local results.
            scoped_locals (dict[str, Any]): Loop/branch-local values.

        Returns:
            int | None: Integer value, or ``None`` when the value is not
                currently resolvable.
        """
        try:
            return int(self._get_value(value, context, results, scoped_locals))
        except ExecutionError:
            return None

    def _get_array_data(
        self,
        array_value: ArrayValue,
        context: ExecutionContext,
        results: dict[str, Any],
        scoped_locals: dict[str, Any],
    ) -> Any:
        """Resolve an array-like container from execution state."""
        if array_value.uuid in results:
            return results[array_value.uuid]
        if context.has(array_value.uuid):
            return context.get(array_value.uuid)
        if array_value.name in scoped_locals:
            return scoped_locals[array_value.name]
        if context.has(array_value.name):
            return context.get(array_value.name)
        const_array = array_value.get_const_array()
        if const_array is not None:
            return const_array
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
        """Resolve a DictValue to concrete key/value pairs."""
        if isinstance(iterable_value, DictValue):
            bound_items = iterable_value.get_bound_data_items()
            if bound_items:
                return list(bound_items)
            if context.has(iterable_value.uuid):
                return list(context.get(iterable_value.uuid).items())
            if iterable_value.name in scoped_locals:
                return list(scoped_locals[iterable_value.name].items())
            if context.has(iterable_value.name):
                return list(context.get(iterable_value.name).items())
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
        op: ForItemsOperation,
        key: Any,
    ) -> None:
        """Bind for-items key variables into the loop scope."""
        if len(op.key_vars) == 1:
            loop_scope[op.key_vars[0]] = key
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

        for name, element in zip(op.key_vars, key, strict=True):
            loop_scope[name] = element
