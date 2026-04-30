"""Classical segment executor for Python-based classical operations."""

from __future__ import annotations

from typing import Any

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    CompOp,
    CondOp,
    NotOp,
    PhiOp,
)
from qamomile.circuit.ir.operation.classical_ops import DecodeQFixedOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.value import ArrayValue, DictValue, TupleValue, Value
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
        elif isinstance(op, PhiOp):
            self._execute_phi(op, context, results, scoped_locals)
        elif isinstance(op, DecodeQFixedOperation):
            self._execute_decode_qfixed(op, context, results, scoped_locals)
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
        """Resolve an array element from its parent container."""
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
            indexed_key = f"{parent.name}[{indices[0]}]"
            if context.has(indexed_key):
                return context.get(indexed_key)

        raise ExecutionError(
            f"Array element {parent.name}{indices} could not be resolved"
        )

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
