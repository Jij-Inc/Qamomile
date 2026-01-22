"""Classical segment executor for Python-based classical operations."""

from __future__ import annotations

from typing import Any

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    CompOp,
    CompOpKind,
    NotOp,
    CondOp,
    CondOpKind,
)
from qamomile.circuit.ir.operation.classical_ops import DecodeQFixedOperation
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.segments import ClassicalSegment
from qamomile.circuit.transpiler.execution_context import ExecutionContext
from qamomile.circuit.transpiler.errors import ExecutionError


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

        for op in segment.operations:
            self._execute_operation(op, context, results)

        return results

    def _execute_operation(
        self,
        op: Operation,
        context: ExecutionContext,
        results: dict[str, Any],
    ) -> None:
        """Execute a single operation."""
        if isinstance(op, BinOp):
            self._execute_binop(op, context, results)
        elif isinstance(op, CompOp):
            self._execute_compop(op, context, results)
        elif isinstance(op, NotOp):
            self._execute_notop(op, context, results)
        elif isinstance(op, CondOp):
            self._execute_condop(op, context, results)
        elif isinstance(op, DecodeQFixedOperation):
            self._execute_decode_qfixed(op, context, results)

    def _execute_binop(
        self,
        op: BinOp,
        context: ExecutionContext,
        results: dict[str, Any],
    ) -> None:
        """Execute binary operation."""
        lhs = self._get_value(op.operands[0], context, results)
        rhs = self._get_value(op.operands[1], context, results)

        match op.kind:
            case BinOpKind.ADD:
                result_value = lhs + rhs
            case BinOpKind.SUB:
                result_value = lhs - rhs
            case BinOpKind.MUL:
                result_value = lhs * rhs
            case BinOpKind.DIV:
                result_value = lhs / rhs
            case BinOpKind.FLOORDIV:
                result_value = lhs // rhs
            case BinOpKind.POW:
                result_value = lhs**rhs
            case _:
                raise ExecutionError(f"Unknown BinOp kind: {op.kind}")

        if op.results:
            results[op.results[0].uuid] = result_value

    def _execute_compop(
        self,
        op: CompOp,
        context: ExecutionContext,
        results: dict[str, Any],
    ) -> None:
        """Execute comparison operation."""
        lhs = self._get_value(op.operands[0], context, results)
        rhs = self._get_value(op.operands[1], context, results)

        match op.kind:
            case CompOpKind.EQ:
                result_value = lhs == rhs
            case CompOpKind.NEQ:
                result_value = lhs != rhs
            case CompOpKind.LT:
                result_value = lhs < rhs
            case CompOpKind.LE:
                result_value = lhs <= rhs
            case CompOpKind.GT:
                result_value = lhs > rhs
            case CompOpKind.GE:
                result_value = lhs >= rhs
            case _:
                raise ExecutionError(f"Unknown CompOp kind: {op.kind}")

        if op.results:
            results[op.results[0].uuid] = result_value

    def _execute_notop(
        self,
        op: NotOp,
        context: ExecutionContext,
        results: dict[str, Any],
    ) -> None:
        """Execute not operation."""
        operand = self._get_value(op.operands[0], context, results)
        result_value = not operand

        if op.results:
            results[op.results[0].uuid] = result_value

    def _execute_condop(
        self,
        op: CondOp,
        context: ExecutionContext,
        results: dict[str, Any],
    ) -> None:
        """Execute conditional operation (AND, OR)."""
        lhs = self._get_value(op.operands[0], context, results)
        rhs = self._get_value(op.operands[1], context, results)

        match op.kind:
            case CondOpKind.AND:
                result_value = lhs and rhs
            case CondOpKind.OR:
                result_value = lhs or rhs
            case _:
                raise ExecutionError(f"Unknown CondOp kind: {op.kind}")

        if op.results:
            results[op.results[0].uuid] = result_value

    def _execute_decode_qfixed(
        self,
        op: DecodeQFixedOperation,
        context: ExecutionContext,
        results: dict[str, Any],
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
            bits = [self._get_value(b, context, results) for b in op.operands]

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

    def _get_value(
        self,
        value: Value,
        context: ExecutionContext,
        results: dict[str, Any],
    ) -> Any:
        """Get the concrete value from context or results."""
        if value.uuid in results:
            return results[value.uuid]
        if context.has(value.uuid):
            return context.get(value.uuid)
        # Check if it's a constant
        if value.is_constant():
            return value.get_const()
        raise ExecutionError(f"Value {value.name} not found in context or results")
