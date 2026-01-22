"""Constant folding pass for compile-time expression evaluation."""

from __future__ import annotations

import dataclasses
from typing import Any

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.ir.value import Value

from . import Pass
from .control_flow_visitor import OperationTransformer


class ConstantFoldingPass(Pass[Block, Block]):
    """Evaluates constant expressions at compile time.

    This pass folds BinOp operations when all operands are constants
    or bound parameters, eliminating unnecessary classical operations
    that would otherwise split quantum segments.

    Example:
        Before (with bindings={"phase": 0.5}):
            BinOp(phase * 2) -> classical segment split

        After:
            Constant 1.0 -> no segment split
    """

    def __init__(self, bindings: dict[str, Any] | None = None):
        self._bindings = bindings or {}

    @property
    def name(self) -> str:
        return "constant_fold"

    def run(self, input: Block) -> Block:
        """Run constant folding on the block."""
        # Track folded values: uuid -> constant Value
        folded_values: dict[str, Value] = {}

        # Process operations
        new_ops = self._fold_operations(input.operations, folded_values)

        return dataclasses.replace(input, operations=new_ops)

    def _fold_operations(
        self,
        operations: list[Operation],
        folded_values: dict[str, Value],
    ) -> list[Operation]:
        """Process operations, folding constant BinOps."""
        outer_self = self

        class ConstantFoldingTransformer(OperationTransformer):
            def transform_operation(self, op: Operation) -> Operation | None:
                if isinstance(op, BinOp):
                    folded = outer_self._try_fold_binop(op, folded_values)
                    if folded is not None:
                        # BinOp was folded to a constant - remove it
                        # Just record the mapping for later substitution
                        folded_values[op.results[0].uuid] = folded
                        return None

                # Substitute folded values in operands
                return outer_self._substitute_folded_operands(op, folded_values)

        transformer = ConstantFoldingTransformer()
        return transformer.transform_operations(operations)

    def _try_fold_binop(
        self,
        op: BinOp,
        folded_values: dict[str, Value],
    ) -> Value | None:
        """Try to fold a BinOp to a constant. Returns None if not foldable."""
        if len(op.operands) != 2:
            return None

        left = self._resolve_value(op.operands[0], folded_values)
        right = self._resolve_value(op.operands[1], folded_values)

        if left is None or right is None:
            return None

        # Both operands are constants, evaluate
        result_value = self._evaluate_binop(op.kind, left, right)
        if result_value is None:
            return None

        # Create constant Value with same uuid for substitution
        result_type = op.results[0].type
        return Value(
            type=result_type,
            name=f"folded_{op.results[0].name}",
            params={"const": result_value},
            uuid=op.results[0].uuid,  # Keep same UUID for substitution
        )

    def _resolve_value(
        self,
        value: Value,
        folded_values: dict[str, Value],
    ) -> float | int | None:
        """Resolve a Value to a constant, or None if not resolvable."""
        # Check if already folded
        if value.uuid in folded_values:
            folded = folded_values[value.uuid]
            return folded.get_const()

        # Check if constant
        if value.is_constant():
            return value.get_const()

        # Check if bound parameter
        if value.is_parameter():
            param_name = value.parameter_name()
            if param_name and param_name in self._bindings:
                return self._bindings[param_name]

        return None

    def _evaluate_binop(
        self,
        kind: BinOpKind | None,
        left: float | int,
        right: float | int,
    ) -> float | int | None:
        """Evaluate a binary operation on two constants."""
        if kind is None:
            return None

        try:
            match kind:
                case BinOpKind.ADD:
                    return left + right
                case BinOpKind.SUB:
                    return left - right
                case BinOpKind.MUL:
                    return left * right
                case BinOpKind.DIV:
                    return left / right if right != 0 else None
                case BinOpKind.FLOORDIV:
                    return left // right if right != 0 else None
                case BinOpKind.POW:
                    return left**right
                case _:
                    return None
        except (TypeError, ValueError, OverflowError):
            return None

    def _substitute_folded_operands(
        self,
        op: Operation,
        folded_values: dict[str, Value],
    ) -> Operation:
        """Substitute folded constant values in operation operands."""
        new_operands: list[Any] = []
        changed = False

        for operand in op.operands:
            if isinstance(operand, Value) and operand.uuid in folded_values:
                new_operands.append(folded_values[operand.uuid])
                changed = True
            else:
                new_operands.append(operand)

        if changed:
            return dataclasses.replace(op, operands=new_operands)
        return op
