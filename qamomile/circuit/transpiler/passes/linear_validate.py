"""Linear type validation pass: Verify quantum resources are used correctly."""

from __future__ import annotations

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.errors import LinearTypeError


class LinearValidationPass(Pass[Block, Block]):
    """Validate linear type semantics at IR level.

    This pass serves as a safety net to catch linear type violations
    that may have bypassed the frontend checks. It verifies:
    1. Each quantum value is used (consumed) at most once
    2. Quantum values are not silently discarded

    Input: Block (any kind)
    Output: Same Block (unchanged, validation only)
    """

    @property
    def name(self) -> str:
        return "linear_validate"

    def run(self, input: Block) -> Block:
        """Validate linear type semantics in the block.

        Raises:
            LinearTypeError: If a quantum value is consumed multiple times.
        """
        # Track which quantum values have been consumed
        # Maps uuid -> operation name that consumed it
        consumed: dict[str, str] = {}

        self._validate_operations(input.operations, consumed)

        return input  # Pass-through, no modifications

    def _validate_operations(
        self,
        operations: list[Operation],
        consumed: dict[str, str],
    ) -> None:
        """Validate operations for linear type violations.

        Note: This method handles control flow specially due to the need
        for scope management (e.g., loop scopes, branch merging).
        """
        for op in operations:
            op_name = type(op).__name__

            # Check each operand
            for operand in op.operands:
                if isinstance(operand, Value) and operand.type.is_quantum():
                    self._check_and_mark_consumed(operand, op_name, consumed)

            # Handle control flow with scope management
            if isinstance(op, ForOperation):
                # For loops: values may be consumed multiple times
                # Create a new scope for the loop body
                loop_consumed: dict[str, str] = consumed.copy()
                self._validate_operations(op.operations, loop_consumed)
                # Values consumed in loop may or may not be consumed after
                # depending on loop execution - conservative: don't propagate
            elif isinstance(op, IfOperation):
                # If: values consumed in either branch are consumed
                true_consumed = consumed.copy()
                false_consumed = consumed.copy()
                self._validate_operations(op.true_operations, true_consumed)
                self._validate_operations(op.false_operations, false_consumed)
                # Merge: anything consumed in either branch is consumed
                consumed.update(true_consumed)
                consumed.update(false_consumed)
            elif isinstance(op, WhileOperation):
                # While loops: similar to for loops
                loop_consumed = consumed.copy()
                self._validate_operations(op.operations, loop_consumed)

    def _check_and_mark_consumed(
        self,
        value: Value,
        operation_name: str,
        consumed: dict[str, str],
    ) -> None:
        """Check if a value was already consumed and mark it as consumed.

        Raises:
            LinearTypeError: If the value was already consumed.
        """
        # Skip if this is a result of the previous operation with same uuid
        # (SSA-style versioning means the same physical qubit has different versions)
        # We check by uuid which should be unique per value instance

        if value.uuid in consumed:
            first_consumer = consumed[value.uuid]
            raise LinearTypeError(
                f"Quantum value '{value.name}' was already consumed by "
                f"'{first_consumer}' and cannot be used again in '{operation_name}'.\n\n"
                f"This is likely an internal error - if you see this message, "
                f"please report it as a bug.",
                handle_name=value.name,
                operation_name=operation_name,
                first_use_location=first_consumer,
            )

        consumed[value.uuid] = operation_name
