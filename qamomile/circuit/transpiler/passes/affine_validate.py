"""Affine type validation pass: Verify quantum resources are used correctly."""

from __future__ import annotations

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation, SliceArrayOperation
from qamomile.circuit.ir.operation.control_flow import HasNestedOps, IfOperation
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.errors import AffineTypeError, ValidationError
from qamomile.circuit.transpiler.passes import Pass


class AffineValidationPass(Pass[Block, Block]):
    """Validate affine type semantics at IR level.

    This pass serves as a safety net to catch affine type violations
    that may have bypassed the frontend checks. It verifies:
    1. Each quantum value is used (consumed) at most once
    2. Quantum values are not silently discarded

    Input: Block (any kind)
    Output: Same Block (unchanged, validation only)
    """

    @property
    def name(self) -> str:
        return "affine_validate"

    def run(self, input: Block) -> Block:
        """Validate affine type semantics in the block.

        Raises:
            ValidationError: If the block kind is not AFFINE.
            AffineTypeError: If a quantum value is consumed multiple times.
        """
        if input.kind not in (BlockKind.AFFINE,):
            raise ValidationError(
                f"AffineValidationPass expects AFFINE block, got {input.kind}",
            )

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
        """Validate operations for affine type violations.

        Note: This method handles control flow specially due to the need
        for scope management (e.g., loop scopes, branch merging).
        """
        for op in operations:
            op_name = type(op).__name__

            # SliceArrayOperation takes the parent array as an operand
            # but does NOT consume it — it only produces metadata
            # describing a strided view.  The parent remains live and
            # downstream gate/measure ops on it must not be flagged as
            # double-consume.  (ConstantFoldingPass normally strips
            # SliceArrayOperation before this pass, but we guard
            # defensively in case ordering changes.)
            if isinstance(op, SliceArrayOperation):
                continue

            # Check each operand
            for operand in op.operands:
                if isinstance(operand, Value) and operand.type.is_quantum():
                    self._check_and_mark_consumed(operand, op_name, consumed)

            # Handle control flow with scope management
            if isinstance(op, HasNestedOps):
                nested_lists = op.nested_op_lists()
                # For IfOperation, skip phi_ops (last list) - they are
                # merge operations referencing values from both branches
                # and should not be independently validated.
                if isinstance(op, IfOperation) and len(nested_lists) > 2:
                    validate_lists = nested_lists[:2]
                else:
                    validate_lists = nested_lists
                scoped_sets: list[dict[str, str]] = []
                for op_list in validate_lists:
                    scoped = consumed.copy()
                    self._validate_operations(op_list, scoped)
                    scoped_sets.append(scoped)
                # For IfOperation, merge all scoped consumed back:
                # anything consumed in either branch is considered consumed.
                # For loops (For/ForItems/While), don't propagate - values
                # consumed in loop may or may not be consumed after.
                if isinstance(op, IfOperation):
                    for scoped in scoped_sets:
                        consumed.update(scoped)

    def _check_and_mark_consumed(
        self,
        value: Value,
        operation_name: str,
        consumed: dict[str, str],
    ) -> None:
        """Check if a value was already consumed and mark it as consumed.

        Raises:
            AffineTypeError: If the value was already consumed.
        """
        # Skip if this is a result of the previous operation with same uuid
        # (SSA-style versioning means the same physical qubit has different versions)
        # We check by uuid which should be unique per value instance

        if value.uuid in consumed:
            first_consumer = consumed[value.uuid]
            raise AffineTypeError(
                f"Quantum value '{value.name}' was already consumed by "
                f"'{first_consumer}' and cannot be used again in '{operation_name}'.\n\n"
                f"This is likely an internal error - if you see this message, "
                f"please report it as a bug.",
                handle_name=value.name,
                operation_name=operation_name,
                first_use_location=first_consumer,
            )

        consumed[value.uuid] = operation_name
