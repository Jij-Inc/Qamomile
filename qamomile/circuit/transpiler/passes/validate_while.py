"""WhileOperation contract validation pass.

Validates that all ``WhileOperation`` conditions are measurement-backed
``Bit`` values.  Only ``while bit:`` where ``bit`` originates from
``qmc.measure()`` is supported.  All other while patterns (classical
variables, constants, comparison results) are rejected with a clear
``ValidationError`` before reaching backend-specific emit passes.

This pass runs after ``lower_compile_time_ifs`` and before ``analyze``.
"""

from __future__ import annotations

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import MeasureOperation
from qamomile.circuit.ir.types.primitives import BitType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.errors import ValidationError

from . import Pass


class ValidateWhileContractPass(Pass[Block, Block]):
    """Validates that all WhileOperation conditions are measurement-backed.

    Builds a producer map (result UUID → producing Operation) and checks
    every WhileOperation operand against it.  A valid condition must be:

    1. A ``Value`` with ``BitType``
    2. Produced by a ``MeasureOperation`` (UUID appears in producer map)

    Raises ``ValidationError`` for any non-measurement while pattern.
    """

    @property
    def name(self) -> str:
        return "validate_while_contract"

    def run(self, block: Block) -> Block:
        """Validate all WhileOperations and return block unchanged."""
        producer_map: dict[str, type[Operation]] = {}
        self._build_producer_map(block.operations, producer_map)
        self._validate_whiles(block.operations, producer_map)
        return block

    def _build_producer_map(
        self,
        operations: list[Operation],
        producer_map: dict[str, type[Operation]],
    ) -> None:
        """Walk operations recursively, mapping result UUIDs to producer types."""
        for op in operations:
            for result in op.results:
                if isinstance(result, Value):
                    producer_map[result.uuid] = type(op)

            # Recurse into control flow bodies
            if isinstance(op, WhileOperation):
                self._build_producer_map(op.operations, producer_map)
            elif isinstance(op, ForOperation):
                self._build_producer_map(op.operations, producer_map)
            elif isinstance(op, ForItemsOperation):
                self._build_producer_map(op.operations, producer_map)
            elif isinstance(op, IfOperation):
                self._build_producer_map(op.true_operations, producer_map)
                self._build_producer_map(op.false_operations, producer_map)

    def _validate_whiles(
        self,
        operations: list[Operation],
        producer_map: dict[str, type[Operation]],
    ) -> None:
        """Find all WhileOperations and validate their conditions."""
        for op in operations:
            if isinstance(op, WhileOperation):
                self._validate_while_condition(op, producer_map)
                # Also recurse into while body
                self._validate_whiles(op.operations, producer_map)
            elif isinstance(op, ForOperation):
                self._validate_whiles(op.operations, producer_map)
            elif isinstance(op, ForItemsOperation):
                self._validate_whiles(op.operations, producer_map)
            elif isinstance(op, IfOperation):
                self._validate_whiles(op.true_operations, producer_map)
                self._validate_whiles(op.false_operations, producer_map)

    def _validate_while_condition(
        self,
        op: WhileOperation,
        producer_map: dict[str, type[Operation]],
    ) -> None:
        """Validate that a WhileOperation's initial condition is measurement-backed.

        Only ``operands[0]`` (the initial condition) is checked.
        ``operands[1]`` (loop-carried condition) may come from ``IfOperation``
        or ``PhiOp`` when there is branching inside the while body, so it is
        not required to be directly produced by ``MeasureOperation``.
        """
        if op.operands:
            self._check_operand(op.operands[0], "condition", producer_map)

    @staticmethod
    def _unwrap_value(operand: object) -> Value | None:
        """Extract the IR Value from an operand (Handle or raw Value)."""
        if isinstance(operand, Value):
            return operand
        # Frontend Handle objects (Bit, UInt, etc.) have a .value attribute
        val = getattr(operand, "value", None)
        if isinstance(val, Value):
            return val
        return None

    def _check_operand(
        self,
        operand: object,
        label: str,
        producer_map: dict[str, type[Operation]],
    ) -> None:
        """Check a single operand is a measurement-backed Bit."""
        value = self._unwrap_value(operand)
        if value is None:
            raise ValidationError(
                f"WhileOperation {label} must be a measurement result (Bit), "
                f"but got unsupported operand {type(operand).__name__}. "
                f"Only 'while bit:' where 'bit' comes from qmc.measure() is supported."
            )

        if not isinstance(value.type, BitType):
            type_name = type(value.type).__name__
            raise ValidationError(
                f"WhileOperation {label} must be a measurement result (Bit), "
                f"but got Value of type {type_name}. "
                f"Only 'while bit:' where 'bit' comes from qmc.measure() is supported.",
                value_name=value.name,
            )

        producer = producer_map.get(value.uuid)
        if producer is not MeasureOperation:
            if producer is None:
                detail = "not produced by any operation in this kernel"
            else:
                detail = f"produced by {producer.__name__}"
            raise ValidationError(
                f"WhileOperation {label} must be a measurement result, "
                f"but Bit '{value.name}' is {detail}. "
                f"Only 'while bit:' where 'bit' comes from qmc.measure() is supported.",
                value_name=value.name,
            )
