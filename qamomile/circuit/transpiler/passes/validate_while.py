"""WhileOperation contract validation pass.

Validates that all ``WhileOperation`` conditions are measurement-backed
``Bit`` values.  A condition is measurement-backed if it originates from
``qmc.measure()`` either directly or through phi-merged branches where
every leaf is itself measurement-backed (e.g. ``if sel: bit = measure(q1)
else: bit = measure(q2)``).

All other while patterns (classical variables, constants, comparison
results, non-measurement branch leaves) are rejected with a clear
``ValidationError`` before reaching backend-specific emit passes.

This pass runs after ``lower_compile_time_ifs`` and before ``analyze``.
"""

from __future__ import annotations

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import PhiOp
from qamomile.circuit.ir.operation.control_flow import (
    HasNestedOps,
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

    Builds a producer map (result UUID → producing Operation instance) and
    checks every WhileOperation operand against it.  A valid condition must be:

    1. A ``Value`` with ``BitType``
    2. Measurement-backed: produced by ``MeasureOperation`` directly, or by
       ``IfOperation`` / ``PhiOp`` where every reachable leaf source is
       itself measurement-backed.

    Both ``operands[0]`` (initial condition) and ``operands[1]``
    (loop-carried condition) are validated.

    Raises ``ValidationError`` for any non-measurement while pattern.
    """

    @property
    def name(self) -> str:
        return "validate_while_contract"

    def run(self, block: Block) -> Block:
        """Validate all WhileOperations and return block unchanged."""
        producer_map: dict[str, Operation] = {}
        self._build_producer_map(block.operations, producer_map)
        self._validate_whiles(block.operations, producer_map)
        return block

    def _build_producer_map(
        self,
        operations: list[Operation],
        producer_map: dict[str, Operation],
    ) -> None:
        """Walk operations recursively, mapping result UUIDs to producer instances."""
        for op in operations:
            for result in op.results:
                if isinstance(result, Value):
                    producer_map[result.uuid] = op

            # Recurse into control flow bodies
            if isinstance(op, HasNestedOps):
                # IfOperation phi_ops results need their own producer entries
                if isinstance(op, IfOperation):
                    for phi in op.phi_ops:
                        for result in phi.results:
                            if isinstance(result, Value):
                                producer_map[result.uuid] = phi
                for op_list in op.nested_op_lists():
                    self._build_producer_map(op_list, producer_map)

    def _validate_whiles(
        self,
        operations: list[Operation],
        producer_map: dict[str, Operation],
    ) -> None:
        """Find all WhileOperations and validate their conditions."""
        for op in operations:
            if isinstance(op, WhileOperation):
                self._validate_while_condition(op, producer_map)
            if isinstance(op, HasNestedOps):
                for op_list in op.nested_op_lists():
                    self._validate_whiles(op_list, producer_map)

    def _validate_while_condition(
        self,
        op: WhileOperation,
        producer_map: dict[str, Operation],
    ) -> None:
        """Validate that a WhileOperation's conditions are measurement-backed.

        Both ``operands[0]`` (initial condition) and ``operands[1]``
        (loop-carried condition) are checked.  A condition is
        measurement-backed if it traces back to ``MeasureOperation``
        through any chain of ``IfOperation`` / ``PhiOp`` merges.
        """
        if op.operands:
            self._check_operand(op.operands[0], "condition", producer_map)
        if len(op.operands) > 1:
            self._check_operand(op.operands[1], "loop-carried condition", producer_map)

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

    def _is_measurement_backed(
        self,
        value: Value,
        producer_map: dict[str, Operation],
        visiting: set[str] | None = None,
    ) -> bool:
        """Recursively check whether a value traces back to measurement operations.

        Returns ``True`` if the value is produced by ``MeasureOperation``
        directly, or by ``IfOperation`` / ``PhiOp`` where every reachable
        leaf source is itself measurement-backed.

        Uses backtracking on ``visiting`` so that the same value can be
        reached from multiple independent phi branches without false negatives.
        """
        if visiting is None:
            visiting = set()
        if value.uuid in visiting:
            return False
        visiting.add(value.uuid)

        producer = producer_map.get(value.uuid)
        if producer is None:
            visiting.discard(value.uuid)
            return False

        if isinstance(producer, MeasureOperation):
            visiting.discard(value.uuid)
            return True

        if isinstance(producer, IfOperation):
            # Find the PhiOp whose output matches this value
            for phi in producer.phi_ops:
                if phi.results and phi.results[0].uuid == value.uuid:
                    result = self._is_measurement_backed(
                        phi.true_value, producer_map, visiting
                    ) and self._is_measurement_backed(
                        phi.false_value, producer_map, visiting
                    )
                    visiting.discard(value.uuid)
                    return result
            visiting.discard(value.uuid)
            return False

        if isinstance(producer, PhiOp):
            result = self._is_measurement_backed(
                producer.true_value, producer_map, visiting
            ) and self._is_measurement_backed(
                producer.false_value, producer_map, visiting
            )
            visiting.discard(value.uuid)
            return result

        visiting.discard(value.uuid)
        return False

    def _check_operand(
        self,
        operand: object,
        label: str,
        producer_map: dict[str, Operation],
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

        if not self._is_measurement_backed(value, producer_map):
            producer = producer_map.get(value.uuid)
            if producer is None:
                detail = "not produced by any operation in this kernel"
            else:
                detail = f"produced by {type(producer).__name__}"
            raise ValidationError(
                f"WhileOperation {label} must be a measurement result, "
                f"but Bit '{value.name}' is {detail}. "
                f"Only 'while bit:' where 'bit' comes from qmc.measure() is supported.",
                value_name=value.name,
            )
