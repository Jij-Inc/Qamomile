"""WhileOperation contract validation pass.

Validates that all ``WhileOperation`` conditions are measurement-backed
``Bit`` values.  A condition is measurement-backed if it originates from
``qmc.measure()`` either directly or through merged branches where
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
from qamomile.circuit.ir.operation.arithmetic_operations import (
    RuntimeClassicalExpr,
    RuntimeOpKind,
)
from qamomile.circuit.ir.operation.control_flow import (
    HasNestedOps,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    MeasureOperation,
    MeasureVectorOperation,
)
from qamomile.circuit.ir.types.primitives import BitType
from qamomile.circuit.ir.value import Value, resolve_root_array_index
from qamomile.circuit.transpiler.errors import ValidationError

from . import Pass


def build_producer_map(
    operations: list[Operation],
    producer_map: dict[str, Operation],
) -> None:
    """Walk operations recursively, mapping result UUIDs to producer instances.

    Exposed at module scope so measurement-backing checks outside this
    pass (e.g. the loop-carried rebind check in ``analyze``) can build
    the same map.

    Args:
        operations (list[Operation]): Operations to walk, recursing into
            all ``HasNestedOps`` bodies. ``IfOperation`` merge outputs
            are on ``results``, so they map to the ``IfOperation``
            itself.
        producer_map (dict[str, Operation]): Mutable map from result UUID
            to the producing operation; updated in place.
    """
    for op in operations:
        for result in op.results:
            if isinstance(result, Value):
                producer_map[result.uuid] = op

        # Recurse into control flow bodies.
        if isinstance(op, HasNestedOps):
            for op_list in op.nested_op_lists():
                build_producer_map(op_list, producer_map)


def is_measurement_backed(
    value: Value,
    producer_map: dict[str, Operation],
    visiting: set[str] | None = None,
) -> bool:
    """Recursively check whether a value traces back to measurement operations.

    Returns ``True`` if the value is produced by ``MeasureOperation`` /
    ``MeasureVectorOperation`` directly, if it is an element access of a
    measured ``Vector[Bit]`` (its ``parent_array`` — possibly through a
    ``slice_of`` view chain — is measurement-backed), or if it is an
    ``IfOperation`` merge output where every reachable leaf source is
    itself measurement-backed.

    Uses backtracking on ``visiting`` so that the same value can be
    reached from multiple independent merge branches without false negatives.

    Args:
        value (Value): The IR value to trace.
        producer_map (dict[str, Operation]): Map from result UUID to the
            operation that produced it.
        visiting (set[str] | None): UUIDs on the current DFS path, used to
            break cycles. Defaults to None (fresh traversal).

    Returns:
        bool: True if the value is measurement-backed.
    """
    if visiting is None:
        visiting = set()
    if value.uuid in visiting:
        return False
    visiting.add(value.uuid)

    producer = producer_map.get(value.uuid)

    if isinstance(producer, (MeasureOperation, MeasureVectorOperation)):
        visiting.discard(value.uuid)
        return True

    # Element access of a measured Vector[Bit]: ``s[i]`` where
    # ``s = qmc.measure(register)``. The element carries its own UUID
    # (not produced by any op) and points to the measured array via
    # ``parent_array``; a sliced view (``s[a:b:c][i]``) reaches the root
    # measured array through the parent's ``slice_of`` chain.
    parent = getattr(value, "parent_array", None)
    if parent is not None:
        cur: Value | None = parent
        while cur is not None:
            if is_measurement_backed(cur, producer_map, visiting):
                visiting.discard(value.uuid)
                return True
            cur = getattr(cur, "slice_of", None)

    if producer is None:
        visiting.discard(value.uuid)
        return False

    if isinstance(producer, IfOperation):
        # Find the merge whose output matches this value
        for merge in producer.iter_merges():
            if merge.result.uuid == value.uuid:
                result = is_measurement_backed(
                    merge.true_value, producer_map, visiting
                ) and is_measurement_backed(merge.false_value, producer_map, visiting)
                visiting.discard(value.uuid)
                return result
        visiting.discard(value.uuid)
        return False

    if (
        isinstance(producer, RuntimeClassicalExpr)
        and producer.kind is RuntimeOpKind.SELECT
    ):
        # ``ClassicalLoweringPass`` lowers a runtime-if merge to
        # ``select(cond, t, f)``; like the merge it lowers, the value is
        # measurement-backed when both branch sources are.
        result = is_measurement_backed(
            producer.operands[1], producer_map, visiting
        ) and is_measurement_backed(producer.operands[2], producer_map, visiting)
        visiting.discard(value.uuid)
        return result

    visiting.discard(value.uuid)
    return False


class ValidateWhileContractPass(Pass[Block, Block]):
    """Validates that all WhileOperation conditions are measurement-backed.

    Builds a producer map (result UUID → producing Operation instance) and
    checks every WhileOperation operand against it.  A valid condition must be:

    1. A ``Value`` with ``BitType``
    2. Measurement-backed: produced by ``MeasureOperation`` directly, or an
       ``IfOperation`` merge output where every reachable leaf source is
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
        """Walk operations recursively, mapping result UUIDs to producer instances.

        Thin wrapper for the module-level :func:`build_producer_map`.

        Args:
            operations (list[Operation]): Operations to walk recursively.
            producer_map (dict[str, Operation]): Mutable UUID-to-producer
                map, updated in place.
        """
        build_producer_map(operations, producer_map)

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
        through any chain of ``IfOperation`` merges.
        """
        if op.operands:
            self._check_operand(op.operands[0], "condition", producer_map)
        if len(op.operands) > 1:
            self._check_operand(op.operands[1], "loop-carried condition", producer_map)
            initial = self._unwrap_value(op.operands[0])
            carried = self._unwrap_value(op.operands[1])
            if initial is not None and carried is not None:
                body_operation_ids = self._collect_operation_ids(op.operations)
                if not self._is_loop_local_update(
                    carried,
                    initial,
                    body_operation_ids,
                    producer_map,
                ):
                    raise ValidationError(
                        "WhileOperation loop-carried Bit conditions must be "
                        "updated by measurements produced inside the loop "
                        "body, or preserve the current condition value. "
                        "Reusing a different measurement taken before the "
                        "loop cannot safely update the backend condition.",
                        value_name=carried.name,
                    )

    @staticmethod
    def _collect_operation_ids(operations: list[Operation]) -> set[int]:
        """Collect operation identities from a nested operation tree.

        Args:
            operations (list[Operation]): Operations to scan recursively.

        Returns:
            set[int]: Python identities of all operations in the tree.
        """
        operation_ids: set[int] = set()
        for operation in operations:
            operation_ids.add(id(operation))
            if isinstance(operation, HasNestedOps):
                for body in operation.nested_op_lists():
                    operation_ids.update(
                        ValidateWhileContractPass._collect_operation_ids(body)
                    )
        return operation_ids

    @staticmethod
    def _same_condition_location(left: Value, right: Value) -> bool:
        """Check whether two values address the same measured bit location.

        Args:
            left (Value): First condition value.
            right (Value): Second condition value.

        Returns:
            bool: True for the same scalar UUID or the same constant-indexed
                element of a shared measured vector, including slice views.
        """
        if left.uuid == right.uuid:
            return True
        if (
            left.parent_array is None
            or right.parent_array is None
            or len(left.element_indices) != 1
            or len(right.element_indices) != 1
        ):
            return False
        left_index = left.element_indices[0]
        right_index = right.element_indices[0]
        if not left_index.is_constant() or not right_index.is_constant():
            return False
        left_location = resolve_root_array_index(
            left.parent_array, int(left_index.get_const())
        )
        right_location = resolve_root_array_index(
            right.parent_array, int(right_index.get_const())
        )
        if left_location is None or right_location is None:
            return False
        left_root, left_root_index = left_location
        right_root, right_root_index = right_location
        return left_root.uuid == right_root.uuid and left_root_index == right_root_index

    def _is_loop_local_update(
        self,
        value: Value,
        initial: Value,
        body_operation_ids: set[int],
        producer_map: dict[str, Operation],
        visiting: set[str] | None = None,
    ) -> bool:
        """Check that a carried condition is updated within its while body.

        The current condition may pass through an identity branch unchanged.
        Every other reachable leaf must be a measurement produced in the
        loop body. This prevents the allocator from aliasing unrelated
        pre-loop measurement clbits onto the mutable while-condition clbit.

        Args:
            value (Value): Candidate carried condition value.
            initial (Value): Initial while-condition value.
            body_operation_ids (set[int]): Identities of operations inside
                the while body.
            producer_map (dict[str, Operation]): Global UUID-to-producer map.
            visiting (set[str] | None): UUIDs on the current recursion path.
                Defaults to None.

        Returns:
            bool: True when every update path is loop-local or preserves the
                initial condition.
        """
        if self._same_condition_location(value, initial):
            return True
        if visiting is None:
            visiting = set()
        if value.uuid in visiting:
            return False
        visiting.add(value.uuid)

        producer = producer_map.get(value.uuid)
        if isinstance(producer, (MeasureOperation, MeasureVectorOperation)):
            result = id(producer) in body_operation_ids
            visiting.discard(value.uuid)
            return result

        parent = value.parent_array
        while parent is not None:
            parent_producer = producer_map.get(parent.uuid)
            if isinstance(parent_producer, (MeasureOperation, MeasureVectorOperation)):
                result = id(parent_producer) in body_operation_ids
                visiting.discard(value.uuid)
                return result
            parent = parent.slice_of

        if isinstance(producer, IfOperation):
            for merge in producer.iter_merges():
                if merge.result.uuid != value.uuid:
                    continue
                result = self._is_loop_local_update(
                    merge.true_value,
                    initial,
                    body_operation_ids,
                    producer_map,
                    visiting,
                ) and self._is_loop_local_update(
                    merge.false_value,
                    initial,
                    body_operation_ids,
                    producer_map,
                    visiting,
                )
                visiting.discard(value.uuid)
                return result

        if (
            isinstance(producer, RuntimeClassicalExpr)
            and producer.kind is RuntimeOpKind.SELECT
        ):
            result = self._is_loop_local_update(
                producer.operands[1],
                initial,
                body_operation_ids,
                producer_map,
                visiting,
            ) and self._is_loop_local_update(
                producer.operands[2],
                initial,
                body_operation_ids,
                producer_map,
                visiting,
            )
            visiting.discard(value.uuid)
            return result

        visiting.discard(value.uuid)
        return False

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

        Thin wrapper for the module-level :func:`is_measurement_backed`.

        Args:
            value (Value): The IR value to trace.
            producer_map (dict[str, Operation]): Map from result UUID to the
                operation that produced it.
            visiting (set[str] | None): UUIDs on the current DFS path, used to
                break cycles. Defaults to None (fresh traversal).

        Returns:
            bool: True if the value is measurement-backed.
        """
        return is_measurement_backed(value, producer_map, visiting)

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
