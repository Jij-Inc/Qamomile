"""Control flow visitor for operation traversal."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.control_flow import HasNestedOps


class ControlFlowVisitor(ABC):
    """Base class for visiting operations with control flow handling.

    Subclasses override `visit_operation` to define per-operation behavior.
    Control flow recursion is handled automatically by the base class.

    Example:
        class MeasurementCounter(ControlFlowVisitor):
            def __init__(self):
                self.count = 0

            def visit_operation(self, op: Operation) -> None:
                if isinstance(op, MeasureOperation):
                    self.count += 1
    """

    def visit_operations(self, operations: list[Operation]) -> None:
        """Visit all operations including nested control flow."""
        for op in operations:
            self.visit_operation(op)
            self._visit_control_flow(op)

    @abstractmethod
    def visit_operation(self, op: Operation) -> None:
        """Process a single operation. Override in subclasses."""
        pass

    def _visit_control_flow(self, op: Operation) -> None:
        """Recursively visit operations inside control flow constructs."""
        if isinstance(op, HasNestedOps):
            for op_list in op.nested_op_lists():
                self.visit_operations(op_list)


class OperationTransformer(ABC):
    """Base class for transforming operations with control flow handling.

    Subclasses override `transform_operation` to define per-operation transformation.
    Control flow recursion and rebuilding is handled automatically.

    Example:
        class OperationRenamer(OperationTransformer):
            def transform_operation(self, op: Operation) -> Operation:
                # Return modified operation
                return dataclasses.replace(op, ...)
    """

    def transform_operations(self, operations: list[Operation]) -> list[Operation]:
        """Transform all operations including nested control flow."""
        result: list[Operation] = []
        for op in operations:
            transformed = self.transform_operation(op)
            if transformed is not None:
                transformed = self._transform_control_flow(transformed)
                result.append(transformed)
        return result

    @abstractmethod
    def transform_operation(self, op: Operation) -> Operation | None:
        """Transform a single operation. Return None to remove it."""
        pass

    def _transform_control_flow(self, op: Operation) -> Operation:
        """Recursively transform operations inside control flow constructs."""
        if isinstance(op, HasNestedOps):
            new_lists = [
                self.transform_operations(op_list) for op_list in op.nested_op_lists()
            ]
            return op.rebuild_nested(new_lists)
        return op


class OperationCollector(ControlFlowVisitor):
    """Collects operations matching a predicate.

    Example:
        collector = OperationCollector(lambda op: isinstance(op, MeasureOperation))
        collector.visit_operations(block.operations)
        measurements = collector.collected
    """

    def __init__(self, predicate: Callable[[Operation], bool]):
        self._predicate = predicate
        self.collected: list[Operation] = []

    def visit_operation(self, op: Operation) -> None:
        if self._predicate(op):
            self.collected.append(op)


class ValueCollector(ControlFlowVisitor):
    """Collects Value UUIDs from operation operands and results.

    Attributes:
        operand_uuids: Set of UUIDs from operands
        result_uuids: Set of UUIDs from results
    """

    def __init__(self):
        self.operand_uuids: set[str] = set()
        self.result_uuids: set[str] = set()

    def visit_operation(self, op: Operation) -> None:
        from qamomile.circuit.ir.value import ValueBase

        for operand in op.operands:
            if isinstance(operand, ValueBase):
                self.operand_uuids.add(operand.uuid)

        for result in op.results:
            self.result_uuids.add(result.uuid)
