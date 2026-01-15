"""Executable program structures for compiled quantum-classical programs."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.operation import OperationKind
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, CompOp, NotOp, CondOp
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.segments import (
    ClassicalSegment,
    QuantumSegment,
    SeparatedProgram,
)
from qamomile.circuit.transpiler.errors import ExecutionError

T = TypeVar("T")  # Backend circuit type


@dataclasses.dataclass
class CompiledQuantumSegment(Generic[T]):
    """A quantum segment with emitted backend circuit."""

    segment: QuantumSegment
    circuit: T

    # Mapping from Value UUIDs to physical qubit indices
    qubit_map: dict[str, int] = dataclasses.field(default_factory=dict)

    # Mapping from Value UUIDs to classical bit indices (for measurements)
    clbit_map: dict[str, int] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class CompiledClassicalSegment:
    """A classical segment ready for Python execution."""

    segment: ClassicalSegment


class ExecutionContext:
    """Holds global state during program execution."""

    def __init__(self, initial_bindings: dict[str, Any] | None = None):
        self._state: dict[str, Any] = initial_bindings.copy() if initial_bindings else {}

    def get(self, key: str) -> Any:
        return self._state.get(key)

    def set(self, key: str, value: Any) -> None:
        self._state[key] = value

    def get_many(self, keys: list[str]) -> dict[str, Any]:
        return {k: self._state[k] for k in keys if k in self._state}

    def update(self, values: dict[str, Any]) -> None:
        self._state.update(values)

    def has(self, key: str) -> bool:
        return key in self._state


class QuantumExecutor(ABC, Generic[T]):
    """Abstract base for quantum backend execution."""

    @abstractmethod
    def submit(
        self,
        compiled_segment: CompiledQuantumSegment[T],
        context: ExecutionContext,
    ) -> Any:
        """Submit quantum circuit for execution.

        Returns a job or result depending on backend.
        """
        pass

    @abstractmethod
    def get_result(self, job: Any) -> dict[str, Any]:
        """Extract results from quantum execution.

        Returns mapping from value refs to measurement results.
        """
        pass


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
        # Add more operation types as needed

    def _execute_binop(
        self,
        op: BinOp,
        context: ExecutionContext,
        results: dict[str, Any],
    ) -> None:
        """Execute binary operation."""
        lhs = self._get_value(op.operands[0], context, results)
        rhs = self._get_value(op.operands[1], context, results)

        # BinOp has an 'op' or 'operation' field indicating the operation type
        # This depends on the actual implementation of BinOp
        result_value: Any = None
        # Placeholder - actual implementation depends on BinOp structure
        # result_value = lhs + rhs  # for ADD
        # etc.

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

        # Placeholder - actual implementation depends on CompOp structure
        result_value: Any = None

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

        # Placeholder - actual implementation depends on CondOp structure
        result_value: Any = None

        if op.results:
            results[op.results[0].uuid] = result_value

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


@dataclasses.dataclass
class ExecutableProgram(Generic[T]):
    """A fully compiled program ready for execution.

    This is the new Orchestrator - manages execution of mixed
    classical/quantum programs.
    """

    compiled_quantum: list[CompiledQuantumSegment[T]] = dataclasses.field(
        default_factory=list
    )
    compiled_classical: list[CompiledClassicalSegment] = dataclasses.field(
        default_factory=list
    )

    # Execution order: indices into compiled_quantum or compiled_classical
    # Tuple of (is_quantum: bool, index: int)
    execution_order: list[tuple[bool, int]] = dataclasses.field(default_factory=list)

    # Final output references
    output_refs: list[str] = dataclasses.field(default_factory=list)

    def run(
        self,
        quantum_executor: QuantumExecutor[T],
        bindings: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Execute the program.

        Args:
            quantum_executor: Backend-specific quantum executor
            bindings: Parameter values to bind

        Returns:
            List of output values
        """
        context = ExecutionContext(bindings)
        classical_executor = ClassicalExecutor()

        for is_quantum, index in self.execution_order:
            if is_quantum:
                compiled = self.compiled_quantum[index]
                job = quantum_executor.submit(compiled, context)
                results = quantum_executor.get_result(job)
                context.update(results)
            else:
                compiled = self.compiled_classical[index]
                results = classical_executor.execute(compiled.segment, context)
                context.update(results)

        return [context.get(ref) for ref in self.output_refs]

    def get_circuits(self) -> list[T]:
        """Get all quantum circuits in execution order."""
        return [seg.circuit for seg in self.compiled_quantum]

    def get_first_circuit(self) -> T | None:
        """Get the first quantum circuit, or None if no quantum segments."""
        if self.compiled_quantum:
            return self.compiled_quantum[0].circuit
        return None
