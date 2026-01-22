"""Hamiltonian evaluation pass.

This module provides a pass to evaluate Hamiltonian operation graphs,
producing ConcreteHamiltonian objects. This is used during transpilation
to convert the symbolic Hamiltonian expressions into concrete form.
"""

from __future__ import annotations

from typing import Dict, Any

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.operation import Operation, CInitOperation
from qamomile.circuit.ir.operation.hamiltonian_ops import (
    PauliCreateOp,
    HamiltonianAddOp,
    HamiltonianMulOp,
    HamiltonianScaleOp,
    HamiltonianNegOp,
    HamiltonianIdentityOp,
)
from qamomile.circuit.ir.operation.control_flow import ForOperation
from qamomile.circuit.ir.types.hamiltonian import HamiltonianExprType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.observable.concrete import ConcreteHamiltonian


class HamiltonianEvalError(Exception):
    """Error during Hamiltonian evaluation."""

    pass


class HamiltonianEvaluator:
    """Evaluates Hamiltonian operation graphs to produce ConcreteHamiltonian.

    This evaluator traverses the operation graph in a block and evaluates
    Hamiltonian operations to produce a concrete representation. It supports:
    - PauliCreateOp: Creates single Pauli terms
    - HamiltonianAddOp: Adds Hamiltonians
    - HamiltonianMulOp: Multiplies Hamiltonians
    - HamiltonianScaleOp: Scales by scalar
    - HamiltonianNegOp: Negates Hamiltonian
    - ForOperation: Unrolls loops (requires concrete bounds)

    The evaluator requires all symbolic values (like loop bounds) to be
    bound to concrete values before evaluation.
    """

    def __init__(self, bindings: Dict[str, Any] | None = None):
        """Initialize the evaluator.

        Args:
            bindings: Dict mapping parameter names to concrete values
        """
        self.bindings = bindings or {}
        # Maps Value to evaluated result (ConcreteHamiltonian or scalar)
        self._value_cache: Dict[int, Any] = {}

    def _get_value_id(self, value: Value) -> int:
        """Get unique ID for a Value."""
        return id(value)

    def _get_scalar(self, value: Value) -> float | complex:
        """Get scalar value from a Float or UInt Value.

        Handles constant values and bound parameters.
        """
        value_id = self._get_value_id(value)
        if value_id in self._value_cache:
            return self._value_cache[value_id]

        # Check for constant in params
        if value.params and "const" in value.params:
            result = value.params["const"]
            self._value_cache[value_id] = result
            return result

        # Check bindings by name
        if value.name in self.bindings:
            result = self.bindings[value.name]
            self._value_cache[value_id] = result
            return result

        raise HamiltonianEvalError(
            f"Cannot evaluate scalar value '{value.name}': "
            f"not a constant and not in bindings"
        )

    def _get_int(self, value: Value) -> int:
        """Get integer value from a UInt Value."""
        scalar = self._get_scalar(value)
        return int(scalar)

    def _get_hamiltonian(self, value: Value) -> ConcreteHamiltonian:
        """Get evaluated Hamiltonian for a Value."""
        value_id = self._get_value_id(value)
        if value_id not in self._value_cache:
            raise HamiltonianEvalError(
                f"Value '{value.name}' has not been evaluated yet"
            )
        result = self._value_cache[value_id]
        if not isinstance(result, ConcreteHamiltonian):
            raise HamiltonianEvalError(
                f"Value '{value.name}' is not a Hamiltonian (got {type(result).__name__})"
            )
        return result

    def _eval_pauli_create(self, op: PauliCreateOp) -> None:
        """Evaluate PauliCreateOp."""
        qubit_idx = self._get_int(op.qubit_index)
        result = ConcreteHamiltonian.single_pauli(op.pauli_kind, qubit_idx)
        self._value_cache[self._get_value_id(op.output)] = result

    def _eval_hamiltonian_add(self, op: HamiltonianAddOp) -> None:
        """Evaluate HamiltonianAddOp."""
        lhs = self._get_hamiltonian(op.lhs)
        rhs = self._get_hamiltonian(op.rhs)
        result = lhs + rhs
        self._value_cache[self._get_value_id(op.output)] = result

    def _eval_hamiltonian_mul(self, op: HamiltonianMulOp) -> None:
        """Evaluate HamiltonianMulOp."""
        lhs = self._get_hamiltonian(op.lhs)
        rhs = self._get_hamiltonian(op.rhs)
        result = lhs * rhs
        self._value_cache[self._get_value_id(op.output)] = result

    def _eval_hamiltonian_scale(self, op: HamiltonianScaleOp) -> None:
        """Evaluate HamiltonianScaleOp."""
        hamiltonian = self._get_hamiltonian(op.hamiltonian_expr)
        scalar = self._get_scalar(op.scalar)
        result = hamiltonian * scalar
        self._value_cache[self._get_value_id(op.output)] = result

    def _eval_hamiltonian_neg(self, op: HamiltonianNegOp) -> None:
        """Evaluate HamiltonianNegOp."""
        hamiltonian = self._get_hamiltonian(op.hamiltonian_expr)
        result = -hamiltonian
        self._value_cache[self._get_value_id(op.output)] = result

    def _eval_hamiltonian_identity(self, op: HamiltonianIdentityOp) -> None:
        """Evaluate HamiltonianIdentityOp."""
        scalar = self._get_scalar(op.scalar)
        result = ConcreteHamiltonian.identity(scalar)
        self._value_cache[self._get_value_id(op.output)] = result

    def _eval_cinit(self, op: CInitOperation) -> None:
        """Evaluate CInitOperation (constants)."""
        result_value = op.results[0]
        if result_value.params and "const" in result_value.params:
            self._value_cache[self._get_value_id(result_value)] = result_value.params["const"]
        elif result_value.name in self.bindings:
            self._value_cache[self._get_value_id(result_value)] = self.bindings[result_value.name]

    def _eval_for_operation(self, op: ForOperation, operations: list[Operation]) -> None:
        """Evaluate ForOperation by unrolling.

        Note: This is a simplified implementation. A full implementation
        would need to properly handle the loop body operations and
        variable scoping.
        """
        start = self._get_int(op.start)
        stop = self._get_int(op.stop)
        step = self._get_int(op.step) if len(op.operands) > 2 else 1

        # Get the loop variable
        loop_var = op.loop_var

        # For each iteration, evaluate the body
        for i in builtins_range(start, stop, step):
            # Bind the loop variable
            self._value_cache[self._get_value_id(loop_var)] = i

            # Evaluate body operations
            if op.body is not None:
                self._eval_block(op.body)

    def _eval_operation(self, op: Operation) -> None:
        """Evaluate a single operation."""
        if isinstance(op, PauliCreateOp):
            self._eval_pauli_create(op)
        elif isinstance(op, HamiltonianAddOp):
            self._eval_hamiltonian_add(op)
        elif isinstance(op, HamiltonianMulOp):
            self._eval_hamiltonian_mul(op)
        elif isinstance(op, HamiltonianScaleOp):
            self._eval_hamiltonian_scale(op)
        elif isinstance(op, HamiltonianNegOp):
            self._eval_hamiltonian_neg(op)
        elif isinstance(op, HamiltonianIdentityOp):
            self._eval_hamiltonian_identity(op)
        elif isinstance(op, CInitOperation):
            self._eval_cinit(op)
        elif isinstance(op, ForOperation):
            self._eval_for_operation(op, [])
        # Skip other operation types (quantum gates, etc.)

    def _eval_block(self, block: Block) -> None:
        """Evaluate all operations in a block."""
        for op in block.operations:
            self._eval_operation(op)

    def evaluate(self, block: Block) -> ConcreteHamiltonian | None:
        """Evaluate a block and return the Hamiltonian result.

        Args:
            block: The block containing Hamiltonian operations

        Returns:
            ConcreteHamiltonian if the block produces one, None otherwise
        """
        self._eval_block(block)

        # Find the result - look for HamiltonianExprType in block outputs
        for value in block.output_values:
            if isinstance(value.type, HamiltonianExprType):
                return self._get_hamiltonian(value)

        # Check if any value in cache is a Hamiltonian
        hamiltonians = [
            v for v in self._value_cache.values()
            if isinstance(v, ConcreteHamiltonian)
        ]
        if hamiltonians:
            return hamiltonians[-1]

        return None


# Keep Python's range accessible
import builtins
builtins_range = builtins.range


def evaluate_hamiltonian(
    block: Block, bindings: Dict[str, Any] | None = None
) -> ConcreteHamiltonian | None:
    """Convenience function to evaluate a Hamiltonian from a block.

    Args:
        block: Block containing Hamiltonian operations
        bindings: Parameter bindings

    Returns:
        Evaluated ConcreteHamiltonian, or None if block doesn't produce one
    """
    evaluator = HamiltonianEvaluator(bindings)
    return evaluator.evaluate(block)
