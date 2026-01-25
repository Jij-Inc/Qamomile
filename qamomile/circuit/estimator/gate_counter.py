"""Gate counting for quantum circuits.

This module provides algebraic gate counting using SymPy expressions,
allowing resource estimates to depend on problem size parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import sympy as sp

from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    GateOperation,
)
from qamomile.circuit.ir.operation.operation import Operation

if TYPE_CHECKING:
    from qamomile.circuit.ir.block import Block
    from qamomile.circuit.ir.block_value import BlockValue


@dataclass
class GateCount:
    """Gate count breakdown for a quantum circuit.

    All counts are SymPy expressions that may contain symbols
    for parametric problem sizes.

    Attributes:
        total: Total number of gates
        single_qubit: Number of single-qubit gates
        two_qubit: Number of two-qubit gates
        t_gates: Number of T gates (critical for fault tolerance)
        clifford_gates: Number of Clifford gates (H, S, CNOT, CZ, etc.)
    """

    total: sp.Expr
    single_qubit: sp.Expr
    two_qubit: sp.Expr
    t_gates: sp.Expr
    clifford_gates: sp.Expr

    def __add__(self, other: GateCount) -> GateCount:
        """Add two gate counts together."""
        return GateCount(
            total=self.total + other.total,
            single_qubit=self.single_qubit + other.single_qubit,
            two_qubit=self.two_qubit + other.two_qubit,
            t_gates=self.t_gates + other.t_gates,
            clifford_gates=self.clifford_gates + other.clifford_gates,
        )

    def __mul__(self, factor: sp.Expr | int) -> GateCount:
        """Multiply gate counts by a factor."""
        factor_expr = sp.Integer(factor) if isinstance(factor, int) else factor
        return GateCount(
            total=self.total * factor_expr,
            single_qubit=self.single_qubit * factor_expr,
            two_qubit=self.two_qubit * factor_expr,
            t_gates=self.t_gates * factor_expr,
            clifford_gates=self.clifford_gates * factor_expr,
        )

    __rmul__ = __mul__

    def max(self, other: GateCount) -> GateCount:
        """Element-wise maximum of two gate counts."""
        return GateCount(
            total=sp.Max(self.total, other.total),
            single_qubit=sp.Max(self.single_qubit, other.single_qubit),
            two_qubit=sp.Max(self.two_qubit, other.two_qubit),
            t_gates=sp.Max(self.t_gates, other.t_gates),
            clifford_gates=sp.Max(self.clifford_gates, other.clifford_gates),
        )

    def simplify(self) -> GateCount:
        """Simplify all SymPy expressions."""
        return GateCount(
            total=sp.simplify(self.total),
            single_qubit=sp.simplify(self.single_qubit),
            two_qubit=sp.simplify(self.two_qubit),
            t_gates=sp.simplify(self.t_gates),
            clifford_gates=sp.simplify(self.clifford_gates),
        )

    @staticmethod
    def zero() -> GateCount:
        """Return a zero gate count."""
        return GateCount(
            total=sp.Integer(0),
            single_qubit=sp.Integer(0),
            two_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=sp.Integer(0),
        )


# Gate categorization
CLIFFORD_GATES = {"h", "s", "sdg", "cx", "cy", "cz", "swap"}
T_GATES = {"t", "tdg"}
SINGLE_QUBIT_GATES = {
    "h",
    "x",
    "y",
    "z",
    "s",
    "sdg",
    "t",
    "tdg",
    "rx",
    "ry",
    "rz",
    "p",
    "u",
    "u1",
    "u2",
    "u3",
}
TWO_QUBIT_GATES = {"cx", "cy", "cz", "swap", "cp", "crx", "cry", "crz", "rzz"}


def _count_gate_operation(op: GateOperation) -> GateCount:
    """Count gates for a single gate operation.

    Args:
        op: The gate operation to count

    Returns:
        Gate counts for this operation
    """
    # Get gate name from enum
    gate_name = op.gate_type.name.lower() if op.gate_type else "unknown"

    # Determine gate type
    is_clifford = gate_name in CLIFFORD_GATES
    is_t_gate = gate_name in T_GATES
    is_single_qubit = gate_name in SINGLE_QUBIT_GATES
    is_two_qubit = gate_name in TWO_QUBIT_GATES

    # Count based on gate type
    single = sp.Integer(1) if is_single_qubit else sp.Integer(0)
    two = sp.Integer(1) if is_two_qubit else sp.Integer(0)
    t_count = sp.Integer(1) if is_t_gate else sp.Integer(0)
    clifford = sp.Integer(1) if is_clifford else sp.Integer(0)

    return GateCount(
        total=sp.Integer(1),
        single_qubit=single,
        two_qubit=two,
        t_gates=t_count,
        clifford_gates=clifford,
    )


def _count_from_operations(operations: list[Operation]) -> GateCount:
    """Count gates from a list of operations.

    Args:
        operations: List of operations to analyze

    Returns:
        Total gate counts as SymPy expressions
    """
    count = GateCount.zero()

    for op in operations:
        match op:
            case GateOperation():
                count = count + _count_gate_operation(op)

            case ForOperation():
                # Multiply inner count by loop iterations
                inner_count = _count_from_operations(op.operations)

                # Get loop bounds: for i in range(start, stop, step)
                # iterations = (stop - start) / step
                if len(op.operands) >= 2:
                    start = op.operands[0]
                    stop = op.operands[1]
                    step = op.operands[2] if len(op.operands) >= 3 else None

                    # Convert values to SymPy expressions
                    def value_to_expr(v: Any) -> sp.Expr:
                        if hasattr(v, "value"):
                            # Constant value
                            return sp.Integer(int(v.value))
                        elif hasattr(v, "name"):
                            # Symbolic value
                            return sp.Symbol(v.name)
                        else:
                            return sp.Symbol("unknown")

                    start_expr = value_to_expr(start)
                    stop_expr = value_to_expr(stop)

                    if step is None:
                        # Default step = 1
                        iterations = stop_expr - start_expr
                    else:
                        step_expr = value_to_expr(step)
                        iterations = (stop_expr - start_expr) / step_expr
                else:
                    # Fallback
                    iterations = sp.Symbol("iter")

                count = count + (inner_count * iterations)

            case WhileOperation():
                # Conservative: assume constant iterations
                # In practice, would need bounds analysis
                inner_count = _count_from_operations(op.operations)
                iterations = sp.Symbol("while_iter")
                count = count + (inner_count * iterations)

            case IfOperation():
                # Take maximum of both branches
                true_count = _count_from_operations(op.true_operations)
                false_count = _count_from_operations(op.false_operations)
                count = count + true_count.max(false_count)

            case CallBlockOperation():
                # Recursively count gates in called block
                from qamomile.circuit.ir.block_value import BlockValue

                block = op.operands[0]
                if isinstance(block, BlockValue):
                    count = count + count_gates(block)

            case ControlledUOperation():
                # Recursively count gates in the unitary block
                from qamomile.circuit.ir.block_value import BlockValue

                block = op.block
                if isinstance(block, BlockValue):
                    count = count + count_gates(block)

            case _:
                continue

    return count.simplify()


def count_gates(block: BlockValue | Block | list[Operation]) -> GateCount:
    """Count gates in a quantum circuit.

    This function analyzes operations and returns algebraic gate counts
    using SymPy expressions. Counts may contain symbols for parametric
    problem sizes (e.g., loop bounds, array dimensions).

    Supports:
    - GateOperation: Single gate counts
    - ForOperation: Multiplies inner count by iterations
    - IfOperation: Takes maximum of branches
    - CallBlockOperation: Recursively counts called blocks
    - ControlledUOperation: Recursively counts unitary block

    Args:
        block: BlockValue, Block, or list of Operations to analyze

    Returns:
        GateCount with total, single_qubit, two_qubit, t_gates, clifford_gates

    Example:
        >>> from qamomile.circuit.estimator import count_gates
        >>> count = count_gates(my_circuit.block)
        >>> print(count.total)  # e.g., "2*n + 5"
        >>> print(count.t_gates)  # e.g., "n"
    """
    from qamomile.circuit.ir.block import Block
    from qamomile.circuit.ir.block_value import BlockValue

    if isinstance(block, (BlockValue, Block)):
        ops = block.operations
    else:
        ops = block

    return _count_from_operations(ops)
