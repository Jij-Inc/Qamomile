"""Circuit depth estimation for quantum circuits.

This module estimates circuit depth by analyzing operation dependencies.
Depth is expressed as SymPy expressions for parametric problem sizes.
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
class CircuitDepth:
    """Circuit depth breakdown for a quantum circuit.

    All depths are SymPy expressions that may contain symbols
    for parametric problem sizes.

    Attributes:
        total_depth: Total circuit depth (all gates)
        t_depth: Depth of T gates only (critical for fault tolerance)
        two_qubit_depth: Depth of two-qubit gates (often the bottleneck)
    """

    total_depth: sp.Expr
    t_depth: sp.Expr
    two_qubit_depth: sp.Expr

    def __add__(self, other: CircuitDepth) -> CircuitDepth:
        """Add two circuit depths (sequential composition)."""
        return CircuitDepth(
            total_depth=self.total_depth + other.total_depth,
            t_depth=self.t_depth + other.t_depth,
            two_qubit_depth=self.two_qubit_depth + other.two_qubit_depth,
        )

    def __mul__(self, factor: sp.Expr | int) -> CircuitDepth:
        """Multiply circuit depth by a factor."""
        factor_expr = sp.Integer(factor) if isinstance(factor, int) else factor
        return CircuitDepth(
            total_depth=self.total_depth * factor_expr,
            t_depth=self.t_depth * factor_expr,
            two_qubit_depth=self.two_qubit_depth * factor_expr,
        )

    __rmul__ = __mul__

    def max(self, other: CircuitDepth) -> CircuitDepth:
        """Element-wise maximum of two circuit depths (parallel composition)."""
        return CircuitDepth(
            total_depth=sp.Max(self.total_depth, other.total_depth),
            t_depth=sp.Max(self.t_depth, other.t_depth),
            two_qubit_depth=sp.Max(self.two_qubit_depth, other.two_qubit_depth),
        )

    def simplify(self) -> CircuitDepth:
        """Simplify all SymPy expressions."""
        return CircuitDepth(
            total_depth=sp.simplify(self.total_depth),
            t_depth=sp.simplify(self.t_depth),
            two_qubit_depth=sp.simplify(self.two_qubit_depth),
        )

    @staticmethod
    def zero() -> CircuitDepth:
        """Return a zero circuit depth."""
        return CircuitDepth(
            total_depth=sp.Integer(0),
            t_depth=sp.Integer(0),
            two_qubit_depth=sp.Integer(0),
        )


# Gate categorization
T_GATES = {"t", "tdg"}
TWO_QUBIT_GATES = {"cx", "cy", "cz", "swap", "cp", "crx", "cry", "crz", "rzz"}


def _estimate_gate_depth(op: GateOperation) -> CircuitDepth:
    """Estimate depth for a single gate operation.

    Args:
        op: The gate operation

    Returns:
        Circuit depths for this operation
    """
    # Get gate name from enum
    gate_name = op.gate_type.name.lower() if op.gate_type else "unknown"

    is_t_gate = gate_name in T_GATES
    is_two_qubit = gate_name in TWO_QUBIT_GATES

    t_depth = sp.Integer(1) if is_t_gate else sp.Integer(0)
    two_qubit_depth = sp.Integer(1) if is_two_qubit else sp.Integer(0)

    return CircuitDepth(
        total_depth=sp.Integer(1),
        t_depth=t_depth,
        two_qubit_depth=two_qubit_depth,
    )


def _estimate_sequential_depth(operations: list[Operation]) -> CircuitDepth:
    """Estimate depth assuming sequential execution.

    This is a conservative estimate that assumes no parallelization.
    For actual parallel depth, would need full dependency analysis.

    Args:
        operations: List of operations

    Returns:
        Sequential circuit depth
    """
    depth = CircuitDepth.zero()

    for op in operations:
        match op:
            case GateOperation():
                depth = depth + _estimate_gate_depth(op)

            case ForOperation():
                # Multiply inner depth by loop iterations
                inner_depth = _estimate_sequential_depth(op.operations)

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

                depth = depth + (inner_depth * iterations)

            case WhileOperation():
                # Conservative: assume constant iterations
                inner_depth = _estimate_sequential_depth(op.operations)
                iterations = sp.Symbol("while_iter")
                depth = depth + (inner_depth * iterations)

            case IfOperation():
                # Take maximum of both branches
                true_depth = _estimate_sequential_depth(op.true_operations)
                false_depth = _estimate_sequential_depth(op.false_operations)
                depth = depth + true_depth.max(false_depth)

            case CallBlockOperation():
                # Recursively estimate depth in called block
                from qamomile.circuit.ir.block_value import BlockValue

                block = op.operands[0]
                if isinstance(block, BlockValue):
                    depth = depth + estimate_depth(block)

            case ControlledUOperation():
                # Recursively estimate depth in the unitary block
                from qamomile.circuit.ir.block_value import BlockValue

                block = op.block
                if isinstance(block, BlockValue):
                    depth = depth + estimate_depth(block)

            case _:
                continue

    return depth.simplify()


def estimate_depth(block: BlockValue | Block | list[Operation]) -> CircuitDepth:
    """Estimate circuit depth.

    This function provides a conservative (upper bound) estimate of circuit depth
    by assuming sequential execution. For actual parallelizable depth, a full
    dependency analysis would be required.

    Depth is expressed using SymPy expressions and may contain symbols for
    parametric problem sizes (e.g., loop bounds).

    Supports:
    - GateOperation: Contributes depth 1
    - ForOperation: Multiplies inner depth by iterations
    - IfOperation: Takes maximum of branches
    - CallBlockOperation: Recursively estimates called blocks
    - ControlledUOperation: Recursively estimates unitary block

    Args:
        block: BlockValue, Block, or list of Operations to analyze

    Returns:
        CircuitDepth with total_depth, t_depth, two_qubit_depth

    Example:
        >>> from qamomile.circuit.estimator import estimate_depth
        >>> depth = estimate_depth(my_circuit.block)
        >>> print(depth.total_depth)  # e.g., "n + 5"
        >>> print(depth.t_depth)  # e.g., "n"

    Note:
        This is a sequential depth estimate (upper bound). Actual depth
        may be lower due to gate parallelization, which requires full
        dependency graph analysis.
    """
    from qamomile.circuit.ir.block import Block
    from qamomile.circuit.ir.block_value import BlockValue

    if isinstance(block, (BlockValue, Block)):
        ops = block.operations
    else:
        ops = block

    return _estimate_sequential_depth(ops)
