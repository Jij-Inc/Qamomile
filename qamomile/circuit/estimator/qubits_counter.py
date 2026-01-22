"""Qubit resource counting for BlockValue."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import sympy as sp

from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import ControlledUOperation
from qamomile.circuit.ir.operation.operation import Operation, QInitOperation
from qamomile.circuit.ir.types.primitives import QubitType
from qamomile.circuit.ir.value import ArrayValue

if TYPE_CHECKING:
    from qamomile.circuit.ir.block_value import BlockValue


def _count_qinit(op: QInitOperation) -> sp.Expr:
    """Count qubits from a QInitOperation.

    Args:
        op: The QInitOperation to count qubits from.

    Returns:
        Number of qubits as a sympy expression (may contain symbols for
        parametric dimensions).
    """
    result = op.results[0]

    # Array of qubits (ArrayValue with quantum element type)
    # Check ArrayValue first before checking single qubit type
    # Note: Use isinstance instead of is_quantum() due to MRO issue
    if isinstance(result, ArrayValue) and isinstance(result.type, QubitType):
        el_count: sp.Expr = sp.Integer(1)
        for dim in result.shape:
            if dim.is_constant():
                const_val = dim.get_const()
                if const_val is not None:
                    el_count *= sp.Integer(int(const_val))
            elif dim.is_parameter():
                param_name = dim.parameter_name()
                if param_name is not None:
                    el_count *= sp.Symbol(param_name)  # type: ignore
            else:
                # Use name as symbol
                el_count *= sp.Symbol(dim.name)  # type: ignore
        return el_count

    # Single qubit
    if isinstance(result.type, QubitType):
        return sp.Integer(1)

    return sp.Integer(0)


def _count_from_operations(operations: list[Operation]) -> sp.Expr:
    """Count qubits from a list of operations.

    Args:
        operations: List of operations to analyze.

    Returns:
        Total qubit count as a sympy expression.
    """
    count: sp.Expr = sp.Integer(0)

    for op in operations:
        match op:
            case QInitOperation():
                count += _count_qinit(op)  # type: ignore

            case ForOperation():
                # Assume qubits are uncomputed after loop, don't multiply by iterations
                inner_count = _count_from_operations(op.operations)
                count += inner_count  # type: ignore

            case WhileOperation():
                # Same as for loop - assume uncomputation
                inner_count = _count_from_operations(op.operations)
                count += inner_count  # type: ignore

            case IfOperation():
                # Take maximum of both branches
                true_count = _count_from_operations(op.true_operations)
                false_count = _count_from_operations(op.false_operations)
                count += sp.Max(true_count, false_count)  # type: ignore

            case CallBlockOperation():
                # Recursively count qubits in called block
                from qamomile.circuit.ir.block_value import BlockValue

                block = op.operands[0]
                if isinstance(block, BlockValue):
                    count += qubits_counter(block)  # type: ignore

            case ControlledUOperation():
                # Recursively count qubits in the unitary block
                from qamomile.circuit.ir.block_value import BlockValue

                block = op.block
                if isinstance(block, BlockValue):
                    count += qubits_counter(block)  # type: ignore

            case _:
                continue

    return sp.simplify(count)


@overload
def qubits_counter(block: BlockValue) -> sp.Expr: ...


@overload
def qubits_counter(block: list[Operation]) -> sp.Expr: ...


def qubits_counter(block: BlockValue | list[Operation]) -> sp.Expr:
    """Count the number of qubits required by a BlockValue.

    This function analyzes the operations in a BlockValue and counts
    the total number of qubits that need to be allocated. It handles:

    - QInitOperation: Counts single qubits and qubit arrays
    - ForOperation/WhileOperation: Counts inner resources (assumes uncomputation)
    - IfOperation: Takes the maximum of both branches
    - CallBlockOperation: Recursively counts called blocks
    - ControlledUOperation: Recursively counts the unitary block

    Args:
        block: The BlockValue to analyze.

    Returns:
        The qubit count as a sympy expression. May contain symbols
        for parametric dimensions (e.g., if array sizes are parameters).

    Example:
        >>> from qamomile.circuit.ir.block_value import BlockValue
        >>> block = some_block_value()
        >>> count = qubits_counter(block)
        >>> print(count)  # e.g., "n + 3" for parametric n
    """

    ops = block.operations if isinstance(block, BlockValue) else block
    return _count_from_operations(ops)
