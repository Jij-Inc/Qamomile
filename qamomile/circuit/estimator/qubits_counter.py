"""Qubit resource counting for BlockValue."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import sympy as sp

from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.operation.composite_gate import CompositeGateOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import ControlledUOperation
from qamomile.circuit.ir.operation.operation import Operation, QInitOperation
from qamomile.circuit.ir.types.primitives import QubitType
from qamomile.circuit.ir.value import ArrayValue, Value

if TYPE_CHECKING:
    from qamomile.circuit.ir.block_value import BlockValue

BINOP_TO_SYMPY = {
    BinOpKind.ADD: lambda l, r: l + r,
    BinOpKind.SUB: lambda l, r: l - r,
    BinOpKind.MUL: lambda l, r: l * r,
    BinOpKind.DIV: lambda l, r: l / r,
    BinOpKind.FLOORDIV: lambda l, r: sp.floor(l / r),
    BinOpKind.POW: lambda l, r: l**r,
}

WHILE_SYMBOL = sp.Symbol("|while|")


def _resolve_value_to_sympy(
    value: Value,
    symbol_map: dict[str, sp.Expr],
) -> sp.Expr:
    """Resolve a Value to a sympy expression.

    Resolution priority:
      1. Constant -> sp.Integer
      2. Parameter -> sp.Symbol (integer=True, positive=True)
      3. Registered in symbol_map (BinOp result etc.) -> cached expression
      4. Fallback -> sp.Symbol(name, integer=True, positive=True)
    """
    if value.is_constant():
        const_val = value.get_const()
        if const_val is not None:
            return sp.Integer(int(const_val))

    if value.is_parameter():
        param_name = value.parameter_name()
        if param_name is not None:
            return sp.Symbol(param_name, integer=True, positive=True)

    if value.uuid in symbol_map:
        return symbol_map[value.uuid]

    return sp.Symbol(value.name, integer=True, positive=True)


def _count_qinit(op: QInitOperation, symbol_map: dict[str, sp.Expr]) -> sp.Expr:
    """Count qubits from a QInitOperation.

    Args:
        op: The QInitOperation to count qubits from.
        symbol_map: Mapping from Value UUIDs to resolved sympy expressions.

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
            el_count *= _resolve_value_to_sympy(dim, symbol_map)
        return el_count

    # Single qubit
    if isinstance(result.type, QubitType):
        return sp.Integer(1)

    return sp.Integer(0)


def _count_input_qubits(input_values: list, symbol_map: dict[str, sp.Expr]) -> sp.Expr:
    """Count qubit-typed values in BlockValue.input_values.

    Only used at the top level to count qubits passed as function arguments.
    Not used in recursive calls to avoid double-counting.

    Args:
        input_values: List of Value objects from BlockValue.input_values.
        symbol_map: Mapping from Value UUIDs to resolved sympy expressions.

    Returns:
        Number of input qubits as a sympy expression.
    """
    count: sp.Expr = sp.Integer(0)
    for v in input_values:
        if isinstance(v, ArrayValue) and isinstance(v.type, QubitType):
            el_count: sp.Expr = sp.Integer(1)
            for dim in v.shape:
                el_count *= _resolve_value_to_sympy(dim, symbol_map)
            count += el_count
        elif isinstance(v.type, QubitType):
            count += sp.Integer(1)
    return count


def _count_from_operations(
    operations: list[Operation],
    symbol_map: dict[str, sp.Expr] | None = None,
) -> sp.Expr:
    """Count qubits from a list of operations.

    Args:
        operations: List of operations to analyze.
        symbol_map: Mapping from Value UUIDs to resolved sympy expressions.

    Returns:
        Total qubit count as a sympy expression.
    """
    if symbol_map is None:
        symbol_map = {}

    count: sp.Expr = sp.Integer(0)

    for op in operations:
        match op:
            case BinOp():
                lhs_expr = _resolve_value_to_sympy(op.lhs, symbol_map)
                rhs_expr = _resolve_value_to_sympy(op.rhs, symbol_map)
                if op.kind in BINOP_TO_SYMPY:
                    symbol_map[op.output.uuid] = BINOP_TO_SYMPY[op.kind](
                        lhs_expr, rhs_expr
                    )

            case QInitOperation():
                count += _count_qinit(op, symbol_map)  # type: ignore

            case ForOperation():
                inner_count = _count_from_operations(op.operations, symbol_map)
                start_expr = _resolve_value_to_sympy(op.operands[0], symbol_map)
                stop_expr = _resolve_value_to_sympy(op.operands[1], symbol_map)
                step_expr = _resolve_value_to_sympy(op.operands[2], symbol_map)
                iterations = (stop_expr - start_expr) / step_expr
                count += inner_count * iterations  # type: ignore

            case WhileOperation():
                inner_count = _count_from_operations(op.operations, symbol_map)
                count += inner_count * WHILE_SYMBOL  # type: ignore

            case IfOperation():
                # Take maximum of both branches
                true_count = _count_from_operations(op.true_operations, symbol_map)
                false_count = _count_from_operations(op.false_operations, symbol_map)
                count += sp.Max(true_count, false_count)  # type: ignore

            case CallBlockOperation():
                # Recursively count only internally-allocated qubits
                # (not input_values, to avoid double-counting passed-in qubits)
                from qamomile.circuit.ir.block_value import BlockValue

                called_block = op.operands[0]
                if isinstance(called_block, BlockValue):
                    # Map formal input array dimensions to actual argument dimensions
                    inner_symbol_map = symbol_map.copy()
                    for formal_idx, formal_input in enumerate(
                        called_block.input_values
                    ):
                        if formal_idx + 1 < len(op.operands):
                            actual_arg = op.operands[formal_idx + 1]
                            if isinstance(actual_arg, ArrayValue) and isinstance(
                                formal_input, ArrayValue
                            ):
                                for dim_formal, dim_actual in zip(
                                    formal_input.shape, actual_arg.shape
                                ):
                                    inner_symbol_map[dim_formal.uuid] = (
                                        _resolve_value_to_sympy(
                                            dim_actual, symbol_map
                                        )
                                    )
                    count += _count_from_operations(called_block.operations, inner_symbol_map)  # type: ignore

            case ControlledUOperation():
                # Recursively count only internally-allocated qubits
                from qamomile.circuit.ir.block_value import BlockValue

                controlled_block = op.block
                if isinstance(controlled_block, BlockValue):
                    inner_symbol_map = symbol_map.copy()
                    if op.has_index_spec:
                        # index_spec mode: skip qubit formals,
                        # map only non-qubit ArrayValue dimensions
                        param_operands = op.operands[2:]
                        param_idx = 0
                        for formal_input in controlled_block.input_values:
                            if not formal_input.type.is_quantum():
                                if param_idx < len(param_operands):
                                    actual_arg = param_operands[param_idx]
                                    if isinstance(
                                        actual_arg, ArrayValue
                                    ) and isinstance(formal_input, ArrayValue):
                                        for dim_formal, dim_actual in zip(
                                            formal_input.shape,
                                            actual_arg.shape,
                                        ):
                                            inner_symbol_map[
                                                dim_formal.uuid
                                            ] = _resolve_value_to_sympy(
                                                dim_actual, symbol_map
                                            )
                                    param_idx += 1
                    else:
                        # Map formal input array dimensions to actual target operand dimensions
                        target_operands = op.target_operands
                        for formal_idx, formal_input in enumerate(
                            controlled_block.input_values
                        ):
                            if formal_idx < len(target_operands):
                                actual_arg = target_operands[formal_idx]
                                if isinstance(
                                    actual_arg, ArrayValue
                                ) and isinstance(formal_input, ArrayValue):
                                    for dim_formal, dim_actual in zip(
                                        formal_input.shape, actual_arg.shape
                                    ):
                                        inner_symbol_map[dim_formal.uuid] = (
                                            _resolve_value_to_sympy(
                                                dim_actual, symbol_map
                                            )
                                        )
                    count += _count_from_operations(controlled_block.operations, inner_symbol_map)  # type: ignore

            case ForItemsOperation():
                inner_count = _count_from_operations(op.operations, symbol_map)
                dict_operand = op.operands[0]
                if (
                    hasattr(dict_operand, "is_parameter")
                    and dict_operand.is_parameter()
                ):
                    dict_name = dict_operand.parameter_name() or dict_operand.name
                else:
                    dict_name = dict_operand.name
                cardinality = sp.Symbol(f"|{dict_name}|", integer=True, positive=True)
                count += inner_count * cardinality  # type: ignore

            case CompositeGateOperation():
                from qamomile.circuit.ir.block_value import BlockValue

                # Count only internally-allocated qubits (not input_values)
                impl = op.implementation
                if isinstance(impl, BlockValue):
                    count += _count_from_operations(impl.operations, symbol_map)  # type: ignore

                # Add ancilla qubits from resource metadata
                if op.resource_metadata is not None:
                    count += sp.Integer(op.resource_metadata.ancilla_qubits)

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
    from qamomile.circuit.ir.block_value import BlockValue

    if isinstance(block, BlockValue):
        symbol_map: dict[str, sp.Expr] = {}
        input_qubits = _count_input_qubits(block.input_values, symbol_map)
        ops_qubits = _count_from_operations(block.operations, symbol_map)
        return sp.simplify(input_qubits + ops_qubits)
    return _count_from_operations(block)
