"""Qubit resource counting for BlockValue."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import sympy as sp

from qamomile.circuit.estimator._utils import BINOP_TO_SYMPY, _strip_nonneg_max
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp
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


def _is_clean_call(block: "BlockValue") -> bool:
    """Check if a BlockValue is a 'clean call'.

    A clean call returns only the qubits it received as inputs.
    Any internally allocated qubits (ancillas) are freed before return,
    meaning ancilla qubits can be reused across loop iterations.

    Detection: every quantum-typed return value must have a logical_id
    that matches one of the input_values' logical_ids.

    Conservative: if a cast/alias produces a new logical_id, this returns
    False (safe fallback — will overcount, never undercount).
    """
    input_logical_ids = {v.logical_id for v in block.input_values}
    for rv in block.return_values:
        if rv.type.is_quantum():
            if rv.logical_id not in input_logical_ids:
                return False
    return True


# ---------------------------------------------------------------------------
# Shared helpers: extracted from _count_loop_body_split / _count_from_operations
# ---------------------------------------------------------------------------


def _register_binop(op: BinOp, symbol_map: dict[str, sp.Expr]) -> None:
    """Register a BinOp result in the symbol map."""
    lhs_expr = _resolve_value_to_sympy(op.lhs, symbol_map)
    rhs_expr = _resolve_value_to_sympy(op.rhs, symbol_map)
    if op.kind in BINOP_TO_SYMPY:
        symbol_map[op.output.uuid] = BINOP_TO_SYMPY[op.kind](lhs_expr, rhs_expr)


def _build_call_block_inner_map(
    op: CallBlockOperation, symbol_map: dict[str, sp.Expr]
) -> tuple[BlockValue | None, dict[str, sp.Expr]]:
    """Build inner symbol map for CallBlockOperation.

    Maps formal input array dimensions to actual argument dimensions
    so that parametric sizes are resolved correctly in the callee.

    Returns:
        (called_block, inner_symbol_map) if called_block is a BlockValue,
        (None, {}) otherwise.
    """
    from qamomile.circuit.ir.block_value import BlockValue

    called_block = op.operands[0]
    if not isinstance(called_block, BlockValue):
        return None, {}
    inner_symbol_map = symbol_map.copy()
    for formal_idx, formal_input in enumerate(called_block.input_values):
        if formal_idx + 1 < len(op.operands):
            actual_arg = op.operands[formal_idx + 1]
            if isinstance(actual_arg, ArrayValue) and isinstance(
                formal_input, ArrayValue
            ):
                for dim_formal, dim_actual in zip(
                    formal_input.shape, actual_arg.shape
                ):
                    inner_symbol_map[dim_formal.uuid] = _resolve_value_to_sympy(
                        dim_actual, symbol_map
                    )
    return called_block, inner_symbol_map


def _build_controlled_u_inner_map(
    op: ControlledUOperation, symbol_map: dict[str, sp.Expr]
) -> tuple[BlockValue | None, dict[str, sp.Expr]]:
    """Build inner symbol map for ControlledUOperation.

    Handles both index_spec mode (qubit operands identified by index)
    and the default mode (target_operands mapping).

    Returns:
        (controlled_block, inner_symbol_map) if controlled_block is a BlockValue,
        (None, {}) otherwise.
    """
    from qamomile.circuit.ir.block_value import BlockValue

    controlled_block = op.block
    if not isinstance(controlled_block, BlockValue):
        return None, {}
    inner_symbol_map = symbol_map.copy()
    if op.has_index_spec:
        param_operands = op.operands[2:]
        param_idx = 0
        for formal_input in controlled_block.input_values:
            if not formal_input.type.is_quantum():
                if param_idx < len(param_operands):
                    actual_arg = param_operands[param_idx]
                    if isinstance(actual_arg, ArrayValue) and isinstance(
                        formal_input, ArrayValue
                    ):
                        for dim_formal, dim_actual in zip(
                            formal_input.shape, actual_arg.shape
                        ):
                            inner_symbol_map[dim_formal.uuid] = (
                                _resolve_value_to_sympy(dim_actual, symbol_map)
                            )
                    param_idx += 1
    else:
        target_operands = op.target_operands
        for formal_idx, formal_input in enumerate(controlled_block.input_values):
            if formal_idx < len(target_operands):
                actual_arg = target_operands[formal_idx]
                if isinstance(actual_arg, ArrayValue) and isinstance(
                    formal_input, ArrayValue
                ):
                    for dim_formal, dim_actual in zip(
                        formal_input.shape, actual_arg.shape
                    ):
                        inner_symbol_map[dim_formal.uuid] = (
                            _resolve_value_to_sympy(dim_actual, symbol_map)
                        )
    return controlled_block, inner_symbol_map


def _compute_for_iterations(
    op: ForOperation, symbol_map: dict[str, sp.Expr]
) -> sp.Expr:
    """Compute the number of iterations for a ForOperation."""
    start_expr = _resolve_value_to_sympy(op.operands[0], symbol_map)
    stop_expr = _resolve_value_to_sympy(op.operands[1], symbol_map)
    step_expr = _resolve_value_to_sympy(op.operands[2], symbol_map)
    return (stop_expr - start_expr) / step_expr


def _resolve_for_items_cardinality(op: ForItemsOperation) -> sp.Symbol:
    """Resolve the cardinality symbol for a ForItemsOperation."""
    dict_operand = op.operands[0]
    if hasattr(dict_operand, "is_parameter") and dict_operand.is_parameter():
        dict_name = dict_operand.parameter_name() or dict_operand.name
    else:
        dict_name = dict_operand.name
    return sp.Symbol(f"|{dict_name}|", integer=True, positive=True)


# ---------------------------------------------------------------------------
# Main counting functions
# ---------------------------------------------------------------------------


def _count_loop_body_split(
    operations: list[Operation],
    symbol_map: dict[str, sp.Expr] | None = None,
) -> tuple[sp.Expr, sp.Expr]:
    """Split loop body qubit count into (persistent, reusable).

    persistent: qubits that accumulate with each iteration
        (QInitOperation, non-clean calls)
    reusable: qubits from clean calls that can be reused across iterations
        (max watermark, counted once)

    Returns:
        (persistent, reusable) tuple of sympy expressions.
    """
    from qamomile.circuit.ir.block_value import BlockValue

    if symbol_map is None:
        symbol_map = {}

    persistent: sp.Expr = sp.Integer(0)
    reusable: sp.Expr = sp.Integer(0)

    for op in operations:
        match op:
            case BinOp():
                _register_binop(op, symbol_map)

            case QInitOperation():
                persistent += _count_qinit(op, symbol_map)  # type: ignore

            case CallBlockOperation():
                called_block, inner_map = _build_call_block_inner_map(
                    op, symbol_map
                )
                if called_block is not None:
                    inner_alloc = _count_from_operations(
                        called_block.operations, inner_map
                    )
                    if _is_clean_call(called_block):
                        reusable = sp.Max(reusable, inner_alloc)
                    else:
                        persistent += inner_alloc  # type: ignore

            case ControlledUOperation():
                controlled_block, inner_map = _build_controlled_u_inner_map(
                    op, symbol_map
                )
                if controlled_block is not None:
                    inner_alloc = _count_from_operations(
                        controlled_block.operations, inner_map
                    )
                    if _is_clean_call(controlled_block):
                        reusable = sp.Max(reusable, inner_alloc)
                    else:
                        persistent += inner_alloc  # type: ignore

            case CompositeGateOperation():
                impl = op.implementation
                meta_ancilla: sp.Expr = sp.Integer(0)
                impl_alloc: sp.Expr = sp.Integer(0)

                if isinstance(impl, BlockValue):
                    impl_alloc = _count_from_operations(impl.operations, symbol_map)

                if op.resource_metadata is not None:
                    meta_ancilla = sp.Integer(op.resource_metadata.ancilla_qubits)

                total_alloc = impl_alloc + meta_ancilla

                if isinstance(impl, BlockValue) and _is_clean_call(impl):
                    reusable = sp.Max(reusable, total_alloc)
                elif impl is None and op.resource_metadata is not None:
                    # Stub with metadata only: ancilla are reusable by definition
                    reusable = sp.Max(reusable, meta_ancilla)
                else:
                    persistent += total_alloc  # type: ignore

            case ForOperation():
                inner_p, inner_r = _count_loop_body_split(
                    op.operations, symbol_map
                )
                iterations = _compute_for_iterations(op, symbol_map)
                reusable_factor = sp.Piecewise(
                    (sp.Integer(1), sp.Gt(iterations, 0)),
                    (sp.Integer(0), True),
                )
                persistent += inner_p * iterations  # type: ignore
                reusable = sp.Max(reusable, inner_r * reusable_factor)

            case WhileOperation():
                inner_p, inner_r = _count_loop_body_split(
                    op.operations, symbol_map
                )
                persistent += inner_p * WHILE_SYMBOL  # type: ignore
                reusable = sp.Max(reusable, inner_r)

            case IfOperation():
                true_p, true_r = _count_loop_body_split(
                    op.true_operations, symbol_map
                )
                false_p, false_r = _count_loop_body_split(
                    op.false_operations, symbol_map
                )
                persistent += sp.Max(true_p, false_p)  # type: ignore
                reusable = sp.Max(reusable, true_r, false_r)

            case ForItemsOperation():
                inner_p, inner_r = _count_loop_body_split(
                    op.operations, symbol_map
                )
                cardinality = _resolve_for_items_cardinality(op)
                persistent += inner_p * cardinality  # type: ignore
                reusable = sp.Max(reusable, inner_r)

            case _:
                continue

    return (
        _strip_nonneg_max(sp.simplify(persistent)),
        _strip_nonneg_max(sp.simplify(reusable)),
    )


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
    from qamomile.circuit.ir.block_value import BlockValue

    if symbol_map is None:
        symbol_map = {}

    count: sp.Expr = sp.Integer(0)

    for op in operations:
        match op:
            case BinOp():
                _register_binop(op, symbol_map)

            case QInitOperation():
                count += _count_qinit(op, symbol_map)  # type: ignore

            case ForOperation():
                persistent, reusable = _count_loop_body_split(
                    op.operations, symbol_map
                )
                iterations = _compute_for_iterations(op, symbol_map)
                reusable_factor = sp.Piecewise(
                    (sp.Integer(1), sp.Gt(iterations, 0)),
                    (sp.Integer(0), True),
                )
                count += persistent * iterations + reusable * reusable_factor  # type: ignore

            case WhileOperation():
                persistent, reusable = _count_loop_body_split(
                    op.operations, symbol_map
                )
                # WHILE_SYMBOL is always positive, so reusable is counted once
                count += persistent * WHILE_SYMBOL + reusable  # type: ignore

            case IfOperation():
                # Take maximum of both branches
                true_count = _count_from_operations(op.true_operations, symbol_map)
                false_count = _count_from_operations(op.false_operations, symbol_map)
                count += sp.Max(true_count, false_count)  # type: ignore

            case CallBlockOperation():
                called_block, inner_map = _build_call_block_inner_map(
                    op, symbol_map
                )
                if called_block is not None:
                    count += _count_from_operations(called_block.operations, inner_map)  # type: ignore

            case ControlledUOperation():
                controlled_block, inner_map = _build_controlled_u_inner_map(
                    op, symbol_map
                )
                if controlled_block is not None:
                    count += _count_from_operations(controlled_block.operations, inner_map)  # type: ignore

            case ForItemsOperation():
                persistent, reusable = _count_loop_body_split(
                    op.operations, symbol_map
                )
                cardinality = _resolve_for_items_cardinality(op)
                # cardinality is always positive, so reusable is counted once
                count += persistent * cardinality + reusable  # type: ignore

            case CompositeGateOperation():
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
