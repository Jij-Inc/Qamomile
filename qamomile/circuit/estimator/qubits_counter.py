"""Qubit resource counting for BlockValue.

This module counts the number of qubits required by a quantum circuit,
using ExprResolver for value resolution and shared engine helpers for
control flow scoping.
"""

from __future__ import annotations

from typing import overload

import sympy as sp

from qamomile.circuit.ir.block_value import BlockValue
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
from qamomile.circuit.ir.value import ArrayValue

from ._engine import (
    build_for_items_scope,
    build_for_loop_scope,
    build_if_scopes,
    build_while_scope,
    resolve_for_items_cardinality,
)
from ._loop_executor import symbolic_iterations
from ._resolver import ExprResolver
from ._utils import _strip_nonneg_max

WHILE_SYMBOL = sp.Symbol("|while|", integer=True, positive=True)


# ------------------------------------------------------------------ #
#  QInit / input qubit counting                                       #
# ------------------------------------------------------------------ #


def _count_qinit(op: QInitOperation, resolver: ExprResolver) -> sp.Expr:
    """Count qubits from a QInitOperation.

    Returns:
        Number of qubits as a sympy expression (may contain symbols for
        parametric dimensions).
    """
    result = op.results[0]

    # Array of qubits
    if isinstance(result, ArrayValue) and isinstance(result.type, QubitType):
        el_count: sp.Expr = sp.Integer(1)
        for dim in result.shape:
            el_count *= resolver.resolve(dim)
        return el_count

    # Single qubit
    if isinstance(result.type, QubitType):
        return sp.Integer(1)

    return sp.Integer(0)


def _count_input_qubits(
    input_values: list,
    resolver: ExprResolver,
) -> sp.Expr:
    """Count qubit-typed values in BlockValue.input_values.

    Only used at the top level to count qubits passed as function arguments.
    """
    count: sp.Expr = sp.Integer(0)
    for v in input_values:
        if isinstance(v, ArrayValue) and isinstance(v.type, QubitType):
            el_count: sp.Expr = sp.Integer(1)
            for dim in v.shape:
                el_count *= resolver.resolve(dim)
            count += el_count
        elif isinstance(v.type, QubitType):
            count += sp.Integer(1)
    return count


# ------------------------------------------------------------------ #
#  Clean call detection                                               #
# ------------------------------------------------------------------ #


def _is_clean_call(block: BlockValue) -> bool:
    """Check if a BlockValue is a 'clean call'.

    A clean call returns only the qubits it received as inputs.
    Any internally allocated qubits (ancillas) are freed before return,
    meaning ancilla qubits can be reused across loop iterations.

    Conservative: if a cast/alias produces a new logical_id, this returns
    False (safe fallback — will overcount, never undercount).
    """
    input_logical_ids = {v.logical_id for v in block.input_values}
    for rv in block.return_values:
        if rv.type.is_quantum():
            if rv.logical_id not in input_logical_ids:
                return False
    return True


# ------------------------------------------------------------------ #
#  ControlledU child resolver                                         #
# ------------------------------------------------------------------ #


def _build_controlled_u_child_resolver(
    op: ControlledUOperation,
    resolver: ExprResolver,
) -> tuple[BlockValue | None, ExprResolver | None]:
    """Build child resolver for ControlledUOperation's inner block.

    Handles both index_spec mode (qubit operands identified by index)
    and the default mode (target_operands mapping).

    Returns:
        (controlled_block, child_resolver) or (None, None).
    """

    controlled_block = op.block
    if not isinstance(controlled_block, BlockValue):
        return None, None

    extra: dict[str, sp.Expr] = {}
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
                        for df, da in zip(formal_input.shape, actual_arg.shape):
                            extra[df.uuid] = resolver.resolve(da)
                    param_idx += 1
    else:
        target_operands = op.target_operands
        for formal_idx, formal_input in enumerate(controlled_block.input_values):
            if formal_idx < len(target_operands):
                actual_arg = target_operands[formal_idx]
                if isinstance(actual_arg, ArrayValue) and isinstance(
                    formal_input, ArrayValue
                ):
                    for df, da in zip(formal_input.shape, actual_arg.shape):
                        extra[df.uuid] = resolver.resolve(da)

    ctx = resolver.context
    ctx.update(extra)
    child = ExprResolver(
        block=controlled_block,
        context=ctx,
        loop_var_names=resolver.loop_var_names,
        parent_blocks=[],
    )
    return controlled_block, child


# ------------------------------------------------------------------ #
#  CompositeGate qubit counting                                       #
# ------------------------------------------------------------------ #


def _count_composite_split(
    op: CompositeGateOperation,
    resolver: ExprResolver,
) -> tuple[sp.Expr, bool]:
    """Count qubits from a CompositeGateOperation.

    Returns (alloc, is_reusable):
        alloc: Total qubit allocation (implementation + metadata ancilla)
        is_reusable: True if the allocation can be reused across loop iterations
    """

    impl = op.implementation
    meta_ancilla: sp.Expr = sp.Integer(0)
    impl_alloc: sp.Expr = sp.Integer(0)

    if isinstance(impl, BlockValue):
        impl_alloc = _count_from_operations(impl.operations, resolver)

    if op.resource_metadata is not None:
        meta_ancilla = sp.Integer(op.resource_metadata.ancilla_qubits)

    total_alloc = impl_alloc + meta_ancilla

    if isinstance(impl, BlockValue) and _is_clean_call(impl):
        return total_alloc, True
    if impl is None and op.resource_metadata is not None:
        # Stub with metadata only: ancilla are reusable by definition
        return meta_ancilla, True
    return total_alloc, False


def _count_composite_total(
    op: CompositeGateOperation,
    resolver: ExprResolver,
) -> sp.Expr:
    """Count qubits from a CompositeGateOperation (non-split)."""

    count: sp.Expr = sp.Integer(0)
    impl = op.implementation
    if isinstance(impl, BlockValue):
        count += _count_from_operations(impl.operations, resolver)
    if op.resource_metadata is not None:
        count += sp.Integer(op.resource_metadata.ancilla_qubits)
    return count


# ------------------------------------------------------------------ #
#  Loop body split counting                                           #
# ------------------------------------------------------------------ #


def _count_loop_body_split(
    operations: list[Operation],
    resolver: ExprResolver,
) -> tuple[sp.Expr, sp.Expr]:
    """Split loop body qubit count into (persistent, reusable).

    persistent: qubits that accumulate with each iteration
        (QInitOperation, non-clean calls)
    reusable: qubits from clean calls that can be reused across iterations
        (max watermark, counted once)

    Returns:
        (persistent, reusable) tuple of sympy expressions.
    """

    persistent: sp.Expr = sp.Integer(0)
    reusable: sp.Expr = sp.Integer(0)

    for op in operations:
        match op:
            case QInitOperation():
                persistent += _count_qinit(op, resolver)  # type: ignore

            case CallBlockOperation():
                called_block = op.operands[0]
                if isinstance(called_block, BlockValue):
                    child = resolver.call_child_scope(op)
                    inner_alloc = _count_from_operations(called_block.operations, child)
                    if _is_clean_call(called_block):
                        reusable = sp.Max(reusable, inner_alloc)
                    else:
                        persistent += inner_alloc  # type: ignore

            case ControlledUOperation():
                controlled_block, child = _build_controlled_u_child_resolver(
                    op, resolver
                )
                if controlled_block is not None and child is not None:
                    inner_alloc = _count_from_operations(
                        controlled_block.operations, child
                    )
                    if _is_clean_call(controlled_block):
                        reusable = sp.Max(reusable, inner_alloc)
                    else:
                        persistent += inner_alloc  # type: ignore

            case CompositeGateOperation():
                alloc, is_reusable = _count_composite_split(op, resolver)
                if is_reusable:
                    reusable = sp.Max(reusable, alloc)
                else:
                    persistent += alloc  # type: ignore

            case ForOperation():
                if len(op.operands) < 2:
                    continue
                child, start, stop, step, _ = build_for_loop_scope(op, resolver)
                inner_p, inner_r = _count_loop_body_split(op.operations, child)
                iterations = symbolic_iterations(start, stop, step)
                reusable_factor = sp.Piecewise(
                    (sp.Integer(1), sp.Gt(iterations, 0)),
                    (sp.Integer(0), True),
                )
                persistent += inner_p * iterations  # type: ignore
                reusable = sp.Max(reusable, inner_r * reusable_factor)

            case WhileOperation():
                child, _ = build_while_scope(op, resolver)
                inner_p, inner_r = _count_loop_body_split(op.operations, child)
                persistent += inner_p * WHILE_SYMBOL  # type: ignore
                reusable = sp.Max(reusable, inner_r)

            case IfOperation():
                true_child, false_child = build_if_scopes(op, resolver)
                true_p, true_r = _count_loop_body_split(op.true_operations, true_child)
                false_p, false_r = _count_loop_body_split(
                    op.false_operations, false_child
                )
                persistent += sp.Max(true_p, false_p)  # type: ignore
                reusable = sp.Max(reusable, true_r, false_r)

            case ForItemsOperation():
                child = build_for_items_scope(op, resolver)
                inner_p, inner_r = _count_loop_body_split(op.operations, child)
                cardinality = resolve_for_items_cardinality(op)
                persistent += inner_p * cardinality  # type: ignore
                reusable = sp.Max(reusable, inner_r)

            case _:
                continue

    return (
        _strip_nonneg_max(sp.simplify(persistent)),
        _strip_nonneg_max(sp.simplify(reusable)),
    )


# ------------------------------------------------------------------ #
#  Main counting function                                             #
# ------------------------------------------------------------------ #


def _count_from_operations(
    operations: list[Operation],
    resolver: ExprResolver,
) -> sp.Expr:
    """Count qubits from a list of operations.

    Returns:
        Total qubit count as a sympy expression.
    """

    count: sp.Expr = sp.Integer(0)

    for op in operations:
        match op:
            case QInitOperation():
                count += _count_qinit(op, resolver)  # type: ignore

            case ForOperation():
                if len(op.operands) < 2:
                    continue
                child, start, stop, step, _ = build_for_loop_scope(op, resolver)
                persistent, reusable = _count_loop_body_split(op.operations, child)
                iterations = symbolic_iterations(start, stop, step)
                reusable_factor = sp.Piecewise(
                    (sp.Integer(1), sp.Gt(iterations, 0)),
                    (sp.Integer(0), True),
                )
                count += persistent * iterations + reusable * reusable_factor  # type: ignore

            case WhileOperation():
                child, _ = build_while_scope(op, resolver)
                persistent, reusable = _count_loop_body_split(op.operations, child)
                count += persistent * WHILE_SYMBOL + reusable  # type: ignore

            case IfOperation():
                true_child, false_child = build_if_scopes(op, resolver)
                true_count = _count_from_operations(op.true_operations, true_child)
                false_count = _count_from_operations(op.false_operations, false_child)
                count += sp.Max(true_count, false_count)  # type: ignore

            case CallBlockOperation():
                called_block = op.operands[0]
                if isinstance(called_block, BlockValue):
                    child = resolver.call_child_scope(op)
                    count += _count_from_operations(  # type: ignore
                        called_block.operations, child
                    )

            case ControlledUOperation():
                controlled_block, child = _build_controlled_u_child_resolver(
                    op, resolver
                )
                if controlled_block is not None and child is not None:
                    count += _count_from_operations(  # type: ignore
                        controlled_block.operations, child
                    )

            case ForItemsOperation():
                child = build_for_items_scope(op, resolver)
                persistent, reusable = _count_loop_body_split(op.operations, child)
                cardinality = resolve_for_items_cardinality(op)
                count += persistent * cardinality + reusable  # type: ignore

            case CompositeGateOperation():
                count += _count_composite_total(op, resolver)  # type: ignore

            case _:
                continue

    return sp.simplify(count)


# ------------------------------------------------------------------ #
#  Public entry point                                                 #
# ------------------------------------------------------------------ #


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

    if isinstance(block, BlockValue):
        resolver = ExprResolver(block=block)
        input_qubits = _count_input_qubits(block.input_values, resolver)
        ops_qubits = _count_from_operations(block.operations, resolver)
        return sp.simplify(input_qubits + ops_qubits)
    resolver = ExprResolver()
    return _count_from_operations(block, resolver)
