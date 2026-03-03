"""Gate counting for quantum circuits.

This module provides algebraic gate counting using SymPy expressions,
allowing resource estimates to depend on problem size parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import sympy as sp
from sympy import Sum

from qamomile.circuit.estimator._utils import _strip_nonneg_max

from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.operation.composite_gate import CompositeGateOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    GateOperation,
)
from qamomile.circuit.ir.operation.operation import Operation

from ._catalog import (
    classify_controlled_u,
    classify_gate,
    extract_gate_count_from_metadata,
    qft_iqft_gate_count,
)
from ._engine import (
    build_for_items_scope,
    build_for_loop_scope,
    build_if_scopes,
    build_while_scope,
    resolve_composite_gate,
    resolve_controlled_u,
    resolve_for_items_cardinality,
)
from ._resolver import ExprResolver

if TYPE_CHECKING:
    from qamomile.circuit.ir.block import Block
    from qamomile.circuit.ir.block_value import BlockValue


# ------------------------------------------------------------------ #
#  GateCount dataclass (public API — unchanged)                       #
# ------------------------------------------------------------------ #


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
    multi_qubit: sp.Expr
    t_gates: sp.Expr
    clifford_gates: sp.Expr
    rotation_gates: sp.Expr
    oracle_calls: dict[str, sp.Expr] = field(default_factory=dict)
    oracle_queries: dict[str, sp.Expr] = field(default_factory=dict)

    def __add__(self, other: GateCount) -> GateCount:
        """Add two gate counts together."""
        merged_oracle = dict(self.oracle_calls)
        for name, count in other.oracle_calls.items():
            if name in merged_oracle:
                merged_oracle[name] = merged_oracle[name] + count
            else:
                merged_oracle[name] = count
        merged_queries = dict(self.oracle_queries)
        for name, count in other.oracle_queries.items():
            if name in merged_queries:
                merged_queries[name] = merged_queries[name] + count
            else:
                merged_queries[name] = count
        return GateCount(
            total=self.total + other.total,
            single_qubit=self.single_qubit + other.single_qubit,
            two_qubit=self.two_qubit + other.two_qubit,
            multi_qubit=self.multi_qubit + other.multi_qubit,
            t_gates=self.t_gates + other.t_gates,
            clifford_gates=self.clifford_gates + other.clifford_gates,
            rotation_gates=self.rotation_gates + other.rotation_gates,
            oracle_calls=merged_oracle,
            oracle_queries=merged_queries,
        )

    def __mul__(self, factor: sp.Expr | int) -> GateCount:
        """Multiply gate counts by a factor."""
        factor_expr = sp.Integer(factor) if isinstance(factor, int) else factor
        return GateCount(
            total=self.total * factor_expr,
            single_qubit=self.single_qubit * factor_expr,
            two_qubit=self.two_qubit * factor_expr,
            multi_qubit=self.multi_qubit * factor_expr,
            t_gates=self.t_gates * factor_expr,
            clifford_gates=self.clifford_gates * factor_expr,
            rotation_gates=self.rotation_gates * factor_expr,
            oracle_calls={
                name: count * factor_expr for name, count in self.oracle_calls.items()
            },
            oracle_queries={
                name: count * factor_expr for name, count in self.oracle_queries.items()
            },
        )

    __rmul__ = __mul__

    def max(self, other: GateCount) -> GateCount:
        """Element-wise maximum of two gate counts."""

        def _merge_max(
            a: dict[str, sp.Expr], b: dict[str, sp.Expr]
        ) -> dict[str, sp.Expr]:
            merged: dict[str, sp.Expr] = {}
            for key in set(a.keys()) | set(b.keys()):
                if key in a and key in b:
                    merged[key] = sp.Max(a[key], b[key])
                elif key in a:
                    merged[key] = a[key]
                else:
                    merged[key] = b[key]
            return merged

        return GateCount(
            total=sp.Max(self.total, other.total),
            single_qubit=sp.Max(self.single_qubit, other.single_qubit),
            two_qubit=sp.Max(self.two_qubit, other.two_qubit),
            multi_qubit=sp.Max(self.multi_qubit, other.multi_qubit),
            t_gates=sp.Max(self.t_gates, other.t_gates),
            clifford_gates=sp.Max(self.clifford_gates, other.clifford_gates),
            rotation_gates=sp.Max(self.rotation_gates, other.rotation_gates),
            oracle_calls=_merge_max(self.oracle_calls, other.oracle_calls),
            oracle_queries=_merge_max(self.oracle_queries, other.oracle_queries),
        )

    def simplify(self) -> GateCount:
        """Simplify all SymPy expressions."""
        return GateCount(
            total=_strip_nonneg_max(sp.simplify(self.total)),
            single_qubit=_strip_nonneg_max(sp.simplify(self.single_qubit)),
            two_qubit=_strip_nonneg_max(sp.simplify(self.two_qubit)),
            multi_qubit=_strip_nonneg_max(sp.simplify(self.multi_qubit)),
            t_gates=_strip_nonneg_max(sp.simplify(self.t_gates)),
            clifford_gates=_strip_nonneg_max(sp.simplify(self.clifford_gates)),
            rotation_gates=_strip_nonneg_max(sp.simplify(self.rotation_gates)),
            oracle_calls={
                name: _strip_nonneg_max(sp.simplify(count))
                for name, count in self.oracle_calls.items()
            },
            oracle_queries={
                name: _strip_nonneg_max(sp.simplify(count))
                for name, count in self.oracle_queries.items()
            },
        )

    @staticmethod
    def zero() -> GateCount:
        """Return a zero gate count."""
        return GateCount(
            total=sp.Integer(0),
            single_qubit=sp.Integer(0),
            two_qubit=sp.Integer(0),
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=sp.Integer(0),
            rotation_gates=sp.Integer(0),
        )


# ------------------------------------------------------------------ #
#  SymPy Sum helper                                                   #
# ------------------------------------------------------------------ #


def _apply_sum_to_count(
    count: GateCount,
    loop_var: sp.Symbol,
    start: sp.Expr,
    stop: sp.Expr,
    step: sp.Expr = sp.Integer(1),
) -> GateCount:
    """Apply SymPy Sum to all fields of a GateCount.

    For forward loops (step > 0): Sum over ``(loop_var, start, stop-1)``
    For reverse loops (step < 0): Sum over ``(loop_var, stop+1, start)``
    """
    is_negative_step = False
    try:
        is_negative_step = bool(step < 0)
    except TypeError:
        pass  # symbolic step, assume positive

    if is_negative_step:
        lower = stop + 1
        upper = start
    else:
        lower = start
        upper = stop - 1

    return GateCount(
        total=Sum(count.total, (loop_var, lower, upper)).doit(),
        single_qubit=Sum(count.single_qubit, (loop_var, lower, upper)).doit(),
        two_qubit=Sum(count.two_qubit, (loop_var, lower, upper)).doit(),
        multi_qubit=Sum(count.multi_qubit, (loop_var, lower, upper)).doit(),
        t_gates=Sum(count.t_gates, (loop_var, lower, upper)).doit(),
        clifford_gates=Sum(count.clifford_gates, (loop_var, lower, upper)).doit(),
        rotation_gates=Sum(count.rotation_gates, (loop_var, lower, upper)).doit(),
        oracle_calls={
            name: Sum(val, (loop_var, lower, upper)).doit()
            for name, val in count.oracle_calls.items()
        },
        oracle_queries={
            name: Sum(val, (loop_var, lower, upper)).doit()
            for name, val in count.oracle_queries.items()
        },
    )


# ------------------------------------------------------------------ #
#  Core dispatch                                                      #
# ------------------------------------------------------------------ #


def _count_from_operations(
    operations: list[Operation],
    resolver: ExprResolver,
    num_controls: int | sp.Expr = 0,
) -> GateCount:
    """Count gates from a list of operations.

    Uses ExprResolver for value resolution and shared engine helpers
    for CompositeGate/ControlledU/loop resolution.
    """
    count = GateCount.zero()

    for op in operations:
        match op:
            case GateOperation():
                count = count + classify_gate(op, num_controls=num_controls)

            case ForOperation():
                count = count + _handle_for(op, resolver, num_controls)

            case WhileOperation():
                child, trip_count = build_while_scope(op, resolver)
                inner = _count_from_operations(
                    op.operations,
                    child,
                    num_controls,
                )
                count = count + inner * trip_count

            case IfOperation():
                true_child, false_child = build_if_scopes(op, resolver)
                true_count = _count_from_operations(
                    op.true_operations,
                    true_child,
                    num_controls,
                )
                false_count = _count_from_operations(
                    op.false_operations,
                    false_child,
                    num_controls,
                )
                count = count + true_count.max(false_count)

            case ForItemsOperation():
                child = build_for_items_scope(op, resolver)
                inner = _count_from_operations(
                    op.operations,
                    child,
                    num_controls,
                )
                cardinality = resolve_for_items_cardinality(op)
                count = count + inner * cardinality

            case CallBlockOperation():
                count = count + _handle_call(op, resolver, num_controls)

            case ControlledUOperation():
                nc, nt = resolve_controlled_u(op, resolver)
                count = count + classify_controlled_u(nc, nt)

            case CompositeGateOperation():
                count = count + _handle_composite(
                    op,
                    resolver,
                    num_controls,
                )

            case _:
                continue

    return count.simplify()


# ------------------------------------------------------------------ #
#  ForOperation handler                                               #
# ------------------------------------------------------------------ #


def _handle_for(
    op: ForOperation,
    resolver: ExprResolver,
    num_controls: int | sp.Expr,
) -> GateCount:
    """Handle ForOperation: multiply or Sum depending on loop-var dependency."""
    if len(op.operands) < 2:
        # Degenerate — no bounds
        return GateCount.zero()

    child, start, stop, step, loop_sym = build_for_loop_scope(op, resolver)

    inner = _count_from_operations(op.operations, child, num_controls)

    # Check if inner count depends on the loop variable
    all_free: set[sp.Symbol] = set()
    for attr in ("total", "two_qubit", "multi_qubit"):
        all_free |= getattr(inner, attr).free_symbols
    for val in inner.oracle_calls.values():
        all_free |= val.free_symbols
    for val in inner.oracle_queries.values():
        all_free |= val.free_symbols

    if loop_sym in all_free:
        return _apply_sum_to_count(inner, loop_sym, start, stop, step)

    iterations = (stop - start) / step
    return inner * iterations


# ------------------------------------------------------------------ #
#  CallBlockOperation handler                                         #
# ------------------------------------------------------------------ #


def _handle_call(
    op: CallBlockOperation,
    resolver: ExprResolver,
    num_controls: int | sp.Expr,
) -> GateCount:
    """Handle CallBlockOperation: recurse into callee with mapped scope."""
    from qamomile.circuit.ir.block_value import BlockValue

    called_block = op.operands[0]
    if not isinstance(called_block, BlockValue):
        return GateCount.zero()

    child = resolver.call_child_scope(op)
    return _count_from_operations(
        called_block.operations,
        child,
        num_controls,
    )


# ------------------------------------------------------------------ #
#  CompositeGateOperation handler                                     #
# ------------------------------------------------------------------ #


def _handle_composite(
    op: CompositeGateOperation,
    resolver: ExprResolver,
    num_controls: int | sp.Expr,
) -> GateCount:
    """Handle CompositeGateOperation: metadata > implementation > formula."""
    res = resolve_composite_gate(op, resolver)

    if res.kind == "metadata":
        gc = extract_gate_count_from_metadata(res.metadata)
        if res.is_stub and res.oracle_name:
            gc.oracle_calls[res.oracle_name] = sp.Integer(1)
            if res.query_complexity is not None:
                gc.oracle_queries[res.oracle_name] = sp.Integer(
                    res.query_complexity,
                )
        return gc

    if res.kind == "implementation":
        return _count_from_operations(
            res.impl_block.operations,
            res.impl_resolver,
            num_controls,
        )

    if res.kind == "qft_iqft":
        return qft_iqft_gate_count(res.n_qubits)

    # error
    raise ValueError(res.error_message)


# ------------------------------------------------------------------ #
#  Public entry point                                                 #
# ------------------------------------------------------------------ #


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
    - ControlledUOperation: Counts as a single opaque gate

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
        block_ref = block
        ops = block.operations
    else:
        block_ref = None
        ops = block

    resolver = ExprResolver(block=block_ref)
    return _count_from_operations(ops, resolver)
