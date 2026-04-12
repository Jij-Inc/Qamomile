"""Gate counting for quantum circuits.

This module provides algebraic gate counting using SymPy expressions,
allowing resource estimates to depend on problem size parameters.
"""

from __future__ import annotations

import warnings
from typing import Any

import sympy as sp
from sympy import Sum

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.operation.composite_gate import CompositeGateOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    GateOperation,
)
from qamomile.circuit.ir.operation.operation import Operation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp

from ._catalog import (
    classify_controlled_u,
    classify_gate,
    classify_pauli_evolve,
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
from ._gate_count import GateCount
from ._loop_executor import symbolic_iterations
from ._resolver import ExprResolver

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

    Uses the variable transformation ``loop_var = start + step * k`` where
    ``k`` ranges from ``0`` to ``iterations - 1``, matching Python ``range()``
    semantics for any step value.
    """
    iterations = symbolic_iterations(start, stop, step)
    k = sp.Dummy("k", integer=True, nonneg=True)

    def _sum_field(expr: sp.Expr) -> sp.Expr:
        transformed = expr.subs(loop_var, start + step * k)
        result = Sum(transformed, (k, 0, iterations - 1)).doit()
        return result  # type: ignore[return-value]

    return GateCount(
        total=_sum_field(count.total),
        single_qubit=_sum_field(count.single_qubit),
        two_qubit=_sum_field(count.two_qubit),
        multi_qubit=_sum_field(count.multi_qubit),
        t_gates=_sum_field(count.t_gates),
        clifford_gates=_sum_field(count.clifford_gates),
        rotation_gates=_sum_field(count.rotation_gates),
        oracle_calls={
            name: _sum_field(val) for name, val in count.oracle_calls.items()
        },
        oracle_queries={
            name: _sum_field(val) for name, val in count.oracle_queries.items()
        },
    )


# ------------------------------------------------------------------ #
#  Core dispatch                                                      #
# ------------------------------------------------------------------ #


def _count_from_operations(
    operations: list[Operation],
    resolver: ExprResolver,
    num_controls: int | sp.Expr = 0,
    bindings: dict[str, Any] | None = None,
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
                count = count + _handle_for(op, resolver, num_controls, bindings)

            case WhileOperation():
                child, trip_count = build_while_scope(op, resolver)
                inner = _count_from_operations(
                    op.operations,
                    child,
                    num_controls,
                    bindings,
                )
                count = count + inner * trip_count

            case IfOperation():
                true_child, false_child = build_if_scopes(op, resolver)
                true_count = _count_from_operations(
                    op.true_operations,
                    true_child,
                    num_controls,
                    bindings,
                )
                false_count = _count_from_operations(
                    op.false_operations,
                    false_child,
                    num_controls,
                    bindings,
                )
                count = count + true_count.max(false_count)

            case ForItemsOperation():
                child = build_for_items_scope(op, resolver)
                inner = _count_from_operations(
                    op.operations,
                    child,
                    num_controls,
                    bindings,
                )
                cardinality = resolve_for_items_cardinality(op)
                count = count + inner * cardinality

            case CallBlockOperation():
                count = count + _handle_call(op, resolver, num_controls, bindings)

            case ControlledUOperation():
                nc, nt = resolve_controlled_u(op, resolver)
                count = count + classify_controlled_u(nc, nt)

            case CompositeGateOperation():
                count = count + _handle_composite(
                    op,
                    resolver,
                    num_controls,
                )

            case PauliEvolveOp():
                count = count + _handle_pauli_evolve(op, bindings)

            case HasNestedOps():
                warnings.warn(
                    f"Unhandled control flow type {type(op).__name__} "
                    f"in gate counting; its gates will not be counted.",
                    stacklevel=2,
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
    bindings: dict[str, Any] | None = None,
) -> GateCount:
    """Handle ForOperation: multiply or Sum depending on loop-var dependency."""
    if len(op.operands) < 2:
        # Degenerate — no bounds
        return GateCount.zero()

    child, start, stop, step, loop_sym = build_for_loop_scope(op, resolver)

    inner = _count_from_operations(op.operations, child, num_controls, bindings)

    # Check if inner count depends on the loop variable
    all_free: set[sp.Symbol] = set()
    for attr in ("total", "two_qubit", "multi_qubit"):
        all_free |= getattr(inner, attr).free_symbols  # type: ignore[arg-type]
    for val in inner.oracle_calls.values():
        all_free |= val.free_symbols  # type: ignore[arg-type]
    for val in inner.oracle_queries.values():
        all_free |= val.free_symbols  # type: ignore[arg-type]

    if loop_sym in all_free:
        return _apply_sum_to_count(inner, loop_sym, start, stop, step)

    iterations = symbolic_iterations(start, stop, step)
    return inner * iterations


# ------------------------------------------------------------------ #
#  CallBlockOperation handler                                         #
# ------------------------------------------------------------------ #


def _handle_call(
    op: CallBlockOperation,
    resolver: ExprResolver,
    num_controls: int | sp.Expr,
    bindings: dict[str, Any] | None = None,
) -> GateCount:
    """Handle CallBlockOperation: recurse into callee with mapped scope."""
    called_block = op.block
    if not isinstance(called_block, Block):
        return GateCount.zero()  # type: ignore[unreachable]

    child = resolver.call_child_scope(op)
    return _count_from_operations(
        called_block.operations,
        child,
        num_controls,
        bindings,
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
        assert res.metadata is not None
        gc = extract_gate_count_from_metadata(res.metadata)
        if res.is_stub and res.oracle_name:
            gc.oracle_calls[res.oracle_name] = sp.Integer(1)
            if res.query_complexity is not None:
                gc.oracle_queries[res.oracle_name] = sp.Integer(
                    res.query_complexity,
                )
        return gc

    if res.kind == "implementation":
        assert res.impl_block is not None
        assert res.impl_resolver is not None
        return _count_from_operations(
            res.impl_block.operations,
            res.impl_resolver,
            num_controls,
        )

    if res.kind == "qft_iqft":
        assert res.n_qubits is not None
        return qft_iqft_gate_count(res.n_qubits)

    # error
    raise ValueError(res.error_message)


# ------------------------------------------------------------------ #
#  PauliEvolveOp handler                                             #
# ------------------------------------------------------------------ #


def _handle_pauli_evolve(
    op: PauliEvolveOp,
    bindings: dict[str, Any] | None,
) -> GateCount:
    """Handle PauliEvolveOp: compute gate count from Hamiltonian structure.

    Requires the Hamiltonian to be available via bindings. If not bound,
    emits a warning and returns zero (consistent with prior behavior).
    """
    import qamomile.observable as qm_o

    if bindings is None:
        warnings.warn(
            "PauliEvolveOp gate count requires bindings with Hamiltonian. "
            "Pass bindings to count_gates() for accurate counts.",
            stacklevel=4,
        )
        return GateCount.zero()

    # Try to resolve Hamiltonian from bindings
    obs_value = op.observable
    hamiltonian = None
    if hasattr(obs_value, "name") and obs_value.name in bindings:
        hamiltonian = bindings[obs_value.name]
    if hamiltonian is None and hasattr(obs_value, "uuid"):
        hamiltonian = bindings.get(obs_value.uuid)

    if not isinstance(hamiltonian, qm_o.Hamiltonian):
        warnings.warn(
            "PauliEvolveOp gate count requires a bound Hamiltonian. "
            "Gate counts for this operation will be missing.",
            stacklevel=4,
        )
        return GateCount.zero()

    return classify_pauli_evolve(hamiltonian)


# ------------------------------------------------------------------ #
#  Public entry point                                                 #
# ------------------------------------------------------------------ #


def count_gates(
    block: Block | list[Operation],
    bindings: dict[str, Any] | None = None,
) -> GateCount:
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
    - PauliEvolveOp: Counts decomposition gates (requires bindings)

    Args:
        block: Block or list of Operations to analyze
        bindings: Optional parameter bindings. Required for PauliEvolveOp
            gate counting (must contain the Hamiltonian).

    Returns:
        GateCount with total, single_qubit, two_qubit, t_gates, clifford_gates

    Example:
        >>> from qamomile.circuit.estimator import count_gates
        >>> count = count_gates(my_circuit.block)
        >>> print(count.total)  # e.g., "2*n + 5"
        >>> print(count.t_gates)  # e.g., "n"
    """
    if isinstance(block, Block):
        block_ref = block
        ops = block.operations
    else:
        block_ref = None
        ops = block

    resolver = ExprResolver(block=block_ref)
    return _count_from_operations(ops, resolver, bindings=bindings)
