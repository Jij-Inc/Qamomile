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
from qamomile.circuit.ir.operation.callable import (
    InvokeOperation,
)
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
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import Operation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.value import ArrayValue

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
    resolve_controlled_u,
    resolve_for_items_cardinality,
    resolve_invoke_resource,
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
    backend: str | None = None,
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
                count = count + _handle_for(
                    op,
                    resolver,
                    num_controls,
                    bindings,
                    backend,
                )

            case WhileOperation():
                child, trip_count = build_while_scope(op, resolver)
                inner = _count_from_operations(
                    op.operations,
                    child,
                    num_controls,
                    bindings,
                    backend,
                )
                count = count + inner * trip_count

            case IfOperation():
                true_child, false_child = build_if_scopes(op, resolver)
                true_count = _count_from_operations(
                    op.true_operations,
                    true_child,
                    num_controls,
                    bindings,
                    backend,
                )
                false_count = _count_from_operations(
                    op.false_operations,
                    false_child,
                    num_controls,
                    bindings,
                    backend,
                )
                count = count + true_count.max(false_count)

            case ForItemsOperation():
                child = build_for_items_scope(op, resolver)
                inner = _count_from_operations(
                    op.operations,
                    child,
                    num_controls,
                    bindings,
                    backend,
                )
                cardinality = resolve_for_items_cardinality(op)
                count = count + inner * cardinality

            case InvokeOperation():
                count = count + _handle_invoke(
                    op,
                    resolver,
                    num_controls,
                    bindings,
                    backend,
                )

            case ControlledUOperation():
                nc, nt = resolve_controlled_u(op, resolver)
                count = count + classify_controlled_u(nc, nt)

            case InverseBlockOperation():
                count = count + _handle_inverse_block(
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
    backend: str | None = None,
) -> GateCount:
    """Handle ForOperation: multiply or Sum depending on loop-var dependency."""
    if len(op.operands) < 2:
        # Degenerate — no bounds
        return GateCount.zero()

    child, start, stop, step, loop_sym = build_for_loop_scope(op, resolver)

    inner = _count_from_operations(
        op.operations,
        child,
        num_controls,
        bindings,
        backend,
    )

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


def _handle_invoke(
    op: InvokeOperation,
    resolver: ExprResolver,
    num_controls: int | sp.Expr,
    bindings: dict[str, Any] | None = None,
    backend: str | None = None,
) -> GateCount:
    """Handle InvokeOperation through resource metadata or an embedded body.

    Args:
        op (InvokeOperation): Invocation to estimate.
        resolver (ExprResolver): Resolver for the current scope.
        num_controls (int | sp.Expr): Number of enclosing controls.
        bindings (dict[str, Any] | None): Optional compile-time bindings.

    Returns:
        GateCount: Estimated gate count.

    Raises:
        ValueError: If the invocation has neither metadata nor body.
    """
    return _handle_boxed_callable(op, resolver, num_controls, bindings, backend)


# ------------------------------------------------------------------ #
#  Boxed callable handler                                             #
# ------------------------------------------------------------------ #


def _handle_boxed_callable(
    op: InvokeOperation,
    resolver: ExprResolver,
    num_controls: int | sp.Expr,
    bindings: dict[str, Any] | None = None,
    backend: str | None = None,
) -> GateCount:
    """Handle a boxed invocation: metadata > implementation > formula."""
    res = resolve_invoke_resource(op, resolver, backend=backend)

    if res.kind == "metadata":
        assert res.metadata is not None
        gc = extract_gate_count_from_metadata(res.metadata)
        if res.is_opaque and res.oracle_name:
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
            bindings,
            backend,
        )

    if res.kind == "qft_iqft":
        assert res.n_qubits is not None
        return qft_iqft_gate_count(res.n_qubits)

    # error
    raise ValueError(res.error_message)


def _handle_inverse_block(
    op: InverseBlockOperation,
    resolver: ExprResolver,
    num_controls: int | sp.Expr,
) -> GateCount:
    """Count an inverse block through its fallback implementation.

    Args:
        op (InverseBlockOperation): Inverse block operation to count.
        resolver (ExprResolver): Resolver for the current estimator scope.
        num_controls (int | sp.Expr): Number of controls already applied by
            the enclosing traversal.

    Returns:
        GateCount: Gate count for the inverse implementation block.

    Raises:
        ValueError: If the inverse operation has no implementation block.
    """
    impl = op.implementation_block
    if not isinstance(impl, Block):
        raise ValueError(
            f"Cannot estimate resources for InverseBlockOperation "
            f"'{op.name}': no implementation block is available."
        )

    extra: dict[str, sp.Expr] = {}
    quantum_actuals = iter(op.target_qubits)
    parameter_actuals = iter(op.parameters)
    for formal in impl.input_values:
        if formal.type.is_quantum():
            actual = next(quantum_actuals, None)
            if (
                isinstance(formal, ArrayValue)
                and isinstance(actual, ArrayValue)
                and formal.shape
                and actual.shape
            ):
                extra[formal.shape[0].uuid] = resolver.resolve(actual.shape[0])
            continue

        actual = next(parameter_actuals, None)
        if actual is not None:
            extra[formal.uuid] = resolver.resolve(actual)

    ctx = resolver.context
    ctx.update(extra)
    child = ExprResolver(
        block=impl,
        context=ctx,
        loop_var_names=resolver.loop_var_names,
        parent_blocks=[],
    )
    return _count_from_operations(
        impl.operations,
        child,
        num_controls + op.num_control_qubits,
    )


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
    if hasattr(obs_value, "name") and obs_value.name and obs_value.name in bindings:
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
    backend: str | None = None,
) -> GateCount:
    """Count gates in a quantum circuit.

    This function analyzes operations and returns algebraic gate counts
    using SymPy expressions. Counts may contain symbols for parametric
    problem sizes (e.g., loop bounds, array dimensions).

    Supports:
    - GateOperation: Single gate counts
    - ForOperation: Multiplies inner count by iterations
    - IfOperation: Takes maximum of branches
    - InvokeOperation: Counts callable bodies or callable resource metadata
    - ControlledUOperation: Counts as a single opaque gate
    - PauliEvolveOp: Counts decomposition gates (requires bindings)

    Args:
        block: Block or list of Operations to analyze
        bindings: Optional parameter bindings. Required for PauliEvolveOp
            gate counting (must contain the Hamiltonian).
        backend: Optional backend name used to select backend-specific
            callable implementation resources or bodies.

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
    return _count_from_operations(ops, resolver, bindings=bindings, backend=backend)
