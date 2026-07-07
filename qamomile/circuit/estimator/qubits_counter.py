"""Qubit resource counting for Block.

This module counts the number of qubits required by a quantum circuit,
using ExprResolver for value resolution and shared engine helpers for
control flow scoping.
"""

from __future__ import annotations

from typing import overload

import sympy as sp

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import (
    InvokeOperation,
    ResourceMetadata,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import ControlledUOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import Operation, QInitOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
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
    """Count qubit-typed values in Block.input_values.

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


def _is_clean_call(block: Block) -> bool:
    """Check if a Block is a 'clean call'.

    A clean call returns only the qubits it received as inputs.
    Any internally allocated qubits (ancillas) are freed before return,
    meaning ancilla qubits can be reused across loop iterations.

    Conservative: if a cast/alias produces a new logical_id, this returns
    False (safe fallback — will overcount, never undercount).
    """
    input_logical_ids = {v.logical_id for v in block.input_values}
    for rv in block.output_values:
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
) -> tuple[Block | None, ExprResolver | None]:
    """Build a child resolver for the inner block of a ``ControlledUOperation``.

    Walks the inner block's input signature against the op's
    ``target_operands`` so any ``ArrayValue`` size that the caller
    bound by passing a sized argument is added to the resolver
    context.  This lets downstream qubit-counting resolve the inner
    block's symbolic ``Vector[Qubit]`` sizes from the call site.

    Returns:
        (controlled_block, child_resolver) or (None, None).
    """

    controlled_block = op.block
    if not isinstance(controlled_block, Block):
        return None, None

    extra: dict[str, sp.Expr] = {}
    # ``op.target_operands`` includes both quantum sub-arg ``Value`` s
    # and classical ``Value`` s, but we only need shape propagation
    # for the ``Vector[Qubit]`` ones.  Filter both the formal inputs
    # and the actuals to the quantum subset so the zip pairs the
    # right things when the wrapped kernel's signature interleaves
    # classical and quantum params (e.g. ``def sub(theta, q: Vector[Qubit])``
    # -- the unfiltered zip used to pair ``theta`` formal with the
    # ``q`` actual, missing the actual ``q`` formal entirely and
    # leaving its symbolic shape unresolved).
    quantum_formals = [
        iv for iv in controlled_block.input_values if iv.type.is_quantum()
    ]
    quantum_actuals = [v for v in op.target_operands if v.type.is_quantum()]
    for formal_input, actual_arg in zip(quantum_formals, quantum_actuals):
        if isinstance(actual_arg, ArrayValue) and isinstance(formal_input, ArrayValue):
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
def _count_inverse_block_split(
    op: InverseBlockOperation,
    resolver: ExprResolver,
) -> tuple[sp.Expr, bool]:
    """Count reusable qubits from an inverse block implementation.

    Args:
        op (InverseBlockOperation): Inverse block operation to count.
        resolver (ExprResolver): Resolver for the current estimator scope.

    Returns:
        tuple[sp.Expr, bool]: Allocation count and whether it is reusable
            across loop iterations.
    """
    impl = op.implementation_block
    impl_alloc: sp.Expr = sp.Integer(0)
    if isinstance(impl, Block):
        impl_alloc = _count_from_operations(impl.operations, resolver)
        if _is_clean_call(impl):
            return impl_alloc, True
    return impl_alloc, False


def _count_invoke_split(
    op: InvokeOperation,
    resolver: ExprResolver,
    *,
    backend: str | None = None,
) -> tuple[sp.Expr, bool]:
    """Count reusable qubits from an InvokeOperation.

    Args:
        op (InvokeOperation): Invocation operation to count.
        resolver (ExprResolver): Resolver for the current estimator scope.
        backend (str | None): Optional backend name used to select
            backend-specific callable implementation resources or bodies.

    Returns:
        tuple[sp.Expr, bool]: Allocation count and whether it is reusable.
    """
    body = op.effective_body(backend=backend)
    resource = op.effective_resource(backend=backend)
    body_alloc: sp.Expr = sp.Integer(0)
    meta_ancilla: sp.Expr = sp.Integer(0)
    if isinstance(body, Block):
        child = resolver.call_child_scope(op, called_block=body)
        body_alloc = _count_from_operations(body.operations, child, backend=backend)
    if isinstance(resource, ResourceMetadata):
        meta_ancilla = sp.Integer(resource.ancilla_qubits)

    total = body_alloc + meta_ancilla
    if isinstance(body, Block) and _is_clean_call(body):
        return total, True
    if body is None and isinstance(resource, ResourceMetadata):
        return total, True
    if body is None and op.attrs.get("kind") == "oracle":
        return total, True
    return total, False


def _count_invoke_total(
    op: InvokeOperation,
    resolver: ExprResolver,
    *,
    backend: str | None = None,
) -> sp.Expr:
    """Count qubits from an InvokeOperation.

    Args:
        op (InvokeOperation): Invocation operation to count.
        resolver (ExprResolver): Resolver for the current estimator scope.
        backend (str | None): Optional backend name used to select
            backend-specific callable implementation resources or bodies.

    Returns:
        sp.Expr: Additional qubit allocation required by the invocation.
    """
    alloc, _ = _count_invoke_split(op, resolver, backend=backend)
    return alloc


def _count_inverse_block_total(
    op: InverseBlockOperation,
    resolver: ExprResolver,
) -> sp.Expr:
    """Count qubits from an inverse block implementation.

    Args:
        op (InverseBlockOperation): Inverse block operation to count.
        resolver (ExprResolver): Resolver for the current estimator scope.

    Returns:
        sp.Expr: Total additional qubit allocation required by the fallback
            implementation block.
    """
    impl = op.implementation_block
    if isinstance(impl, Block):
        return _count_from_operations(impl.operations, resolver)
    return sp.Integer(0)


# ------------------------------------------------------------------ #
#  Loop body split counting                                           #
# ------------------------------------------------------------------ #


def _count_loop_body_split(
    operations: list[Operation],
    resolver: ExprResolver,
    *,
    backend: str | None = None,
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

            case ControlledUOperation():
                controlled_block, child = _build_controlled_u_child_resolver(
                    op, resolver
                )
                if controlled_block is not None and child is not None:
                    inner_alloc = _count_from_operations(
                        controlled_block.operations,
                        child,
                        backend=backend,
                    )
                    if _is_clean_call(controlled_block):
                        reusable = sp.Max(reusable, inner_alloc)
                    else:
                        persistent += inner_alloc  # type: ignore

            case InvokeOperation():
                alloc, is_reusable = _count_invoke_split(
                    op,
                    resolver,
                    backend=backend,
                )
                if is_reusable:
                    reusable = sp.Max(reusable, alloc)
                else:
                    persistent += alloc  # type: ignore

            case InverseBlockOperation():
                alloc, is_reusable = _count_inverse_block_split(op, resolver)
                if is_reusable:
                    reusable = sp.Max(reusable, alloc)
                else:
                    persistent += alloc  # type: ignore

            case ForOperation():
                if len(op.operands) < 2:
                    continue
                child, start, stop, step, _ = build_for_loop_scope(op, resolver)
                inner_p, inner_r = _count_loop_body_split(
                    op.operations,
                    child,
                    backend=backend,
                )
                iterations = symbolic_iterations(start, stop, step)
                reusable_factor = sp.Piecewise(
                    (sp.Integer(1), sp.Gt(iterations, 0)),
                    (sp.Integer(0), True),
                )
                persistent += inner_p * iterations  # type: ignore
                reusable = sp.Max(reusable, inner_r * reusable_factor)

            case WhileOperation():
                child, _ = build_while_scope(op, resolver)
                inner_p, inner_r = _count_loop_body_split(
                    op.operations,
                    child,
                    backend=backend,
                )
                persistent += inner_p * WHILE_SYMBOL  # type: ignore
                reusable = sp.Max(reusable, inner_r)

            case IfOperation():
                true_child, false_child = build_if_scopes(op, resolver)
                true_p, true_r = _count_loop_body_split(
                    op.true_operations,
                    true_child,
                    backend=backend,
                )
                false_p, false_r = _count_loop_body_split(
                    op.false_operations,
                    false_child,
                    backend=backend,
                )
                persistent += sp.Max(true_p, false_p)  # type: ignore
                reusable = sp.Max(reusable, true_r, false_r)

            case ForItemsOperation():
                child = build_for_items_scope(op, resolver)
                inner_p, inner_r = _count_loop_body_split(
                    op.operations,
                    child,
                    backend=backend,
                )
                cardinality = resolve_for_items_cardinality(op)
                persistent += inner_p * cardinality  # type: ignore
                reusable = sp.Max(reusable, inner_r)

            case PauliEvolveOp():
                # PauliEvolveOp operates in-place on existing qubits;
                # no additional qubit allocation required.
                continue

            case HasNestedOps():
                import warnings

                warnings.warn(
                    f"Unhandled control flow type {type(op).__name__} "
                    f"in qubit counting; its qubits will not be counted.",
                    stacklevel=2,
                )

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
    *,
    backend: str | None = None,
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
                persistent, reusable = _count_loop_body_split(
                    op.operations,
                    child,
                    backend=backend,
                )
                iterations = symbolic_iterations(start, stop, step)
                reusable_factor = sp.Piecewise(
                    (sp.Integer(1), sp.Gt(iterations, 0)),
                    (sp.Integer(0), True),
                )
                count += persistent * iterations + reusable * reusable_factor  # type: ignore

            case WhileOperation():
                child, _ = build_while_scope(op, resolver)
                persistent, reusable = _count_loop_body_split(
                    op.operations,
                    child,
                    backend=backend,
                )
                count += persistent * WHILE_SYMBOL + reusable  # type: ignore

            case IfOperation():
                true_child, false_child = build_if_scopes(op, resolver)
                true_count = _count_from_operations(
                    op.true_operations,
                    true_child,
                    backend=backend,
                )
                false_count = _count_from_operations(
                    op.false_operations,
                    false_child,
                    backend=backend,
                )
                count += sp.Max(true_count, false_count)  # type: ignore

            case ControlledUOperation():
                controlled_block, child = _build_controlled_u_child_resolver(
                    op, resolver
                )
                if controlled_block is not None and child is not None:
                    count += _count_from_operations(  # type: ignore
                        controlled_block.operations,
                        child,
                        backend=backend,
                    )

            case ForItemsOperation():
                child = build_for_items_scope(op, resolver)
                persistent, reusable = _count_loop_body_split(
                    op.operations,
                    child,
                    backend=backend,
                )
                cardinality = resolve_for_items_cardinality(op)
                count += persistent * cardinality + reusable  # type: ignore

            case InvokeOperation():
                count += _count_invoke_total(op, resolver, backend=backend)  # type: ignore

            case InverseBlockOperation():
                count += _count_inverse_block_total(op, resolver)  # type: ignore

            case PauliEvolveOp():
                # PauliEvolveOp operates in-place; no new qubits.
                continue

            case HasNestedOps():
                import warnings

                warnings.warn(
                    f"Unhandled control flow type {type(op).__name__} "
                    f"in qubit counting; its qubits will not be counted.",
                    stacklevel=2,
                )

            case _:
                continue

    return sp.simplify(count)


# ------------------------------------------------------------------ #
#  Public entry point                                                 #
# ------------------------------------------------------------------ #


@overload
def qubits_counter(block: Block, *, backend: str | None = None) -> sp.Expr: ...


@overload
def qubits_counter(
    block: list[Operation],
    *,
    backend: str | None = None,
) -> sp.Expr: ...


@overload
def qubits_counter(
    block: Block | list[Operation],
    *,
    backend: str | None = None,
) -> sp.Expr: ...


def qubits_counter(
    block: Block | list[Operation],
    *,
    backend: str | None = None,
) -> sp.Expr:
    """Count the number of qubits required by a Block.

    This function analyzes the operations in a Block and counts
    the total number of qubits that need to be allocated. It handles:

    - QInitOperation: Counts single qubits and qubit arrays
    - ForOperation/WhileOperation: Counts inner resources (assumes uncomputation)
    - IfOperation: Takes the maximum of both branches
    - InvokeOperation: Counts callable bodies and callable resource metadata
    - ControlledUOperation: Recursively counts the unitary block

    Args:
        block: The Block to analyze.
        backend: Optional backend name used to select backend-specific
            callable implementation resources or bodies.

    Returns:
        The qubit count as a sympy expression. May contain symbols
        for parametric dimensions (e.g., if array sizes are parameters).

    Example:
        >>> from qamomile.circuit.ir.block import Block
        >>> block = some_block()
        >>> count = qubits_counter(block)
        >>> print(count)  # e.g., "n + 3" for parametric n
    """

    if isinstance(block, Block):
        resolver = ExprResolver(block=block)
        input_qubits = _count_input_qubits(block.input_values, resolver)
        ops_qubits = _count_from_operations(
            block.operations,
            resolver,
            backend=backend,
        )
        return sp.simplify(input_qubits + ops_qubits)
    resolver = ExprResolver()
    return _count_from_operations(block, resolver, backend=backend)
