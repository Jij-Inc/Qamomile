"""Shared operation-processing helpers for resource estimation.

Provides resolution and scoping utilities used by the estimators
(gate_counter, qubits_counter).  Each estimator has
its own dispatch loop but calls these shared helpers to avoid
duplicating callable resource-priority logic, QFT/IQFT n resolution,
ControlledU resolution, loop scoping, etc.

This module does NOT own traversal policy (unroll vs. multiply vs.
interpolate) — that stays in each estimator / loop_executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import sympy as sp

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import (
    CompositeGateType,
    InvokeOperation,
    ResourceMetadata,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
)
from qamomile.circuit.ir.operation.gate import ControlledUOperation
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.value import ArrayValue

from ._resolver import ExprResolver

# ------------------------------------------------------------------ #
#  Invoke resource resolution                                         #
# ------------------------------------------------------------------ #


@dataclass
class InvokeResourceResolution:
    """Result of resolving an InvokeOperation's resource source.

    Exactly one of the four branches holds useful data:

    - ``"metadata"``       → *metadata*, *is_opaque*, *oracle_name* etc.
    - ``"implementation"`` → *impl_block*, *impl_resolver*
    - ``"qft_iqft"``      → *n_qubits*
    - ``"error"``          → *error_message*
    """

    kind: Literal["metadata", "implementation", "qft_iqft", "error"]

    # metadata branch
    metadata: ResourceMetadata | None = None
    is_opaque: bool = False
    oracle_name: str | None = None
    query_complexity: int | None = None
    num_control_qubits: int = 0

    # implementation branch
    impl_block: Block | None = None
    impl_resolver: ExprResolver | None = None

    # qft_iqft branch
    n_qubits: sp.Expr | None = None

    # error branch
    error_message: str | None = None


def resolve_invoke_resource(
    op: InvokeOperation,
    resolver: ExprResolver,
    *,
    backend: str | None = None,
) -> InvokeResourceResolution:
    """Resolve an InvokeOperation to its resource source.

    Priority (deterministic, not fallback):

    1. Selected implementation resource metadata, when available.
    2. Callable fallback resource metadata, when available.
    3. Selected implementation body or direct callable body.
    4. Bodyless oracle call/query accounting.
    5. Known formula (QFT / IQFT).
    6. Error — no resource info available.

    Args:
        op (InvokeOperation): The invocation to resolve.
        resolver (ExprResolver): Resolver for the current scope.
        backend (str | None): Optional backend name used to select
            backend-specific callable implementations. Defaults to ``None``.

    Returns:
        InvokeResourceResolution: Exactly one of its branches is populated.
    """
    resource = op.effective_resource(backend=backend)
    body = op.effective_body(backend=backend)

    # 1. metadata
    if isinstance(resource, ResourceMetadata):
        oracle_name = None
        qc = None
        if body is None:
            oracle_name = op.custom_name or op.gate_type.value
            qc = resource.query_complexity
        return InvokeResourceResolution(
            kind="metadata",
            metadata=resource,
            is_opaque=body is None,
            oracle_name=oracle_name,
            query_complexity=qc,
            num_control_qubits=op.num_control_qubits,
        )

    # Bodyless oracles remain meaningful without explicit metadata: they
    # contribute one opaque oracle call and zero concrete gate counts.
    if body is None and op.attrs.get("kind") == "oracle":
        oracle_name = op.custom_name or op.gate_type.value
        return InvokeResourceResolution(
            kind="metadata",
            metadata=ResourceMetadata(),
            is_opaque=True,
            oracle_name=oracle_name,
            query_complexity=None,
            num_control_qubits=op.num_control_qubits,
        )

    # 2. implementation
    if body is not None:
        if isinstance(body, Block):
            return InvokeResourceResolution(
                kind="implementation",
                impl_block=body,
                impl_resolver=resolver.call_child_scope(op, called_block=body),
            )

    # 3. QFT / IQFT
    if op.gate_type in (CompositeGateType.QFT, CompositeGateType.IQFT):
        n = _resolve_qft_iqft_n(op, resolver)
        return InvokeResourceResolution(kind="qft_iqft", n_qubits=n)

    # 4. Error
    gate_name = op.custom_name or op.gate_type.value
    return InvokeResourceResolution(
        kind="error",
        error_message=(
            f"Cannot estimate resources for InvokeOperation '{gate_name}': "
            f"No resource metadata, implementation body, or known formula "
            f"is available."
        ),
    )


def _resolve_qft_iqft_n(
    op: InvokeOperation,
    resolver: ExprResolver,
) -> sp.Expr:
    """Resolve qubit count *n* for QFT / IQFT.

    Strategy:
      1. target_qubits → parent_array shape[0] (symbolic) or concrete count
      2. Search block for QInitOperation with ``"counting"`` array (QPE)
      3. ``num_target_qubits`` field
      4. ``sp.Symbol("n")`` fallback

    Args:
        op (InvokeOperation): A QFT or IQFT invocation.
        resolver (ExprResolver): Resolver for the current scope.

    Returns:
        sp.Expr: The qubit count (concrete ``sp.Integer`` or symbolic).
    """
    target_qubits = op.target_qubits
    n_qubits = len(target_qubits)

    if n_qubits > 0:
        first = target_qubits[0]
        if isinstance(first, ArrayValue) and first.shape:
            return resolver.resolve(first.shape[0])
        if hasattr(first, "parent_array") and first.parent_array is not None:
            parent = first.parent_array
            if isinstance(parent, ArrayValue) and parent.shape:
                return resolver.resolve(parent.shape[0])
        return sp.Integer(n_qubits)

    # No target qubits — search block for QPE "counting" array
    block = resolver.block
    if block is not None:
        for block_op in block.operations:
            if isinstance(block_op, QInitOperation):
                if block_op.results and isinstance(block_op.results[0], ArrayValue):
                    array = block_op.results[0]
                    if array.shape and array.name == "counting":
                        return resolver.resolve(array.shape[0])

    # num_target_qubits field
    if isinstance(op.num_target_qubits, int) and op.num_target_qubits > 0:
        return sp.Integer(op.num_target_qubits)

    # Last resort
    return sp.Symbol("n", integer=True, positive=True)


# ------------------------------------------------------------------ #
#  ControlledU resolution                                             #
# ------------------------------------------------------------------ #


def resolve_controlled_u(
    op: ControlledUOperation,
    resolver: ExprResolver,
) -> tuple[int | sp.Expr, int]:
    """Resolve a ControlledUOperation to ``(num_controls, num_targets)``.

    ``num_controls`` may be ``int`` or ``sp.Expr`` (symbolic).
    ``num_targets`` is always a concrete ``int``.

    Args:
        op (ControlledUOperation): The operation to resolve.
        resolver (ExprResolver): Resolver for the current scope.

    Returns:
        tuple[int | sp.Expr, int]: ``(num_controls, num_targets)``.
    """
    if op.is_symbolic_num_controls:
        nc: int | sp.Expr = resolver.resolve(op.num_controls)
    else:
        nc = op.num_controls  # type: ignore[assignment]

    controlled_block = op.block
    if isinstance(controlled_block, Block):
        num_targets = sum(
            1 for inp in controlled_block.input_values if inp.type.is_quantum()
        )
    else:
        target_ops = getattr(op, "target_operands", [])
        num_targets = len(target_ops) if target_ops else 1

    return nc, num_targets


# ------------------------------------------------------------------ #
#  ForItems cardinality                                               #
# ------------------------------------------------------------------ #


def resolve_for_items_cardinality(
    op: ForItemsOperation,
) -> sp.Expr:
    """Return the symbolic cardinality ``|dict_name|`` for a ForItems loop.

    Args:
        op (ForItemsOperation): The for-items loop operation.

    Returns:
        sp.Expr: A positive integer symbol ``|dict_name|``.
    """
    dict_operand = op.operands[0]
    if hasattr(dict_operand, "is_parameter") and dict_operand.is_parameter():
        dict_name = dict_operand.parameter_name() or dict_operand.name
    else:
        dict_name = dict_operand.name
    return sp.Symbol(f"|{dict_name}|", integer=True, positive=True)


# ------------------------------------------------------------------ #
#  ForOperation scoping                                               #
# ------------------------------------------------------------------ #


class _LocalBlock:
    """Lightweight stand-in for Block, used for value tracing in loop bodies.

    Attributes:
        operations (list[Any]): The list of operations in this block.
    """

    __slots__ = ("operations",)

    def __init__(self, operations: list[Any]):
        self.operations = operations


def build_for_loop_scope(
    op: ForOperation,
    resolver: ExprResolver,
) -> tuple[ExprResolver, sp.Expr, sp.Expr, sp.Expr, sp.Symbol]:
    """Build child resolver and resolved bounds for a ForOperation.

    Args:
        op (ForOperation): The for-loop operation.
        resolver (ExprResolver): Resolver for the enclosing scope.

    Returns:
        tuple[ExprResolver, sp.Expr, sp.Expr, sp.Expr, sp.Symbol]:
            ``(child_resolver, start, stop, step, loop_symbol)``.
            The child resolver has ``block`` = local block over
            ``op.operations``, loop variable mapped to ``loop_symbol``,
            and parent blocks propagated.
    """
    loop_sym = sp.Symbol(op.loop_var, integer=True, positive=True)

    # Collect loop-variable name mappings BEFORE creating the child resolver,
    # because the loop var symbol may be needed when resolving bounds that
    # reference the loop variable indirectly.
    loop_var_names = _collect_loop_var_names(
        op.operations,
        op.loop_var,
        loop_sym,
    )

    # Create local block + child resolver (propagates parent blocks)
    local_block = _LocalBlock(op.operations)
    child = resolver.child_scope(
        inner_block=local_block,
        extra_loop_vars=loop_var_names,
    )

    # Resolve bounds in the CHILD resolver so that values defined in the
    # loop body (e.g. BinOps producing bounds for nested loops) can be
    # traced through both the local block and parent blocks.
    start = child.resolve(op.operands[0])
    stop = child.resolve(op.operands[1])
    step = child.resolve(op.operands[2]) if len(op.operands) >= 3 else sp.Integer(1)

    return child, start, stop, step, loop_sym


def _collect_loop_var_names(
    operations: list[Any],
    loop_var_name: str,
    loop_sym: sp.Symbol,
) -> dict[str, sp.Symbol]:
    """Collect ``{value_name: loop_symbol}`` for Values matching the loop variable.

    Scans one level of nesting so that operands referencing the loop
    variable inside common patterns (e.g. nested BinOps) are also
    captured.

    Args:
        operations (list[Any]): Operations inside the loop body.
        loop_var_name (str): Name of the loop variable (e.g. ``"i"``).
        loop_sym (sp.Symbol): SymPy symbol representing the loop variable.

    Returns:
        dict[str, sp.Symbol]: Mapping from value name to *loop_sym* for
            every Value whose ``name`` equals *loop_var_name*.
    """
    result: dict[str, sp.Symbol] = {loop_var_name: loop_sym}
    seen: set[str] = set()

    def check_operands(op: Any) -> None:
        for operand in getattr(op, "operands", []):
            if (
                hasattr(operand, "name")
                and hasattr(operand, "uuid")
                and operand.name == loop_var_name
                and operand.uuid not in seen
            ):
                seen.add(operand.uuid)
                result[operand.name] = loop_sym

    for op in operations:
        check_operands(op)
        # Check one level of nesting for common patterns
        for inner_op in getattr(op, "operations", []):
            check_operands(inner_op)

    return result


# ------------------------------------------------------------------ #
#  WhileOperation scoping                                             #
# ------------------------------------------------------------------ #


def build_while_scope(
    op: Any,
    resolver: ExprResolver,
) -> tuple[ExprResolver, sp.Symbol]:
    """Build child resolver and trip-count symbol for a WhileOperation.

    Args:
        op (Any): A WhileOperation with an ``operations`` attribute.
        resolver (ExprResolver): Resolver for the enclosing scope.

    Returns:
        tuple[ExprResolver, sp.Symbol]: ``(child_resolver, trip_count_symbol)``
            where *trip_count_symbol* is ``|while|``.
    """
    local_block = _LocalBlock(op.operations)
    child = resolver.child_scope(inner_block=local_block)
    trip_count = sp.Symbol("|while|", integer=True, positive=True)
    return child, trip_count


# ------------------------------------------------------------------ #
#  IfOperation scoping                                                #
# ------------------------------------------------------------------ #


def build_if_scopes(
    op: Any,
    resolver: ExprResolver,
) -> tuple[ExprResolver, ExprResolver]:
    """Build child resolvers for both branches of an IfOperation.

    Args:
        op (Any): An IfOperation with ``true_operations`` and
            ``false_operations`` attributes.
        resolver (ExprResolver): Resolver for the enclosing scope.

    Returns:
        tuple[ExprResolver, ExprResolver]: ``(true_resolver, false_resolver)``.
    """
    true_block = _LocalBlock(op.true_operations)
    false_block = _LocalBlock(op.false_operations)
    true_child = resolver.child_scope(inner_block=true_block)
    false_child = resolver.child_scope(inner_block=false_block)
    return true_child, false_child


# ------------------------------------------------------------------ #
#  ForItems scoping                                                   #
# ------------------------------------------------------------------ #


def build_for_items_scope(
    op: ForItemsOperation,
    resolver: ExprResolver,
) -> ExprResolver:
    """Build child resolver for a ForItemsOperation body.

    Args:
        op (ForItemsOperation): The for-items loop operation.
        resolver (ExprResolver): Resolver for the enclosing scope.

    Returns:
        ExprResolver: Child resolver scoped to the loop body.
    """
    local_block = _LocalBlock(op.operations)
    return resolver.child_scope(inner_block=local_block)
