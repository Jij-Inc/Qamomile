"""Trace scalar IR values through their defining classical operations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import sympy as sp

from qamomile.circuit.estimator._classical_expression import (
    _apply_binop,
    _apply_compop,
    _apply_condop,
    _apply_unary_math,
    _as_boolean,
)
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    CompOp,
    CondOp,
    NotOp,
    UnaryMathOp,
)
from qamomile.circuit.ir.operation.operation import Operation
from qamomile.circuit.ir.value import Value


def _trace_classical_value(
    value: Value,
    block: Any,
    visited: set[int],
    concrete: bool,
    *,
    resolve: Callable[[Any, bool], sp.Expr],
    producer_map: Callable[[Any], Mapping[str, Operation]],
) -> sp.Expr | None:
    """Trace a scalar value through a supported defining operation.

    Args:
        value (Value): Value whose defining operation is sought.
        block (Any): Block whose operations are indexed.
        visited (set[int]): Object identities already visited while tracing.
        concrete (bool): Whether unresolved operands must be concrete.
        resolve (Callable[[Any, bool], sp.Expr]): Recursive scalar resolver.
        producer_map (Callable[[Any], Mapping[str, Operation]]): Block-index
            lookup for operation results.

    Returns:
        sp.Expr | None: Resolved expression for a supported defining
        operation, or ``None`` when no supported producer exists.
    """
    value_id = id(value)
    if value_id in visited:
        return None
    visited.add(value_id)

    operation = producer_map(block).get(value.uuid)
    if isinstance(operation, BinOp):
        left = resolve(operation.operands[0], concrete)
        right = resolve(operation.operands[1], concrete)
        assert operation.kind is not None
        return _apply_binop(operation.kind, left, right)

    if isinstance(operation, CompOp):
        left = resolve(operation.operands[0], concrete)
        right = resolve(operation.operands[1], concrete)
        assert operation.kind is not None
        return _apply_compop(operation.kind, left, right)

    if isinstance(operation, CondOp):
        left = resolve(operation.operands[0], concrete)
        right = resolve(operation.operands[1], concrete)
        assert operation.kind is not None
        return _apply_condop(  # type: ignore[return-value]
            operation.kind,
            left,
            right,
        )

    if isinstance(operation, NotOp):
        operand = resolve(operation.input, concrete)
        return sp.Not(_as_boolean(operand))  # type: ignore[return-value]

    if isinstance(operation, UnaryMathOp):
        operand = resolve(operation.input, concrete)
        assert operation.kind is not None
        return _apply_unary_math(operation.kind, operand)

    return None
