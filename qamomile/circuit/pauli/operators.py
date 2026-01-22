"""Pauli operator factory functions.

This module provides factory functions for creating Pauli operators
as HamiltonianExpr handles. Each function creates a single Pauli
operator acting on a specified qubit index.

Usage:
    import qamomile.circuit as qm

    @qm.qkernel
    def my_hamiltonian() -> qm.HamiltonianExpr:
        H = qm.pauli.Z(0) * qm.pauli.Z(1) + qm.pauli.X(0)
        return H
"""

from __future__ import annotations

import typing

from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.types.hamiltonian import HamiltonianExprType, PauliKind
from qamomile.circuit.ir.types.primitives import UIntType
from qamomile.circuit.ir.operation.hamiltonian_ops import PauliCreateOp
from qamomile.circuit.ir.value import Value
from qamomile.circuit.frontend.handle.hamiltonian import HamiltonianExpr

if typing.TYPE_CHECKING:
    from qamomile.circuit.frontend.handle.primitives import UInt


def _make_uint_const(val: int) -> Value:
    """Create a UInt Value with a constant."""
    return Value(type=UIntType(), name="uint_const", params={"const": val})


def _create_pauli(
    index: "int | UInt", pauli_kind: PauliKind
) -> HamiltonianExpr:
    """Create a Pauli operator on the given qubit index.

    Args:
        index: The qubit index (can be int literal or symbolic UInt)
        pauli_kind: The type of Pauli operator

    Returns:
        HamiltonianExpr handle representing the Pauli operator
    """
    from qamomile.circuit.frontend.handle.primitives import UInt

    # Convert index to Value
    if isinstance(index, int):
        index_value = _make_uint_const(index)
    elif isinstance(index, UInt):
        index_value = index.value
    else:
        raise TypeError(
            f"Pauli index must be int or UInt, got {type(index).__name__}"
        )

    # Create result value
    result_value = Value(type=HamiltonianExprType(), name=f"pauli_{pauli_kind.name}")

    # Emit PauliCreateOp
    op = PauliCreateOp(
        operands=[index_value],
        results=[result_value],
        pauli_kind=pauli_kind,
    )
    tracer = get_current_tracer()
    tracer.add_operation(op)

    return HamiltonianExpr(value=result_value)


def X(index: "int | UInt") -> HamiltonianExpr:
    """Create a Pauli X operator on the given qubit index.

    Args:
        index: The qubit index (can be int literal or symbolic UInt from loop)

    Returns:
        HamiltonianExpr representing X_index

    Example:
        qm.pauli.X(0)      # X on qubit 0
        qm.pauli.X(i)      # X on qubit i (symbolic)
        qm.pauli.X(i + 1)  # X on qubit i+1 (symbolic expression)
    """
    return _create_pauli(index, PauliKind.X)


def Y(index: "int | UInt") -> HamiltonianExpr:
    """Create a Pauli Y operator on the given qubit index.

    Args:
        index: The qubit index (can be int literal or symbolic UInt from loop)

    Returns:
        HamiltonianExpr representing Y_index

    Example:
        qm.pauli.Y(0)      # Y on qubit 0
        qm.pauli.Y(i)      # Y on qubit i (symbolic)
    """
    return _create_pauli(index, PauliKind.Y)


def Z(index: "int | UInt") -> HamiltonianExpr:
    """Create a Pauli Z operator on the given qubit index.

    Args:
        index: The qubit index (can be int literal or symbolic UInt from loop)

    Returns:
        HamiltonianExpr representing Z_index

    Example:
        qm.pauli.Z(0)      # Z on qubit 0
        qm.pauli.Z(i)      # Z on qubit i (symbolic)
    """
    return _create_pauli(index, PauliKind.Z)


def I(index: "int | UInt") -> HamiltonianExpr:
    """Create a Pauli I (identity) operator on the given qubit index.

    Args:
        index: The qubit index (can be int literal or symbolic UInt from loop)

    Returns:
        HamiltonianExpr representing I_index

    Note:
        Single-qubit identity is typically not needed explicitly,
        but useful for completeness and certain constructions.

    Example:
        qm.pauli.I(0)      # Identity on qubit 0
    """
    return _create_pauli(index, PauliKind.I)
