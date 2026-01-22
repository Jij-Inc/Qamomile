"""Hamiltonian and Pauli types for quantum observable representation.

This module defines types for representing Pauli operators and Hamiltonians
in the Qamomile IR. These types support symbolic qubit indices, enabling
Hamiltonians to be constructed within qkernels using loop variables.

Design: Single HamiltonianExprType
----------------------------------
All Hamiltonian expressions (single Pauli, scaled Paulis, products, sums)
use a single HamiltonianExprType. The structure is captured in the operation
graph rather than in the type system:

- qm.pauli.Z(0) → HamiltonianExpr via PauliCreateOp
- 3 * qm.pauli.Z(0) → HamiltonianExpr via HamiltonianScaleOp
- Z(0) * Z(1) → HamiltonianExpr via HamiltonianMulOp
- Z(0) + X(1) → HamiltonianExpr via HamiltonianAddOp
"""

import enum

from .primitives import ObjectTypeMixin, ValueType


class PauliKind(enum.Enum):
    """The four Pauli operators."""

    I = 0  # Identity
    X = 1
    Y = 2
    Z = 3

    def __repr__(self) -> str:
        return f"PauliKind.{self.name}"


class HamiltonianExprType(ObjectTypeMixin, ValueType):
    """Type representing a Hamiltonian expression.

    A HamiltonianExprType represents any Hamiltonian expression, including:
    - Single Pauli operators: Z(0), X(i), Y(i+1)
    - Scaled expressions: 0.5 * Z(0)
    - Products of Paulis: Z(0) * Z(1)
    - Sums of expressions: Z(0) + X(1)

    The structure of the expression is captured by the operation graph,
    not by the type. This simplifies the type system while allowing
    complex symbolic Hamiltonians to be constructed in qkernels.

    Example usage in qkernel:
        H = qm.pauli.Z(0) * qm.pauli.Z(1) + qm.pauli.X(0) + qm.pauli.X(1)
        H = J * qm.pauli.Z(i) * qm.pauli.Z(i + 1)  # with loop variable
    """

    pass
