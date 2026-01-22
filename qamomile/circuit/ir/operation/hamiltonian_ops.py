"""Hamiltonian operations for the Qamomile IR.

This module defines operations for constructing and manipulating
Hamiltonians within qkernels. These operations follow the SSA pattern
where each operation produces new values rather than mutating existing ones.

All operations produce HamiltonianExprType values. The expression structure
is captured by the operation graph, allowing the transpiler to evaluate
the Hamiltonian during compilation.
"""

import dataclasses

from qamomile.circuit.ir.types.primitives import FloatType, UIntType
from qamomile.circuit.ir.types.hamiltonian import (
    PauliKind,
    HamiltonianExprType,
)
from qamomile.circuit.ir.value import Value

from .operation import Operation, OperationKind, ParamHint, Signature


@dataclasses.dataclass
class PauliCreateOp(Operation):
    """Create a Pauli operator on a qubit index.

    This operation creates a single Pauli operator (I, X, Y, or Z) applied
    to a specific qubit index. The qubit index can be symbolic (UInt from
    a loop variable), enabling construction of Hamiltonians in loops.

    Attributes:
        pauli_kind: The type of Pauli operator (I, X, Y, Z)
        operands[0]: qubit_index (UInt) - can be symbolic
        results[0]: HamiltonianExpr value
    """

    pauli_kind: PauliKind = PauliKind.I

    @property
    def qubit_index(self) -> Value:
        """The qubit index this Pauli acts on."""
        return self.operands[0]

    @property
    def output(self) -> Value:
        """The created HamiltonianExpr value."""
        return self.results[0]

    @property
    def signature(self) -> Signature:
        return Signature(
            operands=[ParamHint(name="qubit_index", type=UIntType())],
            results=[ParamHint(name="hamiltonian_expr", type=HamiltonianExprType())],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CLASSICAL


@dataclasses.dataclass
class HamiltonianAddOp(Operation):
    """Add two HamiltonianExprs together.

    This operation creates a new HamiltonianExpr that is the sum of
    two input HamiltonianExprs.

    Attributes:
        operands[0]: lhs HamiltonianExpr
        operands[1]: rhs HamiltonianExpr
        results[0]: sum HamiltonianExpr
    """

    @property
    def lhs(self) -> Value:
        """Left-hand side HamiltonianExpr."""
        return self.operands[0]

    @property
    def rhs(self) -> Value:
        """Right-hand side HamiltonianExpr."""
        return self.operands[1]

    @property
    def output(self) -> Value:
        """The sum HamiltonianExpr."""
        return self.results[0]

    @property
    def signature(self) -> Signature:
        return Signature(
            operands=[
                ParamHint(name="lhs", type=HamiltonianExprType()),
                ParamHint(name="rhs", type=HamiltonianExprType()),
            ],
            results=[ParamHint(name="sum", type=HamiltonianExprType())],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CLASSICAL


@dataclasses.dataclass
class HamiltonianMulOp(Operation):
    """Multiply two HamiltonianExprs together.

    This operation computes the tensor product of two HamiltonianExprs,
    handling Pauli algebra rules (e.g., XX=I, XY=iZ) when terms
    share qubit indices during evaluation.

    Attributes:
        operands[0]: lhs HamiltonianExpr
        operands[1]: rhs HamiltonianExpr
        results[0]: product HamiltonianExpr
    """

    @property
    def lhs(self) -> Value:
        """Left-hand side HamiltonianExpr."""
        return self.operands[0]

    @property
    def rhs(self) -> Value:
        """Right-hand side HamiltonianExpr."""
        return self.operands[1]

    @property
    def output(self) -> Value:
        """The product HamiltonianExpr."""
        return self.results[0]

    @property
    def signature(self) -> Signature:
        return Signature(
            operands=[
                ParamHint(name="lhs", type=HamiltonianExprType()),
                ParamHint(name="rhs", type=HamiltonianExprType()),
            ],
            results=[ParamHint(name="product", type=HamiltonianExprType())],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CLASSICAL


@dataclasses.dataclass
class HamiltonianScaleOp(Operation):
    """Scale a HamiltonianExpr by a scalar.

    This operation creates a new HamiltonianExpr with all term coefficients
    multiplied by the given scalar.

    Attributes:
        operands[0]: hamiltonian_expr (HamiltonianExpr)
        operands[1]: scalar (Float)
        results[0]: scaled HamiltonianExpr
    """

    @property
    def hamiltonian_expr(self) -> Value:
        """The input HamiltonianExpr."""
        return self.operands[0]

    @property
    def scalar(self) -> Value:
        """The scalar multiplier."""
        return self.operands[1]

    @property
    def output(self) -> Value:
        """The scaled HamiltonianExpr."""
        return self.results[0]

    @property
    def signature(self) -> Signature:
        return Signature(
            operands=[
                ParamHint(name="hamiltonian_expr", type=HamiltonianExprType()),
                ParamHint(name="scalar", type=FloatType()),
            ],
            results=[ParamHint(name="scaled", type=HamiltonianExprType())],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CLASSICAL


@dataclasses.dataclass
class HamiltonianNegOp(Operation):
    """Negate a HamiltonianExpr.

    This operation creates a new HamiltonianExpr with all term coefficients
    negated.

    Attributes:
        operands[0]: hamiltonian_expr (HamiltonianExpr)
        results[0]: negated HamiltonianExpr
    """

    @property
    def hamiltonian_expr(self) -> Value:
        """The input HamiltonianExpr."""
        return self.operands[0]

    @property
    def output(self) -> Value:
        """The negated HamiltonianExpr."""
        return self.results[0]

    @property
    def signature(self) -> Signature:
        return Signature(
            operands=[ParamHint(name="hamiltonian_expr", type=HamiltonianExprType())],
            results=[ParamHint(name="negated", type=HamiltonianExprType())],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CLASSICAL


@dataclasses.dataclass
class HamiltonianIdentityOp(Operation):
    """Create an identity Hamiltonian (scalar times identity operator).

    This operation creates a HamiltonianExpr representing a constant
    times the identity operator. This is useful as a starting point
    for building Hamiltonians: `H = 0.0 * I()` creates the zero operator.

    Attributes:
        operands[0]: scalar (Float) - the coefficient
        results[0]: HamiltonianExpr representing scalar * I
    """

    @property
    def scalar(self) -> Value:
        """The scalar coefficient."""
        return self.operands[0]

    @property
    def output(self) -> Value:
        """The created HamiltonianExpr."""
        return self.results[0]

    @property
    def signature(self) -> Signature:
        return Signature(
            operands=[ParamHint(name="scalar", type=FloatType())],
            results=[ParamHint(name="identity", type=HamiltonianExprType())],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CLASSICAL
