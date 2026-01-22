"""Expectation value operation for computing <psi|H|psi>.

This module defines the ExpvalOp IR operation that represents
computing the expectation value of a Hamiltonian observable with
respect to a quantum state.
"""

import dataclasses

from qamomile.circuit.ir.types.primitives import FloatType
from qamomile.circuit.ir.types.hamiltonian import HamiltonianExprType
from qamomile.circuit.ir.value import Value

from .operation import Operation, OperationKind, ParamHint, Signature


@dataclasses.dataclass
class ExpvalOp(Operation):
    """Expectation value operation.

    This operation computes the expectation value <psi|H|psi> where
    psi is the quantum state and H is the Hamiltonian observable.

    The operation bridges quantum and classical computation:
    - Input: quantum state (qubits) + Hamiltonian expression
    - Output: classical Float (expectation value)

    Attributes:
        operands[0]: qubits - The quantum register (Vector[Qubit] or individual qubits)
        operands[1]: hamiltonian - The HamiltonianExpr to measure
        results[0]: exp_val - The estimated expectation value (Float)

    Example IR:
        %qubits = ... # quantum state after ansatz
        %H = HamiltonianAddOp(...) # Z0*Z1 + 0.5*X0
        %exp_val = ExpvalOp(%qubits, %H)  # <psi|H|psi>
    """

    @property
    def qubits(self) -> Value:
        """The quantum register operand."""
        return self.operands[0]

    @property
    def hamiltonian(self) -> Value:
        """The Hamiltonian observable operand."""
        return self.operands[1]

    @property
    def output(self) -> Value:
        """The expectation value result."""
        return self.results[0]

    @property
    def signature(self) -> Signature:
        return Signature(
            operands=[
                ParamHint(name="qubits", type=None),  # Any qubit type (Vector or tuple)
                ParamHint(name="hamiltonian", type=HamiltonianExprType()),
            ],
            results=[ParamHint(name="exp_val", type=FloatType())],
        )

    @property
    def operation_kind(self) -> OperationKind:
        """ExpvalOp is HYBRID - bridges quantum state to classical value."""
        return OperationKind.HYBRID
