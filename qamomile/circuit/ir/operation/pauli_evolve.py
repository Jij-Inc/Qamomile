"""Pauli evolution operation for applying exp(-i * gamma * H).

This module defines the PauliEvolveOp IR operation that represents
applying Hamiltonian time evolution to a quantum state.
"""

import dataclasses

from qamomile.circuit.ir.types.hamiltonian import ObservableType
from qamomile.circuit.ir.types.primitives import FloatType
from qamomile.circuit.ir.value import Value

from .operation import Operation, OperationKind, ParamHint, Signature


@dataclasses.dataclass
class PauliEvolveOp(Operation):
    """Pauli evolution operation: exp(-i * gamma * H).

    This operation applies the time evolution of a Pauli Hamiltonian
    to a quantum register.

    Attributes:
        operands[0]: qubits - The quantum register (Vector[Qubit])
        operands[1]: observable - The Observable parameter (Hamiltonian from bindings)
        operands[2]: gamma - The evolution time / variational parameter (Float)
        results[0]: evolved_qubits - The evolved quantum register
    """

    @property
    def qubits(self) -> Value:
        """The quantum register operand."""
        return self.operands[0]

    @property
    def observable(self) -> Value:
        """The Observable parameter operand."""
        return self.operands[1]

    @property
    def gamma(self) -> Value:
        """The evolution time parameter."""
        return self.operands[2]

    @property
    def evolved_qubits(self) -> Value:
        """The evolved quantum register result."""
        return self.results[0]

    @property
    def signature(self) -> Signature:
        return Signature(
            operands=[
                None,  # qubits: any qubit type
                ParamHint(name="observable", type=ObservableType()),
                ParamHint(name="gamma", type=FloatType()),
            ],
            results=[ParamHint(name="evolved_qubits", type=self.evolved_qubits.type)],
        )

    @property
    def operation_kind(self) -> OperationKind:
        """PauliEvolveOp is QUANTUM - transforms quantum state."""
        return OperationKind.QUANTUM
