"""
Qamomile to QuTiP Transpiler Module

This module provides functionality to convert Qamomile hamiltonians to their QuTiP equivalents. It includes a QuTiPTranspiler
class that implements the QuantumSDKTranspiler interface for QuTiP compatibility.

Key features:
- Convert Qamomile Hamiltonians to QuTip Hamiltonians

Usage:
    from qamomile.qutip.transpiler import QuTiPTranspiler

    transpiler = QuTiPTranspiler()
    qp_operator = transpiler.transpile_hamiltonian(qamomile_hamiltonian)


Note: This module requires both Qamomile and QuTiP to be installed.
"""

import qamomile.core.operator as qm_o

import collections
from qamomile.core.transpiler import QuantumSDKTranspiler
from qutip import Qobj, sigmaz, sigmax, sigmay, tensor, qzero, qeye


class QuTiPTranspiler(QuantumSDKTranspiler[tuple[collections.Counter[int], int]]):
    """
    Transpiler class for converting between Qamomile and QuTiP quantum objects.

    This class implements the QuantumSDKTranspiler interface for QuTiP compatibility,
    providing methods to convert Hamiltonians.
    """

    def transpile_hamiltonian(self, operator: qm_o.Hamiltonian) -> Qobj:
        """
        Convert a Qamomile Hamiltonian to a QuTiP Hamiltonian.

        Args:
            operator (qm_o.Hamiltonian): The Qamomile Hamiltonian to convert.

        Returns:
            Qobj: The converted Hamiltonian.

        Raises:
            NotImplementedError: If an unsupported Pauli operator is encountered.
        """
        n = operator.num_qubits
        H = tensor([qzero(2)] * n)
        for term, coeff in operator.terms.items():
            op_list = [qeye(2)] * n
            for pauli in term:
                match pauli.pauli:
                    case qm_o.Pauli.X:
                        op_list[pauli.index] = sigmax()
                    case qm_o.Pauli.Y:
                        op_list[pauli.index] = sigmay()
                    case qm_o.Pauli.Z:
                        op_list[pauli.index] = sigmaz()
                    case _:
                        raise NotImplementedError(
                            "Only Pauli X, Y, and Z are supported"
                        )
            H += coeff * tensor(op_list)
        if operator.constant != 0:
            H += operator.constant * qeye(2**n)
        return Qobj(H)

    def transpile_circuit(self) -> None:
        raise NotImplementedError(
            "Invalid function. The 'transpile_circuit' function is not supported by the QuTiPTranspiler class."
        )

    def convert_result(self) -> None:
        raise NotImplementedError(
            "Invalid function. The 'convert_result' function is not supported by the QuTiPTranspiler class."
        )
