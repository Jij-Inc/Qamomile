"""
Qamomile to Pennylane Transpiler Module

This module provides functionality to convert Qamomile Hamiltonians to their Pennylane equivalents. It includes a `PennylaneTranspiler` class that implements the `QuantumSDKTranspiler` interface for Pennylane compatibility.

Key Features:
- Convert Qamomile Hamiltonians to Pennylane Hamiltonians.

Usage Example:
    ```python
    from qamomile.Pennylane.transpiler import PennylaneTranspiler

    transpiler = PennylaneTranspiler()
    pennylane_operator = transpiler.transpile_hamiltonian(qamomile_hamiltonian)
    ```

Requirements:
- Qamomile
- Pennylane
"""

import qamomile.core.operator 
import pennylane as qml
import collections
import numpy as np
from qamomile.core.transpiler import QuantumSDKTranspiler

class PennylaneTranspiler(QuantumSDKTranspiler[tuple[collections.Counter[int], int]]):
    """
    A transpiler class for converting Qamomile Hamiltonians to Pennylane-compatible Hamiltonians.

    This class implements the `QuantumSDKTranspiler` interface, providing methods to convert Hamiltonians between Qamomile and Pennylane representations.

    Methods:
    --------
    transpile_hamiltonian(operator: qm_o.Hamiltonian) -> qml.Hamiltonian:
        Converts a Qamomile Hamiltonian to a Pennylane Hamiltonian.

    Example Usage:
    --------------
        transpiler = PennylaneTranspiler()
        pennylane_operator = transpiler.transpile_hamiltonian(qamomile_hamiltonian)

    """

    def transpile_hamiltonian(self, operator: qamomile.core.operator.Hamiltonian) -> qml.Hamiltonian :
        """
        Converts a Qamomile Hamiltonian to a Pennylane Hamiltonian.

        Args:
            operator (qm_o.Hamiltonian): The Qamomile Hamiltonian to be converted.

        Returns:
            qml.Hamiltonian: The corresponding Pennylane Hamiltonian.

        Raises:
            NotImplementedError: If an unsupported Pauli operator is encountered during the conversion.
        """
        n = operator.num_qubits
        coeffs = []
        ops = []
        
        for term, coeff in operator.terms.items():
            op_list = [qml.Identity(i) for i in range(n)]
            for pauli in term:
                if pauli.pauli == qamomile.core.operator.Pauli.X:
                    op_list[pauli.index] = qml.PauliX(pauli.index)
                elif pauli.pauli == qamomile.core.operator.Pauli.Y:
                    op_list[pauli.index] = qml.PauliY(pauli.index)
                elif pauli.pauli == qamomile.core.operator.Pauli.Z:
                    op_list[pauli.index] = qml.PauliZ(pauli.index)
                else:
                    raise NotImplementedError(
                        f"Unsupported Pauli operator: {pauli.pauli}"
                    )
            # Combine the operations into a tensor product
            ops.append(qml.operation.Tensor(*op_list))
            coeffs.append(coeff)

        # Construct the Pennylane Hamiltonian
        return qml.Hamiltonian(np.array(coeffs), ops)

    def transpile_circuit(self) -> None:
        raise NotImplementedError(
            "Invalid function. The 'transpile_circuit' function is not supported by the PennylaneTranspiler class."
        )

    def convert_result(self) -> None:
        raise NotImplementedError(
            "Invalid function. The 'convert_result' function is not supported by the PennylaneTranspiler class."
        )
