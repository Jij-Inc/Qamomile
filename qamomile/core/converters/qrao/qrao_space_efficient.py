from __future__ import annotations
import typing as typ
import numpy as np
from qamomile.core.converters.converter import QuantumConverter
from qamomile.core.ising_qubo import IsingModel
import qamomile.core.operator as qm_o
from .graph_coloring import greedy_graph_coloring, check_linear_term


def numbering_space_efficient_encode(
    ising: IsingModel,
) -> dict[int, qm_o.PauliOperator]:
    """
    Encodes the Ising model into a space efficient and provides corresponding Pauli operator.

    Args:
        ising (IsingModel): The Ising model to be encoded.

    Returns:
        dict[int, qm_o.PauliOperator]: A dictionary mapping qubit indices to Pauli operators.
    """
    max_quad_index = max(max(t) for t in ising.quad.keys())
    max_linear_index = max(ising.linear.keys())
    num_vars = max(max_quad_index, max_linear_index) + 1

    encode = {}
    pauli_ope = [qm_o.Pauli.X, qm_o.Pauli.Y]
    for i in range(num_vars):
        qubit_index = i // 2
        color = i % 2
        encode[i] = qm_o.PauliOperator(pauli_ope[color], qubit_index)
    return encode

def qrac_space_efficient_encode_ising(
    ising: IsingModel,
) -> tuple[qm_o.Hamiltonian, dict[int, qm_o.PauliOperator]]:
    encoded_ope = numbering_space_efficient_encode(ising)

    offset = ising.constant

    hamiltonian = qm_o.Hamiltonian()
    hamiltonian.constant = offset

    # convert linear parts of the objective function into Hamiltonian.
    for idx, coeff in ising.linear.items():
        if coeff == 0.0:
            continue

        pauli = encoded_ope[idx]
        hamiltonian.add_term((pauli,), np.sqrt(3) * coeff)

    # create Pauli terms
    for (i, j), coeff in ising.quad.items():
        if coeff == 0.0:
            continue

        if i == j:
            hamiltonian.constant += coeff
            continue

        pauli_i = encoded_ope[i]

        pauli_j = encoded_ope[j]

        if pauli_i.index == pauli_j.index:
            hamiltonian.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, pauli_i.index),), np.sqrt(3) * coeff)
        else:
            hamiltonian.add_term((pauli_i, pauli_j), 3 * coeff)

    return hamiltonian, encoded_ope



class QRACSpaceEfficientConverter(QuantumConverter):

    def ising_encode(
        self,
        multipliers: typ.Optional[dict[str, float]] = None,
        detail_parameters: typ.Optional[dict[str, dict[tuple[int, ...], tuple[float, float]]]] = None
    ) -> IsingModel:
        ising = super().ising_encode(multipliers, detail_parameters)
        return ising

    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        """
        Construct the cost Hamiltonian for Space Efficient QRAC.

        Returns:
            qm_o.Hamiltonian: The cost Hamiltonian.
        """
        ising = self.get_ising()

        hamiltonian, pauli_encoding = qrac_space_efficient_encode_ising(ising)
        self.pauli_encoding = pauli_encoding
        self.num_qubits = hamiltonian.num_qubits
        return hamiltonian

    def get_encoded_pauli_list(self) -> list[qm_o.Hamiltonian]:
        # return the encoded Pauli operators as list
        ising = self.get_ising()
        
        zero_pauli = qm_o.Hamiltonian(num_qubits=self.num_qubits)
        pauli_operators = [zero_pauli] * ising.num_bits()
        for idx, pauli in self.pauli_encoding.items():
            observable = qm_o.Hamiltonian(num_qubits=self.num_qubits)
            observable.add_term((pauli,), 1.0)
            pauli_operators[idx] = observable
        return pauli_operators
