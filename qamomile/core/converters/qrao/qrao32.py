from __future__ import annotations
import typing as typ
import numpy as np
from qamomile.core.converters.converter import QuantumConverter
from qamomile.core.ising_qubo import IsingModel
import qamomile.core.operator as qm_o
from .qrao31 import color_group_to_qrac_encode
from .graph_coloring import greedy_graph_coloring, check_linear_term

def create_x_prime(idx: int) -> qm_o.PauliOperator:
    """
    Creates a X' operator for the given index.

    .. math::
            X' = \\frac{1}{2}X_1X_2 + \\frac{1}{2}X_1Z_2 + Z_1I_2

    Parameters:
    idx (int): The index of the first qubit.

    Returns:
    qm_o.PauliOperator: The X' operator for the given index.
    """
    Xi0 = qm_o.X(idx)
    Xi1 = qm_o.X(idx + 1)
    Zi0 = qm_o.Z(idx)
    Zi1 = qm_o.Z(idx + 1)
    return 1 / 2 * (Xi0 * Xi1) + 1 / 2 * (Xi0 * Zi1) + Zi0

def create_y_prime(idx: int) -> qm_o.Hamiltonian:
    """
    Creates a Y' operator for the given index.

    .. math::
            Y' = \\frac{1}{2}I_1X_2 + I_1Z_2 + \\frac{1}{2}Y_1Y_2

    Parameters:
    idx (int): The index of the first qubit.

    Returns:
    qm_o.PauliOperator: The Y' operator for the given index.
    """
    Xi1 = qm_o.X(idx + 1)
    Zi1 = qm_o.Z(idx + 1)
    Yi0 = qm_o.Y(idx)
    Yi1 = qm_o.Y(idx + 1)
    return 1 / 2 * Xi1 + Zi1 + 1 / 2 * (Yi0 * Yi1)


def create_z_prime(idx: int) -> qm_o.Hamiltonian:
    """
    Creates a Z' operator for the given index.

    .. math::
            Z' = Z_1Z_2 - \\frac{1}{2}X_1I_2 - \\frac{1}{2}Z_1X_2

    Parameters:
    idx (int): The index of the first qubit.

    Returns:
    qm_o.PauliOperator: The Z' operator for the given index.
    """
    Xi0 = qm_o.X(idx)
    Xi1 = qm_o.X(idx + 1)
    Zi0 = qm_o.Z(idx)
    Zi1 = qm_o.Z(idx + 1)
    return Zi0 * Zi1 - 1 / 2 * Xi0 - 1 / 2 * (Zi0 * Xi1)

def create_prime_operator(pauli_op: qm_o.PauliOperator) -> qm_o.Hamiltonian:
    if pauli_op.pauli == qm_o.Pauli.X:
        return create_x_prime(2 * pauli_op.index)
    elif pauli_op.pauli == qm_o.Pauli.Y:
        return create_y_prime(2 * pauli_op.index)
    elif pauli_op.pauli == qm_o.Pauli.Z:
        return create_z_prime(2 * pauli_op.index)
    else:
        raise ValueError("Invalid Pauli operator")

def qrac32_encode_ising(
    ising: IsingModel, color_group: dict[int, list[int]]
) -> tuple[qm_o.Hamiltonian, dict[int, qm_o.PauliOperator]]:
    encoded_ope = color_group_to_qrac_encode(color_group)

    offset = ising.constant

    hamiltonian = qm_o.Hamiltonian()
    hamiltonian.constant = offset

    # convert linear parts of the objective function into Hamiltonian.
    for idx, coeff in ising.linear.items():
        if coeff == 0.0:
            continue

        pauli = encoded_ope[idx]
        prime_i = create_prime_operator(pauli)
        hamiltonian += coeff * prime_i

    # create quad terms
    for (i, j), coeff in ising.quad.items():
        if coeff == 0.0:
            continue

        if i == j:
            hamiltonian.constant += coeff
            continue

        pauli_i = encoded_ope[i]
        prime_i = create_prime_operator(pauli_i)

        pauli_j = encoded_ope[j]
        prime_j = create_prime_operator(pauli_j)

        hamiltonian += coeff * prime_i * prime_j

    return hamiltonian, encoded_ope

class QRAC32Converter(QuantumConverter):

    max_color_group_size = 3

    def ising_encode(
        self,
        multipliers: typ.Optional[dict[str, float]] = None,
        detail_parameters: typ.Optional[dict[str, dict[tuple[int, ...], tuple[float, float]]]] = None
    ) -> IsingModel:
        ising = super().ising_encode(multipliers, detail_parameters)

        _, color_group = greedy_graph_coloring(
            ising.quad.keys(),
            self.max_color_group_size
        )
        color_group = check_linear_term(
            color_group, list(ising.linear.keys()), self.max_color_group_size
        )

        self.color_group = color_group
        return ising

    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        """
        Construct the cost Hamiltonian for QRAC32.

        Returns:
            qm_o.Hamiltonian: The cost Hamiltonian.
        """
        ising = self.get_ising()

        hamiltonian, pauli_encoding = qrac32_encode_ising(ising, self.color_group)
        self.pauli_encoding = pauli_encoding
        return hamiltonian

    def get_encoded_pauli_list(self) -> list[qm_o.Hamiltonian]:
        # return the encoded Pauli operators as list
        ising = self.get_ising()
        num_qubits = len(self.color_group)
        zero_pauli = qm_o.Hamiltonian(num_qubits=num_qubits * 2)
        pauli_operators = [zero_pauli] * ising.num_bits()
        for idx, pauli in self.pauli_encoding.items():
            observable = create_prime_operator(pauli)
            pauli_operators[idx] = observable
        return pauli_operators
