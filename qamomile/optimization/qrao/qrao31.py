"""QRAC(3,1,p) Converter for Quantum Random Access Optimization.

This module provides the QRAC31Converter class that converts optimization
problems into QRAC-encoded quantum circuits and handles result decoding.
"""

from __future__ import annotations
import typing

import numpy as np
import ommx.v1

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.optimization.utils import is_close_zero
from qamomile.optimization.binary_model import BinaryModel, VarType
from qamomile.optimization.converter import MathematicalProblemConverter

from .graph_coloring import greedy_graph_coloring, check_linear_term


def color_group_to_qrac_encode(
    color_group: dict[int, list[int]],
) -> dict[int, qm_o.PauliOperator]:
    """qrac encode

    Args:
        color_group (dict[int, list[int]]): key is color (qubit's index). value is list of bit's index.

    Returns:
        dict[int, tuple[int, Pauli]]: key is bit's index. value is tuple of qubit's index and Pauli operator kind.

    Examples:
        >>> color_group = {0: [0, 1, 2], 1: [3, 4], 2: [6,]}
        >>> color_group_to_qrac_encode(color_group)
        {0: Z0, 1: X0, 2: Y0, 3: Z1, 4: X1, 6: Z2}

    """
    qrac31 = {}
    paulis = [qm_o.Pauli.Z, qm_o.Pauli.X, qm_o.Pauli.Y]
    for color, group in color_group.items():
        for ope_idx, bit_index in enumerate(group):
            qrac31[bit_index] = qm_o.PauliOperator(paulis[ope_idx], color)
    return qrac31



def qrac31_encode_ising(
    ising: BinaryModel[typing.Literal[VarType.SPIN]],
    color_group: dict[int, list[int]]
) -> tuple[qm_o.Hamiltonian, dict[int, qm_o.PauliOperator]]:
    encoded_ope = color_group_to_qrac_encode(color_group)

    offset = ising.constant

    hamiltonian = qm_o.Hamiltonian()
    hamiltonian.constant = offset

    # convert linear parts of the objective function into Hamiltonian.
    for idx, coeff in ising.linear.items():
        if is_close_zero(coeff):
            continue
        
        # TODO: Check coefficients should be depend a size of color group
        pauli = encoded_ope[idx]
        hamiltonian.add_term((pauli,), np.sqrt(3) * coeff)

    # create Pauli terms
    for (i, j), coeff in ising.quad.items():
        if is_close_zero(coeff):
            continue

        if i == j:
            hamiltonian.constant += coeff
            continue

        pauli_i = encoded_ope[i]
        pauli_j = encoded_ope[j]

        hamiltonian.add_term((pauli_i, pauli_j), 3 * coeff)

    for inds, coeff in ising.higher.items():
        if is_close_zero(coeff):
            continue
        paulis = tuple(encoded_ope[i] for i in inds)
        hamiltonian.add_term(paulis, (np.sqrt(3) ** len(inds)) * coeff)

    return hamiltonian, encoded_ope


class QRAC31Converter(MathematicalProblemConverter):
    """QRAC(3,1,p) Converter for Quantum Random Access Optimization.

    Converts optimization problems into QRAC-encoded form with reduced qubit count.
    Up to 3 variables are encoded into a single qubit using Pauli operators.

    Example:
        >>> converter = QRAC31Converter(instance)
        >>> exp_executable = converter.get_cost_hamiltonian()
        >>> transpiler = Transpiler(backend)
        >>> executable = transpler.executable()
    """

    @property
    def _hubo_support(self) -> bool:
        return False

    def __post_init__(self,) -> None:
        # Generate color grouping for QRAC encoding
        _, color_group = greedy_graph_coloring(
            graph=self.spin_model.quad.keys(),
            max_color_group_size=3,
        )
        self.color_group = check_linear_term(
            color_group, self.spin_model.linear.keys(), max_color_group_size=3
        )

        self.pauli_encoding: dict[int, qm_o.PauliOperator] = {}

    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        """Generate the cost Hamiltonian for the QRAC-encoded problem.

        Returns:
            Hamiltonian representing the cost function in QRAC form.
        """
        hamiltonian, pauli_encoding = qrac31_encode_ising(
            self.spin_model,
            self.color_group
        )
        self.pauli_encoding = pauli_encoding
        return hamiltonian

    def get_encoded_pauli_list(self) -> list[qm_o.Hamiltonian]:
        """Get the encoded Pauli operators as a list of Hamiltonians.

        This method returns the Pauli Operators which correspond to the each variable in the Ising model.

        Returns:
            list[Hamiltonian]: List of Hamiltonians for each variable's Pauli operator.
        """
        # return the encoded Pauli operators as list
        ising = self.spin_model
        num_qubits = len(self.color_group)
        zero_pauli = qm_o.Hamiltonian(num_qubits=num_qubits)
        pauli_operators = [zero_pauli] * ising.num_bits
        for idx, pauli in self.pauli_encoding.items():
            observable = qm_o.Hamiltonian(num_qubits=num_qubits)
            observable.add_term((pauli,), 1.0)
            pauli_operators[idx] = observable
        return pauli_operators