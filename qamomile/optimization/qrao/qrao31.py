"""QRAC(3,1,p) Converter for Quantum Random Access Optimization.

This module provides the QRAC31Converter class that converts optimization
problems into QRAC-encoded quantum circuits and handles result decoding.
"""

from __future__ import annotations
import typing

import numpy as np

import qamomile.observable as qm_o
from qamomile.optimization.utils import is_close_zero
from qamomile.optimization.binary_model import BinaryModel, VarType

from .base import QRACConverterBase
from .encoder import (
    GraphColoringQRACEncoder,
    PauliType,
    _build_var_occupancy,
    color_group_to_qrac_encode,
)


class QRAC31Encoder(GraphColoringQRACEncoder):
    """(3,1,p)-QRAC Encoder.

    Encodes Ising model variables into Pauli operators on qubits using
    graph coloring to ensure that interacting variables are assigned to
    different qubits.

    The relaxed Hamiltonian is:
        H̃ = Σ_{ij} √k_i·√k_j·J_{ij}·P_{f(i)}·P_{f(j)} + Σ_i √k_i·h_i·P_{f(i)}

    where f(i) = (qubit_index, pauli_type) maps variable i to a Pauli operator,
    and k_i is the number of variables encoded on the qubit containing variable i.

    Attributes:
        max_color_group_size: Maximum variables per qubit (3 for QRAC31)
    """

    max_color_group_size: int = 3
    paulis: list[PauliType] = ["Z", "X", "Y"]

    @property
    def num_qubits(self) -> int:
        return self.num_logical_qubits


def qrac31_encode_ising(
    ising: BinaryModel[typing.Literal[VarType.SPIN]], color_group: dict[int, list[int]]
) -> tuple[qm_o.Hamiltonian, dict[int, qm_o.PauliOperator]]:
    encoded_ope = color_group_to_qrac_encode(color_group)
    var_occupancy = _build_var_occupancy(color_group)

    hamiltonian = qm_o.Hamiltonian()
    hamiltonian.constant = ising.constant

    # Linear terms: √k_i * h_i * P_{f(i)}
    for idx, coeff in ising.linear.items():
        if is_close_zero(coeff):
            continue
        pauli = encoded_ope[idx]
        k = var_occupancy[idx]
        hamiltonian.add_term((pauli,), np.sqrt(k) * coeff)

    # Quadratic terms: √k_i * √k_j * J_{ij} * P_{f(i)} * P_{f(j)}
    for (i, j), coeff in ising.quad.items():
        if is_close_zero(coeff):
            continue

        if i == j:
            hamiltonian.constant += coeff
            continue

        pauli_i = encoded_ope[i]
        pauli_j = encoded_ope[j]
        ki, kj = var_occupancy[i], var_occupancy[j]
        hamiltonian.add_term((pauli_i, pauli_j), np.sqrt(ki) * np.sqrt(kj) * coeff)

    return hamiltonian, encoded_ope


class QRAC31Converter(QRACConverterBase[QRAC31Encoder]):
    """QRAC(3,1,p) Converter for Quantum Random Access Optimization.

    Converts optimization problems into QRAC-encoded form with reduced qubit count.
    Up to 3 variables are encoded into a single qubit using Pauli operators.

    Example:
        >>> converter = QRAC31Converter(instance)
        >>> hamiltonian = converter.get_cost_hamiltonian()
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self._encoder = QRAC31Encoder(self.spin_model)
        self.color_group = self._encoder.color_group
        self.pauli_encoding: dict[int, qm_o.PauliOperator] = {}

    @property
    def num_qubits(self) -> int:
        return len(self.color_group)

    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        """Generate the cost Hamiltonian for the QRAC-encoded problem.

        Returns:
            Hamiltonian representing the cost function in QRAC form.
        """
        hamiltonian, pauli_encoding = qrac31_encode_ising(
            self.spin_model, self.color_group
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
