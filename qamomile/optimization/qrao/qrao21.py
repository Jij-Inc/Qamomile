"""QRAC(2,1,p) Converter for Quantum Random Access Optimization.

This module provides the QRAC21Converter class that converts optimization
problems into QRAC-encoded quantum circuits using (2,1,p)-QRAC.
Up to 2 variables are encoded into a single qubit using X and Z Pauli operators.
"""

from __future__ import annotations
import typing

import numpy as np

import qamomile.observable as qm_o
from qamomile.optimization.utils import is_close_zero
from qamomile.optimization.binary_model import BinaryModel, VarType

from .base import QRACConverterBase
from .encoder import QRAC21Encoder, _build_var_occupancy, color_group_to_qrac_encode


def qrac21_encode_ising(
    ising: BinaryModel[typing.Literal[VarType.SPIN]],
    color_group: dict[int, list[int]],
) -> tuple[qm_o.Hamiltonian, dict[int, qm_o.PauliOperator]]:
    """Encode a spin model using (2,1,p)-QRAC.

    Args:
        ising: BinaryModel in SPIN vartype.
        color_group: Mapping from qubit index to list of variable indices.

    Returns:
        Tuple of (relaxed Hamiltonian, encoding map).
    """
    encoded_ope = color_group_to_qrac_encode(color_group)
    var_occupancy = _build_var_occupancy(color_group)

    hamiltonian = qm_o.Hamiltonian()
    hamiltonian.constant = ising.constant

    for idx, coeff in ising.linear.items():
        if is_close_zero(coeff):
            continue
        pauli = encoded_ope[idx]
        k = var_occupancy[idx]
        hamiltonian.add_term((pauli,), np.sqrt(k) * coeff)

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


class QRAC21Converter(QRACConverterBase[QRAC21Encoder]):
    """QRAC(2,1,p) Converter for Quantum Random Access Optimization.

    Converts optimization problems into QRAC-encoded form with reduced qubit count.
    Up to 2 variables are encoded into a single qubit using X and Z Pauli operators.

    Example:
        >>> converter = QRAC21Converter(instance)
        >>> hamiltonian = converter.get_cost_hamiltonian()
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self._encoder = QRAC21Encoder(self.spin_model)
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
        hamiltonian, pauli_encoding = qrac21_encode_ising(
            self.spin_model, self.color_group
        )
        self.pauli_encoding = pauli_encoding
        return hamiltonian

    def get_encoded_pauli_list(self) -> list[qm_o.Hamiltonian]:
        """Get the encoded Pauli operators as a list of Hamiltonians.

        Returns:
            list[Hamiltonian]: List of Hamiltonians for each variable's Pauli operator.
        """
        ising = self.spin_model
        num_qubits = len(self.color_group)
        zero_pauli = qm_o.Hamiltonian(num_qubits=num_qubits)
        pauli_operators = [zero_pauli] * ising.num_bits
        for idx, pauli in self.pauli_encoding.items():
            observable = qm_o.Hamiltonian(num_qubits=num_qubits)
            observable.add_term((pauli,), 1.0)
            pauli_operators[idx] = observable
        return pauli_operators
