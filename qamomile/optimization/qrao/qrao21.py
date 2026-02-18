"""QRAC(2,1,p) Converter for Quantum Random Access Optimization.

This module provides the QRAC21Converter class that converts optimization
problems into QRAC-encoded quantum circuits using (2,1,p)-QRAC.
Up to 2 variables are encoded into a single qubit using X and Z Pauli operators.
"""

from __future__ import annotations

import numpy as np

import qamomile.observable as qm_o

from .base import QRACConverterBase
from .encoder import (
    GraphColoringQRACEncoder,
    PauliType,
)


class QRAC21Encoder(GraphColoringQRACEncoder):
    """(2,1,p)-QRAC Encoder.

    Encodes Ising model variables into Pauli operators using 2-coloring.
    Up to 2 variables per qubit, using Z and X Paulis.

    The relaxed Hamiltonian is:
        H̃ = Σ_{ij} √k_i·√k_j·J_{ij}·P_{f(i)}·P_{f(j)} + Σ_i √k_i·h_i·P_{f(i)}

    where k_i is the number of variables encoded on the qubit containing variable i.
    """

    max_color_group_size: int = 2
    paulis: list[PauliType] = ["Z", "X"]

    @property
    def num_qubits(self) -> int:
        return self.num_logical_qubits

    def _get_operator_and_scale(
        self,
        pauli: qm_o.PauliOperator,
        k: int,
    ) -> tuple[qm_o.Hamiltonian, float]:
        h = qm_o.Hamiltonian()
        h.add_term((pauli,), 1.0)
        return h, np.sqrt(k)


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
        return self._encoder.num_qubits

    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        """Generate the cost Hamiltonian for the QRAC-encoded problem.

        Returns:
            Hamiltonian representing the cost function in QRAC form.
        """
        hamiltonian, pauli_encoding = self._encoder.encode_ising(self.spin_model)
        self.pauli_encoding = pauli_encoding
        return hamiltonian

    def get_encoded_pauli_list(self) -> list[qm_o.Hamiltonian]:
        """Get the encoded Pauli operators as a list of Hamiltonians.

        Returns:
            list[Hamiltonian]: List of Hamiltonians for each variable's Pauli operator.
        """
        ising = self.spin_model
        num_qubits = self._encoder.num_qubits
        zero_pauli = qm_o.Hamiltonian(num_qubits=num_qubits)
        pauli_operators = [zero_pauli] * ising.num_bits
        for idx, pauli in self.pauli_encoding.items():
            observable = qm_o.Hamiltonian(num_qubits=num_qubits)
            observable.add_term((pauli,), 1.0)
            pauli_operators[idx] = observable
        return pauli_operators
