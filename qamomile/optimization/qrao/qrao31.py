"""QRAC(3,1,p) Converter for Quantum Random Access Optimization.

This module provides the QRAC31Converter class that converts optimization
problems into QRAC-encoded quantum circuits and handles result decoding.
"""

from __future__ import annotations

import numpy as np

import qamomile.observable as qm_o

from .base_converter import QRACConverterBase
from .base_encoder import (
    GraphColoringQRACEncoder,
    PauliType,
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

    def _get_operator_and_scale(
        self,
        pauli: qm_o.PauliOperator,
        k: int,
    ) -> tuple[qm_o.Hamiltonian, float]:
        h = qm_o.Hamiltonian()
        h.add_term((pauli,), 1.0)
        return h, np.sqrt(k)


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
