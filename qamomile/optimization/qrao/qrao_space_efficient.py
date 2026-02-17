"""Space Efficient QRAC Converter for Quantum Random Access Optimization.

This module implements Space Efficient QRAO which maintains a constant
2:1 compression ratio by using sequential numbering instead of graph coloring.

Variables are alternately assigned X and Y Pauli operators on consecutive qubits.
When interacting variables share the same qubit, a Z operator is used instead
of the product of their Paulis.
"""

from __future__ import annotations
import typing

import numpy as np

import qamomile.observable as qm_o
from qamomile.optimization.utils import is_close_zero
from qamomile.optimization.binary_model import BinaryModel, VarType

from .base import QRACConverterBase
from .encoder import QRACSpaceEfficientEncoder


def numbering_space_efficient_encode(
    ising: BinaryModel[typing.Literal[VarType.SPIN]],
) -> dict[int, qm_o.PauliOperator]:
    """Encode using sequential numbering.

    Variable i is assigned to qubit i//2 with Pauli X (even) or Y (odd).

    Args:
        ising: BinaryModel in SPIN vartype.

    Returns:
        Mapping from variable index to PauliOperator.
    """
    num_vars = ising.num_bits
    encode = {}
    pauli_ope = [qm_o.Pauli.X, qm_o.Pauli.Y]
    for i in range(num_vars):
        qubit_index = i // 2
        color = i % 2
        encode[i] = qm_o.PauliOperator(pauli_ope[color], qubit_index)
    return encode


def qrac_space_efficient_encode_ising(
    ising: BinaryModel[typing.Literal[VarType.SPIN]],
) -> tuple[qm_o.Hamiltonian, dict[int, qm_o.PauliOperator]]:
    """Encode a spin model using space-efficient QRAC.

    For quadratic terms:
    - Same qubit: √3 * J_{ij} * Z_k
    - Different qubit: 3 * J_{ij} * P_{f(i)} * P_{f(j)}

    Args:
        ising: BinaryModel in SPIN vartype.

    Returns:
        Tuple of (relaxed Hamiltonian, encoding map).
    """
    encoded_ope = numbering_space_efficient_encode(ising)

    hamiltonian = qm_o.Hamiltonian()
    hamiltonian.constant = ising.constant

    for idx, coeff in ising.linear.items():
        if is_close_zero(coeff):
            continue
        pauli = encoded_ope[idx]
        hamiltonian.add_term((pauli,), np.sqrt(3) * coeff)

    for (i, j), coeff in ising.quad.items():
        if is_close_zero(coeff):
            continue
        if i == j:
            hamiltonian.constant += coeff
            continue
        pauli_i = encoded_ope[i]
        pauli_j = encoded_ope[j]
        if pauli_i.index == pauli_j.index:
            hamiltonian.add_term(
                (qm_o.PauliOperator(qm_o.Pauli.Z, pauli_i.index),),
                np.sqrt(3) * coeff,
            )
        else:
            hamiltonian.add_term((pauli_i, pauli_j), 3 * coeff)

    return hamiltonian, encoded_ope


class QRACSpaceEfficientConverter(QRACConverterBase):
    """Space Efficient QRAC Converter for Quantum Random Access Optimization.

    Uses sequential numbering with constant 2:1 compression ratio.
    No graph coloring is performed.

    Example:
        >>> converter = QRACSpaceEfficientConverter(instance)
        >>> hamiltonian = converter.get_cost_hamiltonian()
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self._encoder = QRACSpaceEfficientEncoder(self.spin_model)
        self.pauli_encoding: dict[int, qm_o.PauliOperator] = {}
        self._num_qubits: int = 0

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def encoder(self) -> QRACSpaceEfficientEncoder:
        return self._encoder

    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        """Generate the cost Hamiltonian for the space-efficient QRAC-encoded problem.

        Returns:
            Hamiltonian representing the cost function in QRAC form.
        """
        hamiltonian, pauli_encoding = qrac_space_efficient_encode_ising(self.spin_model)
        self.pauli_encoding = pauli_encoding
        self._num_qubits = hamiltonian.num_qubits
        return hamiltonian

    def get_encoded_pauli_list(self) -> list[qm_o.Hamiltonian]:
        """Get the encoded Pauli operators as a list of Hamiltonians.

        Returns:
            list[Hamiltonian]: List of Hamiltonians for each variable's Pauli operator.
        """
        ising = self.spin_model
        zero_pauli = qm_o.Hamiltonian(num_qubits=self._num_qubits)
        pauli_operators = [zero_pauli] * ising.num_bits
        for idx, pauli in self.pauli_encoding.items():
            observable = qm_o.Hamiltonian(num_qubits=self._num_qubits)
            observable.add_term((pauli,), 1.0)
            pauli_operators[idx] = observable
        return pauli_operators
