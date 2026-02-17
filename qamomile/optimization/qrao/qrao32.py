"""QRAC(3,2,p) Converter for Quantum Random Access Optimization.

This module provides the QRAC32Converter class that converts optimization
problems into QRAC-encoded quantum circuits using (3,2,p)-QRAC.

The (3,2,p)-QRAC uses 2-local prime operators (X', Y', Z') where each
logical qubit maps to 2 physical qubits, providing higher fidelity encoding.
"""

from __future__ import annotations
import typing

import numpy as np

import qamomile.observable as qm_o
from qamomile.optimization.utils import is_close_zero
from qamomile.optimization.binary_model import BinaryModel, VarType

from .base import QRACConverterBase
from .encoder import QRAC32Encoder, _build_var_occupancy, color_group_to_qrac_encode


def create_x_prime(idx: int) -> qm_o.Hamiltonian:
    """Create X' operator for the given 2-qubit starting index.

    X' = (1/√6) * (1/2 * X₁X₂ + 1/2 * X₁Z₂ + Z₁I₂)

    Args:
        idx: Index of the first qubit.

    Returns:
        Hamiltonian representing the X' operator.
    """
    Xi0 = qm_o.X(idx)
    Xi1 = qm_o.X(idx + 1)
    Zi0 = qm_o.Z(idx)
    Zi1 = qm_o.Z(idx + 1)
    return 1 / np.sqrt(6) * (1 / 2 * (Xi0 * Xi1) + 1 / 2 * (Xi0 * Zi1) + Zi0)


def create_y_prime(idx: int) -> qm_o.Hamiltonian:
    """Create Y' operator for the given 2-qubit starting index.

    Y' = (1/√6) * (1/2 * I₁X₂ + I₁Z₂ + 1/2 * Y₁Y₂)

    Args:
        idx: Index of the first qubit.

    Returns:
        Hamiltonian representing the Y' operator.
    """
    Xi1 = qm_o.X(idx + 1)
    Zi1 = qm_o.Z(idx + 1)
    Yi0 = qm_o.Y(idx)
    Yi1 = qm_o.Y(idx + 1)
    return 1 / np.sqrt(6) * (1 / 2 * Xi1 + Zi1 + 1 / 2 * (Yi0 * Yi1))


def create_z_prime(idx: int) -> qm_o.Hamiltonian:
    """Create Z' operator for the given 2-qubit starting index.

    Z' = (1/√6) * (Z₁Z₂ - 1/2 * X₁I₂ - 1/2 * Z₁X₂)

    Args:
        idx: Index of the first qubit.

    Returns:
        Hamiltonian representing the Z' operator.
    """
    Xi0 = qm_o.X(idx)
    Xi1 = qm_o.X(idx + 1)
    Zi0 = qm_o.Z(idx)
    Zi1 = qm_o.Z(idx + 1)
    return 1 / np.sqrt(6) * (Zi0 * Zi1 - 1 / 2 * Xi0 - 1 / 2 * (Zi0 * Xi1))


def create_prime_operator(pauli_op: qm_o.PauliOperator) -> qm_o.Hamiltonian:
    """Convert a logical PauliOperator to its prime (2-qubit) form.

    Args:
        pauli_op: PauliOperator with logical qubit index.

    Returns:
        Hamiltonian representing the 2-local prime operator.
    """
    if pauli_op.pauli == qm_o.Pauli.X:
        return create_x_prime(2 * pauli_op.index)
    elif pauli_op.pauli == qm_o.Pauli.Y:
        return create_y_prime(2 * pauli_op.index)
    elif pauli_op.pauli == qm_o.Pauli.Z:
        return create_z_prime(2 * pauli_op.index)
    else:
        raise ValueError("Invalid Pauli operator")


def qrac32_encode_ising(
    ising: BinaryModel[typing.Literal[VarType.SPIN]],
    color_group: dict[int, list[int]],
) -> tuple[qm_o.Hamiltonian, dict[int, qm_o.PauliOperator]]:
    """Encode a spin model using (3,2,p)-QRAC.

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
        prime_i = create_prime_operator(pauli)
        k = var_occupancy[idx]
        hamiltonian += np.sqrt(2 * k) * coeff * prime_i

    for (i, j), coeff in ising.quad.items():
        if is_close_zero(coeff):
            continue
        if i == j:
            hamiltonian.constant += coeff
            continue
        pauli_i = encoded_ope[i]
        prime_i = create_prime_operator(pauli_i)
        pauli_j = encoded_ope[j]
        prime_j = create_prime_operator(pauli_j)
        ki, kj = var_occupancy[i], var_occupancy[j]
        hamiltonian += np.sqrt(2 * ki) * np.sqrt(2 * kj) * coeff * prime_i * prime_j

    return hamiltonian, encoded_ope


class QRAC32Converter(QRACConverterBase[QRAC32Encoder]):
    """QRAC(3,2,p) Converter for Quantum Random Access Optimization.

    Converts optimization problems into QRAC-encoded form using 2-local
    prime operators (X', Y', Z'). Each logical qubit maps to 2 physical qubits.

    Example:
        >>> converter = QRAC32Converter(instance)
        >>> hamiltonian = converter.get_cost_hamiltonian()
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self._encoder = QRAC32Encoder(self.spin_model)
        self.color_group = self._encoder.color_group
        self.pauli_encoding: dict[int, qm_o.PauliOperator] = {}

    @property
    def num_qubits(self) -> int:
        """Physical qubit count (2x logical qubits)."""
        return len(self.color_group) * 2

    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        """Generate the cost Hamiltonian for the QRAC-encoded problem.

        Returns:
            Hamiltonian representing the cost function in QRAC form.
        """
        hamiltonian, pauli_encoding = qrac32_encode_ising(
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
        num_qubits = len(self.color_group) * 2
        zero_pauli = qm_o.Hamiltonian(num_qubits=num_qubits)
        pauli_operators = [zero_pauli] * ising.num_bits
        for idx, pauli in self.pauli_encoding.items():
            observable = create_prime_operator(pauli)
            pauli_operators[idx] = observable
        return pauli_operators
