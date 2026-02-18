"""QRAC(3,2,p) Converter for Quantum Random Access Optimization.

This module provides the QRAC32Converter class that converts optimization
problems into QRAC-encoded quantum circuits using (3,2,p)-QRAC.

Color groups with k>=2 variables use 2-local prime operators (X', Y', Z')
on 2 physical qubits. Groups with k=1 variable fall back to a regular
1-local Pauli on a single physical qubit.
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


def build_physical_qubit_map(
    color_group: dict[int, list[int]],
) -> tuple[dict[int, int], int]:
    """Map color (logical) index to physical qubit start index.

    For (3,2,p)-QRAC, color groups with k=1 variable use 1 physical qubit
    (regular Pauli), while groups with k>=2 variables use 2 physical qubits
    (prime operators).

    Args:
        color_group: Mapping from color index to list of variable indices.

    Returns:
        Tuple of (color_to_phys_start mapping, total_physical_qubits).
    """
    color_to_phys: dict[int, int] = {}
    current = 0
    for color in sorted(color_group.keys()):
        color_to_phys[color] = current
        current += 1 if len(color_group[color]) == 1 else 2
    return color_to_phys, current


class QRAC32Encoder(GraphColoringQRACEncoder):
    """(3,2,p)-QRAC Encoder.

    Same graph coloring as (3,1,p) but uses 2-local prime operators for
    color groups with k>=2 variables. Groups with k=1 use a single
    physical qubit with a regular Pauli operator.

    Physical qubit allocation per color group:
        k=1: 1 physical qubit (regular Pauli, scale=1)
        k=2: 2 physical qubits (prime operators, scale=√(2·2)=2)
        k=3: 2 physical qubits (prime operators, scale=√(2·3)=√6)
    """

    max_color_group_size: int = 3
    paulis: list[PauliType] = ["Z", "X", "Y"]

    @property
    def num_qubits(self) -> int:
        """Number of physical qubits (1 per k=1 group, 2 per k>=2 group)."""
        _, total = build_physical_qubit_map(self._color_group)
        return total


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


def create_prime_operator(
    pauli_op: qm_o.PauliOperator, phys_start: int
) -> qm_o.Hamiltonian:
    """Convert a logical PauliOperator to its prime (2-qubit) form.

    Args:
        pauli_op: PauliOperator with logical qubit index.
        phys_start: Physical qubit start index for the prime operator.

    Returns:
        Hamiltonian representing the 2-local prime operator.
    """
    if pauli_op.pauli == qm_o.Pauli.X:
        return create_x_prime(phys_start)
    elif pauli_op.pauli == qm_o.Pauli.Y:
        return create_y_prime(phys_start)
    elif pauli_op.pauli == qm_o.Pauli.Z:
        return create_z_prime(phys_start)
    else:
        raise ValueError("Invalid Pauli operator")


def _make_operator(
    pauli: qm_o.PauliOperator,
    k: int,
    color_to_phys: dict[int, int],
) -> tuple[qm_o.Hamiltonian, float]:
    """Create the operator and scale factor for a single variable.

    For k=1: regular 1-local Pauli on a single physical qubit, scale=1.
    For k>=2: 2-local prime operator on 2 physical qubits, scale=√(2k).

    Args:
        pauli: PauliOperator with color (logical qubit) index.
        k: Number of variables on this color group.
        color_to_phys: Mapping from color index to physical qubit start.

    Returns:
        Tuple of (operator Hamiltonian, scale factor).
    """
    phys_start = color_to_phys[pauli.index]
    if k == 1:
        h = qm_o.Hamiltonian()
        h.add_term((qm_o.PauliOperator(pauli.pauli, phys_start),), 1.0)
        return h, 1.0
    else:
        return create_prime_operator(pauli, phys_start), np.sqrt(2 * k)


def qrac32_encode_ising(
    ising: BinaryModel[typing.Literal[VarType.SPIN]],
    color_group: dict[int, list[int]],
) -> tuple[qm_o.Hamiltonian, dict[int, qm_o.PauliOperator]]:
    """Encode a spin model using (3,2,p)-QRAC.

    Color groups with k=1 variable use a single physical qubit with a
    regular Pauli operator (scale=1). Groups with k>=2 variables use
    2 physical qubits with prime operators (scale=√(2k)).

    Args:
        ising: BinaryModel in SPIN vartype.
        color_group: Mapping from qubit index to list of variable indices.

    Returns:
        Tuple of (relaxed Hamiltonian, encoding map).
    """
    encoded_ope = color_group_to_qrac_encode(color_group)
    var_occupancy = _build_var_occupancy(color_group)
    color_to_phys, _ = build_physical_qubit_map(color_group)

    hamiltonian = qm_o.Hamiltonian()
    hamiltonian.constant = ising.constant

    for idx, coeff in ising.linear.items():
        if is_close_zero(coeff):
            continue
        pauli = encoded_ope[idx]
        k = var_occupancy[idx]
        op, scale = _make_operator(pauli, k, color_to_phys)
        hamiltonian += scale * coeff * op

    for (i, j), coeff in ising.quad.items():
        if is_close_zero(coeff):
            continue
        if i == j:
            hamiltonian.constant += coeff
            continue
        op_i, scale_i = _make_operator(encoded_ope[i], var_occupancy[i], color_to_phys)
        op_j, scale_j = _make_operator(encoded_ope[j], var_occupancy[j], color_to_phys)
        hamiltonian += scale_i * scale_j * coeff * op_i * op_j

    return hamiltonian, encoded_ope


class QRAC32Converter(QRACConverterBase[QRAC32Encoder]):
    """QRAC(3,2,p) Converter for Quantum Random Access Optimization.

    Converts optimization problems into QRAC-encoded form. Color groups
    with k>=2 variables use 2-local prime operators (X', Y', Z') on
    2 physical qubits. Groups with k=1 use a regular Pauli on 1 physical qubit.

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
        """Physical qubit count (1 per k=1 group, 2 per k>=2 group)."""
        _, total = build_physical_qubit_map(self.color_group)
        return total

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
        color_to_phys, total_phys = build_physical_qubit_map(self.color_group)
        var_occupancy = _build_var_occupancy(self.color_group)

        zero_pauli = qm_o.Hamiltonian(num_qubits=total_phys)
        pauli_operators = [zero_pauli] * ising.num_bits
        for idx, pauli in self.pauli_encoding.items():
            k = var_occupancy[idx]
            phys_start = color_to_phys[pauli.index]
            if k == 1:
                observable = qm_o.Hamiltonian(num_qubits=total_phys)
                observable.add_term((qm_o.PauliOperator(pauli.pauli, phys_start),), 1.0)
            else:
                observable = create_prime_operator(pauli, phys_start)
            pauli_operators[idx] = observable
        return pauli_operators
