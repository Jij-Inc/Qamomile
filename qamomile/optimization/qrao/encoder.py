"""QRAC Encoders for Quantum Random Access Optimization.

This module implements the encoding logic for various QRAC variants:
- (2,1,p)-QRAC: Maps up to 2 variables per qubit using X, Z.
- (3,1,p)-QRAC: Maps up to 3 variables per qubit using X, Y, Z.
- (3,2,p)-QRAC: Maps up to 3 variables per 2 qubits using prime operators.
- Space-efficient QRAC: Sequential numbering with X, Y, constant 2:1 compression.
"""

from __future__ import annotations
from math import sqrt
from typing import Literal

import numpy as np

from qamomile.optimization.binary_model import BinaryModel
from qamomile.optimization.utils import is_close_zero
from .graph_coloring import greedy_graph_coloring, check_linear_term


PauliType = Literal["X", "Y", "Z"]


def _build_var_occupancy(color_group: dict[int, list[int]]) -> dict[int, int]:
    """Map each variable index to the occupancy of its qubit.

    Args:
        color_group: Mapping from qubit index to list of variable indices.

    Returns:
        Mapping from variable index to the number of variables on its qubit.
    """
    var_occupancy: dict[int, int] = {}
    for var_list in color_group.values():
        k = len(var_list)
        for var_idx in var_list:
            var_occupancy[var_idx] = k
    return var_occupancy


class QRAC31Encoder:
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
        linear_coeff_scale: Scaling factor for linear terms (√3)
        quad_coeff_scale: Scaling factor for quadratic terms (3)
    """

    max_color_group_size: int = 3
    linear_coeff_scale: float = sqrt(3)
    quad_coeff_scale: float = 3.0

    def __init__(self, spin_model: BinaryModel) -> None:
        """Initialize encoder with a spin model.

        Args:
            spin_model: BinaryModel in SPIN vartype
        """
        self.spin_model = spin_model
        self._color_group: dict[int, list[int]] = {}
        self._pauli_encoding: dict[int, tuple[int, PauliType]] = {}

        # Relaxed Hamiltonian coefficients
        self._linear_hamiltonian: dict[tuple[int, PauliType], float] = {}
        self._quad_hamiltonian: dict[
            tuple[tuple[int, PauliType], tuple[int, PauliType]], float
        ] = {}
        self._constant: float = 0.0

        self._perform_encoding()

    def _perform_encoding(self) -> None:
        """Perform graph coloring and Pauli encoding."""
        # 1. Graph coloring on interaction graph
        edges = list(self.spin_model.quad.keys())
        _, self._color_group = greedy_graph_coloring(edges, self.max_color_group_size)

        # 2. Add linear-only variables to color groups
        self._color_group = check_linear_term(
            self._color_group,
            list(self.spin_model.linear.keys()),
            self.max_color_group_size,
        )

        # 3. Assign Pauli operators to variables
        paulis: list[PauliType] = ["Z", "X", "Y"]
        for qubit, var_list in self._color_group.items():
            for pauli_idx, var_idx in enumerate(var_list):
                self._pauli_encoding[var_idx] = (qubit, paulis[pauli_idx])

        # 4. Build relaxed Hamiltonian
        self._build_relaxed_hamiltonian()

    def _build_relaxed_hamiltonian(self) -> None:
        """Compute relaxed Hamiltonian coefficients with per-occupancy scaling."""
        var_occupancy = _build_var_occupancy(self._color_group)

        # Linear terms: √k_i * h_i * P_{f(i)}
        for var_idx, coeff in self.spin_model.linear.items():
            if is_close_zero(coeff):
                continue
            qubit, pauli = self._pauli_encoding[var_idx]
            key = (qubit, pauli)
            k = var_occupancy[var_idx]
            self._linear_hamiltonian[key] = (
                self._linear_hamiltonian.get(key, 0.0) + np.sqrt(k) * coeff
            )

        # Quadratic terms: √k_i * √k_j * J_{ij} * P_{f(i)} * P_{f(j)}
        for (i, j), coeff in self.spin_model.quad.items():
            if is_close_zero(coeff):
                continue
            qi, pi = self._pauli_encoding[i]
            qj, pj = self._pauli_encoding[j]
            # Canonical ordering for consistent keys
            term_i = (qi, pi)
            term_j = (qj, pj)
            key = (term_i, term_j) if term_i <= term_j else (term_j, term_i)
            ki, kj = var_occupancy[i], var_occupancy[j]
            self._quad_hamiltonian[key] = (
                self._quad_hamiltonian.get(key, 0.0) + np.sqrt(ki) * np.sqrt(kj) * coeff
            )

        self._constant = self.spin_model.constant

    @property
    def num_qubits(self) -> int:
        """Number of qubits after QRAC encoding."""
        return len(self._color_group)

    @property
    def color_group(self) -> dict[int, list[int]]:
        """Mapping from qubit index to list of variable indices."""
        return self._color_group

    @property
    def pauli_encoding(self) -> dict[int, tuple[int, PauliType]]:
        """Mapping from variable index to (qubit, pauli_type)."""
        return self._pauli_encoding

    @property
    def linear_hamiltonian(self) -> dict[tuple[int, PauliType], float]:
        """Linear terms of relaxed Hamiltonian: {(qubit, pauli): coeff}."""
        return self._linear_hamiltonian

    @property
    def quad_hamiltonian(
        self,
    ) -> dict[tuple[tuple[int, PauliType], tuple[int, PauliType]], float]:
        """Quadratic terms of relaxed Hamiltonian: {((q1, p1), (q2, p2)): coeff}."""
        return self._quad_hamiltonian

    @property
    def constant(self) -> float:
        """Constant term of relaxed Hamiltonian."""
        return self._constant

    def get_pauli_for_variable(self, var_idx: int) -> tuple[int, PauliType]:
        """Get Pauli operator for a variable.

        Args:
            var_idx: Variable index

        Returns:
            Tuple of (qubit_index, pauli_type)
        """
        return self._pauli_encoding[var_idx]


class QRAC21Encoder:
    """(2,1,p)-QRAC Encoder.

    Encodes Ising model variables into Pauli operators using 2-coloring.
    Up to 2 variables per qubit, using Z and X Paulis.

    The relaxed Hamiltonian is:
        H̃ = Σ_{ij} √k_i·√k_j·J_{ij}·P_{f(i)}·P_{f(j)} + Σ_i √k_i·h_i·P_{f(i)}

    where k_i is the number of variables encoded on the qubit containing variable i.
    """

    max_color_group_size: int = 2
    linear_coeff_scale: float = sqrt(2)
    quad_coeff_scale: float = 2.0

    def __init__(self, spin_model: BinaryModel) -> None:
        self.spin_model = spin_model
        self._color_group: dict[int, list[int]] = {}
        self._pauli_encoding: dict[int, tuple[int, PauliType]] = {}
        self._linear_hamiltonian: dict[tuple[int, PauliType], float] = {}
        self._quad_hamiltonian: dict[
            tuple[tuple[int, PauliType], tuple[int, PauliType]], float
        ] = {}
        self._constant: float = 0.0
        self._perform_encoding()

    def _perform_encoding(self) -> None:
        edges = list(self.spin_model.quad.keys())
        _, self._color_group = greedy_graph_coloring(edges, self.max_color_group_size)
        self._color_group = check_linear_term(
            self._color_group,
            list(self.spin_model.linear.keys()),
            self.max_color_group_size,
        )
        paulis: list[PauliType] = ["Z", "X"]
        for qubit, var_list in self._color_group.items():
            for pauli_idx, var_idx in enumerate(var_list):
                self._pauli_encoding[var_idx] = (qubit, paulis[pauli_idx])
        self._build_relaxed_hamiltonian()

    def _build_relaxed_hamiltonian(self) -> None:
        var_occupancy = _build_var_occupancy(self._color_group)

        for var_idx, coeff in self.spin_model.linear.items():
            if is_close_zero(coeff):
                continue
            qubit, pauli = self._pauli_encoding[var_idx]
            key = (qubit, pauli)
            k = var_occupancy[var_idx]
            self._linear_hamiltonian[key] = (
                self._linear_hamiltonian.get(key, 0.0) + np.sqrt(k) * coeff
            )
        for (i, j), coeff in self.spin_model.quad.items():
            if is_close_zero(coeff):
                continue
            qi, pi = self._pauli_encoding[i]
            qj, pj = self._pauli_encoding[j]
            term_i = (qi, pi)
            term_j = (qj, pj)
            key = (term_i, term_j) if term_i <= term_j else (term_j, term_i)
            ki, kj = var_occupancy[i], var_occupancy[j]
            self._quad_hamiltonian[key] = (
                self._quad_hamiltonian.get(key, 0.0) + np.sqrt(ki) * np.sqrt(kj) * coeff
            )
        self._constant = self.spin_model.constant

    @property
    def num_qubits(self) -> int:
        return len(self._color_group)

    @property
    def color_group(self) -> dict[int, list[int]]:
        return self._color_group

    @property
    def pauli_encoding(self) -> dict[int, tuple[int, PauliType]]:
        return self._pauli_encoding

    @property
    def linear_hamiltonian(self) -> dict[tuple[int, PauliType], float]:
        return self._linear_hamiltonian

    @property
    def quad_hamiltonian(
        self,
    ) -> dict[tuple[tuple[int, PauliType], tuple[int, PauliType]], float]:
        return self._quad_hamiltonian

    @property
    def constant(self) -> float:
        return self._constant

    def get_pauli_for_variable(self, var_idx: int) -> tuple[int, PauliType]:
        return self._pauli_encoding[var_idx]


class QRAC32Encoder:
    """(3,2,p)-QRAC Encoder.

    Same graph coloring as (3,1,p) but uses 2-local prime operators.
    Each logical qubit maps to 2 physical qubits.

    The relaxed Hamiltonian is:
        H̃ = Σ_{ij} 6·J_{ij}·P'_{f(i)}·P'_{f(j)} + Σ_i √6·h_i·P'_{f(i)}

    where P' are 2-local prime operators (X', Y', Z').
    """

    max_color_group_size: int = 3
    linear_coeff_scale: float = sqrt(6)
    quad_coeff_scale: float = 6.0

    def __init__(self, spin_model: BinaryModel) -> None:
        self.spin_model = spin_model
        self._color_group: dict[int, list[int]] = {}
        self._pauli_encoding: dict[int, tuple[int, PauliType]] = {}
        self._perform_encoding()

    def _perform_encoding(self) -> None:
        edges = list(self.spin_model.quad.keys())
        _, self._color_group = greedy_graph_coloring(edges, self.max_color_group_size)
        self._color_group = check_linear_term(
            self._color_group,
            list(self.spin_model.linear.keys()),
            self.max_color_group_size,
        )
        paulis: list[PauliType] = ["Z", "X", "Y"]
        for qubit, var_list in self._color_group.items():
            for pauli_idx, var_idx in enumerate(var_list):
                self._pauli_encoding[var_idx] = (qubit, paulis[pauli_idx])

    @property
    def num_qubits(self) -> int:
        """Number of physical qubits (2x logical qubits)."""
        return len(self._color_group) * 2

    @property
    def num_logical_qubits(self) -> int:
        return len(self._color_group)

    @property
    def color_group(self) -> dict[int, list[int]]:
        return self._color_group

    @property
    def pauli_encoding(self) -> dict[int, tuple[int, PauliType]]:
        return self._pauli_encoding

    def get_pauli_for_variable(self, var_idx: int) -> tuple[int, PauliType]:
        return self._pauli_encoding[var_idx]


class QRACSpaceEfficientEncoder:
    """Space Efficient QRAC Encoder.

    No graph coloring. Uses sequential numbering:
    variable i -> qubit i//2, Pauli X (even index) or Y (odd index).

    Always maintains a 2:1 compression ratio.
    """

    linear_coeff_scale: float = sqrt(3)

    def __init__(self, spin_model: BinaryModel) -> None:
        self.spin_model = spin_model
        self._pauli_encoding: dict[int, tuple[int, PauliType]] = {}
        self._perform_encoding()

    def _perform_encoding(self) -> None:
        paulis: list[PauliType] = ["X", "Y"]
        num_vars = self.spin_model.num_bits
        for i in range(num_vars):
            qubit_index = i // 2
            self._pauli_encoding[i] = (qubit_index, paulis[i % 2])

    @property
    def num_qubits(self) -> int:
        num_vars = self.spin_model.num_bits
        return (num_vars + 1) // 2

    @property
    def pauli_encoding(self) -> dict[int, tuple[int, PauliType]]:
        return self._pauli_encoding

    def get_pauli_for_variable(self, var_idx: int) -> tuple[int, PauliType]:
        return self._pauli_encoding[var_idx]
