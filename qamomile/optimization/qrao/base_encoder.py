"""QRAC Encoder base classes and utilities.

This module provides:
- ``BaseQRACEncoder``: Abstract base for all QRAC encoders with validation.
- ``GraphColoringQRACEncoder``: Intermediate base for graph-coloring-based encoders.
- Utility functions for Pauli encoding, occupancy mapping, and physical qubit mapping.

Concrete encoder classes live alongside their converters:
- ``QRAC21Encoder`` in ``qrao21.py``
- ``QRAC31Encoder`` in ``qrao31.py``
- ``QRAC32Encoder`` in ``qrao32.py``
- ``QRACSpaceEfficientEncoder`` in ``qrao_space_efficient.py``
"""

from __future__ import annotations

import abc
from typing import Literal

import qamomile.observable as qm_o
from qamomile.optimization.binary_model import BinaryModel, VarType
from qamomile.optimization.utils import is_close_zero

from .graph_coloring import check_linear_term, greedy_graph_coloring

PauliType = Literal["X", "Y", "Z"]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def color_group_to_qrac_encode(
    color_group: dict[int, list[int]],
) -> dict[int, qm_o.PauliOperator]:
    """Encode a color group mapping into Pauli operator assignments.

    Assigns Z, X, Y Pauli operators (in order) to variables within each
    qubit's color group.

    Args:
        color_group: Mapping from qubit index to list of variable indices.

    Returns:
        Mapping from variable index to PauliOperator.

    Examples:
        >>> color_group = {0: [0, 1, 2], 1: [3, 4], 2: [6,]}
        >>> color_group_to_qrac_encode(color_group)
        {0: Z0, 1: X0, 2: Y0, 3: Z1, 4: X1, 6: Z2}

    """
    encoded = {}
    paulis = [qm_o.Pauli.Z, qm_o.Pauli.X, qm_o.Pauli.Y]
    for color, group in color_group.items():
        for ope_idx, bit_index in enumerate(group):
            encoded[bit_index] = qm_o.PauliOperator(paulis[ope_idx], color)
    return encoded


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


# ---------------------------------------------------------------------------
# Base encoder classes
# ---------------------------------------------------------------------------


class BaseQRACEncoder(abc.ABC):
    """Abstract base for all QRAC encoders.

    Provides:
    - Input validation (spin vartype, no HUBO terms).
    - Common ``pauli_encoding`` property and ``get_pauli_for_variable`` method.
    - Abstract hooks for ``_perform_encoding`` and ``num_qubits``.

    Subclasses must implement ``_perform_encoding`` and ``num_qubits``.
    """

    def __init__(self, spin_model: BinaryModel) -> None:
        self._validate(spin_model)
        self.spin_model = spin_model
        self._pauli_encoding: dict[int, tuple[int, PauliType]] = {}
        self._perform_encoding()

    @staticmethod
    def _validate(spin_model: BinaryModel) -> None:
        """Validate that the model is suitable for QRAC encoding."""
        if spin_model.vartype != VarType.SPIN:
            raise ValueError("Encoder requires a SPIN-type BinaryModel.")
        if spin_model.higher:
            raise ValueError(
                "Encoder does not support higher-order (HUBO) terms. "
                "All interaction terms must be at most quadratic."
            )

    @abc.abstractmethod
    def _perform_encoding(self) -> None:
        """Perform the QRAC encoding (populate ``_pauli_encoding``)."""
        ...

    @property
    @abc.abstractmethod
    def num_qubits(self) -> int:
        """Number of qubits after QRAC encoding."""
        ...

    @property
    def pauli_encoding(self) -> dict[int, tuple[int, PauliType]]:
        """Mapping from variable index to (qubit, pauli_type)."""
        return self._pauli_encoding

    def get_pauli_for_variable(self, var_idx: int) -> tuple[int, PauliType]:
        """Get Pauli operator for a variable.

        Args:
            var_idx: Variable index

        Returns:
            Tuple of (qubit_index, pauli_type)
        """
        return self._pauli_encoding[var_idx]


class GraphColoringQRACEncoder(BaseQRACEncoder, abc.ABC):
    """Base for graph-coloring-based QRAC encoders (21, 31, 32).

    Subclasses only need to set ``max_color_group_size`` and ``paulis``.

    The encoding process:
    1. Graph coloring on the interaction graph.
    2. Adds linear-only variables to color groups.
    3. Assigns Pauli operators to variables within each qubit.
    """

    max_color_group_size: int
    """Maximum number of variables per qubit."""

    paulis: list[PauliType]
    """Pauli types to assign within each color group (in order)."""

    def __init__(self, spin_model: BinaryModel) -> None:
        self._color_group: dict[int, list[int]] = {}
        super().__init__(spin_model)

    def _perform_encoding(self) -> None:
        """Perform graph coloring and Pauli encoding."""
        edges = list(self.spin_model.quad.keys())
        _, self._color_group = greedy_graph_coloring(edges, self.max_color_group_size)
        self._color_group = check_linear_term(
            self._color_group,
            list(self.spin_model.linear.keys()),
            self.max_color_group_size,
        )
        for qubit, var_list in self._color_group.items():
            for pauli_idx, var_idx in enumerate(var_list):
                self._pauli_encoding[var_idx] = (qubit, self.paulis[pauli_idx])

    @property
    def color_group(self) -> dict[int, list[int]]:
        """Mapping from qubit index to list of variable indices."""
        return self._color_group

    @property
    def num_logical_qubits(self) -> int:
        """Number of logical qubits (color groups) after QRAC encoding."""
        return len(self._color_group)

    def encode_ising(
        self,
        ising: BinaryModel,
    ) -> tuple[qm_o.Hamiltonian, dict[int, qm_o.PauliOperator]]:
        """Encode a spin model into a relaxed QRAC Hamiltonian.

        Uses the encoding determined by ``_perform_encoding`` and delegates
        operator/scale computation to ``_get_operator_and_scale``.

        Args:
            ising: BinaryModel in SPIN vartype.

        Returns:
            Tuple of (relaxed Hamiltonian, encoding map).
        """
        encoded_ope = color_group_to_qrac_encode(self._color_group)
        var_occupancy = _build_var_occupancy(self._color_group)

        hamiltonian = qm_o.Hamiltonian()
        hamiltonian.constant = ising.constant

        for idx, coeff in ising.linear.items():
            if is_close_zero(coeff):
                continue
            op, scale = self._get_operator_and_scale(
                encoded_ope[idx], var_occupancy[idx]
            )
            hamiltonian += (coeff * scale) * op

        for (i, j), coeff in ising.quad.items():
            if is_close_zero(coeff):
                continue
            if i == j:
                hamiltonian.constant += coeff
                continue
            op_i, scale_i = self._get_operator_and_scale(
                encoded_ope[i], var_occupancy[i]
            )
            op_j, scale_j = self._get_operator_and_scale(
                encoded_ope[j], var_occupancy[j]
            )
            hamiltonian += (coeff * scale_i * scale_j) * (op_i * op_j)

        return hamiltonian, encoded_ope

    @abc.abstractmethod
    def _get_operator_and_scale(
        self,
        pauli: qm_o.PauliOperator,
        k: int,
    ) -> tuple[qm_o.Hamiltonian, float]:
        """Return the operator and scaling factor for a variable.

        Args:
            pauli: PauliOperator assigned to this variable.
            k: Number of variables sharing this qubit (occupancy).

        Returns:
            Tuple of (operator Hamiltonian, scale factor).
        """
        ...
