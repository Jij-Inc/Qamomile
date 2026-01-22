"""Gate specifications for transpiler testing.

This module defines gate specifications aligned with GateKind enum from
qamomile/circuit/transpiler/gate_emitter.py, including expected unitary
matrices for verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import numpy as np


class GateCategory(Enum):
    """Categories of quantum gates for test organization."""

    SINGLE_QUBIT = auto()  # H, X, Y, Z, S, T
    SINGLE_QUBIT_ROTATION = auto()  # RX, RY, RZ, P
    TWO_QUBIT = auto()  # CX, CZ, SWAP
    TWO_QUBIT_ROTATION = auto()  # CP, RZZ
    THREE_QUBIT = auto()  # TOFFOLI
    CONTROLLED = auto()  # CH, CY, CRX, CRY, CRZ
    MEASUREMENT = auto()


@dataclass
class GateSpec:
    """Specification for a gate type.

    Attributes:
        name: Gate name (matches GateKind enum name)
        category: Gate category for test organization
        num_qubits: Number of qubit operands
        has_angle: Whether the gate has an angle parameter
        matrix_fn: Function to compute the unitary matrix
                   For non-parametric gates: takes no args
                   For parametric gates: takes angle as argument
    """

    name: str
    category: GateCategory
    num_qubits: int
    has_angle: bool = False
    matrix_fn: Callable[..., np.ndarray] | None = None


# Standard qubit states
KET_0: np.ndarray = np.array([1, 0], dtype=complex)
KET_1: np.ndarray = np.array([0, 1], dtype=complex)

# Standard projection matrices
PROJ_0: np.ndarray = np.array([[1, 0], [0, 0]], dtype=complex)  # |0><0|
PROJ_1: np.ndarray = np.array([[0, 0], [0, 1]], dtype=complex)  # |1><1|


# =============================================================================
# Single-qubit gate matrices
# =============================================================================


def h_matrix() -> np.ndarray:
    """Hadamard gate."""
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def x_matrix() -> np.ndarray:
    """Pauli-X gate."""
    return np.array([[0, 1], [1, 0]], dtype=complex)


def y_matrix() -> np.ndarray:
    """Pauli-Y gate."""
    return np.array([[0, -1j], [1j, 0]], dtype=complex)


def z_matrix() -> np.ndarray:
    """Pauli-Z gate."""
    return np.array([[1, 0], [0, -1]], dtype=complex)


def s_matrix() -> np.ndarray:
    """S gate (sqrt(Z))."""
    return np.array([[1, 0], [0, 1j]], dtype=complex)


def t_matrix() -> np.ndarray:
    """T gate (sqrt(S))."""
    return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


# =============================================================================
# Single-qubit rotation gate matrices
# =============================================================================


def rx_matrix(theta: float) -> np.ndarray:
    """RX rotation gate: exp(-i * theta/2 * X)."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry_matrix(theta: float) -> np.ndarray:
    """RY rotation gate: exp(-i * theta/2 * Y)."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz_matrix(theta: float) -> np.ndarray:
    """RZ rotation gate: exp(-i * theta/2 * Z)."""
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex
    )


def p_matrix(theta: float) -> np.ndarray:
    """Phase gate: diag(1, e^(i*theta))."""
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)


# =============================================================================
# Two-qubit gate matrices (Qiskit little-endian convention: q0 is LSB)
# State ordering: |q1 q0> where index = q1*2 + q0
# Index 0 = |00>, Index 1 = |01>, Index 2 = |10>, Index 3 = |11>
# =============================================================================


def cx_matrix() -> np.ndarray:
    """CNOT gate (control=0, target=1) in little-endian convention.

    When q0=1 (control on), flip q1.
    |00> → |00>, |01> → |11>, |10> → |10>, |11> → |01>
    """
    return np.array(
        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex
    )


def cz_matrix() -> np.ndarray:
    """CZ gate - symmetric, same in both conventions."""
    return np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex
    )


def swap_matrix() -> np.ndarray:
    """SWAP gate - symmetric, same in both conventions."""
    return np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
    )


# =============================================================================
# Two-qubit rotation gate matrices (little-endian convention)
# =============================================================================


def cp_matrix(theta: float) -> np.ndarray:
    """Controlled-Phase gate (control=0, target=1) in little-endian convention.

    Phase applied when both qubits are 1 (index 3).
    """
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, np.exp(1j * theta)],
        ],
        dtype=complex,
    )


def rzz_matrix(theta: float) -> np.ndarray:
    """RZZ gate: exp(-i * theta/2 * Z ⊗ Z).

    Diagonal matrix, same in both conventions.
    """
    return np.array(
        [
            [np.exp(-1j * theta / 2), 0, 0, 0],
            [0, np.exp(1j * theta / 2), 0, 0],
            [0, 0, np.exp(1j * theta / 2), 0],
            [0, 0, 0, np.exp(-1j * theta / 2)],
        ],
        dtype=complex,
    )


# =============================================================================
# Three-qubit gate matrices (little-endian convention)
# State ordering: |q2 q1 q0> where index = q2*4 + q1*2 + q0
# =============================================================================


def toffoli_matrix() -> np.ndarray:
    """Toffoli (CCX) gate (control1=0, control2=1, target=2) in little-endian.

    Flip q2 when both q0=1 and q1=1.
    |000> → |000> (0→0), |001> → |001> (1→1), |010> → |010> (2→2),
    |011> → |111> (3→7), |100> → |100> (4→4), |101> → |101> (5→5),
    |110> → |110> (6→6), |111> → |011> (7→3)
    """
    mat = np.eye(8, dtype=complex)
    # Swap indices 3 and 7
    mat[3, 3] = 0
    mat[3, 7] = 1
    mat[7, 7] = 0
    mat[7, 3] = 1
    return mat


# =============================================================================
# Controlled single-qubit gate matrices (little-endian convention)
# For controlled gates with control=0, target=1:
# The gate applies to target when control (q0) is 1.
# In little-endian: indices 1 (|01>) and 3 (|11>) have control=1.
# =============================================================================


def ch_matrix() -> np.ndarray:
    """Controlled-Hadamard gate (control=0, target=1) in little-endian.

    Apply H to q1 when q0=1.
    Affects indices 1 (|01>) and 3 (|11>).
    """
    h = h_matrix()
    mat = np.eye(4, dtype=complex)
    # Apply H in the subspace where q0=1 (indices 1 and 3)
    mat[1, 1] = h[0, 0]  # |01>→|01> component
    mat[1, 3] = h[0, 1]  # |11>→|01> component
    mat[3, 1] = h[1, 0]  # |01>→|11> component
    mat[3, 3] = h[1, 1]  # |11>→|11> component
    return mat


def cy_matrix() -> np.ndarray:
    """Controlled-Y gate (control=0, target=1) in little-endian.

    Apply Y to q1 when q0=1.
    """
    y = y_matrix()
    mat = np.eye(4, dtype=complex)
    mat[1, 1] = y[0, 0]  # |01>→|01> component
    mat[1, 3] = y[0, 1]  # |11>→|01> component
    mat[3, 1] = y[1, 0]  # |01>→|11> component
    mat[3, 3] = y[1, 1]  # |11>→|11> component
    return mat


def crx_matrix(theta: float) -> np.ndarray:
    """Controlled-RX gate (control=0, target=1) in little-endian.

    Apply RX to q1 when q0=1.
    """
    rx = rx_matrix(theta)
    mat = np.eye(4, dtype=complex)
    mat[1, 1] = rx[0, 0]
    mat[1, 3] = rx[0, 1]
    mat[3, 1] = rx[1, 0]
    mat[3, 3] = rx[1, 1]
    return mat


def cry_matrix(theta: float) -> np.ndarray:
    """Controlled-RY gate (control=0, target=1) in little-endian.

    Apply RY to q1 when q0=1.
    """
    ry = ry_matrix(theta)
    mat = np.eye(4, dtype=complex)
    mat[1, 1] = ry[0, 0]
    mat[1, 3] = ry[0, 1]
    mat[3, 1] = ry[1, 0]
    mat[3, 3] = ry[1, 1]
    return mat


def crz_matrix(theta: float) -> np.ndarray:
    """Controlled-RZ gate (control=0, target=1) in little-endian.

    Apply RZ to q1 when q0=1.
    """
    rz = rz_matrix(theta)
    mat = np.eye(4, dtype=complex)
    mat[1, 1] = rz[0, 0]
    mat[1, 3] = rz[0, 1]
    mat[3, 1] = rz[1, 0]
    mat[3, 3] = rz[1, 1]
    return mat


# =============================================================================
# Gate specifications registry
# =============================================================================

GATE_SPECS: dict[str, GateSpec] = {
    # Single-qubit gates
    "H": GateSpec("H", GateCategory.SINGLE_QUBIT, 1, matrix_fn=h_matrix),
    "X": GateSpec("X", GateCategory.SINGLE_QUBIT, 1, matrix_fn=x_matrix),
    "Y": GateSpec("Y", GateCategory.SINGLE_QUBIT, 1, matrix_fn=y_matrix),
    "Z": GateSpec("Z", GateCategory.SINGLE_QUBIT, 1, matrix_fn=z_matrix),
    "S": GateSpec("S", GateCategory.SINGLE_QUBIT, 1, matrix_fn=s_matrix),
    "T": GateSpec("T", GateCategory.SINGLE_QUBIT, 1, matrix_fn=t_matrix),
    # Single-qubit rotation gates
    "RX": GateSpec(
        "RX", GateCategory.SINGLE_QUBIT_ROTATION, 1, has_angle=True, matrix_fn=rx_matrix
    ),
    "RY": GateSpec(
        "RY", GateCategory.SINGLE_QUBIT_ROTATION, 1, has_angle=True, matrix_fn=ry_matrix
    ),
    "RZ": GateSpec(
        "RZ", GateCategory.SINGLE_QUBIT_ROTATION, 1, has_angle=True, matrix_fn=rz_matrix
    ),
    "P": GateSpec(
        "P", GateCategory.SINGLE_QUBIT_ROTATION, 1, has_angle=True, matrix_fn=p_matrix
    ),
    # Two-qubit gates
    "CX": GateSpec("CX", GateCategory.TWO_QUBIT, 2, matrix_fn=cx_matrix),
    "CZ": GateSpec("CZ", GateCategory.TWO_QUBIT, 2, matrix_fn=cz_matrix),
    "SWAP": GateSpec("SWAP", GateCategory.TWO_QUBIT, 2, matrix_fn=swap_matrix),
    # Two-qubit rotation gates
    "CP": GateSpec(
        "CP", GateCategory.TWO_QUBIT_ROTATION, 2, has_angle=True, matrix_fn=cp_matrix
    ),
    "RZZ": GateSpec(
        "RZZ", GateCategory.TWO_QUBIT_ROTATION, 2, has_angle=True, matrix_fn=rzz_matrix
    ),
    # Three-qubit gates
    "TOFFOLI": GateSpec("TOFFOLI", GateCategory.THREE_QUBIT, 3, matrix_fn=toffoli_matrix),
    # Controlled single-qubit gates
    "CH": GateSpec("CH", GateCategory.CONTROLLED, 2, matrix_fn=ch_matrix),
    "CY": GateSpec("CY", GateCategory.CONTROLLED, 2, matrix_fn=cy_matrix),
    "CRX": GateSpec(
        "CRX", GateCategory.CONTROLLED, 2, has_angle=True, matrix_fn=crx_matrix
    ),
    "CRY": GateSpec(
        "CRY", GateCategory.CONTROLLED, 2, has_angle=True, matrix_fn=cry_matrix
    ),
    "CRZ": GateSpec(
        "CRZ", GateCategory.CONTROLLED, 2, has_angle=True, matrix_fn=crz_matrix
    ),
    # Measurement (special case - no matrix)
    "MEASURE": GateSpec("MEASURE", GateCategory.MEASUREMENT, 1, matrix_fn=None),
}

# Test angles for parametric gates
TEST_ANGLES: list[float] = [0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]


# =============================================================================
# Utility functions for statevector comparison
# =============================================================================


def statevectors_equal(
    sv1: np.ndarray, sv2: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8
) -> bool:
    """Check if two statevectors are equal up to global phase.

    Args:
        sv1: First statevector
        sv2: Second statevector
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if statevectors are equal up to global phase
    """
    sv1 = np.asarray(sv1, dtype=complex).flatten()
    sv2 = np.asarray(sv2, dtype=complex).flatten()

    if sv1.shape != sv2.shape:
        return False

    # Find first non-zero element for phase alignment
    for i in range(len(sv1)):
        if np.abs(sv1[i]) > atol and np.abs(sv2[i]) > atol:
            # Align global phase
            phase = sv2[i] / sv1[i]
            sv1_aligned = sv1 * phase
            return np.allclose(sv1_aligned, sv2, rtol=rtol, atol=atol)

    # Both vectors are essentially zero
    return np.allclose(sv1, sv2, rtol=rtol, atol=atol)


def compute_expected_statevector(
    initial_state: np.ndarray, unitary: np.ndarray
) -> np.ndarray:
    """Compute expected statevector after applying unitary.

    Args:
        initial_state: Initial statevector
        unitary: Unitary matrix to apply

    Returns:
        Resulting statevector
    """
    return unitary @ initial_state


def tensor_product(*matrices: np.ndarray) -> np.ndarray:
    """Compute tensor product of multiple matrices.

    Args:
        matrices: Matrices to tensor together (left to right order)

    Returns:
        Tensor product result
    """
    result = matrices[0]
    for mat in matrices[1:]:
        result = np.kron(result, mat)
    return result


def identity(n: int = 2) -> np.ndarray:
    """Return n x n identity matrix."""
    return np.eye(n, dtype=complex)


# =============================================================================
# Initial state preparation
# =============================================================================


def all_zeros_state(num_qubits: int) -> np.ndarray:
    """Return |00...0> state vector."""
    state = np.zeros(2**num_qubits, dtype=complex)
    state[0] = 1
    return state


def computational_basis_state(num_qubits: int, index: int) -> np.ndarray:
    """Return computational basis state |index>.

    Args:
        num_qubits: Number of qubits
        index: Index of the computational basis state (0 to 2^n - 1)

    Returns:
        State vector
    """
    state = np.zeros(2**num_qubits, dtype=complex)
    state[index] = 1
    return state


def plus_state() -> np.ndarray:
    """Return |+> = (|0> + |1>) / sqrt(2)."""
    return np.array([1, 1], dtype=complex) / np.sqrt(2)


def minus_state() -> np.ndarray:
    """Return |-> = (|0> - |1>) / sqrt(2)."""
    return np.array([1, -1], dtype=complex) / np.sqrt(2)


def bell_state(index: int = 0) -> np.ndarray:
    """Return Bell state.

    Args:
        index: 0 for |00>+|11>, 1 for |00>-|11>,
               2 for |01>+|10>, 3 for |01>-|10>

    Returns:
        Two-qubit Bell state vector
    """
    states = [
        np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),  # |00> + |11>
        np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),  # |00> - |11>
        np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),  # |01> + |10>
        np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),  # |01> - |10>
    ]
    return states[index]
