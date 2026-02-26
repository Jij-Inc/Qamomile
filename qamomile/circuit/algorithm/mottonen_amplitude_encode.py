"""Quantum state preparation algorithms.

Available:
    Classes:
        - MottonenAmplitudeEncode: Möttönen amplitude encoding (CompositeGate)

    Functions:
        - amplitude_encoding: Apply Möttönen amplitude encoding to qubits

Reference:
    Möttönen et al., "Transformation of quantum states using uniformly
    controlled rotations", arXiv:quant-ph/0407010
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import qamomile.circuit as qmc
from qamomile.circuit.frontend.composite_gate import CompositeGate
from qamomile.circuit.frontend.handle import Float, Qubit, Vector
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateType,
    ResourceMetadata,
)


def _gray_code(n: int) -> int:
    """Return the Gray code of integer n."""
    return n ^ (n >> 1)


def _get_cnot_controls(k: int) -> list[int]:
    """Generate CNOT control qubit indices for k control bits.

    Returns a list of length 2^k indicating which control qubit
    (0..k-1) to use for each step in the uniformly controlled rotation.
    """
    controls = []
    for i in range(1, 2**k + 1):
        if i == 2**k:
            controls.append(k - 1)
        else:
            controls.append((i & -i).bit_length() - 1)
    return controls


def _compute_angle_transform_matrix(k: int) -> np.ndarray:
    """Compute the 2^k x 2^k angle transformation matrix M.

    This matrix transforms the rotation angles alpha (computed from
    sub-vector norms) into theta angles for the Gray-code-ordered
    uniformly controlled rotations.

    Uses the transposed form M^T so that the decomposition maps
    each control state s directly to alpha[s] without additional
    permutation beyond bit-reversal of the alpha indices.
    """
    size = 2**k
    M = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            bit_count = bin(j & _gray_code(i)).count("1")
            M[i, j] = (2**-k) * ((-1) ** bit_count)
    return M


def _compute_all_thetas(amplitudes: np.ndarray, num_qubits: int) -> list[np.ndarray]:
    """Pre-compute all rotation angles for the Möttönen decomposition.

    Args:
        amplitudes: Normalized state vector of length 2^num_qubits.
        num_qubits: Number of qubits.

    Returns:
        List of theta arrays, one per qubit level k.
    """
    all_thetas: list[np.ndarray] = []
    for k in range(num_qubits):
        chunk_size = 2 ** (num_qubits - k)
        half_chunk = chunk_size // 2

        alphas = []
        for i in range(2**k):
            chunk = amplitudes[i * chunk_size : (i + 1) * chunk_size]
            norm_chunk = np.linalg.norm(chunk)
            norm_bottom = np.linalg.norm(chunk[half_chunk:])
            if norm_chunk > 0:
                alpha = 2 * np.arcsin(norm_bottom / norm_chunk)
            else:
                alpha = 0.0
            alphas.append(alpha)
        alphas_arr = np.array(alphas)

        if k == 0:
            thetas = alphas_arr
        else:
            # Bit-reverse alpha indices to align chunk ordering
            # (MSB-first) with Gray code convention (LSB-first).
            alphas_br = np.zeros_like(alphas_arr)
            for j in range(2**k):
                rev_j = int(format(j, f"0{k}b")[::-1], 2)
                alphas_br[j] = alphas_arr[rev_j]
            M = _compute_angle_transform_matrix(k)
            thetas = M @ alphas_br

        all_thetas.append(thetas)
    return all_thetas


class MottonenAmplitudeEncode(CompositeGate):
    """Möttönen amplitude encoding (arXiv:quant-ph/0407010).

    Prepares an arbitrary quantum state from the |0...0> state using
    uniformly controlled Y-rotations decomposed via Gray code ordering.

    The input amplitudes are automatically normalized.

    Example::

        gate = MottonenAmplitudeEncode([1.0, 0.0, 0.0, 1.0])
        q0, q1 = gate(q0, q1)
    """

    gate_type = CompositeGateType.CUSTOM
    custom_name = "mottonen_state_prep"

    _strategies: dict = {}  # type: ignore[type-arg]
    _default_strategy = "standard"

    def __init__(self, amplitudes: Sequence[float] | np.ndarray):
        arr = np.array(amplitudes, dtype=float)
        n = len(arr)
        if n == 0 or (n & (n - 1)) != 0:
            raise ValueError(f"Length of amplitudes must be a power of 2, got {n}")
        norm = np.linalg.norm(arr)
        if norm == 0:
            raise ValueError("Amplitudes must not be all zeros")
        self._amplitudes = arr / norm
        self._num_qubits = int(np.log2(n))
        self._all_thetas = _compute_all_thetas(self._amplitudes, self._num_qubits)

    @property
    def num_target_qubits(self) -> int:
        return self._num_qubits

    def _decompose(
        self,
        qubits: tuple[Qubit, ...],
    ) -> tuple[Qubit, ...]:
        """Decompose into RY and CNOT gates following Gray code ordering."""
        n = self._num_qubits
        q = list(qubits)

        for k in range(n):
            thetas = self._all_thetas[k]
            tgt = n - 1 - k

            if k == 0:
                q[tgt] = qmc.ry(q[tgt], float(thetas[0]))
            else:
                cnot_seq = _get_cnot_controls(k)
                for i in range(2**k):
                    q[tgt] = qmc.ry(q[tgt], float(thetas[i]))
                    ctrl = n - 1 - cnot_seq[i]
                    q[ctrl], q[tgt] = qmc.cx(q[ctrl], q[tgt])

        return tuple(q)

    def _resources(self) -> ResourceMetadata:
        n = self._num_qubits
        # RY gates: 1 (k=0) + sum_{k=1}^{n-1} 2^k = 2^n - 1
        num_ry = 2**n - 1
        # CNOT gates: sum_{k=1}^{n-1} 2^k = 2^n - 2
        num_cnot = 2**n - 2
        return ResourceMetadata(
            t_gate_count=0,
            custom_metadata={
                "num_ry_gates": num_ry,
                "num_cnot_gates": num_cnot,
                "total_gates": num_ry + num_cnot,
                "num_qubits": n,
            },
        )


def compute_mottonen_thetas(amplitudes: Sequence[float] | np.ndarray) -> np.ndarray:
    """Pre-compute flat rotation angles for Möttönen amplitude encoding.

    Converts a classical amplitude vector into the rotation angles
    needed by :func:`parametric_amplitude_encoding`.

    Args:
        amplitudes: Classical vector of length 2^n. Will be normalized.

    Returns:
        1-D numpy array of length 2^n - 1.
    """
    arr = np.array(amplitudes, dtype=float)
    n = len(arr)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Length of amplitudes must be a power of 2, got {n}")
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("Amplitudes must not be all zeros")
    arr = arr / norm
    num_qubits = int(np.log2(n))
    all_thetas = _compute_all_thetas(arr, num_qubits)
    return np.concatenate(all_thetas)


def _get_size(arr: Vector[Qubit]) -> int:
    """Get array size as Python int."""
    size = arr.shape[0]
    if isinstance(size, int):
        return size
    if hasattr(size, "value") and size.value.is_constant():
        val = size.value.get_const()
        if val is not None:
            return int(val)
    if hasattr(size, "init_value"):
        return int(size.init_value)
    raise ValueError("Array must have fixed size")


def amplitude_encoding(
    amplitudes: Sequence[float] | np.ndarray,
    qubits: Vector[Qubit],
) -> Vector[Qubit]:
    """Apply Möttönen amplitude encoding to a qubit register.

    Prepares the quantum state whose amplitudes match the given
    classical vector (automatically normalized).

    Args:
        amplitudes: Classical vector of length 2^n. Will be normalized.
        qubits: Vector of n qubits, all in |0> state.

    Returns:
        Qubit vector encoding the normalized amplitudes.
    """
    n = _get_size(qubits)
    gate = MottonenAmplitudeEncode(amplitudes)
    if gate.num_target_qubits != n:
        raise ValueError(
            f"Need {gate.num_target_qubits} qubits for {len(amplitudes)} "
            f"amplitudes, got {n}"
        )

    qubit_list: list[Qubit] = [qubits[i] for i in range(n)]
    result = gate(*qubit_list)
    for i in range(n):
        qubits[i] = result[i]
    return qubits


def parametric_amplitude_encoding(
    thetas: Vector[Float],
    qubits: Vector[Qubit],
) -> Vector[Qubit]:
    """Parametric Möttönen amplitude encoding.

    Uses pre-computed rotation angles (from :func:`compute_mottonen_thetas`)
    as ``Vector[Float]`` parameters, enabling parametric quantum circuits.

    Args:
        thetas: Rotation angles of length 2^n - 1, computed by
            :func:`compute_mottonen_thetas`.
        qubits: Vector of n qubits, all in |0> state.

    Returns:
        Qubit vector encoding the state corresponding to *thetas*.
    """
    n = _get_size(qubits)
    q: list[Qubit] = [qubits[i] for i in range(n)]

    theta_idx = 0
    for k in range(n):
        tgt = n - 1 - k
        if k == 0:
            q[tgt] = qmc.ry(q[tgt], thetas[theta_idx])
            theta_idx += 1
        else:
            cnot_seq = _get_cnot_controls(k)
            for i in range(2**k):
                q[tgt] = qmc.ry(q[tgt], thetas[theta_idx])
                theta_idx += 1
                ctrl = n - 1 - cnot_seq[i]
                q[ctrl], q[tgt] = qmc.cx(q[ctrl], q[tgt])

    for i in range(n):
        qubits[i] = q[i]
    return qubits
