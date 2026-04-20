"""Private FWHT helpers used by :mod:`qamomile.linalg.hermitian`.

This module has no public API. It exposes a small set of routines that
compute the exact Pauli decomposition coefficients of a dense Hermitian
matrix via the Fast Walsh-Hadamard Transform, running in
``O(n * 4^n)`` time (``n`` is the number of qubits) with only NumPy.

The caller is responsible for validating that the input is Hermitian;
these helpers only guard against the residual imaginary part that would
appear if the assumption were violated.
"""

from __future__ import annotations

import numpy as np

_PHASE_TABLE = np.array([1.0, -1j, -1.0, 1j], dtype=np.complex128)


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def fwht_inplace(vec: np.ndarray) -> None:
    """Apply an in-place unnormalized Fast Walsh-Hadamard Transform.

    Length must be a power of two. After calling,
    ``vec[s] == sum_q original_vec[q] * (-1)**popcount(q & s)``.
    """
    length = vec.shape[0]
    step = 1
    while step < length:
        for start in range(0, length, 2 * step):
            x = vec[start : start + step].copy()
            y = vec[start + step : start + 2 * step].copy()
            vec[start : start + step] = x + y
            vec[start + step : start + 2 * step] = x - y
        step *= 2


def fwht_pauli_coefficients(
    matrix: np.ndarray,
    *,
    imag_tol: float = 1e-10,
) -> tuple[np.ndarray, int]:
    """Compute real Pauli coefficients ``alpha[x_mask, z_mask]`` via FWHT.

    Uses the symplectic indexing where qubit ``k`` is determined by bit
    ``k`` of each mask::

        (x_bit, z_bit) -> Pauli
        (0, 0) -> I
        (1, 0) -> X
        (0, 1) -> Z
        (1, 1) -> Y

    Qubit 0 corresponds to the least-significant bit of the matrix's
    computational-basis indices.

    Args:
        matrix: Dense ``(N, N)`` array with ``N = 2**n``. Assumed Hermitian.
        imag_tol: If any coefficient has ``|Im| > imag_tol`` a ``ValueError``
            is raised, which acts as a late guard against non-Hermitian input.

    Returns:
        A pair ``(coeffs, num_qubits)`` where ``coeffs`` is an ``(N, N)``
        real ``float64`` ndarray indexed by ``(x_mask, z_mask)``.
    """
    m = np.asarray(matrix, dtype=np.complex128)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"matrix must be 2D square; got shape {m.shape}.")
    dim = m.shape[0]
    if not is_power_of_two(dim):
        raise ValueError(f"matrix dimension must be a power of 2; got {dim}.")
    num_qubits = dim.bit_length() - 1

    coeffs = np.empty((dim, dim), dtype=np.complex128)
    idx = np.arange(dim, dtype=np.int64)

    for r in range(dim):
        v = m[idx ^ r, idx].astype(np.complex128, copy=True)
        fwht_inplace(v)
        # TODO(perf): this Python-level generator runs dim times per row, so
        # the total cost is O(4**n) Python iterations — the dominant
        # bottleneck when scaling up. Replace with a vectorized popcount
        # (e.g. a precomputed table or ``np.unpackbits``-based reduction) if
        # larger ``n`` becomes a target.
        popcounts = np.fromiter(
            ((r & s).bit_count() for s in range(dim)),
            dtype=np.int64,
            count=dim,
        )
        phases = _PHASE_TABLE[popcounts & 3]
        coeffs[r, :] = phases * v / dim

    max_imag = float(np.max(np.abs(coeffs.imag)))
    if max_imag > imag_tol:
        raise ValueError(
            f"Pauli decomposition produced non-negligible imaginary "
            f"coefficients (max |Im| = {max_imag:.3e}); input matrix is "
            f"likely not Hermitian."
        )

    return coeffs.real.astype(np.float64, copy=False), num_qubits
