"""Private FWHT helpers for dense Pauli decompositions.

This module has no public API. It exposes a small set of routines that
compute the Pauli decomposition coefficients of a dense complex matrix via
the Fast Walsh-Hadamard Transform, running in ``O(n * 4^n)`` time (``n`` is
the number of qubits). Binary64 components are accumulated as exact integer
multiples of the smallest subnormal value, then rounded only once per output
coefficient. This prevents both butterfly overflow and loss of small terms
when much larger terms later cancel.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

_PHASE_TABLE = np.array([1.0, -1j, -1.0, 1j], dtype=np.complex128)
_BINARY64_SUBNORMAL_EXPONENT = 1074


def is_power_of_two(n: int) -> bool:
    """Check whether ``n`` is a positive power of two.

    Args:
        n (int): Integer to test.

    Returns:
        bool: ``True`` if ``n`` is a power of two, ``False`` otherwise.
    """
    return n > 0 and (n & (n - 1)) == 0


def _float_to_subnormal_units(value: float) -> int:
    """Convert one finite binary64 value to exact subnormal-sized units.

    Every finite binary64 value is an integer multiple of ``2**-1074``.
    Keeping that integer in Python's arbitrary-precision representation lets
    the FWHT retain small addends across cancellation with values near the
    largest finite exponent.

    Args:
        value (float): Finite Python float to convert.

    Returns:
        int: Exact integer ``k`` such that ``value == k * 2**-1074``.
    """
    numerator, denominator = value.as_integer_ratio()
    denominator_exponent = denominator.bit_length() - 1
    return numerator << (_BINARY64_SUBNORMAL_EXPONENT - denominator_exponent)


def _fwht_integer_inplace(values: list[int]) -> None:
    """Apply an exact unnormalized FWHT to integer accumulators in place.

    Args:
        values (list[int]): Power-of-two-length integer vector to transform.

    Returns:
        None: ``values`` is replaced by its exact Walsh-Hadamard transform.
    """
    step = 1
    while step < len(values):
        for start in range(0, len(values), 2 * step):
            for offset in range(step):
                left = values[start + offset]
                right = values[start + step + offset]
                values[start + offset] = left + right
                values[start + step + offset] = left - right
        step *= 2


def fwht_complex_pauli_coefficients(matrix: ArrayLike) -> tuple[np.ndarray, int]:
    """Compute complex Pauli coefficients via FWHT.

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
        matrix (ArrayLike): Dense finite ``(N, N)`` array with ``N = 2**n``.

    Returns:
        tuple[np.ndarray, int]: Pair ``(coeffs, num_qubits)`` where ``coeffs``
            is an ``(N, N)`` complex128 array indexed by
            ``(x_mask, z_mask)``.

    Raises:
        ValueError: If ``matrix`` is not a finite square matrix whose
            dimension is a positive power of two, or cannot be converted to
            complex numeric data.
    """
    try:
        m = np.asarray(matrix, dtype=np.complex128)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("matrix must contain numeric values.") from exc
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"matrix must be 2D square; got shape {m.shape}.")
    dim = m.shape[0]
    if not is_power_of_two(dim):
        raise ValueError(f"matrix dimension must be a power of 2; got {dim}.")
    if not np.all(np.isfinite(m)):
        raise ValueError("matrix entries must all be finite.")
    num_qubits = dim.bit_length() - 1

    coeffs = np.empty((dim, dim), dtype=np.complex128)
    idx = np.arange(dim, dtype=np.int64)
    coefficient_denominator = 1 << (_BINARY64_SUBNORMAL_EXPONENT + num_qubits)

    for r in range(dim):
        diagonal = m[idx ^ r, idx]
        real_units = [
            _float_to_subnormal_units(float(value)) for value in diagonal.real
        ]
        imag_units = [
            _float_to_subnormal_units(float(value)) for value in diagonal.imag
        ]
        _fwht_integer_inplace(real_units)
        _fwht_integer_inplace(imag_units)
        transformed = np.asarray(
            [
                complex(
                    real / coefficient_denominator,
                    imag / coefficient_denominator,
                )
                for real, imag in zip(real_units, imag_units, strict=True)
            ],
            dtype=np.complex128,
        )
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
        coeffs[r, :] = phases * transformed

    if not np.all(np.isfinite(coeffs)):
        raise ValueError("Pauli coefficients exceed the complex128 finite range.")

    return coeffs, num_qubits


def fwht_pauli_coefficients(
    matrix: np.ndarray,
    *,
    imag_tol: float = 1e-10,
) -> tuple[np.ndarray, int]:
    """Compute real Pauli coefficients for a Hermitian matrix via FWHT.

    Args:
        matrix (np.ndarray): Dense finite ``(N, N)`` Hermitian array with
            ``N = 2**n``.
        imag_tol (float): Maximum tolerated absolute imaginary residue in any
            coefficient. Defaults to ``1e-10``.

    Returns:
        tuple[np.ndarray, int]: Pair ``(coeffs, num_qubits)`` where ``coeffs``
            is an ``(N, N)`` float64 array indexed by
            ``(x_mask, z_mask)``.

    Raises:
        ValueError: If the matrix fails the generic complex decomposition
            validation or produces an imaginary coefficient larger than
            ``imag_tol``.
    """
    coeffs, num_qubits = fwht_complex_pauli_coefficients(matrix)
    max_imag = float(np.max(np.abs(coeffs.imag)))
    if max_imag > imag_tol:
        raise ValueError(
            f"Pauli decomposition produced non-negligible imaginary "
            f"coefficients (max |Im| = {max_imag:.3e}); input matrix is "
            f"likely not Hermitian."
        )

    return coeffs.real.astype(np.float64, copy=False), num_qubits
