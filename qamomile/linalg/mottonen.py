"""Möttönen-Vartiainen amplitude-encoding angle computation.

Pure classical preprocessing for the Möttönen amplitude-encoding
construction.  Takes an amplitude vector and returns the Gray-walk
``Ry`` (and, for complex inputs, ``Rz``) angle vectors that the
Möttönen Gray-code emission consumes — assuming that the target
register starts in :math:`|0\\rangle^{\\otimes n}`.  See
:mod:`qamomile.circuit.stdlib.state_preparation.mottonen_amplitude_encoding`
for the gate-emission side and the all-zero-state pre-condition.

The math here is intentionally separated from gate emission so it
can also be used standalone — e.g., by hybrid-optimisation loops
that pre-compute angle vectors outside any kernel and then feed
them into ``mottonen_amplitude_encoding_from_angles`` via
``parameters=[...]``.

Reference
---------

M. Möttönen, J. J. Vartiainen, V. Bergholm, M. M. Salomaa,
"Transformation of quantum states using uniformly controlled
rotations", arXiv:quant-ph/0407010 (2004).  Equation / section
numbers cited in individual function docstrings below refer to this
paper.  In particular:

* The Gray-code decomposition of a ``k``-control uniformly
  controlled rotation into ``2^k`` single-qubit rotations + ``2^k``
  CNOTs is described in **Section II**, around **Fig. 2** and the
  paragraph immediately after **Eq. (2)** (the paper does not number
  this as a Lemma / Theorem).
* The angle-basis transform ``θ = M^(k) α`` linking per-state
  rotation angles ``α`` to elementary Gray-walk angles ``θ`` is
  **Eq. (3)** of Section II.
* Application to state preparation, including the iterative
  disentangling for general (complex) amplitudes, is in
  **Section III**, **Eqs. (4)-(8)**.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = [
    "validate_and_normalize_amplitudes",
    "compute_mottonen_amplitude_encoding_ry_angles",
    "compute_mottonen_amplitude_encoding_rz_angles",
    "compute_all_ry_angles_per_level",
    "compute_disentangling_angles_per_level",
]


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _gray_code(n: int) -> int:
    """Return the binary-reflected Gray code of *n*.

    Used to index the elementary Gray-walk angles consumed by the
    Möttönen-Vartiainen decomposition (arXiv:quant-ph/0407010,
    Section II — the binary reflected Gray code ``g_i`` appears in
    the exponent of **Eq. (3)**).

    Args:
        n (int): Non-negative integer to encode.

    Returns:
        int: The Gray-code image ``n ^ (n >> 1)``.
    """
    return n ^ (n >> 1)


def _bit_reverse(value: int, num_bits: int) -> int:
    """Reverse the *num_bits* lowest bits of *value*.

    This is **not a step the paper itself names** — it is an
    implementation-side bridge between two indexing conventions.  The
    recursive amplitude-chunk layout used by
    :func:`compute_all_ry_angles_per_level` /
    :func:`compute_disentangling_angles_per_level` numbers chunks
    big-endian (chunk index 0 = controls all zero in the
    most-significant order), while the Walsh-Hadamard transform used for
    paper **Eq. (3)** is
    indexed in the little-endian convention.  Bit-reversing the
    chunk index converts between the two before applying ``M^(k)``.

    Args:
        value (int): Non-negative integer whose low bits are reversed.
        num_bits (int): Width of the field to reverse over.

    Returns:
        int: The integer obtained by reversing the bottom *num_bits* of
            *value*.
    """
    return int(format(value, f"0{num_bits}b")[::-1], 2)


def _fast_walsh_hadamard(values: np.ndarray) -> np.ndarray:
    """Apply the unnormalized Walsh-Hadamard transform to a copy.

    The dense matrix in paper Eq. (3) is a Hadamard matrix whose rows are
    permuted by Gray code. Butterfly additions compute its ordinary
    Walsh-Hadamard part in ``O(N log N)`` time and ``O(N)`` working memory,
    avoiding the former ``O(N**2)`` matrix and unbounded cache.

    Args:
        values (np.ndarray): One-dimensional, power-of-two-length input.

    Returns:
        np.ndarray: Unnormalized Hadamard transform of ``values``.
    """
    transformed = np.asarray(values, dtype=float).copy()
    half_width = 1
    while half_width < len(transformed):
        block_width = 2 * half_width
        for start in range(0, len(transformed), block_width):
            left = transformed[start : start + half_width].copy()
            right = transformed[start + half_width : start + block_width].copy()
            transformed[start : start + half_width] = left + right
            transformed[start + half_width : start + block_width] = left - right
        half_width = block_width
    return transformed


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_and_normalize_amplitudes(
    amplitudes: Sequence[float] | Sequence[complex] | np.ndarray,
) -> tuple[np.ndarray, int, bool]:
    """Validate an amplitude vector and return its normalised form.

    The Möttönen construction (arXiv:quant-ph/0407010, Section III)
    works on a unit-norm amplitude vector indexed by computational
    basis states; this helper enforces that pre-condition so the
    downstream angle computation can assume a well-formed input.

    Inputs may be real or complex.  A complex input whose imaginary
    part is identically zero (within ``np.allclose`` tolerance) is
    coerced to a real array; this preserves the cheaper signed-RY fast
    path for vectors that happen to arrive boxed as ``complex``.

    Args:
        amplitudes (Sequence[float] | Sequence[complex] | np.ndarray):
            Amplitude vector.  Must be a 1-D sequence whose length is a
            power of two and at least 2, with at least one non-zero
            entry.

    Returns:
        tuple[np.ndarray, int, bool]:
            ``(normalized, num_qubits, is_complex)`` where
            ``normalized`` is a unit-norm ``np.ndarray``,
            ``num_qubits`` is ``log2(len(amplitudes))``, and
            ``is_complex`` is ``True`` iff the input has a non-zero
            imaginary component (and therefore needs the
            phase-restoration stage downstream).  When ``is_complex``
            is ``False``, ``normalized.dtype`` is ``float``; otherwise
            it is ``complex``.

    Raises:
        ValueError: If the input is not a 1-D vector (e.g., a nested
            sequence or a 2-D ``np.ndarray``), the length is not a
            power of two (or is less than 2, i.e., would map to a
            zero-qubit register), or all amplitudes are zero.
    """
    arr = np.asarray(amplitudes)

    # Reject nested sequences / 2-D arrays up front so the user sees a
    # clear shape error here rather than a confusing TypeError deeper in
    # the angle-computation pipeline (e.g. `math.atan2(np.ndarray, ...)`).
    if arr.ndim != 1:
        raise ValueError(
            f"Amplitudes must be a 1-D vector, got an array of shape {arr.shape}"
        )

    if np.iscomplexobj(arr):
        if np.allclose(np.imag(arr), 0.0):
            arr = np.real(arr).astype(float)
            is_complex = False
        else:
            arr = arr.astype(complex)
            is_complex = True
    else:
        arr = arr.astype(float)
        is_complex = False

    n = len(arr)
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError(
            f"Length of amplitudes must be a power of 2 and at least 2, got {n}"
        )

    norm = float(np.linalg.norm(arr))
    if math.isclose(norm, 0.0, abs_tol=1e-15):
        raise ValueError("Amplitudes must not be all zeros")

    return arr / norm, int(round(math.log2(n))), is_complex


# ---------------------------------------------------------------------------
# Per-level angle computation
# ---------------------------------------------------------------------------


def _to_gray_walk_basis(
    per_level_angles: list[np.ndarray],
    num_qubits: int,
) -> list[np.ndarray]:
    """Convert per-control-state angles to the Gray-walk angle basis.

    Per-level application of paper **Eq. (3)** (Möttönen et al.,
    arXiv:quant-ph/0407010, Section II):
    ``θ = M^(k) α``.  The Gray-walk emission interleaves ``2^k``
    elementary rotations with ``2^k`` CNOTs in a Gray-code order
    (Section II, Fig. 2); ``M^(k)`` is a normalized Hadamard transform
    with Gray-permuted rows. It is evaluated without materializing a dense
    matrix. The transform maps natural per-control-state angles ``α`` to
    elementary Gray-walk angles ``θ``. Used identically by the Ry
    (magnitude) and Rz (phase) stages.

    The bit-reverse step is **not in the paper as a numbered step**
    (see :func:`_bit_reverse`); it is an implementation bridge
    between the recursive big-endian chunk indexing used by
    :func:`compute_all_ry_angles_per_level` /
    :func:`compute_disentangling_angles_per_level` and the
    little-endian indexing assumed by ``M^(k)``.

    Args:
        per_level_angles (list[np.ndarray]): Per-level angle arrays in
            the per-control-state ordering; the ``k``-th entry has
            length ``2**k``.
        num_qubits (int): Number of qubits (``len(per_level_angles)``).

    Returns:
        list[np.ndarray]: Same shape as *per_level_angles*, with each
            level (for ``k >= 1``) replaced by the Gray-walk-basis
            angles.  The ``k = 0`` entry is returned unchanged.
    """
    transformed: list[np.ndarray] = []
    for k in range(num_qubits):
        raw = per_level_angles[k]
        if k == 0:
            transformed.append(raw)
            continue
        size = 2**k
        bit_reversed = np.empty(size)
        for j in range(size):
            bit_reversed[j] = raw[_bit_reverse(j, k)]
        hadamard = _fast_walsh_hadamard(bit_reversed)
        gray_walk = np.empty(size)
        for i in range(size):
            gray_walk[i] = hadamard[_gray_code(i)] / size
        transformed.append(gray_walk)
    return transformed


def compute_all_ry_angles_per_level(
    amplitudes: np.ndarray,
    num_qubits: int,
) -> list[np.ndarray]:
    """Pre-compute every level's Ry rotation angle vector (magnitude stage).

    Implements the recursive magnitude-stage angle formula given in
    Möttönen et al., arXiv:quant-ph/0407010, **Section III, Eq. (8)**
    (with the leaf case matching the unnumbered angle
    ``2 \\arcsin(|a_{2j}| / \\sqrt{|a_{2j-1}|^2 + |a_{2j}|^2})``
    just before **Eq. (6)**, equivalent to ``2 atan2(|a_1|, |a_0|)``
    on the leaf pair).  The pre-condition is the all-zero state
    :math:`|0\\rangle^{\\otimes n}`; the formulas below describe the
    angles needed to reach a real ``amplitudes`` from there.

    For each level ``k`` (``0 <= k < num_qubits``) the amplitude
    vector is split into ``2**k`` equal chunks.  Each chunk yields one
    per-control-state angle ``α`` (Eq. (8)):

    * **Intermediate levels** (``chunk_size > 2``): the angle rotates
      the target qubit so its ``|1>`` weight matches the lower-half
      block norm.  Using ``arctan2(norm_lower, norm_upper)`` keeps the
      formula well defined when the upper half has zero norm.
    * **Leaf level** (``chunk_size == 2``):
      ``α = 2 * arctan2(a_1, a_0)`` directly (signed, so negative
      amplitudes are preserved without a phase stage).

    For ``k >= 1`` the per-control-state angles are then transformed
    to the Gray-walk basis via :func:`_to_gray_walk_basis` (paper
    **Eq. (3)**).

    Args:
        amplitudes (np.ndarray): Unit-norm real amplitude vector of
            length ``2**num_qubits``.
        num_qubits (int): Number of qubits in the target register.

    Returns:
        list[np.ndarray]: A list of ``num_qubits`` arrays; the ``k``-th
            entry has length ``2**k`` and holds the Gray-walk Ry
            angles for that level.
    """
    ry_angles_per_level: list[np.ndarray] = [np.empty(2**k) for k in range(num_qubits)]
    for k in range(num_qubits):
        chunk_size = 2 ** (num_qubits - k)
        half_chunk = chunk_size // 2

        per_chunk_angles = ry_angles_per_level[k]
        for i in range(2**k):
            chunk = amplitudes[i * chunk_size : (i + 1) * chunk_size]
            if half_chunk == 1:
                per_chunk_angles[i] = 2.0 * math.atan2(float(chunk[1]), float(chunk[0]))
            else:
                norm_upper = float(np.linalg.norm(chunk[:half_chunk]))
                norm_lower = float(np.linalg.norm(chunk[half_chunk:]))
                per_chunk_angles[i] = 2.0 * math.atan2(norm_lower, norm_upper)

    return _to_gray_walk_basis(ry_angles_per_level, num_qubits)


def compute_disentangling_angles_per_level(
    amplitudes: np.ndarray,
    num_qubits: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Iteratively disentangle to obtain both Ry and Rz angles per level.

    Implements the Möttönen disentangling sweep for general (complex)
    amplitudes from Möttönen et al., arXiv:quant-ph/0407010,
    **Section III, Eqs. (4)-(8)**: the phase-equalisation rotation
    ``Ξ_z`` (Eq. (4)) followed by the magnitude rotation, applied
    pair-by-pair from LSB to MSB.  Pre-condition is the all-zero
    state :math:`|0\\rangle^{\\otimes n}`; the angles below are
    those needed to *reach* the input ``amplitudes`` from there
    (computed via the inverse sweep that disentangles the input).

    At each step the amplitude vector is halved by pairing adjacent
    entries; for the pair ``(a_0, a_1)`` we read off

    * ``Ry angle = 2 * arctan2(|a_1|, |a_0|)`` (magnitude split,
      paper **Eq. (8)** restricted to the leaf level — equivalently
      the unnumbered ``2 arcsin(...)`` form just before **Eq. (6)**),
    * ``Rz angle = arg(a_1 / a_0)`` (phase difference, paper
      **Eq. (5)**),

    and replace the pair with the single complex amplitude that
    survives the implicit ``Rz^{-1} Ry^{-1}`` disentangling step
    (paper **Eq. (6)** zeros out one of the pair entries; the
    surviving amplitude is

    .. math::

        \\beta = \\sqrt{|a_0|^2 + |a_1|^2} \\,
                 \\exp\\!\\left(i\\,\\frac{\\arg a_0 + \\arg a_1}{2}\\right),

    which carries the averaged phase needed by the next outer step
    of the sweep — the paper does not write ``β`` in this exact
    closed form but the relation falls out of combining
    Eqs. (5)-(7)).  The full sweep is paper **Eq. (7)**.

    Zero-magnitude halves are handled gracefully by leaving the
    corresponding angle at ``0`` (the underlying state has no support
    there so the angle is immaterial).  The returned per-level arrays
    are already bit-reversed and gray-walk-transformed (paper
    **Eq. (3)** applied per level via :func:`_to_gray_walk_basis`),
    ready for the Möttönen Gray-walk emission.

    Args:
        amplitudes (np.ndarray): Unit-norm complex amplitude vector of
            length ``2**num_qubits``.
        num_qubits (int): Number of qubits in the target register.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]:
            ``(ry_angles_per_level, rz_angles_per_level)`` in
            Gray-walk ordering.  Each list has ``num_qubits`` entries;
            the ``k``-th entry holds ``2**k`` angles for level ``k`` of
            the forward emission.
    """
    ry_angles_per_level: list[np.ndarray] = [np.empty(2**k) for k in range(num_qubits)]
    rz_angles_per_level: list[np.ndarray] = [np.empty(2**k) for k in range(num_qubits)]

    current = amplitudes.astype(complex).copy()
    for j in range(num_qubits):
        # Forward level k = num_qubits - 1 - j has 2**(num_qubits-1-j) = num_pairs angles.
        forward_level = num_qubits - 1 - j
        num_pairs = len(current) // 2
        ry_at_level = ry_angles_per_level[forward_level]
        rz_at_level = rz_angles_per_level[forward_level]
        new_current = np.empty(num_pairs, dtype=complex)

        for c in range(num_pairs):
            a0 = current[2 * c]
            a1 = current[2 * c + 1]
            mag0 = float(abs(a0))
            mag1 = float(abs(a1))
            mag = math.sqrt(mag0 * mag0 + mag1 * mag1)
            ry_at_level[c] = 2.0 * math.atan2(mag1, mag0)

            if mag0 > 1e-15 and mag1 > 1e-15:
                rz_at_level[c] = float(np.angle(a1 / a0))
                avg_phase = float(np.angle(a0)) + rz_at_level[c] / 2.0
            elif mag0 > 1e-15:
                rz_at_level[c] = 0.0
                avg_phase = float(np.angle(a0))
            elif mag1 > 1e-15:
                rz_at_level[c] = 0.0
                avg_phase = float(np.angle(a1))
            else:
                rz_at_level[c] = 0.0
                avg_phase = 0.0

            new_current[c] = mag * np.exp(1j * avg_phase)

        current = new_current

    return (
        _to_gray_walk_basis(ry_angles_per_level, num_qubits),
        _to_gray_walk_basis(rz_angles_per_level, num_qubits),
    )


# ---------------------------------------------------------------------------
# Public flat-vector entry points
# ---------------------------------------------------------------------------


def compute_mottonen_amplitude_encoding_ry_angles(
    amplitudes: Sequence[float] | Sequence[complex] | np.ndarray,
) -> np.ndarray:
    """Pre-compute the flat Ry angle vector for a Möttönen encoding.

    Returns the Gray-walk Ry angles for the magnitude stage of the
    encoding.  For real inputs (or complex inputs with zero imaginary
    part) this corresponds to the single-stage signed-Ry encoding.
    For complex inputs with non-zero imaginary part, these are the
    magnitude-stage angles only — pair with
    :func:`compute_mottonen_amplitude_encoding_rz_angles` to obtain
    the phase-stage angles needed to reproduce the full complex
    state.

    Args:
        amplitudes (Sequence[float] | Sequence[complex] | np.ndarray):
            Amplitude vector of length ``2**n``.  Real or complex; it
            is normalised automatically.

    Returns:
        np.ndarray: 1-D array of length ``2**n - 1`` holding the
            Gray-walk Ry angles laid out level by level.

    Raises:
        ValueError: If the input is not a 1-D vector, the length is
            not a power of two (or is less than 2, i.e., would map to
            a zero-qubit register), or all amplitudes are zero.
    """
    normalized, num_qubits, is_complex = validate_and_normalize_amplitudes(amplitudes)
    if is_complex:
        ry_angles_per_level, _ = compute_disentangling_angles_per_level(
            normalized, num_qubits
        )
    else:
        ry_angles_per_level = compute_all_ry_angles_per_level(normalized, num_qubits)
    return np.concatenate(ry_angles_per_level)


def compute_mottonen_amplitude_encoding_rz_angles(
    amplitudes: Sequence[float] | Sequence[complex] | np.ndarray,
) -> np.ndarray:
    """Pre-compute the flat Rz angle vector for the phase-restoration stage.

    Returns the Gray-walk Rz angles for the phase stage of the
    Möttönen encoding.  For real inputs (or complex inputs with zero
    imaginary part) the phase stage is unnecessary and the returned
    array is all zeros.  For complex inputs with non-zero imaginary
    part, the returned angles together with the Ry angles from
    :func:`compute_mottonen_amplitude_encoding_ry_angles` reproduce
    the full complex state.

    Args:
        amplitudes (Sequence[float] | Sequence[complex] | np.ndarray):
            Amplitude vector of length ``2**n``.  Real or complex; it
            is normalised automatically.

    Returns:
        np.ndarray: 1-D array of length ``2**n - 1`` holding the
            Gray-walk Rz angles laid out level by level (all zeros for
            real inputs).

    Raises:
        ValueError: If the input is not a 1-D vector, the length is
            not a power of two (or is less than 2, i.e., would map to
            a zero-qubit register), or all amplitudes are zero.
    """
    normalized, num_qubits, is_complex = validate_and_normalize_amplitudes(amplitudes)
    if not is_complex:
        return np.zeros(2**num_qubits - 1)
    _, rz_angles_per_level = compute_disentangling_angles_per_level(
        normalized, num_qubits
    )
    return np.concatenate(rz_angles_per_level)
