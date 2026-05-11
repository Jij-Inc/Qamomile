"""State preparation via Möttönen's uniformly-controlled-rotation construction.

Prepares the n-qubit state

.. math::

    |\\psi\\rangle = \\sum_{i=0}^{2^n - 1} a_i \\,|i\\rangle

from :math:`|0\\rangle^{\\otimes n}` for a normalised amplitude vector
``a``.  The construction follows Möttönen et al., "Transformation of
quantum states using uniformly controlled rotations"
(arXiv:quant-ph/0407010).  Both the RY-only "amplitude distribution"
stage and the RZ "phase restoration" stage are supported, so general
complex amplitudes work end-to-end.

Pipeline
--------

1. Validate and normalise the input (length must be a power of two).
2. Determine whether the input is genuinely complex (has a non-zero
   imaginary part).  Real inputs (including complex with zero imag) keep
   the original signed-RY path — negative real amplitudes flow through
   the sign of ``arctan2(a_1, a_0)`` naturally, with no RZ overhead.
   Complex inputs use the iterative disentangling construction described
   below.
3. For real inputs, recursively compute the per-level RY rotation angles
   by splitting each chunk into upper / lower halves and using
   ``arctan2`` of the two sub-block norms (or signed ``arctan2`` at the
   leaf).
4. For complex inputs, iteratively disentangle the target amplitude
   vector qubit-by-qubit from LSB to MSB.  Each disentangling step
   pairs adjacent complex amplitudes and reads off
   ``ry_angle = 2 * arctan2(|a_1|, |a_0|)`` (magnitude split) and
   ``rz_angle = arg(a_1 / a_0)`` (phase difference, taken in
   ``(-π, π]``); the pair is then reduced to a single complex amplitude
   ``sqrt(|a_0|^2 + |a_1|^2) * exp(i * avg_phase)`` that feeds the
   next step, where ``avg_phase = arg(a_0) + rz_angle / 2`` when
   ``a_0 != 0`` (using ``np.angle`` on the principal branch); the
   one-side-zero cases use the non-zero side's argument and set
   ``rz_angle = 0``.
5. At each level ``k >= 1`` apply a uniformly controlled rotation over
   the previously prepared ``k`` qubits.  We use the standard Gray-code
   RY / CNOT decomposition for the magnitude stage and the same
   structure with RZ for the phase stage; the per-level angle vector is
   bit-reversed and then multiplied by the linear transform
   :math:`M^{(k)}` to obtain the angles consumed by the Gray walk.
   The emitted gate order is "all RY layers, then all RZ layers".
   Pairwise ``[U_y^(k), U_z^(k')]`` does NOT commute in general
   (including ``k != k'`` cases, because earlier RY targets can be
   controls of later RZ multiplexers), but the FULL sweep product
   equals the per-level interleaved product
   ``(U_z^(0) U_y^(0)) ... (U_z^(n-1) U_y^(n-1))`` as unitaries — this
   is a structural identity verified by
   :class:`tests.circuit.algorithm.state_preparation.test_mottonen_amplitude_encoding.TestRyRzOrdering`
   with arbitrary (non-disentangling) per-level angles.  Within each
   level the order RY-before-RZ is preserved in both schemes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from qamomile.circuit.frontend.composite_gate import CompositeGate
from qamomile.circuit.frontend.handle import Float, Qubit, Vector
from qamomile.circuit.frontend.handle.utils import get_size
from qamomile.circuit.frontend.operation.qubit_gates import cx, ry, rz
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateType,
    ResourceMetadata,
)

__all__ = [
    "MottonenAmplitudeEncoding",
    "amplitude_encoding",
    "amplitude_encoding_from_angles",
    "compute_mottonen_amplitude_encoding_ry_angles",
    "compute_mottonen_amplitude_encoding_rz_angles",
]


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _gray_code(n: int) -> int:
    """Return the binary-reflected Gray code of *n*.

    Args:
        n (int): Non-negative integer to encode.

    Returns:
        int: The Gray-code image ``n ^ (n >> 1)``.
    """
    return n ^ (n >> 1)


def _bit_reverse(value: int, num_bits: int) -> int:
    """Reverse the *num_bits* lowest bits of *value*.

    Args:
        value (int): Non-negative integer whose low bits are reversed.
        num_bits (int): Width of the field to reverse over.

    Returns:
        int: The integer obtained by reversing the bottom *num_bits* of
            *value*.
    """
    return int(format(value, f"0{num_bits}b")[::-1], 2)


def _get_cnot_controls(k: int) -> list[int]:
    """Compute the CNOT control positions of the Gray-code Ry walk.

    The Gray-code decomposition of a ``k``-control uniformly controlled
    :math:`R_y` interleaves ``2^k`` rotations with ``2^k`` CNOTs.  The
    ``i``-th CNOT toggles the control bit that differs between
    ``Gray(i-1)`` and ``Gray(i)``, which is the trailing-zero position of
    ``i`` for ``i < 2^k`` and ``k-1`` for the closing CNOT that returns
    to the starting Gray word.

    Args:
        k (int): Number of control qubits driving the uniformly
            controlled gate.

    Returns:
        list[int]: A list of length ``2**k`` whose ``i``-th entry is the
            index of the control qubit (in ``0..k-1``) that the ``i``-th
            CNOT acts upon.
    """
    controls = []
    for i in range(1, 2**k + 1):
        if i == 2**k:
            controls.append(k - 1)
        else:
            lowest_set_bit = i & -i
            controls.append(lowest_set_bit.bit_length() - 1)
    return controls


def _compute_angle_transform_matrix(k: int) -> np.ndarray:
    """Build the :math:`M^{(k)}` matrix that maps per-state angles to Gray-walk angles.

    The matrix is

    .. math::

        M^{(k)}_{ij} = \\frac{1}{2^k}\\,(-1)^{\\,b(g_i)\\cdot b(j)},

    where ``g_i`` is the Gray code of ``i`` and ``b(\\cdot)\\cdot b(\\cdot)``
    denotes the bitwise inner product modulo 2.  This is the closed-form
    transform from Möttönen-Vartiainen for decomposing a uniformly
    controlled rotation into elementary RY / CNOT gates.

    Args:
        k (int): Number of control qubits for the uniformly controlled
            rotation.

    Returns:
        np.ndarray: A ``(2**k, 2**k)`` array implementing the linear
            transform.
    """
    size = 2**k
    matrix = np.zeros((size, size))
    for i in range(size):
        g_i = _gray_code(i)
        for j in range(size):
            parity = bin(j & g_i).count("1") & 1
            matrix[i, j] = (2.0**-k) * (1.0 if parity == 0 else -1.0)
    return matrix


# ---------------------------------------------------------------------------
# Validation and angle computation
# ---------------------------------------------------------------------------


def _validate_and_normalize(
    amplitudes: Sequence[float] | Sequence[complex] | np.ndarray,
) -> tuple[np.ndarray, int, bool]:
    """Validate an amplitude vector and return its normalised form.

    Inputs may be real or complex.  A complex input whose imaginary part
    is identically zero (within ``np.allclose`` tolerance) is coerced to
    a real array; this preserves the original signed-RY fast path for
    vectors that happen to arrive boxed as ``complex``.

    Args:
        amplitudes (Sequence[float] | Sequence[complex] | np.ndarray):
            Amplitude vector.  Its length must be a power of two and at
            least one entry must be non-zero.

    Returns:
        tuple[np.ndarray, int, bool]: ``(normalized, num_qubits, is_complex)``
            where ``normalized`` is a unit-norm ``np.ndarray``,
            ``num_qubits`` is ``log2(len(amplitudes))``, and
            ``is_complex`` is True iff the input has a non-zero
            imaginary component (and therefore needs the phase-restoration
            stage).  When ``is_complex`` is False, ``normalized.dtype``
            is ``float``; otherwise it is ``complex``.

    Raises:
        ValueError: If the length is not a power of two (or is less
            than 2, i.e., would map to a zero-qubit register), or all
            amplitudes are zero.
    """
    arr = np.asarray(amplitudes)

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


def _compute_all_ry_angles(
    amplitudes: np.ndarray,
    num_qubits: int,
) -> list[np.ndarray]:
    """Pre-compute every level's Ry rotation angle vector (magnitude stage).

    For each level ``k`` (``0 <= k < num_qubits``) the amplitude vector
    is split into ``2**k`` equal chunks.  Each chunk yields one
    per-control-state angle (called ``alpha`` in Möttönen-Vartiainen):

    * **Intermediate levels** (``chunk_size > 2``): the angle rotates
      the target qubit so its ``|1>`` weight matches the lower-half
      block norm.  Using ``arctan2(norm_lower, norm_upper)`` keeps the
      formula well defined when the upper half has zero norm.
    * **Leaf level** (``chunk_size == 2``):
      ``alpha = 2 * arctan2(a_1, a_0)`` directly, which is signed and
      therefore preserves negative amplitudes.

    For ``k >= 1`` the per-control-state angles are then transformed to
    the Gray-walk basis via :func:`_to_gray_walk_basis`.

    Args:
        amplitudes (np.ndarray): Unit-norm real amplitude vector of
            length ``2**num_qubits``.
        num_qubits (int): Number of qubits in the target register.

    Returns:
        list[np.ndarray]: A list of ``num_qubits`` arrays; the ``k``-th
            entry has length ``2**k`` and holds the Gray-walk Ry angles
            for that level.
    """
    ry_angles_per_level: list[np.ndarray] = [np.empty(0)] * num_qubits
    for k in range(num_qubits):
        chunk_size = 2 ** (num_qubits - k)
        half_chunk = chunk_size // 2

        per_chunk_angles = np.empty(2**k)
        for i in range(2**k):
            chunk = amplitudes[i * chunk_size : (i + 1) * chunk_size]
            if half_chunk == 1:
                per_chunk_angles[i] = 2.0 * math.atan2(float(chunk[1]), float(chunk[0]))
            else:
                norm_upper = float(np.linalg.norm(chunk[:half_chunk]))
                norm_lower = float(np.linalg.norm(chunk[half_chunk:]))
                per_chunk_angles[i] = 2.0 * math.atan2(norm_lower, norm_upper)
        ry_angles_per_level[k] = per_chunk_angles

    return _to_gray_walk_basis(ry_angles_per_level, num_qubits)


def _to_gray_walk_basis(
    per_level_angles: list[np.ndarray],
    num_qubits: int,
) -> list[np.ndarray]:
    """Convert per-control-state angles to the Gray-walk angle basis.

    The Möttönen Gray-walk emission interleaves elementary rotations
    with CNOTs in a Gray-code order; the elementary angles consumed by
    that emission are obtained from the natural per-control-state
    angles by a level-wise bit-reverse permutation followed by
    multiplication by the level's transform matrix
    (see :func:`_compute_angle_transform_matrix`).  Used by both the Ry
    (magnitude) stage and the Rz (phase) stage; the structure is
    identical.

    The bit-reverse step converts between two indexing conventions: the
    recursive chunk layout uses big-endian control labels (chunk index
    0 = controls all zero in the most-significant order), while
    :func:`_emit_mottonen_gates` walks Gray codes in the little-endian
    convention.

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
        bit_reversed = np.empty(2**k)
        for j in range(2**k):
            bit_reversed[j] = raw[_bit_reverse(j, k)]
        transformed.append(_compute_angle_transform_matrix(k) @ bit_reversed)
    return transformed


def _compute_disentangling_angles(
    amplitudes: np.ndarray,
    num_qubits: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Iteratively disentangle to obtain both Ry and Rz angles per level.

    Implements the standard Möttönen disentangling sweep from LSB to
    MSB.  At each step the amplitude vector is halved by pairing
    adjacent entries; for the pair ``(a_0, a_1)`` we read off

    * ``Ry angle = 2 * arctan2(|a_1|, |a_0|)`` (magnitude split),
    * ``Rz angle = arg(a_1 / a_0)`` (phase difference),

    and replace the pair with the single complex amplitude that
    survives the implicit ``Rz^{-1} Ry^{-1}`` disentangling step:

    .. math::

        \\beta = \\sqrt{|a_0|^2 + |a_1|^2} \\,
                 \\exp\\!\\left(i\\,\\frac{\\arg a_0 + \\arg a_1}{2}\\right).

    Zero-magnitude halves are handled gracefully by leaving the
    corresponding angle at ``0`` (the underlying state has no support
    there so the angle is immaterial).  The returned per-level arrays
    are already bit-reversed and gray-walk-transformed, ready for
    :func:`_emit_mottonen_gates`.

    Args:
        amplitudes (np.ndarray): Unit-norm complex amplitude vector of
            length ``2**num_qubits``.
        num_qubits (int): Number of qubits in the target register.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]:
            ``(ry_angles_per_level, rz_angles_per_level)`` in Gray-walk
            ordering.  Each list has ``num_qubits`` entries; the
            ``k``-th entry holds ``2**k`` angles for level ``k`` of the
            forward emission.
    """
    ry_angles_per_level: list[np.ndarray] = [np.empty(0)] * num_qubits
    rz_angles_per_level: list[np.ndarray] = [np.empty(0)] * num_qubits

    current = amplitudes.astype(complex).copy()
    for j in range(num_qubits):
        num_pairs = len(current) // 2
        ry_at_level = np.empty(num_pairs)
        rz_at_level = np.empty(num_pairs)
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

        # Forward level k = num_qubits - 1 - j has 2**(num_qubits-1-j) = num_pairs angles.
        forward_level = num_qubits - 1 - j
        ry_angles_per_level[forward_level] = ry_at_level
        rz_angles_per_level[forward_level] = rz_at_level
        current = new_current

    return (
        _to_gray_walk_basis(ry_angles_per_level, num_qubits),
        _to_gray_walk_basis(rz_angles_per_level, num_qubits),
    )


# ---------------------------------------------------------------------------
# Quantum gate emission
# ---------------------------------------------------------------------------


def _emit_mottonen_gates(
    qubits: list[Qubit],
    num_qubits: int,
    angles: Sequence[float] | np.ndarray | Vector[Float],
    gate: str = "ry",
) -> None:
    """Emit a Gray-walk Ry / CNOT or Rz / CNOT sequence in place on *qubits*.

    Level ``k`` targets qubit ``num_qubits - 1 - k`` so that the first
    rotation lands on the most-significant qubit (consistent with the
    chunk-splitting convention used by :func:`_compute_all_ry_angles`
    and :func:`_compute_disentangling_angles`).

    *angles* may be either a concrete numerical sequence (Python list,
    NumPy array of dtype float) or a ``Vector[Float]`` handle — the
    latter is what makes the runtime-parametric path
    (:func:`amplitude_encoding_from_angles`) work: indexing a
    ``Vector[Float]`` with a Python ``int`` produces a ``Float`` handle
    that ``qmc.ry`` / ``qmc.rz`` accept directly.

    Args:
        qubits (list[Qubit]): Mutable list of length ``num_qubits`` of
            qubit handles.  Entries are overwritten as gates are applied.
        num_qubits (int): Number of qubits in the register.
        angles (Sequence[float] | np.ndarray | Vector[Float]): Flat
            sequence of rotation angles of length ``2**num_qubits - 1``,
            laid out level by level.
        gate (str): ``"ry"`` for the magnitude stage or ``"rz"`` for the
            phase stage.  Defaults to ``"ry"``.

    Returns:
        None: The *qubits* list is mutated in place.

    Raises:
        ValueError: If *gate* is not ``"ry"`` or ``"rz"``.
    """
    match gate:
        case "ry":
            rotation = ry
        case "rz":
            rotation = rz
        case _:
            raise ValueError(f"gate must be 'ry' or 'rz', got {gate!r}")

    idx = 0
    for k in range(num_qubits):
        tgt = num_qubits - 1 - k
        if k == 0:
            qubits[tgt] = rotation(qubits[tgt], angles[idx])
            idx += 1
            continue

        cnot_seq = _get_cnot_controls(k)
        for step in range(2**k):
            qubits[tgt] = rotation(qubits[tgt], angles[idx])
            idx += 1
            ctrl = num_qubits - 1 - cnot_seq[step]
            qubits[ctrl], qubits[tgt] = cx(qubits[ctrl], qubits[tgt])


# ---------------------------------------------------------------------------
# Composite gate
# ---------------------------------------------------------------------------


class MottonenAmplitudeEncoding(CompositeGate):
    """Möttönen amplitude encoding for normalised real or complex vectors.

    Prepares the state :math:`\\sum_i a_i |i\\rangle` from
    :math:`|0\\rangle^{\\otimes n}` using uniformly controlled Y (and,
    for genuinely complex inputs, Z) rotations decomposed into ``RY`` /
    ``RZ`` and ``CNOT`` gates with Gray-code ordering.

    Notes:
        * Input amplitudes are normalised automatically.
        * Real inputs (negative entries allowed) use a single RY stage:
          the sign of ``arctan2`` at the leaf preserves negative signs
          natively, so no RZ overhead is incurred.
        * Complex inputs with non-zero imaginary part use the full
          two-stage construction (RY for magnitudes, RZ for phases).
        * Complex inputs whose imaginary part is identically zero (e.g.
          ``[1+0j, -1+0j]``) are silently coerced to real and follow the
          single-stage path.

    Example::

        # Real amplitudes (signed allowed)
        gate = MottonenAmplitudeEncoding([1.0, 0.0, 0.0, 1.0])
        q0, q1 = gate(q0, q1)

        # Complex amplitudes
        gate = MottonenAmplitudeEncoding([1+0j, 1j, -1+0j, -1j])
        q0, q1 = gate(q0, q1)
    """

    gate_type = CompositeGateType.CUSTOM
    custom_name = "mottonen_amplitude_encoding"

    def __init__(self, amplitudes: Sequence[float] | Sequence[complex] | np.ndarray):
        """Initialise the gate with a concrete amplitude vector.

        Args:
            amplitudes (Sequence[float] | Sequence[complex] | np.ndarray):
                Amplitude vector of length ``2**n``.  Real or complex; it
                is automatically normalised.  Complex inputs with zero
                imaginary part are coerced to real (single-stage RY path).

        Raises:
            ValueError: If the length is not a power of two (or is
                less than 2, i.e., would map to a zero-qubit register),
                or all amplitudes are zero.
        """
        self._amplitudes, self._num_qubits, self._is_complex = _validate_and_normalize(
            amplitudes
        )
        self._rz_angles_per_level: list[np.ndarray] | None
        if self._is_complex:
            (
                self._ry_angles_per_level,
                self._rz_angles_per_level,
            ) = _compute_disentangling_angles(self._amplitudes, self._num_qubits)
        else:
            self._ry_angles_per_level = _compute_all_ry_angles(
                self._amplitudes, self._num_qubits
            )
            self._rz_angles_per_level = None

    @property
    def num_target_qubits(self) -> int:
        """Number of qubits the gate acts on.

        Returns:
            int: The number of qubits (``log2`` of the amplitude vector
                length).
        """
        return self._num_qubits

    def _decompose(
        self,
        qubits: Vector[Qubit] | tuple[Qubit, ...],
    ) -> tuple[Qubit, ...]:
        """Decompose into RY/CNOT (and RZ/CNOT for complex inputs) gates.

        Emits the magnitude stage (RY layers, Gray-walk order) and then,
        if the input is genuinely complex, the phase stage (RZ layers,
        same structure).  Emitting "all RY layers then all RZ layers"
        is mathematically equivalent (as a full unitary, not just on
        ``|0>^n``) to the Möttönen-Vartiainen per-level interleaved
        order — even though pairwise ``U_y`` / ``U_z`` multiplexers do
        not commute in general (an earlier RY target can be a control
        of a later RZ).  The structural identity is locked down by
        :class:`tests.circuit.algorithm.state_preparation.test_mottonen_amplitude_encoding.TestRyRzOrdering`.

        Args:
            qubits (Vector[Qubit] | tuple[Qubit, ...]):
                ``num_target_qubits`` input qubits as a tuple or
                ``Vector`` handle, expected to start in
                :math:`|0\\rangle^{\\otimes n}`.

        Returns:
            tuple[Qubit, ...]: Output qubits in the encoded state.
        """
        qubit_list = [qubits[i] for i in range(self._num_qubits)]
        ry_angles = [float(a) for a in np.concatenate(self._ry_angles_per_level)]
        _emit_mottonen_gates(qubit_list, self._num_qubits, ry_angles, gate="ry")
        if self._rz_angles_per_level is not None:
            rz_angles = [float(a) for a in np.concatenate(self._rz_angles_per_level)]
            _emit_mottonen_gates(qubit_list, self._num_qubits, rz_angles, gate="rz")
        return tuple(qubit_list)

    def _resources(self) -> ResourceMetadata:
        """Return gate counts for the Gray-walk decomposition.

        Returns:
            ResourceMetadata: Carries the ``RY`` count
                (``2**n - 1``), the ``RZ`` count (``0`` for real inputs,
                ``2**n - 1`` for complex), the ``CNOT`` count (``2**n - 2``
                per emitted stage, so ``2 * (2**n - 2)`` for complex),
                and aggregate totals.
        """
        n = self._num_qubits
        num_ry = 2**n - 1
        cnot_per_stage = 2**n - 2
        if self._is_complex:
            num_rz = 2**n - 1
            num_cnot = 2 * cnot_per_stage
        else:
            num_rz = 0
            num_cnot = cnot_per_stage
        num_rot = num_ry + num_rz
        return ResourceMetadata(
            t_gates=0,
            total_gates=num_rot + num_cnot,
            single_qubit_gates=num_rot,
            two_qubit_gates=num_cnot,
            rotation_gates=num_rot,
            clifford_gates=num_cnot,
            custom_metadata={
                "num_ry_gates": num_ry,
                "num_rz_gates": num_rz,
                "num_cnot_gates": num_cnot,
                "num_qubits": n,
                "complex_input": self._is_complex,
            },
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def compute_mottonen_amplitude_encoding_ry_angles(
    amplitudes: Sequence[float] | Sequence[complex] | np.ndarray,
) -> np.ndarray:
    """Pre-compute the flat Ry angle vector for a Möttönen encoding.

    Returns the Gray-walk Ry angles for the magnitude stage of the
    encoding.  For real inputs (or complex with zero imaginary part)
    this corresponds to the single-stage signed-Ry encoding.  For
    complex inputs with non-zero imaginary part, these are the
    magnitude-stage angles only — pair with
    :func:`compute_mottonen_amplitude_encoding_rz_angles` to obtain the
    phase-stage angles needed to reproduce the full complex state.

    Args:
        amplitudes (Sequence[float] | Sequence[complex] | np.ndarray):
            Amplitude vector of length ``2**n``.  Real or complex; it is
            normalised automatically.

    Returns:
        np.ndarray: 1-D array of length ``2**n - 1`` holding the
            Gray-walk Ry angles laid out level by level.

    Raises:
        ValueError: If the length is not a power of two (or is less
            than 2, i.e., would map to a zero-qubit register), or all
            amplitudes are zero.
    """
    normalized, num_qubits, is_complex = _validate_and_normalize(amplitudes)
    if is_complex:
        ry_angles_per_level, _ = _compute_disentangling_angles(normalized, num_qubits)
    else:
        ry_angles_per_level = _compute_all_ry_angles(normalized, num_qubits)
    return np.concatenate(ry_angles_per_level)


def compute_mottonen_amplitude_encoding_rz_angles(
    amplitudes: Sequence[float] | Sequence[complex] | np.ndarray,
) -> np.ndarray:
    """Pre-compute the flat Rz angle vector for the phase-restoration stage.

    Returns the Gray-walk Rz angles for the phase stage of the Möttönen
    encoding.  For real inputs (or complex with zero imaginary part)
    the phase stage is unnecessary and the returned array is all zeros.
    For complex inputs with non-zero imaginary part, the returned
    angles together with the Ry angles from
    :func:`compute_mottonen_amplitude_encoding_ry_angles` reproduce the
    full complex state.

    Args:
        amplitudes (Sequence[float] | Sequence[complex] | np.ndarray):
            Amplitude vector of length ``2**n``.  Real or complex; it is
            normalised automatically.

    Returns:
        np.ndarray: 1-D array of length ``2**n - 1`` holding the
            Gray-walk Rz angles laid out level by level (all zeros for
            real inputs).

    Raises:
        ValueError: If the length is not a power of two (or is less
            than 2, i.e., would map to a zero-qubit register), or all
            amplitudes are zero.
    """
    normalized, num_qubits, is_complex = _validate_and_normalize(amplitudes)
    if not is_complex:
        return np.zeros(2**num_qubits - 1)
    _, rz_angles_per_level = _compute_disentangling_angles(normalized, num_qubits)
    return np.concatenate(rz_angles_per_level)


def amplitude_encoding(
    qubits: Vector[Qubit],
    amplitudes: Sequence[float] | Sequence[complex] | np.ndarray | Vector[Float],
) -> Vector[Qubit]:
    """Apply Möttönen amplitude encoding to *qubits* in place.

    Convenience wrapper around :class:`MottonenAmplitudeEncoding` that
    accepts a ``Vector`` handle and writes the gated qubits back into
    the same vector.  Real and complex amplitudes are both supported;
    see the class docstring for the gate-count tradeoff between the two
    paths.

    *amplitudes* may be supplied as one of three forms:

    * A concrete Python ``Sequence[float]`` / ``Sequence[complex]`` /
      ``np.ndarray``.  Use this when the amplitudes are known where you
      build the kernel (closed over from the surrounding Python scope).
    * A ``Vector[Float]`` handle obtained from a kernel parameter that
      is **bound at compile time** via
      ``transpiler.transpile(kernel, bindings={"amps": [...]})``.  The
      handle's bound concrete values are extracted at trace time and
      flow through the same angle-computation path.  This makes
      ``bindings={"amps": ...}`` ergonomic without forcing the user to
      pre-compute Möttönen angles.
    * **Not** a ``Vector[Float]`` left symbolic by
      ``parameters=["amps"]`` — the angle computation requires concrete
      values at trace time.  Use :func:`amplitude_encoding_from_angles`
      with ``parameters=["ry_angles", "rz_angles"]`` for the
      runtime-parametric case.

    Args:
        qubits (Vector[Qubit]): Vector of ``n`` qubit handles, expected
            to start in :math:`|0\\rangle^{\\otimes n}`.
        amplitudes (Sequence[float] | Sequence[complex] | np.ndarray | Vector[Float]):
            Amplitude vector of length ``2**n``.  Real or complex; it is
            normalised automatically.  ``Vector[Float]`` is accepted only
            when its concrete values are available at trace time (i.e.,
            it came from a ``bindings={...}`` entry, not from
            ``parameters=[...]``).

    Returns:
        Vector[Qubit]: The same *qubits* vector, with each element
            updated to the post-encoding qubit handle.

    Raises:
        ValueError: If the amplitude length is not a power of two (or
            is less than 2, i.e., would map to a zero-qubit register),
            all amplitudes are zero, the qubit count does not match
            ``log2(len(amplitudes))``, or *amplitudes* is a
            ``Vector[Float]`` handle whose concrete values are not
            available at trace time (use
            :func:`amplitude_encoding_from_angles` with
            ``parameters=[...]`` for runtime-parametric angles).

    Example::

        # Concrete Python amplitudes
        @qmc.qkernel
        def prepare() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, name="q")
            q = amplitude_encoding(q, [1.0, 0.0, 0.0, 1.0])
            return qmc.measure(q)

        # Bound Vector[Float] kernel parameter
        @qmc.qkernel
        def prepare(amps: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, name="q")
            q = amplitude_encoding(q, amps)
            return qmc.measure(q)

        transpiler.transpile(prepare, bindings={"amps": [1.0, 0.0, 0.0, 1.0]})
    """
    n = get_size(qubits)

    if isinstance(amplitudes, Vector):
        const_array = amplitudes.value.get_const_array()
        if const_array is None:
            raise ValueError(
                "amplitude_encoding received a Vector[Float] handle without "
                "concrete values at trace time. Bind it via "
                "transpiler.transpile(kernel, bindings={...}) for compile-time "
                "amplitudes, or use amplitude_encoding_from_angles with "
                "parameters=[...] for runtime-parametric angles."
            )
        concrete_amplitudes: Sequence[float] | Sequence[complex] | np.ndarray = (
            np.asarray(const_array)
        )
    else:
        concrete_amplitudes = amplitudes

    gate = MottonenAmplitudeEncoding(concrete_amplitudes)
    if gate.num_target_qubits != n:
        raise ValueError(
            f"amplitude_encoding requires {gate.num_target_qubits} qubits "
            f"for an amplitude vector of length {len(concrete_amplitudes)}, "
            f"got {n}"
        )

    qubit_list: list[Qubit] = [qubits[i] for i in range(n)]
    result = gate(*qubit_list)
    for i in range(n):
        qubits[i] = result[i]
    return qubits


def amplitude_encoding_from_angles(
    qubits: Vector[Qubit],
    ry_angles: Sequence[float] | np.ndarray | Vector[Float],
    rz_angles: Sequence[float] | np.ndarray | Vector[Float] | None = None,
) -> Vector[Qubit]:
    """Apply Möttönen amplitude encoding from pre-computed Ry / Rz angles.

    Companion to :func:`amplitude_encoding` for the **parametric** use
    case: the user pre-computes the Gray-walk Ry (and optionally Rz)
    angles classically with
    :func:`compute_mottonen_amplitude_encoding_ry_angles` and
    :func:`compute_mottonen_amplitude_encoding_rz_angles`, then passes
    them in as either concrete sequences or as ``Vector[Float]``
    handles obtained from kernel parameters.  In the latter case the
    angles can be left as runtime parameters
    (``transpiler.transpile(kernel, parameters=["ry_angles", ...])``)
    so the same compiled circuit can be re-bound to different
    amplitude vectors without recompilation — useful inside hybrid
    optimisation loops.

    Unlike :func:`amplitude_encoding`, this function does NOT wrap the
    emission in a :class:`MottonenAmplitudeEncoding` ``CompositeGate``
    box on the IR side; the Ry / Rz / CNOT Gray-walk gates are emitted
    directly into the surrounding kernel.  Resource estimation /
    visualization will therefore see the elementary gates rather than
    a single high-level op.

    Args:
        qubits (Vector[Qubit]): Vector of ``n`` qubit handles, expected
            to start in :math:`|0\\rangle^{\\otimes n}`.
        ry_angles (Sequence[float] | np.ndarray | Vector[Float]):
            Gray-walk Ry angles for the magnitude stage.  Must have
            length ``2**n - 1``.
        rz_angles (Sequence[float] | np.ndarray | Vector[Float] | None):
            Gray-walk Rz angles for the phase stage.  Pass ``None``
            (default) to skip the Rz stage entirely (real-amplitude
            path); otherwise must have length ``2**n - 1`` as well.

    Returns:
        Vector[Qubit]: The same *qubits* vector, with each element
            updated to the post-encoding qubit handle.

    Raises:
        ValueError: If ``ry_angles`` / ``rz_angles`` is a concrete
            sequence whose length does not match ``2**n - 1``, or if the
            ``qubits`` vector has an unresolved symbolic shape that
            ``get_size`` cannot reduce to a concrete integer.  When the
            angle argument is a ``Vector[Float]`` handle the length check
            is skipped (the shape may be symbolic at trace time); a
            runtime mismatch then surfaces as a backend bind-time error
            instead.

    Example::

        # Pre-compute classically (outside the kernel)
        ry = compute_mottonen_amplitude_encoding_ry_angles(amps)
        rz = compute_mottonen_amplitude_encoding_rz_angles(amps)

        @qmc.qkernel
        def prepare(
            ry_a: qmc.Vector[qmc.Float],
            rz_a: qmc.Vector[qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, name="q")
            q = amplitude_encoding_from_angles(q, ry_a, rz_a)
            return qmc.measure(q)

        exe = transpiler.transpile(prepare, parameters=["ry_a", "rz_a"])
        # Same compiled circuit, re-bound at "runtime":
        exe.run(transpiler.executor(),
                bindings={"ry_a": ry.tolist(), "rz_a": rz.tolist()})
    """
    n = get_size(qubits)
    expected_len = 2**n - 1

    # Length validation only applies to concrete sequences.  ``Vector[Float]``
    # handles obtained from kernel parameters carry a symbolic shape at
    # trace time, so a static check would either spuriously fail (symbolic
    # shape resolves to 0) or require resolving bindings inside this helper.
    # In the parametric case the user contracts to bind exactly ``2**n - 1``
    # values; a mismatch surfaces at backend bind time.
    if not isinstance(ry_angles, Vector):
        ry_len = len(ry_angles)
        if ry_len != expected_len:
            raise ValueError(
                f"ry_angles must have length 2**n - 1 = {expected_len} for "
                f"n={n} qubits, got {ry_len}"
            )
    if rz_angles is not None and not isinstance(rz_angles, Vector):
        rz_len = len(rz_angles)
        if rz_len != expected_len:
            raise ValueError(
                f"rz_angles must have length 2**n - 1 = {expected_len} for "
                f"n={n} qubits, got {rz_len}"
            )

    qubit_list: list[Qubit] = [qubits[i] for i in range(n)]
    _emit_mottonen_gates(qubit_list, n, ry_angles, gate="ry")
    if rz_angles is not None:
        _emit_mottonen_gates(qubit_list, n, rz_angles, gate="rz")
    for i in range(n):
        qubits[i] = qubit_list[i]
    return qubits
