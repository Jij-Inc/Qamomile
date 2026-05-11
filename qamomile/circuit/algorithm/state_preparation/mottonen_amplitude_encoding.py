"""Real-amplitude state preparation via Möttönen's Y-rotation construction.

Prepares the n-qubit state

.. math::

    |\\psi\\rangle = \\sum_{i=0}^{2^n - 1} a_i \\,|i\\rangle

from :math:`|0\\rangle^{\\otimes n}` for a real, normalised amplitude vector
``a``.  The construction follows Möttönen et al., "Transformation of
quantum states using uniformly controlled rotations"
(arXiv:quant-ph/0407010), restricted to the RY-only "amplitude
distribution" half of the algorithm.  Phase restoration (the second
RZ stage needed for general complex amplitudes) is intentionally out of
scope and complex inputs are rejected at validation time.

Pipeline
--------

1. Validate and normalise the input (length must be a power of two).
2. Recursively compute the per-level rotation angles by splitting each
   chunk into upper / lower halves and using ``arctan2`` of the two
   sub-block norms.  At the leaf level the formula degenerates to
   ``arctan2(a_1, a_0)`` so that **negative real amplitudes flow through
   the sign naturally**.
3. At each level ``k >= 1`` apply a uniformly controlled :math:`R_y`
   over the previously prepared ``k`` qubits.  We use the standard
   Gray-code RY / CNOT decomposition; the per-level angle vector
   ``alpha`` is bit-reversed and then multiplied by the linear transform
   :math:`M^{(k)}` to obtain the angles consumed by the Gray walk.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from qamomile.circuit.frontend.composite_gate import CompositeGate
from qamomile.circuit.frontend.handle import Qubit, Vector
from qamomile.circuit.frontend.handle.utils import get_size
from qamomile.circuit.frontend.operation.qubit_gates import cx, ry
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateType,
    ResourceMetadata,
)

__all__ = [
    "MottonenAmplitudeEncoding",
    "amplitude_encoding",
    "compute_mottonen_amplitude_encoding_thetas",
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
    amplitudes: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, int]:
    """Validate a real amplitude vector and return its normalised form.

    Complex inputs are accepted only when their imaginary part is
    identically zero (within ``np.allclose`` tolerance); any non-zero
    imaginary component is rejected because this routine implements the
    RY-only Möttönen variant.

    Args:
        amplitudes (Sequence[float] | np.ndarray): Real amplitude vector.
            Its length must be a power of two and at least one entry must
            be non-zero.

    Returns:
        tuple[np.ndarray, int]: ``(normalized, num_qubits)`` where
            ``normalized`` is a unit-norm ``np.ndarray`` of dtype
            ``float`` and ``num_qubits`` is ``log2(len(amplitudes))``.

    Raises:
        TypeError: If *amplitudes* carries a non-zero imaginary part.
        ValueError: If the length is not a power of two, or all
            amplitudes are zero.
    """
    arr = np.asarray(amplitudes)

    if np.iscomplexobj(arr):
        if not np.allclose(np.imag(arr), 0.0):
            raise TypeError(
                "MottonenAmplitudeEncoding supports only real amplitudes. "
                "Complex amplitudes require an additional phase-restoration "
                "stage that this routine intentionally omits."
            )
        arr = np.real(arr)

    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Length of amplitudes must be a power of 2, got {n}")

    norm = float(np.linalg.norm(arr))
    if math.isclose(norm, 0.0, abs_tol=1e-15):
        raise ValueError("Amplitudes must not be all zeros")

    return arr / norm, int(round(math.log2(n)))


def _compute_all_thetas(
    amplitudes: np.ndarray,
    num_qubits: int,
) -> list[np.ndarray]:
    """Pre-compute every level's Ry angle vector.

    For each level ``k`` (``0 <= k < num_qubits``) the amplitude vector
    is split into ``2**k`` equal chunks.  Each chunk yields one angle
    ``alpha``:

    * **Intermediate levels** (``chunk_size > 2``): ``alpha`` rotates the
      target qubit so its ``|1>`` weight matches the lower-half block
      norm.  Using ``arctan2(norm_lower, norm_upper)`` keeps the formula
      well defined when the upper half has zero norm.
    * **Leaf level** (``chunk_size == 2``): ``alpha = 2 * arctan2(a_1, a_0)``
      directly, which is signed and therefore preserves negative
      amplitudes.

    For ``k >= 1`` the ``alpha`` vector is then transformed to the
    Gray-walk basis via ``M^(k) @ bit_reverse(alpha)``.  The bit-reverse
    step converts between two indexing conventions: the recursive chunk
    layout uses big-endian control labels (chunk index 0 = controls all
    zero in the most-significant order), while ``_emit_mottonen_gates``
    walks Gray codes in the little-endian convention.

    Args:
        amplitudes (np.ndarray): Unit-norm real amplitude vector of
            length ``2**num_qubits``.
        num_qubits (int): Number of qubits in the target register.

    Returns:
        list[np.ndarray]: A list of ``num_qubits`` arrays; the ``k``-th
            entry has length ``2**k`` and holds the Gray-walk angles for
            that level.
    """
    all_thetas: list[np.ndarray] = []
    for k in range(num_qubits):
        chunk_size = 2 ** (num_qubits - k)
        half_chunk = chunk_size // 2

        alphas = np.empty(2**k)
        for i in range(2**k):
            chunk = amplitudes[i * chunk_size : (i + 1) * chunk_size]
            if half_chunk == 1:
                alphas[i] = 2.0 * math.atan2(float(chunk[1]), float(chunk[0]))
            else:
                norm_upper = float(np.linalg.norm(chunk[:half_chunk]))
                norm_lower = float(np.linalg.norm(chunk[half_chunk:]))
                alphas[i] = 2.0 * math.atan2(norm_lower, norm_upper)

        if k == 0:
            all_thetas.append(alphas)
            continue

        alphas_br = np.empty(2**k)
        for j in range(2**k):
            alphas_br[j] = alphas[_bit_reverse(j, k)]
        all_thetas.append(_compute_angle_transform_matrix(k) @ alphas_br)

    return all_thetas


# ---------------------------------------------------------------------------
# Quantum gate emission
# ---------------------------------------------------------------------------


def _emit_mottonen_gates(
    qubits: list[Qubit],
    num_qubits: int,
    angles: Sequence[float],
) -> None:
    """Emit the Gray-walk RY / CNOT sequence in place on *qubits*.

    Level ``k`` targets qubit ``num_qubits - 1 - k`` so that the first
    rotation lands on the most-significant qubit (consistent with the
    chunk-splitting convention used by :func:`_compute_all_thetas`).

    Args:
        qubits (list[Qubit]): Mutable list of length ``num_qubits`` of
            qubit handles.  Entries are overwritten as gates are applied.
        num_qubits (int): Number of qubits in the register.
        angles (Sequence[float]): Flat sequence of rotation angles of
            length ``2**num_qubits - 1``, laid out level by level.

    Returns:
        None: The *qubits* list is mutated in place.
    """
    idx = 0
    for k in range(num_qubits):
        tgt = num_qubits - 1 - k
        if k == 0:
            qubits[tgt] = ry(qubits[tgt], angles[idx])
            idx += 1
            continue

        cnot_seq = _get_cnot_controls(k)
        for step in range(2**k):
            qubits[tgt] = ry(qubits[tgt], angles[idx])
            idx += 1
            ctrl = num_qubits - 1 - cnot_seq[step]
            qubits[ctrl], qubits[tgt] = cx(qubits[ctrl], qubits[tgt])


# ---------------------------------------------------------------------------
# Composite gate
# ---------------------------------------------------------------------------


class MottonenAmplitudeEncoding(CompositeGate):
    """Möttönen amplitude encoding for normalised real vectors.

    Prepares the state :math:`\\sum_i a_i |i\\rangle` from
    :math:`|0\\rangle^{\\otimes n}` using uniformly controlled Y rotations
    decomposed into ``RY`` and ``CNOT`` gates with Gray-code ordering.

    Notes:
        * Input amplitudes are normalised automatically.
        * Negative real amplitudes are supported.
        * Complex amplitudes are rejected; full complex state preparation
          would require an additional phase-restoration stage.

    Example::

        gate = MottonenAmplitudeEncoding([1.0, 0.0, 0.0, 1.0])
        q0, q1 = gate(q0, q1)
    """

    gate_type = CompositeGateType.CUSTOM
    custom_name = "mottonen_amplitude_encoding"

    def __init__(self, amplitudes: Sequence[float] | np.ndarray):
        """Initialise the gate with a concrete amplitude vector.

        Args:
            amplitudes (Sequence[float] | np.ndarray): Real amplitude
                vector of length ``2**n``.  It is automatically
                normalised; complex inputs are rejected.

        Raises:
            TypeError: If *amplitudes* carries a non-zero imaginary part.
            ValueError: If the length is not a power of two, or all
                amplitudes are zero.
        """
        self._amplitudes, self._num_qubits = _validate_and_normalize(amplitudes)
        self._all_thetas = _compute_all_thetas(self._amplitudes, self._num_qubits)

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
        """Decompose into RY / CNOT gates using Gray-code ordering.

        Args:
            qubits (Vector[Qubit] | tuple[Qubit, ...]):
                ``num_target_qubits`` input qubits as a tuple or
                ``Vector`` handle, expected to start in
                :math:`|0\\rangle^{\\otimes n}`.

        Returns:
            tuple[Qubit, ...]: Output qubits in the encoded state.
        """
        qubit_list = [qubits[i] for i in range(self._num_qubits)]
        flat_angles = [float(a) for a in np.concatenate(self._all_thetas)]
        _emit_mottonen_gates(qubit_list, self._num_qubits, flat_angles)
        return tuple(qubit_list)

    def _resources(self) -> ResourceMetadata:
        """Return gate counts for the Gray-walk decomposition.

        Returns:
            ResourceMetadata: Carries the ``RY`` count (``2**n - 1``),
                the ``CNOT`` count (``2**n - 2``), and aggregate totals.
        """
        n = self._num_qubits
        num_ry = 2**n - 1
        num_cnot = 2**n - 2
        return ResourceMetadata(
            t_gates=0,
            total_gates=num_ry + num_cnot,
            single_qubit_gates=num_ry,
            two_qubit_gates=num_cnot,
            rotation_gates=num_ry,
            clifford_gates=num_cnot,
            custom_metadata={
                "num_ry_gates": num_ry,
                "num_cnot_gates": num_cnot,
                "num_qubits": n,
            },
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def compute_mottonen_amplitude_encoding_thetas(
    amplitudes: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Pre-compute the flat Ry angle vector for a Möttönen encoding.

    Useful when the same encoding is to be applied repeatedly with
    different parameter bindings (e.g. inside a hybrid optimisation
    loop) without re-running the recursive angle computation.

    Args:
        amplitudes (Sequence[float] | np.ndarray): Real amplitude vector
            of length ``2**n``.  It is normalised automatically.

    Returns:
        np.ndarray: 1-D array of length ``2**n - 1`` holding the
            Gray-walk angles laid out level by level.

    Raises:
        TypeError: If *amplitudes* carries a non-zero imaginary part.
        ValueError: If the length is not a power of two, or all
            amplitudes are zero.
    """
    normalized, num_qubits = _validate_and_normalize(amplitudes)
    return np.concatenate(_compute_all_thetas(normalized, num_qubits))


def amplitude_encoding(
    qubits: Vector[Qubit],
    amplitudes: Sequence[float] | np.ndarray,
) -> Vector[Qubit]:
    """Apply Möttönen amplitude encoding to *qubits* in place.

    Convenience wrapper around :class:`MottonenAmplitudeEncoding` that
    accepts a ``Vector`` handle and writes the gated qubits back into
    the same vector.

    Args:
        qubits (Vector[Qubit]): Vector of ``n`` qubit handles, expected
            to start in :math:`|0\\rangle^{\\otimes n}`.
        amplitudes (Sequence[float] | np.ndarray): Real amplitude vector
            of length ``2**n``.  It is normalised automatically.

    Returns:
        Vector[Qubit]: The same *qubits* vector, with each element
            updated to the post-encoding qubit handle.

    Raises:
        TypeError: If *amplitudes* carries a non-zero imaginary part.
        ValueError: If the amplitude length is not a power of two, all
            amplitudes are zero, or the qubit count does not match
            ``log2(len(amplitudes))``.

    Example::

        @qmc.qkernel
        def prepare() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, name="q")
            q = amplitude_encoding(q, [1.0, 0.0, 0.0, 1.0])
            return qmc.measure(q)
    """
    n = get_size(qubits)
    gate = MottonenAmplitudeEncoding(amplitudes)
    if gate.num_target_qubits != n:
        raise ValueError(
            f"amplitude_encoding requires {gate.num_target_qubits} qubits "
            f"for an amplitude vector of length {len(amplitudes)}, got {n}"
        )

    qubit_list: list[Qubit] = [qubits[i] for i in range(n)]
    result = gate(*qubit_list)
    for i in range(n):
        qubits[i] = result[i]
    return qubits
