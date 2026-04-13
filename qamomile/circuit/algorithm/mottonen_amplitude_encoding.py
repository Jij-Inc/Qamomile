"""Real-amplitude quantum state preparation via the Möttönen Y-rotation decomposition.

Available:
    Classes:
        - MottonenAmplitudeEncoding: Möttönen amplitude encoding for normalized
          real vectors (CompositeGate)

    Functions:
        - amplitude_encoding: Apply Möttönen amplitude encoding to qubits
        - compute_mottonen_amplitude_encoding_thetas: Pre-compute the RY angles
          for the real-amplitude encoding circuit

Reference:
    Möttönen et al., "Transformation of quantum states using uniformly
    controlled rotations", arXiv:quant-ph/0407010

This module implements the RY-based amplitude-distribution part of the
Möttönen construction for normalized real amplitude vectors. It does not
implement the phase-equalization / phase-restoration stages needed for
arbitrary complex state preparation.
"""

from __future__ import annotations

import math
import typing
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from qamomile.circuit.frontend.composite_gate import CompositeGate
from qamomile.circuit.frontend.handle import Float, Qubit, Vector
from qamomile.circuit.frontend.handle.utils import get_size
from qamomile.circuit.frontend.operation.qubit_gates import cx, ry
from qamomile.circuit.frontend.tracer import Tracer, trace
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateType,
    ResourceMetadata,
)

if TYPE_CHECKING:
    from qamomile.circuit.ir.block_value import BlockValue

# ---------------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------------


def _gray_code(n: int) -> int:
    """Return the Gray code of integer n."""
    return n ^ (n >> 1)


def _bit_reverse(value: int, num_bits: int) -> int:
    """Reverse the bit order of *value* within *num_bits* width."""
    return int(format(value, f"0{num_bits}b")[::-1], 2)


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
            lowest_set_bit = i & -i
            bit_position = lowest_set_bit.bit_length() - 1
            controls.append(bit_position)
    return controls


def _compute_angle_transform_matrix(k: int) -> np.ndarray:
    """Compute the 2^k x 2^k angle transformation matrix for the RY network.

    This matrix transforms the per-control-state angles ``alpha`` into the
    Gray-code-ordered elementary angles ``theta`` used by the decomposition
    of a uniformly controlled RY rotation.

    The implementation uses the transposed indexing convention so that the
    emitted Gray-code walk matches the chunk ordering after the explicit
    bit-reversal step applied to ``alpha``.
    """
    size = 2**k
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            bit_count = bin(j & _gray_code(i)).count("1")
            matrix[i, j] = (2**-k) * ((-1) ** bit_count)
    return matrix


# ---------------------------------------------------------------------------
# Validation & angle computation
# ---------------------------------------------------------------------------


def _validate_and_normalize(
    amplitudes: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, int]:
    """Validate a real amplitude vector and return ``(normalized_array, num_qubits)``.

    Raises:
        TypeError: If the input contains a non-zero imaginary component.
        ValueError: If length is not a power of 2, or all amplitudes are zero.
    """
    arr = np.asarray(amplitudes)

    if np.iscomplexobj(arr):
        if not np.allclose(np.imag(arr), 0.0):
            raise TypeError(
                "MottonenAmplitudeEncoding supports only real amplitudes. "
                "Complex amplitudes require a full state-preparation routine "
                "with explicit phase handling."
            )
        arr = np.real(arr)

    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Length of amplitudes must be a power of 2, got {n}")

    norm = float(np.linalg.norm(arr))
    if math.isclose(norm, 0.0, abs_tol=1e-15):
        raise ValueError("Amplitudes must not be all zeros")

    return arr / norm, int(np.log2(n))


def _compute_all_thetas(amplitudes: np.ndarray, num_qubits: int) -> list[np.ndarray]:
    """Pre-compute all RY rotation angles for real-amplitude encoding.

    This routine implements the amplitude-distribution part of the Möttönen
    construction. Intermediate levels are determined from block norms, and the
    leaf level uses signed real amplitudes directly.

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

        # For intermediate levels, the original Möttönen construction can be
        # written using block norms.  We use arctan2 instead of arcsin for the
        # equivalent norm-ratio formula because:
        #   - it avoids an explicit division by the combined norm,
        #   - it remains well-defined for the zero block case,
        #   - and at the leaf level it naturally supports signed real entries.
        #
        # Intermediate levels:
        #   alpha = 2 * arctan2(norm_bottom, norm_top)
        #
        # Leaf level:
        #   alpha = 2 * arctan2(a_1, a_0)
        #
        # This is a real-amplitude specialization. Full complex state
        # preparation additionally requires explicit phase handling.
        alphas = []
        for i in range(2**k):
            chunk = amplitudes[i * chunk_size : (i + 1) * chunk_size]
            if half_chunk == 1:
                alpha = 2 * np.arctan2(float(chunk[1]), float(chunk[0]))
            else:
                norm_top = np.linalg.norm(chunk[:half_chunk])
                norm_bottom = np.linalg.norm(chunk[half_chunk:])
                alpha = 2 * np.arctan2(norm_bottom, norm_top)
            alphas.append(alpha)
        alphas_arr = np.array(alphas)

        if k == 0:
            thetas = alphas_arr
        else:
            # Bit-reverse alpha indices to align the recursive chunk ordering
            # with the Gray-code walk used by the uniformly controlled network.
            alphas_br = np.zeros_like(alphas_arr)
            for j in range(2**k):
                alphas_br[j] = alphas_arr[_bit_reverse(j, k)]
            matrix = _compute_angle_transform_matrix(k)
            thetas = matrix @ alphas_br

        all_thetas.append(thetas)
    return all_thetas


# ---------------------------------------------------------------------------
# Circuit emission
# ---------------------------------------------------------------------------


def _emit_mottonen_gates(
    q: list[Qubit],
    num_qubits: int,
    angles: Sequence[float] | Vector[Float],
) -> None:
    """Emit the RY + CNOT gate sequence for Möttönen decomposition.

    Reads *angles* sequentially by flat index (2^num_qubits - 1 total).
    Mutates *q* in place.

    Args:
        q: Mutable list of qubit handles.
        num_qubits: Number of qubits.
        angles: Flat sequence of rotation angles, length 2^n - 1.
            Can be concrete floats or parametric Float handles.
    """
    idx = 0
    for k in range(num_qubits):
        tgt = num_qubits - 1 - k
        if k == 0:
            q[tgt] = ry(q[tgt], angles[idx])
            idx += 1
        else:
            cnot_seq = _get_cnot_controls(k)
            for i in range(2**k):
                q[tgt] = ry(q[tgt], angles[idx])
                idx += 1
                ctrl = num_qubits - 1 - cnot_seq[i]
                q[ctrl], q[tgt] = cx(q[ctrl], q[tgt])


# ---------------------------------------------------------------------------
# CompositeGate class
# ---------------------------------------------------------------------------


class MottonenAmplitudeEncoding(CompositeGate):
    """Möttönen-style amplitude encoding for normalized real vectors.

    Prepares a real-amplitude quantum state from the |0...0> state using
    uniformly controlled Y-rotations decomposed into RY and CNOT gates via
    Gray code ordering.

    Notes:
        - Input amplitudes are automatically normalized.
        - Negative real amplitudes are supported.
        - Complex amplitudes are not supported.
        - This class implements the RY-only amplitude-distribution part of the
          general Möttönen state-preparation construction.

    Example::

        gate = MottonenAmplitudeEncoding([1.0, 0.0, 0.0, 1.0])
        q0, q1 = gate(q0, q1)
    """

    gate_type = CompositeGateType.CUSTOM
    custom_name = "mottonen_amplitude_encoding"

    _strategies: dict[str, object] = {}
    _default_strategy = "standard"

    def __init__(
        self,
        amplitudes: Sequence[float] | np.ndarray | None = None,
        *,
        thetas: Vector[Float] | None = None,
        num_qubits: int | None = None,
    ):
        if amplitudes is not None:
            # Concrete mode for normalized real amplitudes.
            self._amplitudes, self._num_qubits = _validate_and_normalize(amplitudes)
            self._all_thetas = _compute_all_thetas(self._amplitudes, self._num_qubits)
            self._parametric = False
        elif thetas is not None and num_qubits is not None:
            # Parametric mode for pre-computed real-amplitude encoding angles.
            self._thetas = thetas
            self._num_qubits = num_qubits
            self._parametric = True
        else:
            raise ValueError(
                "Either amplitudes or (thetas, num_qubits) must be provided"
            )

    @property
    def num_target_qubits(self) -> int:
        return self._num_qubits

    def _decompose(
        self,
        qubits: tuple[Qubit, ...],
    ) -> tuple[Qubit, ...]:
        """Decompose into the RY/CNOT network for real-amplitude encoding."""
        if self._parametric:
            raise NotImplementedError(
                "Parametric mode uses _build_decomposition_block directly"
            )
        q = list(qubits)
        flat_angles = [float(a) for a in np.concatenate(self._all_thetas)]
        _emit_mottonen_gates(q, self._num_qubits, flat_angles)
        return tuple(q)

    def _build_decomposition_block(
        self,
        target_qubits: "tuple[Qubit, ...] | Vector[Qubit]",
        strategy_name: str | None = None,
    ) -> "BlockValue | None":
        if not self._parametric:
            return super()._build_decomposition_block(target_qubits, strategy_name)

        from qamomile.circuit.ir.block_value import BlockValue
        from qamomile.circuit.ir.types.primitives import QubitType
        from qamomile.circuit.ir.value import Value

        decomp_tracer = Tracer()
        input_values: list[Value] = []
        fresh_qubits: list[Qubit] = []
        for i in range(self._num_qubits):
            q_value = Value(type=QubitType(), name=f"_decomp_q{i}")
            fresh_qubits.append(Qubit(value=q_value))
            input_values.append(q_value)

        with trace(decomp_tracer):
            q = list(fresh_qubits)
            _emit_mottonen_gates(q, self._num_qubits, self._thetas)

        return_values = [qi.value for qi in q]
        return BlockValue(
            operations=decomp_tracer.operations,
            input_values=input_values,
            return_values=return_values,
            name=self.custom_name,
        )

    def _resources(self) -> ResourceMetadata:
        n = self._num_qubits
        # RY gates: 1 (k=0) + sum_{k=1}^{n-1} 2^k = 2^n - 1
        num_ry = 2**n - 1
        # CNOT gates: sum_{k=1}^{n-1} 2^k = 2^n - 2
        num_cnot = 2**n - 2
        return ResourceMetadata(
            t_gates=0,
            custom_metadata={
                "num_ry_gates": num_ry,
                "num_cnot_gates": num_cnot,
                "total_gates": num_ry + num_cnot,
                "num_qubits": n,
            },
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_mottonen_amplitude_encoding_thetas(
    amplitudes: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Pre-compute flat RY angles for Möttönen amplitude encoding.

    Converts a normalized real vector into the elementary rotation angles
    needed by :func:`amplitude_encoding` in parametric mode.

    Args:
        amplitudes: Real vector of length 2^n. Will be normalized.

    Returns:
        1-D numpy array of length 2^n - 1.
    """
    arr, num_qubits = _validate_and_normalize(amplitudes)
    all_thetas = _compute_all_thetas(arr, num_qubits)
    return np.concatenate(all_thetas)


def compute_mottonen_thetas(amplitudes: Sequence[float] | np.ndarray) -> np.ndarray:
    """Backward-compatible alias for real-amplitude encoding angles.

    Deprecated:
        Use :func:`compute_mottonen_amplitude_encoding_thetas` instead.
    """
    warnings.warn(
        "compute_mottonen_thetas() is deprecated; use "
        "compute_mottonen_amplitude_encoding_thetas() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_mottonen_amplitude_encoding_thetas(amplitudes)


def _concrete_amplitude_encoding(
    qubits: Vector[Qubit],
    amplitudes: Sequence[float] | np.ndarray,
) -> Vector[Qubit]:
    """Apply Möttönen amplitude encoding with concrete real amplitudes."""
    n = get_size(qubits)
    gate = MottonenAmplitudeEncoding(amplitudes)
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


def _parametric_amplitude_encoding(
    qubits: Vector[Qubit],
    thetas: Vector[Float],
) -> Vector[Qubit]:
    """Apply Möttönen amplitude encoding with pre-computed real-valued angles."""
    n = get_size(qubits)
    gate = MottonenAmplitudeEncoding(thetas=thetas, num_qubits=n)
    qubit_list: list[Qubit] = [qubits[i] for i in range(n)]
    result = gate(*qubit_list)
    for i in range(n):
        qubits[i] = result[i]
    return qubits


@typing.overload
def amplitude_encoding(
    qubits: Vector[Qubit],
    data: Sequence[float] | np.ndarray,
) -> Vector[Qubit]: ...


@typing.overload
def amplitude_encoding(
    qubits: Vector[Qubit],
    data: Vector[Float],
) -> Vector[Qubit]: ...


def amplitude_encoding(
    qubits: Vector[Qubit],
    data: Sequence[float] | np.ndarray | Vector[Float],
) -> Vector[Qubit]:
    """Encode a normalized real vector into quantum amplitudes.

    If *data* is a classical real vector of length 2^n, it is treated as
    amplitudes and converted into the internal RY angle sequence.
    If *data* is a ``Vector[Float]``, it is treated as the pre-computed
    real-valued angle sequence for this decomposition.

    Args:
        qubits: Vector of n qubits, all in |0> state.
        data: Either a classical real vector of length 2^n (will be normalized),
            or a ``Vector[Float]`` of pre-computed rotation angles of length
            2^n - 1.

    Returns:
        Qubit vector encoding the target real-amplitude state.

    Notes:
        Complex amplitudes are not supported by this function.
    """
    if isinstance(data, Vector):
        return _parametric_amplitude_encoding(qubits, data)
    return _concrete_amplitude_encoding(qubits, data)
