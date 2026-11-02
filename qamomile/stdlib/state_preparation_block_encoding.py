# ruff: noqa: F821 -- dependent widths are closed into generated source.

"""Block encode a bounded SciPy sparse array through paired state preparation."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import qamomile as qm

from .block_encoding import BlockEncoding
from .sparse_access_block_encoding import _trigonometric_interval
from .state_preparation import (
    _controlled_rotation_schedule,
    amplitude_angles,
    amplitude_encoding,
)

_MAX_DIMENSION = 16


def state_preparation_block_encoding(
    matrix: object,
    *,
    max_error: float = 1e-8,
) -> BlockEncoding:
    r"""Encode a SciPy sparse array with projected block ``A / alpha``.

    The normalization is ``alpha = s * max(abs(A[i, j]))``, where ``s``
    is the larger maximum number of nonzero entries in any row or column.
    Duplicate coordinates are summed before this calculation. The zero matrix
    uses ``alpha = 1``. A fixed-point interval replay of every preparation
    gate bounds the absolute spectral-norm reconstruction error of the full
    projected block. Device noise and backend gate synthesis are excluded.

    Classical sparse-array processing happens at factory time. The returned
    descriptor owns an ordinary Qamomile unitary and is compiled only when
    bound to an algorithm or compiled directly. This construction uses
    ``O(N**2)`` preparation data for an ``N x N`` input, so the dimension
    is bounded.

    Args:
        matrix (scipy.sparse.sparray): Finite square sparse array with
            power-of-two dimension from 2 through 16. Integer entries must
            fit the exact binary64 integer range.
        max_error (float): Nonnegative finite absolute spectral-norm budget
            for the reconstructed matrix.

    Returns:
        BlockEncoding: First-class unitary with signal width
            ``log2(N) + 2`` and system width ``log2(N)``.

    Raises:
        ImportError: If SciPy is not installed.
        TypeError: If input is not a SciPy sparse array or the error budget is
            not a real scalar.
        ValueError: If shape, size, values, normalization, or budget is invalid.

    Example:
        >>> from scipy.sparse import csr_array
        >>> from qamomile.stdlib import state_preparation_block_encoding
        >>> block = state_preparation_block_encoding(
        ...     csr_array([[0.5, 0], [0, 0.25]]), max_error=1e-8
        ... )
        >>> block.normalization
        0.5
    """

    from scipy.sparse import coo_array, csr_array, sparray

    if not isinstance(matrix, sparray):
        raise TypeError("matrix must be a SciPy sparse array")
    if isinstance(max_error, bool) or not isinstance(max_error, (int, float)):
        raise TypeError("max_error must be a real scalar")
    try:
        checked_error = float(max_error)
    except OverflowError as error:
        raise ValueError("max_error must be nonnegative and finite") from error
    if not math.isfinite(checked_error) or checked_error < 0:
        raise ValueError("max_error must be nonnegative and finite")
    coordinates = coo_array(matrix, copy=True)
    shape = coordinates.shape
    if shape is None or len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("matrix must be square")
    dimension = shape[0]
    if dimension < 2 or dimension & (dimension - 1):
        raise ValueError("matrix dimension must be a power of two of at least two")
    if dimension > _MAX_DIMENSION:
        raise ValueError(f"matrix dimension exceeds the {_MAX_DIMENSION}-row budget")

    kind = coordinates.dtype.kind
    if (
        kind not in "iufc"
        or (kind == "f" and coordinates.dtype.itemsize > 8)
        or (kind == "c" and coordinates.dtype.itemsize > 16)
    ):
        raise TypeError("matrix entries must be real or complex numeric scalars")
    if kind in "iu" and any(abs(int(value)) > (1 << 53) for value in coordinates.data):
        raise ValueError("integer matrix entries exceed exact binary64 range")
    array = csr_array(coordinates, dtype=complex, copy=True)
    array.sum_duplicates()
    array.eliminate_zeros()
    array.sort_indices()
    values = [complex(value) for value in array.data]
    if any(
        not math.isfinite(value.real) or not math.isfinite(value.imag)
        for value in values
    ):
        raise ValueError("matrix entries must be finite")
    max_abs = max((abs(value) for value in values), default=0.0)
    row_counts = [
        int(array.indptr[index + 1] - array.indptr[index]) for index in range(dimension)
    ]
    column_counts = [0] * dimension
    for column in array.indices:
        column_counts[int(column)] += 1
    sparsity = max((*row_counts, *column_counts))
    alpha = float(sparsity) * max_abs if sparsity else 1.0
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("matrix normalization is not finite and positive")

    width = dimension.bit_length() - 1
    prepared_width = width + 1
    total_width = 2 * width + 2
    scale = alpha if sparsity else 1.0
    columns: list[list[complex]] = [[0j] * (2 * dimension) for _ in range(dimension)]
    rows: list[list[complex]] = [[0j] * (2 * dimension) for _ in range(dimension)]
    row_masses = [0.0] * dimension
    column_masses = [0.0] * dimension
    for row in range(dimension):
        for offset in range(array.indptr[row], array.indptr[row + 1]):
            column = int(array.indices[offset])
            value = complex(array.data[offset])
            magnitude = abs(value)
            amplitude = math.sqrt(magnitude / scale)
            rows[row][column] = amplitude
            columns[column][row] = amplitude * value / magnitude
            row_masses[row] += magnitude / scale
            column_masses[column] += magnitude / scale
    for index in range(dimension):
        rows[index][dimension] = math.sqrt(max(0.0, 1.0 - row_masses[index]))
        columns[index][dimension] = math.sqrt(max(0.0, 1.0 - column_masses[index]))
    error_bound = _state_preparation_error_bound(rows, columns, array, alpha)
    if error_bound > checked_error:
        raise ValueError(
            f"max_error={max_error} is below the certified error bound {error_bound}"
        )
    left_cases = tuple(amplitude_encoding(tuple(row)) for row in rows)
    left_inverse_cases = tuple(
        _adjoint_preparation(case, prepared_width) for case in left_cases
    )
    right_cases = tuple(amplitude_encoding(tuple(column)) for column in columns)

    @qm.qkernel
    def prepare_right(
        state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
        """Prepare column amplitudes and their orthogonal failure state.

        Args:
            state (qm.Q[qm.Bits[total_width]]): Joint signal and system state.

        Returns:
            qm.Q[qm.Bits[total_width]]: Column-prepared joint state.
        """

        index, remaining = qm.split_at(state, width)
        left_flag, remaining = qm.split_at(remaining, 1)
        right_flag, system = qm.split_at(remaining, 1)
        system, target = qm.multiplex(system, right_cases, qm.concat(index, right_flag))
        index, right_flag = qm.split_at(target, width)
        return qm.concat(index, left_flag, right_flag, system)

    @qm.qkernel
    def unprepare_left(
        state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
        """Unprepare row amplitudes and their orthogonal failure state.

        Args:
            state (qm.Q[qm.Bits[total_width]]): Joint signal and system state.

        Returns:
            qm.Q[qm.Bits[total_width]]: Row-unprepared joint state.
        """

        index, remaining = qm.split_at(state, width)
        left_flag, remaining = qm.split_at(remaining, 1)
        right_flag, system = qm.split_at(remaining, 1)
        system, target = qm.multiplex(
            system, left_inverse_cases, qm.concat(index, left_flag)
        )
        index, left_flag = qm.split_at(target, width)
        return qm.concat(index, left_flag, right_flag, system)

    @qm.qkernel
    def encoded(
        state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
        """Apply the sparse-array block encoding.

        Args:
            state (qm.Q[qm.Bits[total_width]]): Joint signal and system state.

        Returns:
            qm.Q[qm.Bits[total_width]]: Encoded joint state.
        """

        state = prepare_right(state)
        for bit in range(width):
            state = qm.swap(state, left=bit, right=width + 2 + bit)
        return unprepare_left(state)

    return BlockEncoding(
        encoded, alpha, prepared_width + 1, width, error_bound=error_bound
    )


def _adjoint_preparation(unitary: qm.QKernel, width: int) -> qm.QKernel:
    """Create one direct inverse case for a row preparation.

    Args:
        unitary (qm.QKernel): Row state-preparation unitary.
        width (int): Width of its coherent state.

    Returns:
        qm.QKernel: Inverse unary preparation.
    """

    @qm.qkernel
    def inverse(
        state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Apply one adjointed preparation.

        Args:
            state (qm.Q[qm.Bits[width]]): Owned row-preparation state.

        Returns:
            qm.Q[qm.Bits[width]]: Inverse-prepared state.
        """

        return qm.adjoint(unitary, state)

    return inverse


__all__ = ["state_preparation_block_encoding"]

_FIXED_SCALE = 1 << 256
_RealInterval = tuple[int, int]
_ComplexInterval = tuple[_RealInterval, _RealInterval]


def _fixed_from_fraction(value: Fraction) -> _RealInterval:
    """Enclose one exact rational at 256-bit fixed precision.

    Args:
        value (Fraction): Exact rational input.

    Returns:
        tuple[int, int]: Outward-rounded scaled integer interval.
    """

    numerator = value.numerator * _FIXED_SCALE
    denominator = value.denominator
    return numerator // denominator, -(-numerator // denominator)


def _fixed_add(left: _RealInterval, right: _RealInterval) -> _RealInterval:
    """Add two fixed-point intervals.

    Args:
        left (tuple[int, int]): First interval.
        right (tuple[int, int]): Second interval.

    Returns:
        tuple[int, int]: Sum interval.
    """

    return left[0] + right[0], left[1] + right[1]


def _fixed_neg(value: _RealInterval) -> _RealInterval:
    """Negate one fixed-point interval.

    Args:
        value (tuple[int, int]): Interval to negate.

    Returns:
        tuple[int, int]: Negated interval.
    """

    return -value[1], -value[0]


def _fixed_mul(left: _RealInterval, right: _RealInterval) -> _RealInterval:
    """Multiply two fixed-point intervals with outward rounding.

    Args:
        left (tuple[int, int]): First interval.
        right (tuple[int, int]): Second interval.

    Returns:
        tuple[int, int]: Product interval.
    """

    products = tuple(a * b for a in left for b in right)
    return min(products) // _FIXED_SCALE, -(-max(products) // _FIXED_SCALE)


def _complex_add(left: _ComplexInterval, right: _ComplexInterval) -> _ComplexInterval:
    """Add two rectangular complex intervals.

    Args:
        left (_ComplexInterval): First interval.
        right (_ComplexInterval): Second interval.

    Returns:
        _ComplexInterval: Componentwise sum.
    """

    return _fixed_add(left[0], right[0]), _fixed_add(left[1], right[1])


def _complex_neg(value: _ComplexInterval) -> _ComplexInterval:
    """Negate a rectangular complex interval.

    Args:
        value (_ComplexInterval): Interval to negate.

    Returns:
        _ComplexInterval: Componentwise negative.
    """

    return _fixed_neg(value[0]), _fixed_neg(value[1])


def _complex_mul(left: _ComplexInterval, right: _ComplexInterval) -> _ComplexInterval:
    """Multiply two rectangular complex intervals.

    Args:
        left (_ComplexInterval): First factor.
        right (_ComplexInterval): Second factor.

    Returns:
        _ComplexInterval: Outward-rounded product.
    """

    ac = _fixed_mul(left[0], right[0])
    bd = _fixed_mul(left[1], right[1])
    ad = _fixed_mul(left[0], right[1])
    bc = _fixed_mul(left[1], right[0])
    return _fixed_add(ac, _fixed_neg(bd)), _fixed_add(ad, bc)


def _fixed_trig(angle: Fraction, *, sine: bool) -> _RealInterval:
    """Enclose one trigonometric value of an exact stored gate angle.

    Args:
        angle (Fraction): Exact rational gate angle.
        sine (bool): Select sine instead of cosine.

    Returns:
        tuple[int, int]: Fixed-point enclosure.
    """

    lower, upper = _trigonometric_interval(angle, sine=sine)
    return _fixed_from_fraction(lower)[0], _fixed_from_fraction(upper)[1]


def _phase_interval(angle: Fraction) -> _ComplexInterval:
    """Enclose the complex phase induced by a stored angle.

    Args:
        angle (Fraction): Exact rational phase angle.

    Returns:
        _ComplexInterval: Enclosure of the phase factor.
    """

    return _fixed_trig(angle, sine=False), _fixed_trig(angle, sine=True)


def _prepared_intervals(amplitudes: list[complex]) -> list[_ComplexInterval]:
    """Replay the exact preparation gate schedule with interval arithmetic.

    Args:
        amplitudes (list[complex]): Source vector passed to amplitude_encoding.

    Returns:
        list[_ComplexInterval]: Enclosed prepared amplitudes in basis order.
    """

    angles = amplitude_angles(tuple(amplitudes))
    width = angles.width
    zero = ((0, 0), (0, 0))
    state = [zero for _ in amplitudes]
    state[0] = ((_FIXED_SCALE, _FIXED_SCALE), (0, 0))

    def rotate_y(angle: float, target: int) -> None:
        """Apply one enclosed Y rotation to the local interval state.

        Args:
            angle (float): Stored rotation angle.
            target (int): Target bit index.
        """

        half_angle = Fraction.from_float(angle) / 2
        cosine = _fixed_trig(half_angle, sine=False)
        sine = _fixed_trig(half_angle, sine=True)
        factor_c = (cosine, (0, 0))
        factor_s = (sine, (0, 0))
        for low in range(len(state)):
            if low & (1 << target):
                continue
            high = low | (1 << target)
            first, second = state[low], state[high]
            state[low] = _complex_add(
                _complex_mul(factor_c, first),
                _complex_neg(_complex_mul(factor_s, second)),
            )
            state[high] = _complex_add(
                _complex_mul(factor_s, first), _complex_mul(factor_c, second)
            )

    def rotate_z(angle: float, target: int) -> None:
        """Apply one enclosed Z rotation to the local interval state.

        Args:
            angle (float): Stored rotation angle.
            target (int): Target bit index.
        """

        half_angle = Fraction.from_float(angle) / 2
        negative = _phase_interval(-half_angle)
        positive = _phase_interval(half_angle)
        for basis in range(len(state)):
            state[basis] = _complex_mul(
                positive if basis & (1 << target) else negative, state[basis]
            )

    def controlled_x(control: int, target: int) -> None:
        """Permute one controlled-X pair in the local interval state.

        Args:
            control (int): Control bit index.
            target (int): Target bit index.
        """

        for basis in range(len(state)):
            if basis & (1 << control) and not basis & (1 << target):
                other = basis | (1 << target)
                state[basis], state[other] = state[other], state[basis]

    rotate_y(angles.ry[0], width - 1)
    for angle_index, target, control in _controlled_rotation_schedule(width):
        rotate_y(angles.ry[angle_index], target)
        controlled_x(control, target)
    if any(angles.rz):
        rotate_z(angles.rz[0], width - 1)
        for angle_index, target, control in _controlled_rotation_schedule(width):
            rotate_z(angles.rz[angle_index], target)
            controlled_x(control, target)
    if angles.phase:
        factor = _phase_interval(Fraction.from_float(angles.phase))
        state = [_complex_mul(factor, amplitude) for amplitude in state]
    return state


def _state_preparation_error_bound(
    rows: list[list[complex]],
    columns: list[list[complex]],
    matrix: Any,
    normalization: float,
) -> float:
    """Bound the reconstructed matrix from actual stored preparation angles.

    Args:
        rows (list[list[complex]]): Row preparation vectors.
        columns (list[list[complex]]): Column preparation vectors.
        matrix (object): Canonical SciPy CSR array.
        normalization (float): Stored positive scale.

    Returns:
        float: Upward-rounded absolute spectral-norm error bound.
    """

    dimension = len(rows)
    left = [_prepared_intervals(row) for row in rows]
    right = [_prepared_intervals(column) for column in columns]
    scale = _fixed_from_fraction(Fraction.from_float(normalization))
    largest = 0
    for row in range(dimension):
        for column in range(dimension):
            left_amplitude = left[row][column]
            conjugated = (left_amplitude[0], _fixed_neg(left_amplitude[1]))
            product = _complex_mul(conjugated, right[column][row])
            real = _fixed_mul(scale, product[0])
            imag = _fixed_mul(scale, product[1])
            target = complex(matrix[row, column])
            target_real = _fixed_from_fraction(Fraction.from_float(target.real))
            target_imag = _fixed_from_fraction(Fraction.from_float(target.imag))
            real_error = max(
                abs(real[0] - target_real[1]), abs(real[1] - target_real[0])
            )
            imag_error = max(
                abs(imag[0] - target_imag[1]), abs(imag[1] - target_imag[0])
            )
            largest = max(largest, real_error + imag_error)
    exact_bound = Fraction(dimension * largest, _FIXED_SCALE)
    rounded = float(exact_bound)
    if Fraction.from_float(rounded) < exact_bound:
        rounded = math.nextafter(rounded, math.inf)
    return rounded
