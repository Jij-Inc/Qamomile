# ruff: noqa: F821 -- dependent widths are closed into generated source.

"""Sparse-access block encoding from bounded SciPy sparse-array data."""

from __future__ import annotations

import math
from fractions import Fraction

import qamomile as qm

from .block_encoding import BlockEncoding
from .lookup import sparse_matrix_lookup

_MAX_TABLE_ROWS = 4096
_MAX_VALUE_CASES = 4096


def sparse_access_block_encoding(
    matrix: object, *, max_error: float = 1e-8
) -> BlockEncoding:
    """Build a sparse-access block encoding from a SciPy sparse array.

    Entries are assigned shared row/column edge colors. Reversible column and
    row lookups then select each entry exactly once. The projected block is
    A / (slots * magnitude_scale), including for asymmetric complex matrices.
    Input data is frozen when this factory runs; compilation remains deferred.

    Args:
        matrix (scipy.sparse.sparray): Square finite sparse array with a
            power-of-two dimension of at least two.
        max_error (float): Nonnegative absolute spectral-norm error budget
            for the matrix reconstructed from the projected block.

    Returns:
        BlockEncoding: Sparse-access unitary and its normalization.

    Raises:
        ImportError: If SciPy is unavailable.
        TypeError: If the input type or error budget is invalid.
        ValueError: If shape, values, work budget, or error budget is invalid.
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
    source = coo_array(matrix, copy=True)
    shape = source.shape
    if shape is None or len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("matrix must be square")
    dimension = shape[0]
    if dimension < 2 or dimension & (dimension - 1):
        raise ValueError("matrix dimension must be a power of two of at least two")
    if dimension * 2 > _MAX_TABLE_ROWS:
        raise ValueError("sparse lookup table exceeds the 4096-row work budget")
    kind = source.dtype.kind
    if (
        kind not in "iufc"
        or (kind == "f" and source.dtype.itemsize > 8)
        or (kind == "c" and source.dtype.itemsize > 16)
    ):
        raise TypeError("matrix entries must be real or complex numeric scalars")
    if kind in "iu" and any(abs(int(value)) > (1 << 53) for value in source.data):
        raise ValueError("integer matrix entries exceed exact binary64 range")
    array = csr_array(source, dtype=complex, copy=True)
    array.sum_duplicates()
    array.eliminate_zeros()
    array.sort_indices()
    entries = [
        (row, int(array.indices[offset]), complex(array.data[offset]))
        for row in range(dimension)
        for offset in range(array.indptr[row], array.indptr[row + 1])
    ]
    if any(
        not math.isfinite(value.real) or not math.isfinite(value.imag)
        for _, _, value in entries
    ):
        raise ValueError("matrix entries must be finite")
    row_counts = [0] * dimension
    column_counts = [0] * dimension
    for row, column, _ in entries:
        row_counts[row] += 1
        column_counts[column] += 1
    sparsity = max((*row_counts, *column_counts))
    # An edge meets at most 2s-2 previously colored edges at its endpoints.
    minimum_colors = max(1, 2 * sparsity - 1)
    slots = max(2, 1 << (minimum_colors - 1).bit_length())
    if dimension * slots > _MAX_TABLE_ROWS:
        raise ValueError("sparse lookup table exceeds the 4096-row work budget")
    unique_values = tuple(dict.fromkeys(value for _, _, value in entries))
    value_width = max(1, len(unique_values).bit_length())
    if 1 << value_width > _MAX_VALUE_CASES:
        raise ValueError("value rotation table exceeds the 4096-case work budget")
    magnitude_scale = max((abs(value) for value in unique_values), default=0.0)
    normalization = slots * magnitude_scale if entries else 1.0
    if not math.isfinite(normalization) or normalization <= 0:
        raise ValueError("matrix normalization is not finite and positive")
    error_bound = _certified_matrix_error(unique_values, magnitude_scale, sparsity)
    if error_bound > checked_error:
        raise ValueError(
            f"max_error={max_error} is below the certified error bound {error_bound}"
        )
    value_ids = {value: index + 1 for index, value in enumerate(unique_values)}
    row_used: list[set[int]] = [set() for _ in range(dimension)]
    column_used: list[set[int]] = [set() for _ in range(dimension)]
    from_columns = [[0] * slots for _ in range(dimension)]
    from_values = [[0] * slots for _ in range(dimension)]
    to_columns = [[0] * slots for _ in range(dimension)]
    zeros = [[0] * slots for _ in range(dimension)]
    for row, column, value in entries:
        color = next(
            slot
            for slot in range(slots)
            if slot not in row_used[row] and slot not in column_used[column]
        )
        row_used[row].add(color)
        column_used[column].add(color)
        from_columns[column][color] = row
        from_values[column][color] = value_ids[value]
        to_columns[row][color] = column
    index_width = dimension.bit_length() - 1
    slot_width = slots.bit_length() - 1
    column_lookup = sparse_matrix_lookup(
        from_columns, from_values, column_width=index_width, value_width=value_width
    )
    row_lookup = sparse_matrix_lookup(
        to_columns, zeros, column_width=index_width, value_width=value_width
    )
    copy_index = _copy_index(index_width)
    absent_rotation = _value_rotation(None)
    rotations = [absent_rotation]
    rotations.extend(
        _value_rotation(value / magnitude_scale) for value in unique_values
    )
    rotations.extend([absent_rotation] * ((1 << value_width) - len(rotations)))
    rotation_cases = tuple(rotations)
    signal_width = slot_width + 2 * index_width + value_width + 1
    total_width = signal_width + index_width

    @qm.qkernel
    def encoded(
        state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
        """Apply paired lookup, value rotation, and reversible cleanup.

        Args:
            state (qm.Q[qm.Bits[total_width]]): Signal and system state.

        Returns:
            qm.Q[qm.Bits[total_width]]: Encoded state.
        """

        for bit in range(slot_width):
            state = qm.h(state, target=bit)
        slot, rest = qm.split_at(state, slot_width)
        scratch, rest = qm.split_at(rest, index_width)
        index, rest = qm.split_at(rest, index_width)
        value, rest = qm.split_at(rest, value_width)
        flag, system = qm.split_at(rest, 1)
        address, scratch, value = column_lookup(qm.concat(system, slot), scratch, value)
        system, slot = qm.split_at(address, index_width)
        value, flag = qm.multiplex(value, rotation_cases, flag)
        index, scratch = qm.split_at(copy_index(qm.concat(index, scratch)), index_width)
        address, scratch, value = column_lookup(qm.concat(system, slot), scratch, value)
        system, slot = qm.split_at(address, index_width)
        state = qm.concat(slot, scratch, index, value, flag, system)
        for bit in range(index_width):
            state = qm.swap(
                state, left=slot_width + index_width + bit, right=signal_width + bit
            )
        slot, rest = qm.split_at(state, slot_width)
        scratch, rest = qm.split_at(rest, index_width)
        index, rest = qm.split_at(rest, index_width)
        value, rest = qm.split_at(rest, value_width)
        flag, system = qm.split_at(rest, 1)
        address, index, value = row_lookup(qm.concat(system, slot), index, value)
        system, slot = qm.split_at(address, index_width)
        state = qm.concat(slot, scratch, index, value, flag, system)
        for bit in range(slot_width):
            state = qm.h(state, target=bit)
        return state

    return BlockEncoding(
        encoded, normalization, signal_width, index_width, error_bound=error_bound
    )


def _certified_matrix_error(
    values: tuple[complex, ...], magnitude_scale: float, sparsity: int
) -> float:
    """Bound the full spectral-norm reconstruction error from rounded angles.

    Each row and column has at most sparsity entries, so their maximum entry
    error bounds the spectral norm after multiplication by sparsity. Taylor
    intervals enclose the exact trigonometric values of the stored binary64
    gate angles; this does not include later device noise or backend synthesis.

    Args:
        values (tuple[complex, ...]): Distinct canonical matrix entries.
        magnitude_scale (float): Maximum entry magnitude.
        sparsity (int): Maximum row or column nonzero count.

    Returns:
        float: Upward-rounded absolute spectral-norm error bound.
    """

    if not values:
        return 0.0
    scale = Fraction.from_float(magnitude_scale)
    worst = Fraction(0)
    for value in values:
        normalized = value / magnitude_scale
        angle = 2.0 * math.acos(min(1.0, abs(normalized)))
        phase = math.atan2(normalized.imag, normalized.real)
        cosine = _trigonometric_interval(Fraction.from_float(angle) / 2, sine=False)
        phase_cosine = _trigonometric_interval(Fraction.from_float(phase), sine=False)
        phase_sine = _trigonometric_interval(Fraction.from_float(phase), sine=True)
        real = _interval_product(cosine, phase_cosine)
        imag = _interval_product(cosine, phase_sine)
        target_real = Fraction.from_float(value.real)
        target_imag = Fraction.from_float(value.imag)
        real_error = max(abs(scale * endpoint - target_real) for endpoint in real)
        imag_error = max(abs(scale * endpoint - target_imag) for endpoint in imag)
        worst = max(worst, real_error + imag_error)
    exact_bound = sparsity * worst
    rounded = float(exact_bound)
    if Fraction.from_float(rounded) < exact_bound:
        rounded = math.nextafter(rounded, math.inf)
    return rounded


def _trigonometric_interval(
    argument: Fraction, *, sine: bool
) -> tuple[Fraction, Fraction]:
    """Enclose sine or cosine of a binary-rational gate angle.

    Args:
        argument (Fraction): Exact stored angle, with magnitude at most pi.
        sine (bool): Select sine instead of cosine.

    Returns:
        tuple[Fraction, Fraction]: Rational lower and upper bounds.
    """

    if abs(argument) > 4:
        return Fraction(-1), Fraction(1)
    power = argument if sine else Fraction(1)
    factorial = 1
    total = Fraction(0)
    first_degree = 1 if sine else 0
    for index in range(24):
        term = power / factorial
        total += -term if index & 1 else term
        power *= argument * argument
        factorial *= (first_degree + 2 * index + 1) * (first_degree + 2 * index + 2)
    tail = abs(power / factorial)
    return total - tail, total + tail


def _interval_product(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    """Multiply two closed rational intervals.

    Args:
        left (tuple[Fraction, Fraction]): First interval.
        right (tuple[Fraction, Fraction]): Second interval.

    Returns:
        tuple[Fraction, Fraction]: Enclosure of every product.
    """

    products = tuple(a * b for a in left for b in right)
    return min(products), max(products)


def _copy_index(width: int) -> qm.QKernel:
    """Return a reversible computational-basis register copy.

    Args:
        width (int): Number of bits in each index.

    Returns:
        qm.QKernel: XOR copy from trailing to leading register.
    """

    @qm.qkernel
    def copied(
        state: qm.Q[qm.Bits[2 * width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[2 * width]]:  # type: ignore[valid-type]
        """XOR each trailing source bit into the corresponding target.

        Args:
            state (qm.Q[qm.Bits[2 * width]]): Target and source.

        Returns:
            qm.Q[qm.Bits[2 * width]]: Updated target and retained source.
        """

        for bit in range(width):
            state = qm.cx(state, control=width + bit, target=bit)
        return state

    return copied


def _value_rotation(value: complex | None) -> qm.QKernel:
    """Construct the one-qubit amplitude and complex-phase rotation.

    Args:
        value (complex | None): Normalized entry, or absent slot.

    Returns:
        qm.QKernel: One-qubit unitary with the entry as its zero amplitude.
    """

    if value is None:

        @qm.qkernel
        def absent(
            flag: qm.Q[qm.Bits[1]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[1]]:  # type: ignore[valid-type]
            """Mark an absent slot as orthogonal to the signal projection.

            Args:
                flag (qm.Q[qm.Bit]): Success flag.

            Returns:
                qm.Q[qm.Bit]: Failed flag.
            """

            return qm.x(flag)

        return absent

    angle = 2.0 * math.acos(min(1.0, abs(value)))
    phase = math.atan2(value.imag, value.real)

    @qm.qkernel
    def present(
        flag: qm.Q[qm.Bits[1]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[1]]:  # type: ignore[valid-type]
        """Rotate the success amplitude to the selected complex value.

        Args:
            flag (qm.Q[qm.Bit]): Success flag.

        Returns:
            qm.Q[qm.Bit]: Rotated flag.
        """

        flag = qm.ry(flag, angle)
        flag = qm.p(flag, -phase)
        return qm.global_phase(flag, phase)

    return present


__all__ = ["sparse_access_block_encoding"]
