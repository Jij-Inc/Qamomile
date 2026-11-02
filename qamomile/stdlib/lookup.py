# ruff: noqa: F821 -- widths are closed into dependent source annotations.

"""Coherent immutable lookup tables built from total function selection."""

from __future__ import annotations

from collections.abc import Sequence

import qamomile as qm
from qamomile._native import compiler as native

from ._structure import compose_unary, identity


def xor_lookup(values: Sequence[int], *, value_width: int | None = None) -> qm.QKernel:
    """Construct ``|address>|y> -> |address>|y xor values[address]>``.

    The table length fixes the coherent address width and must be a power of
    two. Each row is an ordinary unary X composition, selected by the existing
    total coherent ``multiplex`` transform. Consequently lookup adds no map/QROM
    instruction, estimator rule, serialization case, or backend branch.

    Args:
        values: Nonnegative integer table in little-endian address order.
        value_width: Positive target width. When omitted it is the minimum
            positive width containing every row.

    Returns:
        A unary unitary over the leading address and trailing target state.

    Raises:
        TypeError: If the table or width has an invalid runtime type.
        ValueError: If the table length, width, or a row is outside its domain.
    """

    if not isinstance(values, Sequence):
        raise TypeError("values must be a sequence of integers")
    frozen: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("lookup values must be integers")
        if value < 0:
            raise ValueError("lookup values must be nonnegative")
        frozen.append(value)
    if value_width is not None and (
        isinstance(value_width, bool) or not isinstance(value_width, int)
    ):
        raise TypeError("value_width must be an integer")
    address_width, width, targets = _native_xor_plan(tuple(frozen), value_width)
    return _xor_lookup_from_targets(
        targets,
        address_width=address_width,
        value_width=width,
    )


def _xor_lookup_from_targets(
    targets: tuple[tuple[int, ...], ...],
    *,
    address_width: int,
    value_width: int,
) -> qm.QKernel:
    """Close one validated native XOR schedule into ordinary functions."""

    total_width = address_width + value_width
    rows = tuple(_xor_targets(targets=row, width=value_width) for row in targets)

    @qm.qkernel
    def lookup(
        state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
        """Apply the configured coherent XOR lookup.

        Args:
            state (qm.Q[qm.Bits[total_width]]): Address and target registers.

        Returns:
            qm.Q[qm.Bits[total_width]]: Retained address and updated target.
        """

        address, target = qm.split_at(state, address_width)
        address, target = qm.multiplex(address, rows, target)
        return qm.concat(address, target)

    return lookup


def sparse_matrix_lookup(
    column_indices: Sequence[Sequence[int]],
    encoded_values: Sequence[Sequence[int]],
    *,
    column_width: int,
    value_width: int | None = None,
) -> qm.QKernel:
    """Construct a coherent row/slot lookup for a sparse matrix.

    The returned total function applies
    ``(|address>, |column>, |value>)`` to
    ``(|address>, |column xor j(row, slot)>, |value xor a(row, slot)>)``.
    The address is the explicit register ``row + row_count * slot``: row bits
    are low-order and slot bits are high-order. Column and value are separate
    owned arguments, so their roles cannot be confused by a packed-state ABI.
    ``column_indices[row][slot]`` and ``encoded_values[row][slot]`` use the
    same ordinary, rectangular Python matrix syntax as Qamomile source. Row
    and slot extents must be powers of two, so every coherent address has a
    defined result. An extent-one axis occupies no address qubit.

    This is intentionally a function factory, not a sparse-matrix descriptor
    or a new IR operation. It can be passed directly to another Qamomile
    function, adjointed, controlled, serialized, and compiled by every target.

    Args:
        column_indices: Rectangular nonnegative column-index matrix.
        encoded_values: Rectangular nonnegative fixed-width value matrix.
        column_width: Positive width of the column output register.
        value_width: Positive width of the encoded value output. When omitted,
            the minimum positive width containing every value is used.

    Returns:
        A three-register reversible transform returning address, column, and
        value with the same ownership partition.

    Raises:
        TypeError: If a matrix, entry, or width has an invalid host type.
        ValueError: If shapes differ, an address extent is not a power of two,
            or an entry does not fit its output width.
    """

    columns = _rectangular_nonnegative_integers(column_indices, label="column_indices")
    values = _rectangular_nonnegative_integers(encoded_values, label="encoded_values")
    checked_column_width = _positive_width(column_width, "column_width")
    if value_width is not None and (
        isinstance(value_width, bool) or not isinstance(value_width, int)
    ):
        raise TypeError("value_width must be an integer")
    address_width, checked_value_width, targets = _native_sparse_plan(
        columns,
        values,
        column_width=checked_column_width,
        value_width=value_width,
    )
    packed_lookup = _xor_lookup_from_targets(
        targets,
        address_width=address_width,
        value_width=checked_column_width + checked_value_width,
    )

    @qm.qkernel
    def lookup(
        address: qm.Q[qm.Bits[address_width]],  # type: ignore[valid-type]
        column: qm.Q[qm.Bits[checked_column_width]],  # type: ignore[valid-type]
        value: qm.Q[qm.Bits[checked_value_width]],  # type: ignore[valid-type]
    ) -> tuple[  # type: ignore[valid-type]
        qm.Q[qm.Bits[address_width]],  # type: ignore[valid-type]
        qm.Q[qm.Bits[checked_column_width]],  # type: ignore[valid-type]
        qm.Q[qm.Bits[checked_value_width]],  # type: ignore[valid-type]
    ]:
        """Apply the sparse oracle while preserving register ownership roles."""

        state = qm.concat(address, column, value)
        state = packed_lookup(state)
        address, output = qm.split_at(state, address_width)
        column, value = qm.split_at(output, checked_column_width)
        return address, column, value

    return lookup


def _rectangular_nonnegative_integers(
    matrix: object,
    *,
    label: str,
) -> tuple[tuple[int, ...], ...]:
    """Freeze and validate one rectangular nonnegative integer matrix."""

    if isinstance(matrix, (str, bytes, bytearray)) or not isinstance(matrix, Sequence):
        raise TypeError(f"{label} must be a sequence of row sequences")
    rows: list[tuple[int, ...]] = []
    for row in matrix:
        if isinstance(row, (str, bytes, bytearray)) or not isinstance(row, Sequence):
            raise TypeError(f"{label} must be a sequence of row sequences")
        frozen_row: list[int] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{label} entries must be integers")
            if value < 0:
                raise ValueError(f"{label} entries must be nonnegative")
            frozen_row.append(value)
        rows.append(tuple(frozen_row))
    if not rows or not rows[0]:
        raise ValueError(f"{label} must not be empty")
    if any(len(row) != len(rows[0]) for row in rows[1:]):
        raise ValueError(f"{label} must be rectangular")
    return tuple(rows)


def _positive_width(value: object, label: str) -> int:
    """Return one validated positive lookup-register width."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _xor_targets(*, targets: tuple[int, ...], width: int) -> qm.QKernel:
    """Construct one native-planned fixed-width bitwise XOR row."""

    if not targets:
        return identity
    return compose_unary(
        width=width,
        steps=tuple(_fixed_x(width=width, target=target) for target in targets),
    )


def _native_xor_plan(
    values: tuple[int, ...],
    value_width: int | None,
) -> tuple[int, int, tuple[tuple[int, ...], ...]]:
    """Cross the required native XOR lookup planning boundary."""

    derive = getattr(native, "plan_xor_lookup", None)
    if not callable(derive):
        raise TypeError("Q7661: native lookup planning is unavailable")
    try:
        projection = derive(
            [str(value) for value in values],
            None if value_width is None else str(value_width),
        )
    except ValueError as error:
        raise ValueError(str(error)) from error
    return _decode_native_plan(projection)


def _native_sparse_plan(
    columns: tuple[tuple[int, ...], ...],
    values: tuple[tuple[int, ...], ...],
    *,
    column_width: int,
    value_width: int | None,
) -> tuple[int, int, tuple[tuple[int, ...], ...]]:
    """Cross the required native sparse lookup planning boundary."""

    derive = getattr(native, "plan_sparse_lookup", None)
    if not callable(derive):
        raise TypeError("Q7661: native lookup planning is unavailable")
    try:
        projection = derive(
            [[str(value) for value in row] for row in columns],
            [[str(value) for value in row] for row in values],
            str(column_width),
            None if value_width is None else str(value_width),
        )
    except ValueError as error:
        raise ValueError(str(error)) from error
    try:
        projected_column_width = projection.column_width
        projected_value_width = projection.value_width
    except AttributeError as error:
        raise TypeError("Q7660: malformed native lookup plan") from error
    if (
        type(projected_column_width) is not int
        or projected_column_width != column_width
        or type(projected_value_width) is not int
        or projected_value_width <= 0
    ):
        raise TypeError("Q7660: malformed native lookup plan")
    return _decode_native_plan(
        projection,
        target_width=projected_column_width + projected_value_width,
    )


def _decode_native_plan(
    projection: object,
    *,
    target_width: int | None = None,
) -> tuple[int, int, tuple[tuple[int, ...], ...]]:
    """Validate one transient native lookup projection."""

    try:
        address_width = getattr(projection, "address_width")
        value_width = getattr(projection, "value_width")
        raw_targets = getattr(projection, "flip_targets")
    except AttributeError as error:
        raise TypeError("Q7660: malformed native lookup plan") from error
    if (
        type(address_width) is not int
        or address_width <= 0
        or type(value_width) is not int
        or value_width <= 0
        or not isinstance(raw_targets, list)
        or len(raw_targets) != 1 << address_width
    ):
        raise TypeError("Q7660: malformed native lookup plan")
    checked_target_width = value_width if target_width is None else target_width
    targets: list[tuple[int, ...]] = []
    for row in raw_targets:
        if (
            not isinstance(row, list)
            or any(type(target) is not int for target in row)
            or row != sorted(set(row))
            or any(target < 0 or target >= checked_target_width for target in row)
        ):
            raise TypeError("Q7660: malformed native lookup plan")
        targets.append(tuple(row))
    return address_width, value_width, tuple(targets)


def _fixed_x(*, width: int, target: int) -> qm.QKernel:
    """Construct one fixed-target X row component."""

    @qm.qkernel
    def flip(
        state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Flip the configured target component.

        Args:
            state (qm.Q[qm.Bits[width]]): Owned register to update.

        Returns:
            qm.Q[qm.Bits[width]]: Register with the configured component flipped.
        """

        return qm.x(state, target=target)

    return flip


__all__ = ["sparse_matrix_lookup", "xor_lookup"]
