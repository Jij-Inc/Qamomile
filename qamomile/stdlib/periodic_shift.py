"""Exact block encodings of finite periodic-shift sums."""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from typing import TypeGuard

import qamomile as qm
from qamomile._native import compiler as native

from ._structure import compose_unary, lift_slice
from .arithmetic import add_constant
from .block_encoding import BlockEncoding
from .lcu import lcu_block_encoding

Offset = int | tuple[int, ...]


def periodic_shift_block_encoding(
    coefficients: Mapping[Offset, complex],
    *,
    register_sizes: Sequence[int],
) -> BlockEncoding:
    r"""Construct an exact LCU block encoding of periodic translations.

    Each mapping entry denotes ``coefficient * T_offset``, where an axis of
    width ``w`` is translated modulo ``2**w``. Axis zero occupies the leading
    and therefore least-significant state components. Signed aliases are
    combined before zero terms are removed.

    The factory closes each translation into ordinary constant-addition
    functions and delegates PREPARE/SELECT to :func:`lcu_block_encoding`.
    It introduces no periodic-shift descriptor, QCore operation, resource
    rule, or backend lowering path.

    Args:
        coefficients: Finite coefficients keyed by one integer in one
            dimension or by one integer component per axis.
        register_sizes: Positive qubit width of every periodic axis.

    Returns:
        A construction-independent exact block-encoding contract.

    Raises:
        TypeError: If the mapping, an offset, coefficient, or width has the
            wrong host type.
        ValueError: If no axis is present or a coefficient is nonfinite.
    """

    sizes = _register_sizes(register_sizes)
    terms = _native_periodic_terms(coefficients, sizes)
    return _periodic_block_encoding(terms, sizes)


def _periodic_block_encoding(
    terms: tuple[tuple[tuple[int, ...], complex], ...],
    sizes: tuple[int, ...],
) -> BlockEncoding:
    """Close canonical native terms into ordinary translation functions."""

    width = sum(sizes)
    return lcu_block_encoding(
        tuple(
            (coefficient, _translation(sizes=sizes, offset=offset))
            for offset, coefficient in terms
        ),
        system_width=width,
    )


def periodic_shift_block_encoding_from_matrix(
    matrix: Sequence[Sequence[object]],
    *,
    register_sizes: Sequence[int],
) -> BlockEncoding:
    """Recognize an exact constant-coefficient periodic matrix and encode it.

    Axis zero is the least-significant flattened axis. Every column must be
    exactly the corresponding multidimensional periodic translation of the
    first column. Recognition returns the existing coefficient-map
    construction, so dense input adds no descriptor, approximation state, IR
    operation, or backend implementation.

    Args:
        matrix: Finite square dense matrix as a sequence of row sequences.
        register_sizes: Positive bit width of every periodic axis.

    Returns:
        The exact periodic-shift block encoding.

    Raises:
        TypeError: If the matrix or one entry is not a numeric sequence.
        ValueError: If its dimension, finiteness, or periodic structure is
            invalid.
    """

    sizes = _register_sizes(register_sizes)
    try:
        rows = tuple(tuple(row) for row in matrix)
    except TypeError as error:
        raise TypeError("matrix must be a sequence of row sequences") from error
    column_count = len(rows[0]) if rows else 0
    if any(len(row) != column_count for row in rows):
        raise ValueError("matrix rows must have one common length")
    dense: list[list[tuple[float, float]]] = []
    for row in rows:
        values: list[tuple[float, float]] = []
        for entry in row:
            if isinstance(entry, bool) or not isinstance(entry, numbers.Complex):
                raise TypeError("matrix entries must be numeric scalars")
            value = complex(entry)
            values.append((value.real, value.imag))
        dense.append(values)
    decompose = getattr(native, "decompose_periodic_shift_matrix", None)
    if decompose is None:
        raise TypeError("Q7530: native periodic shift decomposition is unavailable")
    try:
        projection = decompose(
            sizes,
            (len(rows), column_count),
            tuple(value for row in dense for value in row),
        )
    except OverflowError as error:
        raise ValueError("Q7539: periodic shift matrix exceeds its budget") from error
    terms = _decode_native_terms(projection, sizes)
    return _periodic_block_encoding(terms, sizes)


def _register_sizes(register_sizes: Sequence[int]) -> tuple[int, ...]:
    """Validate and freeze one nonempty periodic register layout."""

    if isinstance(register_sizes, (str, bytes)):
        raise TypeError("register_sizes must be a sequence of integers")
    try:
        raw_sizes = tuple(register_sizes)
    except TypeError as error:
        raise TypeError("register_sizes must be a sequence of integers") from error
    if not raw_sizes:
        raise ValueError("periodic shift block encoding requires at least one axis")
    sizes: list[int] = []
    for axis, width in enumerate(raw_sizes):
        if not _is_integer(width):
            raise TypeError(f"register_sizes[{axis}] must be an integer")
        value = int(width)
        if value <= 0:
            raise ValueError(f"register_sizes[{axis}] must be positive")
        sizes.append(value)
    return tuple(sizes)


def _native_periodic_terms(
    coefficients: object,
    sizes: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], complex], ...]:
    """Cross the required native periodic coefficient-normalization boundary."""

    if not isinstance(coefficients, Mapping):
        raise TypeError("coefficients must be a mapping from offsets to scalars")
    offsets: list[tuple[str, ...]] = []
    values: list[tuple[float, float]] = []
    for raw_offset, raw_coefficient in coefficients.items():
        offset = _offset_components(raw_offset, sizes)
        if isinstance(raw_coefficient, bool) or not isinstance(
            raw_coefficient, numbers.Complex
        ):
            raise TypeError(f"coefficient for offset {raw_offset!r} must be complex")
        coefficient = complex(raw_coefficient)
        offsets.append(tuple(format(component, "x") for component in offset))
        values.append((coefficient.real, coefficient.imag))
    normalize = getattr(native, "normalize_periodic_shift_terms", None)
    if normalize is None:
        raise TypeError("Q7530: native periodic shift decomposition is unavailable")
    try:
        projection = normalize(sizes, offsets, values)
    except OverflowError as error:
        raise ValueError("Q7532: periodic shift exceeds its width budget") from error
    return _decode_native_terms(projection, sizes)


def _offset_components(
    offset: object,
    sizes: tuple[int, ...],
) -> tuple[int, ...]:
    """Validate and return arbitrary-precision signed offset components."""

    if len(sizes) == 1 and _is_integer(offset):
        components = (int(offset),)
    elif isinstance(offset, tuple):
        components = offset
    else:
        expected = "an integer or one-element tuple" if len(sizes) == 1 else "a tuple"
        raise TypeError(f"offset must be {expected}")
    if len(components) != len(sizes):
        raise ValueError("offset must contain one component per register axis")
    values: list[int] = []
    for axis, component in enumerate(components):
        if not _is_integer(component):
            raise TypeError(f"offset component {axis} must be an integer")
        values.append(int(component))
    return tuple(values)


def _decode_native_terms(
    projection: object,
    sizes: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], complex], ...]:
    """Validate the transient native term projection."""

    if not isinstance(projection, list):
        raise TypeError("Q7541: malformed native periodic shift decomposition")
    terms: list[tuple[tuple[int, ...], complex]] = []
    previous: tuple[int, ...] | None = None
    for row in projection:
        if (
            not isinstance(row, tuple)
            or len(row) != 3
            or not isinstance(row[0], list)
            or len(row[0]) != len(sizes)
            or not isinstance(row[1], float)
            or not isinstance(row[2], float)
            or not math.isfinite(row[1])
            or not math.isfinite(row[2])
        ):
            raise TypeError("Q7541: malformed native periodic shift decomposition")
        try:
            offset = tuple(int(component, 16) for component in row[0])
        except (TypeError, ValueError) as error:
            raise TypeError(
                "Q7541: malformed native periodic shift decomposition"
            ) from error
        if any(
            component < 0 or component >= 1 << width
            for component, width in zip(offset, sizes, strict=True)
        ) or (previous is not None and offset <= previous):
            raise TypeError("Q7541: malformed native periodic shift decomposition")
        coefficient = complex(row[1], row[2])
        if not coefficient:
            raise TypeError("Q7541: malformed native periodic shift decomposition")
        terms.append((offset, coefficient))
        previous = offset
    return tuple(terms)


def _is_integer(value: object) -> TypeGuard[numbers.Integral]:
    """Return whether a host value is an integer but not a Boolean."""

    return not isinstance(value, bool) and isinstance(value, numbers.Integral)


def _translation(
    *,
    sizes: tuple[int, ...],
    offset: tuple[int, ...],
) -> qm.QKernel:
    """Compose one multidimensional periodic translation from additions."""

    total_width = sum(sizes)
    start = 0
    steps: list[qm.QKernel] = []
    for width, value in zip(sizes, offset, strict=True):
        if value:
            steps.append(
                lift_slice(
                    total_width=total_width,
                    start=start,
                    width=width,
                    transform=add_constant(width=width, value=value),
                )
            )
        start += width
    return compose_unary(width=total_width, steps=steps)


__all__ = [
    "periodic_shift_block_encoding",
    "periodic_shift_block_encoding_from_matrix",
]
