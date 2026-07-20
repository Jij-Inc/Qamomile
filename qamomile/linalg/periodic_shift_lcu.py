"""Represent periodic-shift matrices as immutable LCU decompositions."""

from __future__ import annotations

import dataclasses
import math
import numbers
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from qamomile.linalg._validation import (
    as_finite_square_complex_matrix,
    coerce_finite_complex_scalar,
    coerce_nonnegative_finite_real,
)

Offset = int | tuple[int, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class PeriodicShiftLCUTerm:
    """Store one nonzero coefficient and one canonical periodic offset.

    Args:
        coefficient (complex): Finite nonzero coefficient multiplying the
            periodic shift.
        offset (tuple[int, ...]): Non-negative offset residue for every axis.
            The enclosing :class:`PeriodicShiftLCU` validates each component
            against its corresponding axis size.

    Raises:
        TypeError: If the coefficient is not numeric or the offset is not an
            iterable of non-Boolean integers.
        ValueError: If the coefficient is zero or non-finite, or an offset
            component is negative.
    """

    coefficient: complex
    offset: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate and freeze the canonical term.

        Raises:
            TypeError: If the coefficient or offset has an invalid type.
            ValueError: If the coefficient or offset is outside its valid
                domain.
        """
        coefficient = coerce_finite_complex_scalar(
            self.coefficient,
            "term coefficient",
        )
        if not coefficient:
            raise ValueError("PeriodicShiftLCUTerm coefficient must be nonzero.")
        try:
            raw_offset = tuple(self.offset)
        except TypeError as error:
            raise TypeError(
                "PeriodicShiftLCUTerm offset must be an iterable."
            ) from error
        offset: list[int] = []
        for axis, component in enumerate(raw_offset):
            if not _is_integer(component):
                raise TypeError(
                    f"PeriodicShiftLCUTerm offset component {axis} must be an integer."
                )
            value = int(component)
            if value < 0:
                raise ValueError(
                    "PeriodicShiftLCUTerm offset components must be non-negative."
                )
            offset.append(value)
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "offset", tuple(offset))


@dataclasses.dataclass(frozen=True, slots=True)
class PeriodicShiftLCU:
    """Store an immutable periodic-shift linear-combination decomposition.

    Axis zero occupies the least-significant bits of the flattened system
    basis. Each term represents ``coefficient * T_offset``, where every axis
    of ``T_offset`` wraps modulo ``2**register_sizes[axis]``.

    Args:
        register_sizes (tuple[int, ...]): Positive qubit width of every
            periodic axis. The empty tuple represents a scalar ``1 x 1``
            matrix at the linalg layer.
        terms (tuple[PeriodicShiftLCUTerm, ...]): Canonical nonzero terms.
            Terms are stored in lexicographic offset order. The empty tuple
            represents the zero operator.
        truncation_error_bound (float): Upper bound on the spectral-norm error
            caused only by coefficients intentionally removed by
            :meth:`from_matrix` or :meth:`from_coefficients`. Defaults to
            ``0.0`` and excludes host floating-point roundoff.

    Raises:
        TypeError: If a field has an invalid runtime type.
        ValueError: If an axis width, term, normalization, or truncation bound
            is invalid, or terms contain duplicate offsets.
    """

    register_sizes: tuple[int, ...]
    terms: tuple[PeriodicShiftLCUTerm, ...]
    truncation_error_bound: float = 0.0

    def __post_init__(self) -> None:
        """Validate and canonicalize the immutable decomposition.

        Raises:
            TypeError: If a field has an invalid runtime type.
            ValueError: If metadata is inconsistent, non-finite, or outside
                its valid range.
        """
        register_sizes = _validate_register_sizes(self.register_sizes)
        try:
            terms = tuple(self.terms)
        except TypeError as error:
            raise TypeError("PeriodicShiftLCU terms must be an iterable.") from error

        validated: list[PeriodicShiftLCUTerm] = []
        seen: set[tuple[int, ...]] = set()
        for term in terms:
            if not isinstance(term, PeriodicShiftLCUTerm):
                raise TypeError(
                    "PeriodicShiftLCU terms must all be PeriodicShiftLCUTerm values."
                )
            if len(term.offset) != len(register_sizes):
                raise ValueError(
                    "PeriodicShiftLCU term offsets must contain one component "
                    "per register axis."
                )
            for axis, (component, width) in enumerate(
                zip(term.offset, register_sizes, strict=True)
            ):
                if component >= 1 << width:
                    raise ValueError(
                        "PeriodicShiftLCU term offset component "
                        f"{axis} is not a canonical modular residue."
                    )
            if term.offset in seen:
                raise ValueError(
                    "PeriodicShiftLCU cannot contain duplicate canonical offsets."
                )
            seen.add(term.offset)
            validated.append(term)

        ordered_terms = tuple(sorted(validated, key=lambda term: term.offset))
        try:
            alpha = math.fsum(abs(term.coefficient) for term in ordered_terms)
        except OverflowError as error:
            raise ValueError(
                "PeriodicShiftLCU normalization exceeds the finite float range."
            ) from error
        if not math.isfinite(alpha):
            raise ValueError("PeriodicShiftLCU normalization must be finite.")

        error_bound = coerce_nonnegative_finite_real(
            self.truncation_error_bound,
            "truncation_error_bound",
        )
        object.__setattr__(self, "register_sizes", register_sizes)
        object.__setattr__(self, "terms", ordered_terms)
        object.__setattr__(self, "truncation_error_bound", error_bound)

    @classmethod
    def from_coefficients(
        cls,
        coefficients: Mapping[Offset, complex],
        *,
        register_sizes: Sequence[int],
        atol: float = 0.0,
    ) -> PeriodicShiftLCU:
        """Build a canonical decomposition from periodic-shift coefficients.

        Equivalent signed or unsigned offsets are combined before pruning.
        Coefficients whose magnitude is at most ``atol`` are omitted, and the
        sum of their magnitudes is stored in ``truncation_error_bound``.

        Args:
            coefficients (Mapping[int | tuple[int, ...], complex]): Finite
                coefficients keyed by a one-dimensional integer offset or one
                integer component per axis.
            register_sizes (Sequence[int]): Positive qubit width of each
                periodic axis. The empty sequence describes a scalar
                decomposition; any supplied offset must then be the empty
                tuple.
            atol (float): Non-negative absolute coefficient threshold.
                Defaults to ``0.0``, which removes exact zeros only.

        Returns:
            PeriodicShiftLCU: Immutable canonical decomposition.

        Raises:
            TypeError: If coefficients, offsets, widths, or ``atol`` have
                invalid types.
            ValueError: If a value is non-finite, an offset has the wrong
                dimension, a width is invalid, or normalization overflows.
        """
        if not isinstance(coefficients, Mapping):
            raise TypeError("coefficients must be a mapping from offsets to values.")
        sizes = _validate_register_sizes(register_sizes)
        threshold = coerce_nonnegative_finite_real(atol, "atol")
        grouped: dict[tuple[int, ...], list[complex]] = {}
        for offset, coefficient in coefficients.items():
            canonical = _canonical_offset(offset, sizes)
            value = coerce_finite_complex_scalar(
                coefficient,
                f"coefficient for offset {offset!r}",
            )
            grouped.setdefault(canonical, []).append(value)

        terms: list[PeriodicShiftLCUTerm] = []
        dropped_magnitudes: list[float] = []
        for offset, values in sorted(grouped.items()):
            coefficient = _sum_complex_values(values, offset)
            magnitude = abs(coefficient)
            if not magnitude:
                continue
            if magnitude <= threshold:
                dropped_magnitudes.append(magnitude)
                continue
            terms.append(PeriodicShiftLCUTerm(coefficient, offset))

        try:
            error_bound = math.fsum(dropped_magnitudes)
        except OverflowError as error:
            raise ValueError(
                "PeriodicShiftLCU truncation error bound exceeds the finite "
                "float range."
            ) from error
        return cls(
            register_sizes=sizes,
            terms=tuple(terms),
            truncation_error_bound=error_bound,
        )

    @classmethod
    def from_matrix(
        cls,
        matrix: ArrayLike,
        *,
        register_sizes: Sequence[int],
        atol: float = 0.0,
    ) -> PeriodicShiftLCU:
        """Decompose an exact constant-coefficient periodic-shift matrix.

        After conversion to ``complex128``, the structural check is exact:
        every column must be the periodic translation of the first column
        implied by ``register_sizes``. The ``atol`` argument has the same
        meaning as in :meth:`from_coefficients`; it prunes extracted
        coefficients but does not relax the structural check. Hermiticity is
        not required.

        Args:
            matrix (ArrayLike): Finite square matrix with dimension
                ``2**sum(register_sizes)``.
            register_sizes (Sequence[int]): Positive qubit width of each axis
                in flattened least-significant-axis-first order. The empty
                sequence describes a scalar ``1 x 1`` matrix.
            atol (float): Non-negative absolute coefficient threshold.
                Defaults to ``0.0``, which removes exact zeros only.

        Returns:
            PeriodicShiftLCU: Immutable canonical shift decomposition.

        Raises:
            TypeError: If widths or ``atol`` have invalid types.
            ValueError: If the matrix is nonnumeric, non-finite, nonsquare,
                has the wrong dimension, violates the periodic-shift
                structure, or produces an unrepresentable normalization.
        """
        sizes = _validate_register_sizes(register_sizes)
        threshold = coerce_nonnegative_finite_real(atol, "atol")
        dense = as_finite_square_complex_matrix(matrix)
        expected_dimension = 1 << sum(sizes)
        if dense.shape[0] != expected_dimension:
            raise ValueError(
                "matrix dimension must equal 2**sum(register_sizes); "
                f"got {dense.shape[0]} and register_sizes={sizes}."
            )

        if expected_dimension == 1:
            coefficients: dict[Offset, complex] = {(): complex(dense[0, 0])}
            return cls.from_coefficients(
                coefficients,
                register_sizes=sizes,
                atol=threshold,
            )

        dimensions = tuple(1 << width for width in sizes)
        indices = np.arange(expected_dimension, dtype=np.int64)
        coordinate_arrays = np.unravel_index(indices, dimensions, order="F")
        coordinates = np.stack(coordinate_arrays, axis=1)
        dimension_array = np.asarray(dimensions, dtype=np.int64)
        strides = np.cumprod(
            np.asarray((1, *dimensions[:-1]), dtype=np.int64),
        )
        first_column = dense[:, 0]
        for column in range(expected_dimension):
            shifted = (coordinates + coordinates[column]) % dimension_array
            rows = shifted @ strides
            aligned_column = dense[rows, column]
            mismatches = np.flatnonzero(aligned_column != first_column)
            if mismatches.size:
                offset_index = int(mismatches[0])
                row = int(rows[offset_index])
                expected = complex(first_column[offset_index])
                actual = complex(dense[row, column])
                raise ValueError(
                    "matrix is not a constant-coefficient periodic-shift "
                    f"matrix for register_sizes={sizes}: entry ({row}, {column}) "
                    f"is {actual!r}, expected {expected!r}."
                )

        coefficients = {
            tuple(int(component) for component in coordinates[index]): complex(value)
            for index, value in enumerate(first_column)
        }
        return cls.from_coefficients(
            coefficients,
            register_sizes=sizes,
            atol=threshold,
        )

    @property
    def alpha(self) -> float:
        """Return the retained LCU one-norm normalization.

        Returns:
            float: Stable sum of retained coefficient magnitudes. The zero
                operator has normalization ``0.0``.
        """
        return math.fsum(abs(term.coefficient) for term in self.terms)

    @property
    def num_terms(self) -> int:
        """Return the number of retained periodic-shift terms.

        Returns:
            int: Number of nonzero retained terms.
        """
        return len(self.terms)

    @property
    def num_qubits(self) -> int:
        """Return the flattened system-register width.

        Returns:
            int: Sum of all periodic-axis qubit widths.
        """
        return sum(self.register_sizes)


def _validate_register_sizes(register_sizes: Sequence[int]) -> tuple[int, ...]:
    """Validate and freeze periodic-axis qubit widths.

    Args:
        register_sizes (Sequence[int]): Candidate per-axis qubit widths.

    Returns:
        tuple[int, ...]: Plain positive Python integer widths. The empty tuple
            is retained for a scalar system.

    Raises:
        TypeError: If the input is not iterable or a width is not an integer.
        ValueError: If a supplied width is non-positive.
    """
    try:
        raw_sizes = tuple(register_sizes)
    except TypeError as error:
        raise TypeError("register_sizes must be an iterable of integers.") from error
    sizes: list[int] = []
    for axis, width in enumerate(raw_sizes):
        if not _is_integer(width):
            raise TypeError(f"register_sizes[{axis}] must be an integer.")
        value = int(width)
        if value <= 0:
            raise ValueError(f"register_sizes[{axis}] must be positive, got {value}.")
        sizes.append(value)
    return tuple(sizes)


def _canonical_offset(
    offset: Any,
    register_sizes: tuple[int, ...],
) -> tuple[int, ...]:
    """Convert one user offset to canonical modular residues.

    Args:
        offset (Any): One integer in one dimension or one component per axis.
        register_sizes (tuple[int, ...]): Validated per-axis qubit widths.

    Returns:
        tuple[int, ...]: Canonical non-negative offset residues.

    Raises:
        TypeError: If the offset form or a component is invalid.
        ValueError: If the offset dimension disagrees with the axes.
    """
    dimensions = len(register_sizes)
    if dimensions == 1 and _is_integer(offset):
        components = (int(offset),)
    elif isinstance(offset, tuple):
        if len(offset) != dimensions:
            raise ValueError(
                f"offset {offset!r} has {len(offset)} dimensions; expected {dimensions}."
            )
        components = offset
    else:
        expected = (
            "an integer or one-element tuple"
            if dimensions == 1
            else f"a {dimensions}-element tuple"
        )
        raise TypeError(f"offset must be {expected}, got {offset!r}.")

    residues: list[int] = []
    for axis, (component, width) in enumerate(
        zip(components, register_sizes, strict=True)
    ):
        if not _is_integer(component):
            raise TypeError(f"offset component {axis} must be an integer.")
        residues.append(int(component) % (1 << width))
    return tuple(residues)


def _sum_complex_values(
    values: Sequence[complex],
    offset: tuple[int, ...],
) -> complex:
    """Combine aliased coefficients in deterministic numeric order.

    Args:
        values (Sequence[complex]): Finite coefficients for one canonical
            offset.
        offset (tuple[int, ...]): Canonical offset used in diagnostics.

    Returns:
        complex: Accurately summed coefficient with canonical signed zeros.

    Raises:
        ValueError: If finite inputs overflow while being combined.
    """
    ordered = sorted(values, key=lambda value: (value.real, value.imag))
    try:
        real = math.fsum(value.real for value in ordered)
        imag = math.fsum(value.imag for value in ordered)
    except OverflowError as error:
        raise ValueError(
            f"combined coefficient for offset {offset!r} must be finite."
        ) from error
    if not math.isfinite(real) or not math.isfinite(imag):
        raise ValueError(f"combined coefficient for offset {offset!r} must be finite.")
    return complex(real if real else 0.0, imag if imag else 0.0)


def _is_integer(value: object) -> bool:
    """Return whether a value is an integer scalar but not a Boolean.

    Args:
        value (object): Runtime value to inspect.

    Returns:
        bool: Whether the value is an accepted integer scalar.
    """
    return not _is_boolean(value) and isinstance(value, (numbers.Integral, np.integer))


def _is_boolean(value: object) -> bool:
    """Return whether a value is a Python or NumPy Boolean scalar.

    Args:
        value (object): Runtime value to inspect.

    Returns:
        bool: Whether the value is Boolean-like.
    """
    return isinstance(value, (bool, np.bool_))


__all__ = ["PeriodicShiftLCU", "PeriodicShiftLCUTerm"]
