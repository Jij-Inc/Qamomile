"""Tests for immutable periodic-shift LCU decompositions."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest

import qamomile.linalg as qml
from qamomile.linalg import PeriodicShiftLCU, PeriodicShiftLCUTerm


def _matrix(
    coefficients: dict[tuple[int, ...], complex],
    register_sizes: tuple[int, ...],
) -> np.ndarray:
    """Return a dense periodic-shift matrix in axis-zero-LSB order.

    Args:
        coefficients (dict[tuple[int, ...], complex]): Canonical shift
            coefficients.
        register_sizes (tuple[int, ...]): Per-axis qubit widths.

    Returns:
        np.ndarray: Dense matrix represented by the supplied shifts.
    """
    dimensions = tuple(1 << width for width in register_sizes)
    dimension = math.prod(dimensions)
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for column in range(dimension):
        coordinates: list[int] = []
        remaining = column
        for axis_dimension in dimensions:
            coordinates.append(remaining % axis_dimension)
            remaining //= axis_dimension
        for offset, coefficient in coefficients.items():
            shifted = tuple(
                (coordinate + component) % axis_dimension
                for coordinate, component, axis_dimension in zip(
                    coordinates,
                    offset,
                    dimensions,
                    strict=True,
                )
            )
            row = 0
            stride = 1
            for coordinate, axis_dimension in zip(
                shifted,
                dimensions,
                strict=True,
            ):
                row += coordinate * stride
                stride *= axis_dimension
            matrix[row, column] += coefficient
    return matrix


def _reconstruct(lcu: PeriodicShiftLCU) -> np.ndarray:
    """Reconstruct the retained dense matrix from one decomposition.

    Args:
        lcu (PeriodicShiftLCU): Canonical decomposition to expand.

    Returns:
        np.ndarray: Retained dense periodic-shift matrix.
    """
    coefficients = {term.offset: term.coefficient for term in lcu.terms}
    if not lcu.register_sizes:
        coefficient = next(iter(coefficients.values()), 0.0j)
        return np.asarray([[coefficient]], dtype=np.complex128)
    return _matrix(coefficients, lcu.register_sizes)


def test_from_coefficients_combines_aliases_in_canonical_order() -> None:
    """Signed aliases combine deterministically before term ordering."""
    items = [
        (1, 1e16),
        (-3, -1e16),
        (5, 1.0),
        (0, -2.0),
    ]
    forward = PeriodicShiftLCU.from_coefficients(
        dict(items),
        register_sizes=(2,),
    )
    reversed_order = PeriodicShiftLCU.from_coefficients(
        dict(reversed(items)),
        register_sizes=(2,),
    )

    expected = (
        PeriodicShiftLCUTerm(-2.0, (0,)),
        PeriodicShiftLCUTerm(1.0, (1,)),
    )
    assert forward.terms == reversed_order.terms == expected
    assert forward.alpha == pytest.approx(3.0, rel=1e-15, abs=0.0)
    assert forward.num_terms == 2
    assert forward.num_qubits == 2


def test_from_coefficients_canonicalizes_signed_zero_components() -> None:
    """Equivalent signed-zero inputs produce exactly equal value objects."""
    positive = PeriodicShiftLCU.from_coefficients(
        {0: complex(-1.0, 0.0)},
        register_sizes=(1,),
    )
    negative = PeriodicShiftLCU.from_coefficients(
        {0: complex(-1.0, -0.0)},
        register_sizes=(1,),
    )

    assert positive == negative
    assert hash(positive) == hash(negative)
    assert math.copysign(1.0, negative.terms[0].coefficient.imag) == 1.0


def test_direct_value_is_frozen_sorted_and_deeply_immutable() -> None:
    """Direct construction freezes widths, terms, and canonical ordering."""
    lcu = PeriodicShiftLCU(
        register_sizes=(np.int64(2), np.int32(1)),
        terms=(
            PeriodicShiftLCUTerm(0.5j, (1, 0)),
            PeriodicShiftLCUTerm(-1.0, (0, 0)),
        ),
    )

    assert lcu.register_sizes == (2, 1)
    assert all(type(width) is int for width in lcu.register_sizes)
    assert tuple(term.offset for term in lcu.terms) == ((0, 0), (1, 0))
    assert not hasattr(lcu, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        lcu.truncation_error_bound = 1.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        lcu.terms[0].offset[0] = 1  # type: ignore[index]


@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 1, 2, 42])
def test_from_matrix_recovers_random_complex_one_dimensional_terms(
    width: int,
    seed: int,
) -> None:
    """Exact complex circulant matrices round-trip through the first column."""
    rng = np.random.default_rng(seed)
    dimension = 1 << width
    coefficients = {
        (offset,): complex(*rng.uniform(-1.0, 1.0, size=2))
        for offset in range(dimension)
    }
    matrix = _matrix(coefficients, (width,))

    lcu = PeriodicShiftLCU.from_matrix(matrix, register_sizes=(width,))

    np.testing.assert_allclose(_reconstruct(lcu), matrix, atol=0.0, rtol=0.0)
    assert {term.offset: term.coefficient for term in lcu.terms} == coefficients


@pytest.mark.parametrize("register_sizes", [(1, 2), (2, 1), (1, 1, 1)])
@pytest.mark.parametrize("seed", [0, 42])
def test_from_matrix_recovers_multidimensional_axis_zero_lsb_terms(
    register_sizes: tuple[int, ...],
    seed: int,
) -> None:
    """Multidimensional extraction follows the flattened axis-zero-LSB basis."""
    rng = np.random.default_rng(seed)
    dimensions = tuple(1 << width for width in register_sizes)
    coefficients: dict[tuple[int, ...], complex] = {}
    for flat_offset in range(math.prod(dimensions)):
        remaining = flat_offset
        offset: list[int] = []
        for dimension in dimensions:
            offset.append(remaining % dimension)
            remaining //= dimension
        coefficients[tuple(offset)] = complex(*rng.uniform(-1.0, 1.0, size=2))
    matrix = _matrix(coefficients, register_sizes)

    lcu = PeriodicShiftLCU.from_matrix(matrix, register_sizes=register_sizes)

    assert lcu.register_sizes == register_sizes
    assert {term.offset: term.coefficient for term in lcu.terms} == coefficients
    np.testing.assert_allclose(_reconstruct(lcu), matrix, atol=0.0, rtol=0.0)


def test_from_matrix_accepts_non_hermitian_periodic_shift() -> None:
    """Periodic structure does not imply or require Hermiticity."""
    matrix = _matrix({(0,): 0.25j, (1,): 2.0 - 0.5j}, (2,))

    lcu = PeriodicShiftLCU.from_matrix(matrix, register_sizes=(2,))

    np.testing.assert_allclose(_reconstruct(lcu), matrix, atol=0.0, rtol=0.0)
    assert not np.allclose(matrix, matrix.conj().T)


def test_matrix_grid_interpretation_changes_canonical_offset_coordinates() -> None:
    """One matrix can have different offsets under different axis layouts."""
    matrix = _matrix({(2,): 1.0}, (2,))

    one_axis = PeriodicShiftLCU.from_matrix(matrix, register_sizes=(2,))
    two_axes = PeriodicShiftLCU.from_matrix(matrix, register_sizes=(1, 1))

    assert one_axis.terms == (PeriodicShiftLCUTerm(1.0, (2,)),)
    assert two_axes.terms == (PeriodicShiftLCUTerm(1.0, (0, 1)),)
    np.testing.assert_array_equal(_reconstruct(one_axis), _reconstruct(two_axes))


def test_periodic_shift_lcu_types_are_publicly_exported() -> None:
    """The decomposition and term types are available from qamomile.linalg."""
    assert qml.PeriodicShiftLCU is PeriodicShiftLCU
    assert qml.PeriodicShiftLCUTerm is PeriodicShiftLCUTerm


def test_from_matrix_rejects_position_dependent_matrix() -> None:
    """A diagonal whose value changes by position fails the exact structure."""
    matrix = np.diag([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match=r"entry \(1, 1\).+expected"):
        PeriodicShiftLCU.from_matrix(matrix, register_sizes=(2,))


def test_from_matrix_atol_does_not_relax_periodic_structure() -> None:
    """The Pauli-compatible threshold prunes terms but never hides noise."""
    matrix = _matrix({(0,): 1.0, (1,): 0.5}, (2,))
    matrix[2, 1] += 1e-12

    with pytest.raises(ValueError, match="not a constant-coefficient"):
        PeriodicShiftLCU.from_matrix(matrix, register_sizes=(2,), atol=1.0)


def test_atol_prunes_coefficients_and_bounds_spectral_error() -> None:
    """Dropped unitary-shift weights provide the documented norm bound."""
    tiny = 1e-8
    matrix = _matrix({(0,): 1.0, (1,): tiny, (2,): -0.5 * tiny}, (2,))

    lcu = PeriodicShiftLCU.from_matrix(
        matrix,
        register_sizes=(2,),
        atol=1e-7,
    )
    error = matrix - _reconstruct(lcu)

    assert lcu.terms == (PeriodicShiftLCUTerm(1.0, (0,)),)
    assert lcu.truncation_error_bound == pytest.approx(1.5 * tiny)
    assert np.linalg.norm(error, ord=2) <= lcu.truncation_error_bound + 1e-15


def test_threshold_boundary_is_pruned() -> None:
    """A coefficient exactly at the absolute threshold is omitted."""
    coefficient = complex(0.345584192064786, -0.32776493753426794)
    threshold = abs(coefficient)

    lcu = PeriodicShiftLCU.from_coefficients(
        {0: coefficient},
        register_sizes=(1,),
        atol=threshold,
    )

    assert lcu.terms == ()
    assert lcu.truncation_error_bound == pytest.approx(
        threshold,
        rel=0.0,
        abs=0.0,
    )


def test_tiny_nonzero_coefficient_is_exact_by_default() -> None:
    """The default threshold retains every representable nonzero term."""
    tiny = np.nextafter(0.0, 1.0)

    lcu = PeriodicShiftLCU.from_coefficients(
        {0: tiny, 1: 1.0},
        register_sizes=(1,),
    )

    assert lcu.terms == (
        PeriodicShiftLCUTerm(tiny, (0,)),
        PeriodicShiftLCUTerm(1.0, (1,)),
    )
    assert lcu.truncation_error_bound == 0.0


def test_zero_cancelled_and_all_pruned_inputs_are_valid() -> None:
    """Every mathematical zero path retains widths and approximation data."""
    empty = PeriodicShiftLCU.from_coefficients({}, register_sizes=(2,))
    cancelled = PeriodicShiftLCU.from_coefficients(
        {1: 1.0, -3: -1.0},
        register_sizes=(2,),
    )
    pruned = PeriodicShiftLCU.from_coefficients(
        {0: 1e-8},
        register_sizes=(2,),
        atol=1e-7,
    )
    matrix_zero = PeriodicShiftLCU.from_matrix(
        np.zeros((4, 4)),
        register_sizes=(2,),
    )

    assert empty.terms == cancelled.terms == matrix_zero.terms == ()
    assert empty.alpha == cancelled.alpha == matrix_zero.alpha == 0.0
    assert empty.truncation_error_bound == cancelled.truncation_error_bound == 0.0
    assert pruned.terms == ()
    assert pruned.truncation_error_bound == pytest.approx(1e-8)


def test_scalar_matrix_matches_pauli_linalg_domain() -> None:
    """A one-by-one matrix remains a valid zero-qubit linalg value."""
    lcu = PeriodicShiftLCU.from_matrix(
        np.asarray([[2.0 - 3.0j]]),
        register_sizes=(),
    )

    assert lcu.num_qubits == 0
    assert lcu.terms == (PeriodicShiftLCUTerm(2.0 - 3.0j, ()),)
    np.testing.assert_allclose(_reconstruct(lcu), [[2.0 - 3.0j]])


@pytest.mark.parametrize(
    ("matrix", "register_sizes", "message"),
    [
        (np.array([1.0, 2.0]), (1,), "2D square"),
        (np.zeros((2, 4)), (1,), "2D square"),
        (np.eye(3), (2,), "dimension"),
        (np.eye(4), (1,), "dimension"),
        (np.array([[np.nan, 0.0], [0.0, 1.0]]), (1,), "finite"),
        (np.array([[np.inf, 0.0], [0.0, 1.0]]), (1,), "finite"),
    ],
)
def test_from_matrix_rejects_invalid_dense_inputs(
    matrix: np.ndarray,
    register_sizes: tuple[int, ...],
    message: str,
) -> None:
    """Invalid matrix shape, dimension, and entries fail clearly."""
    with pytest.raises(ValueError, match=message):
        PeriodicShiftLCU.from_matrix(matrix, register_sizes=register_sizes)


@pytest.mark.parametrize(
    ("coefficients", "register_sizes", "exception", "message"),
    [
        ({0: 1.0}, (), TypeError, "0-element tuple"),
        ({0: 1.0}, (0,), ValueError, "must be positive"),
        ({(0, 1): 1.0}, (2,), ValueError, "dimensions"),
        ({0: "1.0"}, (1,), TypeError, "numeric scalar"),
        ({0: complex(np.inf, 0.0)}, (1,), ValueError, "finite"),
    ],
)
def test_from_coefficients_rejects_invalid_inputs(
    coefficients: dict[Any, Any],
    register_sizes: tuple[int, ...],
    exception: type[Exception],
    message: str,
) -> None:
    """Invalid widths, offsets, and coefficients fail before decomposition."""
    with pytest.raises(exception, match=message):
        PeriodicShiftLCU.from_coefficients(
            coefficients,
            register_sizes=register_sizes,
        )


@pytest.mark.parametrize("atol", [True, -1.0, np.inf, 1j])
def test_invalid_threshold_is_rejected(atol: Any) -> None:
    """The pruning threshold accepts only finite non-negative reals."""
    exception = TypeError if isinstance(atol, (bool, complex)) else ValueError
    with pytest.raises(exception):
        PeriodicShiftLCU.from_coefficients(
            {0: 1.0},
            register_sizes=(1,),
            atol=atol,
        )


def test_direct_terms_reject_noncanonical_or_duplicate_offsets() -> None:
    """Direct values require unique in-range canonical offset residues."""
    with pytest.raises(ValueError, match="canonical modular residue"):
        PeriodicShiftLCU(
            (2,),
            (PeriodicShiftLCUTerm(1.0, (4,)),),
        )
    with pytest.raises(ValueError, match="duplicate"):
        PeriodicShiftLCU(
            (2,),
            (
                PeriodicShiftLCUTerm(1.0, (1,)),
                PeriodicShiftLCUTerm(2.0, (1,)),
            ),
        )


@pytest.mark.parametrize(
    "coefficients",
    [
        {0: 10**1000},
        {0: 1e308, 1: 1e308},
        {1: 1e308, -3: 1e308},
    ],
)
def test_unrepresentable_coefficients_and_normalizations_fail(
    coefficients: dict[int, complex],
) -> None:
    """Coefficient conversion, alias sums, and alpha must remain finite."""
    with pytest.raises(ValueError, match="finite"):
        PeriodicShiftLCU.from_coefficients(
            coefficients,
            register_sizes=(2,),
        )
