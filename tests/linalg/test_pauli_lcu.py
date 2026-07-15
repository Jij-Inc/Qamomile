"""Tests for general complex Pauli LCU decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from qamomile.linalg import PauliLCU, PauliLCUTerm
from qamomile.observable import Pauli, PauliOperator

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_PAULI_MATRIX = {Pauli.I: I2, Pauli.X: X, Pauli.Y: Y, Pauli.Z: Z}


def _term_matrix(term: PauliLCUTerm, num_qubits: int) -> np.ndarray:
    """Return one sparse Pauli word as a dense little-endian matrix.

    Args:
        term (PauliLCUTerm): Sparse term whose coefficient is ignored.
        num_qubits (int): System width of the dense matrix.

    Returns:
        np.ndarray: Dense Pauli-word matrix.
    """
    if num_qubits == 0:
        return np.ones((1, 1), dtype=np.complex128)
    by_index = {operator.index: operator.pauli for operator in term.operators}
    result = np.ones((1, 1), dtype=np.complex128)
    for qubit in reversed(range(num_qubits)):
        result = np.kron(result, _PAULI_MATRIX[by_index.get(qubit, Pauli.I)])
    return result


def _reconstruct(lcu: PauliLCU) -> np.ndarray:
    """Reconstruct a dense matrix from an immutable Pauli LCU.

    Args:
        lcu (PauliLCU): Decomposition to reconstruct.

    Returns:
        np.ndarray: Dense complex matrix.
    """
    dim = 1 << lcu.num_qubits
    result = np.zeros((dim, dim), dtype=np.complex128)
    for term in lcu.terms:
        result += term.coefficient * _term_matrix(term, lcu.num_qubits)
    return result


def _coefficient_map(lcu: PauliLCU) -> dict[tuple[tuple[int, Pauli], ...], complex]:
    """Return coefficients keyed by canonical sparse Pauli words.

    Args:
        lcu (PauliLCU): Decomposition to index.

    Returns:
        dict[tuple[tuple[int, Pauli], ...], complex]: Coefficients by word.
    """
    return {
        tuple((operator.index, operator.pauli) for operator in term.operators): (
            term.coefficient
        )
        for term in lcu.terms
    }


def test_non_hermitian_lowering_operator_has_complex_pauli_coefficient() -> None:
    """The non-Hermitian lowering operator decomposes as ``(X + iY) / 2``."""
    matrix = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    lcu = PauliLCU.from_matrix(matrix)
    coefficients = _coefficient_map(lcu)

    assert coefficients == {
        ((0, Pauli.X),): pytest.approx(0.5),
        ((0, Pauli.Y),): pytest.approx(0.5j),
    }
    assert lcu.alpha == pytest.approx(1.0)
    np.testing.assert_allclose(_reconstruct(lcu), matrix, atol=1e-15, rtol=0.0)


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 1, 2, 42])
def test_random_complex_matrix_matches_trace_oracle(
    num_qubits: int,
    seed: int,
) -> None:
    """Complex FWHT coefficients match the independent trace formula."""
    rng = np.random.default_rng(seed)
    dim = 1 << num_qubits
    matrix = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    lcu = PauliLCU.from_matrix(matrix)

    for term in lcu.terms:
        pauli = _term_matrix(term, num_qubits)
        expected = np.trace(pauli @ matrix) / dim
        assert term.coefficient == pytest.approx(expected, abs=1e-12)
    np.testing.assert_allclose(_reconstruct(lcu), matrix, atol=1e-12, rtol=0.0)
    assert lcu.alpha == pytest.approx(
        sum(abs(term.coefficient) for term in lcu.terms),
        abs=1e-12,
    )


def test_complex_identity_remains_a_normal_term() -> None:
    """Identity coefficients retain their complex phase and sparse empty word."""
    coefficient = -0.25 + 0.75j
    lcu = PauliLCU.from_matrix(coefficient * np.eye(4, dtype=np.complex128))

    assert lcu.terms == (PauliLCUTerm(coefficient, ()),)


def test_large_finite_identity_does_not_overflow_fwht_intermediates() -> None:
    """Exact FWHT accumulation avoids overflow before final averaging."""
    coefficient = 1e308
    lcu = PauliLCU.from_matrix(coefficient * I2)

    assert lcu.terms == (PauliLCUTerm(coefficient, ()),)


def test_subnormal_identity_survives_exact_fwht_accumulation() -> None:
    """Exact FWHT accumulation preserves the smallest complex128 value."""
    coefficient = np.nextafter(0.0, 1.0)
    lcu = PauliLCU.from_matrix(coefficient * np.eye(4, dtype=np.complex128))

    assert lcu.terms == (PauliLCUTerm(coefficient, ()),)


def test_fwht_preserves_tiny_terms_across_extreme_row_cancellation() -> None:
    """Exact integer butterflies retain representable mixed-scale outputs."""
    matrix = np.diag([1e308, -1e308, 1e-300, 0.0]).astype(np.complex128)
    coefficients = _coefficient_map(PauliLCU.from_matrix(matrix))

    assert coefficients == {
        (): 2.5e-301,
        ((0, Pauli.Z),): 5e307,
        ((1, Pauli.Z),): -2.5e-301,
        ((0, Pauli.Z), (1, Pauli.Z)): 5e307,
    }


def test_two_qubit_words_use_qubit_zero_as_lsb() -> None:
    """The existing FWHT LSB convention maps matrix factors to q0 correctly."""
    matrix = (1.25 - 0.5j) * np.kron(X, Z)
    lcu = PauliLCU.from_matrix(matrix)

    assert lcu.terms == (
        PauliLCUTerm(
            1.25 - 0.5j,
            (
                PauliOperator(Pauli.Z, 0),
                PauliOperator(Pauli.X, 1),
            ),
        ),
    )


def test_default_threshold_keeps_tiny_nonzero_coefficients() -> None:
    """The exact-by-default API omits only coefficients equal to zero."""
    tiny = 1e-14
    lcu = PauliLCU.from_matrix(X + tiny * Z)

    assert lcu.num_terms == 2
    assert _coefficient_map(lcu)[((0, Pauli.Z),)] == pytest.approx(tiny)
    assert lcu.truncation_error_bound == 0.0


def test_explicit_threshold_records_operator_norm_error_bound() -> None:
    """Dropped coefficient magnitudes bound the resulting spectral-norm error."""
    tiny = 1e-8
    matrix = X + tiny * Z
    lcu = PauliLCU.from_matrix(matrix, atol=1e-7)
    error = matrix - _reconstruct(lcu)

    assert lcu.num_terms == 1
    assert lcu.truncation_error_bound == pytest.approx(tiny)
    assert np.linalg.norm(error, ord=2) <= lcu.truncation_error_bound + 1e-15


def test_zero_and_all_pruned_matrices_are_valid_lcus() -> None:
    """Zero decompositions retain their system width and approximation bound."""
    zero = PauliLCU.from_matrix(np.zeros((4, 4), dtype=np.complex128))
    pruned = PauliLCU.from_matrix(1e-8 * np.kron(X, I2), atol=1e-7)

    assert zero.num_qubits == 2
    assert zero.terms == ()
    assert zero.alpha == 0.0
    assert zero.truncation_error_bound == 0.0
    assert pruned.terms == ()
    assert pruned.alpha == 0.0
    assert pruned.truncation_error_bound == pytest.approx(1e-8)


def test_scalar_matrix_decomposition_is_supported_by_linalg() -> None:
    """A ``1 x 1`` matrix is represented as a zero-qubit identity term."""
    lcu = PauliLCU.from_matrix(np.array([[2.0 - 3.0j]]))

    assert lcu.num_qubits == 0
    assert lcu.terms == (PauliLCUTerm(2.0 - 3.0j, ()),)


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([1.0, 2.0]),
        np.zeros((2, 4)),
        np.eye(3),
        np.array([[np.nan, 0.0], [0.0, 1.0]]),
        np.array([[np.inf, 0.0], [0.0, 1.0]]),
    ],
)
def test_invalid_matrix_is_rejected(matrix: np.ndarray) -> None:
    """Shape, dimension, and finite-value validation precede decomposition."""
    with pytest.raises(ValueError):
        PauliLCU.from_matrix(matrix)


@pytest.mark.parametrize("atol", [True, "small", "1", 1j, np.complex128(2 + 3j)])
def test_non_real_threshold_is_rejected(atol: object) -> None:
    """The coefficient threshold must be a real scalar."""
    with pytest.raises(TypeError):
        PauliLCU.from_matrix(X, atol=atol)  # type: ignore[arg-type]


@pytest.mark.parametrize("atol", [-1.0, np.nan, np.inf])
def test_invalid_real_threshold_is_rejected(atol: float) -> None:
    """Negative and non-finite real thresholds are invalid."""
    with pytest.raises(ValueError):
        PauliLCU.from_matrix(X, atol=atol)


@pytest.mark.parametrize(
    "bound",
    [True, "1", 1j, np.complex128(2 + 3j)],
)
def test_non_real_truncation_bound_is_rejected(bound: object) -> None:
    """The stored truncation bound accepts only actual real scalars."""
    with pytest.raises(TypeError):
        PauliLCU(1, (), truncation_error_bound=bound)  # type: ignore[arg-type]


def test_unrepresentable_numeric_inputs_raise_documented_errors() -> None:
    """Huge conversions and L1 normalizations fail with public error types."""
    huge_integer = 10**1000
    with pytest.raises(ValueError, match="numeric|finite"):
        PauliLCU.from_matrix([[huge_integer]])
    with pytest.raises(ValueError, match="finite"):
        PauliLCU.from_matrix(X, atol=huge_integer)
    with pytest.raises(ValueError, match="finite"):
        PauliLCUTerm(huge_integer, ())
    with pytest.raises(ValueError, match="normalization"):
        PauliLCU(
            1,
            (
                PauliLCUTerm(1e308, ()),
                PauliLCUTerm(1e308, (PauliOperator(Pauli.X, 0),)),
            ),
        )


def test_term_constructor_canonicalizes_operator_order() -> None:
    """Sparse Pauli words become deterministic increasing-index tuples."""
    term = PauliLCUTerm(
        1j,
        (
            PauliOperator(Pauli.X, 2),
            PauliOperator(Pauli.Z, 0),
        ),
    )

    assert tuple(operator.index for operator in term.operators) == (0, 2)


def test_term_constructor_canonicalizes_signed_complex_zero() -> None:
    """Equal coefficients cannot diverge at the negative-real branch cut."""
    positive_zero = PauliLCUTerm(complex(-1.0, 0.0), ())
    negative_zero = PauliLCUTerm(complex(-1.0, -0.0), ())

    assert positive_zero == negative_zero
    assert np.signbit(positive_zero.coefficient.imag) is np.False_
    assert np.signbit(negative_zero.coefficient.imag) is np.False_


@pytest.mark.parametrize(
    "term",
    [
        lambda: PauliLCUTerm(0.0, ()),
        lambda: PauliLCUTerm(np.nan, ()),
        lambda: PauliLCUTerm("1", ()),
        lambda: PauliLCUTerm(
            1.0,
            (PauliOperator("X", 0),),  # type: ignore[arg-type]
        ),
        lambda: PauliLCUTerm(1.0, (PauliOperator(Pauli.I, 0),)),
        lambda: PauliLCUTerm(
            1.0,
            (
                PauliOperator(Pauli.X, 0),
                PauliOperator(Pauli.Y, 0),
            ),
        ),
    ],
)
def test_invalid_term_is_rejected(term: object) -> None:
    """Terms require finite nonzero coefficients and canonical sparse words."""
    with pytest.raises((TypeError, ValueError)):
        term()  # type: ignore[operator]


def test_lcu_constructor_rejects_duplicate_or_out_of_range_words() -> None:
    """An immutable LCU cannot contain ambiguous or invalid Pauli terms."""
    term = PauliLCUTerm(1.0, (PauliOperator(Pauli.X, 0),))

    with pytest.raises(ValueError, match="duplicate"):
        PauliLCU(1, (term, PauliLCUTerm(2.0, term.operators)))
    with pytest.raises(ValueError, match="outside"):
        PauliLCU(0, (term,))
