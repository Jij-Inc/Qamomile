"""Tests for ``qamomile.linalg.hermitian.HermitianMatrix``.

These cover constructor validation, the FWHT-based ``to_hamiltonian``
decomposition (against a naive trace-based oracle, against round-trip
reconstruction, and on hand-computed small cases), and Hermitian-preserving
arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

import qamomile.observable as qmo
from qamomile.linalg import HermitianMatrix

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _pauli_1q(x_bit: int, z_bit: int) -> np.ndarray:
    return {
        (0, 0): I2,
        (1, 0): X,
        (0, 1): Z,
        (1, 1): Y,
    }[(x_bit, z_bit)]


def _pauli_string_matrix(x_mask: int, z_mask: int, n: int) -> np.ndarray:
    """Matrix of the Pauli string in the little-endian (qubit 0 = LSB) convention."""
    factors = [
        _pauli_1q((x_mask >> k) & 1, (z_mask >> k) & 1) for k in reversed(range(n))
    ]
    result = factors[0]
    for f in factors[1:]:
        result = np.kron(result, f)
    return result


def _naive_decompose(matrix: np.ndarray) -> np.ndarray:
    """Reference decomposition via ``alpha_P = Tr(M P) / 2**n``."""
    dim = matrix.shape[0]
    n = dim.bit_length() - 1
    coeffs = np.zeros((dim, dim), dtype=float)
    for x_mask in range(dim):
        for z_mask in range(dim):
            p = _pauli_string_matrix(x_mask, z_mask, n)
            coeffs[x_mask, z_mask] = float(np.trace(matrix @ p).real / dim)
    return coeffs


def _reconstruct_from_hamiltonian(h: qmo.Hamiltonian) -> np.ndarray:
    n = h.num_qubits
    dim = 1 << n
    result = np.asarray(h.constant, dtype=complex) * np.eye(dim, dtype=complex)
    for ops, coeff in h.terms.items():
        mat = np.eye(dim, dtype=complex)
        factors = [I2] * n
        for op in ops:
            factors[op.index] = {
                qmo.Pauli.X: X,
                qmo.Pauli.Y: Y,
                qmo.Pauli.Z: Z,
            }[op.pauli]
        term = factors[n - 1]
        for f in reversed(factors[: n - 1]):
            term = np.kron(term, f)
        mat = term
        result = result + complex(coeff) * mat
    return result


def _random_hermitian(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim = 1 << n
    a = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    return (a + a.conj().T) / 2.0


class TestConstructorValidation:
    def test_non_2d_rejected(self):
        with pytest.raises(ValueError, match="2D square"):
            HermitianMatrix(np.array([1.0, 2.0]))

    def test_non_square_rejected(self):
        with pytest.raises(ValueError, match="2D square"):
            HermitianMatrix(np.zeros((2, 4), dtype=complex))

    def test_non_power_of_two_rejected(self):
        with pytest.raises(ValueError, match="power of 2"):
            HermitianMatrix(np.eye(3, dtype=complex))

    def test_non_hermitian_rejected(self):
        bad = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
        with pytest.raises(ValueError, match="Hermitian"):
            HermitianMatrix(bad)

    def test_validate_false_accepts_non_hermitian(self):
        bad = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
        h = HermitianMatrix(bad, validate=False)
        assert h.num_qubits == 1


class TestSingleQubitSanity:
    @pytest.mark.parametrize(
        ("matrix", "expected_term"),
        [
            (X, (qmo.Pauli.X, 0)),
            (Y, (qmo.Pauli.Y, 0)),
            (Z, (qmo.Pauli.Z, 0)),
        ],
    )
    def test_single_pauli_matrix(self, matrix, expected_term):
        ham = HermitianMatrix(matrix).to_hamiltonian()
        assert ham.constant == pytest.approx(0.0)
        assert len(ham.terms) == 1
        [(ops, coeff)] = list(ham.terms.items())
        assert len(ops) == 1
        assert ops[0].pauli == expected_term[0]
        assert ops[0].index == expected_term[1]
        assert coeff == pytest.approx(1.0)

    def test_identity_matrix_goes_to_constant(self):
        ham = HermitianMatrix(np.eye(2, dtype=complex)).to_hamiltonian()
        assert ham.terms == {}
        assert ham.constant == pytest.approx(1.0)


class TestTwoQubitHandComputed:
    def test_xz_plus_half_yi(self):
        """M = (X on q1)(Z on q0) + 0.5 * (Y on q1)(I on q0)."""
        matrix = np.kron(X, Z) + 0.5 * np.kron(Y, I2)
        ham = HermitianMatrix(matrix).to_hamiltonian()

        assert ham.constant == pytest.approx(0.0)
        assert len(ham.terms) == 2

        zx_key = (
            qmo.PauliOperator(qmo.Pauli.Z, 0),
            qmo.PauliOperator(qmo.Pauli.X, 1),
        )
        y1_key = (qmo.PauliOperator(qmo.Pauli.Y, 1),)
        assert ham.terms[zx_key] == pytest.approx(1.0)
        assert ham.terms[y1_key] == pytest.approx(0.5)


class TestConstantTerm:
    def test_scaled_identity_3q(self):
        dim = 8
        matrix = 2.5 * np.eye(dim, dtype=complex)
        ham = HermitianMatrix(matrix).to_hamiltonian()
        assert ham.terms == {}
        assert ham.constant == pytest.approx(2.5)
        assert ham.num_qubits == 3


class TestNaiveOracle:
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_random_hermitian_matches_naive(self, n):
        matrix = _random_hermitian(n, seed=n)
        expected = _naive_decompose(matrix)

        ham = HermitianMatrix(matrix).to_hamiltonian(tol=0.0)

        dim = 1 << n
        actual = np.zeros((dim, dim), dtype=float)
        actual[0, 0] = float(np.real(ham.constant))
        for ops, coeff in ham.terms.items():
            x_mask = 0
            z_mask = 0
            for op in ops:
                if op.pauli == qmo.Pauli.X:
                    x_mask |= 1 << op.index
                elif op.pauli == qmo.Pauli.Z:
                    z_mask |= 1 << op.index
                else:
                    x_mask |= 1 << op.index
                    z_mask |= 1 << op.index
            actual[x_mask, z_mask] = float(np.real(coeff))

        np.testing.assert_allclose(actual, expected, atol=1e-12)


class TestRoundTrip:
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_decompose_then_reconstruct(self, n):
        matrix = _random_hermitian(n, seed=100 + n)
        ham = HermitianMatrix(matrix).to_hamiltonian(tol=0.0)
        rebuilt = _reconstruct_from_hamiltonian(ham)
        np.testing.assert_allclose(rebuilt, matrix, atol=1e-12)


class TestTolDropping:
    def test_small_perturbation_is_dropped(self):
        base = np.kron(Z, Z)
        tiny = 1e-14 * np.kron(X, I2)
        big = HermitianMatrix(base).to_hamiltonian()
        with_tiny = HermitianMatrix(base + tiny).to_hamiltonian(tol=1e-10)
        assert len(big.terms) == len(with_tiny.terms)


class TestArithmetic:
    def test_add(self):
        a = HermitianMatrix(np.kron(X, I2))
        b = HermitianMatrix(np.kron(I2, Z))
        c = a + b
        expected = np.kron(X, I2) + np.kron(I2, Z)
        np.testing.assert_allclose(c.matrix, expected, atol=1e-12)

    def test_sub(self):
        a = HermitianMatrix(np.kron(X, I2))
        b = HermitianMatrix(np.kron(I2, Z))
        c = a - b
        expected = np.kron(X, I2) - np.kron(I2, Z)
        np.testing.assert_allclose(c.matrix, expected, atol=1e-12)

    def test_add_shape_mismatch(self):
        a = HermitianMatrix(X)
        b = HermitianMatrix(np.kron(X, Z))
        with pytest.raises(ValueError, match="Shape mismatch"):
            a + b

    def test_real_scalar_mul(self):
        a = HermitianMatrix(np.kron(X, Z))
        b = 1.5 * a
        np.testing.assert_allclose(b.matrix, 1.5 * np.kron(X, Z), atol=1e-12)
        c = a * 2
        np.testing.assert_allclose(c.matrix, 2.0 * np.kron(X, Z), atol=1e-12)

    def test_complex_scalar_mul_rejected(self):
        a = HermitianMatrix(X)
        with pytest.raises(TypeError, match="real scalars"):
            (1 + 1j) * a

    def test_neg(self):
        a = HermitianMatrix(np.kron(X, Z))
        assert (-a) == (-1) * a


class TestProperties:
    def test_num_qubits_and_shape(self):
        a = HermitianMatrix(np.eye(8, dtype=complex))
        assert a.num_qubits == 3
        assert a.shape == (8, 8)

    def test_matrix_is_read_only(self):
        a = HermitianMatrix(X)
        view = a.matrix
        assert view.flags.writeable is False
        with pytest.raises(ValueError):
            view[0, 0] = 99.0


class TestInternalNonHermitianGuard:
    def test_non_hermitian_via_validate_false_raises_in_decompose(self):
        bad = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
        h = HermitianMatrix(bad, validate=False)
        with pytest.raises(ValueError, match="imaginary"):
            h.to_hamiltonian()
