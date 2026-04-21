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

    def test_large_magnitude_non_hermitian_is_deferred_to_fwht_guard(self):
        """Document the known ``rtol`` gap in the constructor check.

        ``np.allclose`` in :class:`HermitianMatrix` uses the default
        ``rtol=1e-5``, so a non-Hermitian input whose per-entry deviation is
        small relative to magnitude but much larger than ``atol`` slips past
        the constructor check and is only caught later by the FWHT
        imaginary-residue guard inside :meth:`to_hamiltonian`.

        Effective tolerance at the constructor is
        ``atol + rtol * |b| = 1e-10 + 1e-5 * 1e6 = 10``, so a 10-unit
        off-diagonal asymmetry at magnitude 1e6 is accepted. Fixing this
        means passing ``rtol=0.0`` to ``np.allclose`` in the constructor.
        """
        magnitude = 1e6
        asymmetry = 10.0  # ≫ atol=1e-10 but ≤ rtol * magnitude with default rtol
        bad = np.array(
            [[0.0, magnitude + asymmetry], [magnitude, 0.0]], dtype=complex
        )
        h = HermitianMatrix(bad)
        with pytest.raises(ValueError, match="imaginary"):
            h.to_hamiltonian()


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
        rebuilt = ham.to_numpy()
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

    def test_numpy_scalar_types_accepted_mul(self):
        """NumPy numeric scalars (``np.float64``, ``np.int64``) should be
        accepted the same way as their Python counterparts."""
        a = HermitianMatrix(np.kron(X, Z))
        np.testing.assert_allclose(
            (np.float64(1.5) * a).matrix, 1.5 * np.kron(X, Z), atol=1e-12
        )
        np.testing.assert_allclose(
            (a * np.int64(3)).matrix, 3.0 * np.kron(X, Z), atol=1e-12
        )
        np.testing.assert_allclose(
            (np.complex128(2.0 + 0j) * a).matrix,
            2.0 * np.kron(X, Z),
            atol=1e-12,
        )

    def test_real_scalar_div(self):
        a = HermitianMatrix(np.kron(X, Z))
        b = a / 2
        np.testing.assert_allclose(b.matrix, 0.5 * np.kron(X, Z), atol=1e-12)

    def test_complex_scalar_div_rejected(self):
        a = HermitianMatrix(X)
        with pytest.raises(TypeError, match="real scalars"):
            a / (1 + 1j)

    def test_numpy_scalar_types_accepted_div(self):
        a = HermitianMatrix(np.kron(X, Z))
        np.testing.assert_allclose(
            (a / np.float64(2.0)).matrix, 0.5 * np.kron(X, Z), atol=1e-12
        )
        np.testing.assert_allclose(
            (a / np.int64(4)).matrix, 0.25 * np.kron(X, Z), atol=1e-12
        )

    def test_neg(self):
        a = HermitianMatrix(np.kron(X, Z))
        assert (-a) == (-1) * a

    def test_add_sub_identity_1q(self):
        a = HermitianMatrix(I2)
        b = HermitianMatrix(I2)
        np.testing.assert_allclose((a + b).matrix, 2 * I2, atol=1e-12)
        np.testing.assert_allclose(
            (a - b).matrix, np.zeros((2, 2), dtype=complex), atol=1e-12
        )
        assert (a + b).num_qubits == 1
        assert (a - b).num_qubits == 1

    def test_div_by_zero_raises(self):
        a = HermitianMatrix(X)
        with pytest.raises(ZeroDivisionError):
            a / 0
        with pytest.raises(ZeroDivisionError):
            a / 0.0
        with pytest.raises(ZeroDivisionError):
            a / 1e-16

    def test_eq_non_hermitian_returns_notimplemented(self):
        a = HermitianMatrix(X)
        assert a.__eq__(5) is NotImplemented
        assert a.__eq__("not a matrix") is NotImplemented
        assert a.__eq__(np.array([[0, 1], [1, 0]], dtype=complex)) is NotImplemented
        assert (a == 5) is False

    @pytest.mark.parametrize("seed", [offset + 901 for offset in range(30)])
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_random_add_matches_hamiltonian_add(self, n, seed):
        rng = np.random.default_rng(seed)
        seed_a, seed_b = (int(x) for x in rng.integers(0, 2**31, size=2))
        a = HermitianMatrix(_random_hermitian(n, seed=seed_a))
        b = HermitianMatrix(_random_hermitian(n, seed=seed_b))
        lhs = (a + b).to_hamiltonian(tol=0.0).to_numpy()
        rhs = (a.to_hamiltonian(tol=0.0) + b.to_hamiltonian(tol=0.0)).to_numpy()
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    @pytest.mark.parametrize("seed", [offset + 901 for offset in range(30)])
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_random_sub_matches_hamiltonian_sub(self, n, seed):
        rng = np.random.default_rng(seed)
        seed_a, seed_b = (int(x) for x in rng.integers(0, 2**31, size=2))
        a = HermitianMatrix(_random_hermitian(n, seed=seed_a))
        b = HermitianMatrix(_random_hermitian(n, seed=seed_b))
        lhs = (a - b).to_hamiltonian(tol=0.0).to_numpy()
        rhs = (a.to_hamiltonian(tol=0.0) - b.to_hamiltonian(tol=0.0)).to_numpy()
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)


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
