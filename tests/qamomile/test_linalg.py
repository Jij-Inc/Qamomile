"""Classical sampled-subspace linear algebra tests."""

from __future__ import annotations

import numpy as np
import pytest

import qamomile as qm
import qamomile.linalg as linalg_module
from qamomile.stdlib import pauli_block_encoding


def _dense_two_qubit_hamiltonian() -> np.ndarray:
    """Build the dense reference in Qamomile's logical-qubit order.

    Returns:
        numpy.ndarray: Four-dimensional Hermitian reference matrix.
    """

    identity = np.eye(2, dtype=np.complex128)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    pauli_z = np.diag([1, -1]).astype(np.complex128)
    return (
        0.25 * np.eye(4, dtype=np.complex128)
        + np.kron(identity, pauli_z)
        + 0.5 * np.kron(pauli_x, identity)
        + 0.2 * np.kron(identity, pauli_y)
        + 0.75 * np.kron(pauli_y, pauli_y)
    )


def _random_hermitian(num_qubits: int, seed: int) -> np.ndarray:
    """Return one deterministic dense Hermitian matrix.

    Args:
        num_qubits (int): Logical width.
        seed (int): NumPy generator seed.

    Returns:
        numpy.ndarray: Dense complex Hermitian matrix.
    """

    generator = np.random.default_rng(seed)
    dimension = 1 << num_qubits
    raw = generator.standard_normal(
        (dimension, dimension)
    ) + 1j * generator.standard_normal((dimension, dimension))
    return (raw + raw.conjugate().T) / 2.0


def _full_basis(num_qubits: int) -> tuple[tuple[int, ...], ...]:
    """Return computational states ordered by their little-endian index.

    Args:
        num_qubits (int): Logical width.

    Returns:
        tuple[tuple[int, ...], ...]: Complete logical-bit basis.
    """

    return tuple(
        tuple((basis >> index) & 1 for index in range(num_qubits))
        for basis in range(1 << num_qubits)
    )


@pytest.mark.parametrize(
    ("matrix", "message"),
    (
        (np.array([1.0, 2.0]), "two-dimensional"),
        (np.zeros((2, 4)), "square"),
        (np.eye(3), "power of two"),
        (np.array([[0.0, 1.0], [0.0, 0.0]]), "not Hermitian"),
        (np.array([[np.inf]]), "finite"),
    ),
)
def test_hermitian_matrix_rejects_invalid_input(
    matrix: np.ndarray,
    message: str,
) -> None:
    """Rust rejects malformed, unsafe, or non-Hermitian dense input."""

    with pytest.raises(ValueError, match=message):
        qm.linalg.HermitianMatrix(matrix)


def test_hermitian_matrix_uses_absolute_not_relative_tolerance() -> None:
    """Large entries cannot hide an absolute Hermitian mismatch."""

    matrix = np.array([[0.0, 1_000_010.0], [1_000_000.0, 0.0]])

    with pytest.raises(ValueError, match="not Hermitian"):
        qm.linalg.HermitianMatrix(matrix, atol=1e-10)


def test_hermitian_matrix_cannot_bypass_its_native_invariant() -> None:
    """The public Hermitian type never stores an unproved general matrix."""

    source = np.array([[0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="not Hermitian"):
        qm.linalg.HermitianMatrix(source)
    with pytest.raises(TypeError, match="validate"):
        qm.linalg.HermitianMatrix(source, validate=False)  # type: ignore[call-arg]


@pytest.mark.parametrize("num_qubits", (1, 2, 3))
def test_hermitian_matrix_round_trip_matches_dense_input(num_qubits: int) -> None:
    """Rust FWHT output reconstructs the original matrix in LSB order."""

    matrix = _random_hermitian(num_qubits, seed=100 + num_qubits)
    hamiltonian = qm.linalg.HermitianMatrix(matrix).to_hamiltonian(tol=0.0)
    rebuilt = qm.linalg.subspace_hamiltonian(
        _full_basis(num_qubits),
        hamiltonian,
    )

    np.testing.assert_allclose(rebuilt, matrix, rtol=0.0, atol=1e-12)


def test_hermitian_matrix_preserves_little_endian_pauli_order() -> None:
    """The left Kronecker factor maps to the higher logical qubit."""

    identity = np.eye(2, dtype=np.complex128)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pauli_z = np.diag([1, -1]).astype(np.complex128)
    hamiltonian = qm.linalg.HermitianMatrix(
        np.kron(pauli_x, pauli_z) + 0.5 * np.kron(identity, pauli_x)
    ).to_hamiltonian(tol=0.0)

    assert hamiltonian == qm.X(1) * qm.Z(0) + 0.5 * qm.X(0)


def test_hermitian_matrix_truncation_and_extreme_identity() -> None:
    """Thresholding is explicit and exact accumulation avoids overflow."""

    identity = np.eye(2, dtype=np.complex128)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pauli_z = np.diag([1, -1]).astype(np.complex128)
    truncated = qm.linalg.HermitianMatrix(pauli_x + 1e-14 * pauli_z).to_hamiltonian(
        tol=1e-12
    )
    extreme = qm.linalg.HermitianMatrix(1e308 * identity).to_hamiltonian(tol=0.0)

    assert truncated == qm.X(0)
    assert extreme == qm.Hamiltonian.identity(1e308, num_qubits=1)


def test_hermitian_matrix_owns_read_only_storage_and_arithmetic() -> None:
    """The wrapper cannot be mutated through either input or property views."""

    source = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    matrix = qm.linalg.HermitianMatrix(source)
    source[0, 1] = 9.0

    assert matrix.num_qubits == 1
    assert matrix.shape == (2, 2)
    assert matrix.matrix[0, 1] == 1.0
    with pytest.raises(ValueError):
        matrix.matrix[0, 1] = 2.0
    np.testing.assert_array_equal((matrix + matrix).matrix, 2.0 * matrix.matrix)
    np.testing.assert_array_equal((matrix - matrix).matrix, np.zeros((2, 2)))
    np.testing.assert_array_equal((-matrix).matrix, -matrix.matrix)
    np.testing.assert_array_equal((matrix * 2).matrix, 2.0 * matrix.matrix)
    np.testing.assert_array_equal((matrix / 2).matrix, 0.5 * matrix.matrix)
    assert matrix == qm.linalg.HermitianMatrix(matrix.matrix)
    assert repr(matrix) == "HermitianMatrix(num_qubits=1, shape=(2, 2))"


def test_hermitian_matrix_arithmetic_rejects_invalid_scalars_and_shapes() -> None:
    """Host convenience algebra preserves the Hermitian invariant."""

    one = qm.linalg.HermitianMatrix(np.eye(2))
    two = qm.linalg.HermitianMatrix(np.eye(4))

    with pytest.raises(ValueError, match="Shape mismatch"):
        one + two
    with pytest.raises(TypeError, match="real scalars"):
        one * (1 + 1j)
    with pytest.raises(ValueError, match="Q7407"):
        one * np.inf
    np.testing.assert_array_equal((one / 1e-16).matrix, np.eye(2) * 1e16)
    np.testing.assert_array_equal(
        (qm.linalg.HermitianMatrix(np.zeros((2, 2))) / np.nextafter(0.0, 1.0)).matrix,
        np.zeros((2, 2)),
    )
    with pytest.raises(ZeroDivisionError, match="Q7416"):
        one / 0.0
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="Q7407"):
        qm.linalg.HermitianMatrix(np.eye(2) * 1e308) / 1e-308


def test_hermitian_matrix_rejects_over_budget_before_element_transfer() -> None:
    """The Rust-owned cap prevents an exponential 512-square transform."""

    with pytest.raises(ValueError, match="decomposition budget"):
        qm.linalg.HermitianMatrix(np.zeros((512, 512), dtype=np.complex128))


def test_hermitian_matrix_fails_closed_without_native_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No Python FWHT silently replaces an absent Rust implementation."""

    matrix = qm.linalg.HermitianMatrix(np.eye(2))
    with monkeypatch.context() as patch:
        patch.setattr(linalg_module.native, "validate_hermitian_matrix", None)
        with pytest.raises(TypeError, match="native dense-matrix algebra"):
            qm.linalg.HermitianMatrix(np.eye(2))
    with monkeypatch.context() as patch:
        patch.setattr(linalg_module.native, "decompose_hermitian_matrix", None)
        with pytest.raises(TypeError, match="native dense-matrix algebra"):
            matrix.to_hamiltonian()


def test_hermitian_matrix_rejects_malformed_native_arithmetic_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arithmetic never stores an unchecked native ABI projection."""

    matrix = qm.linalg.HermitianMatrix(np.eye(2))
    monkeypatch.setattr(
        linalg_module.native,
        "scale_hermitian_matrix",
        lambda *_: [(1.0, 0.0)],
    )

    with pytest.raises(TypeError, match="violates the Python ABI"):
        matrix * 2.0


def test_general_matrix_decomposes_into_complex_pauli_operator() -> None:
    """A non-Hermitian matrix retains the complex Pauli phase."""

    matrix = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    decomposition = qm.linalg.decompose_pauli_matrix(matrix)

    assert decomposition.operator == 0.5 * qm.X(0) + 0.5j * qm.Y(0)
    assert decomposition.operator.is_hermitian is False
    assert decomposition.alpha == 1.0
    assert decomposition.num_terms == 2
    assert decomposition.truncation_error_bound == 0.0


def test_pauli_decomposition_alpha_crosses_native_one_norm_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public LCU norm has no independent Python summation rule."""

    decomposition = qm.linalg.decompose_pauli_matrix(
        np.array([[0.0, 7.0], [-1.0, 0.0]], dtype=np.complex128)
    )
    with pytest.raises(TypeError):
        qm.linalg.PauliDecomposition(  # type: ignore[call-arg]
            3.0 * qm.X(0) + 4.0j * qm.Y(0),
            0.0,
        )
    native_derive = linalg_module.native.observable_lcu_one_norm
    calls = 0
    payload: bytes | None = None

    def record_native_derivation(value: bytes) -> object:
        """Record and delegate one canonical observable norm request."""

        nonlocal calls, payload
        calls += 1
        payload = value
        return native_derive(value)

    monkeypatch.setattr(
        linalg_module.native,
        "observable_lcu_one_norm",
        record_native_derivation,
    )

    assert decomposition.alpha == 7.0
    assert calls == 1
    assert payload is decomposition.operator._native_literal

    monkeypatch.setattr(linalg_module.native, "observable_lcu_one_norm", None)
    with pytest.raises(TypeError, match="Q7400"):
        _ = decomposition.alpha

    monkeypatch.setattr(
        linalg_module.native,
        "observable_lcu_one_norm",
        lambda _payload: -1.0,
    )
    with pytest.raises(TypeError, match="violates the Python ABI"):
        _ = decomposition.alpha


@pytest.mark.parametrize("num_qubits", (1, 2, 3))
def test_general_matrix_pauli_decomposition_round_trips(num_qubits: int) -> None:
    """The complex Rust FWHT reconstructs arbitrary dense matrices."""

    generator = np.random.default_rng(700 + num_qubits)
    dimension = 1 << num_qubits
    matrix = generator.standard_normal(
        (dimension, dimension)
    ) + 1j * generator.standard_normal((dimension, dimension))
    decomposition = qm.linalg.decompose_pauli_matrix(matrix)
    rebuilt = qm.linalg.subspace_hamiltonian(
        _full_basis(num_qubits),
        decomposition.operator,
    )

    np.testing.assert_allclose(rebuilt, matrix, rtol=0.0, atol=1e-12)


def test_general_matrix_decomposition_reports_truncation_bound() -> None:
    """Omitted coefficient magnitudes contribute a conservative bound."""

    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pauli_z = np.diag([1, -1]).astype(np.complex128)
    decomposition = qm.linalg.decompose_pauli_matrix(
        pauli_x + 1e-14 * pauli_z,
        atol=1e-12,
    )

    assert decomposition.operator == qm.X(0)
    assert decomposition.num_terms == 1
    assert decomposition.truncation_error_bound >= 1e-14
    assert decomposition.truncation_error_bound < 1.000000000000001e-14


def test_general_matrix_decomposition_feeds_existing_block_encoding() -> None:
    """Dense conversion adds no second LCU construction path."""

    matrix = np.array([[0.0, 1.0j], [1.0, 0.0]], dtype=np.complex128)
    decomposition = qm.linalg.decompose_pauli_matrix(matrix)
    encoding = pauli_block_encoding(decomposition.operator)

    assert encoding.normalization == pytest.approx(decomposition.alpha)
    assert encoding.system_width == 1


def test_general_matrix_decomposition_rejects_invalid_threshold() -> None:
    """Coefficient filtering accepts only a finite non-negative threshold."""

    with pytest.raises(ValueError, match="tolerances"):
        qm.linalg.decompose_pauli_matrix(np.eye(2), atol=-1.0)
    with pytest.raises(ValueError, match="finite"):
        qm.linalg.decompose_pauli_matrix(np.eye(2), atol=float("nan"))


def test_general_matrix_decomposition_fails_closed_without_native_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An absent Rust complex transform has no NumPy fallback."""

    monkeypatch.setattr(linalg_module.native, "decompose_pauli_matrix", None)
    with pytest.raises(TypeError, match="native dense-matrix algebra"):
        qm.linalg.decompose_pauli_matrix(np.eye(2))


def test_subspace_hamiltonian_matches_dense_pauli_algebra() -> None:
    """XOR/parity projection matches a direct dense two-qubit matrix."""

    samples = ((0, 0), (1, 0), (0, 1), (1, 1))
    hamiltonian = (
        0.25 + qm.Z(0) + 0.5 * qm.X(1) + 0.2 * qm.Y(0) + 0.75 * qm.Y(0) * qm.Y(1)
    )

    projected = qm.linalg.subspace_hamiltonian(samples, hamiltonian)

    np.testing.assert_allclose(
        projected,
        _dense_two_qubit_hamiltonian(),
        rtol=0.0,
        atol=1e-12,
    )


def test_solve_subspace_is_variational_and_exact_at_full_dimension() -> None:
    """Nested QSCI spaces decrease to the exact full-space ground energy."""

    samples = ((0, 0), (1, 0), (0, 1), (1, 1))
    hamiltonian = -qm.Z(0) * qm.Z(1) - 0.7 * qm.X(0) - 0.7 * qm.X(1)
    exact = -np.sqrt(1.0 + 4.0 * 0.7**2)
    energies = tuple(
        float(qm.linalg.solve_subspace(samples[:size], hamiltonian)[0][0])
        for size in range(1, len(samples) + 1)
    )

    assert all(left >= right - 1e-12 for left, right in zip(energies, energies[1:]))
    assert all(energy >= exact - 1e-12 for energy in energies)
    assert np.isclose(energies[-1], exact, rtol=0.0, atol=1e-12)


def test_solve_subspace_returns_typed_empty_arrays() -> None:
    """An empty sample basis has explicit zero-dimensional outputs."""

    eigenvalues, eigenvectors = qm.linalg.solve_subspace((), qm.Z(0))

    assert eigenvalues.shape == (0,)
    assert eigenvalues.dtype == np.float64
    assert eigenvectors.shape == (0, 0)
    assert eigenvectors.dtype == np.complex128


def test_subspace_accepts_integral_numpy_sample_arrays() -> None:
    """NumPy sampler output follows the same validated bitstring path."""

    samples = np.asarray(((0,), (1,)), dtype=np.int8)

    np.testing.assert_allclose(
        qm.linalg.subspace_hamiltonian(samples, qm.X(0)),
        np.array([[0, 1], [1, 0]], dtype=np.complex128),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("samples", "error", "message"),
    (
        (((0,),), ValueError, "exactly 2 bits"),
        (((0, 2),), ValueError, "zero or one"),
        (((0, 0.5),), TypeError, "integers"),
        (("00",), TypeError, "bit sequence"),
    ),
)
def test_subspace_rejects_ambiguous_sample_bases(
    samples: object,
    error: type[Exception],
    message: str,
) -> None:
    """Malformed and repeated bitstrings fail before matrix construction."""

    with pytest.raises(error, match=message):
        qm.linalg.subspace_hamiltonian(samples, qm.Z(0) + qm.Z(1))  # type: ignore[arg-type]


def test_solve_subspace_rejects_nonhermitian_observables() -> None:
    """The eigensolver never silently projects a non-Hermitian observable."""

    with pytest.raises(ValueError, match="Hermitian"):
        qm.linalg.solve_subspace(((0,),), 1j * qm.Z(0))


def test_nonfinite_observable_cannot_reach_subspace_solver() -> None:
    """The canonical observable boundary rejects invalid solver input early."""

    with pytest.raises(ValueError, match="finite"):
        qm.Hamiltonian.identity(float("nan"), num_qubits=1)


def test_generalized_subspace_matches_cross_basis_oracle() -> None:
    """Mixed Z/X states retain their non-orthogonal overlap matrix."""

    inverse_sqrt_two = 1.0 / np.sqrt(2.0)
    hamiltonian, overlap = qm.linalg.generalized_subspace_matrices(
        ((0,), (0,)),
        ((qm.Pauli.Z,), (qm.Pauli.X,)),
        qm.Z(0),
    )

    np.testing.assert_allclose(
        overlap,
        np.array([[1.0, inverse_sqrt_two], [inverse_sqrt_two, 1.0]]),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        hamiltonian,
        np.array([[1.0, inverse_sqrt_two], [inverse_sqrt_two, 0.0]]),
        rtol=0.0,
        atol=1e-15,
    )


def test_subspace_projection_reuses_native_observable_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Projection forwards the compiler-normalized literal without rebuilding it."""

    operator = qm.Z(0)
    native_project = linalg_module.native.project_pauli_subspace
    payload: bytes | None = None

    def record_native_projection(
        value: bytes,
        samples: tuple[tuple[int, ...], ...],
        bases: tuple[tuple[int, ...], ...],
    ) -> object:
        """Record and delegate one sampled-subspace projection."""

        nonlocal payload
        payload = value
        return native_project(value, samples, bases)

    monkeypatch.setattr(
        linalg_module.native,
        "project_pauli_subspace",
        record_native_projection,
    )

    qm.linalg.subspace_hamiltonian(((0,),), operator)

    assert payload is operator._native_literal


def test_generalized_subspace_y_basis_convention() -> None:
    """Y bit zero is the positive-i eigenstate."""

    hamiltonian, overlap = qm.linalg.generalized_subspace_matrices(
        ((0,), (1,)),
        (qm.Pauli.Y,),
        qm.Y(0),
    )

    np.testing.assert_allclose(hamiltonian, np.diag([1.0, -1.0]), atol=1e-15)
    np.testing.assert_allclose(overlap, np.eye(2), atol=1e-15)


def test_generalized_solver_handles_mixed_and_duplicate_states() -> None:
    """Overlap whitening removes null directions instead of rejecting rows."""

    mixed_values, _ = qm.linalg.solve_subspace(
        ((0,),),
        -qm.X(0),
        bases=(qm.Pauli.X,),
    )
    duplicate_values, duplicate_vectors = qm.linalg.solve_subspace(
        ((0,), (0,)),
        qm.Z(0),
    )

    np.testing.assert_allclose(mixed_values, [-1.0], atol=1e-15)
    np.testing.assert_allclose(duplicate_values, [1.0], atol=1e-15)
    assert duplicate_vectors.shape == (2, 1)


@pytest.mark.parametrize(
    ("bases", "error", "message"),
    (
        ((qm.Pauli.X, qm.Pauli.Y), TypeError, "basis row"),
        (((qm.Pauli.X,),), ValueError, "one row"),
        ((3,), ValueError, "zero, one, or two"),
        ((True,), TypeError, "basis entries"),
    ),
)
def test_generalized_subspace_rejects_invalid_bases(
    bases: object,
    error: type[Exception],
    message: str,
) -> None:
    """Basis labels and shapes fail before the native projection."""

    with pytest.raises(error, match=message):
        qm.linalg.generalized_subspace_matrices(
            ((0,), (1,)),
            bases,  # type: ignore[arg-type]
            qm.Z(0),
        )


def test_subspace_projection_fails_closed_without_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Z and mixed projections share one required native implementation."""

    monkeypatch.setattr(linalg_module.native, "project_pauli_subspace", None)
    with pytest.raises(TypeError, match="native Pauli subspace"):
        qm.linalg.subspace_hamiltonian(((0,),), qm.Z(0))
    with pytest.raises(TypeError, match="native Pauli subspace"):
        qm.linalg.generalized_subspace_matrices(
            ((0,),),
            (qm.Pauli.X,),
            qm.Z(0),
        )
