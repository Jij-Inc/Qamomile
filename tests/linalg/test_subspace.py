"""Tests for :mod:`qamomile.linalg.subspace`.

The Z-basis path is checked against the full dense matrix produced by
:meth:`qamomile.observable.Hamiltonian.to_numpy` (round-trip when the
sample list spans the whole Hilbert space, projected oracle for proper
subspaces). The mixed-basis path is checked against
``U^† H_full U`` where ``U`` is the Pauli-eigenbasis rotation matching
the convention documented at the top of ``subspace.py`` (Y eigenstate
``|+i⟩`` is the ``+1`` eigenvector of Y).
"""

from __future__ import annotations

import numpy as np
import pytest

import qamomile.observable as qm_o
from qamomile.linalg import (
    HermitianMatrix,
    generalized_subspace_matrices,
    solve_subspace,
    subspace_hamiltonian,
)

SQRT2_INV = 1.0 / np.sqrt(2.0)

# Per-qubit basis-rotation matrices that take the Z eigenbasis to the
# requested Pauli eigenbasis (column k = |basis, k⟩ expressed in the Z
# basis). These are the oracle rotations against which the mixed-basis
# subspace matrices are validated.
_BASIS_ROTATIONS: dict[qm_o.Pauli, np.ndarray] = {
    qm_o.Pauli.X: np.array(
        [[SQRT2_INV, SQRT2_INV], [SQRT2_INV, -SQRT2_INV]],
        dtype=np.complex128,
    ),
    qm_o.Pauli.Y: np.array(
        [[SQRT2_INV, SQRT2_INV], [1j * SQRT2_INV, -1j * SQRT2_INV]],
        dtype=np.complex128,
    ),
    qm_o.Pauli.Z: np.eye(2, dtype=np.complex128),
}


def _all_bitstrings(num_qubits: int) -> list[tuple[int, ...]]:
    """Enumerate every length-``num_qubits`` bitstring in qubit-0=LSB order."""
    samples = []
    for idx in range(1 << num_qubits):
        samples.append(tuple((idx >> q) & 1 for q in range(num_qubits)))
    return samples


def _random_hermitian_hamiltonian(
    num_qubits: int, *, seed: int = 0
) -> qm_o.Hamiltonian:
    """Build a random dense Hermitian and decompose it into Pauli terms."""
    rng = np.random.default_rng(seed)
    dim = 1 << num_qubits
    a = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    a = 0.5 * (a + a.conj().T)
    return HermitianMatrix(a).to_hamiltonian(tol=1e-12)


def _basis_rotation_matrix(bases: list[qm_o.Pauli]) -> np.ndarray:
    """Tensor product of per-qubit Z→basis rotations, qubit 0 = LSB."""
    matrix = np.array([[1.0 + 0.0j]], dtype=np.complex128)
    for basis in reversed(bases):
        matrix = np.kron(matrix, _BASIS_ROTATIONS[basis])
    return matrix


# -- Z-basis path -----------------------------------------------------


@pytest.mark.parametrize("num_qubits", [1, 2, 3, 4])
def test_z_basis_full_hilbert_round_trip(num_qubits: int) -> None:
    """All-bitstring Z-basis subspace reproduces ``H.to_numpy()``."""
    H = _random_hermitian_hamiltonian(num_qubits, seed=num_qubits)
    samples = _all_bitstrings(num_qubits)
    h_sub = subspace_hamiltonian(samples, H)
    np.testing.assert_allclose(h_sub, H.to_numpy(), atol=1e-10)


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_z_basis_proper_subspace_matches_projection(seed: int) -> None:
    """Subset of bitstrings reproduces ``P^T H_full P``."""
    num_qubits = 3
    H = _random_hermitian_hamiltonian(num_qubits, seed=seed)
    h_full = H.to_numpy()

    rng = np.random.default_rng(seed)
    chosen = rng.choice(1 << num_qubits, size=4, replace=False)
    samples = [tuple((int(idx) >> q) & 1 for q in range(num_qubits)) for idx in chosen]
    h_sub = subspace_hamiltonian(samples, H)
    expected = h_full[np.ix_(chosen, chosen)]
    np.testing.assert_allclose(h_sub, expected, atol=1e-10)


def test_z_basis_constant_only_yields_scaled_identity() -> None:
    """A pure ``c · I`` Hamiltonian projects to ``c · I_n``."""
    n_qubits = 2
    H = qm_o.Hamiltonian.identity(coeff=0.7 + 0.3j, num_qubits=n_qubits)
    samples = [(0, 0), (1, 0), (0, 1)]
    h_sub = subspace_hamiltonian(samples, H)
    np.testing.assert_allclose(
        h_sub, (0.7 + 0.3j) * np.eye(3, dtype=np.complex128), atol=1e-12
    )


def test_z_basis_hermitian() -> None:
    """The projected matrix is Hermitian for a Hermitian Hamiltonian."""
    H = _random_hermitian_hamiltonian(3, seed=11)
    samples = [(0, 0, 0), (1, 0, 1), (1, 1, 0)]
    h_sub = subspace_hamiltonian(samples, H)
    np.testing.assert_allclose(h_sub, h_sub.conj().T, atol=1e-10)


def test_z_basis_empty_samples_returns_empty_matrix() -> None:
    """No samples yields a ``(0, 0)`` matrix without complaint."""
    H = qm_o.X(0)
    h_sub = subspace_hamiltonian([], H)
    assert h_sub.shape == (0, 0)


def test_z_basis_rejects_wrong_width() -> None:
    """Sample shorter than ``num_qubits`` raises ``ValueError``."""
    H = qm_o.Z(0) + qm_o.Z(1)
    with pytest.raises(ValueError):
        subspace_hamiltonian([(0,)], H)


# -- Mixed-basis path -------------------------------------------------


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
def test_all_z_basis_matches_z_fast_path(num_qubits: int) -> None:
    """All-Z bases via the generalized path equals the Z fast path."""
    H = _random_hermitian_hamiltonian(num_qubits, seed=num_qubits + 5)
    samples = _all_bitstrings(num_qubits)
    bases = (qm_o.Pauli.Z,) * num_qubits
    h_sub_general, s_sub = generalized_subspace_matrices(samples, bases, H)
    h_sub_fast = subspace_hamiltonian(samples, H)
    np.testing.assert_allclose(h_sub_general, h_sub_fast, atol=1e-10)
    np.testing.assert_allclose(
        s_sub, np.eye(len(samples), dtype=np.complex128), atol=1e-12
    )


@pytest.mark.parametrize("basis", [qm_o.Pauli.X, qm_o.Pauli.Y, qm_o.Pauli.Z])
@pytest.mark.parametrize("num_qubits", [1, 2, 3])
def test_uniform_basis_matches_rotated_oracle(
    basis: qm_o.Pauli, num_qubits: int
) -> None:
    """Uniform-basis subspace equals ``U^† H_full U`` for the matching ``U``."""
    # Use a deterministic seed (Python's built-in ``hash`` is randomised
    # per interpreter unless ``PYTHONHASHSEED`` is set, which would make
    # the generated Hamiltonian non-reproducible across runs).
    H = _random_hermitian_hamiltonian(
        num_qubits, seed=1000 + 10 * num_qubits + basis.value
    )
    h_full = H.to_numpy()

    samples = _all_bitstrings(num_qubits)
    bases = [basis] * num_qubits
    h_sub, s_sub = generalized_subspace_matrices(samples, bases, H)

    u = _basis_rotation_matrix(bases)
    expected_h = u.conj().T @ h_full @ u
    np.testing.assert_allclose(h_sub, expected_h, atol=1e-9)
    np.testing.assert_allclose(
        s_sub, np.eye(1 << num_qubits, dtype=np.complex128), atol=1e-10
    )


def test_y_basis_eigenstates_diagonalize_y() -> None:
    """``|+i⟩`` is the +1 eigenstate, ``|-i⟩`` the -1 eigenstate of Y."""
    H = qm_o.Y(0)
    samples = [(0,), (1,)]
    h_sub, s_sub = generalized_subspace_matrices(samples, (qm_o.Pauli.Y,), H)
    np.testing.assert_allclose(h_sub, np.diag([1.0 + 0.0j, -1.0 + 0.0j]), atol=1e-12)
    np.testing.assert_allclose(s_sub, np.eye(2), atol=1e-12)


def test_cross_basis_overlap_matches_inner_products() -> None:
    """``⟨0|+⟩ = 1/√2`` and friends fall out of ``S_sub``."""
    H = qm_o.Hamiltonian.identity(coeff=1.0, num_qubits=1)
    samples = [(0,), (0,), (1,), (1,)]
    bases = [qm_o.Pauli.Z, qm_o.Pauli.X, qm_o.Pauli.Z, qm_o.Pauli.X]
    _, s_sub = generalized_subspace_matrices(
        samples,
        np.asarray([[b.value] for b in bases], dtype=np.int8),
        H,
    )
    expected = np.array(
        [
            [1.0, SQRT2_INV, 0.0, SQRT2_INV],
            [SQRT2_INV, 1.0, SQRT2_INV, 0.0],
            [0.0, SQRT2_INV, 1.0, -SQRT2_INV],
            [SQRT2_INV, 0.0, -SQRT2_INV, 1.0],
        ],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(s_sub, expected, atol=1e-12)


def test_per_sample_bases_distinct_per_row() -> None:
    """Each sample can carry its own per-qubit basis tuple."""
    H = qm_o.Z(0) + qm_o.X(1)
    h_full = H.to_numpy()

    z = qm_o.Pauli.Z
    x = qm_o.Pauli.X
    samples = [(0, 0), (1, 0)]
    bases = [[z, x], [z, x]]
    h_sub, s_sub = generalized_subspace_matrices(samples, bases, H)

    u = _basis_rotation_matrix([z, x])
    rotated = u.conj().T @ h_full @ u
    expected_idx = [0, 1]
    np.testing.assert_allclose(
        h_sub, rotated[np.ix_(expected_idx, expected_idx)], atol=1e-10
    )
    np.testing.assert_allclose(s_sub, np.eye(2, dtype=np.complex128), atol=1e-12)


def test_generalized_hermitian() -> None:
    """``H_sub`` and ``S_sub`` are both Hermitian for Hermitian inputs."""
    H = _random_hermitian_hamiltonian(2, seed=99)
    samples = [(0, 0), (1, 1), (0, 1)]
    bases = [qm_o.Pauli.X, qm_o.Pauli.Y]
    h_sub, s_sub = generalized_subspace_matrices(samples, bases, H)
    np.testing.assert_allclose(h_sub, h_sub.conj().T, atol=1e-10)
    np.testing.assert_allclose(s_sub, s_sub.conj().T, atol=1e-10)


def test_generalized_rejects_pauli_identity() -> None:
    """``Pauli.I`` is not a valid measurement basis."""
    H = qm_o.X(0)
    with pytest.raises(ValueError):
        generalized_subspace_matrices([(0,)], (qm_o.Pauli.I,), H)


def test_generalized_rejects_bases_shape_mismatch() -> None:
    """Per-sample ``bases`` must match the sample count."""
    H = qm_o.X(0)
    with pytest.raises(ValueError):
        generalized_subspace_matrices(
            [(0,), (1,)],
            [(qm_o.Pauli.X,)],  # only one row, but two samples
            H,
        )


def test_generalized_empty_samples() -> None:
    """No samples yields zero-sized ``H_sub`` and ``S_sub``."""
    H = qm_o.X(0)
    h_sub, s_sub = generalized_subspace_matrices([], (qm_o.Pauli.X,), H)
    assert h_sub.shape == (0, 0)
    assert s_sub.shape == (0, 0)


# -- solve_subspace ---------------------------------------------------


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
def test_solve_z_basis_full_eigenvalues(num_qubits: int) -> None:
    """Z-basis solver over the full Hilbert space matches ``eigvalsh``."""
    H = _random_hermitian_hamiltonian(num_qubits, seed=num_qubits + 100)
    samples = _all_bitstrings(num_qubits)
    eigvals, _ = solve_subspace(samples, H)
    expected = np.linalg.eigvalsh(H.to_numpy())
    np.testing.assert_allclose(eigvals, expected, atol=1e-10)


def test_solve_z_basis_removes_duplicate_sample_null_space() -> None:
    """Repeated Z-basis samples cannot create non-variational eigenvalues."""
    eigvals, eigvecs = solve_subspace([(0,), (0,), (1,)], qm_o.Z(0))

    np.testing.assert_allclose(eigvals, [-1.0, 1.0], atol=1e-12)
    assert eigvecs.shape == (3, 2)


def test_solve_x_basis_recovers_minus_x_ground_state() -> None:
    """Sampling ``|+⟩`` of ``-X`` recovers ``-1`` ground-state energy."""
    H = -qm_o.X(0)
    eigvals, _ = solve_subspace([(0,)], H, bases=(qm_o.Pauli.X,))
    assert eigvals.shape == (1,)
    np.testing.assert_allclose(eigvals[0], -1.0, atol=1e-12)


def test_solve_x_basis_full_subspace_matches_eigvalsh() -> None:
    """All ``|±⟩^⊗n`` recover every eigenvalue of the rotated Hamiltonian."""
    num_qubits = 2
    H = _random_hermitian_hamiltonian(num_qubits, seed=27)
    samples = _all_bitstrings(num_qubits)
    bases = (qm_o.Pauli.X,) * num_qubits
    eigvals, _ = solve_subspace(samples, H, bases=bases)
    expected = np.linalg.eigvalsh(H.to_numpy())
    np.testing.assert_allclose(eigvals, expected, atol=1e-9)


def test_solve_rank_deficient_overlap_fallback() -> None:
    """Duplicate samples make ``S_sub`` singular; fallback returns the rank."""
    H = qm_o.Z(0)
    samples = [(0,), (0,), (1,)]
    eigvals, eigvecs = solve_subspace(
        samples, H, bases=(qm_o.Pauli.Z,), overlap_tol=1e-10
    )
    # Effective rank is 2 (only two distinct Z bitstrings).
    assert eigvals.shape == (2,)
    assert eigvecs.shape[0] == 3
    # Eigenvalues are ±1 for ``Z``.
    np.testing.assert_allclose(np.sort(eigvals), [-1.0, 1.0], atol=1e-10)


def test_solve_rejects_negative_overlap_tol() -> None:
    """``overlap_tol`` must be non-negative."""
    H = qm_o.X(0)
    with pytest.raises(ValueError):
        solve_subspace([(0,)], H, bases=(qm_o.Pauli.X,), overlap_tol=-1e-12)


def test_solve_raises_when_overlap_tol_kills_all_directions() -> None:
    """If ``overlap_tol`` is so large that every overlap eigenvalue is
    discarded, ``solve_subspace`` must raise rather than silently
    return an empty eigenpair tuple."""
    H = qm_o.X(0)
    # ``overlap_tol`` larger than any eigenvalue of S (which is at most 1
    # since S is a Gram matrix of unit vectors) forces ``keep`` all-False.
    with pytest.raises(ValueError, match="overlap_tol"):
        solve_subspace([(0,)], H, bases=(qm_o.Pauli.X,), overlap_tol=10.0)


def test_solve_empty_samples() -> None:
    """No samples returns empty eigenpair arrays."""
    H = qm_o.X(0)
    eigvals, eigvecs = solve_subspace([], H)
    assert eigvals.shape == (0,)
    assert eigvecs.shape == (0, 0)


# -- Mixed per-sample basis GEVP --------------------------------------


def _basis_state_in_z(basis: qm_o.Pauli, bit: int) -> np.ndarray:
    """Return the column-vector form of ``|basis, bit⟩`` in the Z basis."""
    return _BASIS_ROTATIONS[basis][:, bit].astype(np.complex128)


def _product_state_in_z(
    bits: tuple[int, ...], bases: tuple[qm_o.Pauli, ...]
) -> np.ndarray:
    """Build ``⊗_q |basis_q, bit_q⟩`` as a Z-basis state vector.

    Qubit 0 is the least-significant axis to stay aligned with
    :meth:`qamomile.observable.Hamiltonian.to_numpy`.
    """
    state = np.array([1.0 + 0.0j], dtype=np.complex128)
    for q, b in reversed(list(zip(bits, bases))):
        state = np.kron(state, _basis_state_in_z(b, q))
    return state


def _build_oracle_matrices(
    samples: list[tuple[int, ...]],
    bases: list[tuple[qm_o.Pauli, ...]],
    H: qm_o.Hamiltonian,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation: build ``H_sub`` and ``S_sub`` directly
    from full Z-basis state vectors. Used as an oracle to validate the
    table-based vectorised path.
    """
    h_full = H.to_numpy()
    n = len(samples)
    h_sub = np.zeros((n, n), dtype=np.complex128)
    s_sub = np.zeros((n, n), dtype=np.complex128)
    states = [_product_state_in_z(s, b) for s, b in zip(samples, bases)]
    for i, psi_i in enumerate(states):
        for j, psi_j in enumerate(states):
            h_sub[i, j] = np.vdot(psi_i, h_full @ psi_j)
            s_sub[i, j] = np.vdot(psi_i, psi_j)
    return h_sub, s_sub


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_generalized_per_sample_mixed_xyz_matches_full_state_oracle(
    seed: int,
) -> None:
    """Per-row mixed X/Y/Z bases match the brute-force state-vector oracle."""
    num_qubits = 2
    H = _random_hermitian_hamiltonian(num_qubits, seed=seed)

    rng = np.random.default_rng(seed)
    pauli_pool = (qm_o.Pauli.X, qm_o.Pauli.Y, qm_o.Pauli.Z)
    n_samples = 6
    samples = [
        tuple(int(b) for b in rng.integers(0, 2, size=num_qubits))
        for _ in range(n_samples)
    ]
    bases = [
        tuple(pauli_pool[int(b)] for b in rng.integers(0, 3, size=num_qubits))
        for _ in range(n_samples)
    ]

    h_sub, s_sub = generalized_subspace_matrices(samples, bases, H)
    h_oracle, s_oracle = _build_oracle_matrices(samples, bases, H)
    np.testing.assert_allclose(h_sub, h_oracle, atol=1e-10)
    np.testing.assert_allclose(s_sub, s_oracle, atol=1e-10)


def test_solve_mixed_basis_full_span_recovers_full_spectrum() -> None:
    """Pooling all X^n + Y^n + Z^n bitstrings spans the Hilbert space.

    ``S_sub`` is rank ``2**n`` inside a ``3 * 2**n`` ambient space; the
    overlap-tol fallback should drop the redundant directions and
    return exactly the full spectrum of ``H``.
    """
    num_qubits = 2
    H = _random_hermitian_hamiltonian(num_qubits, seed=2024)
    bitstrings = _all_bitstrings(num_qubits)

    samples: list[tuple[int, ...]] = []
    bases: list[tuple[qm_o.Pauli, ...]] = []
    for basis in (qm_o.Pauli.X, qm_o.Pauli.Y, qm_o.Pauli.Z):
        for bits in bitstrings:
            samples.append(bits)
            bases.append((basis,) * num_qubits)

    eigvals, _ = solve_subspace(samples, H, bases=bases, overlap_tol=1e-9)
    expected = np.linalg.eigvalsh(H.to_numpy())
    assert eigvals.shape == (1 << num_qubits,)
    np.testing.assert_allclose(eigvals, expected, atol=1e-8)


def test_solve_mixed_basis_one_per_basis_two_qubit_full_spectrum() -> None:
    """Three carefully chosen mixed-basis samples per qubit also span 2 qubits."""
    num_qubits = 2
    H = _random_hermitian_hamiltonian(num_qubits, seed=314)

    # 3 X-basis + 3 Y-basis + 3 Z-basis distinct bitstrings = 9 samples.
    # The state set spans the 4-D Hilbert space with redundancy.
    pick = [(0, 0), (1, 0), (0, 1)]
    samples: list[tuple[int, ...]] = []
    bases: list[tuple[qm_o.Pauli, ...]] = []
    for basis in (qm_o.Pauli.X, qm_o.Pauli.Y, qm_o.Pauli.Z):
        for bits in pick:
            samples.append(bits)
            bases.append((basis,) * num_qubits)

    eigvals, _ = solve_subspace(samples, H, bases=bases, overlap_tol=1e-9)
    expected = np.linalg.eigvalsh(H.to_numpy())
    assert eigvals.shape == (1 << num_qubits,)
    np.testing.assert_allclose(eigvals, expected, atol=1e-8)


def test_solve_mixed_basis_qubit_local_bases_differ() -> None:
    """Different qubits can be measured in different Pauli bases per sample."""
    num_qubits = 2
    H = _random_hermitian_hamiltonian(num_qubits, seed=99)

    # Each sample uses a different basis on qubit 0 vs qubit 1.
    samples = [
        (0, 0),  # |+⟩ ⊗ |0⟩ in (X, Z)
        (1, 0),  # |-⟩ ⊗ |0⟩ in (X, Z)
        (0, 0),  # |+⟩ ⊗ |+i⟩ in (X, Y)
        (1, 1),  # |-⟩ ⊗ |-i⟩ in (X, Y)
        (0, 0),  # |0⟩ ⊗ |+⟩ in (Z, X)
        (1, 1),  # |1⟩ ⊗ |-⟩ in (Z, X)
    ]
    bases = [
        (qm_o.Pauli.X, qm_o.Pauli.Z),
        (qm_o.Pauli.X, qm_o.Pauli.Z),
        (qm_o.Pauli.X, qm_o.Pauli.Y),
        (qm_o.Pauli.X, qm_o.Pauli.Y),
        (qm_o.Pauli.Z, qm_o.Pauli.X),
        (qm_o.Pauli.Z, qm_o.Pauli.X),
    ]

    h_sub, s_sub = generalized_subspace_matrices(samples, bases, H)
    h_oracle, s_oracle = _build_oracle_matrices(samples, bases, H)
    np.testing.assert_allclose(h_sub, h_oracle, atol=1e-10)
    np.testing.assert_allclose(s_sub, s_oracle, atol=1e-10)

    eigvals, _ = solve_subspace(samples, H, bases=bases, overlap_tol=1e-9)
    expected = np.linalg.eigvalsh(H.to_numpy())
    assert eigvals.shape == (1 << num_qubits,)
    np.testing.assert_allclose(eigvals, expected, atol=1e-8)


def test_solve_mixed_basis_recovers_known_ground_state() -> None:
    """Sampling the |+⟩|0⟩ eigenstate of -X-Z gives the -2 ground energy."""
    H = -qm_o.X(0) - qm_o.Z(1)

    samples = [
        (0, 0),  # |+⟩ ⊗ |0⟩  — exact ground state
        (1, 0),  # |-⟩ ⊗ |0⟩
        (0, 1),  # |+⟩ ⊗ |1⟩
        (0, 0),  # |0⟩ ⊗ |0⟩  — Z-basis sample for span
    ]
    bases = [
        (qm_o.Pauli.X, qm_o.Pauli.Z),
        (qm_o.Pauli.X, qm_o.Pauli.Z),
        (qm_o.Pauli.X, qm_o.Pauli.Z),
        (qm_o.Pauli.Z, qm_o.Pauli.Z),
    ]
    eigvals, _ = solve_subspace(samples, H, bases=bases, overlap_tol=1e-10)
    assert eigvals.shape[0] >= 1
    np.testing.assert_allclose(eigvals[0], -2.0, atol=1e-9)


def test_generalized_accepts_int_ndarray_bases() -> None:
    """Integer ndarray of basis codes is accepted via the fast path."""
    num_qubits = 2
    H = _random_hermitian_hamiltonian(num_qubits, seed=5)
    samples = [(0, 1), (1, 0)]
    bases_codes = np.asarray(
        [
            [qm_o.Pauli.X.value, qm_o.Pauli.Y.value],
            [qm_o.Pauli.Z.value, qm_o.Pauli.X.value],
        ],
        dtype=np.int8,
    )
    bases_pauli = [
        (qm_o.Pauli.X, qm_o.Pauli.Y),
        (qm_o.Pauli.Z, qm_o.Pauli.X),
    ]
    h_codes, s_codes = generalized_subspace_matrices(samples, bases_codes, H)
    h_pauli, s_pauli = generalized_subspace_matrices(samples, bases_pauli, H)
    np.testing.assert_allclose(h_codes, h_pauli, atol=1e-12)
    np.testing.assert_allclose(s_codes, s_pauli, atol=1e-12)
