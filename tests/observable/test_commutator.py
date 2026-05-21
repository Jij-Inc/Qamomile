"""Tests for ``qamomile.observable.hamiltonian.commutator``.

Verifies known Pauli-algebra identities, antisymmetry, the zero
commutator for terms that share no anticommuting qubits, and
equivalence with the naive ``a * b - b * a`` expansion over random
multi-term Hamiltonians.
"""

import itertools

import numpy as np
import pytest

from qamomile.observable.hamiltonian import (
    Hamiltonian,
    Pauli,
    PauliOperator,
    X,
    Y,
    Z,
    _pauli_strings_anticommute,
    commutator,
)


def _naive_commutator(a: Hamiltonian, b: Hamiltonian) -> Hamiltonian:
    """Reference implementation: compute ``a * b - b * a`` directly.

    Args:
        a (Hamiltonian): The left operand of the commutator.
        b (Hamiltonian): The right operand of the commutator.

    Returns:
        Hamiltonian: The naively expanded commutator, used as the
            ground truth that ``commutator(a, b)`` is checked against.
    """
    return a * b - b * a


def _hamiltonians_close(a: Hamiltonian, b: Hamiltonian, *, atol: float = 1e-10) -> bool:
    """Compare two Hamiltonians up to a numerical tolerance on coefficients.

    Args:
        a (Hamiltonian): The left-hand Hamiltonian to compare.
        b (Hamiltonian): The right-hand Hamiltonian to compare.
        atol (float): Absolute tolerance forwarded to ``np.isclose``
            for both the constant term and every Pauli-string
            coefficient. Defaults to 1e-10.

    Returns:
        bool: True when the constant terms and all term coefficients
            (over the union of both ``terms`` dicts) agree within
            ``atol``; False otherwise.
    """
    if not np.isclose(a.constant, b.constant, atol=atol):
        return False
    keys = set(a.terms) | set(b.terms)
    for key in keys:
        ca = a.terms.get(key, 0.0)
        cb = b.terms.get(key, 0.0)
        if not np.isclose(ca, cb, atol=atol):
            return False
    return True


def _random_hamiltonian(
    rng: np.random.Generator,
    num_qubits: int,
    num_terms: int,
    *,
    include_constant: bool = True,
) -> Hamiltonian:
    """Build a random Hamiltonian for property-based equivalence tests.

    Each term draws an independent Pauli from {X, Y, Z, I} per qubit
    (identities are dropped) and a complex Gaussian coefficient.

    Args:
        rng (np.random.Generator): Seeded RNG used for every random
            choice so the helper stays deterministic per test.
        num_qubits (int): Number of qubits the Hamiltonian acts on.
            Sets both the per-term Pauli string length and the
            ``Hamiltonian._num_qubits`` floor.
        num_terms (int): Number of Pauli-string terms to add. Terms
            that collapse to identity (all-I draws) are still passed
            through ``add_term`` and routed to ``constant``.
        include_constant (bool): When True, also assigns a random
            complex constant to the Hamiltonian. Defaults to True.

    Returns:
        Hamiltonian: A random Hamiltonian instance suitable for
            comparing ``commutator(a, b)`` against the naive
            ``a * b - b * a`` expansion.
    """
    h = Hamiltonian(num_qubits=num_qubits)
    paulis = (Pauli.X, Pauli.Y, Pauli.Z, Pauli.I)
    for _ in range(num_terms):
        ops = []
        for q in range(num_qubits):
            p = paulis[rng.integers(0, 4)]
            if p != Pauli.I:
                ops.append(PauliOperator(p, q))
        coeff = complex(rng.normal(), rng.normal())
        h.add_term(tuple(ops), coeff)
    if include_constant:
        h.constant = complex(rng.normal(), rng.normal())
    return h


class TestPauliAlgebraIdentities:
    """Single-qubit Pauli commutators must match the textbook values."""

    def test_xy_gives_2iz(self):
        """[X, Y] = 2i Z."""
        assert commutator(X(0), Y(0)) == 2j * Z(0)

    def test_yz_gives_2ix(self):
        """[Y, Z] = 2i X."""
        assert commutator(Y(0), Z(0)) == 2j * X(0)

    def test_zx_gives_2iy(self):
        """[Z, X] = 2i Y."""
        assert commutator(Z(0), X(0)) == 2j * Y(0)

    def test_self_commutator_is_zero(self):
        """[P, P] = 0 for any Pauli P."""
        for factory in (X, Y, Z):
            result = commutator(factory(0), factory(0))
            assert len(result) == 0
            assert result.constant == 0.0


class TestCommutingTermsVanish:
    """Terms whose Pauli strings commute must contribute nothing."""

    def test_disjoint_qubits_commute(self):
        """Operators on disjoint qubits commute, so [X0, X1] = 0."""
        result = commutator(X(0), X(1))
        assert len(result) == 0
        assert result.constant == 0.0

    def test_same_pauli_different_qubits(self):
        """[Z0 Z1, Z1 Z2] = 0 because every shared qubit carries the same Pauli."""
        a = Z(0) * Z(1)
        b = Z(1) * Z(2)
        result = commutator(a, b)
        assert len(result) == 0
        assert result.constant == 0.0

    def test_two_anticommuting_qubits_commute(self):
        """X0 X1 and Y0 Y1 anticommute twice, so they commute overall."""
        a = X(0) * X(1)
        b = Y(0) * Y(1)
        assert not _pauli_strings_anticommute(
            tuple(next(iter(a.terms))),
            tuple(next(iter(b.terms))),
        )
        result = commutator(a, b)
        assert len(result) == 0
        assert result.constant == 0.0


class TestAlgebraicProperties:
    """Bilinearity and antisymmetry hold for the commutator."""

    def test_antisymmetry(self):
        """[A, B] = -[B, A]."""
        rng = np.random.default_rng(0)
        a = _random_hamiltonian(rng, num_qubits=3, num_terms=4)
        b = _random_hamiltonian(rng, num_qubits=3, num_terms=4)
        ab = commutator(a, b)
        ba = commutator(b, a)
        assert _hamiltonians_close(ab, -1.0 * ba)

    def test_constant_part_drops_out(self):
        """Adding a constant to either operand leaves the commutator unchanged."""
        a = X(0) + 1.5 * Y(1)
        b = Z(0) * X(1) + 0.5
        without_const = commutator(a, b - 0.5)
        with_const = commutator(a, b)
        assert _hamiltonians_close(without_const, with_const)

    def test_linearity_in_first_argument(self):
        """[αA + βC, B] = α[A, B] + β[C, B]."""
        rng = np.random.default_rng(1)
        a = _random_hamiltonian(rng, num_qubits=2, num_terms=3, include_constant=False)
        c = _random_hamiltonian(rng, num_qubits=2, num_terms=3, include_constant=False)
        b = _random_hamiltonian(rng, num_qubits=2, num_terms=3, include_constant=False)
        alpha, beta = 0.7 + 0.2j, -1.3
        lhs = commutator(alpha * a + beta * c, b)
        rhs = alpha * commutator(a, b) + beta * commutator(c, b)
        assert _hamiltonians_close(lhs, rhs)


class TestEquivalenceWithNaive:
    """The optimized commutator must agree with ``a * b - b * a``."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 42, 1337])
    @pytest.mark.parametrize("num_qubits", [1, 2, 3, 4])
    def test_random_hamiltonians_match_naive(self, seed: int, num_qubits: int):
        """commutator(a, b) == a * b - b * a for randomized inputs."""
        rng = np.random.default_rng(seed)
        a = _random_hamiltonian(rng, num_qubits=num_qubits, num_terms=5)
        b = _random_hamiltonian(rng, num_qubits=num_qubits, num_terms=5)
        assert _hamiltonians_close(commutator(a, b), _naive_commutator(a, b))

    def test_matches_naive_for_known_anticommuting_pair(self):
        """[XY, YX] worked example matches naive expansion exactly."""
        a = X(0) * Y(1)
        b = Y(0) * X(1)
        assert _hamiltonians_close(commutator(a, b), _naive_commutator(a, b))


class TestNumQubitsPropagation:
    """The result must preserve the qubit register from its operands."""

    def test_max_of_two_operands(self):
        """num_qubits should not shrink even if the commutator is sparse."""
        a = X(0) + X(4)
        b = Y(0) + Y(4)
        result = commutator(a, b)
        assert result.num_qubits >= max(a.num_qubits, b.num_qubits)


class TestPauliAnticommuteHelper:
    """Direct coverage for the qubit-parity anticommutation predicate."""

    @pytest.mark.parametrize(
        "p1,p2,expected",
        list(
            itertools.chain.from_iterable(
                [
                    [(Pauli.X, Pauli.Y, True), (Pauli.X, Pauli.Z, True)],
                    [(Pauli.Y, Pauli.Z, True), (Pauli.X, Pauli.X, False)],
                    [(Pauli.X, Pauli.I, False), (Pauli.I, Pauli.Y, False)],
                ]
            )
        ),
    )
    def test_single_qubit_table(self, p1: Pauli, p2: Pauli, expected: bool):
        """Single-qubit Pauli pairs follow the standard anticommutation table."""
        t1 = (PauliOperator(p1, 0),)
        t2 = (PauliOperator(p2, 0),)
        assert _pauli_strings_anticommute(t1, t2) is expected
