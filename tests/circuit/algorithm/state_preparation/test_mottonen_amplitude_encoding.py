"""Tests for Möttönen amplitude encoding."""

from __future__ import annotations

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.state_preparation import (
    MottonenAmplitudeEncoding,
    amplitude_encoding,
    amplitude_encoding_from_angles,
)
from qamomile.circuit.ir.operation import InvokeOperation
from qamomile.circuit.ir.operation.callable import CompositeGateType
from qamomile.circuit.ir.operation.gate import GateOperation
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.value import resolve_root_qubit_address
from qamomile.circuit.transpiler.passes.emit_support import ResourceAllocator
from qamomile.circuit.transpiler.segments import QuantumStep
from qamomile.linalg import (
    compute_mottonen_amplitude_encoding_ry_angles,
    compute_mottonen_amplitude_encoding_rz_angles,
)
from tests.circuit.conftest import run_statevector

# Catalogue of REAL amplitudes shared by every backend test.  Each entry is
# a list of (id, amplitudes) — keeping it module-level lets all
# parametrisations stay aligned.
_FIXED_AMPLITUDES: list[tuple[str, list[float]]] = [
    ("1q-basis-0", [1.0, 0.0]),
    ("1q-basis-1", [0.0, 1.0]),
    ("1q-plus", [1.0, 1.0]),
    ("1q-negative", [1.0, -1.0]),
    ("1q-normalization", [3.0, 4.0]),
    ("2q-uniform", [1.0, 1.0, 1.0, 1.0]),
    ("2q-bell-like", [1.0, 0.0, 0.0, 1.0]),
    ("2q-asymmetric", [1.0, 2.0, 3.0, 4.0]),
    ("2q-signed", [1.0, -1.0, 1.0, -1.0]),
    ("3q-signed", [1.0, 2.0, 3.0, -4.0, 5.0, -6.0, 7.0, 8.0]),
]

# Catalogue of COMPLEX amplitudes — exercises the magnitude + phase
# decomposition (iterative disentangling).  Includes pure-phase, mixed,
# and sparse complex cases.
_FIXED_COMPLEX_AMPLITUDES: list[tuple[str, list[complex]]] = [
    ("1q-imag", [1.0 + 0j, 0.0 + 1j]),
    ("1q-mixed", [1.0 + 1j, 1.0 + 0j]),
    ("2q-phases", [1.0 + 0j, 0.0 + 1j, -1.0 + 0j, 0.0 - 1j]),
    ("2q-mixed", [1.0 + 0j, 1.0 + 1j, 1.0 - 1j, 0.0 + 2j]),
    ("2q-sparse-complex", [1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 1j]),
    ("3q-mixed", [1 + 0j, 1j, -1 + 0j, -1j, 1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]),
]

_SEEDS = [901, 902, 903, 904, 905]
_QUBIT_COUNTS = [1, 2, 3, 4]
_WRAPPER_QUBIT_COUNTS = [1, 2, 3]

# Sampling tolerance: 5 sigma binomial bound at the test shot count, matching
# the convention used elsewhere in the suite (e.g. test_trotter.py's
# CrossBackendDistribution).
_SHOTS = 8192
_STD_TOLERANCE = 5.0

# Hand-derived expectation values for the encoding of [1, 2, 3, 4] in a
# 2-qubit register with little-endian ordering (q0 = LSB).
#
# Normalised state:   a = [1, 2, 3, 4] / sqrt(30)
# Probabilities:      p_i = a_i^2 = [1, 4, 9, 16] / 30
#
#   <Z_0> = p_0 + p_2 - p_1 - p_3 = (1 + 9 - 4 - 16) / 30 = -1/3
#   <Z_1> = p_0 + p_1 - p_2 - p_3 = (1 + 4 - 9 - 16) / 30 = -2/3
#
# The signed encoding [1, -1, 1, -1] / 2 gives:
#   <Z_0> = (1/4 + 1/4) - (1/4 + 1/4) = 0
#   <X_0> on [a_0, a_1] = 2 * a_0 * a_1 (per-pair off-diagonal): the four
#     amplitudes split into pairs (a_0,a_1)=(1/2,-1/2) and (a_2,a_3)=(1/2,-1/2),
#     so <X_0> = 2*(1/2)*(-1/2) + 2*(1/2)*(-1/2) = -1.
_EXPVAL_CASES: list[tuple[str, list[float] | list[complex], str, int, float]] = [
    # (id,        amplitudes,                 pauli, qubit_index, expected)
    # ----- Real amplitudes -----
    ("asymm-Z0", [1.0, 2.0, 3.0, 4.0], "Z", 0, -1.0 / 3.0),
    ("asymm-Z1", [1.0, 2.0, 3.0, 4.0], "Z", 1, -2.0 / 3.0),
    ("signed-Z0", [1.0, -1.0, 1.0, -1.0], "Z", 0, 0.0),
    ("signed-X0", [1.0, -1.0, 1.0, -1.0], "X", 0, -1.0),
    # ----- Complex amplitudes -----
    # |+y> = (|0> + i|1>)/sqrt(2): pure Y eigenstate with eigenvalue +1.
    ("plus-y-Y0", [1.0 + 0j, 0.0 + 1j], "Y", 0, 1.0),
    # Same state, transverse: <Z_0> = <X_0> = 0.
    ("plus-y-Z0", [1.0 + 0j, 0.0 + 1j], "Z", 0, 0.0),
    # 2-qubit complex amps a = [1, 1+1j, 1-1j, 2j] / 3, probs [1, 2, 2, 4] / 9.
    # <Z_0> = p_0 + p_2 - p_1 - p_3 = (1 + 2 - 2 - 4) / 9 = -1/3.
    # <Z_1> = p_0 + p_1 - p_2 - p_3 = (1 + 2 - 2 - 4) / 9 = -1/3.
    ("2q-mixed-Z0", [1 + 0j, 1 + 1j, 1 - 1j, 0 + 2j], "Z", 0, -1.0 / 3.0),
    ("2q-mixed-Z1", [1 + 0j, 1 + 1j, 1 - 1j, 0 + 2j], "Z", 1, -1.0 / 3.0),
]


def _random_real_amplitudes(n_qubits: int, seed: int) -> list[float]:
    """Return a deterministic non-zero random real amplitude vector.

    Args:
        n_qubits (int): Number of qubits represented by the vector.
        seed (int): Seed for ``np.random.default_rng``.

    Returns:
        list[float]: Real amplitude vector of length ``2**n_qubits``.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal(2**n_qubits).tolist()


_WRAPPER_AMPLITUDE_CASES: list[pytest.ParameterSet] = [
    pytest.param([1.0, 0.0], id="boundary-1q-zero"),
    pytest.param([1.0, 2.0, 3.0, 4.0], id="fixed-2q-asymmetric"),
    pytest.param([1.0 + 0j, 0.0 + 1j], id="complex-1q-plus-y"),
    pytest.param(
        [1.0 + 0j, 1.0 + 1j, 1.0 - 1j, 0.0 + 2j],
        id="complex-2q-mixed",
    ),
    pytest.param(
        [1.0, 2.0, 3.0, -4.0, 5.0, -6.0, 7.0, 8.0],
        id="fixed-3q-signed",
    ),
    *[
        pytest.param(
            _random_real_amplitudes(n_qubits, seed),
            id=f"random-{n_qubits}q-seed{seed}",
        )
        for n_qubits in _WRAPPER_QUBIT_COUNTS
        for seed in [0, 1, 2, 42]
    ],
]


def _normalize(amps: list[float] | list[complex]) -> np.ndarray:
    """Return *amps* as a unit-norm array (real or complex per input dtype).

    Args:
        amps (list[float] | list[complex]): Amplitudes (any non-zero norm).

    Returns:
        np.ndarray: Array with unit Euclidean norm.  Complex dtype if any
            element is complex, else float.
    """
    if any(isinstance(x, complex) for x in amps):
        arr = np.asarray(amps, dtype=complex)
    else:
        arr = np.asarray(amps, dtype=float)
    return arr / np.linalg.norm(arr)


def _state_fidelity(got: np.ndarray, expected: np.ndarray) -> float:
    """Phase-invariant fidelity ``|<got|expected>|^2`` between two state vectors.

    Args:
        got (np.ndarray): Statevector returned by the simulator.
        expected (np.ndarray): Reference statevector to compare against.

    Returns:
        float: ``|<got|expected>|^2`` as a Python ``float``.
    """
    return float(np.abs(np.vdot(got, expected)) ** 2)


def _pad_observable(num_qubits: int, term: qm_o.Hamiltonian) -> qm_o.Hamiltonian:
    """Pad a single-qubit Pauli to *num_qubits* width with a zero-weighted Z term.

    Several backend emit paths require the ``Hamiltonian.num_qubits``
    to match the register width.  Adding ``0.0 * qm_o.Z(num_qubits - 1)``
    extends the declared qubit count to ``num_qubits`` without affecting
    any expectation value (the weight is exactly zero).  This is the
    standard trick used elsewhere in the test suite.

    Args:
        num_qubits (int): Target register width.
        term (qm_o.Hamiltonian): Source Hamiltonian whose width is at
            most *num_qubits*.

    Returns:
        qm_o.Hamiltonian: Hamiltonian equal to *term* but reporting
            ``num_qubits == num_qubits``.  If *term* is already wide
            enough, it is returned unchanged.
    """
    if term.num_qubits >= num_qubits:
        return term
    padded = term + 0.0 * qm_o.Z(num_qubits - 1)
    assert padded.num_qubits == num_qubits
    return padded


def _make_pauli(pauli: str, qubit: int) -> qm_o.Hamiltonian:
    """Build a single-qubit Pauli observable from a label / qubit pair.

    Args:
        pauli (str): One of ``"X"``, ``"Y"``, ``"Z"``.
        qubit (int): Index of the qubit the Pauli acts on.

    Returns:
        qm_o.Hamiltonian: A single-term Hamiltonian.

    Raises:
        ValueError: If *pauli* is not one of the three accepted labels.
    """
    match pauli:
        case "Z":
            return qm_o.Z(qubit)
        case "X":
            return qm_o.X(qubit)
        case "Y":
            return qm_o.Y(qubit)
        case _:
            raise ValueError(f"Unknown Pauli label: {pauli}")


# ---------------------------------------------------------------------------
# Class-level metadata / resource estimation
# ---------------------------------------------------------------------------


class TestCompositeGateMetadata:
    """Smoke tests for the CompositeGate plumbing."""

    def test_gate_identifiers(self) -> None:
        """``gate_type`` and ``custom_name`` match what the IR will tag."""
        gate = MottonenAmplitudeEncoding([1.0, 0.0])
        assert gate.gate_type == CompositeGateType.CUSTOM
        assert gate.custom_name == "mottonen_amplitude_encoding"

    @pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
    def test_resources_real(self, n_qubits: int) -> None:
        """Real input: single-stage RY counts match the Gray-walk formulas."""
        amplitudes = np.ones(2**n_qubits)

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(n_qubits, "q")
            q = amplitude_encoding(q, amplitudes)
            return q

        estimate = kernel.estimate_resources()
        assert estimate.gates.t == 0
        assert estimate.gates.total == (2**n_qubits - 1) + (2**n_qubits - 2)
        assert estimate.gates.single_qubit == 2**n_qubits - 1
        assert estimate.gates.two_qubit == 2**n_qubits - 2

    @pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
    def test_resources_complex(self, n_qubits: int) -> None:
        """Complex input: two-stage RY + RZ counts double the gate budget."""
        amplitudes = np.ones(2**n_qubits, dtype=complex) + 1j * np.arange(2**n_qubits)

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(n_qubits, "q")
            q = amplitude_encoding(q, amplitudes)
            return q

        estimate = kernel.estimate_resources()
        rot = 2 * (2**n_qubits - 1)
        cnot = 2 * (2**n_qubits - 2)
        assert estimate.gates.t == 0
        assert estimate.gates.total == rot + cnot
        assert estimate.gates.single_qubit == rot
        assert estimate.gates.two_qubit == cnot


# ---------------------------------------------------------------------------
# Ry / Rz stage ordering
# ---------------------------------------------------------------------------


def _build_unitary_via_emission(
    n: int,
    ry_per_level: list[np.ndarray],
    rz_per_level: list[np.ndarray],
    mode: str,
) -> np.ndarray:
    """Build the n-qubit unitary that the Ry / Rz Gray-walk emission realises.

    Used by :class:`TestRyRzOrdering` to assert sweep/interleave equivalence.

    Args:
        n (int): Number of qubits.
        ry_per_level (list[np.ndarray]): Gray-walk RY angles, per level.
        rz_per_level (list[np.ndarray]): Gray-walk RZ angles, per level.
        mode (str): ``"sweep"`` (all Ry levels then all Rz levels) or
            ``"interleave"`` (per-level Ry then Rz, repeated).

    Returns:
        np.ndarray: The ``2**n x 2**n`` unitary as a column-by-column
            simulation, obtained by applying the emission to each
            computational basis state.
    """
    from qamomile.circuit.algorithm.state_preparation.mottonen_amplitude_encoding import (
        _get_cnot_controls,
    )

    N = 2**n

    def ry_m(t: float) -> np.ndarray:
        c, s = np.cos(t / 2), np.sin(t / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    def rz_m(t: float) -> np.ndarray:
        return np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)])

    def apply_1q(st: np.ndarray, qb: int, m: np.ndarray) -> np.ndarray:
        new = np.zeros_like(st)
        for idx in range(N):
            b = (idx >> qb) & 1
            other = idx & ~(1 << qb)
            for nb in (0, 1):
                new[other | (nb << qb)] += m[nb, b] * st[idx]
        return new

    def apply_cnot(st: np.ndarray, c: int, t: int) -> np.ndarray:
        new = st.copy()
        for idx in range(N):
            if (idx >> c) & 1:
                new[idx], new[idx ^ (1 << t)] = st[idx ^ (1 << t)], st[idx]
        return new

    def emit_one_level(
        st: np.ndarray, k: int, angles: np.ndarray, gate_m
    ) -> np.ndarray:
        tgt = n - 1 - k
        if k == 0:
            return apply_1q(st, tgt, gate_m(float(angles[0])))
        seq = _get_cnot_controls(k)
        for step in range(2**k):
            st = apply_1q(st, tgt, gate_m(float(angles[step])))
            ctrl = n - 1 - seq[step]
            st = apply_cnot(st, ctrl, tgt)
        return st

    U = np.zeros((N, N), dtype=complex)
    for col in range(N):
        st = np.zeros(N, dtype=complex)
        st[col] = 1.0
        match mode:
            case "sweep":
                for k in range(n):
                    st = emit_one_level(st, k, ry_per_level[k], ry_m)
                for k in range(n):
                    st = emit_one_level(st, k, rz_per_level[k], rz_m)
            case "interleave":
                for k in range(n):
                    st = emit_one_level(st, k, ry_per_level[k], ry_m)
                    st = emit_one_level(st, k, rz_per_level[k], rz_m)
            case _:
                raise ValueError(f"mode must be 'sweep' or 'interleave', got {mode!r}")
        U[:, col] = st
    return U


class TestRyRzOrdering:
    """Verify the structural equivalence of sweep vs interleave emission.

    The implementation emits "all Ry levels then all Rz levels"
    (sweep), not the per-level interleaved order suggested by the
    Möttönen-Vartiainen disentangling derivation.  Pairwise
    ``[U_y^(k), U_z^(k')]`` is not zero in general, but the FULL
    products coincide as unitaries — including for arbitrary angles
    unrelated to any specific amplitude vector.
    """

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_sweep_equals_interleave_for_arbitrary_angles(
        self, n: int, seed: int
    ) -> None:
        """For random per-level angles, sweep and interleave produce the same unitary.

        Locks the non-trivial algebraic identity that the implementation
        relies on for ``_decompose``: emitting the two stages as two
        separate Gray-walk sweeps (RY first, then RZ) is mathematically
        equivalent to the Möttönen-Vartiainen interleaved order
        (per-level RY then RZ), not just on ``|0>^n`` but as full
        unitaries.
        """
        rng = np.random.default_rng(seed)
        ry_per_level = [rng.standard_normal(2**k) for k in range(n)]
        rz_per_level = [rng.standard_normal(2**k) for k in range(n)]
        u_sweep = _build_unitary_via_emission(n, ry_per_level, rz_per_level, "sweep")
        u_interleave = _build_unitary_via_emission(
            n, ry_per_level, rz_per_level, "interleave"
        )
        np.testing.assert_allclose(u_sweep, u_interleave, atol=1e-10)


# ---------------------------------------------------------------------------
# Angle pre-computation
# ---------------------------------------------------------------------------


class TestComputeAngles:
    """Validate the classical angle pre-computation."""

    @pytest.mark.parametrize(
        "n_qubits, expected",
        [
            (1, np.array([np.pi / 2])),
            (2, np.array([np.pi / 2, np.pi / 2, 0.0])),
            (3, np.array([np.pi / 2, np.pi / 2, 0.0, np.pi / 2, 0.0, 0.0, 0.0])),
        ],
    )
    def test_uniform_amplitudes_match_analytical(
        self, n_qubits: int, expected: np.ndarray
    ) -> None:
        """Uniform amplitudes reduce to a known closed form."""
        ry_angles = compute_mottonen_amplitude_encoding_ry_angles(np.ones(2**n_qubits))
        np.testing.assert_allclose(ry_angles, expected, atol=1e-12)

    def test_length_invariants(self) -> None:
        """Output length matches ``2**n - 1`` for every supported size."""
        for n in _QUBIT_COUNTS:
            ry_angles = compute_mottonen_amplitude_encoding_ry_angles(np.ones(2**n))
            assert ry_angles.shape == (2**n - 1,)


# ---------------------------------------------------------------------------
# Lazy construction (no eager angle computation)
# ---------------------------------------------------------------------------


class TestLazyConstruction:
    """``MottonenAmplitudeEncoding`` defers angle computation.

    The published Möttönen construction is intrinsically ``O(2^n)``
    (normalisation) and ``O(n * 2^n)`` (angle computation).
    ``__init__`` runs the cheap normalisation pass eagerly so input
    errors surface at construction time, but the dominant
    angle-precomputation cost is deferred:

    * ``__init__`` populates ``_normalized``, ``_is_complex``,
      ``_num_qubits`` (``O(2^n)``).  Angle caches stay empty.
    * The first ``_resources()`` call reads ``_is_complex`` directly
      and does NOT trigger angle precomputation.
    * The first ``_decompose()`` call triggers the
      ``O(n * 2^n)`` angle pass.  Subsequent calls hit the cache.
    """

    def test_init_does_not_compute_angles(self) -> None:
        """``__init__`` normalises eagerly but leaves angle caches empty."""
        gate = MottonenAmplitudeEncoding([1.0, 2.0, 3.0, 4.0])
        assert gate.num_target_qubits == 2
        assert gate._is_complex is False
        # Angles are not yet computed.
        assert gate._ry_angles_per_level_cache is None
        assert gate._rz_angles_per_level_cache is None

    def test_decompose_triggers_angle_computation(self) -> None:
        """``_decompose()`` populates the angle caches."""
        gate = MottonenAmplitudeEncoding([1.0, 2.0, 3.0, 4.0])

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            # invoking the gate inside a kernel call goes through _decompose
            q[0], q[1] = gate(q[0], q[1])
            return qmc.measure(q)

        kernel.build()
        assert gate._ry_angles_per_level_cache is not None
        # Real input → no Rz cache.
        assert gate._rz_angles_per_level_cache is None

    def test_kernel_build_triggers_eager_decomposition_known_limitation(
        self,
    ) -> None:
        """``kernel.build()`` ends up triggering ``_decompose()`` (framework limitation).

        Documented limitation: ``CompositeGate.__call__`` eagerly runs
        ``_build_decomposition_block`` whenever the gate is invoked
        inside a kernel body, in order to populate the ``body`` field on the
        resulting ``InvokeOperation``.  As a consequence, even though
        ``MottonenAmplitudeEncoding._decompose()`` is lazy on the gate
        object itself, going through ``amplitude_encoding(q, amps)``
        inside a ``@qkernel`` will still compute the angles at
        kernel-build time — the lazy contract is partially defeated by
        the surrounding framework.

        This test pins the current behaviour so that, if a future
        refactor of the ``CompositeGate`` framework defers
        ``_build_decomposition_block`` until emit time, this assertion
        fails loudly and the lazy contract can be tightened.
        """
        amps = [float(i + 1) for i in range(2**4)]

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q = amplitude_encoding(q, amps)
            return qmc.measure(q)

        block = kernel.build()
        invoke_ops = [
            op
            for op in block.operations
            if isinstance(op, InvokeOperation)
            and op.target.name == "mottonen_amplitude_encoding"
        ]
        assert len(invoke_ops) == 1
        # Today: the decomposition body IS populated by kernel.build() because
        # CompositeGate.__call__ eagerly builds the decomposition block.
        assert invoke_ops[0].body is not None
        assert invoke_ops[0].body.operations


# ---------------------------------------------------------------------------
# Resource estimation against the published Möttönen-Vartiainen formula
# ---------------------------------------------------------------------------


class TestResourceEstimationAgainstPaper:
    """``kernel.estimate_resources()`` matches the published Möttönen formula.

    Reference: M. Möttönen, J. J. Vartiainen, V. Bergholm, M. M. Salomaa,
    "Transformation of quantum states using uniformly controlled
    rotations", arXiv:quant-ph/0407010 (2004).  **Section II** of
    that paper (Fig. 2 + paragraph after Eq. (2)) describes the
    Gray-code decomposition of a ``k``-controlled uniformly
    controlled rotation into ``2**k`` elementary rotations and
    ``2**k`` CNOTs (the paper does not number this as a Lemma /
    Theorem).  Summing over the ``n`` stages of the
    amplitude-encoding cascade — stage ``k`` for
    ``k = 0, 1, ..., n - 1``, where stage 0 is uncontrolled and
    contributes no CNOTs — yields the closed forms:

    | input    | rotations         | CNOTs                |
    |----------|-------------------|----------------------|
    | real     | ``2**n - 1``      | ``2**n - 2``         |
    | complex  | ``2 (2**n - 1)``  | ``2 (2**n - 2)``     |

    These are the basic Möttönen-Vartiainen counts before any of the
    inter-stage cancellation optimisations described later in the same
    paper (which the Qamomile implementation does NOT apply).  They
    therefore line up exactly with what
    ``MottonenAmplitudeEncoding._resources()`` reports — but here we
    deliberately go through the public ``kernel.estimate_resources()``
    entry point, which walks the IR and resolves the composite gate
    automatically, so this test exercises the full
    composite-gate-aware estimator path on top of the formula.
    """

    @pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 5])
    def test_real_amplitude_encoding_matches_published_formula(
        self, n_qubits: int
    ) -> None:
        """Real input: rotations = ``2^n - 1``, CNOTs = ``max(0, 2^n - 2)``."""
        amps = np.ones(2**n_qubits).tolist()

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n_qubits, "q")
            q = amplitude_encoding(q, amps)
            return qmc.measure(q)

        est = kernel.estimate_resources()
        expected_rot = 2**n_qubits - 1
        # 2^n - 2 underflows to a negative for n=1; clamp to 0 to match
        # the implementation's no-CNOT special case at the leaf level.
        expected_cnot = max(0, 2**n_qubits - 2)
        assert int(est.qubits) == n_qubits
        assert int(est.gates.rotation_gates) == expected_rot
        assert int(est.gates.two_qubit) == expected_cnot
        assert int(est.gates.total) == expected_rot + expected_cnot
        # No T gates and no multi-qubit (3+) gates in the basic Möttönen
        # decomposition: rotations are single-qubit, CNOTs are two-qubit
        # Cliffords.
        assert int(est.gates.t_gates) == 0
        assert int(est.gates.multi_qubit) == 0
        assert int(est.gates.clifford_gates) == expected_cnot
        assert int(est.gates.single_qubit) == expected_rot

    @pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
    def test_complex_amplitude_encoding_matches_published_formula(
        self, n_qubits: int
    ) -> None:
        """Complex input: rotations and CNOTs both double (Ry + Rz stages)."""
        rng = np.random.default_rng(101 + n_qubits)
        # Force the two-stage path with a non-zero imaginary component;
        # complex inputs whose imag part is identically zero are silently
        # coerced to the cheaper real path.
        re = rng.standard_normal(2**n_qubits)
        im = rng.standard_normal(2**n_qubits)
        amps = (re + 1j * im).tolist()

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n_qubits, "q")
            q = amplitude_encoding(q, amps)
            return qmc.measure(q)

        est = kernel.estimate_resources()
        expected_rot = 2 * (2**n_qubits - 1)
        expected_cnot = 2 * max(0, 2**n_qubits - 2)
        assert int(est.qubits) == n_qubits
        assert int(est.gates.rotation_gates) == expected_rot
        assert int(est.gates.two_qubit) == expected_cnot
        assert int(est.gates.total) == expected_rot + expected_cnot
        assert int(est.gates.t_gates) == 0
        assert int(est.gates.multi_qubit) == 0
        assert int(est.gates.clifford_gates) == expected_cnot
        assert int(est.gates.single_qubit) == expected_rot


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Ensure malformed inputs are caught with helpful messages."""

    @pytest.mark.parametrize(
        "amplitudes, match",
        [
            ([1.0, 0.0, 0.0], "power of 2"),
            ([], "power of 2"),
            ([1.0], "power of 2"),  # length 1 = 2**0 = 0 qubits, rejected
            ([0.0, 0.0], "all zeros"),
        ],
    )
    def test_invalid_amplitudes_raise(
        self, amplitudes: list[float], match: str
    ) -> None:
        """Non-power-of-two (or under-length), empty, and all-zero inputs are rejected."""
        with pytest.raises(ValueError, match=match):
            MottonenAmplitudeEncoding(amplitudes)

    @pytest.mark.parametrize(
        "amplitudes",
        [
            pytest.param([[1.0, 0.0], [0.0, 1.0]], id="nested-list-2x2"),
            pytest.param(np.eye(4), id="ndarray-4x4"),
            pytest.param(np.zeros((2, 4), dtype=complex), id="ndarray-2x4-complex"),
            pytest.param(np.array(1.0), id="ndarray-0d-scalar"),
        ],
    )
    def test_non_1d_amplitudes_rejected(self, amplitudes) -> None:
        """Nested sequences and 2-D / 0-D arrays are rejected with a clear shape error.

        Without the early ``arr.ndim == 1`` check, e.g. a 2x2 nested list
        passes ``len(arr) == 2`` (a power of 2 ≥ 2) and only fails much
        later inside the angle computation with a confusing
        ``TypeError`` from ``math.atan2(np.ndarray, ...)``.  This test
        locks in the up-front shape validation in
        ``_validate_and_normalize``.
        """
        with pytest.raises(ValueError, match="1-D vector"):
            MottonenAmplitudeEncoding(amplitudes)

    def test_complex_nonzero_imag_accepted(self) -> None:
        """Complex inputs with non-zero imaginary part are accepted (full path).

        ``[1+1j, 1+0j]`` normalises to ``[(1+1j)/sqrt(3), 1/sqrt(3)]``.
        The magnitudes are ``[sqrt(2)/sqrt(3), 1/sqrt(3)]`` and the
        phase difference is ``arg(1 / (1+1j)) = -pi/4``, so

        * Ry angle: ``2 * arctan2(1/sqrt(3), sqrt(2)/sqrt(3))
                     = 2 * arctan2(1, sqrt(2))``,
        * Rz angle: ``-pi/4``.
        """
        amps = np.array([1.0 + 1j, 1.0 + 0j], dtype=complex)
        ry_angles = compute_mottonen_amplitude_encoding_ry_angles(amps)
        rz_angles = compute_mottonen_amplitude_encoding_rz_angles(amps)
        expected_ry = 2.0 * np.arctan2(1.0, np.sqrt(2.0))
        expected_rz = -np.pi / 4.0
        np.testing.assert_allclose(ry_angles, [expected_ry], atol=1e-12)
        np.testing.assert_allclose(rz_angles, [expected_rz], atol=1e-12)

    def test_complex_zero_imag_takes_real_path(self) -> None:
        """Complex inputs with zero imaginary part coerce cleanly to the real angle.

        ``[1+0j, 0+0j]`` normalises to ``[1, 0]`` so the single Gray-walk
        Ry angle is ``2 * arctan2(0, 1) = 0`` and the phase stage is a
        no-op (Rz angles are all zero by the real-input shortcut).
        """
        amps = np.array([1.0 + 0j, 0.0 + 0j], dtype=complex)
        ry_angles = compute_mottonen_amplitude_encoding_ry_angles(amps)
        rz_angles = compute_mottonen_amplitude_encoding_rz_angles(amps)
        np.testing.assert_allclose(ry_angles, [0.0], atol=1e-12)
        np.testing.assert_allclose(rz_angles, [0.0], atol=1e-12)

    def test_rz_angles_are_zero_for_real_input(self) -> None:
        """``compute_..._rz_angles`` returns zeros for real inputs of any size."""
        for n in _QUBIT_COUNTS:
            rz_angles = compute_mottonen_amplitude_encoding_rz_angles(np.ones(2**n))
            assert rz_angles.shape == (2**n - 1,)
            np.testing.assert_allclose(rz_angles, np.zeros(2**n - 1), atol=1e-12)

    def test_qubit_count_mismatch_raises(self) -> None:
        """``amplitude_encoding`` rejects qubit / amplitude mismatches."""

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q = amplitude_encoding(q, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            return qmc.measure(q)

        with pytest.raises(ValueError, match="qubits"):
            kernel.build()


@qmc.qkernel
def _amp_encode_with_symbolic_qubits(
    qs: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Sub-kernel taking a Vector[Qubit] of symbolic length.

    Used by ``test_symbolic_shape_qubits_raise_clear_error`` to drive
    the ``get_size`` failure path inside ``amplitude_encoding`` and
    confirm the API-specific error message is raised.
    """
    qs = amplitude_encoding(qs, [1.0, 0.0, 0.0, 0.0])
    return qs


class TestSymbolicShapeQubitsRejected:
    """``amplitude_encoding`` raises an API-specific error on symbolic qubits."""

    def test_symbolic_shape_qubits_raise_clear_error(self) -> None:
        """A Vector[Qubit] without a compile-time-known size is rejected
        with an ``amplitude_encoding``-prefixed ValueError that names the
        binding workaround, rather than the bare ``get_size`` message."""
        with pytest.raises(ValueError, match="amplitude_encoding requires"):
            _amp_encode_with_symbolic_qubits.build()


# ---------------------------------------------------------------------------
# Kernel builders shared by every backend
# ---------------------------------------------------------------------------


def _build_measure_kernel(
    amplitudes: list[float] | list[complex],
) -> qmc.QKernel:
    """Build a kernel that prepares *amplitudes* and measures the register.

    Args:
        amplitudes (list[float] | list[complex]): Real or complex
            amplitude vector of length ``2**n``.  Drives the register
            size (``n = log2(len(amplitudes))``).  Complex inputs route
            through the two-stage (Ry + Rz) Möttönen path; real inputs
            (and complex inputs with identically-zero imaginary part)
            stay on the single-stage Ry path.

    Returns:
        qmc.QKernel: Kernel returning ``Vector[Bit]`` (full-register
            measurement).
    """
    n_qubits = int(round(np.log2(len(amplitudes))))

    @qmc.qkernel
    def kernel() -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(n_qubits, "q")
        q = amplitude_encoding(q, amplitudes)
        return qmc.measure(q)

    return kernel


def _build_expval_kernel(
    amplitudes: list[float] | list[complex],
) -> qmc.QKernel:
    """Build a kernel that returns ``<H>`` on the prepared state.

    Args:
        amplitudes (list[float] | list[complex]): Real or complex
            amplitude vector of length ``2**n``.  Drives the register
            size (``n = log2(len(amplitudes))``).  Both Möttönen paths
            (single-stage Ry and two-stage Ry + Rz) feed the same
            ``qmc.expval`` consumer.

    Returns:
        qmc.QKernel: Kernel taking an ``H: Observable`` runtime binding
            and returning the expectation value as a ``Float``.
    """
    n_qubits = int(round(np.log2(len(amplitudes))))

    @qmc.qkernel
    def kernel(H: qmc.Observable) -> qmc.Float:
        q = qmc.qubit_array(n_qubits, "q")
        q = amplitude_encoding(q, amplitudes)
        return qmc.expval(q, H)

    return kernel


def _build_wrapper_measure_kernel(
    amplitudes: list[float] | list[complex],
) -> qmc.QKernel:
    """Build a wrapper-kernel amplitude-encoding sampler.

    Args:
        amplitudes (list[float] | list[complex]): Real or complex
            amplitude vector of length ``2**n``.

    Returns:
        qmc.QKernel: Kernel that calls an inner qkernel wrapper around
            ``amplitude_encoding`` before measuring the register.
    """
    n_qubits = int(round(np.log2(len(amplitudes))))

    @qmc.qkernel
    def layer(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply Möttönen encoding through a qkernel wrapper."""
        q = amplitude_encoding(q, amplitudes)
        return q

    @qmc.qkernel
    def kernel() -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(n_qubits, "q")
        q = layer(q)
        return qmc.measure(q)

    return kernel


def _build_wrapper_expval_kernel(
    amplitudes: list[float] | list[complex],
) -> qmc.QKernel:
    """Build a wrapper-kernel amplitude-encoding expval circuit.

    Args:
        amplitudes (list[float] | list[complex]): Real or complex
            amplitude vector of length ``2**n``.

    Returns:
        qmc.QKernel: Kernel that calls an inner qkernel wrapper around
            ``amplitude_encoding`` before computing ``<H>``.
    """
    n_qubits = int(round(np.log2(len(amplitudes))))

    @qmc.qkernel
    def layer(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply Möttönen encoding through a qkernel wrapper."""
        q = amplitude_encoding(q, amplitudes)
        return q

    @qmc.qkernel
    def kernel(H: qmc.Observable) -> qmc.Float:
        q = qmc.qubit_array(n_qubits, "q")
        q = layer(q)
        return qmc.expval(q, H)

    return kernel


def _build_nested_wrapper_measure_kernel(
    amplitudes: list[float] | list[complex],
) -> qmc.QKernel:
    """Build a two-level wrapper-kernel amplitude-encoding sampler.

    Args:
        amplitudes (list[float] | list[complex]): Real or complex
            amplitude vector of length ``2**n``.

    Returns:
        qmc.QKernel: Kernel whose outer wrapper calls another wrapper
            around ``amplitude_encoding``.
    """
    n_qubits = int(round(np.log2(len(amplitudes))))

    @qmc.qkernel
    def inner(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply Möttönen encoding inside the innermost wrapper."""
        q = amplitude_encoding(q, amplitudes)
        return q

    @qmc.qkernel
    def outer(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Forward to the inner Möttönen wrapper."""
        q = inner(q)
        return q

    @qmc.qkernel
    def kernel() -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(n_qubits, "q")
        q = outer(q)
        return qmc.measure(q)

    return kernel


def _build_nested_wrapper_expval_kernel(
    amplitudes: list[float] | list[complex],
) -> qmc.QKernel:
    """Build a two-level wrapper-kernel amplitude-encoding expval circuit.

    Args:
        amplitudes (list[float] | list[complex]): Real or complex
            amplitude vector of length ``2**n``.

    Returns:
        qmc.QKernel: Kernel whose outer wrapper calls another wrapper
            around ``amplitude_encoding`` before computing ``<H>``.
    """
    n_qubits = int(round(np.log2(len(amplitudes))))

    @qmc.qkernel
    def inner(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Apply Möttönen encoding inside the innermost wrapper."""
        q = amplitude_encoding(q, amplitudes)
        return q

    @qmc.qkernel
    def outer(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        """Forward to the inner Möttönen wrapper."""
        q = inner(q)
        return q

    @qmc.qkernel
    def kernel(H: qmc.Observable) -> qmc.Float:
        q = qmc.qubit_array(n_qubits, "q")
        q = outer(q)
        return qmc.expval(q, H)

    return kernel


def _expected_z0(amplitudes: list[float] | list[complex]) -> float:
    """Return the little-endian ``<Z0>`` expectation for *amplitudes*.

    Args:
        amplitudes (list[float] | list[complex]): Real or complex
            amplitude vector.

    Returns:
        float: Analytic expectation where basis-index bit 0 corresponds
            to qubit 0.
    """
    probabilities = np.abs(_normalize(amplitudes)) ** 2
    return float(
        sum(prob if (idx & 1) == 0 else -prob for idx, prob in enumerate(probabilities))
    )


def _expected_single_pauli(
    amplitudes: list[float] | list[complex],
    pauli: str,
    qubit: int,
) -> float:
    """Return the little-endian single-Pauli expectation for *amplitudes*.

    Args:
        amplitudes (list[float] | list[complex]): Real or complex
            amplitude vector.
        pauli (str): One of ``"X"``, ``"Y"``, ``"Z"``.
        qubit (int): Qubit index in little-endian basis order.

    Returns:
        float: Analytic expectation for the selected single-qubit Pauli.

    Raises:
        ValueError: If *pauli* is not one of ``"X"``, ``"Y"``, or ``"Z"``.
    """
    state = _normalize(amplitudes).astype(complex)
    mask = 1 << qubit

    if pauli == "Z":
        return float(
            sum(
                (1.0 if (idx & mask) == 0 else -1.0) * abs(amplitude) ** 2
                for idx, amplitude in enumerate(state)
            )
        )

    if pauli not in {"X", "Y"}:
        raise ValueError(f"Unknown Pauli label: {pauli}")

    total = 0.0 + 0.0j
    for idx, amplitude in enumerate(state):
        paired_idx = idx ^ mask
        if pauli == "X":
            coefficient = 1.0
        elif (idx & mask) == 0:
            coefficient = 1.0j
        else:
            coefficient = -1.0j
        total += np.conjugate(state[paired_idx]) * coefficient * amplitude
    return float(np.real_if_close(total))


def _bits_to_index(bits: tuple[int, ...] | int) -> int:
    """Convert a measurement outcome to a flat basis-state index.

    Qamomile's sample API yields ``int`` for scalar ``Bit`` results and
    ``tuple[int, ...]`` for ``Vector[Bit]`` results; both forms are
    handled here.

    Args:
        bits (tuple[int, ...] | int): Scalar ``int`` (scalar Bit) or
            tuple of bits (Vector[Bit]).  Little-endian: bit ``i`` of the
            resulting index corresponds to qubit ``i`` of the measured
            register.

    Returns:
        int: The integer basis-state index.
    """
    if isinstance(bits, int):
        return int(bits)
    return sum(int(b) << i for i, b in enumerate(bits))


def _shot_noise_tolerance(p: float, shots: int) -> float:
    """Return the ``_STD_TOLERANCE``-sigma binomial half-width for probability *p*.

    Args:
        p (float): Expected single-shot probability for the bin under test.
        shots (int): Total number of shots in the sampling experiment.

    Returns:
        float: ``_STD_TOLERANCE * sqrt(p * (1-p) / shots)``, with a tiny
            floor on ``p * (1-p)`` so degenerate ``p in {0, 1}`` bins
            still admit a non-zero (but small) tolerance.
    """
    return _STD_TOLERANCE * float(np.sqrt(max(p * (1.0 - p), 1e-12) / shots))


# ---------------------------------------------------------------------------
# Wrapper-kernel regressions for custom-composite boxing
# ---------------------------------------------------------------------------


class TestAmplitudeEncodingWrapperBoxing:
    """Regression tests for amplitude_encoding inside qkernel wrappers."""

    @pytest.mark.parametrize(
        "amplitudes",
        [
            pytest.param(
                [1.0, 2.0, 3.0, -4.0, 5.0, -6.0, 7.0, 8.0],
                id="real-3q",
            ),
            pytest.param(
                [1.0 + 0j, 1.0 + 1j, 1.0 - 1j, 0.0 + 2j],
                id="complex-2q",
            ),
        ],
    )
    def test_invoke_operands_resolve_to_caller_array(
        self, amplitudes: list[float] | list[complex]
    ) -> None:
        """Boxed Möttönen operands must resolve to the caller array."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        kernel = _build_wrapper_measure_kernel(amplitudes)

        block = transpiler.to_block(kernel)
        block = transpiler.substitute(block)
        block = transpiler.resolve_parameter_shapes(block, None)
        affine = transpiler.inline(block)

        qinit_arrays = [
            op.results[0] for op in affine.operations if isinstance(op, QInitOperation)
        ]
        assert qinit_arrays
        qinit_uuid = qinit_arrays[0].uuid

        invoke_ops = [op for op in affine.operations if isinstance(op, InvokeOperation)]
        assert invoke_ops

        root_addresses: list[tuple[str, int]] = []
        for op in affine.operations:
            if not isinstance(op, InvokeOperation):
                continue
            for operand in op.target_qubits:
                if operand.parent_array is None:
                    continue
                root_addr = resolve_root_qubit_address(operand)
                assert root_addr is not None
                root_addresses.append(root_addr)

        assert root_addresses
        assert {root_uuid for root_uuid, _ in root_addresses} == {qinit_uuid}

        affine = transpiler.unroll_recursion(affine, None)
        validated = transpiler.affine_validate(affine)
        partially_evaluated = transpiler.partial_eval(validated, None)
        partially_evaluated = transpiler.slice_borrow_check(partially_evaluated)
        partially_evaluated = transpiler.strip_slice_ops(partially_evaluated)
        analyzed = transpiler.analyze(partially_evaluated)
        analyzed = transpiler.classical_lowering(analyzed)
        analyzed = transpiler.validate_symbolic_shapes(analyzed)
        plan = transpiler.plan(analyzed)
        quantum_segment = next(
            step.segment for step in plan.steps if isinstance(step, QuantumStep)
        )

        ResourceAllocator().allocate(quantum_segment.operations, bindings={})

    @pytest.mark.parametrize("amplitudes", _WRAPPER_AMPLITUDE_CASES)
    def test_forward_wrapper_sampler(self, sdk_transpiler, amplitudes) -> None:
        """Wrapped amplitude_encoding samples the encoded distribution."""
        kernel = _build_wrapper_measure_kernel(amplitudes)
        exe = sdk_transpiler.transpiler.transpile(kernel)
        results = (
            exe.sample(sdk_transpiler.transpiler.executor(), shots=_SHOTS)
            .result()
            .results
        )

        expected_probs = np.abs(_normalize(amplitudes)) ** 2
        observed_probs = np.zeros_like(expected_probs)
        total = 0
        for bits, count in results:
            observed_probs[_bits_to_index(bits)] = count / _SHOTS
            total += count
        assert total == _SHOTS

        for i, p_exp in enumerate(expected_probs):
            assert abs(observed_probs[i] - p_exp) < _shot_noise_tolerance(
                p_exp, _SHOTS
            ), (
                f"{sdk_transpiler.backend_name} bin {i}: "
                f"got {observed_probs[i]:.4f}, expected {p_exp:.4f} "
                f"(amplitudes={amplitudes})"
            )

    @pytest.mark.parametrize("amplitudes", _WRAPPER_AMPLITUDE_CASES)
    def test_forward_wrapper_expval(self, sdk_transpiler, amplitudes) -> None:
        """Wrapped amplitude_encoding gives the analytic little-endian <Z0>."""
        n_qubits = int(round(np.log2(len(amplitudes))))
        H = _pad_observable(n_qubits, _make_pauli("Z", 0))
        exe = sdk_transpiler.transpiler.transpile(
            _build_wrapper_expval_kernel(amplitudes), bindings={"H": H}
        )
        result = exe.run(sdk_transpiler.transpiler.executor()).result()
        assert float(result) == pytest.approx(_expected_z0(amplitudes), abs=1e-8)

    def test_forward_wrapper_phase_expval(self, sdk_transpiler) -> None:
        """Wrapped complex amplitude_encoding preserves phase-sensitive <Y0>."""
        amplitudes = [1.0 + 0j, 0.0 + 1j]
        H = _make_pauli("Y", 0)
        exe = sdk_transpiler.transpiler.transpile(
            _build_wrapper_expval_kernel(amplitudes), bindings={"H": H}
        )
        result = exe.run(sdk_transpiler.transpiler.executor()).result()
        assert float(result) == pytest.approx(
            _expected_single_pauli(amplitudes, "Y", 0), abs=1e-8
        )

    def test_nested_wrapper_sampler(self, sdk_transpiler) -> None:
        """Nested wrapper amplitude_encoding samples the encoded distribution."""
        amplitudes = [1.0, 2.0, 3.0, -4.0, 5.0, -6.0, 7.0, 8.0]
        exe = sdk_transpiler.transpiler.transpile(
            _build_nested_wrapper_measure_kernel(amplitudes)
        )
        results = (
            exe.sample(sdk_transpiler.transpiler.executor(), shots=_SHOTS)
            .result()
            .results
        )

        expected_probs = np.abs(_normalize(amplitudes)) ** 2
        observed_probs = np.zeros_like(expected_probs)
        total = 0
        for bits, count in results:
            observed_probs[_bits_to_index(bits)] = count / _SHOTS
            total += count
        assert total == _SHOTS

        for i, p_exp in enumerate(expected_probs):
            assert abs(observed_probs[i] - p_exp) < _shot_noise_tolerance(
                p_exp, _SHOTS
            ), (
                f"{sdk_transpiler.backend_name} bin {i}: "
                f"got {observed_probs[i]:.4f}, expected {p_exp:.4f}"
            )

    def test_nested_wrapper_expval(self, sdk_transpiler) -> None:
        """Nested wrapper complex amplitude_encoding preserves phase-sensitive <Y0>."""
        amplitudes = [1.0 + 0j, 0.0 + 1j, 0.0 + 0j, 0.0 + 0j]
        n_qubits = int(round(np.log2(len(amplitudes))))
        H = _pad_observable(n_qubits, _make_pauli("Y", 0))
        exe = sdk_transpiler.transpiler.transpile(
            _build_nested_wrapper_expval_kernel(amplitudes), bindings={"H": H}
        )
        result = exe.run(sdk_transpiler.transpiler.executor()).result()
        assert float(result) == pytest.approx(
            _expected_single_pauli(amplitudes, "Y", 0), abs=1e-8
        )


# ---------------------------------------------------------------------------
# Qiskit backend
# ---------------------------------------------------------------------------


class TestEncodingQiskit:
    """Statevector / sampler / expval verification on the Qiskit backend."""

    @pytest.mark.parametrize(
        "amplitudes",
        [pytest.param(amps, id=name) for name, amps in _FIXED_AMPLITUDES],
    )
    def test_fixed_amplitudes_statevector(
        self, qiskit_transpiler, amplitudes: list[float]
    ) -> None:
        """Each catalogued real amplitude vector is faithfully encoded."""
        qc = qiskit_transpiler.to_circuit(_build_measure_kernel(amplitudes))
        sv = run_statevector(qc).real
        expected = _normalize(amplitudes)
        assert _state_fidelity(sv, expected) == pytest.approx(1.0, abs=1e-8)

    @pytest.mark.parametrize(
        "amplitudes",
        [pytest.param(amps, id=name) for name, amps in _FIXED_COMPLEX_AMPLITUDES],
    )
    def test_fixed_complex_amplitudes_statevector(
        self, qiskit_transpiler, amplitudes: list[complex]
    ) -> None:
        """Each catalogued complex amplitude vector is faithfully encoded."""
        qc = qiskit_transpiler.to_circuit(_build_measure_kernel(amplitudes))
        sv = run_statevector(qc)  # keep full complex
        expected = _normalize(amplitudes)
        assert _state_fidelity(sv, expected) == pytest.approx(1.0, abs=1e-8)

    @pytest.mark.parametrize("n_qubits", _QUBIT_COUNTS)
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_random_amplitudes_statevector(
        self, qiskit_transpiler, n_qubits: int, seed: int
    ) -> None:
        """Random signed amplitudes (1..4 qubits) round-trip exactly."""
        rng = np.random.default_rng(seed)
        amplitudes = rng.standard_normal(2**n_qubits).tolist()
        qc = qiskit_transpiler.to_circuit(_build_measure_kernel(amplitudes))
        sv = run_statevector(qc).real
        expected = _normalize(amplitudes)
        assert _state_fidelity(sv, expected) == pytest.approx(1.0, abs=1e-8)

    @pytest.mark.parametrize("n_qubits", _QUBIT_COUNTS)
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_random_complex_amplitudes_statevector(
        self, qiskit_transpiler, n_qubits: int, seed: int
    ) -> None:
        """Random complex amplitudes (1..4 qubits) round-trip exactly."""
        rng = np.random.default_rng(seed + 10000)  # disjoint seed space
        re = rng.standard_normal(2**n_qubits)
        im = rng.standard_normal(2**n_qubits)
        amplitudes = (re + 1j * im).tolist()
        qc = qiskit_transpiler.to_circuit(_build_measure_kernel(amplitudes))
        sv = run_statevector(qc)
        expected = _normalize(amplitudes)
        assert _state_fidelity(sv, expected) == pytest.approx(1.0, abs=1e-8)

    @pytest.mark.parametrize(
        "amplitudes",
        [pytest.param(amps, id=name) for name, amps in _FIXED_AMPLITUDES]
        + [pytest.param(amps, id=name) for name, amps in _FIXED_COMPLEX_AMPLITUDES],
    )
    def test_fixed_amplitudes_sampler(
        self,
        qiskit_transpiler,
        seeded_executor,
        amplitudes: list[float] | list[complex],
    ) -> None:
        """Empirical histogram matches |amplitudes|^2 within shot noise."""
        kernel = _build_measure_kernel(amplitudes)
        exe = qiskit_transpiler.transpile(kernel)
        results = exe.sample(seeded_executor, shots=_SHOTS).result().results

        expected_probs = np.abs(_normalize(amplitudes)) ** 2
        observed_probs = np.zeros_like(expected_probs)
        total = 0
        for bits, count in results:
            observed_probs[_bits_to_index(bits)] = count / _SHOTS
            total += count
        assert total == _SHOTS

        for i, p_exp in enumerate(expected_probs):
            assert abs(observed_probs[i] - p_exp) < _shot_noise_tolerance(
                p_exp, _SHOTS
            ), (
                f"bin {i}: got {observed_probs[i]:.4f}, expected {p_exp:.4f} "
                f"(amplitudes={amplitudes})"
            )

    @pytest.mark.parametrize(
        "amplitudes, pauli, qubit_idx, expected",
        [pytest.param(amps, p, q, e, id=name) for name, amps, p, q, e in _EXPVAL_CASES],
    )
    def test_expval(
        self,
        qiskit_transpiler,
        amplitudes: list[float] | list[complex],
        pauli: str,
        qubit_idx: int,
        expected: float,
    ) -> None:
        """``<H>`` on the encoded state matches the hand-derived value."""
        n_qubits = int(round(np.log2(len(amplitudes))))
        H = _pad_observable(n_qubits, _make_pauli(pauli, qubit_idx))
        exe = qiskit_transpiler.transpile(
            _build_expval_kernel(amplitudes), bindings={"H": H}
        )
        result = exe.run(qiskit_transpiler.executor()).result()
        assert float(result) == pytest.approx(expected, abs=1e-8)


# ---------------------------------------------------------------------------
# QuriParts backend
# ---------------------------------------------------------------------------


@pytest.mark.quri_parts
class TestEncodingQuriParts:
    """Sampler / expval verification on the QuriParts backend (via Qulacs).

    Statevector verification is implicitly covered by the sampler test
    (the Born probabilities match) and by the expval test (the observable
    expectation values match the analytic ones).
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        pytest.importorskip("quri_parts")
        pytest.importorskip("quri_parts.qulacs")
        from qamomile.quri_parts.transpiler import QuriPartsTranspiler

        self.transpiler = QuriPartsTranspiler()

    @pytest.mark.parametrize(
        "amplitudes",
        [pytest.param(amps, id=name) for name, amps in _FIXED_AMPLITUDES]
        + [pytest.param(amps, id=name) for name, amps in _FIXED_COMPLEX_AMPLITUDES],
    )
    def test_fixed_amplitudes_sampler(
        self, amplitudes: list[float] | list[complex]
    ) -> None:
        """Empirical histogram matches |amplitudes|^2 within shot noise."""
        kernel = _build_measure_kernel(amplitudes)
        exe = self.transpiler.transpile(kernel)
        results = exe.sample(self.transpiler.executor(), shots=_SHOTS).result().results

        expected_probs = np.abs(_normalize(amplitudes)) ** 2
        observed_probs = np.zeros_like(expected_probs)
        total = 0
        for bits, count in results:
            observed_probs[_bits_to_index(bits)] = count / _SHOTS
            total += count
        assert total == _SHOTS

        for i, p_exp in enumerate(expected_probs):
            assert abs(observed_probs[i] - p_exp) < _shot_noise_tolerance(
                p_exp, _SHOTS
            ), (
                f"bin {i}: got {observed_probs[i]:.4f}, expected {p_exp:.4f} "
                f"(amplitudes={amplitudes})"
            )

    @pytest.mark.parametrize(
        "amplitudes, pauli, qubit_idx, expected",
        [pytest.param(amps, p, q, e, id=name) for name, amps, p, q, e in _EXPVAL_CASES],
    )
    def test_expval(
        self,
        amplitudes: list[float] | list[complex],
        pauli: str,
        qubit_idx: int,
        expected: float,
    ) -> None:
        """``<H>`` on the encoded state matches the hand-derived value."""
        n_qubits = int(round(np.log2(len(amplitudes))))
        H = _pad_observable(n_qubits, _make_pauli(pauli, qubit_idx))
        exe = self.transpiler.transpile(
            _build_expval_kernel(amplitudes), bindings={"H": H}
        )
        result = exe.run(self.transpiler.executor()).result()
        assert float(result) == pytest.approx(expected, abs=1e-8)


# ---------------------------------------------------------------------------
# CUDA-Q backend
# ---------------------------------------------------------------------------


@pytest.mark.cudaq
class TestEncodingCudaq:
    """Sampler / expval verification on the CUDA-Q backend.

    Statevector verification is implicitly covered by the sampler test
    (the Born probabilities match) and by the expval test (the observable
    expectation values match the analytic ones).  CUDA-Q's sampler is
    intrinsically stochastic, hence the shot-noise tolerance.
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        pytest.importorskip("cudaq")
        from qamomile.cudaq.transpiler import CudaqTranspiler

        self.transpiler = CudaqTranspiler()

    @pytest.mark.parametrize(
        "amplitudes",
        [pytest.param(amps, id=name) for name, amps in _FIXED_AMPLITUDES]
        + [pytest.param(amps, id=name) for name, amps in _FIXED_COMPLEX_AMPLITUDES],
    )
    def test_fixed_amplitudes_sampler(
        self, amplitudes: list[float] | list[complex]
    ) -> None:
        """Empirical histogram matches |amplitudes|^2 within shot noise."""
        kernel = _build_measure_kernel(amplitudes)
        exe = self.transpiler.transpile(kernel)
        results = exe.sample(self.transpiler.executor(), shots=_SHOTS).result().results

        expected_probs = np.abs(_normalize(amplitudes)) ** 2
        observed_probs = np.zeros_like(expected_probs)
        total = 0
        for bits, count in results:
            observed_probs[_bits_to_index(bits)] = count / _SHOTS
            total += count
        assert total == _SHOTS

        for i, p_exp in enumerate(expected_probs):
            assert abs(observed_probs[i] - p_exp) < _shot_noise_tolerance(
                p_exp, _SHOTS
            ), (
                f"bin {i}: got {observed_probs[i]:.4f}, expected {p_exp:.4f} "
                f"(amplitudes={amplitudes})"
            )

    @pytest.mark.parametrize(
        "amplitudes, pauli, qubit_idx, expected",
        [pytest.param(amps, p, q, e, id=name) for name, amps, p, q, e in _EXPVAL_CASES],
    )
    def test_expval(
        self,
        amplitudes: list[float] | list[complex],
        pauli: str,
        qubit_idx: int,
        expected: float,
    ) -> None:
        """``<H>`` on the encoded state matches the hand-derived value."""
        n_qubits = int(round(np.log2(len(amplitudes))))
        H = _pad_observable(n_qubits, _make_pauli(pauli, qubit_idx))
        exe = self.transpiler.transpile(
            _build_expval_kernel(amplitudes), bindings={"H": H}
        )
        result = exe.run(self.transpiler.executor()).result()
        assert float(result) == pytest.approx(expected, abs=1e-8)


# ---------------------------------------------------------------------------
# amplitude_encoding via bound Vector[Float] amplitudes
# ---------------------------------------------------------------------------


def _build_amplitude_encoding_bound_kernel(n_qubits: int) -> qmc.QKernel:
    """Build a kernel taking a Vector[Float] of *amplitudes* (bound, real).

    Args:
        n_qubits (int): Number of qubits in the register.

    Returns:
        qmc.QKernel: Kernel taking ``amps: Vector[Float]`` of length
            ``2**n_qubits`` and returning the measured ``Vector[Bit]``.
    """

    @qmc.qkernel
    def kernel(amps: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(n_qubits, "q")
        q = amplitude_encoding(q, amps)
        return qmc.measure(q)

    return kernel


class TestAmplitudeEncodingBoundVector:
    """``amplitude_encoding`` accepts a Vector[Float] kernel parameter when bound.

    The same ``MottonenAmplitudeEncoding`` CompositeGate path runs;
    only the entry-point ergonomics differ from the closure-based call.
    """

    @pytest.mark.parametrize(
        "amplitudes",
        [pytest.param(amps, id=name) for name, amps in _FIXED_AMPLITUDES],
    )
    def test_bound_amplitudes_statevector(
        self, qiskit_transpiler, amplitudes: list[float]
    ) -> None:
        """Bindings-bound Vector[Float] amplitudes produce the right statevector."""
        n = int(round(np.log2(len(amplitudes))))
        kernel = _build_amplitude_encoding_bound_kernel(n)
        qc = qiskit_transpiler.to_circuit(kernel, bindings={"amps": amplitudes})
        sv = run_statevector(qc).real
        expected = _normalize(amplitudes)
        assert _state_fidelity(sv, expected) == pytest.approx(1.0, abs=1e-8)

    def test_runtime_parameter_rejected(self, qiskit_transpiler) -> None:
        """``parameters=["amps"]`` raises a directing error toward the angle path."""
        kernel = _build_amplitude_encoding_bound_kernel(2)
        with pytest.raises(ValueError, match="amplitude_encoding_from_angles"):
            qiskit_transpiler.transpile(kernel, parameters=["amps"])

    def test_bound_amplitudes_emits_composite_gate(self, qiskit_transpiler) -> None:
        """Bound-Vector path keeps the high-level invoke in the IR.

        Distinguishes the bound path from
        ``amplitude_encoding_from_angles``, which emits flat RY/CNOT
        directly.  Verified by counting block operations in the traced
        kernel — the bound path produces a single composite-gate
        invocation while the angle path expands inline.
        """
        kernel = _build_amplitude_encoding_bound_kernel(2)
        block = kernel.build(amps=[1.0, 2.0, 3.0, 4.0])
        invoke_ops = [
            op
            for op in block.operations
            if isinstance(op, InvokeOperation)
            and op.target.name == "mottonen_amplitude_encoding"
        ]
        assert len(invoke_ops) == 1
        assert invoke_ops[0].attrs["custom_name"] == "mottonen_amplitude_encoding"


# ---------------------------------------------------------------------------
# Parametric amplitude encoding (Vector[Float] angles as kernel parameters)
# ---------------------------------------------------------------------------


def _build_parametric_real_kernel(n_qubits: int) -> qmc.QKernel:
    """Build a kernel that prepares a state from a Vector[Float] of Ry angles.

    Args:
        n_qubits (int): Number of qubits in the register.

    Returns:
        qmc.QKernel: A kernel taking ``ry_angles: Vector[Float]`` of length
            ``2**n_qubits - 1`` and returning the measured ``Vector[Bit]``.
    """

    @qmc.qkernel
    def kernel(ry_angles: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(n_qubits, "q")
        q = amplitude_encoding_from_angles(q, ry_angles)
        return qmc.measure(q)

    return kernel


def _build_parametric_complex_kernel(n_qubits: int) -> qmc.QKernel:
    """Build a kernel that prepares a state from Vector[Float] Ry + Rz angles.

    Args:
        n_qubits (int): Number of qubits in the register.

    Returns:
        qmc.QKernel: A kernel taking ``ry_angles`` and ``rz_angles``
            (each of length ``2**n_qubits - 1``) and returning the measured
            ``Vector[Bit]``.
    """

    @qmc.qkernel
    def kernel(
        ry_angles: qmc.Vector[qmc.Float], rz_angles: qmc.Vector[qmc.Float]
    ) -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(n_qubits, "q")
        q = amplitude_encoding_from_angles(q, ry_angles, rz_angles)
        return qmc.measure(q)

    return kernel


def _build_parametric_real_expval_kernel(n_qubits: int) -> qmc.QKernel:
    """Build an expval kernel parametric over Ry angles.

    Pairs with :func:`_build_parametric_real_kernel` but returns
    ``<H>`` so the estimator code path is exercised.

    Args:
        n_qubits (int): Number of qubits in the register.

    Returns:
        qmc.QKernel: Kernel taking ``ry_angles: Vector[Float]`` and
            ``H: Observable`` and returning the expectation value as a
            ``Float``.
    """

    @qmc.qkernel
    def kernel(ry_angles: qmc.Vector[qmc.Float], H: qmc.Observable) -> qmc.Float:
        q = qmc.qubit_array(n_qubits, "q")
        q = amplitude_encoding_from_angles(q, ry_angles)
        return qmc.expval(q, H)

    return kernel


def _build_parametric_complex_expval_kernel(n_qubits: int) -> qmc.QKernel:
    """Build an expval kernel parametric over Ry + Rz angles.

    Args:
        n_qubits (int): Number of qubits in the register.

    Returns:
        qmc.QKernel: Kernel taking ``ry_angles`` and ``rz_angles``
            (Vector[Float]) plus ``H: Observable`` and returning the
            expectation value as a ``Float``.
    """

    @qmc.qkernel
    def kernel(
        ry_angles: qmc.Vector[qmc.Float],
        rz_angles: qmc.Vector[qmc.Float],
        H: qmc.Observable,
    ) -> qmc.Float:
        q = qmc.qubit_array(n_qubits, "q")
        q = amplitude_encoding_from_angles(q, ry_angles, rz_angles)
        return qmc.expval(q, H)

    return kernel


class TestParametricInputValidation:
    """Length checks at kernel build time for the parametric helper."""

    def test_ry_length_mismatch_raises(self) -> None:
        """``ry_angles`` length other than ``2**n - 1`` is rejected at build."""

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            # 2 qubits → expects 3 angles, we provide 5
            q = amplitude_encoding_from_angles(q, [0.0, 0.0, 0.0, 0.0, 0.0])
            return qmc.measure(q)

        with pytest.raises(ValueError, match="ry_angles must have length"):
            kernel.build()

    def test_rz_length_mismatch_raises(self) -> None:
        """``rz_angles`` length other than ``2**n - 1`` is rejected at build."""

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            # 2 qubits → expects 3 angles for each stage; ry ok, rz wrong
            q = amplitude_encoding_from_angles(q, [0.0, 0.0, 0.0], rz_angles=[0.0, 0.0])
            return qmc.measure(q)

        with pytest.raises(ValueError, match="rz_angles must have length"):
            kernel.build()

    def test_ndarray_angles_stored_as_python_float(self) -> None:
        """``np.ndarray`` angle inputs land in IR metadata as ``float``, not ``np.float64``.

        Regression: ``angles[idx]`` on a NumPy array is ``np.float64``,
        which slips past downstream ``isinstance(c, float)`` checks on
        ``Value.get_const()``.  ``_emit_mottonen_gates`` must coerce
        each per-element angle to ``float`` before handing it to the
        rotation gate.
        """

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            ry = compute_mottonen_amplitude_encoding_ry_angles([1.0, 2.0, 3.0, 4.0])
            assert isinstance(ry, np.ndarray)
            q = amplitude_encoding_from_angles(q, ry)
            return qmc.measure(q)

        block = kernel.build()
        rotation_consts: list[object] = []
        for op in block.operations:
            if not isinstance(op, GateOperation):
                continue
            theta = getattr(op, "theta", None)
            if theta is None:
                continue
            const = theta.get_const()
            if const is not None:
                rotation_consts.append(const)

        # 2 qubits → 2**n - 1 = 3 RY rotations should land in the IR.
        assert len(rotation_consts) == 3
        for c in rotation_consts:
            assert type(c) is float, (
                f"angle stored as {type(c).__name__}, expected builtin float"
            )


class TestParametricEncodingQiskit:
    """Statevector + sampler verification of the parametric path on Qiskit.

    Exercises both ``bindings`` (compile-time fixed angles) and
    ``parameters`` (runtime-parametric angles) entry points, and both
    real and complex catalogues.
    """

    @pytest.mark.parametrize(
        "amplitudes",
        [pytest.param(amps, id=name) for name, amps in _FIXED_AMPLITUDES],
    )
    def test_real_via_bindings(
        self, qiskit_transpiler, amplitudes: list[float]
    ) -> None:
        """Real amps via compile-time-bound Ry-angle Vector[Float]."""
        n = int(round(np.log2(len(amplitudes))))
        ry = compute_mottonen_amplitude_encoding_ry_angles(amplitudes).tolist()
        kernel = _build_parametric_real_kernel(n)
        qc = qiskit_transpiler.to_circuit(kernel, bindings={"ry_angles": ry})
        sv = run_statevector(qc).real
        expected = _normalize(amplitudes)
        assert _state_fidelity(sv, expected) == pytest.approx(1.0, abs=1e-8)

    @pytest.mark.parametrize(
        "amplitudes",
        [pytest.param(amps, id=name) for name, amps in _FIXED_AMPLITUDES],
    )
    def test_real_via_runtime_parameters(
        self, qiskit_transpiler, seeded_executor, amplitudes: list[float]
    ) -> None:
        """Real amps via runtime-bound Ry-angle Vector[Float].

        The circuit is compiled once with ``parameters=["ry_angles"]``
        (resulting in ``2**n - 1`` Qiskit parameters) and then re-bound
        via ``executable.sample(bindings=...)`` — confirms the
        emitted gates accept Float handles and the sampler picks up
        runtime values.
        """
        n = int(round(np.log2(len(amplitudes))))
        kernel = _build_parametric_real_kernel(n)
        exe = qiskit_transpiler.transpile(kernel, parameters=["ry_angles"])
        # Circuit should carry 2**n - 1 parametric rotations
        circuit = exe.compiled_quantum[0].circuit
        assert len(circuit.parameters) == 2**n - 1

        ry = compute_mottonen_amplitude_encoding_ry_angles(amplitudes).tolist()
        results = (
            exe.sample(seeded_executor, shots=_SHOTS, bindings={"ry_angles": ry})
            .result()
            .results
        )
        expected_probs = np.abs(_normalize(amplitudes)) ** 2
        observed_probs = np.zeros_like(expected_probs)
        total = 0
        for bits, count in results:
            observed_probs[_bits_to_index(bits)] = count / _SHOTS
            total += count
        assert total == _SHOTS
        for i, p_exp in enumerate(expected_probs):
            assert abs(observed_probs[i] - p_exp) < _shot_noise_tolerance(
                p_exp, _SHOTS
            ), f"bin {i}: got {observed_probs[i]:.4f}, expected {p_exp:.4f}"

    @pytest.mark.parametrize(
        "amplitudes",
        [pytest.param(amps, id=name) for name, amps in _FIXED_COMPLEX_AMPLITUDES],
    )
    def test_complex_via_bindings(
        self, qiskit_transpiler, amplitudes: list[complex]
    ) -> None:
        """Complex amps via compile-time-bound Ry + Rz Vector[Float] pair."""
        n = int(round(np.log2(len(amplitudes))))
        ry = compute_mottonen_amplitude_encoding_ry_angles(amplitudes).tolist()
        rz = compute_mottonen_amplitude_encoding_rz_angles(amplitudes).tolist()
        kernel = _build_parametric_complex_kernel(n)
        qc = qiskit_transpiler.to_circuit(
            kernel, bindings={"ry_angles": ry, "rz_angles": rz}
        )
        sv = run_statevector(qc)
        expected = _normalize(amplitudes)
        assert _state_fidelity(sv, expected) == pytest.approx(1.0, abs=1e-8)

    @pytest.mark.parametrize(
        "amplitudes",
        [pytest.param(amps, id=name) for name, amps in _FIXED_COMPLEX_AMPLITUDES],
    )
    def test_complex_via_runtime_parameters(
        self,
        qiskit_transpiler,
        seeded_executor,
        amplitudes: list[complex],
    ) -> None:
        """Complex amps via runtime-bound Ry + Rz Vector[Float] pair."""
        n = int(round(np.log2(len(amplitudes))))
        kernel = _build_parametric_complex_kernel(n)
        exe = qiskit_transpiler.transpile(kernel, parameters=["ry_angles", "rz_angles"])
        circuit = exe.compiled_quantum[0].circuit
        # 2 stages of 2**n - 1 parametric rotations each
        assert len(circuit.parameters) == 2 * (2**n - 1)

        ry = compute_mottonen_amplitude_encoding_ry_angles(amplitudes).tolist()
        rz = compute_mottonen_amplitude_encoding_rz_angles(amplitudes).tolist()
        results = (
            exe.sample(
                seeded_executor,
                shots=_SHOTS,
                bindings={"ry_angles": ry, "rz_angles": rz},
            )
            .result()
            .results
        )
        expected_probs = np.abs(_normalize(amplitudes)) ** 2
        observed_probs = np.zeros_like(expected_probs)
        total = 0
        for bits, count in results:
            observed_probs[_bits_to_index(bits)] = count / _SHOTS
            total += count
        assert total == _SHOTS
        for i, p_exp in enumerate(expected_probs):
            assert abs(observed_probs[i] - p_exp) < _shot_noise_tolerance(
                p_exp, _SHOTS
            ), f"bin {i}: got {observed_probs[i]:.4f}, expected {p_exp:.4f}"

    def test_circuit_reuse_with_different_amplitudes(
        self, qiskit_transpiler, seeded_executor
    ) -> None:
        """Same compiled circuit can be re-bound to different amplitude vectors.

        The whole point of the runtime-parametric path: compile once,
        sample many times with different angles.
        """
        n = 2
        kernel = _build_parametric_real_kernel(n)
        exe = qiskit_transpiler.transpile(kernel, parameters=["ry_angles"])

        for amplitudes in [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
        ]:
            ry = compute_mottonen_amplitude_encoding_ry_angles(
                np.asarray(amplitudes)
            ).tolist()
            results = (
                exe.sample(seeded_executor, shots=_SHOTS, bindings={"ry_angles": ry})
                .result()
                .results
            )
            expected_probs = _normalize(amplitudes) ** 2
            observed_probs = np.zeros_like(expected_probs)
            for bits, count in results:
                observed_probs[_bits_to_index(bits)] = count / _SHOTS
            for i, p_exp in enumerate(expected_probs):
                assert abs(observed_probs[i] - p_exp) < _shot_noise_tolerance(
                    p_exp, _SHOTS
                ), f"amps={amplitudes} bin {i}: got {observed_probs[i]:.4f}"

    @pytest.mark.parametrize(
        "amplitudes, pauli, qubit_idx, expected",
        [pytest.param(amps, p, q, e, id=name) for name, amps, p, q, e in _EXPVAL_CASES],
    )
    def test_expval_via_runtime_parameters(
        self,
        qiskit_transpiler,
        amplitudes: list[float] | list[complex],
        pauli: str,
        qubit_idx: int,
        expected: float,
    ) -> None:
        """``<H>`` matches the analytic value when angles + H are runtime bindings."""
        n_qubits = int(round(np.log2(len(amplitudes))))
        is_complex = any(isinstance(a, complex) and a.imag != 0 for a in amplitudes)
        H = _pad_observable(n_qubits, _make_pauli(pauli, qubit_idx))
        ry = compute_mottonen_amplitude_encoding_ry_angles(amplitudes).tolist()

        if is_complex:
            rz = compute_mottonen_amplitude_encoding_rz_angles(amplitudes).tolist()
            kernel = _build_parametric_complex_expval_kernel(n_qubits)
            exe = qiskit_transpiler.transpile(
                kernel,
                parameters=["ry_angles", "rz_angles"],
                bindings={"H": H},
            )
            result = exe.run(
                qiskit_transpiler.executor(),
                bindings={"ry_angles": ry, "rz_angles": rz},
            ).result()
        else:
            kernel = _build_parametric_real_expval_kernel(n_qubits)
            exe = qiskit_transpiler.transpile(
                kernel, parameters=["ry_angles"], bindings={"H": H}
            )
            result = exe.run(
                qiskit_transpiler.executor(), bindings={"ry_angles": ry}
            ).result()

        assert float(result) == pytest.approx(expected, abs=1e-8)


@pytest.mark.quri_parts
class TestParametricEncodingQuriParts:
    """Parametric path on the QuriParts backend (sampler + expval)."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        pytest.importorskip("quri_parts")
        pytest.importorskip("quri_parts.qulacs")
        from qamomile.quri_parts.transpiler import QuriPartsTranspiler

        self.transpiler = QuriPartsTranspiler()

    @pytest.mark.parametrize(
        "amplitudes",
        [pytest.param(amps, id=name) for name, amps in _FIXED_AMPLITUDES]
        + [pytest.param(amps, id=name) for name, amps in _FIXED_COMPLEX_AMPLITUDES],
    )
    def test_runtime_parametric_sampler(
        self, amplitudes: list[float] | list[complex]
    ) -> None:
        """Sampler histogram matches |amplitudes|^2 when angles are bound at run time."""
        n = int(round(np.log2(len(amplitudes))))
        is_complex = any(isinstance(a, complex) and a.imag != 0 for a in amplitudes)
        ry = compute_mottonen_amplitude_encoding_ry_angles(amplitudes).tolist()
        if is_complex:
            rz = compute_mottonen_amplitude_encoding_rz_angles(amplitudes).tolist()
            kernel = _build_parametric_complex_kernel(n)
            exe = self.transpiler.transpile(
                kernel, parameters=["ry_angles", "rz_angles"]
            )
            results = (
                exe.sample(
                    self.transpiler.executor(),
                    shots=_SHOTS,
                    bindings={"ry_angles": ry, "rz_angles": rz},
                )
                .result()
                .results
            )
        else:
            kernel = _build_parametric_real_kernel(n)
            exe = self.transpiler.transpile(kernel, parameters=["ry_angles"])
            results = (
                exe.sample(
                    self.transpiler.executor(),
                    shots=_SHOTS,
                    bindings={"ry_angles": ry},
                )
                .result()
                .results
            )

        expected_probs = np.abs(_normalize(amplitudes)) ** 2
        observed_probs = np.zeros_like(expected_probs)
        total = 0
        for bits, count in results:
            observed_probs[_bits_to_index(bits)] = count / _SHOTS
            total += count
        assert total == _SHOTS
        for i, p_exp in enumerate(expected_probs):
            assert abs(observed_probs[i] - p_exp) < _shot_noise_tolerance(
                p_exp, _SHOTS
            ), f"bin {i}: got {observed_probs[i]:.4f}, expected {p_exp:.4f}"

    @pytest.mark.parametrize(
        "amplitudes, pauli, qubit_idx, expected",
        [pytest.param(amps, p, q, e, id=name) for name, amps, p, q, e in _EXPVAL_CASES],
    )
    def test_runtime_parametric_expval(
        self,
        amplitudes: list[float] | list[complex],
        pauli: str,
        qubit_idx: int,
        expected: float,
    ) -> None:
        """``<H>`` on the encoded state matches the analytic value (runtime params)."""
        n_qubits = int(round(np.log2(len(amplitudes))))
        is_complex = any(isinstance(a, complex) and a.imag != 0 for a in amplitudes)
        H = _pad_observable(n_qubits, _make_pauli(pauli, qubit_idx))
        ry = compute_mottonen_amplitude_encoding_ry_angles(amplitudes).tolist()

        if is_complex:
            rz = compute_mottonen_amplitude_encoding_rz_angles(amplitudes).tolist()
            kernel = _build_parametric_complex_expval_kernel(n_qubits)
            exe = self.transpiler.transpile(
                kernel,
                parameters=["ry_angles", "rz_angles"],
                bindings={"H": H},
            )
            result = exe.run(
                self.transpiler.executor(),
                bindings={"ry_angles": ry, "rz_angles": rz},
            ).result()
        else:
            kernel = _build_parametric_real_expval_kernel(n_qubits)
            exe = self.transpiler.transpile(
                kernel, parameters=["ry_angles"], bindings={"H": H}
            )
            result = exe.run(
                self.transpiler.executor(), bindings={"ry_angles": ry}
            ).result()

        assert float(result) == pytest.approx(expected, abs=1e-8)


@pytest.mark.cudaq
class TestParametricEncodingCudaq:
    """Parametric path on the CUDA-Q backend (sampler + expval)."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        pytest.importorskip("cudaq")
        from qamomile.cudaq.transpiler import CudaqTranspiler

        self.transpiler = CudaqTranspiler()

    @pytest.mark.parametrize(
        "amplitudes",
        [pytest.param(amps, id=name) for name, amps in _FIXED_AMPLITUDES]
        + [pytest.param(amps, id=name) for name, amps in _FIXED_COMPLEX_AMPLITUDES],
    )
    def test_runtime_parametric_sampler(
        self, amplitudes: list[float] | list[complex]
    ) -> None:
        """Sampler histogram matches |amplitudes|^2 when angles are bound at run time."""
        n = int(round(np.log2(len(amplitudes))))
        is_complex = any(isinstance(a, complex) and a.imag != 0 for a in amplitudes)
        ry = compute_mottonen_amplitude_encoding_ry_angles(amplitudes).tolist()
        if is_complex:
            rz = compute_mottonen_amplitude_encoding_rz_angles(amplitudes).tolist()
            kernel = _build_parametric_complex_kernel(n)
            exe = self.transpiler.transpile(
                kernel, parameters=["ry_angles", "rz_angles"]
            )
            results = (
                exe.sample(
                    self.transpiler.executor(),
                    shots=_SHOTS,
                    bindings={"ry_angles": ry, "rz_angles": rz},
                )
                .result()
                .results
            )
        else:
            kernel = _build_parametric_real_kernel(n)
            exe = self.transpiler.transpile(kernel, parameters=["ry_angles"])
            results = (
                exe.sample(
                    self.transpiler.executor(),
                    shots=_SHOTS,
                    bindings={"ry_angles": ry},
                )
                .result()
                .results
            )

        expected_probs = np.abs(_normalize(amplitudes)) ** 2
        observed_probs = np.zeros_like(expected_probs)
        total = 0
        for bits, count in results:
            observed_probs[_bits_to_index(bits)] = count / _SHOTS
            total += count
        assert total == _SHOTS
        for i, p_exp in enumerate(expected_probs):
            assert abs(observed_probs[i] - p_exp) < _shot_noise_tolerance(
                p_exp, _SHOTS
            ), f"bin {i}: got {observed_probs[i]:.4f}, expected {p_exp:.4f}"

    @pytest.mark.parametrize(
        "amplitudes, pauli, qubit_idx, expected",
        [pytest.param(amps, p, q, e, id=name) for name, amps, p, q, e in _EXPVAL_CASES],
    )
    def test_runtime_parametric_expval(
        self,
        amplitudes: list[float] | list[complex],
        pauli: str,
        qubit_idx: int,
        expected: float,
    ) -> None:
        """``<H>`` on the encoded state matches the analytic value (runtime params)."""
        n_qubits = int(round(np.log2(len(amplitudes))))
        is_complex = any(isinstance(a, complex) and a.imag != 0 for a in amplitudes)
        H = _pad_observable(n_qubits, _make_pauli(pauli, qubit_idx))
        ry = compute_mottonen_amplitude_encoding_ry_angles(amplitudes).tolist()

        if is_complex:
            rz = compute_mottonen_amplitude_encoding_rz_angles(amplitudes).tolist()
            kernel = _build_parametric_complex_expval_kernel(n_qubits)
            exe = self.transpiler.transpile(
                kernel,
                parameters=["ry_angles", "rz_angles"],
                bindings={"H": H},
            )
            result = exe.run(
                self.transpiler.executor(),
                bindings={"ry_angles": ry, "rz_angles": rz},
            ).result()
        else:
            kernel = _build_parametric_real_expval_kernel(n_qubits)
            exe = self.transpiler.transpile(
                kernel, parameters=["ry_angles"], bindings={"H": H}
            )
            result = exe.run(
                self.transpiler.executor(), bindings={"ry_angles": ry}
            ).result()

        assert float(result) == pytest.approx(expected, abs=1e-8)
