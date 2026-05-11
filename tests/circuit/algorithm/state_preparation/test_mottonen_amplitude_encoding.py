"""Tests for Möttönen amplitude encoding."""

from __future__ import annotations

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.state_preparation import (
    MottonenAmplitudeEncoding,
    amplitude_encoding,
    compute_mottonen_amplitude_encoding_thetas,
)
from qamomile.circuit.ir.operation.composite_gate import CompositeGateType
from tests.circuit.conftest import run_statevector

# Catalogue of amplitudes shared by every backend test.  Each entry is a list
# of (id, amplitudes) — keeping it module-level lets all parametrisations
# stay aligned.
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

_SEEDS = [901, 902, 903, 904, 905]
_QUBIT_COUNTS = [1, 2, 3, 4]

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
_EXPVAL_CASES: list[tuple[str, list[float], str, int, float]] = [
    # (id,        amplitudes,                 pauli, qubit_index, expected)
    ("asymm-Z0", [1.0, 2.0, 3.0, 4.0], "Z", 0, -1.0 / 3.0),
    ("asymm-Z1", [1.0, 2.0, 3.0, 4.0], "Z", 1, -2.0 / 3.0),
    ("signed-Z0", [1.0, -1.0, 1.0, -1.0], "Z", 0, 0.0),
    ("signed-X0", [1.0, -1.0, 1.0, -1.0], "X", 0, -1.0),
]


def _normalize(amps: list[float]) -> np.ndarray:
    """Return *amps* as a unit-norm float array.

    Args:
        amps (list[float]): Real amplitudes (any non-zero norm).

    Returns:
        np.ndarray: Array of dtype ``float`` with unit Euclidean norm.
    """
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
    """Pad a single-qubit Pauli to *num_qubits* width with a zero-coeff identity.

    Several backend emit paths require the ``Hamiltonian.num_qubits`` to
    match the register width; tacking on a zero-weighted term on the
    highest-index qubit is the standard trick used elsewhere in the test
    suite.

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
    def test_resources(self, n_qubits: int) -> None:
        """Reported gate counts match the Gray-walk decomposition formulas."""
        gate = MottonenAmplitudeEncoding(np.ones(2**n_qubits))
        assert gate.num_target_qubits == n_qubits
        r = gate._resources()
        meta = r.custom_metadata
        assert meta["num_ry_gates"] == 2**n_qubits - 1
        assert meta["num_cnot_gates"] == 2**n_qubits - 2
        assert meta["num_qubits"] == n_qubits
        assert r.t_gates == 0
        assert r.total_gates == (2**n_qubits - 1) + (2**n_qubits - 2)
        assert r.single_qubit_gates == 2**n_qubits - 1
        assert r.two_qubit_gates == 2**n_qubits - 2


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
        thetas = compute_mottonen_amplitude_encoding_thetas(np.ones(2**n_qubits))
        np.testing.assert_allclose(thetas, expected, atol=1e-12)

    def test_length_invariants(self) -> None:
        """Output length matches ``2**n - 1`` for every supported size."""
        for n in _QUBIT_COUNTS:
            thetas = compute_mottonen_amplitude_encoding_thetas(np.ones(2**n))
            assert thetas.shape == (2**n - 1,)


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
            ([0.0, 0.0], "all zeros"),
        ],
    )
    def test_invalid_amplitudes_raise(
        self, amplitudes: list[float], match: str
    ) -> None:
        """Non-power-of-two, empty, and all-zero inputs are rejected."""
        with pytest.raises(ValueError, match=match):
            MottonenAmplitudeEncoding(amplitudes)

    def test_complex_amplitudes_rejected(self) -> None:
        """Complex inputs with non-zero imaginary part raise ``TypeError``."""
        with pytest.raises(TypeError, match="real amplitudes"):
            compute_mottonen_amplitude_encoding_thetas(
                np.array([1.0 + 1j, 0.0], dtype=complex)
            )

    def test_complex_with_zero_imag_accepted(self) -> None:
        """Complex inputs with zero imaginary part coerce cleanly to the real angle.

        ``[1+0j, 0+0j]`` normalises to ``[1, 0]`` so the single Gray-walk
        angle is ``2 * arctan2(0, 1) = 0``.  Asserting the value (not just
        the shape) catches regressions in the complex-to-real coercion.
        """
        thetas = compute_mottonen_amplitude_encoding_thetas(
            np.array([1.0 + 0j, 0.0 + 0j], dtype=complex)
        )
        np.testing.assert_allclose(thetas, [0.0], atol=1e-12)

    def test_qubit_count_mismatch_raises(self) -> None:
        """``amplitude_encoding`` rejects qubit / amplitude mismatches."""

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q = amplitude_encoding(q, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            return qmc.measure(q)

        with pytest.raises(ValueError, match="qubits"):
            kernel.build()


# ---------------------------------------------------------------------------
# Kernel builders shared by every backend
# ---------------------------------------------------------------------------


def _build_measure_kernel(amplitudes: list[float]) -> qmc.QKernel:
    """Build a kernel that prepares *amplitudes* and measures the register.

    Args:
        amplitudes (list[float]): Real amplitude vector of length
            ``2**n``.  Drives the register size
            (``n = log2(len(amplitudes))``).

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


def _build_expval_kernel(amplitudes: list[float]) -> qmc.QKernel:
    """Build a kernel that returns ``<H>`` on the prepared state.

    Args:
        amplitudes (list[float]): Real amplitude vector of length
            ``2**n``.  Drives the register size
            (``n = log2(len(amplitudes))``).

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
        """Each catalogued amplitude vector is faithfully encoded."""
        qc = qiskit_transpiler.to_circuit(_build_measure_kernel(amplitudes))
        sv = run_statevector(qc).real
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

    @pytest.mark.parametrize(
        "amplitudes",
        [pytest.param(amps, id=name) for name, amps in _FIXED_AMPLITUDES],
    )
    def test_fixed_amplitudes_sampler(
        self,
        qiskit_transpiler,
        seeded_executor,
        amplitudes: list[float],
    ) -> None:
        """Empirical histogram matches |amplitudes|^2 within shot noise."""
        kernel = _build_measure_kernel(amplitudes)
        exe = qiskit_transpiler.transpile(kernel)
        results = exe.sample(seeded_executor, shots=_SHOTS).result().results

        expected_probs = _normalize(amplitudes) ** 2
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
        amplitudes: list[float],
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
        [pytest.param(amps, id=name) for name, amps in _FIXED_AMPLITUDES],
    )
    def test_fixed_amplitudes_sampler(self, amplitudes: list[float]) -> None:
        """Empirical histogram matches |amplitudes|^2 within shot noise."""
        kernel = _build_measure_kernel(amplitudes)
        exe = self.transpiler.transpile(kernel)
        results = exe.sample(self.transpiler.executor(), shots=_SHOTS).result().results

        expected_probs = _normalize(amplitudes) ** 2
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
        amplitudes: list[float],
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
        [pytest.param(amps, id=name) for name, amps in _FIXED_AMPLITUDES],
    )
    def test_fixed_amplitudes_sampler(self, amplitudes: list[float]) -> None:
        """Empirical histogram matches |amplitudes|^2 within shot noise."""
        kernel = _build_measure_kernel(amplitudes)
        exe = self.transpiler.transpile(kernel)
        results = exe.sample(self.transpiler.executor(), shots=_SHOTS).result().results

        expected_probs = _normalize(amplitudes) ** 2
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
        amplitudes: list[float],
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
