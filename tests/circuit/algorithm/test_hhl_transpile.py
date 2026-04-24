"""HHL transpile tests — Qiskit backend + shared helpers for other backends.

Tests:
  - Fixed Rz(π) problems with hand-computed A⁻¹|b⟩ (2 clock qubits)
  - P(π) eigenstate test (only invertible eigenstate survives)
  - Post-selection probability verification
  - Random Rz problems verified against numpy (2–3 clock qubits, 100 seeds)

Shared helpers (imported by CUDA-Q and QURI Parts test modules):
  - extract_postselected_system, fidelity
  - compute_rz_qpe_bins, compute_exact_ainv_b, random_rz_problem
  - make_rz_hhl_kernel
  - Module-level qkernels: _rz_u, _rz_u_inv, _p_u, _p_u_inv
"""

import math

import numpy as np
import pytest

pytest.importorskip("qiskit")

import qamomile.circuit as qmc
from qamomile.circuit.algorithm.hhl import hhl
from qamomile.circuit.algorithm.mottonen_amplitude_encoding import amplitude_encoding
from qamomile.qiskit.transpiler import QiskitTranspiler
from tests.circuit.conftest import run_statevector

# ---------------------------------------------------------------------------
# Module-level qkernels (no `from __future__ import annotations`)
# ---------------------------------------------------------------------------


@qmc.qkernel
def _rz_u(q: qmc.Qubit, alpha: qmc.Float) -> qmc.Qubit:
    """Rz(alpha) unitary."""
    return qmc.rz(q, alpha)


@qmc.qkernel
def _rz_u_inv(q: qmc.Qubit, alpha: qmc.Float) -> qmc.Qubit:
    """Rz(-alpha) — adjoint of Rz(alpha)."""
    return qmc.rz(q, -1.0 * alpha)


@qmc.qkernel
def _p_u(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Phase gate P(theta)."""
    return qmc.p(q, theta)


@qmc.qkernel
def _p_u_inv(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Adjoint of P(theta)."""
    return qmc.p(q, -1.0 * theta)


# ---------------------------------------------------------------------------
# Shared helpers — imported by CUDA-Q / QURI Parts test modules
# ---------------------------------------------------------------------------


def extract_postselected_system(
    sv: np.ndarray,
    n_system: int,
    n_clock: int,
) -> np.ndarray:
    """Extract system amplitudes conditioned on ancilla=|1> and clock=|00...0>.

    Qubit allocation order: sys[0..n_sys-1], clock[0..n_clk-1], anc.
    Little-endian: statevector index = q0 + q1*2 + q2*4 + ...
    """
    n_total = n_system + n_clock + 1
    assert len(sv) == 2**n_total

    anc_pos = n_system + n_clock

    system_amps = np.zeros(2**n_system, dtype=complex)

    for idx in range(len(sv)):
        if not ((idx >> anc_pos) & 1):
            continue
        clock_zero = True
        for c in range(n_clock):
            if (idx >> (n_system + c)) & 1:
                clock_zero = False
                break
        if not clock_zero:
            continue
        sys_val = idx & ((1 << n_system) - 1)
        system_amps[sys_val] = sv[idx]

    return system_amps


def fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """State fidelity |<a|b>|^2 (phase-invariant, handles normalization)."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return float(np.abs(np.vdot(a / na, b / nb)) ** 2)


def compute_rz_qpe_bins(
    alpha: float,
    n_clock: int,
    phase_scale: float = 2.0 * math.pi,
) -> tuple[tuple[int, int], tuple[float, float]]:
    """Compute QPE bins and decoded eigenvalues for Rz(alpha).

    Rz(alpha) eigenvalues:
        |0>: e^{-i alpha/2}  ->  eigenphase = -alpha / (4 pi)  mod 1
        |1>: e^{+i alpha/2}  ->  eigenphase = +alpha / (4 pi)

    Returns:
        bins:  (raw_0, raw_1) integer QPE outcomes
        eigenvalues:  (lambda_0, lambda_1) decoded eigenvalue estimates
    """
    N = 2**n_clock
    phi_0 = -alpha / (4.0 * math.pi)
    phi_1 = alpha / (4.0 * math.pi)
    raw_0 = round(phi_0 * N) % N
    raw_1 = round(phi_1 * N) % N
    lambda_0 = phase_scale * raw_0 / N
    lambda_1 = phase_scale * raw_1 / N
    return (raw_0, raw_1), (lambda_0, lambda_1)


def compute_exact_ainv_b(
    b_amps: list[float],
    eigenvalues: tuple[float, float],
) -> np.ndarray:
    """Compute A^{-1}|b> from input amplitudes and decoded eigenvalues.

    Returns unnormalized amplitudes of the solution state.
    """
    b = np.array(b_amps, dtype=float)
    norm = np.linalg.norm(b)
    if norm < 1e-15:
        return np.zeros(2)
    b_norm = b / norm
    result = np.zeros(2)
    for i in range(2):
        lam = eigenvalues[i]
        if abs(lam) > 1e-15:
            result[i] = b_norm[i] / lam
    return result


def random_rz_problem(seed: int):
    """Generate a random Rz HHL problem with exact QPE resolution.

    For exact QPE with m clock qubits, alpha must satisfy:
        alpha / (4 pi) * 2^m is an integer,
    i.e. alpha = k * pi / 2^(m-2) for integer k.

    Returns:
        alpha, b_amps, C, n_clock, supported_bins, eigenvalues, exact
    """
    rng = np.random.default_rng(seed)
    n_clock = int(rng.choice([2, 3]))

    if n_clock == 2:
        alpha = math.pi  # only non-trivial exact choice
    else:
        alpha = float(rng.choice([math.pi / 2, math.pi, 3 * math.pi / 2]))

    # Random positive amplitudes (avoid near-zero)
    b_raw = rng.uniform(0.1, 3.0, size=2)

    bins, eigenvalues = compute_rz_qpe_bins(alpha, n_clock)
    supported_bins = tuple(b for b in bins if b != 0)

    # Choose C so that |C / lambda_hat| <= 1 for all populated eigenvalues
    min_abs_lam = min(abs(lam) for lam in eigenvalues if abs(lam) > 1e-15)
    C = 0.9 * min_abs_lam

    exact = compute_exact_ainv_b(b_raw.tolist(), eigenvalues)

    return alpha, b_raw.tolist(), C, n_clock, supported_bins, eigenvalues, exact


def make_rz_hhl_kernel(
    b_amps: list[float],
    n_clock: int,
    C: float,
    supported_bins: tuple[int, ...],
):
    """Create a @qmc.qkernel for Rz HHL with the given parameters."""

    @qmc.qkernel
    def circuit(alpha: qmc.Float) -> qmc.Bit:
        sys = qmc.qubit_array(1, name="sys")
        sys = amplitude_encoding(sys, b_amps)
        clock = qmc.qubit_array(n_clock, name="clock")
        anc = qmc.qubit("anc")
        sys, clock, anc = hhl(
            sys,
            clock,
            anc,
            unitary=_rz_u,
            inv_unitary=_rz_u_inv,
            scaling=C,
            phase_scale=2.0 * math.pi,
            supported_raw_bins=supported_bins,
            strict=True,
            alpha=alpha,
        )
        return qmc.measure(anc)

    return circuit


# ---------------------------------------------------------------------------
# Qiskit tests
# ---------------------------------------------------------------------------


class TestHHLTranspileQiskit:
    """HHL transpile tests using Qiskit + AerSimulator statevector."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.transpiler = QiskitTranspiler()

    def _simulate(self, kernel, bindings=None):
        """Transpile and run statevector simulation."""
        kw = {"bindings": bindings} if bindings else {}
        exe = self.transpiler.transpile(kernel, **kw)
        qc = exe.compiled_quantum[0].circuit
        return run_statevector(qc)

    # -- Fixed tests: Rz(pi) with 2 clock qubits, hand-computed answers --
    #
    # Rz(pi) eigenvalues with 2 clock qubits (phase_scale=2pi, unsigned):
    #   |0>: eigenphase = -1/4 mod 1 = 3/4 -> raw=3 -> lambda_0 = 3*pi/2
    #   |1>: eigenphase = 1/4              -> raw=1 -> lambda_1 = pi/2
    #
    # A^{-1}|b> proportional to  (b_0 / (3*pi/2)) |0> + (b_1 / (pi/2)) |1>
    #                          = (b_0 / 3) |0> + b_1 |1>

    @pytest.mark.parametrize(
        "amplitudes, expected_unnorm",
        [
            pytest.param([1.0, 1.0], [1.0 / 3, 1.0], id="uniform"),
            pytest.param([1.0, 2.0], [1.0 / 3, 2.0], id="1-2"),
            pytest.param([0.0, 1.0], [0.0, 1.0], id="basis-1"),
            pytest.param([1.0, 0.0], [1.0, 0.0], id="basis-0"),
            pytest.param([3.0, 4.0], [1.0, 4.0], id="3-4"),
        ],
    )
    def test_rz_pi_fixed(
        self,
        amplitudes: list[float],
        expected_unnorm: list[float],
    ) -> None:
        """Rz(pi) with 2 clock qubits — hand-computed A^{-1}|b>."""
        C = 0.4
        kernel = make_rz_hhl_kernel(amplitudes, 2, C, (1, 3))
        sv = self._simulate(kernel, {"alpha": math.pi})
        sys_amps = extract_postselected_system(sv, 1, 2)
        norm = np.linalg.norm(sys_amps)
        assert norm > 1e-10, (
            f"Zero post-selection probability for amplitudes={amplitudes}"
        )

        expected = np.array(expected_unnorm, dtype=float)
        f = fidelity(sys_amps, expected)
        assert np.isclose(f, 1.0, atol=1e-4), (
            f"amplitudes={amplitudes}: fidelity={f}, "
            f"got {sys_amps / norm}, "
            f"expected {expected / np.linalg.norm(expected)}"
        )

    # -- P gate test: only the invertible eigenstate survives --

    def test_p_gate_eigenstate(self) -> None:
        """P(pi) with |b> = |1>: only the invertible eigenstate survives.

        P(theta) eigenvalues: 1 (|0>, raw=0, not invertible),
                              e^{i*theta} (|1>, raw=2 for theta=pi, 2 clocks).
        """

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            sys = qmc.qubit_array(1, name="sys")
            sys = amplitude_encoding(sys, [0.0, 1.0])
            clock = qmc.qubit_array(2, name="clock")
            anc = qmc.qubit("anc")
            sys, clock, anc = hhl(
                sys,
                clock,
                anc,
                unitary=_p_u,
                inv_unitary=_p_u_inv,
                scaling=0.25,
                phase_scale=2.0 * math.pi,
                supported_raw_bins=(2,),
                strict=True,
                theta=theta,
            )
            return qmc.measure(anc)

        sv = self._simulate(circuit, {"theta": math.pi})
        sys_amps = extract_postselected_system(sv, 1, 2)
        norm = np.linalg.norm(sys_amps)
        assert norm > 1e-10, "Zero post-selection probability"

        expected = np.array([0.0, 1.0])
        f = fidelity(sys_amps, expected)
        assert np.isclose(f, 1.0, atol=1e-6), f"fidelity={f}, sys={sys_amps / norm}"

    # -- Post-selection probability --

    def test_post_selection_probability(self) -> None:
        """Verify ancilla post-selection probability matches theory.

        For |b> = (1/sqrt(2))(|0>+|1>) with C=0.4 and phase_scale=2pi:
          P(anc=1) = |b_0|^2 (C/lambda_0)^2 + |b_1|^2 (C/lambda_1)^2
                   = 0.5 * (0.4 / (3pi/2))^2 + 0.5 * (0.4 / (pi/2))^2
        """
        C = 0.4
        kernel = make_rz_hhl_kernel([1.0, 1.0], 2, C, (1, 3))
        sv = self._simulate(kernel, {"alpha": math.pi})
        sys_amps = extract_postselected_system(sv, 1, 2)
        p_post = float(np.linalg.norm(sys_amps) ** 2)

        lambda0 = 3.0 * math.pi / 2
        lambda1 = math.pi / 2
        p_expected = 0.5 * (C / lambda0) ** 2 + 0.5 * (C / lambda1) ** 2

        assert np.isclose(p_post, p_expected, rtol=1e-4), (
            f"Post-selection probability: got {p_post}, expected {p_expected}"
        )

    # -- Random tests: 100 seeds --

    @pytest.mark.parametrize("seed", [901 + i for i in range(100)])
    def test_rz_random(self, seed: int) -> None:
        """Random Rz HHL problem verified against numpy exact solution."""
        alpha, b_amps, C, n_clock, supported_bins, eigenvalues, exact = (
            random_rz_problem(seed)
        )

        kernel = make_rz_hhl_kernel(b_amps, n_clock, C, supported_bins)
        sv = self._simulate(kernel, {"alpha": alpha})
        sys_amps = extract_postselected_system(sv, 1, n_clock)
        norm = np.linalg.norm(sys_amps)
        assert norm > 1e-10, (
            f"seed={seed}: zero post-selection probability "
            f"(alpha={alpha:.4f}, b={b_amps}, C={C:.4f})"
        )

        f = fidelity(sys_amps, exact)
        assert np.isclose(f, 1.0, atol=1e-3), (
            f"seed={seed}: fidelity={f}, "
            f"alpha={alpha:.4f}, b={b_amps}, n_clock={n_clock}, "
            f"got {sys_amps / norm}, "
            f"expected {exact / np.linalg.norm(exact)}"
        )
