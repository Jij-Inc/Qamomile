"""End-to-end tests: amplitude encoding → HHL → statevector verification.

Verifies the full pipeline:
  1. Prepare |b⟩ using Möttönen amplitude encoding
  2. Run HHL circuit (QPE → reciprocal rotation → inverse QPE)
  3. Extract post-selected system state from statevector
  4. Compare with theoretical A⁻¹|b⟩

Test cases:
  - P(θ) unitary (one invertible eigenstate)
  - Rz(α) unitary (two invertible eigenstates, both components survive)
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
# Unitary kernels: P(theta) — eigenvalues 1 (|0⟩) and e^{iθ} (|1⟩)
#   Only |1⟩ has a nonzero decoded eigenvalue.
# ---------------------------------------------------------------------------


@qmc.qkernel
def _p_u(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    return qmc.p(q, theta)


@qmc.qkernel
def _p_u_inv(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    return qmc.p(q, -1.0 * theta)


# ---------------------------------------------------------------------------
# Unitary kernels: Rz(alpha) — eigenvalues e^{-iα/2} (|0⟩) and e^{iα/2} (|1⟩)
#   Both eigenstates have nonzero decoded eigenvalue.
# ---------------------------------------------------------------------------


@qmc.qkernel
def _rz_u(q: qmc.Qubit, alpha: qmc.Float) -> qmc.Qubit:
    return qmc.rz(q, alpha)


@qmc.qkernel
def _rz_u_inv(q: qmc.Qubit, alpha: qmc.Float) -> qmc.Qubit:
    return qmc.rz(q, -1.0 * alpha)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_postselected_system(
    sv: np.ndarray,
    n_system: int,
    n_clock: int,
) -> np.ndarray:
    """Extract system amplitudes conditioned on ancilla=|1⟩ and clock=|00...0⟩.

    Assumes qubit allocation order: sys[0..n_sys-1], clock[0..n_clk-1], anc.
    Qiskit little-endian: statevector index = q0 + q1*2 + q2*4 + ...
    """
    n_total = n_system + n_clock + 1
    assert len(sv) == 2**n_total

    anc_pos = n_system + n_clock  # ancilla is last allocated qubit

    system_amps = np.zeros(2**n_system, dtype=complex)

    for idx in range(len(sv)):
        # Require ancilla = 1
        if not ((idx >> anc_pos) & 1):
            continue
        # Require all clock qubits = 0
        clock_zero = True
        for c in range(n_clock):
            if (idx >> (n_system + c)) & 1:
                clock_zero = False
                break
        if not clock_zero:
            continue
        # Extract system basis state
        sys_val = idx & ((1 << n_system) - 1)
        system_amps[sys_val] = sv[idx]

    return system_amps


def _fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """State fidelity |⟨a|b⟩|² (phase-invariant, handles normalization)."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return float(np.abs(np.vdot(a / na, b / nb)) ** 2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHHLEndToEnd:
    """End-to-end: amplitude_encoding |b⟩ → HHL → verify A⁻¹|b⟩."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.transpiler = QiskitTranspiler()

    def _simulate(self, kernel, bindings=None):
        """Transpile, simulate, return (circuit, statevector)."""
        kw = {"bindings": bindings} if bindings else {}
        exe = self.transpiler.transpile(kernel, **kw)
        qc = exe.compiled_quantum[0].circuit
        return qc, run_statevector(qc)

    # -- Qubit ordering sanity check --

    def test_qubit_ordering(self) -> None:
        """Verify sys[0] maps to Qiskit qubit index 0."""

        @qmc.qkernel
        def check() -> qmc.Bit:
            sys = qmc.qubit_array(1, name="sys")
            sys[0] = qmc.x(sys[0])  # |1⟩
            clock = qmc.qubit_array(2, name="clock")
            anc = qmc.qubit("anc")
            return qmc.measure(anc)

        _, sv = self._simulate(check)
        # 4 qubits → 16 amplitudes
        # |1⟩_sys ⊗ |00⟩_clock ⊗ |0⟩_anc → index = 1 (only bit 0 set)
        assert sv.shape == (16,), f"Unexpected shape {sv.shape}"
        assert np.isclose(abs(sv[1]), 1.0, atol=1e-10), (
            f"Expected all amplitude at index 1, got |sv|={np.abs(sv)}"
        )

    # -- P(θ) tests: one invertible eigenstate --

    def test_p_gate_eigenstate(self) -> None:
        """P(π) with |b⟩ = |1⟩ (the only invertible eigenstate).

        Eigenvalue: λ_hat = π (raw=2 with 2 clock qubits, phase_scale=2π).
        Post-selected system should be |1⟩.
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

        _, sv = self._simulate(circuit, {"theta": math.pi})
        sys_amps = _extract_postselected_system(sv, 1, 2)
        norm = np.linalg.norm(sys_amps)
        assert norm > 1e-10, "Zero post-selection probability"

        expected = np.array([0.0, 1.0])
        f = _fidelity(sys_amps, expected)
        assert np.isclose(f, 1.0, atol=1e-6), (
            f"fidelity={f}, sys={sys_amps / norm}"
        )

    def test_p_gate_superposition_b(self) -> None:
        """P(π) with |b⟩ = (3/5)|0⟩ + (4/5)|1⟩.

        Only the |1⟩ component has nonzero eigenvalue.
        Post-selected system should be |1⟩ regardless of the |0⟩ overlap.
        """

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            sys = qmc.qubit_array(1, name="sys")
            sys = amplitude_encoding(sys, [3.0, 4.0])
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

        _, sv = self._simulate(circuit, {"theta": math.pi})
        sys_amps = _extract_postselected_system(sv, 1, 2)
        norm = np.linalg.norm(sys_amps)
        assert norm > 1e-10, "Zero post-selection probability"

        expected = np.array([0.0, 1.0])
        f = _fidelity(sys_amps, expected)
        assert np.isclose(f, 1.0, atol=1e-6), (
            f"fidelity={f}, sys={sys_amps / norm}"
        )

    # -- Rz(α) tests: two invertible eigenstates --
    #
    # Rz(α) eigenvalues: e^{-iα/2} (|0⟩), e^{iα/2} (|1⟩)
    #
    # With α = π and 2 clock qubits (phase_scale = 2π, unsigned):
    #   |0⟩: eigenphase = -1/4 mod 1 = 3/4  →  raw = 3  →  λ_hat = 3π/2
    #   |1⟩: eigenphase =  1/4          →  raw = 1  →  λ_hat = π/2
    #
    # A⁻¹|b⟩ ∝ (b₀ / (3π/2)) |0⟩ + (b₁ / (π/2)) |1⟩
    #         ∝ (b₀ / 3) |0⟩ + b₁ |1⟩

    @pytest.mark.parametrize(
        "amplitudes, expected_unnorm",
        [
            pytest.param(
                [1.0, 1.0],
                [1.0 / 3, 1.0],
                id="uniform",
            ),
            pytest.param(
                [1.0, 2.0],
                [1.0 / 3, 2.0],
                id="1-2",
            ),
            pytest.param(
                [0.0, 1.0],
                [0.0, 1.0],
                id="basis-1",
            ),
            pytest.param(
                [1.0, 0.0],
                [1.0, 0.0],
                id="basis-0",
            ),
        ],
    )
    def test_rz_gate_amplitude_encoded(
        self,
        amplitudes: list[float],
        expected_unnorm: list[float],
    ) -> None:
        """Rz(π) with amplitude-encoded |b⟩. Both eigenstates invertible."""
        alpha_val = math.pi
        C = 0.4  # well within |C/λ| ≤ 1 for both eigenvalues

        @qmc.qkernel
        def circuit(alpha: qmc.Float) -> qmc.Bit:
            sys = qmc.qubit_array(1, name="sys")
            sys = amplitude_encoding(sys, amplitudes)
            clock = qmc.qubit_array(2, name="clock")
            anc = qmc.qubit("anc")
            sys, clock, anc = hhl(
                sys,
                clock,
                anc,
                unitary=_rz_u,
                inv_unitary=_rz_u_inv,
                scaling=C,
                phase_scale=2.0 * math.pi,
                supported_raw_bins=(1, 3),
                strict=True,

                alpha=alpha,
            )
            return qmc.measure(anc)

        _, sv = self._simulate(circuit, {"alpha": alpha_val})
        sys_amps = _extract_postselected_system(sv, 1, 2)
        norm = np.linalg.norm(sys_amps)
        assert norm > 1e-10, (
            f"Zero post-selection probability for amplitudes={amplitudes}"
        )

        expected = np.array(expected_unnorm, dtype=float)
        f = _fidelity(sys_amps, expected)
        assert np.isclose(f, 1.0, atol=1e-4), (
            f"amplitudes={amplitudes}: fidelity={f}, "
            f"got {sys_amps / norm}, expected {expected / np.linalg.norm(expected)}"
        )

    def test_rz_postselection_probability(self) -> None:
        """Verify the ancilla post-selection probability for Rz(π).

        For |b⟩ = (1/√2)(|0⟩+|1⟩) with C=0.4 and phase_scale=2π:
          P(anc=1) = |b₀|² × (C/λ₀)² + |b₁|² × (C/λ₁)²
                   = 0.5 × (0.4/(3π/2))² + 0.5 × (0.4/(π/2))²
        """
        alpha_val = math.pi
        C = 0.4

        @qmc.qkernel
        def circuit(alpha: qmc.Float) -> qmc.Bit:
            sys = qmc.qubit_array(1, name="sys")
            sys = amplitude_encoding(sys, [1.0, 1.0])
            clock = qmc.qubit_array(2, name="clock")
            anc = qmc.qubit("anc")
            sys, clock, anc = hhl(
                sys,
                clock,
                anc,
                unitary=_rz_u,
                inv_unitary=_rz_u_inv,
                scaling=C,
                phase_scale=2.0 * math.pi,
                supported_raw_bins=(1, 3),
                strict=True,

                alpha=alpha,
            )
            return qmc.measure(anc)

        _, sv = self._simulate(circuit, {"alpha": alpha_val})
        sys_amps = _extract_postselected_system(sv, 1, 2)
        p_post = float(np.linalg.norm(sys_amps) ** 2)

        lambda0 = 3 * math.pi / 2  # raw=3
        lambda1 = math.pi / 2  # raw=1
        p_expected = 0.5 * (C / lambda0) ** 2 + 0.5 * (C / lambda1) ** 2

        assert np.isclose(p_post, p_expected, rtol=1e-4), (
            f"Post-selection probability: got {p_post}, expected {p_expected}"
        )
