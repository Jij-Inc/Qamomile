"""HHL transpile tests — QURI Parts backend.

Mirrors the Qiskit test suite using QURI Parts transpiler + Qulacs
for statevector simulation.  Automatically skipped if quri_parts is
not installed.
"""

import math

import numpy as np
import pytest

pytestmark = pytest.mark.quri_parts

pytest.importorskip("quri_parts")
pytest.importorskip("quri_parts.qulacs")

import qamomile.circuit as qmc  # noqa: E402
from qamomile.circuit.algorithm.hhl import hhl  # noqa: E402
from qamomile.circuit.algorithm.mottonen_amplitude_encoding import (  # noqa: E402
    amplitude_encoding,
)
from qamomile.quri_parts.transpiler import QuriPartsTranspiler  # noqa: E402
from tests.circuit.algorithm.test_hhl_transpile import (  # noqa: E402
    _p_u,
    _p_u_inv,
    extract_postselected_system,
    fidelity,
    make_rz_hhl_kernel,
    random_rz_problem,
)


def _run_statevector_quri(circuit) -> np.ndarray:
    """Extract statevector from a QURI Parts circuit using Qulacs."""
    from quri_parts.core.state import GeneralCircuitQuantumState
    from quri_parts.qulacs.simulator import evaluate_state_to_vector

    if hasattr(circuit, "parameter_count") and circuit.parameter_count > 0:
        bound_circuit = circuit.bind_parameters(
            [0.0] * circuit.parameter_count
        )
    elif hasattr(circuit, "bind_parameters"):
        bound_circuit = circuit.bind_parameters([])
    else:
        bound_circuit = circuit

    circuit_state = GeneralCircuitQuantumState(
        bound_circuit.qubit_count, bound_circuit
    )
    statevector = evaluate_state_to_vector(circuit_state)
    return np.array(statevector.vector)


class TestHHLTranspileQuriParts:
    """HHL transpile tests using QURI Parts + Qulacs statevector."""

    def _simulate(self, kernel, bindings=None):
        """Transpile with QuriPartsTranspiler and extract statevector."""
        kw = {"bindings": bindings} if bindings else {}
        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(kernel, **kw)
        circuit = exe.compiled_quantum[0].circuit
        return _run_statevector_quri(circuit)

    # -- Fixed tests: Rz(pi), 2 clock qubits --

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

    # -- P gate test --

    def test_p_gate_eigenstate(self) -> None:
        """P(pi) with |b> = |1>: only the invertible eigenstate survives."""

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
        assert np.isclose(f, 1.0, atol=1e-6), (
            f"fidelity={f}, sys={sys_amps / norm}"
        )

    # -- Post-selection probability --

    def test_post_selection_probability(self) -> None:
        """Verify ancilla post-selection probability matches theory."""
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
