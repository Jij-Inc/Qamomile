"""Tests for Möttönen amplitude encoding (state preparation)."""

from __future__ import annotations

import numpy as np
import pytest

import qamomile.circuit as qm
from qamomile.circuit.algorithm.mottonen_amplitude_encoding import (
    MottonenAmplitudeEncoding,
    amplitude_encoding,
    compute_mottonen_thetas,
)
from qamomile.circuit.ir.operation.composite_gate import CompositeGateType
from tests.circuit.conftest import run_statevector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_encoding_circuit(amplitudes: list[float], qiskit_transpiler):
    """Build a concrete amplitude-encoding circuit (with measurements)."""
    n_qubits = int(np.log2(len(amplitudes)))

    @qm.qkernel
    def _kernel() -> qm.Vector[qm.Bit]:
        q = qm.qubit_array(n_qubits, "q")
        q = amplitude_encoding(q, amplitudes)
        return qm.measure(q)

    return qiskit_transpiler.to_circuit(_kernel)


def _build_parametric_circuit(
    n_qubits: int, thetas_val: list[float], qiskit_transpiler
):
    """Build a parametric amplitude-encoding circuit (with measurements)."""

    @qm.qkernel
    def _kernel(t: qm.Vector[qm.Float]) -> qm.Vector[qm.Bit]:
        q = qm.qubit_array(n_qubits, "q")
        q = qm.amplitude_encoding(q, t)
        return qm.measure(q)

    executor = qiskit_transpiler.transpile(_kernel, bindings={"t": thetas_val})
    return executor.compiled_quantum[0].circuit


def _assert_amplitudes_match(qc, n_qubits: int, amplitudes: list[float]) -> None:
    """Simulate *qc* and verify the encoded amplitudes match *amplitudes*."""
    sv = run_statevector(qc)
    encoded = sv[: 2**n_qubits]
    expected = np.array(amplitudes, dtype=float)
    expected /= np.linalg.norm(expected)
    fidelity = np.abs(np.vdot(expected, encoded)) ** 2
    assert np.isclose(fidelity, 1.0, atol=1e-8), (
        f"amplitudes={amplitudes}: fidelity={fidelity}, got {encoded}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# Unit tests: MottonenAmplitudeEncoding class
# ---------------------------------------------------------------------------


class TestMottonenAmplitudeEncoding:
    def test_gate_metadata(self) -> None:
        gate = MottonenAmplitudeEncoding([1.0, 0.0])
        assert gate.custom_name == "mottonen_amplitude_encoding"
        assert gate.gate_type == CompositeGateType.CUSTOM

    @pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
    def test_resources(self, n_qubits: int) -> None:
        amplitudes = np.ones(2**n_qubits)
        gate = MottonenAmplitudeEncoding(amplitudes)
        assert gate.num_target_qubits == n_qubits
        r = gate._resources()
        m = r.custom_metadata
        assert m["num_ry_gates"] == 2**n_qubits - 1
        assert m["num_cnot_gates"] == 2**n_qubits - 2
        assert m["num_qubits"] == n_qubits

    @pytest.mark.parametrize(
        "amplitudes, match",
        [
            pytest.param([1.0, 0.0, 0.0], "power of 2", id="non-power-of-2"),
            pytest.param([], "power of 2", id="empty"),
            pytest.param([0.0, 0.0], "all zeros", id="all-zeros"),
        ],
    )
    def test_invalid_amplitudes(self, amplitudes: list[float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            MottonenAmplitudeEncoding(amplitudes)


# ---------------------------------------------------------------------------
# Unit tests: compute_mottonen_thetas
# ---------------------------------------------------------------------------


class TestComputeMottonenThetas:
    @pytest.mark.parametrize("n_qubits", [1, 2, 3])
    def test_output_length(self, n_qubits: int) -> None:
        amplitudes = np.ones(2**n_qubits)
        thetas = compute_mottonen_thetas(amplitudes)
        assert len(thetas) == 2**n_qubits - 1

    @pytest.mark.parametrize(
        "amplitudes, match",
        [
            pytest.param([1.0, 0.0, 0.0], "power of 2", id="non-power-of-2"),
            pytest.param([0.0, 0.0], "all zeros", id="all-zeros"),
        ],
    )
    def test_invalid_amplitudes(self, amplitudes: list[float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            compute_mottonen_thetas(amplitudes)


# ---------------------------------------------------------------------------
# Statevector correctness tests (concrete encoding)
# ---------------------------------------------------------------------------


class TestConcreteEncoding:
    @pytest.mark.parametrize(
        "amplitudes",
        [
            pytest.param([1.0, 0.0], id="1q-basis-0"),
            pytest.param([0.0, 1.0], id="1q-basis-1"),
            pytest.param([1.0, 1.0], id="1q-plus"),
            pytest.param([1.0, 0.0, 0.0, 0.0], id="2q-00"),
            pytest.param([0.0, 0.0, 0.0, 1.0], id="2q-11"),
            pytest.param([1.0, 1.0, 1.0, 1.0], id="2q-uniform"),
            pytest.param([1.0, 0.0, 0.0, 1.0], id="2q-bell-like"),
            pytest.param([3.0, 4.0], id="1q-normalization"),
        ],
    )
    def test_fixed_amplitudes(self, qiskit_transpiler, amplitudes: list[float]) -> None:
        n_qubits = int(np.log2(len(amplitudes)))
        qc = _build_encoding_circuit(amplitudes, qiskit_transpiler)
        _assert_amplitudes_match(qc, n_qubits, amplitudes)

    @pytest.mark.parametrize("n_qubits", [1, 2, 3])
    def test_random_amplitudes(self, qiskit_transpiler, n_qubits: int) -> None:
        rng = np.random.default_rng(42)
        for _ in range(5):
            amplitudes = rng.standard_normal(2**n_qubits).tolist()
            qc = _build_encoding_circuit(amplitudes, qiskit_transpiler)
            _assert_amplitudes_match(qc, n_qubits, amplitudes)

    def test_qubit_mismatch_raises(self) -> None:
        """Qubit count / amplitude count mismatch raises ValueError."""

        @qm.qkernel
        def circuit() -> qm.Vector[qm.Bit]:
            q = qm.qubit_array(2, "q")
            # 8 amplitudes need 3 qubits, but only 2 provided
            q = amplitude_encoding(q, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            return qm.measure(q)

        with pytest.raises(ValueError, match="qubits"):
            circuit.build()

    def test_amplitude_encoding_factory(self, qiskit_transpiler) -> None:
        """Test the qm.amplitude_encoding public API."""
        amplitudes = [1.0, 0.0, 0.0, 1.0]

        @qm.qkernel
        def _kernel() -> qm.Vector[qm.Bit]:
            q = qm.qubit_array(2, "q")
            q = qm.amplitude_encoding(q, amplitudes)
            return qm.measure(q)

        qc = qiskit_transpiler.to_circuit(_kernel)
        _assert_amplitudes_match(qc, 2, amplitudes)


# ---------------------------------------------------------------------------
# Parametric amplitude encoding tests
# ---------------------------------------------------------------------------


class TestParametricEncoding:
    @pytest.mark.parametrize(
        "amplitudes",
        [
            pytest.param([1.0, 0.0], id="1q-basis-0"),
            pytest.param([0.0, 1.0], id="1q-basis-1"),
            pytest.param([1.0, 1.0], id="1q-plus"),
            pytest.param([1.0, 0.0, 0.0, 0.0], id="2q-00"),
            pytest.param([0.0, 0.0, 0.0, 1.0], id="2q-11"),
            pytest.param([1.0, 1.0, 1.0, 1.0], id="2q-uniform"),
            pytest.param([1.0, 0.0, 0.0, 1.0], id="2q-bell-like"),
        ],
    )
    def test_fixed_amplitudes(self, qiskit_transpiler, amplitudes: list[float]) -> None:
        n_qubits = int(np.log2(len(amplitudes)))
        thetas = compute_mottonen_thetas(amplitudes)
        qc = _build_parametric_circuit(n_qubits, thetas.tolist(), qiskit_transpiler)
        _assert_amplitudes_match(qc, n_qubits, amplitudes)

    @pytest.mark.parametrize("n_qubits", [1, 2, 3])
    def test_random_amplitudes(self, qiskit_transpiler, n_qubits: int) -> None:
        rng = np.random.default_rng(123)
        for _ in range(3):
            amplitudes = rng.standard_normal(2**n_qubits).tolist()
            thetas = compute_mottonen_thetas(amplitudes)
            qc = _build_parametric_circuit(n_qubits, thetas.tolist(), qiskit_transpiler)
            _assert_amplitudes_match(qc, n_qubits, amplitudes)
