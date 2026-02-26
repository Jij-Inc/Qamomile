"""Tests for Möttönen amplitude encoding (state preparation)."""

from __future__ import annotations

import numpy as np
import pytest

import qamomile.circuit as qm
from qamomile.circuit.algorithm.mottonen_amplitude_encoding import (
    MottonenAmplitudeEncoding,
    compute_mottonen_thetas,
)
from qamomile.circuit.ir.operation.composite_gate import CompositeGateType

# ---------------------------------------------------------------------------
# Unit tests for class attributes and resources
# ---------------------------------------------------------------------------


class TestMottonenAmplitudeEncodingAttributes:
    def test_custom_name(self) -> None:
        gate = MottonenAmplitudeEncoding([1.0, 0.0])
        assert gate.custom_name == "mottonen_amplitude_encoding"

    def test_gate_type(self) -> None:
        gate = MottonenAmplitudeEncoding([1.0, 0.0])
        assert gate.gate_type == CompositeGateType.CUSTOM

    @pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
    def test_num_target_qubits(self, n_qubits: int) -> None:
        amplitudes = np.ones(2**n_qubits)
        gate = MottonenAmplitudeEncoding(amplitudes)
        assert gate.num_target_qubits == n_qubits

    def test_invalid_length(self) -> None:
        with pytest.raises(ValueError, match="power of 2"):
            MottonenAmplitudeEncoding([1.0, 0.0, 0.0])

    def test_empty_amplitudes(self) -> None:
        with pytest.raises(ValueError, match="power of 2"):
            MottonenAmplitudeEncoding([])

    def test_zero_amplitudes(self) -> None:
        with pytest.raises(ValueError, match="all zeros"):
            MottonenAmplitudeEncoding([0.0, 0.0])


def test_amplitude_encoding_qubit_mismatch() -> None:
    """Qubit count / amplitude count mismatch raises ValueError."""
    from qamomile.circuit.algorithm.mottonen_amplitude_encoding import amplitude_encoding

    @qm.qkernel
    def circuit() -> qm.Vector[qm.Bit]:
        q = qm.qubit_array(2, "q")
        # 8 amplitudes need 3 qubits, but only 2 provided
        q = amplitude_encoding(q, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return qm.measure(q)

    with pytest.raises(ValueError, match="qubits"):
        circuit.build()


class TestMottonenAmplitudeEncodingResources:
    @pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
    def test_resources(self, n_qubits: int) -> None:
        amplitudes = np.ones(2**n_qubits)
        r = MottonenAmplitudeEncoding(amplitudes)._resources()
        m = r.custom_metadata
        assert m["num_ry_gates"] == 2**n_qubits - 1
        assert m["num_cnot_gates"] == 2**n_qubits - 2
        assert m["num_qubits"] == n_qubits


# ---------------------------------------------------------------------------
# Statevector correctness tests via Qamomile pipeline + Qiskit simulation
# ---------------------------------------------------------------------------


def _make_encoding_kernel(n_qubits: int, amplitudes: list[float]):
    """Create an amplitude encoding qkernel using the factory function."""
    from qamomile.circuit.algorithm.mottonen_amplitude_encoding import amplitude_encoding

    @qm.qkernel
    def _kernel() -> qm.Vector[qm.Bit]:
        q = qm.qubit_array(n_qubits, "q")
        q = amplitude_encoding(q, amplitudes)
        return qm.measure(q)

    return _kernel


def _build_encoding_circuit(amplitudes: list[float]):
    """Transpile encoding kernel via Qamomile pipeline.

    Returns the circuit with measurements removed for statevector simulation.
    """
    from qamomile.qiskit import QiskitTranspiler

    n_qubits = int(np.log2(len(amplitudes)))
    kernel = _make_encoding_kernel(n_qubits, amplitudes)
    transpiler = QiskitTranspiler()
    qc = transpiler.to_circuit(kernel)
    qc.remove_final_measurements()
    return qc


def _simulate_statevector(qc):
    """Simulate circuit and return statevector as numpy array."""
    from qiskit_aer import AerSimulator

    qc.save_statevector()
    simulator = AerSimulator(method="statevector")
    result = simulator.run(qc).result()
    return np.array(result.get_statevector())


def _extract_encoded_amplitudes(sv: np.ndarray, n_qubits: int) -> np.ndarray:
    """Extract the amplitudes on the first n_qubits from the statevector.

    The CUSTOM CompositeGate emit path may allocate extra unused qubits
    that remain in |0>. The encoded state lives in the first 2^n_qubits
    basis states (indices 0..2^n_qubits-1) of the full statevector.
    """
    return sv[: 2**n_qubits]


@pytest.mark.parametrize(
    "amplitudes",
    [
        [1.0, 0.0],  # |0>
        [0.0, 1.0],  # |1>
        [1.0, 1.0],  # |+>
    ],
)
def test_1qubit_states(amplitudes: list[float]) -> None:
    """Test 1-qubit amplitude encoding."""
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")

    qc = _build_encoding_circuit(amplitudes)
    sv = _simulate_statevector(qc)
    encoded = _extract_encoded_amplitudes(sv, 1)

    expected = np.array(amplitudes, dtype=float)
    expected = expected / np.linalg.norm(expected)

    assert np.allclose(np.abs(encoded), np.abs(expected), atol=1e-8), (
        f"amplitudes={amplitudes}: got {encoded}, expected {expected}"
    )


@pytest.mark.parametrize(
    "amplitudes",
    [
        [1.0, 0.0, 0.0, 0.0],  # |00>
        [0.0, 0.0, 0.0, 1.0],  # |11>
        [1.0, 1.0, 1.0, 1.0],  # uniform
        [1.0, 0.0, 0.0, 1.0],  # Bell-like
    ],
)
def test_2qubit_states(amplitudes: list[float]) -> None:
    """Test 2-qubit amplitude encoding."""
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")

    qc = _build_encoding_circuit(amplitudes)
    sv = _simulate_statevector(qc)
    n_qubits = int(np.log2(len(amplitudes)))
    encoded = _extract_encoded_amplitudes(sv, n_qubits)

    expected = np.array(amplitudes, dtype=float)
    expected = expected / np.linalg.norm(expected)

    assert np.allclose(np.abs(encoded), np.abs(expected), atol=1e-8), (
        f"amplitudes={amplitudes}: got {encoded}, expected {expected}"
    )


@pytest.mark.parametrize("n_qubits", [1, 2, 3])
def test_random_amplitudes(n_qubits: int) -> None:
    """Test with random amplitudes for various qubit counts."""
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")

    rng = np.random.default_rng(42)
    for _ in range(5):
        amplitudes = rng.standard_normal(2**n_qubits).tolist()
        qc = _build_encoding_circuit(amplitudes)
        sv = _simulate_statevector(qc)
        encoded = _extract_encoded_amplitudes(sv, n_qubits)

        expected = np.array(amplitudes, dtype=float)
        expected = expected / np.linalg.norm(expected)

        assert np.allclose(np.abs(encoded), np.abs(expected), atol=1e-8), (
            f"n_qubits={n_qubits}: statevector mismatch"
        )


def test_normalization() -> None:
    """Non-normalized input should be automatically normalized."""
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")

    amplitudes = [3.0, 4.0]  # norm = 5
    qc = _build_encoding_circuit(amplitudes)
    sv = _simulate_statevector(qc)
    encoded = _extract_encoded_amplitudes(sv, 1)

    expected = np.array([0.6, 0.8])
    assert np.allclose(np.abs(encoded), np.abs(expected), atol=1e-8)


def test_amplitude_encoding_factory() -> None:
    """Test the amplitude_encoding factory function."""
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")

    amplitudes = [1.0, 0.0, 0.0, 1.0]

    @qm.qkernel
    def _kernel() -> qm.Vector[qm.Bit]:
        q = qm.qubit_array(2, "q")
        q = qm.amplitude_encoding(q, amplitudes)
        return qm.measure(q)

    from qamomile.qiskit import QiskitTranspiler

    transpiler = QiskitTranspiler()
    qc = transpiler.to_circuit(_kernel)
    qc.remove_final_measurements()
    sv = _simulate_statevector(qc)
    encoded = _extract_encoded_amplitudes(sv, 2)

    expected = np.array(amplitudes, dtype=float)
    expected = expected / np.linalg.norm(expected)

    assert np.allclose(np.abs(encoded), np.abs(expected), atol=1e-8)


# ---------------------------------------------------------------------------
# compute_mottonen_thetas unit tests
# ---------------------------------------------------------------------------


class TestComputeMottonenThetas:
    @pytest.mark.parametrize("n_qubits", [1, 2, 3])
    def test_output_length(self, n_qubits: int) -> None:
        amplitudes = np.ones(2**n_qubits)
        thetas = compute_mottonen_thetas(amplitudes)
        assert len(thetas) == 2**n_qubits - 1

    def test_invalid_length(self) -> None:
        with pytest.raises(ValueError, match="power of 2"):
            compute_mottonen_thetas([1.0, 0.0, 0.0])

    def test_zero_amplitudes(self) -> None:
        with pytest.raises(ValueError, match="all zeros"):
            compute_mottonen_thetas([0.0, 0.0])


# ---------------------------------------------------------------------------
# Parametric amplitude encoding tests
# ---------------------------------------------------------------------------


def _build_parametric_circuit(n_qubits: int, thetas_val: list[float]):
    """Build a parametric encoding circuit via transpiler."""
    from qamomile.qiskit import QiskitTranspiler

    @qm.qkernel
    def _kernel(t: qm.Vector[qm.Float]) -> qm.Vector[qm.Bit]:
        q = qm.qubit_array(n_qubits, "q")
        q = qm.amplitude_encoding(q, t)
        return qm.measure(q)

    transpiler = QiskitTranspiler()
    executor = transpiler.transpile(_kernel, bindings={"t": thetas_val})
    qc = executor.compiled_quantum[0].circuit
    qc.remove_final_measurements()
    return qc


@pytest.mark.parametrize(
    "amplitudes",
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ],
)
def test_parametric_1qubit(amplitudes: list[float]) -> None:
    """Parametric encoding matches amplitude_encoding for 1-qubit states."""
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")

    thetas = compute_mottonen_thetas(amplitudes)
    qc = _build_parametric_circuit(1, thetas.tolist())
    sv = _simulate_statevector(qc)
    encoded = _extract_encoded_amplitudes(sv, 1)

    expected = np.array(amplitudes, dtype=float)
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(np.abs(encoded), np.abs(expected), atol=1e-8)


@pytest.mark.parametrize(
    "amplitudes",
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
    ],
)
def test_parametric_2qubit(amplitudes: list[float]) -> None:
    """Parametric encoding matches amplitude_encoding for 2-qubit states."""
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")

    thetas = compute_mottonen_thetas(amplitudes)
    qc = _build_parametric_circuit(2, thetas.tolist())
    sv = _simulate_statevector(qc)
    encoded = _extract_encoded_amplitudes(sv, 2)

    expected = np.array(amplitudes, dtype=float)
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(np.abs(encoded), np.abs(expected), atol=1e-8)


@pytest.mark.parametrize("n_qubits", [1, 2, 3])
def test_parametric_random(n_qubits: int) -> None:
    """Parametric encoding with random amplitudes."""
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")

    rng = np.random.default_rng(123)
    for _ in range(3):
        amplitudes = rng.standard_normal(2**n_qubits).tolist()
        thetas = compute_mottonen_thetas(amplitudes)
        qc = _build_parametric_circuit(n_qubits, thetas.tolist())
        sv = _simulate_statevector(qc)
        encoded = _extract_encoded_amplitudes(sv, n_qubits)

        expected = np.array(amplitudes, dtype=float)
        expected = expected / np.linalg.norm(expected)
        assert np.allclose(np.abs(encoded), np.abs(expected), atol=1e-8)
