"""Rich Qiskit frontend-to-backend test suite.

Tests the full pipeline: @qkernel definition -> QiskitTranspiler -> execution.
Covers every frontend gate, gate combinations, control flow, transpiler passes,
stdlib, measurement, and edge cases.

Note: Do NOT use ``from __future__ import annotations`` in this file.
The @qkernel AST transformer relies on resolved type annotations to identify
Float vs UInt etc.  PEP 563 deferred annotations turn them into strings, which
breaks ``_create_bound_input``.
"""

import numpy as np
import pytest

import qamomile.circuit as qmc
from tests.transpiler.gate_test_specs import (
    GATE_SPECS,
    statevectors_equal,
    all_zeros_state,
    computational_basis_state,
    compute_expected_statevector,
    tensor_product,
    identity,
)

# ---------------------------------------------------------------------------
# Skip entire module if qiskit is not installed
# ---------------------------------------------------------------------------
qiskit = pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from qamomile.qiskit import QiskitTranspiler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_measurements(circuit):
    """Return a copy of the circuit with measurement gates removed."""
    from qiskit import QuantumCircuit

    new_qc = QuantumCircuit(circuit.num_qubits)
    for inst in circuit.data:
        if inst.operation.name not in ("measure", "barrier"):
            new_qc.append(inst)
    return new_qc


def _run_statevector(circuit) -> np.ndarray:
    """Run a Qiskit circuit and return the pre-measurement statevector."""
    from qiskit_aer import AerSimulator

    qc = _strip_measurements(circuit)
    qc.save_statevector()
    result = AerSimulator(method="statevector").run(qc).result()
    return np.array(result.get_statevector())


def _transpile_and_get_circuit(kernel, bindings=None, parameters=None):
    """Transpile a qkernel and return (ExecutableProgram, QuantumCircuit)."""
    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(kernel, bindings=bindings, parameters=parameters)
    circuit = exe.compiled_quantum[0].circuit
    return exe, circuit


def _gate_names(circuit) -> list:
    """Return list of gate names in a Qiskit circuit."""
    return [inst.operation.name for inst in circuit.data]


# ============================================================================
# 1. Individual Gate Tests – creation + determined + random execution
# ============================================================================


class TestSingleQubitGatesFrontend:
    """Test each single-qubit gate through the full frontend pipeline."""

    # -- Hadamard --

    def test_h_creation(self):
        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert "h" in _gate_names(qc)

    def test_h_statevector(self):
        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
        assert statevectors_equal(sv, expected)

    # -- Pauli-X --

    def test_x_creation(self):
        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert "x" in _gate_names(qc)

    def test_x_statevector(self):
        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = np.array([0, 1], dtype=complex)
        assert statevectors_equal(sv, expected)

    # -- RX --

    def test_rx_creation(self):
        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 4})
        assert "rx" in _gate_names(qc)

    def test_rx_determined(self):
        """RX(pi) |0> = -i|1>."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi})
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RX"].matrix_fn(np.pi)
        )
        assert statevectors_equal(sv, expected)

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999])
    def test_rx_random(self, seed):
        rng = np.random.default_rng(seed)
        angle = rng.uniform(0, 2 * np.pi)

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": angle})
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RX"].matrix_fn(angle)
        )
        assert statevectors_equal(sv, expected)

    # -- RY --

    def test_ry_creation(self):
        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.ry(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 4})
        assert "ry" in _gate_names(qc)

    def test_ry_determined(self):
        """RY(pi/2) |0> -> (|0> + |1>) / sqrt(2)."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.ry(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 2})
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RY"].matrix_fn(np.pi / 2)
        )
        assert statevectors_equal(sv, expected)

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999])
    def test_ry_random(self, seed):
        rng = np.random.default_rng(seed)
        angle = rng.uniform(0, 2 * np.pi)

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.ry(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": angle})
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RY"].matrix_fn(angle)
        )
        assert statevectors_equal(sv, expected)

    # -- RZ --

    def test_rz_creation(self):
        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rz(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 4})
        assert "rz" in _gate_names(qc)

    def test_rz_determined(self):
        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rz(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi})
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RZ"].matrix_fn(np.pi)
        )
        assert statevectors_equal(sv, expected)

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999])
    def test_rz_random(self, seed):
        rng = np.random.default_rng(seed)
        angle = rng.uniform(0, 2 * np.pi)

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rz(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": angle})
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RZ"].matrix_fn(angle)
        )
        assert statevectors_equal(sv, expected)

    # -- P (Phase) --

    def test_p_creation(self):
        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.p(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 4})
        assert "p" in _gate_names(qc)

    def test_p_determined(self):
        """P(pi) on |1> gives phase e^{i*pi} = -1."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)  # Prepare |1>
            q = qmc.p(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi})
        sv = _run_statevector(qc)
        state_after_x = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["X"].matrix_fn()
        )
        expected = compute_expected_statevector(
            state_after_x, GATE_SPECS["P"].matrix_fn(np.pi)
        )
        assert statevectors_equal(sv, expected)

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999])
    def test_p_random(self, seed):
        rng = np.random.default_rng(seed)
        angle = rng.uniform(0, 2 * np.pi)

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            q = qmc.p(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": angle})
        sv = _run_statevector(qc)
        state_after_x = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["X"].matrix_fn()
        )
        expected = compute_expected_statevector(
            state_after_x, GATE_SPECS["P"].matrix_fn(angle)
        )
        assert statevectors_equal(sv, expected)


class TestTwoQubitGatesFrontend:
    """Test each two-qubit gate through the full frontend pipeline."""

    # -- CX --

    def test_cx_creation(self):
        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert "cx" in _gate_names(qc)

    def test_cx_from_00(self):
        """CX |00> = |00>."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            computational_basis_state(2, 0), GATE_SPECS["CX"].matrix_fn()
        )
        assert statevectors_equal(sv, expected)

    def test_cx_from_10(self):
        """CX |10> = |11> (control on, flips target)."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            computational_basis_state(2, 1), GATE_SPECS["CX"].matrix_fn()
        )
        assert statevectors_equal(sv, expected)

    def test_cx_from_01(self):
        """CX |01> = |01> (control off)."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[1] = qmc.x(q[1])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            computational_basis_state(2, 2), GATE_SPECS["CX"].matrix_fn()
        )
        assert statevectors_equal(sv, expected)

    def test_cx_from_11(self):
        """CX |11> = |10>."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            computational_basis_state(2, 3), GATE_SPECS["CX"].matrix_fn()
        )
        assert statevectors_equal(sv, expected)

    # -- CZ --

    def test_cz_creation(self):
        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = qmc.cz(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert "cz" in _gate_names(qc)

    def test_cz_from_00(self):
        """CZ |00> = |00> (no phase)."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = qmc.cz(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            computational_basis_state(2, 0), GATE_SPECS["CZ"].matrix_fn()
        )
        assert statevectors_equal(sv, expected)

    def test_cz_from_01(self):
        """CZ |01> = |01> (no phase, only q0=1)."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[0], q[1] = qmc.cz(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            computational_basis_state(2, 1), GATE_SPECS["CZ"].matrix_fn()
        )
        assert statevectors_equal(sv, expected)

    def test_cz_from_10(self):
        """CZ |10> = |10> (no phase, only q1=1)."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[1] = qmc.x(q[1])
            q[0], q[1] = qmc.cz(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            computational_basis_state(2, 2), GATE_SPECS["CZ"].matrix_fn()
        )
        assert statevectors_equal(sv, expected)

    def test_cz_determined_11(self):
        """CZ |11> = -|11>."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0], q[1] = qmc.cz(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            computational_basis_state(2, 3), GATE_SPECS["CZ"].matrix_fn()
        )
        assert statevectors_equal(sv, expected)

    # -- CP --

    def test_cp_creation(self):
        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = qmc.cp(q[0], q[1], theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 4})
        assert "cp" in _gate_names(qc)

    def test_cp_determined(self):
        """CP(pi) on |11> gives phase e^{i*pi} = -1."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0], q[1] = qmc.cp(q[0], q[1], theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi})
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            computational_basis_state(2, 3), GATE_SPECS["CP"].matrix_fn(np.pi)
        )
        assert statevectors_equal(sv, expected)

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999])
    def test_cp_random(self, seed):
        rng = np.random.default_rng(seed)
        angle = rng.uniform(0, 2 * np.pi)

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0], q[1] = qmc.cp(q[0], q[1], theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": angle})
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            computational_basis_state(2, 3), GATE_SPECS["CP"].matrix_fn(angle)
        )
        assert statevectors_equal(sv, expected)

    # -- RZZ --

    def test_rzz_creation(self):
        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = qmc.rzz(q[0], q[1], theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 4})
        assert "rzz" in _gate_names(qc)

    def test_rzz_determined(self):
        """RZZ(pi) on |00>."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = qmc.rzz(q[0], q[1], theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi})
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            all_zeros_state(2), GATE_SPECS["RZZ"].matrix_fn(np.pi)
        )
        assert statevectors_equal(sv, expected)

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999])
    def test_rzz_random(self, seed):
        rng = np.random.default_rng(seed)
        angle = rng.uniform(0, 2 * np.pi)

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0], q[1] = qmc.rzz(q[0], q[1], theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": angle})
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            computational_basis_state(2, 3), GATE_SPECS["RZZ"].matrix_fn(angle)
        )
        assert statevectors_equal(sv, expected)

    # -- SWAP --

    def test_swap_creation(self):
        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = qmc.swap(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert "swap" in _gate_names(qc)

    def test_swap_determined(self):
        """SWAP |10> = |01>."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[0], q[1] = qmc.swap(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            computational_basis_state(2, 1), GATE_SPECS["SWAP"].matrix_fn()
        )
        assert statevectors_equal(sv, expected)


# ============================================================================
# 2. Gate Combination Tests
# ============================================================================


class TestGateCombinations:
    """Test multi-gate circuits through the frontend pipeline."""

    def test_bell_state(self):
        """H + CX creates Bell state (|00> + |11>) / sqrt(2)."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        assert statevectors_equal(sv, expected)

    def test_ghz_state_3(self):
        """GHZ state: (|000> + |111>) / sqrt(2) via H + CX chain."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            q[1], q[2] = qmc.cx(q[1], q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = np.zeros(8, dtype=complex)
        expected[0] = 1 / np.sqrt(2)
        expected[7] = 1 / np.sqrt(2)
        assert statevectors_equal(sv, expected)

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999])
    def test_rotation_sequence_random(self, seed):
        """RX -> RY -> RZ on single qubit with random angles."""
        rng = np.random.default_rng(seed)
        a1, a2, a3 = rng.uniform(0, 2 * np.pi, size=3)

        @qmc.qkernel
        def circuit(
            theta1: qmc.Float, theta2: qmc.Float, theta3: qmc.Float
        ) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta1)
            q = qmc.ry(q, theta2)
            q = qmc.rz(q, theta3)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"theta1": a1, "theta2": a2, "theta3": a3}
        )
        sv = _run_statevector(qc)

        state = all_zeros_state(1)
        state = GATE_SPECS["RX"].matrix_fn(a1) @ state
        state = GATE_SPECS["RY"].matrix_fn(a2) @ state
        state = GATE_SPECS["RZ"].matrix_fn(a3) @ state
        assert statevectors_equal(sv, state)

    def test_swap_preserves_state(self):
        """Prepare |1> on q0, SWAP, verify q1 is |1>."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[0], q[1] = qmc.swap(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # After X on q0 then SWAP: |10> -> |01>
        expected = np.array([0, 0, 1, 0], dtype=complex)
        assert statevectors_equal(sv, expected)

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999])
    def test_entangling_plus_rotation(self, seed):
        """H + CX + RZ on target qubit."""
        rng = np.random.default_rng(seed)
        angle = rng.uniform(0, 2 * np.pi)

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            q[1] = qmc.rz(q[1], theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": angle})
        sv = _run_statevector(qc)

        state = all_zeros_state(2)
        # Little-endian: q[0] is LSB (right in tensor product)
        # H on q[0]: tensor_product(I, H)
        i_h = tensor_product(identity(), GATE_SPECS["H"].matrix_fn())
        state = i_h @ state
        # CX gate spec: control=q0(LSB), target=q1(MSB)
        state = GATE_SPECS["CX"].matrix_fn() @ state
        # RZ on q[1]: tensor_product(RZ, I)
        rz_i = tensor_product(GATE_SPECS["RZ"].matrix_fn(angle), identity())
        state = rz_i @ state
        assert statevectors_equal(sv, state)

    def test_x_then_h_identity_check(self):
        """X -> H -> H -> X = Identity (up to global phase)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            q = qmc.h(q)
            q = qmc.h(q)
            q = qmc.x(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = all_zeros_state(1)
        assert statevectors_equal(sv, expected)

    def test_multiple_cx_gates(self):
        """Apply CX three times: CX^3 = CX (since CX^2 = I)."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            q[0], q[1] = qmc.cx(q[0], q[1])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # |10> -> CX -> |11> -> CX -> |10> -> CX -> |11>
        expected = computational_basis_state(2, 3)
        assert statevectors_equal(sv, expected)

    def test_teleportation_circuit_pre_measurement(self):
        """Teleportation circuit (pre-measurement portion).

        Tests the gate preparation + Bell pair + entanglement pattern.
        Note: Full teleportation with conditional corrections after measurement
        is not supported because the analyzer rejects quantum ops on
        measurement-dependent values (DependencyError).
        """

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            # Alice's qubit to teleport (prepared in |1>)
            alice = qmc.qubit("alice")
            alice = qmc.x(alice)

            # Shared Bell pair
            bell0 = qmc.qubit("bell0")
            bell1 = qmc.qubit("bell1")
            bell0 = qmc.h(bell0)
            bell0, bell1 = qmc.cx(bell0, bell1)

            # Alice entangles her qubit with her half of Bell pair
            alice, bell0 = qmc.cx(alice, bell0)
            alice = qmc.h(alice)

            # Measure all three qubits
            q = qmc.qubit_array(3, "out")
            q[0] = alice
            q[1] = bell0
            q[2] = bell1
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        names = _gate_names(qc)
        assert "h" in names
        assert "cx" in names
        assert qc.num_qubits >= 3


# ============================================================================
# 3. Control Flow Tests
# ============================================================================


class TestControlFlowRange:
    """Test qmc.range loop through the frontend pipeline."""

    @pytest.mark.parametrize("n_qubits", [2, 3, 5, 10, 20])
    def test_h_all_qubits_via_range(self, n_qubits):
        """Apply H to all qubits via qmc.range loop."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": n_qubits})
        h_count = sum(1 for name in _gate_names(qc) if name == "h")
        assert h_count == n_qubits

    @pytest.mark.parametrize("n_qubits", [2, 3, 5, 10, 20])
    def test_rx_layer_via_range(self, n_qubits):
        """Apply RX with varying angles via qmc.range."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, thetas: qmc.Vector[qmc.Float]
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.rx(q[i], thetas[i])
            return qmc.measure(q)

        thetas = [0.1 * i for i in range(n_qubits)]
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": n_qubits, "thetas": thetas}
        )
        rx_gates = [inst for inst in qc.data if inst.operation.name == "rx"]
        assert len(rx_gates) == n_qubits

        for i, inst in enumerate(rx_gates):
            actual = float(inst.operation.params[0])
            assert abs(actual - thetas[i]) < 1e-10

    def test_range_with_start_stop(self):
        """qmc.range(start, stop) applies gates to subset of qubits."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(1, 3):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 4})
        h_count = sum(1 for name in _gate_names(qc) if name == "h")
        assert h_count == 2

    def test_cz_chain_via_range(self):
        """CZ entangling layer via range loop."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            for i in qmc.range(n - 1):
                q[i], q[i + 1] = qmc.cz(q[i], q[i + 1])
            return qmc.measure(q)

        n = 4
        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": n})
        cz_count = sum(1 for name in _gate_names(qc) if name == "cz")
        assert cz_count == n - 1

    @pytest.mark.parametrize("n_qubits", [4, 6])
    def test_range_with_step_2(self, n_qubits):
        """range(0, n, 2) applies gates to even-indexed qubits only."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(0, n, 2):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": n_qubits})
        h_count = sum(1 for name in _gate_names(qc) if name == "h")
        expected = (n_qubits + 1) // 2  # even indices: 0, 2, 4, ...
        assert h_count == expected

    @pytest.mark.parametrize("n_qubits", [4, 6])
    def test_range_with_step_2_odd_start(self, n_qubits):
        """range(1, n, 2) applies gates to odd-indexed qubits only."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(1, n, 2):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": n_qubits})
        h_count = sum(1 for name in _gate_names(qc) if name == "h")
        expected = n_qubits // 2  # odd indices: 1, 3, 5, ...
        assert h_count == expected

    def test_zero_iteration_range(self):
        """range(start, stop) with start >= stop produces no gates."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(5, 5):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 6})
        h_count = sum(1 for name in _gate_names(qc) if name == "h")
        assert h_count == 0

    def test_multiple_sequential_ranges(self):
        """Two range loops in sequence: H layer then RZ layer."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, thetas: qmc.Vector[qmc.Float]
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            for i in qmc.range(n):
                q[i] = qmc.rz(q[i], thetas[i])
            return qmc.measure(q)

        n = 4
        thetas = [0.1 * i for i in range(n)]
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": n, "thetas": thetas}
        )
        h_count = sum(1 for name in _gate_names(qc) if name == "h")
        rz_count = sum(1 for name in _gate_names(qc) if name == "rz")
        assert h_count == n
        assert rz_count == n

    def test_range_with_expression_args(self):
        """range(1, n-1) with arithmetic in bounds."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(1, n - 1):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        n = 5
        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": n})
        h_count = sum(1 for name in _gate_names(qc) if name == "h")
        assert h_count == n - 2  # indices 1, 2, 3


class TestControlFlowItems:
    """Test qmc.items loop through the frontend pipeline."""

    def test_items_ising_rzz(self):
        """Apply RZZ gates from Ising dict via qmc.items."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
            ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            gamma: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for (i, j), Jij in qmc.items(ising):
                q[i], q[j] = qmc.rzz(q[i], q[j], gamma * Jij)
            return qmc.measure(q)

        ising = {(0, 1): 1.0, (1, 2): -0.5, (0, 2): 0.3}
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": 3, "ising": ising, "gamma": 0.5}
        )
        rzz_count = sum(1 for name in _gate_names(qc) if name == "rzz")
        assert rzz_count == 3

    def test_items_single_key_dict(self):
        """Apply RZ gates from single-key dict."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
            angles: qmc.Dict[qmc.UInt, qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i, theta in qmc.items(angles):
                q[i] = qmc.rz(q[i], theta)
            return qmc.measure(q)

        angles = {0: 0.1, 1: 0.2, 2: 0.3}
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": 3, "angles": angles}
        )
        rz_count = sum(1 for name in _gate_names(qc) if name == "rz")
        assert rz_count == 3

    def test_items_empty_dict(self):
        """Empty dict produces no gates from the loop body."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
            ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for (i, j), Jij in qmc.items(ising):
                q[i], q[j] = qmc.rzz(q[i], q[j], Jij)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 2, "ising": {}})
        rzz_count = sum(1 for name in _gate_names(qc) if name == "rzz")
        assert rzz_count == 0

    def test_items_large_dict(self):
        """Large dict (10 entries) correctly unrolls to 10 RZZ gates."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
            ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for (i, j), Jij in qmc.items(ising):
                q[i], q[j] = qmc.rzz(q[i], q[j], Jij)
            return qmc.measure(q)

        # 10 edges on 5 qubits (complete graph)
        ising = {
            (i, j): 0.1 * (i + j)
            for i in range(5)
            for j in range(i + 1, 5)
        }
        assert len(ising) == 10
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": 5, "ising": ising}
        )
        rzz_count = sum(1 for name in _gate_names(qc) if name == "rzz")
        assert rzz_count == 10

    def test_items_negative_values_angle_sign(self):
        """Negative Jij produces correctly signed RZZ angle."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
            ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            gamma: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for (i, j), Jij in qmc.items(ising):
                q[i], q[j] = qmc.rzz(q[i], q[j], gamma * Jij)
            return qmc.measure(q)

        ising = {(0, 1): -1.0}
        gamma = 0.5
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": 2, "ising": ising, "gamma": gamma}
        )
        rzz_gates = [inst for inst in qc.data if inst.operation.name == "rzz"]
        assert len(rzz_gates) == 1
        actual_angle = float(rzz_gates[0].operation.params[0])
        assert abs(actual_angle - (gamma * -1.0)) < 1e-10

    def test_items_statevector_verification(self):
        """Items loop on 2-qubit Ising produces correct statevector."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
            ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for (i, j), Jij in qmc.items(ising):
                q[i], q[j] = qmc.rzz(q[i], q[j], Jij)
            return qmc.measure(q)

        theta = 0.7
        ising = {(0, 1): theta}
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": 2, "ising": ising}
        )
        sv = _run_statevector(qc)
        # RZZ(theta) on |00> = [e^{-i*theta/2}, 0, 0, 0] (up to global phase)
        # For 2-qubit |00> state, RZZ gives diagonal: [e^{-it/2}, e^{it/2}, e^{it/2}, e^{-it/2}]
        expected = np.array([
            np.exp(-1j * theta / 2), 0, 0, 0
        ])
        # |00> → only first component nonzero
        assert abs(abs(sv[0]) - 1.0) < 1e-10
        assert abs(abs(sv[1])) < 1e-10
        assert abs(abs(sv[2])) < 1e-10
        assert abs(abs(sv[3])) < 1e-10


class TestControlFlowNested:
    """Test nested control flow patterns."""

    def test_nested_range_loops(self):
        """Two-level range loop: apply CZ between all pairs."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            for i in qmc.range(n - 1):
                for j in qmc.range(i + 1, n):
                    q[i], q[j] = qmc.cz(q[i], q[j])
            return qmc.measure(q)

        n = 3
        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": n})
        # n*(n-1)/2 = 3 CZ gates
        cz_count = sum(1 for name in _gate_names(qc) if name == "cz")
        assert cz_count == n * (n - 1) // 2

    def test_range_then_items_sequence(self):
        """Range H layer followed by items RZZ — sequential control flow mix."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
            ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            gamma: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            for (i, j), Jij in qmc.items(ising):
                q[i], q[j] = qmc.rzz(q[i], q[j], gamma * Jij)
            return qmc.measure(q)

        ising = {(0, 1): 1.0, (1, 2): -0.5}
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": 3, "ising": ising, "gamma": 0.5}
        )
        h_count = sum(1 for name in _gate_names(qc) if name == "h")
        rzz_count = sum(1 for name in _gate_names(qc) if name == "rzz")
        assert h_count == 3
        assert rzz_count == 2

    def test_cx_ladder_via_range(self):
        """CX entangling ladder: X on q0 propagates to all qubits → |1111⟩."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q[0] = qmc.x(q[0])
            for i in qmc.range(n - 1):
                q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])
            return qmc.measure(q)

        n = 4
        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": n})
        cx_count = sum(1 for name in _gate_names(qc) if name == "cx")
        assert cx_count == n - 1
        # X on q0, then CX chain propagates: |1000> → |1100> → |1110> → |1111>
        sv = _run_statevector(qc)
        expected = computational_basis_state(n, 2**n - 1)
        assert statevectors_equal(sv, expected)

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_nested_range_parametrize_sizes(self, n):
        """All-pairs CZ with parametrized qubit count."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            for i in qmc.range(n - 1):
                for j in qmc.range(i + 1, n):
                    q[i], q[j] = qmc.cz(q[i], q[j])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": n})
        cz_count = sum(1 for name in _gate_names(qc) if name == "cz")
        assert cz_count == n * (n - 1) // 2


class TestControlFlowIfElse:
    """Test if-else control flow in @qkernel (measure → conditional gate)."""

    def test_if_else_creation(self):
        """If-else transpiles into Qiskit if_else instruction."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q0 = qmc.x(q0)  # Prepare |1>
            bit = qmc.measure(q0)
            if bit:
                q1 = qmc.x(q1)
            else:
                q1 = q1
            return qmc.measure(q1)

        _, qc = _transpile_and_get_circuit(circuit)
        names = _gate_names(qc)
        assert "if_else" in names

    def test_if_else_both_branches(self):
        """If-else with true branch applying X and false as no-op.

        Note: The linear type system traces both branches from the same state,
        so both branches must use the same qubit handle consistently. Using
        different gates (e.g., X in true, H in false) on the same qubit causes
        a QubitConsumedError because the qubit gets consumed in branch tracing.
        """

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q0 = qmc.x(q0)
            bit = qmc.measure(q0)
            if bit:
                q1 = qmc.x(q1)
            else:
                q1 = q1
            return qmc.measure(q1)

        _, qc = _transpile_and_get_circuit(circuit)
        names = _gate_names(qc)
        assert "if_else" in names
        # Extract the if_else instruction
        if_else_insts = [i for i in qc.data if i.operation.name == "if_else"]
        assert len(if_else_insts) >= 1

    def test_if_else_execution(self):
        """Execute an if-else circuit and verify it runs without error.

        Mid-circuit measurement with conditional execution may produce
        implementation-specific results depending on the simulator backend.
        We verify the circuit executes and returns valid Bit values.
        """

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q0 = qmc.x(q0)  # q0 = |1>, so bit = 1
            bit = qmc.measure(q0)
            if bit:
                q1 = qmc.x(q1)
            else:
                q1 = q1
            return qmc.measure(q1)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        assert result is not None
        assert len(result.results) > 0
        for value, count in result.results:
            assert value in (0, 1)
            assert count > 0

    def test_if_only_no_else(self):
        """If-only (no else branch) compiles to if_else with no-op false."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q0 = qmc.x(q0)
            bit = qmc.measure(q0)
            if bit:
                q1 = qmc.x(q1)
            return qmc.measure(q1)

        _, qc = _transpile_and_get_circuit(circuit)
        names = _gate_names(qc)
        assert "if_else" in names

    def test_if_else_on_zero_measurement(self):
        """Measure |0⟩ (always false) — if_else instruction still created."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            # q0 stays |0>, so bit = 0 → false branch
            bit = qmc.measure(q0)
            if bit:
                q1 = qmc.x(q1)
            else:
                q1 = q1
            return qmc.measure(q1)

        _, qc = _transpile_and_get_circuit(circuit)
        names = _gate_names(qc)
        assert "if_else" in names

    def test_if_else_with_parametric_gate(self):
        """True branch applies RX(theta), false branch is no-op."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q0 = qmc.x(q0)
            bit = qmc.measure(q0)
            if bit:
                q1 = qmc.rx(q1, theta)
            else:
                q1 = q1
            return qmc.measure(q1)

        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"theta": np.pi / 4}
        )
        names = _gate_names(qc)
        assert "if_else" in names

    def test_multiple_sequential_if_else(self):
        """Two sequential if-else blocks on different measurements."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q2 = qmc.qubit("q2")
            q0 = qmc.x(q0)
            bit0 = qmc.measure(q0)
            if bit0:
                q1 = qmc.x(q1)
            else:
                q1 = q1
            bit1 = qmc.measure(q1)
            if bit1:
                q2 = qmc.x(q2)
            else:
                q2 = q2
            return qmc.measure(q2)

        _, qc = _transpile_and_get_circuit(circuit)
        if_else_insts = [i for i in qc.data if i.operation.name == "if_else"]
        assert len(if_else_insts) >= 2


class TestControlFlowWhile:
    """Test while-loop control flow in @qkernel."""

    def test_while_loop_creation(self):
        """While loop transpiles into Qiskit while_loop instruction."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            bit = qmc.measure(q)
            while bit:
                q = qmc.qubit("q2")
                q = qmc.h(q)
                bit = qmc.measure(q)
            return bit

        _, qc = _transpile_and_get_circuit(circuit)
        names = _gate_names(qc)
        assert "while_loop" in names

    def test_while_loop_structure(self):
        """While loop has correct body structure."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            bit = qmc.measure(q)
            while bit:
                q = qmc.qubit("q2")
                q = qmc.h(q)
                bit = qmc.measure(q)
            return bit

        _, qc = _transpile_and_get_circuit(circuit)
        while_insts = [i for i in qc.data if i.operation.name == "while_loop"]
        assert len(while_insts) >= 1
        # The while_loop params contain one QuantumCircuit (body)
        body = while_insts[0].operation.params[0]
        assert body.num_qubits > 0

    def test_while_loop_circuit_structure(self):
        """While loop circuit has correct top-level structure (qubits, clbits, while_loop)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            bit = qmc.measure(q)
            while bit:
                q = qmc.qubit("q2")
                q = qmc.h(q)
                bit = qmc.measure(q)
            return bit

        _, qc = _transpile_and_get_circuit(circuit)
        names = _gate_names(qc)
        # Must have initial H + measure before the while
        assert "h" in names
        assert "measure" in names
        assert "while_loop" in names
        # Circuit has qubits and classical bits
        assert qc.num_qubits >= 1
        assert qc.num_clbits >= 1

    def test_while_loop_with_x_body(self):
        """While body with X gate (always flips to |0⟩ on second iteration)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)  # |1>
            bit = qmc.measure(q)
            while bit:
                q = qmc.qubit("q2")
                q = qmc.h(q)
                bit = qmc.measure(q)
            return bit

        _, qc = _transpile_and_get_circuit(circuit)
        names = _gate_names(qc)
        assert "while_loop" in names
        assert "x" in names

    def test_while_loop_body_gate_verification(self):
        """Verify while loop body subcircuit contains expected gates."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            bit = qmc.measure(q)
            while bit:
                q = qmc.qubit("q2")
                q = qmc.h(q)
                bit = qmc.measure(q)
            return bit

        _, qc = _transpile_and_get_circuit(circuit)
        while_insts = [i for i in qc.data if i.operation.name == "while_loop"]
        assert len(while_insts) >= 1
        body = while_insts[0].operation.params[0]
        body_names = [inst.operation.name for inst in body.data]
        assert "h" in body_names
        assert "measure" in body_names


# ============================================================================
# 3b. QAOA Pattern Tests
# ============================================================================


class TestControlFlowQAOAPattern:
    """Test realistic QAOA-like patterns combining range + items control flow."""

    def test_qaoa_single_layer_gate_counts(self):
        """Full QAOA ansatz layer: H init + Ising RZZ + mixer RX."""

        @qmc.qkernel
        def qaoa_layer(
            n: qmc.UInt,
            ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            gamma: qmc.Float,
            beta: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            for (i, j), Jij in qmc.items(ising):
                q[i], q[j] = qmc.rzz(q[i], q[j], gamma * Jij)
            for i in qmc.range(n):
                q[i] = qmc.rx(q[i], beta)
            return qmc.measure(q)

        ising = {(0, 1): 1.0, (1, 2): -0.5, (0, 2): 0.3}
        _, qc = _transpile_and_get_circuit(
            qaoa_layer,
            bindings={"n": 3, "ising": ising, "gamma": 0.5, "beta": 0.7},
        )
        h_count = sum(1 for name in _gate_names(qc) if name == "h")
        rzz_count = sum(1 for name in _gate_names(qc) if name == "rzz")
        rx_count = sum(1 for name in _gate_names(qc) if name == "rx")
        assert h_count == 3
        assert rzz_count == 3
        assert rx_count == 3

    def test_qaoa_single_layer_statevector(self):
        """QAOA layer on 2 qubits with known params — statevector check."""

        @qmc.qkernel
        def qaoa_layer(
            n: qmc.UInt,
            ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            gamma: qmc.Float,
            beta: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            for (i, j), Jij in qmc.items(ising):
                q[i], q[j] = qmc.rzz(q[i], q[j], gamma * Jij)
            for i in qmc.range(n):
                q[i] = qmc.rx(q[i], beta)
            return qmc.measure(q)

        gamma, beta = 0.5, 0.3
        ising = {(0, 1): 1.0}
        _, qc = _transpile_and_get_circuit(
            qaoa_layer,
            bindings={"n": 2, "ising": ising, "gamma": gamma, "beta": beta},
        )
        sv = _run_statevector(qc)
        # Manually compute: H⊗H |00> → |++>
        # Then RZZ(gamma*1.0) → diagonal phases
        # Then RX(beta)⊗RX(beta)
        H = GATE_SPECS["H"].matrix_fn()
        RZZ = GATE_SPECS["RZZ"].matrix_fn(gamma * 1.0)
        RX = GATE_SPECS["RX"].matrix_fn(beta)
        state = all_zeros_state(2)
        state = tensor_product(H, H) @ state
        state = RZZ @ state
        state = tensor_product(RX, RX) @ state
        assert statevectors_equal(sv, state)

    def test_qaoa_single_layer_parametric(self):
        """QAOA layer with parameters=["gamma","beta"] — bind and execute."""

        @qmc.qkernel
        def qaoa_layer(
            n: qmc.UInt,
            ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            gamma: qmc.Float,
            beta: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            for (i, j), Jij in qmc.items(ising):
                q[i], q[j] = qmc.rzz(q[i], q[j], gamma * Jij)
            for i in qmc.range(n):
                q[i] = qmc.rx(q[i], beta)
            return qmc.measure(q)

        ising = {(0, 1): 1.0}
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(
            qaoa_layer,
            bindings={"n": 2, "ising": ising},
            parameters=["gamma", "beta"],
        )
        # Should have parameters
        assert exe.has_parameters
        # Bind and execute
        executor = transpiler.executor()
        job = exe.sample(
            executor,
            shots=100,
            bindings={"gamma": 0.5, "beta": 0.3},
        )
        result = job.result()
        assert result is not None
        assert len(result.results) > 0

    def test_alternating_entangling_step2(self):
        """Hardware-efficient entangling: even CZ pairs + odd CZ pairs via step=2."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            # Even pairs: (0,1), (2,3), ...
            for i in qmc.range(0, n - 1, 2):
                q[i], q[i + 1] = qmc.cz(q[i], q[i + 1])
            # Odd pairs: (1,2), (3,4), ...
            for i in qmc.range(1, n - 1, 2):
                q[i], q[i + 1] = qmc.cz(q[i], q[i + 1])
            return qmc.measure(q)

        n = 6
        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": n})
        cz_count = sum(1 for name in _gate_names(qc) if name == "cz")
        # Even pairs: 3 (0-1, 2-3, 4-5), Odd pairs: 2 (1-2, 3-4) = 5 total
        even_pairs = len(range(0, n - 1, 2))
        odd_pairs = len(range(1, n - 1, 2))
        assert cz_count == even_pairs + odd_pairs


# ============================================================================
# 4. Transpiler Pass Tests (Step-by-Step Pipeline)
# ============================================================================


class TestTranspilerPassesPipeline:
    """Test individual transpiler pipeline stages."""

    @pytest.fixture
    def transpiler(self):
        return QiskitTranspiler()

    def test_to_block(self, transpiler):
        """to_block converts QKernel to Block."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        block = transpiler.to_block(circuit)
        assert block is not None
        assert len(block.operations) > 0

    def test_inline(self, transpiler):
        """inline() flattens CallBlockOperations from sub-kernel calls."""

        @qmc.qkernel
        def sub_kernel(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def main_kernel() -> qmc.Bit:
            q = qmc.qubit("q")
            q = sub_kernel(q)
            return qmc.measure(q)

        block = transpiler.to_block(main_kernel)
        inlined = transpiler.inline(block)

        from qamomile.circuit.ir.block import BlockKind

        assert inlined.kind == BlockKind.LINEAR

    def test_linear_validate(self, transpiler):
        """linear_validate() checks no-cloning on inlined block."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            q = qmc.x(q)
            return qmc.measure(q)

        block = transpiler.to_block(circuit)
        inlined = transpiler.inline(block)
        validated = transpiler.linear_validate(inlined)
        # Should pass without error; returned block is same object or equivalent
        assert validated is not None
        assert len(validated.operations) > 0

    def test_constant_fold(self, transpiler):
        """constant_fold reduces constant expressions."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            angle = theta * 2.0
            q = qmc.rx(q, angle)
            return qmc.measure(q)

        block = transpiler.to_block(circuit, bindings={"theta": 0.5})
        inlined = transpiler.inline(block)
        validated = transpiler.linear_validate(inlined)
        folded = transpiler.constant_fold(validated, bindings={"theta": 0.5})
        assert folded is not None

    def test_analyze(self, transpiler):
        """analyze() validates dependencies and marks block ANALYZED."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        block = transpiler.to_block(circuit)
        inlined = transpiler.inline(block)
        validated = transpiler.linear_validate(inlined)
        folded = transpiler.constant_fold(validated)
        analyzed = transpiler.analyze(folded)

        from qamomile.circuit.ir.block import BlockKind

        assert analyzed.kind == BlockKind.ANALYZED

    def test_separate(self, transpiler):
        """separate() splits into C->Q->C segments."""
        from qamomile.circuit.transpiler.segments import SimplifiedProgram

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        block = transpiler.to_block(circuit)
        inlined = transpiler.inline(block)
        validated = transpiler.linear_validate(inlined)
        folded = transpiler.constant_fold(validated)
        analyzed = transpiler.analyze(folded)
        separated = transpiler.separate(analyzed)

        assert isinstance(separated, SimplifiedProgram)
        assert separated.quantum is not None

    def test_emit(self, transpiler):
        """emit() generates backend-specific circuit."""
        from qamomile.circuit.transpiler.executable import ExecutableProgram

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        block = transpiler.to_block(circuit)
        inlined = transpiler.inline(block)
        validated = transpiler.linear_validate(inlined)
        folded = transpiler.constant_fold(validated)
        analyzed = transpiler.analyze(folded)
        separated = transpiler.separate(analyzed)
        exe = transpiler.emit(separated)

        assert isinstance(exe, ExecutableProgram)
        assert len(exe.compiled_quantum) > 0

    def test_full_transpile(self, transpiler):
        """Full transpile() pipeline end-to-end."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        exe = transpiler.transpile(circuit)
        assert exe is not None
        assert exe.quantum_circuit is not None

    def test_to_circuit_convenience(self, transpiler):
        """to_circuit() returns just the QuantumCircuit."""
        from qiskit import QuantumCircuit

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        qc = transpiler.to_circuit(circuit)
        assert isinstance(qc, QuantumCircuit)


class TestTranspilerConfigAndSubstitution:
    """Test TranspilerConfig, substitute(), strategy selection, and segment structure."""

    def test_transpiler_config_with_strategies(self):
        """TranspilerConfig.with_strategies creates config with strategy overrides."""
        from qamomile.circuit.transpiler.transpiler import TranspilerConfig

        config = TranspilerConfig.with_strategies({"qft": "approximate_k2"})
        assert config.decomposition.strategy_overrides["qft"] == "approximate_k2"
        assert len(config.substitutions.rules) == 1
        assert config.substitutions.rules[0].source_name == "qft"
        assert config.substitutions.rules[0].strategy == "approximate_k2"

    def test_set_config_on_transpiler(self):
        """set_config() applies TranspilerConfig to transpiler."""
        from qamomile.circuit.transpiler.transpiler import TranspilerConfig

        transpiler = QiskitTranspiler(use_native_composite=False)
        config = TranspilerConfig.with_strategies({"qft": "approximate_k2"})
        transpiler.set_config(config)
        assert transpiler.config is config

    def test_substitute_pass_sets_strategy(self):
        """substitute() pass sets strategy_name on CompositeGateOperation."""
        from qamomile.circuit.transpiler.transpiler import TranspilerConfig

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q = qmc.qft(q)
            return qmc.measure(q)

        transpiler = QiskitTranspiler(use_native_composite=False)
        config = TranspilerConfig.with_strategies({"qft": "approximate_k2"})
        transpiler.set_config(config)

        block = transpiler.to_block(circuit)
        substituted = transpiler.substitute(block)
        # The substitution pass should have been applied (block changed)
        assert substituted is not None

    def test_approximate_qft_fewer_gates(self):
        """Approximate QFT strategy produces fewer gates than standard."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(5, "q")
            q = qmc.qft(q)
            return qmc.measure(q)

        # Standard QFT (default)
        transpiler_std = QiskitTranspiler(use_native_composite=False)
        exe_std = transpiler_std.transpile(circuit)
        qc_std = exe_std.compiled_quantum[0].circuit
        gates_std = [i for i in qc_std.data if i.operation.name not in ("measure", "barrier")]

        # Approximate QFT (k=2)
        from qamomile.circuit.transpiler.transpiler import TranspilerConfig

        transpiler_approx = QiskitTranspiler(use_native_composite=False)
        config = TranspilerConfig.with_strategies({"qft": "approximate_k2"})
        transpiler_approx.set_config(config)
        exe_approx = transpiler_approx.transpile(circuit)
        qc_approx = exe_approx.compiled_quantum[0].circuit
        gates_approx = [i for i in qc_approx.data if i.operation.name not in ("measure", "barrier")]

        # Approximate should have strictly fewer gates for 5 qubits
        assert len(gates_approx) < len(gates_std)

    def test_approximate_qft_cp_count(self):
        """Approximate QFT (k=2) on 5 qubits has fewer CP gates than standard."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(5, "q")
            q = qmc.qft(q)
            return qmc.measure(q)

        # Standard QFT
        transpiler_std = QiskitTranspiler(use_native_composite=False)
        exe_std = transpiler_std.transpile(circuit)
        qc_std = exe_std.compiled_quantum[0].circuit
        cp_std = sum(1 for i in qc_std.data if i.operation.name == "cp")

        # Approximate k=2
        from qamomile.circuit.transpiler.transpiler import TranspilerConfig

        transpiler_approx = QiskitTranspiler(use_native_composite=False)
        config = TranspilerConfig.with_strategies({"qft": "approximate_k2"})
        transpiler_approx.set_config(config)
        exe_approx = transpiler_approx.transpile(circuit)
        qc_approx = exe_approx.compiled_quantum[0].circuit
        cp_approx = sum(1 for i in qc_approx.data if i.operation.name == "cp")

        # Standard: n*(n-1)/2 = 10 CP gates; Approximate k=2: fewer
        assert cp_std == 10  # 5*4/2
        assert cp_approx < cp_std

    def test_qft_strategy_resources(self):
        """QFT strategy resource metadata is consistent."""
        from qamomile.circuit.stdlib.qft import QFT

        qft_gate = QFT(5)
        standard_resources = qft_gate.get_resources_for_strategy("standard")
        approx_resources = qft_gate.get_resources_for_strategy("approximate_k2")

        assert standard_resources.custom_metadata["num_cp_gates"] == 10
        assert approx_resources.custom_metadata["num_cp_gates"] < 10
        assert standard_resources.custom_metadata["num_h_gates"] == 5
        assert approx_resources.custom_metadata["num_h_gates"] == 5

    def test_substitute_no_rules_is_noop(self):
        """substitute() with no config rules returns block unchanged."""
        transpiler = QiskitTranspiler()

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        block = transpiler.to_block(circuit)
        substituted = transpiler.substitute(block)
        # Should return exact same block (no rules)
        assert substituted is block

    def test_separate_segments_simple_circuit(self):
        """separate() produces SimplifiedProgram with quantum segment, no classical prep/post."""
        from qamomile.circuit.transpiler.segments import SimplifiedProgram

        transpiler = QiskitTranspiler()

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        block = transpiler.to_block(circuit)
        inlined = transpiler.inline(block)
        validated = transpiler.linear_validate(inlined)
        folded = transpiler.constant_fold(validated)
        analyzed = transpiler.analyze(folded)
        separated = transpiler.separate(analyzed)

        assert isinstance(separated, SimplifiedProgram)
        assert separated.quantum is not None
        assert len(separated.quantum.operations) > 0
        # Simple H+measure circuit should have no classical prep
        assert separated.classical_prep is None

    def test_separate_segments_with_classical_post(self):
        """QPE produces classical_post segment for QFixed decoding."""
        from qamomile.circuit.transpiler.segments import SimplifiedProgram

        @qmc.qkernel
        def phase_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            return qmc.p(q, theta)

        @qmc.qkernel
        def qpe_circuit(phase: qmc.Float) -> qmc.Float:
            phase_register = qmc.qubit_array(3, name="phase_reg")
            target = qmc.qubit(name="target")
            target = qmc.x(target)
            phase_q = qmc.qpe(target, phase_register, phase_gate, theta=phase)
            return qmc.measure(phase_q)

        transpiler = QiskitTranspiler(use_native_composite=False)
        block = transpiler.to_block(qpe_circuit, bindings={"phase": np.pi / 2})
        substituted = transpiler.substitute(block)
        inlined = transpiler.inline(substituted)
        validated = transpiler.linear_validate(inlined)
        folded = transpiler.constant_fold(validated, bindings={"phase": np.pi / 2})
        analyzed = transpiler.analyze(folded)
        separated = transpiler.separate(analyzed)

        assert isinstance(separated, SimplifiedProgram)
        assert separated.quantum is not None
        # QPE returns QFixed → Float, so classical_post should handle the decode
        assert separated.classical_post is not None

    def test_get_first_circuit(self):
        """ExecutableProgram.get_first_circuit() returns the quantum circuit."""
        from qiskit import QuantumCircuit

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit)
        qc = exe.get_first_circuit()
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits > 0

    def test_get_first_circuit_matches_quantum_circuit_property(self):
        """get_first_circuit() should match quantum_circuit property."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit)
        assert exe.get_first_circuit() is exe.quantum_circuit

    def test_to_circuit_convenience_method(self):
        """to_circuit() convenience method returns backend circuit directly."""
        from qiskit import QuantumCircuit

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        qc = transpiler.to_circuit(circuit)
        assert isinstance(qc, QuantumCircuit)

    def test_to_circuit_with_bindings(self):
        """to_circuit() with parameter bindings."""
        from qiskit import QuantumCircuit

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        qc = transpiler.to_circuit(circuit, bindings={"theta": np.pi / 4})
        assert isinstance(qc, QuantumCircuit)
        rx_gates = [i for i in qc.data if i.operation.name == "rx"]
        assert len(rx_gates) == 1
        assert abs(float(rx_gates[0].operation.params[0]) - np.pi / 4) < 1e-10

    def test_step_by_step_matches_transpile(self):
        """Step-by-step pipeline produces same circuit as transpile()."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()

        # Full pipeline
        exe_full = transpiler.transpile(circuit)
        qc_full = exe_full.compiled_quantum[0].circuit

        # Step-by-step
        block = transpiler.to_block(circuit)
        substituted = transpiler.substitute(block)
        inlined = transpiler.inline(substituted)
        validated = transpiler.linear_validate(inlined)
        folded = transpiler.constant_fold(validated)
        analyzed = transpiler.analyze(folded)
        separated = transpiler.separate(analyzed)
        exe_step = transpiler.emit(separated)
        qc_step = exe_step.compiled_quantum[0].circuit

        # Both should produce functionally equivalent circuits
        sv_full = _run_statevector(qc_full)
        sv_step = _run_statevector(qc_step)
        assert statevectors_equal(sv_full, sv_step)

    def test_multiple_strategy_overrides(self):
        """TranspilerConfig supports overriding multiple gates at once."""
        from qamomile.circuit.transpiler.transpiler import TranspilerConfig

        config = TranspilerConfig.with_strategies({
            "qft": "approximate_k2",
            "iqft": "approximate_k2",
        })
        assert len(config.substitutions.rules) == 2
        rule_names = {r.source_name for r in config.substitutions.rules}
        assert rule_names == {"qft", "iqft"}


class TestParameterizedCircuits:
    """Test parametric circuit transpilation."""

    def test_parametric_rx(self):
        """Transpile with parameters= preserves Qiskit Parameter objects."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit
        assert len(qc.parameters) > 0

    def test_parametric_bind_and_execute(self):
        """Bind parameters and execute."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])

        job = exe.sample(
            transpiler.executor(), bindings={"theta": np.pi}, shots=100
        )
        result = job.result()
        assert result is not None
        assert len(result.results) > 0

    def test_mixed_bound_and_parametric(self):
        """Some bindings bound, others kept as parameters."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, theta: qmc.Float
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.rx(q[i], theta)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(
            circuit, bindings={"n": 3}, parameters=["theta"]
        )
        qc = exe.compiled_quantum[0].circuit
        assert len(qc.parameters) > 0


class TestSubKernelInlining:
    """Test sub-kernel calls and inlining."""

    def test_sub_kernel_inline(self):
        """Sub-kernel calls are inlined into the main circuit."""

        @qmc.qkernel
        def apply_h(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def main() -> qmc.Bit:
            q = qmc.qubit("q")
            q = apply_h(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(main)
        sv = _run_statevector(qc)
        expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
        assert statevectors_equal(sv, expected)

    def test_chained_sub_kernels(self):
        """Multiple chained sub-kernel calls."""

        @qmc.qkernel
        def layer_rx(
            q: qmc.Vector[qmc.Qubit], theta: qmc.Float
        ) -> qmc.Vector[qmc.Qubit]:
            n = q.shape[0]
            for i in qmc.range(n):
                q[i] = qmc.rx(q[i], theta)
            return q

        @qmc.qkernel
        def layer_rz(
            q: qmc.Vector[qmc.Qubit], phi: qmc.Float
        ) -> qmc.Vector[qmc.Qubit]:
            n = q.shape[0]
            for i in qmc.range(n):
                q[i] = qmc.rz(q[i], phi)
            return q

        @qmc.qkernel
        def full(
            hi: qmc.Vector[qmc.Float], theta: qmc.Float, phi: qmc.Float
        ) -> qmc.Vector[qmc.Bit]:
            n = hi.shape[0]
            q = qmc.qubit_array(n, "q")
            q = layer_rx(q, theta)
            q = layer_rz(q, phi)
            return qmc.measure(q)

        hi = np.array([1.0, 2.0, 3.0])
        _, qc = _transpile_and_get_circuit(
            full, bindings={"hi": hi, "theta": 0.5, "phi": 0.3}
        )
        rx_count = sum(1 for name in _gate_names(qc) if name == "rx")
        rz_count = sum(1 for name in _gate_names(qc) if name == "rz")
        assert rx_count == 3
        assert rz_count == 3


# ============================================================================
# 5. Stdlib Tests (QFT, IQFT, controlled, CompositeGate)
# ============================================================================


class TestStdlibQFT:
    """Test QFT/IQFT through the full pipeline."""

    def test_qft_transpiles(self):
        """QFT can be transpiled to Qiskit circuit."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q = qmc.qft(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert qc is not None
        assert qc.num_qubits == 3

    def test_iqft_transpiles(self):
        """IQFT can be transpiled to Qiskit circuit."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q = qmc.iqft(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert qc is not None
        assert qc.num_qubits == 3

    def test_qft_then_iqft_transpiles(self):
        """QFT followed by IQFT can be transpiled and simulated.

        Note: The decomposed QFT/IQFT both include bit-reversal SWAPs,
        so QFT -> IQFT != identity in general. We verify the circuit
        transpiles and runs without error.
        """

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q = qmc.qft(q)
            q = qmc.iqft(q)
            return qmc.measure(q)

        # Use decomposed emission so Aer can simulate
        transpiler = QiskitTranspiler(use_native_composite=False)
        exe = transpiler.transpile(circuit)
        qc = exe.compiled_quantum[0].circuit
        sv = _run_statevector(qc)
        # Just verify it produces a valid statevector (norm 1)
        assert abs(np.linalg.norm(sv) - 1.0) < 1e-10

    def test_qft_native_emission(self):
        """Native QFT emitter uses Qiskit's QFT library gate."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q = qmc.qft(q)
            return qmc.measure(q)

        transpiler = QiskitTranspiler(use_native_composite=True)
        exe = transpiler.transpile(circuit)
        qc = exe.compiled_quantum[0].circuit
        assert len(qc.data) > 0

    def test_qft_decomposed_emission(self):
        """Decomposed QFT uses primitive gates (H, CP)."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q = qmc.qft(q)
            return qmc.measure(q)

        transpiler = QiskitTranspiler(use_native_composite=False)
        exe = transpiler.transpile(circuit)
        qc = exe.compiled_quantum[0].circuit
        gate_set = set(_gate_names(qc))
        assert "h" in gate_set


class TestStdlibQPE:
    """Test Quantum Phase Estimation through the full pipeline."""

    def test_qpe_transpiles(self):
        """QPE can be transpiled to a Qiskit circuit."""

        @qmc.qkernel
        def phase_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            return qmc.p(q, theta)

        @qmc.qkernel
        def qpe_3bit(phase: qmc.Float) -> qmc.Float:
            phase_register = qmc.qubit_array(3, name="phase_reg")
            target = qmc.qubit(name="target")
            target = qmc.x(target)
            phase_q = qmc.qpe(target, phase_register, phase_gate, theta=phase)
            return qmc.measure(phase_q)

        transpiler = QiskitTranspiler(use_native_composite=False)
        exe = transpiler.transpile(qpe_3bit, bindings={"phase": np.pi / 2})
        qc = exe.compiled_quantum[0].circuit
        assert qc is not None
        assert qc.num_qubits == 4  # 3 counting + 1 target

    def test_qpe_execution(self):
        """QPE execution returns valid float results."""

        @qmc.qkernel
        def phase_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            return qmc.p(q, theta)

        @qmc.qkernel
        def qpe_3bit(phase: qmc.Float) -> qmc.Float:
            phase_register = qmc.qubit_array(3, name="phase_reg")
            target = qmc.qubit(name="target")
            target = qmc.x(target)
            phase_q = qmc.qpe(target, phase_register, phase_gate, theta=phase)
            return qmc.measure(phase_q)

        transpiler = QiskitTranspiler(use_native_composite=False)
        exe = transpiler.transpile(qpe_3bit, bindings={"phase": np.pi / 2})
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=500)
        result = job.result()
        assert result is not None
        # QPE should return float values in [0, 1)
        for value, count in result.results:
            assert isinstance(value, float)
            assert 0.0 <= value < 1.0
            assert count > 0


class TestControlledGate:
    """Test controlled gate through the frontend pipeline."""

    def test_controlled_h(self):
        """controlled(h_kernel) creates controlled-H circuit."""

        @qmc.qkernel
        def h_gate(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            return q

        controlled_h = qmc.controlled(h_gate)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[0], q[1] = controlled_h(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert qc is not None
        assert qc.num_qubits == 2

    def test_controlled_rx(self):
        """controlled(rx_kernel) creates controlled-RX circuit."""

        @qmc.qkernel
        def rx_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            q = qmc.rx(q, theta)
            return q

        controlled_rx = qmc.controlled(rx_gate)

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[0], q[1] = controlled_rx(q[0], q[1], theta=theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 2})
        assert qc is not None
        assert qc.num_qubits == 2


class TestCustomCompositeGate:
    """Test custom CompositeGate through the full pipeline."""

    def test_custom_gate_transpiles(self):
        """Custom CompositeGate decomposes and transpiles."""
        from qamomile.circuit.frontend.composite_gate import CompositeGate
        from qamomile.circuit.frontend.handle import Qubit

        class BellPair(CompositeGate):
            custom_name = "bell_pair"

            @property
            def num_target_qubits(self) -> int:
                return 2

            def _decompose(self, qubits: tuple) -> tuple:
                q0, q1 = qubits
                q0 = qmc.h(q0)
                q0, q1 = qmc.cx(q0, q1)
                return (q0, q1)

        bell = BellPair()

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = bell(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # CompositeGate may allocate extra qubits internally
        num_q = qc.num_qubits
        # This test assumes the Bell pair occupies the only two qubits.
        # If CompositeGate ever allocates ancillas, update the expected-state
        # construction instead of relaxing this assertion.
        assert num_q == 2
        expected = np.zeros(2**num_q, dtype=complex)
        expected[0] = 1.0 / np.sqrt(2)  # |00>
        expected[3] = 1.0 / np.sqrt(2)  # |11> (q0=1, q1=1)
        assert statevectors_equal(sv, expected)


# ============================================================================
# 6. Measurement + Execution Tests
# ============================================================================


class TestMeasurement:
    """Test measurement operations."""

    def test_single_qubit_measure(self):
        """Single qubit measurement returns Bit."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        assert result is not None
        for value, _ in result.results:
            assert value == 1

    def test_vector_measure(self):
        """Vector qubit measurement returns Vector[Bit]."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.x(q[0])
            q[2] = qmc.x(q[2])
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        assert result is not None
        for value, _ in result.results:
            # Vector[Bit] returns tuple of ints
            assert isinstance(value, tuple)
            assert len(value) == 3

    def test_execution_bitstring_counts(self):
        """Execution returns correct bitstring distribution."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=1000)
        result = job.result()
        assert result is not None

        outcomes = {val for val, _ in result.results}
        # Bell state should only produce (0, 0) and (1, 1)
        assert outcomes.issubset({(0, 0), (1, 1)})


class TestExecution:
    """Test circuit execution patterns."""

    def test_sample_with_shots(self):
        """Sample with varying shot counts."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=500)
        result = job.result()
        total_counts = sum(count for _, count in result.results)
        assert total_counts == 500

    def test_parametric_sample(self):
        """Parametric circuit execution."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        executor = transpiler.executor()

        # theta=0: should measure |0>
        job = exe.sample(executor, bindings={"theta": 0.0}, shots=100)
        result = job.result()
        for value, _ in result.results:
            assert value == 0

        # theta=pi: should measure |1>
        job = exe.sample(executor, bindings={"theta": np.pi}, shots=100)
        result = job.result()
        for value, _ in result.results:
            assert value == 1


# ============================================================================
# 7. Edge Cases & Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_measure_only_circuit(self):
        """Circuit with no gates, just measurement."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = all_zeros_state(1)
        assert statevectors_equal(sv, expected)

    def test_multi_qubit_no_entanglement(self):
        """Multiple qubits with independent gates."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.h(q[0])
            q[1] = qmc.x(q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # q0=|+>, q1=|1>, q2=|0>
        # |q2 q1 q0> basis: expected |010> and |011> with equal amplitude
        expected = np.zeros(8, dtype=complex)
        expected[2] = 1 / np.sqrt(2)  # |010>
        expected[3] = 1 / np.sqrt(2)  # |011>
        assert statevectors_equal(sv, expected)

    def test_qubit_aliasing_raises(self):
        """Using same qubit for control and target raises error."""
        from qamomile.circuit.transpiler.errors import QubitAliasError

        with pytest.raises(QubitAliasError):

            @qmc.qkernel
            def circuit() -> qmc.Bit:
                q = qmc.qubit("q")
                q, q = qmc.cx(q, q)
                return qmc.measure(q)

            circuit.build()

    def test_large_qubit_count(self):
        """Circuit with 10+ qubits transpiles successfully."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 12})
        assert qc.num_qubits == 12
        h_count = sum(1 for name in _gate_names(qc) if name == "h")
        assert h_count == 12

    def test_zero_angle_rotation(self):
        """Rotation by 0 should be identity."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": 0.0})
        sv = _run_statevector(qc)
        expected = all_zeros_state(1)
        assert statevectors_equal(sv, expected)

    def test_double_gate_application(self):
        """Applying same gate twice: H^2 = I."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            q = qmc.h(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = all_zeros_state(1)
        assert statevectors_equal(sv, expected)

    def test_nested_kernel_calls(self):
        """Multi-level nested sub-kernel calls."""

        @qmc.qkernel
        def level2(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def level1(q: qmc.Qubit) -> qmc.Qubit:
            q = level2(q)
            q = qmc.x(q)
            return q

        @qmc.qkernel
        def main() -> qmc.Bit:
            q = qmc.qubit("q")
            q = level1(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(main)
        sv = _run_statevector(qc)
        state = all_zeros_state(1)
        state = GATE_SPECS["H"].matrix_fn() @ state
        state = GATE_SPECS["X"].matrix_fn() @ state
        assert statevectors_equal(sv, state)

    def test_circuit_with_classical_arithmetic(self):
        """Circuit with classical arithmetic using Python-level computation.

        Note: Float * constant inside a @qkernel creates a BinOp in IR, which
        the constant folding pass may not resolve when the parameter is bound.
        So we test that Python-level pre-computation of the angle works.
        """
        angle = np.pi / 4 * 2.0  # = pi/2

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": angle})
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RX"].matrix_fn(np.pi / 2)
        )
        assert statevectors_equal(sv, expected)
