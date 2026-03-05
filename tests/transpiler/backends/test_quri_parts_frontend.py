"""Rich QURI Parts frontend-to-backend test suite.

Tests the full pipeline: @qkernel definition -> QuriPartsCircuitTranspiler -> execution.
Covers every available frontend gate, gate combinations, transpiler passes,
parametric circuits, basic execution, algorithm modules, and edge cases.

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
    bell_state,
    plus_state,
    minus_state,
)

# ---------------------------------------------------------------------------
# Skip entire module if quri-parts is not installed
# ---------------------------------------------------------------------------
pytest.importorskip("quri_parts")
pytest.importorskip("quri_parts.qulacs")

from qamomile.quri_parts import QuriPartsCircuitTranspiler
from qamomile.circuit.transpiler.executable import ExecutableProgram
from qamomile.circuit.transpiler.segments import SimplifiedProgram
from qamomile.circuit.ir.block import BlockKind

import qamomile.observable as qm_o
from qamomile.circuit.algorithm.basic import (
    rx_layer,
    ry_layer,
    rz_layer,
    cz_entangling_layer,
)
from qamomile.circuit.algorithm.qaoa import (
    ising_cost_circuit,
    x_mixier_circuit,
    qaoa_circuit,
    superposition_vector,
    qaoa_state,
)
from qamomile.circuit.algorithm.fqaoa import (
    initial_occupations,
    givens_rotation,
    hopping_gate,
    mixer_layer,
    cost_layer,
    fqaoa_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_statevector(circuit) -> np.ndarray:
    """Run a QURI Parts circuit and return the statevector via Qulacs.

    Handles both parametric (unbound) and bound circuits.
    For parametric circuits with no parameter values, binds with empty list.
    """
    from quri_parts.core.state import GeneralCircuitQuantumState
    from quri_parts.qulacs.simulator import evaluate_state_to_vector

    if hasattr(circuit, "parameter_count") and circuit.parameter_count > 0:
        bound_circuit = circuit.bind_parameters([0.0] * circuit.parameter_count)
    elif hasattr(circuit, "bind_parameters"):
        bound_circuit = circuit.bind_parameters([])
    else:
        bound_circuit = circuit

    circuit_state = GeneralCircuitQuantumState(
        bound_circuit.qubit_count, bound_circuit
    )
    statevector = evaluate_state_to_vector(circuit_state)
    return np.array(statevector.vector)


def _transpile_and_get_circuit(kernel, bindings=None, parameters=None):
    """Transpile a qkernel and return (ExecutableProgram, QURI Parts circuit)."""
    transpiler = QuriPartsCircuitTranspiler()
    exe = transpiler.transpile(kernel, bindings=bindings, parameters=parameters)
    circuit = exe.compiled_quantum[0].circuit
    return exe, circuit


def _gate_names(circuit) -> list[str]:
    """Return list of gate operation names in a QURI Parts circuit.

    Handles both parametric and non-parametric circuits.
    """
    if hasattr(circuit, "gates"):
        return [gate.name for gate in circuit.gates]
    if hasattr(circuit, "bind_parameters"):
        bound = circuit.bind_parameters([])
        if hasattr(bound, "gates"):
            return [gate.name for gate in bound.gates]
    return []


# ============================================================================
# 1. Individual Gate Statevector Tests
# ============================================================================


class TestSingleQubitGatesFrontend:
    """Test each single-qubit gate through the full frontend pipeline."""

    # -- Creation tests (verify gate names in circuit) --

    def test_h_creation(self):
        """H gate transpiles to circuit containing 'H' gate."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert "H" in _gate_names(qc)

    def test_x_creation(self):
        """X gate transpiles to circuit containing 'X' gate."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert "X" in _gate_names(qc)

    def test_y_creation(self):
        """Y gate transpiles to circuit containing 'Y' gate."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.y(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert "Y" in _gate_names(qc)

    def test_z_creation(self):
        """Z gate transpiles to circuit containing 'Z' gate."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.z(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert "Z" in _gate_names(qc)

    def test_s_creation(self):
        """S gate transpiles to circuit containing 'S' gate."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.s(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert "S" in _gate_names(qc)

    def test_sdg_creation(self):
        """SDG gate transpiles to circuit containing 'Sdag' gate."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.sdg(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert "Sdag" in _gate_names(qc)

    def test_t_creation(self):
        """T gate transpiles to circuit containing 'T' gate."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.t(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert "T" in _gate_names(qc)

    def test_tdg_creation(self):
        """TDG gate transpiles to circuit containing 'Tdag' gate."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.tdg(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert "Tdag" in _gate_names(qc)

    def test_rx_creation(self):
        """RX gate transpiles to circuit containing 'RX' gate."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 4})
        assert "RX" in _gate_names(qc)

    def test_ry_creation(self):
        """RY gate transpiles to circuit containing 'RY' gate."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.ry(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 4})
        assert "RY" in _gate_names(qc)

    def test_rz_creation(self):
        """RZ gate transpiles to circuit containing 'RZ' gate."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rz(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 4})
        assert "RZ" in _gate_names(qc)

    def test_p_creation(self):
        """P gate transpiles to circuit containing 'RZ' (P emitted as RZ)."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.p(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 4})
        assert "RZ" in _gate_names(qc)

    # -- Statevector tests --

    def test_h_statevector(self):
        """H|0> produces equal superposition (|0>+|1>)/sqrt(2)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
        assert statevectors_equal(sv, expected)

    def test_x_statevector(self):
        """X|0> = |1> statevector verification."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = np.array([0, 1], dtype=complex)
        assert statevectors_equal(sv, expected)

    def test_rx_determined(self):
        """RX(pi)|0> = -i|1>."""

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

    @pytest.mark.parametrize(
        "seed", [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
    )
    def test_rx_random(self, seed):
        """RX(random theta) statevector matches analytical RX matrix."""
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

    def test_ry_determined(self):
        """RY(pi/2)|0> -> (|0>+|1>)/sqrt(2)."""

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

    @pytest.mark.parametrize(
        "seed", [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
    )
    def test_ry_random(self, seed):
        """RY(random theta) statevector matches analytical RY matrix."""
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

    def test_rz_determined(self):
        """RZ(pi)|0> statevector matches analytical RZ matrix."""

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

    @pytest.mark.parametrize(
        "seed", [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
    )
    def test_rz_random(self, seed):
        """RZ(random theta) statevector matches analytical RZ matrix."""
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

    @pytest.mark.parametrize(
        "seed", [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
    )
    def test_p_random(self, seed):
        """P(random θ) on |1⟩ statevector matches analytical P matrix."""
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

    # -- Y (Pauli-Y) --

    def test_y_on_zero(self):
        """Y|0⟩ = i|1⟩."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.y(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["Y"].matrix_fn()
        )
        assert statevectors_equal(sv, expected)

    def test_y_on_one(self):
        """Y|1⟩ = -i|0⟩."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            q = qmc.y(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        state_after_x = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["X"].matrix_fn()
        )
        expected = compute_expected_statevector(
            state_after_x, GATE_SPECS["Y"].matrix_fn()
        )
        assert statevectors_equal(sv, expected)

    # -- Z (Pauli-Z) --

    def test_z_on_zero(self):
        """Z|0⟩ = |0⟩ (no change on computational basis |0⟩)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.z(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["Z"].matrix_fn()
        )
        assert statevectors_equal(sv, expected)

    def test_z_on_one(self):
        """Z|1⟩ = −|1⟩ (phase flip on |1⟩)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            q = qmc.z(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        state_after_x = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["X"].matrix_fn()
        )
        expected = compute_expected_statevector(
            state_after_x, GATE_SPECS["Z"].matrix_fn()
        )
        assert statevectors_equal(sv, expected)

    def test_z_on_plus(self):
        """Z|+⟩ = |−⟩ (phase flip converts |+⟩ to |−⟩)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            q = qmc.z(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = minus_state()
        assert statevectors_equal(sv, expected)

    # -- S (sqrt(Z)) --

    def test_s_on_zero(self):
        """S|0⟩ = |0⟩ (no change on computational basis |0⟩)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.s(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["S"].matrix_fn()
        )
        assert statevectors_equal(sv, expected)

    def test_s_on_one(self):
        """S|1⟩ = i|1⟩ (phase +i on |1⟩)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            q = qmc.s(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        state_after_x = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["X"].matrix_fn()
        )
        expected = compute_expected_statevector(
            state_after_x, GATE_SPECS["S"].matrix_fn()
        )
        assert statevectors_equal(sv, expected)

    def test_s_squared_equals_z(self):
        """S·S = Z on |+⟩ (two S gates equal one Z gate)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            q = qmc.s(q)
            q = qmc.s(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = minus_state()
        assert statevectors_equal(sv, expected)

    # -- SDG (S-dagger, inverse of S) --

    def test_sdg_on_zero(self):
        """SDG|0⟩ = |0⟩ (no change on computational basis |0⟩)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.sdg(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["SDG"].matrix_fn()
        )
        assert statevectors_equal(sv, expected)

    def test_sdg_on_one(self):
        """SDG|1⟩ = −i|1⟩ (phase −i on |1⟩)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            q = qmc.sdg(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        state_after_x = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["X"].matrix_fn()
        )
        expected = compute_expected_statevector(
            state_after_x, GATE_SPECS["SDG"].matrix_fn()
        )
        assert statevectors_equal(sv, expected)

    def test_sdg_inverse_of_s(self):
        """S then SDG = Identity on |1⟩."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            q = qmc.s(q)
            q = qmc.sdg(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = np.array([0, 1], dtype=complex)
        assert statevectors_equal(sv, expected)

    # -- T (sqrt(S)) --

    def test_t_on_zero(self):
        """T|0⟩ = |0⟩ (no change on computational basis |0⟩)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.t(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["T"].matrix_fn()
        )
        assert statevectors_equal(sv, expected)

    def test_t_on_one(self):
        """T|1⟩ = e^(iπ/4)|1⟩ (phase e^(iπ/4) on |1⟩)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            q = qmc.t(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        state_after_x = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["X"].matrix_fn()
        )
        expected = compute_expected_statevector(
            state_after_x, GATE_SPECS["T"].matrix_fn()
        )
        assert statevectors_equal(sv, expected)

    def test_t_squared_equals_s(self):
        """T·T = S on |1⟩ (two T gates equal one S gate)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            q = qmc.t(q)
            q = qmc.t(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        state_after_x = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["X"].matrix_fn()
        )
        expected = compute_expected_statevector(
            state_after_x, GATE_SPECS["S"].matrix_fn()
        )
        assert statevectors_equal(sv, expected)

    # -- TDG (T-dagger, inverse of T) --

    def test_tdg_on_zero(self):
        """TDG|0⟩ = |0⟩ (no change on computational basis |0⟩)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.tdg(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["TDG"].matrix_fn()
        )
        assert statevectors_equal(sv, expected)

    def test_tdg_on_one(self):
        """TDG|1⟩ = e^(−iπ/4)|1⟩ (phase e^(−iπ/4) on |1⟩)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            q = qmc.tdg(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        state_after_x = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["X"].matrix_fn()
        )
        expected = compute_expected_statevector(
            state_after_x, GATE_SPECS["TDG"].matrix_fn()
        )
        assert statevectors_equal(sv, expected)

    def test_tdg_inverse_of_t(self):
        """T then TDG = Identity on |1⟩."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            q = qmc.t(q)
            q = qmc.tdg(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = np.array([0, 1], dtype=complex)
        assert statevectors_equal(sv, expected)


class TestTwoQubitGatesFrontend:
    """Test each two-qubit gate through the full frontend pipeline."""

    # -- CX creation --

    def test_cx_creation(self):
        """CX gate transpiles to circuit containing 'CNOT' gate."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert "CNOT" in _gate_names(qc)

    # -- CX basis states --

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
        """CZ gate transpiles to circuit containing 'CZ' gate."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = qmc.cz(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert "CZ" in _gate_names(qc)

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
        """CP gate is decomposed (no native CP — emits RZ + CNOT gates)."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = qmc.cp(q[0], q[1], theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 4})
        names = _gate_names(qc)
        # CP decomposes to 3 RZ + 2 CNOT
        assert names.count("RZ") >= 3
        assert names.count("CNOT") >= 2

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

    @pytest.mark.parametrize(
        "seed", [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
    )
    def test_cp_random(self, seed):
        """CP(random θ) on |11⟩ statevector matches analytical CP matrix."""
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
        """RZZ gate transpiles to circuit containing 'PauliRotation' gate."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = qmc.rzz(q[0], q[1], theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 4})
        assert "PauliRotation" in _gate_names(qc)

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

    @pytest.mark.parametrize(
        "seed", [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
    )
    def test_rzz_random(self, seed):
        """RZZ(random θ) on |11⟩ statevector matches analytical RZZ matrix."""
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
        """SWAP gate transpiles to circuit containing 'SWAP' gate."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = qmc.swap(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert "SWAP" in _gate_names(qc)

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


class TestThreeQubitGatesFrontend:
    """Test each three-qubit gate through the full frontend pipeline."""

    # -- CCX (Toffoli) --

    def test_ccx_creation(self):
        """CCX gate transpiles to circuit containing 'TOFFOLI' gate."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0], q[1], q[2] = qmc.ccx(q[0], q[1], q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert "TOFFOLI" in _gate_names(qc)

    @pytest.mark.parametrize(
        "x_targets, basis_idx",
        [
            pytest.param([], 0, id="000"),
            pytest.param([0], 1, id="001"),
            pytest.param([1], 2, id="010"),
            pytest.param([0, 1], 3, id="011_flips"),
            pytest.param([2], 4, id="100"),
            pytest.param([0, 2], 5, id="101"),
            pytest.param([1, 2], 6, id="110"),
            pytest.param([0, 1, 2], 7, id="111_flips"),
        ],
    )
    def test_ccx_basis_state(self, x_targets, basis_idx):
        """CCX on all 8 basis states; flips target only when both controls |1⟩."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0], q[1], q[2] = qmc.ccx(q[0], q[1], q[2])
            return qmc.measure(q)

        @qmc.qkernel
        def circuit_x0() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.x(q[0])
            q[0], q[1], q[2] = qmc.ccx(q[0], q[1], q[2])
            return qmc.measure(q)

        @qmc.qkernel
        def circuit_x1() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[1] = qmc.x(q[1])
            q[0], q[1], q[2] = qmc.ccx(q[0], q[1], q[2])
            return qmc.measure(q)

        @qmc.qkernel
        def circuit_x01() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0], q[1], q[2] = qmc.ccx(q[0], q[1], q[2])
            return qmc.measure(q)

        @qmc.qkernel
        def circuit_x2() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[2] = qmc.x(q[2])
            q[0], q[1], q[2] = qmc.ccx(q[0], q[1], q[2])
            return qmc.measure(q)

        @qmc.qkernel
        def circuit_x02() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.x(q[0])
            q[2] = qmc.x(q[2])
            q[0], q[1], q[2] = qmc.ccx(q[0], q[1], q[2])
            return qmc.measure(q)

        @qmc.qkernel
        def circuit_x12() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[1] = qmc.x(q[1])
            q[2] = qmc.x(q[2])
            q[0], q[1], q[2] = qmc.ccx(q[0], q[1], q[2])
            return qmc.measure(q)

        @qmc.qkernel
        def circuit_x012() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[2] = qmc.x(q[2])
            q[0], q[1], q[2] = qmc.ccx(q[0], q[1], q[2])
            return qmc.measure(q)

        kernels = {
            0: circuit,
            1: circuit_x0,
            2: circuit_x1,
            3: circuit_x01,
            4: circuit_x2,
            5: circuit_x02,
            6: circuit_x12,
            7: circuit_x012,
        }

        _, qc = _transpile_and_get_circuit(kernels[basis_idx])
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            computational_basis_state(3, basis_idx),
            GATE_SPECS["TOFFOLI"].matrix_fn(),
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

    @pytest.mark.parametrize(
        "seed", [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
    )
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

    @pytest.mark.parametrize(
        "seed", [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
    )
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
        assert "H" in names
        assert "CNOT" in names
        assert qc.qubit_count >= 3


# ============================================================================
# 3. Transpiler Pipeline Tests
# ============================================================================


class TestTranspilerPassesPipeline:
    """Test individual transpiler pipeline stages."""

    @pytest.fixture
    def transpiler(self):
        return QuriPartsCircuitTranspiler()

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
        assert analyzed.kind == BlockKind.ANALYZED

    def test_separate(self, transpiler):
        """separate() splits into C->Q->C segments."""

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
        """to_circuit() returns the QURI Parts circuit directly."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        qc = transpiler.to_circuit(circuit)
        assert qc is not None
        assert qc.qubit_count > 0

    def test_step_by_step_matches_transpile(self, transpiler):
        """Step-by-step pipeline produces same statevector as transpile()."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        # Full pipeline
        exe_full = transpiler.transpile(circuit)
        qc_full = exe_full.compiled_quantum[0].circuit
        sv_full = _run_statevector(qc_full)

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
        sv_step = _run_statevector(qc_step)

        assert statevectors_equal(sv_full, sv_step)


# ============================================================================
# 4. Parametric Circuit Tests
# ============================================================================


class TestParameterizedCircuits:
    """Test parametric circuit transpilation and execution."""

    def test_parametric_rx(self):
        """Transpile with parameters= preserves parametric circuit."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        transpiler = QuriPartsCircuitTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit
        assert qc.parameter_count > 0

    def test_parametric_bind_and_execute(self):
        """Bind parameters and execute."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        transpiler = QuriPartsCircuitTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])

        job = exe.sample(transpiler.executor(), bindings={"theta": np.pi}, shots=100)
        result = job.result()
        assert result is not None
        assert len(result.results) > 0

    def test_mixed_bound_and_parametric(self):
        """Some bindings bound, others kept as parameters."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.rx(q[i], theta)
            return qmc.measure(q)

        transpiler = QuriPartsCircuitTranspiler()
        exe = transpiler.transpile(circuit, bindings={"n": 3}, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit
        assert qc.parameter_count > 0

    def test_parametric_bind_and_rebind(self):
        """Compile once, execute with different parameter values."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.ry(q, theta)
            return qmc.measure(q)

        transpiler = QuriPartsCircuitTranspiler()
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
        sv = _run_statevector(qc)
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-10)


# ============================================================================
# 5. Basic Execution Tests
# ============================================================================


class TestExecution:
    """Test circuit execution through the QURI Parts sampler."""

    def test_single_qubit_x_measure(self):
        """X|0> = |1> — sampling should return all '1'."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            return qmc.measure(q)

        transpiler = QuriPartsCircuitTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        assert result is not None
        for value, _ in result.results:
            assert value == 1

    def test_bell_state_execution(self):
        """Bell state measurement: only |00> and |11> outcomes."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        transpiler = QuriPartsCircuitTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=1000)
        result = job.result()
        assert result is not None

        outcomes = {val for val, _ in result.results}
        # Bell state should only produce (0, 0) and (1, 1)
        assert outcomes.issubset({(0, 0), (1, 1)})

    def test_sample_shot_count(self):
        """Verify total shot count matches requested shots."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        transpiler = QuriPartsCircuitTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=500)
        result = job.result()
        total_counts = sum(count for _, count in result.results)
        assert total_counts == 500

    def test_parametric_sample(self):
        """Parametric circuit execution: theta=0 gives |0>, theta=pi gives |1>."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        transpiler = QuriPartsCircuitTranspiler()
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

    def test_vector_measure(self):
        """Vector qubit measurement returns Vector[Bit]."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.x(q[0])
            q[2] = qmc.x(q[2])
            return qmc.measure(q)

        transpiler = QuriPartsCircuitTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        assert result is not None
        for value, _ in result.results:
            # Vector[Bit] returns tuple of ints
            assert isinstance(value, tuple)
            assert len(value) == 3


# ============================================================================
# 6. Edge Cases and Error Handling
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

    @pytest.mark.parametrize("n_qubits", [20, 50])
    def test_large_qubit_count(self, n_qubits):
        """Circuit with many qubits transpiles successfully."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": n_qubits})
        assert qc.qubit_count == n_qubits

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
        """Circuit with classical arithmetic using Python-level computation."""
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


# ============================================================================
# 7. Algorithm – Basic Layers
# ============================================================================


class TestAlgorithmBasicLayers:
    """Test rotation and entangling layer building blocks."""

    def test_rx_layer_statevector(self):
        """rx_layer applies RX to each qubit with given angles."""

        @qmc.qkernel
        def circuit(thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q = rx_layer(q, thetas, qmc.uint(0))
            return qmc.measure(q)

        angles = np.array([np.pi / 2, np.pi, np.pi / 4])
        _, qc = _transpile_and_get_circuit(circuit, bindings={"thetas": angles})
        sv = _run_statevector(qc)

        # Build expected: RX(a2) ⊗ RX(a1) ⊗ RX(a0) |000>
        state = all_zeros_state(3)
        mat = tensor_product(
            GATE_SPECS["RX"].matrix_fn(angles[2]),
            GATE_SPECS["RX"].matrix_fn(angles[1]),
            GATE_SPECS["RX"].matrix_fn(angles[0]),
        )
        expected = mat @ state
        assert statevectors_equal(sv, expected)

    def test_ry_layer_statevector(self):
        """ry_layer applies RY to each qubit with given angles."""

        @qmc.qkernel
        def circuit(thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q = ry_layer(q, thetas, qmc.uint(0))
            return qmc.measure(q)

        angles = np.array([np.pi / 3, np.pi / 6])
        _, qc = _transpile_and_get_circuit(circuit, bindings={"thetas": angles})
        sv = _run_statevector(qc)

        state = all_zeros_state(2)
        mat = tensor_product(
            GATE_SPECS["RY"].matrix_fn(angles[1]),
            GATE_SPECS["RY"].matrix_fn(angles[0]),
        )
        expected = mat @ state
        assert statevectors_equal(sv, expected)

    def test_rz_layer_statevector(self):
        """rz_layer applies RZ to each qubit with given angles."""

        @qmc.qkernel
        def circuit(thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q = rz_layer(q, thetas, qmc.uint(0))
            return qmc.measure(q)

        angles = np.array([np.pi / 4, np.pi / 2])
        _, qc = _transpile_and_get_circuit(circuit, bindings={"thetas": angles})
        sv = _run_statevector(qc)

        state = all_zeros_state(2)
        mat = tensor_product(
            GATE_SPECS["RZ"].matrix_fn(angles[1]),
            GATE_SPECS["RZ"].matrix_fn(angles[0]),
        )
        expected = mat @ state
        assert statevectors_equal(sv, expected)

    def test_cz_entangling_layer_statevector(self):
        """cz_entangling_layer applies CZ between consecutive qubits."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            # Prepare all |+> to see CZ effects
            for i in qmc.range(3):
                q[i] = qmc.h(q[i])
            q = cz_entangling_layer(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-10)
        # CZ on |+++> produces entangled state — not separable
        # Check it's not equal to |+++>
        plus_all = tensor_product(plus_state(), plus_state(), plus_state())
        assert not statevectors_equal(sv, plus_all)

    def test_rx_layer_with_offset(self):
        """rx_layer with non-zero offset reads from later positions."""

        @qmc.qkernel
        def circuit(thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q = rx_layer(q, thetas, qmc.uint(2))
            return qmc.measure(q)

        # 4 values; offset=2 means use thetas[2] and thetas[3]
        angles = np.array([0.0, 0.0, np.pi / 3, np.pi / 4])
        _, qc = _transpile_and_get_circuit(circuit, bindings={"thetas": angles})
        sv = _run_statevector(qc)

        state = all_zeros_state(2)
        mat = tensor_product(
            GATE_SPECS["RX"].matrix_fn(angles[3]),
            GATE_SPECS["RX"].matrix_fn(angles[2]),
        )
        expected = mat @ state
        assert statevectors_equal(sv, expected)

    def test_rotation_then_entangling(self):
        """Combine rotation layer followed by entangling layer."""

        @qmc.qkernel
        def circuit(thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q = ry_layer(q, thetas, qmc.uint(0))
            q = cz_entangling_layer(q)
            return qmc.measure(q)

        angles = np.array([np.pi / 2, np.pi / 4, np.pi / 3])
        _, qc = _transpile_and_get_circuit(circuit, bindings={"thetas": angles})
        sv = _run_statevector(qc)
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-10)

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_variational_ansatz_pattern(self, seed):
        """Full variational ansatz: RY -> CZ -> RY -> CZ -> RY."""
        rng = np.random.default_rng(seed)
        n_qubits = 3

        @qmc.qkernel
        def ansatz(thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q = ry_layer(q, thetas, qmc.uint(0))
            q = cz_entangling_layer(q)
            q = ry_layer(q, thetas, qmc.uint(3))
            q = cz_entangling_layer(q)
            q = ry_layer(q, thetas, qmc.uint(6))
            return qmc.measure(q)

        params = rng.uniform(0, 2 * np.pi, size=9)
        _, qc = _transpile_and_get_circuit(ansatz, bindings={"thetas": params})
        sv = _run_statevector(qc)
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-10)

    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_ry_layer_gate_count(self, n_qubits):
        """ry_layer produces exactly n RY gates."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, thetas: qmc.Vector[qmc.Float]
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = ry_layer(q, thetas, qmc.uint(0))
            return qmc.measure(q)

        thetas = [0.1 * i for i in range(n_qubits)]
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": n_qubits, "thetas": thetas}
        )
        ry_count = _gate_names(qc).count("RY")
        assert ry_count == n_qubits

    @pytest.mark.parametrize("n_qubits", [3, 4, 5])
    def test_cz_entangling_layer_gate_count(self, n_qubits):
        """cz_entangling_layer emits n-1 CZ gates."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            q = cz_entangling_layer(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": n_qubits})
        cz_count = _gate_names(qc).count("CZ")
        assert cz_count == n_qubits - 1

    def test_cz_entangling_layer_on_plus_state_analytical(self):
        """CZ layer on |+>^3 produces correct entangled statevector (analytical)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            q = cz_entangling_layer(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 3})
        sv = _run_statevector(qc)
        # Analytical: H⊗3 |000>, then CZ(0,1), then CZ(1,2)
        H = GATE_SPECS["H"].matrix_fn()
        CZ = GATE_SPECS["CZ"].matrix_fn()
        I2 = identity(2)
        state = all_zeros_state(3)
        state = tensor_product(H, tensor_product(H, H)) @ state
        cz_01 = tensor_product(I2, CZ)
        state = cz_01 @ state
        cz_12 = tensor_product(CZ, I2)
        state = cz_12 @ state
        assert statevectors_equal(sv, state)

    def test_ry_plus_cz_variational_block(self):
        """ry_layer + cz_entangling_layer: gate structure verification."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, thetas: qmc.Vector[qmc.Float]
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = ry_layer(q, thetas, qmc.uint(0))
            q = cz_entangling_layer(q)
            return qmc.measure(q)

        thetas = [0.1, 0.2, 0.3]
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": 3, "thetas": thetas}
        )
        names = _gate_names(qc)
        ry_count = names.count("RY")
        cz_count = names.count("CZ")
        assert ry_count == 3
        assert cz_count == 2

    def test_ry_plus_cz_statevector(self):
        """ry_layer + cz_entangling_layer statevector correctness."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, thetas: qmc.Vector[qmc.Float]
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = ry_layer(q, thetas, qmc.uint(0))
            q = cz_entangling_layer(q)
            return qmc.measure(q)

        thetas = [0.5, 1.0, 1.5]
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": 3, "thetas": thetas}
        )
        sv = _run_statevector(qc)
        RY0, RY1, RY2 = [GATE_SPECS["RY"].matrix_fn(t) for t in thetas]
        CZ = GATE_SPECS["CZ"].matrix_fn()
        I2 = identity(2)
        state = all_zeros_state(3)
        state = tensor_product(RY2, tensor_product(RY1, RY0)) @ state
        state = tensor_product(I2, CZ) @ state
        state = tensor_product(CZ, I2) @ state
        assert statevectors_equal(sv, state)


# ============================================================================
# 8. Algorithm – QAOA Modules
# ============================================================================


class TestAlgorithmQAOAModules:
    """Test QAOA circuit building blocks and full pipeline."""

    def _simple_ising(self):
        """Return a simple 2-qubit Ising model for testing."""
        quad = {(0, 1): 1.0}
        linear = {0: 0.5, 1: -0.5}
        return quad, linear

    def test_ising_cost_circuit(self):
        """ising_cost_circuit produces a valid unitary evolution."""
        quad, linear = self._simple_ising()

        @qmc.qkernel
        def circuit(
            quad_: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear_: qmc.Dict[qmc.UInt, qmc.Float],
            gamma: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            for i in qmc.range(2):
                q[i] = qmc.h(q[i])
            q = ising_cost_circuit(quad_, linear_, q, gamma)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(
            circuit,
            bindings={"quad_": quad, "linear_": linear, "gamma": 0.5},
        )
        sv = _run_statevector(qc)
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-10)

    def test_x_mixer_circuit(self):
        """x_mixer applies RX(2*beta) to each qubit."""

        @qmc.qkernel
        def circuit(beta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            for i in qmc.range(2):
                q[i] = qmc.h(q[i])
            q = x_mixier_circuit(q, beta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"beta": np.pi / 4})
        sv = _run_statevector(qc)
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-10)

    def test_qaoa_circuit_p1(self):
        """Single-layer QAOA circuit produces valid state."""
        quad, linear = self._simple_ising()

        @qmc.qkernel
        def circuit(
            quad_: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear_: qmc.Dict[qmc.UInt, qmc.Float],
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            for i in qmc.range(2):
                q[i] = qmc.h(q[i])
            q = qaoa_circuit(qmc.uint(1), quad_, linear_, q, gammas, betas)
            return qmc.measure(q)

        gammas = np.array([0.3])
        betas = np.array([0.7])
        _, qc = _transpile_and_get_circuit(
            circuit,
            bindings={
                "quad_": quad,
                "linear_": linear,
                "gammas": gammas,
                "betas": betas,
            },
        )
        sv = _run_statevector(qc)
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-10)

    def test_superposition_vector(self):
        """superposition_vector creates uniform superposition |+...+>."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = superposition_vector(qmc.uint(3))
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = np.ones(8, dtype=complex) / np.sqrt(8)
        assert statevectors_equal(sv, expected)

    def test_qaoa_state_end_to_end(self):
        """Full qaoa_state: superposition + QAOA layers."""
        quad = {(0, 1): 1.0}
        linear = {0: 0.5, 1: -0.5}

        @qmc.qkernel
        def circuit(
            quad_: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear_: qmc.Dict[qmc.UInt, qmc.Float],
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qaoa_state(
                qmc.uint(1), quad_, linear_, qmc.uint(2), gammas, betas
            )
            return qmc.measure(q)

        gammas = np.array([0.5])
        betas = np.array([0.8])
        _, qc = _transpile_and_get_circuit(
            circuit,
            bindings={
                "quad_": quad,
                "linear_": linear,
                "gammas": gammas,
                "betas": betas,
            },
        )
        sv = _run_statevector(qc)
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-10)

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_qaoa_state_random_params(self, seed):
        """QAOA state with random parameters is always normalized."""
        rng = np.random.default_rng(seed)
        quad = {(0, 1): rng.uniform(-1, 1), (1, 2): rng.uniform(-1, 1)}
        linear = {0: rng.uniform(-1, 1), 1: rng.uniform(-1, 1), 2: rng.uniform(-1, 1)}

        @qmc.qkernel
        def circuit(
            quad_: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear_: qmc.Dict[qmc.UInt, qmc.Float],
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qaoa_state(
                qmc.uint(2), quad_, linear_, qmc.uint(3), gammas, betas
            )
            return qmc.measure(q)

        gammas = rng.uniform(0, np.pi, size=2)
        betas = rng.uniform(0, np.pi, size=2)
        _, qc = _transpile_and_get_circuit(
            circuit,
            bindings={
                "quad_": quad,
                "linear_": linear,
                "gammas": gammas,
                "betas": betas,
            },
        )
        sv = _run_statevector(qc)
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-10)

    def test_qaoa_sampling_consistency(self):
        """QAOA circuit produces valid measurement outcomes via sampler."""
        quad = {(0, 1): 1.0}
        linear = {0: 0.5, 1: -0.5}

        @qmc.qkernel
        def circuit(
            quad_: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear_: qmc.Dict[qmc.UInt, qmc.Float],
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qaoa_state(
                qmc.uint(1), quad_, linear_, qmc.uint(2), gammas, betas
            )
            return qmc.measure(q)

        gammas = np.array([0.5])
        betas = np.array([0.8])
        transpiler = QuriPartsCircuitTranspiler()
        exe = transpiler.transpile(
            circuit,
            bindings={
                "quad_": quad,
                "linear_": linear,
                "gammas": gammas,
                "betas": betas,
            },
        )
        executor = transpiler.executor()
        job = exe.sample(executor, bindings={}, shots=500)
        result = job.result()
        total = sum(count for _, count in result.results)
        assert total == 500


# ============================================================================
# 9. All Four Bell States
# ============================================================================


class TestAllFourBellStates:
    """Test creation and verification of all four Bell states."""

    def test_bell_00(self):
        """|Φ+> = (|00> + |11>) / √2."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        assert statevectors_equal(sv, bell_state(0))

    def test_bell_01(self):
        """|Φ-> = (|00> - |11>) / √2 via X on control before H."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        assert statevectors_equal(sv, bell_state(1))

    def test_bell_10(self):
        """|Ψ+> = (|01> + |10>) / √2 via X on target."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[1] = qmc.x(q[1])
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        assert statevectors_equal(sv, bell_state(2))

    def test_bell_11(self):
        """|Ψ-> = (|01> - |10>) / √2 via X on both."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        assert statevectors_equal(sv, bell_state(3))


# ============================================================================
# 10. GHZ State Parametrised
# ============================================================================


class TestGHZStateParametrised:
    """Test GHZ state creation for varying qubit counts."""

    @pytest.mark.parametrize("n_qubits", [2, 3, 4, 5])
    def test_ghz_state(self, n_qubits):
        """GHZ(n) = (|0...0> + |1...1>) / √2."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q[0] = qmc.h(q[0])
            for i in qmc.range(n - 1):
                q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": n_qubits})
        sv = _run_statevector(qc)
        expected = np.zeros(2**n_qubits, dtype=complex)
        expected[0] = 1 / np.sqrt(2)
        expected[-1] = 1 / np.sqrt(2)
        assert statevectors_equal(sv, expected)


# ============================================================================
# 11. Deutsch-Jozsa Algorithm Pattern
# ============================================================================


class TestDeutschJozsaAlgorithm:
    """Test Deutsch-Jozsa algorithm pattern using available gates."""

    def test_constant_zero_oracle(self):
        """Constant-zero oracle: DJ outputs |0...0>."""

        @qmc.qkernel
        def dj_constant_zero() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            # Input qubits in |+>
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            # Ancilla in |->
            q[2] = qmc.x(q[2])
            q[2] = qmc.h(q[2])
            # No oracle gates (constant zero)
            # Final H on input qubits
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(dj_constant_zero)
        sv = _run_statevector(qc)
        # Input qubits should be in |00> -> index 0 and 4 (ancilla in |->)
        # Ancilla qubit (q2) is in |1> state from X gate
        # Measurement of input qubits: should be |00>
        # sv indices: |q2 q1 q0> — for constant oracle, input qubits return to |00>
        # The state is |-> ⊗ |00> = (|100> - |000>)/√2
        # Probabilities: only measure q0=0, q1=0 for input qubits
        prob_00 = np.abs(sv[0]) ** 2 + np.abs(sv[4]) ** 2
        assert np.isclose(prob_00, 1.0, atol=1e-10)

    def test_balanced_oracle(self):
        """Balanced oracle (CX): DJ outputs non-zero on input qubits."""

        @qmc.qkernel
        def dj_balanced() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            # Input qubits in |+>
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            # Ancilla in |->
            q[2] = qmc.x(q[2])
            q[2] = qmc.h(q[2])
            # Balanced oracle: CX(q0, q2)
            q[0], q[2] = qmc.cx(q[0], q[2])
            # Final H on input qubits
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(dj_balanced)
        sv = _run_statevector(qc)
        # For balanced oracle, input qubits should NOT be all |00>
        prob_00 = np.abs(sv[0]) ** 2 + np.abs(sv[4]) ** 2
        assert prob_00 < 0.01  # Should be ~0


# ============================================================================
# 12. Bernstein-Vazirani Algorithm Pattern
# ============================================================================


class TestBernsteinVaziraniAlgorithm:
    """Test Bernstein-Vazirani algorithm: recover secret string via H-oracle-H.

    Oracle CX(q_i, ancilla) encodes bit s_i=1 of the secret string.
    After H on input qubits, measurement reveals the secret.
    """

    def _check_bv_result(self, sv, n_input, expected_secret_idx):
        """Verify BV outcome: ancilla in |-> splits probability over q_a=0,1."""
        ancilla_bit = 2**n_input
        # Probability of measuring the secret on input qubits
        prob = (
            np.abs(sv[expected_secret_idx]) ** 2
            + np.abs(sv[expected_secret_idx + ancilla_bit]) ** 2
        )
        assert np.isclose(prob, 1.0, atol=1e-6)

    def test_bv_secret_10(self):
        """Secret s=(1,0): CX(q0, ancilla) → recover q0=1, q1=0."""

        @qmc.qkernel
        def bv() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[2] = qmc.x(q[2])
            q[2] = qmc.h(q[2])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            q[0], q[2] = qmc.cx(q[0], q[2])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(bv)
        sv = _run_statevector(qc)
        # s=(1,0) → q0=1, q1=0 → little-endian index = 1
        self._check_bv_result(sv, n_input=2, expected_secret_idx=1)

    def test_bv_secret_01(self):
        """Secret s=(0,1): CX(q1, ancilla) → recover q0=0, q1=1."""

        @qmc.qkernel
        def bv() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[2] = qmc.x(q[2])
            q[2] = qmc.h(q[2])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            q[1], q[2] = qmc.cx(q[1], q[2])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(bv)
        sv = _run_statevector(qc)
        # s=(0,1) → q0=0, q1=1 → little-endian index = 2
        self._check_bv_result(sv, n_input=2, expected_secret_idx=2)

    def test_bv_secret_11(self):
        """Secret s=(1,1): CX(q0,anc) + CX(q1,anc) → recover q0=1, q1=1."""

        @qmc.qkernel
        def bv() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[2] = qmc.x(q[2])
            q[2] = qmc.h(q[2])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            q[0], q[2] = qmc.cx(q[0], q[2])
            q[1], q[2] = qmc.cx(q[1], q[2])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(bv)
        sv = _run_statevector(qc)
        # s=(1,1) → q0=1, q1=1 → little-endian index = 3
        self._check_bv_result(sv, n_input=2, expected_secret_idx=3)

    def test_bv_secret_101(self):
        """Secret s=(1,0,1): CX(q0,anc) + CX(q2,anc) → recover 101."""

        @qmc.qkernel
        def bv() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[3] = qmc.x(q[3])
            q[3] = qmc.h(q[3])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            q[2] = qmc.h(q[2])
            q[0], q[3] = qmc.cx(q[0], q[3])
            q[2], q[3] = qmc.cx(q[2], q[3])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            q[2] = qmc.h(q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(bv)
        sv = _run_statevector(qc)
        # s=(1,0,1) → q0=1, q1=0, q2=1 → little-endian index = 1 + 4 = 5
        self._check_bv_result(sv, n_input=3, expected_secret_idx=5)


# ============================================================================
# 13. Expectation Value Pipeline
# ============================================================================


class TestExpvalQuriPartsPipeline:
    """Test the observable expectation value pipeline through QuriParts.

    The Hamiltonian must be passed as an Observable parameter via bindings,
    not constructed inside the @qkernel function.
    """

    def test_expval_z_on_zero(self):
        """<0|Z|0> = 1."""
        H = qm_o.Z(0)

        @qmc.qkernel
        def circuit(obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit("q")
            return qmc.expval((q,), obs)

        transpiler = QuriPartsCircuitTranspiler()
        exe = transpiler.transpile(circuit, bindings={"obs": H})
        executor = transpiler.executor()
        result = exe.run(executor).result()
        assert np.isclose(result, 1.0, atol=1e-6)

    def test_expval_z_on_one(self):
        """<1|Z|1> = -1."""
        H = qm_o.Z(0)

        @qmc.qkernel
        def circuit(obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit("q")
            q = qmc.x(q)
            return qmc.expval((q,), obs)

        transpiler = QuriPartsCircuitTranspiler()
        exe = transpiler.transpile(circuit, bindings={"obs": H})
        executor = transpiler.executor()
        result = exe.run(executor).result()
        assert np.isclose(result, -1.0, atol=1e-6)

    def test_expval_x_on_plus(self):
        """<+|X|+> = 1."""
        H = qm_o.X(0)

        @qmc.qkernel
        def circuit(obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.expval((q,), obs)

        transpiler = QuriPartsCircuitTranspiler()
        exe = transpiler.transpile(circuit, bindings={"obs": H})
        executor = transpiler.executor()
        result = exe.run(executor).result()
        assert np.isclose(result, 1.0, atol=1e-6)

    def test_expval_zz_bell_state(self):
        """<Φ+|Z⊗Z|Φ+> = 1 for Bell state."""
        H = qm_o.Z(0) * qm_o.Z(1)

        @qmc.qkernel
        def circuit(obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.expval(q, obs)

        transpiler = QuriPartsCircuitTranspiler()
        exe = transpiler.transpile(circuit, bindings={"obs": H})
        executor = transpiler.executor()
        result = exe.run(executor).result()
        assert np.isclose(result, 1.0, atol=1e-6)

    def test_expval_multi_term_hamiltonian(self):
        """Expectation of H = 0.5*Z0*Z1 + 0.25*(X0 + X1) on |++>."""
        H = 0.5 * qm_o.Z(0) * qm_o.Z(1) + 0.25 * (qm_o.X(0) + qm_o.X(1))

        @qmc.qkernel
        def circuit(obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            return qmc.expval(q, obs)

        transpiler = QuriPartsCircuitTranspiler()
        exe = transpiler.transpile(circuit, bindings={"obs": H})
        executor = transpiler.executor()
        result = exe.run(executor).result()
        # <++|Z0Z1|++> = 0, <++|X0|++> = 1, <++|X1|++> = 1
        # Expected = 0.5 * 0 + 0.25 * (1 + 1) = 0.5
        assert np.isclose(result, 0.5, atol=1e-6)

    def test_expval_parametric(self):
        """Parametric circuit with expectation value."""
        H = qm_o.Z(0)

        @qmc.qkernel
        def circuit(theta: qmc.Float, obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit("q")
            q = qmc.ry(q, theta)
            return qmc.expval((q,), obs)

        transpiler = QuriPartsCircuitTranspiler()
        executor = transpiler.executor()
        # theta=0: <0|Z|0>=1
        exe_0 = transpiler.transpile(circuit, bindings={"theta": 0.0, "obs": H})
        result_0 = exe_0.run(executor).result()
        assert np.isclose(result_0, 1.0, atol=1e-6)

        # theta=pi: <1|Z|1>=-1
        exe_pi = transpiler.transpile(circuit, bindings={"theta": np.pi, "obs": H})
        result_pi = exe_pi.run(executor).result()
        assert np.isclose(result_pi, -1.0, atol=1e-6)


# ============================================================================
# 14. FQAOA Integration
# ============================================================================


class TestFQAOAIntegration:
    """Test FQAOA circuit building blocks through QuriParts."""

    def test_initial_occupations(self):
        """initial_occupations applies X to first k qubits."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q = initial_occupations(q, qmc.uint(2))
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # First 2 qubits are |1>, rest |0> -> |0011> = index 3
        expected = computational_basis_state(4, 3)
        assert statevectors_equal(sv, expected)

    def test_hopping_gate_normalization(self):
        """hopping_gate preserves state normalization."""

        @qmc.qkernel
        def circuit(beta: qmc.Float, hop: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[0] = qmc.x(q[0])
            q = hopping_gate(q, qmc.uint(0), qmc.uint(1), beta, hop)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"beta": 0.3, "hop": 1.0}
        )
        sv = _run_statevector(qc)
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-10)

    def test_mixer_layer_normalization(self):
        """mixer_layer preserves normalization."""

        @qmc.qkernel
        def circuit(beta: qmc.Float, hop: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q = initial_occupations(q, qmc.uint(2))
            q = mixer_layer(q, beta, hop, qmc.uint(4))
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"beta": 0.5, "hop": 1.0}
        )
        sv = _run_statevector(qc)
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-10)

    def test_cost_layer_normalization(self):
        """cost_layer preserves normalization."""
        linear = {0: 0.5, 1: -0.3}
        quad = {(0, 1): 1.0}

        @qmc.qkernel
        def circuit(
            gamma: qmc.Float,
            linear_: qmc.Dict[qmc.UInt, qmc.Float],
            quad_: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            for i in qmc.range(2):
                q[i] = qmc.h(q[i])
            q = cost_layer(q, gamma, linear_, quad_)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(
            circuit,
            bindings={"gamma": 0.4, "linear_": linear, "quad_": quad},
        )
        sv = _run_statevector(qc)
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-10)

    @pytest.mark.xfail(
        reason="Bug #5: sub-kernel internal qubit_array not in qubit_map",
        strict=True,
    )
    def test_fqaoa_state_end_to_end(self):
        """Full fqaoa_state: initial + Givens + layers."""
        linear = {0: 0.5, 1: -0.3, 2: 0.2, 3: -0.1}
        quad = {(0, 1): 1.0, (2, 3): 0.5}
        # Simple Givens rotation: swap adjacent pairs
        givens_ij = np.array([[0, 1]], dtype=int)
        givens_theta = np.array([np.pi / 4])

        @qmc.qkernel
        def circuit(
            linear_: qmc.Dict[qmc.UInt, qmc.Float],
            quad_: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            givens_ij_: qmc.Matrix[qmc.UInt],
            givens_theta_: qmc.Vector[qmc.Float],
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = fqaoa_state(
                qmc.uint(1),
                linear_,
                quad_,
                qmc.uint(4),
                qmc.uint(2),
                givens_ij_,
                givens_theta_,
                qmc.float_(1.0),
                gammas,
                betas,
            )
            return qmc.measure(q)

        gammas = np.array([0.3])
        betas = np.array([0.7])
        _, qc = _transpile_and_get_circuit(
            circuit,
            bindings={
                "linear_": linear,
                "quad_": quad,
                "givens_ij_": givens_ij,
                "givens_theta_": givens_theta,
                "gammas": gammas,
                "betas": betas,
            },
        )
        sv = _run_statevector(qc)
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-10)


# ============================================================================
# 15. Deep Nested QKernel Composition
# ============================================================================


class TestDeepNestedQKernelComposition:
    """Test deeply nested sub-kernel composition."""

    def test_three_level_nesting(self):
        """Three levels of sub-kernel nesting produce correct result."""

        @qmc.qkernel
        def level_0(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def level_1(q: qmc.Qubit) -> qmc.Qubit:
            q = level_0(q)
            q = qmc.rx(q, 0.5)
            return q

        @qmc.qkernel
        def level_2() -> qmc.Bit:
            q = qmc.qubit("q")
            q = level_1(q)
            q = qmc.ry(q, 0.3)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(level_2)
        sv = _run_statevector(qc)

        # Manual: H -> RX(0.5) -> RY(0.3) on |0>
        state = all_zeros_state(1)
        state = GATE_SPECS["H"].matrix_fn() @ state
        state = GATE_SPECS["RX"].matrix_fn(0.5) @ state
        state = GATE_SPECS["RY"].matrix_fn(0.3) @ state
        assert statevectors_equal(sv, state)

    def test_diamond_sub_kernel_reuse(self):
        """Same sub-kernel reused in multiple places."""

        @qmc.qkernel
        def apply_h(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = apply_h(q[0])
            q[1] = apply_h(q[1])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # H⊗H|00> -> |++>, then CX|++>
        state = all_zeros_state(2)
        hh = tensor_product(GATE_SPECS["H"].matrix_fn(), GATE_SPECS["H"].matrix_fn())
        state = hh @ state
        state = GATE_SPECS["CX"].matrix_fn() @ state
        assert statevectors_equal(sv, state)


# ============================================================================
# 16. Phase Kickback Pattern
# ============================================================================


class TestPhaseKickbackPattern:
    """Test phase kickback — a key quantum computing primitive."""

    def test_phase_kickback_h_cx(self):
        """Phase kickback: CX with target in |-> kicks phase to control."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            # Control in |+>
            q[0] = qmc.h(q[0])
            # Target in |->
            q[1] = qmc.x(q[1])
            q[1] = qmc.h(q[1])
            # CX: kicks phase to control
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # After kickback, control should be in |-> and target stays in |->
        # |-> ⊗ |-> = (|0>-|1>)/√2 ⊗ (|0>-|1>)/√2
        expected = tensor_product(minus_state(), minus_state())
        assert statevectors_equal(sv, expected)


# ============================================================================
# 17. Entanglement and Parity Patterns
# ============================================================================


class TestEntanglementAndParityPatterns:
    """Test entanglement verification and parity computations."""

    def test_parity_check_even(self):
        """Parity check: CX chain computes parity in ancilla."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            # Prepare |110>: q0=0, q1=1, q2=1
            q[1] = qmc.x(q[1])
            q[2] = qmc.x(q[2])
            # Use q0 as parity ancilla
            q[1], q[0] = qmc.cx(q[1], q[0])
            q[2], q[0] = qmc.cx(q[2], q[0])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # Parity of (1,1) = 0, so q0 should be |0>
        # Final state: |110> = index 6
        expected = computational_basis_state(3, 6)
        assert statevectors_equal(sv, expected)

    def test_parity_check_odd(self):
        """Parity check: odd number of 1s flips ancilla."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            # Prepare |100>: q0=0, q1=0, q2=1
            q[2] = qmc.x(q[2])
            # Use q0 as parity ancilla
            q[1], q[0] = qmc.cx(q[1], q[0])
            q[2], q[0] = qmc.cx(q[2], q[0])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # Parity of (0,1) = 1, so q0 flips to |1>
        # Final state: |101> = index 5
        expected = computational_basis_state(3, 5)
        assert statevectors_equal(sv, expected)

    def test_entanglement_witness_via_cz(self):
        """CZ on |++> creates entangled state distinguishable from product."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            q[0], q[1] = qmc.cz(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # CZ|++> = (|00> + |01> + |10> - |11>)/2
        expected = np.array([1, 1, 1, -1], dtype=complex) / 2
        assert statevectors_equal(sv, expected)

    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_full_superposition_normalization(self, n_qubits):
        """H on all qubits produces uniform superposition."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": n_qubits})
        sv = _run_statevector(qc)
        expected = np.ones(2**n_qubits, dtype=complex) / np.sqrt(2**n_qubits)
        assert statevectors_equal(sv, expected)


# ============================================================================
# 18. Advanced Parameter Handling
# ============================================================================


class TestAdvancedParameterHandling:
    """Test advanced parametric circuit features."""

    def test_multiple_parameters_bound(self):
        """Multiple parameters all bound at transpile time."""

        @qmc.qkernel
        def circuit(a: qmc.Float, b: qmc.Float, c: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, a)
            q = qmc.ry(q, b)
            q = qmc.rz(q, c)
            return qmc.measure(q)

        a, b, c = 0.3, 0.7, 1.2
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"a": a, "b": b, "c": c}
        )
        sv = _run_statevector(qc)

        state = all_zeros_state(1)
        state = GATE_SPECS["RX"].matrix_fn(a) @ state
        state = GATE_SPECS["RY"].matrix_fn(b) @ state
        state = GATE_SPECS["RZ"].matrix_fn(c) @ state
        assert statevectors_equal(sv, state)

    def test_parametric_expval_sweep(self):
        """Sweep parameter and verify expectation values follow cos(theta)."""
        H = qm_o.Z(0)

        @qmc.qkernel
        def circuit(theta: qmc.Float, obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit("q")
            q = qmc.ry(q, theta)
            return qmc.expval((q,), obs)

        transpiler = QuriPartsCircuitTranspiler()
        executor = transpiler.executor()

        for theta in [0.0, np.pi / 4, np.pi / 2, np.pi]:
            exe = transpiler.transpile(
                circuit, bindings={"theta": theta, "obs": H}
            )
            result = exe.run(executor).result()
            expected = np.cos(theta)
            assert np.isclose(result, expected, atol=1e-5), (
                f"theta={theta}: got {result}, expected {expected}"
            )


# ============================================================================
# 19. Variational Classifier Pattern
# ============================================================================


class TestVariationalClassifierPattern:
    """Test a minimal variational classifier ansatz."""

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_classifier_ansatz_normalization(self, seed):
        """Variational classifier ansatz produces normalized state."""
        rng = np.random.default_rng(seed)

        @qmc.qkernel
        def classifier(thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            # Feature encoding layer
            q = ry_layer(q, thetas, qmc.uint(0))
            # Entangling layer
            q = cz_entangling_layer(q)
            # Variational layer
            q = ry_layer(q, thetas, qmc.uint(2))
            return qmc.measure(q)

        params = rng.uniform(0, 2 * np.pi, size=4)
        _, qc = _transpile_and_get_circuit(
            classifier, bindings={"thetas": params}
        )
        sv = _run_statevector(qc)
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-10)

    def test_classifier_sampling(self):
        """Variational classifier produces valid measurement outcomes."""

        @qmc.qkernel
        def classifier(thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q = ry_layer(q, thetas, qmc.uint(0))
            q = cz_entangling_layer(q)
            q = ry_layer(q, thetas, qmc.uint(2))
            return qmc.measure(q)

        params = np.array([0.5, 0.3, 0.7, 0.1])
        transpiler = QuriPartsCircuitTranspiler()
        exe = transpiler.transpile(classifier, bindings={"thetas": params})
        executor = transpiler.executor()
        job = exe.sample(executor, bindings={}, shots=200)
        result = job.result()
        total = sum(count for _, count in result.results)
        assert total == 200


# ============================================================================
# 20. GHZ State Parametrised
# ============================================================================


class TestGHZStateParametrised:
    """Test GHZ state creation for varying qubit counts."""

    @pytest.mark.parametrize("n_qubits", [2, 3, 4, 5])
    def test_ghz_state(self, n_qubits):
        """GHZ(n) = (|0...0> + |1...1>) / √2."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q[0] = qmc.h(q[0])
            for i in qmc.range(n - 1):
                q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": n_qubits})
        sv = _run_statevector(qc)
        expected = np.zeros(2**n_qubits, dtype=complex)
        expected[0] = 1 / np.sqrt(2)
        expected[-1] = 1 / np.sqrt(2)
        assert statevectors_equal(sv, expected)


# ============================================================================
# 21. Deutsch-Jozsa Algorithm Pattern
# ============================================================================


class TestDeutschJozsaAlgorithm:
    """Test Deutsch-Jozsa algorithm pattern using available gates."""

    def test_constant_zero_oracle(self):
        """Constant-zero oracle: DJ outputs |0...0>."""

        @qmc.qkernel
        def dj_constant_zero() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            # Input qubits in |+>
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            # Ancilla in |->
            q[2] = qmc.x(q[2])
            q[2] = qmc.h(q[2])
            # No oracle gates (constant zero)
            # Final H on input qubits
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(dj_constant_zero)
        sv = _run_statevector(qc)
        # For constant oracle, input qubits return to |00>
        # The state is |-> ⊗ |00> = (|100> - |000>)/√2
        # Probabilities: only measure q0=0, q1=0 for input qubits
        prob_00 = np.abs(sv[0]) ** 2 + np.abs(sv[4]) ** 2
        assert np.isclose(prob_00, 1.0, atol=1e-10)

    def test_balanced_oracle(self):
        """Balanced oracle (CX): DJ outputs non-zero on input qubits."""

        @qmc.qkernel
        def dj_balanced() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            # Input qubits in |+>
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            # Ancilla in |->
            q[2] = qmc.x(q[2])
            q[2] = qmc.h(q[2])
            # Balanced oracle: CX(q0, q2)
            q[0], q[2] = qmc.cx(q[0], q[2])
            # Final H on input qubits
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(dj_balanced)
        sv = _run_statevector(qc)
        # For balanced oracle, input qubits should NOT be all |00>
        prob_00 = np.abs(sv[0]) ** 2 + np.abs(sv[4]) ** 2
        assert prob_00 < 0.01  # Should be ~0


# ============================================================================
# 22. Bernstein-Vazirani Algorithm Pattern
# ============================================================================


class TestBernsteinVaziraniAlgorithm:
    """Test Bernstein-Vazirani algorithm: recover secret string via H-oracle-H.

    Oracle CX(q_i, ancilla) encodes bit s_i=1 of the secret string.
    After H on input qubits, measurement reveals the secret.
    """

    def _check_bv_result(self, sv, n_input, expected_secret_idx):
        """Verify BV outcome: ancilla in |-> splits probability over q_a=0,1."""
        ancilla_bit = 2**n_input
        # Probability of measuring the secret on input qubits
        prob = (
            np.abs(sv[expected_secret_idx]) ** 2
            + np.abs(sv[expected_secret_idx + ancilla_bit]) ** 2
        )
        assert np.isclose(prob, 1.0, atol=1e-6)

    def test_bv_secret_10(self):
        """Secret s=(1,0): CX(q0, ancilla) → recover q0=1, q1=0."""

        @qmc.qkernel
        def bv() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[2] = qmc.x(q[2])
            q[2] = qmc.h(q[2])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            q[0], q[2] = qmc.cx(q[0], q[2])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(bv)
        sv = _run_statevector(qc)
        # s=(1,0) → q0=1, q1=0 → little-endian index = 1
        self._check_bv_result(sv, n_input=2, expected_secret_idx=1)

    def test_bv_secret_01(self):
        """Secret s=(0,1): CX(q1, ancilla) → recover q0=0, q1=1."""

        @qmc.qkernel
        def bv() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[2] = qmc.x(q[2])
            q[2] = qmc.h(q[2])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            q[1], q[2] = qmc.cx(q[1], q[2])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(bv)
        sv = _run_statevector(qc)
        # s=(0,1) → q0=0, q1=1 → little-endian index = 2
        self._check_bv_result(sv, n_input=2, expected_secret_idx=2)

    def test_bv_secret_11(self):
        """Secret s=(1,1): CX(q0,anc) + CX(q1,anc) → recover q0=1, q1=1."""

        @qmc.qkernel
        def bv() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[2] = qmc.x(q[2])
            q[2] = qmc.h(q[2])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            q[0], q[2] = qmc.cx(q[0], q[2])
            q[1], q[2] = qmc.cx(q[1], q[2])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(bv)
        sv = _run_statevector(qc)
        # s=(1,1) → q0=1, q1=1 → little-endian index = 3
        self._check_bv_result(sv, n_input=2, expected_secret_idx=3)

    def test_bv_secret_101(self):
        """Secret s=(1,0,1): CX(q0,anc) + CX(q2,anc) → recover 101."""

        @qmc.qkernel
        def bv() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[3] = qmc.x(q[3])
            q[3] = qmc.h(q[3])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            q[2] = qmc.h(q[2])
            q[0], q[3] = qmc.cx(q[0], q[3])
            q[2], q[3] = qmc.cx(q[2], q[3])
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            q[2] = qmc.h(q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(bv)
        sv = _run_statevector(qc)
        # s=(1,0,1) → q0=1, q1=0, q2=1 → little-endian index = 1 + 4 = 5
        self._check_bv_result(sv, n_input=3, expected_secret_idx=5)


# ============================================================================
# 23. Error Cases
# ============================================================================


class TestErrorCases:
    """Verify that invalid inputs raise the correct exception types."""

    def test_qubit_aliasing_cx_raises(self):
        """Using the same qubit for both CX operands raises QubitAliasError."""
        from qamomile.circuit.transpiler.errors import QubitAliasError

        with pytest.raises(QubitAliasError):

            @qmc.qkernel
            def circuit() -> qmc.Bit:
                q = qmc.qubit("q")
                q, q = qmc.cx(q, q)
                return qmc.measure(q)

            circuit.build()

    def test_qubit_aliasing_cz_raises(self):
        """Using the same qubit for both CZ operands raises QubitAliasError."""
        from qamomile.circuit.transpiler.errors import QubitAliasError

        with pytest.raises(QubitAliasError):

            @qmc.qkernel
            def circuit() -> qmc.Bit:
                q = qmc.qubit("q")
                q, q = qmc.cz(q, q)
                return qmc.measure(q)

            circuit.build()

    def test_missing_required_binding_raises(self):
        """Transpiling a kernel that needs a binding without it raises."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        transpiler = QuriPartsCircuitTranspiler()
        # Provide a wrong key so 'theta' is still missing
        with pytest.raises(ValueError, match="Unknown argument"):
            transpiler.transpile(circuit, bindings={"alpha": 0.5})

    def test_wrong_binding_type_for_uint_raises(self):
        """Passing wrong binding type for UInt parameter raises an error."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        transpiler = QuriPartsCircuitTranspiler()
        # Provide a string for a UInt param — should fail
        with pytest.raises((TypeError, ValueError)):
            transpiler.transpile(circuit, bindings={"n": "not_a_number"})
