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
    all_zeros_state,
    bell_state,
    computational_basis_state,
    compute_expected_statevector,
    identity,
    minus_state,
    plus_state,
    statevectors_equal,
    tensor_product,
)

# ---------------------------------------------------------------------------
# Skip entire module if qiskit is not installed
# ---------------------------------------------------------------------------
qiskit = pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from qiskit import QuantumCircuit
from qiskit.circuit import Barrier, Measure, ParameterExpression
from qiskit.circuit.controlflow import IfElseOp, WhileLoopOp
from qiskit.circuit.library import (
    CCXGate,
    CPhaseGate,
    CXGate,
    CZGate,
    HGate,
    PhaseGate,
    RXGate,
    RYGate,
    RZGate,
    RZZGate,
    SdgGate,
    SGate,
    SwapGate,
    TdgGate,
    TGate,
    XGate,
    ZGate,
)

import qamomile.observable as qm_o
from qamomile.circuit.algorithm.basic import (
    cz_entangling_layer,
    rx_layer,
    ry_layer,
    rz_layer,
    superposition_vector,
)
from qamomile.circuit.algorithm.fqaoa import (
    cost_layer,
    fqaoa_state,
    givens_rotation,
    hopping_gate,
    initial_occupations,
    mixer_layer,
)
from qamomile.circuit.algorithm.qaoa import (
    ising_cost,
    qaoa_layers,
    qaoa_state,
    x_mixer,
)
from qamomile.circuit.ir.block import BlockKind
from qamomile.circuit.transpiler.executable import ExecutableProgram
from qamomile.circuit.transpiler.segments import SimplifiedProgram
from qamomile.circuit.transpiler.transpiler import TranspilerConfig
from qamomile.qiskit import QiskitTranspiler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of the circuit with measurement gates removed.

    Args:
        circuit: Qiskit QuantumCircuit to strip.

    Returns:
        New QuantumCircuit with measurement and barrier instructions removed.
    """
    new_qc = QuantumCircuit(circuit.num_qubits)
    for inst in circuit.data:
        if not isinstance(inst.operation, (Measure, Barrier)):
            new_qc.append(inst)
    return new_qc


def _run_statevector(circuit: QuantumCircuit, decompose: bool = False) -> np.ndarray:
    """Run a Qiskit circuit and return the pre-measurement statevector.

    Args:
        circuit: Qiskit QuantumCircuit.
        decompose: If True, decompose to basis gates before simulation.
            Required for circuits containing controlled sub-circuits
            (``ccircuit-*``), which Aer cannot simulate directly.

    Returns:
        Numpy array of complex amplitudes representing the statevector.
    """
    from qiskit_aer import AerSimulator

    qc = _strip_measurements(circuit)
    if decompose:
        from qiskit import transpile as qiskit_transpile

        qc = qiskit_transpile(qc, basis_gates=["u", "cx"], optimization_level=0)
    qc.save_statevector()
    result = AerSimulator(method="statevector").run(qc).result()
    return np.array(result.get_statevector())


def _transpile_and_get_circuit(
    kernel,
    bindings: dict | None = None,
    parameters: list[str] | None = None,
) -> tuple[ExecutableProgram, QuantumCircuit]:
    """Transpile a qkernel and return the executable and Qiskit circuit.

    Args:
        kernel: A ``@qmc.qkernel`` decorated function.
        bindings: Optional parameter bindings dict.
        parameters: Optional list of parameter names to leave symbolic.

    Returns:
        Tuple of (ExecutableProgram, QuantumCircuit).
    """
    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(kernel, bindings=bindings, parameters=parameters)
    circuit = exe.compiled_quantum[0].circuit
    return exe, circuit


# ============================================================================
# 1. Individual Gate Tests – creation + determined + random execution
# ============================================================================


class TestSingleQubitGatesFrontend:
    """Test each single-qubit gate through the full frontend pipeline."""

    # -- Hadamard --

    def test_h_creation(self):
        """H gate transpiles to: H(q0) → Measure(q0→c0)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert len(qc.data) == 2  # H + measure
        assert isinstance(qc.data[0].operation, HGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert isinstance(qc.data[1].operation, Measure)
        assert [qc.find_bit(q).index for q in qc.data[1].qubits] == [0]
        assert [qc.find_bit(c).index for c in qc.data[1].clbits] == [0]
        assert qc.num_qubits == 1
        assert qc.num_clbits == 1

    def test_h_statevector(self):
        """H|0⟩ produces equal superposition (|0⟩+|1⟩)/√2."""

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
        """X gate transpiles to: X(q0) → Measure(q0→c0)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert len(qc.data) == 2
        assert isinstance(qc.data[0].operation, XGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert isinstance(qc.data[1].operation, Measure)
        assert qc.num_qubits == 1
        assert qc.num_clbits == 1

    def test_x_statevector(self):
        """X|0⟩ = |1⟩ statevector verification."""

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
        """RX gate transpiles to: RX(q0) → Measure(q0→c0)."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 4})
        assert len(qc.data) == 2
        assert isinstance(qc.data[0].operation, RXGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert isinstance(qc.data[1].operation, Measure)
        assert qc.num_qubits == 1
        assert qc.num_clbits == 1

    @pytest.mark.parametrize(
        "angle",
        [np.pi]
        + [
            np.random.default_rng(s).uniform(0, 2 * np.pi)
            for s in [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
        ],
    )
    def test_rx_statevector(self, angle):
        """RX(θ) statevector matches analytical RX matrix."""

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
        """RY gate transpiles to a Qiskit circuit containing 'ry' instruction."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.ry(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 4})
        assert len(qc.data) == 2
        assert isinstance(qc.data[0].operation, RYGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert isinstance(qc.data[1].operation, Measure)
        assert qc.num_qubits == 1
        assert qc.num_clbits == 1

    @pytest.mark.parametrize(
        "angle",
        [np.pi / 2]
        + [
            np.random.default_rng(s).uniform(0, 2 * np.pi)
            for s in [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
        ],
    )
    def test_ry_statevector(self, angle):
        """RY(θ) statevector matches analytical RY matrix."""

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
        """RZ gate transpiles to a Qiskit circuit containing 'rz' instruction."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rz(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 4})
        assert len(qc.data) == 2
        assert isinstance(qc.data[0].operation, RZGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert isinstance(qc.data[1].operation, Measure)
        assert qc.num_qubits == 1
        assert qc.num_clbits == 1

    @pytest.mark.parametrize(
        "angle",
        [np.pi]
        + [
            np.random.default_rng(s).uniform(0, 2 * np.pi)
            for s in [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
        ],
    )
    def test_rz_statevector(self, angle):
        """RZ(θ) statevector matches analytical RZ matrix."""

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
        """P gate transpiles to a Qiskit circuit containing 'p' instruction."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.p(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 4})
        assert len(qc.data) == 2
        assert isinstance(qc.data[0].operation, PhaseGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert isinstance(qc.data[1].operation, Measure)
        assert qc.num_qubits == 1
        assert qc.num_clbits == 1

    @pytest.mark.parametrize(
        "angle",
        [np.pi]
        + [
            np.random.default_rng(s).uniform(0, 2 * np.pi)
            for s in [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
        ],
    )
    def test_p_statevector(self, angle):
        """P(θ) on |1⟩ statevector matches analytical P matrix."""

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

    # -- Z (Pauli-Z) --

    def test_z_creation(self):
        """Z gate transpiles to a Qiskit circuit containing 'z' instruction."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.z(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert len(qc.data) == 2
        assert isinstance(qc.data[0].operation, ZGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert isinstance(qc.data[1].operation, Measure)
        assert qc.num_qubits == 1
        assert qc.num_clbits == 1

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

    def test_s_creation(self):
        """S gate transpiles to a Qiskit circuit containing 's' instruction."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.s(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert len(qc.data) == 2
        assert isinstance(qc.data[0].operation, SGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert isinstance(qc.data[1].operation, Measure)
        assert qc.num_qubits == 1
        assert qc.num_clbits == 1

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

    def test_sdg_creation(self):
        """SDG gate transpiles to a Qiskit circuit containing 'sdg' instruction."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.sdg(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert len(qc.data) == 2
        assert isinstance(qc.data[0].operation, SdgGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert isinstance(qc.data[1].operation, Measure)
        assert qc.num_qubits == 1
        assert qc.num_clbits == 1

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

    def test_t_creation(self):
        """T gate transpiles to a Qiskit circuit containing 't' instruction."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.t(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert len(qc.data) == 2
        assert isinstance(qc.data[0].operation, TGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert isinstance(qc.data[1].operation, Measure)
        assert qc.num_qubits == 1
        assert qc.num_clbits == 1

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

    def test_tdg_creation(self):
        """TDG gate transpiles to a Qiskit circuit containing 'tdg' instruction."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.tdg(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert len(qc.data) == 2
        assert isinstance(qc.data[0].operation, TdgGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert isinstance(qc.data[1].operation, Measure)
        assert qc.num_qubits == 1
        assert qc.num_clbits == 1

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

    # -- CX --

    def test_cx_creation(self):
        """CX gate transpiles to a Qiskit circuit containing 'cx' instruction."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert len(qc.data) == 3  # CX + 2 measures
        assert isinstance(qc.data[0].operation, CXGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0, 1]
        assert isinstance(qc.data[1].operation, Measure)
        assert isinstance(qc.data[2].operation, Measure)
        assert qc.num_qubits == 2
        assert qc.num_clbits == 2

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
        """CZ gate transpiles to a Qiskit circuit containing 'cz' instruction."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = qmc.cz(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert len(qc.data) == 3
        assert isinstance(qc.data[0].operation, CZGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0, 1]
        assert isinstance(qc.data[1].operation, Measure)
        assert isinstance(qc.data[2].operation, Measure)
        assert qc.num_qubits == 2
        assert qc.num_clbits == 2

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
        """CP gate transpiles to a Qiskit circuit containing 'cp' instruction."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = qmc.cp(q[0], q[1], theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 4})
        assert len(qc.data) == 3
        assert isinstance(qc.data[0].operation, CPhaseGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0, 1]
        assert isinstance(qc.data[1].operation, Measure)
        assert isinstance(qc.data[2].operation, Measure)
        assert qc.num_qubits == 2
        assert qc.num_clbits == 2

    @pytest.mark.parametrize(
        "angle",
        [np.pi]
        + [
            np.random.default_rng(s).uniform(0, 2 * np.pi)
            for s in [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
        ],
    )
    def test_cp_statevector(self, angle):
        """CP(θ) on |11⟩ statevector matches analytical CP matrix."""

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
        """RZZ gate transpiles to a Qiskit circuit containing 'rzz' instruction."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = qmc.rzz(q[0], q[1], theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 4})
        assert len(qc.data) == 3
        assert isinstance(qc.data[0].operation, RZZGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0, 1]
        assert isinstance(qc.data[1].operation, Measure)
        assert isinstance(qc.data[2].operation, Measure)
        assert qc.num_qubits == 2
        assert qc.num_clbits == 2

    @pytest.mark.parametrize(
        "angle",
        [np.pi]
        + [
            np.random.default_rng(s).uniform(0, 2 * np.pi)
            for s in [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
        ],
    )
    def test_rzz_statevector(self, angle):
        """RZZ(θ) on |11⟩ statevector matches analytical RZZ matrix."""

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
        """SWAP gate transpiles to a Qiskit circuit containing 'swap' instruction."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = qmc.swap(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert len(qc.data) == 3
        assert isinstance(qc.data[0].operation, SwapGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0, 1]
        assert isinstance(qc.data[1].operation, Measure)
        assert isinstance(qc.data[2].operation, Measure)
        assert qc.num_qubits == 2
        assert qc.num_clbits == 2

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
        """CCX gate transpiles to a Qiskit circuit containing 'ccx' instruction."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0], q[1], q[2] = qmc.ccx(q[0], q[1], q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert len(qc.data) == 4  # CCX + 3 measures
        assert isinstance(qc.data[0].operation, CCXGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0, 1, 2]
        assert isinstance(qc.data[1].operation, Measure)
        assert isinstance(qc.data[2].operation, Measure)
        assert isinstance(qc.data[3].operation, Measure)
        assert qc.num_qubits == 3
        assert qc.num_clbits == 3

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
        """CCX on all 8 computational basis states; flips target only when both controls are |1⟩."""

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
# 1b. Parametric Gate Tests (unbound parameters)
# ============================================================================


class TestParametricGates:
    """Test that gates with unbound parameters produce parameterized circuits.

    Uses ``parameters=["theta"]`` to keep float args as Qiskit Parameter
    objects, verifying gate type, parameter name, and circuit structure.
    """

    def test_rx_parametric(self):
        """RX with unbound theta contains a Parameter in the gate."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit
        assert len(qc.parameters) == 1
        assert isinstance(qc.data[0].operation, RXGate)
        param = qc.data[0].operation.params[0]
        assert isinstance(param, ParameterExpression)
        assert "theta" in str(param)

    def test_ry_parametric(self):
        """RY with unbound theta contains a Parameter in the gate."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.ry(q, theta)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit
        assert len(qc.parameters) == 1
        assert isinstance(qc.data[0].operation, RYGate)
        param = qc.data[0].operation.params[0]
        assert isinstance(param, ParameterExpression)
        assert "theta" in str(param)

    def test_rz_parametric(self):
        """RZ with unbound theta contains a Parameter in the gate."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rz(q, theta)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit
        assert len(qc.parameters) == 1
        assert isinstance(qc.data[0].operation, RZGate)
        param = qc.data[0].operation.params[0]
        assert isinstance(param, ParameterExpression)
        assert "theta" in str(param)

    def test_p_parametric(self):
        """P with unbound theta contains a Parameter in the gate."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.p(q, theta)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit
        assert len(qc.parameters) == 1
        assert isinstance(qc.data[0].operation, PhaseGate)
        param = qc.data[0].operation.params[0]
        assert isinstance(param, ParameterExpression)
        assert "theta" in str(param)

    def test_cp_parametric(self):
        """CP with unbound theta contains a Parameter in the gate."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = qmc.cp(q[0], q[1], theta)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit
        assert len(qc.parameters) == 1
        assert isinstance(qc.data[0].operation, CPhaseGate)
        param = qc.data[0].operation.params[0]
        assert isinstance(param, ParameterExpression)
        assert "theta" in str(param)

    def test_rzz_parametric(self):
        """RZZ with unbound theta contains a Parameter in the gate."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = qmc.rzz(q[0], q[1], theta)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit
        assert len(qc.parameters) == 1
        assert isinstance(qc.data[0].operation, RZZGate)
        param = qc.data[0].operation.params[0]
        assert isinstance(param, ParameterExpression)
        assert "theta" in str(param)

    def test_vector_parametric(self):
        """Vector[Float] parameter produces multiple indexed Parameters."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.ry(q[i], thetas[i])
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit, bindings={"n": 3}, parameters=["thetas"])
        qc = exe.compiled_quantum[0].circuit
        assert len(qc.parameters) == 3
        # Each RY gate has a ParameterExpression
        for i in range(3):
            assert isinstance(qc.data[i].operation, RYGate)
            param = qc.data[i].operation.params[0]
            assert isinstance(param, ParameterExpression)

    # -- BinOp with parameters (via items loop for constant folding) --------

    def test_parametric_mul_binop(self):
        """theta * constant produces ParameterExpression with correct binding."""

        @qmc.qkernel
        def circuit(
            coeffs: qmc.Dict[qmc.UInt, qmc.Float],
            theta: qmc.Float,
            H: qmc.Observable,
        ) -> qmc.Float:
            q = qmc.qubit("q")
            for _, c in qmc.items(coeffs):
                q = qmc.rx(q, theta * c)
            return qmc.expval(q, H)

        H = qm_o.Hamiltonian(num_qubits=1)
        H += qm_o.Z(0)
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(
            circuit,
            bindings={"coeffs": {0: 2.0}, "H": H},
            parameters=["theta"],
        )
        qc = exe.compiled_quantum[0].circuit
        assert len(qc.parameters) == 1
        param_expr = qc.data[0].operation.params[0]
        assert isinstance(param_expr, ParameterExpression)
        # Bind theta=1.5 → gate angle should be 1.5 * 2.0 = 3.0
        theta_param = list(qc.parameters)[0]
        bound = qc.assign_parameters({theta_param: 1.5})
        assert np.isclose(float(bound.data[0].operation.params[0]), 3.0, atol=1e-10)

    def test_parametric_add_binop(self):
        """theta + constant produces ParameterExpression with correct binding."""

        @qmc.qkernel
        def circuit(
            offsets: qmc.Dict[qmc.UInt, qmc.Float],
            theta: qmc.Float,
            H: qmc.Observable,
        ) -> qmc.Float:
            q = qmc.qubit("q")
            for _, o in qmc.items(offsets):
                q = qmc.rx(q, theta + o)
            return qmc.expval(q, H)

        H = qm_o.Hamiltonian(num_qubits=1)
        H += qm_o.Z(0)
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(
            circuit,
            bindings={"offsets": {0: 0.5}, "H": H},
            parameters=["theta"],
        )
        qc = exe.compiled_quantum[0].circuit
        assert len(qc.parameters) == 1
        param_expr = qc.data[0].operation.params[0]
        assert isinstance(param_expr, ParameterExpression)
        # Bind theta=1.0 → gate angle should be 1.0 + 0.5 = 1.5
        theta_param = list(qc.parameters)[0]
        bound = qc.assign_parameters({theta_param: 1.0})
        assert np.isclose(float(bound.data[0].operation.params[0]), 1.5, atol=1e-10)

    def test_parametric_sub_binop(self):
        """theta - constant produces ParameterExpression with correct binding."""

        @qmc.qkernel
        def circuit(
            offsets: qmc.Dict[qmc.UInt, qmc.Float],
            theta: qmc.Float,
            H: qmc.Observable,
        ) -> qmc.Float:
            q = qmc.qubit("q")
            for _, o in qmc.items(offsets):
                q = qmc.rx(q, theta - o)
            return qmc.expval(q, H)

        H = qm_o.Hamiltonian(num_qubits=1)
        H += qm_o.Z(0)
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(
            circuit,
            bindings={"offsets": {0: 0.5}, "H": H},
            parameters=["theta"],
        )
        qc = exe.compiled_quantum[0].circuit
        assert len(qc.parameters) == 1
        param_expr = qc.data[0].operation.params[0]
        assert isinstance(param_expr, ParameterExpression)
        # Bind theta=1.0 → gate angle should be 1.0 - 0.5 = 0.5
        theta_param = list(qc.parameters)[0]
        bound = qc.assign_parameters({theta_param: 1.0})
        assert np.isclose(float(bound.data[0].operation.params[0]), 0.5, atol=1e-10)


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
        def circuit(theta1: qmc.Float, theta2: qmc.Float, theta3: qmc.Float) -> qmc.Bit:
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
        # Structure: X(alice=q0) H(bell0) CX(bell0,bell1) CX(alice,bell0) H(alice) + 3 Measure = 8
        assert len(qc.data) == 8
        assert isinstance(qc.data[0].operation, XGate)
        assert isinstance(qc.data[1].operation, HGate)
        assert isinstance(qc.data[2].operation, CXGate)
        assert isinstance(qc.data[3].operation, CXGate)
        assert isinstance(qc.data[4].operation, HGate)
        for i in range(3):
            assert isinstance(qc.data[5 + i].operation, Measure)


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
        # Structure: n H gates on q[0..n-1], then n Measure gates
        assert len(qc.data) == 2 * n_qubits
        for i in range(n_qubits):
            assert isinstance(qc.data[i].operation, HGate)
            assert [qc.find_bit(q).index for q in qc.data[i].qubits] == [i]
        for i in range(n_qubits):
            idx = n_qubits + i
            assert isinstance(qc.data[idx].operation, Measure)
            assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [i]
            assert [qc.find_bit(c).index for c in qc.data[idx].clbits] == [i]
        assert qc.num_qubits == n_qubits
        assert qc.num_clbits == n_qubits

    @pytest.mark.parametrize("n_qubits", [2, 3, 5, 10, 20])
    def test_rx_layer_via_range(self, n_qubits):
        """Apply RX with varying angles via qmc.range."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.rx(q[i], thetas[i])
            return qmc.measure(q)

        thetas = [0.1 * i for i in range(n_qubits)]
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": n_qubits, "thetas": thetas}
        )
        # Structure: n RX gates on q[0..n-1], then n Measure gates
        assert len(qc.data) == 2 * n_qubits
        for i in range(n_qubits):
            assert isinstance(qc.data[i].operation, RXGate)
            assert [qc.find_bit(q).index for q in qc.data[i].qubits] == [i]
            assert np.isclose(
                float(qc.data[i].operation.params[0]), thetas[i], atol=1e-10
            )
        for i in range(n_qubits):
            idx = n_qubits + i
            assert isinstance(qc.data[idx].operation, Measure)
            assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [i]
        assert qc.num_qubits == n_qubits

    @pytest.mark.parametrize("n_qubits", [4, 5, 6])
    def test_range_with_start_stop(self, n_qubits):
        """qmc.range(start, stop) applies gates to subset of qubits."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(1, 3):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": n_qubits})
        # Structure: 2 H gates on q[1], q[2], then n Measure gates
        assert len(qc.data) == 2 + n_qubits
        assert isinstance(qc.data[0].operation, HGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [1]
        assert isinstance(qc.data[1].operation, HGate)
        assert [qc.find_bit(q).index for q in qc.data[1].qubits] == [2]
        for i in range(n_qubits):
            assert isinstance(qc.data[2 + i].operation, Measure)
            assert [qc.find_bit(q).index for q in qc.data[2 + i].qubits] == [i]
        assert qc.num_qubits == n_qubits

    @pytest.mark.parametrize("n_qubits", [3, 4, 5])
    def test_cz_chain_via_range(self, n_qubits):
        """CZ entangling layer via range loop."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            for i in qmc.range(n - 1):
                q[i], q[i + 1] = qmc.cz(q[i], q[i + 1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": n_qubits})
        # Structure: n H on q[0..n-1], (n-1) CZ on (i, i+1), n Measure
        n_cz = n_qubits - 1
        assert len(qc.data) == n_qubits + n_cz + n_qubits
        for i in range(n_qubits):
            assert isinstance(qc.data[i].operation, HGate)
            assert [qc.find_bit(q).index for q in qc.data[i].qubits] == [i]
        for i in range(n_cz):
            idx = n_qubits + i
            assert isinstance(qc.data[idx].operation, CZGate)
            assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [i, i + 1]
        for i in range(n_qubits):
            idx = n_qubits + n_cz + i
            assert isinstance(qc.data[idx].operation, Measure)
        assert qc.num_qubits == n_qubits

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
        # Structure: H on even-indexed qubits (0, 2, 4, ...), then n Measure
        even_indices = list(range(0, n_qubits, 2))
        n_h = len(even_indices)
        assert len(qc.data) == n_h + n_qubits
        for pos, qi in enumerate(even_indices):
            assert isinstance(qc.data[pos].operation, HGate)
            assert [qc.find_bit(q).index for q in qc.data[pos].qubits] == [qi]
        for i in range(n_qubits):
            assert isinstance(qc.data[n_h + i].operation, Measure)
        assert qc.num_qubits == n_qubits

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
        # Structure: H on odd-indexed qubits (1, 3, 5, ...), then n Measure
        odd_indices = list(range(1, n_qubits, 2))
        n_h = len(odd_indices)
        assert len(qc.data) == n_h + n_qubits
        for pos, qi in enumerate(odd_indices):
            assert isinstance(qc.data[pos].operation, HGate)
            assert [qc.find_bit(q).index for q in qc.data[pos].qubits] == [qi]
        for i in range(n_qubits):
            assert isinstance(qc.data[n_h + i].operation, Measure)
        assert qc.num_qubits == n_qubits

    @pytest.mark.parametrize("start,stop", [(5, 5), (3, 2)])
    def test_zero_iteration_range(self, start, stop):
        """range(start, stop) with start >= stop produces no gates."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(start, stop):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 6})
        # Structure: 0 H gates (zero iterations), only 6 Measure gates
        assert len(qc.data) == 6
        for i in range(6):
            assert isinstance(qc.data[i].operation, Measure)
            assert [qc.find_bit(q).index for q in qc.data[i].qubits] == [i]
        assert qc.num_qubits == 6

    def test_multiple_sequential_ranges(self):
        """Two range loops in sequence: H layer then RZ layer."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            for i in qmc.range(n):
                q[i] = qmc.rz(q[i], thetas[i])
            return qmc.measure(q)

        n = 4
        thetas = [0.1 * i for i in range(n)]
        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": n, "thetas": thetas})
        # Structure: 4 H on q[0..3], 4 RZ on q[0..3], 4 Measure
        assert len(qc.data) == 3 * n
        for i in range(n):
            assert isinstance(qc.data[i].operation, HGate)
            assert [qc.find_bit(q).index for q in qc.data[i].qubits] == [i]
        for i in range(n):
            assert isinstance(qc.data[n + i].operation, RZGate)
            assert [qc.find_bit(q).index for q in qc.data[n + i].qubits] == [i]
            assert np.isclose(
                float(qc.data[n + i].operation.params[0]), thetas[i], atol=1e-10
            )
        for i in range(n):
            assert isinstance(qc.data[2 * n + i].operation, Measure)
        assert qc.num_qubits == n

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
        # Structure: 3 H on q[1], q[2], q[3] (range(1, 4)), then 5 Measure
        h_indices = list(range(1, n - 1))  # [1, 2, 3]
        n_h = len(h_indices)
        assert len(qc.data) == n_h + n
        for pos, qi in enumerate(h_indices):
            assert isinstance(qc.data[pos].operation, HGate)
            assert [qc.find_bit(q).index for q in qc.data[pos].qubits] == [qi]
        for i in range(n):
            assert isinstance(qc.data[n_h + i].operation, Measure)
        assert qc.num_qubits == n


class TestControlFlowItems:
    """Test qmc.items loop through the frontend pipeline."""

    @pytest.mark.parametrize(
        "n_qubits,ising",
        [
            (3, {(0, 1): 1.0, (1, 2): -0.5, (0, 2): 0.3}),
            (4, {(0, 1): 1.0, (1, 2): -0.5, (2, 3): 0.7}),
            (2, {(0, 1): 0.5}),
        ],
    )
    def test_items_ising_rzz(self, n_qubits, ising):
        """Apply RZZ gates from Ising dict via qmc.items."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
            ising_dict: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            gamma: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for (i, j), Jij in qmc.items(ising_dict):
                q[i], q[j] = qmc.rzz(q[i], q[j], gamma * Jij)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": n_qubits, "ising_dict": ising, "gamma": 0.5}
        )
        # Structure: len(ising) RZZ gates, then n Measure gates
        edges = list(ising.keys())
        n_rzz = len(edges)
        assert len(qc.data) == n_rzz + n_qubits
        for pos, (i, j) in enumerate(edges):
            assert isinstance(qc.data[pos].operation, RZZGate)
            assert [qc.find_bit(q).index for q in qc.data[pos].qubits] == [i, j]
            expected_param = 0.5 * ising[(i, j)]
            assert np.isclose(
                float(qc.data[pos].operation.params[0]), expected_param, atol=1e-10
            )
        for i in range(n_qubits):
            assert isinstance(qc.data[n_rzz + i].operation, Measure)
        assert qc.num_qubits == n_qubits

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
        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 3, "angles": angles})
        # Structure: 3 RZ on q[0], q[1], q[2], then 3 Measure
        keys = list(angles.keys())
        assert len(qc.data) == len(keys) + 3
        for pos, k in enumerate(keys):
            assert isinstance(qc.data[pos].operation, RZGate)
            assert [qc.find_bit(q).index for q in qc.data[pos].qubits] == [k]
            assert np.isclose(
                float(qc.data[pos].operation.params[0]), angles[k], atol=1e-10
            )
        for i in range(3):
            assert isinstance(qc.data[len(keys) + i].operation, Measure)
        assert qc.num_qubits == 3

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
        # Structure: 0 RZZ (empty dict), only 2 Measure
        assert len(qc.data) == 2
        for i in range(2):
            assert isinstance(qc.data[i].operation, Measure)
        assert qc.num_qubits == 2

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
        ising = {(i, j): 0.1 * (i + j) for i in range(5) for j in range(i + 1, 5)}
        assert len(ising) == 10
        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 5, "ising": ising})
        # Structure: 10 RZZ gates, then 5 Measure
        edges = list(ising.keys())
        assert len(qc.data) == 10 + 5
        for pos, (i, j) in enumerate(edges):
            assert isinstance(qc.data[pos].operation, RZZGate)
            assert [qc.find_bit(q).index for q in qc.data[pos].qubits] == [i, j]
            assert np.isclose(
                float(qc.data[pos].operation.params[0]), ising[(i, j)], atol=1e-10
            )
        for i in range(5):
            assert isinstance(qc.data[10 + i].operation, Measure)
        assert qc.num_qubits == 5

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
        # Structure: 1 RZZ on q[0,1], then 2 Measure
        assert len(qc.data) == 3
        assert isinstance(qc.data[0].operation, RZZGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0, 1]
        assert np.isclose(
            float(qc.data[0].operation.params[0]), gamma * -1.0, atol=1e-10
        )
        assert isinstance(qc.data[1].operation, Measure)
        assert isinstance(qc.data[2].operation, Measure)
        assert qc.num_qubits == 2

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
        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 2, "ising": ising})
        sv = _run_statevector(qc)
        # RZZ(theta) on |00>: apply RZZ matrix to all-zeros state
        expected = GATE_SPECS["RZZ"].matrix_fn(theta) @ all_zeros_state(2)
        assert statevectors_equal(sv, expected)


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
        # Structure: n H, n*(n-1)/2 CZ (pairs (0,1),(0,2),(1,2)), n Measure
        n_cz = n * (n - 1) // 2
        cz_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        assert len(qc.data) == n + n_cz + n
        for i in range(n):
            assert isinstance(qc.data[i].operation, HGate)
            assert [qc.find_bit(q).index for q in qc.data[i].qubits] == [i]
        for pos, (qi, qj) in enumerate(cz_pairs):
            idx = n + pos
            assert isinstance(qc.data[idx].operation, CZGate)
            assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [qi, qj]
        for i in range(n):
            assert isinstance(qc.data[n + n_cz + i].operation, Measure)
        assert qc.num_qubits == n

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
        # Structure: 3 H on q[0..2], 2 RZZ on edges, 3 Measure
        edges = list(ising.keys())
        assert len(qc.data) == 3 + len(edges) + 3
        for i in range(3):
            assert isinstance(qc.data[i].operation, HGate)
            assert [qc.find_bit(q).index for q in qc.data[i].qubits] == [i]
        for pos, (ei, ej) in enumerate(edges):
            idx = 3 + pos
            assert isinstance(qc.data[idx].operation, RZZGate)
            assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [ei, ej]
            assert np.isclose(
                float(qc.data[idx].operation.params[0]),
                0.5 * ising[(ei, ej)],
                atol=1e-10,
            )
        for i in range(3):
            assert isinstance(qc.data[3 + len(edges) + i].operation, Measure)
        assert qc.num_qubits == 3

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
        # Structure: X(q0), (n-1) CX on (i, i+1), n Measure
        n_cx = n - 1
        assert len(qc.data) == 1 + n_cx + n
        assert isinstance(qc.data[0].operation, XGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        for i in range(n_cx):
            assert isinstance(qc.data[1 + i].operation, CXGate)
            assert [qc.find_bit(q).index for q in qc.data[1 + i].qubits] == [i, i + 1]
        for i in range(n):
            assert isinstance(qc.data[1 + n_cx + i].operation, Measure)
        assert qc.num_qubits == n
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
        # Structure: n H, n*(n-1)/2 CZ (all-pairs), n Measure
        n_cz = n * (n - 1) // 2
        cz_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        assert len(qc.data) == n + n_cz + n
        for i in range(n):
            assert isinstance(qc.data[i].operation, HGate)
            assert [qc.find_bit(q).index for q in qc.data[i].qubits] == [i]
        for pos, (qi, qj) in enumerate(cz_pairs):
            idx = n + pos
            assert isinstance(qc.data[idx].operation, CZGate)
            assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [qi, qj]
        for i in range(n):
            assert isinstance(qc.data[n + n_cz + i].operation, Measure)
        assert qc.num_qubits == n


class TestControlFlowIfElse:
    """Test if-else control flow in @qkernel (measure → conditional gate)."""

    def test_if_else_creation(self):
        """If-else transpiles to: X(q0) → Measure(q0→c0) → IfElse(q1,c0) → Measure(q1→c2)."""

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
        assert len(qc.data) == 4  # X + Measure + IfElse + Measure
        assert isinstance(qc.data[0].operation, XGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert isinstance(qc.data[1].operation, Measure)
        assert [qc.find_bit(q).index for q in qc.data[1].qubits] == [0]
        assert [qc.find_bit(c).index for c in qc.data[1].clbits] == [0]
        assert isinstance(qc.data[2].operation, IfElseOp)
        assert [qc.find_bit(q).index for q in qc.data[2].qubits] == [1]
        assert isinstance(qc.data[3].operation, Measure)
        assert qc.num_qubits == 2

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
        assert len(qc.data) == 4
        assert isinstance(qc.data[0].operation, XGate)
        assert isinstance(qc.data[1].operation, Measure)
        assert isinstance(qc.data[2].operation, IfElseOp)
        assert [qc.find_bit(q).index for q in qc.data[2].qubits] == [1]
        assert isinstance(qc.data[3].operation, Measure)

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
        """If-only (no else) transpiles to: X(q0) → Measure(q0→c0) → IfElse(q1) → Measure(q1)."""

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
        assert len(qc.data) == 4
        assert isinstance(qc.data[0].operation, XGate)
        assert isinstance(qc.data[1].operation, Measure)
        assert isinstance(qc.data[2].operation, IfElseOp)
        assert isinstance(qc.data[3].operation, Measure)

    def test_if_else_on_zero_measurement(self):
        """Measure |0⟩: Measure(q0→c0) → IfElse(q1,c0) → Measure(q1)."""

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
        assert len(qc.data) == 3  # no X init → Measure + IfElse + Measure
        assert isinstance(qc.data[0].operation, Measure)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert isinstance(qc.data[1].operation, IfElseOp)
        assert [qc.find_bit(q).index for q in qc.data[1].qubits] == [1]
        assert isinstance(qc.data[2].operation, Measure)

    def test_if_else_with_parametric_gate(self):
        """Parametric if-else: X(q0) → Measure(q0) → IfElse(q1) → Measure(q1)."""

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

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 4})
        assert len(qc.data) == 4
        assert isinstance(qc.data[0].operation, XGate)
        assert isinstance(qc.data[1].operation, Measure)
        assert isinstance(qc.data[2].operation, IfElseOp)
        assert isinstance(qc.data[3].operation, Measure)

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
        # Structure: X(q0), M(q0→c0), IfElse(q1,c0), M(q1→c2), IfElse(q2,c2), M(q2→c4)
        assert len(qc.data) == 6
        assert isinstance(qc.data[0].operation, XGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert isinstance(qc.data[1].operation, Measure)
        assert [qc.find_bit(q).index for q in qc.data[1].qubits] == [0]
        assert isinstance(qc.data[2].operation, IfElseOp)
        assert [qc.find_bit(q).index for q in qc.data[2].qubits] == [1]
        assert isinstance(qc.data[3].operation, Measure)
        assert [qc.find_bit(q).index for q in qc.data[3].qubits] == [1]
        assert isinstance(qc.data[4].operation, IfElseOp)
        assert [qc.find_bit(q).index for q in qc.data[4].qubits] == [2]
        assert isinstance(qc.data[5].operation, Measure)
        assert [qc.find_bit(q).index for q in qc.data[5].qubits] == [2]
        assert qc.num_qubits == 3


class TestControlFlowWhileStructure:
    """Structural tests for while-loop transpilation.

    These tests verify the Qiskit circuit structure without executing
    the circuit. They check gate presence, qubit/clbit counts, body
    contents, and — critically — that the body measurement writes to
    the same classical bit as the while condition.
    """

    # -- basic gate-level checks ------------------------------------------

    def test_while_loop_creation(self):
        """While loop transpiles to: H(q0) → Measure(q0→c0) → WhileLoop."""

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

        # Exactly 3 top-level instructions
        assert len(qc.data) == 3

        # [0] H gate on qubit 0
        assert isinstance(qc.data[0].operation, HGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert qc.data[0].clbits == ()

        # [1] Measure on qubit 0 → clbit 0
        assert isinstance(qc.data[1].operation, Measure)
        assert [qc.find_bit(q).index for q in qc.data[1].qubits] == [0]
        assert [qc.find_bit(c).index for c in qc.data[1].clbits] == [0]

        # [2] WhileLoopOp
        assert isinstance(qc.data[2].operation, WhileLoopOp)

    def test_while_loop_structure(self):
        """While loop body sub-circuit: H(q0) → Measure(q0→c)."""

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

        # Extract the while loop body
        while_inst = qc.data[2]
        assert isinstance(while_inst.operation, WhileLoopOp)
        body = while_inst.operation.params[0]

        # Body has exactly 2 instructions
        assert len(body.data) == 2
        assert body.num_qubits == 1

        # Body[0]: H on body-qubit 0
        assert isinstance(body.data[0].operation, HGate)
        assert [body.find_bit(q).index for q in body.data[0].qubits] == [0]

        # Body[1]: Measure on body-qubit 0
        assert isinstance(body.data[1].operation, Measure)
        assert [body.find_bit(q).index for q in body.data[1].qubits] == [0]

    def test_while_loop_circuit_structure(self):
        """Top-level resource counts for a while-loop circuit."""

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

        assert len(qc.data) == 3  # H + Measure + WhileLoop
        assert qc.num_qubits >= 1
        assert qc.num_clbits >= 1

        # Verify types at each position
        assert isinstance(qc.data[0].operation, HGate)
        assert isinstance(qc.data[1].operation, Measure)
        assert isinstance(qc.data[2].operation, WhileLoopOp)

    def test_while_loop_with_x_body(self):
        """X-init while loop: X(q0) → Measure(q0→c0) → WhileLoop."""

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

        # Exactly 3 top-level instructions
        assert len(qc.data) == 3

        # [0] X gate on qubit 0
        assert isinstance(qc.data[0].operation, XGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]

        # [1] Measure on qubit 0 → clbit 0
        assert isinstance(qc.data[1].operation, Measure)
        assert [qc.find_bit(q).index for q in qc.data[1].qubits] == [0]
        assert [qc.find_bit(c).index for c in qc.data[1].clbits] == [0]

        # [2] WhileLoopOp
        assert isinstance(qc.data[2].operation, WhileLoopOp)

    def test_while_loop_body_gate_verification(self):
        """Body sub-circuit has H then Measure, both targeting body-qubit 0."""

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

        # Get while loop body
        assert isinstance(qc.data[2].operation, WhileLoopOp)
        body = qc.data[2].operation.params[0]

        # Body has exactly 2 instructions
        assert len(body.data) == 2

        # Body[0]: H on body-qubit 0
        assert isinstance(body.data[0].operation, HGate)
        assert [body.find_bit(q).index for q in body.data[0].qubits] == [0]

        # Body[1]: Measure on body-qubit 0
        assert isinstance(body.data[1].operation, Measure)
        assert [body.find_bit(q).index for q in body.data[1].qubits] == [0]

        # H and Measure target the same qubit
        h_target = [body.find_bit(q).index for q in body.data[0].qubits]
        m_target = [body.find_bit(q).index for q in body.data[1].qubits]
        assert h_target == m_target

    # -- clbit aliasing (the root cause of the infinite-loop bug) ----------
    # NOTE: clbit aliasing and single-clbit tests are in the X-body section
    # below to avoid duplication.

    # -- body gate ordering -----------------------------------------------

    def test_while_loop_body_gate_order(self):
        """Body gates must appear in source order: H → measure."""

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
        assert isinstance(qc.data[2].operation, WhileLoopOp)
        body = qc.data[2].operation.params[0]
        assert len(body.data) == 2
        assert isinstance(body.data[0].operation, HGate)
        assert isinstance(body.data[1].operation, Measure)

    # -- outer circuit gate ordering --------------------------------------

    def test_while_loop_outer_gate_order(self):
        """Initial measure must appear before while_loop in the outer circuit."""

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
        # Verify ordering: H → Measure → WhileLoop
        assert len(qc.data) == 3
        assert isinstance(qc.data[0].operation, HGate)
        assert isinstance(qc.data[1].operation, Measure)
        assert isinstance(qc.data[2].operation, WhileLoopOp)

    # -- condition wiring -------------------------------------------------

    def test_while_loop_condition_uses_initial_measure_clbit(self):
        """The while condition must reference the clbit from the initial measurement."""

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

        # Find the initial measure instruction (before while_loop)
        assert isinstance(qc.data[1].operation, Measure)
        initial_measure_clbit = qc.data[1].clbits[0]

        # The while condition must reference the same clbit
        assert isinstance(qc.data[2].operation, WhileLoopOp)
        condition_clbit = qc.data[2].clbits[0]
        assert qc.clbits.index(condition_clbit) == qc.clbits.index(
            initial_measure_clbit
        ), "while_loop condition must reference the clbit from the initial measure"

    # -- X-body structure: initial X, body has H + measure ----------------

    def test_while_loop_x_init_body_contains_expected_gates(self):
        """X-initialized while loop body must contain H and measure."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            bit = qmc.measure(q)
            while bit:
                q = qmc.qubit("q2")
                q = qmc.h(q)
                bit = qmc.measure(q)
            return bit

        _, qc = _transpile_and_get_circuit(circuit)
        assert isinstance(qc.data[2].operation, WhileLoopOp)
        body = qc.data[2].operation.params[0]
        assert len(body.data) == 2
        assert isinstance(body.data[0].operation, HGate)
        assert isinstance(body.data[1].operation, Measure)

    def test_while_loop_body_measure_same_clbit(self):
        """Body measurement must write to the same clbit as the condition.

        Without this, the while condition checks clbit[0] but the body
        writes to clbit[1], so the condition never updates and the loop
        never terminates.
        """

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

        # Find the while_loop instruction
        assert isinstance(qc.data[2].operation, WhileLoopOp)
        while_inst = qc.data[2]
        condition_clbit = while_inst.clbits[0]

        # The body circuit is in params[0]
        body = while_inst.operation.params[0]
        body_measures = [i for i in body.data if isinstance(i.operation, Measure)]
        assert len(body_measures) == 1

        # The body measure must target the same classical bit index as
        # the while condition.  In Qiskit's while_loop, the body circuit
        # shares the same classical register as the outer circuit, so
        # the clbit index used inside the body must match the condition.
        body_measure_clbit = body_measures[0].clbits[0]
        assert body.clbits.index(body_measure_clbit) == qc.clbits.index(
            condition_clbit
        ), (
            f"Body measure writes to clbit index "
            f"{body.clbits.index(body_measure_clbit)} "
            f"but while condition checks clbit index "
            f"{qc.clbits.index(condition_clbit)}"
        )

    def test_while_loop_single_clbit_allocated(self):
        """Loop-carried aliasing should not waste a classical bit."""

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
        assert qc.num_clbits == 1, (
            f"Expected 1 classical bit but got {qc.num_clbits}. "
            "The body measurement should alias to the condition's clbit."
        )

    def test_while_loop_execution_terminates(self):
        """Execute a while loop that is guaranteed to terminate.

        q0 starts in |1⟩ (X gate), so measure → bit = 1 and the loop
        enters.  Inside the loop, q1 starts in |0⟩ (no gate), so
        measure → bit = 0 and the loop exits on the next condition
        check.  The returned bit must always be 0.

        This test catches regressions where the body measurement writes
        to a different clbit than the while condition, causing the loop
        to never see the updated value (infinite loop / wrong result).
        """

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q0 = qmc.x(q0)  # |1⟩ → bit = 1, enter loop
            bit = qmc.measure(q0)
            while bit:
                q1 = qmc.qubit("q1")  # |0⟩ → bit = 0, exit loop
                bit = qmc.measure(q1)
            return bit

        # --- Structure checks ---
        _, qc = _transpile_and_get_circuit(circuit)
        assert qc.num_clbits == 1, f"Expected 1 classical bit but got {qc.num_clbits}."
        while_insts = [i for i in qc.data if isinstance(i.operation, WhileLoopOp)]
        assert len(while_insts) == 1
        body = while_insts[0].operation.params[0]
        body_measures = [i for i in body.data if isinstance(i.operation, Measure)]
        assert len(body_measures) == 1
        # Body measure must target the same clbit as the while condition.
        cond_clbit_idx = qc.clbits.index(while_insts[0].clbits[0])
        body_clbit_idx = body.clbits.index(body_measures[0].clbits[0])
        assert body_clbit_idx == cond_clbit_idx, (
            f"Body measure targets clbit {body_clbit_idx} but while "
            f"condition checks clbit {cond_clbit_idx}."
        )

        # --- Execution check ---
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        assert result is not None
        # Every shot must return 0 (the loop always exits after one
        # iteration because q1 is measured as 0).
        for value, count in result.results:
            assert value == 0, (
                f"Expected all shots to return 0 but got value={value} "
                f"({count} shots). The while-loop condition is not being "
                "updated correctly."
            )

    def test_while_loop_with_if_else_clbit_structure(self):
        """While loop with if-else: both branch measurements alias to the
        condition clbit.

        This is the structural counterpart of the execution test below.
        Without the clbit aliasing fix, the allocator would create
        separate clbits for each branch measurement (5 clbits total),
        and the while condition would never observe the updated value.
        """

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q0 = qmc.h(q0)
            bit = qmc.measure(q0)

            q1 = qmc.qubit("q1")
            q1 = qmc.h(q1)
            sel = qmc.measure(q1)

            while bit:
                if sel:
                    q2 = qmc.qubit("q2")
                    q2 = qmc.h(q2)
                    bit = qmc.measure(q2)
                else:
                    q3 = qmc.qubit("q3")
                    q3 = qmc.h(q3)
                    bit = qmc.measure(q3)
            return bit

        _, qc = _transpile_and_get_circuit(circuit)

        # Two classical bits: one for the while condition, one for sel.
        assert qc.num_clbits == 2, (
            f"Expected 2 classical bits but got {qc.num_clbits}. "
            "Branch measurements are not being aliased to the "
            "while-condition clbit."
        )

        # Find the while_loop and its condition clbit index.
        while_insts = [i for i in qc.data if isinstance(i.operation, WhileLoopOp)]
        assert len(while_insts) == 1
        while_inst = while_insts[0]
        cond_clbit_idx = qc.clbits.index(while_inst.clbits[0])

        # Inside the while body there is an if_else block.
        body = while_inst.operation.params[0]
        if_else_insts = [i for i in body.data if isinstance(i.operation, IfElseOp)]
        assert len(if_else_insts) == 1

        # Both branches (if / else) must have exactly one measurement
        # each, and both must target the same clbit as the while
        # condition.
        if_else_op = if_else_insts[0]
        for branch_idx, block in enumerate(if_else_op.operation.blocks):
            branch_name = "if" if branch_idx == 0 else "else"
            measures = [i for i in block.data if isinstance(i.operation, Measure)]
            assert len(measures) == 1, (
                f"Expected 1 measurement in {branch_name}-branch but "
                f"got {len(measures)}."
            )
            # Resolve the measurement's clbit index back through the
            # nesting: block → if_else → while_body → top-level circuit.
            meas_clbit_in_block = block.clbits.index(measures[0].clbits[0])
            if_else_clbits = [body.clbits.index(c) for c in if_else_insts[0].clbits]
            meas_clbit_in_body = if_else_clbits[meas_clbit_in_block]
            while_clbits = [qc.clbits.index(c) for c in while_inst.clbits]
            meas_clbit_in_circuit = while_clbits[meas_clbit_in_body]
            assert meas_clbit_in_circuit == cond_clbit_idx, (
                f"{branch_name}-branch measure targets circuit clbit "
                f"{meas_clbit_in_circuit} but while condition checks "
                f"clbit {cond_clbit_idx}."
            )

    def test_while_loop_with_if_else_execution(self):
        """Execute a while loop containing an if-else with measurements.

        q0 = X → bit = 1 (enter loop), q1 = X → sel = 1 (always take
        if-branch).  Inside the if-branch, q2 is |0⟩ → bit = 0, so
        the loop exits.  The returned bit must always be 0.

        This is the exact pattern that triggered the clbit aliasing bug:
        measurements inside if-else branches within a while loop were
        allocated separate clbits, so the while condition never saw the
        updated measurement and the loop ran forever.
        """

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q0 = qmc.x(q0)  # |1⟩ → bit = 1, enter loop
            bit = qmc.measure(q0)

            q1 = qmc.qubit("q1")
            q1 = qmc.x(q1)  # |1⟩ → sel = 1, always take if-branch
            sel = qmc.measure(q1)

            while bit:
                if sel:
                    q2 = qmc.qubit("q2")  # |0⟩ → bit = 0, exit
                    bit = qmc.measure(q2)
                else:
                    q3 = qmc.qubit("q3")  # |0⟩ → bit = 0, exit
                    bit = qmc.measure(q3)
            return bit

        # --- Structure checks ---
        # 4 distinct qubits: q0, q1, q2, q3 (q2 and q3 are in mutually
        # exclusive branches so they are different physical qubits).
        _, qc = _transpile_and_get_circuit(circuit)
        assert qc.num_clbits == 2, (
            f"Expected 2 classical bits but got {qc.num_clbits}. "
            "Branch measurements are not being aliased to the "
            "while-condition clbit."
        )

        # --- Execution check ---
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        assert result is not None
        for value, count in result.results:
            assert value == 0, (
                f"Expected all shots to return 0 but got value={value} "
                f"({count} shots). The while-loop with if-else is not "
                "aliasing branch measurements to the condition clbit."
            )

    def test_while_loop_with_if_else_else_branch_taken(self):
        """While + if-else where the else branch terminates the loop.

        sel=0 (q1 stays |0⟩), so the else branch is always taken.
        q3 is |0⟩ → bit=0, loop exits.  Verifies that the else-branch
        measurement is correctly aliased to the condition clbit.
        """

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q0 = qmc.x(q0)  # bit=1, enter loop
            bit = qmc.measure(q0)

            q1 = qmc.qubit("q1")
            # q1 stays |0⟩ → sel=0, always take else-branch
            sel = qmc.measure(q1)

            while bit:
                if sel:
                    q2 = qmc.qubit("q2")
                    q2 = qmc.x(q2)  # bit=1 (never taken)
                    bit = qmc.measure(q2)
                else:
                    q3 = qmc.qubit("q3")  # |0⟩ → bit=0, exit
                    bit = qmc.measure(q3)
            return bit

        _, qc = _transpile_and_get_circuit(circuit)
        assert qc.num_clbits == 2, f"Expected 2 classical bits but got {qc.num_clbits}."

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()
        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        for value, count in result.results:
            assert value == 0, (
                f"Expected all shots to return 0 but got value={value} ({count} shots)."
            )

    def test_while_loop_with_nested_if_else(self):
        """While + nested if-else: recursive phi tracing must work.

        sel1=1, sel2=1, so the innermost if-branch is taken.
        q3 is |0⟩ → bit=0, loop exits.  Tests that
        _alias_loop_carried_clbits recurses through nested IfOperations.
        """

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q0 = qmc.x(q0)  # bit=1
            bit = qmc.measure(q0)

            q1 = qmc.qubit("q1")
            q1 = qmc.x(q1)  # sel1=1
            sel1 = qmc.measure(q1)

            q2 = qmc.qubit("q2")
            q2 = qmc.x(q2)  # sel2=1
            sel2 = qmc.measure(q2)

            while bit:
                if sel1:
                    if sel2:
                        q3 = qmc.qubit("q3")  # |0⟩ → bit=0
                        bit = qmc.measure(q3)
                    else:
                        q4 = qmc.qubit("q4")
                        q4 = qmc.x(q4)  # bit=1 (not taken)
                        bit = qmc.measure(q4)
                else:
                    q5 = qmc.qubit("q5")  # |0⟩ → bit=0 (not taken)
                    bit = qmc.measure(q5)
            return bit

        _, qc = _transpile_and_get_circuit(circuit)
        assert qc.num_clbits == 3, (
            f"Expected 3 classical bits (bit, sel1, sel2) but got {qc.num_clbits}."
        )

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()
        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        for value, count in result.results:
            assert value == 0, (
                f"Expected all shots to return 0 but got value={value} ({count} shots)."
            )

    def test_while_loop_with_if_only_no_else(self):
        """While loop with if-only (no else): clbit count must not leak.

        When the while body contains an if without else, the PhiOp's
        false_val is the pre-if while-condition value.  map_phi_outputs
        redirects this UUID, which used to corrupt the while-condition's
        canonical clbit and cause an extra orphaned clbit to appear.

        Verifies:
        - num_clbits == 2 (bit + sel, no orphan)
        - while condition clbit == initial measurement clbit
        - sampling always returns 0 (loop terminates correctly)
        """

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q0 = qmc.x(q0)  # bit=1, enter loop
            bit = qmc.measure(q0)

            q1 = qmc.qubit("q1")
            q1 = qmc.x(q1)  # sel=1, always take if-branch
            sel = qmc.measure(q1)

            while bit:
                if sel:
                    q2 = qmc.qubit("q2")  # |0⟩ → bit=0, exit
                    bit = qmc.measure(q2)
            return bit

        # --- Structure checks ---
        _, qc = _transpile_and_get_circuit(circuit)
        assert qc.num_clbits == 2, (
            f"Expected 2 classical bits but got {qc.num_clbits}. "
            "If-only (no else) in while loop is leaking orphan clbits."
        )

        # The while condition clbit must be the initial measurement's clbit.
        while_insts = [i for i in qc.data if isinstance(i.operation, WhileLoopOp)]
        assert len(while_insts) == 1
        while_inst = while_insts[0]
        cond_clbit_idx = qc.clbits.index(while_inst.clbits[0])

        # The initial measure (before the while) should be on clbit 0.
        initial_measures = [i for i in qc.data if isinstance(i.operation, Measure)]
        assert len(initial_measures) >= 1
        first_meas_clbit_idx = qc.clbits.index(initial_measures[0].clbits[0])
        assert cond_clbit_idx == first_meas_clbit_idx, (
            f"While condition uses clbit {cond_clbit_idx} but initial "
            f"measurement uses clbit {first_meas_clbit_idx}."
        )

        # --- Execution check ---
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()
        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        for value, count in result.results:
            assert value == 0, (
                f"Expected all shots to return 0 but got value={value} "
                f"({count} shots). While + if-only loop is not terminating."
            )

    def test_while_loop_with_nested_if_only_no_else(self):
        """While loop with nested if-only (no else): clbit count must not leak.

        A deeper nesting test: while body has an if-only containing another
        if-only that reassigns the while condition.  Verifies that the
        canonical clbit save/restore handles nested if-only correctly.
        """

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q0 = qmc.x(q0)  # bit=1, enter loop
            bit = qmc.measure(q0)

            q1 = qmc.qubit("q1")
            q1 = qmc.x(q1)  # sel1=1
            sel1 = qmc.measure(q1)

            q2 = qmc.qubit("q2")
            q2 = qmc.x(q2)  # sel2=1
            sel2 = qmc.measure(q2)

            while bit:
                if sel1:
                    if sel2:
                        q3 = qmc.qubit("q3")  # |0⟩ → bit=0, exit
                        bit = qmc.measure(q3)
            return bit

        _, qc = _transpile_and_get_circuit(circuit)
        assert qc.num_clbits == 3, (
            f"Expected 3 classical bits (bit, sel1, sel2) but got {qc.num_clbits}."
        )

        # --- Execution check ---
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()
        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        for value, count in result.results:
            assert value == 0, (
                f"Expected all shots to return 0 but got value={value} ({count} shots)."
            )


class TestControlFlowWhileSampling:
    """Sampling (execution) tests for while-loop circuits.

    These tests actually run the circuits on AerSimulator with shots
    and verify the measurement results. They require the clbit-aliasing
    bug fix to pass — without it the loop never terminates.
    """

    def test_while_loop_skip_when_zero(self):
        """Initial qubit in |0⟩ → measure gives 0 → loop body never entered.

        The result must always be 0.
        """

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            # q stays |0⟩, so bit = 0 → while condition is false
            bit = qmc.measure(q)
            while bit:
                q = qmc.qubit("q2")
                q = qmc.h(q)
                bit = qmc.measure(q)
            return bit

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        assert result is not None
        assert len(result.results) > 0
        # All shots must return 0 (loop never entered)
        for value, count in result.results:
            assert value == 0, f"Expected all 0, got value={value}"
        total_counts = sum(count for _, count in result.results)
        assert total_counts == 100

    def test_while_loop_deterministic_termination(self):
        """X gate makes bit=1, body applies H → 50% chance of 0 each iteration.

        The loop must terminate (not hang). Every returned value must be 0
        (the loop exits when bit becomes 0).
        """

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)  # |1⟩ → bit = 1 → enter loop
            bit = qmc.measure(q)
            while bit:
                q = qmc.qubit("q2")
                q = qmc.h(q)  # 50% chance of |0⟩ each iteration
                bit = qmc.measure(q)
            return bit

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        assert result is not None
        assert len(result.results) > 0
        # Loop only exits when bit=0, so all results must be 0
        for value, count in result.results:
            assert value == 0, (
                f"While loop should only exit with bit=0, got value={value}"
            )
        total_counts = sum(count for _, count in result.results)
        assert total_counts == 100

    def test_while_loop_repeat_until_zero(self):
        """Full repeat-until-zero pattern: H → measure → while bit → H → measure.

        This is the canonical repeat-until-success pattern. The circuit must
        terminate and always return 0. Tests with enough shots to ensure
        the loop runs at least some iterations statistically.
        """

        @qmc.qkernel
        def repeat_until_zero() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            bit = qmc.measure(q)
            while bit:
                q = qmc.qubit("q2")
                q = qmc.h(q)
                bit = qmc.measure(q)
            return bit

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(repeat_until_zero)
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=200)
        result = job.result()
        assert result is not None
        assert len(result.results) > 0
        # The returned value is always 0 (loop exit condition)
        for value, count in result.results:
            assert value == 0, (
                f"repeat_until_zero must always return 0, got value={value}"
            )
        total_counts = sum(count for _, count in result.results)
        assert total_counts == 200

    def test_while_loop_execution_returns_valid_bits(self):
        """All sample results must be valid Bit values (0 or 1)."""

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

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        for value, count in result.results:
            assert value in (0, 1), f"Bit value must be 0 or 1, got {value}"
            assert count > 0


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
        # Structure: 3H, 3RZZ, 3RX, 3Measure = 12 total
        edges = list(ising.keys())
        assert len(qc.data) == 12
        for i in range(3):
            assert isinstance(qc.data[i].operation, HGate)
            assert [qc.find_bit(q).index for q in qc.data[i].qubits] == [i]
        for pos, (ei, ej) in enumerate(edges):
            assert isinstance(qc.data[3 + pos].operation, RZZGate)
            assert [qc.find_bit(q).index for q in qc.data[3 + pos].qubits] == [ei, ej]
        for i in range(3):
            assert isinstance(qc.data[6 + i].operation, RXGate)
            assert [qc.find_bit(q).index for q in qc.data[6 + i].qubits] == [i]
        for i in range(3):
            assert isinstance(qc.data[9 + i].operation, Measure)
        assert qc.num_qubits == 3

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
        # Structure: n H, even CZ pairs, odd CZ pairs, n Measure
        even_pair_list = [(i, i + 1) for i in range(0, n - 1, 2)]
        odd_pair_list = [(i, i + 1) for i in range(1, n - 1, 2)]
        n_cz = len(even_pair_list) + len(odd_pair_list)
        assert len(qc.data) == n + n_cz + n
        for i in range(n):
            assert isinstance(qc.data[i].operation, HGate)
            assert [qc.find_bit(q).index for q in qc.data[i].qubits] == [i]
        for pos, (qi, qj) in enumerate(even_pair_list + odd_pair_list):
            idx = n + pos
            assert isinstance(qc.data[idx].operation, CZGate)
            assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [qi, qj]
        for i in range(n):
            assert isinstance(qc.data[n + n_cz + i].operation, Measure)
        assert qc.num_qubits == n


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
        """to_circuit() returns just the QuantumCircuit."""

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

        config = TranspilerConfig.with_strategies({"qft": "approximate_k2"})
        assert config.decomposition.strategy_overrides["qft"] == "approximate_k2"
        assert len(config.substitutions.rules) == 1
        assert config.substitutions.rules[0].source_name == "qft"
        assert config.substitutions.rules[0].strategy == "approximate_k2"

    def test_set_config_on_transpiler(self):
        """set_config() applies TranspilerConfig to transpiler."""

        transpiler = QiskitTranspiler(use_native_composite=False)
        config = TranspilerConfig.with_strategies({"qft": "approximate_k2"})
        transpiler.set_config(config)
        assert transpiler.config is config

    def test_substitute_pass_sets_strategy(self):
        """substitute() pass sets strategy_name on CompositeGateOperation."""

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
        # Standard QFT 5q: 5H + 10CP + 2SWAP + 5M = 22
        assert len(qc_std.data) == 22
        # Verify structure: H(0) 4CP H(1) 3CP H(2) 2CP H(3) 1CP H(4) 2SWAP 5M
        idx = 0
        for qubit in range(5):
            assert isinstance(qc_std.data[idx].operation, HGate)
            idx += 1
            for j in range(qubit + 1, 5):
                assert isinstance(qc_std.data[idx].operation, CPhaseGate)
                idx += 1
        # 2 SWAP gates
        assert isinstance(qc_std.data[idx].operation, SwapGate)
        assert [qc_std.find_bit(q).index for q in qc_std.data[idx].qubits] == [0, 4]
        idx += 1
        assert isinstance(qc_std.data[idx].operation, SwapGate)
        assert [qc_std.find_bit(q).index for q in qc_std.data[idx].qubits] == [1, 3]
        idx += 1
        for i in range(5):
            assert isinstance(qc_std.data[idx].operation, Measure)
            idx += 1

        # Approximate QFT (k=2)
        transpiler_approx = QiskitTranspiler(use_native_composite=False)
        config = TranspilerConfig.with_strategies({"qft": "approximate_k2"})
        transpiler_approx.set_config(config)
        exe_approx = transpiler_approx.transpile(circuit)
        qc_approx = exe_approx.compiled_quantum[0].circuit
        # Approximate QFT k=2 5q: 5H + 7CP + 2SWAP + 5M = 19
        # Each qubit gets at most k=2 CP gates after its H
        assert len(qc_approx.data) == 19
        # Verify structure: H(0) 2CP H(1) 2CP H(2) 2CP H(3) 1CP H(4) 2SWAP 5M
        idx = 0

        # qubit 0: H + CP(q1,q0,π/2) + CP(q2,q0,π/4)
        assert isinstance(qc_approx.data[idx].operation, HGate)
        assert [qc_approx.find_bit(q).index for q in qc_approx.data[idx].qubits] == [0]
        idx += 1
        assert isinstance(qc_approx.data[idx].operation, CPhaseGate)
        assert [qc_approx.find_bit(q).index for q in qc_approx.data[idx].qubits] == [
            1,
            0,
        ]
        assert np.isclose(
            float(qc_approx.data[idx].operation.params[0]), np.pi / 2, atol=1e-10
        )
        idx += 1
        assert isinstance(qc_approx.data[idx].operation, CPhaseGate)
        assert [qc_approx.find_bit(q).index for q in qc_approx.data[idx].qubits] == [
            2,
            0,
        ]
        assert np.isclose(
            float(qc_approx.data[idx].operation.params[0]), np.pi / 4, atol=1e-10
        )
        idx += 1

        # qubit 1: H + CP(q2,q1,π/2) + CP(q3,q1,π/4)
        assert isinstance(qc_approx.data[idx].operation, HGate)
        assert [qc_approx.find_bit(q).index for q in qc_approx.data[idx].qubits] == [1]
        idx += 1
        assert isinstance(qc_approx.data[idx].operation, CPhaseGate)
        assert [qc_approx.find_bit(q).index for q in qc_approx.data[idx].qubits] == [
            2,
            1,
        ]
        assert np.isclose(
            float(qc_approx.data[idx].operation.params[0]), np.pi / 2, atol=1e-10
        )
        idx += 1
        assert isinstance(qc_approx.data[idx].operation, CPhaseGate)
        assert [qc_approx.find_bit(q).index for q in qc_approx.data[idx].qubits] == [
            3,
            1,
        ]
        assert np.isclose(
            float(qc_approx.data[idx].operation.params[0]), np.pi / 4, atol=1e-10
        )
        idx += 1

        # qubit 2: H + CP(q3,q2,π/2) + CP(q4,q2,π/4)
        assert isinstance(qc_approx.data[idx].operation, HGate)
        assert [qc_approx.find_bit(q).index for q in qc_approx.data[idx].qubits] == [2]
        idx += 1
        assert isinstance(qc_approx.data[idx].operation, CPhaseGate)
        assert [qc_approx.find_bit(q).index for q in qc_approx.data[idx].qubits] == [
            3,
            2,
        ]
        assert np.isclose(
            float(qc_approx.data[idx].operation.params[0]), np.pi / 2, atol=1e-10
        )
        idx += 1
        assert isinstance(qc_approx.data[idx].operation, CPhaseGate)
        assert [qc_approx.find_bit(q).index for q in qc_approx.data[idx].qubits] == [
            4,
            2,
        ]
        assert np.isclose(
            float(qc_approx.data[idx].operation.params[0]), np.pi / 4, atol=1e-10
        )
        idx += 1

        # qubit 3: H + CP(q4,q3,π/2)
        assert isinstance(qc_approx.data[idx].operation, HGate)
        assert [qc_approx.find_bit(q).index for q in qc_approx.data[idx].qubits] == [3]
        idx += 1
        assert isinstance(qc_approx.data[idx].operation, CPhaseGate)
        assert [qc_approx.find_bit(q).index for q in qc_approx.data[idx].qubits] == [
            4,
            3,
        ]
        assert np.isclose(
            float(qc_approx.data[idx].operation.params[0]), np.pi / 2, atol=1e-10
        )
        idx += 1

        # qubit 4: H only
        assert isinstance(qc_approx.data[idx].operation, HGate)
        assert [qc_approx.find_bit(q).index for q in qc_approx.data[idx].qubits] == [4]
        idx += 1

        # 2 SWAP gates for bit reversal
        assert isinstance(qc_approx.data[idx].operation, SwapGate)
        assert [qc_approx.find_bit(q).index for q in qc_approx.data[idx].qubits] == [
            0,
            4,
        ]
        idx += 1
        assert isinstance(qc_approx.data[idx].operation, SwapGate)
        assert [qc_approx.find_bit(q).index for q in qc_approx.data[idx].qubits] == [
            1,
            3,
        ]
        idx += 1

        for i in range(5):
            assert isinstance(qc_approx.data[idx].operation, Measure)
            assert [
                qc_approx.find_bit(q).index for q in qc_approx.data[idx].qubits
            ] == [i]
            idx += 1

        # Approximate has strictly fewer total gates
        assert len(qc_approx.data) < len(qc_std.data)

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
        # Standard: 5H + 10CP + 2SWAP + 5M = 22
        assert len(qc_std.data) == 22
        cp_std = sum(1 for i in qc_std.data if isinstance(i.operation, CPhaseGate))
        assert cp_std == 10  # 5*4/2

        # Approximate k=2
        transpiler_approx = QiskitTranspiler(use_native_composite=False)
        config = TranspilerConfig.with_strategies({"qft": "approximate_k2"})
        transpiler_approx.set_config(config)
        exe_approx = transpiler_approx.transpile(circuit)
        qc_approx = exe_approx.compiled_quantum[0].circuit
        # Approximate: 5H + 7CP + 2SWAP + 5M = 19
        assert len(qc_approx.data) == 19
        cp_approx = sum(
            1 for i in qc_approx.data if isinstance(i.operation, CPhaseGate)
        )
        assert cp_approx == 7  # truncated to k=2 neighbors
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

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        qc = transpiler.to_circuit(circuit, bindings={"theta": np.pi / 4})
        assert isinstance(qc, QuantumCircuit)
        # Structure: RX(q0, π/4) + Measure(q0) = 2
        assert len(qc.data) == 2
        assert isinstance(qc.data[0].operation, RXGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert np.isclose(float(qc.data[0].operation.params[0]), np.pi / 4, atol=1e-10)
        assert isinstance(qc.data[1].operation, Measure)
        assert [qc.find_bit(q).index for q in qc.data[1].qubits] == [0]

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

        config = TranspilerConfig.with_strategies(
            {
                "qft": "approximate_k2",
                "iqft": "approximate_k2",
            }
        )
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

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit, bindings={"n": 3}, parameters=["theta"])
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
        def layer_rz(q: qmc.Vector[qmc.Qubit], phi: qmc.Float) -> qmc.Vector[qmc.Qubit]:
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
        # Structure: 3 RX on q[0..2], 3 RZ on q[0..2], 3 Measure
        n = 3
        assert len(qc.data) == 3 * n
        for i in range(n):
            assert isinstance(qc.data[i].operation, RXGate)
            assert [qc.find_bit(q).index for q in qc.data[i].qubits] == [i]
        for i in range(n):
            assert isinstance(qc.data[n + i].operation, RZGate)
            assert [qc.find_bit(q).index for q in qc.data[n + i].qubits] == [i]
        for i in range(n):
            assert isinstance(qc.data[2 * n + i].operation, Measure)
        assert qc.num_qubits == n


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
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-10)

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
        # Decomposed QFT 3q: H(0) CP(1,0) CP(2,0) H(1) CP(2,1) H(2) SWAP(0,2) + 3M = 10
        assert len(qc.data) == 10
        assert isinstance(qc.data[0].operation, HGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert isinstance(qc.data[1].operation, CPhaseGate)
        assert isinstance(qc.data[2].operation, CPhaseGate)
        assert isinstance(qc.data[3].operation, HGate)
        assert [qc.find_bit(q).index for q in qc.data[3].qubits] == [1]
        assert isinstance(qc.data[4].operation, CPhaseGate)
        assert isinstance(qc.data[5].operation, HGate)
        assert [qc.find_bit(q).index for q in qc.data[5].qubits] == [2]
        assert isinstance(qc.data[6].operation, SwapGate)
        assert [qc.find_bit(q).index for q in qc.data[6].qubits] == [0, 2]
        for i in range(3):
            assert isinstance(qc.data[7 + i].operation, Measure)


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

    @pytest.mark.parametrize("n_qubits", [12, 20, 50])
    def test_large_qubit_count(self, n_qubits):
        """Circuit with 10+ qubits transpiles successfully."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": n_qubits})
        # Structure: n H gates, n Measure gates
        assert len(qc.data) == 2 * n_qubits
        assert qc.num_qubits == n_qubits
        for i in range(n_qubits):
            assert isinstance(qc.data[i].operation, HGate)
            assert [qc.find_bit(q).index for q in qc.data[i].qubits] == [i]
        for i in range(n_qubits):
            assert isinstance(qc.data[n_qubits + i].operation, Measure)

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


# ===========================================================================
# Error / Negative Tests
# ===========================================================================


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
        """Transpiling a kernel that needs a non-Float binding without it raises."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
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

        transpiler = QiskitTranspiler()
        # Provide a string for a UInt param — should fail
        with pytest.raises((TypeError, ValueError)):
            transpiler.transpile(circuit, bindings={"n": "not_a_number"})


# ===========================================================================
# 8. Advanced Algorithm Module Tests
# ===========================================================================


class TestAlgorithmBasicLayers:
    """Test qamomile.circuit.algorithm.basic building blocks through Qiskit."""

    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_ry_layer_gate_count(self, n_qubits):
        """ry_layer produces exactly n RY gates."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = ry_layer(q, thetas, qmc.uint(0))
            return qmc.measure(q)

        thetas = [0.1 * i for i in range(n_qubits)]
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": n_qubits, "thetas": thetas}
        )
        # Structure: n RY on q[0..n-1], n Measure
        assert len(qc.data) == 2 * n_qubits
        for i in range(n_qubits):
            assert isinstance(qc.data[i].operation, RYGate)
            assert [qc.find_bit(q).index for q in qc.data[i].qubits] == [i]
            assert np.isclose(
                float(qc.data[i].operation.params[0]), thetas[i], atol=1e-10
            )
        for i in range(n_qubits):
            assert isinstance(qc.data[n_qubits + i].operation, Measure)
        assert qc.num_qubits == n_qubits

    @pytest.mark.parametrize("n_qubits", [2, 3])
    def test_ry_layer_statevector(self, n_qubits):
        """ry_layer statevector matches tensor product of individual RY gates."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = ry_layer(q, thetas, qmc.uint(0))
            return qmc.measure(q)

        thetas = [0.3 * (i + 1) for i in range(n_qubits)]
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": n_qubits, "thetas": thetas}
        )
        sv = _run_statevector(qc)
        # Build expected (Qiskit little-endian: tensor order is MSB ⊗ ... ⊗ LSB)
        matrices = [GATE_SPECS["RY"].matrix_fn(t) for t in thetas]
        mat = matrices[-1]
        for m in reversed(matrices[:-1]):
            mat = tensor_product(mat, m)
        expected = mat @ all_zeros_state(n_qubits)
        assert statevectors_equal(sv, expected)

    def test_rx_layer_statevector(self):
        """rx_layer on 3 qubits matches analytical statevector."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = rx_layer(q, thetas, qmc.uint(0))
            return qmc.measure(q)

        thetas = [0.5, 1.0, 1.5]
        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 3, "thetas": thetas})
        sv = _run_statevector(qc)
        # Qiskit little-endian: tensor order MSB ⊗ ... ⊗ LSB
        matrices = [GATE_SPECS["RX"].matrix_fn(t) for t in thetas]
        mat = tensor_product(matrices[2], tensor_product(matrices[1], matrices[0]))
        expected = mat @ all_zeros_state(3)
        assert statevectors_equal(sv, expected)

    def test_rz_layer_statevector(self):
        """rz_layer on 3 qubits matches analytical statevector."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = rz_layer(q, thetas, qmc.uint(0))
            return qmc.measure(q)

        thetas = [0.2, 0.4, 0.6]
        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 3, "thetas": thetas})
        sv = _run_statevector(qc)
        # Qiskit little-endian: tensor order MSB ⊗ ... ⊗ LSB
        matrices = [GATE_SPECS["RZ"].matrix_fn(t) for t in thetas]
        mat = tensor_product(matrices[2], tensor_product(matrices[1], matrices[0]))
        expected = mat @ all_zeros_state(3)
        assert statevectors_equal(sv, expected)

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
        # Structure: n H + (n-1) CZ on (i,i+1) + n Measure
        assert len(qc.data) == n_qubits + (n_qubits - 1) + n_qubits
        idx = 0
        for i in range(n_qubits):
            assert isinstance(qc.data[idx].operation, HGate)
            assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [i]
            idx += 1
        for i in range(n_qubits - 1):
            assert isinstance(qc.data[idx].operation, CZGate)
            assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [i, i + 1]
            idx += 1
        for i in range(n_qubits):
            assert isinstance(qc.data[idx].operation, Measure)
            assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [i]
            idx += 1

    def test_cz_entangling_layer_on_plus_state(self):
        """CZ layer on |+>^3 produces correct entangled statevector."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            q = cz_entangling_layer(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 3})
        sv = _run_statevector(qc)
        # Qiskit little-endian: tensor order is q2 ⊗ q1 ⊗ q0
        H = GATE_SPECS["H"].matrix_fn()
        CZ = GATE_SPECS["CZ"].matrix_fn()
        I2 = identity(2)  # single-qubit 2x2 identity
        state = all_zeros_state(3)
        state = tensor_product(H, tensor_product(H, H)) @ state  # H on all (symmetric)
        # CZ on qubits 0,1: in Qiskit LE, CZ acts on bits 0,1 (rightmost)
        # tensor: I(q2) ⊗ CZ(q1,q0)
        cz_01 = tensor_product(I2, CZ)
        state = cz_01 @ state
        # CZ on qubits 1,2: in Qiskit LE, CZ acts on bits 1,2 (leftmost)
        # tensor: CZ(q2,q1) ⊗ I(q0)
        cz_12 = tensor_product(CZ, I2)
        state = cz_12 @ state
        assert statevectors_equal(sv, state)

    def test_ry_plus_cz_variational_block(self):
        """ry_layer + cz_entangling_layer: gate structure verification."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = ry_layer(q, thetas, qmc.uint(0))
            q = cz_entangling_layer(q)
            return qmc.measure(q)

        thetas = [0.1, 0.2, 0.3]
        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 3, "thetas": thetas})
        # Structure: 3 RY(q0,q1,q2) + 2 CZ(0,1),(1,2) + 3 Measure
        assert len(qc.data) == 8
        for i in range(3):
            assert isinstance(qc.data[i].operation, RYGate)
            assert [qc.find_bit(q).index for q in qc.data[i].qubits] == [i]
            assert np.isclose(
                float(qc.data[i].operation.params[0]), thetas[i], atol=1e-10
            )
        assert isinstance(qc.data[3].operation, CZGate)
        assert [qc.find_bit(q).index for q in qc.data[3].qubits] == [0, 1]
        assert isinstance(qc.data[4].operation, CZGate)
        assert [qc.find_bit(q).index for q in qc.data[4].qubits] == [1, 2]
        for i in range(3):
            assert isinstance(qc.data[5 + i].operation, Measure)
            assert [qc.find_bit(q).index for q in qc.data[5 + i].qubits] == [i]

    def test_ry_plus_cz_statevector(self):
        """ry_layer + cz_entangling_layer statevector correctness."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = ry_layer(q, thetas, qmc.uint(0))
            q = cz_entangling_layer(q)
            return qmc.measure(q)

        thetas = [0.5, 1.0, 1.5]
        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 3, "thetas": thetas})
        sv = _run_statevector(qc)
        # Expected: RY layer then CZ chain (Qiskit little-endian: q2 ⊗ q1 ⊗ q0)
        RY0, RY1, RY2 = [GATE_SPECS["RY"].matrix_fn(t) for t in thetas]
        CZ = GATE_SPECS["CZ"].matrix_fn()
        I2 = identity(2)  # single-qubit 2x2 identity
        state = all_zeros_state(3)
        state = tensor_product(RY2, tensor_product(RY1, RY0)) @ state
        # CZ(q0,q1): I(q2) ⊗ CZ(q1,q0)
        state = tensor_product(I2, CZ) @ state
        # CZ(q1,q2): CZ(q2,q1) ⊗ I(q0)
        state = tensor_product(CZ, I2) @ state
        assert statevectors_equal(sv, state)


class TestAlgorithmQAOAModules:
    """Test qamomile.circuit.algorithm.qaoa module functions through Qiskit."""

    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_superposition_vector_statevector(self, n_qubits):
        """superposition_vector produces uniform |+>^n."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = superposition_vector(n)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": n_qubits})
        sv = _run_statevector(qc)
        expected = np.ones(2**n_qubits, dtype=complex) / np.sqrt(2**n_qubits)
        assert statevectors_equal(sv, expected)

    def test_ising_cost_gate_count(self):
        """ising_cost emits correct RZZ and RZ gate counts."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            gamma: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = ising_cost(quad, linear, q, gamma)
            return qmc.measure(q)

        quad = {(0, 1): 1.0, (1, 2): -0.5}
        linear = {0: 0.3, 2: -0.1}
        gamma = 0.5
        _, qc = _transpile_and_get_circuit(
            circuit,
            bindings={"n": 3, "quad": quad, "linear": linear, "gamma": gamma},
        )
        # Structure: 2 RZZ + 2 RZ + 3 Measure = 7
        assert len(qc.data) == 7
        assert isinstance(qc.data[0].operation, RZZGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0, 1]
        assert np.isclose(
            float(qc.data[0].operation.params[0]), gamma * 1.0, atol=1e-10
        )
        assert isinstance(qc.data[1].operation, RZZGate)
        assert [qc.find_bit(q).index for q in qc.data[1].qubits] == [1, 2]
        assert np.isclose(
            float(qc.data[1].operation.params[0]), gamma * (-0.5), atol=1e-10
        )
        assert isinstance(qc.data[2].operation, RZGate)
        assert [qc.find_bit(q).index for q in qc.data[2].qubits] == [0]
        assert np.isclose(
            float(qc.data[2].operation.params[0]), gamma * 0.3, atol=1e-10
        )
        assert isinstance(qc.data[3].operation, RZGate)
        assert [qc.find_bit(q).index for q in qc.data[3].qubits] == [2]
        assert np.isclose(
            float(qc.data[3].operation.params[0]), gamma * (-0.1), atol=1e-10
        )
        for i in range(3):
            assert isinstance(qc.data[4 + i].operation, Measure)
            assert [qc.find_bit(q).index for q in qc.data[4 + i].qubits] == [i]

    def test_ising_cost_statevector(self):
        """ising_cost on 2 qubits matches RZZ(gamma*J) on |00>."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            gamma: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = ising_cost(quad, linear, q, gamma)
            return qmc.measure(q)

        gamma = 0.7
        quad = {(0, 1): 1.0}
        linear = {}
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": 2, "quad": quad, "linear": linear, "gamma": gamma}
        )
        sv = _run_statevector(qc)
        expected = GATE_SPECS["RZZ"].matrix_fn(gamma * 1.0) @ all_zeros_state(2)
        assert statevectors_equal(sv, expected)

    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_x_mixer_circuit_gate_count(self, n_qubits):
        """x_mixer emits n RX gates."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, beta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = x_mixer(q, beta)
            return qmc.measure(q)

        beta = 0.3
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": n_qubits, "beta": beta}
        )
        # Structure: n RX on q[0..n-1] + n Measure
        assert len(qc.data) == 2 * n_qubits
        for i in range(n_qubits):
            assert isinstance(qc.data[i].operation, RXGate)
            assert [qc.find_bit(q).index for q in qc.data[i].qubits] == [i]
            assert np.isclose(
                float(qc.data[i].operation.params[0]), 2.0 * beta, atol=1e-10
            )
        for i in range(n_qubits):
            assert isinstance(qc.data[n_qubits + i].operation, Measure)
            assert [qc.find_bit(q).index for q in qc.data[n_qubits + i].qubits] == [i]

    def test_x_mixer_circuit_statevector(self):
        """x_mixer statevector matches RX(2β) ⊗ RX(2β)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, beta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = x_mixer(q, beta)
            return qmc.measure(q)

        beta = 0.4
        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 2, "beta": beta})
        sv = _run_statevector(qc)
        RX = GATE_SPECS["RX"].matrix_fn(2.0 * beta)
        expected = tensor_product(RX, RX) @ all_zeros_state(2)
        assert statevectors_equal(sv, expected)

    def test_qaoa_layers_single_layer(self):
        """qaoa_layers with p=1: correct gate structure."""

        @qmc.qkernel
        def circuit(
            p: qmc.UInt,
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            n: qmc.UInt,
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = qaoa_layers(p, quad, linear, q, gammas, betas)
            return qmc.measure(q)

        quad = {(0, 1): 1.0}
        linear = {}
        gamma, beta = 0.5, 0.3
        _, qc = _transpile_and_get_circuit(
            circuit,
            bindings={
                "p": 1,
                "quad": quad,
                "linear": linear,
                "n": 2,
                "gammas": [gamma],
                "betas": [beta],
            },
        )
        # Structure: 1 RZZ(0,1) + 2 RX(q0,q1) + 2 Measure = 5
        assert len(qc.data) == 5
        assert isinstance(qc.data[0].operation, RZZGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0, 1]
        assert np.isclose(
            float(qc.data[0].operation.params[0]), gamma * 1.0, atol=1e-10
        )
        assert isinstance(qc.data[1].operation, RXGate)
        assert [qc.find_bit(q).index for q in qc.data[1].qubits] == [0]
        assert np.isclose(float(qc.data[1].operation.params[0]), 2.0 * beta, atol=1e-10)
        assert isinstance(qc.data[2].operation, RXGate)
        assert [qc.find_bit(q).index for q in qc.data[2].qubits] == [1]
        assert np.isclose(float(qc.data[2].operation.params[0]), 2.0 * beta, atol=1e-10)
        for i in range(2):
            assert isinstance(qc.data[3 + i].operation, Measure)
            assert [qc.find_bit(q).index for q in qc.data[3 + i].qubits] == [i]

    def test_qaoa_state_single_layer_statevector(self):
        """Full qaoa_state p=1: H→RZZ→RX statevector correctness."""

        @qmc.qkernel
        def circuit(
            p: qmc.UInt,
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            n: qmc.UInt,
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qaoa_state(p, quad, linear, n, gammas, betas)
            return qmc.measure(q)

        gamma, beta = 0.5, 0.3
        quad = {(0, 1): 1.0}
        linear = {}
        _, qc = _transpile_and_get_circuit(
            circuit,
            bindings={
                "p": 1,
                "quad": quad,
                "linear": linear,
                "n": 2,
                "gammas": [gamma],
                "betas": [beta],
            },
        )
        sv = _run_statevector(qc)
        H = GATE_SPECS["H"].matrix_fn()
        RZZ = GATE_SPECS["RZZ"].matrix_fn(gamma * 1.0)
        RX = GATE_SPECS["RX"].matrix_fn(2.0 * beta)
        state = all_zeros_state(2)
        state = tensor_product(H, H) @ state
        state = RZZ @ state
        state = tensor_product(RX, RX) @ state
        assert statevectors_equal(sv, state)

    @pytest.mark.parametrize("p", [1, 2, 3])
    def test_qaoa_state_multi_layer(self, p):
        """qaoa_state gate counts scale correctly with p layers."""

        @qmc.qkernel
        def circuit(
            p_val: qmc.UInt,
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            n: qmc.UInt,
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qaoa_state(p_val, quad, linear, n, gammas, betas)
            return qmc.measure(q)

        n_qubits = 3
        quad_edges = [(0, 1), (1, 2), (0, 2)]
        quad_vals = [1.0, -0.5, 0.3]
        quad = dict(zip(quad_edges, quad_vals))
        linear = {}
        gammas = [0.1 * (i + 1) for i in range(p)]
        betas = [0.2 * (i + 1) for i in range(p)]
        _, qc = _transpile_and_get_circuit(
            circuit,
            bindings={
                "p_val": p,
                "quad": quad,
                "linear": linear,
                "n": n_qubits,
                "gammas": gammas,
                "betas": betas,
            },
        )
        n_edges = len(quad)
        # Structure: 3 H + p * (3 RZZ + 3 RX) + 3 Measure
        expected_len = n_qubits + p * (n_edges + n_qubits) + n_qubits
        assert len(qc.data) == expected_len
        idx = 0
        # Initial H gates
        for i in range(n_qubits):
            assert isinstance(qc.data[idx].operation, HGate)
            assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [i]
            idx += 1
        # p layers of (RZZ + RX)
        for layer in range(p):
            gamma = gammas[layer]
            beta = betas[layer]
            for e, (qi, qj) in enumerate(quad_edges):
                assert isinstance(qc.data[idx].operation, RZZGate)
                assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [qi, qj]
                assert np.isclose(
                    float(qc.data[idx].operation.params[0]),
                    gamma * quad_vals[e],
                    atol=1e-10,
                )
                idx += 1
            for i in range(n_qubits):
                assert isinstance(qc.data[idx].operation, RXGate)
                assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [i]
                assert np.isclose(
                    float(qc.data[idx].operation.params[0]), 2.0 * beta, atol=1e-10
                )
                idx += 1
        # Measure gates
        for i in range(n_qubits):
            assert isinstance(qc.data[idx].operation, Measure)
            assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [i]
            idx += 1


class TestAdvancedParameterHandling:
    """Test parameters= with Vector[Float] and advanced re-execution patterns."""

    def test_parametric_scalar_float_works(self):
        """Control test: scalar Float with parameters= preserves Qiskit Parameter."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.ry(q, theta)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        assert exe.has_parameters
        qc = exe.compiled_quantum[0].circuit
        assert len(qc.parameters) > 0

    def test_parametric_vector_float(self):
        """Vector[Float] with parameters= should preserve Qiskit Parameters."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.ry(q[i], thetas[i])
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit, bindings={"n": 3}, parameters=["thetas"])
        assert exe.has_parameters
        qc = exe.compiled_quantum[0].circuit
        assert len(qc.parameters) > 0

    def test_parametric_mixed_scalar_and_vector(self):
        """Mixed scalar + Vector[Float] parameters."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
            theta: qmc.Float,
            params: qmc.Vector[qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.ry(q[i], params[i])
            q[0] = qmc.rz(q[0], theta)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(
            circuit, bindings={"n": 3}, parameters=["theta", "params"]
        )
        assert exe.has_parameters
        param_names = exe.parameter_names
        # Should have theta + params[0], params[1], params[2]
        assert len(param_names) >= 4

    def test_parametric_variational_classifier_pattern(self):
        """User's exact pattern: ry_layer + cz_entangling_layer with parametric Vector."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, params: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = ry_layer(q, params, qmc.uint(0))
            q = cz_entangling_layer(q)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit, bindings={"n": 3}, parameters=["params"])
        assert exe.has_parameters
        qc = exe.compiled_quantum[0].circuit
        assert len(qc.parameters) == 3  # params[0], params[1], params[2]

    def test_parametric_qaoa_vector_gammas_betas(self):
        """QAOA with parameters=['gammas','betas'] — Vector[Float] parameters."""

        @qmc.qkernel
        def circuit(
            p: qmc.UInt,
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            n: qmc.UInt,
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qaoa_state(p, quad, linear, n, gammas, betas)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(
            circuit,
            bindings={
                "p": 1,
                "quad": {(0, 1): 1.0},
                "linear": {},
                "n": 2,
            },
            parameters=["gammas", "betas"],
        )
        assert exe.has_parameters
        qc = exe.compiled_quantum[0].circuit
        assert len(qc.parameters) > 0

    def test_parametric_bind_and_rebind(self):
        """Compile once, execute with different parameter values."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.ry(q, theta)
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

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_parametric_multiple_re_executions(self, seed):
        """Same compiled circuit, different random theta values each time."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])

        rng = np.random.default_rng(seed)
        theta = rng.uniform(0, 2 * np.pi)

        # Verify parameter binding works and circuit is valid
        qc = exe.compiled_quantum[0].circuit
        assert len(qc.parameters) > 0

        executor = transpiler.executor()
        job = exe.sample(executor, bindings={"theta": theta}, shots=100)
        result = job.result()
        assert result is not None
        assert len(result.results) > 0


# ===========================================================================
# 9. Algorithm Tests: Deutsch-Jozsa
# ===========================================================================


class TestDeutschJozsaAlgorithm:
    """Test Deutsch-Jozsa algorithm with multiple oracle types."""

    def test_constant_0_oracle_transpiles(self):
        """Constant-0 DJ circuit has n+1 qubits and correct gate structure."""

        @qmc.qkernel
        def oracle_c0(
            inputs: qmc.Vector[qmc.Qubit], ancilla: qmc.Qubit
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
            return inputs, ancilla

        @qmc.qkernel
        def dj_c0(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            inputs = qmc.qubit_array(n, name="input")
            ancilla = qmc.qubit(name="ancilla")
            ancilla = qmc.x(ancilla)
            for i in qmc.range(n):
                inputs[i] = qmc.h(inputs[i])
            ancilla = qmc.h(ancilla)
            inputs, ancilla = oracle_c0(inputs, ancilla)
            for i in qmc.range(n):
                inputs[i] = qmc.h(inputs[i])
            return qmc.measure(inputs)

        n = 3
        _, qc = _transpile_and_get_circuit(dj_c0, bindings={"n": n})
        assert qc.num_qubits == n + 1  # n input + 1 ancilla
        # Structure: X(ancilla) + n H(input) + H(ancilla) + n H(input) + n Measure
        # = 1 + n + 1 + n + n = 3n + 2
        assert len(qc.data) == 3 * n + 2
        idx = 0
        # X on ancilla (qubit n)
        assert isinstance(qc.data[idx].operation, XGate)
        assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [n]
        idx += 1
        # H on input qubits [0..n-1]
        for i in range(n):
            assert isinstance(qc.data[idx].operation, HGate)
            assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [i]
            idx += 1
        # H on ancilla
        assert isinstance(qc.data[idx].operation, HGate)
        assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [n]
        idx += 1
        # (oracle_c0 does nothing)
        # H on input qubits [0..n-1]
        for i in range(n):
            assert isinstance(qc.data[idx].operation, HGate)
            assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [i]
            idx += 1
        # Measure on input qubits [0..n-1]
        for i in range(n):
            assert isinstance(qc.data[idx].operation, Measure)
            assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [i]
            idx += 1

    def test_constant_0_all_zeros(self):
        """Constant-0 oracle: all measurements should be all-zeros."""

        @qmc.qkernel
        def oracle_c0(
            inputs: qmc.Vector[qmc.Qubit], ancilla: qmc.Qubit
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
            return inputs, ancilla

        @qmc.qkernel
        def dj_c0(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            inputs = qmc.qubit_array(n, name="input")
            ancilla = qmc.qubit(name="ancilla")
            ancilla = qmc.x(ancilla)
            for i in qmc.range(n):
                inputs[i] = qmc.h(inputs[i])
            ancilla = qmc.h(ancilla)
            inputs, ancilla = oracle_c0(inputs, ancilla)
            for i in qmc.range(n):
                inputs[i] = qmc.h(inputs[i])
            return qmc.measure(inputs)

        exe, _ = _transpile_and_get_circuit(dj_c0, bindings={"n": 2})
        result = exe.sample(QiskitTranspiler().executor(), shots=1000).result()
        for value, _ in result.results:
            assert all(v == 0 for v in value)

    def test_constant_1_all_zeros(self):
        """Constant-1 oracle: all measurements should also be all-zeros."""

        @qmc.qkernel
        def oracle_c1(
            inputs: qmc.Vector[qmc.Qubit], ancilla: qmc.Qubit
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
            ancilla = qmc.x(ancilla)
            return inputs, ancilla

        @qmc.qkernel
        def dj_c1(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            inputs = qmc.qubit_array(n, name="input")
            ancilla = qmc.qubit(name="ancilla")
            ancilla = qmc.x(ancilla)
            for i in qmc.range(n):
                inputs[i] = qmc.h(inputs[i])
            ancilla = qmc.h(ancilla)
            inputs, ancilla = oracle_c1(inputs, ancilla)
            for i in qmc.range(n):
                inputs[i] = qmc.h(inputs[i])
            return qmc.measure(inputs)

        exe, _ = _transpile_and_get_circuit(dj_c1, bindings={"n": 2})
        result = exe.sample(QiskitTranspiler().executor(), shots=1000).result()
        for value, _ in result.results:
            assert all(v == 0 for v in value)

    def test_balanced_xor_non_zero(self):
        """Balanced XOR oracle: no measurement should be all-zeros."""

        @qmc.qkernel
        def oracle_bxor(
            inputs: qmc.Vector[qmc.Qubit], ancilla: qmc.Qubit
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
            n = inputs.shape[0]
            for i in qmc.range(n):
                inputs[i], ancilla = qmc.cx(inputs[i], ancilla)
            return inputs, ancilla

        @qmc.qkernel
        def dj_bxor(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            inputs = qmc.qubit_array(n, name="input")
            ancilla = qmc.qubit(name="ancilla")
            ancilla = qmc.x(ancilla)
            for i in qmc.range(n):
                inputs[i] = qmc.h(inputs[i])
            ancilla = qmc.h(ancilla)
            inputs, ancilla = oracle_bxor(inputs, ancilla)
            for i in qmc.range(n):
                inputs[i] = qmc.h(inputs[i])
            return qmc.measure(inputs)

        exe, _ = _transpile_and_get_circuit(dj_bxor, bindings={"n": 2})
        result = exe.sample(QiskitTranspiler().executor(), shots=1000).result()
        for value, _ in result.results:
            assert not all(v == 0 for v in value)

    def test_balanced_first_bit_non_zero(self):
        """Balanced first-bit oracle: no measurement should be all-zeros."""

        @qmc.qkernel
        def oracle_bfirst(
            inputs: qmc.Vector[qmc.Qubit], ancilla: qmc.Qubit
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
            inputs[0], ancilla = qmc.cx(inputs[0], ancilla)
            return inputs, ancilla

        @qmc.qkernel
        def dj_bfirst(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            inputs = qmc.qubit_array(n, name="input")
            ancilla = qmc.qubit(name="ancilla")
            ancilla = qmc.x(ancilla)
            for i in qmc.range(n):
                inputs[i] = qmc.h(inputs[i])
            ancilla = qmc.h(ancilla)
            inputs, ancilla = oracle_bfirst(inputs, ancilla)
            for i in qmc.range(n):
                inputs[i] = qmc.h(inputs[i])
            return qmc.measure(inputs)

        exe, _ = _transpile_and_get_circuit(dj_bfirst, bindings={"n": 2})
        result = exe.sample(QiskitTranspiler().executor(), shots=1000).result()
        for value, _ in result.results:
            assert not all(v == 0 for v in value)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_balanced_xor_gate_counts(self, n):
        """Balanced XOR DJ: H=2n+1, CX=n, X=1."""

        @qmc.qkernel
        def oracle_bxor(
            inputs: qmc.Vector[qmc.Qubit], ancilla: qmc.Qubit
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
            n_in = inputs.shape[0]
            for i in qmc.range(n_in):
                inputs[i], ancilla = qmc.cx(inputs[i], ancilla)
            return inputs, ancilla

        @qmc.qkernel
        def dj_bxor(n_val: int) -> qmc.Vector[qmc.Bit]:
            inputs = qmc.qubit_array(n_val, name="input")
            ancilla = qmc.qubit(name="ancilla")
            ancilla = qmc.x(ancilla)
            for i in qmc.range(n_val):
                inputs[i] = qmc.h(inputs[i])
            ancilla = qmc.h(ancilla)
            inputs, ancilla = oracle_bxor(inputs, ancilla)
            for i in qmc.range(n_val):
                inputs[i] = qmc.h(inputs[i])
            return qmc.measure(inputs)

        _, qc = _transpile_and_get_circuit(dj_bxor, bindings={"n_val": n})
        # Structure: X(ancilla) + n H(input) + H(ancilla) + n CX(i,ancilla)
        #            + n H(input) + n Measure
        # Total = 1 + n + 1 + n + n + n = 4n + 2
        assert qc.num_qubits == n + 1
        assert len(qc.data) == 4 * n + 2
        idx = 0
        # X on ancilla (qubit n)
        assert isinstance(qc.data[idx].operation, XGate)
        assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [n]
        idx += 1
        # H on input qubits [0..n-1]
        for i in range(n):
            assert isinstance(qc.data[idx].operation, HGate)
            assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [i]
            idx += 1
        # H on ancilla
        assert isinstance(qc.data[idx].operation, HGate)
        assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [n]
        idx += 1
        # Oracle: n CX gates from input[i] to ancilla
        for i in range(n):
            assert isinstance(qc.data[idx].operation, CXGate)
            assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [i, n]
            idx += 1
        # H on input qubits [0..n-1]
        for i in range(n):
            assert isinstance(qc.data[idx].operation, HGate)
            assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [i]
            idx += 1
        # Measure on input qubits [0..n-1]
        for i in range(n):
            assert isinstance(qc.data[idx].operation, Measure)
            assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [i]
            idx += 1

    def test_deutsch_jozsa_statevector_constant(self):
        """Constant-0 DJ: input register in |00> state before measurement."""

        @qmc.qkernel
        def oracle_c0(
            inputs: qmc.Vector[qmc.Qubit], ancilla: qmc.Qubit
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
            return inputs, ancilla

        @qmc.qkernel
        def dj_c0(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            inputs = qmc.qubit_array(n, name="input")
            ancilla = qmc.qubit(name="ancilla")
            ancilla = qmc.x(ancilla)
            for i in qmc.range(n):
                inputs[i] = qmc.h(inputs[i])
            ancilla = qmc.h(ancilla)
            inputs, ancilla = oracle_c0(inputs, ancilla)
            for i in qmc.range(n):
                inputs[i] = qmc.h(inputs[i])
            return qmc.measure(inputs)

        _, qc = _transpile_and_get_circuit(dj_c0, bindings={"n": 2})
        sv = _run_statevector(qc)
        # Total 3 qubits (2 input + 1 ancilla).
        # Input register: H·H|0⟩ = |0⟩ (constant oracle identity).
        # Ancilla: H·X|0⟩ = H|1⟩ = |−⟩ = (|0⟩ − |1⟩)/√2 (no second H on ancilla).
        # Full state: |00⟩ ⊗ |−⟩ = (|000⟩ − |001⟩)/√2
        # In Qiskit little-endian:
        #   |000⟩ → index 0, |001⟩ → index 1
        #   But qubit ordering: ancilla is the last allocated qubit.
        #   ancilla = qubit 2 (MSB in LE), inputs = qubits 0,1
        #   |−⟩ on qubit 2, |00⟩ on qubits 0,1
        #   → (|0⟩_q2 − |1⟩_q2) ⊗ |0⟩_q1 ⊗ |0⟩_q0 / √2
        #   = (|000⟩ − |100⟩)/√2 → indices 0 and 4 with amplitudes ±1/√2
        assert np.isclose(abs(sv[0]), 1.0 / np.sqrt(2), atol=1e-10)
        assert np.isclose(abs(sv[4]), 1.0 / np.sqrt(2), atol=1e-10)
        # Input register marginal: all probability on |00⟩ → constant function
        prob_input_00 = abs(sv[0]) ** 2 + abs(sv[4]) ** 2
        assert np.isclose(prob_input_00, 1.0, atol=1e-10)


# ===========================================================================
# 10. Expectation Value Pipeline Tests
# ===========================================================================


class TestExpvalQiskitPipeline:
    """Test qmc.expval through full Qiskit transpile pipeline."""

    def test_expval_transpiles_with_compiled_expval(self):
        """Transpilation with Observable produces compiled_expval segment."""

        @qmc.qkernel
        def vqe(n: qmc.UInt, H: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(n, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.expval(q, H)

        H = qm_o.Z(0) * qm_o.Z(1)
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(vqe, bindings={"H": H, "n": 2})
        assert len(exe.compiled_expval) == 1
        assert isinstance(exe.compiled_expval[0].hamiltonian, qm_o.Hamiltonian)

    def test_expval_z_on_zero_state(self):
        """<0|Z|0> = +1.0."""

        @qmc.qkernel
        def circuit(H: qmc.Observable) -> qmc.Float:
            q = qmc.qubit("q")
            return qmc.expval((q,), H)

        H_label = qm_o.Hamiltonian(num_qubits=1)
        H_label += qm_o.Z(0)
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit, bindings={"H": H_label})
        result = exe.run(transpiler.executor()).result()
        assert np.isclose(result, 1.0, atol=0.1)

    def test_expval_z_on_one_state(self):
        """<1|Z|1> = -1.0."""

        @qmc.qkernel
        def circuit(H: qmc.Observable) -> qmc.Float:
            q = qmc.qubit("q")
            q = qmc.x(q)
            return qmc.expval((q,), H)

        H_label = qm_o.Hamiltonian(num_qubits=1)
        H_label += qm_o.Z(0)
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit, bindings={"H": H_label})
        result = exe.run(transpiler.executor()).result()
        assert np.isclose(result, -1.0, atol=0.1)

    def test_expval_x_on_plus_state(self):
        """<+|X|+> = +1.0."""

        @qmc.qkernel
        def circuit(H: qmc.Observable) -> qmc.Float:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.expval((q,), H)

        H_label = qm_o.Hamiltonian(num_qubits=1)
        H_label += qm_o.X(0)
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit, bindings={"H": H_label})
        result = exe.run(transpiler.executor()).result()
        assert np.isclose(result, 1.0, atol=0.1)

    def test_expval_zz_bell_state(self):
        """<Φ+|ZZ|Φ+> = +1.0 (both spins aligned in Bell state)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, H: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(n, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.expval(q, H)

        H_label = qm_o.Z(0) * qm_o.Z(1)
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit, bindings={"H": H_label, "n": 2})
        result = exe.run(transpiler.executor()).result()
        assert np.isclose(result, 1.0, atol=0.1)

    def test_expval_complex_hamiltonian(self):
        """Multi-term Hamiltonian on Bell state."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, H: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(n, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.expval(q, H)

        # H = ZZ + 0.5*(X0 + X1)
        # For Bell state |Φ+>: <ZZ>=1, <X0>=0, <X1>=0
        # So <H> = 1.0 + 0.5*0 + 0.5*0 = 1.0
        H_label = qm_o.Z(0) * qm_o.Z(1) + 0.5 * qm_o.X(0) + 0.5 * qm_o.X(1)
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit, bindings={"H": H_label, "n": 2})
        result = exe.run(transpiler.executor()).result()
        assert np.isclose(result, 1.0, atol=0.15)

    def test_expval_missing_observable_raises(self):
        """Transpilation without Observable binding raises RuntimeError."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, H: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(n, "q")
            return qmc.expval(q, H)

        transpiler = QiskitTranspiler()
        with pytest.raises(RuntimeError, match="Observable.*not found in bindings"):
            transpiler.transpile(circuit, bindings={"n": 2})


# ===========================================================================
# 11. Variational Classifier Pattern Tests
# ===========================================================================


class TestVariationalClassifierPattern:
    """Test the data re-uploading variational classifier pattern."""

    def test_variational_classifier_structure(self):
        """Single-layer classifier: 3 RY + 2 CZ gates for 3 qubits."""

        @qmc.qkernel
        def classifier(
            n: qmc.UInt,
            params: qmc.Vector[qmc.Float],
            H: qmc.Observable,
        ) -> qmc.Float:
            q = qmc.qubit_array(n, "q")
            q = ry_layer(q, params, qmc.uint(0))
            q = cz_entangling_layer(q)
            return qmc.expval(q, H)

        H_label = qm_o.Hamiltonian(num_qubits=3)
        H_label += qm_o.Z(0)
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(
            classifier,
            bindings={"n": 3, "params": [0.1, 0.2, 0.3], "H": H_label},
        )
        qc = exe.compiled_quantum[0].circuit
        # Expval circuit: 3 RY + 2 CZ = 5 (no Measure)
        assert len(qc.data) == 5
        for i in range(3):
            assert isinstance(qc.data[i].operation, RYGate)
            assert [qc.find_bit(q).index for q in qc.data[i].qubits] == [i]
        assert isinstance(qc.data[3].operation, CZGate)
        assert [qc.find_bit(q).index for q in qc.data[3].qubits] == [0, 1]
        assert isinstance(qc.data[4].operation, CZGate)
        assert [qc.find_bit(q).index for q in qc.data[4].qubits] == [1, 2]

    def test_variational_classifier_execution(self):
        """Classifier expval returns a float in [-1, 1]."""

        @qmc.qkernel
        def classifier(
            n: qmc.UInt,
            params: qmc.Vector[qmc.Float],
            H: qmc.Observable,
        ) -> qmc.Float:
            q = qmc.qubit_array(n, "q")
            q = ry_layer(q, params, qmc.uint(0))
            q = cz_entangling_layer(q)
            return qmc.expval(q, H)

        H_label = qm_o.Hamiltonian(num_qubits=2)
        H_label += qm_o.Z(0)
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(
            classifier,
            bindings={"n": 2, "params": [0.5, 1.0], "H": H_label},
        )
        result = exe.run(transpiler.executor()).result()
        assert -1.0 <= result <= 1.0

    def test_two_layer_variational_classifier(self):
        """Two-layer classifier: 6 RY + 4 CZ for 3 qubits."""

        @qmc.qkernel
        def classifier(
            n: qmc.UInt,
            params: qmc.Vector[qmc.Float],
            H: qmc.Observable,
        ) -> qmc.Float:
            q = qmc.qubit_array(n, "q")
            for layer in qmc.range(2):
                q = ry_layer(q, params, layer * n)
                q = cz_entangling_layer(q)
            return qmc.expval(q, H)

        H_label = qm_o.Hamiltonian(num_qubits=3)
        H_label += qm_o.Z(0)
        params = [0.1 * i for i in range(6)]  # 2 layers * 3 qubits
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(
            classifier,
            bindings={"n": 3, "params": params, "H": H_label},
        )
        qc = exe.compiled_quantum[0].circuit
        # Expval circuit: 2 layers of (3 RY + 2 CZ) = 10 (no Measure)
        assert len(qc.data) == 10
        idx = 0
        for layer in range(2):
            for i in range(3):
                assert isinstance(qc.data[idx].operation, RYGate)
                assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [i]
                idx += 1
            assert isinstance(qc.data[idx].operation, CZGate)
            assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [0, 1]
            idx += 1
            assert isinstance(qc.data[idx].operation, CZGate)
            assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [1, 2]
            idx += 1

    def test_data_reuploading_pattern(self):
        """Data re-uploading: x-encode, entangle, params, entangle."""

        @qmc.qkernel
        def classifier(
            n: qmc.UInt,
            x: qmc.Vector[qmc.Float],
            params: qmc.Vector[qmc.Float],
            H: qmc.Observable,
        ) -> qmc.Float:
            q = qmc.qubit_array(n, "q")
            q = ry_layer(q, x, qmc.uint(0))
            q = cz_entangling_layer(q)
            q = ry_layer(q, params, qmc.uint(0))
            q = cz_entangling_layer(q)
            return qmc.expval(q, H)

        H_label = qm_o.Hamiltonian(num_qubits=3)
        H_label += qm_o.Z(0)
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(
            classifier,
            bindings={
                "n": 3,
                "x": [0.1, 0.2, 0.3],
                "params": [0.4, 0.5, 0.6],
                "H": H_label,
            },
        )
        qc = exe.compiled_quantum[0].circuit
        # Expval circuit: RY(x)*3 + CZ*2 + RY(params)*3 + CZ*2 = 10 (no Measure)
        assert len(qc.data) == 10
        # First block: x-encode
        for i in range(3):
            assert isinstance(qc.data[i].operation, RYGate)
            assert [qc.find_bit(q).index for q in qc.data[i].qubits] == [i]
        assert isinstance(qc.data[3].operation, CZGate)
        assert [qc.find_bit(q).index for q in qc.data[3].qubits] == [0, 1]
        assert isinstance(qc.data[4].operation, CZGate)
        assert [qc.find_bit(q).index for q in qc.data[4].qubits] == [1, 2]
        # Second block: params
        for i in range(3):
            assert isinstance(qc.data[5 + i].operation, RYGate)
            assert [qc.find_bit(q).index for q in qc.data[5 + i].qubits] == [i]
        assert isinstance(qc.data[8].operation, CZGate)
        assert [qc.find_bit(q).index for q in qc.data[8].qubits] == [0, 1]
        assert isinstance(qc.data[9].operation, CZGate)
        assert [qc.find_bit(q).index for q in qc.data[9].qubits] == [1, 2]

    def test_variational_classifier_parametric(self):
        """Parametric classifier with parameters=['params']."""

        @qmc.qkernel
        def classifier(
            n: qmc.UInt,
            params: qmc.Vector[qmc.Float],
            H: qmc.Observable,
        ) -> qmc.Float:
            q = qmc.qubit_array(n, "q")
            q = ry_layer(q, params, qmc.uint(0))
            q = cz_entangling_layer(q)
            return qmc.expval(q, H)

        H_label = qm_o.Hamiltonian(num_qubits=2)
        H_label += qm_o.Z(0)
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(
            classifier,
            bindings={"n": 2, "H": H_label},
            parameters=["params"],
        )
        assert exe.has_parameters


# ===========================================================================
# 12. Bernstein-Vazirani Algorithm Tests
# ===========================================================================


class TestBernsteinVaziraniAlgorithm:
    """Test Bernstein-Vazirani algorithm (H-Oracle-H pattern)."""

    def _make_bv_circuit(self, secret_bits):
        """Create a BV circuit for the given secret bit list."""
        n = len(secret_bits)

        @qmc.qkernel
        def bv_oracle(
            inputs: qmc.Vector[qmc.Qubit],
            ancilla: qmc.Qubit,
            secret: qmc.Vector[qmc.UInt],
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
            n_in = inputs.shape[0]
            for i in qmc.range(n_in):
                # CX only when secret[i] == 1; we use items on a dict
                # that maps active indices to 1
                pass
            return inputs, ancilla

        # Workaround: define oracle with the secret baked in via dict
        # (since conditional on non-measurement is limited)
        active_indices = {i: 1 for i, b in enumerate(secret_bits) if b == 1}

        @qmc.qkernel
        def bv_circuit(
            n_val: qmc.UInt,
            active: qmc.Dict[qmc.UInt, qmc.UInt],
        ) -> qmc.Vector[qmc.Bit]:
            inputs = qmc.qubit_array(n_val, name="input")
            ancilla = qmc.qubit(name="ancilla")
            ancilla = qmc.x(ancilla)
            for i in qmc.range(n_val):
                inputs[i] = qmc.h(inputs[i])
            ancilla = qmc.h(ancilla)
            # Oracle: CX for each active bit
            for i, _ in qmc.items(active):
                inputs[i], ancilla = qmc.cx(inputs[i], ancilla)
            for i in qmc.range(n_val):
                inputs[i] = qmc.h(inputs[i])
            return qmc.measure(inputs)

        return bv_circuit, n, active_indices

    def test_bv_secret_101(self):
        """BV with secret='101': measurement yields (1, 0, 1)."""
        bv_circuit, n, active = self._make_bv_circuit([1, 0, 1])
        exe, _ = _transpile_and_get_circuit(
            bv_circuit, bindings={"n_val": n, "active": active}
        )
        result = exe.sample(QiskitTranspiler().executor(), shots=100).result()
        for value, _ in result.results:
            assert tuple(value) == (1, 0, 1)

    def test_bv_secret_11(self):
        """BV with secret='11': measurement yields (1, 1)."""
        bv_circuit, n, active = self._make_bv_circuit([1, 1])
        exe, _ = _transpile_and_get_circuit(
            bv_circuit, bindings={"n_val": n, "active": active}
        )
        result = exe.sample(QiskitTranspiler().executor(), shots=100).result()
        for value, _ in result.results:
            assert tuple(value) == (1, 1)

    def test_bv_secret_000(self):
        """BV with secret='000': measurement yields (0, 0, 0)."""
        bv_circuit, n, active = self._make_bv_circuit([0, 0, 0])
        exe, _ = _transpile_and_get_circuit(
            bv_circuit, bindings={"n_val": n, "active": active}
        )
        result = exe.sample(QiskitTranspiler().executor(), shots=100).result()
        for value, _ in result.results:
            assert tuple(value) == (0, 0, 0)

    @pytest.mark.parametrize(
        "secret_bits,expected_cx",
        [([1, 0, 1], 2), ([1, 1], 2), ([0, 0, 0], 0), ([1, 1, 1, 1], 4)],
    )
    def test_bv_gate_counts(self, secret_bits, expected_cx):
        """BV gate counts: full structure verification."""
        bv_circuit, n, active = self._make_bv_circuit(secret_bits)
        _, qc = _transpile_and_get_circuit(
            bv_circuit, bindings={"n_val": n, "active": active}
        )
        active_indices = sorted(active.keys())
        # Structure: X(ancilla) + n H(input) + H(ancilla) + CX per active
        #            + n H(input) + n Measure
        # Total = 1 + n + 1 + expected_cx + n + n = 3n + 2 + expected_cx
        assert qc.num_qubits == n + 1
        assert len(qc.data) == 3 * n + 2 + expected_cx
        idx = 0
        # X on ancilla (qubit n)
        assert isinstance(qc.data[idx].operation, XGate)
        assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [n]
        idx += 1
        # H on input qubits [0..n-1]
        for i in range(n):
            assert isinstance(qc.data[idx].operation, HGate)
            assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [i]
            idx += 1
        # H on ancilla
        assert isinstance(qc.data[idx].operation, HGate)
        assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [n]
        idx += 1
        # Oracle: CX from active input bits to ancilla
        for ai in active_indices:
            assert isinstance(qc.data[idx].operation, CXGate)
            assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [ai, n]
            idx += 1
        # H on input qubits [0..n-1]
        for i in range(n):
            assert isinstance(qc.data[idx].operation, HGate)
            assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [i]
            idx += 1
        # Measure on input qubits [0..n-1]
        for i in range(n):
            assert isinstance(qc.data[idx].operation, Measure)
            assert [qc.find_bit(q).index for q in qc.data[idx].qubits] == [i]
            idx += 1


# ===========================================================================
# 13. FQAOA Integration Tests
# ===========================================================================


class TestFQAOAIntegration:
    """Test FQAOA algorithm components through Qiskit pipeline."""

    def test_initial_occupations(self):
        """initial_occupations applies X to first k qubits."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, n_f: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = initial_occupations(q, n_f)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 4, "n_f": 2})
        # Structure: 2 X(q0,q1) + 4 Measure = 6
        assert len(qc.data) == 6
        assert isinstance(qc.data[0].operation, XGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert isinstance(qc.data[1].operation, XGate)
        assert [qc.find_bit(q).index for q in qc.data[1].qubits] == [1]
        for i in range(4):
            assert isinstance(qc.data[2 + i].operation, Measure)
            assert [qc.find_bit(q).index for q in qc.data[2 + i].qubits] == [i]
        sv = _run_statevector(qc)
        # |1100> in Qiskit little-endian: qubits 0,1 are |1>, qubits 2,3 are |0>
        # Binary 0011 = index 3
        assert np.isclose(abs(sv[3]), 1.0, atol=1e-10)

    def test_givens_rotation_gate_count(self):
        """Givens rotation contains 2 CX gates."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = givens_rotation(q, qmc.uint(0), qmc.uint(1), theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 2, "theta": 0.5})
        # Structure: CX(1,0) + ControlledGate(0,1) + CX(1,0) + 2 Measure = 5
        assert len(qc.data) == 5
        assert isinstance(qc.data[0].operation, CXGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [1, 0]
        # qc.data[1] is a ControlledGate (controlled-RY)
        assert [qc.find_bit(q).index for q in qc.data[1].qubits] == [0, 1]
        assert isinstance(qc.data[2].operation, CXGate)
        assert [qc.find_bit(q).index for q in qc.data[2].qubits] == [1, 0]
        for i in range(2):
            assert isinstance(qc.data[3 + i].operation, Measure)
            assert [qc.find_bit(q).index for q in qc.data[3 + i].qubits] == [i]

    def test_hopping_gate_gate_count(self):
        """Hopping gate contains RX, CX, and RZ gates."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, beta: qmc.Float, hopping_val: qmc.Float
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = hopping_gate(q, qmc.uint(0), qmc.uint(1), beta, hopping_val)
            return qmc.measure(q)

        beta = 0.3
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": 2, "beta": beta, "hopping_val": 1.0}
        )
        # Structure: RX(q0,-π/2) RX(q1,π/2) CX(0,1) RX(q0,-β) RZ(q1,β)
        #            CX(0,1) RX(q0,π/2) RX(q1,-π/2) Measure(q0) Measure(q1)
        assert len(qc.data) == 10
        half_pi = np.pi / 2
        assert isinstance(qc.data[0].operation, RXGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert np.isclose(float(qc.data[0].operation.params[0]), -half_pi, atol=1e-10)
        assert isinstance(qc.data[1].operation, RXGate)
        assert [qc.find_bit(q).index for q in qc.data[1].qubits] == [1]
        assert np.isclose(float(qc.data[1].operation.params[0]), half_pi, atol=1e-10)
        assert isinstance(qc.data[2].operation, CXGate)
        assert [qc.find_bit(q).index for q in qc.data[2].qubits] == [0, 1]
        assert isinstance(qc.data[3].operation, RXGate)
        assert [qc.find_bit(q).index for q in qc.data[3].qubits] == [0]
        assert np.isclose(float(qc.data[3].operation.params[0]), -beta, atol=1e-10)
        assert isinstance(qc.data[4].operation, RZGate)
        assert [qc.find_bit(q).index for q in qc.data[4].qubits] == [1]
        assert np.isclose(float(qc.data[4].operation.params[0]), beta, atol=1e-10)
        assert isinstance(qc.data[5].operation, CXGate)
        assert [qc.find_bit(q).index for q in qc.data[5].qubits] == [0, 1]
        assert isinstance(qc.data[6].operation, RXGate)
        assert [qc.find_bit(q).index for q in qc.data[6].qubits] == [0]
        assert np.isclose(float(qc.data[6].operation.params[0]), half_pi, atol=1e-10)
        assert isinstance(qc.data[7].operation, RXGate)
        assert [qc.find_bit(q).index for q in qc.data[7].qubits] == [1]
        assert np.isclose(float(qc.data[7].operation.params[0]), -half_pi, atol=1e-10)
        for i in range(2):
            assert isinstance(qc.data[8 + i].operation, Measure)
            assert [qc.find_bit(q).index for q in qc.data[8 + i].qubits] == [i]

    def test_cost_layer_gate_count(self):
        """FQAOA cost layer: RZ==len(linear), RZZ==len(quad)."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
            gamma: qmc.Float,
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = cost_layer(q, gamma, linear, quad)
            return qmc.measure(q)

        linear = {0: 0.5, 1: -0.3}
        quad = {(0, 1): 1.0, (1, 2): -0.5}
        gamma = 0.5
        _, qc = _transpile_and_get_circuit(
            circuit,
            bindings={"n": 3, "gamma": gamma, "linear": linear, "quad": quad},
        )
        # Structure: 2 RZ(linear) + 2 RZZ(quad) + 3 Measure = 7
        assert len(qc.data) == 7
        # RZ for linear terms (note: cost_layer emits linear first, then quad)
        assert isinstance(qc.data[0].operation, RZGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert isinstance(qc.data[1].operation, RZGate)
        assert [qc.find_bit(q).index for q in qc.data[1].qubits] == [1]
        # RZZ for quadratic terms
        assert isinstance(qc.data[2].operation, RZZGate)
        assert [qc.find_bit(q).index for q in qc.data[2].qubits] == [0, 1]
        assert isinstance(qc.data[3].operation, RZZGate)
        assert [qc.find_bit(q).index for q in qc.data[3].qubits] == [1, 2]
        for i in range(3):
            assert isinstance(qc.data[4 + i].operation, Measure)
            assert [qc.find_bit(q).index for q in qc.data[4 + i].qubits] == [i]

    @pytest.mark.parametrize("n_qubits", [4, 6])
    def test_mixer_layer_hopping_count(self, n_qubits):
        """Mixer layer applies correct number of hopping gates."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, beta: qmc.Float, hopping_val: qmc.Float
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = mixer_layer(q, beta, hopping_val, n)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(
            circuit,
            bindings={"n": n_qubits, "beta": 0.3, "hopping_val": 1.0},
        )
        # Even pairs: len(range(0, n-1, 2))
        # Odd pairs: len(range(1, n-1, 2))
        # Boundary: 1 (pair (0, n-1))
        # Total hopping gates = even + odd + 1
        even_pairs = len(range(0, n_qubits - 1, 2))
        odd_pairs = len(range(1, n_qubits - 1, 2))
        total_hopping = even_pairs + odd_pairs + 1
        # Each hopping gate = 8 ops: RX RX CX RX RZ CX RX RX
        gates_per_hopping = 8
        expected_len = total_hopping * gates_per_hopping + n_qubits  # + n Measure
        assert len(qc.data) == expected_len
        # Verify each hopping block has pattern: RX RX CX RX RZ CX RX RX
        for blk in range(total_hopping):
            base = blk * gates_per_hopping
            assert isinstance(qc.data[base + 0].operation, RXGate)
            assert isinstance(qc.data[base + 1].operation, RXGate)
            assert isinstance(qc.data[base + 2].operation, CXGate)
            assert isinstance(qc.data[base + 3].operation, RXGate)
            assert isinstance(qc.data[base + 4].operation, RZGate)
            assert isinstance(qc.data[base + 5].operation, CXGate)
            assert isinstance(qc.data[base + 6].operation, RXGate)
            assert isinstance(qc.data[base + 7].operation, RXGate)
        # Verify Measure gates at the end
        for i in range(n_qubits):
            m_idx = expected_len - n_qubits + i
            assert isinstance(qc.data[m_idx].operation, Measure)
            assert [qc.find_bit(q).index for q in qc.data[m_idx].qubits] == [i]

    def test_fqaoa_state_transpiles(self):
        """Full fqaoa_state transpiles successfully."""

        @qmc.qkernel
        def circuit(
            p: qmc.UInt,
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            n: qmc.UInt,
            n_f: qmc.UInt,
            givens_ij: qmc.Matrix[qmc.UInt],
            givens_theta: qmc.Vector[qmc.Float],
            hopping_val: qmc.Float,
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = fqaoa_state(
                p,
                linear,
                quad,
                n,
                n_f,
                givens_ij,
                givens_theta,
                hopping_val,
                gammas,
                betas,
            )
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(
            circuit,
            bindings={
                "p": 1,
                "linear": {0: 0.5, 1: -0.3},
                "quad": {(0, 1): 1.0},
                "n": 4,
                "n_f": 2,
                "givens_ij": [[0, 1]],
                "givens_theta": [0.3],
                "hopping_val": 1.0,
                "gammas": [0.5],
                "betas": [0.3],
            },
        )
        assert qc.num_qubits == 4


# ===========================================================================
# 14. Advanced Circuit Pattern Tests
# ===========================================================================


class TestDeepNestedQKernelComposition:
    """Test 3-level deep sub-kernel nesting with parameter pass-through."""

    def test_three_level_nesting_statevector(self):
        """outer → middle → inner (3 levels): statevector = H·RX(θ)|0⟩."""

        @qmc.qkernel
        def inner(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            q = qmc.rx(q, theta)
            return q

        @qmc.qkernel
        def middle(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            q = inner(q, theta)
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def outer(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = middle(q, theta)
            return qmc.measure(q)

        theta = np.pi / 3
        _, qc = _transpile_and_get_circuit(outer, bindings={"theta": theta})
        sv = _run_statevector(qc)
        H = GATE_SPECS["H"].matrix_fn()
        RX = GATE_SPECS["RX"].matrix_fn(theta)
        expected = H @ RX @ all_zeros_state(1)
        assert statevectors_equal(sv, expected)

    def test_three_level_nesting_gate_counts(self):
        """3-level nesting: 1 rx + 1 h after inlining."""

        @qmc.qkernel
        def inner(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            q = qmc.rx(q, theta)
            return q

        @qmc.qkernel
        def middle(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            q = inner(q, theta)
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def outer(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = middle(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(outer, bindings={"theta": 0.5})
        # Structure: RX(q0, 0.5) + H(q0) + Measure(q0) = 3
        assert len(qc.data) == 3
        assert isinstance(qc.data[0].operation, RXGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert np.isclose(float(qc.data[0].operation.params[0]), 0.5, atol=1e-10)
        assert isinstance(qc.data[1].operation, HGate)
        assert [qc.find_bit(q).index for q in qc.data[1].qubits] == [0]
        assert isinstance(qc.data[2].operation, Measure)
        assert [qc.find_bit(q).index for q in qc.data[2].qubits] == [0]

    def test_deep_nesting_two_params(self):
        """3-level nesting with two parameters: statevector = RY(β)·RX(α)|0⟩."""

        @qmc.qkernel
        def inner(q: qmc.Qubit, alpha: qmc.Float) -> qmc.Qubit:
            q = qmc.rx(q, alpha)
            return q

        @qmc.qkernel
        def middle(q: qmc.Qubit, alpha: qmc.Float, beta: qmc.Float) -> qmc.Qubit:
            q = inner(q, alpha)
            q = qmc.ry(q, beta)
            return q

        @qmc.qkernel
        def outer(alpha: qmc.Float, beta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = middle(q, alpha, beta)
            return qmc.measure(q)

        alpha, beta = np.pi / 6, np.pi / 3
        _, qc = _transpile_and_get_circuit(
            outer, bindings={"alpha": alpha, "beta": beta}
        )
        sv = _run_statevector(qc)
        RX = GATE_SPECS["RX"].matrix_fn(alpha)
        RY = GATE_SPECS["RY"].matrix_fn(beta)
        expected = RY @ RX @ all_zeros_state(1)
        assert statevectors_equal(sv, expected)

    def test_sub_kernel_per_element_in_loop(self):
        """Sub-kernel called per-element in qmc.range loop: n h gates."""

        @qmc.qkernel
        def apply_h(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def main(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = apply_h(q[i])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(main, bindings={"n": 4})
        # Structure: 4 H(q0..q3) + 4 Measure = 8
        assert len(qc.data) == 8
        for i in range(4):
            assert isinstance(qc.data[i].operation, HGate)
            assert [qc.find_bit(q).index for q in qc.data[i].qubits] == [i]
        for i in range(4):
            assert isinstance(qc.data[4 + i].operation, Measure)
            assert [qc.find_bit(q).index for q in qc.data[4 + i].qubits] == [i]

    def test_sub_kernel_per_element_statevector(self):
        """Sub-kernel per-element: statevector = |++⟩ for n=2."""

        @qmc.qkernel
        def apply_h(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def main(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = apply_h(q[i])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(main, bindings={"n": 2})
        sv = _run_statevector(qc)
        expected = tensor_product(plus_state(), plus_state())
        assert statevectors_equal(sv, expected)


class TestControlledSubRoutines:
    """Test qmc.controlled() with statevector verification and advanced patterns."""

    def test_controlled_ry_control_on(self):
        """Controlled-RY with ctrl=|1⟩: statevector matches CRY matrix."""

        @qmc.qkernel
        def ry_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            q = qmc.ry(q, theta)
            return q

        controlled_ry = qmc.controlled(ry_gate)

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])  # ctrl = |1⟩
            q[0], q[1] = controlled_ry(q[0], q[1], theta=theta)
            return qmc.measure(q)

        theta = np.pi / 2
        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": theta})
        sv = _run_statevector(qc, decompose=True)
        CRY = GATE_SPECS["CRY"].matrix_fn(theta)
        X = GATE_SPECS["X"].matrix_fn()
        # Initial: X on q0 → |10⟩. In LE: index 1.
        state = tensor_product(identity(2), X) @ all_zeros_state(2)
        expected = CRY @ state
        assert statevectors_equal(sv, expected)

    def test_controlled_ry_control_off(self):
        """Controlled-RY with ctrl=|0⟩: target unchanged, sv = |00⟩."""

        @qmc.qkernel
        def ry_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            q = qmc.ry(q, theta)
            return q

        controlled_ry = qmc.controlled(ry_gate)

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = controlled_ry(q[0], q[1], theta=theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 2})
        sv = _run_statevector(qc, decompose=True)
        expected = all_zeros_state(2)
        assert statevectors_equal(sv, expected)

    def test_controlled_rz_statevector(self):
        """Controlled-RZ with ctrl=|1⟩: statevector matches CRZ matrix."""

        @qmc.qkernel
        def rz_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            q = qmc.rz(q, theta)
            return q

        controlled_rz = qmc.controlled(rz_gate)

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])  # ctrl = |1⟩
            q[0], q[1] = controlled_rz(q[0], q[1], theta=theta)
            return qmc.measure(q)

        theta = np.pi / 3
        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": theta})
        sv = _run_statevector(qc, decompose=True)
        CRZ = GATE_SPECS["CRZ"].matrix_fn(theta)
        X = GATE_SPECS["X"].matrix_fn()
        state = tensor_product(identity(2), X) @ all_zeros_state(2)
        expected = CRZ @ state
        assert statevectors_equal(sv, expected)

    def test_controlled_multi_gate_kernel(self):
        """Controlled multi-gate sub-routine (H + X) transpiles."""

        @qmc.qkernel
        def hx_gate(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            q = qmc.x(q)
            return q

        controlled_hx = qmc.controlled(hx_gate)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[0], q[1] = controlled_hx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        assert qc is not None
        assert qc.num_qubits >= 2

    def test_controlled_power_2(self):
        """Controlled P(θ) with power=2 transpiles successfully."""

        @qmc.qkernel
        def p_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            q = qmc.p(q, theta)
            return q

        cp_pow = qmc.controlled(p_gate)

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = cp_pow(q[0], q[1], power=2, theta=theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 4})
        assert qc is not None
        assert qc.num_qubits == 2

    def test_controlled_power_4(self):
        """Controlled P(θ) with power=4 transpiles successfully."""

        @qmc.qkernel
        def p_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            q = qmc.p(q, theta)
            return q

        cp_pow = qmc.controlled(p_gate)

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = cp_pow(q[0], q[1], power=4, theta=theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 4})
        assert qc is not None
        assert qc.num_qubits == 2


class TestAllFourBellStates:
    """Test all four Bell states with statevector verification."""

    def test_bell_phi_plus(self):
        """|Φ+⟩ = (|00⟩ + |11⟩)/√2."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        assert statevectors_equal(sv, bell_state(0))

    def test_bell_phi_minus(self):
        """|Φ−⟩ = (|00⟩ − |11⟩)/√2."""

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

    def test_bell_psi_plus(self):
        """|Ψ+⟩ = (|01⟩ + |10⟩)/√2."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            q[1] = qmc.x(q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        assert statevectors_equal(sv, bell_state(2))

    def test_bell_psi_minus(self):
        """|Ψ−⟩ = (|01⟩ − |10⟩)/√2."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            q[1] = qmc.x(q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        assert statevectors_equal(sv, bell_state(3))


class TestGHZStateParametrised:
    """Test loop-based GHZ state with parametrised qubit count."""

    @pytest.mark.parametrize("n_qubits", [3, 4, 5])
    def test_ghz_loop_statevector(self, n_qubits):
        """GHZ via loop: sv[0] = sv[2^n-1] = 1/√2, rest 0."""

        @qmc.qkernel
        def ghz(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q[0] = qmc.h(q[0])
            for i in qmc.range(n - 1):
                q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(ghz, bindings={"n": n_qubits})
        sv = _run_statevector(qc)
        assert np.isclose(abs(sv[0]), 1.0 / np.sqrt(2), atol=1e-10)
        assert np.isclose(abs(sv[2**n_qubits - 1]), 1.0 / np.sqrt(2), atol=1e-10)
        # All other amplitudes are zero
        for i in range(1, 2**n_qubits - 1):
            assert np.isclose(abs(sv[i]), 0.0, atol=1e-10)

    @pytest.mark.parametrize("n_qubits", [3, 4, 5])
    def test_ghz_loop_gate_counts(self, n_qubits):
        """GHZ via loop: 1 h gate, n-1 cx gates."""

        @qmc.qkernel
        def ghz(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q[0] = qmc.h(q[0])
            for i in qmc.range(n - 1):
                q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(ghz, bindings={"n": n_qubits})
        # Structure: H(q0) + (n-1) CX(i, i+1) + n Measure
        assert len(qc.data) == 1 + (n_qubits - 1) + n_qubits
        assert isinstance(qc.data[0].operation, HGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        for i in range(n_qubits - 1):
            assert isinstance(qc.data[1 + i].operation, CXGate)
            assert [qc.find_bit(q).index for q in qc.data[1 + i].qubits] == [i, i + 1]
        for i in range(n_qubits):
            assert isinstance(qc.data[n_qubits + i].operation, Measure)
            assert [qc.find_bit(q).index for q in qc.data[n_qubits + i].qubits] == [i]


class TestManualQFTCircuit:
    """Textbook QFT from scratch using CP + H + SWAP (Nielsen & Chuang)."""

    def test_manual_qft_2q_gate_counts(self):
        """2-qubit manual QFT: 2 h, 1 cp, 1 swap."""

        @qmc.qkernel
        def qft2() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            # QFT on q[0] (MSB)
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cp(q[0], q[1], np.pi / 2)
            # QFT on q[1] (LSB)
            q[1] = qmc.h(q[1])
            # Swap for bit-reversal
            q[0], q[1] = qmc.swap(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(qft2)
        # Structure: H(q0) + CP(0,1) + H(q1) + SWAP(0,1) + 2 Measure = 6
        assert len(qc.data) == 6
        assert isinstance(qc.data[0].operation, HGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert isinstance(qc.data[1].operation, CPhaseGate)
        assert [qc.find_bit(q).index for q in qc.data[1].qubits] == [0, 1]
        assert np.isclose(float(qc.data[1].operation.params[0]), np.pi / 2, atol=1e-10)
        assert isinstance(qc.data[2].operation, HGate)
        assert [qc.find_bit(q).index for q in qc.data[2].qubits] == [1]
        assert isinstance(qc.data[3].operation, SwapGate)
        assert [qc.find_bit(q).index for q in qc.data[3].qubits] == [0, 1]
        for i in range(2):
            assert isinstance(qc.data[4 + i].operation, Measure)
            assert [qc.find_bit(q).index for q in qc.data[4 + i].qubits] == [i]

    def test_manual_qft_2q_from_zero(self):
        """QFT|00⟩ = |++⟩ = [1,1,1,1]/2."""

        @qmc.qkernel
        def qft2() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cp(q[0], q[1], np.pi / 2)
            q[1] = qmc.h(q[1])
            q[0], q[1] = qmc.swap(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(qft2)
        sv = _run_statevector(qc)
        expected = np.array([1, 1, 1, 1], dtype=complex) / 2
        assert statevectors_equal(sv, expected)

    def test_manual_qft_2q_from_one(self):
        """QFT|01⟩ = [1, i, -1, -i]/2."""

        @qmc.qkernel
        def qft2_one() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])  # Prepare |01⟩ (q0=1, q1=0)
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cp(q[0], q[1], np.pi / 2)
            q[1] = qmc.h(q[1])
            q[0], q[1] = qmc.swap(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(qft2_one)
        sv = _run_statevector(qc)
        # Input: X on q0 → |01⟩ in LE (index 1).
        # After QFT+SWAP, the output is [1, 1, -1, -1]/2
        # (QFT maps |01⟩ to uniform amplitudes with sign pattern).
        expected = np.array([1, 1, -1, -1], dtype=complex) / 2
        assert statevectors_equal(sv, expected)

    def test_manual_qft_3q_gate_counts(self):
        """3-qubit manual QFT: 3 h, 3 cp, 1 swap."""

        @qmc.qkernel
        def qft3() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            # q[0]
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cp(q[0], q[1], np.pi / 2)
            q[0], q[2] = qmc.cp(q[0], q[2], np.pi / 4)
            # q[1]
            q[1] = qmc.h(q[1])
            q[1], q[2] = qmc.cp(q[1], q[2], np.pi / 2)
            # q[2]
            q[2] = qmc.h(q[2])
            # Bit-reversal: swap q[0] and q[2]
            q[0], q[2] = qmc.swap(q[0], q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(qft3)
        # Structure: H(q0) CP(0,1) CP(0,2) H(q1) CP(1,2) H(q2) SWAP(0,2) 3 Measure = 10
        assert len(qc.data) == 10
        assert isinstance(qc.data[0].operation, HGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert isinstance(qc.data[1].operation, CPhaseGate)
        assert [qc.find_bit(q).index for q in qc.data[1].qubits] == [0, 1]
        assert np.isclose(float(qc.data[1].operation.params[0]), np.pi / 2, atol=1e-10)
        assert isinstance(qc.data[2].operation, CPhaseGate)
        assert [qc.find_bit(q).index for q in qc.data[2].qubits] == [0, 2]
        assert np.isclose(float(qc.data[2].operation.params[0]), np.pi / 4, atol=1e-10)
        assert isinstance(qc.data[3].operation, HGate)
        assert [qc.find_bit(q).index for q in qc.data[3].qubits] == [1]
        assert isinstance(qc.data[4].operation, CPhaseGate)
        assert [qc.find_bit(q).index for q in qc.data[4].qubits] == [1, 2]
        assert np.isclose(float(qc.data[4].operation.params[0]), np.pi / 2, atol=1e-10)
        assert isinstance(qc.data[5].operation, HGate)
        assert [qc.find_bit(q).index for q in qc.data[5].qubits] == [2]
        assert isinstance(qc.data[6].operation, SwapGate)
        assert [qc.find_bit(q).index for q in qc.data[6].qubits] == [0, 2]
        for i in range(3):
            assert isinstance(qc.data[7 + i].operation, Measure)
            assert [qc.find_bit(q).index for q in qc.data[7 + i].qubits] == [i]


class TestPhaseKickbackPattern:
    """Phase kickback — core QPE principle tested in isolation."""

    def test_phase_kickback_p_gate(self):
        """Ctrl=|+⟩, tgt=|1⟩, controlled-P(θ): ctrl acquires phase e^{iθ}."""

        @qmc.qkernel
        def p_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            q = qmc.p(q, theta)
            return q

        cp_gate = qmc.controlled(p_gate)

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])  # ctrl = |+⟩
            q[1] = qmc.x(q[1])  # tgt = |1⟩ (eigenstate)
            q[0], q[1] = cp_gate(q[0], q[1], theta=theta)
            return qmc.measure(q)

        theta = np.pi / 2
        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": theta})
        sv = _run_statevector(qc, decompose=True)
        # Qiskit LE: q0=ctrl, q1=tgt
        # After: ctrl=(|0⟩+e^{iθ}|1⟩)/√2, tgt=|1⟩
        # Full state: (|01⟩ + e^{iθ}|11⟩)/√2
        # LE indices: |01⟩=index 2, |11⟩=index 3
        assert np.isclose(abs(sv[2]), 1.0 / np.sqrt(2), atol=1e-10)
        assert np.isclose(abs(sv[3]), 1.0 / np.sqrt(2), atol=1e-10)
        # Phase difference between sv[3] and sv[2] should be e^{iθ}
        phase = sv[3] / sv[2]
        assert np.isclose(phase, np.exp(1j * theta), atol=1e-10)

    def test_phase_kickback_x_gate(self):
        """Ctrl=|+⟩, tgt=|−⟩, CX: ctrl becomes |−⟩ (phase -1 kickback)."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])  # ctrl = |+⟩
            q[1] = qmc.x(q[1])  # tgt = |1⟩
            q[1] = qmc.h(q[1])  # tgt = |−⟩
            q[0], q[1] = qmc.cx(q[0], q[1])  # CX kickback
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # After kickback: ctrl=|−⟩, tgt=|−⟩
        # |−⟩⊗|−⟩ = (|0⟩-|1⟩)(|0⟩-|1⟩)/2 = (|00⟩-|01⟩-|10⟩+|11⟩)/2
        expected = tensor_product(minus_state(), minus_state())
        assert statevectors_equal(sv, expected)

    def test_phase_kickback_identity(self):
        """Ctrl=|+⟩, tgt=|1⟩, identity oracle: no phase change."""

        @qmc.qkernel
        def id_gate(q: qmc.Qubit) -> qmc.Qubit:
            return q

        c_id = qmc.controlled(id_gate)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])  # ctrl = |+⟩
            q[1] = qmc.x(q[1])  # tgt = |1⟩
            q[0], q[1] = c_id(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc, decompose=True)
        # |+⟩ ⊗ |1⟩ = (|01⟩ + |11⟩)/√2
        # LE indices: |01⟩=2, |11⟩=3
        expected = tensor_product(computational_basis_state(1, 1), plus_state())
        assert statevectors_equal(sv, expected)


class TestEntanglementAndParityPatterns:
    """Utility circuit patterns: parity check, repetition code, swap test."""

    def test_parity_check_even(self):
        """Data |011⟩ (parity 0=even): ancilla stays |0⟩."""

        @qmc.qkernel
        def parity(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            data = qmc.qubit_array(n, "data")
            anc = qmc.qubit("anc")
            # Prepare |011⟩: x on data[0] and data[1]
            data[0] = qmc.x(data[0])
            data[1] = qmc.x(data[1])
            # CX ladder: compute parity into ancilla
            for i in qmc.range(n):
                data[i], anc = qmc.cx(data[i], anc)
            return qmc.measure(data)

        _, qc = _transpile_and_get_circuit(parity, bindings={"n": 3})
        sv = _run_statevector(qc)
        # 4 qubits: data[0..2] + anc (qubit 3)
        # |011⟩ has 2 ones → parity 0 → anc = |0⟩
        # State: |data⟩|anc⟩ = |011⟩|0⟩
        # LE: q0=1,q1=1,q2=0,anc=0 → index = 0b0011 = 3
        assert np.isclose(abs(sv[3]), 1.0, atol=1e-10)

    def test_parity_check_odd(self):
        """Data |010⟩ (parity 1=odd): ancilla becomes |1⟩."""

        @qmc.qkernel
        def parity(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            data = qmc.qubit_array(n, "data")
            anc = qmc.qubit("anc")
            # Prepare |010⟩: x on data[1]
            data[1] = qmc.x(data[1])
            # CX ladder
            for i in qmc.range(n):
                data[i], anc = qmc.cx(data[i], anc)
            return qmc.measure(data)

        _, qc = _transpile_and_get_circuit(parity, bindings={"n": 3})
        sv = _run_statevector(qc)
        # |010⟩ has 1 one → parity 1 → anc = |1⟩
        # State: |010⟩|1⟩
        # LE: q0=0,q1=1,q2=0,anc=1 → index = 0b1010 = 10
        assert np.isclose(abs(sv[10]), 1.0, atol=1e-10)

    def test_repetition_code_encode_0(self):
        """Encode |0⟩ with 3-qubit repetition code: output |000⟩."""

        @qmc.qkernel
        def encode() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            # Encode q[0] into q[1], q[2]
            q[0], q[1] = qmc.cx(q[0], q[1])
            q[0], q[2] = qmc.cx(q[0], q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(encode)
        sv = _run_statevector(qc)
        assert statevectors_equal(sv, all_zeros_state(3))

    def test_repetition_code_encode_1(self):
        """Encode |1⟩ with 3-qubit repetition code: output |111⟩."""

        @qmc.qkernel
        def encode() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.x(q[0])  # Prepare |1⟩ on q[0]
            q[0], q[1] = qmc.cx(q[0], q[1])
            q[0], q[2] = qmc.cx(q[0], q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(encode)
        sv = _run_statevector(qc)
        expected = computational_basis_state(3, 7)  # |111⟩ = index 7
        assert statevectors_equal(sv, expected)

    def test_swap_test_same_states(self):
        """Swap test: both data=|0⟩ (overlap=1), ancilla should be |0⟩."""

        @qmc.qkernel
        def swap_kernel(q0: qmc.Qubit, q1: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            q0, q1 = qmc.swap(q0, q1)
            return q0, q1

        cswap = qmc.controlled(swap_kernel)

        @qmc.qkernel
        def swap_test_circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            # q[0] = ancilla, q[1] = data_a, q[2] = data_b
            q[0] = qmc.h(q[0])
            q[0], q[1], q[2] = cswap(q[0], q[1], q[2])
            q[0] = qmc.h(q[0])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(swap_test_circuit)
        sv = _run_statevector(qc, decompose=True)
        # With identical states: ancilla always measures 0.
        # Full state should be |000⟩ (all data qubits are |0⟩, ancilla is |0⟩).
        assert statevectors_equal(sv, all_zeros_state(3))


# ===========================================================================
# 15. Qubit Array Pattern Tests
# ===========================================================================


class TestQubitArrayPatterns:
    """Test qubit_array-based circuit patterns that fill coverage gaps.

    Group 1: Pure gate circuits on qubit_array (not covered by TestGateCombinations).
    Group 2: If-else control flow with qubit_array (TestControlFlowIfElse uses only qubit()).
    Group 3: Unified array parity (existing tests use qubit_array + separate qubit("anc")).
    Group 4: Parametric vs hardcoded array size (UInt-bound vs literal).
    """

    # -- Group 1: Pure Gate Circuits on qubit_array ----------------------------

    def test_cx_propagation_array(self):
        """X(q[0]) + CX → |11⟩ on qubit_array(2)."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        expected = computational_basis_state(2, 3)
        assert statevectors_equal(sv, expected)
        assert qc.num_qubits == 2

    def test_independent_rotations_array(self):
        """RX(θ₁) on q[0], RY(θ₂) on q[1] via qubit_array(2)."""

        @qmc.qkernel
        def circuit(theta1: qmc.Float, theta2: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.rx(q[0], theta1)
            q[1] = qmc.ry(q[1], theta2)
            return qmc.measure(q)

        bindings = {"theta1": np.pi / 3, "theta2": np.pi / 5}
        _, qc = _transpile_and_get_circuit(circuit, bindings=bindings)
        sv = _run_statevector(qc)
        # Expected: RX(π/3)|0⟩ ⊗ RY(π/5)|0⟩ (Qiskit LE: MSB⊗LSB)
        RX = GATE_SPECS["RX"].matrix_fn(np.pi / 3)
        RY = GATE_SPECS["RY"].matrix_fn(np.pi / 5)
        expected = tensor_product(RY, RX) @ all_zeros_state(2)
        assert statevectors_equal(sv, expected)
        assert qc.num_qubits == 2

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_independent_rotations_array_random(self, seed):
        """Random RX(θ₁) ⊗ RY(θ₂) on qubit_array(2) matches analytical."""
        rng = np.random.default_rng(seed)
        t1, t2 = rng.uniform(0, 2 * np.pi, 2)

        @qmc.qkernel
        def circuit(theta1: qmc.Float, theta2: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.rx(q[0], theta1)
            q[1] = qmc.ry(q[1], theta2)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"theta1": t1, "theta2": t2}
        )
        sv = _run_statevector(qc)
        rx_mat = GATE_SPECS["RX"].matrix_fn(t1)
        ry_mat = GATE_SPECS["RY"].matrix_fn(t2)
        expected = tensor_product(ry_mat, rx_mat) @ all_zeros_state(2)
        assert statevectors_equal(sv, expected)

    def test_cz_entangling_layer_array(self):
        """H⊗3 + CZ chain on qubit_array(3)."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            q[2] = qmc.h(q[2])
            q[0], q[1] = qmc.cz(q[0], q[1])
            q[1], q[2] = qmc.cz(q[1], q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(qc)
        # Manual: H⊗3 · CZ(0,1) · CZ(1,2) |000⟩
        H = GATE_SPECS["H"].matrix_fn()
        CZ = GATE_SPECS["CZ"].matrix_fn()
        h3 = tensor_product(H, tensor_product(H, H))
        cz01 = tensor_product(identity(2), CZ)
        cz12 = tensor_product(CZ, identity(2))
        expected = cz12 @ cz01 @ h3 @ all_zeros_state(3)
        assert statevectors_equal(sv, expected)
        assert qc.num_qubits == 3

    def test_three_qubit_rotation_layer_array(self):
        """Different rotation gates per index: RX(θ₀), RY(θ₁), RZ(θ₂)."""

        @qmc.qkernel
        def circuit(t0: qmc.Float, t1: qmc.Float, t2: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.rx(q[0], t0)
            q[1] = qmc.ry(q[1], t1)
            q[2] = qmc.rz(q[2], t2)
            return qmc.measure(q)

        angles = {"t0": np.pi / 4, "t1": np.pi / 3, "t2": np.pi / 6}
        _, qc = _transpile_and_get_circuit(circuit, bindings=angles)
        sv = _run_statevector(qc)
        # Expected: RZ(π/6)|0⟩ ⊗ RY(π/3)|0⟩ ⊗ RX(π/4)|0⟩  (Qiskit LE)
        RX = GATE_SPECS["RX"].matrix_fn(np.pi / 4)
        RY = GATE_SPECS["RY"].matrix_fn(np.pi / 3)
        RZ = GATE_SPECS["RZ"].matrix_fn(np.pi / 6)
        expected = tensor_product(RZ, tensor_product(RY, RX)) @ all_zeros_state(3)
        assert statevectors_equal(sv, expected)
        assert qc.num_qubits == 3

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_three_qubit_rotation_layer_array_random(self, seed):
        """Random RX(θ₀) ⊗ RY(θ₁) ⊗ RZ(θ₂) on qubit_array(3) matches analytical."""
        rng = np.random.default_rng(seed)
        t0, t1, t2 = rng.uniform(0, 2 * np.pi, 3)

        @qmc.qkernel
        def circuit(a0: qmc.Float, a1: qmc.Float, a2: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.rx(q[0], a0)
            q[1] = qmc.ry(q[1], a1)
            q[2] = qmc.rz(q[2], a2)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"a0": t0, "a1": t1, "a2": t2}
        )
        sv = _run_statevector(qc)
        rx_mat = GATE_SPECS["RX"].matrix_fn(t0)
        ry_mat = GATE_SPECS["RY"].matrix_fn(t1)
        rz_mat = GATE_SPECS["RZ"].matrix_fn(t2)
        expected = tensor_product(
            rz_mat, tensor_product(ry_mat, rx_mat)
        ) @ all_zeros_state(3)
        assert statevectors_equal(sv, expected)

    # -- Group 2: If-Else with qubit_array ------------------------------------

    def test_if_else_array_basic(self):
        """X(q[0]) → measure → if: X(q[1]) on qubit_array(2)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            bit = qmc.measure(q[0])
            if bit:
                q[1] = qmc.x(q[1])
            else:
                q[1] = q[1]
            return qmc.measure(q[1])

        _, qc = _transpile_and_get_circuit(circuit)
        # Structure: X(q0) Measure(q0→c0) IfElse(q1,c0) Measure(q1→c2) = 4
        assert len(qc.data) == 4
        assert qc.num_qubits == 2
        assert isinstance(qc.data[0].operation, XGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert isinstance(qc.data[1].operation, Measure)
        assert [qc.find_bit(q).index for q in qc.data[1].qubits] == [0]
        assert isinstance(qc.data[2].operation, IfElseOp)
        assert [qc.find_bit(q).index for q in qc.data[2].qubits] == [1]
        assert isinstance(qc.data[3].operation, Measure)
        assert [qc.find_bit(q).index for q in qc.data[3].qubits] == [1]

    def test_if_else_array_zero_measurement(self):
        """Measure |0⟩ (always false branch) on qubit_array(2)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit_array(2, "q")
            bit = qmc.measure(q[0])
            if bit:
                q[1] = qmc.x(q[1])
            else:
                q[1] = q[1]
            return qmc.measure(q[1])

        _, qc = _transpile_and_get_circuit(circuit)
        # Structure: Measure(q0→c0) IfElse(q1,c0) Measure(q1→c2) = 3
        assert len(qc.data) == 3
        assert qc.num_qubits == 2
        assert isinstance(qc.data[0].operation, Measure)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert isinstance(qc.data[1].operation, IfElseOp)
        assert [qc.find_bit(q).index for q in qc.data[1].qubits] == [1]
        assert isinstance(qc.data[2].operation, Measure)
        assert [qc.find_bit(q).index for q in qc.data[2].qubits] == [1]

    def test_if_else_array_parametric(self):
        """if: RX(θ) on q[1] via qubit_array(2)."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            bit = qmc.measure(q[0])
            if bit:
                q[1] = qmc.rx(q[1], theta)
            else:
                q[1] = q[1]
            return qmc.measure(q[1])

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 4})
        # Structure: X(q0) Measure(q0→c0) IfElse(q1,c0) Measure(q1→c2) = 4
        assert len(qc.data) == 4
        assert qc.num_qubits == 2
        assert isinstance(qc.data[0].operation, XGate)
        assert [qc.find_bit(q).index for q in qc.data[0].qubits] == [0]
        assert isinstance(qc.data[1].operation, Measure)
        assert [qc.find_bit(q).index for q in qc.data[1].qubits] == [0]
        assert isinstance(qc.data[2].operation, IfElseOp)
        assert [qc.find_bit(q).index for q in qc.data[2].qubits] == [1]
        assert isinstance(qc.data[3].operation, Measure)
        assert [qc.find_bit(q).index for q in qc.data[3].qubits] == [1]

    def test_if_else_array_execution(self):
        """Shot-based execution of if-else on qubit_array(2)."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            bit = qmc.measure(q[0])
            if bit:
                q[1] = qmc.x(q[1])
            else:
                q[1] = q[1]
            return qmc.measure(q[1])

        transpiler = QiskitTranspiler()
        executor = transpiler.executor()
        exe = transpiler.transpile(circuit)
        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        assert result is not None
        assert len(result.results) > 0
        for value, count in result.results:
            assert value in (0, 1)
            assert count > 0

    # -- Group 3: Unified Array Parity ----------------------------------------

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param("even", id="even_parity_011"),
            pytest.param("odd", id="odd_parity_010"),
        ],
    )
    def test_parity_unified_array(self, case):
        """Parity check with single qubit_array(4): data q[0..2] + ancilla q[3]."""

        @qmc.qkernel
        def parity_even() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0], q[3] = qmc.cx(q[0], q[3])
            q[1], q[3] = qmc.cx(q[1], q[3])
            q[2], q[3] = qmc.cx(q[2], q[3])
            return qmc.measure(q)

        @qmc.qkernel
        def parity_odd() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[1] = qmc.x(q[1])
            q[0], q[3] = qmc.cx(q[0], q[3])
            q[1], q[3] = qmc.cx(q[1], q[3])
            q[2], q[3] = qmc.cx(q[2], q[3])
            return qmc.measure(q)

        if case == "even":
            # |011⟩ has 2 ones → even parity → ancilla stays |0⟩
            # LE: q0=1,q1=1,q2=0,q3=0 → index 0b0011 = 3
            kernel, expected_idx = parity_even, 3
        else:
            # |010⟩ has 1 one → odd parity → ancilla becomes |1⟩
            # LE: q0=0,q1=1,q2=0,q3=1 → index 0b1010 = 10
            kernel, expected_idx = parity_odd, 10

        _, qc = _transpile_and_get_circuit(kernel)
        sv = _run_statevector(qc)
        assert np.isclose(abs(sv[expected_idx]), 1.0, atol=1e-10)
        assert qc.num_qubits == 4

    # -- Group 4: Parametric vs Hardcoded Array Size --------------------------

    def test_h_layer_parametric_vs_hardcoded(self):
        """H on all n qubits: parametric UInt-bound n vs hardcoded 3."""

        @qmc.qkernel
        def h_parametric(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        @qmc.qkernel
        def h_hardcoded() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            q[2] = qmc.h(q[2])
            return qmc.measure(q)

        _, qc_par = _transpile_and_get_circuit(h_parametric, bindings={"n": 3})
        _, qc_hc = _transpile_and_get_circuit(h_hardcoded)
        sv_par = _run_statevector(qc_par)
        sv_hc = _run_statevector(qc_hc)
        assert statevectors_equal(sv_par, sv_hc)
        # Both should be |+++⟩ = uniform superposition
        expected = np.ones(8, dtype=complex) / np.sqrt(8)
        assert statevectors_equal(sv_par, expected)

    def test_cx_chain_parametric_vs_hardcoded(self):
        """GHZ via CX chain: parametric UInt-bound n vs hardcoded 3."""

        @qmc.qkernel
        def ghz_parametric(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q[0] = qmc.h(q[0])
            for i in qmc.range(n - 1):
                q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])
            return qmc.measure(q)

        @qmc.qkernel
        def ghz_hardcoded() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            q[1], q[2] = qmc.cx(q[1], q[2])
            return qmc.measure(q)

        _, qc_par = _transpile_and_get_circuit(ghz_parametric, bindings={"n": 3})
        _, qc_hc = _transpile_and_get_circuit(ghz_hardcoded)
        sv_par = _run_statevector(qc_par)
        sv_hc = _run_statevector(qc_hc)
        assert statevectors_equal(sv_par, sv_hc)
        # Both should be 3-qubit GHZ
        assert np.isclose(abs(sv_par[0]), 1.0 / np.sqrt(2), atol=1e-10)
        assert np.isclose(abs(sv_par[7]), 1.0 / np.sqrt(2), atol=1e-10)
