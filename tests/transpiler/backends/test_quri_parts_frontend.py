"""Rich QURI Parts frontend-to-backend test suite.

Tests the full pipeline: @qkernel definition -> QuriPartsTranspiler -> execution.
Covers every available frontend gate, gate combinations, transpiler passes,
parametric circuits, basic execution, algorithm modules, and edge cases.

Note: Do NOT use ``from __future__ import annotations`` in this file.
The @qkernel AST transformer relies on resolved type annotations to identify
Float vs UInt etc.  PEP 563 deferred annotations turn them into strings, which
breaks ``_create_bound_input``.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.quri_parts

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
# Skip entire module if quri-parts is not installed
# ---------------------------------------------------------------------------
pytest.importorskip("quri_parts")
pytest.importorskip("quri_parts.qulacs")

from quri_parts.circuit import gate_names
from quri_parts.core.operator import SinglePauli
from qamomile.circuit.transpiler.errors import EmitError

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
from qamomile.quri_parts import QuriPartsTranspiler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_statevector(circuit, parameter_bindings=None) -> np.ndarray:
    """Run a QURI Parts circuit and return the statevector via Qulacs.

    Handles both parametric (unbound) and bound circuits.
    If *parameter_bindings* is given, uses those values; otherwise binds
    with zeros for parametric circuits.
    """
    from quri_parts.core.state import GeneralCircuitQuantumState
    from quri_parts.qulacs.simulator import evaluate_state_to_vector

    if hasattr(circuit, "parameter_count") and circuit.parameter_count > 0:
        values = (
            parameter_bindings
            if parameter_bindings is not None
            else [0.0] * circuit.parameter_count
        )
        bound_circuit = circuit.bind_parameters(values)
    elif hasattr(circuit, "bind_parameters"):
        bound_circuit = circuit.bind_parameters([])
    else:
        bound_circuit = circuit

    circuit_state = GeneralCircuitQuantumState(bound_circuit.qubit_count, bound_circuit)
    statevector = evaluate_state_to_vector(circuit_state)
    return np.array(statevector.vector)


def _transpile_and_get_circuit(
    kernel, bindings=None, parameters=None, *, skip_smoke_test=False
):
    """Transpile a qkernel and return (ExecutableProgram, QURI Parts circuit).

    Also runs a smoke test to verify execution works:
    - sample(shots=1) for measurement circuits
    - run() for expval circuits
    Skipped when parameters != None (runtime bindings required)
    or skip_smoke_test=True (e.g., large qubit counts).
    """
    transpiler = QuriPartsTranspiler()
    exe = transpiler.transpile(kernel, bindings=bindings, parameters=parameters)
    circuit = exe.compiled_quantum[0].circuit
    # Smoke test: verify execution runs without error
    if parameters is None and not skip_smoke_test:
        executor = transpiler.executor()
        if exe.compiled_expval:
            exe.run(executor).result()
        else:
            exe.sample(executor, bindings={}, shots=1).result()
    return exe, circuit


def _get_gates(circuit, parameter_bindings=None) -> list:
    """Return list of gate objects from a QURI Parts circuit.

    Handles both parametric and non-parametric circuits.
    If *parameter_bindings* is given, uses those values; otherwise binds
    with zeros for parametric circuits.
    Returns QuantumGate objects with .name, .target_indices,
    .control_indices, and .params properties.
    """
    # Parametric circuits must be bound first to get concrete gates
    if hasattr(circuit, "bind_parameters"):
        if hasattr(circuit, "parameter_count") and circuit.parameter_count > 0:
            values = (
                parameter_bindings
                if parameter_bindings is not None
                else [0.0] * circuit.parameter_count
            )
            bound = circuit.bind_parameters(values)
            if hasattr(bound, "gates"):
                return list(bound.gates)
        else:
            bound = circuit.bind_parameters([])
            if hasattr(bound, "gates"):
                return list(bound.gates)
    if hasattr(circuit, "gates"):
        gates = list(circuit.gates)
        if gates:
            return gates
    return []


def _gate_names(circuit, parameter_bindings=None) -> list[str]:
    """Compatibility helper for tests that only care about gate names."""
    return [gate.name for gate in _get_gates(circuit, parameter_bindings)]


# ---------------------------------------------------------------------------
# Module-level qkernel definitions for parametrized tests
# ---------------------------------------------------------------------------

# CCX basis state kernels (shared by TestThreeQubitGatesFrontend and TestExecution)


@qmc.qkernel
def _ccx_000() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(3, "q")
    q[0], q[1], q[2] = qmc.ccx(q[0], q[1], q[2])
    return qmc.measure(q)


@qmc.qkernel
def _ccx_001() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.x(q[0])
    q[0], q[1], q[2] = qmc.ccx(q[0], q[1], q[2])
    return qmc.measure(q)


@qmc.qkernel
def _ccx_010() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(3, "q")
    q[1] = qmc.x(q[1])
    q[0], q[1], q[2] = qmc.ccx(q[0], q[1], q[2])
    return qmc.measure(q)


@qmc.qkernel
def _ccx_011() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.x(q[0])
    q[1] = qmc.x(q[1])
    q[0], q[1], q[2] = qmc.ccx(q[0], q[1], q[2])
    return qmc.measure(q)


@qmc.qkernel
def _ccx_100() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(3, "q")
    q[2] = qmc.x(q[2])
    q[0], q[1], q[2] = qmc.ccx(q[0], q[1], q[2])
    return qmc.measure(q)


@qmc.qkernel
def _ccx_101() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.x(q[0])
    q[2] = qmc.x(q[2])
    q[0], q[1], q[2] = qmc.ccx(q[0], q[1], q[2])
    return qmc.measure(q)


@qmc.qkernel
def _ccx_110() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(3, "q")
    q[1] = qmc.x(q[1])
    q[2] = qmc.x(q[2])
    q[0], q[1], q[2] = qmc.ccx(q[0], q[1], q[2])
    return qmc.measure(q)


@qmc.qkernel
def _ccx_111() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.x(q[0])
    q[1] = qmc.x(q[1])
    q[2] = qmc.x(q[2])
    q[0], q[1], q[2] = qmc.ccx(q[0], q[1], q[2])
    return qmc.measure(q)


# Single-qubit deterministic sampling kernels


@qmc.qkernel
def _circ_x() -> qmc.Bit:
    q = qmc.qubit("q")
    q = qmc.x(q)
    return qmc.measure(q)


@qmc.qkernel
def _circ_y() -> qmc.Bit:
    q = qmc.qubit("q")
    q = qmc.y(q)
    return qmc.measure(q)


@qmc.qkernel
def _circ_z() -> qmc.Bit:
    q = qmc.qubit("q")
    q = qmc.z(q)
    return qmc.measure(q)


@qmc.qkernel
def _circ_s() -> qmc.Bit:
    q = qmc.qubit("q")
    q = qmc.s(q)
    return qmc.measure(q)


@qmc.qkernel
def _circ_t() -> qmc.Bit:
    q = qmc.qubit("q")
    q = qmc.t(q)
    return qmc.measure(q)


@qmc.qkernel
def _circ_xx() -> qmc.Bit:
    q = qmc.qubit("q")
    q = qmc.x(q)
    q = qmc.x(q)
    return qmc.measure(q)


@qmc.qkernel
def _circ_xy() -> qmc.Bit:
    q = qmc.qubit("q")
    q = qmc.x(q)
    q = qmc.y(q)
    return qmc.measure(q)


@qmc.qkernel
def _circ_hxh() -> qmc.Bit:
    q = qmc.qubit("q")
    q = qmc.h(q)
    q = qmc.x(q)
    q = qmc.h(q)
    return qmc.measure(q)


# Two-qubit deterministic sampling kernels


@qmc.qkernel
def _cx_00() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q[0], q[1] = qmc.cx(q[0], q[1])
    return qmc.measure(q)


@qmc.qkernel
def _cx_10() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.x(q[0])
    q[0], q[1] = qmc.cx(q[0], q[1])
    return qmc.measure(q)


@qmc.qkernel
def _cx_01() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q[1] = qmc.x(q[1])
    q[0], q[1] = qmc.cx(q[0], q[1])
    return qmc.measure(q)


@qmc.qkernel
def _cx_11() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.x(q[0])
    q[1] = qmc.x(q[1])
    q[0], q[1] = qmc.cx(q[0], q[1])
    return qmc.measure(q)


@qmc.qkernel
def _cz_00() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q[0], q[1] = qmc.cz(q[0], q[1])
    return qmc.measure(q)


@qmc.qkernel
def _cz_10() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.x(q[0])
    q[0], q[1] = qmc.cz(q[0], q[1])
    return qmc.measure(q)


@qmc.qkernel
def _cz_01() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q[1] = qmc.x(q[1])
    q[0], q[1] = qmc.cz(q[0], q[1])
    return qmc.measure(q)


@qmc.qkernel
def _cz_11() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.x(q[0])
    q[1] = qmc.x(q[1])
    q[0], q[1] = qmc.cz(q[0], q[1])
    return qmc.measure(q)


@qmc.qkernel
def _swap_10() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.x(q[0])
    q[0], q[1] = qmc.swap(q[0], q[1])
    return qmc.measure(q)


@qmc.qkernel
def _swap_01() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q[1] = qmc.x(q[1])
    q[0], q[1] = qmc.swap(q[0], q[1])
    return qmc.measure(q)


# Gate combination deterministic sampling kernels


@qmc.qkernel
def _cx_chain_3() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.x(q[0])
    q[0], q[1] = qmc.cx(q[0], q[1])
    q[1], q[2] = qmc.cx(q[1], q[2])
    return qmc.measure(q)


@qmc.qkernel
def _swap_after_x() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.x(q[0])
    q[0], q[1] = qmc.swap(q[0], q[1])
    return qmc.measure(q)


@qmc.qkernel
def _x_cx_x() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.x(q[0])
    q[0], q[1] = qmc.cx(q[0], q[1])
    q[1] = qmc.x(q[1])
    return qmc.measure(q)


# ============================================================================
# 1. Individual Gate Tests
# ============================================================================


class TestSingleQubitGatesFrontend:
    """Test each single-qubit gate through the full frontend pipeline.

    Each test verifies: gate name (creation) + statevector + qubit_array equivalence.
    """

    # -- H (Hadamard) --

    def test_h(self):
        """H|0⟩ = (|0⟩+|1⟩)/√2: creation + statevector + qubit_array."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        gates = _get_gates(qc)
        assert len(gates) == 1
        assert gates[0].name == gate_names.H
        assert gates[0].target_indices == (0,)
        sv = _run_statevector(qc)
        expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
        assert statevectors_equal(sv, expected)

        @qmc.qkernel
        def circuit_arr() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, "q")
            q[0] = qmc.h(q[0])
            return qmc.measure(q)

        _, qc_arr = _transpile_and_get_circuit(circuit_arr)
        assert _get_gates(qc_arr) == gates
        assert statevectors_equal(_run_statevector(qc_arr), expected)

    # -- X (Pauli-X) --

    def test_x(self):
        """X|0⟩ = |1⟩: creation + statevector + qubit_array."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        gates = _get_gates(qc)
        assert len(gates) == 1
        assert gates[0].name == gate_names.X
        assert gates[0].target_indices == (0,)
        sv = _run_statevector(qc)
        expected = np.array([0, 1], dtype=complex)
        assert statevectors_equal(sv, expected)

        @qmc.qkernel
        def circuit_arr() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, "q")
            q[0] = qmc.x(q[0])
            return qmc.measure(q)

        _, qc_arr = _transpile_and_get_circuit(circuit_arr)
        assert _get_gates(qc_arr) == gates
        assert statevectors_equal(_run_statevector(qc_arr), expected)

    # -- Y (Pauli-Y) --

    def test_y_on_zero(self):
        """Y|0⟩ = i|1⟩: creation + statevector + qubit_array."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.y(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        gates = _get_gates(qc)
        assert len(gates) == 1
        assert gates[0].name == gate_names.Y
        assert gates[0].target_indices == (0,)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["Y"].matrix_fn()
        )
        assert statevectors_equal(_run_statevector(qc), expected)

        @qmc.qkernel
        def circuit_arr() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, "q")
            q[0] = qmc.y(q[0])
            return qmc.measure(q)

        _, qc_arr = _transpile_and_get_circuit(circuit_arr)
        assert _get_gates(qc_arr) == gates
        assert statevectors_equal(_run_statevector(qc_arr), expected)

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
        """Z|0⟩ = |0⟩: creation + statevector."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.z(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        gates = _get_gates(qc)
        assert len(gates) == 1
        assert gates[0].name == gate_names.Z
        assert gates[0].target_indices == (0,)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["Z"].matrix_fn()
        )
        assert statevectors_equal(_run_statevector(qc), expected)

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
        """Z|+⟩ = |−⟩: statevector + qubit_array."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            q = qmc.z(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        gates = _get_gates(qc)
        expected = minus_state()
        assert statevectors_equal(_run_statevector(qc), expected)

        @qmc.qkernel
        def circuit_arr() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, "q")
            q[0] = qmc.h(q[0])
            q[0] = qmc.z(q[0])
            return qmc.measure(q)

        _, qc_arr = _transpile_and_get_circuit(circuit_arr)
        assert _get_gates(qc_arr) == gates
        assert statevectors_equal(_run_statevector(qc_arr), expected)

    # -- S (sqrt(Z)) --

    def test_s_on_zero(self):
        """S|0⟩ = |0⟩: creation + statevector."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.s(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        gates = _get_gates(qc)
        assert len(gates) == 1
        assert gates[0].name == gate_names.S
        assert gates[0].target_indices == (0,)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["S"].matrix_fn()
        )
        assert statevectors_equal(_run_statevector(qc), expected)

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
        """S·S = Z on |+⟩: statevector + qubit_array."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            q = qmc.s(q)
            q = qmc.s(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        gates = _get_gates(qc)
        expected = minus_state()
        assert statevectors_equal(_run_statevector(qc), expected)

        @qmc.qkernel
        def circuit_arr() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, "q")
            q[0] = qmc.h(q[0])
            q[0] = qmc.s(q[0])
            q[0] = qmc.s(q[0])
            return qmc.measure(q)

        _, qc_arr = _transpile_and_get_circuit(circuit_arr)
        assert _get_gates(qc_arr) == gates
        assert statevectors_equal(_run_statevector(qc_arr), expected)

    # -- SDG (S-dagger, inverse of S) --

    def test_sdg_on_zero(self):
        """SDG|0⟩ = |0⟩: creation + statevector."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.sdg(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        gates = _get_gates(qc)
        assert len(gates) == 1
        assert gates[0].name == gate_names.Sdag
        assert gates[0].target_indices == (0,)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["SDG"].matrix_fn()
        )
        assert statevectors_equal(_run_statevector(qc), expected)

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
        """S then SDG = Identity on |1⟩: statevector + qubit_array."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            q = qmc.s(q)
            q = qmc.sdg(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        gates = _get_gates(qc)
        expected = np.array([0, 1], dtype=complex)
        assert statevectors_equal(_run_statevector(qc), expected)

        @qmc.qkernel
        def circuit_arr() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, "q")
            q[0] = qmc.x(q[0])
            q[0] = qmc.s(q[0])
            q[0] = qmc.sdg(q[0])
            return qmc.measure(q)

        _, qc_arr = _transpile_and_get_circuit(circuit_arr)
        assert _get_gates(qc_arr) == gates
        assert statevectors_equal(_run_statevector(qc_arr), expected)

    # -- T (sqrt(S)) --

    def test_t_on_zero(self):
        """T|0⟩ = |0⟩: creation + statevector."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.t(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        gates = _get_gates(qc)
        assert len(gates) == 1
        assert gates[0].name == gate_names.T
        assert gates[0].target_indices == (0,)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["T"].matrix_fn()
        )
        assert statevectors_equal(_run_statevector(qc), expected)

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
        """T·T = S on |1⟩: statevector + qubit_array."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            q = qmc.t(q)
            q = qmc.t(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        gates = _get_gates(qc)
        state_after_x = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["X"].matrix_fn()
        )
        expected = compute_expected_statevector(
            state_after_x, GATE_SPECS["S"].matrix_fn()
        )
        assert statevectors_equal(_run_statevector(qc), expected)

        @qmc.qkernel
        def circuit_arr() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, "q")
            q[0] = qmc.x(q[0])
            q[0] = qmc.t(q[0])
            q[0] = qmc.t(q[0])
            return qmc.measure(q)

        _, qc_arr = _transpile_and_get_circuit(circuit_arr)
        assert _get_gates(qc_arr) == gates
        assert statevectors_equal(_run_statevector(qc_arr), expected)

    # -- TDG (T-dagger, inverse of T) --

    def test_tdg_on_zero(self):
        """TDG|0⟩ = |0⟩: creation + statevector."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.tdg(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        gates = _get_gates(qc)
        assert len(gates) == 1
        assert gates[0].name == gate_names.Tdag
        assert gates[0].target_indices == (0,)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["TDG"].matrix_fn()
        )
        assert statevectors_equal(_run_statevector(qc), expected)

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
        """T then TDG = Identity on |1⟩: statevector + qubit_array."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            q = qmc.t(q)
            q = qmc.tdg(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        gates = _get_gates(qc)
        expected = np.array([0, 1], dtype=complex)
        assert statevectors_equal(_run_statevector(qc), expected)

        @qmc.qkernel
        def circuit_arr() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, "q")
            q[0] = qmc.x(q[0])
            q[0] = qmc.t(q[0])
            q[0] = qmc.tdg(q[0])
            return qmc.measure(q)

        _, qc_arr = _transpile_and_get_circuit(circuit_arr)
        assert _get_gates(qc_arr) == gates
        assert statevectors_equal(_run_statevector(qc_arr), expected)

    # -- Parametric gates (RX, RY, RZ, P) --

    @pytest.mark.parametrize(
        "angle",
        [np.pi]
        + [
            np.random.default_rng(s).uniform(0, 2 * np.pi)
            for s in [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
        ],
    )
    def test_rx_statevector(self, angle):
        """RX(angle)|0⟩: creation + statevector + qubit_array (π only)."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": angle})
        gates = _get_gates(qc)
        assert len(gates) == 1
        assert gates[0].name == gate_names.RX
        assert gates[0].target_indices == (0,)
        assert np.isclose(gates[0].params[0], angle, atol=1e-10)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RX"].matrix_fn(angle)
        )
        assert statevectors_equal(_run_statevector(qc), expected)

        if np.isclose(angle, np.pi):

            @qmc.qkernel
            def circuit_arr(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
                q = qmc.qubit_array(1, "q")
                q[0] = qmc.rx(q[0], theta)
                return qmc.measure(q)

            _, qc_arr = _transpile_and_get_circuit(
                circuit_arr, bindings={"theta": angle}
            )
            assert _get_gates(qc_arr) == gates
            assert statevectors_equal(_run_statevector(qc_arr), expected)

    @pytest.mark.parametrize(
        "angle",
        [np.pi / 2]
        + [
            np.random.default_rng(s).uniform(0, 2 * np.pi)
            for s in [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
        ],
    )
    def test_ry_statevector(self, angle):
        """RY(angle)|0⟩: creation + statevector + qubit_array (π/2 only)."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.ry(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": angle})
        gates = _get_gates(qc)
        assert len(gates) == 1
        assert gates[0].name == gate_names.RY
        assert gates[0].target_indices == (0,)
        assert np.isclose(gates[0].params[0], angle, atol=1e-10)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RY"].matrix_fn(angle)
        )
        assert statevectors_equal(_run_statevector(qc), expected)

        if np.isclose(angle, np.pi / 2):

            @qmc.qkernel
            def circuit_arr(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
                q = qmc.qubit_array(1, "q")
                q[0] = qmc.ry(q[0], theta)
                return qmc.measure(q)

            _, qc_arr = _transpile_and_get_circuit(
                circuit_arr, bindings={"theta": angle}
            )
            assert _get_gates(qc_arr) == gates
            assert statevectors_equal(_run_statevector(qc_arr), expected)

    @pytest.mark.parametrize(
        "angle",
        [np.pi]
        + [
            np.random.default_rng(s).uniform(0, 2 * np.pi)
            for s in [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
        ],
    )
    def test_rz_statevector(self, angle):
        """RZ(angle)|0⟩: creation + statevector + qubit_array (π only)."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rz(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": angle})
        gates = _get_gates(qc)
        assert len(gates) == 1
        assert gates[0].name == gate_names.RZ
        assert gates[0].target_indices == (0,)
        assert np.isclose(gates[0].params[0], angle, atol=1e-10)
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RZ"].matrix_fn(angle)
        )
        assert statevectors_equal(_run_statevector(qc), expected)

        if np.isclose(angle, np.pi):

            @qmc.qkernel
            def circuit_arr(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
                q = qmc.qubit_array(1, "q")
                q[0] = qmc.rz(q[0], theta)
                return qmc.measure(q)

            _, qc_arr = _transpile_and_get_circuit(
                circuit_arr, bindings={"theta": angle}
            )
            assert _get_gates(qc_arr) == gates
            assert statevectors_equal(_run_statevector(qc_arr), expected)

    @pytest.mark.parametrize(
        "angle",
        [np.pi]
        + [
            np.random.default_rng(s).uniform(0, 2 * np.pi)
            for s in [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
        ],
    )
    def test_p_statevector(self, angle):
        """P(angle) on |1⟩: creation + statevector + qubit_array (π only)."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            q = qmc.p(q, theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": angle})
        gates = _get_gates(qc)
        assert gates[-1].name == gate_names.U1
        assert np.isclose(gates[-1].params[0], angle, atol=1e-10)
        state_after_x = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["X"].matrix_fn()
        )
        expected = compute_expected_statevector(
            state_after_x, GATE_SPECS["P"].matrix_fn(angle)
        )
        assert statevectors_equal(_run_statevector(qc), expected)

        if np.isclose(angle, np.pi):

            @qmc.qkernel
            def circuit_arr(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
                q = qmc.qubit_array(1, "q")
                q[0] = qmc.x(q[0])
                q[0] = qmc.p(q[0], theta)
                return qmc.measure(q)

            _, qc_arr = _transpile_and_get_circuit(
                circuit_arr, bindings={"theta": angle}
            )
            assert _get_gates(qc_arr) == gates
            assert statevectors_equal(_run_statevector(qc_arr), expected)


class TestTwoQubitGatesFrontend:
    """Test each two-qubit gate through the full frontend pipeline.

    Each test verifies: creation (gate name) + statevector + individual qubit() equivalence
    where applicable.
    """

    # -- CX (CNOT) --

    def test_cx_from_00(self):
        """CX |00⟩ = |00⟩: creation + statevector."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        gates = _get_gates(qc)
        assert len(gates) == 1
        assert gates[0].name == gate_names.CNOT
        assert gates[0].target_indices == (1,)
        assert gates[0].control_indices == (0,)
        expected = compute_expected_statevector(
            computational_basis_state(2, 0), GATE_SPECS["CX"].matrix_fn()
        )
        assert statevectors_equal(_run_statevector(qc), expected)

    def test_cx_from_10(self):
        """CX |10⟩ = |11⟩: statevector + individual qubit() equivalence."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        gates = _get_gates(qc)
        expected = compute_expected_statevector(
            computational_basis_state(2, 1), GATE_SPECS["CX"].matrix_fn()
        )
        assert statevectors_equal(_run_statevector(qc), expected)

        @qmc.qkernel
        def circuit_ind() -> tuple[qmc.Bit, qmc.Bit]:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q0 = qmc.x(q0)
            q0, q1 = qmc.cx(q0, q1)
            return qmc.measure(q0), qmc.measure(q1)

        _, qc_ind = _transpile_and_get_circuit(circuit_ind)
        assert [g.name for g in _get_gates(qc_ind)] == [g.name for g in gates]
        assert statevectors_equal(_run_statevector(qc_ind), expected)

    def test_cx_from_01(self):
        """CX |01⟩ = |01⟩ (control off)."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[1] = qmc.x(q[1])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        expected = compute_expected_statevector(
            computational_basis_state(2, 2), GATE_SPECS["CX"].matrix_fn()
        )
        assert statevectors_equal(_run_statevector(qc), expected)

    def test_cx_from_11(self):
        """CX |11⟩ = |10⟩."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        expected = compute_expected_statevector(
            computational_basis_state(2, 3), GATE_SPECS["CX"].matrix_fn()
        )
        assert statevectors_equal(_run_statevector(qc), expected)

    # -- CZ --

    def test_cz_from_00(self):
        """CZ |00⟩ = |00⟩: creation + statevector."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = qmc.cz(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        gates = _get_gates(qc)
        assert len(gates) == 1
        assert gates[0].name == gate_names.CZ
        assert gates[0].target_indices == (1,)
        assert gates[0].control_indices == (0,)
        expected = compute_expected_statevector(
            computational_basis_state(2, 0), GATE_SPECS["CZ"].matrix_fn()
        )
        assert statevectors_equal(_run_statevector(qc), expected)

    def test_cz_from_01(self):
        """CZ |01⟩ = |01⟩ (no phase, only q0=1)."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[0], q[1] = qmc.cz(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        expected = compute_expected_statevector(
            computational_basis_state(2, 1), GATE_SPECS["CZ"].matrix_fn()
        )
        assert statevectors_equal(_run_statevector(qc), expected)

    def test_cz_from_10(self):
        """CZ |10⟩ = |10⟩ (no phase, only q1=1)."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[1] = qmc.x(q[1])
            q[0], q[1] = qmc.cz(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        expected = compute_expected_statevector(
            computational_basis_state(2, 2), GATE_SPECS["CZ"].matrix_fn()
        )
        assert statevectors_equal(_run_statevector(qc), expected)

    def test_cz_determined_11(self):
        """CZ |11⟩ = -|11⟩: statevector + individual qubit() equivalence."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0], q[1] = qmc.cz(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        gates = _get_gates(qc)
        expected = compute_expected_statevector(
            computational_basis_state(2, 3), GATE_SPECS["CZ"].matrix_fn()
        )
        assert statevectors_equal(_run_statevector(qc), expected)

        @qmc.qkernel
        def circuit_ind() -> tuple[qmc.Bit, qmc.Bit]:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q0 = qmc.x(q0)
            q1 = qmc.x(q1)
            q0, q1 = qmc.cz(q0, q1)
            return qmc.measure(q0), qmc.measure(q1)

        _, qc_ind = _transpile_and_get_circuit(circuit_ind)
        assert [g.name for g in _get_gates(qc_ind)] == [g.name for g in gates]
        assert statevectors_equal(_run_statevector(qc_ind), expected)

    # -- CP --

    @pytest.mark.parametrize(
        "angle",
        [np.pi]
        + [
            np.random.default_rng(s).uniform(0, 2 * np.pi)
            for s in [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
        ],
    )
    def test_cp_statevector(self, angle):
        """CP(angle) on |11⟩: creation (π/4) + statevector + individual qubit() (π only)."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0], q[1] = qmc.cp(q[0], q[1], theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": angle})
        # CP decomposes to RZ + CNOT sequence
        gates = _get_gates(qc)
        assert len(gates) > 0
        expected = compute_expected_statevector(
            computational_basis_state(2, 3), GATE_SPECS["CP"].matrix_fn(angle)
        )
        assert statevectors_equal(_run_statevector(qc), expected)

        if np.isclose(angle, np.pi):

            @qmc.qkernel
            def circuit_ind(theta: qmc.Float) -> tuple[qmc.Bit, qmc.Bit]:
                q0 = qmc.qubit("q0")
                q1 = qmc.qubit("q1")
                q0 = qmc.x(q0)
                q1 = qmc.x(q1)
                q0, q1 = qmc.cp(q0, q1, theta)
                return qmc.measure(q0), qmc.measure(q1)

            _, qc_ind = _transpile_and_get_circuit(
                circuit_ind, bindings={"theta": angle}
            )
            assert [g.name for g in _get_gates(qc_ind)] == [g.name for g in gates]
            assert statevectors_equal(_run_statevector(qc_ind), expected)

    # -- RZZ --

    @pytest.mark.parametrize(
        "angle",
        [np.pi]
        + [
            np.random.default_rng(s).uniform(0, 2 * np.pi)
            for s in [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999]
        ],
    )
    def test_rzz_statevector(self, angle):
        """RZZ(angle) on |11⟩: creation + statevector + individual qubit() (π only)."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0], q[1] = qmc.rzz(q[0], q[1], theta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"theta": angle})
        gates = _get_gates(qc)
        rzz_gates = [g for g in gates if g.name == gate_names.PauliRotation]
        assert len(rzz_gates) >= 1
        assert rzz_gates[0].pauli_ids == (SinglePauli.Z, SinglePauli.Z)
        expected = compute_expected_statevector(
            computational_basis_state(2, 3), GATE_SPECS["RZZ"].matrix_fn(angle)
        )
        assert statevectors_equal(_run_statevector(qc), expected)

        if np.isclose(angle, np.pi):

            @qmc.qkernel
            def circuit_ind(theta: qmc.Float) -> tuple[qmc.Bit, qmc.Bit]:
                q0 = qmc.qubit("q0")
                q1 = qmc.qubit("q1")
                q0 = qmc.x(q0)
                q1 = qmc.x(q1)
                q0, q1 = qmc.rzz(q0, q1, theta)
                return qmc.measure(q0), qmc.measure(q1)

            _, qc_ind = _transpile_and_get_circuit(
                circuit_ind, bindings={"theta": angle}
            )
            assert [g.name for g in _get_gates(qc_ind)] == [g.name for g in gates]
            assert statevectors_equal(_run_statevector(qc_ind), expected)

    # -- SWAP --

    def test_swap_determined(self):
        """SWAP |10⟩ = |01⟩: creation + statevector + individual qubit() equivalence."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[0], q[1] = qmc.swap(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        gates = _get_gates(qc)
        swap_gates = [g for g in gates if g.name == gate_names.SWAP]
        assert len(swap_gates) == 1
        assert swap_gates[0].target_indices == (0, 1)
        expected = compute_expected_statevector(
            computational_basis_state(2, 1), GATE_SPECS["SWAP"].matrix_fn()
        )
        assert statevectors_equal(_run_statevector(qc), expected)

        @qmc.qkernel
        def circuit_ind() -> tuple[qmc.Bit, qmc.Bit]:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q0 = qmc.x(q0)
            q0, q1 = qmc.swap(q0, q1)
            return qmc.measure(q0), qmc.measure(q1)

        _, qc_ind = _transpile_and_get_circuit(circuit_ind)
        assert [g.name for g in _get_gates(qc_ind)] == [g.name for g in gates]
        assert statevectors_equal(_run_statevector(qc_ind), expected)


class TestThreeQubitGatesFrontend:
    """Test each three-qubit gate through the full frontend pipeline."""

    @pytest.mark.parametrize(
        "kernel, basis_idx",
        [
            pytest.param(_ccx_000, 0, id="000"),
            pytest.param(_ccx_001, 1, id="001"),
            pytest.param(_ccx_010, 2, id="010"),
            pytest.param(_ccx_011, 3, id="011_flips"),
            pytest.param(_ccx_100, 4, id="100"),
            pytest.param(_ccx_101, 5, id="101"),
            pytest.param(_ccx_110, 6, id="110"),
            pytest.param(_ccx_111, 7, id="111_flips"),
        ],
    )
    def test_ccx_basis_state(self, kernel, basis_idx):
        """CCX on all 8 basis states; flips target only when both controls |1⟩."""
        _, qc = _transpile_and_get_circuit(kernel)
        sv = _run_statevector(qc)
        expected = compute_expected_statevector(
            computational_basis_state(3, basis_idx),
            GATE_SPECS["TOFFOLI"].matrix_fn(),
        )
        assert statevectors_equal(sv, expected)

    def test_ccx_creation(self):
        """CCX gate transpiles to circuit containing 'TOFFOLI' gate."""
        _, qc = _transpile_and_get_circuit(_ccx_000)
        gates = _get_gates(qc)
        assert len(gates) == 1
        assert gates[0].name == gate_names.TOFFOLI
        assert gates[0].target_indices == (2,)
        assert gates[0].control_indices == (0, 1)

    def test_ccx_individual_qubit_equivalence(self):
        """CCX |011⟩ via individual qubit() matches qubit_array version."""

        @qmc.qkernel
        def circuit_ind() -> tuple[qmc.Bit, qmc.Bit, qmc.Bit]:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q2 = qmc.qubit("q2")
            q0 = qmc.x(q0)
            q1 = qmc.x(q1)
            q0, q1, q2 = qmc.ccx(q0, q1, q2)
            return qmc.measure(q0), qmc.measure(q1), qmc.measure(q2)

        _, qc_arr = _transpile_and_get_circuit(_ccx_011)
        _, qc_ind = _transpile_and_get_circuit(circuit_ind)
        expected = compute_expected_statevector(
            computational_basis_state(3, 3),
            GATE_SPECS["TOFFOLI"].matrix_fn(),
        )
        assert [g.name for g in _get_gates(qc_ind)] == [
            g.name for g in _get_gates(qc_arr)
        ]
        assert statevectors_equal(_run_statevector(qc_ind), expected)


# ============================================================================
# 2. Gate Combination Tests
# ============================================================================


class TestGateCombinations:
    """Test multi-gate circuits through the frontend pipeline."""

    def test_bell_state(self):
        """H + CX creates Bell state: qubit_array + individual qubit() equivalence."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        gates = _get_gates(qc)
        expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        assert statevectors_equal(_run_statevector(qc), expected)

        @qmc.qkernel
        def circuit_ind() -> tuple[qmc.Bit, qmc.Bit]:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q0 = qmc.h(q0)
            q0, q1 = qmc.cx(q0, q1)
            return qmc.measure(q0), qmc.measure(q1)

        _, qc_ind = _transpile_and_get_circuit(circuit_ind)
        assert [g.name for g in _get_gates(qc_ind)] == [g.name for g in gates]
        assert statevectors_equal(_run_statevector(qc_ind), expected)

    def test_ghz_state_3(self):
        """GHZ state: qubit_array + individual qubit() equivalence."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            q[1], q[2] = qmc.cx(q[1], q[2])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        gates = _get_gates(qc)
        expected = np.zeros(8, dtype=complex)
        expected[0] = 1 / np.sqrt(2)
        expected[7] = 1 / np.sqrt(2)
        assert statevectors_equal(_run_statevector(qc), expected)

        @qmc.qkernel
        def circuit_ind() -> tuple[qmc.Bit, qmc.Bit, qmc.Bit]:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q2 = qmc.qubit("q2")
            q0 = qmc.h(q0)
            q0, q1 = qmc.cx(q0, q1)
            q1, q2 = qmc.cx(q1, q2)
            return qmc.measure(q0), qmc.measure(q1), qmc.measure(q2)

        _, qc_ind = _transpile_and_get_circuit(circuit_ind)
        assert [g.name for g in _get_gates(qc_ind)] == [g.name for g in gates]
        assert statevectors_equal(_run_statevector(qc_ind), expected)

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
        """Prepare |1> on q0, SWAP: qubit_array + individual qubit() equivalence."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[0], q[1] = qmc.swap(q[0], q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit)
        gates = _get_gates(qc)
        expected = np.array([0, 0, 1, 0], dtype=complex)
        assert statevectors_equal(_run_statevector(qc), expected)

        @qmc.qkernel
        def circuit_ind() -> tuple[qmc.Bit, qmc.Bit]:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q0 = qmc.x(q0)
            q0, q1 = qmc.swap(q0, q1)
            return qmc.measure(q0), qmc.measure(q1)

        _, qc_ind = _transpile_and_get_circuit(circuit_ind)
        assert [g.name for g in _get_gates(qc_ind)] == [g.name for g in gates]
        assert statevectors_equal(_run_statevector(qc_ind), expected)

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
        Note: Full teleportation requires two sequential conditional corrections
        (``if m1: X(bell1); if m0: Z(bell1)``). A single ``if`` correction
        transpiles successfully, but the second ``if`` sees ``bell1`` as a
        phi-node (SSA merge from the first ``if`` branch), causing the
        analyzer to raise ``DependencyError`` because it treats the merged
        value as measurement-dependent.
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
        gates = _get_gates(qc)
        assert qc.qubit_count >= 3
        # Teleportation structure: X(alice), H(bell0), CNOT(bell0→bell1),
        # CNOT(alice→bell0), H(alice)
        assert len(gates) == 5
        alice_q = gates[0].target_indices[0]
        bell0_q = gates[1].target_indices[0]
        bell1_q = gates[2].target_indices[0]
        assert gates[0].name == gate_names.X and gates[0].target_indices == (alice_q,)
        assert gates[1].name == gate_names.H and gates[1].target_indices == (bell0_q,)
        assert (
            gates[2].name == gate_names.CNOT
            and gates[2].control_indices == (bell0_q,)
            and gates[2].target_indices == (bell1_q,)
        )
        assert (
            gates[3].name == gate_names.CNOT
            and gates[3].control_indices == (alice_q,)
            and gates[3].target_indices == (bell0_q,)
        )
        assert gates[4].name == gate_names.H and gates[4].target_indices == (alice_q,)
        # All three qubits are distinct
        assert len({alice_q, bell0_q, bell1_q}) == 3

        # Also verify qubit_array(3) version transpiles equivalently
        @qmc.qkernel
        def circuit_arr() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.h(q[1])
            q[1], q[2] = qmc.cx(q[1], q[2])
            q[0], q[1] = qmc.cx(q[0], q[1])
            q[0] = qmc.h(q[0])
            return qmc.measure(q)

        _, qc_arr = _transpile_and_get_circuit(circuit_arr)
        assert qc_arr.qubit_count == 3
        gates_arr = _get_gates(qc_arr)
        assert [g.name for g in gates_arr] == [g.name for g in gates]


# ============================================================================
# 3. Control Flow Tests
# ============================================================================


class TestControlFlowRange:
    """Test qmc.range loop through the QuriParts frontend pipeline.

    qmc.range() is trace-time unrolled (Python-level loop), NOT circuit-level
    control flow.  These tests therefore work identically on any backend.
    """

    @pytest.mark.parametrize("n_qubits", [2, 3, 5, 10, 20])
    def test_h_all_qubits_via_range(self, n_qubits):
        """Apply H to all qubits via qmc.range loop."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        _, circ = _transpile_and_get_circuit(circuit, bindings={"n": n_qubits})
        gates = _get_gates(circ)
        h_gates = [g for g in gates if g.name == gate_names.H]
        assert len(h_gates) == n_qubits
        for i, g in enumerate(h_gates):
            assert g.target_indices == (i,)

    @pytest.mark.parametrize("n_qubits", [2, 3, 5])
    def test_rx_layer_via_range(self, n_qubits):
        """Apply RX with varying angles via qmc.range — check gate count."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.rx(q[i], thetas[i])
            return qmc.measure(q)

        thetas = [0.1 * i for i in range(n_qubits)]
        _, circ = _transpile_and_get_circuit(
            circuit, bindings={"n": n_qubits, "thetas": thetas}
        )
        gates = _get_gates(circ)
        rx_gates = [g for g in gates if g.name == gate_names.RX]
        assert len(rx_gates) == n_qubits
        for i, g in enumerate(rx_gates):
            assert g.target_indices == (i,)

    @pytest.mark.parametrize("n_qubits", [4, 5, 6])
    def test_range_with_start_stop(self, n_qubits):
        """qmc.range(start, stop) applies gates to subset of qubits."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(1, 3):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        _, circ = _transpile_and_get_circuit(circuit, bindings={"n": n_qubits})
        gates = _get_gates(circ)
        h_gates = [g for g in gates if g.name == gate_names.H]
        assert len(h_gates) == 2
        assert h_gates[0].target_indices == (1,)
        assert h_gates[1].target_indices == (2,)

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

        _, circ = _transpile_and_get_circuit(circuit, bindings={"n": n_qubits})
        gates = _get_gates(circ)
        cz_gates = [g for g in gates if g.name == gate_names.CZ]
        assert len(cz_gates) == n_qubits - 1
        for i, g in enumerate(cz_gates):
            assert g.target_indices == (i + 1,)
            assert g.control_indices == (i,)

    @pytest.mark.parametrize("n_qubits", [4, 6])
    def test_range_with_step_2(self, n_qubits):
        """range(0, n, 2) applies gates to even-indexed qubits only."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(0, n, 2):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        _, circ = _transpile_and_get_circuit(circuit, bindings={"n": n_qubits})
        gates = _get_gates(circ)
        h_gates = [g for g in gates if g.name == gate_names.H]
        expected = (n_qubits + 1) // 2  # even indices: 0, 2, 4, ...
        assert len(h_gates) == expected
        for idx, g in enumerate(h_gates):
            assert g.target_indices == (idx * 2,)

    @pytest.mark.parametrize("n_qubits", [4, 6])
    def test_range_with_step_2_odd_start(self, n_qubits):
        """range(1, n, 2) applies gates to odd-indexed qubits only."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(1, n, 2):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        _, circ = _transpile_and_get_circuit(circuit, bindings={"n": n_qubits})
        gates = _get_gates(circ)
        h_gates = [g for g in gates if g.name == gate_names.H]
        expected = n_qubits // 2  # odd indices: 1, 3, 5, ...
        assert len(h_gates) == expected
        for idx, g in enumerate(h_gates):
            assert g.target_indices == (idx * 2 + 1,)

    @pytest.mark.parametrize("start,stop", [(5, 5), (3, 2)])
    def test_zero_iteration_range(self, start, stop):
        """range(start, stop) with start >= stop produces no gates."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(start, stop):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        _, circ = _transpile_and_get_circuit(circuit, bindings={"n": 6})
        gates = _get_gates(circ)
        h_gates = [g for g in gates if g.name == gate_names.H]
        assert len(h_gates) == 0

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
        _, circ = _transpile_and_get_circuit(
            circuit, bindings={"n": n, "thetas": thetas}
        )
        gates = _get_gates(circ)
        h_gates = [g for g in gates if g.name == gate_names.H]
        rz_gates = [g for g in gates if g.name == gate_names.RZ]
        assert len(h_gates) == n
        assert len(rz_gates) == n
        for i, g in enumerate(h_gates):
            assert g.target_indices == (i,)
        for i, g in enumerate(rz_gates):
            assert g.target_indices == (i,)

    def test_range_with_expression_args(self):
        """range(1, n-1) with arithmetic in bounds."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(1, n - 1):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        n = 5
        _, circ = _transpile_and_get_circuit(circuit, bindings={"n": n})
        gates = _get_gates(circ)
        h_gates = [g for g in gates if g.name == gate_names.H]
        assert len(h_gates) == n - 2  # indices 1, 2, 3
        for idx, g in enumerate(h_gates):
            assert g.target_indices == (idx + 1,)

    # -- Loop array vs manual qubit() equivalence tests --

    def test_h_loop_array_vs_manual_qubits(self):
        """H loop on qubit_array matches manual qubit() x3."""

        @qmc.qkernel
        def loop_circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        @qmc.qkernel
        def manual_circuit() -> tuple[qmc.Bit, qmc.Bit, qmc.Bit]:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q2 = qmc.qubit("q2")
            q0 = qmc.h(q0)
            q1 = qmc.h(q1)
            q2 = qmc.h(q2)
            return qmc.measure(q0), qmc.measure(q1), qmc.measure(q2)

        _, circ_loop = _transpile_and_get_circuit(loop_circuit, bindings={"n": 3})
        _, circ_manual = _transpile_and_get_circuit(manual_circuit)
        sv_loop = _run_statevector(circ_loop)
        sv_manual = _run_statevector(circ_manual)
        assert statevectors_equal(sv_loop, sv_manual)
        # Both should be |+++> = uniform superposition
        expected = np.ones(8, dtype=complex) / np.sqrt(8)
        assert statevectors_equal(sv_loop, expected)

    def test_rx_loop_array_vs_manual_qubits(self):
        """RX loop on qubit_array matches manual qubit() x2."""

        @qmc.qkernel
        def loop_circuit(
            n: qmc.UInt, thetas: qmc.Vector[qmc.Float]
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.rx(q[i], thetas[i])
            return qmc.measure(q)

        @qmc.qkernel
        def manual_circuit(t0: qmc.Float, t1: qmc.Float) -> tuple[qmc.Bit, qmc.Bit]:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q0 = qmc.rx(q0, t0)
            q1 = qmc.rx(q1, t1)
            return qmc.measure(q0), qmc.measure(q1)

        t0, t1 = np.pi / 3, np.pi / 5
        _, circ_loop = _transpile_and_get_circuit(
            loop_circuit, bindings={"n": 2, "thetas": [t0, t1]}
        )
        _, circ_manual = _transpile_and_get_circuit(
            manual_circuit, bindings={"t0": t0, "t1": t1}
        )
        sv_loop = _run_statevector(circ_loop)
        sv_manual = _run_statevector(circ_manual)
        assert statevectors_equal(sv_loop, sv_manual)

    # -- Execution (sampling) verification for range loops --

    @pytest.mark.parametrize("n_qubits", [2, 3, 5])
    def test_range_x_loop_deterministic_sampling(self, n_qubits):
        """X on all qubits via range loop: sampling returns all 1s."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.x(q[i])
            return qmc.measure(q)

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(circuit, bindings={"n": n_qubits})
        executor = transpiler.executor()
        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        expected = tuple(1 for _ in range(n_qubits))
        for value, _ in result.results:
            assert value == expected

    def test_range_cx_chain_deterministic_sampling(self):
        """CX chain via range loop: X(q[0]) propagates to all qubits."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q[0] = qmc.x(q[0])
            for i in qmc.range(n - 1):
                q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])
            return qmc.measure(q)

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(circuit, bindings={"n": 4})
        executor = transpiler.executor()
        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        for value, _ in result.results:
            assert value == (1, 1, 1, 1)


class TestControlFlowItems:
    """Test qmc.items loop through the QuriParts frontend pipeline.

    qmc.items() is trace-time unrolled, so it works with any backend.
    """

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

        _, circ = _transpile_and_get_circuit(
            circuit, bindings={"n": n_qubits, "ising_dict": ising, "gamma": 0.5}
        )
        # QuriParts emits RZZ as PauliRotation
        gates = _get_gates(circ)
        rzz_gates = [g for g in gates if g.name == gate_names.PauliRotation]
        assert len(rzz_gates) == len(ising)
        for g in rzz_gates:
            assert g.pauli_ids == (SinglePauli.Z, SinglePauli.Z)  # ZZ

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
        _, circ = _transpile_and_get_circuit(
            circuit, bindings={"n": 3, "angles": angles}
        )
        gates = _get_gates(circ)
        rz_gates = [g for g in gates if g.name == gate_names.RZ]
        assert len(rz_gates) == 3
        for i, g in enumerate(rz_gates):
            assert g.target_indices == (i,)

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

        _, circ = _transpile_and_get_circuit(circuit, bindings={"n": 2, "ising": {}})
        gates = _get_gates(circ)
        rzz_gates = [g for g in gates if g.name == gate_names.PauliRotation]
        assert len(rzz_gates) == 0

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
        _, circ = _transpile_and_get_circuit(circuit, bindings={"n": 5, "ising": ising})
        gates = _get_gates(circ)
        rzz_gates = [g for g in gates if g.name == gate_names.PauliRotation]
        assert len(rzz_gates) == 10
        for g in rzz_gates:
            assert g.pauli_ids == (SinglePauli.Z, SinglePauli.Z)  # ZZ

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
        _, circ = _transpile_and_get_circuit(circuit, bindings={"n": 2, "ising": ising})
        sv = _run_statevector(circ)
        # RZZ(theta) on |00>: apply RZZ matrix to all-zeros state
        expected = GATE_SPECS["RZZ"].matrix_fn(theta) @ all_zeros_state(2)
        assert statevectors_equal(sv, expected)

    def test_items_negative_values_angle_sign(self):
        """Negative Jij produces correctly signed RZZ angle (statevector check).

        Uses H|0⟩⊗H|0⟩ initial state so RZZ(±θ) produces distinguishable
        states (on |00⟩ alone they only differ by global phase).
        """

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

        gamma = 0.5
        # Test with negative coupling
        ising_neg = {(0, 1): -1.0}
        _, circ_neg = _transpile_and_get_circuit(
            circuit, bindings={"n": 2, "ising": ising_neg, "gamma": gamma}
        )
        sv_neg = _run_statevector(circ_neg)

        # Test with positive coupling
        ising_pos = {(0, 1): 1.0}
        _, circ_pos = _transpile_and_get_circuit(
            circuit, bindings={"n": 2, "ising": ising_pos, "gamma": gamma}
        )
        sv_pos = _run_statevector(circ_pos)

        # On superposition state, positive and negative angles produce different states
        assert not statevectors_equal(sv_neg, sv_pos)

        # Each should match the analytically computed value:
        # H⊗H|00⟩ → |++⟩, then RZZ(angle)
        H = GATE_SPECS["H"].matrix_fn()
        hh = tensor_product(H, H)
        init_state = hh @ all_zeros_state(2)
        expected_neg = GATE_SPECS["RZZ"].matrix_fn(gamma * -1.0) @ init_state
        expected_pos = GATE_SPECS["RZZ"].matrix_fn(gamma * 1.0) @ init_state
        assert statevectors_equal(sv_neg, expected_neg)
        assert statevectors_equal(sv_pos, expected_pos)

    # -- Execution (sampling) verification for items loops --

    def test_items_x_from_dict_deterministic_sampling(self):
        """Apply X to dict-specified qubits: sampling returns expected bits."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
            targets: qmc.Dict[qmc.UInt, qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i, _val in qmc.items(targets):
                q[i] = qmc.x(q[i])
            return qmc.measure(q)

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(
            circuit, bindings={"n": 4, "targets": {0: 1.0, 2: 1.0}}
        )
        executor = transpiler.executor()
        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        for value, _ in result.results:
            assert value == (1, 0, 1, 0)


class TestControlFlowNested:
    """Test nested control flow patterns (all trace-time unrolled)."""

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
        _, circ = _transpile_and_get_circuit(circuit, bindings={"n": n})
        # n*(n-1)/2 = 3 CZ gates
        gates = _get_gates(circ)
        cz_gates = [g for g in gates if g.name == gate_names.CZ]
        assert len(cz_gates) == n * (n - 1) // 2
        expected_pairs = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
        for g, (ci, ti) in zip(cz_gates, expected_pairs):
            assert g.control_indices == (ci,) and g.target_indices == (ti,)

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
        _, circ = _transpile_and_get_circuit(
            circuit, bindings={"n": 3, "ising": ising, "gamma": 0.5}
        )
        gates = _get_gates(circ)
        h_gates = [g for g in gates if g.name == gate_names.H]
        rzz_gates = [g for g in gates if g.name == gate_names.PauliRotation]
        assert len(h_gates) == 3
        assert len(rzz_gates) == 2
        for i, g in enumerate(h_gates):
            assert g.target_indices == (i,)
        for g in rzz_gates:
            assert g.pauli_ids == (SinglePauli.Z, SinglePauli.Z)  # ZZ

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
        _, circ = _transpile_and_get_circuit(circuit, bindings={"n": n})
        gates = _get_gates(circ)
        cx_gates = [g for g in gates if g.name == gate_names.CNOT]
        assert len(cx_gates) == n - 1
        for i, g in enumerate(cx_gates):
            assert g.target_indices == (i + 1,)
            assert g.control_indices == (i,)
        # X on q0, then CX chain propagates: |1000> → |1100> → |1110> → |1111>
        sv = _run_statevector(circ)
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

        _, circ = _transpile_and_get_circuit(circuit, bindings={"n": n})
        gates = _get_gates(circ)
        cz_gates = [g for g in gates if g.name == gate_names.CZ]
        assert len(cz_gates) == n * (n - 1) // 2
        expected_pairs = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
        for g, (ci, ti) in zip(cz_gates, expected_pairs):
            assert g.control_indices == (ci,) and g.target_indices == (ti,)

    # -- Execution (sampling) verification for nested loops --

    def test_nested_range_x_deterministic_sampling(self):
        """GHZ via nested H+CX pattern: sampling returns only (0,...,0) or (1,...,1)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q[0] = qmc.h(q[0])
            for i in qmc.range(n - 1):
                q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])
            return qmc.measure(q)

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(circuit, bindings={"n": 4})
        executor = transpiler.executor()
        job = exe.sample(executor, bindings={}, shots=500)
        result = job.result()
        all_zeros = (0, 0, 0, 0)
        all_ones = (1, 1, 1, 1)
        for value, _ in result.results:
            assert value in (all_zeros, all_ones)
        # Both outcomes should appear
        outcomes = {val for val, _ in result.results}
        assert outcomes == {all_zeros, all_ones}


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
        _, circ = _transpile_and_get_circuit(
            qaoa_layer,
            bindings={"n": 3, "ising": ising, "gamma": 0.5, "beta": 0.7},
        )
        gates = _get_gates(circ)
        h_gates = [g for g in gates if g.name == gate_names.H]
        rzz_gates = [g for g in gates if g.name == gate_names.PauliRotation]
        rx_gates = [g for g in gates if g.name == gate_names.RX]
        assert len(h_gates) == 3
        assert len(rzz_gates) == 3
        assert len(rx_gates) == 3
        for i, g in enumerate(h_gates):
            assert g.target_indices == (i,)
        for g in rzz_gates:
            assert g.pauli_ids == (SinglePauli.Z, SinglePauli.Z)  # ZZ
        for i, g in enumerate(rx_gates):
            assert g.target_indices == (i,)

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
        _, circ = _transpile_and_get_circuit(
            qaoa_layer,
            bindings={"n": 2, "ising": ising, "gamma": gamma, "beta": beta},
        )
        sv = _run_statevector(circ)
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
        _, circ = _transpile_and_get_circuit(circuit, bindings={"n": n})
        gates = _get_gates(circ)
        cz_gates = [g for g in gates if g.name == gate_names.CZ]
        # Even pairs: 3 (0-1, 2-3, 4-5), Odd pairs: 2 (1-2, 3-4) = 5 total
        even_pairs_list = list(range(0, n - 1, 2))
        odd_pairs_list = list(range(1, n - 1, 2))
        assert len(cz_gates) == len(even_pairs_list) + len(odd_pairs_list)
        # Verify even pairs first, then odd pairs
        for idx, start in enumerate(even_pairs_list):
            g = cz_gates[idx]
            assert g.control_indices == (start,) and g.target_indices == (start + 1,)
        for idx, start in enumerate(odd_pairs_list):
            g = cz_gates[len(even_pairs_list) + idx]
            assert g.control_indices == (start,) and g.target_indices == (start + 1,)

    # NOTE: test_qaoa_single_layer_parametric is NOT ported because it uses
    # `gamma * Jij` where gamma is a Parameter and Jij is a float — this
    # triggers TypeError: unsupported operand type(s) for *: 'Parameter' and
    # 'float' in QuriParts. This is a fundamental limitation of QURI Parts'
    # Parameter type not supporting arithmetic with float operands.


# ============================================================================
# 4. Transpiler Pass Tests
# ============================================================================


class TestTranspilerPassesPipeline:
    """Test individual transpiler pipeline stages."""

    @pytest.fixture
    def transpiler(self):
        return QuriPartsTranspiler()

    def test_to_block(self, transpiler):
        """to_block converts QKernel to Block."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        block = transpiler.to_block(circuit)
        assert block is not None
        assert len(block.operations) == 4

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
        assert inlined.kind == BlockKind.AFFINE

    def test_affine_validate(self, transpiler):
        """affine_validate() checks no-cloning on inlined block."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            q = qmc.x(q)
            return qmc.measure(q)

        block = transpiler.to_block(circuit)
        inlined = transpiler.inline(block)
        validated = transpiler.affine_validate(inlined)
        assert validated is not None
        assert len(validated.operations) == 4

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
        validated = transpiler.affine_validate(inlined)
        folded = transpiler.constant_fold(validated, bindings={"theta": 0.5})
        assert folded is not None
        assert len(folded.operations) == 3

    def test_analyze(self, transpiler):
        """analyze() validates dependencies and marks block ANALYZED."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        block = transpiler.to_block(circuit)
        inlined = transpiler.inline(block)
        validated = transpiler.affine_validate(inlined)
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
        validated = transpiler.affine_validate(inlined)
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
        validated = transpiler.affine_validate(inlined)
        folded = transpiler.constant_fold(validated)
        analyzed = transpiler.analyze(folded)
        separated = transpiler.separate(analyzed)
        exe = transpiler.emit(separated)

        assert isinstance(exe, ExecutableProgram)
        assert len(exe.compiled_quantum) == 1

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
        assert len(_get_gates(exe.quantum_circuit)) == 1

    def test_to_circuit_convenience(self, transpiler):
        """to_circuit() returns the QURI Parts circuit directly."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        qc = transpiler.to_circuit(circuit)
        assert qc is not None
        assert qc.qubit_count == 1

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
        validated = transpiler.affine_validate(inlined)
        folded = transpiler.constant_fold(validated)
        analyzed = transpiler.analyze(folded)
        separated = transpiler.separate(analyzed)
        exe_step = transpiler.emit(separated)
        qc_step = exe_step.compiled_quantum[0].circuit
        sv_step = _run_statevector(qc_step)

        assert statevectors_equal(sv_full, sv_step)


class TestTranspilerConfigPortable:
    """Test TranspilerConfig, substitute(), and segment structure on QuriParts.

    These tests verify the backend-agnostic transpiler pipeline stages
    (to_block, substitute, inline, constant_fold, analyze, separate).
    """

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

        transpiler = QuriPartsTranspiler()
        config = TranspilerConfig.with_strategies({"qft": "approximate_k2"})
        transpiler.set_config(config)
        assert transpiler.config is config

    def test_substitute_no_rules_is_noop(self):
        """substitute() with no config rules returns block unchanged."""
        transpiler = QuriPartsTranspiler()

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        block = transpiler.to_block(circuit)
        substituted = transpiler.substitute(block)
        # Should return exact same block (no rules)
        assert substituted is block

    def test_substitute_pass_sets_strategy(self):
        """substitute() pass with config modifies block for QFT circuits."""
        from qamomile.circuit.transpiler.transpiler import TranspilerConfig

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q = qmc.qft(q)
            return qmc.measure(q)

        transpiler = QuriPartsTranspiler()
        config = TranspilerConfig.with_strategies({"qft": "approximate_k2"})
        transpiler.set_config(config)

        block = transpiler.to_block(circuit)
        substituted = transpiler.substitute(block)
        # The substitution pass should return a new block (not the original)
        assert substituted is not None
        assert substituted is not block

    def test_separate_segments_simple_circuit(self):
        """separate() produces SimplifiedProgram with quantum segment."""
        transpiler = QuriPartsTranspiler()

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        block = transpiler.to_block(circuit)
        inlined = transpiler.inline(block)
        validated = transpiler.affine_validate(inlined)
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

        transpiler = QuriPartsTranspiler()
        block = transpiler.to_block(qpe_circuit, bindings={"phase": np.pi / 2})
        substituted = transpiler.substitute(block)
        inlined = transpiler.inline(substituted)
        validated = transpiler.affine_validate(inlined)
        folded = transpiler.constant_fold(validated, bindings={"phase": np.pi / 2})
        analyzed = transpiler.analyze(folded)
        separated = transpiler.separate(analyzed)

        assert isinstance(separated, SimplifiedProgram)
        assert separated.quantum is not None
        # QPE returns QFixed → Float, so classical_post should handle the decode
        assert separated.classical_post is not None

    def test_approximate_qft_fewer_gates(self):
        """Approximate QFT strategy produces fewer gates than standard."""
        from qamomile.circuit.transpiler.transpiler import TranspilerConfig

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(5, "q")
            q = qmc.qft(q)
            return qmc.measure(q)

        # Standard QFT (default)
        transpiler_std = QuriPartsTranspiler()
        exe_std = transpiler_std.transpile(circuit)
        circ_std = exe_std.compiled_quantum[0].circuit
        gates_std = _get_gates(circ_std)
        gates_std_count = len(gates_std)

        # Approximate QFT (k=2)
        transpiler_approx = QuriPartsTranspiler()
        config = TranspilerConfig.with_strategies({"qft": "approximate_k2"})
        transpiler_approx.set_config(config)
        exe_approx = transpiler_approx.transpile(circuit)
        circ_approx = exe_approx.compiled_quantum[0].circuit
        gates_approx = _get_gates(circ_approx)
        gates_approx_count = len(gates_approx)

        # Approximate should have strictly fewer gates for 5 qubits
        assert gates_approx_count < gates_std_count

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


class TestParameterizedCircuits:
    """Test parametric circuit transpilation and execution."""

    def test_parametric_rx(self):
        """Transpile with parameters= preserves parametric circuit."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit
        assert qc.parameter_count == 1

    def test_parametric_bind_and_execute(self):
        """Bind parameters and execute."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        transpiler = QuriPartsTranspiler()
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

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(circuit, bindings={"n": 3}, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit
        assert qc.parameter_count == 1

    # NOTE: test_parametric_bind_and_rebind is in TestAdvancedParameterHandling
    # to avoid duplication (identical circuit and purpose).

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
        sv = _run_statevector(qc)
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-10)


# ============================================================================
# 1b. Parametric Gate Tests (unbound parameters)
# ============================================================================


class TestParametricGates:
    """Verify that each parameterized gate keeps its parameter unbound
    when transpiled with ``parameters=["theta"]``.

    For QURI Parts, gates are Rust-backed objects — we check ``gate.name``
    (not ``isinstance``) and use ``qc.parameter_count`` (not ``len(qc.parameters)``).
    """

    def test_rx_parametric(self):
        """RX with parameters=['theta'] keeps parameter unbound and produces correct statevector."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit
        assert qc.parameter_count == 1
        angle = np.pi / 3
        gates = _get_gates(qc, parameter_bindings=[angle])
        assert any(g.name == gate_names.RX for g in gates)
        sv = _run_statevector(qc, parameter_bindings=[angle])
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RX"].matrix_fn(angle)
        )
        assert statevectors_equal(sv, expected)

    def test_ry_parametric(self):
        """RY with parameters=['theta'] keeps parameter unbound and produces correct statevector."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.ry(q, theta)
            return qmc.measure(q)

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit
        assert qc.parameter_count == 1
        angle = np.pi / 3
        gates = _get_gates(qc, parameter_bindings=[angle])
        assert any(g.name == gate_names.RY for g in gates)
        sv = _run_statevector(qc, parameter_bindings=[angle])
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RY"].matrix_fn(angle)
        )
        assert statevectors_equal(sv, expected)

    def test_rz_parametric(self):
        """RZ with parameters=['theta'] keeps parameter unbound and produces correct statevector."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rz(q, theta)
            return qmc.measure(q)

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit
        assert qc.parameter_count == 1
        angle = np.pi / 3
        gates = _get_gates(qc, parameter_bindings=[angle])
        assert any(g.name == gate_names.RZ for g in gates)
        sv = _run_statevector(qc, parameter_bindings=[angle])
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["RZ"].matrix_fn(angle)
        )
        assert statevectors_equal(sv, expected)

    def test_p_parametric(self):
        """P with parameters=['theta'] keeps parameter unbound (emits RZ in parametric path).

        P(θ)|0⟩ = |0⟩ (diagonal gate, trivial on |0⟩), so we compare against
        RZ(θ)|0⟩ which differs only by global phase — statevectors_equal handles this.
        """

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.p(q, theta)
            return qmc.measure(q)

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit
        assert qc.parameter_count == 1
        angle = np.pi / 3
        gates = _get_gates(qc, parameter_bindings=[angle])
        # P gate emits as RZ in parametric path (U1 in non-parametric)
        assert any(g.name == gate_names.RZ for g in gates)
        sv = _run_statevector(qc, parameter_bindings=[angle])
        expected = compute_expected_statevector(
            all_zeros_state(1), GATE_SPECS["P"].matrix_fn(angle)
        )
        assert statevectors_equal(sv, expected)

    def test_cp_parametric(self):
        """CP with parameters=['theta'] keeps parameter unbound (decomposes to RZ+CNOT)."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = qmc.cp(q[0], q[1], theta)
            return qmc.measure(q)

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit
        assert qc.parameter_count == 1
        angle = np.pi / 4
        # CP decomposes to RZ + CNOT pattern
        gates = _get_gates(qc, parameter_bindings=[angle])
        gate_name_set = {g.name for g in gates}
        assert gate_names.RZ in gate_name_set
        assert gate_names.CNOT in gate_name_set
        sv = _run_statevector(qc, parameter_bindings=[angle])
        expected = compute_expected_statevector(
            all_zeros_state(2), GATE_SPECS["CP"].matrix_fn(angle)
        )
        assert statevectors_equal(sv, expected)

    def test_rzz_parametric(self):
        """RZZ with parameters=['theta'] keeps parameter unbound (emits PauliRotation)."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0], q[1] = qmc.rzz(q[0], q[1], theta)
            return qmc.measure(q)

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        qc = exe.compiled_quantum[0].circuit
        assert qc.parameter_count == 1
        angle = np.pi / 3
        gates = _get_gates(qc, parameter_bindings=[angle])
        assert any(g.name == gate_names.PauliRotation for g in gates)
        sv = _run_statevector(qc, parameter_bindings=[angle])
        expected = compute_expected_statevector(
            all_zeros_state(2), GATE_SPECS["RZZ"].matrix_fn(angle)
        )
        assert statevectors_equal(sv, expected)

    def test_vector_parametric(self):
        """Vector[Float] with parameters=['thetas'] gives 3 indexed params and correct statevector."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.ry(q[i], thetas[i])
            return qmc.measure(q)

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(circuit, bindings={"n": 3}, parameters=["thetas"])
        qc = exe.compiled_quantum[0].circuit
        assert qc.parameter_count == 3
        angles = [np.pi / 4, np.pi / 3, np.pi / 6]
        sv = _run_statevector(qc, parameter_bindings=angles)
        expected = compute_expected_statevector(
            all_zeros_state(3),
            tensor_product(
                GATE_SPECS["RY"].matrix_fn(angles[2]),
                GATE_SPECS["RY"].matrix_fn(angles[1]),
                GATE_SPECS["RY"].matrix_fn(angles[0]),
            ),
        )
        assert statevectors_equal(sv, expected)

    # -- BinOp parametric tests --

    def test_parametric_mul_binop(self):
        """theta * c via items() loop, bind and verify numerical result.

        Note: QURI Parts' Rust-backed Parameter does not support arithmetic
        operators (*, +, -) with floats, so BinOp with unbound parameters
        is not possible. We test with fully bound parameters instead.
        """

        @qmc.qkernel
        def circuit(
            theta: qmc.Float,
            coeff: qmc.Dict[qmc.UInt, qmc.Float],
        ) -> qmc.Bit:
            q = qmc.qubit("q")
            for _, c in qmc.items(coeff):
                q = qmc.rx(q, theta * c)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"theta": 1.5, "coeff": {0: 2.0}}
        )
        gates = _get_gates(qc)
        rx_gates = [g for g in gates if g.name == gate_names.RX]
        assert len(rx_gates) == 1
        assert np.isclose(rx_gates[0].params[0], 3.0, atol=1e-10)

    def test_parametric_add_binop(self):
        """theta + o via items() loop, bind and verify numerical result.

        Note: QURI Parts' Rust-backed Parameter does not support arithmetic
        operators, so BinOp with unbound parameters is not possible.
        """

        @qmc.qkernel
        def circuit(
            theta: qmc.Float,
            offset: qmc.Dict[qmc.UInt, qmc.Float],
        ) -> qmc.Bit:
            q = qmc.qubit("q")
            for _, o in qmc.items(offset):
                q = qmc.rx(q, theta + o)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"theta": 1.0, "offset": {0: 0.5}}
        )
        gates = _get_gates(qc)
        rx_gates = [g for g in gates if g.name == gate_names.RX]
        assert len(rx_gates) == 1
        assert np.isclose(rx_gates[0].params[0], 1.5, atol=1e-10)

    def test_parametric_sub_binop(self):
        """theta - o via items() loop, bind and verify numerical result.

        Note: QURI Parts' Rust-backed Parameter does not support arithmetic
        operators, so BinOp with unbound parameters is not possible.
        """

        @qmc.qkernel
        def circuit(
            theta: qmc.Float,
            offset: qmc.Dict[qmc.UInt, qmc.Float],
        ) -> qmc.Bit:
            q = qmc.qubit("q")
            for _, o in qmc.items(offset):
                q = qmc.rx(q, theta - o)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"theta": 1.0, "offset": {0: 0.5}}
        )
        gates = _get_gates(qc)
        rx_gates = [g for g in gates if g.name == gate_names.RX]
        assert len(rx_gates) == 1
        assert np.isclose(rx_gates[0].params[0], 0.5, atol=1e-10)


# ============================================================================
# 5. Stdlib Tests (QFT, IQFT, controlled, CompositeGate)
# ============================================================================


class TestStdlibQFT:
    """Test QFT/IQFT through the full QuriParts pipeline.

    QuriParts has no native composite gate emitter for QFT, but
    StandardEmitPass provides a manual decomposition fallback using
    basic gates (H, CP→decomposed, SWAP).
    """

    def test_qft_transpiles(self):
        """QFT can be transpiled to a QURI Parts circuit."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q = qmc.qft(q)
            return qmc.measure(q)

        _, circ = _transpile_and_get_circuit(circuit)
        assert circ is not None
        assert circ.qubit_count == 3

    def test_iqft_transpiles(self):
        """IQFT can be transpiled to a QURI Parts circuit."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q = qmc.iqft(q)
            return qmc.measure(q)

        _, circ = _transpile_and_get_circuit(circuit)
        assert circ is not None
        assert circ.qubit_count == 3

    def test_qft_then_iqft_statevector(self):
        """QFT followed by IQFT is identity on the input state."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q = qmc.qft(q)
            q = qmc.iqft(q)
            return qmc.measure(q)

        _, circ = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(circ)
        # QFT -> IQFT must recover the input state X(q[0]) = |01> in QURI Parts
        # convention (qubit 0 is LSB), statevector index 1 = [0, 1, 0, 0]
        expected = np.array([0, 1, 0, 0], dtype=complex)
        assert np.allclose(sv, expected, atol=1e-10)

    def test_qft_decomposed_gate_set(self):
        """Decomposed QFT uses basic gates (H, RZ, CNOT, SWAP)."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q = qmc.qft(q)
            return qmc.measure(q)

        _, circ = _transpile_and_get_circuit(circuit)
        gates = _get_gates(circ)
        # 3-qubit QFT: 3 H gates + 3 CP (each → 3 RZ + 2 CNOT) + 1 SWAP = 19 gates
        h_gates = [g for g in gates if g.name == gate_names.H]
        rz_gates = [g for g in gates if g.name == gate_names.RZ]
        cx_gates = [g for g in gates if g.name == gate_names.CNOT]
        swap_gates = [g for g in gates if g.name == gate_names.SWAP]
        assert len(h_gates) == 3
        for i, g in enumerate(h_gates):
            assert g.target_indices == (2 - i,)
        assert len(rz_gates) == 9  # 3 CP × 3 RZ each
        assert len(cx_gates) == 6  # 3 CP × 2 CNOT each
        assert len(swap_gates) == 1
        assert swap_gates[0].target_indices == (0, 2)


class TestStdlibQPE:
    """Test Quantum Phase Estimation structure through QuriParts pipeline."""

    def test_qpe_transpiles(self):
        """QPE can be transpiled to a QURI Parts circuit with correct qubit count."""

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

        _, circ = _transpile_and_get_circuit(qpe_3bit, bindings={"phase": np.pi / 2})
        assert circ is not None
        assert circ.qubit_count == 4  # 3 counting + 1 target

    def test_qpe_execution(self):
        """QPE execution returns exact phase estimate via QuriParts sampling."""

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

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(qpe_3bit, bindings={"phase": np.pi / 2})
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=256)
        result = job.result()
        assert result is not None
        # phase = pi/2 → phase/2π = 0.25, representable exactly in 3-bit QPE
        assert len(result.results) == 1
        value, count = result.results[0]
        assert np.isclose(value, 0.25, atol=1e-9)
        assert count == 256

    @pytest.mark.parametrize(
        "phase, expected_value",
        [
            pytest.param(np.pi / 2, 0.25, id="phase_pi_over_2"),
            pytest.param(np.pi, 0.5, id="phase_pi"),
            pytest.param(3 * np.pi / 2, 0.75, id="phase_3pi_over_2"),
        ],
    )
    def test_qpe_decode_correctness(self, phase, expected_value):
        """QPE decode returns numerically correct float for known phases.

        P(phase)|1⟩ = e^{i*phase}|1⟩, so the eigenvalue phase is phase/(2π).
        With 3-bit QPE, the decoded float should match phase/(2π).
        """

        @qmc.qkernel
        def phase_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            return qmc.p(q, theta)

        @qmc.qkernel
        def qpe_3bit(phase_angle: qmc.Float) -> qmc.Float:
            phase_register = qmc.qubit_array(3, name="phase_reg")
            target = qmc.qubit(name="target")
            target = qmc.x(target)
            phase_q = qmc.qpe(target, phase_register, phase_gate, theta=phase_angle)
            return qmc.measure(phase_q)

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(qpe_3bit, bindings={"phase_angle": phase})
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=500)
        result = job.result()
        # Find the most frequent result
        most_frequent_value = max(result.results, key=lambda x: x[1])[0]
        assert np.isclose(most_frequent_value, expected_value, atol=1e-10)


class TestControlledGate:
    """Test qmc.controlled() through the QuriParts pipeline.

    QuriParts' circuit_to_gate() returns None, but the transpiler's
    fallback path in _emit_controlled_u manually decomposes each gate
    in the sub-kernel into its controlled variant (CH, CRX, CRY, etc.).
    This works for single-control cases (num_controls=1).
    """

    def test_controlled_h(self):
        """controlled(h_gate) transpiles to a 2-qubit circuit."""

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

        _, circ = _transpile_and_get_circuit(circuit)
        assert circ.qubit_count == 2
        assert len(_get_gates(circ)) > 0

    def test_controlled_h_statevector(self):
        """controlled(H) with ctrl=|1⟩: statevector matches CH matrix."""

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

        _, circ = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(circ)
        CH = GATE_SPECS["CH"].matrix_fn()
        X = GATE_SPECS["X"].matrix_fn()
        state = tensor_product(identity(2), X) @ all_zeros_state(2)
        expected = CH @ state
        assert statevectors_equal(sv, expected)

    def test_controlled_rx(self):
        """controlled(rx_gate) transpiles with parameter binding."""

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

        _, circ = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 2})
        assert circ.qubit_count == 2
        assert len(_get_gates(circ)) > 0

    def test_controlled_rx_statevector(self):
        """controlled(RX) with ctrl=|1⟩: statevector matches CRX matrix."""

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

        theta = np.pi / 2
        _, circ = _transpile_and_get_circuit(circuit, bindings={"theta": theta})
        sv = _run_statevector(circ)
        CRX = GATE_SPECS["CRX"].matrix_fn(theta)
        X = GATE_SPECS["X"].matrix_fn()
        state = tensor_product(identity(2), X) @ all_zeros_state(2)
        expected = CRX @ state
        assert statevectors_equal(sv, expected)

    # -- Double-control (num_controls=2) tests --
    # QuriParts' gate_controlled() returns None and the fallback decomposition
    # only supports single control.  Verify that multi-control raises EmitError.

    def test_controlled_h_double_control_raises(self):
        """controlled(h_gate, num_controls=2) raises EmitError on QuriParts."""
        from qamomile.circuit.transpiler.errors import EmitError

        @qmc.qkernel
        def h_gate(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            return q

        controlled_h2 = qmc.controlled(h_gate, num_controls=2)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0], q[1], q[2] = controlled_h2(q[0], q[1], q[2])
            return qmc.measure(q)

        with pytest.raises(EmitError, match="multi-controlled"):
            _transpile_and_get_circuit(circuit)

    def test_controlled_rx_double_control_raises(self):
        """controlled(rx_gate, num_controls=2) raises EmitError on QuriParts."""
        from qamomile.circuit.transpiler.errors import EmitError

        @qmc.qkernel
        def rx_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            q = qmc.rx(q, theta)
            return q

        controlled_rx2 = qmc.controlled(rx_gate, num_controls=2)

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0], q[1], q[2] = controlled_rx2(q[0], q[1], q[2], theta=theta)
            return qmc.measure(q)

        with pytest.raises(EmitError, match="multi-controlled"):
            _transpile_and_get_circuit(circuit, bindings={"theta": np.pi})


class TestCustomCompositeGate:
    """Test custom CompositeGate through the QuriParts pipeline.

    When circuit_to_gate() returns None, _emit_custom_composite falls
    back to manually emitting the sub-circuit's operations inline.
    """

    def test_custom_composite_transpiles(self):
        """Custom CompositeGate (BellPair) decomposes and transpiles."""
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

        _, circ = _transpile_and_get_circuit(circuit)
        assert circ.qubit_count == 2
        gates = _get_gates(circ)
        # BellPair: H(0), CNOT(ctrl=0, tgt=1)
        assert len(gates) == 2
        assert gates[0].name == gate_names.H and gates[0].target_indices == (0,)
        assert (
            gates[1].name == gate_names.CNOT
            and gates[1].target_indices == (1,)
            and gates[1].control_indices == (0,)
        )

    def test_composite_statevector(self):
        """BellPair CompositeGate produces Bell state |Φ+⟩."""
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

        _, circ = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(circ)
        assert circ.qubit_count == 2
        expected = np.zeros(4, dtype=complex)
        expected[0] = 1.0 / np.sqrt(2)  # |00⟩
        expected[3] = 1.0 / np.sqrt(2)  # |11⟩
        assert statevectors_equal(sv, expected)

    def test_composite_gate_counts(self):
        """BellPair decomposes to exactly 1 H + 1 CNOT."""
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

        _, circ = _transpile_and_get_circuit(circuit)
        gates = _get_gates(circ)
        h_gates = [g for g in gates if g.name == gate_names.H]
        cx_gates = [g for g in gates if g.name == gate_names.CNOT]
        assert len(h_gates) == 1
        assert len(cx_gates) == 1
        assert h_gates[0].target_indices == (0,)
        assert cx_gates[0].target_indices == (1,)
        assert cx_gates[0].control_indices == (0,)


# ============================================================================
# 6. Execution Tests
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

        transpiler = QuriPartsTranspiler()
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

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=1000)
        result = job.result()
        assert result is not None

        outcomes = {val for val, _ in result.results}
        # Bell state should only produce (0, 0) and (1, 1)
        assert outcomes.issubset({(0, 0), (1, 1)})
        # Both outcomes should appear with reasonable probability (>15%)
        total = sum(c for _, c in result.results)
        for _, count in result.results:
            assert count / total > 0.15

    def test_sample_shot_count(self):
        """Verify total shot count matches requested shots."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        transpiler = QuriPartsTranspiler()
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

        transpiler = QuriPartsTranspiler()
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

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(circuit)
        executor = transpiler.executor()

        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        assert result is not None
        for value, _ in result.results:
            # Vector[Bit] returns tuple of ints
            assert isinstance(value, tuple)
            assert len(value) == 3
            # X(q[0]) and X(q[2]) → deterministic (1, 0, 1)
            assert value == (1, 0, 1)

    # -- Deterministic sampling tests for all gate types --

    @pytest.mark.parametrize(
        "kernel, expected",
        [
            pytest.param(_circ_x, 1, id="X_gives_1"),
            pytest.param(_circ_y, 1, id="Y_gives_1"),
            pytest.param(_circ_z, 0, id="Z_gives_0"),
            pytest.param(_circ_s, 0, id="S_gives_0"),
            pytest.param(_circ_t, 0, id="T_gives_0"),
            pytest.param(_circ_xx, 0, id="XX_identity"),
            pytest.param(_circ_xy, 0, id="XY_gives_0"),
            pytest.param(_circ_hxh, 0, id="HXH_equals_Z"),
        ],
    )
    def test_single_gate_deterministic_sampling(self, kernel, expected):
        """Single-qubit deterministic circuits: sampling matches expected bit."""
        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(kernel)
        executor = transpiler.executor()
        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        for value, _ in result.results:
            assert value == expected

    @pytest.mark.parametrize(
        "kernel, expected",
        [
            pytest.param(_cx_00, (0, 0), id="CX_00"),
            pytest.param(_cx_10, (1, 1), id="CX_10"),
            pytest.param(_cx_01, (0, 1), id="CX_01"),
            pytest.param(_cx_11, (1, 0), id="CX_11"),
            pytest.param(_cz_00, (0, 0), id="CZ_00"),
            pytest.param(_cz_10, (1, 0), id="CZ_10"),
            pytest.param(_cz_01, (0, 1), id="CZ_01"),
            pytest.param(_cz_11, (1, 1), id="CZ_11"),
            pytest.param(_swap_10, (0, 1), id="SWAP_10"),
            pytest.param(_swap_01, (1, 0), id="SWAP_01"),
        ],
    )
    def test_two_qubit_deterministic_sampling(self, kernel, expected):
        """Two-qubit deterministic circuits: sampling matches expected tuple."""
        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(kernel)
        executor = transpiler.executor()
        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        for value, _ in result.results:
            assert value == expected

    @pytest.mark.parametrize(
        "kernel, expected",
        [
            pytest.param(_ccx_000, (0, 0, 0), id="CCX_000"),
            pytest.param(_ccx_001, (1, 0, 0), id="CCX_001"),
            pytest.param(_ccx_010, (0, 1, 0), id="CCX_010"),
            pytest.param(_ccx_011, (1, 1, 1), id="CCX_011_flips"),
            pytest.param(_ccx_100, (0, 0, 1), id="CCX_100"),
            pytest.param(_ccx_101, (1, 0, 1), id="CCX_101"),
            pytest.param(_ccx_110, (0, 1, 1), id="CCX_110"),
            pytest.param(_ccx_111, (1, 1, 0), id="CCX_111_flips"),
        ],
    )
    def test_ccx_deterministic_sampling(self, kernel, expected):
        """CCX on all 8 basis states via sampling."""
        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(kernel)
        executor = transpiler.executor()
        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        for value, _ in result.results:
            assert value == expected

    @pytest.mark.parametrize(
        "kernel, expected",
        [
            pytest.param(_cx_chain_3, (1, 1, 1), id="CX_chain_all_ones"),
            pytest.param(_swap_after_x, (0, 1), id="SWAP_after_X"),
            pytest.param(_x_cx_x, (1, 0), id="X_CX_X_cancel"),
        ],
    )
    def test_gate_combination_deterministic_sampling(self, kernel, expected):
        """Multi-gate combination circuits: sampling matches expected."""
        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(kernel)
        executor = transpiler.executor()
        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        for value, _ in result.results:
            assert value == expected


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

    @pytest.mark.parametrize("n_qubits", [20, 50])
    def test_large_qubit_count(self, n_qubits):
        """Circuit with many qubits transpiles successfully."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": n_qubits}, skip_smoke_test=True
        )
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

    def test_mixed_qubit_and_qubit_array(self):
        """Mixed qubit() + qubit_array() in same circuit: transpile + statevector."""

        @qmc.qkernel
        def circuit() -> tuple[qmc.Bit, qmc.Vector[qmc.Bit]]:
            anc = qmc.qubit("anc")
            q = qmc.qubit_array(2, "q")
            anc = qmc.x(anc)
            anc, q[0] = qmc.cx(anc, q[0])
            return qmc.measure(anc), qmc.measure(q)

        _, circ = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(circ)
        assert circ.qubit_count == 3
        # X(anc) → CX(anc, q[0]) → |anc=1, q[0]=1, q[1]=0⟩ = |110⟩
        # LE ordering: qubit 0 (anc)=1, qubit 1 (q[0])=1, qubit 2 (q[1])=0
        # Index = 0b011 = 3
        assert np.isclose(abs(sv[3]), 1.0, atol=1e-10)


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

        transpiler = QuriPartsTranspiler()
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

        transpiler = QuriPartsTranspiler()
        # Provide a string for a UInt param — should fail
        with pytest.raises((TypeError, ValueError)):
            transpiler.transpile(circuit, bindings={"n": "not_a_number"})


# ============================================================================
# 8. Advanced Algorithm Module Tests
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
        def circuit(n: qmc.UInt, thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = ry_layer(q, thetas, qmc.uint(0))
            return qmc.measure(q)

        thetas = [0.1 * i for i in range(n_qubits)]
        _, qc = _transpile_and_get_circuit(
            circuit, bindings={"n": n_qubits, "thetas": thetas}
        )
        gates = _get_gates(qc)
        ry_gates = [g for g in gates if g.name == gate_names.RY]
        assert len(ry_gates) == n_qubits
        for i, g in enumerate(ry_gates):
            assert g.target_indices == (i,)

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
        gates = _get_gates(qc)
        cz_gates = [g for g in gates if g.name == gate_names.CZ]
        assert len(cz_gates) == n_qubits - 1
        for i, g in enumerate(cz_gates):
            assert g.target_indices == (i + 1,)
            assert g.control_indices == (i,)

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
        def circuit(n: qmc.UInt, thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = ry_layer(q, thetas, qmc.uint(0))
            q = cz_entangling_layer(q)
            return qmc.measure(q)

        thetas = [0.1, 0.2, 0.3]
        _, qc = _transpile_and_get_circuit(circuit, bindings={"n": 3, "thetas": thetas})
        gates = _get_gates(qc)
        ry_gates = [g for g in gates if g.name == gate_names.RY]
        cz_gates = [g for g in gates if g.name == gate_names.CZ]
        assert len(ry_gates) == 3
        assert len(cz_gates) == 2
        for i, g in enumerate(ry_gates):
            assert g.target_indices == (i,)

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
        RY0, RY1, RY2 = [GATE_SPECS["RY"].matrix_fn(t) for t in thetas]
        CZ = GATE_SPECS["CZ"].matrix_fn()
        I2 = identity(2)
        state = all_zeros_state(3)
        state = tensor_product(RY2, tensor_product(RY1, RY0)) @ state
        state = tensor_product(I2, CZ) @ state
        state = tensor_product(CZ, I2) @ state
        assert statevectors_equal(sv, state)


class TestAlgorithmQAOAModules:
    """Test QAOA circuit building blocks and full pipeline."""

    def _simple_ising(self):
        """Return a simple 2-qubit Ising model for testing."""
        quad = {(0, 1): 1.0}
        linear = {0: 0.5, 1: -0.5}
        return quad, linear

    def test_ising_cost(self):
        """ising_cost produces a valid unitary evolution."""
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
            q = ising_cost(quad_, linear_, q, gamma)
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
            q = x_mixer(q, beta)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"beta": np.pi / 4})
        sv = _run_statevector(qc)
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-10)

    def test_qaoa_layers_p1(self):
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
            q = qaoa_layers(qmc.uint(1), quad_, linear_, q, gammas, betas)
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
            q = qaoa_state(qmc.uint(1), quad_, linear_, qmc.uint(2), gammas, betas)
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
            q = qaoa_state(qmc.uint(2), quad_, linear_, qmc.uint(3), gammas, betas)
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
            q = qaoa_state(qmc.uint(1), quad_, linear_, qmc.uint(2), gammas, betas)
            return qmc.measure(q)

        gammas = np.array([0.5])
        betas = np.array([0.8])
        transpiler = QuriPartsTranspiler()
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
        _, qc = _transpile_and_get_circuit(circuit, bindings={"a": a, "b": b, "c": c})
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

        transpiler = QuriPartsTranspiler()
        executor = transpiler.executor()

        for theta in [0.0, np.pi / 4, np.pi / 2, np.pi]:
            exe = transpiler.transpile(circuit, bindings={"theta": theta, "obs": H})
            result = exe.run(executor).result()
            expected = np.cos(theta)
            assert np.isclose(result, expected, atol=1e-5), (
                f"theta={theta}: got {result}, expected {expected}"
            )

    def test_parametric_scalar_float_works(self):
        """Scalar Float with parameters= preserves parametric circuit."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.ry(q, theta)
            return qmc.measure(q)

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])
        assert exe.has_parameters
        qc = exe.compiled_quantum[0].circuit
        assert qc.parameter_count > 0

    def test_parametric_vector_float(self):
        """Vector[Float] with parameters= preserves parametric circuit."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.ry(q[i], thetas[i])
            return qmc.measure(q)

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(circuit, bindings={"n": 3}, parameters=["thetas"])
        assert exe.has_parameters
        qc = exe.compiled_quantum[0].circuit
        assert qc.parameter_count > 0

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

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(
            circuit, bindings={"n": 3}, parameters=["theta", "params"]
        )
        assert exe.has_parameters
        param_names = exe.parameter_names
        # Should have theta + params[0], params[1], params[2]
        assert len(param_names) >= 4

    def test_parametric_variational_classifier_pattern(self):
        """ry_layer + cz_entangling_layer with parametric Vector."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, params: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = ry_layer(q, params, qmc.uint(0))
            q = cz_entangling_layer(q)
            return qmc.measure(q)

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(circuit, bindings={"n": 3}, parameters=["params"])
        assert exe.has_parameters
        qc = exe.compiled_quantum[0].circuit
        assert qc.parameter_count == 3  # params[0], params[1], params[2]

    def test_parametric_bind_and_rebind(self):
        """Compile once, execute with different parameter values."""

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.ry(q, theta)
            return qmc.measure(q)

        transpiler = QuriPartsTranspiler()
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

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(circuit, parameters=["theta"])

        rng = np.random.default_rng(seed)
        theta = rng.uniform(0, 2 * np.pi)

        # Verify parameter binding works and circuit is valid
        qc = exe.compiled_quantum[0].circuit
        assert qc.parameter_count > 0

        executor = transpiler.executor()
        job = exe.sample(executor, bindings={"theta": theta}, shots=100)
        result = job.result()
        assert result is not None
        assert len(result.results) > 0


# ============================================================================
# 9. Algorithm Tests: Deutsch-Jozsa
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

    def test_constant_0_oracle_transpiles(self):
        """Constant-0 DJ circuit has n+1 qubits and correct gate structure."""

        @qmc.qkernel
        def dj_c0(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            inputs = qmc.qubit_array(n, name="input")
            ancilla = qmc.qubit(name="ancilla")
            ancilla = qmc.x(ancilla)
            for i in qmc.range(n):
                inputs[i] = qmc.h(inputs[i])
            ancilla = qmc.h(ancilla)
            # No oracle gates (constant zero)
            for i in qmc.range(n):
                inputs[i] = qmc.h(inputs[i])
            return qmc.measure(inputs)

        n = 3
        _, qc = _transpile_and_get_circuit(dj_c0, bindings={"n": n})
        gates = _get_gates(qc)
        h_gates = [g for g in gates if g.name == gate_names.H]
        x_gates = [g for g in gates if g.name == gate_names.X]
        assert len(h_gates) == 2 * n + 1  # n input H before + n H after + 1 ancilla H
        assert len(x_gates) == 1  # ancilla init
        assert x_gates[0].target_indices == (n,)  # X on ancilla qubit
        # First n H gates on input qubits 0..n-1
        for i in range(n):
            assert h_gates[i].target_indices == (i,)
        # Ancilla H
        assert h_gates[n].target_indices == (n,)
        # Final n H gates on input qubits 0..n-1
        for i in range(n):
            assert h_gates[n + 1 + i].target_indices == (i,)

    def test_constant_1_all_zeros(self):
        """Constant-1 oracle: input qubits still in |00> (constant → all-zeros)."""

        @qmc.qkernel
        def dj_c1() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            # Input qubits in |+>
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            # Ancilla in |->
            q[2] = qmc.x(q[2])
            q[2] = qmc.h(q[2])
            # Constant-1 oracle: X on ancilla
            q[2] = qmc.x(q[2])
            # Final H on input qubits
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(dj_c1)
        sv = _run_statevector(qc)
        # Constant oracle → input qubits return to |00>
        # Ancilla: X|-> = |-(-1)> = |+> ... actually X·H·X|0> = X·H|1> = X|-> = -|+>
        # Anyway, input register marginal: all probability on |00>
        prob_00 = np.abs(sv[0]) ** 2 + np.abs(sv[4]) ** 2
        assert np.isclose(prob_00, 1.0, atol=1e-10)

    def test_balanced_first_bit_non_zero(self):
        """Balanced first-bit oracle: input qubits NOT all |00>."""

        @qmc.qkernel
        def dj_bfirst() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            # Input qubits in |+>
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            # Ancilla in |->
            q[2] = qmc.x(q[2])
            q[2] = qmc.h(q[2])
            # Balanced oracle: CX(q0, ancilla) — first bit only
            q[0], q[2] = qmc.cx(q[0], q[2])
            # Final H on input qubits
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(dj_bfirst)
        sv = _run_statevector(qc)
        prob_00 = np.abs(sv[0]) ** 2 + np.abs(sv[4]) ** 2
        assert prob_00 < 0.01  # balanced → NOT all zeros

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_balanced_xor_gate_counts(self, n):
        """Balanced XOR DJ: H=2n+1, CNOT=n, X=1."""

        @qmc.qkernel
        def dj_bxor(n_val: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            inputs = qmc.qubit_array(n_val, name="input")
            ancilla = qmc.qubit(name="ancilla")
            ancilla = qmc.x(ancilla)
            for i in qmc.range(n_val):
                inputs[i] = qmc.h(inputs[i])
            ancilla = qmc.h(ancilla)
            # Oracle: CX each input to ancilla
            for i in qmc.range(n_val):
                inputs[i], ancilla = qmc.cx(inputs[i], ancilla)
            for i in qmc.range(n_val):
                inputs[i] = qmc.h(inputs[i])
            return qmc.measure(inputs)

        _, qc = _transpile_and_get_circuit(dj_bxor, bindings={"n_val": n})
        gates = _get_gates(qc)
        h_gates = [g for g in gates if g.name == gate_names.H]
        cx_gates = [g for g in gates if g.name == gate_names.CNOT]
        x_gates = [g for g in gates if g.name == gate_names.X]
        assert len(h_gates) == 2 * n + 1
        assert len(cx_gates) == n
        assert len(x_gates) == 1
        assert x_gates[0].target_indices == (n,)  # X on ancilla
        # Input H gates on qubits 0..n-1
        for i in range(n):
            assert h_gates[i].target_indices == (i,)
        # Ancilla H on qubit n
        assert h_gates[n].target_indices == (n,)
        # Oracle CX: each input qubit controls ancilla
        for i, g in enumerate(cx_gates):
            assert g.control_indices == (i,) and g.target_indices == (n,)
        # Final H on input qubits 0..n-1
        for i in range(n):
            assert h_gates[n + 1 + i].target_indices == (i,)

    def test_deutsch_jozsa_statevector_constant(self):
        """Constant-0 DJ: input register in |00> state before measurement."""

        @qmc.qkernel
        def dj_c0() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            q[2] = qmc.x(q[2])
            q[2] = qmc.h(q[2])
            # No oracle (constant zero)
            q[0] = qmc.h(q[0])
            q[1] = qmc.h(q[1])
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(dj_c0)
        sv = _run_statevector(qc)
        # 3 qubits: input[0,1] + ancilla[2]
        # Input: H·H|0⟩ = |0⟩, Ancilla: H·X|0⟩ = |−⟩
        # State: |00⟩ ⊗ |−⟩ = (|000⟩ − |100⟩)/√2
        # Indices 0 and 4 have amplitudes ±1/√2
        assert np.isclose(abs(sv[0]), 1.0 / np.sqrt(2), atol=1e-10)
        assert np.isclose(abs(sv[4]), 1.0 / np.sqrt(2), atol=1e-10)
        # Input register marginal: all probability on |00⟩
        prob_input_00 = abs(sv[0]) ** 2 + abs(sv[4]) ** 2
        assert np.isclose(prob_input_00, 1.0, atol=1e-10)


# ============================================================================
# 10. Expectation Value Pipeline Tests
# ============================================================================


class TestExpvalQuriPartsPipeline:
    """Test the observable expectation value pipeline through QuriParts.

    The Hamiltonian must be passed as an Observable parameter via bindings,
    not constructed inside the @qkernel function.
    """

    def test_expval_transpiles_with_compiled_expval(self):
        """Transpilation with Observable produces compiled_expval segment."""

        @qmc.qkernel
        def vqe(n: qmc.UInt, H: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(n, "q")
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.expval(q, H)

        H = qm_o.Z(0) * qm_o.Z(1)
        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(vqe, bindings={"H": H, "n": 2})
        assert len(exe.compiled_expval) == 1
        assert isinstance(exe.compiled_expval[0].hamiltonian, qm_o.Hamiltonian)

    def test_expval_z_on_zero(self):
        """<0|Z|0> = 1."""
        H = qm_o.Z(0)

        @qmc.qkernel
        def circuit(obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit("q")
            return qmc.expval((q,), obs)

        transpiler = QuriPartsTranspiler()
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

        transpiler = QuriPartsTranspiler()
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

        transpiler = QuriPartsTranspiler()
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

        transpiler = QuriPartsTranspiler()
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

        transpiler = QuriPartsTranspiler()
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

        transpiler = QuriPartsTranspiler()
        executor = transpiler.executor()
        # theta=0: <0|Z|0>=1
        exe_0 = transpiler.transpile(circuit, bindings={"theta": 0.0, "obs": H})
        result_0 = exe_0.run(executor).result()
        assert np.isclose(result_0, 1.0, atol=1e-6)

        # theta=pi: <1|Z|1>=-1
        exe_pi = transpiler.transpile(circuit, bindings={"theta": np.pi, "obs": H})
        result_pi = exe_pi.run(executor).result()
        assert np.isclose(result_pi, -1.0, atol=1e-6)

    def test_expval_parametric_bound_at_run(self):
        """Parametric expval: theta preserved as parameter, bound at run().

        This tests the scenario where theta is left unbound during transpile()
        (using parameters=["theta"]) and bound later at run() time. The executor
        must handle the bound circuit correctly in estimate_expectation().
        """
        H = qm_o.Z(0)

        @qmc.qkernel
        def circuit(theta: qmc.Float, obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit("q")
            q = qmc.ry(q, theta)
            return qmc.expval((q,), obs)

        transpiler = QuriPartsTranspiler()
        executor = transpiler.executor()

        exe = transpiler.transpile(
            circuit,
            bindings={"obs": H},
            parameters=["theta"],
        )
        assert exe.has_parameters

        # theta=0: <0|Z|0>=1
        result_0 = exe.run(executor, bindings={"theta": 0.0}).result()
        assert np.isclose(result_0, 1.0, atol=1e-6)

        # theta=pi/2: <+y|Z|+y>=0 (RY(pi/2)|0> = equal superposition)
        result_half = exe.run(executor, bindings={"theta": np.pi / 2}).result()
        assert np.isclose(result_half, 0.0, atol=1e-5)

        # theta=pi: <1|Z|1>=-1
        result_pi = exe.run(executor, bindings={"theta": np.pi}).result()
        assert np.isclose(result_pi, -1.0, atol=1e-6)

    def test_expval_parametric_sweep_bound_at_run(self):
        """Sweep theta as a runtime parameter for expval, verifying cos(theta)."""
        H = qm_o.Z(0)

        @qmc.qkernel
        def circuit(theta: qmc.Float, obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit("q")
            q = qmc.ry(q, theta)
            return qmc.expval((q,), obs)

        transpiler = QuriPartsTranspiler()
        executor = transpiler.executor()
        exe = transpiler.transpile(circuit, bindings={"obs": H}, parameters=["theta"])

        for theta in [0.0, 0.3, np.pi / 4, np.pi / 2, np.pi, 1.5 * np.pi]:
            result = exe.run(executor, bindings={"theta": theta}).result()
            expected = np.cos(theta)
            assert np.isclose(result, expected, atol=1e-5), (
                f"theta={theta}: got {result}, expected {expected}"
            )

    def test_expval_missing_observable_raises(self):
        """Transpilation without Observable binding raises RuntimeError."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, H: qmc.Observable) -> qmc.Float:
            q = qmc.qubit_array(n, "q")
            return qmc.expval(q, H)

        transpiler = QuriPartsTranspiler()
        with pytest.raises(RuntimeError, match="Observable.*not found in bindings"):
            transpiler.transpile(circuit, bindings={"n": 2})


# ============================================================================
# 11. Variational Classifier Pattern Tests
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
        _, qc = _transpile_and_get_circuit(classifier, bindings={"thetas": params})
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
        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(classifier, bindings={"thetas": params})
        executor = transpiler.executor()
        job = exe.sample(executor, bindings={}, shots=200)
        result = job.result()
        total = sum(count for _, count in result.results)
        assert total == 200

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
        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(
            classifier,
            bindings={"n": 3, "params": [0.1, 0.2, 0.3], "H": H_label},
        )
        qc = exe.compiled_quantum[0].circuit
        gates = _get_gates(qc)
        ry_gates = [g for g in gates if g.name == gate_names.RY]
        cz_gates = [g for g in gates if g.name == gate_names.CZ]
        assert len(ry_gates) == 3
        assert len(cz_gates) == 2
        for i, g in enumerate(ry_gates):
            assert g.target_indices == (i,)

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
        transpiler = QuriPartsTranspiler()
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
        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(
            classifier,
            bindings={"n": 3, "params": params, "H": H_label},
        )
        qc = exe.compiled_quantum[0].circuit
        gates = _get_gates(qc)
        ry_gates = [g for g in gates if g.name == gate_names.RY]
        cz_gates = [g for g in gates if g.name == gate_names.CZ]
        assert len(ry_gates) == 6  # 2 layers × 3 qubits
        assert len(cz_gates) == 4  # 2 layers × 2 CZ each (n-1=2)
        # RY gates target qubits 0,1,2 in each layer
        for layer in range(2):
            for q in range(3):
                assert ry_gates[layer * 3 + q].target_indices == (q,)
        # CZ entangling: (0,1), (1,2) in each layer
        for layer in range(2):
            assert cz_gates[layer * 2].control_indices == (0,)
            assert cz_gates[layer * 2].target_indices == (1,)
            assert cz_gates[layer * 2 + 1].control_indices == (1,)
            assert cz_gates[layer * 2 + 1].target_indices == (2,)

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
        transpiler = QuriPartsTranspiler()
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
        gates = _get_gates(qc)
        ry_gates = [g for g in gates if g.name == gate_names.RY]
        cz_gates = [g for g in gates if g.name == gate_names.CZ]
        assert len(ry_gates) == 6  # 3 x-encode + 3 params
        assert len(cz_gates) == 4  # 2 entangling layers × 2 CZ each (n-1=2)
        # RY gates target qubits 0,1,2 in each of 2 layers
        for layer in range(2):
            for q in range(3):
                assert ry_gates[layer * 3 + q].target_indices == (q,)
        # CZ entangling: (0,1), (1,2) in each of 2 layers
        for layer in range(2):
            assert cz_gates[layer * 2].control_indices == (0,)
            assert cz_gates[layer * 2].target_indices == (1,)
            assert cz_gates[layer * 2 + 1].control_indices == (1,)
            assert cz_gates[layer * 2 + 1].target_indices == (2,)

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
        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(
            classifier,
            bindings={"n": 2, "H": H_label},
            parameters=["params"],
        )
        assert exe.has_parameters


# ============================================================================
# 12. Bernstein-Vazirani Algorithm Tests
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
# 13. FQAOA Integration Tests
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

        _, qc = _transpile_and_get_circuit(circuit, bindings={"beta": 0.3, "hop": 1.0})
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

        _, qc = _transpile_and_get_circuit(circuit, bindings={"beta": 0.5, "hop": 1.0})
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

    def test_givens_rotation_gate_count(self):
        """Givens rotation contains 4 CNOT gates in QuriParts.

        Structure: CX + controlled-RY + CX.
        In QuriParts, controlled-RY is decomposed to 2 CNOT + RY gates,
        giving 2 outer CX + 2 inner CNOT = 4 CNOT total.
        """

        @qmc.qkernel
        def circuit(n: qmc.UInt, theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = givens_rotation(q, qmc.uint(0), qmc.uint(1), theta)
            return qmc.measure(q)

        _, circ = _transpile_and_get_circuit(circuit, bindings={"n": 2, "theta": 0.5})
        gates = _get_gates(circ)
        cx_gates = [g for g in gates if g.name == gate_names.CNOT]
        # 2 outer CX + 2 from CRY decomposition = 4
        assert len(cx_gates) == 4
        # All CNOT gates operate on qubits 0 and 1
        for g in cx_gates:
            assert set(g.target_indices + g.control_indices) == {0, 1}

    def test_hopping_gate_gate_count(self):
        """Hopping gate contains RX, CNOT, and RZ gates."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, beta: qmc.Float, hopping_val: qmc.Float
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            q = hopping_gate(q, qmc.uint(0), qmc.uint(1), beta, hopping_val)
            return qmc.measure(q)

        _, circ = _transpile_and_get_circuit(
            circuit, bindings={"n": 2, "beta": 0.3, "hopping_val": 1.0}
        )
        gates = _get_gates(circ)
        cx_gates = [g for g in gates if g.name == gate_names.CNOT]
        rx_gates = [g for g in gates if g.name == gate_names.RX]
        rz_gates = [g for g in gates if g.name == gate_names.RZ]
        # From code: 4 outer RX (±π/2) + 1 inner RX + 2 CNOT + 1 RZ
        assert len(cx_gates) == 2
        assert len(rx_gates) == 5  # 4 outer + 1 inner
        assert len(rz_gates) == 1
        # All gates operate on qubits 0 and/or 1
        for g in cx_gates:
            assert set(g.target_indices + g.control_indices) == {0, 1}
        for g in rx_gates:
            assert g.target_indices[0] in (0, 1)
        assert rz_gates[0].target_indices[0] in (0, 1)

    def test_cost_layer_gate_count(self):
        """FQAOA cost layer: RZ==len(linear), PauliRotation==len(quad)."""

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
        _, circ = _transpile_and_get_circuit(
            circuit,
            bindings={"n": 3, "gamma": 0.5, "linear": linear, "quad": quad},
        )
        gates = _get_gates(circ)
        rz_gates = [g for g in gates if g.name == gate_names.RZ]
        rzz_gates = [g for g in gates if g.name == gate_names.PauliRotation]
        assert len(rz_gates) == len(linear)
        assert len(rzz_gates) == len(quad)
        # RZ gates target the linear term qubits
        rz_targets = [g.target_indices[0] for g in rz_gates]
        for qubit in linear.keys():
            assert qubit in rz_targets
        # RZZ gates are PauliRotation(ZZ) on the quad term qubit pairs
        for g in rzz_gates:
            assert g.pauli_ids == (SinglePauli.Z, SinglePauli.Z)  # ZZ
            assert len(g.target_indices) == 2

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

        _, circ = _transpile_and_get_circuit(
            circuit,
            bindings={"n": n_qubits, "beta": 0.3, "hopping_val": 1.0},
        )
        # Even pairs: len(range(0, n-1, 2))
        # Odd pairs: len(range(1, n-1, 2))
        # Boundary: 1
        # Total hopping gates = even + odd + 1
        even_pairs = len(range(0, n_qubits - 1, 2))
        odd_pairs = len(range(1, n_qubits - 1, 2))
        total_hopping = even_pairs + odd_pairs + 1
        # Each hopping gate has 2 CNOT
        gates = _get_gates(circ)
        cx_gates = [g for g in gates if g.name == gate_names.CNOT]
        assert len(cx_gates) == total_hopping * 2
        # All CNOT gates should target adjacent qubit pairs
        for g in cx_gates:
            assert len(g.control_indices) == 1
            assert len(g.target_indices) == 1

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

        _, circ = _transpile_and_get_circuit(
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
        assert circ.qubit_count == 4


# ============================================================================
# 14. Advanced Circuit Pattern Tests
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

    def test_sub_kernel_per_element_in_loop(self):
        """Sub-kernel called per-element in qmc.range loop: n H gates."""

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

        _, circ = _transpile_and_get_circuit(main, bindings={"n": 4})
        gates = _get_gates(circ)
        h_gates = [g for g in gates if g.name == gate_names.H]
        assert len(h_gates) == 4
        for i, g in enumerate(h_gates):
            assert g.target_indices == (i,)

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

        _, circ = _transpile_and_get_circuit(main, bindings={"n": 2})
        sv = _run_statevector(circ)
        expected = tensor_product(plus_state(), plus_state())
        assert statevectors_equal(sv, expected)


class TestControlledSubRoutines:
    """Test qmc.controlled() with statevector verification and power.

    All tests use single-control (num_controls=1), which is handled by
    the fallback decomposition in standard_emit._emit_controlled_u.
    Multi-control (num_controls>1) would require circuit_to_gate support.
    """

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
        _, circ = _transpile_and_get_circuit(circuit, bindings={"theta": theta})
        sv = _run_statevector(circ)
        CRY = GATE_SPECS["CRY"].matrix_fn(theta)
        X = GATE_SPECS["X"].matrix_fn()
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

        _, circ = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 2})
        sv = _run_statevector(circ)
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
        _, circ = _transpile_and_get_circuit(circuit, bindings={"theta": theta})
        sv = _run_statevector(circ)
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

        _, circ = _transpile_and_get_circuit(circuit)
        assert circ.qubit_count >= 2

    def test_controlled_multi_gate_statevector(self):
        """Controlled(H·X) with ctrl=|1⟩: statevector verified."""

        @qmc.qkernel
        def hx_gate(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            q = qmc.x(q)
            return q

        controlled_hx = qmc.controlled(hx_gate)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])  # ctrl = |1⟩
            q[0], q[1] = controlled_hx(q[0], q[1])
            return qmc.measure(q)

        _, circ = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(circ)
        # ctrl=|1⟩: apply H then X to target
        # CH|10⟩ = |1⟩⊗H|0⟩ = |1⟩⊗|+⟩ → then CX flips target
        # Full: X on q0 → CH(q0,q1) → CX(q0,q1)
        X = GATE_SPECS["X"].matrix_fn()
        CH = GATE_SPECS["CH"].matrix_fn()
        CX = GATE_SPECS["CX"].matrix_fn()
        state = tensor_product(identity(2), X) @ all_zeros_state(2)
        state = CH @ state
        expected = CX @ state
        assert statevectors_equal(sv, expected)

    def test_controlled_power_2(self):
        """Controlled P(θ) with power=2 transpiles."""

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

        _, circ = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 4})
        assert circ.qubit_count == 2

    def test_controlled_power_4(self):
        """Controlled P(θ) with power=4 transpiles."""

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

        _, circ = _transpile_and_get_circuit(circuit, bindings={"theta": np.pi / 4})
        assert circ.qubit_count == 2

    def test_controlled_power_2_statevector(self):
        """Controlled P(θ)^2 = CP(2θ): statevector on |1+⟩ verified."""

        @qmc.qkernel
        def p_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            q = qmc.p(q, theta)
            return q

        cp_pow = qmc.controlled(p_gate)

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            q[1] = qmc.x(q[1])  # target = |1⟩ to see phase
            q[0], q[1] = cp_pow(q[0], q[1], power=2, theta=theta)
            return qmc.measure(q)

        theta = np.pi / 4
        _, circ = _transpile_and_get_circuit(circuit, bindings={"theta": theta})
        sv = _run_statevector(circ)
        # Initial state: H|0⟩ ⊗ X|0⟩ = |+⟩⊗|1⟩
        # Then controlled-P^2 = apply P(θ) twice when ctrl=|1⟩
        # P(θ)^2 |1⟩ = e^{2iθ}|1⟩
        H = GATE_SPECS["H"].matrix_fn()
        X = GATE_SPECS["X"].matrix_fn()
        # QuriParts P is emitted as RZ; CP is decomposed
        # Build expected via: |+⟩|1⟩ → when ctrl=1, apply P(θ)^2 to target
        # CP(2θ) matrix (equivalent to power=2 of CP(θ))
        CP = GATE_SPECS["CP"].matrix_fn(2 * theta)
        state = tensor_product(X, H) @ all_zeros_state(2)
        expected = CP @ state
        assert statevectors_equal(sv, expected)

    def test_controlled_multi_gate_double_control_raises(self):
        """controlled(H·X, num_controls=2) raises EmitError on QuriParts."""
        from qamomile.circuit.transpiler.errors import EmitError

        @qmc.qkernel
        def hx_gate(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            q = qmc.x(q)
            return q

        controlled_hx2 = qmc.controlled(hx_gate, num_controls=2)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.x(q[0])
            q[1] = qmc.x(q[1])
            q[0], q[1], q[2] = controlled_hx2(q[0], q[1], q[2])
            return qmc.measure(q)

        with pytest.raises(EmitError, match="multi-controlled"):
            _transpile_and_get_circuit(circuit)


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


class TestManualQFTCircuit:
    """Textbook QFT from scratch using CP + H + SWAP (Nielsen & Chuang).

    In QuriParts, each CP gate is decomposed into 5 gates: 3×RZ + 2×CNOT.
    Gate count assertions are adjusted accordingly.
    """

    def test_manual_qft_2q_gate_counts(self):
        """2-qubit manual QFT gate counts with CP decomposition.

        Qiskit: 2H + 1CP + 1SWAP
        QuriParts: 2H + (3RZ + 2CNOT) + 1SWAP = 2H + 3RZ + 2CNOT + 1SWAP
        """

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

        _, circ = _transpile_and_get_circuit(qft2)
        gates = _get_gates(circ)
        h_gates = [g for g in gates if g.name == gate_names.H]
        rz_gates = [g for g in gates if g.name == gate_names.RZ]
        cx_gates = [g for g in gates if g.name == gate_names.CNOT]
        swap_gates = [g for g in gates if g.name == gate_names.SWAP]
        assert len(h_gates) == 2
        # CP decomposed: 3 RZ + 2 CNOT per CP, 1 CP total
        assert len(rz_gates) == 3
        assert len(cx_gates) == 2
        assert len(swap_gates) == 1
        # H gates target qubits 0 and 1
        assert h_gates[0].target_indices == (0,)
        assert h_gates[1].target_indices == (1,)
        # SWAP operates on qubits 0 and 1
        assert swap_gates[0].target_indices == (0, 1)

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

        _, circ = _transpile_and_get_circuit(qft2)
        sv = _run_statevector(circ)
        expected = np.array([1, 1, 1, 1], dtype=complex) / 2
        assert statevectors_equal(sv, expected)

    def test_manual_qft_2q_from_one(self):
        """QFT|01⟩ — verify statevector via global-phase-tolerant comparison."""

        @qmc.qkernel
        def qft2_one() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])  # Prepare |01⟩ (q0=1, q1=0)
            q[0] = qmc.h(q[0])
            q[0], q[1] = qmc.cp(q[0], q[1], np.pi / 2)
            q[1] = qmc.h(q[1])
            q[0], q[1] = qmc.swap(q[0], q[1])
            return qmc.measure(q)

        _, circ = _transpile_and_get_circuit(qft2_one)
        sv = _run_statevector(circ)
        # Input: X on q0 → |01⟩.
        # After QFT+SWAP, expected = [1, 1, -1, -1]/2
        expected = np.array([1, 1, -1, -1], dtype=complex) / 2
        assert statevectors_equal(sv, expected)

    def test_manual_qft_3q_gate_counts(self):
        """3-qubit manual QFT gate counts with CP decomposition.

        Qiskit: 3H + 3CP + 1SWAP
        QuriParts: 3H + 3×(3RZ + 2CNOT) + 1SWAP = 3H + 9RZ + 6CNOT + 1SWAP
        """

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

        _, circ = _transpile_and_get_circuit(qft3)
        gates = _get_gates(circ)
        h_gates = [g for g in gates if g.name == gate_names.H]
        rz_gates = [g for g in gates if g.name == gate_names.RZ]
        cx_gates = [g for g in gates if g.name == gate_names.CNOT]
        swap_gates = [g for g in gates if g.name == gate_names.SWAP]
        assert len(h_gates) == 3
        # 3 CP gates, each decomposed to 3 RZ + 2 CNOT
        assert len(rz_gates) == 9
        assert len(cx_gates) == 6
        assert len(swap_gates) == 1
        # H gates target qubits 0, 1, 2
        for i, g in enumerate(h_gates):
            assert g.target_indices == (i,)
        # SWAP operates on qubits 0 and 2 (bit-reversal)
        assert swap_gates[0].target_indices == (0, 2)


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


# ============================================================================
# 15. Qubit Array Pattern Tests
# ============================================================================


class TestQubitArrayPatterns:
    """Test qubit_array-based circuit patterns (pure gate circuits only).

    Group 2+ tests (if-else with qubit_array) are not portable to QuriParts
    because they require mid-circuit measurement and native if_else support.
    """

    def test_cx_propagation_array(self):
        """X(q[0]) + CX → |11⟩ on qubit_array(2)."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            q[0], q[1] = qmc.cx(q[0], q[1])
            return qmc.measure(q)

        _, circ = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(circ)
        expected = computational_basis_state(2, 3)
        assert statevectors_equal(sv, expected)
        assert circ.qubit_count == 2

    def test_independent_rotations_array(self):
        """RX(θ₁) on q[0], RY(θ₂) on q[1] via qubit_array(2)."""

        @qmc.qkernel
        def circuit(theta1: qmc.Float, theta2: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.rx(q[0], theta1)
            q[1] = qmc.ry(q[1], theta2)
            return qmc.measure(q)

        bindings = {"theta1": np.pi / 3, "theta2": np.pi / 5}
        _, circ = _transpile_and_get_circuit(circuit, bindings=bindings)
        sv = _run_statevector(circ)
        RX = GATE_SPECS["RX"].matrix_fn(np.pi / 3)
        RY = GATE_SPECS["RY"].matrix_fn(np.pi / 5)
        expected = tensor_product(RY, RX) @ all_zeros_state(2)
        assert statevectors_equal(sv, expected)
        assert circ.qubit_count == 2

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

        _, circ = _transpile_and_get_circuit(
            circuit, bindings={"theta1": t1, "theta2": t2}
        )
        sv = _run_statevector(circ)
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

        _, circ = _transpile_and_get_circuit(circuit)
        sv = _run_statevector(circ)
        # Manual: H⊗3 · CZ(0,1) · CZ(1,2) |000⟩
        H = GATE_SPECS["H"].matrix_fn()
        CZ = GATE_SPECS["CZ"].matrix_fn()
        h3 = tensor_product(H, tensor_product(H, H))
        cz01 = tensor_product(identity(2), CZ)
        cz12 = tensor_product(CZ, identity(2))
        expected = cz12 @ cz01 @ h3 @ all_zeros_state(3)
        assert statevectors_equal(sv, expected)
        assert circ.qubit_count == 3

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
        _, circ = _transpile_and_get_circuit(circuit, bindings=angles)
        sv = _run_statevector(circ)
        RX = GATE_SPECS["RX"].matrix_fn(np.pi / 4)
        RY = GATE_SPECS["RY"].matrix_fn(np.pi / 3)
        RZ = GATE_SPECS["RZ"].matrix_fn(np.pi / 6)
        expected = tensor_product(RZ, tensor_product(RY, RX)) @ all_zeros_state(3)
        assert statevectors_equal(sv, expected)
        assert circ.qubit_count == 3

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_three_qubit_rotation_layer_array_random(self, seed):
        """Random RX(θ₀) ⊗ RY(θ₁) ⊗ RZ(θ₂) on qubit_array(3)."""
        rng = np.random.default_rng(seed)
        t0, t1, t2 = rng.uniform(0, 2 * np.pi, 3)

        @qmc.qkernel
        def circuit(a0: qmc.Float, a1: qmc.Float, a2: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, "q")
            q[0] = qmc.rx(q[0], a0)
            q[1] = qmc.ry(q[1], a1)
            q[2] = qmc.rz(q[2], a2)
            return qmc.measure(q)

        _, circ = _transpile_and_get_circuit(
            circuit, bindings={"a0": t0, "a1": t1, "a2": t2}
        )
        sv = _run_statevector(circ)
        rx_mat = GATE_SPECS["RX"].matrix_fn(t0)
        ry_mat = GATE_SPECS["RY"].matrix_fn(t1)
        rz_mat = GATE_SPECS["RZ"].matrix_fn(t2)
        expected = tensor_product(
            rz_mat, tensor_product(ry_mat, rx_mat)
        ) @ all_zeros_state(3)
        assert statevectors_equal(sv, expected)

    # -- Group 3: Parity Check Pattern ----------------------------------------

    @pytest.mark.parametrize(
        "case",
        [
            "even",
            "odd",
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

        _, circ = _transpile_and_get_circuit(kernel)
        sv = _run_statevector(circ)
        assert np.isclose(abs(sv[expected_idx]), 1.0, atol=1e-10)
        assert circ.qubit_count == 4

    # -- Group 4: Parametric vs Hardcoded Array Size ---------------------------

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

        _, circ_par = _transpile_and_get_circuit(h_parametric, bindings={"n": 3})
        _, circ_hc = _transpile_and_get_circuit(h_hardcoded)
        sv_par = _run_statevector(circ_par)
        sv_hc = _run_statevector(circ_hc)
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

        _, circ_par = _transpile_and_get_circuit(ghz_parametric, bindings={"n": 3})
        _, circ_hc = _transpile_and_get_circuit(ghz_hardcoded)
        sv_par = _run_statevector(circ_par)
        sv_hc = _run_statevector(circ_hc)
        assert statevectors_equal(sv_par, sv_hc)
        # Both should be 3-qubit GHZ
        assert np.isclose(abs(sv_par[0]), 1.0 / np.sqrt(2), atol=1e-10)
        assert np.isclose(abs(sv_par[7]), 1.0 / np.sqrt(2), atol=1e-10)


# ============================================================================
# 30. Compile-time constant if with array quantum phi output
# ============================================================================


class TestCompileTimeIfArrayQuantumPhi:
    """Compile-time constant if with array quantum phi must not raise EmitError."""

    def test_dead_branch_different_array(self):
        """Dead branch rebinds qubit array to a different array."""
        flag = True

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            if flag:
                q[0] = qmc.x(q[0])
            else:
                alt = qmc.qubit_array(2, "alt")
                alt[1] = qmc.x(alt[1])
                q = alt
            return qmc.measure(q)

        _, circ = _transpile_and_get_circuit(circuit)
        assert circ is not None


# ============================================================================
# Compile-time if phi propagation (Issue: compile_time_constant_if_phi_propagation)
# ============================================================================


class TestCompileTimeIfPhiPropagation:
    """Compile-time IfOperation lowering before SeparatePass.

    Tests that compile-time resolvable IfOperations (including expression-
    derived conditions) are lowered before separation.
    """

    def test_direct_classical_if_after_qinit(self):
        """Direct ``if flag:`` after quantum init should not raise."""

        @qmc.qkernel
        def circuit(flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            theta = qmc.float_(0.1)
            if flag:
                theta = qmc.float_(1.0)
            else:
                theta = qmc.float_(2.0)
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        _, circ = _transpile_and_get_circuit(circuit, bindings={"flag": 1})
        assert circ is not None

    def test_comparison_derived_classical_if_after_qinit(self):
        """``if flag > 0:`` after quantum init should not raise."""

        @qmc.qkernel
        def circuit(flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            theta = qmc.float_(0.1)
            if flag > 0:
                theta = qmc.float_(1.0)
            else:
                theta = qmc.float_(2.0)
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        _, circ = _transpile_and_get_circuit(circuit, bindings={"flag": 1})
        assert circ is not None

    def test_comparison_derived_classical_if_flag_zero(self):
        """``if flag > 0:`` with flag=0 selects false branch."""

        @qmc.qkernel
        def circuit(flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            theta = qmc.float_(0.1)
            if flag > 0:
                theta = qmc.float_(1.0)
            else:
                theta = qmc.float_(2.0)
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        _, circ = _transpile_and_get_circuit(circuit, bindings={"flag": 0})
        assert circ is not None

    def test_symbolic_parameter_alias_before_qinit(self):
        """Symbolic parameter through compile-time if should be preserved."""
        FLAG = True

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Bit:
            angle = qmc.float_(0.5)
            if FLAG:
                angle = theta
            else:
                angle = qmc.float_(0.5)
            q = qmc.qubit("q")
            q = qmc.rx(q, angle)
            return qmc.measure(q)

        _, circ = _transpile_and_get_circuit(circuit, parameters=["theta"])
        assert circ is not None


# ============================================================================
# 31. Direct cast -> measure (cast element carrier identity)
# ============================================================================


class TestDirectCastMeasure:
    """Verify that direct Vector[Qubit] -> QFixed -> measure resolves carriers.

    QURI Parts measurement is a no-op (handled by sampler), so we verify
    the clbit_map has the expected number of entries.
    """

    def test_direct_cast_measure_resolves_carriers(self):
        """Direct cast then measure must resolve num_bits carrier qubits."""

        @qmc.qkernel
        def circuit() -> qmc.Float:
            q = qmc.qubit_array(2, "q")
            qf = qmc.cast(q, qmc.QFixed, int_bits=0)
            return qmc.measure(qf)

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(circuit)
        circ = exe.compiled_quantum[0].circuit
        clbit_map = exe.compiled_quantum[0].clbit_map
        assert circ.qubit_count == 2
        assert len(clbit_map) == 2, (
            f"Expected 2 clbit_map entries, got {len(clbit_map)}"
        )

    def test_cast_after_gate_measure(self):
        """Gate before cast must not break carrier resolution."""

        @qmc.qkernel
        def circuit() -> qmc.Float:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.h(q[0])
            qf = qmc.cast(q, qmc.QFixed, int_bits=0)
            return qmc.measure(qf)

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(circuit)
        circ = exe.compiled_quantum[0].circuit
        clbit_map = exe.compiled_quantum[0].clbit_map
        gate_names_list = _gate_names(circ)
        assert any(n.lower() == "h" for n in gate_names_list)
        assert len(clbit_map) == 2, (
            f"Expected 2 clbit_map entries, got {len(clbit_map)}"
        )


# ============================================================================
# Unresolved non-measurement if-condition rejection
# ============================================================================


class TestUnresolvedNonMeasurementIfRejection:
    """Non-measurement unresolved if-conditions must raise EmitError.

    When a kernel argument is used as an if-condition and left in
    ``parameters`` (not bound), the transpiler must raise an explicit
    error instead of silently dropping the branch.
    """

    def test_unresolved_flag_parameter_raises(self):
        """``if flag:`` with parameters=["flag", "theta"] must raise EmitError."""

        @qmc.qkernel
        def circuit(flag: qmc.UInt, theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            if flag:
                q = qmc.rx(q, theta)
            return qmc.measure(q)

        with pytest.raises(EmitError, match="measurement results"):
            _transpile_and_get_circuit(circuit, parameters=["flag", "theta"])

    def test_unresolved_flag_only_raises(self):
        """``if flag:`` with parameters=["flag"] must raise EmitError."""

        @qmc.qkernel
        def circuit(flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            if flag:
                q = qmc.x(q)
            return qmc.measure(q)

        with pytest.raises(EmitError, match="measurement results"):
            _transpile_and_get_circuit(circuit, parameters=["flag"])

    def test_bound_flag_still_works(self):
        """``if flag:`` with bindings={"flag": 1} must still work."""

        @qmc.qkernel
        def circuit(flag: qmc.UInt) -> qmc.Bit:
            q = qmc.qubit("q")
            if flag:
                q = qmc.x(q)
            return qmc.measure(q)

        _, qc = _transpile_and_get_circuit(circuit, bindings={"flag": 1})
        gate_names_list = _gate_names(qc)
        assert "X" in gate_names_list


class TestUnresolvedStructuralSize:
    """Unresolved structural UInt must raise, not produce zero-size artifact."""

    def test_qubit_array_with_unresolved_size_raises(self):
        """qubit_array(n) with parameters=["n"] must raise EmitError."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for i in qmc.range(n):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        transpiler = QuriPartsTranspiler()
        with pytest.raises((EmitError, ValueError)):
            transpiler.transpile(circuit, parameters=["n"])


# ============================================================================
# 32. Portable TranspilerConfig Tests
# ============================================================================
