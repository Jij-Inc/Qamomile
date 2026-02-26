"""Minimal essential QURI Parts frontend-to-backend test suite.

Tests the full pipeline: @qkernel definition -> QuriPartsCircuitTranspiler -> execution.
Covers individual gates, gate combinations, transpiler passes, parametric circuits,
basic execution, and edge cases.

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
# Skip entire module if quri-parts is not installed
# ---------------------------------------------------------------------------
pytest.importorskip("quri_parts")
pytest.importorskip("quri_parts.qulacs")

from qamomile.quri_parts import QuriPartsCircuitTranspiler
from qamomile.circuit.transpiler.executable import ExecutableProgram
from qamomile.circuit.transpiler.segments import SimplifiedProgram
from qamomile.circuit.ir.block import BlockKind


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


# ============================================================================
# 1. Individual Gate Statevector Tests
# ============================================================================


class TestSingleQubitGatesFrontend:
    """Test each single-qubit gate through the full frontend pipeline."""

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


class TestTwoQubitGatesFrontend:
    """Test each two-qubit gate through the full frontend pipeline."""

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
