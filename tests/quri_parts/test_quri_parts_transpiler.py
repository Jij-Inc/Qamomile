"""Tests for QuriPartsTranspiler (new qamomile.circuit API)."""

import math

import pytest

pytestmark = pytest.mark.quri_parts

# Skip entire module if QURI Parts circuit support is not installed.
pytest.importorskip("quri_parts.circuit")

import qamomile.circuit as qmc  # noqa: E402
from qamomile.quri_parts import (  # noqa: E402
    QuriPartsExecutor,
    QuriPartsGateEmitter,
)


class TestQuriPartsGateEmitter:
    """Test the gate emitter directly."""

    def test_create_circuit(self) -> None:
        """Test circuit creation."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(3, 2)
        assert circuit.qubit_count == 3
        # Note: QURI Parts doesn't track classical bits in the circuit

    def test_single_qubit_gates(self) -> None:
        """Test single qubit gate emission."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(2, 0)

        emitter.emit_h(circuit, 0)
        emitter.emit_x(circuit, 1)
        emitter.emit_y(circuit, 0)
        emitter.emit_z(circuit, 1)
        emitter.emit_s(circuit, 0)
        emitter.emit_t(circuit, 1)

        gates = list(circuit.gates)
        assert len(gates) == 6

    def test_rotation_gates_with_float(self) -> None:
        """Test rotation gates with float angles."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(1, 0)

        emitter.emit_rx(circuit, 0, math.pi / 2)
        emitter.emit_ry(circuit, 0, math.pi / 4)
        emitter.emit_rz(circuit, 0, math.pi / 8)

        gates = list(circuit.gates)
        assert len(gates) == 3

    def test_rotation_gates_with_parameter(self) -> None:
        """Test rotation gates with parametric angles."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(1, 0)

        param = emitter.create_parameter("theta")
        emitter.emit_rx(circuit, 0, param)
        emitter.emit_ry(circuit, 0, param)

        # Circuit should have parametric gates
        assert circuit.parameter_count == 1

    def test_two_qubit_gates(self) -> None:
        """Test two qubit gate emission."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(2, 0)

        emitter.emit_cx(circuit, 0, 1)
        emitter.emit_cz(circuit, 0, 1)
        emitter.emit_swap(circuit, 0, 1)

        gates = list(circuit.gates)
        assert len(gates) == 3

    def test_controlled_phase_decomposition(self) -> None:
        """Test CP gate decomposition."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(2, 0)

        emitter.emit_cp(circuit, 0, 1, math.pi / 2)

        # CP decomposes into 5 gates: RZ, CNOT, RZ, CNOT, RZ
        gates = list(circuit.gates)
        assert len(gates) == 5

    def test_rzz_gate(self) -> None:
        """Test RZZ gate emission."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(2, 0)

        emitter.emit_rzz(circuit, 0, 1, math.pi / 4)

        gates = list(circuit.gates)
        assert len(gates) == 1

    def test_toffoli_gate(self) -> None:
        """Test Toffoli gate emission."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(3, 0)

        emitter.emit_toffoli(circuit, 0, 1, 2)

        gates = list(circuit.gates)
        assert len(gates) == 1

    def test_controlled_single_qubit_decomposition(self) -> None:
        """Test controlled single-qubit gate decompositions."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(2, 0)

        emitter.emit_ch(circuit, 0, 1)
        emitter.emit_cy(circuit, 0, 1)
        emitter.emit_crx(circuit, 0, 1, math.pi / 2)
        emitter.emit_cry(circuit, 0, 1, math.pi / 2)
        emitter.emit_crz(circuit, 0, 1, math.pi / 2)

        # All controlled gates should decompose
        gates = list(circuit.gates)
        assert len(gates) > 5  # Each gate decomposes to multiple gates

    def test_measure_is_noop(self) -> None:
        """Test that measure is a no-op (doesn't add gates)."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(1, 1)

        emitter.emit_h(circuit, 0)
        emitter.emit_measure(circuit, 0, 0)

        gates = list(circuit.gates)
        # Only H gate, measure is no-op
        assert len(gates) == 1

    def test_barrier_is_noop(self) -> None:
        """Test that barrier is a no-op (doesn't add gates)."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(2, 0)

        emitter.emit_h(circuit, 0)
        emitter.emit_barrier(circuit, [0, 1])
        emitter.emit_h(circuit, 1)

        gates = list(circuit.gates)
        # Only 2 H gates, barrier is no-op
        assert len(gates) == 2

    def test_supports_no_native_control_flow(self) -> None:
        """Test that emitter reports no native control flow support."""
        emitter = QuriPartsGateEmitter()
        assert not emitter.supports_for_loop()
        assert not emitter.supports_if_else()
        assert not emitter.supports_while_loop()

    def test_circuit_to_gate_returns_none(self) -> None:
        """Test that circuit_to_gate returns None (not supported)."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(1, 0)
        emitter.emit_h(circuit, 0)

        gate = emitter.circuit_to_gate(circuit, "test")
        assert gate is None

    def test_gate_power_returns_none(self) -> None:
        """Test that gate_power returns None (not supported)."""
        emitter = QuriPartsGateEmitter()
        result = emitter.gate_power(None, 2)
        assert result is None

    def test_gate_controlled_returns_none(self) -> None:
        """Test that gate_controlled returns None (not supported)."""
        emitter = QuriPartsGateEmitter()
        result = emitter.gate_controlled(None, 1)
        assert result is None


class TestQuriPartsTranspiler:
    """Test the transpiler configuration."""

    def test_transpiler_creation(self) -> None:
        """Test transpiler can be created."""
        from qamomile.quri_parts import QuriPartsTranspiler

        transpiler = QuriPartsTranspiler()
        assert transpiler is not None

    def test_executor_creation(self) -> None:
        """Test executor can be created from transpiler."""
        from qamomile.quri_parts import QuriPartsTranspiler

        transpiler = QuriPartsTranspiler()
        executor = transpiler.executor()
        assert executor is not None

    def test_executor_with_custom_sampler(self) -> None:
        """Test executor with custom sampler."""
        from qamomile.quri_parts import QuriPartsTranspiler

        transpiler = QuriPartsTranspiler()
        # Custom sampler can be passed to executor
        executor = transpiler.executor(sampler=None, estimator=None)
        assert executor is not None

    def test_controlled_cp_with_runtime_parameter_transpiles(self) -> None:
        """Controlled CP fallback preserves a symbolic QURI Parts angle."""
        from qamomile.quri_parts import QuriPartsTranspiler

        @qmc.qkernel
        def cp_layer(
            control: qmc.Qubit,
            target: qmc.Qubit,
            theta: qmc.Float,
        ) -> tuple[qmc.Qubit, qmc.Qubit]:
            """Apply a symbolic controlled-phase body."""
            control, target = qmc.cp(control, target, theta)
            return control, target

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            """Apply a controlled CP body with a runtime angle."""
            qs = qmc.qubit_array(3, "qs")
            qs[0], qs[1], qs[2] = qmc.control(cp_layer)(
                qs[0],
                qs[1],
                qs[2],
                theta=theta,
            )
            return qmc.measure(qs)

        transpiler = QuriPartsTranspiler()
        executable = transpiler.transpile(circuit, parameters=["theta"])
        quri_circuit = executable.compiled_quantum[0].circuit

        assert quri_circuit.parameter_count == 1
        assert len(quri_circuit.gates) > 0

    def test_controlled_inverse_with_runtime_parameter_transpiles(self) -> None:
        """Controlled inverse fallback evaluates symbolic angle negation."""
        from qamomile.quri_parts import QuriPartsTranspiler

        @qmc.qkernel
        def phase_layer(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            """Apply a runtime phase rotation."""
            q = qmc.rz(q, theta)
            return q

        @qmc.qkernel
        def inverse_body(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            """Apply an inverse runtime phase inside a controlled body."""
            q = qmc.inverse(phase_layer)(q, theta)
            return q

        @qmc.qkernel
        def circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
            """Apply a controlled inverse body with a runtime angle."""
            qs = qmc.qubit_array(2, "qs")
            qs[0] = qmc.x(qs[0])
            controlled = qmc.control(inverse_body)
            qs[0], qs[1] = controlled(qs[0], qs[1], theta)
            return qmc.measure(qs)

        transpiler = QuriPartsTranspiler()
        executable = transpiler.transpile(circuit, parameters=["theta"])
        quri_circuit = executable.compiled_quantum[0].circuit

        assert quri_circuit.parameter_count == 1
        assert len(quri_circuit.gates) > 0

    def test_controlled_toffoli_cascade_path_executes(self) -> None:
        """Controlled-Toffoli cascade flips only the intended target qubit."""
        pytest.importorskip("quri_parts.qulacs")

        from qamomile.quri_parts import QuriPartsTranspiler

        @qmc.qkernel
        def toffoli_layer(
            left: qmc.Qubit,
            right: qmc.Qubit,
            target: qmc.Qubit,
        ) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
            """Apply a Toffoli gate inside a controlled body."""
            left, right, target = qmc.ccx(left, right, target)
            return left, right, target

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            """Prepare three controls and apply a controlled Toffoli."""
            qs = qmc.qubit_array(4, "qs")
            qs[0] = qmc.x(qs[0])
            qs[1] = qmc.x(qs[1])
            qs[2] = qmc.x(qs[2])
            qs[0], qs[1], qs[2], qs[3] = qmc.control(toffoli_layer)(
                qs[0],
                qs[1],
                qs[2],
                qs[3],
            )
            return qmc.measure(qs)

        transpiler = QuriPartsTranspiler()
        executable = transpiler.transpile(circuit)
        result = executable.sample(QuriPartsExecutor(seed=123), shots=16).result()

        assert result.results == [((True, True, True, True), 16)]
