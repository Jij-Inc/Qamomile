"""Tests for QuriPartsCircuitTranspiler (new qamomile.circuit API)."""

import math
import pytest

from qamomile.quri_parts import (
    QuriPartsGateEmitter,
)


class TestQuriPartsGateEmitter:
    """Test the gate emitter directly."""

    def test_create_circuit(self):
        """Test circuit creation."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(3, 2)
        assert circuit.qubit_count == 3
        # Note: QURI Parts doesn't track classical bits in the circuit

    def test_single_qubit_gates(self):
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

    def test_rotation_gates_with_float(self):
        """Test rotation gates with float angles."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(1, 0)

        emitter.emit_rx(circuit, 0, math.pi / 2)
        emitter.emit_ry(circuit, 0, math.pi / 4)
        emitter.emit_rz(circuit, 0, math.pi / 8)

        gates = list(circuit.gates)
        assert len(gates) == 3

    def test_rotation_gates_with_parameter(self):
        """Test rotation gates with parametric angles."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(1, 0)

        param = emitter.create_parameter("theta")
        emitter.emit_rx(circuit, 0, param)
        emitter.emit_ry(circuit, 0, param)

        # Circuit should have parametric gates
        assert circuit.parameter_count == 1

    def test_two_qubit_gates(self):
        """Test two qubit gate emission."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(2, 0)

        emitter.emit_cx(circuit, 0, 1)
        emitter.emit_cz(circuit, 0, 1)
        emitter.emit_swap(circuit, 0, 1)

        gates = list(circuit.gates)
        assert len(gates) == 3

    def test_controlled_phase_decomposition(self):
        """Test CP gate decomposition."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(2, 0)

        emitter.emit_cp(circuit, 0, 1, math.pi / 2)

        # CP decomposes into 5 gates: RZ, CNOT, RZ, CNOT, RZ
        gates = list(circuit.gates)
        assert len(gates) == 5

    def test_rzz_gate(self):
        """Test RZZ gate emission."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(2, 0)

        emitter.emit_rzz(circuit, 0, 1, math.pi / 4)

        gates = list(circuit.gates)
        assert len(gates) == 1

    def test_toffoli_gate(self):
        """Test Toffoli gate emission."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(3, 0)

        emitter.emit_toffoli(circuit, 0, 1, 2)

        gates = list(circuit.gates)
        assert len(gates) == 1

    def test_controlled_single_qubit_decomposition(self):
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

    def test_measure_is_noop(self):
        """Test that measure is a no-op (doesn't add gates)."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(1, 1)

        emitter.emit_h(circuit, 0)
        emitter.emit_measure(circuit, 0, 0)

        gates = list(circuit.gates)
        # Only H gate, measure is no-op
        assert len(gates) == 1

    def test_barrier_is_noop(self):
        """Test that barrier is a no-op (doesn't add gates)."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(2, 0)

        emitter.emit_h(circuit, 0)
        emitter.emit_barrier(circuit, [0, 1])
        emitter.emit_h(circuit, 1)

        gates = list(circuit.gates)
        # Only 2 H gates, barrier is no-op
        assert len(gates) == 2

    def test_supports_no_native_control_flow(self):
        """Test that emitter reports no native control flow support."""
        emitter = QuriPartsGateEmitter()
        assert not emitter.supports_for_loop()
        assert not emitter.supports_if_else()
        assert not emitter.supports_while_loop()

    def test_circuit_to_gate_returns_none(self):
        """Test that circuit_to_gate returns None (not supported)."""
        emitter = QuriPartsGateEmitter()
        circuit = emitter.create_circuit(1, 0)
        emitter.emit_h(circuit, 0)

        gate = emitter.circuit_to_gate(circuit, "test")
        assert gate is None

    def test_gate_power_returns_none(self):
        """Test that gate_power returns None (not supported)."""
        emitter = QuriPartsGateEmitter()
        result = emitter.gate_power(None, 2)
        assert result is None

    def test_gate_controlled_returns_none(self):
        """Test that gate_controlled returns None (not supported)."""
        emitter = QuriPartsGateEmitter()
        result = emitter.gate_controlled(None, 1)
        assert result is None


class TestQuriPartsCircuitTranspiler:
    """Test the transpiler configuration."""

    def test_transpiler_creation(self):
        """Test transpiler can be created."""
        from qamomile.quri_parts import QuriPartsCircuitTranspiler

        transpiler = QuriPartsCircuitTranspiler()
        assert transpiler is not None

    def test_executor_creation(self):
        """Test executor can be created from transpiler."""
        from qamomile.quri_parts import QuriPartsCircuitTranspiler

        transpiler = QuriPartsCircuitTranspiler()
        executor = transpiler.executor()
        assert executor is not None

    def test_executor_with_custom_sampler(self):
        """Test executor with custom sampler."""
        from qamomile.quri_parts import QuriPartsCircuitTranspiler

        transpiler = QuriPartsCircuitTranspiler()
        # Custom sampler can be passed to executor
        executor = transpiler.executor(sampler=None, estimator=None)
        assert executor is not None
