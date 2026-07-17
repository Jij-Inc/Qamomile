"""Qiskit transpiler test configuration.

This module configures the transpiler test suite for the Qiskit backend.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import qamomile.circuit as qmc
from tests.transpiler.base_test import TranspilerTestSuite


class TestQiskitTranspiler(TranspilerTestSuite):
    """Test suite for Qiskit transpiler.

    Tests the QiskitGateEmitter from qamomile.circuit against expected
    quantum gate behaviors using AerSimulator for statevector verification.

    Note: Some gates (CH) are not directly supported by AerSimulator's
    statevector method and need to be transpiled to basis gates first.
    """

    backend_name = "qiskit"
    # CH is not directly supported by AerSimulator statevector method
    unsupported_gates: set[str] = {"CH"}

    @classmethod
    def get_emitter(cls) -> Any:
        """Get Qiskit GateEmitter instance."""
        from qamomile.qiskit.emitter import QiskitGateEmitter

        return QiskitGateEmitter()

    @classmethod
    def get_simulator(cls) -> Any:
        """Get Qiskit statevector simulator."""
        from qiskit_aer import AerSimulator

        return AerSimulator(method="statevector")

    def test_global_phase_uses_native_circuit_metadata(self) -> None:
        """The compatibility emitter preserves a standalone phase exactly."""
        emitter = self.get_emitter()
        circuit = emitter.create_circuit(0, 0)

        emitter.emit_global_phase(circuit, 0.25)

        assert float(circuit.global_phase) == pytest.approx(0.25)

    def test_default_executor_does_not_force_single_thread(self) -> None:
        """The default Aer backend leaves parallelism under Aer control."""
        from qamomile.qiskit import QiskitExecutor

        executor = QiskitExecutor()

        assert executor.backend.options.max_parallel_threads != 1

    def test_executor_decomposes_nested_empty_multiplexer_without_unrolling(
        self,
    ) -> None:
        """Aer workaround decomposes poisoned multiplexers inside for_loop blocks."""
        import math

        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import StatePreparation
        from qiskit_aer import AerSimulator

        from qamomile.qiskit import QiskitExecutor
        from qamomile.qiskit.transpiler import (
            _contains_empty_parameter_multiplexer,
            _decompose_empty_parameter_multiplexers,
        )

        backend = AerSimulator()
        amplitudes = [0.5, math.sqrt(0.75)]
        body = QuantumCircuit(1, 1)
        body.append(StatePreparation(amplitudes).inverse(), [0])
        circuit = QuantumCircuit(1, 1)
        circuit.for_loop(range(2), None, body, [0], [0])
        circuit.measure(0, 0)

        transpiled = transpile(circuit, backend)
        fixed = _decompose_empty_parameter_multiplexers(transpiled, backend)

        assert any(inst.operation.name == "for_loop" for inst in fixed.data)
        assert not _contains_empty_parameter_multiplexer(fixed)

        counts = QiskitExecutor(backend).execute(circuit, shots=8)
        assert sum(counts.values()) == 8

    def test_executor_leaves_parameterized_multiplexer_untouched(self) -> None:
        """Aer workaround does not treat valid params-bearing multiplexers as unsafe."""
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import UCGate
        from qiskit_aer import AerSimulator

        from qamomile.qiskit.transpiler import (
            _contains_empty_parameter_multiplexer,
            _decompose_empty_parameter_multiplexers,
        )

        backend = AerSimulator()
        circuit = QuantumCircuit(2, 2)
        circuit.append(UCGate([np.eye(2), np.array([[0, 1], [1, 0]])]), [0, 1])
        circuit.measure([0, 1], [0, 1])

        transpiled = transpile(circuit, backend)
        fixed = _decompose_empty_parameter_multiplexers(transpiled, backend)

        multiplexers = [
            instruction.operation
            for instruction in fixed.data
            if instruction.operation.name == "multiplexer"
        ]
        assert len(multiplexers) == 1
        assert len(multiplexers[0].params) == 2
        assert fixed is transpiled
        assert not _contains_empty_parameter_multiplexer(fixed)
        assert backend.run(fixed, shots=8).result().get_counts() == {"00": 8}

    def test_bare_loop_variable_angle_is_internal_to_public_abi(self) -> None:
        """A native loop induction angle does not become a public parameter."""
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            """Rotate by each native loop induction value."""
            qubit = qmc.qubit("qubit")
            for index in qmc.range(3):
                qubit = qmc.rz(qubit, index)
            return qmc.measure(qubit)

        executable = QiskitTranspiler().transpile(kernel)

        assert executable.parameter_names == []
        assert executable.get_first_circuit().num_parameters == 1

    @classmethod
    def run_circuit_statevector(cls, circuit: Any) -> np.ndarray:
        """Run circuit and extract statevector."""
        from qiskit_aer import AerSimulator

        # Save statevector
        circuit.save_statevector()

        # Run simulation
        simulator = AerSimulator(method="statevector")
        result = simulator.run(circuit).result()
        statevector = result.get_statevector()

        return np.array(statevector)
