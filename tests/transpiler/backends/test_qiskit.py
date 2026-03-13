"""Qiskit transpiler test configuration.

This module configures the transpiler test suite for the Qiskit backend.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

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
