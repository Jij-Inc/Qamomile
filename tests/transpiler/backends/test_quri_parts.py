"""QuriParts transpiler test configuration.

This module configures the transpiler test suite for the QuriParts backend.
"""

from __future__ import annotations

from typing import Any

import pytest
import numpy as np

from tests.transpiler.base_test import TranspilerTestSuite, HamiltonianTestMixin


class TestQuriPartsTranspiler(TranspilerTestSuite, HamiltonianTestMixin):
    """Test suite for QuriParts transpiler.

    QuriParts supports most standard gates but has some limitations:
    - Measurements are not supported in circuits (no-op in emitter)
    - Control flow is not natively supported (loops are unrolled)
    - Controlled gates (CH, CY, CRX, CRY, CRZ, CP) are decomposed
    """

    backend_name = "quri_parts"
    # MEASURE is a no-op in QURI Parts
    unsupported_gates: set[str] = {"MEASURE"}

    @classmethod
    def get_emitter(cls) -> Any:
        """Get QuriParts GateEmitter instance."""
        from qamomile.quri_parts.emitter import QuriPartsGateEmitter

        return QuriPartsGateEmitter()

    @classmethod
    def get_transpiler(cls) -> Any:
        """Get QuriParts transpiler instance."""
        from qamomile.quri_parts import QuriPartsTranspiler

        return QuriPartsTranspiler()

    @classmethod
    def get_simulator(cls) -> Any:
        """Get Qulacs vector simulator."""
        from quri_parts.qulacs.simulator import evaluate_state_to_vector

        return evaluate_state_to_vector

    @classmethod
    def run_circuit_statevector(cls, circuit: Any) -> np.ndarray:
        """Run circuit and extract statevector using Qulacs.

        For parametric circuits with no parameters, we bind with empty list.
        Then use the bound circuit for simulation.
        """
        from quri_parts.core.state import quantum_state, apply_circuit
        from quri_parts.qulacs.simulator import evaluate_state_to_vector

        # If it's a parametric circuit, bind with empty parameters
        # (it has no unbound parameters from our emitter tests)
        if hasattr(circuit, "parameter_count") and circuit.parameter_count > 0:
            # Circuit has parameters - bind them with zeros (shouldn't happen in these tests)
            bound_circuit = circuit.bind_parameters([0.0] * circuit.parameter_count)
        elif hasattr(circuit, "bind_parameters"):
            # It's a parametric circuit but with 0 parameters - still need to bind
            bound_circuit = circuit.bind_parameters([])
        else:
            bound_circuit = circuit

        # Create initial state |0...0>
        cb_state = quantum_state(bound_circuit.qubit_count, bits=0)

        # Apply circuit - use GeneralCircuitQuantumState for bound circuit
        from quri_parts.core.state import GeneralCircuitQuantumState

        circuit_state = GeneralCircuitQuantumState(bound_circuit.qubit_count, bound_circuit)

        # Get statevector
        statevector = evaluate_state_to_vector(circuit_state)

        return np.array(statevector.vector)
