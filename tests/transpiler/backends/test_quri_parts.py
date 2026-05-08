"""QURI Parts transpiler test configuration.

This module configures the transpiler test suite for the QURI Parts backend.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.quri_parts

# Skip entire module if QURI Parts with the Qulacs backend is not installed.
pytest.importorskip("quri_parts")
pytest.importorskip("quri_parts.qulacs")

from tests.transpiler.base_test import TranspilerTestSuite  # noqa: E402


class TestQuriPartsTranspiler(TranspilerTestSuite):
    """Test suite for QURI Parts transpiler.

    QURI Parts supports most standard gates but has some limitations:
    - Measurements are not supported in circuits (no-op in emitter)
    - Control flow is not natively supported (loops are unrolled)
    - Controlled gates (CH, CY, CRX, CRY, CRZ, CP) are decomposed
    """

    backend_name = "quri_parts"
    # MEASURE is a no-op in QURI Parts
    unsupported_gates: set[str] = {"MEASURE"}

    @classmethod
    def get_emitter(cls) -> Any:
        """Get QURI Parts GateEmitter instance."""
        from qamomile.quri_parts.emitter import QuriPartsGateEmitter

        return QuriPartsGateEmitter()

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
        from quri_parts.core.state import GeneralCircuitQuantumState
        from quri_parts.qulacs.simulator import evaluate_state_to_vector

        # If it's a parametric circuit, bind with empty parameters
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
