"""QURI Parts transpiler test configuration.

This module configures the transpiler test suite for the QURI Parts backend.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tests.transpiler.base_test import TranspilerTestSuite


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


# ---------------------------------------------------------------------------
# QuriPartsParamExpr numpy scalar regression tests
# ---------------------------------------------------------------------------


class TestQuriPartsParamExprNumpyScalars:
    """Verify QuriPartsParamExpr arithmetic accepts numpy scalar types.

    numpy scalars like np.float32 and np.int32 are not subclasses of
    Python float/int (except np.float64). The emitter must accept them
    via is_concrete_real_number (numbers.Real) rather than isinstance(..., (int, float)).
    """

    @staticmethod
    def _make_expr() -> Any:
        """Create a QuriPartsParamExpr with a dummy parameter term."""
        from qamomile.quri_parts.emitter import QuriPartsParamExpr

        # Use a string as a stand-in for a Parameter object; arithmetic
        # only needs dict key identity, not QURI Parts runtime.
        return QuriPartsParamExpr(terms={"p": 1.0}, const=0.0)

    def test_coerce_constant_np_float32(self) -> None:
        from qamomile.quri_parts.emitter import QuriPartsParamExpr

        result = QuriPartsParamExpr._coerce_constant(np.float32(1.5))
        assert result == 1.5
        assert isinstance(result, float)

    def test_coerce_constant_np_int32(self) -> None:
        from qamomile.quri_parts.emitter import QuriPartsParamExpr

        result = QuriPartsParamExpr._coerce_constant(np.int32(3))
        assert result == 3.0
        assert isinstance(result, float)

    def test_mul_np_float32(self) -> None:
        expr = self._make_expr()
        result = expr * np.float32(2.0)
        assert result.terms["p"] == 2.0
        assert result.const == 0.0

    def test_add_np_float32(self) -> None:
        expr = self._make_expr()
        result = expr + np.float32(0.5)
        assert result.const == 0.5
        assert result.terms["p"] == 1.0

    def test_sub_np_int32(self) -> None:
        expr = self._make_expr()
        result = expr - np.int32(1)
        assert result.const == -1.0

    def test_rsub_np_float32(self) -> None:
        expr = self._make_expr()
        result = np.float32(3.0) - expr
        assert result.const == 3.0
        assert result.terms["p"] == -1.0

    def test_make_angle_dict_np_float32(self) -> None:
        from qamomile.quri_parts.emitter import QuriPartsGateEmitter

        emitter = QuriPartsGateEmitter()
        result = emitter._make_angle_dict(np.float32(1.5))
        assert result == 1.5
        assert isinstance(result, float)
