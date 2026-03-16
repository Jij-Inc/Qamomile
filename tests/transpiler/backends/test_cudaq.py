"""CUDA-Q transpiler test configuration.

This module configures the transpiler test suite for the CUDA-Q backend.
Tests are automatically skipped if the ``cudaq`` package is not installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

cudaq = pytest.importorskip("cudaq")

from tests.transpiler.base_test import TranspilerTestSuite  # noqa: E402


class TestCudaqTranspiler(TranspilerTestSuite):
    """Test suite for CUDA-Q transpiler.

    CUDA-Q supports most standard gates but has some limitations:
    - Measurements are not supported in circuits (no-op in emitter)
    - c_if (if-then, no else) is supported; while-loops raise EmitError
    - CP and RZZ are decomposed (no native gates)
    - CH and CY are decomposed
    """

    backend_name = "cudaq"
    unsupported_gates: set[str] = {"MEASURE"}

    @classmethod
    def get_emitter(cls) -> Any:
        """Get CUDA-Q GateEmitter instance."""
        from qamomile.cudaq.emitter import CudaqGateEmitter

        return CudaqGateEmitter()

    @classmethod
    def get_simulator(cls) -> Any:
        """Get CUDA-Q simulator (not directly used)."""
        return None

    @classmethod
    def run_circuit_statevector(cls, circuit: Any) -> np.ndarray:
        """Run circuit and extract statevector using CUDA-Q."""
        import cudaq

        state = cudaq.get_state(circuit.kernel)
        return np.array(state)


class TestCudaqControlFlowErrors:
    """Test that unsupported control flow raises explicit errors."""

    def test_c_if_transpiles_ok(self) -> None:
        """c_if (if-then, no else) should transpile without error."""
        import qamomile.circuit as qm
        from qamomile.cudaq import CudaqTranspiler

        @qm.qkernel
        def circuit_with_c_if(q0: qm.Qubit, q1: qm.Qubit) -> qm.Bit:
            q0 = qm.x(q0)
            b = qm.measure(q0)
            if b:
                q1 = qm.x(q1)
            return qm.measure(q1)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(circuit_with_c_if)
        assert exe.circuit.num_qubits == 2

    def test_if_with_else_raises_emit_error(self) -> None:
        """IfOperation with else branch on CUDA-Q must raise EmitError."""
        import qamomile.circuit as qm
        from qamomile.circuit.transpiler.errors import EmitError
        from qamomile.cudaq import CudaqTranspiler

        @qm.qkernel
        def circuit_with_if_else(
            q0: qm.Qubit,
            q1: qm.Qubit,
        ) -> qm.Bit:
            q0 = qm.h(q0)
            b = qm.measure(q0)
            if b:
                q1 = qm.x(q1)
            else:
                q1 = qm.h(q1)
            return qm.measure(q1)

        transpiler = CudaqTranspiler()
        with pytest.raises(EmitError, match="does not support else"):
            transpiler.transpile(circuit_with_if_else)

    def test_while_loop_raises_emit_error(self) -> None:
        """WhileOperation on CUDA-Q backend must raise EmitError."""
        import qamomile.circuit as qm
        from qamomile.circuit.transpiler.errors import EmitError
        from qamomile.cudaq import CudaqTranspiler

        @qm.qkernel
        def _while_body(q: qm.Qubit) -> tuple[qm.Qubit, qm.Bit]:
            q = qm.x(q)
            return q, qm.measure(q)

        @qm.qkernel
        def circuit_with_while(q: qm.Qubit) -> qm.Bit:
            b = qm.measure(q)
            q, b = qm.while_loop(b, _while_body, q)
            return b

        transpiler = CudaqTranspiler()
        with pytest.raises(EmitError, match="while loop control flow"):
            transpiler.transpile(circuit_with_while)
