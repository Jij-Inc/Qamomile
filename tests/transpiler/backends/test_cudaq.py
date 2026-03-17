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
    - Measurement-dependent conditional branching raises EmitError (0.14.x)
    - While-loops raise EmitError
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

    def test_c_if_raises_emit_error(self) -> None:
        """measurement-dependent c_if is rejected under CUDA-Q 0.14.x."""
        import qamomile.circuit as qmc
        from qamomile.circuit.transpiler.errors import EmitError
        from qamomile.cudaq import CudaqTranspiler

        @qmc.qkernel
        def circuit_with_c_if() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q0 = qmc.x(q0)
            b = qmc.measure(q0)
            if b:
                q1 = qmc.x(q1)
            return qmc.measure(q1)

        transpiler = CudaqTranspiler()
        with pytest.raises(EmitError, match="measurement-dependent"):
            transpiler.transpile(circuit_with_c_if)

    def test_if_with_else_raises_emit_error(self) -> None:
        """IfOperation with else branch on CUDA-Q must raise EmitError."""
        import qamomile.circuit as qmc
        from qamomile.circuit.transpiler.errors import EmitError
        from qamomile.cudaq import CudaqTranspiler

        @qmc.qkernel
        def circuit_with_if_else() -> qmc.Bit:
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            q0 = qmc.h(q0)
            b = qmc.measure(q0)
            if b:
                q1 = qmc.x(q1)
            else:
                q1 = qmc.h(q1)
            return qmc.measure(q1)

        transpiler = CudaqTranspiler()
        with pytest.raises(EmitError, match="measurement-dependent"):
            transpiler.transpile(circuit_with_if_else)

    def test_while_loop_raises_emit_error(self) -> None:
        """WhileOperation on CUDA-Q backend must raise EmitError."""
        import qamomile.circuit as qmc
        from qamomile.circuit.transpiler.errors import EmitError
        from qamomile.cudaq import CudaqTranspiler

        @qmc.qkernel
        def circuit_with_while() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            bit = qmc.measure(q)
            while bit:
                q = qmc.qubit("q2")
                q = qmc.h(q)
                bit = qmc.measure(q)
            return bit

        transpiler = CudaqTranspiler()
        with pytest.raises(EmitError, match="while loop control flow"):
            transpiler.transpile(circuit_with_while)
