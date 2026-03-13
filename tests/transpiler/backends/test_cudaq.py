"""CUDA-Q transpiler test configuration.

This module configures the transpiler test suite for the CUDA-Q backend.
Tests are automatically skipped if the ``cudaq`` package is not installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

cudaq = pytest.importorskip("cudaq")

from tests.transpiler.base_test import TranspilerTestSuite


class TestCudaqTranspiler(TranspilerTestSuite):
    """Test suite for CUDA-Q transpiler.

    CUDA-Q supports most standard gates but has some limitations:
    - Measurements are not supported in circuits (no-op in emitter)
    - Control flow (if/while) is not natively supported
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

    def test_if_else_raises_not_implemented(self) -> None:
        """IfOperation on CUDA-Q backend must raise NotImplementedError."""
        import qamomile.circuit as qm
        from qamomile.cudaq import CudaqTranspiler

        @qm.qkernel
        def circuit_with_if(q: qm.Qubit) -> qm.Bit:
            q = qm.h(q)
            b = qm.measure(q)
            q = qm.cond(b, qm.x, q)
            return qm.measure(q)

        transpiler = CudaqTranspiler()
        with pytest.raises(NotImplementedError, match="does not support IfOperation"):
            transpiler.transpile(circuit_with_if)

    def test_while_loop_raises_not_implemented(self) -> None:
        """WhileOperation on CUDA-Q backend must raise NotImplementedError."""
        import qamomile.circuit as qm
        from qamomile.cudaq import CudaqTranspiler

        @qm.qkernel
        def circuit_with_while(q: qm.Qubit) -> qm.Bit:
            b = qm.measure(q)
            q, b = qm.while_loop(b, _while_body, q)
            return b

        @qm.qkernel
        def _while_body(q: qm.Qubit) -> tuple[qm.Qubit, qm.Bit]:
            q = qm.x(q)
            return q, qm.measure(q)

        transpiler = CudaqTranspiler()
        with pytest.raises(
            NotImplementedError, match="does not support WhileOperation"
        ):
            transpiler.transpile(circuit_with_while)
