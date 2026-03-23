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
    - Measurements are no-op in the builder emitter (auto-measured by sample)
    - Runtime control flow uses codegen emitter + cudaq.run()
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


class TestCudaqRuntimeControlFlow:
    """Test runtime measurement-dependent control flow via cudaq.run()."""

    def test_c_if_transpiles_to_runtime_circuit(self) -> None:
        """Runtime if-then produces CudaqRuntimeCircuit."""
        import qamomile.circuit as qmc
        from qamomile.cudaq import CudaqTranspiler
        from qamomile.cudaq.emitter import CudaqRuntimeCircuit

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
        exe = transpiler.transpile(circuit_with_c_if)
        circuit = exe.compiled_quantum[0].circuit
        assert isinstance(circuit, CudaqRuntimeCircuit)

    def test_if_with_else_transpiles_to_runtime_circuit(self) -> None:
        """Runtime if-else produces CudaqRuntimeCircuit."""
        import qamomile.circuit as qmc
        from qamomile.cudaq import CudaqTranspiler
        from qamomile.cudaq.emitter import CudaqRuntimeCircuit

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
        exe = transpiler.transpile(circuit_with_if_else)
        circuit = exe.compiled_quantum[0].circuit
        assert isinstance(circuit, CudaqRuntimeCircuit)

    def test_while_loop_transpiles_to_runtime_circuit(self) -> None:
        """Runtime while loop produces CudaqRuntimeCircuit."""
        import qamomile.circuit as qmc
        from qamomile.cudaq import CudaqTranspiler
        from qamomile.cudaq.emitter import CudaqRuntimeCircuit

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
        exe = transpiler.transpile(circuit_with_while)
        circuit = exe.compiled_quantum[0].circuit
        assert isinstance(circuit, CudaqRuntimeCircuit)
