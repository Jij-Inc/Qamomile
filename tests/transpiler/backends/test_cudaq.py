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
    - Measurements are no-op in STATIC mode (auto-measured by sample)
    - Runtime control flow uses RUNNABLE mode + cudaq.run()
    - CP and RZZ are decomposed (no native gates)
    - CH and CY are decomposed
    """

    backend_name = "cudaq"
    unsupported_gates: set[str] = {"MEASURE"}

    # Shared emitter instance for finalization in run_circuit_statevector
    _shared_emitter: Any = None

    @classmethod
    def get_emitter(cls) -> Any:
        """Get CUDA-Q KernelEmitter instance."""
        from qamomile.cudaq.emitter import CudaqKernelEmitter

        cls._shared_emitter = CudaqKernelEmitter()
        return cls._shared_emitter

    @classmethod
    def get_simulator(cls) -> Any:
        """Get CUDA-Q simulator (not directly used)."""
        return None

    @classmethod
    def run_circuit_statevector(cls, circuit: Any) -> np.ndarray:
        """Run circuit and extract statevector using CUDA-Q.

        Finalizes the circuit in STATIC mode before extracting state.
        """
        import cudaq

        from qamomile.cudaq.emitter import ExecutionMode

        if cls._shared_emitter is not None and circuit.kernel_func is None:
            cls._shared_emitter.finalize(circuit, ExecutionMode.STATIC)

        state = cudaq.get_state(circuit.kernel_func)
        return np.array(state)


class TestCudaqRuntimeControlFlow:
    """Test runtime measurement-dependent control flow via cudaq.run()."""

    def test_c_if_transpiles_to_runnable_mode(self) -> None:
        """Runtime if-then produces a RUNNABLE-mode artifact."""
        import qamomile.circuit as qmc
        from qamomile.cudaq import CudaqTranspiler
        from qamomile.cudaq.emitter import ExecutionMode

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
        assert circuit.execution_mode == ExecutionMode.RUNNABLE

    def test_if_with_else_transpiles_to_runnable_mode(self) -> None:
        """Runtime if-else produces a RUNNABLE-mode artifact."""
        import qamomile.circuit as qmc
        from qamomile.cudaq import CudaqTranspiler
        from qamomile.cudaq.emitter import ExecutionMode

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
        assert circuit.execution_mode == ExecutionMode.RUNNABLE

    def test_while_loop_transpiles_to_runnable_mode(self) -> None:
        """Runtime while loop produces a RUNNABLE-mode artifact."""
        import qamomile.circuit as qmc
        from qamomile.cudaq import CudaqTranspiler
        from qamomile.cudaq.emitter import ExecutionMode

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
        assert circuit.execution_mode == ExecutionMode.RUNNABLE


class TestCudaqSourceInspectRegression:
    """Regression tests for generated kernel source inspect-ability.

    Ensures that ``inspect.getsource()`` works on kernels produced by
    ``CudaqKernelEmitter.finalize()`` and that multiple kernels do not
    share source via filename collision.
    """

    def test_finalized_kernel_is_inspectable(self) -> None:
        """inspect.getsource() succeeds on a finalized kernel."""
        import inspect

        import qamomile.circuit as qmc
        from qamomile.cudaq import CudaqTranspiler

        @qmc.qkernel
        def single_h() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(single_h)
        kernel = exe.compiled_quantum[0].circuit.kernel_func

        # Must not raise OSError.  The kernel is a PyKernelDecorator;
        # inspect the underlying function via .kernelFunction.
        source = inspect.getsource(kernel.kernelFunction)
        assert "_qamomile_kernel" in source

    def test_multiple_kernels_have_distinct_sources(self) -> None:
        """Two consecutively generated kernels have independent sources."""
        import inspect

        import qamomile.circuit as qmc
        from qamomile.cudaq import CudaqTranspiler

        @qmc.qkernel
        def circuit_a() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            return qmc.measure(q)

        @qmc.qkernel
        def circuit_b() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.x(q)
            return qmc.measure(q)

        transpiler = CudaqTranspiler()
        exe_a = transpiler.transpile(circuit_a)
        exe_b = transpiler.transpile(circuit_b)

        kernel_a = exe_a.compiled_quantum[0].circuit.kernel_func
        kernel_b = exe_b.compiled_quantum[0].circuit.kernel_func

        source_a = inspect.getsource(kernel_a.kernelFunction)
        source_b = inspect.getsource(kernel_b.kernelFunction)

        # Sources must differ: circuit_a uses h(), circuit_b uses x()
        assert source_a != source_b
        assert "h(" in source_a
        assert "x(" in source_b
