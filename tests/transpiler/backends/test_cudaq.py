"""CUDA-Q transpiler test configuration.

This module configures the transpiler test suite for the CUDA-Q backend.
Tests are automatically skipped if the ``cudaq`` package is not installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.cudaq

cudaq = pytest.importorskip("cudaq")

from qamomile.circuit.transpiler.circuit_ir import (  # noqa: E402
    CircuitBuilder,
    CircuitProgram,
    ParameterExpr,
    ReusableCircuit,
)
from qamomile.circuit.transpiler.errors import EmitError  # noqa: E402
from qamomile.cudaq.materializer import CudaqMaterializer  # noqa: E402
from tests.transpiler.backends._cudaq_source_assertions import (  # noqa: E402
    TracingCudaqKernelEmitter,
    ValidatingCudaqTranspiler as CudaqTranspiler,
    assert_inspect_source_matches_artifact,
)
from tests.transpiler.base_test import TranspilerTestSuite  # noqa: E402


def _phase_only_program(
    *,
    controls: int = 0,
    inverse: bool = False,
    power: int = 1,
) -> CircuitProgram:
    """Build a call whose reusable body consists only of global phase."""
    body = CircuitBuilder(1, 0, "phase_only")
    body.add_global_phase(ParameterExpr("theta"))
    caller = CircuitBuilder(1 + controls, 0, "phase_caller")
    caller.append_call(
        ReusableCircuit(
            body.freeze(),
            "phase_only",
            controls=controls,
            inverse=inverse,
            power=power,
        ),
        tuple(range(1 + controls)),
    )
    return caller.freeze()


class TestCudaqGlobalPhaseMaterialization:
    """Test CUDA-Q global-phase handling at the circuit-IR boundary."""

    def test_standalone_phase_is_preserved_without_changing_the_abi(self) -> None:
        """A root phase reaches CUDA-Q exactly and keeps its parameter ABI."""
        builder = CircuitBuilder(1, 0)
        theta = ParameterExpr("theta")
        builder.add_global_phase(theta * theta)

        materialized = CudaqMaterializer().materialize(
            builder.freeze(),
            ("theta",),
        )

        assert materialized.parameter_order == ("theta",)
        assert tuple(materialized.parameters) == ("theta",)
        assert materialized.artifact.param_count == 1
        assert "def _qamomile_kernel(thetas: list[float]):" in (
            materialized.artifact.entry_source
        )
        assert "r1(" in materialized.artifact.entry_source
        assert "rz(" in materialized.artifact.entry_source

        angle = 0.41
        state = np.array(
            cudaq.get_state(materialized.artifact.kernel_func, [angle])
        )
        assert np.allclose(
            state,
            np.array([np.exp(1j * angle**2), 0.0], dtype=np.complex128),
            atol=1e-10,
        )

    def test_zero_qubit_standalone_phase_is_rejected_explicitly(self) -> None:
        """CUDA-Q never hides a carrier qubit that would change the ABI."""
        builder = CircuitBuilder(0, 0)
        builder.add_global_phase(0.25)

        with pytest.raises(EmitError, match="at least 1 qubit"):
            CudaqMaterializer().materialize(builder.freeze())

    def test_controlled_phase_gate_preserves_the_exact_unitary(self) -> None:
        """CUDA-Q CP emission retains the global factor of its matrix."""
        from qamomile.cudaq.emitter import CudaqKernelEmitter, ExecutionMode

        emitter = CudaqKernelEmitter()
        artifact = emitter.create_circuit(2, 0)
        emitter.emit_cp(artifact, 0, 1, 0.73)
        artifact = emitter.finalize(artifact, ExecutionMode.STATIC)

        state = np.array(cudaq.get_state(artifact.kernel_func))
        assert np.allclose(
            state,
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
            atol=1e-10,
        )

    def test_controlled_phase_only_call_emits_relative_phase_correction(self) -> None:
        """A controlled phase-only helper emits its observable correction."""
        artifact = (
            CudaqMaterializer()
            .materialize(
                _phase_only_program(controls=1),
                ("theta",),
            )
            .artifact
        )

        assert "cudaq.control(_qamomile_phase_only_0, q[0], q[1], thetas)" in (
            artifact.entry_source
        )
        [correction] = [
            line.strip()
            for line in artifact.entry_source.splitlines()
            if line.strip().startswith("r1(")
        ]
        assert "thetas[0]" in correction
        assert correction.endswith(", q[0])")
        assert not correction.startswith("r1(-")
        assert "pass" in artifact.source

    def test_tiny_controlled_phase_is_not_optimized_away(self) -> None:
        """Every exact nonzero helper phase emits a relative correction."""
        body = CircuitBuilder(1, 0, "tiny_phase")
        body.add_global_phase(1e-16)
        caller = CircuitBuilder(2, 0)
        caller.append_call(
            ReusableCircuit(body.freeze(), "tiny_phase", controls=1),
            (0, 1),
        )

        artifact = CudaqMaterializer().materialize(caller.freeze()).artifact

        assert "r1(1e-16, q[0])" in artifact.entry_source

    def test_inverse_controlled_phase_negates_relative_correction(self) -> None:
        """Inverse transforms negate the phase correction exactly once."""
        artifact = (
            CudaqMaterializer()
            .materialize(
                _phase_only_program(controls=1, inverse=True),
                ("theta",),
            )
            .artifact
        )

        assert "cudaq.adjoint(_qamomile_phase_only_0" in artifact.source
        [correction] = [
            line.strip()
            for line in artifact.entry_source.splitlines()
            if line.strip().startswith("r1(")
        ]
        assert correction.startswith("r1(-")
        assert "thetas[0]" in correction
        assert correction.endswith(", q[0])")

    def test_power_repeats_call_and_phase_correction_together(self) -> None:
        """Integral power repeats both the helper and its relative phase."""
        artifact = (
            CudaqMaterializer()
            .materialize(
                _phase_only_program(controls=1, power=3),
                ("theta",),
            )
            .artifact
        )

        assert artifact.entry_source.count("cudaq.control(") == 3
        corrections = [
            line.strip()
            for line in artifact.entry_source.splitlines()
            if line.strip().startswith("r1(")
        ]
        assert len(corrections) == 3
        assert all("thetas[0]" in correction for correction in corrections)
        assert all(correction.endswith(", q[0])") for correction in corrections)


class TestCudaqTranspiler(TranspilerTestSuite):
    """Test suite for CUDA-Q transpiler.

    CUDA-Q supports most standard gates but has some limitations:
    - Measurements are no-op in STATIC mode (auto-measured by sample)
    - Runtime control flow uses RUNNABLE mode + cudaq.run()
    - RZZ is decomposed; CP uses CUDA-Q's controlled R1
    - CH and CY are decomposed
    """

    backend_name = "cudaq"
    unsupported_gates: set[str] = {"MEASURE"}

    # Shared emitter instance for finalization in run_circuit_statevector
    _shared_emitter: Any = None

    @classmethod
    def get_emitter(cls) -> Any:
        """Get CUDA-Q KernelEmitter instance."""
        cls._shared_emitter = TracingCudaqKernelEmitter()
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

    def test_create_empty_circuit(self) -> None:
        """Test creating an empty circuit and validating its generated source."""
        from qamomile.cudaq.emitter import ExecutionMode

        emitter = self.get_emitter()
        circuit = emitter.create_circuit(num_qubits=2, num_clbits=1)
        emitter.finalize(circuit, ExecutionMode.STATIC)

        assert circuit is not None
        assert circuit.source


class TestCudaqRuntimeControlFlow:
    """Test runtime measurement-dependent control flow via cudaq.run()."""

    def test_c_if_transpiles_to_runnable_mode(self) -> None:
        """Runtime if-then produces a RUNNABLE-mode artifact."""
        import qamomile.circuit as qmc
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
        from qamomile.cudaq.emitter import ExecutionMode

        @qmc.qkernel
        def circuit_with_while() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            bit = qmc.measure(q)
            while bit:
                q2 = qmc.qubit("q2")
                q2 = qmc.h(q2)
                bit = qmc.measure(q2)
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
        assert_inspect_source_matches_artifact(exe.compiled_quantum[0].circuit)

    def test_multiple_kernels_have_distinct_sources(self) -> None:
        """Two consecutively generated kernels have independent sources."""
        import inspect

        import qamomile.circuit as qmc

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
        assert_inspect_source_matches_artifact(exe_a.compiled_quantum[0].circuit)
        assert_inspect_source_matches_artifact(exe_b.compiled_quantum[0].circuit)
