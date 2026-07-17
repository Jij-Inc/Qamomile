"""Tests for transpiler pipeline error paths.

Validates that SeparationError is raised when programs have no quantum
operations, and that measurement-feedback patterns with native control flow
(if/while) are correctly handled without errors.

Note: Do NOT use ``from __future__ import annotations`` in this file.
The @qkernel AST transformer relies on resolved type annotations.
"""

import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.transpiler.errors import DependencyError, SeparationError
from qamomile.circuit.transpiler.segments import MultipleQuantumSegmentsError
from qamomile.qiskit.transpiler import QiskitTranspiler


@pytest.fixture
def qiskit_transpiler():
    pytest.importorskip("qiskit")
    return QiskitTranspiler()


class TestSeparationError:
    """Programs must contain at least one quantum segment."""

    def test_pure_classical_kernel_raises(self, qiskit_transpiler) -> None:
        """A kernel with no quantum operations cannot be transpiled."""

        @qmc.qkernel
        def kernel(x: qmc.Float) -> qmc.Float:
            return x

        with pytest.raises(SeparationError, match="No quantum segment found"):
            qiskit_transpiler.transpile(kernel, bindings={"x": 1.0})

    def test_identity_float_kernel_raises(self, qiskit_transpiler) -> None:
        """Another pure classical kernel variant."""

        @qmc.qkernel
        def kernel(a: qmc.Float, b: qmc.Float) -> qmc.Float:
            return a + b

        with pytest.raises(SeparationError, match="No quantum segment found"):
            qiskit_transpiler.transpile(kernel, bindings={"a": 1.0, "b": 2.0})


class TestDependencyErrorContract:
    """DependencyError is raised when a QUANTUM operation's classical operand
    is derived from a measurement result (JIT compilation requirement).

    Note: The typed frontend prevents most such cases at the Python level
    (e.g., Bit cannot be used as Float rotation angle). These tests verify
    the error class and its attributes are correctly structured.
    """

    def test_error_class_has_expected_attributes(self) -> None:
        """DependencyError carries quantum_op and classical_value context."""
        err = DependencyError(
            "test message",
            quantum_op="GateOperation",
            classical_value="theta",
        )
        assert err.quantum_op == "GateOperation"
        assert err.classical_value == "theta"
        assert "test message" in str(err)

    def test_error_inherits_from_compile_error(self) -> None:
        """DependencyError is a QamomileCompileError."""
        from qamomile.circuit.transpiler.errors import QamomileCompileError

        err = DependencyError("msg")
        assert isinstance(err, QamomileCompileError)

    def test_region_args_propagate_measurement_taint(self) -> None:
        """A loop result inherits taint from its initial and yielded values."""
        from qamomile.circuit.ir.operation.control_flow import ForOperation, RegionArg
        from qamomile.circuit.ir.operation.gate import MeasureOperation
        from qamomile.circuit.ir.types.primitives import BitType, QubitType
        from qamomile.circuit.ir.value import Value
        from qamomile.circuit.transpiler.passes.analyze import (
            build_dependency_graph,
            find_measurement_derived_values,
            find_measurement_results,
        )

        qubit = Value(type=QubitType(), name="q")
        init = Value(type=BitType(), name="init").with_const(False)
        block_arg = Value(type=BitType(), name="carry")
        yielded = Value(type=BitType(), name="yielded")
        result = Value(type=BitType(), name="result")
        measure = MeasureOperation(operands=[qubit], results=[yielded])
        loop = ForOperation(
            operations=[measure],
            region_args=(RegionArg("carry", init, block_arg, yielded, result),),
            results=[result],
        )

        graph = build_dependency_graph([loop])
        derived = find_measurement_derived_values(
            graph, find_measurement_results([loop])
        )

        assert block_arg.uuid in derived
        assert result.uuid in derived


class TestMultipleQuantumSegmentsErrorContract:
    """MultipleQuantumSegmentsError is raised when the program has more than
    one quantum segment (requiring classical interleaving not supported)."""

    def test_error_class_message(self) -> None:
        """Verify error message includes segment count."""
        err = MultipleQuantumSegmentsError(
            "Found 3 quantum segments. Only single quantum execution is supported."
        )
        assert "3 quantum segments" in str(err)

    def test_error_is_exception(self) -> None:
        """MultipleQuantumSegmentsError follows the compile-error contract."""
        from qamomile.circuit.transpiler.errors import QamomileCompileError

        assert issubclass(MultipleQuantumSegmentsError, QamomileCompileError)

    def test_true_multi_segment_program_keeps_segment_count_message(
        self, qiskit_transpiler
    ) -> None:
        """A genuinely multi-quantum-segment program keeps the original message.

        Quantum operations resuming after ``qmc.expval`` produce a second
        quantum segment. This must keep raising ``MultipleQuantumSegmentsError``
        with the segment-count / measurement-dependence wording — the
        runtime-loop-bound case is diagnosed earlier by
        ``SymbolicShapeValidationPass`` and must not have changed this path.
        """

        @qmc.qkernel
        def kernel(obs: qmc.Observable) -> tuple[qmc.Float, qmc.Vector[qmc.Bit]]:
            q = qmc.qubit_array(1, "q")
            q[0] = qmc.h(q[0])
            e = qmc.expval(q, obs)
            q2 = qmc.qubit_array(1, "q2")
            q2[0] = qmc.x(q2[0])
            return e, qmc.measure(q2)

        with pytest.raises(
            MultipleQuantumSegmentsError, match="Found 2 quantum segments"
        ) as exc_info:
            qiskit_transpiler.transpile(kernel, bindings={"obs": qm_o.Z(0)})
        assert "measurement results" in str(exc_info.value)


class TestMeasurementFeedbackWithNativeControlFlow:
    """Qiskit supports native if/while control flow within a single quantum
    segment. These patterns should NOT raise errors."""

    def test_if_with_measurement_condition_succeeds(self, qiskit_transpiler) -> None:
        """Measurement result as if-condition works with native control flow."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q1 = qmc.qubit("q1")
            q1 = qmc.h(q1)
            bit = qmc.measure(q1)

            q2 = qmc.qubit("q2")
            if bit:
                q2 = qmc.x(q2)

            return qmc.measure(q2)

        # Native if-else with measurement condition stays in one quantum segment
        executable = qiskit_transpiler.transpile(kernel)
        assert executable.get_first_circuit() is not None

    def test_while_with_measurement_condition_succeeds(self, qiskit_transpiler) -> None:
        """Measurement result as while-condition works with native control flow."""

        @qmc.qkernel
        def kernel() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            bit = qmc.measure(q)
            while bit:
                q2 = qmc.qubit("q_loop")
                q2 = qmc.h(q2)
                bit = qmc.measure(q2)
            return bit

        executable = qiskit_transpiler.transpile(kernel)
        assert executable.get_first_circuit() is not None

    def test_parameter_binding_passes(self, qiskit_transpiler) -> None:
        """Classical values that are parameters are allowed as rotation angles."""

        @qmc.qkernel
        def kernel(theta: qmc.Float) -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.rx(q, theta)
            return qmc.measure(q)

        executable = qiskit_transpiler.transpile(kernel, bindings={"theta": 0.5})
        assert executable.get_first_circuit() is not None
