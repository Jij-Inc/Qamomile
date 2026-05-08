"""Tests for transpiler pipeline error paths.

Validates that SeparationError is raised when programs have no quantum
operations, and that measurement-feedback patterns with native control flow
(if/while) are correctly handled without errors.

Note: Do NOT use ``from __future__ import annotations`` in this file.
The @qkernel AST transformer relies on resolved type annotations.
"""

import pytest

import qamomile.circuit as qmc
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
        """MultipleQuantumSegmentsError inherits from Exception."""
        assert issubclass(MultipleQuantumSegmentsError, Exception)


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
