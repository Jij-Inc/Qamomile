"""Tests for SimplifiedProgram validation and C→Q→C pattern enforcement."""

import pytest

import qamomile.circuit as qm
from qamomile.circuit.transpiler.segments import SimplifiedProgram
from qamomile.qiskit.transpiler import QiskitTranspiler


class TestSimplifiedProgramStructure:
    """Test that SimplifiedProgram enforces C→Q→C pattern."""

    def test_transpile_succeeds(self):
        """Full transpile pipeline should work and return SimplifiedProgram internally."""

        @qm.qkernel
        def kernel(theta: qm.Float) -> qm.Bit:
            # This kernel has classical prep and quantum segments
            angle = theta * 2.0  # Classical prep

            # Quantum segment
            q = qm.qubit("q")
            q = qm.h(q)
            q = qm.rx(q, angle)

            # Measurement
            b = qm.measure(q)
            return b

        transpiler = QiskitTranspiler()
        executable = transpiler.transpile(kernel, bindings={"theta": 0.5})

        # Should succeed
        assert executable is not None
        assert executable.quantum_circuit is not None

    def test_simplified_program_structure(self):
        """Check that separate() returns SimplifiedProgram."""

        @qm.qkernel
        def kernel() -> qm.Bit:
            q = qm.qubit("q")
            q = qm.h(q)
            b = qm.measure(q)
            return b

        transpiler = QiskitTranspiler()
        block = transpiler.to_block(kernel)
        inlined = transpiler.inline(block)
        analyzed = transpiler.analyze(inlined)
        separated = transpiler.separate(analyzed)

        # Should return SimplifiedProgram
        assert isinstance(separated, SimplifiedProgram)
        assert separated.quantum is not None


class TestExecutableProgram:
    """Test ExecutableProgram quantum_circuit property."""

    def test_quantum_circuit_property(self):
        """Test direct access to quantum circuit via property."""

        @qm.qkernel
        def kernel() -> qm.Bit:
            q = qm.qubit("q")
            q = qm.h(q)
            b = qm.measure(q)
            return b

        transpiler = QiskitTranspiler()
        executable = transpiler.transpile(kernel)

        # Should be able to access circuit directly
        circuit = executable.quantum_circuit
        assert circuit is not None

        # Should be same as get_first_circuit()
        assert circuit is executable.get_first_circuit()

    def test_quantum_circuit_error_on_no_circuit(self):
        """Test that quantum_circuit raises error when no circuit exists."""
        from qamomile.circuit.transpiler.errors import ExecutionError
        from qamomile.circuit.transpiler.executable import ExecutableProgram

        # Create empty program
        empty_program = ExecutableProgram()

        with pytest.raises(ExecutionError, match="No quantum circuit"):
            _ = empty_program.quantum_circuit
