"""Tests for ValidateWhileContractPass.

Verifies that only measurement-backed while conditions pass validation,
and that all other while patterns are rejected with clear ValidationError.

Note: Do NOT use ``from __future__ import annotations`` in this file.
The @qkernel AST transformer relies on resolved type annotations.
"""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.qiskit.transpiler import QiskitTranspiler


class TestWhileContractPositive:
    """Measurement-backed while conditions must pass validation."""

    def test_measurement_backed_while_transpiles(self):
        """bit = measure(q); while bit: ... bit = measure(q) succeeds."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            bit = qmc.measure(q)
            while bit:
                q = qmc.qubit("q2")
                q = qmc.h(q)
                bit = qmc.measure(q)
            return bit

        transpiler = QiskitTranspiler()
        result = transpiler.transpile(circuit)
        assert result is not None

    def test_measurement_while_no_loop_carry_transpiles(self):
        """bit = measure(q); while bit: (body without re-measure) succeeds."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.h(q)
            bit = qmc.measure(q)
            while bit:
                q2 = qmc.qubit("q2")
                q2 = qmc.h(q2)
                _ = qmc.measure(q2)
            return bit

        transpiler = QiskitTranspiler()
        result = transpiler.transpile(circuit)
        assert result is not None


class TestWhileContractNegative:
    """Non-measurement while conditions must be rejected."""

    def test_classical_uint_while_rejected(self):
        """while n: with UInt parameter is rejected."""

        @qmc.qkernel
        def circuit(q: qmc.Qubit, n: qmc.UInt) -> qmc.Qubit:
            while n:
                q = qmc.h(q)
            return q

        transpiler = QiskitTranspiler()
        with pytest.raises(ValidationError, match="measurement result"):
            transpiler.transpile(circuit, bindings={"n": 1})

    def test_mixed_initial_classical_while_rejected(self):
        """flag = True; while flag: ... flag = measure(q) is rejected.

        Even though the body reassigns the condition to a measurement
        result, the initial condition is a boolean constant, not a
        measurement-backed Bit.
        """

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit("q")
            flag = True
            while flag:
                q = qmc.h(q)
                flag = qmc.measure(q)
            return flag

        transpiler = QiskitTranspiler()
        with pytest.raises(ValidationError, match="measurement result"):
            transpiler.transpile(circuit)

    def test_unresolved_classical_while_rejected(self):
        """while flag: with unbound parameter is rejected."""

        @qmc.qkernel
        def circuit(q: qmc.Qubit, flag: qmc.UInt) -> qmc.Qubit:
            while flag:
                q = qmc.h(q)
            return q

        transpiler = QiskitTranspiler()
        with pytest.raises(ValidationError, match="measurement result"):
            transpiler.transpile(circuit, parameters=["flag"])
