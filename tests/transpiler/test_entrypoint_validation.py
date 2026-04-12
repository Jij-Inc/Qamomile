from __future__ import annotations

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.errors import EntrypointValidationError


@pytest.fixture
def qiskit_transpiler():
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    return QiskitTranspiler()


class TestEntrypointValidation:
    def test_transpile_rejects_quantum_input_entrypoint(self, qiskit_transpiler) -> None:
        @qmc.qkernel
        def kernel(q: qmc.Qubit) -> qmc.Bit:
            return qmc.measure(q)

        with pytest.raises(EntrypointValidationError, match="classical inputs and outputs only"):
            qiskit_transpiler.transpile(kernel)

    def test_transpile_rejects_quantum_output_entrypoint(self, qiskit_transpiler) -> None:
        @qmc.qkernel
        def kernel() -> qmc.Qubit:
            q = qmc.qubit(name="q")
            q = qmc.h(q)
            return q

        with pytest.raises(EntrypointValidationError, match="classical inputs and outputs only"):
            qiskit_transpiler.transpile(kernel)

        with pytest.raises(EntrypointValidationError, match="classical inputs and outputs only"):
            qiskit_transpiler.to_circuit(kernel)

    def test_quantum_io_subroutine_can_be_used_from_classical_entrypoint(
        self,
        qiskit_transpiler,
    ) -> None:
        @qmc.qkernel
        def inner(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.h(q)

        @qmc.qkernel
        def outer() -> qmc.Bit:
            q = qmc.qubit(name="q")
            q = inner(q)
            return qmc.measure(q)

        executable = qiskit_transpiler.transpile(outer)
        assert executable.get_first_circuit() is not None

    def test_rejects_tuple_with_qubit_input(self, qiskit_transpiler) -> None:
        @qmc.qkernel
        def kernel(pair: qmc.Tuple[qmc.Float, qmc.Qubit]) -> qmc.Bit:
            q = pair[1]
            return qmc.measure(q)

        with pytest.raises(EntrypointValidationError, match="classical inputs and outputs only"):
            qiskit_transpiler.transpile(kernel)

    def test_rejects_vector_qubit_input(self, qiskit_transpiler) -> None:
        @qmc.qkernel
        def kernel(q: qmc.Vector[qmc.Qubit]) -> qmc.Bit:
            q[0] = qmc.h(q[0])
            return qmc.measure(q[0])

        with pytest.raises(EntrypointValidationError, match="classical inputs and outputs only"):
            qiskit_transpiler.transpile(kernel)

    def test_accepts_classical_tuple_input(self, qiskit_transpiler) -> None:
        """Classical-only tuple should pass entrypoint validation."""
        from qamomile.circuit.transpiler.passes.entrypoint_validation import (
            EntrypointValidationPass,
        )

        @qmc.qkernel
        def kernel(params: qmc.Tuple[qmc.Float, qmc.Float]) -> qmc.Bit:
            q = qmc.qubit(name="q")
            q = qmc.rx(q, params[0])
            q = qmc.ry(q, params[1])
            return qmc.measure(q)

        block = qiskit_transpiler.to_block(kernel)
        # Should not raise - tuple of classical types is fine
        result = EntrypointValidationPass().run(block)
        assert result is block

    def test_rejects_vector_qubit_input(self, qiskit_transpiler) -> None:
        @qmc.qkernel
        def kernel(q: qmc.Vector[qmc.Qubit]) -> qmc.Bit:
            q[0] = qmc.h(q[0])
            return qmc.measure(q[0])

        with pytest.raises(EntrypointValidationError, match="classical inputs and outputs only"):
            qiskit_transpiler.transpile(kernel)

    def test_accepts_vector_bit_output(self, qiskit_transpiler) -> None:
        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit(name="q")
            q = qmc.h(q)
            b = qmc.measure(q)
            return (b,)

        executable = qiskit_transpiler.transpile(kernel)
        assert executable is not None

    def test_build_allows_quantum_io_kernel(self) -> None:
        @qmc.qkernel
        def kernel() -> qmc.Qubit:
            q = qmc.qubit(name="q")
            q = qmc.h(q)
            return q

        graph = kernel.build()
        assert len(graph.output_values) == 1
        assert graph.output_values[0].type.is_quantum()
