"""Tests for shape propagation through qkernel calls.

These tests verify that Vector[Qubit] shape information is correctly
preserved when qubits are passed through qkernel function calls.
This addresses a bug where ArrayValue.next_version() was not preserving
the shape field, causing shape information to be lost.
"""

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.value import ArrayValue


class TestXMixerShapePropagation:
    """Test the x_mixer pattern that originally exposed the shape bug."""

    def test_x_mixer_shape_through_block(self):
        """x_mixer should preserve Vector[Qubit] shape through Block.call()."""
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.ir.value import Value

        @qmc.qkernel
        def x_mixer(
            q: qmc.Vector[qmc.Qubit],
            beta: qmc.Float,
        ) -> qmc.Vector[qmc.Qubit]:
            n = q.shape[0]
            for i in qmc.range(n):
                q[i] = qmc.rx(q[i], 2 * beta)
            return q

        # Get the internal block
        block = x_mixer.block

        # Create an input ArrayValue with known shape
        shape = (Value(type=UIntType(), name="n").with_const(5),)
        input_val = ArrayValue(type=QubitType(), name="q", shape=shape)
        beta_val = Value(type=qmc.Float, name="beta")

        # Call the block
        call_op = block.call(q=input_val, beta=beta_val)

        # The result should preserve shape
        assert len(call_op.results) == 1
        result_val = call_op.results[0]
        assert isinstance(result_val, ArrayValue)
        assert len(result_val.shape) == 1


class TestQAOAPatternShapePropagation:
    """Test QAOA-like patterns that chain ising_cost and x_mixer."""

    def test_qaoa_layer_shape_through_block(self):
        """QAOA layer should preserve Vector[Qubit] shape through chained calls."""
        from qamomile.circuit.ir.types.primitives import FloatType, QubitType, UIntType
        from qamomile.circuit.ir.value import Value

        @qmc.qkernel
        def ising_cost(
            q: qmc.Vector[qmc.Qubit],
            hi: qmc.Vector[qmc.Float],
            gamma: qmc.Float,
        ) -> qmc.Vector[qmc.Qubit]:
            n = hi.shape[0]
            for i in qmc.range(n):
                q[i] = qmc.rz(q[i], gamma * hi[i])
            return q

        # Test that ising_cost preserves shape
        block = ising_cost.block
        shape = (Value(type=UIntType(), name="n").with_const(3),)
        q_val = ArrayValue(type=QubitType(), name="q", shape=shape)
        hi_val = ArrayValue(type=FloatType(), name="hi", shape=shape)
        gamma_val = Value(type=FloatType(), name="gamma")

        call_op = block.call(q=q_val, hi=hi_val, gamma=gamma_val)

        assert len(call_op.results) == 1
        result_val = call_op.results[0]
        assert isinstance(result_val, ArrayValue)
        assert len(result_val.shape) == 1


class TestNewArrayValueShapePropagation:
    """Test that shape dims are resolved when a callee creates a new ArrayValue.

    This covers the inline pass bug where return value mapping used
    cloned-but-not-substituted values, losing concrete shape dimensions
    for newly created ArrayValues (e.g., PauliEvolveOp results).
    """

    @pytest.fixture
    def qiskit_transpiler(self):
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        return QiskitTranspiler(use_native_composite=False)

    def test_new_array_shape_propagates_through_inline(self, qiskit_transpiler):
        """A callee that creates a new ArrayValue and returns it should
        preserve concrete shape dims so that a subsequent callee's
        q.shape[0]-based for loop resolves correctly."""
        import qamomile.observable as qm_o

        @qmc.qkernel
        def apply_evolve(
            q: qmc.Vector[qmc.Qubit],
            H: qmc.Observable,
            gamma: qmc.Float,
        ) -> qmc.Vector[qmc.Qubit]:
            q = qmc.pauli_evolve(q, H, gamma)
            return q

        @qmc.qkernel
        def apply_mixer(
            q: qmc.Vector[qmc.Qubit],
            beta: qmc.Float,
        ) -> qmc.Vector[qmc.Qubit]:
            n = q.shape[0]
            for i in qmc.range(n):
                q[i] = qmc.rx(q[i], 2.0 * beta)
            return q

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
            H: qmc.Observable,
            gamma: qmc.Float,
            beta: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            q = apply_evolve(q, H, gamma)
            q = apply_mixer(q, beta)  # must resolve q.shape[0] after evolve
            return qmc.measure(q)

        H = qm_o.Hamiltonian()
        H.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 0),), 1.0)
        H.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 1),), 0.5)

        exe = qiskit_transpiler.transpile(
            circuit,
            bindings={"n": 2, "H": H, "gamma": 0.5, "beta": 0.3},
        )
        qc = exe.compiled_quantum[0].circuit

        # Count gates: should have RZ (from fallback evolve) + RX (from mixer)
        gate_names = [inst.operation.name for inst in qc.data]
        assert "rz" in gate_names, f"Expected RZ from evolve, got {gate_names}"
        assert "rx" in gate_names, f"Expected RX from mixer, got {gate_names}"
        assert gate_names.count("rx") == 2  # 2 qubits × 1 RX each


class TestShapePropagationWithTranspiler:
    """Integration tests using the Qiskit transpiler."""

    @pytest.fixture
    def qiskit_transpiler(self):
        """Get Qiskit transpiler if available."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        return QiskitTranspiler()

    def test_x_mixer_transpiles_correctly(self, qiskit_transpiler):
        """x_mixer pattern should transpile and execute correctly."""

        @qmc.qkernel
        def x_mixer(
            q: qmc.Vector[qmc.Qubit],
            beta: qmc.Float,
        ) -> qmc.Vector[qmc.Qubit]:
            n = q.shape[0]
            for i in qmc.range(n):
                angle = 2 * beta
                q[i] = qmc.rx(q[i], angle)
            return q

        @qmc.qkernel
        def circuit(
            hi: qmc.Vector[qmc.Float],
            beta: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            n = hi.shape[0]
            q = qmc.qubit_array(n, name="q")
            q = x_mixer(q, beta)
            return qmc.measure(q)

        hi = np.array([0.1, 0.2, 0.3])  # 3 qubits

        executor = qiskit_transpiler.transpile(
            circuit,
            bindings={"hi": hi},
            parameters=["beta"],
        )

        # Run the circuit
        job = executor.sample(
            qiskit_transpiler.executor(),
            bindings={"beta": 0.5},
            shots=100,
        )
        result = job.result()

        assert result is not None
        # Verify measurement results have correct length
        for bitstring, _count in result.results:
            assert len(bitstring) == 3

    def test_chained_kernels_transpile_correctly(self, qiskit_transpiler):
        """Chained qkernel calls should transpile correctly."""

        @qmc.qkernel
        def layer_a(
            q: qmc.Vector[qmc.Qubit],
            theta: qmc.Float,
        ) -> qmc.Vector[qmc.Qubit]:
            n = q.shape[0]
            for i in qmc.range(n):
                q[i] = qmc.rx(q[i], theta)
            return q

        @qmc.qkernel
        def layer_b(
            q: qmc.Vector[qmc.Qubit],
            phi: qmc.Float,
        ) -> qmc.Vector[qmc.Qubit]:
            n = q.shape[0]  # Must still have shape after layer_a
            for i in qmc.range(n):
                q[i] = qmc.rz(q[i], phi)
            return q

        @qmc.qkernel
        def full_circuit(
            hi: qmc.Vector[qmc.Float],
            theta: qmc.Float,
            phi: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            n = hi.shape[0]
            q = qmc.qubit_array(n, name="q")
            q = layer_a(q, theta)
            q = layer_b(q, phi)  # Shape must propagate from layer_a
            return qmc.measure(q)

        hi = np.array([1.0, 2.0, 3.0, 4.0])  # 4 qubits

        executor = qiskit_transpiler.transpile(
            full_circuit,
            bindings={"hi": hi},
            parameters=["theta", "phi"],
        )

        job = executor.sample(
            qiskit_transpiler.executor(),
            bindings={"theta": 0.3, "phi": 0.7},
            shots=100,
        )
        result = job.result()

        assert result is not None
        for bitstring, _count in result.results:
            assert len(bitstring) == 4


class TestNestedElementExtractionPattern:
    """Regression test for the inline pass dropping cloned parent_array refs.

    A @qkernel taking ``Vector[Qubit]`` that extracts individual ``q[i]``
    elements and passes them to a nested @qkernel helper used to crash at
    emit time with ``AssertionError: Array element key '...' not found in
    qubit_map``. The bug was that ``_inline_call`` resolved call_args by raw
    UUID lookup, so cloned-element values whose ``parent_array`` had been
    remapped to the caller's concrete array kept the stale clone reference.
    """

    @pytest.fixture
    def qiskit_transpiler(self):
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        return QiskitTranspiler()

    def test_nested_element_extraction_pattern(self, qiskit_transpiler):
        """Outer kernel takes Vector[Qubit], extracts q[i] elements, passes
        them to a nested helper that takes individual Qubit args."""

        @qmc.qkernel
        def helper(
            a: qmc.Qubit, b: qmc.Qubit, c: qmc.Qubit
        ) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
            a, b = qmc.cx(a, b)
            a, c = qmc.cx(a, c)
            return a, b, c

        @qmc.qkernel
        def outer(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            q[0], q[1], q[2] = helper(q[0], q[1], q[2])
            return q

        @qmc.qkernel
        def main() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(3, name="q")
            q = outer(q)
            return qmc.measure(q)

        # Should compile without the resource_allocator assertion
        executor = qiskit_transpiler.transpile(main)
        qc = executor.compiled_quantum[0].circuit

        # Should emit two CX gates and three measurements
        gate_names = [inst.operation.name for inst in qc.data]
        assert gate_names.count("cx") == 2
        assert gate_names.count("measure") == 3

    def test_doubly_nested_element_extraction(self, qiskit_transpiler):
        """Three levels deep: outer → middle (Vector[Qubit]) → leaf (Qubit args)."""

        @qmc.qkernel
        def leaf(a: qmc.Qubit, b: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            a, b = qmc.cx(a, b)
            return a, b

        @qmc.qkernel
        def middle(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            q[0], q[1] = leaf(q[0], q[1])
            return q

        @qmc.qkernel
        def outer(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            q = middle(q)
            return q

        @qmc.qkernel
        def main() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, name="q")
            q = outer(q)
            return qmc.measure(q)

        executor = qiskit_transpiler.transpile(main)
        qc = executor.compiled_quantum[0].circuit
        gate_names = [inst.operation.name for inst in qc.data]
        assert gate_names.count("cx") == 1
        assert gate_names.count("measure") == 2
