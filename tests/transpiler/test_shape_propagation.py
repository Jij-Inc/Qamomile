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
        """x_mixer should preserve Vector[Qubit] shape through BlockValue.call()."""
        from qamomile.circuit.ir.value import Value
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType

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
        shape = (Value(type=UIntType(), name="n", params={"const": 5}),)
        input_val = ArrayValue(type=QubitType(), name="q", shape=shape)
        beta_val = Value(type=qmc.Float, name="beta", params={})

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
        from qamomile.circuit.ir.value import Value
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType, FloatType

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
        shape = (Value(type=UIntType(), name="n", params={"const": 3}),)
        q_val = ArrayValue(type=QubitType(), name="q", shape=shape)
        hi_val = ArrayValue(type=FloatType(), name="hi", shape=shape)
        gamma_val = Value(type=FloatType(), name="gamma", params={})

        call_op = block.call(q=q_val, hi=hi_val, gamma=gamma_val)

        assert len(call_op.results) == 1
        result_val = call_op.results[0]
        assert isinstance(result_val, ArrayValue)
        assert len(result_val.shape) == 1


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
