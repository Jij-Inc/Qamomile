"""Tests for shape propagation through qkernel calls.

These tests verify that Vector[Qubit] shape information is correctly
preserved when qubits are passed through qkernel function calls.
This addresses a bug where ArrayValue.next_version() was not preserving
the shape field, causing shape information to be lost.
"""

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.value import ArrayValue, Value


class TestXMixerShapePropagation:
    """Test the x_mixer pattern that originally exposed the shape bug."""

    def test_x_mixer_shape_through_block(self):
        """x_mixer should preserve Vector[Qubit] shape through BlockValue.call()."""
        from qamomile.circuit.ir.value import ArrayValue, Value
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
        from qamomile.circuit.ir.value import ArrayValue, Value
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


class TestCallReturnShapeNormalization:
    """Test that BlockValue.call() normalizes shape dims for non-input array returns.

    When a subkernel creates an array internally (not an input-alias return),
    its shape dims may reference callee input Values.  BlockValue.call() must
    substitute those dims with the corresponding caller args so that
    downstream caller operations (e.g. measure) never carry stale
    callee-scoped metadata.
    """

    def test_non_input_return_shape_substituted_with_caller_arg(self):
        """Shape dim referencing callee scalar input is replaced by caller arg."""
        from qamomile.circuit.ir.block_value import BlockValue
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType

        n_callee = Value(type=UIntType(), name="n")
        q_array = ArrayValue(type=QubitType(), name="q", shape=(n_callee,))

        block = BlockValue(
            name="make_array",
            label_args=["n"],
            input_values=[n_callee],
            return_values=[q_array],
            operations=[],
        )

        n_caller = Value(type=UIntType(), name="size", params={"const": 5})
        call_op = block.call(n=n_caller)

        result = call_op.results[0]
        assert isinstance(result, ArrayValue)
        assert len(result.shape) == 1
        assert result.shape[0].uuid == n_caller.uuid
        assert result.shape[0].params.get("const") == 5

    def test_non_input_return_shape_with_array_input_dims(self):
        """Shape dim referencing callee array-input shape dim is replaced."""
        from qamomile.circuit.ir.block_value import BlockValue
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType

        callee_dim = Value(type=UIntType(), name="q_dim0")
        q_callee = ArrayValue(type=QubitType(), name="q", shape=(callee_dim,))
        # Return a new array whose shape references the input array's dim
        r_array = ArrayValue(type=QubitType(), name="r", shape=(callee_dim,))

        block = BlockValue(
            name="clone_array",
            label_args=["q"],
            input_values=[q_callee],
            return_values=[r_array],
            operations=[],
        )

        caller_dim = Value(type=UIntType(), name="caller_dim", params={"const": 3})
        q_caller = ArrayValue(type=QubitType(), name="q_caller", shape=(caller_dim,))
        call_op = block.call(q=q_caller)

        result = call_op.results[0]
        assert isinstance(result, ArrayValue)
        assert result.shape[0].uuid == caller_dim.uuid

    def test_input_alias_return_not_affected(self):
        """Input-alias returns still use next_version (no shape substitution)."""
        from qamomile.circuit.ir.block_value import BlockValue
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType

        callee_dim = Value(type=UIntType(), name="q_dim0")
        q_callee = ArrayValue(type=QubitType(), name="q", shape=(callee_dim,))

        block = BlockValue(
            name="identity",
            label_args=["q"],
            input_values=[q_callee],
            return_values=[q_callee],  # input-alias: returns same value
            operations=[],
        )

        caller_dim = Value(type=UIntType(), name="caller_dim", params={"const": 4})
        q_caller = ArrayValue(type=QubitType(), name="q_caller", shape=(caller_dim,))
        call_op = block.call(q=q_caller)

        result = call_op.results[0]
        assert isinstance(result, ArrayValue)
        # Input-alias uses next_version of caller arg
        assert result.logical_id == q_caller.logical_id


class TestDerivedSizeShapePropagation:
    """Test that derived-size (e.g. m = n + 1) propagates through to allocation."""

    @pytest.fixture
    def qiskit_transpiler(self):
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        return QiskitTranspiler()

    def test_no_loop_derived_size_transpiles(self, qiskit_transpiler):
        """single_plain: m = n + 1; q = qubit_array(m); measure(q) succeeds."""

        @qmc.qkernel
        def single_plain(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            m = n + 1
            q = qmc.qubit_array(m, name="q")
            return qmc.measure(q)

        exe = qiskit_transpiler.transpile(single_plain, bindings={"n": 2})
        job = exe.sample(qiskit_transpiler.executor(), shots=10)
        result = job.result()
        assert result is not None
        for bitstring, _ in result.results:
            assert len(bitstring) == 3

    def test_subkernel_derived_size_transpiles(self, qiskit_transpiler):
        """Subkernel returning qubit_array(n+1), caller measure() succeeds."""

        @qmc.qkernel
        def make_plus_one(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            m = n + qmc.uint(1)
            q = qmc.qubit_array(m, name="q")
            return q

        @qmc.qkernel
        def outer(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = make_plus_one(n)
            return qmc.measure(q)

        exe = qiskit_transpiler.transpile(outer, bindings={"n": 2})
        job = exe.sample(qiskit_transpiler.executor(), shots=10)
        result = job.result()
        assert result is not None
        for bitstring, _ in result.results:
            assert len(bitstring) == 3


class TestInlineRegularResultCanonicalization:
    """Test that inline promotes canonicalized regular-op results into value_map.

    When a regular op (e.g. QInitOperation) produces an ArrayValue and that
    value is subsequently passed through an input-alias-return subkernel,
    the inline pass must ensure the canonicalized (substituted) result is
    used — not the stale clone.
    """

    @pytest.fixture
    def qiskit_transpiler(self):
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        return QiskitTranspiler()

    def test_qinit_through_identity_subkernel_measure(self, qiskit_transpiler):
        """QInit -> inner(q) -> return q -> caller measure(q) transpiles."""

        @qmc.qkernel
        def inner(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            return q

        @qmc.qkernel
        def mid(num_qubits: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(num_qubits, name="q")
            q = inner(q)
            return q

        @qmc.qkernel
        def top(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = mid(n)
            return qmc.measure(q)

        exe = qiskit_transpiler.transpile(top, bindings={"n": 4})
        job = exe.sample(qiskit_transpiler.executor(), shots=10)
        result = job.result()
        assert result is not None
        for bitstring, _ in result.results:
            assert len(bitstring) == 4

    def test_inline_preserves_measure_shape_caller_local(self, qiskit_transpiler):
        """After inline, MeasureVectorOperation.results[0].shape stays caller-local."""
        from qamomile.circuit.ir.operation.gate import MeasureVectorOperation
        from qamomile.circuit.transpiler.passes.inline import InlinePass

        @qmc.qkernel
        def inner(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            return q

        @qmc.qkernel
        def mid(num_qubits: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(num_qubits, name="q")
            q = inner(q)
            return q

        @qmc.qkernel
        def top(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = mid(n)
            return qmc.measure(q)

        block = qiskit_transpiler.to_block(top, bindings={"n": 4})
        inlined = InlinePass().run(block)

        # Find MeasureVectorOperation in inlined block
        measure_ops = [
            op for op in inlined.operations if isinstance(op, MeasureVectorOperation)
        ]
        assert len(measure_ops) == 1
        measure_result = measure_ops[0].results[0]
        assert isinstance(measure_result, ArrayValue)
        assert len(measure_result.shape) == 1
        # Shape dim must NOT be the callee-local 'num_qubits' — it must
        # be the caller-local value (either 'n' or a const=4 derivative).
        dim = measure_result.shape[0]
        assert dim.name != "num_qubits", (
            f"MeasureVectorOperation shape regressed to callee symbol: {dim.name}"
        )


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
