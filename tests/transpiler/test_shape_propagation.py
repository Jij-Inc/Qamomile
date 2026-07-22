"""Tests for shape propagation through qkernel calls.

These tests verify that Vector[Qubit] shape information is correctly
preserved when qubits are passed through qkernel function calls.
This addresses a bug where ArrayValue.next_version() was not preserving
the shape field, causing shape information to be lost.
"""

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import CInitOperation
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.types.primitives import UIntType
from qamomile.circuit.ir.value import ArrayValue, DictValue, TupleValue, Value
from qamomile.circuit.transpiler.passes.inline import InlinePass


def _scalar_returning_block(name: str) -> tuple[Block, Value]:
    """Build a callee block that returns one scalar value."""
    output = Value(type=UIntType(), name=f"{name}_out")
    block = Block(
        name=name,
        output_values=[output],
        operations=[
            CInitOperation(results=[output]),
            ReturnOperation(operands=[output]),
        ],
        kind=BlockKind.HIERARCHICAL,
    )
    return block, output


def _block_with_structural_output_call(name: str) -> tuple[Block, Value, Value]:
    """Build a block whose tuple output contains a call result placeholder."""
    callee, _ = _scalar_returning_block(f"{name}_callee")
    sibling = Value(type=UIntType(), name=f"{name}_sibling")
    call = callee.call()
    call_result = call.results[0]
    output = TupleValue(name=f"{name}_tuple", elements=(call_result, sibling))
    block = Block(
        name=name,
        output_values=[output],
        operations=[call],
        kind=BlockKind.HIERARCHICAL,
    )
    return block, call_result, sibling


class TestInlineStructuralOutputSubstitution:
    """Test structural block outputs updated by InlinePass substitutions."""

    def test_top_level_tuple_output_rewrites_inlined_call_result(self):
        """InlinePass rewrites call result placeholders inside TupleValue
        block outputs."""
        block, call_result, sibling = _block_with_structural_output_call("top")

        inlined = InlinePass().run(block)

        output = inlined.output_values[0]
        assert isinstance(output, TupleValue)
        assert output.elements[0].uuid != call_result.uuid
        assert output.elements[0].name == "top_callee_out"
        assert output.elements[1] == sibling

    def test_nested_block_tuple_output_rewrites_inlined_call_result(self):
        """Nested blocks owned by operations rewrite structural output
        elements after inlining."""
        nested, call_result, sibling = _block_with_structural_output_call("nested")
        outer = Block(
            name="outer",
            operations=[InverseBlockOperation(source_block=nested)],
            kind=BlockKind.HIERARCHICAL,
        )

        inlined = InlinePass().run(outer)

        inverse_op = inlined.operations[0]
        assert isinstance(inverse_op, InverseBlockOperation)
        assert inverse_op.source_block is not None
        output = inverse_op.source_block.output_values[0]
        assert isinstance(output, TupleValue)
        assert output.elements[0].uuid != call_result.uuid
        assert output.elements[0].name == "nested_callee_out"
        assert output.elements[1] == sibling

    def test_dict_output_tuple_key_rewrites_inlined_call_result(self):
        """InlinePass rewrites call results nested inside DictValue keys."""
        callee, _ = _scalar_returning_block("dict_callee")
        sibling = Value(type=UIntType(), name="dict_sibling")
        value = Value(type=UIntType(), name="dict_value")
        call = callee.call()
        call_result = call.results[0]
        key = TupleValue(name="dict_key", elements=(call_result, sibling))
        output = DictValue(name="dict_output", entries=((key, value),))
        block = Block(
            name="dict_outer",
            output_values=[output],
            operations=[call],
            kind=BlockKind.HIERARCHICAL,
        )

        inlined = InlinePass().run(block)

        dict_output = inlined.output_values[0]
        assert isinstance(dict_output, DictValue)
        key_output, value_output = dict_output.entries[0]
        assert isinstance(key_output, TupleValue)
        assert key_output.elements[0].uuid != call_result.uuid
        assert key_output.elements[0].name == "dict_callee_out"
        assert key_output.elements[1] == sibling
        assert value_output == value


class TestStructuralCallResultMaterialization:
    """Test caller-local structural result graphs produced by ``Block.call``."""

    def test_result_metadata_references_caller_local_sibling(self):
        """Metadata references are remapped after the full result graph is known."""
        element = Value(type=UIntType(), name="element")
        dimension = Value(type=UIntType(), name="dimension").with_const(1)
        array = ArrayValue(
            type=UIntType(),
            name="array",
            shape=(dimension,),
        ).with_array_runtime_metadata(
            element_uuids=(element.uuid,),
            element_logical_ids=(element.logical_id,),
        )
        output = TupleValue(name="output", elements=(element, array))
        block = Block(name="metadata", output_values=[output])

        result = block.call().results[0]

        assert isinstance(result, TupleValue)
        result_element, result_array = result.elements
        assert isinstance(result_array, ArrayValue)
        assert result_element.uuid != element.uuid
        assert result_array.metadata.array_runtime is not None
        assert result_array.metadata.array_runtime.element_uuids == (
            result_element.uuid,
        )
        assert result_array.metadata.array_runtime.element_logical_ids == (
            result_element.logical_id,
        )

    def test_empty_formal_dict_pass_through_preserves_actual_entries(self):
        """A symbolic empty formal dict does not erase populated actual data."""
        formal = DictValue(name="formal", entries=())
        key_element = Value(type=UIntType(), name="key_element").with_const(2)
        key = TupleValue(name="key", elements=(key_element,))
        entry = Value(type=UIntType(), name="entry").with_const(7)
        actual = DictValue(name="actual", entries=((key, entry),))
        block = Block(
            name="identity_dict",
            label_args=["mapping"],
            input_values=[formal],
            output_values=[formal],
            operations=[ReturnOperation(operands=[formal])],
        )

        result = block.call(mapping=actual).results[0]

        assert isinstance(result, DictValue)
        assert result.uuid != actual.uuid
        assert result.entries == actual.entries


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

    def test_block_call_inlines_as_invoke_operation(self):
        """Block.call should produce an inline InvokeOperation."""
        from qamomile.circuit.ir.block import Block, BlockKind
        from qamomile.circuit.ir.operation import GateOperation, InvokeOperation
        from qamomile.circuit.ir.types.primitives import QubitType
        from qamomile.circuit.ir.value import Value
        from qamomile.circuit.transpiler.passes.inline import InlinePass

        @qmc.qkernel
        def helper(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            return q

        q = Value(type=QubitType(), name="q")
        call_op = helper.block.call(q=q)

        assert isinstance(call_op, InvokeOperation)

        outer = Block(
            name="outer",
            label_args=["q"],
            input_values=[q],
            output_values=list(call_op.results),
            operations=[call_op],
            kind=BlockKind.HIERARCHICAL,
        )
        inlined = InlinePass().run(outer)

        assert inlined.kind is BlockKind.AFFINE
        assert not any(isinstance(op, InvokeOperation) for op in inlined.operations)
        assert any(isinstance(op, GateOperation) for op in inlined.operations)


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

        return QiskitTranspiler(
            use_native_composite=False,
            use_native_pauli_evolution=False,
        )

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

        # This fixture disables native composites, so evolution is legalized
        # to RZ gadgets while the mixer emits RX.
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
