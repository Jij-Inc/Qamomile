"""Tests for the expval (expectation value) operation with Observable bindings."""

import pytest
import qamomile.circuit as qm
import qamomile.observable as qm_o
from qamomile.circuit.ir.operation import ExpvalOp


class TestExpvalFrontend:
    """Test expval frontend function with Observable parameter."""

    def test_expval_basic(self):
        """Test basic expval with Observable parameter."""

        @qm.qkernel
        def test_kernel(q0: qm.Qubit, q1: qm.Qubit, H: qm.Observable) -> qm.Float:
            q0 = qm.h(q0)
            exp_val = qm.expval((q0, q1), H)
            return exp_val

        block = test_kernel.build()

        # Check that ExpvalOp is in the operations
        has_expval = any(isinstance(op, ExpvalOp) for op in block.operations)
        assert has_expval, "Expected ExpvalOp in block operations"

        # Find the ExpvalOp and check its properties
        expval_op = next(op for op in block.operations if isinstance(op, ExpvalOp))
        assert expval_op.output is not None
        assert expval_op.qubits is not None
        assert expval_op.observable is not None

    def test_expval_with_vector(self):
        """Test expval with Vector[Qubit]."""

        @qm.qkernel
        def test_kernel(n: qm.UInt, H: qm.Observable) -> qm.Float:
            q = qm.qubit_array(n, "q")
            q[0] = qm.h(q[0])
            q[0], q[1] = qm.cx(q[0], q[1])
            exp_val = qm.expval(q, H)
            return exp_val

        # Build with n=2 binding (H can be unbound for IR test)
        block = test_kernel.build(n=2)

        # Should have ExpvalOp
        has_expval = any(isinstance(op, ExpvalOp) for op in block.operations)
        assert has_expval


class TestExpvalOp:
    """Test ExpvalOp IR operation."""

    def test_operation_kind_is_hybrid(self):
        """Test that ExpvalOp has HYBRID operation kind."""
        from qamomile.circuit.ir.operation.operation import OperationKind

        @qm.qkernel
        def test_kernel(q0: qm.Qubit, H: qm.Observable) -> qm.Float:
            exp_val = qm.expval((q0,), H)
            return exp_val

        block = test_kernel.build()
        expval_op = next(op for op in block.operations if isinstance(op, ExpvalOp))

        assert expval_op.operation_kind == OperationKind.HYBRID

    def test_expval_output_type_is_float(self):
        """Test that ExpvalOp output has FloatType."""
        from qamomile.circuit.ir.types.primitives import FloatType

        @qm.qkernel
        def test_kernel(q0: qm.Qubit, H: qm.Observable) -> qm.Float:
            exp_val = qm.expval((q0,), H)
            return exp_val

        block = test_kernel.build()
        expval_op = next(op for op in block.operations if isinstance(op, ExpvalOp))

        assert isinstance(expval_op.output.type, FloatType)


class TestExpvalSegment:
    """Test ExpvalSegment and related structures."""

    def test_segment_kind(self):
        """Test that ExpvalSegment has EXPVAL kind."""
        from qamomile.circuit.transpiler.segments import (
            ExpvalSegment,
            SegmentKind,
        )

        segment = ExpvalSegment()
        assert segment.kind == SegmentKind.EXPVAL

    def test_compiled_expval_segment_fields(self):
        """Test CompiledExpvalSegment has expected fields."""
        from qamomile.circuit.transpiler.compiled_segments import CompiledExpvalSegment
        from qamomile.circuit.transpiler.segments import ExpvalSegment
        import dataclasses

        fields = {f.name for f in dataclasses.fields(CompiledExpvalSegment)}
        assert "segment" in fields
        assert "hamiltonian" in fields
        assert "quantum_segment_index" in fields
        assert "result_ref" in fields
        assert "qubit_map" in fields


class TestExpvalJob:
    """Test ExpvalJob."""

    def test_expval_job_result(self):
        """Test that ExpvalJob correctly returns the expectation value."""
        from qamomile.circuit.transpiler.job import ExpvalJob, JobStatus

        exp_val = 0.75
        job = ExpvalJob(exp_val)

        assert job.status() == JobStatus.COMPLETED
        assert job.result() == 0.75


class TestExpvalTranspiler:
    """Test transpiler with ExpvalOp and Observable bindings."""

    def test_separate_pass_creates_expval_segment(self):
        """Test that separate pass creates ExpvalSegment for ExpvalOp."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler
        from qamomile.circuit.transpiler.segments import SegmentKind

        @qm.qkernel
        def test_kernel(q0: qm.Qubit, q1: qm.Qubit, H: qm.Observable) -> qm.Float:
            q0 = qm.h(q0)
            exp_val = qm.expval((q0, q1), H)
            return exp_val

        transpiler = QiskitTranspiler()

        # Use lower-level API to test separate pass
        block = transpiler.to_block(test_kernel)
        linear = transpiler.inline(block)
        separated = transpiler.separate(linear)

        # Should have expval segment
        assert separated.expval is not None
        assert separated.expval.kind == SegmentKind.EXPVAL
        assert separated.expval.result_ref != ""

    def test_transpile_with_hamiltonian_binding(self):
        """Test full transpilation with Hamiltonian binding."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        # Build Hamiltonian in Python
        H = qm_o.Z(0) * qm_o.Z(1) + 0.5 * (qm_o.X(0) + qm_o.X(1))

        @qm.qkernel
        def vqe(n: qm.UInt, H: qm.Observable) -> qm.Float:
            q = qm.qubit_array(n, "q")
            q[0] = qm.h(q[0])
            q[0], q[1] = qm.cx(q[0], q[1])
            return qm.expval(q, H)

        transpiler = QiskitTranspiler()

        # Transpile with Hamiltonian in bindings
        executable = transpiler.transpile(vqe, bindings={"H": H, "n": 2})

        # Should have compiled expval segment
        assert len(executable.compiled_expval) == 1
        compiled_expval = executable.compiled_expval[0]

        # Check hamiltonian field
        assert isinstance(compiled_expval.hamiltonian, qm_o.Hamiltonian)

    def test_transpile_without_hamiltonian_binding_raises(self):
        """Test that transpilation without Hamiltonian binding raises error."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qm.qkernel
        def vqe(n: qm.UInt, H: qm.Observable) -> qm.Float:
            q = qm.qubit_array(n, "q")
            return qm.expval(q, H)

        transpiler = QiskitTranspiler()

        # Should raise RuntimeError because H is not in bindings
        with pytest.raises(RuntimeError, match="Observable.*not found in bindings"):
            transpiler.transpile(vqe, bindings={"n": 2})


class TestExpvalContractValidation:
    """Test fail-closed validation for expval usage patterns in transpiler."""

    def test_multiple_expval_rejected(self):
        """Two expval operations should be rejected by separate pass."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler
        from qamomile.circuit.transpiler.errors import SeparationError

        @qm.qkernel
        def bad(
            n: qm.UInt, H1: qm.Observable, H2: qm.Observable
        ) -> tuple[qm.Float, qm.Float]:
            q1 = qm.qubit_array(n, "q1")
            q2 = qm.qubit_array(n, "q2")
            q1[0] = qm.h(q1[0])
            e1 = qm.expval(q1, H1)
            e2 = qm.expval(q2, H2)
            return e1, e2

        transpiler = QiskitTranspiler()
        with pytest.raises(SeparationError, match="Multiple expval"):
            transpiler.transpile(
                bad,
                bindings={
                    "n": 2,
                    "H1": qm_o.Z(0),
                    "H2": qm_o.Z(0),
                },
            )

    def test_expval_after_quantum_op_rejected(self):
        """Quantum operations after expval should be rejected."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler
        from qamomile.circuit.transpiler.errors import SeparationError

        @qm.qkernel
        def bad(n: qm.UInt, H: qm.Observable) -> qm.Float:
            q = qm.qubit_array(n, "q")
            q[0] = qm.h(q[0])
            result = qm.expval(q, H)
            q2 = qm.qubit_array(n, "q2")
            q2[0] = qm.x(q2[0])  # quantum op after expval
            return result

        transpiler = QiskitTranspiler()
        with pytest.raises(SeparationError, match="after expval"):
            transpiler.transpile(
                bad,
                bindings={"n": 2, "H": qm_o.Z(0)},
            )

    @pytest.mark.parametrize(
        "kind",
        ["for", "for_items", "if", "while"],
        ids=["ForOperation", "ForItemsOperation", "IfOperation", "WhileOperation"],
    )
    def test_expval_inside_control_flow_rejected(self, kind: str):
        """expval inside any control-flow kind should be rejected."""
        from qamomile.circuit.ir.block import Block
        from qamomile.circuit.ir.operation.control_flow import (
            ForItemsOperation,
            ForOperation,
            IfOperation,
            WhileOperation,
        )
        from qamomile.circuit.ir.operation.expval import ExpvalOp
        from qamomile.circuit.ir.types.primitives import BitType, FloatType, UIntType
        from qamomile.circuit.ir.value import Value
        from qamomile.circuit.transpiler.errors import SeparationError
        from qamomile.circuit.transpiler.passes.separate import SeparatePass

        qubits_val = Value(type=UIntType(), name="q")
        obs_val = Value(type=UIntType(), name="H")
        result_val = Value(type=FloatType(), name="ev")
        inner_expval = ExpvalOp(operands=[qubits_val, obs_val], results=[result_val])

        if kind == "for":
            op = ForOperation(
                operands=[
                    Value(type=UIntType(), name="start"),
                    Value(type=UIntType(), name="stop"),
                    Value(type=UIntType(), name="step"),
                ],
                results=[],
                loop_var="i",
                operations=[inner_expval],
            )
        elif kind == "for_items":
            op = ForItemsOperation(
                operands=[Value(type=UIntType(), name="items")],
                results=[],
                key_vars=["k"],
                value_var="v",
                operations=[inner_expval],
            )
        elif kind == "if":
            op = IfOperation(
                operands=[Value(type=BitType(), name="cond")],
                results=[],
                true_operations=[inner_expval],
                false_operations=[],
                phi_ops=[],
            )
        else:  # while
            op = WhileOperation(
                operands=[Value(type=BitType(), name="cond")],
                results=[],
                operations=[inner_expval],
            )

        sep = SeparatePass()
        with pytest.raises(SeparationError, match="expval inside control flow"):
            sep._validate_expval_contract(Block(operations=[op]))

    def test_expval_inside_if_false_branch_rejected(self):
        """expval in IfOperation.false_operations should be rejected."""
        from qamomile.circuit.ir.block import Block
        from qamomile.circuit.ir.operation.control_flow import IfOperation
        from qamomile.circuit.ir.operation.expval import ExpvalOp
        from qamomile.circuit.ir.types.primitives import BitType, FloatType, UIntType
        from qamomile.circuit.ir.value import Value
        from qamomile.circuit.transpiler.errors import SeparationError
        from qamomile.circuit.transpiler.passes.separate import SeparatePass

        inner_expval = ExpvalOp(
            operands=[
                Value(type=UIntType(), name="q"),
                Value(type=UIntType(), name="H"),
            ],
            results=[Value(type=FloatType(), name="ev")],
        )
        if_op = IfOperation(
            operands=[Value(type=BitType(), name="cond")],
            results=[],
            true_operations=[],
            false_operations=[inner_expval],
            phi_ops=[],
        )

        sep = SeparatePass()
        with pytest.raises(SeparationError, match="expval inside control flow"):
            sep._validate_expval_contract(Block(operations=[if_op]))

    def test_expval_inside_nested_control_flow_rejected(self):
        """expval nested two levels deep in control flow should be rejected."""
        from qamomile.circuit.ir.block import Block
        from qamomile.circuit.ir.operation.control_flow import ForOperation, IfOperation
        from qamomile.circuit.ir.operation.expval import ExpvalOp
        from qamomile.circuit.ir.types.primitives import BitType, FloatType, UIntType
        from qamomile.circuit.ir.value import Value
        from qamomile.circuit.transpiler.errors import SeparationError
        from qamomile.circuit.transpiler.passes.separate import SeparatePass

        inner_expval = ExpvalOp(
            operands=[
                Value(type=UIntType(), name="q"),
                Value(type=UIntType(), name="H"),
            ],
            results=[Value(type=FloatType(), name="ev")],
        )
        # ForOperation -> IfOperation -> ExpvalOp (2-level nesting)
        nested_if = IfOperation(
            operands=[Value(type=BitType(), name="cond")],
            results=[],
            true_operations=[inner_expval],
            false_operations=[],
            phi_ops=[],
        )
        outer_for = ForOperation(
            operands=[
                Value(type=UIntType(), name="start"),
                Value(type=UIntType(), name="stop"),
                Value(type=UIntType(), name="step"),
            ],
            results=[],
            loop_var="i",
            operations=[nested_if],
        )

        sep = SeparatePass()
        with pytest.raises(SeparationError, match="expval inside control flow"):
            sep._validate_expval_contract(Block(operations=[outer_for]))

    def test_expval_classical_after_expval_rejected(self):
        """Classical operations after expval should also be rejected."""
        from qamomile.circuit.ir.block import Block
        from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
        from qamomile.circuit.ir.operation.expval import ExpvalOp
        from qamomile.circuit.ir.types.primitives import FloatType, UIntType
        from qamomile.circuit.ir.value import Value
        from qamomile.circuit.transpiler.errors import SeparationError
        from qamomile.circuit.transpiler.passes.separate import SeparatePass

        qubits_val = Value(type=UIntType(), name="q")
        obs_val = Value(type=UIntType(), name="H")
        result_val = Value(type=FloatType(), name="ev")
        const_val = Value(type=FloatType(), name="one")
        add_result = Value(type=FloatType(), name="sum")

        expval_op = ExpvalOp(
            operands=[qubits_val, obs_val],
            results=[result_val],
        )
        # Classical BinOp (e.g., ev + 1.0) after expval
        binop = BinOp(
            operands=[result_val, const_val],
            results=[add_result],
            kind=BinOpKind.ADD,
        )
        block = Block(operations=[expval_op, binop])

        sep = SeparatePass()
        with pytest.raises(SeparationError, match="after expval"):
            sep._validate_expval_contract(block)

    def test_single_toplevel_expval_still_works(self):
        """Single top-level expval should still work end-to-end."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        H = qm_o.Z(0) * qm_o.Z(1) + 0.5 * (qm_o.X(0) + qm_o.X(1))

        @qm.qkernel
        def vqe(n: qm.UInt, H: qm.Observable) -> qm.Float:
            q = qm.qubit_array(n, "q")
            q[0] = qm.h(q[0])
            q[0], q[1] = qm.cx(q[0], q[1])
            return qm.expval(q, H)

        transpiler = QiskitTranspiler()
        executable = transpiler.transpile(vqe, bindings={"H": H, "n": 2})
        assert len(executable.compiled_expval) == 1


class TestExpvalTupleValidation:
    """Test that invalid tuple members are rejected at build time."""

    def test_tuple_non_qubit_rejected_before_ir(self):
        """Non-Qubit tuple member should be rejected before ExpvalOp is created."""

        @qm.qkernel
        def bad(theta: qm.Float, H: qm.Observable) -> qm.Float:
            x = theta + 1.0
            return qm.expval((x,), H)

        with pytest.raises(TypeError, match="expval tuple expects only Qubit elements"):
            bad.build()

    def test_tuple_invalid_rejected_before_transpile(self):
        """Invalid tuple should fail at build, never reaching transpile."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qm.qkernel
        def bad(theta: qm.Float, H: qm.Observable) -> qm.Float:
            q = qm.qubit(name="q")
            q = qm.h(q)
            return qm.expval((q, theta), H)

        # Should raise TypeError at build time (inside transpile), not silently pass
        with pytest.raises(TypeError, match="expval tuple expects only Qubit elements"):
            QiskitTranspiler().transpile(bad, bindings={"H": qm_o.Z(0)})


class TestExpvalRemapContract:
    """Test that expval Hamiltonian remap is compile-time only (no runtime double remap)."""

    def test_runtime_does_not_double_remap(self):
        """CompiledExpvalSegment.hamiltonian should be passed through as-is at runtime."""
        from qamomile.circuit.transpiler.compiled_segments import (
            CompiledExpvalSegment,
            CompiledQuantumSegment,
        )
        from qamomile.circuit.transpiler.executable import ExecutableProgram
        from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata
        from qamomile.circuit.transpiler.quantum_executor import QuantumExecutor
        from qamomile.circuit.transpiler.segments import ExpvalSegment, QuantumSegment

        class RecorderExecutor(QuantumExecutor[object]):
            def __init__(self):
                self.last_h = None

            def bind_parameters(self, circuit, bindings, metadata):
                return circuit

            def execute(self, circuit, shots):
                return {"0": shots}

            def estimate(self, circuit, hamiltonian, params=None):
                self.last_h = hamiltonian
                return 0.0

        # Compile-time remapped Hamiltonian: Z(0)->Z(1), Z(1)->Z(0)
        compiled_h = (qm_o.Z(0) + 2 * qm_o.Z(1)).remap_qubits({0: 1, 1: 0})

        exe = ExecutableProgram(
            compiled_quantum=[
                CompiledQuantumSegment(
                    segment=QuantumSegment(operations=[]),
                    circuit=object(),
                    parameter_metadata=ParameterMetadata(),
                )
            ],
            compiled_expval=[
                CompiledExpvalSegment(
                    segment=ExpvalSegment(result_ref="ev"),
                    hamiltonian=compiled_h,
                    result_ref="ev",
                    qubit_map={0: 1, 1: 0},
                )
            ],
            execution_order=[("quantum", 0), ("expval", 0)],
        )
        executor = RecorderExecutor()
        exe.run(executor).result()

        # Runtime must pass compiled_h through without additional remap
        assert executor.last_h is compiled_h

    def test_build_qubit_map_vector_prefix_keys(self):
        """_build_qubit_map should resolve Vector[Qubit] via prefix-key scan."""
        pytest.importorskip("qiskit")
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.ir.value import ArrayValue, Value
        from qamomile.circuit.transpiler.compiled_segments import CompiledQuantumSegment
        from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata
        from qamomile.circuit.transpiler.segments import QuantumSegment
        from qamomile.qiskit.transpiler import QiskitEmitPass

        size = Value(type=UIntType(), name="n", params={"const": 2})
        vec = ArrayValue(type=QubitType(), name="q", shape=(size,))

        compiled_quantum = [
            CompiledQuantumSegment(
                segment=QuantumSegment(operations=[]),
                circuit=object(),
                qubit_map={f"{vec.uuid}_0": 5, f"{vec.uuid}_1": 3},
                parameter_metadata=ParameterMetadata(),
            )
        ]

        result = QiskitEmitPass(bindings={})._build_qubit_map(vec, 0, compiled_quantum)
        assert result == {0: 5, 1: 3}

    def test_build_qubit_map_vector_partial_raises(self):
        """_build_qubit_map should raise EmitError on partial vector prefix keys."""
        pytest.importorskip("qiskit")
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.ir.value import ArrayValue, Value
        from qamomile.circuit.transpiler.compiled_segments import CompiledQuantumSegment
        from qamomile.circuit.transpiler.errors import EmitError
        from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata
        from qamomile.circuit.transpiler.segments import QuantumSegment
        from qamomile.qiskit.transpiler import QiskitEmitPass

        size = Value(type=UIntType(), name="n", params={"const": 3})
        vec = ArrayValue(type=QubitType(), name="q", shape=(size,))

        # Only indices 0 and 2 present (missing index 1) → partial map
        compiled_quantum = [
            CompiledQuantumSegment(
                segment=QuantumSegment(operations=[]),
                circuit=object(),
                qubit_map={f"{vec.uuid}_0": 5, f"{vec.uuid}_2": 7},
                parameter_metadata=ParameterMetadata(),
            )
        ]

        with pytest.raises(EmitError, match="Incomplete vector qubit mapping"):
            QiskitEmitPass(bindings={})._build_qubit_map(vec, 0, compiled_quantum)

    def test_build_qubit_map_vector_missing_tail_raises(self):
        """_build_qubit_map should raise EmitError when tail indices are missing."""
        pytest.importorskip("qiskit")
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.ir.value import ArrayValue, Value
        from qamomile.circuit.transpiler.compiled_segments import CompiledQuantumSegment
        from qamomile.circuit.transpiler.errors import EmitError
        from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata
        from qamomile.circuit.transpiler.segments import QuantumSegment
        from qamomile.qiskit.transpiler import QiskitEmitPass

        size = Value(type=UIntType(), name="n", params={"const": 3})
        vec = ArrayValue(type=QubitType(), name="q", shape=(size,))

        # Only indices 0 and 1 present for size-3 vector → missing tail
        compiled_quantum = [
            CompiledQuantumSegment(
                segment=QuantumSegment(operations=[]),
                circuit=object(),
                qubit_map={f"{vec.uuid}_0": 5, f"{vec.uuid}_1": 3},
                parameter_metadata=ParameterMetadata(),
            )
        ]

        with pytest.raises(EmitError, match="Incomplete vector qubit mapping"):
            QiskitEmitPass(bindings={})._build_qubit_map(vec, 0, compiled_quantum)

    def test_build_qubit_map_vector_extra_index_raises(self):
        """_build_qubit_map should raise EmitError on gapless extra indices."""
        pytest.importorskip("qiskit")
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.ir.value import ArrayValue, Value
        from qamomile.circuit.transpiler.compiled_segments import CompiledQuantumSegment
        from qamomile.circuit.transpiler.errors import EmitError
        from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata
        from qamomile.circuit.transpiler.segments import QuantumSegment
        from qamomile.qiskit.transpiler import QiskitEmitPass

        size = Value(type=UIntType(), name="n", params={"const": 3})
        vec = ArrayValue(type=QubitType(), name="q", shape=(size,))

        # Indices 0,1,2,3 present for size-3 vector → extra index
        compiled_quantum = [
            CompiledQuantumSegment(
                segment=QuantumSegment(operations=[]),
                circuit=object(),
                qubit_map={
                    f"{vec.uuid}_0": 5,
                    f"{vec.uuid}_1": 3,
                    f"{vec.uuid}_2": 8,
                    f"{vec.uuid}_3": 9,
                },
                parameter_metadata=ParameterMetadata(),
            )
        ]

        with pytest.raises(EmitError, match="Incomplete vector qubit mapping"):
            QiskitEmitPass(bindings={})._build_qubit_map(vec, 0, compiled_quantum)

    def test_build_qubit_map_vector_empty_map_raises(self):
        """_build_qubit_map should raise EmitError on empty map for positive-size vector."""
        pytest.importorskip("qiskit")
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.ir.value import ArrayValue, Value
        from qamomile.circuit.transpiler.compiled_segments import CompiledQuantumSegment
        from qamomile.circuit.transpiler.errors import EmitError
        from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata
        from qamomile.circuit.transpiler.segments import QuantumSegment
        from qamomile.qiskit.transpiler import QiskitEmitPass

        size = Value(type=UIntType(), name="n", params={"const": 3})
        vec = ArrayValue(type=QubitType(), name="q", shape=(size,))

        # No prefix keys at all for positive-size vector → empty map
        compiled_quantum = [
            CompiledQuantumSegment(
                segment=QuantumSegment(operations=[]),
                circuit=object(),
                qubit_map={},
                parameter_metadata=ParameterMetadata(),
            )
        ]

        with pytest.raises(EmitError, match="Incomplete vector qubit mapping"):
            QiskitEmitPass(bindings={})._build_qubit_map(vec, 0, compiled_quantum)

    def test_build_qubit_map_tuple_still_works(self):
        """_build_qubit_map still resolves tuple synthetic arrays correctly."""
        pytest.importorskip("qiskit")
        from qamomile.circuit.ir.types.primitives import QubitType
        from qamomile.circuit.ir.value import ArrayValue, Value
        from qamomile.circuit.transpiler.compiled_segments import CompiledQuantumSegment
        from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata
        from qamomile.circuit.transpiler.segments import QuantumSegment
        from qamomile.qiskit.transpiler import QiskitEmitPass

        q0 = Value(type=QubitType(), name="q0")
        q1 = Value(type=QubitType(), name="q1")
        arr = ArrayValue(
            type=QubitType(),
            name="expval_qubits",
            shape=(),
            params={"qubit_values": [q0, q1]},
        )

        compiled_quantum = [
            CompiledQuantumSegment(
                segment=QuantumSegment(operations=[]),
                circuit=object(),
                qubit_map={q0.uuid: 3, q1.uuid: 1},
                parameter_metadata=ParameterMetadata(),
            )
        ]

        result = QiskitEmitPass(bindings={})._build_qubit_map(arr, 0, compiled_quantum)
        assert result == {0: 3, 1: 1}

    def test_build_qubit_map_borrowed_scalar_element(self):
        """_build_qubit_map resolves borrowed scalar element (e.g. q[1])."""
        pytest.importorskip("qiskit")
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.ir.value import ArrayValue, Value
        from qamomile.circuit.transpiler.compiled_segments import CompiledQuantumSegment
        from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata
        from qamomile.circuit.transpiler.segments import QuantumSegment
        from qamomile.qiskit.transpiler import QiskitEmitPass

        # Create a qubit array and a borrowed element q[1]
        q_array = ArrayValue(
            type=QubitType(),
            name="q",
            shape=(Value(type=UIntType(), name="n", params={"const": 2}),),
        )
        idx_val = Value(type=UIntType(), name="idx", params={"const": 1})
        borrowed_elem = Value(
            type=QubitType(),
            name="q[1]",
            parent_array=q_array,
            element_indices=(idx_val,),
        )

        compiled_quantum = [
            CompiledQuantumSegment(
                segment=QuantumSegment(operations=[]),
                circuit=object(),
                qubit_map={f"{q_array.uuid}_0": 0, f"{q_array.uuid}_1": 7},
                parameter_metadata=ParameterMetadata(),
            )
        ]

        result = QiskitEmitPass(bindings={})._build_qubit_map(
            borrowed_elem, 0, compiled_quantum
        )
        assert result == {0: 7}

    def test_build_qubit_map_borrowed_tuple_elements(self):
        """_build_qubit_map resolves borrowed elements inside a tuple."""
        pytest.importorskip("qiskit")
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.ir.value import ArrayValue, Value
        from qamomile.circuit.transpiler.compiled_segments import CompiledQuantumSegment
        from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata
        from qamomile.circuit.transpiler.segments import QuantumSegment
        from qamomile.qiskit.transpiler import QiskitEmitPass

        q_array = ArrayValue(
            type=QubitType(),
            name="q",
            shape=(Value(type=UIntType(), name="n", params={"const": 3}),),
        )
        idx0 = Value(type=UIntType(), name="i0", params={"const": 1})
        idx1 = Value(type=UIntType(), name="i1", params={"const": 2})
        borrowed_q1 = Value(
            type=QubitType(),
            name="q[1]",
            parent_array=q_array,
            element_indices=(idx0,),
        )
        borrowed_q2 = Value(
            type=QubitType(),
            name="q[2]",
            parent_array=q_array,
            element_indices=(idx1,),
        )

        arr = ArrayValue(
            type=QubitType(),
            name="expval_qubits",
            shape=(),
            params={"qubit_values": [borrowed_q1, borrowed_q2]},
        )

        compiled_quantum = [
            CompiledQuantumSegment(
                segment=QuantumSegment(operations=[]),
                circuit=object(),
                qubit_map={
                    f"{q_array.uuid}_0": 0,
                    f"{q_array.uuid}_1": 5,
                    f"{q_array.uuid}_2": 3,
                },
                parameter_metadata=ParameterMetadata(),
            )
        ]

        result = QiskitEmitPass(bindings={})._build_qubit_map(arr, 0, compiled_quantum)
        assert result == {0: 5, 1: 3}

    def test_build_qubit_map_unresolved_scalar_raises(self):
        """_build_qubit_map raises EmitError for unresolvable scalar target."""
        pytest.importorskip("qiskit")
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.ir.value import ArrayValue, Value
        from qamomile.circuit.transpiler.compiled_segments import CompiledQuantumSegment
        from qamomile.circuit.transpiler.errors import EmitError
        from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata
        from qamomile.circuit.transpiler.segments import QuantumSegment
        from qamomile.qiskit.transpiler import QiskitEmitPass

        # Borrowed element whose parent array is not in qubit_map
        q_array = ArrayValue(
            type=QubitType(),
            name="q",
            shape=(Value(type=UIntType(), name="n", params={"const": 2}),),
        )
        idx_val = Value(type=UIntType(), name="idx", params={"const": 0})
        borrowed_elem = Value(
            type=QubitType(),
            name="q[0]",
            parent_array=q_array,
            element_indices=(idx_val,),
        )

        compiled_quantum = [
            CompiledQuantumSegment(
                segment=QuantumSegment(operations=[]),
                circuit=object(),
                qubit_map={},  # Empty - nothing to resolve against
                parameter_metadata=ParameterMetadata(),
            )
        ]

        with pytest.raises(EmitError, match="Cannot resolve expval target"):
            QiskitEmitPass(bindings={})._build_qubit_map(
                borrowed_elem, 0, compiled_quantum
            )

    def test_build_qubit_map_unresolved_tuple_element_raises(self):
        """_build_qubit_map raises EmitError for unresolvable tuple element."""
        pytest.importorskip("qiskit")
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.ir.value import ArrayValue, Value
        from qamomile.circuit.transpiler.compiled_segments import CompiledQuantumSegment
        from qamomile.circuit.transpiler.errors import EmitError
        from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata
        from qamomile.circuit.transpiler.segments import QuantumSegment
        from qamomile.qiskit.transpiler import QiskitEmitPass

        q_array = ArrayValue(
            type=QubitType(),
            name="q",
            shape=(Value(type=UIntType(), name="n", params={"const": 2}),),
        )
        # Symbolic (unbound) index
        symbolic_idx = Value(type=UIntType(), name="i")
        borrowed_elem = Value(
            type=QubitType(),
            name="q[i]",
            parent_array=q_array,
            element_indices=(symbolic_idx,),
        )

        arr = ArrayValue(
            type=QubitType(),
            name="expval_qubits",
            shape=(),
            params={"qubit_values": [borrowed_elem]},
        )

        compiled_quantum = [
            CompiledQuantumSegment(
                segment=QuantumSegment(operations=[]),
                circuit=object(),
                qubit_map={f"{q_array.uuid}_0": 0, f"{q_array.uuid}_1": 1},
                parameter_metadata=ParameterMetadata(),
            )
        ]

        with pytest.raises(EmitError, match="Cannot resolve expval target"):
            QiskitEmitPass(bindings={})._build_qubit_map(arr, 0, compiled_quantum)


class TestHamiltonianRemapQubits:
    """Test Hamiltonian.remap_qubits method."""

    def test_hamiltonian_remap_qubits(self):
        """Test Hamiltonian.remap_qubits."""
        # Create Hamiltonian with Z on qubit 0
        H = qm_o.Z(0)

        # Remap qubit 0 to qubit 5
        H_remapped = H.remap_qubits({0: 5})

        # Check that qubit was remapped
        assert H_remapped.num_qubits == 6  # max index is 5

        # Check that the term is now on qubit 5
        assert len(H_remapped.terms) == 1
        operators = list(H_remapped.terms.keys())[0]
        assert len(operators) == 1
        assert operators[0].index == 5
        assert operators[0].pauli == qm_o.Pauli.Z

    def test_hamiltonian_remap_empty_returns_same(self):
        """Test that empty mapping preserves the Hamiltonian."""
        H = qm_o.Z(0)
        H_remapped = H.remap_qubits({})

        # Should be unchanged
        assert H_remapped.num_qubits == H.num_qubits
        assert H_remapped.terms == H.terms
