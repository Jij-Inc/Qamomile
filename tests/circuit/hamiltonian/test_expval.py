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
        import dataclasses

        from qamomile.circuit.transpiler.compiled_segments import CompiledExpvalSegment

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
        from qamomile.circuit.transpiler.segments import ExpvalStep, SegmentKind
        from qamomile.qiskit import QiskitTranspiler

        @qm.qkernel
        def test_kernel(q0: qm.Qubit, q1: qm.Qubit, H: qm.Observable) -> qm.Float:
            q0 = qm.h(q0)
            exp_val = qm.expval((q0, q1), H)
            return exp_val

        transpiler = QiskitTranspiler()

        # Use lower-level API to test separate pass
        block = transpiler.to_block(test_kernel)
        linear = transpiler.inline(block)
        separated = transpiler.plan(linear)

        # Should have expval segment
        expval_step = next(s for s in separated.steps if isinstance(s, ExpvalStep))
        assert expval_step.segment.kind == SegmentKind.EXPVAL
        assert expval_step.segment.result_ref != ""

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
