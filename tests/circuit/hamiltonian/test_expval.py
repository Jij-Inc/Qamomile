"""Tests for the expval (expectation value) operation."""

import pytest
import qamomile.circuit as qm
from qamomile.circuit.ir.operation import ExpvalOp
from qamomile.circuit.ir.operation.hamiltonian_ops import PauliCreateOp


class TestExpvalFrontend:
    """Test expval frontend function."""

    def test_expval_basic(self):
        """Test basic expval with single qubit and Z observable."""

        @qm.qkernel
        def test_kernel(q0: qm.Qubit, q1: qm.Qubit) -> qm.Float:
            q0 = qm.h(q0)
            H = qm.pauli.Z(0)
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
        assert expval_op.hamiltonian is not None

    def test_expval_with_complex_hamiltonian(self):
        """Test expval with multi-term Hamiltonian."""

        @qm.qkernel
        def test_kernel(q0: qm.Qubit, q1: qm.Qubit) -> qm.Float:
            q0 = qm.h(q0)
            q0, q1 = qm.cx(q0, q1)

            # Create ZZ + 0.5*(X0 + X1) Hamiltonian
            H = qm.pauli.Z(0) * qm.pauli.Z(1) + 0.5 * (qm.pauli.X(0) + qm.pauli.X(1))
            exp_val = qm.expval((q0, q1), H)
            return exp_val

        block = test_kernel.build()

        # Should have ExpvalOp
        has_expval = any(isinstance(op, ExpvalOp) for op in block.operations)
        assert has_expval

        # Should have multiple PauliCreateOp operations
        pauli_ops = [op for op in block.operations if isinstance(op, PauliCreateOp)]
        assert len(pauli_ops) >= 3  # Z(0), Z(1), X(0), X(1)


class TestExpvalOp:
    """Test ExpvalOp IR operation."""

    def test_operation_kind_is_hybrid(self):
        """Test that ExpvalOp has HYBRID operation kind."""
        from qamomile.circuit.ir.operation.operation import OperationKind

        @qm.qkernel
        def test_kernel(q0: qm.Qubit) -> qm.Float:
            H = qm.pauli.Z(0)
            exp_val = qm.expval((q0,), H)
            return exp_val

        block = test_kernel.build()
        expval_op = next(op for op in block.operations if isinstance(op, ExpvalOp))

        assert expval_op.operation_kind == OperationKind.HYBRID

    def test_expval_output_type_is_float(self):
        """Test that ExpvalOp output has FloatType."""
        from qamomile.circuit.ir.types.primitives import FloatType

        @qm.qkernel
        def test_kernel(q0: qm.Qubit) -> qm.Float:
            H = qm.pauli.Z(0)
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

        segment = ExpvalSegment()
        # CompiledExpvalSegment expects an Observable, so we can't fully instantiate
        # but we can check the dataclass fields exist
        import dataclasses

        fields = {f.name for f in dataclasses.fields(CompiledExpvalSegment)}
        assert "segment" in fields
        assert "observable" in fields
        assert "quantum_segment_index" in fields
        assert "result_ref" in fields
        assert "qubit_map" in fields  # New field for Pauli index mapping


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
    """Test transpiler with ExpvalOp."""

    def test_separate_pass_creates_expval_segment(self):
        """Test that separate pass creates ExpvalSegment for ExpvalOp."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler
        from qamomile.circuit.transpiler.segments import SegmentKind

        @qm.qkernel
        def test_kernel(q0: qm.Qubit, q1: qm.Qubit) -> qm.Float:
            q0 = qm.h(q0)
            H = qm.pauli.Z(0)
            exp_val = qm.expval((q0, q1), H)
            return exp_val

        transpiler = QiskitTranspiler()

        # Use lower-level API to test separate pass
        block = transpiler.to_block(test_kernel)
        linear = transpiler.inline(block)
        separated = transpiler.separate(linear)

        # Should have expval segment
        assert len(separated.expval_segments()) == 1
        assert separated.expval_segments()[0].kind == SegmentKind.EXPVAL
        assert separated.expval_segments()[0].result_ref != ""


class TestObservableRemapQubits:
    """Test Observable.remap_qubits method."""

    def test_observable_remap_qubits(self):
        """Test Observable.remap_qubits delegates to ConcreteHamiltonian."""
        from qamomile.circuit.observable import Observable, ConcreteHamiltonian
        from qamomile.circuit.ir.types.hamiltonian import PauliKind

        h = ConcreteHamiltonian.single_pauli(PauliKind.Z, 0, 1.0)
        obs = Observable(h)

        obs_remapped = obs.remap_qubits({0: 5})

        assert obs_remapped.num_qubits == 6  # max index is 5
        assert ((PauliKind.Z, 5),) in obs_remapped.hamiltonian.terms

    def test_observable_remap_empty_returns_same_hamiltonian(self):
        """Test that empty mapping preserves the Hamiltonian."""
        from qamomile.circuit.observable import Observable, ConcreteHamiltonian
        from qamomile.circuit.ir.types.hamiltonian import PauliKind

        h = ConcreteHamiltonian.single_pauli(PauliKind.Z, 0, 1.0)
        obs = Observable(h)

        obs_remapped = obs.remap_qubits({})

        # Hamiltonian should be the same object
        assert obs_remapped.hamiltonian is h
