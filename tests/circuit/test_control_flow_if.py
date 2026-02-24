"""Tests for if-else control flow in @qkernel."""

import pytest

import qamomile.circuit as qm
from qamomile.circuit.frontend.handle import Qubit, Vector
from qamomile.circuit.frontend.handle.primitives import Float
from qamomile.circuit.frontend.operation.control_flow import _create_phi_for_values
from qamomile.circuit.frontend.qkernel import qkernel
from qamomile.circuit.ir.operation.arithmetic_operations import PhiOp
from qamomile.circuit.ir.operation.control_flow import IfOperation
from qamomile.circuit.ir.types.primitives import BitType, FloatType, QubitType
from qamomile.circuit.ir.value import Value


class TestIfElseBothBranches:
    """Both true and false branches contain quantum operations."""

    def test_if_else_both_branches_same_qubit(self):
        """Applying different gates to the same qubit in both branches.

        The AST transformer generates:
          def _cond_0(cond, q1): return cond
          def _body_0(cond, q1): q1 = qm.x(q1); return cond, q1
          def _body_1(cond, q1): q1 = qm.h(q1); return cond, q1
          cond, q1 = emit_if(_cond_0, _body_0, _body_1, [cond, q1])

        Previously, emit_if passed the same q1 object to both branches,
        so q1._consumed=True after the true branch caused
        QubitConsumedError in the false branch.
        """

        @qkernel
        def circuit(q0: Qubit, q1: Qubit) -> Qubit:
            cond = qm.measure(q0)
            if cond:
                q1 = qm.x(q1)
            else:
                q1 = qm.h(q1)
            return q1

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) == 1
        assert len(if_ops[0].false_operations) == 1
        assert len(if_ops[0].results) == 2
        results = if_ops[0].results
        assert results[0].type == BitType()
        assert results[1].type == QubitType()
        assert len(if_ops[0].phi_ops) == 2
        for phi in if_ops[0].phi_ops:
            assert isinstance(phi, PhiOp)
            assert len(phi.operands) == 3
            assert len(phi.results) == 1

    def test_if_else_multiple_qubits_in_branches(self):
        """Multiple qubits operated on in both branches should work."""

        @qkernel
        def circuit(q0: Qubit, q1: Qubit, q2: Qubit) -> tuple[Qubit, Qubit]:
            cond = qm.measure(q0)
            if cond:
                q1 = qm.x(q1)
                q2 = qm.h(q2)
            else:
                q1 = qm.h(q1)
                q2 = qm.x(q2)
            return q1, q2

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) == 2
        assert len(if_ops[0].false_operations) == 2
        assert len(if_ops[0].results) == 3
        results = if_ops[0].results
        assert results[0].type == BitType()
        assert results[1].type == QubitType()
        assert results[2].type == QubitType()
        assert len(if_ops[0].phi_ops) == 3
        for phi in if_ops[0].phi_ops:
            assert isinstance(phi, PhiOp)
            assert len(phi.operands) == 3
            assert len(phi.results) == 1

    def test_if_else_diff(self):
        """If without else should work. The false branch is empty (returns vars as-is),
        so no qubit is consumed there."""

        @qkernel
        def circuit(q0: Qubit, q1: Qubit) -> Qubit:
            cond = qm.measure(q0)
            if cond:
                q1 = qm.x(q1)
            else:
                q1 = qm.h(q1)
                q1 = qm.x(q1)
            return q1

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) == 1
        assert len(if_ops[0].false_operations) == 2
        assert len(if_ops[0].results) == 2
        results = if_ops[0].results
        assert results[0].type == BitType()
        assert results[1].type == QubitType()
        assert len(if_ops[0].phi_ops) == 2
        for phi in if_ops[0].phi_ops:
            assert isinstance(phi, PhiOp)
            assert len(phi.operands) == 3
            assert len(phi.results) == 1


class TestIfWithoutElse:
    """If-only or if with an empty else branch (no quantum operations in false branch)."""

    def test_if_only_no_else(self):
        """If without else should work. The false branch is empty (returns vars as-is),
        so no qubit is consumed there."""

        @qkernel
        def circuit(q0: Qubit, q1: Qubit) -> Qubit:
            cond = qm.measure(q0)
            if cond:
                q1 = qm.x(q1)
            return q1

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) == 1
        assert len(if_ops[0].false_operations) == 0
        assert len(if_ops[0].results) == 2
        results = if_ops[0].results
        assert results[0].type == BitType()
        assert results[1].type == QubitType()
        assert len(if_ops[0].phi_ops) == 2
        for phi in if_ops[0].phi_ops:
            assert isinstance(phi, PhiOp)
            assert len(phi.operands) == 3
            assert len(phi.results) == 1

    def test_if_else_classical_only_in_one_branch(self):
        """One branch with no qubit operations should work.

        The false branch only returns the qubit unchanged, producing
        a PhiOp that merges the gate-applied and identity paths.
        """

        @qkernel
        def circuit(q0: Qubit, q1: Qubit) -> Qubit:
            cond = qm.measure(q0)
            if cond:
                q1 = qm.x(q1)
            else:
                pass  # no qubit operations
            return q1

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) == 1
        assert len(if_ops[0].false_operations) == 0
        assert len(if_ops[0].results) == 2
        results = if_ops[0].results
        assert results[0].type == BitType()
        assert results[1].type == QubitType()
        assert len(if_ops[0].phi_ops) == 2
        for phi in if_ops[0].phi_ops:
            assert isinstance(phi, PhiOp)
            assert len(phi.operands) == 3
            assert len(phi.results) == 1


class TestIfElseNested:
    """Nested if-else control flow."""

    def test_nested_if_else(self):
        """Nested if-else inside a branch should build successfully.

        Currently fails due to AST transformer limitation with nested
        control flow.  This test documents the limitation and should be
        updated when nested if-else is supported.
        """

        @qkernel
        def circuit(q0: Qubit, q1: Qubit, q2: Qubit) -> Qubit:
            cond1 = qm.measure(q0)
            if cond1:
                cond2 = qm.measure(q2)
                if cond2:
                    q1 = qm.x(q1)
                else:
                    q1 = qm.h(q1)
            else:
                q1 = qm.h(q1)
            return q1

        graph = circuit.build()

        # Outer IfOperation should exist at the top level
        outer_if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(outer_if_ops) == 1, "Expected 1 outer IfOperation"
        outer_if = outer_if_ops[0]
        assert len(outer_if.true_operations) == 2
        assert len(outer_if.false_operations) == 1
        assert len(outer_if.results) == 3
        results = outer_if.results
        assert results[0].type == BitType()
        assert results[1].type == QubitType()
        assert results[2].type == QubitType()
        assert len(outer_if.phi_ops) == 3
        for phi in outer_if.phi_ops:
            assert isinstance(phi, PhiOp)
            assert len(phi.operands) == 3
            assert len(phi.results) == 1

        # Outer true branch should contain an inner IfOperation
        inner_if_ops = [
            op for op in outer_if.true_operations if isinstance(op, IfOperation)
        ]
        assert len(inner_if_ops) == 1, "Expected 1 inner IfOperation in true branch"
        inner_if = inner_if_ops[0]
        assert len(inner_if.true_operations) == 1
        assert len(inner_if.false_operations) == 1
        assert len(inner_if.results) == 2
        results = inner_if.results
        assert results[0].type == BitType()
        assert results[1].type == QubitType()
        assert len(inner_if.phi_ops) == 2
        for phi in inner_if.phi_ops:
            assert isinstance(phi, PhiOp)
            assert len(phi.operands) == 3
            assert len(phi.results) == 1


class TestIfElseErrorHandling:
    """Error cases for if-else control flow."""

    def test_phi_type_mismatch_raises_type_error(self):
        """_create_phi_for_values should raise TypeError when branch types differ."""
        if_op = IfOperation()
        condition = Value(type=BitType(), name="cond")
        true_val = Qubit(value=Value(type=QubitType(), name="q_true"))
        false_val = Float(value=Value(type=FloatType(), name="f_false"))

        with pytest.raises(TypeError, match="Type mismatch in if-else branches"):
            _create_phi_for_values(condition, true_val, false_val, if_op)


class TestIfElseWithSymbolicVector:
    """Symbolic-sized Vector[Qubit] (parameter) in if-else branches."""

    def test_if_else_symbolic_vector_same_element(self):
        """Different gates on same element of a parameter Vector in each branch."""

        @qkernel
        def circuit(q0: Qubit, qs: Vector[Qubit]) -> Vector[Qubit]:
            cond = qm.measure(q0)
            if cond:
                q = qs[0]
                q = qm.x(q)
                qs[0] = q
            else:
                q = qs[0]
                q = qm.h(q)
                qs[0] = q
            return qs

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) > 0
        assert len(if_ops[0].false_operations) > 0
        assert len(if_ops[0].phi_ops) >= 2  # cond + qs

    def test_if_else_symbolic_vector_different_elements(self):
        """Different elements of a parameter Vector in each branch."""

        @qkernel
        def circuit(q0: Qubit, qs: Vector[Qubit]) -> Vector[Qubit]:
            cond = qm.measure(q0)
            if cond:
                q = qs[0]
                q = qm.x(q)
                qs[0] = q
            else:
                q = qs[1]
                q = qm.h(q)
                qs[1] = q
            return qs

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) > 0
        assert len(if_ops[0].false_operations) > 0

    def test_if_only_symbolic_vector_passthrough(self):
        """Parameter Vector with ops only in true branch, pass-through in else."""

        @qkernel
        def circuit(q0: Qubit, qs: Vector[Qubit]) -> Vector[Qubit]:
            cond = qm.measure(q0)
            if cond:
                q = qs[0]
                q = qm.x(q)
                qs[0] = q
            return qs

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) > 0
        assert len(if_ops[0].false_operations) == 0

    def test_if_else_symbolic_vector_and_qubit_mixed(self):
        """Mixed parameter Vector and individual Qubit in if-else."""

        @qkernel
        def circuit(
            q0: Qubit, q1: Qubit, qs: Vector[Qubit]
        ) -> tuple[Vector[Qubit], Qubit]:
            cond = qm.measure(q0)
            if cond:
                q = qs[0]
                q = qm.x(q)
                qs[0] = q
                q1 = qm.h(q1)
            else:
                q = qs[1]
                q = qm.h(q)
                qs[1] = q
                q1 = qm.x(q1)
            return qs, q1

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].phi_ops) >= 3  # cond, qs, q1

    def test_if_else_symbolic_vector_index_after_merge(self):
        """Indexing a parameter Vector after if-else merge must work."""

        @qkernel
        def circuit(q0: Qubit, qs: Vector[Qubit]) -> Qubit:
            cond = qm.measure(q0)
            if cond:
                q = qs[0]
                q = qm.x(q)
                qs[0] = q
            else:
                q = qs[0]
                q = qm.h(q)
                qs[0] = q
            result = qs[1]
            result = qm.h(result)
            return result

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1

class TestIfElseWithVector:
    """Vector[Qubit] and mixed Vector/Qubit in if-else branches."""

    def test_if_else_vector_element_ops_both_branches(self):
        """Different gates on same Vector element in each branch."""

        @qkernel
        def circuit(q0: Qubit) -> Vector[Qubit]:
            qs = qm.qubit_array(3, "qs")
            cond = qm.measure(q0)
            if cond:
                q = qs[0]
                q = qm.x(q)
                qs[0] = q
            else:
                q = qs[0]
                q = qm.h(q)
                qs[0] = q
            return qs

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) > 0
        assert len(if_ops[0].false_operations) > 0
        assert len(if_ops[0].phi_ops) >= 2  # cond + qs

    def test_if_else_vector_different_elements_per_branch(self):
        """Different elements operated on in each branch (tests _borrowed_indices reset)."""

        @qkernel
        def circuit(q0: Qubit) -> Vector[Qubit]:
            qs = qm.qubit_array(3, "qs")
            cond = qm.measure(q0)
            if cond:
                q = qs[0]
                q = qm.x(q)
                qs[0] = q
            else:
                q = qs[1]
                q = qm.h(q)
                qs[1] = q
            return qs

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1

    def test_if_else_mixed_vector_and_qubit(self):
        """Mixed Vector[Qubit] and individual Qubit in if-else."""

        @qkernel
        def circuit(q0: Qubit, q1: Qubit) -> tuple[Vector[Qubit], Qubit]:
            qs = qm.qubit_array(2, "qs")
            cond = qm.measure(q0)
            if cond:
                q = qs[0]
                q = qm.x(q)
                qs[0] = q
                q1 = qm.h(q1)
            else:
                q = qs[1]
                q = qm.h(q)
                qs[1] = q
                q1 = qm.x(q1)
            return qs, q1

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].phi_ops) >= 3  # cond, qs, q1

    def test_if_else_vector_index_after_merge(self):
        """Indexing Vector after if-else merge must work (directly tests phi merge type)."""

        @qkernel
        def circuit(q0: Qubit) -> Qubit:
            qs = qm.qubit_array(3, "qs")
            cond = qm.measure(q0)
            if cond:
                q = qs[0]
                q = qm.x(q)
                qs[0] = q
            else:
                q = qs[0]
                q = qm.h(q)
                qs[0] = q
            result = qs[1]
            result = qm.h(result)
            return result

        graph = circuit.build()
        # If qs became generic Handle after phi merge, qs[1] would raise AttributeError

    def test_if_only_vector_passthrough(self):
        """Vector with ops in true branch only, pass-through in else."""

        @qkernel
        def circuit(q0: Qubit) -> Vector[Qubit]:
            qs = qm.qubit_array(2, "qs")
            cond = qm.measure(q0)
            if cond:
                q = qs[0]
                q = qm.x(q)
                qs[0] = q
            return qs

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
