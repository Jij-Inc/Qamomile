"""Tests for if-else control flow in @qkernel."""

import pytest

import qamomile.circuit as qm
from qamomile.circuit.frontend.handle import Qubit, UInt, Vector
from qamomile.circuit.frontend.handle.primitives import Float
from qamomile.circuit.frontend.operation.control_flow import _create_phi_for_values
from qamomile.circuit.frontend.qkernel import qkernel
from qamomile.circuit.ir.operation.arithmetic_operations import PhiOp
from qamomile.circuit.ir.operation.control_flow import IfOperation
from qamomile.circuit.ir.operation.gate import (
    GateOperationType,
    MeasureOperation,
    MeasureVectorOperation,
)
from qamomile.circuit.ir.types.primitives import BitType, FloatType, QubitType
from qamomile.circuit.ir.value import Value


class TestIfElseScalarQubit:
    """If-else control flow with scalar Qubit parameters only."""

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
        assert if_ops[0].true_operations[0].gate_type == GateOperationType.X  # type: ignore
        assert len(if_ops[0].false_operations) == 1
        assert if_ops[0].false_operations[0].gate_type == GateOperationType.H  # type: ignore
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
        assert if_ops[0].true_operations[0].gate_type == GateOperationType.X  # type: ignore
        assert if_ops[0].true_operations[1].gate_type == GateOperationType.H  # type: ignore
        assert len(if_ops[0].false_operations) == 2
        assert if_ops[0].false_operations[0].gate_type == GateOperationType.H  # type: ignore
        assert if_ops[0].false_operations[1].gate_type == GateOperationType.X  # type: ignore
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

    def test_if_else_asymmetric_branch_ops(self):
        """If-else with different number of operations in each branch."""

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
        assert if_ops[0].true_operations[0].gate_type == GateOperationType.X  # type: ignore
        assert len(if_ops[0].false_operations) == 2
        assert if_ops[0].false_operations[0].gate_type == GateOperationType.H  # type: ignore
        assert if_ops[0].false_operations[1].gate_type == GateOperationType.X  # type: ignore
        assert len(if_ops[0].results) == 2
        results = if_ops[0].results
        assert results[0].type == BitType()
        assert results[1].type == QubitType()
        assert len(if_ops[0].phi_ops) == 2
        for phi in if_ops[0].phi_ops:
            assert isinstance(phi, PhiOp)
            assert len(phi.operands) == 3
            assert len(phi.results) == 1

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
        assert if_ops[0].true_operations[0].gate_type == GateOperationType.X  # type: ignore
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
        assert if_ops[0].true_operations[0].gate_type == GateOperationType.X  # type: ignore
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

    def test_nested_if_else(self):
        """Nested if-else inside a branch should build successfully."""

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
        assert len(outer_if_ops) == 1
        outer_if = outer_if_ops[0]
        assert len(outer_if.true_operations) == 2
        assert isinstance(outer_if.true_operations[0], MeasureOperation)
        assert isinstance(outer_if.true_operations[1], IfOperation)
        assert len(outer_if.false_operations) == 1
        assert outer_if.false_operations[0].gate_type == GateOperationType.H  # type: ignore
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
        assert len(inner_if_ops) == 1
        inner_if = inner_if_ops[0]
        assert len(inner_if.true_operations) == 1
        assert inner_if.true_operations[0].gate_type == GateOperationType.X  # type: ignore
        assert len(inner_if.false_operations) == 1
        assert inner_if.false_operations[0].gate_type == GateOperationType.H  # type: ignore
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
                qs[0] = qm.x(qs[0])
            else:
                qs[0] = qm.h(qs[0])
            return qs

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) == 1
        assert if_ops[0].true_operations[0].gate_type == GateOperationType.X  # type: ignore
        assert len(if_ops[0].false_operations) == 1
        assert if_ops[0].false_operations[0].gate_type == GateOperationType.H  # type: ignore
        assert len(if_ops[0].results) == 2
        results = if_ops[0].results
        assert results[0].type == BitType()
        assert results[1].type == QubitType()
        assert len(if_ops[0].phi_ops) == 2
        for phi in if_ops[0].phi_ops:
            assert isinstance(phi, PhiOp)
            assert len(phi.operands) == 3
            assert len(phi.results) == 1

    def test_if_else_symbolic_vector_different_elements(self):
        """Different elements of a parameter Vector in each branch."""

        @qkernel
        def circuit(q0: Qubit, qs: Vector[Qubit]) -> Vector[Qubit]:
            cond = qm.measure(q0)
            if cond:
                qs[0] = qm.x(qs[0])
            else:
                qs[1] = qm.h(qs[1])
            return qs

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) == 1
        assert if_ops[0].true_operations[0].gate_type == GateOperationType.X  # type: ignore
        assert len(if_ops[0].false_operations) == 1
        assert if_ops[0].false_operations[0].gate_type == GateOperationType.H  # type: ignore
        assert len(if_ops[0].results) == 2
        results = if_ops[0].results
        assert results[0].type == BitType()
        assert results[1].type == QubitType()
        assert len(if_ops[0].phi_ops) == 2
        for phi in if_ops[0].phi_ops:
            assert isinstance(phi, PhiOp)
            assert len(phi.operands) == 3
            assert len(phi.results) == 1

    def test_if_only_symbolic_vector_passthrough(self):
        """Parameter Vector with ops only in true branch, pass-through in else."""

        @qkernel
        def circuit(q0: Qubit, qs: Vector[Qubit]) -> Vector[Qubit]:
            cond = qm.measure(q0)
            if cond:
                qs[0] = qm.x(qs[0])
            return qs

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) == 1
        assert if_ops[0].true_operations[0].gate_type == GateOperationType.X  # type: ignore
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

    def test_if_else_symbolic_vector_and_qubit_mixed(self):
        """Mixed parameter Vector and individual Qubit in if-else."""

        @qkernel
        def circuit(
            q0: Qubit, q1: Qubit, qs: Vector[Qubit]
        ) -> tuple[Vector[Qubit], Qubit]:
            cond = qm.measure(q0)
            if cond:
                qs[0] = qm.x(qs[0])
                q1 = qm.h(q1)
            else:
                qs[1] = qm.h(qs[1])
                q1 = qm.x(q1)
            return qs, q1

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) == 2
        assert if_ops[0].true_operations[0].gate_type == GateOperationType.X  # type: ignore
        assert if_ops[0].true_operations[1].gate_type == GateOperationType.H  # type: ignore
        assert len(if_ops[0].false_operations) == 2
        assert if_ops[0].false_operations[0].gate_type == GateOperationType.H  # type: ignore
        assert if_ops[0].false_operations[1].gate_type == GateOperationType.X  # type: ignore
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

    def test_if_else_symbolic_vector_index_after_merge(self):
        """Indexing a parameter Vector after if-else merge must work."""

        @qkernel
        def circuit(q0: Qubit, qs: Vector[Qubit]) -> Qubit:
            cond = qm.measure(q0)
            if cond:
                qs[0] = qm.x(qs[0])
            else:
                qs[0] = qm.h(qs[0])
            result = qs[1]
            result = qm.h(result)
            return result

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) == 1
        assert if_ops[0].true_operations[0].gate_type == GateOperationType.X  # type: ignore
        assert len(if_ops[0].false_operations) == 1
        assert if_ops[0].false_operations[0].gate_type == GateOperationType.H  # type: ignore
        assert len(if_ops[0].results) == 2
        results = if_ops[0].results
        assert results[0].type == BitType()
        assert results[1].type == QubitType()
        assert len(if_ops[0].phi_ops) == 2
        for phi in if_ops[0].phi_ops:
            assert isinstance(phi, PhiOp)
            assert len(phi.operands) == 3
            assert len(phi.results) == 1


class TestIfElseWithQubitArray:
    """If-else with qm.qubit_array() — both dynamic (UInt parameter) and fixed-size."""

    # --- Dynamic qubit_array (UInt parameter) ---

    @pytest.mark.parametrize("n", [1, 3, 100])
    def test_if_else_dynamic_qubit_array_both_branches(self, n):
        """Both branches operate on same element of a dynamically-sized array."""

        @qkernel
        def circuit(q0: Qubit, n: UInt) -> Vector[Qubit]:
            qs = qm.qubit_array(n, name="qs")
            cond = qm.measure(q0)
            if cond:
                qs[0] = qm.x(qs[0])
            else:
                qs[0] = qm.h(qs[0])
            return qs

        graph = circuit.build(n=n)
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) == 1
        assert if_ops[0].true_operations[0].gate_type == GateOperationType.X  # type: ignore
        assert len(if_ops[0].false_operations) == 1
        assert if_ops[0].false_operations[0].gate_type == GateOperationType.H  # type: ignore
        assert len(if_ops[0].results) == 2
        results = if_ops[0].results
        assert results[0].type == BitType()
        assert results[1].type == QubitType()
        assert len(if_ops[0].phi_ops) == 2
        for phi in if_ops[0].phi_ops:
            assert isinstance(phi, PhiOp)
            assert len(phi.operands) == 3
            assert len(phi.results) == 1

    @pytest.mark.parametrize("n", [2, 3, 100])
    def test_if_else_dynamic_qubit_array_different_elements(self, n):
        """Different elements in each branch of a dynamically-sized array."""

        @qkernel
        def circuit(q0: Qubit, n: UInt) -> Vector[Qubit]:
            qs = qm.qubit_array(n, name="qs")
            cond = qm.measure(q0)
            if cond:
                qs[0] = qm.x(qs[0])
            else:
                qs[1] = qm.h(qs[1])
            return qs

        graph = circuit.build(n=n)
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) == 1
        assert if_ops[0].true_operations[0].gate_type == GateOperationType.X  # type: ignore
        assert len(if_ops[0].false_operations) == 1
        assert if_ops[0].false_operations[0].gate_type == GateOperationType.H  # type: ignore
        assert len(if_ops[0].results) == 2
        results = if_ops[0].results
        assert results[0].type == BitType()
        assert results[1].type == QubitType()
        assert len(if_ops[0].phi_ops) == 2
        for phi in if_ops[0].phi_ops:
            assert isinstance(phi, PhiOp)
            assert len(phi.operands) == 3
            assert len(phi.results) == 1

    @pytest.mark.parametrize("n", [1, 3, 100])
    def test_if_only_dynamic_qubit_array(self, n):
        """True branch only, no else, with dynamically-sized array."""

        @qkernel
        def circuit(q0: Qubit, n: UInt) -> Vector[Qubit]:
            qs = qm.qubit_array(n, name="qs")
            cond = qm.measure(q0)
            if cond:
                qs[0] = qm.x(qs[0])
            return qs

        graph = circuit.build(n=n)
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) == 1
        assert if_ops[0].true_operations[0].gate_type == GateOperationType.X  # type: ignore
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

    @pytest.mark.parametrize("n", [2, 3, 100])
    def test_if_else_dynamic_qubit_array_mixed_with_qubit(self, n):
        """Dynamically-sized array + individual Qubit in both branches."""

        @qkernel
        def circuit(q0: Qubit, q1: Qubit, n: UInt) -> tuple[Vector[Qubit], Qubit]:
            qs = qm.qubit_array(n, name="qs")
            cond = qm.measure(q0)
            if cond:
                qs[0] = qm.x(qs[0])
                q1 = qm.h(q1)
            else:
                qs[1] = qm.h(qs[1])
                q1 = qm.x(q1)
            return qs, q1

        graph = circuit.build(n=n)
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) == 2
        assert if_ops[0].true_operations[0].gate_type == GateOperationType.X  # type: ignore
        assert if_ops[0].true_operations[1].gate_type == GateOperationType.H  # type: ignore
        assert len(if_ops[0].false_operations) == 2
        assert if_ops[0].false_operations[0].gate_type == GateOperationType.H  # type: ignore
        assert if_ops[0].false_operations[1].gate_type == GateOperationType.X  # type: ignore
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

    # --- Fixed-size qubit_array (element operations) ---

    def test_if_else_vector_element_ops_both_branches(self):
        """Different gates on same Vector element in each branch."""

        @qkernel
        def circuit(q0: Qubit) -> Vector[Qubit]:
            qs = qm.qubit_array(3, "qs")
            cond = qm.measure(q0)
            if cond:
                qs[0] = qm.x(qs[0])
            else:
                qs[0] = qm.h(qs[0])
            return qs

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) == 1
        assert if_ops[0].true_operations[0].gate_type == GateOperationType.X  # type: ignore
        assert len(if_ops[0].false_operations) == 1
        assert if_ops[0].false_operations[0].gate_type == GateOperationType.H  # type: ignore
        assert len(if_ops[0].results) == 2
        results = if_ops[0].results
        assert results[0].type == BitType()
        assert results[1].type == QubitType()
        assert len(if_ops[0].phi_ops) == 2
        for phi in if_ops[0].phi_ops:
            assert isinstance(phi, PhiOp)
            assert len(phi.operands) == 3
            assert len(phi.results) == 1

    def test_if_else_vector_different_elements_per_branch(self):
        """Different elements operated on in each branch (tests _borrowed_indices reset)."""

        @qkernel
        def circuit(q0: Qubit) -> Vector[Qubit]:
            qs = qm.qubit_array(3, "qs")
            cond = qm.measure(q0)
            if cond:
                qs[0] = qm.x(qs[0])
            else:
                qs[1] = qm.h(qs[1])
            return qs

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) == 1
        assert if_ops[0].true_operations[0].gate_type == GateOperationType.X  # type: ignore
        assert len(if_ops[0].false_operations) == 1
        assert if_ops[0].false_operations[0].gate_type == GateOperationType.H  # type: ignore
        assert len(if_ops[0].results) == 2
        results = if_ops[0].results
        assert results[0].type == BitType()
        assert results[1].type == QubitType()
        assert len(if_ops[0].phi_ops) == 2
        for phi in if_ops[0].phi_ops:
            assert isinstance(phi, PhiOp)
            assert len(phi.operands) == 3
            assert len(phi.results) == 1

    def test_if_else_mixed_vector_and_qubit(self):
        """Mixed Vector[Qubit] and individual Qubit in if-else."""

        @qkernel
        def circuit(q0: Qubit, q1: Qubit) -> tuple[Vector[Qubit], Qubit]:
            qs = qm.qubit_array(2, "qs")
            cond = qm.measure(q0)
            if cond:
                qs[0] = qm.x(qs[0])
                q1 = qm.h(q1)
            else:
                qs[1] = qm.h(qs[1])
                q1 = qm.x(q1)
            return qs, q1

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) == 2
        assert if_ops[0].true_operations[0].gate_type == GateOperationType.X  # type: ignore
        assert if_ops[0].true_operations[1].gate_type == GateOperationType.H  # type: ignore
        assert len(if_ops[0].false_operations) == 2
        assert if_ops[0].false_operations[0].gate_type == GateOperationType.H  # type: ignore
        assert if_ops[0].false_operations[1].gate_type == GateOperationType.X  # type: ignore
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

    def test_if_else_vector_index_after_merge(self):
        """Indexing Vector after if-else merge must work (directly tests phi merge type)."""

        @qkernel
        def circuit(q0: Qubit) -> Qubit:
            qs = qm.qubit_array(3, "qs")
            cond = qm.measure(q0)
            if cond:
                qs[0] = qm.x(qs[0])
            else:
                qs[0] = qm.h(qs[0])
            result = qs[1]
            result = qm.h(result)
            return result

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) == 1
        assert if_ops[0].true_operations[0].gate_type == GateOperationType.X  # type: ignore
        assert len(if_ops[0].false_operations) == 1
        assert if_ops[0].false_operations[0].gate_type == GateOperationType.H  # type: ignore
        assert len(if_ops[0].results) == 2
        results = if_ops[0].results
        assert results[0].type == BitType()
        assert results[1].type == QubitType()
        assert len(if_ops[0].phi_ops) == 2
        for phi in if_ops[0].phi_ops:
            assert isinstance(phi, PhiOp)
            assert len(phi.operands) == 3
            assert len(phi.results) == 1

    def test_if_only_vector_passthrough(self):
        """Vector with ops in true branch only, pass-through in else."""

        @qkernel
        def circuit(q0: Qubit) -> Vector[Qubit]:
            qs = qm.qubit_array(2, "qs")
            cond = qm.measure(q0)
            if cond:
                qs[0] = qm.x(qs[0])
            return qs

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) == 1
        assert if_ops[0].true_operations[0].gate_type == GateOperationType.X  # type: ignore
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

    # --- Vector measurement as condition ---

    def test_measure_vector_condition_single_qubit_op(self):
        """Measure entire Vector, use bits[0] as condition, operate on separate Qubit."""

        @qkernel
        def circuit(q: Qubit) -> Qubit:
            qs = qm.qubit_array(2, "qs")
            bits = qm.measure(qs)
            if bits[0]:
                q = qm.x(q)
            else:
                q = qm.h(q)
            return q

        graph = circuit.build()
        # MeasureVectorOperation should precede the IfOperation
        measure_ops = [
            op for op in graph.operations if isinstance(op, MeasureVectorOperation)
        ]
        assert len(measure_ops) == 1
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) == 1
        assert if_ops[0].true_operations[0].gate_type == GateOperationType.X  # type: ignore
        assert len(if_ops[0].false_operations) == 1
        assert if_ops[0].false_operations[0].gate_type == GateOperationType.H  # type: ignore
        assert len(if_ops[0].phi_ops) >= 2
        for phi in if_ops[0].phi_ops:
            assert isinstance(phi, PhiOp)
            assert len(phi.operands) == 3
            assert len(phi.results) == 1

    def test_measure_vector_condition_another_vector_op(self):
        """Measure one Vector, use bit as condition, operate on a second Vector."""

        @qkernel
        def circuit() -> Vector[Qubit]:
            qs = qm.qubit_array(2, "qs")
            targets = qm.qubit_array(2, "targets")
            bits = qm.measure(qs)
            if bits[0]:
                targets[0] = qm.x(targets[0])
            else:
                targets[0] = qm.h(targets[0])
            return targets

        graph = circuit.build()
        measure_ops = [
            op for op in graph.operations if isinstance(op, MeasureVectorOperation)
        ]
        assert len(measure_ops) == 1
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) == 1
        assert if_ops[0].true_operations[0].gate_type == GateOperationType.X  # type: ignore
        assert len(if_ops[0].false_operations) == 1
        assert if_ops[0].false_operations[0].gate_type == GateOperationType.H  # type: ignore
        for phi in if_ops[0].phi_ops:
            assert isinstance(phi, PhiOp)
            assert len(phi.operands) == 3
            assert len(phi.results) == 1

    # --- Whole vector operations ---

    def test_qubit_condition_all_vector_elements(self):
        """Operate on all elements of a Vector in both branches."""

        @qkernel
        def circuit(q0: Qubit) -> Vector[Qubit]:
            qs = qm.qubit_array(3, "qs")
            cond = qm.measure(q0)
            if cond:
                qs[0] = qm.x(qs[0])
                qs[1] = qm.x(qs[1])
                qs[2] = qm.x(qs[2])
            else:
                qs[0] = qm.h(qs[0])
                qs[1] = qm.h(qs[1])
                qs[2] = qm.h(qs[2])
            return qs

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) == 3
        for op in if_ops[0].true_operations:
            assert op.gate_type == GateOperationType.X  # type: ignore
        assert len(if_ops[0].false_operations) == 3
        for op in if_ops[0].false_operations:
            assert op.gate_type == GateOperationType.H  # type: ignore
        assert len(if_ops[0].results) == 2
        results = if_ops[0].results
        assert results[0].type == BitType()
        assert results[1].type == QubitType()
        assert len(if_ops[0].phi_ops) == 2
        for phi in if_ops[0].phi_ops:
            assert isinstance(phi, PhiOp)
            assert len(phi.operands) == 3
            assert len(phi.results) == 1

    def test_qubit_condition_partial_vector_elements(self):
        """Different number of element operations per branch."""

        @qkernel
        def circuit(q0: Qubit) -> Vector[Qubit]:
            qs = qm.qubit_array(3, "qs")
            cond = qm.measure(q0)
            if cond:
                qs[0] = qm.x(qs[0])
                qs[1] = qm.h(qs[1])
            else:
                qs[2] = qm.x(qs[2])
            return qs

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) == 2
        assert if_ops[0].true_operations[0].gate_type == GateOperationType.X  # type: ignore
        assert if_ops[0].true_operations[1].gate_type == GateOperationType.H  # type: ignore
        assert len(if_ops[0].false_operations) == 1
        assert if_ops[0].false_operations[0].gate_type == GateOperationType.X  # type: ignore
        assert len(if_ops[0].results) == 2
        results = if_ops[0].results
        assert results[0].type == BitType()
        assert results[1].type == QubitType()
        assert len(if_ops[0].phi_ops) == 2
        for phi in if_ops[0].phi_ops:
            assert isinstance(phi, PhiOp)
            assert len(phi.operands) == 3
            assert len(phi.results) == 1

    def test_qubit_condition_vector_and_scalar_mixed(self):
        """Operate on all vector elements AND a separate qubit in both branches."""

        @qkernel
        def circuit(q0: Qubit, q1: Qubit) -> tuple[Vector[Qubit], Qubit]:
            qs = qm.qubit_array(2, "qs")
            cond = qm.measure(q0)
            if cond:
                qs[0] = qm.x(qs[0])
                qs[1] = qm.x(qs[1])
                q1 = qm.h(q1)
            else:
                qs[0] = qm.h(qs[0])
                qs[1] = qm.h(qs[1])
                q1 = qm.x(q1)
            return qs, q1

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        assert len(if_ops[0].true_operations) == 3
        assert if_ops[0].true_operations[0].gate_type == GateOperationType.X  # type: ignore
        assert if_ops[0].true_operations[1].gate_type == GateOperationType.X  # type: ignore
        assert if_ops[0].true_operations[2].gate_type == GateOperationType.H  # type: ignore
        assert len(if_ops[0].false_operations) == 3
        assert if_ops[0].false_operations[0].gate_type == GateOperationType.H  # type: ignore
        assert if_ops[0].false_operations[1].gate_type == GateOperationType.H  # type: ignore
        assert if_ops[0].false_operations[2].gate_type == GateOperationType.X  # type: ignore
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


class TestIfElseOneSidedReturnErrors:
    """One-sided return in if-else should raise SyntaxError at AST transform time."""

    def test_if_side_return_only_raises_syntax_error(self):
        """Return only in if-branch (not in else) should raise SyntaxError."""
        with pytest.raises(SyntaxError, match="One-sided 'return'"):

            @qkernel
            def circuit(q0: Qubit, q1: Qubit) -> Qubit:
                cond = qm.measure(q0)
                if cond:
                    return qm.x(q1)
                else:
                    q1 = qm.h(q1)
                return q1

    def test_else_side_return_only_raises_syntax_error(self):
        """Return only in else-branch (not in if) should raise SyntaxError."""
        with pytest.raises(SyntaxError, match="One-sided 'return'"):

            @qkernel
            def circuit(q0: Qubit, q1: Qubit) -> Qubit:
                cond = qm.measure(q0)
                if cond:
                    q1 = qm.x(q1)
                else:
                    return qm.h(q1)
                return q1

    def test_both_branches_return_still_works(self):
        """Return in BOTH branches should still work (regression guard)."""

        @qkernel
        def circuit(q0: Qubit, q1: Qubit) -> Qubit:
            cond = qm.measure(q0)
            if cond:
                return qm.x(q1)
            else:
                return qm.h(q1)

        graph = circuit.build()
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1

    def test_neither_branch_returns_still_works(self):
        """No return in either branch should still work (regression guard)."""

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

    def test_nested_one_sided_return_raises_syntax_error(self):
        """One-sided return in a nested if should also raise SyntaxError."""
        with pytest.raises(SyntaxError, match="One-sided 'return'"):

            @qkernel
            def circuit(q0: Qubit, q1: Qubit, q2: Qubit) -> Qubit:
                cond1 = qm.measure(q0)
                if cond1:
                    cond2 = qm.measure(q2)
                    if cond2:
                        return qm.x(q1)
                    else:
                        q1 = qm.h(q1)
                else:
                    q1 = qm.h(q1)
                return q1

    def test_if_only_return_no_else_raises_syntax_error(self):
        """Return in if-branch with no else clause should raise SyntaxError."""
        with pytest.raises(SyntaxError, match="One-sided 'return'"):

            @qkernel
            def circuit(q0: Qubit, q1: Qubit) -> Qubit:
                cond = qm.measure(q0)
                if cond:
                    return qm.x(q1)
                return q1

    def test_if_elif_else_all_return_works(self):
        """if-elif-else with return in ALL branches should work."""

        @qkernel
        def circuit(q0: Qubit, q1: Qubit, q2: Qubit) -> Qubit:
            cond1 = qm.measure(q0)
            cond2 = qm.measure(q2)
            if cond1:
                return qm.x(q1)
            elif cond2:
                return qm.h(q1)
            else:
                return qm.z(q1)

        graph = circuit.build()
        # Should build without error; nested IfOperations
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) >= 1

    def test_if_elif_no_else_partial_return_raises_syntax_error(self):
        """if-elif with return but no final else should raise SyntaxError.

        AST: if -> orelse=[If(elif)] where the inner If has
        body with return but no orelse. Inner visit_If detects this.
        """
        with pytest.raises(SyntaxError, match="One-sided 'return'"):

            @qkernel
            def circuit(q0: Qubit, q1: Qubit, q2: Qubit) -> Qubit:
                cond1 = qm.measure(q0)
                cond2 = qm.measure(q2)
                if cond1:
                    return qm.x(q1)
                elif cond2:
                    return qm.h(q1)
                return q1

    def test_for_nested_return_in_if_only_raises_syntax_error(self):
        """Return inside for-loop in only one branch should raise SyntaxError."""
        with pytest.raises(SyntaxError, match="inside for/while"):

            @qkernel
            def circuit(q0: Qubit, q1: Qubit) -> Qubit:
                cond = qm.measure(q0)
                if cond:
                    for _ in qm.range(1):
                        return qm.x(q1)
                else:
                    q1 = qm.h(q1)
                return q1

    def test_while_nested_return_in_if_only_raises_syntax_error(self):
        """Return inside while-loop in only one branch should raise SyntaxError."""
        with pytest.raises(SyntaxError, match="inside for/while"):

            @qkernel
            def circuit(q0: Qubit, q1: Qubit) -> Qubit:
                cond = qm.measure(q0)
                if cond:
                    while True:
                        return qm.x(q1)
                else:
                    q1 = qm.h(q1)
                return q1

    def test_for_nested_return_in_both_branches_raises_syntax_error(self):
        """Return inside for-loop in both branches should raise SyntaxError."""
        with pytest.raises(SyntaxError, match="inside for/while"):

            @qkernel
            def circuit(q0: Qubit, q1: Qubit) -> Qubit:
                cond = qm.measure(q0)
                if cond:
                    for _ in qm.range(1):
                        return qm.x(q1)
                else:
                    for _ in qm.range(1):
                        return qm.h(q1)

    def test_while_nested_return_in_both_branches_raises_syntax_error(self):
        """Return inside while-loop in both branches should raise SyntaxError."""
        with pytest.raises(SyntaxError, match="inside for/while"):

            @qkernel
            def circuit(q0: Qubit, q1: Qubit) -> Qubit:
                cond = qm.measure(q0)
                if cond:
                    while True:
                        return qm.x(q1)
                else:
                    while True:
                        return qm.h(q1)


# ===========================================================================
# Input/output variable separation (Issues 2 & 3)
# ===========================================================================


class TestIfBranchVariableMerge:
    """Tests for input/output variable separation in visit_If."""

    def test_if_both_branch_new_local_definition_builds(self):
        """Both branches define a new local; build should not raise NameError."""

        @qkernel
        def circuit(q0: Qubit, q1: Qubit, q2: Qubit) -> qm.Bit:
            cond = qm.measure(q0)
            if cond:
                a = qm.measure(q1)
            else:
                a = qm.measure(q2)
            return a

        # Should not raise NameError
        graph = circuit.build()
        assert graph is not None
        if_ops = [op for op in graph.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        # 'a' should be in the phi_ops (new local merged from both branches)
        assert any(isinstance(p, PhiOp) for p in if_ops[0].phi_ops)

    def test_if_both_branch_new_local_definition_transpile_no_nameerror(self):
        """Both branches define a new local; transpile should not raise NameError."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit.transpiler import QiskitTranspiler

        @qkernel
        def circuit() -> qm.Bit:
            q0 = qm.qubit("q0")
            q1 = qm.qubit("q1")
            q2 = qm.qubit("q2")
            cond = qm.measure(q0)
            if cond:
                a = qm.measure(q1)
            else:
                a = qm.measure(q2)
            return a

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(circuit)
        assert exe is not None

    def test_if_store_only_reassignment_is_merged(self):
        """Store-only reassignment of existing var should be merged, not stale."""

        @qkernel
        def circuit(q0: Qubit, q1: Qubit, q2: Qubit, q3: Qubit) -> qm.Bit:
            b = qm.measure(q1)
            cond = qm.measure(q0)
            if cond:
                b = qm.measure(q2)
            else:
                b = qm.measure(q3)
            # b should be the merged phi, not the original measure
            return b

        block = circuit.block
        if_ops = [op for op in block.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        # The if should have phi_ops for the reassigned variable b
        bit_phis = [
            p
            for p in if_ops[0].phi_ops
            if isinstance(p, PhiOp) and isinstance(p.results[0].type, BitType)
        ]
        # At least one phi for 'b' (BitType) beyond the condition phi
        assert len(bit_phis) >= 1

    def test_if_only_store_only_reassignment_affects_following_if(self):
        """If-only (no else) store-only reassignment should produce a phi merge."""

        @qkernel
        def circuit(q0: Qubit, q1: Qubit, q2: Qubit) -> qm.Bit:
            b = qm.measure(q1)
            cond = qm.measure(q0)
            if cond:
                b = qm.measure(q2)
            # b should be phi-merged (true: new measure, false: original)
            return b

        block = circuit.block
        if_ops = [op for op in block.operations if isinstance(op, IfOperation)]
        assert len(if_ops) == 1
        phi_ops = if_ops[0].phi_ops
        assert len(phi_ops) > 0

    def test_if_one_sided_new_local_definition_raises_syntax_error(self):
        """New local defined in only one branch and read after should raise SyntaxError."""
        with pytest.raises(SyntaxError, match="defined in only one branch"):

            @qkernel
            def circuit(q0: Qubit, q1: Qubit) -> qm.Bit:
                cond = qm.measure(q0)
                if cond:
                    a = qm.measure(q1)
                # a is read after the if but only defined in the true branch
                return a

    def test_if_one_sided_existing_outer_scope_reassignment_is_allowed(self):
        """One-sided reassignment of an existing outer var should not be rejected."""

        @qkernel
        def circuit(q0: Qubit, q1: Qubit, q2: Qubit) -> qm.Bit:
            b = qm.measure(q1)
            cond = qm.measure(q0)
            if cond:
                b = qm.measure(q2)
            return b

        # Should not raise - b exists in outer scope
        graph = circuit.build()
        assert graph is not None

    def test_if_one_sided_new_local_not_read_after_if_is_allowed(self):
        """New local defined in only one branch but not read after should be allowed."""

        @qkernel
        def circuit(q0: Qubit, q1: Qubit) -> qm.Bit:
            cond = qm.measure(q0)
            if cond:
                _unused = qm.measure(q1)
            # _unused is not read after the if
            return cond

        # Should not raise - _unused is not used after the if
        graph = circuit.build()
        assert graph is not None


class TestIfNestedInLoop:
    """Regression: if nested inside for/while must see correct outer scope."""

    def test_if_inside_for_sees_loop_body_definitions(self):
        """Variable defined before if inside for body should be in outer_defined."""

        @qkernel
        def circuit(q0: Qubit, q1: Qubit, n: UInt) -> Qubit:
            for _i in qm.range(n):
                cond = qm.measure(q0)
                if cond:
                    q1 = qm.x(q1)
                else:
                    q1 = qm.h(q1)
            return q1

        graph = circuit.build(n=3)
        assert graph is not None

    def test_if_inside_for_with_reassignment(self):
        """Store-only reassignment inside for body + if should merge correctly."""

        @qkernel
        def circuit(q0: Qubit, q1: Qubit, n: UInt) -> qm.Bit:
            b = qm.measure(q0)
            for _i in qm.range(n):
                if b:
                    q1 = qm.x(q1)
                else:
                    q1 = qm.h(q1)
                b = qm.measure(q1)
            return b

        graph = circuit.build(n=2)
        assert graph is not None
