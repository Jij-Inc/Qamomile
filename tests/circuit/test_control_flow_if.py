"""Tests for if-else control flow in @qkernel."""

import pytest

import qamomile.circuit as qm
from qamomile.circuit.frontend.qkernel import qkernel
from qamomile.circuit.frontend.handle import Qubit
from qamomile.circuit.ir.operation.control_flow import IfOperation
from qamomile.circuit.ir.operation.arithmetic_operations import PhiOp
from qamomile.circuit.transpiler.errors import QubitConsumedError


class TestIfElseQubitConsumed:
    """Both branches operating on the same Qubit should not raise QubitConsumedError."""

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
        assert graph is not None

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
        assert graph is not None

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
        assert graph is not None


class TestIfElseIRStructure:
    """PhiOp should be recorded in IfOperation and the IR should be well-formed."""

    def test_if_operation_in_graph(self):
        """The built graph should contain an IfOperation."""

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
        assert len(if_ops) == 1, f"Expected 1 IfOperation, got {len(if_ops)}"

    def test_phi_ops_recorded_in_if_operation(self):
        """PhiOp instances should be stored in IfOperation.phi_ops.

        Previously, PhiOp was created but discarded (_phi_op was never stored).
        Now it is appended to if_operation.phi_ops.
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

        if_op = if_ops[0]
        assert hasattr(if_op, "phi_ops"), "IfOperation should have phi_ops field"
        assert len(if_op.phi_ops) > 0, "PhiOps should be recorded in IfOperation"

        # Verify each PhiOp structure
        for phi in if_op.phi_ops:
            assert isinstance(phi, PhiOp)
            assert len(phi.operands) == 3  # condition, true_value, false_value
            assert len(phi.results) == 1  # phi_output

    def test_if_operation_has_true_and_false_operations(self):
        """IfOperation should have operations recorded in both branches."""

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
        if_op = if_ops[0]

        assert len(if_op.true_operations) > 0, "True branch should have operations"
        assert len(if_op.false_operations) > 0, "False branch should have operations"
