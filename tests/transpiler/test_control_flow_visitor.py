"""Unit tests for control_flow_visitor.py — phi_ops traversal.

Covers fixes for Bug #6 (IfOperation phi_ops not visited/transformed):
    ControlFlowVisitor.visit_operations() now recurses into
    IfOperation.phi_ops so that visitors see PhiOp nodes.
    OperationTransformer.transform_operations() now transforms
    IfOperation.phi_ops and writes them back via dataclasses.replace().
"""

from __future__ import annotations

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import PhiOp
from qamomile.circuit.ir.operation.control_flow import IfOperation
from qamomile.circuit.ir.types.primitives import BitType, QubitType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.passes.control_flow_visitor import (
    ControlFlowVisitor,
    OperationTransformer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_value(name: str, type_cls: type = QubitType) -> Value:
    """Create a simple Value with the given name and type."""
    return Value(type=type_cls(), name=name)


# ===========================================================================
# ControlFlowVisitor / OperationTransformer phi_ops handling
# ===========================================================================


class TestControlFlowVisitorPhiOps:
    """Tests that ControlFlowVisitor base classes visit/transform phi_ops."""

    def test_visitor_visits_phi_ops(self) -> None:
        """ControlFlowVisitor.visit_operations visits phi_ops."""
        visited: list[Operation] = []

        class Collector(ControlFlowVisitor):
            def visit_operation(self, op: Operation) -> None:
                visited.append(op)

        cond = _make_value("cond", BitType)
        phi_output = _make_value("q_phi", QubitType)
        phi = PhiOp(
            operands=[cond, _make_value("t", QubitType), _make_value("f", QubitType)],
            results=[phi_output],
        )
        if_op = IfOperation(
            operands=[cond],
            results=[phi_output],
            true_operations=[],
            false_operations=[],
            phi_ops=[phi],
        )

        collector = Collector()
        collector.visit_operations([if_op])

        # Should visit: if_op itself + the phi op inside
        assert if_op in visited
        assert phi in visited

    def test_transformer_transforms_phi_ops(self) -> None:
        """OperationTransformer.transform_operations transforms phi_ops."""

        class NoopTransformer(OperationTransformer):
            """Identity transform — just returns ops unchanged."""

            def transform_operation(self, op: Operation) -> Operation:
                return op

        cond = _make_value("cond", BitType)
        phi_output = _make_value("q_phi", QubitType)
        phi = PhiOp(
            operands=[cond, _make_value("t", QubitType), _make_value("f", QubitType)],
            results=[phi_output],
        )
        if_op = IfOperation(
            operands=[cond],
            results=[phi_output],
            true_operations=[],
            false_operations=[],
            phi_ops=[phi],
        )

        transformer = NoopTransformer()
        result = transformer.transform_operations([if_op])

        assert len(result) == 1
        assert isinstance(result[0], IfOperation)
        # phi_ops should be preserved through transform
        assert len(result[0].phi_ops) == 1
        assert isinstance(result[0].phi_ops[0], PhiOp)
