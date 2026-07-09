"""Unit tests for control_flow_visitor.py — if-merge traversal contract.

IfOperation stores branch merges as the parallel ``true_yields`` /
``false_yields`` value lists, NOT as nested operations. These tests pin
the traversal contract that replaced the old phi_ops nested-list walk
(Bug #6): the generic nested-list walk sees exactly the two branch
bodies, and the merge yields reach generic passes through
``all_input_values`` instead.
"""

from __future__ import annotations

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.control_flow import IfOperation
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.types.primitives import BitType, QubitType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.passes.control_flow_visitor import (
    ControlFlowVisitor,
    OperationTransformer,
    ValueCollector,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_value(name: str, type_cls: type = QubitType) -> Value:
    """Create a simple Value with the given name and type."""
    return Value(type=type_cls(), name=name)


def _make_if_with_merge() -> tuple[IfOperation, Value, Value, Value]:
    """Build an IfOperation with one branch body op and one merge slot.

    Returns:
        tuple[IfOperation, Value, Value, Value]: The operation and the
            merge's ``(true_value, false_value, result)`` values.
    """
    cond = _make_value("cond", BitType)
    true_value = _make_value("t", QubitType)
    false_value = _make_value("f", QubitType)
    merge_output = _make_value("q_merged", QubitType)
    body_op = QInitOperation(operands=[], results=[_make_value("q", QubitType)])
    if_op = IfOperation(
        operands=[cond],
        true_operations=[body_op],
        false_operations=[],
    )
    if_op.add_merge(true_value, false_value, merge_output)
    return if_op, true_value, false_value, merge_output


# ===========================================================================
# ControlFlowVisitor / OperationTransformer if-merge handling
# ===========================================================================


class TestControlFlowVisitorIfMerges:
    """Pin how the generic traversal helpers see IfOperation merges."""

    def test_nested_op_lists_is_exactly_the_two_branches(self) -> None:
        """nested_op_lists() exposes the two branch bodies and nothing else."""
        if_op, _, _, _ = _make_if_with_merge()

        nested = if_op.nested_op_lists()

        assert len(nested) == 2
        assert nested[0] is if_op.true_operations
        assert nested[1] is if_op.false_operations

    def test_visitor_visits_branch_bodies_but_no_merge_ops(self) -> None:
        """The generic walk sees the if and its body ops; merges are not ops."""
        visited: list[Operation] = []

        class Collector(ControlFlowVisitor):
            def visit_operation(self, op: Operation) -> None:
                visited.append(op)

        if_op, _, _, _ = _make_if_with_merge()

        Collector().visit_operations([if_op])

        assert visited == [if_op, if_op.true_operations[0]]

    def test_all_input_values_exposes_merge_yields(self) -> None:
        """all_input_values() surfaces the condition and both yield lists."""
        if_op, true_value, false_value, _ = _make_if_with_merge()

        input_uuids = {v.uuid for v in if_op.all_input_values()}

        assert if_op.condition.uuid in input_uuids
        assert true_value.uuid in input_uuids
        assert false_value.uuid in input_uuids

    def test_value_collector_reaches_yields_through_all_input_values(self) -> None:
        """ValueCollector picks up yield UUIDs without a storage special case."""
        if_op, true_value, false_value, merge_output = _make_if_with_merge()

        collector = ValueCollector()
        collector.visit_operations([if_op])

        assert true_value.uuid in collector.operand_uuids
        assert false_value.uuid in collector.operand_uuids
        assert merge_output.uuid in collector.result_uuids

    def test_transformer_preserves_merges_through_rebuild(self) -> None:
        """OperationTransformer round-trips the merge slots untouched."""

        class NoopTransformer(OperationTransformer):
            """Identity transform — just returns ops unchanged."""

            def transform_operation(self, op: Operation) -> Operation:
                return op

        if_op, true_value, false_value, merge_output = _make_if_with_merge()

        result = NoopTransformer().transform_operations([if_op])

        assert len(result) == 1
        assert isinstance(result[0], IfOperation)
        [merge] = result[0].iter_merges()
        assert merge.true_value is true_value
        assert merge.false_value is false_value
        assert merge.result is merge_output
