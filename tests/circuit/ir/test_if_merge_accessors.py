"""Tests for the IfOperation branch-merge accessor API.

``IfOperation.add_merge`` / ``iter_merges`` / ``IfMerge`` are the single
construction / read surface for phi semantics; these tests pin the
accessor contract (slot ordering, selection, identity detection) and the
strict consistency checks that guard against hand-built or corrupted
merge storage.
"""

import pytest

from qamomile.circuit.ir.operation.arithmetic_operations import PhiOp
from qamomile.circuit.ir.operation.control_flow import IfOperation
from qamomile.circuit.ir.types.primitives import BitType, UIntType
from qamomile.circuit.ir.value import Value


def _uint(name: str) -> Value:
    """Create a plain UInt-typed Value.

    Args:
        name (str): Display name for the value.

    Returns:
        Value: Fresh UInt-typed value.
    """
    return Value(type=UIntType(), name=name)


def _if_with_merges(merge_count: int) -> tuple[IfOperation, list[tuple[Value, ...]]]:
    """Build an IfOperation with ``merge_count`` merges via add_merge.

    Args:
        merge_count (int): Number of merge slots to add.

    Returns:
        tuple[IfOperation, list[tuple[Value, ...]]]: The operation and the
            ``(true, false, result)`` value triples added, in order.
    """
    if_op = IfOperation(operands=[Value(type=BitType(), name="cond")])
    triples: list[tuple[Value, ...]] = []
    for i in range(merge_count):
        triple = (_uint(f"t{i}"), _uint(f"f{i}"), _uint(f"r{i}"))
        if_op.add_merge(*triple)
        triples.append(triple)
    return if_op, triples


class TestAddMerge:
    """Construction contract of ``IfOperation.add_merge``."""

    @pytest.mark.parametrize("merge_count", [0, 1, 3])
    def test_add_merge_keeps_phi_ops_and_results_in_lockstep(
        self, merge_count: int
    ) -> None:
        """add_merge mirrors every merge into phi_ops and results."""
        if_op, triples = _if_with_merges(merge_count)

        assert len(if_op.phi_ops) == merge_count
        assert len(if_op.results) == merge_count
        for phi, (true_v, false_v, result) in zip(if_op.phi_ops, triples, strict=True):
            assert phi.operands == [if_op.condition, true_v, false_v]
            assert phi.results == [result]

    def test_add_merge_without_condition_raises(self) -> None:
        """add_merge on a condition-less IfOperation is an internal error."""
        if_op = IfOperation()
        with pytest.raises(RuntimeError, match="condition operand"):
            if_op.add_merge(_uint("t"), _uint("f"), _uint("r"))

    def test_add_merge_type_mismatch_raises(self) -> None:
        """add_merge rejects branch / result values of differing types."""
        if_op = IfOperation(operands=[Value(type=BitType(), name="cond")])
        with pytest.raises(RuntimeError, match="matching branch and result types"):
            if_op.add_merge(_uint("t"), Value(type=BitType(), name="f"), _uint("r"))
        with pytest.raises(RuntimeError, match="matching branch and result types"):
            if_op.add_merge(_uint("t"), _uint("f"), Value(type=BitType(), name="r"))


class TestIterMerges:
    """Read contract and strict checks of ``IfOperation.iter_merges``."""

    def test_iter_merges_on_bare_if_yields_nothing(self) -> None:
        """A condition-less, merge-less IfOperation iterates cleanly."""
        assert list(IfOperation().iter_merges()) == []

    def test_dependency_graph_tolerates_condition_less_if(self) -> None:
        """The dependency graph builder skips a missing condition operand.

        A partially-constructed IfOperation (no operands, no merges) must
        not crash the explicit merge-edge pass with an IndexError.
        """
        from qamomile.circuit.transpiler.passes.analyze import (
            build_dependency_graph,
        )

        assert build_dependency_graph([IfOperation()]) == {}

    def test_iter_merges_yields_slots_in_result_order(self) -> None:
        """Each merge slot exposes its index, branch sources, and result."""
        if_op, triples = _if_with_merges(3)

        merges = list(if_op.iter_merges())

        assert [m.index for m in merges] == [0, 1, 2]
        for merge, (true_v, false_v, result) in zip(merges, triples, strict=True):
            assert merge.true_value is true_v
            assert merge.false_value is false_v
            assert merge.result is result

    @pytest.mark.parametrize("taken", [True, False])
    def test_select_returns_taken_branch_source(self, taken: bool) -> None:
        """select() picks the true source iff the condition is true."""
        if_op, [(true_v, false_v, _)] = _if_with_merges(1)

        [merge] = if_op.iter_merges()

        expected = true_v if taken else false_v
        assert merge.select(taken) is expected

    def test_is_identity_detects_shared_branch_value(self) -> None:
        """is_identity is True only when both branches carry the same UUID."""
        cond = Value(type=BitType(), name="cond")
        if_op = IfOperation(operands=[cond])
        shared = _uint("shared")
        if_op.add_merge(shared, shared, _uint("r0"))
        if_op.add_merge(_uint("t1"), _uint("f1"), _uint("r1"))

        identity_flags = [m.is_identity for m in if_op.iter_merges()]

        assert identity_flags == [True, False]

    def test_iter_merges_count_mismatch_raises(self) -> None:
        """A phi_ops/results length mismatch is reported as IR corruption."""
        if_op, _ = _if_with_merges(1)
        if_op.results.append(_uint("extra"))

        with pytest.raises(RuntimeError, match="1 phi_ops for 2 results"):
            list(if_op.iter_merges())

    def test_iter_merges_malformed_phi_raises(self) -> None:
        """A merge without condition/true/false operands is IR corruption."""
        cond = Value(type=BitType(), name="cond")
        result = _uint("r")
        if_op = IfOperation(
            operands=[cond],
            results=[result],
            phi_ops=[PhiOp(operands=[_uint("only")], results=[result])],
        )

        with pytest.raises(RuntimeError, match="expected"):
            list(if_op.iter_merges())

    def test_iter_merges_result_mismatch_raises(self) -> None:
        """A merge output that is not the positional if-result is corruption."""
        cond = Value(type=BitType(), name="cond")
        if_op = IfOperation(
            operands=[cond],
            results=[_uint("r_op")],
            phi_ops=[
                PhiOp(
                    operands=[cond, _uint("t"), _uint("f")],
                    results=[_uint("r_phi")],
                )
            ],
        )

        with pytest.raises(RuntimeError, match="does not"):
            list(if_op.iter_merges())

    def test_iter_merges_condition_mismatch_raises(self) -> None:
        """A merge carrying a different condition than the if is corruption."""
        cond = Value(type=BitType(), name="cond")
        other_cond = Value(type=BitType(), name="other_cond")
        result = _uint("r")
        if_op = IfOperation(
            operands=[cond],
            results=[result],
            phi_ops=[
                PhiOp(
                    operands=[other_cond, _uint("t"), _uint("f")],
                    results=[result],
                )
            ],
        )

        with pytest.raises(RuntimeError, match="condition"):
            list(if_op.iter_merges())

    def test_iter_merges_non_phi_entry_raises(self) -> None:
        """A non-PhiOp object stored among the merges is IR corruption."""
        cond = Value(type=BitType(), name="cond")
        result = _uint("r")
        if_op = IfOperation(
            operands=[cond],
            results=[result],
            phi_ops=[IfOperation(operands=[cond])],  # type: ignore[list-item]
        )

        with pytest.raises(RuntimeError, match="expected PhiOp"):
            list(if_op.iter_merges())

    def test_iter_merges_missing_condition_with_merges_raises(self) -> None:
        """Merges attached to a condition-less if are IR corruption."""
        cond = Value(type=BitType(), name="cond")
        result = _uint("r")
        if_op = IfOperation(
            operands=[],
            results=[result],
            phi_ops=[
                PhiOp(
                    operands=[cond, _uint("t"), _uint("f")],
                    results=[result],
                )
            ],
        )

        with pytest.raises(RuntimeError, match="condition operand is missing"):
            list(if_op.iter_merges())
