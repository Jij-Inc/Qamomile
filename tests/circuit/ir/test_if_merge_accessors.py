"""Tests for the IfOperation branch-merge accessor API.

``IfOperation.add_merge`` / ``iter_merges`` / ``IfMerge`` are the single
construction / read surface for merge semantics; these tests pin the
accessor contract (slot ordering, selection, identity detection) and the
strict consistency checks that guard against hand-built or corrupted
merge storage.
"""

import pytest

from qamomile.circuit.ir.operation.control_flow import (
    BranchRebind,
    IfOperation,
    LoopCarriedRebind,
    WhileOperation,
    genuine_input_values,
)
from qamomile.circuit.ir.types.primitives import BitType, QubitType, UIntType
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
    def test_add_merge_keeps_yields_and_results_in_lockstep(
        self, merge_count: int
    ) -> None:
        """add_merge mirrors every merge into the yield lists and results."""
        if_op, triples = _if_with_merges(merge_count)

        assert len(if_op.true_yields) == merge_count
        assert len(if_op.false_yields) == merge_count
        assert len(if_op.results) == merge_count
        for i, (true_v, false_v, result) in enumerate(triples):
            assert if_op.true_yields[i] is true_v
            assert if_op.false_yields[i] is false_v
            assert if_op.results[i] is result

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

    def test_iter_merges_result_count_mismatch_raises(self) -> None:
        """A yields/results length mismatch is reported as IR corruption."""
        if_op, _ = _if_with_merges(1)
        if_op.results.append(_uint("extra"))

        with pytest.raises(
            RuntimeError, match="1 true_yields / 1 false_yields for 2 results"
        ):
            list(if_op.iter_merges())

    def test_iter_merges_yield_list_mismatch_raises(self) -> None:
        """Yield lists that disagree with each other are IR corruption."""
        if_op, _ = _if_with_merges(1)
        if_op.true_yields.append(_uint("stray"))

        with pytest.raises(
            RuntimeError, match="2 true_yields / 1 false_yields for 1 results"
        ):
            list(if_op.iter_merges())

    def test_iter_merges_missing_condition_with_merges_raises(self) -> None:
        """Merges attached to a condition-less if are IR corruption.

        The yield lists are attached directly (bypassing ``add_merge``,
        which would already reject the missing condition) to simulate
        hand-built corrupted storage. The old storage's other per-merge
        corruption modes — a foreign entry among the merges or a merge
        carrying a different condition — cannot be represented in the
        yield-list storage and need no counterparts here.
        """
        if_op = IfOperation(operands=[])
        if_op.true_yields.append(_uint("t"))
        if_op.false_yields.append(_uint("f"))
        if_op.results.append(_uint("r"))

        with pytest.raises(RuntimeError, match="condition operand is missing"):
            list(if_op.iter_merges())


def _qubit(name: str) -> Value:
    """Create a plain Qubit-typed Value.

    Args:
        name (str): Display name for the value.

    Returns:
        Value: Fresh Qubit-typed value.
    """
    return Value(type=QubitType(), name=name)


class TestGenuineInputValues:
    """Occurrence-based exclusion of rebind-record values from genuine reads."""

    def test_loop_rebind_record_values_are_excluded(self) -> None:
        """A loop rebind record's before/after are not genuine reads.

        They ride along ``all_input_values`` only for cloning, so a value
        referenced solely through the record must not count as read.
        """
        cond = Value(type=BitType(), name="cond")
        before = _uint("before")
        after = _uint("after")
        while_op = WhileOperation(
            operands=[cond],
            operations=[],
            loop_carried_rebinds=(
                LoopCarriedRebind(var_name="acc", before=before, after=after),
            ),
        )

        read_uuids = {v.uuid for v in genuine_input_values(while_op)}

        assert cond.uuid in read_uuids, "the while condition is a genuine read"
        assert before.uuid not in read_uuids, "rebind-record before is not a read"
        assert after.uuid not in read_uuids, "rebind-record after is not a read"

    def test_yield_shared_with_branch_rebind_is_kept(self) -> None:
        """A value that is both a false yield and a branch-rebind before stays read.

        This is the else-less quantum-discard shape: the false side yields
        the pre-branch value, which is simultaneously the ``branch_rebinds``
        before. A plain UUID-set subtraction would drop the yield read too;
        the occurrence-based removal must keep it.
        """
        cond = Value(type=BitType(), name="cond")
        q_pre = _qubit("q_pre")
        fresh = _qubit("fresh")
        merged = _qubit("merged")
        if_op = IfOperation(operands=[cond], true_operations=[], false_operations=[])
        if_op.add_merge(fresh, q_pre, merged)
        if_op.branch_rebinds = (
            BranchRebind(
                var_name="q",
                before=q_pre,
                rebound_in_true=True,
                rebound_in_false=False,
            ),
        )

        read_uuids = {v.uuid for v in genuine_input_values(if_op)}

        assert q_pre.uuid in read_uuids, "kept via the false yield despite the record"
        assert fresh.uuid in read_uuids, "the true yield is a genuine read"
