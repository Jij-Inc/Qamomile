"""Partial evaluation pass for compile-time simplification."""

from __future__ import annotations

from typing import Any

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.passes.analyze import (
    reject_branch_internal_quantum_discard,
    reject_loop_carried_classical_rebinds,
    reject_self_referential_loop_stores,
)
from qamomile.circuit.transpiler.passes.compile_time_if_lowering import (
    CompileTimeIfLoweringPass,
)
from qamomile.circuit.transpiler.passes.constant_fold import ConstantFoldingPass


class PartialEvaluationPass(Pass[Block, Block]):
    """Fold constants and lower compile-time control flow."""

    def __init__(self, bindings: dict[str, Any] | None = None):
        self._bindings = bindings

    @property
    def name(self) -> str:
        return "partial_eval"

    def run(self, input: Block) -> Block:
        """Fold constants and lower compile-time control flow.

        Args:
            input (Block): Block to evaluate.  ``AFFINE`` in the normal
                pipeline; ``HIERARCHICAL`` is also accepted for the
                self-recursion unroll loop (see inline comment).

        Returns:
            Block: Block with constants folded and compile-time-resolvable
                ``IfOperation``s lowered.

        Raises:
            ValidationError: If the block kind is not ``AFFINE`` /
                ``HIERARCHICAL``, or if one of the classical pre-fold
                rejection checks fires (a classical element store inside a
                loop reads an element of the array it writes, or a loop
                body rebinds a classical scalar whose pre-loop value it
                still reads).
            QubitRebindError: If a runtime if branch rebinds a quantum
                variable to a different quantum value — a fresh allocation
                or another existing register — while the pre-branch state
                is neither consumed on that path nor owned elsewhere.
        """
        # HIERARCHICAL is accepted so that the self-recursion unroll loop
        # can interleave inline (which leaves one CallBlockOperation per
        # self-ref per iteration) with partial_eval (which folds the
        # base-case `if` before the next unroll).  The inner passes
        # ignore CallBlockOperations, so this is safe.
        if input.kind not in (BlockKind.AFFINE, BlockKind.HIERARCHICAL):
            raise ValidationError(
                f"PartialEvaluationPass expects AFFINE or HIERARCHICAL "
                f"block, got {input.kind}",
            )

        # Reject self-referential in-loop classical stores BEFORE folding:
        # ConstantFoldingPass folds bound element reads to plain constants,
        # which both erases the parent-array provenance this check needs
        # and bakes the stale pre-loop value into every loop iteration.
        # Bindings let the check resolve IfOperation conditions the same
        # way CompileTimeIfLoweringPass will: a store inside a
        # compile-time-dead branch is not rejected (the branch is about
        # to be eliminated), while a compile-time-taken branch's store is
        # still caught here, pre-fold.
        reject_self_referential_loop_stores(input.operations, self._bindings)

        # Reject loop-carried classical scalar rebinds BEFORE folding for
        # the same reason: folding an all-constant accumulation like
        # ``total = total + 1`` collapses the in-loop BinOp to a constant,
        # erasing the dependency evidence while keeping the wrong result.
        reject_loop_carried_classical_rebinds(
            input.operations, self._bindings, output_values=input.output_values
        )

        # Reject branch-internal quantum discards BEFORE if-lowering, with
        # bindings, so compile-time branches (dead or taken) are classified
        # exactly the way CompileTimeIfLoweringPass will lower them and only
        # genuine runtime (measurement-derived-condition) branches are
        # checked. Running here also makes the targeted error fire at the
        # earliest pass that sees the pattern.
        reject_branch_internal_quantum_discard(input.operations, self._bindings)

        # Keep ``SliceArrayOperation`` nodes through partial_eval so
        # the downstream ``SliceBorrowCheckPass`` can use them as
        # view-declaration markers and detect direct-access-over-view
        # aliases independently of the order in which the view is
        # first referenced.  They are stripped immediately after the
        # linearity check by ``StripSliceArrayOpsPass`` so segmentation
        # still sees a classical-op-free quantum segment stream.
        folded = ConstantFoldingPass(self._bindings, strip_slice_ops=False).run(input)
        lowered = CompileTimeIfLoweringPass(self._bindings).run(folded)

        # Compile-time-if lowering can hoist a live branch's store out of
        # an IfOperation body. Run constant folding again so stores that
        # were previously nested become top-level fold candidates before
        # segmentation/emit decides whether they belong to a quantum segment.
        return ConstantFoldingPass(self._bindings, strip_slice_ops=False).run(lowered)
