"""Partial evaluation pass for compile-time simplification."""

from __future__ import annotations

from typing import Any

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.passes.analyze import (
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
                ``HIERARCHICAL``, or if a classical element store inside a
                loop reads an element of the array it writes.
        """
        # HIERARCHICAL is accepted so that the self-recursion unroll loop
        # can interleave inline (which leaves one inline InvokeOperation per
        # self-ref per iteration) with partial_eval (which folds the
        # base-case `if` before the next unroll).  The inner passes leave
        # unresolved inline calls untouched, so this is safe.
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
