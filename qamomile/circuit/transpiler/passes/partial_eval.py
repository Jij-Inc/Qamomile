"""Partial evaluation pass for compile-time simplification."""

from __future__ import annotations

from typing import Any

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.passes.analyze import (
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
        reject_self_referential_loop_stores(input.operations)

        # Keep ``SliceArrayOperation`` nodes through partial_eval so
        # the downstream ``SliceBorrowCheckPass`` can use them as
        # view-declaration markers and detect direct-access-over-view
        # aliases independently of the order in which the view is
        # first referenced.  They are stripped immediately after the
        # linearity check by ``StripSliceArrayOpsPass`` so segmentation
        # still sees a classical-op-free quantum segment stream.
        folded = ConstantFoldingPass(self._bindings, strip_slice_ops=False).run(input)
        return CompileTimeIfLoweringPass(self._bindings).run(folded)
