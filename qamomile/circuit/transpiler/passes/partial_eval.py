"""Partial evaluation pass for compile-time simplification."""

from __future__ import annotations

from typing import Any

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.circuit.transpiler.passes import Pass
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

        folded = ConstantFoldingPass(self._bindings).run(input)
        return CompileTimeIfLoweringPass(self._bindings).run(folded)
