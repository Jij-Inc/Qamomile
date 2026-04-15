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
        if input.kind not in (BlockKind.AFFINE,):
            raise ValidationError(
                f"PartialEvaluationPass expects AFFINE block, got {input.kind}",
            )

        folded = ConstantFoldingPass(self._bindings).run(input)
        return CompileTimeIfLoweringPass(self._bindings).run(folded)
