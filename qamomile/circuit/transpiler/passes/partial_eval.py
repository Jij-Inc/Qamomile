"""Partial evaluation pass for compile-time simplification."""

from __future__ import annotations

import dataclasses
from typing import Any

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.control_flow import HasNestedOps
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.transpiler.block_parameter_binding import (
    block_parameter_binding_keys,
    pair_block_parameter_operands,
)
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.passes.analyze import (
    reject_control_flow_quantum_discard,
    reject_loop_carried_classical_rebinds,
    reject_self_referential_loop_stores,
)
from qamomile.circuit.transpiler.passes.compile_time_if_lowering import (
    CompileTimeIfLoweringPass,
)
from qamomile.circuit.transpiler.passes.constant_fold import ConstantFoldingPass
from qamomile.circuit.transpiler.value_resolver import (
    ValueResolver as UnifiedValueResolver,
)


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
            QubitRebindError: If a runtime if branch or a loop body
                rebinds a quantum variable to a different quantum value —
                a fresh allocation or another existing register — in a
                shape the control-flow discard check cannot prove safe
                (see ``reject_control_flow_quantum_discard``).
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

        # SELECT case bodies are operation-owned Blocks with their own formal
        # inputs, rather than same-scope ``HasNestedOps`` lists. Apply the same
        # partial-evaluation pipeline explicitly with case-local bindings so a
        # bound/default parameter can remove compile-time ``if`` nodes before
        # controlled emission without leaking an outer same-named parameter.
        evaluated_operations = self._evaluate_select_case_blocks(input.operations)
        if evaluated_operations is not input.operations:
            input = dataclasses.replace(input, operations=evaluated_operations)

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

        # Reject branch-internal and loop-body quantum discards BEFORE
        # if-lowering, with bindings, so compile-time branches (dead or
        # taken) are classified exactly the way CompileTimeIfLoweringPass
        # will lower them and only genuine runtime
        # (measurement-derived-condition) branches are checked. Running
        # here also makes the targeted error fire at the earliest pass
        # that sees the pattern.
        reject_control_flow_quantum_discard(input.operations, self._bindings)

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

    def _evaluate_select_case_blocks(
        self,
        operations: list[Operation],
    ) -> list[Operation]:
        """Partially evaluate SELECT cases reachable in one operation list.

        Ordinary control-flow bodies share the parent value scope and are
        traversed only to find SELECT nodes. Each SELECT case is then evaluated
        as an independent Block, preserving its formal inputs and outputs.

        Args:
            operations (list[Operation]): Operations to inspect.

        Returns:
            list[Operation]: Operations with evaluated SELECT case Blocks. The
                original list is returned when no reachable SELECT changes.

        Raises:
            ValidationError: If a case has a stage unsupported by partial
                evaluation or violates a pre-fold control-flow invariant.
        """
        rewritten: list[Operation] = []
        changed = False
        for operation in operations:
            current = operation
            if isinstance(current, SelectOperation):
                case_blocks: list[Block] = []
                resolver = UnifiedValueResolver(bindings=self._bindings)
                for case_block in current.case_blocks:
                    if case_block.kind == BlockKind.TRACED:
                        case_block = dataclasses.replace(
                            case_block,
                            kind=BlockKind.HIERARCHICAL,
                        )
                    case_bindings: dict[str, Any] = {}
                    for formal, actual in pair_block_parameter_operands(
                        case_block,
                        current.param_operands,
                    ):
                        resolved = resolver.resolve(actual)
                        if resolved is None:
                            continue
                        for key in block_parameter_binding_keys(formal):
                            case_bindings[key] = resolved
                    case_blocks.append(
                        PartialEvaluationPass(case_bindings).run(case_block)
                    )
                current = dataclasses.replace(current, case_blocks=case_blocks)
            if isinstance(current, HasNestedOps):
                nested_lists = current.nested_op_lists()
                evaluated_nested = [
                    self._evaluate_select_case_blocks(nested) for nested in nested_lists
                ]
                if any(
                    evaluated is not original
                    for evaluated, original in zip(
                        evaluated_nested,
                        nested_lists,
                        strict=True,
                    )
                ):
                    current = current.rebuild_nested(evaluated_nested)
            changed = changed or current is not operation
            rewritten.append(current)
        return rewritten if changed else operations
