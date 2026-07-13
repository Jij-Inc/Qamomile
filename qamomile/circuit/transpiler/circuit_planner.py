"""Target-independent planning for circuit-family compilation."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from functools import partial
from typing import Any

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.transpiler.errors import FrontendTransformError
from qamomile.circuit.transpiler.passes.affine_validate import AffineValidationPass
from qamomile.circuit.transpiler.passes.analyze import AnalyzePass
from qamomile.circuit.transpiler.passes.classical_lowering import (
    ClassicalLoweringPass,
)
from qamomile.circuit.transpiler.passes.inline import (
    InlinePass,
    count_inline_invokes,
    count_unrollable_inline_invokes,
)
from qamomile.circuit.transpiler.passes.partial_eval import PartialEvaluationPass
from qamomile.circuit.transpiler.passes.separate import SegmentationPass
from qamomile.circuit.transpiler.passes.slice_borrow_check import (
    SliceBorrowCheckPass,
)
from qamomile.circuit.transpiler.passes.strip_slice_ops import (
    StripSliceArrayOpsPass,
)
from qamomile.circuit.transpiler.passes.symbolic_shape_validation import (
    SymbolicShapeValidationPass,
)
from qamomile.circuit.transpiler.prepared import EntrypointMode, PreparedModule
from qamomile.circuit.transpiler.segments import ProgramPlan


@dataclasses.dataclass(frozen=True)
class CircuitPlanningHooks:
    """Provide overrideable stages for shared circuit planning.

    Args:
        inline (Callable[[Block], Block]): Hierarchical-call inlining stage.
        unroll_recursion (Callable[[Block, dict[str, Any] | None], Block]):
            Recursive-call expansion stage.
        affine_validate (Callable[[Block], Block]): Affine-safety validation
            stage.
        partial_eval (Callable[[Block, dict[str, Any] | None], Block]):
            Compile-time partial-evaluation stage.
        slice_borrow_check (Callable[[Block], Block]): Slice-borrow validation
            stage.
        strip_slice_ops (Callable[[Block], Block]): Declarative slice cleanup
            stage.
        analyze (Callable[[Block], Block]): Dependency-analysis stage.
        classical_lowering (Callable[[Block], Block]): Runtime-classical
            expression lowering stage.
        validate_symbolic_shapes (Callable[[Block], Block]): Structural-shape
            validation stage.
        plan (Callable[[Block], ProgramPlan]): Segmentation stage.
    """

    inline: Callable[[Block], Block]
    unroll_recursion: Callable[[Block, dict[str, Any] | None], Block]
    affine_validate: Callable[[Block], Block]
    partial_eval: Callable[[Block, dict[str, Any] | None], Block]
    slice_borrow_check: Callable[[Block], Block]
    strip_slice_ops: Callable[[Block], Block]
    analyze: Callable[[Block], Block]
    classical_lowering: Callable[[Block], Block]
    validate_symbolic_shapes: Callable[[Block], Block]
    plan: Callable[[Block], ProgramPlan]


def _bind_planning_stage(
    stage: Callable[[Block, dict[str, Any] | None], Block],
    bindings: dict[str, Any],
    block: Block,
) -> Block:
    """Invoke an overrideable two-argument stage positionally.

    Args:
        stage (Callable[[Block, dict[str, Any] | None], Block]): Planning stage
            supplied by a transpiler subclass.
        bindings (dict[str, Any]): Effective compile-time bindings.
        block (Block): Block being transformed.

    Returns:
        Block: Transformed block.

    Raises:
        Exception: Propagates failures from ``stage`` unchanged.
    """
    return stage(block, bindings)


class CircuitPlanningPipeline:
    """Lower prepared semantics into the shared circuit execution model.

    The pipeline contains no SDK-specific capability or artifact logic. It is
    shared by SDK transpilers and non-execution consumers that need verified
    :class:`CircuitProgram` input, including circuit visualization.

    Args:
        inline_pass (InlinePass | None): Callable inliner. Defaults to a new
            :class:`InlinePass`.
        affine_validation_pass (AffineValidationPass | None): Affine safety
            validator. Defaults to a new :class:`AffineValidationPass`.
        analyze_pass (AnalyzePass | None): Program-mode dependency analyzer.
            Fragment mode always uses an analyzer that permits quantum I/O.
            Defaults to a new :class:`AnalyzePass`.
        segmentation_pass (SegmentationPass | None): Circuit-family
            segmentation strategy. Defaults to a new
            :class:`SegmentationPass`.
        stage_hooks (CircuitPlanningHooks | None): Existing transpiler stage
            methods to preserve subclass overrides. Defaults to ``None``,
            which runs the pipeline's target-independent pass instances.
        max_unroll_depth (int): Maximum recursive inline/partial-evaluation
            iterations. Defaults to ``64``.
    """

    def __init__(
        self,
        *,
        inline_pass: InlinePass | None = None,
        affine_validation_pass: AffineValidationPass | None = None,
        analyze_pass: AnalyzePass | None = None,
        segmentation_pass: SegmentationPass | None = None,
        stage_hooks: CircuitPlanningHooks | None = None,
        max_unroll_depth: int = 64,
    ) -> None:
        """Initialize the circuit planning passes.

        Args:
            inline_pass (InlinePass | None): Callable inliner. Defaults to
                ``None``.
            affine_validation_pass (AffineValidationPass | None): Affine
                validator. Defaults to ``None``.
            analyze_pass (AnalyzePass | None): Program-mode analyzer. Defaults
                to ``None``.
            segmentation_pass (SegmentationPass | None): Segmentation pass.
                Defaults to ``None``.
            stage_hooks (CircuitPlanningHooks | None): Existing transpiler
                stage methods. Defaults to ``None``.
            max_unroll_depth (int): Recursive expansion limit. Defaults to
                ``64``.

        Raises:
            ValueError: If ``max_unroll_depth`` is not positive.
        """
        if max_unroll_depth <= 0:
            raise ValueError("max_unroll_depth must be positive")
        self._inline_pass = inline_pass or InlinePass()
        self._affine_validation_pass = affine_validation_pass or AffineValidationPass()
        self._program_analyze_pass = analyze_pass or AnalyzePass()
        self._segmentation_pass = segmentation_pass or SegmentationPass()
        self._stage_hooks = stage_hooks
        self._max_unroll_depth = max_unroll_depth

    def run(
        self,
        prepared: PreparedModule,
        bindings: dict[str, Any] | None = None,
    ) -> ProgramPlan:
        """Plan one prepared module for circuit-family lowering.

        Args:
            prepared (PreparedModule): Semantic module whose entrypoint is
                hierarchical, affine, or already analyzed.
            bindings (dict[str, Any] | None): Additional compile-time bindings.
                Values override same-named bindings retained by ``prepared``.
                Defaults to ``None``.

        Returns:
            ProgramPlan: Host-orchestrated circuit execution plan.

        Raises:
            FrontendTransformError: If recursive inlining cannot converge.
            QamomileCompileError: If semantic validation, partial evaluation,
                or segmentation rejects the module.
            ValueError: If the prepared entrypoint has an unsupported stage.
        """
        effective_bindings = dict(prepared.bindings)
        if bindings:
            effective_bindings.update(bindings)

        if self._stage_hooks is None:
            inline = self._inline_pass.run
            unroll_recursion = partial(
                self._unroll_recursion,
                bindings=effective_bindings,
            )
            affine_validate = self._affine_validation_pass.run
            partial_eval = PartialEvaluationPass(effective_bindings).run
            slice_borrow_check = SliceBorrowCheckPass().run
            strip_slice_ops = StripSliceArrayOpsPass().run
            analyze = self._program_analyze_pass.run
            classical_lowering = ClassicalLoweringPass().run
            validate_symbolic_shapes = SymbolicShapeValidationPass().run
            plan = self._segmentation_pass.run
        else:
            hooks = self._stage_hooks
            inline = hooks.inline
            unroll_recursion = partial(
                _bind_planning_stage,
                hooks.unroll_recursion,
                effective_bindings,
            )
            affine_validate = hooks.affine_validate
            partial_eval = partial(
                _bind_planning_stage,
                hooks.partial_eval,
                effective_bindings,
            )
            slice_borrow_check = hooks.slice_borrow_check
            strip_slice_ops = hooks.strip_slice_ops
            analyze = hooks.analyze
            classical_lowering = hooks.classical_lowering
            validate_symbolic_shapes = hooks.validate_symbolic_shapes
            plan = hooks.plan

        entrypoint = prepared.entrypoint
        if entrypoint.kind is BlockKind.HIERARCHICAL:
            affine = inline(entrypoint)
            affine = unroll_recursion(affine)
        elif entrypoint.kind is BlockKind.AFFINE:
            affine = entrypoint
        elif entrypoint.kind is BlockKind.ANALYZED:
            affine = dataclasses.replace(entrypoint, kind=BlockKind.AFFINE)
            affine = slice_borrow_check(affine)
            affine = strip_slice_ops(affine)
            analyzed = dataclasses.replace(affine, kind=BlockKind.ANALYZED)
            affine = None
        else:
            raise ValueError(
                "Circuit planning requires a HIERARCHICAL, AFFINE, or "
                f"ANALYZED block, got {entrypoint.kind}"
            )

        if affine is not None:
            affine = affine_validate(affine)
            evaluated = partial_eval(affine)
            evaluated = slice_borrow_check(evaluated)
            evaluated = strip_slice_ops(evaluated)
            if prepared.mode is EntrypointMode.CIRCUIT_FRAGMENT:
                analyze = AnalyzePass(validate_classical_io=False).run
            analyzed = analyze(evaluated)
        analyzed = classical_lowering(analyzed)
        analyzed = validate_symbolic_shapes(analyzed)
        return plan(analyzed)

    def _unroll_recursion(
        self,
        block: Block,
        bindings: dict[str, Any],
    ) -> Block:
        """Inline recursive calls to a partial-evaluation fixed point.

        Args:
            block (Block): Hierarchical or affine block to expand.
            bindings (dict[str, Any]): Compile-time bindings used to select
                recursion base cases.

        Returns:
            Block: Fully expanded affine block.

        Raises:
            FrontendTransformError: If recursion remains trapped inside an
                operation-owned body or exceeds the configured depth.
        """
        if count_inline_invokes(block.operations) == 0:
            return block

        for _ in range(self._max_unroll_depth):
            block = self._inline_pass.run(block)
            block = PartialEvaluationPass(bindings).run(block)
            if count_inline_invokes(block.operations) == 0:
                # Partial evaluation preserves the hierarchical kind. Run the
                # inliner once more to establish the downstream AFFINE stage.
                return self._inline_pass.run(block)
            if count_unrollable_inline_invokes(block.operations) == 0:
                raise FrontendTransformError(
                    "qmc.control / qmc.inverse was given a recursive "
                    "@qkernel: after inlining, an inline callable invocation still "
                    "remains inside the controlled / inverted block, and "
                    "partial_eval cannot fold its base-case `if` there "
                    "(constant folding does not descend into a "
                    "ControlledUOperation.block or an InverseBlockOperation "
                    "block). Controlling or inverting a self-recursive "
                    "qkernel is not supported. Rewrite the qkernel "
                    "non-recursively (manually unrolled to the required "
                    "depth) before passing it to qmc.control / qmc.inverse."
                )

        raise FrontendTransformError(
            f"Recursive @qkernel did not terminate after "
            f"{self._max_unroll_depth} unroll iterations.  Either the "
            f"recursion does not terminate under the provided bindings, "
            f"or the parameter driving the base-case condition was not "
            f"bound to a compile-time constant so partial_eval could "
            f"not fold the base case."
        )
