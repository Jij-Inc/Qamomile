"""Base transpiler class for backend-specific compilation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from qamomile.circuit.frontend.decomposition import DecompositionConfig
from qamomile.circuit.frontend.qkernel import QKernel
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.parameter import ParamKind
from qamomile.circuit.transpiler.errors import (
    FrontendTransformError,
    QamomileCompileError,
)
from qamomile.circuit.transpiler.executable import ExecutableProgram, QuantumExecutor
from qamomile.circuit.transpiler.passes.affine_validate import AffineValidationPass
from qamomile.circuit.transpiler.passes.analyze import AnalyzePass
from qamomile.circuit.transpiler.passes.compile_time_if_lowering import (
    CompileTimeIfLoweringPass,
)
from qamomile.circuit.transpiler.passes.constant_fold import ConstantFoldingPass
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.circuit.transpiler.passes.entrypoint_validation import (
    EntrypointValidationPass,
)
from qamomile.circuit.transpiler.passes.inline import (
    InlinePass,
    count_call_blocks,
    count_unrollable_call_blocks,
)
from qamomile.circuit.transpiler.passes.parameter_shape_resolution import (
    ParameterShapeResolutionPass,
)
from qamomile.circuit.transpiler.passes.partial_eval import PartialEvaluationPass
from qamomile.circuit.transpiler.passes.separate import SegmentationPass
from qamomile.circuit.transpiler.passes.slice_borrow_check import (
    SliceBorrowCheckPass,
)
from qamomile.circuit.transpiler.passes.substitution import (
    SubstitutionConfig,
    SubstitutionPass,
    SubstitutionRule,
)
from qamomile.circuit.transpiler.passes.symbolic_shape_validation import (
    SymbolicShapeValidationPass,
)
from qamomile.circuit.transpiler.segments import ProgramPlan

if TYPE_CHECKING:
    pass

T = TypeVar("T")  # Backend circuit type


def _validate_bindings_parameters_disjoint(
    bindings: dict[str, Any] | None,
    parameters: list[str] | None,
) -> None:
    """Reject argument names that appear in both bindings and parameters.

    A name being in both is ambiguous (placeholder value vs runtime
    symbol) and used to silently miscompile control-flow predicates
    that depended on parameter-array elements; rejecting the overlap up
    front keeps the contract unambiguous. Shared by :meth:`Transpiler.transpile`
    and :meth:`Transpiler.transpile_block`.

    Args:
        bindings (dict[str, Any] | None): Compile-time bindings, or
            None for no bindings.
        parameters (list[str] | None): Runtime parameter names, or None
            for the default parameter contract.

    Raises:
        ValueError: If any name appears in both collections.
    """
    if not bindings or not parameters:
        return
    overlap = set(parameters) & set(bindings.keys())
    if overlap:
        raise ValueError(
            f"Parameter name(s) {sorted(overlap)} appear in both "
            f"`parameters` and `bindings`. A name must be either "
            f"compile-time bound (in `bindings`) or runtime symbolic "
            f"(in `parameters`), not both. "
            f"If you want this value baked into the circuit, remove "
            f"it from `parameters`. If you want it as a runtime "
            f"parameter, remove it from `bindings`."
        )


def _restore_baked_bindings(
    block: Block,
    bindings: dict[str, Any] | None,
    parameters: list[str] | None,
) -> dict[str, Any] | None:
    """Reconstruct the compile-time bindings recorded in ``param_slots``.

    A block that was traced with ``kernel.build(**bindings)`` records
    each folded argument as a ``COMPILE_TIME_BOUND`` slot carrying its
    ``bound_value`` (or Python-signature ``default``). Most of those
    values are already baked into the IR metadata, but some are
    consumed again at emit time from the ``bindings`` dict — most
    notably ``Observable`` Hamiltonians. When the block crosses a
    process boundary via serialization, the original ``bindings`` dict
    is gone; this helper rebuilds it from the slot manifest so
    :meth:`Transpiler.transpile_block` sees the same contract
    :meth:`Transpiler.transpile` saw.

    Args:
        block (Block): The block whose ``param_slots`` carry the
            recorded contract.
        bindings (dict[str, Any] | None): Caller-supplied bindings for
            slots that are still unresolved. Must not re-bind a slot
            that was already compile-time bound at trace time.
        parameters (list[str] | None): Caller-supplied runtime
            parameter names. Must not name a compile-time-bound slot.

    Returns:
        dict[str, Any] | None: ``bindings`` merged with every baked
            slot value, or ``None`` when the result is empty.

    Raises:
        ValueError: If ``bindings`` or ``parameters`` references a slot
            that is already ``COMPILE_TIME_BOUND`` — the baked scalar
            values are already folded into the IR, so a differing
            re-bind (or a runtime re-interpretation) would silently
            diverge from the emitted circuit. Rebuild the kernel with
            new bindings instead.
    """
    merged = dict(bindings) if bindings else {}
    parameter_names = set(parameters or ())
    for slot in block.param_slots:
        if slot.kind is not ParamKind.COMPILE_TIME_BOUND:
            continue
        baked = slot.bound_value if slot.bound_value is not None else slot.default
        if slot.name in merged:
            raise ValueError(
                f"Argument {slot.name!r} was already compile-time bound when "
                f"this block was traced; its value is baked into the IR and "
                f"cannot be re-bound here. Re-build the kernel with the new "
                f"binding instead."
            )
        if slot.name in parameter_names:
            raise ValueError(
                f"Argument {slot.name!r} was compile-time bound when this "
                f"block was traced and cannot be turned into a runtime "
                f"parameter here. Re-build the kernel with "
                f"parameters=[{slot.name!r}] instead."
            )
        if baked is None:
            continue
        merged[slot.name] = baked
    return merged or None


@dataclass
class TranspilerConfig:
    """Configuration for the transpiler pipeline.

    This configuration allows customizing the compilation behavior,
    including decomposition strategies and subroutine substitutions.

    Attributes:
        decomposition: Configuration for decomposition strategies.
            Controls which strategies are used for composite gates.
        substitutions: Configuration for subroutine/gate substitutions.
            Allows replacing blocks or setting gate strategies.

    Example:
        config = TranspilerConfig(
            decomposition=DecompositionConfig(
                strategy_overrides={"qft": "approximate"},
            ),
            substitutions=SubstitutionConfig(
                rules=[
                    SubstitutionRule("my_oracle", target=optimized_oracle),
                ],
            ),
        )
        transpiler = QiskitTranspiler(config=config)
    """

    decomposition: DecompositionConfig = field(default_factory=DecompositionConfig)
    substitutions: SubstitutionConfig = field(default_factory=SubstitutionConfig)

    @classmethod
    def with_strategies(
        cls,
        strategy_overrides: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> "TranspilerConfig":
        """Create config with strategy overrides.

        Args:
            strategy_overrides: Map of gate name to strategy name
            **kwargs: Additional config options

        Returns:
            TranspilerConfig instance

        Example:
            config = TranspilerConfig.with_strategies(
                strategy_overrides={"qft": "approximate", "iqft": "approximate"}
            )
        """
        decomp = DecompositionConfig(
            strategy_overrides=strategy_overrides or {},
        )

        # Convert strategy overrides to substitution rules
        rules = []
        if strategy_overrides:
            for gate_name, strategy in strategy_overrides.items():
                rules.append(SubstitutionRule(source_name=gate_name, strategy=strategy))

        return cls(
            decomposition=decomp,
            substitutions=SubstitutionConfig(rules=rules),
            **kwargs,
        )


class Transpiler(ABC, Generic[T]):
    """Base class for backend-specific transpilers.

    Provides the full compilation pipeline from QKernel to
    executable program.

    Usage:
        transpiler = QiskitTranspiler()

        # Option 1: Full pipeline
        executable = transpiler.compile(kernel, bindings={"theta": 0.5})
        results = executable.run(transpiler.executor())

        # Option 2: Step-by-step
        block = transpiler.to_block(kernel)
        substituted = transpiler.substitute(block)
        affine = transpiler.inline(substituted)
        validated = transpiler.affine_validate(affine)
        folded = transpiler.constant_fold(validated, bindings={"theta": 0.5})
        analyzed = transpiler.analyze(folded)
        plan = transpiler.plan(analyzed)
        executable = transpiler.emit(plan, bindings={"theta": 0.5})

        # Option 3: Just get the circuit (no execution)
        circuit = transpiler.to_circuit(kernel, bindings={"theta": 0.5})

        # With configuration (strategy overrides)
        config = TranspilerConfig.with_strategies({"qft": "approximate"})
        transpiler = QiskitTranspiler(config=config)
    """

    # Generic passes (can be overridden by subclasses)
    _inline_pass: InlinePass = InlinePass()
    _affine_validate_pass: AffineValidationPass = AffineValidationPass()
    _analyze_pass: AnalyzePass = AnalyzePass()
    _config: TranspilerConfig | None = None

    @property
    def config(self) -> TranspilerConfig:
        """Get the transpiler configuration.

        Returns a default TranspilerConfig if not explicitly set.
        This property ensures backward compatibility with subclasses
        that don't call super().__init__().
        """
        if not hasattr(self, "_config") or self._config is None:
            self._config = TranspilerConfig()
        return self._config

    def set_config(self, config: TranspilerConfig) -> None:
        """Set the transpiler configuration.

        Args:
            config: Transpiler configuration to use
        """
        self._config = config

    @abstractmethod
    def _create_segmentation_pass(self) -> SegmentationPass:
        """Create the backend-specific segmentation pass.

        Subclasses must implement this to provide a SegmentationPass
        configured with the backend's capabilities.
        """
        pass

    @abstractmethod
    def _create_emit_pass(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> EmitPass[T]:
        """Create the backend-specific emit pass.

        Args:
            bindings: Parameter values to bind at compile time
            parameters: Parameter names to preserve as backend parameters
        """
        pass

    @abstractmethod
    def executor(self, **kwargs: Any) -> QuantumExecutor[T]:
        """Create a quantum executor for this backend."""
        pass

    # === Conversion Methods ===

    def to_block(
        self,
        kernel: QKernel,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> Block:
        """Convert a QKernel to a Block.

        Args:
            kernel: The QKernel to convert
            bindings: Concrete values to bind at trace time (resolves array shapes)
            parameters: Names to keep as unbound parameters

        When bindings or parameters are provided, uses kernel.build() to properly
        resolve array shapes from the bound data. Otherwise uses the cached
        hierarchical block for efficiency.
        """
        if bindings or parameters:
            # Use build() to properly handle bindings and parameters
            # This resolves array shapes from bound data (e.g., bias.shape[0])
            traced = kernel.build(parameters=parameters, **(bindings or {}))
            return replace(
                traced,
                kind=BlockKind.HIERARCHICAL,
            )
        else:
            # Original behavior for no bindings
            return kernel.block

    # === Pipeline Passes ===

    def substitute(self, block: Block) -> Block:
        """Pass 0.5: Apply substitutions (optional).

        This pass replaces CallBlockOperation targets and sets
        strategy names on CompositeGateOperations based on config.

        Args:
            block: Block to transform

        Returns:
            Block with substitutions applied
        """
        if not self.config.substitutions.rules:
            return block
        return SubstitutionPass(self.config.substitutions).run(block)

    def resolve_parameter_shapes(
        self,
        block: Block,
        bindings: dict[str, Any] | None = None,
    ) -> Block:
        """Pass 0.75: Resolve symbolic Vector parameter shape dims.

        Qamomile circuits are compile-time fixed-structure. Parameter
        ``Vector[Float]`` / ``Vector[UInt]`` inputs carry symbolic
        ``{name}_dim{i}`` shape Values so frontend code like
        ``arr.shape[0]`` returns a usable handle. This pass looks at
        ``bindings`` and, for every parameter array that has a concrete
        binding, substitutes those symbolic dims with constants so that
        downstream loop-bound resolution sees fixed lengths.

        Parameters without a concrete binding are left as-is; their
        symbolic dims are harmless as long as no compile-time structure
        decision depends on them (the library QAOA pattern).
        """
        return ParameterShapeResolutionPass(bindings).run(block)

    # Upper bound on unroll iterations for self-recursive @qkernels.
    # 64 covers Suzuki–Trotter up to order 130 (64 levels at 2-per-level).
    MAX_UNROLL_DEPTH: int = 64

    def inline(self, block: Block) -> Block:
        """Pass 1: Inline all CallBlockOperations."""
        return self._inline_pass.run(block)

    def unroll_recursion(
        self,
        block: Block,
        bindings: dict[str, Any] | None = None,
    ) -> Block:
        """Fixed-point loop of inline ↔ partial_eval for self-recursive kernels.

        Each iteration unrolls one layer of self-referential
        ``CallBlockOperation`` and then folds the base-case
        ``IfOperation`` via ``partial_eval``. Terminates when no
        ``CallBlockOperation`` remains (success), when every residual call
        is trapped inside an operation-owned block where ``partial_eval``
        cannot fold it (control / inverse of a recursive kernel — raises a
        targeted error, see below), or when ``MAX_UNROLL_DEPTH`` is reached
        (genuinely non-terminating top-level recursion — raises).

        Args:
            block (Block): The block to unroll. May be ``HIERARCHICAL``
                (still containing self-referential ``CallBlockOperation``s)
                or already ``AFFINE`` (returned unchanged).
            bindings (dict[str, Any] | None): Compile-time bindings used by
                ``partial_eval`` to fold the base-case condition. Defaults
                to None, meaning no bindings are applied.

        Returns:
            Block: The fully unrolled, ``AFFINE`` block once no
                ``CallBlockOperation`` remains. Returned unchanged when the
                input already has no calls.

        Raises:
            FrontendTransformError: If every remaining ``CallBlockOperation``
                is trapped inside a ``ControlledUOperation.block`` /
                ``InverseBlockOperation`` block (a self-recursive kernel was
                passed to ``qmc.control`` / ``qmc.inverse``), or if a
                genuinely non-terminating top-level recursion does not
                converge within ``MAX_UNROLL_DEPTH`` iterations. The two
                cases carry distinct, cause-specific messages.
        """
        if count_call_blocks(block.operations) == 0:
            return block

        for _ in range(self.MAX_UNROLL_DEPTH):
            block = self.inline(block)
            block = self.partial_eval(block, bindings)
            if count_call_blocks(block.operations) == 0:
                # ``partial_eval`` keeps ``block.kind`` from the input,
                # which stays HIERARCHICAL even after the last
                # CallBlockOperation was folded away.  Re-run ``inline``
                # to refresh the kind to AFFINE so downstream
                # ``affine_validate`` is happy.
                return self.inline(block)
            # After a full inline + partial_eval iteration, if calls remain
            # only inside operation-owned blocks (a ControlledUOperation's
            # ``block`` or an InverseBlockOperation's nested blocks), no
            # further iteration can make progress: ``inline`` already
            # unrolled one layer there, but ``partial_eval`` never descends
            # into those blocks to fold the base-case ``if``. This is the
            # signature of a self-recursive @qkernel passed to
            # ``qmc.control`` / ``qmc.inverse``; fail fast with a targeted
            # message instead of spinning to ``MAX_UNROLL_DEPTH`` and
            # blaming the bindings.
            if count_unrollable_call_blocks(block.operations) == 0:
                raise FrontendTransformError(
                    "qmc.control / qmc.inverse was given a recursive "
                    "@qkernel: after inlining, a CallBlockOperation still "
                    "remains inside the controlled / inverted block, and "
                    "partial_eval cannot fold its base-case `if` there "
                    "(constant folding does not descend into a "
                    "ControlledUOperation.block or an InverseBlockOperation "
                    "block). Controlling or inverting a self-recursive "
                    "kernel is not supported. Rewrite the kernel "
                    "non-recursively (manually unrolled to the required "
                    "depth) before passing it to qmc.control / qmc.inverse."
                )

        raise FrontendTransformError(
            f"Recursive @qkernel did not terminate after "
            f"{self.MAX_UNROLL_DEPTH} unroll iterations.  Either the "
            f"recursion does not terminate under the provided bindings, "
            f"or the parameter driving the base-case condition was not "
            f"bound to a compile-time constant so partial_eval could "
            f"not fold the base case."
        )

    def affine_validate(self, block: Block) -> Block:
        """Pass 1.5: Validate affine type semantics.

        This is a safety net to catch affine type violations that may
        have bypassed frontend checks. Validates that quantum values
        are used at most once.
        """
        return self._affine_validate_pass.run(block)

    def constant_fold(
        self,
        block: Block,
        bindings: dict[str, Any] | None = None,
    ) -> Block:
        """Pass 1.5: Fold constant expressions.

        Evaluates BinOp operations when all operands are constants
        or bound parameters. This prevents quantum segment splitting
        from parametric expressions like `phase * 2`.
        """
        return ConstantFoldingPass(bindings).run(block)

    def partial_eval(
        self,
        block: Block,
        bindings: dict[str, Any] | None = None,
    ) -> Block:
        """Pass 1.75: Fold constants and lower compile-time control flow."""
        return PartialEvaluationPass(bindings).run(block)

    def lower_compile_time_ifs(
        self,
        block: Block,
        bindings: dict[str, Any] | None = None,
    ) -> Block:
        """Pass 1.75: Lower compile-time resolvable IfOperations.

        Evaluates IfOperation conditions (including expression-derived
        conditions via CompOp/CondOp/NotOp) and replaces resolved ones
        with selected-branch operations.  Phi outputs are substituted
        with selected-branch values throughout the block.

        This prevents SegmentationPass from seeing classical-only compile-time
        IfOperations that would otherwise split quantum segments.
        """
        return CompileTimeIfLoweringPass(bindings).run(block)

    def strip_slice_ops(self, block: Block) -> Block:
        """Pass 1.95: Remove ``SliceArrayOperation`` nodes from the block.

        ``PartialEvaluationPass`` keeps these declarative ops through
        constant folding so :meth:`slice_borrow_check` can use them
        as view-declaration markers. Once the linearity check has run,
        segmentation and downstream passes expect a classical-op-free
        quantum stream — this pass performs that cleanup.
        """
        from qamomile.circuit.transpiler.passes.strip_slice_ops import (
            StripSliceArrayOpsPass,
        )

        return StripSliceArrayOpsPass().run(block)

    def slice_borrow_check(self, block: Block) -> Block:
        """Pass 1.9: Post-fold slice-view linearity checker.

        Runs after :meth:`partial_eval` has resolved slice bounds to
        concrete values.  Catches the slice-view linearity violations
        that the trace-time frontend check cannot detect on its own —
        specifically, slices whose bounds were *symbolic* at trace
        time (so the frontend bulk-borrow tracker had to skip them)
        and aliasing scenarios that only become visible once those
        bounds are folded to constants:

        1. A view whose newly-concrete coverage overlaps another live
           view of the same root parent.
        2. A view whose newly-concrete coverage hits a slot that was
           consumed by a destructive view operation earlier in the
           block.
        3. A view that reaches the end of the block while still
           recorded as the owner of the parent's slots (i.e. it was
           never used or never released).

        Direct element borrows (``q[i]``) emit no IR operation, so the
        IR-level pass cannot observe them; the trace-time validation
        in :func:`func_to_block._validate_returned_arrays` covers that
        path.

        The pass is a pass-through for the IR — it only raises on
        violations and leaves the block unchanged on success.
        """
        return SliceBorrowCheckPass().run(block)

    def analyze(self, block: Block) -> Block:
        """Pass 2: Validate and analyze dependencies."""
        return self._analyze_pass.run(block)

    def classical_lowering(self, block: Block) -> Block:
        """Pass 2.25: Lower measurement-derived classical ops.

        Identifies ``CompOp`` / ``CondOp`` / ``NotOp`` / ``BinOp``
        instances whose operand dataflow traces back to a measurement and
        rewrites them to ``RuntimeClassicalExpr``. Compile-time-foldable
        and emit-time-foldable (loop-bound, parameter-bound) classical
        ops are left unchanged.

        Runs after ``analyze`` so the measurement-taint analysis has the
        full dependency graph available, and before
        ``validate_symbolic_shapes`` / ``plan`` / ``emit`` so downstream
        passes can rely on the cleaner IR (in particular: future
        segmentation work can dispatch on ``RuntimeClassicalExpr`` type
        instead of the BitType-only heuristic).
        """
        from qamomile.circuit.transpiler.passes.classical_lowering import (
            ClassicalLoweringPass,
        )

        return ClassicalLoweringPass().run(block)

    def validate_symbolic_shapes(self, block: Block) -> Block:
        """Pass 2.5: Reject unresolved parameter shape dims in loop bounds.

        Runs after ``analyze`` so dependency info is complete. Raises
        ``QamomileCompileError`` with an actionable message when a
        ``gamma_dim0``-style symbolic Value reaches a ``ForOperation``
        bound without being folded to a constant by
        ``ParameterShapeResolutionPass``.
        """
        return SymbolicShapeValidationPass().run(block)

    def plan(self, block: Block) -> ProgramPlan:
        """Pass 3: Lower and split into a program plan.

        Validates C→Q→C pattern with single quantum segment.
        """
        segmentation_pass = self._create_segmentation_pass()
        return segmentation_pass.run(block)

    def emit(
        self,
        separated: ProgramPlan,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> ExecutableProgram[T]:
        """Pass 4: Generate backend-specific code.

        Args:
            separated: The separated program to emit
            bindings: Parameter values to bind at compile time
            parameters: Parameter names to preserve as backend parameters
        """
        emit_pass = self._create_emit_pass(bindings, parameters)
        return emit_pass.run(separated)

    # === Convenience Methods ===

    def transpile(
        self,
        kernel: QKernel,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> ExecutableProgram[T]:
        """Full compilation pipeline from QKernel to executable.

        Args:
            kernel: The QKernel to compile
            bindings: Parameter values to bind (also resolves array shapes).
                Names in ``bindings`` and ``parameters`` must be disjoint —
                a name is either compile-time bound or runtime symbolic,
                never both.
            parameters: Parameter names to preserve as backend parameters

        Returns:
            ExecutableProgram ready for execution

        Raises:
            ValueError: If a name appears in both ``bindings`` and
                ``parameters``. A name being in both is ambiguous (placeholder
                value vs runtime symbol) and used to silently miscompile
                control-flow predicates that depended on parameter-array
                elements; rejecting the overlap up front keeps the contract
                unambiguous.
            QamomileCompileError: If compilation fails (validation, dependency errors)

        Note:
            ``kernel`` is treated as a top-level executable entrypoint here.
            Entry points must have classical inputs/outputs only. Quantum-I/O
            QKernels remain valid as subroutines and for ``build()``.

        Pipeline:
            1. to_block: Convert QKernel to Block
            2. substitute: Apply substitutions (if configured)
            3. resolve_parameter_shapes: Constant-fold symbolic Vector param dims
            4. inline: Inline CallBlockOperations
            5. affine_validate: Validate affine type semantics
            6. partial_eval: Fold constants and lower compile-time control flow
            7. analyze: Validate and analyze dependencies
            8. validate_symbolic_shapes: Reject unresolved parameter shape dims
            9. plan: Build ProgramPlan (segment into C->Q->C steps)
            10. emit: Generate backend-specific code
        """
        _validate_bindings_parameters_disjoint(bindings, parameters)

        entrypoint_validator = EntrypointValidationPass()

        # Pass bindings and parameters to to_block for proper shape resolution
        block = self.to_block(kernel, bindings, parameters)
        entrypoint_validator.run(block)
        # Apply substitutions if configured
        substituted = self.substitute(block)
        shape_resolved = self.resolve_parameter_shapes(substituted, bindings)
        affine = self.inline(shape_resolved)
        # Self-recursive @qkernels need iterated inline ↔ partial_eval so
        # each unroll step can have its base-case `if` folded before the
        # next unroll.  No-op when the block is already affine.
        affine = self.unroll_recursion(affine, bindings)
        return self._compile_affine_to_executable(affine, bindings, parameters)

    def _compile_affine_to_executable(
        self,
        affine: Block,
        bindings: dict[str, Any] | None,
        parameters: list[str] | None,
    ) -> ExecutableProgram[T]:
        """Run the shared AFFINE→executable tail of the pipeline.

        This is the second half of :meth:`transpile` that is identical
        for :meth:`transpile_block`: both reach an ``AFFINE`` block (one
        by tracing + inlining a ``QKernel``, the other by deserializing
        IR) and from there run the same fixed pass sequence
        (``affine_validate`` → ``partial_eval`` → slice checks →
        ``analyze`` → ``classical_lowering`` → ``validate_symbolic_shapes``
        → ``plan`` → ``emit``). Keeping it in one place means a new tail
        pass cannot be added to one entry point and forgotten on the
        other.

        Args:
            affine (Block): A block at ``BlockKind.AFFINE`` (no residual
                ``CallBlockOperation``s; shape dims already resolved by
                the caller).
            bindings (dict[str, Any] | None): Compile-time bindings, or
                None. Threaded through ``partial_eval`` and ``emit``.
            parameters (list[str] | None): Runtime parameter names, or
                None. Threaded through ``emit``.

        Returns:
            ExecutableProgram[T]: The emitted backend executable.

        Raises:
            QamomileCompileError: If any pass in the tail rejects the IR
                (affine, dependency, or shape violations).
        """
        validated = self.affine_validate(affine)
        partially_evaluated = self.partial_eval(validated, bindings)
        partially_evaluated = self.slice_borrow_check(partially_evaluated)
        partially_evaluated = self.strip_slice_ops(partially_evaluated)
        analyzed = self.analyze(partially_evaluated)
        # Lower measurement-derived classical ops to ``RuntimeClassicalExpr``
        # so emit can dispatch them through a dedicated backend hook
        # instead of fold-or-translate logic over ``CompOp``/``CondOp``/
        # ``NotOp``/``BinOp``. Non-runtime (foldable) classical ops are
        # left untouched and continue through the existing emit-time fold
        # path. Runs before symbolic-shape validation and segmentation
        # so those passes see the rewritten IR.
        analyzed = self.classical_lowering(analyzed)
        analyzed = self.validate_symbolic_shapes(analyzed)
        separated = self.plan(analyzed)
        return self.emit(separated, bindings, parameters)

    def transpile_block(
        self,
        block: Block,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> ExecutableProgram[T]:
        """Compile an IR ``Block`` into an executable, without a QKernel.

        This is the entry point for IR that arrives as data rather than
        as a Python function — most importantly a Block reconstructed
        by :func:`qamomile.circuit.ir.serialize.load_json` /
        :func:`~qamomile.circuit.ir.serialize.load_msgpack` in another
        process (e.g. a workflow runner executing a quantum node of a
        larger hybrid program). It runs the same pipeline as
        :meth:`transpile` from the AFFINE stage onward; the
        trace-stage passes (``to_block`` / ``inline`` /
        ``unroll_recursion``) are skipped because an AFFINE block has
        no ``CallBlockOperation`` left by contract.

        The block's classical parameter contract travels with it in
        ``Block.param_slots``, so the caller can bind compile-time
        values (``bindings``) and select runtime parameters
        (``parameters``) exactly as with :meth:`transpile`. Array
        bindings resolve symbolic ``Vector`` shape dims here as well,
        so a client may serialize an unbound kernel and let the
        receiving side supply the problem-sized data.

        Args:
            block (Block): The block to compile. Must be
                ``BlockKind.AFFINE`` or ``BlockKind.ANALYZED`` — the
                kinds the IR serializer accepts. ``ANALYZED`` input is
                re-analyzed (every pipeline pass is idempotent).
            bindings (dict[str, Any] | None): Compile-time parameter
                bindings keyed by kernel argument name, folded into the
                IR by ``resolve_parameter_shapes`` / ``partial_eval``.
                Must be disjoint from ``parameters``. Defaults to None.
            parameters (list[str] | None): Argument names to preserve
                as runtime parameters in the emitted backend circuit.
                Defaults to None, meaning the block's recorded
                ``param_slots`` contract decides (slots already marked
                ``RUNTIME_PARAMETER`` stay runtime).

        Returns:
            ExecutableProgram[T]: Executable wrapping the backend
                circuit and the parameter metadata needed to re-bind
                runtime parameters.

        Raises:
            ValueError: If a name appears in both ``bindings`` and
                ``parameters``, or if ``block.kind`` is not ``AFFINE``
                / ``ANALYZED``.
            EntrypointValidationError: If the block has quantum inputs
                or outputs (executable entrypoints are classical-I/O
                only).
            QamomileCompileError: If compilation fails (validation,
                dependency, or shape errors).

        Example:
            >>> from qamomile.circuit.ir.serialize import load_json
            >>> block = load_json(wire_payload)
            >>> transpiler = QiskitTranspiler()
            >>> exe = transpiler.transpile_block(block)
            >>> result = exe.sample(
            ...     transpiler.executor(),
            ...     shots=1024,
            ...     bindings={"theta": 0.5},
            ... ).result()
        """
        _validate_bindings_parameters_disjoint(bindings, parameters)

        if block.kind not in (BlockKind.AFFINE, BlockKind.ANALYZED):
            raise ValueError(
                f"transpile_block() requires BlockKind.AFFINE or "
                f"BlockKind.ANALYZED, got {block.kind.name}. Use "
                f"transpile() for QKernels and inline HIERARCHICAL "
                f"blocks first."
            )
        # Rewind ANALYZED to AFFINE so the pipeline re-runs analyze
        # itself; the passes are idempotent, and AnalyzePass insists on
        # AFFINE input.
        if block.kind is BlockKind.ANALYZED:
            block = replace(block, kind=BlockKind.AFFINE)

        # Values folded at trace time (kernel.build bindings / defaults)
        # are recorded in ``param_slots``; rebuild the bindings dict so
        # emit-time consumers (e.g. Observable Hamiltonians) see the
        # same contract transpile() saw before the block was
        # serialized.
        bindings = _restore_baked_bindings(block, bindings, parameters)

        EntrypointValidationPass().run(block)
        # ``substitute`` is intentionally skipped: SubstitutionPass
        # rewrites CallBlockOperation targets, which cannot exist in an
        # AFFINE block (they are gone by the serializer's contract), and
        # the pass itself only accepts HIERARCHICAL input. Strategy-name
        # substitution on already-inlined composite gates would need a
        # kind-relaxed pass and is future work.
        shape_resolved = self.resolve_parameter_shapes(block, bindings)
        return self._compile_affine_to_executable(shape_resolved, bindings, parameters)

    def to_circuit(
        self,
        kernel: QKernel,
        bindings: dict[str, Any] | None = None,
    ) -> T:
        """Compile and extract just the quantum circuit.

        This is a convenience method for when you just want the
        backend circuit without the full executable.

        Args:
            kernel: The QKernel to compile
            bindings: Parameter values to bind

        Returns:
            Backend-specific quantum circuit

        Note:
            ``kernel`` is treated as a top-level executable entrypoint and
            must therefore have classical inputs/outputs only. Use a
            classical-I/O wrapper kernel when composing quantum-I/O
            subroutines into an executable circuit.

        Note:
            Only returns the first quantum segment's circuit.
            For programs with multiple quantum segments, use
            compile() and access circuits from ExecutableProgram.
        """
        executable = self.transpile(kernel, bindings)

        circuit = executable.get_first_circuit()
        if circuit is None:
            raise QamomileCompileError("No quantum operations in kernel")

        return circuit
