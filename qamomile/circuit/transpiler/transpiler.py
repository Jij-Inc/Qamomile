"""Base transpiler class for backend-specific compilation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from qamomile.circuit.frontend.decomposition import DecompositionConfig
from qamomile.circuit.frontend.qkernel import QKernel
from qamomile.circuit.ir.block import Block, BlockKind
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
)
from qamomile.circuit.transpiler.passes.parameter_shape_resolution import (
    ParameterShapeResolutionPass,
)
from qamomile.circuit.transpiler.passes.partial_eval import PartialEvaluationPass
from qamomile.circuit.transpiler.passes.separate import SegmentationPass
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
        ``IfOperation`` via ``partial_eval``.  Terminates when no
        ``CallBlockOperation`` remains (success), when the call count
        stops decreasing (symbolic driver — self-calls are left in the
        IR and handled by downstream passes), or when ``MAX_UNROLL_DEPTH``
        is reached (non-terminating recursion — raises).
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
        if bindings and parameters:
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
        validated = self.affine_validate(affine)
        partially_evaluated = self.partial_eval(validated, bindings)
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
