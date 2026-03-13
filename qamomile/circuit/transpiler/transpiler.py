"""Base transpiler class for backend-specific compilation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from qamomile.circuit.frontend.qkernel import QKernel
from qamomile.circuit.frontend.decomposition import DecompositionConfig
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.transpiler.errors import QamomileCompileError
from qamomile.circuit.transpiler.passes.inline import InlinePass
from qamomile.circuit.transpiler.passes.analyze import AnalyzePass
from qamomile.circuit.transpiler.passes.constant_fold import ConstantFoldingPass
from qamomile.circuit.transpiler.passes.affine_validate import AffineValidationPass
from qamomile.circuit.transpiler.passes.separate import SeparatePass
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.circuit.transpiler.passes.substitution import (
    SubstitutionConfig,
    SubstitutionPass,
    SubstitutionRule,
)
from qamomile.circuit.transpiler.segments import SimplifiedProgram
from qamomile.circuit.transpiler.executable import ExecutableProgram, QuantumExecutor

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
        separated = transpiler.separate(analyzed)
        executable = transpiler.emit(separated, bindings={"theta": 0.5})

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
    def _create_separate_pass(self) -> SeparatePass:
        """Create the backend-specific separate pass.

        Subclasses must implement this to provide a SeparatePass
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
        block_value for efficiency.
        """
        if bindings or parameters:
            # Use build() to properly handle bindings and parameters
            # This resolves array shapes from bound data (e.g., bias.shape[0])
            graph = kernel.build(parameters=parameters, **(bindings or {}))
            return Block(
                name=graph.name,
                label_args=[],  # build() doesn't preserve label_args
                input_values=graph.input_values,
                output_values=graph.output_values,
                operations=graph.operations,
                kind=BlockKind.HIERARCHICAL,
                parameters=graph.parameters,
            )
        else:
            # Original behavior for no bindings
            block_value = kernel.block
            return Block.from_block_value(block_value, {})

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

    def inline(self, block: Block) -> Block:
        """Pass 1: Inline all CallBlockOperations."""
        return self._inline_pass.run(block)

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

    def analyze(self, block: Block) -> Block:
        """Pass 2: Validate and analyze dependencies."""
        return self._analyze_pass.run(block)

    def separate(self, block: Block) -> SimplifiedProgram:
        """Pass 3: Lower and split into quantum and classical segments.

        Validates C→Q→C pattern with single quantum segment.
        """
        separate_pass = self._create_separate_pass()
        return separate_pass.run(block)

    def emit(
        self,
        separated: SimplifiedProgram,
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
            bindings: Parameter values to bind (also resolves array shapes)
            parameters: Parameter names to preserve as backend parameters

        Returns:
            ExecutableProgram ready for execution

        Raises:
            QamomileCompileError: If compilation fails (validation, dependency errors)

        Pipeline:
            1. to_block: Convert QKernel to Block
            2. substitute: Apply substitutions (if configured)
            3. inline: Inline CallBlockOperations
            4. affine_validate: Validate affine type semantics
            5. constant_fold: Fold constant expressions
            6. analyze: Validate and analyze dependencies
            7. separate: Split into quantum/classical segments
            8. emit: Generate backend-specific code
        """
        # Pass bindings and parameters to to_block for proper shape resolution
        block = self.to_block(kernel, bindings, parameters)
        # Apply substitutions if configured
        substituted = self.substitute(block)
        affine = self.inline(substituted)
        validated = self.affine_validate(affine)
        folded = self.constant_fold(validated, bindings)
        analyzed = self.analyze(folded)
        separated = self.separate(analyzed)
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
            Only returns the first quantum segment's circuit.
            For programs with multiple quantum segments, use
            compile() and access circuits from ExecutableProgram.
        """
        executable = self.transpile(kernel, bindings)

        circuit = executable.get_first_circuit()
        if circuit is None:
            raise QamomileCompileError("No quantum operations in kernel")

        return circuit
