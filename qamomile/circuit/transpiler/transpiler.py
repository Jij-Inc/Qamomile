"""Base transpiler class for backend-specific compilation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from qamomile.circuit.frontend.qkernel import QKernel
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.transpiler.errors import QamomileCompileError
from qamomile.circuit.transpiler.passes.inline import InlinePass
from qamomile.circuit.transpiler.passes.analyze import AnalyzePass
from qamomile.circuit.transpiler.passes.constant_fold import ConstantFoldingPass
from qamomile.circuit.transpiler.passes.linear_validate import LinearValidationPass
from qamomile.circuit.transpiler.passes.separate import SeparatePass
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.circuit.transpiler.segments import SeparatedProgram
from qamomile.circuit.transpiler.executable import ExecutableProgram, QuantumExecutor

if TYPE_CHECKING:
    pass

T = TypeVar("T")  # Backend circuit type


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
        linear = transpiler.inline(block)
        folded = transpiler.constant_fold(linear, bindings={"theta": 0.5})
        analyzed = transpiler.analyze(folded)
        separated = transpiler.separate(analyzed)
        executable = transpiler.emit(separated, bindings={"theta": 0.5})

        # Option 3: Just get the circuit (no execution)
        circuit = transpiler.to_circuit(kernel, bindings={"theta": 0.5})
    """

    # Generic passes (can be overridden by subclasses)
    _inline_pass: InlinePass = InlinePass()
    _linear_validate_pass: LinearValidationPass = LinearValidationPass()
    _analyze_pass: AnalyzePass = AnalyzePass()

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

    def inline(self, block: Block) -> Block:
        """Pass 1: Inline all CallBlockOperations."""
        return self._inline_pass.run(block)

    def linear_validate(self, block: Block) -> Block:
        """Pass 1.5: Validate linear type semantics.

        This is a safety net to catch linear type violations that may
        have bypassed frontend checks. Validates that quantum values
        are used at most once.
        """
        return self._linear_validate_pass.run(block)

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

    def separate(self, block: Block) -> SeparatedProgram:
        """Pass 3: Lower and split into quantum and classical segments."""
        separate_pass = self._create_separate_pass()
        return separate_pass.run(block)

    def emit(
        self,
        separated: SeparatedProgram,
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
        """
        # Pass bindings and parameters to to_block for proper shape resolution
        block = self.to_block(kernel, bindings, parameters)
        linear = self.inline(block)
        validated = self.linear_validate(linear)
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
