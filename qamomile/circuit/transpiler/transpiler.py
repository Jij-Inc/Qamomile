"""Base transpiler class for backend-specific compilation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from qamomile.circuit.frontend.qkernel import QKernel
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.transpiler.errors import QamomileCompileError
from qamomile.circuit.transpiler.passes.inline import InlinePass
from qamomile.circuit.transpiler.passes.analyze import AnalyzePass
from qamomile.circuit.transpiler.passes.separate import SeparatePass
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.circuit.transpiler.segments import SeparatedProgram
from qamomile.circuit.transpiler.executable import ExecutableProgram, QuantumExecutor

if TYPE_CHECKING:
    from qamomile.circuit.ir.block_value import BlockValue

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
        analyzed = transpiler.analyze(linear)
        separated = transpiler.separate(analyzed)
        executable = transpiler.emit(separated, bindings={"theta": 0.5})

        # Option 3: Just get the circuit (no execution)
        circuit = transpiler.to_circuit(kernel, bindings={"theta": 0.5})
    """

    # Passes (can be overridden by subclasses)
    _inline_pass: InlinePass = InlinePass()
    _analyze_pass: AnalyzePass = AnalyzePass()
    _separate_pass: SeparatePass = SeparatePass()

    @abstractmethod
    def _create_emit_pass(
        self,
        bindings: dict[str, Any] | None = None,
    ) -> EmitPass[T]:
        """Create the backend-specific emit pass."""
        pass

    @abstractmethod
    def executor(self, **kwargs: Any) -> QuantumExecutor[T]:
        """Create a quantum executor for this backend."""
        pass

    # === Conversion Methods ===

    def to_block(
        self,
        kernel: QKernel,
        parameters: dict[str, Any] | None = None,
    ) -> Block:
        """Convert a QKernel to a Block.

        This accesses the BlockValue from the kernel and wraps it
        in a Block structure.
        """
        block_value = kernel.block
        return Block.from_block_value(block_value, parameters)

    # === Pipeline Passes ===

    def inline(self, block: Block) -> Block:
        """Pass 1: Inline all CallBlockOperations."""
        return self._inline_pass.run(block)

    def analyze(self, block: Block) -> Block:
        """Pass 2: Validate and analyze dependencies."""
        return self._analyze_pass.run(block)

    def separate(self, block: Block) -> SeparatedProgram:
        """Pass 3: Split into quantum and classical segments."""
        return self._separate_pass.run(block)

    def emit(
        self,
        separated: SeparatedProgram,
        bindings: dict[str, Any] | None = None,
    ) -> ExecutableProgram[T]:
        """Pass 4: Generate backend-specific code."""
        emit_pass = self._create_emit_pass(bindings)
        return emit_pass.run(separated)

    # === Convenience Methods ===

    def compile(
        self,
        kernel: QKernel,
        bindings: dict[str, Any] | None = None,
    ) -> ExecutableProgram[T]:
        """Full compilation pipeline from QKernel to executable.

        Args:
            kernel: The QKernel to compile
            bindings: Parameter values to bind

        Returns:
            ExecutableProgram ready for execution

        Raises:
            QamomileCompileError: If compilation fails (validation, dependency errors)
        """
        block = self.to_block(kernel)
        linear = self.inline(block)
        analyzed = self.analyze(linear)
        separated = self.separate(analyzed)
        return self.emit(separated, bindings)

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
        executable = self.compile(kernel, bindings)

        circuit = executable.get_first_circuit()
        if circuit is None:
            raise QamomileCompileError("No quantum operations in kernel")

        return circuit
