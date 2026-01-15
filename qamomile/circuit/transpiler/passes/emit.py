"""Emit pass: Generate backend-specific code from separated program."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.segments import (
    ClassicalSegment,
    QuantumSegment,
    SeparatedProgram,
    SegmentKind,
)
from qamomile.circuit.transpiler.executable import (
    CompiledClassicalSegment,
    CompiledQuantumSegment,
    ExecutableProgram,
)

T = TypeVar("T")  # Backend circuit type


class EmitPass(Pass[SeparatedProgram, ExecutableProgram[T]], Generic[T]):
    """Base class for backend-specific emission passes.

    Subclasses implement _emit_quantum_segment() to generate
    backend-specific quantum circuits.

    Input: SeparatedProgram
    Output: ExecutableProgram with compiled segments
    """

    @property
    def name(self) -> str:
        return "emit"

    def __init__(self, bindings: dict[str, Any] | None = None):
        """Initialize with optional parameter bindings.

        Args:
            bindings: Values to bind parameters to. If not provided,
                     parameters must be bound at execution time.
        """
        self.bindings = bindings or {}

    def run(self, input: SeparatedProgram) -> ExecutableProgram[T]:
        """Emit backend code for all segments."""
        compiled_quantum: list[CompiledQuantumSegment[T]] = []
        compiled_classical: list[CompiledClassicalSegment] = []
        execution_order: list[tuple[bool, int]] = []

        for segment in input.segments:
            if segment.kind == SegmentKind.QUANTUM:
                assert isinstance(segment, QuantumSegment)
                compiled = self._compile_quantum(segment)
                compiled_quantum.append(compiled)
                execution_order.append((True, len(compiled_quantum) - 1))
            else:
                assert isinstance(segment, ClassicalSegment)
                compiled = self._compile_classical(segment)
                compiled_classical.append(compiled)
                execution_order.append((False, len(compiled_classical) - 1))

        return ExecutableProgram(
            compiled_quantum=compiled_quantum,
            compiled_classical=compiled_classical,
            execution_order=execution_order,
            output_refs=input.output_refs,
        )

    def _compile_quantum(
        self,
        segment: QuantumSegment,
    ) -> CompiledQuantumSegment[T]:
        """Compile a quantum segment to backend circuit."""
        circuit, qubit_map, clbit_map = self._emit_quantum_segment(
            segment.operations,
            self.bindings,
        )

        return CompiledQuantumSegment(
            segment=segment,
            circuit=circuit,
            qubit_map=qubit_map,
            clbit_map=clbit_map,
        )

    def _compile_classical(
        self,
        segment: ClassicalSegment,
    ) -> CompiledClassicalSegment:
        """Compile a classical segment for Python execution."""
        # Classical segments are interpreted, not compiled
        return CompiledClassicalSegment(segment=segment)

    @abstractmethod
    def _emit_quantum_segment(
        self,
        operations: list[Operation],
        bindings: dict[str, Any],
    ) -> tuple[T, dict[str, int], dict[str, int]]:
        """Generate backend-specific quantum circuit.

        Args:
            operations: List of quantum operations to emit
            bindings: Parameter bindings

        Returns:
            Tuple of (circuit, qubit_map, clbit_map) where:
            - circuit: Backend-specific circuit object
            - qubit_map: Value UUID -> physical qubit index
            - clbit_map: Value UUID -> physical clbit index
        """
        pass
