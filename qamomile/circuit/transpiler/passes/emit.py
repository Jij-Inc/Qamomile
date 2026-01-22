"""Emit pass: Generate backend-specific code from separated program."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
    CompositeGateType,
)
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.segments import (
    ClassicalSegment,
    ExpvalSegment,
    QuantumSegment,
    SeparatedProgram,
    SegmentKind,
)
from qamomile.circuit.transpiler.executable import (
    CompiledClassicalSegment,
    CompiledExpvalSegment,
    CompiledQuantumSegment,
    ExecutableProgram,
    ParameterMetadata,
)
from qamomile.circuit.transpiler.hamiltonian_eval import evaluate_hamiltonian
from qamomile.circuit.observable import Observable

T = TypeVar("T")  # Backend circuit type
C = TypeVar("C", covariant=True)  # Circuit type for emitter


@runtime_checkable
class CompositeGateEmitter(Protocol[C]):
    """Protocol for backend-specific CompositeGate emitters.

    Each backend can implement emitters for specific composite gate types
    (QPE, QFT, IQFT, etc.) using native backend libraries.

    The emitter pattern allows:
    1. Backends to use native implementations when available (e.g., Qiskit QFT)
    2. Fallback to manual decomposition when native is unavailable
    3. Easy addition of new backends without modifying core code

    Example:
        class QiskitQFTEmitter:
            def can_emit(self, gate_type: CompositeGateType) -> bool:
                return gate_type in (CompositeGateType.QFT, CompositeGateType.IQFT)

            def emit(self, circuit, op, qubit_indices, bindings) -> bool:
                from qiskit.circuit.library import QFT
                qft = QFT(len(qubit_indices))
                circuit.compose(qft, qubit_indices, inplace=True)
                return True
    """

    def can_emit(self, gate_type: CompositeGateType) -> bool:
        """Check if this emitter can handle the given gate type.

        Args:
            gate_type: The CompositeGateType to check

        Returns:
            True if this emitter supports native emission for the gate type
        """
        ...

    def emit(
        self,
        circuit: C,
        op: CompositeGateOperation,
        qubit_indices: list[int],
        bindings: dict[str, Any],
    ) -> bool:
        """Emit the composite gate to the circuit.

        Args:
            circuit: The backend-specific circuit to emit to
            op: The CompositeGateOperation to emit
            qubit_indices: Physical qubit indices for the operation
            bindings: Parameter bindings for the operation

        Returns:
            True if emission succeeded, False to fall back to manual decomposition
        """
        ...


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

    def __init__(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ):
        """Initialize with optional parameter bindings.

        Args:
            bindings: Values to bind parameters to. If not provided,
                     parameters must be bound at execution time.
            parameters: List of parameter names to preserve as backend parameters.
        """
        self.bindings = bindings or {}
        self.parameters = set(parameters) if parameters else set()

    def run(self, input: SeparatedProgram) -> ExecutableProgram[T]:
        """Emit backend code for all segments."""
        compiled_quantum: list[CompiledQuantumSegment[T]] = []
        compiled_classical: list[CompiledClassicalSegment] = []
        compiled_expval: list[CompiledExpvalSegment] = []
        execution_order: list[tuple[str, int]] = []

        # Track the last quantum segment index for expval reference
        last_quantum_idx = -1

        for segment in input.segments:
            if segment.kind == SegmentKind.QUANTUM:
                assert isinstance(segment, QuantumSegment)
                compiled = self._compile_quantum(segment)
                compiled_quantum.append(compiled)
                last_quantum_idx = len(compiled_quantum) - 1
                execution_order.append(("quantum", last_quantum_idx))
            elif segment.kind == SegmentKind.EXPVAL:
                assert isinstance(segment, ExpvalSegment)
                compiled = self._compile_expval(
                    segment, last_quantum_idx, input, compiled_quantum
                )
                compiled_expval.append(compiled)
                execution_order.append(("expval", len(compiled_expval) - 1))
            else:
                assert isinstance(segment, ClassicalSegment)
                compiled = self._compile_classical(segment)
                compiled_classical.append(compiled)
                execution_order.append(("classical", len(compiled_classical) - 1))

        return ExecutableProgram(
            compiled_quantum=compiled_quantum,
            compiled_classical=compiled_classical,
            compiled_expval=compiled_expval,
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

        # Build parameter metadata after emission
        param_metadata = self._build_parameter_metadata()

        return CompiledQuantumSegment(
            segment=segment,
            circuit=circuit,
            qubit_map=qubit_map,
            clbit_map=clbit_map,
            parameter_metadata=param_metadata,
        )

    def _build_parameter_metadata(self) -> ParameterMetadata:
        """Build parameter metadata after emission.

        Subclasses should override this to provide parameter information
        for runtime binding.

        Returns:
            ParameterMetadata with parameter information
        """
        return ParameterMetadata()

    def _compile_classical(
        self,
        segment: ClassicalSegment,
    ) -> CompiledClassicalSegment:
        """Compile a classical segment for Python execution."""
        # Classical segments are interpreted, not compiled
        return CompiledClassicalSegment(segment=segment)

    def _compile_expval(
        self,
        segment: ExpvalSegment,
        quantum_segment_index: int,
        program: SeparatedProgram,
        compiled_quantum: list[CompiledQuantumSegment[T]],
    ) -> CompiledExpvalSegment:
        """Compile an expval segment by evaluating the Hamiltonian.

        Args:
            segment: The ExpvalSegment to compile
            quantum_segment_index: Index of the quantum segment providing the state
            program: The full separated program (for parameter access)
            compiled_quantum: List of already compiled quantum segments

        Returns:
            CompiledExpvalSegment with concrete Observable
        """
        # Build a minimal block from the Hamiltonian operations
        # We need to find all Hamiltonian operations that lead to this value
        from qamomile.circuit.ir.block import Block

        # Create a block containing the operations that define the Hamiltonian
        # For now, we'll evaluate from the segment's operations
        block = Block(
            operations=segment.operations,
            parameters=program.parameters,
        )

        # Evaluate the Hamiltonian
        concrete_h = evaluate_hamiltonian(block, self.bindings)

        if concrete_h is None:
            raise RuntimeError(
                "Failed to evaluate Hamiltonian for expval operation. "
                "Make sure all Hamiltonian operations are included."
            )

        # Create Observable from ConcreteHamiltonian
        observable = Observable(concrete_h)

        # Build qubit mapping from Pauli index to physical qubit index
        qubit_map = self._build_qubit_map(
            segment.qubits_value,
            quantum_segment_index,
            compiled_quantum,
        )

        return CompiledExpvalSegment(
            segment=segment,
            observable=observable,
            quantum_segment_index=quantum_segment_index,
            result_ref=segment.result_ref,
            qubit_map=qubit_map,
        )

    def _build_qubit_map(
        self,
        qubits_value: Value | None,
        quantum_segment_index: int,
        compiled_quantum: list[CompiledQuantumSegment[T]],
    ) -> dict[int, int]:
        """Build mapping from Pauli index to physical qubit index.

        For expval((q0, q1), H), the Pauli index i maps to qubits[i].
        This method resolves the mapping through the compiled quantum
        segment's qubit_map.

        Args:
            qubits_value: The Value representing the qubit tuple/array
            quantum_segment_index: Index of the quantum segment
            compiled_quantum: List of compiled quantum segments

        Returns:
            Dict mapping Pauli index -> physical qubit index
        """
        from qamomile.circuit.ir.value import ArrayValue

        qubit_map: dict[int, int] = {}

        if qubits_value is None or quantum_segment_index < 0:
            return qubit_map

        if quantum_segment_index >= len(compiled_quantum):
            return qubit_map

        compiled_seg = compiled_quantum[quantum_segment_index]
        uuid_to_physical = compiled_seg.qubit_map

        # Check if qubits_value is an ArrayValue with qubit_values
        # (created when tuple of qubits is passed to expval)
        if isinstance(qubits_value, ArrayValue):
            qubit_values = qubits_value.params.get("qubit_values", [])
            for i, qv in enumerate(qubit_values):
                # qv is a Value object; look up its UUID in the qubit_map
                if hasattr(qv, "uuid") and qv.uuid in uuid_to_physical:
                    qubit_map[i] = uuid_to_physical[qv.uuid]
        else:
            # Single qubit or Vector case - map index 0 to the qubit
            if hasattr(qubits_value, "uuid") and qubits_value.uuid in uuid_to_physical:
                qubit_map[0] = uuid_to_physical[qubits_value.uuid]

        return qubit_map

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
