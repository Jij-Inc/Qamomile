"""Emit pass: Generate backend-specific code from separated program."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import qamomile.observable as qm_o
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
    CompositeGateType,
)
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.executable import (
    CompiledClassicalSegment,
    CompiledExpvalSegment,
    CompiledQuantumSegment,
    ExecutableProgram,
    ParameterMetadata,
)
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.passes.emit_support.qubit_address import (
    ClbitMap,
    QubitAddress,
    QubitMap,
)
from qamomile.circuit.transpiler.passes.emit_support.value_resolver import (
    ValueResolver,
)
from qamomile.circuit.transpiler.segments import (
    ClassicalSegment,
    ClassicalStep,
    ExpvalSegment,
    ExpvalStep,
    ProgramPlan,
    QuantumSegment,
    QuantumStep,
)

T = TypeVar("T")  # Backend circuit type
C = TypeVar("C", contravariant=True)  # Circuit type for emitter


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
                from qiskit.circuit.library import QFTGate
                qft_gate = QFTGate(len(qubit_indices))
                circuit.append(qft_gate, qubit_indices)
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


class EmitPass(Pass[ProgramPlan, ExecutableProgram[T]], Generic[T]):
    """Base class for backend-specific emission passes.

    Subclasses implement _emit_quantum_segment() to generate
    backend-specific quantum circuits.

    Input: ProgramPlan
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
        # Wrap user bindings in an ``EmitContext`` so emit-time writers can
        # progressively migrate to typed methods (``set_value``, ``set_runtime_expr``,
        # ``push_loop_var``) while existing dict-style writes continue to
        # work. ``EmitContext`` IS a ``dict``, so downstream signatures
        # typed as ``dict[str, Any]`` accept it unchanged.
        from qamomile.circuit.transpiler.emit_context import EmitContext

        if isinstance(bindings, EmitContext):
            self.bindings = bindings
        else:
            self.bindings = EmitContext.from_user_bindings(bindings)
        self.parameters = set(parameters) if parameters else set()
        self._resolver = ValueResolver(self.parameters)

    def run(self, input: ProgramPlan) -> ExecutableProgram[T]:
        """Emit backend code from a program plan."""
        quantum_segments: list[QuantumSegment] = []
        compiled_classical: list[CompiledClassicalSegment] = []
        classical_segments: list[ClassicalSegment] = []
        compiled_expval: list[CompiledExpvalSegment] = []
        expval_segments: list[ExpvalSegment] = []
        compiled_quantum: list[CompiledQuantumSegment[T]] = []

        for step in input.steps:
            if isinstance(step, ClassicalStep):
                classical_segments.append(step.segment)
                compiled_classical.append(self._compile_classical(step.segment))
            elif isinstance(step, QuantumStep):
                quantum_segments.append(step.segment)
                compiled_quantum.append(self._compile_quantum(step.segment))
            elif isinstance(step, ExpvalStep):
                expval_segments.append(step.segment)
                compiled_expval.append(
                    self._compile_expval(
                        step.segment,
                        step.quantum_step_index,
                        input,
                        compiled_quantum,
                    )
                )

        return ExecutableProgram(
            plan=input,
            compiled_quantum=compiled_quantum,
            compiled_classical=compiled_classical,
            compiled_expval=compiled_expval,
            output_refs=input.abi.output_refs,
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

        # Retrieve measurement qubit map if available (StandardEmitPass sets it)
        measurement_qubit_map: dict[int, int] = getattr(
            self, "_measurement_qubit_map", {}
        )

        return CompiledQuantumSegment(
            segment=segment,
            circuit=circuit,
            qubit_map=qubit_map,
            clbit_map=clbit_map,
            measurement_qubit_map=dict(measurement_qubit_map),
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
        program: ProgramPlan,
        compiled_quantum: list[CompiledQuantumSegment[T]],
    ) -> CompiledExpvalSegment:
        """Compile expval segment by retrieving bound Hamiltonian.

        Args:
            segment: The ExpvalSegment to compile
            quantum_segment_index: Index of the quantum segment providing the state
            program: The full simplified program (for parameter access)
            compiled_quantum: List of already compiled quantum segments

        Returns:
            CompiledExpvalSegment with qm_o.Hamiltonian
        """
        observable_value = segment.hamiltonian_value

        # Retrieve Hamiltonian from bindings
        if observable_value is None:
            raise RuntimeError("ExpvalSegment has no hamiltonian_value set.")

        hamiltonian = self._resolve_observable_binding(observable_value)

        # Validate type
        if not isinstance(hamiltonian, qm_o.Hamiltonian):
            raise TypeError(
                f"Expected qamomile.observable.Hamiltonian, got {type(hamiltonian)}"
            )

        # Snapshot the binding.  Without this, a user who calls
        # ``H.add_term(...)`` between ``transpile()`` and ``run()``
        # would silently change the observable that the compiled
        # executable evaluates — a mutable-default-argument-style
        # trap.  The ``executable`` should reflect the binding at
        # the moment of compile.
        hamiltonian = hamiltonian.copy()

        # Build qubit mapping from Pauli index to physical qubit index.
        # The ``ProgramOrchestrator`` applies ``hamiltonian.remap_qubits``
        # with this map once at run time; we intentionally do NOT pre-
        # remap here to avoid the prior double-remap, which silently
        # produced wrong observables when the map's keys collided with
        # its remapped values (e.g. view qubit_map {0:1, 1:3} mapped
        # ``Z(0)`` to ``Z(3)`` instead of ``Z(1)``).
        qubit_map = self._build_qubit_map(
            segment.qubits_value,
            quantum_segment_index,
            compiled_quantum,
        )

        # Create CompiledExpvalSegment with qm_o.Hamiltonian directly
        return CompiledExpvalSegment(
            segment=segment,
            hamiltonian=hamiltonian,
            quantum_segment_index=quantum_segment_index,
            result_ref=segment.result_ref,
            qubit_map=qubit_map,
        )

    def _resolve_observable_binding(self, observable_value: Value) -> Any:
        """Resolve an observable Value to its bound Hamiltonian."""
        hamiltonian = self._resolver.resolve_bound_value(
            observable_value, self.bindings
        )
        if hamiltonian is None:
            raise RuntimeError(
                f"Observable '{observable_value.name}' not found in bindings. "
                f"Hamiltonians must be provided as bindings."
            )
        return hamiltonian

    def _build_qubit_map(
        self,
        qubits_value: Value | None,
        quantum_segment_index: int,
        compiled_quantum: list[CompiledQuantumSegment[T]],
    ) -> dict[int, int]:
        """Build mapping from Pauli index to physical qubit index.

        For ``expval((q0, q1), H)``, the Pauli index i maps to
        ``qubits[i]``. When ``qubits_value`` is a sliced view
        (``expval(q[1::2], Z(0))``), the view's ``slice_of`` chain is
        walked so Pauli index i resolves to the root parent's
        ``start + step * i``-th physical qubit. The caller then feeds
        this mapping into ``hamiltonian.remap_qubits(...)`` which
        rewrites observable qubit indices to physical-space indices in
        one go.

        Args:
            qubits_value: The Value representing the qubit tuple /
                array / view to evaluate the observable over.
            quantum_segment_index: Index of the quantum segment whose
                compiled ``qubit_map`` supplies the physical qubit
                positions.
            compiled_quantum: List of already compiled quantum segments.

        Returns:
            Dict mapping Pauli index -> physical qubit index. Empty
            when the segment index is invalid or when no entry
            resolved.
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
            # For a sliced view, the view's own element uuids are not
            # keys in uuid_to_physical (only the root parent's are), so
            # we walk the slice_of chain once and build root-space
            # addresses. For a root array this simply returns
            # (qubits_value, 0, 1) and behaves identically to the
            # pre-view lookup path.
            if qubits_value.slice_of is None:
                # Non-sliced array: ``resolve_slice_chain`` will not
                # walk and cannot raise on bound resolution.  Wrap
                # defensively against unexpected errors but keep the
                # legacy fallback semantics — never observed in
                # practice, just paranoia for backwards compat.
                try:
                    root_av, start, step = self._resolver.resolve_slice_chain(
                        qubits_value, self.bindings, operation="ExpvalSegment"
                    )
                except Exception:
                    root_av, start, step = qubits_value, 0, 1
            else:
                # Sliced view: let ``EmitError`` propagate.  Catching
                # here used to silently downgrade the lookup to the
                # element_uuid path, which then produced an empty
                # qubit_map and a backend-side observable / circuit
                # width mismatch far away from the real cause.  When
                # slice bounds genuinely cannot be resolved under the
                # active bindings the user needs the EmitError that
                # ``resolve_slice_chain`` already raises.
                root_av, start, step = self._resolver.resolve_slice_chain(
                    qubits_value, self.bindings, operation="ExpvalSegment"
                )

            if root_av is qubits_value:
                # Non-view case: match legacy direct element_uuid lookup
                # so arrays whose elements were registered under explicit
                # UUIDs (e.g. tuple-packed expval qubits) still resolve.
                for i, qubit_uuid in enumerate(qubits_value.get_element_uuids()):
                    addr = QubitAddress(qubit_uuid)
                    if addr in uuid_to_physical:
                        qubit_map[i] = uuid_to_physical[addr]
            else:
                # View case: enumerate by view length (via element_uuids
                # or shape) and look up each position in the root's
                # registered (root_uuid, index) composite keys.
                length = len(qubits_value.get_element_uuids())
                if length == 0 and qubits_value.shape:
                    resolved = self._resolver.resolve_int_value(
                        qubits_value.shape[0], self.bindings
                    )
                    length = resolved if resolved is not None else 0
                for i in range(length):
                    addr = QubitAddress(root_av.uuid, start + step * i)
                    if addr in uuid_to_physical:
                        qubit_map[i] = uuid_to_physical[addr]
        else:
            # Single qubit or Vector case - map index 0 to the qubit
            if hasattr(qubits_value, "uuid"):
                addr = QubitAddress(qubits_value.uuid)
                if addr in uuid_to_physical:
                    qubit_map[0] = uuid_to_physical[addr]

        return qubit_map

    @abstractmethod
    def _emit_quantum_segment(
        self,
        operations: list[Operation],
        bindings: dict[str, Any],
    ) -> tuple[T, QubitMap, ClbitMap]:
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
