"""Emit pass: Generate backend-specific code from separated program."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import qamomile.observable as qm_o
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
    CompositeGateType,
)
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.executable import (
    CompiledClassicalSegment,
    CompiledExpvalSegment,
    CompiledQuantumSegment,
    ExecutableProgram,
    ParameterMetadata,
)
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.segments import (
    ClassicalSegment,
    ExpvalSegment,
    QuantumSegment,
    SimplifiedProgram,
)

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


class EmitPass(Pass[SimplifiedProgram, ExecutableProgram[T]], Generic[T]):
    """Base class for backend-specific emission passes.

    Subclasses implement _emit_quantum_segment() to generate
    backend-specific quantum circuits.

    Input: SimplifiedProgram (enforces C→Q→C pattern)
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

    def run(self, input: SimplifiedProgram) -> ExecutableProgram[T]:
        """Emit backend code from simplified program."""
        # Pre-evaluate classical_prep to resolve values before quantum emission
        if input.classical_prep:
            self._pre_evaluate_classical(input.classical_prep)

        # Compile quantum segment (always present)
        compiled_quantum = [self._compile_quantum(input.quantum)]

        # Compile optional segments
        compiled_classical: list[CompiledClassicalSegment] = []
        compiled_expval: list[CompiledExpvalSegment] = []
        execution_order: list[tuple[str, int]] = []

        # Build execution order based on what's present
        if input.classical_prep:
            compiled_classical.append(self._compile_classical(input.classical_prep))
            execution_order.append(("classical", 0))

        # Quantum always present
        execution_order.append(("quantum", 0))

        # Expval or classical post (mutually exclusive in practice)
        if input.expval:
            compiled_expval.append(
                self._compile_expval(input.expval, 0, input, compiled_quantum)
            )
            execution_order.append(("expval", 0))

        if input.classical_post:
            idx = 1 if input.classical_prep else 0
            compiled_classical.append(self._compile_classical(input.classical_post))
            execution_order.append(("classical", idx))

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

    def _pre_evaluate_classical(self, segment: ClassicalSegment) -> None:
        """Pre-evaluate classical segment operations to populate bindings.

        Evaluates BinOps using current bindings so that their results
        are available during quantum emission (e.g., for gate angles
        computed from user-provided parameters).

        The default implementation resolves only concrete values (constants
        and already-bound names/UUIDs).  Built-in backends (via
        ``StandardEmitPass``) override this to use the full symbolic
        resolver so that unbound parameters produce backend parameter
        expressions instead of being silently dropped.
        """
        for op in segment.operations:
            if isinstance(op, BinOp):
                lhs_val = self._resolve_value(op.lhs)
                rhs_val = self._resolve_value(op.rhs)
                if lhs_val is None or rhs_val is None:
                    continue
                result = None
                match op.kind:
                    case BinOpKind.ADD:
                        result = lhs_val + rhs_val
                    case BinOpKind.SUB:
                        result = lhs_val - rhs_val
                    case BinOpKind.MUL:
                        result = lhs_val * rhs_val
                    case BinOpKind.DIV:
                        if rhs_val == 0:
                            raise EmitError(
                                "Division by zero during classical pre-evaluation"
                            )
                        result = lhs_val / rhs_val
                    case BinOpKind.FLOORDIV:
                        if rhs_val == 0:
                            raise EmitError(
                                "Floor division by zero during classical pre-evaluation"
                            )
                        result = lhs_val // rhs_val
                    case BinOpKind.POW:
                        result = lhs_val**rhs_val
                if result is not None and op.results:
                    output = op.results[0]
                    self.bindings[output.uuid] = result
                    self.bindings[output.name] = result

    def _resolve_value(self, value: Value) -> Any:
        """Resolve a Value to a concrete number from bindings or constants.

        Returns None if the value cannot be resolved.
        """
        if value.is_constant():
            return value.get_const()
        if value.uuid in self.bindings:
            return self.bindings[value.uuid]
        if value.name in self.bindings:
            return self.bindings[value.name]
        return None

    def _compile_expval(
        self,
        segment: ExpvalSegment,
        quantum_segment_index: int,
        program: SimplifiedProgram,
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
        if observable_value.name not in self.bindings:
            raise RuntimeError(
                f"Observable '{observable_value.name}' not found in bindings. "
                f"Hamiltonians must be provided as bindings."
            )

        hamiltonian = self.bindings[observable_value.name]

        # Validate type
        if not isinstance(hamiltonian, qm_o.Hamiltonian):
            raise TypeError(
                f"Expected qamomile.observable.Hamiltonian, got {type(hamiltonian)}"
            )

        # Build qubit mapping from Pauli index to physical qubit index
        qubit_map = self._build_qubit_map(
            segment.qubits_value,
            quantum_segment_index,
            compiled_quantum,
        )

        # Apply qubit remapping if needed
        if qubit_map:
            hamiltonian = hamiltonian.remap_qubits(qubit_map)

        # Create CompiledExpvalSegment with qm_o.Hamiltonian directly
        return CompiledExpvalSegment(
            segment=segment,
            hamiltonian=hamiltonian,
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

        Resolves the logical-to-physical qubit mapping for the expval target
        through the compiled quantum segment's ``qubit_map``.  Three target
        shapes are supported:

        * **Tuple synthetic array** – ``ArrayValue`` with
          ``params["qubit_values"]``.  Each element's UUID is looked up
          directly in ``uuid_to_physical``.
        * **Vector[Qubit] plain array** – ``ArrayValue`` *without*
          ``params["qubit_values"]``.  The physical mapping is recovered by
          scanning ``uuid_to_physical`` for keys matching the prefix
          ``"{array_uuid}_"``, then parsing the integer suffix as the logical
          index.
        * **Scalar Qubit** – A plain ``Value`` whose UUID is looked up
          directly.

        Args:
            qubits_value: The Value representing the qubit tuple/array/scalar.
            quantum_segment_index: Index of the quantum segment providing the
                state.
            compiled_quantum: List of already compiled quantum segments.

        Returns:
            Dict mapping Pauli index -> physical qubit index.

        Raises:
            EmitError: If a vector prefix-key scan finds keys that do not
                exactly match the declared shape, or if a positive-size
                vector has no prefix keys at all.
        """
        from qamomile.circuit.ir.value import ArrayValue
        from qamomile.circuit.transpiler.value_resolution import (
            BindingLookup,
            resolve_int_value,
        )

        qubit_map: dict[int, int] = {}

        if qubits_value is None or quantum_segment_index < 0:
            return qubit_map

        if quantum_segment_index >= len(compiled_quantum):
            return qubit_map

        compiled_seg = compiled_quantum[quantum_segment_index]
        uuid_to_physical = compiled_seg.qubit_map

        if isinstance(qubits_value, ArrayValue):
            qubit_values = qubits_value.params.get("qubit_values", [])
            if qubit_values:
                # Tuple synthetic array: each element's UUID is in qubit_map
                for i, qv in enumerate(qubit_values):
                    if hasattr(qv, "uuid") and qv.uuid in uuid_to_physical:
                        qubit_map[i] = uuid_to_physical[qv.uuid]
            else:
                # Vector[Qubit] plain array: scan for prefix keys
                prefix = f"{qubits_value.uuid}_"
                for key, physical in uuid_to_physical.items():
                    if key.startswith(prefix):
                        suffix = key[len(prefix) :]
                        try:
                            logical_idx = int(suffix)
                        except ValueError:
                            continue
                        qubit_map[logical_idx] = physical

                # Resolve declared size from shape for exact validation
                expected_size: int | None = None
                if qubits_value.shape:
                    lookup = BindingLookup(self.bindings)
                    expected_size = resolve_int_value(qubits_value.shape[0], lookup)

                if expected_size is not None and expected_size > 0:
                    # Fail-closed: found indices must exactly match
                    # {0, 1, ..., expected_size - 1}.
                    expected_indices = set(range(expected_size))
                    if qubit_map.keys() != expected_indices:
                        raise EmitError(
                            f"Incomplete vector qubit mapping: "
                            f"declared size {expected_size}, "
                            f"found prefix keys for {sorted(qubit_map.keys())} "
                            f"but expected indices {sorted(expected_indices)}."
                        )
                elif qubit_map:
                    # Fallback when declared size is unavailable:
                    # reject internal gaps up to observed max.
                    expected = set(range(max(qubit_map.keys()) + 1))
                    missing = expected - qubit_map.keys()
                    if missing:
                        raise EmitError(
                            f"Incomplete vector qubit mapping: "
                            f"found prefix keys for {sorted(qubit_map.keys())} "
                            f"but logical indices {sorted(missing)} are missing."
                        )
        else:
            # Scalar qubit
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
