"""Emit pass: Generate backend-specific code from separated program."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import qamomile.observable as qm_o
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.callable import (
    CompositeGateType,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.gate import (
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
    ProjectOperation,
    ResetOperation,
)
from qamomile.circuit.ir.types.primitives import BitType
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
    collect_value_like_uuids,
    resolve_root_qubit_address,
)
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.executable import (
    CompiledClassicalSegment,
    CompiledExpvalSegment,
    CompiledQuantumSegment,
    ExecutableProgram,
    ParameterMetadata,
)
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.passes.control_flow_visitor import OperationCollector
from qamomile.circuit.transpiler.passes.emit_support.condition_resolution import (
    resolve_condition_address_detailed,
)
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
    """Protocol for preserving or lowering boxed callable operations.

    The concrete compiler installs a semantic emitter that boxes every
    executable callable into circuit IR with its identity and fallback body.
    Native SDK selection happens later, through target capabilities and
    legalization; this traversal hook must not import or select a backend.
    """

    def can_emit(self, gate_type: CompositeGateType) -> bool:
        """Check whether this emitter handles the given callable kind.

        Args:
            gate_type (CompositeGateType): Callable kind to check.

        Returns:
            bool: True if this emitter handles the callable kind.
        """
        ...

    def emit(
        self,
        circuit: C,
        op: InvokeOperation,
        qubit_indices: list[int],
        bindings: dict[str, Any],
    ) -> bool:
        """Preserve or lower the boxed callable into the output builder.

        Args:
            circuit (C): Output builder receiving the callable representation.
            op (InvokeOperation): Invocation to preserve or lower.
            qubit_indices (list[int]): Physical qubit indices for the operation.
            bindings (dict[str, Any]): Parameter bindings for the operation.

        Returns:
            bool: True if handled, or False to use the generic fallback
            decomposition.
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

        Raises:
            ValueError: If a name appears in both ``bindings`` and
                ``parameters``. This is the innermost emit-side choke point:
                it catches the overlap even when an ``EmitPass`` is constructed
                directly (e.g. via ``Transpiler._create_emit_pass``), bypassing
                the ``transpile`` / ``emit`` wrappers. A name in both is
                ambiguous and would otherwise silently bake the binding while
                dropping the runtime parameter (see #354).
        """
        from qamomile.circuit.frontend.param_validation import (
            validate_bindings_parameters_disjoint,
        )

        # Validate before wrapping ``bindings`` into an ``EmitContext``: on a
        # re-entrant call ``bindings`` may already be an ``EmitContext`` (a dict
        # subclass), but ``.keys()`` still exposes the raw names for the check.
        validate_bindings_parameters_disjoint(bindings, parameters)

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
        """Emit backend code from a program plan.

        Args:
            input (ProgramPlan): Segmented plan whose quantum and classical
                steps should be compiled.

        Returns:
            ExecutableProgram[T]: Executable program containing all compiled
                segments and the public output contract.

        """
        # ``Block.parameters`` includes special values such as Observable
        # inputs even when they are supplied as compile-time bindings. The
        # public bindings/parameters disjointness check has already rejected
        # genuine user overlap; subtract bound manifest entries here so only
        # unbound symbols are promoted into the backend runtime ABI.
        planned_parameters = set(input.parameters) - set(self.bindings)
        if not planned_parameters.issubset(self.parameters):
            self.parameters.update(planned_parameters)
            self._resolver = ValueResolver(self.parameters)

        self._program_output_values = tuple(input.abi.output_values)
        self._program_output_refs = frozenset(
            uuid
            for value in self._program_output_values
            for uuid in collect_value_like_uuids(value)
        )
        quantum_segments: list[QuantumSegment] = []
        compiled_classical: list[CompiledClassicalSegment] = []
        classical_segments: list[ClassicalSegment] = []
        compiled_expval: list[CompiledExpvalSegment] = []
        expval_segments: list[ExpvalSegment] = []
        compiled_quantum: list[CompiledQuantumSegment[T]] = []

        if any(isinstance(step, ExpvalStep) for step in input.steps):
            nonunitary_types = (
                MeasureOperation,
                MeasureQFixedOperation,
                MeasureVectorOperation,
                ProjectOperation,
                ResetOperation,
            )
            for step in input.steps:
                if not isinstance(step, QuantumStep):
                    continue
                collector = OperationCollector(
                    lambda operation: isinstance(operation, nonunitary_types)
                )
                collector.visit_operations(step.segment.operations)
                if collector.collected:
                    operation_names = sorted(
                        {type(operation).__name__ for operation in collector.collected}
                    )
                    raise EmitError(
                        "Programs that compute expval cannot also contain "
                        "measurement, projection, or reset operations in the "
                        "same quantum execution. Split sampling and expectation "
                        "evaluation into separate kernels. Found: "
                        f"{operation_names}."
                    )

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
            output_values=list(input.abi.output_values),
        )

    def _compile_quantum(
        self,
        segment: QuantumSegment,
    ) -> CompiledQuantumSegment[T]:
        """Compile a quantum segment to a backend circuit.

        Args:
            segment (QuantumSegment): Segment whose operations and live
                quantum-to-classical boundary outputs should be emitted.

        Returns:
            CompiledQuantumSegment[T]: Compiled circuit plus physical resource
                and parameter metadata.
        """
        self._current_quantum_output_refs = frozenset(segment.output_refs)
        try:
            circuit, qubit_map, clbit_map = self._emit_quantum_segment(
                segment.operations,
                self.bindings,
            )
        finally:
            del self._current_quantum_output_refs

        self._register_public_output_clbit_aliases(clbit_map)

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

    def _register_public_output_clbit_aliases(self, clbit_map: ClbitMap) -> None:
        """Alias structural public Bit outputs onto their physical clbits.

        Vector measurements allocate classical resources under root-array
        addresses such as ``QubitAddress(bits.uuid, i)``. A later element or
        slice access has its own UUID, while the public ABI historically kept
        only that UUID. Registering aliases after emission (when loop-carried
        indices have their final emit-time bindings) lets measurement loading
        populate both the physical root address and the public output identity.

        Args:
            clbit_map (ClbitMap): Physical clbit map to extend in place.

        Returns:
            None: The supplied map receives zero or more alias addresses.

        Raises:
            EmitError: If a public Bit view carries slice bounds or a length
                that cannot be resolved at emit time.
        """
        output_values = getattr(self, "_program_output_values", ())
        for descriptor in output_values:
            for output in self._flatten_public_output_leaves(descriptor):
                if isinstance(output, Value) and isinstance(output.type, BitType):
                    address, resolved_as_element = resolve_condition_address_detailed(
                        output,
                        self.bindings,
                        self._resolver,
                    )
                    if resolved_as_element and address in clbit_map:
                        clbit_map[QubitAddress(output.uuid)] = clbit_map[address]
                    continue

                if not (
                    isinstance(output, ArrayValue)
                    and isinstance(output.type, BitType)
                    and output.slice_of is not None
                ):
                    continue

                root, start, step = self._resolver.resolve_slice_chain(
                    output,
                    self.bindings,
                    operation="public Bit view output",
                )
                if not output.shape:
                    from qamomile.circuit.transpiler.errors import EmitError

                    raise EmitError(
                        "A public Vector[Bit] view has no shape information.",
                        operation="public Bit view output",
                    )
                length = self._resolver.resolve_int_value(
                    output.shape[0], self.bindings
                )
                if length is None:
                    from qamomile.circuit.transpiler.errors import EmitError

                    raise EmitError(
                        "A public Vector[Bit] view has an unresolved length.",
                        operation="public Bit view output",
                    )
                for local_index in range(length):
                    source = QubitAddress(root.uuid, start + step * local_index)
                    if source in clbit_map:
                        clbit_map[QubitAddress(output.uuid, local_index)] = clbit_map[
                            source
                        ]

    @staticmethod
    def _flatten_public_output_leaves(value: ValueBase) -> tuple[ValueBase, ...]:
        """Flatten tuple and dictionary output descriptors to typed leaves.

        Frontend return values normally arrive as already-flattened ABI
        descriptors. Low-level blocks may instead expose a ``TupleValue`` or
        ``DictValue`` directly, so emit-time output handling must inspect the
        scalar and array leaves stored in both dictionary keys and values.
        Array ancestry is deliberately not traversed here: a returned element
        is one leaf, while its parent vector is only addressing metadata and
        must not be treated as a separate whole-vector output.

        Args:
            value (ValueBase): Public output descriptor to flatten.

        Returns:
            tuple[ValueBase, ...]: Non-container leaves in deterministic
                tuple/dictionary entry order.
        """
        leaves: list[ValueBase] = []
        pending: list[ValueBase] = [value]
        visited: set[int] = set()
        while pending:
            current = pending.pop()
            identity = id(current)
            if identity in visited:
                continue
            visited.add(identity)
            if isinstance(current, TupleValue):
                pending.extend(reversed(current.elements))
                continue
            if isinstance(current, DictValue):
                children = [child for entry in current.entries for child in entry]
                pending.extend(reversed(children))
                continue
            leaves.append(current)
        return tuple(leaves)

    def _allocator_live_output_refs(self) -> frozenset[str] | None:
        """Return outputs that must survive quantum resource allocation.

        The public ABI output may be a classical expression produced after the
        quantum segment.  In that shape its measurement input is absent from
        ``ProgramABI.output_refs`` but present in ``QuantumSegment.output_refs``.
        Both sets must therefore seed allocator liveness.  Returning ``None``
        when neither context exists keeps direct low-level emit calls
        fail-closed for mixed prior/fresh Bit merges.

        Returns:
            frozenset[str] | None: Union of public and current-segment output
                UUIDs, or None when emission has no complete plan context.
        """
        program_outputs = getattr(self, "_program_output_refs", None)
        segment_outputs = getattr(self, "_current_quantum_output_refs", None)
        if program_outputs is None and segment_outputs is None:
            return None
        return frozenset(program_outputs or ()) | frozenset(segment_outputs or ())

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
        observable_indices = {
            operator.index
            for operators, _coefficient in hamiltonian
            for operator in operators
        }
        invalid_indices = sorted(observable_indices.difference(qubit_map))
        if invalid_indices:
            raise ValueError(
                "Observable qubit indices are outside the register passed to "
                f"expval: {invalid_indices}. Valid logical indices are "
                f"{sorted(qubit_map)}."
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
        ``start + step * i``-th physical qubit. For a tuple of Vector
        elements whose own UUIDs are not registered in the quantum
        segment (e.g. an ungated ancilla, or a gate/composite result),
        the flat UUID lookup misses and i falls back to the element's
        root ``(array_uuid, index)`` address captured at trace time,
        which the root array's ``QInitOperation`` always registers. The
        caller then feeds this mapping into ``hamiltonian.remap_qubits(...)``
        which rewrites observable qubit indices to physical-space
        indices in one go.

        Args:
            qubits_value (Value | None): The Value representing the qubit
                tuple / array / view to evaluate the observable over.
                ``None`` yields an empty map.
            quantum_segment_index (int): Index of the quantum segment whose
                compiled ``qubit_map`` supplies the physical qubit
                positions.
            compiled_quantum (list[CompiledQuantumSegment[T]]): Already
                compiled quantum segments.

        Returns:
            dict[int, int]: Mapping from Pauli index to physical qubit
                index. Empty when the segment index is invalid or when no
                entry resolved.
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
                #
                # On a miss, fall back to the element's root
                # ``(root_uuid, index)`` address captured at trace time: the
                # root array's QInitOperation always registers that composite
                # key, so this resolves a Vector element whose own (per-version)
                # UUID was never registered in the quantum segment -- e.g. an
                # ungated ancilla, or an element that is a gate/composite
                # result.  Standalone qubits carry the ``("", -1)`` sentinel
                # (``None`` here) and stay on the flat-lookup path above.
                parent_addrs = qubits_value.get_element_parent_addresses()
                for i, qubit_uuid in enumerate(qubits_value.get_element_uuids()):
                    addr = QubitAddress(qubit_uuid)
                    if addr in uuid_to_physical:
                        qubit_map[i] = uuid_to_physical[addr]
                        continue
                    # ``get_element_parent_addresses()`` returns exactly one
                    # entry per ``get_element_uuids()`` element, so indexing by
                    # ``i`` here is always in range.
                    root_addr_pair = parent_addrs[i]
                    if root_addr_pair is not None:
                        root_uuid, root_idx = root_addr_pair
                        root_addr = QubitAddress(root_uuid, root_idx)
                        if root_addr in uuid_to_physical:
                            qubit_map[i] = uuid_to_physical[root_addr]

                # Whole Vector[Qubit] operands created by
                # ``qubit_array(...)`` do not carry explicit
                # element_uuids; QInit registers them under
                # QubitAddress(array_uuid, i) instead.  If the legacy
                # tuple-style lookup above did not resolve anything,
                # fall through to the same root-address enumeration used
                # for views so offset whole registers remap observable
                # indices to their actual physical qubits.
                if qubit_map:
                    return qubit_map

            # View case, or whole Vector fallback: enumerate by operand
            # length (via element_uuids or shape) and look up each
            # position in the root's registered (root_uuid, index)
            # composite keys.  For non-view whole arrays, start=0 and
            # step=1.
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
                else:
                    # Bare single ``Qubit`` operand (``expval(q, H)`` without an
                    # enclosing tuple). When it is a Vector element whose own
                    # UUID was never registered (e.g. an ungated ancilla passed
                    # as ``expval(anc[0], H)``), fall back to its root
                    # ``(root_uuid, index)`` address. Unlike the tuple form, the
                    # bare element Value keeps its ``parent_array`` into emit, so
                    # it is resolved directly here without trace-time capture.
                    resolved = resolve_root_qubit_address(qubits_value)
                    if resolved is not None:
                        root_uuid, root_idx = resolved
                        root_addr = QubitAddress(root_uuid, root_idx)
                        if root_addr in uuid_to_physical:
                            qubit_map[0] = uuid_to_physical[root_addr]

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
