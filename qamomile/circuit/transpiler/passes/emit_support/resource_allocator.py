"""Resource allocation helpers for emission."""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING, Any

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    IfOperation,
    WhileOperation,
    genuine_input_values,
)
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    ControlledUOperation,
    GateOperation,
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
    ProjectOperation,
    ResetOperation,
    SymbolicControlledU,
)
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.types.primitives import BitType
from qamomile.circuit.ir.value import (
    ArrayValue,
    Value,
    ValueBase,
    array_physical_region,
    resolve_root_qubit_address,
)
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.passes.emit_support.condition_resolution import (
    _is_unresolved_bit_element,
    _merge_source_is_branch_local_measurement,
    map_merge_outputs,
    remap_static_merge_outputs,
    resolve_condition_address_detailed,
    resolve_if_condition,
)
from qamomile.circuit.transpiler.passes.emit_support.physical_index_map import (
    copy_array_element_aliases,
)
from qamomile.circuit.transpiler.passes.emit_support.qubit_address import (
    ClbitMap,
    QubitAddress,
    QubitMap,
)
from qamomile.circuit.transpiler.passes.emit_support.value_resolver import (
    ValueResolver,
    resolve_qubit_key,
)

if TYPE_CHECKING:
    from qamomile.circuit.ir.value import Value


class ResourceAllocator:
    """Allocates qubit and classical bit indices from operations.

    This class handles the first pass of circuit emission: determining
    how many physical qubits and classical bits are needed and mapping
    Value UUIDs to their physical indices.

    New physical indices are assigned via monotonic counters
    (``_next_qubit_index`` / ``_next_clbit_index``) so that alias
    entries — which reuse an existing physical index — never inflate
    the counter.  Using ``len(map)`` would cause sparse (gapped)
    physical indices because alias keys increase the map size without
    adding new physical resources.
    """

    def __init__(self, resolver: ValueResolver | None = None) -> None:
        """Initialize allocator state.

        Args:
            resolver (ValueResolver | None): Emit value resolver that carries
                runtime parameter names and binding lookup rules. Defaults to
                None, which creates a resolver without runtime parameters.
        """
        self._next_qubit_index: int = 0
        self._next_clbit_index: int = 0
        self._resolver = resolver or ValueResolver()
        self._safe_mixed_bit_merge_outputs: frozenset[str] = frozenset()

    @property
    def safe_mixed_bit_merge_outputs(self) -> frozenset[str]:
        """Return mixed-provenance Bit merges proven safe to alias.

        Returns:
            frozenset[str]: Merge-result UUIDs whose pre-existing clbit source
                is dead after the merge on every enclosing path.
        """
        return self._safe_mixed_bit_merge_outputs

    def _find_safe_mixed_bit_merge_outputs(
        self,
        operations: list[Operation],
        bindings: dict[str, Any],
        dependency_graph: dict[str, set[str]],
        public_output_uuids: frozenset[str] | set[str] | None,
    ) -> frozenset[str]:
        """Find prior/fresh Bit merges whose prior clbit is provably dead.

        Args:
            operations (list[Operation]): Complete operation tree to inspect.
            bindings (dict[str, Any]): Bindings used to resolve vector-element
                clbit addresses.
            dependency_graph (dict[str, set[str]]): Classical value dependency
                graph used to trace lazy expressions to measured sources.
            public_output_uuids (frozenset[str] | set[str] | None): Program
                outputs live after this operation tree. ``None`` means a full
                liveness proof is unavailable, so no mixed merge is approved.

        Returns:
            frozenset[str]: Merge-result UUIDs safe for clbit aliasing.
        """
        if public_output_uuids is None:
            return frozenset()

        merges: list[tuple[IfOperation, Value, Value, Value]] = []
        reads_after_if: dict[int, set[str]] = {}
        value_table: dict[str, ValueBase] = {}

        def register_value(value: ValueBase) -> None:
            """Register a value and all structural child Values.

            Args:
                value (ValueBase): Value to add to ``value_table``.
            """
            if value.uuid in value_table:
                return
            value_table[value.uuid] = value
            for attr in ("parent_array", "slice_of", "slice_start", "slice_step"):
                referenced = getattr(value, attr, None)
                if isinstance(referenced, ValueBase):
                    register_value(referenced)
            for attr in ("element_indices", "shape", "elements"):
                for referenced in getattr(value, attr, ()):
                    if isinstance(referenced, ValueBase):
                        register_value(referenced)
            for key, item in getattr(value, "entries", ()):
                if isinstance(key, ValueBase):
                    register_value(key)
                if isinstance(item, ValueBase):
                    register_value(item)

        def direct_references(value: ValueBase) -> set[str]:
            """Collect direct dataflow references without parent widening.

            Args:
                value (ValueBase): Value read at one program point.

            Returns:
                set[str]: Its UUID plus index, shape, and container members.
            """
            references = {value.uuid}
            for attr in ("element_indices", "shape", "elements"):
                for referenced in getattr(value, attr, ()):
                    if isinstance(referenced, ValueBase):
                        references.update(direct_references(referenced))
            for attr in ("slice_start", "slice_step"):
                referenced = getattr(value, attr, None)
                if isinstance(referenced, ValueBase):
                    references.update(direct_references(referenced))
            for key, item in getattr(value, "entries", ()):
                if isinstance(key, ValueBase):
                    references.update(direct_references(key))
                if isinstance(item, ValueBase):
                    references.update(direct_references(item))
            return references

        def operation_reads(operation: Operation) -> set[str]:
            """Collect direct reads from an operation subtree.

            Args:
                operation (Operation): Operation tree to inspect.

            Returns:
                set[str]: UUIDs read by the operation or nested bodies.
            """
            references: set[str] = set()
            for value in genuine_input_values(operation):
                references.update(direct_references(value))
            if isinstance(operation, HasNestedOps):
                for body in operation.nested_op_lists():
                    for nested in body:
                        references.update(operation_reads(nested))
            return references

        def record_suffix_reads(
            scope: list[Operation],
            inherited: set[str],
            *,
            cyclic: bool = False,
            cycle_prefix: set[str] | None = None,
        ) -> None:
            """Record path-aware values observed after each nested if.

            A loop body is a cyclic scope.  Reads before an if in the next
            iteration are therefore live after that if in the current
            iteration: once a branch-local measurement overwrites an external
            clbit, those prefix reads would observe the new value rather than
            the snapshot captured before the loop.  The current if's own merge
            sources are deliberately not part of its prefix; a while-condition
            phi is expected to read its prior value at the update itself.

            Args:
                scope (list[Operation]): Sequential scope to walk backwards.
                inherited (set[str]): Reads after the scope returns.
                cyclic (bool): Whether control returns to the start of this
                    scope after its end. Defaults to False.
                cycle_prefix (set[str] | None): Reads executed before this
                    nested scope on the next traversal of an enclosing cycle.
                    Defaults to None.
            """
            prefix_reads: dict[int, set[str]] = {}
            prefix = set(cycle_prefix or ())
            if cyclic:
                for operation in scope:
                    prefix_reads[id(operation)] = set(prefix)
                    prefix.update(operation_reads(operation))

            suffix = set(inherited)
            for operation in reversed(scope):
                if isinstance(operation, IfOperation):
                    reads_after_if[id(operation)] = set(suffix) | prefix_reads.get(
                        id(operation), set()
                    )
                    result_uuids = {
                        merge.result.uuid for merge in operation.iter_merges()
                    }
                    branch_suffix = suffix - result_uuids
                    true_exit: set[str] = set()
                    false_exit: set[str] = set()
                    for merge in operation.iter_merges():
                        true_exit.update(direct_references(merge.true_value))
                        false_exit.update(direct_references(merge.false_value))
                    nested_cycle_prefix = set(prefix_reads.get(id(operation), ()))
                    if operation.operands and isinstance(
                        operation.operands[0], ValueBase
                    ):
                        nested_cycle_prefix.update(
                            direct_references(operation.operands[0])
                        )
                    record_suffix_reads(
                        operation.true_operations,
                        branch_suffix | true_exit,
                        cyclic=cyclic,
                        cycle_prefix=nested_cycle_prefix,
                    )
                    record_suffix_reads(
                        operation.false_operations,
                        branch_suffix | false_exit,
                        cyclic=cyclic,
                        cycle_prefix=nested_cycle_prefix,
                    )
                elif isinstance(
                    operation, (ForOperation, ForItemsOperation, WhileOperation)
                ):
                    enclosing_cycle_prefix = prefix_reads.get(id(operation), set())
                    for body in operation.nested_op_lists():
                        record_suffix_reads(
                            body,
                            suffix,
                            cyclic=True,
                            cycle_prefix=enclosing_cycle_prefix,
                        )
                elif isinstance(operation, HasNestedOps):
                    for body in operation.nested_op_lists():
                        record_suffix_reads(
                            body,
                            suffix,
                            cyclic=cyclic,
                            cycle_prefix=prefix_reads.get(id(operation), set()),
                        )
                suffix.update(operation_reads(operation))

        def walk(scope: list[Operation]) -> None:
            """Collect merge records and the Values needed for lookup.

            Args:
                scope (list[Operation]): Current nested operation list.
            """
            for operation in scope:
                for value in (*genuine_input_values(operation), *operation.results):
                    register_value(value)
                if isinstance(operation, IfOperation):
                    for merge in operation.iter_merges():
                        register_value(merge.true_value)
                        register_value(merge.false_value)
                        register_value(merge.result)
                        merges.append(
                            (
                                operation,
                                merge.true_value,
                                merge.false_value,
                                merge.result,
                            )
                        )
                if isinstance(operation, HasNestedOps):
                    for body in operation.nested_op_lists():
                        walk(body)

        walk(operations)
        record_suffix_reads(operations, set(public_output_uuids))

        def depends_on_address(
            uuid: str,
            target: QubitAddress,
            barriers: frozenset[str],
            visiting: frozenset[str] = frozenset(),
        ) -> bool:
            """Return whether a read reaches a pre-merge clbit address.

            Args:
                uuid (str): Read UUID to trace backwards.
                target (QubitAddress): Pre-existing clbit address.
                barriers (frozenset[str]): Current merge results representing
                    updated phi values.
                visiting (frozenset[str]): UUIDs on the recursion path.

            Returns:
                bool: Whether ``target`` is read without crossing a barrier.
            """
            if uuid in barriers or uuid in visiting:
                return False
            value = value_table.get(uuid)
            if value is not None and isinstance(value.type, BitType):
                if isinstance(value, ArrayValue):
                    root = value
                    while root.slice_of is not None:
                        root = root.slice_of
                    if root.uuid == target.uuid:
                        region = array_physical_region(value)
                        if region is None or target.element_index is None:
                            return True
                        return target.element_index in region[1]
                elif isinstance(value, Value):
                    address, resolved_as_element = resolve_condition_address_detailed(
                        value, bindings, self._resolver
                    )
                    if not _is_unresolved_bit_element(value, resolved_as_element):
                        if address == target:
                            return True
                        if resolved_as_element:
                            return False
            next_visiting = visiting | {uuid}
            return any(
                depends_on_address(dependency, target, barriers, next_visiting)
                for dependency in dependency_graph.get(uuid, ())
            )

        allowed: set[str] = set()
        for if_operation, true_source, false_source, result in merges:
            if not isinstance(result.type, BitType):
                continue
            true_local = _merge_source_is_branch_local_measurement(
                true_source, if_operation.true_operations
            )
            false_local = _merge_source_is_branch_local_measurement(
                false_source, if_operation.false_operations
            )
            if true_local == false_local:
                continue
            external_source = false_source if true_local else true_source
            external_addresses: tuple[QubitAddress, ...]
            if isinstance(external_source, ArrayValue):
                region = array_physical_region(external_source)
                if region is None:
                    continue
                root = external_source
                while root.slice_of is not None:
                    root = root.slice_of
                external_addresses = tuple(
                    QubitAddress(root.uuid, root_index) for root_index in region[1]
                )
            else:
                external_address, resolved_as_element = (
                    resolve_condition_address_detailed(
                        external_source, bindings, self._resolver
                    )
                )
                if _is_unresolved_bit_element(external_source, resolved_as_element):
                    continue
                external_addresses = (external_address,)

            # A pre-existing clbit cannot serve as the in-place update slot
            # for one merge while another output of the same runtime branch
            # still selects that old value.  The two merge results are
            # simultaneous: treating every result as a liveness barrier would
            # incorrectly approve crossed shapes such as
            # ``(fresh, old)`` / ``(old, fresh)`` and collapse two independent
            # outputs onto one clbit.  Fail closed whenever another merge
            # source in this If reaches the candidate external address.
            same_if_sources = tuple(
                source
                for other_if, other_true, other_false, other_result in merges
                if other_if is if_operation and other_result.uuid != result.uuid
                for source in (other_true, other_false)
            )
            if any(
                depends_on_address(source.uuid, external_address, frozenset())
                for external_address in external_addresses
                for source in same_if_sources
            ):
                continue
            barriers = frozenset(
                merge.result.uuid for merge in if_operation.iter_merges()
            )
            if any(
                depends_on_address(read, external_address, barriers)
                for external_address in external_addresses
                for read in reads_after_if.get(id(if_operation), ())
            ):
                continue
            allowed.add(result.uuid)
        return frozenset(allowed)

    def _validate_while_condition_lineages(
        self,
        operations: list[Operation],
        bindings: dict[str, Any],
    ) -> None:
        """Reject while-condition updates that require copying an old clbit.

        Qamomile implements a runtime while-condition update by making every
        branch-local measurement write into the condition's existing physical
        clbit.  That is a phi update, not a classical copy: a pre-measured Bit
        from elsewhere cannot be assigned to the condition because its value
        already lives in another clbit before the loop begins.  Aliasing that
        foreign address would instead retarget the earlier measurement and can
        change the loop's *initial* condition.

        Args:
            operations (list[Operation]): Complete operation tree to validate.
            bindings (dict[str, Any]): Active bindings used to resolve measured
                vector element addresses.

        Raises:
            EmitError: If a loop-carried while condition has a pre-existing
                leaf that is neither the current condition nor a measurement
                executed in the loop body branch that selects it.
        """

        def validate_while(operation: WhileOperation) -> None:
            """Validate one while operation's updated-condition lineage.

            Args:
                operation (WhileOperation): While operation to validate.

            Raises:
                EmitError: If its updated condition contains a foreign
                    pre-measured Bit leaf.
            """
            if len(operation.operands) != 2:
                return
            initial = operation.operands[0]
            updated = operation.operands[1]

            initial_address, initial_resolved = resolve_condition_address_detailed(
                initial, bindings, self._resolver
            )
            initial_unresolved = _is_unresolved_bit_element(initial, initial_resolved)
            merge_producers: dict[str, tuple[IfOperation, Value, Value]] = {}

            def collect_merges(scope: list[Operation]) -> None:
                """Collect if-merge producers in the while body.

                Args:
                    scope (list[Operation]): Nested operation list to scan.
                """
                for nested in scope:
                    if isinstance(nested, IfOperation):
                        for merge in nested.iter_merges():
                            merge_producers[merge.result.uuid] = (
                                nested,
                                merge.true_value,
                                merge.false_value,
                            )
                    if isinstance(nested, HasNestedOps):
                        for body in nested.nested_op_lists():
                            collect_merges(body)

            collect_merges(operation.operations)

            def is_current_condition(source: Value) -> bool:
                """Return whether a source denotes the current condition slot.

                Args:
                    source (Value): Candidate pre-existing condition source.

                Returns:
                    bool: True for the same UUID or the same resolved vector
                        element address as the initial condition.
                """
                if source.uuid == initial.uuid:
                    return True
                source_address, source_resolved = resolve_condition_address_detailed(
                    source, bindings, self._resolver
                )
                if initial_unresolved or _is_unresolved_bit_element(
                    source, source_resolved
                ):
                    return False
                return source_address == initial_address

            def validate_source(
                source: Value,
                scope: list[Operation],
                visiting: frozenset[str] = frozenset(),
            ) -> None:
                """Validate one updated-condition source recursively.

                Args:
                    source (Value): Source selected for the next condition.
                    scope (list[Operation]): Branch scope in which the source
                        may have been freshly measured.
                    visiting (frozenset[str]): Merge-result UUIDs on the current
                        recursion path. Defaults to an empty set.

                Raises:
                    EmitError: If the source is a foreign pre-measured Bit or a
                        cyclic/malformed merge lineage.
                """
                if is_current_condition(source):
                    return
                if _merge_source_is_branch_local_measurement(source, scope):
                    return
                if source.uuid in visiting:
                    raise EmitError(
                        "Runtime while-condition update has a cyclic Bit merge "
                        "lineage that cannot be mapped to one condition clbit.",
                        operation="WhileOperation",
                    )
                producer = merge_producers.get(source.uuid)
                if producer is not None:
                    if_operation, true_source, false_source = producer
                    next_visiting = visiting | {source.uuid}
                    validate_source(
                        true_source,
                        if_operation.true_operations,
                        next_visiting,
                    )
                    validate_source(
                        false_source,
                        if_operation.false_operations,
                        next_visiting,
                    )
                    return
                raise EmitError(
                    "Runtime while-condition update selects a pre-existing "
                    "measured Bit that is not the current condition. Clbit "
                    "aliasing can route a branch-local remeasurement into the "
                    "condition slot, but it cannot copy a value already stored "
                    "in another clbit. Remeasure the condition in that branch "
                    "or restructure the loop.",
                    operation="WhileOperation",
                )

            validate_source(updated, operation.operations)

        def walk(scope: list[Operation]) -> None:
            """Validate every while nested in an operation tree.

            Args:
                scope (list[Operation]): Operation list to scan recursively.
            """
            for operation in scope:
                if isinstance(operation, WhileOperation):
                    validate_while(operation)
                if isinstance(operation, HasNestedOps):
                    for body in operation.nested_op_lists():
                        walk(body)

        walk(operations)

    @staticmethod
    def _coerce_nonnegative_integral_size(value: Any) -> int | None:
        """Coerce a non-negative, non-boolean integral value to a Python int.

        Args:
            value (Any): Candidate structural size value resolved from IR
                constants, compile-time bindings, or bound array elements.

        Returns:
            int | None: The coerced integer size, or None when ``value`` is
                negative, not an integral numeric value, or is a boolean.
        """
        # Python bool is an Integral subclass, but it is never a valid size.
        # NumPy bool scalars are rejected because they are not Integral values.
        if isinstance(value, bool):
            return None
        if isinstance(value, numbers.Integral):
            size = int(value)
            return size if size >= 0 else None
        return None

    def _compact_new_clbit_indices(
        self,
        clbit_map: ClbitMap,
        first_new_index: int,
    ) -> None:
        """Remove holes among clbits allocated by the current call.

        Runtime branch merging first allocates a clbit for each branch-local
        measurement and then aliases the mutually exclusive writes onto one
        physical slot.  Later measurements would otherwise retain the skipped
        numeric indices, inflating the backend circuit's classical register.
        Only indices at or above ``first_new_index`` are renumbered; every
        caller-supplied ``initial_clbit_map`` entry therefore keeps its exact
        physical index.

        Args:
            clbit_map (ClbitMap): Address map to compact in place.
            first_new_index (int): First index available to this allocation
                call, equal to one past the maximum initial-map index.
        """
        allocated = sorted(
            {physical for physical in clbit_map.values() if physical >= first_new_index}
        )
        remapping = {
            physical: first_new_index + offset
            for offset, physical in enumerate(allocated)
        }
        if any(old != new for old, new in remapping.items()):
            for address, physical in tuple(clbit_map.items()):
                replacement = remapping.get(physical)
                if replacement is not None:
                    clbit_map[address] = replacement
        self._next_clbit_index = first_new_index + len(allocated)

    def allocate(
        self,
        operations: list[Operation],
        bindings: dict[str, Any] | None = None,
        initial_qubit_map: QubitMap | None = None,
        initial_clbit_map: ClbitMap | None = None,
        public_output_uuids: frozenset[str] | set[str] | None = None,
    ) -> tuple[QubitMap, ClbitMap]:
        """Allocate qubit and clbit indices for all operations.

        Args:
            operations (list[Operation]): Operations to allocate resources
                for.
            bindings (dict[str, Any] | None): Optional variable bindings
                for resolving dynamic sizes. Defaults to None (treated as
                an empty mapping).
            initial_qubit_map (QubitMap | None): Optional pre-populated
                qubit address mapping. Used by callers that need to seed
                the allocator with bindings established outside the
                operation list — for instance, the inner-block emitter
                in ``blockvalue_to_gate`` pre-allocates ``Vector[Qubit]``
                input elements (the inner block has no ``QInitOperation``
                for inputs, so per-element entries must be supplied here
                or the assertion in ``_allocate_gate`` fires). The map is
                copied; allocation continues from
                ``max(values) + 1`` so new ``QInitOperation`` allocations
                inside ``operations`` do not collide. Defaults to None
                (treated as empty).
            initial_clbit_map (ClbitMap | None): Optional pre-populated
                clbit address mapping. Same semantics as
                ``initial_qubit_map`` but for classical bits. Defaults to
                None.
            public_output_uuids (frozenset[str] | set[str] | None): Program
                outputs that must remain live after this operation tree.
                Defaults to None, which disables dead-source approval for
                mixed prior/fresh Bit merges.

        Returns:
            tuple[QubitMap, ClbitMap]: ``(qubit_map, clbit_map)`` where
                each maps ``QubitAddress`` to a physical index. If an
                initial map was supplied, its entries are preserved
                verbatim in the returned map.
        """
        qubit_map: QubitMap = dict(initial_qubit_map) if initial_qubit_map else {}
        clbit_map: ClbitMap = dict(initial_clbit_map) if initial_clbit_map else {}
        first_new_clbit_index = max(clbit_map.values(), default=-1) + 1
        # Distinguish genuine runtime (measurement-derived) if conditions
        # from loop-bound predicates that remain symbolic until unrolled at
        # emit time. Only the former may need the pre-measured Bit mux guard.
        from qamomile.circuit.transpiler.passes.analyze import (
            build_dependency_graph,
            find_measurement_derived_values,
            find_measurement_results,
        )

        dependency_graph = build_dependency_graph(operations)
        self._measurement_tainted = find_measurement_derived_values(
            dependency_graph, find_measurement_results(operations)
        )
        resolved_bindings = bindings or {}
        self._validate_while_condition_lineages(operations, resolved_bindings)
        self._safe_mixed_bit_merge_outputs = self._find_safe_mixed_bit_merge_outputs(
            operations,
            resolved_bindings,
            dependency_graph,
            public_output_uuids,
        )
        self._next_qubit_index = max(qubit_map.values(), default=-1) + 1
        self._next_clbit_index = first_new_clbit_index
        self._allocate_recursive(operations, qubit_map, clbit_map, resolved_bindings)
        self._compact_new_clbit_indices(clbit_map, first_new_clbit_index)
        return qubit_map, clbit_map

    def resolve_iteration_maps(
        self,
        operations: list[Operation],
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> tuple[QubitMap, ClbitMap]:
        """Resolve resource aliases for one statically replayed iteration.

        The initial allocation walks an unrolled loop body only once, before
        its loop variable has a concrete value.  A nested runtime condition
        such as ``while measured[index]:`` therefore cannot select its root
        clbit during that first walk.  Replaying the allocator over copies of
        the already-complete maps with the iteration binding present resolves
        those aliases without allocating another physical resource.  The
        caller uses the returned maps for that iteration only; the same body
        UUIDs may legitimately alias different measured-vector elements in
        later iterations.

        Args:
            operations (list[Operation]): Statically replayed loop body.
            qubit_map (QubitMap): Complete enclosing qubit map.
            clbit_map (ClbitMap): Complete enclosing clbit map.
            bindings (dict[str, Any]): Iteration-local UUID bindings, including
                the concrete range index or items-loop key/value identities.

        Returns:
            tuple[QubitMap, ClbitMap]: Per-iteration copies with conditional
                aliases resolved for the supplied bindings.

        Raises:
            EmitError: If resolving the bound body exposes an invalid quantum
                merge, classical-bit merge, array size, or while-condition
                lineage.
            AssertionError: If replay unexpectedly allocates a new physical
                qubit or clbit instead of only adding or changing aliases.
        """
        iteration_qubit_map = dict(qubit_map)
        iteration_clbit_map = dict(clbit_map)
        next_qubit = self._next_qubit_index
        next_clbit = self._next_clbit_index
        existing_qubits = set(qubit_map.values())
        existing_clbits = set(clbit_map.values())
        try:
            self._allocate_recursive(
                operations,
                iteration_qubit_map,
                iteration_clbit_map,
                bindings,
            )
        finally:
            # This is an alias-resolution replay, not a second allocation.
            # Restore the monotonic counters even if malformed IR fails.
            self._next_qubit_index = next_qubit
            self._next_clbit_index = next_clbit

        assert set(iteration_qubit_map.values()) <= existing_qubits, (
            "[FOR DEVELOPER] Iteration resource replay allocated a new physical "
            "qubit; the initial allocation must cover every loop-body resource."
        )
        assert set(iteration_clbit_map.values()) <= existing_clbits, (
            "[FOR DEVELOPER] Iteration resource replay allocated a new physical "
            "clbit; the initial allocation must cover every loop-body resource."
        )
        return iteration_qubit_map, iteration_clbit_map

    def _allocate_recursive(
        self,
        operations: list[Operation],
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Recursively allocate resources from operations.

        Args:
            operations (list[Operation]): Operations to walk, including
                nested control-flow bodies.
            qubit_map (QubitMap): Qubit address mapping mutated in place.
            clbit_map (ClbitMap): Clbit address mapping mutated in place.
            bindings (dict[str, Any]): Parameter bindings used to resolve
                symbolic array sizes.

        Raises:
            EmitError: If a quantum array's size cannot be resolved from
                ``bindings``, or if a ``QInitOperation`` allocates a
                rank>1 quantum register — the ``QubitAddress`` keys
                registered here carry a single flat index, so a
                higher-rank register would silently alias distinct
                elements onto the same physical qubit. The frontend
                rejects such registers at construction time; this guard
                covers hand-built or deserialized IR.
        """
        for op in operations:
            if isinstance(op, QInitOperation):
                result = op.results[0]
                if isinstance(result, ArrayValue):
                    # ``len()`` is read into a variable so zuban does not
                    # narrow the variadic shape tuple to fixed-length
                    # forms that make ``result.shape[0]`` look out of
                    # range below.
                    rank = len(result.shape)
                    if rank > 1:
                        raise EmitError(
                            f"Cannot allocate qubits for rank-{rank} "
                            f"quantum register {result.name!r}: the qubit "
                            f"addressing path is rank-1, so a higher-rank "
                            f"register would silently alias distinct "
                            f"elements onto the same physical qubit. Use a "
                            f"1-D Vector[Qubit] with explicit index "
                            f"arithmetic instead."
                        )
                    # Allocate physical qubits for array elements using
                    # QubitAddress(array_uuid, i) keys.  At this stage only
                    # these composite keys are registered; the individual
                    # element Values (which carry their own UUIDs) are created
                    # dynamically during frontend tracing and their UUID
                    # mapping is deferred to _allocate_gate / _allocate_qubit_list.
                    if result.shape:
                        size_val = result.shape[0]
                        size = self._resolve_size(size_val, bindings)
                        if size is None:
                            raise EmitError(
                                "Cannot resolve array size for qubit allocation. "
                                "Structural UInt parameters must be bound at transpile time."
                            )
                        for i in range(size):
                            qubit_addr = QubitAddress(result.uuid, i)
                            if qubit_addr not in qubit_map:
                                qubit_map[qubit_addr] = self._next_qubit_index
                                self._next_qubit_index += 1
                    continue
                scalar_addr = QubitAddress(result.uuid)
                if scalar_addr not in qubit_map:
                    qubit_map[scalar_addr] = self._next_qubit_index
                    self._next_qubit_index += 1

            elif isinstance(op, MeasureOperation):
                result = op.results[0]
                clbit_addr = QubitAddress(result.uuid)
                if clbit_addr not in clbit_map:
                    clbit_map[clbit_addr] = self._next_clbit_index
                    self._next_clbit_index += 1

            elif isinstance(op, ProjectOperation):
                qubit_in = op.operands[0]
                qubit_out = op.results[0]
                bit_out = op.results[1]
                self._allocate_qubit_list([qubit_in], [qubit_out], qubit_map)
                clbit_addr = QubitAddress(bit_out.uuid)
                if clbit_addr not in clbit_map:
                    clbit_map[clbit_addr] = self._next_clbit_index
                    self._next_clbit_index += 1

            elif isinstance(op, ResetOperation):
                self._allocate_qubit_list(op.operands, op.results, qubit_map)

            elif isinstance(op, MeasureVectorOperation):
                result = op.results[0]
                if isinstance(result, ArrayValue) and result.shape:
                    size_val = result.shape[0]
                    size = self._resolve_size(size_val, bindings)
                    if size is not None:
                        for i in range(size):
                            clbit_addr = QubitAddress(result.uuid, i)
                            if clbit_addr not in clbit_map:
                                clbit_map[clbit_addr] = self._next_clbit_index
                                self._next_clbit_index += 1

            elif isinstance(op, MeasureQFixedOperation):
                qfixed = op.operands[0]
                qubit_uuids = qfixed.get_qfixed_qubit_uuids()
                result = op.results[0]
                for i, qubit_uuid in enumerate(qubit_uuids):
                    clbit_addr = QubitAddress(result.uuid, i)
                    if clbit_addr not in clbit_map:
                        clbit_map[clbit_addr] = self._next_clbit_index
                        self._next_clbit_index += 1

            elif isinstance(op, GateOperation):
                self._allocate_gate(op, qubit_map)

            elif isinstance(op, IfOperation):
                resolved = resolve_if_condition(op.condition, bindings)
                if resolved is not None:
                    # Compile-time constant: only allocate the selected
                    # branch and remap merge outputs directly.
                    selected = op.true_operations if resolved else op.false_operations
                    self._allocate_recursive(selected, qubit_map, clbit_map, bindings)
                    self._remap_static_merge_outputs(
                        op, resolved, qubit_map, clbit_map, bindings
                    )
                else:
                    self._allocate_recursive(
                        op.true_operations, qubit_map, clbit_map, bindings
                    )
                    self._allocate_recursive(
                        op.false_operations, qubit_map, clbit_map, bindings
                    )
                    self._allocate_if_merges(op, qubit_map, clbit_map, bindings)

            elif isinstance(op, WhileOperation):
                # WhileOperation operands:
                #   operands[0]: initial condition (always present)
                #   operands[1]: loop-carried condition (present only when the
                #                body reassigns the condition variable)
                # No other operand count is valid.
                if len(op.operands) == 1:
                    # Invariant condition: the condition variable is not
                    # reassigned inside the loop body.  No loop-carried
                    # clbit aliasing is needed; just allocate the body.
                    self._allocate_recursive(
                        op.operations, qubit_map, clbit_map, bindings
                    )
                elif len(op.operands) == 2:
                    initial_cond = op.operands[0]
                    loop_carried = op.operands[1]
                    init_val = (
                        initial_cond.value
                        if hasattr(initial_cond, "value")
                        else initial_cond
                    )
                    init_addr = self._condition_source_address(init_val, bindings)

                    # Save the canonical clbit for the initial condition
                    # BEFORE body allocation.  An if-only (no else) inside
                    # the loop body produces a merge whose false_value is
                    # the pre-if while-condition value.  map_merge_outputs
                    # will redirect that false-source UUID to the
                    # true-branch clbit,
                    # overwriting clbit_map[init_addr] and making the
                    # post-body mismatch detection ineffective.
                    saved_init_clbit = clbit_map.get(init_addr)

                    # Allocate the loop body so that IfOperation merge
                    # mappings inside the body are fully resolved.
                    self._allocate_recursive(
                        op.operations, qubit_map, clbit_map, bindings
                    )

                    # Restore the canonical clbit for init_addr if it was
                    # overwritten during body allocation.
                    if saved_init_clbit is not None:
                        clbit_map[init_addr] = saved_init_clbit

                    carried_val = (
                        loop_carried.value
                        if hasattr(loop_carried, "value")
                        else loop_carried
                    )
                    carried_addr = self._condition_source_address(carried_val, bindings)
                    carried_clbit = clbit_map.get(carried_addr)

                    # Alias the loop-carried condition to the initial
                    # while-condition clbit.  After body allocation the
                    # loop-carried UUID may point to a different clbit
                    # (e.g. a merged measurement from an if-else).
                    # We recursively trace IfOperation merges and map all
                    # upstream branch-measurement UUIDs to the canonical clbit.
                    if (
                        saved_init_clbit is not None
                        and carried_clbit is not None
                        and saved_init_clbit != carried_clbit
                    ):
                        clbit_map[carried_addr] = saved_init_clbit
                        self._alias_loop_carried_clbits(
                            op.operations,
                            carried_addr,
                            saved_init_clbit,
                            clbit_map,
                        )
                    elif saved_init_clbit is not None and carried_addr not in clbit_map:
                        clbit_map[carried_addr] = saved_init_clbit
                else:
                    assert False, (
                        "[FOR DEVELOPER] WhileOperation must have exactly 2 "
                        "operands to reach this branch, but got "
                        f"{len(op.operands)}. This indicates a bug in the "
                        "WhileOperation construction."
                    )

            elif isinstance(op, HasNestedOps):
                # Generic recursion for For/ForItems: recurse into all nested bodies.
                for body in op.nested_op_lists():
                    self._allocate_recursive(body, qubit_map, clbit_map, bindings)

            elif isinstance(op, PauliEvolveOp):
                self._allocate_pauli_evolve(op, qubit_map)

            elif isinstance(op, (InverseBlockOperation, InvokeOperation)):
                self._allocate_composite(op, qubit_map)

            elif isinstance(op, ControlledUOperation):
                self._allocate_controlled_u(op, qubit_map, bindings)

            elif isinstance(op, CastOperation):
                self._allocate_cast(op, qubit_map)

    def _condition_source_address(
        self,
        value: Any,
        bindings: dict[str, Any],
    ) -> QubitAddress:
        """Resolve a while / merge condition source to its ``clbit_map`` key.

        A measured ``Vector[Bit]`` element (``s[i]`` from
        ``s = qmc.measure(register)``) registers its clbit under
        ``QubitAddress(root_array.uuid, index)``, not the element's own
        UUID. Resolving the source the same way the emit-time condition
        lookup does keeps the loop-carried / merge clbit alias consistent
        with where the clbit was actually allocated. The allocator's
        ``ValueResolver`` folds symbolic indices / slice bounds through
        ``bindings`` (e.g. an unrolled loop variable), so a loop-indexed
        element resolves to its root clbit; a genuinely unresolvable
        element still falls back to its scalar UUID (which is not
        registered, so the caller's ``clbit_map`` lookup misses and the
        aliasing is skipped rather than pointed at a wrong slot).

        Args:
            value (Any): The condition / merge source — an IR ``Value`` or, in
                degenerate cases, a non-Value (handled via ``str``).
            bindings (dict[str, Any]): Active bindings for folding symbolic
                element indices / slice bounds via the resolver.

        Returns:
            QubitAddress: The address the source's classical bit is
                registered under.
        """
        if isinstance(value, Value):
            address, _ = resolve_condition_address_detailed(
                value, bindings, self._resolver
            )
            return address
        return QubitAddress(str(value))

    def _alias_loop_carried_clbits(
        self,
        operations: list[Operation],
        target_addr: QubitAddress,
        canonical_clbit: int,
        clbit_map: ClbitMap,
    ) -> None:
        """Recursively trace if-merge sources and alias them to *canonical_clbit*.

        When a while loop body contains an if-else with measurements in
        both branches, the merged result (the loop-carried condition)
        and all its upstream branch-measurement UUIDs must write to the
        same classical bit as the initial while condition.
        """
        for op in operations:
            if isinstance(op, HasNestedOps) and not isinstance(op, IfOperation):
                for body in op.nested_op_lists():
                    self._alias_loop_carried_clbits(
                        body, target_addr, canonical_clbit, clbit_map
                    )
                continue
            if not isinstance(op, IfOperation):
                continue
            for merge in op.iter_merges():
                if merge.result.uuid != target_addr.uuid:
                    continue
                true_addr = self._condition_source_address(merge.true_value, {})
                false_addr = self._condition_source_address(merge.false_value, {})
                if true_addr in clbit_map:
                    clbit_map[true_addr] = canonical_clbit
                if false_addr in clbit_map:
                    clbit_map[false_addr] = canonical_clbit
                # Recurse into branches for nested if-else
                self._alias_loop_carried_clbits(
                    op.true_operations, true_addr, canonical_clbit, clbit_map
                )
                self._alias_loop_carried_clbits(
                    op.false_operations, false_addr, canonical_clbit, clbit_map
                )

    def _allocate_if_merges(
        self,
        op: IfOperation,
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Register merge output UUIDs via the shared ``map_merge_outputs`` utility.

        Args:
            op (IfOperation): The runtime if-else whose merged outputs need
                physical-resource registration.
            qubit_map (QubitMap): QubitAddress-to-physical-qubit mapping
                (mutated in place).
            clbit_map (ClbitMap): QubitAddress-to-physical-clbit mapping
                (mutated in place).
            bindings (dict[str, Any]): Active bindings passed with the
                allocator's ``ValueResolver`` so a merge source that is a
                loop-indexed measured ``Vector[Bit]`` element resolves to
                its root clbit.

        Raises:
            EmitError: If a genuine runtime merge selects two distinct
                pre-existing measured Bit/clbit regions, or if quantum merge
                sources do not denote identical physical qubits.
        """
        map_merge_outputs(
            op,
            qubit_map,
            clbit_map,
            bindings=bindings,
            resolver=self._resolver,
            reject_runtime_bit_mux=(
                bool(op.operands) and op.operands[0].uuid in self._measurement_tainted
            ),
            allowed_mixed_bit_outputs=self._safe_mixed_bit_merge_outputs,
        )

    def _remap_static_merge_outputs(
        self,
        op: IfOperation,
        condition_value: bool,
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Remap merge outputs for a compile-time constant ``IfOperation``.

        Delegates to the module-level ``remap_static_merge_outputs`` helper,
        which is shared with the emit pass to ensure scalar and array
        quantum merge outputs are handled identically at both allocation
        and emission time.

        Args:
            op (IfOperation): The if-else whose condition resolved at
                compile time.
            condition_value (bool): The resolved boolean condition.
            qubit_map (QubitMap): QubitAddress-to-physical-qubit mapping
                (mutated in place).
            clbit_map (ClbitMap): QubitAddress-to-physical-clbit mapping
                (mutated in place).
            bindings (dict[str, Any]): Active bindings passed with the
                allocator's ``ValueResolver`` so a merge source that is a
                loop-indexed measured ``Vector[Bit]`` element resolves to
                its root clbit.
        """
        remap_static_merge_outputs(
            op,
            condition_value,
            qubit_map,
            clbit_map,
            bindings=bindings,
            resolver=self._resolver,
        )

    def _resolve_size(
        self,
        size_val: "Value",
        bindings: dict[str, Any],
    ) -> int | None:
        """Resolve a size Value to a concrete integer.

        Args:
            size_val (Value): IR value used as a qubit-array size.
                Constants, bound scalar values, shape-dimension values, and
                array-element values are supported when compile-time concrete.
            bindings (dict[str, Any]): Compile-time bindings available to the
                emit pass, keyed by parameter names or value UUIDs.

        Returns:
            int | None: Concrete integer size, or None when the value cannot
                be resolved at allocation time.
        """
        import re

        if size_val.is_constant():
            return self._coerce_nonnegative_integral_size(size_val.get_const())

        # UUID identity always wins over display labels. Inlined locals and
        # nested-loop temporaries can legitimately share a human-readable name
        # with an unrelated kernel binding.
        if size_val.uuid in bindings:
            bound = bindings[size_val.uuid]
            if (size := self._coerce_nonnegative_integral_size(bound)) is not None:
                return size

        # Declared parameter provenance is the only scalar-name lookup with
        # semantic meaning. A bare Value.name is display-only and must not
        # capture an unrelated user binding.
        if size_val.is_parameter():
            parameter_name = size_val.parameter_name()
            if parameter_name and parameter_name in bindings:
                bound = bindings[parameter_name]
                if (size := self._coerce_nonnegative_integral_size(bound)) is not None:
                    return size

        # Array element (e.g., sizes[0]): delegate to the emit value resolver
        # so bound containers and VectorView slices follow the same lookup
        # rules as other emit-time value resolution paths. Resolver refusal
        # is final here; symbolic array-element sizes must stay unresolved.
        if size_val.parent_array is not None and size_val.element_indices:
            return self._coerce_nonnegative_integral_size(
                self._resolver.resolve_bound_value(size_val, bindings)
            )

        # Dimension naming pattern (e.g., "hi_dim0" -> array "hi", dimension 0).
        # Handles cases where parent_array is None after inlining.
        if size_val.name:
            match = re.match(r"^(.+)_dim(\d+)$", size_val.name)
            if match:
                array_name = match.group(1)
                dim_index = int(match.group(2))
                if array_name in bindings:
                    array_data = bindings[array_name]
                    if hasattr(array_data, "shape"):
                        if dim_index < len(array_data.shape):
                            return int(array_data.shape[dim_index])
                    elif dim_index == 0 and hasattr(array_data, "__len__"):
                        return len(array_data)

        return None

    def _allocate_gate(
        self,
        op: GateOperation,
        qubit_map: QubitMap,
    ) -> None:
        """Allocate resources for a GateOperation."""
        # GateOperation represents a unitary gate: qubit count is preserved.
        qubit_ops = op.qubit_operands
        assert len(op.results) == len(qubit_ops), (
            f"GateOperation must have equal qubit operands and results, "
            f"got {len(qubit_ops)} qubit operands and {len(op.results)} results."
        )

        # Phase 1: Register all operands in qubit_map.
        # Element Values are created dynamically during frontend tracing
        # (handle/array.py _get_element), so their UUIDs are unknown at
        # QInitOperation time.  Here we lazily map each element UUID to
        # the physical qubit already allocated under the root parent's
        # QubitAddress key — the chain walk in
        # ``_resolve_root_qubit_address`` handles sliced views so that
        # e.g. ``view[0]`` for ``view = q[1:3]`` resolves to the
        # physical qubit for ``q[1]``.
        for operand in qubit_ops:
            operand_addr = QubitAddress(operand.uuid)
            if operand_addr not in qubit_map:
                chain_addr = self._resolve_root_qubit_address(operand)
                if chain_addr is not None:
                    assert chain_addr in qubit_map, (
                        f"Array element key {str(chain_addr)!r} not found in qubit_map. "
                        f"This indicates a bug in the transpiler pipeline: "
                        f"QInitOperation for the parent array was not processed "
                        f"before this GateOperation."
                    )
                    qubit_map[operand_addr] = qubit_map[chain_addr]
                elif operand.parent_array is None or not operand.element_indices:
                    # Scalar qubit: allocate new index.
                    # This path is used for @qkernel input parameters created with
                    # emit_init=False (func_to_block.py), which have no QInitOperation.
                    # ResourceAllocator.allocate() receives only operations, not the
                    # block's input_values, so these qubits are first registered here.
                    qubit_map[operand_addr] = self._next_qubit_index
                    self._next_qubit_index += 1
                # Non-constant indices (symbolic loop vars) are resolved
                # at emit time via ValueResolver.resolve_qubit_index_detailed.

        # Phase 2: Map each result to its corresponding qubit operand (1:1)
        for i, result in enumerate(op.results):
            result_addr = QubitAddress(result.uuid)
            if result_addr not in qubit_map:
                operand = qubit_ops[i]
                operand_addr = QubitAddress(operand.uuid)
                if operand_addr in qubit_map:
                    qubit_map[result_addr] = qubit_map[operand_addr]

    def _allocate_qubit_list(
        self,
        all_qubits: list["Value"],
        results: list["Value"],
        qubit_map: QubitMap,
    ) -> None:
        """Allocate qubits for a list of qubit Values and their results.

        For view elements (operand whose ``parent_array.slice_of`` is
        set), the root-space ``QubitAddress`` is derived by walking the
        slice chain and composing the affine map.  No new physical
        qubit is allocated for a view element — the root parent's
        pre-existing physical qubit is reused — which prevents the
        previous bug where e.g. ``qft(q[1::2])`` inflated the
        circuit's qubit count beyond the true number of physical
        qubits because every view element was mistakenly registered
        under its view-local ``(view_uuid, i)`` key.
        """
        for qubit in all_qubits:
            chain_addr = self._resolve_root_qubit_address(qubit)
            if chain_addr is not None:
                qubit_addr = chain_addr
                if qubit_addr not in qubit_map:
                    # Element key missing under the root parent — this
                    # means the QInitOperation for the root was never
                    # allocated.  Treat it as a hard bug rather than
                    # silently allocating a fresh index (the prior
                    # behaviour), because doing so inflates the
                    # circuit's qubit count when a view is passed to
                    # a composite gate.
                    raise AssertionError(
                        f"Root qubit address '{str(qubit_addr)}' not found in qubit_map "
                        f"when allocating for element '{qubit.uuid}'. "
                        "The root array's QInitOperation must be allocated first."
                    )
            elif isinstance(qubit, ArrayValue):
                # Whole ``Vector[Qubit]`` operand (not a per-element
                # access): its per-element addresses are already in
                # ``qubit_map`` (placed there by the upstream
                # ``QInitOperation`` or by a prior allocator that
                # produced this array as its result, see the
                # ArrayValue→ArrayValue copy in the result loop
                # below).  Do **not** fall through to the scalar
                # fresh-allocate branch -- doing so would allocate a
                # single wire for the whole vector and leave the
                # element keys missing, which then trips
                # ``_resolve_root_qubit_address`` on a downstream op
                # that addresses elements of this vector (e.g.
                # ``qmc.iqft`` after ``ctrl_qft(c, q_out, coef)``).
                qubit_addr = None
            else:
                qubit_addr, is_array = resolve_qubit_key(qubit)
                if qubit_addr is None:
                    continue
                if qubit_addr not in qubit_map:
                    # Scalar qubit or symbolic-index element: fall back
                    # to the legacy allocation behaviour so qkernel
                    # input parameters (emit_init=False) and symbolic
                    # loop-var indices keep working.
                    qubit_map[qubit_addr] = self._next_qubit_index
                    self._next_qubit_index += 1

            if qubit_addr is not None:
                scalar_addr = QubitAddress(qubit.uuid)
                if scalar_addr not in qubit_map:
                    qubit_map[scalar_addr] = qubit_map[qubit_addr]

        for i, result in enumerate(results):
            result_addr = QubitAddress(result.uuid)
            if i < len(all_qubits):
                operand = all_qubits[i]
                # ArrayValue input → ArrayValue result: alias every
                # per-element address from the input's UUID to the
                # result's UUID.  Mirrors :meth:`_allocate_pauli_evolve`
                # and the ``SymbolicControlledU`` control-prefix branch
                # in :meth:`_allocate_controlled_u`.  Without this copy,
                # a ``ConcreteControlledU`` with a ``Vector[Qubit]``
                # sub-kernel argument leaves the next-version vector's
                # element keys unmapped, and any downstream op that
                # walks ``parent_array.uuid -> (uuid, i)`` (e.g.
                # ``qmc.iqft`` expanded into per-element CP / H gates,
                # or a ``MeasureVectorOperation`` on the result)
                # trips the ``_resolve_root_qubit_address`` assertion.
                if isinstance(operand, ArrayValue) and isinstance(result, ArrayValue):
                    copy_array_element_aliases(operand.uuid, result.uuid, qubit_map)
                    continue

                chain_addr = self._resolve_root_qubit_address(operand)
                if chain_addr is not None:
                    qubit_addr = chain_addr
                else:
                    qubit_addr, _ = resolve_qubit_key(operand)
                if qubit_addr is not None and qubit_addr in qubit_map:
                    physical = qubit_map[qubit_addr]
                    if result_addr not in qubit_map:
                        qubit_map[result_addr] = physical
                    # If the result is itself an array element, also
                    # register the (parent_array.uuid, idx) key so that
                    # downstream operations referencing the result's
                    # parent ``ArrayValue`` (e.g. ``MeasureVectorOperation``
                    # on a controlled-U's next-version control vector)
                    # can resolve each element through the same path
                    # used for QInit-allocated arrays.
                    result_chain_addr = self._resolve_root_qubit_address(result)
                    if (
                        result_chain_addr is not None
                        and result_chain_addr not in qubit_map
                    ):
                        qubit_map[result_chain_addr] = physical
                elif qubit_addr is not None and result_addr not in qubit_map:
                    raise AssertionError(
                        f"Missing qubit address '{str(qubit_addr)}' in qubit_map when "
                        f"allocating result '{result.uuid}'. "
                        "This indicates a bug in operand allocation."
                    )

    def _resolve_root_qubit_address(
        self,
        operand: "Value",
    ) -> QubitAddress | None:
        """Walk the slice_of chain and return the root-space QubitAddress.

        Thin wrapper over :func:`resolve_root_qubit_address` (shared with the
        frontend's ``expval`` lowering) that wraps the resolved
        ``(root_uuid, index)`` pair in a ``QubitAddress``.

        Args:
            operand (Value): The qubit operand to resolve; expected to be an
                array element with a constant index.

        Returns:
            QubitAddress | None: ``QubitAddress(root_uuid, index)`` for a
                resolvable array element, or ``None`` when the operand is not an
                array element, its index is non-constant, or the slice chain has
                a non-constant ``slice_start`` / ``slice_step`` (deferred to the
                emit-time resolver, which has bindings available).
        """
        resolved = resolve_root_qubit_address(operand)
        if resolved is None:
            return None
        root_uuid, idx = resolved
        return QubitAddress(root_uuid, idx)

    def _allocate_pauli_evolve(
        self,
        op: PauliEvolveOp,
        qubit_map: QubitMap,
    ) -> None:
        """Allocate resources for a PauliEvolveOp.

        Maps result array elements to the same physical qubits as
        the input array (identity mapping -- same qubits, new SSA values).
        """
        input_qubits = op.qubits
        result_qubits = op.evolved_qubits
        copy_array_element_aliases(input_qubits.uuid, result_qubits.uuid, qubit_map)

    def _allocate_composite(
        self,
        op: InverseBlockOperation | InvokeOperation,
        qubit_map: QubitMap,
    ) -> None:
        """Allocate resources for a composite-like quantum operation."""
        all_qubits = op.control_qubits + op.target_qubits
        self._allocate_qubit_list(all_qubits, list(op.results), qubit_map)

    def _allocate_controlled_u(
        self,
        op: ControlledUOperation,
        qubit_map: QubitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Allocate resources for a controlled-U operation.

        Args:
            op (ControlledUOperation): Controlled-U operation whose quantum
                operands and pass-through results need physical qubit slots.
            qubit_map (QubitMap): Mutable mapping from IR qubit addresses to
                physical qubit indices.
            bindings (dict[str, Any]): Active compile-time bindings used to
                resolve symbolic array-element controls.
        """
        if isinstance(op, SymbolicControlledU):
            from qamomile.circuit.ir.value import ArrayValue

            # Three shapes can land here without having been promoted to
            # ``ConcreteControlledU`` by ``ConstantFoldingPass``:
            #
            #   * single-pool + ``control_indices``: one ``ArrayValue``
            #     control operand whose pass-through slots cannot be
            #     represented in ``ConcreteControlledU``'s scalar layout.
            #   * multi-arg control prefix (``num_control_args > 1``):
            #     a heterogeneous mix of scalar ``Value`` and
            #     ``ArrayValue`` operands whose qubit-count sum equals
            #     ``num_controls``.
            #   * single-pool with no ``control_indices`` but a
            #     ``num_controls`` that depends on a loop variable
            #     (``num_controls = n - 1 - k`` inside a ``qmc.range``):
            #     constant folding cannot resolve the loop variable so
            #     the promotion never fires; each unrolled iteration
            #     instead arrives at the emit pass with a fully
            #     resolvable ``num_controls``.
            #
            # All three flow through the same per-operand allocation:
            # each input operand keeps its physical mapping onto the
            # corresponding result operand.  Whether ``num_controls``
            # ultimately resolves is the emit pass's responsibility;
            # the allocator only needs to thread per-element addresses
            # through.
            for i in range(op.num_control_args):
                src = op.operands[i]
                dst = op.results[i]
                if isinstance(src, ArrayValue):
                    copy_array_element_aliases(src.uuid, dst.uuid, qubit_map)
                else:
                    src_addr = QubitAddress(src.uuid)
                    physical = qubit_map.get(src_addr)
                    if (
                        physical is None
                        and src.parent_array is not None
                        and src.element_indices
                    ):
                        physical = self._resolver.resolve_qubit_index(
                            src, qubit_map, bindings
                        )
                        if physical is not None:
                            qubit_map[src_addr] = physical
                    elif physical is None:
                        # Scalar control whose UUID is first introduced
                        # at this ``SymbolicControlledU`` -- typically a
                        # top-level ``@qkernel`` ``Qubit`` input
                        # (``emit_init=False``, so no ``QInitOperation``
                        # pre-registers it).  ``_allocate_qubit_list``
                        # already handles the same edge case for the
                        # concrete-controlled-U path (line ~527); mirror
                        # the fresh-slot allocation here so emit does
                        # not later trip on a missing scalar mapping
                        # for the control prefix.
                        physical = self._next_qubit_index
                        qubit_map[src_addr] = physical
                        self._next_qubit_index += 1
                    if physical is not None:
                        dst_addr = QubitAddress(dst.uuid)
                        if dst_addr not in qubit_map:
                            qubit_map[dst_addr] = physical
            sub_quantum_operands = [
                v for v in op.operands[op.num_control_args :] if v.type.is_quantum()
            ]
            sub_quantum_results = [
                r for r in op.results[op.num_control_args :] if r.type.is_quantum()
            ]
            if sub_quantum_operands:
                self._allocate_qubit_list(
                    sub_quantum_operands, sub_quantum_results, qubit_map
                )
            return

        assert isinstance(op, ConcreteControlledU)
        control_qubits = list(op.control_operands)
        target_qubits = [v for v in op.target_operands if v.type.is_quantum()]
        all_qubits = control_qubits + target_qubits
        self._allocate_qubit_list(all_qubits, list(op.results), qubit_map)

    @staticmethod
    def _parse_composite_key(key: str) -> QubitAddress:
        """Parse a legacy composite key string into a QubitAddress.

        Delegates to ``QubitAddress.from_composite_key``.
        """
        return QubitAddress.from_composite_key(key)

    def _allocate_cast(
        self,
        op: CastOperation,
        qubit_map: QubitMap,
    ) -> None:
        """Allocate resources for a CastOperation (alias mapping)."""
        resolved = 0
        for i, qubit_uuid in enumerate(op.qubit_mapping):
            qubit_addr = self._parse_composite_key(qubit_uuid)
            if qubit_addr not in qubit_map:
                # Fallback: try as plain scalar UUID (the qubit_mapping may
                # store element UUIDs that were registered via _allocate_gate)
                qubit_addr = QubitAddress(qubit_uuid)
            if qubit_addr in qubit_map:
                result_element_addr = QubitAddress(op.results[0].uuid, i)
                qubit_map[result_element_addr] = qubit_map[qubit_addr]
                result_base_addr = QubitAddress(op.results[0].uuid)
                if result_base_addr not in qubit_map:
                    qubit_map[result_base_addr] = qubit_map[qubit_addr]
                resolved += 1
        total = len(op.qubit_mapping)
        if total > 0 and resolved < total:
            import warnings

            warnings.warn(
                f"CastOperation: {total - resolved}/{total} carrier qubits "
                f"unresolved in qubit_map. "
                f"Downstream measurements may be silently dropped.",
                stacklevel=2,
            )
