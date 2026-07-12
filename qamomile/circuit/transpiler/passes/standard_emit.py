"""Standard emit pass using GateEmitter protocol.

This module provides StandardEmitPass, a reusable emit pass implementation
that uses the GateEmitter protocol for backend-specific gate emission.

The actual emission logic is decomposed into focused modules under
``emit_support/``. This class serves as the orchestrator with thin
wrappers that delegate to those module functions while preserving
subclass override points (used by QiskitEmitPass, CudaqEmitPass).
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator
from typing import Any, Generic, TypeVar, cast

from qamomile.circuit.ir.operation import (
    Operation,
    ReleaseSliceViewOperation,
    SliceArrayOperation,
)
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    CompOp,
    CondOp,
    NotOp,
    RuntimeClassicalExpr,
)
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.classical_ops import (
    DictGetItemOperation,
    StoreArrayElementOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    GateOperation,
    GateOperationType,
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
    ProjectOperation,
    ResetOperation,
)
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.emit_context import EmitContext
from qamomile.circuit.transpiler.executable import ParameterInfo, ParameterMetadata
from qamomile.circuit.transpiler.gate_emitter import GateEmitter
from qamomile.circuit.transpiler.passes.emit import CompositeGateEmitter, EmitPass
from qamomile.circuit.transpiler.passes.emit_support import (
    ClbitMap,
    CompositeDecomposer,
    LoopAnalyzer,
    MultiControlAncillaPool,
    QubitAddress,
    QubitMap,
    ResourceAllocator,
)
from qamomile.circuit.transpiler.passes.emit_support.cast_binop_emission import (
    evaluate_binop,
    evaluate_classical_predicate,
    handle_cast,
)
from qamomile.circuit.transpiler.passes.emit_support.composite_gate_emission import (
    emit_invoke_operation,
)
from qamomile.circuit.transpiler.passes.emit_support.control_flow_emission import (
    emit_for,
    emit_for_items,
    emit_if,
    emit_while,
    evaluate_dict_getitem,
    mark_updated_while_condition,
    reject_stale_runtime_condition,
)
from qamomile.circuit.transpiler.passes.emit_support.controlled_emission import (
    blockvalue_to_gate,
    emit_controlled_fallback,
    emit_controlled_u,
)
from qamomile.circuit.transpiler.passes.emit_support.counting_emitter import (
    CountingEmitter,
)
from qamomile.circuit.transpiler.passes.emit_support.gate_emission import emit_gate
from qamomile.circuit.transpiler.passes.emit_support.inverse_emission import (
    emit_inverse_block,
)
from qamomile.circuit.transpiler.passes.emit_support.measurement_emission import (
    emit_measure,
    emit_measure_qfixed,
    emit_measure_vector,
)
from qamomile.circuit.transpiler.passes.emit_support.pauli_evolve_emission import (
    emit_pauli_evolve,
)

T = TypeVar("T")  # Backend circuit type


def _segment_may_reserve_ancillas(operations: list[Operation]) -> bool:
    """Return True if the segment could reach the multi-control cascade.

    The counting dry-run that sizes the ancilla pool roughly doubles a
    segment's emission work, so it is only worth running when the segment
    can actually consume the pool. Only a ``ControlledUOperation`` (a
    ``qmc.control(...)`` call), an ``InverseBlockOperation`` (whose block
    may hold controlled gates), or an ``InvokeOperation`` (whose selected
    callable body may be controlled) reaches the irreducible
    multi-controlled hook, directly or after control composition. A segment
    of plain gates, measurements, and classical ops — the common case —
    never does, so the dry run is skipped. This is only a presence check,
    not a demand computation: when it says "maybe", the real count still
    comes from the dry run, so it cannot drift from the emitter.

    Args:
        operations (list[Operation]): The quantum segment's operations.

    Returns:
        bool: True if any operation, recursively through control-flow
            bodies, is a controlled-U, inverse block, or callable invocation.
    """
    for op in operations:
        if isinstance(
            op,
            (ControlledUOperation, InverseBlockOperation, InvokeOperation),
        ):
            return True
        if isinstance(op, HasNestedOps):
            for body in op.nested_op_lists():
                if _segment_may_reserve_ancillas(body):
                    return True
    return False


class StandardEmitPass(EmitPass[T], Generic[T]):
    """Standard emit pass implementation using GateEmitter protocol.

    This class provides the orchestration logic for circuit emission
    while delegating backend-specific operations to a GateEmitter.

    Subclasses (QiskitEmitPass, CudaqEmitPass) override specific methods
    to provide native backend support. The thin wrappers here delegate to
    module functions in ``emit_support/`` by default.

    Args:
        gate_emitter: Backend-specific gate emitter
        bindings: Parameter bindings for the circuit
        parameters: List of parameter names to preserve as backend parameters
        composite_emitters: Optional list of CompositeGateEmitter for native implementations
    """

    def __init__(
        self,
        gate_emitter: GateEmitter[T],
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
        composite_emitters: list[CompositeGateEmitter[T]] | None = None,
        backend_name: str | None = None,
    ):
        super().__init__(bindings, parameters)
        self._emitter = gate_emitter
        self._composite_emitters = composite_emitters or []
        self.backend_name = backend_name

        # Helper classes (``_resolver`` is built by ``EmitPass.__init__``).
        self._allocator = ResourceAllocator(self._resolver)
        self._loop_analyzer = LoopAnalyzer()
        self._decomposer = CompositeDecomposer()

        # Cache for backend parameter objects
        self._parameter_map: dict[str, Any] = {}
        self._parameter_sources: dict[str, str] = {}

        # Mapping from classical bit index to physical qubit index.
        # Populated during measurement emission to support backends
        # where emit_measure is a no-op (e.g., QURI Parts).
        self._measurement_qubit_map: dict[int, int] = {}

        # Clean ancilla qubits reserved for the shared Toffoli-cascade
        # lowering of irreducible multi-controlled gates. Populated per
        # quantum segment when ``_reserves_multi_control_ancillas()``
        # is True; None on backends with native multi-control support.
        self._mc_ancilla_pool: MultiControlAncillaPool | None = None

        # True only during the count-only dry-run walk in
        # ``_count_multi_control_ancilla_demand``. Backend paths that build
        # or manipulate real circuit objects (e.g. QURI Parts' native
        # inverse) consult this to stay on a counting-safe route.
        self._counting_emission = False

        # Path-local measurement snapshots whose physical clbits may have
        # been overwritten by runtime while-condition updates. The state is
        # forked/merged by runtime If emission and reset for each segment.
        self._overwritten_runtime_condition_sources: set[tuple[str, int]] = set()

    # ------------------------------------------------------------------
    # Core orchestration (stays on this class)
    # ------------------------------------------------------------------

    def _build_parameter_metadata(self) -> ParameterMetadata:
        """Build parameter metadata from created parameter objects."""
        params = []
        for name, backend_param in self._parameter_map.items():
            match = re.match(r"(\w+)\[(\d+)\]", name)
            if match:
                array_name = match.group(1)
                index = int(match.group(2))
            else:
                array_name = name
                index = None

            params.append(
                ParameterInfo(
                    name=name,
                    array_name=array_name,
                    index=index,
                    backend_param=backend_param,
                    source_ref=self._parameter_sources.get(name),
                )
            )

        return ParameterMetadata(parameters=params)

    def _get_or_create_parameter(
        self,
        name: str,
        source_ref: str | None = None,
    ) -> Any:
        """Get or create a backend parameter while tracking its IR source."""
        if name not in self._parameter_map:
            self._parameter_map[name] = self._emitter.create_parameter(name)
        if source_ref is not None:
            self._parameter_sources.setdefault(name, source_ref)
        return self._parameter_map[name]

    def _reject_overwritten_live_condition_outputs(self) -> None:
        """Reject exposing a measurement snapshot overwritten by a while.

        A runtime while update is implemented by remeasuring into the
        condition's physical clbit.  The updated condition Value is safe to
        expose, but the immutable measurement snapshot that originally owned
        that clbit is not.  Structural live-output refs include parent vector
        UUIDs, so returning the whole measured vector is caught as well as
        returning one scalar snapshot.

        Raises:
            EmitError: If a public or inter-segment live output still refers
                to an overwritten initial condition snapshot.
        """
        from qamomile.circuit.ir.types.primitives import BitType
        from qamomile.circuit.ir.value import ArrayValue, ValueBase
        from qamomile.circuit.transpiler.passes.emit_support.condition_resolution import (
            resolve_condition_address_detailed,
        )

        live_refs = self._allocator_live_output_refs()
        if live_refs is None:
            return
        output_values = getattr(self, "_program_output_values", {})
        precise_output_addresses: set[QubitAddress] = set()
        scalar_output_identities: set[str] = set()
        coarse_output_roots: set[str] = set()
        covered_output_refs: set[str] = set()

        def collect_structural_refs(
            value: ValueBase,
            visiting: frozenset[str] = frozenset(),
        ) -> None:
            """Collect UUID refs explained by one public output descriptor.

            Args:
                value (ValueBase): Public output or structural ancestor.
                visiting (frozenset[str]): UUIDs already visited on this path.
                    Defaults to an empty set.
            """
            if value.uuid in visiting:
                return
            covered_output_refs.add(value.uuid)
            next_visiting = visiting | {value.uuid}
            for attr in ("parent_array", "slice_of", "slice_start", "slice_step"):
                child = getattr(value, attr, None)
                if isinstance(child, ValueBase):
                    collect_structural_refs(child, next_visiting)
            for attr in ("element_indices", "shape", "elements"):
                for child in getattr(value, attr, ()):
                    if isinstance(child, ValueBase):
                        collect_structural_refs(child, next_visiting)
            for key, item in getattr(value, "entries", ()):
                if isinstance(key, ValueBase):
                    collect_structural_refs(key, next_visiting)
                if isinstance(item, ValueBase):
                    collect_structural_refs(item, next_visiting)

        for descriptor in output_values.values():
            if not isinstance(descriptor, ValueBase):
                continue
            collect_structural_refs(descriptor)
            for output in self._flatten_public_output_leaves(descriptor):
                if (
                    isinstance(output, Value)
                    and not isinstance(output, ArrayValue)
                    and isinstance(output.type, BitType)
                ):
                    address, resolved_as_element = resolve_condition_address_detailed(
                        output,
                        self.bindings,
                        self._resolver,
                    )
                    if resolved_as_element:
                        precise_output_addresses.add(address)
                    else:
                        scalar_output_identities.add(output.uuid)
                    continue
                if not (
                    isinstance(output, ArrayValue) and isinstance(output.type, BitType)
                ):
                    continue
                if output.slice_of is None:
                    root = output
                    start = 0
                    step = 1
                else:
                    root, start, step = self._resolver.resolve_slice_chain(
                        output,
                        self.bindings,
                        operation="public Bit view output liveness",
                    )
                length = (
                    self._resolver.resolve_int_value(output.shape[0], self.bindings)
                    if output.shape
                    else None
                )
                if length is None:
                    coarse_output_roots.add(root.uuid)
                    continue
                precise_output_addresses.update(
                    QubitAddress(root.uuid, start + step * index)
                    for index in range(length)
                )

        # Structural ancestry for a direct public ``bits[1]`` output includes
        # the root vector UUID. Remove descriptor-covered refs before applying
        # the conservative segment-boundary fallback, otherwise overwriting
        # ``bits[0]`` would incorrectly reject the disjoint ``bits[1]`` output.
        residual_live_refs = set(live_refs) - covered_output_refs
        for source_identity, _physical in self._overwritten_runtime_condition_sources:
            source_address = QubitAddress.from_composite_key(source_identity)
            if (
                source_address not in precise_output_addresses
                and source_identity not in scalar_output_identities
                and source_address.uuid not in coarse_output_roots
                and source_identity not in residual_live_refs
                and source_address.uuid not in residual_live_refs
            ):
                continue
            from qamomile.circuit.transpiler.errors import EmitError

            raise EmitError(
                "A measured Bit snapshot remains live after a runtime while "
                "update reused its physical clbit. Return or consume the "
                "updated condition Value instead of the original scalar or "
                "Vector measurement snapshot.",
                operation="WhileOperation",
            )

    def _emit_quantum_segment(
        self,
        operations: list[Operation],
        bindings: dict[str, Any],
    ) -> tuple[T, QubitMap, ClbitMap]:
        """Generate a backend circuit from one quantum segment.

        Args:
            operations (list[Operation]): Quantum/control-flow operation tree
                to allocate and emit.
            bindings (dict[str, Any]): Compile-time and emit-local value
                bindings.

        Returns:
            tuple[T, QubitMap, ClbitMap]: Emitted backend circuit and its
                logical-to-physical qubit and clbit maps.

        Raises:
            EmitError: If resource allocation or backend emission cannot
                represent the operation tree safely.
            RuntimeError: If an IR marker that should have been removed by an
                earlier pass reaches emission.
            ValueError: If a loop that requires unrolling has unresolved or
                invalid bounds.
        """
        self._overwritten_runtime_condition_sources.clear()
        qubit_map, clbit_map = self._allocator.allocate(
            operations,
            bindings,
            public_output_uuids=self._allocator_live_output_refs(),
        )
        self._safe_mixed_bit_merge_outputs = (
            self._allocator.safe_mixed_bit_merge_outputs
        )

        qubit_count = max(qubit_map.values()) + 1 if qubit_map else 0
        clbit_count = max(clbit_map.values()) + 1 if clbit_map else 0

        self._mc_ancilla_pool = None
        if self._reserves_multi_control_ancillas() and _segment_may_reserve_ancillas(
            operations
        ):
            ancilla_demand = self._count_multi_control_ancilla_demand(
                operations, qubit_count, qubit_map, clbit_map, bindings
            )
            if ancilla_demand > 0:
                self._mc_ancilla_pool = MultiControlAncillaPool(
                    qubit_count, ancilla_demand
                )
                qubit_count += ancilla_demand

        circuit = self._emitter.create_circuit(qubit_count, clbit_count)
        self._measurement_qubit_map.clear()

        self._emit_operations(circuit, operations, qubit_map, clbit_map, bindings)
        self._reject_overwritten_live_condition_outputs()

        return circuit, qubit_map, clbit_map

    def _count_multi_control_ancilla_demand(
        self,
        operations: list[Operation],
        data_qubit_count: int,
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> int:
        """Measure the clean-ancilla demand by a count-only dry-run emission.

        Runs the real ``_emit_operations`` walk once against a
        :class:`CountingEmitter` (every gate a no-op, every reusable-gate /
        native path declined) and a counting :class:`MultiControlAncillaPool`
        that records peak concurrent usage instead of a real circuit. Because
        the walk is the same code the real emission runs, the count cannot
        drift from the emitter the way a separate mirror estimate could; the
        capability answers are biased toward the cascade path so the peak is
        an upper bound. All pass and context state the walk mutates is
        snapshotted and restored so the throwaway run leaves no trace.

        Args:
            operations (list[Operation]): The quantum segment's operations.
            data_qubit_count (int): Number of data qubits already allocated;
                the counting pool's indices start here (their exact values
                are irrelevant, only the peak count matters).
            qubit_map (QubitMap): Data-qubit map; copied for the dry run.
            clbit_map (ClbitMap): Classical-bit map; copied for the dry run.
            bindings (dict[str, Any]): Emit bindings; mutations are rolled
                back after counting.

        Returns:
            int: The peak number of clean ancilla qubits a real emission of
                ``operations`` would hold at once.
        """
        saved_emitter = self._emitter
        saved_pool = self._mc_ancilla_pool
        saved_composites = self._composite_emitters
        saved_params = dict(self._parameter_map)
        saved_sources = dict(self._parameter_sources)
        saved_measure = dict(self._measurement_qubit_map)
        saved_overwritten_conditions = set(self._overwritten_runtime_condition_sources)
        saved_context = (
            bindings.snapshot_state()
            if isinstance(bindings, EmitContext)
            else dict(bindings)
        )
        counting_pool = MultiControlAncillaPool(data_qubit_count, 0, counting=True)
        self._emitter = cast(GateEmitter[T], CountingEmitter(saved_emitter))
        self._mc_ancilla_pool = counting_pool
        self._counting_emission = True
        # Native composite emitters would construct real circuit objects on
        # the dummy circuit; disable them so counting stays on the library
        # decomposition path (which reserves the same or more ancillas).
        self._composite_emitters = []
        try:
            dummy = self._emitter.create_circuit(0, 0)
            self._emit_operations(
                dummy, operations, dict(qubit_map), dict(clbit_map), bindings
            )
        finally:
            self._emitter = saved_emitter
            self._mc_ancilla_pool = saved_pool
            self._composite_emitters = saved_composites
            self._counting_emission = False
            self._parameter_map.clear()
            self._parameter_map.update(saved_params)
            self._parameter_sources.clear()
            self._parameter_sources.update(saved_sources)
            self._measurement_qubit_map.clear()
            self._measurement_qubit_map.update(saved_measure)
            self._overwritten_runtime_condition_sources.clear()
            self._overwritten_runtime_condition_sources.update(
                saved_overwritten_conditions
            )
            if isinstance(bindings, EmitContext):
                bindings.restore_state(saved_context)
            else:
                bindings.clear()
                bindings.update(saved_context)
        return counting_pool.peak

    def _emit_operations(
        self,
        circuit: T,
        operations: list[Operation],
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
        force_unroll: bool = False,
        emit_qinit_reset: bool = False,
    ) -> None:
        """Dispatch an operation list into a backend circuit.

        Args:
            circuit (T): Backend circuit being mutated.
            operations (list[Operation]): Operations in program order.
            qubit_map (QubitMap): Current logical-to-physical qubit map.
            clbit_map (ClbitMap): Current logical-to-physical clbit map.
            bindings (dict[str, Any]): Active compile-time and loop-local
                bindings.
            force_unroll (bool): Whether range loops must use emit-time
                unrolling. Defaults to False.
            emit_qinit_reset (bool): Whether nested allocations should reset
                their persistent backend wires at each body entry. Defaults to
                False.

        Raises:
            EmitError: If an operation or control-flow shape cannot be emitted
                safely by the backend.
            RuntimeError: If slice-lifetime markers or another invalid
                pipeline-stage artifact reaches this dispatcher.
            ValueError: If required loop values cannot be resolved.
        """
        for op in operations:
            if isinstance(op, QInitOperation):
                if emit_qinit_reset:
                    self._emit_qinit_reset(circuit, op, qubit_map, bindings)
                continue
            elif isinstance(op, (SliceArrayOperation, ReleaseSliceViewOperation)):
                # SliceArrayOperation / ReleaseSliceViewOperation are
                # intentionally preserved through partial_eval /
                # constant_fold so SliceBorrowCheckPass can validate
                # view borrow / release post-fold; StripSliceArrayOpsPass
                # (invoked from ``Transpiler.strip_slice_ops`` after the
                # linearity check) is responsible for removing both
                # before this point.  Reaching here means the strip
                # stage was skipped or ran out of order — a
                # compiler-internal invariant violation.  Fail loudly
                # rather than silently emitting nothing.
                raise RuntimeError(
                    f"{type(op).__name__} reached emit — "
                    f"StripSliceArrayOpsPass should have stripped it "
                    f"after SliceBorrowCheckPass.  This is a "
                    f"compiler bug; please report it."
                )
            elif isinstance(op, GateOperation):
                emit_gate(self, circuit, op, qubit_map, bindings)
            elif isinstance(op, MeasureOperation):
                emit_measure(self, circuit, op, qubit_map, clbit_map, bindings)
            elif isinstance(op, ProjectOperation):
                self._emit_project(circuit, op, qubit_map, clbit_map, bindings)
            elif isinstance(op, ResetOperation):
                self._emit_reset(circuit, op, qubit_map, bindings)
            elif isinstance(op, MeasureVectorOperation):
                emit_measure_vector(self, circuit, op, qubit_map, clbit_map, bindings)
            elif isinstance(op, MeasureQFixedOperation):
                emit_measure_qfixed(self, circuit, op, qubit_map, clbit_map)
            elif isinstance(op, ForOperation):
                self._emit_for(
                    circuit, op, qubit_map, clbit_map, bindings, force_unroll
                )
            elif isinstance(op, ForItemsOperation):
                emit_for_items(self, circuit, op, qubit_map, clbit_map, bindings)
            elif isinstance(op, IfOperation):
                reject_stale_runtime_condition(
                    self,
                    op.condition,
                    clbit_map,
                    bindings,
                )
                bound_qubit_map, bound_clbit_map = (
                    self._allocator.resolve_iteration_maps(
                        [op],
                        qubit_map,
                        clbit_map,
                        bindings,
                    )
                )
                self._emit_if(
                    circuit,
                    op,
                    bound_qubit_map,
                    bound_clbit_map,
                    bindings,
                )
                qubit_map.update(bound_qubit_map)
                clbit_map.update(bound_clbit_map)
            elif isinstance(op, WhileOperation):
                condition = op.operands[0] if op.operands else None
                condition_value = (
                    condition.value if hasattr(condition, "value") else condition
                )
                if isinstance(condition_value, Value):
                    reject_stale_runtime_condition(
                        self,
                        condition_value,
                        clbit_map,
                        bindings,
                    )
                bound_qubit_map, bound_clbit_map = (
                    self._allocator.resolve_iteration_maps(
                        [op],
                        qubit_map,
                        clbit_map,
                        bindings,
                    )
                )
                self._emit_while(
                    circuit,
                    op,
                    bound_qubit_map,
                    bound_clbit_map,
                    bindings,
                )
                mark_updated_while_condition(
                    self,
                    op,
                    bound_clbit_map,
                    bindings,
                )
                qubit_map.update(bound_qubit_map)
                clbit_map.update(bound_clbit_map)
            elif isinstance(op, InvokeOperation):
                emit_invoke_operation(self, circuit, op, qubit_map, bindings)
            elif isinstance(op, InverseBlockOperation):
                self._emit_inverse_block(circuit, op, qubit_map, bindings)
            elif isinstance(op, ControlledUOperation):
                emit_controlled_u(self, circuit, op, qubit_map, bindings)
            elif isinstance(op, PauliEvolveOp):
                self._emit_pauli_evolve(circuit, op, qubit_map, bindings)
            elif isinstance(op, StoreArrayElementOperation):
                # Classical element stores execute host-side in classical
                # segments (ClassicalExecutor) or fold at compile time.
                # One reaching a quantum segment means the stored contents
                # feed a quantum op without being compile-time resolvable;
                # silently skipping it would emit stale gate parameters.
                from qamomile.circuit.transpiler.errors import EmitError

                raise EmitError(
                    f"Classical array element store into "
                    f"'{op.array.name or 'array'}' reached the quantum "
                    f"segment. Stored elements consumed by quantum gates "
                    f"must be compile-time resolvable: bind the array and "
                    f"the stored value via `bindings` instead of "
                    f"`parameters`, or restructure the kernel so the "
                    f"stored elements are not used as gate parameters."
                )
            elif isinstance(op, CastOperation):
                handle_cast(self, op, qubit_map)
            elif isinstance(op, BinOp):
                evaluate_binop(self, op, bindings)
            elif isinstance(op, DictGetItemOperation):
                evaluate_dict_getitem(self, op, bindings)
            elif isinstance(op, RuntimeClassicalExpr):
                # Pre-emit ``ClassicalLoweringPass`` already identified this
                # op as runtime-only. Hand off to the backend hook directly;
                # no fold attempt — the IR has already declared the verdict.
                self._emit_runtime_classical_expr(circuit, op, clbit_map, bindings)
            elif isinstance(op, (CompOp, CondOp, NotOp)):
                evaluate_classical_predicate(self, op, bindings)
                # If the predicate could not be folded at compile time (e.g.
                # operands are runtime measurement bits), give the backend a
                # chance to build a runtime expression that downstream
                # if/while emission can consume as a classical condition.
                # Note: post-``ClassicalLoweringPass``, most measurement-
                # derived predicates have already been rewritten to
                # ``RuntimeClassicalExpr`` (handled above). This fallback
                # remains for predicates that depend on emit-time-bound
                # values not visible to the lowering pass (e.g. computed
                # from a loop variable that wraps a measurement somehow).
                if op.results and op.results[0].uuid not in bindings:
                    runtime_expr = self._build_runtime_predicate_expr(
                        circuit, op, clbit_map, bindings
                    )
                    if runtime_expr is not None:
                        set_runtime_expr = getattr(bindings, "set_runtime_expr", None)
                        if callable(set_runtime_expr):
                            set_runtime_expr(op.results[0].uuid, runtime_expr)
                        else:
                            bindings[op.results[0].uuid] = runtime_expr
            elif isinstance(op, HasNestedOps):
                raise NotImplementedError(
                    f"Unhandled control flow: {type(op).__name__}"
                )

    def _emit_qinit_reset(
        self,
        circuit: T,
        op: QInitOperation,
        qubit_map: QubitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Emit prepare-zero operations for nested fresh allocations.

        Args:
            circuit (T): Backend circuit currently being emitted.
            op (QInitOperation): Fresh logical allocation to prepare.
            qubit_map (QubitMap): Current logical-to-physical qubit map.
            bindings (dict[str, Any]): Emit-time bindings for array sizes.

        Raises:
            EmitError: If a symbolic array allocation size cannot be
                resolved during emit.
        """
        from qamomile.circuit.ir.value import ArrayValue
        from qamomile.circuit.transpiler.errors import EmitError

        result = op.results[0]
        if isinstance(result, ArrayValue):
            if not result.shape:
                return
            size = self._resolver.resolve_int_value(result.shape[0], bindings)
            if size is None:
                raise EmitError(
                    "Cannot emit nested fresh allocation for a symbolic-size "
                    "qubit array. Bind the array size before transpilation.",
                    operation="QInitOperation",
                )
            for index in range(size):
                qubit = qubit_map[QubitAddress(result.uuid, index)]
                self._checked_emit_reset(circuit, qubit, "QInitOperation")
            return
        qubit = qubit_map[QubitAddress(result.uuid)]
        self._checked_emit_reset(circuit, qubit, "QInitOperation")

    def _resolve_qubit_operand(
        self,
        qubit_val: Value,
        qubit_map: QubitMap,
        bindings: dict[str, Any],
        operation: str,
    ) -> int:
        """Resolve a single qubit operand to its physical qubit index.

        Uses the full resolver (which handles array-element qubits with
        composite keys, e.g. ``q[i]`` after loop unrolling) with a fallback
        to a direct scalar-UUID lookup. This is the shared resolution path
        for reset / projection so they behave identically to
        ``emit_measure`` — a plain ``qubit_map[QubitAddress(uuid)]`` misses
        array elements, whose addresses are composite ``(root_uuid, index)``
        keys rather than the element's own UUID.

        Args:
            qubit_val (Value): The qubit operand to resolve.
            qubit_map (QubitMap): Current logical-to-physical qubit map.
            bindings (dict[str, Any]): Emit-time bindings for index/size
                resolution.
            operation (str): Operation name for the error message.

        Returns:
            int: The physical qubit index.

        Raises:
            EmitError: If the operand cannot be resolved to a physical
                qubit index (e.g. an unresolved native-loop index that
                should have forced unrolling, or a missing allocation).
        """
        from qamomile.circuit.transpiler.errors import EmitError

        index = self._resolver.resolve_qubit_index(qubit_val, qubit_map, bindings)
        if index is None:
            scalar_addr = QubitAddress(qubit_val.uuid)
            if scalar_addr in qubit_map:
                index = qubit_map[scalar_addr]
        if index is None:
            raise EmitError(
                f"{operation} could not resolve qubit "
                f"'{qubit_val.name}' (uuid: {qubit_val.uuid[:8]}...) to a "
                f"physical qubit index. A qubit indexed by a native-loop "
                f"variable (e.g. `q[i]`) must be unrolled before emit; if you "
                f"reached this error the loop was not unrolled or the qubit "
                f"was never allocated.",
                operation=operation,
            )
        return index

    def _emit_project(
        self,
        circuit: T,
        op: ProjectOperation,
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Emit a projective measurement that keeps the projected qubit.

        A ``project_z`` is emitted as a measurement whose qubit survives (the
        handle is rebound to the projected state). The qubit operand is
        resolved through ``_resolve_qubit_operand`` so array-element qubits
        (``project_z(q[i])`` after loop unrolling) resolve correctly, matching
        ``emit_measure``.

        Args:
            circuit (T): Backend circuit currently being emitted.
            op (ProjectOperation): The projection operation to emit. Must have
                already lowered ``project_x`` / ``project_y`` to ``project_z``
                plus basis-change gates.
            qubit_map (QubitMap): Current logical-to-physical qubit map.
            clbit_map (ClbitMap): Current classical-bit address map.
            bindings (dict[str, Any]): Emit-time bindings for index/size
                resolution.

        Raises:
            EmitError: If ``op.axis`` is not ``"z"`` (an unlowered
                ``project_x`` / ``project_y`` reached emit), if the qubit
                operand cannot be resolved to a physical index, or if the
                result classical bit was never allocated.
        """
        from qamomile.circuit.transpiler.errors import EmitError
        from qamomile.circuit.transpiler.gate_emitter import MeasurementMode

        if op.axis != "z":
            raise EmitError(
                f"ProjectOperation axis {op.axis!r} reached emit; only "
                "project_z should appear because project_x/project_y lower "
                "through basis-change gates.",
                operation="ProjectOperation",
            )
        qubit = self._resolve_qubit_operand(
            op.operands[0], qubit_map, bindings, "ProjectOperation"
        )
        clbit_addr = QubitAddress(op.results[1].uuid)
        if clbit_addr not in clbit_map:
            raise EmitError(
                f"ProjectOperation could not resolve its result classical bit "
                f"(uuid: {op.results[1].uuid[:8]}...): it was never allocated "
                f"in the clbit map.",
                operation="ProjectOperation",
            )
        bit = clbit_map[clbit_addr]
        self._emitter.emit_measure(circuit, qubit, bit)
        if self._emitter.measurement_mode == MeasurementMode.STATIC:
            self._measurement_qubit_map[bit] = qubit

    def _emit_reset(
        self,
        circuit: T,
        op: ResetOperation,
        qubit_map: QubitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Emit a reset of a qubit to the |0> state.

        The qubit operand is resolved through ``_resolve_qubit_operand`` so
        array-element qubits (``reset(q[i])`` after loop unrolling) resolve
        correctly.

        Args:
            circuit (T): Backend circuit currently being emitted.
            op (ResetOperation): The reset operation to emit.
            qubit_map (QubitMap): Current logical-to-physical qubit map.
            bindings (dict[str, Any]): Emit-time bindings for index/size
                resolution.

        Raises:
            EmitError: If the qubit operand cannot be resolved to a physical
                qubit index (e.g. an unresolved native-loop index that should
                have forced unrolling, or a missing allocation).
        """
        qubit = self._resolve_qubit_operand(
            op.operands[0], qubit_map, bindings, "ResetOperation"
        )
        self._checked_emit_reset(circuit, qubit, "ResetOperation")

    def _checked_emit_reset(self, circuit: T, qubit: int, operation: str) -> None:
        """Emit a reset, converting backend refusal into an ``EmitError``.

        ``GateEmitter.emit_reset`` raises ``NotImplementedError`` on backends
        with no reset primitive (e.g. QURI Parts). Letting that raw Python
        exception escape a normal qkernel compile is a UX bug, so every reset
        emission funnels through here and surfaces an actionable compile
        error instead.

        Args:
            circuit (T): Backend circuit currently being emitted.
            qubit (int): Physical qubit index to reset.
            operation (str): Operation label for the error message
                (``"ResetOperation"`` or ``"QInitOperation"``).

        Raises:
            EmitError: If the backend emitter does not support reset. The
                message tells the user to use a reset-capable backend or
                avoid ``qmc.reset`` / fresh in-loop allocation on this one.
        """
        from qamomile.circuit.transpiler.errors import EmitError

        try:
            self._emitter.emit_reset(circuit, qubit)
        except NotImplementedError as e:
            raise EmitError(
                "This backend cannot emit a qubit reset. "
                "`qmc.reset(...)` and fresh qubit allocation inside a "
                "runtime loop (which requires a per-iteration reset) need a "
                "backend with a native reset primitive, e.g. Qiskit or "
                "CUDA-Q. Either switch backend or restructure the kernel to "
                "avoid reset on this one.",
                operation=operation,
            ) from e

    # ------------------------------------------------------------------
    # Methods overridden by backend subclasses (QiskitEmitPass, CudaqEmitPass).
    # Only methods with actual overrides are kept as instance methods.
    # ------------------------------------------------------------------

    def _emit_for(
        self,
        circuit: T,
        op: ForOperation,
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
        force_unroll: bool = False,
    ) -> None:
        emit_for(self, circuit, op, qubit_map, clbit_map, bindings, force_unroll)

    def _emit_for_unrolled(
        self,
        circuit: T,
        op: ForOperation,
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> None:
        from qamomile.circuit.transpiler.passes.emit_support.control_flow_emission import (
            emit_for_unrolled,
        )

        emit_for_unrolled(self, circuit, op, qubit_map, clbit_map, bindings)

    def _emit_if(
        self,
        circuit: T,
        op: IfOperation,
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> None:
        emit_if(self, circuit, op, qubit_map, clbit_map, bindings)

    def _emit_while(
        self,
        circuit: T,
        op: WhileOperation,
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> None:
        emit_while(self, circuit, op, qubit_map, clbit_map, bindings)

    def _emit_pauli_evolve(
        self,
        circuit: T,
        op: PauliEvolveOp,
        qubit_map: QubitMap,
        bindings: dict[str, Any],
    ) -> None:
        emit_pauli_evolve(self, circuit, op, qubit_map, bindings)

    def _emit_inverse_block(
        self,
        circuit: T,
        op: InverseBlockOperation,
        qubit_map: QubitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Emit a first-class inverse block operation.

        Args:
            circuit (T): Backend circuit being built.
            op (InverseBlockOperation): Inverse block operation to emit.
            qubit_map (QubitMap): Current quantum value to physical qubit map.
            bindings (dict[str, Any]): Active emit bindings.

        Raises:
            EmitError: If neither backend-native inverse emission nor the
                fallback implementation can be emitted.
        """
        emit_inverse_block(self, circuit, op, qubit_map, bindings)

    def _emit_runtime_classical_expr(
        self,
        circuit: T,
        op: RuntimeClassicalExpr,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Backend hook: lower ``RuntimeClassicalExpr`` to a backend expr.

        The default implementation raises ``EmitError``. A backend that
        supports runtime classical expressions (e.g. Qiskit 2.x with
        ``qiskit.circuit.classical.expr``) overrides this to translate the
        IR op to its native expression type and store it in
        ``bindings`` (preferably via ``EmitContext.set_runtime_expr``)
        keyed by the op's result UUID, so that ``_emit_if`` /
        ``_emit_while`` can consume it.

        Args:
            circuit: The backend circuit being built.
            op: The runtime classical expression to lower.
            clbit_map: Map from ``QubitAddress`` → physical clbit index.
            bindings: Current bindings; the result should be written here.

        Raises:
            EmitError: If the backend does not support runtime classical
                expressions.
        """
        from qamomile.circuit.transpiler.errors import EmitError

        raise EmitError(
            f"Backend {type(self).__name__!r} does not support runtime "
            f"classical expressions (RuntimeClassicalExpr). The IR contains "
            f"a measurement-derived classical op (kind={op.kind}) that "
            f"cannot be folded at compile time. Either bind the upstream "
            f"parameters to compile-time constants or use a backend with "
            f"dynamic-circuit support."
        )

    def _build_runtime_predicate_expr(
        self,
        circuit: T,
        op: "CompOp | CondOp | NotOp",
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> Any:
        """Build a backend-specific runtime expression for a classical predicate.

        Hook for backends that support classical-bit-level expressions in
        ``if`` / ``while`` conditions (e.g. Qiskit's
        ``qiskit.circuit.classical.expr``). When a ``CompOp`` / ``CondOp`` /
        ``NotOp`` cannot be folded at compile time because its operands are
        runtime measurement bits, the emit dispatch calls this hook to give
        the backend a chance to express the predicate as a clbit-level
        expression. The returned object is stored in ``bindings`` keyed by
        the op's result UUID and later consumed by ``_emit_if`` /
        ``_emit_while``.

        Args:
            circuit: The backend circuit being built (for clbit lookups).
            op: The unresolved classical predicate.
            clbit_map: Map from ``QubitAddress`` → physical clbit index.
            bindings: Current bindings (read-only here; mutation happens at
                the caller).

        Returns:
            A backend-native expression object, or ``None`` if the backend
            does not support runtime classical predicates (default for the
            base class).
        """
        return None

    def _emit_controlled_fallback(
        self,
        circuit: T,
        block_value: Any,
        num_controls: int,
        control_indices: list[int],
        target_indices: list[int],
        power: int,
        bindings: dict[str, Any],
    ) -> None:
        emit_controlled_fallback(
            self,
            circuit,
            block_value,
            num_controls,
            control_indices,
            target_indices,
            power,
            bindings,
        )

    def _reserves_multi_control_ancillas(self) -> bool:
        """Backend hook: opt in to the shared multi-controlled lowering.

        When True, ``_emit_quantum_segment`` statically estimates the
        clean-ancilla demand of the segment's multi-controlled gates,
        appends that many extra qubits to the circuit, and the base
        ``_emit_irreducible_multi_controlled_gate`` lowers irreducible
        multi-controlled gates through the shared Toffoli-cascade
        decomposition (arXiv:2307.07478, Appendix A.3) on those
        ancillas.

        Backends with a native multi-controlled primitive (e.g. Qiskit's
        ``gate.control(k)``, CUDA-Q's ``ctrl`` variants) keep the
        default False so their circuits carry no unused ancilla qubits.
        A new backend without native multi-control support should
        override this to return True instead of implementing its own
        ``_emit_irreducible_multi_controlled_gate``.

        Returns:
            bool: False in the base implementation.
        """
        return False

    @contextlib.contextmanager
    def _suspended_mc_ancilla_pool(self) -> Iterator[None]:
        """Suspend the segment's multi-control ancilla pool during nested emission.

        The reserved pool's indices are physical addresses in the *parent*
        segment's circuit (they sit past that circuit's data qubits). Any
        helper that emits a block into an independent sub-circuit — the
        reusable-gate probe (``blockvalue_to_gate``) or a backend-native
        inverse (``_try_emit_backend_inverse``) — MUST run its sub-circuit
        emission inside this context so that an irreducible
        multi-controlled gate in the block does not index the parent pool.
        Reusing a parent index in the narrower sub-circuit would either
        exceed its width or silently alias one of its data qubits.

        With the pool suspended the base
        ``_emit_irreducible_multi_controlled_gate`` finds no pool and
        raises ``EmitError``; the sub-circuit helpers catch it and fall
        back to gate-by-gate emission on the parent circuit, where the
        pool and the composed control set are both valid. Backends that do
        not reserve a pool (native multi-control) already hold ``None``,
        so the suspension is a no-op for them.

        Yields:
            None: Runs the ``with`` body with ``_mc_ancilla_pool`` cleared;
                the previous pool is restored on exit, including on error.
        """
        saved = self._mc_ancilla_pool
        self._mc_ancilla_pool = None
        try:
            yield
        finally:
            self._mc_ancilla_pool = saved

    def _emit_irreducible_multi_controlled_gate(
        self,
        circuit: T,
        gate_type: GateOperationType,
        control_indices: list[int],
        target_idx: int,
        angle: Any,
    ) -> None:
        """Emit one irreducible multi-controlled gate.

        The shared controlled fallback reduces multi-qubit gate types
        structurally and covers up to two controls on X / Z via
        Toffoli. Single-qubit gates that still carry two or more
        controls after those reductions land here.

        When the backend reserved a clean-ancilla pool (see
        ``_reserves_multi_control_ancillas``), the gate is lowered
        through the shared Toffoli-cascade decomposition. Otherwise the
        backend is expected to have native multi-control support and
        never reach this hook; reaching it raises a descriptive error.

        Args:
            circuit (T): Backend circuit being built.
            gate_type (GateOperationType): The single-qubit gate to
                control.
            control_indices (list[int]): Physical control qubits.
            target_idx (int): Physical target qubit.
            angle (Any): Resolved rotation angle for rotation-like
                gates, or ``None`` for fixed gates.

        Raises:
            EmitError: If no ancilla pool was reserved for this segment,
                or the pool is smaller than ``len(control_indices) - 1``
                (a bug in ``_count_multi_control_ancilla_demand``).
        """
        from qamomile.circuit.transpiler.errors import EmitError
        from qamomile.circuit.transpiler.passes.emit_support.controlled_emission import (
            emit_multi_controlled_on_clean_ancillas,
        )

        if self._mc_ancilla_pool is None:
            raise EmitError(
                f"Cannot emit {len(control_indices)}-controlled "
                f"{gate_type.name}: the shared fallback reduces to Toffoli "
                f"for up to two controls only, and backend "
                f"{type(self).__name__!r} neither reserves ancillas for the "
                f"shared Toffoli-cascade decomposition "
                f"(``_reserves_multi_control_ancillas``) nor overrides "
                f"``_emit_irreducible_multi_controlled_gate``. Run this "
                f"kernel on a backend with native multi-control support, or "
                f"enable the shared decomposition in the backend's emit "
                f"pass.",
                operation="ControlledGate",
            )

        ancillas = self._mc_ancilla_pool.take(len(control_indices) - 1)
        if ancillas is None:
            raise EmitError(
                f"Multi-controlled {gate_type.name} over "
                f"{len(control_indices)} controls needs "
                f"{len(control_indices) - 1} clean ancilla qubit(s), but "
                f"only {self._mc_ancilla_pool.count} were reserved for this "
                f"segment. This means the count-only demand walk "
                f"(``_count_multi_control_ancilla_demand``) under-measured "
                f"the segment's demand — a compiler bug; please report it.",
                operation="ControlledGate",
            )
        emit_multi_controlled_on_clean_ancillas(
            self, circuit, gate_type, control_indices, target_idx, angle, ancillas
        )

    def _blockvalue_to_gate(
        self,
        block_value: Any,
        num_qubits: int,
        bindings: dict[str, Any],
        input_operands: list[Any] | None = None,
        operation_name: str = "ControlledUOperation",
    ) -> Any:
        """Convert a nested block into a reusable backend gate.

        Args:
            block_value (Any): Block-like object to emit into a temporary
                backend circuit.
            num_qubits (int): Number of qubits in the temporary circuit.
            bindings (dict[str, Any]): Active emit bindings.
            input_operands (list[Any] | None): Optional call-site operands
                used to bind block inputs. Defaults to None.
            operation_name (str): Operation name used in diagnostics when
                input binding fails. Defaults to ``"ControlledUOperation"``.

        Returns:
            Any: Backend gate object, or None when conversion fails.
        """
        return blockvalue_to_gate(
            self,
            block_value,
            num_qubits,
            bindings,
            input_operands=input_operands,
            operation_name=operation_name,
        )

    # ------------------------------------------------------------------
    # Helpers called from emit_support modules
    # ------------------------------------------------------------------

    def _resolve_angle(self, op: GateOperation, bindings: dict[str, Any]) -> Any:
        from qamomile.circuit.transpiler.passes.emit_support.gate_emission import (
            resolve_angle,
        )

        return resolve_angle(self, op, bindings)

    def _emit_custom_composite(
        self,
        circuit: T,
        op: Any,
        impl: Any,
        qubit_indices: list[int],
        bindings: dict[str, Any],
    ) -> None:
        from qamomile.circuit.transpiler.passes.emit_support.controlled_emission import (
            emit_custom_composite,
        )

        emit_custom_composite(self, circuit, op, impl, qubit_indices, bindings)

    def _emit_controlled_powers(
        self,
        circuit: T,
        block_value: Any,
        counting_indices: list[int],
        target_indices: list[int],
        bindings: dict[str, Any],
    ) -> None:
        from qamomile.circuit.transpiler.passes.emit_support.controlled_emission import (
            emit_controlled_powers,
        )

        emit_controlled_powers(
            self, circuit, block_value, counting_indices, target_indices, bindings
        )
