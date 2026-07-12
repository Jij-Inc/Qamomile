"""Segmentation pass: Split a block into executable program steps."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import RuntimeClassicalExpr
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.classical_ops import DecodeQFixedOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    IfOperation,
    WhileOperation,
    genuine_input_values,
)
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.operation.gate import (
    MeasureQFixedOperation,
    MeasureVectorOperation,
)
from qamomile.circuit.ir.operation.operation import OperationKind
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.types.primitives import BitType, QubitType, UIntType
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
    array_physical_region,
    arrays_share_physical_region,
)
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.passes.analyze import (
    build_dependency_graph,
    find_measurement_derived_values,
    find_measurement_results,
)
from qamomile.circuit.transpiler.passes.control_flow_visitor import ControlFlowVisitor
from qamomile.circuit.transpiler.passes.validate_while import ValidateWhileContractPass
from qamomile.circuit.transpiler.segments import (
    ClassicalSegment,
    ClassicalStep,
    ExpvalSegment,
    ExpvalStep,
    HybridBoundary,
    MultipleQuantumSegmentsError,
    ProgramABI,
    ProgramPlan,
    QuantumSegment,
    QuantumStep,
    Segment,
)
from qamomile.circuit.transpiler.value_resolver import ValueResolver

# =========================================================================
# Pre-segmentation transformations (called by SegmentationPass.run)
# =========================================================================


def materialize_return(block: Block) -> Block:
    """Synchronize output_values from ReturnOperation.

    Uses ReturnOperation.operands as the source of truth for return values,
    ensuring consistency between the operation stream and block metadata.
    """
    for op in reversed(block.operations):
        if isinstance(op, ReturnOperation):
            return dataclasses.replace(block, output_values=list(op.operands))
    return block


def _collect_compile_time_outputs(output_values: list[Value]) -> dict[str, object]:
    """Resolve outputs already materialized in IR metadata.

    Args:
        output_values (list[Value]): Block outputs in public return order.

    Returns:
        dict[str, object]: Output UUIDs mapped to constant scalar or array
            values. Structural array elements resolve through their parent
            array's compile-time contents.
    """
    resolver = ValueResolver()
    resolved_outputs: dict[str, object] = {}
    for value in output_values:
        resolved = resolver.resolve(value)
        if resolved is None and isinstance(value, ArrayValue):
            resolved = value.get_const_array()
        if resolved is not None:
            resolved_outputs[value.uuid] = resolved
    return resolved_outputs


def lower_measure_qfixed(
    op: MeasureQFixedOperation,
    cast_source: ArrayValue | None = None,
) -> list[Operation]:
    """Lower MeasureQFixedOperation to MeasureVectorOperation + decode.

    Args:
        op (MeasureQFixedOperation): QFixed measurement to lower.
        cast_source (ArrayValue | None): Source vector from the preceding
            ``CastOperation`` when the QFixed carries a deferred, non-enumerated
            vector alias. Defaults to ``None`` for concrete carrier lists.

    Returns:
        list[Operation]: Operations ``[MeasureVectorOperation,
        DecodeQFixedOperation]``.
    """
    qfixed = op.operands[0]

    qubit_uuids = qfixed.get_cast_qubit_uuids() or qfixed.get_qfixed_qubit_uuids()
    num_bits = op.num_bits or len(qubit_uuids)
    int_bits = op.int_bits

    if qubit_uuids:
        size_value = Value(
            type=UIntType(),
            name="qfixed_size",
        ).with_const(num_bits)

        qubits_array = ArrayValue(
            type=QubitType(),
            name="qfixed_qubits",
            shape=(size_value,),
        ).with_array_runtime_metadata(
            element_uuids=qubit_uuids,
            element_logical_ids=qfixed.get_cast_qubit_logical_ids() or (),
        )
        cast_source_uuid = qfixed.get_cast_source_uuid()
        if cast_source_uuid:
            qubits_array = qubits_array.with_cast_metadata(
                source_uuid=cast_source_uuid,
                source_logical_id=qfixed.get_cast_source_logical_id(),
                qubit_uuids=qubit_uuids,
                qubit_logical_ids=qfixed.get_cast_qubit_logical_ids() or (),
            )
    elif cast_source is not None and cast_source.shape:
        qubits_array = cast_source
        size_value = cast_source.shape[0]
        if num_bits == 0 and size_value.is_constant():
            const_bits = size_value.get_const()
            if const_bits is not None:
                num_bits = int(const_bits)
    else:
        size_value = Value(
            type=UIntType(),
            name="qfixed_size",
        ).with_const(num_bits)
        qubits_array = ArrayValue(
            type=QubitType(),
            name="qfixed_qubits",
            shape=(size_value,),
        )

    bits_array = ArrayValue(
        type=BitType(),
        name="qfixed_bits",
        shape=(size_value,),
    )

    measure_vec_op = MeasureVectorOperation(
        operands=[qubits_array],
        results=[bits_array],
    )

    decode_op = DecodeQFixedOperation(
        num_bits=num_bits,
        int_bits=int_bits,
        operands=[bits_array],
        results=list(op.results),
    )

    return [measure_vec_op, decode_op]


def lower_operations(block: Block) -> Block:
    """Lower high-level operations like MeasureQFixedOperation.

    MeasureQFixedOperation is lowered to:
    1. MeasureVectorOperation for each qubit (HYBRID -> QUANTUM segment)
    2. DecodeQFixedOperation to convert bits to float (CLASSICAL segment)
    """
    lowered_ops: list[Operation] = []
    qfixed_cast_sources: dict[str, ArrayValue] = {}
    for op in block.operations:
        if isinstance(op, CastOperation):
            if op.results and isinstance(op.operands[0], ArrayValue):
                qfixed_cast_sources[op.results[0].uuid] = op.operands[0]
            lowered_ops.append(op)
        elif isinstance(op, MeasureQFixedOperation):
            cast_source = qfixed_cast_sources.get(op.operands[0].uuid)
            lowered_ops.extend(lower_measure_qfixed(op, cast_source))
        else:
            lowered_ops.append(op)
    return dataclasses.replace(block, operations=lowered_ops)


# =========================================================================
# Segmentation strategy and pass
# =========================================================================


class SegmentationStrategy(ABC):
    """Execution-model-specific strategy for building a ProgramPlan."""

    @abstractmethod
    def create_plan(
        self,
        segments: list[Segment],
        block: Block,
        boundaries: list[HybridBoundary],
    ) -> ProgramPlan:
        """Build a program plan from segmented operations.

        Args:
            segments (list[Segment]): Ordered execution segments.
            block (Block): Lowered source block carrying the public ABI.
            boundaries (list[HybridBoundary]): Quantum/classical boundaries.

        Returns:
            ProgramPlan: Strategy-specific executable plan.

        Raises:
            NotImplementedError: Always; subclasses must implement planning.
        """
        raise NotImplementedError


class NisqSegmentationStrategy(SegmentationStrategy):
    """Single-quantum-segment planning strategy for current Qamomile runtimes."""

    def create_plan(
        self,
        segments: list[Segment],
        block: Block,
        boundaries: list[HybridBoundary],
    ) -> ProgramPlan:
        """Build the current single-quantum-segment NISQ plan.

        Args:
            segments (list[Segment]): Ordered classical, quantum, and expval
                segments.
            block (Block): Lowered block providing parameters and outputs.
            boundaries (list[HybridBoundary]): Recorded hybrid transitions.

        Returns:
            ProgramPlan: Plan with one quantum step and surrounding classical
                or expectation-value steps.

        Raises:
            SeparationError: If no quantum segment exists.
            MultipleQuantumSegmentsError: If more than one quantum segment
                would require unsupported JIT-style execution.
        """
        quantum_segs = [s for s in segments if isinstance(s, QuantumSegment)]
        if len(quantum_segs) == 0:
            from qamomile.circuit.transpiler.errors import SeparationError

            raise SeparationError("No quantum segment found")
        if len(quantum_segs) > 1:
            raise MultipleQuantumSegmentsError(
                f"Found {len(quantum_segs)} quantum segments. "
                f"Only single quantum execution is supported. "
                f"This typically happens when quantum operations depend on "
                f"measurement results (JIT compilation not supported)."
            )

        quantum = quantum_segs[0]
        quantum_idx = segments.index(quantum)

        steps: list[ClassicalStep | QuantumStep | ExpvalStep] = []
        for i, segment in enumerate(segments):
            if isinstance(segment, ClassicalSegment):
                role = "prep" if i < quantum_idx else "post"
                steps.append(ClassicalStep(segment=segment, role=role))
            elif isinstance(segment, QuantumSegment):
                steps.append(QuantumStep(segment=segment))
            elif isinstance(segment, ExpvalSegment):
                steps.append(ExpvalStep(segment=segment, quantum_step_index=0))

        abi = ProgramABI(
            public_inputs={name: value for name, value in block.parameters.items()},
            output_refs=[v.uuid for v in block.output_values],
            output_values={v.uuid: v for v in block.output_values},
            constant_outputs=_collect_compile_time_outputs(block.output_values),
        )

        return ProgramPlan(
            steps=steps,
            abi=abi,
            boundaries=boundaries,
            parameters=block.parameters,
        )


class SegmentationPass(Pass[Block, ProgramPlan]):
    """Segment a block into a strategy-specific executable program plan.

    This pass:
    1. Materializes return operations (syncs output_values from ReturnOperation)
    2. Splits the operation list into quantum and classical segments
    3. Builds a ProgramPlan via the configured segmentation strategy

    Input: Block (typically ANALYZED or AFFINE)
    Output: ProgramPlan
    """

    def __init__(
        self,
        strategy: SegmentationStrategy | None = None,
    ) -> None:
        self._strategy = strategy or NisqSegmentationStrategy()

    @property
    def name(self) -> str:
        return "segment"

    def run(self, input: Block) -> ProgramPlan:
        """Lower and segment a block into a program plan.

        Args:
            input (Block): Block whose while contract and hybrid operations
                should be lowered before segmentation.

        Returns:
            ProgramPlan: Strategy-produced execution plan.

        Raises:
            ValidationError: If a runtime while violates its contract.
            SeparationError: If the strategy finds no quantum segment.
            MultipleQuantumSegmentsError: If the strategy finds multiple
                quantum segments.
        """
        input = ValidateWhileContractPass().run(input)
        block = materialize_return(input)
        block = lower_operations(block)
        return self._segment(block)

    def _segment(self, block: Block) -> ProgramPlan:
        """Build a plan from an already-lowered block.

        Args:
            block (Block): Lowered block to split into segments.

        Returns:
            ProgramPlan: Strategy-produced execution plan.

        Raises:
            SeparationError: If the strategy finds no quantum segment.
            MultipleQuantumSegmentsError: If the strategy finds multiple
                quantum segments.
        """
        segments = self._build_segments_list(block)
        return self._strategy.create_plan(segments, block, self._boundaries)

    def _build_segments_list(self, block: Block) -> list[Segment]:
        """Build list of segments from block operations.

        Extracted from _separate_segments to allow reuse.

        Args:
            block (Block): Lowered block whose operations are segmented.

        Returns:
            list[Segment]: Ordered execution segments.

        Raises:
            SeparationError: If an operation cannot be assigned to a valid
                segment shape.
        """
        segments: list[Segment] = []
        self._boundaries: list[HybridBoundary] = []

        # Measurement-taint set: every value transitively derived from a
        # measurement result. Computed with the same dataflow utilities as
        # ``AnalyzePass`` / ``ClassicalLoweringPass`` so the three passes
        # agree on what "measurement-derived" means. Used below to decide
        # whether a classical op interleaved in a quantum region is a
        # genuine post-measurement computation (must split off into its own
        # classical segment) or a parameter / structural expression that
        # merely feeds a quantum gate (can stay in the quantum segment).
        dependency_graph = build_dependency_graph(block.operations)
        self._dependency_graph = dependency_graph
        self._quantum_value_uuids = self._collect_quantum_value_uuids(block.operations)
        measurement_uuids = find_measurement_results(block.operations)
        self._measurement_tainted = find_measurement_derived_values(
            dependency_graph, measurement_uuids
        )

        # Quantum-needed set: every value transitively required to compute
        # the operands of a quantum / hybrid operation. A non-measurement
        # classical op may only be absorbed into a quantum segment when its
        # result feeds a quantum gate (directly or through a chain of other
        # classical expressions). A classical op whose result is *not* needed
        # by any quantum op — e.g. a block output or a value consumed only by
        # downstream classical post-processing — must stay in its own
        # classical segment so the executor actually runs it and the
        # orchestrator can surface its result; absorbing it into a quantum
        # segment would silently drop its value.
        self._quantum_needed = self._compute_quantum_needed(
            block.operations, dependency_graph
        )

        # Block-output set: UUIDs the block returns. A classical op producing
        # one of these must stay in a classical segment even if it also feeds
        # a quantum gate — the executor only runs classical segments, so
        # absorbing such an op would drop its returned value (the orchestrator
        # would surface ``None``). Keeping it classical instead surfaces the
        # value, and if that forces a quantum-segment split the user gets an
        # explicit error rather than a silently wrong result.
        self._block_output_uuids = {
            v.uuid for v in block.output_values if isinstance(v, ValueBase)
        }
        self._block_output_values = tuple(block.output_values)

        # Absorbable set: classical ops (top-level or nested) whose results are
        # consumed exclusively by quantum / hybrid ops, or by other absorbable
        # classical ops that themselves ultimately feed quantum — never by a
        # classical sink (a classical-segment op, a measurement-derived op, or a
        # block output). Only top-level members are actually folded into a
        # segment below; nested members ride inside their control-flow op and
        # are tracked here only so the sink analysis treats a nested chain link
        # correctly. Computing this transitively is what makes a chain like
        # ``(phase * 2) - 1`` feeding a gate fully absorbable while a value read
        # back by later classical work stays in its own classical segment.
        self._absorbable_op_ids = self._compute_absorbable(block.operations)

        current_ops: list[Operation] = []
        current_kind: OperationKind | None = None
        # Parameter-expression classical ops (the gate-angle expressions this
        # work absorbs) that appear before / outside the quantum segment. They
        # are held here and prepended to the quantum segment once it starts, so
        # they end up computed inside the quantum circuit regardless of where
        # the user wrote them — never stranded in a classical prep segment.
        pending_absorbable: list[Operation] = []

        for op in block.operations:
            # Skip ReturnOperation - it's a terminal operation handled separately
            if isinstance(op, ReturnOperation):
                continue

            op_kind = self._effective_kind(op)

            # Handle ExpvalOp specially - it creates its own segment
            if isinstance(op, ExpvalOp):
                # Flush current quantum segment first
                if current_ops:
                    segment = self._create_segment(current_kind, current_ops)
                    segments.append(segment)
                    current_ops = []

                # Create ExpvalSegment
                expval_segment = ExpvalSegment(
                    operations=[op],
                    hamiltonian_value=op.hamiltonian,
                    qubits_value=op.qubits,
                    result_ref=op.output.uuid,
                )
                segments.append(expval_segment)

                # Reset state - next operations start fresh
                current_kind = None
                continue

            if op_kind == OperationKind.HYBRID:
                # HYBRID operations (measurements) need special handling:
                # - They take quantum input and produce classical output
                # - The measurement itself belongs to the quantum segment
                # - We accumulate consecutive measurements in the same segment

                # If we're in classical mode, flush and start quantum
                if current_kind == OperationKind.CLASSICAL and current_ops:
                    segment = self._create_segment(current_kind, current_ops)
                    segments.append(segment)
                    current_ops = []

                # If no current kind, start as quantum
                if current_kind is None:
                    current_kind = OperationKind.QUANTUM

                # Add the measurement to the current (quantum) segment
                # Note: we stay in QUANTUM mode to accumulate consecutive measurements
                if current_kind != OperationKind.QUANTUM:
                    current_kind = OperationKind.QUANTUM
                if pending_absorbable:
                    current_ops.extend(pending_absorbable)
                    pending_absorbable = []
                current_ops.append(op)

                # Create boundary for tracking quantum-classical transition
                boundary = HybridBoundary(
                    operation=op,
                    source_segment_index=len(segments),  # Current segment being built
                    target_segment_index=len(segments) + 1,  # Next segment after flush
                    value_ref=op.results[0].uuid if op.results else "",
                )
                self._boundaries.append(boundary)
                continue

            # Non-measurement parameter / structural classical ops (the
            # gate-angle expressions this segmentation work exists for) belong
            # in the quantum segment regardless of where they appear. Without
            # this, an interleaved op forces a spurious
            # quantum→classical→quantum split (``MultipleQuantumSegmentsError``)
            # and an op written *before* the quantum segment (e.g. ``angle =
            # -phase`` computed before ``qmc.qubit_array``) is stranded in a
            # classical prep segment, where the backend has no gate to attach
            # the parameter expression to and silently emits a zero angle.
            # ``_absorbable_op_ids`` (from :meth:`_compute_absorbable`) holds
            # exactly the ops safe to absorb: non-measurement classical ops
            # whose results flow only to quantum ops (directly or through other
            # absorbable classical ops) and are not block outputs. When already
            # inside the quantum segment we append directly; otherwise we hold
            # the op and prepend it when the quantum segment starts. Ops read
            # back by a classical segment or returned are *not* in the set, so
            # they stay classical (an explicit error rather than a dropped
            # value if that forces a split).
            if op_kind == OperationKind.CLASSICAL and id(op) in self._absorbable_op_ids:
                if current_kind == OperationKind.QUANTUM:
                    current_ops.append(op)
                else:
                    pending_absorbable.append(op)
                continue

            # A non-absorbable, non-measurement classical op whose result is an
            # input to a quantum / hybrid op (a gate angle, or a controlled-gate
            # structural parameter) cannot be placed under the single-quantum-
            # segment model: it is *not* absorbable because it is also read by
            # classical work or returned as a block output, so it must stay in a
            # classical segment — but then the quantum op reads a value the
            # classical segment never threads into the circuit and the backend
            # silently emits a zero/garbage parameter. (When the op sits between
            # quantum ops this also manifests as a spurious extra quantum
            # segment; before the quantum region it produces a legal C→Q plan
            # with a stranded value, so the multi-segment check alone misses
            # it.) Reject explicitly here so the user gets an error instead of a
            # wrong result. Measurement-derived ops are handled separately
            # (``RuntimeClassicalExpr`` / mid-circuit measurement) and excluded.
            if (
                op_kind == OperationKind.CLASSICAL
                and not isinstance(op, RuntimeClassicalExpr)
                and not self._is_measurement_tainted(op)
                and self._feeds_quantum(op)
            ):
                raise MultipleQuantumSegmentsError(
                    "A classical value used as a quantum gate parameter is also "
                    "consumed by classical work or returned as a block output, so "
                    "it cannot be absorbed into the single quantum segment (it "
                    "would otherwise be stranded in a classical segment and the "
                    "gate would silently receive a zero parameter). Compute the "
                    "value so it flows only to quantum gates, or restructure the "
                    "kernel so the classical consumer does not share it."
                )

            if current_kind is None:
                current_kind = op_kind

            # Runtime classical expressions bridge a measurement to a
            # runtime IfOperation/WhileOperation inside a single quantum
            # segment. After ``ClassicalLoweringPass``, every measurement-
            # derived classical op is represented as ``RuntimeClassicalExpr``,
            # so this is a single type-check — no operand-typing heuristic
            # needed. Examples that work under this rule but the old
            # BitType-only heuristic could not handle:
            #   ``if (s0 + 2 * s1 + 4 * s2) == 5:``  (UInt-typed BinOp/CompOp)
            #   ``if measure(q) == bound_uint_param:`` (mixed Bit/UInt)
            if (
                op_kind == OperationKind.CLASSICAL
                and current_kind == OperationKind.QUANTUM
                and isinstance(op, RuntimeClassicalExpr)
            ):
                current_ops.append(op)
                continue

            if op_kind != current_kind and op_kind in (
                OperationKind.QUANTUM,
                OperationKind.CLASSICAL,
            ):
                # Context switch - flush current segment
                if current_ops:
                    segment = self._create_segment(current_kind, current_ops)
                    segments.append(segment)
                    current_ops = []
                current_kind = op_kind

            # Entering (or continuing) the quantum segment: pull in any held
            # parameter-expression ops so they are computed inside the quantum
            # circuit, ahead of the gate that consumes them.
            if current_kind == OperationKind.QUANTUM and pending_absorbable:
                current_ops.extend(pending_absorbable)
                pending_absorbable = []

            current_ops.append(op)

        # Drain any held parameter-expression ops. They are normally emptied
        # when the quantum segment started; a non-empty remainder here means a
        # malformed multi-quantum stream (rejected downstream by the
        # single-quantum-segment strategy), so fold them into the final segment
        # rather than dropping them.
        if pending_absorbable:
            if current_kind is None:
                current_kind = OperationKind.QUANTUM
            current_ops.extend(pending_absorbable)
            pending_absorbable = []

        # Flush final segment
        if current_ops:
            segment = self._create_segment(current_kind, current_ops)
            segments.append(segment)

        self._reject_quantum_segment_carry_escapes(segments)

        # Compute input/output refs for each segment
        self._compute_segment_io(segments, block)

        return segments

    def _collect_carry_results(
        self,
        op: Operation,
        collected: dict[str, str],
    ) -> None:
        """Collect loop-carry result UUIDs from an op tree.

        Args:
            op (Operation): Operation to inspect (control-flow bodies
                are walked recursively).
            collected (dict[str, str]): Mutable map from carry-result
                UUID to the carried variable's display name, updated in
                place.
        """
        if isinstance(op, (ForOperation, ForItemsOperation, WhileOperation)):
            for region_arg in op.region_args:
                collected[region_arg.result.uuid] = region_arg.var_name
        if isinstance(op, HasNestedOps):
            for body in op.nested_op_lists():
                for nested in body:
                    self._collect_carry_results(nested, collected)

    def _segment_read_uuids(self, operations: list[Operation]) -> set[str]:
        """Collect every genuine input-value UUID a segment's op tree reads.

        Cloning/substitution metadata exposed by ``all_input_values`` is not a
        data read. In particular, loop RegionArg formals/results and rebind
        records must not make an overwritten or otherwise unused value appear
        live and trigger an escape/divergent-array rejection.

        Args:
            operations (list[Operation]): Segment operations (control-flow
                bodies are walked recursively).

        Returns:
            set[str]: UUIDs of all referenced input values.
        """
        reads: set[str] = set()
        for op in operations:
            for value in genuine_input_values(op):
                if isinstance(value, ValueBase):
                    reads |= self._value_reference_uuids(value)
            if isinstance(op, HasNestedOps):
                for body in op.nested_op_lists():
                    reads |= self._segment_read_uuids(body)
        return reads

    def _value_reference_uuids(self, value: ValueBase) -> set[str]:
        """Collect a value UUID and UUIDs embedded in its addressing metadata.

        Array elements and views carry dataflow in structural fields rather
        than ordinary operation operands.  In particular, ``values[index]``
        stores ``index`` in ``element_indices``; looking only at the element's
        own UUID would therefore miss an emit-time loop carry used as the
        index of a block output or expectation-value operand.

        Args:
            value (ValueBase): Value whose structural references to walk.

        Returns:
            set[str]: The value's UUID plus every recursively referenced
                parent array, element index, shape, slice-bound, tuple-element,
                and dictionary key/value UUID. A scalar classical value
                carrying compile-time constant metadata omits only its own UUID
                because it needs no runtime producer; any structural references
                attached to it remain included.
        """
        references: set[str] = set()
        visited_objects: set[int] = set()
        pending: list[ValueBase] = [value]
        while pending:
            current = pending.pop()
            identity = id(current)
            if identity in visited_objects:
                continue
            visited_objects.add(identity)
            materialized_scalar_constant = (
                isinstance(current, Value)
                and not isinstance(current, ArrayValue)
                and current.type.is_classical()
                and current.is_constant()
            )
            if not materialized_scalar_constant:
                references.add(current.uuid)
            for field_name in (
                "parent_array",
                "slice_of",
                "slice_start",
                "slice_step",
            ):
                referenced = getattr(current, field_name, None)
                if isinstance(referenced, ValueBase):
                    pending.append(referenced)
            for field_name in ("element_indices", "shape"):
                for referenced in getattr(current, field_name, ()):
                    if isinstance(referenced, ValueBase):
                        pending.append(referenced)
            if isinstance(current, TupleValue):
                pending.extend(current.elements)
            if isinstance(current, DictValue):
                for key, item_value in current.entries:
                    pending.append(key)
                    pending.append(item_value)
        return references

    def _collect_classical_merge_results(
        self,
        op: Operation,
        collected: dict[str, str],
    ) -> None:
        """Collect non-Bit classical merges embedded in a quantum op tree.

        Measurement-backed ``Bit`` merges have a physical clbit and are
        handled by backend aliasing.  Other classical merge results created by
        a runtime ``if`` inside a quantum segment exist only while emitting the
        circuit; the classical executor cannot later surface them.

        Args:
            op (Operation): Operation tree to inspect.
            collected (dict[str, str]): Mutable result-UUID to display-name map,
                updated in place.
        """
        if isinstance(op, IfOperation):
            condition = op.operands[0] if op.operands else None
            condition_uuid = (
                condition.uuid if isinstance(condition, ValueBase) else None
            )
            merges = (
                op.iter_merges() if condition_uuid in self._measurement_tainted else ()
            )
            for merge in merges:
                if not merge.result.type.is_quantum() and not isinstance(
                    merge.result.type, BitType
                ):
                    collected[merge.result.uuid] = merge.result.name or "<anonymous>"
        if isinstance(op, HasNestedOps):
            for body in op.nested_op_lists():
                for nested in body:
                    self._collect_classical_merge_results(nested, collected)

    def _collect_divergent_quantum_array_merges(
        self,
        op: Operation,
        collected: dict[str, str],
    ) -> None:
        """Collect runtime array merges with statically different regions.

        Args:
            op (Operation): Operation tree to inspect recursively.
            collected (dict[str, str]): Mutable result-UUID to display-name
                map, updated in place.
        """
        if isinstance(op, IfOperation):
            condition = op.operands[0] if op.operands else None
            condition_uuid = (
                condition.uuid if isinstance(condition, ValueBase) else None
            )
            merges = (
                op.iter_merges() if condition_uuid in self._measurement_tainted else ()
            )
            for merge in merges:
                result = merge.result
                true_value = merge.true_value
                false_value = merge.false_value
                if not (
                    isinstance(result, ArrayValue)
                    and result.type.is_quantum()
                    and isinstance(true_value, ArrayValue)
                    and isinstance(false_value, ArrayValue)
                ):
                    continue
                true_region = array_physical_region(true_value)
                false_region = array_physical_region(false_value)
                # Symbolic coverage may become concrete at emit time, where
                # the physical map remains the source of truth. Reject here
                # only when both regions are known and demonstrably differ.
                if (
                    true_region is not None
                    and false_region is not None
                    and not arrays_share_physical_region(true_value, false_value)
                ):
                    collected[result.uuid] = result.name or "<anonymous>"
        if isinstance(op, HasNestedOps):
            for body in op.nested_op_lists():
                for nested in body:
                    self._collect_divergent_quantum_array_merges(nested, collected)

    def _collect_quantum_value_uuids(
        self,
        operations: list[Operation],
    ) -> set[str]:
        """Collect quantum result UUIDs from an operation tree.

        Args:
            operations (list[Operation]): Operations to inspect recursively.

        Returns:
            set[str]: UUIDs of quantum scalar and array results.
        """
        quantum_uuids: set[str] = set()
        for op in operations:
            quantum_uuids.update(
                result.uuid for result in op.results if result.type.is_quantum()
            )
            if isinstance(op, HasNestedOps):
                for body in op.nested_op_lists():
                    quantum_uuids.update(self._collect_quantum_value_uuids(body))
        return quantum_uuids

    def _propagate_escape_origins(
        self,
        seeds: dict[str, str],
    ) -> dict[str, set[str]]:
        """Propagate emit-only value origins through the dependency graph.

        Args:
            seeds (dict[str, str]): Seed UUIDs mapped to user-facing variable
                names.

        Returns:
            dict[str, set[str]]: Every transitively dependent UUID mapped to
                the seed names from which it derives.
        """
        dependents: dict[str, set[str]] = {}
        for result_uuid, dependencies in self._dependency_graph.items():
            for dependency_uuid in dependencies:
                dependents.setdefault(dependency_uuid, set()).add(result_uuid)

        origins = {uuid: {name} for uuid, name in seeds.items()}
        pending = list(seeds)
        while pending:
            current = pending.pop()
            current_origins = origins[current]
            for dependent in dependents.get(current, ()):
                # A classical value that affects a gate has a dependency path
                # through the gate's quantum result and eventually its final
                # measurement. That is an ordinary emit-time use, not a value
                # escaping to the classical executor. Stop propagation at the
                # quantum SSA boundary while retaining purely classical paths
                # such as a carried value flowing through a runtime-if merge.
                if dependent in self._quantum_value_uuids:
                    continue
                merged_origins = origins.setdefault(dependent, set())
                previous_size = len(merged_origins)
                merged_origins.update(current_origins)
                if len(merged_origins) != previous_size:
                    pending.append(dependent)
        return origins

    def _reject_quantum_segment_carry_escapes(
        self,
        segments: list[Segment],
    ) -> None:
        """Reject quantum-segment carry results that escape the segment.

        A loop placed in the quantum segment has its carried classical
        values computed at EMIT time (per-iteration unrolling) — the
        runtime executor never runs the loop, so a carry result read by
        a classical segment or returned as a block output has no
        runtime representation and would silently surface as ``None``.

        Args:
            segments (list[Segment]): The segments just built.

        Raises:
            MultipleQuantumSegmentsError: If a quantum-segment loop's
                carried result is a block output or is read by a
                non-quantum segment.
        """
        quantum_carry_results: dict[str, str] = {}
        quantum_classical_merges: dict[str, str] = {}
        divergent_quantum_array_merges: dict[str, str] = {}
        for segment in segments:
            if not isinstance(segment, QuantumSegment):
                continue
            for op in segment.operations:
                self._collect_carry_results(op, quantum_carry_results)
                self._collect_classical_merge_results(op, quantum_classical_merges)
                self._collect_divergent_quantum_array_merges(
                    op, divergent_quantum_array_merges
                )
        if (
            not quantum_carry_results
            and not quantum_classical_merges
            and not divergent_quantum_array_merges
        ):
            return

        quantum_reads: set[str] = set()
        for segment in segments:
            if isinstance(segment, QuantumSegment):
                quantum_reads |= self._segment_read_uuids(segment.operations)

        outside_reads: set[str] = set()
        for segment in segments:
            if isinstance(segment, QuantumSegment):
                continue
            outside_reads |= self._segment_read_uuids(segment.operations)

        block_output_reads: set[str] = set()
        for output in self._block_output_values:
            if isinstance(output, ValueBase):
                block_output_reads |= self._value_reference_uuids(output)
        escaped_reads = outside_reads | block_output_reads

        divergent_origins = self._propagate_escape_origins(
            divergent_quantum_array_merges
        )
        used_divergent_merges = sorted(
            {
                var_name
                for uuid in quantum_reads | escaped_reads
                for var_name in divergent_origins.get(uuid, ())
            }
        )
        if used_divergent_merges:
            raise MultipleQuantumSegmentsError(
                "A runtime if inside the quantum segment merges arrays from "
                "different physical qubit regions "
                f"({', '.join(used_divergent_merges)}). A single merged "
                "quantum value cannot dynamically select one physical "
                "register or slice at runtime. Use the selected view only "
                "inside each branch, or make the condition compile-time."
            )

        carry_origins = self._propagate_escape_origins(quantum_carry_results)

        escaped_names = sorted(
            {
                var_name
                for uuid in escaped_reads
                for var_name in carry_origins.get(uuid, ())
            }
        )
        if escaped_names:
            raise MultipleQuantumSegmentsError(
                "A loop inside the quantum segment carries classical "
                f"values ({', '.join(escaped_names)}) that are read "
                "outside it (by classical post-processing or as block "
                "outputs). Emit-time unrolling computes carried values "
                "while building the circuit, so they have no runtime "
                "representation the classical executor could surface. "
                "Compute the reduction in a classical-only loop (before "
                "or after the quantum operations), or drop it from the "
                "outputs."
            )

        merge_origins = self._propagate_escape_origins(quantum_classical_merges)
        escaped_merges = sorted(
            {
                var_name
                for uuid in escaped_reads
                for var_name in merge_origins.get(uuid, ())
            }
        )
        if escaped_merges:
            raise MultipleQuantumSegmentsError(
                "A runtime if inside the quantum segment has classical merges "
                f"({', '.join(escaped_merges)}) that are read outside it "
                "(by classical post-processing or as block outputs). These "
                "values exist only while emitting the circuit and have no "
                "runtime representation the classical executor could surface. "
                "Keep the merged value inside backend-supported quantum "
                "control flow, or compute it in a classical-only segment."
            )

    def _compute_quantum_needed(
        self,
        operations: list[Operation],
        dependency_graph: dict[str, set[str]],
    ) -> set[str]:
        """Collect every value required to compute a quantum op's inputs.

        Seeds from the input values of every quantum / hybrid operation
        (walked recursively through control flow) and back-propagates through
        ``dependency_graph`` so that a chain of classical expressions feeding
        a gate angle (e.g. ``(phase * 2) - 1``) is fully covered, not just the
        expression directly wired to the gate.

        Args:
            operations (list[Operation]): Top-level operations of the block.
            dependency_graph (dict[str, set[str]]): ``result_uuid ->
                set(operand_uuid, ...)`` as produced by
                ``build_dependency_graph``.

        Returns:
            set[str]: UUIDs of all values transitively needed by a quantum or
                hybrid operation.
        """

        effective_kind = self._effective_kind

        class QuantumInputCollector(ControlFlowVisitor):
            """Collect input-value UUIDs of every quantum / hybrid operation."""

            def __init__(self) -> None:
                """Initialize the empty input-UUID accumulator."""
                self.inputs: set[str] = set()

            def visit_operation(self, op: Operation) -> None:
                """Record the input-value UUIDs of a quantum / hybrid op.

                Control-flow ops whose bodies contain quantum work count
                too: their structural inputs (loop bounds, loop-carried
                ``iter_args``) must be resolvable at emit time, so a
                classical op computing them feeds the quantum segment
                exactly like a gate-angle expression.

                Args:
                    op (Operation): The operation being visited.

                Returns:
                    None: Mutates ``self.inputs`` in place.
                """
                kind = op.operation_kind
                if kind == OperationKind.CONTROL:
                    kind = effective_kind(op)
                if kind in (
                    OperationKind.QUANTUM,
                    OperationKind.HYBRID,
                ):
                    for v in op.all_input_values():
                        if isinstance(v, ValueBase):
                            self.inputs.add(v.uuid)

        collector = QuantumInputCollector()
        collector.visit_operations(operations)

        needed: set[str] = set()
        worklist = list(collector.inputs)
        while worklist:
            current = worklist.pop()
            if current in needed:
                continue
            needed.add(current)
            for dep in dependency_graph.get(current, ()):
                if dep not in needed:
                    worklist.append(dep)
        return needed

    def _is_measurement_tainted(self, op: Operation) -> bool:
        """Return whether ``op`` participates in the measurement dataflow.

        An operation is measurement-tainted when any of its input values or
        results is transitively derived from a measurement result (per the
        taint set computed in :meth:`_build_segments_list`). Such ops are
        genuine classical post-processing and must be kept in their own
        classical segment; non-tainted classical ops are parameter /
        structural expressions that can stay inside a quantum segment.

        Args:
            op (Operation): The operation to classify.

        Returns:
            bool: ``True`` if any input value or result UUID is in the
                measurement-taint set, ``False`` otherwise.
        """
        tainted = self._measurement_tainted
        for v in op.results:
            if isinstance(v, ValueBase) and v.uuid in tainted:
                return True
        for v in op.all_input_values():
            if isinstance(v, ValueBase) and v.uuid in tainted:
                return True
        return False

    def _feeds_quantum(self, op: Operation) -> bool:
        """Return whether ``op``'s result is needed by a quantum operation.

        Checks the quantum-needed set computed in
        :meth:`_build_segments_list`. Only classical ops that feed a quantum
        gate (directly or through a chain of other classical expressions) may
        be absorbed into a quantum segment; ops whose results are block
        outputs or feed only classical post-processing must stay classical.

        Args:
            op (Operation): The operation to classify.

        Returns:
            bool: ``True`` if any result UUID is in the quantum-needed set,
                ``False`` otherwise.
        """
        needed = self._quantum_needed
        for v in op.results:
            if isinstance(v, ValueBase) and v.uuid in needed:
                return True
        return False

    def _produces_block_output(self, op: Operation) -> bool:
        """Return whether any result of ``op`` is a block output value.

        Block outputs must be produced inside a classical segment so the
        executor runs the op and the orchestrator can surface the value;
        such ops are therefore never absorbed into a quantum segment.

        Args:
            op (Operation): The operation to classify.

        Returns:
            bool: ``True`` if any result UUID is in the block-output set,
                ``False`` otherwise.
        """
        outputs = self._block_output_uuids
        for v in op.results:
            if isinstance(v, ValueBase) and v.uuid in outputs:
                return True
        return False

    def _compute_absorbable(self, operations: list[Operation]) -> set[int]:
        """Compute the set of classical ops safe to fold into a quantum segment.

        A classical op is *absorbable* when it is non-measurement, feeds a
        quantum gate, is not a block output, and — transitively — every consumer
        of its results is a quantum / hybrid op or another absorbable classical
        op. Candidates include classical ops nested inside control-flow bodies
        (e.g. ``angle = base + 1`` inside ``qmc.range``), not just top-level
        ones: a nested chain link that ultimately feeds only quantum ops must be
        able to qualify so a top-level op feeding it stays absorbable, while a
        nested classical op whose result is a block output (or otherwise flows
        to a classical sink) must NOT qualify so the op feeding it is kept out
        of the quantum segment. Only top-level ops are actually folded into a
        segment at segmentation time; nested ops ride inside their enclosing
        control-flow op, so their membership here is used purely for the sink
        analysis.

        This iterate-until-stable computation keeps a chain of gate-angle
        expressions (e.g. ``(phase * 2) - 1``) absorbable while refusing to
        absorb a value that is also read back by later classical work or
        returned, which would be dropped by the (classical-op-free) quantum
        segment at execution time.

        Args:
            operations (list[Operation]): Top-level operations of the block.

        Returns:
            set[int]: ``id()`` of every classical op (top-level or nested) that
                is safe with respect to the quantum-segment absorption.
        """

        class ConsumerCollector(ControlFlowVisitor):
            """Map each value UUID to the operations that read it."""

            def __init__(self) -> None:
                """Initialize the empty consumer map."""
                self.consumers: dict[str, list[Operation]] = {}

            def visit_operation(self, op: Operation) -> None:
                """Record ``op`` as a consumer of each of its input values.

                Args:
                    op (Operation): The operation being visited.

                Returns:
                    None: Mutates ``self.consumers`` in place.
                """
                for v in op.all_input_values():
                    if isinstance(v, ValueBase):
                        self.consumers.setdefault(v.uuid, []).append(op)

        collector = ConsumerCollector()
        collector.visit_operations(operations)
        consumers = collector.consumers

        # Candidates: every classical op (recursing into control-flow bodies)
        # that is non-measurement, quantum-feeding, and not a block output.
        classical_ops = self._collect_classical_ops(operations)
        absorbable: set[int] = {
            id(op)
            for op in classical_ops
            if not self._is_measurement_tainted(op)
            and self._feeds_quantum(op)
            and not self._produces_block_output(op)
        }

        # Iteratively drop any candidate that has a result consumed by a
        # classical sink — a consumer that is neither a quantum/hybrid op nor a
        # (still-)absorbable classical op. Removing one op can disqualify the
        # ops feeding it, so iterate until the set stops changing.
        changed = True
        while changed:
            changed = False
            for op in classical_ops:
                if id(op) not in absorbable:
                    continue
                if self._has_classical_sink(op, consumers, absorbable):
                    absorbable.discard(id(op))
                    changed = True
        return absorbable

    def _collect_classical_ops(self, operations: list[Operation]) -> list[Operation]:
        """Collect every classical-kind op, recursing into control-flow bodies.

        Args:
            operations (list[Operation]): Operations to walk (a block's
                top-level list or a control-flow op's nested body).

        Returns:
            list[Operation]: Every ``OperationKind.CLASSICAL`` op reachable from
                ``operations``, including those nested inside control flow.
        """
        result: list[Operation] = []
        for op in operations:
            if op.operation_kind == OperationKind.CLASSICAL:
                result.append(op)
            elif (
                isinstance(op, (ForOperation, ForItemsOperation))
                and self._effective_kind(op) == OperationKind.CLASSICAL
            ):
                # A classical-body loop is one classical computation from
                # the segmentation's point of view — with loop-carried
                # value slots it can produce a gate parameter (e.g. an
                # accumulated angle) exactly like a BinOp chain, and emit
                # evaluates it inside the quantum segment the same way.
                result.append(op)
            if isinstance(op, HasNestedOps):
                for body in op.nested_op_lists():
                    result.extend(self._collect_classical_ops(body))
        return result

    def _has_classical_sink(
        self,
        op: Operation,
        consumers: dict[str, list[Operation]],
        absorbable: set[int],
    ) -> bool:
        """Return whether any result of ``op`` is read by a classical sink.

        A classical sink is a consumer that will run in a classical segment
        (measurement post-processing, a non-absorbable classical expression, a
        block output, classical control flow) and would therefore fail to read a
        value buried in a quantum segment. A consumer is safe (not a sink) when
        it is a quantum / hybrid op — including a gate nested inside a quantum
        loop, whose own effective kind is QUANTUM/HYBRID — or another currently-
        absorbable classical op (which may itself be nested). A nested classical
        consumer is therefore *not* trusted on the strength of its enclosing
        loop alone: it counts as safe only when it is itself absorbable (i.e.
        its results, too, flow only to quantum ops and are not block outputs).

        Args:
            op (Operation): The candidate op whose results are checked.
            consumers (dict[str, list[Operation]]): Map from value UUID to the
                operations that read it.
            absorbable (set[int]): ``id()`` of ops still considered absorbable
                in the current iteration.

        Returns:
            bool: ``True`` if some result of ``op`` is consumed by a classical
                sink, ``False`` otherwise.
        """
        for result in op.results:
            if not isinstance(result, ValueBase):
                continue
            for consumer in consumers.get(result.uuid, ()):
                if consumer is op:
                    continue
                if self._effective_kind(consumer) in (
                    OperationKind.QUANTUM,
                    OperationKind.HYBRID,
                ):
                    continue
                if id(consumer) in absorbable:
                    continue
                return True
        return False

    def _effective_kind(self, op: Operation) -> OperationKind:
        """Determine the effective kind of an operation.

        Control flow inherits the kinds of its nested operations. An
        operation-empty loop with hidden quantum ownership/carry records stays
        quantum so it cannot create a spurious quantum-classical boundary.

        Args:
            op (Operation): Operation or nested control-flow node to classify.

        Returns:
            OperationKind: Effective segmentation kind for ``op``.
        """
        kind = op.operation_kind

        if kind != OperationKind.CONTROL:
            return kind

        # Control flow - determine by inner operations
        if isinstance(op, HasNestedOps):
            inner_kinds: set[OperationKind] = set()
            for body in op.nested_op_lists():
                inner_kinds.update(self._effective_kind(inner) for inner in body)
        else:
            # Other control - treat as classical for now
            return OperationKind.CLASSICAL

        # Remove CONTROL from inner kinds (nested control flow)
        inner_kinds.discard(OperationKind.CONTROL)

        if len(inner_kinds) == 0:
            # A marker-only loop may become operation-empty after
            # ``StripSliceArrayOpsPass`` while still carrying quantum
            # ownership records for ``AnalyzePass``.  Those records do not
            # represent host-side classical work and must not split one
            # quantum stream into Q -> C -> Q segments merely because the
            # loop body is empty.  Classical carries (including a measured
            # Bit selected by the final loop index) intentionally remain
            # classical so the post-quantum executor evaluates them.
            if isinstance(op, (ForOperation, ForItemsOperation, WhileOperation)):
                if any(
                    record.before.type.is_quantum() or record.after.type.is_quantum()
                    for record in op.loop_carried_rebinds
                ) or any(
                    region_arg.init.type.is_quantum()
                    or region_arg.block_arg.type.is_quantum()
                    or region_arg.result.type.is_quantum()
                    for region_arg in op.region_args
                ):
                    return OperationKind.QUANTUM
            return OperationKind.CLASSICAL  # Empty classical control flow
        elif len(inner_kinds) == 1:
            return inner_kinds.pop()
        else:
            # Mixed operations in control flow
            # If any QUANTUM or HYBRID operation exists, treat as QUANTUM
            # because measurements belong in the quantum circuit
            # EmitPass will handle backend-specific behavior
            if (
                OperationKind.QUANTUM in inner_kinds
                or OperationKind.HYBRID in inner_kinds
            ):
                return OperationKind.QUANTUM
            return OperationKind.CLASSICAL

    def _create_segment(
        self,
        kind: OperationKind | None,
        operations: list[Operation],
    ) -> Segment:
        """Create a segment of the appropriate type."""
        if kind == OperationKind.QUANTUM:
            return QuantumSegment(operations=operations)
        else:
            return ClassicalSegment(operations=operations)

    def _compute_segment_io(
        self,
        segments: list[Segment],
        block: Block,
    ) -> None:
        """Compute input/output refs for each segment.

        A segment reads values defined outside it (inputs)
        and writes values used outside it or returned (outputs).

        Args:
            segments (list[Segment]): Ordered segments whose boundary refs are
                populated in place.
            block (Block): Lowered block whose public outputs seed final
                liveness.
        """
        # Track which segment defines each value
        value_definitions: dict[str, int] = {}  # uuid -> segment index

        # Initialize with block inputs
        for v in block.input_values:
            value_definitions[v.uuid] = -1  # -1 = external input

        # Initialize with parameters
        for v in block.parameters.values():
            value_definitions[v.uuid] = -1  # -1 = external input

        for i, segment in enumerate(segments):
            segment_inputs: set[str] = set()
            segment_outputs: set[str] = set()

            self._collect_segment_io(
                segment.operations,
                i,
                value_definitions,
                segment_inputs,
                segment_outputs,
            )

            segment.input_refs = sorted(segment_inputs)
            segment.output_refs = sorted(segment_outputs)

        block_output_refs: set[str] = set()
        for output in block.output_values:
            block_output_refs.update(self._value_reference_uuids(output))

        live_after = set(block_output_refs)
        for segment in reversed(segments):
            definitions = segment.output_refs
            segment.output_refs = [
                reference for reference in definitions if reference in live_after
            ]
            live_after.update(segment.input_refs)

    def _collect_segment_io(
        self,
        operations: list[Operation],
        segment_index: int,
        value_definitions: dict[str, int],
        segment_inputs: set[str],
        segment_outputs: set[str],
    ) -> None:
        """Recursively collect definitions and external structural reads.

        Args:
            operations (list[Operation]): Segment operation tree to inspect.
            segment_index (int): Index of the segment owning local definitions.
            value_definitions (dict[str, int]): Mutable UUID-to-defining-segment
                map, including ``-1`` entries for external block inputs.
            segment_inputs (set[str]): Mutable set receiving external reads.
            segment_outputs (set[str]): Mutable set receiving local
                definitions before reverse liveness filtering.
        """

        value_reference_uuids = self._value_reference_uuids

        class SegmentIOCollector(ControlFlowVisitor):
            def visit_operation(self, op: Operation) -> None:
                """Record one operation's reads and definitions.

                Args:
                    op (Operation): The visited operation.
                """
                # Loop iteration identities and RegionArg block arguments are
                # definitions owned by the enclosing control-flow op, not
                # boundary inputs. Register them before walking the body so a
                # body read cannot be misclassified as an external segment
                # dependency. Keeping these definitions also lets a structural
                # public output retain a final loop key/index UUID when needed.
                if isinstance(op, ForOperation) and op.loop_var_value is not None:
                    self._record_definition(op.loop_var_value)
                if isinstance(op, ForItemsOperation):
                    for key_value in op.key_var_values or ():
                        self._record_definition(key_value)
                    if op.value_var_value is not None:
                        self._record_definition(op.value_var_value)
                if isinstance(op, (ForOperation, ForItemsOperation, WhileOperation)):
                    for region_arg in op.region_args:
                        self._record_definition(region_arg.block_arg)

                # Inputs not defined in this segment are boundary reads.
                # If yields are deferred until after visiting both branches,
                # when branch-local definitions are known; recording them here
                # would misclassify local measurements as segment inputs.
                inputs = (
                    list(op.operands)
                    if isinstance(op, IfOperation)
                    else genuine_input_values(op)
                )
                for operand in inputs:
                    self._record_read(operand)

                # Results are defined by this segment
                for result in op.results:
                    self._record_definition(result)

            def _visit_control_flow(self, op: Operation) -> None:
                """Recurse into control flow, handling if-merges explicitly.

                Args:
                    op (Operation): The operation whose nested bodies are
                        visited.
                """
                if isinstance(op, IfOperation):
                    # Explicit merge handling via iter_merges — the
                    # collector must not rely on merge storage being
                    # reachable through the generic nested-list walk.
                    # Branch bodies are visited first so branch-defined
                    # merge sources register as in-segment definitions
                    # before the merges read them.
                    self.visit_operations(op.true_operations)
                    self.visit_operations(op.false_operations)
                    for merge in op.iter_merges():
                        self._record_read(merge.true_value)
                        self._record_read(merge.false_value)
                        self._record_definition(merge.result)
                    return
                super()._visit_control_flow(op)

            @staticmethod
            def _record_read(operand: object) -> None:
                """Mark a value read as a segment input unless locally defined.

                Args:
                    operand (object): Candidate operand; ignored unless it
                        is a ``ValueBase``.
                """
                if isinstance(operand, ValueBase):
                    for reference in value_reference_uuids(operand):
                        if (
                            reference not in value_definitions
                            or value_definitions[reference] != segment_index
                        ):
                            segment_inputs.add(reference)

            @staticmethod
            def _record_definition(result: ValueBase) -> None:
                """Mark a value as defined by (and output of) this segment.

                Args:
                    result (ValueBase): The defined result value.
                """
                value_definitions[result.uuid] = segment_index
                segment_outputs.add(result.uuid)

        collector = SegmentIOCollector()
        collector.visit_operations(operations)
