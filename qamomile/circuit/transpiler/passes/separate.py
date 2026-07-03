"""Segmentation pass: Split a block into executable program steps."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import (
    PhiOp,
    RuntimeClassicalExpr,
)
from qamomile.circuit.ir.operation.classical_ops import DecodeQFixedOperation
from qamomile.circuit.ir.operation.control_flow import (
    HasNestedOps,
    IfOperation,
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
    ValueLike,
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


def lower_measure_qfixed(op: MeasureQFixedOperation) -> list[Operation]:
    """Lower MeasureQFixedOperation to MeasureVectorOperation + decode.

    Returns:
        List of operations: [MeasureVectorOperation, DecodeQFixedOperation]
    """
    qfixed = op.operands[0]

    qubit_uuids = qfixed.get_cast_qubit_uuids() or qfixed.get_qfixed_qubit_uuids()
    num_bits = op.num_bits or len(qubit_uuids)
    int_bits = op.int_bits

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
    for op in block.operations:
        if isinstance(op, MeasureQFixedOperation):
            lowered_ops.extend(lower_measure_qfixed(op))
        else:
            lowered_ops.append(op)
    return dataclasses.replace(block, operations=lowered_ops)


def collect_value_like_uuids(value: ValueLike) -> set[str]:
    """Collect UUIDs contained in a value-like IR object.

    Args:
        value (ValueLike): Value-like object to inspect.

    Returns:
        set[str]: UUIDs for ``value`` itself and any recursively contained
            tuple/dict elements.
    """
    uuids = {value.uuid}
    if isinstance(value, TupleValue):
        for element in value.elements:
            uuids.update(collect_value_like_uuids(element))
    elif isinstance(value, DictValue):
        for key, entry_value in value.entries:
            uuids.update(collect_value_like_uuids(key))
            uuids.update(collect_value_like_uuids(entry_value))
    return uuids


def _is_public_runtime_input(value: ValueLike) -> bool:
    """Return whether a block input belongs in ``ProgramABI.public_inputs``.

    Args:
        value (ValueLike): Block input or parameter value to classify.

    Returns:
        bool: ``True`` for classical or classical structural values that can
            be bound by users at runtime; ``False`` for quantum values, which
            are allocated and consumed by the quantum segment.
    """
    return not value.type.is_quantum()


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
        """Build a ProgramPlan from segmented operations."""
        raise NotImplementedError


class NisqSegmentationStrategy(SegmentationStrategy):
    """Single-quantum-segment planning strategy for current Qamomile runtimes."""

    def create_plan(
        self,
        segments: list[Segment],
        block: Block,
        boundaries: list[HybridBoundary],
    ) -> ProgramPlan:
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

        public_inputs = {
            name: value
            for name, value in zip(block.label_args, block.input_values, strict=True)
            if _is_public_runtime_input(value)
        }
        for name, value in block.parameters.items():
            if _is_public_runtime_input(value):
                public_inputs.setdefault(name, value)

        abi = ProgramABI(
            public_inputs=public_inputs,
            output_values=list(block.output_values),
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
        """Segment the block into a ProgramPlan."""
        input = ValidateWhileContractPass().run(input)
        block = materialize_return(input)
        block = lower_operations(block)
        return self._segment(block)

    def _segment(self, block: Block) -> ProgramPlan:
        """Build a ProgramPlan from a lowered block."""
        segments = self._build_segments_list(block)
        return self._strategy.create_plan(segments, block, self._boundaries)

    def _build_segments_list(self, block: Block) -> list[Segment]:
        """Build list of segments from block operations.

        Extracted from _separate_segments to allow reuse.
        """
        segments: list[Segment] = []
        self._boundaries: list[HybridBoundary] = []
        self._current_block_operations = block.operations

        # Measurement-taint set: every value transitively derived from a
        # measurement result. Computed with the same dataflow utilities as
        # ``AnalyzePass`` / ``ClassicalLoweringPass`` so the three passes
        # agree on what "measurement-derived" means. Used below to decide
        # whether a classical op interleaved in a quantum region is a
        # genuine post-measurement computation (must split off into its own
        # classical segment) or a parameter / structural expression that
        # merely feeds a quantum gate (can stay in the quantum segment).
        dependency_graph = build_dependency_graph(block.operations)
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
        self._block_output_uuids = set()
        for value in block.output_values:
            self._block_output_uuids.update(collect_value_like_uuids(value))

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
        # The consumers map is computed once and shared with
        # ``_classify_runtime_exprs`` below to avoid a second traversal of the
        # nested control flow.
        consumers = self._build_consumers_map(block.operations)
        self._consumers = consumers
        self._absorbable_op_ids = self._compute_absorbable(block.operations, consumers)

        # Runtime-expression placement sets: a ``RuntimeClassicalExpr``
        # (measurement-derived classical op) may be consumed in-circuit
        # (runtime if/while conditions), host-side (block outputs, classical
        # post-processing), or both. Computed once here; used by the routing
        # loop below to decide quantum-segment vs classical-segment placement.
        (
            self._quantum_consumed_expr_ids,
            self._host_evaluated_expr_ids,
        ) = self._classify_runtime_exprs(block.operations, consumers)

        current_ops: list[Operation] = []
        current_kind: OperationKind | None = None
        # Parameter-expression classical ops (the gate-angle expressions this
        # work absorbs) that appear before / outside the quantum segment. They
        # are held here and prepended to the quantum segment once it starts, so
        # they end up computed inside the quantum circuit regardless of where
        # the user wrote them — never stranded in a classical prep segment.
        pending_absorbable: list[Operation] = []
        # Host-evaluated runtime expressions encountered while inside the
        # quantum segment. They are held here and flushed into the first
        # post-quantum classical segment (creating one at the end of the
        # stream if no classical op follows), so a runtime expression written
        # between measurements never splits the quantum segment and never
        # gets stranded inside it.
        pending_post: list[Operation] = []

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

            # Runtime classical expressions (measurement-derived classical
            # ops, lowered by ``ClassicalLoweringPass``) are placed by
            # consumer (see :meth:`_classify_runtime_exprs`):
            #
            # - An expr bridging a measurement to a runtime IfOperation /
            #   WhileOperation condition (directly or through other
            #   in-circuit exprs) stays inside the quantum segment so the
            #   backend can lower it to a native classical expression.
            #   Examples that work under this rule but the old BitType-only
            #   heuristic could not handle:
            #     ``if (s0 + 2 * s1 + 4 * s2) == 5:``  (UInt-typed BinOp/CompOp)
            #     ``if measure(q) == bound_uint_param:`` (mixed Bit/UInt)
            # - An expr whose result is a block output or feeds host-side
            #   classical post-processing is deferred to a post-quantum
            #   classical segment (``pending_post``) where
            #   ``ClassicalExecutor`` computes it per shot. Keeping it in
            #   the quantum segment would silently drop its value — nothing
            #   in the quantum runtime surfaces expression results, so the
            #   orchestrator would resolve the output to ``None``.
            # - An expr consumed by both worlds is placed in both; it is
            #   pure, so duplicate evaluation is safe.
            if (
                op_kind == OperationKind.CLASSICAL
                and current_kind == OperationKind.QUANTUM
                and isinstance(op, RuntimeClassicalExpr)
            ):
                if id(op) in self._quantum_consumed_expr_ids:
                    current_ops.append(op)
                    if id(op) in self._host_evaluated_expr_ids:
                        pending_post.append(op)
                else:
                    pending_post.append(op)
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

            # Entering (or continuing) a post-quantum classical segment:
            # flush deferred runtime expressions first so ops that consume
            # their results execute after them. ``pending_post`` only fills
            # while the quantum segment is active, so any classical segment
            # reached with a non-empty list is post-quantum.
            if current_kind == OperationKind.CLASSICAL and pending_post:
                current_ops.extend(pending_post)
                pending_post = []

            if op_kind == OperationKind.QUANTUM:
                op, shadow_ops = self._rewrite_post_shadow_control_flow(
                    op,
                    self._current_block_operations,
                )
                if shadow_ops:
                    pending_post.extend(shadow_ops)

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

        # Deferred runtime expressions with no classical op after them in
        # the stream (e.g. ``s = b0 & b1; return s``) still must run
        # host-side: close the plan with a classical segment holding them.
        if pending_post:
            segments.append(ClassicalSegment(operations=pending_post))

        # Compute input/output refs for each segment
        self._compute_segment_io(segments, block)

        return segments

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

        segmentation = self

        class QuantumInputCollector(ControlFlowVisitor):
            """Collect input-value UUIDs of every quantum / hybrid operation."""

            def __init__(self) -> None:
                """Initialize the empty input-UUID accumulator."""
                self.inputs: set[str] = set()

            def visit_operation(self, op: Operation) -> None:
                """Record the input-value UUIDs of a quantum / hybrid op.

                Args:
                    op (Operation): The operation being visited.

                Returns:
                    None: Mutates ``self.inputs`` in place.
                """
                if (
                    op.operation_kind
                    in (
                        OperationKind.QUANTUM,
                        OperationKind.HYBRID,
                    )
                    or segmentation._effective_kind(op) == OperationKind.QUANTUM
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

    def _compute_absorbable(
        self,
        operations: list[Operation],
        consumers: dict[str, list[Operation]],
    ) -> set[int]:
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
            consumers (dict[str, list[Operation]]): ``value_uuid ->
                [consumer, ...]`` map over the whole op tree, as built by
                :meth:`_build_consumers_map`.

        Returns:
            set[int]: ``id()`` of every classical op (top-level or nested) that
                is safe with respect to the quantum-segment absorption.
        """
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

    def _build_consumers_map(
        self,
        operations: list[Operation],
    ) -> dict[str, list[Operation]]:
        """Map each value UUID to the operations that read it.

        Walks the full operation tree (recursing through control-flow
        bodies) and records every operation as a consumer of each of its
        input values, including extra value fields exposed via
        ``all_input_values`` (e.g. an ``IfOperation``'s condition).

        Args:
            operations (list[Operation]): Top-level operations of the block.

        Returns:
            dict[str, list[Operation]]: ``value_uuid -> [consumer, ...]``
                over the whole tree.
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
        return collector.consumers

    def _build_producer_map(
        self,
        operations: list[Operation],
    ) -> dict[str, Operation]:
        """Map each produced value UUID to the operation that writes it.

        Args:
            operations (list[Operation]): Top-level operations of the block.

        Returns:
            dict[str, Operation]: Mapping from result UUID to the producing
                operation.
        """

        class ProducerCollector(ControlFlowVisitor):
            """Collect result producers across nested operation trees."""

            def __init__(self) -> None:
                """Initialize the empty producer map."""
                self.producers: dict[str, Operation] = {}

            def visit_operation(self, op: Operation) -> None:
                """Record result producers for ``op``.

                Args:
                    op (Operation): Operation being visited.

                Returns:
                    None: Mutates ``self.producers`` in place.
                """
                for result in op.results:
                    if isinstance(result, ValueBase):
                        self.producers[result.uuid] = op

        collector = ProducerCollector()
        collector.visit_operations(operations)
        return collector.producers

    def _rewrite_post_shadow_control_flow(
        self,
        op: Operation,
        scope_operations: list[Operation],
    ) -> tuple[Operation, list[Operation]]:
        """Rewrite quantum control flow and build host-side shadow ops.

        Args:
            op (Operation): Quantum-segment operation to inspect.
            scope_operations (list[Operation]): Operations in the lexical
                scope that contains ``op``. Used to collect pure-classical
                producers for runtime conditions.

        Returns:
            tuple[Operation, list[Operation]]: The operation with host-only
                Phi outputs removed from nested quantum ``IfOperation`` nodes,
                plus classical shadow operations that recompute those Phi
                outputs after the quantum segment.
        """
        if isinstance(op, IfOperation):
            true_operations, true_shadows = self._rewrite_post_shadow_body(
                op.true_operations
            )
            false_operations, false_shadows = self._rewrite_post_shadow_body(
                op.false_operations
            )
            rewritten: IfOperation = dataclasses.replace(
                op,
                true_operations=true_operations,
                false_operations=false_operations,
            )

            shadow_ops: list[Operation] = []
            if true_shadows or false_shadows:
                condition_prelude = self._collect_classical_dependency_ops(
                    [op.condition],
                    self._shadow_condition_scope(scope_operations),
                )
                if condition_prelude is None:
                    condition_prelude = []
                shadow_ops.extend(condition_prelude)
                shadow_ops.append(
                    dataclasses.replace(
                        op,
                        true_operations=true_shadows,
                        false_operations=false_shadows,
                        phi_ops=[],
                        results=[],
                    )
                )

            if self._effective_kind(rewritten) == OperationKind.QUANTUM:
                local_shadow_ops = self._build_post_shadow_if_ops(
                    rewritten,
                    scope_operations,
                )
                if local_shadow_ops:
                    shadow_ops.extend(local_shadow_ops)
                    rewritten = self._drop_host_only_phi_outputs(
                        rewritten,
                        local_shadow_ops,
                    )
            return rewritten, shadow_ops

        if isinstance(op, HasNestedOps):
            shadow_lists: list[list[Operation]] = []
            rewritten_lists: list[list[Operation]] = []
            has_shadow = False
            changed = False
            for body in op.nested_op_lists():
                rewritten_body, body_shadows = self._rewrite_post_shadow_body(body)
                rewritten_lists.append(rewritten_body)
                shadow_lists.append(body_shadows)
                has_shadow = has_shadow or bool(body_shadows)
                changed = changed or any(
                    new is not old for new, old in zip(rewritten_body, body)
                )
            rewritten_op = op.rebuild_nested(rewritten_lists) if changed else op
            if not has_shadow:
                return rewritten_op, []
            return rewritten_op, [op.rebuild_nested(shadow_lists)]

        return op, []

    def _rewrite_post_shadow_body(
        self,
        operations: list[Operation],
    ) -> tuple[list[Operation], list[Operation]]:
        """Rewrite a control-flow body and collect its shadow operations.

        Args:
            operations (list[Operation]): Operations in one lexical body.

        Returns:
            tuple[list[Operation], list[Operation]]: Rewritten body for the
                quantum segment and classical-only shadow body for the
                post-quantum segment.
        """
        rewritten_body: list[Operation] = []
        shadow_body: list[Operation] = []
        for body_op in operations:
            rewritten_op, shadow_ops = self._rewrite_post_shadow_control_flow(
                body_op,
                operations,
            )
            rewritten_body.append(rewritten_op)
            shadow_body.extend(shadow_ops)
        return rewritten_body, shadow_body

    def _drop_host_only_phi_outputs(
        self,
        op: IfOperation,
        shadow_ops: list[Operation],
    ) -> IfOperation:
        """Remove host-only Phi outputs from a quantum ``IfOperation``.

        Args:
            op (IfOperation): Rewritten ``IfOperation`` candidate.
            shadow_ops (list[Operation]): Shadow operations returned by
                :meth:`_build_post_shadow_if_ops`.

        Returns:
            IfOperation: ``op`` with host-only Phi outputs removed when
                applicable.
        """
        shadow_if = shadow_ops[-1]
        if not isinstance(shadow_if, IfOperation):
            return op
        host_phi_output_ids = {
            result.uuid
            for result in shadow_if.results
            if result.uuid not in self._quantum_needed
        }
        if not host_phi_output_ids:
            return op
        return dataclasses.replace(
            op,
            results=[
                result
                for result in op.results
                if result.uuid not in host_phi_output_ids
            ],
            phi_ops=[
                phi for phi in op.phi_ops if phi.output.uuid not in host_phi_output_ids
            ],
        )

    def _build_post_shadow_if_ops(
        self,
        op: IfOperation,
        condition_scope: list[Operation] | None = None,
    ) -> list[Operation]:
        """Build host-side operations that recompute selected Phi outputs.

        Args:
            op (IfOperation): Quantum-effective ``IfOperation`` whose Phi
                outputs may be needed after the quantum segment.
            condition_scope (list[Operation] | None): Lexical scope used to
                collect pure-classical producers for the condition. Defaults
                to the whole block for top-level compatibility.

        Returns:
            list[Operation]: Pure-classical operations to append to the
                post-quantum segment.
        """
        target_phis: list[PhiOp] = []
        true_seeds: list[ValueBase] = []
        false_seeds: list[ValueBase] = []

        for phi in op.phi_ops:
            if not self._phi_needs_host_value(phi):
                continue
            if not isinstance(phi.true_value, ValueBase) or not isinstance(
                phi.false_value, ValueBase
            ):
                continue
            true_ops = self._collect_classical_dependency_ops(
                [phi.true_value],
                op.true_operations,
            )
            false_ops = self._collect_classical_dependency_ops(
                [phi.false_value],
                op.false_operations,
            )
            if true_ops is None or false_ops is None:
                continue
            target_phis.append(phi)
            true_seeds.append(phi.true_value)
            false_seeds.append(phi.false_value)

        if not target_phis:
            return []

        condition_prelude = self._collect_classical_dependency_ops(
            [op.condition],
            self._shadow_condition_scope(condition_scope)
            if condition_scope is not None
            else self._current_block_operations,
        )
        if condition_prelude is None:
            condition_prelude = []

        true_operations = self._collect_classical_dependency_ops(
            true_seeds,
            op.true_operations,
        )
        false_operations = self._collect_classical_dependency_ops(
            false_seeds,
            op.false_operations,
        )
        if true_operations is None or false_operations is None:
            return []

        shadow_if = dataclasses.replace(
            op,
            true_operations=true_operations,
            false_operations=false_operations,
            phi_ops=target_phis,
            results=[phi.output for phi in target_phis],
        )
        return [*condition_prelude, shadow_if]

    def _shadow_condition_scope(
        self,
        scope_operations: list[Operation],
    ) -> list[Operation]:
        """Return operations visible to a nested shadow condition.

        Args:
            scope_operations (list[Operation]): Operations from the immediate
                lexical body that contains the shadowed ``IfOperation``.

        Returns:
            list[Operation]: Top-level block operations followed by immediate
                scope operations, deduplicated by object identity.
        """
        combined: list[Operation] = []
        seen: set[int] = set()
        for candidate in [*self._current_block_operations, *scope_operations]:
            candidate_id = id(candidate)
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            combined.append(candidate)
        return combined

    def _phi_needs_host_value(self, phi: PhiOp) -> bool:
        """Return whether a Phi output must be available host-side.

        Args:
            phi (PhiOp): Phi operation to inspect.

        Returns:
            bool: ``True`` when the Phi output is a block output or feeds a
                host-side classical consumer.
        """
        output = phi.output
        if isinstance(output, ArrayValue):
            return False
        if isinstance(output.type, QubitType):
            return False
        if output.uuid in self._block_output_uuids:
            return True
        for consumer in self._consumers.get(output.uuid, ()):
            if isinstance(consumer, RuntimeClassicalExpr):
                if id(consumer) in self._host_evaluated_expr_ids:
                    return True
                continue
            if self._effective_kind(consumer) == OperationKind.CLASSICAL:
                return True
        return False

    def _collect_classical_dependency_ops(
        self,
        seeds: list[ValueBase],
        operations: list[Operation],
    ) -> list[Operation] | None:
        """Collect pure-classical producers needed by ``seeds``.

        Args:
            seeds (list[ValueBase]): Values whose producers should be
                available.
            operations (list[Operation]): Operation scope from which producers
                may be copied.

        Returns:
            list[Operation] | None: Ordered pure-classical producer operations,
                or ``None`` when a needed producer is a nested operation that
                cannot be copied safely.
        """
        local_producers = self._build_producer_map(operations)
        required_ids: set[int] = set()

        def require(value: ValueBase) -> bool:
            producer = local_producers.get(value.uuid)
            if producer is None:
                return True
            if self._effective_kind(producer) != OperationKind.CLASSICAL:
                return True
            required_ids.add(id(producer))
            for operand in producer.all_input_values():
                if not require(operand):
                    return False
            return True

        for seed in seeds:
            if not require(seed):
                return None

        ordered: list[Operation] = []
        remaining = set(required_ids)
        for candidate in operations:
            if id(candidate) in required_ids:
                ordered.append(candidate)
                remaining.discard(id(candidate))
        if remaining:
            return None
        return ordered

    def _classify_runtime_exprs(
        self,
        operations: list[Operation],
        consumers: dict[str, list[Operation]],
    ) -> tuple[set[int], set[int]]:
        """Classify each ``RuntimeClassicalExpr`` by where it must execute.

        A runtime expression (a measurement-derived classical op lowered by
        ``ClassicalLoweringPass``) can be consumed by two different worlds:

        - **In-circuit**: its result feeds a quantum / hybrid operation —
          in practice a runtime ``IfOperation`` / ``WhileOperation``
          condition — or another in-circuit runtime expression. Such an
          expr must ride inside the quantum segment so the backend emits
          it as a native classical expression.
        - **Host-side**: its result is a block output or is read by
          classical post-processing (a classical-segment op, classical
          control flow, or another host-evaluated expression). Such an
          expr must land in a classical segment so ``ClassicalExecutor``
          computes its value; stranded in a quantum segment it would be
          silently dropped and the orchestrator would surface ``None``.

        Both sets are transitive closures over the expression dataflow: an
        expr feeding an in-circuit expr is itself in-circuit (the backend
        expression tree needs its operands emitted too), and an expr read
        by a host-evaluated expr must itself be host-evaluated (the host
        evaluation needs its operand values). An expr may be in both sets
        (e.g. used as an if condition *and* returned); segmentation then
        places it in both segments, which is safe because the op is pure.

        Args:
            operations (list[Operation]): Top-level operations of the block.
            consumers (dict[str, list[Operation]]): ``value_uuid ->
                [consumer, ...]`` map over the whole op tree, as built by
                :meth:`_build_consumers_map`.

        Returns:
            tuple[set[int], set[int]]: ``(quantum_consumed, host_evaluated)``
                — ``id()`` sets of the runtime expressions that must be
                placed in the quantum segment and evaluated host-side,
                respectively.
        """
        exprs = [
            op
            for op in self._collect_classical_ops(operations)
            if isinstance(op, RuntimeClassicalExpr)
        ]
        if not exprs:
            return set(), set()

        # Ops that execute host-side when reached: every top-level op whose
        # effective kind is CLASSICAL, plus everything nested inside one.
        # Consumers nested inside a QUANTUM-effective control op (e.g. the
        # phi merges of a runtime if) execute in-circuit, not host-side,
        # and must not drag their operand expressions into a classical
        # segment.
        top_level_ids = {id(op) for op in operations}
        host_scope_ids: set[int] = set()
        for top in operations:
            if self._effective_kind(top) == OperationKind.CLASSICAL:
                host_scope_ids |= self._collect_op_ids([top])

        def result_consumers(op: Operation) -> list[Operation]:
            """Collect the operations reading any result of ``op``.

            Args:
                op (Operation): The producer operation.

            Returns:
                list[Operation]: Consumers of ``op``'s results (``op``
                    itself excluded).
            """
            found: list[Operation] = []
            for result in op.results:
                if isinstance(result, ValueBase):
                    found.extend(
                        c for c in consumers.get(result.uuid, ()) if c is not op
                    )
            return found

        # In-circuit set: seeded by consumption from a quantum / hybrid op
        # (a runtime if/while's effective kind is QUANTUM when its body
        # holds gates); grown backward through expression chains until
        # stable, so every operand-expr of an emitted expr is emitted too.
        quantum_consumed: set[int] = set()
        changed = True
        while changed:
            changed = False
            for op in exprs:
                if id(op) in quantum_consumed:
                    continue
                for consumer in result_consumers(op):
                    if self._effective_kind(consumer) in (
                        OperationKind.QUANTUM,
                        OperationKind.HYBRID,
                    ) or (
                        isinstance(consumer, RuntimeClassicalExpr)
                        and id(consumer) in quantum_consumed
                    ):
                        quantum_consumed.add(id(op))
                        changed = True
                        break

        # Host-side set: an expr that will execute host-side (a top-level
        # expr not consumed in-circuit, or an expr nested in host-executing
        # classical control flow) must be evaluated by the executor; a
        # block output must be available host-side even when also consumed
        # in-circuit. Grown backward through expression chains and
        # host-executing non-expression consumers until stable.
        host_evaluated: set[int] = {
            id(op)
            for op in exprs
            if (
                (id(op) in top_level_ids or id(op) in host_scope_ids)
                and id(op) not in quantum_consumed
            )
            or self._produces_block_output(op)
        }
        changed = True
        while changed:
            changed = False
            for op in exprs:
                if id(op) in host_evaluated:
                    continue
                for consumer in result_consumers(op):
                    if isinstance(consumer, RuntimeClassicalExpr):
                        needs_host = id(consumer) in host_evaluated
                    else:
                        needs_host = id(consumer) in host_scope_ids
                    if needs_host:
                        host_evaluated.add(id(op))
                        changed = True
                        break

        return quantum_consumed, host_evaluated

    def _collect_op_ids(self, operations: list[Operation]) -> set[int]:
        """Collect ``id()`` of every op in the tree, recursing control flow.

        Args:
            operations (list[Operation]): Operations to walk (including
                each op's nested control-flow bodies).

        Returns:
            set[int]: ``id()`` of every operation reachable from
                ``operations``.
        """
        ids: set[int] = set()

        class IdCollector(ControlFlowVisitor):
            """Record the identity of every visited operation."""

            def visit_operation(self, op: Operation) -> None:
                """Add ``op``'s identity to the enclosing set.

                Args:
                    op (Operation): The operation being visited.

                Returns:
                    None: Mutates the enclosing ``ids`` set in place.
                """
                ids.add(id(op))

        IdCollector().visit_operations(operations)
        return ids

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

        For control flow, examine the inner operations.
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
            return OperationKind.CLASSICAL  # Empty control flow
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

            segment.input_refs = list(segment_inputs)
            segment.output_refs = list(segment_outputs)

    def _collect_segment_io(
        self,
        operations: list[Operation],
        segment_index: int,
        value_definitions: dict[str, int],
        segment_inputs: set[str],
        segment_outputs: set[str],
    ) -> None:
        """Recursively collect input/output refs from operations."""

        class SegmentIOCollector(ControlFlowVisitor):
            def visit_operation(self, op: Operation) -> None:
                # Operands not defined in this segment are inputs
                for operand in op.operands:
                    if isinstance(operand, ValueBase):
                        if (
                            operand.uuid not in value_definitions
                            or value_definitions[operand.uuid] != segment_index
                        ):
                            segment_inputs.add(operand.uuid)

                # Results are defined by this segment
                for result in op.results:
                    value_definitions[result.uuid] = segment_index
                    segment_outputs.add(result.uuid)

        collector = SegmentIOCollector()
        collector.visit_operations(operations)
