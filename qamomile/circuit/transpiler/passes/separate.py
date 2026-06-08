"""Segmentation pass: Split a block into executable program steps."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import RuntimeClassicalExpr
from qamomile.circuit.ir.operation.classical_ops import DecodeQFixedOperation
from qamomile.circuit.ir.operation.control_flow import (
    HasNestedOps,
)
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.operation.gate import (
    MeasureQFixedOperation,
    MeasureVectorOperation,
)
from qamomile.circuit.ir.operation.operation import OperationKind
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.types.primitives import BitType, QubitType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value, ValueBase
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

        abi = ProgramABI(
            public_inputs={name: value for name, value in block.parameters.items()},
            output_refs=[v.uuid for v in block.output_values],
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
        self._block_output_uuids = {
            v.uuid for v in block.output_values if isinstance(v, ValueBase)
        }

        # Absorbable set: top-level classical ops that may be folded into the
        # surrounding quantum segment. An op qualifies only when its results
        # are consumed exclusively by quantum / hybrid ops, or by other
        # absorbable classical ops that themselves ultimately feed quantum —
        # never by a classical sink (a classical-segment op, a measurement-
        # derived op, or a block output). Computing this transitively is what
        # makes a chain like ``(phase * 2) - 1`` feeding a gate fully
        # absorbable while a value read back by later classical work stays in
        # its own classical segment.
        self._absorbable_op_ids = self._compute_absorbable(block.operations)

        current_ops: list[Operation] = []
        current_kind: OperationKind | None = None

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

            # Parameter / structural classical ops that are *not* derived
            # from a measurement (e.g. a gate-angle expression like
            # ``theta=-phase`` over a runtime parameter, which constant
            # folding cannot collapse because ``phase`` stays symbolic) must
            # not split the surrounding quantum region. They are emit-time
            # foldable into a backend parameter expression and belong with
            # the gate they feed. Without this carve-out such an op forces a
            # spurious quantum→classical→quantum boundary, raising
            # ``MultipleQuantumSegmentsError`` for valid pure-quantum
            # kernels (e.g. a symbolic multi-control phase inside a loop).
            #
            # ``_absorbable_op_ids`` (computed in :meth:`_compute_absorbable`)
            # holds exactly the ops for which absorption is safe: non-measurement
            # classical ops whose results are consumed *only* by quantum/hybrid
            # ops (directly or through other absorbed classical ops) and are not
            # block outputs. Ops read back by a classical segment or returned
            # stay classical so the executor runs them; if that forces a split
            # the user gets an explicit error rather than a dropped value.
            if (
                op_kind == OperationKind.CLASSICAL
                and current_kind == OperationKind.QUANTUM
                and id(op) in self._absorbable_op_ids
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

            current_ops.append(op)

        # Flush final segment
        if current_ops:
            segment = self._create_segment(current_kind, current_ops)
            segments.append(segment)

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
                if op.operation_kind in (
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

        A top-level classical op is *absorbable* when it is non-measurement,
        feeds a quantum gate, is not a block output, and — transitively — every
        consumer of its results is a quantum / hybrid op or another absorbable
        classical op. This greatest-fixpoint keeps a chain of gate-angle
        expressions (e.g. ``(phase * 2) - 1``) absorbable while refusing to
        absorb a value that is also read back by later classical work or
        returned, which would be dropped by the (classical-op-free) quantum
        segment at execution time.

        Args:
            operations (list[Operation]): Top-level operations of the block.

        Returns:
            set[int]: ``id()`` of every top-level classical op that may be
                absorbed into the surrounding quantum segment.
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

        # Map every op (nested or top-level) to its top-level ancestor. A
        # consumer's segment is decided by the top-level op that contains it:
        # an op nested inside a quantum-effective control-flow body (e.g. a
        # ``BinOp`` inside ``qmc.range``) rides in the quantum segment, not a
        # classical one. Classifying a nested consumer by its own kind would
        # wrongly treat such a chain link as a classical sink and reject a
        # valid pure-quantum nested angle expression.
        top_level_owner = self._build_top_level_owner(operations)

        # Seed candidates: non-measurement, quantum-feeding, non-output
        # top-level classical ops.
        absorbable: set[int] = {
            id(op)
            for op in operations
            if op.operation_kind == OperationKind.CLASSICAL
            and not self._is_measurement_tainted(op)
            and self._feeds_quantum(op)
            and not self._produces_block_output(op)
        }

        # Iteratively drop any candidate that has a result consumed by a
        # classical sink — a consumer that is neither a quantum/hybrid op nor a
        # (still-)absorbable classical op. Removing one op can disqualify the
        # ops feeding it, so iterate to a fixpoint.
        changed = True
        while changed:
            changed = False
            for op in operations:
                if id(op) not in absorbable:
                    continue
                if self._has_classical_sink(
                    op, consumers, absorbable, top_level_owner
                ):
                    absorbable.discard(id(op))
                    changed = True
        return absorbable

    def _build_top_level_owner(
        self, operations: list[Operation]
    ) -> dict[int, Operation]:
        """Map every operation to the top-level operation that contains it.

        Walks each top-level op and, recursively, the bodies of any nested
        control-flow op, recording the top-level ancestor for every
        descendant. A top-level op maps to itself. Used to decide which
        segment a (possibly nested) consumer executes in.

        Args:
            operations (list[Operation]): Top-level operations of the block.

        Returns:
            dict[int, Operation]: ``id(op) -> top-level ancestor op`` for every
                op reachable from ``operations`` (including the top-level ops
                themselves).
        """
        owner: dict[int, Operation] = {}

        def record(op: Operation, top: Operation) -> None:
            """Record ``top`` as the top-level owner of ``op`` and its descendants.

            Args:
                op (Operation): The current (possibly nested) operation.
                top (Operation): The top-level ancestor of ``op``.

            Returns:
                None: Mutates the enclosing ``owner`` map in place.
            """
            owner[id(op)] = top
            if isinstance(op, HasNestedOps):
                for body in op.nested_op_lists():
                    for inner in body:
                        record(inner, top)

        for op in operations:
            record(op, op)
        return owner

    def _has_classical_sink(
        self,
        op: Operation,
        consumers: dict[str, list[Operation]],
        absorbable: set[int],
        top_level_owner: dict[int, Operation],
    ) -> bool:
        """Return whether any result of ``op`` is read by a classical sink.

        A classical sink is a consumer that will run in a classical segment
        (measurement post-processing, a non-absorbable classical expression,
        classical control flow) and would therefore fail to read a value buried
        in a quantum segment. The segment a consumer runs in is decided by its
        top-level ancestor: a consumer nested inside a quantum/hybrid-effective
        control-flow body rides in the quantum segment (not a sink), and a
        top-level absorbable classical op joins the quantum segment too. Only
        consumers owned by a genuinely classical top-level op are sinks.

        Args:
            op (Operation): The candidate op whose results are checked.
            consumers (dict[str, list[Operation]]): Map from value UUID to the
                operations that read it.
            absorbable (set[int]): ``id()`` of ops still considered absorbable
                in the current fixpoint iteration.
            top_level_owner (dict[int, Operation]): Map from ``id(op)`` to the
                top-level op that contains it, as built by
                :meth:`_build_top_level_owner`.

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
                owner = top_level_owner.get(id(consumer), consumer)
                if self._effective_kind(owner) in (
                    OperationKind.QUANTUM,
                    OperationKind.HYBRID,
                ):
                    continue
                if id(owner) in absorbable:
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
