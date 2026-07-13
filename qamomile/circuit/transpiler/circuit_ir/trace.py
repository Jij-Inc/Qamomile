"""Immutable drawing provenance captured beside flat circuit IR.

The executable :class:`~qamomile.circuit.transpiler.circuit_ir.model.CircuitProgram`
remains the sole semantic source of truth.  These nodes only retain source regions
that the shared lowering walk must specialize, such as value-dependent unrolled
loops and inlined qkernel calls.  Every trace region can therefore be flattened
back to the exact instruction tuple it annotates.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, TypeAlias

from qamomile.circuit.transpiler.circuit_ir.model import (
    CircuitBuilder,
    CircuitInstruction,
    CircuitProgram,
    ForInstruction,
    IfInstruction,
    ScalarExpr,
    WhileInstruction,
)


class SpecializedLoopKind(enum.Enum):
    """Classify source loops that lowering specialized iteration by iteration."""

    RANGE = "range"
    ITEMS = "items"


@dataclasses.dataclass(frozen=True)
class RangeLoopOrigin:
    """Describe one concretely specialized range loop.

    Args:
        variable (str): Source induction-variable display name.
        indexset (range): Exact concrete values visited by lowering.
    """

    variable: str
    indexset: range


@dataclasses.dataclass(frozen=True)
class ItemsLoopOrigin:
    """Describe one concretely specialized dictionary-items loop.

    Args:
        key_variables (tuple[str, ...]): Source key-variable display names.
        value_variable (str): Source value-variable display name.
    """

    key_variables: tuple[str, ...]
    value_variable: str


SpecializedLoopOrigin: TypeAlias = RangeLoopOrigin | ItemsLoopOrigin


@dataclasses.dataclass(frozen=True)
class TraceRegion:
    """Store an immutable ordered drawing-provenance region.

    Args:
        nodes (tuple[TraceNode, ...]): Source-aware nodes in execution order.
    """

    nodes: tuple[TraceNode, ...]

    def flatten(self) -> tuple[CircuitInstruction, ...]:
        """Flatten provenance to its authoritative circuit instructions.

        Returns:
            tuple[CircuitInstruction, ...]: Exact instruction sequence covered
                by this region.
        """
        result: list[CircuitInstruction] = []
        for node in self.nodes:
            result.extend(node.flatten())
        return tuple(result)


@dataclasses.dataclass(frozen=True)
class TracedInstruction:
    """Annotate one ordinary circuit instruction and its structured regions.

    Args:
        instruction (CircuitInstruction): Authoritative circuit instruction.
        regions (tuple[TraceRegion, ...]): Traces for structured child regions,
            ordered like the instruction's body fields. Defaults to empty.
    """

    instruction: CircuitInstruction
    regions: tuple[TraceRegion, ...] = ()

    def flatten(self) -> tuple[CircuitInstruction, ...]:
        """Return the single authoritative instruction represented here.

        Returns:
            tuple[CircuitInstruction, ...]: One-element instruction tuple.
        """
        return (self.instruction,)


@dataclasses.dataclass(frozen=True)
class SpecializedLoopIteration:
    """Retain the exact emitted body of one specialized iteration.

    Args:
        value_label (str): Deterministic source-value label for display.
        region (TraceRegion): Exact instructions emitted for this iteration.
    """

    value_label: str
    region: TraceRegion


@dataclasses.dataclass(frozen=True)
class SpecializedLoopTrace:
    """Group flat emitted instructions by their source loop iterations.

    Args:
        kind (SpecializedLoopKind): Range or dictionary-items loop category.
        origin (SpecializedLoopOrigin): Immutable source loop description.
        iterations (tuple[SpecializedLoopIteration, ...]): Exact concrete
            iterations in lowering order.
    """

    kind: SpecializedLoopKind
    origin: SpecializedLoopOrigin
    iterations: tuple[SpecializedLoopIteration, ...]

    def flatten(self) -> tuple[CircuitInstruction, ...]:
        """Concatenate exact per-iteration instructions.

        Returns:
            tuple[CircuitInstruction, ...]: Flat lowering result.
        """
        result: list[CircuitInstruction] = []
        for iteration in self.iterations:
            result.extend(iteration.region.flatten())
        return tuple(result)


@dataclasses.dataclass(frozen=True)
class InlineRegionTrace:
    """Retain a source qkernel boundary erased by normal inlining.

    Args:
        label (str): Source callable display name.
        region_id (int): Deterministic lowering-local region identity.
        call_arguments (tuple[tuple[str, ScalarExpr], ...]): Ordered scalar
            actual arguments retained for the collapsed call label.
        quantum_slots (tuple[int, ...]): Exact source-call quantum interface
            slots in first-occurrence input/output order.
        region (TraceRegion): Exact instructions emitted inside the call.
    """

    label: str
    region_id: int
    call_arguments: tuple[tuple[str, ScalarExpr], ...]
    quantum_slots: tuple[int, ...]
    region: TraceRegion

    def flatten(self) -> tuple[CircuitInstruction, ...]:
        """Return exact instructions emitted inside the source call.

        Returns:
            tuple[CircuitInstruction, ...]: Flat inlined call body.
        """
        return self.region.flatten()


TraceNode: TypeAlias = TracedInstruction | SpecializedLoopTrace | InlineRegionTrace


@dataclasses.dataclass(frozen=True)
class CircuitProgramTrace:
    """Store drawing-only provenance for one verified circuit program.

    Args:
        root (TraceRegion): Root source-aware instruction region.
    """

    root: TraceRegion


@dataclasses.dataclass
class _SpecializedLoopContext:
    """Track mutable construction of one specialized loop trace.

    Args:
        kind (SpecializedLoopKind): Specialized loop category.
        origin (SpecializedLoopOrigin): Source loop description.
        iterations (list[SpecializedLoopIteration]): Completed iterations.
        active_label (str | None): Current iteration label, if open.
    """

    kind: SpecializedLoopKind
    origin: SpecializedLoopOrigin
    iterations: list[SpecializedLoopIteration] = dataclasses.field(default_factory=list)
    active_label: str | None = None


@dataclasses.dataclass
class _InlineRegionContext:
    """Track mutable construction of one inlined source region.

    Args:
        label (str): Source callable display name.
        region_id (int): Deterministic lowering-local identity.
        call_arguments (tuple[tuple[str, ScalarExpr], ...]): Ordered scalar
            arguments resolved at the start marker.
        quantum_slots (list[int]): Exact quantum interface slots accumulated
            from start inputs and end outputs.
    """

    label: str
    region_id: int
    call_arguments: tuple[tuple[str, ScalarExpr], ...]
    quantum_slots: list[int]


class _TracingOperationList(list[CircuitInstruction]):
    """Record every semantic-list append into its owning trace builder.

    Args:
        owner (TracingCircuitBuilder): Builder receiving append callbacks.
    """

    def __init__(self, owner: TracingCircuitBuilder) -> None:
        """Initialize an empty callback list.

        Args:
            owner (TracingCircuitBuilder): Builder receiving append callbacks.
        """
        super().__init__()
        self._owner = owner

    def append(self, instruction: CircuitInstruction) -> None:
        """Append an instruction and record its source-aware trace node.

        Args:
            instruction (CircuitInstruction): Instruction to append.
        """
        super().append(instruction)
        self._owner._record_instruction(instruction)


class TracingCircuitBuilder(CircuitBuilder):
    """Build ordinary circuit IR while recording drawing-only provenance.

    The inherited operation lists remain byte-for-byte equivalent to a normal
    :class:`CircuitBuilder`.  A parallel target stack groups those same
    instructions into specialized loops and source inline regions.

    Args:
        num_qubits (int): Number of virtual qubit slots.
        num_clbits (int): Number of classical bit slots.
        name (str): Circuit name. Defaults to ``"main"``.
    """

    def __init__(self, num_qubits: int, num_clbits: int, name: str = "main") -> None:
        """Initialize circuit and trace construction state.

        Args:
            num_qubits (int): Number of virtual qubit slots.
            num_clbits (int): Number of classical bit slots.
            name (str): Circuit name. Defaults to ``"main"``.
        """
        super().__init__(num_qubits, num_clbits, name)
        root_nodes: list[TraceNode] = []
        self._trace_targets: list[list[TraceNode]] = [root_nodes]
        self._regions[0].operations = _TracingOperationList(self)
        self._pending_regions: tuple[TraceRegion, ...] | None = None
        self._if_true_traces: dict[int, TraceRegion] = {}
        self._specialized_loops: list[_SpecializedLoopContext] = []
        self._inline_regions: list[_InlineRegionContext] = []

    def _open_structured_region(self) -> None:
        """Replace the current empty operation list and open its trace target.

        Raises:
            RuntimeError: If the inherited builder did not open an empty
                structured region.
        """
        if self._regions[-1].operations:
            raise RuntimeError("A newly opened structured region is not empty")
        target: list[TraceNode] = []
        self._regions[-1].operations = _TracingOperationList(self)
        self._trace_targets.append(target)

    def _close_trace_target(self) -> TraceRegion:
        """Close and freeze the innermost non-root trace target.

        Returns:
            TraceRegion: Immutable completed trace region.

        Raises:
            RuntimeError: If no nested trace target is open.
        """
        if len(self._trace_targets) <= 1:
            raise RuntimeError("No nested trace target is open")
        return TraceRegion(tuple(self._trace_targets.pop()))

    def _record_instruction(self, instruction: CircuitInstruction) -> None:
        """Record one appended instruction in the active trace region.

        Args:
            instruction (CircuitInstruction): Newly appended instruction.
        """
        regions = self._pending_regions or ()
        self._pending_regions = None
        self._trace_targets[-1].append(TracedInstruction(instruction, regions))

    def begin_for(self, indexset: range) -> Any:
        """Open a native structured for loop and its child trace.

        Args:
            indexset (range): Concrete native-loop range.

        Returns:
            Any: Inherited loop-variable expression.
        """
        variable = super().begin_for(indexset)
        self._open_structured_region()
        return variable

    def end_for(self) -> None:
        """Close a native structured for loop and attach its body trace."""
        self._pending_regions = (self._close_trace_target(),)
        super().end_for()

    def begin_if(self, condition: Any) -> Any:
        """Open a structured conditional and its true-branch trace.

        Args:
            condition (Any): Target-neutral scalar condition.

        Returns:
            Any: Inherited conditional context token.
        """
        context = super().begin_if(condition)
        self._open_structured_region()
        return context

    def begin_else(self, context: Any) -> None:
        """Switch a structured conditional to its false-branch trace.

        Args:
            context (Any): Inherited conditional context token.
        """
        self._if_true_traces[id(context)] = self._close_trace_target()
        super().begin_else(context)
        self._open_structured_region()

    def end_if(self, context: Any) -> None:
        """Close a structured conditional and attach both branch traces.

        Args:
            context (Any): Inherited conditional context token.
        """
        current = self._close_trace_target()
        true_trace = self._if_true_traces.pop(id(context), None)
        if true_trace is None:
            true_trace = current
            false_trace = TraceRegion(())
        else:
            false_trace = current
        self._pending_regions = (true_trace, false_trace)
        super().end_if(context)

    def begin_while(self, condition: Any) -> Any:
        """Open a structured while loop and its child trace.

        Args:
            condition (Any): Target-neutral scalar condition.

        Returns:
            Any: Inherited while-loop context token.
        """
        context = super().begin_while(condition)
        self._open_structured_region()
        return context

    def end_while(self, context: Any) -> None:
        """Close a structured while loop and attach its body trace.

        Args:
            context (Any): Inherited while-loop context token.
        """
        self._pending_regions = (self._close_trace_target(),)
        super().end_while(context)

    def begin_specialized_loop(
        self,
        kind: SpecializedLoopKind,
        origin: SpecializedLoopOrigin,
    ) -> _SpecializedLoopContext:
        """Open a source loop that will emit distinct flat iterations.

        Args:
            kind (SpecializedLoopKind): Specialized loop category.
            origin (SpecializedLoopOrigin): Source loop description.

        Returns:
            _SpecializedLoopContext: Opaque context token.
        """
        context = _SpecializedLoopContext(kind, origin)
        self._specialized_loops.append(context)
        return context

    def begin_specialized_iteration(
        self,
        context: _SpecializedLoopContext,
        value_label: str,
    ) -> None:
        """Open one exact specialized-loop iteration.

        Args:
            context (_SpecializedLoopContext): Innermost specialized loop.
            value_label (str): Deterministic iteration-value label.

        Raises:
            RuntimeError: If contexts are unbalanced or an iteration is open.
        """
        if not self._specialized_loops or self._specialized_loops[-1] is not context:
            raise RuntimeError("Specialized loop context is not innermost")
        if context.active_label is not None:
            raise RuntimeError("A specialized loop iteration is already open")
        context.active_label = value_label
        self._trace_targets.append([])

    def end_specialized_iteration(
        self,
        context: _SpecializedLoopContext,
    ) -> None:
        """Close one exact specialized-loop iteration.

        Args:
            context (_SpecializedLoopContext): Innermost specialized loop.

        Raises:
            RuntimeError: If no matching iteration is open.
        """
        if not self._specialized_loops or self._specialized_loops[-1] is not context:
            raise RuntimeError("Specialized loop context is not innermost")
        if context.active_label is None:
            raise RuntimeError("No specialized loop iteration is open")
        region = self._close_trace_target()
        context.iterations.append(
            SpecializedLoopIteration(context.active_label, region)
        )
        context.active_label = None

    def end_specialized_loop(self, context: _SpecializedLoopContext) -> None:
        """Close a specialized loop and append one grouped trace node.

        Args:
            context (_SpecializedLoopContext): Innermost specialized loop.

        Raises:
            RuntimeError: If contexts are unbalanced or an iteration is open.
        """
        if not self._specialized_loops or self._specialized_loops[-1] is not context:
            raise RuntimeError("Specialized loop context is not innermost")
        if context.active_label is not None:
            raise RuntimeError("Cannot close a specialized loop iteration")
        self._specialized_loops.pop()
        self._trace_targets[-1].append(
            SpecializedLoopTrace(
                kind=context.kind,
                origin=context.origin,
                iterations=tuple(context.iterations),
            )
        )

    def begin_inline_region(
        self,
        label: str,
        region_id: int,
        call_arguments: tuple[tuple[str, ScalarExpr], ...],
        quantum_slots: tuple[int, ...],
    ) -> _InlineRegionContext:
        """Open one source qkernel region erased from executable IR.

        Args:
            label (str): Source callable display name.
            region_id (int): Deterministic lowering-local identity.
            call_arguments (tuple[tuple[str, ScalarExpr], ...]): Ordered
                resolved scalar actual arguments.
            quantum_slots (tuple[int, ...]): Exact quantum input slots.

        Returns:
            _InlineRegionContext: Opaque region token.
        """
        context = _InlineRegionContext(
            label,
            region_id,
            call_arguments,
            list(dict.fromkeys(quantum_slots)),
        )
        self._inline_regions.append(context)
        self._trace_targets.append([])
        return context

    def end_inline_region(
        self,
        label: str,
        region_id: int,
        quantum_slots: tuple[int, ...],
    ) -> None:
        """Close one source qkernel region and group its exact instructions.

        Args:
            label (str): Source callable display name.
            region_id (int): Deterministic lowering-local identity.
            quantum_slots (tuple[int, ...]): Exact quantum output slots.

        Raises:
            RuntimeError: If start/end marker nesting does not match.
        """
        if not self._inline_regions:
            raise RuntimeError("No inline source region is open")
        context = self._inline_regions.pop()
        if (context.label, context.region_id) != (label, region_id):
            raise RuntimeError("Inline source region markers are unbalanced")
        for slot in quantum_slots:
            if slot not in context.quantum_slots:
                context.quantum_slots.append(slot)
        region = self._close_trace_target()
        self._trace_targets[-1].append(
            InlineRegionTrace(
                context.label,
                context.region_id,
                context.call_arguments,
                tuple(context.quantum_slots),
                region,
            )
        )

    def freeze_trace(self, program: CircuitProgram) -> CircuitProgramTrace:
        """Freeze and verify provenance against an authoritative program.

        Args:
            program (CircuitProgram): Program produced by :meth:`freeze`.

        Returns:
            CircuitProgramTrace: Immutable verified drawing provenance.

        Raises:
            RuntimeError: If any trace region remains open.
            ValueError: If trace flattening differs from ``program``.
        """
        if (
            len(self._trace_targets) != 1
            or self._specialized_loops
            or self._inline_regions
            or self._pending_regions is not None
        ):
            raise RuntimeError("Cannot freeze a circuit trace with open regions")
        trace = CircuitProgramTrace(TraceRegion(tuple(self._trace_targets[0])))
        verify_circuit_trace(program, trace)
        return trace


def verify_circuit_trace(
    program: CircuitProgram,
    trace: CircuitProgramTrace,
) -> None:
    """Verify that provenance is lossless and structurally aligned.

    Args:
        program (CircuitProgram): Authoritative target-neutral circuit.
        trace (CircuitProgramTrace): Drawing provenance to validate.

    Raises:
        ValueError: If any trace region flattens differently from its circuit
            instruction region or has the wrong structured-child arity.
    """
    _verify_region(trace.root, program.operations, path="root")


def _verify_region(
    region: TraceRegion,
    expected: tuple[CircuitInstruction, ...],
    *,
    path: str,
) -> None:
    """Verify one trace region and all structured descendants recursively.

    Args:
        region (TraceRegion): Trace region to verify.
        expected (tuple[CircuitInstruction, ...]): Exact authoritative body.
        path (str): Diagnostic location.

    Raises:
        ValueError: If flattened instructions or child regions differ.
    """
    if region.flatten() != expected:
        raise ValueError(f"Circuit trace does not match instructions at {path}")
    for node_index, node in enumerate(region.nodes):
        node_path = f"{path}/{node_index}"
        if isinstance(node, SpecializedLoopTrace):
            for iteration_index, iteration in enumerate(node.iterations):
                _verify_region(
                    iteration.region,
                    iteration.region.flatten(),
                    path=f"{node_path}/iteration/{iteration_index}",
                )
            continue
        if isinstance(node, InlineRegionTrace):
            _verify_region(
                node.region,
                node.region.flatten(),
                path=f"{node_path}/inline",
            )
            continue
        instruction = node.instruction
        if isinstance(instruction, ForInstruction):
            expected_regions = (instruction.body,)
        elif isinstance(instruction, IfInstruction):
            expected_regions = (instruction.true_body, instruction.false_body)
        elif isinstance(instruction, WhileInstruction):
            expected_regions = (instruction.body,)
        else:
            expected_regions = ()
        if len(node.regions) != len(expected_regions):
            raise ValueError(
                f"Circuit trace structured-region arity differs at {node_path}"
            )
        for child_index, (child, expected_child) in enumerate(
            zip(node.regions, expected_regions, strict=True)
        ):
            _verify_region(
                child,
                expected_child,
                path=f"{node_path}/region/{child_index}",
            )
