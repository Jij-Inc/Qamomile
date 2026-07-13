"""Convert target-neutral circuit IR into renderer-ready Visual IR.

The adapter is deliberately independent of frontend ``Block`` objects. It
uses the verified linear wire lineage carried by :class:`CircuitProgram` and
therefore never guesses an unresolved qubit, slice, or callable operand.
"""

from __future__ import annotations

import dataclasses
import math
import operator
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from qamomile._utils import is_close_zero
from qamomile.circuit.ir.operation.gate import GateOperationType
from qamomile.circuit.transpiler.circuit_ir.model import (
    BarrierInstruction,
    BinaryExpr,
    BinaryOperator,
    CallInstruction,
    CircuitInstruction,
    CircuitProgram,
    ClassicalBitExpr,
    ForInstruction,
    GateInstruction,
    IfInstruction,
    LiteralExpr,
    LoopVariableExpr,
    MeasureInstruction,
    MeasureVectorInstruction,
    ParameterExpr,
    PauliEvolutionInstruction,
    ResetInstruction,
    ReusableCircuit,
    ScalarExpr,
    UnaryExpr,
    UnaryOperator,
    WhileInstruction,
    WireId,
)
from qamomile.circuit.transpiler.circuit_ir.trace import (
    CircuitProgramTrace,
    InlineRegionTrace,
    ItemsLoopOrigin,
    RangeLoopOrigin,
    SpecializedLoopKind,
    SpecializedLoopTrace,
    TracedInstruction,
    TraceRegion,
    verify_circuit_trace,
)
from qamomile.circuit.transpiler.circuit_ir.verify import verify_circuit
from qamomile.circuit.transpiler.gate_emitter import GateKind

from .geometry import compute_border_padding
from .style import DEFAULT_STYLE, CircuitStyle
from .text_metrics import measure_text_width
from .visual_ir import (
    VFoldedBlock,
    VFoldedKind,
    VGate,
    VGateKind,
    VInlineBlock,
    VisualCircuit,
    VisualNode,
    VSkip,
    VUnfoldedKind,
    VUnfoldedSequence,
)

__all__ = ["CircuitProgramAdapter", "circuit_program_to_visual_ir"]


_GATE_TYPES: dict[GateKind, GateOperationType] = {
    GateKind.H: GateOperationType.H,
    GateKind.X: GateOperationType.X,
    GateKind.Y: GateOperationType.Y,
    GateKind.Z: GateOperationType.Z,
    GateKind.S: GateOperationType.S,
    GateKind.SDG: GateOperationType.SDG,
    GateKind.T: GateOperationType.T,
    GateKind.TDG: GateOperationType.TDG,
    GateKind.RX: GateOperationType.RX,
    GateKind.RY: GateOperationType.RY,
    GateKind.RZ: GateOperationType.RZ,
    GateKind.P: GateOperationType.P,
    GateKind.CX: GateOperationType.CX,
    GateKind.CZ: GateOperationType.CZ,
    GateKind.SWAP: GateOperationType.SWAP,
    GateKind.CP: GateOperationType.CP,
    GateKind.RZZ: GateOperationType.RZZ,
    GateKind.TOFFOLI: GateOperationType.TOFFOLI,
}

_CONTROLLED_TARGET_TYPES: dict[GateKind, GateOperationType] = {
    GateKind.CH: GateOperationType.H,
    GateKind.CY: GateOperationType.Y,
    GateKind.CRX: GateOperationType.RX,
    GateKind.CRY: GateOperationType.RY,
    GateKind.CRZ: GateOperationType.RZ,
}

_GATE_LABELS: dict[GateKind, str] = {
    GateKind.H: r"$H$",
    GateKind.X: r"$X$",
    GateKind.Y: r"$Y$",
    GateKind.Z: r"$Z$",
    GateKind.S: r"$S$",
    GateKind.SDG: r"$S^{\dagger}$",
    GateKind.T: r"$T$",
    GateKind.TDG: r"$T^{\dagger}$",
    GateKind.RX: r"$R_x$",
    GateKind.RY: r"$R_y$",
    GateKind.RZ: r"$R_z$",
    GateKind.P: r"$P$",
    GateKind.CX: r"$CX$",
    GateKind.CZ: r"$CZ$",
    GateKind.SWAP: r"$SWAP$",
    GateKind.CP: r"$CP$",
    GateKind.RZZ: r"$R_{zz}$",
    GateKind.TOFFOLI: r"$CCX$",
    GateKind.CH: r"$H$",
    GateKind.CY: r"$Y$",
    GateKind.CRX: r"$R_x$",
    GateKind.CRY: r"$R_y$",
    GateKind.CRZ: r"$R_z$",
}

_GREEK_PARAMETER_NAMES = frozenset(
    {
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "pi",
        "rho",
        "sigma",
        "tau",
        "upsilon",
        "phi",
        "chi",
        "psi",
        "omega",
    }
)

_BINARY_SYMBOLS: dict[BinaryOperator, str] = {
    BinaryOperator.ADD: "+",
    BinaryOperator.SUB: "-",
    BinaryOperator.MUL: "*",
    BinaryOperator.DIV: "/",
    BinaryOperator.FLOORDIV: "//",
    BinaryOperator.MOD: "%",
    BinaryOperator.POW: "**",
    BinaryOperator.EQ: "==",
    BinaryOperator.NEQ: "!=",
    BinaryOperator.LT: "<",
    BinaryOperator.LE: "<=",
    BinaryOperator.GT: ">",
    BinaryOperator.GE: ">=",
    BinaryOperator.AND: "and",
    BinaryOperator.OR: "or",
}

_LITERAL_BINARY_OPERATORS: dict[
    BinaryOperator, Callable[[bool | int | float, bool | int | float], Any]
] = {
    BinaryOperator.ADD: operator.add,
    BinaryOperator.SUB: operator.sub,
    BinaryOperator.MUL: operator.mul,
    BinaryOperator.DIV: operator.truediv,
    BinaryOperator.FLOORDIV: operator.floordiv,
    BinaryOperator.MOD: operator.mod,
    BinaryOperator.POW: operator.pow,
    BinaryOperator.EQ: operator.eq,
    BinaryOperator.NEQ: operator.ne,
    BinaryOperator.LT: operator.lt,
    BinaryOperator.LE: operator.le,
    BinaryOperator.GT: operator.gt,
    BinaryOperator.GE: operator.ge,
    BinaryOperator.AND: lambda left, right: bool(left) and bool(right),
    BinaryOperator.OR: lambda left, right: bool(left) or bool(right),
}

_NodeKey = tuple[str | int, ...]


@dataclasses.dataclass(frozen=True)
class _MeasurementWriter:
    """Identify one visible static measurement producing a classical bit.

    Args:
        node_key (_NodeKey): Stable Visual IR node identity.
        qubit_index (int): Display slot measured by the node.
        visible (bool): Whether the measurement is present in the visual tree.
    """

    node_key: _NodeKey
    qubit_index: int
    visible: bool


_WriterMap = dict[int, set[_MeasurementWriter]]


class _WireTopology:
    """Resolve every module-local wire version to an immutable display slot."""

    def __init__(self, program: CircuitProgram, input_slots: Sequence[int]) -> None:
        """Build an exact wire-version topology for one circuit module.

        Args:
            program (CircuitProgram): Verified circuit module to index.
            input_slots (Sequence[int]): Display slot corresponding to each
                module input wire.

        Raises:
            ValueError: If input arity differs or an instruction references an
                unmapped wire despite circuit verification.
        """
        if len(input_slots) != program.num_qubits:
            raise ValueError(
                "Circuit input slot count does not match program.num_qubits"
            )
        self._slots = dict(zip(program.input_wires, input_slots, strict=True))
        self._visit_operations(program.operations)

    @property
    def slots(self) -> Mapping[WireId, int]:
        """Return the complete immutable-by-contract lineage mapping.

        Returns:
            Mapping[WireId, int]: Wire versions mapped to display slots.
        """
        return self._slots

    def slot(self, wire: WireId) -> int:
        """Return the display slot for one wire version.

        Args:
            wire (WireId): Module-local wire version.

        Returns:
            int: Exact display slot.

        Raises:
            ValueError: If ``wire`` is absent from the verified lineage.
        """
        try:
            return self._slots[wire]
        except KeyError as error:
            raise ValueError(
                f"Wire {wire.value} has no display-slot lineage"
            ) from error

    def slots_for(self, wires: Sequence[WireId]) -> list[int]:
        """Resolve an ordered wire sequence without dropping operands.

        Args:
            wires (Sequence[WireId]): Wire versions in semantic operand order.

        Returns:
            list[int]: Display slots in the same order.

        Raises:
            ValueError: If any wire lacks a lineage entry.
        """
        return [self.slot(wire) for wire in wires]

    def _publish_outputs(
        self,
        inputs: Sequence[WireId],
        outputs: Sequence[WireId],
    ) -> None:
        """Propagate slot identity through a linear instruction.

        Args:
            inputs (Sequence[WireId]): Consumed wire versions.
            outputs (Sequence[WireId]): Positionally corresponding outputs.

        Raises:
            ValueError: If arities differ, an input is unknown, or an output
                was already assigned.
        """
        if len(inputs) != len(outputs):
            raise ValueError("Circuit instruction wire arities do not match")
        for input_wire, output_wire in zip(inputs, outputs, strict=True):
            if output_wire in self._slots:
                raise ValueError(f"Wire {output_wire.value} has duplicate lineage")
            self._slots[output_wire] = self.slot(input_wire)

    def _visit_operations(
        self,
        operations: Sequence[CircuitInstruction],
    ) -> None:
        """Index all wire definitions in one structured region.

        Args:
            operations (Sequence[CircuitInstruction]): Region instructions.

        Raises:
            TypeError: If the closed CircuitInstruction union is extended
                without an adapter implementation.
            ValueError: If a verified wire invariant is not satisfied.
        """
        for instruction in operations:
            if isinstance(instruction, GateInstruction):
                self._publish_outputs(instruction.inputs, instruction.outputs)
            elif isinstance(instruction, MeasureInstruction):
                self._publish_outputs((instruction.input,), (instruction.output,))
            elif isinstance(instruction, MeasureVectorInstruction):
                self._publish_outputs(instruction.inputs, instruction.outputs)
            elif isinstance(instruction, ResetInstruction):
                self._publish_outputs((instruction.input,), (instruction.output,))
            elif isinstance(instruction, BarrierInstruction):
                self.slots_for(instruction.wires)
            elif isinstance(instruction, PauliEvolutionInstruction):
                self._publish_outputs(instruction.inputs, instruction.outputs)
            elif isinstance(instruction, CallInstruction):
                self._publish_outputs(instruction.inputs, instruction.outputs)
            elif isinstance(instruction, ForInstruction):
                self._visit_operations(instruction.body)
                self._publish_outputs(instruction.inputs, instruction.outputs)
            elif isinstance(instruction, IfInstruction):
                self._visit_operations(instruction.true_body)
                self._visit_operations(instruction.false_body)
                self._publish_outputs(instruction.inputs, instruction.outputs)
            elif isinstance(instruction, WhileInstruction):
                self._visit_operations(instruction.body)
                self._publish_outputs(instruction.inputs, instruction.outputs)
            else:
                raise TypeError(
                    f"Unsupported CircuitInstruction: {type(instruction).__name__}"
                )


def _copy_writers(writers: _WriterMap) -> _WriterMap:
    """Copy a classical reaching-definition map.

    Args:
        writers (_WriterMap): Map to copy.

    Returns:
        _WriterMap: Independent sets for branch-local mutation.
    """
    return {clbit: set(sources) for clbit, sources in writers.items()}


def _merge_writers(*writer_maps: _WriterMap) -> _WriterMap:
    """Join classical reaching definitions from alternative paths.

    Args:
        *writer_maps (_WriterMap): Path states to union.

    Returns:
        _WriterMap: Union of writers for every classical slot.
    """
    merged: _WriterMap = {}
    for writer_map in writer_maps:
        for clbit, sources in writer_map.items():
            merged.setdefault(clbit, set()).update(sources)
    return merged


def _proxy_changed_writers(
    entry: _WriterMap,
    exit: _WriterMap,
    node_key: _NodeKey,
    *,
    visible: bool,
) -> _WriterMap:
    """Expose changed reaching definitions through a collapsed container.

    Args:
        entry (_WriterMap): Reaching definitions before the container.
        exit (_WriterMap): Exact reaching definitions after its hidden body.
        node_key (_NodeKey): Visible container identity.
        visible (bool): Whether this container is present in the visual tree.

    Returns:
        _WriterMap: Exit state where only definitions changed by the body are
            attributed to the container boundary. Unchanged definitions keep
            their original identity.
    """
    proxied = _copy_writers(exit)
    for clbit, sources in exit.items():
        if sources == entry.get(clbit, set()):
            continue
        proxied[clbit] = {
            _MeasurementWriter(node_key, source.qubit_index, visible)
            for source in sources
        }
    return proxied


def _format_scalar(
    expression: ScalarExpr,
    loop_values: Mapping[str, int] | None = None,
) -> str:
    """Format a complete circuit scalar expression without semantic loss.

    Args:
        expression (ScalarExpr): Expression to format.
        loop_values (Mapping[str, int] | None): Concrete induction values for
            an unrolled visual iteration. Defaults to ``None``.

    Returns:
        str: Fully parenthesized, deterministic expression text.

    Raises:
        TypeError: If the closed ScalarExpr union is extended without a
            formatter implementation.
    """
    if isinstance(expression, LiteralExpr):
        return repr(expression.value)
    if isinstance(expression, ParameterExpr):
        return expression.name
    if isinstance(expression, ClassicalBitExpr):
        return f"c[{expression.index}]"
    if isinstance(expression, LoopVariableExpr):
        if loop_values is not None and expression.name in loop_values:
            return str(loop_values[expression.name])
        return expression.name
    if isinstance(expression, UnaryExpr):
        operand = _format_scalar(expression.operand, loop_values)
        if expression.operator is UnaryOperator.NOT:
            return f"not ({operand})"
        if expression.operator is UnaryOperator.NEG:
            return f"-({operand})"
        raise TypeError(f"Unsupported unary operator: {expression.operator}")
    if isinstance(expression, BinaryExpr):
        left = _format_scalar(expression.left, loop_values)
        right = _format_scalar(expression.right, loop_values)
        try:
            symbol = _BINARY_SYMBOLS[expression.operator]
        except KeyError as error:
            raise TypeError(
                f"Unsupported binary operator: {expression.operator}"
            ) from error
        return f"({left} {symbol} {right})"
    raise TypeError(f"Unsupported ScalarExpr: {type(expression).__name__}")


def _format_call_parameter_name(name: str) -> str:
    """Format one call parameter name for a compact box label.

    Args:
        name (str): Source-level formal parameter name.

    Returns:
        str: Mathtext for a Greek name, otherwise the unchanged name.
    """
    if name in _GREEK_PARAMETER_NAMES:
        return rf"$\{name}$"
    return name


def _format_call_argument(
    expression: ScalarExpr,
    loop_values: Mapping[str, int] | None = None,
) -> str:
    """Format one scalar call argument without hiding symbolic meaning.

    Args:
        expression (ScalarExpr): Target-neutral call-site expression.
        loop_values (Mapping[str, int] | None): Concrete induction values for
            an unfolded visual iteration. Defaults to ``None``.

    Returns:
        str: Compact numeric text or a complete symbolic expression.
    """
    value = _literal_value(expression)
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, (int, float)):
        if value == 0 or math.isclose(float(value), 0.0, abs_tol=1e-15):
            return "0"
        magnitude = abs(value)
        if magnitude >= 1000 or magnitude < 0.01:
            return f"{value:.0e}"
        if magnitude >= 10:
            return f"{value:.1f}"
        return f"{value:.2f}"
    if isinstance(expression, ParameterExpr):
        return _format_call_parameter_name(expression.name)
    return _format_scalar(expression, loop_values)


def _classical_bits(expression: ScalarExpr) -> set[int]:
    """Collect classical slots referenced by an expression.

    Args:
        expression (ScalarExpr): Expression to inspect.

    Returns:
        set[int]: Referenced classical bit indices.

    Raises:
        TypeError: If the closed ScalarExpr union is extended unexpectedly.
    """
    if isinstance(expression, ClassicalBitExpr):
        return {expression.index}
    if isinstance(expression, BinaryExpr):
        return _classical_bits(expression.left) | _classical_bits(expression.right)
    if isinstance(expression, UnaryExpr):
        return _classical_bits(expression.operand)
    if isinstance(expression, (LiteralExpr, ParameterExpr, LoopVariableExpr)):
        return set()
    raise TypeError(f"Unsupported ScalarExpr: {type(expression).__name__}")


def _literal_value(expression: ScalarExpr) -> bool | int | float | None:
    """Evaluate a scalar expression only when every leaf is literal.

    Args:
        expression (ScalarExpr): Expression to evaluate.

    Returns:
        bool | int | float | None: Concrete value, or ``None`` when symbolic.

    Raises:
        TypeError: If the closed ScalarExpr union is extended unexpectedly.
    """
    if isinstance(expression, LiteralExpr):
        return expression.value
    if isinstance(expression, (ParameterExpr, ClassicalBitExpr, LoopVariableExpr)):
        return None
    if isinstance(expression, UnaryExpr):
        operand = _literal_value(expression.operand)
        if operand is None:
            return None
        if expression.operator is UnaryOperator.NOT:
            return not bool(operand)
        if expression.operator is UnaryOperator.NEG:
            return -operand
        raise TypeError(f"Unsupported unary operator: {expression.operator}")
    if isinstance(expression, BinaryExpr):
        left = _literal_value(expression.left)
        right = _literal_value(expression.right)
        if left is None or right is None:
            return None
        try:
            evaluator = _LITERAL_BINARY_OPERATORS[expression.operator]
            value = evaluator(left, right)
        except (ArithmeticError, TypeError, ValueError):
            return None
        return value if isinstance(value, (bool, int, float)) else None
    raise TypeError(f"Unsupported ScalarExpr: {type(expression).__name__}")


def _is_provably_zero_phase(expression: ScalarExpr) -> bool:
    """Determine whether a literal phase is numerically zero.

    Args:
        expression (ScalarExpr): Phase expression to inspect.

    Returns:
        bool: ``True`` only for a fully literal value within Qamomile's
            numerical zero tolerance.

    Raises:
        TypeError: If the closed ScalarExpr union is extended unexpectedly.
    """
    literal = _literal_value(expression)
    return literal is not None and is_close_zero(float(literal))


class CircuitProgramAdapter:
    """Build existing Visual IR directly from one verified CircuitProgram."""

    def __init__(
        self,
        program: CircuitProgram,
        *,
        trace: CircuitProgramTrace | None = None,
        style: CircuitStyle | None = None,
        qubit_names: Mapping[int, str] | None = None,
        output_names: Sequence[str] | None = None,
        expectation_value_qubits: Sequence[Sequence[int]] | None = None,
        expand_calls: bool = False,
        inline_depth: int | None = None,
        fold_loops: bool = True,
        fold_ifs: bool = False,
        fold_whiles: bool = False,
    ) -> None:
        """Initialize an exact circuit-to-visual adapter.

        Args:
            program (CircuitProgram): Target-neutral circuit program.
            trace (CircuitProgramTrace | None): Optional lossless drawing-only
                source provenance aligned with ``program``. Defaults to
                ``None``.
            style (CircuitStyle | None): Drawing style used for text metrics.
                Defaults to the standard style.
            qubit_names (Mapping[int, str] | None): Optional display names by
                physical slot. Missing entries use ``q{slot}``.
            output_names (Sequence[str] | None): Optional public output labels.
            expectation_value_qubits (Sequence[Sequence[int]] | None): Exact
                physical slots for symbolic terminal expectation-value boxes.
                Defaults to ``None``.
            expand_calls (bool): Whether source qkernel regions and direct,
                untransformed reusable calls should expose their bodies.
                Bodyless opaque calls always remain boxed. Defaults to
                ``False``.
            inline_depth (int | None): Maximum source/reusable call expansion
                depth. ``None`` permits arbitrary finite nesting.
            fold_loops (bool): Whether concrete for loops use folded boxes.
            fold_ifs (bool): Whether runtime if regions use folded boxes.
            fold_whiles (bool): Whether runtime while regions use folded boxes.

        Raises:
            ValueError: If ``inline_depth`` is negative or circuit structural
                verification fails.
        """
        if inline_depth is not None and inline_depth < 0:
            raise ValueError("inline_depth must be non-negative or None")
        verify_circuit(program)
        if trace is not None:
            verify_circuit_trace(program, trace)
        self.program = program
        self.trace = trace
        self.style = style or DEFAULT_STYLE
        self.qubit_names = {
            slot: (qubit_names or {}).get(slot, f"q{slot}")
            for slot in range(program.num_qubits)
        }
        self.output_names = list(output_names or ())
        self.expectation_value_qubits = tuple(
            tuple(qubits) for qubits in (expectation_value_qubits or ())
        )
        for qubits in self.expectation_value_qubits:
            if not qubits:
                raise ValueError(
                    "Expectation-value drawing requires at least one qubit"
                )
            if any(slot < 0 or slot >= program.num_qubits for slot in qubits):
                raise ValueError(
                    "Expectation-value drawing references a qubit outside "
                    "the verified circuit width"
                )
        self.expand_calls = expand_calls
        self.inline_depth = inline_depth
        self.fold_loops = fold_loops
        self.fold_ifs = fold_ifs
        self.fold_whiles = fold_whiles
        self._topology = _WireTopology(program, range(program.num_qubits))
        self._root_outputs = set(program.output_wires)

    def build(self) -> VisualCircuit:
        """Convert the complete program to Visual IR.

        Returns:
            VisualCircuit: Renderer-ready tree and exact wire metadata.

        Raises:
            TypeError: If the circuit/scalar closed unions contain an unknown
                variant.
            ValueError: If verified wire lineage cannot be resolved.
        """
        children, _ = self._build_nodes(
            self.program.operations,
            self._topology,
            trace_region=self.trace.root if self.trace is not None else None,
            path=("circuit",),
            writers={},
            loop_values={},
            visual_depth=0,
            call_depth=0,
            conditional_depth=0,
            inside_call=False,
            visible=True,
        )
        for index, qubits in enumerate(self.expectation_value_qubits):
            label = "<H>"
            width = self._label_width(label)
            children.append(
                VGate(
                    node_key=("expval", index),
                    label=label,
                    qubit_indices=list(qubits),
                    estimated_width=width,
                    kind=VGateKind.EXPVAL,
                    box_width=width,
                )
            )
        global_phase = (
            None
            if _is_provably_zero_phase(self.program.global_phase)
            else _format_scalar(self.program.global_phase)
        )
        qubit_map = {
            f"wire:{wire.value}": slot
            for wire, slot in sorted(
                self._topology.slots.items(), key=lambda item: item[0].value
            )
        }
        return VisualCircuit(
            children=children,
            qubit_map=qubit_map,
            qubit_names=dict(self.qubit_names),
            num_qubits=self.program.num_qubits,
            output_names=list(self.output_names),
            global_phase=global_phase,
        )

    def _build_nodes(
        self,
        operations: Sequence[CircuitInstruction],
        topology: _WireTopology,
        *,
        trace_region: TraceRegion | None = None,
        path: _NodeKey,
        writers: _WriterMap,
        loop_values: Mapping[str, int],
        visual_depth: int,
        call_depth: int,
        conditional_depth: int,
        inside_call: bool,
        visible: bool,
    ) -> tuple[list[VisualNode], _WriterMap]:
        """Convert one structured region and propagate classical writers.

        Args:
            operations (Sequence[CircuitInstruction]): Region instructions.
            topology (_WireTopology): Exact module-local wire resolver.
            trace_region (TraceRegion | None): Optional source-aware partition
                whose flattening equals ``operations``. Defaults to ``None``.
            path (_NodeKey): Stable key prefix for the region.
            writers (_WriterMap): Reaching measurement definitions at entry.
            loop_values (Mapping[str, int]): Materialized loop values.
            visual_depth (int): Nesting depth used for visual geometry.
            call_depth (int): Number of enclosing expanded source or reusable
                calls.
            conditional_depth (int): Number of enclosing control-flow regions.
            inside_call (bool): Whether this region is a reusable-body preview.
            visible (bool): Whether produced measurement nodes are rendered.

        Returns:
            tuple[list[VisualNode], _WriterMap]: Visual children and exit
                reaching definitions.

        Raises:
            TypeError: If an unknown CircuitInstruction occurs.
            ValueError: If exact wire lineage or a call invariant fails.
        """
        if trace_region is not None and trace_region.flatten() != tuple(operations):
            raise ValueError("Circuit trace region does not match adapter input")
        source_nodes = (
            trace_region.nodes
            if trace_region is not None
            else tuple(TracedInstruction(instruction) for instruction in operations)
        )
        result: list[VisualNode] = []
        current_writers = _copy_writers(writers)
        for index, source_node in enumerate(source_nodes):
            node_key = (*path, index)
            if isinstance(source_node, SpecializedLoopTrace):
                node, current_writers = self._build_specialized_loop(
                    source_node,
                    node_key,
                    topology,
                    current_writers,
                    loop_values,
                    visual_depth,
                    call_depth,
                    conditional_depth,
                    inside_call,
                    visible,
                )
                result.append(node)
                continue
            if isinstance(source_node, InlineRegionTrace):
                node, current_writers = self._build_inline_region(
                    source_node,
                    node_key,
                    topology,
                    current_writers,
                    loop_values,
                    visual_depth,
                    call_depth,
                    conditional_depth,
                    inside_call,
                    visible,
                )
                result.append(node)
                continue
            instruction = source_node.instruction
            child_regions = source_node.regions
            if isinstance(instruction, GateInstruction):
                result.append(
                    self._build_gate(instruction, node_key, topology, loop_values)
                )
            elif isinstance(instruction, MeasureInstruction):
                slot = topology.slot(instruction.input)
                result.append(
                    VGate(
                        node_key=node_key,
                        label="M",
                        qubit_indices=[slot],
                        estimated_width=self.style.gate_width,
                        kind=VGateKind.MEASURE,
                        terminates_wire=(
                            conditional_depth == 0
                            and not inside_call
                            and instruction.output in self._root_outputs
                        ),
                    )
                )
                current_writers[instruction.clbit] = {
                    _MeasurementWriter(node_key, slot, visible)
                }
            elif isinstance(instruction, MeasureVectorInstruction):
                slots = topology.slots_for(instruction.inputs)
                result.append(
                    VGate(
                        node_key=node_key,
                        label="M",
                        qubit_indices=slots,
                        estimated_width=self.style.gate_width,
                        kind=VGateKind.MEASURE_VECTOR,
                        terminates_wire=(
                            conditional_depth == 0
                            and not inside_call
                            and all(
                                output in self._root_outputs
                                for output in instruction.outputs
                            )
                        ),
                    )
                )
                for clbit, slot in zip(instruction.clbits, slots, strict=True):
                    current_writers[clbit] = {
                        _MeasurementWriter(node_key, slot, visible)
                    }
            elif isinstance(instruction, ResetInstruction):
                result.append(
                    self._generic_gate(
                        node_key,
                        "RESET",
                        [topology.slot(instruction.input)],
                    )
                )
            elif isinstance(instruction, BarrierInstruction):
                result.append(
                    self._generic_gate(
                        node_key,
                        "BARRIER",
                        topology.slots_for(instruction.wires),
                    )
                )
            elif isinstance(instruction, PauliEvolutionInstruction):
                time = _format_scalar(instruction.time, loop_values)
                label = f"EVOLVE({instruction.hamiltonian!s}, t={time})"
                result.append(
                    self._generic_gate(
                        node_key,
                        label,
                        topology.slots_for(instruction.inputs),
                    )
                )
            elif isinstance(instruction, CallInstruction):
                result.append(
                    self._build_call(
                        instruction,
                        node_key,
                        topology,
                        loop_values,
                        visual_depth,
                        call_depth,
                    )
                )
            elif isinstance(instruction, ForInstruction):
                node, current_writers = self._build_for(
                    instruction,
                    node_key,
                    topology,
                    current_writers,
                    loop_values,
                    visual_depth,
                    call_depth,
                    conditional_depth,
                    inside_call,
                    visible,
                    child_regions[0] if child_regions else None,
                )
                result.append(node)
            elif isinstance(instruction, IfInstruction):
                node, current_writers = self._build_if(
                    instruction,
                    node_key,
                    topology,
                    current_writers,
                    loop_values,
                    visual_depth,
                    call_depth,
                    conditional_depth,
                    inside_call,
                    visible,
                    child_regions if child_regions else None,
                )
                result.append(node)
            elif isinstance(instruction, WhileInstruction):
                node, current_writers = self._build_while(
                    instruction,
                    node_key,
                    topology,
                    current_writers,
                    loop_values,
                    visual_depth,
                    call_depth,
                    conditional_depth,
                    inside_call,
                    visible,
                    child_regions[0] if child_regions else None,
                )
                result.append(node)
            else:
                raise TypeError(
                    f"Unsupported CircuitInstruction: {type(instruction).__name__}"
                )
        return result, current_writers

    def _build_gate(
        self,
        instruction: GateInstruction,
        node_key: _NodeKey,
        topology: _WireTopology,
        loop_values: Mapping[str, int],
    ) -> VGate:
        """Convert one primitive instruction, including controlled variants.

        Args:
            instruction (GateInstruction): Primitive gate instruction.
            node_key (_NodeKey): Stable visual identity.
            topology (_WireTopology): Exact wire resolver.
            loop_values (Mapping[str, int]): Unrolled induction bindings.

        Returns:
            VGate: Existing renderer-compatible gate node.

        Raises:
            ValueError: If GateKind.MEASURE bypassed its dedicated instruction.
            TypeError: If an unknown GateKind occurs.
        """
        if instruction.kind is GateKind.MEASURE:
            raise ValueError("GateKind.MEASURE must use MeasureInstruction")
        try:
            base_label = _GATE_LABELS[instruction.kind]
        except KeyError as error:
            raise TypeError(f"Unsupported GateKind: {instruction.kind}") from error
        parameters = [
            _format_scalar(parameter, loop_values)
            for parameter in instruction.parameters
        ]
        label = (
            base_label if not parameters else f"{base_label}({', '.join(parameters)})"
        )
        slots = topology.slots_for(instruction.inputs)
        width = self._label_width(label) if parameters else self.style.gate_width
        if instruction.kind in _CONTROLLED_TARGET_TYPES:
            box_width = self._label_width(label)
            return VGate(
                node_key=node_key,
                label=label,
                qubit_indices=slots,
                estimated_width=box_width,
                kind=VGateKind.CONTROLLED_U_BOX,
                gate_type=_CONTROLLED_TARGET_TYPES[instruction.kind],
                has_param=bool(parameters),
                box_width=box_width,
                control_count=1,
            )
        try:
            gate_type = _GATE_TYPES[instruction.kind]
        except KeyError as error:
            raise TypeError(f"Unsupported GateKind: {instruction.kind}") from error
        return VGate(
            node_key=node_key,
            label=label,
            qubit_indices=slots,
            estimated_width=width,
            kind=VGateKind.GATE,
            gate_type=gate_type,
            has_param=bool(parameters),
        )

    def _generic_gate(
        self,
        node_key: _NodeKey,
        label: str,
        qubit_indices: list[int],
    ) -> VGate:
        """Create a clearly labelled generic semantic operation box.

        Args:
            node_key (_NodeKey): Stable visual identity.
            label (str): Complete operation label.
            qubit_indices (list[int]): Exact participating display slots.

        Returns:
            VGate: Generic box node understood by existing layout/renderer.

        Raises:
            ValueError: If the semantic operation has no participating wires.
        """
        if not qubit_indices:
            raise ValueError(f"Visual operation {label!r} has no quantum wires")
        width = self._label_width(label)
        return VGate(
            node_key=node_key,
            label=label,
            qubit_indices=qubit_indices,
            estimated_width=width,
            kind=VGateKind.GATE,
            gate_type=None,
            box_width=width,
        )

    def _build_call(
        self,
        instruction: CallInstruction,
        node_key: _NodeKey,
        topology: _WireTopology,
        loop_values: Mapping[str, int],
        visual_depth: int,
        call_depth: int,
    ) -> VGate | VInlineBlock:
        """Convert a reusable call without applying lossy transformations.

        Args:
            instruction (CallInstruction): Reusable circuit invocation.
            node_key (_NodeKey): Stable visual identity.
            topology (_WireTopology): Caller wire resolver.
            loop_values (Mapping[str, int]): Active loop bindings.
            visual_depth (int): Nesting depth used for visual geometry.
            call_depth (int): Number of enclosing expanded source or reusable
                calls.

        Returns:
            VGate | VInlineBlock: Collapsed call or exact direct-body preview.

        Raises:
            ValueError: If call controls/targets violate verified arity.
            TypeError: If the body contains an unsupported instruction.
        """
        callee = instruction.callee
        preview_body = self._direct_preview_body(callee)
        slots = topology.slots_for(instruction.inputs)
        if callee.controls > len(slots):
            raise ValueError("Reusable call control count exceeds its input arity")
        controls = slots[: callee.controls]
        targets = slots[callee.controls :]
        if len(targets) != callee.body.num_qubits:
            raise ValueError("Reusable call target arity does not match its body")
        label = callee.name
        if not label and callee.identity is not None:
            label = callee.identity.symbol or callee.identity.key.name
        if not label:
            label = callee.native_realization or "U"
        if len(callee.body.operations) == 1:
            body_instruction = callee.body.operations[0]
            if (
                isinstance(body_instruction, GateInstruction)
                and label.casefold() == body_instruction.kind.name.casefold()
            ):
                label = _GATE_LABELS[body_instruction.kind]
        if callee.call_arguments:
            arguments = ", ".join(
                f"{_format_call_parameter_name(name)}="
                f"{_format_call_argument(expression, loop_values)}"
                for name, expression in callee.call_arguments
            )
            label = f"{label}({arguments})"
        if callee.inverse:
            label = f"{label}^-1"
        if callee.power != 1 and callee.controls == 0:
            label = f"{label}^{callee.power}"

        can_expand = (
            self.expand_calls
            and callee.power == 1
            and not callee.inverse
            and (self.inline_depth is None or call_depth < self.inline_depth)
            and callee.native_realization is None
            and not callee.opaque
        )
        if can_expand:
            body_topology = _WireTopology(preview_body, targets)
            children, _ = self._build_nodes(
                preview_body.operations,
                body_topology,
                path=(*node_key, "body"),
                writers={},
                loop_values=loop_values,
                visual_depth=visual_depth + 1,
                call_depth=call_depth + 1,
                conditional_depth=0,
                inside_call=True,
                visible=True,
            )
            if not _is_provably_zero_phase(preview_body.global_phase):
                label = f"{label} [phase={_format_scalar(preview_body.global_phase)}]"
            affected = self._visual_footprint(children)
            if not affected:
                affected = list(targets)
            border_padding = compute_border_padding(self.style, visual_depth)
            max_gate_width = max(
                (self._node_width(child) for child in children),
                default=self.style.gate_width,
            )
            content_width = self._sequence_width(children)
            label_width = self._label_width(label)
            final_width = max(
                label_width,
                content_width + 2 * border_padding,
                self.style.gate_width,
            )
            return VInlineBlock(
                node_key=node_key,
                label=label,
                children=children,
                affected_qubits=affected,
                control_qubit_indices=controls,
                power=1,
                depth=visual_depth,
                border_padding=border_padding,
                max_gate_width=max_gate_width,
                label_width=label_width,
                content_width=content_width,
                final_width=final_width,
            )

        box_width = self._label_width(label)
        gate_type = (
            self._single_body_gate_type(callee.body) if callee.power == 1 else None
        )
        estimated_width = box_width
        if callee.controls and callee.power > 1:
            estimated_width = max(
                box_width + 2 * self.style.power_wrapper_margin,
                self._label_width(f"pow={callee.power}"),
            )
        return VGate(
            node_key=node_key,
            label=label,
            qubit_indices=controls + targets,
            estimated_width=estimated_width,
            kind=(
                VGateKind.CONTROLLED_U_BOX
                if callee.controls
                else VGateKind.COMPOSITE_BOX
            ),
            gate_type=gate_type if callee.controls else None,
            box_width=box_width,
            control_count=callee.controls,
            power=callee.power if callee.controls else 1,
        )

    @staticmethod
    def _direct_preview_body(callee: ReusableCircuit) -> CircuitProgram:
        """Remove one target-neutral implementation-only forwarding wrapper.

        Semantic composite emission can retain a named outer call whose
        fallback program consists solely of one direct reusable call on the
        same inputs and outputs. The inner call is an implementation detail,
        not another source nesting level, so expanding the semantic call may
        preview its body directly. Any phase, transform, or non-forwarding
        wire map keeps the original body intact.

        Args:
            callee (ReusableCircuit): Reusable call selected for display.

        Returns:
            CircuitProgram: Direct implementation body when exact forwarding
                is proven, otherwise ``callee.body``.
        """
        body = callee.body
        if (
            callee.identity is None
            or not _is_provably_zero_phase(body.global_phase)
            or len(body.operations) != 1
        ):
            return body
        instruction = body.operations[0]
        if not isinstance(instruction, CallInstruction):
            return body
        implementation = instruction.callee
        if (
            implementation.controls != 0
            or implementation.power != 1
            or implementation.inverse
            or implementation.native_realization is not None
            or instruction.inputs != body.input_wires
            or instruction.outputs != body.output_wires
        ):
            return body
        return implementation.body

    @staticmethod
    def _single_body_gate_type(program: CircuitProgram) -> GateOperationType | None:
        """Return a native target symbol for an exact one-gate body.

        Args:
            program (CircuitProgram): Reusable fallback body.

        Returns:
            GateOperationType | None: Supported primitive symbol or ``None``.
        """
        if not _is_provably_zero_phase(program.global_phase):
            return None
        if len(program.operations) != 1:
            return None
        instruction = program.operations[0]
        if not isinstance(instruction, GateInstruction):
            return None
        gate_type = _GATE_TYPES.get(instruction.kind)
        if gate_type in {
            GateOperationType.X,
            GateOperationType.Z,
            GateOperationType.CX,
            GateOperationType.CZ,
            GateOperationType.SWAP,
            GateOperationType.TOFFOLI,
        }:
            return gate_type
        return None

    def _label_width(self, label: str) -> float:
        """Measure a text box with the same renderer font metrics.

        Args:
            label (str): Text or mathtext label.

        Returns:
            float: Full padded box width.
        """
        text_width = measure_text_width(
            label,
            font_size=self.style.font_size,
            fallback_char_width=self.style.char_width_gate,
        )
        return max(
            self.style.gate_width,
            text_width + 2 * self.style.text_padding,
        )

    def _folded_width(self, header: str, body_lines: Sequence[str]) -> float:
        """Measure a folded control-flow summary without truncation.

        Args:
            header (str): Bold control-flow header.
            body_lines (Sequence[str]): Complete direct-body summaries.

        Returns:
            float: Padded folded box width.
        """
        header_width = measure_text_width(
            header,
            font_size=self.style.font_size,
            font_weight="bold",
            fallback_char_width=self.style.char_width_bold,
        )
        body_width = max(
            (
                measure_text_width(
                    line,
                    font_size=self.style.subfont_size,
                    fallback_char_width=self.style.char_width_monospace,
                )
                for line in body_lines
            ),
            default=0.0,
        )
        return max(
            self.style.folded_loop_width,
            max(header_width, body_width) + 2 * self.style.text_padding,
        )

    def _node_width(self, node: VisualNode) -> float:
        """Return one Visual IR node's reserved width estimate.

        Args:
            node (VisualNode): Node to inspect.

        Returns:
            float: Non-negative estimated width.

        Raises:
            TypeError: If VisualNode is extended without adapter support.
        """
        if isinstance(node, VGate):
            return node.estimated_width
        if isinstance(node, VInlineBlock):
            return node.final_width
        if isinstance(node, VFoldedBlock):
            return node.folded_width
        if isinstance(node, VUnfoldedSequence):
            return sum(node.iteration_widths)
        if isinstance(node, VSkip):
            return 0.0
        raise TypeError(f"Unsupported VisualNode: {type(node).__name__}")

    def _sequence_width(self, nodes: Sequence[VisualNode]) -> float:
        """Estimate a sequential Visual IR region's full horizontal width.

        Args:
            nodes (Sequence[VisualNode]): Nodes in program order.

        Returns:
            float: Width plus gaps between space-bearing nodes.

        Raises:
            TypeError: If a VisualNode variant is unsupported.
        """
        widths = [self._node_width(node) for node in nodes]
        visible = [width for width in widths if width > 0]
        return sum(visible) + self.style.gate_gap * max(0, len(visible) - 1)

    @staticmethod
    def _visual_footprint(nodes: Sequence[VisualNode]) -> list[int]:
        """Collect exact participating slots from a Visual IR subtree.

        Args:
            nodes (Sequence[VisualNode]): Visual nodes to inspect.

        Returns:
            list[int]: Sorted unique display slots.

        Raises:
            TypeError: If a VisualNode variant is unsupported.
        """
        slots: set[int] = set()
        for node in nodes:
            if isinstance(node, VGate):
                slots.update(node.qubit_indices)
            elif isinstance(node, VInlineBlock):
                slots.update(node.affected_qubits)
                slots.update(node.control_qubit_indices)
            elif isinstance(node, VFoldedBlock):
                slots.update(node.affected_qubits)
            elif isinstance(node, VUnfoldedSequence):
                slots.update(node.affected_qubits)
            elif not isinstance(node, VSkip):
                raise TypeError(f"Unsupported VisualNode: {type(node).__name__}")
        return sorted(slots)

    @staticmethod
    def _summary_lines(nodes: Sequence[VisualNode]) -> list[str]:
        """Return deterministic, complete direct-node summaries.

        Args:
            nodes (Sequence[VisualNode]): Nodes to summarize.

        Returns:
            list[str]: One readable line per direct node, or ``(empty)``.

        Raises:
            TypeError: If a VisualNode variant is unsupported.
        """
        lines: list[str] = []
        for node in nodes:
            if isinstance(node, (VGate, VInlineBlock)):
                lines.append(node.label)
            elif isinstance(node, VFoldedBlock):
                lines.append(node.header_label)
            elif isinstance(node, VUnfoldedSequence):
                lines.append(node.condition_label or node.kind.name.lower())
            elif not isinstance(node, VSkip):
                raise TypeError(f"Unsupported VisualNode: {type(node).__name__}")
        return lines or ["(empty)"]

    @staticmethod
    def _condition_measure_info(
        condition: ScalarExpr,
        writers: _WriterMap,
    ) -> tuple[_NodeKey | None, list[int]]:
        """Resolve exact visible sources from one measurement node.

        Args:
            condition (ScalarExpr): Runtime condition expression.
            writers (_WriterMap): Reaching measurement definitions.

        Returns:
            tuple[_NodeKey | None, list[int]]: Measurement key and exact source
                slots, or no connector when any bit has ambiguous provenance
                or the condition spans multiple measurement nodes.

        Raises:
            TypeError: If ScalarExpr is extended unexpectedly.
        """
        measure_key: _NodeKey | None = None
        source_qubits: list[int] = []
        for clbit in sorted(_classical_bits(condition)):
            candidates = writers.get(clbit, set())
            if not candidates or any(not source.visible for source in candidates):
                return None, []
            candidate_keys = {source.node_key for source in candidates}
            if len(candidate_keys) != 1:
                return None, []
            candidate_key = next(iter(candidate_keys))
            if measure_key is None:
                measure_key = candidate_key
            elif candidate_key != measure_key:
                return None, []
            for source in sorted(candidates, key=lambda item: item.qubit_index):
                if source.qubit_index not in source_qubits:
                    source_qubits.append(source.qubit_index)
        return measure_key, source_qubits

    def _build_inline_region(
        self,
        trace: InlineRegionTrace,
        node_key: _NodeKey,
        topology: _WireTopology,
        writers: _WriterMap,
        loop_values: Mapping[str, int],
        visual_depth: int,
        call_depth: int,
        conditional_depth: int,
        inside_call: bool,
        visible: bool,
    ) -> tuple[VGate | VInlineBlock | VSkip, _WriterMap]:
        """Convert one source qkernel region erased from executable IR.

        Args:
            trace (InlineRegionTrace): Lossless inlined source region.
            node_key (_NodeKey): Stable visual identity.
            topology (_WireTopology): Exact caller wire resolver.
            writers (_WriterMap): Reaching measurement writers at entry.
            loop_values (Mapping[str, int]): Active native-loop bindings.
            visual_depth (int): Current visual nesting depth.
            call_depth (int): Number of expanded source/circuit calls.
            conditional_depth (int): Enclosing structured-region depth.
            inside_call (bool): Whether this region is already in a call.
            visible (bool): Whether descendants are rendered.

        Returns:
            tuple[VGate | VInlineBlock | VSkip, _WriterMap]: Collapsed or
                expanded source call and its exact exit writer state.

        Raises:
            TypeError: If a traced instruction is unsupported.
            ValueError: If exact wire lineage fails.
        """
        label = trace.label or "qkernel"
        if trace.call_arguments:
            arguments = ", ".join(
                f"{_format_call_parameter_name(name)}="
                f"{_format_call_argument(expression, loop_values)}"
                for name, expression in trace.call_arguments
            )
            label = f"{label}({arguments})"
        can_expand = self.expand_calls and (
            self.inline_depth is None or call_depth < self.inline_depth
        )
        children, exit_writers = self._build_nodes(
            trace.region.flatten(),
            topology,
            trace_region=trace.region,
            path=(*node_key, "body"),
            writers=writers,
            loop_values=loop_values,
            visual_depth=visual_depth + 1,
            call_depth=call_depth + 1,
            conditional_depth=conditional_depth,
            inside_call=inside_call,
            visible=visible and can_expand,
        )
        affected = sorted(
            set(self._visual_footprint(children)).union(trace.quantum_slots)
        )
        if not affected:
            return VSkip(node_key), exit_writers
        if not can_expand:
            width = self._label_width(label)
            return (
                VGate(
                    node_key=node_key,
                    label=label,
                    qubit_indices=affected,
                    estimated_width=width,
                    kind=VGateKind.BLOCK_BOX,
                    box_width=width,
                ),
                _proxy_changed_writers(
                    writers,
                    exit_writers,
                    node_key,
                    visible=visible,
                ),
            )

        border_padding = compute_border_padding(self.style, visual_depth)
        max_gate_width = max(
            (self._node_width(child) for child in children),
            default=self.style.gate_width,
        )
        content_width = self._sequence_width(children)
        label_width = self._label_width(label)
        final_width = max(
            label_width,
            content_width + 2 * border_padding,
            self.style.gate_width,
        )
        return (
            VInlineBlock(
                node_key=node_key,
                label=label,
                children=children,
                affected_qubits=affected,
                control_qubit_indices=[],
                power=1,
                depth=visual_depth,
                border_padding=border_padding,
                max_gate_width=max_gate_width,
                label_width=label_width,
                content_width=content_width,
                final_width=final_width,
            ),
            exit_writers,
        )

    def _build_specialized_loop(
        self,
        trace: SpecializedLoopTrace,
        node_key: _NodeKey,
        topology: _WireTopology,
        writers: _WriterMap,
        loop_values: Mapping[str, int],
        visual_depth: int,
        call_depth: int,
        conditional_depth: int,
        inside_call: bool,
        visible: bool,
    ) -> tuple[VFoldedBlock | VUnfoldedSequence | VSkip, _WriterMap]:
        """Convert exact per-iteration lowering provenance into one loop node.

        Args:
            trace (SpecializedLoopTrace): Exact source loop provenance.
            node_key (_NodeKey): Stable visual identity.
            topology (_WireTopology): Exact wire resolver.
            writers (_WriterMap): Reaching measurement writers at entry.
            loop_values (Mapping[str, int]): Enclosing native-loop bindings.
            visual_depth (int): Current visual nesting depth.
            call_depth (int): Number of enclosing expanded source or reusable
                calls.
            conditional_depth (int): Enclosing structured-region depth.
            inside_call (bool): Whether the loop occurs inside a call preview.
            visible (bool): Whether descendants are rendered.

        Returns:
            tuple[VFoldedBlock | VUnfoldedSequence | VSkip, _WriterMap]: Exact
                folded summary or materialized iteration sequence and exit
                writer state.

        Raises:
            TypeError: If a traced instruction is unsupported.
            ValueError: If source metadata or wire lineage is malformed.
        """
        if not trace.iterations:
            return VSkip(node_key), _copy_writers(writers)
        iterations: list[list[VisualNode]] = []
        widths: list[float] = []
        current = _copy_writers(writers)
        child_visible = visible and not self.fold_loops
        for ordinal, iteration in enumerate(trace.iterations):
            children, current = self._build_nodes(
                iteration.region.flatten(),
                topology,
                trace_region=iteration.region,
                path=(*node_key, "iteration", ordinal),
                writers=current,
                loop_values=loop_values,
                visual_depth=visual_depth + 1,
                call_depth=call_depth,
                conditional_depth=conditional_depth + 1,
                inside_call=inside_call,
                visible=child_visible,
            )
            iterations.append(children)
            widths.append(self._sequence_width(children))
        affected = self._visual_footprint(
            [child for iteration in iterations for child in iteration]
        )
        if isinstance(trace.origin, RangeLoopOrigin):
            header = f"for {trace.origin.variable} in {trace.origin.indexset!s}:"
        elif isinstance(trace.origin, ItemsLoopOrigin):
            keys = ", ".join(trace.origin.key_variables) or "key"
            header = f"for {keys}, {trace.origin.value_variable} in items:"
        else:
            raise TypeError(
                f"Unsupported specialized loop origin: {type(trace.origin).__name__}"
            )
        if not self.fold_loops:
            kind = (
                VUnfoldedKind.FOR
                if trace.kind is SpecializedLoopKind.RANGE
                else VUnfoldedKind.FOR_ITEMS
            )
            return (
                VUnfoldedSequence(
                    node_key=node_key,
                    iterations=iterations,
                    affected_qubits=affected,
                    kind=kind,
                    iteration_widths=widths,
                ),
                current,
            )

        summaries = [tuple(self._summary_lines(children)) for children in iterations]
        distinct_summaries = list(dict.fromkeys(summaries))
        if len(distinct_summaries) == 1:
            body_lines = list(distinct_summaries[0])
        else:
            body_lines = [
                f"{len(iterations)} specialized iterations "
                f"({len(distinct_summaries)} distinct bodies)"
            ]
            body_lines.extend(
                f"body {index + 1}: {', '.join(summary)}"
                for index, summary in enumerate(distinct_summaries)
            )
        kind = (
            VFoldedKind.FOR
            if trace.kind is SpecializedLoopKind.RANGE
            else VFoldedKind.FOR_ITEMS
        )
        return (
            VFoldedBlock(
                node_key=node_key,
                header_label=header,
                body_lines=body_lines,
                affected_qubits=affected,
                folded_width=self._folded_width(header, body_lines),
                kind=kind,
            ),
            _proxy_changed_writers(
                writers,
                current,
                node_key,
                visible=visible,
            ),
        )

    def _build_for(
        self,
        instruction: ForInstruction,
        node_key: _NodeKey,
        topology: _WireTopology,
        writers: _WriterMap,
        loop_values: Mapping[str, int],
        visual_depth: int,
        call_depth: int,
        conditional_depth: int,
        inside_call: bool,
        visible: bool,
        body_trace: TraceRegion | None = None,
    ) -> tuple[VFoldedBlock | VUnfoldedSequence | VSkip, _WriterMap]:
        """Convert one concrete structured for loop.

        Args:
            instruction (ForInstruction): Concrete loop instruction.
            node_key (_NodeKey): Stable visual identity.
            topology (_WireTopology): Exact wire resolver.
            writers (_WriterMap): Reaching measurement writers at entry.
            loop_values (Mapping[str, int]): Enclosing induction bindings.
            visual_depth (int): Nesting depth used for visual geometry.
            call_depth (int): Number of enclosing expanded source or reusable
                calls.
            conditional_depth (int): Enclosing structured-region depth.
            inside_call (bool): Whether the loop is inside a call preview.
            visible (bool): Whether this loop's descendants are rendered.
            body_trace (TraceRegion | None): Optional lossless trace for the
                native loop body. Defaults to ``None``.

        Returns:
            tuple[VFoldedBlock | VUnfoldedSequence | VSkip, _WriterMap]: Loop
                node and exact exit reaching definitions.

        Raises:
            TypeError: If a nested instruction is unsupported.
            ValueError: If exact wire lineage fails.
        """
        iteration_count = len(instruction.indexset)
        if iteration_count == 0:
            return VSkip(node_key), _copy_writers(writers)

        if not self.fold_loops:
            iterations: list[list[VisualNode]] = []
            iteration_widths: list[float] = []
            current = _copy_writers(writers)
            for ordinal, value in enumerate(instruction.indexset):
                bindings = dict(loop_values)
                bindings[instruction.loop_variable.name] = value
                children, current = self._build_nodes(
                    instruction.body,
                    topology,
                    trace_region=body_trace,
                    path=(*node_key, "iteration", ordinal),
                    writers=current,
                    loop_values=bindings,
                    visual_depth=visual_depth + 1,
                    call_depth=call_depth,
                    conditional_depth=conditional_depth + 1,
                    inside_call=inside_call,
                    visible=visible,
                )
                iterations.append(children)
                iteration_widths.append(self._sequence_width(children))
            affected = self._visual_footprint(
                [child for iteration in iterations for child in iteration]
            )
            return (
                VUnfoldedSequence(
                    node_key=node_key,
                    iterations=iterations,
                    affected_qubits=affected,
                    kind=VUnfoldedKind.FOR,
                    iteration_widths=iteration_widths,
                ),
                current,
            )

        children, current = self._build_nodes(
            instruction.body,
            topology,
            trace_region=body_trace,
            path=(*node_key, "body"),
            writers=writers,
            loop_values=loop_values,
            visual_depth=visual_depth + 1,
            call_depth=call_depth,
            conditional_depth=conditional_depth + 1,
            inside_call=inside_call,
            visible=False,
        )
        # A folded body represents one static instruction site. Reapply its
        # writer transfer until either the concrete trip count is exhausted or
        # the finite reaching-definition state reaches a fixed point.
        for _ in range(1, iteration_count):
            _, next_writers = self._build_nodes(
                instruction.body,
                topology,
                trace_region=body_trace,
                path=(*node_key, "body"),
                writers=current,
                loop_values=loop_values,
                visual_depth=visual_depth + 1,
                call_depth=call_depth,
                conditional_depth=conditional_depth + 1,
                inside_call=inside_call,
                visible=False,
            )
            if next_writers == current:
                break
            current = next_writers
        affected = self._visual_footprint(children)
        header = f"for {instruction.loop_variable.name} in {instruction.indexset!s}:"
        body_lines = self._summary_lines(children)
        return (
            VFoldedBlock(
                node_key=node_key,
                header_label=header,
                body_lines=body_lines,
                affected_qubits=affected,
                folded_width=self._folded_width(header, body_lines),
                kind=VFoldedKind.FOR,
            ),
            _proxy_changed_writers(
                writers,
                current,
                node_key,
                visible=visible,
            ),
        )

    def _build_if(
        self,
        instruction: IfInstruction,
        node_key: _NodeKey,
        topology: _WireTopology,
        writers: _WriterMap,
        loop_values: Mapping[str, int],
        visual_depth: int,
        call_depth: int,
        conditional_depth: int,
        inside_call: bool,
        visible: bool,
        branch_traces: tuple[TraceRegion, ...] | None = None,
    ) -> tuple[VFoldedBlock | VUnfoldedSequence, _WriterMap]:
        """Convert a structured runtime conditional and join its writers.

        Args:
            instruction (IfInstruction): Runtime conditional instruction.
            node_key (_NodeKey): Stable visual identity.
            topology (_WireTopology): Exact wire resolver.
            writers (_WriterMap): Reaching measurement writers at entry.
            loop_values (Mapping[str, int]): Active induction bindings.
            visual_depth (int): Nesting depth used for visual geometry.
            call_depth (int): Number of enclosing expanded source or reusable
                calls.
            conditional_depth (int): Enclosing structured-region depth.
            inside_call (bool): Whether the conditional is in a call preview.
            visible (bool): Whether descendants are present in the visual tree.
            branch_traces (tuple[TraceRegion, ...] | None): Optional true and
                false source-aware regions. Defaults to ``None``.

        Returns:
            tuple[VFoldedBlock | VUnfoldedSequence, _WriterMap]: Conditional
                node and the branch-joined writer state.

        Raises:
            TypeError: If a nested instruction or scalar is unsupported.
            ValueError: If exact wire lineage fails.
        """
        condition = _format_scalar(instruction.condition, loop_values)
        header = f"if {condition}:"
        measure_key, measure_qubits = self._condition_measure_info(
            instruction.condition, writers
        )
        child_visible = visible and not self.fold_ifs
        true_trace = branch_traces[0] if branch_traces is not None else None
        false_trace = branch_traces[1] if branch_traces is not None else None
        true_nodes, true_writers = self._build_nodes(
            instruction.true_body,
            topology,
            trace_region=true_trace,
            path=(*node_key, "true"),
            writers=writers,
            loop_values=loop_values,
            visual_depth=visual_depth + 1,
            call_depth=call_depth,
            conditional_depth=conditional_depth + 1,
            inside_call=inside_call,
            visible=child_visible,
        )
        false_nodes, false_writers = self._build_nodes(
            instruction.false_body,
            topology,
            trace_region=false_trace,
            path=(*node_key, "false"),
            writers=writers,
            loop_values=loop_values,
            visual_depth=visual_depth + 1,
            call_depth=call_depth,
            conditional_depth=conditional_depth + 1,
            inside_call=inside_call,
            visible=child_visible,
        )
        joined = _merge_writers(true_writers, false_writers)
        affected = self._visual_footprint([*true_nodes, *false_nodes])
        if not affected and measure_qubits:
            affected = list(measure_qubits)
        if self.fold_ifs:
            body_lines = [
                "true:",
                *self._summary_lines(true_nodes),
                "else:",
                *self._summary_lines(false_nodes),
            ]
            return (
                VFoldedBlock(
                    node_key=node_key,
                    header_label=header,
                    body_lines=body_lines,
                    affected_qubits=affected,
                    folded_width=self._folded_width(header, body_lines),
                    kind=VFoldedKind.IF,
                    condition_measure_node_key=measure_key,
                    condition_measure_qubit_indices=measure_qubits,
                ),
                _proxy_changed_writers(
                    writers,
                    joined,
                    node_key,
                    visible=visible,
                ),
            )
        iterations = [true_nodes, false_nodes]
        label_width = self._label_width(header)
        branch_widths = [label_width, self._label_width("else:")]
        return (
            VUnfoldedSequence(
                node_key=node_key,
                iterations=iterations,
                affected_qubits=affected,
                kind=VUnfoldedKind.IF,
                iteration_widths=[
                    self._sequence_width(true_nodes),
                    self._sequence_width(false_nodes),
                ],
                condition_label=header,
                condition_label_width=label_width,
                branch_label_widths=branch_widths,
                condition_measure_node_key=measure_key,
                condition_measure_qubit_indices=measure_qubits,
            ),
            joined,
        )

    def _build_while(
        self,
        instruction: WhileInstruction,
        node_key: _NodeKey,
        topology: _WireTopology,
        writers: _WriterMap,
        loop_values: Mapping[str, int],
        visual_depth: int,
        call_depth: int,
        conditional_depth: int,
        inside_call: bool,
        visible: bool,
        body_trace: TraceRegion | None = None,
    ) -> tuple[VFoldedBlock | VUnfoldedSequence, _WriterMap]:
        """Convert a runtime while loop with loop-carried writer analysis.

        Args:
            instruction (WhileInstruction): Runtime loop instruction.
            node_key (_NodeKey): Stable visual identity.
            topology (_WireTopology): Exact wire resolver.
            writers (_WriterMap): Reaching measurement writers at entry.
            loop_values (Mapping[str, int]): Active induction bindings.
            visual_depth (int): Nesting depth used for visual geometry.
            call_depth (int): Number of enclosing expanded source or reusable
                calls.
            conditional_depth (int): Enclosing structured-region depth.
            inside_call (bool): Whether the loop is in a call preview.
            visible (bool): Whether descendants are present in the visual tree.
            body_trace (TraceRegion | None): Optional source-aware body region.
                Defaults to ``None``.

        Returns:
            tuple[VFoldedBlock | VUnfoldedSequence, _WriterMap]: While node and
                fixed-point exit writer state.

        Raises:
            RuntimeError: If finite classical writer analysis does not reach a
                fixed point.
            TypeError: If a nested instruction or scalar is unsupported.
            ValueError: If exact wire lineage fails.
        """
        head = _copy_writers(writers)
        max_iterations = max(2, self.program.num_clbits * 4 + 4)
        for _ in range(max_iterations):
            _, body_exit = self._build_nodes(
                instruction.body,
                topology,
                trace_region=body_trace,
                path=(*node_key, "body"),
                writers=head,
                loop_values=loop_values,
                visual_depth=visual_depth + 1,
                call_depth=call_depth,
                conditional_depth=conditional_depth + 1,
                inside_call=inside_call,
                visible=False,
            )
            next_head = _merge_writers(writers, body_exit)
            if next_head == head:
                break
            head = next_head
        else:
            raise RuntimeError("While measurement-writer analysis did not converge")

        condition = _format_scalar(instruction.condition, loop_values)
        header = f"while {condition}:"
        # The visible header represents the first predicate evaluation. Its
        # connector must therefore originate from the entry reaching
        # definition; loop-carried measurements belong to later evaluations
        # and would require a separate back-edge visual.
        measure_key, measure_qubits = self._condition_measure_info(
            instruction.condition, writers
        )
        child_visible = visible and not self.fold_whiles
        body_nodes, body_exit = self._build_nodes(
            instruction.body,
            topology,
            trace_region=body_trace,
            path=(*node_key, "body"),
            writers=head,
            loop_values=loop_values,
            visual_depth=visual_depth + 1,
            call_depth=call_depth,
            conditional_depth=conditional_depth + 1,
            inside_call=inside_call,
            visible=child_visible,
        )
        exit_writers = _merge_writers(writers, body_exit)
        affected = self._visual_footprint(body_nodes)
        if not affected and measure_qubits:
            affected = list(measure_qubits)
        if self.fold_whiles:
            body_lines = self._summary_lines(body_nodes)
            return (
                VFoldedBlock(
                    node_key=node_key,
                    header_label=header,
                    body_lines=body_lines,
                    affected_qubits=affected,
                    folded_width=self._folded_width(header, body_lines),
                    kind=VFoldedKind.WHILE,
                    condition_measure_node_key=measure_key,
                    condition_measure_qubit_indices=measure_qubits,
                ),
                _proxy_changed_writers(
                    writers,
                    exit_writers,
                    node_key,
                    visible=visible,
                ),
            )
        label_width = self._label_width(header)
        return (
            VUnfoldedSequence(
                node_key=node_key,
                iterations=[body_nodes],
                affected_qubits=affected,
                kind=VUnfoldedKind.WHILE,
                iteration_widths=[self._sequence_width(body_nodes)],
                condition_label=header,
                condition_label_width=label_width,
                branch_label_widths=[label_width],
                condition_measure_node_key=measure_key,
                condition_measure_qubit_indices=measure_qubits,
            ),
            exit_writers,
        )


def circuit_program_to_visual_ir(
    program: CircuitProgram,
    *,
    trace: CircuitProgramTrace | None = None,
    style: CircuitStyle | None = None,
    qubit_names: Mapping[int, str] | None = None,
    output_names: Sequence[str] | None = None,
    expectation_value_qubits: Sequence[Sequence[int]] | None = None,
    expand_calls: bool = False,
    inline_depth: int | None = None,
    fold_loops: bool = True,
    fold_ifs: bool = False,
    fold_whiles: bool = False,
) -> VisualCircuit:
    """Convert a target-neutral circuit program to existing Visual IR.

    Args:
        program (CircuitProgram): Program to verify and convert.
        trace (CircuitProgramTrace | None): Optional lossless drawing-only
            source provenance aligned with ``program``. Defaults to ``None``.
        style (CircuitStyle | None): Style used for text measurement.
        qubit_names (Mapping[int, str] | None): Optional names by slot.
        output_names (Sequence[str] | None): Optional output labels.
        expectation_value_qubits (Sequence[Sequence[int]] | None): Exact
            physical slots for symbolic terminal expectation-value boxes.
        expand_calls (bool): Expand source qkernel regions and direct,
            untransformed reusable bodies. Bodyless opaque calls remain boxed.
        inline_depth (int | None): Maximum source/reusable call expansion
            depth.
        fold_loops (bool): Fold concrete for-loop bodies.
        fold_ifs (bool): Fold runtime conditional bodies.
        fold_whiles (bool): Fold runtime while bodies.

    Returns:
        VisualCircuit: Exact renderer-ready circuit tree.

    Raises:
        ValueError: If circuit verification or exact lineage resolution fails.
        TypeError: If a closed Circuit IR union has an unsupported variant.
    """
    return CircuitProgramAdapter(
        program,
        trace=trace,
        style=style,
        qubit_names=qubit_names,
        output_names=output_names,
        expectation_value_qubits=expectation_value_qubits,
        expand_calls=expand_calls,
        inline_depth=inline_depth,
        fold_loops=fold_loops,
        fold_ifs=fold_ifs,
        fold_whiles=fold_whiles,
    ).build()
