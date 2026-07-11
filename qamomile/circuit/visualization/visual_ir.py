"""Visual IR: pre-resolved intermediate representation for circuit visualization.

This module defines the Visual IR node types that carry all resolved information
(labels, qubit indices, widths) needed by Layout and Renderer. The Visual IR
serves as the decoupling boundary between Analyzer (which understands IR semantics)
and Layout/Renderer (which only need pre-resolved visual information).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from qamomile.circuit.ir.operation.gate import GateOperationType

__all__ = [
    "GateOperationType",
    "VFoldedBlock",
    "VFoldedKind",
    "VGate",
    "VGateKind",
    "VInlineBlock",
    "VSkip",
    "VUnfoldedKind",
    "VUnfoldedSequence",
    "VisualCircuit",
    "VisualNode",
]


class VGateKind(enum.Enum):
    """Classify gate-like nodes for rendering dispatch."""

    GATE = enum.auto()
    MEASURE = enum.auto()
    MEASURE_VECTOR = enum.auto()
    BLOCK_BOX = enum.auto()
    COMPOSITE_BOX = enum.auto()
    CONTROLLED_U_BOX = enum.auto()
    EXPVAL = enum.auto()


class VFoldedKind(enum.Enum):
    """Classify folded control-flow blocks."""

    FOR = enum.auto()
    WHILE = enum.auto()
    FOR_ITEMS = enum.auto()
    IF = enum.auto()


class VUnfoldedKind(enum.Enum):
    """Classify unfolded control-flow sequences."""

    FOR = enum.auto()
    FOR_ITEMS = enum.auto()
    IF = enum.auto()
    WHILE = enum.auto()


@dataclass
class VGate:
    """Represent a pre-resolved gate, measurement, block, or expval node.

    Args:
        node_key (tuple): Stable identity shared by analysis, layout, and
            rendering.
        label (str): TeX-formatted display label.
        qubit_indices (list[int]): Resolved operand wire indices.
        estimated_width (float): Full horizontal width reserved by layout,
            including any powered wrapper.
        kind (VGateKind): Rendering strategy for this node.
        gate_type (GateOperationType | None): Concrete gate type used for
            dedicated multi-qubit symbols. Defaults to None.
        has_param (bool): Whether the label contains gate parameters. Defaults
            to False.
        box_width (float | None): Width of the inner rendered box for block-like
            nodes. Defaults to None.
        control_count (int): Number of leading ``qubit_indices`` that are
            controls. Defaults to zero.
        power (int): Wrapped-unitary power shown by an outer box. Defaults to
            one.
        terminates_wire (bool): Whether this node ends its measured wires.
            Defaults to True.
    """

    node_key: tuple
    label: str
    qubit_indices: list[int]
    estimated_width: float
    kind: VGateKind
    gate_type: GateOperationType | None = None
    has_param: bool = False
    box_width: float | None = None
    control_count: int = 0  # For CONTROLLED_U_BOX: first N indices are controls
    # For CONTROLLED_U_BOX: ``power`` value of the wrapped unitary.
    # When > 1, the renderer draws an outer ``pow=N`` wrapper box
    # around the inner controlled-U rectangle (matching the
    # ``VInlineBlock`` rendering of an expanded controlled-U block).
    power: int = 1
    # Whether a measurement node terminates its wire at this x-position.
    # True for a top-level measurement (the wire ends after it, the usual
    # final-measure convention). False for a measurement inside an if/else
    # branch: that is a mid-circuit measurement on one alternative, so the
    # wire must keep going (the other branch never measured this qubit).
    terminates_wire: bool = True


@dataclass
class VInlineBlock:
    """Represent an inlined call, controlled-U, or composite block.

    Args:
        node_key (tuple): Stable identity shared across visualization phases.
        label (str): Block label displayed in the inner border.
        children (list[VisualNode]): Expanded child nodes.
        affected_qubits (list[int]): Wires semantically used by the block.
        control_qubit_indices (list[int]): Explicit control wires outside the
            target border.
        power (int): Wrapped-block power. Values above one add an outer wrapper.
        depth (int): Visual nesting depth assigned by analysis.
        border_padding (float): Inner border padding.
        max_gate_width (float): Widest descendant gate width.
        label_width (float): Minimum inner width required by the block label.
        content_width (float): Estimated child-content width.
        final_width (float): Minimum full width including any power wrapper.
    """

    node_key: tuple
    label: str
    children: list[VisualNode]
    affected_qubits: list[int]
    control_qubit_indices: list[int]
    power: int
    depth: int
    border_padding: float
    max_gate_width: float
    label_width: float
    content_width: float
    final_width: float


@dataclass
class VFoldedBlock:
    """Represent a folded FOR, WHILE, FOR_ITEMS, or IF block.

    Rendered as a single box with header label and body summary text.

    Args:
        node_key (tuple): Stable identity shared across visualization phases.
        header_label (str): Bold control-flow header.
        body_lines (list[str]): Summary expressions rendered inside the box.
        affected_qubits (list[int]): Wires semantically used by the block.
        folded_width (float): Full horizontal width required by the summary.
        kind (VFoldedKind): Folded control-flow category.
        affected_qubits_precise (bool): Whether ``affected_qubits`` is exact.
            Defaults to True; conservative sets omit participation markers.
        condition_measure_node_key (tuple | None): Measurement node feeding an
            IF/WHILE condition. Defaults to None.
        condition_measure_qubit_indices (list[int]): Candidate condition-source
            wires. Defaults to an empty list.
    """

    node_key: tuple
    header_label: str
    body_lines: list[str]
    affected_qubits: list[int]
    folded_width: float
    kind: VFoldedKind
    affected_qubits_precise: bool = True
    condition_measure_node_key: tuple | None = None
    condition_measure_qubit_indices: list[int] = field(default_factory=list)


@dataclass
class VUnfoldedSequence:
    """Represent an unfolded FOR, FOR_ITEMS, IF, or WHILE sequence.

    ``iterations`` contains materialized iterations for FOR/FOR_ITEMS, true and
    optional false regions for IF, or one displayed body region for WHILE.

    Args:
        node_key (tuple): Stable identity shared across visualization phases.
        iterations (list[list[VisualNode]]): Ordered iteration or region nodes.
        affected_qubits (list[int]): Wires semantically used by the sequence.
        kind (VUnfoldedKind): Sequence or boxed-control-flow category.
        iteration_widths (list[float]): Analyzer width estimates aligned with
            ``iterations``. Defaults to an empty list.
        condition_label (str | None): IF/WHILE header. Defaults to None.
        affected_qubits_precise (bool): Whether ``affected_qubits`` is exact.
            Defaults to True.
        condition_label_width (float): Minimum first-region header width for IF
            or WHILE. Defaults to zero.
        branch_label_widths (list[float]): Minimum header widths aligned with
            boxed IF/WHILE regions. Defaults to an empty list.
        condition_measure_node_key (tuple | None): Measurement node feeding the
            condition. Defaults to None.
        condition_measure_qubit_indices (list[int]): Candidate source wires for
            the condition connector. Defaults to an empty list.
    """

    node_key: tuple
    iterations: list[list[VisualNode]]
    affected_qubits: list[int]
    kind: VUnfoldedKind
    iteration_widths: list[float] = field(default_factory=list)
    condition_label: str | None = None
    affected_qubits_precise: bool = True
    condition_label_width: float = 0.0
    branch_label_widths: list[float] = field(default_factory=list)
    condition_measure_node_key: tuple | None = None
    condition_measure_qubit_indices: list[int] = field(default_factory=list)


@dataclass
class VSkip:
    """Represent a zero-space QInit, Cast, or empty loop.

    Args:
        node_key (tuple): Stable identity for compatibility position maps.
            Defaults to an empty tuple.
    """

    node_key: tuple = ()


VisualNode = VGate | VInlineBlock | VFoldedBlock | VUnfoldedSequence | VSkip


@dataclass
class VisualCircuit:
    """Contain the root Visual IR tree and its wire metadata.

    Args:
        children (list[VisualNode]): Root visual nodes in program order.
        qubit_map (dict[str, int]): Logical qubit identifiers to wire indices.
        qubit_names (dict[int, str]): Display labels by wire index.
        num_qubits (int): Total number of wire rows.
        output_names (list[str]): Optional output labels retained for clients.
            Defaults to an empty list.
    """

    children: list[VisualNode]
    qubit_map: dict[str, int]
    qubit_names: dict[int, str]
    num_qubits: int
    output_names: list[str] = field(default_factory=list)
