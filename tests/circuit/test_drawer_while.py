"""Tests for WhileOperation circuit visualization.

A ``while`` loop is measurement-backed and cannot be unrolled at compile time,
so it is drawn like a single-branch ``if``: the loop body is shown once inside a
``while <cond>:`` box connected to the measurement that produces the condition.
These tests cover:

- the default unfolded rendering (body gates inside a ``while <cond>:`` box
  with a measurement connector, matching a single-branch ``if``);
- the folded rendering under ``fold_whiles=True`` (compact summary box,
  matching folded ``for``);
- nesting of ``while`` with ``if`` and ``for``.
"""

import matplotlib

matplotlib.use("Agg")

from collections.abc import Iterable, Iterator
from typing import Any

from matplotlib import colors as mcolors
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

import qamomile.circuit as qmc
from qamomile.circuit.visualization.analyzer import CircuitAnalyzer
from qamomile.circuit.visualization.drawer import _prepare_graph_for_visualization
from qamomile.circuit.visualization.style import DEFAULT_STYLE
from qamomile.circuit.visualization.visual_ir import (
    VFoldedBlock,
    VFoldedKind,
    VGate,
    VGateKind,
    VInlineBlock,
    VisualCircuit,
    VisualNode,
    VUnfoldedKind,
    VUnfoldedSequence,
)


@qmc.qkernel
def measurement_while() -> qmc.Bit:
    """Build a measurement-backed while loop with a loop-carried condition.

    Returns:
        qmc.Bit: The final loop-condition bit.
    """
    q = qmc.qubit("q")
    q = qmc.h(q)
    bit = qmc.measure(q)
    while bit:
        q2 = qmc.qubit("q2")
        q2 = qmc.h(q2)
        bit = qmc.measure(q2)
    return bit


@qmc.qkernel
def while_in_if() -> qmc.Bit:
    """Build a while loop nested inside the true branch of an if.

    Returns:
        qmc.Bit: The outer if-condition bit.
    """
    q0 = qmc.qubit("q0")
    q0 = qmc.x(q0)
    b0 = qmc.measure(q0)
    if b0:
        q1 = qmc.qubit("q1")
        q1 = qmc.h(q1)
        b1 = qmc.measure(q1)
        while b1:
            q2 = qmc.qubit("q2")
            q2 = qmc.h(q2)
            b1 = qmc.measure(q2)
    return b0


@qmc.qkernel
def for_in_while() -> qmc.Bit:
    """Build a for loop nested inside a while loop body.

    Returns:
        qmc.Bit: The final loop-condition bit.
    """
    q = qmc.qubit("q")
    q = qmc.h(q)
    bit = qmc.measure(q)
    while bit:
        q2 = qmc.qubit("q2")
        for _ in qmc.range(2):
            q2 = qmc.x(q2)
        bit = qmc.measure(q2)
    return bit


def _walk(nodes: Iterable[VisualNode]) -> Iterator[VisualNode]:
    """Yield every VisualNode reachable from a list of root nodes.

    Args:
        nodes (Iterable[VisualNode]): Root visual nodes to traverse.

    Yields:
        VisualNode: Each visual node reachable from ``nodes``.
    """
    for node in nodes:
        yield node
        if isinstance(node, VInlineBlock):
            yield from _walk(node.children)
        elif isinstance(node, VUnfoldedSequence):
            for iteration in node.iterations:
                yield from _walk(iteration)


def _visual_circuit(
    kernel: Any,
    *,
    fold_loops: bool = True,
    fold_whiles: bool = False,
    **bindings: Any,
) -> VisualCircuit:
    """Trace ``kernel`` and run the visualization analyzer in isolation.

    Args:
        kernel (Any): QKernel-like object to trace.
        fold_loops (bool): Whether ``for`` loop operations should be folded.
        fold_whiles (bool): Whether ``while`` loop operations should be folded.
            Defaults to False (the drawer default), so a while expands into a
            body-in-box.
        **bindings (Any): Concrete draw-time bindings for kernel parameters.

    Returns:
        VisualCircuit: Visual IR produced by ``CircuitAnalyzer``.
    """
    block = _prepare_graph_for_visualization(
        kernel._build_graph_for_visualization(**bindings)
    )
    analyzer = CircuitAnalyzer(
        block,
        DEFAULT_STYLE,
        inline=False,
        fold_loops=fold_loops,
        fold_ifs=False,
        fold_whiles=fold_whiles,
        expand_composite=False,
        inline_depth=None,
    )
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(block)
    return analyzer.build_visual_ir(block, qubit_map, qubit_names, num_qubits)


def _folded_whiles(vc: VisualCircuit) -> list[VFoldedBlock]:
    """Collect every folded WHILE box in a visual circuit.

    Args:
        vc (VisualCircuit): Visual circuit to scan.

    Returns:
        list[VFoldedBlock]: Folded WHILE blocks in traversal order.
    """
    return [
        n
        for n in _walk(vc.children)
        if isinstance(n, VFoldedBlock) and n.kind == VFoldedKind.WHILE
    ]


def _unfolded_whiles(vc: VisualCircuit) -> list[VUnfoldedSequence]:
    """Collect every unfolded WHILE sequence in a visual circuit.

    Args:
        vc (VisualCircuit): Visual circuit to scan.

    Returns:
        list[VUnfoldedSequence]: Unfolded WHILE sequences in traversal order.
    """
    return [
        n
        for n in _walk(vc.children)
        if isinstance(n, VUnfoldedSequence) and n.kind == VUnfoldedKind.WHILE
    ]


def _while_branch_boxes(fig: Figure) -> list[FancyBboxPatch]:
    """Collect WHILE-styled dashed boxes from a rendered figure.

    Args:
        fig (Figure): Rendered circuit figure.

    Returns:
        list[FancyBboxPatch]: While-colored dashed boxes.
    """
    ax = fig._qm_ax  # type: ignore[attr-defined]
    while_edge = mcolors.to_rgba(DEFAULT_STYLE.while_loop_edge_color)
    return [
        patch
        for patch in ax.patches
        if isinstance(patch, FancyBboxPatch)
        and patch.get_edgecolor() == while_edge
        and patch.get_linestyle() in ("--", (0, (6.4, 1.6)))
    ]


def _while_connector_lines(fig: Figure) -> list[Line2D]:
    """Collect solid WHILE-colored connector lines from a rendered figure.

    Args:
        fig (Figure): Rendered circuit figure.

    Returns:
        list[Line2D]: Lines that match the measurement-to-WHILE connector style.
    """
    ax = fig._qm_ax  # type: ignore[attr-defined]
    return [
        line
        for line in ax.lines
        if line.get_color() == DEFAULT_STYLE.while_loop_edge_color
        and line.get_linestyle() == "-"
    ]


class TestUnfoldedWhile:
    """The default renders a while as a body-in-box, like a single-branch if."""

    def test_unfolded_while_is_single_body_sequence(self):
        """A while unfolds to one WHILE sequence with a single body iteration."""
        vc = _visual_circuit(measurement_while)
        unfolded = _unfolded_whiles(vc)
        assert len(unfolded) == 1
        assert len(unfolded[0].iterations) == 1
        assert not _folded_whiles(vc)

    def test_unfolded_while_condition_label_spells_bit(self):
        """The unfolded box header names the measured condition bit."""
        vc = _visual_circuit(measurement_while)
        seq = _unfolded_whiles(vc)[0]
        assert seq.condition_label is not None
        assert seq.condition_label.startswith("while ")
        assert "measured" in seq.condition_label

    def test_unfolded_while_body_carries_its_gates(self):
        """The single iteration holds the loop body's H and measurement."""
        vc = _visual_circuit(measurement_while)
        seq = _unfolded_whiles(vc)[0]
        body = seq.iterations[0]
        gates = [n for n in _walk(body) if isinstance(n, VGate)]
        assert any(g.kind == VGateKind.GATE for g in gates)
        assert any(g.kind == VGateKind.MEASURE for g in gates)
        # The body operates on the loop-local wire (q2 -> wire 1), not q's wire.
        assert seq.affected_qubits == [1]

    def test_unfolded_while_carries_measurement_connector_metadata(self):
        """The unfolded box records the measurement feeding its condition."""
        vc = _visual_circuit(measurement_while)
        seq = _unfolded_whiles(vc)[0]
        assert seq.condition_measure_node_key is not None
        assert seq.condition_measure_qubit_indices == [0]

    def test_body_measurement_keeps_wire_running(self):
        """A measurement inside the while body must not terminate its wire.

        The loop body re-executes every iteration, so its measurement is
        mid-circuit — the wire has to continue under the rest of the while
        box, exactly like a measurement inside an if branch. Regression: the
        conditional-scope check only recognized the if markers, so the body
        measure carried ``terminates_wire=True`` and the wire was cut inside
        the box.
        """
        vc = _visual_circuit(measurement_while)
        seq = _unfolded_whiles(vc)[0]
        body_measures = [
            n
            for n in _walk(seq.iterations[0])
            if isinstance(n, VGate)
            and n.kind in (VGateKind.MEASURE, VGateKind.MEASURE_VECTOR)
        ]
        assert body_measures
        assert all(not m.terminates_wire for m in body_measures)
        # The top-level condition measurement still terminates normally.
        top_measures = [
            n
            for n in vc.children
            if isinstance(n, VGate) and n.kind == VGateKind.MEASURE
        ]
        assert all(m.terminates_wire for m in top_measures)


class TestFoldedWhile:
    """``fold_whiles=True`` collapses a while into a compact summary box."""

    def test_folded_while_is_single_box(self):
        """A measurement-backed while folds to one WHILE box."""
        vc = _visual_circuit(measurement_while, fold_whiles=True)
        folded = _folded_whiles(vc)
        assert len(folded) == 1
        assert not _unfolded_whiles(vc)

    def test_folded_while_header_spells_condition(self):
        """The folded box header names the measured condition bit."""
        vc = _visual_circuit(measurement_while, fold_whiles=True)
        folded = _folded_whiles(vc)[0]
        assert folded.header_label.startswith("while ")
        assert "measured" in folded.header_label

    def test_folded_while_body_summarizes_operations(self):
        """The folded box body lists the loop-body operations."""
        vc = _visual_circuit(measurement_while, fold_whiles=True)
        folded = _folded_whiles(vc)[0]
        assert any("h(" in line for line in folded.body_lines)
        assert any("measure(" in line for line in folded.body_lines)

    def test_folded_while_carries_measurement_connector_metadata(self):
        """The folded box records which measurement feeds its condition."""
        vc = _visual_circuit(measurement_while, fold_whiles=True)
        folded = _folded_whiles(vc)[0]
        assert folded.condition_measure_node_key is not None
        assert folded.condition_measure_qubit_indices == [0]


class TestNestedWhile:
    """While composes with if and for when nested either way."""

    def test_while_inside_if_unfolds_both_boxes(self):
        """The if's true branch contains the while as its own box."""
        vc = _visual_circuit(while_in_if)
        top_ifs = [
            n
            for n in vc.children
            if isinstance(n, VUnfoldedSequence) and n.kind == VUnfoldedKind.IF
        ]
        assert len(top_ifs) == 1
        inner_whiles = _unfolded_whiles(vc)
        assert len(inner_whiles) == 1
        # The nested while lives inside the if's true branch, not at top level.
        assert inner_whiles[0] not in vc.children

    def test_for_inside_while_body_unfolds_nested_for(self):
        """An unfolded while body carries the nested for (unrolled) and measure."""
        vc = _visual_circuit(for_in_while, fold_loops=False)
        seq = _unfolded_whiles(vc)[0]
        body = seq.iterations[0]
        nested_for = [
            n
            for n in body
            if isinstance(n, VUnfoldedSequence) and n.kind == VUnfoldedKind.FOR
        ]
        assert len(nested_for) == 1
        # ``for _ in range(2)`` unrolls to two iterations inside the body.
        assert len(nested_for[0].iterations) == 2
        assert any(
            isinstance(n, VGate) and n.kind == VGateKind.MEASURE for n in _walk(body)
        )


class TestDrawEndToEnd:
    """``QKernel.draw`` returns a Figure for every while flavor without raising."""

    def test_measurement_while_draws_in_both_modes(self):
        """A measurement-backed while draws unfolded (default) and folded."""
        assert isinstance(measurement_while.draw(), Figure)
        assert isinstance(measurement_while.draw(fold_whiles=True), Figure)

    def test_unfolded_while_draws_box_and_connector(self):
        """The default while draws its dashed box and measurement connector."""
        fig = measurement_while.draw()
        assert len(_while_branch_boxes(fig)) == 1
        assert len(_while_connector_lines(fig)) == 1

    def test_folded_while_draws_connector(self):
        """The folded while draws a connector from its condition measurement."""
        fig = measurement_while.draw(fold_whiles=True)
        assert len(_while_connector_lines(fig)) == 1

    def test_while_in_if_draws(self):
        """A while nested in an if draws in both modes without raising."""
        assert isinstance(while_in_if.draw(), Figure)
        assert isinstance(while_in_if.draw(fold_whiles=True), Figure)

    def test_for_in_while_draws(self):
        """A for nested in a while draws in both modes without raising."""
        assert isinstance(for_in_while.draw(), Figure)
        assert isinstance(for_in_while.draw(fold_loops=False), Figure)
