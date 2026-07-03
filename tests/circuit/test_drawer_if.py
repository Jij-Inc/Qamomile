"""Tests for IfOperation circuit visualization.

Covers the three behaviors that make if/else drawable:

- compile-time resolvable conditions (bindings or traced constants) are lowered
  to the selected branch, so no if box is drawn;
- surviving conditions (measurement-backed runtime ``if``, symbolic classical
  ``if``) render as side-by-side if/else branch boxes by default, or as a
  folded summary box under ``fold_ifs=True``;
- nested ifs render recursively.
"""

import matplotlib

matplotlib.use("Agg")

from collections.abc import Iterable, Iterator
from dataclasses import replace
import math
from typing import Any

from matplotlib import colors as mcolors
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

import qamomile.circuit as qmc
from qamomile.circuit.frontend.handle import Qubit, UInt
from qamomile.circuit.ir.operation.control_flow import IfOperation
from qamomile.circuit.ir.operation.gate import GateOperationType
from qamomile.circuit.visualization.analyzer import CircuitAnalyzer
from qamomile.circuit.visualization.drawer import (
    MatplotlibDrawer,
    _prepare_graph_for_visualization,
)
from qamomile.circuit.visualization.geometry import compute_border_padding
from qamomile.circuit.visualization.layout import CircuitLayoutEngine
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
def runtime_if(q0: Qubit, q1: Qubit) -> Qubit:
    """Build a measurement-backed if/else example.

    Args:
        q0 (Qubit): Qubit measured to decide the branch.
        q1 (Qubit): Qubit transformed by the selected branch.

    Returns:
        Qubit: Updated ``q1``.
    """
    cond = qmc.measure(q0)
    if cond:
        q1 = qmc.x(q1)
    else:
        q1 = qmc.h(q1)
    return q1


@qmc.qkernel
def inline_measure_if(q0: Qubit, q1: Qubit) -> Qubit:
    """Build an if whose condition is an inline measurement call.

    Args:
        q0 (Qubit): Qubit measured directly in the condition.
        q1 (Qubit): Qubit transformed by the selected branch.

    Returns:
        Qubit: Updated ``q1``.
    """
    if qmc.measure(q0):
        q1 = qmc.x(q1)
    else:
        q1 = qmc.h(q1)
    return q1


@qmc.qkernel
def vector_measure_element_if() -> Qubit:
    """Build an if whose condition is one bit of a vector measurement.

    Returns:
        Qubit: Updated target qubit.
    """
    qs = qmc.qubit_array(3, "qs")
    target = qmc.qubit("target")
    bits = qmc.measure(qs)
    if bits[1]:
        target = qmc.x(target)
    else:
        target = qmc.h(target)
    return target


@qmc.qkernel
def symbolic_vector_measure_element_if(index: UInt) -> Qubit:
    """Build an if whose vector-measurement element index is symbolic.

    Args:
        index (UInt): Symbolic measured-bit index used as the condition.

    Returns:
        Qubit: Updated target qubit.
    """
    qs = qmc.qubit_array(3, "qs")
    target = qmc.qubit("target")
    bits = qmc.measure(qs)
    if bits[index]:
        target = qmc.x(target)
    else:
        target = qmc.h(target)
    return target


@qmc.qkernel
def classical_if(q0: Qubit, flag: UInt) -> Qubit:
    """Build a symbolic classical-condition if/else example.

    Args:
        q0 (Qubit): Qubit transformed by the selected branch.
        flag (UInt): Classical flag compared with ``1``.

    Returns:
        Qubit: Updated ``q0``.
    """
    if flag == 1:
        q0 = qmc.x(q0)
    else:
        q0 = qmc.h(q0)
    return q0


@qmc.qkernel
def if_no_else(q0: Qubit, q1: Qubit) -> Qubit:
    """Build a measurement-backed if example with no else branch.

    Args:
        q0 (Qubit): Qubit measured to decide the branch.
        q1 (Qubit): Qubit transformed only when the condition is true.

    Returns:
        Qubit: Updated ``q1``.
    """
    cond = qmc.measure(q0)
    if cond:
        q1 = qmc.x(q1)
    return q1


@qmc.qkernel
def empty_true_if(q0: Qubit, q1: Qubit) -> Qubit:
    """Build a measurement-backed if whose true branch is empty.

    Args:
        q0 (Qubit): Qubit measured to decide the branch.
        q1 (Qubit): Qubit transformed only by the else branch.

    Returns:
        Qubit: Updated ``q1``.
    """
    cond = qmc.measure(q0)
    if cond:
        pass
    else:
        q1 = qmc.x(q1)
    return q1


@qmc.qkernel
def empty_single_branch_if(q0: Qubit, q1: Qubit) -> Qubit:
    """Build a measurement-backed if whose only branch is empty.

    Args:
        q0 (Qubit): Qubit measured to decide the branch.
        q1 (Qubit): Qubit returned unchanged.

    Returns:
        Qubit: Unchanged ``q1``.
    """
    if qmc.measure(q0):
        pass
    return q1


@qmc.qkernel
def symbolic_empty_single_branch_if(q0: Qubit, flag: UInt) -> Qubit:
    """Build a symbolic if whose only branch is empty.

    Args:
        q0 (Qubit): Qubit returned unchanged.
        flag (UInt): Symbolic value that keeps the IF at draw time.

    Returns:
        Qubit: Unchanged ``q0``.
    """
    if flag == 1:
        pass
    return q0


@qmc.qkernel
def nested_if(q0: Qubit, q1: Qubit, q2: Qubit) -> Qubit:
    """Build a two-level nested if example.

    Args:
        q0 (Qubit): Qubit measured by the outer if.
        q1 (Qubit): Qubit measured by the nested if.
        q2 (Qubit): Qubit transformed by the selected leaf branch.

    Returns:
        Qubit: Updated ``q2``.
    """
    c0 = qmc.measure(q0)
    if c0:
        c1 = qmc.measure(q1)
        if c1:
            q2 = qmc.x(q2)
        else:
            q2 = qmc.h(q2)
    else:
        q2 = qmc.z(q2)
    return q2


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
    kernel: Any, *, fold_loops: bool = True, fold_ifs: bool = False, **bindings: Any
) -> VisualCircuit:
    """Trace ``kernel`` and run the visualization analyzer in isolation.

    Args:
        kernel (Any): QKernel-like object to trace.
        fold_loops (bool): Whether loop operations should be folded.
        fold_ifs (bool): Whether if operations should be folded.
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
        fold_ifs=fold_ifs,
        expand_composite=False,
        inline_depth=None,
    )
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(block)
    return analyzer.build_visual_ir(block, qubit_map, qubit_names, num_qubits)


def _folded_ifs(vc: VisualCircuit) -> list[VFoldedBlock]:
    """Collect every folded IF box in a visual circuit.

    Args:
        vc (VisualCircuit): Visual circuit to scan.

    Returns:
        list[VFoldedBlock]: Folded IF blocks in traversal order.
    """
    return [
        n
        for n in _walk(vc.children)
        if isinstance(n, VFoldedBlock) and n.kind == VFoldedKind.IF
    ]


def _unfolded_ifs(vc: VisualCircuit) -> list[VUnfoldedSequence]:
    """Collect every unfolded IF sequence in a visual circuit.

    Args:
        vc (VisualCircuit): Visual circuit to scan.

    Returns:
        list[VUnfoldedSequence]: Unfolded IF sequences in traversal order.
    """
    return [
        n
        for n in _walk(vc.children)
        if isinstance(n, VUnfoldedSequence) and n.kind == VUnfoldedKind.IF
    ]


def _branch_gate_types(iteration: Iterable[VisualNode]) -> list[GateOperationType]:
    """Return the gate types of the non-measurement gates in one branch.

    Args:
        iteration (Iterable[VisualNode]): Visual nodes inside one IF branch.

    Returns:
        list[GateOperationType]: Gate types seen in traversal order.
    """
    return [
        n.gate_type
        for n in _walk(iteration)
        if isinstance(n, VGate) and n.gate_type is not None
    ]


def _if_connector_lines(fig: Figure) -> list[Line2D]:
    """Collect solid IF-colored connector lines from a rendered figure.

    Args:
        fig (Figure): Rendered circuit figure.

    Returns:
        list[Line2D]: Lines that match the measurement-to-IF connector style.
    """
    ax = fig._qm_ax  # type: ignore[attr-defined]
    return [
        line
        for line in ax.lines
        if line.get_color() == DEFAULT_STYLE.if_edge_color
        and line.get_linestyle() == "-"
    ]


def _if_branch_boxes(fig: Figure) -> list[FancyBboxPatch]:
    """Collect IF branch boxes from a rendered figure.

    Args:
        fig (Figure): Rendered circuit figure.

    Returns:
        list[FancyBboxPatch]: IF-colored branch boxes.
    """
    ax = fig._qm_ax  # type: ignore[attr-defined]
    if_edge = mcolors.to_rgba(DEFAULT_STYLE.if_edge_color)
    return [
        patch
        for patch in ax.patches
        if isinstance(patch, FancyBboxPatch)
        and patch.get_edgecolor() == if_edge
        and patch.get_facecolor()[3] == 0
        and patch.get_linestyle() == "-."
    ]


class TestFoldedIf:
    """``fold_ifs=True`` renders a surviving if as a single folded box."""

    def test_runtime_if_folds_to_one_if_box(self):
        """A measurement-backed if becomes one folded IF box with both branches."""
        vc = _visual_circuit(runtime_if, fold_ifs=True)
        folded = _folded_ifs(vc)
        assert len(folded) == 1
        # The condition label spells the measurement bit, not a placeholder.
        assert folded[0].header_label.startswith("if ")
        assert "measured" in folded[0].header_label
        assert any(line.startswith("else:") for line in folded[0].body_lines)
        assert any("h(" in line for line in folded[0].body_lines)

    def test_runtime_if_keeps_measurement_connector_metadata(self):
        """A folded measurement-backed if knows which measurement feeds it."""
        vc = _visual_circuit(runtime_if, fold_ifs=True)
        folded = _folded_ifs(vc)[0]
        assert folded.condition_measure_node_key is not None
        assert folded.condition_measure_qubit_indices == [0]

    def test_symbolic_classical_if_label_shows_predicate(self):
        """An unbound classical if renders its predicate, e.g. ``if flag == 1:``."""
        vc = _visual_circuit(classical_if, fold_ifs=True)
        folded = _folded_ifs(vc)
        assert len(folded) == 1
        assert folded[0].header_label == "if flag == 1:"
        assert any(line.startswith("else:") for line in folded[0].body_lines)

    def test_empty_true_branch_folded_summary_keeps_else(self):
        """An empty true branch still shows pass and else in folded form."""
        vc = _visual_circuit(empty_true_if, fold_ifs=True)
        folded = _folded_ifs(vc)
        assert len(folded) == 1
        assert folded[0].body_lines[0] == "pass"
        assert any(line.startswith("else:") for line in folded[0].body_lines)


class TestUnfoldedIf:
    """The default renders surviving ifs as side-by-side branches."""

    def test_runtime_if_unfolds_into_two_branches(self):
        """A measurement-backed if/else yields one IF sequence with two branches."""
        vc = _visual_circuit(runtime_if)
        unfolded = _unfolded_ifs(vc)
        assert len(unfolded) == 1
        assert len(unfolded[0].iterations) == 2
        assert unfolded[0].condition_label is not None
        assert "measured" in unfolded[0].condition_label

    def test_assigned_measurement_condition_carries_connector_metadata(self):
        """``b = measure(q); if b:`` records the measurement feeding the IF."""
        vc = _visual_circuit(runtime_if)
        seq = _unfolded_ifs(vc)[0]
        assert seq.condition_measure_node_key is not None
        assert seq.condition_measure_qubit_indices == [0]

    def test_inline_measurement_condition_carries_connector_metadata(self):
        """``if measure(q):`` records the measurement feeding the IF."""
        vc = _visual_circuit(inline_measure_if)
        seq = _unfolded_ifs(vc)[0]
        assert seq.condition_measure_node_key is not None
        assert seq.condition_measure_qubit_indices == [0]

    def test_vector_measurement_element_condition_uses_selected_wire(self):
        """``if bits[1]:`` connects from the measured vector's second wire."""
        vc = _visual_circuit(vector_measure_element_if)
        measure = next(
            n
            for n in _walk(vc.children)
            if isinstance(n, VGate) and n.kind == VGateKind.MEASURE_VECTOR
        )
        seq = _unfolded_ifs(vc)[0]
        assert measure.qubit_indices == [0, 1, 2]
        assert seq.condition_measure_node_key == measure.node_key
        assert seq.condition_measure_qubit_indices == [1]

    def test_symbolic_vector_measurement_element_keeps_all_candidate_wires(self):
        """``if bits[i]:`` records all candidates when ``i`` is unresolved."""
        vc = _visual_circuit(symbolic_vector_measure_element_if)
        measure = next(
            n
            for n in _walk(vc.children)
            if isinstance(n, VGate) and n.kind == VGateKind.MEASURE_VECTOR
        )
        seq = _unfolded_ifs(vc)[0]
        assert measure.qubit_indices == [0, 1, 2]
        assert seq.condition_measure_node_key == measure.node_key
        assert seq.condition_measure_qubit_indices == [0, 1, 2]

    def test_symbolic_classical_condition_has_no_measurement_connector(self):
        """A non-measurement condition does not draw as measurement-derived."""
        vc = _visual_circuit(classical_if)
        seq = _unfolded_ifs(vc)[0]
        assert seq.condition_measure_node_key is None
        assert seq.condition_measure_qubit_indices == []

    def test_branches_carry_their_own_gates(self):
        """Iteration 0 holds the true-branch X; iteration 1 holds the else H."""
        vc = _visual_circuit(runtime_if)
        seq = _unfolded_ifs(vc)[0]
        assert _branch_gate_types(seq.iterations[0]) == [GateOperationType.X]
        assert _branch_gate_types(seq.iterations[1]) == [GateOperationType.H]

    def test_if_without_else_has_single_branch(self):
        """An if with no else unfolds to a single-branch IF sequence."""
        vc = _visual_circuit(if_no_else)
        unfolded = _unfolded_ifs(vc)
        assert len(unfolded) == 1
        assert len(unfolded[0].iterations) == 1
        assert _branch_gate_types(unfolded[0].iterations[0]) == [GateOperationType.X]

    def test_empty_true_branch_is_kept_for_rendering(self):
        """An empty true branch still occupies the IF branch slot."""
        vc = _visual_circuit(empty_true_if)
        unfolded = _unfolded_ifs(vc)
        assert len(unfolded) == 1
        assert len(unfolded[0].iterations) == 2
        assert unfolded[0].iterations[0] == []
        assert _branch_gate_types(unfolded[0].iterations[1]) == [GateOperationType.X]

    def test_empty_single_branch_uses_condition_measure_wire(self):
        """An empty single-branch IF anchors to its condition measurement wire."""
        vc = _visual_circuit(empty_single_branch_if)
        unfolded = _unfolded_ifs(vc)
        assert len(unfolded) == 1
        assert unfolded[0].iterations == [[]]
        assert unfolded[0].affected_qubits == [0]
        assert unfolded[0].condition_measure_qubit_indices == [0]

    def test_condition_label_width_reserved_for_layout(self):
        """The IF sequence carries a positive header width for layout to reserve."""
        vc = _visual_circuit(classical_if)
        seq = _unfolded_ifs(vc)[0]
        assert seq.condition_label == "if flag == 1:"
        assert seq.condition_label_width > 0.0

    def test_branch_label_widths_align_with_branches(self):
        """One header-box width per branch; the first equals condition_label_width.

        The renderer sizes each branch box to these estimates (the same numbers
        the layout reserves), which is what keeps the boxes gap-free and the
        labels from overflowing.
        """
        vc = _visual_circuit(classical_if)
        seq = _unfolded_ifs(vc)[0]
        assert len(seq.branch_label_widths) == len(seq.iterations) == 2
        assert all(w > 0.0 for w in seq.branch_label_widths)
        assert seq.branch_label_widths[0] == seq.condition_label_width

    def test_fold_loops_does_not_fold_if(self):
        """Loop folding leaves if/else rendering expanded unless fold_ifs is set."""
        vc = _visual_circuit(runtime_if, fold_loops=True)
        assert not _folded_ifs(vc)
        assert len(_unfolded_ifs(vc)) == 1


class TestUnfoldedIfSpacing:
    """The if/else header reserves vertical room so it clears the wire above."""

    def test_header_label_does_not_crowd_wire_above(self):
        """The gap above the if's top wire exceeds the default wire spacing."""
        vc = _visual_circuit(runtime_if, fold_loops=False)
        layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(vc)
        # runtime_if: q0 carries the measurement, q1 carries the if/else box.
        # The "if q0_measured:" header sits above q1 (between q0 and q1), so the
        # q0->q1 gap must exceed the default wire spacing to make room for it.
        gap = layout.qubit_y[0] - layout.qubit_y[1]
        assert gap > DEFAULT_STYLE.qubit_base_spacing

    def test_folded_if_body_does_not_crowd_wire_above(self):
        """A multi-line folded IF box reserves vertical room above its wire."""
        vc = _visual_circuit(runtime_if, fold_ifs=True)
        layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(vc)
        # The folded IF is centered on q1 but has three text lines, so q0->q1
        # spacing must grow to keep the box from overlapping q0's measurement.
        gap = layout.qubit_y[0] - layout.qubit_y[1]
        assert gap > DEFAULT_STYLE.qubit_base_spacing

    def test_plain_two_qubit_circuit_keeps_base_spacing(self):
        """Without an if header, adjacent wires keep the default spacing."""

        @qmc.qkernel
        def plain(q0: Qubit, q1: Qubit) -> tuple[Qubit, Qubit]:
            """Build a plain two-qubit circuit without if labels.

            Args:
                q0 (Qubit): Qubit receiving an H gate.
                q1 (Qubit): Qubit receiving an X gate.

            Returns:
                tuple[Qubit, Qubit]: Updated ``q0`` and ``q1``.
            """
            q0 = qmc.h(q0)
            q1 = qmc.x(q1)
            return q0, q1

        vc = _visual_circuit(plain, fold_loops=False)
        layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(vc)
        gap = layout.qubit_y[0] - layout.qubit_y[1]
        assert gap == DEFAULT_STYLE.qubit_base_spacing


class TestBranchMeasurementWire:
    """A measurement inside an if branch is mid-circuit and keeps its wire."""

    def _measures(self, vc: VisualCircuit) -> list[VGate]:
        """Collect every measurement VGate in a visual circuit.

        Args:
            vc (VisualCircuit): Visual circuit to scan.

        Returns:
            list[VGate]: Measurement gates in traversal order.
        """
        return [
            n
            for n in _walk(vc.children)
            if isinstance(n, VGate)
            and n.kind in (VGateKind.MEASURE, VGateKind.MEASURE_VECTOR)
        ]

    def test_branch_measure_flagged_mid_circuit(self):
        """nested_if: the top-level measure terminates, the branch one doesn't."""
        vc = _visual_circuit(nested_if, fold_loops=False)
        measures = self._measures(vc)
        # nested_if measures q0 at top level and q1 inside the outer true branch.
        terminating = [m for m in measures if m.terminates_wire]
        mid_circuit = [m for m in measures if not m.terminates_wire]
        assert len(terminating) == 1
        assert len(mid_circuit) == 1

    def test_branch_measure_wire_continues_in_layout(self):
        """The mid-circuit measure's wire is not terminated by the layout."""
        vc = _visual_circuit(nested_if, fold_loops=False)
        measures = self._measures(vc)
        terminating = next(m for m in measures if m.terminates_wire)
        mid_circuit = next(m for m in measures if not m.terminates_wire)

        layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(vc)
        # The top-level measure ends its wire; the branch measure does not, so
        # the qubit's wire keeps running through the else branch's x-range.
        assert terminating.qubit_indices[0] in layout.qubit_end_positions
        assert mid_circuit.qubit_indices[0] not in layout.qubit_end_positions

    def test_top_level_measure_still_terminates(self):
        """A plain top-level measurement keeps the usual wire termination."""
        vc = _visual_circuit(runtime_if, fold_loops=False)
        measures = self._measures(vc)
        # runtime_if measures only q0, at top level.
        assert len(measures) == 1
        assert measures[0].terminates_wire is True
        layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(vc)
        assert measures[0].qubit_indices[0] in layout.qubit_end_positions


class TestCompileTimeResolution:
    """Bound or constant conditions are lowered away before drawing."""

    def test_bound_condition_is_lowered_in_block(self):
        """Binding ``flag`` removes the IfOperation from the prepared block."""
        block = _prepare_graph_for_visualization(
            classical_if._build_graph_for_visualization(flag=1)
        )
        assert not any(isinstance(op, IfOperation) for op in block.operations)

    def test_symbolic_condition_keeps_ifoperation(self):
        """Without a binding the prepared graph keeps the IfOperation."""
        block = _prepare_graph_for_visualization(
            classical_if._build_graph_for_visualization()
        )
        assert any(isinstance(op, IfOperation) for op in block.operations)

    def test_bound_true_draws_only_selected_branch(self):
        """``flag=1`` collapses to the true branch (X) with no if box."""
        for fold_ifs in (False, True):
            vc = _visual_circuit(classical_if, fold_ifs=fold_ifs, flag=1)
            assert not _folded_ifs(vc)
            assert not _unfolded_ifs(vc)
            gates = [
                n.gate_type
                for n in _walk(vc.children)
                if isinstance(n, VGate) and n.gate_type is not None
            ]
            assert gates == [GateOperationType.X]

    def test_bound_false_draws_only_else_branch(self):
        """``flag=0`` collapses to the else branch (H) with no if box."""
        vc = _visual_circuit(classical_if, fold_ifs=True, flag=0)
        assert not _folded_ifs(vc)
        gates = [
            n.gate_type
            for n in _walk(vc.children)
            if isinstance(n, VGate) and n.gate_type is not None
        ]
        assert gates == [GateOperationType.H]


class TestNestedIf:
    """Nested ifs render recursively."""

    def test_nested_if_unfolds_recursively(self):
        """The outer if's true branch contains the inner if as its own box."""
        vc = _visual_circuit(nested_if, fold_loops=False)
        top_level = [
            n
            for n in vc.children
            if isinstance(n, VUnfoldedSequence) and n.kind == VUnfoldedKind.IF
        ]
        assert len(top_level) == 1
        true_branch = top_level[0].iterations[0]
        inner = [
            n
            for n in _walk(true_branch)
            if isinstance(n, VUnfoldedSequence) and n.kind == VUnfoldedKind.IF
        ]
        assert len(inner) == 1
        # Inner if/else carries X (true) and H (else); outer else carries Z.
        assert _branch_gate_types(inner[0].iterations[0]) == [GateOperationType.X]
        assert _branch_gate_types(inner[0].iterations[1]) == [GateOperationType.H]
        assert _branch_gate_types(top_level[0].iterations[1]) == [GateOperationType.Z]


class TestDrawEndToEnd:
    """``QKernel.draw`` returns a Figure for every if flavor without raising."""

    def test_runtime_if_draws(self):
        """Measurement-backed if draws in unfolded and folded-if modes."""
        assert isinstance(runtime_if.draw(), Figure)
        assert isinstance(runtime_if.draw(fold_ifs=True), Figure)

    def test_runtime_if_draws_measurement_connector(self):
        """Measurement-backed if draws a solid connector from measure to IF."""
        assert len(_if_connector_lines(runtime_if.draw())) == 1
        assert len(_if_connector_lines(runtime_if.draw(fold_ifs=True))) == 1

    def test_inline_measurement_if_draws_connector(self):
        """An inline measurement condition draws the same connector."""
        assert len(_if_connector_lines(inline_measure_if.draw())) == 1

    def test_vector_measurement_element_if_draws_connector(self):
        """A vector-measurement element condition draws one connector."""
        assert len(_if_connector_lines(vector_measure_element_if.draw())) == 1

    def test_symbolic_vector_measurement_element_skips_connector(self):
        """Ambiguous vector-measurement elements do not draw a connector."""
        assert len(_if_connector_lines(symbolic_vector_measure_element_if.draw())) == 0

    def test_symbolic_classical_if_draws(self):
        """Unbound classical if draws in unfolded and folded-if modes."""
        assert isinstance(classical_if.draw(), Figure)
        assert isinstance(classical_if.draw(fold_ifs=True), Figure)

    def test_bound_classical_if_draws(self):
        """Bound classical if (compile-time resolved) draws without an if box."""
        assert isinstance(classical_if.draw(fold_loops=False, flag=1), Figure)

    def test_nested_if_draws(self):
        """Nested if draws in unfolded and folded-if modes."""
        assert isinstance(nested_if.draw(), Figure)
        assert isinstance(nested_if.draw(fold_ifs=True), Figure)

    def test_empty_true_branch_draws_if_label(self):
        """The renderer keeps the IF label even when the true branch is empty."""
        fig = empty_true_if.draw(fold_loops=False)
        ax = fig._qm_ax  # type: ignore[attr-defined]
        labels = [text.get_text() for text in ax.texts]
        assert any(label.startswith("if ") for label in labels)
        assert "else:" in labels

    def test_empty_single_branch_draws_if_label(self):
        """The renderer keeps an empty single-branch IF visible."""
        for fig in (
            empty_single_branch_if.draw(fold_loops=False),
            empty_single_branch_if.draw(fold_loops=False, fold_ifs=True),
        ):
            ax = fig._qm_ax  # type: ignore[attr-defined]
            labels = [text.get_text() for text in ax.texts]
            assert any(label.startswith("if ") for label in labels)
            assert len(_if_connector_lines(fig)) == 1

    def test_empty_single_branch_uses_layout_reserved_span(self):
        """Empty IF branch boxes prefer the layout-reserved x-span."""
        style = replace(DEFAULT_STYLE, gate_gap=0.1)
        fig = MatplotlibDrawer(
            empty_single_branch_if._build_graph_for_visualization(), style
        ).draw(fold_loops=False)
        connector = _if_connector_lines(fig)[0]
        measure_right = connector.get_xdata()[0]
        expected_left = measure_right + compute_border_padding(style, depth=0)
        branch_box = _if_branch_boxes(fig)[0]
        assert math.isclose(branch_box.get_x(), expected_left)

    def test_symbolic_empty_single_branch_draws_if_label(self):
        """The renderer keeps an empty symbolic single-branch IF visible."""
        for fig in (
            symbolic_empty_single_branch_if.draw(fold_loops=False),
            symbolic_empty_single_branch_if.draw(fold_loops=False, fold_ifs=True),
        ):
            ax = fig._qm_ax  # type: ignore[attr-defined]
            labels = [text.get_text() for text in ax.texts]
            assert any(label.startswith("if ") for label in labels)
