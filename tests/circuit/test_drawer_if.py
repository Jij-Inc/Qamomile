"""Tests for IfOperation circuit visualization.

Covers the three behaviors that make if/else drawable:

- compile-time resolvable conditions (bindings or traced constants) are lowered
  to the selected branch, so no if box is drawn;
- surviving conditions (measurement-backed runtime ``if``, symbolic classical
  ``if``) render as a folded summary box under ``fold_loops=True`` or as
  side-by-side if/else branch boxes under ``fold_loops=False``;
- nested ifs render recursively.
"""

import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure

import qamomile.circuit as qmc
from qamomile.circuit.frontend.handle import Qubit, UInt
from qamomile.circuit.ir.operation.control_flow import IfOperation
from qamomile.circuit.ir.operation.gate import GateOperationType
from qamomile.circuit.visualization.analyzer import CircuitAnalyzer
from qamomile.circuit.visualization.drawer import _prepare_graph_for_visualization
from qamomile.circuit.visualization.layout import CircuitLayoutEngine
from qamomile.circuit.visualization.style import DEFAULT_STYLE
from qamomile.circuit.visualization.visual_ir import (
    VFoldedBlock,
    VFoldedKind,
    VGate,
    VGateKind,
    VInlineBlock,
    VisualCircuit,
    VUnfoldedKind,
    VUnfoldedSequence,
)


@qmc.qkernel
def runtime_if(q0: Qubit, q1: Qubit) -> Qubit:
    """Measurement-backed if/else: X on the true branch, H on the else branch."""
    cond = qmc.measure(q0)
    if cond:
        q1 = qmc.x(q1)
    else:
        q1 = qmc.h(q1)
    return q1


@qmc.qkernel
def classical_if(q0: Qubit, flag: UInt) -> Qubit:
    """Classical-condition if/else driven by an integer ``flag`` parameter."""
    if flag == 1:
        q0 = qmc.x(q0)
    else:
        q0 = qmc.h(q0)
    return q0


@qmc.qkernel
def if_no_else(q0: Qubit, q1: Qubit) -> Qubit:
    """Measurement-backed if with no else branch."""
    cond = qmc.measure(q0)
    if cond:
        q1 = qmc.x(q1)
    return q1


@qmc.qkernel
def nested_if(q0: Qubit, q1: Qubit, q2: Qubit) -> Qubit:
    """Two-level nested if so recursive branch rendering can be checked."""
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


def _walk(nodes):
    """Yield every VisualNode reachable from a list of root nodes."""
    for node in nodes:
        yield node
        if isinstance(node, VInlineBlock):
            yield from _walk(node.children)
        elif isinstance(node, VUnfoldedSequence):
            for iteration in node.iterations:
                yield from _walk(iteration)


def _visual_circuit(kernel, *, fold_loops, **bindings) -> VisualCircuit:
    """Trace ``kernel`` and run the visualization analyzer in isolation."""
    block = _prepare_graph_for_visualization(
        kernel._build_graph_for_visualization(**bindings)
    )
    analyzer = CircuitAnalyzer(
        block,
        DEFAULT_STYLE,
        inline=False,
        fold_loops=fold_loops,
        expand_composite=False,
        inline_depth=None,
    )
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(block)
    return analyzer.build_visual_ir(block, qubit_map, qubit_names, num_qubits)


def _folded_ifs(vc: VisualCircuit) -> list[VFoldedBlock]:
    """Collect every folded IF box in a visual circuit."""
    return [
        n
        for n in _walk(vc.children)
        if isinstance(n, VFoldedBlock) and n.kind == VFoldedKind.IF
    ]


def _unfolded_ifs(vc: VisualCircuit) -> list[VUnfoldedSequence]:
    """Collect every unfolded IF sequence in a visual circuit."""
    return [
        n
        for n in _walk(vc.children)
        if isinstance(n, VUnfoldedSequence) and n.kind == VUnfoldedKind.IF
    ]


def _branch_gate_types(iteration) -> list[GateOperationType]:
    """Return the gate types of the (non-measurement) gates in one branch."""
    return [
        n.gate_type
        for n in _walk(iteration)
        if isinstance(n, VGate) and n.gate_type is not None
    ]


class TestFoldedIf:
    """``fold_loops=True`` renders a surviving if as a single folded box."""

    def test_runtime_if_folds_to_one_if_box(self):
        """A measurement-backed if becomes exactly one folded IF box."""
        vc = _visual_circuit(runtime_if, fold_loops=True)
        folded = _folded_ifs(vc)
        assert len(folded) == 1
        # The condition label spells the measurement bit, not a placeholder.
        assert folded[0].header_label.startswith("if ")
        assert "measured" in folded[0].header_label

    def test_symbolic_classical_if_label_shows_predicate(self):
        """An unbound classical if renders its predicate, e.g. ``if flag == 1:``."""
        vc = _visual_circuit(classical_if, fold_loops=True)
        folded = _folded_ifs(vc)
        assert len(folded) == 1
        assert folded[0].header_label == "if flag == 1:"


class TestUnfoldedIf:
    """``fold_loops=False`` renders a surviving if as side-by-side branches."""

    def test_runtime_if_unfolds_into_two_branches(self):
        """A measurement-backed if/else yields one IF sequence with two branches."""
        vc = _visual_circuit(runtime_if, fold_loops=False)
        unfolded = _unfolded_ifs(vc)
        assert len(unfolded) == 1
        assert len(unfolded[0].iterations) == 2
        assert unfolded[0].condition_label is not None
        assert "measured" in unfolded[0].condition_label

    def test_branches_carry_their_own_gates(self):
        """Iteration 0 holds the true-branch X; iteration 1 holds the else H."""
        vc = _visual_circuit(runtime_if, fold_loops=False)
        seq = _unfolded_ifs(vc)[0]
        assert _branch_gate_types(seq.iterations[0]) == [GateOperationType.X]
        assert _branch_gate_types(seq.iterations[1]) == [GateOperationType.H]

    def test_if_without_else_has_single_branch(self):
        """An if with no else unfolds to a single-branch IF sequence."""
        vc = _visual_circuit(if_no_else, fold_loops=False)
        unfolded = _unfolded_ifs(vc)
        assert len(unfolded) == 1
        assert len(unfolded[0].iterations) == 1
        assert _branch_gate_types(unfolded[0].iterations[0]) == [GateOperationType.X]

    def test_condition_label_width_reserved_for_layout(self):
        """The IF sequence carries a positive header width for layout to reserve."""
        vc = _visual_circuit(classical_if, fold_loops=False)
        seq = _unfolded_ifs(vc)[0]
        assert seq.condition_label == "if flag == 1:"
        assert seq.condition_label_width > 0.0

    def test_branch_label_widths_align_with_branches(self):
        """One header-box width per branch; the first equals condition_label_width.

        The renderer sizes each branch box to these estimates (the same numbers
        the layout reserves), which is what keeps the boxes gap-free and the
        labels from overflowing.
        """
        vc = _visual_circuit(classical_if, fold_loops=False)
        seq = _unfolded_ifs(vc)[0]
        assert len(seq.branch_label_widths) == len(seq.iterations) == 2
        assert all(w > 0.0 for w in seq.branch_label_widths)
        assert seq.branch_label_widths[0] == seq.condition_label_width


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

    def test_plain_two_qubit_circuit_keeps_base_spacing(self):
        """Without an if header, adjacent wires keep the default spacing."""

        @qmc.qkernel
        def plain(q0: Qubit, q1: Qubit) -> tuple[Qubit, Qubit]:
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
        """Collect every measurement VGate in a visual circuit."""
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
        for fold in (True, False):
            vc = _visual_circuit(classical_if, fold_loops=fold, flag=1)
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
        vc = _visual_circuit(classical_if, fold_loops=True, flag=0)
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
        """Measurement-backed if draws in both folded and unfolded modes."""
        assert isinstance(runtime_if.draw(fold_loops=True), Figure)
        assert isinstance(runtime_if.draw(fold_loops=False), Figure)

    def test_symbolic_classical_if_draws(self):
        """Unbound classical if draws in both modes."""
        assert isinstance(classical_if.draw(fold_loops=True), Figure)
        assert isinstance(classical_if.draw(fold_loops=False), Figure)

    def test_bound_classical_if_draws(self):
        """Bound classical if (compile-time resolved) draws without an if box."""
        assert isinstance(classical_if.draw(fold_loops=False, flag=1), Figure)

    def test_nested_if_draws(self):
        """Nested if draws in both modes."""
        assert isinstance(nested_if.draw(fold_loops=True), Figure)
        assert isinstance(nested_if.draw(fold_loops=False), Figure)
