"""Regression tests for loop-carried values in circuit visualization."""

import matplotlib

matplotlib.use("Agg")

import qamomile.circuit as qmc
from qamomile.circuit.visualization.analyzer import CircuitAnalyzer
from qamomile.circuit.visualization.drawer import _prepare_graph_for_visualization
from qamomile.circuit.visualization.style import DEFAULT_STYLE
from qamomile.circuit.visualization.visual_ir import (
    VFoldedBlock,
    VGate,
    VInlineBlock,
    VUnfoldedKind,
    VUnfoldedSequence,
)


@qmc.qkernel
def _angle_carry() -> qmc.Bit:
    """Apply each successive carried angle in a two-trip loop."""
    q = qmc.qubit("q")
    angle = qmc.float_(0.0)
    for _i in qmc.range(2):
        q = qmc.rx(q, angle)
        angle = angle + 1.0
    return qmc.measure(q)


@qmc.qkernel
def _post_loop_angle() -> qmc.Bit:
    """Use the final carried angle after a two-trip loop."""
    q = qmc.qubit("q")
    angle = qmc.float_(0.0)
    for _i in qmc.range(2):
        q = qmc.rx(q, angle)
        angle = angle + 1.0
    q = qmc.rz(q, angle)
    return qmc.measure(q)


@qmc.qkernel
def _nested_angle_carry() -> qmc.Bit:
    """Carry one angle through nested two-trip range loops."""
    q = qmc.qubit("q")
    angle = qmc.float_(0.0)
    for _i in qmc.range(2):
        for _j in qmc.range(2):
            q = qmc.rx(q, angle)
            angle = angle + 1.0
    return qmc.measure(q)


@qmc.qkernel
def _items_angle_carry(
    deltas: qmc.Dict[qmc.UInt, qmc.Float],
) -> qmc.Bit:
    """Carry an angle through materialized dictionary entries."""
    q = qmc.qubit("q")
    angle = qmc.float_(0.0)
    for _key, delta in deltas.items():
        q = qmc.rx(q, angle)
        angle = angle + delta
    return qmc.measure(q)


@qmc.qkernel
def _branch_angle_carry() -> qmc.Bit:
    """Update a carried angle through a loop-index-dependent branch."""
    q = qmc.qubit("q")
    angle = qmc.float_(0.0)
    for i in qmc.range(2):
        if i == 0:
            angle = angle + 1.0
        else:
            angle = angle + 2.0
        q = qmc.rx(q, angle)
    return qmc.measure(q)


@qmc.qkernel
def _carryless_outer_with_inner_carry() -> qmc.Vector[qmc.Bit]:
    """Consume an inner loop result inside a carry-less outer loop body."""
    q = qmc.qubit_array(3, "q")
    for _outer in qmc.range(2):
        index = qmc.uint(0)
        for _inner in qmc.range(2):
            index = index + 1
        q[index] = qmc.x(q[index])
    return qmc.measure(q)


def _walk(nodes):
    """Yield every visual node recursively in display order."""
    for node in nodes:
        yield node
        if isinstance(node, VInlineBlock):
            yield from _walk(node.children)
        elif isinstance(node, VUnfoldedSequence):
            for iteration in node.iterations:
                yield from _walk(iteration)


def _gate_labels(kernel, *, fold_loops=False, **bindings) -> list[str]:
    """Return all visual gate labels for an unfolded kernel."""
    graph = _prepare_graph_for_visualization(
        kernel._build_graph_for_visualization(**bindings)
    )
    analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, fold_loops=fold_loops)
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
    visual = analyzer.build_visual_ir(graph, qubit_map, qubit_names, num_qubits)
    return [node.label for node in _walk(visual.children) if isinstance(node, VGate)]


def _unfolded_angle_labels() -> list[str]:
    """Return gate labels from each materialized loop iteration."""
    graph = _prepare_graph_for_visualization(
        _angle_carry._build_graph_for_visualization()
    )
    analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, fold_loops=False)
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
    visual = analyzer.build_visual_ir(graph, qubit_map, qubit_names, num_qubits)
    loop = next(
        node
        for node in visual.children
        if isinstance(node, VUnfoldedSequence) and node.kind is VUnfoldedKind.FOR
    )
    return [
        gate.label
        for iteration in loop.iterations
        for gate in iteration
        if isinstance(gate, VGate)
    ]


def test_analyzer_threads_region_arg_between_unfolded_iterations() -> None:
    """The second loop iteration reads the first iteration's carried result."""
    assert _unfolded_angle_labels() == ["$R_x$(0)", "$R_x$(1.00)"]


def test_public_draw_threads_region_arg_between_unfolded_iterations() -> None:
    """The public draw path renders each loop-carried angle exactly once."""
    figure = _angle_carry.draw(fold_loops=False)
    labels = [text.get_text() for axes in figure.axes for text in axes.texts]

    assert labels.count("$R_x$(0)") == 1
    assert labels.count("$R_x$(1.00)") == 1


def test_analyzer_publishes_final_region_arg_after_loop() -> None:
    """A gate after the loop sees the final carried result value."""
    assert _gate_labels(_post_loop_angle) == [
        "$R_x$(0)",
        "$R_x$(1.00)",
        "$R_z$(2.00)",
        "M",
    ]


def test_folded_analyzer_publishes_final_region_arg_after_loop() -> None:
    """A folded finite loop still exposes its final carry to later gates."""
    assert _gate_labels(_post_loop_angle, fold_loops=True) == [
        "$R_z$(2.00)",
        "M",
    ]


def test_analyzer_threads_region_arg_through_nested_loops() -> None:
    """Nested range loops share the outer carried value recurrence."""
    assert _gate_labels(_nested_angle_carry) == [
        "$R_x$(0)",
        "$R_x$(1.00)",
        "$R_x$(2.00)",
        "$R_x$(3.00)",
        "M",
    ]


def test_analyzer_threads_for_items_region_arg_between_entries() -> None:
    """ForItems entries advance RegionArgs with each concrete delta."""
    assert _gate_labels(_items_angle_carry, deltas={0: 1.0, 1: 2.0}) == [
        "$R_x$(0)",
        "$R_x$(1.00)",
        "M",
    ]


def test_analyzer_selects_nested_branch_when_advancing_region_arg() -> None:
    """A concrete loop predicate selects the matching branch yield each trip."""
    assert _gate_labels(_branch_angle_carry) == [
        "$R_x$(1.00)",
        "$R_x$(3.00)",
        "M",
    ]


def test_carryless_outer_loop_keeps_inner_result_for_unfolded_body() -> None:
    """An inner carry still resolves an index inside each outer iteration."""
    graph = _prepare_graph_for_visualization(
        _carryless_outer_with_inner_carry._build_graph_for_visualization()
    )
    analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, fold_loops=False)
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
    visual = analyzer.build_visual_ir(graph, qubit_map, qubit_names, num_qubits)
    x_gates = [
        node
        for node in _walk(visual.children)
        if isinstance(node, VGate) and node.label == "$X$"
    ]

    assert [gate.qubit_indices for gate in x_gates] == [[2], [2]]


def test_carryless_outer_loop_keeps_inner_result_for_folded_body() -> None:
    """Folded text and affected wires retain an inner carry's final index."""
    graph = _prepare_graph_for_visualization(
        _carryless_outer_with_inner_carry._build_graph_for_visualization()
    )
    analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, fold_loops=True)
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
    visual = analyzer.build_visual_ir(graph, qubit_map, qubit_names, num_qubits)
    folded = next(node for node in visual.children if isinstance(node, VFoldedBlock))

    assert folded.body_lines == ["q[2] = x(q[2])"]
    assert folded.affected_qubits == [2]
