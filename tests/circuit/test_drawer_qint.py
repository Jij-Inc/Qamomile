"""Check QInt measurement wires through both analyzer and rendered drawer."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Arc

import qamomile.circuit as qmc
from qamomile.circuit.frontend.qkernel import QKernel
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.gate import GateOperation
from qamomile.circuit.serialization import deserialize, serialize
from qamomile.circuit.visualization.analyzer import CircuitAnalyzer
from qamomile.circuit.visualization.drawer import MatplotlibDrawer
from qamomile.circuit.visualization.layout import CircuitLayoutEngine
from qamomile.circuit.visualization.style import DEFAULT_STYLE
from qamomile.circuit.visualization.visual_ir import VGate, VGateKind


@qmc.qkernel
def _measure_qint_root() -> qmc.UInt:
    """Measure all six root-register wires as an unsigned integer.

    Returns:
        qmc.UInt: Unsigned integer measurement.
    """
    qubits = qmc.qubit_array(6, "q")
    return qmc.measure(qmc.cast(qubits, qmc.QInt))


@qmc.qkernel
def _measure_qint_stride() -> qmc.UInt:
    """Measure odd root-register wires as an unsigned integer.

    Returns:
        qmc.UInt: Unsigned integer measurement over wires one, three, five.
    """
    qubits = qmc.qubit_array(6, "q")
    return qmc.measure(qmc.cast(qubits[1::2], qmc.QInt))


@qmc.qkernel
def _measure_symbolic_qint_root(n: qmc.UInt) -> qmc.UInt:
    """Measure a root register whose width is supplied at build time."""
    qubits = qmc.qubit_array(n, "q")
    return qmc.measure(qmc.cast(qubits, qmc.QInt))


@qmc.qkernel
def _measure_symbolic_qint_stride(n: qmc.UInt) -> qmc.UInt:
    """Measure selected root wires of a register with a bound width."""
    qubits = qmc.qubit_array(n, "q")
    return qmc.measure(qmc.cast(qubits[1:6:2], qmc.QInt))


@qmc.qkernel
def _measure_symbolic_qint_empty(n: qmc.UInt) -> qmc.UInt:
    """Measure an empty view without consuming any root wire."""
    qubits = qmc.qubit_array(n, "q")
    return qmc.measure(qmc.cast(qubits[1:1], qmc.QInt))


@qmc.qkernel
def _pack_qint(qubits: qmc.Vector[qmc.Qubit]) -> qmc.QInt:
    """Return a packed alias of the caller's qubits."""
    return qmc.cast(qubits, qmc.QInt)


@qmc.qkernel
def _forward_qint(qubits: qmc.Vector[qmc.Qubit]) -> qmc.QInt:
    """Forward a packed alias through a second callable boundary."""
    return _pack_qint(qubits)


@qmc.qkernel
def _measure_returned_qint(n: qmc.UInt) -> qmc.UInt:
    """Measure a packed register returned by a helper."""
    qubits = qmc.qubit_array(n, "q")
    return qmc.measure(_pack_qint(qubits))


@qmc.qkernel
def _measure_merged_qint(n: qmc.UInt) -> qmc.UInt:
    """Measure a packed alias merged from two runtime branches."""
    qubits = qmc.qubit_array(n, "q")
    control = qmc.qubit("control")
    control = qmc.h(control)
    condition = qmc.measure(control)
    if condition:
        packed = qmc.cast(qubits, qmc.QInt)
    else:
        packed = qmc.cast(qubits, qmc.QInt)
    return qmc.measure(packed)


@qmc.qkernel
def _measure_distinct_qint_aliases(
    n: qmc.UInt,
) -> tuple[qmc.UInt, qmc.UInt, qmc.UInt]:
    """Keep distinct roots and views separate across repeated nested calls."""
    left = qmc.qubit_array(n, "left")
    right = qmc.qubit_array(n, "right")
    even = _forward_qint(left[0::2])
    odd = _forward_qint(left[1::2])
    other = _forward_qint(right[1::2])
    return qmc.measure(odd), qmc.measure(other), qmc.measure(even)


@pytest.mark.parametrize("restored", [False, True], ids=["original", "restored"])
@pytest.mark.parametrize("inline", [False, True], ids=["boxed", "inlined"])
@pytest.mark.parametrize(
    ("kernel", "width", "expected_measurements", "expected_num_qubits"),
    [
        (_measure_returned_qint, 0, [[]], 0),
        (_measure_returned_qint, 6, [[0, 1, 2, 3, 4, 5]], 6),
        (_measure_merged_qint, 0, [[0], []], 1),
        (_measure_merged_qint, 6, [[6], [0, 1, 2, 3, 4, 5]], 7),
        (_measure_distinct_qint_aliases, 6, [[1, 3, 5], [7, 9, 11], [0, 2, 4]], 12),
    ],
)
def test_bound_qint_alias_measurements_drawer_roundtrip(
    kernel: QKernel,
    width: int,
    expected_measurements: list[list[int]],
    expected_num_qubits: int,
    restored: bool,
    inline: bool,
) -> None:
    """Returned and merged packed aliases retain their exact measured wires."""
    if restored:
        kernel = deserialize(serialize(kernel))
    drawer = MatplotlibDrawer(kernel.build(n=width))
    analyzer = CircuitAnalyzer(drawer.graph, DEFAULT_STYLE, inline=inline)
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(drawer.graph)
    visual = analyzer.build_visual_ir(drawer.graph, qubit_map, qubit_names, num_qubits)
    measurements = [
        node.qubit_indices
        for node in visual.children
        if isinstance(node, VGate)
        and node.kind in (VGateKind.MEASURE, VGateKind.MEASURE_VECTOR)
    ]
    assert num_qubits == expected_num_qubits
    assert measurements == expected_measurements

    layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(visual)
    figure = drawer.draw(inline=inline)
    try:
        figure.canvas.draw()
        meters = [patch for patch in figure.axes[0].patches if isinstance(patch, Arc)]
        expected_wires = [index for wires in expected_measurements for index in wires]
        assert len(meters) == len(expected_wires)
        drawn_wires = [
            int(np.argmin(np.abs(np.asarray(layout.qubit_y) - meter.center[1])))
            for meter in meters
        ]
        assert sorted(drawn_wires) == sorted(expected_wires)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("restored", [False, True], ids=["original", "restored"])
@pytest.mark.parametrize(
    ("kernel", "width", "expected_wires"),
    [
        (_measure_symbolic_qint_root, 0, []),
        (_measure_symbolic_qint_root, 3, [0, 1, 2]),
        (_measure_symbolic_qint_root, 6, [0, 1, 2, 3, 4, 5]),
        (_measure_symbolic_qint_stride, 6, [1, 3, 5]),
        (_measure_symbolic_qint_stride, 8, [1, 3, 5]),
        (_measure_symbolic_qint_empty, 6, []),
    ],
)
def test_bound_qint_measurement_drawer_roundtrip(
    kernel: QKernel, width: int, expected_wires: list[int], restored: bool
) -> None:
    """Binding a serialized cast draws exactly the source's ordered root wires."""
    if restored:
        kernel = deserialize(serialize(kernel))
    drawer = MatplotlibDrawer(kernel.build(n=width))
    analyzer = CircuitAnalyzer(drawer.graph, DEFAULT_STYLE)
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(drawer.graph)
    visual = analyzer.build_visual_ir(drawer.graph, qubit_map, qubit_names, num_qubits)
    measurements = [
        node
        for node in visual.children
        if isinstance(node, VGate) and node.kind == VGateKind.MEASURE_VECTOR
    ]
    assert num_qubits == width
    assert len(measurements) == 1
    assert measurements[0].qubit_indices == expected_wires
    assert [qubit_names[index] for index in measurements[0].qubit_indices] == [
        f"q[{index}]" for index in expected_wires
    ]

    layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(visual)
    figure = drawer.draw()
    try:
        figure.canvas.draw()
        meters = [patch for patch in figure.axes[0].patches if isinstance(patch, Arc)]
        assert len(meters) == len(expected_wires)
        # The meter arc sits slightly below its wire, so identify the closest
        # wire rather than depending on the symbol's internal geometry.
        drawn_wires = [
            int(np.argmin(np.abs(np.asarray(layout.qubit_y) - meter.center[1])))
            for meter in meters
        ]
        assert sorted(drawn_wires) == sorted(expected_wires)
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ("kernel", "expected_wires"),
    [(_measure_qint_root, [0, 1, 2, 3, 4, 5]), (_measure_qint_stride, [1, 3, 5])],
)
def test_qint_measurement_analyzer_and_drawer(
    kernel: QKernel, expected_wires: list[int]
) -> None:
    """One vector-measurement node draws one meter on each selected root wire."""
    graph = kernel._build_graph_for_visualization()
    analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE)
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
    visual = analyzer.build_visual_ir(graph, qubit_map, qubit_names, num_qubits)
    measurements = [
        node
        for node in visual.children
        if isinstance(node, VGate) and node.kind == VGateKind.MEASURE_VECTOR
    ]
    assert num_qubits == 6
    assert len(measurements) == 1
    assert measurements[0].qubit_indices == expected_wires

    figure = MatplotlibDrawer.draw_kernel(kernel)
    try:
        figure.canvas.draw()
        meters = [patch for patch in figure.axes[0].patches if isinstance(patch, Arc)]
        assert len(meters) == len(expected_wires)
        # Meter arcs share their measurement box's vertical center, so the
        # stride must leave two wire spacings between consecutive meters.
        centers = sorted(float(meter.center[1]) for meter in meters)
        spacing = DEFAULT_STYLE.qubit_base_spacing
        np.testing.assert_allclose(
            np.diff(centers),
            np.diff(expected_wires) * spacing,
            atol=1e-12,
            rtol=0,
        )
    finally:
        plt.close(figure)


@qmc.qkernel
def _recursive_draw_helper(depth: qmc.UInt, qubit: qmc.Qubit) -> qmc.Qubit:
    """Retain a self-referential callable behind a summary box."""
    if depth == 0:
        qubit = qmc.h(qubit)
    else:
        qubit = _recursive_draw_helper(depth - 1, qubit)
    return qubit


@qmc.qkernel
def _recursive_draw_driver(depth: qmc.UInt) -> qmc.Bit:
    """Measure a recursive call without expanding its body for display."""
    qubit = qmc.qubit("q")
    qubit = _recursive_draw_helper(depth, qubit)
    return qmc.measure(qubit)


def test_source_index_preserves_boxed_recursive_drawing() -> None:
    """Source indexing terminates before rendering a recursive callable box."""
    drawer = MatplotlibDrawer(_recursive_draw_driver.build(depth=2))
    analyzer = CircuitAnalyzer(drawer.graph, DEFAULT_STYLE)
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(drawer.graph)
    visual = analyzer.build_visual_ir(drawer.graph, qubit_map, qubit_names, num_qubits)
    assert num_qubits == 1
    assert any(
        isinstance(node, VGate) and node.kind == VGateKind.BLOCK_BOX
        for node in visual.children
    )

    figure = drawer.draw(inline=False)
    try:
        figure.canvas.draw()
        meters = [patch for patch in figure.axes[0].patches if isinstance(patch, Arc)]
        assert len(meters) == 1
    finally:
        plt.close(figure)


@qmc.qkernel
def _shared_draw_leaf(qubit: qmc.Qubit) -> qmc.Qubit:
    """Apply one gate in a callable body shared by several invocations."""
    return qmc.x(qubit)


@qmc.qkernel
def _shared_draw_pair(qubit: qmc.Qubit) -> qmc.Qubit:
    """Reference the same leaf body twice."""
    qubit = _shared_draw_leaf(qubit)
    return _shared_draw_leaf(qubit)


@qmc.qkernel
def _shared_draw_nested(qubit: qmc.Qubit) -> qmc.Qubit:
    """Reference the same pair body twice."""
    qubit = _shared_draw_pair(qubit)
    return _shared_draw_pair(qubit)


def test_source_index_visits_shared_callable_operations_once(monkeypatch) -> None:
    """Shared call paths do not multiply visits to a body's operations."""
    graph = _shared_draw_nested.build()
    calls = [op for op in graph.operations if isinstance(op, InvokeOperation)]
    assert len(calls) == 2
    assert calls[0].body is calls[1].body
    visits: dict[int, int] = {}
    original_inputs = GateOperation.all_input_values

    def counted_inputs(operation):
        """Record how often indexing reads each shared gate operation."""
        identity = id(operation)
        visits[identity] = visits.get(identity, 0) + 1
        return original_inputs(operation)

    monkeypatch.setattr(GateOperation, "all_input_values", counted_inputs)
    CircuitAnalyzer(graph, DEFAULT_STYLE)
    assert list(visits.values()) == [1]
