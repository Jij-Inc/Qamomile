"""Regression tests for correctness-first qkernel drawing."""

from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")

import pytest
from matplotlib.figure import Figure

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.control_flow import ForItemsOperation, ForOperation
from qamomile.circuit.ir.operation.gate import GateOperationType
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.types.primitives import FloatType, QubitType, UIntType
from qamomile.circuit.ir.value import DictValue, Value
from qamomile.circuit.transpiler.circuit_ir import ForInstruction
from qamomile.circuit.transpiler.passes.analyze import AnalyzePass
from qamomile.circuit.transpiler.passes.inline import InlinePass
from qamomile.circuit.visualization import CircuitDrawingError, MatplotlibDrawer
from qamomile.circuit.visualization.circuit_adapter import (
    circuit_program_to_visual_ir,
)
from qamomile.circuit.visualization.drawing_compiler import (
    compile_block_for_drawing,
    compile_qkernel_for_drawing,
)
from qamomile.circuit.visualization.layout import CircuitLayoutEngine
from qamomile.circuit.visualization.style import DEFAULT_STYLE
from qamomile.circuit.visualization.visual_ir import VGate, VGateKind


@qmc.qkernel
def _scalar_quantum_input(q: qmc.Qubit) -> qmc.Qubit:
    """Apply a gate to a scalar external quantum input.

    Args:
        q (qmc.Qubit): External qubit to update.

    Returns:
        qmc.Qubit: Updated external qubit.
    """
    return qmc.h(q)


@qmc.qkernel
def _identity_quantum_input(q: qmc.Qubit) -> qmc.Qubit:
    """Return an external qubit without inventing an initialization.

    Args:
        q (qmc.Qubit): External qubit to preserve.

    Returns:
        qmc.Qubit: The unchanged external qubit.
    """
    return q


@qmc.qkernel
def _vector_quantum_input(
    q: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Apply a gate to every external quantum-vector element.

    Args:
        q (qmc.Vector[qmc.Qubit]): External qubit register to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated external qubit register.
    """
    return qmc.h(q)


@qmc.qkernel
def _symbolic_quantum_index(
    q: qmc.Vector[qmc.Qubit],
    index: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Apply X at a draw-time-selected quantum index.

    Args:
        q (qmc.Vector[qmc.Qubit]): External qubit register to update.
        index (qmc.UInt): Index selecting the target qubit.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated external qubit register.
    """
    q[index] = qmc.x(q[index])
    return q


@qmc.qkernel
def _symbolic_quantum_slice(
    q: qmc.Vector[qmc.Qubit],
    stop: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Apply H to a draw-time-selected prefix.

    Args:
        q (qmc.Vector[qmc.Qubit]): External qubit register to update.
        stop (qmc.UInt): Exclusive prefix bound.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated external qubit register.
    """
    prefix = q[0:stop]
    prefix = qmc.h(prefix)
    q[0:stop] = prefix
    return q


@qmc.qkernel
def _controlled_quantum_slice(
    q: qmc.Vector[qmc.Qubit],
    system_size: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Apply a controlled modular increment to an external slice.

    Args:
        q (qmc.Vector[qmc.Qubit]): System register followed by one control.
        system_size (qmc.UInt): Width of the controlled target slice.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated external register.
    """
    controlled_increment = qmc.control(qmc.modular_increment)
    q[system_size], q[0:system_size] = controlled_increment(
        q[system_size],
        q[0:system_size],
    )
    return q


def _zero_trip_qinit_graph(kind: str) -> Block:
    """Build an exact zero-trip loop with one unreachable local allocation.

    Args:
        kind (str): ``"for"`` or ``"for-items"``.

    Returns:
        Block: Quantum-shaped graph whose body must allocate no physical wire.

    Raises:
        ValueError: If ``kind`` is unsupported.
    """
    local = Value(type=QubitType(), name="local")
    if kind == "for":
        operation = ForOperation(
            operands=[
                Value(type=UIntType(), name="start").with_const(0),
                Value(type=UIntType(), name="stop").with_const(0),
                Value(type=UIntType(), name="step").with_const(1),
            ],
            loop_var="i",
            loop_var_value=Value(type=UIntType(), name="i"),
            operations=[QInitOperation(results=[local])],
        )
    elif kind == "for-items":
        operation = ForItemsOperation(
            operands=[
                DictValue(name="items", entries=()).with_dict_runtime_metadata({})
            ],
            key_vars=["key"],
            value_var="item",
            key_var_values=(Value(type=UIntType(), name="key"),),
            value_var_value=Value(type=FloatType(), name="item"),
            operations=[QInitOperation(results=[local])],
        )
    else:
        raise ValueError(f"Unsupported zero-trip loop kind: {kind}")
    return Block(name=f"zero_trip_{kind}", operations=[operation])


@qmc.qkernel
def _multiple_quantum_segments(
    observable: qmc.Observable,
) -> tuple[qmc.Float, qmc.Vector[qmc.Bit]]:
    """Build a program whose expval separates two quantum segments.

    Args:
        observable (qmc.Observable): Observable used by the expectation value.

    Returns:
        tuple[qmc.Float, qmc.Vector[qmc.Bit]]: Expectation value and a later
            independent measurement.
    """
    first = qmc.qubit_array(1, "first")
    first[0] = qmc.h(first[0])
    expectation = qmc.expval(first, observable)
    second = qmc.qubit_array(1, "second")
    second[0] = qmc.x(second[0])
    return expectation, qmc.measure(second)


@qmc.qkernel
def _single_expval(observable: qmc.Observable) -> qmc.Float:
    """Build one terminal expectation value over a two-qubit state.

    Args:
        observable (qmc.Observable): Observable displayed symbolically as H.

    Returns:
        qmc.Float: Expectation value of the prepared state.
    """
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.h(q[0])
    return qmc.expval(q, observable)


@qmc.qkernel
def _sliced_expval(observable: qmc.Observable) -> qmc.Float:
    """Build a terminal expectation value over a strided register view.

    Args:
        observable (qmc.Observable): Symbolic observable to display.

    Returns:
        qmc.Float: Expectation value over physical wires one and three.
    """
    q = qmc.qubit_array(4, "q")
    q = qmc.h(q)
    return qmc.expval(q[1::2], observable)


@qmc.qkernel
def _stage_ready_draw() -> qmc.Bit:
    """Build a simple program usable at later compiler stages.

    Returns:
        qmc.Bit: Measurement of one prepared qubit.
    """
    q = qmc.qubit("q")
    q = qmc.h(q)
    return qmc.measure(q)


@qmc.qkernel
def _classical_only_draw(theta: qmc.Float) -> qmc.Float:
    """Build a valid qkernel with no quantum interface or operations.

    Args:
        theta (qmc.Float): Classical input value.

    Returns:
        qmc.Float: Shifted classical result.
    """
    return theta + 1.0


@qmc.qkernel
def _classical_helper(value: qmc.Float) -> qmc.Float:
    """Apply one reusable purely classical computation.

    Args:
        value (qmc.Float): Input value.

    Returns:
        qmc.Float: Shifted value.
    """
    return value + 1.0


@qmc.qkernel
def _classical_helper_draw(theta: qmc.Float) -> qmc.Float:
    """Invoke a classical-only helper from a classical-only entrypoint.

    Args:
        theta (qmc.Float): Input value.

    Returns:
        qmc.Float: Helper result scaled by two.
    """
    return _classical_helper(theta) * 2.0


@qmc.qkernel
def _dynamic_named_helper(q: qmc.Qubit) -> qmc.Bit:
    """Allocate named branch and loop-local qubits inside a helper.

    Args:
        q (qmc.Qubit): Qubit measured as the runtime condition.

    Returns:
        qmc.Bit: Final loop condition.
    """
    bit = qmc.measure(q)
    if bit:
        branch_q = qmc.qubit("branch_q")
        branch_q = qmc.x(branch_q)
        qmc.measure(branch_q)
    while bit:
        loop_q = qmc.qubit("loop_q")
        loop_q = qmc.x(loop_q)
        bit = qmc.measure(loop_q)
    return bit


@qmc.qkernel
def _use_dynamic_named_helper() -> qmc.Bit:
    """Invoke a helper whose inlined body owns named quantum allocations.

    Returns:
        qmc.Bit: Helper result.
    """
    q = qmc.qubit("q")
    return _dynamic_named_helper(q)


@qmc.qkernel
def _large_static_loop() -> qmc.Bit:
    """Build a large backend-native loop over one fixed qubit.

    Returns:
        qmc.Bit: Measurement after the loop.
    """
    q = qmc.qubit("q")
    for _ in qmc.range(10**9):
        q = qmc.h(q)
    return qmc.measure(q)


def _q_wire_labels(figure: Figure) -> list[str]:
    """Return rendered labels belonging to the external ``q`` input.

    Args:
        figure (Figure): Rendered circuit figure.

    Returns:
        list[str]: Labels whose text begins with ``q``.
    """
    axis = figure._qm_ax  # type: ignore[attr-defined]
    return [
        text.get_text()
        for text in axis.texts
        if re.fullmatch(r"q(?:\[\d+\]|\d+)?", text.get_text())
    ]


def test_draw_accepts_scalar_quantum_input() -> None:
    """A scalar quantum input remains a single named display wire."""
    figure = _scalar_quantum_input.draw()

    assert isinstance(figure, Figure)
    assert _q_wire_labels(figure) == ["q"]


def test_draw_accepts_identity_quantum_input() -> None:
    """An empty quantum fragment preserves its external wire interface."""
    figure = _identity_quantum_input.draw()

    assert isinstance(figure, Figure)
    assert _q_wire_labels(figure) == ["q"]


def test_draw_accepts_classical_only_qkernel() -> None:
    """A classical-only qkernel retains the established empty drawing."""
    figure = _classical_only_draw.draw(theta=1.0)
    drawing = compile_qkernel_for_drawing(_classical_only_draw, {"theta": 1.0})

    assert isinstance(figure, Figure)
    assert drawing.circuit.num_qubits == 0
    assert drawing.qubit_names == {}
    axis = figure._qm_ax  # type: ignore[attr-defined]
    assert "Empty circuit" in {text.get_text() for text in axis.texts}


def test_draw_accepts_classical_only_helper_call() -> None:
    """A classical helper invocation does not invent a quantum segment."""
    figure = _classical_helper_draw.draw(theta=1.0)
    drawing = compile_qkernel_for_drawing(
        _classical_helper_draw,
        {"theta": 1.0},
    )

    assert isinstance(figure, Figure)
    assert drawing.circuit.num_qubits == 0
    assert drawing.circuit.operations == ()
    axis = figure._qm_ax  # type: ignore[attr-defined]
    assert "Empty circuit" in {text.get_text() for text in axis.texts}


def test_inlined_helper_allocations_keep_source_wire_names() -> None:
    """Post-plan allocation UUIDs retain helper-local display names."""
    drawing = compile_qkernel_for_drawing(_use_dynamic_named_helper)

    assert drawing.circuit.num_qubits == 3
    assert drawing.qubit_names == {0: "q", 1: "branch_q", 2: "loop_q"}


@pytest.mark.parametrize("kind", ["for", "for-items"])
def test_zero_trip_loop_body_adds_no_phantom_wire(kind: str) -> None:
    """Exact allocation skips unreachable body-local qubits.

    Args:
        kind (str): Zero-trip loop operation variant.
    """
    drawing = compile_block_for_drawing(_zero_trip_qinit_graph(kind))

    assert drawing.circuit.num_qubits == 0
    assert drawing.circuit.operations == ()
    assert drawing.qubit_names == {}


def test_draw_preserves_large_backend_native_loop() -> None:
    """Resource discovery does not materialize a backend-native loop."""
    drawing = compile_qkernel_for_drawing(_large_static_loop)
    figure = _large_static_loop.draw()

    assert drawing.circuit.num_qubits == 1
    assert any(
        isinstance(operation, ForInstruction)
        for operation in drawing.circuit.operations
    )
    assert isinstance(figure, Figure)


def test_draw_accepts_vector_quantum_input() -> None:
    """A sized quantum-vector input keeps exactly its external wires."""
    figure = _vector_quantum_input.draw(q=3)

    assert isinstance(figure, Figure)
    assert _q_wire_labels(figure) == ["q[0]", "q[1]", "q[2]"]


def test_draw_rejects_unresolved_quantum_index() -> None:
    """Drawing never guesses which wire an unresolved index denotes."""
    with pytest.raises(CircuitDrawingError, match="index"):
        _symbolic_quantum_index.draw(q=3)


def test_draw_accepts_bound_quantum_index() -> None:
    """A concrete quantum index can be lowered to an exact target wire."""
    figure = _symbolic_quantum_index.draw(q=3, index=2)

    assert isinstance(figure, Figure)
    assert _q_wire_labels(figure) == ["q[0]", "q[1]", "q[2]"]
    drawing = compile_qkernel_for_drawing(
        _symbolic_quantum_index,
        {"q": 3, "index": 2},
    )
    gates = [
        node
        for node in circuit_program_to_visual_ir(drawing.circuit).children
        if isinstance(node, VGate) and node.gate_type is GateOperationType.X
    ]
    assert drawing.circuit.num_qubits == 3
    assert [gate.qubit_indices for gate in gates] == [[2]]


def test_block_draw_rejects_unresolved_quantum_index() -> None:
    """The public raw-Block drawer never guesses a symbolic target wire."""
    graph = _symbolic_quantum_index._build_graph_for_visualization(q=3)

    with pytest.raises(CircuitDrawingError, match="index"):
        MatplotlibDrawer(graph).draw()


def test_block_draw_accepts_resolved_quantum_index() -> None:
    """A raw Block with a resolved index uses the exact circuit pipeline."""
    graph = _symbolic_quantum_index._build_graph_for_visualization(q=3, index=2)

    figure = MatplotlibDrawer(graph).draw()

    assert isinstance(figure, Figure)
    assert _q_wire_labels(figure) == ["q[0]", "q[1]", "q[2]"]
    drawing = compile_block_for_drawing(graph)
    gates = [
        node
        for node in circuit_program_to_visual_ir(drawing.circuit).children
        if isinstance(node, VGate) and node.gate_type is GateOperationType.X
    ]
    assert drawing.circuit.num_qubits == 3
    assert [gate.qubit_indices for gate in gates] == [[2]]


def test_draw_rejects_unresolved_quantum_slice() -> None:
    """Drawing never widens an unresolved slice to candidate wires."""
    with pytest.raises(CircuitDrawingError, match="slice"):
        _symbolic_quantum_slice.draw(q=4)


def test_draw_accepts_bound_quantum_slice() -> None:
    """A concrete quantum slice lowers to its exact physical wires."""
    figure = _symbolic_quantum_slice.draw(q=4, stop=2)

    assert isinstance(figure, Figure)
    assert _q_wire_labels(figure) == ["q[0]", "q[1]", "q[2]", "q[3]"]
    drawing = compile_qkernel_for_drawing(
        _symbolic_quantum_slice,
        {"q": 4, "stop": 2},
    )
    gates = [
        node
        for node in circuit_program_to_visual_ir(drawing.circuit).children
        if isinstance(node, VGate) and node.gate_type is GateOperationType.H
    ]
    assert drawing.circuit.num_qubits == 4
    assert [gate.qubit_indices for gate in gates] == [[0], [1]]


def test_draw_controlled_slice_has_no_phantom_wires() -> None:
    """A controlled slice call keeps exact wires and its semantic name."""
    figure = _controlled_quantum_slice.draw(
        q=5,
        system_size=4,
        inline=True,
    )

    assert isinstance(figure, Figure)
    assert _q_wire_labels(figure) == [
        "q[0]",
        "q[1]",
        "q[2]",
        "q[3]",
        "q[4]",
    ]
    axis = figure._qm_ax  # type: ignore[attr-defined]
    assert "modular_increment" in {text.get_text() for text in axis.texts}
    drawing = compile_qkernel_for_drawing(
        _controlled_quantum_slice,
        {"q": 5, "system_size": 4},
    )
    [call] = circuit_program_to_visual_ir(drawing.circuit).children
    assert drawing.circuit.num_qubits == 5
    assert len(drawing.qubit_names) == 5
    assert isinstance(call, VGate)
    assert call.kind is VGateKind.CONTROLLED_U_BOX
    assert call.control_count == 1
    assert call.qubit_indices == [4, 0, 1, 2, 3]


def test_draw_wraps_multiple_quantum_segment_error() -> None:
    """Planning failures are exposed through the public drawing error."""
    with pytest.raises(CircuitDrawingError, match="2 quantum segments"):
        _multiple_quantum_segments.draw(observable=qm_o.Z(0))


@pytest.mark.parametrize("bindings", [{}, {"observable": qm_o.Z(0)}])
def test_draw_preserves_terminal_expval(
    bindings: dict[str, object],
) -> None:
    """Bound and symbolic observables retain the terminal H box.

    Args:
        bindings (dict[str, object]): Optional concrete observable binding.
    """
    figure = _single_expval.draw(**bindings)
    drawing = compile_qkernel_for_drawing(_single_expval, bindings)
    visual = circuit_program_to_visual_ir(
        drawing.circuit,
        expectation_value_qubits=drawing.expectation_value_qubits,
    )

    assert isinstance(figure, Figure)
    axis = figure._qm_ax  # type: ignore[attr-defined]
    assert "<H>" in {text.get_text() for text in axis.texts}
    assert _q_wire_labels(figure) == ["q[0]", "q[1]"]
    assert drawing.expectation_value_qubits == ((0, 1),)
    expval_nodes = [
        node
        for node in visual.children
        if isinstance(node, VGate) and node.kind is VGateKind.EXPVAL
    ]
    assert [node.qubit_indices for node in expval_nodes] == [[0, 1]]


def test_draw_expval_uses_exact_sliced_slots() -> None:
    """A sliced expectation value keeps root-register physical slots."""
    drawing = compile_qkernel_for_drawing(_sliced_expval)
    visual = circuit_program_to_visual_ir(
        drawing.circuit,
        expectation_value_qubits=drawing.expectation_value_qubits,
    )
    layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(visual)
    [expval] = [
        node
        for node in visual.children
        if isinstance(node, VGate) and node.kind is VGateKind.EXPVAL
    ]
    placement = layout.expval_layouts[expval.node_key]
    skipped_y = layout.qubit_y[2]
    figure = _sliced_expval.draw()

    assert drawing.circuit.num_qubits == 4
    assert drawing.expectation_value_qubits == ((1, 3),)
    assert len(placement.boxes) == 2
    assert all(not (box.bottom < skipped_y < box.top) for box in placement.boxes)
    assert len(placement.connector_segments) == 1
    assert isinstance(figure, Figure)


@pytest.mark.parametrize("stage", ["affine", "analyzed"])
def test_block_draw_accepts_later_compiler_stages(stage: str) -> None:
    """Exact raw-Block drawing resumes from AFFINE and ANALYZED stages.

    Args:
        stage (str): Compiler stage to provide to ``MatplotlibDrawer``.
    """
    affine = InlinePass().run(_stage_ready_draw.block)
    block = affine if stage == "affine" else AnalyzePass().run(affine)

    figure = MatplotlibDrawer(block).draw()

    assert isinstance(figure, Figure)
    axis = figure._qm_ax  # type: ignore[attr-defined]
    assert "$H$" in {text.get_text() for text in axis.texts}


def test_block_draw_normalizes_analyzed_slice_markers() -> None:
    """An analyzed slice block resumes with exact marker normalization."""
    graph = _symbolic_quantum_slice._build_graph_for_visualization(q=4, stop=2)
    affine = InlinePass().run(graph)
    analyzed = AnalyzePass(validate_classical_io=False).run(affine)

    figure = MatplotlibDrawer(analyzed).draw()
    drawing = compile_block_for_drawing(analyzed)
    gates = [
        node
        for node in circuit_program_to_visual_ir(drawing.circuit).children
        if isinstance(node, VGate) and node.gate_type is GateOperationType.H
    ]

    assert isinstance(figure, Figure)
    assert drawing.circuit.num_qubits == 4
    assert [gate.qubit_indices for gate in gates] == [[0], [1]]
