"""Test visualization of the unified callable invocation model.

The callable IR uses :class:`InvokeOperation` for ordinary qkernel calls,
named composites, and bodyless oracles.  These tests exercise those callables
through their public frontend APIs so visualization cannot accidentally depend
on one obsolete callable operation shape.  They also cover recursive control
flow inside an expanded invocation and concrete substitution of a
loop-carried ``RegionArg`` while drawing an unrolled loop.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.control_flow import ForOperation
from qamomile.circuit.ir.operation.gate import GateOperationType
from qamomile.circuit.visualization.analyzer import CircuitAnalyzer
from qamomile.circuit.visualization.drawer import _prepare_graph_for_visualization
from qamomile.circuit.visualization.style import DEFAULT_STYLE
from qamomile.circuit.visualization.visual_ir import (
    VGate,
    VGateKind,
    VInlineBlock,
    VisualCircuit,
    VisualNode,
    VUnfoldedKind,
    VUnfoldedSequence,
)


@qmc.qkernel
def _inline_phase(q: qmc.Qubit) -> qmc.Qubit:
    """Apply the body of an inline-by-default helper.

    Args:
        q (qmc.Qubit): Target qubit.

    Returns:
        qmc.Qubit: Updated target qubit.
    """
    q = qmc.h(q)
    return qmc.rz(q, 0.25)


@qmc.qkernel
def _use_inline_phase() -> qmc.Qubit:
    """Invoke an ordinary qkernel from an entrypoint.

    Returns:
        qmc.Qubit: Qubit returned by the helper.
    """
    q = qmc.qubit("q")
    return _inline_phase(q)


@qmc.composite_gate(name="named_phase")
def _named_phase(q: qmc.Qubit) -> qmc.Qubit:
    """Apply the fallback body of a preserve-box composite.

    Args:
        q (qmc.Qubit): Target qubit.

    Returns:
        qmc.Qubit: Updated target qubit.
    """
    q = qmc.x(q)
    return qmc.ry(q, 0.5)


@qmc.qkernel
def _use_named_phase() -> qmc.Qubit:
    """Invoke a named composite from an entrypoint.

    Returns:
        qmc.Qubit: Qubit returned by the composite.
    """
    q = qmc.qubit("q")
    return _named_phase(q)


@qmc.composite_gate(name="nested_named_phase")
def _nested_named_phase(q: qmc.Qubit) -> qmc.Qubit:
    """Invoke one preserve-box composite from another.

    Args:
        q (qmc.Qubit): Target qubit.

    Returns:
        qmc.Qubit: Qubit returned by the nested composite.
    """
    return _named_phase(q)


@qmc.qkernel
def _use_nested_named_phase() -> qmc.Qubit:
    """Invoke the nested preserve-box composite from an entrypoint.

    Returns:
        qmc.Qubit: Qubit returned by the nested composite.
    """
    q = qmc.qubit("q")
    return _nested_named_phase(q)


@qmc.qkernel
def _use_controlled_named_phase() -> qmc.Vector[qmc.Qubit]:
    """Invoke a preserve-box composite through its controlled transform.

    Returns:
        qmc.Vector[qmc.Qubit]: Control and target qubits after invocation.
    """
    q = qmc.qubit_array(2, "q")
    q[0], q[1] = qmc.control(_named_phase)(q[0], q[1])
    return q


_OPAQUE_PHASE = qmc.opaque("opaque_phase", num_qubits=1)


@qmc.qkernel
def _use_opaque_phase() -> qmc.Qubit:
    """Invoke a bodyless oracle from an entrypoint.

    Returns:
        qmc.Qubit: Qubit returned by the oracle.
    """
    q = qmc.qubit("q")
    (q,) = _OPAQUE_PHASE(q)
    return q


@qmc.qkernel
def _dynamic_helper(q: qmc.Qubit) -> qmc.Bit:
    """Nest runtime IF and WHILE regions inside an ordinary qkernel call.

    Args:
        q (qmc.Qubit): Qubit measured to seed runtime control flow.

    Returns:
        qmc.Bit: Final loop-condition bit.
    """
    q = qmc.h(q)
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
def _use_dynamic_helper() -> qmc.Bit:
    """Invoke the helper whose body owns nested runtime regions.

    Returns:
        qmc.Bit: Final condition returned by the helper.
    """
    q = qmc.qubit("q")
    return _dynamic_helper(q)


@qmc.qkernel
def _carried_rotation() -> qmc.Qubit:
    """Drive each unrolled rotation with the current loop-carried angle.

    Returns:
        qmc.Qubit: Qubit after three parameterized rotations.
    """
    q = qmc.qubit("q")
    rx_angle = qmc.float_(0.0)
    for _iteration in qmc.range(3):
        rx_angle = rx_angle + 0.25
        q = qmc.rx(q, rx_angle)
    return q


def _visual_circuit(
    kernel: Any,
    *,
    inline: bool = False,
    expand_composite: bool = False,
    fold_loops: bool = True,
    inline_depth: int | None = None,
    **bindings: Any,
) -> VisualCircuit:
    """Build Visual IR for one public-API qkernel fixture.

    Args:
        kernel (Any): QKernel-like object to visualize.
        inline (bool): Whether inline-by-default invocations expand.
        expand_composite (bool): Whether preserve-box composites expand.
        fold_loops (bool): Whether finite loops remain folded.
        inline_depth (int | None): Maximum callable expansion depth, or
            ``None`` for no limit.
        **bindings (Any): Concrete visualization bindings.

    Returns:
        VisualCircuit: Analyzed visual tree for assertions.
    """
    graph = _prepare_graph_for_visualization(
        kernel._build_graph_for_visualization(**bindings)
    )
    analyzer = CircuitAnalyzer(
        graph,
        DEFAULT_STYLE,
        inline=inline,
        fold_loops=fold_loops,
        expand_composite=expand_composite,
        inline_depth=inline_depth,
        fold_ifs=False,
        fold_whiles=False,
    )
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
    return analyzer.build_visual_ir(graph, qubit_map, qubit_names, num_qubits)


def _walk(nodes: Iterable[VisualNode]) -> Iterator[VisualNode]:
    """Yield nodes recursively through inline bodies and unfolded regions.

    Args:
        nodes (Iterable[VisualNode]): Root nodes to traverse.

    Yields:
        VisualNode: Each reachable visual node in preorder.
    """
    for node in nodes:
        yield node
        if isinstance(node, VInlineBlock):
            yield from _walk(node.children)
        elif isinstance(node, VUnfoldedSequence):
            for region in node.iterations:
                yield from _walk(region)


def _assert_draws(kernel: Any, **options: Any) -> None:
    """Assert the public draw path renders and close the resulting figure.

    Args:
        kernel (Any): QKernel-like object whose public ``draw`` method to call.
        **options (Any): Visualization options forwarded to ``draw``.
    """
    figure = kernel.draw(**options)
    assert isinstance(figure, Figure)
    plt.close(figure)


def test_inline_qkernel_invoke_switches_between_box_and_body() -> None:
    """An ordinary qkernel Invoke is boxed by default and expands on demand."""
    boxed = _visual_circuit(_use_inline_phase)
    boxed_calls = [
        node
        for node in boxed.children
        if isinstance(node, VGate) and node.kind is VGateKind.BLOCK_BOX
    ]
    assert [node.label for node in boxed_calls] == ["_INLINE_PHASE"]

    expanded = _visual_circuit(_use_inline_phase, inline=True)
    [inline_call] = [
        node for node in expanded.children if isinstance(node, VInlineBlock)
    ]
    assert inline_call.label == "_inline_phase"
    assert [
        node.gate_type
        for node in _walk(inline_call.children)
        if isinstance(node, VGate)
    ] == [GateOperationType.H, GateOperationType.RZ]
    _assert_draws(_use_inline_phase)
    _assert_draws(_use_inline_phase, inline=True)


def test_preserve_box_composite_expands_only_with_expand_composite() -> None:
    """A named composite keeps its box until expand_composite is requested."""
    boxed = _visual_circuit(_use_named_phase, inline=True)
    composite_boxes = [
        node
        for node in boxed.children
        if isinstance(node, VGate) and node.kind is VGateKind.COMPOSITE_BOX
    ]
    assert [node.label for node in composite_boxes] == ["NAMED_PHASE"]

    expanded = _visual_circuit(_use_named_phase, expand_composite=True)
    [inline_call] = [
        node for node in expanded.children if isinstance(node, VInlineBlock)
    ]
    assert inline_call.label == "named_phase"
    assert [
        node.gate_type
        for node in _walk(inline_call.children)
        if isinstance(node, VGate)
    ] == [GateOperationType.X, GateOperationType.RY]
    _assert_draws(_use_named_phase, inline=True)
    _assert_draws(_use_named_phase, expand_composite=True)


def test_composite_expansion_honors_inline_depth() -> None:
    """Nested preserve-box calls stop expanding at the requested depth."""
    limited = _visual_circuit(
        _use_nested_named_phase,
        expand_composite=True,
        inline_depth=1,
    )
    [outer] = [node for node in limited.children if isinstance(node, VInlineBlock)]
    assert [
        node.label
        for node in outer.children
        if isinstance(node, VGate) and node.kind is VGateKind.COMPOSITE_BOX
    ] == ["NAMED_PHASE"]

    unlimited = _visual_circuit(
        _use_nested_named_phase,
        expand_composite=True,
    )
    assert not any(
        isinstance(node, VGate) and node.kind is VGateKind.COMPOSITE_BOX
        for node in _walk(unlimited.children)
    )
    assert [
        node.gate_type for node in _walk(unlimited.children) if isinstance(node, VGate)
    ] == [GateOperationType.X, GateOperationType.RY]
    _assert_draws(
        _use_nested_named_phase,
        expand_composite=True,
        inline_depth=1,
    )


def test_controlled_composite_keeps_control_outside_fallback_body() -> None:
    """A controlled Invoke expands its direct fallback only on target wires."""
    boxed = _visual_circuit(_use_controlled_named_phase)
    [controlled_box] = [
        node
        for node in boxed.children
        if isinstance(node, VGate) and node.kind is VGateKind.CONTROLLED_U_BOX
    ]
    assert controlled_box.qubit_indices == [0, 1]
    assert controlled_box.control_count == 1

    expanded = _visual_circuit(
        _use_controlled_named_phase,
        expand_composite=True,
    )
    [inline_call] = [
        node for node in expanded.children if isinstance(node, VInlineBlock)
    ]
    assert inline_call.control_qubit_indices == [0]
    assert [
        (node.gate_type, node.qubit_indices)
        for node in _walk(inline_call.children)
        if isinstance(node, VGate)
    ] == [
        (GateOperationType.X, [1]),
        (GateOperationType.RY, [1]),
    ]
    _assert_draws(_use_controlled_named_phase, expand_composite=True)


def test_bodyless_oracle_remains_boxed_under_all_expansion_options() -> None:
    """A bodyless oracle remains a box even when every expansion flag is set."""
    circuit = _visual_circuit(
        _use_opaque_phase,
        inline=True,
        expand_composite=True,
        fold_loops=False,
    )
    assert not any(isinstance(node, VInlineBlock) for node in circuit.children)
    oracle_boxes = [
        node
        for node in circuit.children
        if isinstance(node, VGate) and node.kind is VGateKind.COMPOSITE_BOX
    ]
    assert [node.label for node in oracle_boxes] == ["OPAQUE_PHASE"]
    _assert_draws(_use_opaque_phase, inline=True, expand_composite=True)


def test_expanded_invoke_recursively_draws_if_and_while_regions() -> None:
    """An expanded Invoke preserves nested runtime IF and WHILE visual nodes."""
    circuit = _visual_circuit(_use_dynamic_helper, inline=True, fold_loops=False)
    [inline_call] = [
        node for node in circuit.children if isinstance(node, VInlineBlock)
    ]
    region_kinds = {
        node.kind
        for node in _walk(inline_call.children)
        if isinstance(node, VUnfoldedSequence)
    }
    assert region_kinds == {VUnfoldedKind.IF, VUnfoldedKind.WHILE}
    _assert_draws(
        _use_dynamic_helper,
        inline=True,
        fold_loops=False,
        fold_ifs=False,
        fold_whiles=False,
    )


def test_unfolded_loop_substitutes_region_arg_for_each_rotation() -> None:
    """Each unrolled iteration uses its current loop-carried scalar value."""
    loops = [
        op for op in _carried_rotation.block.operations if isinstance(op, ForOperation)
    ]
    assert len(loops) == 1
    assert loops[0].region_args, "fixture must exercise RegionArg substitution"

    circuit = _visual_circuit(_carried_rotation, fold_loops=False)
    [loop] = [
        node
        for node in circuit.children
        if isinstance(node, VUnfoldedSequence) and node.kind is VUnfoldedKind.FOR
    ]
    labels = [
        node.label
        for iteration in loop.iterations
        for node in _walk(iteration)
        if isinstance(node, VGate) and node.gate_type is GateOperationType.RX
    ]
    assert labels == ["$R_x$(0.25)", "$R_x$(0.50)", "$R_x$(0.75)"]
    _assert_draws(_carried_rotation, fold_loops=False)
