"""Regression tests for zero-qubit global-phase visualization."""

from collections.abc import Iterator
from typing import Any

import matplotlib
import matplotlib.patches as mpatches

import qamomile.circuit as qmc
from qamomile.circuit.visualization.analyzer import CircuitAnalyzer
from qamomile.circuit.visualization.drawer import (
    MatplotlibDrawer,
    _prepare_graph_for_visualization,
)
from qamomile.circuit.visualization.style import DEFAULT_STYLE
from qamomile.circuit.visualization.visual_ir import (
    VFoldedBlock,
    VGate,
    VGateKind,
    VInlineBlock,
    VisualCircuit,
    VisualNode,
    VUnfoldedSequence,
)

matplotlib.use("Agg")


@qmc.qkernel
def _identity(q: qmc.Qubit) -> qmc.Qubit:
    """Return a qubit unchanged.

    Args:
        q (qmc.Qubit): Qubit to preserve.

    Returns:
        qmc.Qubit: The unchanged qubit.
    """
    return q


@qmc.qkernel
def _symbolic_phase_circuit(theta: qmc.Float) -> qmc.Bit:
    """Apply a symbolic global phase to one qubit.

    Args:
        theta (qmc.Float): Global phase angle in radians.

    Returns:
        qmc.Bit: Measurement of the phased qubit.
    """
    q = qmc.qubit("q")
    q = qmc.global_phase(_identity, theta)(q)
    return qmc.measure(q)


@qmc.qkernel
def _phase_helper(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Apply a global phase inside an inlineable helper.

    Args:
        q (qmc.Qubit): Qubit passed through the helper.
        theta (qmc.Float): Global phase angle in radians.

    Returns:
        qmc.Qubit: Phased qubit.
    """
    return qmc.global_phase(_identity, theta)(q)


@qmc.qkernel
def _inline_scope_circuit(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Call a phased helper on only the second of two wires.

    Args:
        theta (qmc.Float): Global phase angle in radians.

    Returns:
        qmc.Vector[qmc.Bit]: Measurements of both qubits.
    """
    qs = qmc.qubit_array(2, "qs")
    qs[1] = _phase_helper(qs[1], theta)
    return qmc.measure(qs)


@qmc.qkernel
def _nested_phase_circuit(theta: qmc.Float, flag: qmc.UInt) -> qmc.Bit:
    """Place a global phase under nested symbolic for/if control flow.

    Args:
        theta (qmc.Float): Global phase angle in radians.
        flag (qmc.UInt): Predicate value selecting the phase branch.

    Returns:
        qmc.Bit: Measurement of the phased qubit.
    """
    q = qmc.qubit("q")
    for i in qmc.range(2):
        if flag == 1:
            q = qmc.global_phase(_identity, theta)(q)
    return qmc.measure(q)


def _visual_circuit(
    kernel: Any,
    *,
    inline: bool = False,
    fold_loops: bool = True,
    fold_ifs: bool = False,
    **bindings: Any,
) -> VisualCircuit:
    """Build Visual IR for a kernel with the requested expansion settings.

    Args:
        kernel (Any): QKernel to analyze.
        inline (bool): Whether to inline helper calls. Defaults to False.
        fold_loops (bool): Whether to fold loop bodies. Defaults to True.
        fold_ifs (bool): Whether to fold conditional branches. Defaults to
            False.
        **bindings (Any): Compile-time bindings passed to visualization.

    Returns:
        VisualCircuit: Visual IR generated with the requested settings.
    """
    block = _prepare_graph_for_visualization(
        kernel._build_graph_for_visualization(**bindings)
    )
    analyzer = CircuitAnalyzer(
        block,
        DEFAULT_STYLE,
        inline=inline,
        fold_loops=fold_loops,
        fold_ifs=fold_ifs,
    )
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(block)
    return analyzer.build_visual_ir(block, qubit_map, qubit_names, num_qubits)


def _walk(nodes: list[VisualNode]) -> Iterator[VisualNode]:
    """Yield every visual node recursively in display order.

    Args:
        nodes (list[VisualNode]): Visual nodes to traverse.

    Yields:
        VisualNode: Each node in recursive display order.
    """
    for node in nodes:
        yield node
        if isinstance(node, VInlineBlock):
            yield from _walk(node.children)
        elif isinstance(node, VUnfoldedSequence):
            for iteration in node.iterations:
                yield from _walk(iteration)


def _phase_nodes(circuit: VisualCircuit) -> list[VGate]:
    """Collect the global-phase annotation nodes in a visual circuit.

    Args:
        circuit (VisualCircuit): Visual circuit to search.

    Returns:
        list[VGate]: Global-phase annotation nodes in display order.
    """
    return [
        node
        for node in _walk(circuit.children)
        if isinstance(node, VGate) and node.kind == VGateKind.GLOBAL_PHASE
    ]


class TestGlobalPhaseVisualIr:
    """Verify labels, scope, and nested-control representations."""

    def test_constant_phase_value_is_visible(self):
        """A bound phase appears numerically instead of disappearing."""
        circuit = _visual_circuit(_symbolic_phase_circuit, theta=0.375)

        phases = _phase_nodes(circuit)
        assert len(phases) == 1
        assert phases[0].label == "global phase: 0.38"
        assert phases[0].qubit_indices == [0]

    def test_symbolic_phase_name_is_visible(self):
        """An unbound Greek parameter remains symbolic in the phase label."""
        circuit = _visual_circuit(_symbolic_phase_circuit)

        phases = _phase_nodes(circuit)
        assert len(phases) == 1
        assert phases[0].label == r"global phase: $\theta$"

    def test_inline_phase_uses_helper_quantum_scope(self):
        """An inlined helper's phase reserves only its participating wire."""
        circuit = _visual_circuit(_inline_scope_circuit, inline=True)

        phases = _phase_nodes(circuit)
        assert len(phases) == 1
        assert phases[0].qubit_indices == [1]

    def test_unfolded_nested_for_if_keeps_each_phase_annotation(self):
        """Materialized loop iterations retain phases in their true branches."""
        circuit = _visual_circuit(
            _nested_phase_circuit,
            fold_loops=False,
            fold_ifs=False,
        )

        phases = _phase_nodes(circuit)
        assert len(phases) == 2
        assert all("true" in phase.node_key for phase in phases)
        assert all(r"$\theta$" in phase.label for phase in phases)

    def test_folded_nested_for_if_names_phase_in_summary(self):
        """A folded loop summary exposes the phase operation and parameter."""
        circuit = _visual_circuit(
            _nested_phase_circuit,
            fold_loops=True,
            fold_ifs=True,
        )

        folded = [node for node in circuit.children if isinstance(node, VFoldedBlock)]
        assert len(folded) == 1
        summary = "\n".join(folded[0].body_lines)
        assert "global_phase" in summary
        assert r"$\theta$" in summary


class TestGlobalPhaseRendering:
    """Verify the renderer uses a floating non-gate annotation."""

    def test_badge_floats_above_an_unbroken_wire(self):
        """The phase badge stays above the wire and displays its value."""
        figure = MatplotlibDrawer.draw_kernel(_symbolic_phase_circuit, theta=0.5)
        axes = figure.axes[0]

        assert "global phase: 0.50" in [text.get_text() for text in axes.texts]
        badges = [
            patch
            for patch in axes.patches
            if isinstance(patch, mpatches.FancyBboxPatch)
            and patch.get_linestyle() == ":"
        ]
        assert len(badges) == 1
        assert badges[0].get_bbox().ymin > 0.0
