"""Regression tests for drawing kernels that use Vector slicing.

The visualization analyzer must walk the ``ArrayValue.slice_of`` chain
when an operand's ``parent_array`` is itself a slice view (rather than
the root register), so that view-local element indices compose with the
chain's affine map and resolve to root-space qubit wires.  Without this
walk the analyzer drops every gate inside a broadcast like
``qmc.h(qs[0::2])`` — the parent's logical id is the view's, which the
qubit map only ever records for the root — and the rendered figure
shows only the measurement at the end of the kernel.

Each test below pins down a different slice topology (literal-bound
slice + slice-assignment broadcast, the canonical top-level
``even = q[0::2]; for i in ...; q[0::2] = even`` pattern, and a nested
``q[0::2][0:2]`` two-hop slice) and asserts that the analyzer reports
the correct affected-qubit set for the emitted broadcast loop.  When
``fold_loops=False`` is requested, the inner H gates also unfold onto
the exact root-space wires.
"""

import matplotlib

matplotlib.use("Agg")

import qamomile.circuit as qmc
from qamomile.circuit.visualization.analyzer import CircuitAnalyzer
from qamomile.circuit.visualization.drawer import MatplotlibDrawer
from qamomile.circuit.visualization.style import DEFAULT_STYLE
from qamomile.circuit.visualization.visual_ir import (
    VFoldedBlock,
    VFoldedKind,
    VGate,
    VGateKind,
    VUnfoldedKind,
    VUnfoldedSequence,
)


@qmc.qkernel
def _explicit_slice_release() -> qmc.Vector[qmc.Bit]:
    """``view = q[0::2]; view = h(view); q[0::2] = view`` over 4 qubits."""
    q = qmc.qubit_array(4, "q")
    view = q[0::2]
    view = qmc.h(view)
    q[0::2] = view
    return qmc.measure(q)


@qmc.qkernel
def _inline_slice_assignment() -> qmc.Vector[qmc.Bit]:
    """``q[0:2] = h(q[0:2])`` contiguous literal slice over 4 qubits."""
    q = qmc.qubit_array(4, "q")
    q[0:2] = qmc.h(q[0:2])
    return qmc.measure(q)


@qmc.qkernel
def _top_level_loop_pattern() -> qmc.Vector[qmc.Bit]:
    """Canonical broadcast-loop: top view, body element access, top release."""
    q = qmc.qubit_array(4, "q")
    even = q[0::2]
    for i in qmc.range(even.shape[0]):
        even[i] = qmc.h(even[i])
    q[0::2] = even
    return qmc.measure(q)


@qmc.qkernel
def _nested_slice() -> qmc.Vector[qmc.Bit]:
    """``even = q[0::2]; even[0:2] = h(even[0:2])`` over 8 qubits."""
    q = qmc.qubit_array(8, "q")
    even = q[0::2]
    even[0:2] = qmc.h(even[0:2])
    return qmc.measure(q)


def _build_visual_circuit(kernel, **kwargs):
    """Build a VisualCircuit straight from the analyzer for assertions."""
    graph = kernel._build_graph_for_visualization()
    analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, **kwargs)
    qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
    return analyzer.build_visual_ir(graph, qubit_map, qubit_names, num_qubits)


class TestSliceDrawAnalyzer:
    """Analyzer-level invariants for slice-bearing kernels.

    Before the slice-aware ``_resolve_view_chain_to_root`` walk, the
    analyzer dropped every gate whose operand had a sliced parent
    (because the view's logical id was not in qubit_map and the
    resolver returned None up the stack).  These tests assert the
    broadcast loop's affected-qubit set lands on the right root-space
    wires for each slice topology.
    """

    def test_explicit_release_pattern_affects_strided_wires(self):
        """``view = q[0::2]; view = h(view); q[0::2] = view`` touches root {0, 2}."""
        vc = _build_visual_circuit(_explicit_slice_release)
        folded = [
            c
            for c in vc.children
            if isinstance(c, VFoldedBlock) and c.kind == VFoldedKind.FOR
        ]
        assert len(folded) == 1, vc.children
        assert sorted(folded[0].affected_qubits) == [0, 2]
        assert folded[0].affected_qubits_precise is True

    def test_inline_contiguous_assignment_affects_contiguous_wires(self):
        """``q[0:2] = h(q[0:2])`` touches root {0, 1}."""
        vc = _build_visual_circuit(_inline_slice_assignment)
        folded = [
            c
            for c in vc.children
            if isinstance(c, VFoldedBlock) and c.kind == VFoldedKind.FOR
        ]
        assert len(folded) == 1, vc.children
        assert sorted(folded[0].affected_qubits) == [0, 1]

    def test_top_level_loop_pattern_affects_strided_wires(self):
        """top-level ``even = q[0::2]; for i in ...:`` touches root {0, 2}."""
        vc = _build_visual_circuit(_top_level_loop_pattern)
        folded = [
            c
            for c in vc.children
            if isinstance(c, VFoldedBlock) and c.kind == VFoldedKind.FOR
        ]
        assert len(folded) == 1
        assert sorted(folded[0].affected_qubits) == [0, 2]

    def test_nested_slice_composes_affine_chain(self):
        """``even = q[0::2]; even[0:2] = h(even[0:2])`` touches root {0, 2}.

        ``even`` is ``q[0::2]`` (start=0, step=2) and ``even[0:2]`` is
        the inner two-element prefix of that view.  Composing the
        chain back to ``q``, the inner indices 0 and 1 map to root
        positions ``0 + 2*0 = 0`` and ``0 + 2*1 = 2``.  Verifying this
        result pins down that the analyzer walks the full
        ``slice_of`` chain rather than only the immediate parent.
        """
        vc = _build_visual_circuit(_nested_slice)
        folded = [
            c
            for c in vc.children
            if isinstance(c, VFoldedBlock) and c.kind == VFoldedKind.FOR
        ]
        assert len(folded) == 1
        assert sorted(folded[0].affected_qubits) == [0, 2]


class TestSliceDrawUnfolded:
    """When the broadcast loop is unfolded the inner H gates must land
    on the exact root-space wires the slice covers.

    With ``fold_loops=False`` the analyzer evaluates the loop range
    iteration by iteration and emits per-iteration children.  Each
    iteration's H gate is then a ``VGate`` whose ``qubit_indices``
    list is the resolved root-space index for that view-local i.
    """

    def test_explicit_release_unfolded_emits_h_on_strided_wires(self):
        """Unfolded ``view=h(view)`` emits H on q[0] and q[2]."""
        vc = _build_visual_circuit(_explicit_slice_release, fold_loops=False)
        h_qubits: list[int] = []
        for top in vc.children:
            if isinstance(top, VUnfoldedSequence) and top.kind == VUnfoldedKind.FOR:
                for iteration in top.iterations:
                    for node in iteration:
                        if isinstance(node, VGate) and node.kind == VGateKind.GATE:
                            h_qubits.extend(node.qubit_indices)
        assert sorted(h_qubits) == [0, 2], h_qubits

    def test_nested_slice_unfolded_emits_h_on_composed_wires(self):
        """Unfolded ``even[0:2] = h(even[0:2])`` with ``even = q[0::2]``
        emits H on q[0] and q[2]."""
        vc = _build_visual_circuit(_nested_slice, fold_loops=False)
        h_qubits: list[int] = []
        for top in vc.children:
            if isinstance(top, VUnfoldedSequence) and top.kind == VUnfoldedKind.FOR:
                for iteration in top.iterations:
                    for node in iteration:
                        if isinstance(node, VGate) and node.kind == VGateKind.GATE:
                            h_qubits.extend(node.qubit_indices)
        assert sorted(h_qubits) == [0, 2], h_qubits


class TestSliceCastQFixedDraw:
    """``cast(view, QFixed) -> measure`` must render the measurement on
    the carrier qubits the cast spans.

    Two analyzer bugs combined to drop the measurement entirely from
    the rendered figure: ``MeasureQFixedOperation`` was not in the
    visual-IR dispatch list (so the op silently produced no node), and
    even if it had been, the cast's ``CastMetadata.qubit_logical_ids``
    is formatted as ``f"{root_lid}_{idx}"`` while ``QInitOperation``
    registers root elements in ``qubit_map`` as
    ``f"{root_lid}_[{idx}]"``.  ``_resolve_qfixed_carrier_indices``
    bridges the two encodings; the regressions below pin down both
    fixes together — a measure must appear, and it must land on the
    physical wires the slice actually covers.
    """

    def test_cast_view_qfixed_measure_targets_slice_wires(self):
        """``cast(q[1::2], QFixed); measure`` renders M on q[1] and q[3]."""

        @qmc.qkernel
        def kern() -> qmc.Float:
            q = qmc.qubit_array(4, "q")
            qf = qmc.cast(q[1::2], qmc.QFixed, int_bits=0)
            return qmc.measure(qf)

        vc = _build_visual_circuit(kern)
        m_indices: list[int] = []
        for c in vc.children:
            if isinstance(c, VGate) and c.kind == VGateKind.MEASURE_VECTOR:
                m_indices.extend(c.qubit_indices)
        assert sorted(m_indices) == [1, 3], m_indices

    def test_cast_view_qfixed_measure_renders_a_gate(self):
        """The QFixed measure must produce a VGate node — silent drop
        was the original symptom (only VSkip nodes survived)."""

        @qmc.qkernel
        def kern() -> qmc.Float:
            q = qmc.qubit_array(4, "q")
            qf = qmc.cast(q[1::2], qmc.QFixed, int_bits=0)
            return qmc.measure(qf)

        vc = _build_visual_circuit(kern)
        measure_count = sum(
            1
            for c in vc.children
            if isinstance(c, VGate) and c.kind == VGateKind.MEASURE_VECTOR
        )
        assert measure_count == 1, [type(c).__name__ for c in vc.children]

    def test_cast_root_qfixed_measure_targets_all_wires(self):
        """``cast(q, QFixed); measure`` still targets every wire when
        the cast source is the whole register (not a slice)."""

        @qmc.qkernel
        def kern() -> qmc.Float:
            q = qmc.qubit_array(4, "q")
            qf = qmc.cast(q, qmc.QFixed, int_bits=0)
            return qmc.measure(qf)

        vc = _build_visual_circuit(kern)
        m_indices: list[int] = []
        for c in vc.children:
            if isinstance(c, VGate) and c.kind == VGateKind.MEASURE_VECTOR:
                m_indices.extend(c.qubit_indices)
        assert sorted(m_indices) == [0, 1, 2, 3], m_indices


class TestSliceDrawSmoke:
    """End-to-end smoke: ``MatplotlibDrawer.draw_kernel`` runs without
    raising for every slice topology.

    The drawer end-to-end exercises analyzer + layout + renderer; this
    guard catches regressions where, for instance, layout chokes on a
    folded block whose affected qubits are non-contiguous, or the
    renderer fails to handle ``affected_qubits_precise`` dots.
    """

    def test_draw_all_slice_patterns_succeeds(self):
        """All four kernels must produce a non-empty matplotlib Figure."""
        kernels = [
            _explicit_slice_release,
            _inline_slice_assignment,
            _top_level_loop_pattern,
            _nested_slice,
        ]
        for kern in kernels:
            fig = MatplotlibDrawer.draw_kernel(kern)
            # The figure must have non-zero area and an axes attached.
            assert fig.get_size_inches()[0] > 0
            assert fig.get_size_inches()[1] > 0
