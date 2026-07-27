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

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.visualization.analyzer import CircuitAnalyzer
from qamomile.circuit.visualization.drawer import MatplotlibDrawer
from qamomile.circuit.visualization.layout import CircuitLayoutEngine
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
    # Strict-return: ``even`` still owns its non-overlap slots {4, 6};
    # discharge it before measuring the parent.
    q[0::2] = even
    return qmc.measure(q)


@qmc.qkernel
def _h_all_helper(v: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Whole-register Hadamard helper called with a slice view."""
    for i in qmc.range(v.shape[0]):
        v[i] = qmc.h(v[i])
    return v


@qmc.qkernel
def _h_broadcast_helper(v: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Helper that broadcasts H over the whole input via ``qmc.h(v)``.

    Distinct from :func:`_h_all_helper` (which uses an explicit index
    loop): this body exercises the ``ArrayValue`` + ``shape`` operand
    resolution path in ``_resolve_non_element_operand`` (whole-vector
    broadcast), as opposed to the per-element loop body.
    """
    return qmc.h(v)


@qmc.qkernel
def _stride_slice_broadcast_helper_call() -> qmc.Vector[qmc.Bit]:
    """Stride slice handed to a broadcast helper over 6 qubits.

    Exercises the ``ArrayValue`` + ``shape`` path with a non-
    contiguous slice-aliased parameter (the helper's ``v`` aliases
    root wires ``{0, 2, 4}``).  The ``base + k`` formula in that path
    would render H on contiguous wires ``{0, 1, 2}``; the element-
    key lookup added in the same fix routes it to ``{0, 2, 4}``.
    """
    q = qmc.qubit_array(6, "q")
    evens = q[0::2]
    evens = _h_broadcast_helper(evens)
    q[0::2] = evens
    return qmc.measure(q)


@qmc.qkernel
def _stride_slice_helper_call() -> qmc.Vector[qmc.Bit]:
    """Stride slice ``q[0::2]`` handed to a sub-kernel over 6 qubits."""
    q = qmc.qubit_array(6, "q")
    evens = q[0::2]
    evens = _h_all_helper(evens)
    q[0::2] = evens
    return qmc.measure(q)


@qmc.qkernel
def _window_slice_helper_call() -> qmc.Vector[qmc.Bit]:
    """Contiguous window ``q[1:4]`` handed to a sub-kernel over 6 qubits."""
    q = qmc.qubit_array(6, "q")
    mid = q[1:4]
    mid = _h_all_helper(mid)
    q[1:4] = mid
    return qmc.measure(q)


@qmc.qkernel
def _x_all_helper(v: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Whole-register X helper, used alongside _h_all_helper for disjoint slices."""
    for i in qmc.range(v.shape[0]):
        v[i] = qmc.x(v[i])
    return v


@qmc.qkernel
def _disjoint_slice_helper_calls() -> qmc.Vector[qmc.Bit]:
    """Two disjoint slices each handed to a different sub-kernel."""
    q = qmc.qubit_array(6, "q")
    evens = q[0::2]
    odds = q[1::2]
    evens = _h_all_helper(evens)
    odds = _x_all_helper(odds)
    q[0::2] = evens
    q[1::2] = odds
    return qmc.measure(q)


@qmc.qkernel
def _h_all_outer_helper(v: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Two-level helper that forwards its slice-aliased parameter."""
    v = _h_all_helper(v)
    return v


@qmc.qkernel
def _stride_slice_nested_helper_call() -> qmc.Vector[qmc.Bit]:
    """Stride slice forwarded through a two-level helper chain."""
    q = qmc.qubit_array(6, "q")
    evens = q[0::2]
    evens = _h_all_outer_helper(evens)
    q[0::2] = evens
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
            assert len(fig.axes) > 0


class TestSliceSubKernelArgumentDraw:
    """Slice views handed as sub-kernel arguments must alias root wires.

    Before the slice-view-aware ``InvokeOperation`` handling in
    ``build_qubit_map`` and the matching slice-of walk in
    ``_resolve_non_element_operand``, passing a slice view to a
    ``@qkernel`` registered the view's logical id as a single fresh
    wire.  The sub-kernel's invoke box then claimed phantom wire
    indices outside the actual qubit range, and the renderer crashed
    with ``IndexError`` translating those phantom indices to
    y-coordinates.  The tests below pin the affected-qubit set of the
    call-block box for each slice topology so the regression cannot
    silently come back.
    """

    def test_stride_slice_helper_box_targets_strided_wires(self):
        """``_h_all_helper(q[0::2])`` box lands on root {0, 2, 4}."""
        vc = _build_visual_circuit(_stride_slice_helper_call)
        boxes = [
            c
            for c in vc.children
            if isinstance(c, VGate) and c.kind == VGateKind.BLOCK_BOX
        ]
        assert len(boxes) == 1, vc.children
        assert sorted(boxes[0].qubit_indices) == [0, 2, 4]

    def test_window_slice_helper_box_targets_contiguous_wires(self):
        """``_h_all_helper(q[1:4])`` box lands on root {1, 2, 3}."""
        vc = _build_visual_circuit(_window_slice_helper_call)
        boxes = [
            c
            for c in vc.children
            if isinstance(c, VGate) and c.kind == VGateKind.BLOCK_BOX
        ]
        assert len(boxes) == 1, vc.children
        assert sorted(boxes[0].qubit_indices) == [1, 2, 3]

    def test_nested_helper_chain_preserves_slice_alias(self):
        """Slice forwarded through two helpers keeps its root wires.

        The outer helper passes its slice-aliased parameter on to the
        inner helper.  The outer box must still cover the original
        root wires {0, 2, 4} — i.e., the alias must propagate.
        """
        vc = _build_visual_circuit(_stride_slice_nested_helper_call)
        boxes = [
            c
            for c in vc.children
            if isinstance(c, VGate) and c.kind == VGateKind.BLOCK_BOX
        ]
        assert len(boxes) == 1, vc.children
        assert sorted(boxes[0].qubit_indices) == [0, 2, 4]

    def test_inline_stride_slice_helper_unfolds_on_strided_wires(self):
        """With ``inline=True`` the sub-kernel's H gates unfold onto
        the same root wires the slice covers."""
        vc = _build_visual_circuit(
            _stride_slice_helper_call, inline=True, fold_loops=False
        )

        def collect_gate_wires(children) -> list[int]:
            hits: list[int] = []
            for c in children:
                if isinstance(c, VGate) and c.kind == VGateKind.GATE:
                    hits.extend(c.qubit_indices)
                elif isinstance(c, VUnfoldedSequence):
                    for iter_children in c.iterations:
                        hits.extend(collect_gate_wires(iter_children))
                elif hasattr(c, "children"):
                    hits.extend(collect_gate_wires(c.children))
            return hits

        assert sorted(collect_gate_wires(vc.children)) == [0, 2, 4]

    def test_draw_all_slice_helper_patterns_succeeds(self):
        """End-to-end smoke for slice + sub-kernel: ``draw_kernel``
        must produce a non-empty Figure without raising."""
        kernels = [
            _stride_slice_helper_call,
            _window_slice_helper_call,
            _stride_slice_nested_helper_call,
            _disjoint_slice_helper_calls,
        ]
        for kern in kernels:
            fig = MatplotlibDrawer.draw_kernel(kern)
            assert fig.get_size_inches()[0] > 0
            assert fig.get_size_inches()[1] > 0
            assert len(fig.axes) > 0

    def test_disjoint_slice_helper_calls_have_no_phantom_wires(self):
        """Two disjoint slice views passed to different sub-kernels must
        not allocate phantom wires past the root register.

        Regression: ``map_block_results`` previously treated a slice
        view operand as an unregistered array and allocated a fresh
        wire for the call's qubit-typed result.  With two such calls
        on disjoint slices the rendered figure grew an extra wire per
        call past the actual qubit count.  The fix resolves a whole
        slice view to its root's first slice-covered element key in
        ``_resolve_array_element_lid`` so the existing wire is reused.
        """
        graph = _disjoint_slice_helper_calls._build_graph_for_visualization()
        analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE)
        _, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
        assert num_qubits == 6, (num_qubits, qubit_names)
        assert set(qubit_names.values()) == {f"q[{i}]" for i in range(6)}

    def test_disjoint_slice_helper_boxes_target_correct_wires(self):
        """Each call-block box must cover only its slice's root wires."""
        vc = _build_visual_circuit(_disjoint_slice_helper_calls)
        boxes = [
            c
            for c in vc.children
            if isinstance(c, VGate) and c.kind == VGateKind.BLOCK_BOX
        ]
        assert len(boxes) == 2, vc.children
        box_wires = sorted(sorted(b.qubit_indices) for b in boxes)
        assert box_wires == [[0, 2, 4], [1, 3, 5]], box_wires

    @pytest.mark.parametrize("fold_loops", [True, False])
    def test_disjoint_inline_blocks_do_not_overlap_in_layout(self, fold_loops):
        """Two interleaved-but-disjoint inline blocks must be laid out
        sequentially, not overlap horizontally.

        Regression: ``_place_vinline_block`` reserved column space for
        ``affected_qubits`` only — the wires the block directly touched.
        For ``h_all(q[0::2])`` followed by ``x_all(q[1::2])`` the two
        blocks share *no* affected wire, so x_all slid into h_all's
        x-slot, and their dashed boxes (visually spanning q[0..4] and
        q[1..5] respectively) overlapped on top of each other.  The fix
        reserves the full visual span ``min(affected)..max(affected)``
        when placing a ``VInlineBlock``.
        """
        graph = _disjoint_slice_helper_calls._build_graph_for_visualization()
        analyzer = CircuitAnalyzer(
            graph, DEFAULT_STYLE, inline=True, fold_loops=fold_loops
        )
        qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
        vc = analyzer.build_visual_ir(graph, qubit_map, qubit_names, num_qubits)
        layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(vc)

        ranges_by_name: dict[str, tuple[float, float]] = {
            br["name"]: (br["start_x"], br["end_x"]) for br in layout.block_ranges
        }
        assert (
            "_h_all_helper" in ranges_by_name and "_x_all_helper" in ranges_by_name
        ), ranges_by_name
        h_start, h_end = ranges_by_name["_h_all_helper"]
        x_start, x_end = ranges_by_name["_x_all_helper"]
        # _h_all_helper is invoked first; its right edge must lie
        # strictly before _x_all_helper's left edge.
        assert h_end < x_start, (
            f"_h_all_helper=[{h_start}, {h_end}] overlaps "
            f"_x_all_helper=[{x_start}, {x_end}]"
        )

    def test_broadcast_in_helper_lands_on_slice_wires(self):
        """``qmc.h(v)`` inside a helper called with a slice view must
        render on the slice's root-space wires, not phantom contiguous
        ones.

        Regression: ``_resolve_non_element_operand``'s
        ``ArrayValue`` + ``shape`` branch always returned
        ``[base_idx + k for k in range(size)]``.  For a slice-aliased
        callee parameter (``build_qubit_map`` pre-populates
        per-element keys to non-contiguous root wires), that formula
        renders H on consecutive wires starting from the slice's
        first wire instead of the actual stride wires the slice
        covers.  The fix prefers ``f"{lid}_[{k}]"`` lookups when
        present and only falls back to the formula for truly
        contiguous arrays.
        """
        vc = _build_visual_circuit(_stride_slice_broadcast_helper_call)
        # The helper call renders as a BLOCK_BOX whose qubit_indices
        # must cover the stride wires (root q[0], q[2], q[4]).
        boxes = [
            c
            for c in vc.children
            if isinstance(c, VGate) and c.kind == VGateKind.BLOCK_BOX
        ]
        assert len(boxes) == 1, vc.children
        assert sorted(boxes[0].qubit_indices) == [0, 2, 4], boxes[0].qubit_indices


class TestSymbolicSliceBoundsInLoopUnfold:
    """Slices whose bounds clamp through ``_uint_min`` must still
    resolve to root wires once the surrounding loop is unrolled.

    Writing ``q[0:target_index]`` with a symbolic ``target_index``
    runs the frontend's ``_uint_min(0, min(target_index, parent_len))``
    clamp; the slice's ``slice_start`` becomes a symbolic ``BinOp``
    result rather than a constant ``0``.  Before
    ``_resolve_view_chain_to_root`` learned to fall back to
    ``_evaluate_value`` for non-constant slice bounds, the analyzer
    silently dropped every control qubit of a ``SymbolicControlledU``
    whose pool was such a slice — the rendered figure showed a bare
    target X gate (``control_count == 0``) on each unrolled iteration
    instead of the MCX ladder the IR actually expresses.  These
    tests pin both the per-iteration ``control_count`` and the
    expected control + target wire mapping.
    """

    @staticmethod
    def _multi_arg_mcx_ladder():
        """Return a kernel that drives the MCX ladder this regression
        guards: each iteration's pool is ``q[0:target_index]`` with a
        symbolic, loop-variable-dependent ``target_index``."""

        @qmc.qkernel
        def kern(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, "q")
            for k in qmc.range(n - 1):
                target_index = n - 1 - k
                mcx = qmc.control(qmc.x, num_controls=target_index)
                q[0:target_index], q[target_index] = mcx(
                    q[0:target_index],
                    q[target_index],
                )
            return qmc.measure(q)

        return kern

    @staticmethod
    def _build_with_kwargs(kernel, **build_kwargs):
        """Build a VisualCircuit, forwarding kernel-side kwargs through
        ``_build_graph_for_visualization``.

        The shared :func:`_build_visual_circuit` helper threads its
        kwargs into :class:`CircuitAnalyzer` instead of the kernel,
        which is the right surface for analyzer flags but cannot
        carry ``n=5`` to the kernel's Vector-size parameter.  This
        helper does both: passes ``n=5`` to the kernel and sets
        ``fold_loops=False`` on the analyzer so the loop unrolls.
        """
        graph = kernel._build_graph_for_visualization(**build_kwargs)
        analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, fold_loops=False)
        qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
        return analyzer.build_visual_ir(graph, qubit_map, qubit_names, num_qubits)

    def test_unrolled_ladder_emits_descending_control_counts(self):
        """Each unrolled iteration must expose the MCX size for that step.

        With ``n=5`` the loop body runs four times and the wrapped X
        gains one control fewer per iteration: 4, 3, 2, 1.  The
        analyzer's slice-aware chain walk produces the same ladder
        the full transpile pipeline emits.
        """
        kern = self._multi_arg_mcx_ladder()
        vc = self._build_with_kwargs(kern, n=5)
        unfolded = [c for c in vc.children if isinstance(c, VUnfoldedSequence)]
        assert len(unfolded) == 1, vc.children
        control_counts: list[int] = []
        for iteration in unfolded[0].iterations:
            for node in iteration:
                if isinstance(node, VGate) and node.kind == VGateKind.CONTROLLED_U_BOX:
                    control_counts.append(node.control_count)
        assert control_counts == [4, 3, 2, 1], control_counts

    def test_unrolled_ladder_targets_one_wire_per_iteration(self):
        """Each unrolled iteration places the X on the iteration's
        target wire while the prefix wires appear as controls.

        For ``n=5`` and ``target_index = n - 1 - k``, iteration ``k``
        targets ``q[target_index]`` with prefix controls
        ``q[0:target_index]``.  After resolution the
        ``qubit_indices`` list is ``controls + [target]``: e.g. for
        ``k=0`` it is ``[0, 1, 2, 3, 4]``, for ``k=1`` it is
        ``[0, 1, 2, 3]``.
        """
        kern = self._multi_arg_mcx_ladder()
        vc = self._build_with_kwargs(kern, n=5)
        unfolded = next(c for c in vc.children if isinstance(c, VUnfoldedSequence))
        expected = [
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3],
            [0, 1, 2],
            [0, 1],
        ]
        observed: list[list[int]] = []
        for iteration in unfolded.iterations:
            for node in iteration:
                if isinstance(node, VGate) and node.kind == VGateKind.CONTROLLED_U_BOX:
                    observed.append(list(node.qubit_indices))
        assert observed == expected, observed
