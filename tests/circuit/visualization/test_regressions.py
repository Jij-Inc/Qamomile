import io

import pytest

import qamomile.circuit as qm
from qamomile.circuit.visualization.analyzer import CircuitAnalyzer
from qamomile.circuit.visualization.style import DEFAULT_STYLE
from qamomile.circuit.visualization.visual_ir import (
    VFoldedBlock,
    VGate,
    VGateKind,
    VInlineBlock,
    VUnfoldedSequence,
)


class TestFormatSymbolicParam:
    """Unit tests for _format_symbolic_param."""

    def setup_method(self):
        # _format_symbolic_param only calls @staticmethod _extract_greek_prefix,
        # so we can skip __init__ entirely.
        self.analyzer = object.__new__(CircuitAnalyzer)

    def _fmt(self, name: str) -> str:
        return self.analyzer._format_symbolic_param(name)

    # --- Exact-match behaviour (unchanged) ---

    def test_exact_greek(self):
        assert self._fmt("theta") == r"$\theta$"
        assert self._fmt("phi") == r"$\phi$"

    def test_exact_greek_bracket(self):
        assert self._fmt("phi[0]") == r"$\phi[0]$"

    def test_exact_greek_underscore(self):
        assert self._fmt("theta_2") == r"$\theta_{2}$"

    def test_nested_subscripts(self):
        assert self._fmt("beta_a_b_c") == r"$\beta_{a_{b_{c}}}$"

    def test_already_tex(self):
        assert self._fmt(r"\theta") == r"$\theta$"

    # --- Non-Greek subscript (issue: non_greek_underscore_no_subscript) ---

    def test_non_greek_underscore(self):
        assert self._fmt("x_2") == r"$\mathrm{x}_{2}$"

    def test_non_greek_multi_underscore(self):
        assert self._fmt("my_var_1") == r"$\mathrm{my}_{var_{1}}$"

    def test_non_greek_bracket(self):
        assert self._fmt("data[i]") == r"$\mathrm{data}[i]$"

    # --- Greek prefix extraction (issue: greek_prefix_extraction) ---

    def test_greek_prefix_simple(self):
        assert self._fmt("phis") == r"${\phi}s$"
        assert self._fmt("thetas") == r"${\theta}s$"
        assert self._fmt("gammas") == r"${\gamma}s$"

    def test_greek_prefix_bracket(self):
        assert self._fmt("phis[i]") == r"${\phi}s[i]$"
        assert self._fmt("thetas[0]") == r"${\theta}s[0]$"

    def test_greek_prefix_underscore(self):
        assert self._fmt("phis_0") == r"${\phi}s_{0}$"
        assert self._fmt("gammas_0") == r"${\gamma}s_{0}$"

    # --- Plain text fallback ---

    def test_plain_text(self):
        assert self._fmt("params") == "params"
        assert self._fmt("x") == "x"


class TestDoubleSubscriptRegression:
    """Regression tests for double subscript error in Greek parameter names.

    See: .claude/workspace/reviewed/double_subscript_error_in_greek_parameter_names.md
    """

    @pytest.mark.mpl_image_compare(style="default")
    def test_greek_param_with_multiple_underscores(self):
        """Greek letter parameter with multiple underscores should not raise."""

        @qm.qkernel
        def circuit(q: qm.Qubit, beta_a_b_c: float) -> qm.Qubit:
            q = qm.rx(q, beta_a_b_c)
            return q

        fig = circuit.draw()
        # Force TeX parsing even without --mpl (e.g. in CI)
        fig.savefig(io.BytesIO(), format="png")
        return fig

    @pytest.mark.mpl_image_compare(style="default")
    def test_greek_param_with_three_underscores(self):
        """gamma_x_y_z should also render without error."""

        @qm.qkernel
        def circuit(q: qm.Qubit, gamma_x_y_z: float) -> qm.Qubit:
            q = qm.ry(q, gamma_x_y_z)
            return q

        fig = circuit.draw()
        # Force TeX parsing even without --mpl (e.g. in CI)
        fig.savefig(io.BytesIO(), format="png")
        return fig


class TestFloatArithmeticLabel:
    """Regression test: Float arithmetic should display symbolic expression, not 'float_tmp'."""

    @staticmethod
    def _get_labels(kernel):
        """Build visual IR and extract labels from a kernel."""
        graph = kernel._build_graph_for_visualization()
        analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE)
        qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
        vc = analyzer.build_visual_ir(graph, qubit_map, qubit_names, num_qubits)
        return [n.label for n in vc.children if hasattr(n, "label")]

    def test_float_div_label(self):
        @qm.qkernel
        def circuit(theta: qm.Float) -> qm.Bit:
            q = qm.qubit(name="q")
            q = qm.rx(q, theta / 2)
            return qm.measure(q)

        labels = self._get_labels(circuit)
        for label in labels:
            assert "float_tmp" not in label

    def test_float_mul_label(self):
        @qm.qkernel
        def circuit(theta: qm.Float) -> qm.Bit:
            q = qm.qubit(name="q")
            q = qm.ry(q, theta * 2)
            return qm.measure(q)

        labels = self._get_labels(circuit)
        for label in labels:
            assert "float_tmp" not in label


class TestDictParamSymbolicDraw:
    """Regression test: Dict type params should not block draw() when not provided."""

    def test_draw_dict_symbolic(self):
        """Dict型パラメータなしでdraw()できることを確認."""

        @qm.qkernel
        def items_demo(
            n_qubits: qm.UInt,
            ising: qm.Dict[qm.Tuple[qm.UInt, qm.UInt], qm.Float],
            gamma: qm.Float,
        ) -> qm.Vector[qm.Bit]:
            q = qm.qubit_array(n_qubits, name="q")
            for (i, j), Jij in qm.items(ising):
                q[i], q[j] = qm.rzz(q[i], q[j], gamma * Jij)
            return qm.measure(q)

        fig = items_demo.draw(n_qubits=3)
        assert fig is not None


class TestFreshQubitReturnCallBlock:
    """Regression: CallBlock returning fresh qubits should not crash draw()."""

    def test_draw_callblock_fresh_qubit_array_return(self):
        """superposition_vector(n) returns fresh qubits - draw should not crash."""
        from qamomile.circuit.algorithm.basic import superposition_vector

        @qm.qkernel
        def demo(n: int) -> qm.Vector[qm.Bit]:
            q = superposition_vector(n)
            return qm.measure(q)

        graph = demo._build_graph_for_visualization(n=3)
        analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE)
        qm_map, qm_names, num_q = analyzer.build_qubit_map(graph)
        assert num_q == 3

    def test_draw_callblock_pass_through_qubit_still_works(self):
        """Existing pass-through CallBlock must not regress."""

        @qm.qkernel
        def identity(q: qm.Qubit) -> qm.Qubit:
            return q

        @qm.qkernel
        def demo() -> qm.Bit:
            q = qm.qubit(name="q")
            q = identity(q)
            return qm.measure(q)

        graph = demo._build_graph_for_visualization()
        analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE)
        qm_map, qm_names, num_q = analyzer.build_qubit_map(graph)
        assert num_q == 1


class TestQPEPowerResolution:
    """Regression: ControlledU power=2**k should resolve per iteration."""

    def test_inline_unfolded_qpe_power_values(self):
        """QPE inline unfolded: oracle power should be 1, 2, 4 for 3 counting qubits."""

        @qm.qkernel
        def oracle(q: qm.Qubit, phi: float) -> qm.Qubit:
            return qm.p(q, phi)

        @qm.qkernel
        def qpe_circuit(phase: float) -> qm.Float:
            counting = qm.qubit_array(3, name="cnt")
            target = qm.qubit(name="tgt")
            target = qm.x(target)
            result: qm.QFixed = qm.qpe(target, counting, oracle, phi=phase)
            return qm.measure(result)

        graph = qpe_circuit._build_graph_for_visualization(phase=0.25)
        analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=True, fold_loops=False)
        qm_map, qm_names, num_q = analyzer.build_qubit_map(graph)
        vc = analyzer.build_visual_ir(graph, qm_map, qm_names, num_q)

        # Find unfolded sequence containing controlled-U blocks
        powers = []
        for node in vc.children:
            if isinstance(node, VUnfoldedSequence):
                for iteration in node.iterations:
                    for child in iteration:
                        if isinstance(child, VInlineBlock) and child.power >= 1:
                            powers.append(child.power)
        assert powers == [1, 2, 4], f"Expected [1, 2, 4], got {powers}"


class TestFoldedForCompositeGate:
    """Regression: Folded for with CompositeGateOperation should have body_lines."""

    def test_folded_for_with_composite_gate_has_body(self):
        """Folded for containing stub composite gate should show body expression."""

        @qm.composite_gate(stub=True, name="oracle", num_qubits=2)
        def oracle():
            pass

        @qm.qkernel
        def circuit() -> qm.Vector[qm.Bit]:
            q = qm.qubit_array(2, name="q")
            for i in qm.range(2):
                q[0], q[1] = oracle(q[0], q[1])
            return qm.measure(q)

        graph = circuit._build_graph_for_visualization()
        analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=False, fold_loops=True)
        qm_map, qm_names, num_q = analyzer.build_qubit_map(graph)
        vc = analyzer.build_visual_ir(graph, qm_map, qm_names, num_q)

        folded_blocks = [n for n in vc.children if isinstance(n, VFoldedBlock)]
        assert len(folded_blocks) >= 1
        for block in folded_blocks:
            if "oracle" in str(block.body_lines):
                assert len(block.body_lines) > 0, "body_lines should not be empty"
                assert any("oracle" in line for line in block.body_lines)
                return
        # If no oracle-containing block found, assert fails
        assert False, "No folded block with oracle body found"


class TestFoldedCallBlockSymbolicTex:
    """Regression: Folded CallBlock params should use TeX formatting."""

    def test_folded_callblock_param_has_tex(self):
        """Folded for + nested CallBlock: symbolic param should be TeX-formatted."""

        @qm.qkernel
        def rotate(q: qm.Qubit, omega: qm.Float) -> qm.Qubit:
            q = qm.rx(q, omega)
            return q

        @qm.qkernel
        def circuit(n: int, omegas: qm.Vector[qm.Float]) -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(n, name="q")
            for i in qm.range(n):
                q[i] = rotate(q[i], omegas[i])
            return q

        graph = circuit._build_graph_for_visualization(n=3)
        analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=False, fold_loops=True)
        qm_map, qm_names, num_q = analyzer.build_qubit_map(graph)
        vc = analyzer.build_visual_ir(graph, qm_map, qm_names, num_q)

        folded_blocks = [n for n in vc.children if isinstance(n, VFoldedBlock)]
        assert len(folded_blocks) >= 1
        # Body lines should contain TeX omega, not raw "omegas[i]"
        for block in folded_blocks:
            for line in block.body_lines:
                if "rotate" in line:
                    assert "omegas[i]" not in line, (
                        f"Raw string 'omegas[i]' found in body line: {line}"
                    )
                    return
        assert False, "No folded block with rotate body found"


class TestInlineUnfoldedCallBlockArrayIndex:
    """Regression: Inlined CallBlock array params should resolve per iteration."""

    def test_inline_unfolded_array_index_resolved_per_iteration(self):
        """Inlined CallBlock: omegas[i] should resolve to omegas[0/1/2]."""

        @qm.qkernel
        def rotate(q: qm.Qubit, omega: qm.Float) -> qm.Qubit:
            q = qm.rx(q, omega)
            return q

        @qm.qkernel
        def circuit(n: int, omegas: qm.Vector[qm.Float]) -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(n, name="q")
            for i in qm.range(n):
                q[i] = rotate(q[i], omegas[i])
            return q

        graph = circuit._build_graph_for_visualization(n=3)
        analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=True, fold_loops=False)
        qm_map, qm_names, num_q = analyzer.build_qubit_map(graph)
        vc = analyzer.build_visual_ir(graph, qm_map, qm_names, num_q)

        # Collect gate labels from unfolded iterations
        gate_labels: list[str] = []
        for node in vc.children:
            if isinstance(node, VUnfoldedSequence):
                for iteration in node.iterations:
                    for child in iteration:
                        if isinstance(child, VInlineBlock):
                            for grandchild in child.children:
                                if hasattr(grandchild, "label"):
                                    gate_labels.append(grandchild.label)

        # Labels should NOT all be identical (index should vary per iteration)
        if len(gate_labels) >= 2:
            assert len(set(gate_labels)) > 1, (
                f"All gate labels are identical: {gate_labels}. "
                f"Array index should resolve per iteration."
            )


class TestQAOARangeAliasDisplay:
    """Regression: Unbound array loop stop should show name, not '?'."""

    def test_named_value_fallback_no_question_mark(self):
        """_format_value_as_expression should return value.name, not '?'."""
        from qamomile.circuit.ir.types.primitives import UIntType
        from qamomile.circuit.ir.value import Value

        analyzer = object.__new__(CircuitAnalyzer)
        analyzer.graph = None

        # Simulate an unbound array shape value (e.g., edges_dim0)
        value = Value(type=UIntType(), name="edges_dim0")
        result = analyzer._format_value_as_expression(value, set())
        assert result == "edges_dim0", f"Expected 'edges_dim0', got '{result}'"
        assert "?" not in result

    def test_anonymous_value_still_returns_question_mark(self):
        """Anonymous value (empty name) should still return '?'."""
        from qamomile.circuit.ir.types.primitives import UIntType
        from qamomile.circuit.ir.value import Value

        analyzer = object.__new__(CircuitAnalyzer)
        analyzer.graph = None

        value = Value(type=UIntType(), name="")
        result = analyzer._format_value_as_expression(value, set())
        assert result == "?"

    def test_bound_range_shows_concrete(self):
        """Bound array: header should show concrete value like qm.range(3)."""

        @qm.qkernel
        def circuit(n: int) -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(n, name="q")
            for e in qm.range(n):
                q[0] = qm.rz(q[0], 0.5)
            return q

        graph = circuit._build_graph_for_visualization(n=3)
        analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=False, fold_loops=True)
        qm_map, qm_names, num_q = analyzer.build_qubit_map(graph)
        vc = analyzer.build_visual_ir(graph, qm_map, qm_names, num_q)

        folded_blocks = [n for n in vc.children if isinstance(n, VFoldedBlock)]
        assert len(folded_blocks) >= 1
        found_range_3 = any(
            "qm.range(3)" in block.header_label for block in folded_blocks
        )
        assert found_range_3, (
            f"Expected 'qm.range(3)' in a folded header, got: "
            f"{[b.header_label for b in folded_blocks]}"
        )

    def test_nested_loop_range_not_regressed(self):
        """Nested loop: inner range(j) should still display correctly."""

        @qm.qkernel
        def circuit(n: int) -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(n, name="q")
            for j in qm.range(n):
                for k in qm.range(j):
                    q[j], q[k] = qm.cx(q[j], q[k])
            return q

        graph = circuit._build_graph_for_visualization(n=3)
        analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=False, fold_loops=True)
        qm_map, qm_names, num_q = analyzer.build_qubit_map(graph)
        vc = analyzer.build_visual_ir(graph, qm_map, qm_names, num_q)

        folded_blocks = [n for n in vc.children if isinstance(n, VFoldedBlock)]
        assert len(folded_blocks) >= 1
        for block in folded_blocks:
            if "qm.range(3)" in block.header_label:
                for line in block.body_lines:
                    if "qm.range" in line:
                        assert "?" not in line, (
                            f"Nested loop range contains '?': {line}"
                        )


class TestFreshQubitCallBlockVisualization:
    """Regression: superposition_vector(n) fresh-qubit CallBlock visualization."""

    @staticmethod
    def _make_demo_kernel():
        from qamomile.circuit.algorithm.basic import superposition_vector

        @qm.qkernel
        def demo(n: int) -> qm.Vector[qm.Bit]:
            q = superposition_vector(n)
            return qm.measure(q)

        return demo

    def test_inline_true_no_crash(self):
        """inline=True: build_qubit_map + build_visual_ir should not raise."""
        demo = self._make_demo_kernel()
        graph = demo._build_graph_for_visualization(n=3)
        analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=True)
        qm_map, qm_names, num_q = analyzer.build_qubit_map(graph)
        assert num_q == 3
        vc = analyzer.build_visual_ir(graph, qm_map, qm_names, num_q)
        assert vc.num_qubits == 3

    def test_inline_depth_1_no_crash(self):
        """inline=True with inline_depth=1: boundary case."""
        demo = self._make_demo_kernel()
        graph = demo._build_graph_for_visualization(n=3)
        analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=True, inline_depth=1)
        qm_map, qm_names, num_q = analyzer.build_qubit_map(graph)
        assert num_q == 3

    def test_block_box_has_qubit_indices(self):
        """inline=False: BLOCK_BOX should span all fresh-return qubits."""
        demo = self._make_demo_kernel()
        graph = demo._build_graph_for_visualization(n=3)
        analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=False)
        qm_map, qm_names, num_q = analyzer.build_qubit_map(graph)
        vc = analyzer.build_visual_ir(graph, qm_map, qm_names, num_q)
        block_boxes = [
            n
            for n in vc.children
            if isinstance(n, VGate) and n.kind == VGateKind.BLOCK_BOX
        ]
        assert len(block_boxes) >= 1, "Expected at least one BLOCK_BOX"
        assert sorted(block_boxes[0].qubit_indices) == [0, 1, 2]

    def test_measure_vector_expands(self):
        """inline=False: MEASURE_VECTOR should span all fresh-return qubits."""
        demo = self._make_demo_kernel()
        graph = demo._build_graph_for_visualization(n=3)
        analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=False)
        qm_map, qm_names, num_q = analyzer.build_qubit_map(graph)
        vc = analyzer.build_visual_ir(graph, qm_map, qm_names, num_q)
        measure_nodes = [
            n
            for n in vc.children
            if isinstance(n, VGate) and n.kind == VGateKind.MEASURE_VECTOR
        ]
        assert len(measure_nodes) >= 1, "Expected at least one MEASURE_VECTOR"
        assert sorted(measure_nodes[0].qubit_indices) == [0, 1, 2]

    def test_boundary_n1_n2(self):
        """Boundary: n=1 and n=2 should work for both inline modes."""
        demo = self._make_demo_kernel()
        for n_val in (1, 2):
            for inline in (True, False):
                graph = demo._build_graph_for_visualization(n=n_val)
                analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=inline)
                qm_map, qm_names, num_q = analyzer.build_qubit_map(graph)
                assert num_q == n_val, (
                    f"n={n_val}, inline={inline}: expected {n_val} qubits, got {num_q}"
                )
                vc = analyzer.build_visual_ir(graph, qm_map, qm_names, num_q)
                assert vc.num_qubits == n_val

    @pytest.mark.parametrize("n_val", [2, 3])
    def test_inline_affected_qubits_fold_true(self, n_val):
        """inline=True, fold_loops=True: parent VInlineBlock spans all fresh qubits."""
        demo = self._make_demo_kernel()
        graph = demo._build_graph_for_visualization(n=n_val)
        analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=True, fold_loops=True)
        qm_map, qm_names, num_q = analyzer.build_qubit_map(graph)
        vc = analyzer.build_visual_ir(graph, qm_map, qm_names, num_q)
        inline_blocks = [n for n in vc.children if isinstance(n, VInlineBlock)]
        assert len(inline_blocks) >= 1, "Expected at least one VInlineBlock"
        assert sorted(inline_blocks[0].affected_qubits) == list(range(n_val))

    @pytest.mark.parametrize("n_val", [2, 3])
    def test_inline_affected_qubits_fold_false(self, n_val):
        """inline=True, fold_loops=False: parent VInlineBlock spans all fresh qubits."""
        demo = self._make_demo_kernel()
        graph = demo._build_graph_for_visualization(n=n_val)
        analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=True, fold_loops=False)
        qm_map, qm_names, num_q = analyzer.build_qubit_map(graph)
        vc = analyzer.build_visual_ir(graph, qm_map, qm_names, num_q)
        inline_blocks = [n for n in vc.children if isinstance(n, VInlineBlock)]
        assert len(inline_blocks) >= 1, "Expected at least one VInlineBlock"
        assert sorted(inline_blocks[0].affected_qubits) == list(range(n_val))

    @pytest.mark.parametrize("fold_loops", [True, False])
    def test_inline_layout_block_ranges(self, fold_loops):
        """inline=True: layout block_ranges qubit_indices should span all fresh qubits."""
        from qamomile.circuit.visualization.layout import CircuitLayoutEngine

        demo = self._make_demo_kernel()
        graph = demo._build_graph_for_visualization(n=3)
        analyzer = CircuitAnalyzer(
            graph, DEFAULT_STYLE, inline=True, fold_loops=fold_loops
        )
        qm_map, qm_names, num_q = analyzer.build_qubit_map(graph)
        vc = analyzer.build_visual_ir(graph, qm_map, qm_names, num_q)
        layout_engine = CircuitLayoutEngine(DEFAULT_STYLE)
        layout = layout_engine.compute_layout(vc)
        # At least one block_range should cover all 3 qubits
        found = any(
            sorted(br["qubit_indices"]) == [0, 1, 2] for br in layout.block_ranges
        )
        assert found, (
            f"No block_range spans [0,1,2]; got {[br['qubit_indices'] for br in layout.block_ranges]}"
        )


class TestDeutschPassthroughNoRegression:
    """Negative regression: pass-through CallBlock should retain correct affected_qubits."""

    def test_deutsch_all_h_inline(self):
        """deutsch kernel's _all_h pass-through CallBlock keeps affected_qubits == [0, 1]."""
        from tests.circuit.qkernel_catalog import deutsch

        graph = deutsch._build_graph_for_visualization()
        analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=True)
        qm_map, qm_names, num_q = analyzer.build_qubit_map(graph)
        vc = analyzer.build_visual_ir(graph, qm_map, qm_names, num_q)
        inline_blocks = [n for n in vc.children if isinstance(n, VInlineBlock)]
        # _all_h should span both qubits [0, 1]
        all_h_blocks = [b for b in inline_blocks if "all_h" in b.label.lower()]
        assert len(all_h_blocks) >= 1, "Expected _all_h inline block"
        assert sorted(all_h_blocks[0].affected_qubits) == [0, 1]


_TMP_TOKENS = ("uint_tmp", "float_tmp", "bit_tmp")


def _iter_labels(node):
    """Yield every human-readable label string reachable from a VisualNode."""
    if isinstance(node, VGate):
        yield node.label
    elif isinstance(node, VInlineBlock):
        yield node.label
        for child in node.children:
            yield from _iter_labels(child)
    elif isinstance(node, VFoldedBlock):
        yield node.header_label
        for line in node.body_lines:
            yield line
    elif isinstance(node, VUnfoldedSequence):
        for iteration in node.iterations:
            for child in iteration:
                yield from _iter_labels(child)


def _assert_no_tmp_leakage(vc, context=""):
    """Assert no IR-internal temp name leaks into any rendered label."""
    for child in vc.children:
        for label in _iter_labels(child):
            for token in _TMP_TOKENS:
                assert token not in label, (
                    f"{context}: internal temp '{token}' leaked into label {label!r}"
                )


class TestNoInternalTempLeakage:
    """Regression: IR-internal placeholder names must not reach rendered labels."""

    def _build_vqe_kernel(self):
        """The VQE-ansatz shape the user reported."""

        @qm.qkernel
        def ry_layer(
            q: qm.Vector[qm.Qubit], thetas: qm.Vector[qm.Float], base: qm.UInt
        ) -> qm.Vector[qm.Qubit]:
            for i in qm.range(2):
                q[i] = qm.ry(q[i], thetas[base + i])
            return q

        @qm.qkernel
        def rz_layer(
            q: qm.Vector[qm.Qubit], thetas: qm.Vector[qm.Float], base: qm.UInt
        ) -> qm.Vector[qm.Qubit]:
            for i in qm.range(2):
                q[i] = qm.rz(q[i], thetas[base + i])
            return q

        @qm.qkernel
        def cx_entangling_layer(
            q: qm.Vector[qm.Qubit],
        ) -> qm.Vector[qm.Qubit]:
            for i in qm.range(1):
                q[i], q[i + 1] = qm.cx(q[i], q[i + 1])
            return q

        @qm.qkernel
        def vqe_ansatz(
            n: qm.UInt, reps: qm.UInt, thetas: qm.Vector[qm.Float]
        ) -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(n, name="q")
            for r in qm.range(reps):
                base = r * 2 * n
                q = ry_layer(q, thetas, base)
                q = rz_layer(q, thetas, base + n)
                q = cx_entangling_layer(q)
            final_base = reps * 2 * n
            q = ry_layer(q, thetas, final_base)
            q = rz_layer(q, thetas, final_base + n)
            return q

        return vqe_ansatz

    def test_bug_a_folded_body_no_tmp(self):
        """Folded for-body with BinOp CallBlock arg resolves symbolically."""
        kernel = self._build_vqe_kernel()
        graph = kernel._build_graph_for_visualization(n=2, reps=2)
        analyzer = CircuitAnalyzer(
            graph, DEFAULT_STYLE, inline=False, fold_loops=True
        )
        qm_map, qm_names, num_q = analyzer.build_qubit_map(graph)
        vc = analyzer.build_visual_ir(graph, qm_map, qm_names, num_q)

        _assert_no_tmp_leakage(vc, context="bug_a folded")

        # Positive check: folded body should carry a symbolic arg like "r*2*...".
        folded = [n for n in vc.children if isinstance(n, VFoldedBlock)]
        assert folded, "Expected at least one VFoldedBlock"
        joined = " ".join(line for b in folded for line in b.body_lines)
        assert "ry_layer" in joined
        assert "r*2" in joined, (
            f"Expected symbolic BinOp form in folded body; got {joined!r}"
        )

    def test_bug_b_inline_unfolded_no_tmp(self):
        """Inline/unfolded top-level CallBlock resolves concrete theta indices."""
        kernel = self._build_vqe_kernel()
        graph = kernel._build_graph_for_visualization(n=2, reps=2)
        analyzer = CircuitAnalyzer(
            graph, DEFAULT_STYLE, inline=True, fold_loops=False
        )
        qm_map, qm_names, num_q = analyzer.build_qubit_map(graph)
        vc = analyzer.build_visual_ir(graph, qm_map, qm_names, num_q)

        _assert_no_tmp_leakage(vc, context="bug_b inline unfolded")

        # Positive check: some gate label should reference a concrete index
        # like "thetas[8]" or "thetas[9]" from the post-loop ry_layer.
        all_labels = [lbl for c in vc.children for lbl in _iter_labels(c)]
        assert any("8" in lbl or "9" in lbl for lbl in all_labels), (
            f"Expected concrete index in post-loop gate label; got {all_labels!r}"
        )

    def test_controlled_u_folded_binop_arg(self):
        """ControlledU inside a folded for-body with a BinOp arg — no tmp leak."""

        @qm.qkernel
        def inner(q: qm.Qubit, phi: qm.Float) -> qm.Qubit:
            q = qm.rz(q, phi)
            return q

        @qm.qkernel
        def outer(n: qm.UInt, phis: qm.Vector[qm.Float]) -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(n, name="q")
            for i in qm.range(n):
                shift = i * 2
                q[i] = inner(q[i], phis[shift])
            return q

        graph = outer._build_graph_for_visualization(n=3)
        analyzer = CircuitAnalyzer(
            graph, DEFAULT_STYLE, inline=False, fold_loops=True
        )
        qm_map, qm_names, num_q = analyzer.build_qubit_map(graph)
        vc = analyzer.build_visual_ir(graph, qm_map, qm_names, num_q)
        _assert_no_tmp_leakage(vc, context="controlled_u folded")

    def test_nested_callblock_passes_binop_through(self):
        """Outer CallBlock computes a BinOp and passes it to inner CallBlock."""

        @qm.qkernel
        def leaf(q: qm.Qubit, k: qm.UInt) -> qm.Qubit:
            q = qm.h(q)
            return q

        @qm.qkernel
        def middle(q: qm.Qubit, m: qm.UInt) -> qm.Qubit:
            k = m + 1
            q = leaf(q, k)
            return q

        @qm.qkernel
        def top(n: qm.UInt) -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(n, name="q")
            for i in qm.range(n):
                q[i] = middle(q[i], i * 2)
            return q

        graph = top._build_graph_for_visualization(n=3)
        for fold, inline in [(True, False), (False, False), (False, True)]:
            analyzer = CircuitAnalyzer(
                graph, DEFAULT_STYLE, inline=inline, fold_loops=fold
            )
            qm_map, qm_names, num_q = analyzer.build_qubit_map(graph)
            vc = analyzer.build_visual_ir(graph, qm_map, qm_names, num_q)
            _assert_no_tmp_leakage(
                vc, context=f"nested callblock fold={fold} inline={inline}"
            )

    def test_catalog_sweep_no_tmp_any_mode(self):
        """Every catalog kernel renders without tmp-name leakage in any mode."""
        from tests.circuit.qkernel_catalog import QKERNEL_CATALOG

        modes = [
            (True, False),
            (False, False),
            (True, True),
            (False, True),
        ]
        for entry in QKERNEL_CATALOG:
            try:
                graph = entry.qkernel._build_graph_for_visualization(
                    **entry.min_params
                )
            except Exception:
                # Skip kernels that can't be built with min_params
                # (e.g., kernels requiring unsupported param types).
                continue
            for fold_loops, inline in modes:
                analyzer = CircuitAnalyzer(
                    graph,
                    DEFAULT_STYLE,
                    inline=inline,
                    fold_loops=fold_loops,
                )
                qm_map, qm_names, num_q = analyzer.build_qubit_map(graph)
                try:
                    vc = analyzer.build_visual_ir(graph, qm_map, qm_names, num_q)
                except NotImplementedError:
                    # Some kernels use IfOperation/WhileOperation which
                    # the visualizer does not yet support.
                    continue
                _assert_no_tmp_leakage(
                    vc,
                    context=(
                        f"catalog[{entry.id}] fold={fold_loops} inline={inline}"
                    ),
                )


class TestNestedLoopAffectedQubits:
    """Regression (bug_002): nested control-flow qubits must not be dropped by precise walk."""

    def test_mixed_rotation_plus_nested_entangling_loop(self):
        """Outer loop with parameterized rotation + nested entangling For covers all touched qubits."""

        @qm.qkernel
        def circuit(theta: qm.Float) -> qm.Vector[qm.Bit]:
            ancilla = qm.qubit_array(2, name="ancilla")
            data = qm.qubit_array(4, name="data")
            for i in qm.range(2):
                ancilla[i] = qm.rx(ancilla[i], theta)
                for j in qm.range(4):
                    ancilla[i], data[j] = qm.cx(ancilla[i], data[j])
            return qm.measure(data)

        graph = circuit._build_graph_for_visualization(theta=0.5)
        analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=False, fold_loops=True)
        qm_map, qm_names, num_q = analyzer.build_qubit_map(graph)
        vc = analyzer.build_visual_ir(graph, qm_map, qm_names, num_q)

        folded_blocks = [n for n in vc.children if isinstance(n, VFoldedBlock)]
        assert len(folded_blocks) >= 1, "Expected at least one folded for-block"
        outer = folded_blocks[0]
        # Outer loop must span both ancilla (indices 0, 1) and data (2, 3, 4, 5).
        assert sorted(outer.affected_qubits) == [0, 1, 2, 3, 4, 5]

    def test_nested_for_precision_over_fallback(self):
        """Hybrid precise recursion beats fallback: box should span only touched wires."""

        @qm.qkernel
        def circuit() -> qm.Vector[qm.Bit]:
            q = qm.qubit_array(10, name="q")
            data = qm.qubit_array(10, name="data")
            for i in qm.range(2):
                for j in qm.range(3):
                    q[i], data[j] = qm.cx(q[i], data[j])
            return qm.measure(data)

        graph = circuit._build_graph_for_visualization()
        analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE, inline=False, fold_loops=True)
        qm_map, qm_names, num_q = analyzer.build_qubit_map(graph)
        vc = analyzer.build_visual_ir(graph, qm_map, qm_names, num_q)

        folded_blocks = [n for n in vc.children if isinstance(n, VFoldedBlock)]
        assert len(folded_blocks) >= 1
        outer = folded_blocks[0]
        # q has base index 0, data has base index 10. Loop touches q[0], q[1],
        # data[0], data[1], data[2] — precise recursion should return exactly
        # those 5 wires, not the 20-wire over-approximation a pure fallback
        # would yield.
        assert sorted(outer.affected_qubits) == [0, 1, 10, 11, 12]
