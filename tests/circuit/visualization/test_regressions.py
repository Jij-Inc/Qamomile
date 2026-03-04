import io

import pytest

import qamomile.circuit as qm
from qamomile.circuit.visualization.analyzer import CircuitAnalyzer
from qamomile.circuit.visualization.style import DEFAULT_STYLE


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
