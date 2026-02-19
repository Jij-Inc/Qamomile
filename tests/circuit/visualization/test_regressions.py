import io

import pytest

import qamomile.circuit as qm
from qamomile.circuit.visualization.analyzer import CircuitAnalyzer


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
