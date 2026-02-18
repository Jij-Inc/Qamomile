import io

import pytest

import qamomile.circuit as qm


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
