import numpy as np
import pytest

from qamomile.optimization.binary_model.model import BinaryModel
from qamomile.optimization.post_process.local_search import LocalSearch


@pytest.fixture
def spin_model() -> BinaryModel:
    """Ising model: 2*z0*z1 + 2*z1*z2 + 4*z0.

    Minimum at z=(-1, 1, -1) with energy -8.
    """
    return BinaryModel.from_ising(
        linear={0: 4.0},
        quad={(0, 1): 2.0, (1, 2): 2.0},
        constant=0.0,
    )


@pytest.fixture
def binary_model() -> BinaryModel:
    """QUBO model: 8*x0*x1 + 8*x1*x2 - 12*x0 - 8*x1 - 4*x2 + 8.

    Equivalent to the spin_model above after variable transformation.
    Minimum at x=(1, 0, 1) with energy -8.
    """
    return BinaryModel.from_qubo(
        qubo={
            (0, 1): 8.0,
            (1, 2): 8.0,
            (0, 0): -12.0,
            (1, 1): -8.0,
            (2, 2): -4.0,
        },
        constant=8.0,
    )


class TestLocalSearchSpin:
    def test_best_improvement_converges(self, spin_model):
        ls = LocalSearch(spin_model)
        result = ls.run([1, -1, -1], method="best_improvement")
        sample, energy, _ = result.lowest()
        assert energy == -8.0

    def test_first_improvement_converges(self, spin_model):
        ls = LocalSearch(spin_model)
        result = ls.run([1, -1, -1], method="first_improvement")
        sample, energy, _ = result.lowest()
        assert energy == -8.0

    def test_max_iter_limits_search(self, spin_model):
        ls = LocalSearch(spin_model)
        result = ls.run([1, 1, -1], max_iter=1, method="best_improvement")
        sample, energy, _ = result.lowest()
        # After 1 iteration of best-improvement from [1,1,-1]:
        # flipping index 0 gives delta_E = -12, which is the best flip
        assert energy == spin_model.calc_energy([-1, 1, -1])

    def test_invalid_method_raises(self, spin_model):
        ls = LocalSearch(spin_model)
        with pytest.raises(ValueError, match="Invalid method"):
            ls.run([1, 1, 1], method="nonexistent")


class TestLocalSearchBinary:
    def test_best_improvement_converges(self, binary_model):
        ls = LocalSearch(binary_model)
        result = ls.run([0, 0, 0], method="best_improvement")
        sample, energy, _ = result.lowest()
        assert energy == -8.0
        assert sample == {0: 1, 1: 0, 2: 1}

    def test_first_improvement_converges(self, binary_model):
        ls = LocalSearch(binary_model)
        result = ls.run([1, 1, 1], method="first_improvement")
        sample, energy, _ = result.lowest()
        assert energy == -8.0

    def test_already_optimal_no_change(self, binary_model):
        ls = LocalSearch(binary_model)
        result = ls.run([1, 0, 1], method="best_improvement")
        sample, energy, _ = result.lowest()
        assert energy == -8.0
        assert sample == {0: 1, 1: 0, 2: 1}


class TestCalcEDiff:
    def test_energy_diff_matches_actual(self, spin_model):
        ls = LocalSearch(spin_model)
        state = np.array([1.0, 1.0, -1.0])
        delta = ls._calc_e_diff(state, ls._quad, ls._linear, 0)
        e_before = spin_model.calc_energy([1, 1, -1])
        e_after = spin_model.calc_energy([-1, 1, -1])
        assert np.isclose(delta, e_after - e_before)
