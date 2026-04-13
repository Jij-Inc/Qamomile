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


@pytest.fixture
def single_var_model() -> BinaryModel:
    """Single-variable SPIN model: 3*z0. Minimum at z0=-1 with energy -3."""
    return BinaryModel.from_ising(linear={0: 3.0}, quad={}, constant=0.0)


@pytest.fixture
def linear_only_model() -> BinaryModel:
    """SPIN model with only linear terms: 2*z0 - 3*z1. Minimum at (-1,1), energy -5."""
    return BinaryModel.from_ising(linear={0: 2.0, 1: -3.0}, quad={}, constant=0.0)


@pytest.fixture
def quad_only_model() -> BinaryModel:
    """SPIN model with only quadratic terms: -2*z0*z1. Minimum energy -2."""
    return BinaryModel.from_ising(linear={}, quad={(0, 1): -2.0}, constant=0.0)


class TestLocalSearchSpin:
    @pytest.mark.parametrize("method", ["best_improvement", "first_improvement"])
    def test_converges_to_optimum(self, spin_model, method):
        """Both methods reach the global optimum from the same starting state."""
        ls = LocalSearch(spin_model)
        result = ls.run([1, -1, -1], method=method)
        _, energy, _ = result.lowest()
        assert np.isclose(energy, -8.0)

    def test_max_iter_limits_search(self, spin_model):
        """After 1 iteration only the best single flip is applied."""
        ls = LocalSearch(spin_model)
        result = ls.run([1, 1, -1], max_iter=1, method="best_improvement")
        _, energy, _ = result.lowest()
        # From [1,1,-1] the best flip is index 0 (delta_E=-12), giving [-1,1,-1]
        assert np.isclose(energy, spin_model.calc_energy([-1, 1, -1]))

    def test_max_iter_zero_returns_initial(self, spin_model):
        """max_iter=0 returns the initial state unchanged."""
        ls = LocalSearch(spin_model)
        initial = [1, 1, -1]
        result = ls.run(initial, max_iter=0, method="best_improvement")
        _, energy, _ = result.lowest()
        assert np.isclose(energy, spin_model.calc_energy(initial))

    def test_invalid_method_raises(self, spin_model):
        """Unknown method name raises ValueError."""
        ls = LocalSearch(spin_model)
        with pytest.raises(ValueError, match="Invalid method"):
            ls.run([1, 1, 1], method="nonexistent")

    def test_invalid_initial_state_length_raises(self, spin_model):
        """Initial state with the wrong length raises ValueError."""
        ls = LocalSearch(spin_model)
        with pytest.raises(ValueError, match="initial_state|length|size"):
            ls.run([1, -1], method="best_improvement")

    @pytest.mark.parametrize("initial_state", ([1, 0, -1], [1, 2, -1], [1, -1, 3]))
    def test_invalid_initial_state_spin_domain_raises(self, spin_model, initial_state):
        """Initial state values outside the SPIN domain raise ValueError."""
        ls = LocalSearch(spin_model)
        with pytest.raises(ValueError, match="initial_state|SPIN|domain|value"):
            ls.run(initial_state, method="best_improvement")

    @pytest.mark.parametrize("max_iter", [-2, -3, -10])
    def test_invalid_max_iter_raises(self, spin_model, max_iter):
        """max_iter values less than -1 raise ValueError."""
        ls = LocalSearch(spin_model)
        with pytest.raises(ValueError, match="max_iter|-1|non-negative"):
            ls.run([1, 1, -1], max_iter=max_iter, method="best_improvement")
    @pytest.mark.parametrize("seed", range(10))
    def test_random_initial_never_increases_energy(self, spin_model, seed):
        """Local search from a random initial state never increases energy."""
        rng = np.random.default_rng(seed)
        n = spin_model.num_bits
        initial = rng.choice([-1, 1], size=n).tolist()
        ls = LocalSearch(spin_model)
        result = ls.run(initial, method="best_improvement")
        _, energy, _ = result.lowest()
        assert energy <= spin_model.calc_energy(initial) + 1e-10


class TestLocalSearchBinary:
    @pytest.mark.parametrize("method", ["best_improvement", "first_improvement"])
    def test_converges_to_optimum(self, binary_model, method):
        """Both methods reach the global optimum from a suitable starting state."""
        initial = [0, 0, 0] if method == "best_improvement" else [1, 1, 1]
        ls = LocalSearch(binary_model)
        result = ls.run(initial, method=method)
        sample, energy, _ = result.lowest()
        assert np.isclose(energy, -8.0)
        assert sample == {0: 1, 1: 0, 2: 1}

    def test_already_optimal_no_change(self, binary_model):
        """Starting at the optimum returns it unchanged."""
        ls = LocalSearch(binary_model)
        result = ls.run([1, 0, 1], method="best_improvement")
        sample, energy, _ = result.lowest()
        assert np.isclose(energy, -8.0)
        assert sample == {0: 1, 1: 0, 2: 1}

    @pytest.mark.parametrize("seed", range(10))
    def test_random_initial_never_increases_energy(self, binary_model, seed):
        """Local search from a random BINARY initial state never increases energy."""
        rng = np.random.default_rng(seed)
        n = binary_model.num_bits
        initial = rng.choice([0, 1], size=n).tolist()
        ls = LocalSearch(binary_model)
        result = ls.run(initial, method="best_improvement")
        _, energy, _ = result.lowest()
        assert energy <= binary_model.calc_energy(initial) + 1e-10


class TestInputValidation:
    def test_initial_state_wrong_length_raises(self, spin_model):
        """Initial state with wrong length raises ValueError."""
        ls = LocalSearch(spin_model)
        with pytest.raises(ValueError, match="initial_state length"):
            ls.run([1, -1], method="best_improvement")

    @pytest.mark.parametrize("state", [[1, 0, -1], [1, 2, -1], [1, -1, 3]])
    def test_invalid_spin_values_raises(self, spin_model, state):
        """SPIN initial state with values other than ±1 raises ValueError."""
        ls = LocalSearch(spin_model)
        with pytest.raises(ValueError, match="must be \\+1 or -1"):
            ls.run(state, method="best_improvement")

    @pytest.mark.parametrize("state", [[0, 2, 1], [0, -1, 1], [0, 0, 3]])
    def test_invalid_binary_values_raises(self, binary_model, state):
        """BINARY initial state with values other than 0/1 raises ValueError."""
        ls = LocalSearch(binary_model)
        with pytest.raises(ValueError, match="must be 0 or 1"):
            ls.run(state, method="best_improvement")

    @pytest.mark.parametrize("max_iter", [-2, -3, -10])
    def test_invalid_max_iter_raises(self, spin_model, max_iter):
        """max_iter less than -1 raises ValueError."""
        ls = LocalSearch(spin_model)
        with pytest.raises(ValueError, match="max_iter"):
            ls.run([1, 1, -1], max_iter=max_iter, method="best_improvement")


class TestEdgeCases:
    def test_single_variable_model(self, single_var_model):
        """Single-variable model converges to the correct minimum."""
        ls = LocalSearch(single_var_model)
        result = ls.run([1], method="best_improvement")
        _, energy, _ = result.lowest()
        assert np.isclose(energy, -3.0)

    def test_linear_only_model(self, linear_only_model):
        """Model with no quadratic terms converges correctly."""
        ls = LocalSearch(linear_only_model)
        result = ls.run([1, 1], method="best_improvement")
        _, energy, _ = result.lowest()
        assert np.isclose(energy, -5.0)

    def test_quad_only_model(self, quad_only_model):
        """Model with no linear terms converges correctly."""
        ls = LocalSearch(quad_only_model)
        result = ls.run([1, -1], method="best_improvement")
        _, energy, _ = result.lowest()
        assert np.isclose(energy, -2.0)


class TestCalcEDiff:
    def test_energy_diff_matches_actual(self, spin_model):
        """Incremental energy diff equals the brute-force difference."""
        ls = LocalSearch(spin_model)
        state = np.array([1.0, 1.0, -1.0])
        delta = ls._calc_e_diff(state, ls._neighbors, ls._linear_dict, 0)
        e_before = spin_model.calc_energy([1, 1, -1])
        e_after = spin_model.calc_energy([-1, 1, -1])
        assert np.isclose(delta, e_after - e_before)

    @pytest.mark.parametrize("flip_idx", [0, 1, 2])
    def test_energy_diff_all_indices(self, spin_model, flip_idx):
        """Energy diff is correct for every index in the model."""
        ls = LocalSearch(spin_model)
        state = np.array([-1.0, 1.0, 1.0])
        delta = ls._calc_e_diff(state, ls._neighbors, ls._linear_dict, flip_idx)
        state_list = state.astype(int).tolist()
        e_before = spin_model.calc_energy(state_list)
        flipped = state_list.copy()
        flipped[flip_idx] = -flipped[flip_idx]
        e_after = spin_model.calc_energy(flipped)
        assert np.isclose(delta, e_after - e_before)
