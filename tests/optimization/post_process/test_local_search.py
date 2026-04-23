from itertools import combinations

import numpy as np
import ommx.v1
import pytest

from qamomile.optimization.binary_model.expr import BinaryExpr, VarType
from qamomile.optimization.binary_model.model import BinaryModel
from qamomile.optimization.binary_model.sampleset import BinarySampleSet
from qamomile.optimization.post_process.local_search import LocalSearch

# ---------------------------------------------------------------------------
# Random-model helpers (for tests that should not depend on a specific
# Hamiltonian structure — only on the invariant "energy never increases").
# ---------------------------------------------------------------------------


def _random_spin_model(seed: int, n: int = 4) -> BinaryModel:
    """Return a quadratic SPIN model with seeded N(0,1) linear/quad coefficients."""
    rng = np.random.default_rng(seed)
    linear = {i: float(rng.normal()) for i in range(n)}
    quad = {(i, j): float(rng.normal()) for i in range(n) for j in range(i + 1, n)}
    return BinaryModel.from_ising(linear=linear, quad=quad, constant=0.0)


def _random_binary_model(seed: int, n: int = 4) -> BinaryModel:
    """Return a QUBO (BINARY) model with seeded N(0,1) coefficients on all (i,j)."""
    rng = np.random.default_rng(seed)
    qubo = {(i, j): float(rng.normal()) for i in range(n) for j in range(i, n)}
    return BinaryModel.from_qubo(qubo=qubo, constant=0.0)


def _random_hubo_model(seed: int, n: int = 4, max_order: int = 3) -> BinaryModel:
    """Return a BINARY HUBO with seeded N(0,1) coefficients up to *max_order*."""
    rng = np.random.default_rng(seed)
    hubo: dict[tuple[int, ...], float] = {}
    for order in range(1, max_order + 1):
        for term in combinations(range(n), order):
            hubo[term] = float(rng.normal())
    return BinaryModel.from_hubo(hubo=hubo, constant=0.0)


# ---------------------------------------------------------------------------
# Deterministic fixtures — each docstring states the objective and the
# unique minimum (or enumerates the value landscape) so assertions can be
# justified without recomputing.
# ---------------------------------------------------------------------------


@pytest.fixture
def spin_model() -> BinaryModel:
    """Ising model: 2*z0*z1 + 2*z1*z2 + 4*z0.

    Brute-forced energies over 8 states:
      (+,+,+)=8;   (+,+,-)=4;   (+,-,+)=0;   (+,-,-)=4;
      (-,+,+)=-4;  (-,+,-)=-8 (unique min); (-,-,+)=-4; (-,-,-)=0.
    """
    return BinaryModel.from_ising(
        linear={0: 4.0},
        quad={(0, 1): 2.0, (1, 2): 2.0},
        constant=0.0,
    )


@pytest.fixture
def binary_model() -> BinaryModel:
    """QUBO model: 8*x0*x1 + 8*x1*x2 - 12*x0 - 8*x1 - 4*x2 + 8.

    Equivalent to ``spin_model`` under ``x = (1 - s) / 2`` mapping.
    Brute-forced energies over 8 states:
      (0,0,0)=8; (0,0,1)=4;  (0,1,0)=0;  (0,1,1)=4;
      (1,0,0)=-4; (1,0,1)=-8 (unique min); (1,1,0)=-4; (1,1,1)=0.
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
    """SPIN model with only quadratic terms: -2*z0*z1. Two minima at (+1,+1) and (-1,-1), energy -2."""
    return BinaryModel.from_ising(linear={}, quad={(0, 1): -2.0}, constant=0.0)


class TestLocalSearchSpin:
    @pytest.mark.parametrize("method", ["best", "first"])
    def test_converges_to_optimum(self, spin_model, method):
        """Both methods reach the unique global optimum z=(-1, 1, -1), E=-8.

        Derivation (``2 z0 z1 + 2 z1 z2 + 4 z0``, start [1, -1, -1]):
          1. ΔE at [1,-1,-1]: i=0 gives -4, i=1 gives 0, i=2 gives -4.
             Best/first both pick i=0 (tie broken by index order) → [-1,-1,-1].
          2. ΔE at [-1,-1,-1]: i=0 gives +4, i=1 gives -8, i=2 gives -4.
             Best/first both pick i=1 → [-1, 1,-1], E=-8.
          3. ΔE at [-1, 1,-1]: every index has ΔE > 0, so no further flips.
        """
        ls = LocalSearch(spin_model)
        result = ls.run([1, -1, -1], method=method)
        sample, energy, _ = result.lowest()
        assert np.isclose(energy, -8.0)
        assert sample == {0: -1, 1: 1, 2: -1}

    def test_max_iter_limits_search(self, spin_model):
        """One iteration applies the best flip then stops.

        From [1,1,-1]: ΔE at i=0 is -12, at i=1 is 0, at i=2 is +4.
        best-improvement picks i=0 → [-1,1,-1] with E=-8. max_iter=1 halts
        the loop before any further improvement check.
        """
        ls = LocalSearch(spin_model)
        result = ls.run([1, 1, -1], max_iter=1, method="best")
        sample, energy, _ = result.lowest()
        assert np.isclose(energy, spin_model.calc_energy([-1, 1, -1]))
        assert sample == {0: -1, 1: 1, 2: -1}

    def test_max_iter_zero_returns_initial(self, spin_model):
        """max_iter=0 skips the search loop entirely, returning initial state as-is."""
        ls = LocalSearch(spin_model)
        initial = [1, 1, -1]
        result = ls.run(initial, max_iter=0, method="best")
        sample, energy, _ = result.lowest()
        assert np.isclose(energy, spin_model.calc_energy(initial))
        assert sample == {0: 1, 1: 1, 2: -1}

    def test_invalid_method_raises(self, spin_model):
        """Unknown method name raises ValueError listing valid choices."""
        ls = LocalSearch(spin_model)
        with pytest.raises(ValueError, match="Invalid method"):
            ls.run([1, 1, 1], method="nonexistent")

    @pytest.mark.parametrize("method", ["best", "first"])
    @pytest.mark.parametrize("model_seed", range(5))
    @pytest.mark.parametrize("init_seed", range(5))
    def test_random_initial_never_increases_energy(self, model_seed, init_seed, method):
        """Energy-monotonic invariant holds across random models AND random inits.

        For 5 seeded random SPIN models × 5 seeded random initial states × 2
        methods, the returned energy must never exceed the starting energy —
        only strictly improving flips are accepted.
        """
        model = _random_spin_model(model_seed, n=4)
        rng = np.random.default_rng(init_seed)
        initial = rng.choice([-1, 1], size=model.num_bits).tolist()
        ls = LocalSearch(model)
        result = ls.run(initial, method=method)
        _, energy, _ = result.lowest()
        assert energy <= model.calc_energy(initial) + 1e-10


class TestLocalSearchBinary:
    @pytest.mark.parametrize("method", ["best", "first"])
    def test_converges_to_optimum(self, binary_model, method):
        """Both methods reach the unique optimum x=(1, 0, 1), E=-8.

        ``binary_model`` is the ``x = (1 - s) / 2`` image of ``spin_model``, so
        the local-search trajectory in SPIN space mirrors the Ising case:
          - ``method="best"`` starts at x=[0,0,0] (spin [+1,+1,+1]); greedy
            descent in spin-space converges to spin [-1,1,-1], i.e. x=[1,0,1].
          - ``method="first"`` starts at x=[1,1,1] (spin [-1,-1,-1]); the
            first improving flip (i=1 in spin-space, ΔE=-8) reaches the same
            optimum.
        """
        initial = [0, 0, 0] if method == "best" else [1, 1, 1]
        ls = LocalSearch(binary_model)
        result = ls.run(initial, method=method)
        sample, energy, _ = result.lowest()
        assert np.isclose(energy, -8.0)
        assert sample == {0: 1, 1: 0, 2: 1}

    def test_already_optimal_no_change(self, binary_model):
        """Starting at the unique optimum x=(1,0,1), no flip has ΔE<0, so state is unchanged."""
        ls = LocalSearch(binary_model)
        result = ls.run([1, 0, 1], method="best")
        sample, energy, _ = result.lowest()
        assert np.isclose(energy, -8.0)
        assert sample == {0: 1, 1: 0, 2: 1}

    @pytest.mark.parametrize("method", ["best", "first"])
    @pytest.mark.parametrize("model_seed", range(5))
    @pytest.mark.parametrize("init_seed", range(5))
    def test_random_initial_never_increases_energy(self, model_seed, init_seed, method):
        """Energy-monotonic invariant holds across random QUBOs and random inits."""
        model = _random_binary_model(model_seed, n=4)
        rng = np.random.default_rng(init_seed)
        initial = rng.choice([0, 1], size=model.num_bits).tolist()
        ls = LocalSearch(model)
        result = ls.run(initial, method=method)
        _, energy, _ = result.lowest()
        assert energy <= model.calc_energy(initial) + 1e-10


class TestInputValidation:
    def test_initial_state_wrong_length_raises(self, spin_model):
        """Initial state with wrong length raises ValueError."""
        ls = LocalSearch(spin_model)
        with pytest.raises(ValueError, match="initial_state length"):
            ls.run([1, -1], method="best")

    @pytest.mark.parametrize("state", [[1, 0, -1], [1, 2, -1], [1, -1, 3]])
    def test_invalid_spin_values_raises(self, spin_model, state):
        """SPIN initial state with values other than ±1 raises ValueError."""
        ls = LocalSearch(spin_model)
        with pytest.raises(ValueError, match="must be \\+1 or -1"):
            ls.run(state, method="best")

    @pytest.mark.parametrize("state", [[0, 2, 1], [0, -1, 1], [0, 0, 3]])
    def test_invalid_binary_values_raises(self, binary_model, state):
        """BINARY initial state with values other than 0/1 raises ValueError."""
        ls = LocalSearch(binary_model)
        with pytest.raises(ValueError, match="must be 0 or 1"):
            ls.run(state, method="best")

    @pytest.mark.parametrize("max_iter", [-2, -3, -10])
    def test_invalid_max_iter_raises(self, spin_model, max_iter):
        """max_iter less than -1 raises ValueError."""
        ls = LocalSearch(spin_model)
        with pytest.raises(ValueError, match="max_iter"):
            ls.run([1, 1, -1], max_iter=max_iter, method="best")


class TestEdgeCases:
    def test_single_variable_model(self, single_var_model):
        """Single-variable model 3*z0 converges to z0=-1, E=-3.

        From [1]: ΔE(0) = -2·1·3 = -6, so flip to [-1], E=-3.
        From [-1]: ΔE(0) = +6, no further flip — converged.
        """
        ls = LocalSearch(single_var_model)
        result = ls.run([1], method="best")
        sample, energy, _ = result.lowest()
        assert np.isclose(energy, -3.0)
        assert sample == {0: -1}

    def test_linear_only_model(self, linear_only_model):
        """Purely linear model 2*z0 - 3*z1 converges to (-1, 1), E=-5.

        From [1,1]: ΔE(0) = -2·1·2 = -4, ΔE(1) = -2·1·(-3) = +6.
        Best-improvement flips i=0 → [-1, 1], E = 2·(-1) - 3·1 = -5.
        From [-1, 1]: both ΔE > 0, so no further flip.
        """
        ls = LocalSearch(linear_only_model)
        result = ls.run([1, 1], method="best")
        sample, energy, _ = result.lowest()
        assert np.isclose(energy, -5.0)
        assert sample == {0: -1, 1: 1}

    def test_quad_only_model(self, quad_only_model):
        """Purely quadratic model -2*z0*z1 converges to one local minimum, E=-2.

        From [1,-1]: ΔE(0) = -2·1·(-2·-1) = -4, ΔE(1) = -2·-1·(-2·1) = -4.
        Best-improvement ties at -4; ``argmin`` breaks ties on the smaller
        index, so i=0 flips first → [-1,-1], E = -2·(-1)·(-1) = -2.
        From [-1,-1]: both ΔE = +4, so the (+1,+1) minimum is unreachable
        from this path — local search stops at (-1,-1).
        """
        ls = LocalSearch(quad_only_model)
        result = ls.run([1, -1], method="best")
        sample, energy, _ = result.lowest()
        assert np.isclose(energy, -2.0)
        assert sample == {0: -1, 1: -1}


class TestLocalSearchHubo:
    """Higher-order (HUBO) support: linear/quad/higher terms in one Hamiltonian."""

    @pytest.fixture
    def cubic_spin_model(self) -> BinaryModel:
        """Cubic SPIN model: s_0*s_1*s_2 + 2*s_0.

        Brute force over 8 states:
          (+,+,+)=3;   (+,+,-)=1;   (+,-,+)=1;   (+,-,-)=3;
          (-,+,+)=-3 (min); (-,+,-)=-1; (-,-,+)=-1; (-,-,-)=-3 (min).
        """
        return BinaryModel(
            BinaryExpr(
                vartype=VarType.SPIN,
                constant=0.0,
                coefficients={(0, 1, 2): 1.0, (0,): 2.0},
            )
        )

    @pytest.mark.parametrize("method", ["best", "first"])
    def test_cubic_spin_converges_to_min(self, cubic_spin_model, method):
        """HUBO cubic SPIN model from [1,1,1] lands in the (-,+,+) local min, E=-3.

        _terms built by LocalSearch for ``s_0*s_1*s_2 + 2*s_0``:
          _terms[0] = [((1,2), 1.0), ((), 2.0)]  — cubic + linear
          _terms[1] = [((0,2), 1.0)]
          _terms[2] = [((0,1), 1.0)]
        From [1,1,1]:
          1. ΔE(0) = -2·1·(1·1 + 2) = -6 (improves most);
             ΔE(1) = -2·1·(1·1) = -2; ΔE(2) = -2·1·(1·1) = -2.
             Best/first both flip i=0 → [-1, 1, 1], E = -1 + (-2) = -3.
          2. From [-1, 1, 1]: ΔE(0) = -2·(-1)·(1·1 + 2) = +6;
             ΔE(1) = -2·1·((-1)·1) = +2; ΔE(2) = -2·1·((-1)·1) = +2.
             All non-negative, so search halts.
        """
        ls = LocalSearch(cubic_spin_model)
        result = ls.run([1, 1, 1], method=method)
        sample, energy, _ = result.lowest()
        assert np.isclose(energy, -3.0)
        assert sample == {0: -1, 1: 1, 2: 1}

    @pytest.mark.parametrize("method", ["best", "first"])
    @pytest.mark.parametrize("model_seed", range(5))
    @pytest.mark.parametrize("init_seed", range(5))
    def test_random_hubo_never_increases_energy(self, model_seed, init_seed, method):
        """Energy-monotonic invariant for random BINARY HUBO up to cubic terms."""
        model = _random_hubo_model(model_seed, n=4, max_order=3)
        rng = np.random.default_rng(init_seed)
        initial = rng.choice([0, 1], size=model.num_bits).tolist()
        ls = LocalSearch(model)
        result = ls.run(initial, method=method)
        _, energy, _ = result.lowest()
        assert energy <= model.calc_energy(initial) + 1e-10

    @pytest.mark.parametrize("seed", range(5))
    def test_cubic_energy_diff_matches_brute_force(self, cubic_spin_model, seed):
        """Incremental ΔE on cubic terms matches full-energy brute force."""
        rng = np.random.default_rng(seed)
        state = rng.choice([-1.0, 1.0], size=3)
        ls = LocalSearch(cubic_spin_model)
        for flip_idx in range(3):
            delta = ls._calc_e_diff(state, ls._terms, flip_idx)
            before = cubic_spin_model.calc_energy(state.astype(int).tolist())
            flipped = state.copy()
            flipped[flip_idx] = -flipped[flip_idx]
            after = cubic_spin_model.calc_energy(flipped.astype(int).tolist())
            assert np.isclose(delta, after - before)


class TestOmmxInput:
    """LocalSearch accepts ommx.v1.Instance, mirroring other converters."""

    @pytest.fixture
    def quadratic_ommx(self) -> ommx.v1.Instance:
        """Minimize ``-x0 - x1 + 3·x0·x1`` over binary {x0, x1}.

        Brute force:
          (0,0)=0; (0,1)=-1 (min); (1,0)=-1 (min); (1,1)=1.
        """
        x0 = ommx.v1.DecisionVariable.binary(id=0, name="x0")
        x1 = ommx.v1.DecisionVariable.binary(id=1, name="x1")
        return ommx.v1.Instance.from_components(
            decision_variables=[x0, x1],
            objective=-x0 - x1 + 3 * x0 * x1,
            constraints=[],
            sense=ommx.v1.Instance.MINIMIZE,
        )

    @pytest.fixture
    def cubic_ommx(self) -> ommx.v1.Instance:
        """Minimize ``2·x0·x1·x2 - x0 - x1 - x2`` over binary {x0, x1, x2}.

        Brute-forced energies (minimization sense):
          (0,0,0)=0;  (1,0,0)=-1; (0,1,0)=-1; (0,0,1)=-1;
          (1,1,0)=-2; (1,0,1)=-2; (0,1,1)=-2;
          (1,1,1)=2 + (-3) = -1.
        Minimum is -2 at any pair being 1 and the third 0.
        """
        x0 = ommx.v1.DecisionVariable.binary(id=0)
        x1 = ommx.v1.DecisionVariable.binary(id=1)
        x2 = ommx.v1.DecisionVariable.binary(id=2)
        return ommx.v1.Instance.from_components(
            decision_variables=[x0, x1, x2],
            objective=2 * x0 * x1 * x2 - x0 - x1 - x2,
            constraints=[],
            sense=ommx.v1.Instance.MINIMIZE,
        )

    def test_ommx_input_returns_solution(self, quadratic_ommx):
        """ommx input yields ommx.v1.Solution output (not BinarySampleSet).

        Starting from x=(0,0) (spin [+1,+1]) on ``-x0 - x1 + 3*x0*x1``:
          1. to_hubo + SPIN conversion gives linear={0:-1/4, 1:-1/4},
             quad={(0,1): 3/4}, constant=-1/4. ΔE(0)=ΔE(1)=-1 at spin
             [+1,+1]; argmin ties on i=0, so spin_0 flips → [-1, +1].
          2. At spin [-1, +1]: ΔE(0)=+1, ΔE(1)=+2 — no improvement.
        Spin [-1,+1] maps back to BINARY {0: 1, 1: 0}, objective = -1.
        """
        result = LocalSearch(quadratic_ommx).run([0, 0], method="best")
        assert isinstance(result, ommx.v1.Solution)
        assert np.isclose(result.objective, -1.0)
        assert result.feasible
        assert dict(result.state.entries) == {0: 1.0, 1: 0.0}

    def test_binary_model_input_still_returns_sampleset(self, spin_model):
        """BinaryModel input keeps the pre-ommx return type (BinarySampleSet) and same solution."""
        result = LocalSearch(spin_model).run([1, -1, -1], method="best")
        assert isinstance(result, BinarySampleSet)
        sample, energy, _ = result.lowest()
        assert np.isclose(energy, -8.0)
        assert sample == {0: -1, 1: 1, 2: -1}

    def test_quadratic_ommx_matches_binary_model(self, quadratic_ommx):
        """ommx quadratic solution agrees with the equivalent BinaryModel."""
        bm = BinaryModel.from_qubo(
            qubo={(0, 0): -1.0, (1, 1): -1.0, (0, 1): 3.0}, constant=0.0
        )
        result_bm = LocalSearch(bm).run([0, 0], method="best")
        result_ommx = LocalSearch(quadratic_ommx).run([0, 0], method="best")

        sample_bm, energy_bm, _ = result_bm.lowest()
        assert np.isclose(energy_bm, -1.0)
        assert np.isclose(result_ommx.objective, energy_bm)
        assert dict(result_ommx.state.entries) == {
            i: float(v) for i, v in sample_bm.items()
        }

    @pytest.mark.parametrize("method", ["best", "first"])
    def test_cubic_ommx_preserves_higher_order(self, cubic_ommx, method):
        """HUBO terms in ommx input survive the handoff via to_hubo.

        From x=(0,0,0): any single flip to 1 yields E=-1 (ΔE=-1, strictly
        improving); a second flip to any other 1-bit yields E=-2 (ΔE=-1);
        a third flip to (1,1,1) yields E=-1 (ΔE=+1, rejected). So local
        search halts at two 1-bits with E=-2.
        """
        result = LocalSearch(cubic_ommx).run([0, 0, 0], method=method)
        assert isinstance(result, ommx.v1.Solution)
        assert np.isclose(result.objective, -2.0)
        assert result.feasible

    def test_invalid_model_type_raises(self):
        """Passing anything other than BinaryModel / ommx.v1.Instance raises TypeError."""
        with pytest.raises(TypeError, match="ommx.v1.Instance or BinaryModel"):
            LocalSearch("not a model")  # type: ignore[arg-type]


class TestConstantOnlyModel:
    @pytest.mark.parametrize("method", ["best", "first"])
    def test_constant_only_model_no_crash(self, method):
        """n == 0 constant-only model returns unchanged state without crashing.

        With no variables, ``_best_improvement`` / ``_first_improvement``
        early-return without calling ``np.argmin([])``; ``run`` then wraps
        the empty SPIN state into a single sample with ``E = constant``.
        """
        model = BinaryModel.from_ising(linear={}, quad={}, constant=5.0)
        ls = LocalSearch(model)
        result = ls.run([], method=method)
        sample, energy, _ = result.lowest()
        assert np.isclose(energy, 5.0)
        assert sample == {}


class TestCalcEDiff:
    def test_energy_diff_matches_actual(self, spin_model):
        """Single-index incremental ΔE equals the brute-force difference.

        Hand-derived: at [1,1,-1] flipping i=0 takes E(1,1,-1)=4 to
        E(-1,1,-1)=-8, so ΔE must be -12 — matching ``_calc_e_diff``.
        """
        ls = LocalSearch(spin_model)
        state = np.array([1.0, 1.0, -1.0])
        delta = ls._calc_e_diff(state, ls._terms, 0)
        e_before = spin_model.calc_energy([1, 1, -1])
        e_after = spin_model.calc_energy([-1, 1, -1])
        assert np.isclose(delta, e_after - e_before)
        assert np.isclose(delta, -12.0)

    @pytest.mark.parametrize("flip_idx", [0, 1, 2])
    def test_energy_diff_all_indices(self, spin_model, flip_idx):
        """Energy diff is correct for every index in the model (brute-force check)."""
        ls = LocalSearch(spin_model)
        state = np.array([-1.0, 1.0, 1.0])
        delta = ls._calc_e_diff(state, ls._terms, flip_idx)
        state_list = state.astype(int).tolist()
        e_before = spin_model.calc_energy(state_list)
        flipped = state_list.copy()
        flipped[flip_idx] = -flipped[flip_idx]
        e_after = spin_model.calc_energy(flipped)
        assert np.isclose(delta, e_after - e_before)
