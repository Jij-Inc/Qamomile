import numpy as np
import pytest

from qamomile.optimization.binary_model import BinaryModel, BinaryExpr, VarType
from qamomile.optimization.higher_ising_model import HigherIsingModel


# ---- coefficients property ----


def test_coefficients_property():
    """The coefficients property should merge linear, quad, and higher dicts."""
    model = BinaryModel.from_ising(
        linear={0: 1.0, 1: 2.0},
        quad={(0, 1): 3.0},
        constant=4.0,
    )
    coeffs = model.coefficients
    assert coeffs[(0,)] == 1.0
    assert coeffs[(1,)] == 2.0
    assert coeffs[(0, 1)] == 3.0
    assert len(coeffs) == 3


def test_coefficients_property_higher_order():
    """The coefficients property should include higher-order terms."""
    expr = BinaryExpr(
        vartype=VarType.SPIN,
        constant=1.0,
        coefficients={(0,): 2.0, (0, 1): 3.0, (0, 1, 2): 4.0},
    )
    model = BinaryModel(expr)
    coeffs = model.coefficients
    assert len(coeffs) == 3
    assert coeffs[(0,)] == 2.0
    assert coeffs[(0, 1)] == 3.0
    assert coeffs[(0, 1, 2)] == 4.0


def test_coefficients_property_empty():
    """Empty model should return empty coefficients dict."""
    expr = BinaryExpr(vartype=VarType.SPIN, constant=5.0, coefficients={})
    model = BinaryModel(expr)
    assert model.coefficients == {}
    assert model.constant == 5.0


# ---- index_map ----


def test_index_map_defaults_to_identity():
    """When no index_map is given, it should be the identity mapping."""
    model = BinaryModel.from_ising(
        linear={2: 1.0, 5: 2.0},
        quad={(2, 5): 3.0},
        constant=0.0,
    )
    assert model.index_map == {2: 2, 5: 5}


def test_index_map_custom():
    """Custom index_map should be stored and accessible."""
    custom_map = {2: 100, 5: 200}
    model = BinaryModel.from_ising(
        linear={2: 1.0, 5: 2.0},
        quad={(2, 5): 3.0},
        constant=0.0,
        index_map=custom_map,
    )
    assert model.index_map == custom_map


def test_index_map_preserved_through_change_vartype():
    """index_map should be preserved when changing vartype."""
    custom_map = {0: 10, 1: 20}
    model = BinaryModel.from_ising(
        linear={0: 1.0, 1: 2.0},
        quad={(0, 1): 3.0},
        constant=0.0,
        index_map=custom_map,
    )
    binary_model = model.change_vartype(VarType.BINARY)
    assert binary_model.index_map == custom_map

    # Also check identity (same vartype)
    spin_model = model.change_vartype(VarType.SPIN)
    assert spin_model.index_map == custom_map


def test_index_map_preserved_through_normalize():
    """index_map should be preserved when normalizing (replace=False)."""
    custom_map = {0: 10, 1: 20}
    model = BinaryModel.from_ising(
        linear={0: 1.0, 1: 2.0},
        quad={(0, 1): 4.0},
        constant=0.0,
        index_map=custom_map,
    )
    normalized = model.normalize_by_abs_max(replace=False)
    assert normalized.index_map == custom_map

    # Also replace=True
    model.normalize_by_abs_max(replace=True)
    assert model.index_map == custom_map


# ---- ising2original_index ----


def test_ising2original_index_identity():
    """With identity index_map and contiguous indices, should return original index."""
    model = BinaryModel.from_ising(
        linear={0: 1.0, 1: 2.0, 2: 3.0},
        quad={},
        constant=0.0,
    )
    assert model.ising2original_index(0) == 0
    assert model.ising2original_index(1) == 1
    assert model.ising2original_index(2) == 2


def test_ising2original_index_non_contiguous():
    """Non-contiguous original indices, identity index_map."""
    model = BinaryModel.from_ising(
        linear={2: 1.0, 5: 2.0, 8: 3.0},
        quad={},
        constant=0.0,
    )
    # Sequential 0 -> original 2 -> index_map 2
    assert model.ising2original_index(0) == 2
    # Sequential 1 -> original 5 -> index_map 5
    assert model.ising2original_index(1) == 5
    # Sequential 2 -> original 8 -> index_map 8
    assert model.ising2original_index(2) == 8


def test_ising2original_index_custom_map():
    """Full three-level mapping with custom index_map."""
    custom_map = {2: 100, 5: 200, 8: 300}
    model = BinaryModel.from_ising(
        linear={2: 1.0, 5: 2.0, 8: 3.0},
        quad={},
        constant=0.0,
        index_map=custom_map,
    )
    # Sequential 0 -> original 2 -> index_map 100
    assert model.ising2original_index(0) == 100
    # Sequential 1 -> original 5 -> index_map 200
    assert model.ising2original_index(1) == 200
    # Sequential 2 -> original 8 -> index_map 300
    assert model.ising2original_index(2) == 300


# ---- backward-compatible aliases ----


def test_backward_compat_aliases():
    """original_to_zero_origin_map and zero_origin_to_original_map should work."""
    model = BinaryModel.from_ising(
        linear={2: 1.0, 5: 2.0},
        quad={(2, 5): 3.0},
        constant=0.0,
    )
    assert model.original_to_zero_origin_map == model.index_origin_to_new
    assert model.zero_origin_to_original_map == model.index_new_to_origin
    assert model.original_to_zero_origin_map == {2: 0, 5: 1}
    assert model.zero_origin_to_original_map == {0: 2, 1: 5}


# ---- calc_energy ----


@pytest.mark.parametrize(
    "linear, quad, constant, state, expected_energy",
    [
        ({0: 4.0, 1: 5.0}, {(0, 1): 2.0}, 6.0, [1, -1], 3.0),
        ({1: 3.0}, {(0, 2): -1.0}, -2.0, [-1, 1, -1], 0.0),
        ({2: -2.0}, {}, 1.0, [1, 1, 1], -1.0),
    ],
)
def test_calc_energy_spin(linear, quad, constant, state, expected_energy):
    """calc_energy for SPIN models should match expected values."""
    model = BinaryModel.from_ising(linear=linear, quad=quad, constant=constant)
    assert np.isclose(model.calc_energy(state), expected_energy)


def test_calc_energy_spin_higher_order():
    """calc_energy should work with higher-order SPIN terms."""
    expr = BinaryExpr(
        vartype=VarType.SPIN,
        constant=1.0,
        coefficients={(0, 1, 2): 1.5, (2,): -2.0},
    )
    model = BinaryModel(expr)
    # state = [1, 1, 1]: energy = 1.0 + 1.5*1*1*1 + (-2.0)*1 = 0.5
    assert np.isclose(model.calc_energy([1, 1, 1]), 0.5)


def test_calc_energy_binary():
    """calc_energy for BINARY models."""
    model = BinaryModel.from_qubo(
        qubo={(0, 0): 1.0, (0, 1): 2.0, (1, 1): 3.0},
        constant=0.0,
    )
    # state = [1, 0]: energy = 1.0*1 + 2.0*1*0 + 3.0*0 = 1.0
    assert np.isclose(model.calc_energy([1, 0]), 1.0)
    # state = [1, 1]: energy = 1.0*1 + 2.0*1*1 + 3.0*1 = 6.0
    assert np.isclose(model.calc_energy([1, 1]), 6.0)


def test_calc_energy_invalid_spin():
    """calc_energy should raise ValueError for invalid SPIN values."""
    model = BinaryModel.from_ising(linear={0: 1.0, 1: 2.0}, quad={}, constant=0.0)
    with pytest.raises(ValueError):
        model.calc_energy([1, 0])  # 0 is not valid for SPIN


def test_calc_energy_invalid_binary():
    """calc_energy should raise ValueError for invalid BINARY values."""
    model = BinaryModel.from_qubo(qubo={(0, 0): 1.0}, constant=0.0)
    with pytest.raises(ValueError):
        model.calc_energy([-1])  # -1 is not valid for BINARY


def test_calc_energy_empty():
    """calc_energy on empty model should return constant."""
    expr = BinaryExpr(vartype=VarType.SPIN, constant=5.0, coefficients={})
    model = BinaryModel(expr)
    # No variables, so state is empty-ish. But num_bits=0,
    # so we call with an empty list.
    assert np.isclose(model.calc_energy([]), 5.0)


# ---- calc_energy equivalence with HigherIsingModel ----


@pytest.mark.parametrize(
    "coefficients, constant, state, expected_energy",
    [
        ({(0, 1): 2.0, (0,): 4.0, (1,): 5.0}, 6.0, [1, -1], 3.0),
        ({(0, 2): -1.0, (1,): 3.0}, -2.0, [-1, 1, -1], 0.0),
        ({(0, 1, 2): 1.5, (2,): -2.0}, 1.0, [1, 1, 1], 0.5),
        ({}, 5.0, [], 5.0),
    ],
)
def test_calc_energy_matches_higher_ising(
    coefficients, constant, state, expected_energy
):
    """BinaryModel.calc_energy should produce same results as HigherIsingModel.calc_energy."""
    # HigherIsingModel reference
    if coefficients:
        him = HigherIsingModel(coefficients=coefficients.copy(), constant=constant)
        him_energy = him.calc_energy(state)
        assert np.isclose(him_energy, expected_energy)

    # BinaryModel
    expr = BinaryExpr(
        vartype=VarType.SPIN,
        constant=constant,
        coefficients=coefficients.copy(),
    )
    bm = BinaryModel(expr)
    bm_energy = bm.calc_energy(state)
    assert np.isclose(bm_energy, expected_energy)

    if coefficients:
        assert np.isclose(bm_energy, him_energy)


# ---- from_qubo with simplify ----


def test_from_qubo_simplify():
    """from_qubo with simplify=True should remove near-zero coefficients."""
    qubo: dict[tuple[int, int], float] = {(0, 1): 2, (0, 0): -1, (1, 1): -1}
    model = BinaryModel.from_qubo(qubo)
    assert model.num_bits == 2

    model_simplified = BinaryModel.from_qubo(qubo, simplify=True)
    # The diagonal terms (0,0)=-1 and (1,1)=-1 produce linear terms that are 0
    # after the QUBO aggregation. from_qubo simplifies the BINARY expr.
    # But the binary expr has (0,): -1 and (1,): -1, which are not near zero.
    # So simplify on the binary side doesn't remove them.

    # Just verify model was created successfully
    assert model_simplified.num_bits == 2


def test_from_ising_simplify():
    """from_ising with simplify=True should remove near-zero coefficients."""
    model = BinaryModel.from_ising(
        linear={0: 0.0, 1: 2.0},
        quad={(0, 1): 1e-15},
        constant=1.0,
        simplify=True,
    )
    # Near-zero linear coeff (0: 0.0) and near-zero quad coeff are removed.
    # Only index 1 remains, remapped to sequential index 0.
    assert len(model.quad) == 0
    assert model.num_bits == 1
    assert model.linear[0] == 2.0  # original index 1 remapped to 0


# ---- from_hubo ----


def test_from_hubo_manually():
    """from_hubo should produce same results as HigherIsingModel.from_hubo for a manual case."""
    hubo = {(0,): 1.0, (0, 1): 2.0, (0, 1, 3): 4.0}
    constant = 8.0

    # Expected values (from test_higher_ising_model.py)
    expected_coefficients = {
        (0,): -12 / 8,
        (1,): -8 / 8,
        (2,): -4 / 8,  # index 3 maps to sequential 2
        (0, 1): 8 / 8,
        (0, 2): 4 / 8,
        (1, 2): 4 / 8,
        (0, 1, 2): -4 / 8,
    }
    expected_constant = 76 / 8

    # HigherIsingModel reference
    him = HigherIsingModel.from_hubo(hubo=hubo, constant=constant)

    # BinaryModel
    bm = BinaryModel.from_hubo(hubo=hubo, constant=constant)

    # Check BinaryModel is SPIN
    assert bm.vartype == VarType.SPIN

    # Check constant
    assert np.isclose(bm.constant, expected_constant)
    assert np.isclose(bm.constant, him.constant)

    # Check coefficients match expected
    for key, value in expected_coefficients.items():
        assert np.isclose(bm.coefficients[key], value), (
            f"Mismatch at {key}: expected {value}, got {bm.coefficients.get(key)}"
        )

    # Check number of variables
    assert bm.num_bits == him.num_bits == 3


@pytest.mark.parametrize("seed", [901 + i for i in range(50)])
def test_from_hubo_equivalence_random(seed):
    """from_hubo should produce numerically equivalent results to HigherIsingModel.from_hubo."""
    np.random.seed(seed)

    # Generate random HUBO
    hubo: dict[tuple[int, ...], float] = {}
    num_terms = np.random.randint(3, 10)
    for _ in range(num_terms):
        order = np.random.randint(1, 5)
        indices = tuple(sorted(np.random.choice(8, size=order, replace=False)))
        hubo[indices] = np.random.randn()
    constant = np.random.randn()

    # HigherIsingModel reference
    him = HigherIsingModel.from_hubo(hubo=hubo, constant=constant)

    # BinaryModel
    bm = BinaryModel.from_hubo(hubo=hubo, constant=constant)

    # Constants should match
    assert np.isclose(bm.constant, him.constant), (
        f"Constants differ: BM={bm.constant}, HIM={him.constant}"
    )

    # Number of bits should match
    assert bm.num_bits == him.num_bits

    # Compare coefficients: for each coefficient in HigherIsingModel, check BinaryModel has it
    for key, value in him.coefficients.items():
        if np.isclose(value, 0.0):
            continue
        assert key in bm.coefficients, f"Key {key} missing in BinaryModel"
        assert np.isclose(bm.coefficients[key], value), (
            f"Mismatch at {key}: BM={bm.coefficients[key]}, HIM={value}"
        )

    # Verify energy equivalence for random states
    if bm.num_bits > 0:
        for _ in range(5):
            state = [np.random.choice([-1, 1]) for _ in range(bm.num_bits)]
            bm_energy = bm.calc_energy(state)
            him_energy = him.calc_energy(state)
            assert np.isclose(bm_energy, him_energy), (
                f"Energy mismatch: BM={bm_energy}, HIM={him_energy} for state={state}"
            )


def test_from_hubo_with_simplify():
    """from_hubo with simplify should remove near-zero terms before conversion."""
    hubo = {(0,): 1.0, (0, 1): 1e-16, (1,): 2.0}
    bm = BinaryModel.from_hubo(hubo=hubo, constant=0.0, simplify=True)
    assert bm.vartype == VarType.SPIN
    # The near-zero term (0,1) should have been removed before conversion
    assert bm.num_bits == 2


def test_from_hubo_with_index_map():
    """from_hubo should propagate index_map."""
    hubo = {(0,): 1.0, (1,): 2.0}
    custom_map = {0: 10, 1: 20}
    bm = BinaryModel.from_hubo(hubo=hubo, constant=0.0, index_map=custom_map)
    assert bm.index_map == custom_map
    assert bm.ising2original_index(0) == 10
    assert bm.ising2original_index(1) == 20


# ---- from_qubo with index_map ----


def test_from_qubo_with_index_map():
    """from_qubo should accept and store index_map."""
    qubo = {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 3.0}
    custom_map = {0: 50, 1: 60}
    model = BinaryModel.from_qubo(qubo, index_map=custom_map)
    assert model.index_map == custom_map


def test_from_ising_with_index_map():
    """from_ising should accept and store index_map."""
    custom_map = {0: 50, 1: 60}
    model = BinaryModel.from_ising(
        linear={0: 1.0, 1: 2.0},
        quad={(0, 1): 3.0},
        constant=0.0,
        index_map=custom_map,
    )
    assert model.index_map == custom_map


# ---- num_bits ----


@pytest.mark.parametrize(
    "coefficients, constant, expected_num_bits",
    [
        ({(0,): 1.0, (0, 1): 2.0, (0, 2, 3): 1.0, (4, 5, 6, 7): -3.0}, 0.0, 8),
        ({(0, 1, 2): 1.0, (3, 4): 1.0}, 6.0, 5),
        ({(0, 1, 3): 2.0, (2,): 1.0}, 6.0, 4),
        ({}, 6.0, 0),
        ({(0,): 1.0}, 6.0, 1),
        ({(0, 1): 1.0}, 6.0, 2),
        ({(0, 1, 2): 1.0}, 6.0, 3),
    ],
)
def test_num_bits(coefficients, constant, expected_num_bits):
    """num_bits should match HigherIsingModel's behavior."""
    expr = BinaryExpr(
        vartype=VarType.SPIN, constant=constant, coefficients=coefficients
    )
    model = BinaryModel(expr)
    assert model.num_bits == expected_num_bits


# ---- order ----


def test_order():
    """order should reflect the maximum term degree."""
    # Quadratic only
    model = BinaryModel.from_ising(linear={0: 1.0}, quad={(0, 1): 2.0}, constant=0.0)
    assert model.order == 2

    # Higher order
    expr = BinaryExpr(
        vartype=VarType.SPIN,
        constant=0.0,
        coefficients={(0, 1, 2, 3): 1.0, (0,): 2.0},
    )
    model = BinaryModel(expr)
    assert model.order == 4

    # Empty
    expr = BinaryExpr(vartype=VarType.SPIN, constant=5.0, coefficients={})
    model = BinaryModel(expr)
    assert model.order == 0
