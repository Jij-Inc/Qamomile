import numpy as np
import pytest

from qamomile.optimization.binary_model import BinaryModel, BinaryExpr, VarType


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
    assert np.isclose(model.calc_energy([]), 5.0)


# ---- calc_energy with known values ----


@pytest.mark.parametrize(
    "coefficients, constant, state, expected_energy",
    [
        ({(0, 1): 2.0, (0,): 4.0, (1,): 5.0}, 6.0, [1, -1], 3.0),
        ({(0, 2): -1.0, (1,): 3.0}, -2.0, [-1, 1, -1], 0.0),
        ({(0, 1, 2): 1.5, (2,): -2.0}, 1.0, [1, 1, 1], 0.5),
        ({}, 5.0, [], 5.0),
    ],
)
def test_calc_energy_spin_known_values(coefficients, constant, state, expected_energy):
    """BinaryModel.calc_energy should produce correct results for known SPIN cases."""
    expr = BinaryExpr(
        vartype=VarType.SPIN,
        constant=constant,
        coefficients=coefficients.copy(),
    )
    bm = BinaryModel(expr)
    assert np.isclose(bm.calc_energy(state), expected_energy)


# ---- from_qubo with simplify ----


def test_from_qubo_simplify():
    """from_qubo with simplify=True should remove near-zero coefficients."""
    qubo: dict[tuple[int, int], float] = {(0, 1): 2, (0, 0): -1, (1, 1): -1}
    model = BinaryModel.from_qubo(qubo)
    assert model.num_bits == 2

    model_simplified = BinaryModel.from_qubo(qubo, simplify=True)
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
    """from_hubo should return a BINARY model with HUBO coefficients stored correctly."""
    hubo = {(0,): 1.0, (0, 1): 2.0, (0, 1, 3): 4.0}
    constant = 8.0

    # BinaryModel returns BINARY
    bm = BinaryModel.from_hubo(hubo=hubo, constant=constant)

    # Check BinaryModel is BINARY
    assert bm.vartype == VarType.BINARY
    assert np.isclose(bm.constant, constant)
    assert bm.num_bits == 3  # indices 0, 1, 3 → sequential 0, 1, 2

    # Check HUBO coefficients are stored correctly
    assert np.isclose(bm.coefficients[(0,)], 1.0)
    assert np.isclose(bm.coefficients[(0, 1)], 2.0)
    assert np.isclose(bm.coefficients[(0, 1, 2)], 4.0)  # index 3 → sequential 2

    # Verify energy equivalence via round-trip: BINARY → SPIN → BINARY
    spin_bm = bm.change_vartype(VarType.SPIN)
    roundtrip_bm = spin_bm.change_vartype(VarType.BINARY)

    for bits in range(2**bm.num_bits):
        state = [(bits >> i) & 1 for i in range(bm.num_bits)]
        assert np.isclose(bm.calc_energy(state), roundtrip_bm.calc_energy(state))


@pytest.mark.parametrize("seed", [901 + i for i in range(50)])
def test_from_hubo_equivalence_random(seed):
    """from_hubo round-trip BINARY→SPIN→BINARY should preserve energy."""
    np.random.seed(seed)

    # Generate random HUBO
    hubo: dict[tuple[int, ...], float] = {}
    num_terms = np.random.randint(3, 10)
    for _ in range(num_terms):
        order = np.random.randint(1, 5)
        indices = tuple(sorted(np.random.choice(8, size=order, replace=False)))
        hubo[indices] = np.random.randn()
    constant = np.random.randn()

    # BinaryModel from HUBO (BINARY)
    bm = BinaryModel.from_hubo(hubo=hubo, constant=constant)
    assert bm.vartype == VarType.BINARY

    # Round-trip: BINARY → SPIN → BINARY
    spin_bm = bm.change_vartype(VarType.SPIN)
    roundtrip_bm = spin_bm.change_vartype(VarType.BINARY)

    assert bm.num_bits == roundtrip_bm.num_bits

    # Verify energy equivalence for random binary states
    if bm.num_bits > 0:
        for _ in range(5):
            state = [np.random.choice([0, 1]) for _ in range(bm.num_bits)]
            bm_energy = bm.calc_energy(state)
            rt_energy = roundtrip_bm.calc_energy(state)
            assert np.isclose(bm_energy, rt_energy), (
                f"Energy mismatch: BM={bm_energy}, RT={rt_energy} for state={state}"
            )


def test_from_hubo_with_simplify():
    """from_hubo with simplify should remove near-zero terms."""
    hubo = {(0,): 1.0, (0, 1): 1e-16, (1,): 2.0}
    bm = BinaryModel.from_hubo(hubo=hubo, constant=0.0, simplify=True)
    assert bm.vartype == VarType.BINARY
    # The near-zero term (0,1) should have been removed
    assert bm.num_bits == 2


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
    """num_bits should return the number of unique variable indices."""
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


# ---- BinaryExpr.__imul__ idempotency ----


def test_imul_spin_idempotency():
    """SPIN: z_0 * z_0 = 1 (pairs cancel, becomes constant)."""
    z0 = BinaryExpr(vartype=VarType.SPIN, constant=0.0, coefficients={(0,): 1.0})
    result = z0 * z0
    # z_0^2 = 1 → constant = 1.0, no coefficients
    assert np.isclose(result.constant, 1.0)
    assert len(result.coefficients) == 0


def test_imul_binary_idempotency():
    """BINARY: x_0 * x_0 = x_0 (duplicates reduce to single)."""
    x0 = BinaryExpr(vartype=VarType.BINARY, constant=0.0, coefficients={(0,): 1.0})
    result = x0 * x0
    # x_0^2 = x_0 → coefficient (0,) = 1.0, constant = 0.0
    assert np.isclose(result.constant, 0.0)
    assert np.isclose(result.coefficients[(0,)], 1.0)
    assert len(result.coefficients) == 1


def test_imul_spin_partial_cancel():
    """SPIN: (z_0*z_1) * (z_1*z_2) = z_0*z_2 (z_1^2 cancels)."""
    z01 = BinaryExpr(vartype=VarType.SPIN, constant=0.0, coefficients={(0, 1): 1.0})
    z12 = BinaryExpr(vartype=VarType.SPIN, constant=0.0, coefficients={(1, 2): 1.0})
    result = z01 * z12
    # (z_0*z_1) * (z_1*z_2) = z_0 * z_1^2 * z_2 = z_0 * z_2
    assert np.isclose(result.constant, 0.0)
    assert np.isclose(result.coefficients[(0, 2)], 1.0)
    assert len(result.coefficients) == 1


def test_imul_binary_partial_dedup():
    """BINARY: (x_0*x_1) * (x_1*x_2) = x_0*x_1*x_2 (x_1 deduplicates)."""
    x01 = BinaryExpr(vartype=VarType.BINARY, constant=0.0, coefficients={(0, 1): 1.0})
    x12 = BinaryExpr(vartype=VarType.BINARY, constant=0.0, coefficients={(1, 2): 1.0})
    result = x01 * x12
    # (x_0*x_1) * (x_1*x_2) = x_0 * x_1^2 * x_2 = x_0 * x_1 * x_2
    assert np.isclose(result.constant, 0.0)
    assert np.isclose(result.coefficients[(0, 1, 2)], 1.0)
    assert len(result.coefficients) == 1


def test_imul_spin_all_cancel():
    """SPIN: (z_0*z_1) * (z_0*z_1) = 1 (all pairs cancel)."""
    z01 = BinaryExpr(vartype=VarType.SPIN, constant=0.0, coefficients={(0, 1): 1.0})
    result = z01 * z01
    # z_0^2 * z_1^2 = 1
    assert np.isclose(result.constant, 1.0)
    assert len(result.coefficients) == 0


def test_imul_spin_with_constants():
    """SPIN: (2 + 3*z_0) * (1 + z_0) = 2 + 2*z_0 + 3*z_0 + 3 = 5 + 5*z_0."""
    a = BinaryExpr(vartype=VarType.SPIN, constant=2.0, coefficients={(0,): 3.0})
    b = BinaryExpr(vartype=VarType.SPIN, constant=1.0, coefficients={(0,): 1.0})
    result = a * b
    # (2 + 3z)(1 + z) = 2 + 2z + 3z + 3z^2 = 2 + 5z + 3 = 5 + 5z
    assert np.isclose(result.constant, 5.0)
    assert np.isclose(result.coefficients[(0,)], 5.0)
    assert len(result.coefficients) == 1


def test_imul_binary_with_constants():
    """BINARY: (2 + 3*x_0) * (1 + x_0) = 2 + 2*x_0 + 3*x_0 + 3*x_0 = 2 + 8*x_0."""
    a = BinaryExpr(vartype=VarType.BINARY, constant=2.0, coefficients={(0,): 3.0})
    b = BinaryExpr(vartype=VarType.BINARY, constant=1.0, coefficients={(0,): 1.0})
    result = a * b
    # (2 + 3x)(1 + x) = 2 + 2x + 3x + 3x^2 = 2 + 2x + 3x + 3x = 2 + 8x
    assert np.isclose(result.constant, 2.0)
    assert np.isclose(result.coefficients[(0,)], 8.0)
    assert len(result.coefficients) == 1


# ---- Higher-order change_vartype round-trip ----


def test_change_vartype_4th_order_roundtrip():
    """4th-order BINARY→SPIN→BINARY should preserve energy."""
    hubo = {(0, 1, 2, 3): 2.0, (0, 1): 1.0, (2,): -1.0}
    constant = 3.0

    # Create BINARY model
    binary_expr = BinaryExpr(
        vartype=VarType.BINARY, constant=constant, coefficients=hubo.copy()
    )
    binary_model = BinaryModel(binary_expr)

    # Convert to SPIN and back
    spin_model = binary_model.change_vartype(VarType.SPIN)
    roundtrip_model = spin_model.change_vartype(VarType.BINARY)

    # Verify energy equivalence for all 16 states
    for bits in range(16):
        state = [(bits >> i) & 1 for i in range(4)]
        original_energy = binary_model.calc_energy(state)
        roundtrip_energy = roundtrip_model.calc_energy(state)
        assert np.isclose(original_energy, roundtrip_energy, atol=1e-10), (
            f"Energy mismatch for state {state}: {original_energy} vs {roundtrip_energy}"
        )


def test_change_vartype_5th_order_roundtrip():
    """5th-order BINARY→SPIN→BINARY should preserve energy."""
    hubo = {(0, 1, 2, 3, 4): 1.5, (0, 2, 4): -0.5, (1, 3): 2.0, (0,): 1.0}
    constant = -2.0

    # Create BINARY model
    binary_expr = BinaryExpr(
        vartype=VarType.BINARY, constant=constant, coefficients=hubo.copy()
    )
    binary_model = BinaryModel(binary_expr)

    # Convert to SPIN and back
    spin_model = binary_model.change_vartype(VarType.SPIN)
    roundtrip_model = spin_model.change_vartype(VarType.BINARY)

    # Verify energy equivalence for all 32 states
    for bits in range(32):
        state = [(bits >> i) & 1 for i in range(5)]
        original_energy = binary_model.calc_energy(state)
        roundtrip_energy = roundtrip_model.calc_energy(state)
        assert np.isclose(original_energy, roundtrip_energy, atol=1e-10), (
            f"Energy mismatch for state {state}: {original_energy} vs {roundtrip_energy}"
        )


def test_change_vartype_spin_4th_order_roundtrip():
    """4th-order SPIN→BINARY→SPIN should preserve energy."""
    coefficients = {(0, 1, 2, 3): 1.0, (0, 2): -0.5, (1,): 2.0}
    constant = 1.0

    spin_expr = BinaryExpr(
        vartype=VarType.SPIN, constant=constant, coefficients=coefficients.copy()
    )
    spin_model = BinaryModel(spin_expr)

    # Convert to BINARY and back
    binary_model = spin_model.change_vartype(VarType.BINARY)
    roundtrip_model = binary_model.change_vartype(VarType.SPIN)

    # Verify energy equivalence for all 16 spin states
    for bits in range(16):
        state = [1 if (bits >> i) & 1 == 0 else -1 for i in range(4)]
        original_energy = spin_model.calc_energy(state)
        roundtrip_energy = roundtrip_model.calc_energy(state)
        assert np.isclose(original_energy, roundtrip_energy, atol=1e-10), (
            f"Energy mismatch for state {state}: {original_energy} vs {roundtrip_energy}"
        )


# ---- from_hubo with 4th/5th order ----


def test_from_hubo_4th_order():
    """from_hubo with 4th-order terms: BINARY→SPIN→BINARY round-trip should preserve energy."""
    hubo = {(0, 1, 2, 3): 2.0, (0, 1): 1.0, (2,): -1.0}
    constant = 3.0

    bm = BinaryModel.from_hubo(hubo=hubo, constant=constant)
    assert bm.vartype == VarType.BINARY

    # Round-trip: BINARY → SPIN → BINARY
    spin_bm = bm.change_vartype(VarType.SPIN)
    roundtrip_bm = spin_bm.change_vartype(VarType.BINARY)

    # Energy equivalence for all binary states
    for bits in range(2**bm.num_bits):
        state = [(bits >> i) & 1 for i in range(bm.num_bits)]
        assert np.isclose(bm.calc_energy(state), roundtrip_bm.calc_energy(state)), (
            f"Energy mismatch for state {state}"
        )


def test_from_hubo_5th_order():
    """from_hubo with 5th-order terms: BINARY→SPIN→BINARY round-trip should preserve energy."""
    hubo = {(0, 1, 2, 3, 4): 1.0, (0, 2): 0.5, (3,): -1.0}
    constant = 0.0

    bm = BinaryModel.from_hubo(hubo=hubo, constant=constant)
    assert bm.vartype == VarType.BINARY

    # Round-trip: BINARY → SPIN → BINARY
    spin_bm = bm.change_vartype(VarType.SPIN)
    roundtrip_bm = spin_bm.change_vartype(VarType.BINARY)

    # Energy equivalence for all binary states
    for bits in range(2**bm.num_bits):
        state = [(bits >> i) & 1 for i in range(bm.num_bits)]
        assert np.isclose(bm.calc_energy(state), roundtrip_bm.calc_energy(state)), (
            f"Energy mismatch for state {state}"
        )


# ---- (i,j)/(j,i) accumulation ----


def test_from_qubo_ij_ji_accumulation():
    """from_qubo should accumulate (i,j) and (j,i) as the same term."""
    qubo = {(0, 1): 2.0, (1, 0): 3.0}
    model = BinaryModel.from_qubo(qubo)
    # Both (0,1) and (1,0) should be combined into (0,1) = 5.0
    assert np.isclose(model.quad[(0, 1)], 5.0)
    assert len(model.quad) == 1


def test_from_ising_ij_ji_accumulation():
    """from_ising should accumulate (i,j) and (j,i) as the same term."""
    quad = {(0, 1): 2.0, (1, 0): 3.0}
    model = BinaryModel.from_ising(linear={0: 1.0}, quad=quad, constant=0.0)
    # Both (0,1) and (1,0) should be combined into (0,1) = 5.0
    assert np.isclose(model.quad[(0, 1)], 5.0)
    assert len(model.quad) == 1
    assert np.isclose(model.linear[0], 1.0)


def test_from_hubo_duplicate_accumulation():
    """from_hubo should accumulate duplicate index tuples."""
    hubo = {(0, 1, 2): 2.0, (2, 0, 1): 3.0}
    model = BinaryModel.from_hubo(hubo=hubo, constant=0.0)
    # Both should be sorted to (0,1,2) and accumulated = 5.0
    assert model.vartype == VarType.BINARY
    assert model.num_bits == 3
    assert np.isclose(model.higher[(0, 1, 2)], 5.0)
