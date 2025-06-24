import numpy as np
import pytest

from qamomile.core.ising_qubo import qubo_to_ising, IsingModel


# >>> qubo_to_ising >>>
@pytest.mark.parametrize(
    "qubo",
    [
        # Standard QUBO with off-diagonal and diagonal terms
        {(0, 1): 2, (0, 0): -1, (1, 1): -1},
        # Only diagonal terms
        {(0, 0): 2.0, (1, 1): 3.0},
        # Only off-diagonal terms
        {(0, 1): 4.0, (1, 2): -2.0},
        # Empty QUBO
        {},
        # Zero coefficients
        {(0, 0): 0.0, (1, 1): 0.0, (0, 1): 0.0},
        # Single variable QUBO
        {(0, 0): 3.0},
        # Mixed zero and nonzero
        {(0, 1): 0.0, (1, 2): 2.0, (2, 2): 0.0},
    ],
)
def test_qubo_to_ising_without_simplify(qubo):
    """Run qubo_to_ising with various QUBOs and check Ising conversion (no simplify).

    Check if
    1. The returned value is an IsingModel,
    2. The constant term is correctly calculated,
    3. The linear terms are correctly converted to Ising model,
    4. The quadratic terms are correctly converted to Ising model.
    """
    ising = qubo_to_ising(qubo, simplify=False)
    # 1. The returned value is an IsingModel,
    assert isinstance(ising, IsingModel)

    # Calculate the expected constant, linear, and quadratic terms.
    #    The quadratic terms, which are of i == j, are divided by 4.0,
    expected_quad = {(i, j): value / 4.0 for (i, j), value in qubo.items() if i != j}
    #    The constant term is computed as the sum of the double of the diagonal terms divided by 4.0
    #    and the off-diagonal terms divided by 4.0.
    expected_constant = sum(
        2 * (value / 4.0) if i == j else (value / 4.0) for (i, j), value in qubo.items()
    )
    #    The linear terms are computed as the sum of the off-diagonal terms divided by 4.0
    expected_linear = {}
    for (i, j), value in qubo.items():
        expected_linear[i] = expected_linear.get(i, 0) - value / 4.0
        expected_linear[j] = expected_linear.get(j, 0) - value / 4.0

    # 2. The constant term is correctly calculated,
    assert ising.constant == pytest.approx(expected_constant)
    # 3. The linear terms are correctly converted to Ising model,
    assert ising.linear == pytest.approx(expected_linear)
    # 4. The quadratic terms are correctly converted to Ising model,
    assert ising.quad == pytest.approx(expected_quad)


@pytest.mark.parametrize(
    "qubo",
    [
        # Standard QUBO with off-diagonal and diagonal terms
        {(0, 1): 2, (0, 0): -1, (1, 1): 0, (0, 2): 0, (1, 3): 1},
        # Only diagonal terms
        {(0, 0): 2.0, (1, 1): 0},
        # Only off-diagonal terms
        {(0, 1): 4.0, (1, 2): 0},
        # Empty QUBO
        {},
        # Zero coefficients
        {(0, 0): 0.0, (1, 1): 0.0, (0, 1): 0.0},
        # Single variable QUBO
        {(0, 0): 0},
        # Cancel h[2] out but J[0, 2] remains (for the coverage rate)
        {(0, 0): 4.0, (1, 2): 4.0, (3, 2): -4.0, (0, 1): 12.0},
    ],
)
def test_qubo_to_ising_with_simplify(qubo):
    """Run qubo_to_ising with various QUBOs and check Ising conversion (with simplify).

    Check if
    1. The returned value is an IsingModel,
    2. The constant term is correctly calculated,
    3. Each linear term is at least in the expected linear term,
    4. Each quadratic term is at least in the expected quadratic term,
    5. The length of the index map is correct.
    """
    ising = qubo_to_ising(qubo, simplify=True)
    # 1. The returned value is an IsingModel,
    assert isinstance(ising, IsingModel)

    # Calculate the expected constant, linear, quadratic terms, and number of elements of idnex_map.
    #    The quadratic terms, which are of i == j, are divided by 4.0,
    expected_quad = {
        (i, j): value / 4.0 for (i, j), value in qubo.items() if i != j and value != 0
    }
    #    The constant term is computed as the sum of the double of the diagonal terms divided by 4.0
    #    and the off-diagonal terms divided by 4.0.
    expected_constant = sum(
        2 * (value / 4.0) if i == j else (value / 4.0)
        for (i, j), value in qubo.items()
        if value != 0
    )
    #    The linear terms are computed as the sum of the off-diagonal terms divided by 4.0
    expected_linear = {}
    for (i, j), value in qubo.items():
        if value != 0:
            expected_linear[i] = expected_linear.get(i, 0) - value / 4.0
            expected_linear[j] = expected_linear.get(j, 0) - value / 4.0
    expected_linear = {i: v for i, v in expected_linear.items() if v != 0}
    #    The valid indices are the keys of the expected linear terms and the keys of the expected quadratic terms.
    expected_valid_indices = set([i for i in expected_linear.keys()])
    for i, j in expected_quad.keys():
        expected_valid_indices.add(i)
        expected_valid_indices.add(j)
    expected_num_index_map = len(expected_valid_indices)

    # 2. The constant term is correctly calculated,
    assert ising.constant == pytest.approx(expected_constant)
    # 3. Each linear term is at least in the expected linear term,
    for value in ising.linear.values():
        assert any(
            value == expected_value for expected_value in expected_linear.values()
        )
    # 4. Each quadratic term is at least in the expected quadratic term,
    for value in ising.quad.values():
        assert any(value == expected_value for expected_value in expected_quad.values())
    # 5. The length of the index map is correct.
    assert len(ising.index_map) == expected_num_index_map


# <<< qubo_to_ising <<<


# >>> IsingModel >>>
@pytest.mark.parametrize(
    "quad, linear, constant",
    [
        # Standard case with both quad and linear terms
        (
            {(0, 1): 2.0, (1, 2): -4.0},
            {0: 1.0, 1: 3.0},
            6.0,
        ),
        # Only constant term
        ({}, {}, 42.0),
        # Only linear terms
        ({}, {0: 1.0, 2: 2.0}, 1.0),
        # Only quad terms
        ({(0, 1): 1.0, (2, 3): 2.0}, {}, 0.0),
        # Empty model
        ({}, {}, 0.0),
    ],
)
def test_ising_model_creation_without_index_map(quad, linear, constant):
    """Create an IsingModel instance and verify its attributes.

    Check if
    1. The constant term is set correctly,
    2. The linear terms are set correctly,
    3. The quadratic terms are set correctly,
    4. index_map is set correctly.
    """
    # Setup: Create an Ising model with given coefficients
    ising = IsingModel(quad=quad, linear=linear, constant=constant)
    # 1. The constant term is set correctly
    assert ising.constant == constant
    # 2. The linear terms are set correctly
    assert ising.linear == linear
    # 3. The quadratic terms are set correctly
    assert ising.quad == quad

    # index_map should be set as same as the given linear and quad terms.
    expected_index_map = {}
    for i in linear.keys():
        expected_index_map[i] = i
    for i, j in quad.keys():
        expected_index_map[i] = i
        expected_index_map[j] = j
    # 4. index_map is set correctly.
    assert ising.index_map == expected_index_map


@pytest.mark.parametrize(
    "quad, linear, constant, initial_index_map",
    [
        # Standard case with both quad and linear terms and custom index_map
        (
            {(0, 1): 2.0, (1, 2): -4.0},
            {0: 1.0, 1: 3.0},
            6.0,
            {0: 10, 1: 11, 2: 12},
        ),
        # Only constant term and empty index_map
        (
            {},
            {},
            42.0,
            {},
        ),
        # Only linear terms and custom index_map
        (
            {},
            {0: 1.0, 2: 2.0},
            1.0,
            {0: 5, 2: 7},
        ),
        # Only quad terms and custom index_map
        (
            {(0, 1): 1.0, (2, 3): 2.0},
            {},
            0.0,
            {0: 2, 1: 3, 2: 4, 3: 5},
        ),
        # Empty model and empty index_map
        (
            {},
            {},
            0.0,
            {},
        ),
    ],
)
def test_ising_model_creation_with_index_map(quad, linear, constant, initial_index_map):
    """Create an IsingModel instance and verify its attributes.

    Check if
    1. The constant term is set correctly,
    2. The linear terms are set correctly,
    3. The quadratic terms are set correctly,
    4. index_map is set correctly.
    """
    # Setup: Create an Ising model with given coefficients
    ising = IsingModel(
        quad=quad, linear=linear, constant=constant, index_map=initial_index_map
    )
    # 1. The constant term is set correctly
    assert ising.constant == constant
    # 2. The linear terms are set correctly
    assert ising.linear == linear
    # 3. The quadratic terms are set correctly
    assert ising.quad == quad
    # 4. index_map is set correctly.
    assert ising.index_map == initial_index_map


# <<< IsingModel <<<


def test_num_bits():
    ising = IsingModel(
        {(0, 1): 2.0, (0, 2): 1.0},
        {2: 5.0, 3: 2.0, 4: 1.0, 5: 1.0, 6: 1.0},
        6.0,
    )
    assert ising.num_bits() == 7

    ising = IsingModel({}, {0: 1.0, 1: 1.0, 2: 5.0, 3: 2.0}, 6.0)
    assert ising.num_bits() == 4

    ising = IsingModel(
        {(0, 1): 2.0, (0, 2): 1.0},
        {},
        6.0,
    )
    assert ising.num_bits() == 3

    ising = IsingModel(
        {},
        {},
        6.0,
    )
    assert ising.num_bits() == 0

    ising = IsingModel(
        {},
        {0: 1.0},
        6.0,
    )
    assert ising.num_bits() == 1


def test_normalize_by_abs_max():
    # Setup: Create an Ising model with known coefficients
    ising = IsingModel(
        quad={(0, 1): 2.0, (1, 2): -4.0},  # max quad coeff is 4.0
        linear={0: 1.0, 1: 3.0},  # max linear coeff is 3.0
        constant=6.0,
    )

    # Execute normalization
    ising.normalize_by_abs_max()

    # Max coefficient was 4.0, so everything should be divided by 4.0
    assert ising.quad[(0, 1)] == 0.5  # 2.0 / 4.0
    assert ising.quad[(1, 2)] == -1.0  # -4.0 / 4.0
    assert ising.linear[0] == 0.25  # 1.0 / 4.0
    assert ising.linear[1] == 0.75  # 3.0 / 4.0
    assert ising.constant == 1.5  # 6.0 / 4.0


def test_normalize_by_abs_max_linear_dominant():
    # Setup: Create an Ising model where linear term has the max coefficient
    ising = IsingModel(
        quad={(0, 1): 2.0},  # max quad coeff is 2.0
        linear={0: 4.0, 1: -5.0},  # max linear coeff is 5.0
        constant=10.0,
    )

    # Execute normalization
    ising.normalize_by_abs_max()

    # Max coefficient was 5.0, so everything should be divided by 5.0
    assert ising.quad[(0, 1)] == 0.4  # 2.0 / 5.0
    assert ising.linear[0] == 0.8  # 4.0 / 5.0
    assert ising.linear[1] == -1.0  # -5.0 / 5.0
    assert ising.constant == 2.0  # 10.0 / 5.0


def test_normalize_by_abs_max_empty():
    # Setup: Create an Ising model with no coefficients
    ising = IsingModel(quad={}, linear={}, constant=0.0)

    # Empty model should not be normalized
    ising.normalize_by_abs_max()

    # Values should remain unchanged
    assert ising.constant == 0.0
    assert len(ising.quad) == 0
    assert len(ising.linear) == 0


def test_normalize_by_abs_max_only_constant():
    # Setup: Create an Ising model with only constant term
    ising = IsingModel(quad={}, linear={}, constant=5.0)

    # Model with only constant should not be normalized
    ising.normalize_by_abs_max()

    # Constant should remain unchanged
    assert ising.constant == 5.0
    assert len(ising.quad) == 0
    assert len(ising.linear) == 0


def test_normalize_by_rms():
    # Setup: Create an Ising model with known coefficients
    ising = IsingModel(
        quad={(0, 1): 2.0, (1, 2): -2.0},  # sum(w_ij^2) = 8, E2 = 2
        linear={0: 1.0, 1: -1.0},  # sum(w_i^2) = 2, E1 = 2
        constant=6.0,
    )

    # Calculate expected normalization factor
    # sqrt(8/2 + 2/2) = sqrt(4 + 1) = sqrt(5)
    expected_factor = np.sqrt(5)

    # Execute normalization
    ising.normalize_by_rms()

    # Check normalized values with numpy's tolerance
    np.testing.assert_allclose(ising.quad[(0, 1)], 2.0 / expected_factor)
    np.testing.assert_allclose(ising.quad[(1, 2)], -2.0 / expected_factor)
    np.testing.assert_allclose(ising.linear[0], 1.0 / expected_factor)
    np.testing.assert_allclose(ising.linear[1], -1.0 / expected_factor)
    np.testing.assert_allclose(ising.constant, 6.0 / expected_factor)
