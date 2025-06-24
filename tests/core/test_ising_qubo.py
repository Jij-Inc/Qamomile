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


@pytest.mark.parametrize(
    "quad, linear, constant",
    [
        # Both quad and linear terms, max index is 6
        ({(0, 1): 2.0, (0, 2): 1.0}, {2: 5.0, 3: 2.0, 4: 1.0, 5: 1.0, 6: 1.0}, 6.0),
        # Only linear terms, max index is 3
        ({}, {0: 1.0, 1: 1.0, 2: 5.0, 3: 2.0}, 6.0),
        # Only quad terms, max index is 2
        ({(0, 1): 2.0, (0, 2): 1.0}, {}, 6.0),
        # Empty model
        ({}, {}, 6.0),
        # Single linear term, index 0
        ({}, {0: 1.0}, 6.0),
        # Only constant term, no quad or linear terms
        ({}, {}, 42.0),
    ],
)
def test_num_bits(quad, linear, constant):
    """Run IsingModel.num_bits for various IsingModel instances.

    Check if
    1. The number of bits is correctly calculated from quad and linear terms.
    """
    ising = IsingModel(quad=quad, linear=linear, constant=constant)

    # num_qubits should be the maximum index of linear and quad terms + 1.
    max_index = -1
    for i in ising.linear.keys():
        max_index = max(max_index, i)
    for i, j in ising.quad.keys():
        max_index = max(max_index, i, j)
    expected_num_bits = max_index + 1

    # 1. The number of bits is correctly calculated from quad and linear terms.
    assert ising.num_bits() == expected_num_bits


@pytest.mark.parametrize(
    "quad, linear, constant, state",
    [
        # Simple case: 2 bits
        ({(0, 1): 1.0}, {0: 2.0, 1: -1.0}, 3.0, [1, -1]),
        # Simple case: 3 bits
        ({(0, 1): 1.0, (1, 2): -2.0}, {0: 1.0, 2: 2.0}, 3.0, [1, -1, 1]),
        # Only constant
        ({}, {}, 5.0, [1, -1]),
        # Only linear
        ({}, {0: 2.0, 1: -3.0}, 0.0, [1, -1]),
        # Only quad
        ({(0, 1): 2.0}, {}, 0.0, [1, -1]),
        # Empty model
        ({}, {}, 0.0, []),
    ],
)
def test_calc_energy(quad, linear, constant, state):
    """Run IsingModel.calc_energy and check the energy calculation.

    Check if
    1. The energy is calculated correctly for given state.
    """
    # Setup: Create an Ising model with given coefficients
    ising = IsingModel(quad=quad, linear=linear, constant=constant)

    # expected_energy should be calculated as
    #    quadratic terms * corresponding state (+1/-1) + linear terms * corresponding state (+1/-1) + constant.
    expected_quad_energy = sum(
        value * state[i] * state[j] for (i, j), value in ising.quad.items()
    )
    expected_linear_energy = sum(value * state[i] for i, value in ising.linear.items())
    expected_energy = constant + expected_quad_energy + expected_linear_energy

    # 1. The energy is calculated correctly for given state
    assert ising.calc_energy(state) == expected_energy


@pytest.mark.parametrize(
    "index_map",
    [
        # Standard mapping
        {0: 10, 1: 11, 2: 12},
        # Identity mapping
        {0: 0, 1: 1, 2: 2},
        # Non-contiguous mapping
        {0: 5, 2: 7},
        # Single element
        {0: 42},
    ],
)
def test_ising2qubo_index(index_map):
    """Run IsingModel.ising2qubo_index.

    Check if
    1. The index is correctly mapped to the QUBO index.
    """
    quad = {}
    linear = {}
    constant = 0
    ising = IsingModel(quad=quad, linear=linear, constant=constant, index_map=index_map)

    for ising_index, qubo_index in index_map.items():
        # 1. The index is correctly mapped to the QUBO index.
        assert ising.ising2qubo_index(ising_index) == qubo_index


@pytest.mark.parametrize(
    "quad, linear, constant",
    [
        # Quad-dominant normalization
        ({(0, 1): 2.0, (1, 2): -4.0}, {0: 1.0, 1: 3.0}, 6.0),
        # Linear-dominant normalization
        ({(0, 1): 2.0}, {0: 4.0, 1: -5.0}, 10.0),
        # Only constant (should remain unchanged)
        ({}, {}, 5.0),
        # Only quad
        ({(0, 1): 8.0, (1, 2): -4.0}, {}, 4.0),
        # Only linear
        ({}, {0: 2.0, 1: -6.0}, 12.0),
        # Empty model
        ({}, {}, 0.0),
        # 0 coefficients (should remain unchanged)
        ({(0, 1): 0.0}, {0: 0, 1: 0}, 0.0),
    ],
)
def test_normalize_by_abs_max(quad, linear, constant):
    """Run IsingModel.normalize_by_abs_max and check normalization by max coefficient.

    Check if
    1. The quadratic terms are normalized.
    2. The linear terms are normalized,
    3. The constant term is normalized,
    """
    ising = IsingModel(quad=quad, linear=linear, constant=constant)
    ising.normalize_by_abs_max()

    # Calculate the absolute value of the maximum coefficient.
    max_abs_coeff_quad = 0
    if len(ising.quad) != 0:
        max_abs_coeff_quad = max(abs(value) for value in ising.quad.values())
    max_abs_coeff_linear = 0
    if len(ising.linear) != 0:
        max_abs_coeff_linear = max(abs(value) for value in ising.linear.values())
    max_abs_coeff = max(max_abs_coeff_quad, max_abs_coeff_linear)
    # Calculate the expected values after normalization.
    expected_quad = quad.copy()
    expected_linear = linear.copy()
    expected_constant = constant
    if max_abs_coeff != 0:
        expected_quad = {
            (i, j): value / max_abs_coeff for (i, j), value in ising.quad.items()
        }
        expected_linear = {
            i: value / max_abs_coeff for i, value in ising.linear.items()
        }
        expected_constant = ising.constant / max_abs_coeff

    # 1. The quadratic terms are normalized.
    assert ising.quad == pytest.approx(expected_quad)
    # 2. The linear terms are normalized,
    assert ising.linear == pytest.approx(expected_linear)
    # 3. The constant term is normalized,
    assert ising.constant == pytest.approx(expected_constant)


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


# <<< IsingModel <<<
