from qamomile.core.ising_qubo import qubo_to_ising, IsingModel
import pytest
import numpy as np


# >>> IsingModel >>>
@pytest.mark.parametrize(
    "quad, linear, constant, expected_index_map",
    [
        (  # Standard case with both quad and linear terms
            {(0, 1): 2.0, (1, 2): -4.0},
            {0: 1.0, 1: 3.0},
            6.0,
            {0: 0, 1: 1, 2: 2},
        ),
        (  # Only constant term
            {},
            {},
            42.0,
            {},
        ),
        (  # Only linear terms
            {},
            {0: 1.0, 2: 2.0},
            1.0,
            {0: 0, 2: 2},
        ),
        (  # Only quad terms
            {(0, 1): 1.0, (2, 3): 2.0},
            {},
            0.0,
            {0: 0, 1: 1, 2: 2, 3: 3},
        ),
        (  # Empty model
            {},
            {},
            0.0,
            {},
        ),
    ],
)
def test_ising_model_creation_without_index_map(
    quad, linear, constant, expected_index_map
):
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
    # 4. index_map is set correctly.
    print(ising)
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
    "quad, linear, constant, expected_num_bits",
    [
        # Both quad and linear terms, max index is 6
        ({(0, 1): 2.0, (0, 2): 1.0}, {2: 5.0, 3: 2.0, 4: 1.0, 5: 1.0, 6: 1.0}, 6.0, 7),
        # Only linear terms, max index is 3
        ({}, {0: 1.0, 1: 1.0, 2: 5.0, 3: 2.0}, 6.0, 4),
        # Only quad terms, max index is 2
        ({(0, 1): 2.0, (0, 2): 1.0}, {}, 6.0, 3),
        # Empty model
        ({}, {}, 6.0, 0),
        # Single linear term, index 0
        ({}, {0: 1.0}, 6.0, 1),
        # Only constant term, no quad or linear terms
        ({}, {}, 42.0, 0),
    ],
)
def test_num_bits(quad, linear, constant, expected_num_bits):
    """Run IsingModel.num_bits for various IsingModel instances.

    Check if
    1. The number of bits is correctly calculated from quad and linear terms.
    """
    ising = IsingModel(quad=quad, linear=linear, constant=constant)
    # 1. The number of bits is correctly calculated from quad and linear terms.
    assert ising.num_bits() == expected_num_bits


@pytest.mark.parametrize(
    "quad, linear, constant, state, expected_energy",
    [
        # Simple case: 2 bits
        (
            {(0, 1): 1.0},
            {0: 2.0, 1: -1.0},
            3.0,
            [1, -1],
            1.0 * 1 * (-1) + 2.0 * 1 + (-1.0) * (-1) + 3.0,
        ),
        # Simple case: 3 bits
        (
            {(0, 1): 1.0, (1, 2): -2.0},
            {0: 1.0, 2: 2.0},
            3.0,
            [1, -1, 1],
            1.0 * 1 * -1 + (-2.0) * -1 * 1 + 1.0 * 1 + 2.0 * 1 + 3.0,
        ),
        # Only constant
        ({}, {}, 5.0, [1, -1], 5.0),
        # Only linear
        ({}, {0: 2.0, 1: -3.0}, 0.0, [1, -1], 2.0 * 1 + (-3.0) * (-1)),
        # Only quad
        ({(0, 1): 2.0}, {}, 0.0, [1, -1], 2.0 * 1 * -1),
        # Empty model
        ({}, {}, 0.0, [], 0.0),
    ],
)
def test_calc_energy(quad, linear, constant, state, expected_energy):
    """Run IsingModel.calc_energy and check the energy calculation.

    Check if
    1. The energy is calculated correctly for given state.
    """
    # Setup: Create an Ising model with given coefficients
    ising = IsingModel(quad=quad, linear=linear, constant=constant)
    # 1. The energy is calculated correctly for given state
    assert ising.calc_energy(state) == expected_energy


@pytest.mark.parametrize(
    "index_map, index, expected_qubo_index",
    [
        # Standard mapping
        ({0: 10, 1: 11, 2: 12}, 0, 10),
        ({0: 10, 1: 11, 2: 12}, 1, 11),
        ({0: 10, 1: 11, 2: 12}, 2, 12),
        # Identity mapping
        ({0: 0, 1: 1, 2: 2}, 0, 0),
        ({0: 0, 1: 1, 2: 2}, 1, 1),
        ({0: 0, 1: 1, 2: 2}, 2, 2),
        # Non-contiguous mapping
        ({0: 5, 2: 7}, 0, 5),
        ({0: 5, 2: 7}, 2, 7),
        # Single element
        ({0: 42}, 0, 42),
    ],
)
def test_ising2qubo_index(index_map, index, expected_qubo_index):
    """Run IsingModel.ising2qubo_index and check index mapping.

    Check if
    1. The index is correctly mapped to the QUBO index.
    """
    quad = {}
    linear = {}
    constant = 0
    ising = IsingModel(quad=quad, linear=linear, constant=constant, index_map=index_map)
    # 1. The index is correctly mapped to the QUBO index.
    assert ising.ising2qubo_index(index) == expected_qubo_index


@pytest.mark.parametrize(
    "quad, linear, constant, expected_quad, expected_linear, expected_constant",
    [
        # Quad-dominant normalization
        (
            {(0, 1): 2.0, (1, 2): -4.0},
            {0: 1.0, 1: 3.0},
            6.0,
            {(0, 1): 0.5, (1, 2): -1.0},
            {0: 0.25, 1: 0.75},
            1.5,
        ),
        # Linear-dominant normalization
        (
            {(0, 1): 2.0},
            {0: 4.0, 1: -5.0},
            10.0,
            {(0, 1): 0.4},
            {0: 0.8, 1: -1.0},
            2.0,
        ),
        # Only constant (should remain unchanged)
        (
            {},
            {},
            5.0,
            {},
            {},
            5.0,
        ),
        # Only quad
        (
            {(0, 1): 8.0, (1, 2): -4.0},
            {},
            4.0,
            {(0, 1): 1.0, (1, 2): -0.5},
            {},
            0.5,
        ),
        # Only linear
        (
            {},
            {0: 2.0, 1: -6.0},
            12.0,
            {},
            {0: 0.3333333333333333, 1: -1.0},
            2.0,
        ),
        # Empty model
        (
            {},
            {},
            0.0,
            {},
            {},
            0.0,
        ),
        # 0 coefficients for quad (should remain unchanged)
        (
            {(0, 1): 0.0},
            {},
            0.0,
            {(0, 1): 0.0},
            {},
            0.0,
        ),
        # 0 coefficients for linear (should remain unchanged)
        (
            {},
            {0: 0.0},
            0.0,
            {},
            {0: 0.0},
            0.0,
        ),
    ],
)
def test_normalize_by_abs_max(
    quad, linear, constant, expected_quad, expected_linear, expected_constant
):
    """Run IsingModel.normalize_by_abs_max and check normalization by max coefficient.

    Check if
    1. The constant term is normalized,
    2. The linear terms are normalized,
    3. The quadratic terms are normalized.
    """
    ising = IsingModel(quad=quad, linear=linear, constant=constant)
    ising.normalize_by_abs_max()
    # 1. The constant term is normalized,
    assert ising.quad == pytest.approx(expected_quad)
    # 2. The linear terms are normalized,
    assert ising.linear == pytest.approx(expected_linear)
    # 3. The quadratic terms are normalized.
    assert ising.constant == pytest.approx(expected_constant)


@pytest.mark.parametrize(
    "quad, linear, constant",
    [
        # Quad and linear terms (quad-dominant)
        ({(0, 1): 2.0, (1, 2): -2.0}, {0: 1.0, 1: -1.0}, 6.0),
        # Quad and linear terms (linear-dominant)
        ({(0, 1): 2}, {0: 3.0, 1: -4.0}, 7.0),
        # Only constant (should remain unchanged)
        ({}, {}, 5.0),
        # Only quad
        ({(0, 1): 8.0, (1, 2): -4.0}, {}, 4.0),
        # Only linear
        ({}, {0: 2.0, 1: -6.0}, 12.0),
        # Empty model
        ({}, {}, 0.0),
        # 0 coefficients for quad (should remain unchanged)
        ({(0, 1): 0.0}, {}, 0.0),
        # 0 coefficients for linear (should remain unchanged)
        ({}, {0: 0.0}, 0.0),
    ],
)
def test_normalize_by_rms(quad, linear, constant):
    """Run IsingModel.normalize_by_rms and check normalization by RMS value.

    Check if
    1. The quadratic terms are divided by the RMS value,
    2. The linear terms are divided by the RMS value,
    3. The constant term is normalized by the RMS value.
    """
    ising = IsingModel(quad=quad, linear=linear, constant=constant)
    ising.normalize_by_rms()

    # Calcuate the RMS value.
    rms_quad = 0
    rms_linear = 0
    if len(ising.quad) != 0:
        rms_quad = np.sqrt(
            sum(value**2 for value in ising.quad.values()) / len(ising.quad)
        )
    if len(ising.linear) != 0:
        rms_linear = np.sqrt(
            sum(value**2 for value in ising.linear.values()) / len(ising.linear)
        )
    rms_value = np.sqrt(rms_quad**2 + rms_linear**2)
    # Calcuate the expected values after normalization.
    expected_quad = quad.copy()
    expected_linear = linear.copy()
    expected_constant = constant
    if rms_value != 0:
        expected_quad = {
            (i, j): value / rms_value for (i, j), value in ising.quad.items()
        }
        expected_linear = {i: value / rms_value for i, value in ising.linear.items()}
        expected_constant = ising.constant / rms_value

    # 1. The constant term is normalized,
    assert ising.quad == pytest.approx(expected_quad)
    # 2. The linear terms are normalized,
    assert ising.linear == pytest.approx(expected_linear)
    # 3. The quadratic terms are normalized.
    assert ising.constant == pytest.approx(expected_constant)


# <<< IsingModel <<<


# >>> qubo_to_ising >>>
def test_onehot_conversion():
    """Run qubo_to_ising with a simple QUBO.

    Check if
    1. The constant term is correctly calculated,
    2. The linear terms are correctly converted to Ising model,
    3. The quadratic terms are correctly converted to Ising model,
    4. The simplified version has the correct constant term,
    5. The simplified version removes linear terms,
    6. The simplified version has the correct quadratic terms.
    """
    qubo: dict[tuple[int, int], float] = {(0, 1): 2, (0, 0): -1, (1, 1): -1}
    ising = qubo_to_ising(qubo)
    # 1. The constant term is correctly calculated,
    assert ising.constant == -0.5
    # 2. The linear terms are correctly converted to Ising model,
    assert ising.linear == {0: 0, 1: 0}
    # 3. The quadratic terms are correctly converted to Ising model,
    assert ising.quad == {(0, 1): 0.5}

    ising = qubo_to_ising(qubo, simplify=True)
    # 4. The simplified version has the correct constant term,
    assert ising.constant == -0.5
    # 5. The simplified version removes linear terms,
    assert ising.linear == {}
    # 6. The simplified version has the correct quadratic terms.
    assert ising.quad == {(0, 1): 0.5}


# <<< qubo_to_ising <<<

# >>> calc_qubo_energy >>>
# <<< calc_qubo_energy <<<
