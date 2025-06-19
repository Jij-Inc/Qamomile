from qamomile.core.ising_qubo import qubo_to_ising, IsingModel
import numpy as np


# >>> IsingModel >>>
def test_num_bits():
    """Run IsingModel.num_bits for various IsingModel instances.

    Check if
    1. The number of bits is correctly calculated from quad and linear terms,
    2. The function returns 0 for empty models,
    3. The function returns 1 for a single linear term.
    """
    ising = IsingModel(
        {(0, 1): 2.0, (0, 2): 1.0},
        {2: 5.0, 3: 2.0, 4: 1.0, 5: 1.0, 6: 1.0},
        6.0,
    )
    # 1. The number of bits is correctly calculated from quad and linear terms
    assert ising.num_bits() == 7

    ising = IsingModel({}, {0: 1.0, 1: 1.0, 2: 5.0, 3: 2.0}, 6.0)
    assert ising.num_bits() == 4

    ising = IsingModel(
        {(0, 1): 2.0, (0, 2): 1.0},
        {},
        6.0,
    )
    assert ising.num_bits() == 3

    # 2. The function returns 0 for empty models
    ising = IsingModel(
        {},
        {},
        6.0,
    )
    assert ising.num_bits() == 0

    # 3. The function returns 1 for a single linear term
    ising = IsingModel(
        {},
        {0: 1.0},
        6.0,
    )
    assert ising.num_bits() == 1


def test_normalize_by_abs_max():
    """Run IsingModel.normalize_by_abs_max and check normalization by max coefficient.

    Check if
    1. All coefficients are divided by the maximum absolute value among quad and linear terms,
    2. The constant term is also normalized,
    3. The function works for both quad-dominant and linear-dominant cases.
    """
    # Setup: Create an Ising model with known coefficients
    ising = IsingModel(
        quad={(0, 1): 2.0, (1, 2): -4.0},  # max quad coeff is 4.0
        linear={0: 1.0, 1: 3.0},  # max linear coeff is 3.0
        constant=6.0,
    )

    # Execute normalization
    ising.normalize_by_abs_max()

    # 1. All coefficients are divided by the maximum absolute value among quad and linear terms
    assert ising.quad[(0, 1)] == 0.5  # 2.0 / 4.0
    assert ising.quad[(1, 2)] == -1.0  # -4.0 / 4.0
    assert ising.linear[0] == 0.25  # 1.0 / 4.0
    assert ising.linear[1] == 0.75  # 3.0 / 4.0
    # 2. The constant term is also normalized
    assert ising.constant == 1.5  # 6.0 / 4.0


def test_normalize_by_abs_max_linear_dominant():
    """Run IsingModel.normalize_by_abs_max for a linear-dominant model.

    Check if
    1. All coefficients are divided by the maximum absolute value among quad and linear terms,
    2. The constant term is also normalized.
    """
    # Setup: Create an Ising model where linear term has the max coefficient
    ising = IsingModel(
        quad={(0, 1): 2.0},  # max quad coeff is 2.0
        linear={0: 4.0, 1: -5.0},  # max linear coeff is 5.0
        constant=10.0,
    )

    # Execute normalization
    ising.normalize_by_abs_max()

    # 1. All coefficients are divided by the maximum absolute value among quad and linear terms
    assert ising.quad[(0, 1)] == 0.4  # 2.0 / 5.0
    assert ising.linear[0] == 0.8  # 4.0 / 5.0
    assert ising.linear[1] == -1.0  # -5.0 / 5.0
    # 2. The constant term is also normalized
    assert ising.constant == 2.0  # 10.0 / 5.0


def test_normalize_by_abs_max_empty():
    """Run IsingModel.normalize_by_abs_max for an empty model.

    Check if
    1. The function does not change the model if there are no coefficients.
    """
    # Setup: Create an Ising model with no coefficients
    ising = IsingModel(quad={}, linear={}, constant=0.0)

    # Empty model should not be normalized
    ising.normalize_by_abs_max()

    # 1. The function does not change the model if there are no coefficients.
    assert ising.constant == 0.0
    assert len(ising.quad) == 0
    assert len(ising.linear) == 0


def test_normalize_by_abs_max_only_constant():
    """Run IsingModel.normalize_by_abs_max for a model with only a constant term.

    Check if
    1. The function does not change the model if there are no quad or linear terms.
    """
    # Setup: Create an Ising model with only constant term
    ising = IsingModel(quad={}, linear={}, constant=5.0)

    # Model with only constant should not be normalized
    ising.normalize_by_abs_max()

    # 1. The function does not change the model if there are no quad or linear terms.
    assert ising.constant == 5.0
    assert len(ising.quad) == 0
    assert len(ising.linear) == 0


def test_normalize_by_rms():
    """Run IsingModel.normalize_by_rms and check normalization by RMS value.

    Check if
    1. All coefficients are divided by the RMS value of quad and linear terms,
    2. The constant term is also normalized.
    """
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

    # 1. All coefficients are divided by the RMS value of quad and linear terms
    np.testing.assert_allclose(ising.quad[(0, 1)], 2.0 / expected_factor)
    np.testing.assert_allclose(ising.quad[(1, 2)], -2.0 / expected_factor)
    np.testing.assert_allclose(ising.linear[0], 1.0 / expected_factor)
    np.testing.assert_allclose(ising.linear[1], -1.0 / expected_factor)
    # 2. The constant term is also normalized
    np.testing.assert_allclose(ising.constant, 6.0 / expected_factor)


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
