from qamomile.core.ising_qubo import IsingModel
import numpy as np


def test_onehot_conversion():
    qubo: dict[tuple[int, int], float] = {(0, 1): 2, (0, 0): -1, (1, 1): -1}
    ising = IsingModel.from_qubo(qubo)
    assert ising.constant == -0.5
    assert ising.linear == {0: 0, 1: 0}
    assert ising.quad == {(0, 1): 0.5}

    ising = IsingModel.from_qubo(qubo, simplify=True)
    assert ising.constant == -0.5
    assert ising.linear == {}
    assert ising.quad == {(0, 1): 0.5}


def test_num_bits():
    ising = IsingModel(
        {(0, 1): 2.0, (0, 2): 1.0},
        {2: 5.0, 3: 2.0, 4: 1.0, 5: 1.0, 6: 1.0},
        6.0,
    )
    assert ising.num_bits == 7

    ising = IsingModel({}, {0: 1.0, 1: 1.0, 2: 5.0, 3: 2.0}, 6.0)
    assert ising.num_bits == 4

    ising = IsingModel(
        {(0, 1): 2.0, (0, 2): 1.0},
        {},
        6.0,
    )
    assert ising.num_bits == 3

    ising = IsingModel(
        {},
        {},
        6.0,
    )
    assert ising.num_bits == 0

    ising = IsingModel(
        {},
        {0: 1.0},
        6.0,
    )
    assert ising.num_bits == 1


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
