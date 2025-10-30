import numpy
import pytest

from qamomile.core.higher_ising_model import HigherIsingModel


@pytest.mark.parametrize("seed", [901 + i for i in range(100)])
def test_creation_with_defaults(seed):
    """Create a HigherIsingModel.

    Check if
    - the coefficients are the same as the input,
    - the constant term is the same as the input,
    - the keys of the index_map are the unique values of the integers of the coefficients' key.
    - the values of the index_map are the positions of the sorted unique integers.
    """
    # Fix the numpy.random.seed for reproducibility.

    # Create the random coefficients.

    # Create the random constant term.

    # Create the HigherIsingModel.

    # - the coefficients are the same as the input,
    # - the constant term is the same as the input,
    # - the keys of the index_map are the unique values of the integers of the coefficients' key.
    # - the values of the index_map are the positions of the sorted unique integers.


@pytest.mark.parametrize("seed", [901 + i for i in range(100)])
def test_creation_with_index_map(seed):
    """Create a HigherIsingModel by giving index_map.

    Check if
    - the index_map is the same as the input.
    """
    # Fix the numpy.random.seed for reproducibility.

    # Create the random coefficients.

    # Create the random constant term.

    # Create a random index_map.

    # Create the HigherIsingModel.

    # - the index_map is the same as the input.


@pytest.mark.parametrize(
    "coefficients, constant, expected_num_bits",
    [
        [
            {
                (0,): 1.2,
                (0, 1): 2.0,
                (0, 2, 3): 1.0,
                (4, 5, 6, 7): -3.0,
            },
            0.0,
            8,
        ],
        [{(0, 1, 2): 1.0, (3, 4): 1.0}, 6.0, 5],
        [{(0, 1, 3): 2.0, (2,): 1.0}, 6.0, 4],
        [{}, 6.0, 0],
        [{(0,): 1.0}, 6.0, 1],
        [{(0, 1): 1.0}, 6.0, 2],
        [{(0, 1, 2): 1.0}, 6.0, 3],
    ],
)
def test_num_bits_manually(coefficients, constant, expected_num_bits):
    """Create a HigherIsingModel with given coefficients and constant.

    Check if
    - the num_bits property returns the expected number of bits.
    """
    # Create the HigherIsingModel.

    # - the num_bits property returns the expected number of bits.


@pytest.mark.parametrize(
    "coefficients, constant, state, expected_energy",
    [
        ({(0, 1): 2.0, (0,): 4.0, (1,): 5.0}, 6.0, [1, -1], -1.0),
        ({(0, 2): -1.0, (1,): 3.0}, -2.0, [-1, 1, -1], 0.0),
        ({(0, 1, 2): 1.5, (2,): -2.0}, 1.0, [1, 1, 1], 0.5),
        ({}, 5.0, [1, -1, 1], 5.0),
    ],
)
def test_calc_energy_manually(coefficients, constant, state, expected_energy):
    """Run calc_energy method.

    Check if
    - the calc_energy method returns the expected energy.
    """
    # Create the HigherIsingModel.

    # - the calc_energy method returns the expected energy.


def test_normalize_by_abs_max_empty():
    """Run normalize_by_abs_max on an empty HigherIsingModel.

    Check if
    - the coefficients remain empty,
    - the constant term remains unchanged.
    """
    # Create an empty HigherIsingModel.

    # Run normalize_by_abs_max.

    # - the coefficients remain empty,
    # - the constant term remains unchanged.


@pytest.mark.parametrize("seed", [901 + i for i in range(100)])
def test_normalize_by_abs_max(seed):
    """Run normalize_by_abs_max on a non-empty HigherIsingModel.

    Check if
    - all coefficients are scaled correctly,
    - the constant term is scaled correctly.
    """
    # Fix the numpy.random.seed for reproducibility.

    # Create random coefficients.

    # Create a random constant term.

    # Create the HigherIsingModel.

    # Calculate the expected normalization factor before normalization.

    # Run normalize_by_abs_max.

    # - all coefficients are scaled correctly,
    # - the constant term is scaled correctly.


def test_normalize_by_rms_empty():
    """Run normalize_by_rms on an empty HigherIsingModel.

    Check if
    - the coefficients remain empty,
    - the constant term remains unchanged.
    """
    # Create an empty HigherIsingModel.

    # Run normalize_by_rms.

    # - the coefficients remain empty,
    # - the constant term remains unchanged.


@pytest.mark.parametrize("seed", [901 + i for i in range(100)])
def test_normalize_by_rms(seed):
    """Run normalize_by_rms on a non-empty HigherIsingModel.

    Check if
    - all coefficients are scaled correctly,
    - the constant term is scaled correctly.
    """
    # Fix the numpy.random.seed for reproducibility.

    # Create random coefficients.

    # Create a random constant term.

    # Create the HigherIsingModel.

    # Calculate the expected normalization factor before normalization.

    # Run normalize_by_rms.

    # - all coefficients are scaled correctly,
    # - the constant term is scaled correctly.


def test_normalize_by_factor_empty():
    """Run normalize_by_factor on an empty HigherIsingModel.

    Check if
    - the coefficients remain empty,
    - the constant term remains unchanged.
    """
    # Create an empty HigherIsingModel.

    # Run normalize_by_factor.

    # - the coefficients remain empty,
    # - the constant term remains unchanged.


def test_normalize_by_factor_0():
    """Run normalize_by_factor with factor 0 on a non-empty HigherIsingModel.

    Check if
    - the coefficients remain unchanged,
    - the constant term remains unchanged.
    """
    # Create random coefficients.

    # Create a random constant term.

    # Create the HigherIsingModel.

    # Store the coefficients and constant term before normalization.

    # Run normalize_by_factor with factor 0.

    # - the coefficients remain unchanged,
    # - the constant term remains unchanged.


@pytest.mark.parametrize("seed", [901 + i for i in range(100)])
def test_normalize_by_factor(seed):
    """Run normalize_by_rms on a non-empty HigherIsingModel.

    Check if
    - all coefficients are scaled correctly,
    - the constant term is scaled correctly.
    """
    # Fix the numpy.random.seed for reproducibility.

    # Create random coefficients.

    # Create a random constant term.

    # Create the HigherIsingModel.

    # Calculate the expected normalization factor before normalization.

    # Run normalize_by_rms.

    # - all coefficients are scaled correctly,
    # - the constant term is scaled correctly.


def test_from_hubo_manually():
    """Run from_hubo with manual data.

    Check if
    - the coefficients are as expected,
    - the constant term is as expected.
    - the index_map is as expected.
    """
    # Define the HUBO coefficients and constant term.
    hubo = {(0,): 1.0, (0, 1): 2.0, (0, 1, 3): 4.0}
    constant = 8.0
    # Define the expected coefficients, constant term, and index_map.
    expected_coefficients = {
        (0,): -12 / 8,
        (1,): -8 / 8,
        (3,): -4 / 8,
        (0, 1): 8 / 8,
        (0, 3): 4 / 8,
        (1, 3): 4 / 8,
        (0, 1, 3): -4 / 8,
    }
    expected_constant = 76 / 8
    expected_index_map = {0: 0, 1: 1, 3: 3}
    # Create the HigherIsingModel from the HUBO.
    higher_ising = HigherIsingModel.from_hubo(hubo=hubo, constant=constant)

    # - the coefficients are as expected,
    for key, value in expected_coefficients.items():
        assert numpy.isclose(higher_ising.coefficients[key], value)
    # - the constant term is as expected.
    assert numpy.isclose(higher_ising.constant, expected_constant)
    # - the index_map is as expected.
    assert higher_ising.index_map == expected_index_map
