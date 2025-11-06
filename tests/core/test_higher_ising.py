import numpy
import pytest

from qamomile.core.higher_ising_model import HigherIsingModel


@pytest.mark.parametrize("seed", [901 + i for i in range(100)])
def test_creation_with_defaults(seed):
    """Create a HigherIsingModel.

    Check if
    - the coefficients are the same as the input through original_to_zero_origin_map,
    - the constant term is the same as the input,
    - the keys of the index_map are the unique values of the integers of the coefficients' key.
    - the values of the index_map are the positions of the sorted unique integers.
    """
    # Fix the numpy.random.seed for reproducibility.
    numpy.random.seed(seed)

    # Create the random coefficients (up to 5th order with random indices).
    coefficients = {}
    num_terms = numpy.random.randint(5, 15)  # 5-14 terms
    for _ in range(num_terms):
        order = numpy.random.randint(1, 6)  # 1st to 5th order
        indices = tuple(sorted(numpy.random.choice(20, size=order, replace=False)))
        coefficients[indices] = numpy.random.randn()

    # Create the random constant term.
    constant = numpy.random.rand()

    # Create the HigherIsingModel.
    model = HigherIsingModel(coefficients=coefficients.copy(), constant=constant)

    # - the coefficients are the same as the input through original_to_zero_origin_map,
    for original_indices, coefficient in coefficients.items():
        zero_origin_indices = tuple(
            model.original_to_zero_origin_map[idx] for idx in original_indices
        )
        assert numpy.isclose(coefficient, model.coefficients[zero_origin_indices])
    # - the constant term is the same as the input,
    assert model.constant == constant
    # - the keys of the index_map are the unique values of the integers of the coefficients' key.
    unique_indices = {idx for key in coefficients.keys() for idx in key}
    assert set(model.index_map.keys()) == unique_indices
    # - the values of the index_map are the positions of the sorted unique integers.
    for index in unique_indices:
        assert model.index_map[index] == index


@pytest.mark.parametrize("seed", [901 + i for i in range(100)])
def test_creation_with_index_map(seed):
    """Create a HigherIsingModel by giving index_map.

    Check if
    - the index_map is the same as the input.
    """
    # Fix the numpy.random.seed for reproducibility.
    numpy.random.seed(seed)

    # Create the random coefficients (up to 5th order with random indices).
    coefficients = {}
    num_terms = numpy.random.randint(5, 15)  # 5-14 terms
    for _ in range(num_terms):
        order = numpy.random.randint(1, 6)  # 1st to 5th order
        indices = tuple(sorted(numpy.random.choice(20, size=order, replace=False)))
        coefficients[indices] = numpy.random.randn()

    # Create the random constant term.
    constant = numpy.random.rand()

    # Create a random index_map for all unique indices in coefficients.
    unique_indices = {idx for key in coefficients.keys() for idx in key}
    index_map = {idx: numpy.random.randint(100, 200) for idx in unique_indices}

    # Create the HigherIsingModel.
    model = HigherIsingModel(
        coefficients=coefficients, constant=constant, index_map=index_map
    )

    # - the index_map is the same as the input.
    assert model.index_map == index_map


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
        [{(1,): 1}, 0, 1],
    ],
)
def test_num_bits_manually(coefficients, constant, expected_num_bits):
    """Create a HigherIsingModel with given coefficients and constant.

    Check if
    - the num_bits property returns the expected number of bits.
    """
    # Create the HigherIsingModel.
    model = HigherIsingModel(coefficients=coefficients, constant=constant)

    # - the num_bits property returns the expected number of bits.
    assert model.num_bits == expected_num_bits


def test_calc_energy_non_spin():
    """Run calc_energy method with a non-spin state.

    Check if
    - the calc_energy method raises a ValueError.
    """
    # Create the HigherIsingModel.
    higher_ising = HigherIsingModel({(0, 1): 2.0, (0,): 4.0, (1,): 5.0}, 6.0)

    # - the calc_energy method raises a ValueError.
    with pytest.raises(ValueError):
        higher_ising.calc_energy([1, 0])  # 0 is not a valid spin value


@pytest.mark.parametrize(
    "coefficients, constant, state, expected_energy",
    [
        ({(0, 1): 2.0, (0,): 4.0, (1,): 5.0}, 6.0, [1, -1], 3.0),
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
    model = HigherIsingModel(coefficients=coefficients, constant=constant)

    # - the calc_energy method returns the expected energy.
    assert numpy.isclose(model.calc_energy(state), expected_energy)


def test_normalize_by_abs_max_empty():
    """Run normalize_by_abs_max on an empty HigherIsingModel.

    Check if
    - the coefficients remain empty,
    - the constant term remains unchanged.
    """
    # Create an empty HigherIsingModel.
    constant = 5.0
    model = HigherIsingModel(coefficients={}, constant=constant)

    # Run normalize_by_abs_max.
    model.normalize_by_abs_max()

    # - the coefficients remain empty,
    assert model.coefficients == {}
    # - the constant term remains unchanged.
    assert model.constant == constant


@pytest.mark.parametrize("seed", [901 + i for i in range(100)])
def test_normalize_by_abs_max(seed):
    """Run normalize_by_abs_max on a non-empty HigherIsingModel.

    Check if
    - all coefficients are scaled correctly,
    - the constant term is scaled correctly.
    """
    # Fix the numpy.random.seed for reproducibility.
    numpy.random.seed(seed)

    # Create random coefficients (up to 5th order with random indices).
    coefficients = {}
    num_terms = numpy.random.randint(5, 15)  # 5-14 terms
    for _ in range(num_terms):
        order = numpy.random.randint(1, 6)  # 1st to 5th order
        indices = tuple(sorted(numpy.random.choice(20, size=order, replace=False)))
        coefficients[indices] = numpy.random.randn()

    # Create a random constant term.
    constant = numpy.random.randn()

    # Create the HigherIsingModel.
    model = HigherIsingModel(coefficients=coefficients.copy(), constant=constant)

    # Calculate the expected normalization factor before normalization.
    max_abs = max(abs(v) for v in coefficients.values())

    # Run normalize_by_abs_max.
    model.normalize_by_abs_max()

    # - all coefficients are scaled correctly,
    for original_indices, coefficient in coefficients.items():
        zero_origin_indices = tuple(
            model.original_to_zero_origin_map[idx] for idx in original_indices
        )
        assert numpy.isclose(
            model.coefficients[zero_origin_indices], coefficient / max_abs
        )
    # - the constant term is scaled correctly.
    assert numpy.isclose(model.constant, constant / max_abs)


def test_normalize_by_rms_empty():
    """Run normalize_by_rms on an empty HigherIsingModel.

    Check if
    - the coefficients remain empty,
    - the constant term remains unchanged.
    """
    # Create an empty HigherIsingModel.
    constant = 5.0
    model = HigherIsingModel(coefficients={}, constant=constant)

    # Run normalize_by_rms.
    model.normalize_by_rms()

    # - the coefficients remain empty,
    assert model.coefficients == {}
    # - the constant term remains unchanged.
    assert model.constant == constant


@pytest.mark.parametrize("seed", [901 + i for i in range(100)])
def test_normalize_by_rms(seed):
    """Run normalize_by_rms on a non-empty HigherIsingModel.

    Check if
    - all coefficients are scaled correctly,
    - the constant term is scaled correctly.
    """
    # Fix the numpy.random.seed for reproducibility.
    numpy.random.seed(seed)

    # Create random coefficients (up to 5th order with random indices).
    coefficients = {}
    num_terms = numpy.random.randint(5, 15)  # 5-14 terms
    for _ in range(num_terms):
        order = numpy.random.randint(1, 6)  # 1st to 5th order
        indices = tuple(sorted(numpy.random.choice(20, size=order, replace=False)))
        coefficients[indices] = numpy.random.randn()

    # Create a random constant term.
    constant = numpy.random.randn()

    # Create the HigherIsingModel.
    model = HigherIsingModel(coefficients=coefficients.copy(), constant=constant)

    # Calculate the expected normalization factor before normalization.
    counts = {}
    for indices, coeff in coefficients.items():
        order = len(indices)
        if order not in counts:
            counts[order] = [0.0, 0]
        counts[order][0] += coeff**2
        counts[order][1] += 1

    rms_components = 0.0
    for order, (sum_squares, count) in counts.items():
        if count > 0:
            mean_square = sum_squares / count
            rms_components += mean_square

    rms = numpy.sqrt(rms_components)

    # Run normalize_by_rms.
    model.normalize_by_rms()

    # - all coefficients are scaled correctly,
    for original_indices, coefficient in coefficients.items():
        zero_origin_indices = tuple(
            model.original_to_zero_origin_map[idx] for idx in original_indices
        )
        assert numpy.isclose(model.coefficients[zero_origin_indices], coefficient / rms)
    # - the constant term is scaled correctly.
    assert numpy.isclose(model.constant, constant / rms)


def test_normalize_by_factor_empty():
    """Run normalize_by_factor on an empty HigherIsingModel.

    Check if
    - the coefficients remain empty,
    - the constant term remains unchanged.
    """
    # Create an empty HigherIsingModel.
    constant = 5.0
    model = HigherIsingModel(coefficients={}, constant=constant)

    # Run normalize_by_factor.
    model.normalize_by_factor(factor=2.0)

    # - the coefficients remain empty,
    assert model.coefficients == {}
    # - the constant term remains unchanged.
    assert model.constant == constant


def test_normalize_by_factor_0():
    """Run normalize_by_factor with factor 0 on a non-empty HigherIsingModel.

    Check if
    - the coefficients remain unchanged,
    - the constant term remains unchanged.
    """
    # Create random coefficients.
    coefficients = {
        (0, 1): 2.0,
        (2,): 3.0,
        (1, 3, 5): -1.5,
    }

    # Create a random constant term.
    constant = 7.0

    # Create the HigherIsingModel.
    model = HigherIsingModel(coefficients=coefficients.copy(), constant=constant)

    # Store the coefficients and constant term before normalization.
    original_constant = constant

    # Run normalize_by_factor with factor 0.
    model.normalize_by_factor(factor=0.0)

    # - the coefficients remain unchanged,
    for original_indices, coefficient in coefficients.items():
        zero_origin_indices = tuple(
            model.original_to_zero_origin_map[idx] for idx in original_indices
        )
        assert numpy.isclose(coefficient, model.coefficients[zero_origin_indices])
    # - the constant term remains unchanged.
    assert model.constant == original_constant


@pytest.mark.parametrize("seed", [901 + i for i in range(100)])
def test_normalize_by_factor(seed):
    """Run normalize_by_factor on a non-empty HigherIsingModel.

    Check if
    - all coefficients are scaled correctly,
    - the constant term is scaled correctly.
    """
    # Fix the numpy.random.seed for reproducibility.
    numpy.random.seed(seed)

    # Create random coefficients (up to 5th order with random indices).
    coefficients = {}
    num_terms = numpy.random.randint(5, 15)  # 5-14 terms
    for _ in range(num_terms):
        order = numpy.random.randint(1, 6)  # 1st to 5th order
        indices = tuple(sorted(numpy.random.choice(20, size=order, replace=False)))
        coefficients[indices] = numpy.random.randn()

    # Create a random constant term.
    constant = numpy.random.randn()

    # Create the HigherIsingModel.
    model = HigherIsingModel(coefficients=coefficients.copy(), constant=constant)

    # Create a random normalization factor.
    factor = numpy.random.rand() + 0.5  # Ensure factor is not too close to zero

    # Run normalize_by_factor.
    model.normalize_by_factor(factor=factor)

    # - all coefficients are scaled correctly,
    for original_indices, coefficient in coefficients.items():
        zero_origin_indices = tuple(
            model.original_to_zero_origin_map[idx] for idx in original_indices
        )
        assert numpy.isclose(
            model.coefficients[zero_origin_indices], coefficient / factor
        )
    # - the constant term is scaled correctly.
    assert numpy.isclose(model.constant, constant / factor)


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
        zero_origin_indices = tuple(
            higher_ising.original_to_zero_origin_map[idx] for idx in key
        )
        assert numpy.isclose(higher_ising.coefficients[zero_origin_indices], value)
    # - the constant term is as expected.
    assert numpy.isclose(higher_ising.constant, expected_constant)
    # - the index_map is as expected.
    assert higher_ising.index_map == expected_index_map
