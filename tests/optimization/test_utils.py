"""Tests for the spin/binary conversion helpers in qamomile.optimization.utils."""

import numpy as np
import pytest

from qamomile.optimization.utils import binary_to_spin, spin_to_binary


def test_binary_to_spin_scalar_uses_z_eigenvalue_convention():
    """0 maps to +1 and 1 maps to -1 (s = 1 - 2x)."""
    assert binary_to_spin(0) == 1
    assert binary_to_spin(1) == -1


def test_spin_to_binary_scalar_uses_z_eigenvalue_convention():
    """+1 maps to 0 and -1 maps to 1 (x = (1 - s) / 2)."""
    assert spin_to_binary(1) == 0
    assert spin_to_binary(-1) == 1


def test_scalar_return_type_is_plain_int():
    """Scalar conversions return builtin ints, not NumPy scalars."""
    assert isinstance(binary_to_spin(0), int)
    assert isinstance(spin_to_binary(1), int)


def test_bool_scalars_are_accepted_as_bits():
    """Python bools are treated as 0/1 binary values."""
    assert binary_to_spin(False) == 1
    assert binary_to_spin(True) == -1


def test_list_preserves_shape_and_returns_list():
    """A list of bits converts element-wise into a list of spins."""
    result = binary_to_spin([0, 1, 1, 0])
    assert result == [1, -1, -1, 1]
    assert isinstance(result, list)


def test_tuple_input_returns_list():
    """A tuple input is converted into a list of spins."""
    assert spin_to_binary((1, -1, 1)) == [0, 1, 0]


def test_nested_list_of_samples_is_converted_recursively():
    """Decoded sample sets shaped as list[list[int]] convert per element."""
    samples = [[1, -1, 1], [-1, -1, 1]]
    assert spin_to_binary(samples) == [[0, 1, 0], [1, 1, 0]]


def test_mapping_converts_values_and_keeps_keys():
    """A {index: value} sample keeps its keys and converts its values."""
    assert binary_to_spin({0: 0, 3: 1, 7: 0}) == {0: 1, 3: -1, 7: 1}


def test_numpy_array_returns_integer_array():
    """A NumPy binary array converts into an int spin array of same shape."""
    arr = np.array([[0, 1], [1, 0]])
    result = binary_to_spin(arr)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([[1, -1], [-1, 1]]))
    assert np.issubdtype(result.dtype, np.integer)


def test_numpy_bool_array_is_accepted():
    """A boolean NumPy array is treated as 0/1 bits."""
    arr = np.array([True, False, True])
    np.testing.assert_array_equal(binary_to_spin(arr), np.array([-1, 1, -1]))


def test_spin_to_binary_array_returns_integer_dtype():
    """The spin-to-binary array path also yields an integer array."""
    result = spin_to_binary(np.array([1, -1, 1]))
    assert np.issubdtype(result.dtype, np.integer)
    np.testing.assert_array_equal(result, np.array([0, 1, 0]))


def test_numpy_scalar_inputs_return_plain_int():
    """NumPy integer/boolean scalars decode to builtin ints, not NumPy scalars."""
    for value in (np.int64(0), np.int32(1), np.bool_(True)):
        assert isinstance(binary_to_spin(value), int)
    assert binary_to_spin(np.int64(0)) == 1
    assert binary_to_spin(np.bool_(True)) == -1
    assert isinstance(spin_to_binary(np.int64(-1)), int)


def test_empty_containers_are_preserved():
    """Empty list, dict, and integer array convert to empty results."""
    assert binary_to_spin([]) == []
    assert spin_to_binary({}) == {}
    empty = binary_to_spin(np.array([], dtype=int))
    assert isinstance(empty, np.ndarray)
    assert empty.size == 0


def test_mapping_values_may_be_nested_containers():
    """Mapping values that are lists/arrays convert recursively."""
    assert spin_to_binary({0: [1, -1], 1: [-1, 1]}) == {0: [0, 1], 1: [1, 0]}


@pytest.mark.parametrize("array_fn", [binary_to_spin, spin_to_binary])
def test_float_arrays_are_rejected(array_fn):
    """Float NumPy arrays raise TypeError like float scalars do."""
    with pytest.raises(TypeError):
        array_fn(np.array([0.0, 1.0]))


@pytest.mark.parametrize("bits", [0, 1, [0, 1, 0], [1, 1, 0]])
def test_round_trip_binary_spin_binary(bits):
    """spin_to_binary(binary_to_spin(x)) is the identity on valid bits."""
    assert spin_to_binary(binary_to_spin(bits)) == bits


@pytest.mark.parametrize("spins", [1, -1, [1, -1, 1], [-1, -1, 1]])
def test_round_trip_spin_binary_spin(spins):
    """binary_to_spin(spin_to_binary(s)) is the identity on valid spins."""
    assert binary_to_spin(spin_to_binary(spins)) == spins


def test_tuple_round_trips_to_list():
    """A tuple input round-trips to an equal list (containers normalize to list)."""
    assert spin_to_binary(binary_to_spin((0, 1, 0))) == [0, 1, 0]


def test_numpy_round_trip():
    """Array conversions round-trip back to the original bits/spins."""
    arr = np.array([[0, 1, 1], [1, 0, 0]])
    np.testing.assert_array_equal(spin_to_binary(binary_to_spin(arr)), arr)


@pytest.mark.parametrize("bad", [2, -1, [0, 2], {0: 3}])
def test_binary_to_spin_rejects_non_binary(bad):
    """Values outside {0, 1} raise ValueError."""
    with pytest.raises(ValueError):
        binary_to_spin(bad)


@pytest.mark.parametrize("bad", [0, 2, [1, 0], {0: 2}])
def test_spin_to_binary_rejects_non_spin(bad):
    """Values outside {+1, -1} raise ValueError."""
    with pytest.raises(ValueError):
        spin_to_binary(bad)


def test_numpy_array_rejects_out_of_domain_entries():
    """An array containing an invalid entry raises ValueError."""
    with pytest.raises(ValueError):
        binary_to_spin(np.array([0, 1, 2]))
    with pytest.raises(ValueError):
        spin_to_binary(np.array([1, -1, 0]))


@pytest.mark.parametrize("convert", [binary_to_spin, spin_to_binary])
@pytest.mark.parametrize("bad", ["01", b"01", bytearray(b"01"), 1.0, None])
def test_unsupported_types_raise_type_error(convert, bad):
    """Strings, bytes, floats, and None are rejected by both directions."""
    with pytest.raises(TypeError):
        convert(bad)
