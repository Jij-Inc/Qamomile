import numpy as np
import pytest

from qamomile.optimization.schedules.dicke import (
    _scs_schedule,
    bartschi_eidenbenz_schedule,
    dicke_state_composition_schedule,
)


def test_scs_schedule_for_weight_two_builds_one_pair_and_one_triplet():
    """Tests that _scs_schedule for n=4, k=2 returns exactly one pair and one triplet with correct indices and angles."""
    pair_indices, triplets_indices, pair_angles, triplets_angles = _scs_schedule(4, 2)

    np.testing.assert_array_equal(pair_indices, np.asarray([[2, 3]], dtype=np.uint32))
    np.testing.assert_array_equal(
        triplets_indices, np.asarray([[1, 2, 3]], dtype=np.uint32)
    )
    np.testing.assert_allclose(pair_angles, [2 * np.arccos(0.5)])
    np.testing.assert_allclose(triplets_angles, [2 * np.arccos(np.sqrt(0.5))])


def test_bartschi_eidenbenz_schedule_stacks_columns_in_descending_size_order():
    """Tests that bartschi_eidenbenz_schedule concatenates SCS columns in descending qubit-register order."""
    pair_indices, triplets_indices, pair_angles, triplets_angles = (
        bartschi_eidenbenz_schedule(4, 2)
    )

    expected_pairs = np.asarray([[2, 3], [1, 2], [0, 1]], dtype=np.uint32)
    expected_triplets = np.asarray([[1, 2, 3], [0, 1, 2]], dtype=np.uint32)

    np.testing.assert_array_equal(pair_indices, expected_pairs)
    np.testing.assert_array_equal(triplets_indices, expected_triplets)
    assert pair_angles.shape == (3,)
    assert triplets_angles.shape == (2,)


def test_dicke_state_composition_schedule_repeats_local_schedule_per_block():
    """Tests that dicke_state_composition_schedule tiles the local BE schedule across blocks with correct index offsets."""
    initial_ones, pair_indices, triplets_indices, pair_angles, triplets_angles = (
        dicke_state_composition_schedule(n_qubits=6, block_size=3, hamming_weight=2)
    )
    local_pairs, local_triplets, local_pair_angles, local_triplets_angles = (
        bartschi_eidenbenz_schedule(3, 2)
    )

    expected_initial_ones = np.asarray([1, 2, 4, 5], dtype=np.uint32)
    expected_pair_indices = np.vstack([local_pairs, local_pairs + 3]).astype(np.uint32)
    expected_triplets_indices = np.vstack([local_triplets, local_triplets + 3]).astype(
        np.uint32
    )
    expected_pair_angles = np.tile(local_pair_angles, 2)
    expected_triplets_angles = np.tile(local_triplets_angles, 2)

    np.testing.assert_array_equal(initial_ones, expected_initial_ones)
    np.testing.assert_array_equal(pair_indices, expected_pair_indices)
    np.testing.assert_array_equal(triplets_indices, expected_triplets_indices)
    np.testing.assert_allclose(pair_angles, expected_pair_angles)
    np.testing.assert_allclose(triplets_angles, expected_triplets_angles)


@pytest.mark.parametrize(
    ("n_dicke", "k_dicke"),
    [(1, 0), (4, 0), (4, 4), (6, 6)],
)
def test_bartschi_eidenbenz_schedule_early_return_for_trivial_weights(n_dicke, k_dicke):
    """Tests that k=0 and k=n_dicke return empty arrays (Dicke state is already a basis state)."""
    pair_indices, triplets_indices, pair_angles, triplets_angles = bartschi_eidenbenz_schedule(
        n_dicke, k_dicke
    )

    assert pair_indices.size == 0
    assert triplets_indices.size == 0
    assert pair_angles.size == 0
    assert triplets_angles.size == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_qubits": 0, "block_size": 2, "hamming_weight": 1}, "n_qubits must be > 0"),
        (
            {"n_qubits": 5, "block_size": 0, "hamming_weight": 1},
            "block_size must be > 0",
        ),
        (
            {"n_qubits": 5, "block_size": 2, "hamming_weight": 1},
            "divisible by block_size",
        ),
        (
            {"n_qubits": 6, "block_size": 3, "hamming_weight": 4},
            "Require 0 <= hamming_weight <= block_size",
        ),
    ],
)
def test_dicke_state_composition_schedule_validates_inputs(kwargs, message):
    """Tests that dicke_state_composition_schedule raises ValueError for invalid n_qubits, block_size, or hamming_weight."""
    with pytest.raises(ValueError, match=message):
        dicke_state_composition_schedule(**kwargs)
