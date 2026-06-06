import numpy as np
import pytest

from qamomile.optimization.schedules.dicke import (
    bartschi_eidenbenz_schedule,
    dicke_state_composition_schedule,
)


def test_bartschi_eidenbenz_schedule_first_column_matches_scs_formula():
    """Tests that the first column of bartschi_eidenbenz_schedule(4,2) matches the SCS angle formula.

    The topmost column (n=4, k_sub=2) yields pair (2,3,3) with angle 2*arccos(1/sqrt(4)),
    and triplet (1,2,3) with angle 2*arccos(sqrt(2/4)). These values are derived from
    the Bartschi-Eidenbenz SCS formula.
    """
    schedule = bartschi_eidenbenz_schedule(4, 2)

    # Pair gate encoded as (t, c, c).
    assert (2, 3, 3) in schedule
    np.testing.assert_allclose(schedule[(2, 3, 3)], 2 * np.arccos(1 / np.sqrt(4)))

    # Triplet gate has c1 != c2.
    assert (1, 2, 3) in schedule
    np.testing.assert_allclose(schedule[(1, 2, 3)], 2 * np.arccos(np.sqrt(2 / 4)))


def test_bartschi_eidenbenz_schedule_stacks_columns_in_descending_size_order():
    """Tests that bartschi_eidenbenz_schedule concatenates SCS columns in descending qubit-register order."""
    schedule = bartschi_eidenbenz_schedule(4, 2)

    expected_pair_keys = {(2, 3, 3), (1, 2, 2), (0, 1, 1)}
    expected_triplet_keys = {(1, 2, 3), (0, 1, 2)}
    all_keys = set(schedule.keys())

    assert expected_pair_keys <= all_keys
    assert expected_triplet_keys <= all_keys
    assert len(schedule) == 5  # 3 pairs + 2 triplets


def test_dicke_state_composition_schedule_repeats_local_schedule_per_block():
    """Tests that dicke_state_composition_schedule tiles the local BE schedule across blocks with correct index offsets."""
    initial_ones, schedule = dicke_state_composition_schedule(
        n_qubits=6, block_size=3, hamming_weight=2
    )
    local_schedule = bartschi_eidenbenz_schedule(3, 2)

    expected_initial_ones = np.asarray([1, 2, 4, 5], dtype=np.uint32)
    np.testing.assert_array_equal(initial_ones, expected_initial_ones)

    # Both blocks should appear: block 0 (offset 0) and block 1 (offset 3).
    for (t, c1, c2), angle in local_schedule.items():
        assert (t, c1, c2) in schedule
        np.testing.assert_allclose(schedule[(t, c1, c2)], angle)
        assert (t + 3, c1 + 3, c2 + 3) in schedule
        np.testing.assert_allclose(schedule[(t + 3, c1 + 3, c2 + 3)], angle)

    assert len(schedule) == 2 * len(local_schedule)


@pytest.mark.parametrize(
    ("n_dicke", "k_dicke"),
    [(1, 0), (4, 0), (4, 4), (6, 6)],
)
def test_bartschi_eidenbenz_schedule_early_return_for_trivial_weights(n_dicke, k_dicke):
    """Tests that k=0 and k=n_dicke return an empty dict (Dicke state is already a basis state)."""
    schedule = bartschi_eidenbenz_schedule(n_dicke, k_dicke)

    assert len(schedule) == 0


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
