"""Tests for the clean-ancilla pool of the multi-controlled decomposition.

The pool's *size* is measured by a count-only dry-run of the real emission
(``StandardEmitPass._count_multi_control_ancilla_demand``); those
end-to-end demand checks live in the QURI Parts backend suite
(``tests/transpiler/backends/test_quri_parts_frontend.py``). This file
covers the pool data structure itself — its offset / hold discipline and
its counting mode — in isolation.
"""

import pytest

from qamomile.circuit.transpiler.passes.emit_support.multi_control_ancilla import (
    MultiControlAncillaPool,
)


def test_pool_take_returns_leading_indices() -> None:
    """take() hands out the first k reserved indices."""
    pool = MultiControlAncillaPool(first_index=7, count=3)
    assert pool.take(2) == [7, 8]
    assert pool.take(3) == [7, 8, 9]


def test_pool_take_shortfall_returns_none() -> None:
    """take() signals a shortfall with None instead of raising."""
    pool = MultiControlAncillaPool(first_index=7, count=1)
    assert pool.take(2) is None


def test_pool_take_rejects_negative_count() -> None:
    """A negative request is a caller bug and raises, not a silent empty list."""
    pool = MultiControlAncillaPool(first_index=7, count=3)
    with pytest.raises(ValueError, match="non-negative"):
        pool.take(-1)
    with pytest.raises(ValueError, match="non-negative"):
        with pool.try_hold(-1):
            pass


def test_pool_try_hold_advances_offset_so_take_draws_after_held_range() -> None:
    """While a hold is active, take() hands out qubits after the held range."""
    pool = MultiControlAncillaPool(first_index=10, count=5)
    with pool.try_hold(2) as held:
        assert held == [10, 11]
        # A leaf cascade inside the batched body draws from the offset.
        assert pool.take(2) == [12, 13]


def test_pool_try_hold_rewinds_offset_on_exit() -> None:
    """Sibling batches reuse the same range because the offset rewinds."""
    pool = MultiControlAncillaPool(first_index=10, count=5)
    with pool.try_hold(3) as first:
        assert first == [10, 11, 12]
    with pool.try_hold(3) as second:
        assert second == [10, 11, 12]


def test_pool_try_hold_rewinds_offset_on_exception() -> None:
    """An exception inside the hold still rewinds the offset."""
    pool = MultiControlAncillaPool(first_index=10, count=5)
    with pytest.raises(RuntimeError):
        with pool.try_hold(2):
            raise RuntimeError("boom")
    assert pool.take(2) == [10, 11]


def test_pool_try_hold_nests() -> None:
    """Nested holds stack, each drawing after the previous."""
    pool = MultiControlAncillaPool(first_index=0, count=6)
    with pool.try_hold(2) as outer:
        assert outer == [0, 1]
        with pool.try_hold(2) as inner:
            assert inner == [2, 3]
            assert pool.take(2) == [4, 5]


def test_pool_try_hold_yields_none_when_remainder_too_small() -> None:
    """try_hold yields None (no raise) when the unheld remainder is too small."""
    pool = MultiControlAncillaPool(first_index=0, count=2)
    with pool.try_hold(1):
        with pool.try_hold(2) as shortfall:
            assert shortfall is None
    # Offset fully rewound: the whole pool is available again.
    assert pool.take(2) == [0, 1]


def test_pool_counting_mode_records_peak_including_held_ladders() -> None:
    """In counting mode take/try_hold never fail and record peak usage.

    A held ladder plus a take inside it must contribute their sum to the
    peak (mirroring a nested cascade drawn from behind a batched ladder),
    and the peak must ignore usage that has since been released.
    """
    pool = MultiControlAncillaPool(first_index=0, count=0, counting=True)
    assert pool.take(3) == [0, 1, 2]
    assert pool.peak == 3
    with pool.try_hold(2):
        # A nested cascade behind the held ladder pushes the peak to 2 + 4.
        pool.take(4)
        assert pool.peak == 6
    # Releasing the hold does not lower the recorded peak.
    pool.take(1)
    assert pool.peak == 6
