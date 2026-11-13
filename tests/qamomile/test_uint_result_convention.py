"""Pin the two bit-order conventions that meet at a measured ``UInt``.

Coherent arithmetic is little-endian: lane 0 is the least significant bit, so
``qm.prepare((0, 1))`` reinterpreted with ``qm.as_uint`` represents two. The
result ABI decodes a measured ``UInt`` reading lane 0 as the most significant
bit, and ``qm.prepare`` of an integer ``UInt`` value uses the same order. These
tests fix both conventions as they are today so that changing either one is a
deliberate, visible ABI decision.
"""

from __future__ import annotations

import pytest

import qamomile as qm

SOURCE = """
from __future__ import annotations

import qamomile as qm

@qm.qkernel
def reinterpret_two() -> qm.Shot[qm.UInt[2]]:
    return qm.measure(qm.as_uint(qm.prepare((0, 1))))

@qm.qkernel
def one_plus_one() -> qm.Shot[qm.UInt[2]]:
    value = qm.as_uint(qm.prepare((1, 0)))
    value = value + 1
    return qm.measure(value)

@qm.qkernel
def one_plus_one_lanes() -> qm.Shot[qm.Bits[2]]:
    value = qm.as_uint(qm.prepare((1, 0)))
    value = value + 1
    return qm.measure(qm.as_bits(value))

@qm.qkernel
def prepared(value: qm.Param[qm.UInt[3]]) -> qm.Shot[qm.UInt[3]]:
    return qm.measure(qm.prepare(value))
"""


def _result(entry: str, **parameters: object) -> qm.SampleResult:
    program = qm.build_source(SOURCE).specialize(entry)
    return qm.run(
        program,
        on=qm.Reference(),
        shots=4,
        seed=3,
        parameters=parameters or None,
    ).result


def test_coherent_arithmetic_is_little_endian() -> None:
    """1 + 1 sets lane 1, the weight-two lane of a little-endian word."""

    assert _result("one_plus_one_lanes").counts == {(0, 1): 4}


def test_measured_uint_decodes_lane_zero_as_most_significant() -> None:
    """The decoded value of lanes ``(0, 1)`` is one, not the arithmetic two."""

    assert _result("reinterpret_two").counts == {(0, 1): 4}
    assert _result("reinterpret_two").values == {1: 4}
    assert _result("one_plus_one").counts == {(0, 1): 4}
    assert _result("one_plus_one").values == {1: 4}


@pytest.mark.parametrize(("value", "lanes"), [(6, (1, 1, 0)), (1, (0, 0, 1))])
def test_prepare_and_decode_share_the_result_order(
    value: int, lanes: tuple[int, ...]
) -> None:
    """``prepare`` of an integer and the decoder round-trip in result order."""

    result = _result("prepared", value=value)
    assert result.counts == {lanes: 4}
    assert result.values == {value: 4}
