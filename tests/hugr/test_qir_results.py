"""Verify H2 output decoding without provider connections or quantum execution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from qamomile.circuit.transpiler.errors import ExecutionError
from qamomile.hugr._qir_results import decode_qir_results, parse_qir_output_manifest

pytestmark = pytest.mark.hugr


def _result(bits, rows):
    """Return a SDK-shaped result that preserves explicit readout ordering."""
    return SimpleNamespace(
        get_bitlist=Mock(return_value=bits), get_shots=Mock(return_value=rows)
    )


def _bit(tag, index):
    """Provide the public fields used by a native pytket bit identifier."""
    return SimpleNamespace(reg_name=tag, index=[index])


def test_boolean_decode_preserves_explicit_provider_column_order():
    """Reordered labels retain their meaning, including indices above nine."""
    tags = [f"qamomile.output.0.{index}" for index in range(12)]
    expected = {tag: index in (1, 4, 9) for index, tag in enumerate(tags)}
    bits = [_bit(tag, index) for index, tag in reversed(list(enumerate(tags)))]
    row = [int(expected[bit.reg_name]) for bit in bits]
    raw = _result(bits, np.asarray([row, [1 - value for value in row]], dtype=np.uint8))

    actual = decode_qir_results(raw, 2, dict.fromkeys(tags, ("bool", 1)))

    assert actual == [expected, {tag: not value for tag, value in expected.items()}]
    assert all(type(value) is bool for record in actual for value in record.values())
    raw.get_shots.assert_called_once_with(cbits=bits)


@pytest.mark.parametrize("indices", [(0, 0, 0), (0, 1, 10)])
def test_native_pytket_readout_order_and_empty_output(indices):
    """Local and global SDK indices retain tag identity across serialization."""
    pytest.importorskip("pytket")
    from pytket.backends.backendresult import BackendResult
    from pytket.circuit import Bit
    from pytket.utils.outcomearray import OutcomeArray

    tags = ["qamomile.output.0.0", "qamomile.output.1.0", "qamomile.output.0.10"]
    bits = [Bit(tag, index) for tag, index in reversed(list(zip(tags, indices)))]
    raw = BackendResult(
        c_bits=bits,
        shots=OutcomeArray.from_readouts(
            np.array([[1, 1, 0], [0, 0, 1]], dtype=np.uint8)
        ),
    )
    stored = BackendResult.from_dict(raw.to_dict())
    assert decode_qir_results(stored, 2, dict.fromkeys(tags, ("bool", 1))) == [
        {tags[0]: False, tags[1]: True, tags[2]: True},
        {tags[0]: True, tags[1]: False, tags[2]: False},
    ]
    empty = BackendResult(
        c_bits=[],
        shots=OutcomeArray.from_readouts(np.empty((3, 0), dtype=np.uint8)),
    )
    assert decode_qir_results(empty, 3, {}) == [{}, {}, {}]


@pytest.mark.parametrize("interleaved", [False, True])
def test_native_h_series_registers_decode_index_zero_in_provider_order(interleaved):
    """Complete 64-bit registers decode by tag and index, ignoring upper bits."""
    pytest.importorskip("pytket")
    from pytket.backends.backendresult import BackendResult
    from pytket.circuit import Bit
    from pytket.utils.outcomearray import OutcomeArray

    tags = ["qamomile.output.1.0", "qamomile.output.0.0", "qamomile.output.0.10"]
    positions = (
        [(tag, index) for index in reversed(range(64)) for tag in tags]
        if interleaved
        else [(tag, index) for tag in tags for index in reversed(range(64))]
    )
    expected = [
        dict(zip(tags, [True, False, False], strict=True)),
        dict(zip(tags, [False, True, True], strict=True)),
    ]
    raw = BackendResult(
        c_bits=[Bit(tag, index) for tag, index in positions],
        shots=OutcomeArray.from_readouts(
            np.asarray(
                [
                    [
                        int(record[tag]) if index == 0 else (index + shot) % 2
                        for tag, index in positions
                    ]
                    for shot, record in enumerate(expected)
                ],
                dtype=np.uint8,
            )
        ),
    )

    stored = BackendResult.from_dict(raw.to_dict())

    assert decode_qir_results(stored, 2, dict.fromkeys(tags, ("bool", 1))) == expected


def test_scalar_and_complete_h_series_registers_can_coexist():
    """Each tag independently selects its supported native storage layout."""
    bits = [_bit("scalar", 10), *[_bit("wide", index) for index in reversed(range(64))]]
    raw = _result(bits, [[0, *([1] * 63), 1], [1, *([1] * 63), 0]])
    assert decode_qir_results(raw, 2, {"scalar": ("bool", 1), "wide": ("bool", 1)}) == [
        {"scalar": False, "wide": True},
        {"scalar": True, "wide": False},
    ]


def test_empty_output_still_validates_provider_shot_count():
    """A result without public leaves produces one empty mapping for each real shot."""
    raw = _result([], np.empty((3, 0), dtype=np.uint8))
    assert decode_qir_results(raw, 3, {}) == [{}, {}, {}]
    with pytest.raises(ExecutionError, match="shape"):
        decode_qir_results(raw, 2, {})


@pytest.mark.parametrize(
    "bits,outputs,message",
    [
        ([], {"flag": ("bool", 1)}, "Missing"),
        ([_bit("other", 0)], {"flag": ("bool", 1)}, "Unexpected"),
        ([_bit("flag", 0), _bit("flag", 0)], {"flag": ("bool", 1)}, "Duplicate"),
        ([_bit("flag", 0), _bit("flag", 1)], {"flag": ("bool", 1)}, "QIR"),
        ([_bit("flag", 10), _bit("flag", 2)], {"flag": ("bool", 1)}, "QIR"),
        ([_bit("flag", -1)], {"flag": ("bool", 1)}, "index"),
        ([_bit("flag", True)], {"flag": ("bool", 1)}, "index"),
        ([_bit("flag", 1.0)], {"flag": ("bool", 1)}, "index"),
        ([_bit("flag", "1")], {"flag": ("bool", 1)}, "index"),
        ([_bit("flag", None)], {"flag": ("bool", 1)}, "index"),
        (
            [SimpleNamespace(reg_name="flag", index=[])],
            {"flag": ("bool", 1)},
            "index",
        ),
        (
            [SimpleNamespace(reg_name="flag", index=[0, 0])],
            {"flag": ("bool", 1)},
            "index",
        ),
    ],
)
def test_invalid_registers_fail_before_fetching_readouts(bits, outputs, message):
    """Incomplete or ambiguous registers cannot silently become public values."""
    raw = _result(bits, [])
    with pytest.raises(ExecutionError, match=message):
        decode_qir_results(raw, 1, outputs)
    raw.get_shots.assert_not_called()


@pytest.mark.parametrize(
    "indices",
    [
        list(range(63)),
        list(range(1, 64)),
        [*range(63), 64],
        list(range(65)),
        [*range(63), 0],
    ],
)
def test_malformed_h_series_registers_fail_before_fetching_readouts(indices):
    """Wide registers require every index from zero through 63 exactly once."""
    raw = _result([_bit("flag", index) for index in indices], [])
    with pytest.raises(ExecutionError):
        decode_qir_results(raw, 1, {"flag": ("bool", 1)})
    raw.get_shots.assert_not_called()


@pytest.mark.parametrize(
    "rows", [[[0], [1]], [0], [[0, 1]], [[2]], [[-1]], [[0.0]], [["1"]]]
)
def test_invalid_readouts_and_shot_counts_are_rejected(rows):
    """Wrong dimensions, non-bits, and noninteger values fail instead of being coerced."""
    raw = _result([_bit("flag", 0)], rows)
    with pytest.raises(ExecutionError):
        decode_qir_results(raw, 1, {"flag": ("bool", 1)})


@pytest.mark.parametrize(
    "raw", [None, object(), SimpleNamespace(results="OUTPUT\tBOOL\ttrue\tflag")]
)
def test_other_provider_result_formats_are_rejected(raw):
    """The H2 decoder never treats arbitrary QIR text as typed register readouts."""
    with pytest.raises(ExecutionError, match="bit-register"):
        decode_qir_results(raw, 1, {"flag": ("bool", 1)})


def test_provider_readout_failure_is_an_execution_error():
    """Unavailable individual shots produce a stable execution diagnosis."""
    raw = _result([_bit("flag", 0)], [])
    raw.get_shots.side_effect = RuntimeError("shots unavailable")
    with pytest.raises(ExecutionError, match="Cannot read QIR shot"):
        decode_qir_results(raw, 1, {"flag": ("bool", 1)})


@pytest.mark.parametrize("shots", [0, -1, True, 1.5])
def test_invalid_shots_fail_before_provider_access(shots):
    """The decoder enforces the same positive count boundary as submission."""
    raw = _result([], [])
    with pytest.raises(ValueError, match="shots"):
        decode_qir_results(raw, shots, {})
    raw.get_bitlist.assert_not_called()


def test_output_manifest_parses_owned_scalar_descriptors():
    """Saved JSON descriptors retain their labels and supported scalar kinds."""
    assert parse_qir_output_manifest('{"flag":["bool",1],"other":["bool",1]}') == {
        "flag": ("bool", 1),
        "other": ("bool", 1),
    }
    assert parse_qir_output_manifest("{}") == {}


@pytest.mark.parametrize(
    "encoded",
    [
        None,
        "",
        "null",
        "[]",
        "true",
        "{",
        '{"flag":["bool",1],"flag":["bool",1]}',
        '{"flag":["bool",true]}',
        '{"flag":["bool",1.0]}',
        '{"flag":["bool",64]}',
        '{"flag":["uint",1]}',
        '{"flag":["uint",64]}',
        '{"flag":["float",64]}',
        '{"flag":["uint",64,0]}',
        '{"flag":"bool"}',
        '{"":["bool",1]}',
        '{"flag":["uint",null]}',
    ],
)
def test_invalid_output_manifests_are_rejected(encoded):
    """Malformed, duplicate, unsupported, and loosely typed metadata cannot restore jobs."""
    with pytest.raises(ValueError):
        parse_qir_output_manifest(encoded)


def test_manifest_validation_precedes_provider_access():
    """Unsupported scalar recording never causes result retrieval."""
    raw = _result([], [])
    with pytest.raises(ValueError, match="Unsupported"):
        decode_qir_results(raw, 1, {"float": ("float", 64)})
    raw.get_bitlist.assert_not_called()
