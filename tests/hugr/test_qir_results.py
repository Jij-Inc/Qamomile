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
    bits = [_bit(tag, 0) for tag in reversed(tags)]
    row = [int(expected[bit.reg_name]) for bit in bits]
    raw = _result(bits, np.asarray([row, [1 - value for value in row]], dtype=np.uint8))

    actual = decode_qir_results(raw, 2, dict.fromkeys(tags, ("bool", 1)))

    assert actual == [expected, {tag: not value for tag, value in expected.items()}]
    assert all(type(value) is bool for record in actual for value in record.values())
    raw.get_shots.assert_called_once_with(cbits=bits)


def test_native_pytket_readout_order_and_empty_output():
    """The installed provider SDK follows the decoder's explicit column contract."""
    pytest.importorskip("pytket")
    from pytket.backends.backendresult import BackendResult
    from pytket.circuit import Bit
    from pytket.utils.outcomearray import OutcomeArray

    first, second = "qamomile.output.0.2", "qamomile.output.0.10"
    bits = [Bit(second, 0), Bit(first, 0)]
    raw = BackendResult(
        c_bits=bits,
        shots=OutcomeArray.from_readouts(np.array([[1, 0], [0, 1]], dtype=np.uint8)),
    )
    assert decode_qir_results(raw, 2, {first: ("bool", 1), second: ("bool", 1)}) == [
        {first: False, second: True},
        {first: True, second: False},
    ]
    empty = BackendResult(
        c_bits=[],
        shots=OutcomeArray.from_readouts(np.empty((3, 0), dtype=np.uint8)),
    )
    assert decode_qir_results(empty, 3, {}) == [{}, {}, {}]


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
        ([_bit("flag", 1)], {"flag": ("bool", 1)}, "width"),
        ([_bit("flag", -1)], {"flag": ("bool", 1)}, "width"),
        ([_bit("flag", True)], {"flag": ("bool", 1)}, "index"),
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
