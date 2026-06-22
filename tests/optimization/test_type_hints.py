"""Tests for runtime-resolvable optimization type annotations."""

from __future__ import annotations

from typing import get_origin, get_type_hints

from qamomile.circuit.transpiler.job import SampleResult
from qamomile.optimization.binary_model.model import BinaryModel
from qamomile.optimization.converter import MathematicalProblemConverter


def test_sample_result_type_hints_are_runtime_resolvable():
    """SampleResult annotations can be resolved with ``typing.get_type_hints``."""
    converter_decode = get_type_hints(MathematicalProblemConverter.decode)
    converter_decode_to_binary = get_type_hints(
        MathematicalProblemConverter.decode_to_binary_sampleset
    )
    model_decode = get_type_hints(BinaryModel.decode_from_sampleresult)

    assert get_origin(converter_decode["samples"]) is SampleResult
    assert get_origin(converter_decode_to_binary["samples"]) is SampleResult
    assert get_origin(model_decode["result"]) is SampleResult
