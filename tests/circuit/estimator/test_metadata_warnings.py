"""Tests for ResourceMetadata consistency warnings."""

import warnings

import pytest

from qamomile.circuit.estimator._catalog import (
    extract_gate_count_from_metadata as _extract_gate_count_from_metadata,
)
from qamomile.circuit.ir.operation.composite_gate import ResourceMetadata


class TestMetadataValidationWarnings:
    """Test that _extract_gate_count_from_metadata warns on inconsistent metadata."""

    def test_warns_when_total_set_but_subcategories_none(self):
        """total_gates=10, all sub-categories None -> warn."""
        meta = ResourceMetadata(total_gates=10)
        with pytest.warns(UserWarning, match="total_gates=10"):
            _extract_gate_count_from_metadata(meta)

    def test_warns_when_total_exceeds_known_subcategories(self):
        """total_gates=10, two_qubit_gates=3, rest None -> warn (3 < 10)."""
        meta = ResourceMetadata(total_gates=10, two_qubit_gates=3)
        with pytest.warns(UserWarning, match="total_gates"):
            result = _extract_gate_count_from_metadata(meta)
        assert result.total == 10
        assert result.two_qubit == 3

    def test_no_warning_when_subcategories_sum_to_total(self):
        """total_gates=10, sub-categories sum to 10 -> no warn."""
        meta = ResourceMetadata(
            total_gates=10,
            single_qubit_gates=4,
            two_qubit_gates=6,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _extract_gate_count_from_metadata(meta)

    def test_no_warning_when_total_not_set(self):
        """total_gates is None -> no warn."""
        meta = ResourceMetadata(single_qubit_gates=5)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _extract_gate_count_from_metadata(meta)

    def test_no_warning_when_all_fields_set(self):
        """All fields explicitly set -> no warn."""
        meta = ResourceMetadata(
            total_gates=10,
            single_qubit_gates=4,
            two_qubit_gates=5,
            multi_qubit_gates=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _extract_gate_count_from_metadata(meta)

    def test_total_gates_zero_is_respected(self):
        """total_gates=0 should be treated as explicit 0, not as None."""
        meta = ResourceMetadata(
            total_gates=0,
            single_qubit_gates=0,
            two_qubit_gates=0,
            multi_qubit_gates=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = _extract_gate_count_from_metadata(meta)
        assert result.total == 0
