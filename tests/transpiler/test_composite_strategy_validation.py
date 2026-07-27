"""Tests for strict composite decomposition strategy validation."""

import pytest

from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.passes.emit_support.composite_gate_emission import (
    _qft_truncation_depth,
)


@pytest.mark.parametrize("strategy", ["approximate", "approximate_kx", "unknown"])
def test_qft_strategy_rejects_malformed_names(strategy: str) -> None:
    """Malformed strategy names cannot silently select exact or depth three."""
    with pytest.raises(EmitError, match="Invalid QFT strategy"):
        _qft_truncation_depth(strategy, "QFT")


def test_qft_strategy_accepts_exact_and_positive_depth() -> None:
    """Only the documented exact and approximate forms are accepted."""
    assert _qft_truncation_depth(None, "QFT") is None
    assert _qft_truncation_depth("exact", "QFT") is None
    assert _qft_truncation_depth("approximate_k3", "QFT") == 3
