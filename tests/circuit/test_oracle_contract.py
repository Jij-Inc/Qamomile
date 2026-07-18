"""Tests for consistent scalar and vector oracle control contracts."""

import pytest

import qamomile.circuit as qmc

_CONTROLLED_ORACLE = qmc.Oracle(
    "controlled_vector_contract",
    num_qubits=2,
    num_control_qubits=1,
)


@qmc.qkernel
def _invalid_vector_call() -> qmc.Vector[qmc.Qubit]:
    """Attempt to bypass an oracle's explicit control requirement."""
    qubits = qmc.qubit_array(2, "qubits")
    return _CONTROLLED_ORACLE(qubits)


def test_vector_oracle_cannot_bypass_declared_controls() -> None:
    """Vector syntax rejects an oracle that requires explicit controls."""
    with pytest.raises(ValueError, match="requires 1 explicit control"):
        _invalid_vector_call.build()
