"""Tests for consistent scalar and vector oracle control contracts."""

import numpy as np
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


def test_oracle_normalizes_numpy_control_count() -> None:
    """NumPy integer control counts use the same public integral contract."""
    oracle = qmc.opaque(
        "numpy_control_count",
        num_qubits=1,
        num_control_qubits=np.int64(2),
    )

    assert oracle.num_control_qubits == 2
    assert isinstance(oracle.num_control_qubits, int)
