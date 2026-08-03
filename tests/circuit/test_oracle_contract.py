"""Tests for consistent scalar and vector oracle control contracts."""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.value import ArrayValue, array_static_length

_CONTROLLED_ORACLE = qmc.Oracle(
    "controlled_vector_contract",
    num_qubits=2,
    num_control_qubits=1,
)
_VECTOR_ORACLE = qmc.Oracle(
    "vector_contract",
    signature=qmc.CallableSignature(
        inputs=[qmc.Vector[qmc.Qubit]],
        outputs=[qmc.Vector[qmc.Qubit]],
    ),
)


@qmc.qkernel
def _valid_vector_call() -> qmc.Vector[qmc.Qubit]:
    """Return a vector passed through a vector-signature oracle."""
    qubits = qmc.qubit_array(2, "qubits")
    return _VECTOR_ORACLE(qubits)


@qmc.qkernel
def _invalid_vector_call() -> qmc.Vector[qmc.Qubit]:
    """Attempt to bypass an oracle's explicit control requirement."""
    qubits = qmc.qubit_array(2, "qubits")
    return _CONTROLLED_ORACLE(qubits)


def test_vector_oracle_cannot_bypass_declared_controls() -> None:
    """Vector syntax rejects an oracle that requires explicit controls."""
    with pytest.raises(ValueError, match="requires 1 explicit control"):
        _invalid_vector_call.build()


def test_vector_oracle_preserves_vector_result_shape() -> None:
    """Vector oracle calls retain one vector output at runtime."""
    block = _valid_vector_call.build()

    assert len(block.output_values) == 1
    output = block.output_values[0]
    assert isinstance(output, ArrayValue)
    assert array_static_length(output) == 2
    assert output.type.label() == "QubitType"
