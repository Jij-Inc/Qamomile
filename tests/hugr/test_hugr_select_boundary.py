"""Dependency-free validation of malformed SELECT at the native boundary."""

import pytest

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.hugr.lowerer import _lower_operation


def test_hugr_lowerer_rejects_select_without_quantum_operands() -> None:
    """Malformed SELECT fails before constructing any native graph nodes."""
    operation = SelectOperation(
        num_index_qubits=1,
        case_blocks=[Block(name="zero"), Block(name="one")],
    )

    with pytest.raises(EmitError, match="quantum operand/result arity") as error:
        _lower_operation(operation, None, {}, {}, {})

    assert error.value.operation == "SelectOperation"
