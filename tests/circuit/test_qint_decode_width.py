"""Distinguish unresolved decoder widths from known empty quantum integers."""

import pytest

from qamomile.circuit.ir.operation import DecodeQIntOperation, MeasureQIntOperation
from qamomile.circuit.ir.types import BitType, QUIntType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.classical_executor import ClassicalExecutor
from qamomile.circuit.transpiler.errors import ExecutionError
from qamomile.circuit.transpiler.execution_context import ExecutionContext


@pytest.mark.parametrize("width", [0, 3, None])
def test_qint_width_properties_preserve_unknown_and_zero(width):
    """Derived widths return None only for unknown registers and bit arrays."""
    size = Value(type=UIntType(), name="size")
    if width is not None:
        size = size.with_const(width)
    register = Value(type=QUIntType(width=size), name="register")
    bits = ArrayValue(type=BitType(), name="bits", shape=(size,))
    assert MeasureQIntOperation(operands=[register]).num_bits == width
    assert DecodeQIntOperation(operands=[bits]).num_bits == width
    assert MeasureQIntOperation().num_bits is None
    assert DecodeQIntOperation().num_bits is None


def test_symbolic_qint_decode_raises_instead_of_returning_zero():
    """An unresolved array length cannot silently decode as the integer zero."""
    size = Value(type=UIntType(), name="size")
    bits = ArrayValue(type=BitType(), name="bits", shape=(size,))
    output = Value(type=UIntType(), name="decoded")
    decode = DecodeQIntOperation(operands=[bits], results=[output])
    results = {}
    with pytest.raises(ExecutionError, match="unknown bit width"):
        ClassicalExecutor()._execute_decode_qint(
            decode, ExecutionContext(), results, {}
        )
    assert output.uuid not in results


def test_zero_width_qint_decode_returns_zero():
    """A known empty array represents the integer zero without measured bits."""
    size = Value(type=UIntType(), name="size").with_const(0)
    bits = ArrayValue(type=BitType(), name="bits", shape=(size,))
    output = Value(type=UIntType(), name="decoded")
    decode = DecodeQIntOperation(operands=[bits], results=[output])
    results = {}
    ClassicalExecutor()._execute_decode_qint(decode, ExecutionContext(), results, {})
    assert results[output.uuid] == 0
