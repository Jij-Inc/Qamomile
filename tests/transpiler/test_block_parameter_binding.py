"""Tests for operation-owned block parameter pairing."""

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.gate import ConcreteControlledU
from qamomile.circuit.ir.types.hamiltonian import ObservableType
from qamomile.circuit.ir.types.primitives import QubitType, UIntType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.block_parameter_binding import (
    pair_block_parameter_operands,
)
from qamomile.circuit.transpiler.passes.emit_support.value_resolver import (
    ValueResolver,
)


def test_pair_block_parameter_operands_skips_quantum_inputs() -> None:
    """Pair only classical/object inputs in declaration order."""
    target = Value(type=QubitType(), name="target")
    first = Value(type=UIntType(), name="first")
    second = Value(type=UIntType(), name="second")
    first_actual = Value(type=UIntType(), name="first_actual")
    second_actual = Value(type=UIntType(), name="second_actual")
    block = Block(input_values=[target, first, second])

    pairs = pair_block_parameter_operands(block, [first_actual, second_actual])

    assert pairs == [(first, first_actual), (second, second_actual)]


def test_emit_block_param_binding_uses_shared_pairing_contract() -> None:
    """Bind the same positional pairs that compile-time lowering sees."""
    target = Value(type=QubitType(), name="target")
    first = Value(type=UIntType(), name="first")
    second = Value(type=UIntType(), name="second")
    first_actual = Value(type=UIntType(), name="first_actual")
    second_actual = Value(type=UIntType(), name="second_actual")
    block = Block(input_values=[target, first, second])

    bindings = ValueResolver().bind_block_params(
        block,
        [first_actual, second_actual],
        {"first_actual": 1, "second_actual": 2},
    )

    assert bindings["first"] == 1
    assert bindings["second"] == 2


def test_controlled_param_operands_include_object_parameters() -> None:
    """Keep classical and object parameters in one signature-ordered list."""
    control = Value(type=QubitType(), name="control")
    target = Value(type=QubitType(), name="target")
    selector = Value(type=UIntType(), name="selector")
    observable = Value(type=ObservableType(), name="observable")
    op = ConcreteControlledU(
        operands=[control, target, selector, observable],
        results=[control.next_version(), target.next_version()],
        block=Block(),
    )

    assert op.param_operands == [selector, observable]
