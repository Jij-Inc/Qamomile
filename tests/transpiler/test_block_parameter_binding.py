"""Tests for operation-owned block parameter pairing."""

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.gate import ConcreteControlledU
from qamomile.circuit.ir.types.hamiltonian import ObservableType
from qamomile.circuit.ir.types.primitives import QubitType, UIntType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.block_parameter_binding import (
    pair_block_operands,
    pair_block_parameter_operands,
)
from qamomile.circuit.transpiler.passes.emit_support.controlled_emission import (
    _bind_block_inputs,
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


def test_pair_block_operands_reweaves_interleaved_formals() -> None:
    """Grouped actual pools align with an interleaved block signature."""
    first_qubit = Value(type=QubitType(), name="first_qubit")
    selector = Value(type=UIntType(), name="selector")
    second_qubit = Value(type=QubitType(), name="second_qubit")
    count = Value(type=UIntType(), name="count")
    actual_first = Value(type=QubitType(), name="actual_first")
    actual_second = Value(type=QubitType(), name="actual_second")
    actual_selector = Value(type=UIntType(), name="actual_selector")
    actual_count = Value(type=UIntType(), name="actual_count")
    block = Block(input_values=[first_qubit, selector, second_qubit, count])

    pairs = pair_block_operands(
        block,
        [actual_first, actual_second, actual_selector, actual_count],
    )

    assert pairs == [
        (first_qubit, actual_first),
        (selector, actual_selector),
        (second_qubit, actual_second),
        (count, actual_count),
    ]


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


def test_emit_binding_overrides_outer_same_name_with_actual_operand() -> None:
    """A fresh inner parameter cannot capture an outer same-name binding."""
    target = Value(type=QubitType(), name="target")
    inner_theta = Value(type=UIntType(), name="theta").with_parameter("theta")
    actual_theta = Value(type=UIntType(), name="actual_theta").with_const(2)
    block = Block(input_values=[target, inner_theta])
    resolver = ValueResolver()

    bindings = resolver.bind_block_params(
        block,
        [actual_theta],
        {"theta": 1},
    )

    assert bindings[inner_theta.uuid] == 2
    assert bindings["theta"] == 2
    assert resolver.resolve_classical_value(inner_theta, bindings) == 2


def test_missing_inner_actual_does_not_capture_outer_same_name() -> None:
    """An unpaired inner formal remains symbolic across a name collision."""
    target = Value(type=QubitType(), name="target")
    inner_theta = Value(type=UIntType(), name="theta").with_parameter("theta")
    block = Block(input_values=[target, inner_theta])
    resolver = ValueResolver()

    bindings = resolver.bind_block_params(block, [], {"theta": 1})

    assert "theta" not in bindings
    assert inner_theta.uuid not in bindings
    assert resolver.resolve_classical_value(inner_theta, bindings) is None


def test_missing_actual_clears_distinct_provenance_and_display_names() -> None:
    """A fresh formal clears both name channels before emit resolution."""
    target = Value(type=QubitType(), name="target")
    inner_theta = Value(type=UIntType(), name="display_theta").with_parameter(
        "parameter_theta"
    )
    block = Block(input_values=[target, inner_theta])
    resolver = ValueResolver()

    bindings = resolver.bind_block_params(
        block,
        [],
        {"display_theta": 1, "parameter_theta": 2},
    )

    assert "display_theta" not in bindings
    assert "parameter_theta" not in bindings
    assert inner_theta.uuid not in bindings
    assert resolver.resolve_classical_value(inner_theta, bindings) is None


def test_generic_block_binder_overrides_outer_same_name() -> None:
    """The controlled composite binder applies the same fresh-scope rule."""

    class EmitPass:
        """Provide the resolver surface used by ``_bind_block_inputs``."""

        def __init__(self) -> None:
            """Initialize a real value resolver."""
            self._resolver = ValueResolver()

        def _get_or_create_parameter(self, name: str, uuid: str) -> object:
            """Return a stable stand-in for an unresolved backend parameter.

            Args:
                name (str): Parameter name.
                uuid (str): Parameter UUID.

            Returns:
                object: Tuple-shaped stand-in used only if resolution falls
                through to backend parameter creation.
            """
            return (name, uuid)

    target = Value(type=QubitType(), name="target")
    inner_theta = Value(type=UIntType(), name="theta").with_parameter("theta")
    actual_theta = Value(type=UIntType(), name="actual_theta").with_const(2)
    block = Block(input_values=[target, inner_theta])

    bindings = _bind_block_inputs(
        EmitPass(),
        block,
        [Value(type=QubitType(), name="actual_target"), actual_theta],
        {"theta": 1},
    )

    assert bindings[inner_theta.uuid] == 2
    assert bindings["theta"] == 2


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
