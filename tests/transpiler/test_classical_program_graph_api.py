"""Verify public host execution helpers without backend-specific semantics."""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.types import FloatType, UIntType
from qamomile.circuit.ir.value import ArrayValue, DictValue, TupleValue, Value
from qamomile.circuit.transpiler import (
    ClassicalExecutor,
    ExecutionContext,
    ExecutionError,
    QamomileCompiler,
    inline_callables,
)


@qmc.qkernel
def _flip(q: qmc.Qubit) -> qmc.Qubit:
    """Provide a quantum call boundary to preserve during selective inlining.

    Args:
        q (qmc.Qubit): State to flip.

    Returns:
        qmc.Qubit: State after applying the X gate.
    """
    return qmc.x(q)


@qmc.qkernel
def _double(value: qmc.Float) -> qmc.Float:
    """Provide a classical helper whose body is needed by a host interpreter.

    Args:
        value (qmc.Float): Classical input to double.

    Returns:
        qmc.Float: Twice the input value.
    """
    return value * 2.0


@qmc.qkernel
def _with_calls(value: qmc.Float) -> tuple[qmc.Bit, qmc.Float]:
    """Call independent quantum and classical helpers.

    Args:
        value (qmc.Float): Input passed to the classical helper.

    Returns:
        tuple: Independent quantum and classical results:
            measured (qmc.Bit): Measurement of the flipped zero state.
            doubled (qmc.Float): Twice the classical input.
    """
    q = qmc.qubit("q")
    q = _flip(q)
    doubled = _double(value)
    return qmc.measure(q), doubled


def test_public_inlining_selector_preserves_unselected_callable_identity():
    """A selected host helper is expanded while an unrelated quantum call remains."""
    source = QamomileCompiler().prepare(_with_calls, parameters=["value"]).entrypoint
    calls = [op for op in source.operations if isinstance(op, InvokeOperation)]
    quantum, classical = calls
    lowered = inline_callables(
        source,
        body_selector=lambda call: (
            call.effective_body() if call.target == classical.target else None
        ),
    )
    remaining = [op for op in lowered.operations if isinstance(op, InvokeOperation)]
    assert len(remaining) == 1
    assert remaining[0].target == quantum.target
    assert remaining[0].definition is quantum.definition
    assert remaining[0].operands[0].uuid == quantum.operands[0].uuid
    assert len([op for op in source.operations if isinstance(op, InvokeOperation)]) == 2


def test_public_value_resolution_preserves_containers_and_identity():
    """Computed values and array slots resolve inside structured tuples and dicts."""
    computed = Value(type=FloatType(), name="same_name")
    array = ArrayValue(type=FloatType(), name="values").with_parameter("values")
    element = Value(
        type=FloatType(),
        name="element",
        parent_array=array,
        element_indices=(Value(type=UIntType(), name="index").with_const(1),),
    )
    key = Value(type=UIntType(), name="key").with_const(3)
    dictionary = DictValue(name="mapping", entries=((key, computed),))
    output = TupleValue(name="outputs", elements=(dictionary, element))
    context = ExecutionContext(
        {computed.uuid: 0.25, "same_name": 99.0, "values": [1.0, 2.0]}
    )
    resolved = ClassicalExecutor().resolve_value(output, context)
    assert resolved[0] == pytest.approx({3: 0.25}, abs=1e-12)
    assert resolved[1] == pytest.approx(2.0, abs=1e-12)


def test_public_value_resolution_rejects_unavailable_classical_values():
    """Missing computed outputs fail through the shared execution error contract."""
    with pytest.raises(ExecutionError, match="not found"):
        ClassicalExecutor().resolve_value(
            Value(type=FloatType(), name="missing"), ExecutionContext()
        )
