"""Preserve QInt widths and caller ownership across kernel input boundaries."""

from __future__ import annotations

import dataclasses

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.func_to_block import create_dummy_input
from qamomile.circuit.frontend.tracer import trace
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import (
    InvokeOperation,
    MeasureQIntOperation,
    ReturnOperation,
)
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.types import QUIntType, UIntType
from qamomile.circuit.ir.value import Value, packed_register_type_width
from qamomile.circuit.serialization import deserialize, serialize
from qamomile.circuit.serialization.validation import validate_qkernel_ir


@qmc.qkernel
def _identity(value: qmc.QInt) -> qmc.QInt:
    """Return the same packed register supplied by the caller.

    Args:
        value (qmc.QInt): Caller-owned packed quantum register.

    Returns:
        qmc.QInt: Unmodified input register.
    """
    return value


@qmc.qkernel
def _read(value: qmc.QInt) -> qmc.UInt:
    """Measure the caller-owned packed register.

    Args:
        value (qmc.QInt): Packed quantum register to consume.

    Returns:
        qmc.UInt: Unsigned integer decoded from the measured carriers.
    """
    return qmc.measure(value)


@qmc.qkernel
def _roundtrip(width: qmc.UInt) -> qmc.UInt:
    """Pass an all-ones register through QInt input and output boundaries.

    Args:
        width (qmc.UInt): Compile-time non-negative register width.

    Returns:
        qmc.UInt: Integer ``2**width - 1``, including zero for an empty register.
    """
    register = qmc.qubit_array(width, "register")
    for index in qmc.range(width):
        register[index] = qmc.x(register[index])
    return _read(_identity(qmc.cast(register, qmc.QInt)))


@qmc.qkernel
def _forward(value: qmc.QInt) -> qmc.QInt:
    """Return an input through another callable rather than a direct alias.

    Args:
        value (qmc.QInt): Caller-owned packed quantum register.

    Returns:
        qmc.QInt: Input register forwarded through the identity helper.
    """
    return _identity(value)


@qmc.qkernel
def _read_forwarded(value: qmc.QInt) -> qmc.UInt:
    """Measure an input after two identity calls advance its SSA versions.

    Args:
        value (qmc.QInt): Packed quantum register to forward and consume.

    Returns:
        qmc.UInt: Unsigned integer decoded from the original register state.
    """
    value = _forward(value)
    value = _forward(value)
    return qmc.measure(value)


@qmc.qkernel
def _through_forwarded() -> qmc.UInt:
    """Pass a concrete register into a helper with nested packed inputs.

    Returns:
        qmc.UInt: Integer one, with only the least significant carrier set.
    """
    register = qmc.qubit_array(3, "register")
    register[0] = qmc.x(register[0])
    return _read_forwarded(qmc.cast(register, qmc.QInt))


@qmc.qkernel
def _allocate() -> qmc.QInt:
    """Create a packed carrier without inheriting a formal input.

    Returns:
        qmc.QInt: Newly allocated three-qubit register initialized to zero.
    """
    return qmc.cast(qmc.qubit_array(3, "register"), qmc.QInt)


@qmc.qkernel
def _forward_allocation() -> qmc.QInt:
    """Forward qubits allocated by a nested callable.

    Returns:
        qmc.QInt: Zero-initialized three-qubit register returned by the allocator.
    """
    return _allocate()


@qmc.qkernel
def _measure_allocation() -> qmc.UInt:
    """Measure a register allocated two callable boundaries away.

    Returns:
        qmc.UInt: Integer zero decoded from the newly allocated register.
    """
    return qmc.measure(_forward_allocation())


@qmc.qkernel
def _scalar(value: qmc.Qubit) -> qmc.Qubit:
    """Return a single qubit without treating it as a packed register.

    Args:
        value (qmc.Qubit): Caller-owned scalar qubit.

    Returns:
        qmc.Qubit: Unmodified input qubit.
    """
    return value


@qmc.qkernel
def _qubit_as_qint() -> qmc.UInt:
    """Pass a scalar qubit where a packed register is required.

    Returns:
        qmc.UInt: Unreachable because the argument type is rejected.

    Raises:
        TypeError: If the scalar qubit reaches the QInt input boundary.
    """
    return _read(qmc.qubit("q"))


@qmc.qkernel
def _vector_as_qint() -> qmc.UInt:
    """Pass an unpacked vector where a packed register is required.

    Returns:
        qmc.UInt: Unreachable because the argument type is rejected.

    Raises:
        TypeError: If the unpacked vector reaches the QInt input boundary.
    """
    return _read(qmc.qubit_array(2, "q"))


@qmc.qkernel
def _qint_as_qubit() -> qmc.Bit:
    """Pass a packed register where one scalar qubit is required.

    Returns:
        qmc.Bit: Unreachable because the argument type is rejected.

    Raises:
        TypeError: If the QInt reaches the scalar-qubit input boundary.
    """
    number = qmc.cast(qmc.qubit_array(2, "q"), qmc.QInt)
    return qmc.measure(_scalar(number))


def test_qint_parameter_remains_symbolic_without_a_caller() -> None:
    """An unbound packed input is quantum and never becomes an empty register."""
    block = _read.build()
    assert packed_register_type_width(block.input_values[0].type) is None
    assert block.input_values[0].metadata.cast is None
    assert block.param_slots == ()
    assert not any(isinstance(op, QInitOperation) for op in block.operations)

    restored = deserialize(serialize(_read)).block
    assert packed_register_type_width(restored.input_values[0].type) is None
    assert restored.input_values[0].metadata.cast is None


@pytest.mark.parametrize("width", [0, 1, 3, 65])
def test_qint_call_specialization_keeps_exact_width_and_ownership(width: int) -> None:
    """Call sites specialize packed widths without allocation or a global cap."""
    block = _roundtrip.build(width=width)
    calls = [op for op in block.operations if isinstance(op, InvokeOperation)]
    assert len(calls) == 2
    for call in calls:
        body = call.definition.body
        assert packed_register_type_width(body.input_values[0].type) == width
        assert body.input_values[0].metadata.cast is None
        assert body.param_slots == ()
        assert not any(isinstance(op, QInitOperation) for op in body.operations)
    assert calls[0].results[0].type == QUIntType(width)


@pytest.mark.parametrize("kernel", [_qubit_as_qint, _vector_as_qint, _qint_as_qubit])
def test_qint_input_mismatches_fail_at_the_call_boundary(kernel) -> None:
    """Packed inputs reject unpacked handles, and scalar inputs reject QInts."""
    with pytest.raises(TypeError, match="declared as"):
        kernel.build()


@pytest.mark.parametrize(
    "transform", ["control", "select", "power", "controlled_inverse"]
)
@pytest.mark.parametrize("symbolic_width", [False, True])
@pytest.mark.parametrize("keyword_target", [False, True])
@pytest.mark.parametrize("target_width", [0, 3])
def test_qint_controlled_targets_fail_without_consuming_handles(
    transform: str, symbolic_width: bool, keyword_target: bool, target_width: int
) -> None:
    """Reject packed targets before emitting operations or transferring ownership.

    Args:
        transform (str): Controlled operation variant to exercise.
        symbolic_width (bool): Whether the control or index width is unresolved.
        keyword_target (bool): Whether to pass the packed target by keyword.
        target_width (int): Number of qubits in the packed target.
    """
    with trace() as tracer:
        width = qmc.uint("width") if symbolic_width else 1
        prefix = qmc.qubit_array(width, "prefix")
        target = qmc.cast(qmc.qubit_array(target_width, "target"), qmc.QInt)
        if transform == "select":
            operation = qmc.select([_identity, _identity], num_index_qubits=width)
        else:
            kernel = (
                qmc.inverse(_identity)
                if transform == "controlled_inverse"
                else _identity
            )
            operation = qmc.control(kernel, num_controls=width)
        modifiers = {"power": 2} if transform == "power" else {}
        operations_before = list(tracer.operations)

        with pytest.raises(
            TypeError, match=r"control\(\).*select\(\).*QInt.*direct qkernel call"
        ):
            if keyword_target:
                operation(prefix, value=target, **modifiers)
            else:
                operation(prefix, target, **modifiers)

        assert tracer.operations == operations_before
        assert not prefix._consumed
        assert not target._consumed
        qmc.measure(prefix)
        qmc.measure(_identity(target))


@pytest.mark.parametrize("transform", ["control", "select"])
def test_qint_symbolic_control_prefix_fails_without_consuming_handles(
    transform: str,
) -> None:
    """Reject a packed handle among symbolic controls before consuming any input.

    Args:
        transform (str): Whether to exercise a controlled gate or SELECT.
    """
    with trace() as tracer:
        width = qmc.uint("width")
        scalar_prefix = qmc.qubit("scalar_prefix")
        packed_prefix = qmc.cast(qmc.qubit_array(2, "packed_prefix"), qmc.QInt)
        target = qmc.qubit("target")
        operation = (
            qmc.select([_scalar, _scalar], num_index_qubits=width)
            if transform == "select"
            else qmc.control(_scalar, num_controls=width)
        )
        operations_before = list(tracer.operations)

        with pytest.raises(
            TypeError, match=r"control\(\).*select\(\).*QInt.*direct qkernel call"
        ):
            operation(scalar_prefix, packed_prefix, target)

        assert tracer.operations == operations_before
        assert not scalar_prefix._consumed
        assert not packed_prefix._consumed
        assert not target._consumed
        qmc.measure(qmc.h(scalar_prefix))
        qmc.measure(_identity(packed_prefix))
        qmc.measure(qmc.x(target))


@pytest.mark.parametrize("width", [0, 3])
def test_qint_identity_inverse_remains_supported(width: int) -> None:
    """Keep ordinary packed-input inversion outside the controlled-call restriction.

    Args:
        width (int): Number of qubits in the packed register.
    """
    with trace():
        target = qmc.cast(qmc.qubit_array(width, "target"), qmc.QInt)
        result = qmc.inverse(_identity)(target)
        assert isinstance(result, qmc.QInt)
        assert packed_register_type_width(result.value.type) == width
        qmc.measure(result)


@pytest.mark.parametrize("width", [None, 0, 1, 65])
def test_serialization_accepts_formal_qint_inputs_without_cast_metadata(width) -> None:
    """Only formal packed inputs can defer physical carriers to their caller."""
    shape = None if width is None else (width,)
    formal = create_dummy_input(qmc.QInt, "number", emit_init=False, shape=shape).value
    output = Value(type=UIntType(), name="output")
    block = Block(
        input_values=[formal],
        output_values=[output],
        operations=[MeasureQIntOperation(operands=[formal], results=[output])],
    )
    validate_qkernel_ir(block)
    assert packed_register_type_width(formal.type) == width
    assert formal.metadata.cast is None

    block.input_values = []
    with pytest.raises(ValueError, match="requires cast metadata"):
        validate_qkernel_ir(block)


def test_serialization_still_rejects_malformed_formal_qint_metadata() -> None:
    """The formal-input exception does not suppress invalid present metadata."""
    formal = Value(type=QUIntType(1), name="number").with_cast_metadata(
        source_uuid="", source_logical_id="", qubit_uuids=(), qubit_logical_ids=()
    )
    output = Value(type=UIntType(), name="output")
    block = Block(
        input_values=[formal],
        output_values=[output],
        operations=[MeasureQIntOperation(operands=[formal], results=[output])],
    )
    with pytest.raises(ValueError, match="source identity is incomplete"):
        validate_qkernel_ir(block)


def test_serialization_preserves_nested_qint_input_forwarding() -> None:
    """Direct calls retain caller-owned carriers without synthetic metadata."""
    restored = deserialize(serialize(_through_forwarded))
    call = next(
        op for op in restored.block.operations if isinstance(op, InvokeOperation)
    )
    body = call.definition.body
    measurement = next(
        op for op in body.operations if isinstance(op, MeasureQIntOperation)
    )
    assert measurement.operands[0].metadata.cast is None
    assert measurement.operands[0].uuid != body.input_values[0].uuid
    assert serialize(restored) == serialize(_through_forwarded)


def test_serialization_rejects_metadata_free_qint_from_a_bodyless_call() -> None:
    """The callable-boundary exception requires a validated implementation."""
    restored = deserialize(serialize(_through_forwarded))
    call = next(
        op for op in restored.block.operations if isinstance(op, InvokeOperation)
    )
    body = call.definition.body
    calls = [op for op in body.operations if isinstance(op, InvokeOperation)]
    calls[-1].definition.body = None
    with pytest.raises(ValueError, match="requires cast metadata"):
        validate_qkernel_ir(restored.block)


def test_serialization_rejects_qint_call_output_without_a_producer() -> None:
    """A typed Return alone cannot establish a packed callable carrier."""
    restored = deserialize(serialize(_through_forwarded))
    call = next(
        op for op in restored.block.operations if isinstance(op, InvokeOperation)
    )
    forwarded = next(
        op for op in call.definition.body.operations if isinstance(op, InvokeOperation)
    )
    fake = Value(type=QUIntType(3), name="unproduced")
    forwarded.definition.body.output_values = [fake]
    forwarded.definition.body.operations = [ReturnOperation(operands=[fake])]
    with pytest.raises(ValueError, match="requires cast metadata"):
        validate_qkernel_ir(restored.block)


def test_serialization_accepts_call_carriers_allocated_in_nested_callees() -> None:
    """The producer proof permits allocated QInts as well as input forwarding."""
    restored = deserialize(serialize(_measure_allocation))
    call = next(
        op for op in restored.block.operations if isinstance(op, InvokeOperation)
    )
    measurement = next(
        op for op in restored.block.operations if isinstance(op, MeasureQIntOperation)
    )
    carrier = dataclasses.replace(
        call.results[0],
        metadata=dataclasses.replace(call.results[0].metadata, cast=None),
    )
    call.results[0] = carrier
    measurement.operands[0] = carrier
    validate_qkernel_ir(restored.block)


def test_serialization_rejects_qint_call_input_without_a_carrier() -> None:
    """A genuine identity body cannot legitimize an unproduced caller argument."""
    restored = deserialize(serialize(_through_forwarded))
    call = next(
        op for op in restored.block.operations if isinstance(op, InvokeOperation)
    )
    call.operands[0] = Value(type=QUIntType(3), name="unproduced")
    with pytest.raises(ValueError, match="requires cast metadata"):
        validate_qkernel_ir(restored.block)


@pytest.mark.parametrize("width", [0, 1, 3])
def test_qint_parameters_execute_on_qiskit(width: int) -> None:
    """The engine-independent input change preserves integer circuit decoding."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    transpiler = QiskitTranspiler()
    executable = transpiler.transpile(_roundtrip, bindings={"width": width})
    result = executable.sample(transpiler.executor(), shots=4).result()
    assert result.results == [((1 << width) - 1, 4)]


def test_qint_parameters_above_64_bits_still_compile_on_qiskit() -> None:
    """The HUGR output limit does not restrict another engine's QInt inputs."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    executable = QiskitTranspiler().transpile(_roundtrip, bindings={"width": 65})
    assert executable.compiled_quantum[0].circuit.num_qubits == 65
