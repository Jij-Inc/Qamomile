"""Tests for the public QInt quantum-register handle."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import (
    CastOperation,
    DecodeQIntOperation,
    MeasureQIntOperation,
    MeasureVectorOperation,
)
from qamomile.circuit.ir.operation.control_flow import IfOperation
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.types import BitType, QubitType, QUIntType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.serialization import deserialize, serialize
from qamomile.circuit.transpiler.errors import QubitConsumedError, SeparationError
from qamomile.circuit.transpiler.passes.separate import lower_operations


@qmc.qkernel
def _measure_two_bit_qint() -> qmc.UInt:
    """Measure a two-carrier unsigned quantum integer."""
    register = qmc.qubit_array(2, "register")
    return qmc.measure(qmc.cast(register, qmc.QInt))


@qmc.qkernel
def _measure_non_palindromic_qint() -> qmc.UInt:
    """Measure carriers zero and one as the unsigned integer three."""
    register = qmc.qubit_array(3, "register")
    register[0] = qmc.x(register[0])
    register[1] = qmc.x(register[1])
    return qmc.measure(qmc.cast(register, qmc.QInt))


@qmc.qkernel
def _measure_zero_width_qint() -> qmc.UInt:
    """Measure the unique value represented by an empty QInt register."""
    return qmc.measure(qmc.cast(qmc.qubit_array(0, "register"), qmc.QInt))


@qmc.qkernel
def _pack_qint(register: qmc.Vector[qmc.Qubit]) -> qmc.QInt:
    """Reinterpret a caller-owned register as a QInt."""
    return qmc.cast(register, qmc.QInt)


@qmc.qkernel
def _measure_invoked_qint() -> qmc.UInt:
    """Measure a QInt returned from another qkernel."""
    register = qmc.qubit_array(3, "register")
    register[0] = qmc.x(register[0])
    register[1] = qmc.x(register[1])
    return qmc.measure(_pack_qint(register))


def test_qint_cast_and_measure_build_typed_ir() -> None:
    """QInt cast and measurement use QUInt and UInt IR types."""
    block = _measure_two_bit_qint.build()
    cast_op = next(op for op in block.operations if isinstance(op, CastOperation))
    measure_op = next(
        op for op in block.operations if isinstance(op, MeasureQIntOperation)
    )

    assert cast_op.target_type == QUIntType(width=2)
    assert cast_op.results[0].type == QUIntType(width=2)
    assert measure_op.operands == cast_op.results
    assert isinstance(measure_op.results[0].type, UIntType)
    assert measure_op.num_bits == 2


def test_qint_num_bits_is_derived_from_operands() -> None:
    """QInt operations derive num_bits from their operands and store nothing."""
    concrete = Value(type=QUIntType(width=3), name="register")
    assert MeasureQIntOperation(operands=[concrete]).num_bits == 3

    width = Value(type=UIntType(), name="width")
    symbolic = Value(type=QUIntType(width=width), name="register")
    assert MeasureQIntOperation(operands=[symbolic]).num_bits is None

    size = Value(type=UIntType(), name="size").with_const(4)
    bits = ArrayValue(type=BitType(), name="bits", shape=(size,))
    assert DecodeQIntOperation(operands=[bits]).num_bits == 4

    with pytest.raises(TypeError, match="num_bits"):
        MeasureQIntOperation(operands=[concrete], num_bits=3)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="num_bits"):
        DecodeQIntOperation(operands=[bits], num_bits=4)  # type: ignore[call-arg]


def test_qint_is_valid_qkernel_return_type() -> None:
    """A qkernel may return a QInt whose concrete width comes from its cast."""

    @qmc.qkernel
    def kernel() -> qmc.QInt:
        return qmc.cast(qmc.qubit_array(3, "register"), qmc.QInt)

    block = kernel.build()
    assert block.output_values[0].type == QUIntType(width=3)


def test_qint_plan_lowering_preserves_decoder_result() -> None:
    """Planning splits QInt measurement into vector measure and UInt decode."""
    block = _measure_two_bit_qint.build()
    measured_result = next(
        op.results[0] for op in block.operations if isinstance(op, MeasureQIntOperation)
    )

    lowered = lower_operations(block)
    measure_vector = next(
        op for op in lowered.operations if isinstance(op, MeasureVectorOperation)
    )
    decode = next(
        op for op in lowered.operations if isinstance(op, DecodeQIntOperation)
    )

    assert decode.operands == measure_vector.results
    assert decode.results == [measured_result]
    assert decode.num_bits == 2


@qmc.qkernel
def _measure_merged_symbolic_width_qint(n: qmc.UInt) -> qmc.UInt:
    """Measure a symbolic-width QInt merged from equivalent runtime branches."""
    register = qmc.qubit_array(n, "register")
    register[0] = qmc.x(register[0])
    selector = qmc.measure(qmc.qubit("selector"))
    if selector:
        value = qmc.cast(register, qmc.QInt)
    else:
        value = qmc.cast(register, qmc.QInt)
    return qmc.measure(value)


def _measurement_block(
    qint: Value, operations: list[Any] | None = None
) -> tuple[Block, Value]:
    """Build a block that measures ``qint`` after ``operations``.

    Args:
        qint (Value): Packed register operand of the measurement.
        operations (list[Any] | None): Operations preceding the measurement.
            Defaults to none.

    Returns:
        tuple[Block, Value]: The block and its unsigned-integer result.
    """
    result = Value(type=UIntType(), name="result")
    block = Block(
        operations=[
            *(operations or []),
            MeasureQIntOperation(operands=[qint], results=[result]),
            ReturnOperation(operands=[result]),
        ],
        output_values=[result],
    )
    return block, result


def test_lower_operations_rejects_unresolvable_symbolic_qint() -> None:
    """Plan lowering refuses to synthesize an empty register for unknown width."""
    width = Value(type=UIntType(), name="n")
    qint = Value(type=QUIntType(width=width), name="qint").with_cast_metadata(
        source_uuid="missing",
        source_logical_id="missing",
        qubit_uuids=(),
        qubit_logical_ids=(),
    )
    block, _ = _measurement_block(qint)

    with pytest.raises(SeparationError, match="no resolvable carrier"):
        lower_operations(block)


def test_lower_operations_accepts_known_zero_width_qint_without_source() -> None:
    """A register whose type declares width zero lowers to an empty measurement."""
    qint = Value(type=QUIntType(width=0), name="qint").with_cast_metadata(
        source_uuid="missing",
        source_logical_id="missing",
        qubit_uuids=(),
        qubit_logical_ids=(),
    )
    block, result = _measurement_block(qint)

    lowered = lower_operations(block)

    measure_vector = next(
        op for op in lowered.operations if isinstance(op, MeasureVectorOperation)
    )
    decode = next(
        op for op in lowered.operations if isinstance(op, DecodeQIntOperation)
    )
    assert measure_vector.operands[0].shape[0].get_const() == 0
    assert decode.operands == measure_vector.results
    assert decode.results == [result]
    assert decode.num_bits == 0


def test_lower_operations_rejects_symbolic_cast_source_width() -> None:
    """A cast source whose length is still symbolic cannot be measured."""
    width = Value(type=UIntType(), name="n")
    source = ArrayValue(type=QubitType(), name="register", shape=(width,))
    qint = Value(type=QUIntType(width=width), name="qint").with_cast_metadata(
        source_uuid=source.uuid,
        source_logical_id=source.logical_id,
        qubit_uuids=(),
        qubit_logical_ids=(),
    )
    cast_op = CastOperation(
        operands=[source],
        results=[qint],
        source_type=QubitType(),
        target_type=qint.type,
        qubit_mapping=[],
    )
    block, _ = _measurement_block(qint, [cast_op])

    with pytest.raises(SeparationError, match="symbolic at plan time"):
        lower_operations(block)


def test_lower_operations_resolves_nested_cast_source_by_metadata() -> None:
    """A merged register finds its source through nested-cast metadata."""
    restored = deserialize(serialize(_measure_merged_symbolic_width_qint))
    block = restored.build(n=3)
    measurement = next(
        op for op in block.operations if isinstance(op, MeasureQIntOperation)
    )
    merged_qint = measurement.operands[0]
    assert not merged_qint.get_cast_qubit_uuids()

    lowered = lower_operations(block)

    measure_vector = next(
        op for op in lowered.operations if isinstance(op, MeasureVectorOperation)
    )
    decode = next(
        op for op in lowered.operations if isinstance(op, DecodeQIntOperation)
    )
    assert measure_vector.operands[0].uuid == merged_qint.get_cast_source_uuid()
    assert decode.num_bits == 3


def test_qint_resource_estimation_counts_all_carriers() -> None:
    """Resource estimation treats QInt measurement as destructive on every carrier."""

    @qmc.qkernel
    def kernel() -> qmc.UInt:
        register = qmc.qubit_array(3, "register")
        register[0] = qmc.h(register[0])
        return qmc.measure(qmc.cast(register, qmc.QInt))

    estimate = kernel.estimate_resources()
    assert estimate.gates.total == 1
    assert estimate.measurements.total == 3
    assert estimate.depth.depth == 2
    assert estimate.width.peak_qubits == 3


def test_qint_bit_order_executes_across_engines(sdk_transpiler: Any) -> None:
    """Every SDK engine decodes carrier zero as the least-significant bit."""
    transpiler = sdk_transpiler.transpiler
    executable = transpiler.transpile(_measure_non_palindromic_qint)
    result = executable.sample(transpiler.executor(), shots=16).result()
    assert result.results == [(3, 16)], (
        f"{sdk_transpiler.engine_name}: got {result.results}"
    )


def test_qint_returned_from_qkernel_executes_across_engines(
    sdk_transpiler: Any,
) -> None:
    """QInt carrier metadata survives qkernel invocation and inlining."""
    transpiler = sdk_transpiler.transpiler
    executable = transpiler.transpile(_measure_invoked_qint)
    result = executable.sample(transpiler.executor(), shots=16).result()
    assert result.results == [(3, 16)], (
        f"{sdk_transpiler.engine_name}: got {result.results}"
    )


def test_zero_width_qint_executes_across_engines(sdk_transpiler: Any) -> None:
    """Every SDK engine returns zero for an empty QInt carrier sequence."""
    transpiler = sdk_transpiler.transpiler
    executable = transpiler.transpile(_measure_zero_width_qint)
    result = executable.sample(transpiler.executor(), shots=4).result()
    assert result.results == [(0, 4)], (
        f"{sdk_transpiler.engine_name}: got {result.results}"
    )


def test_qint_slice_carriers_keep_source_order() -> None:
    """A strided cast records root carriers in LSB-to-MSB source order."""

    @qmc.qkernel
    def kernel() -> qmc.UInt:
        register = qmc.qubit_array(4, "register")
        return qmc.measure(qmc.cast(register[1::2], qmc.QInt))

    block = kernel.build()
    root = next(
        op.results[0] for op in block.operations if isinstance(op, QInitOperation)
    )
    cast_op = next(op for op in block.operations if isinstance(op, CastOperation))

    assert cast_op.qubit_mapping == [f"{root.uuid}_1", f"{root.uuid}_3"]
    assert cast_op.results[0].get_cast_qubit_logical_ids() == (
        f"{root.logical_id}_1",
        f"{root.logical_id}_3",
    )


def test_qint_runtime_branch_merge_preserves_carriers() -> None:
    """Merging equivalent branch-local QInts retains ordered cast metadata."""

    @qmc.qkernel
    def kernel() -> qmc.UInt:
        register = qmc.qubit_array(2, "register")
        selector = qmc.measure(qmc.qubit("selector"))
        if selector:
            value = qmc.cast(register, qmc.QInt)
        else:
            value = qmc.cast(register, qmc.QInt)
        return qmc.measure(value)

    block = kernel.build()
    branch = next(op for op in block.operations if isinstance(op, IfOperation))
    merged_qint = next(
        result for result in branch.results if isinstance(result.type, QUIntType)
    )
    assert merged_qint.get_cast_qubit_uuids()
    measurement = next(
        op for op in block.operations if isinstance(op, MeasureQIntOperation)
    )
    assert measurement.operands == [merged_qint]


def test_plain_merge_of_quantum_arrays_of_different_static_width_is_allowed() -> None:
    """Branch-dependent widths stay legal when the array is not packed-cast."""

    @qmc.qkernel
    def kernel(flag: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        if flag == 1:
            work = qmc.qubit_array(4, "large")
        else:
            work = qmc.qubit_array(2, "small")
        return qmc.measure(work)

    kernel.build()


def test_packed_cast_of_view_within_both_branch_widths_is_allowed() -> None:
    """A literal view covering slots present in both branches may be cast."""

    @qmc.qkernel
    def kernel(flag: qmc.UInt) -> qmc.UInt:
        if flag == 1:
            work = qmc.qubit_array(4, "large")
        else:
            work = qmc.qubit_array(2, "small")
        return qmc.measure(qmc.cast(work[0:2], qmc.QInt))

    kernel.build()


def test_if_merge_allows_symbolic_and_static_quantum_array_lengths() -> None:
    """A symbolic-vs-static width mismatch is undecidable at trace time."""

    @qmc.qkernel
    def kernel(n: qmc.UInt) -> qmc.UInt:
        selector = qmc.measure(qmc.qubit("selector"))
        if selector:
            work = qmc.qubit_array(n, "symbolic")
        else:
            work = qmc.qubit_array(2, "static")
        return qmc.measure(qmc.cast(work, qmc.QInt))

    kernel.build()


def test_qint_cast_consumes_source_vector() -> None:
    """Reusing a vector after casting it to QInt raises a consume error."""

    @qmc.qkernel
    def kernel() -> qmc.QInt:
        register = qmc.qubit_array(2, "register")
        value = qmc.cast(register, qmc.QInt)
        _ = register[0]
        return value

    with pytest.raises(QubitConsumedError):
        kernel.build()


@pytest.mark.parametrize("int_bits", [0, 1, -1, None, False])
def test_qint_cast_rejects_fixed_point_layout_argument(int_bits: Any) -> None:
    """QInt rejects every explicit layout keyword, including zero and None."""

    @qmc.qkernel
    def kernel() -> qmc.QInt:
        """Attempt an unsupported explicit QInt layout.

        Returns:
            qmc.QInt: Unreachable because the keyword is rejected.
        """
        register = qmc.qubit_array(2, "register")
        return qmc.cast(register, qmc.QInt, int_bits=int_bits)

    with pytest.raises(TypeError, match="only supported when casting to QFixed"):
        kernel.build()


@pytest.mark.parametrize("int_bits", [0, 1])
def test_qfixed_cast_keeps_default_and_explicit_integer_bits(int_bits: int) -> None:
    """The low two set bits decode as 3 / 2**(3 - int_bits), defaulting to 3/8."""
    from qamomile.qiskit import QiskitTranspiler

    @qmc.qkernel
    def default_layout() -> qmc.Float:
        """Measure a register with the default all-fractional layout.

        Returns:
            qmc.Float: The fraction three eighths.
        """
        register = qmc.qubit_array(3, "register")
        register[0] = qmc.x(register[0])
        register[1] = qmc.x(register[1])
        return qmc.measure(qmc.cast(register, qmc.QFixed))

    @qmc.qkernel
    def explicit_layout() -> qmc.Float:
        """Measure the same carriers with an explicit fixed-point layout.

        Returns:
            qmc.Float: Three scaled by the selected fractional-bit count.
        """
        register = qmc.qubit_array(3, "register")
        register[0] = qmc.x(register[0])
        register[1] = qmc.x(register[1])
        return qmc.measure(qmc.cast(register, qmc.QFixed, int_bits=int_bits))

    transpiler = QiskitTranspiler()
    for kernel, expected in (
        (default_layout, 0.375),
        (explicit_layout, 3 / 2 ** (3 - int_bits)),
    ):
        executable = transpiler.transpile(kernel)
        samples = executable.sample(transpiler.executor(), shots=8).result().results
        assert len(samples) == 1
        value, count = samples[0]
        np.testing.assert_allclose(value, expected, atol=1e-12, rtol=0)
        assert count == 8


@pytest.mark.parametrize(
    ("flag", "expected_qubits"),
    [(1, [1, 3]), (0, [0, 2])],
)
def test_qint_compile_time_branch_rebuilds_slice_carriers(
    flag: int,
    expected_qubits: list[int],
) -> None:
    """Compile-time branch selection refreshes QInt cast carrier metadata."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    @qmc.qkernel
    def kernel(flag: qmc.UInt) -> qmc.UInt:
        register = qmc.qubit_array(4, "register")
        if flag == 1:
            selected = register[1::2]
        else:
            selected = register[0::2]
        return qmc.measure(qmc.cast(selected, qmc.QInt))

    executable = QiskitTranspiler().transpile(kernel, bindings={"flag": flag})
    circuit = executable.compiled_quantum[0].circuit
    measured_qubits = sorted(
        circuit.find_bit(instruction.qubits[0]).index
        for instruction in circuit.data
        if instruction.operation.name == "measure"
    )
    assert measured_qubits == expected_qubits
