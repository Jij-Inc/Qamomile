"""Validate packed register reinterpretation and canonical carrier identities."""

from __future__ import annotations

import dataclasses

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.gate import MeasureQFixedOperation
from qamomile.circuit.ir.types import QFixedType, QUIntType, UIntType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.serialization import SerializedQKernel, deserialize, serialize
from qamomile.circuit.serialization.canonical import canonicalize_graph
from qamomile.circuit.serialization.validation import validate_qkernel_ir
from qamomile.qiskit import QiskitTranspiler


@qmc.qkernel
def _measure_qfixed() -> qmc.Float:
    """Expose a concrete fixed-point cast and its measurement layout.

    Returns:
        qmc.Float: Measured fixed-point value with the lowest bit set.
    """
    register = qmc.qubit_array(3, "register")
    register[0] = qmc.x(register[0])
    return qmc.measure(qmc.cast(register, qmc.QFixed, int_bits=1))


@qmc.qkernel
def _measure_qint() -> qmc.UInt:
    """Expose a concrete unsigned register cast and its measurement.

    Returns:
        qmc.UInt: Measured unsigned value with the lowest bit set.
    """
    register = qmc.qubit_array(3, "register")
    register[0] = qmc.x(register[0])
    return qmc.measure(qmc.cast(register, qmc.QInt))


def _packed_cast_kernel(target_kind: str) -> SerializedQKernel:
    """Insert a reinterpretation from the other packed encoding.

    Args:
        target_kind (str): Target encoding, either ``"qfixed"`` or ``"qint"``.

    Returns:
        SerializedQKernel: Kernel measuring the reinterpreted source bits.
    """
    kernel = deserialize(
        serialize(_measure_qfixed if target_kind == "qfixed" else _measure_qint)
    )
    cast_index, target_cast = next(
        (index, operation)
        for index, operation in enumerate(kernel.block.operations)
        if isinstance(operation, CastOperation)
    )
    original_result = target_cast.results[0]
    source_type = QUIntType(3) if target_kind == "qfixed" else QFixedType(1, 2)
    metadata = dataclasses.replace(original_result.metadata, qfixed=None)
    source = Value(type=source_type, name="packed_source", metadata=metadata)
    if isinstance(source_type, QFixedType):
        source = source.with_qfixed_metadata(target_cast.qubit_mapping, 3, 1)
    source_cast = dataclasses.replace(
        target_cast, results=[source], target_type=source_type
    )
    result = original_result.with_cast_metadata(
        source.uuid,
        target_cast.qubit_mapping,
        source.logical_id,
        original_result.get_cast_qubit_logical_ids(),
    )
    kernel.block.operations.insert(cast_index, source_cast)
    target_cast.source_type = source_type
    target_cast.operands = [source]
    target_cast.results = [result]
    for operation in kernel.block.operations[cast_index + 2 :]:
        operation.operands = [
            result if operand.uuid == result.uuid else operand
            for operand in operation.operands
        ]
    return kernel


def _reinterpretation(kernel: SerializedQKernel) -> CastOperation:
    """Return the packed-to-packed operation in one test kernel.

    Args:
        kernel (SerializedQKernel): Fixture containing one packed conversion.

    Returns:
        CastOperation: Conversion consuming the intermediate packed value.

    Raises:
        StopIteration: If the fixture contains no packed conversion.
    """
    return next(
        operation
        for operation in kernel.block.operations
        if isinstance(operation, CastOperation)
        and isinstance(operation.source_type, (QFixedType, QUIntType))
    )


@pytest.mark.parametrize("target_kind", ["qfixed", "qint"])
def test_packed_to_packed_cast_roundtrip(target_kind: str) -> None:
    """Round trips preserve both encodings over the same ordered carriers."""
    kernel = _packed_cast_kernel(target_kind)
    payload = serialize(kernel)
    restored = deserialize(payload)
    operation = _reinterpretation(restored)
    source, result = operation.operands[0], operation.results[0]

    assert type(source.type) is (QUIntType if target_kind == "qfixed" else QFixedType)
    assert type(result.type) is (QFixedType if target_kind == "qfixed" else QUIntType)
    assert source.get_cast_qubit_uuids() == tuple(operation.qubit_mapping)
    assert result.get_cast_qubit_uuids() == source.get_cast_qubit_uuids()
    assert result.get_cast_qubit_logical_ids() == source.get_cast_qubit_logical_ids()
    assert result.get_cast_source_uuid() == source.uuid
    assert result.get_cast_source_logical_id() == source.logical_id
    fixed = result if target_kind == "qfixed" else source
    assert fixed.metadata.qfixed is not None
    assert fixed.metadata.qfixed.num_bits == 3
    assert fixed.metadata.qfixed.int_bits == 1
    assert serialize(restored) == payload


@pytest.mark.parametrize("target_kind, expected", [("qfixed", 0.25), ("qint", 1)])
def test_packed_to_packed_roundtrip_executes(target_kind: str, expected: float) -> None:
    """Reinterpretation retains asymmetric bit significance during execution."""
    kernel = _packed_cast_kernel(target_kind)
    transpiler = QiskitTranspiler()
    for candidate in (kernel, deserialize(serialize(kernel))):
        result = (
            transpiler.transpile(candidate)
            .sample(transpiler.executor(), shots=16)
            .result()
        )
        assert len(result.results) == 1
        actual, count = result.results[0]
        assert count == 16
        if target_kind == "qfixed":
            assert actual == pytest.approx(expected, abs=1e-12, rel=0.0)
        else:
            assert actual == expected


@pytest.mark.parametrize("target_kind", ["qfixed", "qint"])
@pytest.mark.parametrize("channel", ["physical", "logical", "mapping", "layout"])
def test_packed_to_packed_cast_rejects_inconsistent_carriers(
    target_kind: str, channel: str
) -> None:
    """Reinterpretation validates ordered identities, mapping, and layout."""
    kernel = _packed_cast_kernel(target_kind)
    operation = _reinterpretation(kernel)
    source, result = operation.operands[0], operation.results[0]
    metadata = result.metadata.cast
    assert metadata is not None
    if channel == "mapping":
        operation.qubit_mapping.reverse()
        expected = "qubit_mapping disagrees"
    elif channel == "layout":
        fixed = result if target_kind == "qfixed" else source
        assert fixed.metadata.qfixed is not None
        changed = dataclasses.replace(
            fixed,
            metadata=dataclasses.replace(
                fixed.metadata,
                qfixed=dataclasses.replace(fixed.metadata.qfixed, int_bits=2),
            ),
        )
        if target_kind == "qfixed":
            operation.results = [changed]
        else:
            operation.operands = [changed]
        expected = "layout width disagrees"
    else:
        changed_metadata = dataclasses.replace(
            metadata,
            **{
                "qubit_uuids" if channel == "physical" else "qubit_logical_ids": tuple(
                    reversed(
                        metadata.qubit_uuids
                        if channel == "physical"
                        else metadata.qubit_logical_ids
                    )
                )
            },
        )
        changed = dataclasses.replace(
            result, metadata=dataclasses.replace(result.metadata, cast=changed_metadata)
        )
        if channel == "physical":
            operation.qubit_mapping.reverse()
            if isinstance(changed.type, QFixedType):
                changed = changed.with_qfixed_metadata(operation.qubit_mapping, 3, 1)
        operation.results = [changed]
        expected = "carrier order disagrees"
    with pytest.raises(ValueError, match=expected):
        validate_qkernel_ir(Block(operations=[operation]))


@pytest.mark.parametrize("target_kind", ["qfixed", "qint"])
def test_packed_to_packed_cast_rejects_width_mismatch(target_kind: str) -> None:
    """Equal target metadata cannot hide a wider source register."""
    operation = _reinterpretation(_packed_cast_kernel(target_kind))
    source = operation.operands[0]
    source_type = QUIntType(4) if target_kind == "qfixed" else QFixedType(1, 3)
    source = dataclasses.replace(source, type=source_type).with_cast_metadata(
        source.get_cast_source_uuid(),
        (*source.get_cast_qubit_uuids(), "extra_carrier"),
        source.get_cast_source_logical_id(),
        (*source.get_cast_qubit_logical_ids(), "extra_logical"),
    )
    if isinstance(source_type, QFixedType):
        source = source.with_qfixed_metadata(source.get_cast_qubit_uuids(), 4, 1)
    operation.source_type = source_type
    operation.operands = [source]
    with pytest.raises(ValueError, match="width disagrees with its source register"):
        validate_qkernel_ir(Block(operations=[operation]))


@pytest.mark.parametrize("target_kind", ["qfixed", "qint"])
@pytest.mark.parametrize("side", ["source", "result"])
def test_packed_to_packed_cast_requires_cast_metadata(
    target_kind: str, side: str
) -> None:
    """Both scalar encodings require independent cast provenance."""
    operation = _reinterpretation(_packed_cast_kernel(target_kind))
    values = operation.operands if side == "source" else operation.results
    value = values[0]
    values[0] = dataclasses.replace(
        value, metadata=dataclasses.replace(value.metadata, cast=None)
    )
    with pytest.raises(ValueError, match="carrier requires cast metadata"):
        validate_qkernel_ir(Block(operations=[operation]))


@pytest.mark.parametrize("target_kind", ["qfixed", "qint"])
def test_packed_to_packed_cast_rejects_unresolved_width(target_kind: str) -> None:
    """Reinterpretation requires provable source and target total widths."""
    operation = _reinterpretation(_packed_cast_kernel(target_kind))
    width = Value(type=UIntType(), name="width")
    for values in (operation.operands, operation.results):
        value = values[0]
        register_type = (
            QFixedType(0, width)
            if isinstance(value.type, QFixedType)
            else QUIntType(width)
        )
        value = dataclasses.replace(value, type=register_type).with_cast_metadata(
            value.get_cast_source_uuid(), [], value.get_cast_source_logical_id(), []
        )
        if isinstance(register_type, QFixedType):
            value = value.with_qfixed_metadata([], 0, 0)
        values[0] = value
    operation.source_type = operation.operands[0].type
    operation.target_type = operation.results[0].type
    operation.qubit_mapping = []
    with pytest.raises(ValueError, match="unsupported symbolic packed-to-packed"):
        validate_qkernel_ir(Block(operations=[operation]))


def _qfixed_constant_integer_bits_kernel() -> SerializedQKernel:
    """Replace a fixed-point integer width with its constant UInt spelling.

    Returns:
        SerializedQKernel: Kernel with a type-only constant integer-bit value.
    """
    kernel = deserialize(serialize(_measure_qfixed))
    integer_bits = Value(type=UIntType(), name="integer_bits").with_const(1)
    for operation in kernel.block.operations:
        for value in [*operation.operands, *operation.results]:
            if isinstance(value.type, QFixedType):
                value.type.integer_bits = integer_bits
        if isinstance(operation, CastOperation):
            assert isinstance(operation.target_type, QFixedType)
            operation.target_type.integer_bits = integer_bits
    return kernel


def test_qfixed_constant_uint_integer_bits_roundtrip() -> None:
    """Constant UInt integer bits agree with integer metadata and op fields."""
    payload = serialize(_qfixed_constant_integer_bits_kernel())
    restored = deserialize(payload)
    measurement = next(
        operation
        for operation in restored.block.operations
        if isinstance(operation, MeasureQFixedOperation)
    )
    integer_bits = measurement.operands[0].type.integer_bits
    assert isinstance(integer_bits, Value)
    assert integer_bits.get_const() == 1
    assert measurement.int_bits == 1
    assert measurement.num_bits == 3
    assert serialize(restored) == payload


@pytest.mark.parametrize("operation_kind", ["cast", "measure"])
def test_qfixed_symbolic_integer_bits_are_explicitly_unsupported(
    operation_kind: str,
) -> None:
    """The metadata schema cannot encode an unresolved integer-bit count."""
    kernel = _qfixed_constant_integer_bits_kernel()
    kind = CastOperation if operation_kind == "cast" else MeasureQFixedOperation
    operation = next(op for op in kernel.block.operations if isinstance(op, kind))
    fixed_type = (
        operation.target_type
        if isinstance(operation, CastOperation)
        else operation.operands[0].type
    )
    fixed_type.integer_bits = Value(type=UIntType(), name="integer_bits")
    if isinstance(operation, CastOperation):
        operation.results[0].type.integer_bits = fixed_type.integer_bits
    with pytest.raises(
        ValueError, match="unsupported symbolic QFixedType integer_bits"
    ):
        validate_qkernel_ir(Block(operations=[operation]))


def test_canonical_numeric_suffix_identities_match_value_table() -> None:
    """Numeric suffixes on element, logical, and parent IDs remain identities."""
    envelope = {
        "value_table": [
            {"uuid": "root_7", "logical_id": "logical_root_7"},
            {"uuid": "elem", "logical_id": "logical_elem"},
            {"uuid": "elem_2", "logical_id": "logical_elem_2"},
        ],
        "body": {
            "metadata": {
                "array_runtime": {
                    "element_uuids": ["elem_2"],
                    "element_logical_ids": ["logical_elem_2"],
                    "element_parent_uuids": ["root_7"],
                },
                "cast": {
                    "qubit_uuids": ["elem_2", "root_7_2"],
                    "qubit_logical_ids": ["logical_elem_2", "logical_root_7_2"],
                },
            },
            "qubit_mapping": ["elem_2", "root_7_2"],
        },
    }
    canonical = canonicalize_graph(envelope)
    root, _, element = canonical["value_table"]
    metadata = canonical["body"]["metadata"]
    runtime = metadata["array_runtime"]
    assert runtime["element_uuids"] == [element["uuid"]]
    assert runtime["element_logical_ids"] == [element["logical_id"]]
    assert runtime["element_parent_uuids"] == [root["uuid"]]
    assert metadata["cast"]["qubit_uuids"] == [element["uuid"], f"{root['uuid']}_2"]
    assert metadata["cast"]["qubit_logical_ids"] == [
        element["logical_id"],
        f"{root['logical_id']}_2",
    ]
    assert canonical["body"]["qubit_mapping"] == metadata["cast"]["qubit_uuids"]
