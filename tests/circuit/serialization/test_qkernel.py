"""Tests for static qkernel protobuf serialization."""

from __future__ import annotations

import inspect
import subprocess
import sys
from importlib.metadata import version

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.composite_gate import configure_composite
from qamomile.circuit.frontend.qkernel_callable import (
    qkernel_callable_attrs,
    qkernel_callable_ref,
)
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import (
    CallableBodyRef,
    CallableImplementation,
    CallableRef,
    CallPolicy,
    CallTransform,
    CompositeGateType,
)
from qamomile.circuit.serialization import (
    QAMOMILE_VERSION,
    SerializedQKernel,
    deserialize,
    serialize,
)
from qamomile.circuit.serialization.encode import to_dict as kernel_to_dict
from qamomile.circuit.serialization.proto import qamomile_ir_pb2 as pb
from qamomile.qiskit import QiskitTranspiler
from tests.circuit.qkernel_catalog import QKERNEL_BY_ID


@qmc.qkernel
def _parameterized(n: qmc.UInt, theta: qmc.Float = 0.25) -> qmc.Bit:
    """Apply a parameterized rotation a compile-time number of times."""
    q = qmc.qubit("q")
    for _ in qmc.range(n):
        q = qmc.rx(q, theta)
    return qmc.measure(q)


@qmc.qkernel
def _array_shaped(values: qmc.Vector[qmc.Float]) -> qmc.Bit:
    """Use a bound array shape and elements to define a circuit."""
    q = qmc.qubit_array(values.shape[0], "q")
    for i in qmc.range(values.shape[0]):
        q[i] = qmc.rx(q[i], values[i])
    return qmc.measure(q[0])


@qmc.qkernel
def _child(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Nested qkernel used to exercise callable graph preservation."""
    return qmc.ry(q, theta)


@qmc.qkernel
def _parent(theta: qmc.Float) -> qmc.Bit:
    """Invoke another qkernel from the serialized body."""
    q = qmc.qubit("q")
    q = _child(q, theta)
    return qmc.measure(q)


@qmc.qkernel
def _containers(
    pair: qmc.Tuple[qmc.UInt, qmc.Float],
    values: qmc.Dict[qmc.UInt, qmc.Float],
) -> qmc.Float:
    """Expose structural container annotations in the static interface."""
    return pair[1] + values[qmc.uint(0)]


@qmc.qkernel
def _native_annotations(n: int, theta: float, flag: bool) -> bool:
    """Expose Python-native scalar annotations in the static interface."""
    q = qmc.qubit("q")
    for _ in qmc.range(n):
        q = qmc.rx(q, theta)
    if flag:
        q = qmc.x(q)
    return qmc.measure(q)


@qmc.qkernel
def _native_tuple_return(theta: float) -> tuple[bool, float]:
    """Expose a Python tuple return annotation in the static interface."""
    q = qmc.rx(qmc.qubit("q"), theta)
    return qmc.measure(q), theta


@qmc.qkernel
def _custom_composite(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Provide a composite with non-empty serialized callable metadata."""
    return qmc.rz(q, theta)


configure_composite(
    _custom_composite,
    name="custom_rotation",
    namespace="tests.serialization",
    gate_type=CompositeGateType.CUSTOM,
    policy=CallPolicy.PRESERVE_BOX,
    implementations=(
        CallableImplementation(
            transform=CallTransform.DIRECT,
            backend="test-backend",
            strategy="named-strategy",
            body_ref=CallableBodyRef(
                ref=CallableRef("tests.serialization", "custom_rotation_body"),
                attrs={"layout": (0, 1)},
            ),
            attrs={"priority": 2},
        ),
    ),
    semantic_arguments={"axis_order": ("z", "x")},
)


@qmc.qkernel
def _calls_controlled_composite(theta: qmc.Float) -> qmc.Bit:
    """Call a preserved composite through its high-level controlled transform."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(_custom_composite)(control, target, theta)
    return qmc.measure(target)


@qmc.qkernel
def _carried_scalar(n: qmc.UInt) -> qmc.UInt:
    """Carry one scalar through a loop region."""
    total = qmc.uint(0)
    for i in qmc.range(n):
        total = total + i
    return total


_controlled_oracle = qmc.Oracle(
    name="serialization_controlled_oracle",
    num_qubits=1,
    num_control_qubits=1,
)


@qmc.qkernel
def _calls_controlled_oracle() -> qmc.Bit:
    """Invoke an oracle whose signature already includes its control qubit."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = _controlled_oracle(target, controls=(control,))
    return qmc.measure(control)


_DESERIALIZED_CHILD = deserialize(serialize(_child))
_DESERIALIZED_QFT = deserialize(serialize(qmc.qft))
_DESERIALIZED_IQFT = deserialize(serialize(qmc.iqft))


@qmc.qkernel
def _calls_deserialized(theta: qmc.Float) -> qmc.Bit:
    """Invoke a deserialized qkernel as a normal frontend subroutine."""
    q = qmc.qubit("q")
    q = _DESERIALIZED_CHILD(q, theta)
    return qmc.measure(q)


@qmc.qkernel
def _calls_deserialized_qft(n: qmc.UInt) -> qmc.Bit:
    """Invoke a deserialized composite qkernel as a named callable."""
    qubits = qmc.qubit_array(n, "q")
    qubits = _DESERIALIZED_QFT(qubits)
    return qmc.measure(qubits[0])


@qmc.qkernel
def _calls_original_qft(n: qmc.UInt) -> qmc.Bit:
    """Invoke the original QFT for backend-output comparison."""
    qubits = qmc.qubit_array(n, "q")
    qubits = qmc.qft(qubits)
    return qmc.measure(qubits[0])


@qmc.qkernel
def _calls_original_iqft(n: qmc.UInt) -> qmc.Bit:
    """Invoke the original IQFT for backend-output comparison."""
    qubits = qmc.qubit_array(n, "q")
    qubits = qmc.iqft(qubits)
    return qmc.measure(qubits[0])


@qmc.qkernel
def _calls_inverse_deserialized_qft(n: qmc.UInt) -> qmc.Bit:
    """Invoke the inverse of a deserialized QFT."""
    qubits = qmc.qubit_array(n, "q")
    qubits = qmc.inverse(_DESERIALIZED_QFT)(qubits)
    return qmc.measure(qubits[0])


@qmc.qkernel
def _calls_inverse_deserialized_iqft(n: qmc.UInt) -> qmc.Bit:
    """Invoke the inverse of a deserialized IQFT."""
    qubits = qmc.qubit_array(n, "q")
    qubits = qmc.inverse(_DESERIALIZED_IQFT)(qubits)
    return qmc.measure(qubits[0])


def _circuit(kernel: object, **kwargs: object):
    """Transpile a kernel and return its first Qiskit circuit.

    Args:
        kernel (object): QKernel-like entrypoint.
        **kwargs (object): Keyword arguments forwarded to ``transpile``.

    Returns:
        object: First backend circuit.
    """
    executable = QiskitTranspiler().transpile(kernel, **kwargs)  # type: ignore[arg-type]
    circuit = executable.get_first_circuit()
    assert circuit is not None
    return circuit


def _message(kernel: object) -> pb.QKernel:
    """Serialize a qkernel and parse its generated protobuf message.

    Args:
        kernel (object): QKernel-like object to serialize.

    Returns:
        pb.QKernel: Parsed protobuf message.
    """
    message = pb.QKernel()
    message.ParseFromString(serialize(kernel))  # type: ignore[arg-type]
    return message


def _restore(message: pb.QKernel) -> SerializedQKernel:
    """Deserialize a generated qkernel message through the public bytes API.

    Args:
        message (pb.QKernel): Message to serialize into bytes.

    Returns:
        SerializedQKernel: Reconstructed qkernel-like object.
    """
    return deserialize(message.SerializeToString(deterministic=True))


def _fresh_equivalent_kernel():
    """Trace a new qkernel instance with stable source-level semantics.

    Returns:
        object: Independently traced qkernel with the same callable identity and
            algorithm on every invocation.
    """

    @qmc.qkernel
    def independently_traced(theta: qmc.Float) -> qmc.Bit:
        """Apply one rotation and measure the result."""
        q = qmc.qubit("q")
        q = qmc.rx(q, theta)
        return qmc.measure(q)

    return independently_traced


def test_schema_has_one_qkernel_root() -> None:
    """The wire schema exposes no Block or PreparedModule root artifacts."""
    message_names = set(pb.DESCRIPTOR.message_types_by_name)

    assert "QKernel" in message_names
    assert "Algorithm" not in message_names
    assert "PreparedModule" not in message_names
    assert "ParamSlot" not in message_names
    assert [field.name for field in pb.QKernel.DESCRIPTOR.fields] == [
        "qamomile_version",
        "name",
        "parameters",
        "results",
        "body",
        "value_table",
        "callable_table",
        "callable_definition",
        "return_annotation",
    ]


def test_ir_serialize_package_imports_in_a_fresh_interpreter() -> None:
    """The low-level IR package does not depend on serialization adapters."""
    completed = subprocess.run(
        [sys.executable, "-c", "import qamomile.circuit.ir.serialize"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_qkernel_round_trip_preserves_static_ir_and_interface() -> None:
    """Static body identity, annotations, defaults, and call graph survive."""
    message = _message(_parent)
    restored = _restore(message)

    assert message.qamomile_version == QAMOMILE_VERSION
    assert restored.name == _parent.name
    assert list(restored.signature.parameters) == list(_parent.signature.parameters)
    assert restored.input_types == _parent.input_types
    assert restored.output_types == _parent.output_types
    assert qkernel_callable_ref(restored) == qkernel_callable_ref(_parent)
    assert qkernel_callable_attrs(restored) == qkernel_callable_attrs(_parent)
    restored_body = kernel_to_dict(restored)["artifact"]["body"]
    original_body = kernel_to_dict(_parent)["artifact"]["body"]
    assert restored_body == original_body
    assert len(message.callable_table) > 0


def test_scalar_bindings_and_runtime_parameters_work_after_load() -> None:
    """A loaded qkernel follows the ordinary transpile input contract."""
    restored = deserialize(serialize(_parameterized))

    original = _circuit(_parameterized, bindings={"n": 3}, parameters=["theta"])
    round_tripped = _circuit(
        restored,
        bindings={"n": 3},
        parameters=["theta"],
    )

    assert original.num_qubits == round_tripped.num_qubits
    assert original.count_ops() == round_tripped.count_ops()
    assert [str(item) for item in original.parameters] == [
        str(item) for item in round_tripped.parameters
    ]


def test_default_is_applied_after_load() -> None:
    """Signature defaults are static interface data, not saved invocation data."""
    restored = deserialize(serialize(_parameterized))

    original = _circuit(_parameterized, bindings={"n": 2})
    round_tripped = _circuit(restored, bindings={"n": 2})

    assert original.count_ops() == round_tripped.count_ops()
    assert not round_tripped.parameters


def test_array_binding_shape_and_elements_work_after_load() -> None:
    """Array-dependent static structure survives without storing the array."""
    values = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    restored = deserialize(serialize(_array_shaped))

    original = _circuit(_array_shaped, bindings={"values": values})
    round_tripped = _circuit(restored, bindings={"values": values})

    assert original.num_qubits == round_tripped.num_qubits == 3
    assert original.count_ops() == round_tripped.count_ops()


def test_nested_callable_graph_transpiles_after_load() -> None:
    """Shared hierarchical callable definitions remain executable."""
    restored = deserialize(serialize(_parent))

    original = _circuit(_parent, parameters=["theta"])
    round_tripped = _circuit(restored, parameters=["theta"])

    assert original.count_ops() == round_tripped.count_ops()


def test_controlled_callable_transform_round_trips_at_high_level() -> None:
    """Controlled composite invocation remains a validated high-level call."""
    restored = deserialize(serialize(_calls_controlled_composite))

    original = _circuit(_calls_controlled_composite, parameters=["theta"])
    round_tripped = _circuit(restored, parameters=["theta"])

    assert original.count_ops() == round_tripped.count_ops()


def test_controlled_oracle_signature_includes_controls() -> None:
    """Controlled oracle signatures validate without dropping their controls."""
    restored = deserialize(serialize(_calls_controlled_oracle))

    assert kernel_to_dict(restored) == kernel_to_dict(_calls_controlled_oracle)


@pytest.mark.parametrize(
    "catalog_id",
    [
        "maj_loop",
        "uma_2_cnot_loop",
        "uma_3_cnot_loop",
        "simple_ripple_carry_adder_2_cnot",
        "simple_ripple_carry_adder_3_cnot",
    ],
)
def test_independent_callable_bodies_have_separate_ssa_scopes(
    catalog_id: str,
) -> None:
    """Callable bodies may reuse cached semantic values in separate scopes."""
    kernel = QKERNEL_BY_ID[catalog_id].qkernel

    assert serialize(deserialize(serialize(kernel))) == serialize(kernel)


def test_loaded_qkernel_can_be_invoked_inside_another_qkernel() -> None:
    """The deserialized object supports the normal frontend call surface."""
    circuit = _circuit(_calls_deserialized, parameters=["theta"])

    assert circuit.count_ops()["ry"] == 1


def test_root_composite_callable_semantics_round_trip() -> None:
    """QFT retains stable identity, native policy, and composite metadata."""
    restored = deserialize(serialize(qmc.qft))

    assert qkernel_callable_ref(restored) == qkernel_callable_ref(qmc.qft)
    assert qkernel_callable_attrs(restored) == qkernel_callable_attrs(qmc.qft)
    assert restored._callable_kind == "composite"
    assert restored._callable_policy is CallPolicy.NATIVE_FIRST
    assert restored._callable_gate_type is CompositeGateType.QFT
    assert restored._callable_namespace == "qamomile.stdlib"

    block = _calls_deserialized_qft.block
    invoke = next(
        operation
        for operation in block.operations
        if operation.__class__.__name__ == "InvokeOperation"
    )
    assert invoke.target == qkernel_callable_ref(qmc.qft)
    assert invoke.definition is not None
    assert invoke.definition.default_policy is CallPolicy.NATIVE_FIRST
    assert invoke.attrs == qkernel_callable_attrs(qmc.qft)

    original = _circuit(_calls_original_qft, bindings={"n": 3})
    round_tripped = _circuit(_calls_deserialized_qft, bindings={"n": 3})
    assert round_tripped.count_ops() == original.count_ops()


@pytest.mark.parametrize(
    ("restored", "expected", "round_tripped_kernel", "original_kernel"),
    [
        (
            _DESERIALIZED_QFT,
            qmc.iqft,
            _calls_inverse_deserialized_qft,
            _calls_original_iqft,
        ),
        (
            _DESERIALIZED_IQFT,
            qmc.qft,
            _calls_inverse_deserialized_iqft,
            _calls_original_qft,
        ),
    ],
)
def test_inverse_deserialized_qft_family_uses_known_counterpart(
    restored: SerializedQKernel,
    expected: object,
    round_tripped_kernel: object,
    original_kernel: object,
) -> None:
    """Restored QFT/IQFT preserve inverse mapping and backend emission."""
    assert qmc.inverse(restored) is expected

    original = _circuit(original_kernel, bindings={"n": 3})
    round_tripped = _circuit(round_tripped_kernel, bindings={"n": 3})
    assert round_tripped.count_ops() == original.count_ops()


def test_root_callable_implementations_and_semantic_arguments_round_trip() -> None:
    """Non-empty implementation contracts and semantic attrs remain exact."""
    restored = deserialize(serialize(_custom_composite))

    assert qkernel_callable_ref(restored) == qkernel_callable_ref(_custom_composite)
    assert qkernel_callable_attrs(restored) == qkernel_callable_attrs(_custom_composite)
    assert restored._callable_semantic_arguments == {"axis_order": ("z", "x")}
    assert len(restored._callable_implementations) == 1
    implementation = restored._callable_implementations[0]
    assert implementation.transform is CallTransform.DIRECT
    assert implementation.backend == "test-backend"
    assert implementation.strategy == "named-strategy"
    assert implementation.attrs == {"priority": 2}
    assert implementation.body_ref is not None
    assert implementation.body_ref.ref == CallableRef(
        "tests.serialization",
        "custom_rotation_body",
    )
    assert implementation.body_ref.attrs == {"layout": (0, 1)}


def test_container_annotations_round_trip() -> None:
    """Tuple and Dict annotations retain their nested element types."""
    restored = deserialize(serialize(_containers))

    assert list(restored.signature.parameters) == list(_containers.signature.parameters)
    assert restored.input_types == _containers.input_types
    assert restored.output_types == _containers.output_types


def test_python_native_annotations_round_trip_without_normalization() -> None:
    """Python scalar annotations remain distinct from Qamomile handle types."""
    restored = deserialize(serialize(_native_annotations))

    assert restored.input_types == {"n": int, "theta": float, "flag": bool}
    assert restored.output_types == [bool]
    assert [
        parameter.annotation for parameter in restored.signature.parameters.values()
    ] == [int, float, bool]
    assert restored.signature.return_annotation is bool


def test_python_tuple_return_annotation_round_trips_exactly() -> None:
    """A multi-result Python tuple retains its complete return annotation."""
    restored = deserialize(serialize(_native_tuple_return))

    assert restored.output_types == [bool, float]
    assert restored.signature.return_annotation == tuple[bool, float]


def test_wire_payload_contains_no_invocation_values() -> None:
    """The qkernel message contains interface defaults but no bindings fields."""
    message = _message(_parameterized)
    schema_text = message.DESCRIPTOR.file.serialized_pb

    assert b"bindings" not in schema_text
    assert b"bound_value" not in schema_text
    assert b"runtime_parameters" not in schema_text
    assert message.parameters[1].has_default
    assert message.parameters[1].default.float_value.bits


def test_serialization_is_deterministic_for_one_built_qkernel() -> None:
    """Repeated serialization of the cached semantic graph is byte-identical."""
    assert serialize(_parent) == serialize(_parent)


def test_independent_equivalent_traces_have_canonical_bytes() -> None:
    """Random trace identities do not change the canonical protobuf payload."""
    left = _fresh_equivalent_kernel()
    right = _fresh_equivalent_kernel()

    assert left.block.input_values[0].uuid != right.block.input_values[0].uuid
    assert serialize(left) == serialize(right)


def test_canonical_payload_is_idempotent_across_round_trip() -> None:
    """Deserializing and serializing a canonical payload does not change bytes."""
    payload = serialize(_parent)

    assert serialize(deserialize(payload)) == payload


def test_noncanonical_duplicate_wire_field_is_rejected() -> None:
    """Equivalent protobuf fields cannot be appended to canonical bytes."""
    payload = serialize(_parent)
    encoded_version = QAMOMILE_VERSION.encode()
    assert len(encoded_version) < 128
    duplicate_version = bytes((0x0A, len(encoded_version))) + encoded_version

    with pytest.raises(ValueError, match="bytes are not in canonical form"):
        deserialize(payload + duplicate_version)


def test_operation_with_invalid_arity_is_rejected_during_deserialize() -> None:
    """A protobuf gate cannot bypass its semantic operand/result contract."""
    message = _message(_parameterized)
    gate = message.body.operations[1].body[0]
    del gate.operand_refs[:]
    del gate.result_refs[:]

    with pytest.raises(ValueError, match="GateOperation.*requires 2 operands"):
        _restore(message)


def test_operation_with_invalid_operand_type_is_rejected_during_deserialize() -> None:
    """A forged reference cannot feed a Float value to a measurement op."""
    message = _message(_parameterized)
    measure = message.body.operations[2]
    measure.operand_refs[0] = message.body.input_value_refs[1]

    with pytest.raises(ValueError, match="MeasureOperation.*expected QubitType"):
        _restore(message)


def test_operation_reusing_operand_as_result_is_rejected_during_deserialize() -> None:
    """An operation cannot forge an in-place update outside SSA semantics."""
    message = _message(_parameterized)
    measure = message.body.operations[2]
    measure.result_refs[0] = measure.operand_refs[0]

    with pytest.raises(ValueError, match="reuses an operand UUID as an SSA result"):
        _restore(message)


def test_operation_repeating_quantum_operand_is_rejected_during_deserialize() -> None:
    """A forged controlled invocation cannot alias two quantum inputs."""
    message = _message(_calls_controlled_composite)
    invoke = message.body.operations[2]
    invoke.operand_refs[1] = invoke.operand_refs[0]

    with pytest.raises(ValueError, match="repeats a quantum operand UUID"):
        _restore(message)


def test_invoke_kind_disagreeing_with_definition_is_rejected() -> None:
    """Call attrs cannot spoof oracle signature semantics for a composite."""
    message = _message(_calls_controlled_composite)
    invoke = message.body.operations[2]
    invoke.attrs.map_value.entries[0].value.string_value = "oracle"

    with pytest.raises(ValueError, match="kind disagrees with its definition"):
        _restore(message)


def test_duplicate_producer_in_one_block_is_rejected_during_deserialize() -> None:
    """Two operations in the same block cannot produce one SSA identity."""
    message = _message(_calls_controlled_composite)
    message.body.operations[1].result_refs[0] = message.body.operations[0].result_refs[
        0
    ]

    with pytest.raises(ValueError, match="already produced"):
        _restore(message)


def test_classical_operation_with_quantum_types_is_rejected() -> None:
    """Changing a quantum gate tag cannot forge an invalid classical op."""
    message = _message(_parameterized)
    gate = message.body.operations[1].body[0]
    gate.operation_type = pb.BIN_OPERATION
    gate.ClearField("gate_type")
    gate.expression_kind = "ADD"

    with pytest.raises(ValueError, match="BinOp.*operands must be"):
        _restore(message)


def test_classical_initializer_cannot_produce_a_qubit() -> None:
    """A canonical operation tag swap cannot classify a qubit as classical."""
    message = _message(_parameterized)
    message.body.operations[0].operation_type = pb.CINIT_OPERATION

    with pytest.raises(ValueError, match="cannot initialize a quantum value"):
        _restore(message)


def test_unreachable_value_table_entry_is_rejected() -> None:
    """Canonical payloads cannot carry unreferenced value-table nodes."""
    message = _message(_parent)
    extra = message.value_table.add()
    extra.CopyFrom(message.value_table[0])
    extra.uuid = "unused-value"
    extra.logical_id = "unused-logical"

    with pytest.raises(ValueError, match="not in canonical form"):
        _restore(message)


def test_inconsistent_region_result_is_rejected_during_deserialize() -> None:
    """A loop region result must be one of the owning operation's SSA results."""
    message = _message(_carried_scalar)
    loop = next(
        operation for operation in message.body.operations if operation.region_args
    )
    loop.region_args[0].result_ref = message.body.input_value_refs[0]

    with pytest.raises(ValueError, match="region result is not owned by the loop"):
        _restore(message)


def test_version_mismatch_is_rejected() -> None:
    """The format has an explicit same-Qamomile-release contract."""
    message = _message(_parameterized)
    message.qamomile_version = "different-version"

    with pytest.raises(ValueError, match="qamomile_version mismatch"):
        _restore(message)


def test_version_marker_uses_complete_distribution_version() -> None:
    """Development and local build metadata remain part of compatibility."""
    assert QAMOMILE_VERSION == version("qamomile")


def test_duplicate_interface_parameter_is_rejected() -> None:
    """Repeated protobuf parameter names cannot overwrite interface data."""
    message = _message(_parameterized)
    message.parameters.add().CopyFrom(message.parameters[0])

    with pytest.raises(ValueError, match="duplicate qkernel parameter"):
        _restore(message)


def test_missing_result_descriptor_is_rejected() -> None:
    """The declared return interface must match the Block output ABI."""
    message = _message(_parameterized)
    del message.results[:]

    with pytest.raises(ValueError, match="result count"):
        _restore(message)


def test_missing_parameter_descriptor_is_rejected() -> None:
    """The declared parameter interface must match the Block input ABI."""
    message = _message(_parameterized)
    del message.parameters[:]

    with pytest.raises(ValueError, match="parameter count"):
        _restore(message)


def test_parameter_type_inconsistent_with_body_is_rejected() -> None:
    """A forged parameter descriptor cannot change the Block input ABI."""
    message = _message(_parameterized)
    message.parameters[0].type.value_type.kind = pb.FLOAT_TYPE

    with pytest.raises(ValueError, match="parameter 'n'.*type does not match"):
        _restore(message)


def test_result_type_inconsistent_with_body_is_rejected() -> None:
    """A forged result descriptor cannot change the Block output ABI."""
    message = _message(_parameterized)
    message.results[0].value_type.kind = pb.FLOAT_TYPE

    with pytest.raises(ValueError, match="result at index 0 type does not match"):
        _restore(message)


def test_parameter_annotation_inconsistent_with_ir_type_is_rejected() -> None:
    """A forged annotation cannot reinterpret the declared parameter type."""
    message = _message(_parameterized)
    message.parameters[0].type.annotation.kind = pb.QAMOMILE_FLOAT

    with pytest.raises(ValueError, match="parameter 'n'.*type does not match"):
        _restore(message)


def test_return_annotation_inconsistent_with_results_is_rejected() -> None:
    """A forged return annotation cannot reinterpret result descriptors."""
    message = _message(_parameterized)
    message.return_annotation.kind = pb.QAMOMILE_FLOAT

    with pytest.raises(ValueError, match="return annotation does not match"):
        _restore(message)


def test_missing_frontend_annotations_are_rejected() -> None:
    """Every interface position must carry its lossless frontend annotation."""
    parameter_message = _message(_parameterized)
    parameter_message.parameters[0].type.ClearField("annotation")

    with pytest.raises(ValueError, match="ValueType or annotation"):
        _restore(parameter_message)

    return_message = _message(_parameterized)
    return_message.ClearField("return_annotation")

    with pytest.raises(ValueError, match="missing its return annotation"):
        _restore(return_message)


def test_missing_root_callable_definition_is_rejected() -> None:
    """Every qkernel carries one explicit root callable descriptor."""
    message = _message(_parameterized)
    message.ClearField("callable_definition")

    with pytest.raises(ValueError, match="missing its callable definition"):
        _restore(message)


def test_block_and_prepared_artifacts_are_not_serializable() -> None:
    """The public API accepts qkernels only."""
    with pytest.raises(TypeError, match="qkernel-like"):
        serialize(Block())  # type: ignore[arg-type]


def test_explicit_parameter_binding_overlap_is_rejected_after_load() -> None:
    """Loaded qkernels retain the strict disjoint-input contract."""
    restored = deserialize(serialize(_parameterized))

    with pytest.raises(ValueError, match="appear in both"):
        restored.build(parameters=["theta"], theta=0.5, n=2)


def test_signature_parameter_kind_is_preserved() -> None:
    """The reconstructed signature retains Python parameter kinds."""
    restored = deserialize(serialize(_parameterized))

    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in restored.signature.parameters.values()
    )
