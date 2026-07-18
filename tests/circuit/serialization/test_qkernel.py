"""Tests for static qkernel protobuf serialization."""

from __future__ import annotations

import dataclasses
import inspect
import struct
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
from qamomile.circuit.ir.operation.arithmetic_operations import CompOp, CompOpKind
from qamomile.circuit.ir.operation.callable import (
    CallableBodyRef,
    CallableImplementation,
    CallableRef,
    CallPolicy,
    CallTransform,
    CompositeGateType,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.classical_ops import (
    ReturnQuantumArrayElementOperation,
)
from qamomile.circuit.ir.operation.control_flow import HasNestedOps
from qamomile.circuit.ir.operation.gate import GateOperation, GateOperationType
from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.serialize.encode import (
    _OP_ENCODERS,
    _encode_block,
    _EncodeContext,
)
from qamomile.circuit.ir.types.primitives import FloatType, QubitType, UIntType
from qamomile.circuit.ir.uuid_remapper import UUIDRemapper
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.serialization import (
    QAMOMILE_VERSION,
    SerializedQKernel,
    deserialize,
    serialize,
)
from qamomile.circuit.serialization.decode import from_dict as kernel_from_dict
from qamomile.circuit.serialization.encode import to_dict as kernel_to_dict
from qamomile.circuit.serialization.graph_protobuf import _OPERATION_TO_PROTO
from qamomile.circuit.serialization.proto import qamomile_ir_pb2 as pb
from qamomile.circuit.serialization.validation import validate_qkernel_ir
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
def _phase_identity(q: qmc.Qubit) -> qmc.Qubit:
    """Return one qubit unchanged for global-phase serialization tests."""
    return q


@qmc.qkernel
def _global_phase_kernel(q: qmc.Qubit) -> qmc.Qubit:
    """Append a zero-result global-phase operation."""
    return qmc.global_phase(_phase_identity, 0.375)(q)


@qmc.qkernel
def _symbolic_vector_x(
    qubits: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Apply X through a loop over a symbolic vector width."""
    for index in qmc.range(qubits.shape[0]):
        qubits[index] = qmc.x(qubits[index])
    return qubits


@qmc.qkernel
def _atomic_symbolic_vector_inverse() -> qmc.Vector[qmc.Bit]:
    """Inverse-call a symbolic-vector kernel at a concrete width."""
    qubits = qmc.qubit_array(2, "qubits")
    qubits = qmc.inverse(_symbolic_vector_x)(qubits)
    return qmc.measure(qubits)


@qmc.qkernel
def _vector_rotation_layer(
    qubits: qmc.Vector[qmc.Qubit],
    angles: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Qubit]:
    """Rotate each qubit by the corresponding vector element."""
    for index in qmc.range(qubits.shape[0]):
        qubits[index] = qmc.rx(qubits[index], angles[index])
    return qubits


@qmc.qkernel
def _vector_parameter_inverse_round_trip(
    angles: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    """Apply a vector-parameter layer and its atomic inverse."""
    qubits = qmc.qubit_array(2, "qubits")
    qubits = _vector_rotation_layer(qubits, angles)
    qubits = qmc.inverse(_vector_rotation_layer)(qubits, angles)
    return qmc.measure(qubits)


@qmc.qkernel
def _inverse_with_free_classical_capture(theta: qmc.Float) -> qmc.Bit:
    """Inverse-call a nested kernel that captures a parent parameter."""

    @qmc.qkernel
    def rotation(qubit: qmc.Qubit) -> qmc.Qubit:
        """Rotate by the enclosing runtime parameter."""
        return qmc.rx(qubit, theta)

    qubit = qmc.qubit("qubit")
    qubit = qmc.inverse(rotation)(qubit)
    return qmc.measure(qubit)


@qmc.qkernel
def _branch_selected_array_return() -> qmc.Vector[qmc.Bit]:
    """Return statically selected qubits to loop-indexed array slots."""
    qubits = qmc.qubit_array(2, "qubits")
    for index in qmc.range(2):
        if index == 0:
            selected = qubits[0]
        else:
            selected = qubits[1]
        selected = qmc.x(selected)
        qubits[index] = selected
    return qmc.measure(qubits)


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
def _comparison_operations(
    integer: qmc.UInt,
) -> tuple[qmc.Bit, qmc.Bit, qmc.Bit, qmc.Bit]:
    """Expose serializable Bit and mixed Bit-UInt comparisons.

    Args:
        integer (qmc.UInt): Unsigned-integer equality operand.

    Returns:
        tuple[qmc.Bit, qmc.Bit, qmc.Bit, qmc.Bit]: Bit equality, Bit
            inequality, mixed equality, and reflected mixed inequality.
    """
    left = qmc.measure(qmc.qubit("left"))
    right = qmc.measure(qmc.qubit("right"))
    return left == right, left != right, left == integer, integer != right


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
def _value_controlled_x(
    control_0: qmc.Qubit,
    control_1: qmc.Qubit,
    target: qmc.Qubit,
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    """Apply X for the LSB-first control value two."""
    return qmc.control(qmc.x, num_controls=2, control_value=2)(
        control_0,
        control_1,
        target,
    )


@qmc.qkernel
def _ordinary_controlled_x(
    control_0: qmc.Qubit,
    control_1: qmc.Qubit,
    target: qmc.Qubit,
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    """Apply X for the canonical all-ones control value."""
    return qmc.control(qmc.x, num_controls=2)(
        control_0,
        control_1,
        target,
    )


@qmc.qkernel
def _zero_value_controlled_x(
    control: qmc.Qubit,
    target: qmc.Qubit,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Apply X when one control qubit is zero."""
    return qmc.control(qmc.x, control_value=0)(control, target)


@qmc.qkernel
def _wide_value_controlled_x() -> qmc.Bit:
    """Apply X under an activation value wider than 64 bits."""
    controls = qmc.qubit_array(70, "controls")
    target = qmc.qubit("target")
    controls, target = qmc.control(
        qmc.x,
        num_controls=70,
        control_value=1 << 69,
    )(controls, target)
    return qmc.measure(target)


@qmc.qkernel
def _serialization_identity(target: qmc.Qubit) -> qmc.Qubit:
    """Return one serialization-test target unchanged."""
    return target


@qmc.qkernel
def _global_phase_program(
    target: qmc.Qubit,
    angle: qmc.Float,
) -> qmc.Qubit:
    """Apply a serializable global phase to an identity body."""
    return qmc.global_phase(_serialization_identity, angle)(target)


@qmc.qkernel
def _select_program(
    index: qmc.Qubit,
    target: qmc.Qubit,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Select identity or X from one index qubit."""
    return qmc.select([_serialization_identity, qmc.x])(index, target)


@qmc.qkernel
def _wide_select_program() -> qmc.Bit:
    """Preserve a concrete SELECT index width greater than 64."""
    index = qmc.qubit_array(70, "index")
    target = qmc.qubit("target")
    index, target = qmc.select(
        [_serialization_identity, qmc.x],
        num_index_qubits=70,
    )(index, target)
    return qmc.measure(target)


@qmc.qkernel
def _select_pair_identity(
    scalar: qmc.Qubit,
    vector: qmc.Vector[qmc.Qubit],
) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit]]:
    """Return a scalar and vector SELECT target unchanged."""
    return scalar, vector


@qmc.qkernel
def _select_pair_x(
    scalar: qmc.Qubit,
    vector: qmc.Vector[qmc.Qubit],
) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit]]:
    """Apply X to the scalar while retaining a vector SELECT target."""
    return qmc.x(scalar), vector


@qmc.qkernel
def _symbolic_select_program(width: qmc.UInt) -> qmc.Bit:
    """Use a symbolic width across scalar and array index arguments."""
    index_scalar = qmc.qubit("index_scalar")
    index_array = qmc.qubit_array(width - 1, "index_array")
    target_scalar = qmc.qubit("target_scalar")
    target_array = qmc.qubit_array(2, "target_array")
    index_scalar, index_array, target_scalar, target_array = qmc.select(
        [_select_pair_identity, _select_pair_x],
        num_index_qubits=width,
    )(
        index_scalar,
        index_array,
        target_scalar,
        target_array,
    )
    return qmc.measure(target_scalar)


@qmc.qkernel
def _ordered_select_program() -> qmc.Bit:
    """Select four distinct gates in ascending index order."""
    index = qmc.qubit_array(2, "index")
    target = qmc.qubit("target")
    index, target = qmc.select([qmc.x, qmc.y, qmc.z, qmc.h])(index, target)
    return qmc.measure(target)


@qmc.qkernel
def _carried_scalar(n: qmc.UInt) -> qmc.UInt:
    """Carry one scalar through a loop region."""
    total = qmc.uint(0)
    for i in qmc.range(n):
        total = total + i
    return total


@qmc.qkernel
def _constant_one() -> qmc.UInt:
    """Return one caller-local compile-time UInt value."""
    return qmc.uint(1)


@qmc.qkernel
def _calls_constant_one_twice() -> qmc.UInt:
    """Consume two independently materialized constant qkernel results."""
    return _constant_one() + _constant_one()


@qmc.qkernel
def _array_parent_metadata(obs: qmc.Observable) -> qmc.Float:
    """Attach root-array addresses to tuple-form expectation operands."""
    qubits = qmc.qubit_array(2, "qubits")
    return qmc.expval((qubits[0], qubits[1]), obs)


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


def test_every_encodable_operation_has_a_protobuf_mapping() -> None:
    """The IR encoder and the protobuf operation table cover the same ops.

    The two tables are edited in different modules, so an operation added to
    only one of them still merges cleanly and fails at runtime instead. Both
    are private, and the invariant relates them directly, so it is asserted
    here rather than through a public entry point.
    """
    encodable = {operation.__name__ for operation in _OP_ENCODERS}
    mapped = set(_OPERATION_TO_PROTO)

    assert encodable == mapped


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


def test_repeated_constant_qkernel_calls_get_distinct_results() -> None:
    """Each call materializes a fresh SSA result for a constant output."""
    payload = serialize(_calls_constant_one_twice)
    restored = deserialize(payload)
    invokes = [
        operation
        for operation in restored.block.operations
        if isinstance(operation, InvokeOperation)
    ]

    assert len(invokes) == 2
    assert invokes[0].results[0].uuid != invokes[1].results[0].uuid
    assert invokes[0].results[0].logical_id != invokes[1].results[0].logical_id
    assert serialize(restored) == payload


def test_bit_comparison_operations_round_trip() -> None:
    """Bit and mixed Bit-UInt CompOps survive protobuf serialization."""
    payload = serialize(_comparison_operations)
    restored = deserialize(payload)
    comparisons = [
        operation
        for operation in restored.block.operations
        if isinstance(operation, CompOp)
    ]

    assert [operation.kind for operation in comparisons] == [
        CompOpKind.EQ,
        CompOpKind.NEQ,
        CompOpKind.EQ,
        CompOpKind.NEQ,
    ]
    assert serialize(restored) == payload


def test_bit_ordering_operation_is_rejected_during_deserialize() -> None:
    """A forged ordering relation cannot use Bit comparison operands."""
    message = _message(_comparison_operations)
    message.body.operations[4].expression_kind = CompOpKind.LT.name

    with pytest.raises(ValueError, match="numeric scalars.*equality"):
        _restore(message)


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


def test_control_value_uses_an_optional_arbitrary_integer_field() -> None:
    """A non-default control value uses the typed BigInteger wire field."""
    patterned = _message(_value_controlled_x)
    patterned_operation = next(
        operation
        for operation in patterned.body.operations
        if operation.operation_type == pb.CONCRETE_CONTROLLED_OPERATION
    )
    ordinary = _message(_ordinary_controlled_x)
    ordinary_operation = next(
        operation
        for operation in ordinary.body.operations
        if operation.operation_type == pb.CONCRETE_CONTROLLED_OPERATION
    )

    assert patterned_operation.HasField("control_value")
    assert not patterned_operation.control_value.negative
    assert int.from_bytes(patterned_operation.control_value.magnitude, "big") == 2
    assert not ordinary_operation.HasField("control_value")

    restored = _restore(patterned)
    controlled = next(
        operation
        for operation in restored.block.operations
        if operation.__class__.__name__ == "ConcreteControlledU"
    )
    assert controlled.control_value == 2


@pytest.mark.parametrize(
    ("kernel", "expected"),
    [
        pytest.param(_zero_value_controlled_x, 0, id="zero"),
        pytest.param(_wide_value_controlled_x, 1 << 69, id="wider-than-64-bits"),
    ],
)
def test_control_value_big_integer_boundaries_round_trip(
    kernel: object,
    expected: int,
) -> None:
    """Zero and values wider than 64 bits retain field presence and value."""
    message = _message(kernel)
    encoded = next(
        operation
        for operation in message.body.operations
        if operation.operation_type == pb.CONCRETE_CONTROLLED_OPERATION
    )

    assert encoded.HasField("control_value")
    assert int.from_bytes(encoded.control_value.magnitude, "big") == expected
    if expected == 0:
        assert encoded.control_value.magnitude == b""

    restored = _restore(message)
    controlled = next(
        operation
        for operation in restored.block.operations
        if operation.__class__.__name__ == "ConcreteControlledU"
    )
    assert controlled.control_value == expected


def test_out_of_range_control_value_is_rejected_during_deserialize() -> None:
    """A forged activation integer cannot exceed its declared control width."""
    message = _message(_value_controlled_x)
    operation = next(
        operation
        for operation in message.body.operations
        if operation.operation_type == pb.CONCRETE_CONTROLLED_OPERATION
    )
    operation.control_value.negative = False
    operation.control_value.magnitude = b"\x04"

    with pytest.raises(ValueError, match="control_value 4 does not fit"):
        _restore(message)


def test_select_without_a_quantum_target_is_rejected() -> None:
    """Serialized SELECT requires a target beyond its index and parameters."""
    index = Value(type=QubitType(), name="index")
    parameter = Value(type=FloatType(), name="parameter")
    index_result = index.next_version()
    cases = [
        Block(input_values=[parameter], output_values=[]),
        Block(input_values=[parameter], output_values=[]),
    ]
    operation = SelectOperation(
        operands=[index, parameter],
        results=[index_result],
        num_index_qubits=1,
        case_blocks=cases,
    )
    block = Block(
        input_values=[index, parameter],
        output_values=[index_result],
        operations=[operation],
    )

    with pytest.raises(ValueError, match="requires at least one quantum target"):
        validate_qkernel_ir(block)


def test_concrete_select_width_greater_than_64_round_trips() -> None:
    """The original concrete-width field preserves a large overwide SELECT."""
    message = _message(_wide_select_program)
    encoded = next(
        operation
        for operation in message.body.operations
        if operation.operation_type == pb.SELECT_OPERATION
    )

    assert encoded.HasField("num_index_qubits")
    assert encoded.num_index_qubits == 70
    assert not encoded.HasField("num_index_qubits_ref")
    assert not encoded.HasField("num_index_args")

    restored = _restore(message)
    select = next(
        operation
        for operation in restored.block.operations
        if isinstance(operation, SelectOperation)
    )
    assert select.num_index_qubits == 70
    assert select.num_index_args == 70


def test_symbolic_select_width_and_argument_groups_round_trip() -> None:
    """A UInt width references its Value and retains mixed index groups."""
    message = _message(_symbolic_select_program)
    encoded = next(
        operation
        for operation in message.body.operations
        if operation.operation_type == pb.SELECT_OPERATION
    )

    assert not encoded.HasField("num_index_qubits")
    assert encoded.HasField("num_index_qubits_ref")
    assert encoded.num_index_args == 2
    assert encoded.num_index_qubits_ref in {value.uuid for value in message.value_table}

    restored = _restore(message)
    select = next(
        operation
        for operation in restored.block.operations
        if isinstance(operation, SelectOperation)
    )
    assert isinstance(select.num_index_qubits, Value)
    assert select.num_index_qubits.name == "width"
    assert select.num_index_args == 2
    assert not isinstance(select.index_operands[0], ArrayValue)
    assert isinstance(select.index_operands[1], ArrayValue)
    assert not isinstance(select.target_operands[0], ArrayValue)
    assert isinstance(select.target_operands[1], ArrayValue)

    original_circuit = _circuit(_symbolic_select_program, bindings={"width": 3})
    restored_circuit = _circuit(restored, bindings={"width": 3})
    assert original_circuit.num_qubits == restored_circuit.num_qubits
    assert original_circuit.count_ops() == restored_circuit.count_ops()


def test_select_case_order_round_trips() -> None:
    """SELECT case blocks remain in ascending index order."""
    restored = _restore(_message(_ordered_select_program))
    select = next(
        operation
        for operation in restored.block.operations
        if isinstance(operation, SelectOperation)
    )
    gate_types = [
        next(
            operation.gate_type
            for operation in case.operations
            if isinstance(operation, GateOperation)
        )
        for case in select.case_blocks
    ]

    assert gate_types == [
        GateOperationType.X,
        GateOperationType.Y,
        GateOperationType.Z,
        GateOperationType.H,
    ]


@pytest.mark.parametrize("retain_concrete", [False, True])
def test_select_width_union_rejects_missing_or_mutually_present_fields(
    retain_concrete: bool,
) -> None:
    """SELECT requires exactly one concrete or symbolic width field."""
    message = _message(_wide_select_program)
    encoded = next(
        operation
        for operation in message.body.operations
        if operation.operation_type == pb.SELECT_OPERATION
    )
    if retain_concrete:
        encoded.num_index_qubits_ref = encoded.operand_refs[0]
    else:
        encoded.ClearField("num_index_qubits")

    with pytest.raises(ValueError, match="requires exactly one"):
        _restore(message)


def test_symbolic_select_rejects_a_missing_width_reference() -> None:
    """A symbolic SELECT width must resolve through the Value table."""
    message = _message(_symbolic_select_program)
    encoded = next(
        operation
        for operation in message.body.operations
        if operation.operation_type == pb.SELECT_OPERATION
    )
    encoded.num_index_qubits_ref = "missing-width-value"

    with pytest.raises(ValueError, match="value_table is missing entry"):
        _restore(message)


def test_symbolic_select_requires_its_index_argument_count() -> None:
    """A symbolic SELECT cannot infer its operand-slot boundary on load."""
    message = _message(_symbolic_select_program)
    encoded = next(
        operation
        for operation in message.body.operations
        if operation.operation_type == pb.SELECT_OPERATION
    )
    encoded.ClearField("num_index_args")

    with pytest.raises(ValueError, match="requires num_index_args"):
        _restore(message)


def test_symbolic_select_rejects_changed_result_grouping() -> None:
    """Scalar and array index result slots cannot be interchanged."""
    message = _message(_symbolic_select_program)
    encoded = next(
        operation
        for operation in message.body.operations
        if operation.operation_type == pb.SELECT_OPERATION
    )
    result_refs = list(encoded.result_refs)
    result_refs[0], result_refs[1] = result_refs[1], result_refs[0]
    del encoded.result_refs[:]
    encoded.result_refs.extend(result_refs)

    with pytest.raises(ValueError, match="preserve quantum argument grouping"):
        _restore(message)


@pytest.mark.parametrize(
    ("kernel", "operation_type"),
    [
        (_global_phase_program, pb.GLOBAL_PHASE_OPERATION),
        (_select_program, pb.SELECT_OPERATION),
    ],
)
def test_semantic_quantum_operations_round_trip_as_typed_wire_nodes(
    kernel: object,
    operation_type: int,
) -> None:
    """Global phase and SELECT retain dedicated protobuf operation types."""
    message = _message(kernel)

    assert any(
        operation.operation_type == operation_type
        for operation in message.body.operations
    )
    assert serialize(_restore(message)) == serialize(kernel)  # type: ignore[arg-type]


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
    """The qkernel message contains defaults but no concrete binding values."""
    message = _message(_parameterized)
    pending = list(pb.DESCRIPTOR.message_types_by_name.values())
    message_descriptors = []
    while pending:
        descriptor = pending.pop()
        message_descriptors.append(descriptor)
        pending.extend(descriptor.nested_types)
    forbidden_fields = {"bindings", "bound_value", "runtime_parameters"}

    for descriptor in message_descriptors:
        assert forbidden_fields.isdisjoint(field.name for field in descriptor.fields), (
            descriptor.full_name
        )
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


def test_if_decoder_internal_error_is_normalized_to_value_error() -> None:
    """Malformed branch operands cannot leak a developer RuntimeError."""
    message = _message(_native_annotations)
    operation = next(
        item
        for item in message.body.operations
        if item.operation_type == pb.IF_OPERATION
    )
    del operation.operand_refs[:]

    with pytest.raises(ValueError, match="payload is malformed"):
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

    with pytest.raises(ValueError, match="result_ref does not match result_refs"):
        _restore(message)


@pytest.mark.parametrize(
    "field_name",
    ["element_parent_uuids", "element_parent_indices"],
)
def test_missing_array_parent_value_is_rejected_during_deserialize(
    field_name: str,
) -> None:
    """Missing optional parent values cannot leak ``None`` into semantic IR."""
    message = _message(_array_parent_metadata)
    runtime = next(
        value.metadata.array_runtime
        for value in message.value_table
        if value.metadata.HasField("array_runtime")
        and value.metadata.array_runtime.element_parent_indices
    )
    getattr(runtime, field_name)[0].ClearField("value")

    with pytest.raises(ValueError, match="not in canonical form"):
        _restore(message)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("element_parent_uuids", 1),
        ("element_parent_indices", "0"),
        ("element_parent_indices", True),
    ],
)
def test_invalid_array_parent_type_is_rejected(
    field_name: str,
    invalid_value: object,
) -> None:
    """Internal graph decoding rejects invalid array-parent entry types."""
    envelope = kernel_to_dict(_array_parent_metadata)
    runtime = next(
        value["metadata"]["array_runtime"]
        for value in envelope["value_table"]
        if value["metadata"].get("array_runtime") is not None
        and value["metadata"]["array_runtime"]["element_parent_indices"]
    )
    runtime[field_name][0] = invalid_value

    with pytest.raises(ValueError, match=rf"{field_name}\[0\]"):
        kernel_from_dict(envelope)


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


def test_global_phase_round_trip_preserves_zero_result_operand() -> None:
    """Public protobuf serialization preserves a standalone global phase."""
    restored = deserialize(serialize(_global_phase_kernel))
    block = restored.build()

    phase_operations = [
        operation
        for operation in block.operations
        if isinstance(operation, GlobalPhaseOperation)
    ]
    assert len(phase_operations) == 1
    phase_operation = phase_operations[0]
    assert phase_operation.results == []
    assert phase_operation.phase.get_const() == pytest.approx(0.375)


def test_branch_selected_quantum_return_round_trip() -> None:
    """Serialization preserves deferred quantum array return validation."""
    restored = deserialize(serialize(_branch_selected_array_return))
    block = restored.build()

    pending = list(block.operations)
    return_operations = []
    while pending:
        operation = pending.pop()
        if isinstance(operation, ReturnQuantumArrayElementOperation):
            return_operations.append(operation)
        if isinstance(operation, HasNestedOps):
            for nested in operation.nested_op_lists():
                pending.extend(nested)

    assert len(return_operations) == 1
    executable = QiskitTranspiler().transpile(restored)
    result = executable.sample(
        QiskitTranspiler().executor(),
        shots=16,
    ).result()
    assert result.results == [((1, 1), 16)]


def test_signature_parameter_kind_is_preserved() -> None:
    """The reconstructed signature retains Python parameter kinds."""
    restored = deserialize(serialize(_parameterized))

    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in restored.signature.parameters.values()
    )


def test_value_encoder_rejects_active_duplicate_uuid_structure() -> None:
    """A recursive value edge cannot hide a conflicting duplicate UUID."""
    one = Value(type=UIntType(), name="one").with_const(1)
    two = Value(type=UIntType(), name="two").with_const(2)
    root = ArrayValue(type=QubitType(), name="root", shape=(one,))
    conflicting = dataclasses.replace(root, shape=(two,))
    sliced = dataclasses.replace(
        root,
        slice_of=conflicting,
        slice_start=Value(type=UIntType(), name="start").with_const(0),
        slice_step=one,
    )

    with pytest.raises(ValueError, match="conflicting structures"):
        _EncodeContext().register_value(sliced)


def test_value_encoder_rejects_conflict_across_owned_blocks() -> None:
    """Independent owned blocks cannot reuse one UUID for different arrays."""
    one = Value(type=UIntType(), name="one").with_const(1)
    two = Value(type=UIntType(), name="two").with_const(2)
    source_value = ArrayValue(type=QubitType(), name="q", shape=(one,))
    fallback_value = dataclasses.replace(source_value, shape=(two,))
    source = Block(
        name="source",
        label_args=["q"],
        input_values=[source_value],
        output_values=[source_value],
    )
    fallback = Block(
        name="fallback",
        label_args=["q"],
        input_values=[fallback_value],
        output_values=[fallback_value],
    )
    root = Block(
        operations=[
            InverseBlockOperation(
                source_block=source,
                implementation_block=fallback,
            )
        ]
    )

    with pytest.raises(ValueError, match="conflicting structures"):
        _encode_block(root, _EncodeContext())


def test_uuid_remapper_rejects_conflicting_source_structures() -> None:
    """Alpha-renaming cannot launder one UUID with incompatible shapes."""
    one = Value(type=UIntType(), name="one").with_const(1)
    two = Value(type=UIntType(), name="two").with_const(2)
    source_value = ArrayValue(type=QubitType(), name="q", shape=(one,))
    conflicting = dataclasses.replace(source_value, shape=(two,))
    block = Block(
        label_args=["first", "second"],
        input_values=[source_value, conflicting],
    )

    with pytest.raises(ValueError, match="conflicting structures"):
        UUIDRemapper().clone_block(block)


def test_uuid_remapper_rejects_recursive_inverse_source() -> None:
    """Alpha-renaming rejects an inverse source edge back to its own block."""
    block = Block(name="recursive")
    block.operations.append(
        InverseBlockOperation(
            source_block=block,
            implementation_block=Block(name="fallback"),
        )
    )

    with pytest.raises(ValueError, match="recursive inverse source"):
        UUIDRemapper().clone_block(block)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (0.0, -0.0),
        (
            struct.unpack(">d", struct.pack(">Q", 0x7FF8000000000001))[0],
            struct.unpack(">d", struct.pack(">Q", 0x7FF8000000000002))[0],
        ),
    ],
    ids=["signed_zero", "nan_payload"],
)
def test_value_encoder_compares_float_metadata_by_bits(
    first: float,
    second: float,
) -> None:
    """Same-UUID metadata with different binary64 bits is rejected."""
    value = Value(type=FloatType(), name="value").with_const(first)
    conflicting = dataclasses.replace(
        value,
        metadata=Value(type=FloatType(), name="other").with_const(second).metadata,
    )
    context = _EncodeContext()
    context.register_value(value)

    with pytest.raises(ValueError, match="conflicting structures"):
        context.register_value(conflicting)


def test_value_encoder_accepts_repeated_nan_with_identical_bits() -> None:
    """Same-UUID NaN metadata is stable when its payload bits match."""
    nan = struct.unpack(">d", struct.pack(">Q", 0x7FF8000000000001))[0]
    value = Value(type=FloatType(), name="value").with_const(nan)
    repeated = dataclasses.replace(
        value,
        metadata=Value(type=FloatType(), name="other").with_const(nan).metadata,
    )
    context = _EncodeContext()

    assert context.register_value(value) == value.uuid
    assert context.register_value(repeated) == value.uuid


def test_symbolic_vector_inverse_round_trips_with_disjoint_fallback_values() -> None:
    """Atomic inverse serialization alpha-renames its specialized fallback."""
    inverse_operation = next(
        operation
        for operation in _atomic_symbolic_vector_inverse.block.operations
        if isinstance(operation, InverseBlockOperation)
    )
    assert inverse_operation.source_block is not None
    assert inverse_operation.implementation_block is not None
    assert {
        value.uuid for value in inverse_operation.source_block.input_values
    }.isdisjoint(
        value.uuid for value in inverse_operation.implementation_block.input_values
    )

    restored = deserialize(serialize(_atomic_symbolic_vector_inverse))
    executable = QiskitTranspiler().transpile(restored)
    result = executable.sample(QiskitTranspiler().executor(), shots=16).result()

    assert result.results == [((1, 1), 16)]


def test_inverse_round_trip_preserves_free_classical_capture() -> None:
    """Fallback alpha-renaming leaves enclosing runtime parameters shared."""
    restored = deserialize(serialize(_inverse_with_free_classical_capture))
    block = restored.build(parameters=["theta"])

    assert list(block.parameters) == ["theta"]
    executable = QiskitTranspiler().transpile(restored, parameters=["theta"])
    result = executable.sample(
        QiskitTranspiler().executor(),
        shots=16,
        bindings={"theta": np.pi},
    ).result()
    assert result.results == [(1, 16)]


def test_inverse_round_trip_remaps_vector_parameter_elements() -> None:
    """Fallback vector elements point to its fresh classical input array."""
    restored = deserialize(serialize(_vector_parameter_inverse_round_trip))
    executable = QiskitTranspiler().transpile(
        restored,
        bindings={"angles": [0.37, -0.81]},
    )
    result = executable.sample(QiskitTranspiler().executor(), shots=16).result()

    assert result.results == [((0, 0), 16)]
