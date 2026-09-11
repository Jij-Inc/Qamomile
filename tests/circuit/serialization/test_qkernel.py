"""Tests for static qkernel protobuf serialization."""

from __future__ import annotations

import dataclasses
import inspect
import struct
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

import qamomile.circuit as qmc
from qamomile.circuit.estimator import ResourceTraceNode
from qamomile.circuit.estimator._resource_constraints import (
    _ConstraintOrigin,
    _ConstraintProvenance,
    _ConstraintRange,
    _ResourceConstraint,
)
from qamomile.circuit.frontend.composite_gate import configure_composite
from qamomile.circuit.frontend.qkernel_callable import (
    qkernel_callable_attrs,
    qkernel_callable_ref,
)
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.arithmetic_operations import (
    CompOp,
    CompOpKind,
    UnaryMathOp,
    UnaryMathOpKind,
)
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
    DecodeQIntOperation,
    ReturnQuantumArrayElementOperation,
)
from qamomile.circuit.ir.operation.control_flow import HasNestedOps
from qamomile.circuit.ir.operation.gate import (
    GateOperation,
    GateOperationType,
    MeasureQFixedOperation,
    MeasureQIntOperation,
)
from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.serialize.decode import _decode_block, _DecodeContext
from qamomile.circuit.ir.serialize.encode import (
    _OP_ENCODERS,
    _encode_block,
    _EncodeContext,
)
from qamomile.circuit.ir.types.primitives import (
    BitType,
    FloatType,
    QubitType,
    UIntType,
)
from qamomile.circuit.ir.types.q_register import QFixedType, QUIntType
from qamomile.circuit.ir.uuid_remapper import UUIDRemapper
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.serialization import (
    QAMOMILE_VERSION,
    SerializedQKernel,
    deserialize,
    serialize,
)
from qamomile.circuit.serialization.canonical import canonicalize_graph
from qamomile.circuit.serialization.decode import from_dict as kernel_from_dict
from qamomile.circuit.serialization.encode import to_dict as kernel_to_dict
from qamomile.circuit.serialization.graph_protobuf import (
    _OPERATION_TO_PROTO,
    _operation_from_proto,
    _operation_to_proto,
    _validate_operation_fields,
)
from qamomile.circuit.serialization.kernel import _StaticBindingResolver
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
def _qint_register() -> qmc.QInt:
    """Return a three-carrier unsigned quantum integer."""
    return qmc.cast(qmc.qubit_array(3, "qint"), qmc.QInt)


@qmc.qkernel
def _qint_measurement() -> qmc.UInt:
    """Measure a three-carrier unsigned quantum integer."""
    register = qmc.cast(qmc.qubit_array(3, "qint"), qmc.QInt)
    return qmc.measure(register)


@qmc.qkernel
def _asymmetric_qint_measurement() -> qmc.UInt:
    """Measure an endian-sensitive QInt value after serialization."""
    register = qmc.qubit_array(3, "qint")
    register[0] = qmc.x(register[0])
    register[1] = qmc.x(register[1])
    return qmc.measure(qmc.cast(register, qmc.QInt))


@qmc.qkernel
def _sliced_qint_measurement() -> qmc.UInt:
    """Measure a strided carrier view after serialization."""
    register = qmc.qubit_array(4, "qint")
    register[1] = qmc.x(register[1])
    return qmc.measure(qmc.cast(register[1::2], qmc.QInt))


@qmc.qkernel
def _zero_width_qint_measurement() -> qmc.UInt:
    """Measure the unique value represented by an empty QInt register."""
    register = qmc.cast(qmc.qubit_array(0, "qint"), qmc.QInt)
    return qmc.measure(register)


@qmc.qkernel
def _symbolic_width_qint_measurement(
    register: qmc.Vector[qmc.Qubit],
) -> qmc.UInt:
    """Measure a QInt whose carrier width is supplied by the caller."""
    return qmc.measure(qmc.cast(register, qmc.QInt))


@qmc.qkernel
def _merged_symbolic_width_qint_measurement(n: qmc.UInt) -> qmc.UInt:
    """Measure a symbolic-width QInt merged from equivalent runtime branches."""
    register = qmc.qubit_array(n, "register")
    register[0] = qmc.x(register[0])
    selector = qmc.measure(qmc.qubit("selector"))
    if selector:
        value = qmc.cast(register, qmc.QInt)
    else:
        value = qmc.cast(register, qmc.QInt)
    return qmc.measure(value)


@qmc.qkernel
def _merged_symbolic_width_qfixed_measurement(n: qmc.UInt) -> qmc.Float:
    """Measure a symbolic-width QFixed merged from equivalent runtime branches."""
    register = qmc.qubit_array(n, "register")
    register[0] = qmc.x(register[0])
    selector = qmc.measure(qmc.qubit("selector"))
    if selector:
        value = qmc.cast(register, qmc.QFixed)
    else:
        value = qmc.cast(register, qmc.QFixed)
    return qmc.measure(value)


@qmc.qkernel
def _qfixed_measurement() -> qmc.Float:
    """Measure a three-carrier fixed-point register reading 0.75."""
    register = qmc.qubit_array(3, "qfixed")
    register[0] = qmc.x(register[0])
    register[1] = qmc.x(register[1])
    return qmc.measure(qmc.cast(register, qmc.QFixed, int_bits=1))


@qmc.qkernel
def _sliced_qfixed_measurement() -> qmc.Float:
    """Measure a strided QFixed carrier view after serialization."""
    register = qmc.qubit_array(4, "qfixed")
    register[1] = qmc.x(register[1])
    return qmc.measure(qmc.cast(register[1::2], qmc.QFixed, int_bits=1))


@qmc.qkernel
def _symbolic_width_qfixed_measurement(
    register: qmc.Vector[qmc.Qubit],
) -> qmc.Float:
    """Measure a QFixed whose carrier width is supplied by the caller."""
    return qmc.measure(qmc.cast(register, qmc.QFixed))


@qmc.qkernel
def _zero_width_qfixed_measurement() -> qmc.Float:
    """Measure the unique value represented by an empty QFixed register."""
    register = qmc.cast(qmc.qubit_array(0, "qfixed"), qmc.QFixed)
    return qmc.measure(register)


@qmc.qkernel
def _merged_qfixed_measurement() -> qmc.Float:
    """Measure a QFixed merged from equivalent runtime-if branches."""
    register = qmc.qubit_array(2, "qfixed")
    register[1] = qmc.x(register[1])
    selector = qmc.measure(qmc.qubit("selector"))
    if selector:
        value = qmc.cast(register, qmc.QFixed, int_bits=1)
    else:
        value = qmc.cast(register, qmc.QFixed, int_bits=1)
    return qmc.measure(value)


@qmc.qkernel
def _qfixed_register() -> qmc.QFixed:
    """Return a two-carrier fixed-point register built by a sub-kernel."""
    register = qmc.qubit_array(2, "qfixed")
    register[1] = qmc.x(register[1])
    return qmc.cast(register, qmc.QFixed, int_bits=1)


@qmc.qkernel
def _invoked_qfixed_measurement() -> qmc.Float:
    """Measure a QFixed returned by an invoked sub-kernel."""
    return qmc.measure(_qfixed_register())


@qmc.qkernel
def _unary_math_width(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Allocate a register from independent log2 and ceil operations."""
    return qmc.qubit_array(qmc.ceil(qmc.log2(n)), "q")


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


@qmc.composite_gate(name="symbolic_width_box")
def _symbolic_width_box(
    qubits: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Apply X to the first qubit of a symbolic-width register.

    Args:
        qubits (qmc.Vector[qmc.Qubit]): Nonempty symbolic-width register.

    Returns:
        qmc.Vector[qmc.Qubit]: Register with its first qubit flipped.
    """
    qubits[0] = qmc.x(qubits[0])
    return qubits


@qmc.qkernel
def _symbolic_width_inverse_helper(
    qubits: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Route a symbolic-width register through a preserved composite.

    Args:
        qubits (qmc.Vector[qmc.Qubit]): Symbolic-width register to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated register from the preserved composite.
    """
    return _symbolic_width_box(qubits)


@qmc.qkernel
def _bound_symbolic_vector_inverse(width: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Inverse-call a symbolic-vector kernel at a bound width.

    Args:
        width (qmc.UInt): Compile-time vector width.

    Returns:
        qmc.Vector[qmc.Bit]: Measured inverted all-zero register.
    """
    qubits = qmc.qubit_array(width, "qubits")
    qubits = qmc.inverse(_symbolic_width_inverse_helper)(qubits)
    return qmc.measure(qubits)


@qmc.qkernel
def _asymmetric_block_encoding_unitary(
    signal: qmc.Vector[qmc.Qubit],
    system: qmc.Vector[qmc.Qubit],
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Flip one system qubit while preserving an independent signal.

    Args:
        signal (qmc.Vector[qmc.Qubit]): Pass-through signal register.
        system (qmc.Vector[qmc.Qubit]): One-qubit system register.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Preserved signal
            and flipped system registers.
    """
    system[0] = qmc.x(system[0])
    return signal, system


@qmc.qkernel
def _apply_static_block_encoding(
    signal: qmc.Vector[qmc.Qubit],
    system: qmc.Vector[qmc.Qubit],
    encoding: qmc.LCUBlockEncoding,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Apply a unitary reached through a static block-encoding slot.

    Args:
        signal (qmc.Vector[qmc.Qubit]): Descriptor-sized signal register.
        system (qmc.Vector[qmc.Qubit]): Descriptor-sized system register.
        encoding (qmc.LCUBlockEncoding): Compile-time block encoding.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Registers after
            the bound unitary.
    """
    return encoding.unitary(signal, system)


@qmc.qkernel
def _controlled_static_inverse(
    encoding: qmc.LCUBlockEncoding,
) -> qmc.Vector[qmc.Bit]:
    """Control an inverse helper after static descriptor specialization.

    Args:
        encoding (qmc.LCUBlockEncoding): Compile-time block encoding.

    Returns:
        qmc.Vector[qmc.Bit]: Measured system register.
    """

    @qmc.qkernel
    def inverse_member(
        signal: qmc.Vector[qmc.Qubit],
        system: qmc.Vector[qmc.Qubit],
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        """Expose the static-argument inverse through outer control.

        Args:
            signal (qmc.Vector[qmc.Qubit]): Descriptor-sized signal register.
            system (qmc.Vector[qmc.Qubit]): Descriptor-sized system register.

        Returns:
            tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Inverted
                signal and system registers.
        """
        return qmc.inverse(_apply_static_block_encoding)(
            signal,
            system,
            encoding,
        )

    outer = qmc.x(qmc.qubit("outer"))
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    outer, signal, system = qmc.control(inverse_member)(outer, signal, system)
    return qmc.measure(system)


@qmc.qkernel
def _symbolic_pair_inverse_helper(
    signal: qmc.Vector[qmc.Qubit],
    system: qmc.Vector[qmc.Qubit],
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Route two symbolic registers through one preserved system operation.

    Args:
        signal (qmc.Vector[qmc.Qubit]): Pass-through symbolic register.
        system (qmc.Vector[qmc.Qubit]): Symbolic register whose first qubit is
            flipped.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Preserved signal
            and updated system registers.
    """
    system = _symbolic_width_box(system)
    return signal, system


@qmc.qkernel
def _controlled_bound_symbolic_pair_inverse(
    signal_width: qmc.UInt,
    system_width: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    """Control a nested inverse after two register widths are bound.

    Args:
        signal_width (qmc.UInt): Compile-time signal width.
        system_width (qmc.UInt): Compile-time system width.

    Returns:
        qmc.Vector[qmc.Bit]: Measured system register.
    """

    @qmc.qkernel
    def inverse_member(
        signal: qmc.Vector[qmc.Qubit],
        system: qmc.Vector[qmc.Qubit],
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        """Expose a symbolic two-register inverse through outer control.

        Args:
            signal (qmc.Vector[qmc.Qubit]): Symbolic signal register.
            system (qmc.Vector[qmc.Qubit]): Symbolic system register.

        Returns:
            tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Inverted
                signal and system registers.
        """
        return qmc.inverse(_symbolic_pair_inverse_helper)(signal, system)

    outer = qmc.x(qmc.qubit("outer"))
    signal = qmc.qubit_array(signal_width, "signal")
    system = qmc.qubit_array(system_width, "system")
    outer, signal, system = qmc.control(inverse_member)(outer, signal, system)
    return qmc.measure(system)


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
def _nested_container_return(
    value: qmc.Tuple[
        qmc.Dict[qmc.UInt, qmc.Bit],
        qmc.Tuple[qmc.UInt, qmc.Float],
    ],
) -> qmc.Tuple[
    qmc.Dict[qmc.UInt, qmc.Bit],
    qmc.Tuple[qmc.UInt, qmc.Float],
]:
    """Return nested structural values with an unbound Dict.

    Args:
        value (qmc.Tuple[qmc.Dict[qmc.UInt, qmc.Bit], qmc.Tuple[qmc.UInt, qmc.Float]]):
            Nested structural input.

    Returns:
        qmc.Tuple[qmc.Dict[qmc.UInt, qmc.Bit], qmc.Tuple[qmc.UInt, qmc.Float]]:
            Unchanged structural value.
    """
    return value


_MUTABLE_RETURN_ALIAS = qmc.Bit


@qmc.qkernel
def _cached_return_alias() -> _MUTABLE_RETURN_ALIAS:
    """Return a Bit through a mutable deferred global alias.

    Returns:
        _MUTABLE_RETURN_ALIAS: Bit resolved when the kernel is decorated.
    """
    return qmc.bit(False)


@qmc.qkernel
def _late_serialized_return_alias() -> _LATE_SERIALIZED_RETURN_ALIAS:
    """Return a Bit through an alias defined after decoration.

    Returns:
        _LATE_SERIALIZED_RETURN_ALIAS: Return contract resolved by the first
        serialization trace.
    """
    return qmc.bit(False)


_LATE_SERIALIZED_RETURN_ALIAS = qmc.Bit


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
def _singleton_tuple_return() -> tuple[qmc.Bit]:
    """Expose a one-element Python tuple return annotation."""
    return (qmc.bit(False),)


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
            engine="test-engine",
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
def _select_rx_parameter_first(
    angle: qmc.Float,
    target: qmc.Qubit,
) -> qmc.Qubit:
    """Apply an X rotation with a parameter-first case signature.

    Args:
        angle (qmc.Float): Rotation angle in radians.
        target (qmc.Qubit): Qubit to rotate.

    Returns:
        qmc.Qubit: Rotated target qubit.
    """
    return qmc.rx(target, angle)


@qmc.qkernel
def _select_rz_parameter_first(
    angle: qmc.Float,
    target: qmc.Qubit,
) -> qmc.Qubit:
    """Apply a Z rotation with a parameter-first case signature.

    Args:
        angle (qmc.Float): Rotation angle in radians.
        target (qmc.Qubit): Qubit to rotate.

    Returns:
        qmc.Qubit: Rotated target qubit.
    """
    return qmc.rz(target, angle)


@qmc.qkernel
def _parameter_before_target_select_program() -> qmc.Bit:
    """Select between parameter-first rotation cases.

    Returns:
        qmc.Bit: Measurement of the selected rotation target.
    """
    index = qmc.qubit("index")
    target = qmc.qubit("target")
    index, target = qmc.select(
        [_select_rx_parameter_first, _select_rz_parameter_first]
    )(index, 0.5, target)
    return qmc.measure(target)


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

_explicit_signature_oracle = qmc.Oracle(
    name="serialization_explicit_signature_oracle",
    num_qubits=1,
    num_control_qubits=1,
    signature=qmc.CallableSignature(
        inputs=[qmc.Qubit],
        outputs=[qmc.Qubit],
    ),
)

_nested_serialization_oracle = qmc.Oracle(
    name="nested_serialization_oracle",
    num_qubits=1,
    cost=qmc.ResourceEstimate(
        calls=qmc.CallResources(
            queries_by_name={"nested_serialization_oracle": 1},
        ),
    ),
)


@qmc.qkernel
def _nested_oracle_helper(target: qmc.Qubit) -> qmc.Qubit:
    """Invoke an Oracle from a callable-table forward reference."""
    (target,) = _nested_serialization_oracle(target)
    return target


@qmc.qkernel
def _calls_nested_oracle_helper() -> qmc.Bit:
    """Call a nested qkernel whose body contains an Oracle invocation."""
    target = _nested_oracle_helper(qmc.qubit("target"))
    return qmc.measure(target)


@qmc.qkernel
def _calls_controlled_oracle() -> qmc.Bit:
    """Invoke an oracle whose signature already includes its control qubit."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = _controlled_oracle(target, controls=(control,))
    return qmc.measure(control)


_OPAQUE_COST_SIZE = sp.Symbol(
    "opaque_size",
    integer=True,
    nonnegative=True,
)
_OPAQUE_COST_INDEX = sp.Dummy(
    "opaque_size",
    integer=True,
    nonnegative=True,
)
_OPAQUE_COST_ASSUMPTION = qmc.ResourceAssumption(
    "Serialized opaque cost uses a test model.",
    source="serialization_fixed_cost",
)
_FIXED_OPAQUE_COST = qmc.ResourceEstimate(
    width=qmc.WidthResources(
        input_qubits=1,
        allocated_qubits=2,
        clean_ancilla_qubits=3,
        dirty_ancilla_qubits=4,
        peak_qubits=10,
    ),
    gates=qmc.GateResources(
        total=11,
        single_qubit=12,
        two_qubit=13,
        multi_qubit=14,
        clifford=15,
        rotation=16,
        t=17,
        toffoli=18,
        non_clifford=19,
    ),
    depth=qmc.DepthResources(
        depth=20,
        clifford_depth=21,
        rotation_depth=22,
        t_depth=23,
        toffoli_depth=24,
        non_clifford_depth=25,
        measurement_depth=26,
        gate_depth=27,
        reset_depth=28,
    ),
    calls=qmc.CallResources(
        calls_by_name={"inner_call": _OPAQUE_COST_SIZE + 1},
        queries_by_name={"inner_query": 2 * _OPAQUE_COST_SIZE},
    ),
    measurements=qmc.MeasurementResources(total=29),
    resets=qmc.ResetResources(total=30),
    assumptions=(_OPAQUE_COST_ASSUMPTION,),
    trace=ResourceTraceNode(
        name="fixed_opaque_cost",
        source_kind="opaque_cost",
        strategy="serialized",
        summary="all resource fields",
        assumptions=(_OPAQUE_COST_ASSUMPTION,),
        children=(
            ResourceTraceNode(
                name="inner",
                source_kind="primitive",
            ),
        ),
    ),
    derivation=qmc.EstimateDerivation.MODELED,
    quality=qmc.EstimateQuality.CONSERVATIVE,
    approximation=qmc.ApproximationStatus.APPROXIMATE,
    control_decomposition=qmc.ControlDecomposition.ABSTRACT,
    _constraints=(
        _ResourceConstraint(
            expression=_OPAQUE_COST_INDEX + 1,
            minimum=None,
            expected=_OPAQUE_COST_INDEX + 1,
            label="Opaque test index",
            ranges=(
                _ConstraintRange(
                    symbol=_OPAQUE_COST_INDEX,
                    start=0,
                    step=1,
                    iterations=_OPAQUE_COST_SIZE,
                ),
            ),
        ),
    ),
).repeat(_OPAQUE_COST_SIZE)
_fixed_cost_oracle = qmc.Oracle(
    name="serialization_fixed_cost_oracle",
    num_qubits=1,
    cost=_FIXED_OPAQUE_COST,
)


@qmc.qkernel
def _calls_fixed_cost_oracle() -> qmc.Bit:
    """Invoke an oracle carrying a fixed symbolic resource estimate."""
    target = qmc.qubit("target")
    (target,) = _fixed_cost_oracle(target)
    return qmc.measure(target)


def _callback_opaque_cost(
    ctx: qmc.OpaqueCostContext,
) -> qmc.ResourceEstimate:
    """Return one process-local callback cost for rejection testing.

    Args:
        ctx (qmc.OpaqueCostContext): Definition-level opaque call context.

    Returns:
        qmc.ResourceEstimate: One single-qubit gate for the definition.
    """
    return qmc.ResourceEstimate(
        gates=qmc.GateResources(
            total=ctx.target_qubits,
            single_qubit=ctx.target_qubits,
        )
    )


_callback_cost_oracle = qmc.Oracle(
    name="serialization_callback_cost_oracle",
    num_qubits=1,
    cost=_callback_opaque_cost,
)


@qmc.qkernel
def _calls_callback_cost_oracle() -> qmc.Bit:
    """Invoke an oracle carrying a process-local cost callback."""
    target = qmc.qubit("target")
    (target,) = _callback_cost_oracle(target)
    return qmc.measure(target)


@qmc.qkernel
def _calls_controlled_explicit_signature_oracle() -> qmc.Bit:
    """Add one call-site control to a declared-controlled explicit Oracle."""
    added = qmc.qubit("added")
    declared = qmc.qubit("declared")
    target = qmc.qubit("target")
    added, declared, target = qmc.control(_explicit_signature_oracle)(
        added,
        declared,
        target,
    )
    return qmc.measure(target)


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
    """Invoke the original QFT for engine-output comparison."""
    qubits = qmc.qubit_array(n, "q")
    qubits = qmc.qft(qubits)
    return qmc.measure(qubits[0])


@qmc.qkernel
def _calls_original_iqft(n: qmc.UInt) -> qmc.Bit:
    """Invoke the original IQFT for engine-output comparison."""
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
        object: First engine circuit.
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


def test_unary_math_operation_type_is_appended_to_wire_enum() -> None:
    """Adding unary math leaves every previously assigned enum value intact."""
    assert pb.RETURN_QUANTUM_ARRAY_ELEMENT_OPERATION == 33
    assert pb.UNARY_MATH_OPERATION == 34


def test_qint_wire_enums_are_appended() -> None:
    """QInt additions leave every previously assigned enum value intact."""
    assert pb.QAMOMILE_QINT == 16
    assert pb.MEASURE_QINT_OPERATION == 35
    assert pb.DECODE_QINT_OPERATION == 36


def test_qint_frontend_annotation_roundtrips() -> None:
    """Preserve the QInt annotation and its concrete carrier width."""
    message = _message(_qint_register)

    assert message.results[0].annotation.kind == pb.QAMOMILE_QINT

    restored = _restore(message)
    assert restored.output_types == [qmc.QInt]
    assert restored.block.output_values[0].type == QUIntType(width=3)


def test_measure_qint_operation_roundtrips() -> None:
    """Derive QInt measurement width from the operand after a protobuf roundtrip."""
    message = _message(_qint_measurement)
    encoded = next(
        operation
        for operation in message.body.operations
        if operation.operation_type == pb.MEASURE_QINT_OPERATION
    )

    assert not encoded.HasField("num_bits")

    restored = _restore(message)
    decoded = next(
        operation
        for operation in restored.block.operations
        if isinstance(operation, MeasureQIntOperation)
    )
    assert decoded.num_bits == 3
    assert decoded.operands[0].type == QUIntType(width=3)
    assert decoded.results[0].type == UIntType()


def test_qint_roundtrip_executes_with_preserved_carrier_order() -> None:
    """Public protobuf roundtrip retains QInt carrier identities and order."""
    restored = deserialize(serialize(_asymmetric_qint_measurement))
    transpiler = QiskitTranspiler()

    result = (
        transpiler.transpile(restored).sample(transpiler.executor(), shots=16).result()
    )

    assert result.results == [(3, 16)]


def test_decode_qint_semantic_operation_roundtrips() -> None:
    """Preserve the integer decoder in the closed semantic IR codec."""
    width = Value(type=UIntType(), name="width").with_const(3)
    bits = ArrayValue(type=BitType(), name="bits", shape=(width,))
    result = Value(type=UIntType(), name="result")
    block = Block(
        operations=[DecodeQIntOperation(operands=[bits], results=[result])],
        output_values=[result],
    )
    encode_context = _EncodeContext()

    encoded = _encode_block(block, encode_context)
    restored = _decode_block(
        encoded,
        _DecodeContext(encode_context.value_table_dicts),
    )

    decoded = restored.operations[0]
    assert isinstance(decoded, DecodeQIntOperation)
    assert decoded.num_bits == 3
    assert isinstance(decoded.operands[0], ArrayValue)
    assert decoded.operands[0].type == BitType()
    assert decoded.results[0].type == UIntType()


def test_decode_qint_protobuf_operation_roundtrips() -> None:
    """Preserve the integer decoder through its typed protobuf mapping."""
    record = {
        "$type": "DecodeQIntOperation",
        "operand_refs": ["bits"],
        "result_refs": ["result"],
    }

    message = _operation_to_proto(record)

    assert message.operation_type == pb.DECODE_QINT_OPERATION
    assert not message.HasField("num_bits")
    assert _operation_from_proto(message) == record


def test_qint_operations_reject_stored_num_bits_field() -> None:
    """Reject a stored num_bits on QInt operations; the width is derived."""
    message = _message(_qint_measurement)
    encoded = next(
        operation
        for operation in message.body.operations
        if operation.operation_type == pb.MEASURE_QINT_OPERATION
    )
    encoded.num_bits = 3

    with pytest.raises(ValueError, match="has unrelated fields"):
        _restore(message)

    decode_message = _operation_to_proto(
        {
            "$type": "DecodeQIntOperation",
            "operand_refs": ["bits"],
            "result_refs": ["result"],
        }
    )
    decode_message.num_bits = 3

    with pytest.raises(ValueError, match="has unrelated fields"):
        _validate_operation_fields(decode_message)


def test_qint_cast_rejects_reordered_carrier_metadata() -> None:
    """Reject carrier metadata whose order diverges from the cast operation."""
    message = _message(_asymmetric_qint_measurement)
    encoded_cast = next(
        operation
        for operation in message.body.operations
        if operation.operation_type == pb.CAST_OPERATION
    )
    result = next(
        value
        for value in message.value_table
        if value.uuid == encoded_cast.result_refs[0]
    )
    reordered = list(reversed(result.metadata.cast.qubit_uuids))
    del result.metadata.cast.qubit_uuids[:]
    result.metadata.cast.qubit_uuids.extend(reordered)

    with pytest.raises(
        ValueError,
        match="QInt qubit_mapping disagrees with cast metadata",
    ):
        _restore(message)


def test_qint_cast_rejects_jointly_reordered_mapping_and_metadata() -> None:
    """Reject a self-consistent carrier list that disagrees with source order."""
    message = _message(_asymmetric_qint_measurement)
    encoded_cast = next(
        operation
        for operation in message.body.operations
        if operation.operation_type == pb.CAST_OPERATION
    )
    result = next(
        value
        for value in message.value_table
        if value.uuid == encoded_cast.result_refs[0]
    )
    reordered_uuids = list(reversed(encoded_cast.qubit_mapping))
    reordered_logical_ids = list(reversed(result.metadata.cast.qubit_logical_ids))
    del encoded_cast.qubit_mapping[:]
    encoded_cast.qubit_mapping.extend(reordered_uuids)
    del result.metadata.cast.qubit_uuids[:]
    result.metadata.cast.qubit_uuids.extend(reordered_uuids)
    del result.metadata.cast.qubit_logical_ids[:]
    result.metadata.cast.qubit_logical_ids.extend(reordered_logical_ids)

    with pytest.raises(
        ValueError,
        match="QInt carrier order disagrees with its source",
    ):
        _restore(message)


def test_qint_slice_roundtrip_executes_with_source_order() -> None:
    """Preserve a strided QInt carrier order through public protobuf bytes."""
    restored = deserialize(serialize(_sliced_qint_measurement))
    transpiler = QiskitTranspiler()

    result = (
        transpiler.transpile(restored).sample(transpiler.executor(), shots=16).result()
    )

    assert result.results == [(1, 16)]


def test_decode_qint_rejects_symbolic_bit_array_width() -> None:
    """Reject a decoder whose lowered bit-array width remains unresolved."""
    width = Value(type=UIntType(), name="width").with_parameter("width")
    bits = ArrayValue(type=BitType(), name="bits", shape=(width,))
    result = Value(type=UIntType(), name="result")
    block = Block(
        input_values=[bits, width],
        operations=[DecodeQIntOperation(operands=[bits], results=[result])],
        output_values=[result],
    )

    with pytest.raises(
        ValueError,
        match="QInt decoder bit-array width must be concrete",
    ):
        validate_qkernel_ir(block)


def test_zero_width_qint_measurement_roundtrips() -> None:
    """Preserve the empty QInt whose unsigned value is necessarily zero."""
    restored = _restore(_message(_zero_width_qint_measurement))
    measurement = next(
        operation
        for operation in restored.block.operations
        if isinstance(operation, MeasureQIntOperation)
    )

    assert measurement.num_bits == 0
    assert measurement.operands[0].type == QUIntType(width=0)


def test_symbolic_width_qint_measurement_roundtrips() -> None:
    """Preserve deferred QInt width until a caller supplies its register."""
    restored = _restore(_message(_symbolic_width_qint_measurement))
    measurement = next(
        operation
        for operation in restored.block.operations
        if isinstance(operation, MeasureQIntOperation)
    )

    assert measurement.num_bits is None
    assert isinstance(measurement.operands[0].type, QUIntType)
    assert isinstance(measurement.operands[0].type.width, Value)


def _sample(kernel: object, **bindings: object) -> list[tuple[object, int]]:
    """Sample a kernel returning one measured scalar on the Qiskit simulator.

    Args:
        kernel (object): QKernel-like object returning a measured scalar.
        **bindings (object): Compile-time bindings forwarded to ``transpile``.

    Returns:
        list[tuple[object, int]]: Sampled ``(value, count)`` pairs.
    """
    transpiler = QiskitTranspiler()
    return (
        transpiler.transpile(kernel, bindings=bindings)  # type: ignore[arg-type]
        .sample(transpiler.executor(), shots=16)
        .result()
        .results
    )


def _qfixed_sample(kernel: object) -> list[tuple[float, int]]:
    """Sample a QFixed-measuring kernel on the Qiskit simulator.

    Args:
        kernel (object): QKernel-like object returning a measured ``Float``.

    Returns:
        list[tuple[float, int]]: Sampled ``(value, count)`` pairs.
    """
    return _sample(kernel)  # type: ignore[return-value]


def test_symbolic_width_qint_merged_by_runtime_if_roundtrips() -> None:
    """A runtime-if merged symbolic-width QInt decodes its bound register.

    At trace time the carrier list is empty, so after serialization the
    measurement operand is the branch merge result whose casts live inside
    the ``IfOperation`` bodies. Plan-time lowering must resolve the source
    vector through the cast metadata instead of decoding an empty register.
    """
    restored = deserialize(serialize(_merged_symbolic_width_qint_measurement))

    assert _sample(_merged_symbolic_width_qint_measurement, n=3) == [(1, 16)]
    assert _sample(restored, n=3) == [(1, 16)]


def test_symbolic_width_qfixed_merged_by_runtime_if_roundtrips() -> None:
    """A runtime-if merged symbolic-width QFixed decodes its bound register."""
    restored = deserialize(serialize(_merged_symbolic_width_qfixed_measurement))

    assert _sample(_merged_symbolic_width_qfixed_measurement, n=3) == [(0.125, 16)]
    assert _sample(restored, n=3) == [(0.125, 16)]


def _tampered_qfixed_cast(
    kernel: object,
) -> tuple[pb.QKernel, pb.Operation, pb.Value]:
    """Serialize a kernel and locate its QFixed cast for protobuf tampering.

    Args:
        kernel (object): QKernel-like object containing one cast operation.

    Returns:
        tuple[pb.QKernel, pb.Operation, pb.Value]: The parsed message, its
            cast operation, and the cast result's value-table entry.
    """
    message = _message(kernel)
    encoded_cast = next(
        operation
        for operation in message.body.operations
        if operation.operation_type == pb.CAST_OPERATION
    )
    result = next(
        value
        for value in message.value_table
        if value.uuid == encoded_cast.result_refs[0]
    )
    return message, encoded_cast, result


def _reverse_repeated(field: object) -> None:
    """Reverse a repeated protobuf field in place.

    Args:
        field (object): Repeated scalar container to reverse.
    """
    reordered = list(reversed(field))  # type: ignore[call-overload]
    del field[:]  # type: ignore[index]
    field.extend(reordered)  # type: ignore[attr-defined]


def test_qfixed_measurement_roundtrip_executes() -> None:
    """A QFixed measurement survives serialization and reads the same Float."""
    restored = deserialize(serialize(_qfixed_measurement))

    assert _qfixed_sample(restored) == _qfixed_sample(_qfixed_measurement)
    assert _qfixed_sample(restored) == [(0.75, 16)]


def test_qfixed_slice_roundtrip_executes_with_source_order() -> None:
    """Preserve a strided QFixed carrier order through public protobuf bytes."""
    restored = deserialize(serialize(_sliced_qfixed_measurement))

    # Carrier 0 of the view is ``register[1]``, the fractional (LSB) bit.
    assert _qfixed_sample(restored) == _qfixed_sample(_sliced_qfixed_measurement)
    assert _qfixed_sample(restored) == [(0.5, 16)]


def test_symbolic_width_qfixed_measurement_roundtrips() -> None:
    """Preserve deferred QFixed width until a caller supplies its register."""
    restored = _restore(_message(_symbolic_width_qfixed_measurement))
    measurement = next(
        operation
        for operation in restored.block.operations
        if isinstance(operation, MeasureQFixedOperation)
    )

    assert measurement.num_bits == 0
    assert measurement.int_bits == 0
    assert isinstance(measurement.operands[0].type, QFixedType)
    assert isinstance(measurement.operands[0].type.fractional_bits, Value)


def test_zero_width_qfixed_measurement_roundtrips() -> None:
    """Preserve the empty QFixed whose measured value is necessarily zero."""
    restored = _restore(_message(_zero_width_qfixed_measurement))
    measurement = next(
        operation
        for operation in restored.block.operations
        if isinstance(operation, MeasureQFixedOperation)
    )

    assert measurement.num_bits == 0
    assert measurement.operands[0].type == QFixedType(0, 0)


def test_merged_qfixed_measurement_roundtrips() -> None:
    """A runtime-if merged QFixed keeps its carriers through serialization."""
    restored = deserialize(serialize(_merged_qfixed_measurement))

    assert _qfixed_sample(restored) == _qfixed_sample(_merged_qfixed_measurement)
    assert _qfixed_sample(restored) == [(1.0, 16)]


def test_invoked_qfixed_measurement_roundtrips() -> None:
    """A QFixed returned across a qkernel call boundary restores cleanly."""
    restored = _restore(_message(_invoked_qfixed_measurement))

    assert any(
        isinstance(operation, MeasureQFixedOperation)
        for operation in restored.block.operations
    )


def test_qfixed_measurement_rejects_num_bits_mismatch() -> None:
    """Reject a stored QFixed num_bits that disagrees with the operand type."""
    message = _message(_qfixed_measurement)
    encoded = next(
        operation
        for operation in message.body.operations
        if operation.operation_type == pb.MEASURE_QFIXED_OPERATION
    )
    encoded.num_bits = 2

    with pytest.raises(
        ValueError, match="num_bits disagrees with its QFixedType width"
    ):
        _restore(message)


def test_qfixed_measurement_rejects_int_bits_mismatch() -> None:
    """Reject a stored QFixed int_bits that disagrees with the operand type."""
    message = _message(_qfixed_measurement)
    encoded = next(
        operation
        for operation in message.body.operations
        if operation.operation_type == pb.MEASURE_QFIXED_OPERATION
    )
    encoded.int_bits = 2

    with pytest.raises(ValueError, match="int_bits disagrees"):
        _restore(message)


def test_qfixed_cast_rejects_reordered_carrier_metadata() -> None:
    """Reject QFixed carrier metadata whose order diverges from the cast."""
    message, _, result = _tampered_qfixed_cast(_qfixed_measurement)
    _reverse_repeated(result.metadata.cast.qubit_uuids)

    with pytest.raises(
        ValueError,
        match="QFixed qubit_mapping disagrees with cast metadata",
    ):
        _restore(message)


def test_qfixed_cast_rejects_jointly_reordered_mapping_and_metadata() -> None:
    """Reject a self-consistent QFixed carrier list that disagrees with source."""
    message, encoded_cast, result = _tampered_qfixed_cast(_qfixed_measurement)
    _reverse_repeated(encoded_cast.qubit_mapping)
    _reverse_repeated(result.metadata.cast.qubit_uuids)
    _reverse_repeated(result.metadata.cast.qubit_logical_ids)
    _reverse_repeated(result.metadata.qfixed.qubit_uuids)

    with pytest.raises(
        ValueError,
        match="QFixed carrier order disagrees with its source",
    ):
        _restore(message)


def test_qfixed_cast_rejects_missing_cast_metadata() -> None:
    """Reject a QFixed cast result that carries only layout metadata."""
    message, _, result = _tampered_qfixed_cast(_qfixed_measurement)
    result.metadata.ClearField("cast")

    with pytest.raises(ValueError, match="QFixed carrier requires cast metadata"):
        _restore(message)


def test_qfixed_cast_rejects_duplicate_carriers() -> None:
    """Reject a QFixed cast whose carriers alias one physical qubit."""
    message, encoded_cast, result = _tampered_qfixed_cast(_qfixed_measurement)
    first = result.metadata.cast.qubit_uuids[0]
    count = len(result.metadata.cast.qubit_uuids)
    del result.metadata.cast.qubit_uuids[:]
    result.metadata.cast.qubit_uuids.extend([first] * count)
    del encoded_cast.qubit_mapping[:]
    encoded_cast.qubit_mapping.extend([first] * count)

    with pytest.raises(ValueError, match="must be unique"):
        _restore(message)


def test_qfixed_cast_rejects_layout_carriers_disagreeing_with_cast() -> None:
    """Reject QFixed layout metadata whose carriers differ from cast metadata."""
    message, _, result = _tampered_qfixed_cast(_qfixed_measurement)
    _reverse_repeated(result.metadata.qfixed.qubit_uuids)

    with pytest.raises(
        ValueError,
        match="QFixed layout carriers disagree with cast metadata",
    ):
        _restore(message)


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


def test_unary_math_operations_round_trip() -> None:
    """Independent log2 and ceil operations survive protobuf serialization."""
    payload = serialize(_unary_math_width)
    restored = deserialize(payload)
    operations = [
        operation
        for operation in restored.block.operations
        if isinstance(operation, UnaryMathOp)
    ]

    assert [operation.kind for operation in operations] == [
        UnaryMathOpKind.LOG2,
        UnaryMathOpKind.CEIL,
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


def test_inverse_block_vector_control_is_rejected_during_validation() -> None:
    """Serialization validation independently rejects a vector control operand."""
    scalar_control = Value(type=QubitType(), name="scalar_control")
    target = Value(type=QubitType(), name="target")
    scalar_control_result = scalar_control.next_version()
    target_result = target.next_version()
    operation = InverseBlockOperation(
        operands=[scalar_control, target],
        results=[scalar_control_result, target_result],
        num_control_qubits=1,
        num_target_qubits=1,
        source_block=Block(),
        implementation_block=Block(),
        control_value=0,
    )

    # Corrupt an otherwise valid operation after construction to exercise the
    # serialization boundary's independent defense against forged IR.
    dimension = Value(type=UIntType(), name="dimension").with_const(3)
    vector_control = ArrayValue(
        type=QubitType(),
        name="vector_control",
        shape=(dimension,),
    )
    vector_control_result = vector_control.next_version()
    operation.operands[0] = vector_control
    operation.results[0] = vector_control_result
    block = Block(
        input_values=[vector_control, target],
        output_values=[vector_control_result, target_result],
        operations=[operation],
    )

    with pytest.raises(ValueError, match="control operands must be scalar qubits"):
        validate_qkernel_ir(block)


def test_select_parameter_before_target_round_trips_through_dict() -> None:
    """SELECT case declaration order survives dict encoding and decoding."""
    payload = kernel_to_dict(_parameter_before_target_select_program)

    restored = kernel_from_dict(payload)

    assert kernel_to_dict(restored) == payload


def test_select_rejects_unrecognized_callable_attrs_during_dict_decode() -> None:
    """SELECT decoding fails closed on callable metadata outside ``cases``."""
    payload = kernel_to_dict(_parameter_before_target_select_program)
    operation = next(
        operation
        for operation in payload["artifact"]["body"]["operations"]
        if operation["$type"] == "SelectOperation"
    )
    operation["callable_attrs"]["$map"].append(["unexpected", "payload"])

    with pytest.raises(ValueError, match="supports only the 'cases' key"):
        kernel_from_dict(payload)


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


def test_nested_oracle_definition_round_trips_after_forward_linking() -> None:
    """Nested Oracle calls validate after every definition header is linked."""
    restored = deserialize(serialize(_calls_nested_oracle_helper))

    assert kernel_to_dict(restored) == kernel_to_dict(_calls_nested_oracle_helper)


def test_fixed_opaque_resource_cost_round_trips_with_provenance() -> None:
    """Fixed opaque costs retain metrics, symbols, requirements, and guards."""
    message = _message(_calls_fixed_cost_oracle)
    encoded_definition = next(
        entry.definition
        for entry in message.callable_table
        if entry.definition.ref.name == "serialization_fixed_cost_oracle"
    )

    assert encoded_definition.HasField("opaque_cost")

    restored = _restore(message)
    invoke = next(
        operation
        for operation in restored.block.operations
        if isinstance(operation, InvokeOperation)
    )
    assert invoke.definition is not None
    restored_cost = invoke.definition.opaque_cost
    assert isinstance(restored_cost, qmc.ResourceEstimate)
    assert restored_cost.to_dict() == _FIXED_OPAQUE_COST.to_dict()
    assert restored_cost.explain() == _FIXED_OPAQUE_COST.explain()
    assert serialize(restored) == serialize(_calls_fixed_cost_oracle)

    inactive = restored_cost.substitute(opaque_size=0)
    expected_inactive = _FIXED_OPAQUE_COST.substitute(opaque_size=0)
    assert inactive.gates.total == 0
    assert inactive.assumptions == ()
    assert inactive.quality is qmc.EstimateQuality.EXACT
    assert inactive.approximation is qmc.ApproximationStatus.EXACT
    assert inactive.trace == expected_inactive.trace


def test_stale_fixed_opaque_cost_is_rejected_by_public_serializers() -> None:
    """Public qkernel serializers expose stale domain-state rejection."""
    length = sp.Symbol("stale_length", integer=True)
    stale = qmc.ResourceEstimate(
        gates=qmc.GateResources(total=1 + sp.Max(0, length - 1)),
        _constraints=(
            _ResourceConstraint(
                expression=length - 1,
                minimum=0,
                label="stale serialization input",
                provenance=_ConstraintProvenance(
                    origin=_ConstraintOrigin.ARRAY_ACCESS,
                    source_expressions=(length,),
                    root_formal_names=("stale_length",),
                ),
            ),
        ),
    ).simplify()
    stale.gates.total = 999
    oracle = qmc.Oracle(
        name="serialization_stale_cost_oracle",
        num_qubits=1,
        cost=stale,
    )

    @qmc.qkernel
    def calls_stale_cost() -> qmc.Bit:
        """Invoke an oracle whose fixed cost was mutated after simplification.

        Returns:
            qmc.Bit: Measurement of the oracle target.
        """
        target = qmc.qubit("target")
        (target,) = oracle(target)
        return qmc.measure(target)

    with pytest.raises(RuntimeError, match="domain rewrite state"):
        kernel_to_dict(calls_stale_cost)
    with pytest.raises(RuntimeError, match="domain rewrite state"):
        serialize(calls_stale_cost)


def test_fixed_opaque_resource_cost_has_process_deterministic_bytes() -> None:
    """Quantified Dummy identities never leak process randomness into bytes."""
    script = """
from qamomile.circuit.serialization import serialize
from tests.circuit.serialization.test_qkernel import _calls_fixed_cost_oracle

print(serialize(_calls_fixed_cost_oracle).hex())
"""
    repository = Path(__file__).resolve().parents[3]
    payloads = [
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        for _ in range(2)
    ]

    assert payloads[1:] == payloads[:-1]


@pytest.mark.parametrize(
    "expression",
    [
        "factorial(Integer(1000000000))",
        "Pow(Integer(2), Integer(4097))",
        "Float('1e1000000', precision=53)",
    ],
)
def test_deserialize_rejects_unbounded_opaque_cost_arithmetic_promptly(
    expression: str,
) -> None:
    """Untrusted opaque costs cannot trigger unbounded eager arithmetic."""
    script = f"""
from qamomile.circuit.serialization import deserialize
from qamomile.circuit.serialization.encode import to_dict
from qamomile.circuit.serialization.graph_protobuf import qkernel_from_graph_dict
from tests.circuit.serialization.test_qkernel import _calls_fixed_cost_oracle

envelope = to_dict(_calls_fixed_cost_oracle)
definition = next(
    entry["definition"]
    for entry in envelope["callable_table"]
    if entry["definition"]["ref"]["name"]
    == "serialization_fixed_cost_oracle"
)
opaque_cost = dict(definition["opaque_cost"]["$map"])
gate_total = next(
    pair for pair in opaque_cost["gates"]["$map"] if pair[0] == "total"
)
gate_total[1] = {expression!r}
message = qkernel_from_graph_dict(envelope)
try:
    deserialize(message.SerializeToString(deterministic=True))
except ValueError:
    print("rejected")
else:
    raise AssertionError("unsafe opaque cost was accepted")
"""
    repository = Path(__file__).resolve().parents[3]

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert completed.stdout.strip() == "rejected"


def test_opaque_resource_callback_fails_serialization_explicitly() -> None:
    """Process-local opaque cost callbacks never disappear on serialization."""
    with pytest.raises(
        TypeError,
        match="opaque_cost callback objects cannot be serialized",
    ):
        serialize(_calls_callback_cost_oracle)


def test_scalar_oracle_explicit_signature_survives_added_control_round_trip() -> None:
    """The base ABI preserves target hints while excluding added controls."""
    restored = deserialize(serialize(_calls_controlled_explicit_signature_oracle))

    for kernel in (_calls_controlled_explicit_signature_oracle, restored):
        [invoke] = [
            operation
            for operation in kernel.block.operations
            if isinstance(operation, InvokeOperation)
        ]
        assert invoke.num_declared_control_qubits == 1
        assert invoke.num_added_control_qubits == 1
        assert invoke.definition is not None
        assert invoke.definition.signature is not None
        assert [hint.name for hint in invoke.definition.signature.operands] == [
            "control_0",
            "arg_0",
        ]
        assert [hint.name for hint in invoke.definition.signature.results] == [
            "control_0",
            "result_0",
        ]
    assert kernel_to_dict(restored) == kernel_to_dict(
        _calls_controlled_explicit_signature_oracle
    )


@pytest.mark.parametrize(
    "signature",
    [
        qmc.CallableSignature(
            inputs=[qmc.Qubit],
            outputs=[],
        ),
        qmc.CallableSignature(
            inputs=[qmc.Float],
            outputs=[qmc.Float],
        ),
    ],
)
def test_scalar_oracle_rejects_mismatched_explicit_signature(
    signature: qmc.CallableSignature,
) -> None:
    """A scalar Oracle does not replace an incompatible explicit signature."""
    oracle = qmc.Oracle(
        name="mismatched_explicit_signature_oracle",
        num_qubits=1,
        signature=signature,
    )

    @qmc.qkernel
    def circuit() -> qmc.Qubit:
        """Invoke the malformed explicit Oracle signature."""
        target = qmc.qubit("target")
        (target,) = oracle(target)
        return target

    with pytest.raises(ValueError, match="explicit CallableSignature"):
        circuit.build()


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
    """Restored QFT/IQFT preserve inverse mapping and engine emission."""
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
    assert implementation.engine == "test-engine"
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


def test_nested_container_return_round_trips() -> None:
    """Nested Dict wildcards remain compatible through deserialization."""
    restored = deserialize(serialize(_nested_container_return))

    assert restored.output_types == _nested_container_return.output_types
    assert restored.signature.return_annotation == _nested_container_return.return_type


def test_cached_return_annotation_is_stable_after_global_rebinding() -> None:
    """Serialization uses the return annotation resolved at decoration time."""
    global _MUTABLE_RETURN_ALIAS

    _ = _cached_return_alias.block
    original_alias = _MUTABLE_RETURN_ALIAS
    _MUTABLE_RETURN_ALIAS = qmc.UInt
    try:
        restored = deserialize(serialize(_cached_return_alias))
    finally:
        _MUTABLE_RETURN_ALIAS = original_alias

    assert restored.signature.return_annotation is qmc.Bit
    assert restored.output_types == [qmc.Bit]


def test_late_return_annotation_is_frozen_during_serialization() -> None:
    """Serialization resolves a late alias once and retains that contract."""
    global _LATE_SERIALIZED_RETURN_ALIAS

    first = deserialize(serialize(_late_serialized_return_alias))
    original_alias = _LATE_SERIALIZED_RETURN_ALIAS
    _LATE_SERIALIZED_RETURN_ALIAS = qmc.UInt
    try:
        second = deserialize(serialize(_late_serialized_return_alias))
    finally:
        _LATE_SERIALIZED_RETURN_ALIAS = original_alias

    assert first.signature.return_annotation is qmc.Bit
    assert first.output_types == [qmc.Bit]
    assert second.signature.return_annotation is qmc.Bit
    assert second.output_types == [qmc.Bit]


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


def test_singleton_python_tuple_round_trips_and_invokes_exactly() -> None:
    """A restored singleton Python tuple remains a tuple during invocation."""
    restored = deserialize(serialize(_singleton_tuple_return))

    @qmc.qkernel
    def caller() -> tuple[qmc.Bit]:
        """Invoke the restored singleton-tuple qkernel.

        Returns:
            tuple[qmc.Bit]: Restored qkernel's one-element result tuple.
        """
        return restored()

    block = caller.build()

    assert restored.output_types == [qmc.Bit]
    assert restored.signature.return_annotation == tuple[qmc.Bit]
    assert block.output_values[0].type.label() == "BitType"


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


def test_canonical_graph_preserves_indexed_carrier_suffixes() -> None:
    """Canonicalization remaps carrier roots without losing their indices."""
    envelope = {
        "value_table": [{"uuid": "root", "logical_id": "logical"}],
        "body": {
            "metadata": {
                "cast": {
                    "source_uuid": "root",
                    "source_logical_id": "logical",
                    "qubit_uuids": ["root_0", "root_2"],
                    "qubit_logical_ids": ["logical_0", "logical_2"],
                },
                "qfixed": {
                    "qubit_uuids": ["root_0", "root_2"],
                },
            },
            "qubit_mapping": ["root_0", "root_2"],
        },
    }

    canonical = canonicalize_graph(envelope)
    canonical_root = canonical["value_table"][0]["uuid"]
    canonical_logical = canonical["value_table"][0]["logical_id"]
    metadata = canonical["body"]["metadata"]

    assert metadata["cast"]["source_uuid"] == canonical_root
    assert metadata["cast"]["source_logical_id"] == canonical_logical
    assert metadata["cast"]["qubit_uuids"] == [
        f"{canonical_root}_0",
        f"{canonical_root}_2",
    ]
    assert metadata["cast"]["qubit_logical_ids"] == [
        f"{canonical_logical}_0",
        f"{canonical_logical}_2",
    ]
    assert metadata["qfixed"]["qubit_uuids"] == [
        f"{canonical_root}_0",
        f"{canonical_root}_2",
    ]
    assert canonical["body"]["qubit_mapping"] == [
        f"{canonical_root}_0",
        f"{canonical_root}_2",
    ]


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


def test_serialized_symbolic_inverse_refreshes_bound_scalar_width() -> None:
    """Value replacement refreshes inverse scalar-width metadata before emit."""
    restored = deserialize(serialize(_bound_symbolic_vector_inverse))
    specialized = restored.build(width=3)
    inverse_operation = next(
        operation
        for operation in specialized.operations
        if isinstance(operation, InverseBlockOperation)
    )

    assert inverse_operation.num_target_qubits == 3
    executable = QiskitTranspiler().transpile(restored, bindings={"width": 3})
    result = executable.sample(QiskitTranspiler().executor(), shots=16).result()
    assert result.results == [((1, 0, 0), 16)]


@pytest.mark.parametrize(
    "register_type",
    [
        QUIntType(width=3),
        QFixedType(integer_bits=1, fractional_bits=2),
    ],
)
def test_serialized_inverse_preserves_packed_register_width(
    register_type: QUIntType | QFixedType,
) -> None:
    """Static-binding validation retains packed-register scalar widths."""
    target = Value(type=register_type, name="target")
    operation = InverseBlockOperation(
        operands=[target],
        results=[target.next_version()],
        num_target_qubits=3,
        callable_attrs={
            "resource_contract": {
                "quantum_operand_widths": [
                    {"index": 0, "name": "target", "width": 3},
                ],
            },
        },
    )
    resolver = object.__new__(_StaticBindingResolver)

    resolver._validate_operation_call_widths(operation, {}, {})

    assert operation.num_target_qubits == 3


def test_serialized_controlled_static_inverse_refreshes_call_site_width() -> None:
    """Call-site widths refresh nested inverse metadata before outer control."""
    encoding = qmc.LCUBlockEncoding(
        unitary=_asymmetric_block_encoding_unitary,
        normalization=1.0,
        num_signal_qubits=2,
        num_system_qubits=1,
    )
    restored = deserialize(serialize(_controlled_static_inverse))

    executable = QiskitTranspiler().transpile(
        restored,
        bindings={"encoding": encoding},
    )
    result = executable.sample(QiskitTranspiler().executor(), shots=16).result()
    assert result.results == [((1,), 16)]


@pytest.mark.parametrize(("signal_width", "system_width"), [(2, 1), (2, 3)])
def test_serialized_controlled_inverse_materializes_owned_block_widths(
    signal_width: int,
    system_width: int,
) -> None:
    """Bound call edges materialize inverse widths without static bindings."""
    restored = deserialize(serialize(_controlled_bound_symbolic_pair_inverse))

    executable = QiskitTranspiler().transpile(
        restored,
        bindings={
            "signal_width": signal_width,
            "system_width": system_width,
        },
    )
    result = executable.sample(QiskitTranspiler().executor(), shots=16).result()
    expected = (1, *(0 for _ in range(system_width - 1)))
    assert result.results == [(expected, 16)]


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
