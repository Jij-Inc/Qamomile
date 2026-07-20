"""Measurement operations for quantum circuits."""

from typing import Union, overload

from qamomile.circuit.frontend.handle import Bit, Float, QFixed, Qubit, Vector
from qamomile.circuit.frontend.handle.array import Vector as VectorClass
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.gate import (
    MeasureOperation as IRMeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
    ProjectOperation,
    ResetOperation,
)
from qamomile.circuit.ir.types import BitType, FloatType
from qamomile.circuit.ir.types.primitives import UIntType
from qamomile.circuit.ir.value import ArrayValue, Value


@overload
def measure(target: Qubit) -> Bit: ...


@overload
def measure(target: QFixed) -> Float: ...


@overload
def measure(target: Vector[Qubit]) -> Vector[Bit]: ...


def measure(
    target: Union[Qubit, QFixed, Vector[Qubit]],
) -> Union[Bit, Float, Vector[Bit]]:
    """Measure a qubit or QFixed in the computational basis.

    Performs a projective measurement in the Z-basis.
    The quantum resource is consumed by this operation and cannot be used afterwards.

    Args:
        target (Qubit | QFixed | Vector[Qubit]): Quantum resource to measure.
            - Qubit: Returns a classical Bit
            - QFixed: Returns a Float (decoded from measured bits)

    Returns:
        Bit | Float | Vector[Bit]: Classical result matching the input shape.

    Raises:
        TypeError: If ``target`` is not a supported quantum handle.
        QubitConsumedError: If the quantum resource was already consumed.
        RuntimeError: If no tracer is active.

    Example:
        ```python
        @qkernel
        def measure_qubit(q: Qubit) -> Bit:
            q = h(q)
            return measure(q)

        @qkernel
        def measure_qfixed(qf: QFixed) -> Float:
            # After QPE, qf holds phase bits
            return measure(qf)
        ```
    """
    if isinstance(target, QFixed):
        return _measure_qfixed(target)
    elif isinstance(target, VectorClass) and target.element_type == Qubit:
        return _measure_vector_qubit(target)
    elif isinstance(target, Qubit):
        return _measure_qubit(target)
    else:
        raise TypeError(
            f"Unsupported type for measurement: {type(target)}. "
            "Expected Qubit, QFixed, or Vector[Qubit]."
        )


def _measure_qubit(qubit: Qubit) -> Bit:
    """Measure a single qubit.

    Args:
        qubit (Qubit): Qubit to measure destructively.

    Returns:
        Bit: Measurement result.

    Raises:
        QubitConsumedError: If ``qubit`` was already consumed.
        RuntimeError: If no tracer is active.
    """
    tracer = get_current_tracer()
    qubit.validate_consumable("measure")
    bit_out_value = Value(type=BitType(), name=f"{qubit.value.name}_measured")
    bit_out = Bit(value=bit_out_value)
    measure_op = IRMeasureOperation(operands=[qubit.value], results=[bit_out_value])
    qubit.consume(operation_name="measure")
    tracer.add_operation(measure_op)
    return bit_out


def project_z(qubit: Qubit) -> tuple[Qubit, Bit]:
    """Project a qubit in the Z basis and keep the projected state.

    Args:
        qubit (Qubit): Qubit to project. The input handle is consumed.

    Returns:
        tuple[Qubit, Bit]: Projected qubit handle and measurement bit.

    Raises:
        QubitConsumedError: If ``qubit`` was already consumed.
        RuntimeError: If no tracer is active.
    """
    tracer = get_current_tracer()
    qubit.validate_consumable("project_z")
    qubit_out_value = qubit.value.next_version()
    bit_out_value = Value(type=BitType(), name=f"{qubit.value.name}_projected")
    qubit_out = Qubit(
        value=qubit_out_value,
        parent=qubit.parent,
        indices=qubit.indices,
    )
    bit_out = Bit(value=bit_out_value)

    project_op = ProjectOperation(
        operands=[qubit.value],
        results=[qubit_out_value, bit_out_value],
        axis="z",
    )
    qubit = qubit.consume(operation_name="project_z")
    tracer.add_operation(project_op)
    qubit._handoff_direct_borrow_to(qubit_out)
    return qubit_out, bit_out


def project_x(qubit: Qubit) -> tuple[Qubit, Bit]:
    """Project a qubit in the X basis and keep the projected state.

    Args:
        qubit (Qubit): Qubit to project. The input handle is consumed.

    Returns:
        tuple[Qubit, Bit]: Projected qubit handle and measurement bit.

    Raises:
        QubitConsumedError: If ``qubit`` was already consumed.
        RuntimeError: If no tracer is active.
    """
    from qamomile.circuit.frontend.operation.qubit_gates import h

    qubit = h(qubit)
    qubit, bit = project_z(qubit)
    qubit = h(qubit)
    return qubit, bit


def project_y(qubit: Qubit) -> tuple[Qubit, Bit]:
    """Project a qubit in the Y basis and keep the projected state.

    Args:
        qubit (Qubit): Qubit to project. The input handle is consumed.

    Returns:
        tuple[Qubit, Bit]: Projected qubit handle and measurement bit.

    Raises:
        QubitConsumedError: If ``qubit`` was already consumed.
        RuntimeError: If no tracer is active.
    """
    from qamomile.circuit.frontend.operation.qubit_gates import h, s, sdg

    qubit = sdg(qubit)
    qubit = h(qubit)
    qubit, bit = project_z(qubit)
    qubit = h(qubit)
    qubit = s(qubit)
    return qubit, bit


def reset(qubit: Qubit) -> Qubit:
    """Reset a qubit to the |0> state.

    Args:
        qubit (Qubit): Qubit to reset. The input handle is consumed.

    Returns:
        Qubit: Fresh handle for the reset qubit.

    Raises:
        QubitConsumedError: If ``qubit`` was already consumed.
        RuntimeError: If no tracer is active.
    """
    tracer = get_current_tracer()
    qubit.validate_consumable("reset")
    qubit_out_value = qubit.value.next_version()
    qubit_out = Qubit(
        value=qubit_out_value,
        parent=qubit.parent,
        indices=qubit.indices,
    )
    reset_op = ResetOperation(operands=[qubit.value], results=[qubit_out_value])
    qubit = qubit.consume(operation_name="reset")
    tracer.add_operation(reset_op)
    qubit._handoff_direct_borrow_to(qubit_out)
    return qubit_out


def measure_reset(qubit: Qubit) -> tuple[Qubit, Bit]:
    """Measure a qubit in the Z basis and reset it to |0>.

    Args:
        qubit (Qubit): Qubit to measure and reset.

    Returns:
        tuple[Qubit, Bit]: Reset qubit handle and measurement bit.

    Raises:
        QubitConsumedError: If ``qubit`` was already consumed.
        RuntimeError: If no tracer is active.
    """
    qubit, bit = project_z(qubit)
    qubit = reset(qubit)
    return qubit, bit


def _measure_qfixed(qfixed: QFixed) -> Float:
    """Measure a QFixed (quantum fixed-point number) and decode to Float.

    The QFixed wraps multiple qubits representing a fixed-point number.
    This operation measures all qubits and decodes the bitstring to a float.

    For QPE phase (int_bits=0):
        Bits are ordered least-significant first. For ``n`` bits,
        ``float_value = b0*2**(-n) + ... + b[n-1]*2**(-1)``.

    Args:
        qfixed (QFixed): Fixed-point quantum register to measure.

    Returns:
        Float: Decoded measurement result.

    Raises:
        QubitConsumedError: If ``qfixed`` was already consumed.
        RuntimeError: If no tracer is active.
    """
    tracer = get_current_tracer()
    qfixed.validate_consumable("measure")
    float_out_value = Value(type=FloatType(), name="qfixed_measured")
    float_out = Float(value=float_out_value)
    qubit_values = qfixed.value.get_qfixed_qubit_uuids()
    num_bits = qfixed.value.get_qfixed_num_bits() or len(qubit_values)
    int_bits = qfixed.value.get_qfixed_int_bits() or 0

    # Create MeasureQFixedOperation
    measure_op = MeasureQFixedOperation(
        operands=[qfixed.value],
        results=[float_out_value],
        num_bits=num_bits,
        int_bits=int_bits,
    )
    qfixed.consume(operation_name="measure")
    tracer.add_operation(measure_op)
    return float_out


def _measure_vector_qubit(qubits: Vector[Qubit]) -> Vector[Bit]:
    """Measure a vector of qubits.

    Args:
        qubits (Vector[Qubit]): Qubit register to measure destructively.

    Returns:
        Vector[Bit]: Measurement bits with the same shape.

    Raises:
        QubitConsumedError: If the register or a covered slot was consumed.
        UnreturnedBorrowError: If a register borrow remains outstanding.
        RuntimeError: If no tracer is active.
    """
    tracer = get_current_tracer()
    qubits.validate_consumable("measure")
    if isinstance(qubits.value, ArrayValue) and qubits.value.shape:
        shape_values = qubits.value.shape
    else:
        # Convert frontend shape to IR shape values
        shape_values = tuple(
            Value(type=UIntType(), name=f"dim_{i}").with_const(dim)
            if isinstance(dim, int)
            else dim.value
            for i, dim in enumerate(qubits.shape)
        )

    # Create output ArrayValue with same shape
    bits_value = ArrayValue(
        type=BitType(),
        name=f"{qubits.value.name}_measured",
        shape=shape_values,
    )

    # Create MeasureVectorOperation
    measure_op = MeasureVectorOperation(
        operands=[qubits.value],
        results=[bits_value],
    )
    qubits = qubits.consume(operation_name="measure")
    tracer.add_operation(measure_op)

    # Create and return Vector[Bit] using _create_from_value
    return VectorClass[Bit]._create_from_value(
        value=bits_value,
        shape=qubits.shape,
    )
