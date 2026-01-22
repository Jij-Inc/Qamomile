"""Measurement operations for quantum circuits."""

from typing import Union, overload

from qamomile.circuit.frontend.handle import Bit, Float, QFixed, Qubit, Vector
from qamomile.circuit.frontend.handle.array import Vector as VectorClass
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.gate import (
    MeasureOperation as IRMeasureOperation,
    MeasureVectorOperation,
    MeasureQFixedOperation,
)
from qamomile.circuit.ir.types import BitType, FloatType
from qamomile.circuit.ir.types.primitives import UIntType
from qamomile.circuit.ir.value import Value, ArrayValue


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
        target: The quantum resource to measure.
            - Qubit: Returns a classical Bit
            - QFixed: Returns a Float (decoded from measured bits)

    Returns:
        Bit for Qubit input, Float for QFixed input.

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
    elif hasattr(target, "element_type") and target.element_type == Qubit:
        return _measure_vector_qubit(target)
    else:
        return _measure_qubit(target)


def _measure_qubit(qubit: Qubit) -> Bit:
    """Measure a single qubit.

    Args:
        qubit: The qubit to measure.

    Returns:
        Bit containing the measurement result.
    """
    # Consume the input handle (enforces linear type - measurement is destructive)
    qubit = qubit.consume(operation_name="measure")

    # Create output bit value
    bit_out_value = Value(type=BitType(), name=f"{qubit.value.name}_measured")
    bit_out = Bit(value=bit_out_value)

    # Create IR MeasureOperation
    measure_op = IRMeasureOperation(operands=[qubit.value], results=[bit_out_value])

    tracer = get_current_tracer()
    tracer.add_operation(measure_op)

    return bit_out


def _measure_qfixed(qfixed: QFixed) -> Float:
    """Measure a QFixed (quantum fixed-point number) and decode to Float.

    The QFixed wraps multiple qubits representing a fixed-point number.
    This operation measures all qubits and decodes the bitstring to a float.

    For QPE phase (int_bits=0):
        float_value = 0.b0b1b2... = b0*0.5 + b1*0.25 + b2*0.125 + ...

    Args:
        qfixed: The QFixed to measure.

    Returns:
        Float containing the decoded measurement result.
    """
    # Consume the input handle (enforces linear type - measurement is destructive)
    qfixed = qfixed.consume(operation_name="measure")

    # Create Float output value
    float_out_value = Value(type=FloatType(), name="qfixed_measured")
    float_out = Float(value=float_out_value)

    # Extract QFixed parameters
    qubit_values = qfixed.value.params.get("qubit_values", [])
    num_bits = qfixed.value.params.get("num_bits", len(qubit_values))
    int_bits = qfixed.value.params.get("int_bits", 0)

    # Create MeasureQFixedOperation
    measure_op = MeasureQFixedOperation(
        operands=[qfixed.value],
        results=[float_out_value],
        num_bits=num_bits,
        int_bits=int_bits,
    )

    tracer = get_current_tracer()
    tracer.add_operation(measure_op)

    return float_out


def _measure_vector_qubit(qubits: Vector[Qubit]) -> Vector[Bit]:
    """Measure a vector of qubits.

    Args:
        qubits: The Vector[Qubit] to measure.

    Returns:
        Vector[Bit] containing the measurement results.
    """
    # Consume the input handle (enforces linear type - measurement is destructive)
    qubits = qubits.consume(operation_name="measure")

    # Get shape values - prefer IR shape from ArrayValue, fallback to frontend shape
    if isinstance(qubits.value, ArrayValue) and qubits.value.shape:
        shape_values = qubits.value.shape
    else:
        # Convert frontend shape to IR shape values
        shape_values = tuple(
            Value(type=UIntType(), name=f"dim_{i}", params={"const": dim})
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

    tracer = get_current_tracer()
    tracer.add_operation(measure_op)

    # Create and return Vector[Bit] using _create_from_value
    return VectorClass[Bit]._create_from_value(
        value=bits_value,
        shape=qubits.shape,
    )
