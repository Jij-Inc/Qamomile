"""Measurement operations for quantum circuits."""

from qamomile.circuit.frontend.handle import Bit, Qubit
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.gate import MeasureOperation as IRMeasureOperation
from qamomile.circuit.ir.types import BitType
from qamomile.circuit.ir.value import Value


def measure(qubit: Qubit) -> Bit:
    """Measure a qubit in the computational basis.

    Performs a projective measurement on the qubit in the Z-basis.
    The qubit is consumed by this operation and cannot be used afterwards.

    Args:
        qubit: The qubit to measure. Will be consumed.

    Returns:
        Bit: The classical bit containing the measurement result.
             The value is determined at runtime when the circuit is executed.

    Example:
        ```python
        @qkernel
        def measure_qubit(q: Qubit) -> Bit:
            q = h(q)
            return measure(q)
        ```
    """
    # Create output bit value
    bit_out_value = Value(type=BitType(), name=f"{qubit.value.name}_measured")

    bit_out = Bit(value=bit_out_value)

    # Create IR MeasureOperation
    measure_op = IRMeasureOperation(operands=[qubit.value], results=[bit_out_value])

    tracer = get_current_tracer()
    tracer.add_operation(measure_op)

    return bit_out
