"""Cast operation for type conversions over the same quantum resources."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, overload

from qamomile.circuit.frontend.handle.primitives import QFixed, Qubit
from qamomile.circuit.frontend.handle.array import Vector
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.types.q_register import QFixedType
from qamomile.circuit.ir.value import Value

if TYPE_CHECKING:
    pass


T = TypeVar("T")


@overload
def cast(
    source: Vector[Qubit],
    target_type: type[QFixed],
    *,
    int_bits: int = 0,
) -> QFixed: ...


def cast(
    source: Vector[Qubit],
    target_type: type,
    *,
    int_bits: int = 0,
) -> QFixed:
    """Cast a quantum value to a different type without allocating new qubits.

    The cast creates an alias over the same quantum resources. Both the
    source and target refer to the same physical qubits.

    Args:
        source: The value to cast (currently supports Vector[Qubit])
        target_type: The target type class (currently supports QFixed)
        int_bits: For QFixed, number of integer bits (default: 0 = all fractional)

    Returns:
        A new handle of the target type referencing the same qubits.

    Example:
        @qmc.qkernel
        def my_circuit():
            phase_register = qmc.qubit_array(5, name="phase")
            # ... apply some operations ...

            # Cast the qubit array to QFixed for measurement
            phase_qfixed = qmc.cast(phase_register, qmc.QFixed, int_bits=0)
            phase_value = qmc.measure(phase_qfixed)
            return phase_value

    Raises:
        TypeError: If the source type or target type is not supported
        ValueError: If int_bits is negative or larger than the number of qubits
    """
    # Validate source type
    if not isinstance(source, Vector):
        raise TypeError(f"cast source must be a Vector, got {type(source).__name__}")

    if source.element_type != Qubit:
        raise TypeError(
            f"cast source must be Vector[Qubit], got Vector[{source.element_type.__name__}]"
        )

    # Dispatch based on target type
    if target_type == QFixed:
        return _cast_vector_qubit_to_qfixed(source, int_bits)
    else:
        raise TypeError(
            f"Unsupported target type for cast: {target_type}. Supported types: QFixed"
        )


def _cast_vector_qubit_to_qfixed(
    source: Vector[Qubit],
    int_bits: int = 0,
) -> QFixed:
    """Cast Vector[Qubit] to QFixed.

    Args:
        source: The qubit array to cast
        int_bits: Number of integer bits (rest are fractional)

    Returns:
        QFixed handle referencing the same qubits
    """
    # Get the number of qubits
    size = source.shape[0]
    if isinstance(size, int):
        num_qubits = size
    elif hasattr(size, "value") and size.value.is_constant():
        num_qubits = int(size.value.get_const())
    elif hasattr(size, "init_value"):
        num_qubits = int(size.init_value)
    else:
        raise ValueError(
            "cast requires a fixed-size Vector. Dynamic sizes are not supported."
        )

    # Validate int_bits
    if int_bits < 0:
        raise ValueError(f"int_bits must be non-negative, got {int_bits}")
    if int_bits > num_qubits:
        raise ValueError(
            f"int_bits ({int_bits}) cannot exceed number of qubits ({num_qubits})"
        )

    frac_bits = num_qubits - int_bits

    # Collect qubit UUIDs and logical_ids from the source array
    qubit_uuids: list[str] = []
    qubit_logical_ids: list[str] = []
    for i in range(num_qubits):
        element = source[i]
        qubit_uuids.append(element.value.uuid)
        qubit_logical_ids.append(element.value.logical_id)

    # Create the result QFixed value
    result_type = QFixedType(integer_bits=int_bits, fractional_bits=frac_bits)
    result_value = Value(
        type=result_type,
        name=f"{source.value.name}_as_qfixed",
        params={
            # Cast metadata - using logical_id for physical qubit tracking
            "cast_source_uuid": source.value.uuid,
            "cast_source_logical_id": source.value.logical_id,
            "cast_qubit_uuids": qubit_uuids,
            "cast_qubit_logical_ids": qubit_logical_ids,
            # QFixed-specific metadata (for backward compatibility with measurement)
            "num_bits": num_qubits,
            "int_bits": int_bits,
            "qubit_values": qubit_uuids,
        },
    )

    # Create and emit the CastOperation
    cast_op = CastOperation(
        operands=[source.value],
        results=[result_value],
        source_type=source.value.type,
        target_type=result_type,
        qubit_mapping=qubit_uuids,
    )

    tracer = get_current_tracer()
    tracer.add_operation(cast_op)

    return QFixed(value=result_value)
