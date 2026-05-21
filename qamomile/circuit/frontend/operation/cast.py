"""Cast operation for type conversions over the same quantum resources."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from qamomile.circuit.frontend.handle.array import Vector, VectorView
from qamomile.circuit.frontend.handle.primitives import QFixed, Qubit
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.types.q_register import QFixedType
from qamomile.circuit.ir.value import Value

if TYPE_CHECKING:
    pass


T = TypeVar("T")


def cast(
    source: Vector[Qubit],
    target_type: type,
    *,
    int_bits: int = 0,
) -> QFixed:
    """Cast a quantum value to a different type without allocating new qubits.

    The cast performs a move: the source handle is consumed and cannot be
    reused after the cast. The returned handle references the same physical
    qubits.

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
    """Cast Vector[Qubit] to QFixed (move semantics).

    Args:
        source: The qubit array to cast. Consumed after the cast.
        int_bits: Number of integer bits (rest are fractional)

    Returns:
        QFixed handle referencing the same qubits
    """
    # Ensure all borrowed elements have been returned before casting
    source.validate_all_returned()

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

    # Compose carrier keys in **root-space**.  When ``source`` is a
    # ``VectorView`` over ``q[start::step]`` (possibly nested), the
    # physical qubit slots backing the cast are root-space indices
    # ``root[i_0], root[i_1], ..., root[i_{N-1}]`` — not
    # ``view[0..N-1]``.  The allocator registers qubits under the root
    # array's uuid; using the view's own uuid here would miss in
    # ``qubit_map`` and the cast would emit no measurements.
    #
    # ``VectorView._slice_covered_indices`` already enumerates the
    # root-space indices for compile-time-known slices (the same data
    # the borrow tracker uses), so we use it directly: no IR chain walk,
    # no symbolic-bound resolution.  When it is ``None`` the slice has
    # symbolic bounds and we cannot enumerate carriers at trace time;
    # cast on such a view is rejected here rather than producing a
    # silently broken QFixed downstream.
    if isinstance(source, VectorView):
        covered = source._slice_covered_indices
        if covered is None:
            raise ValueError(
                f"cast() on a view with symbolic slice bounds is not "
                f"supported (view of '{source._slice_parent.value.name}'). "
                f"Use literal-bounded slicing for cast operands."
            )
        if len(covered) != num_qubits:
            raise ValueError(
                f"cast() internal error: view length {num_qubits} does not "
                f"match covered-index count {len(covered)}."
            )
        root_av = source._slice_parent.value
        root_indices = covered
    else:
        root_av = source.value
        root_indices = tuple(range(num_qubits))

    qubit_uuids: list[str] = []
    qubit_logical_ids: list[str] = []
    for root_idx in root_indices:
        qubit_uuids.append(f"{root_av.uuid}_{root_idx}")
        qubit_logical_ids.append(f"{root_av.logical_id}_{root_idx}")

    # Consume the source (move semantics - prevents reuse)
    source = source.consume(operation_name="cast")

    # Create the result QFixed value
    result_type = QFixedType(integer_bits=int_bits, fractional_bits=frac_bits)
    result_value = (
        Value(
            type=result_type,
            name=f"{source.value.name}_as_qfixed",
        )
        .with_cast_metadata(
            source_uuid=source.value.uuid,
            source_logical_id=source.value.logical_id,
            qubit_uuids=qubit_uuids,
            qubit_logical_ids=qubit_logical_ids,
        )
        .with_qfixed_metadata(
            qubit_uuids=qubit_uuids,
            num_bits=num_qubits,
            int_bits=int_bits,
        )
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
