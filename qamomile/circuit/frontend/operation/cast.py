"""Cast operation for type conversions over the same quantum resources."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, cast as type_cast, overload

from qamomile.circuit.frontend.handle.array import Vector, VectorView
from qamomile.circuit.frontend.handle.primitives import QFixed, QInt, Qubit, UInt
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.types.primitives import UIntType
from qamomile.circuit.ir.types.q_register import QFixedType, QUIntType
from qamomile.circuit.ir.value import Value, composite_carrier_key

if TYPE_CHECKING:
    pass


T = TypeVar("T")
_INT_BITS_UNSET = object()


@overload
def cast(
    source: Vector[Qubit],
    target_type: type[QFixed],
    *,
    int_bits: int = 0,
) -> QFixed: ...


@overload
def cast(
    source: Vector[Qubit],
    target_type: type[QInt],
) -> QInt: ...


def cast(
    source: Vector[Qubit],
    target_type: type,
    *,
    int_bits: int | object = _INT_BITS_UNSET,
) -> QFixed | QInt:
    """Cast a qubit vector to a packed quantum-number handle.

    The cast performs a move: the source handle is consumed and cannot be
    reused after the cast. The returned handle references the same physical
    qubits. ``QInt`` uses every carrier as an unsigned integer bit, while
    ``QFixed`` divides the carriers into integer and fractional bits.

    Args:
        source (Vector[Qubit]): Qubit vector to reinterpret and consume.
        target_type (type): Target handle class, either ``QInt`` or ``QFixed``.
        int_bits (int): Number of integer bits for ``QFixed``. Defaults to 0
            when omitted for ``QFixed``. Must be omitted for ``QInt``.

    Returns:
        QInt | QFixed: Target handle referencing the source qubits in the same
            order.

    Raises:
        TypeError: If ``source`` is not ``Vector[Qubit]``, ``target_type`` is
            unsupported, or ``int_bits`` is explicitly supplied for ``QInt``.
        ValueError: If ``int_bits`` is invalid for the selected target or the
            source has an unsupported symbolic slice layout.
        QubitConsumedError: If ``source`` has already been consumed.
        UnreturnedBorrowError: If an element borrowed from ``source`` has not
            been returned.
        RuntimeError: If no tracer is active.

    Example:
        ```python
        @qmc.qkernel
        def read_integer() -> qmc.UInt:
            register = qmc.qubit_array(3, name="value")
            register[0] = qmc.x(register[0])
            return qmc.measure(qmc.cast(register, qmc.QInt))
        ```
    """
    # Validate source type
    if not isinstance(source, Vector):
        raise TypeError(f"cast source must be a Vector, got {type(source).__name__}")

    if source.element_type != Qubit:
        raise TypeError(
            f"cast source must be Vector[Qubit], got Vector[{source.element_type.__name__}]"
        )

    # Dispatch based on target type
    if target_type is QFixed:
        fixed_int_bits = 0 if int_bits is _INT_BITS_UNSET else type_cast(int, int_bits)
        return _cast_vector_qubit_to_qfixed(source, fixed_int_bits)
    if target_type is QInt:
        if int_bits is not _INT_BITS_UNSET:
            raise TypeError(
                "int_bits is only supported when casting to QFixed; omit it for QInt."
            )
        return _cast_vector_qubit_to_qint(source)
    raise TypeError(
        f"Unsupported target type for cast: {target_type}. "
        "Supported types: QInt, QFixed"
    )


def _resolve_vector_qubit_cast_layout(
    source: Vector[Qubit],
) -> tuple[int | None, Value | None, list[str], list[str]]:
    """Resolve the width and ordered carriers of a packed-register cast.

    Serves both packed register targets (``QInt`` and ``QFixed``); the
    carrier keys are spelled by :func:`composite_carrier_key`.

    Args:
        source (Vector[Qubit]): Source vector or literal-bounded vector view.

    Returns:
        tuple[int | None, Value | None, list[str], list[str]]: Concrete width
            when known, symbolic width value when present, ordered carrier
            UUID keys, and parallel logical-ID keys. Carrier position zero is
            the least-significant bit for both ``QInt`` and ``QFixed``.

    Raises:
        UnreturnedBorrowError: If an element borrow has not been returned.
        ValueError: If the vector width or a slice view cannot be resolved
            safely enough to identify physical carrier qubits.
    """
    source.validate_all_returned()

    source_any: Any = source
    if isinstance(source_any, VectorView) and source_any._slice_covered_indices is None:
        raise ValueError(
            f"cast() on a view with symbolic slice bounds is not supported "
            f"(view of '{source_any._slice_parent.value.name}'). Use "
            f"literal-bounded slicing for cast operands."
        )

    size = source.shape[0]
    size_value: Value | None = None
    if isinstance(size, int):
        num_qubits = size
    elif hasattr(size, "value") and size.value.is_constant():
        num_qubits = int(size.value.get_const())
        size_value = size.value
    elif hasattr(size, "value"):
        num_qubits = None
        size_value = size.value
    elif hasattr(size, "init_value"):
        num_qubits = int(size.init_value)
    else:
        raise ValueError(
            "cast requires a fixed-size Vector. Dynamic sizes are not supported."
        )

    if num_qubits is None:
        root_av = source.value
        root_indices: tuple[int, ...] = ()
    elif isinstance(source, VectorView):  # type: ignore[unreachable]
        covered = source._slice_covered_indices  # type: ignore[unreachable]
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

    # ``root_indices`` already lives in the root array's index space (the
    # VectorView supplies covered root slots), so this site only spells the
    # keys instead of re-folding the view chain through ``root_carrier_keys``.
    qubit_uuids = [composite_carrier_key(root_av.uuid, index) for index in root_indices]
    qubit_logical_ids = [
        composite_carrier_key(root_av.logical_id, index) for index in root_indices
    ]
    return num_qubits, size_value, qubit_uuids, qubit_logical_ids


def _cast_vector_qubit_to_qfixed(
    source: Vector[Qubit],
    int_bits: int = 0,
) -> QFixed:
    """Cast Vector[Qubit] to QFixed (move semantics).

    Args:
        source (Vector[Qubit]): Qubit vector to cast and consume.
        int_bits (int): Number of integer bits; remaining bits are fractional.

    Returns:
        QFixed: Fixed-point handle referencing the same ordered carriers.

    Raises:
        UnreturnedBorrowError: If an element borrow has not been returned.
        ValueError: If the integer-bit count, vector width, or source view is
            invalid.
        QubitConsumedError: If ``source`` has already been consumed.
        RuntimeError: If no tracer is active.
    """
    num_qubits, size_value, qubit_uuids, qubit_logical_ids = (
        _resolve_vector_qubit_cast_layout(source)
    )

    # Validate int_bits
    if int_bits < 0:
        raise ValueError(f"int_bits must be non-negative, got {int_bits}")
    if num_qubits is not None and int_bits > num_qubits:
        raise ValueError(
            f"int_bits ({int_bits}) cannot exceed number of qubits ({num_qubits})"
        )
    if num_qubits is not None:
        frac_bits: int | Value[UIntType] = num_qubits - int_bits
    else:
        assert size_value is not None
        symbolic_width = UInt(value=type_cast(Value[UIntType], size_value))
        frac_bits = (
            (symbolic_width - int_bits).value if int_bits else symbolic_width.value
        )

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
            num_bits=num_qubits or 0,
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


def _cast_vector_qubit_to_qint(source: Vector[Qubit]) -> QInt:
    """Cast a qubit vector to an unsigned QInt with move semantics.

    Args:
        source (Vector[Qubit]): Qubit vector to cast and consume.

    Returns:
        QInt: Unsigned integer handle whose bit zero is the source's first
            carrier qubit.

    Raises:
        UnreturnedBorrowError: If an element borrow has not been returned.
        ValueError: If the vector width or source view cannot be resolved.
        QubitConsumedError: If ``source`` has already been consumed.
        RuntimeError: If no tracer is active.
    """
    num_qubits, size_value, qubit_uuids, qubit_logical_ids = (
        _resolve_vector_qubit_cast_layout(source)
    )
    if num_qubits is None:
        assert size_value is not None
        width: int | Value[UIntType] = type_cast(Value[UIntType], size_value)
    else:
        width = num_qubits

    source = source.consume(operation_name="cast")
    result_type = QUIntType(width=width)
    result_value = Value(
        type=result_type,
        name=f"{source.value.name}_as_qint",
    ).with_cast_metadata(
        source_uuid=source.value.uuid,
        source_logical_id=source.value.logical_id,
        qubit_uuids=qubit_uuids,
        qubit_logical_ids=qubit_logical_ids,
    )
    get_current_tracer().add_operation(
        CastOperation(
            operands=[source.value],
            results=[result_value],
            source_type=source.value.type,
            target_type=result_type,
            qubit_mapping=qubit_uuids,
        )
    )
    return QInt(value=result_value)
