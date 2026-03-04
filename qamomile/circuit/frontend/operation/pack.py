"""Pack individual qubits into a Vector[Qubit].

This module provides the ``pack_qubits`` function which collects one or more
``Qubit`` handles (or fixed-size ``Vector[Qubit]`` handles) into a single
``Vector[Qubit]``.  The resulting vector carries the canonical
``element_uuids`` key that downstream passes (separate, emit) use to resolve
the packed qubit ordering.

Linear-type contract:
    All input handles are *consumed* by ``pack_qubits``.  After the call the
    original handles must not be reused; only the returned ``Vector[Qubit]``
    is valid.
"""

from __future__ import annotations

from qamomile.circuit.frontend.handle.primitives import Qubit
from qamomile.circuit.frontend.handle.array import ArrayBase, Vector
from qamomile.circuit.ir.types.primitives import QubitType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value


def _resolve_vector_size(vec: ArrayBase) -> int | None:
    """Try to resolve the concrete size of a 1-D array handle.

    Returns the integer size if it can be determined, or ``None`` for
    symbolic (dynamic) sizes.
    """
    size = vec.shape[0]
    if isinstance(size, int):
        return size
    if hasattr(size, "value") and size.value.is_constant():
        return int(size.value.get_const())
    # Fall back to IR-level shape (ArrayValue.shape)
    if vec.value.shape:
        dim_val = vec.value.shape[0]
        if dim_val.is_constant():
            return int(dim_val.get_const())
    return None


def pack_qubits(*items: Qubit | Vector[Qubit]) -> Vector[Qubit]:
    """Pack individual qubits and/or fixed-size qubit vectors into one vector.

    Args:
        *items: One or more ``Qubit`` or fixed-size ``Vector[Qubit]`` handles.

    Returns:
        A new ``Vector[Qubit]`` whose ``element_uuids`` param records the
        UUID ordering of the packed qubits.

    Raises:
        ValueError: If no arguments are given, or a ``Vector[Qubit]`` with a
            dynamic (symbolic) size is passed.
        TypeError: If an argument is neither ``Qubit`` nor ``Vector[Qubit]``.
    """
    if len(items) == 0:
        raise ValueError("pack_qubits requires at least one qubit")

    element_uuids: list[str] = []
    element_logical_ids: list[str] = []

    for item in items:
        if isinstance(item, Qubit):
            # Consume the qubit handle (linear type enforcement)
            item.consume("pack_qubits")
            element_uuids.append(item.value.uuid)
            element_logical_ids.append(item.value.logical_id)

        elif isinstance(item, ArrayBase) and item.element_type is Qubit:
            # Fixed-size Vector[Qubit] only
            n = _resolve_vector_size(item)
            if n is None:
                raise ValueError("pack_qubits requires fixed-size Vector[Qubit]")

            # Consume the vector handle
            item.consume("pack_qubits")

            # Collect element UUIDs
            for i in range(n):
                elem = item[i]
                element_uuids.append(elem.value.uuid)
                element_logical_ids.append(elem.value.logical_id)
                item[i] = elem  # return the borrow so validate_all_returned won't fire

        else:
            raise TypeError("pack_qubits accepts only Qubit or Vector[Qubit]")

    # Build the packed ArrayValue
    total = len(element_uuids)
    size_value = Value(
        type=UIntType(),
        name="pack_size",
        params={"const": total},
    )
    packed_value = ArrayValue(
        type=QubitType(),
        name="packed_qubits",
        shape=(size_value,),
        params={
            "element_uuids": element_uuids,
            "element_logical_ids": element_logical_ids,
        },
    )

    return Vector._create_from_value(
        value=packed_value,
        shape=(total,),
        name="packed_qubits",
    )
