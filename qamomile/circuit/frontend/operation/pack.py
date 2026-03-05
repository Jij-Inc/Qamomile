"""Pack and unpack qubits into/from a Vector[Qubit].

This module provides ``pack_qubits`` and ``unpack_qubits`` for combining and
splitting qubit handles.  Both functions enforce a linear-type contract:
input handles are *consumed* and must not be reused after the call.

The resulting vectors carry the canonical ``element_uuids`` key that
downstream passes (separate, emit) use to resolve qubit ordering.
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
    # Check frontend-level shape first (if non-empty)
    if vec.shape:
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

            # Collect element UUIDs FIRST (before consume, so _get_element works)
            for i in range(n):
                elem = item[i]
                element_uuids.append(elem.value.uuid)
                element_logical_ids.append(elem.value.logical_id)
                item[i] = elem  # return the borrow so validate_all_returned won't fire

            # Consume the vector handle AFTER collection
            item.consume("pack_qubits")

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


def unpack_qubits(
    packed: Vector[Qubit],
    *,
    num_unpacked: int,
    num_elements: list[int] | None = None,
    indices: list[list[int]] | None = None,
) -> tuple[Vector[Qubit], ...]:
    """Split a packed qubit vector into multiple sub-vectors.

    Exactly one of ``num_elements`` or ``indices`` must be provided.

    Args:
        packed: A ``Vector[Qubit]`` to split (typically from ``pack_qubits``).
        num_unpacked: Number of output groups.
        num_elements: Sizes of consecutive groups (splits from the front).
            ``len(num_elements)`` must equal ``num_unpacked`` and
            ``sum(num_elements)`` must equal the total qubit count.
        indices: Per-group index lists for arbitrary reordering.
            Each element must be used exactly once across all groups.

    Returns:
        A tuple of ``Vector[Qubit]`` handles, one per group.
        Even single-element groups return a length-1 ``Vector``.

    Raises:
        TypeError: If ``packed`` is not a ``Vector[Qubit]``.
        ValueError: For invalid arguments (see detailed rules below).
    """
    # --- Input type validation ---
    if not isinstance(packed, Vector) or packed.element_type is not Qubit:
        raise TypeError("unpack_qubits requires a Vector[Qubit]")

    # --- Resolve total size ---
    total = _resolve_vector_size(packed)
    if total is None:
        raise ValueError("unpack_qubits requires a fixed-size Vector[Qubit]")

    # --- num_unpacked validation ---
    if num_unpacked < 1:
        raise ValueError("num_unpacked must be >= 1")

    # --- Mutual exclusion ---
    if num_elements is not None and indices is not None:
        raise ValueError(
            "Cannot specify both num_elements and indices; use exactly one"
        )
    if num_elements is None and indices is None:
        raise ValueError("Must specify either num_elements or indices")

    # --- Build resolved index groups ---
    if num_elements is not None:
        if len(num_elements) != num_unpacked:
            raise ValueError(
                f"len(num_elements)={len(num_elements)} != num_unpacked={num_unpacked}"
            )
        if any(n < 1 for n in num_elements):
            raise ValueError("All num_elements values must be >= 1")
        if sum(num_elements) != total:
            raise ValueError(
                f"sum(num_elements)={sum(num_elements)} != total qubit count={total}"
            )
        # Build consecutive index groups
        resolved_indices: list[list[int]] = []
        offset = 0
        for n in num_elements:
            resolved_indices.append(list(range(offset, offset + n)))
            offset += n
    else:
        assert indices is not None
        if len(indices) != num_unpacked:
            raise ValueError(
                f"len(indices)={len(indices)} != num_unpacked={num_unpacked}"
            )
        # Validate partition: all indices in range, no duplicates, full coverage
        seen: set[int] = set()
        for group in indices:
            if not group:
                raise ValueError("Empty index group is not allowed")
            for idx in group:
                if idx < 0 or idx >= total:
                    raise ValueError(f"Index {idx} out of range [0, {total})")
                if idx in seen:
                    raise ValueError(f"Duplicate index {idx}")
                seen.add(idx)
        if len(seen) != total:
            raise ValueError(
                f"Indices must cover all {total} qubits exactly once, "
                f"but only {len(seen)} were specified"
            )
        resolved_indices = [list(g) for g in indices]

    # --- Read canonical keys from packed vector ---
    all_uuids: list[str] = packed.value.params.get("element_uuids", [])
    all_logical_ids: list[str] = packed.value.params.get("element_logical_ids", [])

    if len(all_uuids) != total:
        raise ValueError(
            f"element_uuids length ({len(all_uuids)}) does not match "
            f"vector size ({total})"
        )

    # --- Consume the packed handle ---
    packed.consume("unpack_qubits")

    # --- Build output sub-vectors ---
    results: list[Vector[Qubit]] = []
    for group_idx, group in enumerate(resolved_indices):
        group_uuids = [all_uuids[i] for i in group]
        group_logical_ids = (
            [all_logical_ids[i] for i in group] if all_logical_ids else []
        )

        group_size = len(group)
        size_value = Value(
            type=UIntType(),
            name=f"unpack_size_{group_idx}",
            params={"const": group_size},
        )
        group_value = ArrayValue(
            type=QubitType(),
            name=f"unpacked_{group_idx}",
            shape=(size_value,),
            params={
                "element_uuids": group_uuids,
                "element_logical_ids": group_logical_ids,
            },
        )
        results.append(
            Vector._create_from_value(
                value=group_value,
                shape=(group_size,),
                name=f"unpacked_{group_idx}",
            )
        )

    return tuple(results)
