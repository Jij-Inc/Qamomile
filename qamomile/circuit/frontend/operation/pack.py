"""Pack and unpack qubits into/from a Vector[Qubit].

This module provides ``pack_qubits`` and ``unpack_qubits`` for combining and
splitting qubit handles.  Both functions enforce a linear-type contract:
input handles are *consumed* and must not be reused after the call.

The resulting vectors carry the canonical ``element_uuids`` key that
downstream passes (separate, emit) use to resolve qubit ordering.
"""

from __future__ import annotations

from qamomile.circuit.frontend.handle.primitives import Qubit, UInt as UIntHandle
from qamomile.circuit.frontend.handle.array import ArrayBase, Vector
from qamomile.circuit.ir.types.primitives import QubitType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value


def _is_symbolic(val: int | UIntHandle) -> bool:
    """Check if a num_elements entry is symbolic (not a concrete int)."""
    if isinstance(val, int):
        return False
    if hasattr(val, "value") and hasattr(val.value, "is_constant"):
        return not val.value.is_constant()
    return True


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


def _get_root_source_logical_id(item: Qubit | ArrayBase) -> str | None:
    """Get the root source logical_id of a pack item.

    For items produced by ``unpack_qubits``, traces back through
    ``split_spec.source_logical_id`` (sub-vectors) or
    ``parent_array`` → ``split_spec`` (element access on sub-vectors).
    For other items, returns the item's own logical_id.
    """
    if isinstance(item, Qubit):
        val = item.value
        if val.parent_array is not None:
            split_spec = val.parent_array.params.get("split_spec")
            if split_spec:
                return split_spec["source_logical_id"]
        return val.logical_id
    elif isinstance(item, ArrayBase):
        val = item.value
        split_spec = val.params.get("split_spec")
        if split_spec:
            return split_spec["source_logical_id"]
        return val.logical_id
    return None


def pack_qubits(*items: Qubit | Vector[Qubit]) -> Vector[Qubit]:
    """Pack individual qubits and/or qubit vectors into one vector.

    Accepts both concrete-size and symbolic-size ``Vector[Qubit]`` inputs.
    For concrete inputs, ``element_uuids`` metadata is collected and stored.
    For symbolic inputs, metadata resolution is deferred to the emit layer.

    Args:
        *items: One or more ``Qubit`` or ``Vector[Qubit]`` handles.

    Returns:
        A new ``Vector[Qubit]`` whose ``element_uuids`` param records the
        UUID ordering of the packed qubits (for concrete inputs).

    Raises:
        ValueError: If no arguments are given.
        TypeError: If an argument is neither ``Qubit`` nor ``Vector[Qubit]``.
    """
    if len(items) == 0:
        raise ValueError("pack_qubits requires at least one qubit")

    element_uuids: list[str] = []
    element_logical_ids: list[str] = []
    has_symbolic = False
    symbolic_shape_parts: list[int | Value] = []

    for item in items:
        if isinstance(item, Qubit):
            item.consume("pack_qubits")
            element_uuids.append(item.value.uuid)
            element_logical_ids.append(item.value.logical_id)

        elif isinstance(item, ArrayBase) and item.element_type is Qubit:
            n = _resolve_vector_size(item)
            if n is None:
                # Symbolic vector — defer metadata to emit time
                has_symbolic = True
                # Track shape for symbolic total computation
                if item.shape:
                    s = item.shape[0]
                    symbolic_shape_parts.append(
                        s.value if isinstance(s, UIntHandle) else s
                    )
                elif item.value.shape:
                    symbolic_shape_parts.append(item.value.shape[0])
                item.consume("pack_qubits")
            else:
                # Concrete — collect element metadata
                for i in range(n):
                    elem = item[i]
                    element_uuids.append(elem.value.uuid)
                    element_logical_ids.append(elem.value.logical_id)
                    item[i] = elem
                item.consume("pack_qubits")

        else:
            raise TypeError("pack_qubits accepts only Qubit or Vector[Qubit]")

    if has_symbolic:
        # Build symbolic total using UInt handle arithmetic (emits BinOps)
        concrete_count = len(element_uuids)
        acc_handle: UIntHandle | None = None

        if concrete_count > 0:
            acc_handle = UIntHandle(
                value=Value(
                    type=UIntType(),
                    name="pack_concrete_count",
                    params={"const": concrete_count},
                ),
                init_value=concrete_count,
            )

        for part in symbolic_shape_parts:
            part_handle = UIntHandle(value=part) if isinstance(part, Value) else part
            if acc_handle is None:
                acc_handle = part_handle
            else:
                acc_handle = acc_handle + part_handle

        assert acc_handle is not None
        size_value = acc_handle.value
        packed_value = ArrayValue(
            type=QubitType(),
            name="packed_qubits",
            shape=(size_value,),
            params={
                "element_uuids": element_uuids,
                "element_logical_ids": element_logical_ids,
                "has_symbolic_components": True,
            },
        )

        # Propagate source logical_id if all items trace to the same source
        root_sources = set()
        for item in items:
            root = _get_root_source_logical_id(item)
            if root is not None:
                root_sources.add(root)
        if len(root_sources) == 1:
            packed_value.logical_id = root_sources.pop()

        return Vector._create_from_value(
            value=packed_value,
            shape=(acc_handle,),
            name="packed_qubits",
        )
    else:
        # Fully concrete path
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

        # Propagate source logical_id if all items trace to the same source
        root_sources = set()
        for item in items:
            root = _get_root_source_logical_id(item)
            if root is not None:
                root_sources.add(root)
        if len(root_sources) == 1:
            packed_value.logical_id = root_sources.pop()

        return Vector._create_from_value(
            value=packed_value,
            shape=(total,),
            name="packed_qubits",
        )


def unpack_qubits(
    packed: Vector[Qubit],
    *,
    num_unpacked: int,
    num_elements: list[int | UIntHandle] | None = None,
    indices: list[list[int]] | None = None,
) -> tuple[Vector[Qubit], ...]:
    """Split a packed qubit vector into multiple sub-vectors.

    Exactly one of ``num_elements`` or ``indices`` must be provided.

    When all sizes are concrete, index groups are computed immediately and
    ``element_uuids`` metadata is propagated to each output vector.

    When the packed vector has a symbolic size or ``num_elements`` contains
    symbolic values (``UInt`` handles), index computation is deferred: each
    output vector stores a ``split_spec`` in its params for later resolution
    by the transpiler/emit layer.

    Args:
        packed: A ``Vector[Qubit]`` to split.
        num_unpacked: Number of output groups.
        num_elements: Sizes of consecutive groups.  Entries may be ``int``
            or ``UInt`` handles for symbolic sizes.
            ``len(num_elements)`` must equal ``num_unpacked``.
        indices: Per-group index lists for arbitrary reordering.
            Requires a concrete-size vector.  All values must be ``int``.

    Returns:
        A tuple of ``Vector[Qubit]`` handles, one per group.

    Raises:
        TypeError: If ``packed`` is not a ``Vector[Qubit]``, or if
            ``indices`` contains non-int values.
        ValueError: For invalid arguments.
    """
    # --- Input type validation ---
    if not isinstance(packed, Vector) or packed.element_type is not Qubit:
        raise TypeError("unpack_qubits requires a Vector[Qubit]")

    # --- Resolve total size (may be None for symbolic vectors) ---
    total = _resolve_vector_size(packed)

    # Capture symbolic total expression for split_spec
    total_expr: Value | None = None
    if total is None:
        if packed.shape:
            s = packed.shape[0]
            total_expr = s.value if isinstance(s, UIntHandle) else s
        elif packed.value.shape:
            total_expr = packed.value.shape[0]

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
    symbolic_mode = False
    resolved_indices: list[list[int]] | None = None

    if num_elements is not None:
        if len(num_elements) != num_unpacked:
            raise ValueError(
                f"len(num_elements)={len(num_elements)} != num_unpacked={num_unpacked}"
            )

        any_symbolic = any(_is_symbolic(n) for n in num_elements)
        total_is_symbolic = total is None

        if total_is_symbolic or any_symbolic:
            # === SYMBOLIC PATH ===
            # Do NOT call range(offset, offset + n) — store split_spec instead.
            # Only validate what can be checked immediately.
            symbolic_mode = True
            for n in num_elements:
                if isinstance(n, int) and n < 1:
                    raise ValueError("All num_elements values must be >= 1")
        else:
            # === CONCRETE PATH ===
            if any(n < 1 for n in num_elements):
                raise ValueError("All num_elements values must be >= 1")
            if sum(num_elements) != total:
                raise ValueError(
                    f"sum(num_elements)={sum(num_elements)} "
                    f"!= total qubit count={total}"
                )
            resolved_indices = []
            offset = 0
            for n in num_elements:
                resolved_indices.append(list(range(offset, offset + n)))
                offset += n
    else:
        assert indices is not None
        # --- Indices mode: requires concrete total ---
        if total is None:
            raise ValueError(
                "unpack_qubits with indices= requires a fixed-size "
                "Vector[Qubit]. Use num_elements= for symbolic vectors."
            )
        if len(indices) != num_unpacked:
            raise ValueError(
                f"len(indices)={len(indices)} != num_unpacked={num_unpacked}"
            )
        # Validate all index values are int
        for group in indices:
            for idx in group:
                if not isinstance(idx, int):
                    raise TypeError(
                        f"indices values must be int, got {type(idx).__name__}"
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

    # Phase 1 fallback: derive element_uuids from source vector UUID
    if not all_uuids and total is not None:
        source_uuid = packed.value.uuid
        source_logical_id = packed.value.logical_id
        all_uuids = [f"{source_uuid}_{i}" for i in range(total)]
        all_logical_ids = [f"{source_logical_id}_{i}" for i in range(total)]

    # Validate length only when both are available
    if total is not None and all_uuids and len(all_uuids) != total:
        raise ValueError(
            f"element_uuids length ({len(all_uuids)}) does not match "
            f"vector size ({total})"
        )

    # --- Consume the packed handle ---
    packed.consume("unpack_qubits")

    # --- Build output sub-vectors ---
    results: list[Vector[Qubit]] = []

    if symbolic_mode:
        assert num_elements is not None
        # Build split_spec for deferred resolution
        ne_exprs: list[int | Value] = []
        for ne in num_elements:
            if isinstance(ne, int):
                ne_exprs.append(ne)
            elif isinstance(ne, UIntHandle):
                ne_exprs.append(ne.value)
            else:
                ne_exprs.append(ne)

        source_uuid = packed.value.uuid
        source_logical_id = packed.value.logical_id

        for group_idx in range(num_unpacked):
            n_expr = num_elements[group_idx]

            # Determine the size Value for this group
            if isinstance(n_expr, int):
                size_value = Value(
                    type=UIntType(),
                    name=f"unpack_size_{group_idx}",
                    params={"const": n_expr},
                )
                frontend_shape: int | UIntHandle = n_expr
            elif isinstance(n_expr, UIntHandle):
                size_value = n_expr.value
                frontend_shape = n_expr
            else:
                size_value = n_expr
                frontend_shape = UIntHandle(value=n_expr)

            group_value = ArrayValue(
                type=QubitType(),
                name=f"unpacked_{group_idx}",
                shape=(size_value,),
                params={
                    "split_spec": {
                        "mode": "num_elements",
                        "source_uuid": source_uuid,
                        "source_logical_id": source_logical_id,
                        "group_index": group_idx,
                        "num_elements_expr": ne_exprs,
                        "total_expr": total_expr,
                    },
                },
            )
            results.append(
                Vector._create_from_value(
                    value=group_value,
                    shape=(frontend_shape,),
                    name=f"unpacked_{group_idx}",
                )
            )
    else:
        # Concrete path
        assert resolved_indices is not None
        for group_idx, group in enumerate(resolved_indices):
            group_uuids = [all_uuids[i] for i in group] if all_uuids else []
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
