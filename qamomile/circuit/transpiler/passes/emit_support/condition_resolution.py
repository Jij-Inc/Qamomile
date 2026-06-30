"""Helpers for compile-time condition resolution and phi remapping."""

from __future__ import annotations

from typing import Any

from qamomile.circuit.ir.operation.arithmetic_operations import PhiOp
from qamomile.circuit.ir.types.primitives import BitType
from qamomile.circuit.ir.value import ArrayValue, Value

from .physical_index_map import (
    array_element_mapping,
    copy_array_element_aliases,
    map_array_element_aliases,
)
from .qubit_address import ClbitMap, QubitAddress, QubitMap
from .value_resolver import ValueResolver, resolve_qubit_key


def resolve_condition_address_detailed(
    condition: Value,
    bindings: dict[str, Any],
    resolver: ValueResolver | None = None,
) -> tuple[QubitAddress, bool]:
    """Resolve a condition / source ``Value`` to its ``clbit_map`` key.

    Single source of truth (shared by ``control_flow_emission.emit_if`` /
    ``emit_while``, the Qiskit / CUDA-Q backends, ``ResourceAllocator``'s
    loop-carried / phi aliasing, and the phi-output mapping helpers here)
    for turning a runtime control-flow condition — or a phi source Value —
    into the address its classical bit is registered under.

    Scalar measurement results register under ``QubitAddress(bit.uuid)``,
    while a ``Vector[Bit]`` element ``s[i]`` (``s = qmc.measure(register)``)
    registers under ``QubitAddress(root_array.uuid, root_index)``. The
    element index and every ``slice_start`` / ``slice_step`` along the
    parent's ``slice_of`` chain are resolved the same way — taken directly
    when constant, otherwise folded through ``bindings`` via ``resolver`` —
    and composed into the root index via ``root_index = start + step *
    view_local_index`` repeated along the chain.

    Args:
        condition (Value): The condition operand or phi source Value.
        bindings (dict[str, Any]): Active emit-time bindings used to fold
            symbolic indices / slice bounds. May be empty for callers that
            only have constant addresses to resolve (e.g. the allocator's
            pre-emit pass).
        resolver (ValueResolver | None): Resolver exposing
            ``resolve_int_value``; ``None`` restricts resolution to
            constants. Defaults to None.

    Returns:
        tuple[QubitAddress, bool]: ``(address, resolved_as_element)``.
            ``resolved_as_element`` is ``True`` only when ``condition`` is a
            ``Vector[Bit]`` element whose index (and any slice bounds)
            resolved to concrete ints, giving the root-space
            ``QubitAddress(root.uuid, root_index)``. It is ``False`` for a
            plain scalar, or when an element's index / slice bound could
            not be resolved (the scalar UUID fallback). Callers that mutate
            ``clbit_map`` should trust the address as a vector key only when
            this flag is ``True``, to avoid aliasing an unresolved element
            onto an unrelated scalar slot.
    """

    def _resolve_int(value: Value | None) -> int | None:
        """Resolve an index / slice-bound Value to a concrete int or None.

        Args:
            value (Value | None): The index or slice-bound Value, or None.

        Returns:
            int | None: The concrete integer when constant or resolvable
                through ``bindings`` via ``resolver``; ``None`` otherwise.
        """
        if value is None:
            return None
        if value.is_constant():
            return int(value.get_const())
        if resolver is not None:
            return resolver.resolve_int_value(value, bindings)
        return None

    if condition.parent_array is None or not condition.element_indices:
        return QubitAddress(condition.uuid), False
    idx = _resolve_int(condition.element_indices[0])
    if idx is None:
        return QubitAddress(condition.uuid), False
    parent = condition.parent_array
    while parent.slice_of is not None:
        start = _resolve_int(parent.slice_start)
        step = _resolve_int(parent.slice_step)
        if start is None or step is None:
            return QubitAddress(condition.uuid), False
        idx = start + step * idx
        parent = parent.slice_of
    return QubitAddress(parent.uuid, idx), True


def _coerce_to_bool(value: Any) -> bool | None:
    """Coerce a Python scalar to bool; return None for non-scalar values.

    A backend-specific runtime expression (e.g. ``qiskit.circuit.classical.expr.Expr``)
    may be stored in ``bindings`` for the same UUID slot as a compile-time
    Python bool. This guard ensures we don't accidentally call ``bool()`` on
    such an object — that would either raise or return a misleading truthy
    value.

    Args:
        value: The value found in bindings (or the condition's constant).

    Returns:
        ``True`` / ``False`` for ``bool``/``int`` inputs, ``None`` otherwise.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    return None


def resolve_if_condition(
    condition: Any,
    bindings: dict[str, Any],
) -> bool | None:
    """Resolve an if-condition to a compile-time boolean."""
    if not hasattr(condition, "uuid"):
        return _coerce_to_bool(condition)
    if hasattr(condition, "is_constant") and condition.is_constant():
        return _coerce_to_bool(condition.get_const())
    if condition.uuid in bindings:
        return _coerce_to_bool(bindings[condition.uuid])
    if hasattr(condition, "name") and condition.name and condition.name in bindings:
        return _coerce_to_bool(bindings[condition.name])
    return None


def remap_static_phi_outputs(
    phi_ops: list,
    condition_value: bool,
    qubit_map: QubitMap,
    clbit_map: ClbitMap,
) -> None:
    """Remap phi outputs for a compile-time constant ``IfOperation``."""
    for phi in phi_ops:
        if not isinstance(phi, PhiOp):
            continue
        output = phi.results[0]
        selected_val = phi.operands[1] if condition_value else phi.operands[2]

        if (
            QubitAddress(output.uuid) in qubit_map
            or QubitAddress(output.uuid) in clbit_map
        ):
            continue

        if output.type.is_quantum():
            if isinstance(output, ArrayValue):
                if isinstance(selected_val, ArrayValue):
                    copy_array_element_aliases(
                        selected_val.uuid, output.uuid, qubit_map
                    )
            else:
                key, _ = resolve_qubit_key(selected_val)
                scalar_addr = QubitAddress(selected_val.uuid)
                phys = qubit_map.get(scalar_addr)
                if phys is None and key is not None:
                    phys = qubit_map.get(key)
                if phys is not None:
                    qubit_map[QubitAddress(output.uuid)] = phys
        else:
            # Scalar Bit source: resolve vector-element sources (``s[i]``
            # from a measured ``Vector[Bit]``) to their root clbit key, not
            # the element's own UUID (which is not registered in clbit_map).
            src_addr, _ = resolve_condition_address_detailed(selected_val, {}, None)
            if src_addr in clbit_map:
                clbit_map[QubitAddress(output.uuid)] = clbit_map[src_addr]


def map_phi_outputs(
    phi_ops: list,
    qubit_map: QubitMap,
    clbit_map: ClbitMap,
    resolve_scalar_qubit: Any = None,
) -> None:
    """Register phi output UUIDs to the same physical resources as their sources."""
    for phi in phi_ops:
        if not isinstance(phi, PhiOp):
            continue
        output = phi.results[0]
        true_val = phi.operands[1]
        false_val = phi.operands[2]

        if (
            QubitAddress(output.uuid) in qubit_map
            or QubitAddress(output.uuid) in clbit_map
        ):
            continue

        if output.type.is_quantum():
            if isinstance(output, ArrayValue):
                true_mapping = (
                    array_element_mapping(true_val.uuid, qubit_map)
                    if isinstance(true_val, ArrayValue)
                    else {}
                )
                false_mapping = (
                    array_element_mapping(false_val.uuid, qubit_map)
                    if isinstance(false_val, ArrayValue)
                    else {}
                )

                if true_mapping or false_mapping:
                    all_indices = set(true_mapping) | set(false_mapping)
                    output_mapping: dict[int, int] = {}
                    for idx in sorted(all_indices):
                        true_idx = true_mapping.get(idx)
                        false_idx = false_mapping.get(idx)
                        if (
                            true_idx is None
                            or false_idx is None
                            or true_idx != false_idx
                        ):
                            from qamomile.circuit.transpiler.errors import EmitError

                            raise EmitError(
                                "Quantum PhiOp merge requires identical physical "
                                "resources across branches",
                                operation="PhiOp",
                            )
                        output_mapping[idx] = true_idx
                    map_array_element_aliases(output.uuid, output_mapping, qubit_map)
            else:

                def _resolve_one(source: Any) -> int | None:
                    src_addr = QubitAddress(source.uuid)
                    if src_addr in qubit_map:
                        return qubit_map[src_addr]
                    if resolve_scalar_qubit is not None:
                        return resolve_scalar_qubit(source, qubit_map)
                    key, _ = resolve_qubit_key(source)
                    if key is not None:
                        return qubit_map.get(key)
                    return None

                true_phys = _resolve_one(true_val)
                false_phys = _resolve_one(false_val)

                if true_phys is not None and false_phys is not None:
                    if true_phys != false_phys:
                        from qamomile.circuit.transpiler.errors import EmitError

                        raise EmitError(
                            "Quantum PhiOp merge requires identical physical "
                            "resources across branches",
                            operation="PhiOp",
                        )
                    qubit_map[QubitAddress(output.uuid)] = true_phys
                elif true_phys is not None or false_phys is not None:
                    from qamomile.circuit.transpiler.errors import EmitError

                    raise EmitError(
                        "Quantum PhiOp merge requires identical physical "
                        "resources across branches",
                        operation="PhiOp",
                    )

        elif isinstance(output.type, BitType):
            if isinstance(output, ArrayValue):
                true_src = true_val if isinstance(true_val, ArrayValue) else None
                false_src = false_val if isinstance(false_val, ArrayValue) else None
                primary = true_src or false_src
                secondary = false_src if true_src is not None else true_src

                if primary is not None:
                    primary_mapping = array_element_mapping(primary.uuid, clbit_map)
                    map_array_element_aliases(
                        output.uuid,
                        primary_mapping,
                        clbit_map,
                        map_base_address=False,
                    )
                    if secondary is not None:
                        for element_index, phys_idx in primary_mapping.items():
                            sec_addr = QubitAddress(secondary.uuid, element_index)
                            if sec_addr in clbit_map:
                                clbit_map[sec_addr] = phys_idx
            else:
                # Scalar Bit phi: resolve vector-element sources to their
                # root clbit key so a measured ``Vector[Bit]`` element merged
                # through a phi (``if sel: bit = s[0] else: bit = t[0]``)
                # maps to the right clbit instead of the element's own UUID.
                true_addr, _ = resolve_condition_address_detailed(true_val, {}, None)
                false_addr, _ = resolve_condition_address_detailed(false_val, {}, None)
                true_clbit = clbit_map.get(true_addr)
                false_clbit = clbit_map.get(false_addr)

                if true_clbit is not None:
                    clbit_map[QubitAddress(output.uuid)] = true_clbit
                    if false_clbit is not None and false_clbit != true_clbit:
                        clbit_map[false_addr] = true_clbit
                elif false_clbit is not None:
                    clbit_map[QubitAddress(output.uuid)] = false_clbit
