"""Helpers for compile-time condition resolution and phi remapping."""

from __future__ import annotations

from typing import Any

from qamomile.circuit.ir.operation.arithmetic_operations import PhiOp
from qamomile.circuit.ir.types.primitives import BitType
from qamomile.circuit.ir.value import ArrayValue

from .qubit_address import ClbitMap, QubitAddress, QubitMap
from .value_resolver import resolve_qubit_key


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
                is_array_src = isinstance(selected_val, ArrayValue)
                for addr, phys_idx in list(qubit_map.items()):
                    if is_array_src and addr.matches_array(selected_val.uuid):
                        out_addr = QubitAddress(output.uuid, addr.element_index)
                        if out_addr not in qubit_map:
                            qubit_map[out_addr] = phys_idx
            else:
                key, _ = resolve_qubit_key(selected_val)
                scalar_addr = QubitAddress(selected_val.uuid)
                phys = qubit_map.get(scalar_addr)
                if phys is None and key is not None:
                    phys = qubit_map.get(key)
                if phys is not None:
                    qubit_map[QubitAddress(output.uuid)] = phys
        else:
            src_addr = QubitAddress(selected_val.uuid)
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
                true_is_array = isinstance(true_val, ArrayValue)
                false_is_array = isinstance(false_val, ArrayValue)
                true_mapping: dict[int, int] = {}
                false_mapping: dict[int, int] = {}
                for addr, phys_idx in qubit_map.items():
                    if true_is_array and addr.matches_array(true_val.uuid):
                        assert addr.element_index is not None
                        true_mapping[addr.element_index] = phys_idx
                    if false_is_array and addr.matches_array(false_val.uuid):
                        assert addr.element_index is not None
                        false_mapping[addr.element_index] = phys_idx

                if true_mapping or false_mapping:
                    all_indices = set(true_mapping) | set(false_mapping)
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
                        out_addr = QubitAddress(output.uuid, idx)
                        if out_addr not in qubit_map:
                            qubit_map[out_addr] = true_idx
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
                    for addr, phys_idx in list(clbit_map.items()):
                        if addr.matches_array(primary.uuid):
                            assert addr.element_index is not None
                            out_addr = QubitAddress(output.uuid, addr.element_index)
                            if out_addr not in clbit_map:
                                clbit_map[out_addr] = phys_idx
                            if secondary is not None:
                                sec_addr = QubitAddress(
                                    secondary.uuid, addr.element_index
                                )
                                if sec_addr in clbit_map:
                                    clbit_map[sec_addr] = phys_idx
            else:
                true_clbit = clbit_map.get(QubitAddress(true_val.uuid))
                false_clbit = clbit_map.get(QubitAddress(false_val.uuid))

                if true_clbit is not None:
                    clbit_map[QubitAddress(output.uuid)] = true_clbit
                    if false_clbit is not None and false_clbit != true_clbit:
                        clbit_map[QubitAddress(false_val.uuid)] = true_clbit
                elif false_clbit is not None:
                    clbit_map[QubitAddress(output.uuid)] = false_clbit
