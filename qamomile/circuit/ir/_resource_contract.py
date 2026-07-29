"""Decode serializer-friendly resource contracts carried by callable attrs."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any


@dataclasses.dataclass(frozen=True)
class QuantumOperandWidth:
    """Describe one exact source-callable quantum operand width.

    Args:
        index (int): Position among quantum operands only.
        name (str): User-facing register name.
        width (int): Required scalar qubit width.
    """

    index: int
    name: str
    width: int


def quantum_operand_widths(
    attrs: Mapping[str, Any],
    *,
    source: str,
) -> tuple[QuantumOperandWidth, ...]:
    """Decode exact quantum-operand widths from callable resource metadata.

    Args:
        attrs (Mapping[str, Any]): Callable definition or operation attrs.
        source (str): Callable name used in malformed-contract diagnostics.

    Returns:
        tuple[QuantumOperandWidth, ...]: Validated exact-width entries, or an
            empty tuple when the callable declares no such contract.

    Raises:
        ValueError: If present resource metadata is malformed or repeats an
            operand index.
    """
    resource_contract = attrs.get("resource_contract")
    if resource_contract is None:
        return ()
    if not isinstance(resource_contract, Mapping):
        raise ValueError(f"{source} resource_contract must be a mapping.")
    raw_widths = resource_contract.get("quantum_operand_widths")
    if raw_widths is None:
        return ()
    if not isinstance(raw_widths, (list, tuple)):
        raise ValueError(
            f"{source} quantum_operand_widths resource contract must be a list."
        )

    decoded: list[QuantumOperandWidth] = []
    seen_indices: set[int] = set()
    for position, entry in enumerate(raw_widths):
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"{source} quantum operand width entry {position} must be a mapping."
            )
        index = entry.get("index")
        width = entry.get("width")
        name = entry.get("name")
        if type(index) is not int or index < 0:
            raise ValueError(
                f"{source} quantum operand width entry {position} has an invalid index."
            )
        if index in seen_indices:
            raise ValueError(
                f"{source} quantum operand width contract repeats index {index}."
            )
        if type(width) is not int or width < 0:
            raise ValueError(
                f"{source} quantum operand width entry {position} has an invalid width."
            )
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"{source} quantum operand width entry {position} has an invalid name."
            )
        seen_indices.add(index)
        decoded.append(QuantumOperandWidth(index=index, name=name, width=width))
    return tuple(decoded)


def merge_quantum_operand_widths(
    attrs: Mapping[str, Any],
    widths: Sequence[QuantumOperandWidth],
    *,
    source: str,
    operand_count: int | None = None,
    conflict_labels: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Merge exact quantum widths into serializer-friendly callable attrs.

    Existing resource-contract keys are preserved. Width entries are merged by
    quantum-operand index, compatible partial declarations are completed, and
    the encoded list is canonicalized by index.

    Args:
        attrs (Mapping[str, Any]): Existing callable attributes.
        widths (Sequence[QuantumOperandWidth]): Width declarations to merge.
        source (str): Callable name used in malformed-contract diagnostics.
        operand_count (int | None): Optional quantum operand count used to
            reject out-of-range entries. Defaults to ``None``.
        conflict_labels (Mapping[int, str] | None): Optional caller-facing
            field labels used for width-conflict diagnostics. Defaults to
            operand-index diagnostics.

    Returns:
        dict[str, Any]: Copied attributes with the merged resource contract.

    Raises:
        ValueError: If the existing or requested contract is malformed,
            repeats or conflicts at an operand index, or references an index
            outside ``operand_count``.
    """
    merged_attrs = dict(attrs)
    existing_contract = merged_attrs.get("resource_contract")
    if existing_contract is None:
        contract: dict[str, Any] = {}
    elif isinstance(existing_contract, Mapping):
        contract = dict(existing_contract)
    else:
        raise ValueError(f"{source} resource_contract must be a mapping.")

    requested_payload = {
        "resource_contract": {
            "quantum_operand_widths": [
                {
                    "index": entry.index,
                    "name": entry.name,
                    "width": entry.width,
                }
                for entry in widths
            ]
        }
    }
    requested = quantum_operand_widths(requested_payload, source=source)
    merged = {
        entry.index: entry
        for entry in quantum_operand_widths(
            {"resource_contract": contract},
            source=source,
        )
    }
    for entry in requested:
        previous = merged.get(entry.index)
        if previous is not None and previous != entry:
            field_label = (
                conflict_labels.get(entry.index)
                if conflict_labels is not None
                else None
            )
            if field_label is not None:
                raise ValueError(
                    f"{field_label} conflicts with {source}'s existing "
                    "resource contract."
                )
            raise ValueError(
                f"{source} quantum operand {entry.index} resource contract "
                f"conflicts: existing ({previous.name!r}, {previous.width}) "
                f"versus requested ({entry.name!r}, {entry.width})."
            )
        merged[entry.index] = entry

    if operand_count is not None:
        if type(operand_count) is not int or operand_count < 0:
            raise ValueError(f"{source} operand_count must be a nonnegative integer.")
        out_of_range = [index for index in merged if index >= operand_count]
        if out_of_range:
            raise ValueError(
                f"{source} resource contract references quantum operand "
                f"{min(out_of_range)}, but the callable has only "
                f"{operand_count} quantum operand(s)."
            )

    contract["quantum_operand_widths"] = [
        {
            "index": entry.index,
            "name": entry.name,
            "width": entry.width,
        }
        for _, entry in sorted(merged.items())
    ]
    merged_attrs["resource_contract"] = contract
    return merged_attrs


__all__ = [
    "QuantumOperandWidth",
    "merge_quantum_operand_widths",
    "quantum_operand_widths",
]
