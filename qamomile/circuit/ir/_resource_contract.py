"""Decode serializer-friendly resource contracts carried by callable attrs."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
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


__all__ = ["QuantumOperandWidth", "quantum_operand_widths"]
