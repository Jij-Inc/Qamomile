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


@dataclasses.dataclass(frozen=True)
class ProductFormulaContract:
    """Describe one product-formula family and its semantic operand roles.

    Operand positions use the callable's untransformed input ABI. Coherent
    controls added by a later call transform are therefore not included.

    Args:
        kind (str): Product-formula family identifier.
        operands (Mapping[str, int]): Semantic role to callable operand
            position.
    """

    kind: str
    operands: Mapping[str, int]


def product_formula_contract(
    attrs: Mapping[str, Any],
    *,
    source: str,
    operand_count: int | None = None,
) -> ProductFormulaContract | None:
    """Decode product-formula metadata from callable resource attributes.

    Args:
        attrs (Mapping[str, Any]): Callable definition or invocation attrs.
        source (str): Callable name used in malformed-contract diagnostics.
        operand_count (int | None): Optional untransformed input count used to
            reject out-of-range positions. Defaults to ``None``.

    Returns:
        ProductFormulaContract | None: Validated contract, or ``None`` when no
        product formula is declared.

    Raises:
        ValueError: If the resource contract is malformed or references an
            operand outside ``operand_count``.
    """
    resource_contract = attrs.get("resource_contract")
    if resource_contract is None:
        return None
    if not isinstance(resource_contract, Mapping):
        raise ValueError(f"{source} resource_contract must be a mapping.")
    raw_formula = resource_contract.get("product_formula")
    if raw_formula is None:
        return None
    if not isinstance(raw_formula, Mapping):
        raise ValueError(
            f"{source} product_formula resource contract must be a mapping."
        )

    kind = raw_formula.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ValueError(f"{source} product_formula kind must be a nonempty string.")

    positions: dict[str, int] = {}
    for role, position in raw_formula.items():
        if role == "kind":
            continue
        if not isinstance(role, str) or not role:
            raise ValueError(
                f"{source} product_formula operand roles must be nonempty strings."
            )
        if type(position) is not int or position < 0:
            raise ValueError(
                f"{source} product_formula {role} must be a nonnegative integer."
            )
        if operand_count is not None and position >= operand_count:
            raise ValueError(
                f"{source} product_formula {role} references operand "
                f"{position}, but the callable has only {operand_count} input(s)."
            )
        positions[role] = position
    return ProductFormulaContract(kind=kind, operands=positions)


def merge_product_formula_contract(
    attrs: Mapping[str, Any],
    formula: ProductFormulaContract,
    *,
    source: str,
    operand_count: int | None = None,
) -> dict[str, Any]:
    """Merge one product-formula declaration into callable attrs.

    Args:
        attrs (Mapping[str, Any]): Existing callable attributes.
        formula (ProductFormulaContract): Product formula to declare.
        source (str): Callable name used in conflict diagnostics.
        operand_count (int | None): Optional untransformed input count used to
            validate operand positions. Defaults to ``None``.

    Returns:
        dict[str, Any]: Copied attributes with the product-formula contract.

    Raises:
        ValueError: If existing resource metadata is malformed or conflicts
            with ``formula``.
    """
    if "kind" in formula.operands:
        raise ValueError(f"{source} product_formula operand role 'kind' is reserved.")
    formula_payload: dict[str, Any] = {"kind": formula.kind}
    formula_payload.update(formula.operands)
    payload = {
        "resource_contract": {
            "product_formula": formula_payload,
        }
    }
    requested = product_formula_contract(
        payload,
        source=source,
        operand_count=operand_count,
    )
    assert requested is not None

    merged_attrs = dict(attrs)
    existing_contract = merged_attrs.get("resource_contract")
    if existing_contract is None:
        contract: dict[str, Any] = {}
    elif isinstance(existing_contract, Mapping):
        contract = dict(existing_contract)
    else:
        raise ValueError(f"{source} resource_contract must be a mapping.")
    existing = product_formula_contract(
        {"resource_contract": contract},
        source=source,
        operand_count=operand_count,
    )
    if existing is not None and existing != requested:
        raise ValueError(f"{source} product_formula resource contract conflicts.")
    contract["product_formula"] = {
        "kind": requested.kind,
        **requested.operands,
    }
    merged_attrs["resource_contract"] = contract
    return merged_attrs


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
    "ProductFormulaContract",
    "QuantumOperandWidth",
    "merge_product_formula_contract",
    "merge_quantum_operand_widths",
    "product_formula_contract",
    "quantum_operand_widths",
]
