"""Helpers for mutating physical-index maps during emission."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping

from qamomile.circuit.transpiler.passes.emit_support.qubit_address import (
    QubitAddress,
)

PhysicalIndexMap = MutableMapping[QubitAddress, int]


def map_array_result_group(
    result_uuid: str,
    index_group: list[int],
    index_map: PhysicalIndexMap,
) -> None:
    """Map an array-valued result to a physical index group.

    Args:
        result_uuid (str): UUID of the array-valued result whose element
            addresses should be registered.
        index_group (list[int]): Physical indices for the result elements, in
            result-local element order.
        index_map (PhysicalIndexMap): Mutable physical-index map updated in
            place with element addresses and, when available, the base result
            address.
    """
    for element_index, physical in enumerate(index_group):
        index_map[QubitAddress(result_uuid, element_index)] = physical
    if index_group:
        index_map[QubitAddress(result_uuid)] = index_group[0]


def array_element_mapping(
    source_uuid: str,
    index_map: Mapping[QubitAddress, int],
) -> dict[int, int]:
    """Collect physical indices for the elements of an array value.

    Args:
        source_uuid (str): UUID of the array whose element addresses should be
            collected.
        index_map (Mapping[QubitAddress, int]): Physical-index map to read.

    Returns:
        dict[int, int]: Mapping from element index to physical index.
    """
    mapping: dict[int, int] = {}
    for addr, physical in index_map.items():
        if addr.matches_array(source_uuid):
            element_index = addr.element_index
            assert element_index is not None
            mapping[element_index] = physical
    return mapping


def array_element_mappings(
    source_uuids: set[str],
    index_map: Mapping[QubitAddress, int],
) -> dict[str, dict[int, int]]:
    """Collect array element mappings for multiple source arrays.

    Args:
        source_uuids (set[str]): UUIDs of the arrays whose element addresses
            should be collected.
        index_map (Mapping[QubitAddress, int]): Physical-index map to read.

    Returns:
        dict[str, dict[int, int]]: Mapping from each requested source UUID to
            its element-index-to-physical-index mapping.
    """
    if not source_uuids:
        return {}

    mappings: dict[str, dict[int, int]] = {uuid: {} for uuid in source_uuids}
    for addr, physical in index_map.items():
        if addr.uuid in source_uuids and addr.element_index is not None:
            mappings[addr.uuid][addr.element_index] = physical
    return mappings


def map_array_element_aliases(
    result_uuid: str,
    element_mapping: Mapping[int, int],
    index_map: PhysicalIndexMap,
    *,
    map_base_address: bool = True,
) -> None:
    """Map explicit array element aliases to a result array.

    Args:
        result_uuid (str): UUID of the result array that should receive the
            element aliases.
        element_mapping (Mapping[int, int]): Physical indices keyed by result
            element index.
        index_map (PhysicalIndexMap): Mutable physical-index map updated in
            place with missing result element addresses and, when requested,
            the result base address.
        map_base_address (bool): Whether to map the scalar base address for
            the result array to its lowest-index element. Defaults to True.
    """
    first_element_index: int | None = None
    first_physical: int | None = None
    for element_index, physical in element_mapping.items():
        result_addr = QubitAddress(result_uuid, element_index)
        if result_addr not in index_map:
            index_map[result_addr] = physical
        if first_element_index is None or element_index < first_element_index:
            first_element_index = element_index
            first_physical = physical
    if (
        map_base_address
        and first_element_index is not None
        and first_physical is not None
    ):
        base_addr = QubitAddress(result_uuid)
        first_element_addr = QubitAddress(result_uuid, first_element_index)
        if base_addr not in index_map:
            index_map[base_addr] = index_map.get(first_element_addr, first_physical)


def copy_array_element_aliases(
    source_uuid: str,
    result_uuid: str,
    index_map: PhysicalIndexMap,
) -> None:
    """Copy array element aliases from a source array to a result array.

    Args:
        source_uuid (str): UUID of the source array whose element addresses are
            already present in ``index_map``.
        result_uuid (str): UUID of the result array that should alias the same
            physical resources element by element.
        index_map (PhysicalIndexMap): Mutable physical-index map updated in
            place with any missing result element addresses and the result base
            address.
    """
    map_array_element_aliases(
        result_uuid,
        array_element_mapping(source_uuid, index_map),
        index_map,
    )
