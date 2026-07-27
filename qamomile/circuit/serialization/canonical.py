"""Canonicalize process-local identities in serialized qkernel graphs."""

from __future__ import annotations

import copy
import uuid
from typing import Any

_VALUE_NAMESPACE = uuid.UUID("9529c31d-33c5-59c2-9f1a-518667dcf104")
_LOGICAL_NAMESPACE = uuid.UUID("81394e4d-0599-5dfd-873a-59eb7bc42c80")


def canonicalize_graph(envelope: dict[str, Any]) -> dict[str, Any]:
    """Replace random value identities with deterministic graph-local IDs.

    Value-table order is the semantic encoder's deterministic first-visit order.
    The rewrite preserves every equality and reference relationship while making
    independently traced instances of the same qkernel byte-identical.

    Args:
        envelope (dict[str, Any]): Complete internal qkernel graph envelope.

    Returns:
        dict[str, Any]: A deep-copied envelope with canonical UUID and logical-ID
            strings.

    Raises:
        ValueError: If the value table or an identity-bearing field is malformed.
    """
    value_table = envelope.get("value_table")
    if not isinstance(value_table, list):
        raise ValueError("canonical qkernel graph requires a value_table list")

    uuid_remap: dict[str, str] = {}
    logical_id_remap: dict[str, str] = {}
    for index, entry in enumerate(value_table):
        if not isinstance(entry, dict):
            raise ValueError("value_table entries must be dictionaries")
        value_uuid = entry.get("uuid")
        logical_id = entry.get("logical_id")
        if not isinstance(value_uuid, str) or not isinstance(logical_id, str):
            raise ValueError("value_table entries require uuid and logical_id strings")
        if value_uuid in uuid_remap:
            raise ValueError(f"duplicate UUID in value_table: {value_uuid!r}")
        uuid_remap[value_uuid] = str(uuid.uuid5(_VALUE_NAMESPACE, f"value-{index}"))
        if logical_id not in logical_id_remap:
            logical_id_remap[logical_id] = str(
                uuid.uuid5(
                    _LOGICAL_NAMESPACE,
                    f"logical-{len(logical_id_remap)}",
                )
            )

    result = copy.deepcopy(envelope)
    _rewrite_node(result, uuid_remap, logical_id_remap)
    return result


def _rewrite_node(
    node: Any,
    uuid_remap: dict[str, str],
    logical_id_remap: dict[str, str],
) -> None:
    """Rewrite identity-bearing fields recursively in one graph node.

    Args:
        node (Any): Current internal graph node.
        uuid_remap (dict[str, str]): Original-to-canonical value UUID map.
        logical_id_remap (dict[str, str]): Original-to-canonical logical-ID map.

    Raises:
        ValueError: If an identity-bearing field is malformed.
    """
    if isinstance(node, list):
        for item in node:
            _rewrite_node(item, uuid_remap, logical_id_remap)
        return
    if not isinstance(node, dict):
        return

    for key, value in node.items():
        if key in {"uuid", "source_uuid"} and isinstance(value, str):
            node[key] = _mapped(value, uuid_remap, key, _VALUE_NAMESPACE)
        elif key in {"logical_id", "source_logical_id"} and isinstance(value, str):
            node[key] = _mapped(value, logical_id_remap, key, _LOGICAL_NAMESPACE)
        elif key == "entry_refs" and isinstance(value, list):
            node[key] = [
                (
                    _mapped(pair[0], uuid_remap, key, _VALUE_NAMESPACE),
                    _mapped(pair[1], uuid_remap, key, _VALUE_NAMESPACE),
                )
                for pair in value
            ]
        elif (
            key.endswith("_ref") and key != "definition_ref" and isinstance(value, str)
        ):
            node[key] = _mapped(value, uuid_remap, key, _VALUE_NAMESPACE)
        elif key.endswith("_refs") and isinstance(value, list):
            node[key] = [
                _mapped(item, uuid_remap, key, _VALUE_NAMESPACE) for item in value
            ]
        elif key.endswith("_uuids") and isinstance(value, list):
            node[key] = [
                None
                if item is None
                else _mapped(item, uuid_remap, key, _VALUE_NAMESPACE)
                for item in value
            ]
        elif key.endswith("_logical_ids") and isinstance(value, list):
            node[key] = [
                _mapped(item, logical_id_remap, key, _LOGICAL_NAMESPACE)
                for item in value
            ]
        elif key == "qubit_mapping" and isinstance(value, list):
            node[key] = [
                _mapped(item, uuid_remap, key, _VALUE_NAMESPACE) for item in value
            ]
        elif key == "parameters" and isinstance(value, dict):
            node[key] = {
                name: _mapped(reference, uuid_remap, key, _VALUE_NAMESPACE)
                for name, reference in value.items()
            }
        elif key == "$value_ref" and isinstance(value, str):
            node[key] = _mapped(value, uuid_remap, key, _VALUE_NAMESPACE)
        else:
            _rewrite_node(value, uuid_remap, logical_id_remap)


def _mapped(
    value: Any,
    mapping: dict[str, str],
    field: str,
    namespace: uuid.UUID,
) -> str:
    """Resolve one identity through a canonical remapping.

    Args:
        value (Any): Identity value found in the graph.
        mapping (dict[str, str]): Applicable canonical ID map.
        field (str): Field name used for diagnostics.
        namespace (uuid.UUID): Namespace for metadata-only identities that do
            not have a node in the value table.

    Returns:
        str: Canonical identity.

    Raises:
        ValueError: If ``value`` is not a string identity.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field} has a non-string identity {value!r}")
    if value not in mapping:
        mapping[value] = str(uuid.uuid5(namespace, f"external-{len(mapping)}"))
    return mapping[value]
