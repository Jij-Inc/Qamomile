"""Persist ordered execution structure without SDK objects or callables."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qamomile.circuit.transpiler.execution_handle import (
        ExecutionHandle,
        ExecutionReference,
    )

_MAX_NESTING = 100


class ExecutionSnapshotKind(StrEnum):
    """Identify the reconstruction contract of an execution snapshot node."""

    REMOTE = "remote"
    LOCAL = "local"
    COMPOSITE = "composite"


class _ValueKind(StrEnum):
    """Identify native scalar and container types in the local value codec."""

    NONE = "none"
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STR = "str"
    LIST = "list"
    TUPLE = "tuple"
    DICT = "dict"


_SCALAR_TYPES = {
    _ValueKind.NONE: type(None),
    _ValueKind.BOOL: bool,
    _ValueKind.INT: int,
    _ValueKind.FLOAT: float,
    _ValueKind.STR: str,
}


def _validate_fields(
    data: Any, required: set[str], optional: set[str], path: str
) -> Mapping[str, Any]:
    """Check a serialized mapping before interpreting its fields.

    Args:
        data (Any): Candidate serialized mapping.
        required (set[str]): Fields that must be present.
        optional (set[str]): Additional permitted fields.
        path (str): Diagnostic location within the snapshot.

    Returns:
        Mapping[str, Any]: Validated field mapping.

    Raises:
        TypeError: If the value is not a mapping with string keys.
        ValueError: If fields are missing or unknown.
    """
    if not isinstance(data, Mapping) or any(type(key) is not str for key in data):
        raise TypeError(f"{path} must be a mapping with string keys")
    missing = required - data.keys()
    unknown = data.keys() - required - optional
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{path} has unknown fields: {', '.join(sorted(unknown))}")
    return data


def _encode_value(value: Any, path: str = "value", depth: int = 0) -> dict[str, Any]:
    """Encode supported native values with explicit container and scalar tags.

    Args:
        value (Any): Native scalar or recursively supported container.
        path (str): Diagnostic location. Defaults to ``value``.
        depth (int): Current nesting depth. Defaults to zero.

    Returns:
        dict[str, Any]: Tagged JSON-compatible value.

    Raises:
        TypeError: If a value or mapping key has an unsupported type.
        ValueError: If a float is non-finite or nesting exceeds the safe limit.
    """
    if depth > _MAX_NESTING:
        raise ValueError(f"{path} exceeds supported nesting or contains a cycle")
    value_type = type(value)
    for tag, scalar_type in _SCALAR_TYPES.items():
        if value_type is scalar_type:
            if scalar_type is float and not math.isfinite(value):
                raise ValueError(f"{path} must contain only finite floats")
            return {"type": tag.value, "value": value}
    if value_type in (list, tuple):
        return {
            "type": (_ValueKind.LIST if value_type is list else _ValueKind.TUPLE).value,
            "items": [
                _encode_value(item, f"{path}[{index}]", depth + 1)
                for index, item in enumerate(value)
            ],
        }
    if value_type is dict:
        if any(type(key) is not str for key in value):
            raise TypeError(f"{path} must contain only string dictionary keys")
        return {
            "type": _ValueKind.DICT.value,
            "items": {
                key: _encode_value(item, f"{path}[{key!r}]", depth + 1)
                for key, item in value.items()
            },
        }
    raise TypeError(
        f"{path} has unsupported local result type {value_type.__name__}; "
        "supported values are None, bool, int, finite float, str, list, tuple, "
        "and dictionaries with string keys"
    )


def _decode_value(data: Any, path: str = "value", depth: int = 0) -> Any:
    """Decode tagged values without executing user-defined deserializers.

    Args:
        data (Any): Tagged JSON value.
        path (str): Diagnostic location. Defaults to ``value``.
        depth (int): Current nesting depth. Defaults to zero.

    Returns:
        Any: Detached native result with its original container types.

    Raises:
        TypeError: If serialized fields have incompatible types.
        ValueError: If tags, fields, floats, or nesting are invalid.
    """
    if depth > _MAX_NESTING:
        raise ValueError(f"{path} exceeds supported nesting or contains a cycle")
    data = _validate_fields(data, {"type"}, {"value", "items"}, path)
    raw_tag = data["type"]
    if type(raw_tag) is not str:
        raise TypeError(f"{path}.type must be a string")
    try:
        tag = _ValueKind(raw_tag)
    except ValueError as error:
        raise ValueError(f"{path}.type has unsupported value {raw_tag!r}") from error
    if tag in _SCALAR_TYPES:
        _validate_fields(data, {"type", "value"}, set(), path)
        value = data["value"]
        if type(value) is not _SCALAR_TYPES[tag]:
            raise TypeError(f"{path}.value does not match its {tag!r} type tag")
        if tag is _ValueKind.FLOAT and not math.isfinite(value):
            raise ValueError(f"{path}.value must be a finite float")
        return value
    _validate_fields(data, {"type", "items"}, set(), path)
    items = data["items"]
    if tag is _ValueKind.DICT:
        if not isinstance(items, Mapping) or any(type(key) is not str for key in items):
            raise TypeError(f"{path}.items must be a mapping with string keys")
        return {
            key: _decode_value(item, f"{path}[{key!r}]", depth + 1)
            for key, item in items.items()
        }
    if type(items) is not list:
        raise TypeError(f"{path}.items must be a list")
    values = [
        _decode_value(item, f"{path}[{index}]", depth + 1)
        for index, item in enumerate(items)
    ]
    return tuple(values) if tag is _ValueKind.TUPLE else values


def _reference_from_dict(data: Any) -> ExecutionReference:
    """Validate and detach provider-reference fields.

    Args:
        data (Any): Serialized provider reference.

    Returns:
        ExecutionReference: Validated provider identifiers.

    Raises:
        TypeError: If identifiers or context have incompatible types.
        ValueError: If fields are missing, unknown, or invalid.
    """
    from qamomile.circuit.transpiler.execution_handle import ExecutionReference

    data = _validate_fields(
        data, {"provider", "job_ids"}, {"target", "group_id", "context"}, "reference"
    )
    return ExecutionReference.from_dict(data)


@dataclasses.dataclass(frozen=True)
class ExecutionSnapshot:
    """Store a remote leaf, a local value, or an ordered execution group.

    Provider leaves may identify several physical jobs or produce native batch
    results. Composite children retain their result boundaries independently of
    the number of provider identifiers. Local values contain raw engine-neutral
    results, before the executable applies its public result conversion. Trees
    and local values support at most 100 levels of nesting.

    Args:
        kind (str | ExecutionSnapshotKind): One of ``remote``, ``local``, or
            ``composite``, normalized to an enum member.
        reference (ExecutionReference | None): Required only for remote leaves.
        value (Any): Supported native result for local leaves. Defaults to None.
        children (tuple[ExecutionSnapshot, ...]): Ordered composite children.
            Defaults to an empty tuple.

    Raises:
        TypeError: If fields or local result types are unsupported.
        ValueError: If fields conflict with the node kind or values are invalid.
    """

    kind: str | ExecutionSnapshotKind
    reference: ExecutionReference | None = None
    value: Any = None
    children: tuple[ExecutionSnapshot, ...] = ()

    def __post_init__(self) -> None:
        """Validate node shape and detach mutable local data.

        Raises:
            TypeError: If fields or local result types are unsupported.
            ValueError: If node fields conflict or a local value is invalid.
        """
        from qamomile.circuit.transpiler.execution_handle import ExecutionReference

        if not isinstance(self.kind, (str, ExecutionSnapshotKind)):
            raise TypeError("ExecutionSnapshot.kind must be a string")
        try:
            kind = ExecutionSnapshotKind(self.kind)
        except ValueError as error:
            raise ValueError(
                f"Unsupported ExecutionSnapshot.kind: {self.kind!r}; "
                "expected remote, local, or composite"
            ) from error
        object.__setattr__(self, "kind", kind)
        if not isinstance(self.children, (list, tuple)) or not all(
            isinstance(child, ExecutionSnapshot) for child in self.children
        ):
            raise TypeError(
                "ExecutionSnapshot.children must contain ExecutionSnapshot values"
            )
        object.__setattr__(self, "children", tuple(self.children))
        if kind is ExecutionSnapshotKind.REMOTE:
            if not isinstance(self.reference, ExecutionReference):
                raise TypeError(
                    "Remote ExecutionSnapshot requires an ExecutionReference"
                )
            if self.value is not None or self.children:
                raise ValueError(
                    "Remote ExecutionSnapshot must not contain value or children"
                )
            object.__setattr__(
                self, "reference", _reference_from_dict(self.reference.to_dict())
            )
        elif kind is ExecutionSnapshotKind.LOCAL:
            if self.reference is not None or self.children:
                raise ValueError(
                    "Local ExecutionSnapshot must not contain reference or children"
                )
            object.__setattr__(self, "value", _decode_value(_encode_value(self.value)))
        elif self.reference is not None or self.value is not None:
            raise ValueError(
                "Composite ExecutionSnapshot must not contain reference or value"
            )
        self._validate_tree_depth()

    def _validate_tree_depth(self) -> None:
        """Reject trees that cannot pass the bounded JSON decoder.

        Raises:
            ValueError: If the tree exceeds the supported nesting limit.
        """
        pending: list[tuple[ExecutionSnapshot, int]] = [(self, 0)]
        while pending:
            node, depth = pending.pop()
            if depth > _MAX_NESTING:
                raise ValueError(
                    "ExecutionSnapshot exceeds supported nesting or contains a cycle"
                )
            pending.extend((child, depth + 1) for child in node.children)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the execution tree and type-preserving local values.

        Returns:
            dict[str, Any]: JSON-compatible execution tree.

        Raises:
            TypeError: If mutable local data was changed to unsupported types.
            ValueError: If mutable local data or references became invalid.
        """
        if self.kind is ExecutionSnapshotKind.REMOTE:
            assert self.reference is not None
            reference = _reference_from_dict(self.reference.to_dict())
            return {"kind": self.kind.value, "reference": reference.to_dict()}
        if self.kind is ExecutionSnapshotKind.LOCAL:
            return {"kind": self.kind.value, "value": _encode_value(self.value)}
        return {
            "kind": ExecutionSnapshotKind.COMPOSITE.value,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExecutionSnapshot:
        """Reconstruct an execution tree with strict node and value validation.

        Args:
            data (Mapping[str, Any]): Mapping produced by :meth:`to_dict`.

        Returns:
            ExecutionSnapshot: Validated execution structure.

        Raises:
            TypeError: If node fields have incompatible types.
            ValueError: If kinds, fields, references, or local values are invalid.
        """
        return cls._from_dict(data, 0)

    @classmethod
    def _from_dict(cls, data: Any, depth: int) -> ExecutionSnapshot:
        """Decode a node while bounding untrusted tree nesting.

        Args:
            data (Any): Candidate serialized node.
            depth (int): Current node depth.

        Returns:
            ExecutionSnapshot: Validated execution subtree.

        Raises:
            TypeError: If serialized fields have incompatible types.
            ValueError: If fields or nesting are invalid.
        """
        if depth > _MAX_NESTING:
            raise ValueError(
                "ExecutionSnapshot exceeds supported nesting or contains a cycle"
            )
        data = _validate_fields(
            data, {"kind"}, {"reference", "value", "children"}, "execution"
        )
        raw_kind = data["kind"]
        if type(raw_kind) is not str:
            raise TypeError("ExecutionSnapshot.kind must be a string")
        try:
            kind = ExecutionSnapshotKind(raw_kind)
        except ValueError as error:
            raise ValueError(
                f"Unsupported ExecutionSnapshot.kind: {raw_kind!r}; "
                "expected remote, local, or composite"
            ) from error
        if kind is ExecutionSnapshotKind.REMOTE:
            _validate_fields(data, {"kind", "reference"}, set(), "remote execution")
            return cls(kind, reference=_reference_from_dict(data["reference"]))
        if kind is ExecutionSnapshotKind.LOCAL:
            _validate_fields(data, {"kind", "value"}, set(), "local execution")
            return cls(kind, value=_decode_value(data["value"]))
        _validate_fields(data, {"kind", "children"}, set(), "composite execution")
        if type(data["children"]) is not list:
            raise TypeError("Composite ExecutionSnapshot.children must be a list")
        return cls(
            kind,
            children=tuple(
                cls._from_dict(child, depth + 1) for child in data["children"]
            ),
        )

    def references(self) -> tuple[ExecutionReference, ...]:
        """Collect provider leaves in order without discarding tree structure.

        This list supports diagnostics; restoration uses the complete tree.

        Returns:
            tuple[ExecutionReference, ...]: Detached remote references in order.

        Raises:
            TypeError: If mutable reference fields became incompatible.
            ValueError: If mutable reference fields became invalid.
        """
        if self.kind is ExecutionSnapshotKind.REMOTE:
            assert self.reference is not None
            return (_reference_from_dict(self.reference.to_dict()),)
        return tuple(
            reference for child in self.children for reference in child.references()
        )

    def restore(
        self, restore_reference: Callable[[ExecutionReference], ExecutionHandle[Any]]
    ) -> ExecutionHandle[Any]:
        """Reattach remote leaves and rebuild local values and ordered groups.

        The callback must reattach an existing provider execution. This method
        neither retrieves remote results nor submits any execution.

        Args:
            restore_reference (Callable[[ExecutionReference], ExecutionHandle[Any]]):
                Provider-specific callback for one complete remote leaf.

        Returns:
            ExecutionHandle[Any]: Reconstructed raw execution lifecycle.

        Raises:
            TypeError: If local data is unsupported or the callback returns an
                incompatible handle.
            ValueError: If local values or references became invalid.
            Exception: If the provider restoration callback fails.
        """
        from qamomile.circuit.transpiler.execution_handle import (
            CompletedExecutionHandle,
            CompositeExecutionHandle,
            ExecutionHandle,
        )

        if self.kind is ExecutionSnapshotKind.REMOTE:
            handle = restore_reference(self.references()[0])
            if not isinstance(handle, ExecutionHandle):
                raise TypeError("restore_reference must return an ExecutionHandle")
            return handle
        if self.kind is ExecutionSnapshotKind.LOCAL:
            return CompletedExecutionHandle(_decode_value(_encode_value(self.value)))
        return CompositeExecutionHandle(
            tuple(child.restore(restore_reference) for child in self.children)
        )
