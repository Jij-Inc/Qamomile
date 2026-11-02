"""Host layouts connecting fixed dictionary keys to ordinary Param vectors.

These helpers prepare portable data. The native compiler still validates the
Const Map and the parameter ABI still validates each packed Vector value.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Generic, TypeVar

from ._native import compiler as native

_Key = TypeVar("_Key", bound=Hashable)
_Value = TypeVar("_Value")


@dataclass(frozen=True, slots=True)
class ParameterMap(Generic[_Key]):
    """Fix dictionary key order for a Const index map and Param value vector.

    Args:
        keys (tuple[Hashable, ...]): Nonempty unique immutable keys in vector
            order. They must also match the kernel's declared Const Map key type.

    Attributes:
        keys (tuple[Hashable, ...]): Fixed vector order, independent of the order
            of subsequently supplied dictionaries.

    Raises:
        TypeError: If keys are not a tuple of hashable values.
        ValueError: If keys are empty, duplicated, or exceed the node budget.
    """

    keys: tuple[_Key, ...]

    def __post_init__(self) -> None:
        """Check the fixed host layout before allocating an index mapping.

        Raises:
            TypeError: If keys are not a tuple of hashable values.
            ValueError: If the layout is empty, duplicated, or over budget.
        """
        if not isinstance(self.keys, tuple):
            raise TypeError("keys must be a tuple")
        if not self.keys or len(self.keys) > native.MAX_QCORE_MESSAGE_NODES:
            raise ValueError("key layout must be nonempty and fit the node budget")
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("keys must be unique")

    @property
    def indices(self) -> Mapping[_Key, int]:
        """Return the immutable key-to-position map to bind as Const data.

        Returns:
            Mapping[Hashable, int]: Positions following the declared key order.
        """
        return MappingProxyType({key: index for index, key in enumerate(self.keys)})

    def pack(self, values: Mapping[_Key, _Value]) -> tuple[_Value, ...]:
        """Pack exactly one value for every fixed key into Param vector order.

        Args:
            values (Mapping[Hashable, object]): Dictionary with precisely the
                fixed keys. Its insertion order is irrelevant.

        Returns:
            tuple[object, ...]: Values to pass as an ordinary Param Vector.

        Raises:
            TypeError: If values is not a mapping.
            ValueError: If keys are missing or extra.
        """
        if not isinstance(values, Mapping):
            raise TypeError("values must be a mapping")
        if len(values) != len(self.keys) or any(key not in values for key in self.keys):
            raise ValueError("values must contain exactly the fixed keys")
        return tuple(values[key] for key in self.keys)


__all__ = ["ParameterMap"]
