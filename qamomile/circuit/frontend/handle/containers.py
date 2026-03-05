"""Container types for qkernel: Tuple and Dict handles."""

from __future__ import annotations

import dataclasses
from typing import Generic, TypeVar, Iterator, Literal, overload

from qamomile.circuit.ir.value import Value, TupleValue, DictValue
from qamomile.circuit.ir.types.primitives import UIntType

from .handle import Handle
from .primitives import UInt


# Type variables for generic containers
K = TypeVar("K", bound=Handle)
V = TypeVar("V", bound=Handle)


@dataclasses.dataclass
class Tuple(Handle, Generic[K, V]):
    """Tuple handle for qkernel functions.

    Represents a tuple of values, commonly used for multi-index keys
    like (i, j) in Ising models.

    Example:
        ```python
        @qmc.qkernel
        def my_kernel(idx: qmc.Tuple[qmc.UInt, qmc.UInt]) -> qmc.UInt:
            i, j = idx
            return i + j
        ```
    """

    value: TupleValue
    _elements: tuple[Handle, ...] = dataclasses.field(default_factory=tuple)

    @overload
    def __getitem__(self, index: Literal[0]) -> K: ...
    @overload
    def __getitem__(self, index: Literal[1]) -> V: ...

    def __getitem__(self, index: int) -> K | V:
        """Get element at the given index."""
        if not self._elements:
            raise IndexError("Tuple elements not initialized")
        return self._elements[index]  # type: ignore[return-value]

    def __iter__(self) -> Iterator[K | V]:
        """Iterate over tuple elements."""
        return iter(self._elements)  # type: ignore[arg-type]

    def __len__(self) -> int:
        """Return the number of elements."""
        return len(self._elements)


@dataclasses.dataclass
class DictItemsIterator(Generic[K, V]):
    """Iterator for Dict.items() that yields (key, value) pairs.

    This is used internally for iterating over Dict entries in qkernel.
    """

    dict_handle: "Dict[K, V]"
    _index: int = 0

    def __iter__(self) -> "DictItemsIterator[K, V]":
        return self

    def __next__(self) -> tuple[K, V]:
        if self._index >= len(self.dict_handle._entries):
            raise StopIteration
        key, value = self.dict_handle._entries[self._index]
        self._index += 1
        return key, value  # type: ignore[return-value]


@dataclasses.dataclass
class Dict(Handle, Generic[K, V]):
    """Dict handle for qkernel functions.

    Represents a dictionary mapping keys to values, commonly used
    for Ising coefficients like {(i, j): Jij}.

    Example:
        ```python
        @qmc.qkernel
        def ising_cost(
            q: qmc.Vector[qmc.Qubit],
            ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            gamma: qmc.Float,
        ) -> qmc.Vector[qmc.Qubit]:
            for (i, j), Jij in qmc.items(ising):
                q[i], q[j] = qmc.rzz(q[i], q[j], gamma * Jij)
            return q
        ```
    """

    value: DictValue
    _entries: list[tuple[Handle, Handle]] = dataclasses.field(default_factory=list)
    _size: UInt | None = None
    _key_type: type | None = None

    def items(self) -> DictItemsIterator[K, V]:
        """Return an iterator over (key, value) pairs."""
        return DictItemsIterator(dict_handle=self)

    def __len__(self) -> int:
        """Return the number of entries."""
        return len(self._entries)

    @property
    def size(self) -> UInt:
        """Return the number of entries as a UInt handle."""
        if self._size is None:
            self._size = UInt(
                value=Value(
                    type=UIntType(),
                    name=f"{self.value.name}_size",
                    params={"const": len(self._entries)} if self._entries else {},
                )
            )
        return self._size
