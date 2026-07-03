"""Container types for qkernel: Tuple and Dict handles."""

from __future__ import annotations

import dataclasses
from typing import Generic, Iterator, Literal, TypeVar, overload

from qamomile.circuit.ir.types.primitives import UIntType, ValueType
from qamomile.circuit.ir.value import DictValue, TupleValue, Value

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

    value: TupleValue  # type: ignore[assignment]  # intentional narrowing from Value
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
    for Ising coefficients like {(i, j): Jij}. Supports iteration via
    ``items()`` and subscript lookup (``d[key]``), including indexing
    one dict with the iteration keys of another.

    Example:
        ```python
        @qmc.qkernel
        def ising_cost(
            q: qmc.Vector[qmc.Qubit],
            ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            gammas: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
        ) -> qmc.Vector[qmc.Qubit]:
            for (i, j), Jij in qmc.items(ising):
                q[i], q[j] = qmc.rzz(q[i], q[j], Jij * gammas[(i, j)])
            return q
        ```
    """

    value: DictValue  # type: ignore[assignment]  # intentional narrowing from Value
    _entries: list[tuple[Handle, Handle]] = dataclasses.field(default_factory=list)
    _size: UInt | None = None
    _key_type: type | None = None
    _value_type: type | None = None

    def items(self) -> DictItemsIterator[K, V]:
        """Return an iterator over (key, value) pairs."""
        return DictItemsIterator(dict_handle=self)

    def _result_spec(self) -> "tuple[ValueType, type, type]":
        """Resolve the (IR type, handle class, Python coercer) for values.

        Reads ``_value_type`` (recorded from the ``Dict[K, V]``
        annotation at input-handle creation) and maps the scalar handle
        types to their IR type and Python coercer. ``None`` falls back
        to ``Float`` for legacy handles created without the annotation.

        Returns:
            tuple[ValueType, type, type]: ``(ir_type, handle_class,
                python_coercer)`` for the dict's value type.

        Raises:
            NotImplementedError: If the value type is a non-scalar
                (container) type, which subscript lookup does not yet
                support.
        """
        from qamomile.circuit.ir.types.primitives import (
            BitType,
            FloatType,
            UIntType,
        )

        from .primitives import Bit, Float

        if self._value_type is None or self._value_type is Float:
            return FloatType(), Float, float
        if self._value_type is UInt:
            return UIntType(), UInt, int
        if self._value_type is Bit:
            return BitType(), Bit, bool
        raise NotImplementedError(
            f"Dict subscript lookup supports scalar value types "
            f"(Float / UInt / Bit); got {self._value_type!r}. Container "
            f"values (Tuple / Vector) are not yet supported."
        )

    def __getitem__(self, key) -> V:
        """Look up a dict entry by key (``d[key]``).

        Key components may be ``UInt`` handles (including symbolic loop
        variables of a for-items loop), or arbitrary hashable Python
        constants. A tuple key (``d[(i, j)]``) is matched
        component-wise.

        When every key component is a compile-time constant and the
        dict carries bound data, the lookup folds eagerly to a constant
        handle — any hashable Python key works on this path. When a
        component is symbolic (known only at emit time), a
        :class:`DictGetItemOperation` is traced instead; symbolic
        components must be ``UInt`` because they need an IR ``Value``
        representation.

        Args:
            key: A single key component or a tuple of components. Each
                component is a ``UInt`` handle or a hashable Python
                constant (``int``, ``str``, ``float``, ...).

        Returns:
            V: A handle for the looked-up value. The handle type
                follows the dict's declared value type (``Float`` /
                ``UInt`` / ``Bit``).

        Raises:
            TypeError: If a symbolic key component is not a ``UInt``
                handle, or a constant component is unhashable.
            KeyError: If the key components are constant, bound data is
                available, and the key is not present.
            NotImplementedError: If the dict's value type is a
                container type, or a non-``int`` constant key is mixed
                with symbolic components (no IR representation).
        """
        from qamomile.circuit.frontend.tracer import get_current_tracer
        from qamomile.circuit.ir.operation.classical_ops import (
            DictGetItemOperation,
        )

        components = list(key) if isinstance(key, tuple) else [key]

        const_components: list[object] = []
        all_const = True
        for component in components:
            if isinstance(component, Handle):
                if not isinstance(component, UInt):
                    raise TypeError(
                        f"Symbolic dict key components must be UInt "
                        f"handles, got {type(component).__name__}"
                    )
                const = component.value.get_const()
                if const is None:
                    all_const = False
                else:
                    const_components.append(int(const))
            else:
                hash(component)  # raises TypeError for unhashable keys
                const_components.append(component)

        ir_type, handle_class, coerce = self._result_spec()
        result_value = Value(type=ir_type, name=f"{self.value.name}_item")

        # Eager fold: constant key + bound data -> constant handle.
        if all_const:
            bound_items = self.value.get_bound_data_items()
            if bound_items:
                lookup_key = (
                    tuple(const_components)
                    if len(const_components) > 1
                    else const_components[0]
                )
                table = {
                    (tuple(k) if isinstance(k, list) else k): v for k, v in bound_items
                }
                if lookup_key not in table:
                    raise KeyError(
                        f"Key {lookup_key!r} not found in dict '{self.value.name}'"
                    )
                folded = coerce(table[lookup_key])
                return handle_class(  # type: ignore[return-value]
                    value=result_value.with_const(folded),
                    init_value=folded,
                )

        # Symbolic path: every component needs an IR Value. UInt
        # handles carry their own Value; int constants are lifted to
        # constant UInt Values. Other constant types have no IR scalar
        # representation yet, so they only work on the eager-fold path.
        key_values: list[Value] = []
        for component in components:
            if isinstance(component, UInt):
                key_values.append(component.value)
            elif isinstance(component, int):
                key_values.append(
                    Value(type=UIntType(), name="dict_key").with_const(component)
                )
            else:
                raise NotImplementedError(
                    f"Dict key component {component!r} "
                    f"({type(component).__name__}) cannot be traced "
                    f"symbolically; non-int constant keys require the "
                    f"dict data to be bound at trace time (pass it via "
                    f"bindings)."
                )

        op = DictGetItemOperation(
            operands=[self.value] + key_values,  # type: ignore[list-item]
            results=[result_value],
            key_arity=len(components),
        )
        get_current_tracer().add_operation(op)
        return handle_class(value=result_value, init_value=coerce(0))  # type: ignore[return-value]

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
                ).with_const(len(self._entries))
            )
        return self._size
