"""Value types for the Qamomile IR.

This module defines the core value types used in the intermediate representation:
- ValueBase: Protocol for all value types (enables isinstance checks)
- Value[T]: Generic typed value with SSA versioning
- ArrayValue[T]: Array of values with shape information
- TupleValue: Tuple of values (for structured data like Ising indices)
- DictValue: Dictionary of key-value pairs (for Ising coefficients)
"""

from __future__ import annotations

import dataclasses
import typing
import uuid

from .types import DictType, ValueType

T = typing.TypeVar("T", bound=ValueType)


# =============================================================================
# ValueBase Protocol - Common interface for all value types
# =============================================================================


@typing.runtime_checkable
class ValueBase(typing.Protocol):
    """Protocol defining the common interface for all value types.

    This protocol enables:
    - isinstance(v, ValueBase) checks for any value type
    - Unified type annotations with ValueBase instead of union types
    - Duck typing for value operations in transpiler passes

    All value types (Value, ArrayValue, TupleValue, DictValue) implement this protocol.
    """

    uuid: str
    """Unique identifier for this specific value instance."""

    logical_id: str
    """Identifies the same physical qubit/value across SSA versions."""

    name: str
    """Human-readable name for this value."""

    params: dict[str, typing.Any]
    """Flexible parameter storage for metadata (const, parameter, etc.)."""

    def next_version(self) -> ValueBase:
        """Create a new SSA version with fresh uuid but same logical_id."""
        ...

    def is_parameter(self) -> bool:
        """Check if this value is an unbound parameter."""
        ...

    def parameter_name(self) -> str | None:
        """Get the parameter name if this is a parameter."""
        ...

    def is_constant(self) -> bool:
        """Check if this value is a constant."""
        ...


# Type alias for all value types (explicit union for when needed)
ValueLike: typing.TypeAlias = "Value | ArrayValue | TupleValue | DictValue"
"""Type alias for any value type. Prefer using ValueBase for type hints."""


# =============================================================================
# Value - Core typed value with SSA versioning
# =============================================================================


@dataclasses.dataclass
class Value(typing.Generic[T]):
    """A typed value in the IR with SSA-style versioning.

    Value represents a single typed value (qubit, float, int, bit, etc.)
    with support for:
    - SSA versioning via uuid/logical_id
    - Parameter binding
    - Constant folding
    - Array element tracking
    """

    type: T
    name: str
    version: int = 0
    # params can contain: "const" (int|float), "parameter" (str),
    # "lowered_qubits" (list[Value]), "lowered_bits" (list[Value]), "const_array" (list)
    params: dict[str, typing.Any] = dataclasses.field(default_factory=dict)
    uuid: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    # logical_id identifies the same physical qubit across SSA versions
    logical_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    # For array elements: reference to parent array and indices
    parent_array: ArrayValue | None = None
    element_indices: tuple[Value, ...] = ()

    def next_version(self) -> Value[T]:
        """Create a new Value with incremented version (SSA style)."""
        return Value(
            type=self.type,
            name=self.name,
            version=self.version + 1,
            params=self.params.copy(),
            uuid=str(uuid.uuid4()),  # New uuid for each version
            logical_id=self.logical_id,  # Same logical_id for physical identity
            parent_array=self.parent_array,
            element_indices=self.element_indices,
        )

    def is_array_element(self) -> bool:
        """Check if this value is an element of an array."""
        return self.parent_array is not None

    def is_parameter(self) -> bool:
        """Check if this value is an unbound parameter."""
        return "parameter" in self.params

    def parameter_name(self) -> str | None:
        """Get the parameter name if this is a parameter, otherwise None."""
        return self.params.get("parameter")  # type: ignore

    def is_constant(self) -> bool:
        """Check if this value is a constant."""
        return "const" in self.params

    # Accessor methods for common params

    def get_const(self) -> int | float | None:
        """Get constant value if available, otherwise None."""
        return self.params.get("const")

    def get_lowered_qubits(self) -> list[Value] | None:
        """Get lowered qubit list if available, otherwise None."""
        return self.params.get("lowered_qubits")  # type: ignore

    def set_lowered_qubits(self, qubits: list[Value]) -> None:
        """Set lowered qubit list."""
        self.params["lowered_qubits"] = qubits  # type: ignore

    def get_lowered_bits(self) -> list[Value] | None:
        """Get lowered bit list if available, otherwise None."""
        return self.params.get("lowered_bits")  # type: ignore

    def set_lowered_bits(self, bits: list[Value]) -> None:
        """Set lowered bit list."""
        self.params["lowered_bits"] = bits  # type: ignore

    # Cast metadata accessors

    def is_cast_result(self) -> bool:
        """Check if this value is the result of a CastOperation."""
        return (
            "cast_source_logical_id" in self.params or "cast_source_uuid" in self.params
        )

    def get_cast_source_uuid(self) -> str | None:
        """Get the source value UUID if this is a cast result."""
        return self.params.get("cast_source_uuid")

    def get_cast_source_logical_id(self) -> str | None:
        """Get the source value logical_id if this is a cast result."""
        return self.params.get("cast_source_logical_id")

    def get_cast_qubit_uuids(self) -> list[str] | None:
        """Get the underlying qubit UUIDs for this cast value."""
        return self.params.get("cast_qubit_uuids")

    def get_cast_qubit_logical_ids(self) -> list[str] | None:
        """Get the underlying qubit logical_ids for this cast value."""
        return self.params.get("cast_qubit_logical_ids")

    def set_cast_metadata(
        self,
        source_uuid: str,
        qubit_uuids: list[str],
        source_logical_id: str | None = None,
        qubit_logical_ids: list[str] | None = None,
    ) -> None:
        """Set cast metadata for this value."""
        self.params["cast_source_uuid"] = source_uuid
        self.params["cast_qubit_uuids"] = qubit_uuids
        if source_logical_id is not None:
            self.params["cast_source_logical_id"] = source_logical_id
        if qubit_logical_ids is not None:
            self.params["cast_qubit_logical_ids"] = qubit_logical_ids


# =============================================================================
# ArrayValue - Array of values with shape
# =============================================================================


@dataclasses.dataclass
class ArrayValue(Value[T]):
    """An array of values with shape information.

    ArrayValue extends Value to represent multi-dimensional arrays
    of typed values (e.g., qubit registers, parameter vectors).
    """

    type: T
    name: str
    uuid: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    logical_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    shape: tuple[Value, ...] = dataclasses.field(default_factory=tuple)
    # Override params with Any type to support complex values (arrays, lists, etc.)
    params: dict[str, typing.Any] = dataclasses.field(default_factory=dict)

    def next_version(self) -> ArrayValue[T]:
        """Create a new ArrayValue with incremented version, preserving shape."""
        return ArrayValue(
            type=self.type,
            name=self.name,
            version=self.version + 1,
            params=self.params.copy(),
            uuid=str(uuid.uuid4()),
            logical_id=self.logical_id,
            shape=self.shape,
        )


# =============================================================================
# TupleValue - Tuple of values for structured data
# =============================================================================


@dataclasses.dataclass
class TupleValue:
    """A tuple of values for structured data.

    Used for structured data like Ising model indices (i, j).
    Implements the ValueBase protocol for unified handling.
    """

    name: str
    elements: tuple[Value, ...] = dataclasses.field(default_factory=tuple)
    params: dict[str, typing.Any] = dataclasses.field(default_factory=dict)
    uuid: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    logical_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))

    def next_version(self) -> TupleValue:
        """Create a new TupleValue with a fresh uuid but same logical_id."""
        return TupleValue(
            name=self.name,
            elements=self.elements,
            params=self.params.copy(),
            uuid=str(uuid.uuid4()),
            logical_id=self.logical_id,
        )

    def is_parameter(self) -> bool:
        """Check if this tuple is a parameter (has symbolic elements)."""
        return "parameter" in self.params

    def parameter_name(self) -> str | None:
        """Get the parameter name if this is a parameter."""
        return self.params.get("parameter")

    def is_constant(self) -> bool:
        """Check if all elements are constants."""
        return all(isinstance(e, Value) and e.is_constant() for e in self.elements)


# =============================================================================
# DictValue - Dictionary of key-value pairs
# =============================================================================


@dataclasses.dataclass
class DictValue:
    """A dictionary mapping keys to values.

    Used for structured data like Ising coefficients {(i, j): Jij}.
    Entries are stored as a list of (key, value) pairs for consistent ordering.
    Implements the ValueBase protocol for unified handling.
    """

    name: str
    entries: list[tuple[TupleValue | Value, Value]] = dataclasses.field(
        default_factory=list
    )
    params: dict[str, typing.Any] = dataclasses.field(default_factory=dict)
    uuid: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    logical_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))

    def next_version(self) -> DictValue:
        """Create a new DictValue with a fresh uuid but same logical_id."""
        return DictValue(
            name=self.name,
            entries=self.entries,
            params=self.params.copy(),
            uuid=str(uuid.uuid4()),
            logical_id=self.logical_id,
        )

    @property
    def type(self) -> DictType:
        """Return type interface for compatibility with Value."""
        return DictType()

    def is_parameter(self) -> bool:
        """Check if this dict is a parameter (bound at transpile time)."""
        return "parameter" in self.params

    def parameter_name(self) -> str | None:
        """Get the parameter name if this is a parameter."""
        return self.params.get("parameter")

    def is_constant(self) -> bool:
        """Check if all entries have constant values."""
        return all(v.is_constant() for _, v in self.entries)

    def __len__(self) -> int:
        """Return the number of entries."""
        return len(self.entries)
