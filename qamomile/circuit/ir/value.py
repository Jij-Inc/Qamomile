import dataclasses
import typing
import uuid

from .types import ValueType

T = typing.TypeVar("T", bound=ValueType)


@dataclasses.dataclass
class Value(typing.Generic[T]):
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
    parent_array: "ArrayValue | None" = None
    element_indices: tuple["Value", ...] = ()

    def next_version(self) -> "Value[T]":
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

    def get_lowered_qubits(self) -> list["Value"] | None:
        """Get lowered qubit list if available, otherwise None."""
        return self.params.get("lowered_qubits")  # type: ignore

    def set_lowered_qubits(self, qubits: list["Value"]) -> None:
        """Set lowered qubit list."""
        self.params["lowered_qubits"] = qubits  # type: ignore

    def get_lowered_bits(self) -> list["Value"] | None:
        """Get lowered bit list if available, otherwise None."""
        return self.params.get("lowered_bits")  # type: ignore

    def set_lowered_bits(self, bits: list["Value"]) -> None:
        """Set lowered bit list."""
        self.params["lowered_bits"] = bits  # type: ignore

    # Cast metadata accessors

    def is_cast_result(self) -> bool:
        """Check if this value is the result of a CastOperation."""
        return "cast_source_logical_id" in self.params or "cast_source_uuid" in self.params

    def get_cast_source_uuid(self) -> str | None:
        """Get the source value UUID if this is a cast result."""
        return self.params.get("cast_source_uuid")

    def get_cast_source_logical_id(self) -> str | None:
        """Get the source value logical_id if this is a cast result."""
        return self.params.get("cast_source_logical_id")

    def get_cast_qubit_uuids(self) -> list[str] | None:
        """Get the underlying qubit UUIDs for this cast value.

        For cast results, this returns the list of qubit UUIDs that
        the cast target references (the same physical qubits as the source).
        """
        return self.params.get("cast_qubit_uuids")

    def get_cast_qubit_logical_ids(self) -> list[str] | None:
        """Get the underlying qubit logical_ids for this cast value.

        For cast results, this returns the list of qubit logical_ids that
        the cast target references (the same physical qubits as the source).
        """
        return self.params.get("cast_qubit_logical_ids")

    def set_cast_metadata(
        self,
        source_uuid: str,
        qubit_uuids: list[str],
        source_logical_id: str | None = None,
        qubit_logical_ids: list[str] | None = None,
    ) -> None:
        """Set cast metadata for this value.

        Args:
            source_uuid: UUID of the source value being cast from
            qubit_uuids: List of underlying qubit UUIDs
            source_logical_id: Logical ID of the source value (physical identity)
            qubit_logical_ids: List of underlying qubit logical IDs
        """
        self.params["cast_source_uuid"] = source_uuid
        self.params["cast_qubit_uuids"] = qubit_uuids
        if source_logical_id is not None:
            self.params["cast_source_logical_id"] = source_logical_id
        if qubit_logical_ids is not None:
            self.params["cast_qubit_logical_ids"] = qubit_logical_ids


@dataclasses.dataclass
class ArrayValue(Value[T]):
    type: T
    name: str
    uuid: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    logical_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    shape: tuple[Value, ...] = dataclasses.field(default_factory=lambda: tuple([]))
    # Override params with Any type to support complex values (arrays, lists, etc.)
    params: dict[str, typing.Any] = dataclasses.field(default_factory=dict)

    def next_version(self) -> "ArrayValue[T]":
        """Create a new ArrayValue with incremented version, preserving shape."""
        return dataclasses.replace(
            self,
            version=self.version + 1,
            params=self.params.copy(),
            uuid=str(uuid.uuid4()),  # New uuid for each version
            # logical_id stays the same (inherited from dataclasses.replace)
        )
