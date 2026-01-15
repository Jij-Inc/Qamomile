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
    # For array elements: reference to parent array and indices
    parent_array: "ArrayValue | None" = None
    element_indices: tuple["Value", ...] = ()

    def next_version(self) -> "Value[T]":
        return Value(
            type=self.type,
            name=self.name,
            version=self.version + 1,
            params=self.params.copy(),
            uuid=self.uuid,
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


@dataclasses.dataclass
class ArrayValue(Value[T]):
    type: T
    name: str
    uuid: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    shape: tuple[Value, ...] = dataclasses.field(default_factory=lambda: tuple([]))
    # Override params with Any type to support complex values (arrays, lists, etc.)
    params: dict[str, typing.Any] = dataclasses.field(default_factory=dict)
