import abc
from dataclasses import dataclass


class QuantumTypeMixin:
    def is_quantum(self) -> bool:
        return True


class ClassicalTypeMixin:
    def is_classical(self) -> bool:
        return True


class ObjectTypeMixin:
    def is_object(self) -> bool:
        return True


class ValueType(abc.ABC):
    """Base class for all value types in the IR.

    Type instances are compared by class - all instances of the same
    type class are considered equal. This allows using type instances
    as dictionary keys where all QubitType() instances match.
    """

    def label(self) -> str:
        return self.__class__.__name__

    def is_classical(self) -> bool:
        return False

    def is_quantum(self) -> bool:
        return False

    def is_object(self) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        """Two type instances are equal if they are the same class."""
        if not isinstance(other, ValueType):
            return NotImplemented
        return self.__class__ is other.__class__

    def __hash__(self) -> int:
        """Hash based on class, so all instances of same type hash equally."""
        return hash(self.__class__)


class QubitType(QuantumTypeMixin, ValueType):
    """Type representing a quantum bit (qubit)."""

    pass


class UIntType(ClassicalTypeMixin, ValueType):
    """Type representing an unsigned integer."""

    pass


class FloatType(ClassicalTypeMixin, ValueType):
    """Type representing a floating-point number."""

    pass


class BitType(ClassicalTypeMixin, ValueType):
    """Type representing a classical bit."""

    pass


class BlockType(ObjectTypeMixin, ValueType):
    """Type representing a block/function reference."""

    pass


@dataclass
class TupleType(ClassicalTypeMixin, ValueType):
    """Type representing a tuple of values.

    Unlike simple types, TupleType stores the types of its elements,
    so equality and hashing depend on the element types.
    """

    element_types: tuple["ValueType", ...]

    def label(self) -> str:
        element_labels = ", ".join(t.label() for t in self.element_types)
        return f"Tuple[{element_labels}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TupleType):
            return NotImplemented
        return self.element_types == other.element_types

    def __hash__(self) -> int:
        return hash((self.__class__, self.element_types))


@dataclass
class DictType(ClassicalTypeMixin, ValueType):
    """Type representing a dictionary mapping keys to values.

    Unlike simple types, DictType stores the key and value types,
    so equality and hashing depend on those types.
    """

    key_type: "ValueType"
    value_type: "ValueType"

    def label(self) -> str:
        return f"Dict[{self.key_type.label()}, {self.value_type.label()}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DictType):
            return NotImplemented
        return self.key_type == other.key_type and self.value_type == other.value_type

    def __hash__(self) -> int:
        return hash((self.__class__, self.key_type, self.value_type))
