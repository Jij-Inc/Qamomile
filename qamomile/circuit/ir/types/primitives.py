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
class TupleType(ValueType):
    """Type representing a tuple of values.

    Unlike simple types, TupleType stores the types of its elements,
    so equality and hashing depend on the element types.

    Quantum/classical classification is derived from element types:
    quantum if any element is quantum, classical if all are classical.
    """

    element_types: tuple["ValueType", ...]

    def is_quantum(self) -> bool:
        return any(t.is_quantum() for t in self.element_types)

    def is_classical(self) -> bool:
        return all(t.is_classical() for t in self.element_types)

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
class DictType(ValueType):
    """Type representing a dictionary mapping keys to values.

    Unlike simple types, DictType stores the key and value types,
    so equality and hashing depend on those types.
    When key_type and value_type are None, represents a generic Dict type.

    Quantum/classical classification is derived from key/value types.
    """

    key_type: "ValueType | None" = None
    value_type: "ValueType | None" = None

    def is_quantum(self) -> bool:
        return (self.key_type is not None and self.key_type.is_quantum()) or (
            self.value_type is not None and self.value_type.is_quantum()
        )

    def is_classical(self) -> bool:
        if self.key_type is None and self.value_type is None:
            return True
        return (self.key_type is None or self.key_type.is_classical()) and (
            self.value_type is None or self.value_type.is_classical()
        )

    def label(self) -> str:
        if self.key_type is None or self.value_type is None:
            return "Dict"
        return f"Dict[{self.key_type.label()}, {self.value_type.label()}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DictType):
            return NotImplemented
        return self.key_type == other.key_type and self.value_type == other.value_type

    def __hash__(self) -> int:
        return hash((self.__class__, self.key_type, self.value_type))
