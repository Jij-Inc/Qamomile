import abc


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
