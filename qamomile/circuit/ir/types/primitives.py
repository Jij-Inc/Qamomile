import abc
import dataclasses


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
    def label(self) -> str:
        return self.__class__.__name__

    def is_classical(self) -> bool:
        return False
    
    def is_quantum(self) -> bool:
        return False

    def is_object(self) -> bool:
        return False

class QubitType(ValueType, QuantumTypeMixin):
    pass


class UIntType(ValueType, ClassicalTypeMixin):
    pass


class FloatType(ValueType, ClassicalTypeMixin):
    pass


class BitType(ValueType, ClassicalTypeMixin):
    pass


class BlockType(ValueType, ObjectTypeMixin):
    pass
