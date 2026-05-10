from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from .primitives import QuantumTypeMixin, UIntType, ValueType

if TYPE_CHECKING:
    # ``Value`` is referenced only in annotations below.  Keeping the import
    # behind ``TYPE_CHECKING`` breaks the circular dependency between
    # ``qamomile.circuit.ir.value`` (which imports ``.types``) and this
    # module (which lives under ``.types``).  See
    # ``qamomile/circuit/ir/value.py`` for the other side of the cycle.
    from qamomile.circuit.ir.value import Value


@dataclasses.dataclass
class QUIntType(QuantumTypeMixin, ValueType):
    """Quantum unsigned integer type.

    Represents a quantum register encoding an unsigned integer value
    using binary encoding (little-endian by default).

    Attributes:
        width: Number of qubits in the register. Can be symbolic.
    """

    width: int | Value[UIntType]

    def label(self) -> str:
        return f"QUInt[{self.width}]"


@dataclasses.dataclass
class QFixedType(QuantumTypeMixin, ValueType):
    """Quantum fixed-point type.

    Represents a quantum register encoding a fixed-point number
    with specified integer and fractional bits.

    Attributes:
        integer_bits: Number of bits for the integer part.
        fractional_bits: Number of bits for the fractional part.
    """

    integer_bits: int | Value[UIntType] = 0
    fractional_bits: int | Value[UIntType] = 0

    def label(self) -> str:
        return f"QFixed[{self.integer_bits}.{self.fractional_bits}]"
