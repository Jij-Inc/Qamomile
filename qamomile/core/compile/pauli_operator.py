from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class PauliType(Enum):
    """Pauli Type Enum"""

    X = "X"
    Y = "Y"
    Z = "Z"


@dataclass(frozen=True)
class PauliOperator:
    """Pauli Operator class"""

    pauli_type: PauliType
    qubit_index: int
