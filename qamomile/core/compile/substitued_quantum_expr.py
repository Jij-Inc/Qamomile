from __future__ import annotations
from dataclasses import dataclass
from .pauli_operator import PauliOperator, PauliType


@dataclass
class SubstitutedQuantumExpression:
    coeff: dict[tuple[PauliOperator, ...], complex]
    constant: float
    order: int
