from __future__ import annotations
import dataclasses
import enum
from typing import Generic, TypeVar

import numpy as np
from qamomile.circuit.transpiler.job import SampleResult


class VarType(enum.StrEnum):
    BINARY = "BINARY"
    SPIN = "SPIN"


VT = TypeVar('VT', bound=VarType)


@dataclasses.dataclass
class BinaryExpr(Generic[VT]):
    vartype: VT = VarType.BINARY  # type: ignore
    constant: float = 0.0
    coefficients: dict[tuple[int, ...], float] = dataclasses.field(default_factory=dict)

    def copy(self) -> BinaryExpr[VT]:
        return BinaryExpr(
            vartype=self.vartype,
            constant=self.constant,
            coefficients=self.coefficients.copy()
        )

    # Support *= operation
    def __imul__(self, other: int | float | BinaryExpr[VT]) -> BinaryExpr[VT]:
        if isinstance(other, (int, float)):
            for inds in self.coefficients:
                self.coefficients[inds] *= other
            self.constant *= other
        elif isinstance(other, BinaryExpr):
            if self.vartype != other.vartype:
                raise ValueError("Cannot multiply BinaryExpr with different vartypes.")
            new_coefficients: dict[tuple[int, ...], float] = {}
            for inds1, coeff1 in self.coefficients.items():
                for inds2, coeff2 in other.coefficients.items():
                    new_inds = tuple(sorted(inds1 + inds2))
                    new_coefficients[new_inds] = new_coefficients.get(new_inds, 0.0) + coeff1 * coeff2
            # Handle constant term multiplication
            for inds, coeff in self.coefficients.items():
                new_coefficients[inds] = new_coefficients.get(inds, 0.0) + coeff * other.constant
            for inds, coeff in other.coefficients.items():
                new_coefficients[inds] = new_coefficients.get(inds, 0.0) + coeff * self.constant
            self.constant *= other.constant
            self.coefficients = new_coefficients
        else:
            raise TypeError("Unsupported type for multiplication with BinaryExpr.")
        return self

    def __mul__(self, other: int | float | BinaryExpr[VT]) -> BinaryExpr[VT]:
        result = self.copy()
        result *= other
        return result

    def __rmul__(self, other: int | float | BinaryExpr[VT]) -> BinaryExpr[VT]:
        return self.__mul__(other)

    # Support += operation
    def __iadd__(self, other: int | float | BinaryExpr[VT]) -> BinaryExpr[VT]:
        if isinstance(other, (int, float)):
            self.constant += other
        elif isinstance(other, BinaryExpr):
            if self.vartype != other.vartype:
                raise ValueError("Cannot add BinaryExpr with different vartypes.")
            for inds, coeff in other.coefficients.items():
                self.coefficients[inds] = self.coefficients.get(inds, 0.0) + coeff
            self.constant += other.constant
        else:
            raise TypeError("Unsupported type for addition with BinaryExpr.")
        return self

    def __add__(self, other: int | float | BinaryExpr[VT]) -> BinaryExpr[VT]:
        result = self.copy()
        result += other
        return result

    def __radd__(self, other: int | float | BinaryExpr[VT]) -> BinaryExpr[VT]:
        return self.__add__(other)


def binary(index: int) -> BinaryExpr[VarType]:
    """Create a binary variable expression.

    Args:
        index (int): The index of the binary variable.

    Returns:
        BinaryExpr[VarType.BINARY]: The binary variable expression.
    """
    return BinaryExpr(
        vartype=VarType.BINARY,
        constant=0.0,
        coefficients={(index,): 1.0}
    )

def spin(index: int) -> BinaryExpr[VarType]:
    """Create a spin variable expression.

    Args:
        index (int): The index of the spin variable.

    Returns:
        BinaryExpr[VarType.SPIN]: The spin variable expression.
    """
    return BinaryExpr(
        vartype=VarType.SPIN,
        constant=0.0,
        coefficients={(index,): 1.0}
    )