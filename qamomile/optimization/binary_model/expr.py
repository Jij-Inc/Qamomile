from __future__ import annotations

import dataclasses
import enum
from collections import Counter
from typing import Generic, TypeVar

from qamomile.optimization.utils import is_close_zero


class VarType(enum.StrEnum):
    BINARY = "BINARY"
    SPIN = "SPIN"


VT = TypeVar("VT", bound=VarType)


@dataclasses.dataclass
class BinaryExpr(Generic[VT]):
    vartype: VT = VarType.BINARY  # type: ignore
    constant: float = 0.0
    coefficients: dict[tuple[int, ...], float] = dataclasses.field(default_factory=dict)

    def copy(self) -> BinaryExpr[VT]:
        return BinaryExpr(
            vartype=self.vartype,
            constant=self.constant,
            coefficients=self.coefficients.copy(),
        )

    def _reduce_indices(self, inds: tuple[int, ...]) -> tuple[int, ...]:
        """Apply idempotency rules to reduce repeated indices.

        For SPIN: z_i^2 = 1, so pairs of identical indices cancel.
        For BINARY: x_i^2 = x_i, so duplicates reduce to single.
        """
        if self.vartype == VarType.SPIN:
            counts = Counter(inds)
            return tuple(idx for idx, count in sorted(counts.items()) if count % 2 == 1)
        elif self.vartype == VarType.BINARY:
            return tuple(sorted(set(inds)))
        return inds

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
            constant_delta = 0.0
            for inds1, coeff1 in self.coefficients.items():
                for inds2, coeff2 in other.coefficients.items():
                    new_inds = self._reduce_indices(tuple(sorted(inds1 + inds2)))
                    product = coeff1 * coeff2
                    if len(new_inds) == 0:
                        constant_delta += product
                    else:
                        new_coefficients[new_inds] = (
                            new_coefficients.get(new_inds, 0.0) + product
                        )
            # Handle constant term multiplication
            for inds, coeff in self.coefficients.items():
                new_coefficients[inds] = (
                    new_coefficients.get(inds, 0.0) + coeff * other.constant
                )
            for inds, coeff in other.coefficients.items():
                new_coefficients[inds] = (
                    new_coefficients.get(inds, 0.0) + coeff * self.constant
                )
            self.constant = self.constant * other.constant + constant_delta
            self.coefficients = {
                k: v for k, v in new_coefficients.items() if not is_close_zero(v)
            }
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
        vartype=VarType.BINARY, constant=0.0, coefficients={(index,): 1.0}
    )


def spin(index: int) -> BinaryExpr[VarType]:
    """Create a spin variable expression.

    Args:
        index (int): The index of the spin variable.

    Returns:
        BinaryExpr[VarType.SPIN]: The spin variable expression.
    """
    return BinaryExpr(vartype=VarType.SPIN, constant=0.0, coefficients={(index,): 1.0})
