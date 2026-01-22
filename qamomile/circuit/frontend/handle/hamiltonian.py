"""HamiltonianExpr handle for building Hamiltonian expressions.

This module provides the HamiltonianExpr handle class that supports
arithmetic operations to build Hamiltonian expressions within qkernels.
Each operation emits the corresponding IR operation to the tracer.
"""

from __future__ import annotations

import dataclasses
import typing

from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.types.hamiltonian import HamiltonianExprType
from qamomile.circuit.ir.types.primitives import FloatType
from qamomile.circuit.ir.operation.hamiltonian_ops import (
    HamiltonianAddOp,
    HamiltonianMulOp,
    HamiltonianScaleOp,
    HamiltonianNegOp,
)
from qamomile.circuit.ir.value import Value

from .handle import Handle

if typing.TYPE_CHECKING:
    from .primitives import Float


def _make_float_const(val: float) -> Value:
    """Create a Float Value with a constant."""
    return Value(type=FloatType(), name="float_const", params={"const": val})


def _make_hamiltonian_result() -> Value:
    """Create a HamiltonianExpr Value for an operation result."""
    return Value(type=HamiltonianExprType(), name="hamiltonian_tmp")


@dataclasses.dataclass
class HamiltonianExpr(Handle):
    """Handle for Hamiltonian expressions.

    Supports arithmetic operations that emit corresponding IR operations:
    - __add__: HamiltonianAddOp
    - __mul__: HamiltonianMulOp (with HamiltonianExpr) or HamiltonianScaleOp (with scalar)
    - __neg__: HamiltonianNegOp
    - __sub__: Implemented via neg + add

    The expression structure is captured in the operation graph.

    Example:
        H = qm.pauli.Z(0) * qm.pauli.Z(1) + qm.pauli.X(0)
        H = 0.5 * qm.pauli.Z(0)
    """

    def __add__(self, other: "HamiltonianExpr") -> "HamiltonianExpr":
        """Add two HamiltonianExprs."""
        if not isinstance(other, HamiltonianExpr):
            return NotImplemented

        result_value = _make_hamiltonian_result()
        op = HamiltonianAddOp(
            operands=[self.value, other.value],
            results=[result_value],
        )
        tracer = get_current_tracer()
        tracer.add_operation(op)

        return HamiltonianExpr(value=result_value)

    def __radd__(self, other: "HamiltonianExpr | int | float") -> "HamiltonianExpr":
        """Reverse add - handle `0 + expr` case for sum() support."""
        # Support `sum([expr1, expr2])` which starts with 0
        if isinstance(other, (int, float)) and other == 0:
            return self
        if isinstance(other, HamiltonianExpr):
            return other.__add__(self)
        return NotImplemented

    def __mul__(
        self, other: "HamiltonianExpr | Float | float | int"
    ) -> "HamiltonianExpr":
        """Multiply HamiltonianExpr by another HamiltonianExpr or scalar."""
        from .primitives import Float

        result_value = _make_hamiltonian_result()

        if isinstance(other, HamiltonianExpr):
            # HamiltonianExpr * HamiltonianExpr -> HamiltonianMulOp
            op = HamiltonianMulOp(
                operands=[self.value, other.value],
                results=[result_value],
            )
        elif isinstance(other, Float):
            # HamiltonianExpr * Float -> HamiltonianScaleOp
            op = HamiltonianScaleOp(
                operands=[self.value, other.value],
                results=[result_value],
            )
        elif isinstance(other, (int, float)):
            # HamiltonianExpr * literal -> HamiltonianScaleOp with const
            scalar_value = _make_float_const(float(other))
            op = HamiltonianScaleOp(
                operands=[self.value, scalar_value],
                results=[result_value],
            )
        else:
            return NotImplemented

        tracer = get_current_tracer()
        tracer.add_operation(op)

        return HamiltonianExpr(value=result_value)

    def __rmul__(
        self, other: "Float | float | int"
    ) -> "HamiltonianExpr":
        """Reverse multiply - scalar * HamiltonianExpr."""
        from .primitives import Float

        result_value = _make_hamiltonian_result()

        if isinstance(other, Float):
            scalar_value = other.value
        elif isinstance(other, (int, float)):
            scalar_value = _make_float_const(float(other))
        else:
            return NotImplemented

        op = HamiltonianScaleOp(
            operands=[self.value, scalar_value],
            results=[result_value],
        )
        tracer = get_current_tracer()
        tracer.add_operation(op)

        return HamiltonianExpr(value=result_value)

    def __neg__(self) -> "HamiltonianExpr":
        """Negate the HamiltonianExpr."""
        result_value = _make_hamiltonian_result()
        op = HamiltonianNegOp(
            operands=[self.value],
            results=[result_value],
        )
        tracer = get_current_tracer()
        tracer.add_operation(op)

        return HamiltonianExpr(value=result_value)

    def __sub__(self, other: "HamiltonianExpr") -> "HamiltonianExpr":
        """Subtract: self - other = self + (-other)."""
        if not isinstance(other, HamiltonianExpr):
            return NotImplemented
        return self.__add__(other.__neg__())

    def __rsub__(self, other: "HamiltonianExpr") -> "HamiltonianExpr":
        """Reverse subtract: other - self = (-self) + other."""
        if isinstance(other, HamiltonianExpr):
            return self.__neg__().__add__(other)
        return NotImplemented
