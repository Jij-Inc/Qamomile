"""Classical operations for quantum-classical hybrid programs."""

import dataclasses

from qamomile.circuit.ir.types.primitives import BitType, FloatType

from .operation import Operation, OperationKind, ParamHint, Signature


@dataclasses.dataclass
class DecodeQFixedOperation(Operation):
    """Decode measured bits to float (classical operation).

    This operation converts a sequence of classical bits from qubit measurements
    into a floating-point number using fixed-point encoding.

    The decoding formula:
        float_value = Σ bit[i] * 2^(int_bits - 1 - i)

    For QPE phase (int_bits=0):
        float_value = 0.b0b1b2... = b0*0.5 + b1*0.25 + b2*0.125 + ...

    Example:
        bits = [1, 0, 1] with int_bits=0
        → 0.101 (binary) = 0.5 + 0.125 = 0.625

    Attributes:
        num_bits: Total number of bits to decode.
        int_bits: Number of integer bits (0 for pure fractional like QPE phase).

    operands: [ArrayValue of bits (vec[bit])]
    results: [Float value]
    """

    num_bits: int = 0
    int_bits: int = 0

    @property
    def signature(self) -> Signature:
        # Accept a single ArrayValue[Bit] as operand
        return Signature(
            operands=[ParamHint(name="bits", type=BitType())],
            results=[ParamHint(name="float_out", type=FloatType())],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CLASSICAL
