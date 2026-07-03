"""Classical operations for quantum-classical hybrid programs."""

import dataclasses

from qamomile.circuit.ir.types.primitives import BitType, FloatType
from qamomile.circuit.ir.value import ArrayValue, Value

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


@dataclasses.dataclass
class StoreArrayElementOperation(Operation):
    """Store a classical scalar into one element of a classical array.

    This is the IR form of ``array[index] = value`` for classical element
    types (``Bit`` / ``UInt`` / ``Float``).  Classical values are freely
    copyable, so the store is an ordinary SSA rewrite: the operation
    consumes the current array version and produces a new ``ArrayValue``
    version (same ``logical_id``, fresh ``uuid``) whose contents equal the
    input array with the addressed element replaced.  Quantum arrays never
    use this operation — qubit element assignment is the return half of
    the borrow-return idiom and emits no IR.

    The operation is evaluated in one of two places:

    - **Compile time**: ``ConstantFoldingPass`` folds the store when the
      source array contents, the index, and the stored value are all
      compile-time resolvable, attaching the updated ``const_array``
      metadata to the result value.
    - **Runtime**: otherwise the store executes host-side in a classical
      segment via ``ClassicalExecutor`` (e.g. for measurement-derived
      ``Vector[Bit]`` contents).  It must never reach a quantum segment;
      backend emit rejects it explicitly.

    Operand convention:
        operands: ``[array (ArrayValue), stored_value (Value), *index_values]``
        results: ``[new_array (ArrayValue)]``

    Example:
        ```python
        @qmc.qkernel
        def k() -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(2, "qs")
            qs[0] = qmc.x(qs[0])
            bits = qmc.measure(qs)
            bits[1] = bits[0]   # emits StoreArrayElementOperation
            return bits
        ```
    """

    @property
    def array(self) -> ArrayValue:
        """ArrayValue: The array version the store reads from.

        Returns:
            ArrayValue: ``operands[0]``.
        """
        return self.operands[0]  # type: ignore[return-value]

    @property
    def stored_value(self) -> Value:
        """Value: The scalar being written into the array.

        Returns:
            Value: ``operands[1]``.
        """
        return self.operands[1]

    @property
    def index_values(self) -> tuple[Value, ...]:
        """tuple[Value, ...]: The element indices being written.

        Returns:
            tuple[Value, ...]: ``operands[2:]`` — one entry per array
                dimension (a single entry for ``Vector``).
        """
        return tuple(self.operands[2:])

    @property
    def signature(self) -> Signature:
        return Signature(
            operands=[
                ParamHint(name="array", type=self.operands[0].type),
                ParamHint(name="stored_value", type=self.operands[1].type),
                *[
                    ParamHint(name=f"index_{i}", type=idx.type)
                    for i, idx in enumerate(self.operands[2:])
                ],
            ],
            results=[ParamHint(name="new_array", type=self.results[0].type)],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CLASSICAL


@dataclasses.dataclass
class DictGetItemOperation(Operation):
    """Look up one entry of a Dict by a (possibly symbolic) key.

    This is the IR form of ``d[key]`` on a ``Dict`` handle.  The key
    components may be symbolic (e.g. loop variables of a for-items
    loop); the lookup is resolved at emit time when the key values and
    the dict's bound data are both concrete.

    Attributes:
        key_arity: Number of key components (1 for scalar keys, N for
            tuple keys like ``d[(i, j)]``).

    operands: [DictValue, *key_component_values]
    results: [looked-up scalar value]
    """

    key_arity: int = 1

    @property
    def dict_value(self) -> Value:
        """Value: The DictValue being indexed (``operands[0]``).

        Returns:
            Value: ``operands[0]`` (a ``DictValue`` stored as operand).
        """
        return self.operands[0]

    @property
    def key_values(self) -> tuple[Value, ...]:
        """tuple[Value, ...]: The key component values.

        Returns:
            tuple[Value, ...]: ``operands[1:]``.
        """
        return tuple(self.operands[1:])

    @property
    def signature(self) -> Signature:
        return Signature(
            operands=[
                ParamHint(name="dict", type=self.operands[0].type),
                *[
                    ParamHint(name=f"key_{i}", type=kv.type)
                    for i, kv in enumerate(self.operands[1:])
                ],
            ],
            results=[ParamHint(name="value", type=self.results[0].type)],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CLASSICAL
