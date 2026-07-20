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

    The decoding formula for least-significant-first storage:
        float_value = Σ bit[i] * 2^(int_bits - num_bits + i)

    For QPE phase (int_bits=0):
        bit[0] has weight ``2**(-num_bits)`` and bit[-1] has weight 0.5.

    Example:
        bits = [1, 0, 1] with int_bits=0
        → 0.101 (MSB-first display) = 0.5 + 0.125 = 0.625

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
        """Return the operation's dynamic array/qubit/index signature.

        Returns:
            Signature: Operand-only signature with no SSA results.
        """
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
class ReturnQuantumArrayElementOperation(Operation):
    """Validate a branch-selected quantum element's array return at emit time.

    Most quantum element assignments are verified structurally by the
    frontend and emit no IR. A compile-time conditional can instead select
    different element indices on its branches; only the unrolled emit context
    knows which source index survived. This operation carries both the
    requested target indices and the conditional source indices so emission
    can prove they resolve to the same physical slot before treating the
    assignment as a borrow return.

    Operand convention:
        ``[array, returned_qubit, *target_indices, *source_indices]``. The
        target and source halves have equal nonzero arity, inferred from the
        operand count. The operation has no results and emits no backend gate.
    """

    @property
    def index_arity(self) -> int:
        """Return the number of target (and source) index operands.

        Returns:
            int: Half of the operands following the array and qubit.
        """
        return (len(self.operands) - 2) // 2

    @property
    def array(self) -> ArrayValue:
        """Return the quantum array receiving the borrowed element.

        Returns:
            ArrayValue: Root array operand.
        """
        return self.operands[0]  # type: ignore[return-value]

    @property
    def returned_value(self) -> Value:
        """Return the quantum value being returned.

        Returns:
            Value: Returned qubit operand.
        """
        return self.operands[1]

    @property
    def target_indices(self) -> tuple[Value, ...]:
        """Return the user-written assignment indices.

        Returns:
            tuple[Value, ...]: Target index operands.
        """
        arity = self.index_arity
        return tuple(self.operands[2 : 2 + arity])

    @property
    def source_indices(self) -> tuple[Value, ...]:
        """Return the branch-merged borrow-source indices.

        Returns:
            tuple[Value, ...]: Conditional source index operands.
        """
        arity = self.index_arity
        return tuple(self.operands[2 + arity :])

    @property
    def signature(self) -> Signature:
        """Return the deferred validator's operand-only signature.

        Returns:
            Signature: Dynamic array, qubit, and index operands with no SSA
                results.
        """
        return Signature(
            operands=[
                ParamHint(name="array", type=self.array.type),
                ParamHint(name="returned_value", type=self.returned_value.type),
                *[
                    ParamHint(name=f"target_index_{index}", type=value.type)
                    for index, value in enumerate(self.target_indices)
                ],
                *[
                    ParamHint(name=f"source_index_{index}", type=value.type)
                    for index, value in enumerate(self.source_indices)
                ],
            ],
            results=[],
        )

    @property
    def operation_kind(self) -> OperationKind:
        """Classify the return validator as a quantum operation.

        Returns:
            OperationKind: ``OperationKind.QUANTUM``.
        """
        return OperationKind.QUANTUM


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
