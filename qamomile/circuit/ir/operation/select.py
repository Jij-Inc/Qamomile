"""SELECT (quantum multiplexer) operation.

``SelectOperation`` is the IR node behind ``qmc.select``: a quantum
multiplexer that applies a *different* unitary ``U_i`` to a shared target
register depending on the computational-basis value ``i`` read off an
index (control) register::

    SELECT = sum_i |i><i| (x) U_i

Following the project's IR-abstraction principle, the op stays as a single
high-level box. Circuit-family lowering preserves that identity as one reusable
call whose fallback contains controlled case calls. A target can therefore
select a native realization without changing the frontend or semantic IR.
"""

from __future__ import annotations

import dataclasses

from qamomile._utils import is_plain_int
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.types.primitives import QubitType
from qamomile.circuit.ir.value import Value

from .operation import Operation, OperationKind, ParamHint, Signature


@dataclasses.dataclass
class SelectOperation(Operation):
    """Quantum multiplexer: apply ``case_blocks[i]`` when the index reads ``i``.

    Operand layout: ``[idx_0, ..., idx_{k-1}, tgt_0, ..., tgt_m, params...]``
    Result layout:  ``[idx_0', ..., idx_{k-1}', tgt_0', ..., tgt_m']``

    Like :class:`ConcreteControlledU`, the index register is normalised to
    one scalar ``Qubit`` operand per physical index qubit
    (``operands[:num_index_qubits]``); whole-``Vector[Qubit]`` / scalar
    targets follow and keep their shapes, and classical parameters (shared
    across every case and forwarded to each case block) come last.

    Index bit order is **LSB-first**: ``idx_0`` is the least-significant
    bit, matching Qamomile's qubit-zero convention. Case ``i`` is selected
    when index qubit ``j`` reads bit ``j`` of ``i``. ``len(case_blocks)``
    need not be a power of two; index values ``>= len(case_blocks)`` apply
    no operation (identity).

    Attributes:
        num_index_qubits (int): Number of physical index (select) qubits.
            Equals ``ceil(log2(len(case_blocks)))`` (at least ``1``).
        case_blocks (list[Block]): One unitary ``Block`` per index value,
            in ascending index order. Every block shares the same quantum
            target arity and classical-parameter signature.
    """

    num_index_qubits: int = 0
    case_blocks: list[Block] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate the case-count and index-qubit invariants.

        Guards against malformed externally-decoded or hand-built IR:

        * there must be at least two case blocks (a zero- or one-case SELECT
          is not a multiplexer and must be represented directly);
        * the index register must have enough qubits to address every
          case (``2 ** num_index_qubits >= len(case_blocks)``).

        The frontend always constructs a valid op, and value-substitution
        passes preserve ``case_blocks`` verbatim via ``dataclasses.replace``,
        so this only fires on hand-built or corrupt IR.

        Raises:
            ValueError: If fewer than two case blocks are present,
                ``num_index_qubits`` is not a positive Python int, or there
                are more case blocks than ``2 ** num_index_qubits`` index
                values can address.
        """
        if not is_plain_int(self.num_index_qubits):
            raise ValueError(
                "SelectOperation.num_index_qubits must be a Python int, "
                f"got {self.num_index_qubits!r}."
            )
        if self.num_index_qubits < 1:
            raise ValueError(
                f"SelectOperation.num_index_qubits must be positive, "
                f"got {self.num_index_qubits}."
            )
        if len(self.case_blocks) < 2:
            raise ValueError(
                "SelectOperation requires at least two case blocks; a zero- "
                "or one-case operation is not a multiplexer."
            )
        if len(self.case_blocks) > (1 << self.num_index_qubits):
            raise ValueError(
                f"SelectOperation has {len(self.case_blocks)} case blocks but "
                f"only {1 << self.num_index_qubits} index value(s) "
                f"({self.num_index_qubits} index qubit(s)) to address them."
            )

    @property
    def index_operands(self) -> list[Value]:
        """Return the scalar index (select) qubit operands.

        Returns:
            list[Value]: The leading ``num_index_qubits`` operands, one
                per physical index qubit.
        """
        return list(self.operands[: self.num_index_qubits])

    @property
    def target_operands(self) -> list[Value]:
        """Return the quantum target operands applied by every case.

        Returns:
            list[Value]: The quantum operands following the index prefix
                (scalar ``Qubit`` Values and/or whole ``Vector[Qubit]``
                ``ArrayValue``s).
        """
        return [
            op for op in self.operands[self.num_index_qubits :] if op.type.is_quantum()
        ]

    @property
    def param_operands(self) -> list[Value]:
        """Return the shared classical parameter operands.

        Returns:
            list[Value]: The non-quantum operands following the index
                prefix, forwarded identically to every case block.
        """
        return [
            op
            for op in self.operands[self.num_index_qubits :]
            if op.type.is_classical() or op.type.is_object()
        ]

    @property
    def num_cases(self) -> int:
        """Return the number of selectable cases.

        Returns:
            int: ``len(case_blocks)``.
        """
        return len(self.case_blocks)

    @property
    def signature(self) -> Signature:
        """Return the operation signature.

        Returns:
            Signature: Index qubits followed by the post-index operands
                (targets then params), with quantum results mirrored on
                the result side.
        """
        k = self.num_index_qubits
        return Signature(
            operands=[
                *[ParamHint(name=f"index_{i}", type=QubitType()) for i in range(k)],
                *[
                    ParamHint(name=f"arg_{i}", type=op.type)
                    for i, op in enumerate(self.operands[k:])
                ],
            ],
            results=[
                *[ParamHint(name=f"index_{i}", type=QubitType()) for i in range(k)],
                *[
                    ParamHint(name=f"target_{i}", type=r.type)
                    for i, r in enumerate(self.results[k:])
                ],
            ],
        )

    @property
    def operation_kind(self) -> OperationKind:
        """Return the operation kind.

        Returns:
            OperationKind: Always ``OperationKind.QUANTUM``.
        """
        return OperationKind.QUANTUM


def _control_pattern_for_index(
    index_value: int, num_index_qubits: int
) -> tuple[int, ...]:
    """Return the LSB-first control pattern selecting ``index_value``.

    The returned tuple aligns positionally with a :class:`SelectOperation`'s
    ``index_operands``: entry ``j`` is the basis state index qubit ``j`` must
    read for case ``index_value`` to fire. Index qubit ``0`` is the
    least-significant bit.

    Args:
        index_value (int): The case index ``i`` to select. Must satisfy
            ``0 <= index_value < 2 ** num_index_qubits``.
        num_index_qubits (int): Number of index qubits ``k``.

    Returns:
        tuple[int, ...]: A length-``num_index_qubits`` tuple of ``0``/``1``
            ints, least-significant bit first.

    Raises:
        ValueError: If either argument is not a Python int,
            ``num_index_qubits`` is negative, or ``index_value`` does not fit
            in ``num_index_qubits`` bits.

    Example:
        >>> _control_pattern_for_index(2, 2)
        (0, 1)
        >>> _control_pattern_for_index(1, 3)
        (1, 0, 0)
    """
    if not is_plain_int(num_index_qubits):
        raise ValueError(
            f"num_index_qubits must be a Python int, got {num_index_qubits!r}."
        )
    if not is_plain_int(index_value):
        raise ValueError(f"index_value must be a Python int, got {index_value!r}.")
    if num_index_qubits < 0:
        raise ValueError(
            f"num_index_qubits must be non-negative, got {num_index_qubits}."
        )
    if not (0 <= index_value < (1 << num_index_qubits)):
        raise ValueError(
            f"index_value {index_value} does not fit in {num_index_qubits} "
            f"index qubit(s) (valid range 0..{(1 << num_index_qubits) - 1})."
        )
    return tuple((index_value >> j) & 1 for j in range(num_index_qubits))


__all__ = ["SelectOperation"]
