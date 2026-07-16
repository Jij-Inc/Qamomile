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
from qamomile.circuit.ir.types.primitives import QubitType, UIntType
from qamomile.circuit.ir.value import Value, ValueBase

from .operation import Operation, OperationKind, ParamHint, Signature


@dataclasses.dataclass
class SelectOperation(Operation):
    """Quantum multiplexer: apply ``case_blocks[i]`` when the index reads ``i``.

    Concrete operand layout:
    ``[idx_0, ..., idx_{k-1}, tgt_0, ..., tgt_m, params...]``.
    Symbolic-width operand layout:
    ``[idx_arg_0, ..., idx_arg_{a-1}, tgt_0, ..., tgt_m, params...]``.
    Results mirror the quantum operand grouping.

    A concrete index register is normalized to one scalar ``Qubit`` operand
    per physical index qubit. A symbolic-width register instead retains each
    leading caller argument as one scalar or array operand until its bound
    shape is known. Whole-``Vector[Qubit]`` / scalar targets follow and keep
    their shapes, and classical parameters shared across every case come last.

    Index bit order is **LSB-first**: ``idx_0`` is the least-significant
    bit, matching Qamomile's qubit-zero convention. Case ``i`` is selected
    when index qubit ``j`` reads bit ``j`` of ``i``. ``len(case_blocks)``
    need not be a power of two; index values ``>= len(case_blocks)`` apply
    no operation (identity).

    Attributes:
        num_index_qubits (int | Value): Number of physical index (select)
            qubits. A concrete width must be positive and wide enough to
            address every case. A symbolic ``UInt`` value is resolved during
            compilation and validated before circuit lowering.
        case_blocks (list[Block]): One unitary ``Block`` per index value,
            in ascending index order. Every block shares the same quantum
            target arity and classical-parameter signature.
        num_index_args (int): Number of leading operand/result slots occupied
            by index arguments. Concrete SELECT stores one scalar slot per
            physical index qubit, so this equals ``num_index_qubits``.
            Symbolic SELECT may retain heterogeneous scalar/array arguments
            whose flattened width is checked after bindings are available.
    """

    num_index_qubits: int | Value = 0
    case_blocks: list[Block] = dataclasses.field(default_factory=list)
    num_index_args: int = 0

    def __post_init__(self) -> None:
        """Validate the case-count and index-qubit invariants.

        Guards against malformed externally-decoded or hand-built IR:

        * there must be at least two case blocks (a zero- or one-case SELECT
          is not a multiplexer and must be represented directly);
        * a concrete index register must have enough qubits to address every
          case (``2 ** num_index_qubits >= len(case_blocks)``);
        * a symbolic width must be an unsigned-integer ``Value`` and retain at
          least one index argument for later flattening.

        The frontend always constructs a valid op, and value-substitution
        passes preserve ``case_blocks`` verbatim via ``dataclasses.replace``,
        so this only fires on hand-built or corrupt IR.

        Raises:
            ValueError: If fewer than two case blocks are present, the width
                or index-argument count is malformed, or a concrete width
                cannot address every case.
        """
        if len(self.case_blocks) < 2:
            raise ValueError(
                "SelectOperation requires at least two case blocks; a zero- "
                "or one-case operation is not a multiplexer."
            )

        width_value = self.num_index_qubits
        if is_plain_int(width_value):
            assert isinstance(width_value, int)
            width = width_value
            if width < 1:
                raise ValueError(
                    f"SelectOperation.num_index_qubits must be positive, got {width}."
                )
            minimum_width = (len(self.case_blocks) - 1).bit_length()
            if width < minimum_width:
                raise ValueError(
                    f"SelectOperation has {len(self.case_blocks)} case blocks but "
                    f"num_index_qubits={width}; at least {minimum_width} index "
                    f"qubit(s) are required to address every case."
                )
            if self.num_index_args == 0:
                self.num_index_args = width
            elif self.num_index_args != width:
                raise ValueError(
                    "A concrete SelectOperation requires one scalar index "
                    "operand per index qubit; num_index_args must equal "
                    f"num_index_qubits ({width}), got {self.num_index_args}."
                )
        elif isinstance(width_value, Value):
            if not isinstance(width_value.type, UIntType):
                raise ValueError(
                    "SelectOperation.num_index_qubits must be an int or a "
                    "UInt Value, got a Value of type "
                    f"{type(width_value.type).__name__}."
                )
        else:
            raise ValueError(
                "SelectOperation.num_index_qubits must be a Python int or "
                "UInt Value, "
                f"got {width_value!r}."
            )

        if not is_plain_int(self.num_index_args) or self.num_index_args < 1:
            raise ValueError(
                "SelectOperation.num_index_args must be a positive Python "
                f"int, got {self.num_index_args!r}."
            )

    @property
    def is_symbolic_num_index_qubits(self) -> bool:
        """Return whether the index width is a symbolic IR value.

        Returns:
            bool: ``True`` when ``num_index_qubits`` is a ``Value``.
        """
        return isinstance(self.num_index_qubits, Value)

    @property
    def index_operands(self) -> list[Value]:
        """Return the grouped index-prefix operands.

        Returns:
            list[Value]: The leading ``num_index_args`` operands. Concrete
                SELECT has one scalar per physical index qubit; symbolic
                SELECT may retain scalar and array arguments as groups.
        """
        return list(self.operands[: self.num_index_args])

    @property
    def target_operands(self) -> list[Value]:
        """Return the quantum target operands applied by every case.

        Returns:
            list[Value]: The quantum operands following the index prefix
                (scalar ``Qubit`` Values and/or whole ``Vector[Qubit]``
                ``ArrayValue``s).
        """
        return [
            op for op in self.operands[self.num_index_args :] if op.type.is_quantum()
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
            for op in self.operands[self.num_index_args :]
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
        if self.is_symbolic_num_index_qubits:
            raise NotImplementedError(
                "Cannot compute a scalar signature for symbolic-width SelectOperation."
            )
        k = self.num_index_args
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

    def all_input_values(self) -> list[ValueBase]:
        """Return every value consumed by the SELECT operation.

        Returns:
            list[ValueBase]: Ordinary operands plus the symbolic index-width
                value when present.
        """
        values = super().all_input_values()
        if isinstance(self.num_index_qubits, Value):
            values.append(self.num_index_qubits)
        return values

    def replace_values(self, mapping: dict[str, ValueBase]) -> Operation:
        """Replace operand and symbolic-width values by UUID.

        Args:
            mapping (dict[str, ValueBase]): UUID-keyed replacement values.

        Returns:
            Operation: Rebuilt SELECT operation with matching values replaced.
        """
        result = super().replace_values(mapping)
        assert isinstance(result, SelectOperation)
        width = result.num_index_qubits
        if isinstance(width, Value) and width.uuid in mapping:
            replacement = mapping[width.uuid]
            if isinstance(replacement, Value):
                return dataclasses.replace(result, num_index_qubits=replacement)
        return result


__all__ = ["SelectOperation"]
