"""First-class global-phase block operation.

``GlobalPhaseBlockOperation`` represents "apply this block's unitary and
multiply the whole state by ``e^{i*phase}``" as a single IR operation. It
is produced by the :func:`qamomile.circuit.global_phase` combinator and
mirrors the operand layout of
:class:`~qamomile.circuit.ir.operation.inverse_block.InverseBlockOperation`
(control qubits, then quantum targets, then classical/object parameters).

The scalar phase angle is stored in a dedicated ``phase`` field rather than
as an operand, because it is **not** an input of the wrapped block -- it is
metadata describing the extra global phase to apply on top of the block's
unitary. ``phase`` still participates in value substitution / dependency
analysis via the :meth:`all_input_values` / :meth:`replace_values`
overrides (mirroring ``ControlledUOperation.power``), so a symbolic phase
that references a runtime parameter is resolved like any other Value.
"""

from __future__ import annotations

import dataclasses
import typing
from typing import TYPE_CHECKING

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.types.primitives import BlockType, QubitType
from qamomile.circuit.ir.value import ArrayValue, ValueBase

from .operation import Operation, OperationKind, ParamHint, Signature

if TYPE_CHECKING:
    from qamomile.circuit.ir.value import Value


@dataclasses.dataclass
class GlobalPhaseBlockOperation(Operation):
    """Apply a block's unitary together with a global phase ``e^{i*phase}``.

    The operation stores the forward block whose unitary should be emitted
    plus a scalar ``phase`` angle. Standalone, the global phase is
    physically unobservable, so backends either fold it into a native
    circuit-level global-phase accumulator (Qiskit) or drop it (CUDA-Q /
    QURI Parts) while still emitting the wrapped block's body. Under an
    enclosing control, the global phase becomes a *relative* phase on the
    control qubit(s) and is realized as an ordinary phase gate.

    Operands are ordered as control qubits, target quantum operands, then
    classical/object parameters of the wrapped block. Results mirror the
    quantum operand layout: control results first, then one target result
    per target operand. Vector target operands count as one operand/result
    while contributing their scalar width to ``num_target_qubits``. The
    phase angle is **not** an operand (see module docstring).

    Attributes:
        num_control_qubits (int): Number of leading control operands and
            pass-through control results. The :func:`global_phase`
            combinator always emits ``0``; controls are supplied by an
            enclosing :class:`ControlledUOperation`.
        num_target_qubits (int): Scalar qubit width occupied by target
            operands at emit time. Vector operands count by static scalar
            width here but still produce one vector result operand.
        custom_name (str): Human-readable operation name.
        source_block (Block | None): Forward block whose unitary should be
            emitted. The wrapped block's own body, never an inverse of it.
        phase (Value | None): Scalar ``FloatType`` Value holding the global
            phase angle ``theta`` (in radians). May be a compile-time
            constant or a symbolic expression of runtime parameters.
    """

    num_control_qubits: int = 0
    num_target_qubits: int = 0
    custom_name: str = ""
    source_block: Block | None = None
    phase: "Value | None" = None

    def __post_init__(self) -> None:
        """Validate global-phase-block operand and result layout invariants.

        Raises:
            ValueError: If qubit counts are negative, if control operands
                are not quantum values, if a quantum target operand appears
                after a non-quantum parameter, if the results do not mirror
                the quantum operand layout, or if ``phase`` is a quantum
                value.
        """
        if self.num_control_qubits < 0 or self.num_target_qubits < 0:
            raise ValueError("global-phase block qubit counts must be non-negative.")
        if self.num_control_qubits > len(self.operands):
            raise ValueError(
                "global-phase block control count exceeds the number of operands."
            )

        for operand in self.operands[: self.num_control_qubits]:
            if not operand.type.is_quantum():
                raise ValueError("global-phase block control operands must be quantum.")

        seen_parameter = False
        for operand in self.operands[self.num_control_qubits :]:
            if operand.type.is_quantum():
                if seen_parameter:
                    raise ValueError(
                        "global-phase block quantum target operands must precede "
                        "classical/object parameters."
                    )
            else:
                seen_parameter = True

        num_targets = len(self.target_qubits)
        expected_results = self.num_control_qubits + num_targets
        if len(self.results) != expected_results:
            raise ValueError(
                "global-phase block results must mirror the quantum operand "
                f"layout: expected {expected_results} "
                f"({self.num_control_qubits} control + {num_targets} target), "
                f"got {len(self.results)}."
            )
        quantum_operands = [
            *self.operands[: self.num_control_qubits],
            *self.target_qubits,
        ]
        for operand, result in zip(quantum_operands, self.results):
            if not result.type.is_quantum():
                raise ValueError("global-phase block results must be quantum values.")
            if isinstance(operand, ArrayValue) != isinstance(result, ArrayValue):
                raise ValueError(
                    "global-phase block results must mirror operand array-ness: "
                    f"operand {operand.name!r} and result {result.name!r} "
                    "disagree on being a vector."
                )

        if self.phase is not None and self.phase.type.is_quantum():
            raise ValueError("global-phase block phase must be a classical value.")

    @property
    def control_qubits(self) -> list["Value"]:
        """Return control quantum operands.

        Returns:
            list[Value]: Leading control operands.
        """
        return list(self.operands[: self.num_control_qubits])

    @property
    def target_qubits(self) -> list["Value"]:
        """Return target quantum operands.

        Returns:
            list[Value]: Quantum operands consumed by the operation after
                control operands. A vector operand counts as one operand
                here even though ``num_target_qubits`` stores its scalar
                backend width.
        """
        start = self.num_control_qubits
        targets: list["Value"] = []
        for operand in self.operands[start:]:
            if not operand.type.is_quantum():
                break
            targets.append(operand)
        return targets

    @property
    def parameters(self) -> list["Value"]:
        """Return classical/object parameter operands of the wrapped block.

        Returns:
            list[Value]: Non-quantum operands after the quantum targets.
                This does **not** include the global-phase angle, which is
                stored in the dedicated ``phase`` field.
        """
        start = self.num_control_qubits + len(self.target_qubits)
        return list(self.operands[start:])

    @property
    def name(self) -> str:
        """Return a human-readable operation name.

        Returns:
            str: Explicit custom name, or ``"global_phase"`` when unnamed.
        """
        return self.custom_name or "global_phase"

    @property
    def signature(self) -> Signature:
        """Return the operation signature.

        Returns:
            Signature: Signature with the source block hint, target qubit
                operands, parameter operands, and target qubit results.
        """
        operand_hints: list[ParamHint | None] = [ParamHint("source", BlockType())]
        for i in range(self.num_control_qubits):
            operand_hints.append(ParamHint(f"control_{i}", QubitType()))
        for i, target in enumerate(self.target_qubits):
            operand_hints.append(ParamHint(f"target_{i}", target.type))
        for i, param in enumerate(self.parameters):
            operand_hints.append(ParamHint(f"param_{i}", param.type))

        result_hints = [
            ParamHint(f"result_{i}", result.type)
            for i, result in enumerate(self.results)
        ]
        return Signature(operands=operand_hints, results=result_hints)

    @property
    def operation_kind(self) -> OperationKind:
        """Return the operation kind.

        Returns:
            OperationKind: Always ``OperationKind.QUANTUM``.
        """
        return OperationKind.QUANTUM

    def all_input_values(self) -> list[ValueBase]:
        """Return all input Values, including the dedicated ``phase`` field.

        Returns:
            list[ValueBase]: Operand input Values followed by the global
                phase Value when it is present, so generic passes (analyze,
                canonicalize) see the phase as a dependency.
        """
        values = super().all_input_values()
        if isinstance(self.phase, ValueBase):
            values.append(self.phase)
        return values

    def replace_values(self, mapping: dict[str, ValueBase]) -> Operation:
        """Return a copy with all Values substituted, including ``phase``.

        Args:
            mapping (dict[str, ValueBase]): UUID-keyed substitution map.

        Returns:
            Operation: A new ``GlobalPhaseBlockOperation`` with operands,
                results, and the ``phase`` field remapped.
        """
        result = super().replace_values(mapping)
        assert isinstance(result, GlobalPhaseBlockOperation)
        if isinstance(self.phase, ValueBase) and self.phase.uuid in mapping:
            mapped = mapping[self.phase.uuid]
            if isinstance(mapped, ValueBase):
                return dataclasses.replace(result, phase=typing.cast("Value", mapped))
        return result
