"""First-class inverse block operation.

``InverseBlockOperation`` represents "apply the inverse of this block"
as a single IR operation. It shares the operand layout convention of
:class:`~qamomile.circuit.ir.operation.callable.InvokeOperation`
(control qubits, then quantum targets, then classical/object
parameters) but is an independent :class:`Operation` subclass, not a
composite-gate variant.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import CallableRef
from qamomile.circuit.ir.types.primitives import BlockType, QubitType
from qamomile.circuit.ir.value import ArrayValue

from .control_value import normalize_control_value
from .operation import Operation, OperationKind, ParamHint, Signature

if TYPE_CHECKING:
    from qamomile.circuit.ir.value import Value


@dataclasses.dataclass
class InverseBlockOperation(Operation):
    """Represent an inverse qkernel/block as a first-class IR operation.

    The operation stores both the original forward block and a Qamomile-built
    inverse implementation block. Emitters may use ``source_block`` with a
    backend-native inverse/adjoint primitive, then fall back to
    ``implementation_block`` when native inversion is unavailable.

    Operands are ordered as scalar control qubits, target quantum operands,
    then classical/object parameters. Results mirror the quantum operand
    layout: control results first, then one target result per target operand.
    Vector target operands therefore count as one operand/result while
    contributing their scalar width to ``num_target_qubits``.

    Attributes:
        num_control_qubits (int): Number of leading scalar control operands
            and pass-through control results.
        num_target_qubits (int): Scalar qubit width occupied by target
            operands at emit time. Vector operands count by static scalar
            width here but still produce one vector result operand.
        custom_name (str): Human-readable inverse operation name.
        source_block (Block): Forward block whose inverse should be emitted.
        implementation_block (Block): Fallback block that already contains
            the gate-by-gate inverse implementation.
        callable_ref (CallableRef | None): Stable identity of the source
            callable being inverted.
        callable_attrs (dict[str, Any]): Serializer-friendly attrs copied from
            the source callable definition.
        control_value (int | None): LSB-first activation value for the leading
            controls. ``None`` is the ordinary all-ones state.
    """

    num_control_qubits: int = 0
    num_target_qubits: int = 0
    custom_name: str = ""
    source_block: Block | None = None
    implementation_block: Block | None = None
    callable_ref: CallableRef | None = None
    callable_attrs: dict[str, Any] = dataclasses.field(default_factory=dict)
    control_value: int | None = None

    def __post_init__(self) -> None:
        """Validate inverse-block operand and result layout invariants.

        Raises:
            TypeError: If ``control_value`` is not a Python ``int`` or
                ``None``.
            ValueError: If control operands are not scalar quantum values, if
                a quantum target operand appears after a non-quantum
                parameter, or if the results do not mirror the quantum
                operand layout (one quantum result per control operand
                followed by one per target operand), or if ``control_value``
                does not fit the control width.
        """
        if self.num_control_qubits < 0 or self.num_target_qubits < 0:
            raise ValueError("inverse block qubit counts must be non-negative.")
        if self.num_control_qubits == 0:
            if self.control_value is not None:
                raise ValueError(
                    "inverse block control_value requires at least one control."
                )
        else:
            self.control_value = normalize_control_value(
                self.control_value,
                self.num_control_qubits,
            )
        if self.num_control_qubits > len(self.operands):
            raise ValueError(
                "inverse block control count exceeds the number of operands."
            )

        for operand in self.operands[: self.num_control_qubits]:
            if isinstance(operand, ArrayValue):
                raise ValueError(
                    "inverse block control operands must be scalar qubits."
                )
            if not operand.type.is_quantum():
                raise ValueError("inverse block control operands must be quantum.")

        seen_parameter = False
        for operand in self.operands[self.num_control_qubits :]:
            if operand.type.is_quantum():
                if seen_parameter:
                    raise ValueError(
                        "inverse block quantum target operands must precede "
                        "classical/object parameters."
                    )
            else:
                seen_parameter = True

        # Results must mirror the quantum operand layout: downstream
        # passes pair operands with results by ``zip``, so a missing or
        # extra result would otherwise be silently part-processed.
        num_targets = len(self.target_qubits)
        expected_results = self.num_control_qubits + num_targets
        if len(self.results) != expected_results:
            raise ValueError(
                "inverse block results must mirror the quantum operand "
                f"layout: expected {expected_results} "
                f"({self.num_control_qubits} control + {num_targets} "
                f"target), got {len(self.results)}."
            )
        quantum_operands = [
            *self.operands[: self.num_control_qubits],
            *self.target_qubits,
        ]
        for operand, result in zip(quantum_operands, self.results):
            if not result.type.is_quantum():
                raise ValueError("inverse block results must be quantum values.")
            if isinstance(operand, ArrayValue) != isinstance(result, ArrayValue):
                raise ValueError(
                    "inverse block results must mirror operand array-ness: "
                    f"operand {operand.name!r} and result {result.name!r} "
                    "disagree on being a vector."
                )

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
            list[Value]: Quantum operands consumed by the inverse operation
                after control operands. A vector operand counts as one
                operand here even though ``num_target_qubits`` stores its
                scalar backend width.
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
        """Return classical/object parameter operands.

        Returns:
            list[Value]: Non-quantum operands after the quantum targets.
        """
        start = self.num_control_qubits + len(self.target_qubits)
        return list(self.operands[start:])

    @property
    def name(self) -> str:
        """Return a human-readable inverse operation name.

        Returns:
            str: Explicit custom name, or ``"inverse"`` when unnamed.
        """
        return self.custom_name or "inverse"

    @property
    def signature(self) -> Signature:
        """Return the operation signature.

        Returns:
            Signature: Signature with source/fallback block hints, target
                qubit operands, parameter operands, and target qubit results.
        """
        operand_hints: list[ParamHint | None] = [
            ParamHint("source", BlockType()),
            ParamHint("implementation", BlockType()),
        ]
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
