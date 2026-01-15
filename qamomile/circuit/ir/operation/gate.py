import dataclasses
import enum

from qamomile.circuit.ir.types.primitives import BitType, BlockType, QubitType

from .operation import Operation, OperationKind, ParamHint, Signature


class GateOperationType(enum.Enum):
    H = enum.auto()
    X = enum.auto()
    Y = enum.auto()
    Z = enum.auto()
    T = enum.auto()
    S = enum.auto()
    P = enum.auto()  # Phase gate: P(θ)|1⟩ = e^{iθ}|1⟩
    RX = enum.auto()
    RY = enum.auto()
    RZ = enum.auto()
    CX = enum.auto()
    CZ = enum.auto()
    SWAP = enum.auto()
    CP = enum.auto()
    RZZ = enum.auto()
    TOFFOLI = enum.auto()


@dataclasses.dataclass
class GateOperation(Operation):
    gate_type: GateOperationType | None = None

    def __post_init__(self):
        if not self.gate_type:
            raise ValueError("gate_type must be specified for GateOperation.")

    @property
    def signature(self) -> Signature:
        return Signature(
            operands=[
                ParamHint(name=f"qubit_{i}", type=QubitType())
                for i in range(len(self.operands))
            ],
            results=[
                ParamHint(name=f"qubit_{i}", type=QubitType())
                for i in range(len(self.operands))
            ],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.QUANTUM


@dataclasses.dataclass
class MeasureOperation(Operation):
    @property
    def signature(self) -> Signature:
        return Signature(
            operands=[ParamHint(name="qubit", type=QubitType())],
            results=[ParamHint(name="bit", type=BitType())],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.HYBRID


@dataclasses.dataclass
class ControlledUOperation(Operation):
    """Controlled-U operation that applies a unitary block conditionally.

    The operands structure is:
    - operands[0]: BlockValue (the unitary U to apply)
    - operands[1:1+num_controls]: Control qubits
    - operands[1+num_controls:]: Target qubits (arguments to U)

    The results structure is:
    - results[0:num_controls]: Control qubits (returned)
    - results[num_controls:]: Target qubits (returned from U)
    """

    num_controls: int = 1

    @property
    def block(self):
        """Get the BlockValue (unitary U)."""
        return self.operands[0]

    @property
    def control_operands(self):
        """Get the control qubit values."""
        return self.operands[1 : 1 + self.num_controls]

    @property
    def target_operands(self):
        """Get the target qubit values (arguments to U)."""
        return self.operands[1 + self.num_controls :]

    @property
    def signature(self) -> Signature:
        num_targets = len(self.operands) - 1 - self.num_controls
        return Signature(
            operands=[
                ParamHint("U", BlockType()),
                *[
                    ParamHint(name=f"control_{i}", type=QubitType())
                    for i in range(self.num_controls)
                ],
                *[
                    ParamHint(name=f"target_{i}", type=QubitType())
                    for i in range(num_targets)
                ],
            ],
            results=[
                *[
                    ParamHint(name=f"control_{i}", type=QubitType())
                    for i in range(self.num_controls)
                ],
                *[
                    ParamHint(name=f"target_{i}", type=QubitType())
                    for i in range(num_targets)
                ],
            ],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.QUANTUM
