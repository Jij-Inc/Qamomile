import abc
import enum
from dataclasses import dataclass, field

from qamomile.circuit.ir.types import ValueType
from qamomile.circuit.ir.value import Value


class OperationKind(enum.Enum):
    """Classification of operations for classical/quantum separation.

    This enum is used to categorize operations during compilation to
    determine which parts run on classical hardware vs quantum hardware.

    Values:
        QUANTUM: Pure quantum operations (gates, qubit allocation)
        CLASSICAL: Pure classical operations (arithmetic, comparisons)
        HYBRID: Operations that bridge classical and quantum (measurement, encode/decode)
        CONTROL: Control flow structures (for, while, if)
    """

    QUANTUM = "quantum"
    CLASSICAL = "classical"
    HYBRID = "hybrid"
    CONTROL = "control"


@dataclass
class ParamHint:
    name: str
    type: ValueType


@dataclass
class Signature:
    operands: list[ParamHint | None] = field(default_factory=list)
    results: list[ParamHint] = field(default_factory=list)


@dataclass
class Operation(abc.ABC):
    operands: list[Value] = field(default_factory=list)
    results: list[Value] = field(default_factory=list)

    @property
    @abc.abstractmethod
    def signature(self) -> Signature:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def operation_kind(self) -> OperationKind:
        """Return the kind of this operation for classical/quantum classification."""
        raise NotImplementedError()


# Initialize Operations
@dataclass
class CInitOperation(Operation):
    """Initialize the classical values (const, arguments etc)"""

    @property
    def signature(self) -> Signature:
        # operants []
        # results [value]
        return Signature(
            [], results=[ParamHint(name="value", type=self.results[0].type)]
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CLASSICAL


@dataclass
class QInitOperation(Operation):
    """Initialize the qubit"""

    @property
    def signature(self) -> Signature:
        # operants []
        # results [qubit]
        return Signature(
            [], results=[ParamHint(name="value", type=self.results[0].type)]
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.QUANTUM
