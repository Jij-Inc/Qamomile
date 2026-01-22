import dataclasses
import enum

from qamomile.circuit.ir.types.primitives import BitType
from qamomile.circuit.ir.value import Value

from .operation import Operation, OperationKind, ParamHint, Signature


@dataclasses.dataclass
class BinaryOperationBase(Operation):
    """Base for binary operations with lhs, rhs, and output.

    Provides common properties and validation for operations
    that take two operands and produce one result.
    """

    kind: enum.Enum | None = None

    def _validate_kind(self, operation_name: str):
        """Validate that kind is specified. Call from __post_init__."""
        if not self.kind:
            raise ValueError(f"kind must be specified for {operation_name}.")

    @property
    def lhs(self) -> Value:
        """Left-hand side operand."""
        return self.operands[0]

    @property
    def rhs(self) -> Value:
        """Right-hand side operand."""
        return self.operands[1]

    @property
    def output(self) -> Value:
        """Output result."""
        return self.results[0]

    def _create_binary_signature(self) -> Signature:
        """Create signature for binary operation."""
        return Signature(
            operands=[
                ParamHint(name="lhs", type=self.operands[0].type),
                ParamHint(name="rhs", type=self.operands[1].type),
            ],
            results=[ParamHint(name="output", type=self.results[0].type)],
        )


class BinOpKind(enum.Enum):
    ADD = enum.auto()
    SUB = enum.auto()
    MUL = enum.auto()
    DIV = enum.auto()
    FLOORDIV = enum.auto()
    POW = enum.auto()


@dataclasses.dataclass
class BinOp(BinaryOperationBase):
    """Binary arithmetic operation (ADD, SUB, MUL, DIV, FLOORDIV, POW)."""

    kind: BinOpKind | None = None

    def __post_init__(self):
        self._validate_kind("BinOp")

    @property
    def signature(self) -> Signature:
        return self._create_binary_signature()

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CLASSICAL


class CompOpKind(enum.Enum):
    EQ = enum.auto()
    NEQ = enum.auto()
    LT = enum.auto()
    LE = enum.auto()
    GT = enum.auto()
    GE = enum.auto()


@dataclasses.dataclass
class CompOp(BinaryOperationBase):
    """Comparison operation (EQ, NEQ, LT, LE, GT, GE)."""

    kind: CompOpKind | None = None

    def __post_init__(self):
        self._validate_kind("CompOp")

    @property
    def signature(self) -> Signature:
        return self._create_binary_signature()

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CLASSICAL


class CondOpKind(enum.Enum):
    AND = enum.auto()
    OR = enum.auto()


@dataclasses.dataclass
class CondOp(BinaryOperationBase):
    """Conditional logical operation (AND, OR)."""

    kind: CondOpKind | None = None

    def __post_init__(self):
        self._validate_kind("CondOp")

    @property
    def signature(self) -> Signature:
        return self._create_binary_signature()

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CLASSICAL


@dataclasses.dataclass
class NotOp(Operation):
    @property
    def input(self) -> Value:
        return self.operands[0]

    @property
    def output(self) -> Value:
        return self.results[0]

    @property
    def signature(self) -> Signature:
        return Signature(
            operands=[ParamHint(name="input", type=self.operands[0].type)],
            results=[ParamHint(name="output", type=self.results[0].type)],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CLASSICAL


@dataclasses.dataclass
class PhiOp(Operation):
    """SSA Phi function: merge point after conditional branch.

    This operation selects one of two values based on a condition.
    Used to merge values from different branches of an if-else statement.

    Attributes:
        operands[0]: condition (Bit) - which branch was taken
        operands[1]: true_value - value from the true branch
        operands[2]: false_value - value from the false branch
        results[0]: output - merged value

    Example:
        if condition:
            x = x + 1  # true_value
        else:
            x = x + 2  # false_value
        # x is now PhiOp(condition, true_value, false_value)
    """

    @property
    def condition(self) -> Value:
        return self.operands[0]

    @property
    def true_value(self) -> Value:
        return self.operands[1]

    @property
    def false_value(self) -> Value:
        return self.operands[2]

    @property
    def output(self) -> Value:
        return self.results[0]

    @property
    def signature(self) -> Signature:
        return Signature(
            operands=[
                ParamHint(name="condition", type=BitType()),
                ParamHint(name="true_value", type=self.operands[1].type),
                ParamHint(name="false_value", type=self.operands[2].type),
            ],
            results=[ParamHint(name="output", type=self.results[0].type)],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CLASSICAL
