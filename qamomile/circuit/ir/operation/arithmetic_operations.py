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
    MIN = enum.auto()


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


class RuntimeOpKind(enum.Enum):
    """Unified kind for ``RuntimeClassicalExpr`` covering all classical
    op families that can appear at runtime.

    The split between this enum and the per-family ``BinOpKind`` /
    ``CompOpKind`` / ``CondOpKind`` is intentional: compile-time-foldable
    classical ops keep their original IR types so the existing fold
    pipeline (``constant_fold`` → ``compile_time_if_lowering`` → emit-time
    ``evaluate_classical_predicate``) is undisturbed. Only ops identified
    as runtime-evaluation-only by ``ClassicalLoweringPass`` get rewritten
    to ``RuntimeClassicalExpr`` with a member of this enum.
    """

    # Comparison
    EQ = enum.auto()
    NEQ = enum.auto()
    LT = enum.auto()
    LE = enum.auto()
    GT = enum.auto()
    GE = enum.auto()
    # Logical
    AND = enum.auto()
    OR = enum.auto()
    NOT = enum.auto()
    # Arithmetic
    ADD = enum.auto()
    SUB = enum.auto()
    MUL = enum.auto()
    DIV = enum.auto()
    FLOORDIV = enum.auto()
    POW = enum.auto()


@dataclasses.dataclass
class RuntimeClassicalExpr(Operation):
    """A classical expression known to require runtime evaluation.

    Lowered from ``CompOp`` / ``CondOp`` / ``NotOp`` / ``BinOp`` by
    ``ClassicalLoweringPass`` when the op's operand dataflow traces back
    to a ``MeasureOperation`` (i.e. cannot be folded at compile-time, by
    emit-time loop unrolling, or by ``compile_time_if_lowering``). Backend
    emit translates this 1:1 to a backend-native runtime expression
    (e.g. ``qiskit.circuit.classical.expr.Expr``).

    Operand convention:
    - Binary kinds (EQ/NEQ/LT/LE/GT/GE/AND/OR/ADD/SUB/MUL/DIV/FLOORDIV/POW):
      ``operands = [lhs, rhs]``.
    - Unary kind (NOT): ``operands = [val]``.
    - Result: ``results = [output_value]``.

    The single-node + unified-kind shape (vs four parallel subclasses)
    keeps the backend dispatch a single ``match op.kind`` instead of four
    parallel hooks, and makes the IR self-documenting: a single
    ``RuntimeClassicalExpr`` instance signals "runtime evaluation
    required" regardless of which classical family it came from.
    """

    kind: RuntimeOpKind | None = None

    def __post_init__(self):
        if self.kind is None:
            raise ValueError("kind must be specified for RuntimeClassicalExpr")

    @property
    def signature(self) -> Signature:
        if self.kind is RuntimeOpKind.NOT:
            return Signature(
                operands=[ParamHint(name="input", type=self.operands[0].type)],
                results=[ParamHint(name="output", type=self.results[0].type)],
            )
        return Signature(
            operands=[
                ParamHint(name="lhs", type=self.operands[0].type),
                ParamHint(name="rhs", type=self.operands[1].type),
            ],
            results=[ParamHint(name="output", type=self.results[0].type)],
        )

    @property
    def operation_kind(self) -> OperationKind:
        return OperationKind.CLASSICAL


# Mapping from the per-family kinds to ``RuntimeOpKind``. Used by
# ``ClassicalLoweringPass`` to translate compile-time IR ops to their
# runtime equivalent in one place.
_BINOP_KIND_TO_RUNTIME: dict[BinOpKind, RuntimeOpKind] = {
    BinOpKind.ADD: RuntimeOpKind.ADD,
    BinOpKind.SUB: RuntimeOpKind.SUB,
    BinOpKind.MUL: RuntimeOpKind.MUL,
    BinOpKind.DIV: RuntimeOpKind.DIV,
    BinOpKind.FLOORDIV: RuntimeOpKind.FLOORDIV,
    BinOpKind.POW: RuntimeOpKind.POW,
}


def runtime_kind_from_binop(kind: BinOpKind) -> RuntimeOpKind:
    """Map a ``BinOpKind`` to its ``RuntimeOpKind`` counterpart."""
    return _BINOP_KIND_TO_RUNTIME[kind]


_COMPOP_KIND_TO_RUNTIME: dict["CompOpKind", RuntimeOpKind] = {}
_CONDOP_KIND_TO_RUNTIME: dict["CondOpKind", RuntimeOpKind] = {}


def runtime_kind_from_compop(kind: "CompOpKind") -> RuntimeOpKind:
    """Map a ``CompOpKind`` to its ``RuntimeOpKind`` counterpart."""
    if not _COMPOP_KIND_TO_RUNTIME:
        _COMPOP_KIND_TO_RUNTIME.update(
            {
                CompOpKind.EQ: RuntimeOpKind.EQ,
                CompOpKind.NEQ: RuntimeOpKind.NEQ,
                CompOpKind.LT: RuntimeOpKind.LT,
                CompOpKind.LE: RuntimeOpKind.LE,
                CompOpKind.GT: RuntimeOpKind.GT,
                CompOpKind.GE: RuntimeOpKind.GE,
            }
        )
    return _COMPOP_KIND_TO_RUNTIME[kind]


def runtime_kind_from_condop(kind: "CondOpKind") -> RuntimeOpKind:
    """Map a ``CondOpKind`` to its ``RuntimeOpKind`` counterpart."""
    if not _CONDOP_KIND_TO_RUNTIME:
        _CONDOP_KIND_TO_RUNTIME.update(
            {
                CondOpKind.AND: RuntimeOpKind.AND,
                CondOpKind.OR: RuntimeOpKind.OR,
            }
        )
    return _CONDOP_KIND_TO_RUNTIME[kind]


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
