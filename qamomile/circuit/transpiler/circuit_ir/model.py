"""Immutable circuit IR nodes and the private mutable construction builder."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping
from typing import Any, TypeAlias

from qamomile.circuit.transpiler.gate_emitter import GateKind


@dataclasses.dataclass(frozen=True, order=True)
class WireId:
    """Identify one version of a virtual quantum wire.

    Args:
        value (int): Non-negative module-local wire number.
    """

    value: int


class BinaryOperator(enum.Enum):
    """Enumerate scalar operations preserved until target materialization."""

    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    FLOORDIV = "floordiv"
    MOD = "mod"
    POW = "pow"
    EQ = "eq"
    NEQ = "neq"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    AND = "and"
    OR = "or"


class UnaryOperator(enum.Enum):
    """Enumerate unary scalar operations preserved for materialization."""

    NOT = "not"
    NEG = "neg"


class _ScalarOperators:
    """Provide target-neutral arithmetic construction for scalar expressions."""

    def _binary(
        self,
        operator: BinaryOperator,
        other: ScalarExpr | bool | int | float,
    ) -> BinaryExpr:
        """Build a binary expression from this scalar expression.

        Args:
            operator (BinaryOperator): Operation to apply.
            other (ScalarExpr | bool | int | float): Right operand.

        Returns:
            BinaryExpr: Target-neutral binary expression.
        """
        return BinaryExpr(operator, self, as_scalar_expr(other))  # type: ignore[arg-type]

    def __add__(self, other: ScalarExpr | bool | int | float) -> BinaryExpr:
        """Build addition with a right operand.

        Args:
            other (ScalarExpr | bool | int | float): Right operand.

        Returns:
            BinaryExpr: Addition expression.
        """
        return self._binary(BinaryOperator.ADD, other)

    def __sub__(self, other: ScalarExpr | bool | int | float) -> BinaryExpr:
        """Build subtraction with a right operand.

        Args:
            other (ScalarExpr | bool | int | float): Right operand.

        Returns:
            BinaryExpr: Subtraction expression.
        """
        return self._binary(BinaryOperator.SUB, other)

    def __mul__(self, other: ScalarExpr | bool | int | float) -> BinaryExpr:
        """Build multiplication with a right operand.

        Args:
            other (ScalarExpr | bool | int | float): Right operand.

        Returns:
            BinaryExpr: Multiplication expression.
        """
        return self._binary(BinaryOperator.MUL, other)

    def __truediv__(self, other: ScalarExpr | bool | int | float) -> BinaryExpr:
        """Build division with a right operand.

        Args:
            other (ScalarExpr | bool | int | float): Right operand.

        Returns:
            BinaryExpr: Division expression.
        """
        return self._binary(BinaryOperator.DIV, other)

    def __floordiv__(self, other: ScalarExpr | bool | int | float) -> BinaryExpr:
        """Build floor division with a right operand.

        Args:
            other (ScalarExpr | bool | int | float): Right operand.

        Returns:
            BinaryExpr: Floor-division expression.
        """
        return self._binary(BinaryOperator.FLOORDIV, other)

    def __mod__(self, other: ScalarExpr | bool | int | float) -> BinaryExpr:
        """Build modulo with a right operand.

        Args:
            other (ScalarExpr | bool | int | float): Right operand.

        Returns:
            BinaryExpr: Modulo expression.
        """
        return self._binary(BinaryOperator.MOD, other)

    def __pow__(self, other: ScalarExpr | bool | int | float) -> BinaryExpr:
        """Build exponentiation with a right operand.

        Args:
            other (ScalarExpr | bool | int | float): Exponent.

        Returns:
            BinaryExpr: Exponentiation expression.
        """
        return self._binary(BinaryOperator.POW, other)

    def __radd__(self, other: bool | int | float) -> BinaryExpr:
        """Build addition with a left Python scalar.

        Args:
            other (bool | int | float): Left operand.

        Returns:
            BinaryExpr: Addition expression.
        """
        return BinaryExpr(BinaryOperator.ADD, as_scalar_expr(other), self)  # type: ignore[arg-type]

    def __rsub__(self, other: bool | int | float) -> BinaryExpr:
        """Build subtraction from a left Python scalar.

        Args:
            other (bool | int | float): Left operand.

        Returns:
            BinaryExpr: Subtraction expression.
        """
        return BinaryExpr(BinaryOperator.SUB, as_scalar_expr(other), self)  # type: ignore[arg-type]

    def __rmul__(self, other: bool | int | float) -> BinaryExpr:
        """Build multiplication with a left Python scalar.

        Args:
            other (bool | int | float): Left operand.

        Returns:
            BinaryExpr: Multiplication expression.
        """
        return BinaryExpr(BinaryOperator.MUL, as_scalar_expr(other), self)  # type: ignore[arg-type]

    def __rtruediv__(self, other: bool | int | float) -> BinaryExpr:
        """Build division of a left Python scalar by this expression.

        Args:
            other (bool | int | float): Left operand.

        Returns:
            BinaryExpr: Division expression.
        """
        return BinaryExpr(BinaryOperator.DIV, as_scalar_expr(other), self)  # type: ignore[arg-type]


@dataclasses.dataclass(frozen=True)
class LiteralExpr(_ScalarOperators):
    """Represent a concrete scalar literal.

    Args:
        value (bool | int | float): Concrete scalar value.
    """

    value: bool | int | float


@dataclasses.dataclass(frozen=True)
class ParameterExpr(_ScalarOperators):
    """Reference a runtime circuit parameter.

    Args:
        name (str): Stable external parameter name.
    """

    name: str

    def _binary(
        self,
        operator: BinaryOperator,
        other: ScalarExpr | bool | int | float,
    ) -> BinaryExpr:
        """Build a binary expression rooted at this parameter.

        Args:
            operator (BinaryOperator): Operation to apply.
            other (ScalarExpr | bool | int | float): Right operand.

        Returns:
            BinaryExpr: Target-neutral binary expression.
        """
        return BinaryExpr(operator, self, as_scalar_expr(other))

    def __add__(self, other: ScalarExpr | bool | int | float) -> BinaryExpr:
        """Return an addition expression.

        Args:
            other (ScalarExpr | bool | int | float): Right operand.

        Returns:
            BinaryExpr: Addition of this parameter and ``other``.
        """
        return self._binary(BinaryOperator.ADD, other)

    def __sub__(self, other: ScalarExpr | bool | int | float) -> BinaryExpr:
        """Return a subtraction expression.

        Args:
            other (ScalarExpr | bool | int | float): Right operand.

        Returns:
            BinaryExpr: Subtraction of ``other`` from this parameter.
        """
        return self._binary(BinaryOperator.SUB, other)

    def __mul__(self, other: ScalarExpr | bool | int | float) -> BinaryExpr:
        """Return a multiplication expression.

        Args:
            other (ScalarExpr | bool | int | float): Right operand.

        Returns:
            BinaryExpr: Multiplication of this parameter and ``other``.
        """
        return self._binary(BinaryOperator.MUL, other)

    def __truediv__(self, other: ScalarExpr | bool | int | float) -> BinaryExpr:
        """Return a division expression.

        Args:
            other (ScalarExpr | bool | int | float): Right operand.

        Returns:
            BinaryExpr: Division of this parameter by ``other``.
        """
        return self._binary(BinaryOperator.DIV, other)

    def __floordiv__(self, other: ScalarExpr | bool | int | float) -> BinaryExpr:
        """Return a floor-division expression.

        Args:
            other (ScalarExpr | bool | int | float): Right operand.

        Returns:
            BinaryExpr: Floor division of this parameter by ``other``.
        """
        return self._binary(BinaryOperator.FLOORDIV, other)

    def __mod__(self, other: ScalarExpr | bool | int | float) -> BinaryExpr:
        """Return a modulo expression.

        Args:
            other (ScalarExpr | bool | int | float): Right operand.

        Returns:
            BinaryExpr: Modulo of this parameter by ``other``.
        """
        return self._binary(BinaryOperator.MOD, other)

    def __pow__(self, other: ScalarExpr | bool | int | float) -> BinaryExpr:
        """Return a power expression.

        Args:
            other (ScalarExpr | bool | int | float): Exponent.

        Returns:
            BinaryExpr: This parameter raised to ``other``.
        """
        return self._binary(BinaryOperator.POW, other)


@dataclasses.dataclass(frozen=True)
class ClassicalBitExpr(_ScalarOperators):
    """Reference a measured classical bit.

    Args:
        index (int): Circuit-local classical bit index.
    """

    index: int


@dataclasses.dataclass(frozen=True)
class LoopVariableExpr(_ScalarOperators):
    """Reference the induction value of a structured loop.

    Args:
        name (str): Circuit-local loop variable name.
    """

    name: str


@dataclasses.dataclass(frozen=True)
class BinaryExpr(_ScalarOperators):
    """Apply a binary scalar operation.

    Args:
        operator (BinaryOperator): Operation kind.
        left (ScalarExpr): Left operand.
        right (ScalarExpr): Right operand.
    """

    operator: BinaryOperator
    left: ScalarExpr
    right: ScalarExpr


@dataclasses.dataclass(frozen=True)
class UnaryExpr(_ScalarOperators):
    """Apply a unary scalar operation.

    Args:
        operator (UnaryOperator): Operation kind.
        operand (ScalarExpr): Input expression.
    """

    operator: UnaryOperator
    operand: ScalarExpr


ScalarExpr: TypeAlias = (
    LiteralExpr
    | ParameterExpr
    | ClassicalBitExpr
    | LoopVariableExpr
    | BinaryExpr
    | UnaryExpr
)


def as_scalar_expr(value: ScalarExpr | bool | int | float) -> ScalarExpr:
    """Normalize a Python scalar or existing expression.

    Args:
        value (ScalarExpr | bool | int | float): Value to normalize.

    Returns:
        ScalarExpr: Existing expression or a new literal expression.
    """
    if isinstance(value, (bool, int, float)):
        return LiteralExpr(value)
    return value


@dataclasses.dataclass(frozen=True)
class GateInstruction:
    """Apply one primitive gate to versioned virtual wires.

    Args:
        kind (GateKind): Primitive gate kind.
        inputs (tuple[WireId, ...]): Consumed wire versions.
        outputs (tuple[WireId, ...]): Produced wire versions.
        parameters (tuple[ScalarExpr, ...]): Gate parameters.
    """

    kind: GateKind
    inputs: tuple[WireId, ...]
    outputs: tuple[WireId, ...]
    parameters: tuple[ScalarExpr, ...] = ()


@dataclasses.dataclass(frozen=True)
class MeasureInstruction:
    """Measure a wire into a classical bit.

    Args:
        input (WireId): Measured wire version.
        output (WireId): Post-measurement wire version.
        clbit (int): Destination classical bit index.
    """

    input: WireId
    output: WireId
    clbit: int


@dataclasses.dataclass(frozen=True)
class MeasureVectorInstruction:
    """Measure an ordered group of wires into classical bits.

    This instruction preserves vector measurement as one semantic operation
    until target materialization. A backend with a vector measurement
    primitive can consume it directly; scalar-only backends expand it at
    their own boundary.

    Args:
        inputs (tuple[WireId, ...]): Measured wire versions in result order.
        outputs (tuple[WireId, ...]): Post-measurement wire versions.
        clbits (tuple[int, ...]): Destination classical bits in result order.
    """

    inputs: tuple[WireId, ...]
    outputs: tuple[WireId, ...]
    clbits: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class ResetInstruction:
    """Reset a wire and produce a fresh zero-state wire.

    Args:
        input (WireId): Wire version before reset.
        output (WireId): Fresh wire version after reset.
    """

    input: WireId
    output: WireId


@dataclasses.dataclass(frozen=True)
class BarrierInstruction:
    """Separate scheduling regions without changing wire versions.

    Args:
        wires (tuple[WireId, ...]): Wires participating in the barrier.
    """

    wires: tuple[WireId, ...]


class PauliEvolutionRealization(enum.Enum):
    """Enumerate legalization states for abstract Pauli evolution."""

    ABSTRACT = "abstract"
    NATIVE = "native"
    GADGET = "gadget"


@dataclasses.dataclass(frozen=True)
class PauliEvolutionInstruction:
    """Apply an abstract Hamiltonian evolution to selected wires.

    Args:
        hamiltonian (Any): Immutable Qamomile Hamiltonian value.
        time (ScalarExpr): Evolution time in radians.
        inputs (tuple[WireId, ...]): Consumed wire versions.
        outputs (tuple[WireId, ...]): Produced wire versions.
        realization (PauliEvolutionRealization): Target realization selected by
            legalization. Defaults to ``ABSTRACT`` during shared lowering.
    """

    hamiltonian: Any
    time: ScalarExpr
    inputs: tuple[WireId, ...]
    outputs: tuple[WireId, ...]
    realization: PauliEvolutionRealization = PauliEvolutionRealization.ABSTRACT


@dataclasses.dataclass(frozen=True, order=True)
class SemanticOpKey:
    """Identify an abstract operation independently of any backend.

    The key is deliberately open rather than an enum. Standard-library,
    algorithm, provider, and user callables can therefore participate in
    native realization without modifying the compiler's closed vocabulary.

    Args:
        namespace (str): Stable owner namespace such as ``qamomile.stdlib``.
        name (str): Stable operation name within the namespace.
        version (str): Semantic contract version. Defaults to ``"1"``.
        variant (str | None): Optional exact semantic variant, such as a
            decomposition strategy. Defaults to ``None``.
    """

    namespace: str
    name: str
    version: str = "1"
    variant: str | None = None


QFT_SEMANTIC_KEY = SemanticOpKey("qamomile.stdlib", "qft")
"""Semantic key for the exact standard quantum Fourier transform."""

IQFT_SEMANTIC_KEY = SemanticOpKey("qamomile.stdlib", "iqft")
"""Semantic key for the exact inverse quantum Fourier transform."""

STATE_PREPARATION_SEMANTIC_KEY = SemanticOpKey(
    "qamomile.stdlib",
    "state_preparation",
)
"""Semantic key for preparing one concrete normalized state vector."""

RIPPLE_CARRY_ADD_SEMANTIC_KEY = SemanticOpKey(
    "qamomile.stdlib",
    "ripple_carry_add",
)
"""Semantic key for the full reversible ripple-carry adder."""

MULTI_CONTROLLED_X_SEMANTIC_KEY = SemanticOpKey(
    "qamomile.stdlib",
    "multi_controlled_x",
)
"""Semantic key for an arbitrary-width multi-controlled X operation."""


SemanticValue: TypeAlias = None | bool | int | float | str | tuple["SemanticValue", ...]


def _freeze_semantic_value(value: Any) -> SemanticValue:
    """Convert serializer-friendly metadata to an immutable semantic value.

    Args:
        value (Any): Primitive or nested list, tuple, or mapping value.

    Returns:
        SemanticValue: Immutable representation suitable for circuit IR.

    Raises:
        TypeError: If the value cannot cross the semantic circuit boundary.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_semantic_value(item) for item in value)
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_semantic_value(item))
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        )
    raise TypeError(
        f"Semantic argument values must be serializer-friendly primitives, "
        f"received {type(value).__name__}"
    )


@dataclasses.dataclass(frozen=True)
class SemanticArguments:
    """Store immutable named arguments belonging to an operation's meaning.

    Args:
        entries (tuple[tuple[str, SemanticValue], ...]): Sorted name-value
            entries. Defaults to an empty tuple.
    """

    entries: tuple[tuple[str, SemanticValue], ...] = ()

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "SemanticArguments":
        """Freeze one mapping of semantic operation arguments.

        Args:
            values (Mapping[str, Any] | None): Serializer-friendly arguments,
                or ``None`` for no arguments.

        Returns:
            SemanticArguments: Immutable, deterministically ordered arguments.

        Raises:
            TypeError: If a nested value is not serializer-friendly.
        """
        if not values:
            return cls()
        return cls(
            tuple(
                (str(key), _freeze_semantic_value(value))
                for key, value in sorted(values.items())
            )
        )

    def get(self, name: str, default: SemanticValue = None) -> SemanticValue:
        """Return one semantic argument by name.

        Args:
            name (str): Argument name.
            default (SemanticValue): Value returned when absent. Defaults to
                ``None``.

        Returns:
            SemanticValue: Stored value or ``default``.
        """
        for key, value in self.entries:
            if key == name:
                return value
        return default

    def names(self) -> frozenset[str]:
        """Return all semantic argument names.

        Returns:
            frozenset[str]: Immutable set of argument names.
        """
        return frozenset(key for key, _ in self.entries)


@dataclasses.dataclass(frozen=True)
class CallableIdentity:
    """Preserve the semantic identity of a reusable circuit body.

    Args:
        key (SemanticOpKey): Open semantic identity used by target-native
            realization registries.
        symbol (str): Human-readable callable name used for diagnostics.
        arguments (SemanticArguments): Immutable arguments that define this
            invocation's meaning. Defaults to no arguments.
    """

    key: SemanticOpKey
    symbol: str
    arguments: SemanticArguments = SemanticArguments()


@dataclasses.dataclass(frozen=True)
class ReusableCircuit:
    """Describe a reusable circuit body and requested transforms.

    Args:
        body (CircuitProgram): Reusable circuit body.
        name (str): Display and linkage name.
        power (int): Integral repetition count. Defaults to one.
        controls (int): Added control-wire count. Defaults to zero.
        inverse (bool): Whether to apply the inverse body. Defaults to false.
        identity (CallableIdentity | None): Semantic identity preserved for
            target legalization. ``None`` marks an anonymous body. Defaults
            to ``None``.
        native_realization (str | None): Target-owned realization identifier
            selected during legalization. ``None`` keeps the reusable body as
            the fallback implementation. Defaults to ``None``.
        operand_widths (tuple[int, ...]): Flattened width of each semantic
            quantum operand before backend lowering. A vector contributes its
            element count and a scalar qubit contributes one. An empty tuple
            means the source boundary did not expose operand grouping.
    """

    body: CircuitProgram
    name: str
    power: int = 1
    controls: int = 0
    inverse: bool = False
    identity: CallableIdentity | None = None
    native_realization: str | None = None
    operand_widths: tuple[int, ...] = ()

    @property
    def num_qubits(self) -> int:
        """Return the transformed call arity.

        Returns:
            int: Body qubits plus deferred control wires.
        """
        return self.body.num_qubits + self.controls


@dataclasses.dataclass(frozen=True)
class CallInstruction:
    """Invoke a reusable circuit over versioned wires.

    Args:
        callee (ReusableCircuit): Reusable circuit and transforms.
        inputs (tuple[WireId, ...]): Consumed wire versions.
        outputs (tuple[WireId, ...]): Produced wire versions.
    """

    callee: ReusableCircuit
    inputs: tuple[WireId, ...]
    outputs: tuple[WireId, ...]


@dataclasses.dataclass(frozen=True)
class ForInstruction:
    """Repeat a structured circuit region over a concrete range.

    Args:
        indexset (range): Concrete iteration range.
        loop_variable (LoopVariableExpr): Induction expression used by the body.
        inputs (tuple[WireId, ...]): Wire versions entering the loop.
        body (tuple[CircuitInstruction, ...]): Single-iteration body.
        body_outputs (tuple[WireId, ...]): Body wire versions yielded to the
            next iteration.
        outputs (tuple[WireId, ...]): Wire versions after the loop.
    """

    indexset: range
    loop_variable: LoopVariableExpr
    inputs: tuple[WireId, ...]
    body: tuple[CircuitInstruction, ...]
    body_outputs: tuple[WireId, ...]
    outputs: tuple[WireId, ...]


@dataclasses.dataclass(frozen=True)
class IfInstruction:
    """Select between two structured circuit regions.

    Args:
        condition (ScalarExpr): Runtime branch predicate.
        inputs (tuple[WireId, ...]): Wires entering both branches.
        true_body (tuple[CircuitInstruction, ...]): True branch body.
        false_body (tuple[CircuitInstruction, ...]): False branch body.
        true_outputs (tuple[WireId, ...]): Wires yielded by the true branch.
        false_outputs (tuple[WireId, ...]): Wires yielded by the false branch.
        outputs (tuple[WireId, ...]): Merged post-branch wires.
    """

    condition: ScalarExpr
    inputs: tuple[WireId, ...]
    true_body: tuple[CircuitInstruction, ...]
    false_body: tuple[CircuitInstruction, ...]
    true_outputs: tuple[WireId, ...]
    false_outputs: tuple[WireId, ...]
    outputs: tuple[WireId, ...]


@dataclasses.dataclass(frozen=True)
class WhileInstruction:
    """Repeat a structured region while a runtime predicate is true.

    Args:
        condition (ScalarExpr): Runtime loop predicate.
        inputs (tuple[WireId, ...]): Wires entering the loop.
        body (tuple[CircuitInstruction, ...]): Loop body.
        body_outputs (tuple[WireId, ...]): Wires yielded to the next iteration.
        outputs (tuple[WireId, ...]): Wires available after loop termination.
    """

    condition: ScalarExpr
    inputs: tuple[WireId, ...]
    body: tuple[CircuitInstruction, ...]
    body_outputs: tuple[WireId, ...]
    outputs: tuple[WireId, ...]


CircuitInstruction: TypeAlias = (
    GateInstruction
    | MeasureInstruction
    | MeasureVectorInstruction
    | ResetInstruction
    | BarrierInstruction
    | PauliEvolutionInstruction
    | CallInstruction
    | ForInstruction
    | IfInstruction
    | WhileInstruction
)


@dataclasses.dataclass(frozen=True)
class CircuitProgram:
    """Store one immutable backend-neutral circuit program.

    Args:
        name (str): Circuit entrypoint name.
        num_qubits (int): Number of virtual input qubit slots.
        num_clbits (int): Number of classical bit slots.
        input_wires (tuple[WireId, ...]): Initial wire version per qubit slot.
        output_wires (tuple[WireId, ...]): Final wire version per qubit slot.
        operations (tuple[CircuitInstruction, ...]): Structured instruction
            sequence.
        global_phase (ScalarExpr): Program-level phase in radians. Defaults to
            zero and becomes observable when the program is controlled.
    """

    name: str
    num_qubits: int
    num_clbits: int
    input_wires: tuple[WireId, ...]
    output_wires: tuple[WireId, ...]
    operations: tuple[CircuitInstruction, ...]
    global_phase: ScalarExpr = dataclasses.field(
        default_factory=lambda: LiteralExpr(0.0)
    )


@dataclasses.dataclass
class _RegionState:
    """Track one mutable builder region.

    Args:
        operations (list[CircuitInstruction]): Instructions appended to the
            region.
        wires (dict[int, WireId]): Current wire version per physical slot.
    """

    operations: list[CircuitInstruction]
    wires: dict[int, WireId]


@dataclasses.dataclass
class _ForContext:
    """Track construction state for one structured for loop.

    Args:
        indexset (range): Concrete loop range.
        loop_variable (LoopVariableExpr): Body induction expression.
        inputs (dict[int, WireId]): Wire state entering the loop.
    """

    indexset: range
    loop_variable: LoopVariableExpr
    inputs: dict[int, WireId]


@dataclasses.dataclass
class _IfContext:
    """Track construction state for one structured conditional.

    Args:
        condition (ScalarExpr): Runtime branch predicate.
        inputs (dict[int, WireId]): Wire state entering both branches.
        true_region (_RegionState | None): Completed true branch, when an
            else branch has started. Defaults to ``None``.
    """

    condition: ScalarExpr
    inputs: dict[int, WireId]
    true_region: _RegionState | None = None


@dataclasses.dataclass
class _WhileContext:
    """Track construction state for one structured while loop.

    Args:
        condition (ScalarExpr): Runtime loop predicate.
        inputs (dict[int, WireId]): Wire state entering the loop.
    """

    condition: ScalarExpr
    inputs: dict[int, WireId]


class CircuitBuilder:
    """Build immutable circuit IR while assigning fresh wire versions.

    Args:
        num_qubits (int): Number of virtual qubit slots.
        num_clbits (int): Number of classical bit slots.
        name (str): Circuit name. Defaults to ``"main"``.
    """

    def __init__(self, num_qubits: int, num_clbits: int, name: str = "main") -> None:
        """Initialize a circuit builder.

        Args:
            num_qubits (int): Number of virtual qubit slots.
            num_clbits (int): Number of classical bit slots.
            name (str): Circuit name. Defaults to ``"main"``.

        Raises:
            ValueError: If either slot count is negative.
        """
        if num_qubits < 0 or num_clbits < 0:
            raise ValueError("Circuit slot counts must be non-negative")
        self.name = name
        self.num_qubits = num_qubits
        self.num_clbits = num_clbits
        self._next_wire = num_qubits
        input_wires = {index: WireId(index) for index in range(num_qubits)}
        self._input_wires = tuple(input_wires.values())
        self._regions = [_RegionState([], input_wires)]
        self._controls: list[_ForContext | _IfContext | _WhileContext] = []
        self._global_phase: ScalarExpr = LiteralExpr(0.0)

    @property
    def operations(self) -> list[CircuitInstruction]:
        """Return the current region instruction list.

        Returns:
            list[CircuitInstruction]: Mutable list used only during lowering.
        """
        return self._regions[-1].operations

    def current_wire(self, qubit: int) -> WireId:
        """Return the current wire version for a qubit slot.

        Args:
            qubit (int): Physical slot index assigned by circuit lowering.

        Returns:
            WireId: Current version of the slot.

        Raises:
            KeyError: If ``qubit`` is outside the allocated slot range.
        """
        return self._regions[-1].wires[qubit]

    def fresh_wire(self) -> WireId:
        """Allocate a fresh module-local virtual wire version.

        Returns:
            WireId: Newly allocated wire identifier.
        """
        wire = WireId(self._next_wire)
        self._next_wire += 1
        return wire

    def append_gate(
        self,
        kind: GateKind,
        qubits: tuple[int, ...],
        parameters: tuple[ScalarExpr, ...] = (),
    ) -> None:
        """Append a primitive gate and advance all participating wires.

        Args:
            kind (GateKind): Primitive gate kind.
            qubits (tuple[int, ...]): Participating qubit slots.
            parameters (tuple[ScalarExpr, ...]): Gate parameters. Defaults to
                an empty tuple.
        """
        inputs = tuple(self.current_wire(qubit) for qubit in qubits)
        outputs = tuple(self.fresh_wire() for _ in qubits)
        self.operations.append(GateInstruction(kind, inputs, outputs, parameters))
        for qubit, output in zip(qubits, outputs, strict=True):
            self._regions[-1].wires[qubit] = output

    def append_measure(self, qubit: int, clbit: int) -> None:
        """Append a measurement.

        Args:
            qubit (int): Measured qubit slot.
            clbit (int): Destination classical bit slot.

        Raises:
            IndexError: If ``clbit`` is outside the allocated classical slots.
        """
        if clbit < 0 or clbit >= self.num_clbits:
            raise IndexError(f"Classical bit index {clbit} is out of range")
        input_wire = self.current_wire(qubit)
        output_wire = self.fresh_wire()
        self.operations.append(MeasureInstruction(input_wire, output_wire, clbit))
        self._regions[-1].wires[qubit] = output_wire

    def append_measure_vector(
        self,
        qubits: tuple[int, ...],
        clbits: tuple[int, ...],
    ) -> None:
        """Append one ordered vector measurement.

        Args:
            qubits (tuple[int, ...]): Measured qubit slots in result order.
            clbits (tuple[int, ...]): Destination classical slots.

        Raises:
            ValueError: If qubit and classical-bit arities differ or either
                sequence contains duplicate slots.
            IndexError: If a classical-bit slot is outside the circuit.
        """
        if len(qubits) != len(clbits):
            raise ValueError("Vector measurement qubit/clbit arities must match")
        if len(set(qubits)) != len(qubits):
            raise ValueError("Vector measurement qubit slots must be unique")
        if len(set(clbits)) != len(clbits):
            raise ValueError("Vector measurement classical slots must be unique")
        if any(clbit < 0 or clbit >= self.num_clbits for clbit in clbits):
            raise IndexError("Vector measurement classical bit is out of range")
        inputs = tuple(self.current_wire(qubit) for qubit in qubits)
        outputs = tuple(self.fresh_wire() for _ in qubits)
        self.operations.append(MeasureVectorInstruction(inputs, outputs, clbits))
        for qubit, output in zip(qubits, outputs, strict=True):
            self._regions[-1].wires[qubit] = output

    def append_reset(self, qubit: int) -> None:
        """Append reset and advance the affected wire.

        Args:
            qubit (int): Qubit slot to reset.
        """
        input_wire = self.current_wire(qubit)
        output_wire = self.fresh_wire()
        self.operations.append(ResetInstruction(input_wire, output_wire))
        self._regions[-1].wires[qubit] = output_wire

    def append_barrier(self, qubits: tuple[int, ...]) -> None:
        """Append a scheduling barrier without changing wire versions.

        Args:
            qubits (tuple[int, ...]): Participating qubit slots.
        """
        self.operations.append(
            BarrierInstruction(tuple(self.current_wire(qubit) for qubit in qubits))
        )

    def append_pauli_evolution(
        self,
        qubits: tuple[int, ...],
        hamiltonian: Any,
        time: ScalarExpr | bool | int | float,
    ) -> None:
        """Append an abstract Pauli evolution and advance its wires.

        Args:
            qubits (tuple[int, ...]): Participating qubit slots.
            hamiltonian (Any): Qamomile Hamiltonian value.
            time (ScalarExpr | bool | int | float): Evolution time.
        """
        inputs = tuple(self.current_wire(qubit) for qubit in qubits)
        outputs = tuple(self.fresh_wire() for _ in qubits)
        self.operations.append(
            PauliEvolutionInstruction(
                hamiltonian=hamiltonian,
                time=as_scalar_expr(time),
                inputs=inputs,
                outputs=outputs,
            )
        )
        for qubit, output in zip(qubits, outputs, strict=True):
            self._regions[-1].wires[qubit] = output

    def append_call(self, callee: ReusableCircuit, qubits: tuple[int, ...]) -> None:
        """Append a reusable-circuit call and advance its wires.

        Args:
            callee (ReusableCircuit): Reusable circuit and transforms.
            qubits (tuple[int, ...]): Participating qubit slots.
        """
        inputs = tuple(self.current_wire(qubit) for qubit in qubits)
        outputs = tuple(self.fresh_wire() for _ in qubits)
        self.operations.append(CallInstruction(callee, inputs, outputs))
        for qubit, output in zip(qubits, outputs, strict=True):
            self._regions[-1].wires[qubit] = output

    def add_global_phase(
        self,
        phase: ScalarExpr | bool | int | float,
    ) -> None:
        """Accumulate a program-level global phase in radians.

        Args:
            phase (ScalarExpr | bool | int | float): Phase contribution.
        """
        self._global_phase = BinaryExpr(
            BinaryOperator.ADD,
            self._global_phase,
            as_scalar_expr(phase),
        )

    def begin_for(self, indexset: range) -> LoopVariableExpr:
        """Open a structured for-loop body.

        Args:
            indexset (range): Concrete iteration range.

        Returns:
            LoopVariableExpr: Induction expression available inside the body.
        """
        inputs = dict(self._regions[-1].wires)
        variable = LoopVariableExpr(f"loop_{len(self._controls)}")
        self._controls.append(_ForContext(indexset, variable, inputs))
        self._regions.append(_RegionState([], dict(inputs)))
        return variable

    def end_for(self) -> None:
        """Close the innermost structured for-loop body.

        Raises:
            RuntimeError: If the innermost open region is not a for loop.
        """
        if not self._controls or not isinstance(self._controls[-1], _ForContext):
            raise RuntimeError("No structured for loop is open")
        context = self._controls[-1]
        assert isinstance(context, _ForContext)
        self._controls.pop()
        body = self._regions.pop()
        parent = self._regions[-1]
        outputs = {index: self.fresh_wire() for index in range(self.num_qubits)}
        parent.operations.append(
            ForInstruction(
                indexset=context.indexset,
                loop_variable=context.loop_variable,
                inputs=tuple(context.inputs.values()),
                body=tuple(body.operations),
                body_outputs=tuple(body.wires.values()),
                outputs=tuple(outputs.values()),
            )
        )
        parent.wires = outputs

    def begin_if(self, condition: ScalarExpr) -> _IfContext:
        """Open the true region of a structured conditional.

        Args:
            condition (ScalarExpr): Runtime branch predicate.

        Returns:
            _IfContext: Opaque builder token used to select the else branch.
        """
        inputs = dict(self._regions[-1].wires)
        context = _IfContext(condition, inputs)
        self._controls.append(context)
        self._regions.append(_RegionState([], dict(inputs)))
        return context

    def begin_else(self, context: _IfContext) -> None:
        """Close a true region and open its false region.

        Args:
            context (_IfContext): Token returned by :meth:`begin_if`.

        Raises:
            RuntimeError: If ``context`` is not the innermost open conditional
                or an else branch has already started.
        """
        if not self._controls or self._controls[-1] is not context:
            raise RuntimeError("Conditional context is not the innermost region")
        if context.true_region is not None:
            raise RuntimeError("Conditional else region has already started")
        context.true_region = self._regions.pop()
        self._regions.append(_RegionState([], dict(context.inputs)))

    def end_if(self, context: _IfContext) -> None:
        """Close a structured conditional and merge its wire states.

        Args:
            context (_IfContext): Token returned by :meth:`begin_if`.

        Raises:
            RuntimeError: If ``context`` is not the innermost open conditional.
        """
        if not self._controls or self._controls[-1] is not context:
            raise RuntimeError("Conditional context is not the innermost region")
        self._controls.pop()
        current = self._regions.pop()
        if context.true_region is None:
            true_region = current
            false_region = _RegionState([], dict(context.inputs))
        else:
            true_region = context.true_region
            false_region = current
        parent = self._regions[-1]
        outputs = {index: self.fresh_wire() for index in range(self.num_qubits)}
        parent.operations.append(
            IfInstruction(
                condition=context.condition,
                inputs=tuple(context.inputs.values()),
                true_body=tuple(true_region.operations),
                false_body=tuple(false_region.operations),
                true_outputs=tuple(true_region.wires.values()),
                false_outputs=tuple(false_region.wires.values()),
                outputs=tuple(outputs.values()),
            )
        )
        parent.wires = outputs

    def begin_while(self, condition: ScalarExpr) -> _WhileContext:
        """Open a structured while-loop body.

        Args:
            condition (ScalarExpr): Runtime loop predicate.

        Returns:
            _WhileContext: Opaque builder token used to close the loop.
        """
        inputs = dict(self._regions[-1].wires)
        context = _WhileContext(condition, inputs)
        self._controls.append(context)
        self._regions.append(_RegionState([], dict(inputs)))
        return context

    def end_while(self, context: _WhileContext) -> None:
        """Close a structured while-loop body.

        Args:
            context (_WhileContext): Token returned by :meth:`begin_while`.

        Raises:
            RuntimeError: If ``context`` is not the innermost open while loop.
        """
        if not self._controls or self._controls[-1] is not context:
            raise RuntimeError("While context is not the innermost region")
        self._controls.pop()
        body = self._regions.pop()
        parent = self._regions[-1]
        outputs = {index: self.fresh_wire() for index in range(self.num_qubits)}
        parent.operations.append(
            WhileInstruction(
                condition=context.condition,
                inputs=tuple(context.inputs.values()),
                body=tuple(body.operations),
                body_outputs=tuple(body.wires.values()),
                outputs=tuple(outputs.values()),
            )
        )
        parent.wires = outputs

    def freeze(self) -> CircuitProgram:
        """Finalize the root region into immutable circuit IR.

        Returns:
            CircuitProgram: Immutable circuit program.

        Raises:
            RuntimeError: If a structured region is still open.
        """
        if len(self._regions) != 1 or self._controls:
            raise RuntimeError("Cannot freeze a circuit with open structured regions")
        root = self._regions[0]
        return CircuitProgram(
            name=self.name,
            num_qubits=self.num_qubits,
            num_clbits=self.num_clbits,
            input_wires=self._input_wires,
            output_wires=tuple(root.wires[index] for index in range(self.num_qubits)),
            operations=tuple(root.operations),
            global_phase=self._global_phase,
        )
