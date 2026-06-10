"""CompositeGate operation for representing complex multi-gate operations."""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING, Any

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.types.primitives import BlockType, QubitType
from qamomile.circuit.ir.value import ArrayValue

from .operation import Operation, OperationKind, ParamHint, Signature

if TYPE_CHECKING:
    from qamomile.circuit.ir.value import Value


class CompositeGateType(enum.Enum):
    """Registry of known composite gate types."""

    QPE = "qpe"  # Quantum Phase Estimation
    QFT = "qft"  # Quantum Fourier Transform
    IQFT = "iqft"  # Inverse QFT
    CUSTOM = "custom"  # User-defined composite gate


@dataclasses.dataclass
class ResourceMetadata:
    """Resource estimation metadata for composite gates.

    Gate count fields mirror GateCount categories.

    None semantics:
        Fields left as None mean "unknown/unspecified". During extraction,
        gate_counter treats None as 0, which may undercount resources if
        the true value is nonzero. To ensure accurate resource estimates,
        set all relevant fields explicitly.

        When total_gates is set but some of single_qubit_gates,
        two_qubit_gates, or multi_qubit_gates are None, the extractor
        emits a UserWarning if the known sub-total is less than
        total_gates, indicating potentially missing gate category data.

    Attributes:
        query_complexity: Number of oracle/unitary queries per call.
            Used by gate_counter to produce oracle_queries metric.
        t_gates: Estimated T-gate count (None -> 0)
        ancilla_qubits: Number of ancilla qubits required
        total_gates: Total number of gates. If None and sub-categories
            are set, computed as single_qubit + two_qubit + multi_qubit.
        single_qubit_gates: Number of single-qubit gates (None -> 0)
        two_qubit_gates: Number of two-qubit gates (None -> 0)
        multi_qubit_gates: Number of multi-qubit gates, 3+ qubits (None -> 0)
        clifford_gates: Number of Clifford gates (None -> 0)
        rotation_gates: Number of rotation gates (None -> 0)
        custom_metadata: Additional metadata (strategy, precision, etc.)
    """

    query_complexity: int | None = None
    t_gates: int | None = None
    ancilla_qubits: int = 0
    total_gates: int | None = None
    single_qubit_gates: int | None = None
    two_qubit_gates: int | None = None
    multi_qubit_gates: int | None = None
    clifford_gates: int | None = None
    rotation_gates: int | None = None
    custom_metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class CompositeGateOperation(Operation):
    """Represents a composite gate (QPE, QFT, etc.) as a single operation.

    CompositeGate allows representing complex multi-gate operations as a single
    atomic operation in the IR. This enables:
    - Resource estimation without full implementation
    - Backend-native conversion (e.g., Qiskit's QPE)
    - User-defined complex gates

    The operands structure is:
    - operands[0:num_control_qubits]: Control qubits (if any)
    - operands[num_control_qubits:num_control_qubits+num_target_qubits]: Target qubits
    - operands[num_control_qubits+num_target_qubits:]: Parameters

    The results structure:
    - results[0:num_control_qubits]: Control qubits (returned)
    - results[num_control_qubits:]: Target qubits (returned)

    Attributes:
        gate_type: The type of composite gate (QPE, QFT, CUSTOM, etc.)
        num_control_qubits: Number of control qubits
        num_target_qubits: Number of target qubits
        custom_name: Name for CUSTOM gate types (used for identification)
        resource_metadata: Optional resource estimation metadata
        has_implementation: Whether this operation has an implementation Block
        composite_gate_instance: Optional reference to the CompositeGate class instance
            that created this operation (for accessing _decompose() at emit time)
        strategy_name: Optional name of the decomposition strategy to use.
            If None, the default strategy is used during emission.
    """

    gate_type: CompositeGateType = CompositeGateType.CUSTOM
    num_control_qubits: int = 0
    num_target_qubits: int = 0
    custom_name: str = ""
    resource_metadata: ResourceMetadata | None = None
    has_implementation: bool = True
    implementation_block: Block | None = None
    composite_gate_instance: Any = None  # Reference to CompositeGate instance
    strategy_name: str | None = None  # Selected decomposition strategy

    @property
    def implementation(self) -> Block | None:
        """Get the implementation block, if any."""
        if not self.has_implementation:
            return None
        return self.implementation_block

    @property
    def control_qubits(self) -> list["Value"]:
        """Get the control qubit operands."""
        return list(self.operands[: self.num_control_qubits])

    @property
    def target_qubits(self) -> list["Value"]:
        """Get the target qubit operands."""
        start = self.num_control_qubits
        end = start + self.num_target_qubits
        return list(self.operands[start:end])

    @property
    def parameters(self) -> list["Value"]:
        """Get the parameter operands (angles, etc.)."""
        start = self.num_control_qubits + self.num_target_qubits
        return list(self.operands[start:])

    @property
    def name(self) -> str:
        """Human-readable name of this composite gate."""
        if self.gate_type == CompositeGateType.CUSTOM:
            return self.custom_name or "custom"
        return self.gate_type.value

    @property
    def signature(self) -> Signature:
        """Return the operation signature."""
        operand_hints: list[ParamHint | None] = []

        if self.has_implementation:
            operand_hints.append(ParamHint("implementation", BlockType()))

        for i in range(self.num_control_qubits):
            operand_hints.append(ParamHint(f"control_{i}", QubitType()))

        for i in range(self.num_target_qubits):
            operand_hints.append(ParamHint(f"target_{i}", QubitType()))

        # Parameters - use their actual types
        for i, param in enumerate(self.parameters):
            operand_hints.append(ParamHint(f"param_{i}", param.type))

        result_hints = [
            ParamHint(f"control_out_{i}", QubitType())
            for i in range(self.num_control_qubits)
        ] + [
            ParamHint(f"target_out_{i}", QubitType())
            for i in range(self.num_target_qubits)
        ]

        return Signature(operands=operand_hints, results=result_hints)

    @property
    def operation_kind(self) -> OperationKind:
        """Return the operation kind (always QUANTUM)."""
        return OperationKind.QUANTUM


@dataclasses.dataclass
class InverseBlockOperation(Operation):
    """Represent an inverse qkernel/block as a first-class IR operation.

    The operation stores both the original forward block and a Qamomile-built
    inverse implementation block. Emitters may use ``source_block`` with a
    backend-native inverse/adjoint primitive, then fall back to
    ``implementation_block`` when native inversion is unavailable.

    Operands are ordered as control qubits, target quantum operands, then
    classical/object parameters. Results mirror the quantum operand layout:
    control results first, then one target result per target operand. Vector
    target operands therefore count as one operand/result while contributing
    their scalar width to ``num_target_qubits``.

    Attributes:
        num_control_qubits (int): Number of leading control operands and
            pass-through control results.
        num_target_qubits (int): Scalar qubit width occupied by target
            operands at emit time. Vector operands count by static scalar
            width here but still produce one vector result operand.
        custom_name (str): Human-readable inverse operation name.
        source_block (Block): Forward block whose inverse should be emitted.
        implementation_block (Block): Fallback block that already contains
            the gate-by-gate inverse implementation.
    """

    num_control_qubits: int = 0
    num_target_qubits: int = 0
    custom_name: str = ""
    source_block: Block | None = None
    implementation_block: Block | None = None

    def __post_init__(self) -> None:
        """Validate inverse-block operand and result layout invariants.

        Raises:
            ValueError: If control operands are not quantum values, if a
                quantum target operand appears after a non-quantum
                parameter, or if the results do not mirror the quantum
                operand layout (one quantum result per control operand
                followed by one per target operand).
        """
        if self.num_control_qubits < 0 or self.num_target_qubits < 0:
            raise ValueError("inverse block qubit counts must be non-negative.")
        if self.num_control_qubits > len(self.operands):
            raise ValueError(
                "inverse block control count exceeds the number of operands."
            )

        for operand in self.operands[: self.num_control_qubits]:
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
