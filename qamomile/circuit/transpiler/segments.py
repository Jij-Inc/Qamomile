"""Data structures for segmented quantum/classical execution plans."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TypeAlias

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.value import Value


class SegmentKind(Enum):
    """Type of computation segment."""

    QUANTUM = auto()
    CLASSICAL = auto()
    EXPVAL = auto()


@dataclasses.dataclass
class Segment(ABC):
    """Base class for separated computation segments."""

    operations: list[Operation] = dataclasses.field(default_factory=list)

    # Values this segment reads from global state
    input_refs: list[str] = dataclasses.field(default_factory=list)

    # Values this segment writes to global state
    output_refs: list[str] = dataclasses.field(default_factory=list)

    @property
    @abstractmethod
    def kind(self) -> SegmentKind:
        """Return the kind of this segment."""
        pass


@dataclasses.dataclass
class QuantumSegment(Segment):
    """A segment of pure quantum operations.

    Contains quantum gates and qubit allocations.
    Will be emitted to a quantum circuit.
    """

    # Qubit values used in this segment
    qubit_values: list[Value] = dataclasses.field(default_factory=list)

    # Number of qubits needed
    num_qubits: int = 0

    @property
    def kind(self) -> SegmentKind:
        return SegmentKind.QUANTUM


@dataclasses.dataclass
class ClassicalSegment(Segment):
    """A segment of pure classical operations.

    Contains arithmetic, comparisons, and control flow.
    Will be executed directly in Python.
    """

    @property
    def kind(self) -> SegmentKind:
        return SegmentKind.CLASSICAL


@dataclasses.dataclass
class ExpvalSegment(Segment):
    """A segment for expectation value computation.

    Represents computing <psi|H|psi> where psi is the quantum state
    and H is a Hamiltonian observable.

    This segment bridges a quantum circuit (state preparation) to
    a classical expectation value.

    Attributes:
        hamiltonian_value: The IR Value representing the Hamiltonian
        qubits_value: The IR Value representing the quantum state
        result_ref: UUID of the result Float value
    """

    hamiltonian_value: Value | None = None
    qubits_value: Value | None = None
    result_ref: str = ""

    @property
    def kind(self) -> SegmentKind:
        return SegmentKind.EXPVAL


@dataclasses.dataclass
class HybridBoundary:
    """Represents a measurement or encode operation at quantum/classical boundary.

    These operations bridge quantum and classical segments.
    """

    operation: Operation

    # Which segment produces the input
    source_segment_index: int

    # Which segment consumes the output
    target_segment_index: int

    # Value being transferred
    value_ref: str


class MultipleQuantumSegmentsError(Exception):
    """Raised when program has multiple quantum segments.

    Qamomile enforces a single quantum circuit execution pattern:
        [Classical Prep] → Quantum Circuit → [Classical Post/Expval]

    Your program has multiple quantum segments, suggesting quantum operations
    that depend on measurement results (JIT compilation not supported).
    """

    pass


@dataclasses.dataclass
class ProgramABI:
    """Runtime-visible ABI for a segmented program."""

    public_inputs: dict[str, Value] = dataclasses.field(default_factory=dict)
    output_refs: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ClassicalStep:
    """A classical execution step."""

    segment: ClassicalSegment
    role: str = "classical"


@dataclasses.dataclass
class QuantumStep:
    """A quantum execution step."""

    segment: QuantumSegment


@dataclasses.dataclass
class ExpvalStep:
    """An expectation-value execution step."""

    segment: ExpvalSegment
    quantum_step_index: int = 0


ProgramStep: TypeAlias = ClassicalStep | QuantumStep | ExpvalStep


@dataclasses.dataclass
class ProgramPlan:
    """Execution plan for a hybrid quantum/classical program.

    Structure:
    - [Optional] Classical preprocessing (parameter computation, etc.)
    - Single quantum segment (REQUIRED)
    - [Optional] Expval segment OR classical postprocessing

    This plan enforces Qamomile's current execution model:
    all quantum operations must be in a single quantum circuit.
    """
    steps: list[ProgramStep] = dataclasses.field(default_factory=list)
    abi: ProgramABI = dataclasses.field(default_factory=ProgramABI)

    # Boundaries for tracking quantum-classical transitions
    boundaries: list[HybridBoundary] = dataclasses.field(default_factory=list)

    # Original parameters
    parameters: dict[str, Value] = dataclasses.field(default_factory=dict)

