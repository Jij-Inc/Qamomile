"""Data structures for separated quantum/classical segments."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.value import Value

if TYPE_CHECKING:
    from qamomile.circuit.observable import Observable


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


@dataclasses.dataclass
class SeparatedProgram:
    """Result of the separation pass.

    Contains alternating quantum and classical segments with
    explicit boundaries between them.
    """

    segments: list[Segment] = dataclasses.field(default_factory=list)
    boundaries: list[HybridBoundary] = dataclasses.field(default_factory=list)

    # Original parameters for binding
    parameters: dict[str, Value] = dataclasses.field(default_factory=dict)

    # Final output references
    output_refs: list[str] = dataclasses.field(default_factory=list)

    def quantum_segments(self) -> list[QuantumSegment]:
        """Get all quantum segments."""
        return [s for s in self.segments if isinstance(s, QuantumSegment)]

    def classical_segments(self) -> list[ClassicalSegment]:
        """Get all classical segments."""
        return [s for s in self.segments if isinstance(s, ClassicalSegment)]

    def expval_segments(self) -> list[ExpvalSegment]:
        """Get all expectation value segments."""
        return [s for s in self.segments if isinstance(s, ExpvalSegment)]
