"""Compiled segment structures for transpiled quantum circuits."""

from __future__ import annotations

import dataclasses
from typing import Generic, TypeVar, TYPE_CHECKING

from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata
from qamomile.circuit.transpiler.segments import (
    ClassicalSegment,
    ExpvalSegment,
    QuantumSegment,
)

if TYPE_CHECKING:
    from qamomile.circuit.observable import Observable

T = TypeVar("T")  # Backend circuit type


@dataclasses.dataclass
class CompiledQuantumSegment(Generic[T]):
    """A quantum segment with emitted backend circuit."""

    segment: QuantumSegment
    circuit: T

    # Mapping from Value UUIDs to physical qubit indices
    qubit_map: dict[str, int] = dataclasses.field(default_factory=dict)

    # Mapping from Value UUIDs to classical bit indices (for measurements)
    clbit_map: dict[str, int] = dataclasses.field(default_factory=dict)

    # Parameter metadata for runtime binding
    parameter_metadata: ParameterMetadata = dataclasses.field(
        default_factory=ParameterMetadata
    )


@dataclasses.dataclass
class CompiledClassicalSegment:
    """A classical segment ready for Python execution."""

    segment: ClassicalSegment


@dataclasses.dataclass
class CompiledExpvalSegment:
    """A compiled expectation value segment with concrete observable.

    This segment computes <psi|H|psi> where psi is the quantum state
    from a quantum circuit and H is the concrete Hamiltonian observable.

    Attributes:
        segment: The original ExpvalSegment
        observable: The concrete Observable to measure
        quantum_segment_index: Index of the quantum segment providing the state
        result_ref: UUID where to store the expectation value result
        qubit_map: Mapping from Pauli index to physical qubit index.
            e.g., {0: 5, 1: 3} means Z(0) acts on qubit 5, Z(1) on qubit 3
    """

    segment: ExpvalSegment
    observable: "Observable"
    quantum_segment_index: int = 0
    result_ref: str = ""
    qubit_map: dict[int, int] = dataclasses.field(default_factory=dict)
