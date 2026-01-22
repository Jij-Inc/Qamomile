"""Observable wrapper for expectation value context.

This module provides the Observable class which wraps a ConcreteHamiltonian
and provides conversion utilities for different backends.
"""

from __future__ import annotations

import dataclasses
from typing import TypeVar, Generic, Protocol

from .concrete import ConcreteHamiltonian

T = TypeVar("T")


class ObservableEmitter(Protocol[T]):
    """Protocol for emitting observables to backend-specific formats.

    Backend modules (qiskit, quri_parts, etc.) should implement this
    protocol to convert ConcreteHamiltonian to their native format.
    """

    def emit_observable(self, hamiltonian: ConcreteHamiltonian) -> T:
        """Convert a ConcreteHamiltonian to native backend format.

        Args:
            hamiltonian: The Hamiltonian to convert

        Returns:
            Backend-specific observable representation
        """
        ...


@dataclasses.dataclass
class Observable:
    """Observable wrapper for expectation value calculations.

    This class wraps a ConcreteHamiltonian and provides utilities for
    converting to backend-specific formats and computing expectation values.

    Example:
        # Build a Hamiltonian
        @qm.qkernel
        def cost() -> qm.HamiltonianExpr:
            return qm.pauli.Z(0) * qm.pauli.Z(1) + qm.pauli.X(0)

        # Evaluate and wrap in Observable
        concrete_h = evaluate_hamiltonian(cost.build())
        obs = Observable(concrete_h)

        # Convert to Qiskit format
        qiskit_obs = obs.to_native(qiskit_emitter)
    """

    hamiltonian: ConcreteHamiltonian

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits this observable acts on."""
        return self.hamiltonian.num_qubits

    @property
    def num_terms(self) -> int:
        """Return the number of Pauli terms."""
        return len(self.hamiltonian)

    def to_native(self, emitter: ObservableEmitter[T]) -> T:
        """Convert to backend-specific format using an emitter.

        Args:
            emitter: Backend-specific emitter implementing ObservableEmitter

        Returns:
            Backend-specific observable representation
        """
        return emitter.emit_observable(self.hamiltonian)

    def remap_qubits(self, qubit_map: dict[int, int]) -> "Observable":
        """Remap qubit indices according to the given mapping.

        This is used to translate Pauli indices (logical indices within an
        expval call) to physical qubit indices in the actual quantum circuit.

        Args:
            qubit_map: Mapping from logical index to physical index.
                       e.g., {0: 5, 1: 3} maps logical index 0 → physical qubit 5

        Returns:
            New Observable with remapped qubit indices.
        """
        remapped_h = self.hamiltonian.remap_qubits(qubit_map)
        return Observable(remapped_h)

    def __repr__(self) -> str:
        return f"Observable({self.hamiltonian!r})"
