"""QuriParts observable support.

This module provides QuriParts-specific implementations for:
- QuriPartsObservableEmitter: Converts ConcreteHamiltonian to QuriParts Operator
- QuriPartsExpectationEstimator: Estimates expectation values
"""

from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

from qamomile.circuit.observable import (
    ConcreteHamiltonian,
    Observable,
    ExpectationEstimator,
)
from qamomile.circuit.ir.types.hamiltonian import PauliKind

if TYPE_CHECKING:
    from quri_parts.core.operator import Operator
    from quri_parts.circuit import NonParametricQuantumCircuit


# Mapping from PauliKind to QuriParts single-qubit Pauli functions
_PAULI_TO_QURI = {
    PauliKind.I: lambda q: {},  # Identity doesn't contribute
    PauliKind.X: lambda q: {(1, q)},  # X
    PauliKind.Y: lambda q: {(2, q)},  # Y
    PauliKind.Z: lambda q: {(3, q)},  # Z
}


class QuriPartsObservableEmitter:
    """Emitter that converts ConcreteHamiltonian to QuriParts Operator.

    Example:
        emitter = QuriPartsObservableEmitter()
        operator = emitter.emit_observable(hamiltonian)
    """

    def emit_observable(self, hamiltonian: ConcreteHamiltonian) -> "Operator":
        """Convert ConcreteHamiltonian to QuriParts Operator.

        Args:
            hamiltonian: The Hamiltonian to convert

        Returns:
            QuriParts Operator representation
        """
        from quri_parts.core.operator import Operator, pauli_label, PAULI_IDENTITY

        terms = {}

        for pauli_string, coeff in hamiltonian:
            if not pauli_string:
                # Identity term
                terms[PAULI_IDENTITY] = terms.get(PAULI_IDENTITY, 0) + coeff
            else:
                # Build QuriParts pauli label
                # pauli_label format: frozenset of (pauli_id, qubit_index)
                # pauli_id: 1=X, 2=Y, 3=Z
                pauli_indices = []
                for pauli_kind, qubit_idx in pauli_string:
                    if pauli_kind == PauliKind.I:
                        continue  # Skip identity
                    pauli_id = {PauliKind.X: 1, PauliKind.Y: 2, PauliKind.Z: 3}[pauli_kind]
                    pauli_indices.append((qubit_idx, pauli_id))

                if pauli_indices:
                    label = pauli_label(pauli_indices)
                    terms[label] = terms.get(label, 0) + coeff
                else:
                    # All identities reduced to scalar
                    terms[PAULI_IDENTITY] = terms.get(PAULI_IDENTITY, 0) + coeff

        return Operator(terms)


class QuriPartsExpectationEstimator(ExpectationEstimator["NonParametricQuantumCircuit"]):
    """Expectation value estimator using QuriParts.

    This estimator uses QuriParts to compute expectation values.

    Example:
        from quri_parts.qulacs.estimator import create_qulacs_vector_estimator

        qulacs_estimator = create_qulacs_vector_estimator()
        estimator = QuriPartsExpectationEstimator(qulacs_estimator)
        exp_val = estimator.estimate(circuit, observable)
    """

    def __init__(self, quri_estimator=None):
        """Initialize the estimator.

        Args:
            quri_estimator: QuriParts quantum estimator function.
                            Should be a ConcurrentQuantumEstimator or similar.
        """
        self._quri_estimator = quri_estimator
        self._emitter = QuriPartsObservableEmitter()

    def _get_estimator(self):
        """Get or create the QuriParts estimator."""
        if self._quri_estimator is not None:
            return self._quri_estimator

        # Try to create a default estimator using Qulacs
        try:
            from quri_parts.qulacs.estimator import create_qulacs_vector_estimator
            self._quri_estimator = create_qulacs_vector_estimator()
            return self._quri_estimator
        except ImportError:
            raise ImportError(
                "quri-parts-qulacs is required for default estimator. "
                "Install it with: pip install quri-parts-qulacs"
            )

    def estimate(
        self,
        circuit: "NonParametricQuantumCircuit",
        observable: Observable,
        params: Sequence[float] | None = None,
    ) -> float:
        """Estimate the expectation value of an observable.

        Args:
            circuit: QuriParts circuit (state preparation ansatz)
            observable: The observable to measure
            params: Ignored for non-parametric circuits

        Returns:
            The estimated expectation value
        """
        estimator = self._get_estimator()
        operator = self._emitter.emit_observable(observable.hamiltonian)

        # Run estimation
        estimate = estimator(operator, circuit)
        return float(estimate.value.real)

    def estimate_batch(
        self,
        circuit: "NonParametricQuantumCircuit",
        observables: Sequence[Observable],
        params: Sequence[float] | None = None,
    ) -> list[float]:
        """Estimate expectation values for multiple observables.

        Args:
            circuit: QuriParts circuit (state preparation ansatz)
            observables: List of observables to measure
            params: Ignored for non-parametric circuits

        Returns:
            List of estimated expectation values
        """
        estimator = self._get_estimator()
        operators = [
            self._emitter.emit_observable(obs.hamiltonian)
            for obs in observables
        ]

        # Estimate each observable
        results = []
        for operator in operators:
            estimate = estimator(operator, circuit)
            results.append(float(estimate.value.real))

        return results


# Convenience function
def to_quri_operator(hamiltonian: ConcreteHamiltonian) -> "Operator":
    """Convert ConcreteHamiltonian to QuriParts Operator.

    Convenience function that creates an emitter and converts.

    Args:
        hamiltonian: The Hamiltonian to convert

    Returns:
        QuriParts Operator representation
    """
    emitter = QuriPartsObservableEmitter()
    return emitter.emit_observable(hamiltonian)
