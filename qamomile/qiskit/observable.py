"""Qiskit observable support.

This module provides Qiskit-specific implementations for:
- QiskitObservableEmitter: Converts ConcreteHamiltonian to SparsePauliOp
- QiskitExpectationEstimator: Estimates expectation values using Qiskit Estimator
"""

from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

from qamomile.circuit.observable import (
    ConcreteHamiltonian,
    Observable,
    ObservableEmitter,
    ExpectationEstimator,
)
from qamomile.circuit.ir.types.hamiltonian import PauliKind

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp


class QiskitObservableEmitter:
    """Emitter that converts ConcreteHamiltonian to Qiskit SparsePauliOp.

    Example:
        emitter = QiskitObservableEmitter()
        sparse_pauli_op = emitter.emit_observable(hamiltonian)
    """

    def emit_observable(self, hamiltonian: ConcreteHamiltonian) -> "SparsePauliOp":
        """Convert ConcreteHamiltonian to Qiskit SparsePauliOp.

        Args:
            hamiltonian: The Hamiltonian to convert

        Returns:
            Qiskit SparsePauliOp representation
        """
        from qiskit.quantum_info import SparsePauliOp

        n_qubits = hamiltonian.num_qubits
        if n_qubits == 0:
            # Handle empty Hamiltonian
            return SparsePauliOp.from_list([("I", 0.0)])

        pauli_list = []

        for pauli_string, coeff in hamiltonian:
            # Build Pauli string in Qiskit format (reverse order)
            # Qiskit uses little-endian: rightmost character is qubit 0
            paulis = ["I"] * n_qubits
            for pauli_kind, qubit_idx in pauli_string:
                paulis[qubit_idx] = pauli_kind.name

            # Qiskit expects string in reverse order (qubit 0 is rightmost)
            pauli_str = "".join(reversed(paulis))
            pauli_list.append((pauli_str, coeff))

        if not pauli_list:
            # Empty Hamiltonian (zero operator)
            pauli_list = [("I" * max(1, n_qubits), 0.0)]

        return SparsePauliOp.from_list(pauli_list)


class QiskitExpectationEstimator(ExpectationEstimator["QuantumCircuit"]):
    """Expectation value estimator using Qiskit Estimator primitive.

    This estimator uses Qiskit's Estimator primitive to compute expectation
    values. It can use either the statevector simulator or a real backend.

    Example:
        from qiskit_aer import AerSimulator
        from qiskit.primitives import Estimator

        estimator = QiskitExpectationEstimator()  # Uses default Estimator
        exp_val = estimator.estimate(circuit, observable)

        # Or with custom Estimator
        backend = AerSimulator()
        qiskit_estimator = Estimator(backend)
        estimator = QiskitExpectationEstimator(qiskit_estimator)
    """

    def __init__(
        self,
        estimator: "Estimator | None" = None,
        shots: int | None = None,
    ):
        """Initialize the estimator.

        Args:
            estimator: Qiskit Estimator primitive. If None, creates default.
            shots: Number of shots for sampling. If None, uses exact simulation.
        """
        self._estimator = estimator
        self._shots = shots
        self._emitter = QiskitObservableEmitter()

    def _get_estimator(self):
        """Get or create the Qiskit Estimator."""
        if self._estimator is not None:
            return self._estimator

        # Create default estimator
        from qiskit.primitives import Estimator
        self._estimator = Estimator()
        return self._estimator

    def estimate(
        self,
        circuit: "QuantumCircuit",
        observable: Observable,
        params: Sequence[float] | None = None,
    ) -> float:
        """Estimate the expectation value of an observable.

        Args:
            circuit: Qiskit QuantumCircuit (state preparation ansatz)
            observable: The observable to measure
            params: Optional parameter values for parametric circuits

        Returns:
            The estimated expectation value
        """
        estimator = self._get_estimator()
        sparse_pauli_op = self._emitter.emit_observable(observable.hamiltonian)

        # Handle parametric circuits
        if params is not None:
            param_values = list(params)
        else:
            param_values = []

        # Run estimation
        job = estimator.run([(circuit, sparse_pauli_op, param_values)])
        result = job.result()

        # Extract expectation value
        return float(result[0].data.evs)

    def estimate_batch(
        self,
        circuit: "QuantumCircuit",
        observables: Sequence[Observable],
        params: Sequence[float] | None = None,
    ) -> list[float]:
        """Estimate expectation values for multiple observables.

        Args:
            circuit: Qiskit QuantumCircuit (state preparation ansatz)
            observables: List of observables to measure
            params: Optional parameter values for parametric circuits

        Returns:
            List of estimated expectation values
        """
        estimator = self._get_estimator()

        # Convert all observables
        sparse_pauli_ops = [
            self._emitter.emit_observable(obs.hamiltonian)
            for obs in observables
        ]

        # Handle parametric circuits
        if params is not None:
            param_values = list(params)
        else:
            param_values = []

        # Create PUBs (Primitive Unified Blocs) for batch execution
        pubs = [(circuit, op, param_values) for op in sparse_pauli_ops]

        # Run estimation
        job = estimator.run(pubs)
        result = job.result()

        # Extract expectation values
        return [float(r.data.evs) for r in result]


# Convenience function
def to_sparse_pauli_op(hamiltonian: ConcreteHamiltonian) -> "SparsePauliOp":
    """Convert ConcreteHamiltonian to Qiskit SparsePauliOp.

    Convenience function that creates an emitter and converts.

    Args:
        hamiltonian: The Hamiltonian to convert

    Returns:
        Qiskit SparsePauliOp representation
    """
    emitter = QiskitObservableEmitter()
    return emitter.emit_observable(hamiltonian)
