"""Abstract base class for quantum backend execution.

This module provides a simple interface for implementing custom quantum executors.

Example:
    class MyExecutor(QuantumExecutor[QuantumCircuit]):
        def __init__(self, backend):
            self.backend = backend

        def execute(self, circuit: QuantumCircuit, shots: int) -> dict[str, int]:
            job = self.backend.run(circuit, shots=shots)
            return job.result().get_counts()

        def bind_parameters(self, circuit, bindings, metadata):
            param_map = {p.backend_param: bindings[p.name]
                         for p in metadata.parameters}
            return circuit.assign_parameters(param_map)

        def estimate(self, circuit, observable, params=None):
            # Use backend's estimator primitive
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Sequence, TypeVar, TYPE_CHECKING

from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata

if TYPE_CHECKING:
    from qamomile.circuit.observable import Observable

T = TypeVar("T")  # Backend circuit type


class QuantumExecutor(ABC, Generic[T]):
    """Abstract base class for quantum backend execution.

    To implement a custom executor:
    1. Implement execute() (required) - Execute circuit and return bitstring counts
    2. Override bind_parameters() (optional) - For parametric circuit support

    Example:
        class MyExecutor(QuantumExecutor[QuantumCircuit]):
            def __init__(self, backend):
                self.backend = backend

            def execute(self, circuit, shots):
                return self.backend.run(circuit, shots=shots).result().get_counts()
    """

    @abstractmethod
    def execute(self, circuit: T, shots: int) -> dict[str, int]:
        """Execute the circuit and return bitstring counts.

        Args:
            circuit: The quantum circuit to execute
            shots: Number of measurement shots

        Returns:
            Dictionary mapping bitstrings to counts.
            Example: {"00": 512, "11": 512}
        """
        pass

    def bind_parameters(
        self,
        circuit: T,
        bindings: dict[str, Any],
        parameter_metadata: ParameterMetadata,
    ) -> T:
        """Bind parameter values to the circuit.

        Default implementation returns the circuit unchanged.
        Override for backends that support parametric circuits.

        Args:
            circuit: The parameterized circuit
            bindings: Dict mapping parameter names (indexed format) to values.
                     e.g., {"gammas[0]": 0.1, "gammas[1]": 0.2}
            parameter_metadata: Metadata about circuit parameters

        Returns:
            New circuit with parameters bound
        """
        return circuit

    def estimate(
        self,
        circuit: T,
        observable: "Observable",
        params: Sequence[float] | None = None,
    ) -> float:
        """Estimate the expectation value of an observable.

        This method computes <psi|H|psi> where psi is the quantum state
        prepared by the circuit and H is the observable Hamiltonian.

        Backends can override this method to provide optimized implementations
        using their native estimator primitives.

        Args:
            circuit: The quantum circuit (state preparation ansatz)
            observable: The observable to measure
            params: Optional parameter values for parametric circuits

        Returns:
            The estimated expectation value

        Raises:
            NotImplementedError: If the executor does not support estimation
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support expectation value estimation. "
            "Use an executor with estimator support (e.g., QiskitExecutor with estimator)."
        )
