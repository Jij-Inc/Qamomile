"""Expectation value estimator protocol.

This module defines the ExpectationEstimator protocol that backend
modules should implement to support expectation value calculations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Sequence

from .observable import Observable

# Type variable for backend-specific circuit type
CircuitT = TypeVar("CircuitT")


class ExpectationEstimator(ABC, Generic[CircuitT]):
    """Abstract base class for expectation value estimation.

    Backend modules should implement this class to support computing
    expectation values of observables with respect to quantum circuits.

    The type parameter CircuitT represents the backend-specific circuit
    type (e.g., qiskit.QuantumCircuit, quri_parts.Circuit).

    Example implementation for Qiskit:
        class QiskitExpectationEstimator(ExpectationEstimator[QuantumCircuit]):
            def __init__(self, backend, shots=1024):
                self.backend = backend
                self.shots = shots

            def estimate(self, circuit, observable, params=None):
                # Use Qiskit's Estimator primitive
                ...
    """

    @abstractmethod
    def estimate(
        self,
        circuit: CircuitT,
        observable: Observable,
        params: Sequence[float] | None = None,
    ) -> float:
        """Estimate the expectation value of an observable.

        Args:
            circuit: The quantum circuit (state preparation ansatz)
            observable: The observable to measure
            params: Optional parameter values for parametric circuits

        Returns:
            The estimated expectation value <psi|H|psi>
        """
        raise NotImplementedError

    @abstractmethod
    def estimate_batch(
        self,
        circuit: CircuitT,
        observables: Sequence[Observable],
        params: Sequence[float] | None = None,
    ) -> list[float]:
        """Estimate expectation values for multiple observables.

        This method allows efficient batching of multiple observable
        measurements on the same circuit state.

        Args:
            circuit: The quantum circuit (state preparation ansatz)
            observables: List of observables to measure
            params: Optional parameter values for parametric circuits

        Returns:
            List of estimated expectation values
        """
        raise NotImplementedError

    def estimate_gradient(
        self,
        circuit: CircuitT,
        observable: Observable,
        params: Sequence[float],
    ) -> list[float]:
        """Estimate the gradient of the expectation value.

        Default implementation uses parameter-shift rule.
        Backend implementations may override with more efficient methods.

        Args:
            circuit: The quantum circuit (state preparation ansatz)
            observable: The observable to measure
            params: Parameter values at which to compute gradient

        Returns:
            List of partial derivatives d<H>/d(param_i)
        """
        # Default: numerical differentiation via parameter shift
        # Can be overridden by backends with analytic gradient support
        shift = 0.5  # pi/2 shift for parameter-shift rule
        gradient = []

        for i in range(len(params)):
            # Forward shift
            params_plus = list(params)
            params_plus[i] += shift

            # Backward shift
            params_minus = list(params)
            params_minus[i] -= shift

            # Parameter-shift rule: gradient = (f(+) - f(-)) / (2 * sin(shift))
            exp_plus = self.estimate(circuit, observable, params_plus)
            exp_minus = self.estimate(circuit, observable, params_minus)

            import math
            grad_i = (exp_plus - exp_minus) / (2 * math.sin(shift))
            gradient.append(grad_i)

        return gradient
