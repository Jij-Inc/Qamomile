"""Observable module for Hamiltonian/expectation value support.

This module provides:
- ConcreteHamiltonian: Concrete representation of Hamiltonians after evaluation
- Observable: Wrapper for ConcreteHamiltonian with conversion utilities
- ExpectationEstimator: Protocol for expectation value estimation
"""

from .concrete import ConcreteHamiltonian, PauliString
from .observable import Observable, ObservableEmitter
from .estimator import ExpectationEstimator

__all__ = [
    "ConcreteHamiltonian",
    "PauliString",
    "Observable",
    "ObservableEmitter",
    "ExpectationEstimator",
]
