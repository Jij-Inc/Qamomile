"""Resource estimation module for Qamomile circuits.

This module provides comprehensive resource estimation for quantum circuits,
including:

1. **Qubit counting**: Algebraic qubit count with SymPy
2. **Gate counting**: Breakdown by gate type (single/two-qubit, T gates, Clifford)
3. **Algorithmic estimates**: Theoretical bounds for QAOA, QPE, Hamiltonian simulation

All estimates are expressed as SymPy symbolic expressions, allowing
dependency on problem size parameters.

Usage:
    Basic circuit analysis (method API — recommended):
        >>> estimate = my_circuit.estimate_resources()
        >>> print(estimate.qubits)  # e.g., "n + 3"
        >>> print(estimate.gates.total)  # e.g., "2*n"

    Function API (also supported):
        >>> from qamomile.circuit.estimator import estimate_resources
        >>> estimate = estimate_resources(my_circuit.block)
        >>> print(estimate.qubits)  # e.g., "n + 3"
        >>> print(estimate.gates.total)  # e.g., "2*n"

    Formula-based algorithm estimates live in ``qamomile.resource_estimation``:
        >>> from qamomile.resource_estimation import estimate_qaoa
        >>> import sympy as sp
        >>> n, p = sp.symbols('n p', positive=True, integer=True)
        >>> est = estimate_qaoa(n, p, num_edges=n*(n-1)/2)
        >>> print(est.gates.total)
"""

# Core estimators
from qamomile.circuit.estimator.gate_counter import GateCount, count_gates
from qamomile.circuit.estimator.qubits_counter import qubits_counter
from qamomile.circuit.estimator.resource_estimator import (
    ResourceEstimate,
    estimate_resources,
)

__all__ = [
    # Core types
    "ResourceEstimate",
    "GateCount",
    # Core estimators
    "qubits_counter",
    "count_gates",
    "estimate_resources",
]
