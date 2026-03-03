"""Resource estimation module for Qamomile circuits.

This module provides comprehensive resource estimation for quantum circuits,
including:

1. **Qubit counting**: Algebraic qubit count with SymPy
2. **Gate counting**: Breakdown by gate type (single/two-qubit, T gates, Clifford)
3. **Algorithmic estimates**: Theoretical bounds for QAOA, QPE, Hamiltonian simulation

All estimates are expressed as SymPy symbolic expressions, allowing
dependency on problem size parameters.

Usage:
    Basic circuit analysis:
        >>> from qamomile.circuit.estimator import estimate_resources
        >>> estimate = estimate_resources(my_circuit.block)
        >>> print(estimate.qubits)  # e.g., "n + 3"
        >>> print(estimate.gates.total)  # e.g., "2*n"

    Algorithmic estimates:
        >>> from qamomile.circuit.estimator.algorithmic import estimate_qaoa
        >>> import sympy as sp
        >>> n, p = sp.symbols('n p', positive=True, integer=True)
        >>> est = estimate_qaoa(n, p, num_edges=n*(n-1)/2)
        >>> print(est.gates.total)

References:
    Based on "Quantum algorithms: A survey of applications and
    end-to-end complexities" (arXiv:2310.03011v2)
"""

# Core estimators
from qamomile.circuit.estimator.qubits_counter import qubits_counter
from qamomile.circuit.estimator.gate_counter import GateCount, count_gates
from qamomile.circuit.estimator.resource_estimator import (
    ResourceEstimate,
    estimate_resources,
)

# Algorithmic estimators are in submodule
# from qamomile.circuit.estimator.algorithmic import ...

__all__ = [
    # Core types
    "ResourceEstimate",
    "GateCount",
    # Core estimators
    "qubits_counter",
    "count_gates",
    "estimate_resources",
]
