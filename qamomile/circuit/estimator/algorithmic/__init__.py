"""Algorithmic resource estimators based on theoretical complexity formulas.

These estimators provide resource bounds based on published complexity
results from quantum algorithms literature, particularly from:

arXiv:2310.03011v2 - "Quantum algorithms: A survey of applications
and end-to-end complexities"

Unlike the circuit-based estimators (gate_counter, qubits_counter),
these provide theoretical estimates based on algorithm parameters
without needing the actual circuit implementation.
"""

from qamomile.circuit.estimator.algorithmic.hamiltonian_simulation import (
    estimate_qdrift,
    estimate_qsvt,
    estimate_trotter,
)
from qamomile.circuit.estimator.algorithmic.qaoa import estimate_qaoa
from qamomile.circuit.estimator.algorithmic.qpe import estimate_qpe

__all__ = [
    "estimate_qaoa",
    "estimate_qpe",
    "estimate_trotter",
    "estimate_qsvt",
    "estimate_qdrift",
]
