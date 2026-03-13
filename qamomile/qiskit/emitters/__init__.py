"""Qiskit-specific CompositeGate emitters.

This package contains emitters for native Qiskit implementations
of composite gates like QFT and IQFT.

Note: QPE (PhaseEstimation) is NOT included here because it's deprecated
in Qiskit 2.1 and will be removed in 3.0. QPE uses manual decomposition.
"""

from qamomile.qiskit.emitters.qft import QiskitQFTEmitter

__all__ = [
    "QiskitQFTEmitter",
]
