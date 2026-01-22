"""Pauli operator submodule for Hamiltonian construction.

This module provides factory functions for Pauli operators. Usage via
the `qm.pauli` namespace avoids confusion with quantum gate functions
(qm.x, qm.z).

Example:
    import qamomile.circuit as qm

    @qm.qkernel
    def ising_hamiltonian(n: qm.UInt, J: qm.Float) -> qm.HamiltonianExpr:
        H = 0.0 * qm.pauli.I(0)  # Start with zero
        for i in qm.range(n - 1):
            H = H + J * qm.pauli.Z(i) * qm.pauli.Z(i + 1)
        return H
"""

from .operators import X, Y, Z, I

__all__ = ["X", "Y", "Z", "I"]
