"""Matrix types upstream of :mod:`qamomile.observable`.

Currently exposes :class:`HermitianMatrix`, which wraps a dense Hermitian
NumPy matrix and produces an exact Pauli decomposition as a
:class:`qamomile.observable.Hamiltonian` via the Fast Walsh-Hadamard
Transform.
"""

from qamomile.linalg.hermitian import HermitianMatrix

__all__ = ["HermitianMatrix"]
