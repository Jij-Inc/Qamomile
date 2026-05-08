"""Matrix types upstream of :mod:`qamomile.observable`.

Exposes:

* :class:`HermitianMatrix`, which wraps a dense Hermitian NumPy matrix
  and produces an exact Pauli decomposition as a
  :class:`qamomile.observable.Hamiltonian` via the Fast Walsh-Hadamard
  Transform.
* Sample-based partial-diagonalization helpers
  (:func:`subspace_hamiltonian`, :func:`generalized_subspace_matrices`,
  :func:`solve_subspace`) that build Hamiltonian / overlap matrices on
  a subspace spanned by a set of computational-basis or mixed-Pauli-basis
  product states and (optionally) solve the resulting (generalised)
  eigenvalue problem.
"""

from qamomile.linalg.hermitian import HermitianMatrix
from qamomile.linalg.subspace import (
    generalized_subspace_matrices,
    solve_subspace,
    subspace_hamiltonian,
)

__all__ = [
    "HermitianMatrix",
    "generalized_subspace_matrices",
    "solve_subspace",
    "subspace_hamiltonian",
]
