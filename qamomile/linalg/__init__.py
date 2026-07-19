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
* Möttönen amplitude-encoding angle precomputation
  (:func:`compute_mottonen_amplitude_encoding_ry_angles`,
  :func:`compute_mottonen_amplitude_encoding_rz_angles`,
  :func:`validate_and_normalize_amplitudes`).  The actual quantum
  gate emission lives under
  :mod:`qamomile.circuit.stdlib.state_preparation`; this module
  provides only the classical angle math so that hybrid loops can
  pre-compute angle vectors outside any kernel and feed them to
  ``mottonen_amplitude_encoding_from_angles`` via ``parameters=[...]``.
* :class:`PauliLCU`, which decomposes any finite complex power-of-two square
  matrix into immutable complex-weighted Pauli terms for LCU algorithms.
* :class:`PeriodicShiftLCU`, which validates constant-coefficient periodic
  shift structure and stores its immutable complex-weighted shift terms.
"""

from qamomile.linalg.hermitian import HermitianMatrix
from qamomile.linalg.mottonen import (
    compute_mottonen_amplitude_encoding_ry_angles,
    compute_mottonen_amplitude_encoding_rz_angles,
    validate_and_normalize_amplitudes,
)
from qamomile.linalg.pauli_lcu import PauliLCU, PauliLCUTerm
from qamomile.linalg.periodic_shift_lcu import (
    PeriodicShiftLCU,
    PeriodicShiftLCUTerm,
)
from qamomile.linalg.subspace import (
    generalized_subspace_matrices,
    solve_subspace,
    subspace_hamiltonian,
)

__all__ = [
    "HermitianMatrix",
    "PauliLCU",
    "PauliLCUTerm",
    "PeriodicShiftLCU",
    "PeriodicShiftLCUTerm",
    "compute_mottonen_amplitude_encoding_ry_angles",
    "compute_mottonen_amplitude_encoding_rz_angles",
    "generalized_subspace_matrices",
    "solve_subspace",
    "subspace_hamiltonian",
    "validate_and_normalize_amplitudes",
]
