"""State-preparation building blocks.

Available routines:

* :func:`computational_basis_state`: prepare ``|bits>`` from
  :math:`|0\\rangle^{\\otimes n}` via per-qubit ``Rx(pi * bits[i])``.
  Equivalent to a conditional ``X`` (up to a global phase) but emitted
  as a parameterised rotation so ``bits`` can be left as a runtime
  parameter at transpile time.
* :class:`MottonenAmplitudeEncoding` / :func:`amplitude_encoding`: prepare
  an arbitrary real- or complex-amplitude state from
  :math:`|0\\rangle^{\\otimes n}` using Möttönen's Ry (and, for complex
  inputs, Rz) decomposition.  The amplitudes are concrete and the
  angle computation runs at compile time (lazily — see the class
  docstring for the deferred-evaluation contract).
* :func:`amplitude_encoding_from_angles`: the parametric companion.
  Accepts pre-computed Ry (and optional Rz) angles as either concrete
  sequences or ``Vector[Float]`` kernel parameters, so the same
  compiled circuit can be re-bound to different amplitude vectors at
  runtime (e.g. inside a hybrid optimisation loop).

The classical Möttönen angle precomputation
(``compute_mottonen_amplitude_encoding_ry_angles`` /
``compute_mottonen_amplitude_encoding_rz_angles``) lives in
:mod:`qamomile.linalg`.  Import them from there directly when you need
to feed pre-computed angles into ``amplitude_encoding_from_angles``::

    from qamomile.linalg import (
        compute_mottonen_amplitude_encoding_ry_angles,
        compute_mottonen_amplitude_encoding_rz_angles,
    )
"""

from .computational_basis_state import computational_basis_state
from .mottonen_amplitude_encoding import (
    MottonenAmplitudeEncoding,
    amplitude_encoding,
    amplitude_encoding_from_angles,
)

__all__ = [
    "computational_basis_state",
    "MottonenAmplitudeEncoding",
    "amplitude_encoding",
    "amplitude_encoding_from_angles",
]
