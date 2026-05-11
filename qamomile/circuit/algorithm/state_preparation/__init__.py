"""State-preparation building blocks.

Available routines:

* :class:`MottonenAmplitudeEncoding` / :func:`amplitude_encoding`: prepare
  an arbitrary real- or complex-amplitude state from
  :math:`|0\\rangle^{\\otimes n}` using Möttönen's Ry (and, for complex
  inputs, Rz) decomposition.  The amplitudes are concrete and the
  angle computation runs at compile time.
* :func:`amplitude_encoding_from_angles`: the parametric companion.
  Accepts pre-computed Ry (and optional Rz) angles as either concrete
  sequences or ``Vector[Float]`` kernel parameters, so the same
  compiled circuit can be re-bound to different amplitude vectors at
  runtime (e.g. inside a hybrid optimisation loop).
* :func:`compute_mottonen_amplitude_encoding_ry_angles`: classical
  pre-computation of the magnitude-stage Ry rotation angles.
* :func:`compute_mottonen_amplitude_encoding_rz_angles`: classical
  pre-computation of the phase-stage Rz rotation angles (all zeros for
  real inputs).
"""

from .mottonen_amplitude_encoding import (
    MottonenAmplitudeEncoding,
    amplitude_encoding,
    amplitude_encoding_from_angles,
    compute_mottonen_amplitude_encoding_ry_angles,
    compute_mottonen_amplitude_encoding_rz_angles,
)

__all__ = [
    "MottonenAmplitudeEncoding",
    "amplitude_encoding",
    "amplitude_encoding_from_angles",
    "compute_mottonen_amplitude_encoding_ry_angles",
    "compute_mottonen_amplitude_encoding_rz_angles",
]
