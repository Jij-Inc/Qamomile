"""State-preparation building blocks.

Available routines:

* :class:`MottonenAmplitudeEncoding` / :func:`amplitude_encoding`: prepare
  an arbitrary real- or complex-amplitude state from
  :math:`|0\\rangle^{\\otimes n}` using Möttönen's Ry (and, for complex
  inputs, Rz) decomposition.
* :func:`compute_mottonen_amplitude_encoding_ry_angles`: classical
  pre-computation of the magnitude-stage Ry rotation angles.
* :func:`compute_mottonen_amplitude_encoding_rz_angles`: classical
  pre-computation of the phase-stage Rz rotation angles (all zeros for
  real inputs).
"""

from .mottonen_amplitude_encoding import (
    MottonenAmplitudeEncoding,
    amplitude_encoding,
    compute_mottonen_amplitude_encoding_ry_angles,
    compute_mottonen_amplitude_encoding_rz_angles,
)

__all__ = [
    "MottonenAmplitudeEncoding",
    "amplitude_encoding",
    "compute_mottonen_amplitude_encoding_ry_angles",
    "compute_mottonen_amplitude_encoding_rz_angles",
]
