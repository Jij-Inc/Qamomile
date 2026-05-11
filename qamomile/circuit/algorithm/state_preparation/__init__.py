"""State-preparation building blocks.

Available routines:

* :class:`MottonenAmplitudeEncoding` / :func:`amplitude_encoding`: prepare
  an arbitrary real-amplitude state from :math:`|0\\rangle^{\\otimes n}`
  using Möttönen's Y-rotation decomposition.
* :func:`compute_mottonen_amplitude_encoding_thetas`: classical angle
  pre-computation for the Möttönen encoding.
"""

from .mottonen_amplitude_encoding import (
    MottonenAmplitudeEncoding,
    amplitude_encoding,
    compute_mottonen_amplitude_encoding_thetas,
)

__all__ = [
    "MottonenAmplitudeEncoding",
    "amplitude_encoding",
    "compute_mottonen_amplitude_encoding_thetas",
]
