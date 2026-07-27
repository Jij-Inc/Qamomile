"""State-preparation building blocks.

Available routines:

* :func:`computational_basis_state`: prepare ``|bits>`` from
  :math:`|0\\rangle^{\\otimes n}` via exact conditional ``X`` powers.
  The parameterized RX implementation includes phase compensation, so
  ``bits`` can remain a runtime parameter and the routine can be controlled
  without exposing an unintended relative phase.
* :func:`amplitude_encoding`: prepare an arbitrary real- or
  complex-amplitude state from :math:`|0\\rangle^{\\otimes n}`. The
  synthesis method is unspecified, so a backend may select a native
  state-preparation implementation. Qamomile's portable fallback currently
  uses the Möttönen construction.
* :func:`mottonen_amplitude_encoding`: prepare the same target state while
  making Möttönen's uniformly controlled Ry/Rz construction part of the API
  contract.
* :func:`mottonen_amplitude_encoding_from_angles`: the parametric Möttönen
  companion. It accepts pre-computed Ry and optional Rz angles as concrete
  sequences or ``Vector[Float]`` kernel parameters.
* :func:`amplitude_encoding_from_angles`: compatibility name for
  :func:`mottonen_amplitude_encoding_from_angles`.

The classical Möttönen angle precomputation
(``compute_mottonen_amplitude_encoding_ry_angles`` /
``compute_mottonen_amplitude_encoding_rz_angles``) lives in
:mod:`qamomile.linalg`.  Import them from there directly when you need
to feed pre-computed angles into
``mottonen_amplitude_encoding_from_angles``::

    from qamomile.linalg import (
        compute_mottonen_amplitude_encoding_ry_angles,
        compute_mottonen_amplitude_encoding_rz_angles,
    )
"""

from .computational_basis_state import computational_basis_state
from .mottonen_amplitude_encoding import (
    amplitude_encoding,
    amplitude_encoding_from_angles,
    mottonen_amplitude_encoding,
    mottonen_amplitude_encoding_from_angles,
)

__all__ = [
    "computational_basis_state",
    "amplitude_encoding",
    "amplitude_encoding_from_angles",
    "mottonen_amplitude_encoding",
    "mottonen_amplitude_encoding_from_angles",
]
