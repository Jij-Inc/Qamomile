"""Pure-classical binary/spin polynomial problem representation.

Design intent: this is the quantum-agnostic problem model that the
optimization converters consume. ``BinaryExpr`` (``expr.py``) is a plain
coefficient dict ``{index tuple: coeff}`` over BINARY or SPIN variables
(``VarType`` tracked at the type level via the ``VT`` generic, so
binary↔spin conversions are explicit, not implicit re-interpretation).
``BinaryModel`` (``model.py``) wraps an expression with a bidirectional
original↔dense index mapping and sample decoding; ``BinarySampleSet``
(``sampleset.py``) is the decoded-result container; ``normalize.py``
holds coefficient-scaling strategies.

Constraints:
- No quantum imports beyond ``qamomile.circuit``'s ``SampleResult`` type
  (for decoding) — this package must import without any SDK installed.
- Arbitrary-order (HUBO) terms are first-class: nothing here may assume
  degree <= 2.
"""

from .expr import BinaryExpr, VarType, binary, spin
from .model import BinaryModel
from .sampleset import BinarySampleSet

__all__ = [
    "BinaryModel",
    "binary",
    "spin",
    "BinaryExpr",
    "VarType",
    "BinarySampleSet",
]
