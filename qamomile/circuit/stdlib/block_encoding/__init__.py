"""Expose exact block-encoding descriptors and construction factories.

The subpackage groups algorithm-independent operator encodings. Consumers
normally use the stable top-level ``qamomile.circuit`` exports, while direct
imports may use this namespace when producer-specific types are needed.
"""

from .ising_z import IsingZBlockEncoding, ising_z_block_encoding
from .lcu import (
    LCUBlockEncoding,
    LCUBlockEncodingTerm,
    identity_block_encoding,
    lcu_block_encoding,
)
from .pauli import PauliLCUBlockEncoding, pauli_lcu_block_encoding
from .periodic_shift import (
    PeriodicShiftLCUBlockEncoding,
    periodic_shift_lcu_block_encoding,
)

__all__ = [
    "LCUBlockEncoding",
    "LCUBlockEncodingTerm",
    "identity_block_encoding",
    "lcu_block_encoding",
    "IsingZBlockEncoding",
    "ising_z_block_encoding",
    "PauliLCUBlockEncoding",
    "pauli_lcu_block_encoding",
    "PeriodicShiftLCUBlockEncoding",
    "periodic_shift_lcu_block_encoding",
]
