"""Compatibility exports for structured block-encoding factories.

The implementations live in :mod:`qamomile.circuit.stdlib`. Qamomile does not
define one universal base for every block-encoding data-access model, while
static exact LCU producers share :class:`qamomile.circuit.LCUBlockEncoding`.
"""

from .periodic_shift_lcu_block_encoding import (
    PeriodicShiftLCUBlockEncoding,
    periodic_shift_lcu_block_encoding,
)

__all__ = [
    "PeriodicShiftLCUBlockEncoding",
    "periodic_shift_lcu_block_encoding",
]
