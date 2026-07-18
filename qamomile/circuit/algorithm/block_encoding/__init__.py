"""Compatibility exports for structured block-encoding factories.

The implementations live in :mod:`qamomile.circuit.stdlib`. Qamomile does not
define one universal base for every block-encoding data-access model, while
static exact LCU producers share :class:`qamomile.circuit.LCUBlockEncoding`.
"""

from .periodic_stencil import (
    PeriodicStencilBlockEncoding,
    periodic_stencil_block_encoding,
)

__all__ = [
    "PeriodicStencilBlockEncoding",
    "periodic_stencil_block_encoding",
]
