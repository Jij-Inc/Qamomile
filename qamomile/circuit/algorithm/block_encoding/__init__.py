"""Structured block-encoding factories for quantum algorithms.

This package contains method-specific constructions.  It intentionally does
not define a universal block-encoding base class because different data-access
models have different signal-register and workspace contracts.
"""

from .periodic_stencil import (
    PeriodicStencilEncoding,
    periodic_stencil_block_encoding,
)

__all__ = [
    "PeriodicStencilEncoding",
    "periodic_stencil_block_encoding",
]
