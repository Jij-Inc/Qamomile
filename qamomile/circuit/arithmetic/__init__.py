"""Expose arithmetic qkernels for circuit-level imports.

This namespace currently provides X-gate modular increment and
decrement primitives for little-endian basis-state registers. Use
``qmc.control(...)`` on these qkernels to build controlled variants.
"""

from qamomile.circuit.algorithm.arithmetic import (
    modular_decrement,
    modular_increment,
)

__all__ = [
    "modular_decrement",
    "modular_increment",
]
