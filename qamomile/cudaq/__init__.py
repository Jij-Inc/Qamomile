"""Expose CUDA-Q compilation and execution support for Qamomile.

QCore and the Rust compiler remain the semantic authorities. This namespace
only exposes self-contained Python source emission and CUDA-Q execution endpoints; SDK
objects stay out of persistent Qamomile artifacts.
"""

from ._integration import (
    FORMAT as FORMAT,
    Cudaq as Cudaq,
    CudaqLocal as CudaqLocal,
    load_kernel as load_kernel,
)

__all__ = ["Cudaq", "CudaqLocal", "FORMAT", "load_kernel"]
