"""Expose Amazon Braket compilation and execution support for Qamomile.

QCore and the Rust compiler remain the semantic authorities. This namespace
only exposes OpenQASM emission and Braket execution endpoints; SDK objects stay
out of persistent Qamomile artifacts.
"""

from ._integration import (
    FORMAT as FORMAT,
    Braket as Braket,
    BraketDevice as BraketDevice,
    BraketLocal as BraketLocal,
    load_openqasm as load_openqasm,
)

__all__ = ["Braket", "BraketDevice", "BraketLocal", "FORMAT", "load_openqasm"]
