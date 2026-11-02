"""Expose Quration emission, local simulation, and FTQC compilation.

Rust owns QCore semantics, circuit projection, resources and result ABIs.
This optional adapter emits concrete PyQret IR and executes it; SDK objects
and machine-specific resource reports never enter persistent Qamomile values.
"""

from ._integration import (
    FORMAT as FORMAT,
    Quration as Quration,
    QurationLocal as QurationLocal,
    QurationResourceResult as QurationResourceResult,
    compile_resources as compile_resources,
    load_circuit as load_circuit,
)

__all__ = [
    "FORMAT",
    "Quration",
    "QurationLocal",
    "QurationResourceResult",
    "compile_resources",
    "load_circuit",
]
