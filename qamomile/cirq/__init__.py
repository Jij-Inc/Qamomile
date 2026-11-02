"""Cirq realization and execution support for Qamomile.

QCore and the Rust compiler remain the semantic authorities. This namespace
exposes Cirq JSON emission, local simulation, explicit noise, and Google
processor endpoints without placing SDK objects in persistent artifacts.
"""

from ._integration import (
    FORMAT as FORMAT,
    Cirq as Cirq,
    CirqProcessor as CirqProcessor,
    CirqSampler as CirqSampler,
    CirqSimulator as CirqSimulator,
    DepolarizingNoise as DepolarizingNoise,
    GoogleQVMNoise as GoogleQVMNoise,
    _execute_local as _execute_local,
    bind_circuit as bind_circuit,
    google_engine as google_engine,
    google_qvm as google_qvm,
    load_circuit as load_circuit,
)

__all__ = [
    "Cirq",
    "CirqProcessor",
    "CirqSampler",
    "CirqSimulator",
    "DepolarizingNoise",
    "FORMAT",
    "GoogleQVMNoise",
    "bind_circuit",
    "google_engine",
    "google_qvm",
    "load_circuit",
]
