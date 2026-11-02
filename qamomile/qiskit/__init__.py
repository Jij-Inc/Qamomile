"""Expose Qiskit compilation and execution support for Qamomile.

QCore and the Rust compiler remain the semantic authorities. This namespace
emits and retains Qiskit circuits and exposes execution endpoints. `to_circuit`
inspects quantum components without creating an executable artifact. QPY is used
only for explicit portable artifact export; SDK handles never enter QCore.
`profile_from_backend` derives an immutable, Rust-validated ``TargetProfile``
(coupling map, native operations, calibration rows) from a live backend target;
its digest enters the target fingerprint and extension passes read it as
``context.profile``.
"""

from ._integration import (
    FORMAT as FORMAT,
    Aer as Aer,
    IBMRuntime as IBMRuntime,
    Qiskit as Qiskit,
    QiskitBackend as QiskitBackend,
    bind_circuit as bind_circuit,
    decode_counts as decode_counts,
    load_circuit as load_circuit,
    profile_from_backend as profile_from_backend,
    to_circuit as to_circuit,
)

__all__ = [
    "Aer",
    "FORMAT",
    "IBMRuntime",
    "Qiskit",
    "QiskitBackend",
    "bind_circuit",
    "decode_counts",
    "load_circuit",
    "profile_from_backend",
    "to_circuit",
]
