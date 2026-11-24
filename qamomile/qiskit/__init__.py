"""Expose Qiskit compilation and execution support for Qamomile.

QCore and the Rust compiler remain the semantic authorities. This namespace
emits and retains Qiskit circuits and exposes execution endpoints. `to_circuit`
inspects quantum components without creating an executable artifact. QPY is used
only for explicit portable artifact export; SDK handles never enter QCore.
Logical current-shot invocations consume Rust-checked indices in one SDK
submission. Fixed templates use one prepared circuit with numeric parameter
rows or ordered shot memory; topology-dependent recipes retain checked child
circuits. Native result plans run before counts are combined, and adapters
never reconstruct indices from histograms. Measurement-free completion retains
only the certified coordinate count; Rust evaluates each coordinate without
allocating empty input rows. Preparation or endpoint limits
reject oversized requests without silently splitting provider jobs.
``IBMRuntime(dry_run=True)`` selects IBM cloud validation in job or Batch mode.
It preserves the compilation target and returns ``ValidationResult`` with
explicit usage estimates; randomized provider data never becomes Qamomile
measurement samples. Pending usage remains unknown. Saved jobs retain their
dry-run identity, and local-only workloads do not fabricate cloud validation.
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
