"""Expose standard-library quantum callables.

The reader-facing API is function-oriented: use :func:`qft`, :func:`iqft`,
and :func:`qpe` inside qkernels. Internally these functions emit named
callables with Qamomile bodies, resource metadata, and optional backend-native
implementations.

The ``QFT`` and ``IQFT`` classes remain public for advanced strategy selection
and backend tests, but custom user-defined named operations should normally use
the ``qamomile.circuit.composite_gate`` decorator rather than subclassing
``CompositeGate`` directly.

Example:
    ```python
    import qamomile.circuit as qmc

    @qmc.qkernel
    def my_algorithm(qubits: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        qubits = qmc.qft(qubits)
        return qmc.iqft(qubits)
    ```
"""

# Advanced class-based stdlib implementations.
from .qft import IQFT, QFT, iqft, qft

# Strategy objects for advanced stdlib configuration.
from .qft_strategies import (
    ApproximateIQFTStrategy,
    ApproximateQFTStrategy,
    StandardIQFTStrategy,
    StandardQFTStrategy,
)
from .qpe import qpe

__all__ = [
    # Classes
    "QFT",
    "IQFT",
    # Strategies
    "StandardQFTStrategy",
    "ApproximateQFTStrategy",
    "StandardIQFTStrategy",
    "ApproximateIQFTStrategy",
    # Functions
    "qft",
    "iqft",
    "qpe",
]
