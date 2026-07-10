"""Expose standard-library quantum callables.

The reader-facing API is function-oriented: use :func:`qft`, :func:`iqft`,
and :func:`qpe` inside qkernels. Internally these functions emit named
callables with Qamomile bodies, resource metadata, and optional backend-native
implementations.

QFT and IQFT use the same ``composite_gate`` mechanism as user callables; there
is no separate class-based gate hierarchy.

Example:
    ```python
    import qamomile.circuit as qmc

    @qmc.qkernel
    def my_algorithm(qubits: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        qubits = qmc.qft(qubits)
        return qmc.iqft(qubits)
    ```
"""

from .arithmetic import modmul_const
from .grover import grover_iteration_count, grover_search
from .qft import iqft, qft
from .qpe import qpe
from .shor import shor_order_finding

__all__ = [
    "qft",
    "iqft",
    "qpe",
    # Arithmetic and algorithms
    "modmul_const",
    "shor_order_finding",
    "grover_search",
    "grover_iteration_count",
]
