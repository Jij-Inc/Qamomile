"""Amazon Braket engine for Qamomile.

Design intent: this package concretizes the engine-neutral circuit IR as
native Amazon Braket ``Circuit`` objects. The transpiler and executor depend
only on Qamomile's public circuit and observable APIs plus the optional
``amazon-braket-sdk`` dependency; the compiler core never depends on Braket.

Static terminal measurements are retained as Qamomile mapping metadata and
performed by the executor. Engine-specific gate control, inversion, global
phase, parameter binding, and Pauli-evolution lowering stay at this emit
boundary rather than leaking into the shared IR.
"""

from qamomile.braket.execution import BraketExecutionOptions
from qamomile.braket.observable import hamiltonian_to_braket_observable
from qamomile.braket.transpiler import BraketExecutor, BraketTranspiler

__all__ = [
    "BraketExecutor",
    "BraketExecutionOptions",
    "BraketTranspiler",
    "hamiltonian_to_braket_observable",
]
