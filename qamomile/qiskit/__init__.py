"""Qiskit backend for Qamomile: emit pass, executor, and observable conversion.

Design intent: this package concretizes circuit's abstract IR for Qiskit
by implementing circuit's extension protocols — ``QiskitTranspiler``
(``transpiler.py``) plugs a Qiskit ``EmitPass`` into the shared pipeline,
``QiskitGateEmitter`` (``emitter.py``) implements the ``GateEmitter``
protocol (``MeasurementMode.NATIVE``; runtime ``if``/``while`` support
reported via the emitter capability methods), ``emitters/`` holds
optional native composite-gate emitters (e.g. QFT), and ``observable.py``
converts Hamiltonians to ``SparsePauliOp``.

Constraints: depend only on ``qamomile.circuit`` public APIs plus the
``qiskit`` SDK — never on ``qamomile.optimization`` or other backends.
Backend-specific lowering (decompositions, runtime control flow) belongs
here at emit time, not in the IR; reuse circuit's shared decomposition
recipes as the fallback for gates without a native Qiskit equivalent.
"""

from qamomile.qiskit.observable import hamiltonian_to_sparse_pauli_op
from qamomile.qiskit.transpiler import QiskitExecutor, QiskitTranspiler

__all__ = [
    "QiskitTranspiler",
    "QiskitExecutor",
    "hamiltonian_to_sparse_pauli_op",
]
