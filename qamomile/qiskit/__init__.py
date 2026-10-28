"""Qiskit engine for Qamomile.

Design intent: this package concretizes circuit's abstract IR for Qiskit
through ``QiskitMaterializer``. ``QiskitTranspiler`` plugs the materializer
into the shared compiler pipeline, while ``observable.py`` converts
Hamiltonians to ``SparsePauliOp``.

``QiskitExecutor`` uses a local simulator by default and adapts IBM Runtime
V2 primitives for named or preconfigured IBM backends, including account
access checks, ISA compilation, and remote job lifecycles.

Constraints: depend only on ``qamomile.circuit`` public APIs plus the
``qiskit`` SDK — never on ``qamomile.optimization`` or other engines.
Engine-specific lowering (decompositions, runtime control flow) belongs
here at emit time, not in the IR; reuse circuit's shared decomposition
recipes as the fallback for gates without a native Qiskit equivalent.
"""

from qamomile.qiskit.execution import QiskitExecutionOptions
from qamomile.qiskit.observable import hamiltonian_to_sparse_pauli_op
from qamomile.qiskit.transpiler import QiskitExecutor, QiskitTranspiler

__all__ = [
    "QiskitTranspiler",
    "QiskitExecutor",
    "QiskitExecutionOptions",
    "hamiltonian_to_sparse_pauli_op",
]
