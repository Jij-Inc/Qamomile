"""qBraid execution backend for Qamomile (executor-only, no transpiler).

Design intent: unlike the other backend packages, qBraid provides no
emit pass or ``GateEmitter`` of its own. ``QBraidExecutor``
(``executor.py``) implements circuit's ``QuantumExecutor[QuantumCircuit]``
protocol over circuits *already emitted by the Qiskit backend*, routing
them to qBraid-accessible devices through the qBraid runtime. Compile
with ``QiskitTranspiler``, execute with ``QBraidExecutor``.

Constraints:
- Depends only on ``qamomile.circuit`` (plus the optional ``qbraid`` and
  ``qiskit`` SDKs); never imported by circuit or other backends.
- Execution requires qBraid credentials (API key / ``QbraidProvider``),
  so this backend is out of scope for the mandatory cross-backend test
  matrix; tests must gate on credentials and skip otherwise.
- ``estimate()`` is counts-based and rejects circuits with pre-existing
  classical bits rather than risk silently wrong results.
"""

from qamomile.qbraid.executor import QBraidExecutor

__all__ = [
    "QBraidExecutor",
]
