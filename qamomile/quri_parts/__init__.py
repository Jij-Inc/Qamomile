"""QURI Parts backend for Qamomile.

Design intent: this package concretizes circuit's abstract IR for QURI
Parts through ``QuriPartsMaterializer``. ``QuriPartsTranspiler`` plugs the
materializer into the shared compiler pipeline, while ``observable.py``
converts Hamiltonians to QURI Parts ``Operator``s.

Backend-specific quirks stay confined here: parametric circuits use
``LinearMappedUnboundParametricQuantumCircuit``, so symbolic angles are
normalized to linear-combination form (``{param: coeff, CONST: offset}``)
at emit time, and controlled gates without a native equivalent fall back
to circuit's shared decomposition recipes. Failures surface as
``QamomileQuriPartsTranspileError`` (``exceptions.py``) rather than raw
SDK errors.

Constraints: depend only on ``qamomile.circuit`` public APIs plus the
``quri-parts`` SDK — never on ``qamomile.optimization`` or other
backends. Lowering decisions belong at emit time, not in the IR.
"""

from .emitter import QuriPartsGateEmitter
from .exceptions import QamomileQuriPartsTranspileError
from .observable import hamiltonian_to_quri_operator, to_quri_operator
from .transpiler import QuriPartsExecutor, QuriPartsTranspiler

__all__ = [
    "QuriPartsTranspiler",
    "QuriPartsExecutor",
    "QuriPartsGateEmitter",
    # Observable support
    "hamiltonian_to_quri_operator",
    "to_quri_operator",
    # Exceptions
    "QamomileQuriPartsTranspileError",
]
