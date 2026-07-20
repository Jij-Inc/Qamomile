"""Handle type system: user-facing typed wrappers around IR ``Value``s.

Handles are what qkernel code manipulates: quantum primitives (``Qubit``,
``QFixed``), classical scalars (``UInt``, ``Float``, ``Bit``), arrays
(``Vector``, ``VectorView``, ``Matrix``, ``Tensor``), structural
containers (``Tuple``, ``Dict``), and ``Observable`` (Hamiltonian). Each
handle wraps one IR ``Value`` and forwards Python operators (arithmetic,
comparison, indexing) to tracer-emitted IR operations, so user code
reads as ordinary Python while building IR.

Design constraints:

- Quantum handles are linear: operations consume them and return fresh
  handles wrapping a new-version ``Value``; reusing a consumed handle
  raises ``QubitConsumedError``. Classical handles may be read freely.
- Classical arithmetic folds eagerly when both operands are known
  compile-time constants; otherwise a symbolic ``BinOp`` / ``CompOp`` is
  traced for ``partial_eval`` to resolve later.
- Handles are trace-time objects only — they never survive into the
  transpiled program and carry no backend or layout information. The IR
  ``Value`` (with its type and metadata) is the durable representation.
"""

from .array import Matrix, Tensor, Vector, VectorView
from .containers import Dict, Tuple
from .hamiltonian import Observable
from .handle import Handle
from .primitives import Bit, Float, QFixed, Qubit, UInt
from .utils import get_size

__all__ = [
    "Handle",
    "Qubit",
    "QFixed",
    "UInt",
    "Float",
    "Bit",
    "Vector",
    "VectorView",
    "Matrix",
    "Tensor",
    "Observable",
    "Tuple",
    "Dict",
    "get_size",
]
