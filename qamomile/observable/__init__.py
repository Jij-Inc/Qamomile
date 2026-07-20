"""Backend-neutral observable and Hamiltonian representation.

Design center
-------------

This package defines *what* an observable is — a ``Hamiltonian`` as a
sum of Pauli-operator products with (near-)real coefficients — with no
dependency on any quantum SDK or on the circuit IR. Backend packages
(``qamomile.qiskit`` / ``qamomile.quri_parts`` / ...) convert it to
their native operator types for estimation; the circuit layer wraps it
for expectation-value ops, Pauli evolution, and canonical hashing.

Design principles
-----------------

- **One shared representation, converted at the edge.** Consumers
  (estimators, executors, emit passes, ``qamomile.linalg``) all speak
  this ``Hamiltonian``; each backend owns its own conversion module.
- **Operator semantics live here, not in backends.** Numerical
  tolerances (zero-coefficient dropping, Hermiticity slack), term
  simplification, and qubit-index remapping are properties of the
  operator itself and are defined once in ``hamiltonian.py`` so every
  consumer agrees on them.
"""

from .hamiltonian import (
    PAULI_TO_CHAR,
    Hamiltonian,
    Pauli,
    PauliOperator,
    X,
    Y,
    Z,
    commutator,
)

__all__ = [
    "PAULI_TO_CHAR",
    "Hamiltonian",
    "Pauli",
    "PauliOperator",
    "X",
    "Y",
    "Z",
    "commutator",
]
