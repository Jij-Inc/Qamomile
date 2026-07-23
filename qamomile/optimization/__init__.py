"""Domain-specific helper layer for quantum combinatorial optimization.

Design center
-------------

This package turns classical combinatorial optimization problems
(OMMX ``Instance`` or Qamomile ``BinaryModel``) into quantum programs and
decodes measurement results back into classical solutions. It hosts the
converter family: ``QAOAConverter`` (``qaoa.py``, QUBO/Ising and HUBO via
phase gadgets), ``FQAOAConverter`` (``fqaoa.py``, fermionic mixers with
particle-number constraints), the QRAC converters (``qrao/``),
``PCEConverter`` (``pce.py``, Pauli correlation encoding), and
``QSVTFilterConverter`` (``qsvt_filter.py``, QSVT eigenstate filtering for the
Lin & Tong ground-energy search).

Layering constraint (inviolable)
--------------------------------

``optimization → circuit ← backends``. This package is a *consumer* of
``qamomile.circuit``'s public transpiler and algorithm APIs
(``Transpiler``, ``ExecutableProgram``, ``qamomile.circuit.algorithm``
ansatz builders) — it must never be imported by ``qamomile.circuit`` or
by backend packages, and it must never reach into circuit internals
(passes, IR rewriting). A converter builds a qkernel / Hamiltonian and
hands it to whatever backend ``Transpiler`` the caller supplies; backend
choice stays out of this layer entirely.

Extension points
----------------

- New algorithms subclass ``MathematicalProblemConverter``
  (``converter.py``), whose ``normalize_problem_input`` is the single
  canonical entry point for OMMX/BinaryModel intake (deep-copy +
  ``to_hubo`` semantics live there, in exactly one place). Converters
  whose contract genuinely differs (no single cost Hamiltonian, no
  ``SampleResult`` decode — e.g. PCE) may skip the base class, but
  should still delegate intake to ``normalize_problem_input``.
- ``binary_model/`` is the problem representation; ``post_process/``
  holds classical refinement of decoded samples. Both are
  quantum-agnostic and must stay importable without any SDK installed.
"""
