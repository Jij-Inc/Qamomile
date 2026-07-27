"""Operation builders: tracer-facing functions that emit IR operations.

These are the free functions a qkernel body calls (re-exported from
``qamomile.circuit``): gate builders (``qubit_gates.py``), measurement /
reset / projection (``measurement.py``), the control-flow builders that
the frontend AST transform targets (``control_flow.py``: ``range`` /
``for_items`` / if- and while-machinery, loop region args),
meta-operations (``control.py``, ``inverse.py``), type conversion
(``cast.py``), and Hamiltonian helpers (``expval.py``,
``pauli_evolve.py``).

A builder's job is thin and uniform: validate handle arguments at trace
time (e.g. ``QubitAliasError`` when one qubit fills two roles of a
gate), construct one abstract IR ``Operation`` with next-version output
``Value``s, append it to the active ``Tracer``, and return fresh
handles. Builders must not pre-expand into lower-level operations — one
user call maps to one abstract IR operation (a vector measurement stays
a single ``MeasureVectorOperation``); decomposition and lowering are the
transpiler's and the backends' job.
"""
