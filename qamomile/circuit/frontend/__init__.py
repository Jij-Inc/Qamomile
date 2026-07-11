"""Frontend: Python-embedded tracing that turns qkernel functions into IR Blocks.

The ``qkernel`` decorator (``qkernel.py``) first rewrites the user's
function with an AST transform (``ast_transform.py``) that replaces
native ``if`` / ``while`` / ``for`` statements with tracer-visible
builder calls, then executes the transformed function under a ``Tracer``
(``tracer.py``) with handle arguments. Every handle method and operation
builder appends abstract IR ``Operation``s to the active tracer, and
``func_to_block.py`` packages the trace into a ``HIERARCHICAL`` ``Block``
(including the ``ParamSlot`` manifest for classical arguments).

Design constraints:

- Handles (``handle/``) are linear-typed wrappers over IR ``Value``s:
  gates consume their quantum arguments and return fresh handles that
  must be re-bound (SSA-like versioning). Reuse of a consumed handle
  raises ``QubitConsumedError`` at trace time, before the transpiler's
  ``affine_validate`` re-checks the same invariant on the IR.
- The frontend owns tracing, the handle type system, and Python-syntax
  lowering (control-flow rewriting, loop region args). It emits abstract
  IR only; whole-block validation, IR rewriting, segmentation, and
  backend concretization belong to ``qamomile.circuit.transpiler``.
- Calls to nested qkernels stay as ``InvokeOperation`` boxes carrying a
  ``CallPolicy``; the frontend never inlines — ``inline`` is a
  transpiler pass.
"""
