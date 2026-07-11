"""Multi-pass transpiler pipeline turning traced IR into backend executables.

Design center
-------------

The transpiler is a sequence of small, idempotent IR-rewriting passes
(under ``passes/``), orchestrated by ``Transpiler.transpile()``
(``transpiler.py``). Each pass declares the ``BlockKind`` it expects
(``HIERARCHICAL → AFFINE → ANALYZED`` state machine) and validates it on
entry, so passes cannot silently run out of order. Every pass other than
the inline entrypoint validation is also exposed as a public method on
``Transpiler`` for step-by-step debugging.

Canonical pass sequence
-----------------------

::

    QKernel
       │  to_block                    (trace Python AST → IR)
       │  validate_entrypoint         (internal: EntrypointValidationPass —
       │                               requires classical I/O on entrypoints)
       ▼
    Block [HIERARCHICAL]
       │  substitute                  (optional rule-based block replacement)
       │  resolve_parameter_shapes    (concretize Vector shape dims from bindings)
       │  inline                      (remove inline InvokeOperations)
       ▼
    Block [AFFINE]
       │  unroll_recursion            (iterated inline ↔ partial_eval for
       │                               self-recursive kernels)
       │  affine_validate             (enforce "each quantum value used at most once")
       │  partial_eval                (constant fold + CompileTimeIfLoweringPass)
       │  analyze                     (dependency graph + operand kind check)
       ▼
    Block [ANALYZED]
       │  classical_lowering          (measurement-derived classical ops →
       │                               RuntimeClassicalExpr)
       │  validate_symbolic_shapes    (reject unresolved Vector dims at loop bounds)
       │  plan                        (segment into C→Q→C; pre-segmentation
       │                               lowering of MeasureQFixed etc.)
       ▼
    ProgramPlan
       │  emit                        (backend-specific codegen; LoopAnalyzer
       │                               decides unroll vs runtime loop)
       ▼
    ExecutableProgram[T]

Design principles
-----------------

- **Lower as late as possible.** The IR stays abstract through the
  pipeline; backend-specific concretization (per-qubit encoding, native
  composite gates, runtime control flow) happens only at ``emit``.
  ``plan`` lowers only when segmentation forces a split (HYBRID ops →
  pure-quantum + pure-classical halves), and each half stays as abstract
  as the next stage allows.
- **Backends extend via protocols, not subclass hooks into passes.**
  A backend package implements ``GateEmitter`` (``gate_emitter.py``) —
  including ``MeasurementMode`` for measurement handling and
  ``supports_if_else()`` / ``supports_while_loop()`` capability
  reporting — plus optional composite-gate emitters, and reuses the
  shared decomposition recipes in ``decompositions.py`` as fallback.
  The pass pipeline itself is backend-agnostic.
- **Segmentation is a pluggable strategy.** ``plan`` delegates the
  C→Q→C split to a ``SegmentationStrategy`` (``passes/separate.py``);
  ``NisqSegmentationStrategy`` enforces a single quantum segment. New execution models (JIT, distributed) add
  strategies without touching the core.
- **``bindings`` and ``parameters`` are strictly disjoint.** Compile-time
  bindings are folded into the IR; runtime parameters survive to the
  emitted circuit. ``transpile()`` raises ``ValueError`` on overlap
  (see ``frontend/param_validation.py``); compile-time structural
  decisions (classical ``if`` conditions, ``qmc.range`` bounds) must
  come from ``bindings``.

See ``docs/en/tutorial/09_compilation_and_transpilation.py`` for a
step-by-step walk-through with IR dumps after each pass.
"""
