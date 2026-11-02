# Quantum authoring branches

Read the branch that matches the quantum part of the task:

- [Ownership and register windows](quantum/ownership.md): indexed gates,
  splitting, concatenation, and dependent-width helpers.
- [Coherent control](quantum/control.md): callable control, basis patterns,
  and selecting a control lane.
- [Measured and classical flow](quantum/flow.md): scalar results, per-shot
  calculations, measure-reset loops, and typed sparse maps.
- [Reusable components](quantum/reuse.md): annotated helpers, callable
  parameters, and composition inside kernels.
- [Coherent integer arithmetic](quantum/arithmetic.md): little-endian
  preparation, retained operands, modular operations, and comparison.

Qamomile compiles source rather than executing a kernel to trace operations.
Compiler diagnostics identify the source, type, stage, or ownership contract
that needs repair. Keep `Const < Param < Shot` through aggregates and branches.
