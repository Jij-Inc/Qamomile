# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: qamomile
#     language: python
#     name: qamomile
# ---

# %% [markdown]
# # Compilation and Transpilation: Under the Hood
#
# Tutorials 01–07 used the transpiler as a black box: write a `@qkernel`, call
# `transpiler.transpile(...)`, get an executable. This chapter opens the box.
#
# It is aimed at **contributors** — readers who want to:
#
# - Debug a kernel that fails somewhere between tracing and emission
# - Write a custom compiler pass
# - Add a new backend (e.g., a different quantum SDK)
# - Simply understand what `transpile()` actually does
#
# We will walk a small `@qkernel` through the pipeline stage by stage using the
# step-by-step public API on `Transpiler`, inspect the intermediate
# representation at each step, and compare how two backends (Qiskit and
# QURI Parts) turn the same plan into different circuits.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %% [markdown]
# ## 1. The Pipeline at a Glance
#
# `Transpiler.transpile()` is documented in
# `qamomile/circuit/transpiler/transpiler.py` as a composition of ten passes.
# The stages fall into four bands — **frontend → inlining → analysis →
# emission** — separated by **`BlockKind`** transitions:
#
# ```
# QKernel
#    │  to_block                    (tracing: Python AST → IR)
#    ▼
# Block [HIERARCHICAL]
#    │  substitute                  (optional rule-based replacement)
#    │  resolve_parameter_shapes    (concretise Vector shape dims)
#    │  inline                      (remove CallBlockOperations)
#    ▼
# Block [AFFINE]
#    │  unroll_recursion            (iterated inline ↔ partial_eval)
#    │  affine_validate             (safety net for affine types)
#    │  partial_eval                (constant fold + compile-time ifs)
#    │  analyze                     (dependency graph + I/O validation)
#    ▼
# Block [ANALYZED]
#    │  validate_symbolic_shapes    (reject unresolved Vector dims)
#    │  plan                        (segment into C→Q→C)
#    ▼
# ProgramPlan
#    │  emit                        (backend-specific code generation)
#    ▼
# ExecutableProgram[T]
# ```
#
# Every pass is idempotent and exposed as a public method on `Transpiler`, so
# you can run them one at a time and print the `Block` in between. That is the
# single most useful debugging technique in Qamomile.

# %% [markdown]
# ## 2. IR Vocabulary
#
# Before we run any pass, let's name the things we will be printing.
#
# **`Block`** (`qamomile.circuit.ir.block`) is the container that flows through
# the pipeline. It holds:
#
# - `operations`: ordered list of `Operation` instances
# - `input_values` / `output_values`: SSA `Value`s for the kernel's signature
# - `parameters`: dict of unbound parameter names to their `Value`s
# - `kind`: a `BlockKind` tag (`TRACED`, `HIERARCHICAL`, `AFFINE`, or
#   `ANALYZED`) indicating which invariants currently hold
#
# **`BlockKind`** is the pipeline's state machine. Each pass has a precondition
# on `kind` and advances it on success. The progression is monotone:
#
# ```
# TRACED  →  HIERARCHICAL  →  AFFINE  →  ANALYZED
# ```
#
# **`Value`** (`qamomile.circuit.ir.value`) is an SSA-style typed value. Each
# time a qubit passes through a gate, a *new* `Value` is created via
# `Value.next_version()` — the `version` and `uuid` change, but `logical_id`
# (the stable "which physical qubit is this") is preserved, along with the
# type and metadata. Metadata can tag a value as a parameter
# (`with_parameter("theta")`) or a constant (`with_const(2.0)`).
#
# **`Operation`** is the base of the operation hierarchy. Subclasses include:
#
# | Subclass | Purpose | File |
# |----------|---------|------|
# | `GateOperation` | `H`, `RX`, `CX`, … | `ir/operation/gate.py` |
# | `MeasureOperation` | Measurement | `ir/operation/measurement.py` |
# | `ForOperation`, `IfOperation`, `WhileOperation` | Control flow | `ir/operation/control_flow.py` |
# | `CallBlockOperation` | Call to another `Block` (removed by `inline`) | `ir/operation/call_block_ops.py` |
#
# All control-flow ops implement the `HasNestedOps` protocol
# (`nested_op_lists()` / `rebuild_nested()`) so passes can walk into loop and
# branch bodies uniformly, without special-casing each operation type.
#
# Every `Operation` also reports an `operation_kind` (`QUANTUM`, `CLASSICAL`,
# `HYBRID`, `CONTROL`) — this is what the `plan` stage uses to segment the
# block into classical / quantum / expval steps.

# %%
import qamomile.circuit as qmc
from qamomile.circuit.ir.block import BlockKind
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.operation.control_flow import ForOperation
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## 3. The Running Example
#
# We need a kernel small enough to print but rich enough to exercise multiple
# stages:
#
# - A helper `@qkernel` (to exercise **inline**)
# - A `UInt` parameter that drives `qmc.range(n)` (to exercise **partial_eval**)
# - A `Float` parameter we will keep unbound (to exercise **emit**'s parameter
#   handling)


# %%
@qmc.qkernel
def entangle_pair(q0: qmc.Qubit, q1: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Helper subroutine. Inlined into its caller."""
    q0 = qmc.h(q0)
    q0, q1 = qmc.cx(q0, q1)
    return q0, q1


@qmc.qkernel
def demo_kernel(n: qmc.UInt, theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")

    q[0] = qmc.h(q[0])
    for i in qmc.range(n - 1):
        q[i], q[i + 1] = entangle_pair(q[i], q[i + 1])
        q[i + 1] = qmc.rz(q[i + 1], theta)

    return qmc.measure(q)


# %% [markdown]
# We will transpile with `n=3` bound at compile time and `theta` kept as a
# backend parameter.


# %%
def summarise(block):
    """Compact summary of a Block — we will call this after every pass."""
    by_kind = {}
    for op in block.operations:
        by_kind[type(op).__name__] = by_kind.get(type(op).__name__, 0) + 1
    return (
        f"kind={block.kind.name:13s} "
        f"ops={len(block.operations):>2d} "
        f"breakdown={by_kind}"
    )


# %% [markdown]
# ## 4. Stage-by-Stage Walkthrough
#
# We will now run each pass by hand. The helper above prints one line per
# stage so the `BlockKind` and operation mix are easy to compare.
#
# ### 4.1 `to_block` — tracing the Python function
#
# `to_block` executes the decorated function under a tracer context. Every
# `qmc.h(...)`, `qmc.range(...)`, and `entangle_pair(...)` call records an
# `Operation` into the Block. Calls to other `@qkernel`s become
# `CallBlockOperation`s — the body is **not** inlined yet.

# %%
bindings = {"n": 3}
parameters = ["theta"]

block = transpiler.to_block(demo_kernel, bindings=bindings, parameters=parameters)
print("after to_block:   ", summarise(block))
print("parameters:       ", list(block.parameters))
print("CallBlockOps:     ", sum(1 for op in block.operations if isinstance(op, CallBlockOperation)))
# Note: `CallBlockOperation`s may live inside a `ForOperation` body too —
# they are not necessarily in the top-level list.

# %% [markdown]
# The block is `HIERARCHICAL`: it may still contain calls to other blocks and
# composite gates. `block.parameters` mirrors the `parameters=["theta"]`
# argument we passed in. Any input **not** in `parameters` must either be bound
# in `bindings` (like `n`) or consumed by trace-time Python code.
#
# ### 4.2 `inline` — flattening nested block calls
#
# `inline` replaces every `CallBlockOperation` with the operations of the
# target block, substituting SSA values so the result stays well-formed. Once
# no `CallBlockOperation` remains, the block transitions to `AFFINE`.


# %%
def count_calls(ops):
    total = 0
    for op in ops:
        if isinstance(op, CallBlockOperation):
            total += 1
        # Walk nested control-flow bodies so we count calls inside loops.
        for child in getattr(op, "nested_op_lists", lambda: [])():
            total += count_calls(child)
    return total


block = transpiler.inline(block)
print("after inline:     ", summarise(block))
print("CallBlockOps (deep):", count_calls(block.operations))
print("is_affine:        ", block.is_affine())

# %% [markdown]
# Notice what `inline` does **not** do: the `ForOperation` body still has
# `GateOperation`s, one per iteration of the original `for` inside
# `entangle_pair`'s body. Inlining preserves control flow; unrolling happens
# next.
#
# ### 4.3 `partial_eval` — constant folding and compile-time control flow
#
# `partial_eval` has two jobs: fold `BinOp`/`CompOp` nodes whose operands are
# all constants (or bound parameters), and lower `IfOperation` and
# `ForOperation` when their bounds/conditions are resolvable at compile time.
# Because we bound `n=3`, the outer `qmc.range(n - 1)` loop unrolls into two
# copies of its body.

# %%
block = transpiler.partial_eval(block, bindings=bindings)
print("after partial_eval:", summarise(block))
print("ForOperations:    ", sum(1 for op in block.operations if isinstance(op, ForOperation)))

# %% [markdown]
# If you left a `UInt` unbound and tried to use it as a loop bound, the
# downstream `validate_symbolic_shapes` pass would raise
# `QamomileCompileError` with the name of the offending value. That is the
# pass whose job it is to convert "this kernel isn't actually compile-time
# structured" into a readable error rather than a confusing crash later.
#
# ### 4.4 `analyze` — dependency graph and I/O validation
#
# `analyze` builds a dependency graph over values and checks two invariants:
#
# 1. The block's inputs and outputs are classical (quantum I/O is only allowed
#    for *subroutine* blocks, not entrypoints).
# 2. No quantum operation depends on a classical value that was computed from
#    a measurement. That would require JIT compilation, which Qamomile does
#    not currently support — the `plan` stage enforces a single quantum
#    segment.
#
# On success, the block transitions to `ANALYZED`.

# %%
block = transpiler.analyze(block)
print("after analyze:    ", summarise(block))

# %% [markdown]
# ### 4.5 `plan` — segmenting into a `ProgramPlan`
#
# `plan` walks the analyzed block, groups operations by `OperationKind`, and
# assembles a `ProgramPlan` of `ClassicalStep` / `QuantumStep` / `ExpvalStep`
# entries. The `NisqSegmentationStrategy` used by the default transpilers
# enforces **at most one `QuantumStep`** — the canonical C→Q→C pattern.

# %%
plan = transpiler.plan(block)
for i, step in enumerate(plan.steps):
    seg = step.segment
    print(f"  step {i}: {type(step).__name__} ({type(seg).__name__}, {len(seg.operations)} ops)")
print("total unbound parameters:", list(plan.parameters))

# %% [markdown]
# The quantum segment also carries `qubit_values` and `num_qubits` so `emit`
# knows how many qubit lines the backend circuit needs before it starts
# placing gates.
#
# ### 4.6 `emit` — backend-specific code generation
#
# `emit` hands the plan to an `EmitPass` for the target backend. The emit pass
# allocates concrete qubit indices, then walks the quantum segment and calls
# the backend's `GateEmitter` protocol methods (`emit_h`, `emit_rx`, …) to
# construct a native circuit.

# %%
executable = transpiler.emit(plan, bindings=bindings, parameters=parameters)
print("parameter_names:  ", executable.parameter_names)
print()
print(executable.quantum_circuit)

# %% [markdown]
# The surviving parameter is exactly the one we kept (`theta`). All structural
# decisions — qubit count, loop unrolling, which CX sits where — were resolved
# at compile time.
#
# ### 4.7 Passes we skipped
#
# Five passes are part of `transpile()` but we did not call them explicitly:
#
# - **`substitute`** — applies user-configured `SubstitutionRule`s to replace
#   block targets or override composite-gate strategies. No-op when the
#   `TranspilerConfig` has no rules.
# - **`resolve_parameter_shapes`** — fills in `{name}_dim{i}` shape dims when
#   `bindings` provides a concrete `Vector` or `Matrix` value, so that
#   `arr.shape[0]` resolves to a concrete `UInt` downstream.
# - **`unroll_recursion`** — fixed-point loop of `inline ↔ partial_eval` for
#   self-recursive `@qkernel`s (e.g. Suzuki–Trotter — see Tutorial 07).
#   Terminates when the recursion bottoms out or raises if the bindings do
#   not make the base case reachable.
# - **`affine_validate`** — safety net that catches affine-type violations
#   that slipped past frontend checks.
# - **`validate_symbolic_shapes`** — rejects unresolved `Vector` shape dims
#   reaching a `ForOperation` bound, with an actionable error message.
#
# They are idempotent and cheap, so `transpile()` always runs them. As a pass
# author you mostly care about the order: `substitute` and
# `resolve_parameter_shapes` run **before** `inline`; `affine_validate` runs
# **after** it; `validate_symbolic_shapes` runs **after** `analyze` so the
# dependency graph is available.

# %% [markdown]
# ## 5. Backend Emission: Qiskit vs QURI Parts
#
# Every backend plugs into the pipeline by implementing two protocols defined
# in `qamomile/circuit/transpiler/`:
#
# - **`GateEmitter[T]`** (`gate_emitter.py`): the "how do I draw a gate" API.
#   Methods include `create_circuit(num_qubits, num_clbits) -> T`,
#   `create_parameter(name) -> Any`, and ~40 per-gate entry points
#   (`emit_h`, `emit_rx`, `emit_cx`, …). It also advertises a
#   `measurement_mode: MeasurementMode`:
#
#   | Mode | Meaning | Used by |
#   |------|---------|---------|
#   | `NATIVE` | Backend has an explicit measurement instruction the emit pass should call. | Qiskit |
#   | `STATIC` | Backend takes the unmeasured state vector/operator; the sampler handles measurement externally. | QURI Parts |
#   | `RUNNABLE` | Backend supports mid-circuit measurement with runtime control flow. | CUDA-Q (`cudaq.run()` path) |
#
# - **`CompositeGateEmitter[C]`** (`passes/emit.py`): optional. Lets a
#   backend short-circuit composite gates (QFT, QPE, …) with a native
#   implementation. The `can_emit(gate_type) -> bool` / `emit(...) -> bool`
#   contract returns `False` to opt out, in which case the emit pass falls
#   back to the library-level decomposition.
#
# A `Transpiler` subclass wires these together by overriding
# `_create_segmentation_pass` and `_create_emit_pass`, plus `executor()` for
# the runtime side. `qamomile/qiskit/transpiler.py` is the canonical ~50-line
# reference implementation.
#
# Let's transpile the same kernel through QURI Parts and compare.
# QURI Parts is an optional dependency — install with
# `pip install 'qamomile[quri_parts]'` to reproduce this section locally.

# %%
try:
    from qamomile.quri_parts import QuriPartsTranspiler

    quri_transpiler = QuriPartsTranspiler()
    quri_exe = quri_transpiler.transpile(
        demo_kernel, bindings=bindings, parameters=parameters
    )

    print("backend circuit type: ", type(quri_exe.quantum_circuit).__name__)
    print("parameter_names:      ", quri_exe.parameter_names)
    print()
    for gate in quri_exe.quantum_circuit.gates:
        print(" ", gate)
except ModuleNotFoundError:
    # ``qamomile[quri_parts]`` is an optional dependency group — skip the
    # side-by-side when it's not installed so this notebook still runs.
    print("QURI Parts is not installed; skipping the side-by-side output.")

# %% [markdown]
# Three differences worth calling out:
#
# 1. **Circuit type.** Qiskit emits a `QuantumCircuit` with embedded
#    `Parameter` objects; QURI Parts emits a `LinearMappedUnboundParametricQuantumCircuit`
#    whose parameters are QURI Parts `Parameter` instances. Both round-trip
#    through Qamomile's `parameter_names` the same way.
# 2. **Measurement.** Qiskit's circuit ends in `measure` instructions
#    (`measurement_mode=NATIVE`). QURI Parts' circuit has no measurement gates
#    — its executor handles sampling at run time
#    (`measurement_mode=STATIC`).
# 3. **Composite gates.** If the kernel used `qmc.qft(...)`, Qiskit's
#    `QiskitQFTEmitter` would drop in a `QFTGate` box, whereas the QURI Parts
#    backend decomposes via the library pass — same IR, different realised
#    circuit. You can override this per kernel via
#    `TranspilerConfig.with_strategies({"qft": "approximate"})`.

# %% [markdown]
# ## 6. Pointers for Contributors
#
# **Writing a custom pass.** Put it in `qamomile/circuit/transpiler/passes/`,
# take a `Block` in and return a `Block` out, and assert your input `kind`
# precondition up front. When you walk operations, use `HasNestedOps` —
# never `isinstance(op, ForOperation)` chains — so future control-flow ops
# are handled automatically:
#
# ```python
# def rewrite(ops):
#     new_ops = []
#     for op in ops:
#         if hasattr(op, "nested_op_lists"):
#             op = op.rebuild_nested([rewrite(child) for child in op.nested_op_lists()])
#         new_ops.append(transform(op))
#     return new_ops
# ```
#
# **Adding a new backend.** Minimum checklist:
#
# 1. Implement `GateEmitter[T]` for your target SDK (`T` is the SDK's circuit
#    type). Start from `qamomile/qiskit/emitter.py`.
# 2. Subclass `Transpiler[T]` and implement `_create_segmentation_pass` (use
#    `NisqSegmentationStrategy` unless you need something else) and
#    `_create_emit_pass` returning `StandardEmitPass(your_emitter)`.
# 3. Implement a `QuantumExecutor[T]` subclass so users can call `executor()`.
# 4. Optional: add `CompositeGateEmitter`s for QFT/QPE/etc. to preserve
#    high-level structure in the emitted circuit.
#
# **Debugging a transpile error.** Run the passes one at a time with
# `summarise(block)` between them. The stage where `BlockKind` fails to
# advance, the operation count explodes, or an exception is raised is the
# stage you should look at first.

# %% [markdown]
# ## 7. Summary
#
# The pipeline is an SSA-style IR moving through four kinds:
#
# - `HIERARCHICAL` — the raw trace, with block calls still unexpanded
# - `AFFINE` — flat operations + control flow, no block calls
# - `ANALYZED` — validated, dependency-graphed, ready to segment
# - `ProgramPlan` → `ExecutableProgram[T]` — segmented and emitted
#
# Each pass has a narrow job and a precondition on `BlockKind`. The step-by-step
# API on `Transpiler` exposes every pass publicly — treat it as your primary
# debugging tool when a kernel misbehaves, and as the extension surface when
# adding a pass or a backend.
