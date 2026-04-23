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
# ##### Aside: why is it called `AFFINE`?
#
# In programming-language type theory, types are classified by how many times
# a value may be used:
#
# | Flavour | Uses allowed | Example |
# |---------|--------------|---------|
# | Unrestricted (ordinary types) | 0 or more times | Python `int`, a classical bit — values you can copy freely. |
# | **Affine** | **at most once (0 or 1 times)** | **A qubit.** |
# | Linear | exactly once (discarding is forbidden) | Values you must consume. |
#
# Qubits are affine because of the **no-cloning theorem**: the same quantum
# state cannot be duplicated. Once `q` has been consumed by `qmc.h(q)`, the
# old `q` value is gone — only the new version `q'` is usable. That is
# exactly what "at most once" means.
#
# Discarding a quantum value without using it is still allowed (the final
# measurement is what typically consumes it, but forgetting a qubit is not a
# type error). This is why Qamomile picks *affine* rather than *linear* —
# linear types would require every qubit to be explicitly consumed.
#
# `BlockKind.AFFINE` means the block has reached a shape where this affine
# invariant ("each quantum value is used at most once") can be checked.
# `AffineValidationPass` does the actual check and raises `AffineTypeError`
# on a violation.
#
# **`Value`** (`qamomile.circuit.ir.value`) is an SSA-style typed value. Not
# only `Qubit`s but every IR value — `Float`, `UInt`, `Bit`, and so on — is
# represented as a `Value`. Whenever the value is updated (by a gate, a
# classical operation, or assignment), `Value.next_version()` produces a
# fresh copy with a new `version` and `uuid`; the `logical_id`, type, and
# metadata are preserved.
#
# `logical_id` is a stable identifier that says "this is still the same
# logical variable across SSA versions" — e.g. `q = qmc.h(q)` creates a new
# `Value` whose `logical_id` matches the old one. It is **not** a mapping to
# a physical qubit; backend qubit allocation happens later in `emit` via the
# `ResourceAllocator`. The same mechanism is reused for classical values
# such as `Float` parameters and `Bit` measurement results.
#
# Metadata can tag a value as a parameter (`with_parameter("theta")`) or a
# constant (`with_const(2.0)`).
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
from qamomile.circuit.ir import pretty_print_block
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
# When the one-line summary is not enough,
# `qamomile.circuit.ir.pretty_print_block` returns an MLIR-style textual
# dump of the block — the fastest way to see *what changed* between two
# passes. The `depth` argument controls how many layers of
# `CallBlockOperation` to expand inline, so e.g. `depth=1` previews what
# `inline` will produce without running it.

# %% [markdown]
# ## 4. Stage-by-Stage Walkthrough
#
# We will now run each pass by hand. The `summarise` helper prints one line
# per stage so the `BlockKind` and operation mix are easy to compare; drop
# in `pretty_print_block` wherever you want the full picture.
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
# Let's look at the block itself. `pretty_print_block` renders it as
# MLIR-style text; you can see that the `for` body still contains a live
# `call entangle_pair(...)`.

# %%
print(pretty_print_block(block))

# %% [markdown]
# With `depth=1`, the `CallBlockOperation` is expanded inside its call line
# — the same shape `inline` will produce in the next stage, so you can
# preview it without actually running the pass.

# %%
print(pretty_print_block(block, depth=1))

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
# Pretty-printing again confirms that `call entangle_pair(...)` has vanished
# and its body (`h` / `cx`) sits directly inside the `for`. The block's
# `kind` has advanced to `AFFINE`.

# %%
print(pretty_print_block(block))

# %% [markdown]
# Notice what `inline` does **not** do: the `ForOperation` body still has
# `GateOperation`s, one per iteration of the original `for` inside
# `entangle_pair`'s body. Inlining preserves control flow; unrolling happens
# later in `emit`.
#
# ### 4.3 `partial_eval` — constant folding and compile-time `if` removal
#
# `partial_eval` is composed of two sub-passes:
#
# 1. **`ConstantFoldingPass`** — folds `BinOp`/`CompOp` nodes whose operands
#    are all constants (or bound parameters) into literal values. Because we
#    bound `n=3`, the `n - 1` inside `qmc.range(n - 1)` collapses to `2`,
#    making the `ForOperation` bounds concrete.
# 2. **`CompileTimeIfLoweringPass`** — when an `IfOperation`'s condition
#    resolves at compile time, it is replaced by the selected branch's
#    operations. Measurement-backed `IfOperation`s are left alone.
#
# Note that `ForOperation` itself is **not** unrolled here. Loop unrolling, if
# needed, is decided later by `LoopAnalyzer` during `emit` (see section 5),
# so the `ForOperations` count does not drop in this stage.

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
# 2. No **`OperationKind.QUANTUM`** operation receives a **classical-typed
#    operand** whose value was computed from a measurement — concretely, the
#    rotation angle in `rx(q, theta)` cannot be a classical value derived
#    from an earlier measurement, because the backend would have to JIT a
#    classical computation between measurement and gate.
#
# This rule **does not forbid dynamic quantum circuits**: `IfOperation` and
# `WhileOperation` are `OperationKind.CONTROL`, not QUANTUM, so control flow
# conditioned on a measurement `Bit` (`if bit: ...`, `while bit: ...`)
# passes the check. Quantum-typed values that survive a phi merge are also
# explicitly exempt. Section 5 walks through which dynamic patterns are
# allowed and which are rejected, with code examples.
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
# ## 5. Control Flow (`if` / `for` / `while`) Through the Pipeline
#
# How the pipeline handles control flow spans several layers — what the
# frontend accepts, how each pass transforms it, and whether the backend
# supports runtime branching. This section ties those layers together. See
# [tutorial 05](05_classical_flow_patterns) for the user-facing patterns;
# here we focus on the compiler's view.
#
# ### 5.1 What the Frontend Accepts
#
# `@qkernel` rewrites the function AST before tracing (see
# `ControlFlowTransformer` in `qamomile/circuit/frontend/ast_transform.py`):
#
# - Python `if` → `emit_if(cond, true_branch, false_branch, ...)`
# - Python `for` → `for_loop(start, stop, step)` or `for_items(dict)`
#   context managers
#
# Because of this, using native Python `if` / `for` on runtime values behaves
# intuitively: both branches / every iteration are traced. Supported loop
# sources:
#
# - **`qmc.range(n)`** — symbolic bounds; `n` can stay unbound and survive
#   as a `ForOperation` in the IR.
# - **`qmc.items(d)`** — for dicts / sparse data. **Always unrolled at
#   compile time** (`ForItemsOperation`).
# - A bare `for i in <runtime_value>:` — rejected. Always go through
#   `qmc.range(...)` or `qmc.items(...)`.
#
# `while` loops use the `while_loop` context manager. The condition must be
# a **measurement-backed `Bit`** — classical variables or constants are
# rejected downstream by `ValidateWhileContractPass`.
#
# ### 5.2 IR Representation
#
# From `qamomile/circuit/ir/operation/control_flow.py`:
#
# | Operation | Nested lists | Condition / bounds | Notes |
# |-----------|-------------|--------------------|-------|
# | `ForOperation` | `operations` (body) | `operands = [start, stop, step]` (all `UInt`) | carries `loop_var` name |
# | `ForItemsOperation` | `operations` (body) | `operands[0]` is a `DictValue` | always unrolled at transpile time |
# | `IfOperation` | `true_operations`, `false_operations` | `operands[0]` is a `Bit` | `phi_ops` merge values post-branch |
# | `WhileOperation` | `operations` (body) | `operands[0]` (initial), `operands[1]` (loop-carried) | measurement-backed `Bit` required; optional `max_iterations` hint |
#
# All four implement `HasNestedOps`, so passes walk into bodies uniformly via
# `nested_op_lists()` / `rebuild_nested()` — no isinstance chains.
#
# `IfOperation` carries **Phi nodes** (`PhiOp`) to merge values. When both
# branches update the same logical value, readers past the if refer through
# the phi to know which branch's version they get.
#
# ### 5.3 Per-Pass Behavior
#
# | Pass | `IfOperation` | `ForOperation` | `WhileOperation` |
# |------|-------------|---------------|-----------------|
# | `inline` | recurses into both branch bodies | recurses into body | recurses into body |
# | `partial_eval` | constant condition → **replaced by selected branch** (`CompileTimeIfLoweringPass`); measurement condition preserved | bound `BinOp`s folded. **No unrolling** | untouched here |
# | `analyze` | phi edges enter the dependency graph | `loop_var` enters body deps | measurement-condition treated like a quantum operand |
# | `validate_symbolic_shapes` | — | unresolved `Vector` shape dim as a bound → rejected | — |
# | `plan` | `OperationKind.CONTROL` — creates a segment boundary | same | same |
# | `emit` | emitted as runtime `if` if the backend supports it | `LoopAnalyzer.should_unroll()` decides unroll vs native-loop | emitted as runtime `while` |
#
# **`LoopAnalyzer.should_unroll()`**
# (`transpiler/passes/emit_support/loop_analyzer.py`) unrolls when:
#
# 1. Loop bounds depend on an outer loop variable (dynamic nesting)
# 2. The body indexes an array with `loop_var` (e.g. `q[i]`)
# 3. `loop_var` appears in a `BinOp` (e.g. `i + 1`, `2 * i`)
#
# Our `demo_kernel` uses both `q[i]` and `q[i + 1]`, so all three triggers
# fire and the loop is unrolled at emit time — that's why
# `executable.quantum_circuit` is a flat sequence of CXs for two iterations.
# A loop that hits none of these stays in the circuit as a native runtime
# loop for backends that support it.
#
# ### 5.4 Quantum ↔ Classical Dependency Rule (`analyze`)
#
# `analyze` enforces the invariant that **quantum operations must not depend
# on classical values derived from measurements**:
#
# ```python
# # OK: a measurement Bit conditions a quantum gate
# b = qmc.measure(q)
# if b:
#     q = qmc.x(q)
#
# # NG: a classical value derived from a measurement feeds a quantum gate
# b = qmc.measure(q)
# x = some_classical(b)
# q = qmc.rx(q, x)   # rejected by analyze
# ```
#
# In the first case the `Bit` is only used as an `IfOperation` condition —
# no quantum operand type is rewritten. The second case requires JIT
# compilation, which Qamomile does not support today. The `plan` stage
# enforcing a single quantum segment is the other side of this guarantee.
#
# ### 5.5 Backend Runtime-Branching Support
#
# Whether runtime `if` / `while` survives into the emitted circuit depends
# on the backend's `MeasurementMode`
# (`qamomile/circuit/transpiler/gate_emitter.py`):
#
# | Mode | Runtime if/while | Example |
# |------|------------------|---------|
# | `NATIVE` | Supported — conditional gates are emitted explicitly | Qiskit (e.g. `QuantumCircuit.if_test`) |
# | `STATIC` | Not supported — returns the pre-measurement state / operator | QURI Parts |
# | `RUNNABLE` | Fully supported, including runtime loops / branches | CUDA-Q (`cudaq.run()` path) |
#
# Compiling a kernel that contains an `IfOperation` / `WhileOperation` on a
# non-supporting backend will raise at emit time. It is up to the
# contributor to know which mode applies when writing runtime branching.
#
# ### 5.6 Common Errors
#
# - **`ValidationError` (analyze)** — a classical value derived from a
#   measurement was used as a quantum operand. Rewrite the pattern, or
#   redesign to keep state on the quantum side.
# - **`ValidateWhileContractPass` error** — the `while` condition is not a
#   measurement-backed `Bit`. Classical variables or constants are not
#   supported as `while` conditions.
# - **`QamomileCompileError` (validate_symbolic_shapes)** — an unresolved
#   `Vector` shape dim reached a `ForOperation` bound. Concretise the
#   `Vector` via `bindings`, or switch to `qmc.items`.
# - **Emit-time error** — a runtime `if` reached a `MeasurementMode.STATIC`
#   backend. Switch backend, or express the kernel differently.

# %% [markdown]
# ## 6. Backend Emission: Qiskit vs QURI Parts
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
# ## 7. Pointers for Contributors
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
# `summarise(block)` between them to track counts, and reach for
# `pretty_print_block(block)` whenever you need to see the actual IR. The
# stage where `BlockKind` fails to advance, the operation count explodes,
# or an exception is raised is the stage you should look at first. Varying
# `pretty_print_block(block, depth=N)` before and after `inline` makes it
# much easier to spot where a value got disconnected or a phi got dropped.

# %% [markdown]
# ## 8. Summary
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
#
# Control-flow highlights:
#
# - `if` / `for` are rewritten at trace time into `IfOperation` /
#   `ForOperation` / `ForItemsOperation` / `WhileOperation` IR nodes
# - `partial_eval` removes compile-time `if`s; `for`-loop unrolling is
#   decided later by `LoopAnalyzer` during `emit`
# - `analyze` guarantees that quantum operations do not depend on classical
#   values derived from measurements
# - Whether runtime branching survives into the circuit depends on the
#   backend's `MeasurementMode` (`NATIVE` or `RUNNABLE` required)
