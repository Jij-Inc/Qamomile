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
# # Inside the Transpiler: From @qkernel to Executable Program
#
# This tutorial takes you on a guided tour through Qamomile's transpiler pipeline.
# In earlier tutorials, you called `transpiler.transpile()` as a single step.
# Here, you will see what happens under the hood: each pass that transforms
# your `@qkernel` Python function into a backend-specific executable program.
#
# ## What You Will Learn
# - The full transpiler pipeline and the role of each pass
# - The difference between `kernel.block` and `transpiler.to_block()`
# - How `draw()` visualizes circuits at different levels of detail
# - How kernel calls are inlined into a flat instruction sequence
# - How validation, constant folding, and dependency analysis work
# - How the program is separated into quantum and classical segments
# - How the emit pass generates backend-specific code
# - How `TranspilerConfig` controls composite gate decomposition strategies

# %% [markdown]
# ## 1. Overview
#
# When you write a `@qkernel` and call `transpiler.transpile()`, Qamomile runs
# the following pipeline of passes:
#
# ```
# @qkernel
#     |  to_block
#     v
#   Block (HIERARCHICAL)
#     |  inline
#     v
#   Block (LINEAR)
#     |  linear_validate
#     v
#   Block (validated)
#     |  constant_fold
#     v
#   Block (constants evaluated)
#     |  analyze
#     v
#   Block (ANALYZED)
#     |  separate
#     v
#   SeparatedProgram (classical_prep -> quantum -> classical_post)
#     |  emit
#     v
#   ExecutableProgram (backend-specific circuits + post-processing)
# ```
#
# Each pass transforms an intermediate representation (IR) into a more refined
# form. The key insight is that quantum and classical operations start out
# interleaved in a single block, and the pipeline progressively validates,
# simplifies, and finally separates them so the quantum portion can be sent
# to a real device while classical post-processing runs on a CPU.
#
# We will use a **single circuit** throughout Sections 2–6, so you can
# follow every transformation from start to finish. Then in Section 7 we
# show QPE as a complete end-to-end example.

# %% [markdown]
# ## 2. Building a Block and Drawing
#
# The first step is converting a `@qkernel` into a `Block` — Qamomile's
# intermediate representation (IR). Let's define the circuit we will use
# throughout this tutorial.

# %%
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler


@qmc.qkernel
def flip(q: qmc.Qubit) -> qmc.Qubit:
    """Apply X gate."""
    return qmc.x(q)


@qmc.qkernel
def prepare(q: qmc.Qubit) -> qmc.Qubit:
    """Flip then apply H."""
    q = flip(q)
    q = qmc.h(q)
    return q


@qmc.qkernel
def my_circuit(n: qmc.UInt, theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Prepare, rotate, entangle, and measure."""
    doubled_angle = theta * 2
    qs = qmc.qubit_array(n, name="qs")
    qs[0] = prepare(qs[0])
    qs[0] = qmc.rz(qs[0], doubled_angle)
    qs[0], qs[1] = qmc.cx(qs[0], qs[1])
    return qmc.measure(qs)


# %% [markdown]
# This circuit has three layers of nesting: `my_circuit` calls `prepare`,
# which calls `flip`. It also computes `theta * 2` before qubit
# allocation — this classical arithmetic will show up as a `BinOp`
# operation and play a key role in the constant folding and segment
# separation passes later.
#
# The `draw()` method lets you visualize the circuit at different levels
# of detail.

# %%
# Default: sub-kernel calls shown as boxes
my_circuit.draw(n=2)

# %%
# inline=True with inline_depth=1: expand one level only.
# prepare is expanded (showing flip as a box + H), but flip stays as a box.
my_circuit.draw(n=2, inline=True, inline_depth=1)

# %%
# inline=True (unlimited depth): fully expanded to primitive gates.
my_circuit.draw(n=2, inline=True)

# %% [markdown]
# The `inline_depth` parameter controls how many levels of nesting are
# expanded:
# - `None` (default when `inline=True`) — unlimited, expand everything
# - `0` — no inlining (same as `inline=False`)
# - `1` — expand only the top-level calls

# %% [markdown]
# ### kernel.block vs transpiler.to_block()
#
# There are two ways to get the IR block:
#
# - **`kernel.block`** — A cached property that builds the block without
#   any parameter bindings.
# - **`transpiler.to_block(kernel, bindings=...)`** — Builds the block with
#   concrete parameter bindings. This is important when the kernel uses
#   parameters to determine array sizes (e.g., `qmc.qubit_array(n, ...)`).
#
# Since `my_circuit` uses `n` for the qubit array size, these two
# approaches produce **different** results.

# %%
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.operation.gate import GateOperation


def print_block_operations(block: Block):
    """Print all operations in a block."""
    for op in block.operations:
        print(op.__class__.__name__ + ":", end="")
        if isinstance(op, CallBlockOperation):
            print(op.operands[0].name)
        elif isinstance(op, GateOperation):
            print(op.gate_type)
        else:
            print("")


# %%
transpiler = QiskitTranspiler()

# Without bindings — n is symbolic, array size unresolved
print("=== kernel.block (no bindings) ===")
print_block_operations(my_circuit.block)

# %%
# With bindings — n=2 resolves the qubit array to 2 elements
print("=== transpiler.to_block(bindings={'n': 2}) ===")
block = transpiler.to_block(my_circuit, bindings={"n": 2})
print_block_operations(block)

# %% [markdown]
# Notice that `QInitOperation` appears in the output. This is the qubit
# allocation and initialization step — it tells the compiler how many
# physical qubits to create. The emit pass uses this information to set up
# the backend circuit.
#
# The `CallBlockOperation` for `prepare` has not yet been expanded — the
# block is still in its **hierarchical** form (it may contain nested calls).

# %% [markdown]
# ## 3. The Inline Pass
#
# The **inline** pass takes the hierarchical block and expands all
# `CallBlockOperation`s (calls to other qkernels) into a flat, linear
# sequence of primitive operations. After inlining, no nested calls remain.

# %%
print("=== Before inlining ===")
print_block_operations(block)

# %%
inlined = transpiler.inline(block)
print("=== After inlining ===")
print_block_operations(inlined)

# %% [markdown]
# The `CallBlockOperation` for `prepare` has been fully expanded:
# `prepare` called `flip` (which is an X gate) and then applied an H gate.
# Both are now visible as `GateOperation`s in the flat block.
# The `BinOp` for `theta * 2` is also visible — it is a classical
# operation that will be handled by `constant_fold` in the next step.

# %% [markdown]
# ## 4. Validation, Folding, and Analysis — Step by Step
#
# After inlining, the transpiler runs three passes that validate correctness
# and prepare the block for segment separation. Let's look at each one.
#
# ### 4a. `linear_validate` — Enforcing No-Cloning
#
# The `linear_validate` pass enforces **linear type semantics**: each quantum
# value (qubit) must be consumed exactly once. This enforces the no-cloning
# principle at compile time rather than at execution time.

# %%
validated = transpiler.linear_validate(inlined)
print("=== After linear_validate ===")
print_block_operations(validated)

# %% [markdown]
# The operations are unchanged — validation passed because each qubit
# in `my_circuit` is used exactly once per operation. If you tried to use
# the same qubit twice (e.g., `qmc.cx(q, q)`), the pass would raise a
# `LinearTypeError` before the circuit ever reaches the quantum hardware.
#
# ### 4b. `constant_fold` — Evaluating Known Constants
#
# The `constant_fold` pass evaluates arithmetic expressions when all operands
# are known constants or bound parameters. Our circuit contains
# `doubled_angle = theta * 2`, which appears as a `BinOp` operation. When
# we bind `theta = 0.5`, the pass evaluates `0.5 * 2 = 1.0` and removes
# the `BinOp`.

# %%
print("=== Before constant_fold ===")
print_block_operations(validated)

# %%
folded = transpiler.constant_fold(validated, bindings={"theta": 0.5})
print("=== After constant_fold ===")
print_block_operations(folded)

# %% [markdown]
# The `BinOp` for `theta * 2` has been evaluated and removed. The Rz gate
# now uses the concrete value `1.0` directly. If `theta` were left unbound,
# the `BinOp` would survive and end up in a `classical_prep` segment
# during separation — we will demonstrate this in Section 5.
#
# ### 4c. `analyze` — Dependency Graph and I/O Validation
#
# The `analyze` pass builds a dependency graph and validates:
# - All block inputs and outputs are classical types (no quantum I/O at
#   the top level)
# - No quantum operation depends on a measurement result (which would require
#   mid-circuit measurement support, not yet available)

# %%
analyzed = transpiler.analyze(folded)
print("=== After analyze ===")
print_block_operations(analyzed)

# %% [markdown]
# The operations are unchanged again — the analysis confirmed that:
# - Inputs `n` (UInt) and `theta` (Float) are classical
# - Output `Vector[Bit]` is classical
# - No quantum gate depends on a measurement result
#
# The block is now in the **ANALYZED** state, with a dependency graph
# attached. If any validation fails, you get a clear error message pointing
# to the problematic operation.

# %% [markdown]
# ## 5. Segment Separation
#
# The **separate** pass splits the analyzed block into distinct segments:
#
# - **`classical_prep`**: Classical operations that run *before* the quantum
#   circuit (e.g., pre-computation of parameters that could not be folded
#   at compile time). When all parameters are bound, `constant_fold` resolves
#   them, so this segment is typically `None`.
#
# - **`quantum`**: The quantum operations (gates, measurements).
#
# - **`classical_post`**: Classical operations that run *after* measurement
#   (e.g., `DecodeQFixedOperation` which converts a `QFixed` bitstring into a
#   `Float` value). This segment handles type conversions that let users
#   receive high-level results like `Float` instead of raw bitstrings.
#
# This three-part structure reflects the reality of current quantum hardware:
# classical preparation happens on a CPU, the quantum circuit runs on a QPU,
# and classical post-processing runs on a CPU again.
#
# ### 5a. All parameters bound
#
# When all parameters are bound, `constant_fold` resolves all arithmetic,
# so no classical preparation is needed.

# %%
separated = transpiler.separate(analyzed)

print(f"classical_prep:  {separated.classical_prep}")
print(f"quantum:         {separated.quantum.kind.name}")
print(f"classical_post:  {separated.classical_post}")
print(f"boundaries:      {len(separated.boundaries)}")
print()
print("=== Quantum segment operations ===")
for op in separated.quantum.operations:
    print(f"  {op.__class__.__name__}")

# %% [markdown]
# As expected:
# - No `classical_prep` — `theta * 2` was folded to `1.0` by `constant_fold`.
# - One **QUANTUM** segment with gate and measurement operations.
# - No `classical_post` — `Bit` measurements need no decoding.
#
# ### 5b. Runtime parameter — `classical_prep` in action
#
# What happens when `theta` is *not* bound at compile time? The `BinOp`
# for `theta * 2` cannot be resolved by `constant_fold` and must be
# computed at runtime *before* the quantum circuit is sent to the QPU.
# The `separate` pass detects this and places the `BinOp` into a
# `classical_prep` segment.

# %%
# Re-run the pipeline without binding theta
block_param = transpiler.to_block(my_circuit, bindings={"n": 2})
inlined_param = transpiler.inline(block_param)
validated_param = transpiler.linear_validate(inlined_param)
folded_param = transpiler.constant_fold(validated_param, bindings={})  # theta unbound
analyzed_param = transpiler.analyze(folded_param)
separated_param = transpiler.separate(analyzed_param)

# Show all segments
segments_param = []
if separated_param.classical_prep:
    segments_param.append(("classical_prep", separated_param.classical_prep))
segments_param.append(("quantum", separated_param.quantum))
if separated_param.classical_post:
    segments_param.append(("classical_post", separated_param.classical_post))

for name, segment in segments_param:
    print(f"=== {name} ({segment.kind.name}) ===")
    for op in segment.operations:
        print(f"  {op.__class__.__name__}")

# %% [markdown]
# Now `classical_prep` is present and contains the `BinOp` for
# `theta * 2`. This computation will run on a CPU at execution time,
# just before the quantum circuit is submitted to the QPU.
#
# In Section 7, we'll see QPE which produces `classical_post` with
# `DecodeQFixedOperation` for `QFixed → Float` conversion.

# %% [markdown]
# ## 6. The Emit Pass
#
# The final pass is **emit**. It takes the `SeparatedProgram` and converts
# each segment into backend-specific code:
#
# - **Quantum segments** become backend-specific circuits (e.g., a Qiskit
#   `QuantumCircuit`).
# - **Classical segments** become post-processing functions that transform
#   raw measurement results into high-level types.
#
# Let's emit our circuit and execute it.

# %%
executable = transpiler.emit(separated, bindings={"n": 2, "theta": 0.5})

circuit = executable.get_first_circuit()
print("=== Qiskit circuit ===")
print(circuit.draw(output="text"))

# %%
# Execute and see measurement results
executor = transpiler.executor()
job = executable.sample(executor)
result = job.result()

print("\n=== Results ===")
for value, count in result.results:
    print(f"Measured: {value}, Count: {count}")

# %% [markdown]
# The gates become a Qiskit `QuantumCircuit`: X, H, Rz(1.0), CX, followed
# by measurement. Notice that `theta * 2` has already been evaluated by
# `constant_fold`, so the Rz gate uses the pre-computed value `1.0`
# (not the original `0.5`). The results are returned as `Bit` values — no
# post-processing needed.
#
# Note that you can also run the full pipeline in a single call with
# `transpiler.transpile()`:
# ```python
# executable = transpiler.transpile(my_circuit, bindings={"n": 2, "theta": 0.5})
# ```
# The step-by-step approach above is equivalent but lets you inspect
# each intermediate stage.

# %% [markdown]
# ## 7. Full Pipeline — QPE
#
# Now let's see the entire pipeline at work with a more complex example:
# **Quantum Phase Estimation (QPE)**. Unlike `my_circuit`, QPE returns a
# `QFixed` (quantum fixed-point number), which is measured and automatically
# decoded into a `Float`. This decoding happens in the `classical_post`
# segment.
#
# Note that `qpe()` in the standard library is a **regular function**, not
# a `CompositeGate`. It emits operations (including an IQFT
# `CompositeGateOperation`) into the block. The `CompositeGate`s in the
# standard library are QFT and IQFT.

# %%
import math


@qmc.qkernel
def phase_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Phase gate: P(theta)|1> = e^{i*theta}|1>"""
    return qmc.p(q, theta)


@qmc.qkernel
def qpe_3bit(phase: qmc.Float) -> qmc.Float:
    """3-bit Quantum Phase Estimation."""
    phase_register = qmc.qubit_array(3, name="phase_reg")
    target = qmc.qubit(name="target")
    target = qmc.x(target)

    phase_q: qmc.QFixed = qmc.qpe(target, phase_register, phase_gate, theta=phase)

    return qmc.measure(phase_q)


# %%
# Transpile the entire QPE pipeline in one call
test_phase = math.pi / 2
executable_qpe = transpiler.transpile(qpe_3bit, bindings={"phase": test_phase})

# Show the Qiskit circuit
circuit_qpe = executable_qpe.get_first_circuit()
print("=== QPE circuit ===")
print(circuit_qpe.draw(output="text"))

# %%
# Execute and see decoded Float results
job_qpe = executable_qpe.sample(executor)
result_qpe = job_qpe.result()

print("\n=== QPE Results ===")
for value, count in result_qpe.results:
    print(f"Measured value: {value}, Count: {count}")

# %% [markdown]
# The measurement result is returned as a `Float`, not a raw bitstring.
# For `theta = pi/2`, the expected phase is `theta / (2*pi) = 0.25`, and
# the QPE algorithm correctly estimates this value.
#
# ### Inspecting QPE Segments
#
# Let's peek inside to see the segment structure that makes this possible.

# %%
# Run the pipeline step-by-step to inspect segments
block_qpe = transpiler.to_block(qpe_3bit, bindings={"phase": test_phase})
inlined_qpe = transpiler.inline(block_qpe)
validated_qpe = transpiler.linear_validate(inlined_qpe)
folded_qpe = transpiler.constant_fold(validated_qpe, bindings={"phase": test_phase})
analyzed_qpe = transpiler.analyze(folded_qpe)
separated_qpe = transpiler.separate(analyzed_qpe)

# Show the segment structure
segments = []
if separated_qpe.classical_prep:
    segments.append(("classical_prep", separated_qpe.classical_prep))
segments.append(("quantum", separated_qpe.quantum))
if separated_qpe.classical_post:
    segments.append(("classical_post", separated_qpe.classical_post))

for name, segment in segments:
    print(f"=== {name} ({segment.kind.name}) ===")
    for op in segment.operations:
        print(f"  {op.__class__.__name__}")

# %% [markdown]
# QPE produces two segments:
#
# 1. **quantum** — All gates (H, controlled-P, inverse QFT) plus measurement.
#    QPE measures the entire phase register as a single `MeasureVectorOperation`.
#
# 2. **classical_post** — A `DecodeQFixedOperation` that converts the measured
#    bitstring into a `Float` value. This is why QPE can return `Float`
#    instead of raw bitstrings.
#
# Compare this with `my_circuit` from Sections 2–6, which had no
# `classical_post` because `Bit` results need no decoding.

# %% [markdown]
# ## 8. TranspilerConfig and Strategies
#
# The transpiler pipeline is not fixed — you can influence how composite
# gates are decomposed by configuring **strategies**.
#
# `CompositeGate`s in the standard library (like QFT and IQFT) can have
# multiple decomposition strategies registered. For example, the QFT gate
# supports both a standard (full precision, O(n²) gates) and an approximate
# strategy (truncated rotations, O(n·k) gates).
#
# You control which strategy is used during the emit pass via
# `TranspilerConfig`.

# %%
from qamomile.circuit.stdlib.qft import QFT
from qamomile.circuit.stdlib.qft_strategies import (
    ApproximateQFTStrategy,
    StandardQFTStrategy,
)

# Register strategies on the QFT class
QFT.register_strategy("standard", StandardQFTStrategy())
QFT.register_strategy("approximate_k2", ApproximateQFTStrategy(truncation_depth=2))


# Define a kernel that uses QFT and measures the result
@qmc.qkernel
def qft_and_measure() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(4, name="q")
    for i in range(4):
        q[i] = qmc.h(q[i])
    q = qmc.qft(q)
    return qmc.measure(q)


# %%
# Transpile with the default (standard) strategy.
# use_native_composite=False forces Qamomile's own decomposition
# so that strategy selection takes effect.
transpiler_standard = QiskitTranspiler(use_native_composite=False)
circuit_standard = transpiler_standard.to_circuit(qft_and_measure)

print("=== Standard QFT (full precision) ===")
print(circuit_standard.draw(output="text"))

# %%
# Transpile with the approximate strategy (truncation_depth=2).
# For 4 qubits, this skips the CP(pi/8) rotation whose exponent (3)
# exceeds the truncation depth (2).
from qamomile.circuit.transpiler.transpiler import TranspilerConfig

config = TranspilerConfig.with_strategies({"qft": "approximate_k2"})

transpiler_approx = QiskitTranspiler(use_native_composite=False)
transpiler_approx.set_config(config)

circuit_approx = transpiler_approx.to_circuit(qft_and_measure)

print("=== Approximate QFT (truncation_depth=2) ===")
print(circuit_approx.draw(output="text"))

# %%
# Compare gate counts
print("=== Gate Count Comparison ===")
print(f"Standard QFT gates:    {circuit_standard.size()}")
print(f"Approximate QFT gates: {circuit_approx.size()}")

# %% [markdown]
# The approximate strategy produces fewer gates by truncating small-angle
# controlled-phase rotations. For 4 qubits with `truncation_depth=2`, the
# CP(π/8) rotation is skipped because its exponent (3) exceeds the depth.
# This trades precision for efficiency — useful when exact QFT precision
# is not required.
#
# `TranspilerConfig.with_strategies()` creates a configuration that maps
# gate names to strategy names. The transpiler uses this mapping during the
# emit pass to select the appropriate decomposition for each composite gate.

# %% [markdown]
# ## 9. Summary
#
# In this tutorial, you traced the complete path from a `@qkernel` function
# to an `ExecutableProgram`. Here is a recap of each pass:
#
# | Pass | Input | Output | Purpose |
# |------|-------|--------|---------|
# | `to_block()` | `QKernel` | Block (HIERARCHICAL) | Convert Python function to IR |
# | `inline()` | Block | Block (LINEAR) | Expand all kernel calls |
# | `linear_validate()` | Block | Block (validated) | Enforce no-cloning principle |
# | `constant_fold()` | Block | Block (folded) | Evaluate known constants |
# | `analyze()` | Block | Block (ANALYZED) | Build dependency graph, validate I/O |
# | `separate()` | Block | SeparatedProgram | Split into classical/quantum segments |
# | `emit()` | SeparatedProgram | ExecutableProgram | Generate backend-specific code |
#
# Key takeaways:
#
# - **`transpile()` is not magic** — it is a well-defined sequence of passes,
#   each of which you can call individually for debugging.
# - **`to_block()` vs `kernel.block`** — Use `to_block(kernel, bindings=...)`
#   when your kernel has parameterized array shapes. `kernel.block` is a
#   shortcut that builds without bindings.
# - **`draw()` with `inline` and `inline_depth`** — Visualize circuits at
#   different levels of detail before entering the transpiler pipeline.
# - **Segment separation** splits the program into `classical_prep`
#   (runtime parameter computation, as seen when `theta` is unbound),
#   `quantum` (gates and measurements), and `classical_post` (e.g.,
#   QFixed → Float decoding as seen in QPE).
# - **`TranspilerConfig`** controls which decomposition strategy is used for
#   composite gates like QFT and IQFT during the emit pass.
# - **QPE** uses `qpe()`, a regular function (not a CompositeGate). The
#   CompositeGates in the standard library are QFT and IQFT, which support
#   pluggable strategies.
#
# ### Next Steps
#
# - [Custom Executor](10_custom_executor.ipynb): Run circuits on cloud quantum hardware
# - [Resource Estimation](08_resource_estimation.ipynb): Analyze gate counts and circuit depth
# - [QAOA](../optimization/qaoa.ipynb): Optimization with variational circuits

# %% [markdown]
# ## What We Learned
#
# - **The full transpiler pipeline and the role of each pass** — Seven passes (`to_block` → `inline` → `validate` → `constant_fold` → `analyze` → `separate` → `emit`) transform a `@qkernel` into backend-specific code.
# - **The difference between `kernel.block` and `transpiler.to_block()`** — `kernel.block` builds without bindings; `to_block(kernel, bindings=...)` is needed when array shapes depend on parameters.
# - **How `draw()` visualizes circuits** — `inline=True` expands sub-kernel calls; `inline_depth` controls how many nesting levels are expanded.
# - **How kernel calls are inlined into a flat instruction sequence** — The `inline()` pass recursively expands all `CallBlockOperation`s, producing a single linear block.
# - **How validation, constant folding, and dependency analysis work** — `linear_validate` enforces no-cloning, `constant_fold` evaluates `BinOp` expressions like `theta * 2` into concrete values, and `analyze` builds a dependency graph for I/O validation.
# - **How the program is separated into quantum and classical segments** — `separate()` splits the block into `classical_prep` (runtime parameter computation when values are unbound), `quantum`, and `classical_post` segments. QPE demonstrates `classical_post` with `DecodeQFixedOperation` for QFixed → Float conversion.
# - **How the emit pass generates backend-specific code** — `emit()` walks quantum segments and produces native circuit objects (e.g. Qiskit `QuantumCircuit`) for the target backend.
# - **How `TranspilerConfig` controls composite gate decomposition strategies** — `TranspilerConfig.with_strategies()` maps gate names to strategy names, selecting decompositions during the emit pass.
