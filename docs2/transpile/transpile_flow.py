# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# # Transpile Flow
#
# In this section, we describe the flow of transpiling Qamomile qmc.qkernels into backend-specific quantum circuits. The transpilation process consists of several passes that transform the intermediate representation (IR) step by step.
#
# ## Inline Pass
#
# The first step in the transpilation process is the **Inline Pass**. This pass is responsible for inlining all `CallBlockOperations` within the qmc.qkernel.
#
# ### What is CallBlockOperation?
#
# When you call one `@qmc.qkernel` function from another, Qamomile creates a `CallBlockOperation` in the IR. This represents a function call that needs to be expanded (inlined) before the circuit can be executed.

# %%
import qamomile.circuit as qmc

# Sub-kernel: Creates a Bell pair
@qmc.qkernel
def bell_pair(q0: qmc.Qubit, q1: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
    q0 = qmc.h(q0)
    q0, q1 = qmc.cx(q0, q1)
    return q0, q1

# Main kernel: Calls bell_pair
@qmc.qkernel
def main(q0: qmc.Qubit, q1: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
    q0, q1 = bell_pair(q0, q1)  # <- This creates a CallBlockOperation
    return q0, q1


# %% [markdown]
# ### Before and After Inlining
#
# Let's see how the IR changes before and after the Inline Pass:

# %%
from qamomile.circuit.ir.block import Block
from qamomile.circuit.transpiler.passes.inline import InlinePass
from qamomile.circuit.ir.operation.gate import GateOperation

# Convert qmc.qkernel to Block
block_value = main.block
block = Block.from_block_value(block_value)

print("=== Before Inlining ===")
print(f"Block kind: {block.kind}")
print(f"Operations ({len(block.operations)}):")
for i, op in enumerate(block.operations):
    print(f"  [{i}] {type(op).__name__}")

# %%
# Apply InlinePass
inline_pass = InlinePass()
linear_block = inline_pass.run(block)

print("=== After Inlining ===")
print(f"Block kind: {linear_block.kind}")
print(f"Operations ({len(linear_block.operations)}):")
for i, op in enumerate(linear_block.operations):
    if isinstance(op, GateOperation):
        print(f"  [{i}] {op.gate_type.name}")
    else:
        print(f"  [{i}] {type(op).__name__}")


# %% [markdown]
# ### How Inlining Works
#
# The `InlinePass` transforms the block structure as follows:
#
# ```
# HIERARCHICAL Block                    LINEAR Block
# ┌─────────────────────┐              ┌─────────────────────┐
# │ CallBlockOperation  │     =>       │ H gate              │
# │   └─> bell_pair     │              │ CX gate             │
# └─────────────────────┘              └─────────────────────┘
# ```
#
# The key steps in `InlinePass._inline_call()` are:
#
# 1. **Extract the called block**: Get the `BlockValue` and arguments from `CallBlockOperation`
# 2. **Map input values**: Create a mapping from block's input values to the call's arguments
# 3. **Clone operations with fresh UUIDs**: Avoid UUID collisions when the same function is called multiple times
# 4. **Recursively serialize**: Handle nested function calls
# 5. **Map return values**: Connect the block's return values to the call's results
#
# ### Handling Multiple Calls
#
# When the same kernel is called multiple times, each call gets fresh UUIDs to avoid conflicts:

# %%
@qmc.qkernel
def apply_twice(q0: qmc.Qubit, q1: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
    q0, q1 = bell_pair(q0, q1)  # First call
    q0, q1 = bell_pair(q0, q1)  # Second call
    return q0, q1


inline = InlinePass()
block_value = apply_twice.block
block = Block.from_block_value(block_value)
linear_block = inline.run(block)
print("=== After Inlining apply_twice ===")
print(f"Block kind: {linear_block.kind}")
print(f"Operations ({len(linear_block.operations)}):")
for i, op in enumerate(linear_block.operations):
    if isinstance(op, GateOperation):
        print(f"  [{i}] {op.gate_type.name}")
    else:
        print(f"  [{i}] {type(op).__name__}")

# After inlining:
# [0] H    (from first call)
# [1] CX   (from first call)
# [2] H    (from second call)  <- fresh UUIDs
# [3] CX   (from second call)  <- fresh UUIDs


# %% [markdown]
# ### Control Flow Preservation
#
# The Inline Pass preserves control flow structures (`For`, `If`, `While`) while still inlining any `CallBlockOperations` inside them:

# %%
from qamomile.circuit.ir.operation.control_flow import ForOperation

@qmc.qkernel
def repeated_bell(q0: qmc.Qubit, q1: qmc.Qubit, n: int) -> tuple[qmc.Qubit, qmc.Qubit]:
    for i in range(n):
        q0, q1 = bell_pair(q0, q1)  # Inlined inside the loop
    return q0, q1

inline = InlinePass()
block_value = repeated_bell.block
block = Block.from_block_value(block_value)
linear_block = inline.run(block)
print("=== After Inlining repeated_bell ===")
print(f"Block kind: {linear_block.kind}")
print(f"Operations ({len(linear_block.operations)}):")
for i, op in enumerate(linear_block.operations):
    if isinstance(op, ForOperation):
        print(f"  [{i}] ForOperation (loop_var={op.loop_var})")
        print("    Loop body operations:")
        for j, body_op in enumerate(op.operations):
            if isinstance(body_op, GateOperation):
                print(f"      [{j}] {body_op.gate_type.name}")
            else:
                print(f"      [{j}] {type(body_op).__name__}")
    elif isinstance(op, GateOperation):
        print(f"  [{i}] {op.gate_type.name}")
    else:
        print(f"  [{i}] {type(op).__name__}")


# %% [markdown]
# After inlining, the `ForOperation` structure is preserved, but the `CallBlockOperation` inside is replaced with the actual gate operations.


# %% [markdown]
# ## Analyze Pass
#
# Before separating operations into segments, we need to analyze the block to validate dependencies. The **Analyze Pass** performs the following:
#
# 1. Builds a dependency graph between values
# 2. Validates that quantum operations don't depend on measurement results
# 3. Ensures block inputs/outputs are classical types
#

# %%
from qamomile.circuit.transpiler.passes.analyze import AnalyzePass

# First, let's use the simple main kernel from earlier
block = Block.from_block_value(main.block)
linear_block = InlinePass().run(block)

# Apply AnalyzePass
analyze_pass = AnalyzePass()
analyzed_block = analyze_pass.run(linear_block)

print("=== After AnalyzePass ===")
print(f"Block kind: {analyzed_block.kind}")
print(f"Operations: {len(analyzed_block.operations)}")


# %% [markdown]
# ## Separate Pass
#
# The **Separate Pass** splits operations into quantum and classical segments. This is essential for:
#
# - **Quantum segments**: Pure quantum gates that will be compiled to backend circuits
# - **Classical segments**: Classical computations (arithmetic, conditions)
# - **Hybrid boundaries**: Measurement operations that bridge quantum and classical
#
# ```
# SeparatedProgram
# ├─ segments: [QuantumSegment, ClassicalSegment, ...]
# ├─ boundaries: [HybridBoundary, ...]  (measurements)
# └─ output_refs: [uuid, ...]
# ```

# %%
from qamomile.circuit.transpiler.passes.separate import SeparatePass
from qamomile.circuit.transpiler.segments import QuantumSegment, ClassicalSegment

separate_pass = SeparatePass()
separated = separate_pass.run(analyzed_block)

print("=== After SeparatePass ===")
print(f"Number of segments: {len(separated.segments)}")
print(f"Number of boundaries: {len(separated.boundaries)}")
print()
for i, seg in enumerate(separated.segments):
    print(f"Segment [{i}]: {seg.kind.name}")
    for j, op in enumerate(seg.operations):
        if isinstance(op, GateOperation):
            print(f"  [{j}] {op.gate_type.name}")
        else:
            print(f"  [{j}] {type(op).__name__}")


# %% [markdown]
# ### Quantum-Only Kernel
#
# When a kernel contains only quantum operations, the `SeparatePass` produces a single `QuantumSegment`:

# %%
print("=== Pure Quantum Kernel ===")
print(f"Quantum segments: {len(separated.quantum_segments())}")
print(f"Classical segments: {len(separated.classical_segments())}")


# %% [markdown]
# ### Mixed Quantum and Classical Operations
#
# Let's create a kernel with measurement to see how boundaries work:

# %%
@qmc.qkernel
def measure_kernel(q0: qmc.Qubit) -> qmc.Bit:
    q0 = qmc.h(q0)
    result = qmc.measure(q0)
    return result

# Process through the pipeline
block = Block.from_block_value(measure_kernel.block)
linear_block = InlinePass().run(block)
analyzed_block = AnalyzePass().run(linear_block)
separated = SeparatePass().run(analyzed_block)

print("=== Kernel with Measurement ===")
print(f"Number of segments: {len(separated.segments)}")
print(f"Number of boundaries: {len(separated.boundaries)}")
print()

for i, seg in enumerate(separated.segments):
    print(f"Segment [{i}]: {seg.kind.name}")
    for j, op in enumerate(seg.operations):
        if isinstance(op, GateOperation):
            print(f"  [{j}] {op.gate_type.name}")
        else:
            print(f"  [{j}] {type(op).__name__}")

print()
for i, boundary in enumerate(separated.boundaries):
    print(f"Boundary [{i}]: {type(boundary.operation).__name__}")
    print(f"  source_segment: {boundary.source_segment_index}")
    print(f"  target_segment: {boundary.target_segment_index}")


# %% [markdown]
# ### How Separation Works
#
# The `SeparatePass` processes operations as follows:
#
# ```
# Input Block Operations          SeparatedProgram
# ┌─────────────────────┐         ┌─────────────────────┐
# │ H gate (quantum)    │    =>   │ QuantumSegment      │
# │ CX gate (quantum)   │         │   └─ H, CX          │
# │ Measure (hybrid)    │         │ HybridBoundary      │
# │ Add (classical)     │         │   └─ Measure        │
# └─────────────────────┘         │ ClassicalSegment    │
#                                 │   └─ Add            │
#                                 └─────────────────────┘
# ```
#
# Key points:
# - **Quantum operations** are grouped into `QuantumSegment`
# - **Measurement** creates a `HybridBoundary` (not in any segment)
# - **Classical operations** are grouped into `ClassicalSegment`
# - Segment boundaries are created when the operation kind changes
