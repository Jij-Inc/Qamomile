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
# # Transpile & Execute
#
# This tutorial explains the transpilation flow in Qamomile.
#
# ## Basic Usage
#
# We will use Quantum Phase Estimation (QPE) as an example to explain the transpilation and execution flow in Qamomile.
# Let's start by confirming the basic usage.

# %%
import qamomile.circuit as qmc


# %% [markdown]
# ### QPE Overview
#
# Quantum Phase Estimation is an algorithm that estimates the phase φ of the eigenvalue e^{2πiφ} of a unitary operator U.
#
# In Qamomile, you can easily implement QPE using the `qpe()` function:
# - Input: target state, phase register, unitary operation
# - Output: `QFixed` (quantum fixed-point number)
#
# When you measure `QFixed` with `measure()`, it is automatically decoded to `Float`.

# %%
import math

# Define a phase gate as the unitary
# P(θ)|1⟩ = e^{iθ}|1⟩, so |1⟩ is an eigenstate with eigenvalue e^{iθ}
@qmc.qkernel
def phase_gate(q: qmc.Qubit, theta: float) -> qmc.Qubit:
    """Phase gate: P(θ)|1⟩ = e^{iθ}|1⟩"""
    return qmc.p(q, theta)


# 3-bit precision QPE
@qmc.qkernel
def qpe_3bit(phase: float) -> qmc.Float:
    """3-bit Quantum Phase Estimation.

    Args:
        phase: The phase angle θ (the algorithm estimates θ/(2π))

    Returns:
        Float: Estimated phase as a fraction (0 to 1)
    """
    # Create phase register (3-bit precision)
    phase_register = qmc.qubit_array(3, name="phase_reg")

    # Initialize target state to |1⟩ (eigenstate of P(θ))
    target = qmc.qubit(name="target")
    target = qmc.x(target)  # |0⟩ → |1⟩

    # Apply QPE
    phase_q: qmc.QFixed = qmc.qpe(target, phase_register, phase_gate, theta=phase)

    # Measure QFixed and convert to Float
    return qmc.measure(phase_q)

# %%
# Transpile and Execute
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

test_phase = math.pi / 2  # θ = π/2, expected output ≈ 0.25 (since θ/(2π) = 0.25)
executable = transpiler.transpile(qpe_3bit, bindings={"phase": test_phase})

executor = transpiler.executor()
job = executable.sample(executor)
result = job.result()

for value, count in result.results:
    print(f"Measured value: {value}, Count: {count}")

# %% [markdown]
# As you can see, the measurement result is returned as a Float, not a bitstring.
#
# ## Inline Pass
# Let's look at each step of the transpilation process in detail.
# First is the `Inline` pass. This expands all `CallBlockOperation`s inline.
# Since there's nothing to inline in the QPE example above, let's look at a different example.

# %%
@qmc.qkernel
def add_one(q: qmc.Qubit) -> qmc.Qubit:
    """Add one to a qubit (|0⟩ → |1⟩, |1⟩ → |0⟩)"""
    return qmc.x(q)

@qmc.qkernel
def add_two(q: qmc.Qubit) -> qmc.Qubit:
    """Add two to a qubit by calling add_one twice"""
    q = add_one(q)
    q = add_one(q)
    return q

@qmc.qkernel
def add_three(q: qmc.Qubit) -> qmc.Qubit:
    """Add three to a qubit by calling add_two and add_one"""
    q = add_two(q)
    q = add_one(q)
    return q

# %# [markdown]
# Let's try inlining these kernels.

# %%
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.operation.gate import GateOperation

transpiler = QiskitTranspiler()

def print_block_operations(block: Block):
    for op in block.operations:
        print(op.__class__.__name__ + ":", end="")
        if isinstance(op, CallBlockOperation):
            print(op.operands[0].name)
        elif isinstance(op, GateOperation):
            print(op.gate_type)
        else:
            print("")

# Before inlining
block = transpiler.to_block(add_three)
print_block_operations(block)

# %% [markdown]
# As you can see, there are just two `CallBlockOperation`s.
# Now let's perform the inline expansion.

# %%
inlined_block = transpiler.inline(block)
print_block_operations(inlined_block)

# %% [markdown]
# You can see that `add_three` has been expanded into the contents of `add_two` and `add_one`, which is three X gates.

# %% [markdown]
# ## Analyze Pass and Separate Pass
# Next is the `Analyze` pass. This performs dependency analysis and validation. It doesn't make any changes to the computation path.
# After that comes the `Separate` pass. This separates the program into quantum segments and classical segments.
# Let's look at these passes using the QPE example.

# %%
block = transpiler.to_block(qpe_3bit)
inlined_block = transpiler.inline(block)
analyzed_block = transpiler.analyze(inlined_block)
separated_program = transpiler.separate(analyzed_block)


# %%
for i, segment in enumerate(separated_program.segments):
    print(f"Segment {i}: {segment.kind.name}")
    for op in segment.operations:
        print(" ", op.__class__.__name__)

# %% [markdown]
# Quantum operations (gates, measurements, etc.) are separated into `QUANTUM` segments, and classical operations (decoding, etc.) are separated into `CLASSICAL` segments.
#
# `boundaries` tracks the boundaries between quantum and classical (mainly measurements):

# %%
print(f"Boundaries: {len(separated_program.boundaries)}")
for boundary in separated_program.boundaries:
    print(f"  {boundary.operation.__class__.__name__}: segment {boundary.source_segment_index} → {boundary.target_segment_index}")

# %% [markdown]
# ## Emit Pass
#
# Finally, the `Emit` pass. This pass converts the separated program into backend-specific code.
#
# ### Emitting Quantum Segments
# Operations in QUANTUM segments are emitted to backend-specific quantum circuits.
# For Qiskit, a `QuantumCircuit` object is generated.
#
# ### Classical Segment Post-processing
# CLASSICAL segments are added as post-processing for measurement results.
# For example, when measuring `QFixed` in QPE:
# 1. QUANTUM segment: Measure each qubit → raw bitstring
# 2. CLASSICAL segment: Decode bitstring to Float
#
# This allows users to receive `Float` directly rather than raw bitstrings.

# %%
executable = transpiler.emit(separated_program, bindings={"phase": test_phase})

# Check the quantum circuit
print("=== Quantum Circuit ===")
circuit = executable.get_first_circuit()
print(circuit.draw(output="text"))

# Check classical processing
print("\n=== Classical Post-processing ===")
print(f"Total segments: {len(separated_program.segments)}")
for i, segment in enumerate(separated_program.segments):
    print(f"Segment {i}: {segment.kind.name}")
    if segment.kind.name == "CLASSICAL":
        for op in segment.operations:
            print(f"  {op.__class__.__name__}")

# %% [markdown]
# As shown above, in QPE execution:
#
# 1. **QUANTUM segment** → Emitted to Qiskit `QuantumCircuit`
#    - H gates, controlled phase gates, inverse QFT
#    - Three `MeasureOperation`s (measurement of each qubit)
#
# 2. **CLASSICAL segment** → Classical processing after measurement
#    - `DecodeQFixedOperation` decodes bitstring to Float
#
# This is how we get `Float` values like `Measured value: 0.25` as shown in the first example.
# Users can receive results in high-level types (`QFixed` → `Float`) without being aware of raw bitstrings.
