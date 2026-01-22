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
# # Single-Qubit Gates and Superposition
#
# In this tutorial, you will learn about **superposition**, the core concept of quantum computing,
# and the various quantum gates that create it.
#
# ## What You Will Learn
# - The Hadamard gate (H gate) and superposition states
# - Interpreting probabilistic measurement results
# - Rotation gates (RX, RY, RZ) and parameters
# - The phase gate (P gate)
# - Parameterized quantum circuits

# %%
import math
import qamomile.circuit as qm
from qamomile.qiskit import QiskitTranspiler

# Prepare the transpiler
transpiler = QiskitTranspiler()

# %% [markdown]
# ## 1. The Hadamard Gate (H Gate)
#
# The **Hadamard gate** is one of the most important gates in quantum computing.
# This gate puts a qubit into a **superposition state**.
#
# ### Mathematical Definition
#
# The Hadamard gate performs the following transformations:
#
# $$H|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$$
#
# $$H|1\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}$$
#
# In other words, applying the H gate to `|0⟩` creates a state where `|0⟩` and `|1⟩` have equal probability.

# %%
@qm.qkernel
def hadamard_circuit() -> qm.Bit:
    """Apply Hadamard gate to create a superposition state"""
    q = qm.qubit(name="q")

    # Create superposition with H gate
    q = qm.h(q)

    return qm.measure(q)


# %%
# Let's run it
executable = transpiler.transpile(hadamard_circuit)
job = executable.sample(transpiler.executor(), shots=1000)
result = job.result()

print("=== Hadamard Gate Results ===")
for value, count in result.results:
    percentage = count / 1000 * 100
    print(f"  Measurement result: {value}, Count: {count} ({percentage:.1f}%)")

# %% [markdown]
# ### Interpreting the Results
#
# Looking at the results, you can see that `0` and `1` each appear about 50% of the time.
# This is the essence of **superposition**.
#
# - Until measured, the qubit "simultaneously" holds both 0 and 1 states
# - At the moment of measurement, it "collapses" to one or the other
# - Which one it collapses to is determined probabilistically
#
# It's like a quantum coin flip!

# %% [markdown]
# ### Circuit Visualization

# %%
qiskit_circuit = transpiler.to_circuit(hadamard_circuit)
print("=== Hadamard Gate Circuit ===")
print(qiskit_circuit.draw(output="text"))

# %% [markdown]
# ## 2. Combining Multiple Gates
#
# By combining quantum gates, you can create various states.
# What happens if we apply the H gate twice?

# %%
@qm.qkernel
def double_hadamard() -> qm.Bit:
    """Apply H gate twice"""
    q = qm.qubit(name="q")

    q = qm.h(q)  # First: |0⟩ → (|0⟩+|1⟩)/√2
    q = qm.h(q)  # Second: (|0⟩+|1⟩)/√2 → |0⟩

    return qm.measure(q)


# %%
executable2 = transpiler.transpile(double_hadamard)
job2 = executable2.sample(transpiler.executor(), shots=1000)
result2 = job2.result()

print("=== Double H Gate Results ===")
for value, count in result2.results:
    print(f"  Measurement result: {value}, Count: {count}")

# %% [markdown]
# ### Why Only 0?
#
# Applying the H gate twice returns to the original state!
#
# $$H \cdot H = I$$
#
# This is because the H gate is its own "inverse" (self-inverse).
# Mathematically:
#
# $$H \cdot H |0\rangle = H \cdot \frac{|0\rangle + |1\rangle}{\sqrt{2}} = |0\rangle$$

# %% [markdown]
# ## 3. The |+⟩ and |−⟩ States
#
# Superposition states have names:
#
# - **|+⟩ state**: $\frac{|0\rangle + |1\rangle}{\sqrt{2}}$ (H|0⟩)
# - **|−⟩ state**: $\frac{|0\rangle - |1\rangle}{\sqrt{2}}$ (H|1⟩)
#
# These have the same measurement probability distribution (50/50), but they are different quantum states.

# %%
@qm.qkernel
def plus_state() -> qm.Bit:
    """Create |+⟩ state"""
    q = qm.qubit(name="q")
    q = qm.h(q)  # |0⟩ → |+⟩
    return qm.measure(q)


@qm.qkernel
def minus_state() -> qm.Bit:
    """Create |−⟩ state"""
    q = qm.qubit(name="q")
    q = qm.x(q)  # |0⟩ → |1⟩
    q = qm.h(q)  # |1⟩ → |−⟩
    return qm.measure(q)


# %%
# Execute both and compare
exec_plus = transpiler.transpile(plus_state)
exec_minus = transpiler.transpile(minus_state)

result_plus = exec_plus.sample(transpiler.executor(), shots=1000).result()
result_minus = exec_minus.sample(transpiler.executor(), shots=1000).result()

print("=== |+⟩ State Measurement Results ===")
for value, count in result_plus.results:
    print(f"  Measurement result: {value}, Count: {count}")

print("\n=== |−⟩ State Measurement Results ===")
for value, count in result_minus.results:
    print(f"  Measurement result: {value}, Count: {count}")

print("\nBoth are approximately 50/50, but they are different quantum states!")

# %% [markdown]
# ## 4. Rotation Gates (RX, RY, RZ)
#
# For finer control, we use **rotation gates**.
# These take a rotation angle as a parameter.
#
# ### Types of Rotation Gates
#
# - **RX(θ)**: Rotate θ radians around the X-axis
# - **RY(θ)**: Rotate θ radians around the Y-axis
# - **RZ(θ)**: Rotate θ radians around the Z-axis
#
# ### Bloch Sphere Image
#
# A qubit state can be represented as a point on a sphere called the "Bloch sphere."
# Rotation gates rotate the state on this sphere.

# %%
@qm.qkernel
def rx_circuit(theta: qm.Float) -> qm.Bit:
    """Apply RX rotation gate"""
    q = qm.qubit(name="q")
    q = qm.rx(q, theta)  # Rotate theta around X-axis
    return qm.measure(q)


# %%
# Let's run with different angles
angles = [0, math.pi / 4, math.pi / 2, math.pi]
angle_names = ["0", "π/4", "π/2", "π"]

print("=== RX Gate Behavior at Different Angles ===\n")

for angle, name in zip(angles, angle_names):
    executable = transpiler.transpile(rx_circuit, bindings={"theta": angle})
    result = executable.sample(transpiler.executor(), shots=1000).result()

    print(f"RX({name}):")
    for value, count in result.results:
        percentage = count / 1000 * 100
        print(f"  {value}: {count} ({percentage:.1f}%)")
    print()

# %% [markdown]
# ### Interpreting the Results
#
# - **RX(0)**: No rotation → always 0
# - **RX(π/4)**: Small rotation → 1 starts appearing
# - **RX(π/2)**: 45-degree rotation → approximately 50/50 (similar effect to H gate)
# - **RX(π)**: 180-degree rotation → completely flipped, always 1 (same effect as X gate)

# %% [markdown]
# ## 5. Parameterized Circuits
#
# In Qamomile, you can treat parameters as variables and specify values at execution time.
# This is an important feature for Variational Quantum Algorithms (VQA).

# %%
@qm.qkernel
def parameterized_circuit(theta: qm.Float, phi: qm.Float) -> qm.Bit:
    """Circuit with multiple parameters"""
    q = qm.qubit(name="q")

    q = qm.ry(q, theta)  # Y-axis rotation
    q = qm.rz(q, phi)    # Z-axis rotation

    return qm.measure(q)


# %%
# Compile with parameters specified
# parameters: list of parameter names that can be changed at runtime
executable_param = transpiler.transpile(
    parameterized_circuit,
    parameters=["theta", "phi"]  # These values will be specified later
)

# Execute with different parameters
params_list = [
    {"theta": 0, "phi": 0},
    {"theta": math.pi / 2, "phi": 0},
    {"theta": math.pi / 2, "phi": math.pi},
]

print("=== Parameterized Circuit Execution ===\n")

for params in params_list:
    result = executable_param.sample(
        transpiler.executor(),
        bindings=params,
        shots=1000
    ).result()

    print(f"theta={params['theta']:.2f}, phi={params['phi']:.2f}:")
    for value, count in result.results:
        print(f"  {value}: {count}")
    print()

# %% [markdown]
# ## 6. The Phase Gate (P Gate)
#
# The **phase gate** P(θ) adds a phase $e^{i\theta}$ to the `|1⟩` state.
#
# $$P(\theta)|0\rangle = |0\rangle$$
# $$P(\theta)|1\rangle = e^{i\theta}|1\rangle$$
#
# Phase doesn't directly affect measurement results, but it plays an important role in quantum interference.

# %%
@qm.qkernel
def phase_example() -> qm.Bit:
    """Phase gate example"""
    q = qm.qubit(name="q")

    q = qm.h(q)           # Create superposition
    q = qm.p(q, math.pi)  # Add phase π to |1⟩ (sign flip)
    q = qm.h(q)           # Interfere

    return qm.measure(q)


# %%
exec_phase = transpiler.transpile(phase_example)
result_phase = exec_phase.sample(transpiler.executor(), shots=1000).result()

print("=== Phase Gate Example (H-P(π)-H = X) ===")
for value, count in result_phase.results:
    print(f"  Measurement result: {value}, Count: {count}")

# %% [markdown]
# ### Result Explanation
#
# The combination H → P(π) → H has the same effect as the X gate!
#
# This is an example of quantum interference:
# 1. H gate creates superposition
# 2. P(π) flips the sign of the |1⟩ component
# 3. The second H gate causes interference, leaving only |1⟩

# %% [markdown]
# ## 7. Circuit Visualization: Rotation Gates

# %%
qiskit_param = transpiler.to_circuit(
    parameterized_circuit,
    bindings={"theta": math.pi / 4, "phi": math.pi / 2}
)
print("=== Parameterized Circuit Structure ===")
print(qiskit_param.draw(output="text"))

# %% [markdown]
# ## 8. Summary
#
# In this tutorial, you learned:
#
# ### Quantum Gates
# | Gate | Qamomile | Effect |
# |------|----------|--------|
# | Hadamard | `qm.h(q)` | Creates superposition |
# | X rotation | `qm.rx(q, θ)` | Rotates around X-axis |
# | Y rotation | `qm.ry(q, θ)` | Rotates around Y-axis |
# | Z rotation | `qm.rz(q, θ)` | Rotates around Z-axis |
# | Phase | `qm.p(q, θ)` | Adds phase to |1⟩ |
#
# ### Key Concepts
# - **Superposition**: A state holding both 0 and 1 simultaneously. Collapses probabilistically upon measurement
# - **Quantum interference**: Gate combinations cause probability amplitudes to reinforce or cancel
# - **Parameterization**: Values can be changed at runtime using the `parameters` argument
#
# ### Parameterized Circuit Pattern
#
# ```python
# # 1. Define circuit with parameters
# @qm.qkernel
# def circuit(theta: qm.Float) -> qm.Bit:
#     q = qm.qubit(name="q")
#     q = qm.rx(q, theta)
#     return qm.measure(q)
#
# # 2. Compile with parameter names specified
# executable = transpiler.transpile(circuit, parameters=["theta"])
#
# # 3. Specify values at runtime
# result = executable.sample(executor, bindings={"theta": 0.5}, shots=1000)
# ```
#
# In the next tutorial (`03_entanglement.py`), you will work with multiple qubits
# and learn about **entanglement**, the most mysterious phenomenon in quantum mechanics.
