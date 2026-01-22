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
# # Multiple Qubits and Entanglement
#
# In this tutorial, you will learn about **entanglement**,
# the most mysterious and powerful concept in quantum computing.
#
# ## What You Will Learn
# - How to work with multiple qubits
# - The CNOT gate (Controlled-NOT gate)
# - Bell states: The most basic entangled states
# - Working with arrays using `qubit_array()` and `Vector[Qubit]`
# - Using loops with `qm.range()`
# - GHZ states: Multi-qubit entanglement

# %%
import math
import qamomile.circuit as qm
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## 1. Using Multiple Qubits
#
# So far, we have only worked with a single qubit.
# To use multiple qubits, simply call `qm.qubit()` multiple times.

# %%
@qm.qkernel
def two_qubits_independent() -> tuple[qm.Bit, qm.Bit]:
    """Two independent qubits"""
    q0 = qm.qubit(name="q0")
    q1 = qm.qubit(name="q1")

    # Independent operations on each
    q0 = qm.h(q0)  # Put q0 in superposition
    q1 = qm.x(q1)  # Flip q1

    # Measure separately and return
    return qm.measure(q0), qm.measure(q1)


# %%
exec_two = transpiler.transpile(two_qubits_independent)
result_two = exec_two.sample(transpiler.executor(), shots=1000).result()

print("=== Two Independent Qubits ===")
for value, count in result_two.results:
    print(f"  Result: {value}, Count: {count}")

# %% [markdown]
# ### Interpreting the Results
#
# The result is a tuple `(bit0, bit1)`.
# - `q0` is in superposition due to H gate → 0 or 1 with about 50% each
# - `q1` is flipped by X gate → always 1
#
# Therefore, `(0, 1)` and `(1, 1)` should each appear about 50% of the time.

# %% [markdown]
# ## 2. The CNOT Gate (Controlled-NOT Gate)
#
# The **CNOT gate** (also called Controlled-NOT or CX gate) is the most fundamental two-qubit gate.
#
# ### Operation
# - **Control bit**: When this qubit is `|1⟩`, it acts on the target
# - **Target bit**: Flips if the control bit is `|1⟩`
#
# ```
# |00⟩ → |00⟩  (control is 0, so target unchanged)
# |01⟩ → |01⟩  (control is 0, so target unchanged)
# |10⟩ → |11⟩  (control is 1, so target flips)
# |11⟩ → |10⟩  (control is 1, so target flips)
# ```

# %%
@qm.qkernel
def cnot_example() -> tuple[qm.Bit, qm.Bit]:
    """Basic usage of CNOT gate"""
    q0 = qm.qubit(name="control")
    q1 = qm.qubit(name="target")

    # Set control bit to |1⟩
    q0 = qm.x(q0)

    # Apply CNOT gate
    # Important: Receive both qubits as return values!
    q0, q1 = qm.cx(q0, q1)

    return qm.measure(q0), qm.measure(q1)


# %%
exec_cnot = transpiler.transpile(cnot_example)
result_cnot = exec_cnot.sample(transpiler.executor(), shots=1000).result()

print("=== CNOT Gate Example (control bit = 1) ===")
for value, count in result_cnot.results:
    print(f"  Result: {value}, Count: {count}")

print("\nSince the control bit is 1, the target also flips to (1, 1)")

# %% [markdown]
# ### Important: Return Values of Two-Qubit Gates
#
# In Qamomile's linear type system, two-qubit gates **return both qubits**.
#
# ```python
# # Correct way
# q0, q1 = qm.cx(q0, q1)  # Receive both
#
# # Wrong ways
# qm.cx(q0, q1)           # Ignoring return value causes error
# q0 = qm.cx(q0, q1)      # Receiving only one makes q1 unusable
# ```

# %% [markdown]
# ## 3. Bell States: The Basics of Entanglement
#
# **Entanglement** is a phenomenon where multiple qubits have strong correlations,
# and measuring one instantly determines the state of the others.
#
# The most basic entangled state is the **Bell state**.

# %%
@qm.qkernel
def bell_state() -> tuple[qm.Bit, qm.Bit]:
    """Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2"""
    q0 = qm.qubit(name="q0")
    q1 = qm.qubit(name="q1")

    # Step 1: Put the first qubit in superposition
    q0 = qm.h(q0)

    # Step 2: Create "entanglement" with CNOT
    q0, q1 = qm.cx(q0, q1)

    return qm.measure(q0), qm.measure(q1)


# %%
exec_bell = transpiler.transpile(bell_state)
result_bell = exec_bell.sample(transpiler.executor(), shots=1000).result()

print("=== Bell State Measurement Results ===")
for value, count in result_bell.results:
    percentage = count / 1000 * 100
    print(f"  Result: {value}, Count: {count} ({percentage:.1f}%)")

# %% [markdown]
# ### Characteristics of Bell States
#
# Looking at the results, only `(0, 0)` and `(1, 1)` appear, never `(0, 1)` or `(1, 0)`.
#
# This is evidence of entanglement:
# - The two qubits are **perfectly correlated**
# - If one measures to 0, the other is definitely 0
# - If one measures to 1, the other is definitely 1
# - However, which one appears is unknown until measurement (50/50)
#
# This correlation is a uniquely quantum mechanical phenomenon that cannot be explained by classical probability.

# %% [markdown]
# ### Circuit Visualization

# %%
qiskit_bell = transpiler.to_circuit(bell_state)
print("=== Circuit for Creating Bell State ===")
print(qiskit_bell.draw(output="text"))

# %% [markdown]
# ## 4. The Four Bell States
#
# There are four Bell states, which are fundamental states in quantum information.

# %%
@qm.qkernel
def bell_phi_plus() -> tuple[qm.Bit, qm.Bit]:
    """|Φ+⟩ = (|00⟩ + |11⟩)/√2"""
    q0, q1 = qm.qubit(name="q0"), qm.qubit(name="q1")
    q0 = qm.h(q0)
    q0, q1 = qm.cx(q0, q1)
    return qm.measure(q0), qm.measure(q1)


@qm.qkernel
def bell_phi_minus() -> tuple[qm.Bit, qm.Bit]:
    """|Φ−⟩ = (|00⟩ − |11⟩)/√2"""
    q0, q1 = qm.qubit(name="q0"), qm.qubit(name="q1")
    q0 = qm.h(q0)
    q0, q1 = qm.cx(q0, q1)
    q0 = qm.rz(q0, math.pi)  # Phase flip
    return qm.measure(q0), qm.measure(q1)


@qm.qkernel
def bell_psi_plus() -> tuple[qm.Bit, qm.Bit]:
    """|Ψ+⟩ = (|01⟩ + |10⟩)/√2"""
    q0, q1 = qm.qubit(name="q0"), qm.qubit(name="q1")
    q0 = qm.h(q0)
    q0, q1 = qm.cx(q0, q1)
    q1 = qm.x(q1)  # Flip target
    return qm.measure(q0), qm.measure(q1)


@qm.qkernel
def bell_psi_minus() -> tuple[qm.Bit, qm.Bit]:
    """|Ψ−⟩ = (|01⟩ − |10⟩)/√2"""
    q0, q1 = qm.qubit(name="q0"), qm.qubit(name="q1")
    q0 = qm.h(q0)
    q0, q1 = qm.cx(q0, q1)
    q1 = qm.x(q1)
    q0 = qm.rz(q0, math.pi)
    return qm.measure(q0), qm.measure(q1)


# %%
bell_states = [
    ("Φ+", bell_phi_plus),
    ("Φ−", bell_phi_minus),
    ("Ψ+", bell_psi_plus),
    ("Ψ−", bell_psi_minus),
]

print("=== The Four Bell States ===\n")

for name, circuit in bell_states:
    exec_b = transpiler.transpile(circuit)
    result_b = exec_b.sample(transpiler.executor(), shots=1000).result()

    print(f"|{name}⟩:")
    for value, count in result_b.results:
        print(f"  {value}: {count}")
    print()

# %% [markdown]
# ## 5. Qubit Arrays: `qubit_array()` and `Vector[Qubit]`
#
# When working with many qubits, it's convenient to create them as an array using `qubit_array()`.

# %%
@qm.qkernel
def array_example() -> qm.Vector[qm.Bit]:
    """Basic usage of qubit arrays"""
    # Create an array of 3 qubits
    qubits = qm.qubit_array(3, name="q")

    # Access by index
    qubits[0] = qm.h(qubits[0])
    qubits[1] = qm.x(qubits[1])
    # qubits[2] is left as |0⟩

    # Measure the entire array
    return qm.measure(qubits)


# %%
exec_arr = transpiler.transpile(array_example)
result_arr = exec_arr.sample(transpiler.executor(), shots=1000).result()

print("=== Qubit Array Example ===")
print("q[0]: H (superposition), q[1]: X (flip), q[2]: nothing")
print()
for value, count in result_arr.results:
    print(f"  Result: {value}, Count: {count}")

# %% [markdown]
# ### Result Format
#
# Results returned as `Vector[Bit]` are displayed in tuple format.
# Example: `(1, 1, 0)` means q[0]=1, q[1]=1, q[2]=0.

# %% [markdown]
# ## 6. Loop Processing: `qm.range()`
#
# To apply operations to each element of an array, write loops using `qm.range()`.
#
# **Note**: Use `qm.range()` instead of Python's regular `range()`.

# %%
@qm.qkernel
def loop_example(n: int) -> qm.Vector[qm.Bit]:
    """Apply H gate to all qubits using a loop"""
    qubits = qm.qubit_array(n, name="q")

    # Apply H gate to all qubits
    for i in qm.range(n):
        qubits[i] = qm.h(qubits[i])

    return qm.measure(qubits)


# %%
exec_loop = transpiler.transpile(loop_example, bindings={"n": 4})
result_loop = exec_loop.sample(transpiler.executor(), shots=1000).result()

print("=== Loop: All Qubits in Superposition (n=4) ===")
print("All 2^4=16 patterns should appear equally\n")

# Sort and display results
sorted_results = sorted(result_loop.results, key=lambda x: str(x[0]))
for value, count in sorted_results:
    percentage = count / 1000 * 100
    print(f"  {value}: {count} ({percentage:.1f}%)")

# %% [markdown]
# ## 7. GHZ State: Multi-Qubit Entanglement
#
# The **GHZ state** (Greenberger–Horne–Zeilinger state) is entanglement of 3 or more qubits.
#
# $$|GHZ\rangle = \frac{|00...0\rangle + |11...1\rangle}{\sqrt{2}}$$
#
# It's an extension of the Bell state, where all qubits are either "all 0" or "all 1".

# %%
@qm.qkernel
def ghz_state(n: int) -> qm.Vector[qm.Bit]:
    """Create N-qubit GHZ state"""
    qubits = qm.qubit_array(n, name="q")

    # Put the first qubit in superposition
    qubits[0] = qm.h(qubits[0])

    # Apply CNOT in a chain to spread entanglement
    for i in qm.range(n - 1):
        qubits[i], qubits[i + 1] = qm.cx(qubits[i], qubits[i + 1])

    return qm.measure(qubits)


# %%
# 3-qubit GHZ state
exec_ghz3 = transpiler.transpile(ghz_state, bindings={"n": 3})
result_ghz3 = exec_ghz3.sample(transpiler.executor(), shots=1000).result()

print("=== 3-Qubit GHZ State ===")
print("|GHZ⟩ = (|000⟩ + |111⟩)/√2\n")
for value, count in result_ghz3.results:
    percentage = count / 1000 * 100
    print(f"  {value}: {count} ({percentage:.1f}%)")

# %%
# 5-qubit GHZ state
exec_ghz5 = transpiler.transpile(ghz_state, bindings={"n": 5})
result_ghz5 = exec_ghz5.sample(transpiler.executor(), shots=1000).result()

print("\n=== 5-Qubit GHZ State ===")
print("|GHZ⟩ = (|00000⟩ + |11111⟩)/√2\n")
for value, count in result_ghz5.results:
    percentage = count / 1000 * 100
    print(f"  {value}: {count} ({percentage:.1f}%)")

# %% [markdown]
# ### Characteristics of GHZ State
#
# - All N qubits are perfectly correlated
# - Measuring even one determines all the rest
# - Output is either "all 0" or "all 1" only

# %% [markdown]
# ### GHZ Circuit Visualization

# %%
qiskit_ghz = transpiler.to_circuit(ghz_state, bindings={"n": 4})
print("=== 4-Qubit GHZ Circuit ===")
print(qiskit_ghz.draw(output="text"))

# %% [markdown]
# ## 8. Other Two-Qubit Gates
#
# Here are the main two-qubit gates available in Qamomile.

# %% [markdown]
# ### SWAP Gate
#
# Exchanges the states of two qubits.

# %%
@qm.qkernel
def swap_example() -> tuple[qm.Bit, qm.Bit]:
    """Exchange states with SWAP gate"""
    q0 = qm.qubit(name="q0")
    q1 = qm.qubit(name="q1")

    # Set q0 to |1⟩, keep q1 as |0⟩
    q0 = qm.x(q0)

    # Exchange with SWAP
    q0, q1 = qm.swap(q0, q1)

    # After swap: q0=|0⟩, q1=|1⟩
    return qm.measure(q0), qm.measure(q1)


# %%
exec_swap = transpiler.transpile(swap_example)
result_swap = exec_swap.sample(transpiler.executor(), shots=1000).result()

print("=== SWAP Gate Example ===")
print("Before SWAP: q0=|1⟩, q1=|0⟩")
print("After SWAP: q0=|0⟩, q1=|1⟩\n")
for value, count in result_swap.results:
    print(f"  Result: {value}, Count: {count}")

# %% [markdown]
# ### RZZ Gate
#
# Applies Z rotation to both qubits simultaneously.
# This is an important gate for quantum optimization algorithms (QAOA).

# %%
@qm.qkernel
def rzz_example(theta: qm.Float) -> tuple[qm.Bit, qm.Bit]:
    """RZZ gate example"""
    q0 = qm.qubit(name="q0")
    q1 = qm.qubit(name="q1")

    # Create superposition
    q0 = qm.h(q0)
    q1 = qm.h(q1)

    # Apply RZZ gate
    q0, q1 = qm.rzz(q0, q1, angle=theta)

    # Apply H for interference
    q0 = qm.h(q0)
    q1 = qm.h(q1)

    return qm.measure(q0), qm.measure(q1)


# %%
exec_rzz = transpiler.transpile(rzz_example, bindings={"theta": math.pi / 2})
result_rzz = exec_rzz.sample(transpiler.executor(), shots=1000).result()

print("=== RZZ Gate Example (θ=π/2) ===")
for value, count in result_rzz.results:
    print(f"  Result: {value}, Count: {count}")

# %% [markdown]
# ### CP Gate (Controlled Phase Gate)
#
# Adds phase to the target when the control bit is |1⟩.
# Used in Quantum Fourier Transform (QFT).

# %%
@qm.qkernel
def cp_example() -> tuple[qm.Bit, qm.Bit]:
    """CP gate example"""
    q0 = qm.qubit(name="q0")
    q1 = qm.qubit(name="q1")

    # Put both in superposition
    q0 = qm.h(q0)
    q1 = qm.h(q1)

    # Apply CP gate (phase of π/2)
    q0, q1 = qm.cp(q0, q1, math.pi / 2)

    # Apply H for interference
    q0 = qm.h(q0)
    q1 = qm.h(q1)

    return qm.measure(q0), qm.measure(q1)


# %%
exec_cp = transpiler.transpile(cp_example)
result_cp = exec_cp.sample(transpiler.executor(), shots=1000).result()

print("=== CP Gate Example (θ=π/2) ===")
for value, count in result_cp.results:
    print(f"  Result: {value}, Count: {count}")

# %% [markdown]
# ## 9. Summary
#
# In this tutorial, you learned:
#
# ### Working with Multiple Qubits
# ```python
# # Create individually
# q0 = qm.qubit(name="q0")
# q1 = qm.qubit(name="q1")
#
# # Create as array
# qubits = qm.qubit_array(n, name="q")
# ```
#
# ### Two-Qubit Gates
# | Gate | Qamomile | Effect |
# |------|----------|--------|
# | CNOT | `qm.cx(ctrl, tgt)` | Controlled-NOT gate |
# | SWAP | `qm.swap(q0, q1)` | Exchange states |
# | RZZ | `qm.rzz(q0, q1, θ)` | ZZ interaction |
# | CP | `qm.cp(ctrl, tgt, θ)` | Controlled phase gate |
#
# ### Important: Return Values of Two-Qubit Gates
# ```python
# # Always receive both!
# q0, q1 = qm.cx(q0, q1)
# ```
#
# ### Loop Processing
# ```python
# for i in qm.range(n):
#     qubits[i] = qm.h(qubits[i])
# ```
#
# ### Important States
# - **Bell states**: Maximum entanglement of 2 qubits
# - **GHZ states**: Entanglement of N qubits
#
# In the next tutorial (`04_algorithms.py`), you will use this knowledge
# to implement your first quantum algorithm: the Deutsch-Jozsa algorithm.
