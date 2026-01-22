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
# # Introduction to Qamomile: Your First Steps in Quantum Computing
#
# In this tutorial, you will learn the basics of quantum computing using Qamomile.
# If you have programming experience, no prior knowledge of quantum computing is required.
#
# ## What You Will Learn
# - What Qamomile is and its key features
# - Basic concepts of qubits
# - Creating and running your first quantum circuit
# - Qamomile's important design philosophy: the linear type system

# %% [markdown]
# ## 1. What is Qamomile?
#
# **Qamomile** is a library for writing quantum circuits in Python.
# It has the following features:
#
# 1. **Pythonic syntax**: Use the `@qkernel` decorator to write quantum circuits like regular Python functions
# 2. **Type safety**: Type annotations enable compile-time error detection
# 3. **Multi-backend**: The same code can run on various SDKs like Qiskit, QuriParts, PennyLane, etc.
# 4. **Linear type system**: Safely tracks qubit states and prevents common bugs

# %% [markdown]
# ## 2. What is a Qubit?
#
# ### Classical Bits vs Qubits
#
# In regular computers, a **bit** is the smallest unit of information.
# A bit can only have one of two values: **0** or **1**.
#
# ```
# Classical bit: 0 or 1
# ```
#
# In quantum computers, we use **qubits** (quantum bits).
# Qubits can exist in a **superposition** of 0 and 1.
#
# ```
# Qubit: Superposition of |0⟩ and |1⟩ (undetermined until measured)
# ```
#
# Notation explanation:
# - `|0⟩` (ket 0): The "0" state of a qubit
# - `|1⟩` (ket 1): The "1" state of a qubit
#
# ### Measurement
#
# When you **measure** a qubit, the superposition state collapses, and you get either 0 or 1.
# Which value you get is determined probabilistically.

# %% [markdown]
# ## 3. Your First Quantum Circuit
#
# Let's create your first quantum circuit using Qamomile.
# First, import the necessary modules.

# %%
import qamomile.circuit as qm
from qamomile.qiskit import QiskitTranspiler

# %% [markdown]
# ### X Gate (NOT Gate)
#
# One of the most basic quantum gates is the **X gate**.
# The X gate flips the state of a qubit:
# - `|0⟩` → `|1⟩`
# - `|1⟩` → `|0⟩`
#
# It works the same as a "NOT gate" in classical computers.

# %%
@qm.qkernel
def x_gate_circuit() -> qm.Bit:
    """First quantum circuit applying the X gate"""
    # Create a qubit (initial state is |0⟩)
    q = qm.qubit(name="q")

    # Apply X gate to transform |0⟩ → |1⟩
    q = qm.x(q)

    # Measure and return the result
    return qm.measure(q)


# %% [markdown]
# ### Code Explanation
#
# 1. **`@qm.qkernel`**: Decorator indicating this function is a quantum kernel (a function defining a quantum circuit)
# 2. **`-> qm.Bit`**: Return type. The measurement result becomes a classical bit (`Bit`)
# 3. **`qm.qubit(name="q")`**: Creates one qubit. Initial state is `|0⟩`
# 4. **`q = qm.x(q)`**: Applies the X gate. **Important: You must reassign the result**
# 5. **`qm.measure(q)`**: Measures the qubit and gets the result as a classical bit

# %% [markdown]
# ## 4. Linear Type System: Qamomile's Important Design Philosophy
#
# You might wonder why we write `q = qm.x(q)` in the code above.
# This is due to Qamomile's **linear type system**.
#
# ### Why Reassignment is Required
#
# Qubits have a property that they cannot be "copied" (the "no-cloning theorem" in quantum mechanics).
# Qamomile expresses this property through its type system.
#
# ```python
# # Correct way
# q = qm.x(q)  # Reassign the new state to q after applying the gate
#
# # Wrong way (will cause an error)
# qm.x(q)      # If you don't reassign, you'll be referencing the old state
# ```
#
# This design prevents bugs like "accidentally using the same qubit twice."

# %% [markdown]
# ### Linear Type Error Examples: Let's Try It
#
# Let's see what happens when you write code incorrectly.

# %%
# Bad example 1: Using the same qubit twice
@qm.qkernel
def bad_example_reuse() -> tuple[qm.Bit, qm.Bit]:
    q = qm.qubit(name="q")
    q1 = qm.h(q)   # Consume q into q1
    q2 = qm.x(q)   # Bad! q was already used by q1
    return qm.measure(q1), qm.measure(q2)


# %%
# Trying to transpile this circuit will cause an error
try:
    transpiler_test = QiskitTranspiler()
    transpiler_test.transpile(bad_example_reuse)
    print("No error occurred (unexpected behavior)")
except Exception as e:
    print(f"Error occurred (this is the correct behavior):")
    print(f"  {type(e).__name__}: {e}")

# %% [markdown]
# ### Bad Example 2: Ignoring the Return Value
#
# Ignoring the return value of a gate and continuing to use the old variable is also wrong.

# %%
@qm.qkernel
def bad_example_ignore_return() -> qm.Bit:
    q = qm.qubit(name="q")
    qm.h(q)        # Bad! Ignoring the return value
    qm.x(q)        # Bad! Using the old q
    return qm.measure(q)  # This is also the old q


# %%
try:
    transpiler_test = QiskitTranspiler()
    transpiler_test.transpile(bad_example_ignore_return)
    print("No error occurred (unexpected behavior)")
except Exception as e:
    print(f"Error occurred (this is the correct behavior):")
    print(f"  {type(e).__name__}: {e}")

# %% [markdown]
# ### The Correct Way
#
# Always reassign the return value of a gate to the same variable.

# %%
@qm.qkernel
def good_example() -> qm.Bit:
    q = qm.qubit(name="q")
    q = qm.h(q)    # Correct! Reassign the result to q
    q = qm.x(q)    # Correct! Use the updated q
    return qm.measure(q)


# %%
# The correct circuit transpiles without problems
transpiler_test = QiskitTranspiler()
executable_good = transpiler_test.transpile(good_example)
result_good = executable_good.sample(transpiler_test.executor(), shots=100).result()
print("The correct circuit runs without problems:")
for value, count in result_good.results:
    print(f"  Measurement result: {value}, Count: {count}")

# %% [markdown]
# ### Summary: Linear Type Rules
#
# | Code | Correct? | Reason |
# |------|----------|--------|
# | `q = qm.h(q)` | OK | Reassigns the return value |
# | `qm.h(q)` | NG | Ignores the return value |
# | `q1 = qm.h(q); q2 = qm.x(q)` | NG | Uses the same q twice |
# | `q = qm.h(q); q = qm.x(q)` | OK | Updates sequentially |

# %% [markdown]
# ## 5. Running the Quantum Circuit
#
# Let's run the quantum circuit we created.
# In Qamomile, we use a **transpiler** to convert the circuit to a backend (Qiskit in this case) and execute it.

# %%
# Create a transpiler
transpiler = QiskitTranspiler()

# Compile the quantum circuit
executable = transpiler.transpile(x_gate_circuit)

# Run on simulator (1000 measurements)
job = executable.sample(transpiler.executor(), shots=1000)
result = job.result()

# Display results
print("=== X Gate Circuit Results ===")
for value, count in result.results:
    print(f"  Measurement result: {value}, Count: {count}")

# %% [markdown]
# ### Result Explanation
#
# Applying the X gate to `|0⟩` produces `|1⟩`.
# Therefore, the measurement result should always be **1**.
#
# Confirm that in the results above, all 1000 measurements yielded `1`.

# %% [markdown]
# ## 6. Visualizing the Quantum Circuit
#
# You can visualize the structure of the circuit you created.

# %%
# Get the circuit in Qiskit format
qiskit_circuit = transpiler.to_circuit(x_gate_circuit)

# Display the circuit
print("=== Quantum Circuit Structure ===")
print(qiskit_circuit.draw(output="text"))

# %% [markdown]
# ### How to Read the Circuit Diagram
#
# ```
#      ┌───┐┌─┐
#   q: ┤ X ├┤M├
#      └───┘└╥┘
# c: 1/══════╩═
#            0
# ```
#
# - `q`: The qubit line
# - `X`: X gate
# - `M`: Measurement
# - `c`: Classical bit (where the measurement result is stored)

# %% [markdown]
# ## 7. Another Example: Identity Circuit
#
# For comparison, let's create a circuit that applies no gates.

# %%
@qm.qkernel
def identity_circuit() -> qm.Bit:
    """Circuit that does nothing (measures the initial state directly)"""
    q = qm.qubit(name="q")
    # Apply no gates
    return qm.measure(q)


# %%
# Execute
executable_id = transpiler.transpile(identity_circuit)
job_id = executable_id.sample(transpiler.executor(), shots=1000)
result_id = job_id.result()

print("=== Identity Circuit Results ===")
for value, count in result_id.results:
    print(f"  Measurement result: {value}, Count: {count}")

# %% [markdown]
# Since the initial state of a qubit is `|0⟩`, the measurement result is always **0** if nothing is applied.

# %% [markdown]
# ## 8. Summary
#
# In this tutorial, you learned:
#
# 1. **Qamomile basics**: Define quantum circuits with the `@qm.qkernel` decorator
# 2. **Qubits**: Create with `qm.qubit()`, initial state is `|0⟩`
# 3. **Quantum gates**: Manipulate states with gates like `qm.x()`
# 4. **Linear type system**: **Always reassign** after applying gates (`q = qm.x(q)`)
# 5. **Measurement**: Convert quantum states to classical bits with `qm.measure()`
# 6. **Execution**: Compile with `QiskitTranspiler` and run with `sample()`
#
# ### Key Points
#
# ```python
# @qm.qkernel
# def my_circuit() -> qm.Bit:
#     q = qm.qubit(name="q")  # Create qubit
#     q = qm.x(q)              # Apply gate (reassignment required!)
#     return qm.measure(q)     # Measure
# ```
#
# In the next tutorial (`02_single_qubit.py`), you will learn about the **Hadamard gate** that creates superposition states,
# and **rotation gates** that accept angle parameters.
