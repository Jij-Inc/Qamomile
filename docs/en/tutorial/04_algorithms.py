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
# # Introduction to Quantum Algorithms: The Deutsch-Jozsa Algorithm
#
# In this tutorial, you will learn about the **Deutsch-Jozsa algorithm**,
# your first real quantum algorithm.
# This algorithm is historically important as the first example showing
# that quantum computers can outperform classical computers.
#
# ## What You Will Learn
# - What quantum algorithms are
# - The concept of oracles (black-box functions)
# - The Deutsch-Jozsa problem and its quantum solution
# - Exploiting quantum parallelism and interference
# - Next steps: More advanced algorithms

# %%
import qamomile.circuit as qm
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## 1. What is a Quantum Algorithm?
#
# A **quantum algorithm** is a procedure for solving problems by leveraging the properties of quantum computers.
#
# Key quantum properties that quantum algorithms exploit:
#
# 1. **Superposition**: Processing multiple states simultaneously
# 2. **Entanglement**: Strong correlations between qubits
# 3. **Quantum interference**: Increasing the probability of correct answers and decreasing that of wrong ones
#
# The Deutsch-Jozsa algorithm is the simplest example that uses all of these.

# %% [markdown]
# ## 2. The Deutsch-Jozsa Problem
#
# ### Problem Setup
#
# Suppose we have a function $f: \{0,1\}^n \rightarrow \{0,1\}$.
# This function is guaranteed to be one of the following:
#
# - **Constant function**: Returns the same value for all inputs
#   - Example: f(00)=0, f(01)=0, f(10)=0, f(11)=0 (all 0)
#   - Example: f(00)=1, f(01)=1, f(10)=1, f(11)=1 (all 1)
#
# - **Balanced function**: Returns 0 for half the inputs and 1 for the other half
#   - Example: f(00)=0, f(01)=0, f(10)=1, f(11)=1
#   - Example: f(00)=0, f(01)=1, f(10)=0, f(11)=1
#
# **Problem**: Determine whether the function is "constant" or "balanced"
#
# ### Classical Approach
#
# On a classical computer, in the worst case, $2^{n-1} + 1$ function calls are needed.
# (Try half+1 inputs; if all same → constant, if different results appear → balanced)
#
# ### Quantum Approach
#
# On a quantum computer, we can determine this with **just one** function call!

# %% [markdown]
# ## 3. What is an Oracle?
#
# In quantum algorithms, we treat the function $f$ as an "oracle."
# An oracle is given as a **black box**, with unknown internal structure.
#
# ### Structure of a Quantum Oracle
#
# A quantum oracle acts on an input register and an auxiliary bit (ancilla):
#
# $$U_f |x\rangle |y\rangle = |x\rangle |y \oplus f(x)\rangle$$
#
# - $|x\rangle$: Input (unchanged)
# - $|y\rangle$: Auxiliary bit
# - $\oplus$: XOR operation
# - $f(x)$: Function output

# %% [markdown]
# ## 4. Algorithm Implementation
#
# First, let's define various oracles.

# %%
# === Constant oracle (always returns 0) ===
@qm.qkernel
def oracle_constant_0(
    inputs: qm.Vector[qm.Qubit], ancilla: qm.Qubit
) -> tuple[qm.Vector[qm.Qubit], qm.Qubit]:
    """Constant function: f(x) = 0 for all x"""
    # Do nothing (always returns 0)
    return inputs, ancilla


# === Constant oracle (always returns 1) ===
@qm.qkernel
def oracle_constant_1(
    inputs: qm.Vector[qm.Qubit], ancilla: qm.Qubit
) -> tuple[qm.Vector[qm.Qubit], qm.Qubit]:
    """Constant function: f(x) = 1 for all x"""
    # Flip the ancilla (effect of always returning 1)
    ancilla = qm.x(ancilla)
    return inputs, ancilla


# === Balanced oracle (XOR parity) ===
@qm.qkernel
def oracle_balanced_xor(
    inputs: qm.Vector[qm.Qubit], ancilla: qm.Qubit
) -> tuple[qm.Vector[qm.Qubit], qm.Qubit]:
    """Balanced function: f(x) = x_0 XOR x_1 XOR ... XOR x_{n-1}"""
    n = inputs.shape[0]
    for i in qm.range(n):
        inputs[i], ancilla = qm.cx(inputs[i], ancilla)
    return inputs, ancilla


# === Balanced oracle (first bit only) ===
@qm.qkernel
def oracle_balanced_first_bit(
    inputs: qm.Vector[qm.Qubit], ancilla: qm.Qubit
) -> tuple[qm.Vector[qm.Qubit], qm.Qubit]:
    """Balanced function: f(x) = x_0 (value of first bit)"""
    inputs[0], ancilla = qm.cx(inputs[0], ancilla)
    return inputs, ancilla


# %% [markdown]
# ### The Main Deutsch-Jozsa Algorithm
#
# Algorithm steps:
#
# 1. Initialize input register to $|0\rangle^{\otimes n}$, ancilla to $|1\rangle$
# 2. Apply Hadamard gates to all
# 3. Apply the oracle
# 4. Apply Hadamard to input register again
# 5. Measure the input register
#
# **Interpreting results**:
# - All $|0\rangle$ → constant function
# - Otherwise → balanced function

# %%
@qm.qkernel
def deutsch_jozsa_constant_0(n: int) -> qm.Vector[qm.Bit]:
    """Deutsch-Jozsa with constant oracle (f=0)"""
    inputs = qm.qubit_array(n, name="input")
    ancilla = qm.qubit(name="ancilla")

    # Step 1: Initialize
    ancilla = qm.x(ancilla)  # Set ancilla to |1⟩

    # Step 2: Apply Hadamard to all
    for i in qm.range(n):
        inputs[i] = qm.h(inputs[i])
    ancilla = qm.h(ancilla)

    # Step 3: Apply oracle
    inputs, ancilla = oracle_constant_0(inputs, ancilla)

    # Step 4: Apply Hadamard to input register
    for i in qm.range(n):
        inputs[i] = qm.h(inputs[i])

    # Step 5: Measure input register
    return qm.measure(inputs)


@qm.qkernel
def deutsch_jozsa_constant_1(n: int) -> qm.Vector[qm.Bit]:
    """Deutsch-Jozsa with constant oracle (f=1)"""
    inputs = qm.qubit_array(n, name="input")
    ancilla = qm.qubit(name="ancilla")

    ancilla = qm.x(ancilla)

    for i in qm.range(n):
        inputs[i] = qm.h(inputs[i])
    ancilla = qm.h(ancilla)

    inputs, ancilla = oracle_constant_1(inputs, ancilla)

    for i in qm.range(n):
        inputs[i] = qm.h(inputs[i])

    return qm.measure(inputs)


@qm.qkernel
def deutsch_jozsa_balanced_xor(n: int) -> qm.Vector[qm.Bit]:
    """Deutsch-Jozsa with balanced oracle (XOR)"""
    inputs = qm.qubit_array(n, name="input")
    ancilla = qm.qubit(name="ancilla")

    ancilla = qm.x(ancilla)

    for i in qm.range(n):
        inputs[i] = qm.h(inputs[i])
    ancilla = qm.h(ancilla)

    inputs, ancilla = oracle_balanced_xor(inputs, ancilla)

    for i in qm.range(n):
        inputs[i] = qm.h(inputs[i])

    return qm.measure(inputs)


@qm.qkernel
def deutsch_jozsa_balanced_first(n: int) -> qm.Vector[qm.Bit]:
    """Deutsch-Jozsa with balanced oracle (first bit)"""
    inputs = qm.qubit_array(n, name="input")
    ancilla = qm.qubit(name="ancilla")

    ancilla = qm.x(ancilla)

    for i in qm.range(n):
        inputs[i] = qm.h(inputs[i])
    ancilla = qm.h(ancilla)

    inputs, ancilla = oracle_balanced_first_bit(inputs, ancilla)

    for i in qm.range(n):
        inputs[i] = qm.h(inputs[i])

    return qm.measure(inputs)


# %% [markdown]
# ## 5. Execution and Results

# %%
n = 3  # Number of input bits

test_cases = [
    ("Constant (f=0)", deutsch_jozsa_constant_0),
    ("Constant (f=1)", deutsch_jozsa_constant_1),
    ("Balanced (XOR)", deutsch_jozsa_balanced_xor),
    ("Balanced (first bit)", deutsch_jozsa_balanced_first),
]

print(f"=== Deutsch-Jozsa Algorithm (n={n}) ===\n")
print("Decision rule: all 0 → constant, otherwise → balanced\n")

for name, circuit in test_cases:
    exec_dj = transpiler.transpile(circuit, bindings={"n": n})
    result_dj = exec_dj.sample(transpiler.executor(), shots=1000).result()

    print(f"{name}:")
    for value, count in result_dj.results:
        # Judgment
        all_zero = all(v == 0 for v in value)
        judgment = "constant" if all_zero else "balanced"
        print(f"  Result: {value}, Count: {count}, Judgment: {judgment}")
    print()

# %% [markdown]
# ### Interpreting the Results
#
# - **Constant oracle**: Measurement result is always `(0, 0, 0)` → correctly identified as "constant"
# - **Balanced oracle**: Measurement result is not `(0, 0, 0)` → correctly identified as "balanced"
#
# The important point is that we can determine this with **just one measurement**!
# Classically, the worst case requires $2^{n-1}+1 = 5$ function calls.

# %% [markdown]
# ## 6. Circuit Visualization

# %%
# Visualize the balanced oracle (XOR) circuit
qiskit_dj = transpiler.to_circuit(deutsch_jozsa_balanced_xor, bindings={"n": 3})
print("=== Deutsch-Jozsa Circuit (Balanced Oracle XOR, n=3) ===")
print(qiskit_dj.draw(output="text"))

# %% [markdown]
# ### Circuit Structure
#
# 1. First column: Hadamard gates (create superposition)
# 2. Middle: Oracle (implemented with CNOT gates)
# 3. Last column: Hadamard gates (cause interference)
# 4. Measurement

# %% [markdown]
# ## 7. Why Does It Work? (Intuitive Explanation)
#
# ### Quantum Parallelism
#
# When we create superposition with Hadamard gates, the input register becomes
# a superposition of all possible inputs $|00...0\rangle, |00...1\rangle, ..., |11...1\rangle$.
#
# With just one oracle call, **information about $f(x)$ for all inputs**
# is encoded in the quantum state.
#
# ### Quantum Interference
#
# The second Hadamard causes probability amplitudes of correct answers to **reinforce**
# and those of wrong answers to **cancel**.
#
# - Constant function: All amplitudes have the same phase → concentrate on $|00...0\rangle$
# - Balanced function: Half have opposite phase → $|00...0\rangle$ cancels out

# %% [markdown]
# ## 8. The Power of Quantum Algorithms
#
# Deutsch-Jozsa is a "toy problem," but it teaches important lessons:
#
# | Metric | Classical | Quantum |
# |--------|-----------|---------|
# | Query count | $O(2^n)$ | $O(1)$ |
# | Certainty | Deterministic | Deterministic (100% correct) |
#
# This **exponential speedup** is achieved in more practical algorithms as well:
#
# - **Grover's algorithm**: $\sqrt{N}$ speedup for database search
# - **Shor's algorithm**: Exponential speedup for integer factorization
# - **QAOA**: Approximate solutions for combinatorial optimization problems

# %% [markdown]
# ## 9. Next Steps
#
# Congratulations! You have now mastered the basics of quantum programming with Qamomile.
#
# ### What You Learned
#
# 1. **01_introduction.py**: Qamomile basics, linear type system
# 2. **02_single_qubit.py**: Superposition, rotation gates, parameterization
# 3. **03_entanglement.py**: Entanglement, Bell states, GHZ states
# 4. **04_algorithms.py**: Deutsch-Jozsa algorithm
#
# ### Tutorials to Learn Next
#
# You are now ready to learn more advanced quantum algorithms:
#
# - **`qpe.py` (Quantum Phase Estimation)**: Important subroutine for estimating eigenvalues
#   - Create controlled gates with `qm.controlled()`
#   - Automatically decode phases with the `QFixed` type
#   - Foundation for Shor's algorithm and quantum chemistry simulations
#
# - **`qaoa.py` (Quantum Approximate Optimization Algorithm)**: Solving combinatorial optimization problems
#   - Advanced usage of parameterized circuits
#   - Combination with classical optimization (hybrid algorithms)
#   - Encoding cost functions using `rzz` gates

# %% [markdown]
# ## 10. Summary
#
# ### Qamomile Programming Pattern
#
# ```python
# import qamomile.circuit as qm
# from qamomile.qiskit import QiskitTranspiler
#
# # 1. Define quantum kernel
# @qm.qkernel
# def my_algorithm(n: int) -> qm.Vector[qm.Bit]:
#     qubits = qm.qubit_array(n, name="q")
#
#     # Quantum operations (linear type: always reassign!)
#     for i in qm.range(n):
#         qubits[i] = qm.h(qubits[i])
#
#     return qm.measure(qubits)
#
# # 2. Transpile and execute
# transpiler = QiskitTranspiler()
# executable = transpiler.transpile(my_algorithm, bindings={"n": 3})
# result = executable.sample(transpiler.executor(), shots=1000).result()
#
# # 3. Analyze results
# for value, count in result.results:
#     print(f"{value}: {count}")
# ```
#
# ### Key Points
#
# 1. **`@qm.qkernel`**: Decorator for defining quantum circuits
# 2. **Linear type**: Always reassign like `q = qm.h(q)`
# 3. **Two-qubit gates**: Receive both with `q0, q1 = qm.cx(q0, q1)`
# 4. **Arrays**: `qm.qubit_array(n, name="q")` and `qm.range(n)`
# 5. **Transpilation**: Convert to backend with `QiskitTranspiler`
#
# Use this knowledge to tackle `qpe.py` and `qaoa.py`!
