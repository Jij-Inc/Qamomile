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
# # Our First Quantum Algorithm: The Deutsch-Jozsa Algorithm
#
# In this tutorial, we will learn about the **Deutsch-Jozsa algorithm**,
# our first real quantum algorithm.
# The Deutsch-Jozsa algorithm is one of the earliest quantum algorithms,
# generalizing Deutsch's original single-bit algorithm (1985) to $n$ bits.
# It provided one of the first clear demonstrations that quantum computers
# can solve certain problems exponentially faster than classical ones.
#
# ## What We Will Learn
# - What quantum algorithms are
# - The concept of oracles (black-box functions)
# - The Deutsch-Jozsa problem and its quantum solution
# - Passing oracles as arguments to build reusable quantum algorithms
# - Exploiting quantum parallelism and interference

# %%
import qamomile.circuit as qmc
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
# **Constant functions** — same output for every input:
#
# | x  | f(x)=0 | f(x)=1 |
# |----|--------|--------|
# | 00 | 0      | 1      |
# | 01 | 0      | 1      |
# | 10 | 0      | 1      |
# | 11 | 0      | 1      |
#
# **Balanced functions** — 0 for exactly half the inputs, 1 for the other half:
#
# | x  | f₁(x) | f₂(x) |
# |----|-------|-------|
# | 00 | 0     | 0     |
# | 01 | 0     | 1     |
# | 10 | 1     | 0     |
# | 11 | 1     | 1     |
#
# **Problem**: Determine whether the function is "constant" or "balanced"
#
# ### Classical Approach
#
# On a classical computer, in the worst case, $2^{n-1} + 1$ function calls are needed.
# (Try half+1 inputs; if all same -> constant, if different results appear -> balanced)
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
# ## 4. Defining Oracles
#
# Let's define various oracles as QKernel functions.


# %%
# === Constant oracle (always returns 0) ===
@qmc.qkernel
def oracle_constant_0(
    inputs: qmc.Vector[qmc.Qubit], ancilla: qmc.Qubit
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
    """Constant function: f(x) = 0 for all x."""
    # Do nothing (always returns 0)
    return inputs, ancilla


oracle_constant_0.draw(inputs=2)


# %%
# === Constant oracle (always returns 1) ===
@qmc.qkernel
def oracle_constant_1(
    inputs: qmc.Vector[qmc.Qubit], ancilla: qmc.Qubit
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
    """Constant function: f(x) = 1 for all x."""
    # Flip the ancilla (effect of always returning 1)
    ancilla = qmc.x(ancilla)
    return inputs, ancilla


oracle_constant_1.draw(inputs=2)


# %%
# === Balanced oracle (XOR parity) ===
@qmc.qkernel
def oracle_balanced_xor(
    inputs: qmc.Vector[qmc.Qubit], ancilla: qmc.Qubit
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
    """Balanced function: f(x) = x_0 XOR x_1 XOR ... XOR x_{n-1}."""
    n = inputs.shape[0]
    for i in qmc.range(n):
        inputs[i], ancilla = qmc.cx(inputs[i], ancilla)
    return inputs, ancilla


oracle_balanced_xor.draw(inputs=2, fold_loops=False)


# %%
# === Balanced oracle (first bit only) ===
@qmc.qkernel
def oracle_balanced_first_bit(
    inputs: qmc.Vector[qmc.Qubit], ancilla: qmc.Qubit
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
    """Balanced function: f(x) = x_0 (value of first bit)."""
    inputs[0], ancilla = qmc.cx(inputs[0], ancilla)
    return inputs, ancilla


oracle_balanced_first_bit.draw(inputs=2)

# %% [markdown]
# ## 5. The Deutsch-Jozsa Algorithm
#
# Instead of writing a separate function for each oracle, we create a **reusable**
# Deutsch-Jozsa function that accepts any oracle as an argument.
#
# ### Algorithm Steps
#
# 1. Initialize input register to $|0\rangle^{\otimes n}$, ancilla to $|1\rangle$
# 2. Apply Hadamard gates to all
# 3. Apply the oracle
# 4. Apply Hadamard to input register again
# 5. Measure the input register
#
# **Interpreting results**:
# - All $|0\rangle$ -> constant function
# - Otherwise -> balanced function


# %%
def deutsch_jozsa(oracle):
    """Create a Deutsch-Jozsa circuit with the given oracle.

    This is a factory function that returns a QKernel circuit.
    The oracle is captured in the closure and called during tracing.
    """

    @qmc.qkernel
    def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        inputs = qmc.qubit_array(n, name="input")
        ancilla = qmc.qubit(name="ancilla")

        # Step 1: Initialize ancilla to |1⟩
        ancilla = qmc.x(ancilla)

        # Step 2: Apply Hadamard to all
        for i in qmc.range(n):
            inputs[i] = qmc.h(inputs[i])
        ancilla = qmc.h(ancilla)

        # Step 3: Apply oracle
        inputs, ancilla = oracle(inputs, ancilla)

        # Step 4: Apply Hadamard to input register
        for i in qmc.range(n):
            inputs[i] = qmc.h(inputs[i])

        # Step 5: Measure input register
        return qmc.measure(inputs)

    return circuit


# %% [markdown]
# Now we can create Deutsch-Jozsa circuits for any oracle:

# %%
dj_constant_0 = deutsch_jozsa(oracle_constant_0)
dj_constant_1 = deutsch_jozsa(oracle_constant_1)
dj_balanced_xor = deutsch_jozsa(oracle_balanced_xor)
dj_balanced_first = deutsch_jozsa(oracle_balanced_first_bit)

# %% [markdown]
# The `draw()` method has an `inline` parameter that controls how called kernels
# (like the oracle) are displayed:
#
# - `inline=True`: Expands the oracle's gates directly into the circuit diagram,
#   showing every gate in a single flat view.
# - `inline=False` (default): Shows the oracle as a labelled box, preserving the
#   modular structure. This is useful for understanding the high-level algorithm
#   without being distracted by implementation details.

# %%
# With inlining — all gates visible:
dj_balanced_xor.draw(n=2, fold_loops=False, inline=True)

# %%
# Without inlining — the oracle appears as a named box:
dj_balanced_xor.draw(n=2, fold_loops=False)

# %% [markdown]
# ## 6. Execution and Results

# %%
n = 2  # Number of input bits

test_cases = [
    ("Constant (f=0)", dj_constant_0),
    ("Constant (f=1)", dj_constant_1),
    ("Balanced (XOR)", dj_balanced_xor),
    ("Balanced (first bit)", dj_balanced_first),
]

print(f"=== Deutsch-Jozsa Algorithm (n={n}) ===\n")
print("Decision rule: all 0 -> constant, otherwise -> balanced\n")

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
# - **Constant oracle**: Measurement result is always `(0, 0)` -> correctly identified as "constant"
# - **Balanced oracle**: Measurement result is not `(0, 0)` -> correctly identified as "balanced"
#
# The important point is that we can determine this with **just one measurement**!
# Classically, the worst case requires $2^{n-1}+1 = 3$ function calls.

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
# ### Phase Kickback (the Role of Entanglement)
#
# The ancilla qubit is prepared in $|-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$.
# When the oracle applies $U_f$, it briefly entangles the input register with the ancilla.
# The key identity is:
#
# $$U_f |x\rangle |-\rangle = (-1)^{f(x)} |x\rangle |-\rangle$$
#
# The ancilla returns to $|-\rangle$ (unchanged), but the input register picks up
# a phase factor $(-1)^{f(x)}$. This is called **phase kickback**: information about
# $f(x)$ is "kicked back" as a phase onto the input register through the momentary
# entanglement with the ancilla.
#
# Without this entanglement-mediated phase transfer, the algorithm would not work.
#
# ### Quantum Interference
#
# The second Hadamard causes probability amplitudes of correct answers to **reinforce**
# and those of wrong answers to **cancel**.
#
# - Constant function: All amplitudes have the same phase -> concentrate on $|00...0\rangle$
# - Balanced function: Half have opposite phase -> $|00...0\rangle$ cancels out

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
# This **quantum speedup** is achieved in more practical algorithms as well:
#
# - **Grover's algorithm**: $\sqrt{N}$ speedup for database search
# - **Shor's algorithm**: Exponential speedup for integer factorization

# %% [markdown]
# ## 9. Summary
#
# This tutorial covered the **Deutsch-Jozsa algorithm**, our first complete
# quantum algorithm built with Qamomile.
#
# ### Key Takeaways
#
# 1. **Oracle pattern**: Oracles are defined as `@qkernel` functions with a standard
#    signature `(inputs, ancilla) -> (inputs, ancilla)`, making them interchangeable.
#
# 2. **Oracle-as-argument**: Wrapping the algorithm in a factory function that accepts
#    an oracle separates algorithm structure from problem-specific logic:
#
#    ```python
#    def deutsch_jozsa(oracle):
#        @qmc.qkernel
#        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
#            ...
#            inputs, ancilla = oracle(inputs, ancilla)
#            ...
#            return qmc.measure(inputs)
#        return circuit
#    ```
#
# 3. **Quantum speedup**: The algorithm determines constant vs. balanced in $O(1)$
#    oracle queries, compared to $O(2^n)$ classically.
#
# 4. **Three quantum ingredients**: Superposition (parallel evaluation), phase kickback
#    via entanglement (information transfer), and interference (answer extraction).
#
# ### Next Tutorials
#
# - [Standard Library](05_stdlib.ipynb): QFT, IQFT, and QPE with `qmc.controlled()` and `QFixed`
# - [Parametric Circuits](08_parametric_circuits.ipynb): Variational quantum algorithms and hybrid optimization

# %% [markdown]
# ## What We Learned
#
# - **What quantum algorithms are** — Structured sequences of quantum gates designed to exploit superposition and interference for computational advantage over classical approaches.
# - **The concept of oracles (black-box functions)** — Oracles encode problem-specific logic as `@qkernel` functions that can be passed as arguments to algorithm templates.
# - **The Deutsch-Jozsa problem and its quantum solution** — Determines whether a function is constant or balanced in one query, compared to $2^{n-1}+1$ classically.
# - **Passing oracles as arguments to build reusable quantum algorithms** — The oracle-as-argument pattern separates algorithm structure from problem specifics, enabling reuse with different oracles.
# - **Exploiting quantum parallelism and interference** — Hadamard gates create superposition over all inputs; the oracle marks phases; final Hadamards convert phase differences into measurable outcomes.
