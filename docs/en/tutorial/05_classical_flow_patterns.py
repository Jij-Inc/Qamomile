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
# # Classical Flow Patterns
#
# Quantum circuits often have structure that depends on classical data:
# iterating over qubits, applying gates based on a graph's edges,
# or choosing between gate sequences. Qamomile supports these patterns
# through `qmc.range`, `qmc.items`, `if` branching, and `while` loops.
#
# This chapter covers:
#
# - `qmc.range()` for loops (recap and deeper usage)
# - `qmc.items()` for iterating over dictionaries
# - `if` and `while` on measurement results for mid-circuit branching

# %%
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## `qmc.range` Loops
#
# We saw `qmc.range(n)` in Tutorial 02 for simple loops.
# Here is a slightly richer example: applying H to all qubits, then
# entangling adjacent pairs with CX.


# %%
@qmc.qkernel
def hadamard_chain(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")

    # Apply H to every qubit
    for i in qmc.range(n):
        q[i] = qmc.h(q[i])

    # Entangle adjacent pairs
    for i in qmc.range(n - 1):
        q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])

    return qmc.measure(q)


# %%
hadamard_chain.draw(n=4)

# %% [markdown]
# ## `qmc.items` for Sparse Interaction Data
#
# Many quantum algorithms (QAOA, VQE) apply gates only on specific pairs
# of qubits, determined by a graph or interaction map. Rather than looping
# over all pairs, you can pass a **dictionary** of interactions and iterate
# with `qmc.items()`.
#
# The dictionary type uses Qamomile's symbolic types:
# `qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]` — keys are qubit
# index pairs, values are interaction weights.


# %%
@qmc.qkernel
def sparse_coupling(
    n: qmc.UInt,
    edges: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")

    # Initial superposition
    for i in qmc.range(n):
        q[i] = qmc.h(q[i])

    # Apply RZZ interactions only on specified edges
    for (i, j), weight in qmc.items(edges):
        q[i], q[j] = qmc.rzz(q[i], q[j], gamma * weight)

    return qmc.measure(q)


# %% [markdown]
# ## Inspecting with `to_circuit()`
#
# `draw()` does not yet support all patterns (particularly `items` with
# complex types). In such cases, use `to_circuit()` to see the concrete
# backend circuit after all parameters are bound.

# %%
edge_data = {(0, 1): 1.0, (1, 2): -0.7, (0, 2): 0.3}

circuit = transpiler.to_circuit(
    sparse_coupling,
    bindings={"n": 3, "edges": edge_data, "gamma": 0.4},
)
print(circuit)

# %% [markdown]
# Only the three edges in `edge_data` produce RZZ gates — no wasted operations.

# %% [markdown]
# ## `if` Branching and `while` Loops
#
# Qamomile supports **mid-circuit measurement** followed by classical
# branching. The condition must be a **measurement result** (`Bit`),
# not a kernel parameter.
#
# This maps directly to hardware-level conditional execution:
# measure a qubit, then decide what to do next based on the outcome.

# %% [markdown]
# ### `if` on a measurement result
#
# A common pattern: measure one qubit and conditionally apply a gate
# to another qubit based on the outcome.

# %%
@qmc.qkernel
def conditional_flip() -> qmc.Bit:
    q0 = qmc.qubit("q0")
    q1 = qmc.qubit("q1")

    q0 = qmc.x(q0)  # Prepare |1⟩
    bit = qmc.measure(q0)

    # Conditionally flip q1 based on q0's measurement
    if bit:
        q1 = qmc.x(q1)
    else:
        q1 = q1  # no-op — both branches must handle q1

    return qmc.measure(q1)


# %% [markdown]
# > **Note**: Both `if` and `else` branches must handle the same qubit
# > handles due to the affine type system. If the true branch applies
# > a gate to `q1`, the false branch must also reassign `q1` (even as
# > a no-op `q1 = q1`).

# %% [markdown]
# This transpiles to a Qiskit `if_else` instruction and can be executed:

# %%
exe = transpiler.transpile(conditional_flip)
executor = transpiler.executor()
job = exe.sample(executor, bindings={}, shots=100)
result = job.result()
for value, count in result.results:
    print(f"  bit={value}: {count} shots")

# %% [markdown]
# Since `q0` is prepared as |1⟩, the measurement always yields 1,
# so `q1` always gets flipped — every shot should return 1.

# %% [markdown]
# ### `while` on a measurement result
#
# A `while` loop repeats until the measurement condition becomes false.
# This is useful for repeat-until-success protocols.

# %%
@qmc.qkernel
def repeat_until_zero() -> qmc.Bit:
    q = qmc.qubit("q")
    q = qmc.h(q)  # 50/50 chance of |0⟩ or |1⟩
    bit = qmc.measure(q)

    while bit:
        # Re-prepare and re-measure until we get 0
        q = qmc.qubit("q2")
        q = qmc.h(q)
        bit = qmc.measure(q)

    return bit


# %% [markdown]
# This transpiles to a Qiskit `while_loop` instruction:

# %%
exe = transpiler.transpile(repeat_until_zero)
job = exe.sample(executor, bindings={}, shots=100)
result = job.result()
for value, count in result.results:
    print(f"  bit={value}: {count} shots")

# %% [markdown]
# The loop keeps running until the measurement yields 0,
# so the final result is always 0.

# %% [markdown]
# ### Combining `if` and `while`
#
# You can combine both patterns. Here is a protocol that repeatedly
# measures and conditionally applies a correction gate:

# %%
@qmc.qkernel
def measure_and_correct() -> qmc.Bit:
    q0 = qmc.qubit("q0")
    q1 = qmc.qubit("q1")

    q0 = qmc.h(q0)
    bit = qmc.measure(q0)

    while bit:
        # If bit is 1, apply correction to q1
        if bit:
            q1 = qmc.x(q1)
        else:
            q1 = q1
        # Re-prepare and re-measure
        q0 = qmc.qubit("q0_retry")
        q0 = qmc.h(q0)
        bit = qmc.measure(q0)

    return qmc.measure(q1)


# %%
exe = transpiler.transpile(measure_and_correct)
job = exe.sample(executor, bindings={}, shots=100)
result = job.result()
for value, count in result.results:
    print(f"  bit={value}: {count} shots")

# %% [markdown]
# ## Summary
#
# - `qmc.range(n)` for looping over symbolic ranges.
# - `qmc.items(dict)` for iterating over sparse key-value data (edges, weights).
# - `if bit:` and `while bit:` for branching on **measurement results**.
#   Both branches must handle the same qubit handles (affine rule).
# - These control flow patterns transpile to native backend instructions
#   (e.g., Qiskit `if_else` and `while_loop`).
#
# **Next**: [Reuse Patterns](06_reuse_patterns.ipynb) — helper kernels,
# composite gates, and stub gates for top-down design.
