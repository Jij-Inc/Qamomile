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
# - `if` and `while` for conditional and iterative circuit construction
# - Current limitations of `draw()` and `to_circuit()` with control flow

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
# Python `if` and `while` statements work inside kernels.
# Conditions must be classical expressions (on kernel parameters),
# not quantum measurement results.
#
# Here is an example that combines both: build layers of rotation
# gates in a `while` loop, and optionally add entanglement with `if`.


# %%
@qmc.qkernel
def layered_circuit(
    n: qmc.UInt, theta: qmc.Float, depth: qmc.UInt, entangle: qmc.UInt
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")

    d = qmc.uint(0)
    while d < depth:
        for i in qmc.range(n):
            q[i] = qmc.ry(q[i], theta)
        if entangle > 0:
            for i in qmc.range(n - 1):
                q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])
        d = d + 1

    return qmc.measure(q)


# %% [markdown]
# `estimate_resources()` works with `if` and `while`:

# %%
est = layered_circuit.estimate_resources()
print("qubits:", est.qubits)
print("total gates:", est.gates.total)

# %% [markdown]
# ### Current `draw()` and `to_circuit()` Limitations
#
# `if` and `while` are valid in kernels, but `draw()` and `to_circuit()`
# do not yet fully support them. If you encounter an error, use
# `estimate_resources()` to verify the circuit structure.

# %%
try:
    layered_circuit.draw(n=3, theta=0.5, depth=2, entangle=1)
except Exception as e:
    print(f"draw() limitation: {type(e).__name__}: {e}")

# %% [markdown]
# ## Summary
#
# - `qmc.range(n)` for looping over symbolic ranges.
# - `qmc.items(dict)` for iterating over sparse key-value data (edges, weights).
# - `if` and `while` for conditional and iterative gate application.
# - `if`/`while` work in kernel definitions and `estimate_resources()`,
#   but `draw()` and `to_circuit()` support is still limited.
#
# **Next**: [Reuse Patterns](06_reuse_patterns.ipynb) — helper kernels,
# composite gates, and stub gates for top-down design.
