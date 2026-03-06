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
# through `qmc.range`, `qmc.items`, and `if` branching.
#
# This chapter covers:
#
# - `qmc.range()` for loops (recap and deeper usage)
# - `qmc.items()` for iterating over dictionaries
# - `if` branching on kernel parameters
# - When `draw()` cannot handle a pattern and what to do instead

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
# ## `if` Branching
#
# You can use `if` statements on kernel parameters to conditionally apply
# different gates. Both branches must produce compatible return types.

# %%
@qmc.qkernel
def choose_basis(flag: qmc.UInt) -> qmc.Bit:
    q = qmc.qubit(name="q")

    if flag > 0:
        q = qmc.h(q)    # Hadamard basis
    else:
        q = qmc.x(q)    # Flip to |1>

    return qmc.measure(q)

# %% [markdown]
# ### Current `draw()` Limitations
#
# Some branch patterns are valid in kernels but not yet fully supported
# by `draw()`. If you encounter this, use `to_circuit()` or execute
# the kernel directly to verify behavior.

# %%
try:
    choose_basis.draw(flag=1)
except NotImplementedError as e:
    print(f"draw() limitation: {e}")
    print()
    print("Workaround: use to_circuit() to inspect the concrete circuit:")

# %%
print("flag=1 (H basis):")
print(transpiler.to_circuit(choose_basis, bindings={"flag": 1}))

print("flag=0 (X flip):")
print(transpiler.to_circuit(choose_basis, bindings={"flag": 0}))

# %% [markdown]
# ## Summary
#
# - `qmc.range(n)` for looping over symbolic ranges.
# - `qmc.items(dict)` for iterating over sparse key-value data (edges, weights).
# - `if` branching for conditional gate application.
# - When `draw()` hits a limitation, use `to_circuit()` with concrete bindings
#   to inspect the circuit, or simply execute the kernel to verify behavior.
#
# **Next**: [Reuse Patterns](06_reuse_patterns.ipynb) — helper kernels,
# composite gates, and stub gates for top-down design.
