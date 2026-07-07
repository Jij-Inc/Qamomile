# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# tags: [tutorial]
# ---
#
# # Classical Control Flow Patterns
#
# Quantum circuits often have structure that depends on classical control flow: iterating over qubits, applying gates based on a graph's edges, or choosing between gate sequences. Qamomile supports these patterns through `qmc.range`, `qmc.items`, `if` branching, and `while` loops.
#
# This chapter covers:
#
# - `qmc.range()` for loops
# - `qmc.items()` for iterating over dictionaries
# - `d[key]` subscript lookup on dictionaries
# - `if` and `while` on measurement results for mid-circuit branching

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import os

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## `qmc.range` Loops
#
# `qmc.range` may take `start`, `stop`, and `step` arguments. Here we create a qkernel that applies H to every other qubit and then entangles adjacent pairs with CX.


# %%
@qmc.qkernel
def hadamard_chain(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")

    # Apply H to every other qubit
    q[0::2] = qmc.h(q[0::2])

    # Entangle adjacent pairs
    for i in qmc.range(n - 1):
        q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])

    return qmc.measure(q)


# %%
hadamard_chain.draw(n=5, fold_loops=False)

# %% [markdown]
# :::{note}
# The loop variable in `qmc.range` must be a **single variable** (e.g. `for i in qmc.range(n)`).
# Tuple or list unpacking such as `for [i, j] in qmc.range(n)` is not supported and will raise a `SyntaxError`.
# :::

# %% [markdown]
# ## `qmc.items` for Sparse Interaction Data
#
# Many variational algorithms apply gates only on specific pairs of qubits, determined by a graph or interaction map. Rather than looping over all pairs, you can pass a **dictionary** of interactions and iterate with `qmc.items()`.
#
# The dictionary type uses Qamomile's symbolic types: `qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]` — keys are qubit index pairs, values are interaction weights.


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
# :::{note}
# `qmc.items` supports these loop patterns:
#
# - `for key, value in qmc.items(d)` — scalar key
# - `for (i, j), value in qmc.items(d)` — tuple key
# - `for key, value in d.items()` — method-call form
#
# The **value** target must be a single variable. Tuple unpacking in the value position
# (e.g. `for _, (i, j) in qmc.items(d)`) is **not** supported and will raise a `SyntaxError`.
# Similarly, single-target patterns like `for pair in qmc.items(d)` are not supported.
# :::

# %% [markdown]
# ## Dict Subscript Lookup (`d[key]`)
#
# Besides iterating with `qmc.items()`, a `qmc.Dict` can be indexed directly with `d[key]`. The most useful pattern is indexing **one dict with the iteration keys of another**: iterate over sparse interaction terms in one dict while looking up per-edge scale factors in a second dict.


# %%
@qmc.qkernel
def per_edge_angles(
    n: qmc.UInt,
    edges: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    gammas: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")

    for i in qmc.range(n):
        q[i] = qmc.h(q[i])

    # Each edge gets its own angle, looked up by the same (i, j) key
    for (i, j), weight in qmc.items(edges):
        q[i], q[j] = qmc.rzz(q[i], q[j], weight * gammas[(i, j)])

    return qmc.measure(q)


# %%
edge_data = {(0, 1): 1.0, (1, 2): -0.7}
gamma_data = {(0, 1): 0.3, (1, 2): 0.5}

circuit = transpiler.to_circuit(
    per_edge_angles,
    bindings={"n": 3, "edges": edge_data, "gammas": gamma_data},
)
_rzz_angles = sorted(
    float(_instr.operation.params[0])
    for _instr in circuit.data
    if _instr.operation.name == "rzz"
)
# Each RZZ angle is weight * gamma for its own edge.
assert _rzz_angles == sorted([1.0 * 0.3, -0.7 * 0.5])

# %% [markdown]
# :::{note}
# What `d[key]` supports:
#
# - **Keys**: a loop variable of an items loop (`gammas[i]`, `gammas[(i, j)]`), an `int` constant, or a mix of both (`gammas[(0, i)]`). When every key component is a compile-time constant and the dict data is bound, any hashable Python key (e.g. a `str`) also works — the lookup folds to a constant at trace time. Symbolic key components must be `UInt`.
# - **Value types**: scalar values (`qmc.Float`, `qmc.UInt`, `qmc.Bit`). Container values (`qmc.Tuple` / `qmc.Vector`) are not yet supported and raise `NotImplementedError`.
# - A missing key raises `KeyError` at build time, just like a Python dict.
# :::

# %% [markdown]
# ## Inspecting with `transpiler.to_circuit()`
#
# `draw()` does not yet support all patterns (particularly `items` with complex types, `if`, and `while`). In such cases, use `transpiler.to_circuit()` to see the concrete transpiled circuit after all parameters are bound.

# %%
edge_data = {(0, 1): 1.0, (1, 2): -0.7, (0, 2): 0.3}

circuit = transpiler.to_circuit(
    sparse_coupling,
    bindings={"n": 3, "edges": edge_data, "gamma": 0.4},
)
print(circuit)
assert circuit.num_qubits == 3
# n=3 -> 3 initial H + 3 measurements; len(edge_data)=3 -> exactly 3 RZZ.
_ops = {}
for _instr in circuit.data:
    _ops[_instr.operation.name] = _ops.get(_instr.operation.name, 0) + 1
assert _ops == {"h": 3, "rzz": 3, "measure": 3}

# %% [markdown]
# Only the three edges in `edge_data` produce RZZ gates — no wasted operations.

# %% [markdown]
# ## `if` Branching and `while` Loops
#
# Qamomile supports **mid-circuit measurement** followed by classical branching. The condition must be a **measurement result** (`Bit`), not an argument of qkernels.
#
# This maps directly to hardware-level conditional execution: measure a qubit, then decide what to do next based on the outcome.

# %% [markdown]
# ### `if` on a measurement result
#
# A common pattern: measure one qubit and conditionally apply a gate to another qubit based on the outcome.


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
        pass

    return qmc.measure(q1)


# %% [markdown]
# This transpiles to a Qiskit `if_else` instruction and can be executed:

# %%
exe = transpiler.transpile(conditional_flip)
if os.environ.get("QAMOMILE_DOCS_TEST") == "1":
    print("Skipping dynamic-circuit execution in docs test mode.")
else:
    executor = transpiler.executor()
    job = exe.sample(executor, bindings={}, shots=100)
    result = job.result()
    for value, count in result.results:
        print(f"  bit={value}: {count} shots")
    # q0 prepared as |1>; the if-branch flips q1 to |1> on every shot.
    assert result.shots == 100
    assert result.results == [(1, 100)]

# %% [markdown]
# Since `q0` is prepared as |1⟩, the measurement always yields 1, so `q1` always gets flipped — every shot should return 1.

# %% [markdown]
# ### `while` on a measurement result
#
# A `while` loop repeats until the measurement condition becomes false. This is useful for repeat-until-success protocols.


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
# This transpiles to a Qiskit `while_loop` instruction. We can inspect the generated circuit structure:

# %%
exe_while = transpiler.transpile(repeat_until_zero)
qc_while = exe_while.compiled_quantum[0].circuit
print(qc_while)
assert qc_while.num_qubits == 2
# The `while bit:` lowers to a Qiskit `while_loop` instruction.
assert "while_loop" in {instr.operation.name for instr in qc_while.data}

# %% [markdown]
# ### Combining `if` and `while`
#
# You can combine both patterns. Here is a protocol that repeatedly measures and conditionally applies a correction gate:


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
exe_combined = transpiler.transpile(measure_and_correct)
qc_combined = exe_combined.compiled_quantum[0].circuit
print(qc_combined)
assert qc_combined.num_qubits == 3
assert "while_loop" in {instr.operation.name for instr in qc_combined.data}

# %% [markdown]
# ## Summary
#
# - `qmc.range(n)` for looping over symbolic ranges.
# - `qmc.items(dict)` for iterating over sparse key-value data (edges, weights).
# - `d[key]` for looking up one dict by the iteration keys of another
#   (per-edge coefficients or calibration scales).
# - `if bit:` and `while bit:` for branching on **measurement results**.
#   Both branches must handle the same qubit handles (affine rule).
# - These control flow patterns transpile to native quantum SDK instructions
#   (e.g., Qiskit `if_else` and `while_loop`).
#
# **Next**: [Reuse Patterns](08_reuse_patterns.ipynb) — helper qkernels, composite gate callables, and opaque oracles for top-down design.
