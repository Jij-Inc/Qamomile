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
# # Parameterized Kernels
#
# In Tutorial 01 we built kernels with a fixed number of qubits.
# Real quantum programs often need to vary both the **circuit structure**
# (how many qubits, how many layers) and **runtime values** (rotation angles).
#
# This chapter teaches:
#
# - The typical roles of `UInt` and `Float` in kernel inputs
# - `qubit_array()` and `qmc.range()` for parameterized circuits
# - The **bind/sweep** pattern: compile once, execute many times

# %% [markdown]
# ## Typical Roles of `UInt` and `Float`
#
# Kernel parameters come in two flavors:
#
# | Type | Typical role |
# |------|-------------|
# | `qmc.UInt` | Circuit structure (qubit count, loop bound) |
# | `qmc.Float` | Continuous values (rotation angles, weights) |
#
# In practice, `UInt` values that control `qubit_array` size or `qmc.range`
# bounds **must** be bound at transpile time, because the backend needs a
# fixed circuit structure. `Float` values can stay as sweepable parameters.
#
# The common pattern is: bind structure at transpile time, sweep continuous
# values at execution time.

# %%
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## `qubit_array` and `qmc.range`
#
# When the number of qubits depends on a parameter `n`, use `qubit_array(n)`.
# To loop over the array, use `qmc.range(n)` instead of Python's built-in `range()`.
#
# > **Why not Python `range()`?** At trace time, `n` is a symbol, not a Python
# > integer — the kernel body is traced to build an IR, and Python's `range()`
# > cannot iterate over a symbol. `qmc.range()` emits a **loop node** in the IR
# > that the transpiler expands when `n` is bound to a concrete value.


# %%
@qmc.qkernel
def rotation_layer(n: qmc.UInt, theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")

    for i in qmc.range(n):
        q[i] = qmc.h(q[i])
        q[i] = qmc.ry(q[i], theta)

    return qmc.measure(q)


# %% [markdown]
# `draw()` accepts keyword arguments that bind parameters to concrete values
# for visualization. Here `n=4` fixes the circuit to 4 qubits and `theta=0.3`
# provides a placeholder angle.

# %%
rotation_layer.draw(n=4, theta=0.3, fold_loops=False)

# %% [markdown]
# ## Index-Based Updates
#
# Notice the pattern: `q[i] = qmc.h(q[i])`.
# You index into the array, apply the gate, and write the result back to the
# same index. This follows the affine rule from Tutorial 01 — the old handle
# at `q[i]` is consumed, and the returned handle takes its place.
#
# **Anti-pattern: iterating directly over the array.**
# Writing `for qi in q:` does not work because it bypasses the ownership
# tracking that Qamomile uses to enforce the affine rule.

# %%
try:

    @qmc.qkernel
    def bad_iteration(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
        q = qmc.qubit_array(n, name="q")
        for qi in q:  # Direct iteration — bad!
            qi = qmc.h(qi)
        return q

    bad_iteration.draw(n=4)
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")

# %% [markdown]
# Always use index-based access: `for i in qmc.range(n): q[i] = qmc.h(q[i])`.

# %% [markdown]
# ## The Bind/Sweep Pattern
#
# This is the central workflow pattern for parameterized kernels:
#
# 1. **Transpile once**: bind structure parameters (`n`) and declare runtime
#    parameters (`theta`) as sweepable.
# 2. **Execute many times**: reuse the same compiled executable with different
#    runtime bindings.
#
# This avoids recompiling the circuit for every parameter value.

# %%
# Transpile: fix n=4, keep theta as a sweepable parameter
exe = transpiler.transpile(
    rotation_layer,
    bindings={"n": 4},
    parameters=["theta"],
)

# Sweep: run the same executable with different theta values
for theta in [0.1, 0.5, 1.0]:
    result = exe.sample(
        transpiler.executor(),
        shots=128,
        bindings={"theta": theta},
    ).result()
    print(f"theta={theta:.1f} -> {result.results}")

# %% [markdown]
# The transpiled executable is reused across all three runs — only the runtime
# binding `{"theta": theta}` changes.
#
# To recap:
#
# - **`bindings={"n": 4}`** at transpile time: fixes the circuit structure.
# - **`parameters=["theta"]`** at transpile time: declares `theta` as sweepable.
# - **`bindings={"theta": ...}`** at execution time: provides the concrete value.

# %% [markdown]
# ## Summary
#
# - `qmc.UInt` values that control circuit structure (qubit count, loop bounds)
#   typically need to be bound at transpile time. `qmc.Float` values (rotation
#   angles) are natural candidates for runtime sweeping.
# - Use `qmc.qubit_array(n)` and `qmc.range(n)` for parameterized circuits.
#   Always use index-based updates: `q[i] = qmc.gate(q[i])`.
# - The bind/sweep pattern — `transpile(bindings=..., parameters=...)` then loop —
#   compiles once and executes many times.
#
# **Next**: [Resource Estimation](03_resource_estimation.ipynb) — symbolic cost
# analysis, gate breakdowns, and comparing design candidates.
