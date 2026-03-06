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
# # Resource Estimation
#
# Before running a quantum circuit on real hardware, you want to know its
# cost: how many qubits, how many gates, and what kinds of gates.
# Qamomile's `estimate_resources()` answers these questions **without
# executing the circuit** — and it works with both concrete and symbolic
# (parameterized) kernels.
#
# This chapter covers:
#
# - Basic resource estimation for fixed kernels
# - Symbolic estimation for parameterized kernels
# - The full `ResourceEstimate` field reference
# - `.substitute()` and `.simplify()` for working with symbolic expressions
# - Comparing design candidates by their resource costs

# %%
import qamomile.circuit as qmc

# %% [markdown]
# ## Estimating a Fixed Kernel
#
# For a kernel with no parameters, `estimate_resources()` returns
# concrete numbers.


# %%
@qmc.qkernel
def fixed_circuit() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(3, name="q")

    q[0] = qmc.h(q[0])
    q[0], q[1] = qmc.cx(q[0], q[1])
    q[1], q[2] = qmc.cx(q[1], q[2])

    return qmc.measure(q)


# %%
fixed_circuit.draw()

# %%
est = fixed_circuit.estimate_resources()
print("qubits:", est.qubits)
print("total gates:", est.gates.total)
print("single-qubit gates:", est.gates.single_qubit)
print("two-qubit gates:", est.gates.two_qubit)

# %% [markdown]
# ## Symbolic Estimation
#
# When a kernel has unbound parameters (like `n: qmc.UInt`),
# `estimate_resources()` returns **SymPy expressions** that show how
# costs scale with the parameter. This lets you analyze scaling
# without picking a specific value.


# %%
@qmc.qkernel
def scalable_circuit(n: qmc.UInt, theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")

    for i in qmc.range(n):
        q[i] = qmc.h(q[i])
        q[i] = qmc.ry(q[i], theta)

    for i in qmc.range(n - 1):
        q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])

    return qmc.measure(q)


# %%
scalable_circuit.draw(n=4, fold_loops=False)

# %%
est = scalable_circuit.estimate_resources()
print("qubits:", est.qubits)
print("total gates:", est.gates.total)
print("single-qubit gates:", est.gates.single_qubit)
print("two-qubit gates:", est.gates.two_qubit)
print("rotation gates:", est.gates.rotation_gates)
print("parameters:", est.parameters)

# %% [markdown]
# The output contains SymPy expressions like `n` for qubits and `3*n - 1`
# for total gates. These are exact — not approximations.

# %% [markdown]
# ## `ResourceEstimate` Fields Reference
#
# | Field | Description |
# |-------|------------|
# | `est.qubits` | Logical qubit count |
# | `est.gates.total` | Total gate count |
# | `est.gates.single_qubit` | Single-qubit gates |
# | `est.gates.two_qubit` | Two-qubit gates |
# | `est.gates.multi_qubit` | Multi-qubit gates (3+ qubits) |
# | `est.gates.t_gates` | T-gate count |
# | `est.gates.clifford_gates` | Clifford gate count |
# | `est.gates.rotation_gates` | Rotation gate count |
# | `est.gates.oracle_calls` | Oracle call counts (dict by name) |
# | `est.parameters` | Dict of symbol names → SymPy symbols |
#
# All fields are SymPy expressions. For fixed kernels they evaluate to
# plain integers.

# %% [markdown]
# ## Scaling Analysis with `.substitute()`
#
# The symbolic expressions tell you the *formula*, but often you want
# concrete numbers at specific sizes. Use `.substitute()` to evaluate:

# %%
for n_val in [4, 8, 16, 32]:
    c = est.substitute(n=n_val)
    print(
        f"n={n_val:2d}: {int(c.gates.total):>3} gates total, {int(c.gates.two_qubit):>2} two-qubit"
    )

# %% [markdown]
# This lets you check whether a circuit is feasible at your target scale
# before committing to execution.

# %% [markdown]
# ## Summary
#
# - `estimate_resources()` reports qubit and gate costs without executing.
# - For parameterized kernels, results are SymPy expressions showing exact scaling.
# - Use `.substitute(n=...)` to evaluate at specific sizes and check feasibility.
#
# **Next**: [Execution Models](04_execution_models.ipynb) — `sample()` vs `run()`,
# observables, and bit ordering.
