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
# # Debugging and Backend Workflow
#
# This chapter is a practical reference for when things go wrong — or when
# you want to make sure things go right before scaling up. It covers:
#
# - A step-by-step debug workflow
# - Common error messages and what they mean
# - The transpile-once, sweep-many pattern as a workflow recommendation
# - A quick reference card of all types, gates, and key methods

# %%
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## The Debug Mindset
#
# When a kernel doesn't work as expected, follow this sequence:
#
# 1. **Draw** — visualize the circuit to catch structural mistakes.
# 2. **Check parameter names** — make sure runtime bindings match.
# 3. **Inspect with `to_circuit()`** — see exactly what the backend receives.
# 4. **Start small** — use small `n` and low `shots` to iterate quickly.
# 5. **Estimate before executing** — compare design candidates cheaply.
#
# We will walk through each step with a concrete example.

# %% [markdown]
# ## Step 1: Draw the Circuit
#
# Always start by drawing. This catches obvious mistakes like missing gates
# or wrong qubit connections.

# %%
@qmc.qkernel
def layered_rotation(n: qmc.UInt, theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")

    for i in qmc.range(n):
        q[i] = qmc.h(q[i])
        q[i] = qmc.ry(q[i], theta)

    return qmc.measure(q)

# %%
layered_rotation.draw(n=3, theta=0.3)

# %% [markdown]
# ## Step 2: Check Parameter Names
#
# After transpiling, inspect `parameter_names` to confirm which runtime
# parameters the executable expects.

# %%
exe = transpiler.transpile(
    layered_rotation,
    bindings={"n": 3},
    parameters=["theta"],
)

print("parameter_names:", exe.parameter_names)
print("has_parameters:", exe.has_parameters)

# %% [markdown]
# If `parameter_names` doesn't match what you expect, check your
# `transpile(parameters=[...])` call.

# %% [markdown]
# ## Step 3: Inspect with `to_circuit()`
#
# `to_circuit()` compiles with **all** parameters bound and returns the
# backend-native circuit. This shows exactly what the backend will execute.

# %%
circuit = transpiler.to_circuit(
    layered_rotation,
    bindings={"n": 3, "theta": 0.6},
)
print(circuit)

# %% [markdown]
# ## Common Error Messages
#
# Here are the most common mistakes and the errors they produce.

# %% [markdown]
# ### Wrong binding key name
#
# If you misspell a parameter name in `bindings`, you get an error:

# %%
try:
    exe.sample(
        transpiler.executor(),
        shots=64,
        bindings={"thetaa": 0.6},  # Typo: "thetaa" instead of "theta"
    ).result()
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")

# %% [markdown]
# **Fix**: Check `exe.parameter_names` and make sure your binding key matches exactly.

# %% [markdown]
# ### Missing required binding
#
# If `to_circuit()` is called without all parameters, it fails:

# %%
try:
    transpiler.to_circuit(layered_rotation, bindings={"n": 3})
    # theta is missing — to_circuit needs ALL parameters bound
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")

# %% [markdown]
# **Fix**: Provide all parameter values when using `to_circuit()`.

# %% [markdown]
# ### Forgotten rebind (stale handle)
#
# Forgetting `q = qmc.gate(q)` — the most common beginner mistake:

# %%
try:

    @qmc.qkernel
    def bad_rebind() -> qmc.Bit:
        q = qmc.qubit(name="q")
        qmc.h(q)           # Consumed q, but didn't rebind!
        q = qmc.x(q)       # Using stale handle
        return qmc.measure(q)

    bad_rebind.draw()
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")

# %% [markdown]
# **Fix**: Always write `q = qmc.h(q)`, not `qmc.h(q)`.

# %% [markdown]
# ### Ignored tuple return from two-qubit gate

# %%
try:

    @qmc.qkernel
    def bad_cx() -> tuple[qmc.Bit, qmc.Bit]:
        q0 = qmc.qubit(name="q0")
        q1 = qmc.qubit(name="q1")
        q0 = qmc.h(q0)
        qmc.cx(q0, q1)     # Both outputs ignored!
        return qmc.measure(q0), qmc.measure(q1)

    bad_cx.draw()
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")

# %% [markdown]
# **Fix**: Always unpack both handles: `q0, q1 = qmc.cx(q0, q1)`.

# %% [markdown]
# ## Transpile-Once, Sweep-Many
#
# As a workflow recommendation: if you are sweeping over parameter values,
# always transpile once and reuse the executable. This avoids redundant
# compilation.

# %%
for theta in [0.2, 0.6, 1.0]:
    result = exe.sample(
        transpiler.executor(),
        shots=128,
        bindings={"theta": theta},
    ).result()
    print(f"theta={theta:.1f} -> {result.results}")

# %% [markdown]
# ## Quick Reference
#
# ### Handle Types
#
# | Type | Description | Constructor |
# |------|------------|-------------|
# | `qmc.Qubit` | Single qubit handle | `qmc.qubit(name="q")` |
# | `qmc.Vector[qmc.Qubit]` | Qubit array | `qmc.qubit_array(n, name="q")` |
# | `qmc.Float` | Floating-point parameter | Kernel argument type |
# | `qmc.UInt` | Unsigned integer parameter | Kernel argument type |
# | `qmc.Bit` | Classical bit (measurement result) | From `qmc.measure()` |
# | `qmc.Vector[qmc.Bit]` | Bit array | From `qmc.measure(array)` |
# | `qmc.Observable` | Observable handle | Kernel argument type |
#
# ### Single-Qubit Gates
#
# | Gate | Signature | Description |
# |------|----------|-------------|
# | `h` | `(Qubit) -> Qubit` | Hadamard |
# | `x`, `y`, `z` | `(Qubit) -> Qubit` | Pauli gates |
# | `t`, `tdg` | `(Qubit) -> Qubit` | T and T-dagger |
# | `s`, `sdg` | `(Qubit) -> Qubit` | S and S-dagger |
# | `rx`, `ry`, `rz` | `(Qubit, Float) -> Qubit` | Rotation gates |
# | `p` | `(Qubit, Float) -> Qubit` | Phase gate |
#
# ### Two-Qubit Gates
#
# | Gate | Signature | Description |
# |------|----------|-------------|
# | `cx` | `(Qubit, Qubit) -> (Qubit, Qubit)` | CNOT |
# | `cz` | `(Qubit, Qubit) -> (Qubit, Qubit)` | Controlled-Z |
# | `cp` | `(Qubit, Qubit, Float) -> (Qubit, Qubit)` | Controlled-Phase |
# | `rzz` | `(Qubit, Qubit, Float) -> (Qubit, Qubit)` | RZZ interaction |
# | `swap` | `(Qubit, Qubit) -> (Qubit, Qubit)` | SWAP |
#
# ### Three-Qubit Gates
#
# | Gate | Signature | Description |
# |------|----------|-------------|
# | `ccx` | `(Qubit, Qubit, Qubit) -> (Qubit, Qubit, Qubit)` | Toffoli |
#
# ### Key Methods
#
# | Method | Description |
# |--------|------------|
# | `kernel.draw(...)` | Visualize the circuit |
# | `kernel.estimate_resources(...)` | Symbolic resource estimation |
# | `transpiler.transpile(kernel, bindings, parameters)` | Compile to executable |
# | `transpiler.to_circuit(kernel, bindings)` | Get backend-native circuit |
# | `exe.sample(executor, shots, bindings)` | Execute and sample |
# | `exe.run(executor, bindings)` | Execute for expectation value |
# | `job.result()` | Get result from a Job |
# | `result.most_common(n)` | Top-n outcomes |
# | `result.probabilities()` | Outcome probabilities |
