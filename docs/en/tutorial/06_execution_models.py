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
# # Execution Models: `sample()` vs `run()`
#
# Qamomile provides two execution methods depending on what your qkernel returns:
#
# | Qkernel returns | Use | You get back |
# |----------------|-----|-------------|
# | `Bit`, `Vector[Bit]`, `tuple[Bit, ...]` | `sample()` | `SampleResult` — counted outcomes |
# | `Float` (from `expval`) | `run()` | `float` — expectation value |
#
# This chapter explains both methods and introduces **observables** for expectation-value computation.

# %%
# Install the latest Qamomile through pip!
# # !pip install "qamomile[qiskit,visualization]"

# %%
import math

import qamomile.circuit as qmc
import qamomile.observable as qmo
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## Multi-Qubit `sample()`
#
# In Tutorial 01 we sampled a single `Bit`. When a qkernel returns multiple bits, each outcome is a **tuple** of integer values (`0` or `1`).


# %%
@qmc.qkernel
def parity_probe(theta: qmc.Float) -> tuple[qmc.Bit, qmc.Bit]:
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")

    q0 = qmc.h(q0)
    q1 = qmc.ry(q1, theta)
    q0, q1 = qmc.cx(q0, q1)

    return qmc.measure(q0), qmc.measure(q1)


# %%
parity_probe.draw(theta=0.7)

# %%
exe_sample = transpiler.transpile(parity_probe, parameters=["theta"])
sample_result = exe_sample.sample(
    transpiler.executor(),
    shots=256,
    bindings={"theta": 0.7},
).result()

for outcome, count in sample_result.results:
    print(f"  outcome={outcome}, count={count}")
assert sample_result.shots == 256
assert sum(count for _, count in sample_result.results) == 256
# parity_probe returns tuple[Bit, Bit] -> each outcome is a 2-element tuple.
assert all(
    isinstance(outcome, tuple) and len(outcome) == 2
    for outcome, _ in sample_result.results
)

# %% [markdown]
# Each `outcome` is a tuple like `(0, 1)` or `(1, 0)`. The first element corresponds to `q0`, the second to `q1`, matching the order in the `return` statement.

# %% [markdown]
# ## Measurement Result Ordering
#
# When Qamomile measures qubits, each outcome returned by `sample()` is a tuple of bits in the order specified by the qkernel's `return` statement. When measuring `qs: Vector[Qubit]` directly with `measure(qs)`, each outcome is ordered as `bits = (measure(qs[0]), measure(qs[1]), measure(qs[2]), ...)`. When interpreting a bit tuple as one bitstring, treat `bits[0]` as the least-significant bit.

# %% [markdown]
# ## When You Need Expectation Values
#
# Sometimes you don't want individual measurement outcomes — you want the **average value** of a quantum observable. This is common in:
#
# - **VQE** (Variational Quantum Eigensolver): minimize $\langle \psi \rvert H \lvert \psi \rangle$
# - **QAOA**: evaluate cost function expectation values
# - Any optimization loop over quantum parameters
#
# For this, Qamomile provides `expval()` and the `run()` execution method.

# %% [markdown]
# ## The Observable Type
#
# There are two related things to understand:
#
# 1. **`qmc.Observable`** — a **handle type** used in qkernel signatures.
#    Like `qmc.Float`, you use it in type annotations for qkernel arguments and return values.
#
# 2. **`qamomile.observable` module** — where you build **concrete** observable values
#    that you pass via bindings. For example:
#
# ```python
# import qamomile.observable as qmo
#
# H = qmo.Z(0)                        # Pauli Z on qubit 0
# H = qmo.Z(0) * qmo.Z(1)             # ZZ interaction
# H = 0.5 * qmo.X(0) + 0.3 * qmo.Y(1) # Linear combination
# ```

# %% [markdown]
# ## `expval()`: Measuring an Observable
#
# `expval(qubit, hamiltonian)` computes the expectation value $\langle \psi \rvert H \lvert \psi \rangle$ ( $\lvert \psi \rangle$ represents `qubit` and $H$ represents `hamiltonian` ) and returns a `qmc.Float`. A qkernel that returns `Float` from `expval` should be executed with `run()`.


# %%
@qmc.qkernel
def z_expectation(theta: qmc.Float, hamiltonian: qmc.Observable) -> qmc.Float:
    q = qmc.qubit(name="q")
    q = qmc.ry(q, theta)
    return qmc.expval(q, hamiltonian)


H = qmo.Z(0)

# %%
z_expectation.draw(theta=0.7, hamiltonian=H)

# %% [markdown]
# ## Running with `run()`
#
# For expectation-value qkernels, use `run()` instead of `sample()`. The observable is bound at transpile time (it affects the measurement circuit), while `theta` remains a sweepable runtime parameter.

# %%
exe_run = transpiler.transpile(
    z_expectation,
    bindings={"hamiltonian": H},  # Observable bound at transpile time
    parameters=["theta"],  # theta remains sweepable
)

run_result = exe_run.run(
    transpiler.executor(),
    bindings={"theta": 0.7},
).result()

print("expectation value:", run_result)
print("python type:", type(run_result))
assert isinstance(run_result, float)
# Ry(theta)|0> has <Z> = cos(theta) exactly; the statevector estimator
# returns it to floating-point precision.
assert math.isclose(run_result, math.cos(0.7), abs_tol=1e-10)

# %% [markdown]
# `run().result()` returns a plain `float` — the estimated $\langle \psi \rvert Z \lvert \psi \rangle$ value. For $\theta = 0.7$, the RY gate rotates the qubit as
# $$
# R_Y(\theta) \lvert 0 \rangle = \cos\left( \frac{\theta}{2} \right) \lvert 0 \rangle + \sin\left( \frac{\theta}{2} \right) \lvert 1 \rangle
# $$
# so the Z expectation is $\cos^2\left( \frac{\theta}{2} \right) - \sin^2\left( \frac{\theta}{2} \right) = \cos(\theta) \approx 0.765$.

# %% [markdown]
# ## `sample()` vs `run()`
#
# | Qkernel returns | Execution method | `.result()` returns |
# |----------------|-----------------|-------------------|
# | `Bit` | `sample()` | `SampleResult` with `.results: list[tuple[int, int]]` |
# | `tuple[Bit, Bit]` | `sample()` | `SampleResult` with `.results: list[tuple[tuple[int, int], int]]` |
# | `Vector[Bit]` | `sample()` | `SampleResult` with `.results: list[tuple[tuple[int, ...], int]]` |
# | `Float` (from `expval`) | `run()` | `float` |
#
# **Rule of thumb**: if your qkernel ends with `measure()`, use `sample()`. If it ends with `expval()`, use `run()`.

# %% [markdown]
# ## Summary
#
# - `sample()` is for qkernels returning measured bits — you get a distribution
#   of outcomes with counts.
# - `run()` is for qkernels returning `Float` via `expval()` — you get a single
#   expectation value.
# - `qmc.Observable` is the handle type; `qamomile.observable.Z(0)` etc. are
#   the concrete values. Bind observables at transpile time.
# - Bit ordering is big-endian: position in the return tuple matches qubit order.
#
# **Next**: [Classical Flow Patterns](07_classical_flow_patterns.ipynb) — loops with `qmc.range`, sparse data with `qmc.items`, and conditional branching.
