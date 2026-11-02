# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
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
# # Running Quantum Programs and Estimating Expectation Values
#
# Once you have created a quantum program in Qamomile, the next step is to run it and inspect the results.
# This tutorial covers obtaining a single result, collecting repeated measurements, and estimating the expectation value of an observable.
# You will learn to:
#
# - Use `qm.evaluate(...)` to obtain the result of a single execution.
# - Use `qm.run(...)` to execute a quantum circuit repeatedly and read its measurement results.
# - Use `qm.statistics` to calculate means and standard errors from measurement results.
# - Use `Estimator` to estimate a Hamiltonian's expectation value.
#
# This tutorial uses Qamomile’s built-in simulator, `Reference`, for local execution.
# `qm.evaluate(...)` uses it automatically.
# For `qm.run(...)` and `estimator.estimate(...)`, we explicitly select it with `on=reference`.

# %%
from __future__ import annotations

import numpy as np

import qamomile as qm
from qamomile.measurement import Estimator

reference = qm.Reference()


# %% [markdown]
# ## 1. `qm.evaluate(...)`: Obtain a Single Result
#
# We start with a two-qubit circuit.
# `prepare_parity_state(...)` applies an H gate to qubit 0 and an RY gate with angle $\theta$ to qubit 1, followed by a CX gate with qubit 0 as the control and qubit 1 as the target.
# `parity_probe(...)` then measures both qubits in the computational basis.\
# The rotation angle `theta` is declared as `qm.Param[qm.Real]`, so we can specialize the kernel without binding its value.
# We supply that value to `qm.evaluate(...)` when running the program.
# The return annotation `qm.Shot[qm.Bits[2]]` indicates that each execution returns a two-bit measurement result.


# %%
@qm.qkernel
def prepare_parity_state(theta: qm.Param[qm.Real]) -> qm.Q[qm.Bits[2]]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qm.ry(state, theta, target=1)
    state = qm.cx(state, control=0, target=1)
    return state


@qm.qkernel
def parity_probe(theta: qm.Param[qm.Real]) -> qm.Shot[qm.Bits[2]]:
    return qm.measure(prepare_parity_state(theta))


theta = 0.7
parity_program = qm.specialize(parity_probe)
one_result = qm.evaluate(parity_program, theta=theta)
print("One-shot result:", one_result)


# %% [markdown]
# `qm.specialize(parity_probe)` prepares the program, and `qm.evaluate(parity_program, theta=theta)` executes it once.
# The result is a Python tuple containing two measured bits, with qubit 0 first and qubit 1 second.\
# Quantum measurements are probabilistic.
# However, `qm.evaluate(...)` uses `seed=0` by default, so this example produces the same result when rerun with unchanged inputs.
#
# ## 2. `qm.run(...)`: Execute a Quantum Circuit Repeatedly
#
# We now use `qm.run(...)` to execute the same `parity_program` 128 times, keeping $\theta=0.7$ radians.
# Each shot produces a two-bit measurement result.\
# `on=reference` selects the simulator, `parameters={"theta": theta}` supplies the rotation angle, and `shots=128` sets the number of executions.
# We set `seed=7` to make the simulation reproducible.


# %%
record = qm.run(
    parity_program,
    on=reference,
    parameters={"theta": theta},
    shots=128,
    seed=7,
)
samples = record.result
counts = samples.values

print("Shots:", samples.shots)
print("Outcome counts:", counts)
assert sum(counts.values()) == samples.shots

# %% [markdown]
# `samples.values` provides a read-only mapping, a dictionary-like object whose entries cannot be added, changed, or removed.
# We use it directly to display and sum the outcome counts.
# Each key is a two-bit tuple representing a measurement outcome, and each value is the number of times that outcome occurred.
# For example, `(0, 1): 7` means that qubit 0 was measured as 0 and qubit 1 as 1 in seven shots.
# `shots` specifies the number of measurement shots.
# We set `seed` for reproducibility, but measurement counts generally vary with the number of shots and the random draws.
#
# ## 3. Calculate an Expectation Value
#
# In quantum computing, we often estimate an observable’s expectation value by averaging repeated measurements.
# The `qm.statistics` module calculates statistics from results already collected, without running the quantum program again.
# Here, we use it to estimate $ZZ = Z_0 Z_1$.\
# In the computational (Z) basis, bit 0 corresponds to the Z eigenvalue +1, and bit 1 to −1. Their product is +1 for 00 and 11, and −1 for 01 and 10.\
# This product encodes the parity of the outcome: +1 for an even number of 1s and −1 for an odd number.
# `qm.statistics.parity_mean(samples, indices=(0, 1))` computes this average from the outcome counts.
# Since these values are the ZZ measurement outcomes, the average estimates $\langle ZZ \rangle$.
# The function also reports the standard error of the estimate.

# %%
parity = qm.statistics.parity_mean(samples, indices=(0, 1))

print("Estimated ZZ:", parity.value)
print("Standard error:", parity.standard_error)
print("Theoretical value (cos(theta)):", np.cos(theta))


# %% [markdown]
# In the computational (Z) basis, with qubit 0 written first, the circuit prepares
#
# $$
# \lvert \psi \rangle
# = \frac{\cos (\theta / 2)}{\sqrt{2} } (\lvert 00 \rangle + \lvert 11 \rangle) + \frac{\sin (\theta / 2)}{\sqrt{2} } (\lvert 01 \rangle + \lvert 10 \rangle) \tag{1}
# $$
#
# Matching bits occur with probability $\cos^2(\theta/2)$, and different bits with probability $\sin^2(\theta/2)$.
# Their contributions to ZZ are +1 and -1, respectively, so
#
# $$
# \langle ZZ \rangle
# = \cos^2 (\theta / 2) - \sin^2 (\theta / 2)
# = \cos \theta \tag{2}
# $$
#
# A finite-shot estimate may differ from the theoretical value.
# The standard error describes the sampling uncertainty in the estimate.
# More shots generally reduce this uncertainty.
#
# :::{note}
# `qm.statistics` summarizes measurement results that have already been collected.
# Since these measurements were taken in the Z basis, they can be used to estimate ZZ. You cannot calculate an X expectation value from the same Z-basis measurement results.
# :::
#
#
# ## 4. `Estimator`: Estimate a Hamiltonian's Expectation Value
#
# `Estimator` is useful when a Hamiltonian contains terms, such as Z and X, that require different measurement bases.
#
# Pass two components to `Estimator`:
#
# 1. State preparation: a QKernel that prepares a quantum state.
# 2. Observable: the Hamiltonian whose expectation value you want to estimate.
#
# Define a state preparation that rotates one qubit around the Y axis.
# This function returns a quantum state without measuring it. It serves as a state-preparation definition for `Estimator`, rather than a standalone measurement program.


# %%
@qm.qkernel
def rotated_state(theta: qm.Param[qm.Real]) -> qm.Q[qm.Bits[1]]:
    return qm.ry(qm.qubits(1), theta)


preparation = rotated_state.specialize()

# %% [markdown]
# Next, define the Hamiltonian whose expectation value you want to estimate:
#
# $$
# H
# = Z_0 + 0.5 X_0 \tag{3}
# $$
#
# $Z_0$ and $X_0$ are Pauli operators acting on qubit 0.
# Write them as `qm.Z(0)` and `qm.X(0)`, respectively.\
# For each shot, `Estimator` uses the state-preparation kernel to prepare a fresh quantum state.
# It measures some shots in the Z basis and others in the X basis, then combines the estimates as $\langle Z_0 \rangle + 0.5 \langle X_0 \rangle$.

# %%
hamiltonian = qm.Z(0) + 0.5 * qm.X(0)
estimator = Estimator(preparation, hamiltonian, grouping="qwc")

energy = estimator.estimate(
    on=reference,
    parameters={"theta": theta},
    shots=128,
    seed=11,
)

expected = np.cos(theta) + 0.5 * np.sin(theta)

assert energy.shots == 128
assert energy.standard_error is not None
assert abs(energy.value - expected) <= max(6 * energy.standard_error, 0.02)

print("Estimated expectation value:", energy.value)
print("Standard error:", energy.standard_error)
print("Total shots:", energy.shots)
print("Theoretical value:", expected)

# %% [markdown]
# `energy.value` is the estimated expectation value, `energy.standard_error` is its sampling standard error, and `energy.shots` is the total number of shots used.
#
# With grouping="qwc", terms that use compatible measurement bases on each qubit are grouped together.
# Here, $Z_0$ and $X_0$ require separate groups, so the total budget of 128 shots is split evenly: 64 for Z and 64 for X.
# For this state preparation, the theoretical value is
#
# $$
# \langle H \rangle
# = \cos(\theta) + 0.5\sin(\theta) \tag{4}
# $$
#
# The estimate may differ from the theoretical value because it uses a finite number of shots.
# More shots generally reduce the sampling uncertainty.\
# A QKernel describes the work within one execution.
# Results from multiple shots are combined outside the kernel: we used `qm.statistics` to summarize the samples returned by `qm.run(...)`, while Estimator manages the measurements and combines their results for us.
#
# :::{note}
# Applying a calculation before averaging can give a different result from applying it to the average.
# For equal numbers of +1 and −1, the mean of the squares is 1, whereas the square of the mean is 0.
# Both quantities can be computed in Python from the outcomes and counts in `SampleResult`.
# Per-shot arithmetic can also be written inside a QKernel when it forms part of each execution.
# :::

# %% [markdown]
# ## Summary
#
# | Goal | Interface |
# |---|---|
# | Execute once and obtain the result | `qm.evaluate(...)` |
# | Measure a quantum circuit repeatedly | `qm.run(...)` |
# | Calculate means and errors from collected samples | `qm.statistics` |
# | Estimate an expectation value from a state preparation and Hamiltonian | `Estimator` |
#
# Start by collecting measurement results with `qm.run(...)` and summarizing them with `qm.statistics`.
# Use `Estimator` when you want to automate the measurements required for multiple Pauli terms.
#
# **Next:** [Classical Control Flow](06_classical_flow_patterns.ipynb) introduces `for`, `if`, and conditional branching based on measurement results.

# %% [markdown]
#
