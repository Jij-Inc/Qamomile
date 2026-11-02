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
# # Using Python Control Flow
#
# When designing computations with quantum circuits, you may need to apply the same gate to several qubits or choose operations based on a condition.
# Qamomile's `@qm.qkernel` lets you express these operations with familiar Python constructs such as `for` and `if`.
# This tutorial focuses on three patterns:
#
# * Apply quantum gates repeatedly with `for` and `range()`.
# * Choose which operations to apply using `if` and runtime parameters.
# * Select the next operation based on a qubit's measurement result.
#
# We will also briefly introduce `break` and `continue` at the end.

# %%
# Install the latest Qamomile from pip.
# # !pip install qamomile

# %%
from __future__ import annotations

import qamomile as qm

# %% [markdown]
# ## Repeating Quantum Gates with `for`
#
# First, write a program that applies gates in sequence to a register with a specified number of qubits.
# This example applies a Hadamard gate $H$ to each even-indexed qubit, then applies CX gates to adjacent pairs of qubits.
# Define the following `qm.qkernel`:


# %%
@qm.qkernel
def hadamard_chain(
    width: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[width]]:  # noqa: F821
    state = qm.qubits(width)
    for target in range(0, width, 2):
        state = qm.h(state, target=target)
    for control in range(width - 1):
        state = qm.cx(state, control=control, target=control + 1)
    return qm.measure(state)


# %% [markdown]
# The loop `for target in range(0, width, 2)` selects the even-indexed qubits: 0, 2, 4, and so on.
# The next loop, `for control in range(width-1)`, selects control qubits from 0 through `width-2` in order.
# Assign the result of each operation back to the same variable, as in `state = qm.h(state, ...)`, and pass it to the next gate.
# This is the basic pattern for working with quantum states in Qamomile.
# The `Const` in `width: qm.Const[qm.Int]` marks a value that determines the circuit structure, such as the number of qubits.
# In the following example, `qm.specialize` specializes the program with `width=5`.

# %%
chain_program = qm.specialize(hadamard_chain, width=5)
chain_result = qm.run(
    chain_program,
    on=qm.Reference(),
    shots=32,
    seed=5,
).result

print("Number of shots:", chain_result.shots)
print("Measurement counts:", chain_result.counts)

assert chain_result.width == 5
assert chain_result.shots == 32


# %% [markdown]
# With `width=5`, the $H$ gates act on qubits `0, 2, 4`, and the CX gates act on `0→1`, `1→2`, `2→3`, and `3→4`.
# The measurement results are stored in `counts`, which shows how many times each bit string was measured.
# Because the program samples multiple measurements, the counts can vary with the random draws and execution environment.
#
# ## Switching Gates with `if`
#
# Next, use `if` to choose whether to apply an X gate.
# `enabled` is declared as `qm.Param[qm.Bool]`, so you must supply a Boolean value before execution.
# You can reuse the same program with a different value for each run.


# %%
@qm.qkernel
def choose(enabled: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    state = qm.qubits(1)
    if enabled:
        state = qm.x(state)
    return qm.measure(state)


# %% [markdown]
# Here, `qm.evaluate(program, ...)` executes one shot using Qamomile’s built-in Reference simulator and returns the result directly as a Python value.
# Run `choice_program` with `enabled=False`, then with `enabled=True`.

# %%
choice_program = qm.specialize(choose)

disabled_result = qm.evaluate(choice_program, enabled=False)
enabled_result = qm.evaluate(choice_program, enabled=True)

print("enabled=False:", disabled_result)
print("enabled=True: ", enabled_result)

assert disabled_result == (0,)
assert enabled_result == (1,)


# %% [markdown]
# The qubit starts in $\vert 0 \rangle$. Without the X gate (`False`), it stays in $\vert 0 \rangle$; with the X gate (`True`), it becomes $\vert 1 \rangle$.
# The bit strings returned by `qm.evaluate` are therefore `(0,)` and `(1,)`, respectively.
#
# Qamomile uses `Const` and `Param` for different purposes:
#
# | Type | When you must supply the value | Example in this notebook |
# | --- | --- | --- |
# | `Const` | By the time the kernel is specialized | `width` (number of qubits) |
# | `Param` | Before execution | `enabled` (whether to apply X) |
#
#
# ## Choosing the Next Operation from a Measurement Result
#
# Inside a quantum kernel, you can also use a qubit’s measurement result as an `if` condition.
# In this example, `qm.prepare(True)` prepares the first qubit in $\lvert 1 \rangle$, so measuring it returns `True`.
# `qm.prepare(False)` prepares the target qubit in $\lvert 0 \rangle$.
# If the first measurement returns `True`, an X gate is applied to the target.


# %%
@qm.qkernel
def conditional_flip() -> qm.Shot[qm.Bool]:
    condition = qm.measure(qm.prepare(True))
    target = qm.prepare(False)
    if condition:
        target = qm.x(target)
    return qm.measure(target)


# %%
feedback_program = qm.specialize(conditional_flip)
feedback_result = qm.evaluate(feedback_program)

print("Measurement feedback result:", feedback_result)

assert feedback_result is True


# %% [markdown]
# Since the first measurement returns `True`, the output is also `True`.
#
# The sequence is: measure, check the measurement result, then apply a gate if the condition is met.
# The value returned by `qm.measure(...)` becomes available during execution.
# Here, if uses that classical value to choose an operation.
# In contrast, `qm.control(...)` applies a quantum-controlled operation without measuring the control qubits.
#
# :::{note}
# Support for executing measurement-dependent branches varies by backend.
# This example uses `qm.evaluate(...)` with the Reference simulator.
# :::
#
# ## Additional Control: `continue` and `break`
#
# Use `continue` to skip the rest of a particular iteration of a `for` loop, or `break` to end the loop early.
# The following example uses no quantum gates, so you can focus on how these control statements work.
# The loop would normally iterate over `for index in range(limit)`, but `continue` skips the iteration where `index == 2`.
# Once the running sum of `index` values reaches 5 or more, `break` ends the loop.


# %%
@qm.qkernel
def sum_until(limit: qm.Const[qm.Int]) -> qm.Int:
    total = 0
    for index in range(limit):
        if index == 2:
            continue
        total += index
        if total >= 5:
            break
    return total


# %% [markdown]
# Use `qm.specialize` to fix `limit = 6`, then run the QKernel defined above.

# %%
summary_program = qm.specialize(sum_until, limit=6)
summary_result = qm.evaluate(summary_program)
print("Sum until threshold:", summary_result)

assert summary_result == 8

# %% [markdown]
# The loop reaches `break` when the sum becomes 0 + 1 + 3 + 4 = 8, so the result is 8.
# Even within a `for` loop with a predetermined upper bound, conditions can skip iterations or end the loop early.
#
# ## Summary
#
# - Use `for` and `range()` to apply gates repeatedly to multiple qubits.
# - `Const` input values must be specified by the time the kernel is specialized; `Param` input values are supplied before execution and can differ between runs.
# - Use `if` to choose quantum operations based on runtime parameters or measurement results.
# - Use `continue` and `break` to skip iterations or end a bounded loop early.
#
# **Next**: [Reuse Patterns](07_reuse_patterns.ipynb) shows how to extract common quantum circuits into separate QKernels and reuse them across programs.

# %% [markdown]
#
