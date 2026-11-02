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
# # Parameterized Quantum Kernels
#
# In [*Your First Quantum Kernel*](01_your_first_quantum_kernel.ipynb), we defined quantum kernels with a fixed number of qubits. Qamomile also lets you describe kernels whose structure depends on symbolic parameters, such as the number of qubits or circuit layers. For example, you can write a kernel that applies Hadamard (H) gates to `width` qubits, or repeats a sequence of gates `layers` times.
#
# `Const` and `Param` indicate when you need to provide values for kernel inputs. `Const` values must be supplied by specialization, the stage that fixes those inputs in the kernel. When you pass a kernel directly to `qm.compile(...)`, specialization happens within that call, so you supply `Const` values as arguments to `qm.compile(...)`. You provide `Param` values prior to execution.
#
# :::{note}
# To specialize separately from compilation, pass the `Const` values as arguments to `qm.specialize(...)`. You can then pass the returned `Program` to `qm.compile(...)`.
# :::
#
# In the examples in this chapter, we declare the number of qubits as `Const` and the rotation angle as `Param`. `Shot` indicates that a value must be known by the time it is used within a single execution, called a shot.
#
# By keeping the rotation angle as a `Param`, we can compile the kernel for a chosen number of qubits and reuse the compiled artifact with different angles.
#
# This chapter covers:
#
# - The roles of `Const`, `Param`, and `Shot` in quantum kernels
# - Using `qm.qubits()` and `range()` to define parameterized circuits
# - Applying a single-qubit gate to every qubit in a state through **broadcasting**
# - Compiling once and reusing the compiled artifact with different parameter values

# %% [markdown]
# ## Setup

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
from __future__ import annotations

import os

import qamomile as qm

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
sample_shots = 1 if docs_test_mode else 128

# %% [markdown]
# ## Roles of `Const`, `Param`, and `Shot`
#
# `Const`, `Param`, and `Shot` indicate when a classical value must be available, rather than how it is obtained. The table below summarizes these requirements, with examples from this chapter.
#
# | Annotation | Value requirements and examples |
# |---|---|
# | `qm.Const[T]` | Inputs you specify by specialization, such as the number of qubits or the number of iterations |
# | `qm.Param[T]` | Inputs you specify prior to execution, such as rotation angles |
# | `qm.Shot[T]` | Values that must be known by the time they are used during a shot, such as measurement results |
#
# `T` is a placeholder for the underlying classical type, such as `qm.Int`, `qm.Real`, or `qm.Bits[4]`. For example, `qm.Const[qm.Int]` describes an integer input specified by specialization, while `qm.Bits[4]` describes a four-bit value.
#
# `Shot[T]` values need not be known before execution begins, but must be available when used within a shot. Typical examples include measurement results from `qm.measure(...)` and values computed from them. A value known before execution can also be used where `Shot[T]` is expected.
#
# In the examples below, `width` determines the number of qubits and loop iterations. In the execution example, we supply its value as an argument to `qm.compile(...)`. The rotation angle, `theta`, remains a `Param`, so we can choose a different value for each run.
#
# `Const` is not limited to values that determine the circuit’s structure; it can also be used for rotation angles. Declaring an angle as `qm.Const[qm.Real]` requires its value to be supplied by specialization, whereas `qm.Param[qm.Real]` lets you choose a value for each run.

# %% [markdown]
# ## `qm.qubits` and `range`
#
# Use `qm.qubits(width)` to create a quantum state containing `width` qubits. To apply gates to individual qubits, iterate over their indices with `range(width)`. The `target` argument specifies the index of the qubit to which the gate is applied.
#
# You can also use `range(start, stop, step)`. For example, `range(0, width, 2)` selects every other qubit, starting at index 0.
#
# The following kernel applies H and RY to each qubit, then measures the state:


# %%
@qm.qkernel
def rotation_layer(
    width: qm.Const[qm.Int],
    theta: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[width]]:  # noqa: F821
    state = qm.qubits(width)
    for target in range(width):
        state = qm.h(state, target=target)
        state = qm.ry(state, theta, target=target)
    return qm.measure(state)


# %% [markdown]
# :::{note}
# The ` # noqa: F821 ` comment suppresses a linter warning about `width` in the return annotation. It does not affect execution or Qamomile's type checking.
# :::
#
# You can analyze the kernel's logical resources before specifying a value for `width`. This kernel applies two gates per qubit, so its logical gate count is `2 * width`.
#
# Evaluating that expression at `width=4` gives eight logical gates. This evaluates the resource expression without specializing the kernel.

# %%
symbolic_study = qm.resources.logical(rotation_layer)
symbolic_gates = symbolic_study.metric(qm.resources.LOGICAL_GATES)

print("symbolic gates:", symbolic_gates.expression)
assert symbolic_gates.evaluate({"width": 4}).numeric_enclosure() == (8, 8)
assert symbolic_study.metric(qm.resources.LOGICAL_PEAK_QUBITS).evaluate(
    {"width": 4}
).numeric_enclosure() == (4, 4)

# %% [markdown]
# `numeric_enclosure()` returns the lower and upper bounds of the resource value. For the logical gate count, both bounds are 8, so it is exactly eight gates.

# %% [markdown]
# ## Applying Gates by Qubit Index
#
# In `state = qm.h(state, target=target)`, `qm.h` receives a value representing the quantum state and returns a value representing the state after the gate. The original input value cannot be reused in subsequent operations. Here, we receive the returned value under the same name, `state`, and pass it to the next operation.
#
# Qamomile does not support direct iteration over the qubits in `state`. The following example intentionally demonstrates this error:


# %%
@qm.qkernel
def bad_iteration(
    width: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[width]]:  # noqa: F821
    state = qm.qubits(width)
    for qubit in state:  # Direct iteration over qubits is not supported.
        qubit = qm.h(qubit)
    return state


# %%
try:
    qm.specialize(bad_iteration, width=4)
except qm.FrontendError as error:
    print(f"Error type: {type(error).__name__}")
    print(f"Error message: {error}")
    assert "for expects range" in str(error)

# %% [markdown]
# Instead, use `range(width)` to iterate over the qubit indices, as in `rotation_layer`, and pass the updated `state` to each successive gate.

# %% [markdown]
# ## Broadcasting Single-Qubit Gates Over a State
#
# For single-qubit gates such as H and RY, omitting `target=` applies the gate to every qubit in the state. This is called **gate broadcasting**.
#
# For example, `qm.h(state)` applies H to every qubit, and `qm.ry(state, theta)` applies RY to every qubit using the same angle. Use `target=` to select one qubit, or a loop to apply gates to a subset of qubits or use different angles for different qubits.


# %%
@qm.qkernel
def rotation_layer_broadcast(
    width: qm.Const[qm.Int],
    theta: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[width]]:  # noqa: F821
    state = qm.qubits(width)
    state = qm.h(state)
    state = qm.ry(state, theta)
    return qm.measure(state)


# %% [markdown]
# Both `rotation_layer` and `rotation_layer_broadcast` apply H followed by RY to each qubit. With `width=4`, both use four qubits and eight logical gates. The following checks confirm these resource counts for the broadcast version:

# %%
all_program = qm.specialize(rotation_layer_broadcast, width=4)
all_study = qm.resources.logical(all_program)

assert all_study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure() == (4, 4)
assert all_study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (
    8,
    8,
)

# %% [markdown]
# ## Compile Once, Run with Different Parameter Values
#
# A useful pattern for parameterized quantum kernels is to fix the circuit's structure once, then run it with different parameter values:
#
# 1. **Compile once**: fix `width` while keeping `theta` as a runtime parameter in the compiled artifact.
# 2. **Run with different values**: reuse the artifact and supply a new value for `theta` in each run.
#
# We use the built-in Reference simulator to run this example locally.

# %%
# Compile with width=4 while keeping theta as a runtime parameter.
endpoint = qm.Reference()
artifact = qm.compile(rotation_layer, to=endpoint.target, width=4)

parameter_names = tuple(slot.name for slot in artifact.parameter_abi.slots)
print("runtime parameters:", parameter_names)

assert parameter_names == ("theta",)

# %% [markdown]
# The compiled artifact retains `theta` as its only runtime parameter.
#
# The `width=4` argument fixes the number of qubits and loop iterations during the specialization performed by `qm.compile(...)`. A separate `qm.specialize(...)` call is not required.
#
# Next, we run the same artifact with three different angles:

# %%
# Run the same artifact with different theta values.
for theta in (0.1, 0.5, 1.0):
    record = qm.run(
        artifact,
        on=endpoint,
        parameters={"theta": theta},
        shots=sample_shots,
        seed=7,
    )
    result = record.result
    print(f"theta={theta:.1f} -> {result.counts}")
    assert record.artifact is artifact
    assert result.shots == sample_shots
    assert sum(result.counts.values()) == sample_shots
    # Measuring four qubits produces four-element tuples of classical bits.
    assert all(len(outcome) == 4 for outcome in result.counts)

# %% [markdown]
# The same compiled artifact is reused across all three runs. Only the value supplied through `parameters={"theta": theta}` changes.
#
# To recap:
#
# - **`width=4` in `qm.compile(...)`** fixes the number of qubits and loop iterations.
# - **`theta: qm.Param[qm.Real]`** declares `theta` as a real-valued runtime input. This example leaves the angle unbound during compilation and supplies its value for each run through `parameters={"theta": ...}` in `qm.run(...)`.
# - **`parameters={"theta": ...}` in `qm.run(...)`** supplies the angle for that run.
#
# Changing `theta` does not require recompilation in this example, but changing `width` requires specializing and compiling the kernel again.

# %% [markdown]
# ## Summary
#
# - `Const` and `Param` indicate when you provide values for kernel inputs: by specialization for `Const`, and prior to execution for `Param`. In this chapter, we used `Shot` for the measurement results returned by `qm.measure(...)`.
# - Use `qm.qubits(width)` to create a state with a parameterized number of qubits, and `range(width)` to iterate over their indices.
# - Each gate returns an updated state value. Subsequent operations use that returned value; the original input value cannot be reused.
# - Omit `target=` to broadcast a single-qubit gate across the state. Use a qubit index or a loop when selecting individual qubits.
# - Compile for a fixed circuit structure, then reuse the compiled artifact with different runtime parameter values.
#
# **Next**: [Controlled Operations with `qm.control`](03_controlled_gates.ipynb).
#
# To select qubits or split a qubit array, see the usage guide on
# [Working with Qubit Arrays](../usage/qubit_arrays.ipynb).
