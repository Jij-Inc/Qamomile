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
# # Reusing quantum kernels
#
# When building complex quantum circuits, repeating the same gate operations makes code longer and harder to maintain.
# In Qamomile, you can define reusable operations as functions with `@qm.qkernel` and call them from another QKernel.
# Prepare a four-qubit GHZ state by calling a helper QKernel from another kernel.
# A helper QKernel is a quantum function that another kernel calls to reuse its operations.
# An optional example shows how to pass a kernel as an argument.

# %%
# Install the latest Qamomile from pip.
# # !pip install qamomile

# %%
import qamomile as qm

# %% [markdown]
# ## 1. Define a helper QKernel
#
# First, define `entangle_once` to apply a CX gate to two specified qubits.
# `state` is a four-qubit quantum register.
# `control` and `target` each take a single integer index: `control` selects the control qubit, and `target` selects the qubit on which X acts when the control is 1.
# Indices start at 0.


# %%
@qm.qkernel
def entangle_once(
    state: qm.Q[qm.Bits[4]],
    control: qm.Const[qm.Int],
    target: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[4]]:
    return qm.cx(state, control=control, target=target)


# %% [markdown]
# A quantum operation returns the updated quantum register.
# The caller must receive it, as in `state = entangle_once(...)`, and pass the updated `state` to the next operation.
# This lets you compose QKernels like ordinary functions.
#
# ## 2. Prepare a GHZ state with the helper
#
# The four-qubit GHZ state is written as:
#
# $$
# |\mathrm{GHZ}_4\rangle = \frac{|0000\rangle + |1111\rangle}{\sqrt{2}}
# $$
#
# To prepare this state, apply an H gate to the first qubit, then apply CX gates to successive pairs of adjacent qubits.
# The example below uses the `entangle_once()` helper defined above for each CX operation.


# %%
@qm.qkernel
def ghz_with_helper() -> qm.Shot[qm.Bits[4]]:
    state = qm.qubits(4)
    state = qm.h(state, target=0)
    for control in range(3):
        state = entangle_once(state, control, control + 1)
    return qm.measure(state)


# %% [markdown]
# The loop `for control in range(3)` applies CX gates in the order `(control, target) = (0, 1), (1, 2), (2, 3)`.
# Here, ordinary Python integers such as `0` and `1` can be passed as QKernel arguments.
# Finally, `qm.measure(state)` measures the register after the quantum operations.
# Let's run the program.

# %%
helper_program = qm.specialize(ghz_with_helper)
helper_result = qm.run(
    helper_program,
    on=qm.Reference(),
    shots=64,
    seed=7,
).result

expected_outcomes = {(0, 0, 0, 0), (1, 1, 1, 1)}

assert helper_result.shots == 64
assert sum(helper_result.counts.values()) == 64
assert set(helper_result.counts) <= expected_outcomes

print("Helper QKernel counts:", helper_result.counts)


# %% [markdown]
# `helper_result.counts` associates each observed bit tuple with the number of times it occurred.
# To look up a count, use `helper_result.counts.get((0, 0, 0, 0), 0)`, as you would with a dictionary.
# The counts cannot be changed.
# Measuring an ideal GHZ state in the computational basis produces only `(0, 0, 0, 0)` and `(1, 1, 1, 1)`, each with probability 1/2.
# The observed counts fluctuate, so 64 shots need not produce exactly 32 of each.
# Defining `entangle_once()` just once keeps repeated operations concise.
#
# ## 3. Pass a kernel as an argument (optional)
#
# A QKernel can also accept another QKernel as an argument.
# `qm.QKernel[A, B]` describes a kernel that accepts a value of type `A` and returns a value of type `B`.
# Here, `apply_twice(...)` applies the supplied one-qubit kernel twice.
# Passing a different compatible kernel changes the operation without changing this helper.


# %%
@qm.qkernel
def apply_twice(
    operation: qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]],
    state: qm.Q[qm.Bits[1]],
) -> qm.Q[qm.Bits[1]]:
    state = operation(state)
    return operation(state)


# %% [markdown]
# To try this helper, define a one-qubit kernel that applies X, then pass it to `apply_twice`.
# The qubit starts in `0`. X changes `0` to `1` and `1` to `0`, so two applications return it to `0`.


# %%
@qm.qkernel
def flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)


@qm.qkernel
def flip_twice() -> qm.Shot[qm.Bits[1]]:
    state = qm.qubits(1)
    state = apply_twice(flip, state)
    return qm.measure(state)


# %%
twice_result = qm.run(flip_twice, on=qm.Reference(), shots=16, seed=7).result

assert twice_result.shots == 16
assert twice_result.counts == {(0,): 16}

print("Two applications of X:", twice_result.counts)

# %% [markdown]
# With `flip` as the supplied kernel, `apply_twice` produces the measured result `(0,)` on every shot.
# The supplied kernel must accept and return the types declared by `operation`.
#
# For resource estimates of kernels without implementations, see [Resource Estimation](04_resource_estimation.ipynb).
#
# ## 4. Run the same kernels with Qiskit (optional)
#
# The first chapter uses Qiskit Aer. You can also run both kernels from this chapter with it.
# This optional section requires `qamomile[qiskit]`, which you can install with `pip install "qamomile[qiskit]"`.
# Keep the kernel definitions unchanged and choose `Aer()` as the execution endpoint, the simulator that runs the program.
#

# %%
from qamomile.qiskit import Aer  # noqa: E402 -- Optional SDK import.

qiskit_result = qm.run(helper_program, on=Aer(), shots=64, seed=7).result
qiskit_twice_result = qm.run(flip_twice, on=Aer(), shots=16, seed=7).result

assert qiskit_result.shots == 64
assert sum(qiskit_result.counts.values()) == 64
assert set(qiskit_result.counts) <= expected_outcomes
assert qiskit_twice_result.shots == 16
assert qiskit_twice_result.counts == {(0,): 16}

print("Qiskit GHZ counts:", qiskit_result.counts)
print("Qiskit X twice counts:", qiskit_twice_result.counts)

# %% [markdown]
# The kernel definitions, helper calls, and `qm.QKernel[...]` arguments are independent of Qiskit.
# Other supported simulators can run the same kernels through their own execution endpoints.
# The GHZ counts can differ between simulators because measurement is random. Check the possible outcomes and total shots rather than identical counts.
#
# ## Summary
#
# - Operations defined with `@qm.qkernel` can be called repeatedly from another QKernel.
# - After applying an operation to a quantum register, receive the updated `state` and pass it to the next operation.
# - Pass a kernel as an argument with `qm.QKernel[...]` to reuse a helper with different compatible operations.
#
# **Next**: [From a Hermitian Matrix to a Quantum Circuit](08_hermitian_decomposition.ipynb) shows how to decompose a Hermitian matrix given as a NumPy array into Pauli terms and use them to build a quantum circuit for time evolution.
