# ---
# jupyter:
#   jupytext:
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
# tags: [usage]
# ---
#
# # Working with Qubit Arrays
#
# Qamomile represents an ordered group of qubits as a **qubit array**. You can
# apply gates to selected qubits, split an array to pass part of it to a quantum
# kernel, and combine the returned parts for subsequent operations. This lets
# you reuse the same quantum kernel on different parts of a larger array.
#
# This guide shows how to select qubits by index, split and recombine arrays,
# and change their logical order. It also shows how to apply a quantum kernel
# to qubits whose indices are not consecutive. Here, consecutiveness refers
# to the logical indices of the qubit array and does not imply physical
# adjacency of the qubits.
#
# Splitting and combining change how the program groups and indexes existing
# qubits. These operations add no quantum gates and preserve the physical
# quantum state, including entanglement between the parts.

# %% [markdown]
# ## Setup
#
# The examples in this guide run locally using Qamomile's built-in Reference
# simulator. Install Qamomile:

# %%
# # !pip install qamomile

# %%
from __future__ import annotations

import qamomile as qm
from qamomile.stdlib import on_qubits

endpoint = qm.Reference()

# %% [markdown]
# ## Apply Gates to Selected Qubits
#
# For a single-qubit gate such as `qm.x(...)`, the `target=` argument selects
# one qubit by its index in the array. Indices start at zero, so a six-qubit
# array has indices 0 through 5.
#
# ### Select one qubit
#
# `qm.qubits(6)` prepares six qubits in state $|0\rangle$. The X gate exchanges
# $|0\rangle$ and $|1\rangle$, so applying it with `target=5` flips the last
# qubit to $|1\rangle$.


# %%
@qm.qkernel
def flip_last() -> qm.Shot[qm.Bits[6]]:
    state = qm.qubits(6)
    state = qm.x(state, target=5)
    return qm.measure(state)


last_result = qm.run(flip_last, on=endpoint, shots=1, seed=7).result
print(last_result.values)
assert last_result.values == {(0, 0, 0, 0, 0, 1): 1}

# %% [markdown]
# `last_result.values` returns a read-only mapping from tuples of measured bits
# to their occurrence counts.
# The bits follow the array's index order; here, the single shot produces a 1
# only at index 5.
#
# ### Select several qubits
#
# Use a loop to apply the same gate to several selected qubits. You can use
# Python's `range(...)` inside a quantum kernel.


# %%
@qm.qkernel
def flip_even() -> qm.Shot[qm.Bits[6]]:
    state = qm.qubits(6)
    for index in range(0, 6, 2):
        state = qm.x(state, target=index)
    return qm.measure(state)


even_result = qm.run(flip_even, on=endpoint, shots=1, seed=7).result
print(even_result.values)
assert even_result.values == {(1, 0, 1, 0, 1, 0): 1}

# %% [markdown]
# ### Apply a gate to every qubit
#
# Single-qubit gates such as `qm.x(...)` support broadcasting. Omitting
# `target=` applies the gate to every qubit in the array.


# %%
# Qamomile resolves width in this dependent return annotation.
@qm.qkernel
def flip_every_qubit(width: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[width]]:  # noqa: F821
    state = qm.qubits(width)
    state = qm.x(state)
    return qm.measure(state)


# Fix the array length at four.
all_program = qm.specialize(flip_every_qubit, width=4)
all_result = qm.run(all_program, on=endpoint, shots=1, seed=7).result
print(all_result.values)
assert all_result.values == {(1, 1, 1, 1): 1}

# %% [markdown]
# These examples select qubits while keeping the array together. The next
# section shows how to split an array so that a quantum kernel can process
# just one part.

# %% [markdown]
# ## Apply a Quantum Kernel to Part of an Array
#
# Use `qm.split_at(...)` to split a qubit array into two arrays. Use
# `qm.concat(...)` to combine multiple qubit arrays into one larger array.
# Quantum kernels provided by the library can also be applied to part of a
# qubit array in the same way.
#
# Note that the following example calls one quantum kernel from another.


# %%
@qm.qkernel
def flip_three(state: qm.Q[qm.Bits[3]]) -> qm.Q[qm.Bits[3]]:
    return qm.x(state)


@qm.qkernel
def rejoin_halves() -> qm.Shot[qm.Bits[6]]:
    state = qm.qubits(6)
    # Split the six-qubit array into its first three and last three qubits.
    # front: original[0:3], back: original[3:6]
    front, back = qm.split_at(state, 3)

    front = flip_three(front)

    # Combine in the supplied order to restore the original index order.
    state = qm.concat(front, back)
    return qm.measure(state)


# %%
rejoined_result = qm.run(rejoin_halves, on=endpoint, shots=1, seed=7).result
print(rejoined_result.values)
assert rejoined_result.values == {(1, 1, 1, 0, 0, 0): 1}

# %% [markdown]
# Each part has its own indices starting at zero. For example, index 0 of
# `back` refers to index 3 of the original array.
#
# `qm.split_at(...)` consumes the original array value, so subsequent
# operations use the values it returns (`front` and `back` here). Similarly,
# `qm.concat(...)` consumes the supplied values and returns the combined
# array, received here again as `state`.
#
# The split must leave both parts nonempty. For an array of `n` qubits, choose
# a boundary with `0 < boundary < n`. `qm.concat(...)` accepts two or more
# arrays, including arrays prepared separately with `qm.qubits(...)`.

# %% [markdown]
# :::{note}
# **Current limitations:** These operations require concrete array lengths
# when Qamomile reads the quantum kernel source. `qm.split_at(...)` also
# requires a concrete split boundary. Quantum kernel arguments annotated
# with `Const` or `Param` cannot currently specify these lengths or the
# boundary; use concrete integers as shown above.
# :::

# %% [markdown]
# ## Split in Stages and Change the Order
#
# You can split either part of an array again to select a smaller range. Each
# split boundary counts from the beginning of the array being split.
#
# In the following example, three splits isolate the qubits at original
# indices 2 and 3. We apply a Hadamard (H) gate to each, then combine the split
# qubit arrays in a different order.


# %%
@qm.qkernel
def nested_split() -> qm.Shot[qm.Bits[6]]:
    state = qm.qubits(6)

    # Slice notation below refers to the original six-qubit array.
    prefix, remainder = qm.split_at(state, 1)
    # prefix: original[0:1], remainder: original[1:6]

    window, suffix = qm.split_at(remainder, 3)
    # window: original[1:4], suffix: original[4:6]

    first, inner = qm.split_at(window, 1)
    # first: original[1:2], inner: original[2:4]

    inner = qm.h(inner)

    window = qm.concat(first, inner)
    state = qm.concat(suffix, window, prefix)
    # Original qubit indices in the new order: [4, 5, 1, 2, 3, 0].

    return qm.measure(state)


# %%
nested_result = qm.run(nested_split, on=endpoint, shots=32, seed=7).result
print(nested_result.values)

assert sum(nested_result.values.values()) == 32
# The qubits that received H gates appear at positions 3 and 4 in each
# measured tuple; all other bits remain zero.
assert all(
    len(bits) == 6 and bits[0] == bits[1] == bits[2] == bits[5] == 0
    for bits in nested_result.values
)

# %% [markdown]
# For example, `qm.split_at(remainder, 3)` splits `remainder` into `window`,
# containing its first three qubits, and `suffix`, containing its remaining
# two qubits. Since `remainder` contains the last five qubits of the original
# array, `window` corresponds to original indices 1, 2, and 3. Splitting that
# `window` at 1 then separates original index 1 from indices 2 and 3.
#
# As a result, `qm.concat(suffix, window, prefix)` places the qubits in the
# order `[4, 5, 1, 2, 3, 0]`, expressed using the original array's indices.
# Subsequent gate indices and measurement results follow this new order. For
# example, `target=0` now selects index 4 of the original array.
#
# This reordering changes only the mapping between array indices and qubits.
# It therefore adds no SWAP gates and does not change the physical
# quantum state or entanglement. In this example, we eventually form `state`
# again with the same length as the original array, but you do not always need
# to restore the original length.

# %% [markdown]
# ## Apply a Quantum Kernel to Nonconsecutive Qubits
#
# `on_qubits(...)` applies a quantum kernel to selected qubits without writing
# the splits and recombinations yourself. The selected indices do not need
# to be consecutive.
#
# Here we apply the previously defined `flip_three` to indices 0, 2, and 5 of
# a six-qubit array. `width=6` specifies the total array length, and
# `indices=(0, 2, 5)` identifies the three qubits passed to `flip_three`.

# %%
flip_selected = on_qubits(flip_three, width=6, indices=(0, 2, 5))


@qm.qkernel
def nonconsecutive_demo() -> qm.Shot[qm.Bits[6]]:
    state = qm.qubits(6)
    state = flip_selected(state)
    return qm.measure(state)


selected_result = qm.run(nonconsecutive_demo, on=endpoint, shots=1, seed=7).result
print(selected_result.values)
assert selected_result.values == {(1, 0, 1, 0, 0, 1): 1}

# %% [markdown]
# `on_qubits(...)` returns a new quantum kernel, received here as
# `flip_selected`.
#
# Local indices 0, 1, and 2 in `flip_three` correspond to original indices
# 0, 2, and 5, in the order supplied by `indices`.
#
# The array returned by `flip_selected` contains all six qubits in their
# original index order, so the measured tuple has 1 only at positions 0, 2,
# and 5.

# %% [markdown]
# :::{note}
# **Current limitations:** In the current implementation, `width` and
# `indices` must have concrete Python values when you call `on_qubits(...)`.
# These values are used to construct the quantum kernel and cannot be
# supplied later through `qm.specialize(...)` or `qm.run(...)`.
#
# The indices must be distinct and within the array. The number of selected
# indices must match the number of qubits accepted by the quantum kernel.
# The quantum kernel passed to `on_qubits(...)` must be unitary, accept a
# single qubit array, and return an array of the same width. Quantum kernels
# that perform measurements cannot be used. Also, `on_qubits(...)` does not
# accept quantum kernels with `Const` or `Param` arguments. To use such a
# kernel, wrap it in another quantum kernel that accepts only the qubit array
# and calls the original kernel with fixed values for those arguments.
# :::

# %% [markdown]
# ## Summary
#
# - Use a gate's `target=` argument to select one qubit, or a loop to apply the gate to several selected indices.
# - Use `qm.split_at(...)` to separate a consecutive part of an array and pass it to a quantum kernel.
# - Use `qm.concat(...)` to combine arrays in the required index order, including arrays prepared separately.
# - Use `on_qubits(...)` to create a quantum kernel that applies a quantum kernel to selected indices, including nonconsecutive ones. Supply concrete `width` and `indices` values when creating it.
#
# Splitting and combining change how qubits are grouped and indexed. They
# add no quantum gates and preserve the physical quantum state, including
# entanglement between the parts.
