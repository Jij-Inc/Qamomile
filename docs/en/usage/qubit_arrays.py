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
# tags: [usage]
# ---
#
# # Working with Qubit Arrays
#
# A quantum program often needs different operations on different qubits. For
# example, you might flip only the last of six qubits, process the first three
# in a separate function, or combine two prepared groups for the next step.
# Qamomile represents an ordered group of qubits as a **qubit array**.
#
# This guide covers three operations: applying gates to selected positions,
# splitting an array into smaller parts, and combining arrays. Each section
# starts with a use case and shows a small example you can adapt.

# %%
# Install Qamomile with Qiskit support for the local simulator used below.
# # !pip install "qamomile[qiskit]"

# %%
from __future__ import annotations

import qamomile as qm
from qamomile.qiskit import Aer
from qamomile.stdlib import qft

# %% [markdown]
# ## Apply gates to part of a qubit array
#
# Use a gate's `target=` argument when you want to choose which qubit changes
# while keeping the array together. The positions start at zero, just like
# Python sequence indices.
#
# ### Select one qubit
#
# Suppose only the last qubit should start in state 1. `qm.qubits(6)` creates
# six qubits in state 0, and the X gate flips a qubit between 0 and 1. Applying
# it with `target=5` changes the last position.


# %%
@qm.qkernel
def flip_last() -> qm.Bits[6]:
    state = qm.qubits(6)
    state = qm.x(state, target=5)
    return qm.measure(state)


# %% [markdown]
# To check that X applies only to the last qubit, simulate the quantum program
# with `Aer`.

# %%
endpoint = Aer()
last_result = qm.run(flip_last, on=endpoint, shots=1, seed=7).result
print(last_result.values)
assert last_result.values == {(0, 0, 0, 0, 0, 1): 1}

# %% [markdown]
# To vary the array length or selected position, use symbolic values. The
# following example takes the length as `width` and the position as `position`.
# `qm.Const[qm.Int]` marks an integer argument that determines the quantum
# program's structure. Fix its value with `qm.specialize` before running the
# program.
#
# Here we flip position 1 of a four-qubit array. The position must satisfy
# `0 <= position < width`.


# %%
@qm.qkernel
def flip_at(width: qm.Const[qm.Int], position: qm.Const[qm.Int]) -> qm.Bits[width]:
    state = qm.qubits(width)
    state = qm.x(state, target=position)
    return qm.measure(state)


# %%
selected_program = qm.specialize(flip_at, width=4, position=1)
selected_result = qm.run(selected_program, on=endpoint, shots=1, seed=7).result
print(selected_result.values)
assert selected_result.values == {(0, 1, 0, 0): 1}

# %% [markdown]
# ### Select several qubits
#
# A loop is useful when several positions need the same gate. For example,
# `range(0, 6, 2)` visits positions 0, 2, and 4, so the following function flips
# only the even-indexed qubits.


# %%
@qm.qkernel
def flip_even() -> qm.Bits[6]:
    state = qm.qubits(6)
    for target in range(0, 6, 2):
        state = qm.x(state, target=target)
    return qm.measure(state)


# %%
even_result = qm.run(flip_even, on=endpoint, shots=1, seed=7).result
print(even_result.values)
assert even_result.values == {(1, 0, 1, 0, 1, 0): 1}

# %% [markdown]
# For a contiguous range, use consecutive indices. Here `range(3)` visits
# positions 0, 1, and 2, which flips the first three qubits.


# %%
@qm.qkernel
def flip_first_three() -> qm.Bits[6]:
    state = qm.qubits(6)
    for target in range(3):
        state = qm.x(state, target=target)
    return qm.measure(state)


# %%
first_three_result = qm.run(flip_first_three, on=endpoint, shots=1, seed=7).result
print(first_three_result.values)
assert first_three_result.values == {(1, 1, 1, 0, 0, 0): 1}

# %% [markdown]
# You can also specify the array length and selected range through arguments.
# `start` is the first position, `stop` is the end position (excluded), and
# `step` is the spacing between positions. Here we use a positive `step` and
# a range that satisfies `0 <= start < stop <= width`.


# %%
@qm.qkernel
def flip_range(
    width: qm.Const[qm.Int],
    start: qm.Const[qm.Int],
    stop: qm.Const[qm.Int],
    step: qm.Const[qm.Int],
) -> qm.Bits[width]:
    state = qm.qubits(width)
    for target in range(start, stop, step):
        state = qm.x(state, target=target)
    return qm.measure(state)


# %% [markdown]
# Setting `step=2` selects every other qubit. Let's flip positions 1, 3, 5,
# and 7 of an eight-qubit array.

# %%
spaced_program = qm.specialize(flip_range, width=8, start=1, stop=8, step=2)
spaced_result = qm.run(spaced_program, on=endpoint, shots=1, seed=7).result
print(spaced_result.values)
assert spaced_result.values == {(0, 1, 0, 1, 0, 1, 0, 1): 1}

# %% [markdown]
# The same function selects a contiguous range with `step=1`. This time,
# we flip the first two qubits of a five-qubit array.

# %%
contiguous_program = qm.specialize(flip_range, width=5, start=0, stop=2, step=1)
contiguous_result = qm.run(contiguous_program, on=endpoint, shots=1, seed=7).result
print(contiguous_result.values)
assert contiguous_result.values == {(1, 1, 0, 0, 0): 1}

# %% [markdown]
# You do not need a loop to apply the same operation to the entire array.
# For a single-qubit gate such as `qm.x`, omitting `target=` applies the gate
# independently to every qubit in the array.
# For example, `qm.x(state)` flips every qubit in the array.
#
# This also works when the length is an argument. The next example flips
# all four qubits.


# %%
@qm.qkernel
def flip_every_qubit(width: qm.Const[qm.Int]) -> qm.Bits[width]:
    state = qm.qubits(width)
    state = qm.x(state)
    return qm.measure(state)


# %%
all_program = qm.specialize(flip_every_qubit, width=4)
all_result = qm.run(all_program, on=endpoint, shots=1, seed=7).result
print(all_result.values)
assert all_result.values == {(1, 1, 1, 1): 1}

# %% [markdown]
# ## Split a qubit array
#
# Split an array when you want to handle its parts separately, for example to
# measure them as separate results or pass only one part to another function.
# `qm.split_at` gives each part its own value to use in later operations.
#
# ### Split at a chosen position
#
# To handle the first and last three qubits separately, split a six-qubit
# array at position 3. In `qm.split_at(state, 3)`, the first result contains
# positions 0, 1, and 2; the second contains positions 3, 4, and 5.
#
# Each part then has its own indices starting at zero. The example below flips
# original position 2 before splitting, then position 0 of `back`, which was
# position 3 in the original array. It measures and returns both parts.


# %%
@qm.qkernel
def split_halves() -> tuple[qm.Bits[3], qm.Bits[3]]:
    state = qm.qubits(6)
    state = qm.x(state, target=2)
    front, back = qm.split_at(state, 3)
    back = qm.x(back, target=0)
    return qm.measure(front), qm.measure(back)


# %% [markdown]
# `result.values` preserves the two-part return shape. The first tuple is the
# measured `front`, and the second is the measured `back`.

# %%
halves_result = qm.run(split_halves, on=endpoint, shots=1, seed=7).result
print(halves_result.values)
assert halves_result.values == {((0, 0, 1), (1, 0, 0)): 1}

# %% [markdown]
# :::{note}
# After the split, continue with the returned qubit arrays, `front` and `back`
# in this example. The original `state` cannot be reused.
#
# A qubit array is a linear value. `qm.split_at` also consumes its input
# `state` and returns two arrays in its place. Gate operations work the same
# way: `back = qm.x(back, target=0)`
# assigns the new value returned by the operation to `back`. You can apply
# further operations to this new value. You can measure the resulting arrays
# separately, return them from a function, or combine them as shown in the
# next section.
# :::
#
# The split must leave both parts nonempty. For an array of `n` qubits, choose
# a boundary with `0 < boundary < n`; 0 and `n` are rejected. Splitting only
# changes how the existing qubits are grouped and adds no quantum gates.

# %% [markdown]
# :::{note}
# Currently, `qm.split_at` does not support symbolic array lengths or split
# positions expressed as `qm.Const` arguments. Use concrete integers for the
# array length and split position, as in the example above.
# :::

# %% [markdown]
# ### Extract a range from the middle
#
# A function that processes three qubits may need positions 1, 2, and 3 of a
# larger array. First split off position 0, then split the remaining array at
# 3 to separate the desired window from the final two qubits.
#
# The helper `flip_three` below accepts and returns `qm.Q[qm.Bits[3]]`, the type
# of a three-qubit quantum array. It applies X to all three qubits. Passing the
# extracted `window` lets us reuse that helper on just the desired part.


# %%
@qm.qkernel
def flip_three(state: qm.Q[qm.Bits[3]]) -> qm.Q[qm.Bits[3]]:
    return qm.x(state)


@qm.qkernel
def split_window() -> tuple[qm.Bits[1], qm.Bits[3], qm.Bits[2]]:
    state = qm.qubits(6)
    prefix, remainder = qm.split_at(state, 1)
    window, suffix = qm.split_at(remainder, 3)
    window = flip_three(window)
    return qm.measure(prefix), qm.measure(window), qm.measure(suffix)


# %% [markdown]
# The three results correspond to original position 0, positions 1–3, and
# positions 4–5. Only the middle part changes. The second split uses positions
# within `remainder`, so its boundary 3 counts from that part's beginning.

# %%
window_result = qm.run(split_window, on=endpoint, shots=1, seed=7).result
print(window_result.values)
assert window_result.values == {((0,), (1, 1, 1), (0, 0)): 1}

# %% [markdown]
# ## Combine qubit arrays
#
# Combine arrays when the next operation expects one array, for example when
# you want to measure the complete result or pass it to another function.
# `qm.concat` takes two or more arrays and joins them in the order supplied.
#
# :::{note}
# Currently, `qm.concat` does not support symbolic input array lengths
# expressed as `qm.Const` arguments. This section uses arrays whose lengths
# are specified as concrete integers.
# :::
#
# ### Rejoin parts after processing them
#
# If you split an array to process one part, you can then combine the parts
# again. This example passes the first half to `flip_three`, the helper from
# the previous section, and rejoins it with the unchanged second half.


# %%
@qm.qkernel
def rejoin_halves() -> qm.Bits[6]:
    state = qm.qubits(6)
    front, back = qm.split_at(state, 3)
    front = flip_three(front)
    state = qm.concat(front, back)
    return qm.measure(state)


# %% [markdown]
# `qm.concat(front, back)` puts `front` first and `back` second. Measuring the
# result now produces one six-bit tuple rather than two three-bit tuples.

# %%
rejoined_result = qm.run(rejoin_halves, on=endpoint, shots=1, seed=7).result
print(rejoined_result.values)
assert rejoined_result.values == {(1, 1, 1, 0, 0, 0): 1}

# %% [markdown]
# The same sequence works with library functions. For example, `qft` applies
# the quantum Fourier transform (QFT) to an array of the specified width.
# Here we extract a three-qubit window, pass it to `qft`, and combine it with
# the prefix and suffix in their original order.


# %%
@qm.qkernel
def qft_on_window() -> qm.Bits[6]:
    state = qm.qubits(6)
    prefix, remainder = qm.split_at(state, 1)
    window, suffix = qm.split_at(remainder, 3)
    window = qft(window, 3)
    state = qm.concat(prefix, window, suffix)
    return qm.measure(state)


# %% [markdown]
# QFT creates a superposition of the middle three qubits, so their measured
# values can vary from run to run. We sample 32 times and check that the
# surrounding qubits remain 0. The possible changes are confined to original
# positions 1, 2, and 3.

# %%
qft_result = qm.run(qft_on_window, on=endpoint, shots=32, seed=7).result
print(qft_result.values)
assert sum(qft_result.values.values()) == 32
assert all(
    len(bits) == 6 and bits[0] == bits[4] == bits[5] == 0 for bits in qft_result.values
)

# %% [markdown]
# :::{note}
# Combining consumes the input arrays. After `state = qm.concat(front, back)`,
# use `state` for the next operation. The previous `front` and `back` values
# cannot be reused or passed to `concat` again. Combining adds no quantum gates
# and does not copy qubits.
# :::

# %% [markdown]
# ### Combine separately prepared arrays
#
# The inputs to `qm.concat` can also come from separate preparations. This is
# useful when two parts of a calculation prepare their own qubits and a later
# function needs both parts as one array.
#
# Here we prepare a two-qubit array with values `(1, 0)` and a one-qubit array
# with value `(1,)`, then combine them.


# %%
@qm.qkernel
def combine_prepared() -> qm.Bits[3]:
    first = qm.qubits(2)
    first = qm.x(first, target=0)
    second = qm.qubits(1)
    second = qm.x(second, target=0)
    state = qm.concat(first, second)
    return qm.measure(state)


# %%
prepared_result = qm.run(combine_prepared, on=endpoint, shots=1, seed=7).result
print(prepared_result.values)
assert prepared_result.values == {(1, 0, 1): 1}

# %% [markdown]
# In `qm.concat(first, second)`, the two qubits from `first` occupy positions
# 0 and 1, and the qubit from `second` occupies position 2. Writing
# `qm.concat(second, first)` instead would produce `(1, 1, 0)`: the order of
# the arguments determines the order of qubits in the combined array.

# %% [markdown]
# For more about `qm.Const` and `qm.specialize`, see
# [Parameters and Specialization](../tutorial/02_parameterized_kernels.ipynb).
# For more ways to compose quantum functions, see
# [Reusable Quantum Functions](../tutorial/07_reuse_patterns.ipynb).
