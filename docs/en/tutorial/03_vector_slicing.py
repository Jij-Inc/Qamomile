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
# # Vector Slicing
#
# [Tutorial 02](02_parameterized_kernels.ipynb) introduced how to build
# parameterized qkernels and how to broadcast single-qubit gates over a whole
# register. This chapter introduces **slicing**, a Qamomile feature that helps
# you write more complex qkernels.

# %%
# Install the latest Qamomile from pip.
# # !pip install "qamomile[qiskit,visualization]"

# %%
import qamomile.circuit as qmc

from qamomile.circuit.transpiler.errors import (
    AffineTypeError,
    QubitBorrowConflictError,
    UnreturnedBorrowError,
)

# %% [markdown]
# ## Basic slice syntax
#
# Indexing a `Vector[Qubit]` with a Python `slice` (`start:stop:step`) returns
# a **VectorView** — a handle that refers to a sub-range of the parent vector.
# In most situations you can use it just like an ordinary `Vector[Qubit]`.


# %%
@qmc.qkernel
def demo() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(6, name="q")
    # Pick out the even-indexed qubits via slicing.
    evens = q[0::2]
    # Pick out the odd-indexed qubits via slicing.
    odds = q[1::2]
    # Loop over the slice's shape and apply an H gate to each element.
    for i in qmc.range(evens.shape[0]):
        evens[i] = qmc.h(evens[i])
    # Broadcast X (a single-qubit gate) over the whole slice.
    odds = qmc.x(odds)
    # Return the slice views back to the original qubit array.
    q[0::2] = evens
    q[1::2] = odds
    return qmc.measure(q)


demo.draw(fold_loops=False)

# %% [markdown]
# A few things to notice in `demo`:
#
# - `evens = q[0::2]` creates a `VectorView` covering the even-indexed qubits
#   of `q`. Similarly, `odds = q[1::2]` covers the odd-indexed qubits.
# - The `evens[i] = qmc.h(evens[i])` inside the loop follows the same affine
#   pattern you saw in Tutorial 02 — each element is consumed and a new
#   element handle replaces it.
# - `VectorView` can be used in much the same way as `Vector[Qubit]`.

# %% [markdown]
# ## Inline shorthand
#
# When the body is a single broadcastable operation, you don't need to name
# the `VectorView`. Borrow, transform, and return in one statement.


# %%
@qmc.qkernel
def demo_inline() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(6, name="q")
    q[0::2] = qmc.h(q[0::2])
    q[1::2] = qmc.x(q[1::2])
    return qmc.measure(q)


demo_inline.draw(fold_loops=False)

# %% [markdown]
# ## Nested slices
#
# A `VectorView` can itself be sliced again to produce another `VectorView`.
# The result covers a sub-range of the parent `VectorView`. Each level borrows
# from the one above, and each level must be returned to its immediate parent
# before its grandparent can be touched.
#
# Concretely: if `outer = q[0::2]` and `inner = outer[1:3]`, the return order
# is **inner → outer → root**. Return `inner` to `outer` first, then `outer`
# to `q`.


# %%
@qmc.qkernel
def nested_slice() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(6, name="q")
    outer = q[0::2]
    inner = outer[1:3]
    for i in qmc.range(inner.shape[0]):
        inner[i] = qmc.h(inner[i])
    outer[1:3] = inner
    q[0::2] = outer
    return qmc.measure(q)


nested_slice.draw(fold_loops=False)

# %% [markdown]
# ## Passing a `VectorView` to a helper kernel
#
# When a `VectorView` is passed to an external qkernel, it is treated the same
# as a `Vector[Qubit]`.


# %%
@qmc.qkernel
def h_all(v: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    return qmc.h(v)


@qmc.qkernel
def x_all(v: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    return qmc.x(v)


@qmc.qkernel
def demo_helper() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(6, name="q")
    evens = q[0::2]
    odds = q[1::2]
    evens = h_all(evens)
    odds = x_all(odds)
    q[0::2] = evens
    q[1::2] = odds
    return qmc.measure(q)


demo_helper.draw(inline=True, fold_loops=False)

# %% [markdown]
# The same convention extends to the built-in helpers in
# `qamomile.circuit.stdlib`. `qmc.qft`, `qmc.iqft`, and `qmc.qpe` all take a
# `Vector[Qubit]` and work equally well on a `VectorView`.


# %%
@qmc.qkernel
def qft_on_window() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(6, name="q")
    q[1:4] = qmc.qft(q[1:4])
    return qmc.measure(q)


qft_on_window.draw()

# %% [markdown]
# ## Splitting a qubit array via slicing
#
# In Qamomile, you cannot directly split or merge the qubit array
# `Vector[Qubit]` itself. Slicing, however, lets you extract several
# `VectorView`s from one large qubit array, so you can write an algorithm as
# if you had multiple registers. Below, we apply QFT to the whole register
# and then apply QFT again to just a portion. Qamomile's QFT (`qmc.qft`)
# takes a `Vector[Qubit]` as input, so without slicing you would have to
# write QFT from scratch inside your qkernel. With slicing, you can pull
# out a portion of the qubit array and skip that work.


# %%
@qmc.qkernel
def demo_separation() -> qmc.Vector[qmc.Bit]:
    qs = qmc.qubit_array(6, name="qs")
    # Apply QFT to the whole register.
    qs = qmc.qft(qs)
    # Apply QFT again to just the front half.
    partial_qs = qs[0:3]
    partial_qs = qmc.qft(partial_qs)
    # Return the slice view back to the original qubit array.
    qs[0:3] = partial_qs
    return qmc.measure(qs)


demo_separation.draw(inline=True, fold_loops=False)

# %% [markdown]
# ## Error patterns with `VectorView`
#
# A `VectorView` is created by **borrowing** the qubits specified by the
# slice from its parent qubit array `Vector[Qubit]`. From the parent's
# perspective, those qubits are lent out; until they are returned, the
# parent `Vector[Qubit]` cannot access them directly. For the same reason,
# operations that consume the parent `Vector[Qubit]` as a whole are not
# allowed either. To return the lent qubits, you must perform a slice
# assignment.
#
# Below we walk through error patterns where the lent qubits are accessed
# without first being returned.


# %%
@qmc.qkernel
def direct_parent_access() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(6, name="q")
    v = q[1:3]
    q[1] = qmc.h(q[1])  # direct access to a lent qubit raises an error
    q[1:3] = v
    return qmc.measure(q)


try:
    direct_parent_access.draw()
except QubitBorrowConflictError as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
else:
    raise AssertionError(
        "expected QubitBorrowConflictError, but draw() returned normally"
    )


# %%
@qmc.qkernel
def forgot_return() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(6, name="q")
    evens = q[0::2]
    for i in qmc.range(evens.shape[0]):
        evens[i] = qmc.h(evens[i])
    return qmc.measure(q)  # VectorView not returned — raises


try:
    forgot_return.draw()
except UnreturnedBorrowError as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
else:
    raise AssertionError("expected UnreturnedBorrowError, but draw() returned normally")


# %%
@qmc.qkernel
def overlapping_views() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(6, name="q")
    a = q[0:3]
    b = q[2:5]  # double-lending the qubits already lent to `a` — raises
    q[0:3] = a
    q[2:5] = b
    return qmc.measure(q)


try:
    overlapping_views.draw()
except QubitBorrowConflictError as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
else:
    raise AssertionError(
        "expected QubitBorrowConflictError, but draw() returned normally"
    )


# %%
@qmc.qkernel
def invalid_nested_return() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(6, name="q")
    outer = q[0::2]
    inner = outer[1:3]
    for i in qmc.range(inner.shape[0]):
        inner[i] = qmc.h(inner[i])
    q[0::2] = outer  # returning outer without first returning inner — raises
    return qmc.measure(q)


try:
    invalid_nested_return.draw()
except AffineTypeError as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
else:
    raise AssertionError("expected AffineTypeError, but draw() returned normally")

# %% [markdown]
# **Next**: [Controlled Gates](04_controlled_gates.ipynb) — `qmc.control`
# for built-in gates and sub-kernels, concrete vs symbolic control counts,
# and the catalogue of patterns that do not compose.
