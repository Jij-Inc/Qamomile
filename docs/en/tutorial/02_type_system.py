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
# # Qamomile's Type System
#
# Qamomile uses a rich type system to distinguish quantum and classical data,
# enforce correctness before execution, and make circuits self-documenting.
# In this tutorial we will learn every type that Qamomile provides and
# when to reach for each one.
#
# ## What We Will Learn
# - The full catalogue of Qamomile types
# - Quantum types vs classical types
# - How to create qubits and qubit arrays
# - Linear type errors: what they look like and how to fix them
# - Classical scalar types: `Float`, `UInt`, `Bit` (type annotations only)
# - Symbolic values and iteration with `qmc.range()` and `qmc.items()`
# - Container types: `Vector`, `Dict`, `Tuple`, `Matrix`, `Tensor`
# - Special types: `QFixed`, `Observable`

# %%

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## 1. Type Overview
#
# Every value inside a `@qmc.qkernel` function has a Qamomile type.
# These types are used as **type annotations** in function signatures
# (e.g., `theta: qmc.Float`) and as **runtime handles** that track
# data flow through the circuit.
#
# ### Qubit Types
#
# Only qubit types have constructors that we call directly inside
# `@qkernel` functions:
#
# | Type | Constructor | Description |
# |------|-------------|-------------|
# | `Qubit` | `qmc.qubit(name=)` | Single qubit, initialised to $\|0\rangle$ |
# | `Vector[Qubit]` | `qmc.qubit_array(n, name=)` | 1-D qubit register |
#
# ### Type annotation-only types
#
# All other types are used as **type annotations** in `@qkernel`
# signatures. Their values are provided via function arguments,
# `bindings` at transpile time, or as return values from operations.
# They do not have user-facing constructors.
#
# | Type | Category | Description |
# |------|----------|-------------|
# | `Float` | Classical | Floating-point parameter (rotation angles, etc.) |
# | `UInt` | Classical | Unsigned integer (loop bounds, array indices) |
# | `Bit` | Classical | Measurement result (returned by `qmc.measure()`) |
# | `Vector[T]` | Container | 1-D array (`Vector[Float]`, `Vector[Bit]`, etc.) |
# | `Dict[K, V]` | Container | Key-value mapping (passed via `bindings`) |
# | `Tuple[K, V]` | Container | Fixed-size pair (unpacked from Dict keys) |
# | `Matrix[T]` | Container | 2-D array |
# | `Tensor[T]` | Container | N-D array (3+ dimensions) |
# | `QFixed` | Quantum | Fixed-point quantum number (returned by `qmc.qpe()`) |
# | `Observable` | Special | Hamiltonian reference (passed via `bindings`, used with `qmc.expval()`) |
#
# ### Quantum vs Classical
#
# The most important distinction is between **quantum** and **classical** types:
#
# - **Quantum types** (`Qubit`, `QFixed`, `Vector[Qubit]`) obey the
#   **linear type rule**: each handle can only be used once, and we must
#   reassign after every gate (`q = qmc.h(q)`).
# - **Classical types** (`Float`, `UInt`, `Bit`) can be freely reused and
#   copied -- they behave like ordinary Python values.

# %% [markdown]
# ## 2. Qubit Types
#
# `Qubit` and `Vector[Qubit]` are the **only types with user-facing
# constructors**. All other types receive their values through:
#
# - Function arguments with type annotations (`theta: qmc.Float`)
# - The `bindings` dict at transpile time
# - Return values from operations (`qmc.measure()` returns `Bit`)
#
# ### Single Qubit: `qmc.qubit()`
#
# Creates one qubit initialised to $|0\rangle$.


# %%
@qmc.qkernel
def single_qubit_demo() -> qmc.Bit:
    """Create a single qubit, apply H, and measure."""
    q = qmc.qubit(name="q")
    q = qmc.h(q)
    return qmc.measure(q)


single_qubit_demo.draw()

# %% [markdown]
# ### Qubit Array: `qmc.qubit_array()`
#
# Creates a `Vector[Qubit]` -- a 1-D register of qubits.
# We access individual qubits by index and query the size via `.shape[0]`.


# %%
@qmc.qkernel
def qubit_array_demo(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Create an array of n qubits, apply H to each, and measure."""
    qubits = qmc.qubit_array(n, name="q")

    # qubits.shape[0] gives the symbolic array size
    size = qubits.shape[0]

    for i in qmc.range(size):
        qubits[i] = qmc.h(qubits[i])

    return qmc.measure(qubits)


qubit_array_demo.draw(n=3, fold_loops=False)

# %% [markdown]
# #### `draw()` options for parametric circuits
#
# The `draw()` method accepts keyword arguments to control visualization:
#
# - **`n=3`**: Binds the symbolic parameter `n` to a concrete value.
#   For circuits using `qmc.qubit_array(n, ...)`, we **must** provide the
#   qubit count -- otherwise `draw()` cannot determine the circuit layout
#   and will raise an error.
# - **`fold_loops=False`**: By default (`fold_loops=True`), loops created
#   by `qmc.range()` are displayed as compact blocks. Setting it to `False`
#   unrolls the loop so we can see each iteration as individual gates.

# %%
# What happens when you don't specify n?
try:
    qubit_array_demo.draw()  # No n specified — error!
except Exception as e:
    print(f"Error (expected): {type(e).__name__}: {e}")

# %% [markdown]
# ### Linear Type Errors
#
# Because qubits obey the linear type rule, Qamomile catches common
# mistakes at trace time -- before our circuit ever reaches a backend.
# There are three error types:
#
# | Error | Cause |
# |-------|-------|
# | `QubitConsumedError` | Reusing a qubit handle after it was consumed by a gate |
# | `QubitAliasError` | Using the same qubit as both inputs to a two-qubit gate |
# | `UnreturnedBorrowError` | Borrowing a second array element before returning the first |
#
# Let's see each one in action.


# %%
# QubitConsumedError: reusing a consumed qubit
@qmc.qkernel
def consumed_error_demo() -> tuple[qmc.Bit, qmc.Bit]:
    q = qmc.qubit(name="q")
    q1 = qmc.h(q)  # consumes q
    q2 = qmc.x(q)  # ERROR: q was already consumed
    return qmc.measure(q1), qmc.measure(q2)


try:
    consumed_error_demo.draw()
except Exception as e:
    print(f"QubitConsumedError (expected): {type(e).__name__}: {e}")


# %%
# QubitAliasError: same qubit as both control and target
@qmc.qkernel
def alias_error_demo() -> tuple[qmc.Bit, qmc.Bit]:
    q = qmc.qubit(name="q")
    q1, q2 = qmc.cx(q, q)  # ERROR: same qubit twice
    return qmc.measure(q1), qmc.measure(q2)


try:
    alias_error_demo.draw()
except Exception as e:
    print(f"QubitAliasError (expected): {type(e).__name__}: {e}")


# %%
# UnreturnedBorrowError: borrowing without returning
@qmc.qkernel
def borrow_error_demo() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(3, name="q")
    q0 = q[0]  # borrow q[0]
    q0 = qmc.h(q0)
    q1 = q[1]  # ERROR: q[0] not returned yet
    q1 = qmc.x(q1)
    q[0] = q0
    q[1] = q1
    return qmc.measure(q)


try:
    borrow_error_demo.draw()
except Exception as e:
    print(f"UnreturnedBorrowError (expected): {type(e).__name__}: {e}")

# %% [markdown]
# These errors are caught immediately when the `@qkernel` is traced
# (during `draw()` or `transpile()`), not at execution time. This means
# we get clear, actionable error messages before any quantum hardware
# is involved.

# %% [markdown]
# ## 3. Classical Scalar Types
#
# Classical scalars carry data that is known (or will be known) at circuit
# transpilation time. They are used as gate parameters, loop bounds, and
# array indices.
#
# We do not construct classical types directly. Instead, we declare
# them as **type annotations** in `@qkernel` signatures and provide
# values via `bindings` or `parameters` at transpile time.
#
# ### `Float` -- Floating-Point Values
#
# `Float` is the type for continuous gate parameters such as rotation angles.
# Use it as a type annotation: `theta: qmc.Float` in a `@qkernel` signature.
# (Python `float` also works as an alias and is automatically promoted
# to `Float`, but we use `qmc.Float` throughout these tutorials for
# clarity.)
#
# `Float` supports the usual arithmetic operators (`+`, `-`, `*`, `/`),
# which emit symbolic arithmetic into the IR so that the entire expression
# is evaluated during transpilation.


# %%
@qmc.qkernel
def float_arithmetic(theta: qmc.Float) -> qmc.Bit:
    """Demonstrate Float arithmetic inside a qkernel."""
    q = qmc.qubit(name="q")

    # Arithmetic on Float values produces new Float handles
    half_theta = theta / 2
    q = qmc.rx(q, half_theta)

    double_theta = theta * 2
    q = qmc.ry(q, double_theta)

    return qmc.measure(q)


float_arithmetic.draw()

# %% [markdown]
# ### `UInt` -- Unsigned Integers
#
# `UInt` is the type for non-negative integers. It is used for:
#
# - **Array indices**: indexing into `Vector[Qubit]`
# - **Loop bounds**: the argument to `qmc.range()`
# - **Symbolic sizes**: `qubits.shape[0]` returns a `UInt`
#
# Like `Float`, it supports arithmetic (`+`, `-`, `*`, `//`, `**`)
# and comparisons (`<`, `>`, `<=`, `>=`).
#
# Use `n: qmc.UInt` as the type annotation in a `@qkernel` signature.
# (Python `int` also works as an alias and is automatically promoted
# to `UInt`, but we use `qmc.UInt` throughout these tutorials for
# clarity.)


# %%
@qmc.qkernel
def uint_demo(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Use UInt for symbolic array size and loop bounds."""
    qubits = qmc.qubit_array(n, name="q")

    # n is available as a UInt inside the kernel
    for i in qmc.range(n):
        qubits[i] = qmc.h(qubits[i])

    return qmc.measure(qubits)


uint_demo.draw(n=4, fold_loops=False)

# %% [markdown]
# ### `Bit` -- Measurement Results
#
# `Bit` is the type returned by `qmc.measure()`. It represents a classical
# bit -- the outcome of measuring a qubit.
#
# - A single qubit measurement returns `Bit`
# - Measuring a `Vector[Qubit]` returns `Vector[Bit]`
# - `Bit` is typically used only as a return type


# %%
@qmc.qkernel
def bit_demo() -> tuple[qmc.Bit, qmc.Bit]:
    """Measure two qubits independently."""
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")

    q0 = qmc.h(q0)
    q1 = qmc.x(q1)

    # Each qmc.measure() returns a Bit
    b0 = qmc.measure(q0)
    b1 = qmc.measure(q1)
    return b0, b1


bit_demo.draw()

# %% [markdown]
# ### Classical Types are Freely Reusable
#
# Unlike quantum types, classical handles (`Float`, `UInt`, `Bit`)
# can be read multiple times without being consumed.


# %%
@qmc.qkernel
def reuse_classical(theta: qmc.Float) -> tuple[qmc.Bit, qmc.Bit]:
    """The same Float handle can be used in multiple gates."""
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")

    # theta is used twice -- this is perfectly fine for classical types
    q0 = qmc.rx(q0, theta)
    q1 = qmc.ry(q1, theta)

    return qmc.measure(q0), qmc.measure(q1)


reuse_classical.draw()

# %% [markdown]
# ## 4. Symbolic Values and Iteration
#
# When we write `n: qmc.UInt` or `theta: qmc.Float` in a `@qkernel` signature,
# the values are **symbolic** at trace time. The actual values are supplied
# later via `bindings` during transpilation.
#
# This design lets us define circuits once and reuse them with different
# sizes and parameters.
#
# ### `qmc.range()` -- Looping with Symbolic Bounds
#
# Python's built-in `range()` does not work with symbolic `UInt` values.
# Qamomile provides `qmc.range()` which accepts both `int` and `UInt`
# arguments and is expanded into a quantum-aware loop during transpilation.
#
# ```python
# for i in qmc.range(n):           # 0 to n-1
# for i in qmc.range(start, stop): # start to stop-1
# ```
#
# The loop variable `i` is a `UInt` that can be used to index into arrays.


# %%
@qmc.qkernel
def range_demo(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Apply H gates to all qubits using qmc.range()."""
    qubits = qmc.qubit_array(n, name="q")

    for i in qmc.range(n):
        qubits[i] = qmc.h(qubits[i])

    return qmc.measure(qubits)


range_demo.draw(n=4, fold_loops=False)

# %% [markdown]
# ### `qmc.items()` -- Iterating Over Dictionaries
#
# `qmc.items()` iterates over the key-value pairs of a `Dict` handle.
# This is essential for problems where the circuit structure depends on
# data (such as Ising model coefficients).
#
# The Dict and its contents are supplied via `bindings` at transpile time,
# and the loop is **unrolled** -- each iteration becomes concrete gates in
# the final circuit.
#
# ```python
# for (i, j), Jij in qmc.items(ising):
#     q[i], q[j] = qmc.rzz(q[i], q[j], gamma * Jij)
# ```


# %%
@qmc.qkernel
def items_demo(
    n_qubits: qmc.UInt,
    ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    """Apply RZZ gates based on Ising coefficients from a Dict."""
    q = qmc.qubit_array(n_qubits, name="q")
    for (i, j), Jij in qmc.items(ising):
        q[i], q[j] = qmc.rzz(q[i], q[j], gamma * Jij)
    return qmc.measure(q)


items_demo.draw(n_qubits=1)


# %% [markdown]
# The `ising` dictionary maps qubit-pair indices to coupling strengths.
# We provide the data at transpile time via `bindings`.

# %%
# Define Ising coefficients: J_{01} = 1.0, J_{12} = -0.5
ising_data = {(0, 1): 1.0, (1, 2): -0.5}

exec_items = transpiler.transpile(
    items_demo,
    bindings={"n_qubits": 3, "ising": ising_data, "gamma": 0.5},
)

# %% [markdown]
# We can inspect the transpiled circuit to verify that the loop was
# unrolled into two concrete RZZ gates.

# %%
qiskit_circuit = exec_items.get_first_circuit()
print("=== Transpiled Circuit ===")
print(qiskit_circuit.draw(output="text"))

# %% [markdown]
# ## 5. Container Types
#
# Qamomile provides container types for grouping values.
#
# ### `Vector[T]` -- 1-D Arrays
#
# `Vector` is a one-dimensional array parameterised by its element type.
# The most common usage is `Vector[Qubit]` (qubit registers) and
# `Vector[Bit]` (measurement results).
#
# ```python
# qubits: qmc.Vector[qmc.Qubit] = qmc.qubit_array(n, name="q")
# q = qubits[i]       # get element (borrows the qubit)
# qubits[i] = q       # return element (gives the qubit back)
# qubits.shape[0]     # symbolic size as UInt
# ```
#
# `Vector[Float]` can also hold classical arrays (e.g., parameter vectors).

# %% [markdown]
# ### `Dict[K, V]` -- Key-Value Mappings
#
# `Dict` maps keys to values. In Qamomile it is typically used to pass
# problem data into a circuit, such as Ising coupling coefficients.
#
# - The key type `K` is often `Tuple[UInt, UInt]` (a pair of qubit indices)
#   or just `UInt` (a single index).
# - The value type `V` is typically `Float` (a coefficient).
# - Iterate with `qmc.items(dict_handle)`.
# - The actual data is provided via `bindings` as a plain Python `dict`.
#
# ```python
# # Type annotation
# ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]
#
# # Iteration
# for (i, j), Jij in qmc.items(ising):
#     ...
#
# # Providing data at transpile time
# transpiler.transpile(kernel, bindings={"ising": {(0, 1): 1.0, (1, 2): -0.5}})
# ```

# %% [markdown]
# ### `Tuple[K, V]` -- Fixed-Size Pairs
#
# `Tuple` represents a pair of values. It is primarily used as the key type
# in `Dict` for multi-index entries (e.g., qubit pairs `(i, j)`).
#
# Inside a `for ... in qmc.items(d):` loop, tuple keys are automatically
# unpacked:
#
# ```python
# for (i, j), Jij in qmc.items(ising):
#     # i and j are UInt handles
#     # Jij is a Float handle
# ```

# %% [markdown]
# ## 6. Special Types
#
# ### `QFixed` -- Quantum Fixed-Point Number
#
# `QFixed` represents a quantum register interpreted as a fixed-point
# binary number. It is used in **Quantum Phase Estimation (QPE)** to
# automatically decode the estimated phase from the measurement results.
#
# `QFixed` is a quantum type (subject to linear type rules). It is
# returned by `qmc.qpe()` -- we do not construct it directly.
#
# ```python
# # QFixed is returned by qmc.qpe()
# phase: qmc.QFixed = qmc.qpe(target, counting, unitary, **params)
# result: qmc.Float = qmc.measure(phase)
# ```
#
# See the QPE tutorial (`05_stdlib.ipynb`) for a working example.
#
# ### `Observable` -- Hamiltonian Reference
#
# `Observable` is a handle that refers to a Hamiltonian operator.
# It is a **type annotation-only** type -- it cannot be constructed inside
# a `@qkernel`. Instead, we build the Hamiltonian in Python using
# `qamomile.observable` and pass it into the kernel via `bindings`.
#
# `Observable` is used with `qmc.expval()` to compute expectation values
# -- a key operation in variational quantum algorithms (VQE, QAOA).
#
# ```python
# import qamomile.observable as qm_o
#
# # Build Hamiltonian in Python (outside @qkernel)
# H = qm_o.Z(0) * qm_o.Z(1) + 0.5 * (qm_o.X(0) + qm_o.X(1))
#
# @qmc.qkernel
# def vqe_step(
#     q: qmc.Vector[qmc.Qubit],
#     H: qmc.Observable,
#     theta: qmc.Float,
# ) -> qmc.Float:
#     q[0] = qmc.ry(q[0], theta)
#     q[0], q[1] = qmc.cx(q[0], q[1])
#     return qmc.expval(q, H)  # <psi|H|psi>
#
# # Pass Hamiltonian via bindings (requires retranspilation if changed)
# executable = transpiler.transpile(
#     vqe_step,
#     bindings={"H": H},
#     parameters=["theta"],
# )
# ```
#
# We will encounter `Observable` in the optimisation tutorials.

# %% [markdown]
# ## 7. Summary: When to Use Each Type
#
# | I need to... | Use this type | Example |
# |-------------|--------------|---------|
# | Create a single qubit | `Qubit` | `q = qmc.qubit(name="q")` |
# | Create a qubit register | `Vector[Qubit]` | `qubits = qmc.qubit_array(n, name="q")` |
# | Pass a rotation angle | `Float` | `theta: qmc.Float` |
# | Index into an array | `UInt` | `i: qmc.UInt` or loop variable from `qmc.range()` |
# | Store a measurement result | `Bit` | `b = qmc.measure(q)` |
# | Store multiple measurements | `Vector[Bit]` | `bits = qmc.measure(qubits)` |
# | Pass problem data (coefficients) | `Dict[K, V]` | `ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]` |
# | Represent a multi-index key | `Tuple[K, V]` | `qmc.Tuple[qmc.UInt, qmc.UInt]` |
# | Iterate over problem data | `qmc.items(d)` | `for (i, j), Jij in qmc.items(ising):` |
# | Loop with symbolic bounds | `qmc.range(n)` | `for i in qmc.range(n):` |
# | Decode a QPE phase | `QFixed` | (see QPE tutorial) |
# | Compute expectation values | `Observable` | (see optimisation tutorials) |
#
# ### Key Rules to Remember
#
# 1. Only **qubit types** have user-facing constructors: `qmc.qubit()` and
#    `qmc.qubit_array()`. All other types are type annotations.
# 2. **Quantum types** (`Qubit`, `QFixed`) enforce the linear type rule:
#    always reassign after gates.
# 3. **Classical types** (`Float`, `UInt`, `Bit`) can be freely reused.
# 4. Use `qmc.range()` (not Python `range()`) for loops inside `@qkernel`.
# 5. Use `qmc.items()` to iterate over `Dict` handles.
# 6. All data-dependent values (`Dict`, `UInt`, `Float`) are provided
#    via `bindings` at transpile time.
#
# ### Quick Reference: Constructors
#
# ```python
# import qamomile.circuit as qmc
#
# # The only two constructors we need:
# q = qmc.qubit(name="q")               # Qubit
# qubits = qmc.qubit_array(n, name="q")  # Vector[Qubit]
#
# # Everything else is a type annotation:
# # theta: qmc.Float           — rotation angle
# # n: qmc.UInt                — integer parameter
# # b: qmc.Bit                 — measurement result
# # ising: qmc.Dict[K, V]      — problem data (via bindings)
# # H: qmc.Observable           — Hamiltonian (via bindings)
#
# # Iteration inside @qkernel:
# for i in qmc.range(n):                      # symbolic for-loop
# for (i, j), v in qmc.items(dict_handle):    # dict iteration
# ```
#
# In the next tutorials we will see these types in action:
# - **`03_gates.ipynb`**: Complete gate reference (all 11 gates)
# - **`04_superposition_entanglement.ipynb`**: Superposition, interference, Bell/GHZ states
# - **`05_stdlib.ipynb`**: QFT, QPE with `QFixed`
# - **`optimization/qaoa.ipynb`**: QAOA with `Dict`, `Tuple`, and `qmc.items()`

# %% [markdown]
# ## What We Learned
#
# - **The full catalogue of Qamomile types** — Qamomile provides quantum types (`Qubit`, `QFixed`), classical scalars (`Float`, `UInt`, `Bit`), containers (`Vector`, `Dict`, `Tuple`, `Matrix`, `Tensor`), and special types (`Observable`).
# - **Quantum types vs classical types** — Quantum types enforce linear ownership (consume-and-return), while classical types can be freely reused.
# - **How to create qubits and qubit arrays** — `qmc.qubit(name=...)` and `qmc.qubit_array(n, name=...)` are the only constructors; all other types are type annotations.
# - **Linear type errors: what they look like and how to fix them** — `QubitConsumedError`, `QubitAliasError`, and `UnreturnedBorrowError` are caught at trace time with clear messages; the fix is always to capture gate return values and avoid reusing consumed handles.
# - **Classical scalar types: `Float`, `UInt`, `Bit` (type annotations only)** — These appear as function parameters or measurement results but have no standalone constructors.
# - **Symbolic values and iteration with `qmc.range()` and `qmc.items()`** — `qmc.range(n)` creates symbolic loops over `UInt` bounds, and `qmc.items(d)` iterates over `Dict` handles inside `@qkernel`.
# - **Container types: `Vector`, `Dict`, `Tuple`, `Matrix`, `Tensor`** — `Vector` holds qubit registers and measurement arrays; `Dict` and `Tuple` pass structured problem data via `bindings`.
# - **Special types: `QFixed`, `Observable`** — `QFixed` represents fixed-point quantum values from QPE; `Observable` is used for expectation value computation in variational algorithms.
