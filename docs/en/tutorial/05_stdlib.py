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
# # Qamomile's Standard Library
#
# Qamomile provides a **standard library** (`stdlib`) of commonly used quantum
# subroutines, as well as an `algorithm` module with reusable circuit patterns.
#
# The `stdlib` contains the most fundamental, widely used building blocks.
# The `algorithm` module holds additional circuit patterns that arise
# frequently in practice. As the ecosystem evolves, commonly used patterns
# in `algorithm` may be promoted to `stdlib`, and the boundary between the
# two may shift between versions.
#
# ## What We Will Learn
# - Using `qmc.qft()`, `qmc.iqft()`, and `qmc.qpe()` from the standard library
# - Decomposition strategies for controlling precision vs. gate count
# - IR-level vs. backend-level circuits (and why they can differ)
# - Building Quantum Phase Estimation with `qmc.qpe()`
# - Overview of the `qamomile.circuit.algorithm` module
#
# ## Current Standard Library
#
# | Function | Description |
# |----------|-------------|
# | `qmc.qft()` | Quantum Fourier Transform |
# | `qmc.iqft()` | Inverse Quantum Fourier Transform |
# | `qmc.qpe()` | Quantum Phase Estimation |
#
# QFT and IQFT are implemented as `CompositeGate`s with pluggable
# decomposition strategies. Backends can provide native implementations
# (e.g., Qiskit's built-in QFT) or fall back to the gate-level
# decomposition. QPE is a higher-level function that orchestrates
# controlled unitaries and IQFT internally.
#
# The standard library will grow as the quantum computing ecosystem matures
# and new patterns become established.

# %%
import math

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## 1. QFT and IQFT
#
# The **Quantum Fourier Transform** (QFT) is the quantum analog of the
# discrete Fourier transform. It is a key subroutine in many quantum
# algorithms including Shor's algorithm and Quantum Phase Estimation.
#
# Qamomile provides `qmc.qft()` and `qmc.iqft()` as frontend functions
# that operate on `Vector[Qubit]` and return `Vector[Qubit]`. This follows
# the same factory function pattern we have seen with other gates:
#
# ```python
# qubits = qmc.qft(qubits)
# ```
#
# ### QFT Circuit


# %%
@qmc.qkernel
def qft_demo() -> qmc.Vector[qmc.Bit]:
    """Apply QFT to a 3-qubit register."""
    qubits = qmc.qubit_array(3, name="q")

    # Prepare a non-trivial input state: |101>
    # Ket notation uses big-endian: |q[2] q[1] q[0]>
    # So |101> means q[0]=1, q[1]=0, q[2]=1
    qubits[0] = qmc.x(qubits[0])
    qubits[2] = qmc.x(qubits[2])

    # Apply QFT
    qubits = qmc.qft(qubits)

    return qmc.measure(qubits)


qft_demo.draw()

# %% [markdown]
# By default, `draw()` shows `CompositeGate`s (like QFT) as single labelled
# boxes. Passing `expand_composite=True` "opens" each box to reveal the
# individual gates that make up the composite operation.
#
# **Important**: `expand_composite=True` shows Qamomile's **IR-level**
# decomposition — not the final backend circuit. The actual gates emitted
# to a specific backend (e.g., Qiskit) may differ, because the backend
# transpiler is free to substitute, optimize, or rearrange gates. We will
# see a concrete example of this difference in Section 2.

# %%
qft_demo.draw(expand_composite=True)

# %%
exec_qft = transpiler.transpile(qft_demo)
result_qft = exec_qft.sample(transpiler.executor(), shots=1000).result()

print("=== QFT on |101> ===")
print("After QFT, the state is spread across all basis states\n")
sorted_results = sorted(result_qft.results, key=lambda x: str(x[0]))
for value, count in sorted_results:
    percentage = count / 1000 * 100
    print(f"  {value}: {count} ({percentage:.1f}%)")

# %% [markdown]
# ### IQFT Circuit
#
# The Inverse QFT undoes the QFT. Applying QFT followed by IQFT should
# return the original state.


# %%
@qmc.qkernel
def qft_iqft_roundtrip() -> qmc.Vector[qmc.Bit]:
    """Apply QFT then IQFT: should recover original state."""
    qubits = qmc.qubit_array(3, name="q")

    # Prepare |110>
    qubits[1] = qmc.x(qubits[1])
    qubits[2] = qmc.x(qubits[2])

    # QFT followed by IQFT
    qubits = qmc.qft(qubits)
    qubits = qmc.iqft(qubits)

    return qmc.measure(qubits)


qft_iqft_roundtrip.draw()

# %%
exec_roundtrip = transpiler.transpile(qft_iqft_roundtrip)
result_roundtrip = exec_roundtrip.sample(transpiler.executor(), shots=1000).result()

print("=== QFT -> IQFT Roundtrip (should recover |110>) ===")
for value, count in result_roundtrip.results:
    print(f"  {value}: {count}")

# %% [markdown]
# The result is `(0, 1, 1)` with certainty, confirming that IQFT perfectly
# undoes QFT.

# %% [markdown]
# ## 2. Decomposition Strategies
#
# The QFT and IQFT implementations support multiple **decomposition strategies**.
# The standard QFT uses $O(n^2)$ gates, but for large circuits, an approximate
# version with $O(nk)$ gates may be preferable.
#
# ### Listing Available Strategies
#
# Import the `QFT` class from `qamomile.circuit.stdlib.qft` to access
# strategy management methods.

# %%
from qamomile.circuit.stdlib.qft import IQFT, QFT

# List available strategies
print("QFT strategies:", QFT.list_strategies())
print("IQFT strategies:", IQFT.list_strategies())

# %% [markdown]
# ### Comparing Resource Counts
#
# Use `get_resources_for_strategy()` to compare the gate counts of different
# strategies without building the full circuit.

# %%
# Compare resources for an 8-qubit QFT
qft_8 = QFT(8)

print("=== Resource Comparison: 8-qubit QFT ===\n")

for strategy_name in QFT.list_strategies():
    resources = qft_8.get_resources_for_strategy(strategy_name)
    meta = resources.custom_metadata
    print(f"Strategy: {strategy_name}")
    print(f"  H gates:    {meta['num_h_gates']}")
    print(f"  CP gates:   {meta['num_cp_gates']}")
    print(f"  SWAP gates: {meta['num_swap_gates']}")
    print(f"  Total:      {meta['total_gates']}")
    print()

# %% [markdown]
# The approximate strategies significantly reduce the number of controlled
# phase gates by truncating small-angle rotations. The parameter $k$
# (truncation depth) controls the trade-off: larger $k$ gives higher
# precision but more gates.
#
# | Strategy | Truncation depth (k) | CP Gates (n=8) | Error |
# |----------|---------------------|---------------|-------|
# | standard | -- (full) | n(n-1)/2 = 28 | 0 |
# | approximate_k2 | 2 | 13 | O(n/2^2) |
# | approximate (k=3) | 3 | ~18 | O(n/2^3) |
# | approximate_k4 | 4 | ~22 | O(n/2^4) |

# %% [markdown]
# ### Using a Specific Strategy
#
# To apply QFT with a specific strategy, use the `QFT` class directly
# and pass the `strategy` keyword argument.
#
# Let's compare the standard and approximate_k2 decompositions visually:


# %%
@qmc.qkernel
def qft_standard_4() -> qmc.Vector[qmc.Bit]:
    """QFT with standard (full) strategy."""
    qubits = qmc.qubit_array(4, name="q")
    qft_gate = QFT(4)
    qubits[0], qubits[1], qubits[2], qubits[3] = qft_gate(
        qubits[0],
        qubits[1],
        qubits[2],
        qubits[3],
        strategy="standard",
    )
    return qmc.measure(qubits)


@qmc.qkernel
def qft_approx_k2_4() -> qmc.Vector[qmc.Bit]:
    """QFT with approximate_k2 strategy."""
    qubits = qmc.qubit_array(4, name="q")
    qft_gate = QFT(4)
    qubits[0], qubits[1], qubits[2], qubits[3] = qft_gate(
        qubits[0],
        qubits[1],
        qubits[2],
        qubits[3],
        strategy="approximate_k2",
    )
    return qmc.measure(qubits)


# %% [markdown]
# Using `expand_composite=True` we can inspect how each strategy decomposes
# the QFT at the **Qamomile IR level**. The standard version includes all 6
# controlled-phase gates for 4 qubits, whereas the approximate $k=2$ version
# has only 5. Note that this is the **IR-level view** — the actual circuit
# emitted to a particular backend may look different because backends are
# free to substitute their own native implementations (as we will see below).

# %%
print("Qamomile IR — Standard QFT (all CP gates):")
qft_standard_4.draw(expand_composite=True)

# %%
print("Qamomile IR — Approximate QFT k=2 (fewer CP gates):")
qft_approx_k2_4.draw(expand_composite=True)

# %% [markdown]
# ### IR-level vs. Backend-level Circuits
#
# Now let's transpile both kernels to Qiskit and see what the backend
# actually emits.

# %%
qiskit_standard = transpiler.to_circuit(qft_standard_4)
print("Qiskit — Standard QFT:")
print(qiskit_standard.draw(output="text"))

# %%
qiskit_approx = transpiler.to_circuit(qft_approx_k2_4)
print("Qiskit — Approximate QFT k=2:")
print(qiskit_approx.draw(output="text"))

# %% [markdown]
# Both transpiled circuits show an identical `QFT` box. This is because the
# Qiskit backend provides a **native QFT emitter** that substitutes Qiskit's
# own `qiskit.circuit.library.QFT` gate for every QFT operation, regardless
# of the requested strategy.
#
# This is a concrete example of the IR-vs-backend distinction: Qamomile's IR
# records *which strategy was requested* and the `expand_composite=True` view
# shows the corresponding decomposition, while the Qiskit backend freely
# replaces it with its own optimized implementation.
#
# The strategy difference is still meaningful:
#
# - **Resource estimation** via `get_resources_for_strategy()` accurately
#   reports the gate count each strategy would use.
# - **Other backends** that do not provide a native QFT emitter will fall
#   back to Qamomile's decomposition and respect the strategy.
# - Qiskit's `QFTGate` already supports an `approximation_degree` parameter;
#   the Qamomile Qiskit emitter does not yet forward the strategy to it.
#
# **Key takeaway**: Approximation strategies control the trade-off between
# gate count and precision. The `get_resources_for_strategy()` API (shown
# above) is the reliable way to compare strategies regardless of backend.

# %% [markdown]
# ## 3. QPE Using `qmc.qpe()`
#
# **Quantum Phase Estimation** (QPE) is one of the most important quantum
# subroutines. Given a unitary $U$ and its eigenstate $|\psi\rangle$ where
# $U|\psi\rangle = e^{2\pi i \varphi}|\psi\rangle$, QPE estimates the
# phase $\varphi$.
#
# Qamomile provides `qmc.qpe()` which handles:
# 1. Hadamard gates on the counting register
# 2. Controlled-$U^{2^k}$ operations (automatic power repetition)
# 3. Inverse QFT on the counting register
# 4. Conversion to `QFixed` type for automatic decoding
#
# ### Signature
#
# ```python
# qmc.qpe(target, counting, unitary, **params) -> QFixed
# ```
#
# - `target`: The eigenstate qubit(s)
# - `counting`: `Vector[Qubit]` for the phase estimate
# - `unitary`: A `@qkernel` defining a single application of $U$
# - `**params`: Parameters passed to the unitary
#
# **Important**: The unitary should define a **single** application of $U$.
# `qmc.qpe()` internally performs the $U^{2^k}$ repetitions using the
# `power` parameter of `qmc.controlled()`.
#
# ### Defining the Unitary


# %%
@qmc.qkernel
def p_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Phase gate: P(theta)|1> = e^{i*theta}|1>."""
    return qmc.p(q, theta)


# %% [markdown]
# ### Building QPE
#
# The `QFixed` type returned by `qmc.qpe()` represents a quantum
# fixed-point number. When measured with `qmc.measure()`, it is
# automatically decoded into a `Float` value representing the
# estimated phase $\varphi = \theta / (2\pi)$.


# %%
@qmc.qkernel
def qpe_demo(theta: qmc.Float) -> qmc.Float:
    """Estimate the phase of P(theta)."""
    counting = qmc.qubit_array(3, name="counting")
    target = qmc.qubit(name="target")

    # Prepare eigenstate |1>
    target = qmc.x(target)

    # Run QPE
    phase: qmc.QFixed = qmc.qpe(target, counting, p_gate, theta=theta)

    # Measure and decode
    return qmc.measure(phase)


qpe_demo.draw(fold_loops=False, inline=True)

# %% [markdown]
# ### Verifying QPE Results
#
# For the phase gate $P(\theta)$, the eigenvalue is $e^{i\theta}$, so the
# phase is $\varphi = \theta / (2\pi)$.
#
# - $\theta = \pi/2$: expected $\varphi = 0.25$
# - $\theta = \pi/4$: expected $\varphi = 0.125$

# %%
# Test 1: theta = pi/2, expected phase = 0.25
test_theta_1 = math.pi / 2
exec_qpe1 = transpiler.transpile(qpe_demo, bindings={"theta": test_theta_1})
result_qpe1 = exec_qpe1.sample(transpiler.executor(), shots=1024).result()

print("=== QPE with theta = pi/2 (expected phase = 0.25) ===")
for value, count in result_qpe1.results:
    print(f"  Measured phase: {value}, Count: {count}")

# %%
# Test 2: theta = pi/4, expected phase = 0.125
test_theta_2 = math.pi / 4
exec_qpe2 = transpiler.transpile(qpe_demo, bindings={"theta": test_theta_2})
result_qpe2 = exec_qpe2.sample(transpiler.executor(), shots=1024).result()

print("=== QPE with theta = pi/4 (expected phase = 0.125) ===")
for value, count in result_qpe2.results:
    print(f"  Measured phase: {value}, Count: {count}")

# %% [markdown]
# Both results match the expected phases exactly. The `QFixed` type and
# `qmc.measure()` handle all the binary-fraction decoding automatically,
# so we get the phase value directly without manual bit manipulation.

# %% [markdown]
# ## 4. Algorithm Building Blocks
#
# The `qamomile.circuit.algorithm` module provides `@qkernel` building blocks
# for common circuit patterns. These are regular kernels
# and can be composed inside our own kernels **or executed standalone**.
#
# | Module | Examples |
# |--------|---------|
# | `qaoa` | `superposition_vector`, `qaoa_circuit`, `qaoa_state`, ... |
# | `basic` | `rx_layer`, `ry_layer`, `rz_layer`, `cz_entangling_layer` |
# | `fqaoa` | `fqaoa_layers`, `fqaoa_state`, ... |
#
# For the full API, see the API reference documentation.
#
# Contents may change between versions: frequently used patterns may be
# promoted to `stdlib`, and the boundary shifts as the ecosystem evolves.

# %% [markdown]
# ### Example: Using `superposition_vector`
#
# `superposition_vector` creates a uniform superposition of $n$ qubits.
# It is itself a `@qkernel`, so it can be **transpiled and executed
# standalone** or **called from within another kernel**.

# %%
from qamomile.circuit.algorithm import superposition_vector


@qmc.qkernel
def superposition_demo(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Create uniform superposition using the algorithm module."""
    q = superposition_vector(n)
    return qmc.measure(q)


# %%
exec_sup = transpiler.transpile(superposition_demo, bindings={"n": 4})
result_sup = exec_sup.sample(transpiler.executor(), shots=1000).result()

print("=== Superposition Vector (n=4) ===")
print("All 2^4=16 states should appear with roughly equal probability\n")
sorted_results = sorted(result_sup.results, key=lambda x: str(x[0]))
for value, count in sorted_results:
    percentage = count / 1000 * 100
    print(f"  {value}: {count} ({percentage:.1f}%)")

# %% [markdown]
# All algorithm building blocks follow the same pattern: they are ordinary
# `@qkernel` functions that we can compose into our own circuits.
# We will see them in action in the
# [QAOA optimization tutorial](../optimization/qaoa.ipynb).

# %% [markdown]
# ## 5. Summary
#
# ### Standard Library Functions
#
# | Function | Module | Input | Output | Description |
# |----------|--------|-------|--------|-------------|
# | `qmc.qft()` | `qamomile.circuit` | `Vector[Qubit]` | `Vector[Qubit]` | Quantum Fourier Transform |
# | `qmc.iqft()` | `qamomile.circuit` | `Vector[Qubit]` | `Vector[Qubit]` | Inverse QFT |
# | `qmc.qpe()` | `qamomile.circuit` | target, counting, unitary | `QFixed` | Quantum Phase Estimation |
#
# ### Key Classes
#
# | Class | Module | Purpose |
# |-------|--------|---------|
# | `QFT` | `qamomile.circuit.stdlib.qft` | QFT with strategy support |
# | `IQFT` | `qamomile.circuit.stdlib.qft` | IQFT with strategy support |
#
# ### Decomposition Strategies
#
# ```python
# from qamomile.circuit.stdlib.qft import QFT
#
# # List strategies
# QFT.list_strategies()  # ['standard', 'approximate', 'approximate_k2', ...]
#
# # Compare resources
# qft = QFT(8)
# resources = qft.get_resources_for_strategy("approximate")
#
# # Use a specific strategy
# qft_gate = QFT(n)
# results = qft_gate(q0, q1, ..., strategy="approximate")
# ```
#
# ### Next Tutorials
#
# - [Composite Gates](06_composite_gate.ipynb): Writing custom `CompositeGate`s and `@composite_gate`
# - [Our First Quantum Algorithm](07_first_algorithm.ipynb): The Deutsch-Jozsa algorithm
# - [Resource Estimation](09_resource_estimation.ipynb): Estimate gate counts and circuit depth
# - [QAOA](../optimization/qaoa.ipynb): Solve combinatorial optimization problems with QAOA

# %% [markdown]
# ## What We Learned
#
# - **Using `qmc.qft()`, `qmc.iqft()`, and `qmc.qpe()` from the standard library** — These ready-made building blocks handle Quantum Fourier Transform and Phase Estimation with a single function call.
# - **Decomposition strategies for controlling precision vs. gate count** — `QFT.list_strategies()` and `get_resources_for_strategy()` let us compare and select trade-offs like `"standard"` vs. `"approximate"`.
# - **IR-level vs. backend-level circuits** — `expand_composite=True` shows Qamomile's IR decomposition, which may differ from the actual transpiled circuit. Use `transpiler.to_circuit()` to inspect the backend-level result.
# - **Building Quantum Phase Estimation with `qmc.qpe()`** — `qmc.qpe()` takes a unitary kernel, target qubits, and counting qubits, returning a `QFixed` value decoded automatically.
# - **Overview of the `qamomile.circuit.algorithm` module** — Pre-built variational building blocks (QAOA layers, rotation layers, entangling layers) that compose into larger algorithms.
