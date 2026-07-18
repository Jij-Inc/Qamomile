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
# tags: [tutorial, resource-estimation]
# ---
#
# # Resource Estimation
#
# Before running a quantum kernel on real hardware, you may want to know its required resources, such as qubit count and gate count. Or, you may want to know the resource requirements of a quantum kernel you defined in the first place. Qamomile's `estimate_resources()` fills this need **without executing the qkernel**. It works with both concrete and symbolic (parameterized) qkernels.
#
# This chapter covers:
#
# - Basic resource estimation for fixed qkernels
# - Symbolic resource estimation for parameterized qkernels
# - The full `ResourceEstimate` field reference
# - Scaling analysis with `.substitute()`
# - Applying body-derived estimates to Shor order finding

# %%
# Install the latest Qamomile through pip!
# # !pip install "qamomile[qiskit,visualization]"

# %%
import qamomile.circuit as qmc

# %% [markdown]
# ## Estimating Resources of a Fixed QKernel
#
# For a qkernel with no parameters, `estimate_resources()` returns concrete numbers.


# %%
@qmc.qkernel
def fixed_circuit() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(3, name="q")

    q[0] = qmc.h(q[0])
    q[0], q[1] = qmc.cx(q[0], q[1])
    q[1], q[2] = qmc.cx(q[1], q[2])

    return qmc.measure(q)


# %%
fixed_circuit.draw()

# %%
est = fixed_circuit.estimate_resources()
print("qubits:", est.qubits)
assert est.qubits == 3
print("total gates:", est.gates.total)
assert est.gates.total == 3
print("single-qubit gates:", est.gates.single_qubit)
assert est.gates.single_qubit == 1
print("two-qubit gates:", est.gates.two_qubit)
assert est.gates.two_qubit == 2

# %% [markdown]
# ## Symbolic Resource Estimation
#
# When a qkernel has unbound parameters (like `n: qmc.UInt`), `estimate_resources()` returns **SymPy expressions** that show how costs scale with the parameter. This lets you analyze scaling without picking a specific value.


# %%
@qmc.qkernel
def scalable_circuit(n: qmc.UInt, theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")

    q = qmc.h(q)
    q = qmc.ry(q, theta)

    for i in qmc.range(n - 1):
        q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])

    return qmc.measure(q)


# %%
scalable_circuit.draw(n=4, fold_loops=False)

# %%
est = scalable_circuit.estimate_resources()
print("qubits:", est.qubits)
assert str(est.qubits) == "n"
print("total gates:", est.gates.total)
assert str(est.gates.total) == "2*n + Max(0, n - 1)"
print("single-qubit gates:", est.gates.single_qubit)
assert str(est.gates.single_qubit) == "2*n"
print("two-qubit gates:", est.gates.two_qubit)
assert str(est.gates.two_qubit) == "Max(0, n - 1)"
print("rotation gates:", est.gates.rotation_gates)
assert str(est.gates.rotation_gates) == "n"
print("parameters:", est.parameters)
assert set(est.parameters.keys()) == {"n"}

# %% [markdown]
# The output contains SymPy expressions like `n` for qubits and `2*n + Max(0, n - 1)` for total gates. These are exact — not approximations.
#
# The `Max(0, ...)` comes from the trip count of `qmc.range(n - 1)`. Since `n` is unbound, the estimator cannot assume `n >= 1`, so it clamps the count at zero rather than letting `n = 0` contribute `-1` iterations. Substituting any concrete `n >= 1` collapses the guard, which is why the totals below come out as plain integers.

# %% [markdown]
# ## `ResourceEstimate` Fields Reference
#
# | Field | Description |
# |-------|------------|
# | `est.qubits` | Logical qubit count |
# | `est.gates.total` | Total gate count |
# | `est.gates.single_qubit` | Single-qubit gates |
# | `est.gates.two_qubit` | Two-qubit gates |
# | `est.gates.multi_qubit` | Multi-qubit gates (3+ qubits) |
# | `est.gates.t_gates` | T-gate count |
# | `est.gates.clifford_gates` | Clifford gate count |
# | `est.gates.rotation_gates` | Rotation gate count |
# | `est.gates.oracle_calls` | Oracle call counts (dict by name) |
# | `est.parameters` | Dict of symbol names → SymPy symbols |
#
# All fields are SymPy expressions. For fixed qkernels they evaluate to plain integers.

# %% [markdown]
# ## Scaling Analysis with `.substitute()`
#
# The symbolic expressions tell you the *formula*, but often you want concrete numbers at specific sizes. Use `.substitute()` to evaluate:

# %%
for n_val in [4, 8, 16, 32]:
    c = est.substitute(n=n_val)
    print(
        f"n={n_val:2d}: {int(c.gates.total):>3} gates total, {int(c.gates.two_qubit):>2} two-qubit"
    )
    assert int(c.gates.total) == 3 * n_val - 1
    assert int(c.gates.two_qubit) == n_val - 1

# %% [markdown]
# ## Deriving Shor resources from the circuit body
#
# As a practical example, consider order finding, the quantum part of Shor's algorithm.
# `qmc.shor_order_finding()` accepts a base and modulus and returns one qkernel that can both run and estimate resources. The register width comes from `modulus.bit_length()`, so the returned kernel has no artificial `n` argument.

# %%
order_finding = qmc.shor_order_finding(base=2, modulus=15)
shor_est = order_finding.estimate_resources()

print("logical qubits:", shor_est.qubits)
print("total logical gates:", shor_est.gates.total)
print("estimate quality:", shor_est.quality)

assert shor_est.parameters == {}
assert shor_est.qubits == 21
assert str(shor_est.quality) == "upper_bound"

# %% [markdown]
# This implementation does not keep a `2*n`-qubit counting register at once. It measures and resets one phase qubit for reuse, applying semiclassical inverse-QFT phase corrections from the previously observed bits.
#
# With a fixed lookup-window width `w`, peak-live allocation in the circuit body gives `3*n + w + 7` logical qubits.
#
# | Purpose | Width |
# |---|---:|
# | Reused phase qubit | `1` |
# | Modular-value work register | `n` |
# | Modular-multiplication accumulator | `n` |
# | Window-lookup output | `n` |
# | Lookup address | `w` |
# | Carry, vent, overflow, reduction, domain, and enable | `6` |
#
# The default is `w=2`, giving `3*n + 9`. For modulus 15, `n=4`, which agrees with the estimate of 21 qubits. This is the peak-live allocation of the executed qkernel body, not an external cost formula registered with `estimate_resources()`.

# %%
import sympy as sp
from IPython.display import Math, display

symbolic_n, symbolic_w = sp.symbols("n w", integer=True, positive=True)
shor_width_formula = 3 * symbolic_n + symbolic_w + 7
display(Math(rf"N_\mathrm{{qubit}} = {sp.latex(shor_width_formula)}"))
assert shor_width_formula.subs({symbolic_n: 4, symbolic_w: 2}) == shor_est.qubits

# %% [markdown]
# The gate breakdown for modulus 15 is derived by traversing that same body to completion.

# %%
print("single-qubit gates:", shor_est.gates.single_qubit)
print("two-qubit gates:", shor_est.gates.two_qubit)
print("multi-qubit gates:", shor_est.gates.multi_qubit)
print("Toffoli gates:", shor_est.gates.toffoli)

assert shor_est.gates.total == (
    shor_est.gates.single_qubit
    + shor_est.gates.two_qubit
    + shor_est.gates.multi_qubit
)

# %% [markdown]
# `quality` is `upper_bound` because branches selected by mid-circuit measurements and classical feed-forward are counted conservatively. These are logical-circuit resources: they do not include decomposition to device-native gates, routing, error correction, or magic-state production.

# %% [markdown]
# ### Why the gate count is `O(n^3)`
#
# `qmc.modmul_const()` reads the source `w` bits at a time, looks up a classical multiple, and adds it into an accumulator. One modular multiplication has about `n / w` windows. The ripple-carry addition, constant subtraction, comparison, and conditional restoration used by every window are each `O(n)` gates. Constant addition and subtraction use [Gidney's carry-venting adder](https://arxiv.org/abs/2507.23079), which borrows the work register as dirty workspace rather than allocating another `n`-qubit register.
#
# Therefore, at fixed `w`, one modular multiplication is `O(n^2)`, and the default order-finding schedule performs `2*n` controlled multiplications for `O(n^3)` total cost. Semiclassical inverse-QFT feed-forward is `O(n^2)` and does not change the leading order. Including the lookup dependence gives approximately `O(2^w * n^3 / w)`.

# %% [markdown]
# ### Estimating a modular multiplication from the same body
#
# `qmc.modmul_const()` is the public primitive. FTQC arithmetic is specialized to a concrete problem instance before construction, so here we estimate the width-four kernel directly.

# %%
@qmc.qkernel
def modular_multiplier() -> qmc.Vector[qmc.Qubit]:
    reg = qmc.qubit_array(4, name="reg")
    return qmc.modmul_const(
        reg,
        multiplier=2,
        modulus=15,
        window_size=2,
    )


# %%
window_est = modular_multiplier.estimate_resources()
print("windowed arithmetic qubits:", window_est.qubits)
print("windowed arithmetic gates:", window_est.gates.total)
assert window_est.qubits == 3 * 4 + 2 + 7

# %% [markdown]
# In the standalone primitive, an internal control for the unconditional case takes the phase-qubit role, so it has the same `3*n + w + 7` width as the full order-finding circuit. For `x < modulus`, `modmul_const()` implements `|x> -> |a*x mod modulus>`; it leaves basis states outside that domain unchanged to preserve unitarity.

# %% [markdown]
# ### Ekerå–Håstad short-exponent schedule
#
# `qmc.ekera_hastad_factoring()` constructs the short-discrete-logarithm quantum stage of the [Ekerå–Håstad method](https://arxiv.org/abs/1702.00249) for factoring a product of two similarly sized primes. Qamomile sets `m = ceil(n / 2) + 1` and measures schedules of lengths `2*m` and `m` sequentially.
#
# Rather than retain two exponent registers coherently, it reuses the same phase qubit and arithmetic workspace. Its width is consequently the same `3*n + w + 7` as Shor's order finding; the number of controlled modular multiplications differs. The first `2*m` little-endian return bits are the long schedule and the remaining `m` bits are the short schedule.

# %%
short_dlp = qmc.ekera_hastad_factoring(
    generator=2,
    modulus=5,
    window_size=2,
)
short_dlp_est = short_dlp.estimate_resources()

print("Ekerå–Håstad logical qubits:", short_dlp_est.qubits)
print("Ekerå–Håstad total logical gates:", short_dlp_est.gates.total)
assert short_dlp_est.qubits == 3 * 3 + 2 + 7
assert len(short_dlp.output_types) == 9

# %% [markdown]
# ## Summary
#
# - `estimate_resources()` reports qubit and gate costs without executing.
# - For parameterized qkernels, results are SymPy expressions showing exact scaling.
# - Use `.substitute(n=...)` to evaluate at specific sizes and check feasibility.
# - Concrete FTQC factories can still expose body-derived width and gate estimates after specialization.
#
# **Next**: [Execution Models](06_execution_models.ipynb) — `sample()` vs `run()`, observables, and bit ordering.
