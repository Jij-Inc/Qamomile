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
# # Hamiltonian Simulation with Suzuki–Trotter (Rabi oscillation)
#
# Simulating time evolution $e^{-iHt}$ under a Hamiltonian $H$ is one of the
# canonical applications of a quantum computer. When $H$ decomposes as a sum of
# non-commuting terms $H = \sum_j H_j$, we cannot exponentiate it in one shot;
# the standard remedy is a **Trotter–Suzuki product formula**, which
# approximates $e^{-iHt}$ by interleaving short evolutions of each $H_j$.
#
# This tutorial builds the classic example end-to-end in Qamomile:
#
# - **Model**: single-qubit Rabi oscillation with $H = \tfrac{\omega}{2} Z +
#   \tfrac{\Omega}{2} X$. The exact propagator is known in closed form, which
#   gives us a clean benchmark.
# - **Three splittings**: 1st-order Lie–Trotter (`S1`), 2nd-order symmetric
#   Strang (`S2`), and 4th-order Suzuki (`S4`).
# - **Convergence plot**: fidelity error vs step size on a log-log axis, with
#   the expected power-law slopes.
# - **Discussion**: what works cleanly and what does not in the current
#   `@qkernel` frontend.
#
# What you will learn:
#
# - How to express a time-stepping loop with `Vector[Observable]` and
#   `pauli_evolve`.
# - How to compose higher-order Trotter formulas by **calling one `@qkernel`
#   from another** — the sub-kernel pattern from Tutorial 06 applied to a real
#   numerical recipe.
# - The order of the method is directly observable in a simple log-log plot,
#   and Qamomile's symbolic parameter binding lets you sweep step size without
#   re-transpiling the structure.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile as qk_transpile
from qiskit_aer import AerSimulator
from scipy.linalg import expm

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.qiskit import QiskitTranspiler

# %% [markdown]
# ## The Rabi Hamiltonian
#
# A single two-level system driven on resonance is governed by
#
# $$ H = \underbrace{\tfrac{\omega}{2} Z}_{H_z} +
#        \underbrace{\tfrac{\Omega}{2} X}_{H_x}. $$
#
# Because $[Z, X] \neq 0$, splitting $H$ into $H_z$ and $H_x$ introduces a
# Trotter error that is easy to see even on a single qubit. For starting state
# $|0\rangle$, the excitation probability oscillates as
# $P_{|1\rangle}(t) = (\Omega/E)^2 \sin^2(Et/2)$ with $E = \sqrt{\omega^2 +
# \Omega^2}$.
#
# In Qamomile we build $H_z$ and $H_x$ as two separate `Observable`s and pack
# them into a Python list. A `@qkernel` declares that list as
# `Vector[Observable]`; binding the list at transpile time unrolls any
# iteration over `Hs.shape[0]` into per-term `pauli_evolve` calls.

# %%
omega = 1.2
Omega = 0.8
T = 1.5

Hz = 0.5 * omega * qm_o.Z(0)
Hx = 0.5 * Omega * qm_o.X(0)
Hs = [Hz, Hx]

# %% [markdown]
# ## Exact reference state
#
# A 2x2 matrix exponential gives the exact state $|\psi(T)\rangle =
# e^{-iHT}|0\rangle$. We compare every Trotter approximation against this
# vector via the **fidelity error** $1 - |\langle\psi_\text{exact}|\psi_\text{trotter}\rangle|$.

# %%
X_mat = np.array([[0, 1], [1, 0]], dtype=complex)
Z_mat = np.array([[1, 0], [0, -1]], dtype=complex)
H_mat = 0.5 * omega * Z_mat + 0.5 * Omega * X_mat
sv_exact = expm(-1j * T * H_mat) @ np.array([1.0, 0.0], dtype=complex)


def statevector(circuit) -> np.ndarray:
    """Strip measurements, lower PauliEvolutionGate, and read out the state.

    The default `pauli_evolve` emitter produces a `PauliEvolutionGate`, which
    is not in AerSimulator's native basis. We run a shallow Qiskit transpile
    pass to expand it into elementary rotations."""
    stripped = QuantumCircuit(*circuit.qregs)
    for instr in circuit.data:
        if instr.operation.name not in ("measure", "save_statevector"):
            stripped.append(instr)
    stripped = qk_transpile(
        stripped,
        basis_gates=["u", "cx", "rx", "ry", "rz", "h", "p", "sx", "x", "y", "z"],
    )
    stripped.save_statevector()
    sim = AerSimulator(method="statevector")
    return np.asarray(sim.run(stripped).result().get_statevector())


# %% [markdown]
# ## `S1`: first-order Lie–Trotter
#
# The simplest split is
#
# $$ U_{S_1}(\Delta t) = e^{-i H_z \Delta t} e^{-i H_x \Delta t}, $$
#
# applied $N$ times for a total evolution time $T = N \Delta t$. We express one
# step as a helper `@qkernel` that takes the qubit register through and
# evolves it under each term in turn. The outer kernel just repeats the step.

# %%


@qmc.qkernel
def s1_step(
    q: qmc.Vector[qmc.Qubit], Hs: qmc.Vector[qmc.Observable], dt: qmc.Float
) -> qmc.Vector[qmc.Qubit]:
    q = qmc.pauli_evolve(q, Hs[0], dt)
    q = qmc.pauli_evolve(q, Hs[1], dt)
    return q


@qmc.qkernel
def rabi_s1(
    Hs: qmc.Vector[qmc.Observable], dt: qmc.Float, n_steps: qmc.UInt
) -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(1, "q")
    for _ in qmc.range(n_steps):
        q = s1_step(q, Hs, dt)
    return q


@qmc.qkernel
def rabi_s1_meas(
    Hs: qmc.Vector[qmc.Observable], dt: qmc.Float, n_steps: qmc.UInt
) -> qmc.Vector[qmc.Bit]:
    q = rabi_s1(Hs, dt, n_steps)
    return qmc.measure(q)


# %% [markdown]
# ## `S2`: symmetric (Strang) splitting
#
# Symmetrising the step around the middle term cancels the leading error:
#
# $$ U_{S_2}(\Delta t) = e^{-i H_z \Delta t/2}\, e^{-i H_x \Delta t}\,
#    e^{-i H_z \Delta t/2}. $$
#
# The step kernel stays short — three `pauli_evolve` calls — and the outer
# kernel is identical to the `S1` case modulo the name of the step.

# %%


@qmc.qkernel
def s2_step(
    q: qmc.Vector[qmc.Qubit], Hs: qmc.Vector[qmc.Observable], dt: qmc.Float
) -> qmc.Vector[qmc.Qubit]:
    q = qmc.pauli_evolve(q, Hs[0], 0.5 * dt)
    q = qmc.pauli_evolve(q, Hs[1], dt)
    q = qmc.pauli_evolve(q, Hs[0], 0.5 * dt)
    return q


@qmc.qkernel
def rabi_s2(
    Hs: qmc.Vector[qmc.Observable], dt: qmc.Float, n_steps: qmc.UInt
) -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(1, "q")
    for _ in qmc.range(n_steps):
        q = s2_step(q, Hs, dt)
    return q


@qmc.qkernel
def rabi_s2_meas(
    Hs: qmc.Vector[qmc.Observable], dt: qmc.Float, n_steps: qmc.UInt
) -> qmc.Vector[qmc.Bit]:
    q = rabi_s2(Hs, dt, n_steps)
    return qmc.measure(q)


# %% [markdown]
# ## `S4`: 4th-order Suzuki, explicit
#
# The Suzuki recursion constructs higher-order formulas by chaining `S2` at
# rescaled step sizes:
#
# $$ U_{S_4}(\Delta t) = U_{S_2}(p \Delta t)^2\,
#    U_{S_2}\bigl((1 - 4p) \Delta t\bigr)\, U_{S_2}(p \Delta t)^2, $$
#
# where $p = 1/(4 - 4^{1/3}) \approx 0.4145$ is chosen so that the 3rd-order
# error cancels. Because the coefficient and the midpoint weight are plain
# Python floats, we compute them outside any kernel and let them flow in as
# `Float` parameters.
#
# The kernel below is the direct translation: it calls `s2_step` five times
# with the rescaled `dt`. Qamomile inlines each sub-kernel call, producing a
# flat circuit.

# %%
P1 = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
W = 1.0 - 4.0 * P1


@qmc.qkernel
def s4_step(
    q: qmc.Vector[qmc.Qubit], Hs: qmc.Vector[qmc.Observable], dt: qmc.Float
) -> qmc.Vector[qmc.Qubit]:
    q = s2_step(q, Hs, P1 * dt)
    q = s2_step(q, Hs, P1 * dt)
    q = s2_step(q, Hs, W * dt)
    q = s2_step(q, Hs, P1 * dt)
    q = s2_step(q, Hs, P1 * dt)
    return q


@qmc.qkernel
def rabi_s4(
    Hs: qmc.Vector[qmc.Observable], dt: qmc.Float, n_steps: qmc.UInt
) -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(1, "q")
    for _ in qmc.range(n_steps):
        q = s4_step(q, Hs, dt)
    return q


@qmc.qkernel
def rabi_s4_meas(
    Hs: qmc.Vector[qmc.Observable], dt: qmc.Float, n_steps: qmc.UInt
) -> qmc.Vector[qmc.Bit]:
    q = rabi_s4(Hs, dt, n_steps)
    return qmc.measure(q)


# %% [markdown]
# ## Quick sanity check at `N = 8`
#
# Before the full sweep, transpile each kernel once and confirm the statevectors
# land in the right ball park.

# %%
tr = QiskitTranspiler()
N_demo = 8
kernels = {"S1": rabi_s1_meas, "S2": rabi_s2_meas, "S4": rabi_s4_meas}

for name, ker in kernels.items():
    exe = tr.transpile(ker, bindings={"Hs": Hs, "dt": T / N_demo, "n_steps": N_demo})
    sv = statevector(exe.compiled_quantum[0].circuit)
    err = 1.0 - abs(np.vdot(sv_exact, sv))
    print(f"{name} at N={N_demo}: fidelity error = {err:.3e}")

# %% [markdown]
# ## Convergence sweep
#
# We now sweep the number of Trotter steps `N` and plot the fidelity error
# against the step size $\Delta t = T / N$ on a log-log axis. The expected
# slopes are:
#
# | Formula | Local error | Global norm error | Fidelity error ($1 -$ overlap) |
# |---------|-------------|-------------------|---------------------------------|
# | `S1`    | $O(\Delta t^2)$ | $O(\Delta t)$ | $O(\Delta t^2)$ |
# | `S2`    | $O(\Delta t^3)$ | $O(\Delta t^2)$ | $O(\Delta t^4)$ |
# | `S4`    | $O(\Delta t^5)$ | $O(\Delta t^4)$ | $O(\Delta t^8)$ |
#
# Fidelity error scales as the *square* of the state-norm error, because
# $1 - |\langle a | b \rangle| \approx \tfrac{1}{2}\lVert a - b \rVert^2$ when
# the two vectors are close. That is why the plot below shows slopes of 2, 4,
# and 8 rather than 1, 2, and 4.

# %%
Ns = np.array([2, 4, 8, 16, 32, 64])

errors = {"S1": [], "S2": [], "S4": []}
for name, ker in kernels.items():
    for N in Ns:
        exe = tr.transpile(
            ker, bindings={"Hs": Hs, "dt": T / int(N), "n_steps": int(N)}
        )
        sv = statevector(exe.compiled_quantum[0].circuit)
        errors[name].append(1.0 - abs(np.vdot(sv_exact, sv)))

errors = {k: np.asarray(v) for k, v in errors.items()}
dts = T / Ns


# %%
# Fit log-log slopes. S4 hits machine precision quickly, so only fit the
# first few points where the signal is above floor.
def fit_slope(dts, errs, n_points):
    return np.polyfit(np.log(dts[:n_points]), np.log(errs[:n_points]), 1)[0]


slope_s1 = fit_slope(dts, errors["S1"], len(Ns))
slope_s2 = fit_slope(dts, errors["S2"], len(Ns))
slope_s4 = fit_slope(dts, errors["S4"], 3)
print(f"Fitted slopes:  S1 = {slope_s1:.2f}  S2 = {slope_s2:.2f}  S4 = {slope_s4:.2f}")

# Guard the expected orders so doc-tests catch regressions in pauli_evolve.
assert 1.7 < slope_s1 < 2.3, slope_s1
assert 3.7 < slope_s2 < 4.3, slope_s2
assert 7.0 < slope_s4 < 9.0, slope_s4

# %%
fig, ax = plt.subplots(figsize=(6, 4))
markers = {"S1": "o", "S2": "s", "S4": "^"}
for name in ("S1", "S2", "S4"):
    ax.loglog(dts, errors[name], marker=markers[name], label=name)
# Floor at machine precision — the S4 curve flattens here.
ax.axhline(1e-15, color="grey", linestyle=":", linewidth=0.8, label="float64 floor")
ax.set_xlabel(r"step size $\Delta t = T / N$")
ax.set_ylabel(
    r"fidelity error $1 - |\langle \psi_{\rm exact} | \psi_{\rm trotter} \rangle|$"
)
ax.set_title("Trotter convergence on Rabi oscillation")
ax.grid(True, which="both", linewidth=0.3)
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# The lines on the plot have slopes $\approx 2$, $4$, and $8$, matching the
# fidelity-error orders in the table above. `S4` hits the float64 floor
# already at $N = 16$: beyond that the curve bends because the exact and the
# Trotter state agree to machine precision.

# %% [markdown]
# ## What about a recursive `S(k)` kernel?
#
# The Suzuki recursion is naturally a self-referential function
# `S(k, dt) = S(k-1, p dt) ...`. The Qamomile equivalent would look like:
#
# ```python
# @qmc.qkernel
# def suzuki(
#     k: qmc.UInt, q: qmc.Vector[qmc.Qubit],
#     Hs: qmc.Vector[qmc.Observable], dt: qmc.Float,
# ) -> qmc.Vector[qmc.Qubit]:
#     if k == 0:
#         q = s2_step(q, Hs, dt)
#     else:
#         q = suzuki(k - 1, q, Hs, P1 * dt)
#         q = suzuki(k - 1, q, Hs, P1 * dt)
#         q = suzuki(k - 1, q, Hs,  W * dt)
#         q = suzuki(k - 1, q, Hs, P1 * dt)
#         q = suzuki(k - 1, q, Hs, P1 * dt)
#     return q
# ```
#
# Transpiling this today raises `RecursionError` inside the frontend tracer.
# The cause is mechanical: when a `@qkernel` contains an `if` over a `UInt`,
# the tracer descends into **both** branches to build the IR, and the `else`
# branch calls `suzuki` again, which itself contains the same `if`, and so on
# without a base case ever reached at trace time. The `k` value is bound at
# `transpile`-time, but the branch-folding happens later in the pipeline —
# tracing sees a symbol, not `0`.
#
# For now the idiomatic way to write arbitrary-order Suzuki formulas is the
# explicit pattern above: compose `S2` into `S4`, and if you need `S6` or
# higher, compose `S4` into `S6` the same way. A future change that lowers
# constant-`UInt` branches *before* tracing the untaken side would let the
# recursion work directly; until then, explicit composition is both the
# supported path and, for fixed order, essentially free at runtime since the
# transpiler inlines each call.

# %% [markdown]
# ## Summary
#
# - **Model**: a single-qubit Rabi Hamiltonian $H = H_z + H_x$ whose non-zero
#   commutator makes Trotter error measurable.
# - **`Vector[Observable]` + `pauli_evolve`**: the natural primitive for
#   time-stepping. Binding the Hamiltonian list at transpile time expands any
#   iteration over `Hs.shape[0]` into per-term evolutions.
# - **Sub-kernel composition**: `S4` is `S2` called five times with rescaled
#   step sizes. No special-cased plumbing — just normal function calls from
#   Tutorial 06 applied to numerical recipes.
# - **Convergence**: fidelity-error slopes of 2, 4, 8 on log-log match
#   textbook Trotter orders, and the symbolic `dt` / `n_steps` parameters let
#   you sweep step sizes without rebuilding the circuit structure.
# - **Current limitation**: self-recursive `@qkernel`s are not yet supported.
#   Express higher-order formulas by explicit composition of lower-order ones.
