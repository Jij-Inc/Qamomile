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
#     name: python3
# ---

# %% [markdown]
# # Hamiltonian Simulation with Suzuki–Trotter (Rabi oscillation)
#
# Simulating time evolution $e^{-iHt}$ under a Hamiltonian $H = A + B$ where
# $[A, B] \neq 0$ is one of the canonical applications of a quantum computer.
# Because the two terms do not commute, we cannot write
# $e^{-i(A+B)t} = e^{-iAt}\,e^{-iBt}$ exactly; instead we approximate the
# evolution by **interleaving short evolutions** of each term.  The family of
# such approximations — Lie–Trotter at first order, Strang at second, and the
# Suzuki *fractal* recursion at higher even orders — is what this tutorial
# builds end-to-end in Qamomile.
#
# This tutorial is deliberately written *from scratch*: we build $S_1$, $S_2$,
# and the higher-order Suzuki formulas as ordinary `@qkernel`s, derive the
# coefficients we need, and verify the convergence orders empirically against
# the exact propagator.  Two ways of expressing the recursion are compared:
#
# - **Self-recursive `@qkernel`** — the natural mathematical translation, but
#   the current Qamomile frontend rejects it with a clear error and tells you
#   why.  We trigger the error on purpose to look at the message.
# - **Python-level recursion** — define a Python function that returns a
#   freshly built `@qkernel`, and let Python's normal recursion bottom out at
#   `S_2`.  This is the supported pattern and the one to reach for whenever a
#   formula is "recursive in some integer order".

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
# Trotter error that is easy to see even on a single qubit.  For the starting
# state $|0\rangle$, the excitation probability oscillates as
# $P_{|1\rangle}(t) = (\Omega/E)^2 \sin^2(Et/2)$ with
# $E = \sqrt{\omega^2 + \Omega^2}$.
#
# In Qamomile we build $H_z$ and $H_x$ as two separate `Observable`s and pack
# them into a Python list.  A `@qkernel` declares that list as
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
# A 2x2 matrix exponential gives the exact state
# $|\psi(T)\rangle = e^{-iHT}|0\rangle$.  Each Trotter approximation is judged
# against this vector via the **fidelity error**
# $1 - |\langle\psi_\text{exact}|\psi_\text{trotter}\rangle|$.

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
# ## $S_1$: 1st-order Lie–Trotter
#
# The simplest split is
#
# $$ S_1(\Delta t) = e^{-i H_z \Delta t}\, e^{-i H_x \Delta t}, $$
#
# applied $N$ times for a total evolution time $T = N \Delta t$.  Per step the
# error is $O(\Delta t^2)$; integrated over $N$ steps the global state-norm
# error is $O(\Delta t)$.
#
# We write one step as a small helper `@qkernel`.  The qubit register is
# threaded through, evolved under $H_z$ and then $H_x$.  The outer kernel
# repeats the step $N$ times — the count `n_steps` is a `UInt` parameter, so
# the same kernel transpiles for any $N$.

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
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(1, "q")
    for _ in qmc.range(n_steps):
        q = s1_step(q, Hs, dt)
    return qmc.measure(q)


# %% [markdown]
# ## $S_2$: symmetric (Strang) splitting
#
# Symmetrising the step around the middle term cancels the leading error:
#
# $$ S_2(\Delta t) = e^{-i H_z \Delta t/2}\, e^{-i H_x \Delta t}\,
#    e^{-i H_z \Delta t/2}. $$
#
# The local error drops to $O(\Delta t^3)$ and the global state-norm error to
# $O(\Delta t^2)$.  The step kernel is just three `pauli_evolve` calls.

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
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(1, "q")
    for _ in qmc.range(n_steps):
        q = s2_step(q, Hs, dt)
    return qmc.measure(q)


# %% [markdown]
# ## Higher orders: the Suzuki fractal recursion
#
# Masuo Suzuki showed that an arbitrary even-order approximation can be built
# **recursively** from $S_2$ by nesting five rescaled copies at each level:
#
# $$ S_{2k}(x) = S_{2k-2}(p_k x)^2 \,
#                S_{2k-2}\bigl((1 - 4 p_k) x\bigr) \,
#                S_{2k-2}(p_k x)^2, $$
#
# with the level-specific coefficient
#
# $$ p_k = \frac{1}{4 - 4^{1/(2k-1)}}. $$
#
# $p_k$ is chosen so that the $(2k-1)$-th-order error of the lower formula
# cancels, leaving a local error of $O(x^{2k+1})$ per step.  **The
# coefficient depends on $k$** — a single constant reused at every level
# does *not* give the Suzuki fractal.  Concretely,
#
# - $k=2$ (4th order): $p_2 = 1/(4 - 4^{1/3}) \approx 0.4145$,
# - $k=3$ (6th order): $p_3 = 1/(4 - 4^{1/5}) \approx 0.3731$,
# - $k=4$ (8th order): $p_4 = 1/(4 - 4^{1/7}) \approx 0.3596$.
#
# We will build $S_4$ explicitly first, then write a Python function that
# applies the recursion for any even order.

# %% [markdown]
# ### $S_4$ written out
#
# Setting $k = 2$ in the recursion gives the standard 4th-order formula:
#
# $$ S_4(\Delta t) = S_2(p_2 \Delta t)^2 \, S_2((1 - 4 p_2)\Delta t) \,
#    S_2(p_2 \Delta t)^2. $$
#
# In Qamomile, "squared" just means two back-to-back calls: the formula
# becomes five `s2_step` calls with carefully chosen step sizes.  The
# coefficients are plain Python floats, so we evaluate them once and let
# them flow into the kernel as constants.

# %%
P2 = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
W2 = 1.0 - 4.0 * P2


@qmc.qkernel
def s4_step(
    q: qmc.Vector[qmc.Qubit], Hs: qmc.Vector[qmc.Observable], dt: qmc.Float
) -> qmc.Vector[qmc.Qubit]:
    q = s2_step(q, Hs, P2 * dt)
    q = s2_step(q, Hs, P2 * dt)
    q = s2_step(q, Hs, W2 * dt)
    q = s2_step(q, Hs, P2 * dt)
    q = s2_step(q, Hs, P2 * dt)
    return q


@qmc.qkernel
def rabi_s4(
    Hs: qmc.Vector[qmc.Observable], dt: qmc.Float, n_steps: qmc.UInt
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(1, "q")
    for _ in qmc.range(n_steps):
        q = s4_step(q, Hs, dt)
    return qmc.measure(q)


# %% [markdown]
# ## Two ways to write the recursion
#
# Writing $S_6$, $S_8$, … by hand is tedious and easy to get wrong (the most
# common slip is reusing $p_2$ at every level).  A natural impulse is to
# express the recursion *inside* a `@qkernel`:
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
#         q = suzuki(k - 1, q, Hs, p_k * dt)
#         ...
#     return q
# ```
#
# This pattern is rejected by the current frontend.  The reason is mechanical:
# `if` over a `UInt` is lowered as an `IfOperation` with both branches traced
# at build time (so that constant folding in a later pass can pick the right
# one).  The `else` branch contains a self-call, which would re-enter the same
# kernel's tracer without ever bottoming out.  We can confirm this by
# attempting the construction inside a `try/except`:

# %%


@qmc.qkernel
def _suzuki_self_recursive(
    k: qmc.UInt,
    q: qmc.Vector[qmc.Qubit],
    Hs: qmc.Vector[qmc.Observable],
    dt: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    if k == 0:
        q = s2_step(q, Hs, dt)
    else:
        q = _suzuki_self_recursive(k - 1, q, Hs, P2 * dt)
        q = _suzuki_self_recursive(k - 1, q, Hs, P2 * dt)
        q = _suzuki_self_recursive(k - 1, q, Hs, W2 * dt)
        q = _suzuki_self_recursive(k - 1, q, Hs, P2 * dt)
        q = _suzuki_self_recursive(k - 1, q, Hs, P2 * dt)
    return q


@qmc.qkernel
def _outer_suzuki(
    k: qmc.UInt, Hs: qmc.Vector[qmc.Observable], dt: qmc.Float
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(1, "q")
    q = _suzuki_self_recursive(k, q, Hs, dt)
    return qmc.measure(q)


try:
    _outer_suzuki.build(k=1, Hs=Hs, dt=T / 4)
except Exception as e:  # FrontendTransformError
    print(f"{type(e).__name__}:")
    print(str(e))


# %% [markdown]
# The frontend bails out with an actionable
# `FrontendTransformError("Self-recursive @qkernel ... is not supported …")`
# rather than a Python `RecursionError`, and the message points at the
# supported alternative: **build the recursion at Python level**.  Each
# call to a `@qkernel` from inside another is inlined by the transpiler, so
# Python-level composition produces a flat circuit at the end.

# %% [markdown]
# ### Python-level recursive builder
#
# A short Python function that returns a `@qkernel` does the job.  At each
# call it computes the level-specific $p_k$, fetches the lower-order kernel
# by recursing on a Python `int`, and decorates a closure that calls the
# lower kernel five times with the rescaled step.  When the recursion bottoms
# out at order 2 we just return `s2_step`.

# %%


def make_suzuki_step(order: int) -> qmc.QKernel:
    """Return a ``@qkernel`` implementing :math:`S_{\\text{order}}`.

    ``order`` must be an even integer >= 2.  The construction recurses on
    a Python ``int``, so the base case is reached at Python time and the
    decorator is applied to a fully expanded closure.
    """
    if order == 2:
        return s2_step

    lower = make_suzuki_step(order - 2)
    p = 1.0 / (4.0 - 4.0 ** (1.0 / (order - 1)))
    w = 1.0 - 4.0 * p

    @qmc.qkernel
    def step(
        q: qmc.Vector[qmc.Qubit],
        Hs: qmc.Vector[qmc.Observable],
        dt: qmc.Float,
    ) -> qmc.Vector[qmc.Qubit]:
        q = lower(q, Hs, p * dt)
        q = lower(q, Hs, p * dt)
        q = lower(q, Hs, w * dt)
        q = lower(q, Hs, p * dt)
        q = lower(q, Hs, p * dt)
        return q

    return step


# %% [markdown]
# `make_suzuki_step(4)` should agree with the hand-written `s4_step`, and
# `make_suzuki_step(6)` gives us $S_6$ for free with the correct $p_3$.

# %%
s4_from_helper = make_suzuki_step(4)
s6_from_helper = make_suzuki_step(6)


@qmc.qkernel
def rabi_s6(
    Hs: qmc.Vector[qmc.Observable], dt: qmc.Float, n_steps: qmc.UInt
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(1, "q")
    for _ in qmc.range(n_steps):
        q = s6_from_helper(q, Hs, dt)
    return qmc.measure(q)


# %% [markdown]
# ## Quick sanity check at $N = 8$
#
# Before the convergence sweep, transpile each kernel once and confirm the
# statevectors land in the right ball park.

# %%
tr = QiskitTranspiler()
N_demo = 8
kernels = {
    "S1": rabi_s1,
    "S2": rabi_s2,
    "S4": rabi_s4,
    "S6": rabi_s6,
}

for name, ker in kernels.items():
    exe = tr.transpile(ker, bindings={"Hs": Hs, "dt": T / N_demo, "n_steps": N_demo})
    sv = statevector(exe.compiled_quantum[0].circuit)
    err = 1.0 - abs(np.vdot(sv_exact, sv))
    print(f"{name} at N={N_demo}: fidelity error = {err:.3e}")

# %% [markdown]
# ## Convergence sweep
#
# We now sweep the number of Trotter steps $N$ and plot the fidelity error
# against the step size $\Delta t = T / N$ on a log-log axis.  The expected
# slopes are:
#
# | Formula | Local error | Global norm error | Fidelity error ($1 -$ overlap) |
# |---------|-------------|-------------------|---------------------------------|
# | $S_1$   | $O(\Delta t^2)$ | $O(\Delta t)$   | $O(\Delta t^2)$  |
# | $S_2$   | $O(\Delta t^3)$ | $O(\Delta t^2)$ | $O(\Delta t^4)$  |
# | $S_4$   | $O(\Delta t^5)$ | $O(\Delta t^4)$ | $O(\Delta t^8)$  |
# | $S_6$   | $O(\Delta t^7)$ | $O(\Delta t^6)$ | $O(\Delta t^{12})$ |
#
# Fidelity error scales as the *square* of the state-norm error, because
# $1 - |\langle a | b \rangle| \approx \tfrac{1}{2}\lVert a - b \rVert^2$ when
# the two vectors are close — so the plot below shows slopes of 2, 4, 8
# rather than 1, 2, 4.

# %%
Ns = np.array([2, 4, 8, 16, 32, 64])

errors = {name: [] for name in kernels}
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
def fit_slope(dts, errs, n_points):
    return np.polyfit(np.log(dts[:n_points]), np.log(errs[:n_points]), 1)[0]


slope_s1 = fit_slope(dts, errors["S1"], len(Ns))
slope_s2 = fit_slope(dts, errors["S2"], len(Ns))
slope_s4 = fit_slope(dts, errors["S4"], 3)
print(f"Fitted slopes:  S1 = {slope_s1:.2f}  S2 = {slope_s2:.2f}  S4 = {slope_s4:.2f}")
print(f"S6 fidelity error at largest dt: {errors['S6'][0]:.3e}")

# Guard the expected orders so doc-tests catch regressions in pauli_evolve.
assert 1.7 < slope_s1 < 2.3, slope_s1
assert 3.7 < slope_s2 < 4.3, slope_s2
assert 7.0 < slope_s4 < 9.0, slope_s4
# S6 sits at the float64 floor on this 1-qubit problem; just check it is
# nowhere near S4's leading-error magnitude at the same dt.
assert abs(errors["S6"][0]) < 1e-10, errors["S6"][0]

# %%
fig, ax = plt.subplots(figsize=(6, 4))
markers = {"S1": "o", "S2": "s", "S4": "^", "S6": "D"}
for name in ("S1", "S2", "S4", "S6"):
    ax.loglog(dts, errors[name], marker=markers[name], label=name)
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
# The lines on the plot have slopes $\approx 2$, $4$, $8$, matching the
# fidelity-error orders in the table above.  $S_4$ hits the float64 floor
# already at $N = 16$; $S_6$ is at the floor across the entire sweep, so its
# line appears flat (the expected $\Delta t^{12}$ slope is not resolvable on
# this 1-qubit problem in double precision).

# %% [markdown]
# ## Why per-level $p_k$ matters
#
# A tempting shortcut is to write the recursion with a single fixed
# coefficient — typically $p_2 = 1/(4 - 4^{1/3})$ — at every nesting level.
# That does **not** produce the Suzuki fractal: the cancellation argument
# behind Suzuki's construction picks $p_k$ specifically to kill the
# $(2k-1)$-th-order error of $S_{2k-2}$, so reusing $p_2$ when stacking on
# top of $S_4$ leaves a non-zero error term at order $5$, and the resulting
# formula is no better than $S_4$.  `make_suzuki_step` recomputes $p_k$ at
# every level via ``p = 1.0 / (4.0 - 4.0 ** (1.0 / (order - 1)))``.

# %% [markdown]
# ## Summary
#
# - **Model**: a single-qubit Rabi Hamiltonian $H = H_z + H_x$ whose non-zero
#   commutator makes Trotter error measurable.
# - **`Vector[Observable]` + `pauli_evolve`**: the natural primitive for
#   time-stepping; binding the Hamiltonian list at transpile time expands
#   any iteration over `Hs.shape[0]` into per-term evolutions.
# - **Suzuki fractal**: $S_{2k}$ is built by nesting five rescaled copies of
#   $S_{2k-2}$, using the level-specific coefficient
#   $p_k = 1/(4 - 4^{1/(2k-1)})$; reusing one constant across levels is
#   wrong.
# - **Recursion idiom**: write the recursion at Python level (a function
#   that returns a fresh `@qkernel` per order).  A self-recursive
#   `@qkernel` is rejected by the frontend with a clear
#   `FrontendTransformError` that points at this same pattern.
# - **Convergence**: fidelity-error slopes of 2, 4, 8 on log-log match
#   textbook Trotter orders, and the symbolic `dt` / `n_steps` parameters
#   let you sweep step sizes without rebuilding the circuit structure.
