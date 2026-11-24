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
# tags: [algorithm, primitive, simulation]
# ---
#
# # Hamiltonian Simulation with Suzuki–Trotter Decomposition: Rabi Oscillations
#
# The time evolution of a quantum system with Hamiltonian $H$ is described by
# $U(t) = e^{-iHt}$, where $\hbar = 1$.
# Implementing this evolution as a sequence of quantum gates is one of the main
# applications of quantum computers.
# When the Hamiltonian consists of noncommuting terms, such as $H = H_z + H_x$,
# we generally have $e^{-i(H_z + H_x)t} \neq e^{-iH_z t}e^{-iH_x t}$.
# Trotter–Suzuki product formulas approximate the evolution by alternating short
# evolutions under the individual terms.
# The error decreases by taking smaller steps (first-order Lie–Trotter), making
# each step symmetric (second-order Strang), or recursively nesting symmetric
# steps using Suzuki's construction (any even order).\
# In this article, we explore Trotter approximations for a one-qubit Rabi
# Hamiltonian. We start with $S_1$ and $S_2$, then use the standard-library
# `suzuki_trotter` function for higher-order formulas.

# %%
# Install the latest Qamomile with pip.
# # !pip install qamomile

# %%
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

import qamomile as qm
from qamomile.stdlib import suzuki_trotter

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
FORMULA_STEPS = 1 if docs_test_mode else 4
SAMPLE_SHOTS = 1 if docs_test_mode else 2048

# %% [markdown]
# ## Problem Settings
#
# ### The Rabi Hamiltonian
#
# Consider the effective Hamiltonian of a driven two-level system in a rotating
# frame:
#
# $$
# H
# = \underbrace{\frac{\omega}{2} Z}_{H_z} + \underbrace{\frac{\Omega}{2} X}_{H_x}.
# $$
#
# Here, $Z$ and $X$ are Pauli operators, $\omega$ is the detuning, and $\Omega$
# is the drive strength (Rabi frequency).
# Since $[Z, X] \neq 0$, splitting $H$ into $H_z$ and $H_x$ introduces a
# Trotter error.


# %%
omega = 1.2
drive = 0.8
evolution_time = 1.5

drift_hamiltonian = 0.5 * omega * qm.Z(0)
drive_hamiltonian = 0.5 * drive * qm.X(0)
hamiltonians = (drift_hamiltonian, drive_hamiltonian)

commutator = qm.commutator(*hamiltonians)
expected_commutator = 0.5j * omega * drive * qm.Y(0)

assert commutator == expected_commutator
print("[H_z, H_x] =", commutator)

# %% [markdown]
# The commutator is defined as $[A, B] = AB - BA$.
# For the Pauli operators, $[Z, X] = 2iY$, so
#
# $$
# [H_z, H_x]
# = \frac{\omega \Omega}{4}[Z, X]
# = i \frac{\omega \Omega}{2} Y
# $$
#
# This expression agrees with the result of `qm.commutator(...)` above.
# Because the commutator is nonzero, simply multiplying the individual
# evolutions does not give the exact time evolution.
#
# ### Exact Time Evolution
#
# The Hamiltonian in this example is a small $2 \times 2$ matrix, so we can use
# NumPy to compute the evolution $U(t) = e^{-iHt}$ numerically.
# Using the eigendecomposition of the Hermitian matrix,
# $H = V \Lambda V^\dagger$, we obtain
#
# $$
# e^{-iHt}
# = V \mathrm{diag}(e^{-i \lambda_j t}) V^\dagger
# $$
#
# We start in $\lvert 0 \rangle$ and evolve for a total time $T = 1.5$.
# We store the exact final state for comparison with the approximations below.

# %%
pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

matrix = 0.5 * omega * pauli_z + 0.5 * drive * pauli_x
initial_state = np.array([1.0, 0.0], dtype=np.complex128)


def matrix_evolution(hamiltonian: np.ndarray, time: float) -> np.ndarray:
    """Return the exact time-evolution matrix for a Hermitian Hamiltonian."""
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    phases = np.exp(-1j * time * eigenvalues)
    return (eigenvectors * phases) @ eigenvectors.conj().T


exact_state = matrix_evolution(matrix, evolution_time) @ initial_state
print("Exact excitation probability P(1):", abs(exact_state[1]) ** 2)

# %% [markdown]
# The excitation probability in this model—the probability of measuring
# $\lvert 1 \rangle$—also has an analytical expression:
#
# $$
# P_1 (t)
# = \left( \frac{\Omega}{E} \right)^2 \sin^2 \left( \frac{Et}{2} \right), \qquad E
# = \sqrt{\omega^2 + \Omega^2}
# $$
#
# When the detuning $\omega$ is nonzero, the maximum excitation probability is
# less than 1.
#
# ## Algorithm
#
# Let $T$ be the total evolution time and $N$ the number of steps. The duration
# of each step is $\Delta t = T / N$.
#
# ### $S_1$: First-Order Suzuki–Trotter Decomposition (Lie–Trotter)
#
# The simplest decomposition is
#
# $$
# S_1(\Delta t)
# = e^{-i H_x \Delta t} e^{-i H_z \Delta t}, \qquad
# U(T)
# \approx [S_1(\Delta t)]^N
# $$
#
# This formula applies the evolution under $H_z$ first, followed by the
# evolution under $H_x$.
# The local error per step is $O(\Delta t^2)$, while the global state-norm error
# over $N = T/\Delta t$ steps is $O(\Delta t)$.
#
# ### $S_2$: Second-Order Suzuki–Trotter Decomposition (Strang Splitting)
#
# Making the step symmetric around the middle term cancels the leading error
# term:
#
# $$
# S_2(\Delta t)=e^{-iH_z\Delta t/2}
# e^{-iH_x\Delta t}e^{-iH_z\Delta t/2}
# $$
#
# The local error is $O(\Delta t^3)$, and the global state-norm error is
# $O(\Delta t^2)$.
#
# ### Fourth Order and Beyond: Suzuki's Recursive Construction
#
# Combining second-order formulas gives formulas of fourth, sixth, eighth,
# and higher even orders. For $k \geq 2$, the recursion is
#
# $$
# S_{2k} (\Delta t)
# = [S_{2k-2} (p_k \Delta t)]^2 S_{2k-2} ((1-4p_k) \Delta t) [S_{2k-2} (p_k \Delta t)]^2, \qquad
# p_k
# = \frac{1}{4 - 4^{1/(2k-1)}}
# $$
#
# For example, $p_2 \sim 0.4145$ for fourth order and $p_3 \sim 0.3731$ for
# sixth order. The coefficient must be updated at each order.
# The middle coefficient, $1 - 4p_k$, is negative: part of the evolution runs
# backward in time to reduce the approximation error.
# This recursion gives $S_{2k}$ a local error of
# $\mathcal{O}(\Delta t^{2k+1})$.
#
# ## Implementation
#
# ### Implementing Fourth-Order Evolution in Qamomile
#
# Qamomile's `suzuki_trotter` function constructs the operations for a product
# formula. In the following example, we specify 2 Hamiltonian terms, order 4,
# and 4 time steps.
# `hamiltonians` is a static input (`Const`) that specifies the separate
# operators, while `time` is an input that can change at execution time (`Param`).
# This lets us try different evolution times while keeping the program
# structure fixed.


# %%
evolve = suzuki_trotter(term_count=2, order=4, steps=FORMULA_STEPS)


@qm.qkernel
def rabi_samples(
    hamiltonians: qm.Const[qm.Vector[qm.Observable, 2]],
    time: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[1]]:
    """Evolve one qubit with a fourth-order product formula and measure it."""
    state = qm.qubits(1)
    state = evolve(state, hamiltonians, time)
    return qm.measure(state)


program = qm.specialize(rabi_samples, hamiltonians=hamiltonians)

# %% [markdown]
# ### The Tradeoff Between Accuracy and Logical Operation Count
#
# Increasing the formula order or the number of time steps usually improves
# the approximation, but also increases the number of operations.
# For this two-term Hamiltonian, combining adjacent evolutions under the same
# term gives 2 Pauli evolutions per step at first order, 3 at second order,
# and 11 at fourth order.
# The fourth-order program with 4 steps should therefore use
# $11 \times 4 = 44$ Pauli evolutions.
# Let's check that Qamomile's logical resource analysis gives the same result.

# %%
study = qm.resources.logical(program)
evolution_metric = qm.resources.logical_operation_count("quantum.pauli_evolution")
evolution_count = study.metric(evolution_metric).numeric_enclosure()

print("Logical Pauli-evolution count:", evolution_count)
assert evolution_count == (11 * FORMULA_STEPS, 11 * FORMULA_STEPS)

# %% [markdown]
# ## Result
#
# ### Checking Product-Formula Convergence
#
# We now use an independent $2\times2$ matrix calculation to examine how
# accurately the first-, second-, and fourth-order product formulas approach
# the exact solution.
# Let $\lvert \psi_{\mathrm{exact}} \rangle$ be the exact state and
# $\lvert \psi_{\mathrm{approx}}\rangle$ the state produced by a product formula.
# The pure-state fidelity and infidelity are defined as
#
# $$
# F
# =\lvert \langle\psi_{\mathrm{exact}} \vert \psi_{\mathrm{approx}}\rangle \rvert^2,
# \qquad \epsilon_F
# =1-F.
# $$
#
# For normalized states $a$ and $b$, the same infidelity is
# $\epsilon_F = \|b - \langle a|b\rangle a\|^2$.
#
# Fidelity does not depend on the global phase of a state.
# If the global state error is $\mathcal{O}(\Delta t^p)$, the infidelity is
# bounded by $\mathcal{O}(\Delta t^{2p})$.
#
# | Product formula | Local error per step | Global state error | Order of the infidelity bound |
# |---|---|---|---|
# | First order $S_1$ | $O(\Delta t^2)$ | $O(\Delta t)$ | $O(\Delta t^2)$ |
# | Second order $S_2$ | $O(\Delta t^3)$ | $O(\Delta t^2)$ | $O(\Delta t^4)$ |
# | Fourth order $S_4$ | $O(\Delta t^5)$ | $O(\Delta t^4)$ | $O(\Delta t^8)$ |
#
# For some initial states, the leading error term vanishes and the error can
# be even smaller. Let's check whether this example exhibits the slopes
# predicted by the table.

# %%
drift_matrix = 0.5 * omega * pauli_z
drive_matrix = 0.5 * drive * pauli_x


def suzuki_step(order: int, duration: float) -> np.ndarray:
    """Construct one Suzuki product-formula step using only NumPy."""
    if order == 1:
        return matrix_evolution(drive_matrix, duration) @ matrix_evolution(
            drift_matrix, duration
        )
    if order == 2:
        half_drift = matrix_evolution(drift_matrix, duration / 2.0)
        return half_drift @ matrix_evolution(drive_matrix, duration) @ half_drift
    if order < 4 or order % 2 != 0:
        raise ValueError("order must be 1 or a positive even integer (at least 2)")

    k = order // 2
    coefficient = 1.0 / (4.0 - 4.0 ** (1.0 / (2 * k - 1)))
    outer = suzuki_step(order - 2, coefficient * duration)
    middle = suzuki_step(order - 2, (1.0 - 4.0 * coefficient) * duration)
    return outer @ outer @ middle @ outer @ outer


step_counts = np.array([1] if docs_test_mode else [2, 4, 8, 16], dtype=np.int64)
step_widths = evolution_time / step_counts
infidelities: dict[int, list[float]] = {1: [], 2: [], 4: []}
reference_state = exact_state / np.linalg.norm(exact_state)

for steps in step_counts:
    for order in infidelities:
        step = suzuki_step(order, evolution_time / int(steps))
        approximate_state = np.linalg.matrix_power(step, int(steps)) @ initial_state
        approximate_state /= np.linalg.norm(approximate_state)
        residual = (
            approximate_state
            - np.vdot(reference_state, approximate_state) * reference_state
        )
        error = float(np.vdot(residual, residual).real)
        infidelities[order].append(error)

slopes = {}
if not docs_test_mode:
    for order, errors in infidelities.items():
        slope = np.polyfit(
            np.log(step_widths),
            np.log(np.asarray(errors)),
            1,
        )[0]
        slopes[order] = float(slope)

print("Convergence slopes (expected: order 1: 2, order 2: 4, order 4: 8):", slopes)

# %% [markdown]
# The measured slopes reproduce the expected $\Delta t^2$, $\Delta t^4$,
# and $\Delta t^8$ scaling of the infidelity.\
# Next, we plot the infidelity $1-F$ against the step width $\Delta t=T/N$
# on logarithmic axes.

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
for order, values in infidelities.items():
    ax.loglog(step_widths, values, "o-", label=f"Order {order}")

ax.set_xlabel(r"Time-step width $\Delta t = T/N$")
ax.set_ylabel(r"Infidelity $1-F$")
ax.set_title("Suzuki–Trotter convergence (NumPy)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# In this example, the slopes are approximately 2, 4, and 8, in good agreement
# with the theoretical orders.
#
# ### Running the Qamomile Simulator
#
# We now run the Qamomile program defined earlier with
# `suzuki_trotter(order=4, steps=4)`.
# We use `qm.Reference()` as the execution endpoint, evolve for a total time
# $T = 1.5$, and take 2048 measurement shots.
# The NumPy results come from ideal state vectors, while the following results
# are counts from a finite number of measurements and therefore include
# sampling fluctuations.

# %%
reference = qm.Reference()
artifact = qm.compile(program, to=reference)
record = qm.run(
    artifact,
    on=reference,
    parameters={"time": evolution_time},
    shots=SAMPLE_SHOTS,
    seed=2,
)

assert artifact.source_digest.hex() == program.semantic_digest
assert tuple(slot.name for slot in artifact.parameter_abi.slots) == ("time",)
assert record.result.shots == SAMPLE_SHOTS
assert sum(record.result.counts.values()) == SAMPLE_SHOTS
print("Measurement counts:", record.result.counts)

# %% [markdown]
# Let's compare the excitation probability estimated from the Qamomile
# measurements with the probability computed using the same fourth-order
# formula in NumPy.

# %%
energy = np.hypot(omega, drive)
exact_probability = (drive / energy) ** 2 * np.sin(energy * evolution_time / 2.0) ** 2

one_step = suzuki_step(4, evolution_time / FORMULA_STEPS)
numerical_state = np.linalg.matrix_power(one_step, FORMULA_STEPS) @ initial_state
numerical_probability = float(abs(numerical_state[1]) ** 2)

observed_probability = record.result.probability((1,))
shots = record.result.shots
sample_standard_error = np.sqrt(
    observed_probability * (1.0 - observed_probability) / shots
)

print(f"Excitation probability (exact):          {exact_probability:.4f}")
print(f"Excitation probability (NumPy, order 4): {numerical_probability:.4f}")
print(f"Excitation probability (Qamomile):       {observed_probability:.4f}")
print(f"Estimated sampling standard error:      {sample_standard_error:.4f}")

# %% [markdown]
# ### Extension: Visualizing Rabi Oscillations Over Time
#
# Finally, let's plot the excitation probability as a function of time $t$.
# For comparison, we include the exact solution and NumPy results for the
# first- and fourth-order formulas, both with 4 time steps.
# This plot illustrates Rabi oscillations and approximation accuracy. It is
# a separate calculation from the `qm.Reference()` sampling example above.

# %%
times = np.linspace(0.0, 6.0, 2 if docs_test_mode else 121)
exact_probabilities = (drive / energy) ** 2 * np.sin(energy * times / 2.0) ** 2

first_probabilities = []
fourth_probabilities = []
for current_time in times:
    for order, series in ((1, first_probabilities), (4, fourth_probabilities)):
        step = suzuki_step(order, current_time / 4.0)
        state = np.linalg.matrix_power(step, 4) @ initial_state
        series.append(float(abs(state[1]) ** 2))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(times, exact_probabilities, label="Exact", linewidth=2)
ax.plot(times, first_probabilities, "--", label="Order 1 (4 steps)")
ax.plot(times, fourth_probabilities, ":", label="Order 4 (4 steps)")
ax.set_xlabel("Evolution time t")
ax.set_ylabel("Excitation probability P(1)")
ax.set_title("Detuned Rabi oscillations (NumPy)")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# - **Time evolution:** When the terms in $H=H_z+H_x$ do not commute, the exact
#   evolution generally cannot be factored into a simple product of the
#   individual exponentials.
# - **Product formulas:** Increasing the order from first to second to fourth
#   generally improves accuracy at a fixed step width, at the cost of more
#   operations.
# - **Higher-order recursion:** $S_{2k}$ combines five copies of $S_{2k-2}$.
#   The coefficient $p_k$ is computed for each order.
# - **Qamomile:** `suzuki_trotter(term_count=2, order=4, steps=4)` defines a
#   product formula that can be called from a quantum kernel.
# - **Numerical validation:** Comparing NumPy state vectors reveals the
#   convergence slopes of the infidelity $1-F$. Finite-shot measurements in
#   Qamomile let us examine the excitation probability and its statistical
#   fluctuations.

# %% [markdown]
#
