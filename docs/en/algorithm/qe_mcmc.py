# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
# tags: [algorithm, sample-based]
# ---
#
# # Quantum-enhanced Markov chain Monte Carlo
#
# This tutorial demonstrates how to implement Quantum-enhanced Markov chain
# Monte Carlo (QeMCMC) [](https://doi.org/10.1038/s41586-023-06095-4) using
# Qamomile.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import cmath
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np

import qamomile as qm
from qamomile.stdlib import suzuki_trotter

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# ---
# ## Background


# %% [markdown]
# ### Sampling the Boltzmann Distribution


# %% [markdown]
# In many physics and engineering problems, drawing samples $\bm{x}$ from a
# probability distribution $\mu(\bm{x})$ is a central computational task.
# A classic example is the **Boltzmann distribution** from statistical
# mechanics:
# $$
# \mu(\bm{x}) = \frac{1}{Z} \exp\bigl(-\beta E(\bm{x})\bigr), \quad Z = \sum_{\bm{x}} \exp\bigl(-\beta E(\bm{x})\bigr).
# $$
# Here, $E(\bm{x})$ is the energy of state $\bm{x}$, $\beta = 1/T$ is the
# inverse temperature, and $Z$ is the normalization constant known as the
# partition function.
# Beyond giving the probability distribution of state $\bm{x}$ at thermal
# equilibrium, sampling that targets the Boltzmann distribution is also
# widely used as an approach for combinatorial optimization problems.
#
# As a concrete example of an energy function for the Boltzmann distribution,
# consider the **Ising model**.
# The Ising model is a system with a spin variable $x_i \in \{-1, +1\}$ at
# each site $i$, whose energy is given by:
# $$
# E(\bm{x}) = -\sum_{\langle i, j \rangle} J_{ij} \, x_i x_j - \sum_i h_i \, x_i.
# $$
# Here, $J_{ij}$ is the interaction between spins and $h_i$ is the external
# magnetic field at site $i$.
# Since the total number of states grows exponentially as $2^n$, computing
# the partition function $Z$ exactly becomes intractable for large $n$.
# MCMC, introduced below, is therefore used to draw samples directly from
# $\mu(\bm{x})$.
#
# First, let us actually visualize the Boltzmann distribution for a small
# Ising model.
# We take a 1D ferromagnetic Ising chain ($J_{i,i+1} = 1$, $h_i = 0$) and plot
# histograms of the probability aggregated by energy $E(\bm{x})$ for several
# values of the inverse temperature $\beta$.


# %%
N_SPINS = 6
ISING_COUPLING = 1.0


def spin_chain_energy(state: np.ndarray) -> float:
    """Return the energy of a 1D ferromagnetic Ising chain."""

    return -ISING_COUPLING * float(np.sum(state[:-1] * state[1:]))


all_spin_states = np.array(
    [
        [1 - 2 * ((basis >> index) & 1) for index in range(N_SPINS)]
        for basis in range(1 << N_SPINS)
    ]
)
spin_energies = np.array([spin_chain_energy(state) for state in all_spin_states])
assert all_spin_states.shape == (1 << N_SPINS, N_SPINS)
assert spin_energies.min() == -(N_SPINS - 1) * ISING_COUPLING
assert spin_energies.max() == (N_SPINS - 1) * ISING_COUPLING

unique_energies = np.unique(spin_energies)
betas = [0.5, 1.0, 2.0]
figure, axes = plt.subplots(1, len(betas), figsize=(12, 3.5), sharey=True)
for axis, inverse_temperature in zip(axes, betas, strict=True):
    boltzmann_weights = np.exp(-inverse_temperature * spin_energies)
    probabilities = boltzmann_weights / boltzmann_weights.sum()
    energy_probabilities = np.array(
        [probabilities[spin_energies == energy].sum() for energy in unique_energies]
    )
    axis.bar(unique_energies, energy_probabilities, width=0.8)
    axis.set_xlabel(r"Energy $E(\mathbf{x})$")
    axis.set_title(rf"$\beta = {inverse_temperature}$")
axes[0].set_ylabel(r"Probability $\mu(E)$")
figure.suptitle(f"Boltzmann distribution of {N_SPINS}-spin Ising chain")
figure.tight_layout()
plt.show()

# %% [markdown]
# ### Markov chain Monte Carlo (MCMC)
#
# Markov chain Monte Carlo (MCMC) is a general-purpose method for sampling
# from probability distributions.
# It achieves sampling from a target distribution $\mu(\bm{x})$ by exploiting
# a stochastic process called a Markov chain.
# Here we introduce the Metropolis-Hastings (MH) algorithm
# [](https://doi.org/10.1093/biomet/57.1.97) , a common implementation of MCMC.
#
# The MH algorithm generates a new transition $\bm{x} \rightarrow \bm{y}$
# of the Markov chain according to a proposal probability $Q(\bm{y}|\bm{x})$,
# and then accepts or rejects this transition according to the acceptance
# probability
# $$
# A(\bm{y} | \bm{x}) = \min \left(1, \frac{\mu(\bm{y})}{\mu(\bm{x})} \cdot \frac{Q(\bm{x} | \bm{y})}{Q(\bm{y} | \bm{x})} \right).
# $$
# These two steps generate the state at time $t+1$ from the state at time
# $t$.
# After sufficient time, the states of this Markov chain follow the target
# distribution $\mu(\bm{x})$.
# Thus, by running enough transitions, the resulting chain states
# $\{\bm{x}^{(t)}\}$ can be taken as the desired samples.
#
# Let us verify this in practice.
# We will sample from the Boltzmann distribution we just prepared using the
# MH method.
# Many choices of proposal distribution are possible; here we use the
# simplest one, which picks a single random spin and flips it.
#
# Next, we implement the step that stochastically processes the proposed
# transition according to the acceptance probability.
# Since this proposal satisfies $Q(\bm{x} \mid \bm{y}) = Q(\bm{y} \mid \bm{x})$,
# the ratio of $Q$ cancels out in the acceptance probability.
# Therefore, when the target distribution is the Boltzmann distribution,
# the acceptance probability takes the following simple form
# $$
# A(\bm{y} | \bm{x}) = \min \left(1, \frac{\mu(\bm{y})}{\mu(\bm{x})}\right) = \min \left(1, \exp(-\beta (E(\bm{y}) - E(\bm{x})))\right).
# $$
#
# Now we have a working MCMC implementation.
# Let us run it to draw samples.
#
# Let us confirm that the sampled sequence follows the Boltzmann
# distribution.
# Using the samples, we estimate a physical quantity associated with the
# Boltzmann distribution $\mu(\bm{x})$.
# Here we estimate the spin's average magnetization:
# $$
# \langle m \rangle = \sum_{\bm{x}} \mu(\bm{x}) m(\bm{x})
# $$
# where the magnetization is
# $$
# m(\bm{x}) = \frac{1}{n} \sum_{i=1}^n x_i.
# $$
# Since the average magnetization is the expectation of the magnetization
# with respect to the Boltzmann distribution, the estimator should become
# more accurate as the number of samples grows and as the sample
# distribution gets closer to the Boltzmann distribution.
# Let us plot the estimator $\bar{m}_t$ obtained from the first $t$ MCMC
# samples.

# %%
classical_rng = random.Random(0)


def local_spin_update(state: np.ndarray) -> np.ndarray:
    """Propose a spin configuration by flipping one selected spin."""

    proposal = state.copy()
    index = classical_rng.randrange(len(state))
    proposal[index] *= -1
    return proposal


def metropolis_hastings_spins(
    state: np.ndarray,
    candidate: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Accept or reject a spin proposal under the Boltzmann target."""

    delta_energy = spin_chain_energy(candidate) - spin_chain_energy(state)
    if delta_energy < 0.0 or classical_rng.random() < math.exp(-beta * delta_energy):
        return candidate
    return state


CLASSICAL_STEPS = 12 if docs_test_mode else 1_000
CLASSICAL_BETA = 0.5
classical_samples = np.zeros((CLASSICAL_STEPS, N_SPINS))
classical_state = np.ones(N_SPINS)
for step in range(CLASSICAL_STEPS):
    classical_candidate = local_spin_update(classical_state)
    classical_state = metropolis_hastings_spins(
        classical_state,
        classical_candidate,
        CLASSICAL_BETA,
    )
    classical_samples[step] = classical_state


def average_magnetization(samples: np.ndarray) -> float:
    """Return the average magnetization of a two-dimensional sample array."""

    return float(np.mean(np.mean(samples, axis=1)))


classical_magnetization = np.array(
    [
        average_magnetization(classical_samples[:step])
        for step in range(1, CLASSICAL_STEPS + 1)
    ]
)
boltzmann_weights = np.exp(-CLASSICAL_BETA * spin_energies)
boltzmann_probabilities = boltzmann_weights / boltzmann_weights.sum()
theoretical_magnetization = float(
    np.sum(boltzmann_probabilities * all_spin_states.mean(axis=1))
)
assert math.isclose(theoretical_magnetization, 0.0, abs_tol=1e-15)

plt.plot(classical_magnetization, label="MCMC estimate")
plt.axhline(
    theoretical_magnetization,
    color="red",
    linestyle="--",
    label=f"Theoretical ({theoretical_magnetization:.3f})",
)
plt.xlabel("Step")
plt.ylabel("Magnetization")
plt.title("Magnetization vs. Step")
plt.legend()
plt.show()

# %% [markdown]
# ---
# ## Algorithm


# %% [markdown]
# The Quantum-enhanced MCMC algorithm is an MCMC that uses sampling from a
# quantum circuit as its proposal distribution
# [](https://doi.org/10.1038/s41586-023-06095-4).
# Starting from the current state $\bm{x}$, we apply a quantum circuit $U$
# and measure in the computational basis to obtain a new state $\bm{y}$.
# The resulting proposal distribution $Q(\bm{y}|\bm{x})$ is:
# $$
# Q(\bm{y}|\bm{x}) = \| \langle \bm{y} | U | \bm{x} \rangle \|^2
# $$
# Computing this probability directly is difficult, but when the quantum
# circuit satisfies $U = U^\top$, the proposal distribution satisfies
# $Q(\bm{x} \mid \bm{y}) = Q(\bm{y} \mid \bm{x})$, so the $Q$ terms cancel
# out in the acceptance probability, eliminating the need to explicitly
# compute $Q$.
# For example, to sample from the Boltzmann distribution of the Ising model,
# we can use a trotterized time evolution under a time-independent Hamiltonian:
# $$
# U(\gamma, t) = \exp(-i H t), \quad \quad
# H = (1-\gamma) \alpha H_M + \gamma H_C.
# $$
# Here, $H_M$ is called the mixer Hamiltonian and generates quantum
# transitions between states, while $H_C$ is the Ising Hamiltonian.
# $\gamma \in [0,1]$ is a parameter that controls the relative weights
# of the two terms.
# $\alpha$ is a normalization factor used to ensure that the eigenvalues
# of the mixer and cost Hamiltonians are on the same scale.
# $(\gamma, t)$ are tunable parameters that determine the efficiency of the
# MCMC process.

# %%
GAMMA = 0.45
EVOLUTION_TIME = 12.0
BETA = 1.0
N_SPINS = 6
TROTTER_STEPS = 1 if docs_test_mode else 15
INPUT_BITS = (1, 0, 1, 0, 1, 0)
INPUT_ANGLES = tuple(math.pi * bit for bit in INPUT_BITS)
HAMILTONIANS = (
    (1.0 - GAMMA)
    * sum(
        (qm.X(index) for index in range(N_SPINS)),
        qm.Hamiltonian.zero(num_qubits=N_SPINS),
    ),
    -GAMMA
    * sum(
        (qm.Z(index) * qm.Z(index + 1) for index in range(N_SPINS - 1)),
        qm.Hamiltonian.zero(num_qubits=N_SPINS),
    ),
)
EVOLUTION = suzuki_trotter(term_count=2, order=2, steps=TROTTER_STEPS)

assert len(HAMILTONIANS) == 2
assert TROTTER_STEPS == (1 if docs_test_mode else 15)

# %% [markdown]
# ## Author one typed proposal
#
# A zero-or-$\pi$ `RX` rotation prepares each current-state bit. Fixed vector
# lengths make both the input and the two Hamiltonian terms part of the checked
# function contract. The linear `Q[Bits[6]]` value then flows through the shared
# product formula and into one typed shot result.


# %%
@qm.qkernel
def qemcmc_samples(
    input_angles: qm.Param[qm.Vector[qm.Real, 6]],  # type: ignore[valid-type]
    hamiltonians: qm.Const[qm.Vector[qm.Observable, 2]],  # type: ignore[valid-type]
    time: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[6]]:  # type: ignore[valid-type]
    """Sample one runtime-rebindable quantum-enhanced MCMC proposal.

    Args:
        input_angles (qm.Param[qm.Vector[qm.Real, 6]]): Zero-or-pi rotations
            encoding the current Markov-chain bitstring.
        hamiltonians (qm.Const[qm.Vector[qm.Observable, 2]]): Mixer and Ising
            cost Hamiltonian terms.
        time (qm.Param[qm.Real]): Total proposal evolution time.

    Returns:
        qm.Shot[qm.Bits[6]]: Proposed six-spin bitstring.
    """

    state = qm.qubits(6)
    for index in range(6):
        state = qm.rx(state, input_angles[index], target=index)
    state = EVOLUTION(state, hamiltonians, time)
    return qm.measure(state)


program = qemcmc_samples.specialize(hamiltonians=HAMILTONIANS)

# %% [markdown]
# Specialization records the fixed Ising model without erasing the authored
# signature. Only `input_angles` and `time` remain in the eventual artifact
# ABI. Inspection also shows that QeMCMC is composition, not a privileged
# compiler operation.

# %%
assert tuple(parameter.name for parameter in program.entry.parameters) == (
    "input_angles",
    "hamiltonians",
    "time",
)
assert tuple(name for name, _ in program.const_bindings) == ("hamiltonians",)
assert "quantum.qemcmc" not in program.inspect()
assert program.inspect().count("quantum.pauli_evolution") == 1

# %% [markdown]
# ## Estimate the same body
#
# Every step of the second-order two-term formula has the palindromic schedule
# $H_0/2,H_1,H_0/2$. Logical analysis therefore reports exactly three abstract
# Pauli evolutions per Trotter step. Backend-specific gate synthesis happens
# later and cannot change this source-level count.

# %%
study = qm.resources.logical(program)
pauli_evolutions = qm.resources.logical_operation_count("quantum.pauli_evolution")

expected_evolutions = 3 * TROTTER_STEPS
assert study.metric(pauli_evolutions).numeric_enclosure() == (
    expected_evolutions,
    expected_evolutions,
)
assert qm.resources.ResourceStudy.from_bytes(study.to_bytes()) == study

# %% [markdown]
# ## Compile once and sample one proposal
#
# The executable example performs one 128-shot launch on six qubits. A real
# Markov chain would repeatedly convert its current bits to zero-or-$\pi$
# angles and rebind the same artifact. Keeping that loop out of the page avoids
# turning documentation validation into a long stochastic workload.

# %%
reference = qm.Reference()
artifact = qm.compile(program, to=reference.target)
execution = qm.run(
    artifact,
    on=reference,
    parameters={"input_angles": INPUT_ANGLES, "time": EVOLUTION_TIME},
    shots=128,
    seed=2,
)

assert tuple(slot.name for slot in artifact.parameter_abi.slots) == (
    "input_angles",
    "time",
)
assert execution.artifact is artifact
assert execution.invocation.values() == {
    "input_angles": INPUT_ANGLES,
    "time": EVOLUTION_TIME,
}
assert execution.result.shots == 128

# %% [markdown]
# ## Verify the hybrid Markov semantics exactly
#
# Sampling demonstrates the execution path, but it is a poor way to prove
# reversibility or stationarity. For this six-spin example we can construct all
# 64 proposal rows exactly on the host. The helper below follows the same
# Suzuki factors used by the authored program; it is only an oracle for the
# tutorial assertions, not another compiler implementation.


# %%
def apply_x_evolution(
    amplitudes: list[complex],
    *,
    target: int,
    angle: float,
) -> list[complex]:
    """Apply exp(-i angle X_target) to a six-qubit host vector."""

    cosine = math.cos(angle)
    sine = -1.0j * math.sin(angle)
    return [
        cosine * amplitudes[basis] + sine * amplitudes[basis ^ (1 << target)]
        for basis in range(1 << N_SPINS)
    ]


def proposal_probabilities(
    input_bits: tuple[int, ...],
) -> tuple[float, ...]:
    """Return exact address-ordered proposal probabilities."""

    basis_index = sum(bit << index for index, bit in enumerate(input_bits))
    amplitudes = [0.0j] * (1 << N_SPINS)
    amplitudes[basis_index] = 1.0 + 0.0j
    step_time = EVOLUTION_TIME / TROTTER_STEPS
    mixer_angle = (1.0 - GAMMA) * step_time / 2.0
    for _ in range(TROTTER_STEPS):
        for target in range(N_SPINS):
            amplitudes = apply_x_evolution(
                amplitudes,
                target=target,
                angle=mixer_angle,
            )
        for basis in range(1 << N_SPINS):
            interaction = sum(
                (-1 if basis & (1 << index) else 1)
                * (-1 if basis & (1 << (index + 1)) else 1)
                for index in range(N_SPINS - 1)
            )
            amplitudes[basis] *= cmath.exp(1.0j * GAMMA * step_time * interaction)
        for target in range(N_SPINS):
            amplitudes = apply_x_evolution(
                amplitudes,
                target=target,
                angle=mixer_angle,
            )
    return tuple(abs(amplitude) ** 2 for amplitude in amplitudes)


def ising_energy(bits: tuple[int, ...]) -> float:
    """Return the six-spin ferromagnetic Ising-chain energy."""

    spins = tuple(1 - 2 * bit for bit in bits)
    return -float(sum(spins[index] * spins[index + 1] for index in range(N_SPINS - 1)))


states = tuple(
    tuple((basis >> index) & 1 for index in range(N_SPINS))
    for basis in range(1 << N_SPINS)
)
proposal = tuple(proposal_probabilities(bits) for bits in states)
energies = tuple(ising_energy(bits) for bits in states)
weights = tuple(math.exp(-BETA * energy) for energy in energies)
partition = sum(weights)
stationary = tuple(weight / partition for weight in weights)
state_count = 1 << N_SPINS
transition: list[list[float]] = [[0.0] * state_count for _ in range(state_count)]

for source in range(state_count):
    for target in range(state_count):
        if source == target:
            continue
        acceptance = min(
            1.0,
            math.exp(-BETA * (energies[target] - energies[source])),
        )
        transition[source][target] = proposal[source][target] * acceptance
    transition[source][source] = 1.0 - sum(transition[source])

for source in range(state_count):
    assert math.isclose(sum(proposal[source]), 1.0, abs_tol=1e-12)
    assert math.isclose(sum(transition[source]), 1.0, abs_tol=1e-12)
    for target in range(state_count):
        assert math.isclose(
            proposal[source][target],
            proposal[target][source],
            abs_tol=1e-12,
        )
        assert math.isclose(
            stationary[source] * transition[source][target],
            stationary[target] * transition[target][source],
            abs_tol=1e-12,
        )

input_basis = sum(bit << index for index, bit in enumerate(INPUT_BITS))
for basis, expected in enumerate(proposal[input_basis]):
    bits = tuple((basis >> index) & 1 for index in range(N_SPINS))
    sampling_error = 6.0 * math.sqrt(expected * (1.0 - expected) / 128)
    assert abs(execution.result.probability(bits) - expected) <= sampling_error + 0.01

# %% [markdown]
# The first equality proves that every proposal and transition row is
# normalized. Proposal symmetry makes the Metropolis simplification valid; the
# final equality is detailed balance with the Boltzmann distribution. Thus the
# quantum and classical halves compose without moving randomness, acceptance,
# or Markov-chain policy into QCore.

# %% [markdown]
# ---
# ## Example Run


# %% [markdown]
# Let us run the QeMCMC algorithm we just implemented.
# We switch to a lower temperature $\beta = 1.0$ where classical local
# updates mix slowly, so that we can observe the behavior of the quantum
# proposal distribution under conditions that are harder for the classical
# baseline. For a fair comparison, we also run a classical MCMC at the same
# $\beta = 1.0$ alongside the quantum run.


# %% [markdown]
# We compute the estimator of the average magnetization and compare it with
# the classical MCMC result obtained at the same $\beta = 1.0$.

# %%
rng = random.Random(7)
current = (0,) * N_SPINS
chain = [current]
accepted = 0
CHAIN_STEPS = 12 if docs_test_mode else 1_000

for step in range(CHAIN_STEPS):
    proposal_result = qm.run(
        artifact,
        on=reference,
        parameters={
            "input_angles": tuple(math.pi * bit for bit in current),
            "time": EVOLUTION_TIME,
        },
        shots=1,
        seed=100 + step,
    ).result
    candidate = next(iter(proposal_result.counts))
    acceptance = min(
        1.0,
        math.exp(-BETA * (ising_energy(candidate) - ising_energy(current))),
    )
    if rng.random() < acceptance:
        current = candidate
        accepted += 1
    chain.append(current)

print("chain endpoints:", chain[:5], "...", chain[-5:])
print("accepted transitions:", accepted)
assert len(chain) == CHAIN_STEPS + 1
assert all(state in states for state in chain)

# %% [markdown]
# ### Compare with classical local-update MCMC
#
# The original tutorial compares magnetization convergence for classical MCMC
# and QeMCMC at the same inverse temperature. The six-spin model preserves the
# original comparison while remaining small enough to verify the complete
# quantum transition matrix above. Spin-flip symmetry makes the exact
# equilibrium magnetization zero.


# %%
def local_update(bits: tuple[int, ...], generator: random.Random) -> tuple[int, ...]:
    """Return a proposal with one uniformly selected spin flipped."""

    index = generator.randrange(N_SPINS)
    proposal_bits = list(bits)
    proposal_bits[index] = 1 - proposal_bits[index]
    return tuple(proposal_bits)


def accept_metropolis(
    source: tuple[int, ...],
    candidate: tuple[int, ...],
    generator: random.Random,
) -> tuple[int, ...]:
    """Accept or reject one proposal using the shared Ising objective."""

    probability = min(
        1.0,
        math.exp(-BETA * (ising_energy(candidate) - ising_energy(source))),
    )
    return candidate if generator.random() < probability else source


classical_rng = random.Random(7)
classical_current = (0,) * N_SPINS
classical_chain = [classical_current]
for _ in range(CHAIN_STEPS):
    classical_candidate = local_update(classical_current, classical_rng)
    classical_current = accept_metropolis(
        classical_current,
        classical_candidate,
        classical_rng,
    )
    classical_chain.append(classical_current)


def cumulative_magnetization(samples: list[tuple[int, ...]]) -> np.ndarray:
    """Return the running mean magnetization of a bitstring chain."""

    per_sample = np.asarray(
        [sum(1 - 2 * bit for bit in bits) / N_SPINS for bits in samples]
    )
    return np.cumsum(per_sample) / np.arange(1, len(per_sample) + 1)


quantum_magnetization = cumulative_magnetization(chain)
classical_magnetization = cumulative_magnetization(classical_chain)
theoretical_magnetization = sum(
    probability * sum(1 - 2 * bit for bit in bits) / N_SPINS
    for probability, bits in zip(stationary, states, strict=True)
)

assert math.isclose(theoretical_magnetization, 0.0, abs_tol=1e-12)
assert len(classical_chain) == len(chain) == CHAIN_STEPS + 1

plt.plot(classical_magnetization, label="MCMC estimate")
plt.plot(quantum_magnetization, label="QeMCMC estimate")
plt.axhline(theoretical_magnetization, linestyle="--", label="Theoretical")
plt.xlabel("Markov step")
plt.ylabel("Average magnetization")
plt.legend()
plt.show()

# %% [markdown]
# ---
# ## Summary
#
# In this tutorial, we began with a review of classical Metropolis-Hastings
# MCMC and then implemented Quantum-enhanced MCMC (QeMCMC) on top of
# Qamomile, using the quantum circuit $U(\gamma, t) = \exp(-i t H)$ with
# $H = (1-\gamma) H_M + \gamma H_C$ as the proposal distribution.
# Specifically, after preparing the mixer and cost Hamiltonians with
# Qamomile observables, we built the proposal circuit via Suzuki-Trotter
# time evolution using `suzuki_trotter` inside a `qkernel`.
# Finally, we compiled the quantum proposal once for the `Reference` target,
# plugged it into the existing MH loop, and compared the convergence of the
# average magnetization for classical MCMC and QeMCMC on the same Ising
# chain, confirming that the end-to-end quantum-classical hybrid loop
# behaves as intended.
