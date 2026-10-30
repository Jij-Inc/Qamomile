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
# tags: [algorithm, optimization, variational]
# ---
#
# # QAOA for MaxCut: Building the Circuit from Scratch
#
# This is a complete reference example showing the structural shape every
# Qamomile algorithm tutorial must follow. It builds the Quantum
# Approximate Optimization Algorithm (QAOA) for MaxCut from Qamomile's
# low-level circuit primitives instead of the high-level `QAOAConverter`.
# We state MaxCut directly as an Ising model on spin variables, hand-wire
# the superposition, cost, and mixer layers as `@qkernel`s, transpile the
# ansatz with `QiskitTranspiler`, tune the variational angles with a
# classical optimizer, and decode the sampled bitstrings into a partition.
#
# At the end we replace the entire hand-wired ansatz with the single-call
# `qamomile.circuit.algorithm.qaoa_state` helper and confirm the two
# routes produce the same circuit.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import itertools
import os
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

import qamomile.circuit as qmc
from qamomile.circuit.algorithm import qaoa_state
from qamomile.optimization.binary_model import BinaryModel
from qamomile.qiskit import QiskitTranspiler

# %% [markdown]
# ## Background

# %% [markdown]
# ### What is MaxCut?
#
# Given an undirected graph $G = (V, E)$, the **MaxCut** problem asks us
# to partition the vertices into two sets so that the number of edges
# crossing between the sets is maximized. MaxCut is naturally a spin
# problem: assign each vertex $i$ a spin $s_i \in \{+1, -1\}$ for the side
# of the cut it lands on. An edge $(i, j)$ is cut exactly when
# $s_i \ne s_j$, so the number of cut edges is
#
# $$
# \text{MaxCut}(\boldsymbol{s}) = \sum_{(i,j) \in E} \frac{1 - s_i s_j}{2}.
# $$

# %% [markdown]
# ### Create the Graph
#
# We use a 5-node graph with 6 edges. It is large enough to be
# non-trivial and small enough to brute-force for comparison in
# [](#qaoa-result).

# %%
G = nx.Graph()
G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)])
num_nodes = G.number_of_nodes()

pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(5, 4))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="white",
    node_size=700,
    edgecolors="black",
)
plt.title(f"Graph: {num_nodes} nodes, {G.number_of_edges()} edges")
plt.show()

# %% [markdown]
# ## Algorithm

# %% [markdown]
# ### QAOA Ansatz
#
# QAOA prepares a $p$-layer parameterized state
#
# $$
# |\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle
# = \prod_{l=1}^{p} e^{-i \beta_l H_M}\, e^{-i \gamma_l H_C}\,
#   |{+}\rangle^{\otimes n},
# $$
#
# alternating a cost unitary $e^{-i \gamma_l H_C}$ and a mixer unitary
# $e^{-i \beta_l H_M}$ on top of the uniform superposition
# $|{+}\rangle^{\otimes n}$, with mixer Hamiltonian $H_M = \sum_i X_i$.
# The variational angles are:
#
# - $\boldsymbol{\gamma} \in \mathbb{R}^p$, the cost-layer angles, held in
#   the `gammas` parameter in [](#qaoa-implementation).
# - $\boldsymbol{\beta} \in \mathbb{R}^p$, the mixer-layer angles, held in
#   the `betas` parameter.
# - $p$, the number of layers (ansatz depth), held in `p`.

# %% [markdown]
# ### Cost Hamiltonian
#
# Maximizing $\sum_{(i,j) \in E} (1 - s_i s_j)/2$ is equivalent, up to a
# constant, to minimizing the antiferromagnetic Ising Hamiltonian
#
# $$
# H_C(\boldsymbol{s}) = \sum_{(i,j) \in E} s_i s_j.
# $$
#
# Against the general Ising form
# $H = \sum_i h_i s_i + \sum_{i<j} J_{ij} s_i s_j$, unweighted MaxCut has
# no linear terms ($h_i = 0$) and uniform couplings ($J_{ij} = 1$ on every
# edge). The spin-to-basis correspondence is the standard convention
# $Z|0\rangle = |0\rangle$, $Z|1\rangle = -|1\rangle$, so a measurement
# outcome of $0$ maps to spin $+1$ and $1$ maps to spin $-1$, and the
# ground state of $H_C$ encodes the optimal cut.

# %% [markdown]
# (qaoa-implementation)=
# ## Implementation

# %% [markdown]
# ### Step 1: Build the Spin Ising Model
#
# MaxCut is spin-native, so we skip the QUBO and binary-variable detour
# and hand the Ising coefficients straight to `BinaryModel.from_ising`.
# Every edge contributes a coupling $J_{ij} = 1$ and there are no linear
# terms. We print the coefficients so the structure is visible.

# %%
ising_quad: dict[tuple[int, int], float] = {
    tuple(sorted((i, j))): 1.0 for i, j in G.edges()
}
ising_linear: dict[int, float] = {}

spin_model = BinaryModel.from_ising(linear=ising_linear, quad=ising_quad)

print(f"Variable type:          {spin_model.vartype}")
print(f"Linear terms (h_i):     {spin_model.linear}")
print(f"Quadratic terms (J_ij): {spin_model.quad}")
print(f"Constant:               {spin_model.constant}")

# %% [markdown]
# ### Step 2: Circuit Definition
#
# We build the ansatz bottom-up, one `@qkernel` per conceptual piece: the
# uniform superposition, the cost layer, the mixer layer, and the full
# layered ansatz.

# %% [markdown]
# #### Step 1: Uniform Superposition
#
# Apply a Hadamard to every qubit to prepare $|{+}\rangle^{\otimes n}$.

# %%
@qmc.qkernel
def superposition(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(n, name="q")
    for i in qmc.range(n):
        q[i] = qmc.h(q[i])
    return q


# %% [markdown]
# #### Step 2: Cost Layer
#
# Apply the cost unitary $e^{-i \gamma H_C}$.
#
# :::{note}
# **Gate convention.** Qamomile's rotation gates carry the standard $1/2$
# factor: $\text{RZ}(\theta) = e^{-i \theta Z/2}$ and
# $\text{RZZ}(\theta) = e^{-i \theta Z \otimes Z / 2}$. To match
# $e^{-i \gamma H_C}$ exactly one would pass $2 J_{ij} \gamma$ as the
# angle, but since $\gamma$ is a variational parameter that the optimizer
# tunes freely, the constant factor is absorbed into the optimal $\gamma$.
# We therefore pass $J_{ij} \cdot \gamma$ (and $h_i \cdot \gamma$)
# directly.
# :::
#
# The `linear` argument stays in the signature even though it is empty
# for unweighted MaxCut, so the kernel is reusable for weighted MaxCut
# and spin-glass Hamiltonians that do carry linear $h_i$ terms.

# %%
@qmc.qkernel
def cost_layer(
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    q: qmc.Vector[qmc.Qubit],
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    for (i, j), Jij in quad.items():
        q[i], q[j] = qmc.rzz(q[i], q[j], angle=Jij * gamma)
    for i, hi in linear.items():
        q[i] = qmc.rz(q[i], angle=hi * gamma)
    return q


# %% [markdown]
# #### Step 3: Mixer Layer
#
# Apply the mixer unitary $e^{-i \beta H_M}$ with $H_M = \sum_i X_i$.
# Since $\text{RX}(\theta) = e^{-i \theta X / 2}$, we pass $\theta = 2\beta$
# to realize $e^{-i \beta X_i}$ on each qubit.

# %%
@qmc.qkernel
def mixer_layer(
    q: qmc.Vector[qmc.Qubit],
    beta: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    n = q.shape[0]
    for i in qmc.range(n):
        q[i] = qmc.rx(q[i], angle=2.0 * beta)
    return q


# %% [markdown]
# #### Step 4: Full Ansatz
#
# Compose the pieces: prepare the superposition, apply $p$ rounds of cost
# and mixer layers, and measure.

# %%
@qmc.qkernel
def qaoa_ansatz(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    n: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    q = superposition(n)
    for layer in qmc.range(p):
        q = cost_layer(quad, linear, q, gammas[layer])
        q = mixer_layer(q, betas[layer])
    return qmc.measure(q)


# %% [markdown]
# ### Step 3: Transpile to an Executable Circuit
#
# `transpiler.transpile` compiles the ansatz for a backend. We use
# `QiskitTranspiler`, which ships with the core install. The `bindings`
# argument fixes the problem structure at compile time (the Ising
# coefficients, the qubit count `n`, and the layer count `p`), while
# `parameters=["gammas", "betas"]` keeps the variational angles as runtime
# parameters the optimizer can change on every call.

# %%
transpiler = QiskitTranspiler()
p = 3  # number of QAOA layers

executable = transpiler.transpile(
    qaoa_ansatz,
    bindings={
        "p": p,
        "quad": spin_model.quad,
        "linear": spin_model.linear,
        "n": num_nodes,
    },
    parameters=["gammas", "betas"],
)

# %% [markdown]
# ### Step 4: Optimize the Variational Parameters
#
# We minimize the mean energy with COBYLA. Each iteration samples the
# circuit and decodes the spin-domain energy. Heavy settings are gated on
# `QAMOMILE_DOCS_TEST` so the docs build stays fast. For reproducibility
# we seed `AerSimulator` with a fixed `seed_simulator`, seed the NumPy
# generator with the same value, and set `max_parallel_threads=1` so the
# simulator does not interleave random draws across threads.

# %%
SEED = 42


def make_seeded_backend() -> AerSimulator:
    """Return a fresh AerSimulator with deterministic per-shot sampling."""
    return AerSimulator(seed_simulator=SEED, max_parallel_threads=1)


executor = transpiler.executor(backend=make_seeded_backend())
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
sample_shots = 256 if docs_test_mode else 2048
maxiter = 20 if docs_test_mode else 500
cost_history: list[float] = []


def cost_fn(params: np.ndarray) -> float:
    gammas = list(params[:p])
    betas = list(params[p:])
    result = executable.sample(
        executor,
        shots=sample_shots,
        bindings={"gammas": gammas, "betas": betas},
    ).result()
    decoded = spin_model.decode_from_sampleresult(result)
    energy = decoded.energy_mean()
    cost_history.append(energy)
    return energy


rng = np.random.default_rng(SEED)
initial_params = rng.uniform(-np.pi / 2, np.pi / 2, 2 * p)

res = minimize(cost_fn, initial_params, method="COBYLA", options={"maxiter": maxiter})

print(f"Optimized cost: {res.fun:.4f}")
print(f"Optimal params: {[round(v, 4) for v in res.x]}")

# %%
plt.figure(figsize=(8, 4))
plt.plot(cost_history, color="#2696EB")
plt.xlabel("Iteration")
plt.ylabel("Cost (mean energy)")
plt.title("QAOA Optimization Progress")
plt.show()

# %% [markdown]
# ### Step 5: Sample with Optimized Parameters
#
# One final sample draw at the optimized angles. This is the sample set
# [](#qaoa-result) analyzes; the per-iteration draws inside
# `cost_fn` were discarded.

# %%
gammas_opt = list(res.x[:p])
betas_opt = list(res.x[p:])

final_result = executable.sample(
    executor,
    shots=sample_shots,
    bindings={"gammas": gammas_opt, "betas": betas_opt},
).result()

# %% [markdown]
# ### Step 6: Decode
#
# `spin_model.decode_from_sampleresult` returns samples already in the
# spin domain ($+1$ / $-1$), so we can count cut edges directly without a
# binary conversion.

# %%
decoded = spin_model.decode_from_sampleresult(final_result)
print(f"Distinct samples: {len(decoded.samples)}")
print(f"Mean energy:      {decoded.energy_mean():.4f}")

# %% [markdown]
# (qaoa-result)=
# ## Result

# %% [markdown]
# (qaoa-classical-baseline)=
# ### Classical Baseline (Brute Force)
#
# With only $2^5 = 32$ spin configurations we can enumerate every
# partition and record the true optimum to compare the QAOA result
# against.

# %%
best_cut = 0
optimal_partitions: list[tuple[int, ...]] = []

for spins in itertools.product([+1, -1], repeat=num_nodes):
    cut = sum(1 for i, j in G.edges() if spins[i] != spins[j])
    if cut > best_cut:
        best_cut = cut
        optimal_partitions = [spins]
    elif cut == best_cut:
        optimal_partitions.append(spins)

print(f"Optimal MaxCut value: {best_cut}")
print(f"Number of optimal partitions: {len(optimal_partitions)}")
for part in optimal_partitions:
    print(f"  {part}")

# %% [markdown]
# ### Decode and Analyze Results

# %% [markdown]
# #### Best Cut
#
# Scan the decoded samples for the partition with the largest cut and
# compare it to the brute-force optimum from
# [](#qaoa-classical-baseline).

# %%
cut_distribution: Counter[int] = Counter()
best_qaoa_cut = 0
best_qaoa_sample: list[int] | None = None

for sample, _energy, occ in zip(
    decoded.samples, decoded.energy, decoded.num_occurrences
):
    spins = [sample[i] for i in range(num_nodes)]
    cut = sum(1 for i, j in G.edges() if spins[i] != spins[j])
    cut_distribution[cut] += occ
    if cut > best_qaoa_cut:
        best_qaoa_cut = cut
        best_qaoa_sample = spins

print(f"Best QAOA cut: {best_qaoa_cut}  (optimal: {best_cut})")
print(f"Best partition (spins): {best_qaoa_sample}")

# %% [markdown]
# #### Objective Value Distribution

# %%
cuts = sorted(cut_distribution.keys())
counts = [cut_distribution[c] for c in cuts]

plt.figure(figsize=(8, 4))
plt.bar([str(c) for c in cuts], counts, color="#2696EB")
plt.xlabel("Cut size")
plt.ylabel("Frequency")
plt.title("Distribution of MaxCut Values from QAOA")
plt.show()

# %% [markdown]
# #### Visualize the Best Solution

# %%
if best_qaoa_sample is not None:
    color_map = [
        "#FF6B6B" if best_qaoa_sample[i] == +1 else "#4ECDC4"
        for i in range(num_nodes)
    ]
    plt.figure(figsize=(5, 4))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=color_map,
        node_size=700,
        edgecolors="black",
    )
    plt.title(f"QAOA partition (cut = {best_qaoa_cut})")
    plt.show()

# %% [markdown]
# ### Using the Built-in `qaoa_state`
#
# Everything we hand-wired above (superposition, cost layer, mixer layer,
# and the layered loop) is already provided by
# `qamomile.circuit.algorithm.qaoa_state`. It takes the same Ising
# coefficients (`quad`, `linear`) and variational angles (`gammas`,
# `betas`). We build the same circuit with the helper and confirm the two
# routes agree. Under a fixed `seed_simulator`, bit-identical circuits
# yield identical samples and therefore identical mean energies, so a
# difference in the two printed values would indicate the manual and
# built-in routes emitted different circuits rather than residual
# shot-noise.

# %%
@qmc.qkernel
def qaoa_builtin(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    n: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    q = qaoa_state(p=p, quad=quad, linear=linear, n=n, gammas=gammas, betas=betas)
    return qmc.measure(q)


# %%
exe_builtin = transpiler.transpile(
    qaoa_builtin,
    bindings={
        "p": p,
        "quad": spin_model.quad,
        "linear": spin_model.linear,
        "n": num_nodes,
    },
    parameters=["gammas", "betas"],
)

executor_manual = transpiler.executor(backend=make_seeded_backend())
executor_builtin = transpiler.executor(backend=make_seeded_backend())

result_manual = executable.sample(
    executor_manual,
    shots=sample_shots,
    bindings={"gammas": gammas_opt, "betas": betas_opt},
).result()

result_builtin = exe_builtin.sample(
    executor_builtin,
    shots=sample_shots,
    bindings={"gammas": gammas_opt, "betas": betas_opt},
).result()

decoded_manual = spin_model.decode_from_sampleresult(result_manual)
decoded_builtin = spin_model.decode_from_sampleresult(result_builtin)
print(f"Manual   mean energy: {decoded_manual.energy_mean():.4f}")
print(f"Built-in mean energy: {decoded_builtin.energy_mean():.4f}")

# %% [markdown]
# ## Summary
#
# In this tutorial we built the QAOA variational ansatz for MaxCut from
# scratch and ran it through the full Qamomile workflow, taking a 5-node
# graph from a spin Ising model to a decoded partition that matched the
# brute-force optimum.
#
# - **Spin-first formulation:** MaxCut went straight into
#   `BinaryModel.from_ising` as an antiferromagnetic Ising model, with no
#   QUBO or binary-variable detour.
# - **Hand-wired, then built-in:** the ansatz was assembled from `qmc.h`,
#   `qmc.rzz`, `qmc.rz`, and `qmc.rx` inside `@qkernel`s, and
#   `qamomile.circuit.algorithm.qaoa_state` then reproduced the whole
#   construction in a single call with an identical mean energy.
# - **End-to-end Qamomile path:** `QiskitTranspiler` produced the
#   executable that `executable.sample` evaluated each iteration, and
#   `spin_model.decode_from_sampleresult` lifted the bitstrings back to
#   spins.
#
# The same spin-first recipe applies to any Ising problem where you can
# write down the $h_i$ and $J_{ij}$ coefficients (weighted MaxCut,
# spin-glass ground states, the Sherrington-Kirkpatrick model): plug them
# into `BinaryModel.from_ising` and reuse the circuit components above.
