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
# # QAOA for MaxCut with `QAOAConverter`
#
# This skeleton demonstrates the structural shape every Qamomile algorithm
# tutorial must follow. We solve a small MaxCut instance end-to-end using
# `qamomile.optimization.QAOAConverter`, the high-level converter that
# folds QUBO → Ising conversion, `@qkernel` construction, and decode into
# three method calls. The instance is a 4-node triangle-plus-pendant graph
# kept deliberately small so that a brute-force enumeration in the
# `## Result` section yields the ground-truth optimum to compare the
# quantum result against.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

import itertools
import os
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import minimize

from qamomile.optimization.binary_model import BinaryModel
from qamomile.optimization.qaoa import QAOAConverter
from qamomile.qiskit import QiskitTranspiler

# %% [markdown]
# ## Background

# %% [markdown]
# ### What is MaxCut?
#
# Given an undirected graph $G = (V, E)$, **MaxCut** asks us to split the
# vertices into two sets $S, \bar{S}$ so that the number of edges between
# the sets is maximised:
#
# $$
# \text{MaxCut}(x) = \sum_{(i,j) \in E} \bigl[x_i (1 - x_j) + x_j (1 - x_i)\bigr],
# $$
#
# where $x_i \in \{0, 1\}$ selects which set vertex $i$ belongs to.

# %% [markdown]
# ### Create the Graph
#
# We use a 4-node triangle $\{0, 1, 2\}$ with a pendant node $3$ attached
# to vertex $2$. The graph is small enough that the `## Result` section
# can brute-force the ground-truth optimum and — thanks to the odd
# triangle — not bipartite, so the MaxCut is strictly less than the edge
# count.

# %%
G = nx.Graph()
G.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3)])
num_nodes = G.number_of_nodes()

pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(4, 3))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="white",
    node_size=600,
    edgecolors="black",
)
plt.title(f"Graph: {num_nodes} nodes, {G.number_of_edges()} edges")
plt.show()

# %% [markdown]
# ## Algorithm

# %% [markdown]
# ### QAOA Ansatz
#
# QAOA prepares a $p$-layer parameterised state
#
# $$
# |\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle
# = \prod_{l=1}^{p} e^{-i \beta_l H_M}\, e^{-i \gamma_l H_C}\, |{+}\rangle^{\otimes n},
# $$
#
# alternating the **cost unitary** $e^{-i \gamma_l H_C}$ and the **mixer
# unitary** $e^{-i \beta_l H_M}$ with $H_M = \sum_i X_i$. The angles
# $\gamma_l$ and $\beta_l$ are the variational parameters that a classical
# optimiser tunes.

# %% [markdown]
# ### Cost Hamiltonian
#
# For MaxCut, substituting $x_i = (1 - s_i)/2$ with $s_i \in \{\pm 1\}$
# turns the objective into an Ising Hamiltonian
#
# $$
# H_C = \sum_{i < j} J_{ij}\, Z_i Z_j + \sum_i h_i\, Z_i + \text{const},
# $$
#
# whose ground state encodes the optimal cut. Each $Z$-term comes directly
# from an edge of $G$.

# %% [markdown]
# ### Parameters
#
# - $p$ — the number of QAOA layers (ansatz depth). Larger $p$ gives
#   strictly more expressive states but more optimiser work.
# - $\boldsymbol{\gamma} \in \mathbb{R}^p$ — cost-layer angles (`gammas`
#   in code).
# - $\boldsymbol{\beta} \in \mathbb{R}^p$ — mixer-layer angles (`betas`
#   in code).

# %% [markdown]
# ## Implementation

# %% [markdown]
# ### Step 1: QUBO to Ising
#
# We build a `BinaryModel` from the MaxCut QUBO. Each edge $(i, j)$
# contributes $-x_i - x_j + 2 x_i x_j$ — the MaxCut objective negated so
# the optimiser minimises. `QAOAConverter` then converts internally to
# the spin domain and builds the Ising cost Hamiltonian. We print the
# Hamiltonian so the reader sees the Pauli-$Z$ coefficients.

# %%
qubo: dict[tuple[int, int], float] = {}
for i, j in G.edges():
    qubo[(i, i)] = qubo.get((i, i), 0.0) - 1.0
    qubo[(j, j)] = qubo.get((j, j), 0.0) - 1.0
    qubo[(i, j)] = qubo.get((i, j), 0.0) + 2.0

model = BinaryModel.from_qubo(qubo)
converter = QAOAConverter(model)
hamiltonian = converter.get_cost_hamiltonian()
print(hamiltonian)

# %% [markdown]
# ### Step 2: Transpile to an Executable Circuit
#
# `converter.transpile(transpiler, p=p)` builds a $p$-layer QAOA
# `@qkernel` and hands it to the backend transpiler — here
# `QiskitTranspiler`, which ships with the core install. The angles
# `gammas` and `betas` stay as *runtime parameters* so the classical
# optimiser can vary them each call; every other input (Ising
# coefficients, qubit count, `p`) is bound at compile time.
#
# `QAOAConverter` builds the cost and mixer unitaries internally, so no
# rotation gates are exposed by name in this skeleton. Tutorials that
# *hand-wire* the cost or mixer layer with `qmc.rzz`, `qmc.rx`, etc.
# must place the Qamomile-specific factor-of-2 call-out from the
# section guide directly above the rotation cell in this step.

# %%
transpiler = QiskitTranspiler()
p = 2  # number of QAOA layers
executable = converter.transpile(transpiler, p=p)

# %% [markdown]
# ### Step 3: Optimize the Variational Parameters
#
# We minimise the mean energy with COBYLA. Heavy compute is gated on
# `QAMOMILE_DOCS_TEST` so CI builds stay fast; when running the notebook
# locally the shot count and iteration budget jump to the production
# values.

# %%
executor = transpiler.executor()
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
sample_shots = 256 if docs_test_mode else 1024
maxiter = 20 if docs_test_mode else 200

rng = np.random.default_rng(42)
initial_params = rng.uniform(-np.pi / 2, np.pi / 2, 2 * p)
cost_history: list[float] = []


def cost_fn(params: np.ndarray) -> float:
    gammas = list(params[:p])
    betas = list(params[p:])
    result = executable.sample(
        executor,
        shots=sample_shots,
        bindings={"gammas": gammas, "betas": betas},
    ).result()
    decoded = converter.decode(result)
    energy = decoded.energy_mean()
    cost_history.append(energy)
    return energy


res = minimize(cost_fn, initial_params, method="COBYLA", options={"maxiter": maxiter})
print(f"Optimised mean energy: {res.fun:.4f}")

plt.figure(figsize=(6, 3))
plt.plot(cost_history, color="#2696EB")
plt.xlabel("Iteration")
plt.ylabel("Mean energy")
plt.title("QAOA Optimisation Progress")
plt.show()

# %% [markdown]
# ### Step 4: Sample with Optimized Parameters
#
# One final sample draw at the optimum is what the `## Result` section
# will analyse. The earlier per-iteration samples were consumed inside
# `cost_fn` and discarded.

# %%
gammas_opt = list(res.x[:p])
betas_opt = list(res.x[p:])
final_result = executable.sample(
    executor,
    shots=sample_shots,
    bindings={"gammas": gammas_opt, "betas": betas_opt},
).result()

# %% [markdown]
# ### Step 5: Decode
#
# `converter.decode(result)` converts the bitstring sample set back to
# the problem's original vartype (here `BINARY`, because we built
# `BinaryModel` from a QUBO) and tags each sample with its Ising energy.

# %%
decoded = converter.decode(final_result)
print(f"Number of distinct samples: {len(decoded.samples)}")
print(f"Mean energy: {decoded.energy_mean():.4f}")

# %% [markdown]
# ## Result

# %% [markdown]
# ### Classical Baseline (Brute Force)
#
# Enumerating all $2^4 = 16$ assignments gives us the ground-truth
# MaxCut value to compare the quantum result against. We keep the
# baseline in `## Result` (next to the comparison itself) rather than in
# `## Background`, where it would be disconnected from the quantum
# numbers it is being compared with.

# %%
best_cut = 0
optimal_partitions: list[tuple[int, ...]] = []
for bits in itertools.product([0, 1], repeat=num_nodes):
    cut = sum(1 for i, j in G.edges() if bits[i] != bits[j])
    if cut > best_cut:
        best_cut = cut
        optimal_partitions = [bits]
    elif cut == best_cut:
        optimal_partitions.append(bits)

print(f"Optimal MaxCut value: {best_cut}")
for part in optimal_partitions:
    print(f"  {part}")

# %% [markdown]
# ### Decode and Analyze Results

# %% [markdown]
# #### Best Cut
#
# We scan the decoded samples for the one with the largest MaxCut value
# and compare it to the brute-force optimum from
# `### Classical Baseline (Brute Force)`.

# %%
best_qaoa_cut = 0
best_qaoa_bits: list[int] | None = None
for sample in decoded.samples:
    bits = [sample[i] for i in range(num_nodes)]
    cut = sum(1 for i, j in G.edges() if bits[i] != bits[j])
    if cut > best_qaoa_cut:
        best_qaoa_cut = cut
        best_qaoa_bits = bits

print(f"Best QAOA cut: {best_qaoa_cut}  (optimal: {best_cut})")
print(f"Best partition: {best_qaoa_bits}")

# %% [markdown]
# #### Cut-Value Distribution

# %%
cut_counts: Counter[int] = Counter()
for sample, occ in zip(decoded.samples, decoded.num_occurrences):
    bits = [sample[i] for i in range(num_nodes)]
    cut = sum(1 for i, j in G.edges() if bits[i] != bits[j])
    cut_counts[cut] += occ

cuts = sorted(cut_counts.keys())
counts = [cut_counts[c] for c in cuts]
plt.figure(figsize=(6, 3))
plt.bar([str(c) for c in cuts], counts, color="#2696EB")
plt.xlabel("Cut size")
plt.ylabel("Frequency")
plt.title("Distribution of cut values")
plt.show()

# %% [markdown]
# #### Visualize the Best Partition

# %%
if best_qaoa_bits is not None:
    color_map = [
        "#FF6B6B" if best_qaoa_bits[i] == 1 else "#4ECDC4" for i in range(num_nodes)
    ]
    plt.figure(figsize=(4, 3))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=color_map,
        node_size=600,
        edgecolors="black",
    )
    plt.title(f"QAOA partition (cut = {best_qaoa_cut})")
    plt.show()

# %% [markdown]
# ## Summary
#
# This skeleton walked through the Quantum Approximate Optimization
# Algorithm for MaxCut on a 4-node triangle-plus-pendant graph,
# starting from a QUBO description, training the variational angles
# with a classical optimiser, and recovering a discrete partition
# whose cut value can be compared against the brute-force optimum
# computed in the `## Result` section.
#
# The Qamomile machinery that did the work was a small high-level
# surface: `BinaryModel.from_qubo` folded the QUBO into the Ising
# domain, `QAOAConverter` built the cost and mixer `@qkernel`s and
# managed the gate-convention factor of two internally, and the
# `QiskitTranspiler` produced the executable that
# `executable.sample` drove through a Qiskit estimator each
# optimisation step. The final sample set was lifted back to the
# problem's original vartype with `converter.decode`.
