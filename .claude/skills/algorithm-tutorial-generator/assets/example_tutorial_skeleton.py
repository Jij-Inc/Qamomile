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
# `qamomile.optimization.QAOAConverter` — the high-level converter that
# folds QUBO → Ising conversion, `@qkernel` construction, and decode into
# three method calls. Along the way the tutorial exercises every section
# the section guide prescribes: Backgrounds, Algorithm, Implementation,
# Run example, Conclusion.
#
# 1. Define a 4-node MaxCut instance on a triangle-plus-pendant graph.
# 2. Derive the Ising cost Hamiltonian conceptually.
# 3. Drive the `QAOAConverter` pipeline — transpile, optimize, sample, decode.
# 4. Inspect the sampled distribution and the best cut against brute force.
# 5. Recap and point to the hand-wired counterpart (`qaoa_maxcut.py`).

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %% [markdown]
# ## Backgrounds

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
# to vertex $2$. The graph is small enough to brute-force, and — thanks
# to the odd triangle — not bipartite, so the MaxCut is strictly less
# than the edge count.

# %%
import matplotlib.pyplot as plt
import networkx as nx

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
# ### Exact Solution (Brute Force)
#
# Enumerating all $2^4 = 16$ assignments gives us a ground-truth value to
# compare the quantum result against in §5.

# %%
import itertools

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
#
# Qamomile's rotation gates include a $1/2$ factor
# ($\text{RZ}(\theta) = e^{-i \theta Z / 2}$), but because $\gamma$ and
# $\beta$ are free variational parameters the constant is absorbed into
# the optimum — `QAOAConverter` handles this internally.

# %% [markdown]
# ## Implementation

# %% [markdown]
# ### QUBO to Ising
#
# We build a `BinaryModel` from the MaxCut QUBO. Each edge $(i, j)$
# contributes $-x_i - x_j + 2 x_i x_j$ (this is the MaxCut objective
# negated so the optimiser minimises). `QAOAConverter` then converts
# internally to the spin domain and builds the Ising cost Hamiltonian.
# We print the Hamiltonian so the reader sees the Pauli-Z coefficients.

# %%
from qamomile.optimization.binary_model import BinaryModel
from qamomile.optimization.qaoa import QAOAConverter

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
# ### Transpile to an Executable Circuit
#
# `converter.transpile(transpiler, p=p)` builds a $p$-layer QAOA
# `@qkernel` and hands it to the backend transpiler — here
# `QiskitTranspiler`, which ships with the core install. The angles
# `gammas` and `betas` stay as *runtime parameters* so the classical
# optimiser can vary them each call; every other input (Ising
# coefficients, qubit count, `p`) is bound at compile time.

# %%
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()
p = 2  # number of QAOA layers
executable = converter.transpile(transpiler, p=p)

# %% [markdown]
# ### Optimize the Variational Parameters
#
# We minimise the mean energy with COBYLA. Heavy compute is gated on
# `QAMOMILE_DOCS_TEST` so CI builds stay fast; when running the notebook
# locally the shot count and iteration budget jump to the production
# values.

# %%
import os

import numpy as np
from scipy.optimize import minimize

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
# ### Sample with Optimized Parameters
#
# One final sample draw at the optimum is what §5 will analyse. The
# earlier per-iteration samples were consumed inside `cost_fn` and
# discarded.

# %%
gammas_opt = list(res.x[:p])
betas_opt = list(res.x[p:])
final_result = executable.sample(
    executor,
    shots=sample_shots,
    bindings={"gammas": gammas_opt, "betas": betas_opt},
).result()

# %% [markdown]
# ### Decode
#
# `converter.decode(result)` converts the bitstring sample set back to
# the problem's original vartype (here `BINARY`, because we built
# `BinaryModel` from a QUBO) and tags each sample with its Ising energy.

# %%
decoded = converter.decode(final_result)
print(f"Number of distinct samples: {len(decoded.samples)}")
print(f"Mean energy: {decoded.energy_mean():.4f}")

# %% [markdown]
# ## Run example

# %% [markdown]
# ### Decode and Analyze Results

# %% [markdown]
# #### Best Cut
#
# We scan the decoded samples for the one with the largest MaxCut value
# and compare it to the brute-force optimum from §2.

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
from collections import Counter

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
# ## Conclusion
#
# In this skeleton we:
#
# 1. Defined a small MaxCut instance on a 4-node triangle-plus-pendant
#    graph and brute-forced the optimum.
# 2. Stated the QAOA ansatz and its Ising cost Hamiltonian on paper,
#    without touching any Qamomile API.
# 3. Drove `QAOAConverter` through the full pipeline — set-up,
#    transpile, optimise, sample, decode.
# 4. Compared the sampled distribution to the brute-force optimum.
#
# **Known limitations.** QAOA at shallow depth (small $p$) is known to
# under-perform classical heuristics on dense graphs, and the classical
# optimiser's landscape becomes harder with growing $p$ and $n$ — the
# mean energy plot flattening out near a local minimum is normal.
#
# **Next steps.**
#
# - For a hand-wired ansatz that exposes every QAOA gate (and recovers
#   the same circuit via `qamomile.circuit.algorithm.qaoa_state`), see
#   [QAOA for MaxCut](../vqa/qaoa_maxcut).
# - For a constrained optimisation problem solved with the same
#   converter, see [QAOA for Graph Partitioning](../optimization/qaoa_graph_partition).
