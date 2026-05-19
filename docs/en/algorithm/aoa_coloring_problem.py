# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: qamomile-env
#     language: python
#     name: qamomile-env
# ---

# %% [markdown]
# ---
# tags: [algorithm, optimization, variational]
# ---
#
# # Alternating Operator Ansatz for Graph Coloring
#
# This tutorial demonstrates how to solve the **K - graph coloring problem**
# with Qamomile using the Alternating Operator Ansatz (AOA).
#
# >Hadfield, S.; Wang, Z.; O’Gorman, B.; Rieffel, E.G.; Venturelli, D.; Biswas, R. From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz. Algorithms 2019, 12, 34. https://doi.org/10.3390/a12020034
#
# The AOA algorithm extends QAOA by using more general mixers and initial states.
#
# ### Why does this matter for graph coloring? 
# The one-hot encoding uses $N \times K$ binary variables but only $K^N$
# of the $2^{NK}$ bitstrings are feasible (one color per node). 
# On the 5-node, 3-color instance below, that is 243 feasible states inside 
# a $2^{15} = 32768$-dimensional Hilbert space, i.e. less than $1\%$. 
# Standard QAOA starts in a uniform superposition over the full space and uses a transverse-field mixer 
# ($\sum X_i$) that freely rotates qubits in and out of the feasible subspace, 
# so most of the sampled bitstrings violate the one-hot constraint and get discarded. 
# AOA fixes both ends: it starts inside the feasible subspace and uses an XY mixer that 
# preserves Hamming weight, so every sample is feasible by construction.
#
# We will proceed as follows:
#
# 1. Formulate the problem with [JijModeling](https://jij-inc-jijmodeling-tutorials-en.readthedocs-hosted.com/en/latest/introduction.html).
# 2. Create an instance with concrete data.
# 3. Use `AOAConverter` to build the AOA circuit with designated mixer and initial state.
# 4. Optimize the variational parameters with a classical optimizer.
# 5. Sample the optimized circuit and decode the results. We will highlight the feasibility advantage over QAOA

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %% [markdown]
# ## Problem Formulation
#
# Given an undirected graph $G = (V, E)$, the goal is to find a coloring of $G$ using a given number $K$ of colors such that the number of adjacent vertex of $G$ having the same color is minimal.
#
# **Objective:**
#
# $$
# \min \sum_{(u, v) \in E} \sum_{i=0}^{K-1} x_{u, i} x_{v, i}
# $$
#
# **Constraint:**
#
# $$
# \sum_{i=0}^{K-1} x_{u, i}=1, \forall u \in\{0, \ldots, N-1\}
# $$
#
# where $x_{u, i} \in\{0,1\}$ indicates whether color $i$ is used for vertex $u$ or not.

# %% [markdown]
# ## Define the Problem with JijModeling

# %%
import jijmodeling as jm

@jm.Problem.define("Graph Coloring", sense=jm.ProblemSense.MINIMIZE)
def graph_coloring_decorated(problem : jm.DecoratedProblem):
    N = problem.Length()
    K = problem.Natural()

    E = problem.Graph()

    x = problem.BinaryVar(
        shape=(N, K),
        description="$x_{i,k}$ is 1 if node $i$ is colored with color $k$, 0 otherwise",
    )

    problem += jm.sum(
        x[u, i] * x[v, i] for (u, v) in E for i in K
    )

    problem += problem.Constraint(
        "ColoringConstraint",
        (jm.sum(x[u, i] for i in K) == 1 for u in N),
        description="Each node must be colored with exactly one color"
    )

graph_coloring_decorated

# %% [markdown]
# ### Reading the constraint as a Hamming-weight condition
#
# The constraint $\sum_{i=0}^{K-1} x_{u,i} = 1$ says: among the $K$ binary variables attached to node $u$, exactly one is $1$. When we lay these out as a bitstring, each node occupies a *block* of $K$ consecutive qubits, and feasibility means **each block has Hamming weight exactly 1**.
#
# For our 5-node, 3-color instance the qubit layout is:
#
# ```
#   [q0 q1 q2] [q3 q4 q5] [q6 q7 q8] [q9 q10 q11] [q12 q13 q14]
#    node 0     node 1     node 2     node 3        node 4
# ```
#
# A feasible state has one `1` in each bracketed block. This block structure is exactly what `block_size` will refer to later: we tell the converter to prepare a Dicke state *inside each block* and to mix *only within each block*, which keeps every block at Hamming weight 1 throughout the circuit.

# %% [markdown]
# ## Graph Instance
#
# We use a fixed 5-node graph with 6 edges for reproducibility.

# %%
import networkx as nx
import matplotlib.pyplot as plt

num_nodes = 5
edge_list = [(0, 2), (0, 3), (0, 4), (1, 3), (2, 4), (3, 4)]

G = nx.Graph()
G.add_nodes_from(range(num_nodes))
G.add_edges_from(edge_list)

pos = nx.spring_layout(G, seed=1)
plt.figure(figsize=(5, 5))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="white",
    node_size=700,
    edgecolors="black",
)
plt.title(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
plt.show()


# %% [markdown]
# ## Create the Instance
#
# We extract the edge list from the graph and evaluate the JijModeling
# problem with the concrete data.

# %%
num_colors = 3
instance_data = {"N": num_nodes, "E": edge_list, "K": num_colors}
instance = graph_coloring_decorated.eval(instance_data)

# %% [markdown]
# ## Set Up the AOAConverter
#
# `AOAConverter` takes an OMMX instance and internally convert it into a QUBO form and build the Hamiltonian (see QAOA tutorial for more details).
#
# Recall:
# 1. The energy values from the decoded samples are **not** the original objective, it includes **penalty terms**.
# 2. We need to re-evaluate the true objective values separately

# %%
from qamomile.optimization.aoa import AOAConverter

converter = AOAConverter(instance)
converter.spin_model = converter.spin_model.normalize_by_abs_max()
hamiltonian = converter.get_cost_hamiltonian()
print(hamiltonian)

# %% [markdown]
# ## Transpile to an Executable Circuit
#
# `converter.transpile()` works exactly as in the QAOA case.
#
# In addition, we can select what kind of XY mixer and initial state we want to use.
#
# For initial state, we can choose between:
#
# - `single_basis_state` : a single computational-basis state with the right Hamming weight in each block (e.g. $|100\rangle|100\rangle\ldots$, meaning every node starts colored with color 0). Feasible, deterministic, cheap to prepare. The mixer is responsible for spreading amplitude to other feasible states. This is the simplest choice and a good first thing to try.
#
# - `dicke` : a Dicke state in each block: an equal superposition over all bitstrings with the chosen Hamming weight. For Hamming weight 1 and block size $K$, this is $\frac{1}{\sqrt{K}}(|10\ldots 0\rangle + |01\ldots 0\rangle + \cdots + |0\ldots 01\rangle)$ in each block, so the full initial state is already a uniform superposition over the *entire feasible space*. Costs extra gates to prepare ($O(K)$ XY rotations per block) but gives the optimizer an unbiased starting point.
#
# - `uniform` : Hadamard on every qubit, the standard QAOA initial state. **This puts amplitude on infeasible states**, so the feasibility guarantee of AOA is lost; samples will need to be filtered like in plain QAOA. Useful mainly as a baseline or for ablations comparing initial states.
#
# The `hamming_weight` parameter sets the target Hamming weight per block. For graph coloring with one-hot encoding it's always `1`. For other problems (e.g. cardinality-constrained optimization where exactly $k$ items must be selected), it would be different.
#
# For mixer, we can select:
#
# - `ring` : connects each qubit in a block only to its two neighbors in a cycle. Implemented via a *parity* decomposition (odd pairs, then even pairs, then the wrap-around), so each layer costs $O(K)$ two-qubit gates per block. Shallower circuit, but amplitude takes more layers to spread across the block.
#
# - `fully-connected` : connects every pair of qubits within a block. Implemented via a *partition* decomposition (the $\binom{K}{2}$ pairs are split into non-overlapping rounds), so each layer costs $O(K^2)$ two-qubit gates per block. Deeper circuit, but amplitude mixes across the block in a single layer.
#
# Rule of thumb: prefer `ring` when $K$ is large and circuit depth is the bottleneck; prefer `fully-connected` when $K$ is small (say $\leq 4$) and you want faster mixing and more diverse samples. In practice on small instances `fully-connected` tends to produce more distinct feasible solutions per shot.
#
# The parameter `block_size` can be a bit difficult to understand.
# It is used to define the size of each block on which a Dicke state is built and the mixers apply.
# In our case for the graph coloring problem:
# - we have $K$ qubits for one node which can take $K$ colors
# - the constraint states that each node can only take a single color. So within each subset of $K$ qubits, the hamming_weight (number of ones) should remain equal to $1$.
# - we set `block_size = num_colors` to state: "prepare a Dicke state in each subset of $K$ qubits and mix only within each subset".
#
# The behavior is illustrated thoroughly in the next section using the draw function.

# %%
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()
p = 5  # number of AOA layers

executable_aoa_dicke = converter.transpile(
    transpiler,
    p=p,
    initial_state="dicke",
    hamming_weight=1,
    mixer="fully-connected",
    block_size=num_colors,
)

# %% [markdown]
# ## Visualize the AOA Circuit
#
# We can draw the AOA circuit with the same method as in the QAOA tutorial, using `Transpiler.to_block`, `Transpiler.inline` and then `MatplotlibDrawer.draw`.
#
# For readability we use `p=1`.
#
# We need to compute the indices consumed by the qkernel by calling the method `compute_dicke_composition_schedule` and `resolve_pair_indices` of the `converter` class.

# %%
import qamomile.circuit as qmc
from qamomile.circuit.algorithm.aoa import aoa_state_dicke
from qamomile.circuit.visualization import MatplotlibDrawer

#We can access the internal logic of the convertor to get the indices for dicke state preparation and mixer construction.
#This is useful for visualizing but not part of the normal user workflow.
initial_ones,pair_indices_dicke,triplets_indices_dicke,pair_angles_dicke,triplets_angles_dicke=converter.compute_dicke_composition_schedule(hamming_weight=1, block_size=num_colors)
resolved_pair = converter.resolve_pair_indices(mixer="fully-connected", pair_indices=None, block_size=num_colors)

block = transpiler.to_block(
    aoa_state_dicke,
    bindings={
        "linear": converter.spin_model.linear,
        "quad": converter.spin_model.quad,
        "n": converter.spin_model.num_bits,
        "p": 1,
        "pair_indices_mixer": resolved_pair,
        "initial_ones": initial_ones,
        "pair_indices_dicke": pair_indices_dicke,
        "triplets_indices_dicke": triplets_indices_dicke,
        "pair_angles_dicke": pair_angles_dicke,
        "triplets_angles_dicke": triplets_angles_dicke,
    },
    parameters=["gammas", "betas"],
)

block = transpiler.inline(block)
fig = MatplotlibDrawer(block).draw(
    inline=True,
    fold_loops=False,
    expand_composite=True,
    inline_depth=None,
)
fig

# %% [markdown]
# ### Inspecting the Building Blocks
#
# Let's take some time to explain this circuit and illustrate the use of `block_size`.
# `block_size` is set on $K=3$ (the number of color) in this example.
#
# Inside `aoa_state_dicke`, we call several qkernels:
#
# - `prepare_dicke(n,initial_ones,...)` — on the first column several $X$ gates to create the original basis state with given hamming weight for each block. Then a sequence of $R_Y$ and CNOT gates, is used for building the Dicke state inside each block: superposition of all state of same hamming weight.
#
# - `ising_cost(quad, linear, q, gamma)` — the cost layer: same as QAOA. Uses $R_Z$ and $R_ZZ$ rotation gates.
#
# - `xy_mixer(q, betas[layer], pair_indices_mixer)` — a bunch of CNOT and $R_X$ gates are applied to each block for the mixer layer.
#
# `aoa_layers(p, ...)` is just the alternation of `ising_cost` and
# `xy_mixer`, repeated `p` times.

# %%
from qamomile.circuit.algorithm.state_preparation import prepare_dicke
from qamomile.circuit.algorithm.qaoa import ising_cost
from qamomile.circuit.algorithm.aoa import xy_mixer

block = transpiler.to_block(
    prepare_dicke,
    bindings={
        "n": converter.spin_model.num_bits,
        "initial_ones": initial_ones,
        "pair_indices": pair_indices_dicke,
        "triplets_indices": triplets_indices_dicke,
        "pair_angles": pair_angles_dicke,
        "triplets_angles": triplets_angles_dicke,
    },
    parameters=[],
)
block = transpiler.inline(block)
fig = MatplotlibDrawer(block).draw(
    inline=True,
    fold_loops=False,
    expand_composite=True,
    inline_depth=None,
)
fig.set_size_inches(100, 8)
fig

# %%
ising_cost.draw(
    q=converter.spin_model.num_bits,
    quad=converter.spin_model.quad,
    linear=converter.spin_model.linear,
    fold_loops=False,
)

# %%
fig = xy_mixer.draw(
    q=converter.spin_model.num_bits,
    pair_indices_mixer=resolved_pair,
    inline=True,
    fold_loops=False,
    expand_composite=True,
    inline_depth=None,
)
fig.set_size_inches(100, 8)
fig

# %% [markdown]
# ## Optimizing the Alternating Operator Ansatz parameters
#
# We use `executable.sample()` to evaluate the cost at each iteration of the
# classical optimizer. The optimizer explores different `gammas` and `betas`
# to minimize the mean energy of the sampled bitstrings.

# %%
import os

import numpy as np
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

# Seed the simulator so re-executing the notebook reproduces the same
# COBYLA trajectory and final sampling distribution. Without a seed,
# every shot draws fresh randomness, COBYLA sees a noisy cost surface,
# and each notebook run converges to a different (but equivalent) local
# optimum.
executor = transpiler.executor(
    backend=AerSimulator(seed_simulator=901, max_parallel_threads=1)
)
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
sample_shots = 256 if docs_test_mode else 2048
maxiter = 25 if docs_test_mode else 1000

rng = np.random.default_rng(900)
initial_params = rng.uniform(0, np.pi, 2 * p)

cost_history = []


def cost_fn(params):
    gammas = list(params[:p])
    betas = list(params[p:])
    job = executable_aoa_dicke.sample(
        executor,
        shots=sample_shots,
        bindings={"gammas": gammas, "betas": betas},
    )
    result = job.result()
    # decode_to_binary_sampleset returns the QUBO-domain BinarySampleSet
    # whose `energy` is the penalized objective — what COBYLA needs to
    # see infeasibility costs. The polymorphic decode() returns an
    # ommx.v1.SampleSet whose `objective` is the un-penalized true
    # objective; using it here would let the optimizer settle on
    # infeasible all-zero / all-one bitstrings.
    decoded = converter.decode_to_binary_sampleset(result)
    energy = decoded.energy_mean()
    cost_history.append(energy)
    return energy


res = minimize(
    cost_fn,
    initial_params,
    method="COBYLA",
    options={"maxiter": maxiter},
)

print(f"Optimized cost: {res.fun:.3f}")
print(f"Optimal params: {[round(v, 4) for v in res.x]}")
print(f"Function evaluations: {res.nfev}")

# %%
plt.figure(figsize=(8, 4))
plt.plot(cost_history, color="#2696EB")
plt.xlabel("Iteration")
plt.ylabel("Cost (mean energy)")
plt.title("AOA Optimization Progress")
plt.show()

# %% [markdown]
# ## Sample with Optimized Parameters
#
# With the optimized parameters, we sample the circuit to collect
# candidate solutions as bitstrings.

# %%
gammas_opt = list(res.x[:p])
betas_opt = list(res.x[p:])

sample_result = executable_aoa_dicke.sample(
    executor,
    shots=1000,
    bindings={"gammas": gammas_opt, "betas": betas_opt},
).result()

# `decode()` on an OMMX-backed converter returns an `ommx.v1.SampleSet`
# evaluated against the *original* (un-penalized) instance, so
# feasibility, the true objective, and per-constraint diagnostics are
# available through OMMX's own API — no hand-rolled feasibility or
# objective helper is needed.
sample_set = converter.decode(sample_result)

# %% [markdown]
# ## Analyze the Results
#
# ### Feasibility Check
#
# We can now check the advantage of AOA compared to QAOA.
#  
# The problem-tailored XY mixer that we implemented should
# allow one to remain within the feasible space. Thus, 
# unlike the QAOA case, **candidate solutions** proposed by AOA
# should all live into the feasible space.
# That means, they respect the constraint hamming weight $=1$

# %%
summary = sample_set.summary
total_feasible = int(summary["feasible"].sum())
total_samples = len(summary)

print(
    f"Feasible samples: {total_feasible} / {total_samples} "
    f"({100 * total_feasible / total_samples:.1f}%)"
)

# %% [markdown]
# ### Best Feasible Solution
#
# `SampleSet.best_feasible` returns the feasible sample with the
# best (here: smallest) objective. 

# %%
best = sample_set.best_feasible
df = best.decision_variables_df
x_rows = df[df["name"] == "x"]

best_coloring = {}
for _, row in x_rows.iterrows():
    node, color = row["subscripts"]
    if row["value"] > 0.5:
        best_coloring[int(node)] = int(color)

print("Best coloring:", best_coloring)
print("Number of Conflicts (neighbor nodes with the same color):", int(round(best.objective)))

# %% [markdown]
# ### Objective Value Distribution
#
# We plot the distribution of the true objective value (number of conflicts). 

# %%
obj_counts = summary.value_counts().sort_index()

plt.figure(figsize=(8, 4))
plt.bar([str(o) for o in obj_counts.index], obj_counts.values, color="#2696EB")
plt.xlabel("Conflicts (objective value)")
plt.ylabel("Frequency")
plt.title("Distribution of Solutions")
plt.show()


# %% [markdown]
# ### Visualize the Best 3-Coloring
#
# We color the graph nodes according to the best feasible coloring found by the Alternating Operator Ansatz algorithm.

# %%
def get_color_map_from_solution(best_coloring, num_colors):
    """ Creates a color map for the best solution."""
    
    colors = ["#FF6B6B", "#4ECDC4", "#1A535C"]
    color_map = []
    for u in range(num_nodes):
        color_idx = best_coloring.get(u, 0)  # default to 0 if node not found
        color_map.append(colors[color_idx])
    return color_map

color_map = get_color_map_from_solution(best_coloring, num_colors)

pos = nx.spring_layout(G, seed=1)
plt.figure(figsize=(5, 5))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=color_map if color_map else "white",
    node_size=700,
    edgecolors="black",
)
plt.title(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
plt.show()
