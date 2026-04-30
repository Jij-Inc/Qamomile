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
# # PCE for MaxCut: Solving 20 Variables on 3 Qubits
#
# This tutorial walks through Pauli Correlation Encoding (PCE) for the
# MaxCut problem using Qamomile's `PCEConverter`. PCE compresses $N$
# binary variables into the expectation values of $k$-body Pauli
# correlators on a much smaller register of $n = \mathcal{O}(N^{1/k})$
# qubits, enabling variational optimization at problem sizes that would
# not fit on near-term hardware under one-qubit-per-variable schemes
# like QAOA.
#
# We will solve a 20-vertex MaxCut instance on **just 3 qubits** with
# $k = 2$, and verify the result against a brute-force optimum. Steps:
#
# 1. Define a 20-node MaxCut problem and brute-force its optimum.
# 2. Build the PCE encoding via `PCEConverter(instance, k=2)` and read
#    out the per-variable Pauli observables with `get_encoded_pauli_list()`.
# 3. Write a hardware-efficient `@qkernel` ansatz and read each
#    correlator's expectation with `qm.expval(q, P)`.
# 4. Train the ansatz against a tanh-relaxed surrogate of the MaxCut
#    objective using `scipy.optimize.minimize`.
# 5. Decode the optimized expectations into a bitstring with
#    `converter.decode(expectations)` and visualize the partition.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %% [markdown]
# ## Backgrounds

# %% [markdown]
# ### What is MaxCut?
#
# Given an undirected graph $G = (V, E)$, the **MaxCut** problem asks us
# to partition the vertices into two sets $S$ and $\bar{S}$ so that the
# number of edges between the two sets is maximized:
#
# $$
# \text{MaxCut}(x) = \sum_{(i,j) \in E} \bigl[\,x_i (1 - x_j) + x_j (1 - x_i)\,\bigr]
# $$
#
# where $x_i \in \{0, 1\}$ indicates which set vertex $i$ belongs to.

# %% [markdown]
# ### Create the Graph
#
# We use a 20-node random graph with 30 edges (density $\approx 0.16$).
# This is large enough that the PCE compression is meaningful — 20
# variables encoded into 3 qubits — and small enough that brute force
# still gives us a ground truth.
#
# `nx.gnm_random_graph` can produce disconnected graphs (an isolated
# vertex contributes a free spin and degenerates the optimization
# landscape), so we bump the seed until we land on a connected graph.

# %%
import matplotlib.pyplot as plt
import networkx as nx

seed = 42
while True:
    G = nx.gnm_random_graph(20, 30, seed=seed)
    if nx.is_connected(G):
        break
    seed += 1
print(f"Using seed = {seed} (smallest seed >= 42 producing a connected graph)")

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(6, 5))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="white",
    node_size=500,
    edgecolors="black",
)
plt.title(f"Graph: {num_nodes} nodes, {num_edges} edges")
plt.show()

# %% [markdown]
# ### Exact Solution (Brute Force)
#
# Enumerating all $2^{20} = 1{,}048{,}576$ partitions is just over a
# million bitstrings — too many for a Python loop, but a single pass of
# vectorised NumPy bit arithmetic finishes in a fraction of a second.
# This gives us a ground truth to compare PCE against in §5.

# %%
import numpy as np

assignments = np.arange(2**num_nodes, dtype=np.int64)
cuts = np.zeros(2**num_nodes, dtype=np.int32)
for i, j in G.edges():
    cuts += np.bitwise_xor(
        np.right_shift(assignments, i) & 1,
        np.right_shift(assignments, j) & 1,
    ).astype(np.int32)

best_cut = int(cuts.max())
optimal_assignment_ints = np.flatnonzero(cuts == best_cut)
print(f"Optimal MaxCut value         : {best_cut}")
print(f"Number of optimal partitions : {len(optimal_assignment_ints)}")

# %% [markdown]
# ## Algorithm
#
# PCE was introduced by Sciorilli, Borges, Patti, García-Martín, Camilo,
# Anandkumar, and Aolita (https://doi.org/10.48550/arXiv.2401.09421) as
# a way of pushing combinatorial optimization deep into the qubit-count
# regime where QAOA-style one-qubit-per-variable encodings cannot fit.
# For an $N$-variable problem PCE uses only $n = \mathcal{O}(N^{1/k})$
# qubits.

# %% [markdown]
# ### PCE Encoding
#
# PCE picks a **compression rate** $k > 1$ and assigns each binary
# variable $i \in \{1, \dots, N\}$ to a distinct $k$-body Pauli
# correlator $P_i$ — a tensor product of $k$ non-identity Paulis from
# $\{X, Y, Z\}$ acting on $k$ of the $n$ qubits. The number of distinct
# such correlators on $n$ qubits is $\binom{n}{k} \cdot 3^k$, so $n$ is
# chosen as the smallest integer with
#
# $$
# \binom{n}{k} \cdot 3^k \;\ge\; N.
# $$
#
# At $k = 2$ this gives $n = \mathcal{O}(\sqrt{N})$; at $k = 3$ it gives
# $n = \mathcal{O}(N^{1/3})$. For our $N = 20$ instance, $k = 2$ requires
# only $n = 3$ qubits — the smallest integer with
# $\binom{3}{2} \cdot 9 = 27 \ge 20$. The mapping is deterministic —
# Qamomile's `PCEEncoder` enumerates correlators in a fixed
# lexicographic order (first qubit indices, then Pauli labels) and
# assigns the first $N$ of them to variables $0, \dots, N-1$.

# %% [markdown]
# ### Cost Function
#
# Given a parameterized ansatz state $|\Psi(\boldsymbol{\theta})\rangle$,
# PCE turns the discrete spin objective
#
# $$
# C(\mathbf{s}) \;=\; \sum_i h_i \, s_i \,+\, \sum_{i<j} J_{ij} \, s_i s_j
# $$
#
# into a smooth surrogate by replacing each spin $s_i$ with the
# **tanh-relaxed** correlator expectation
# $\sigma_i(\boldsymbol{\theta}) = \tanh\bigl(\alpha\, \langle P_i \rangle\bigr)$,
# and adds a quartic **regularizer** that keeps every correlator inside
# the open interval $(-1, +1)$ — the open domain where every bitstring
# is still expressible:
#
# $$
# \mathcal{L}(\boldsymbol{\theta})
# \;=\; \underbrace{\sum_i h_i \, \sigma_i \,+\, \sum_{i<j} J_{ij} \, \sigma_i \sigma_j}_{\mathcal{L}_{\text{data}}}
#       \,+\, \mathcal{L}_{\text{reg}}, \qquad
# \mathcal{L}_{\text{reg}}
# \;=\; \beta \cdot \nu \cdot \!\left[ \frac{1}{N} \sum_i \sigma_i^2 \right]^{\!2}.
# $$
#
# Intuition: the data term pulls $\sigma_i$ and $\sigma_j$ toward
# opposite signs for every connected pair (so $J_{ij} \sigma_i \sigma_j$
# is negative); the regularizer counterbalances by penalizing premature
# saturation, keeping the optimizer in the smooth interior of the
# correlator domain instead of locking onto a poor candidate bitstring
# early.
#
# Hyperparameters, following https://doi.org/10.48550/arXiv.2401.09421:
#
# - **$\alpha$** (tanh sharpness): scaled as
#   $\alpha \approx n^{\lfloor k/2 \rfloor}$. For $k = 2$ this is
#   $\alpha \approx n$, so we use $\alpha = n = 3$ for our 3-qubit run.
# - **$\beta = 1/2$** (regularizer strength): a fixed value the paper
#   tunes once on random graphs and reuses across experiments.
# - **$\nu$** (overall scale): not a free hyperparameter — it is the
#   Edwards–Erdős MaxCut bound,
#   $\nu = |E|/2 + (N - 1)/4$, computed directly from the graph.
#
# For MaxCut specifically, the spin model has $h_i = 0$ and
# $J_{ij} = +\tfrac{1}{2}$ on every edge, so the data term is minimized
# precisely when adjacent $\sigma_i, \sigma_j$ disagree in sign.

# %% [markdown]
# ### Decoding
#
# After convergence, PCE rounds each correlator's optimized expectation
# back to a discrete spin via the sign function:
#
# $$
# s_i \;=\; \operatorname{sgn}\!\bigl\langle P_i \bigr\rangle_{\boldsymbol{\theta}^*}
# \;\in\; \{+1, -1\},
# $$
#
# and the binary assignment is recovered as $x_i = (1 - s_i) / 2$.

# %% [markdown]
# ### Ansatz Choice
#
# PCE does not prescribe a fixed circuit — the original paper uses a
# **hardware-efficient brickwork ansatz**: alternating layers of
# single-qubit rotations and two-qubit entanglers, with the parameter
# count scaling linearly in the number of variables. We use Qamomile's
# pre-built layers (`ry_layer`, `rz_layer`, `cz_entangling_layer`) from
# `qamomile.circuit.algorithm.basic` and stack them `depth` times,
# giving $2 \cdot n \cdot \text{depth}$ variational angles in total.

# %% [markdown]
# ### Gate-Convention Note
#
# Qamomile's rotation gates carry the standard $1/2$ factor:
# $\text{RY}(\theta) = e^{-i \theta Y / 2}$ and
# $\text{RZ}(\theta) = e^{-i \theta Z / 2}$. Because every entry of the
# `thetas` vector is a **purely variational parameter** (the optimizer
# is free to scale it), this constant factor is simply absorbed into
# the optimal `thetas` values — we pass `thetas[i]` directly without
# inserting an explicit factor of $2$.

# %% [markdown]
# ## Implementation

# %% [markdown]
# ### Build the BinaryModel and PCEConverter
#
# We map MaxCut to a QUBO and wrap it in a `BinaryModel`, then hand the
# model and the chosen compression rate $k = 2$ to `PCEConverter`. The
# converter immediately builds its `PCEEncoder`, internally rewrites
# the model to spin form (using $x_i = (1 - s_i) / 2$), and computes
# the qubit count.

# %%
from qamomile.optimization.binary_model import BinaryModel
from qamomile.optimization.pce import PCEConverter

# Build QUBO: minimize -MaxCut. Pre-seed every node with a zero diagonal
# entry so BinaryModel.from_qubo registers all N variables even if the
# generator ever produces an isolated vertex.
qubo: dict[tuple[int, int], float] = {(v, v): 0.0 for v in G.nodes()}
for i, j in G.edges():
    qubo[(i, i)] -= 1.0
    qubo[(j, j)] -= 1.0
    qubo[(i, j)] = qubo.get((i, j), 0.0) + 2.0

binary_model = BinaryModel.from_qubo(qubo)
converter = PCEConverter(binary_model, k=2)

spin_model = converter.spin_model
print(f"Number of variables  : {spin_model.num_bits}")
print(f"PCE qubit count      : {converter.num_qubits}")
print(f"Compression rate     : k = {converter.k}")
print(f"Compression factor   : {spin_model.num_bits / converter.num_qubits:.1f}x")

# %% [markdown]
# ### Inspect the Per-Variable Pauli Observables
#
# `get_encoded_pauli_list()` returns one Hamiltonian per variable, each
# containing exactly one $k$-body Pauli string with coefficient $1$.
# These are the $P_i$ observables of §3 — the optimizer will read them
# with `qm.expval` inside the ansatz kernel. The same enumeration also
# lives on the underlying `PCEEncoder` (`converter.encoder`) for
# inspection without going through the converter.

# %%
observables = converter.get_encoded_pauli_list()

print(f"Total observables : {len(observables)}")
for i, P_i in enumerate(observables):
    print(f"  P_{i:2d}: {P_i}")

# %% [markdown]
# ### Define the Hardware-Efficient Ansatz
#
# The ansatz starts in the uniform superposition and applies `depth`
# brickwork layers of `ry_layer` + `rz_layer` + `cz_entangling_layer`.
# The kernel returns the expectation value $\langle P \rangle$ where
# `P` is one specific observable supplied via bindings — the same
# kernel is transpiled once per $P_i$.

# %%
import qamomile.circuit as qmc
from qamomile.circuit.algorithm.basic import (
    cz_entangling_layer,
    ry_layer,
    rz_layer,
)


@qmc.qkernel
def pce_ansatz(
    n: qmc.UInt,
    depth: qmc.UInt,
    thetas: qmc.Vector[qmc.Float],
    P: qmc.Observable,
) -> qmc.Float:
    q = qmc.qubit_array(n, name="q")
    for i in qmc.range(n):
        q[i] = qmc.h(q[i])
    for d in qmc.range(depth):
        offset = d * 2 * n
        q = ry_layer(q, thetas, offset)  # type: ignore[arg-type]
        q = rz_layer(q, thetas, offset + n)  # type: ignore[arg-type,operator]
        q = cz_entangling_layer(q)
    return qmc.expval(q, P)


# %% [markdown]
# ### Transpile One Executable per Observable
#
# Each $P_i$ produces a different expectation-value path through the
# transpiler, so we transpile the kernel once per observable and cache
# the resulting `ExecutableProgram`s. The compile-time `bindings` fix
# the structural inputs (`n`, `depth`, `P`); `parameters=["thetas"]`
# leaves the variational angles as runtime parameters that the
# optimizer will set on every call.

# %%
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

n = converter.num_qubits
depth = 3
num_thetas = 2 * n * depth

executables = [
    transpiler.transpile(
        pce_ansatz,
        bindings={"n": n, "depth": depth, "P": P_i},
        parameters=["thetas"],
    )
    for P_i in observables
]

print(f"Executables cached : {len(executables)}")
print(f"Variational params : {num_thetas} (= 2 * n * depth)")

# %% [markdown]
# ### Optimize the Variational Parameters
#
# The classical loop evaluates $\langle P_i \rangle$ for every
# observable at the current `thetas`, plugs the values into the
# tanh-relaxed loss from §3 (data term + regularizer), and asks
# `scipy.optimize.minimize` for an update. We track the loss history
# so the reader can see the optimizer converge.
#
# `QAMOMILE_DOCS_TEST` keeps `maxiter` small for the CI doc build.

# %%
import os

from scipy.optimize import minimize

executor = transpiler.executor()
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
maxiter = 30 if docs_test_mode else 300

# Hyperparameters from https://doi.org/10.48550/arXiv.2401.09421:
#   alpha = n^floor(k/2) — for k=2 this is alpha = n = num_qubits
#   beta  = 1/2 (fixed, paper tunes once on random graphs)
#   nu    = |E| / 2 + (N - 1) / 4 (Edwards-Erdős unweighted MaxCut bound)
N = spin_model.num_bits
alpha = float(n)
beta = 0.5
nu = num_edges / 2 + (N - 1) / 4
print(f"alpha = {alpha}, beta = {beta}, nu = {nu}")

cost_history: list[float] = []


def measure_expectations(thetas: list[float]) -> list[float]:
    return [
        exe.run(executor, bindings={"thetas": thetas}).result()
        for exe in executables
    ]


def loss(params: np.ndarray) -> float:
    thetas = list(params)
    expvals = measure_expectations(thetas)
    relaxed = [np.tanh(alpha * e) for e in expvals]

    # Data term: smooth surrogate of the spin objective.
    L_data = 0.0
    for (i, j), J_ij in spin_model.quad.items():
        L_data += J_ij * relaxed[i] * relaxed[j]
    for i, h_i in spin_model.linear.items():
        L_data += h_i * relaxed[i]

    # Regularizer: beta * nu * [(1/N) sum tanh^2(alpha <P_i>)]^2.
    mean_sq = sum(r**2 for r in relaxed) / N
    L_reg = beta * nu * mean_sq**2

    L_total = L_data + L_reg
    cost_history.append(L_total)
    return L_total


rng = np.random.default_rng(42)
initial_params = rng.uniform(-np.pi, np.pi, num_thetas)

res = minimize(
    loss, initial_params, method="COBYLA", options={"maxiter": maxiter}
)

print(f"Final loss: {res.fun:+.4f}")

# %%
plt.figure(figsize=(8, 4))
plt.plot(cost_history, color="#2696EB")
plt.xlabel("Iteration")
plt.ylabel("Tanh-relaxed loss")
plt.title("PCE Optimization Progress")
plt.show()

# %% [markdown]
# ### Decode the Optimized Expectations
#
# `PCEConverter.decode(expectations)` consumes the per-variable
# expectation values, sign-rounds each one to a spin, and returns a
# single-sample `BinarySampleSet` already converted back to the
# original vartype (BINARY, since we built `binary_model` from a QUBO).

# %%
final_expectations = measure_expectations(list(res.x))
sampleset = converter.decode(final_expectations)

print("Final per-variable expectations:")
for i, e in enumerate(final_expectations):
    print(f"  <P_{i:2d}> = {e:+.4f}")
print()
print(f"Decoded vartype : {sampleset.vartype}")
print(f"Decoded energy  : {sampleset.energy[0]:+.4f}")

# %% [markdown]
# ## Run example

# %% [markdown]
# ### Decode and Analyze Results

# %% [markdown]
# #### Best Cut
#
# Convert the decoded bitstring into a graph partition and compare its
# cut value against the brute-force optimum from §2.

# %%
sample = sampleset.samples[0]
bits = [sample[i] for i in range(num_nodes)]
pce_cut = sum(1 for i, j in G.edges() if bits[i] != bits[j])

print(f"PCE bitstring       : {bits}")
print(f"PCE cut value       : {pce_cut}")
print(f"Brute-force optimum : {best_cut}")
print(f"Approximation ratio : {pce_cut / best_cut:.3f}")

# %% [markdown]
# #### Visualize the Best Solution
#
# Colour each node by which side of the partition it landed on. Nodes
# of different colours sit on opposite sides of the cut.

# %%
color_map = ["#FF6B6B" if bits[i] == 1 else "#4ECDC4" for i in range(num_nodes)]
plt.figure(figsize=(6, 5))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=color_map,
    node_size=500,
    edgecolors="black",
)
plt.title(f"PCE partition (cut = {pce_cut} / optimum = {best_cut})")
plt.show()

# %% [markdown]
# ## Conclusion
#
# In this tutorial we:
#
# 1. Defined a 20-node MaxCut instance and brute-forced its optimum
#    over all $2^{20}$ partitions with vectorised NumPy.
# 2. Encoded the 20 spin variables into 2-body Pauli correlators on
#    just **3 qubits** — a roughly 7× compression — with
#    `PCEConverter(binary_model, k=2)` and read the per-variable
#    observables from `get_encoded_pauli_list()`.
# 3. Built a hardware-efficient `@qkernel` ansatz that returns
#    $\langle P \rangle$ via `qm.expval`, transpiled it once per
#    observable, and trained it against the tanh-relaxed MaxCut loss
#    plus the paper's quartic regularizer
#    ($\alpha = n$, $\beta = 1/2$, $\nu = |E|/2 + (N-1)/4$).
# 4. Recovered a discrete bitstring by feeding the optimized
#    expectations into `PCEConverter.decode(...)` and verified the
#    approximation ratio against the brute-force optimum.
#
# **Limitations:**
#
# - **Hyperparameters and ansatz depth still need tuning.** Even with
#   the paper's $\alpha$, $\beta$, $\nu$ scheme, the tanh surrogate can
#   stagnate if the ansatz capacity is too small or the optimizer's
#   budget too short; sweeping over $\alpha$ on random graphs (as the
#   paper does in its SI) is a typical first move.
# - **One transpile per variable.** Estimating $N$ expectations means
#   $N$ separate `transpile` + `run` calls, which dominates the wall
#   clock on small problems even though the qubit count is tiny.
#
# **Next steps:**
#
# - For the canonical one-qubit-per-variable contrast, see
#   [QAOA for MaxCut](qaoa_maxcut), which solves a small graph with
#   a fixed cost Hamiltonian and bitstring sampling.
# - For sub-qubit encodings using a graph-coloring scheme rather than
#   PCE's combinatorial Pauli enumeration, see Qamomile's QRAO
#   converters under `qamomile.optimization.qrao`.
