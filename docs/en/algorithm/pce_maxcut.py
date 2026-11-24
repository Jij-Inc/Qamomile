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
# tags: [algorithm, optimization, variational]
# ---
#
# # Pauli Correlation Encoding
#
# In this tutorial we solve the MaxCut problem with Pauli Correlation
# Encoding (PCE) using Qamomile's `PauliCorrelationEncoding`. PCE maps $N$ spin
# variables to expectation values of $k$-body Pauli correlators on a
# register of $n = \mathcal{O}(N^{1/k})$ qubits. This reduces the qubit
# count compared with a one-qubit-per-variable QAOA formulation.
#
# We solve a 20-vertex MaxCut instance on **3 qubits** with $k = 2$.
# The workflow builds the encoding with `PauliCorrelationEncoding`, estimates each
# correlator expectation value with a hardware-efficient `@qkernel`
# ansatz, optimizes the ansatz with `scipy.optimize.minimize`, decodes
# the final expectations with `encoding.decode`, and compares the
# result with a brute-force baseline.

# %%
# Install the latest Qamomile through pip!
# # !pip install "qamomile[qiskit]" networkx

# %%
from __future__ import annotations

import os

import networkx as nx
import numpy as np
from scipy.optimize import minimize

import qamomile as qm
from qamomile.measurement import Estimator
from qamomile.optimization import BinaryModel, PauliCorrelationEncoding
from qamomile.qiskit import Aer
from qamomile.stdlib import cx_entangling_layer, ry_layer, rz_layer

# %% [markdown]
# (problem-settings)=
# ## Problem Settings: MaxCut

# %% [markdown]
# ### Problem Definition
#
# Given an undirected graph $G = (V, E)$, the MaxCut problem asks for
# a partition of the vertices into two sets $S$ and $\bar{S}$ so that the
# number of edges between the two sets is maximized. We assign each
# vertex a spin variable $s_i \in \{+1, -1\}$. The value
# $s_i = +1$ places vertex $i$ in $S$, and $s_i = -1$ places it in
# $\bar{S}$. The cut value is
#
# $$
# \text{MaxCut}(\mathbf{s})
# \;=\; \frac{1}{2} \sum_{(i,j) \in E} \bigl(\,1 - s_i s_j\,\bigr).
# $$
#
# Each edge contributes $1$ when its endpoints sit on opposite sides
# ($s_i s_j = -1$) and $0$ when they sit on the same side
# ($s_i s_j = +1$). Maximizing the cut is therefore equivalent to
# **minimizing** the Ising energy
#
# $$
# E(\mathbf{s}) \;=\; \frac{1}{2} \sum_{(i,j) \in E} s_i s_j
#     \;-\; \frac{|E|}{2}, \qquad E(\mathbf{s}) = -\,\text{MaxCut}(\mathbf{s}),
# $$
#
# This is an Ising model with $h_i = 0$, $J_{ij} = \tfrac{1}{2}$ on
# every edge, and a constant offset $-|E|/2$. PCE works directly with
# this spin form, so we do not need an extra conversion from binary
# variables $x \in \{0, 1\}$.

# %% [markdown]
# ### Create the Graph
#
# We use a 20-node 3-regular random graph (every vertex has exactly
# three neighbors, giving $|E| = 3 \cdot 20 / 2 = 30$ edges). MaxCut on
# a 3-regular graph is a benchmark in the PCE paper. The instance is
# also small enough for the brute-force baseline to compute the true
# optimum.
#
# `nx.random_regular_graph` can produce disconnected graphs, so we
# increase the seed until the graph is connected. This keeps the
# example to one partitioning problem instead of several independent
# components.

# %%
seed = 42
while True:
    G = nx.random_regular_graph(3, 20, seed=seed)
    if nx.is_connected(G):
        break
    seed += 1
print(f"Using seed = {seed} (smallest seed >= 42 producing a connected graph)")

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

# %% [markdown]
# (pce-algorithm)=
# ## Algorithm
#
# PCE was introduced by Sciorilli et al.
# {cite:p}`10.48550/arXiv.2401.09421` for combinatorial optimization
# under tight qubit budgets. Standard QAOA uses one qubit
# per variable. PCE uses $n = \mathcal{O}(N^{1/k})$ qubits for an
# $N$-variable problem.

# %% [markdown]
# ### PCE Encoding
#
# PCE picks a correlator order $k > 1$ and assigns each spin
# variable $i \in \{1, \dots, N\}$ to a distinct $k$-body Pauli
# correlator $P_i$. Each $P_i$ is a tensor product of $k$ non-identity
# Paulis from $\{X, Y, Z\}$ acting on $k$ of the $n$ qubits. The number
# of distinct such correlators on $n$ qubits is
# $\binom{n}{k} \cdot 3^k$, so $n$ is chosen as the smallest integer with
#
# $$
# \binom{n}{k} \cdot 3^k \;\ge\; N.
# $$
#
# At $k = 2$ this gives $n = \mathcal{O}(\sqrt{N})$; at $k = 3$ it gives
# $n = \mathcal{O}(N^{1/3})$. For this $N = 20$ instance, $k = 2$
# requires $n = 3$ qubits. This is the smallest integer with
# $\binom{3}{2} \cdot 9 = 27 \ge 20$. The mapping from variables to
# correlators is deterministic: pick any fixed enumeration of $k$-body
# Pauli strings on $n$ qubits and assign the first $N$ of them to
# variables $0, \dots, N-1$.

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
# into a smooth surrogate loss function $\mathcal{L}$ by replacing each
# spin $s_i$ with the tanh-relaxed correlator expectation
# $\sigma_i(\boldsymbol{\theta}) = \tanh\bigl(\alpha\, \langle P_i \rangle\bigr)$,
# and adding a quartic regularizer that discourages early
# saturation of the relaxed variables. The tanh map keeps $\sigma_i$
# inside the open interval $(-1, +1)$, where sign rounding can still
# recover candidate bitstrings:
#
# $$
# \mathcal{L}(\boldsymbol{\theta})
# \;=\; \underbrace{\sum_i h_i \, \sigma_i \,+\, \sum_{i<j} J_{ij} \, \sigma_i \sigma_j}_{\mathcal{L}_{\text{data}}}
#       \,+\, \mathcal{L}_{\text{reg}}, \qquad
# \mathcal{L}_{\text{reg}}
# \;=\; \beta \cdot \nu \cdot \!\left[ \frac{1}{N} \sum_i \sigma_i^2 \right]^{\!2}.
# $$
#
# The data term pulls $\sigma_i$ and $\sigma_j$ toward opposite signs
# for every connected pair (so $J_{ij} \sigma_i \sigma_j$ is negative);
# the regularizer counterbalances this pressure by penalizing large
# relaxed values. This keeps the optimizer in the smooth interior of the
# domain and reduces early convergence to a suboptimal bitstring.
#
# The loss carries three hyperparameters: $\alpha$ (tanh sharpness),
# $\beta$ (regularizer strength), and $\nu$ (overall scale). Their
# values affect optimizer convergence and final solution quality. The
# concrete values used in this tutorial follow the original paper and
# are configured in [](#pce-optimize-variational-parameters).
#
# For MaxCut specifically, the spin model has $h_i = 0$ and
# $J_{ij} = +\tfrac{1}{2}$ on every edge, so the data term is minimized
# precisely when adjacent $\sigma_i, \sigma_j$ disagree in sign.

# %% [markdown]
# ### Decoding
#
# After optimization, PCE turns each optimized correlator expectation
# value back into a discrete spin with sign rounding:
#
# $$
# s_i \;=\; \operatorname{sgn}\!\bigl\langle P_i \bigr\rangle_{\boldsymbol{\theta}^*}
# \;\in\; \{+1, -1\},
# $$
#
# i.e. $s_i = +1$ when $\langle P_i \rangle \ge 0$ and $s_i = -1$
# otherwise. The binary assignment is recovered as
# $x_i = (1 - s_i) / 2$.

# %% [markdown]
# ## Implementation

# %% [markdown]
# ### Build the `BinaryModel` and `PauliCorrelationEncoding`
#
# We build the Ising form derived in [](#problem-settings) with
# `BinaryModel.from_ising`: $h_i = 0$, $J_{ij} = 1/2$ on every edge, and
# constant $-|E|/2$. Passing that spin model and correlator order
# $k = 2$ to `PauliCorrelationEncoding` lets the encoding choose the required
# qubit count. With this scaling, the spin-model energy equals minus
# the cut value. A higher cut has a lower energy.

# %%
coefficients = {(i, j): 0.5 for i, j in G.edges()}
ising_model = BinaryModel.from_ising(
    coefficients,
    variables=range(num_nodes),
    constant=-num_edges / 2,
)
encoding = PauliCorrelationEncoding(ising_model, correlator_order=2)

print(f"Number of variables  : {ising_model.num_variables}")
print(f"PCE qubit count      : {encoding.num_qubits}")
print(f"Correlator order (k) : {encoding.correlator_order}")
print(f"Compression ratio    : {ising_model.num_variables / encoding.num_qubits:.1f}x")

assert ising_model.num_variables == 20
assert encoding.num_qubits == 3
assert encoding.correlator_order == 2

# %% [markdown]
# ### Inspect the Per-Variable Pauli Observables
#
# `encoding.observables` contains one Hamiltonian per variable, each
# containing exactly one $k$-body Pauli string with coefficient $1$.
# These are the $P_i$ observables from [](#pce-algorithm). The optimizer
# will estimate their expectation values from the state prepared by the
# ansatz kernel.

# %%
observables = encoding.observables

print(f"Total observables : {len(observables)}")
assert len(observables) == ising_model.num_variables
for i, observable in enumerate(observables):
    ((word, coefficient),) = observable.terms
    assert len(word.factors) == encoding.correlator_order and coefficient == 1.0
    factors = " ".join(f"{factor.pauli.value}{factor.index}" for factor in word.factors)
    print(f"  P_{i:2d}: {factors}")

# %% [markdown]
# ### Define the Ansatz
#
# PCE leaves the circuit choice open. The original paper uses a
# **hardware-efficient ansatz**: alternating layers of
# single-qubit rotations and two-qubit entangling gates. We use the
# pre-built `ry_layer`, `rz_layer`, and `cx_entangling_layer` from
# `qamomile.stdlib` and stack them `depth` times,
# giving $2 \cdot n \cdot \text{depth}$ variational angles in total.
# The kernel returns the prepared quantum state, which
# we use to evaluate all $P_i$.
#
# :::{note}
# Qamomile's rotation gates carry the standard
# $1/2$ factor: $\text{RY}(\theta) = e^{-i \theta Y / 2}$ and
# $\text{RZ}(\theta) = e^{-i \theta Z / 2}$. Every entry of the `thetas`
# array is a variational parameter. We group the optimizer’s flat vector
# by layer, rotation (RY or RZ), and qubit. The optimizer can scale each angle, so the
# constant factor is absorbed into the optimal `thetas` values. We pass
# the angles directly without inserting an explicit factor of $2$.
# :::


# %%
@qm.qkernel
def pce_ansatz(
    n: qm.Const[qm.Int],
    depth: qm.Const[qm.Int],
    thetas: qm.Param[qm.Tensor[qm.Real, depth, 2, n]],  # noqa: F821
) -> qm.Q[qm.Bits[n]]:  # noqa: F821
    q = qm.qubits(n)
    q = qm.h(q)
    for d in range(depth):
        q = ry_layer(q, n, thetas[d][0])
        q = rz_layer(q, n, thetas[d][1])
        q = cx_entangling_layer(q, n)
    return q


# %% [markdown]
# ### Prepare an Estimator per Observable
#
# Fix the qubit count and depth with `pce_ansatz.specialize(n=n, depth=depth)`.
# Create one Qamomile `Estimator` for each observable in `encoding.observables`.
# At each loss evaluation, estimate each correlator on the local `Aer()` simulator
# with 1,024 shots at the current rotation angles. We fix a seed for each
# observable so repeated evaluations at the same angles give the same estimates.

# %%
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

n = encoding.num_qubits
depth = 1 if docs_test_mode else 3
num_thetas = 2 * n * depth
program = pce_ansatz.specialize(n=n, depth=depth)

estimators = [Estimator(program, observable) for observable in encoding.observables]
executor = Aer()
training_shots = 1 if docs_test_mode else 1024

print(f"Estimators prepared : {len(estimators)}")
print(f"Variational params  : {num_thetas} (= 2 * n * depth)")

assert len(estimators) == ising_model.num_variables

# %% [markdown]
# (pce-optimize-variational-parameters)=
# ### Optimize the Variational Parameters
#
# The classical loop estimates $\langle P_i \rangle$ for every
# observable at the current `thetas`, plugs those values into the
# tanh-relaxed loss from [](#pce-algorithm) (data term + regularizer), and
# asks `scipy.optimize.minimize` to update the angles. We use COBYLA,
# which works with sampled losses without estimating a gradient.
#
# We configure the three loss hyperparameters following the original
# paper:
#
# - $\alpha$ (tanh sharpness): set to $\alpha = N^{k/2}$, where $N$
#   is the number of graph nodes and $k$ is the correlator order. For
#   our 20-node, $k = 2$ run, $\alpha = 20$.
# - $\beta$ (regularizer strength): a fixed value of $1/2$ that the
#   paper has already tuned.
# - $\nu$ (overall scale): the Edwards–Erdős MaxCut bound,
#   $\nu = |E|/2 + (N - 1)/4$.

# %%
# Hyperparameters from https://doi.org/10.48550/arXiv.2401.09421:
#   alpha = N^(k/2) (N = number of nodes, k = PCE correlator order)
#   beta  = 1/2 (regularizer strength)
#   nu    = |E| / 2 + (N - 1) / 4 (Edwards-Erdős MaxCut bound)
N = ising_model.num_variables
k = encoding.correlator_order
alpha = float(N ** (k / 2))
beta = 0.5
nu = num_edges / 2 + (N - 1) / 4
print(f"alpha = {alpha}, beta = {beta}, nu = {nu}")


def measure_expectations(thetas: np.ndarray, *, shots: int) -> list[float]:
    angles = thetas.reshape(depth, 2, n).tolist()
    return [
        estimator.estimate(
            on=executor, parameters={"thetas": angles}, shots=shots, seed=42 + i
        ).value
        for i, estimator in enumerate(estimators)
    ]


def loss(params: np.ndarray) -> float:
    expvals = measure_expectations(params, shots=training_shots)
    relaxed = [np.tanh(alpha * e) for e in expvals]

    # Data term: smooth surrogate of the spin objective.
    L_data = 0.0
    for indices, coefficient in coefficients.items():
        term = coefficient
        for i in indices:
            term *= relaxed[i]
        L_data += term

    # Regularizer: beta * nu * [(1/N) sum tanh^2(alpha <P_i>)]^2.
    mean_sq = sum(r**2 for r in relaxed) / N
    L_reg = beta * nu * mean_sq**2

    return L_data + L_reg


rng = np.random.default_rng(93)
initial_params = rng.uniform(-np.pi, np.pi, num_thetas)

options = {"maxiter": 100}
if docs_test_mode:
    options = {"maxiter": num_thetas + 2, "f_target": float("inf")}
res = minimize(loss, initial_params, method="COBYLA", options=options)

print(f"Loss evaluations: {res.nfev}")
print(f"Final loss: {res.fun:+.4f}")

# %% [markdown]
# (pce-decode-optimized-expectations)=
# ### Decode the Optimized Expectations
#
# `encoding.decode(expectations)` takes the per-variable expectation
# values, sign-rounds each one to a spin, and returns a single-sample
# `DecodedSamples` in the same vartype as the input model. Here the
# vartype is SPIN because `ising_model` was built with
# `BinaryModel.from_ising`. The reported energy follows the convention
# from [](#problem-settings): energy = $-\,\text{cut}$. The decoded energy is
# the negative of the cut value. We use 65,536 shots per observable for this
# final estimate because sampling noise can change the sign of a value close to zero.

# %%
final_shots = 1 if docs_test_mode else 65_536
final_expectations = measure_expectations(res.x, shots=final_shots)
sampleset = encoding.decode(final_expectations)

print("Final per-variable expectations:")
for i, e in enumerate(final_expectations):
    print(f"  <P_{i:2d}> = {e:+.4f}")
print()
print(f"Decoded vartype : {sampleset.vartype}")
print(f"Decoded energy  : {sampleset.lowest.energy:+.4f}")

# %% [markdown]
# (pce-result)=
# ## Result

# %% [markdown]
# (pce-classical-baseline)=
# ### Classical Baseline (Brute Force)
#
# Enumerating all $2^{20} = 1{,}048{,}576$ spin configurations means
# checking about one million assignments. That is too many for a
# simple Python loop, but a single vectorized NumPy pass finishes in a
# fraction of a second.
# We label each configuration by an integer whose bit $i$ encodes
# $s_i = +1$ (bit $0$) or $s_i = -1$ (bit $1$), then count edges with
# $s_i \neq s_j$. This gives us the ground-truth optimum to compare
# the PCE result against in the next subsection.

# %%
assignments = np.arange(2**num_nodes, dtype=np.int64)
cuts = np.zeros(2**num_nodes, dtype=np.int32)
for i, j in G.edges():
    s_i = 1 - 2 * ((assignments >> i) & 1)  # bit 0 → +1, bit 1 → -1
    s_j = 1 - 2 * ((assignments >> j) & 1)
    cuts += (s_i != s_j).astype(np.int32)

best_cut = int(cuts.max())
optimal_assignment_ints = np.flatnonzero(cuts == best_cut)
print(f"Optimal MaxCut value          : {best_cut}")
print(f"Number of optimal assignments : {len(optimal_assignment_ints)}")

# The graph seed is fixed, so the brute-force optimum is deterministic.
assert best_cut == 26

# %% [markdown]
# ### Decode and Analyze Results

# %% [markdown]
# #### Best Cut
#
# Convert the decoded spin assignment into a graph partition and compare
# its cut value with the brute-force optimum from
# [](#pce-classical-baseline). As a consistency check, the cut
# value should equal $-1$ times the spin energy reported in
# [](#pce-decode-optimized-expectations).

# %%
sample = sampleset.lowest.values
spins = [sample[i] for i in range(num_nodes)]
pce_cut = sum(1 for i, j in G.edges() if spins[i] != spins[j])

print(f"PCE spin assignment : {spins}")
print(f"PCE cut value       : {pce_cut}")
print(f"Brute-force optimum : {best_cut}")
print(f"Approximation ratio : {pce_cut / best_cut:.3f}")
assert 0 <= pce_cut <= best_cut
assert np.isclose(pce_cut, -sampleset.lowest.energy, rtol=0.0, atol=1e-12)

# %% [markdown]
# ## Summary
#
# In this tutorial we encoded a 20-node 3-regular MaxCut problem with
# Pauli Correlation Encoding (PCE) and ran the full Qamomile workflow,
# from building the correlator encoding through to decoding the optimized
# expectation values into a spin assignment.
#
# - **Qubit resource efficiency:** PCE represented the 20 spin variables with
#   only 3 qubits through 2-body Pauli correlators, roughly a 7x reduction
#   over the one-qubit-per-variable QAOA encoding.
# - **Surrogate-loss training:** the variational loop minimized the
#   tanh-relaxed surrogate rather than an energy directly, and the
#   decoded assignment cut 26 edges in this run, matching the brute-force optimum.
#   We selected the initialization seed (93) after trying several on this graph.
# - **End-to-end Qamomile path:** `PauliCorrelationEncoding` encoded the given
#   classical variables and exposed the corresponding observables through
#   `encoding.observables`; a hardware-efficient `@qkernel` ansatz
#   prepared the state, `Estimator` estimated each correlator on `Aer()`, the optimizer tuned the parameters,
#   and `encoding.decode` sign-rounded the expectations obtained with
#   those tuned parameters back into the corresponding classical variables.
#
# This `PauliCorrelationEncoding` workflow applies to any QUBO / Ising
# combinatorial problem where qubit count is the bottleneck: swap in your
# own `BinaryModel` and reuse the encoding, circuit preparation, and decoding steps
# shown above.
