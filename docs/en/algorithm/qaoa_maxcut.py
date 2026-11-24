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
# # Quantum Approximate Optimization Algorithm (QAOA) for MaxCut
#
# In this tutorial, we solve a small **MaxCut** problem with QAOA: split the
# vertices of a graph into two groups so that as many edges as possible run
# between the two groups. We build the QAOA circuit ourselves from Hadamard and
# rotation gates, so that every part of the algorithm appears in the code.
#
# The tutorial follows these steps:
#
# 1. Write MaxCut for a small graph as an Ising model on spin variables, and
#    find the exact answer by trying every assignment.
# 2. Build the QAOA circuit as `@qkernel` functions, one part at a time.
# 3. Run the circuit repeatedly while a classical optimizer tunes its angles.
# 4. Measure the circuit once more at the angles the optimizer returned. Each
#    measurement gives one cut. Take the largest measured cut as the answer and
#    compare its size with the exact answer. Then count how often each cut was
#    measured.
# 5. Build the same state in one call with Qamomile's built-in `qaoa_state`.
#
# **Prerequisites**: qubits, quantum gates and measurement. If you are new to
# these, start with
# [Your First Quantum Kernel](../tutorial/01_your_first_quantum_kernel.ipynb).
# No prior knowledge of QAOA is needed. For a problem with a constraint, solved
# with Qamomile's higher-level `QAOAExperiment`, see
# [QAOA for Graph Partitioning](qaoa_graph_partition).

# %% [markdown]
# ## Installation

# %%
# Install the latest Qamomile and NetworkX through pip!
# # !pip install qamomile networkx

# %%
# ruff: noqa: F821 -- dimensions are Qamomile dependent shape binders.

from __future__ import annotations

import itertools
import os
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import minimize

import qamomile as qm
from qamomile.optimization import BinaryModel
from qamomile.stdlib import qaoa_state

# %% [markdown]
# ## Problem Settings
#
# (qaoa-maxcut-problem-definition)=
# ### Problem Definition
#
# Let $G = (V, E)$ be an undirected graph with vertex set $V$ and edge set $E$,
# and let $n = |V|$. An edge is a pair of vertices without an order: $(i, j)$
# and $(j, i)$ are the same edge, and a sum over $E$ counts it once. The graph
# is fixed. The only choice is which of two groups each vertex goes to. Such a
# split of the vertices into two groups is called a *cut*. An edge *crosses*
# the cut when its two ends lie in different groups. Picture a line drawn
# between the two groups: the crossing edges are exactly the edges it cuts
# through. The number of crossing edges is the *size* of the cut, and the
# **MaxCut** problem asks for a cut of the largest size.
#
# We write the choice with spin variables. Each vertex $i$ gets a spin
# $s_i \in \{+1, -1\}$ that names its group. An edge $(i, j)$ crosses the cut
# exactly when $s_i \ne s_j$. Then $s_i s_j = -1$, and otherwise
# $s_i s_j = +1$, so $(1 - s_i s_j)/2$ is 1 for a crossing edge and 0 for any
# other edge. The size of the cut is therefore
#
# $$
# C(\boldsymbol{s}) = \sum_{(i,j) \in E} \frac{1 - s_i s_j}{2},
# $$
#
# and the goal is to find a spin assignment $\boldsymbol{s}$ that maximizes
# $C(\boldsymbol{s})$. The largest value is the MaxCut value of the graph.
#
# MaxCut can also be written with binary variables $x_i \in \{0, 1\}$, through
# $s_i = 1 - 2x_i$, as a Quadratic Unconstrained Binary Optimization (QUBO)
# problem. An edge then contributes $x_i + x_j - 2x_i x_j$ instead of
# $(1 - s_i s_j)/2$. We use spin variables throughout this tutorial: each edge
# then gives a single product $s_i s_j$, which is the form of the Ising model
# used below.

# %% [markdown]
# ### Problem Instance
#
# We use a fixed graph with 5 vertices and 6 edges. Vertices 0, 1 and 2 form a
# triangle, and so do vertices 1, 2 and 3. Vertex 4 is attached to vertex 3
# only. With 5 vertices, there are only $2^5 = 32$ spin assignments, so below
# we can try every one of them and compare QAOA with the exact answer.

# %%
G = nx.Graph()
G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)])
num_nodes = G.number_of_nodes()
# The number of vertices is taken from the edges, so the graph needs an edge.
assert G.number_of_edges() > 0, "The graph must have at least one edge."
# The code uses the vertex labels as positions, so they must be 0, 1, ..., n-1.
assert sorted(G.nodes) == list(range(num_nodes)), "Label the vertices 0, 1, ..., n-1."
assert num_nodes == 5
assert G.number_of_edges() == 6

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
plt.title(f"Graph: {num_nodes} vertices, {G.number_of_edges()} edges")
plt.show()

# %% [markdown]
# To try another graph, change the edge list. Label the vertices
# $0, 1, \dots, n-1$ without gaps, because the code uses these labels as
# positions in the spin assignments. Counts in the assertions in this tutorial,
# such as the 6 edges here and the cut size 5 in the exact solution, hold for
# this graph only, so update them too. The exact solution tries all $2^n$ spin
# assignments, so it slows down quickly as $n$ grows. The
# [Result](#qaoa-maxcut-result) section checks the run against the exact
# solution with assertions: the largest measured cut must be a largest cut of
# the graph, and so must the cut measured most often. QAOA guarantees neither.
# If one of them fails for your graph, the notebook stops there with an
# `AssertionError`. This reports a weaker QAOA result on that graph, not a
# mistake in the code.

# %% [markdown]
# ### Ising Formulation
#
# Since $\sum_{(i,j) \in E} 1 = |E|$, the size of the cut can be rewritten as
#
# $$
# C(\boldsymbol{s}) = \frac{|E|}{2} - \frac{1}{2} H_C(\boldsymbol{s}),
# \qquad
# H_C(\boldsymbol{s}) = \sum_{(i,j) \in E} s_i s_j.
# $$
#
# The first term is fixed by the graph, so maximizing $C$ is the same as
# minimizing $H_C$, and a cut of size $C$ has $H_C = |E| - 2C$. $H_C$ is the
# energy of an Ising model, a model of interacting spins. In the general Ising
# form
#
# $$
# H = \sum_i h_i s_i + \sum_{i < j} J_{ij} s_i s_j,
# $$
#
# unweighted MaxCut has no linear terms ($h_i = 0$ for every vertex), the
# coupling $J_{ij} = 1$ for every edge $(i, j) \in E$, and $J_{ij} = 0$ for
# pairs that are not edges. A positive coupling lowers the energy when the two
# spins differ, which is exactly when the edge crosses the cut. Such a model is
# called antiferromagnetic.
#
# In the [Algorithm](#qaoa-maxcut-algorithm) section, $H_C$ becomes the cost
# Hamiltonian of QAOA. Below, `BinaryModel.from_ising` builds the model from
# these coefficients directly, without converting to binary variables. We then
# read the coefficients back from the model, in the model's own order, and use
# them to build the circuit. For a problem written with binary variables,
# `BinaryModel.from_hubo` plays the same role.

# %%
ising_quad: dict[tuple[int, int], float] = {
    tuple(sorted((i, j))): 1.0 for i, j in G.edges()
}
ising_linear: dict[int, float] = {}

spin_model = BinaryModel.from_ising(
    {**ising_quad, **{(i,): value for i, value in ising_linear.items()}},
    variables=range(num_nodes),
)
# Read the coefficients back from the model, in the model's own order.
ising_quad = {
    indices: coefficient
    for indices, coefficient in spin_model.coefficients.items()
    if len(indices) == 2
}
ising_linear = {
    indices[0]: coefficient
    for indices, coefficient in spin_model.coefficients.items()
    if len(indices) == 1
}

print(f"Variable type:          {spin_model.vartype}")
print(f"Linear terms (h_i):     {ising_linear}")
print(f"Quadratic terms (J_ij): {ising_quad}")
print(f"Constant:               {spin_model.constant}")
# Spin-domain model with no linear field, one J_ij per graph edge, no offset.
assert spin_model.vartype.name == "SPIN"
assert ising_linear == {}
assert len(ising_quad) == G.number_of_edges()
assert spin_model.constant == 0.0
# The model is unchanged by saving it to bytes and loading it back.
assert BinaryModel.from_bytes(spin_model.to_bytes()) == spin_model

# %% [markdown]
# ### Exact Solution (Brute Force)
#
# Before running QAOA, we find the exact answer by trying all $2^n = 32$ spin
# assignments. It is the reference value for the QAOA result later.

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
print(f"Number of optimal assignments: {len(optimal_partitions)}")
for part in optimal_partitions:
    print(f"  {part}")

# The Ising energy of an optimal assignment is |E| - 2C.
optimal_energy = spin_model.energy(optimal_partitions[0])
print(f"Ising energy of the optimal assignments: {optimal_energy}")

# For this graph, the largest cut size is 5. Two spin assignments reach it,
# and one is the other with every spin flipped.
assert best_cut == 5
assert len(optimal_partitions) == 2
for part in optimal_partitions:
    assert tuple(-s for s in part) in optimal_partitions
assert optimal_energy == -4
assert all(
    spin_model.energy(part) == G.number_of_edges() - 2 * best_cut
    for part in optimal_partitions
)

# %% [markdown]
# The largest cut size is 5, one less than the number of edges, and the
# corresponding Ising energy is $H_C = 6 - 2 \cdot 5 = -4$. No cut can make all
# 6 edges cross. In a triangle, at least two of the three vertices share a
# group, so at least one edge of each triangle does not cross. The two
# triangles share the edge $(1, 2)$, so putting vertices 1 and 2 in one group,
# vertices 0 and 3 in the other, and vertex 4 opposite vertex 3 lets every
# other edge cross. The two optimal assignments describe this same cut:
# flipping every spin only swaps the names of the two groups, so both count as
# the exact answer.

# %% [markdown]
# (qaoa-maxcut-algorithm)=
# ## Algorithm
#
# The Quantum Approximate Optimization Algorithm was introduced by
# [Farhi, Goldstone and Gutmann, arXiv:1411.4028](https://doi.org/10.48550/arXiv.1411.4028).
# It prepares a quantum state with a circuit of $p$ layers, built from $H_C$
# and $2p$ angles, two per layer. The state is measured, and a classical
# optimizer tunes the angles so that the measurements are likely to give a
# large cut. This section explains the state, how it becomes gates, how the
# angles are tuned, and how the answer is read out.
#
# ### The QAOA State
#
# QAOA looks for the lowest-energy state of $H_C$ with a circuit on $n$ qubits,
# one qubit per vertex. We first turn $H_C$ into an operator by replacing each
# spin $s_i$ with the Pauli operator $Z_i$ on qubit $i$:
#
# $$
# H_C = \sum_{(i,j) \in E} Z_i Z_j .
# $$
#
# Since $Z|0\rangle = |0\rangle$ and $Z|1\rangle = -|1\rangle$, every bitstring
# $z$ is an eigenstate of $H_C$, and its eigenvalue is the energy of the spin
# assignment with $s_i = 1 - 2z_i$. We write this energy $E(z)$, so that
# $H_C |z\rangle = E(z)\, |z\rangle$. Bit 0 stands for spin $+1$ and bit 1 for
# spin $-1$, the same rule as $s_i = 1 - 2x_i$ in the
# [Problem Definition](#qaoa-maxcut-problem-definition).
#
# The qubits can hold all bitstrings at once, in a superposition
#
# $$
# |\psi\rangle = \sum_{z} a_z\, |z\rangle ,
# $$
#
# and measuring them returns the bitstring $z$ with probability $|a_z|^2$. The
# aim is to prepare a state in which low-energy bitstrings, that is, large
# cuts, have large probabilities, so that a modest number of measurements is
# likely to return a good cut.
#
# **Initial state.** The circuit starts from the uniform superposition
# $|+\rangle^{\otimes n}$, which gives every bitstring the same probability. A
# Hadamard gate on each qubit prepares it.
#
# **Cost unitary.** For an angle $\gamma$, the cost unitary is
# $U_C(\gamma) = e^{-i\gamma H_C}$. It gives each bitstring a phase set by its
# energy:
#
# $$
# U_C(\gamma) |z\rangle = e^{-i\gamma E(z)}\, |z\rangle .
# $$
#
# A phase alone changes no measurement probability. It only records the energy
# of each bitstring in the state.
#
# **Mixer unitary.** For an angle $\beta$, the mixer unitary is
#
# $$
# U_M(\beta) = e^{-i\beta \sum_i X_i}
# = \prod_i \bigl(\cos\beta\; I - i \sin\beta\; X_i\bigr) ,
# $$
#
# where $I$ is the identity and the Pauli operator $X_i$ flips bit $i$. Write
# $y^{(i)}$ for a bitstring $y$ with bit $i$ flipped. The factor for qubit $i$
# turns $|y\rangle$ into a combination of $|y\rangle$ and $|y^{(i)}\rangle$:
#
# $$
# \bigl(\cos\beta\; I - i \sin\beta\; X_i\bigr)\, |y\rangle
# = \cos\beta\, |y\rangle - i \sin\beta\, |y^{(i)}\rangle .
# $$
#
# Each factor therefore passes amplitude between two bitstrings that differ in
# one bit. Unless $\sin\beta$ or $\cos\beta$ is zero, the product of the $n$
# factors connects every pair of bitstrings. Amplitudes that arrive at the same
# bitstring with similar phases add up, and those with opposite phases cancel.
# This interference is what changes the probabilities.
#
# **Layers.** A cost unitary followed by a mixer is one *layer*. With $p$
# layers and the angles $\boldsymbol{\gamma} = (\gamma_1, \dots, \gamma_p)$ and
# $\boldsymbol{\beta} = (\beta_1, \dots, \beta_p)$, the QAOA state is
#
# $$
# |\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle
# = U_M(\beta_p)\, U_C(\gamma_p) \cdots U_M(\beta_1)\, U_C(\gamma_1)\,
# |+\rangle^{\otimes n} ,
# $$
#
# read from right to left. The graph fixes the structure of the circuit through
# $H_C$, and the $2p$ angles are the only free parameters.
#
# With suitable angles, this interference can increase the probability of
# low-energy bitstrings. The [appendix](#qaoa-maxcut-small-angle-analysis)
# derives this effect for one layer with small angles and explains the role
# of their signs.
#
# (qaoa-maxcut-unitaries-to-gates)=
# ### From Unitaries to Gates
#
# For a general Ising model, the cost Hamiltonian is
# $H_C = \sum_i h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j$. All its terms commute,
# because they are all made of $Z$ operators. The cost unitary therefore splits
# exactly into one factor per term,
#
# $$
# U_C(\gamma) = \prod_i e^{-i\gamma h_i Z_i}
# \prod_{i<j} e^{-i\gamma J_{ij} Z_i Z_j} ,
# $$
#
# and the mixer splits into one factor per qubit. Qamomile's rotation gates are
# defined with a factor $1/2$ in the exponent,
#
# $$
# R_Z(\theta) = e^{-i\theta Z/2}, \qquad
# R_{ZZ}(\theta) = e^{-i\theta Z \otimes Z/2}, \qquad
# R_X(\theta) = e^{-i\theta X/2},
# $$
#
# so each factor is one gate with twice the angle:
#
# $$
# e^{-i\gamma J_{ij} Z_i Z_j} = R_{ZZ}(2\gamma J_{ij}), \qquad
# e^{-i\gamma h_i Z_i} = R_Z(2\gamma h_i), \qquad
# e^{-i\beta X_i} = R_X(2\beta).
# $$
#
# On this graph, $J_{ij} = 1$ on every edge and $h_i = 0$, so one layer has 6
# $R_{ZZ}$ gates and 5 $R_X$ gates. Its 5 $R_Z$ gates have angle 0 and do
# nothing here. We keep them so that the same code also works for Ising models
# with linear terms. Weighted MaxCut needs no change either, because its
# weights enter as $J_{ij}$.
#
# A measurement returns bit 0 for spin $+1$ and bit 1 for spin $-1$, by the
# rule $s_i = 1 - 2z_i$ above. `BinaryModel.decode` applies this rule when it
# turns measured bitstrings back into spin assignments.
#
# ### Optimizing the Angles
#
# We choose the angles to minimize the expected energy of the QAOA state:
#
# $$
# F_p(\boldsymbol{\gamma}, \boldsymbol{\beta})
# = \langle \boldsymbol{\gamma}, \boldsymbol{\beta} |\, H_C \,|
# \boldsymbol{\gamma}, \boldsymbol{\beta} \rangle
# = \sum_z \bigl|\langle z | \boldsymbol{\gamma}, \boldsymbol{\beta}
# \rangle\bigr|^2\, E(z) .
# $$
#
# Because $C = |E|/2 - H_C/2$, this maximizes the expected cut size
# $|E|/2 - F_p/2$. We estimate $F_p$ from $N$ measurements at the same angles.
# Each measurement returns a bitstring $z^{(k)}$, whose energy we compute:
#
# $$
# F_p(\boldsymbol{\gamma}, \boldsymbol{\beta})
# \approx \frac{1}{N} \sum_{k=1}^{N} E\bigl(z^{(k)}\bigr) .
# $$
#
# A classical optimizer selects the angles, receives this average from the
# quantum circuit, and selects the next angles. Each evaluation uses a fresh
# batch of $N$ shots, so the total shot count is $N$ times the number of
# evaluations. Increasing $N$ reduces the statistical uncertainty in each
# estimate.
#
# We use SciPy's Constrained Optimization BY Linear Approximation (COBYLA)
# method, which needs no gradients. It searches locally using sampled
# energies, so it does not guarantee a global minimum. The result depends on
# the number of layers, starting angles, optimizer and shot count. The
# [Result](#qaoa-maxcut-result) section shows the optimization loop.
#
# **Starting angles.** Random starting angles can lead to poor local minima
# in MaxCut ([Sack and Serbyn, Quantum 5, 491 (2021)](https://doi.org/10.22331/q-2021-07-01-491)).
# We use angles motivated by quantum annealing, which slowly changes the
# Hamiltonian
#
# $$
# H(s) = -(1 - s) \sum_i X_i + s\, H_C
# $$
#
# from $s = 0$ to $s = 1$. A ground state is a state of lowest energy.
# Initially it is the uniform superposition. The final ground states encode
# maximum cuts. The adiabatic theorem motivates following the ground state
# when the change is
# slow enough and the relevant energy gap remains open
# ([Albash and Lidar, Reviews of Modern Physics 90, 015002 (2018)](https://doi.org/10.1103/RevModPhys.90.015002)).
# Here the Hamiltonian and initial state are unchanged by flipping all spins,
# so the relevant gap is to other states with the same symmetry. We do not
# verify this gap condition; annealing only guides our choice of starting
# angles.
#
# For a total time $T$, divide the evolution into $p$ steps of duration
# $\Delta t = T/p$. Approximate $s = t/T$ within each step by its midpoint:
#
# $$
# s_k = \frac{k - 1/2}{p} ,
# \qquad
# k = 1, \dots, p .
# $$
#
# Splitting the evolution into a cost step followed by a mixer step gives
#
# $$
# e^{-i \Delta t\, H(s_k)}
# \approx e^{\,i \Delta t\, (1 - s_k) \sum_i X_i}\; e^{-i \Delta t\, s_k H_C}
# = U_M(\beta_k)\, U_C(\gamma_k) ,
# $$
#
# so the starting angles are
#
# $$
# \gamma_k = \Delta t\, s_k ,
# \qquad
# \beta_k = -\Delta t\, (1 - s_k) .
# $$
#
# The minus sign in $\beta_k$ comes from the minus sign before $\sum_i X_i$
# in $H(s)$. These angles vary linearly across layers, giving a *linear ramp*.
# With only $p = 3$ steps, neither slow evolution nor an accurate splitting is
# established. The ramp supplies starting angles, not a guarantee of a
# maximum cut.
#
# We use $\Delta t = 0.75$, the time step that Sack and Serbyn found for
# MaxCut on 3-regular graphs, where every vertex has three neighbors. Their
# cost Hamiltonian is the same, but their mixer uses $-\sum_i X_i$, so their
# $\beta$ corresponds to $-\beta$ here. They use $s_k = k/p$; we use the
# midpoint instead. Our graph is not 3-regular, and this step size has not
# been tuned for it. With $p = 3$, the starting angles are
# $\boldsymbol{\gamma} = (0.125, 0.375, 0.625)$ and
# $\boldsymbol{\beta} = (-0.625, -0.375, -0.125)$.
#
# ### Reading Out the Cut
#
# COBYLA returns the angles with the smallest sampled energy estimate among
# those it evaluated. We run the circuit again at these angles to obtain
# cuts. Samples collected during optimization are used only to estimate
# $F_p$; the cut analysis uses this final batch.
#
# Every measured bitstring $z$ gives a valid cut through $s_i = 1 - 2z_i$.
# Flipping all bits swaps the two groups, so we count a bitstring and its
# flipped copy together. Counting how often each cut occurs gives its
# measured frequency.
#
# For each measured cut, we count the crossing edges and take the largest
# cut as the answer. This follows the sampling-and-evaluation procedure in
# [Farhi, Goldstone and Gutmann, arXiv:1411.4028](https://doi.org/10.48550/arXiv.1411.4028),
# Sec. I. The largest measured cut is at least as large as the sample
# average, but it need not be the graph's maximum cut. The
# [Result](#qaoa-maxcut-result) section compares it with the exact solution
# and checks whether the maximum cut is also the most frequently measured.

# %% [markdown]
# ## Implementation with Qamomile
#
# We now write the circuit of the [Algorithm](#qaoa-maxcut-algorithm) section
# as `@qkernel` functions, one step at a time: the uniform superposition, the
# cost layer, the mixer layer, and the full circuit with $p$ layers and a final
# measurement. Two kinds of input appear, as explained in
# [Parameterized Quantum Kernels](../tutorial/02_parameterized_kernels.ipynb).
# Inputs marked `qm.Const`, such as the number of qubits and the edges of the
# graph, are fixed when the kernel is specialized. Inputs marked `qm.Param`,
# here the angles $\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$, stay free and
# are given at each run.

# %% [markdown]
# ### Step 1: Prepare the Uniform Superposition State
#
# `qm.qubits(size)` creates `size` qubits in $|0\rangle$, and a Hadamard gate
# on each qubit turns them into the uniform superposition
# $|{+}\rangle^{\otimes n}$, the initial state of the
# [Algorithm](#qaoa-maxcut-algorithm) section. A gate consumes the quantum
# state it receives and returns the updated state, so each gate call is written
# as `q = qm.h(q, target=i)` (see
# [Your First Quantum Kernel](../tutorial/01_your_first_quantum_kernel.ipynb)).
# In the annotation `qm.Q[qm.Bits[size]]`, `size` is the kernel's own input, so
# the annotation says that the kernel returns `size` qubits. The import cell at
# the top of the page imports `annotations` from `__future__`, so that Python
# leaves such annotations to Qamomile and does not evaluate them itself.


# %%
@qm.qkernel
def superposition(size: qm.Const[qm.Int]) -> qm.Q[qm.Bits[size]]:
    q = qm.qubits(size)
    for i in range(size):
        q = qm.h(q, target=i)
    return q


# %% [markdown]
# (qaoa-maxcut-step-2)=
# ### Step 2: Cost Layer
#
# The cost layer applies $U_C(\gamma)$ as one $R_{ZZ}(2\gamma J_{ij})$ gate per
# edge and one $R_Z(2\gamma h_i)$ gate per vertex, as derived in
# [From Unitaries to Gates](#qaoa-maxcut-unitaries-to-gates) above. The kernel
# reads the Ising model from three inputs. `edges` is an `edge_count` × 2 array
# whose row $k$ holds the two vertices of edge $k$, `quad[k]` holds the
# coupling $J_{ij}$ of that edge, and `linear[i]` holds $h_i$. On this graph,
# `linear` is all zeros, and From Unitaries to Gates explains why the $R_Z$
# gates are kept. The argument `gamma` is the angle of one layer, which
# [Step 4](#qaoa-maxcut-step-4) takes from `gammas`. It carries neither the
# `qm.Const` nor the `qm.Param` marker, because the kernel accepts whichever
# kind of value its caller passes.


# %%
@qm.qkernel
def cost_layer(
    q: qm.Q[qm.Bits[size]],
    size: qm.Const[qm.Int],
    edge_count: qm.Const[qm.Int],
    edges: qm.Const[qm.Matrix[qm.Int, edge_count, 2]],
    quad: qm.Const[qm.Vector[qm.Real, edge_count]],
    linear: qm.Const[qm.Vector[qm.Real, size]],
    gamma: qm.Real,
) -> qm.Q[qm.Bits[size]]:
    for edge in range(edge_count):
        q = qm.rzz(
            q, 2.0 * quad[edge] * gamma, left=edges[edge][0], right=edges[edge][1]
        )
    for i in range(size):
        q = qm.rz(q, 2.0 * linear[i] * gamma, target=i)
    return q


# %% [markdown]
# ### Step 3: Mixer Layer
#
# The mixer layer applies $U_M(\beta)$ as one $R_X(2\beta)$ gate per qubit, as
# derived in [From Unitaries to Gates](#qaoa-maxcut-unitaries-to-gates) above.
# Like `gamma` in [Step 2](#qaoa-maxcut-step-2), the argument `beta` is the
# angle of one layer, which [Step 4](#qaoa-maxcut-step-4) takes from `betas`.


# %%
@qm.qkernel
def mixer_layer(
    q: qm.Q[qm.Bits[size]],
    size: qm.Const[qm.Int],
    beta: qm.Real,
) -> qm.Q[qm.Bits[size]]:
    for i in range(size):
        q = qm.rx(q, 2.0 * beta, target=i)
    return q


# %% [markdown]
# (qaoa-maxcut-step-4)=
# ### Step 4: Build the QAOA Ansatz
#
# `qaoa_ansatz` puts the three steps together into the QAOA state of the
# [Algorithm](#qaoa-maxcut-algorithm) section, followed by a measurement. An
# ansatz is a circuit with free parameters, here the angles. The kernel
# prepares the uniform superposition and then applies $p$ layers, `layers` in
# the code. The loop runs over `layer` $= 0, \dots, p-1$, so `gammas[layer]`
# and `betas[layer]` are $\gamma_k$ and $\beta_k$ for $k =$ `layer` $+ 1$.
# Finally, it measures every qubit and returns the bitstring.
#
# `qm.specialize` then fixes the `qm.Const` inputs: the number of layers
# $p = 3$, the number of qubits, and the edges and coefficients of the Ising
# model, read from the model in its own order. The angles remain free.


# %%
@qm.qkernel
def qaoa_ansatz(
    layers: qm.Const[qm.Int],
    size: qm.Const[qm.Int],
    edge_count: qm.Const[qm.Int],
    edges: qm.Const[qm.Matrix[qm.Int, edge_count, 2]],
    quad: qm.Const[qm.Vector[qm.Real, edge_count]],
    linear: qm.Const[qm.Vector[qm.Real, size]],
    gammas: qm.Param[qm.Vector[qm.Real, layers]],
    betas: qm.Param[qm.Vector[qm.Real, layers]],
) -> qm.Shot[qm.Bits[size]]:
    q = superposition(size)
    for layer in range(layers):
        q = cost_layer(q, size, edge_count, edges, quad, linear, gammas[layer])
        q = mixer_layer(q, size, betas[layer])
    return qm.measure(q)


p = 3  # number of QAOA layers
edge_indices = tuple(ising_quad)
quad_values = tuple(ising_quad[edge] for edge in edge_indices)
linear_values = tuple(ising_linear.get(i, 0.0) for i in range(num_nodes))
program = qm.specialize(
    qaoa_ansatz,
    layers=p,
    size=num_nodes,
    edge_count=len(edge_indices),
    edges=edge_indices,
    quad=quad_values,
    linear=linear_values,
)


# %% [markdown]
# ### Compile the Circuit
#
# `qm.compile` converts the specialized program into a circuit for the engine
# that will run it. Here we use Qamomile's built-in simulator, `qm.Reference()`.
# The angles `gammas` and `betas` are still free in the compiled circuit, so
# the optimizer can run it with new angles each time without another call to
# `qm.compile`. We pass the same random seed to each `qm.run` call to make
# sampling repeatable.

# %%
SEED = 42
executor = qm.Reference()
executable = qm.compile(program, to=executor)
# Kept to check later that the same compiled object is still in use.
artifact_identity = id(executable)
# The free parameters of the compiled circuit are the two vectors of angles.
assert tuple(slot.name for slot in executable.parameter_abi.slots) == (
    "gammas",
    "betas",
)

# %% [markdown]
# The next cell counts the gates of the specialized program and checks them
# against the [Algorithm](#qaoa-maxcut-algorithm) section: 5 Hadamard gates
# and, for each of the 3 layers, 6 $R_{ZZ}$, 5 $R_Z$ and 5 $R_X$ gates.

# %%
study = qm.resources.logical(program)
expected_counts = {
    "quantum.h": num_nodes,
    "quantum.rzz": p * len(ising_quad),
    "quantum.rz": p * num_nodes,
    "quantum.rx": p * num_nodes,
}
# numeric_enclosure() returns the lower and upper bounds of a count. Here the
# two are equal, so the count is exact.
for name, expected in expected_counts.items():
    count = study.metric(qm.resources.logical_operation_count(name)).numeric_enclosure()
    print(f"{name}: {count[0]}")
    assert count == (expected, expected)
# The resource study is unchanged by saving it to bytes and loading it back.
assert qm.resources.ResourceStudy.from_bytes(study.to_bytes()) == study

# %% [markdown]
# (qaoa-maxcut-result)=
# ## Result
#
# We now run the optimization loop of the [Algorithm](#qaoa-maxcut-algorithm)
# section on the compiled circuit, and then read out the cuts.
#
# ### Optimize the Angles
#
# `cost_fn` is the function that COBYLA evaluates. Each call runs the compiled
# circuit with `sample_shots` shots at the given angles. `spin_model.decode`
# then converts every measured bitstring into a spin assignment by
# $s_i = 1 - 2z_i$ and computes its energy, and `cost_fn` returns the mean of
# these energies over the shots, which is the estimate of $F_p$. It also
# appends the value to `cost_history`, so that the list holds one entry per
# evaluation. `scipy.optimize.minimize` receives `cost_fn` and the starting
# angles, and calls `cost_fn` each time COBYLA needs a value.
#
# The starting angles are the linear ramp of the
# [Algorithm](#qaoa-maxcut-algorithm) section with $\Delta t = 0.75$. The array
# `initial_params` holds the $p$ values of $\gamma$ first and then the $p$
# values of $\beta$, the order in which `cost_fn` splits it.
#
# After the run, the cell prints the number of evaluations and the number of
# shots they used, the reason why COBYLA stopped, the smallest estimate of
# $F_p$, and the angles at which it was obtained. The second cell plots the
# estimate at every evaluation.

# %%
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
sample_shots = 256 if docs_test_mode else 2048
maxiter = 20 if docs_test_mode else 500
cost_history: list[float] = []


def cost_fn(params):
    gammas = tuple(float(v) for v in params[:p])
    betas = tuple(float(v) for v in params[p:])
    result = qm.run(
        executable,
        on=executor,
        shots=sample_shots,
        seed=SEED,
        parameters={"gammas": gammas, "betas": betas},
    ).result
    decoded = spin_model.decode(result)
    energy = decoded.mean_energy
    cost_history.append(energy)
    return energy


# Linear ramp: gamma_k = delta_t * s_k and beta_k = -delta_t * (1 - s_k),
# with s_k = (k - 1/2) / p for k = 1, ..., p.
delta_t = 0.75
ramp = (np.arange(1, p + 1) - 0.5) / p
initial_params = np.concatenate([delta_t * ramp, -delta_t * (1 - ramp)])
assert initial_params.shape == (2 * p,)

res = minimize(cost_fn, initial_params, method="COBYLA", options={"maxiter": maxiter})

print(f"Evaluations: {res.nfev}, shots used: {res.nfev * sample_shots}")
print(f"Why COBYLA stopped: {res.message}")
print(f"Smallest estimate of F_p: {res.fun:.4f}")
print(f"Returned angles: {[round(float(v), 4) for v in res.x]}")
assert len(cost_history) == res.nfev
assert len(res.x) == 2 * p
# COBYLA returns the angles with the smallest estimate among all evaluations.
assert res.fun == min(cost_history)
if docs_test_mode:
    assert res.nfev == maxiter

# %%
plt.figure(figsize=(8, 4))
plt.plot(cost_history, color="#2696EB")
plt.xlabel("Evaluation")
plt.ylabel("Estimate of $F_p$ (mean energy)")
plt.title("QAOA Optimization Progress")
plt.show()

# %% [markdown]
# Each point in the plot shows the mean energy estimated at one objective
# evaluation. As reported in the output, COBYLA stopped when its trust-region
# radius, which limits the search step, reached its lower bound.

# %% [markdown]
# ### Sample and Evaluate
#
# We run the compiled circuit once more at the returned angles and decode this
# final batch. Because we pass the same seed as during optimization, the batch contains
# exactly the bitstrings of the evaluation that gave the smallest estimate, so
# its mean energy equals that estimate. On a quantum device it would be a new
# batch.
#
# The second cell counts the samples per cut. `decoded.samples` lists each
# distinct spin assignment once, with its number of occurrences. An assignment
# and its flipped copy are the same cut. To count them together, we represent
# each cut by the one of its two assignments in which vertex 0 has spin $+1$:
# an assignment with spin $-1$ on vertex 0 is flipped first. Then we add up the
# occurrences. We compute the size of every measured cut, take the largest as
# the answer of QAOA, and compare it with the exact solution.

# %%
final_gammas = tuple(float(v) for v in res.x[:p])
final_betas = tuple(float(v) for v in res.x[p:])

final_result = qm.run(
    executable,
    on=executor,
    shots=sample_shots,
    seed=SEED,
    parameters={"gammas": final_gammas, "betas": final_betas},
).result

decoded = spin_model.decode(final_result)
print(f"Mean energy of the final batch: {decoded.mean_energy:.4f}")
# The same compiled object is still in use, and the fixed seed reproduces the
# samples of the evaluation at these angles.
assert id(executable) == artifact_identity
assert decoded.mean_energy == res.fun


# %%
def cut_size(spins):
    return sum(1 for i, j in G.edges() if spins[i] != spins[j])


# Count the samples per cut. A spin assignment and its flipped copy are the
# same cut, represented here by the one with spin +1 on vertex 0.
cut_counts: Counter[tuple[int, ...]] = Counter()
for sample in decoded.samples:
    assignment = dict(sample.assignment)  # {vertex: spin}
    spins = tuple(assignment[i] for i in range(num_nodes))
    cut_counts[tuple(s * spins[0] for s in spins)] += sample.occurrences

best_qaoa_sample = max(cut_counts, key=cut_size)
best_qaoa_cut = cut_size(best_qaoa_sample)
# Every exact largest cut, written with spin +1 on vertex 0.
exact_cuts = {tuple(s * part[0] for s in part) for part in optimal_partitions}

print(f"Distinct cuts measured: {len(cut_counts)} of {2 ** (num_nodes - 1)}")
print(f"Largest measured cut size: {best_qaoa_cut}  (exact: {best_cut})")
print(f"Its spin assignment: {best_qaoa_sample}")
# Every shot is counted, and the answer is an exact largest cut.
assert sum(cut_counts.values()) == sample_shots
assert best_qaoa_cut == best_cut
assert best_qaoa_sample in exact_cuts

# %% [markdown]
# ### Distribution of Cuts
#
# The answer alone does not show how the circuit spreads its samples over the
# cuts. The chart has one bar per measured cut, from the most to the least
# frequent, with the cut size written above each bar. Each cut is labeled by
# the vertices that are not in the group of vertex 0, so `{1,2,4}` is the cut
# that separates vertices 1, 2 and 4 from vertices 0 and 3. The label `{}` is
# the split with every vertex in one group, which has size 0.
#
# The cell prints the most frequent cut with its number of shots, and checks
# that it is the exact largest cut. So the largest cut is not only found among
# the samples. It is also the cut measured most often in this batch.

# %%
# One bar per measured cut, most frequent first. A cut is labeled by the
# vertices that are not in the group of vertex 0.
ranked = cut_counts.most_common()
labels = [
    "{" + ",".join(str(i) for i in range(num_nodes) if cut[i] == -1) + "}"
    for cut, _ in ranked
]
most_frequent_cut, most_frequent_count = ranked[0]
print(
    f"Most frequent cut: {labels[0]}, size {cut_size(most_frequent_cut)}, "
    f"{most_frequent_count} of {sample_shots} shots"
)
assert most_frequent_cut in exact_cuts

plt.figure(figsize=(9, 4))
bars = plt.bar(labels, [count for _, count in ranked], color="#2696EB")
for bar, (cut, _) in zip(bars, ranked):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(cut_size(cut)),
        ha="center",
        va="bottom",
    )
plt.xlabel("Cut, labeled by the vertices outside the group of vertex 0")
plt.ylabel("Number of shots")
plt.title("Measured cuts, with the cut size above each bar")
plt.xticks(rotation=45)
plt.margins(y=0.12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Visualize the Best Cut
#
# Finally, we draw the answer on the graph. The two node colors are the two
# groups. Solid edges cross the cut, and dashed edges have both ends in the
# same group. The title gives the cut size, which is the number of solid edges.

# %%
color_map = ["#FF6B6B" if spin == +1 else "#4ECDC4" for spin in best_qaoa_sample]
crossing = [(i, j) for i, j in G.edges() if best_qaoa_sample[i] != best_qaoa_sample[j]]
not_crossing = [edge for edge in G.edges() if edge not in crossing]
# The cut size is the number of crossing edges.
assert len(crossing) == best_qaoa_cut

plt.figure(figsize=(5, 4))
# The colors are listed in the order of the vertex labels, so the nodes are too.
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=list(range(num_nodes)),
    node_color=color_map,
    node_size=700,
    edgecolors="black",
)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos, edgelist=crossing, width=2)
nx.draw_networkx_edges(G, pos, edgelist=not_crossing, style="dashed", edge_color="gray")
plt.title(f"Largest measured cut (size {best_qaoa_cut})")
plt.axis("off")
plt.show()

# %% [markdown]
# ## Qamomile Built-ins
#
# ### Using `qaoa_state`
#
# Steps 1 to 4 built the QAOA state by hand. `qamomile.stdlib.qaoa_state`
# builds it in one call. It takes the number of qubits, the number of layers,
# the cost Hamiltonian and the angles, and returns the qubits in the QAOA
# state. The Hamiltonian comes from the model as `spin_model.hamiltonian`, so
# the edges and coefficients no longer have to be passed one by one.


# %%
@qm.qkernel
def qaoa_builtin(
    layers: qm.Const[qm.Int],
    size: qm.Const[qm.Int],
    hamiltonian: qm.Const[qm.Observable],
    gammas: qm.Param[qm.Vector[qm.Real, layers]],
    betas: qm.Param[qm.Vector[qm.Real, layers]],
) -> qm.Shot[qm.Bits[size]]:
    q = qaoa_state(size, layers, hamiltonian, gammas, betas)
    return qm.measure(q)


builtin_program = qm.specialize(
    qaoa_builtin,
    layers=p,
    size=num_nodes,
    hamiltonian=spin_model.hamiltonian,
)

# %% [markdown]
# We compile this kernel and run both circuits at the final angles of the
# [Result](#qaoa-maxcut-result) section. We use the same `Reference` simulator and seed for
# both runs. The built-in program has no explicit $R_Z$ gates because this
# Hamiltonian has no linear terms, while our
# `cost_layer` keeps its 15 $R_Z$ gates with angle 0. A rotation by angle 0
# does nothing, so leaving these gates out does not change the state. With the
# same seed, this simulator returns exactly the same counts for the two
# circuits, which the cell checks, and so the two mean energies are equal.

# %%
exe_builtin = qm.compile(builtin_program, to=executor)

result_manual = qm.run(
    executable,
    on=executor,
    shots=sample_shots,
    seed=SEED,
    parameters={"gammas": final_gammas, "betas": final_betas},
).result
result_builtin = qm.run(
    exe_builtin,
    on=executor,
    shots=sample_shots,
    seed=SEED,
    parameters={"gammas": final_gammas, "betas": final_betas},
).result

decoded_manual = spin_model.decode(result_manual)
decoded_builtin = spin_model.decode(result_builtin)
print(f"Hand-built mean energy: {decoded_manual.mean_energy:.4f}")
print(f"Built-in mean energy:   {decoded_builtin.mean_energy:.4f}")
# The built-in program contains no explicit R_Z gates.
builtin_study = qm.resources.logical(builtin_program)
rz_count = qm.resources.logical_operation_count("quantum.rz")
assert rz_count.identifier not in builtin_study.values
# With the same seed, the two circuits give the same samples here, and so the
# same mean.
assert result_manual.counts == result_builtin.counts
assert decoded_manual.mean_energy == decoded_builtin.mean_energy

# %% [markdown]
# ## Summary
#
# In this tutorial we:
#
# 1. Wrote MaxCut on a 5-vertex graph as an Ising model on spin variables with
#    `BinaryModel.from_ising`, and found the exact largest cut, of size 5, by
#    trying every assignment.
# 2. Built the QAOA circuit from `@qkernel` functions: the uniform
#    superposition, the cost layer, the mixer layer, and the full circuit with
#    3 layers.
# 3. Tuned the 6 angles with COBYLA, starting from a linear ramp. Every
#    evaluation ran the compiled circuit at a new set of angles and returned
#    the mean energy of its samples.
# 4. Sampled the circuit at the returned angles and counted the samples per
#    cut. The largest measured cut was the exact one, and it was also the cut
#    measured most often.
# 5. Built the same state in one call with `qamomile.stdlib.qaoa_state`, and
#    obtained the same measurement counts.
#
# The kernels of Steps 1 to 4 take the coefficients $h_i$ and $J_{ij}$ as
# inputs, so they also build the circuit for weighted MaxCut and for Ising
# models with linear terms: pass the coefficients to `BinaryModel.from_ising`
# and reuse the kernels. The time step $\Delta t$ of the ramp was taken from
# MaxCut on 3-regular graphs and may need to change with the model.
#
# For a problem with a constraint, written with JijModeling and solved with the
# higher-level `QAOAExperiment`, see
# [QAOA for Graph Partitioning](qaoa_graph_partition).

# %% [markdown]
# (qaoa-maxcut-small-angle-analysis)=
# ## Appendix: Small-Angle Analysis of One Layer
#
# Take one layer with small angles. After the cost
# unitary, the amplitude of every bitstring $z$ is
# $e^{-i\gamma E(z)}/\sqrt{2^n}$. To first order in $\beta$, the mixer is
# $I - i\beta \sum_i X_i$, so it adds to the amplitude of $y$ the amplitudes of
# the $n$ bitstrings $y^{(i)}$:
#
# $$
# a_y \approx \frac{1}{\sqrt{2^n}}
# \Bigl(e^{-i\gamma E(y)} - i\beta \sum_i e^{-i\gamma E(y^{(i)})}\Bigr) .
# $$
#
# Bitstrings that differ from $y$ in two or more bits also send amplitude to
# $y$. Each flipped bit brings one factor $\sin\beta$, so these contributions
# are of order $\beta^2$ or smaller and are dropped here.
#
# The probability is the squared magnitude $|a_y|^2$. It has no term in $\beta$
# alone or in $\gamma$ alone, because the mixer alone and the cost unitary
# alone leave the uniform probabilities unchanged. Its first term that depends
# on $y$ is proportional to the product $\beta\gamma$. Keeping that term and
# dropping those with higher powers of the angles gives
#
# $$
# |a_y|^2 \approx \frac{1}{2^n}
# \Bigl(1 + 2\beta\gamma \sum_i \bigl(E(y) - E(y^{(i)})\bigr)\Bigr) .
# $$
#
# The probability of $y$ depends on how its energy compares with the energies
# of these $n$ bitstrings. Flipping spin $i$ changes the sign of every term
# that contains $s_i$ and leaves the other terms unchanged, so
#
# $$
# E(y) - E(y^{(i)}) = 2\, s_i \sum_{j \ne i} J_{ij}\, s_j ,
# $$
#
# where $s_i = 1 - 2y_i$ are the spins of $y$, and $J_{ij} = J_{ji}$ is 1 if
# $i$ and $j$ are joined by an edge and 0 otherwise. Summing over $i$ gives a
# double sum,
#
# $$
# \sum_i \bigl(E(y) - E(y^{(i)})\bigr)
# = 2 \sum_i \sum_{j \ne i} J_{ij}\, s_i s_j .
# $$
#
# The double sum contains each pair twice, as $(i, j)$ and as $(j, i)$, while
# $E(y) = \sum_{i<j} J_{ij}\, s_i s_j$ contains it once. The double sum
# therefore equals $2E(y)$, and the sum over $i$ is $4E(y)$. Hence
#
# $$
# |a_y|^2 \approx \frac{1}{2^n}\bigl(1 + 8\beta\gamma\, E(y)\bigr) .
# $$
#
# With $\gamma > 0$ and $\beta < 0$, the factor $\beta\gamma$ is negative, so
# bitstrings with lower energy, that is, larger cuts, become more likely than
# in the uniform state. With $\beta > 0$ the same formula favors high energies
# instead, so the sign of $\beta$ matters.
#
# Both unitaries are needed. Acting on the uniform state, the cost unitary
# alone changes only phases, and the mixer alone changes nothing that can be
# measured.
