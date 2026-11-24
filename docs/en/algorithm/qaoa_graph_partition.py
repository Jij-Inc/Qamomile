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
# # QAOA for Graph Partitioning
#
# In this tutorial, we solve a **graph partitioning** problem with the Quantum
# Approximate Optimization Algorithm (QAOA): split the vertices of a graph into
# two groups of equal size so that as few edges as possible run between the two
# groups. The equal-size requirement is a constraint, and the standard QAOA
# circuit has no direct way to impose one. We therefore add it to the cost as a
# penalty term, and check every sampled solution against the original problem
# afterwards.
#
# Graph partitioning is not the same problem as MaxCut. It *minimizes* the
# number of edges between the two groups, where MaxCut maximizes it, and it
# requires the groups to have equal size.
#
# The tutorial follows these steps:
#
# 1. Write the problem with JijModeling, and fill in the graph data to get a
#    concrete problem instance.
# 2. Pass the instance to `OMMXProblem`, which turns the constraint into a
#    penalty term.
# 3. Build the QAOA circuit from the penalized problem with Qamomile's
#    `QAOAExperiment`, and compile it.
# 4. Run the circuit repeatedly while a classical optimizer tunes its angles.
# 5. Measure the circuit once more at the angles the optimizer returned, and
#    evaluate every sample on the original problem.
#
# **Prerequisites**: qubits, quantum gates and measurement. If you are new to
# these, start with
# [Your First Quantum Kernel](../tutorial/01_your_first_quantum_kernel.ipynb).
# No prior knowledge of QAOA is needed. We use JijModeling only to write the
# problem down. See its
# [documentation](https://jij-inc-jijmodeling-tutorials-en.readthedocs-hosted.com/en/latest/introduction.html)
# for the syntax.
#

# %% [markdown]
# ## Installation

# %%
# Install the latest Qamomile, JijModeling and NetworkX through pip!
# # !pip install qamomile jijmodeling networkx

# %%
import itertools
import os

import jijmodeling as jm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import minimize

import qamomile as qm
from qamomile.optimization import BinaryModel, OMMXProblem, QAOAExperiment, VarType

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# ## Problem Settings
#
# ### Problem Definition
#
# A graph $G = (V, E)$ consists of a set $V$ of vertices and a set $E$ of
# edges, where an edge $(u, v)$ joins the two vertices $u$ and $v$. Our graph
# is undirected: $(u, v)$ and $(v, u)$ are the same edge, and a sum over $E$
# counts it once. The number of vertices $|V|$ is even. The graph is fixed: no
# edge is added or removed.
#
# The only choice we make is how to split the vertices into two groups, group 0
# and group 1. Such a split is called a *partition*. For each vertex $u$, a
# binary variable $x_u$ records the choice: $x_u = 0$ if $u$ is in group 0, and
# $x_u = 1$ if $u$ is in group 1. A choice of values for all the variables is
# an *assignment* $\boldsymbol{x}$. An edge *crosses* the partition when its
# two ends lie in different groups, and the number of crossing edges is the
# *cut size* of the partition. The task is to choose the groups so that
#
# - both groups contain exactly $|V|/2$ vertices, and
# - the cut size is as small as possible.
#
# **Objective function.** The quantity we want to make small is the cut size.
# Written with the variables $x_u$, it is
#
# $$
# C(\boldsymbol{x})
# = \sum_{(u,v) \in E} \bigl[x_u(1 - x_v) + x_v(1 - x_u)\bigr] .
# $$
#
# Each term is $1$ exactly when the edge crosses the partition, and $0$
# otherwise, so the sum counts the crossing edges.
#
# **Constraint.** The two groups must have the same size:
#
# $$
# \sum_{u \in V} x_u = \frac{|V|}{2} .
# $$
#
# The sum counts the vertices in group 1, so this says that group 1 holds
# exactly half of the vertices, and group 0 holds the rest.
#
# **Goal.** Among the assignments that satisfy the constraint, find one that
# achieves the smallest value of the objective function:
#
# $$
# \min_{\boldsymbol{x}} C(\boldsymbol{x})
# \quad \text{subject to} \quad
# \sum_{u \in V} x_u = \frac{|V|}{2} .
# $$
#
# Without the constraint the problem would be trivial: with every vertex in one
# group, no edge crosses, and the cut size is 0.
#
# ### Problem Instance
#
# We use a fixed graph with 8 vertices and 16 edges.
#

# %%
NUM_NODES = 8
EDGE_LIST = (
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 7),
    (2, 3),
    (2, 6),
    (3, 5),
    (4, 5),
    (4, 6),
    (5, 6),
    (5, 7),
    (6, 7),
)

# The two groups have equal size, so the number of vertices must be even.
assert NUM_NODES % 2 == 0, "NUM_NODES must be even: the two groups have equal size."
# JijModeling cannot fill in an empty edge list, so the graph needs an edge.
assert len(EDGE_LIST) > 0, "EDGE_LIST must contain at least one edge."

graph = nx.Graph()
graph.add_nodes_from(range(NUM_NODES))
graph.add_edges_from(EDGE_LIST)
positions = nx.spring_layout(graph, seed=1)

assert graph.number_of_nodes() == NUM_NODES
assert graph.number_of_edges() == len(EDGE_LIST) == 16

# %%
plt.figure(figsize=(5, 5))
nx.draw(
    graph,
    positions,
    with_labels=True,
    node_color="white",
    node_size=700,
    edgecolors="black",
)
plt.title(f"Graph: {graph.number_of_nodes()} vertices, {graph.number_of_edges()} edges")
plt.show()

# %% [markdown]
# To try another graph, change `NUM_NODES` and `EDGE_LIST`. Keep the number of
# vertices even, and keep at least one edge. Counts in the assertions in this
# tutorial, such as the 16 edges here and the 28 terms of the Hamiltonian
# below, hold for this graph only, so update them too. Also check that the
# penalty weight is still large enough, as explained in
# [Penalty Formulation](#qaoa-graph-partition-penalty-formulation). If it is
# too small, the brute-force check under
# [Create the Penalty Model](#qaoa-graph-partition-penalty-model) stops the
# notebook.
#
# The [Result](#qaoa-graph-partition-result) section checks the run with three
# assertions. The share of feasible samples must be larger than among uniformly
# random bitstrings. The best feasible sample must have the smallest cut size
# that the exact solution found. And that cut size must be sampled more often
# than uniformly random bitstrings would give in the same number of shots. QAOA
# guarantees none of these. If one of them fails for your graph or settings,
# the notebook stops there with an `AssertionError`. This reports a weaker QAOA
# result, not a mistake in the code.
#
# Larger graphs or more shots can exceed the work limit of `Reference`. If that
# happens, reduce `SAMPLE_SHOTS` and `FINAL_SHOTS`.
#
# ### Formulation with JijModeling
#
# We now write the same problem with [JijModeling](https://jij-inc-jijmodeling-tutorials-en.readthedocs-hosted.com/en/latest/introduction.html).
# The comments in the code show which line corresponds to which part of the problem.

# %%
# Create an empty problem named "Graph Partitioning". By default it is minimized.
formulation = jm.Problem("Graph Partitioning")


# `update` runs this function once, and the function fills in the problem.
@formulation.update
def _(problem: jm.DecoratedProblem):
    # Placeholders for the graph data. `eval` fills them in later.
    vertices = problem.Dim()  # number of vertices, |V|
    edges = problem.Natural(ndim=2)  # edge list, one row [u, v] per edge

    # One binary variable x_u per vertex.
    x = problem.BinaryVar(shape=(vertices,))

    # Objective function C(x): one term per edge, equal to 1 when the edge
    # crosses the partition.
    problem += (
        edges.rows()
        .map(lambda edge: x[edge[0]] * (1 - x[edge[1]]) + x[edge[1]] * (1 - x[edge[0]]))
        .sum()
    )

    # Constraint: group 1 holds exactly half of the vertices.
    problem += problem.Constraint("Equal Partition", x.sum() == vertices / 2)


formulation

# %% [markdown]
# Filling in the graph data gives a concrete problem instance.

# %%
instance = formulation.eval(
    {
        "vertices": NUM_NODES,
        "edges": [list(edge) for edge in EDGE_LIST],
    }
)

# %% [markdown]
# ### Exact Solution (Brute Force)
#
# With 8 vertices, there are only $2^8 = 256$ assignments, so we can find the
# exact answer by trying them all. We keep it as a reference value for checking
# the QAOA result later. This does not scale, because the number of assignments
# doubles with every added vertex.

# %%
# Try every assignment of the vertices to groups 0 and 1.
cuts = {}
for bits in itertools.product((0, 1), repeat=NUM_NODES):
    # bits[u] is the group of vertex u. Skip assignments with unequal groups.
    if sum(bits) != NUM_NODES // 2:
        continue
    # Count the edges that cross: their two ends lie in different groups.
    cuts[bits] = sum(bits[left] != bits[right] for left, right in EDGE_LIST)

# Find the smallest cut size and every assignment that reaches it.
best_feasible_cut = min(cuts.values())
optimal_partitions = [bits for bits, cut in cuts.items() if cut == best_feasible_cut]

print(f"Equal-size assignments: {len(cuts)}")
print(f"Smallest cut size: {best_feasible_cut}")
for bits in optimal_partitions:
    print(f"Optimal assignment: {bits}")

# Swapping the two groups of an optimal assignment gives another optimal one.
assert all(
    tuple(1 - bit for bit in bits) in optimal_partitions for bits in optimal_partitions
)

# %% [markdown]
# The two optimal assignments are the same partition with the names of the two
# groups swapped. Swapping the names does not change which edges cross, so
# optimal assignments always come in such pairs.

# %% [markdown]
# (qaoa-graph-partition-algorithm)=
# ## Algorithm
#
# The Quantum Approximate Optimization Algorithm was introduced by
# [Farhi, Goldstone and Gutmann, arXiv:1411.4028](https://doi.org/10.48550/arXiv.1411.4028).
# Its input is a cost function: a rule that gives a number, the cost, to every
# assignment of the binary variables. From this function QAOA builds a circuit
# with $2p$ adjustable angles, where $p$ is a number of repeated layers that we
# choose. Running the circuit and measuring the qubits returns one assignment.
# A classical optimizer tunes the angles so that the measurements are likely to
# return assignments of low cost. This section first turns the constrained
# problem into an unconstrained one with a single cost function, the form that
# QAOA needs, and writes that function as an Ising energy. It then explains the
# circuit and its state, how the angles are tuned, and how the samples are
# checked against the original problem.
#
# (qaoa-graph-partition-penalty-formulation)=
# ### Penalty Formulation
#
# QAOA minimizes a single cost over all assignments, with no side conditions.
# We therefore move the constraint into the cost as a quadratic penalty with
# weight $\lambda > 0$. The result is the penalized cost $E_\lambda$, called the
# energy, because it becomes the energy of an Ising model below:
#
# $$
# E_\lambda(\boldsymbol{x})
# = C(\boldsymbol{x})
# + \lambda \Bigl(\sum_{u \in V} x_u - \frac{|V|}{2}\Bigr)^2 .
# $$
#
# An assignment is called *feasible* if it satisfies the constraint. The
# penalty is zero on every feasible assignment and at least $\lambda$ on every
# other one. It does not forbid infeasible assignments. It only makes them
# expensive.
#
# The weight must be large enough. If $\lambda$ is too small, the minimum of
# $E_\lambda$ can be infeasible. For example, at $\lambda = 0$ the minimum puts
# every vertex in one group, where no edge crosses.
#
# A simple estimate looks at one move. Take an equal-size partition and move
# one vertex to the other group. The penalty rises from 0 to $\lambda$. The cut
# size falls by at most the number of crossing edges at that vertex, which is
# at most $\min(\Delta, |V|/2)$, where $\Delta$ is the largest number of edges
# at any vertex. So the move does not lower the energy if
# $\lambda \ge \min(\Delta, |V|/2)$. This is the comparison made by
# [Lucas, Frontiers in Physics 2, 5 (2014)](https://doi.org/10.3389/fphy.2014.00005) (Section
# 2.2). This graph has $\Delta = 5$ and $|V| = 8$, so the estimate asks for
# $\lambda \ge 4$. It considers a single move only, so it is an estimate and
# not a proof. We use $\lambda = 4$, and
# [Create the Penalty Model](#qaoa-graph-partition-penalty-model) checks by
# brute force that this value works.
#
# ### Ising Form
#
# The Pauli operator $Z$ has eigenvalue $+1$ on $|0\rangle$ and $-1$ on
# $|1\rangle$. To match these values, we replace each binary variable $x_u$ by a
# spin $s_u \in \{+1, -1\}$:
#
# $$
# x_u = \frac{1 - s_u}{2} ,
# $$
#
# so $x_u = 0$ means $s_u = +1$, and $x_u = 1$ means $s_u = -1$. Each edge term
# of $C$ then becomes $(1 - s_u s_v)/2$, and $\sum_u x_u - |V|/2$ becomes
# $-\tfrac{1}{2}\sum_u s_u$. The energy is therefore
#
# $$
# E_\lambda(\boldsymbol{s})
# = \sum_{(u,v) \in E} \frac{1 - s_u s_v}{2}
# + \frac{\lambda}{4}\Bigl(\sum_{u \in V} s_u\Bigr)^2 .
# $$
#
# Lucas writes the cost in this form, with a weight $A$ on the squared sum of
# the spins and a weight $B$ on the cut size, so our cost is his with
# $A = \lambda/4$ and $B = 1$. His Eq. (10), $A/B \ge \min(2\Delta, N)/8$ with
# $N = |V|$, is the estimate $\lambda \ge \min(\Delta, |V|/2)$ of the previous
# section.
#
# Expanding the square with $s_u^2 = 1$ leaves a constant and one term for every
# pair of vertices:
#
# $$
# E_\lambda(\boldsymbol{s}) = c_0 + \sum_{u < v} J_{uv}\, s_u s_v ,
# \qquad
# c_0 = \frac{|E|}{2} + \frac{\lambda |V|}{4} ,
# \qquad
# J_{uv} =
# \begin{cases}
# \lambda/2 - 1/2 & (u, v) \in E \\
# \lambda/2 & \text{otherwise.}
# \end{cases}
# $$
#
# This is the energy of an Ising model, a model of interacting spins, in which
# $J_{uv}$ is the coupling between the spins $u$ and $v$. There is no term with
# a single spin, because swapping the two groups,
# $\boldsymbol{s} \to -\boldsymbol{s}$, does not change the energy. The
# constant $c_0$ is the average of the energy over all $2^{|V|}$ assignments,
# because each product $s_u s_v$ averages to zero.
#
# Replacing each spin $s_u$ by the operator $Z_u$ on qubit $u$ gives the *cost
# Hamiltonian*
#
# $$
# H_C = c_0 + \sum_{u < v} J_{uv}\, Z_u Z_v .
# $$
#
# Each $Z_u$ is diagonal in the computational basis, so $H_C$ is diagonal too.
# For a bitstring $z$, which is an assignment of all $x_u$,
#
# $$
# H_C |z\rangle = E_\lambda(z)\, |z\rangle .
# $$
#
# Every bitstring is therefore an eigenstate of $H_C$, with its own energy as
# the eigenvalue. The assignments with the lowest energy are the eigenstates
# with the lowest eigenvalue, the *ground states* of $H_C$. There are at least
# two, because swapping the groups keeps the energy. Solving the problem means
# finding one of them.
#
# For this graph with $\lambda = 4$, we get $c_0 = 16$, $J_{uv} = 1.5$ on the 16
# edges and $J_{uv} = 2$ on the other 12 pairs. So $H_C$ has $\binom{8}{2} = 28$
# terms of the form $Z_u Z_v$, even though the graph has only 16 edges, because
# the penalty couples every pair of vertices.
#
# ### The QAOA State
#
# QAOA looks for a ground state of $H_C$ with a circuit on $n = |V|$ qubits,
# one qubit per vertex. The qubits can hold all bitstrings at once, in a
# superposition
#
# $$
# |\psi\rangle = \sum_{z} a_z\, |z\rangle ,
# $$
#
# and measuring them returns the bitstring $z$ with probability $|a_z|^2$. The
# aim is to prepare a state in which low-energy bitstrings have large
# probabilities, so that a modest number of measurements is likely to return a
# good partition.
#
# **Initial state.** The circuit starts from the uniform superposition
# $|+\rangle^{\otimes n}$, which gives every bitstring the same probability. A
# Hadamard gate on each qubit prepares it.
#
# **Cost unitary.** For an angle $\gamma$, the cost unitary is
# $U_C(\gamma) = e^{-i\gamma H_C}$. Because
# $H_C |z\rangle = E_\lambda(z) |z\rangle$, it gives each bitstring a phase set by
# its energy:
#
# $$
# U_C(\gamma) |z\rangle = e^{-i\gamma E_\lambda(z)}\, |z\rangle .
# $$
#
# A phase alone changes no measurement probability. It only records the energy
# of each bitstring in the state.
#
# **Mixer unitary.** For an angle $\beta$, the mixer unitary is
#
# $$
# U_M(\beta) = e^{-i\beta \sum_u X_u}
# = \prod_u \bigl(\cos\beta\; I - i \sin\beta\; X_u\bigr) ,
# $$
#
# where $I$ is the identity and the Pauli operator $X_u$ flips bit $u$. Write
# $y^{(u)}$ for a bitstring $y$ with bit $u$ flipped. The factor for qubit $u$
# turns $|y\rangle$ into a combination of $|y\rangle$ and $|y^{(u)}\rangle$:
#
# $$
# \bigl(\cos\beta\; I - i \sin\beta\; X_u\bigr)\, |y\rangle
# = \cos\beta\, |y\rangle - i \sin\beta\, |y^{(u)}\rangle .
# $$
#
# Each factor therefore passes amplitude between two bitstrings that differ in
# one bit. Unless $\sin\beta$ or $\cos\beta$ is zero, the product of the $n$
# factors connects every pair of bitstrings. Amplitudes that arrive at the same
# bitstring with similar phases add up, and those with opposite phases cancel.
# This interference is what changes the probabilities.
#
# **Layers.** A cost unitary followed by a mixer is one *layer*. With $p$ layers
# and the angles $\boldsymbol{\gamma} = (\gamma_1, \dots, \gamma_p)$ and
# $\boldsymbol{\beta} = (\beta_1, \dots, \beta_p)$, the QAOA state is
#
# $$
# |\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle
# = U_M(\beta_p)\, U_C(\gamma_p) \cdots U_M(\beta_1)\, U_C(\gamma_1)\,
# |+\rangle^{\otimes n} ,
# $$
#
# read from right to left. The problem fixes the structure of the circuit through
# $H_C$, and the $2p$ angles are the only free parameters.
#
# **Why low energies gain.** Take one layer with small angles. The cost unitary
# gives the amplitude of every bitstring a phase set by its energy, and to
# first order in $\beta$ the mixer adds to the amplitude of $y$ the amplitudes
# of the $n$ bitstrings $y^{(u)}$. In the probability of $y$, the terms in
# $\beta$ alone and in $\gamma$ alone vanish, because the mixer alone and the
# cost unitary alone leave the uniform probabilities unchanged. The
# lowest-order term that depends on $y$ is proportional to the product
# $\beta\gamma$. Keeping it and dropping the terms with higher powers of the
# angles gives
#
# $$
# |a_y|^2 \approx \frac{1}{2^n}
# \bigl(1 + 8\beta\gamma\,(E_\lambda(y) - c_0)\bigr) .
# $$
#
# With $\gamma > 0$ and $\beta < 0$, the factor $\beta\gamma$ is negative, so
# bitstrings with energy below the average $c_0$ become more likely than in the
# uniform state, and those above it less likely. With $\beta > 0$ the same
# formula favors high energies instead, so the sign of $\beta$ matters.
# [QAOA for MaxCut](qaoa_maxcut) derives the formula step by step, in its
# section The QAOA State, for an energy without a constant. The same steps give
# the formula here. It is for one layer and small angles, so it shows the
# direction of the effect, not the probabilities of the circuit used below.
#
# Both unitaries are needed. Acting on the uniform state, the cost unitary
# alone changes only phases, and the mixer alone changes nothing that can be
# measured.
#
# **Gates.** All terms of $H_C$ commute, because they are all made of $Z$
# operators, so $U_C$ splits exactly into one factor per term, and the mixer
# splits into one factor per qubit. Qamomile's rotation gates are defined with
# a factor $1/2$ in the exponent, $R_{ZZ}(\theta) = e^{-i\theta Z \otimes Z/2}$
# and $R_X(\theta) = e^{-i\theta X/2}$, so each factor is one gate with twice
# the angle:
#
# $$
# e^{-i\gamma J_{uv} Z_u Z_v} = R_{ZZ}(2\gamma J_{uv}) ,
# \qquad
# e^{-i\beta X_u} = R_X(2\beta) .
# $$
#
# The constant $c_0$ adds only a global phase, which has no measurable effect.
# Each layer therefore has 28 $R_{ZZ}$ gates and 8 $R_X$ gates.
#
# ### Optimizing the Angles
#
# The angles are chosen to make the expected energy of the QAOA state as small as
# possible:
#
# $$
# F_p(\boldsymbol{\gamma}, \boldsymbol{\beta})
# = \langle \boldsymbol{\gamma}, \boldsymbol{\beta} |\, H_C \,|
# \boldsymbol{\gamma}, \boldsymbol{\beta} \rangle
# = \sum_z \bigl|\langle z | \boldsymbol{\gamma}, \boldsymbol{\beta}
# \rangle\bigr|^2\, E_\lambda(z) .
# $$
#
# Because $E_\lambda$ is the cut size plus the penalty, $F_p$ is the expected
# cut size plus the expected penalty. It can come close to its lower limit, the
# minimum of $E_\lambda$, only if infeasible samples are rare and the cut sizes
# of the feasible samples are small.
#
# $F_p$ is a sum over all $2^n$ bitstrings, but we never need the individual
# probabilities. A quantum computer does not output them. Each run of the
# circuit returns one bitstring, and the bitstring $z$ appears with probability
# $|\langle z | \boldsymbol{\gamma}, \boldsymbol{\beta} \rangle|^2$. Running
# the circuit $N$ times gives bitstrings $z^{(1)}, \dots, z^{(N)}$. The energy
# of each is easy to compute, and their average estimates $F_p$:
#
# $$
# F_p(\boldsymbol{\gamma}, \boldsymbol{\beta})
# \approx \frac{1}{N} \sum_{k=1}^{N} E_\lambda\bigl(z^{(k)}\bigr) .
# $$
#
# This estimate carries statistical noise, which shrinks as $N$ grows. A
# classical optimizer searches for the angles and treats $F_p$ as a black box.
# It chooses a set of angles, the circuit is run $N$ times at those angles, and
# the average energy comes back as the value. From the values received so far,
# the optimizer chooses the next set of angles, and the loop repeats until the
# optimizer stops. Every evaluation is a new batch of $N$ shots, so the total
# cost is the number of evaluations times $N$. QAOA is called a hybrid
# algorithm for this reason: the quantum circuit evaluates the cost at given
# angles, and a classical computer searches for the angles.
#
# This tutorial uses SciPy's Constrained Optimization BY Linear Approximation
# (COBYLA) method, which needs no gradients. It keeps the values at $2p + 1$
# sets of angles, fits a linear function of the angles through them, and uses
# that function to choose the next set. Each step costs a single evaluation.
# The [Result](#qaoa-graph-partition-result) section shows how it starts and
# stops.
#
# The circuit is sampled in two phases. During optimization, the samples at
# each set of angles serve only to estimate $F_p$, and their individual
# bitstrings are not used. After optimization, a final batch at the returned
# angles gives the candidates for the answer.
#
# More layers can only help in principle. Let $M_p$ be the smallest value of
# $F_p$ over all angles. Then $M_p \le M_{p-1}$, and $M_p$ approaches the
# minimum of $E_\lambda$ as $p \to \infty$
# ([Farhi, Goldstone and Gutmann, arXiv:1411.4028](https://doi.org/10.48550/arXiv.1411.4028),
# Eqs. (9) and (10), written there for maximization). At a small, fixed $p$
# there is no such guarantee, and the result also depends on the optimizer, the
# starting angles, the number of shots and $\lambda$.
#
# **Starting angles.** Optimization from random starting angles tends to end in
# poor local minima, as reported for MaxCut
# ([Sack and Serbyn, Quantum 5, 491 (2021)](https://doi.org/10.22331/q-2021-07-01-491)).
# We therefore start from angles taken from quantum annealing.
#
# Quantum annealing prepares the lowest-energy state of one Hamiltonian and
# then changes the Hamiltonian slowly into the one of the problem. We take
#
# $$
# H(s) = -(1 - s) \sum_u X_u + s\, H_C
# $$
#
# and let $s$ go from 0 to 1. At $s = 0$ the lowest-energy state is the uniform
# state, which is where the circuit starts. At $s = 1$ the lowest energy
# belongs to the optimal partition, and its two assignments share it. A system
# that starts in the lowest-energy state stays close to the lowest-energy state
# while the Hamiltonian changes, if the change is slow enough and an energy gap
# keeps that state apart from the others
# ([Albash and Lidar, Reviews of Modern Physics 90, 015002 (2018)](https://doi.org/10.1103/RevModPhys.90.015002)).
# The two assignments that share the lowest energy do not break this condition.
# Swapping the two groups changes neither $H(s)$ nor the uniform state, so the
# state keeps this symmetry along the way, and only the gap to other states
# with the same symmetry matters. If that gap stays open, a slow enough change
# ends in the equal superposition of the two assignments, and a measurement
# then returns the optimal partition. The [QAOA for MaxCut](qaoa_maxcut)
# tutorial uses the same idea to choose its starting angles. We do not check the
# condition here, and use the idea only as a guide for choosing the angles.
#
# To turn this into layers, let the change take a total time $T$, so that
# $s = t/T$ at time $t$, and cut it into $p$ steps of equal duration
# $\Delta t = T/p$. In step $k$ we replace $s$ by its value in the middle of
# the step,
#
# $$
# s_k = \frac{k - 1/2}{p} ,
# \qquad
# k = 1, \dots, p ,
# $$
#
# and apply the two parts of $H(s_k)$ one after the other, each for the time
# $\Delta t$:
#
# $$
# e^{-i \Delta t\, H(s_k)}
# \approx e^{\,i \Delta t\, (1 - s_k) \sum_u X_u}\; e^{-i \Delta t\, s_k H_C}
# = U_M(\beta_k)\, U_C(\gamma_k) .
# $$
#
# This is one QAOA layer with the angles
#
# $$
# \gamma_k = \Delta t\, s_k ,
# \qquad
# \beta_k = -\Delta t\, (1 - s_k) .
# $$
#
# The minus sign in $\beta_k$ comes from the minus sign in front of
# $\sum_u X_u$, and it is the sign that favors low energies in the formula
# above. These angles change steadily from layer to layer, so this choice is
# called a *linear ramp*. With only $p = 5$ steps, nothing shows that the
# change is slow enough, so the ramp is a starting point, which the optimizer
# then improves.
#
# The step $\Delta t$ has to fit the energies of the problem. Sack and Serbyn
# found $\Delta t = 0.75$ for MaxCut on 3-regular graphs, where every coupling
# is 1 and each spin has three of them. Here every spin is coupled to all seven
# others, with strength 1.5 or 2. With $\Delta t = 0.75$ the ramp alone gives
# an expected energy of 19.4, above the 16 of uniformly random bitstrings, and
# with $\Delta t = 0.4$ it gives 8.6. We use $\Delta t = 0.4$, chosen by trying
# several values on this graph, so it is a setting of this tutorial and not a
# general rule. With $p = 5$, the starting angles are
# $\boldsymbol{\gamma} = (0.04, 0.12, 0.2, 0.28, 0.36)$ and
# $\boldsymbol{\beta} = (-0.36, -0.28, -0.2, -0.12, -0.04)$.
#
# ### Feasibility after Sampling
#
# Optimization returns angles, not a partition. They are the angles with the
# smallest estimate of $F_p$ among all the angles COBYLA evaluated. To obtain
# partitions, we run the circuit once more at these angles and keep every
# measured bitstring of this final batch.
#
# The penalty only makes infeasible assignments expensive, so some of these
# samples can still violate the constraint. We therefore evaluate every sample
# on the original problem. Infeasible samples are discarded, and each feasible
# sample is scored by its cut size $C(\boldsymbol{x})$. On a feasible sample,
# $C$ and $E_\lambda$ are equal, because the penalty is zero there.
#
# The answer is the feasible sample with the smallest cut size. QAOA does not
# guarantee that this is the optimum, so the
# [Result](#qaoa-graph-partition-result) section compares it with the exact
# solution.

# %% [markdown]
# ## Implementation with Qamomile
#
# (qaoa-graph-partition-penalty-model)=
# ### Create the Penalty Model
#
# The next step adds the penalty. `OMMXProblem.from_instance` takes the instance
# and the weight $\lambda$, given as `uniform_penalty_weight`, and returns a
# `problem` object that holds two things:
#
# - `problem.model`, the penalized cost $E_\lambda$, which QAOA will minimize.
#   Its Hamiltonian $H_C$ is `problem.model.hamiltonian`.
# - The original problem with its constraint, which `problem.evaluate` uses later
#   to check the samples.

# %%
PENALTY_WEIGHT = 4.0

# Remember the instance, to check that the conversion does not change it.
original_payload = instance.to_bytes()

problem = OMMXProblem.from_instance(instance, uniform_penalty_weight=PENALTY_WEIGHT)

assert instance.to_bytes() == original_payload
assert problem.model.vartype is VarType.BINARY
assert problem.model.num_variables == NUM_NODES
assert problem.model.order == 2
assert BinaryModel.from_bytes(problem.model.to_bytes()) == problem.model

# Compare the Hamiltonian with the expanded form in the Ising Form section.
hamiltonian = problem.model.hamiltonian
couplings = [coefficient.real for _, coefficient in hamiltonian.terms]

print(f"Constant c_0: {hamiltonian.constant.real}")
print(f"Terms with J = lambda/2 - 1/2: {couplings.count(PENALTY_WEIGHT / 2 - 0.5)}")
print(f"Terms with J = lambda/2: {couplings.count(PENALTY_WEIGHT / 2)}")
print(f"QAOA Hamiltonian terms: {len(hamiltonian.terms)}")

assert hamiltonian.constant == len(EDGE_LIST) / 2 + PENALTY_WEIGHT * NUM_NODES / 4
assert couplings.count(PENALTY_WEIGHT / 2) == (
    NUM_NODES * (NUM_NODES - 1) // 2 - len(EDGE_LIST)
)
assert len(hamiltonian.terms) == 28

# %% [markdown]
# Next we check the choice of $\lambda$ by brute force. Over all $2^8$
# assignments, the minimum of $E_\lambda$ should equal the smallest cut size
# from the exact solution, and every assignment that reaches it should be
# feasible. Then minimizing $E_\lambda$ solves the original problem.

# %%
# Compute the energy of every assignment, including the infeasible ones.
penalty_energies = {}
for bits in itertools.product((0, 1), repeat=NUM_NODES):
    penalty_energies[bits] = problem.model.energy(bits)

# Find the smallest energy and every assignment that reaches it.
min_penalty_energy = min(penalty_energies.values())
minimizers = [
    bits for bits, energy in penalty_energies.items() if energy == min_penalty_energy
]

print(f"Minimum of E_lambda: {min_penalty_energy}")
print(f"Smallest cut size: {best_feasible_cut}")
print(f"Number of minimizers: {len(minimizers)}")

assert min_penalty_energy == best_feasible_cut
assert all(sum(bits) == NUM_NODES // 2 for bits in minimizers)

# %% [markdown]
# ### Build and Compile the Experiment
#
# `QAOAExperiment` builds the QAOA circuit of the
# [Algorithm](#qaoa-graph-partition-algorithm) section from `problem.model`,
# here with $p = 5$ layers. In this tutorial, `qm.Reference()`, Qamomile's
# built-in simulator, takes the place of a quantum computer. We compile the
# circuit once. The angles `gammas` and `betas` stay open as parameters and get
# their values at run time, so every evaluation can reuse the same compiled
# circuit.

# %%
LAYERS = 1 if docs_test_mode else 5

experiment = QAOAExperiment(problem.model, layers=LAYERS)

# Compile once for the simulator.
endpoint = qm.Reference()
compiled = experiment.compile(to=endpoint.target)

# Only the angles remain as runtime parameters.
assert tuple(slot.name for slot in compiled.artifact.parameter_abi.slots) == (
    "gammas",
    "betas",
)

# The compiled circuit belongs to this experiment and this problem.
assert compiled.identity == experiment.identity
assert compiled.identity.problem_digest == problem.model.semantic_digest

# QAOA is composed of ordinary operations, not one special operation.
assert "quantum.qaoa" not in experiment.program.inspect()

# %% [markdown]
# ### Inspect the Circuit
#
# Qamomile can count operations without converting the circuit for any
# engine. `qm.resources.logical` reads the program directly. With `LAYERS` layers,
# it reports one Hadamard per qubit, one Pauli evolution per cost layer (the whole
# 28-term cost unitary $U_C$), and one $R_X$ rotation per qubit and layer.

# %%
study = qm.resources.logical(experiment.program)

# One Hadamard per qubit.
assert study.metric(
    qm.resources.logical_operation_count("quantum.h")
).numeric_enclosure() == (NUM_NODES, NUM_NODES)

# One RX rotation per qubit and layer.
assert study.metric(
    qm.resources.logical_operation_count("quantum.rx")
).numeric_enclosure() == (NUM_NODES * LAYERS, NUM_NODES * LAYERS)

# One Pauli evolution per cost layer.
assert study.metric(
    qm.resources.logical_operation_count("quantum.pauli_evolution")
).numeric_enclosure() == (LAYERS, LAYERS)

# %% [markdown]
# (qaoa-graph-partition-result)=
# ## Result
#
# ### Optimize the Angles
#
# We now run the optimization loop of the
# [Algorithm](#qaoa-graph-partition-algorithm) section on the compiled circuit.
# `cost_function` is the function that COBYLA evaluates. Each call runs the
# compiled circuit with `SAMPLE_SHOTS` shots at the given angles and returns
# the mean energy of the samples, which is the estimate of $F_p$. It also
# appends the value to `cost_history`, so that the list holds one entry per
# evaluation. Every call uses a new seed, so every evaluation is a new batch of
# samples.
#
# The starting angles are the linear ramp of the
# [Algorithm](#qaoa-graph-partition-algorithm) section with $\Delta t = 0.4$,
# `RAMP_STEP` in the code. The array `initial_parameters` holds the $p$ values
# of $\gamma$ first and then the $p$ values of $\beta$, the order in which
# `cost_function` splits it. The option `rhobeg` sets the size of COBYLA's
# first steps to 0.1 instead of the default 1.0, which is larger than every
# starting angle. The run may use up to 1,000 evaluations. A run of 256 shots
# stays within the work limit of `Reference`.
#
# After the run, the cell prints the number of evaluations and the number of
# shots they used, the reason why COBYLA stopped, the estimate at the starting
# angles, the smallest estimate of $F_p$, and the angles at which it was
# obtained. The second cell plots the estimate at every evaluation.

# %%
SAMPLE_SHOTS = 1 if docs_test_mode else 256
MAX_ITERATIONS = 2 * LAYERS + 2 if docs_test_mode else 1_000
FINAL_SHOTS = 1 if docs_test_mode else 256
cost_history = []


def cost_function(parameters) -> float:
    """Run the circuit at the given angles and return the mean sample energy."""

    # The optimizer passes all angles in one array: gammas, then betas.
    run = compiled.run(
        on=endpoint,
        gammas=tuple(float(value) for value in parameters[:LAYERS]),
        betas=tuple(float(value) for value in parameters[LAYERS:]),
        shots=SAMPLE_SHOTS,
        seed=901 + len(cost_history),
    )
    cost_history.append(run.samples.mean_energy)
    return run.samples.mean_energy


# Start from the linear ramp of the Algorithm section.
RAMP_STEP = 0.4
ramp = (np.arange(LAYERS) + 0.5) / LAYERS
initial_parameters = np.concatenate([RAMP_STEP * ramp, -RAMP_STEP * (1 - ramp)])

optimizer_options = {"maxiter": MAX_ITERATIONS, "rhobeg": 0.1}
if docs_test_mode:
    optimizer_options["f_target"] = float("inf")

optimization = minimize(
    cost_function,
    initial_parameters,
    method="COBYLA",
    options=optimizer_options,
)
if docs_test_mode:
    assert optimization.nfev == 1

print(
    f"Evaluations: {optimization.nfev}, shots used: {optimization.nfev * SAMPLE_SHOTS}"
)
print(f"Why COBYLA stopped: {optimization.message}")
print(f"Estimate at the starting angles: {cost_history[0]:.4f}")
print(f"Smallest estimate of F_p: {optimization.fun:.4f}")
print(f"Returned gammas: {[round(float(v), 4) for v in optimization.x[:LAYERS]]}")
print(f"Returned betas: {[round(float(v), 4) for v in optimization.x[LAYERS:]]}")
assert len(optimization.x) == 2 * LAYERS
assert len(cost_history) == optimization.nfev

# %%
plt.figure(figsize=(8, 4))
plt.plot(cost_history, color="#2696EB")
plt.xlabel("Energy evaluation")
plt.ylabel("Mean energy")
plt.title("QAOA optimization progress")
plt.show()

# %% [markdown]
# The plot shows how COBYLA proceeds. A linear function of the 10 angles has 11
# coefficients, so COBYLA first needs the values at 11 sets of angles. The
# first 11 points of the plot are these: the starting angles, and then ten sets
# in which one angle at a time is changed by `rhobeg`, each time from the best
# set found so far. In this run nine of the ten estimates lie above the
# starting value, some far above it. The ramp is already a good set of angles,
# and a change of 0.1 in one angle can spoil it.
#
# From then on, COBYLA takes steps guided by the fitted linear function. It
# limits their length to a distance that it reduces as the search settles, and
# it stops once that distance has come down to its lower bound, or when the
# budget of evaluations is used up. The printed reason shows that the first
# ended this run. [QAOA for MaxCut](qaoa_maxcut) describes these steps in more
# detail, under Optimize the Angles.
#
# For comparison, a uniformly random bitstring has mean energy $c_0 = 16$, and
# the minimum energy is 6. The ramp alone already brings the estimate down to
# 8.56, the first point of the plot. In this run the optimizer adds little to
# that. The smallest estimate is 8.13, and one estimate from 256 shots is
# uncertain by about 0.25, so a difference of this size is not far above the
# noise. Indeed, when the next section measures a new batch of 256 shots at the
# same returned angles, its mean energy is 8.70.

# %% [markdown]
# ### Sample and Evaluate
#
# This is the second sampling phase. A final run at the returned angles gives
# the candidate partitions. It has a seed of its own, so it is a new batch, and
# its mean energy is a new estimate of $F_p$. This estimate can lie above the
# smallest estimate of the optimization, which is the lowest of many noisy
# values, and it does in this run. The cell also prints the standard error of
# this mean, the standard deviation of the sample energies divided by
# $\sqrt{256}$. This is the uncertainty of one estimate mentioned under
# Optimize the Angles. `problem.evaluate` then checks each sample against the
# original problem and reports whether it is feasible and what its cut size is.
# The cell compares the share of feasible samples with the share among
# uniformly random bitstrings. These are what the circuit returns when every
# angle is zero, so the comparison shows what the layers add.

# %%
optimized_run = compiled.run(
    on=endpoint,
    gammas=tuple(float(value) for value in optimization.x[:LAYERS]),
    betas=tuple(float(value) for value in optimization.x[LAYERS:]),
    shots=FINAL_SHOTS,
    seed=1901,
)
evaluation = problem.evaluate(optimized_run.samples)

# Count the shots whose sample satisfies the constraint.
feasible_shots = sum(
    evaluation.occurrences[sample_id]
    for sample_id, feasible in evaluation.feasible.items()
    if feasible
)

# `cuts` from the exact solution holds every feasible assignment.
uniform_feasible = len(cuts) / 2**NUM_NODES

# Standard error of the mean energy: the standard deviation of the sample
# energies divided by the square root of the number of shots.
sample_energies = np.repeat(
    [sample.energy for sample in optimized_run.samples.samples],
    [sample.occurrences for sample in optimized_run.samples.samples],
)
standard_error = sample_energies.std() / np.sqrt(FINAL_SHOTS)

print(f"Mean energy of the final batch: {optimized_run.samples.mean_energy:.4f}")
print(f"Standard error of this mean: {standard_error:.4f}")
print(
    f"Feasible samples: {feasible_shots} / {FINAL_SHOTS} "
    f"({100.0 * feasible_shots / FINAL_SHOTS:.1f}%)"
)
print(f"Feasible share of uniformly random bitstrings: {100.0 * uniform_feasible:.1f}%")

assert optimized_run.execution.artifact is compiled.artifact
assert optimized_run.samples.total_occurrences == FINAL_SHOTS
assert evaluation.total_occurrences == FINAL_SHOTS
if not docs_test_mode:
    assert feasible_shots / FINAL_SHOTS > uniform_feasible

# %% [markdown]
# The share of feasible samples is far above that of uniformly random bitstrings,
# so the optimized circuit concentrates on equal-size partitions.

# %% [markdown]
# ### Best Feasible Solution
#
# `evaluation.best_feasible` returns the feasible sample with the smallest cut
# size. We read off the group of each vertex, count the crossing edges again
# from the edge list, and compare the result with the exact solution.

# %%
best = (
    evaluation.get(evaluation.sample_ids[0])
    if docs_test_mode
    else evaluation.best_feasible
)

# Read off the group of each vertex.
values = best.decision_variables_df["value"]
best_assignment = {
    index: int(round(float(values.loc[index]))) for index in range(NUM_NODES)
}
best_objective = int(round(best.objective))

# Count the crossing edges again from the edge list.
cut_edges = sum(
    best_assignment[left] != best_assignment[right] for left, right in EDGE_LIST
)

print("Best feasible partition:", best_assignment)
print(f"Cut size: {best_objective}")
print(f"Exact smallest cut size: {best_feasible_cut}")

assert best_objective == cut_edges
if not docs_test_mode:
    assert sum(best_assignment.values()) == NUM_NODES // 2
    assert best_objective == best_feasible_cut

# %% [markdown]
# The best sample has the exact smallest cut size. This alone does not show
# that the circuit favors good partitions, because random guessing would
# usually find the optimum too. Only 2 of the 256 bitstrings are optimal, but
# 256 uniformly random bitstrings would contain at least one of them with a
# probability of $1 - (254/256)^{256} \approx 87\%$, since each draw misses
# both with probability $254/256$. The next section therefore looks at how
# often each cut size is sampled.

# %% [markdown]
# ### Distribution of Cut Sizes
#
# The evaluation stores each distinct sample once, together with how often it
# occurred. We add up those counts for each cut size among the feasible
# samples, and compare them with what uniformly random bitstrings would give in
# the same number of shots.

# %%
# Count the feasible samples for each cut size.
objective_counts = {}
for sample_id in evaluation.sample_ids:
    if not evaluation.feasible[sample_id]:
        continue
    objective = int(round(evaluation.objectives[sample_id]))
    objective_counts[objective] = (
        objective_counts.get(objective, 0) + evaluation.occurrences[sample_id]
    )

assert sum(objective_counts.values()) == feasible_shots

# Expected counts for uniformly random bitstrings. `cuts` from the exact solution
# holds the cut size of every feasible assignment.
uniform_counts = {}
for cut in cuts.values():
    uniform_counts[cut] = uniform_counts.get(cut, 0) + FINAL_SHOTS / 2**NUM_NODES

cut_values = sorted(uniform_counts)
positions_on_axis = np.arange(len(cut_values))
bar_width = 0.4

plt.figure(figsize=(8, 4))
plt.bar(
    positions_on_axis - bar_width / 2,
    [objective_counts.get(value, 0) for value in cut_values],
    width=bar_width,
    color="#2696EB",
    label="QAOA samples",
)
plt.bar(
    positions_on_axis + bar_width / 2,
    [uniform_counts[value] for value in cut_values],
    width=bar_width,
    color="#eb6834",
    label="Uniformly random bitstrings (expected)",
)
plt.xticks(positions_on_axis, [str(value) for value in cut_values])
plt.xlabel("Cut size")
plt.ylabel("Number of feasible samples")
plt.title("Distribution of feasible solutions")
plt.legend()
plt.show()

print("Cut size: QAOA samples, uniformly random (expected)")
for value in cut_values:
    print(f"{value}: {objective_counts.get(value, 0)}, {uniform_counts[value]:.1f}")

if not docs_test_mode:
    assert (
        objective_counts.get(best_feasible_cut, 0) > uniform_counts[best_feasible_cut]
    )

# %% [markdown]
# QAOA moves the samples towards small cut sizes. The optimal cut size 6
# appears in 24 of the 256 samples, against 2 expected for uniformly random
# bitstrings. This comparison, not the single best sample, shows that the
# optimized circuit favors good partitions. The cut size 8 is still the most
# frequent. It belongs to 16 feasible assignments, against 2 with the cut size
# 6, and that outweighs the higher probability of each optimal assignment: 12
# samples each on average, against 5. So the answer has to be picked out of the
# samples.

# %% [markdown]
# ### Visualize the Best Partition
#
# The colors show the two groups of the best feasible sample. The 6 crossing
# edges are the ones that join vertices of different colors.

# %%
# Color each vertex by its group.
colors = [
    "#FF6B6B" if best_assignment[index] == 1 else "#4ECDC4"
    for index in range(NUM_NODES)
]
plt.figure(figsize=(5, 5))
nx.draw(
    graph,
    positions,
    with_labels=True,
    node_color=colors,
    node_size=700,
    edgecolors="black",
)
plt.title(f"Best feasible partition (cut size = {best_objective})")
plt.show()

# %% [markdown]
# ## Summary
#
# In this tutorial we:
#
# 1. Wrote a graph partitioning problem with JijModeling, and found its exact
#    solution, with cut size 6, by trying every assignment.
# 2. Turned the equal-size constraint into a quadratic penalty with
#    `OMMXProblem`, and checked by brute force that the weight $\lambda = 4$ is
#    large enough.
# 3. Built and compiled the QAOA circuit with `QAOAExperiment`. The penalty
#    couples every pair of vertices, so each of the 5 layers has 28 $R_{ZZ}$
#    rotations, although the graph has only 16 edges.
# 4. Tuned the 10 angles with COBYLA, starting from a linear ramp. Every
#    evaluation ran the compiled circuit at a new set of angles and returned
#    the mean energy of its samples.
# 5. Sampled the circuit at the returned angles and evaluated every sample on
#    the original problem. Compared with uniformly random bitstrings, the
#    feasible share rose from 27.3% to 92.2%, and the optimal cut size appeared
#    in 24 of 256 samples instead of 2.
#
# The penalty weight $\lambda$ and the time step $\Delta t$ of the ramp were
# chosen for this graph, and both may need to change with the problem.
#
# To build a QAOA circuit gate by gate, see [QAOA for MaxCut](qaoa_maxcut).
