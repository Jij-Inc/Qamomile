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
# ---

# %% [markdown]
# ---
# tags: [algorithm, optimization, oracle-based]
# ---
#
# # QSVT Filtering to solve Graph Partitioning
#
# This tutorial explains how to use Quantum Singular Value Transform (QSVT) 
# and chemistry inspired methods to find the ground energy of a given Hamiltonian
# and solve the Graph Partitioning problem.
#
# > L. Lin and Y. Tong, *Near-optimal ground state preparation*, Quantum 4, 361 (2020) [arXiv:2002.12508](https://arxiv.org/abs/2002.12508)
#
# Lin & Tong's ground-energy algorithm finds the lowest eigenvalue of a cost
# Hamiltonian by *filtering*. For a given threshold parameter $\mu$, we use
# Linear Combination of Unitaries (LCU) to build a block encoding of $H - \mu I$.
# A QSVT approximation of the sign function is used to build a projector 
# onto the eigenspace on one side of the threshold parameter $\mu$. The fraction of
# shots surviving post-selection tells a classical binary search which way to move $\mu$.
#
# This notebook runs that whole loop on a graph-partitioning instance, and 
# demonstrates how to use the Qamomile library through the `QSVTFilterConverter`: 
# build the block encoding, probe one threshold, then let
# the binary search locate the ground energy on its own.

# %%
# Install the latest Qamomile through pip!
# # # !pip install qamomile

# %%
import jijmodeling as jm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import qamomile.circuit as qmc
from qamomile.circuit.algorithm.qsvt_filter import eigenstate_filter_projector
from qamomile.optimization.qsvt_filter import QSVTFilterConverter
from qamomile.qiskit import QiskitTranspiler

# %% [markdown]
# ## Problem Settings
#
# Graph partitioning splits the vertices into two equal halves while cutting as
# few edges as possible. We minimise the number of cut edges subject to an
# equal-size constraint, which `to_hubo` absorbs into the objective as a penalty.
#
# (see the QAOA tutorial for more details about Graph Partitioning itself)

# %%
problem = jm.Problem("Graph Partitioning")


@problem.update
def _(problem: jm.DecoratedProblem):
    V = problem.Dim()
    E = problem.Natural(ndim=2)  # edge list: [[u1,v1], [u2,v2], ...]
    x = problem.BinaryVar(shape=(V,))

    # Objective: minimize edges cut between partitions
    problem += (
        E.rows().map(lambda e: x[e[0]] * (1 - x[e[1]]) + x[e[1]] * (1 - x[e[0]])).sum()
    )

    # Constraint: equal partition sizes
    problem += problem.Constraint("Equal Partition", x.sum() == V / 2)


problem

# %%
num_nodes = 6
# Triangular prism: the triangles {0, 3, 4} and {1, 2, 5} joined by a perfect
# matching (0-5, 1-4, 2-3). Separating the two triangles cuts only the three
# matching edges; every other equal split breaks a triangle and cuts more, so
# the optimal partition is unique.
edge_list = [
    [0, 3],
    [0, 4],
    [0, 5],
    [1, 2],
    [1, 4],
    [1, 5],
    [2, 3],
    [2, 5],
    [3, 4],
]

G = nx.Graph()
G.add_nodes_from(range(num_nodes))
G.add_edges_from(edge_list)
assert G.number_of_nodes() == num_nodes
assert G.number_of_edges() == len(edge_list)

pos = nx.spring_layout(G, seed=1)
plt.figure(figsize=(5, 5))
nx.draw(G, pos, with_labels=True, node_color="white", node_size=700, edgecolors="black")
plt.title(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
plt.show()

# %%
instance_data = {"V": num_nodes, "E": edge_list}
instance = problem.eval(instance_data)

# %% [markdown]
# ## Algorithm
#
# The converter block encodes the Ising cost Hamiltonian $H$, composes a second
# LCU for $H - \mu I$, and applies a QSVT polynomial $p \approx \mathrm{sign}$ to
# it. One Hadamard-test qubit turns the resulting reflection into a projector
# $P_\mu$. Starting from the uniform superposition
# $|\varphi_0\rangle = H^{\otimes n}|0\rangle$, the fraction of shots with every
# ancilla measuring zero estimates
#
# $$P(\mu)=\lVert P_\mu |\varphi_0\rangle \rVert^2 = \frac{\#\{i : E_i < \mu\}}{2^n},$$
#
# the fraction of the spectrum below $\mu$. That single number is the predicate a
# classical binary search needs: it is bounded away from zero exactly when an
# eigenstate below $\mu$ carries weight in $|\varphi_0\rangle$.
#

# %% [markdown]
# ## Using the `QSVTFilterConverter()`
#
# To use the `QSVTFilterConverter()`, it is as simple as calling it on the problem instance.
# However, a lot of internal parameter a decisive on the solution quality. We explain each of 
# them in the tutorial, as well as how to chose them cleverly to improve the solution quality.
#
# The constant term of the Ising Hamiltonian is held out of the block encoding,
# indeed the methods performances are tied to the normalization parameter ($\alpha$)
# and the constant term makes it grow. On the other hand, a constant offset shifts 
# every eigenvalue equally without perturbating the solutions. 
# `normalization` ($\alpha$) and `energy_offset` ($c$) together bound the spectrum, and that
# interval is what a threshold gets chosen against.

# %%
converter = QSVTFilterConverter(instance)
transpiler = QiskitTranspiler()

low = converter.energy_offset - converter.normalization
high = converter.energy_offset + converter.normalization
print(f"alpha={converter.normalization}  offset={converter.energy_offset}")
print(f"every eigenvalue lies in [{low}, {high}]")

# Reference spectrum: the Ising operator is diagonal, so we can read the exact
# energies off directly and check the search against them later.
# !!! This is only a check and the computation takes exponential time in general.
spin = converter.spin_model
energies = np.full(1 << spin.num_bits, float(spin.constant))
for state in range(energies.size):
    for word, coeff in spin.coefficients.items():
        parity = sum((state >> i) & 1 for i in word) & 1
        energies[state] += coeff * (-1.0 if parity else 1.0)

ground_energy = float(energies.min())
degeneracy = int((energies == ground_energy).sum())
print(f"levels={np.unique(energies)}  ground={ground_energy} ({degeneracy}-fold)")

assert low <= energies.min() and energies.max() <= high

# %% [markdown]
# ### One threshold
#
# `transpile` bakes $\mu$ into the circuit, so each threshold needs its own
# compilation. The filter keeps the eigenspace *below* $\mu$ by default, which is
# what a search for the minimum eigenvalue (ground energy) wants.
#
# `degree` and `delta` are paired: `delta` sets the width of the polynomial's
# transition and `degree`, is the degree of the Tchebychev expansion, 
# and must be large enough to represent it. 
#
# ::: if not chosen, `degree` and `delta` are set to a default value by the converter.
#
# The resolution is given by 
# $$
# \Delta = \frac{\min_i |E_i-\mu|}{\alpha + |\mu - c|}
# $$
# where $E_i$ are the eigenvalues, and the other parameter were defined previously.
# As often, knowing the resolution will require knowing the ground energy in advance; in 
# practice we need to "guess" or estimate a bound to ensure beating the resolution.
# The `degree` roughly scales as $O(1/\Delta)$.
#
# Below we demonstrates the algorithm for $\mu=4$. The resolution to beat is then $\Delta = \frac{|4-3|}{3 + |4 - 6|} = 1/5$, a choice of `delta=8` matches the transition to the gap and `degree=21` represents it. A bad 
# pairing is not a silent loss of accuracy, it will be rejected outright by the $|p| \le 1$ check.

# %%
DEGREE, DELTA = 21, 8


def probe(mu, shots=2000):
    """Estimate the fraction of the spectrum below `mu`."""
    executable = converter.transpile(
        transpiler,
        mu=mu,
        degree=DEGREE,
        delta=DELTA,
    )
    result = executable.sample(transpiler.executor(), shots=shots).result()
    return converter.success_probability(result)


# Two of the 64 states sit at the ground energy, so the predicate saturates at
# 2 / 64 once mu clears it.
print(f"p(mu=4.0) = {probe(4.0):.4f}   (exact: {(energies < 4.0).mean():.4f})")

# %% [markdown]
# An external library `pyqsp` is used for computing the phase factors 
# corresponding to the polynomial approximation defined by our parameters.
#
# The converter build the quantum circuit of the projector. The circuit is
# ran and we compute the success probability based on the samples count 
# and using the `success_probability` method.
# (exact computation is used as a reference to check the precision) 

# %% [markdown]
# ## Building blocks
#
# `transpile` composes three circuits, each wrapping the previous one. Drawing
# them in that order shows how a block encoding becomes a projector.
#
# To keep the pictures readable the phase vector is shortened to four entries (instead of the 22 used above).

# %%
DRAW_MU = 4.0

# `_shifted_encoding` is internal: it is not part of the converter's public API,
# but it is the descriptor every circuit below is built from, so the drawings
# match the circuit that actually runs at this threshold.
shifted = converter._shifted_encoding(DRAW_MU)
n_signal = shifted.num_signal_qubits
n_system = shifted.num_system_qubits

# Four phases instead of the 22 the degree-21 filter uses; only the count
# affects the circuit's shape, not the values.
phi_demo = [0.1, 0.2, 0.3, 0.4]

print(f"alpha' = {shifted.normalization}   signal={n_signal}  system={n_system}")


# %% [markdown]
# ### 1. The block encoding $U$
#
# The Ising Hamiltonian is written as a linear combination of unitaries and
# realised as $\operatorname{PREPARE}$, $\operatorname{SELECT}$, $\operatorname{PREPARE}^\dagger$. $\operatorname{PREPARE}$ loads the coefficient
# amplitudes onto the signal register, $\operatorname{SELECT}$ applies the corresponding Pauli-$Z$
# word to the system conditioned on the signal, and $\operatorname{PREPARE}^\dagger$ un-computes.
#
# The result is a unitary whose top-left block, the part selected by the signal
# register measuring all zero, is $(H - \mu I)/\alpha'$. Nothing outside that
# block is useful, which is why every shot must be post-selected on it.

# %%
@qmc.qkernel
def block_encoding() -> qmc.Vector[qmc.Bit]:
    """Apply the block encoding of (H - mu*I) / alpha' to |0>."""
    signal = qmc.qubit_array(n_signal, "signal")
    system = qmc.qubit_array(n_system, "system")
    signal, system = shifted.unitary(signal, system)
    return qmc.measure(signal)


block_encoding.draw(inline=True, inline_depth=1, fold_loops=False)


# %% [markdown]
# ### 2. The QSVT alternation
#
# Given the block encoding, `qmc.qsvt` applies
#
# $$R(\phi_0),\; U,\; R(\phi_1),\; U^\dagger,\; R(\phi_2),\; U,\; \dots$$
#
# with $U$ the block encoding built above, $R(\phi) = e^{i\phi(2\Pi - I)}$, where $\Pi$ projects the signal register
# onto all zero. The primitive owns the whole sequence: it allocates its own
# rotation auxiliary qubit, alternates $U$ with $U^\dagger$, and applies the
# phases.
#
# The effect is to replace the encoded operator by a polynomial of it. With the
# phases `pyqsp` produced, that polynomial approximates
# $\mathrm{sign}$, so the good block becomes a *reflection* about the eigenspace
# on one side of $\mu$.

# %%
@qmc.qkernel
def qsvt_reflector(
    phi: qmc.Vector[qmc.Float],
) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
    """QSVT alternation implementing sign((H - mu*I) / alpha')."""
    signal = qmc.qubit_array(n_signal, "signal")
    system = qmc.qubit_array(n_system, "system")
    signal, system = qmc.qsvt(signal, system, phi, shifted)
    return qmc.measure(signal), qmc.measure(system)


qsvt_reflector.draw(phi=phi_demo, inline=True, inline_depth=0, fold_loops=False)

# %% [markdown]
# ### 3. The Lin & Tong projector
#
# The goal is to turn the reflection into projection $P_\mu = (I \pm R)/2$. 
# However, this projector is **not unitary**, so no circuit can simply apply it.
#
# The trick is that it is a sum of two unitaries, $I$ and $R$, so a Hadamard test
# can realise it probabilistically. One probe qubit is put in superposition, the
# QSVT reflection is applied *under its control*, and a second Hadamard
# interferes the two branches. Measuring the probe qubit as zero heralds the
# branch where $P_\mu$ was applied; that is why the algorithm has a success
# probability at all, and why the post-selection is on the probe **and** the
# signal register together.

# %%
eigenstate_filter_projector(shifted).draw(
    proj=1,
    signal=n_signal,
    system=n_system,
    phi=phi_demo,
    inline=True,
    inline_depth=0,
    fold_loops=False,
)


# %% [markdown]
# ## Binary search
#
# We want to use the above algorithm to find a tight bound the problem solution, i.e. the ground state of the Hamiltonian. To do so, we find the correct value of the threshold parameter $\mu$ using the same Binary Search Algorithm as in Li & Tong's paper.
#
# We previously compute the range in which the eigenvalues live $[c-\alpha,c+\alpha)$. For a given threshold $\mu$, we can compare it to the ground energy $\lambda_0$. If $\mu$ lies in $[c-\alpha, \lambda_0)$, all the eigenvalues are filtered out and $P(\mu)=0$. On the intervalle $[\lambda_0,\lambda_1)$, only the ground energy survives. Because our Hamiltonian is diagonal, this probability is given by $\gamma^2 = d/2^n$ where $n$ is the space size and $d$ the degeneracy of the ground state. On the intervalle $[\lambda_1,c+\alpha)$, the probability $P(\mu)$ is always greater than $\gamma^2$.
#
# The parameter $\gamma$ is the key for the search to suceed, yet, $\gamma$ is unknown to us and need to be approximated. 
#
# Once $\gamma$ is settled, we divide the search space $[c-\alpha,c+\alpha)$ into an equally spaced grid. Let $G$ be the range of the space and $h$ is the discretization parameter, input of the algorithm, then the grid contains $\lfloor G/h \rfloor$ possibilities for $\mu$. We explore those possibilities through a classical binary search. The criteria to search above or below the current threshold is whether $P(\mu)$, estimate by our quantum algorithm, is greater or lower than $\tau = \gamma^2/2$.
#
# $h$ also needs to be carefully chosen as it should be smaller than the (unknown) spectral gap.
#
# Finally notice that the approximation error can make the function non strictly decreasing. To overcome this issue, we evaluate a bracket of two consecutive threshold in the grid and return a single bit for each $(B_k,B_{k+1})$ flagging if $P(\mu) > \tau$, and we move according to these bracket.

# %%
def binary_search_ground_energy(success_prob, low, high, gamma, h=1.0):
    """Algorithm 1 of Lin & Tong: paired (B_k, B_k+1) probes.

    Grid x_k = low + (k + 0.5) * h: the half-cell offset keeps thresholds
    between candidate energies, since the strict comparison in the predicate
    would read zero on a threshold sitting exactly on an eigenvalue. B_k = 1
    iff the estimated probability exceeds tau = gamma**2 / 2.

    The paired-probe loop narrows to at most four cells; a linear scan over
    those finishes the job, so the returned bracket is one cell wide.
    Returns (x_L, x_U) with x_L <= lambda_0 <= x_U -- closed on the left so a
    ground energy sitting exactly on `low` is still bracketed.
    """
    n_grid = int(round((high - low) / h))
    grid = [low + (k + 0.5) * h for k in range(n_grid)]
    tau = 0.5 * gamma**2
    cache = {}

    def B(k):
        # Boundary sentinels: B_-1 = 0 (nothing below), B_G = 1 (all below).
        if not 0 <= k < n_grid:
            return 0 if k < 0 else 1
        if k not in cache:
            cache[k] = success_prob(grid[k])
        return int(cache[k] > tau)

    L, U = 0, n_grid - 1
    print(f"grid {grid[0]:+.2f}..{grid[-1]:+.2f}  G={n_grid}  h={h}  tau={tau:.4f}")

    while U - L > 3:
        k = (L + U) // 2
        bk, bk1 = B(k), B(k + 1)
        print(f"  k={k}: B({grid[k]:+.2f})={bk}  B({grid[k + 1]:+.2f})={bk1}", end="  ->  ")
        if bk == 1 and bk1 == 1:  # lambda_0 < x_k+1
            U = k + 1
            print(f"(1,1) U<-{U}  [{grid[L]:+.2f}, {grid[U]:+.2f}]")
        elif bk == 0 and bk1 == 0:  # lambda_0 > x_k
            L = k
            print(f"(0,0) L<-{L}  [{grid[L]:+.2f}, {grid[U]:+.2f}]")
        elif bk == 0 and bk1 == 1:  # x_k-1 < lambda_0 < x_k+2
            L, U = max(k - 1, 0), min(k + 2, n_grid - 1)
            print(f"(0,1) -> [{grid[L]:+.2f}, {grid[U]:+.2f}]")
            break
        else:  # x_k < lambda_0 < x_k+1
            L, U = k, k + 1
            print(f"(1,0) -> [{grid[L]:+.2f}, {grid[U]:+.2f}]")
            break

    # Tighten to a single cell: the first grid point whose predicate fires is
    # the smallest threshold that still sees the ground eigenspace. Clamping to
    # the spectrum bound keeps the bracket physical -- no probe is needed below
    # `low`, because no eigenvalue can be there.
    for k in range(L, U + 1):
        if B(k):
            print(f"  refine: first B=1 at k={k} ({grid[k]:+.2f})")
            return max(grid[k] - h, low), grid[k]
    return grid[U], min(grid[U] + h, high)


# %%
gamma = np.sqrt(degeneracy / energies.size)
x_lower, x_upper = binary_search_ground_energy(probe, low, high, gamma=gamma, h=1.0)

print(f"\nlambda_0 in [{x_lower:+.2f}, {x_upper:+.2f}]   true lambda_0 = {ground_energy:+.2f}")

# %% [markdown]
# ## Optional
#
# Finally, we can call the quantum algorithm with the final threshold parameter found by the binary search to prepare the ground state with high accuracy. From the ground states measurement sample, we extract the optimal solution. 
#
# ::: Note : Deciding whether $P(\mu)>\tau$ in the binary search tolerates a blury filter and that's why `degree=21` was enough for every probe. To keep *only* the ground eigenspace, at $\mu = 3.5$ the optimal resolution is $\Delta = 0.09$ rather than the $0.20$ used by the search, so the final decode exploits a sharper polynomial. Raising `degree` alone does not help: `degree=61` with a mismatched `delta=20` scores *worse* (85%) than `degree=41` with `delta=11` (91%).

# %%
# The predicate needed degree 21; extracting the state itself needs a sharper
# filter at the same threshold.
DECODE_DEGREE, DECODE_DELTA = 41, 11

executable = converter.transpile(
    transpiler,
    mu=x_upper,
    degree=DECODE_DEGREE,
    delta=DECODE_DELTA,
)
result = executable.sample(transpiler.executor(), shots=20000).result()
sampleset = converter.decode_to_binary_sampleset(result)

kept = sum(sampleset.num_occurrences)
optimal = sum(
    n
    for e, n in zip(sampleset.energy, sampleset.num_occurrences)
    if np.isclose(e, ground_energy)
)
print(f"kept {kept} of {result.shots} shots, {optimal / kept:.1%} at the ground energy")

best = max(zip(sampleset.samples, sampleset.num_occurrences), key=lambda z: z[1])[0]
side = sorted(v for v, bit in best.items() if bit == 1)
cut = sum(1 for u, v in edge_list if best[u] != best[v])
print(f"most sampled partition: {side} | cut={cut}")

assert optimal / kept > 0.85
assert cut == 3 and len(side) == num_nodes // 2

# %% [markdown]
# From the final sample distribution `sampleset`, we recover the most sampled partition that we use as the solution to the partition problem. The partition is $[0,3,4]$ and the coresponding objective (minimal cut) is $3$. 

# %% [markdown]
# ## Summary
#
# In this notebook, we:
#
# - Solved a graph-partitioning instance end to end with `QSVTFilterConverter`,
#   going from a JijModeling problem to the optimal partition of the triangular
#   prism, the two triangles, cutting only the three matching edges.
# - Opened up the three nested circuits the converter composes: the LCU block
#   encoding of $H - \mu I$, the `qmc.qsvt` alternation that turns it into a
#   reflection about the eigenspace on one side of $\mu$, and the Hadamard test
#   that turns that reflection into a projector. A projector is not unitary,
#   which is exactly why the algorithm has a *success probability* and why every
#   shot has to be post-selected on the probe and signal registers together.
# - Used that post-selection rate as the Lin & Tong predicate and let a classical
#   binary search drive $\mu$, bracketing the ground energy in three probes
#   without ever being told where it was. Its threshold $\tau = \gamma^2/2$ rests
#   on an assumed lower bound $\gamma$ on the initial overlap, an input to the
#   algorithm, not a universal constant, and the notebook's ground level would be
#   missed entirely by a generic choice.
# - Saw that every knob traces back to one number, the normalized gap
#   $\Delta = \min_i |E_i - \mu| / \alpha'$: `delta` must match it, `degree` must
#   be large enough to represent `delta`, the grid spacing `h` cannot shrink below
#   it, and extracting the state demands more of it than answering the predicate
#   does. Holding the Ising constant out of the block encoding is what buys
#   $\Delta$ back, which is why `normalization` and `energy_offset` are reported
#   as a pair, and why they, not the eigenvalues, are what a threshold is chosen
#   against.
