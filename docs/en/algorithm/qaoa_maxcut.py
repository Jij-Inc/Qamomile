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
# ---
# tags: [algorithm, optimization, variational]
# ---
#
# # QAOA for MaxCut: Building the Circuit from Scratch
#
# This tutorial walks through the Quantum Approximate Optimization Algorithm
# (QAOA) pipeline step by step, using Qamomile's low-level circuit primitives.
# Rather than using the high-level `QAOAConverter`, we will:
#
# 1. Define a MaxCut problem for a small graph.
# 2. Formulate it directly as an Ising model on spin variables.
# 3. Write the QAOA circuit step by step using `@qkernel`.
# 4. Optimize variational parameters with a classical optimizer.
# 5. Decode and visualize the results.
#
# At the end, we show that `qamomile.stdlib.qaoa_state` provides
# the same circuit in a single function call.

# %% [markdown]
# ## Installation
#
# ```bash
# pip install qamomile
# ```

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
from __future__ import annotations

import itertools
import math
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import minimize

import qamomile as qm
from qamomile.optimization import BinaryModel, VarType
from qamomile.stdlib import ising_cost, qaoa_state, uniform_superposition, x_mixer

# %% [markdown]
# ## What is MaxCut?
#
# Given an undirected graph $G = (V, E)$, the **MaxCut** problem asks us to
# partition the vertices into two sets so that the number of edges crossing
# between the two sets is maximized.
#
# MaxCut is naturally a **spin** problem. Assign each vertex $i$ a spin
# $s_i \in \{+1, -1\}$ indicating which side of the cut it belongs to.
# An edge $(i, j)$ is *cut* exactly when $s_i \ne s_j$, so the number of
# cut edges is
#
# $$
# \text{MaxCut}(\boldsymbol{s})
# = \sum_{(i,j) \in E} \frac{1 - s_i s_j}{2}.
# $$
#
# Spin-based problems such as MaxCut, spin-glass ground states, and Ising
# model benchmarks are most cleanly written in the spin domain. We therefore
# skip the QUBO / binary encoding detour and work directly with spin variables
# throughout this tutorial.

# %% [markdown]
# ## Create the Graph
#
# We use a small 5-node graph with 6 edges. This is large enough to be
# non-trivial, yet small enough to brute-force for comparison.

# %%
EDGES = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4))
graph = nx.Graph()
graph.add_edges_from(EDGES)
num_nodes = graph.number_of_nodes()

assert num_nodes == 5
assert graph.number_of_edges() == 6

positions = nx.spring_layout(graph, seed=42)
plt.figure(figsize=(5, 4))
nx.draw(
    graph,
    positions,
    with_labels=True,
    node_color="white",
    node_size=700,
    edgecolors="black",
)
plt.title(f"Graph: {num_nodes} nodes, {graph.number_of_edges()} edges")
plt.show()

# %% [markdown]
# ## Ising Formulation
#
# Maximizing the cut count is equivalent, up to a constant, to minimizing the
# antiferromagnetic Ising Hamiltonian
#
# $$
# H_C(\mathbf{s})=\sum_{(i,j)\in E}s_i s_j.
# $$
#
# Compared with the general Ising form
# $H=\sum_i h_i s_i+\sum_{i<j}J_{ij}s_i s_j$, unweighted MaxCut has no linear
# terms and one uniform coupling for every edge. `BinaryModel.from_ising`
# accepts those coefficients directly; there is no need to go through QUBO.

# %%
model = BinaryModel.from_ising(
    {edge: 1.0 for edge in EDGES},
    variables=range(num_nodes),
)

assert model.vartype is VarType.SPIN
assert model.num_variables == 5
assert model.order == 2
assert BinaryModel.from_bytes(model.to_bytes()) == model

# %% [markdown]
# `BinaryModel` also provides `from_qubo()` and `from_hubo()` for problems that
# are naturally expressed in the binary domain, such as assignment problems
# and constrained problems with penalty terms.

# %% [markdown]
# ## Exact Solution (Brute Force)
#
# Before running QAOA, let's find the optimal partition by trying all
# $2^n = 32$ spin configurations. This gives us a ground truth to compare
# against.

# %%
best_cut = 0
optimal_partitions = []
for spins in itertools.product((1, -1), repeat=num_nodes):
    cut = sum(spins[left] != spins[right] for left, right in EDGES)
    if cut > best_cut:
        best_cut = cut
        optimal_partitions = [spins]
    elif cut == best_cut:
        optimal_partitions.append(spins)

print(f"Optimal MaxCut value: {best_cut}")
print(f"Number of optimal partitions: {len(optimal_partitions)}")

assert best_cut == 5
assert len(optimal_partitions) == 2
assert all(
    tuple(-spin for spin in item) in optimal_partitions for item in optimal_partitions
)

# %% [markdown]
# `model.hamiltonian` is the same objective represented as an immutable,
# backend-neutral sum of commuting Pauli-Z words. Rust derives it from the
# canonical polynomial, so the source page does not need a second coefficient
# conversion.

# %%
cost_hamiltonian = model.hamiltonian

assert cost_hamiltonian.num_qubits == 5
assert cost_hamiltonian.is_pairwise_commuting
assert min(model.energy(item) for item in optimal_partitions) == -4.0

# %% [markdown]
# ## QAOA Circuit: The Idea
#
# The QAOA ansatz prepares a parameterized quantum state:
#
# $$
# |\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle
# = \prod_{l=1}^{p}
#   e^{-i \beta_l H_M} \, e^{-i \gamma_l H_C}
#   \; |{+}\rangle^{\otimes n}
# $$
#
# where:
# - $|{+}\rangle^{\otimes n}$: uniform superposition (Hadamard on every qubit)
# - $e^{-i \gamma H_C}$: **cost unitary** — for the Ising cost $H_C$, this
#   decomposes into $\text{RZZ}$ gates for quadratic terms and $\text{RZ}$
#   gates for linear terms (see Steps 2–3 for the gate convention details)
# - $e^{-i \beta H_M}$: **mixer unitary** — with $H_M = \sum_i X_i$, this
#   becomes $\text{RX}(2\beta)$ on every qubit
# - $p$: number of layers (depth of the ansatz)
#
# The spin $\leftrightarrow$ computational-basis correspondence is the
# standard quantum convention $Z|0\rangle = |0\rangle$,
# $Z|1\rangle = -|1\rangle$, so measurement outcome $0$ maps to spin $+1$
# and outcome $1$ maps to spin $-1$.
#
# We will now build each component as a `@qkernel`.
#
# ### Step 1: Uniform Superposition
#
# Apply a Hadamard gate to every qubit to start from the equal
# superposition state $|{+}\rangle^{\otimes n}$.
#
# ### Step 2: Cost Layer
#
# Apply the cost unitary $e^{-i \gamma H_C}$.
#
# Qamomile's rotation gates include a $1/2$ factor:
# $\text{RZ}(\theta) = e^{-i \theta Z / 2}$ and
# $\text{RZZ}(\theta) = e^{-i \theta Z \otimes Z / 2}$.
# To match $e^{-i \gamma H_C}$ exactly, we therefore pass
# $2 J_{ij} \gamma$ to `rzz` and $2 h_i \gamma$ to `rz`.
#
# We keep the `linear` argument even though it is empty for unweighted
# MaxCut — this makes the kernel immediately reusable for weighted MaxCut
# and generic spin-glass Hamiltonians, which do include linear $h_i$ terms.
#
# ### Step 3: Mixer Layer
#
# Apply the mixer unitary $e^{-i \beta H_M}$ where $H_M = \sum_i X_i$.
# Since $\text{RX}(\theta) = e^{-i \theta X / 2}$, we need $\theta = 2\beta$
# to implement $e^{-i \beta X_i}$ on each qubit.
#
# ### Step 4: Full QAOA Ansatz
#
# Compose the three pieces: superposition, then $p$ rounds of
# cost + mixer, and finally measurement.
#


# %%
@qm.qkernel
def maxcut_qaoa(
    hamiltonian: qm.Const[qm.Observable],  # type: ignore[valid-type]
    layers: qm.Const[qm.Int],
    gammas: qm.Param[qm.Vector[qm.Real, layers]],  # type: ignore[valid-type]  # noqa: F821
    betas: qm.Param[qm.Vector[qm.Real, layers]],  # type: ignore[valid-type]  # noqa: F821
) -> qm.Shot[qm.Bits[5]]:  # type: ignore[valid-type]
    """Sample a typed QAOA state for the five-node MaxCut graph.

    Args:
        hamiltonian (qm.Const[qm.Observable]): Compile-time MaxCut cost.
        layers (qm.Const[qm.Int]): Compile-time number of QAOA layers.
        gammas (qm.Param[qm.Vector[qm.Real, layers]]): Runtime cost angles.
        betas (qm.Param[qm.Vector[qm.Real, layers]]): Runtime mixer angles.

    Returns:
        qm.Shot[qm.Bits[5]]: One logical five-bit observation per shot.
    """

    state = uniform_superposition(qm.qubits(5), 5)
    for layer in range(layers):
        state = ising_cost(state, hamiltonian, gammas[layer])
        state = x_mixer(state, 5, betas[layer])
    return qm.measure(state)


docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
layers = 1 if docs_test_mode else 3
program = maxcut_qaoa.specialize(hamiltonian=cost_hamiltonian, layers=layers)

# %% [markdown]
# The program retains its source signature and records the Hamiltonian as a
# `Const` binding. Compilation later derives an artifact ABI containing only
# the two runtime coordinates, without erasing source identity.

# %%
assert tuple(parameter.name for parameter in program.entry.parameters) == (
    "hamiltonian",
    "layers",
    "gammas",
    "betas",
)
assert tuple(name for name, _ in program.const_bindings) == (
    "hamiltonian",
    "layers",
)
assert "quantum.qaoa" not in program.inspect()

# %% [markdown]
# ## Using the Built-in `qaoa_state`
#
# The explicit construction above makes every QAOA step visible. The standard
# library also provides the same composition for applications that only need
# to supply a width, layer count, cost Hamiltonian, and runtime angles.


# %%
@qm.qkernel
def maxcut_qaoa_builtin(
    hamiltonian: qm.Const[qm.Observable],  # type: ignore[valid-type]
    layers: qm.Const[qm.Int],
    gammas: qm.Param[qm.Vector[qm.Real, layers]],  # type: ignore[valid-type]  # noqa: F821
    betas: qm.Param[qm.Vector[qm.Real, layers]],  # type: ignore[valid-type]  # noqa: F821
) -> qm.Shot[qm.Bits[5]]:  # type: ignore[valid-type]
    """Sample the standard-library form of the same MaxCut QAOA state.

    Args:
        hamiltonian (qm.Const[qm.Observable]): Compile-time MaxCut cost.
        layers (qm.Const[qm.Int]): Compile-time number of QAOA layers.
        gammas (qm.Param[qm.Vector[qm.Real, layers]]): Runtime cost angles.
        betas (qm.Param[qm.Vector[qm.Real, layers]]): Runtime mixer angles.

    Returns:
        qm.Shot[qm.Bits[5]]: One logical five-bit observation per shot.
    """

    return qm.measure(qaoa_state(5, layers, hamiltonian, gammas, betas))


builtin_program = maxcut_qaoa_builtin.specialize(
    hamiltonian=cost_hamiltonian,
    layers=layers,
)
assert (
    qm.resources.logical(builtin_program)
    .metric(qm.resources.LOGICAL_GATES)
    .numeric_enclosure()
    == qm.resources.logical(program)
    .metric(qm.resources.LOGICAL_GATES)
    .numeric_enclosure()
)

# %% [markdown]
# ## Estimate the authored program
#
# Logical resource analysis follows this same body. It sees five Hadamards,
# five mixer rotations and one abstract commuting-Pauli evolution per layer.
# Target lowering may realize that evolution differently, but cannot change
# its source meaning.

# %%
study = qm.resources.logical(program)

assert study.metric(
    qm.resources.logical_operation_count("quantum.h")
).numeric_enclosure() == (5, 5)
assert study.metric(
    qm.resources.logical_operation_count("quantum.rx")
).numeric_enclosure() == (5 * layers, 5 * layers)
assert study.metric(
    qm.resources.logical_operation_count("quantum.pauli_evolution")
).numeric_enclosure() == (layers, layers)
assert qm.resources.ResourceStudy.from_bytes(study.to_bytes()) == study

# %% [markdown]
# ## Transpile and Optimize
#
# Compile the program once while keeping `gammas` and `betas` as runtime
# parameters that the optimizer will tune. At each iteration, the classical
# optimizer samples that same artifact and minimizes its mean Ising energy.
#
# The simulator seed and NumPy generator make the run reproducible. Normal
# notebook execution uses three QAOA layers, 256 shots per objective call, and
# a useful optimization budget. Documentation tests retain the complete
# optimizer loop but reduce those three values so every page remains quick.

# %%
endpoint = qm.Reference()
artifact = qm.compile(program, to=endpoint.target)
assert tuple(slot.name for slot in artifact.parameter_abi.slots) == (
    "gammas",
    "betas",
)
objective_shots = 8 if docs_test_mode else 256
max_iterations = 4 if docs_test_mode else 80
cost_history = []
evaluation_counter = [0]


def cost_function(parameters):
    """Return sampled mean Ising energy for one variational point."""

    gammas = tuple(float(value) for value in parameters[:layers])
    betas = tuple(float(value) for value in parameters[layers:])
    record = qm.run(
        artifact,
        on=endpoint,
        parameters={"gammas": gammas, "betas": betas},
        shots=objective_shots,
        seed=42 + evaluation_counter[0],
    )
    decoded = model.decode(record.result)
    evaluation_counter[0] += 1
    cost_history.append(decoded.mean_energy)
    return decoded.mean_energy


rng = np.random.default_rng(42)
initial_parameters = rng.uniform(-math.pi / 2.0, math.pi / 2.0, 2 * layers)
optimization = minimize(
    cost_function,
    initial_parameters,
    method="COBYLA",
    options={"maxiter": max_iterations},
)

print(f"Optimized cost: {optimization.fun:+.4f}")
print(f"Optimal parameters: {[round(value, 4) for value in optimization.x]}")

assert len(initial_parameters) == 2 * layers
assert len(optimization.x) == 2 * layers
assert len(cost_history) == optimization.nfev

# %%
plt.figure(figsize=(8, 4))
plt.plot(cost_history, color="#2696EB")
plt.xlabel("Iteration")
plt.ylabel("Cost (mean energy)")
plt.title("QAOA Optimization Progress")
plt.show()

# %% [markdown]
# ## Decode and Analyze Results
#
# Sample with the optimized parameters and interpret the outcomes.
# `BinaryModel.decode` maps logical measurement order back to the original
# variable IDs and evaluates every unique assignment in the spin domain, so no
# binary conversion is needed.

# %%
optimized_gammas = tuple(float(value) for value in optimization.x[:layers])
optimized_betas = tuple(float(value) for value in optimization.x[layers:])
final_shots = 128 if docs_test_mode else 2048
final_record = qm.run(
    artifact,
    on=endpoint,
    parameters={"gammas": optimized_gammas, "betas": optimized_betas},
    shots=final_shots,
    seed=42,
)
decoded = model.decode(final_record.result)

cut_distribution = {}
best_qaoa_cut = 0
best_qaoa_assignment = None
for sample in decoded.samples:
    spins = dict(sample.assignment)
    cut = sum(spins[left] != spins[right] for left, right in EDGES)
    cut_distribution[cut] = cut_distribution.get(cut, 0) + sample.occurrences
    if cut > best_qaoa_cut:
        best_qaoa_cut = cut
        best_qaoa_assignment = spins

print(f"Best QAOA cut: {best_qaoa_cut} (optimal: {best_cut})")
print(f"Best partition: {best_qaoa_assignment}")

assert best_qaoa_assignment is not None
assert best_qaoa_cut <= best_cut
assert sum(cut_distribution.values()) == final_shots

# %% [markdown]
# Plot the distribution of observed cut values and the best partition found.

# %%
cuts = sorted(cut_distribution)
counts = [cut_distribution[cut] for cut in cuts]

plt.figure(figsize=(8, 4))
plt.bar([str(cut) for cut in cuts], counts, color="#2696EB")
plt.xlabel("Cut size")
plt.ylabel("Frequency")
plt.title("Distribution of MaxCut Values from QAOA")
plt.show()

# %%
node_colors = [
    "#FF6B6B" if best_qaoa_assignment[node] == 1 else "#4ECDC4"
    for node in range(num_nodes)
]
plt.figure(figsize=(5, 4))
nx.draw(
    graph,
    positions,
    with_labels=True,
    node_color=node_colors,
    node_size=700,
    edgecolors="black",
)
plt.title(f"QAOA partition (cut = {best_qaoa_cut})")
plt.show()

# %% [markdown]
# The source function did not change between inspection, resource estimation,
# compilation, repeated simulation, and decoding. Replacing `Reference` with a
# supported SDK or hardware endpoint changes target realization and transport,
# not the typed QAOA program or the meaning of its result.

# %% [markdown]
# ## Summary
#
# In this tutorial we:
#
# 1. Defined a MaxCut problem and wrote it directly as an Ising Hamiltonian on
#    spin variables—without a QUBO detour.
# 2. Built the spin-domain `BinaryModel` with `BinaryModel.from_ising`.
# 3. Built every component of the QAOA program—superposition, cost layer,
#    mixer layer, and the full layered ansatz.
# 4. Ran a classical optimization loop and decoded and visualized the results.
# 5. Verified that `qamomile.stdlib.qaoa_state` provides the same composition.
#
# The same spin-first recipe applies to weighted MaxCut and spin-glass problems:
# provide the $h_i$ and $J_{ij}$ coefficients to `BinaryModel.from_ising` and
# reuse the program above.
