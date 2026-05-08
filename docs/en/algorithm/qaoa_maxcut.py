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
# At the end, we show that `qamomile.circuit.algorithm.qaoa_state` provides
# the same circuit in a single function call.

# %%
# Install the latest Qamomile through pip!
# (Google Colab) Pick the line that matches your chosen Transpiler tab
# below and remove the leading "# " from it to run.
# # !pip install qamomile                  # Qiskit (default)
# # !pip install "qamomile[quri_parts]"    # QURI Parts
# # !pip install "qamomile[cudaq-cu12]"    # CUDA-Q on a CUDA 12.x toolchain (use cudaq-cu13 on CUDA 13.x). Linux / macOS-arm64 / WSL2 only.

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
import matplotlib.pyplot as plt
import networkx as nx

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
# ## Ising Formulation
#
# Maximizing $\sum_{(i,j) \in E} (1 - s_i s_j) / 2$ is equivalent (up to a
# constant) to *minimizing* the **antiferromagnetic Ising Hamiltonian**
#
# $$
# H_C(\boldsymbol{s}) = \sum_{(i,j) \in E} s_i s_j.
# $$
#
# Compared to the general Ising form
# $H = \sum_i h_i s_i + \sum_{i < j} J_{ij} s_i s_j$, unweighted MaxCut has:
#
# - **no linear terms**: $h_i = 0$ for every vertex, and
# - **uniform couplings**: $J_{ij} = 1$ for every edge $(i, j) \in E$.
#
# `BinaryModel.from_ising` takes the Ising coefficients directly — there is
# no need to go through a QUBO and convert variable types.

# %%
from qamomile.optimization.binary_model import BinaryModel

ising_quad: dict[tuple[int, int], float] = {
    tuple(sorted((i, j))): 1.0 for i, j in G.edges()
}
ising_linear: dict[int, float] = {}

# For weighted MaxCut or spin-glass instances where the J_{ij} are not all
# of the same magnitude, append `.normalize_by_abs_max()` here to keep the
# cost-landscape scale comparable across runs (helps gradient-free
# optimizers such as COBYLA converge consistently).
spin_model = BinaryModel.from_ising(linear=ising_linear, quad=ising_quad)

print(f"Variable type:          {spin_model.vartype}")
print(f"Linear terms (h_i):     {spin_model.linear}")
print(f"Quadratic terms (J_ij): {spin_model.quad}")
print(f"Constant:               {spin_model.constant}")

# %% [markdown]
# > **Note:** `BinaryModel` also provides `from_qubo()` and `from_hubo()` for
# > problems that are naturally expressed in the binary domain (e.g.,
# > assignment problems, constrained problems with penalty terms). See
# > [QAOA for Graph Partitioning](qaoa_graph_partition) for a
# > QUBO / JijModeling-based workflow.

# %% [markdown]
# ## Exact Solution (Brute Force)
#
# Before running QAOA, let's find the optimal partition by trying all
# $2^n = 32$ spin configurations. This gives us a ground truth to compare
# against.

# %%
import itertools

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

# %% [markdown]
# ### Step 1: Uniform Superposition
#
# Apply a Hadamard gate to every qubit to start from the equal
# superposition state $|{+}\rangle^{\otimes n}$.

# %%
import qamomile.circuit as qmc


@qmc.qkernel
def superposition(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(n, name="q")
    for i in qmc.range(n):
        q[i] = qmc.h(q[i])
    return q


# %% [markdown]
# ### Step 2: Cost Layer
#
# Apply the cost unitary $e^{-i \gamma H_C}$.
#
# Qamomile's rotation gates include a $1/2$ factor:
# $\text{RZ}(\theta) = e^{-i \theta Z / 2}$ and
# $\text{RZZ}(\theta) = e^{-i \theta Z \otimes Z / 2}$.
# To match $e^{-i \gamma H_C}$ exactly one would pass $2 J_{ij} \gamma$
# as the angle. However, since $\gamma$ is a **variational parameter**
# that the classical optimizer tunes freely, this constant factor is
# simply absorbed into the optimal $\gamma$ values. We therefore pass
# $J_{ij} \cdot \gamma$ (and $h_i \cdot \gamma$) directly.
#
# We keep the `linear` argument even though it is empty for unweighted
# MaxCut — this makes the kernel immediately reusable for weighted MaxCut
# and generic spin-glass Hamiltonians, which do include linear $h_i$ terms.


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
# ### Step 3: Mixer Layer
#
# Apply the mixer unitary $e^{-i \beta H_M}$ where $H_M = \sum_i X_i$.
# Since $\text{RX}(\theta) = e^{-i \theta X / 2}$, we need $\theta = 2\beta$
# to implement $e^{-i \beta X_i}$ on each qubit.


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
# ### Step 4: Full QAOA Ansatz
#
# Compose the three pieces: superposition, then $p$ rounds of
# cost + mixer, and finally measurement.


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
# ## Transpile and Optimize
#
# We transpile the kernel, binding the problem structure (Ising coefficients,
# number of qubits, number of layers) while keeping `gammas` and `betas`
# as runtime parameters that the optimizer will tune.

# %% [markdown]
# This article uses Qiskit by default. Qamomile transpiles the same
# `@qkernel` to multiple quantum SDKs, so you can follow it with another
# SDK by swapping the import shown below — the rest of the article code
# is identical regardless of the SDK you pick. On Colab, uncomment the
# matching `pip install` line in the cell above first.
#
# ::::{tab-set}
# :::{tab-item} Qiskit
# :sync: qiskit
#
# ```python
# from qamomile.qiskit import QiskitTranspiler
#
# transpiler = QiskitTranspiler()
# ```
# :::
#
# :::{tab-item} QURI Parts
# :sync: quri_parts
#
# ```python
# from qamomile.quri_parts import QuriPartsTranspiler
#
# transpiler = QuriPartsTranspiler()
# ```
# :::
#
# :::{tab-item} CUDA-Q
# :sync: cudaq
#
# Use `qamomile[cudaq-cu12]` for a CUDA 12.x toolchain or
# `qamomile[cudaq-cu13]` for a CUDA 13.x toolchain — pick the one that
# matches your installed CUDA Toolkit. CUDA-Q is supported on Linux,
# macOS arm64, and Windows-via-WSL2 only.
#
# ```python
# from qamomile.cudaq import CudaqTranspiler
#
# transpiler = CudaqTranspiler()
# ```
# :::
# ::::

# %%
# Transpiler — by default this article uses Qiskit. If you picked a
# different tab above (QURI Parts / CUDA-Q), copy the two lines from
# that tab into this cell in place of the two below, and make sure the
# matching pip install line further up has been uncommented.
from qamomile.qiskit import QiskitTranspiler

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
# We use `scipy.optimize.minimize` with the COBYLA method. At each
# iteration, the optimizer samples the circuit and evaluates the mean
# energy. We seed the NumPy generator (so the initial variational
# parameters are stable) and the executor's underlying simulator (so
# per-shot draws are reproducible across runs). How to seed the
# simulator depends on the SDK you picked at the top — copy the
# matching snippet from the tab block below into the executor cell
# further down if you swapped tabs.
#
# Later sections of this article construct two more fresh executors
# (`executor_manual`, `executor_builtin`) that need to start from the
# same RNG state — we factor that into a `make_executor()` helper so
# swapping SDKs only touches one place.
#
# ::::{tab-set}
# :::{tab-item} Qiskit
# :sync: qiskit
#
# ```python
# from qiskit_aer import AerSimulator
#
# def make_executor():
#     return transpiler.executor(
#         backend=AerSimulator(seed_simulator=SEED, max_parallel_threads=1)
#     )
# ```
#
# `seed_simulator=SEED` makes per-shot draws reproducible;
# `max_parallel_threads=1` is needed because Aer's parallel sampling
# can otherwise shuffle draws across threads. In production code you
# can drop the threads constraint (or only enable it in tests / docs
# builds) and trade a little determinism for performance.
# :::
#
# :::{tab-item} QURI Parts
# :sync: quri_parts
#
# ```python
# def make_executor():
#     # qulacs (QURI Parts' default simulator) does not expose seedable
#     # sampling, so per-shot counts vary between runs. The optimisation
#     # still converges to roughly the same neighbourhood at the chosen
#     # shot count, but the cost-history curve below is NOT bit-for-bit
#     # reproducible across runs.
#     return transpiler.executor()
# ```
# :::
#
# :::{tab-item} CUDA-Q
# :sync: cudaq
#
# ```python
# import cudaq
#
# def make_executor():
#     # cudaq's RNG is process-global; reseeding before each executor
#     # call gives in-notebook reproducibility but is NOT safe across
#     # concurrent kernels in the same process.
#     cudaq.set_random_seed(SEED)
#     return transpiler.executor()
# ```
# :::
# ::::

# %%
import os

import numpy as np
from scipy.optimize import minimize

SEED = 42

# %%
# Executor factory — by default this article uses Qiskit's AerSimulator
# with a fixed seed. If you picked a different tab above, copy that
# tab's `make_executor` definition over the lines below (and make sure
# the matching pip install line at the top of this article is
# uncommented).
from qiskit_aer import AerSimulator


def make_executor():
    """Fresh executor with deterministic sampling for this tutorial."""
    return transpiler.executor(
        backend=AerSimulator(seed_simulator=SEED, max_parallel_threads=1)
    )


executor = make_executor()

# %%
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
sample_shots = 256 if docs_test_mode else 2048
maxiter = 20 if docs_test_mode else 500
cost_history: list[float] = []


def cost_fn(params):
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
# ## Decode and Analyze Results
#
# We sample the circuit with the optimized parameters and interpret the
# measurement outcomes. `decode_from_sampleresult` returns samples already
# in the spin domain (+1 / -1), so we can count cut edges directly —
# no binary conversion needed.

# %%
gammas_opt = list(res.x[:p])
betas_opt = list(res.x[p:])

final_result = executable.sample(
    executor,
    shots=sample_shots,
    bindings={"gammas": gammas_opt, "betas": betas_opt},
).result()

decoded = spin_model.decode_from_sampleresult(final_result)

# %%
from collections import Counter

cut_distribution: Counter[int] = Counter()
best_qaoa_cut = 0
best_qaoa_sample = None

for sample, _energy, occ in zip(
    decoded.samples, decoded.energy, decoded.num_occurrences
):
    # sample is a dict {vertex_index: spin_value (+1 or -1)}
    spins = [sample[i] for i in range(num_nodes)]
    cut = sum(1 for i, j in G.edges() if spins[i] != spins[j])
    cut_distribution[cut] += occ
    if cut > best_qaoa_cut:
        best_qaoa_cut = cut
        best_qaoa_sample = spins

print(f"Best QAOA cut: {best_qaoa_cut}  (optimal: {best_cut})")
print(f"Best partition (spins): {best_qaoa_sample}")

# %%
cuts = sorted(cut_distribution.keys())
counts = [cut_distribution[c] for c in cuts]

plt.figure(figsize=(8, 4))
plt.bar([str(c) for c in cuts], counts, color="#2696EB")
plt.xlabel("Cut size")
plt.ylabel("Frequency")
plt.title("Distribution of MaxCut Values from QAOA")
plt.show()

# %%
if best_qaoa_sample is not None:
    color_map = [
        "#FF6B6B" if best_qaoa_sample[i] == +1 else "#4ECDC4" for i in range(num_nodes)
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
# ## Using the Built-in `qaoa_state`
#
# Everything we implemented above — superposition, cost layer, mixer layer,
# and the layered loop — is already provided by
# `qamomile.circuit.algorithm.qaoa_state`. It accepts exactly the same
# Ising coefficients (`quad`, `linear`) and variational parameters
# (`gammas`, `betas`).
#
# Let's build the same circuit using the built-in function to confirm
# that it implements the same structure. Each executor below is
# instantiated with the same `seed_simulator=SEED`. Under a fixed seed,
# **identical circuits yield identical samples and therefore identical
# mean energies**. With finite shots the per-circuit estimate still
# carries shot-noise — that does not vanish under seeding — so if the
# two printed mean energies *do* differ, it indicates that the manual
# and built-in routes did not emit bit-identical circuits (e.g., a gate
# ordering or compilation difference), not residual sampling noise.

# %%
from qamomile.circuit.algorithm import qaoa_state


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


# %% [markdown]
# We transpile and sample with the same optimized parameters.

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

executor_manual = make_executor()
executor_builtin = make_executor()

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
# In this tutorial we:
#
# 1. Defined a MaxCut problem and wrote it *directly* as an Ising
#    Hamiltonian on spin variables — no QUBO / binary-variable detour.
# 2. Built the spin-domain `BinaryModel` with `BinaryModel.from_ising`.
# 3. Built every component of the QAOA circuit as a `@qkernel` —
#    superposition, cost layer, mixer layer, and the full ansatz.
# 4. Ran a classical optimization loop and decoded the spin-domain
#    results.
# 5. Verified that `qamomile.circuit.algorithm.qaoa_state` provides the
#    same circuit with a single function call.
#
# The same spin-first recipe applies to any Ising-like problem —
# spin-glass ground-state search, weighted MaxCut, Sherrington–Kirkpatrick
# model, and so on: plug the $h_i$ and $J_{ij}$ coefficients into
# `BinaryModel.from_ising` and reuse the circuit components above.
#
# **Next steps:**
#
# - For problems that are naturally expressed with **binary variables** or
#   that require **constraints** (penalty terms), see
#   [QAOA for Graph Partitioning](qaoa_graph_partition),
#   which uses the higher-level `QAOAConverter` together with JijModeling.
