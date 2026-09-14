# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # Quantum-enhanced Markov chain Monte Carlo for combinatorial optimization
#
# Quantum computers offer an approach to solving combinatorial optimization problems that differs from classical computing and may lead to more efficient solutions.
# However, quantum optimization at scales that challenge even state-of-the-art classical solvers remains difficult.
# Recently, quantum-enhanced Markov chain Monte Carlo (QeMCMC) has shown promising results in approximating complex probability distributions.
# Building on these results, [Marshall et al. (2026)](https://arxiv.org/abs/2602.06171) proposed a method that combines device sampling and QeMCMC with warm starts and parallel tempering.
# This tutorial demonstrates an implementation of their method using Qamomile.

# %%
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit_aer import AerSimulator

import qamomile.circuit as qmc
from qamomile.circuit.algorithm.qaoa import ising_cost
from qamomile.qiskit import QiskitTranspiler

# %% [markdown]
# ## Background
#
# ### Combinatorial Optimization and Quantum Computing
#
# Combinatorial optimization is an important application of quantum computing.
# Quantum optimization algorithms have been widely studied for their potential to offer substantial speedups over classical algorithms.
# A prominent example is the Quantum Approximate Optimization Algorithm (QAOA; see, for example, [Farhi et al. (2014)](https://arxiv.org/abs/1411.4028)).
# However, noise in current quantum hardware limits the number of qubits and circuit depth that can be used.
# Proposed improvements to QAOA include warm-starting techniques that prepare an initial state from a classical optimization solution.
# Warm-starting QAOA is known to depend strongly on the solution used to prepare its initial state.
# Its search is biased toward the neighborhood of that solution: a good initial solution allows detailed exploration nearby, but reaching a better solution across an energy barrier can be difficult.
#
# ### Previous Work: QeMCMC
#
# An important precursor to [Marshall et al. (2026)](https://arxiv.org/abs/2602.06171) is the work of [Layden et al. (2023)](https://www.nature.com/articles/s41586-023-06095-4), who introduced QeMCMC.
# QeMCMC uses a quantum computer to generate proposals for MCMC.
# The idea is to use quantum dynamics to propose states that are far apart in configuration space but close in energy.  
# Given the current MCMC state $\mathbf{s}$, we prepare the corresponding computational basis state $\vert \mathbf{s} \rangle$ on a quantum computer.
# Applying $U = e^{-iHt}$ gives
#
# $$
# \vert \mathbf{s} \rangle \ \xrightarrow{U} \ U \vert \mathbf{s} \rangle 
# = \sum_{\mathbf{s}'} c_{\mathbf{s}'} \vert \mathbf{s}' \rangle \tag{1}
# $$
#
# Measuring in the computational basis then yields $\mathbf{s}'$ with probability
#
# $$
# Q_\mathrm{Q} (\mathbf{s}' \vert \mathbf{s}) 
# = \vert \langle \mathbf{s}' \vert U \vert \mathbf{s} \rangle \vert^2 \tag{2}
# $$
#
# This outcome is used as an MCMC proposal.
# The quantum computer generates the proposal distribution, and a classical computer decides whether to accept or reject the proposed state.
# Now impose on $U$ the symmetry condition $\vert \langle \mathbf{s}' \vert U \vert \mathbf{s} \rangle \vert = \vert \langle \mathbf{s} \vert U \vert \mathbf{s}' \rangle \vert$, which supports detailed balance.
# The acceptance probability then becomes
#
# $$
# A(\mathbf{s}' \vert \mathbf{s}) 
# = \min \left[ 1, e^{-\Delta E / T} \right] \tag{3}
# $$
#
# We can therefore sample from a quantum distribution that is difficult to sample classically without explicitly computing the quantum probabilities $Q_\mathrm{Q}$.  
# To construct the unitary $U$ introduced above, [Layden et al. (2023)](https://www.nature.com/articles/s41586-023-06095-4) use a Hamiltonian of the form
#
# $$
# H 
# = (1 - \kappa) \alpha H_\mathrm{prob} + \kappa H_\mathrm{mix} \tag{4}
# $$
#
# Here,
#
# $$
# H_\mathrm{prob} 
# = - \sum_{i>j} J_{ij} Z_i Z_j - \sum_i h_i Z_i 
# = \sum_\mathbf{s} E(\mathbf{s}) \vert \mathbf{s} \rangle \langle \mathbf{s} \vert \tag{5}
# $$
#
# encodes the energy landscape of a classical Ising model in a quantum Hamiltonian.
# The simple mixer Hamiltonian
#
# $$
# H_\mathrm{mix} 
# = \sum_i X_i \tag{6}
# $$
#
# promotes quantum transitions.
# The factor $\alpha = \| H_\mathrm{mix} \|_F / \| H_\mathrm{prob} \|_F$ provides normalization.  
# Unlike in QAOA, $\kappa, t$ are not variational parameters to be optimized in QeMCMC.
# Instead, they are sampled at each MCMC iteration, for example as $\kappa = \mathrm{Uniform} [0.25, 0.6], t = \mathrm{Uniform} [2, 20]$.
# For more details and an implementation, see the [QeMCMC tutorial](qe_mcmc.ipynb).
#
# ## Proposed Method
#
# ### Connection to QAOA
#
# Building on the idea of using a quantum circuit to generate proposal distributions in [Layden et al. (2023)](https://www.nature.com/articles/s41586-023-06095-4), [Marshall et al. (2026)](https://arxiv.org/abs/2602.06171) propose a method for efficiently searching for ground states in combinatorial optimization.
# We replace $H_\mathrm{prob}$ with an Ising Hamiltonian that represents the combinatorial optimization problem:
#
# $$
# H 
# = (1 - \kappa) \alpha H_\mathrm{cost} + \kappa H_\mathrm{mix} \tag{7}
# $$
#
# This gives
#
# $$
# U 
# = e^{-it\{(1-\kappa) \alpha H_\mathrm{cost} + \kappa H_\mathrm{mix}\}} 
# \underbrace{\approx}_{\mathrm{Trotter \ decomposition}} e^{-it \kappa H_\mathrm{mix}} e^{-it(1-\kappa) \alpha H_\mathrm{cost}} \tag{8}
# $$
#
# Splitting the evolution into two identical Trotter steps yields
#
# $$
# e^{-itH} 
# \approx \left(e^{-i\beta H_\mathrm{mix}} e^{-i\gamma H_\mathrm{cost}} \right)^2, \quad 
# \left( \gamma = \frac{t}{2} (1 - \kappa) \alpha, \ \beta = \frac{t}{2} \kappa \right) \tag{9}
# $$
#
# This has the form of a QAOA-like circuit with p = 2.
#
# ### Warm Starting
#
# Unlike the QeMCMC procedure described above, [Marshall et al. (2026)](https://arxiv.org/abs/2602.06171) use the currently accepted state $\vert \mathbf{s}_k \rangle$ as a warm start.
# We introduce a regularization parameter $0 < \epsilon < 1/2$ and soften the bit values as follows:
#
# $$
# \tilde{s}_i 
# = \left\{ \begin{array}{ll}
# \epsilon & s_i = 0 \\
# 1 - \epsilon & s_i = 1  
# \end{array} \right. \tag{10}
# $$
#
# Setting the angles to $\theta_i = 2 \mathrm{arcsin} \sqrt{\tilde{s}_i}$, we prepare the corresponding quantum state
#
# $$
# \vert \psi_\mathrm{WS} (s_k) \rangle 
# = \bigotimes_i R_y (\theta_i) \vert 0 \rangle \tag{11}
# $$
#
# As in WS-QAOA, we also modify the mixer to match this initial state.
# Specifically, choosing
#
# $$
# H_{\mathrm{mix}, i}^{(\mathrm{WS})} 
# = - \sin \theta_i X_i - \cos \theta_i Z_i \tag{12}
# $$
#
# makes $\vert \psi_\mathrm{WS} \rangle$ the ground state of the mixer.
# Thus, in [Marshall et al. (2026)](https://arxiv.org/abs/2602.06171), whenever the current state $s_k$ changes during an iteration, the angles $\theta_i (s_k)$ change as well, and both the initial state and the mixer are updated accordingly.
# In this way, WS-QAOA acts as a dynamic proposal generator for the Markov chain.
#
# ### Parallel Tempering (Replica Exchange)
#
# In standard MCMC, the temperature $T$ in Eq. (3) is held fixed.
# This can leave the chain trapped in a local optimum.
# Instead, we can run several MCMC chains in parallel at different temperatures $T_a$.
# These chains at different temperatures are called replicas.
# In the high-temperature limit, $e^{-\Delta E / T} \rightarrow 1$, allowing MCMC to explore broadly regardless of energy differences.
# In the low-temperature limit, MCMC explores the neighborhood of low-energy solutions more closely.
# Consider two replicas with temperatures $T_r < T_{r'}$: the replica at $T_{r'}$ explores broadly.
# When it finds a promising energy basin, transferring its state to temperature $T_r$ allows a more detailed search for low-energy solutions within that basin.
# This is the idea behind parallel tempering.
# The exchange probability for replicas $r, r'$ is
#
# $$
# A_\mathrm{exchange} (\mathbf{s}_r, \mathbf{s}_{r'})
# = \min \left[ 1, \exp \left\{ \left( \frac{1}{T_r} - \frac{1}{T_{r'}} \right) (E(\mathbf{s}_{r}) - E(\mathbf{s}_{r'})) \right\} \right] \tag{13}
# $$
#
# ### Algorithm Overview
#
# We can now summarize the algorithm proposed by [Marshall et al. (2026)](https://arxiv.org/abs/2602.06171):
#
# 1. Choose the QAOA parameters $\beta, \gamma$, the warm-start regularization parameter $\epsilon$, and the replica temperatures $T_r \ (r = 1, 2, \dots, R)$.
# 2. Initialize replicas at several temperatures.
# 3. From the current solution $\mathbf{s}_k^{(r)}$ of each replica, prepare $\vert \psi_\mathrm{WS}(\mathbf{s}_k^{(r)}) \rangle$ and set the corresponding warm-start mixer.
# 4. Run a QAOA-like quantum circuit with $p = 2$ to generate many candidates.
# 5. Retain only the low-energy candidates among the measurement outcomes and select the next proposal from this subset.
# 6. Decide whether to accept the proposal using the Metropolis–Hastings algorithm.
# 7. At regular intervals, exchange states between replicas at neighboring temperatures.
# 8. Repeat steps 3–7.
#
# The method searches for an optimal solution by repeatedly running shallow quantum circuits rather than executing a single deep circuit.
# Unlike standard warm-starting QAOA, the state used for warm-start preparation is not fixed.
# Rebuilding the warm-start state at every iteration works together with the QeMCMC search to guide the exploration toward better solutions.

# %% [markdown]
# ## Implementation with Qamomile
#
# Let us implement the QeMCMC approach to combinatorial optimization described above using Qamomile.
#
# ### Generating a problem instance
#
# We will generate a grid graph and solve its maximum independent set (MIS) problem.

# %%
# 18-node graph: 3 x 6 grid
n_rows = 3
n_cols = 6
n = n_rows * n_cols

edges = []
for row in range(n_rows):
    for col in range(n_cols):
        node = row * n_cols + col
        if col + 1 < n_cols:
            edges.append((node, node + 1))
        if row + 1 < n_rows:
            edges.append((node, node + n_cols))

# %% [markdown]
# Let us visualize the generated graph.

# %%
graph = nx.Graph()
graph.add_nodes_from(range(n))
graph.add_edges_from(edges)

positions = {
    row * n_cols + col: (col, -row)
    for row in range(n_rows)
    for col in range(n_cols)
}

plt.figure(figsize=(7, 7))
nx.draw(
    graph,
    pos=positions,
    with_labels=True,
    node_size=700,
    font_size=9,
)
plt.title("18-node MIS instance: 3 x 6 grid graph")
plt.show()

# %% [markdown]
# ### Setting the parameters
#
# We initialize the parameters used in the following code.
#
# * `penalty`: Coefficient of the penalty term in the QUBO formulation of MIS.
# * `epsilon`: Regularization parameter for the warm start.
# * `gamma, beta`: Parameters of the QAOA-like circuit defined in Eq. (9).
# * `shots`: Number of measurement shots for the quantum circuit.
# * `top_k`: Number of low-energy samples to select as proposal candidates.
# * `num_replicas`: Number of replicas used for parallel tempering.
# * `temperatures`: Temperatures of the replicas.
# * `swap_interval`: Number of iterations between replica exchanges.
# * `max_iterations`: Maximum number of iterations.

# %%
# Set seed of random numbers
rng = np.random.default_rng()

# MIS penalty. lambda = 2 is sufficient for this educational QUBO.
penalty = 2.0

# Warm-start / QAOA-like proposal parameters
# Paper: p=2 with gamma_1 = gamma_2 and beta_1 = beta_2.
epsilon = 0.25
gamma = 0.70
beta = 0.40

# Multi-shot low-energy selection.
shots = 128
top_k = 10

# Parallel tempering
num_replicas = 5
temperatures = np.geomspace(0.05, 2.0, num_replicas)
swap_interval = 1
max_iterations = 20

print("n             =", n)
print("number edges  =", len(edges))
print("temperatures  =", temperatures)

# %% [markdown]
# ### Constructing the Ising Hamiltonian
#
# Let us construct the Ising Hamiltonian for MIS, starting from its QUBO formulation.
#
# $$
# H_\mathrm{QUBO} 
# = - \sum_{v_i \in V} x_i + \lambda \sum_{(v_i, v_j) \in E} x_i x_j \tag{14}
# $$
#
# To convert this to an Ising Hamiltonian, we use the correspondence between binary and spin variables, $x_i = \frac{1 - Z_i}{2}$.
# This gives
#
# $$
# H_\mathrm{Ising} 
# = - \frac{n}{2} + \frac{1}{2} \sum_{v_i \in V} Z_i + \sum_{(v_i, v_j) \in E} \left( \frac{\lambda}{4} - \frac{\lambda}{4} Z_i - \frac{\lambda}{4} Z_j + \frac{\lambda}{4} Z_i Z_j \right) 
# = - \frac{n}{2} + \frac{\lambda \vert E \vert}{4} + \sum_i \left( \frac{1}{2} - \frac{\lambda d_i}{4} \right) Z_i + \frac{\lambda}{4} \sum_{(v_i, v_j) \in E} Z_i Z_j \tag{15}
# $$
#
# Here, $d_i$ is the degree of vertex $i$, or the number of edges incident to it.
# We use the degrees to collect the linear terms and separate them from the quadratic terms.

# %%
# Degree of each vertex
degree = np.zeros(n, dtype=int)
for i, j in edges:
    degree[i] += 1
    degree[j] += 1

# H_cost = constant + sum_i linear[i] Z_i + sum_(i,j) quad[(i,j)] Z_i Z_j
linear = {
    i: float(0.5 - penalty * degree[i] / 4.0)
    for i in range(n)
}
quad = {
    (i, j): float(penalty / 4.0)
    for i, j in edges
}
constant = float(-n / 2.0 + penalty * len(edges) / 4.0)

print("constant =", constant)
print("linear   =", linear)
print("quad      =", quad)

# %% [markdown]
# ### Computing the optimal solution
#
# For this grid graph, selecting alternating vertices in a checkerboard pattern gives an optimal MIS solution.
# We use this fact to compute the optimal energy as a reference for the Qamomile implementation below.

# %%
# One exact optimum for a 3 x 6 grid: checkerboard selection.
exact_state = np.zeros(n, dtype=int)
for row in range(n_rows):
    for col in range(n_cols):
        if (row + col) % 2 == 0:
            exact_state[row * n_cols + col] = 1

exact_energy = float(
    -exact_state.sum()
    + penalty * sum(exact_state[i] * exact_state[j] for i, j in edges)
)

print("QUBO/Ising random checks passed")
print("known optimum energy =", exact_energy)
print("MIS size             =", int(exact_state.sum()))
print("one optimum          =", np.flatnonzero(exact_state).tolist())


# %% [markdown]
# ### Implementing the quantum kernel
#
# We prepare the warm-start initial state using the angles $\theta_i$ derived from the current MCMC state.
# We then apply $e^{-i \gamma H_\mathrm{Ising}}$ using Qamomile's `ising_cost` and introduce a mixer matched to the warm-start initial state.

# %%
@qmc.qkernel
def marshall_warm_qaoa_sampling(
    n: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    thetas: qmc.Vector[qmc.Float],
    gamma: qmc.Float,
    beta: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")

    # Warm-start initial state
    for i in qmc.range(n):
        q[i] = qmc.ry(q[i], thetas[i])

    # p = 2, gamma_1 = gamma_2 = gamma, beta_1 = beta_2 = beta
    q = ising_cost(quad, linear, q, gamma)
    for i in qmc.range(n):
        q[i] = qmc.ry(q[i], -thetas[i])
        q[i] = qmc.rz(q[i], -2.0 * beta)
        q[i] = qmc.ry(q[i], thetas[i])

    q = ising_cost(quad, linear, q, gamma)
    for i in qmc.range(n):
        q[i] = qmc.ry(q[i], -thetas[i])
        q[i] = qmc.rz(q[i], -2.0 * beta)
        q[i] = qmc.ry(q[i], thetas[i])

    return qmc.measure(q)


# %% [markdown]
# ### Transpiling the quantum circuit
#
# Let us transpile the quantum kernel for execution on a Qiskit backend.
# We fix the number of qubits `n` and the Ising Hamiltonian coefficients `linear` and `quad`, while keeping `thetas`, `gamma`, and `beta` as parameters that can be changed at runtime.

# %%
transpiler = QiskitTranspiler()

executable = transpiler.transpile(
    marshall_warm_qaoa_sampling,
    bindings={
        "n": n,
        "quad": quad,
        "linear": linear,
    },
    parameters=["thetas", "gamma", "beta"],
)

backend = AerSimulator(method="matrix_product_state")
executor = transpiler.executor(backend=backend)

print("transpilation complete")
print("Aer method = matrix_product_state")

# %% [markdown]
# ### Preparing the initial state of each replica
#
# We generate `num_replicas` random bitstrings as the initial states for parallel tempering.
# We also compute their energies and record which replica has the lowest energy at initialization.

# %%
replica_states = [rng.integers(0, 2, size=n) for _ in range(num_replicas)]
replica_energies = np.array([
    -state.sum() + penalty * sum(state[i] * state[j] for i, j in edges)
    for state in replica_states
], dtype=float)

best_replica = int(np.argmin(replica_energies))
best_state = replica_states[best_replica].copy()
best_energy = float(replica_energies[best_replica])

# Store the best-so-far curve and every temperature slot after each full iteration.
history_best = [best_energy]
history_replicas = [replica_energies.copy()]

print("initial energies =", replica_energies.tolist())
print("initial best     =", best_energy)

# %% [markdown]
# ### Main loop
#
# We can now assemble the main loop of the algorithm.
# For each replica, we run warm-start QAOA and measure the circuit to generate proposal candidates.
# We select the `top_k` lowest-energy samples and choose a proposal from them.
# A Metropolis–Hastings acceptance step at the replica's temperature determines whether to accept or reject the proposal, and the accepted state becomes the next warm start.
# After updating all replicas, we attempt exchanges between neighboring temperatures.

# %%
for iteration in range(1, max_iterations + 1):
    # ----- Local QeMCMC-inspired update for every replica -----
    for r in range(num_replicas):
        state = replica_states[r]
        current_energy = float(replica_energies[r])
        temperature = float(temperatures[r])

        # Dynamic warm start from the currently accepted state.
        soft = np.where(state == 0, epsilon, 1.0 - epsilon)
        thetas = (2.0 * np.arcsin(np.sqrt(soft))).tolist()

        sample_result = executable.sample(
            executor,
            shots=shots,
            bindings={
                "thetas": thetas,
                "gamma": gamma,
                "beta": beta,
            },
        ).result()

        # Preserve multiplicity of measured bitstrings.
        shot_states = []
        for outcome, count in sample_result.results:
            bits = np.asarray(outcome, dtype=int)
            for _ in range(count):
                shot_states.append(bits.copy())

        shot_energies = np.array([
            -bits.sum() + penalty * sum(bits[i] * bits[j] for i, j in edges)
            for bits in shot_states
        ], dtype=float)

        # Keep the low-energy tail. Include all ties at the top_k cutoff.
        sorted_energy = np.sort(shot_energies)
        cutoff_energy = sorted_energy[min(top_k - 1, len(sorted_energy) - 1)]
        pool_indices = np.flatnonzero(shot_energies <= cutoff_energy)
        proposal_index = int(rng.choice(pool_indices))
        proposal_state = shot_states[proposal_index].copy()
        proposal_energy = float(shot_energies[proposal_index])

        # Heuristic Metropolis step (effective Q is asymmetric after filtering).
        delta_energy = proposal_energy - current_energy
        if delta_energy <= 0.0:
            acceptance = 1.0
        else:
            acceptance = np.exp(-delta_energy / temperature)

        if rng.random() < acceptance:
            replica_states[r] = proposal_state
            replica_energies[r] = proposal_energy

        if replica_energies[r] < best_energy:
            best_energy = float(replica_energies[r])
            best_state = replica_states[r].copy()

    # ----- Replica exchange: alternate even / odd neighboring pairs -----
    if iteration % swap_interval == 0:
        start = (iteration // swap_interval - 1) % 2

        for i in range(start, num_replicas - 1, 2):
            j = i + 1
            log_ratio = (
                (1.0 / temperatures[i] - 1.0 / temperatures[j])
                * (replica_energies[i] - replica_energies[j])
            )
            swap_acceptance = 1.0 if log_ratio >= 0.0 else np.exp(log_ratio)

            if rng.random() < swap_acceptance:
                replica_states[i], replica_states[j] = (
                    replica_states[j].copy(),
                    replica_states[i].copy(),
                )
                replica_energies[i], replica_energies[j] = (
                    replica_energies[j],
                    replica_energies[i],
                )

    history_best.append(best_energy)
    history_replicas.append(replica_energies.copy())

    print(
        f"iter={iteration:2d}  "
        f"cold={replica_energies[0]:6.1f}  "
        f"best={best_energy:5.1f}"
    )

    if best_energy <= exact_energy + 1e-12:
        print("Reached the exact optimum.")
        break

# %% [markdown]
# ## Results
#
# Let us display the results of the main loop.

# %%
selected_vertices = np.flatnonzero(best_state).tolist()
feasible = all(not (best_state[i] == 1 and best_state[j] == 1) for i, j in edges)

print("best energy       =", best_energy)
print("known optimum     =", exact_energy)
print("best bitstring    =", best_state.tolist())
print("selected vertices =", selected_vertices)
print("independent set?  =", feasible)
print("matches optimum?  =", np.isclose(best_energy, exact_energy))

# %% [markdown]
# Let us visualize how the energy of each replica changes over the iterations.

# %%
history_replicas_array = np.asarray(history_replicas)

plt.figure(figsize=(11, 6))

# Optimization trajectory of every temperature slot.
for r in range(num_replicas):
    plt.plot(
        history_replicas_array[:, r],
        label=f"replica {r} (T={temperatures[r]:.3f})",
        alpha=0.75,
    )

# Best solution found anywhere up to each iteration.
plt.plot(
    history_best,
    linewidth=2.5,
    label="best found",
)

# Known optimum for the 5 x 5 grid.
plt.axhline(
    exact_energy,
    linestyle="--",
    linewidth=2,
    label="known optimum",
)

plt.xlabel("iteration")
plt.ylabel("QUBO energy")
plt.title("Optimization history of all replicas")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# %% [markdown]
# Finally, let us visualize the solution on the graph.

# %%
selected_mask = [bool(best_state[i]) for i in range(n)]
node_sizes = [900 if selected_mask[i] else 550 for i in range(n)]

plt.figure(figsize=(7, 7))
nx.draw_networkx_edges(graph, pos=positions, alpha=0.6)
nx.draw_networkx_nodes(
    graph,
    pos=positions,
    nodelist=[i for i in range(n) if not selected_mask[i]],
    node_size=[node_sizes[i] for i in range(n) if not selected_mask[i]],
)
nx.draw_networkx_nodes(
    graph,
    pos=positions,
    nodelist=[i for i in range(n) if selected_mask[i]],
    node_size=[node_sizes[i] for i in range(n) if selected_mask[i]],
    node_shape="s",
)
nx.draw_networkx_labels(graph, pos=positions, font_size=9)
plt.title(f"Best candidate: |S| = {len(selected_vertices)}, E = {best_energy:.1f}")
plt.axis("off")
plt.show()

# %% [markdown]
# As expected, the selected vertices form a checkerboard pattern.

# %% [markdown]
# ## Summary
#
# * We introduced the method of [Marshall et al. (2026)](https://arxiv.org/abs/2602.06171), which combines QeMCMC with warm-start QAOA and parallel tempering.
# * The method uses shallow quantum circuits to generate proposals in an iterative search for a solution.
# * Using the state obtained after each iteration as the next warm start can improve the search for an optimal solution.
# * Qamomile's `ising_cost` function makes it straightforward to implement $e^{-i H_\mathrm{Ising} t}$.

# %% [markdown]
#
