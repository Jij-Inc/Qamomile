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
#     display_name: qamomile (3.11.16.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# tags: [algorithm, optimization, variational]
# ---
#
# # Feedback-Based Algorithm for Quantum Optimization (FALQON)
#
# Quantum computers are expected to offer advantages over classical computers in combinatorial optimization.
# The Quantum Approximate Optimization Algorithm (QAOA) was proposed as a method for solving combinatorial optimization problems, but it requires classical optimization.
# This page introduces the Feedback-based ALgorithm for Quantum OptimizatioN (FALQON), proposed by [Magann et al. (2022)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.250502). FALQON is a quantum optimization method that uses measurement-based feedback without classical optimization of variational parameters.
# We also implement FALQON with Qamomile and verify the implementation on a MaxCut problem.

# %%
# Install the latest Qamomile through pip!
# # !pip install "qamomile[qiskit]"

# %%
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from scipy.sparse import SparseEfficiencyWarning
from qiskit_aer import AerSimulator

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.qiskit import QiskitTranspiler


warnings.filterwarnings(
    "ignore",
    category=SparseEfficiencyWarning,
)

# %% [markdown] vscode={"languageId": "raw"}
# ## Background
#
# ### Combinatorial Optimization and Quantum Optimization Methods
#
# Combinatorial optimization has a wide range of valuable applications, including logistics, supply chains, and drug discovery.
# General combinatorial optimization problems are NP-hard, so finding their exact optimal solutions is not easy.
# Consequently, practical methods that aim to find high-quality approximate solutions have attracted considerable attention.
# Quantum annealing and the Quantum Approximate Optimization Algorithm (QAOA) have been developed as quantum-based approaches.
# Although no rigorous quantum advantage has yet been established for these approaches, an advantage may emerge for some problem sizes.
#
# ### Prior Work: Quantum Lyapunov Control (QLC)
#
# QAOA is a well-known quantum optimization method, but it requires the classical optimization of many variational parameters, which presents a scaling challenge.
# To address this issue, [Magann et al. (2022)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.250502) proposed applying Quantum Lyapunov Control (QLC), which uses feedback to steer a quantum system in a desired direction.
# A representative study of controlling quantum systems through an explicit Lyapunov function is [Grivopoulos & Bamieh (2003)](https://ieeexplore.ieee.org/document/1272601).
# That paper considered the Schrödinger equation
#
# $$
# i \frac{d}{dt} \vert \psi \rangle 
# = \left( H_0 + H_1 u_1 (t) \right) \vert \psi \rangle 
# \equiv H \vert \psi \rangle \tag{1}
# $$
#
# For this single-control setting, we write $u_1(t)$ as $u(t)$ below. The paper discussed how to stabilize a target eigenstate by designing a feedback law $u(t)$ that decreases the Lyapunov function $V(\psi)$.
# Here, $H_0$ is the system Hamiltonian and describes the time evolution of the quantum system in the absence of external control.
# The control Hamiltonian $H_1$ is an operator that describes how the external classical control $u(t)$ acts on the quantum system.
# [Grivopoulos & Bamieh (2003)](https://ieeexplore.ieee.org/document/1272601) proposed using
#
# $$
# V(\psi) 
# = \langle \psi \vert P \vert \psi \rangle \tag{2}
# $$
#
# as the Lyapunov function.
# Here, $P$ is a Hermitian operator, and its expectation value is used as the Lyapunov function.
# The operator $P$ is designed to satisfy $[H_0, P] = 0$.
# We can then rewrite the time derivative as
#
# $$
# \frac{dV}{dt} 
# = \left( \frac{d}{dt} \langle \psi \vert \right) P \vert \psi \rangle + \langle \psi \vert P \left( \frac{d}{dt} \vert \psi \rangle \right) 
# = i \langle \psi \vert H P \vert \psi \rangle + \langle \psi \vert P (-iH) \vert \psi \rangle 
# = i \langle \psi \vert [H, P] \vert \psi \rangle 
# = i \langle \psi \vert [H_0 + H_1 u, P] \vert \psi \rangle 
# = i u \langle \psi \vert [H_1, P] \vert \psi \rangle \tag{3}
# $$
#
# Defining $A(t) \equiv i \langle \psi \vert [H_1, P] \vert \psi \rangle$ gives
#
# $$
# \frac{dV}{dt} 
# = A(t) u(t) \tag{4}
# $$
#
# Furthermore, choosing $u(t) = -A(t)$ gives
#
# $$
# \frac{dV}{dt} 
# = - A^2 \leq 0 \tag{5}
# $$
#
# which makes it possible to control the system so that $V$ is monotonically non-increasing over time.
#
# ## Proposed Method
#
# ### Applying QLC
#
# [Magann et al. (2022)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.250502) considered the following quantum system:
#
# $$
# i \frac{d}{dt} \vert \psi \rangle 
# = \{H_p + \beta (t) H_d \} \vert \psi \rangle \tag{6}
# $$
#
# Here, $H_p$ is the problem Hamiltonian to be minimized, $H_d$ is the driver Hamiltonian, and $\beta(t)$ is a control parameter.
# To minimize $E(t) = \langle \psi \vert H_p \vert \psi \rangle$, we design the control parameter $\beta(t)$ so that $\frac{dE}{dt} \leq 0$.
# From the preceding QLC discussion, substituting $P \rightarrow H_p$ and $H_1 \rightarrow H_d$ into Eq. (3) gives
#
# $$
# \frac{dE}{dt} 
# = i \beta(t) \langle \psi \vert [H_d, H_p] \vert \psi \rangle \tag{7}
# $$
#
# Defining $A(t) \equiv i \langle \psi \vert [H_d, H_p] \vert \psi \rangle$ gives
#
# $$
# \frac{dE}{dt} 
# = A(t) \beta(t) \tag{8}
# $$
#
# Therefore, the simple choice
#
# $$
# \beta(t) 
# = - A(t) \tag{9}
# $$
#
# yields $\frac{dE}{dt} = - A(t)^2 \leq 0$ and makes $E(t)$ monotonically non-increasing.
#
# ### Constructing the Quantum Circuit
#
# To implement this method as a quantum circuit, we discretize time into intervals of width $\Delta t$.
# We define the time-evolution operators for the problem and driver Hamiltonians as
#
# $$
# U_p = e^{-i H_p \Delta t}, \quad U_d(\beta_k) 
# = e^{-i\beta_k H_d \Delta t} \tag{10}
# $$
#
# where $\beta_k$ is the value of $\beta(t)$ at step $k$.
# Using these operators, the state after $k$ steps is
#
# $$
# \vert \psi_k \rangle 
# = U_d(\beta_k) U_p U_d(\beta_{k-1}) U_p \cdots U_d(\beta_1) U_p \vert \psi_0 \rangle \tag{11}
# $$
#
# For the state after $k$ steps, we estimate $A_k = i \langle \psi_k \vert [H_d, H_p] \vert \psi_k \rangle$ from measurements and set $\beta_{k+1} = - A_k$ as the control parameter for step $k+1$.
# FALQON's non-increasing-energy guarantee applies in continuous time.
# A quantum-circuit implementation, however, discretizes time.
# [Magann et al. (2022)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.250502) noted that monotonicity can break down when $\Delta t$ is too large.
# Choosing a sufficiently small $\Delta t$ approximately preserves the continuous-time monotonic behavior.
#
# ### Measuring $A_k$
#
# To estimate $i[H_d, H_p]$ using Pauli measurements, we expand it in Pauli strings $P_j \in \{X, Y, Z, I\}^{\otimes n}$.
# Writing $i [H_d, H_p] = \sum_j \alpha_j P_j$ gives
#
# $$
# A_k 
# = \sum_j \alpha_j \langle \psi_k \vert P_j \vert \psi_k \rangle \tag{12}
# $$
#
# Let us examine this measurement procedure using MaxCut as an example.
# For MaxCut, the problem Hamiltonian can be written as
#
# $$
# H_p 
# = - \sum_{(i, j) \in \mathcal{E}} \frac{1}{2} (1 - Z_i Z_j) \tag{13}
# $$
#
# where $\mathcal{E}$ is the set of graph edges.
# If we choose the driver Hamiltonian as
#
# $$
# H_d 
# = \sum_{j=1}^n X_j \tag{14}
# $$
#
# then
#
# $$
# i [H_d, H_p] 
# = \sum_{(i, j) \in \mathcal{E}} (Y_i Z_j + Z_i Y_j) \tag{15}
# $$
#
# Therefore, for MaxCut,
#
# $$
# A_k 
# = \sum_{(i, j) \in \mathcal{E}} ( \langle \psi_k \vert Y_i Z_j \vert \psi_k \rangle + \langle \psi_k \vert Z_i Y_j \vert \psi_k \rangle ) \tag{16}
# $$
#
# so we only need to measure the expectation values of two-qubit Pauli strings such as $YZ$ and $ZY$.
# Under the usual local Pauli-basis measurement scheme, $Y_i Z_j$ and $Z_i Y_j$ require different product bases.
# Each shot consumes a freshly prepared copy of $\vert \psi_k \rangle$, so their expectation values are estimated from separate batches of shots.
# More generally, terms that are jointly measurable under the chosen measurement scheme can be grouped to reduce the number of distinct measurement settings.
# Even with such grouping, FALQON tends to incur a high measurement cost.
#

# %% [markdown]
# ## Implementation with Qamomile
#
# Let us implement the FALQON method proposed by [Magann et al. (2022)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.250502) with Qamomile.
#
# ### Creating the Problem Instance
#
# As in [Magann et al. (2022)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.250502), we solve a MaxCut problem.

# %%
G = nx.Graph()

G.add_edges_from(
    [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 3),
        (3, 4),
    ]
)

n = G.number_of_nodes()
m = G.number_of_edges()

print("Number of vertices:", n)
print("Number of edges:", m)

pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(5, 4))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=700,
)
plt.show()

# %% [markdown]
# ### Creating $H_p$, $H_d$, and $[H_d, H_p]$
#
# The problem Hamiltonian for MaxCut is given by Eq. (13).
# Here, we implement $H_p = \frac{1}{2} \sum_{(i, j) \in \mathcal{E}} Z_i Z_j$, omitting the constant term.
# We also implement the driver Hamiltonian from Eq. (14).
# Qamomile's `commutator` is convenient for constructing the commutator $[H_d, H_p]$.

# %%
Hp = qm_o.Hamiltonian(num_qubits=n)

for i, j in G.edges():
    Hp += 0.5 * qm_o.Z(i) * qm_o.Z(j)

Hd = qm_o.Hamiltonian(num_qubits=n)

for i in range(n):
    Hd += qm_o.X(i)

print("Hp =", Hp)
print("Hd =", Hd)

commutator_h = qm_o.commutator(Hd, Hp)

feedback_h = 1j * commutator_h

print("[Hd, Hp] =")
print(commutator_h)

print("\ni[Hd, Hp] =")
print(feedback_h)


# %% [markdown]
# ### Creating the FALQON Kernels
#
# Let us implement the FALQON circuit described by Eq. (11).
# `falqon_state` is a kernel that implements the FALQON layers and prepares the resulting quantum state.
# Using that kernel, `falqon_expval` is a qkernel that evaluates $A_k$ or the energy, while `falqon_sampling` performs the final sampling.

# %%
@qmc.qkernel
def falqon_state(
    n: qmc.UInt,
    depth: qmc.UInt,
    betas: qmc.Vector[qmc.Float],
    delta_t: qmc.Float,
    Hp: qmc.Observable,
    Hd: qmc.Observable,
) -> qmc.Vector[qmc.Qubit]:

    q = qmc.qubit_array(n, name="q")

    # Initial state: ground state of Hd = sum_i X_i
    q = qmc.h(q)
    q = qmc.z(q)

    # FALQON layers
    for k in qmc.range(depth):

        # Up = exp(-i Hp delta_t)
        q = qmc.pauli_evolve(
            q,
            Hp,
            delta_t,
        )

        # Ud(beta_k) = exp(-i beta_k Hd delta_t)
        q = qmc.pauli_evolve(
            q,
            Hd,
            betas[k] * delta_t,
        )

    return q

@qmc.qkernel
def falqon_expval(
    n: qmc.UInt,
    depth: qmc.UInt,
    betas: qmc.Vector[qmc.Float],
    delta_t: qmc.Float,
    Hp: qmc.Observable,
    Hd: qmc.Observable,
    obs: qmc.Observable,
) -> qmc.Float:

    q = falqon_state(
        n,
        depth,
        betas,
        delta_t,
        Hp,
        Hd,
    )

    return qmc.expval(q, obs)

@qmc.qkernel
def falqon_sampling(
    n: qmc.UInt,
    depth: qmc.UInt,
    betas: qmc.Vector[qmc.Float],
    delta_t: qmc.Float,
    Hp: qmc.Observable,
    Hd: qmc.Observable,
) -> qmc.Vector[qmc.Bit]:

    q = falqon_state(
        n,
        depth,
        betas,
        delta_t,
        Hp,
        Hd,
    )

    return qmc.measure(q)


# %% [markdown]
# In this implementation, `qmc.expval` evaluates $A_k$ and $\langle H_p \rangle$.
# This is an idealized setting: the simulation does not include the statistical error or measurement cost associated with a finite number of shots.
#
# ### Setting the FALQON Parameters
#
# We set the values required to run FALQON, including $\Delta t$, the maximum number of FALQON layers, and the initial value of $\beta$.

# %%
delta_t = 0.1
max_layers = 20

betas = [0.0]

A_history = []
energy_history = []
cut_history = []
beta_history = [0.0]

# %% [markdown]
# ### Transpiling to Qiskit
#
# We use Qamomile to transpile the kernels to Qiskit.

# %%
transpiler = QiskitTranspiler()

backend = AerSimulator(
    method="statevector",
    seed_simulator=42,
)

executor = transpiler.executor(backend=backend)

# %% [markdown]
# ### The Main FALQON Loop
#
# Using the functions defined above, we construct the FALQON feedback loop.

# %%
for depth in range(1, max_layers + 1):

    # -----------------------------------------
    # A_k = < i [Hd, Hp] >
    # -----------------------------------------

    feedback_executable = transpiler.transpile(
        falqon_expval,
        bindings={
            "n": n,
            "depth": depth,
            "delta_t": delta_t,
            "Hp": Hp,
            "Hd": Hd,
            "obs": feedback_h,
        },
        parameters=["betas"],
    )

    A_k = feedback_executable.run(
        executor,
        bindings={
            "betas": betas,
        },
    ).result()

    A_k = float(np.real(A_k))

    A_history.append(A_k)

    # -----------------------------------------
    # <Hp>
    # -----------------------------------------

    energy_executable = transpiler.transpile(
        falqon_expval,
        bindings={
            "n": n,
            "depth": depth,
            "delta_t": delta_t,
            "Hp": Hp,
            "Hd": Hd,
            "obs": Hp,
        },
        parameters=["betas"],
    )

    energy_reduced = energy_executable.run(
        executor,
        bindings={
            "betas": betas,
        },
    ).result()

    energy_reduced = float(np.real(energy_reduced))

    # Original paper:
    # Hp_full = Hp - |E|/2 I
    energy_full = energy_reduced - 0.5 * m

    energy_history.append(energy_full)

    # Since Hp_full = - MaxCut operator,
    # expected cut value = - <Hp_full>
    expected_cut = -energy_full
    cut_history.append(expected_cut)

    print(
        f"depth={depth:2d}  "
        f"beta={betas[-1]: .6f}  "
        f"A={A_k: .6f}  "
        f"<Hp>={energy_full: .6f}  "
        f"<cut>={expected_cut: .6f}"
    )

    # -----------------------------------------
    # beta_{k+1} = -A_k
    # -----------------------------------------

    if depth < max_layers:
        beta_next = -A_k
        betas.append(beta_next)
        beta_history.append(beta_next)

# %% [markdown]
# ## Results
#
# Let us confirm that the energy $\langle H_p \rangle$ is monotonically non-increasing.

# %%
plt.figure(figsize=(7, 4))

plt.plot(
    range(1, max_layers + 1),
    energy_history,
    marker="o",
)

plt.xlabel("FALQON layer")
plt.ylabel("<Hp>")
plt.title("FALQON energy")

plt.grid()
plt.show()

# %% [markdown]
# Similarly, let us visualize how $\beta_k$ changes over the iterations.

# %%
plt.figure(figsize=(7, 4))

plt.plot(
    range(1, len(beta_history) + 1),
    beta_history,
    marker="o",
)

plt.xlabel("FALQON layer")
plt.ylabel("beta_k")
plt.title("FALQON feedback parameters")

plt.grid()
plt.show()

# %% [markdown]
# Finally, we sample the final state.

# %%
sampling_executable = transpiler.transpile(
    falqon_sampling,
    bindings={
        "n": n,
        "depth": max_layers,
        "delta_t": delta_t,
        "Hp": Hp,
        "Hd": Hd,
    },
    parameters=["betas"],
)

shots = 5000

sample_result = sampling_executable.sample(
    executor,
    bindings={
        "betas": betas,
    },
    shots=shots,
).result()

print(sample_result.results)

# %% [markdown]
# We compute the MaxCut value from each sampled bit string and inspect the final solution candidate.

# %%
best_cut = -1
best_bits = None
best_count = 0

for value, count in sample_result.results:

    # value is already a tuple such as (0, 1, 0, 1, ...)
    bits = list(value)

    cut = sum(
        bits[i] != bits[j]
        for i, j in G.edges()
    )

    if cut > best_cut:
        best_cut = cut
        best_bits = bits
        best_count = count

print("Best bit string:", best_bits)
print("Best cut:", best_cut)
print("Count:", best_count)

# %% [markdown]
# We obtained a solution that cuts five of the six edges.
#
# ## Summary
#
# This page explained FALQON, as proposed by [Magann et al. (2022)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.250502), and showed how to implement it with Qamomile.
# The key points are summarized below:
#
# * FALQON is based on Quantum Lyapunov Control and feeds quantum-circuit measurement results back into the parameters of the next circuit.
# * Whereas QAOA uses a classical optimizer to search over multiple variational parameters, FALQON directly determines the next parameter from values obtained through quantum measurements.
# * Qamomile lets us directly express the problem Hamiltonian $H_p$, the driver Hamiltonian $H_d$, and the commutator $[H_d, H_p]$.
# * The `pauli_evolve` operation provides a concise expression for $e^{-i H \Delta t}$, making the correspondence between the FALQON equations and the quantum-circuit implementation clear.

# %% [markdown]
#
