# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: qamomile
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Solving the MaxCut Problem with QRAO
#
# In this section, we use JijModeling and Qamomile to solve the MaxCut problem with QRAO.
#
# First, we import the main libraries we'll be using.

# %%
import jijmodeling as jm
import ommx.v1
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ## What is the MaxCut Problem?
#
# The MaxCut problem involves partitioning the nodes of a graph into two groups to maximize
# the number of edges that are cut (or the total weight of cut edges if edges are weighted).
# Applications include network partitioning and image processing (segmentation).

# %%
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
num_nodes = 5
edges = [(0, 1), (0, 4), (1, 2), (1, 3), (2, 3), (3, 4)]
G.add_nodes_from(range(num_nodes))
G.add_edges_from(edges)
pos = {0: (1, 1), 1: (0, 1), 2: (-1, 0.5), 3: (0, 0), 4: (1, 0)}

cut_solution = {(1,): 1.0, (2,): 1.0, (4,): 1.0}
edge_colors = []


def get_edge_colors(
    graph, cut_solution, in_cut_color="r", not_in_cut_color="b"
) -> tuple[list[str], list[str]]:
    cut_set_1 = [node[0] for node, value in cut_solution.items() if value == 1.0]
    cut_set_2 = [node for node in graph.nodes() if node not in cut_set_1]

    edge_colors = []
    for u, v, _ in graph.edges(data=True):
        if (u in cut_set_1 and v in cut_set_2) or (u in cut_set_2 and v in cut_set_1):
            edge_colors.append(in_cut_color)
        else:
            edge_colors.append(not_in_cut_color)
    node_colors = ["#2696EB" if node in cut_set_1 else "#EA9b26" for node in G.nodes()]
    return edge_colors, node_colors


edge_colors, node_colors = get_edge_colors(G, cut_solution)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].set_title("Original Graph G=(V,E)")
nx.draw_networkx(G, pos, ax=axes[0], node_size=500, width=3, with_labels=True)
axes[1].set_title("MaxCut Solution Visualization")
nx.draw_networkx(
    G,
    pos,
    ax=axes[1],
    node_size=500,
    width=3,
    with_labels=True,
    edge_color=edge_colors,
    node_color=node_colors,
)

plt.tight_layout()
# plt.show()


# %% [markdown]
# ## Building the Mathematical Model
#
# The MaxCut problem can be formulated as follows:
#
# $$
#   \max \quad \frac{1}{2} \sum_{(i,j) \in E} (1 - s_i s_j)
# $$
#
# This expression uses Ising variables $ s \in \{ +1, -1 \} $. Since we want to formulate
# with JijModeling's binary variables $ x \in \{ 0, 1 \} $, we use the following
# conversion between Ising and binary variables:
#
# $$
#     x_i = \frac{1 + s_i}{2} \quad \Rightarrow \quad s_i = 2x_i - 1
# $$
#

# %%
def Maxcut_problem() -> jm.Problem:
    V = jm.Placeholder("V")
    E = jm.Placeholder("E", ndim=2)
    x = jm.BinaryVar("x", shape=(V,))
    e = jm.Element("e", belong_to=E)
    i = jm.Element("i", belong_to=V)
    j = jm.Element("j", belong_to=V)

    problem = jm.Problem("Maxcut", sense=jm.ProblemSense.MAXIMIZE)
    si = 2 * x[e[0]] - 1
    sj = 2 * x[e[1]] - 1
    si.set_latex("s_{e[0]}")
    sj.set_latex("s_{e[1]}")
    obj = 1 / 2 * jm.sum(e, (1 - si * sj))
    problem += obj
    return problem


problem = Maxcut_problem()
problem

# %% [markdown]
# ## Preparing Instance Data
#
# Next, we solve the MaxCut problem for the following graph. The data for the specific
# problem to be solved is called instance data.

# %%
import networkx as nx
import numpy as np
from IPython.display import display, Latex

G = nx.Graph()
num_nodes = 5
edges = [(0, 1), (0, 4), (1, 2), (1, 3), (2, 3), (3, 4)]
G.add_nodes_from(range(num_nodes))
G.add_edges_from(edges)

weight_matrix = nx.to_numpy_array(G, nodelist=list(range(num_nodes)))

plt.title("G=(V,E)")
plt.plot(figsize=(5, 4))

nx.draw_networkx(G, pos, node_size=500)

# %%
V = num_nodes
E = edges

data = {"V": V, "E": E}

data

# %% [markdown]
# ## Creating a Compiled Instance
# Using the formulation and instance data prepared above, we compile using
# `JijModeling.Interpreter` and `ommx.Instance`. This process yields an intermediate
# representation of the problem with instance data substituted.

# %%
interpreter = jm.Interpreter(data)
instance = interpreter.eval_problem(problem)

# %% [markdown]
# ## Converting Compiled Instance to QAOA Circuit and Hamiltonian
#
# We generate the QAOA circuit and Hamiltonian from the compiled instance. The converter
# used for this is `qm.optimization.qaoa.QAOAConverter`.
#
# Creating an instance of this class and using `ising_encode`, we can internally generate
# an Ising Hamiltonian from the compiled instance. Parameters that occur during conversion
# to QUBO can also be set here. If not set, default values are used.
#
# Once the Ising Hamiltonian is generated, we can generate the QAOA quantum circuit and
# Hamiltonian respectively. These can be created using the `get_qaoa_ansatz` and
# `get_cost_hamiltonian` methods. Here we fix the QAOA depth $p$ to 3.

# %%
import qamomile.circuit as qmc
from qamomile.optimization.qrao.qrao31 import QRAC31Converter
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# Create the QRAO31 Hamiltonian
converter = QRAC31Converter(instance)
hamiltonin = converter.get_cost_hamiltonian()


# %% [markdown]
# We will use VQE to search for the ground state of this Hamiltonian.
from qamomile.circuit.algorithm.basic import ry_layer, rz_layer, cz_entangling_layer


@qmc.qkernel
def vqe(n: qmc.UInt, h: qmc.Observable, depth: qmc.UInt, theta: qmc.Vector[qmc.Float]) -> qmc.Float:
    q = qmc.qubit_array(n, "q")
    for layer in qmc.range(depth):
        q = ry_layer(q, theta, 2 * n * layer)
        q = rz_layer(q, theta, 2 * n * layer + n)
        q = cz_entangling_layer(q)
    return qmc.expval(q, h)


transpiler = QiskitTranspiler()
executable = transpiler.transpile(
    vqe,
    bindings={
        "n": hamiltonin.num_qubits,
        "h": hamiltonin,
        "depth": 2,
    },
    parameters=["theta"]
)

# %% [markdown]
# ## VQE Optimization
#
# We set up a variational optimization loop. Using scipy's COBYLA optimizer,
# we find the optimal VQE parameters.

# %%
from scipy.optimize import minimize

# Calculate parameter count
depth = 2
n_qubits = hamiltonin.num_qubits
n_params = 2 * n_qubits * depth  # ry_layer + rz_layer each consume n_qubits params per layer

print(f"Number of qubits: {n_qubits}")
print(f"Depth: {depth}")
print(f"Number of parameters: {n_params}")

# List to store optimization history
energy_history = []


def objective_function(params, executable):
    """
    Objective function for VQE optimization.
    """
    job = executable.run(
        transpiler.executor(),
        bindings={
            "theta": params,
        },
    )
    energy = job.result()
    energy_history.append(energy)
    return energy


# %%
# Run optimization
np.random.seed(42)

# Initial parameters
init_params = np.random.uniform(0, 2 * np.pi, size=n_params)

# Clear history
energy_history = []

print(f"Starting VQE optimization...")
print(f"Initial parameter count: {len(init_params)}")

# Optimize with COBYLA
result_opt = minimize(
    objective_function,
    init_params,
    args=(executable,),
    method="COBYLA",
    options={"maxiter": 100, "disp": True},
)

print(f"\nOptimization complete")
print(f"Final energy: {result_opt.fun:.4f}")

# %% [markdown]
# ## Visualizing Optimization Results
#
# Let's visualize the convergence of the optimization process.

# %%
plt.figure(figsize=(10, 5))
plt.plot(energy_history, marker='o', markersize=3)
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("VQE Optimization Convergence")
plt.grid(True)
plt.tight_layout()
# plt.show()

# %% [markdown]
# ## QRAO31 Decoding Process
#
# In QRAO31, each variable is encoded into a specific Pauli operator (X, Y, Z).
# From the optimized state, we measure the expectation value of each Pauli operator
# to recover the original variable values.
#
# If the expectation value is positive, we infer +1 (spin representation), and if negative, -1.

# %%
# Create circuits to measure expectation values of Pauli operators for each variable
pauli_observables = converter.get_encoded_pauli_list()

print(f"Number of Pauli operators to measure: {len(pauli_observables)}")
print(f"Variable index to Pauli operator mapping:")
for idx, pauli_op in converter.pauli_encoding.items():
    print(f"  Variable {idx} -> {pauli_op}")


# %%
# Measure expectation values of each Pauli operator
@qmc.qkernel
def measure_pauli(n: qmc.UInt, h: qmc.Observable, depth: qmc.UInt, theta: qmc.Vector[qmc.Float]) -> qmc.Float:
    q = qmc.qubit_array(n, "q")
    for layer in qmc.range(depth):
        q = ry_layer(q, theta, 2 * n * layer)
        q = rz_layer(q, theta, 2 * n * layer + n)
        q = cz_entangling_layer(q)
    return qmc.expval(q, h)


expectations = []
optimal_params = result_opt.x

for i, pauli_obs in enumerate(pauli_observables):
    executable_pauli = transpiler.transpile(
        measure_pauli,
        bindings={
            "n": n_qubits,
            "h": pauli_obs,
            "depth": depth,
        },
        parameters=["theta"]
    )

    job = executable_pauli.run(
        transpiler.executor(),
        bindings={
            "theta": optimal_params,
        },
    )
    expectation = job.result()
    expectations.append(expectation)

print("Pauli expectation values for each variable:")
for i, exp in enumerate(expectations):
    print(f"  Variable {i}: {exp:.4f}")

# %% [markdown]
# ## Recovering the Solution
#
# We use SignRounder to recover the original variable values (spins) from expectation values.

# %%
from qamomile.optimization.qrao import SignRounder

rounder = SignRounder()
spins = rounder.round(expectations)

print("Recovered spin values (+1 or -1):")
for i, spin in enumerate(spins):
    print(f"  Variable {i}: {spin}")

# Convert spins to binary (spin=+1 -> binary=0, spin=-1 -> binary=1)
binary_solution = [(1 - s) // 2 for s in spins]

print("\nBinary solution (0 or 1):")
for i, bit in enumerate(binary_solution):
    print(f"  Variable {i}: {bit}")

# %% [markdown]
# ## Visualizing the Solution
#
# Let's visualize the solution found by QRAO31 on the original graph.

# %%
# Convert solution to dictionary format
solution_dict = {(i,): float(bit) for i, bit in enumerate(binary_solution)}

# Calculate energy
# Energy calculation in spin representation
def calculate_maxcut_value(graph, binary_solution):
    """Calculate the objective function value for the MaxCut problem"""
    cut_count = 0
    for u, v in graph.edges():
        if binary_solution[u] != binary_solution[v]:
            cut_count += 1
    return cut_count

cut_value = calculate_maxcut_value(G, binary_solution)

print(f"\nSolution found:")
print(f"  Binary string: {''.join(map(str, binary_solution))}")
print(f"  Number of cut edges: {cut_value}")

# Visualize solution
edge_colors, node_colors = get_edge_colors(G, solution_dict)

fig, ax = plt.subplots(figsize=(6, 5))
ax.set_title(f"QRAO31 Solution (Cut Edges: {cut_value})")
nx.draw_networkx(
    G,
    pos,
    ax=ax,
    node_size=500,
    width=3,
    with_labels=True,
    edge_color=edge_colors,
    node_color=node_colors,
)
plt.tight_layout()
# plt.show()

# %% [markdown]
# ## Summary
#
# In this tutorial, we demonstrated how to solve the MaxCut problem with QRAO31 using Qamomile:
#
# 1. **Problem formulation**: We formulated MaxCut as a QUBO/Ising problem using JijModeling
# 2. **QRAO31 encoding**: `QRAC31Converter` encoded 3 variables into 1 qubit, reducing qubit count
# 3. **VQE optimization**: We found optimal parameters using a hardware-efficient ansatz
# 4. **Decoding**: We measured expectation values of Pauli operators and recovered original variable values with SignRounder
# 5. **Solution analysis**: We analyzed the measurement results and visualized the solution
#
# Key advantages of using Qamomile for QRAO31:
# - **Qubit reduction**: Can represent up to 3x more variables with fewer qubits
# - **Automatic conversion from mathematical formulations**: Automatically generates QRAC-encoded Hamiltonian from JijModeling
# - **Backend-agnostic**: Currently Qiskit; CUDA-Q and QURI Parts coming soon
# - **Integration with classical optimization**: Easy integration with classical optimization libraries like scipy
#
# QRAO31 is a powerful technique for efficiently using the limited qubits available on NISQ devices.

# %%
