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
#     name: qamomile
# ---

# %% [markdown]
# # qBraid Support - QBraidExecutor
#
# In this section, we will explain what qBraid is and our QBraidExecutor.
# This must contain our support is from Qiskit, so if users want to use qBraid,
# They need to transpile their qamomile circuit with QiskitTranspiler.

# %% [markdown]
# ## Installation
#
# Write how to install qbraid executor support.
# This must be based on the current pyproject.toml.

# %%
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import minimize

import qamomile.circuit as qmc
from qamomile.circuit.algorithm.qaoa import qaoa_state
from qamomile.optimization.binary_model.expr import VarType
from qamomile.optimization.binary_model.model import BinaryModel
from qamomile.qbraid import QBraidExecutor
from qamomile.qiskit import QiskitTranspiler

seed = 901
rng = np.random.default_rng(seed)

# %% [markdown]
# ## QBraid Executor
# Write the description to explain what is qbraid and how to use it with qamomile.

# %%
# device_id = "qbraid:qbraid:sim:qir-sv"
# api_key = "YOUR_API_KEY"
# qbraid_executor = QBraidExecutor(
#     device_id=device_id,
#     api_key=api_key,
# )

# %% [markdown]
# ## MaxCut Example
# Write the maxcut explanation.

# %% [markdown]
# ### Constructing the problem
# Write about the networkxs to create a random graph.
# %%
# Create a random graph using networkx
graph = nx.random_regular_graph(d=3, n=8, seed=0)

# Visualize the random graph
fig, ax = plt.subplots(figsize=(6, 5))
ax.set_title("Graph")
pos = nx.spring_layout(graph, seed=seed)
nx.draw_networkx(graph, pos, ax=ax, node_size=500, width=2, with_labels=True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Constructing the Ising Hamiltonian
# Write the discription to explain how to construct the Ising Hamiltonian from the graph as a minimization problem.

# %%
quad = {}
linear = {node: 0.0 for node in graph.nodes()}
constant = 0.0

for u, v, data in graph.edges(data=True):
    key = (u, v) if u <= v else (v, u)
    quad[key] = quad.get(key, 0.0) + 1 / 2
    constant -= 1 / 2

spin_model = BinaryModel.from_ising(linear=linear, quad=quad, constant=constant)
spin_model_normalized = spin_model.normalize_by_rms()
spin_model_normalized._expr


# %% [markdown]
# ### Constructing the QAOA circuit
# Write the description to explain how to construct the QAOA circuit with qamomile.


# %%
@qmc.qkernel
def qaoa_circuit(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    n: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    q = qaoa_state(p=p, quad=quad, linear=linear, n=n, gammas=gammas, betas=betas)
    return qmc.measure(q)


# %%
transpiler = QiskitTranspiler()
p = 5  # Number of QAOA layers

executable = transpiler.transpile(
    qaoa_circuit,
    bindings={
        "p": p,
        "quad": spin_model_normalized.quad,
        "linear": spin_model_normalized.linear,
        "n": len(graph.nodes()),
    },
    parameters=["gammas", "betas"],
)

# %% [markdown]
# ### Optimization
# Write the description to explain how to optimize the parameters with the qbraid executor.

# %%
# List to save optimization history
energy_history = []
# Convert the spin model into the corresponding binary model to evaluate
# the energy of the sampled bitstrings
binary_model = spin_model_normalized.change_vartype(VarType.BINARY)


# Define the objective function for optimization
def objective_function(params, executable, executor, shots=2000):
    """
    Objective function for VQE optimization.

    Args:
        params: Concatenated [gammas, betas] parameters
        executable: Compiled QAOA circuit
        converter: QAOAConverter for decoding results
        shots: Number of measurement shots

    Returns:
        Expected energy value
    """
    p = len(params) // 2
    gammas = params[:p]
    betas = params[p:]

    job = executable.sample(
        executor,
        bindings={
            "gammas": gammas,
            "betas": betas,
        },
        shots=shots,
    )
    result = job.result()

    energies = []
    for bit_list, counts in result.results:
        energy = binary_model.calc_energy(bit_list)
        for _ in range(counts):
            energies.append(energy)
    energy_avg = np.average(energies)
    energy_history.append(energy_avg)

    return energy_avg


# %%
init_params = rng.uniform(low=-np.pi / 4, high=np.pi / 4, size=2 * p)

# Clear history
energy_history = []

print(f"Starting QAOA optimization with p={p} layers...")
print(f"Initial parameters: gammas={init_params[:p]}, betas={init_params[p:]}")

# Optimize with COBYLA method
result_opt = minimize(
    objective_function,
    init_params,
    # QBraidExecutor is used as the executor for sampling in the objective function
    args=(executable, transpiler.executor()),
    method="Nelder-Mead",
    options={"disp": True},
)

print("\nOptimized parameters:")
print(f"  gammas: {result_opt.x[:p]}")
print(f"  betas: {result_opt.x[p:]}")
print(f"Final energy: {result_opt.fun:.4f}")

# %%
plt.figure(figsize=(10, 5))
plt.plot(energy_history, marker="o", markersize=3)
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("QAOA Optimization Convergence")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Evaluation
# Write the description to explain how to evaluate the optimized parameters with the qbraid executor.

# %%
# Sample with optimized parameters
optimal_gammas = result_opt.x[:p]
optimal_betas = result_opt.x[p:]

job_final = executable.sample(
    transpiler.executor(),
    bindings={
        "gammas": optimal_gammas,
        "betas": optimal_betas,
    },
    shots=2000,
)
result_final = job_final.result()

# %%
# Build frequency distribution over all sampled bitstrings
energy_vs_counts = defaultdict(int)
lowest_energy = float("inf")
best_solution = None
for bit_list, counts in result_final.results:
    energy = binary_model.calc_energy(bit_list)
    energy_vs_counts[energy] += counts
    if energy < lowest_energy:
        lowest_energy = energy
        best_solution = bit_list

# Extract energies and counts for plotting
energies = list(energy_vs_counts.keys())
counts = list(energy_vs_counts.values())
# Sort by energy for better visualization
sorted_indices = np.argsort(energies)[::-1]
counts = np.array(counts)[sorted_indices]
energies = np.array(energies)[sorted_indices]

# Plot frequency distribution
fig, ax = plt.subplots(figsize=(12, 5))
x_pos = np.arange(len(energy_vs_counts))
bars = ax.bar(x_pos, counts)

ax.set_xticks(x_pos)
ax.set_xticklabels(energies, rotation=90)
ax.set_xlabel("Energy")
ax.set_ylabel("Counts")
ax.set_title("QAOA Measurement Frequency Distribution")
plt.tight_layout()
plt.show()


# %%
# Define functions to visualize the solution on the graph.
def get_edge_colors(
    graph, solution: list[int], in_cut_color: str = "r", not_in_cut_color: str = "b"
) -> tuple[list[str], list[str]]:
    cut_set_1 = [node for node, value in enumerate(solution) if value == 1.0]
    cut_set_2 = [node for node in graph.nodes() if node not in cut_set_1]

    edge_colors = []
    for u, v, _ in graph.edges(data=True):
        if (u in cut_set_1 and v in cut_set_2) or (u in cut_set_2 and v in cut_set_1):
            edge_colors.append(in_cut_color)
        else:
            edge_colors.append(not_in_cut_color)
    node_colors = [
        "#2696EB" if node in cut_set_1 else "#EA9b26" for node in graph.nodes()
    ]
    return edge_colors, node_colors


def show_solution(graph, solution, title):
    edge_colors, node_colors = get_edge_colors(graph, solution)
    cut_edges = sum(1 for c in edge_colors if c == "r")
    _, ax = plt.subplots(figsize=(6, 5))
    ax.set_title(f"{title} (Cut edges: {cut_edges})")
    nx.draw_networkx(
        graph,
        pos,
        ax=ax,
        node_size=500,
        width=3,
        with_labels=True,
        edge_color=edge_colors,
        node_color=node_colors,
    )
    plt.tight_layout()
    plt.show()


# %%
# Visualize the best solution found by QAOA
show_solution(graph, best_solution, "QAOA")

# %% [markdown]
# ## Additional Notes
# From here, we will use JijModeling and OMMXPySCIPOptAdapter, which is a SCIP adapter for OMMX, to solve the same problem as a classical optimization problem and compare the results with QAOA. This additional part shows if the above QAOA optimization with qBraidExecutor could find good parameters to sample low-energy states.

# %%
import jijmodeling as jm
from ommx_pyscipopt_adapter import OMMXPySCIPOptAdapter

# %%
# Define the MaxCut problem using JijModeling
problem = jm.Problem("Maxcut", sense=jm.ProblemSense.MAXIMIZE)


@problem.update
def _(problem: jm.DecoratedProblem):
    V = problem.Dim()
    E = problem.Graph()
    x = problem.BinaryVar(shape=(V,))

    obj = (
        E.rows()
        .map(lambda e: 1 / 2 * (1 - (2 * x[e[0]] - 1) * (2 * x[e[1]] - 1)))
        .sum()
    )
    problem += obj


problem

# %%
# Create OMMX format with the defined graph
V = graph.number_of_nodes()
E = graph.edges()
data = {"V": V, "E": E}
print(data)
instance = problem.eval(data)

# %%
# Solve the problem with OMMXPySCIPOptAdapter
scip_solution = OMMXPySCIPOptAdapter.solve(instance)
scip_solution_entries = [bit for bit in scip_solution.state.entries.values()]
print(f"SCIP Solution: {scip_solution_entries}")

# %%
# Visualize the solution found by SCIP
show_solution(graph, scip_solution_entries, "SCIP")

# %%
# Visualize the best solution found by QAOA again for comparison
show_solution(graph, best_solution, "QAOA")
