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
# # QAOA for the Max-Cut
#
# In this section, we will solve the Maxcut Problem using QAOA with the help of the JijModeling and Qamomile libraries.
#
# First, let’s install and import the main libraries we will be using.

# %%
import jijmodeling as jm
import matplotlib.pyplot as plt

# %% [markdown]
# ## What is the Max-Cut Problem
#
# The Max-Cut problem is the problem of dividing the nodes of a graph into two groups such that the number of edges cut (or the total weight of the edges cut, if the edges have weights) is maximized. Applications include network partitioning and image processing (segmentation), among others.
# %%
import networkx as nx
import numpy as np

G = nx.Graph()
num_nodes = 5
edges = [(0, 1), (0, 4), (1, 2), (1, 3), (2, 3), (3, 4)]
G.add_nodes_from(range(num_nodes))
G.add_edges_from(edges)
pos = {0: (1, 1), 1: (0, 1), 2: (-1, 0.5), 3: (0, 0), 4: (1, 0)}

fig, ax = plt.subplots(figsize=(5, 4))
ax.set_title("Original Graph G=(V,E)")
nx.draw_networkx(G, pos, ax=ax, node_size=500, width=3, with_labels=True)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Constructing the Mathematical Model
#
# The Max-Cut problem can be formulated with the following equation:
#
# $$
#   \max \quad \frac{1}{2} \sum_{(i,j) \in E} (1 - s_i s_j)
# $$
#
# Note that this equation is expressed using Ising variables $ s \in \{ +1, -1 \} $. In this case, we want to formulate it using the binary variables $ x \in \{ 0, 1 \} $ from JijModeling. Therefore, we perform the conversion between Ising variables and binary variables using the following equations:
#
# $$
#     x_i = \frac{1 + s_i}{2} \quad \Rightarrow \quad s_i = 2x_i - 1
# $$
#


# %%
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

# %% [markdown]
# ## Creating a Compiled Instance
# We compile the mathematical model together with the instance data using `problem.eval()`. This process yields an intermediate representation of the problem with the instance data substituted.

# %%
V = num_nodes
E = edges
data = {"V": V, "E": E}
instance = problem.eval(data)

# %% [markdown]
# ## Converting Compiled Instance to QAOA Circuit and Hamiltonian
#
# We generate the QAOA circuit and Hamiltonian from the compiled Instance. The converter used for this is `QAOAConverter` from `qamomile.optimization.qaoa`.
#
# By creating an instance of this class, the Ising Hamiltonian is internally generated from the compiled Instance. We can then:
# - Use `transpile()` to generate the QAOA quantum circuit
# - Use `get_cost_hamiltonian()` to inspect the cost Hamiltonian
#
# The number of QAOA layers, $p$, is set to $3$ here.

# %%
from qamomile.optimization.qaoa import QAOAConverter
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# Create the QAOA converter and transpile
p = 3  # Number of QAOA layers
converter = QAOAConverter(instance)
executable = converter.transpile(
    transpiler=transpiler,
    p=p,
)

# %% [markdown]
# Let's inspect the cost Hamiltonian. Since the Max-Cut objective is expressed with Ising variables $s_i \in \{+1, -1\}$, the cost Hamiltonian consists of Pauli-Z operators.

# %%
cost_hamiltonian = converter.get_cost_hamiltonian()
cost_hamiltonian

# %% [markdown]
# Our graph has edges $E = \{(0,1),(0,4),(1,2),(1,3),(2,3),(3,4)\}$. Since the Max-Cut objective in Ising form is $\frac{1}{2}\sum_{(i,j) \in E}(1 - Z_i Z_j)$, the cost Hamiltonian should contain $Z_i Z_j$ terms for each edge. Indeed, we can confirm that the Hamiltonian above matches the expected Ising formulation.

# %% [markdown]
# Let's look at the generated quantum circuit. The circuit implements the QAOA ansatz with alternating cost and mixer layers.

# %%
qiskit_circuit = executable.get_first_circuit()
qiskit_circuit.draw()

# %% [markdown]
# ## VQE Optimization
#
# Now we set up the variational optimization loop. We use scipy's COBYLA optimizer
# to find the optimal QAOA parameters (gamma and beta for each layer).

# %%
from scipy.optimize import minimize

# List to save optimization history
energy_history = []


def objective_function(params, transpiler, executable, converter, shots=1024):
    """
    Objective function for VQE optimization.

    Args:
        params: Concatenated [gammas, betas] parameters
        transpiler: Quantum transpiler
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
        transpiler.executor(),
        bindings={
            "gammas": gammas,
            "betas": betas,
        },
        shots=shots,
    )
    result = job.result()

    sampleset = converter.decode(result)
    energy = sampleset.energy_mean()
    energy_history.append(energy)

    return energy


# %%
# Run optimization
np.random.seed(901)

# Initial parameters: gamma in [0, 2π], beta in [0, π]
init_params = np.concatenate(
    [
        np.random.uniform(0, 2 * np.pi, size=p),  # gammas
        np.random.uniform(0, np.pi, size=p),  # betas
    ]
)

# Clear history
energy_history = []

print(f"Starting QAOA optimization with p={p} layers...")
print(f"Initial parameters: gammas={init_params[:p]}, betas={init_params[p:]}")

# Optimize with COBYLA method
result_opt = minimize(
    objective_function,
    init_params,
    args=(transpiler, executable, converter),
    method="COBYLA",
    options={"maxiter": 100, "disp": True},
)

print("\nOptimized parameters:")
print(f"  gammas: {result_opt.x[:p]}")
print(f"  betas: {result_opt.x[p:]}")
print(f"Final energy: {result_opt.fun:.4f}")

# %% [markdown]
# ## Visualizing Optimization Results
#
# Let's visualize the convergence of the optimization process.
#
# > **Note:** The energy values are negative because Qamomile internally converts the maximization problem into a minimization problem. An energy of $-5$ corresponds to a Max-Cut objective value of $5$.

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
# ## Final Solution Analysis
#
# Now let's sample from the optimized circuit and analyze the results.

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
    shots=4096,
)
result_final = job_final.result()

# Decode results using the converter
sampleset = converter.decode(result_final)

# Build frequency distribution over all sampled bitstrings
bitstrings = []
counts = []
energies = []
for i in range(len(sampleset.samples)):
    sample = sampleset.samples[i]
    bitstring_str = "".join(str(sample[j]) for j in range(num_nodes))
    bitstrings.append(bitstring_str)
    counts.append(sampleset.num_occurrences[i])
    energies.append(sampleset.energy[i])

# Sort by bitstring for consistent display
sorted_order = np.argsort(bitstrings)
bitstrings = [bitstrings[i] for i in sorted_order]
counts = [counts[i] for i in sorted_order]
energies = [energies[i] for i in sorted_order]

# Plot frequency distribution
fig, ax = plt.subplots(figsize=(12, 5))
x_pos = np.arange(len(bitstrings))
bars = ax.bar(x_pos, counts)

# Highlight optimal solutions (energy = -5) with red bars
for i, e in enumerate(energies):
    if np.isclose(e, -5.0):
        bars[i].set_color("red")

ax.set_xticks(x_pos)
ax.set_xticklabels(bitstrings, rotation=90)
ax.set_xlabel("Bitstring")
ax.set_ylabel("Counts")
ax.set_title("QAOA Measurement Frequency Distribution (red = optimal, energy = -5)")
plt.tight_layout()
plt.show()

# %% [markdown]
# The red bars indicate bitstrings with energy $= -5$, which correspond to the optimal Max-Cut solutions (cutting 5 out of 6 edges). The frequency distribution shows that QAOA successfully concentrates measurement probability on these optimal solutions.

# %% [markdown]
# ## Visualizing the Solution
#
# Let's visualize the best solution found by QAOA on the original graph.

# %%
# Get the best solution (lowest energy)
best_sample, best_energy, best_count = sampleset.lowest()
best_solution = {(i,): float(best_sample[i]) for i in range(num_nodes)}

print("Best solution found:")
print(f"  Bitstring: {''.join(str(best_sample[i]) for i in range(num_nodes))}")
print(f"  Energy: {best_energy:.4f}")


# Visualize the solution
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
    node_colors = [
        "#2696EB" if node in cut_set_1 else "#EA9b26" for node in graph.nodes()
    ]
    return edge_colors, node_colors


edge_colors, node_colors = get_edge_colors(G, best_solution)
cut_edges = sum(1 for c in edge_colors if c == "r")

fig, ax = plt.subplots(figsize=(6, 5))
ax.set_title(f"QAOA Solution (Cut edges: {cut_edges})")
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
plt.show()

# %% [markdown]
# ## Comparison with the Exact Solution
#
# Since Qamomile's converters accept `ommx.v1.Instance`, we can easily compare
# our quantum result with a classical solver. Let's solve the same instance
# exactly with SCIP and see how the QAOA solution compares.

# %%
from ommx_pyscipopt_adapter import OMMXPySCIPOptAdapter

solution = OMMXPySCIPOptAdapter.solve(instance)

print(f"Exact optimal value (Max-Cut): {int(solution.objective)}")
print(f"QAOA solution value:           {cut_edges}")

# %% [markdown]
# ## Summary
#
# In this tutorial, we demonstrated how to solve the Max-Cut problem using QAOA with Qamomile:
#
# 1. **Problem Formulation**: We formulated Max-Cut as an Ising problem using JijModeling
# 2. **Hamiltonian & Circuit Generation**: `QAOAConverter` automatically generated the cost Hamiltonian and QAOA circuit
# 3. **VQE Optimization**: We used scipy's COBYLA optimizer to find optimal QAOA parameters with Qamomile
# 4. **Solution Analysis**: The frequency distribution confirmed that QAOA concentrates measurement probability on the optimal Max-Cut solutions
