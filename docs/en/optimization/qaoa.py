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
import ommx.v1
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ## What is the Max-Cut Problem
#
# The Max-Cut problem is the problem of dividing the nodes of a graph into two groups such that the number of edges cut (or the total weight of the edges cut, if the edges have weights) is maximized. Applications include network partitioning and image processing (segmentation), among others.

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
# Next, we will solve the Max-Cut Problem for the following graph. The data for the specific problem being solved is referred to as instance data.

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
# We perform compilation using `JijModeling.Interpreter` and `ommx.Instance` by providing the formulation and the instance data prepared earlier. This process yields an intermediate representation of the problem with the instance data substituted.

# %%
interpreter = jm.Interpreter(data)
instance = interpreter.eval_problem(problem)

# %% [markdown]
# ## Converting Compiled Instance to QAOA Circuit and Hamiltonian
#
# We generate the QAOA circuit and Hamiltonian from the compiled Instance. The converter used to generate these is `qm.optimization.qaoa.QAOAConverter`.
#
# By creating an instance of this class and using `ising_encode`, you can internally generate the Ising Hamiltonian from the compiled Instance. Parameters that arise during the conversion to QUBO can also be set here. If not set, default values are used.
#
# Once the Ising Hamiltonian is generated, you can generate the QAOA quantum circuit and the Hamiltonian respectively. These can be executed using the `get_qaoa_ansatz` and `get_cost_hamiltonian` methods. The number of QAOA layers, $p$, is fixed to be $7$ here.  

# %%
import qamomile.circuit as qmc
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
# Let's look at the generated quantum circuit. The circuit implements the QAOA ansatz with alternating cost and mixer layers.

# %%
qiskit_circuit = executable.get_first_circuit()
if qiskit_circuit is not None:
    print(f"Number of qubits: {qiskit_circuit.num_qubits}")
    print(f"Circuit depth: {qiskit_circuit.depth()}")

# %% [markdown]
# ## Energy Calculation
#
# To optimize QAOA, we need to calculate the expected energy from measurement results.
# The converter provides access to the Ising model, which we use to calculate energy.

# %%
def calculate_ising_energy(bitstring: list[int], ising_model) -> float:
    """
    Calculate the Ising model energy from a bitstring.

    Converts bitstring z_i ∈ {0, 1} to spin s_i ∈ {-1, +1}.
    Convention: s_i = 1 - 2*z_i (z_i=0 → s_i=+1, z_i=1 → s_i=-1)
    """
    # Convert bits to spins
    spins = [1 - 2 * b for b in bitstring]
    return ising_model.calc_energy(spins)


def calculate_expectation_value(sample_result, ising_model) -> float:
    """
    Calculate the expected energy value from measurement results.
    """
    total_energy = 0.0
    total_counts = 0

    for bitstring, count in sample_result.results:
        energy = calculate_ising_energy(bitstring, ising_model)
        total_energy += energy * count
        total_counts += count

    return total_energy / total_counts


# %% [markdown]
# ## VQE Optimization
#
# Now we set up the variational optimization loop. We use scipy's COBYLA optimizer
# to find the optimal QAOA parameters (gamma and beta for each layer).

# %%
from scipy.optimize import minimize

# List to save optimization history
energy_history = []


def objective_function(params, transpiler, executable, ising_model, shots=1024):
    """
    Objective function for VQE optimization.

    Args:
        params: Concatenated [gammas, betas] parameters
        transpiler: Quantum transpiler
        executable: Compiled QAOA circuit
        ising_model: Ising model for energy calculation
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

    energy = calculate_expectation_value(result, ising_model)
    energy_history.append(energy)

    return energy


# %%
# Run optimization
np.random.seed(42)

# Initial parameters: gamma in [0, 2π], beta in [0, π]
init_params = np.concatenate([
    np.random.uniform(0, 2 * np.pi, size=p),  # gammas
    np.random.uniform(0, np.pi, size=p),       # betas
])

# Clear history
energy_history = []

print(f"Starting QAOA optimization with p={p} layers...")
print(f"Initial parameters: gammas={init_params[:p]}, betas={init_params[p:]}")

# Optimize with COBYLA method
result_opt = minimize(
    objective_function,
    init_params,
    args=(transpiler, executable, converter.ising),
    method="COBYLA",
    options={"maxiter": 100, "disp": True},
)

print(f"\nOptimized parameters:")
print(f"  gammas: {result_opt.x[:p]}")
print(f"  betas: {result_opt.x[p:]}")
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
plt.title("QAOA Optimization Convergence")
plt.grid(True)
plt.tight_layout()
# plt.show()

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

# Sort results by energy
results_with_energy = []
for bitstring, count in result_final.results:
    energy = calculate_ising_energy(bitstring, converter.ising)
    results_with_energy.append((bitstring, count, energy))

results_with_energy.sort(key=lambda x: x[2])

print("Measurement results (sorted by energy):")
print("-" * 60)
for bitstring, count, energy in results_with_energy[:10]:
    bitstring_str = "".join(map(str, bitstring))
    probability = count / 4096
    print(f"  {bitstring_str}: count={count:4d}, probability={probability:.3f}, energy={energy:.4f}")

# %% [markdown]
# ## Visualizing the Solution
#
# Let's visualize the best solution found by QAOA on the original graph.

# %%
# Get the best solution (lowest energy)
best_bitstring, best_count, best_energy = results_with_energy[0]
best_solution = {(i,): float(bit) for i, bit in enumerate(best_bitstring)}

print(f"\nBest solution found:")
print(f"  Bitstring: {''.join(map(str, best_bitstring))}")
print(f"  Energy: {best_energy:.4f}")

# Visualize the solution
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
# plt.show()

# %% [markdown]
# ## Summary
#
# In this tutorial, we demonstrated how to solve the Max-Cut problem using QAOA with Qamomile:
#
# 1. **Problem Formulation**: We formulated Max-Cut as a QUBO/Ising problem using JijModeling
# 2. **Circuit Generation**: Qamomile's `QAOAConverter` automatically generated the QAOA circuit
# 3. **VQE Optimization**: We used scipy's COBYLA optimizer to find optimal QAOA parameters
# 4. **Solution Analysis**: We analyzed the measurement results and visualized the solution
#
# Key advantages of using Qamomile for QAOA:
# - Automatic Ising model generation from mathematical formulations
# - Backend-agnostic circuit definition (works with Qiskit, Quri-Parts, PennyLane, etc.)
# - Easy integration with classical optimization libraries
