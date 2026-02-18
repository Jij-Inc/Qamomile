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
# # QRAO for the Max-Cut
#
# **Quantum Random Access Optimization (QRAO)** is an approach that encodes multiple classical variables into a single qubit using Quantum Random Access Coding (QRAC). This dramatically reduces the number of qubits required compared to standard QAOA, where each variable occupies one qubit.
#
# For example, with QRAC(3,1,p) encoding, up to 3 variables are packed into 1 qubit — so a 12-variable problem that would need 12 qubits with QAOA may require only around 4 qubits with QRAO.
#
# In this tutorial, we solve the Max-Cut problem using QRAO31 (the QRAC(3,1,p) variant) with Qamomile.

# %% [markdown]
# ## QRAO Variants in Qamomile
#
# Qamomile provides several QRAO variants, each trading off encoding density and approximation quality. All converters are available from `qamomile.optimization.qrao`.
#
# | Variant | Converter Class | Variables / Qubit | Description |
# |---------|----------------|-------------------|-------------|
# | QRAC(2,1,p) | `QRAC21Converter` | up to 2 | Encodes 2 variables per qubit using Z and X operators |
# | QRAC(3,1,p) | `QRAC31Converter` | up to 3 | Encodes 3 variables per qubit using Z, X, and Y operators |
# | QRAC(3,2,p) | `QRAC32Converter` | up to 3 | Uses 2-qubit prime operators for higher fidelity |
# | Space-efficient | `QRACSpaceEfficientConverter` | 2 (fixed) | No graph coloring needed; constant 2:1 compression |
#
# In this tutorial, we use **QRAC(3,1,p)** for the highest single-qubit compression ratio.

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
num_nodes = 12
# Generalized Petersen graph GP(6,2): 12 nodes, 18 edges, 3-regular
edges = [
    # Outer hexagon
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 0),
    # Spokes
    (0, 6),
    (1, 7),
    (2, 8),
    (3, 9),
    (4, 10),
    (5, 11),
    # Inner connections (step 2)
    (6, 8),
    (8, 10),
    (10, 6),
    (7, 9),
    (9, 11),
    (11, 7),
]
G.add_nodes_from(range(num_nodes))
G.add_edges_from(edges)
pos = {
    # Outer hexagon (radius 2)
    0: (0.00, 2.00),
    1: (1.73, 1.00),
    2: (1.73, -1.00),
    3: (0.00, -2.00),
    4: (-1.73, -1.00),
    5: (-1.73, 1.00),
    # Inner hexagon (radius 0.8)
    6: (0.00, 0.80),
    7: (0.69, 0.40),
    8: (0.69, -0.40),
    9: (0.00, -0.80),
    10: (-0.69, -0.40),
    11: (-0.69, 0.40),
}

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
# ## Converting Compiled Instance to QRAO31 Hamiltonian and VQE Circuit
#
# We generate the QRAO31-encoded Hamiltonian from the compiled Instance. The converter used for this is `QRAC31Converter` from `qamomile.optimization.qrao.qrao31`.
#
# QRAO31 uses Quantum Random Access Coding (QRAC) to encode up to 3 classical variables into a single qubit using different Pauli operators (X, Y, Z). The converter internally performs:
#
# 1. **Graph coloring** on the variable interaction graph to group non-adjacent variables
# 2. **Pauli assignment** — each variable in a group is mapped to a distinct Pauli operator (Z, X, or Y) on the same qubit
# 3. **Hamiltonian relaxation** — the original Ising Hamiltonian is rewritten in terms of the encoded Pauli operators
#
# We can then use `get_cost_hamiltonian()` to obtain the encoded Hamiltonian and build a VQE ansatz to find its ground state.

# %%
import qamomile.circuit as qmc
from qamomile.optimization.qrao.qrao31 import QRAC31Converter
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# Create the QRAO31 converter
converter = QRAC31Converter(instance)

# %% [markdown]
# ### Qubit Reduction
#
# The key advantage of QRAO is the reduction in qubit count. Let's see how many qubits are needed compared to QAOA (which requires one qubit per variable).

# %%
print(f"Number of classical variables: {num_nodes}")
print(f"Number of qubits (QAOA):      {num_nodes}")
print(f"Number of qubits (QRAO31):    {converter.num_qubits}")
print(
    f"Compression ratio:            {converter.num_qubits}/{num_nodes} = {converter.num_qubits / num_nodes:.0%}"
)

# %% [markdown]
# The converter also exposes the variable-to-qubit mapping via `color_group` (which qubit each variable is assigned to) and `pauli_encoding` (which Pauli operator represents each variable).

# %%
print("Color groups (qubit -> variables):")
for qubit_idx, var_indices in converter.color_group.items():
    print(f"  Qubit {qubit_idx}: variables {var_indices}")

print("\nPauli encoding (variable -> (qubit, Pauli)):")
for var_idx, pauli_op in converter.pauli_encoding.items():
    print(f"  Variable {var_idx} -> {pauli_op}")

# %% [markdown]
# Let's inspect the cost Hamiltonian. Unlike QAOA which uses only Pauli-Z operators on $n$ qubits (one per variable), the QRAC-encoded Hamiltonian acts on a much smaller number of qubits and uses mixed Pauli operators (X, Y, Z) because each qubit hosts multiple variables.

# %%
hamiltonian = converter.get_cost_hamiltonian()
hamiltonian

# %% [markdown]
# Now we build a hardware-efficient VQE ansatz to search for the ground state of this Hamiltonian.

# %%
from qamomile.circuit.algorithm.basic import cz_entangling_layer, ry_layer, rz_layer

depth = 2


@qmc.qkernel
def vqe(
    n: qmc.UInt, h: qmc.Observable, depth: qmc.UInt, theta: qmc.Vector[qmc.Float]
) -> qmc.Float:
    q = qmc.qubit_array(n, "q")
    for layer in qmc.range(depth - 1):
        q = ry_layer(q, theta, 2 * n * layer)
        q = rz_layer(q, theta, 2 * n * layer + n)
        q = cz_entangling_layer(q)

    q = ry_layer(q, theta, 2 * n * (depth - 1))
    q = rz_layer(q, theta, 2 * n * (depth - 1) + n)
    return qmc.expval(q, h)


executable = transpiler.transpile(
    vqe,
    bindings={
        "n": hamiltonian.num_qubits,
        "h": hamiltonian,
        "depth": depth,
    },
    parameters=["theta"],
)

# %% [markdown]
# Let's look at the generated quantum circuit.

# %%
qiskit_circuit = executable.get_first_circuit()
qiskit_circuit.draw()

# %% [markdown]
# ## VQE Optimization
#
# Now we set up the variational optimization loop. We use scipy's COBYLA optimizer
# to find the optimal VQE parameters.

# %%
from scipy.optimize import minimize

# Calculate parameter count
n_qubits = hamiltonian.num_qubits
n_params = (
    2 * n_qubits * depth
)  # ry_layer + rz_layer each consume n_qubits params per layer

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

print("Starting VQE optimization...")
print(f"Initial parameter count: {len(init_params)}")

# Optimize with COBYLA
result_opt = minimize(
    objective_function,
    init_params,
    args=(executable,),
    method="COBYLA",
    options={"maxiter": 100, "disp": True},
)

print("\nOptimization complete")
print(f"Final energy: {result_opt.fun:.4f}")

# %% [markdown]
# ## Visualizing Optimization Results
#
# Let's visualize the convergence of the optimization process.
#
# > **Note:** The energy values are negative because Qamomile internally converts the maximization problem into a minimization problem.

# %%
plt.figure(figsize=(10, 5))
plt.plot(energy_history, marker="o", markersize=3)
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("VQE Optimization Convergence")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## QRAO31 Decoding Process
#
# In standard QAOA, each variable corresponds to one qubit, so we can directly read out solutions by sampling bitstrings. In QRAO, however, multiple variables share a single qubit — we cannot simply measure the qubit in the computational basis to recover all variable values simultaneously.
#
# Instead, QRAO uses a two-stage decoding process:
#
# 1. **Expectation value measurement**: For each variable $x_i$, measure the expectation value $\langle P_i \rangle$ of its assigned Pauli operator $P_i \in \{X, Y, Z\}$ on the optimized quantum state.
# 2. **Rounding**: Convert continuous expectation values into discrete spin values using a rounding scheme.
#
# Since the Pauli operators on a single qubit do not commute (e.g., $[X, Y] \neq 0$), the expectation values must be estimated from separate measurement circuits — one for each Pauli basis.

# %%
# Create circuits to measure expectation values of Pauli operators for each variable
pauli_observables = converter.get_encoded_pauli_list()

print(f"Number of Pauli operators to measure: {len(pauli_observables)}")
print("Variable index to Pauli operator mapping:")
for idx, pauli_op in converter.pauli_encoding.items():
    print(f"  Variable {idx} -> {pauli_op}")


# %%
# Measure expectation values of each Pauli operator
@qmc.qkernel
def measure_pauli(
    n: qmc.UInt, h: qmc.Observable, depth: qmc.UInt, theta: qmc.Vector[qmc.Float]
) -> qmc.Float:
    q = qmc.qubit_array(n, "q")
    for layer in qmc.range(depth - 1):
        q = ry_layer(q, theta, 2 * n * layer)
        q = rz_layer(q, theta, 2 * n * layer + n)
        q = cz_entangling_layer(q)

    q = ry_layer(q, theta, 2 * n * (depth - 1))
    q = rz_layer(q, theta, 2 * n * (depth - 1) + n)
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
        parameters=["theta"],
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
# ## Recovering the Solution: Rounding
#
# The expectation values $\langle P_i \rangle$ are continuous numbers in $[-1, 1]$, but we need discrete spin values $s_i \in \{+1, -1\}$. This is where **rounding** comes in.
#
# `SignRounder` applies the simplest rounding rule:
#
# $$
# s_i = \begin{cases} +1 & \text{if } \langle P_i \rangle \geq 0 \\ -1 & \text{if } \langle P_i \rangle < 0 \end{cases}
# $$
#
# Because the QRAC-encoded Hamiltonian is a **relaxation** of the original problem (the encoding is not exact), rounding can introduce a gap between the relaxed optimum and the recovered solution. This is inherent to the QRAO approach — we trade exactness for a dramatic reduction in qubit count.

# %%
from qamomile.optimization.qrao import SignRounder

rounder = SignRounder()
spins = rounder.round(expectations)

print("Recovered spin values (+1 or -1):")
for i, spin in enumerate(spins):
    print(f"  Variable {i}: ⟨P⟩ = {expectations[i]:+.4f}  →  spin = {spin:+d}")

# Convert spins to binary (spin=+1 -> binary=0, spin=-1 -> binary=1)
binary_solution = [(1 - s) // 2 for s in spins]

print("\nBinary solution (0 or 1):")
for i, bit in enumerate(binary_solution):
    print(f"  Variable {i}: {bit}")

# %% [markdown]
# ## Visualizing the Solution
#
# Let's visualize the best solution found by QRAO31 on the original graph.

# %%
# Convert solution to dictionary format
solution_dict = {(i,): float(bit) for i, bit in enumerate(binary_solution)}


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


# Calculate number of cut edges
def calculate_maxcut_value(graph, binary_solution):
    """Calculate the objective function value for the MaxCut problem"""
    cut_count = 0
    for u, v in graph.edges():
        if binary_solution[u] != binary_solution[v]:
            cut_count += 1
    return cut_count


cut_value = calculate_maxcut_value(G, binary_solution)

print("Solution found:")
print(f"  Binary string: {''.join(map(str, binary_solution))}")
print(f"  Number of cut edges: {cut_value}")

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
plt.show()

# %% [markdown]
# ## Comparison with the Exact Solution
#
# Since Qamomile's converters accept `ommx.v1.Instance`, we can easily compare
# our quantum result with a classical solver. Let's solve the same instance
# exactly with SCIP and see how the QRAO solution compares.

# %%
from ommx_pyscipopt_adapter import OMMXPySCIPOptAdapter

solution = OMMXPySCIPOptAdapter.solve(instance)

print(f"Exact optimal value (Max-Cut): {int(solution.objective)}")
print(f"QRAO solution value:           {cut_value}")

# %% [markdown]
# ## Summary
#
# In this tutorial, we demonstrated how to solve the Max-Cut problem using QRAO31 with Qamomile:
#
# 1. **Problem Formulation**: We formulated Max-Cut as an Ising problem using JijModeling
# 2. **QRAO31 Encoding**: `QRAC31Converter` encoded up to 3 variables per qubit via graph coloring and Pauli assignment, reducing the qubit count from 12 (QAOA) to just a few
# 3. **VQE Optimization**: We found optimal parameters using a hardware-efficient ansatz on the reduced-size Hamiltonian
# 4. **Decoding**: Since multiple variables share a qubit, we measured Pauli expectation values and applied `SignRounder` to recover discrete spin values
# 5. **Solution Analysis**: We visualized the recovered Max-Cut solution on the original graph
#
# The key takeaway is that QRAO enables solving larger combinatorial optimization problems on near-term quantum hardware by significantly reducing the number of qubits required. And since Qamomile uses `ommx.v1.Instance`, it is straightforward to benchmark against classical solvers.
