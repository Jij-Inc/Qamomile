# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# # Quantum Approximate Optimization Algorithm (QAOA)
# This tutorial explains how to implement the Quantum Approximate Optimization Algorithm (QAOA) using Qamomile.
# QAOA is a hybrid quantum-classical algorithm inspired by quantum annealing and is a heuristic for combinatorial optimization problems.
# QAOA uses parameterized quantum circuits, and the parameters are adjusted using classical optimization methods.
# In this tutorial, you will learn how to build parameterized quantum circuits and optimize them using Qamomile through QAOA.

# %% [markdown]
# ## Basic Concepts of QAOA
# QAOA consists of the following main steps:
# 1. **Problem Formulation**: Formulate the combinatorial optimization problem you want to solve. For example, the MaxCut problem.
# 2. **Quantum Circuit Construction**: Build a parameterized quantum circuit based on the problem. This includes applying the cost Hamiltonian and mixer Hamiltonian.
# 3. **Measurement**: Execute the quantum circuit and measure the results.
# 4. **Classical Optimization**: Update parameters based on measurement results and reconstruct the quantum circuit.
# 5. **Iteration**: Repeat steps 2-4 until an optimal solution is obtained.

# %% [markdown]
# ## QAOA Implementation from Scratch
# First, let's implement QAOA using Qamomile's basic quantum gates.
# We show an example of solving an energy minimization problem for a random Ising model.

# %%
import numpy as np
import qamomile.circuit as qmc


@qmc.qkernel
def qaoa_cost_operator(
    qubits: qmc.Vector[qmc.Qubit],
    edges: qmc.Matrix[qmc.UInt],
    weights: qmc.Vector[qmc.Float],
    bias: qmc.Vector[qmc.Float],
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    e = edges.shape[0]
    for _e in qmc.range(e):
        i = edges[_e, 0]
        j = edges[_e, 1]
        wij = weights[_e]
        qubits[i], qubits[j] = qmc.rzz(qubits[i], qubits[j], angle=gamma*wij)

    n = qubits.shape[0]
    for i in qmc.range(n):
        bi = bias[i]
        qubits[i] = qmc.rz(qubits[i], angle=gamma*bi)
    return qubits


# %% [markdown]
# Next, we define the QAOA Mixer operator.

# %%
@qmc.qkernel
def qaoa_mixer_operator(
    qubits: qmc.Vector[qmc.Qubit],
    beta: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    n = qubits.shape[0]
    for i in qmc.range(n):
        qubits[i] = qmc.rx(qubits[i], angle=2*beta)
    return qubits


# %% [markdown]
# Finally, we define the entire QAOA circuit.

# %%
@qmc.qkernel
def qaoa_circuit(
    edges: qmc.Matrix[qmc.UInt],
    weights: qmc.Vector[qmc.Float],
    bias: qmc.Vector[qmc.Float],
    p: int,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    n = bias.shape[0]
    qubits = qmc.qubit_array(n, name="qaoa_qubits")

    # Prepare initial state (uniform superposition)
    for i in qmc.range(n):
        qubits[i] = qmc.h(qubits[i])

    # Apply QAOA layers
    for layer in qmc.range(p):
        qubits = qaoa_cost_operator(
            qubits, edges, weights, bias, gammas[layer]
        )
        qubits = qaoa_mixer_operator(qubits, betas[layer])

    return qmc.measure(qubits)

# %% [markdown]
# ## Running QAOA with Different Quantum SDKs
#
# Qamomile supports multiple quantum SDKs. The same circuit definition works across all backends.
# Select your preferred SDK:
#
# ::::{tab-set}
# :::{tab-item} Qiskit
# :sync: sdk
#
# ```python
# from qamomile.qiskit import QiskitTranspiler
#
# transpiler = QiskitTranspiler()
# executable = transpiler.transpile(
#     qaoa_circuit,
#     bindings={"edges": edges, "weights": weights, "bias": bias, "p": 2},
#     parameters=["gammas", "betas"],
# )
#
# job = executable.sample(
#     transpiler.executor(),
#     bindings={"gammas": init_gammas, "betas": init_betas},
#     shots=1024,
# )
# result = job.result()
# ```
#
# :::
# :::{tab-item} Quri-Parts
# :sync: sdk
#
# ```python
# from qamomile.quri_parts import QuriPartsCircuitTranspiler
#
# transpiler = QuriPartsCircuitTranspiler()
# executable = transpiler.transpile(
#     qaoa_circuit,
#     bindings={"edges": edges, "weights": weights, "bias": bias, "p": 2},
#     parameters=["gammas", "betas"],
# )
#
# # Requires quri-parts-qulacs for simulation
# job = executable.sample(
#     transpiler.executor(),
#     bindings={"gammas": init_gammas, "betas": init_betas},
#     shots=1024,
# )
# result = job.result()
# ```
#
# :::
# :::{tab-item} PennyLane
# :sync: sdk
#
# ```python
# from qamomile.pennylane import PennylaneTranspiler
#
# transpiler = PennylaneTranspiler()
# executable = transpiler.transpile(
#     qaoa_circuit,
#     bindings={"edges": edges, "weights": weights, "bias": bias, "p": 2},
#     parameters=["gammas", "betas"],
# )
#
# job = executable.sample(
#     transpiler.executor(),
#     bindings={"gammas": init_gammas, "betas": init_betas},
#     shots=1024,
# )
# result = job.result()
# ```
#
# :::
# :::{tab-item} CUDA-Q
# :sync: sdk
#
# ```{note}
# CUDA-Q is only available on Linux systems with NVIDIA GPUs.
# ```
#
# ```python
# from qamomile.cudaq import CudaqTranspiler
#
# transpiler = CudaqTranspiler()
# executable = transpiler.transpile(
#     qaoa_circuit,
#     bindings={"edges": edges, "weights": weights, "bias": bias, "p": 2},
#     parameters=["gammas", "betas"],
# )
#
# job = executable.sample(
#     transpiler.executor(),
#     bindings={"gammas": init_gammas, "betas": init_betas},
#     shots=1024,
# )
# result = job.result()
# ```
#
# :::
# ::::
#
# The following code executes the QAOA circuit using Qiskit (the primary example):

# %% [markdown]
# ## QAOA Optimization with Qiskit
# Now that the QAOA circuit is defined, let's optimize the parameters using Qiskit.

# %%
import random
from qamomile.qiskit import QiskitTranspiler

def random_ising(n: int, sparsity: float = 0.5):
    edges = []
    weights = []
    bias = []
    for i in range(n):
        bi = round(random.uniform(-1.0, 1.0), 2)
        bias.append(bi)
        for j in range(i + 1, n):
            if random.random() < sparsity:
                wij = round(random.uniform(-1.0, 1.0), 2)
                edges.append([i, j])
                weights.append(wij)

    return (
        edges,
        weights,
        bias,
    )

n = 5
edges, weights, bias = random_ising(n=n, sparsity=0.7)


# %%
transpiler = QiskitTranspiler()
executable = transpiler.transpile(
    qaoa_circuit,
    bindings={
        "edges": edges,
        "weights": weights,
        "bias": bias,
        "p": 2,
    },
    parameters=["gammas", "betas"],
)


init_gammas = np.random.uniform(0, np.pi, size=2)
init_betas = np.random.uniform(0, np.pi/2, size=2)

job = executable.sample(
    transpiler.executor(),
    bindings={
        "gammas": init_gammas,
        "betas": init_betas,
    },
    shots=1024,
)


# %%
result = job.result()
print(result)

# %% [markdown]
# Let's check what quantum circuit was generated.
qiskit_circuit = executable.get_first_circuit()
print(qiskit_circuit.draw(output="text"))

# %% [markdown]
# ## Energy Calculation and Classical Optimization
# In QAOA, we calculate the energy (expected value of the cost function) from measurement results
# and optimize parameters to minimize it.
# First, let's define a function to calculate the Ising model energy.

# %%
def calculate_ising_energy(bitstring: list[int], edges: list[list[int]], weights: list[float], bias: list[float]) -> float:
    """
    Calculate the Ising model energy.

    Convert bitstring z_i ∈ {0, 1} to spin s_i ∈ {-1, +1} for calculation.
    s_i = 1 - 2*z_i (z_i=0 → s_i=1, z_i=1 → s_i=-1)

    E = Σ_{(i,j)} w_ij * s_i * s_j + Σ_i b_i * s_i
    """
    spins = [1 - 2 * b for b in bitstring]

    energy = 0.0
    # Interaction terms
    for (i, j), wij in zip(edges, weights):
        energy += wij * spins[i] * spins[j]
    # Bias terms
    for i, bi in enumerate(bias):
        energy += bi * spins[i]

    return energy


def calculate_expectation_value(
    sample_result,
    edges: list[list[int]],
    weights: list[float],
    bias: list[float],
) -> float:
    """
    Calculate the expected energy value from measurement results.
    """
    total_energy = 0.0
    total_counts = 0

    for bitstring, count in sample_result.results:
        energy = calculate_ising_energy(bitstring, edges, weights, bias)
        total_energy += energy * count
        total_counts += count

    return total_energy / total_counts


# %% [markdown]
# Next, we use scipy.optimize to optimize the parameters.

# %%
from scipy.optimize import minimize

# List to save optimization history
energy_history = []

def objective_function(params, transpiler, executable, edges, weights, bias, shots=1024):
    """
    Objective function to optimize.
    Takes parameters, runs the QAOA circuit, and returns the energy expectation value.
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

    energy = calculate_expectation_value(result, edges, weights, bias)
    energy_history.append(energy)

    return energy


# %%
# Run optimization
p = 2  # Number of QAOA layers

# Initial parameters
np.random.seed(42)
init_params = np.concatenate([
    np.random.uniform(0, np.pi, size=p),      # gammas
    np.random.uniform(0, np.pi/2, size=p),    # betas
])

# Clear history
energy_history = []

# Optimize with COBYLA method
result_opt = minimize(
    objective_function,
    init_params,
    args=(transpiler, executable, edges, weights, bias),
    method="COBYLA",
    options={"maxiter": 100, "disp": True},
)

print(f"\nOptimized parameters:")
print(f"  gammas: {result_opt.x[:p]}")
print(f"  betas: {result_opt.x[p:]}")
print(f"Final energy: {result_opt.fun:.4f}")

# %% [markdown]
# ## Visualizing Optimization Results
# Let's visualize the convergence of the optimization.

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(energy_history, marker='o', markersize=3)
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("QAOA Optimization Convergence")
plt.grid(True)
plt.show()

# %% [markdown]
# ## Checking Solutions with Optimized Parameters
# Let's check the distribution of final solutions using the optimized parameters.

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
    energy = calculate_ising_energy(bitstring, edges, weights, bias)
    results_with_energy.append((bitstring, count, energy))

results_with_energy.sort(key=lambda x: x[2])

print("Measurement results (sorted by energy):")
print("-" * 50)
for bitstring, count, energy in results_with_energy[:10]:
    bitstring_str = "".join(map(str, bitstring))
    print(f"  {bitstring_str}: count={count:4d}, energy={energy:.4f}")

# %% [markdown]
# ## Interpreting Results
# From the QAOA final results, the bitstring with the lowest energy is a candidate for the optimal solution.
# Let's compare with the exact solution to see how good a solution QAOA found.

# %%
from itertools import product

def find_exact_ground_state(n: int, edges: list[list[int]], weights: list[float], bias: list[float]) -> tuple[tuple[int, ...], float]:
    """
    Find the exact ground state by exhaustive search (only for small-scale problems).
    """
    min_energy = float('inf')
    best_bitstring: tuple[int, ...] = tuple([0] * n)

    for bitstring in product([0, 1], repeat=n):
        energy = calculate_ising_energy(list(bitstring), edges, weights, bias)
        if energy < min_energy:
            min_energy = energy
            best_bitstring = bitstring

    return best_bitstring, min_energy


exact_solution, exact_energy = find_exact_ground_state(n, edges, weights, bias)
qaoa_best = results_with_energy[0]

print("Comparison with exact solution:")
print("-" * 50)
print(f"Exact solution:    {''.join(map(str, exact_solution))}, energy={exact_energy:.4f}")
print(f"QAOA best solution: {''.join(map(str, qaoa_best[0]))}, energy={qaoa_best[2]:.4f}")
print(f"Energy difference: {qaoa_best[2] - exact_energy:.4f}")

# Calculate approximation ratio (considering negative energy cases)
if exact_energy != 0:
    approx_ratio = qaoa_best[2] / exact_energy
    print(f"Approximation ratio: {approx_ratio:.4f}")

# %% [markdown]
# ## Summary
# In this tutorial, we learned how to implement QAOA using Qamomile.
#
# Key points:
# 1. **qkernel decorator**: Quantum circuits can be defined as functions, and quantum gates can be applied using Python-like syntax
# 2. **Parameterized circuits**: By using `qmc.Float` type, quantum circuits with optimizable parameters can be created
# 3. **Transpiler**: Using `QiskitTranspiler`, defined circuits can be converted to Qiskit quantum circuits and executed
# 4. **Classical optimization**: Variational quantum algorithms can be implemented by combining with classical optimization libraries like `scipy.optimize`
#
# QAOA is a promising algorithm that can run on NISQ (Noisy Intermediate-Scale Quantum) devices,
# and using Qamomile allows for concise and intuitive QAOA implementation.
# %%
