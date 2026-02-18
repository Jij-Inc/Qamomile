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
# # FQAOA for Constrained Optimization
#
# In this tutorial we solve a constrained binary optimization problem using
# **Fermionic QAOA (FQAOA)** with JijModeling and Qamomile.
#
# FQAOA encodes binary variables into fermionic occupation numbers.
# Equality constraints of the form $\sum_i x_i = M$ are then enforced
# *exactly* by conserving the number of fermions $M$, eliminating the
# need for penalty terms that standard QAOA requires.
#
# **Reference**: Yoshioka et al., *Fermionic Quantum Approximate Optimization Algorithm* (2023).

# %%
import jijmodeling as jm
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Defining the Optimization Problem
#
# We consider a quadratic binary optimization problem with an equality
# constraint:
#
# $$
#   \min \quad \sum_{i,j} J_{i,j}
#       \sum_{d, d'} x_{i,d}\, x_{j,d'}
#   \qquad \text{s.t.} \quad \sum_{i,d} x_{i,d} = M
# $$
#
# where $x_{i,d} \in \{0, 1\}$ and $M$ is the number of fermions.
#
# With standard QAOA the constraint must be added as a penalty term
# $\lambda \bigl(\sum_{i,d} x_{i,d} - M\bigr)^2$, and tuning $\lambda$ is
# non-trivial.  FQAOA avoids this entirely.


# %%
problem = jm.Problem("qubo")


@problem.update
def _(problem: jm.DecoratedProblem):
    J = problem.Float(ndim=2)
    n = J.len_at(0, latex="n")
    D = problem.Dim()
    x = problem.BinaryVar(shape=(n, D))

    # Quadratic objective
    problem += J.ndenumerate().map(
        lambda ij_v: ij_v[1] * x[ij_v[0][0]].sum() * x[ij_v[0][1]].sum()
    ).sum()

    # Equality constraint: total number of selected bits equals M
    problem += problem.Constraint("constraint", x.sum() == 4)


problem

# %% [markdown]
# ## Preparing Instance Data
#
# We prepare a small instance with a $4 \times 4$ coefficient matrix $J$
# and $D = 2$ bits per integer.  The equality constraint requires
# exactly $M = 4$ bits to be set to 1.

# %%
instance_data = {
    "J": [
        [0.0, 0.4, 0.0, 0.0],
        [0.0, 0.0, 0.8, 0.0],
        [0.0, 0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0, 0.0],
    ],
    "D": 2,
}

num_fermions = 4  # must match the constraint sum

# %% [markdown]
# ## Creating a Compiled Instance
#
# We compile the mathematical model together with the instance data using
# `problem.eval()`.

# %%
instance = problem.eval(instance_data)

# %% [markdown]
# ## Converting to an FQAOA Circuit and Hamiltonian
#
# `FQAOAConverter` takes the compiled instance **and** the number of
# fermions $M$. The equality constraint is enforced by the fermionic
# encoding itself, so no penalty terms are needed.
#
# We can then:
# - Use `transpile()` to generate the FQAOA quantum circuit
# - Use `get_cost_hamiltonian()` to inspect the cost Hamiltonian
#
# The number of FQAOA layers, $p$, is set to $2$ here.

# %%
from qamomile.optimization.fqaoa import FQAOAConverter
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

p = 2  # Number of FQAOA layers
converter = FQAOAConverter(instance, num_fermions=num_fermions)
executable = converter.transpile(transpiler, p=p)

# %% [markdown]
# Let's inspect the cost Hamiltonian. The Hamiltonian is constructed from the Ising representation of the QUBO objective.

# %%
cost_hamiltonian = converter.get_cost_hamiltonian()
cost_hamiltonian

# %% [markdown]
# Let's look at the generated quantum circuit. Unlike standard QAOA, the FQAOA circuit includes a Givens-rotation initial state preparation and a fermionic hopping mixer.

# %%
qiskit_circuit = executable.get_first_circuit()
qiskit_circuit.draw()

# %% [markdown]
# ## VQE Optimization
#
# We use scipy's COBYLA optimizer to minimize the energy expectation value
# over the variational parameters `gammas` and `betas`.

# %%
from scipy.optimize import minimize

energy_history = []


def objective_function(params, transpiler, executable, converter, shots=1024):
    p = len(params) // 2
    gammas = params[:p]
    betas = params[p:]

    job = executable.sample(
        transpiler.executor(),
        bindings={"gammas": gammas, "betas": betas},
        shots=shots,
    )
    result = job.result()

    sampleset = converter.decode(result)
    energy = sampleset.energy_mean()
    energy_history.append(energy)
    return energy


# %%
np.random.seed(42)

init_params = np.concatenate(
    [
        np.random.uniform(0, 2 * np.pi, size=p),  # gammas
        np.random.uniform(0, np.pi, size=p),  # betas
    ]
)

energy_history = []

print(f"Starting FQAOA optimization with p={p} layers...")

result_opt = minimize(
    objective_function,
    init_params,
    args=(transpiler, executable, converter),
    method="COBYLA",
    options={"maxiter": 100, "disp": True},
)

print("\nOptimized parameters:")
print(f"  gammas: {result_opt.x[:p]}")
print(f"  betas:  {result_opt.x[p:]}")
print(f"Final energy: {result_opt.fun:.4f}")

# %% [markdown]
# ## Visualizing Optimization Results
#
# Let's visualize the convergence of the optimization process.

# %%
plt.figure(figsize=(10, 5))
plt.plot(energy_history, marker="o", markersize=3)
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("FQAOA Optimization Convergence")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Final Solution Analysis
#
# Now let's sample from the optimized circuit and analyze the results.

# %%
optimal_gammas = result_opt.x[:p]
optimal_betas = result_opt.x[p:]

job_final = executable.sample(
    transpiler.executor(),
    bindings={"gammas": optimal_gammas, "betas": optimal_betas},
    shots=4096,
)
result_final = job_final.result()

# Decode results using the converter
sampleset = converter.decode(result_final)

num_vars = converter.num_qubits

# Build frequency distribution over all sampled bitstrings
bitstrings = []
counts = []
energies = []
for i in range(len(sampleset.samples)):
    sample = sampleset.samples[i]
    bitstring_str = "".join(str(sample[j]) for j in range(num_vars))
    bitstrings.append(bitstring_str)
    counts.append(sampleset.num_occurrences[i])
    energies.append(sampleset.energy[i])

# Sort by bitstring for consistent display
sorted_order = np.argsort(bitstrings)
bitstrings = [bitstrings[i] for i in sorted_order]
counts = [counts[i] for i in sorted_order]
energies = [energies[i] for i in sorted_order]

# Determine optimal energy
best_sample, best_energy, best_count = sampleset.lowest()

# Plot frequency distribution
fig, ax = plt.subplots(figsize=(12, 5))
x_pos = np.arange(len(bitstrings))
bars = ax.bar(x_pos, counts)

# Highlight optimal solutions with red bars
for i, e in enumerate(energies):
    if np.isclose(e, best_energy):
        bars[i].set_color("red")

ax.set_xticks(x_pos)
ax.set_xticklabels(bitstrings, rotation=90)
ax.set_xlabel("Bitstring")
ax.set_ylabel("Counts")
ax.set_title(f"FQAOA Measurement Frequency Distribution (red = optimal, energy = {best_energy:.2f})")
plt.tight_layout()
plt.show()

# %% [markdown]
# Notice that **every bitstring has exactly $M = 4$ bits set to 1**. This confirms that the fermion number conservation enforces the equality constraint without any penalty terms. The red bars indicate the optimal solutions, showing that FQAOA successfully concentrates measurement probability on them.

# %%
print("Best solution found:")
print(f"  Variable assignment: {best_sample}")
print(f"  Energy: {best_energy:.4f}")
print(f"  Occurrences: {best_count}")

# %% [markdown]
# ## Comparison with the Exact Solution
#
# Since Qamomile's converters accept `ommx.v1.Instance`, we can easily compare
# our quantum result with a classical solver. Let's solve the same instance
# exactly with SCIP and see how the FQAOA solution compares.

# %%
from ommx_pyscipopt_adapter import OMMXPySCIPOptAdapter

solution = OMMXPySCIPOptAdapter.solve(instance)

print(f"Exact optimal value: {solution.objective:.4f}")
print(f"FQAOA best energy:   {best_energy:.4f}")

# %% [markdown]
# ## Summary
#
# In this tutorial we demonstrated how to solve a constrained optimization
# problem using FQAOA with Qamomile:
#
# 1. **Problem Formulation**: We defined a QUBO with an equality constraint using JijModeling
# 2. **Hamiltonian & Circuit Generation**: `FQAOAConverter` automatically generated the cost Hamiltonian and FQAOA circuit with Givens-rotation initial state and fermionic mixer
# 3. **VQE Optimization**: We used scipy's COBYLA optimizer to find optimal FQAOA parameters
# 4. **Solution Analysis**: The frequency distribution confirmed that FQAOA concentrates measurement probability on the optimal solutions, with all bitstrings satisfying the constraint exactly
#
# Key advantages of FQAOA over standard QAOA:
# - Equality constraints are satisfied **exactly** by construction
# - No penalty weight $\lambda$ to tune
# - The search space is restricted to feasible solutions, improving convergence
