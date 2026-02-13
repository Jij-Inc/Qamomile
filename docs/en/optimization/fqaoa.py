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
def constrained_qubo_problem() -> jm.Problem:
    J = jm.Placeholder("J", ndim=2)
    n = J.len_at(0, latex="n")
    D = jm.Placeholder("D")
    x = jm.BinaryVar("x", shape=(n, D))

    problem = jm.Problem("qubo")
    i, j = jm.Element("i", n), jm.Element("j", n)
    d, d_dash = jm.Element("d", D), jm.Element("d'", D)

    # Quadratic objective
    problem += jm.sum([i, j], J[i, j] * jm.sum([d, d_dash], x[i, d] * x[j, d_dash]))

    # Equality constraint: total number of selected bits equals M
    problem += jm.Constraint("constraint", jm.sum([i, d], x[i, d]) == 4)

    return problem


problem = constrained_qubo_problem()
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
# `jm.Interpreter`.

# %%
interpreter = jm.Interpreter(instance_data)
instance = interpreter.eval_problem(problem)

# %% [markdown]
# ## Converting to an FQAOA Circuit
#
# `FQAOAConverter` takes the compiled instance **and** the number of
# fermions $M$.  Internally, the QUBO is generated with
# `uniform_penalty_weight=0.0` because the constraint is enforced by the
# fermionic encoding itself.
#
# The `transpile` method produces an executable program whose only free
# parameters are the variational angles `gammas` and `betas`.

# %%
from qamomile.optimization.fqaoa import FQAOAConverter
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

p = 2  # Number of FQAOA layers
converter = FQAOAConverter(instance, num_fermions=num_fermions)
executable = converter.transpile(transpiler, p=p)

# %%
qiskit_circuit = executable.get_first_circuit()
if qiskit_circuit is not None:
    print(f"Number of qubits: {qiskit_circuit.num_qubits}")
    print(f"Number of variational parameters: {len(qiskit_circuit.parameters)}")
    print(f"Circuit depth: {qiskit_circuit.depth()}")

# %% [markdown]
# ## Energy Calculation
#
# To run the VQE loop we need a function that maps measurement results to
# the Ising energy.  The converter stores the spin model in
# `converter.spin_model`.

# %%
def calculate_ising_energy(bitstring: list[int], spin_model) -> float:
    """Convert a measurement bitstring to an Ising energy.

    Convention: z_i in {0, 1} -> s_i = 1 - 2*z_i in {+1, -1}.
    """
    spins = [1 - 2 * b for b in bitstring]
    return spin_model.calc_energy(spins)


def calculate_expectation_value(sample_result, spin_model) -> float:
    """Weighted average energy over all measurement outcomes."""
    total_energy = 0.0
    total_counts = 0
    for bitstring, count in sample_result.results:
        total_energy += calculate_ising_energy(bitstring, spin_model) * count
        total_counts += count
    return total_energy / total_counts


# %% [markdown]
# ## VQE Optimization
#
# We use scipy's COBYLA optimizer to minimize the energy expectation value
# over the variational parameters `gammas` and `betas`.

# %%
from scipy.optimize import minimize

energy_history: list[float] = []


def objective_function(params, spin_model, shots=1024):
    """Objective function for the VQE loop."""
    p = len(params) // 2
    gammas = params[:p]
    betas = params[p:]

    job = executable.sample(
        transpiler.executor(),
        bindings={"gammas": gammas, "betas": betas},
        shots=shots,
    )
    result = job.result()

    energy = calculate_expectation_value(result, spin_model)
    energy_history.append(energy)
    return energy


# %%
np.random.seed(42)

init_params = np.concatenate([
    np.random.uniform(0, 2 * np.pi, size=p),  # gammas
    np.random.uniform(0, np.pi, size=p),       # betas
])

energy_history = []

print(f"Starting FQAOA optimization with p={p} layers...")
print(f"Number of qubits: {converter.num_qubits}")
print(f"Number of fermions: {converter.num_fermions}")

result_opt = minimize(
    objective_function,
    init_params,
    args=(converter.spin_model,),
    method="COBYLA",
    options={"maxiter": 100, "disp": True},
)

print("\nOptimized parameters:")
print(f"  gammas: {result_opt.x[:p]}")
print(f"  betas:  {result_opt.x[p:]}")
print(f"Final energy: {result_opt.fun:.4f}")

# %% [markdown]
# ## Visualizing Optimization Convergence

# %%
plt.figure(figsize=(10, 5))
plt.plot(energy_history, marker="o", markersize=3)
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("FQAOA Optimization Convergence")
plt.grid(True)
plt.tight_layout()
# plt.show()

# %% [markdown]
# ## Final Solution Analysis
#
# We sample from the optimized circuit with more shots and decode the
# measurement results back into the original binary variable domain.

# %%
optimal_gammas = result_opt.x[:p]
optimal_betas = result_opt.x[p:]

job_final = executable.sample(
    transpiler.executor(),
    bindings={"gammas": optimal_gammas, "betas": optimal_betas},
    shots=4096,
)
result_final = job_final.result()

# Decode measurement results into binary variable assignments
sampleset = converter.decode(result_final)

# Show the best solution
best_sample, best_energy, best_count = sampleset.lowest()
print("Best solution found:")
print(f"  Variable assignment: {best_sample}")
print(f"  Energy: {best_energy:.4f}")
print(f"  Occurrences: {best_count}")

# %% [markdown]
# We can also inspect the top measurement outcomes sorted by energy.

# %%
results_with_energy = []
for bitstring, count in result_final.results:
    energy = calculate_ising_energy(bitstring, converter.spin_model)
    results_with_energy.append((bitstring, count, energy))

results_with_energy.sort(key=lambda x: x[2])

print("Top measurement results (sorted by energy):")
print("-" * 60)
for bitstring, count, energy in results_with_energy[:10]:
    bitstring_str = "".join(map(str, bitstring))
    probability = count / 4096
    # Verify fermion number conservation: all bitstrings should have
    # exactly num_fermions bits set to 1.
    num_ones = sum(bitstring)
    print(
        f"  {bitstring_str}: count={count:4d}, "
        f"prob={probability:.3f}, energy={energy:.4f}, "
        f"ones={num_ones}"
    )

# %% [markdown]
# Notice that **every bitstring has exactly $M = 4$ bits set to 1**.
# This confirms that the fermion number conservation enforces the
# equality constraint without any penalty terms.

# %% [markdown]
# ## Summary
#
# In this tutorial we demonstrated how to solve a constrained optimization
# problem using FQAOA with Qamomile:
#
# 1. **Problem formulation**: We defined a QUBO with an equality constraint using JijModeling.
# 2. **Constraint handling**: `FQAOAConverter` encodes the constraint via fermion number conservation — no penalty weight tuning needed.
# 3. **Circuit generation**: `converter.transpile()` produces the full FQAOA ansatz with Givens-rotation initial state and fermionic mixer.
# 4. **VQE optimization**: We found optimal variational parameters using scipy's COBYLA optimizer.
# 5. **Solution decoding**: `converter.decode()` maps measurement results back to variable assignments.
#
# Key advantages of FQAOA over standard QAOA:
# - Equality constraints are satisfied **exactly** by construction
# - No penalty weight $\lambda$ to tune
# - The search space is restricted to feasible solutions, improving convergence
