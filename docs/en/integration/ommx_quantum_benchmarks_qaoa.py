# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
# tags: [integration, optimization, variational]
# ---
#
# # Using OMMX Quantum Benchmarks: Implementing and Benchmarking Quantum Algorithms with Qamomile
#
# This tutorial shows how to run a Qamomile quantum algorithm on a problem
# from a public benchmark dataset and compare its solution quality with a
# classical solver in the same workflow.
#
# **Goal.** Build a QAOA solver in Qamomile, run it on a
# **Low Autocorrelation Binary Sequences (LABS)** instance loaded from the
# [OMMX Quantum Benchmarks](https://jij-inc.github.io/OmmxQuantumBenchmarks/en/)
# dataset, and compare the result with the classical SCIP solver
# accessed through the [`ommx-pyscipopt-adapter`](https://ommx-en-book.readthedocs.io/en/latest/user_guide/supported_ommx_adapters.html).
# Because both the QAOA path and the SCIP path consume the *same*
# `ommx.v1.Instance`, the main difference is the algorithm itself. This
# makes the comparison direct.

# %%
# Install the additional packages used in this tutorial.
# # !pip install "qamomile[qiskit]" ommx-quantum-benchmarks ommx-pyscipopt-adapter matplotlib scipy

# %%
# ruff: noqa: F821 -- dimensions are Qamomile dependent shape binders.
from __future__ import annotations

import os
import time

# `ommx-quantum-benchmarks` uses `minto` internally and, by default,
# `minto` prints a host-environment dump (Python version, virtualenv
# path, etc.) at experiment start. Suppress it so the rendered
# notebook outputs do not leak local-machine details.
os.environ["MINTO_TESTING"] = "true"

import matplotlib.pyplot as plt
import numpy as np
import ommx.v1
import ommx_pyscipopt_adapter
from ommx_quantum_benchmarks.qoblib import Labs
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from scipy.optimize import minimize

import qamomile as qm
from qamomile.optimization import BinaryModel

# %% [markdown]
# ## What is OMMX Quantum Benchmarks?
#
# **OMMX** ([Open Mathematical prograMming eXchange](https://jij-inc.github.io/ommx/en/introduction.html))
# is a data format for exchanging mathematical optimization problems across
# tools. Its `ommx.v1.Instance` stores the objective, constraints, and
# decision-variable metadata.
#
# **OMMX Quantum Benchmarks** is a curated collection of optimization
# benchmark instances distributed in this `ommx.v1.Instance` format. The
# first dataset shipped is **QOBLIB** (Quantum Optimization Benchmarking
# Library) {cite:p}`10.48550/arXiv.2504.03832`, which
# defines ten problem families used in the recent quantum-optimization
# literature, including LABS, Market Split, Independent Set, and
# Steiner-tree packing.
#
# Each benchmark instance is represented as an `ommx.v1.Instance`.
# The LABS instance used here includes a reference solution in the
# `ommx.v1.Solution` format.
# The same `Instance` can also be passed to classical OMMX adapters such
# as `ommx-pyscipopt-adapter`, so one problem definition can support both
# quantum and classical workflows.


# %% [markdown]
# ## Problem: Low Autocorrelation Binary Sequences (LABS)
#
# **LABS** asks for a binary sequence
# $\boldsymbol{s} = (s_0, s_1, \dots, s_{n-1}) \in \{-1, +1\}^n$ whose
# off-diagonal autocorrelations
#
# $$
# c_k(\boldsymbol{s}) = \sum_{i=0}^{n-k-1} s_i \, s_{i+k},
# \qquad k = 1, 2, \dots, n-1
# $$
#
# are as close to zero as possible. The benchmark objective is the
# **sum of squared autocorrelations**
#
# $$
# E(\boldsymbol{s}) = \sum_{k=1}^{n-1} c_k(\boldsymbol{s})^2,
# $$
#
# which we want to *minimize*. LABS has long served as a
# stress test for both classical and quantum heuristics.
#
# ### Loading a LABS instance
#
# `Labs` exposes two models: `"integer"` (uses integer decision variables
# for $c_k$ plus the constraints that tie them to $\boldsymbol{s}$) and
# `"quadratic_unconstrained"` (a quadratic unconstrained binary optimization (QUBO) reformulation that introduces
# auxiliary binary variables $z_{i,k}$ encoding the products
# $x_i x_{i+k+1}$ via a quadratic penalty). The QUBO form is the natural
# target for QAOA, so we use it here.

# %%
dataset = Labs()
instances = dataset.available_instances["quadratic_unconstrained"]
print(f"Dataset:           {dataset.name}")
print(f"Available models:  {dataset.model_names}")
print(f"Instance count:    {len(instances)}")
print(f"First 5 instances: {instances[:5]}")

# %% [markdown]
# We pick `labs005`, the $n=5$ instance. The QUBO reformulation
# declares $n + n(n-1) = 25$ binary variables (5 sequence bits
# $x_i$ plus $n(n-1) = 20$ auxiliary $z_{i,k}$ bits). The objective
# already contains the penalties and uses just 10 of the auxiliaries,
# one for each pair of sequence positions. This gives a 15-qubit problem:
# small enough to simulate locally, but large enough that QAOA is
# non-trivial.

# %%
instance_name = "labs005"
instance, reference_solution = dataset("quadratic_unconstrained", instance_name)

print(f"OMMX variables:    {instance.num_variables}")
print(f"OMMX constraints:  {instance.num_constraints}")
print(f"Reference E(s):    {reference_solution.objective}")
print(f"Reference feasible: {reference_solution.feasible}")

# %% [markdown]
# The bundled reference solution gives the known optimum
# $E^\star = 2$ for $n=5$. We will compare both QAOA and SCIP
# against this value.

# %% [markdown]
# ## Algorithm: QAOA
#
# We build the QAOA circuit from scratch with `@qm.qkernel`, following the recipe in
# [QAOA for MaxCut: Building the Circuit from Scratch](../algorithm/qaoa_maxcut).
# Refer to that tutorial for the gate-by-gate derivation; here we
# focus on the implementation.


# %% [markdown]
# ### Spin model from the OMMX instance
#
# We extract the QUBO with `Instance.to_qubo()` and express each binary
# variable as $x_i=(1-Z_i)/2$. To normalize the energy scale, we divide
# the resulting spin coefficients and constant by the largest absolute
# nonconstant coefficient (10,028 for `labs005`).


# %%
# `Instance.to_qubo()` mutates the instance (it absorbs constraints into
# the objective via the penalty method). Round-trip through bytes to keep
# the caller's instance untouched.
instance_for_qubo = ommx.v1.Instance.from_bytes(instance.to_bytes())
qubo, qubo_constant = instance_for_qubo.to_qubo()
binary_model = BinaryModel.from_hubo(qubo, qubo_constant)
spin_model = binary_model.to_ising().normalize_by_abs_max()
num_qubits = spin_model.num_variables
print(f"QAOA qubits: {num_qubits}")

# %% [markdown]
# ### Cost Hamiltonian
#
# To use the exact expectation value rather than a shot estimate for the
# optimization, we build the Ising cost Hamiltonian directly from the
# spin-model coefficients: $Z_i$ terms for the linear part and
# $Z_i Z_j$ terms for the quadratic part.

# %%
# Map OMMX variable IDs to consecutive qubit indices.
qubit_index = {var: i for i, var in enumerate(spin_model.variables)}
coefficients = spin_model.coefficients
couplings = {
    (qubit_index[ids[0]], qubit_index[ids[1]]): value
    for ids, value in coefficients.items()
    if len(ids) == 2
}
fields = tuple(coefficients.get((var,), 0.0) for var in spin_model.variables)

cost_hamiltonian = SparsePauliOp.from_sparse_list(
    [("ZZ", [i, j], value) for (i, j), value in couplings.items()]
    + [("Z", [i], value) for i, value in enumerate(fields)]
    + [("I", [0], spin_model.constant)],
    num_qubits=num_qubits,
)

# %% [markdown]
# ### QAOA qkernels
#
# The circuit uses three small Qamomile kernels. Hadamard gates prepare a
# uniform superposition. The cost layer applies $R_{ZZ}(J_{ij}\gamma)$ and
# $R_Z(h_i\gamma)$ using the normalized couplings and fields. These gates
# use half their rotation angle in the exponent. The mixer applies
# $R_X(2\beta)$ to each qubit. The sampling kernel measures the resulting
# state in the computational basis.


# %%
@qm.qkernel
def superposition(n: qm.Const[qm.Int]) -> qm.Q[qm.Bits[n]]:
    q = qm.qubits(n)
    for i in range(n):
        q = qm.h(q, target=i)
    return q


@qm.qkernel
def cost_layer(
    q: qm.Q[qm.Bits[n]],
    n: qm.Const[qm.Int],
    edge_count: qm.Const[qm.Int],
    edges: qm.Const[qm.Matrix[qm.Int, edge_count, 2]],
    quad: qm.Const[qm.Vector[qm.Real, edge_count]],
    linear: qm.Const[qm.Vector[qm.Real, n]],
    gamma: qm.Real,
) -> qm.Q[qm.Bits[n]]:
    for k in range(edge_count):
        q = qm.rzz(q, quad[k] * gamma, left=edges[k][0], right=edges[k][1])
    for i in range(n):
        q = qm.rz(q, linear[i] * gamma, target=i)
    return q


@qm.qkernel
def mixer_layer(
    q: qm.Q[qm.Bits[n]], n: qm.Const[qm.Int], beta: qm.Real
) -> qm.Q[qm.Bits[n]]:
    for i in range(n):
        q = qm.rx(q, 2.0 * beta, target=i)
    return q


@qm.qkernel
def qaoa_state(
    n: qm.Const[qm.Int],
    edge_count: qm.Const[qm.Int],
    edges: qm.Const[qm.Matrix[qm.Int, edge_count, 2]],
    quad: qm.Const[qm.Vector[qm.Real, edge_count]],
    linear: qm.Const[qm.Vector[qm.Real, n]],
    layers: qm.Const[qm.Int],
    gammas: qm.Param[qm.Vector[qm.Real, layers]],
    betas: qm.Param[qm.Vector[qm.Real, layers]],
) -> qm.Q[qm.Bits[n]]:
    q = superposition(n)
    for layer in range(layers):
        q = cost_layer(q, n, edge_count, edges, quad, linear, gammas[layer])
        q = mixer_layer(q, n, betas[layer])
    return q


@qm.qkernel
def qaoa_sample(
    n: qm.Const[qm.Int],
    edge_count: qm.Const[qm.Int],
    edges: qm.Const[qm.Matrix[qm.Int, edge_count, 2]],
    quad: qm.Const[qm.Vector[qm.Real, edge_count]],
    linear: qm.Const[qm.Vector[qm.Real, n]],
    layers: qm.Const[qm.Int],
    gammas: qm.Param[qm.Vector[qm.Real, layers]],
    betas: qm.Param[qm.Vector[qm.Real, layers]],
) -> qm.Shot[qm.Bits[n]]:
    q = qaoa_state(n, edge_count, edges, quad, linear, layers, gammas, betas)
    return qm.measure(q)


# %% [markdown]
# ### Compile and optimize
#
# We fix the problem and the number of layers ($p=3$) with `qm.specialize`
# and compile the sampling circuit once. For the energy calculation we
# remove the final measurements from a copy of this circuit.
# At each set of angles, Qiskit Aer's local `EstimatorV2` computes the
# exact expectation of the normalized Hamiltonian from the simulated state.
# BFGS uses finite differences of these values to choose the angles.
# The plot records one point per objective evaluation. We seed NumPy
# to reproduce the initial angles.


# %%
p = 3
fixed = {
    "n": num_qubits,
    "edge_count": len(couplings),
    "edges": tuple(couplings),
    "quad": tuple(couplings.values()),
    "linear": fields,
    "layers": p,
}
endpoint = qm.qiskit.Aer()
sampling = qm.compile(qm.specialize(qaoa_sample, **fixed), to=endpoint.target)
# The energy calculation uses the same gates up to the final measurements.
state_circuit = qm.qiskit.load_circuit(sampling).remove_final_measurements(
    inplace=False
)
state_circuit = transpile(state_circuit, AerSimulator(), optimization_level=0)
estimator = EstimatorV2(
    options={"backend_options": {"method": "statevector", "max_parallel_threads": 1}}
)

SEED = 42
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
maxiter = 1 if docs_test_mode else 50

rng = np.random.default_rng(SEED)
initial_params = rng.uniform(0, np.pi, 2 * p)
cost_history: list[float] = []


def angles(params: np.ndarray) -> dict[str, tuple[float, ...]]:
    return {"gammas": tuple(params[:p]), "betas": tuple(params[p:])}


def cost_fn(params: np.ndarray) -> float:
    """Return the exact expectation of the normalized cost Hamiltonian."""
    invocation = qm.Invocation.create(parameters=angles(params))
    circuit = qm.qiskit.bind_circuit(state_circuit, sampling, invocation)
    result = estimator.run([(circuit, cost_hamiltonian)], precision=0.0).result()
    energy = float(result[0].data.evs)
    cost_history.append(energy)
    return energy


t0 = time.perf_counter()
res = minimize(
    cost_fn,
    initial_params,
    method="BFGS",
    options={"maxiter": maxiter},
)
qaoa_optimize_time = time.perf_counter() - t0

print(f"Optimized expectation value (normalized): {res.fun:.4f}")
print(f"Function evaluations:                     {res.nfev}")
print(f"Wall time:                                {qaoa_optimize_time:.2f} s")

# %%
plt.figure(figsize=(8, 4))
plt.plot(cost_history, color="#2696EB")
plt.xlabel("Objective evaluation")
plt.ylabel("Expectation value (normalized)")
plt.title(f"QAOA Optimization Progress ({instance_name})")
plt.show()

# %% [markdown]
# ### Final sampling
#
# We run the compiled sampling circuit on local `qm.qiskit.Aer()` with
# `qm.run`. We collect 4,096 shots at the optimized
# angles and decode them against the original `ommx.v1.Instance`, so the
# returned `ommx.v1.SampleSet` reports the original QUBO objective
# directly. In this QUBO formulation, samples whose auxiliary $z$
# variables correctly encode the products $x_i x_{i+k+1}$ incur no
# penalty, and the objective equals the LABS energy
# $E(\boldsymbol{s}) = \sum_k c_k^2$. Samples that violate the
# implicit $z = x_i x_{i+k+1}$ relation incur an additive penalty.


# %%
def evaluate_with_ommx(
    result: qm.SampleResult, model: BinaryModel, ommx_instance: ommx.v1.Instance
) -> ommx.v1.SampleSet:
    """Score each measured bitstring in the original OMMX variable order."""
    samples = ommx.v1.Samples({})
    next_id = 0
    for row in model.decode(result).samples:
        ids = list(range(next_id, next_id + row.occurrences))
        samples.append(ids, ommx.v1.State(dict(row.assignment)))
        next_id += row.occurrences
    return ommx_instance.evaluate_samples(samples)


final_shots = 1 if docs_test_mode else 4096
final_result = qm.run(
    sampling, on=endpoint, parameters=angles(res.x), shots=final_shots, seed=SEED
).result
qaoa_sample_set = evaluate_with_ommx(final_result, binary_model, instance)

qaoa_summary = qaoa_sample_set.summary
qaoa_best = qaoa_sample_set.best_feasible
qaoa_best_E = int(round(qaoa_best.objective))
ref_E = int(reference_solution.objective)

assert len(qaoa_summary) == final_shots

print(f"Shots:                {len(qaoa_summary)}")
print(f"QAOA best objective:  {qaoa_best_E}")
print(f"Reference E*:         {ref_E}")

# %% [markdown]
# ### Distribution of objective values
#
# The histogram shows the original QUBO objective of each shot at the
# optimized angles. The red dashed line marks the optimum $E^\star=2$.
# The first bin also covers 6, 10 and 30, the other energies of sequences
# with consistent auxiliaries. Count only shots with objective 2 as hits.
# For `labs005`, any inconsistent auxiliary raises the objective to at
# least 10,002.


# %%
objectives = qaoa_summary["objective"].to_numpy()

plt.figure(figsize=(8, 4))
plt.hist(objectives, bins=40, color="#2696EB", edgecolor="white")
plt.axvline(
    ref_E,
    color="red",
    linestyle="--",
    label=f"Reference $E^\\star$ = {ref_E}",
)
plt.xlabel("Objective value (QUBO energy)")
plt.ylabel("Frequency")
plt.title(f"QAOA Output Distribution (p={p}, shots={final_shots})")
plt.legend()
plt.show()

# %% [markdown]
# ## Classical baseline: SCIP via the OMMX adapter
#
# **SCIP** is a branch-and-bound-based MILP/QUBO solver that can find the
# exact optimal solution. The same `ommx.v1.Instance`
# is consumed by
# `ommx_pyscipopt_adapter.OMMXPySCIPOptAdapter.solve`, which hands the
# problem to SCIP via PySCIPOpt and returns an `ommx.v1.Solution`
# evaluated against the original instance. Its `.objective` is
# directly comparable to the original QUBO objective of each QAOA shot.

# %%
t0 = time.perf_counter()
scip_solution = ommx_pyscipopt_adapter.OMMXPySCIPOptAdapter.solve(instance)
scip_solve_time = time.perf_counter() - t0

scip_E = int(round(scip_solution.objective))
assert scip_solution.feasible
assert scip_solution.objective == reference_solution.objective
print(f"SCIP objective: {scip_E}")
print(f"SCIP feasible: {scip_solution.feasible}")
print(f"Wall time:    {scip_solve_time:.3f} s")

# %% [markdown]
# ## Results comparison
#
# SCIP returns an optimal assignment for this instance. QAOA samples a
# distribution over bitstrings. We report QAOA's **best shot**
# (the lowest-objective bitstring seen across all samples) and its
# **hit rate** on the reference optimum (the fraction of shots that
# achieved $E^\star$).


# %%
# Hit rate: fraction of shots that achieved the reference optimum.
hits = int((qaoa_summary["objective"] == reference_solution.objective).sum())
hit_rate = hits / final_shots

print(f"{'Solver':<22} {'QUBO':>8} {'Time (s)':>12}")
print("-" * 46)
print(f"{'Reference (bundled)':<22} {ref_E:>8} {'-':>12}")
print(f"{'SCIP (classical)':<22} {scip_E:>8} {scip_solve_time:>12.3f}")
print(f"{'QAOA (best shot)':<22} {qaoa_best_E:>8} {qaoa_optimize_time:>12.2f}")
print()
print(
    f"QAOA hit rate on QUBO optimum {ref_E}: {hit_rate:.1%}  ({hits}/{final_shots} shots)"
)

# %% [markdown]
# Two observations make this benchmark meaningful:
#
# 1. **Optimum.** Both SCIP and the best QAOA shot reach the reference
#    optimum $E^\star = 2$, so QAOA is capable of finding the optimal
#    sequence at $n = 5$ with only $p = 3$.
# 2. **Hit rate.** About 4.6% of the 4,096 final measurements reached
#    the QUBO optimum, including the auxiliary bits.
#
# Note that the wall-time figures are only indicative. SCIP's timing is
# just the solver's runtime on the CPU, while the QAOA timing covers the
# full classical-quantum optimization loop on a state-vector simulator;
# both depend on the execution environment. A genuinely fair comparison
# would therefore require a more careful methodology. Either way, combining
# Qamomile with datasets like OMMX Quantum Benchmarks makes it easy to
# compare quantum algorithms and classical solvers across a variety of
# metrics.

# %% [markdown]
# ## Summary
#
# In this tutorial we:
#
# 1. Loaded a LABS instance straight from the OMMX Quantum Benchmarks
#    dataset as an `ommx.v1.Instance`.
# 2. Extracted the QUBO with `Instance.to_qubo()`, normalized its spin
#    coefficients, and ran a hand-written Qamomile QAOA circuit on local
#    Qiskit Aer for both exact energy evaluation and final sampling.
# 3. Compared the QAOA output (best shot, hit rate, sampling
#    distribution) against SCIP via
#    `ommx_pyscipopt_adapter.OMMXPySCIPOptAdapter.solve` on the same
#    instance, plus the reference optimum bundled with the benchmark.
#
# The same circuit can be reused for other QOBLIB models available as
# QUBOs. Integer or constrained formulations may need an encoding first.
# The number of active binary variables determines the circuit width.
