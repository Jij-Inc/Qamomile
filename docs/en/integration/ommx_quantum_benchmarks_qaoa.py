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
# OMMX Quantum Benchmarks
# dataset, and compare the result with the classical SCIP solver
# accessed through the `ommx-pyscipopt-adapter`.
# Because both the QAOA path and the SCIP path consume the *same*
# `ommx.v1.Instance`, the main difference is the algorithm itself. This
# makes the comparison direct.

# %% [markdown]
# ## Installation
#
# ```bash
# pip install "qamomile[qiskit]" ommx-quantum-benchmarks ommx-pyscipopt-adapter
# ```

# %%
# Install the additional packages used in this tutorial.
# # !pip install "qamomile[qiskit]" ommx-quantum-benchmarks ommx-pyscipopt-adapter

# %%
import os
import time

# Suppress the benchmark runner's host-environment report in rendered output.
os.environ["MINTO_TESTING"] = "true"

import matplotlib.pyplot as plt
import numpy as np
import ommx.v1
import ommx_pyscipopt_adapter
from ommx_quantum_benchmarks.qoblib import Labs
from scipy.optimize import minimize

import qamomile as qm
from qamomile.measurement import Estimator
from qamomile.optimization import BinaryModel, QAOAExperiment
from qamomile.qiskit import Aer, load_circuit
from qamomile.stdlib.variational import qaoa_state

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# ## What is OMMX Quantum Benchmarks?
#
# **OMMX** (Open Mathematical prograMming eXchange)
# is a data format for exchanging mathematical optimization problems across
# tools. Its `ommx.v1.Instance` stores the objective, constraints, and
# decision-variable metadata.
#
# **OMMX Quantum Benchmarks** is a curated collection of optimization
# benchmark instances distributed in this `ommx.v1.Instance` format. The
# first dataset shipped is **QOBLIB** (Quantum Optimization Benchmarking
# Library) arXiv:2504.03832, which
# contains nine problem families used in the recent quantum-optimization
# literature, including LABS, Market Split, Independent Set, and
# Steiner-tree packing.
#
# Because each benchmark instance is represented as an `ommx.v1.Instance`,
# any Qamomile workflow that accepts `ommx.v1.Instance`, including
# `QAOAConverter`, can consume these problems without writing any extra code.
# In addition, a reference solution is provided in the `ommx.v1.Solution`
# format and can be used to evaluate benchmark results.
# The same `Instance` can also be passed to classical OMMX adapters such
# as `ommx-pyscipopt-adapter`, so one problem definition can support both
# quantum and classical workflows.
#
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
# which we want to *minimize*. LABS is NP-hard and has long served as a
# stress test for both classical and quantum heuristics.
#
# ### Loading a LABS instance
#
# `Labs` exposes two models: `"integer"` (uses integer decision variables
# for $c_k$ plus the constraints that tie them to $\boldsymbol{s}$) and
# `"quadratic_unconstrained"` (a QUBO reformulation that introduces
# auxiliary binary variables $z_{i,k}$ encoding the products
# $x_i x_{i+k+1}$ via a quadratic penalty). The QUBO form is the natural
# target for QAOA, so we use it here.
#
# We pick `labs005`, the $n=5$ instance. The QUBO reformulation
# uses $n + n(n-1) = 25$ binary variables (5 sequence bits
# $x_i$ plus $n(n-1) = 20$ auxiliary $z_{i,k}$ bits). After
# `Instance.to_qubo()` folds the penalty terms into the objective
# and inactive variables are pruned, this becomes a 15-qubit problem:
# small enough to simulate locally, but large enough that QAOA is
# non-trivial.
#
# The bundled reference solution gives the known optimum
# $E^\star = 2$ for $n=5$. We will compare both QAOA and SCIP
# against this value.
#

# %%
dataset = Labs()
instance, reference_solution = dataset("quadratic_unconstrained", "labs005")
original_payload = instance.to_bytes()

print(f"Dataset: {dataset.name}")
print(f"Available models: {dataset.model_names}")
print(f"OMMX variables: {instance.num_variables}")
print(f"OMMX constraints: {instance.num_constraints}")
print(f"Reference objective: {reference_solution.objective}")

assert instance.num_variables == 25
assert instance.num_constraints == 0
assert reference_solution.feasible
assert reference_solution.objective == 2.0

# %% [markdown]
# ## Extract an Immutable Active QUBO
#
# `to_qubo` is allowed to transform its receiver, so we first round-trip the
# benchmark bytes. `BinaryModel.from_hubo` canonicalizes the resulting
# coefficients in Rust and infers only the 15 variable IDs that occur in
# nonzero terms. The caller-owned benchmark remains unchanged.

# %%
instance_for_qubo = ommx.v1.Instance.from_bytes(original_payload)
qubo, qubo_constant = instance_for_qubo.to_qubo()
model = BinaryModel.from_hubo(qubo, qubo_constant)

assert instance.to_bytes() == original_payload
assert model.num_variables == 15
assert model.order == 2
assert BinaryModel.from_bytes(model.to_bytes()) == model

print(f"Active QAOA variables: {model.variables}")
print(f"QAOA qubits: {model.num_variables}")

# %% [markdown]
# ## Build and Compile Three-Layer QAOA
#
# `QAOAExperiment` fixes the active binary Hamiltonian and three-layer topology in a measured classical-output program for final sampling. The public `qaoa_state` building block defines the same state preparation for the energy estimator. Both use the declared `gammas` and `betas` vectors; no SDK parameter-name mapping is needed.

# %%
LAYERS = 3
experiment = QAOAExperiment(model, layers=LAYERS)
endpoint = Aer()
compiled = experiment.compile(to=endpoint.target)
study = qm.resources.logical(experiment.program)

assert compiled.identity.problem_digest == model.semantic_digest
assert study.metric(
    qm.resources.logical_operation_count("quantum.h")
).numeric_enclosure() == (15, 15)
assert study.metric(
    qm.resources.logical_operation_count("quantum.rx")
).numeric_enclosure() == (45, 45)
assert study.metric(
    qm.resources.logical_operation_count("quantum.pauli_evolution")
).numeric_enclosure() == (3, 3)

measured_circuit = load_circuit(compiled.artifact)
assert measured_circuit.num_qubits == model.num_variables
assert measured_circuit.num_parameters == 2 * LAYERS
print(measured_circuit.draw(output="text", fold=120))

# %% [markdown]
# ## Optimize a Finite-Shot Energy Estimate
#
# `Estimator` prepares measurements of the active QUBO Hamiltonian. All terms are Z products and can share one QWC measurement group, so covariance is retained when scoring each shot. `ENERGY_SHOTS` is the total budget per objective evaluation.
#
# COBYLA works without a finite-difference gradient. Its objective is noisy, and its last value is not an independent estimate of solution quality. We use fresh samples for the selected parameters afterward. Sampling uncertainty does not bound optimization error or guarantee convergence to the benchmark optimum.

# %%
preparation = qaoa_state.specialize(
    width=model.num_variables, layers=LAYERS, hamiltonian=model.hamiltonian
)
estimator = Estimator(preparation, model.hamiltonian, grouping="qwc")
ENERGY_SHOTS = 32 if docs_test_mode else 2_048
cost_history = []


def qaoa_parameters(values):
    """Keep aggregate parameter names from the Qamomile signature."""
    return {
        "gammas": tuple(float(value) for value in values[:LAYERS]),
        "betas": tuple(float(value) for value in values[LAYERS:]),
    }


def cost_function(parameters) -> float:
    """Estimate and record one finite-shot QAOA energy."""
    estimate = estimator.estimate(
        on=endpoint,
        parameters=qaoa_parameters(parameters),
        shots=ENERGY_SHOTS,
        seed=100 + len(cost_history),
    )
    assert estimate.shots == ENERGY_SHOTS
    cost_history.append(estimate.value)
    return estimate.value


SEED = 42
rng = np.random.default_rng(SEED)
initial_parameters = rng.uniform(0.0, np.pi, 2 * LAYERS)
max_iterations = 10 if docs_test_mode else 80

start = time.perf_counter()
optimization = minimize(
    cost_function,
    initial_parameters,
    method="COBYLA",
    options={"maxiter": max_iterations, "rhobeg": 0.5},
)
qaoa_optimize_time = time.perf_counter() - start

print(f"Optimized expectation: {optimization.fun:+.6f}")
print(f"Function evaluations: {optimization.nfev}")
print(f"Wall time: {qaoa_optimize_time:.2f} s")

assert len(optimization.x) == 2 * LAYERS
assert len(cost_history) == optimization.nfev

# %%
plt.figure(figsize=(8, 4))
plt.plot(cost_history, color="#2696EB")
plt.xlabel("Energy evaluation")
plt.ylabel("Sampled expectation value")
plt.title("QAOA optimization progress (LABS, n=5)")
plt.show()

# %% [markdown]
# ## Final Sampling and OMMX Evaluation
#
# Reusing the compiled experiment, `compiled.run` samples through Qamomile
# and its selected Aer endpoint. A separate fresh estimate reports the energy.
# Qamomile decodes the 15 active variables in their original IDs. We then expand their
# multiplicities into OMMX samples and evaluate them against the untouched
# benchmark instance, so reported objectives use the published LABS meaning.

# %%
FINAL_SHOTS = 1 if docs_test_mode else 4_096
final_run = compiled.run(
    on=endpoint,
    **qaoa_parameters(optimization.x),
    shots=FINAL_SHOTS,
    seed=SEED,
)
decoded_samples = final_run.samples
final_energy = estimator.estimate(
    on=endpoint,
    parameters=qaoa_parameters(optimization.x),
    shots=ENERGY_SHOTS,
    seed=1_000_000,
)
print(
    f"Fresh energy estimate: {final_energy.value:+.6f} ± {final_energy.standard_error:.6f}"
)

ommx_samples = ommx.v1.Samples({})
next_sample_id = 0
for decoded_sample in decoded_samples.samples:
    sample_ids = list(
        range(
            next_sample_id,
            next_sample_id + decoded_sample.occurrences,
        )
    )
    next_sample_id += decoded_sample.occurrences
    ommx_samples.append(
        sample_ids,
        ommx.v1.State(
            {variable: float(value) for variable, value in decoded_sample.assignment}
        ),
    )

qaoa_sample_set = instance.evaluate_samples(ommx_samples)
qaoa_summary = qaoa_sample_set.summary
qaoa_best = qaoa_sample_set.best_feasible
qaoa_best_objective = int(round(qaoa_best.objective))
reference_objective = int(round(reference_solution.objective))

print(f"Shots: {len(qaoa_summary)}")
print(f"QAOA best objective: {qaoa_best_objective}")
print(f"Reference objective: {reference_objective}")

assert decoded_samples.total_occurrences == FINAL_SHOTS
assert len(qaoa_summary) == FINAL_SHOTS
assert qaoa_best_objective >= reference_objective

# %% [markdown]
# QAOA returns a distribution, not one answer. The histogram shows every
# original QUBO objective, with the benchmark optimum marked in red.

# %%
objectives = qaoa_summary["objective"].to_numpy()

plt.figure(figsize=(8, 4))
plt.hist(objectives, bins=40, color="#2696EB", edgecolor="white")
plt.axvline(
    reference_objective,
    color="red",
    linestyle="--",
    label=rf"Reference $E^\star$ = {reference_objective}",
)
plt.xlabel("Objective value (QUBO energy)")
plt.ylabel("Frequency")
plt.title(f"QAOA output distribution (p={LAYERS}, shots={FINAL_SHOTS})")
plt.legend()
plt.show()

# %% [markdown]
# ## Classical Baseline: SCIP
#
# The OMMX PySCIPOpt adapter solves the same immutable instance with SCIP. Its
# objective is directly comparable with the bundled reference and the best
# QAOA shot.

# %%
start = time.perf_counter()
scip_solution = ommx_pyscipopt_adapter.OMMXPySCIPOptAdapter.solve(instance)
scip_solve_time = time.perf_counter() - start
scip_objective = int(round(scip_solution.objective))

print(f"SCIP objective: {scip_objective}")
print(f"SCIP feasible: {scip_solution.feasible}")
print(f"Wall time: {scip_solve_time:.3f} s")

assert scip_solution.feasible
assert scip_objective == reference_objective

# %% [markdown]
# ## Results Comparison
#
# SCIP returns one optimum; QAOA returns a distribution. We therefore report
# QAOA's best shot and the fraction of shots that reached the published
# optimum. Wall times are illustrative only because they cover different
# algorithmic work.

# %%
hit_rate = float(
    (qaoa_summary["objective"].round().astype(int) == reference_objective).mean()
)

print(f"{'Solver':<22} {'E(s)':>8} {'Time (s)':>12}")
print("-" * 46)
print(f"{'Reference (bundled)':<22} {reference_objective:>8} {'-':>12}")
print(f"{'SCIP (classical)':<22} {scip_objective:>8} {scip_solve_time:>12.3f}")
print(f"{'QAOA (best shot)':<22} {qaoa_best_objective:>8} {qaoa_optimize_time:>12.2f}")
print(f"QAOA hit rate on E*={reference_objective}: {hit_rate:.1%}")

assert 0.0 <= hit_rate <= 1.0

# %% [markdown]
# ## Summary
#
# - `labs005` supplies one immutable benchmark and known optimum.
# - Its 15 active QUBO variables feed a three-layer typed QAOA experiment.
# - COBYLA optimizes finite-shot Qamomile energy estimates; the measured program supplies a
#   4,096-shot output distribution.
# - OMMX restores published objective meaning, while SCIP independently solves
#   the same instance.
# - Best shot, optimum hit rate, distribution, and wall times reproduce the
#   original quantum-versus-classical benchmark UX.
