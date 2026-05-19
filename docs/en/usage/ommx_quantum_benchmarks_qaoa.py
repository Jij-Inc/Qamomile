# ---
# jupyter:
#   jupytext:
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
# tags: [usage, optimization, variational]
# ---
#
# # Using OMMX Benchmarks (1): Implementing and Benchmarking Quantum Algorithms with Qamomile
#
# This tutorial shows how to drive a Qamomile-built quantum algorithm with
# a problem taken straight from a public benchmark dataset, and how to
# compare its solution quality against a classical solver in the same
# end-to-end pipeline.
#
# **Goal.** Build a QAOA solver in Qamomile, run it on a
# **Low Autocorrelation Binary Sequences (LABS)** instance loaded from the
# [OMMX Quantum Benchmarks](https://github.com/Jij-Inc/OmmxQuantumBenchmarks)
# dataset, and benchmark the result against the classical SCIP solver
# accessed through the [`ommx-pyscipopt-adapter`](https://github.com/Jij-Inc/ommx-pyscipopt-adapter).
# Because both the QAOA path and the SCIP path consume the *same*
# `ommx.v1.Instance`, the only thing that differs between them is the
# algorithm itself — making this a clean apples-to-apples comparison.

# %%
# Install the additional packages used in this tutorial.
# # !pip install qamomile ommx-quantum-benchmarks ommx-pyscipopt-adapter

# %% [markdown]
# ## What is OMMX Quantum Benchmarks?
#
# **OMMX** ([Open Mathematical prograMming eXchange](https://jij-inc.github.io/ommx/en/introduction.html))
# is a data format for exchanging mathematical optimization problems across
# tools — its `ommx.v1.Instance` carries the objective, constraints,
# decision-variable metadata, and (optionally) a reference solution.
#
# **OMMX Quantum Benchmarks** is a curated collection of optimization
# benchmark instances distributed in this `ommx.v1.Instance` format. The
# first dataset shipped is **QOBLIB** (Quantum Optimization Benchmarking
# Library) [](https://arxiv.org/abs/2504.03832), which contains nine
# problem families used in the recent quantum-optimization literature,
# including LABS, Market Split, Independent Set, and Steiner-tree
# packing.
#
# Because each instance is just an `ommx.v1.Instance`, every Qamomile
# entry point that already accepts `ommx.v1.Instance` — most importantly
# `QAOAConverter` — can consume these benchmark problems with no extra
# adapter code. The same `Instance` can also be fed to classical OMMX
# adapters such as `ommx-pyscipopt-adapter`, which lets us reuse one
# problem definition across quantum and classical workflows.

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
# which we want to *minimize* (equivalently: maximize the **merit factor**
# $F = n^2 / (2 E)$). LABS is NP-hard and has long served as a stress test
# for both classical and quantum heuristics.
#
# ### Loading a LABS instance
#
# `Labs` exposes two models — `"integer"` (uses integer decision variables
# for $c_k$ plus the constraints that tie them to $\boldsymbol{s}$) and
# `"quadratic_unconstrained"` (a QUBO reformulation that introduces
# auxiliary binary variables $z_{i,k}$ encoding the products
# $x_i x_{i+k+1}$ via a quadratic penalty). The QUBO form is the natural
# target for QAOA, so we use it here.

# %%
from ommx_quantum_benchmarks.qoblib import Labs

dataset = Labs()
print(f"Dataset:           {dataset.name}")
print(f"Available models:  {dataset.model_names}")
print(f"Instance count:    {len(dataset.available_instances['quadratic_unconstrained'])}")
print(f"First 5 instances: {dataset.available_instances['quadratic_unconstrained'][:5]}")

# %% [markdown]
# We pick `labs005`, the $n=5$ instance. The QUBO reformulation
# uses $n + n(n-1) = 25$ binary variables (5 sequence bits
# $x_i$ plus $n(n-1) = 20$ auxiliary $z_{i,k}$ bits). After
# `Instance.to_qubo()` folds the penalty terms into the objective
# and inactive variables are pruned, this lands at 15 qubits —
# small enough to simulate locally, large enough that QAOA is
# non-trivial.

# %%
instance, reference_solution = dataset("quadratic_unconstrained", "labs005")
n = 5

print(f"OMMX variables:    {instance.num_variables}")
print(f"OMMX constraints:  {instance.num_constraints}")
print(f"Reference E(s):    {reference_solution.objective}")
print(f"Reference feasible: {reference_solution.feasible}")

# %% [markdown]
# The bundled reference solution gives the known optimum
# $E^\star = 2$ for $n=5$ — equivalently, merit factor
# $F^\star = 25 / (2 \cdot 2) = 6.25$. We will compare both
# QAOA and SCIP against this value.

# %% [markdown]
# ## Algorithm: QAOA
#
# Rather than reach for the high-level `QAOAConverter`, we build the
# QAOA pipeline from scratch with `@qkernel`, following the recipe in
# [QAOA for MaxCut: Building the Circuit from Scratch](../algorithm/qaoa_maxcut).
# Refer to that tutorial for the gate-by-gate derivation; here we
# focus on the implementation.

# %% [markdown]
# ### Spin model from the OMMX instance
#
# `Instance.to_qubo()` folds the (penalty-form) QUBO out of the
# `ommx.v1.Instance`. We then wrap it in a `BinaryModel` and switch to
# the spin (-1/+1) domain, which is what the QAOA cost layer expects.
# We also normalize the coefficients so the cost-landscape scale stays
# comparable across runs.

# %%
import ommx.v1

from qamomile.optimization.binary_model import BinaryModel, VarType

# `Instance.to_qubo()` mutates the instance (it absorbs constraints into
# the objective via the penalty method). Round-trip through bytes to keep
# the caller's instance untouched.
instance_for_qubo = ommx.v1.Instance.from_bytes(instance.to_bytes())
qubo, qubo_constant = instance_for_qubo.to_qubo()
spin_model = (
    BinaryModel.from_qubo(qubo, qubo_constant)
    .change_vartype(VarType.SPIN)
    .normalize_by_abs_max()
)
print(f"QAOA qubits: {spin_model.num_bits}")

# %% [markdown]
# ### QAOA qkernels
#
# Three small qkernels — uniform superposition, cost layer, mixer layer —
# composed into the full ansatz.

# %%
import qamomile.circuit as qmc


@qmc.qkernel
def superposition(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(n, name="q")
    for i in qmc.range(n):
        q[i] = qmc.h(q[i])
    return q


@qmc.qkernel
def cost_layer(
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    q: qmc.Vector[qmc.Qubit],
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    for (i, j), Jij in quad.items():
        q[i], q[j] = qmc.rzz(q[i], q[j], angle=Jij * gamma)
    for i, hi in linear.items():
        q[i] = qmc.rz(q[i], angle=hi * gamma)
    return q


@qmc.qkernel
def mixer_layer(
    q: qmc.Vector[qmc.Qubit],
    beta: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    n = q.shape[0]
    for i in qmc.range(n):
        q[i] = qmc.rx(q[i], angle=2.0 * beta)
    return q


@qmc.qkernel
def qaoa_ansatz(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    n: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    q = superposition(n)
    for layer in qmc.range(p):
        q = cost_layer(quad, linear, q, gammas[layer])
        q = mixer_layer(q, betas[layer])
    return qmc.measure(q)


# %% [markdown]
# ### Transpile and optimize
#
# Transpile with $p = 3$ layers, then run COBYLA on the mean spin-model
# energy of the sampled bitstrings. We seed both the AerSimulator and
# NumPy so the trajectory is reproducible.

# %%
import os
import time

import numpy as np
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from qamomile.qiskit import QiskitTranspiler

p = 3
transpiler = QiskitTranspiler()
executable = transpiler.transpile(
    qaoa_ansatz,
    bindings={
        "p": p,
        "quad": spin_model.quad,
        "linear": spin_model.linear,
        "n": spin_model.num_bits,
    },
    parameters=["gammas", "betas"],
)

SEED = 42
executor = transpiler.executor(
    backend=AerSimulator(seed_simulator=SEED, max_parallel_threads=1)
)

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
sample_shots = 256 if docs_test_mode else 1024
maxiter = 20 if docs_test_mode else 300

rng = np.random.default_rng(SEED)
initial_params = rng.uniform(0, np.pi, 2 * p)

cost_history: list[float] = []


def cost_fn(params: np.ndarray) -> float:
    """Sample the QAOA circuit at `params` and return the mean spin-model energy."""
    gammas = list(params[:p])
    betas = list(params[p:])
    job = executable.sample(
        executor,
        shots=sample_shots,
        bindings={"gammas": gammas, "betas": betas},
    )
    result = job.result()
    decoded = spin_model.decode_from_sampleresult(result)
    energy = decoded.energy_mean()
    cost_history.append(energy)
    return energy


t0 = time.perf_counter()
res = minimize(
    cost_fn,
    initial_params,
    method="COBYLA",
    options={"maxiter": maxiter},
)
qaoa_optimize_time = time.perf_counter() - t0

print(f"Optimized mean energy (normalized): {res.fun:.4f}")
print(f"Function evaluations:               {res.nfev}")
print(f"Wall time:                          {qaoa_optimize_time:.2f} s")

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(cost_history, color="#2696EB")
plt.xlabel("Iteration")
plt.ylabel("Mean energy (normalized)")
plt.title("QAOA Optimization Progress (LABS, n=5)")
plt.show()

# %% [markdown]
# ### Final sampling
#
# We sample once more with the optimized parameters and a larger shot
# count, then decode against the original `ommx.v1.Instance` so the
# returned `ommx.v1.SampleSet` reports the original QUBO objective
# directly. For this QUBO formulation, samples whose auxiliary $z$
# variables correctly encode the products $x_i x_{i+k+1}$ incur zero
# penalty and the objective equals the true LABS energy
# $E(\boldsymbol{s}) = \sum_k c_k^2$; samples that violate the
# implicit $z = x_i x_{i+k+1}$ relation pay an additive penalty
# proportional to the placeholder $P$ baked into the instance.

# %%
def evaluate_with_ommx(
    sample_result, spin_model: BinaryModel, ommx_instance: ommx.v1.Instance
) -> ommx.v1.SampleSet:
    """Decode SPIN samples, flip to BINARY, and evaluate against the OMMX instance."""
    binary_ss = spin_model.decode_from_sampleresult(sample_result)
    ommx_samples = ommx.v1.Samples({})
    next_id = 0
    for sample, occ in zip(binary_ss.samples, binary_ss.num_occurrences):
        if occ <= 0:
            continue
        # SPIN (+/-1) -> BINARY (0/1): x = (1 - s) / 2
        binary_state = {idx: (1 - val) // 2 for idx, val in sample.items()}
        sample_ids = list(range(next_id, next_id + occ))
        next_id += occ
        ommx_samples.append(
            sample_ids,
            ommx.v1.State({idx: float(val) for idx, val in binary_state.items()}),
        )
    return ommx_instance.evaluate_samples(ommx_samples)


gammas_opt = list(res.x[:p])
betas_opt = list(res.x[p:])
final_shots = 256 if docs_test_mode else 4096

final_result = executable.sample(
    executor,
    shots=final_shots,
    bindings={"gammas": gammas_opt, "betas": betas_opt},
).result()
qaoa_sample_set = evaluate_with_ommx(final_result, spin_model, instance)

qaoa_summary = qaoa_sample_set.summary
qaoa_best = qaoa_sample_set.best_feasible
qaoa_best_E = int(round(qaoa_best.objective))
ref_E = int(reference_solution.objective)

print(f"Shots:                {len(qaoa_summary)}")
print(f"QAOA best objective:  {qaoa_best_E}")
print(f"Reference E*:         {ref_E}")

# %% [markdown]
# ### Distribution of objective values
#
# QAOA returns a distribution over bitstrings, not a single answer. The
# histogram below shows the QUBO objective of every shot at the
# optimized parameters. The red dashed line marks the reference
# optimum $E^\star$. Samples that fall on (or just to the right of) that
# line are the ones whose $x$'s actually minimize the LABS sum and
# whose $z$'s correctly encode the products; samples far to the right
# are paying penalty for inconsistent $z$'s.

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
# The same `ommx.v1.Instance` is consumed by
# `ommx_pyscipopt_adapter.OMMXPySCIPOptAdapter.solve`, which hands the
# problem to the SCIP MILP/QUBO solver via PySCIPOpt and returns an
# `ommx.v1.Solution` evaluated against the *original* instance — so its
# `.objective` is directly comparable to QAOA's.

# %%
import ommx_pyscipopt_adapter

t0 = time.perf_counter()
scip_solution = ommx_pyscipopt_adapter.OMMXPySCIPOptAdapter.solve(instance)
scip_solve_time = time.perf_counter() - t0

scip_E = int(round(scip_solution.objective))
print(f"SCIP E(s):    {scip_E}")
print(f"SCIP feasible: {scip_solution.feasible}")
print(f"Wall time:    {scip_solve_time:.3f} s")

# %% [markdown]
# ## Results comparison
#
# SCIP returns a single optimum deterministically, while QAOA returns a
# *distribution* over bitstrings — so we report QAOA's **best shot**
# (the lowest-objective bitstring seen across all samples) and its
# **hit rate** on the reference optimum (the fraction of shots that
# achieved $E^\star$).

# %%
# Hit rate: fraction of shots that achieved the reference optimum.
hit_rate = float((qaoa_summary["objective"].round().astype(int) == ref_E).mean())

print(f"{'Solver':<22} {'E(s)':>8} {'Time (s)':>12}")
print("-" * 46)
print(f"{'Reference (bundled)':<22} {ref_E:>8} {'-':>12}")
print(f"{'SCIP (classical)':<22} {scip_E:>8} {scip_solve_time:>12.3f}")
print(f"{'QAOA (best shot)':<22} {qaoa_best_E:>8} {qaoa_optimize_time:>12.2f}")
print()
print(f"QAOA hit rate on E* = {ref_E}: {hit_rate:.1%}  ({final_shots} shots)")

# %% [markdown]
# Two observations make this benchmark meaningful:
#
# 1. **Optimum.** Both SCIP and the best QAOA shot reach the reference
#    optimum $E^\star = 2$, so QAOA is *capable* of finding the optimal
#    sequence at $n = 5$ with only $p = 3$ layers.
# 2. **Concentration.** QAOA's value lies in concentrating sampling
#    probability on low-energy bitstrings — the hit rate above (and
#    the left tail of the histogram) is the quantitative version of
#    that statement.
#
# The wall-time column should be read for the *shape* of the comparison,
# not as a head-to-head verdict: SCIP runs natively against the QUBO on
# the CPU, while the QAOA timing covers the full classical-quantum
# optimization loop on a state-vector simulator. As $n$ grows, the
# *qualitative* trade-off — exact-but-exponential branch-and-bound vs.
# heuristic-but-polynomial-depth circuits — is what makes benchmark
# datasets like OMMX Quantum Benchmarks useful for evaluating both
# sides.

# %% [markdown]
# ## Summary
#
# In this tutorial we:
#
# 1. Loaded a LABS instance straight from the **OMMX Quantum Benchmarks**
#    dataset as an `ommx.v1.Instance`.
# 2. Extracted the QUBO with `Instance.to_qubo()`, wrapped it in a
#    `BinaryModel`, switched to the spin domain, and ran a hand-written
#    QAOA ansatz (`@qkernel`) against it through
#    `QiskitTranspiler` + `AerSimulator`.
# 3. Compared the QAOA output (best shot, hit rate, sampling
#    distribution) against SCIP via
#    `ommx_pyscipopt_adapter.OMMXPySCIPOptAdapter.solve` on the *same*
#    instance, plus the reference optimum bundled with the benchmark.
#
# The pattern generalizes: any other QOBLIB dataset
# (`Marketsplit`, `IndependentSet`, `Network`, …) plugs into the same
# pipeline — load with the corresponding `BaseDataset` subclass,
# extract the QUBO via `Instance.to_qubo()`, and reuse the same
# `BinaryModel` + QAOA ansatz + transpile loop. Larger instances will
# eventually outgrow local simulators, at which point the same
# `executable` can be re-targeted to other Qamomile backends
# (`QuriPartsTranspiler`, `CudaqTranspiler`, …) or real hardware.
#
# **Next steps:**
#
# - For the QAOA mathematics itself, see
#   [QAOA for MaxCut](../algorithm/qaoa_maxcut) and
#   [QAOA for Graph Partitioning](../algorithm/qaoa_graph_partition).
# - To swap in a different benchmark family, inspect
#   `Labs().available_instances`, `IndependentSet().available_instances`,
#   etc., from `ommx_quantum_benchmarks.qoblib`.
