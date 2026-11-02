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
# tags: [algorithm, optimization, variational]
# ---
#
# # QAOA for Graph Partitioning
#
# This tutorial demonstrates how to solve the **graph partitioning problem**
# using the Quantum Approximate Optimization Algorithm (QAOA) with Qamomile.
#
# The workflow is:
#
# ```
# JijModeling problem → problem.eval() → QAOAConverter → transpile → sample → decode
# ```
#
# 1. Formulate the problem with [JijModeling](https://jij-inc-jijmodeling-tutorials-en.readthedocs-hosted.com/en/latest/introduction.html).
# 2. Create an instance with concrete data.
# 3. Use `QAOAConverter` to build the QAOA circuit and Hamiltonian.
# 4. Optimize the variational parameters with a classical optimizer.
# 5. Sample the optimized circuit and decode the results.
#

# %% [markdown]
# ## Installation
#
# ```bash
# pip install "qamomile[qiskit,visualization]" jijmodeling
# ```

# %%
# Install the latest Qamomile and JijModeling through pip!
# # !pip install "qamomile[qiskit,visualization]" jijmodeling

# %%
import os

import jijmodeling as jm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import minimize

import qamomile as qm
from qamomile.optimization import BinaryModel, OMMXProblem, QAOAExperiment, VarType
from qamomile.qiskit import Qiskit, load_circuit

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# ## Problem Formulation
#
# Given an undirected graph $G = (V, E)$, the goal is to partition the vertices
# into two groups of equal size while minimizing the number of edges between
# the two groups.
#
# **Objective:**
#
# $$
# \min \sum_{(u,v) \in E} \bigl[x_u(1 - x_v) + x_v(1 - x_u)\bigr]
# $$
#
# **Constraint:**
#
# $$
# \sum_{u \in V} x_u = \frac{|V|}{2}
# $$
#
# where $x_u \in \{0, 1\}$ indicates which partition vertex $u$ belongs to.
#
# ## Define the Problem with JijModeling
#

# %%
formulation = jm.Problem("Graph Partitioning")


@formulation.update
def _(problem: jm.DecoratedProblem):
    vertices = problem.Dim()
    edges = problem.Natural(ndim=2)
    x = problem.BinaryVar(shape=(vertices,))

    problem += (
        edges.rows()
        .map(lambda edge: x[edge[0]] * (1 - x[edge[1]]) + x[edge[1]] * (1 - x[edge[0]]))
        .sum()
    )
    problem += problem.Constraint("Equal Partition", x.sum() == vertices / 2)


formulation

# %% [markdown]
# ## Graph Instance
#
# We use a fixed 8-node graph with 16 edges for reproducibility.
#

# %%
NUM_NODES = 8
EDGE_LIST = (
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 7),
    (2, 3),
    (2, 6),
    (3, 5),
    (4, 5),
    (4, 6),
    (5, 6),
    (5, 7),
    (6, 7),
)

graph = nx.Graph()
graph.add_nodes_from(range(NUM_NODES))
graph.add_edges_from(EDGE_LIST)
positions = nx.spring_layout(graph, seed=1)

assert graph.number_of_nodes() == NUM_NODES
assert graph.number_of_edges() == len(EDGE_LIST) == 16

# %%
plt.figure(figsize=(5, 5))
nx.draw(
    graph,
    positions,
    with_labels=True,
    node_color="white",
    node_size=700,
    edgecolors="black",
)
plt.title(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
plt.show()

# %% [markdown]
# ## Create the Instance and Penalty Model
#
# `OMMXProblem.from_instance` copies the caller's instance, absorbs the equal
# partition constraint into a binary penalty model, and retains the original
# bytes for evaluation after sampling. The caller-owned instance is never
# mutated. Penalty energy is used only for optimization; feasibility and the
# reported cut count always come from the original OMMX instance.

# %%
instance = formulation.eval(
    {
        "vertices": NUM_NODES,
        "edges": [list(edge) for edge in EDGE_LIST],
    }
)
original_payload = instance.to_bytes()
problem = OMMXProblem.from_instance(instance, uniform_penalty_weight=16.0)

assert instance.to_bytes() == original_payload
assert problem.model.vartype is VarType.BINARY
assert problem.model.num_variables == NUM_NODES
assert problem.model.order == 2
assert BinaryModel.from_bytes(problem.model.to_bytes()) == problem.model

print(f"Penalty terms: {len(problem.model.coefficients)}")
print(f"QAOA Hamiltonian terms: {len(problem.model.hamiltonian.terms)}")

# %% [markdown]
# ## Build and Compile the QAOA Experiment
#
# Normal execution uses the original five QAOA layers. Documentation tests use
# one layer but follow the same compilation, optimizer, sampling, and OMMX
# evaluation path. Only fixed-length `gammas` and `betas` remain in the runtime
# ABI, so every optimizer proposal reuses one artifact.

# %%
LAYERS = 1 if docs_test_mode else 5
experiment = QAOAExperiment(problem.model, layers=LAYERS)
endpoint = qm.Reference()
compiled = experiment.compile(to=endpoint.target)

assert tuple(slot.name for slot in compiled.artifact.parameter_abi.slots) == (
    "gammas",
    "betas",
)
assert compiled.identity == experiment.identity
assert compiled.identity.problem_digest == problem.model.semantic_digest
assert "quantum.qaoa" not in experiment.program.inspect()

# %% [markdown]
# ## Visualize the Circuit and Inspect Its Building Blocks
#
# QAOA is ordinary composition: an initial Hadamard on every qubit, followed by
# alternating cost evolution and X-mixer rotations. For two layers, the Qiskit
# projection makes the repeated `rzz`/`rz` cost realization and `rx` mixers
# visible without exposing a mutable compiler-pass API.

# %%
visualization_experiment = QAOAExperiment(problem.model, layers=2)
visualization_artifact = qm.compile(visualization_experiment.program, to=Qiskit())
visualization_circuit = load_circuit(visualization_artifact)

print(visualization_circuit.draw(output="text", fold=120))
print("Circuit operations:", dict(visualization_circuit.count_ops()))
visualization_circuit.draw(output="mpl", fold=-1)

# %% [markdown]
# Resource analysis sees the same composed body. With `LAYERS` layers it reports
# one Hadamard per problem qubit, one abstract Pauli evolution per cost layer,
# and one RX mixer rotation per qubit and layer.

# %%
study = qm.resources.logical(experiment.program)

assert study.metric(
    qm.resources.logical_operation_count("quantum.h")
).numeric_enclosure() == (NUM_NODES, NUM_NODES)
assert study.metric(
    qm.resources.logical_operation_count("quantum.rx")
).numeric_enclosure() == (NUM_NODES * LAYERS, NUM_NODES * LAYERS)
assert study.metric(
    qm.resources.logical_operation_count("quantum.pauli_evolution")
).numeric_enclosure() == (LAYERS, LAYERS)

# %% [markdown]
# ## Optimize the QAOA Parameters
#
# COBYLA minimizes the occurrence-weighted penalty energy returned by each
# sampled QAOA run. This is the correct optimizer coordinate: using the
# unpenalized original cut objective here would allow infeasible all-zero or
# all-one assignments. Normal execution retains the original 1,000-proposal
# optimizer and uses 256 shots per launch, which stays inside the Reference
# endpoint's per-launch work guard. Documentation tests reduce both budgets.

# %%
SAMPLE_SHOTS = 16 if docs_test_mode else 256
MAX_ITERATIONS = 6 if docs_test_mode else 1_000
FINAL_SHOTS = 128 if docs_test_mode else 256
cost_history = []


def cost_function(parameters) -> float:
    """Sample one QAOA coordinate and return its mean penalty energy."""

    run = compiled.run(
        on=endpoint,
        gammas=tuple(float(value) for value in parameters[:LAYERS]),
        betas=tuple(float(value) for value in parameters[LAYERS:]),
        shots=SAMPLE_SHOTS,
        seed=901 + len(cost_history),
    )
    cost_history.append(run.samples.mean_energy)
    return run.samples.mean_energy


rng = np.random.default_rng(900)
initial_parameters = rng.uniform(0.0, np.pi, 2 * LAYERS)
optimization = minimize(
    cost_function,
    initial_parameters,
    method="COBYLA",
    options={"maxiter": MAX_ITERATIONS, "rhobeg": 0.5},
)

print(f"Optimized penalty cost: {optimization.fun:+.6f}")
print(f"Function evaluations: {optimization.nfev}")
assert len(optimization.x) == 2 * LAYERS
assert len(cost_history) == optimization.nfev

# %%
plt.figure(figsize=(8, 4))
plt.plot(cost_history, color="#2696EB")
plt.xlabel("Energy evaluation")
plt.ylabel("Penalty mean energy")
plt.title("QAOA optimization progress")
plt.show()

# %% [markdown]
# ## Sample with Optimized Parameters
#
# A final, larger run collects candidate partitions. The decoded distribution
# remains in the penalty model's BINARY domain until `problem.evaluate` asks the
# untouched original OMMX instance for feasibility, constraint diagnostics,
# and the true objective.

# %%
optimized_run = compiled.run(
    on=endpoint,
    gammas=tuple(float(value) for value in optimization.x[:LAYERS]),
    betas=tuple(float(value) for value in optimization.x[LAYERS:]),
    shots=FINAL_SHOTS,
    seed=1901,
)
evaluation = problem.evaluate(optimized_run.samples)

feasible_shots = sum(
    evaluation.occurrences[sample_id]
    for sample_id, feasible in evaluation.feasible.items()
    if feasible
)

print(
    f"Feasible samples: {feasible_shots} / {FINAL_SHOTS} "
    f"({100.0 * feasible_shots / FINAL_SHOTS:.1f}%)"
)

assert optimized_run.execution.artifact is compiled.artifact
assert optimized_run.samples.total_occurrences == FINAL_SHOTS
assert evaluation.total_occurrences == FINAL_SHOTS
assert feasible_shots > 0

# %% [markdown]
# ## Analyze the Results
#
# ### Best Feasible Solution
#
# `best_feasible` returns the smallest original cut objective among feasible
# unique rows. Decision-variable values use the original IDs, and the equality
# constraint can be checked directly.

# %%
best = evaluation.best_feasible
values = best.decision_variables_df["value"]
best_assignment = {
    index: int(round(float(values.loc[index]))) for index in range(NUM_NODES)
}
best_objective = int(round(best.objective))
cut_edges = sum(
    best_assignment[left] != best_assignment[right] for left, right in EDGE_LIST
)

print("Best feasible partition:", best_assignment)
print(f"Cut edges: {best_objective}")

assert sum(best_assignment.values()) == NUM_NODES // 2
assert best_objective == cut_edges
assert best_objective >= 0

# %% [markdown]
# ### Objective Value Distribution
#
# The immutable evaluation stores one row per unique solution plus its original
# occurrence count. Grouping feasible occurrences by the original objective
# reproduces the shot-level histogram without expanding duplicate rows.

# %%
objective_counts = {}
for sample_id in evaluation.sample_ids:
    if not evaluation.feasible[sample_id]:
        continue
    objective = int(round(evaluation.objectives[sample_id]))
    objective_counts[objective] = (
        objective_counts.get(objective, 0) + evaluation.occurrences[sample_id]
    )

assert sum(objective_counts.values()) == feasible_shots

plt.figure(figsize=(8, 4))
plt.bar(
    [str(value) for value in sorted(objective_counts)],
    [objective_counts[value] for value in sorted(objective_counts)],
    color="#2696EB",
)
plt.xlabel("Cut edges (objective value)")
plt.ylabel("Frequency")
plt.title("Distribution of feasible solutions")
plt.show()

# %% [markdown]
# ### Visualize the Best Partition
#
# Node colors show the two equal groups of the best feasible QAOA sample.

# %%
colors = [
    "#FF6B6B" if best_assignment[index] == 1 else "#4ECDC4"
    for index in range(NUM_NODES)
]
plt.figure(figsize=(5, 5))
nx.draw(
    graph,
    positions,
    with_labels=True,
    node_color=colors,
    node_size=700,
    edgecolors="black",
)
plt.title(f"Best feasible partition (cut edges = {best_objective})")
plt.show()

# %% [markdown]
# ## Summary
#
# - The original eight-node constrained JijModeling instance is preserved.
# - A five-layer QAOA program is compiled once and continuously optimized with
#   the penalty-domain mean energy.
# - Full-circuit visualization and logical resources expose the Hadamard, cost,
#   and mixer building blocks.
# - Final sampling reports original OMMX feasibility, best objective,
#   objective-value distribution, and the graph partition without conflating
#   penalty energy with application meaning.
