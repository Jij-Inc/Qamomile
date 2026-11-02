# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # QURI Parts Support
#
# This page shows how to use Qamomile's QURI Parts integration through a
# concrete optimization problem. We use QAOA for the same small MaxCut
# instance as the algorithm tutorial, compile one typed program to a portable
# logical circuit, and run sampling and expectation-value evaluation with
# Qulacs. We also demonstrate QURI Parts noise and sampler integration.
#
# Qamomile keeps QURI Parts at the realization boundary. QCore remains the only
# persistent program semantics; the adapter constructs SDK circuit and
# simulator objects only when an invocation is launched.

# %% [markdown]
# ## Installation
#
# ```bash
# pip install "qamomile[quri_parts]"
# ```

# %%
# Install the latest Qamomile and its QURI Parts integration through pip!
# # !pip install "qamomile[quri_parts]"

# %%
import math
import os
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from quri_parts.circuit.noise import DepolarizingNoise, NoiseModel
from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label
from quri_parts.core.state import (
    GeneralCircuitQuantumState,
    ParametricCircuitQuantumState,
)
from quri_parts.qulacs.estimator import (
    create_qulacs_vector_estimator,
    create_qulacs_vector_parametric_estimator,
)
from quri_parts.qulacs.sampler import create_qulacs_noisesimulator_sampler
from scipy.optimize import minimize

import qamomile as qm

from qamomile.persistence import deserialize, serialize
from qamomile.optimization import BinaryModel, QAOAExperiment
from qamomile.quri_parts import (
    Qulacs,
    QuriParts,
    QuriPartsSampler,
    load_circuit,
    load_logical_circuit,
)
from qamomile.stdlib import qaoa_state

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# ## Author one runtime-parametric quantum state
#
# `entangled_state` is the sole state-preparation definition. The sampling
# function composes it with measurement, while expectation evaluation consumes
# the state directly. `theta` stays a typed runtime parameter in both workflows.

# %%
@qm.qkernel
def entangled_state(
    theta: qm.Param[qm.Real],
) -> qm.Q[qm.Bits[2]]:  # type: ignore[valid-type]
    """Prepare one runtime-parametric two-qubit entangled state.

    Args:
        theta (qm.Param[qm.Real]): First-qubit Y-rotation angle in radians.

    Returns:
        qm.Q[qm.Bits[2]]: Affine two-qubit state in Qamomile source order.
    """

    state = qm.qubits(2)
    state = qm.ry(state, theta, target=0)
    return qm.cx(state, control=0, target=1)


@qm.qkernel
def entangled_sample(
    theta: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Measure the shared entangled state.

    Args:
        theta (qm.Param[qm.Real]): First-qubit Y-rotation angle in radians.

    Returns:
        qm.Shot[qm.Bits[2]]: Bits in Qamomile source order.
    """

    return qm.measure(entangled_state(theta))


sample_program = entangled_sample.specialize()
study = qm.resources.logical(sample_program)

assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (2, 2)
assert study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure() == (2, 2)

# %% [markdown]
# ## Compile once and retain the executable
#
# `qm.compile` returns a process-local `CompiledProgram` containing the QURI Parts circuit,
# its SDK lifetime owners, and compiler-derived ABI and resource facts. Ordinary
# execution does not export or restore it. The optional `serialize`/`deserialize`
# calls below demonstrate saving through `qamomile.persistence`; loading creates
# a new executable owner. Compare compilation identities, not Python objects.

# %%
target = QuriParts()
artifact = qm.compile(sample_program, to=target)
restored = deserialize(serialize(artifact))

assert artifact.format == "quri-parts.logical-circuit.v2"
assert restored.execution_identity == artifact.execution_identity
assert artifact.execution_identity == qm.compile(sample_program, to=target).execution_identity
assert tuple(slot.name for slot in artifact.parameter_abi.slots) == ("theta",)
assert artifact.result_abi.width == 2

# %% [markdown]
# `load_logical_circuit` is an explicit inspection helper. It decodes the
# adapter payload as ordinary data; it does not construct a second program or
# bypass the common execution contract. Runtime parameter expressions and
# measurement destinations remain inside this portable representation until
# launch.

# %%
logical = load_logical_circuit(restored)
operations = [instruction["op"] for instruction in logical["body"]]

print("schema:", logical["schema"])
print("width :", logical["width"])
print("ops   :", operations)

assert logical["schema"] == 2
assert logical["width"] == 2
assert operations.count("ry") == 1
assert operations.count("cx") == 1
assert operations.count("measure") == 2

# %% [markdown]
# ## Reuse the artifact on isolated Qulacs
#
# Passing an artifact to `qm.run` does not invoke the compiler again. The
# endpoint validates the invocation, materializes a QURI Parts circuit inside a
# worker, binds `theta`, samples Qulacs, and normalizes native bit order once at
# the adapter boundary. The returned `SampleResult` uses the same typed result
# contract as every other Qamomile endpoint.

# %%
endpoint = Qulacs()
cases = ((0.0, (0, 0)), (math.pi, (1, 1)))

for theta, expected in cases:
    record = qm.run(
        restored,
        on=endpoint,
        parameters={"theta": theta},
        shots=8,
        seed=7,
    )
    print(f"theta={theta:.6f} -> {record.result.counts}")
    assert record.artifact is restored
    assert record.result.counts == {expected: 8}
    assert record.result.values == {expected: 8}

# %% [markdown]
# ## The MaxCut problem
#
# We use the same small five-node, six-edge graph from the
# [QAOA for MaxCut tutorial](../algorithm/qaoa_maxcut.ipynb). Maximizing
# $\sum_{(i,j)\in E}(1-s_i s_j)/2$ is equivalent, up to a constant, to
# minimizing $H_C=\sum_{(i,j)\in E}s_i s_j$. The graph therefore maps directly
# to an immutable Ising `BinaryModel`.
#
# ## Building the QAOA ansatz
#
# The ansatz prepares a uniform superposition, alternates cost and mixer layers
# three times, and measures in the computational basis. `QAOAExperiment`
# supplies that same composition while leaving `gammas` and `betas` as runtime
# parameters. Normal execution retains the continuous COBYLA optimization;
# documentation tests reduce only the execution budgets.

# %%
MAXCUT_EDGES = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4))
graph = nx.Graph(MAXCUT_EDGES)
positions = nx.spring_layout(graph, seed=42)

maxcut_model = BinaryModel.from_ising(
    {edge: 0.5 for edge in MAXCUT_EDGES},
    constant=-len(MAXCUT_EDGES) / 2.0,
    variables=range(5),
)
LAYERS = 1 if docs_test_mode else 3
qaoa_experiment = QAOAExperiment(maxcut_model, layers=LAYERS)
compiled_qaoa = qaoa_experiment.compile(to=endpoint.target)

plt.figure(figsize=(5, 4))
nx.draw(
    graph,
    positions,
    with_labels=True,
    node_color="white",
    node_size=700,
    edgecolors="black",
)
plt.title(
    f"MaxCut graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
)
plt.show()

# %% [markdown]
# The portable logical circuit keeps the two QAOA angle vectors unbound. The
# artifact can therefore be reused at every optimizer coordinate without
# rebuilding the program or circuit.

# %%
qaoa_logical = load_logical_circuit(compiled_qaoa.artifact)
qaoa_operations = [instruction["op"] for instruction in qaoa_logical["body"]]

print("QAOA width:", qaoa_logical["width"])
print("QAOA operations:", Counter(qaoa_operations))
assert qaoa_logical["width"] == 5
assert tuple(slot.name for slot in compiled_qaoa.artifact.parameter_abi.slots) == (
    "gammas",
    "betas",
)

# %% [markdown]
# ## Sampling and continuous optimization
#
# Each objective evaluation reuses `compiled_qaoa`, samples Qulacs, and
# minimizes the occurrence-weighted mean Ising energy. The normal page follows
# main with three QAOA layers; the CI path uses one layer and fewer shots.

# %%
OBJECTIVE_SHOTS = 8 if docs_test_mode else 2_000
OPTIMIZER_ITERATIONS = 4 if docs_test_mode else 100
objective_history: list[float] = []


def cost_function(parameters: np.ndarray) -> float:
    """Return sampled mean MaxCut energy for one QAOA coordinate vector."""

    run = compiled_qaoa.run(
        on=endpoint,
        gammas=tuple(float(value) for value in parameters[:LAYERS]),
        betas=tuple(float(value) for value in parameters[LAYERS:]),
        shots=OBJECTIVE_SHOTS,
        seed=42 + len(objective_history),
    )
    objective_history.append(run.samples.mean_energy)
    return run.samples.mean_energy


rng = np.random.default_rng(42)
initial_parameters = rng.uniform(-math.pi / 2.0, math.pi / 2.0, 2 * LAYERS)
optimization = minimize(
    cost_function,
    initial_parameters,
    method="COBYLA",
    options={"maxiter": OPTIMIZER_ITERATIONS, "rhobeg": 0.5},
)
optimized_gammas = tuple(float(value) for value in optimization.x[:LAYERS])
optimized_betas = tuple(float(value) for value in optimization.x[LAYERS:])

print("QAOA objective history:", objective_history)
print("optimized gammas:", optimized_gammas)
print("optimized betas:", optimized_betas)
assert len(objective_history) == optimization.nfev

# %%
plt.figure(figsize=(8, 4))
plt.plot(objective_history, color="#2696EB")
plt.xlabel("Energy evaluation")
plt.ylabel("Cost (mean Ising energy)")
plt.title("QAOA optimization on Qulacs through QURI Parts")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Final distribution and best cut
#
# The optimized artifact is sampled again with the main tutorial's 2,000-shot
# normal budget. Decoded spin values identify the partition and its cut size.

# %%
FINAL_SHOTS = 32 if docs_test_mode else 2_000
best_qaoa_run = compiled_qaoa.run(
    on=endpoint,
    gammas=optimized_gammas,
    betas=optimized_betas,
    shots=FINAL_SHOTS,
    seed=113,
)
best_qaoa_sample = best_qaoa_run.samples.lowest
best_spins = best_qaoa_sample.values
best_cut = sum(best_spins[left] != best_spins[right] for left, right in MAXCUT_EDGES)

energy_counts = Counter()
for sample in best_qaoa_run.samples.samples:
    energy_counts[sample.energy] += sample.occurrences

print("best observed MaxCut sample:", best_qaoa_sample)
print("best observed cut:", best_cut)
assert best_qaoa_run.samples.total_occurrences == FINAL_SHOTS
assert best_qaoa_sample.energy == -float(best_cut)

# %%
energies = sorted(energy_counts)
figure, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(energies, [energy_counts[energy] for energy in energies], width=0.6)
axes[0].axvline(
    best_qaoa_run.samples.mean_energy,
    color="#2B2B2B",
    linestyle="--",
    label=f"mean = {best_qaoa_run.samples.mean_energy:+.3f}",
)
axes[0].set_xlabel("Ising energy")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Optimized QAOA distribution")
axes[0].legend()

node_colors = [
    "#2696EB" if best_spins[node] == 1 else "#EA9B26" for node in graph.nodes()
]
edge_colors = [
    "r" if best_spins[left] != best_spins[right] else "b"
    for left, right in MAXCUT_EDGES
]
nx.draw(
    graph,
    positions,
    ax=axes[1],
    with_labels=True,
    node_color=node_colors,
    edge_color=edge_colors,
    node_size=700,
    edgecolors="black",
)
axes[1].set_title(f"Best partition (cut = {best_cut})")
figure.tight_layout()
plt.show()

# %% [markdown]
# ## Evaluate an observable from the same state source
#
# `qm.expectation` consumes the same standard-library QAOA state composition.
# Rust derives the grouped Pauli measurement work while QURI Parts and Qulacs
# remain the realization and execution route.

# %%
energy_estimate = qm.expectation(
    qaoa_state,
    maxcut_model.hamiltonian,
    on=endpoint,
    parameters={"gammas": optimized_gammas, "betas": optimized_betas},
    shots=16 if docs_test_mode else 4_096,
    seed=211,
    grouping="qwc",
    width=maxcut_model.num_variables,
    layers=LAYERS,
    hamiltonian=maxcut_model.hamiltonian,
)

print(
    f"QURI Parts expectation: {energy_estimate.value:+.6f} "
    f"± {energy_estimate.standard_error:.6f}"
)
assert math.isfinite(energy_estimate.value)

# %% [markdown]
# ## Advanced QURI Parts features
#
# ### Unbound and bound expectation circuits
#
# QURI Parts exposes both parametric and non-parametric estimator paths.
# `load_circuit` projects the portable artifact to an immutable
# `LinearMappedParametricQuantumCircuit`; the artifact itself remains portable
# data. The runtime coordinates are flattened in their first gate-use order,
# with names retained on QURI Parts' input parameters.
#
# An unbound circuit works with a parametric estimator and can be reused across
# optimizer coordinates without making a bound copy. Calling
# `bind_parameters` produces the corresponding bound circuit for a
# non-parametric estimator. Terminal measurement is omitted from this state
# preparation projection, as expected by QURI Parts' estimator state objects.

# %%
quri_hamiltonian = Operator()
quri_hamiltonian.add_term(PAULI_IDENTITY, -len(MAXCUT_EDGES) / 2.0)
for left, right in MAXCUT_EDGES:
    quri_hamiltonian.add_term(pauli_label(f"Z{left} Z{right}"), 0.5)

unbound_circuit = load_circuit(compiled_qaoa.artifact)
parameter_names = tuple(
    parameter.name for parameter in unbound_circuit.param_mapping.in_params
)
named_values = {
    **{f"gammas[{index}]": optimized_gammas[index] for index in range(LAYERS)},
    **{f"betas[{index}]": optimized_betas[index] for index in range(LAYERS)},
}
flat_parameters = [named_values[name] for name in parameter_names]

print(f"unbound type           : {type(unbound_circuit).__name__}")
print(f"unbound parameter_count: {unbound_circuit.parameter_count}")
print(f"circuit parameter order: {parameter_names}")
assert unbound_circuit.parameter_count == 2 * LAYERS

bound_circuit = unbound_circuit.bind_parameters(flat_parameters)
print(f"bound type             : {type(bound_circuit).__name__}")

parametric_state = ParametricCircuitQuantumState(
    maxcut_model.num_variables,
    unbound_circuit,
)
energy_unbound = create_qulacs_vector_parametric_estimator()(
    quri_hamiltonian,
    parametric_state,
    flat_parameters,
).value.real

bound_state = GeneralCircuitQuantumState(
    maxcut_model.num_variables,
    circuit=bound_circuit,
)
energy_bound = create_qulacs_vector_estimator()(
    quri_hamiltonian,
    bound_state,
).value.real

print(f"parametric estimator: {energy_unbound:+.10f}")
print(f"non-param. estimator: {energy_bound:+.10f}")
assert np.isclose(energy_unbound, energy_bound, atol=1e-10)

# %% [markdown]
# Both SDK estimator routes evaluate the same QAOA state and agree to numerical
# precision. `qm.expectation` remains the backend-neutral route; direct QURI
# Parts estimators are available when an application intentionally manages SDK
# circuits or injects another estimator.

# %% [markdown]
# ## Use another QURI Parts sampler
#
# `QuriPartsSampler` accepts a QURI Parts sampler callable while keeping that
# live SDK object outside the portable artifact. The same `compiled_qaoa`
# artifact can therefore run through a simulator, noise model, or provider
# sampler without recompilation. A live sampler has no portable seed contract,
# so this endpoint uses `seed=0` and leaves reproducibility to the injected
# sampler.
#
# The original tutorial demonstrated this extension point with Qulacs'
# `NoiseSimulator`. We restore the same comparison here: the optimized
# parameters and artifact stay fixed, and only the sampler changes.

# %%
noise_model = NoiseModel([DepolarizingNoise(error_prob=0.02)])
noisy_sampler = create_qulacs_noisesimulator_sampler(noise_model)
noisy_endpoint = QuriPartsSampler(noisy_sampler, label="noise-simulator")

noisy_qaoa_run = compiled_qaoa.run(
    on=noisy_endpoint,
    gammas=optimized_gammas,
    betas=optimized_betas,
    shots=FINAL_SHOTS,
    seed=0,
)

clean_energy = best_qaoa_run.samples.mean_energy
noisy_energy = noisy_qaoa_run.samples.mean_energy
print(f"noiseless sampler mean energy: {clean_energy:+.4f}")
print(f"noisy     sampler mean energy: {noisy_energy:+.4f}")
assert noisy_qaoa_run.samples.total_occurrences == FINAL_SHOTS

# %% [markdown]
# Both runs return ordinary Qamomile sample sets, so their energy distributions
# can be compared directly. Depolarizing noise pushes the state toward the
# maximally mixed state; increasing the error probability therefore tends to
# pull this Ising energy toward zero.

# %%
def energy_distribution(samples):
    """Aggregate an optimization sample set by energy."""

    counts = Counter()
    for sample in samples.samples:
        counts[sample.energy] += sample.occurrences
    energies = sorted(counts)
    return energies, [counts[energy] for energy in energies]


figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for axis, samples, mean_energy, title, color in (
    (axes[0], best_qaoa_run.samples, clean_energy, "Noiseless sampler", "#2696EB"),
    (
        axes[1],
        noisy_qaoa_run.samples,
        noisy_energy,
        "NoiseSimulator sampler",
        "#FF8A3D",
    ),
):
    distribution_energies, distribution_counts = energy_distribution(samples)
    axis.bar(distribution_energies, distribution_counts, width=0.6, color=color)
    axis.axvline(
        mean_energy,
        color="#2B2B2B",
        linestyle="--",
        linewidth=1.5,
        label=f"mean = {mean_energy:+.3f}",
    )
    axis.set_xticks(distribution_energies)
    axis.set_title(title)
    axis.set_xlabel("Ising energy")
    axis.legend()

axes[0].set_ylabel("Frequency")
figure.suptitle("Sampled energy distributions by QURI Parts sampler")
figure.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# - QCore is the sole persistent meaning of the typed source program.
# - `QuriParts` emits deterministic portable data, not a stored SDK object.
# - One artifact accepts multiple runtime parameter values on isolated Qulacs.
# - Sampling and expectation reuse one state-preparation definition and the
#   common Qamomile result contracts.
# - The page runs the original five-node, three-layer continuous MaxCut
#   optimization, final distribution, partition, and expectation workflow.
# - `load_circuit` restores direct unbound/bound QURI Parts circuit inspection
#   and both native estimator paths without storing an SDK object in artifacts.
# - The same artifact also runs through an injected QURI Parts
#   `NoiseSimulator` sampler for the original noise comparison.
