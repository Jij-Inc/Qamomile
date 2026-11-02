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
# # Qiskit Support
#
# This page shows how to use Qamomile's Qiskit integration through a concrete
# optimization problem. Qiskit support is optional; install the `qiskit` extra
# to use the `Qiskit` target, local `Aer`, and live `QiskitBackend` endpoints.
# We use QAOA optimization for a small MaxCut instance, compile one Qamomile
# qkernel to a QuantumCircuit, and run sampling and expectation-value evaluation on Qiskit.
# Along the way, we also examine advanced Qiskit features and IBM Quantum
# execution through the same artifact and result contracts.
#
# Qamomile keeps Qiskit at the realization boundary. QCore remains the only
# persistent program semantics; a Qiskit `QuantumCircuit` is a temporary
# inspection or provider object, never a second source representation.

# %% [markdown]
# ## Installation
#
# ```bash
# pip install "qamomile[qiskit]"
# ```

# %%
# Install the latest Qamomile and its Qiskit integration through pip!
# # !pip install "qamomile[qiskit]"

# %%
import math
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit.providers.basic_provider import BasicSimulator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from scipy.optimize import minimize

import qamomile as qm
from qamomile.optimization import BinaryModel, QAOAExperiment
from qamomile.persistence import deserialize, serialize
from qamomile.qiskit import Aer, Qiskit, QiskitBackend, load_circuit, to_circuit
from qamomile.stdlib import qaoa_state, qft

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# ## Author one runtime-parametric source program
#
# The program rotates the first qubit, entangles the second, and returns a typed
# two-bit sample. `theta` remains a runtime parameter after compilation, so one
# compiled circuit can serve several invocations.


# %%
@qm.qkernel
def entangled_sample(
    theta: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Measure one runtime-parametric two-qubit entangled state.

    Args:
        theta (qm.Param[qm.Real]): First-qubit Y-rotation angle in radians.

    Returns:
        qm.Shot[qm.Bits[2]]: Bits in Qamomile source order.
    """

    state = qm.qubits(2)
    state = qm.ry(state, theta, target=0)
    state = qm.cx(state, control=0, target=1)
    return qm.measure(state)


program = entangled_sample.specialize()
study = qm.resources.logical(program)

assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (2, 2)
assert study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure() == (2, 2)

# %% [markdown]
# ## Compile once and retain the executable
#
# `qm.compile` returns a process-local `CompiledProgram` containing the Qiskit QuantumCircuit,
# its SDK lifetime owners, and compiler-derived ABI and resource facts. Ordinary
# execution does not export or restore it. The optional `serialize`/`deserialize`
# calls below demonstrate saving through `qamomile.persistence`; loading creates
# a new executable owner. Compare compilation identities, not Python objects.

# %%
target = Qiskit()
artifact = qm.compile(program, to=target)
restored = deserialize(serialize(artifact))

assert artifact.format == "qiskit.qpy.v1"
assert restored.execution_identity == artifact.execution_identity
assert artifact.execution_identity == qm.compile(program, to=target).execution_identity
assert tuple(slot.name for slot in artifact.parameter_abi.slots) == ("theta",)
assert artifact.result_abi.width == 2

# %% [markdown]
# `load_circuit` is an explicit adapter inspection helper. It reconstructs a
# copy of the retained Qiskit circuit without changing the artifact or becoming an alternate
# execution path. Qamomile parameters stay unbound until invocation. The circuit
# projection uses compact positional names such as `p0`; the sealed parameter
# ABI retains the user-facing `theta` name and owns that mapping.

# %%
circuit = load_circuit(restored)

print(circuit.draw(output="text", fold=80))
print("Qiskit operations:", dict(circuit.count_ops()))

assert circuit.num_qubits == 2
assert circuit.num_clbits == 2
assert circuit.num_parameters == 1
assert {str(parameter) for parameter in circuit.parameters} == {"p0"}
assert circuit.count_ops()["measure"] == 2

# %% [markdown]
# ## Reuse the artifact on isolated Aer
#
# Passing an artifact to `qm.run` never invokes the compiler again. The Aer
# endpoint binds the canonical `theta` value, executes a bound circuit in the calling process,
# normalizes Qiskit's native bit order once, and returns the same typed
# `SampleResult` used by every Qamomile endpoint.

# %%
endpoint = Aer()
smoke_endpoint = qm.Reference() if docs_test_mode else endpoint
cases = ((0.0, (0, 0)), (math.pi, (1, 1)))

for theta, expected in cases:
    record = qm.run(
        program if docs_test_mode else restored,
        on=smoke_endpoint,
        parameters={"theta": theta},
        shots=8,
        seed=7,
    )
    print(f"theta={theta:.6f} -> {record.result.counts}")
    if not docs_test_mode:
        assert record.artifact is restored
    assert record.result.counts == {expected: 8}
    assert record.result.values == {expected: 8}

# %% [markdown]
# ## The MaxCut problem
#
# We use the same small five-node graph from the
# [QAOA for MaxCut tutorial](../algorithm/qaoa_maxcut.ipynb) so the focus stays
# on the Qiskit integration. Maximizing
# $\sum_{(i,j)\in E}(1-s_i s_j)/2$ is equivalent, up to a constant, to
# minimizing the antiferromagnetic Ising Hamiltonian
# $H_C=\sum_{(i,j)\in E}s_i s_j$. For unweighted MaxCut every edge has the same
# coupling and there are no linear terms, so those coefficients are passed
# directly to `BinaryModel.from_ising`.
#
# ## Building the QAOA ansatz
#
# The recipe is the same as in the MaxCut tutorial: prepare a uniform
# superposition, alternate cost and mixer layers three times, and measure in
# the computational basis. `QAOAExperiment` closes that immutable spin model
# and topology into one typed program, while `gammas` and `betas` remain runtime
# parameters. Documentation tests reduce only the layer and execution budgets.

# %%
MAXCUT_EDGES = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4))
graph = nx.Graph()
graph.add_edges_from(MAXCUT_EDGES)
positions = nx.spring_layout(graph, seed=42)
nx.draw(graph, positions, with_labels=True, node_color="white", edgecolors="black")
plt.title("Five-node MaxCut instance")
plt.show()

maxcut_model = BinaryModel.from_ising(
    {edge: 0.5 for edge in MAXCUT_EDGES},
    constant=-len(MAXCUT_EDGES) / 2.0,
    variables=range(5),
)
LAYERS = 1 if docs_test_mode else 3
qaoa_experiment = QAOAExperiment(maxcut_model, layers=LAYERS)
optimization_endpoint = QiskitBackend(
    AerSimulator(
        max_parallel_threads=1,
        max_parallel_experiments=1,
        max_parallel_shots=1,
    )
)
qaoa_endpoint = optimization_endpoint
compiled_qaoa = qaoa_experiment.compile(to=qaoa_endpoint.target)
optimization_qaoa = qaoa_experiment.compile(to=optimization_endpoint.target)

OBJECTIVE_SHOTS = 8 if docs_test_mode else 2_000
OPTIMIZER_ITERATIONS = 4 if docs_test_mode else 100
rng = np.random.default_rng(42)
initial_parameters = rng.uniform(-math.pi / 2.0, math.pi / 2.0, 2 * LAYERS)
objective_history: list[float] = []


def cost_function(parameters: np.ndarray) -> float:
    """Return sampled mean MaxCut energy for one QAOA coordinate vector."""

    run = optimization_qaoa.run(
        on=optimization_endpoint,
        gammas=tuple(float(value) for value in parameters[:LAYERS]),
        betas=tuple(float(value) for value in parameters[LAYERS:]),
        shots=OBJECTIVE_SHOTS,
        seed=0,
    )
    objective_history.append(run.samples.mean_energy)
    return run.samples.mean_energy


optimization = minimize(
    cost_function,
    initial_parameters,
    method="COBYLA",
    options={"maxiter": OPTIMIZER_ITERATIONS},
)
optimized_gammas = tuple(float(value) for value in optimization.x[:LAYERS])
optimized_betas = tuple(float(value) for value in optimization.x[LAYERS:])

FINAL_SHOTS = 32 if docs_test_mode else 2_000
best_qaoa_run = compiled_qaoa.run(
    on=qaoa_endpoint,
    gammas=optimized_gammas,
    betas=optimized_betas,
    shots=FINAL_SHOTS,
    seed=0,
)
best_qaoa_sample = best_qaoa_run.samples.lowest

print("QAOA objective history:", objective_history)
print("optimized gammas:", optimized_gammas)
print("optimized betas:", optimized_betas)
print("best observed MaxCut sample:", best_qaoa_sample)
assert best_qaoa_run.samples.total_occurrences == FINAL_SHOTS
assert best_qaoa_sample.energy <= -4.0

# %% [markdown]
# ### Expectation values from the optimized state
#
# The sampling objective above is an occurrence-weighted energy estimate. The
# same optimized coordinates can also enter Qamomile's observable expectation
# path. This restores the main tutorial's explicit estimator workflow while
# keeping Qiskit only at the realization boundary.


# %%
@qm.qkernel
def qaoa_energy_state(
    hamiltonian: qm.Const[qm.Observable],
    gammas: qm.Param[qm.Vector[qm.Real, 3]],  # type: ignore[valid-type]
    betas: qm.Param[qm.Vector[qm.Real, 3]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[5]]:  # type: ignore[valid-type]
    """Prepare a fixed three-layer state for expectation estimation."""

    return qaoa_state(5, 3, hamiltonian, gammas, betas)


expectation_gammas = optimized_gammas + (0.0,) * (3 - LAYERS)
expectation_betas = optimized_betas + (0.0,) * (3 - LAYERS)
energy_estimate = qm.expectation(
    qaoa_energy_state,
    maxcut_model.hamiltonian,
    on=qaoa_endpoint,
    hamiltonian=maxcut_model.hamiltonian,
    parameters={"gammas": expectation_gammas, "betas": expectation_betas},
    shots=8 if docs_test_mode else 256,
    seed=0,
    grouping="qwc",
)

print("optimized expectation:", energy_estimate.value)
assert math.isfinite(energy_estimate.value)

# %% [markdown]
# ### Reuse the optimized program with an Aer noise model
#
# A live `QiskitBackend` can wrap an `AerSimulator` configured with SDK-native
# noise. We compile the same QAOA program to that backend snapshot, bind the
# same optimized coordinates, and compare sampled mean energies. Noise remains
# endpoint configuration rather than QCore semantics.

# %%
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(
    depolarizing_error(0.01, 1),
    ["h", "rx", "rz"],
)
noise_model.add_all_qubit_quantum_error(
    depolarizing_error(0.02, 2),
    ["rzz"],
)
noisy_endpoint = QiskitBackend(
    AerSimulator(
        noise_model=noise_model,
        max_parallel_threads=1,
        max_parallel_experiments=1,
        max_parallel_shots=1,
    )
)
noisy_qaoa = qaoa_experiment.compile(to=noisy_endpoint.target)
noisy_run = noisy_qaoa.run(
    on=noisy_endpoint,
    gammas=optimized_gammas,
    betas=optimized_betas,
    shots=16 if docs_test_mode else 2_000,
    seed=0,
)

print("clean mean energy:", best_qaoa_run.samples.mean_energy)
print("noisy mean energy:", noisy_run.samples.mean_energy)
assert math.isfinite(noisy_run.samples.mean_energy)

# %% [markdown]
# `DecodedSamples.mean_energy` is the sampled expectation value of the cost
# Hamiltonian. COBYLA minimizes the same scalar by reusing `compiled_qaoa`;
# neither the Qamomile program nor the compiled circuit is rebuilt between
# coordinates. Native Qiskit primitives or provider estimators remain
# available after loading the circuit, while `qm.expectation` provides the
# backend-neutral route when a state-returning source is preferred.

# %% [markdown]
# ## Advanced Qiskit features
#
# ### Classical control flow and runtime expressions
#
# A measurement-dependent Python branch remains part of QCore until the Qiskit
# emitter converts it to a native `if_else` region. The private feedback bit is
# not exposed in the declared two-bit result.


# %%
@qm.qkernel
def feedback_sample() -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Correlate two output bits through measurement feedback.

    Returns:
        qm.Shot[qm.Bits[2]]: Two equal source-ordered bits.
    """

    state = qm.h(qm.qubits(2), target=0)
    flag, state = qm.measure_z(state, 0)
    if flag:
        state = qm.x(state, target=1)
    else:
        state = qm.z(state, target=1)
    return qm.measure(state)


feedback_artifact = qm.compile(feedback_sample, to=qaoa_endpoint.target)
feedback_circuit = load_circuit(feedback_artifact)
feedback_result = qm.run(
    feedback_artifact,
    on=qaoa_endpoint,
    shots=32,
    seed=0,
).result

assert "if_else" in feedback_circuit.count_ops()
assert set(feedback_result.counts) == {(0, 0), (1, 1)}

# %% [markdown]
# A structural `for` loop that does not use its induction value can likewise
# remain one native `for_loop`. Qiskit owns this realization choice; the QCore
# source still carries an ordinary verified `qcore.for`.


# %%
@qm.qkernel
def repeated_h(
    repetitions: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[1]]:  # type: ignore[valid-type]
    """Apply Hadamard repeatedly through one structural loop.

    Args:
        repetitions: Compile-time loop trip count.

    Returns:
        qm.Shot[qm.Bits[1]]: One measured output bit.
    """

    state = qm.qubits(1)
    for _ in range(repetitions):
        state = qm.h(state, target=0)
    return qm.measure(state)


for_artifact = qm.compile(qm.specialize(repeated_h, repetitions=3), to=target)
for_circuit = load_circuit(for_artifact)
for_result = qm.run(for_artifact, on=Aer(), shots=16, seed=1).result

print("structural-loop Qiskit operations:", dict(for_circuit.count_ops()))
assert for_circuit.count_ops().get("for_loop", 0) == 1
assert for_result.shots == 16


# %% [markdown]
# ### Pauli evolution realization
#
# The QAOA cost layer stays one abstract commuting-Pauli evolution in QCore.
# Qiskit preserves each complete cost layer as one native
# `PauliEvolutionGate` in the retained circuit, including its runtime angle. Aer expands the
# gate's verified exact fallback schedule only at execution. Helper definitions
# and loops may enclose these operations, so top-level counts are not a flattened
# count of all Pauli evolution gates.

# %%
native_qaoa_artifact = qaoa_experiment.compile(to=target).artifact
qaoa_circuit = load_circuit(native_qaoa_artifact)
print("QAOA Qiskit operations:", dict(qaoa_circuit.count_ops()))
assert qaoa_circuit.num_qubits == native_qaoa_artifact.result_abi.width
assert len(qaoa_circuit.parameters) == 2 * LAYERS

# %% [markdown]
# ### QFT realization
#
# A named composite preserves the verified standard-library QFT body as one
# labeled SDK gate. Calling the ordinary qkernel emits that same little-endian
# `h`/`cp`/`swap` body directly. A name such as `qft` does not establish a
# mathematical contract and cannot authorize substitution with `QFTGate`.
# Native arithmetic and Pauli choices instead use compiler-verified semantic
# operations and explicit target capabilities.

# %%
boxed_qft = qm.composite_gate(qft, name="qft")


@qm.qkernel
def qft_sample() -> qm.Shot[qm.Bits[3]]:  # type: ignore[valid-type]
    """Measure the boxed three-qubit QFT of the zero state.

    Returns:
        qm.Shot[qm.Bits[3]]: Three source-ordered output bits.
    """

    return qm.measure(boxed_qft(qm.qubits(3), 3))


@qm.qkernel
def decomposed_qft_sample() -> qm.Shot[qm.Bits[3]]:  # type: ignore[valid-type]
    """Measure the portable three-qubit QFT fallback body.

    Returns:
        qm.Shot[qm.Bits[3]]: Three source-ordered output bits.
    """

    return qm.measure(qft(qm.qubits(3), 3))


qft_artifact = qm.compile(qft_sample, to=target)
qft_circuit = load_circuit(qft_artifact)
decomposed_qft_circuit = to_circuit(decomposed_qft_sample, target=target)
boxed_qft_operations = qft_circuit.count_ops()
from qiskit import transpile

expanded_qft = transpile(
    decomposed_qft_circuit, basis_gates=["h", "cp", "swap"], optimization_level=0
)
decomposed_qft_operations = expanded_qft.count_ops()
qft_result = qm.run(qft_artifact, on=Aer(), shots=8, seed=0).result

print("boxed QFT operations:", dict(boxed_qft_operations))
print("decomposed QFT operations:", dict(decomposed_qft_operations))
assert sum(row.operation.name != "measure" for row in qft_circuit.data) == 1
assert decomposed_qft_operations.get("h", 0) == 3
assert decomposed_qft_operations.get("cp", 0) == 3
assert decomposed_qft_operations.get("swap", 0) == 1
assert qft_result.shots == 8

# %% [markdown]
# ## Use the same source with a live BackendV2 adapter
#
# `QiskitBackend` owns a live SDK object only at the endpoint boundary. Its
# `target` snapshots backend capabilities for routing and fingerprinting; the
# object itself never enters QCore or serialized artifact data. Real providers
# use the same adapter. This local example uses Qiskit's lightweight
# `BasicSimulator` and deliberately leaves `seed=0`, because live backends do not
# promise a portable seed contract.

# %%
live_endpoint = QiskitBackend(BasicSimulator())
live_record = qm.run(
    program,
    on=live_endpoint,
    parameters={"theta": math.pi},
    shots=4,
    seed=0,
)

print("live backend:", live_record.result.backend)
print("routed stages:", live_record.artifact.report.stages)

assert live_record.result.counts == {(1, 1): 4}
assert "qiskit-routing" in live_record.artifact.report.stages
assert live_record.artifact.target_fingerprint == live_endpoint.target.fingerprint

# %% [markdown]
# ## Run on IBM Quantum hardware
#
# A provider backend uses the same live adapter. Credentials and queueing stay
# outside artifacts, so this cost-bearing example is intentionally not run by
# the documentation build:
#
# ```python
# from qiskit_ibm_runtime import QiskitRuntimeService
#
# service = QiskitRuntimeService()
# backend = service.least_busy(operational=True, simulator=False)
# hardware = QiskitBackend(backend)
# hardware_artifact = qm.compile(qaoa_experiment.program, to=hardware.target)
# job = qm.submit(
#     hardware_artifact,
#     on=hardware,
#     parameters={"gammas": optimized_gammas, "betas": optimized_betas},
#     shots=1_000,
#     seed=0,
# )
# print(job.job_id, job.status)
# ```

# %% [markdown]
# ## Summary
#
# - QCore is the sole persistent meaning of the typed source program.
# - `Qiskit` changes only realization by retaining SDK operations in a QuantumCircuit.
# - One compiled artifact accepts multiple runtime parameter values on `Aer`.
# - `QiskitBackend` keeps provider objects, routing, and native jobs at the
#   endpoint boundary while preserving the common Qamomile result ABI.
# - The page also executes the original five-node, three-layer continuous
#   MaxCut optimization, native dynamic `for_loop` / `if_else`,
#   native `PauliEvolutionGate`, boxed-versus-decomposed QFT inspection, and a
#   live BackendV2 route. Documentation tests reduce only layers, shots, and
#   optimizer iterations.
