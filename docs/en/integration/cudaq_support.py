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
# # CUDA-Q Support
#
# This page shows how to use Qamomile's CUDA-Q integration through a concrete
# optimization problem. We use QAOA for the same small MaxCut instance as the
# algorithm tutorial, compile one typed program to serialized CUDA-Q kernel
# data, and run sampling and expectation-value evaluation on CUDA-Q targets.
# We also compare CPU and optional NVIDIA targets and examine kernels with
# classical control flow.
#
# Qamomile keeps CUDA-Q at the realization boundary. QCore remains the only
# persistent program semantics; a CUDA-Q kernel is a temporary SDK object
# used for inspection or execution, never a second source representation.

# %% [markdown]
# ## Installation
#
# Choose the optional dependency group matching the installed CUDA toolkit:
#
# ```bash
# pip install "qamomile[cudaq-cu12]"  # CUDA 12.x on Linux
# pip install "qamomile[cudaq-cu13]"  # CUDA 13.x on Linux or macOS ARM64
# ```

# %%
# Install the latest Qamomile and its CUDA-Q integration through pip!
# # !pip install "qamomile[cudaq-cu13]"

# %%
import math
import os
import time
from collections import Counter

import cudaq
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import minimize

import qamomile as qm

from qamomile.persistence import deserialize, export_payload, serialize
from qamomile.cudaq import Cudaq, CudaqLocal, load_kernel
from qamomile.optimization import BinaryModel, QAOAExperiment
from qamomile.stdlib import qaoa_state

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# ## Author one runtime-parametric quantum state
#
# `parameterized_state` is the sole state-preparation definition. Its two basis
# endpoints are deliberately asymmetric: `theta = 0` produces `01`, while
# `theta = pi` produces `10`. That makes a reversed native bit order visible.


# %%
@qm.qkernel
def parameterized_state(
    theta: qm.Param[qm.Real],
) -> qm.Q[qm.Bits[2]]:  # type: ignore[valid-type]
    """Prepare one runtime-parametric asymmetric two-qubit state.

    Args:
        theta (qm.Param[qm.Real]): First-qubit Y-rotation angle in radians.

    Returns:
        qm.Q[qm.Bits[2]]: Affine two-qubit state in Qamomile source order.
    """

    state = qm.qubits(2)
    state = qm.ry(state, theta, target=0)
    state = qm.x(state, target=1)
    return qm.cx(state, control=0, target=1)


@qm.qkernel
def entangled_sample(
    theta: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Measure the shared parameterized state.

    Args:
        theta (qm.Param[qm.Real]): First-qubit Y-rotation angle in radians.

    Returns:
        qm.Shot[qm.Bits[2]]: Bits in Qamomile source order.
    """

    return qm.measure(parameterized_state(theta))


sample_program = entangled_sample.specialize()
study = qm.resources.logical(sample_program)

assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (3, 3)
assert study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure() == (2, 2)

# %% [markdown]
# ## Compile once and retain the executable
#
# `qm.compile` returns a process-local `CompiledProgram` containing the CUDA-Q kernel,
# its SDK lifetime owners, and compiler-derived ABI and resource facts. Ordinary
# execution does not export or restore it. The optional `serialize`/`deserialize`
# calls below demonstrate saving through `qamomile.persistence`; loading creates
# a new executable owner. Compare compilation identities, not Python objects.

# %%
target = Cudaq()
artifact = qm.compile(sample_program, to=target)
restored = deserialize(serialize(artifact))

assert artifact.format == "cudaq.python-source.v1"
assert restored.execution_identity == artifact.execution_identity
assert artifact.execution_identity == qm.compile(sample_program, to=target).execution_identity
assert tuple(slot.name for slot in artifact.parameter_abi.slots) == ("theta",)
assert artifact.result_abi.width == 2

# %% [markdown]
# Generated Python source is available through explicit export. `load_kernel`
# accesses the retained kernel for inspection. Repeated launches reuse the kernel
# and helper owners for the selected CUDA-Q target.

# %%
generated_source = export_payload(restored).decode("utf-8")
kernel = load_kernel(restored)

print(generated_source)
print("CUDA-Q kernel type:", type(kernel).__name__)

assert isinstance(generated_source, str)
assert "@cudaq.kernel" in generated_source
assert "q = cudaq.qvector(2)" in generated_source
assert "ry(p0, q[0])" in generated_source
assert "x(q[1])" in generated_source
assert "x.ctrl(q[0], q[1])" in generated_source

# %% [markdown]
# ## Run the artifact on the CPU simulator
#
# Passing an artifact to `qm.run` does not invoke the compiler again. The
# endpoint validates the invocation and executes in the calling process. It
# restores the kernel, selects CUDA-Q's `qpp-cpu` target, binds `theta`, and
# normalizes native bit order once. Qamomile serializes its CUDA-Q launches and
# restores the previous SDK target afterward. One deterministic asymmetric
# launch proves both the runtime parameter path and source-order result contract.

# %%
endpoint = CudaqLocal(runtime_target="qpp-cpu")
record = qm.run(
    restored,
    on=endpoint,
    parameters={"theta": math.pi},
    shots=8,
    seed=7,
)

print(record.result.counts)
assert record.artifact is restored
assert record.result.backend == "cudaq.qpp-cpu"
assert record.result.counts == {(1, 0): 8}
assert record.result.values == {(1, 0): 8}

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
# parameters. Normal execution uses 2,000 shots per proposal and a 100-proposal
# COBYLA loop; documentation tests reduce only those budgets.

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
# The serialized CUDA-Q artifact retains only two runtime vectors. Generated
# source exposes the layered Hadamard, Pauli-evolution realization, mixer, and
# final measurements while QCore remains the persistent semantics.

# %%
qaoa_source = export_payload(compiled_qaoa.artifact).decode("utf-8")
print(qaoa_source)

assert compiled_qaoa.artifact.format == "cudaq.python-source.v1"
assert tuple(slot.name for slot in compiled_qaoa.artifact.parameter_abi.slots) == (
    "gammas",
    "betas",
)
assert isinstance(qaoa_source, str) and "@cudaq.kernel" in qaoa_source

# %% [markdown]
# ## Sampling and continuous optimization
#
# Each objective call reuses the one compiled artifact, samples one coordinate
# on `qpp-cpu`, and minimizes the decoded mean Ising energy. Qamomile keeps the
# aggregate vector ABI and source-order bit normalization independent of the
# selected CUDA-Q runtime target.

# %%
SAMPLE_SHOTS = 8 if docs_test_mode else 2_000
MAX_ITERATIONS = 4 if docs_test_mode else 100
cost_history = []


def cost_function(parameters) -> float:
    """Sample one QAOA coordinate and return its mean Ising energy."""

    run = compiled_qaoa.run(
        on=endpoint,
        gammas=tuple(float(value) for value in parameters[:LAYERS]),
        betas=tuple(float(value) for value in parameters[LAYERS:]),
        shots=SAMPLE_SHOTS,
        seed=20 + len(cost_history),
    )
    cost_history.append(run.samples.mean_energy)
    return run.samples.mean_energy


rng = np.random.default_rng(42)
initial_parameters = rng.uniform(-np.pi / 2.0, np.pi / 2.0, 2 * LAYERS)
optimization = minimize(
    cost_function,
    initial_parameters,
    method="COBYLA",
    options={"maxiter": MAX_ITERATIONS, "rhobeg": 0.5},
)

optimized_gammas = tuple(float(value) for value in optimization.x[:LAYERS])
optimized_betas = tuple(float(value) for value in optimization.x[LAYERS:])

print(f"Optimized mean energy: {optimization.fun:+.4f}")
print(f"Optimal gammas: {optimized_gammas}")
print(f"Optimal betas: {optimized_betas}")

assert len(optimization.x) == 2 * LAYERS
assert len(cost_history) == optimization.nfev

# %%
plt.figure(figsize=(8, 4))
plt.plot(cost_history, color="#2696EB")
plt.xlabel("Energy evaluation")
plt.ylabel("Cost (mean Ising energy)")
plt.title("QAOA optimization on CUDA-Q")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Final distribution and best cut
#
# The optimized artifact is sampled again with a larger budget. The decoded
# distribution retains exact occurrence counts and spin energies, so the
# lowest row maps directly to the best observed graph cut.

# %%
FINAL_SHOTS = 64 if docs_test_mode else 2_000
final_run = compiled_qaoa.run(
    on=endpoint,
    gammas=optimized_gammas,
    betas=optimized_betas,
    shots=FINAL_SHOTS,
    seed=113,
)
best_qaoa_sample = final_run.samples.lowest
best_spins = best_qaoa_sample.values
best_cut = sum(best_spins[left] != best_spins[right] for left, right in MAXCUT_EDGES)

energy_counts = Counter()
for sample in final_run.samples.samples:
    energy_counts[sample.energy] += sample.occurrences

print("best observed MaxCut sample:", best_qaoa_sample)
print(f"best observed cut: {best_cut}")

assert final_run.samples.total_occurrences == FINAL_SHOTS
assert best_qaoa_sample.energy == -float(best_cut)

# %%
energies = sorted(energy_counts)
plt.figure(figsize=(8, 4))
plt.bar(energies, [energy_counts[energy] for energy in energies], width=0.6)
plt.xlabel("Ising energy")
plt.ylabel("Frequency")
plt.title("Optimized CUDA-Q QAOA distribution")
plt.show()

node_colors = [
    "#2696EB" if best_spins[node] == 1 else "#EA9B26" for node in graph.nodes()
]
edge_colors = [
    "r" if best_spins[left] != best_spins[right] else "b"
    for left, right in MAXCUT_EDGES
]
plt.figure(figsize=(5, 4))
nx.draw(
    graph,
    positions,
    with_labels=True,
    node_color=node_colors,
    edge_color=edge_colors,
    node_size=700,
    edgecolors="black",
)
plt.title(f"Best QAOA partition (cut = {best_cut})")
plt.show()

# %% [markdown]
# ## Expectation value
#
# `qm.expectation` builds the required Pauli measurement programs from the same
# state-returning standard-library QAOA composition. The result is the sampled
# expectation and its statistical uncertainty through the same CUDA-Q endpoint.

# %%
expectation = qm.expectation(
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
    f"CUDA-Q expectation: {expectation.value:+.6f} ± {expectation.standard_error:.6f}"
)
assert expectation.backend == endpoint.name
assert abs(expectation.value - final_run.samples.mean_energy) <= (
    6 * expectation.standard_error + 0.5
)

# %% [markdown]
# ## Compare CUDA-Q runtime targets
#
# Runtime selection belongs to the endpoint, not the source or artifact. We
# time 100,000 shots on `qpp-cpu`; when an NVIDIA target is available, the same
# artifact and parameters are also run there and the finite-shot means are
# compared.

# %%
BENCHMARK_SHOTS = 1 if docs_test_mode else 100_000


def timed_target_run(runtime_target: str):
    """Run the optimized artifact on one CUDA-Q runtime target and time it."""

    runtime_endpoint = CudaqLocal(runtime_target=runtime_target)
    start = time.perf_counter()
    run = compiled_qaoa.run(
        on=runtime_endpoint,
        gammas=optimized_gammas,
        betas=optimized_betas,
        shots=BENCHMARK_SHOTS,
        seed=13,
    )
    return run, time.perf_counter() - start


cpu_run, cpu_seconds = timed_target_run("qpp-cpu")
print(f"qpp-cpu mean energy: {cpu_run.samples.mean_energy:+.4f}")
print(f"qpp-cpu sample time: {cpu_seconds:.4f} s")

target_runs = [("qpp-cpu", cpu_run.samples, cpu_seconds, "#2696EB")]
if cudaq.num_available_gpus() > 0 and cudaq.has_target("nvidia"):
    gpu_run, gpu_seconds = timed_target_run("nvidia")
    target_runs.append(("nvidia", gpu_run.samples, gpu_seconds, "#FF8A3D"))
    difference = abs(cpu_run.samples.mean_energy - gpu_run.samples.mean_energy)
    print(f"nvidia mean energy: {gpu_run.samples.mean_energy:+.4f}")
    print(f"nvidia sample time: {gpu_seconds:.4f} s")
    print(f"mean-energy difference: {difference:.4f}")
    if not docs_test_mode:
        assert difference < 0.15
else:
    print("No NVIDIA GPU detected; skipping the nvidia target run.")

# %%
figure, axes = plt.subplots(
    1,
    len(target_runs),
    figsize=(5.5 * len(target_runs), 4),
    sharey=True,
)
if len(target_runs) == 1:
    axes = [axes]
for axis, (target_name, samples, elapsed, color) in zip(
    axes,
    target_runs,
    strict=True,
):
    counts = Counter()
    for sample in samples.samples:
        counts[sample.energy] += sample.occurrences
    values = sorted(counts)
    axis.bar(values, [counts[value] for value in values], width=0.6, color=color)
    axis.axvline(
        samples.mean_energy,
        color="#2B2B2B",
        linestyle="--",
        label=f"mean = {samples.mean_energy:+.3f}",
    )
    axis.set_title(f"{target_name} ({elapsed:.3f} s)")
    axis.set_xlabel("Ising energy")
    axis.legend()
axes[0].set_ylabel("Frequency")
figure.suptitle(f"QAOA samples by CUDA-Q target ({BENCHMARK_SHOTS} shots)")
figure.tight_layout()
plt.show()

# %% [markdown]
# ## Kernels with classical control flow
#
# QCore effects state that this program needs mid-circuit measurement and a
# dynamic branch. Capability planning checks that requirement and CUDA-Q
# emission generates the matching runtime `if`.


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


feedback_artifact = qm.compile(feedback_sample, to=target)
feedback_source = export_payload(feedback_artifact).decode("utf-8")
feedback_result = qm.run(feedback_artifact, on=endpoint, shots=32, seed=3).result

print(feedback_source)
assert isinstance(feedback_source, str) and "if m0:" in feedback_source
assert set(feedback_result.counts) == {(0, 0), (1, 1)}

# %% [markdown]
# A target that lacks the required effects fails capability negotiation before
# launch; a capable target receives the verified artifact without an additional
# user-selected execution mode.

# %% [markdown]
# ## Summary
#
# - QCore is the sole persistent meaning of the typed source program.
# - `Cudaq` changes only realization by emitting serialized CUDA-Q kernel data.
# - One compiled artifact binds a runtime parameter on the in-process
#   `qpp-cpu` endpoint and returns Qamomile source-order bits.
# - The original five-node, three-layer MaxCut workflow includes continuous
#   optimization, expectation estimation, final sampling, and visualization.
# - The same artifact is benchmarked for 100,000 shots on CPU and, when
#   available, NVIDIA GPU; target selection changes only the endpoint.
# - Dynamic classical control executes through the same verified artifact
#   contract after capability negotiation.
