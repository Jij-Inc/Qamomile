# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
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
# This page introduces Qamomile's [QURI Parts](https://quri-parts.qunasys.com/) backend through a concrete optimization problem. We model a small MaxCut instance as an Ising problem with `BinaryModel.from_ising`, write the QAOA ansatz directly as a `@qkernel`, and run it through `QuriPartsTranspiler` / `QuriPartsExecutor`. `QuriPartsExecutor` uses [Qulacs](https://docs.qulacs.org/), a fast C++ state-vector simulator, by default, so the examples below run on a local CPU without any extra configuration.

# %%
# Install the latest Qamomile with the QURI Parts extras through pip!
# # !pip install "qamomile[quri_parts]"

# %% [markdown]
# ## The MaxCut problem
#
# We reuse the small 5-node graph from the [QAOA for MaxCut tutorial](../algorithm/qaoa_maxcut.ipynb) so the focus stays on the QURI Parts integration rather than on QAOA itself. Maximizing $\sum_{(i,j) \in E}(1 - s_i s_j)/2$ is equivalent, up to a constant, to *minimizing* the antiferromagnetic Ising Hamiltonian $H_C = \sum_{(i,j) \in E} s_i s_j$. For unweighted MaxCut, every $J_{ij} = 1$ and every $h_i = 0$, so we pass these coefficients directly to `BinaryModel.from_ising`. The model object is used purely as a problem container — it gives us `quad`/`linear` dictionaries that we feed into the QAOA kernel below, and a `decode_from_sampleresult` helper that converts measurement counts into spin (+1 / -1) samples for energy evaluation.

# %%
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from qamomile.optimization.binary_model import BinaryModel

G = nx.Graph()
G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)])
num_nodes = G.number_of_nodes()

ising_quad: dict[tuple[int, int], float] = {
    tuple(sorted((i, j))): 1.0 for i, j in G.edges()
}
ising_linear: dict[int, float] = {}
spin_model = BinaryModel.from_ising(linear=ising_linear, quad=ising_quad)

pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(5, 4))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="white",
    node_size=700,
    edgecolors="black",
)
plt.title(f"MaxCut graph: {num_nodes} nodes, {G.number_of_edges()} edges")
plt.show()

# %% [markdown]
# ## Building the QAOA ansatz with `@qkernel`
#
# Rather than going through `QAOAConverter` (which transpiles internally and hides the kernel), we write the QAOA ansatz directly as a small `@qkernel`. This gives us a Qamomile-side view of the IR via `kernel.draw()` before we hand the program off to QURI Parts.
#
# The recipe is the same as in the [QAOA for MaxCut tutorial](../algorithm/qaoa_maxcut.ipynb): a uniform superposition, $p$ rounds of cost + mixer layers, then a terminal measurement. Qamomile's rotation gates follow the $e^{-i\theta/2}$ convention, so the $1/2$ factor is handled differently in the two layers. For the mixer we pass $2\beta$ to `rx`, which cancels the $1/2$ exactly and gives the textbook $e^{-i\beta X}$. For the cost layer we pass $J_{ij} \cdot \gamma$ to `rzz`, which does **not** cancel the $1/2$; instead, the missing factor of $2$ is absorbed into the variational parameter $\gamma$ itself, so the $\gamma$ used here is half of the textbook QAOA $\gamma$.

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
# `qaoa_ansatz.draw(...)` renders the IR at the Qamomile level. We fix the problem-structure arguments (`p`, `quad`, `linear`, `n`) to concrete values so the layered shape is visible, and let `gammas` / `betas` remain symbolic.

# %%
p = 3  # number of QAOA layers
qaoa_ansatz.draw(
    p=p,
    quad=spin_model.quad,
    linear=spin_model.linear,
    n=num_nodes,
)

# %% [markdown]
# ## Transpile to QURI Parts
#
# `QuriPartsTranspiler` is plugged into `transpile()` the same way as any other backend. We bind the problem-structure arguments and keep `gammas` / `betas` as runtime parameters:

# %%
from qamomile.quri_parts import QuriPartsExecutor, QuriPartsTranspiler

transpiler = QuriPartsTranspiler()
executor = QuriPartsExecutor()

executable = transpiler.transpile(
    qaoa_ansatz,
    bindings={
        "p": p,
        "quad": spin_model.quad,
        "linear": spin_model.linear,
        "n": num_nodes,
    },
    parameters=["gammas", "betas"],
)

# %% [markdown]
# `executable.get_first_circuit()` returns the underlying QURI Parts circuit. It really is a QURI Parts `LinearMappedParametricQuantumCircuit` (exposed in the public API as `LinearMappedUnboundParametricQuantumCircuit`), and the $2p$ QAOA angles (`gammas[0..p-1]`, `betas[0..p-1]`) remain available as named runtime parameters. We can confirm both with `type(...)` / the parameter count, and inspect the circuit directly with QURI Parts' built-in `draw_circuit`:

# %%
from quri_parts.circuit.utils.circuit_drawer import draw_circuit

quri_circuit = executable.get_first_circuit()
assert quri_circuit is not None  # transpile() always emits one quantum segment here
print(type(quri_circuit).__name__)
print("qubit_count    :", quri_circuit.qubit_count)
print("parameter_count:", quri_circuit.parameter_count)

draw_circuit(quri_circuit, line_length=200)

# %% [markdown]
# Each runtime parameter slot stays unbound until execution time, so binding `gammas` / `betas` is a cheap numeric update on the QURI Parts side rather than a circuit rebuild. The problem structure (Ising coefficients, qubit count, and depth) is fixed at compile time, leaving only the variational angles as runtime inputs.

# %% [markdown]
# ## Sampling QAOA with `QuriPartsExecutor`
#
# `executable.sample(executor, bindings=..., shots=...)` returns a `SampleJob`. Calling `.result()` gives a `SampleResult`, which `BinaryModel.decode_from_sampleresult` decodes into a `BinarySampleSet` already in the spin domain (+1 / -1). That means cut edges can be counted directly, with no additional conversion. `QuriPartsExecutor()` runs against the Qulacs state-vector simulator by default.

# %%
rng = np.random.default_rng(42)
init_params = rng.uniform(-np.pi / 2, np.pi / 2, 2 * p)
init_gammas = list(init_params[:p])
init_betas = list(init_params[p:])
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
sample_shots = 256 if docs_test_mode else 2000
maxiter = 20 if docs_test_mode else 100

sample_result = executable.sample(
    executor,
    bindings={"gammas": init_gammas, "betas": init_betas},
    shots=sample_shots,
).result()

decoded = spin_model.decode_from_sampleresult(sample_result)
print(f"Mean energy at random init: {decoded.energy_mean():+.4f}")

# %% [markdown]
# ## Optimizing the QAOA parameters
#
# A typical QAOA optimization loop reuses the same `executable` across many `(gammas, betas)` vectors: call `transpiler.transpile()` once, then call `executable.sample()` many times. Below, we wrap the sample-and-decode step in SciPy's `minimize` driver. The classical optimizer updates `(gammas, betas)` to minimize the mean sampled Ising energy, while every iteration reuses the same `executable` and `QuriPartsExecutor`.

# %%
from scipy.optimize import minimize

cost_history: list[float] = []


def cost_fn(params: np.ndarray) -> float:
    result = executable.sample(
        executor,
        bindings={"gammas": list(params[:p]), "betas": list(params[p:])},
        shots=sample_shots,
    ).result()
    energy = spin_model.decode_from_sampleresult(result).energy_mean()
    cost_history.append(energy)
    return energy


res = minimize(cost_fn, init_params, method="COBYLA", options={"maxiter": maxiter})

opt_gammas = list(res.x[:p])
opt_betas = list(res.x[p:])
print(f"Optimized mean energy: {res.fun:+.4f}")
print(f"Optimal gammas       : {[round(float(v), 4) for v in opt_gammas]}")
print(f"Optimal betas        : {[round(float(v), 4) for v in opt_betas]}")

# %%
plt.figure(figsize=(8, 4))
plt.plot(cost_history, color="#2696EB")
plt.xlabel("Iteration")
plt.ylabel("Cost (mean Ising energy)")
plt.title("QAOA optimization on Qulacs via `QuriPartsExecutor`")
plt.tight_layout()
plt.show()

# %% [markdown]
# The Qulacs sampler used inside `QuriPartsExecutor` does not seed its per-shot RNG, so the exact trajectory and final energy will vary slightly across runs. The optimizer should still converge close to the ground-state energy of $H_C$ on this 5-node graph. The optimized parameters above (`opt_gammas`, `opt_betas`) are reused throughout the rest of this page.

# %% [markdown]
# ## Expectation values: unbound-parametric vs. bound-circuit dispatch
#
# `QuriPartsExecutor.estimate_expectation(circuit, hamiltonian, param_values)` is the QURI Parts-native entry point for expectation-value evaluation. It automatically dispatches between two different QURI Parts estimators depending on the **state of the circuit it receives**:
#
# - **Unbound parametric circuit**: the linear-mapped circuit produced by `transpile()` still has free parameters. QURI Parts' `apply_circuit` wraps it as a `ParametricCircuitQuantumState`, and the executor routes the call through QURI Parts' **parametric estimator**, using `param_values` to bind the parameters at evaluation time.
# - **Bound or non-parametric circuit**: once parameters are bound (for example, via `circuit.bind_parameters([...])`), the same `apply_circuit` call produces a `GeneralCircuitQuantumState`. The executor falls back to QURI Parts' **non-parametric estimator**, and `param_values` is ignored.
#
# Knowing which path is used matters because the two estimators have different cost profiles. The parametric estimator is the right tool inside a tight optimization loop because it avoids a per-iteration circuit copy. The non-parametric estimator avoids parametric bookkeeping when the parameters are already concrete.
#
# QURI Parts treats `measure` as a no-op at the circuit level, so the parametric circuit emitted by `transpiler.transpile(qaoa_ansatz, ...)` represents exactly the QAOA state preparation $|\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle$. We can pass it directly to `estimate_expectation` together with the cost Hamiltonian to compute $\langle H_C \rangle$ analytically, with no sampling noise. Inside a QAOA optimizer, you can therefore keep the same circuit and replace `executable.sample()` + decode with `executor.estimate(circuit, hamiltonian, params=...)`.
#
# To exercise both estimator paths directly, build a Qamomile `Hamiltonian` for $H_C = \sum_{(i,j) \in E} Z_i Z_j$ (plus the linear $h_i$ terms, which are empty for unweighted MaxCut), convert it to a QURI Parts operator, and call `estimate_expectation` on each circuit variant:

# %%
import qamomile.observable as qm_o
from qamomile.quri_parts.observable import hamiltonian_to_quri_operator

cost_hamiltonian = qm_o.Hamiltonian()
for (i, j), Jij in spin_model.quad.items():
    cost_hamiltonian.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.Z, i), qm_o.PauliOperator(qm_o.Pauli.Z, j)),
        Jij,
    )
for i, hi in spin_model.linear.items():
    cost_hamiltonian.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, i),), hi)

quri_H = hamiltonian_to_quri_operator(cost_hamiltonian)

# The unbound parametric circuit straight out of transpiler.transpile().
unbound_circuit = executable.get_first_circuit()
assert unbound_circuit is not None
print(f"unbound type           : {type(unbound_circuit).__name__}")
print(f"unbound parameter_count: {unbound_circuit.parameter_count}")

# QURI Parts requires the runtime parameters as a flat list in the order
# they were registered on the circuit, which is first-use order during
# emission. For QAOA that means gammas[0], betas[0], gammas[1], betas[1],
# ..., interleaved per layer, NOT all gammas followed by all betas.
# Read the order from the executable and build the flat list by lookup so
# we never have to guess the convention.
named_values = {f"gammas[{i}]": opt_gammas[i] for i in range(p)}
named_values.update({f"betas[{i}]": opt_betas[i] for i in range(p)})
flat_params = [named_values[name] for name in executable.parameter_names]
print(f"circuit parameter order: {executable.parameter_names}")

# Manually bind the same numeric values using QURI Parts' native binder.
bound_circuit = unbound_circuit.bind_parameters(flat_params)
print(f"bound   type           : {type(bound_circuit).__name__}")

# Path 1: unbound -> parametric estimator. param_values is consumed.
energy_unbound = executor.estimate_expectation(
    unbound_circuit, quri_H, flat_params
)

# Path 2: bound -> non-parametric estimator. param_values is ignored.
energy_bound = executor.estimate_expectation(bound_circuit, quri_H, [])

print(f"parametric  estimator: {energy_unbound:+.10f}")
print(f"non-param.  estimator: {energy_bound:+.10f}")
assert np.isclose(energy_unbound, energy_bound, atol=1e-10)

# %% [markdown]
# Both paths agree to numerical precision because they evaluate the same QAOA state against the same Ising cost Hamiltonian. The resulting noise-free expectation value at the optimized parameters should also match the optimized sample-mean energy printed earlier within shot noise. The dispatch is invisible to callers who stay inside Qamomile's `executor.estimate()` interface. Reach for `estimate_expectation` directly only when you need to manage the QURI Parts circuit yourself.
#
# `executor.estimate(circuit, hamiltonian, params=...)` is the higher-level variant. It accepts a `qamomile.observable.Hamiltonian` directly, converts it automatically, and then delegates to `estimate_expectation`:

# %%
energy_via_estimate = executor.estimate(
    unbound_circuit, cost_hamiltonian, params=flat_params
)
print(f"executor.estimate     : {energy_via_estimate:+.10f}")
assert np.isclose(energy_via_estimate, energy_unbound, atol=1e-10)

# %% [markdown]
# ## Customizing the sampler and estimator
#
# `QuriPartsExecutor()` lazily creates the default Qulacs vector sampler and parametric estimator on first use. To plug in a different QURI Parts backend, pass the sampler (and/or estimator) through `QuriPartsTranspiler.executor(sampler=..., estimator=...)` or instantiate `QuriPartsExecutor(sampler=..., estimator=...)` directly. The custom executor is a drop-in replacement everywhere `executor` appeared above, and swapping the sampler does not require re-transpiling the kernel: the executable carries the circuit, while the executor carries the simulation backend.
#
# As a concrete check, we build a noise-aware sampler with QURI Parts' Qulacs `NoiseSimulator` and compare the noiseless and noisy sample-mean energies at the **same** optimized parameters. With a non-trivial depolarizing channel applied to every gate, the noisy mean energy should drift away from the noise-free value — confirming that the custom sampler is actually in the loop.

# %%
from quri_parts.circuit.noise import DepolarizingNoise, NoiseModel
from quri_parts.qulacs.sampler import create_qulacs_noisesimulator_sampler

noise_model = NoiseModel([DepolarizingNoise(error_prob=0.02)])
noisy_sampler = create_qulacs_noisesimulator_sampler(noise_model)
noisy_executor = transpiler.executor(sampler=noisy_sampler)

clean_result = executable.sample(
    executor,
    bindings={"gammas": opt_gammas, "betas": opt_betas},
    shots=sample_shots,
).result()
noisy_result = executable.sample(
    noisy_executor,
    bindings={"gammas": opt_gammas, "betas": opt_betas},
    shots=sample_shots,
).result()

clean_energy = spin_model.decode_from_sampleresult(clean_result).energy_mean()
noisy_energy = spin_model.decode_from_sampleresult(noisy_result).energy_mean()
print(f"noiseless sampler mean energy: {clean_energy:+.4f}")
print(f"noisy     sampler mean energy: {noisy_energy:+.4f}")

# %% [markdown]
# The depolarizing channel mixes the QAOA state toward the maximally mixed state, where every spin configuration is equally likely and the mean of $H_C$ collapses to $0$. The noisy mean energy is therefore higher (closer to zero) than the noiseless one. Increasing `error_prob` or stacking additional noise channels pushes the noisy energy further toward $0$; setting `error_prob=0.0` recovers the noiseless value within shot noise. The same pattern applies when swapping in any other QURI Parts sampler — a remote device, a density-matrix simulator, a stochastic state-vector sampler — without re-transpiling the kernel.

# %% [markdown]
# ## Summary
#
# - Writing the QAOA ansatz directly as a `@qkernel` keeps the IR inspectable via `kernel.draw()`. `BinaryModel.from_ising` is then used purely as a problem container that supplies `quad`/`linear` to the kernel and decodes sample results back into the spin domain.
# - `QuriPartsTranspiler().transpile(kernel, bindings=..., parameters=[...])` lowers the kernel to a QURI Parts `LinearMappedParametricQuantumCircuit`, which can be inspected with QURI Parts' native `draw_circuit`.
# - `QuriPartsExecutor` supports both `executable.sample()` for QAOA-style sampling and `executor.estimate(...)` for noise-free expectation values, against the Qulacs state-vector simulator by default.
# - `estimate_expectation` dispatches between QURI Parts' parametric and non-parametric estimators depending on whether the input circuit still has free parameters; `executor.estimate()` hides the dispatch, but you can also invoke it directly when integrating outside Qamomile's higher-level interface.
# - Custom samplers and estimators — including QURI Parts' `NoiseSimulator`-backed sampler — can be passed through `transpiler.executor(...)` without re-transpiling the kernel, and they actually do change the sampled distribution as shown above.
