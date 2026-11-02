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
# # QURI Partsサポート
#
# このpageでは、具体的なoptimization problemを通してQamomileのQURI Parts integrationを紹介します。
# algorithm tutorialと同じ小さなMaxCut instanceにQAOAを適用し、1つの型付きprogramをportableな
# logical circuitへcompileして、Qulacsでsamplingとexpectation-value evaluationを行います。
# QURI Partsのnoise modelとsampler integrationも扱います。
#
# QamomileはQURI Partsをrealization boundaryに保ちます。QCoreだけがpersistent program semanticsであり、
# adapterがSDK circuitとsimulator objectを構築するのはinvocationをlaunchするときだけです。

# %% [markdown]
# ## インストール
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
# ## runtime parameterを持つ量子状態を1つ書く
#
# `entangled_state`だけが状態準備を定義します。sampling関数はこれにmeasurementを合成し、期待値評価は状態を直接使います。どちらのworkflowでも`theta`は型付きruntime parameterのままです。


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
# ## 一度コンパイルして実行対象を保持する
#
# `qm.compile`は、QURI Parts circuit、SDKオブジェクトの所有者、コンパイラが導出したABIとリソース情報を保持する`CompiledProgram`を返します。通常の実行では保存・復元を行いません。以下の`serialize`と`deserialize`は、必要な場合だけ`qamomile.persistence`で保存する例です。復元時には新しい実行対象を生成するため、Pythonオブジェクトの等値ではなくコンパイル識別子を比較します。

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
# `load_logical_circuit`はadapterを明示的にinspectionするhelperです。adapter payloadを通常のdataとしてdecodeし、2つ目のprogramを作ったり、共通のexecution contractを迂回したりしません。runtime parameterの式とmeasurement先はlaunchまでこのportable表現に残ります。

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
# ## 分離されたQulacsでartifactを再利用する
#
# artifactを`qm.run`へ渡してもcompilerは再実行されません。endpointはinvocationを検証し、保持した回路のコピーに`theta`をbindし、Qulacsでsamplingします。native bit順序はadapter境界で1回だけ正規化されます。返される`SampleResult`は、ほかのQamomile endpointと同じ型付きresult contractに従います。

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
# ## MaxCut問題
#
# [MaxCutに対するQAOA tutorial](../algorithm/qaoa_maxcut.ipynb)と同じ小さな5-node、6-edge graphを使います。
# $\sum_{(i,j)\in E}(1-s_i s_j)/2$の最大化は、constantを除けば
# $H_C=\sum_{(i,j)\in E}s_i s_j$の最小化に対応するため、graphをimmutable Ising
# `BinaryModel`へ直接変換できます。
#
# ## QAOA ansatzの構築
#
# uniform superpositionを準備し、cost layerとmixer layerを3回交互に適用してmeasureします。
# `QAOAExperiment`は同じcompositionを提供し、`gammas`と`betas`だけをruntime parameterとして残します。
# 通常実行ではcontinuous COBYLA optimizationを維持し、documentation testではexecution budgetだけを減らします。

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
# portable logical circuitは2つのQAOA angle vectorをunboundのまま保持します。そのため、programやcircuitを再構築せずに、すべてのoptimizer coordinateでartifactを再利用できます。

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
# ## samplingと連続最適化
#
# 各objective evaluationは`compiled_qaoa`を再利用してQulacsでsamplingし、occurrence-weighted mean Ising energyを最小化します。通常実行はmainと同じ3層QAOAを使用し、CI pathだけを1層かつ少ないshot数にします。

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
# ## 最終分布と最良cut
#
# 最適化したartifactを、通常実行ではmainと同じ2,000 shotで再度samplingします。decodeしたspin値からpartitionとcut sizeを求めます。

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
# ## 同じ状態sourceからobservableを評価する
#
# `qm.expectation`は同じ標準ライブラリのQAOA state compositionを使用します。group化したPauli measurementの処理はRustが導出し、QURI PartsとQulacsはrealizationとexecution routeだけを担います。

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
# ## 高度なQURI Parts機能
#
# ### unboundとboundのexpectation circuit
#
# QURI Partsはparametric estimatorとnon-parametric estimatorの両方を公開しています。
# `load_circuit`はportable artifactからimmutableな
# `LinearMappedParametricQuantumCircuit`を投影し、artifact自体はportable dataの
# まま保ちます。runtime coordinateは最初にgateで使われた順にflattenされ、
# QURI Partsのinput parameterには名前が保持されます。
#
# unbound circuitはparametric estimatorで使え、bound copyを作らずoptimizerの
# coordinate間で再利用できます。`bind_parameters`を呼ぶと、non-parametric
# estimator用の対応するbound circuitが得られます。QURI Partsのestimator state
# objectが期待する通り、このstate-preparation projectionではterminal measurementを
# 省略します。

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
# どちらのSDK estimator routeも同じQAOA stateを評価し、数値精度の範囲で一致します。
# `qm.expectation`はbackend-neutralなrouteとして残り、applicationが意図的にSDK
# circuitを管理したり別のestimatorを注入したりするときは、QURI Parts estimatorを
# 直接利用できます。

# %% [markdown]
# ## 別のQURI Parts samplerを使う
#
# `QuriPartsSampler`はQURI Parts sampler callableを受け取り、そのlive SDK objectをportable artifactの外側に保ちます。そのため同じ`compiled_qaoa` artifactを再compileせず、simulator、noise model、provider samplerで実行できます。live samplerにはportableなseed contractがないため、このendpointでは`seed=0`を使い、再現性は注入したsamplerに委ねます。
#
# 元のtutorialでは、このextension pointをQulacsの`NoiseSimulator`で示していました。ここでも同じ比較を復元します。最適化済みparameterとartifactは固定したまま、samplerだけを交換します。

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
# どちらの実行も通常のQamomile sample setを返すため、energy distributionを直接比較できます。depolarizing noiseは状態を最大混合状態へ近づけます。このためerror probabilityを増やすと、このIsing energyは0へ近づく傾向があります。


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
# ## まとめ
#
# - QCoreが型付きsource programの唯一のpersistentな意味です。
# - `QuriParts`は保存されたSDK objectではなく、deterministicなportable dataをemitします。
# - 1つのartifactが、分離されたQulacs上で複数のruntime parameter値を受け取ります。
# - samplingと期待値評価は1つの状態準備定義と共通Qamomile result contractを再利用します。
# - 元と同じ5-node、3層の連続MaxCut最適化、最終分布、partition、期待値workflowを実行します。
# - `load_circuit`により、artifactへSDK objectを保存せず、unbound/bound QURI Parts circuitの直接inspectionと両方のnative estimator routeを復元します。
# - 同じartifactを注入したQURI Parts `NoiseSimulator` samplerでも実行し、元のnoise比較を行います。
