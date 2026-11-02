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
# # CUDA-Qサポート
#
# このpageでは、具体的なoptimization problemを通してQamomileのCUDA-Q integrationを紹介します。
# algorithm tutorialと同じ小さなMaxCut instanceにQAOAを適用し、1つの型付きprogramを
# serialized CUDA-Q kernel dataへcompileして、CUDA-Q target上でsamplingとexpectation-value
# evaluationを行います。CPUとoptional NVIDIA targetの比較、classical control flowを持つkernelも扱います。
#
# QamomileはCUDA-Qをrealization boundaryに保ちます。QCoreだけがpersistent program semanticsであり、
# CUDA-Q kernelはinspectionまたは実行で使う一時的なSDK objectです。

# %% [markdown]
# ## インストール
#
# インストール済みのCUDA toolkitに合うオプション依存グループを選んでください。
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
# ## runtime parameterを持つ量子状態を1つ書く
#
# `parameterized_state`だけが状態準備を定義します。2つの基底endpointは意図的に非対称で、`theta = 0`では`01`、`theta = pi`では`10`を生成します。これによりnative bit順序の反転を検出できます。


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
# ## 一度コンパイルして実行対象を保持する
#
# `qm.compile`は、CUDA-Q kernel、SDKオブジェクトの所有者、コンパイラが導出したABIとリソース情報を保持する`CompiledProgram`を返します。通常の実行では保存・復元を行いません。以下の`serialize`と`deserialize`は、必要な場合だけ`qamomile.persistence`で保存する例です。復元時には新しい実行対象を生成するため、Pythonオブジェクトの等値ではなくコンパイル識別子を比較します。

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
# 生成したPythonソースは明示的なexportで取得できます。`load_kernel`は保持したkernelを参照します。同じCUDA-Q targetへの実行ではkernelとhelperの所有者を再利用します。

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
# ## CPU simulatorでartifactを実行する
#
# artifactを`qm.run`へ渡してもcompilerは再実行されません。endpointはinvocationを検証し、呼び出し元のプロセスでkernelを復元してCUDA-Qの`qpp-cpu` targetを選び、`theta`をbindします。QamomileのCUDA-Q実行を直列化し、実行後には元のSDK targetを復元します。native bit順序は1回だけ正規化されます。決定的で非対称な1回のlaunchにより、runtime parameter経路とsource-order result contractの両方を確認します。

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
# 通常実行は候補ごとに2,000 shots、COBYLA 100候補です。documentation testではbudgetだけを減らします。

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
# serialized CUDA-Q artifactには2つのruntime vectorだけが残ります。生成sourceから
# layered Hadamard、Pauli-evolution realization、mixer、final measurementを確認できます。

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
# ## Samplingと連続最適化
#
# objectiveごとに1つのcompiled artifactを再利用し、`qpp-cpu`でsampleしたdecoded mean
# Ising energyを最小化します。aggregate vector ABIとsource-order bit正規化はCUDA-Q
# runtime targetから独立しています。

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
# ## 最終分布とBest Cut
#
# 最適化済みartifactをより大きいbudgetでsampleし、最低energy rowをgraph cutへ戻します。

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
# ## Expectation Value
#
# `qm.expectation`は同じstate-returning standard-library QAOA compositionから必要な
# Pauli measurement programを作り、同じCUDA-Q endpointでexpectationと統計誤差を返します。

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
# ## CUDA-Q Runtime Targetを比較する
#
# runtime選択はsourceやartifactではなくendpointが担います。`qpp-cpu`で100,000 shotsを
# 計測し、NVIDIA targetが利用できる場合は同じartifactとparameterをそこで実行して
# finite-shot meanを比較します。

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
# ## classical control flowを含むkernel
#
# QCore effectが、このprogramにmid-circuit measurementとdynamic branchが必要なことを表します。capability planningがそのrequirementを検査し、CUDA-Q emissionが対応するruntime `if`を生成します。


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
# 必要なeffectを持たないtargetはlaunch前のcapability negotiationで失敗し、capable targetには追加のuser-selected execution modeなしでverified artifactが渡ります。

# %% [markdown]
# ## まとめ
#
# - QCoreが型付きsource programの唯一のpersistentな意味です。
# - `Cudaq`が変更するのはrealizationだけであり、serialized CUDA-Q kernel dataをemitします。
# - 1つのcompiled artifactが同一プロセスの`qpp-cpu` endpointでruntime parameterをbindし、Qamomile source-orderのbitを返します。
# - 元と同じ5-node、3-layer MaxCut workflowで連続最適化、expectation、final sampling、可視化を行います。
# - 同じartifactをCPUと、利用可能ならNVIDIA GPUで100,000 shots実行し、target選択がendpointだけを変えることを確認します。
# - dynamic classical controlはcapability negotiation後、同じverified artifact contractで実行します。
