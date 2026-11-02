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
# # Qiskitサポート
#
# このpageでは、具体的なoptimization problemを通してQamomileのQiskit integrationを紹介します。
# Qiskit integrationはoptionalです。`qiskit` extraをinstallすると、`Qiskit` target、
# local `Aer`、live `QiskitBackend` endpointを利用できます。小さなMaxCut instanceに対する
# QAOA optimizationを例に、1つのQamomile qkernelをQuantumCircuitへcompileし、Qiskit上でsamplingと
# expectation-value evaluationを実行します。advanced Qiskit featureとIBM Quantum executionも、
# 同じartifact/result contractを通して確認します。
#
# QamomileはQiskitをrealization boundaryに保ちます。QCoreだけがpersistent program semanticsであり、
# Qiskit `QuantumCircuit`は一時的なinspection/provider objectです。

# %% [markdown]
# ## インストール
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
# ## runtime parameterを持つsource programを1つ書く
#
# 最初の量子ビットを回転し、2番目の量子ビットとentangleして、型付き2 bit sampleを返します。`theta`はコンパイル後もruntime parameterとして残るため、1つのコンパイル済み回路を複数回のinvocationに利用できます。


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
# ## 一度コンパイルして実行対象を保持する
#
# `qm.compile`は、Qiskit QuantumCircuit、SDKオブジェクトの所有者、コンパイラが導出したABIとリソース情報を保持する`CompiledProgram`を返します。通常の実行では保存・復元を行いません。以下の`serialize`と`deserialize`は、必要な場合だけ`qamomile.persistence`で保存する例です。復元時には新しい実行対象を生成するため、Pythonオブジェクトの等値ではなくコンパイル識別子を比較します。

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
# `load_circuit`はadapterを明示的にinspectionするhelperです。artifactを変更せず、別の実行経路にもせず、保持したQiskit circuitのコピーを返します。Qamomile parameterはinvocationまで未束縛のままです。回路のparameter projectionは`p0`のような小さな位置名を使い、sealされたparameter ABIがユーザー向けの`theta`名と対応関係を保持します。

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
# ## 分離されたAerでartifactを再利用する
#
# artifactを`qm.run`へ渡してもcompilerは再実行されません。Aer endpointはcanonicalな`theta`値をbindし、呼び出し元のプロセスで束縛済み回路を実行し、Qiskit native bit順序を1回だけ正規化して、すべてのQamomile endpointで共通の型付き`SampleResult`を返します。

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
# ## MaxCut問題
#
# Qiskit integrationの説明に集中するため、
# [MaxCutに対するQAOA tutorial](../algorithm/qaoa_maxcut.ipynb)と同じ小さな5-node graphを使います。
# $\sum_{(i,j)\in E}(1-s_i s_j)/2$の最大化は、constantを除けば、
# antiferromagnetic Ising Hamiltonian $H_C=\sum_{(i,j)\in E}s_i s_j$の最小化に対応します。
# unweighted MaxCutでは全edgeが同じcouplingを持ちlinear termはないため、その係数を
# `BinaryModel.from_ising`へ直接渡します。
#
# ## QAOA ansatzの構築
#
# recipeはMaxCut tutorialと同じです。uniform superpositionを準備し、cost layerとmixer layerを
# 3回交互に適用して、computational basisでmeasureします。`QAOAExperiment`はimmutable spin modelと
# topologyを1つの型付きprogramへ閉じ、`gammas`と`betas`だけをruntime parameterとして残します。
# documentation testではlayer数とexecution budgetだけを減らします。

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
# ### optimized stateの期待値
#
# 上のsampling objectiveはoccurrence-weightedなenergy推定です。同じoptimized coordinateをQamomileのobservable expectation経路にも入力できます。Qiskitをrealization境界だけに保ちながら、main tutorialの明示的なestimator workflowを復元します。


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
# ### optimized programをAer noise modelで再利用する
#
# live `QiskitBackend`はSDK-nativeなnoiseを設定した`AerSimulator`を包めます。同じQAOA programをそのbackend snapshot向けにcompileし、同じoptimized coordinateをbindして、sampled mean energyを比較します。noiseはQCore semanticsではなくendpoint設定のままです。

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
# `DecodedSamples.mean_energy`はcost Hamiltonianのsampled expectation valueです。COBYLAは`compiled_qaoa`を再利用して同じscalarを最小化し、coordinate間でQamomile programもコンパイル済み回路も再構築しません。circuitをloadした後はnative Qiskit primitiveやprovider estimatorも使え、stateを返すsourceを使う場合は`qm.expectation`がbackend-neutralな経路を提供します。

# %% [markdown]
# ## 高度なQiskit機能
#
# ### Classical control flowとruntime expression
#
# measurement-dependentなPython branchは、Qiskit emitterがnative `if_else` regionへ変換するまでQCoreに保持されます。privateなfeedback bitは宣言した2-bit resultへ露出しません。


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
# induction valueを使わないstructural `for` loopも、同様に1つのnative
# `for_loop`として保持できます。このrealization choiceはQiskitが所有し、QCore
# sourceには通常の検証済み`qcore.for`が残ります。


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
# ### Pauli evolutionのrealization
#
# QAOA cost layerはQCore内で1つのabstract commuting-Pauli evolutionのままです。Qiskitはruntime angleを含む各cost layer全体を、保持した回路内で1つのnative `PauliEvolutionGate`として保持します。Aerがgateの検証済みexact fallback scheduleを展開するのはexecution時だけです。

# helper定義やloopの内部に演算を保持するため、最上位の演算数は展開後のPauli evolutionゲート数とは一致しません。

# %%
native_qaoa_artifact = qaoa_experiment.compile(to=target).artifact
qaoa_circuit = load_circuit(native_qaoa_artifact)
print("QAOA Qiskit operations:", dict(qaoa_circuit.count_ops()))
assert qaoa_circuit.num_qubits == native_qaoa_artifact.result_abi.width
assert len(qaoa_circuit.parameters) == 2 * LAYERS

# %% [markdown]
# ### QFTのrealization
#
# named compositeを使うと、検証済みの標準ライブラリQFTの本体をラベル付きSDKゲートとして保持できます。通常のQamomile qkernelとして呼ぶ場合は、同じlittle-endianの`h`/`cp`/`swap`列を直接出力します。`qft`という名前だけでは数学的な意味を保証できないため、`QFTGate`への置換は行いません。算術演算やPauli evolutionのnative実装は、compilerが検証した意味論とtargetの明示的なcapabilityに基づいて選択します。

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
# ## 同じsourceをlive BackendV2 adapterで使う
#
# `QiskitBackend`がlive SDK objectを所有するのはendpoint境界だけです。`target`はroutingとfingerprintのためにbackend capabilityをsnapshot化しますが、object自体はQCoreにもserialized artifact dataにも入りません。実providerも同じadapterを使います。このlocal例はQiskitの軽量な`BasicSimulator`を使い、意図的に`seed=0`とします。live backendはportable seed contractを約束しないためです。

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
# ## IBM Quantum hardwareで実行する
#
# provider backendにも同じlive adapterを使います。credentialとqueueingはartifact外に残るため、costを伴う次の例はdocumentation buildでは実行しません。
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
# ## まとめ
#
# - QCoreが型付きsource programの唯一のpersistentな意味です。
# - `Qiskit`はQuantumCircuit内にSDKの演算を保持してrealizationだけを変更します。
# - 1つのcompiled artifactが`Aer`上で複数のruntime parameter値を受け取ります。
# - `QiskitBackend`はprovider object、routing、native jobをendpoint境界に保ち、共通Qamomile result ABIを維持します。
# - 元の5-node、3-layer continuous MaxCut optimization、native dynamic `for_loop` / `if_else`、native `PauliEvolutionGate`、boxedとdecomposed QFTのinspection、live BackendV2 routeも実行します。documentation testではlayer、shot、optimizer iterationだけを縮小します。
