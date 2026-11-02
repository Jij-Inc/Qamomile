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
# # OMMX Quantum Benchmarksの活用: Qamomileによる量子アルゴリズムの実装とベンチマーク
#
# 本チュートリアルでは、公開ベンチマークデータセットの問題にQamomileの量子アルゴリズムを適用し、その解の品質を同じワークフロー上で古典ソルバーと比較する方法を示します。
#
# **目標:** QamomileでQAOAソルバーを構築し、OMMX Quantum BenchmarksデータセットからロードしたLow Autocorrelation Binary Sequences (LABS)インスタンスに対して実行し、`ommx-pyscipopt-adapter`経由で利用できる古典ソルバーSCIPと結果を比較します。QAOA側とSCIP側はどちらも*同じ*`ommx.v1.Instance`を入力とするため、主な違いはアルゴリズムそのものです。そのため直接比較できます。

# %% [markdown]
# ## インストール
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
# ## OMMX Quantum Benchmarksとは?
#
# **OMMX**(Open Mathematical prograMming eXchange)は、数理最適化問題をツール間で受け渡すためのデータフォーマットです。`ommx.v1.Instance`には目的関数、制約条件、決定変数のメタデータが格納されます。
#
# **OMMX Quantum Benchmarks**は、この`ommx.v1.Instance`形式で配布される最適化ベンチマークインスタンス集です。最初に提供されているデータセットは**QOBLIB**(Quantum Optimization Benchmarking Library)arXiv:2504.03832で、近年の量子最適化研究で使われている9つの問題ファミリーを収録しています。LABS、Market Split、Independent Set、Steiner Tree Packingなどが含まれます。
#
# 各ベンチマークインスタンスは`ommx.v1.Instance`として表現されるため、`QAOAConverter`を含む`ommx.v1.Instance`対応のQamomileワークフローは、追加の実装なしにこれらの問題を扱えます。さらに、参照解も`ommx.v1.Solution`形式で含まれており、ベンチマークの評価に使用できます。同じ`Instance`は`ommx-pyscipopt-adapter`のような古典側のOMMXアダプタにも渡せるので、1つの問題定義を量子・古典の両方のワークフローで使えます。
#
# ## 問題: Low Autocorrelation Binary Sequences (LABS)
#
# **LABS**は、$\boldsymbol{s} = (s_0, s_1, \dots, s_{n-1}) \in \{-1, +1\}^n$というバイナリ系列のうち、非対角の自己相関
#
# $$
# c_k(\boldsymbol{s}) = \sum_{i=0}^{n-k-1} s_i \, s_{i+k},
# \qquad k = 1, 2, \dots, n-1
# $$
#
# をできるだけ0に近づけるものを求める問題です。ベンチマークの目的関数は**自己相関の二乗和**
#
# $$
# E(\boldsymbol{s}) = \sum_{k=1}^{n-1} c_k(\boldsymbol{s})^2,
# $$
#
# であり、これを*最小化*します。LABSはNP-hardであり、古典・量子の両方のヒューリスティクスのストレステストとして長く使われてきました。
#
# ### LABSインスタンスのロード
#
# `Labs`は2つのモデルを公開しています。一つは`"integer"`($c_k$を整数決定変数として導入し、それらを$\boldsymbol{s}$と結びつける制約を加える形式)、もう一つは`"quadratic_unconstrained"`(積$x_i x_{i+k+1}$を表す補助バイナリ変数$z_{i,k}$を2次のペナルティで導入したQUBO定式化)です。QAOAにはQUBO形式が自然に対応するため、本チュートリアルでは後者を使用します。
#
# ここでは$n=5$のインスタンスである`labs005`を選びます。このQUBO定式化では合計$n + n(n-1) = 25$個のバイナリ変数(5個の系列ビット$x_i$と$n(n-1) = 20$個の補助$z_{i,k}$)を使います。`Instance.to_qubo()`がペナルティ項を目的関数に畳み込み、未使用変数を取り除くと、15量子ビットの問題になります。ローカルでシミュレートできる程度に小さく、かつQAOAとして非自明なサイズです。
#
# 同梱の参照解から、$n=5$における既知の最適値が$E^\star = 2$であることが分かります。QAOAとSCIPの結果はこの値と比較していきます。
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
# ## ImmutableなActive QUBOを抽出する
#
# `to_qubo`はreceiverを変換できるため、最初にbenchmark bytesをround-tripします。
# `BinaryModel.from_hubo`は得られた係数をRustでcanonicalizeし、非zero termに現れる
# 15個のvariable IDだけを推論します。caller-owned benchmarkは変更されません。

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
# ## 3層QAOAを構築してコンパイルする
#
# `QAOAExperiment`は有効な二値Hamiltonianと3層の構造を固定し、最終サンプリング用の古典出力を持つ測定プログラムを作ります。公開の部品`qaoa_state`で、エネルギー推定器にも同じ状態準備を渡します。どちらも宣言された`gammas`と`betas`のベクトルを使うため、SDKのパラメータ名に手動で対応付ける必要はありません。

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
# ## 有限ショットのエネルギー推定値を最適化する
#
# `Estimator`で有効なQUBO Hamiltonianを測定します。すべての項がZの積なので1つのQWC測定グループを共有でき、各ショットのスコア計算で共分散を保持します。`ENERGY_SHOTS`は目的関数の1回の評価に使う総予算です。
#
# COBYLAは有限差分の勾配を必要としません。目的関数にはノイズがあり、最後の値は解の品質を独立に推定した結果ではありません。選んだパラメータは後から新しい標本で評価します。サンプリングの不確かさは最適化誤差の上限ではなく、ベンチマークの最適解への収束を保証しません。

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
# ## 最終SamplingとOMMX Evaluation
#
# コンパイル済みの実験を再利用し、`compiled.run`からQamomileのAer実行先で最適化した状態をサンプリングします。別の新しい標本でエネルギーと不確かさも推定します。
# Qamomileが15個のactive変数を元のIDでdecodeします。multiplicityをOMMX sampleへ
# 展開し、変更していないbenchmark instanceで評価するため、objectiveはpublished
# LABSの意味を持ちます。

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
# QAOAは1つのanswerではなくdistributionを返します。histogramに元のQUBO objectiveを
# すべて表示し、benchmark optimumを赤線で示します。

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
# ## 古典Baseline：SCIP
#
# OMMX PySCIPOpt adapterは同じimmutable instanceをSCIPで解きます。このobjectiveは
# bundled referenceおよびQAOA best shotと直接比較できます。

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
# ## 結果の比較
#
# SCIPは1つのoptimumを返し、QAOAはdistributionを返します。そのためQAOAについては
# best shotとpublished optimumに到達したshotの割合を報告します。wall timeは異なる
# algorithmic workを含むため参考値です。

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
# ## まとめ
#
# - `labs005`から1つのimmutable benchmarkと既知のoptimumを得ます。
# - 15個のactive QUBO変数を3層typed QAOA experimentへ渡します。
# - COBYLAで有限ショットのエネルギー推定値を最適化し、測定プログラムから4,096-shot
#   output distributionを得ます。
# - OMMXがpublished objectiveの意味を復元し、SCIPは同じinstanceを独立に解きます。
# - best shot、optimum hit rate、distribution、wall timeを含む、元と同じ量子対古典
#   benchmark UXを再現します。
