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
# # OMMXベンチマークの活用 (1): Qamomileによる量子アルゴリズムの実装とベンチマーク
#
# 本チュートリアルでは、Qamomileで構築した量子アルゴリズムを公開ベンチマークデータセットから取得した問題で動かし、その解の品質を古典ソルバーと同一のパイプライン上で比較する方法を示します。
#
# **目標。** Qamomileで自前のQAOAソルバーを構築し、[OMMX Quantum Benchmarks](https://github.com/Jij-Inc/OmmxQuantumBenchmarks)データセットからロードした**Low Autocorrelation Binary Sequences (LABS)** インスタンスに対して実行し、[`ommx-pyscipopt-adapter`](https://github.com/Jij-Inc/ommx-pyscipopt-adapter)経由で利用できる古典ソルバーSCIPと結果をベンチマークします。QAOAパスとSCIPパスはどちらも*同一*の`ommx.v1.Instance`を入力として受け取るため、両者の差は実質的にアルゴリズムそのものだけになり、フェアな比較が可能です。

# %%
# 本チュートリアルで追加で必要なパッケージのインストール
# # !pip install qamomile ommx-quantum-benchmarks ommx-pyscipopt-adapter

# %% [markdown]
# ## OMMX Quantum Benchmarksとは?
#
# **OMMX** ([Open Mathematical prograMming eXchange](https://jij-inc.github.io/ommx/en/introduction.html))は、数理最適化問題をツール間で受け渡すためのデータフォーマットです。`ommx.v1.Instance`には目的関数、制約条件、決定変数のメタデータ、そして必要に応じて参照解が格納されます。
#
# **OMMX Quantum Benchmarks**は、この`ommx.v1.Instance`形式で配布される最適化ベンチマークインスタンス集です。最初に提供されているデータセットは**QOBLIB** (Quantum Optimization Benchmarking Library) [arXiv:2504.03832](https://arxiv.org/abs/2504.03832)で、近年の量子最適化研究で頻繁に取り上げられる9つの問題ファミリーを収録しています。具体的にはLABS、Market Split、Independent Set、Steiner Tree Packingなどが含まれます。
#
# 各インスタンスはあくまで`ommx.v1.Instance`であるため、`ommx.v1.Instance`を受け取るQamomileのエントリーポイント、特に`QAOAConverter`は追加のアダプタコードなしにこれらのベンチマーク問題を扱えます。同じ`Instance`は`ommx-pyscipopt-adapter`のような古典側のOMMXアダプタにも渡せるので、ひとつの問題定義を量子・古典の両方のワークフローで使い回せます。

# %% [markdown]
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
# であり、これを*最小化*します(等価的に、**メリットファクター** $F = n^2 / (2 E)$を最大化することに相当します)。LABSはNP-hardであり、古典・量子の両方のヒューリスティクスのストレステストとして長く使われてきました。
#
# ### LABSインスタンスのロード
#
# `Labs`は2つのモデルを公開しています。一つは`"integer"`($c_k$を整数決定変数として導入し、それらを$\boldsymbol{s}$と結びつける制約を加える形式)、もう一つは`"quadratic_unconstrained"`(積$x_i x_{i+k+1}$を表す補助バイナリ変数$z_{i,k}$を2次のペナルティで導入したQUBO定式化)です。QAOAにはQUBO形式が自然に対応するため、本チュートリアルでは後者を使用します。

# %%
from ommx_quantum_benchmarks.qoblib import Labs

dataset = Labs()
print(f"Dataset:           {dataset.name}")
print(f"Available models:  {dataset.model_names}")
print(f"Instance count:    {len(dataset.available_instances['quadratic_unconstrained'])}")
print(f"First 5 instances: {dataset.available_instances['quadratic_unconstrained'][:5]}")

# %% [markdown]
# ここでは$n=5$のインスタンスである`labs005`を選びます。このQUBO定式化では合計$n + n(n-1) = 25$個のバイナリ変数(5個の系列ビット$x_i$と$n(n-1) = 20$個の補助$z_{i,k}$)を使います。`Instance.to_qubo()`がペナルティ項を目的関数に畳み込んだ上で未使用変数を取り除くと、最終的に15量子ビットに収まります。これはローカルでシミュレートできる程度に小さく、それでいてQAOAとして非自明なサイズです。

# %%
instance, reference_solution = dataset("quadratic_unconstrained", "labs005")
n = 5

print(f"OMMX variables:    {instance.num_variables}")
print(f"OMMX constraints:  {instance.num_constraints}")
print(f"Reference E(s):    {reference_solution.objective}")
print(f"Reference feasible: {reference_solution.feasible}")

# %% [markdown]
# 同梱の参照解から、$n=5$における既知の最適値が$E^\star = 2$(等価的にメリットファクター$F^\star = 25 / (2 \cdot 2) = 6.25$)であることが分かります。QAOAとSCIPの結果はこの値と比較していきます。

# %% [markdown]
# ## アルゴリズム: QAOA
#
# ここでは高レベルの`QAOAConverter`を使わず、[QAOA for MaxCut: Building the Circuit from Scratch](../algorithm/qaoa_maxcut)と同じレシピに従って`@qkernel`で一から回路を構築します。各ゲートの導出はそちらのチュートリアルに任せ、ここでは実装にフォーカスします。

# %% [markdown]
# ### OMMXインスタンスからスピンモデルを作る
#
# `Instance.to_qubo()`で`ommx.v1.Instance`からペナルティ形式のQUBOを取り出し、`BinaryModel`にラップしてからスピン(-1/+1)領域に変換します。これがQAOAのコストレイヤーが想定する形式です。また、コストランドスケープのスケールを揃えて実行ごとの再現性を保つため、係数を正規化します。

# %%
import ommx.v1

from qamomile.optimization.binary_model import BinaryModel, VarType

# `Instance.to_qubo()`はインスタンスをmutateする(ペナルティ法で制約を目的関数に畳み込む)。
# 呼び出し元のインスタンスを保つために、bytes経由でラウンドトリップさせる。
instance_for_qubo = ommx.v1.Instance.from_bytes(instance.to_bytes())
qubo, qubo_constant = instance_for_qubo.to_qubo()
spin_model = (
    BinaryModel.from_qubo(qubo, qubo_constant)
    .change_vartype(VarType.SPIN)
    .normalize_by_abs_max()
)
print(f"QAOA qubits: {spin_model.num_bits}")

# %% [markdown]
# ### QAOAのqkernel
#
# 一様重ね合わせ・コストレイヤー・ミキサーレイヤーの3つの小さなqkernelを定義し、フルansatzに組み合わせます。

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
# ### トランスパイルと最適化
#
# $p = 3$レイヤーでトランスパイルし、サンプリングされたビット列のスピンモデルでの平均エネルギーをCOBYLAで最小化します。軌跡を再現できるよう、AerSimulatorとNumPyの両方にシードを設定します。

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
    """`params`でQAOA回路をサンプリングし、スピンモデルでの平均エネルギーを返す。"""
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
# ### 最終サンプリング
#
# 最適化されたパラメータでショット数を増やして再度サンプリングし、元の`ommx.v1.Instance`に対してデコードします。返ってくる`ommx.v1.SampleSet`の目的値は、このインスタンスの元のQUBO目的値そのものです。このQUBO定式化では、補助変数$z$が積$x_i x_{i+k+1}$を正しく表現しているサンプルではペナルティが0となり、目的値が真のLABSエネルギー$E(\boldsymbol{s}) = \sum_k c_k^2$と一致します。一方、$z = x_i x_{i+k+1}$という暗黙の関係を破ったサンプルは、インスタンスに埋め込まれたプレースホルダー$P$に比例した加算ペナルティを払います。

# %%
def evaluate_with_ommx(
    sample_result, spin_model: BinaryModel, ommx_instance: ommx.v1.Instance
) -> ommx.v1.SampleSet:
    """SPINサンプルをデコードし、BINARYに変換してOMMXインスタンスで評価する。"""
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
# ### 目的値の分布
#
# QAOAは一つの解ではなくビット列の分布を返します。下のヒストグラムは、最適化済みパラメータで取得した各ショットのQUBO目的値です。赤の点線は参照最適値$E^\star$を示しており、この線上(または直右)に乗るサンプルが「$x$が実際にLABS和を最小化し、$z$が積を正しく符号化している」ショットです。線から右に離れたサンプルは、$z$が不整合なためペナルティを払っているサンプルになります。

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
# ## 古典ベースライン: OMMXアダプタ経由のSCIP
#
# 同じ`ommx.v1.Instance`を`ommx_pyscipopt_adapter.OMMXPySCIPOptAdapter.solve`に渡すと、PySCIPOpt経由でSCIP(MILP/QUBOソルバー)に問題が引き渡され、*元の*インスタンスに対して評価された`ommx.v1.Solution`が返ってきます。したがってその`.objective`はQAOA側と直接比較できます。

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
# ## 結果の比較
#
# SCIPは最適解を一意に返す決定論的なソルバーですが、QAOAはビット列の*分布*を返します。そこでQAOA側は**ベストショット**(全サンプル中で最も低い目的値を達成したビット列)と、参照最適値に対する**ヒット率**(その値を達成したショットの割合)を報告します。

# %%
# ヒット率: 参照最適値を達成したショットの割合
hit_rate = float((qaoa_summary["objective"].round().astype(int) == ref_E).mean())

print(f"{'Solver':<22} {'E(s)':>8} {'Time (s)':>12}")
print("-" * 46)
print(f"{'Reference (bundled)':<22} {ref_E:>8} {'-':>12}")
print(f"{'SCIP (classical)':<22} {scip_E:>8} {scip_solve_time:>12.3f}")
print(f"{'QAOA (best shot)':<22} {qaoa_best_E:>8} {qaoa_optimize_time:>12.2f}")
print()
print(f"QAOA hit rate on E* = {ref_E}: {hit_rate:.1%}  ({final_shots} shots)")

# %% [markdown]
# このベンチマークから読み取れることは大きく2点あります。
#
# 1. **最適値の到達。** SCIPとQAOAベストショットのどちらも参照最適値$E^\star = 2$に到達しており、QAOAは$n = 5$かつ$p = 3$レイヤーという軽い設定でも最適系列を*見つけられる*ことが分かります。
# 2. **集中度。** QAOAの価値はサンプリング確率を低エネルギーのビット列に集中させる点にあります。上記のヒット率(およびヒストグラム左端の集中)が、その性質を定量化したものです。
#
# 経過時間の列は両者の性質の*違い*を捉えるための値で、勝敗判定として読むものではありません。SCIPはQUBOに対してCPU上で直接動作するのに対し、QAOA側の計測値はステートベクタシミュレータ上での古典・量子最適化ループ全体を含みます。$n$が大きくなったときに、厳密だが指数的に伸びるbranch-and-boundと、ヒューリスティックだが多項式深さで済む回路という*定性的*なトレードオフが立ち現れる、そここそがOMMX Quantum Benchmarksのようなデータセットが両陣営の評価に役立つ理由です。

# %% [markdown]
# ## まとめ
#
# 本チュートリアルでは次のことを行いました。
#
# 1. **OMMX Quantum Benchmarks**データセットからLABSインスタンスをそのまま`ommx.v1.Instance`として読み込みました。
# 2. `Instance.to_qubo()`でQUBOを取り出して`BinaryModel`にラップし、スピン領域に切り替えた上で、自作のQAOA ansatz(`@qkernel`)を`QiskitTranspiler` + `AerSimulator`を通じて実行しました。
# 3. QAOAの出力(ベストショット、ヒット率、サンプリング分布)を、*同じ*インスタンスに対する`ommx_pyscipopt_adapter.OMMXPySCIPOptAdapter.solve`の結果、およびベンチマークに同梱された参照最適値と比較しました。
#
# 同じパイプラインは他のQOBLIBデータセット(`Marketsplit`、`IndependentSet`、`Network`など)にもそのまま適用できます。対応する`BaseDataset`サブクラスでロードし、`Instance.to_qubo()`でQUBOを取り出して、同じ`BinaryModel` + QAOA ansatz + transpileのループを再利用するだけです。インスタンスサイズがローカルシミュレータの能力を超えた場合は、同じ`executable`をQamomileの他のバックエンド(`QuriPartsTranspiler`、`CudaqTranspiler`など)や実機にそのまま切り替えられます。
#
# **次のステップ:**
#
# - QAOAそのものの数学については、[QAOA for MaxCut](../algorithm/qaoa_maxcut)と[QAOA for Graph Partitioning](../algorithm/qaoa_graph_partition)を参照してください。
# - 別のベンチマークファミリーに差し替えたい場合は、`ommx_quantum_benchmarks.qoblib`から`Labs().available_instances`、`IndependentSet().available_instances`などを覗いてみてください。
