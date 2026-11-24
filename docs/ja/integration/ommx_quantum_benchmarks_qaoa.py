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
# # OMMX Quantum Benchmarksの活用: Qamomileによる量子アルゴリズムの実装とベンチマーク
#
# 本チュートリアルでは、公開ベンチマークデータセットの問題にQamomileの量子アルゴリズムを適用し、その解の品質を同じワークフロー上で古典ソルバーと比較する方法を示します。
#
# **目標:** QamomileでQAOAソルバーを構築し、[OMMX Quantum Benchmarks](https://jij-inc.github.io/OmmxQuantumBenchmarks/ja/)データセットからロードしたLow Autocorrelation Binary Sequences (LABS)インスタンスに対して実行し、[`ommx-pyscipopt-adapter`](https://ommx-ja-book.readthedocs.io/ja/latest/user_guide/supported_ommx_adapters.html)経由で利用できる古典ソルバーSCIPと結果を比較します。QAOA側とSCIP側はどちらも*同じ*`ommx.v1.Instance`を入力とするため、主な違いはアルゴリズムそのものです。そのため直接比較できます。

# %%
# 本チュートリアルで追加で必要なパッケージのインストール
# # !pip install "qamomile[qiskit]" ommx-quantum-benchmarks ommx-pyscipopt-adapter matplotlib scipy

# %%
# ruff: noqa: F821 -- dimensions are Qamomile dependent shape binders.
from __future__ import annotations

import os
import time

# `ommx-quantum-benchmarks`は内部で`minto`を使用しており、`minto`は
# 既定で実験開始時にホスト環境の情報(Pythonバージョン、仮想環境のパスなど)を
# 標準出力に書き出します。レンダリングされたノートブック出力にローカル環境の
# 情報が漏れないよう、ここで抑制しておきます。
os.environ["MINTO_TESTING"] = "true"

import matplotlib.pyplot as plt
import numpy as np
import ommx.v1
import ommx_pyscipopt_adapter
from ommx_quantum_benchmarks.qoblib import Labs
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from scipy.optimize import minimize

import qamomile as qm
from qamomile.optimization import BinaryModel

# %% [markdown]
# ## OMMX Quantum Benchmarksとは?
#
# **OMMX**([Open Mathematical prograMming eXchange](https://jij-inc.github.io/ommx/en/introduction.html))は、数理最適化問題をツール間で受け渡すためのデータフォーマットです。`ommx.v1.Instance`には目的関数、制約条件、決定変数のメタデータが格納されます。
#
# **OMMX Quantum Benchmarks**は、この`ommx.v1.Instance`形式で配布される最適化ベンチマークインスタンス集です。最初に提供されたデータセットは**QOBLIB**（Quantum Optimization Benchmarking Library）です。QOBLIBの原論文は、LABS、Market Split、Independent Set、Steiner Tree Packingなど10種類の問題を定義しています{cite:p}`10.48550/arXiv.2504.03832`。
#
# 各インスタンスは`ommx.v1.Instance`形式です。ここで使うLABSインスタンスには、`ommx.v1.Solution`形式の参照解も含まれます。同じ`Instance`は`ommx-pyscipopt-adapter`のような古典側のOMMXアダプタにも渡せるので、1つの問題定義を量子・古典の両方のワークフローで使えます。


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
# であり、これを*最小化*します。LABSは古典・量子の両方のヒューリスティクスのストレステストとして長く使われてきました。
#
# ### LABSインスタンスのロード
#
# `Labs`は2つのモデルを公開しています。一つは`"integer"`($c_k$を整数決定変数として導入し、それらを$\boldsymbol{s}$と結びつける制約を加える形式)、もう一つは`"quadratic_unconstrained"`(積$x_i x_{i+k+1}$を表す補助バイナリ変数$z_{i,k}$を2次のペナルティで導入した制約なし二次バイナリ最適化（QUBO）の定式化)です。QAOAにはQUBO形式が自然に対応するため、本チュートリアルでは後者を使用します。

# %%
dataset = Labs()
instances = dataset.available_instances["quadratic_unconstrained"]
print(f"Dataset:           {dataset.name}")
print(f"Available models:  {dataset.model_names}")
print(f"Instance count:    {len(instances)}")
print(f"First 5 instances: {instances[:5]}")

# %% [markdown]
# ここでは$n=5$のインスタンスである`labs005`を選びます。このQUBO定式化では合計$n + n(n-1) = 25$個のバイナリ変数(5個の系列ビット$x_i$と$n(n-1) = 20$個の補助$z_{i,k}$)が宣言されています。目的関数にはすでにペナルティが含まれ、式に現れる補助ビットは系列の位置の組に対応する10個です。そのため、回路は15量子ビットを使います。ローカルでシミュレートできる程度に小さく、かつQAOAとして非自明なサイズです。

# %%
instance_name = "labs005"
instance, reference_solution = dataset("quadratic_unconstrained", instance_name)

print(f"OMMX variables:    {instance.num_variables}")
print(f"OMMX constraints:  {instance.num_constraints}")
print(f"Reference E(s):    {reference_solution.objective}")
print(f"Reference feasible: {reference_solution.feasible}")

# %% [markdown]
# 同梱の参照解から、$n=5$における既知の最適値が$E^\star = 2$であることが分かります。QAOAとSCIPの結果はこの値と比較していきます。

# %% [markdown]
# ## アルゴリズム: QAOA
#
# ここでは[QAOA for MaxCut: Building the Circuit from Scratch](../algorithm/qaoa_maxcut)と同じ手順で、`@qm.qkernel`を使ってQAOA回路を一から構築します。各ゲートの導出はそちらのチュートリアルに任せ、ここでは実装に焦点を当てます。


# %% [markdown]
# ### OMMXインスタンスからスピンモデルを作る
#
# `Instance.to_qubo()`でQUBOを取り出し、各バイナリ変数をスピンで$x_i=(1-Z_i)/2$と表します。エネルギーの尺度を揃えるため、スピンの係数と定数項を、定数項以外で絶対値が最大の係数（`labs005`では10,028）で割ります。


# %%
# `Instance.to_qubo()`はインスタンスをmutateする(ペナルティ法で制約を目的関数に畳み込む)。
# 呼び出し元のインスタンスを保つために、bytes経由でラウンドトリップさせる。
instance_for_qubo = ommx.v1.Instance.from_bytes(instance.to_bytes())
qubo, qubo_constant = instance_for_qubo.to_qubo()
binary_model = BinaryModel.from_hubo(qubo, qubo_constant)
spin_model = binary_model.to_ising().normalize_by_abs_max()
num_qubits = spin_model.num_variables
print(f"QAOA qubits: {num_qubits}")

# %% [markdown]
# ### コストハミルトニアン
#
# 最適化に「ショット推定」ではなく「厳密な期待値」を用いるため、スピンモデルの係数からIsing型のコストハミルトニアンを直接構築します。線形項は$Z_i$、2次項は$Z_i Z_j$に対応します。

# %%
# Map OMMX variable IDs to consecutive qubit indices.
qubit_index = {var: i for i, var in enumerate(spin_model.variables)}
coefficients = spin_model.coefficients
couplings = {
    (qubit_index[ids[0]], qubit_index[ids[1]]): value
    for ids, value in coefficients.items()
    if len(ids) == 2
}
fields = tuple(coefficients.get((var,), 0.0) for var in spin_model.variables)

cost_hamiltonian = SparsePauliOp.from_sparse_list(
    [("ZZ", [i, j], value) for (i, j), value in couplings.items()]
    + [("Z", [i], value) for i, value in enumerate(fields)]
    + [("I", [0], spin_model.constant)],
    num_qubits=num_qubits,
)

# %% [markdown]
# ### QAOAのqkernel
#
# 回路は3つの小さなQamomileカーネルを組み合わせます。まずアダマールゲートで全ビット列の一様な重ね合わせを作ります。コスト層では、正規化した二次・一次の係数をそれぞれ$J_{ij}$、$h_i$として$R_{ZZ}(J_{ij}\gamma)$と$R_Z(h_i\gamma)$を適用します。これらのゲートでは、指数に入る角度は指定した回転角の半分です。ミキサー層では各量子ビットに$R_X(2\beta)$を適用します。サンプリング用のカーネルは、得られた状態を計算基底（各量子ビットの0と1）で測定します。


# %%
@qm.qkernel
def superposition(n: qm.Const[qm.Int]) -> qm.Q[qm.Bits[n]]:
    q = qm.qubits(n)
    for i in range(n):
        q = qm.h(q, target=i)
    return q


@qm.qkernel
def cost_layer(
    q: qm.Q[qm.Bits[n]],
    n: qm.Const[qm.Int],
    edge_count: qm.Const[qm.Int],
    edges: qm.Const[qm.Matrix[qm.Int, edge_count, 2]],
    quad: qm.Const[qm.Vector[qm.Real, edge_count]],
    linear: qm.Const[qm.Vector[qm.Real, n]],
    gamma: qm.Real,
) -> qm.Q[qm.Bits[n]]:
    for k in range(edge_count):
        q = qm.rzz(q, quad[k] * gamma, left=edges[k][0], right=edges[k][1])
    for i in range(n):
        q = qm.rz(q, linear[i] * gamma, target=i)
    return q


@qm.qkernel
def mixer_layer(
    q: qm.Q[qm.Bits[n]], n: qm.Const[qm.Int], beta: qm.Real
) -> qm.Q[qm.Bits[n]]:
    for i in range(n):
        q = qm.rx(q, 2.0 * beta, target=i)
    return q


@qm.qkernel
def qaoa_state(
    n: qm.Const[qm.Int],
    edge_count: qm.Const[qm.Int],
    edges: qm.Const[qm.Matrix[qm.Int, edge_count, 2]],
    quad: qm.Const[qm.Vector[qm.Real, edge_count]],
    linear: qm.Const[qm.Vector[qm.Real, n]],
    layers: qm.Const[qm.Int],
    gammas: qm.Param[qm.Vector[qm.Real, layers]],
    betas: qm.Param[qm.Vector[qm.Real, layers]],
) -> qm.Q[qm.Bits[n]]:
    q = superposition(n)
    for layer in range(layers):
        q = cost_layer(q, n, edge_count, edges, quad, linear, gammas[layer])
        q = mixer_layer(q, n, betas[layer])
    return q


@qm.qkernel
def qaoa_sample(
    n: qm.Const[qm.Int],
    edge_count: qm.Const[qm.Int],
    edges: qm.Const[qm.Matrix[qm.Int, edge_count, 2]],
    quad: qm.Const[qm.Vector[qm.Real, edge_count]],
    linear: qm.Const[qm.Vector[qm.Real, n]],
    layers: qm.Const[qm.Int],
    gammas: qm.Param[qm.Vector[qm.Real, layers]],
    betas: qm.Param[qm.Vector[qm.Real, layers]],
) -> qm.Shot[qm.Bits[n]]:
    q = qaoa_state(n, edge_count, edges, quad, linear, layers, gammas, betas)
    return qm.measure(q)


# %% [markdown]
# ### コンパイルと最適化
#
# `qm.specialize`で問題と層数（$p=3$）を固定し、サンプリング回路を一度だけコンパイルします。期待値の計算には、この回路のコピーから最後の測定を取り除いたものを使います。各角度について、ローカルのQiskit Aerの`EstimatorV2`がシミュレーションした状態から正規化ハミルトニアンの厳密な期待値を求めます。BFGSは角度を少し変えたときの期待値の差を使って角度を更新します。グラフの各点は目的関数を1回評価した値です。初期角度を再現できるよう、NumPyの乱数シードを固定します。


# %%
p = 3
fixed = {
    "n": num_qubits,
    "edge_count": len(couplings),
    "edges": tuple(couplings),
    "quad": tuple(couplings.values()),
    "linear": fields,
    "layers": p,
}
endpoint = qm.qiskit.Aer()
sampling = qm.compile(qm.specialize(qaoa_sample, **fixed), to=endpoint.target)
# The energy calculation uses the same gates up to the final measurements.
state_circuit = qm.qiskit.load_circuit(sampling).remove_final_measurements(
    inplace=False
)
state_circuit = transpile(state_circuit, AerSimulator(), optimization_level=0)
estimator = EstimatorV2(
    options={"backend_options": {"method": "statevector", "max_parallel_threads": 1}}
)

SEED = 42
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
maxiter = 1 if docs_test_mode else 50

rng = np.random.default_rng(SEED)
initial_params = rng.uniform(0, np.pi, 2 * p)
cost_history: list[float] = []


def angles(params: np.ndarray) -> dict[str, tuple[float, ...]]:
    return {"gammas": tuple(params[:p]), "betas": tuple(params[p:])}


def cost_fn(params: np.ndarray) -> float:
    """Return the exact expectation of the normalized cost Hamiltonian."""
    invocation = qm.Invocation.create(parameters=angles(params))
    circuit = qm.qiskit.bind_circuit(state_circuit, sampling, invocation)
    result = estimator.run([(circuit, cost_hamiltonian)], precision=0.0).result()
    energy = float(result[0].data.evs)
    cost_history.append(energy)
    return energy


t0 = time.perf_counter()
res = minimize(
    cost_fn,
    initial_params,
    method="BFGS",
    options={"maxiter": maxiter},
)
qaoa_optimize_time = time.perf_counter() - t0

print(f"Optimized expectation value (normalized): {res.fun:.4f}")
print(f"Function evaluations:                     {res.nfev}")
print(f"Wall time:                                {qaoa_optimize_time:.2f} s")

# %%
plt.figure(figsize=(8, 4))
plt.plot(cost_history, color="#2696EB")
plt.xlabel("Objective evaluation")
plt.ylabel("Expectation value (normalized)")
plt.title(f"QAOA Optimization Progress ({instance_name})")
plt.show()

# %% [markdown]
# ### 最終サンプリング
#
# コンパイル済みのサンプリング回路を、`qm.run`でローカルの`qm.qiskit.Aer()`上で実行します。選んだ角度で4,096回測定し、各結果を元の`ommx.v1.Instance`で採点すると、返ってくる`ommx.v1.SampleSet`から元のQUBO目的関数の値を直接読み取れます。このQUBO定式化では、補助変数$z$が積$x_i x_{i+k+1}$を正しく表現しているサンプルではペナルティが発生せず、目的関数値がLABSエネルギー$E(\boldsymbol{s}) = \sum_k c_k^2$と一致します。一方、$z = x_i x_{i+k+1}$という暗黙の関係を破ったサンプルには、ペナルティが加算されます。


# %%
def evaluate_with_ommx(
    result: qm.SampleResult, model: BinaryModel, ommx_instance: ommx.v1.Instance
) -> ommx.v1.SampleSet:
    """Score each measured bitstring in the original OMMX variable order."""
    samples = ommx.v1.Samples({})
    next_id = 0
    for row in model.decode(result).samples:
        ids = list(range(next_id, next_id + row.occurrences))
        samples.append(ids, ommx.v1.State(dict(row.assignment)))
        next_id += row.occurrences
    return ommx_instance.evaluate_samples(samples)


final_shots = 1 if docs_test_mode else 4096
final_result = qm.run(
    sampling, on=endpoint, parameters=angles(res.x), shots=final_shots, seed=SEED
).result
qaoa_sample_set = evaluate_with_ommx(final_result, binary_model, instance)

qaoa_summary = qaoa_sample_set.summary
qaoa_best = qaoa_sample_set.best_feasible
qaoa_best_E = int(round(qaoa_best.objective))
ref_E = int(reference_solution.objective)

assert len(qaoa_summary) == final_shots

print(f"Shots:                {len(qaoa_summary)}")
print(f"QAOA best objective:  {qaoa_best_E}")
print(f"Reference E*:         {ref_E}")

# %% [markdown]
# ### 目的関数値の分布
#
# ヒストグラムは、選んだ角度で測定した各ビット列の元のQUBO目的関数値を示します。赤い破線が最適値$E^\star=2$です。左端の棒には、補助ビットが整合しているものの系列が最適ではない値6、10、30も含まれます。ヒット率に数えるのは値が2のショットだけです。`labs005`では、補助ビットに不整合があると目的関数値は少なくとも10,002になります。


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
# **SCIP**は分枝限定法ベースのMILP/QUBOソルバーで、厳密解を求めることができます。同じ`ommx.v1.Instance`を`ommx_pyscipopt_adapter.OMMXPySCIPOptAdapter.solve`に渡すと、PySCIPOpt経由で問題がSCIPに渡され、元のインスタンスに対して評価された`ommx.v1.Solution`が返ります。そのため、`.objective`をQAOAの各ショットで得た元のQUBO目的関数値と直接比較できます。

# %%
t0 = time.perf_counter()
scip_solution = ommx_pyscipopt_adapter.OMMXPySCIPOptAdapter.solve(instance)
scip_solve_time = time.perf_counter() - t0

scip_E = int(round(scip_solution.objective))
assert scip_solution.feasible
assert scip_solution.objective == reference_solution.objective
print(f"SCIP objective: {scip_E}")
print(f"SCIP feasible: {scip_solution.feasible}")
print(f"Wall time:    {scip_solve_time:.3f} s")

# %% [markdown]
# ## 結果の比較
#
# このインスタンスに対し、SCIPは最適解を1つ返します。QAOAではビット列の分布から解を測定します。そこでQAOA側は**ベストショット**(全サンプル中で最も低い目的関数値を達成したビット列)と、参照最適値に対する**ヒット率**(その値を達成したショットの割合)を見てみます。


# %%
# ヒット率: 参照最適値を達成したショットの割合
hits = int((qaoa_summary["objective"] == reference_solution.objective).sum())
hit_rate = hits / final_shots

print(f"{'Solver':<22} {'QUBO':>8} {'Time (s)':>12}")
print("-" * 46)
print(f"{'Reference (bundled)':<22} {ref_E:>8} {'-':>12}")
print(f"{'SCIP (classical)':<22} {scip_E:>8} {scip_solve_time:>12.3f}")
print(f"{'QAOA (best shot)':<22} {qaoa_best_E:>8} {qaoa_optimize_time:>12.2f}")
print()
print(
    f"QAOA hit rate on QUBO optimum {ref_E}: {hit_rate:.1%}  ({hits}/{final_shots} shots)"
)

# %% [markdown]
# このベンチマークから読み取れることは大きく2点あります。
#
# 1. **最適値の到達:** SCIPとQAOAベストショットのどちらも参照最適値$E^\star = 2$に到達しており、QAOAは$n = 5$かつ$p = 3$という軽い設定でも最適系列を見つけられることが分かります。
# 2. **ヒット率:** 最後の4,096回の測定のうち、補助ビットも含めたQUBOの目的関数値が2になったのは約4.6%でした。
#
# なお、経過時間の値はあくまでも参考に留まります。SCIPの計測値はCPU上でのソルバーの実行時間であるのに対し、QAOA側の計測値は状態ベクトルシミュレータ上での古典・量子最適化ループ全体を含み、いずれも実行環境に依存します。このため、実際はより公正な評価方法が必要です。いずれにせよ、OMMX Quantum BenchmarksのようなデータセットとQamomileを組み合わせることで、量子アルゴリズムと古典ソルバーを様々な指標のもとで容易に比較することができます。

# %% [markdown]
# ## まとめ
#
# 本チュートリアルでは次のことを行いました。
#
# 1. OMMX Quantum BenchmarksデータセットからLABSインスタンスをそのまま`ommx.v1.Instance`として読み込みました。
# 2. `Instance.to_qubo()`でQUBOを取り出し、スピンの係数を正規化しました。Qamomileで組んだQAOA回路をローカルのQiskit Aerで実行し、厳密な期待値の計算と最終測定を行いました。
# 3. QAOAの出力(ベストショット、ヒット率、サンプリング分布)を、同じインスタンスに対する`ommx_pyscipopt_adapter.OMMXPySCIPOptAdapter.solve`の結果、およびベンチマークに同梱された参照最適値と比較しました。
#
# QUBOで配布された他のQOBLIBモデルにも同じ回路を使えます。整数変数や制約を含むモデルでは、先にQUBOへの変換が必要な場合があります。必要な量子ビット数は目的関数に現れるバイナリ変数の数で決まります。
