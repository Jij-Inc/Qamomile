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
# tags: [algorithm, optimization, variational]
# ---
#
# # 量子近似最適化アルゴリズム（QAOA）によるMaxCut
#
# このチュートリアルでは、Qamomileの低レベル回路プリミティブを使って、**量子近似最適化アルゴリズム（Quantum Approximate Optimization Algorithm; QAOA）** のパイプラインをステップごとに構築します。高レベルな`QAOAConverter`は使わずに、以下の手順で進めます:
#
# 1. 小さなグラフでMaxCut問題を定義する。
# 2. スピン変数上のイジングモデルとして直接定式化する。
# 3. `@qkernel`を使ってQAOA回路をステップごとに記述する。
# 4. 古典オプティマイザで変分パラメータを最適化する。
# 5. 結果をデコードして可視化する。
#
# また、`qamomile.circuit.algorithm.qaoa_state`が同じ回路を1つの関数呼び出しで提供することを示します。

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install "qamomile[qiskit,visualization]"

# %%
import itertools
import os
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import qamomile.circuit as qmc
from qamomile.circuit.algorithm import qaoa_state
from qamomile.optimization.binary_model import BinaryModel
from qamomile.qiskit import QiskitTranspiler
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

# %% [markdown]
# ## 問題設定: MaxCut
#
# ### 問題の定義
#
# 無向グラフ$G = (V, E)$が与えられたとき、**MaxCut**問題は頂点を2つの集合に分割し、2つの集合間を横切る辺の数を最大化する問題です。
#
# MaxCutは二次無制約二値最適化(Quadratic Unconstrained Binary Optimization; QUBO)と呼ばれる問題クラスであり、バイナリ変数による2次関数の目的関数として表現されます。しかし、MaxCutは、スピン変数を用いることで、より自然に記述することができます。各頂点$i$にスピン$s_i \in \{+1, -1\}$を割り当て、頂点がどちら側に属するかを表現します。辺$(i, j)$が「カットされる」のは$s_i \ne s_j$のときなので、カットされる辺数は
#
# $$
# \text{MaxCut}(\boldsymbol{s})
# = \sum_{(i,j) \in E} \frac{1 - s_i s_j}{2}
# $$
#
# と書けます。
#
# MaxCutやスピングラスの基底状態探索、イジングモデルのベンチマークといったスピン変数で自然に定義される問題は、スピン領域で記述するのが最も扱いやすいです。そこで本チュートリアルではQUBOやバイナリ変数への変換などは挟まず、最初から最後までスピン変数で扱います。

# %% [markdown]
# ### 使用するインスタンス
#
# 5頂点、6辺の小さなグラフを使います。全探索が可能な規模でありながら、自明でない構造を持っています。

# %%
G = nx.Graph()
G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)])
num_nodes = G.number_of_nodes()
assert num_nodes == 5
assert G.number_of_edges() == 6

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
plt.title(f"Graph: {num_nodes} nodes, {G.number_of_edges()} edges")
plt.show()

# %% [markdown]
# ### イジング定式化
#
# $\sum_{(i,j) \in E} (1 - s_i s_j) / 2$を最大化することは、定数を除いて以下の反強磁性イジングハミルトニアンを*最小化*することと等価です:
#
# $$
# H_C(\boldsymbol{s}) = \sum_{(i,j) \in E} s_i s_j.
# $$
#
# 一般のイジング形式$H = \sum_i h_i s_i + \sum_{i < j} J_{ij} s_i s_j$と比較すると、重みなしMaxCutは以下の特徴を持ちます:
#
# - **線形項なし**: 全頂点について$h_i = 0$
# - **一様な相互作用**: 全辺$(i, j) \in E$について$J_{ij} = 1$
#
# `BinaryModel.from_ising`はイジング係数を直接受け取ります。QUBOを経由して変数型を変換する必要はありません。

# %%
ising_quad: dict[tuple[int, int], float] = {
    tuple(sorted((i, j))): 1.0 for i, j in G.edges()
}
ising_linear: dict[int, float] = {}

# 重みつきMaxCutやスピングラスでは係数のスケールが揃っていないため、
# COBYLAなど勾配フリーな最適化での収束を安定させる目的で
# `.normalize_by_abs_max()`を末尾に挟むとよい
spin_model = BinaryModel.from_ising(linear=ising_linear, quad=ising_quad)

print(f"Variable type:          {spin_model.vartype}")
print(f"Linear terms (h_i):     {spin_model.linear}")
print(f"Quadratic terms (J_ij): {spin_model.quad}")
print(f"Constant:               {spin_model.constant}")
# スピン領域のモデルで、線形項は無し、各辺に 1 つの J_ij、定数項も無し。
assert spin_model.vartype.name == "SPIN"
assert spin_model.linear == {}
assert len(spin_model.quad) == G.number_of_edges()
assert spin_model.constant == 0.0

# %% [markdown]
# :::{note}
# `BinaryModel`はQUBO向けの`from_qubo()`や高次版の`from_hubo()`も提供しており、割当問題や制約（ペナルティ項）を伴う問題のようにバイナリ領域で自然に定式化される問題に利用できます。QUBO/JijModelingベースのワークフローについては[QAOAによるグラフ分割](qaoa_graph_partition)を参照してください。
# :::

# %% [markdown]
# ### 厳密解（全探索）
#
# QAOAを実行する前に、すべての$2^n = 32$通りのスピン配置を試して最適な分割を求めておきます。QAOAの結果と比較するための基準になります。

# %%
best_cut = 0
optimal_partitions: list[tuple[int, ...]] = []

for spins in itertools.product([+1, -1], repeat=num_nodes):
    cut = sum(1 for i, j in G.edges() if spins[i] != spins[j])
    if cut > best_cut:
        best_cut = cut
        optimal_partitions = [spins]
    elif cut == best_cut:
        optimal_partitions.append(spins)

print(f"Optimal MaxCut value: {best_cut}")
print(f"Number of optimal partitions: {len(optimal_partitions)}")
for part in optimal_partitions:
    print(f"  {part}")
# この固定された 5 ノード 6 辺のグラフでは MaxCut は 5、それを達成する
# スピン配置は大域反転 s -> -s で結ばれる 2 通りだけ。
assert best_cut == 5
assert len(optimal_partitions) == 2
for part in optimal_partitions:
    assert tuple(-s for s in part) in optimal_partitions

# %% [markdown]
# ## アルゴリズム
#
# **量子近似最適化アルゴリズム（QAOA）** は、Farhi、Goldstone、Gutmannが提案した、組合せ最適化のための変分量子アルゴリズムです{cite:p}`10.48550/arXiv.1411.4028`。目的関数をコストハミルトニアン$H_C$として表現し、ミキサーハミルトニアン$H_M$を使って候補解を探索します。パラメータ付きの量子回路（アンザッツ）でパラメータ付き状態を準備し、古典オプティマイザがコストの期待値$\langle H_C \rangle$を最小化するようにパラメータを更新します。
#
# 深さ$p$のQAOAでは、コストハミルトニアンとミキサーハミルトニアンに関する時間発展演算子を$p$回交互に適用します。コストハミルトニアンの時間発展は目的関数値に応じた位相を与えます。ミキサーハミルトニアンの時間発展は計算基底状態の間で振幅を「ミキシング」し、探索空間内での遷移を促進します。最適化後の状態を測定すると、コストハミルトニアンの低エネルギー状態を得やすくなります。MaxCutでは、この低エネルギー状態がより大きなカットに対応します。
#
# QAOAのアンザッツはパラメータ付き量子状態を準備します:
#
# $$
# |\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle
# = \prod_{l=1}^{p}
#   e^{-i \beta_l H_M} \, e^{-i \gamma_l H_C}
#   \; |{+}\rangle^{\otimes n}
# $$
#
# ここで:
#
# - $|{+}\rangle^{\otimes n}$: 一様重ね合わせ状態（全量子ビットにアダマールゲート）
# - $e^{-i \gamma H_C}$: **コストユニタリ** — イジングコスト$H_C$の場合、二次項は$\text{RZZ}$ゲート、一次項は$\text{RZ}$ゲートに分解されます。
# - $e^{-i \beta H_M}$: **ミキサーユニタリ** — $H_M = \sum_i X_i$の場合、各量子ビットへの$\text{RX}(2\beta)$になります。
# - $p$: レイヤー数（アンザッツの深さ）
#
# スピンと計算基底の対応は量子力学の標準的な規約$Z|0\rangle = |0\rangle$, $Z|1\rangle = -|1\rangle$に従います。すなわち、測定結果$0$はスピン$+1$、測定結果$1$はスピン$-1$に対応します。

# %% [markdown]
# ## Qamomileによる実装
#
# それでは、各コンポーネントを`@qkernel`として実装していきましょう。
#
# ### スクラッチ実装
#
# まず、Qamomileの低レベル回路プリミティブを使ってQAOA回路の各要素を自分で構築し、完成したアンザッツを最適化のためにトランスパイルします。

# %% [markdown]
# #### ステップ1: 一様重ね合わせ状態の準備
#
# 全量子ビットにアダマールゲートを適用し、一様な重ね合わせ状態$|{+}\rangle^{\otimes n}$を作ります。

# %%
@qmc.qkernel
def superposition(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(n, name="q")
    for i in qmc.range(n):
        q[i] = qmc.h(q[i])
    return q


# %% [markdown]
# #### ステップ2: コスト層
#
# コストユニタリ$e^{-i \gamma H_C}$を適用します。
#
# :::{note}
# Qamomileの回転ゲートは$1/2$の因子を含む規約を使います。$\text{RZ}(\theta) = e^{-i \theta Z / 2}$、$\text{RZZ}(\theta) = e^{-i \theta Z \otimes Z / 2}$です。$e^{-i \gamma H_C}$と厳密に一致させるため、`rzz`には$2 J_{ij} \gamma$、`rz`には$2 h_i \gamma$を渡します。
# :::
#
# 重みなしMaxCutでは`linear`引数は空ですが、そのまま引数として残しておきます。こうすることで、線形項$h_i$を持つ重みつきMaxCutや一般のスピングラスハミルトニアンにそのまま流用できます。


# %%
@qmc.qkernel
def cost_layer(
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    q: qmc.Vector[qmc.Qubit],
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    for (i, j), Jij in quad.items():
        q[i], q[j] = qmc.rzz(q[i], q[j], angle=2.0 * Jij * gamma)
    for i, hi in linear.items():
        q[i] = qmc.rz(q[i], angle=2.0 * hi * gamma)
    return q


# %% [markdown]
# #### ステップ3: ミキサー層
#
# ミキサーユニタリ$e^{-i \beta H_M}$を適用します（$H_M = \sum_i X_i$）。$\text{RX}(\theta) = e^{-i \theta X / 2}$なので、$e^{-i \beta X_i}$を実装するには$\theta = 2\beta$とします。


# %%
@qmc.qkernel
def mixer_layer(
    q: qmc.Vector[qmc.Qubit],
    beta: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    n = q.shape[0]
    for i in qmc.range(n):
        q[i] = qmc.rx(q[i], angle=2.0 * beta)
    return q


# %% [markdown]
# #### ステップ4: QAOAアンザッツを作る
#
# `qaoa_ansatz`では、ここまでに定義した回路要素を1つの量子カーネルにまとめます。まず$n$個の量子ビットを一様重ね合わせ状態に準備します。続いて$p$個のレイヤーを順に実行し、各レイヤー$l$で変分パラメータ$\gamma_l$を使うコスト層と、$\beta_l$を使うミキサー層を適用します。最後にすべての量子ビットを測定し、候補となるカットを評価するためのビット列を返します。


# %%
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


qaoa_ansatz.draw(
    p=3,
    quad=spin_model.quad,
    linear=spin_model.linear,
    n=num_nodes,
    inline=True,
    fold_loops=True,
)


# %% [markdown]
# #### 回路のトランスパイル
#
# Qamomileで書いた量子カーネルを、シミュレータで実行するためにトランスパイルします。問題の構造（イジング係数、量子ビット数、レイヤー数）はバインドし、`gammas`と`betas`はオプティマイザがチューニングするランタイムパラメータとして残します。
# `QiskitTranspiler`はQamomileの量子カーネルをQiskitの回路に変換し、Qiskitの`AerSimulator`で実行可能な形式にします。

# %%
transpiler = QiskitTranspiler()
p = 3  # QAOAレイヤー数

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
# ### `qaoa_state`を使う
#
# `qamomile.circuit.algorithm.qaoa_state`には、上で実装した重ね合わせ、コスト層、ミキサー層、レイヤーのループがまとめられています。引数には同じイジング係数（`quad`, `linear`）と変分パラメータ（`gammas`, `betas`）を渡します。そのため、各要素を個別に定義せずにQAOA状態を作れます。

# %%
@qmc.qkernel
def qaoa_builtin(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    n: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    q = qaoa_state(p=p, quad=quad, linear=linear, n=n, gammas=gammas, betas=betas)
    return qmc.measure(q)


qaoa_builtin.draw(
    p=3,
    quad=spin_model.quad,
    linear=spin_model.linear,
    n=num_nodes,
    inline=True,
    fold_loops=True,
)


# %% [markdown]
# ## 結果
#
# それでは、実装したQAOA回路を使って、実際にMaxCut問題を解いてみましょう。
#
# ### パラメータの最適化
#
# QAOAは目的関数を最小化するようにパラメータを更新します。この例では、勾配計算が不要なCOBYLA法を使います。
# `scipy.optimize.minimize`は、パラメータを引数に持つ目的関数を与えることでパラメータの最適化を行います。`cost_fn`は、与えられたパラメータに対して回路をサンプリングし、サンプルから平均エネルギーを評価します。この`cost_fn`を`scipy.optimize.minimize`に渡すことで、パラメータの最適化が行われます。また、後で最適化の過程を確認するために、各パラメータセットに対する平均エネルギーを`cost_history`に記録します。
#
# 本チュートリアルでは再現性を確保するため、`AerSimulator`に`seed_simulator=SEED`を渡してショットごとの擬似乱数サンプリングを決定的にし、NumPyの乱数生成器も同じ値でシードして初期変分パラメータを固定し、スレッド間で乱数ドローが交錯しないよう`max_parallel_threads=1`に設定します。シングルスレッド化は若干の性能低下と引き換えに完全な再現性を得るための設定で、実運用コードでは省略するか、テスト/ドキュメントビルド時のみ有効化する形で構いません。

# %%
SEED = 42


def make_seeded_backend() -> AerSimulator:
    """本チュートリアル用に決定的サンプリングを行うAerSimulatorを返す。"""
    return AerSimulator(seed_simulator=SEED, max_parallel_threads=1)


executor = transpiler.executor(backend=make_seeded_backend())
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
sample_shots = 256 if docs_test_mode else 2048
maxiter = 20 if docs_test_mode else 500
cost_history: list[float] = []


def cost_fn(params):
    gammas = list(params[:p])
    betas = list(params[p:])
    result = executable.sample(
        executor,
        shots=sample_shots,
        bindings={"gammas": gammas, "betas": betas},
    ).result()
    decoded = spin_model.decode_from_sampleresult(result)
    energy = decoded.energy_mean()
    cost_history.append(energy)
    return energy


rng = np.random.default_rng(SEED)
initial_params = rng.uniform(-np.pi / 2, np.pi / 2, 2 * p)
assert initial_params.shape == (2 * p,)

res = minimize(cost_fn, initial_params, method="COBYLA", options={"maxiter": maxiter})

print(f"Optimized cost: {res.fun:.4f}")
print(f"Optimal params: {[round(v, 4) for v in res.x]}")
assert len(cost_history) == res.nfev
assert len(res.x) == 2 * p
if docs_test_mode:
    # docs テストモードでは COBYLA は maxiter の予算で打ち切られる。
    assert res.nfev == maxiter

# %%
plt.figure(figsize=(8, 4))
plt.plot(cost_history, color="#2696EB")
plt.xlabel("Iteration")
plt.ylabel("Cost (mean energy)")
plt.title("QAOA Optimization Progress")
plt.show()

# %% [markdown]
# ### 結果のデコードと分析
#
# 最適化されたパラメータで回路をサンプリングし結果を得ます。量子回路の測定結果は、対応する古典変数（バイナリ変数またはスピン変数）に応じてデコードされます。`decode_from_sampleresult`はスピン変数（+1 / -1）のサンプルを返すので、直接カット辺数を数えられます。

# %%
gammas_opt = list(res.x[:p])
betas_opt = list(res.x[p:])

final_result = executable.sample(
    executor,
    shots=sample_shots,
    bindings={"gammas": gammas_opt, "betas": betas_opt},
).result()

decoded = spin_model.decode_from_sampleresult(final_result)

# %%
cut_distribution: Counter[int] = Counter()
best_qaoa_cut = 0
best_qaoa_sample = None

for sample, _energy, occ in zip(
    decoded.samples, decoded.energy, decoded.num_occurrences
):
    # sampleは{頂点インデックス: スピン値 (+1 or -1)}の辞書
    spins = [sample[i] for i in range(num_nodes)]
    cut = sum(1 for i, j in G.edges() if spins[i] != spins[j])
    cut_distribution[cut] += occ
    if cut > best_qaoa_cut:
        best_qaoa_cut = cut
        best_qaoa_sample = spins

print(f"Best QAOA cut: {best_qaoa_cut}  (optimal: {best_cut})")
print(f"Best partition (spins): {best_qaoa_sample}")
# QAOA は全探索の最適値を超えられず、分布は最終サンプルの全ショットを
# 漏れなく数え上げているはず。
assert best_qaoa_cut <= best_cut
assert sum(cut_distribution.values()) == sample_shots

# %% [markdown]
# ここまで見てきた流れは、組み込みの`qaoa_state`でも同様に実行できます。
#
# 組み込み関数を使って同じ構造の回路が実装されていることを確認します。下記の各executorは同じ`seed_simulator=SEED`で初期化されているため、シードを揃えた条件下では回路が完全一致しているならばサンプル列も平均エネルギーも完全一致します。なお、有限ショットによる推定誤差（ショットノイズ）はシードを揃えても消えるわけではなく、各回路に対して常に存在します。したがって両者の平均エネルギーに差分が残る場合は、ショットノイズが原因ではなく手動版と組み込み版の回路が（ゲート順序やコンパイル過程の違いなどで）ビット単位では一致していないことを示します。

# %%
exe_builtin = transpiler.transpile(
    qaoa_builtin,
    bindings={
        "p": p,
        "quad": spin_model.quad,
        "linear": spin_model.linear,
        "n": num_nodes,
    },
    parameters=["gammas", "betas"],
)

executor_manual = transpiler.executor(backend=make_seeded_backend())
executor_builtin = transpiler.executor(backend=make_seeded_backend())

result_manual = executable.sample(
    executor_manual,
    shots=sample_shots,
    bindings={"gammas": gammas_opt, "betas": betas_opt},
).result()

result_builtin = exe_builtin.sample(
    executor_builtin,
    shots=sample_shots,
    bindings={"gammas": gammas_opt, "betas": betas_opt},
).result()

decoded_manual = spin_model.decode_from_sampleresult(result_manual)
decoded_builtin = spin_model.decode_from_sampleresult(result_builtin)
print(f"Manual   mean energy: {decoded_manual.energy_mean():.4f}")
print(f"Built-in mean energy: {decoded_builtin.energy_mean():.4f}")
# 同じシード + ビット単位で同一の回路 ⇒ サンプル列も平均も完全一致するはず。
# 差分が出る場合は、手動版と組み込み版がビット単位で異なる回路を生成したことを
# 意味する（ゲート順序やコンパイル経路の違いなど）。ショットノイズの残差ではない。
assert decoded_manual.energy_mean() == decoded_builtin.energy_mean()

# %% [markdown]
# まず、最終サンプルをカット辺数ごとに集計します。棒の高さは各カット値が得られた回数を表し、サンプル分布が最適値付近に集中しているかを確認できます。

# %%
cuts = sorted(cut_distribution.keys())
counts = [cut_distribution[c] for c in cuts]
near_optimal_shots = sum(
    count for cut, count in cut_distribution.items() if cut >= best_cut - 2
)
# ここでは既知の最適値との差が2以内のカットを「最適値付近」とする。
assert near_optimal_shots / sample_shots > 0.8

plt.figure(figsize=(8, 4))
plt.bar([str(c) for c in cuts], counts, color="#2696EB")
plt.xlabel("Cut size")
plt.ylabel("Frequency")
plt.title("Distribution of MaxCut Values from QAOA")
plt.show()

# %% [markdown]
# 最終サンプルの分布は、事前に計算した最適値の5付近に集中していることがわかります。
#
# 次に、最終サンプルで観測された最良の分割を可視化します。ノードの色は2つのスピングループを表すため、異なる色のノードを結ぶ辺がカットされた辺です。タイトルには得られたカット辺数を表示します。

# %%
if best_qaoa_sample is not None:
    color_map = [
        "#FF6B6B" if best_qaoa_sample[i] == +1 else "#4ECDC4" for i in range(num_nodes)
    ]
    plt.figure(figsize=(5, 4))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=color_map,
        node_size=700,
        edgecolors="black",
    )
    plt.title(f"QAOA partition (cut = {best_qaoa_cut})")
    plt.show()

# %% [markdown]
# ## まとめ
#
# このチュートリアルでは:
#
# 1. MaxCut問題を定義し、QUBO/バイナリ変数を経由せずに*直接*スピン変数上のイジングハミルトニアンとして記述しました。
# 2. `BinaryModel.from_ising`でスピン領域の`BinaryModel`を構築しました。
# 3. QAOA回路の全コンポーネント — 重ね合わせ、コスト層、ミキサー層、完全なアンザッツ — を`@qkernel`として実装しました。
# 4. 古典最適化ループを実行し、スピン領域のまま結果をデコードしました。
# 5. `qamomile.circuit.algorithm.qaoa_state`が同じ回路を1つの関数呼び出しで提供することを確認しました。
#
# この「スピンから始める」レシピは、スピングラス基底状態探索、重みつきMaxCut、Sherrington–Kirkpatrickモデルといった任意のイジング型問題にそのまま適用できます。$h_i$と$J_{ij}$を`BinaryModel.from_ising`に渡し、上で作成した回路コンポーネントを再利用するだけです。バイナリ変数で自然に定式化される問題や、制約（ペナルティ項）が必要な問題については、JijModelingと高レベルの`QAOAConverter`を使う[QAOAによるグラフ分割](qaoa_graph_partition)を参照してください。
