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
# # Pauli Correlation Encoding
#
# このチュートリアルでは、Qamomileの`PauliCorrelationEncoding`を用いて、Pauli Correlation Encoding(PCE)でMaxCut問題を解いてみます。PCEは$N$個のスピン変数を、$n = \mathcal{O}(N^{1/k})$量子ビットのレジスタ上の$k$体Pauli相関演算子の期待値に写像します。これにより、変数1つあたり1量子ビットを使うQAOA定式化よりも必要な量子ビット数を減らします。
#
# 本チュートリアルでは、20頂点のMaxCutインスタンスを$k = 2$で**3量子ビット**に符号化します。`PauliCorrelationEncoding`で符号化を構築し、ハードウェア効率の良い`@qkernel`アンザッツで各相関演算子の期待値を推定します。次に`scipy.optimize.minimize`でアンザッツを最適化し、最後に`encoding.decode`で最終的な期待値をデコードして、全探索によるベースラインと比較します。

# %% [markdown]
# ## インストール
#
# ```bash
# pip install "qamomile[qiskit,visualization]"
# ```

# %%
# Install the latest Qamomile through pip!
# # !pip install "qamomile[qiskit,visualization]"

# %%
import itertools
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import minimize

import qamomile as qm
from qamomile.measurement import Estimator
from qamomile.optimization import BinaryModel, PauliCorrelationEncoding
from qamomile.qiskit import Aer, to_circuit

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# (problem-settings)=
# ## 問題設定: MaxCut


# %% [markdown]
# ### 問題の定義
#
# 無向グラフ$G = (V, E)$が与えられたとき、MaxCut問題は頂点を2つの集合$S$と$\bar{S}$に分割し、2つの集合間の辺の数を最大化する問題です。各頂点にスピン変数$s_i \in \{+1, -1\}$を割り当てます。$s_i = +1$なら頂点$i$は$S$に入り、$s_i = -1$なら$\bar{S}$に入ります。カット値は次のように書けます。
#
# $$
# \text{MaxCut}(\mathbf{s})
# \;=\; \frac{1}{2} \sum_{(i,j) \in E} \bigl(\,1 - s_i s_j\,\bigr).
# $$
#
# 各辺は両端のスピンが反対側にあるとき（$s_i s_j = -1$）に$1$、同じ側にあるとき（$s_i s_j = +1$）に$0$を寄与します。したがってカットを最大化することは、Isingエネルギー
#
# $$
# E(\mathbf{s}) \;=\; \frac{1}{2} \sum_{(i,j) \in E} s_i s_j
#     \;-\; \frac{|E|}{2}, \qquad E(\mathbf{s}) = -\,\text{MaxCut}(\mathbf{s}),
# $$
#
# を**最小化**することと等価です。これは$h_i = 0$、各辺で$J_{ij} = \tfrac{1}{2}$、定数項$-|E|/2$を持つIsingモデルです。PCEはこのスピン形式を直接扱えるため、二値変数$x \in \{0, 1\}$からの追加変換は不要です。


# %% [markdown]
# ### グラフの作成
#
# 20頂点の3-正則ランダムグラフを使用します。各頂点はちょうど3つの隣接頂点を持つので、$|E| = 3 \cdot 20 / 2 = 30$辺となります。3-正則グラフのMaxCutはPCE論文のベンチマークです。このインスタンスは、全探索によって真の最適値を計算できる規模です。
#
# `nx.random_regular_graph`は非連結なグラフを生成することがあるため、連結なグラフが得られるまでseedを増やします。これにより、複数の独立した成分ではなく、1つの分割問題を扱います。

# %%
seed = 42
while True:
    graph = nx.random_regular_graph(3, 20, seed=seed)
    if nx.is_connected(graph):
        break
    seed += 1

NUM_NODES = graph.number_of_nodes()
NUM_EDGES = graph.number_of_edges()
EDGES = tuple(sorted(tuple(sorted(edge)) for edge in graph.edges()))
positions = nx.spring_layout(graph, seed=42)

print(f"Using graph seed {seed}: {NUM_NODES} nodes, {NUM_EDGES} edges")
assert NUM_NODES == 20
assert NUM_EDGES == 30

# %%
plt.figure(figsize=(6, 5))
nx.draw(
    graph,
    positions,
    with_labels=True,
    node_color="white",
    node_size=500,
    edgecolors="black",
)
plt.title(f"Graph: {NUM_NODES} nodes, {NUM_EDGES} edges")
plt.show()

# %% [markdown]
# (pce-algorithm)=
# ## アルゴリズム
#
# PCEはSciorilliら{cite:p}`10.48550/arXiv.2401.09421`が提案した、少ない量子ビット数で組合せ最適化を扱う手法です。標準的なQAOAは変数ごとに1量子ビットを使います。PCEは$N$変数の問題に対し$n = \mathcal{O}(N^{1/k})$量子ビットを使います。


# %% [markdown]
# ### PCE符号化
#
# PCEは相関演算子の次数$k > 1$を選び、スピン変数$i \in \{1, \dots, N\}$をそれぞれ別個の$k$体Pauli相関演算子$P_i$に割り当てます。各$P_i$は$\{X, Y, Z\}$から選んだ恒等演算子でない$k$個のPauli演算子のテンソル積で、$n$量子ビットのうち$k$個に作用します。$n$量子ビット上に存在するこのような相関演算子は$\binom{n}{k} \cdot 3^k$個なので、$n$は次を満たす最小の整数として選びます。
#
# $$
# \binom{n}{k} \cdot 3^k \;\ge\; N.
# $$
#
# $k = 2$ならば$n = \mathcal{O}(\sqrt{N})$、$k = 3$ならば$n = \mathcal{O}(N^{1/3})$となります。この$N = 20$のインスタンスでは、$k = 2$で$n = 3$量子ビットが必要です。これは$\binom{3}{2} \cdot 9 = 27 \ge 20$を満たす最小の整数です。変数から相関演算子への割り当ては決定的に決まります。$n$量子ビット上の$k$体Pauli文字列の固定列挙を1つ用意し、その先頭$N$個を変数$0, \dots, N-1$へ割り当てます。

# %%
model = BinaryModel.from_ising(
    {edge: 0.5 for edge in EDGES},
    constant=-NUM_EDGES / 2.0,
    variables=range(NUM_NODES),
)
encoding = PauliCorrelationEncoding(model, correlator_order=2)

assert model.num_variables == NUM_NODES
assert encoding.num_qubits == 3
assert len(encoding.observables) == NUM_NODES
assert tuple(encoding.correlations) == model.variables
assert all(
    len(observable.terms) == 1
    and len(observable.terms[0][0].factors) == 2
    and observable.terms[0][1] == 1.0
    for observable in encoding.observables
)

print(f"Classical variables: {model.num_variables}")
print(f"PCE qubits: {encoding.num_qubits}")
print(f"Compression ratio: {model.num_variables / encoding.num_qubits:.1f}x")
for variable, observable in encoding.correlations.items():
    print(f"P_{variable:2d}: {observable}")

# %% [markdown]
# ### コスト関数
#
# パラメータ付きアンザッツ状態$|\Psi(\boldsymbol{\theta})\rangle$が与えられたとき、PCEは離散的なスピン目的関数
#
# $$
# C(\mathbf{s}) \;=\; \sum_i h_i \, s_i \,+\, \sum_{i<j} J_{ij} \, s_i s_j
# $$
#
# を、各スピン$s_i$をtanh緩和した相関演算子期待値$\sigma_i(\boldsymbol{\theta}) = \tanh\bigl(\alpha\, \langle P_i \rangle\bigr)$に置き換え、さらに緩和変数の早い飽和を抑える4次の正則化項を加えることで、滑らかな代理損失関数$\mathcal{L}$へ変換します。tanh写像により$\sigma_i$は開区間$(-1, +1)$に入り、この領域では符号丸めで候補ビット列を復元できます。
#
# $$
# \mathcal{L}(\boldsymbol{\theta})
# \;=\; \underbrace{\sum_i h_i \, \sigma_i \,+\, \sum_{i<j} J_{ij} \, \sigma_i \sigma_j}_{\mathcal{L}_{\text{data}}}
#       \,+\, \mathcal{L}_{\text{reg}}, \qquad
# \mathcal{L}_{\text{reg}}
# \;=\; \beta \cdot \nu \cdot \!\left[ \frac{1}{N} \sum_i \sigma_i^2 \right]^{\!2}.
# $$
#
# データ項は接続された各ペアで$\sigma_i$と$\sigma_j$を逆符号へ引き寄せます。つまり$J_{ij} \sigma_i \sigma_j$が負になる方向へ働きます。正則化項は大きな緩和値にペナルティーを与えてこの圧力と釣り合いを取ります。これにより、オプティマイザを領域の滑らかな内部に保ち、劣ったビット列への早期収束を抑えます。
#
# この損失には3つのハイパーパラメータ$\alpha$（tanhの鋭さ）、$\beta$（正則化項の強さ）、$\nu$（全体スケール）が含まれます。これらの値はオプティマイザの収束と最終的なビット列の品質に影響します。本チュートリアルで用いる具体的な値は元論文に従い、[](#pce-optimize-variational-parameters)で設定します。
#
# MaxCutに限れば、スピンモデルは$h_i = 0$、各辺で$J_{ij} = +\tfrac{1}{2}$なので、データ項は隣接する$\sigma_i, \sigma_j$が逆符号となるとき最小になります。


# %% [markdown]
# ### デコード
#
# 収束後、PCEは最適化された各相関演算子の期待値を、符号丸めによって離散スピンへ戻します。
#
# $$
# s_i \;=\; \operatorname{sgn}\!\bigl\langle P_i \bigr\rangle_{\boldsymbol{\theta}^*}
# \;\in\; \{+1, -1\},
# $$
#
# すなわち$\langle P_i \rangle \ge 0$なら$s_i = +1$、それ以外なら$s_i = -1$とします。二値割り当ては$x_i = (1 - s_i) / 2$として復元されます。

# %% [markdown]
# ### Ansatzを定義する
#
# 各ビットへのHに続けて、RY、RZ、隣接ビット間のCXを3層適用する状態準備を使います。3ビット・3層の例には18個の実行時角度があります。Qamomile qkernelは準備した量子変数を合成先へ返します。期待値を計算したり、実行中の状態をホスト側Pythonへ返したりする関数ではありません。
#
# 後で`Estimator`がこの定義を各相関演算子と組み合わせます。図は測定前の3層すべての状態準備を示します。QamomileのRY/RZは標準の半角規約を使うため、`thetas`の各値をそのまま回転へ渡します。

# %% [markdown]
# 測定や実行先を選ぶ前でも、状態準備の定義を検査し、リソースを解析できます。

# %%
DEPTH = 3
NUM_PARAMETERS = 2 * 3 * DEPTH


@qm.qkernel
def pce_state(
    thetas: qm.Param[qm.Vector[qm.Real, 18]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[3]]:  # type: ignore[valid-type]
    """Prepare the three-qubit hardware-efficient PCE trial state.

    Args:
        thetas (qm.Param[qm.Vector[qm.Real, 18]]): Runtime rotation angles.

    Returns:
        qm.Q[qm.Bits[3]]: Prepared PCE state.
    """

    state = qm.qubits(3)
    for target in range(3):
        state = qm.h(state, target=target)
    for depth in range(3):
        base = depth * 6
        for target in range(3):
            state = qm.ry(state, thetas[base + target], target=target)
        for target in range(3):
            state = qm.rz(state, thetas[base + 3 + target], target=target)
        for control in range(2):
            state = qm.cx(state, control=control, target=control + 1)
    return state


program = pce_state.specialize()
study = qm.resources.logical(program)

assert NUM_PARAMETERS == 18
assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (27, 27)
assert study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure() == (3, 3)

# %% [markdown]
# (pce-optimize-variational-parameters)=
# ## 推定器を準備してパラメータを最適化する
#
# 符号化した相関演算子ごとに、再利用可能なQamomileの`Estimator`を作ります。推定器は対応するバックエンドの機能か測定プログラムを生成する経路を選びます。このチュートリアルで実行プログラムの測定を削除したり、オブザーバブルやパラメータの順序を手動変換したりする必要はありません。
#
# 各呼び出しには1相関演算子分の有限ショット予算を渡します。相関演算子が20個なので、損失の1回の評価には`20 * SHOTS_PER_OBSERVABLE`ショットを使います。呼び出しごとに異なる非ゼロのseedで新しい乱数系列を使います。非線形な損失には標本平均を代入するため、ノイズを含む平均へのtanhの適用には偏りが生じ得ます。最終評価では新しい標本と大きめの予算で各相関の不確かさを報告してから、符号で丸めます。ゼロ付近の相関は再測定で符号が変わり得ます。

# %%
state_circuit = to_circuit(program)
assert state_circuit.num_qubits == encoding.num_qubits
assert state_circuit.num_parameters == NUM_PARAMETERS
print(state_circuit.draw(output="text", fold=100))

endpoint = Aer()
estimators = tuple(
    Estimator(program, observable) for observable in encoding.observables
)
SHOTS_PER_OBSERVABLE = 32 if docs_test_mode else 512
FINAL_SHOTS_PER_OBSERVABLE = 128 if docs_test_mode else 2_048
measurement_seeds = itertools.count(100)


def measure_expectations(values, *, shots=SHOTS_PER_OBSERVABLE):
    """Estimate correlators with fresh finite-shot measurements."""
    parameters = {"thetas": tuple(float(value) for value in values)}
    return tuple(
        estimator.estimate(
            on=endpoint,
            parameters=parameters,
            shots=shots,
            seed=next(measurement_seeds),
        )
        for estimator in estimators
    )


ALPHA = float(NUM_NODES ** (2 / 2))
BETA = 0.5
NU = NUM_EDGES / 2.0 + (NUM_NODES - 1) / 4.0
cost_history = []


def loss(values) -> float:
    """Return the tanh-relaxed PCE loss including quartic regularization."""

    estimates = measure_expectations(values)
    expectations = np.array([estimate.value for estimate in estimates])
    relaxed = tuple(float(value) for value in np.tanh(ALPHA * expectations))
    data_term = encoding.relaxed_cost(relaxed) - model.constant
    mean_square = sum(value**2 for value in relaxed) / NUM_NODES
    total = data_term + BETA * NU * mean_square**2
    cost_history.append(total)
    return total


print(f"alpha={ALPHA}, beta={BETA}, nu={NU}")
rng = np.random.default_rng(42)
initial_parameters = rng.uniform(-np.pi, np.pi, NUM_PARAMETERS)
max_iterations = 20 if docs_test_mode else 100
optimization = minimize(
    loss,
    initial_parameters,
    method="COBYLA",
    options={"maxiter": max_iterations, "rhobeg": 0.5},
)

print(f"Final loss: {optimization.fun:+.6f}")
assert len(optimization.x) == NUM_PARAMETERS
assert len(cost_history) == optimization.nfev

# %%
plt.figure(figsize=(8, 4))
plt.plot(cost_history, color="#2696EB")
plt.xlabel("Energy evaluation")
plt.ylabel("Loss")
plt.title("PCE optimization progress")
plt.show()

# %% [markdown]
# ## 最適化したExpectationをDecodeする
#
# 最後のexpectationを元のvariable順にsign-roundします。`DecodedSamples`は
# immutable assignmentと正確なIsing energyを返します。このmodelではenergyは
# cut値の符号反転です。

# %%
final_estimates = measure_expectations(optimization.x, shots=FINAL_SHOTS_PER_OBSERVABLE)
final_expectations = tuple(estimate.value for estimate in final_estimates)
assert all(estimate.shots == FINAL_SHOTS_PER_OBSERVABLE for estimate in final_estimates)
decoded = encoding.decode(final_expectations)
solution = decoded.lowest
spins = solution.values
pce_cut = sum(spins[left] != spins[right] for left, right in EDGES)

print("Final per-variable expectations:")
for index, estimate in enumerate(final_estimates):
    print(f"  <P_{index:2d}> = {estimate.value:+.4f} ± {estimate.standard_error:.4f}")
print("Decoded spins:", dict(spins))
print(f"Decoded energy: {solution.energy:+.1f}")
print(f"PCE cut: {pce_cut}")

assert model.energy(spins) == solution.energy
assert np.isclose(pce_cut, -solution.energy, rtol=0.0, atol=1e-12)

# %% [markdown]
# (pce-classical-baseline)=
# ### 古典ベースライン（全探索）
#
# $2^{20} = 1{,}048{,}576$通りすべてのスピン配置を列挙すると、約100万個の割り当てを調べることになります。単純なPythonループでは多すぎますが、ベクトル化したNumPyを1回通すだけなら短時間で完了します。各配置は整数でラベル付けし、ビット$i$が$0$なら$s_i = +1$、$1$なら$s_i = -1$と対応付けたうえで、$s_i \neq s_j$となる辺の数を数えます。これで次のサブセクションでPCEの結果と比較するための真の最適値が得られます。

# %%
assignments = np.arange(2**NUM_NODES, dtype=np.int64)
cuts = np.zeros(2**NUM_NODES, dtype=np.int32)
for left, right in EDGES:
    left_spins = 1 - 2 * ((assignments >> left) & 1)
    right_spins = 1 - 2 * ((assignments >> right) & 1)
    cuts += (left_spins != right_spins).astype(np.int32)

best_cut = int(cuts.max())
optimal_assignments = np.flatnonzero(cuts == best_cut)
approximation_ratio = pce_cut / best_cut

print(f"Brute-force optimum: {best_cut}")
print(f"Number of optimal partitions: {len(optimal_assignments)}")
print(f"PCE approximation ratio: {approximation_ratio:.3f}")

assert best_cut == 26
assert 0.0 <= approximation_ratio <= 1.0

# %% [markdown]
# #### 解の可視化
#
# 各ノードを所属するパーティション側で色分けします。色が異なるノード同士は、カットの反対側に位置します。

# %%
colors = ["#FF6B6B" if spins[index] == 1 else "#4ECDC4" for index in graph]
plt.figure(figsize=(6, 5))
nx.draw(
    graph,
    positions,
    with_labels=True,
    node_color=colors,
    node_size=500,
    edgecolors="black",
)
plt.title(f"PCE partition (cut = {pce_cut} / optimum = {best_cut})")
plt.show()

# %% [markdown]
# ## まとめ
#
# 本チュートリアルでは、20ノードの3-正則グラフ上のMaxCut問題をPauli Correlation Encoding（PCE）でエンコードし、相関演算子の符号化からデコードによるスピン割り当ての復元までを行う、Qamomileのワークフロー全体を扱いました。
#
# - **量子ビットのリソース効率:** PCEは20個のスピン変数を、2体のPauli相関演算子によってわずか3量子ビットで表現しました。変数あたり1量子ビットを使うQAOA符号化と比べて約7倍の削減です。
# - **代理損失関数による最適化:** 変分ループでは、エネルギーを直接最小化するのではなく、tanh緩和による代理損失関数を最小化しました。デコードした割り当てを全探索の最適値と比較します。有限ショットの学習が最適解を保証するわけではありません。
# - **Qamomileによるエンドツーエンドの流れ:** `PauliCorrelationEncoding`が与えられた古典変数を符号化し、対応するオブザーバブルを`encoding.observables`で取得しました。ハードウェア効率の良い`@qkernel`アンザッツとQamomileの`Estimator`で、明示したショット予算により各相関演算子の期待値を推定し、オプティマイザを使ってパラメータを調整しました。最後に`encoding.decode`を用いて期待値を符号丸めし、対応する古典変数を取り出しました。
#
# 今回の`PauliCorrelationEncoding`のワークフローは、量子ビット数がボトルネックになる任意のQUBO/Ising組合せ最適化問題に応用できます。自分の`BinaryModel`を差し替えれば、上記の符号化・トランスパイル・デコードの手順をそのまま再利用できます。
