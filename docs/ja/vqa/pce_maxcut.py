# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: qamomile
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PCEでMaxCutを解く: 20変数を3量子ビットで
#
# このチュートリアルではQamomileの`PCEConverter`を使ったPauli Correlation Encoding (PCE) によるMaxCut問題の解法を扱います。PCEは$N$個の二値変数を、$n = \mathcal{O}(N^{1/k})$量子ビットというはるかに小さなレジスタ上の$k$体Pauli相関子の期待値へ圧縮します。これにより、QAOAのような変数1つあたり1量子ビットの符号化方式ではNISQ機器に載らない規模の問題でも変分最適化が可能になります。
#
# 本チュートリアルでは20頂点のMaxCutを$k = 2$で**わずか3量子ビット**で解き、結果を全探索による厳密解と比較して検証します。手順は以下の通りです。
#
# 1. 20頂点のMaxCut問題を定義し、厳密解を全探索で求めます。
# 2. `PCEConverter(instance, k=2)`でPCE符号化を構築し、`get_encoded_pauli_list()`で変数ごとのPauliオブザーバブルを取り出します。
# 3. ハードウェア効率の良い`@qkernel`アンザッツを記述し、各相関子の期待値を`qm.expval(q, P)`で読み出します。
# 4. MaxCut目的関数のtanh緩和した代理損失に対し、`scipy.optimize.minimize`でアンザッツを学習します。
# 5. 最適化済みの期待値を`converter.decode(expectations)`でスピン列にデコードし、分割を可視化します。

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install qamomile

# %% [markdown]
# ## Backgrounds

# %% [markdown]
# ### MaxCut問題とは？
#
# 無向グラフ$G = (V, E)$が与えられたとき、**MaxCut**問題は頂点を2つの集合$S$と$\bar{S}$に分割し、2つの集合間の辺の数を最大化する問題です。各頂点に**スピン変数**$s_i \in \{+1, -1\}$を割り当て、$s_i = +1$なら頂点$i$は$S$に、$s_i = -1$なら$\bar{S}$に入るとすると、カット値はスピンを使って次のように直接書けます。
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
# を**最小化**することと等価です。すなわち$h_i = 0$、各辺で$J_{ij} = \tfrac{1}{2}$、定数項$-|E|/2$を持つIsingモデルです。これがまさにPCEが受け取る形式であり、$x \in \{0, 1\}$ ↔ $s \in \{+1, -1\}$の書き換えは不要です。

# %% [markdown]
# ### グラフの作成
#
# 20頂点の**3-正則**ランダムグラフを使用します（各頂点はちょうど3つの隣接頂点を持つので、$|E| = 3 \cdot 20 / 2 = 30$辺となります）。3-正則MaxCutはPCE論文における代表的なベンチマークです。次数が一様であることでEdwards-Erdős正規化項のスケールがクリーンに定まり、かつ全探索で厳密解を求められる程度の規模に収まります。
#
# `nx.random_regular_graph`は非連結なグラフを生成することがあります（孤立した連結成分は自由なスピンを残し、最適化のランドスケープを退化させます）。そこで連結なグラフが得られるまでseedをずらします。

# %%
import matplotlib.pyplot as plt
import networkx as nx

seed = 42
while True:
    G = nx.random_regular_graph(3, 20, seed=seed)
    if nx.is_connected(G):
        break
    seed += 1
print(f"Using seed = {seed} (smallest seed >= 42 producing a connected graph)")

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(6, 5))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="white",
    node_size=500,
    edgecolors="black",
)
plt.title(f"Graph: {num_nodes} nodes, {num_edges} edges")
plt.show()

# %% [markdown]
# ### 厳密解（全探索）
#
# $2^{20} = 1{,}048{,}576$通りすべてのスピン配置を列挙すると100万強の割り当てになります。Pythonのループでは多すぎますが、ベクトル化したNumPyを一回流すだけで1秒未満で完了します。各配置は整数でラベル付けし、ビット$i$が$0$なら$s_i = +1$、$1$なら$s_i = -1$と対応付けたうえで、$s_i \neq s_j$となる辺の数を数えます。これで§5でPCEの結果と比較するための厳密解が得られます。

# %%
import numpy as np

assignments = np.arange(2**num_nodes, dtype=np.int64)
cuts = np.zeros(2**num_nodes, dtype=np.int32)
for i, j in G.edges():
    s_i = 1 - 2 * ((assignments >> i) & 1)  # bit 0 → +1, bit 1 → -1
    s_j = 1 - 2 * ((assignments >> j) & 1)
    cuts += (s_i != s_j).astype(np.int32)

best_cut = int(cuts.max())
optimal_assignment_ints = np.flatnonzero(cuts == best_cut)
print(f"Optimal MaxCut value         : {best_cut}")
print(f"Number of optimal partitions : {len(optimal_assignment_ints)}")

# %% [markdown]
# ## Algorithm
#
# PCEはSciorilli, Borges, Patti, García-Martín, Camilo, Anandkumar, Aolitaによって提案された手法で (https://doi.org/10.48550/arXiv.2401.09421)、QAOAのような変数1つあたり1量子ビットの符号化方式では収まりきらない量子ビット数の領域へ組合せ最適化を押し進めるためのアプローチです。$N$変数の問題に対しPCEは$n = \mathcal{O}(N^{1/k})$量子ビットしか使いません。

# %% [markdown]
# ### PCE符号化
#
# PCEは**圧縮率**$k > 1$を選び、二値変数$i \in \{1, \dots, N\}$をそれぞれ別個の$k$体Pauli相関子$P_i$に割り当てます。$P_i$は$\{X, Y, Z\}$から選んだ恒等演算子でない$k$個のPauli演算子のテンソル積で、$n$量子ビットのうち$k$個に作用します。$n$量子ビット上にこのような相関子は$\binom{n}{k} \cdot 3^k$個存在するので、$n$は次を満たす最小の整数として選びます。
#
# $$
# \binom{n}{k} \cdot 3^k \;\ge\; N.
# $$
#
# $k = 2$ならば$n = \mathcal{O}(\sqrt{N})$、$k = 3$ならば$n = \mathcal{O}(N^{1/3})$となります。本チュートリアルの$N = 20$では、$k = 2$で必要な量子ビット数はわずか$n = 3$です（$\binom{3}{2} \cdot 9 = 27 \ge 20$を満たす最小の整数）。割り当ては決定的に行われます。Qamomileの`PCEEncoder`は固定の辞書式順序（先に量子ビットの組、次にPauliラベル）で相関子を列挙し、最初の$N$個を変数$0, \dots, N-1$へ割り当てます。

# %% [markdown]
# ### コスト関数
#
# パラメータ付きアンザッツ状態$|\Psi(\boldsymbol{\theta})\rangle$が与えられたとき、PCEは離散的なスピン目的関数
#
# $$
# C(\mathbf{s}) \;=\; \sum_i h_i \, s_i \,+\, \sum_{i<j} J_{ij} \, s_i s_j
# $$
#
# を、各スピン$s_i$を**tanh緩和**した相関子期待値$\sigma_i(\boldsymbol{\theta}) = \tanh\bigl(\alpha\, \langle P_i \rangle\bigr)$に置き換え、さらにすべての相関子を開区間$(-1, +1)$内に保つための4次の**正規化項**を加えることで、滑らかな代理損失へ変換します（この開領域では任意のビット列が表現可能です）。
#
# $$
# \mathcal{L}(\boldsymbol{\theta})
# \;=\; \underbrace{\sum_i h_i \, \sigma_i \,+\, \sum_{i<j} J_{ij} \, \sigma_i \sigma_j}_{\mathcal{L}_{\text{data}}}
#       \,+\, \mathcal{L}_{\text{reg}}, \qquad
# \mathcal{L}_{\text{reg}}
# \;=\; \beta \cdot \nu \cdot \!\left[ \frac{1}{N} \sum_i \sigma_i^2 \right]^{\!2}.
# $$
#
# 直観的には、データ項は接続された各ペアで$\sigma_i$と$\sigma_j$を逆符号へ引き寄せ（$J_{ij} \sigma_i \sigma_j$が負になる方向）、正規化項は早すぎる飽和を抑える counterweight として働き、相関子領域の滑らかな内部にオプティマイザを留めることで、誤った候補ビット列に早期に収束しないようにします。
#
# ハイパーパラメータは https://doi.org/10.48550/arXiv.2401.09421 に従います。
#
# - **$\alpha$**（tanhの鋭さ）: $\alpha \sim N^{k/2}$でスケールします。ここで$N$はグラフのノード数（つまりスピン変数の数）、$k$はPCEの圧縮率です。本チュートリアルの20ノード・$k = 2$では$\alpha = 20^{1} = 20$となります。
# - **$\beta = 1/2$**（正規化項の強さ）: 論文がランダムグラフ上で一度だけ調整し、全実験で固定の値です。
# - **$\nu$**（全体スケール）: 自由なハイパーパラメータではなく、グラフから直接計算されるEdwards-ErdősのMaxCut下界$\nu = |E|/2 + (N - 1)/4$です。
#
# MaxCutに限れば、スピンモデルは$h_i = 0$、各辺で$J_{ij} = +\tfrac{1}{2}$なので、データ項は隣接する$\sigma_i, \sigma_j$が逆符号となるとき正確に最小化されます。

# %% [markdown]
# ### デコード
#
# 収束後、PCEは最適化された各相関子期待値を符号関数で離散スピンへ丸めます。
#
# $$
# s_i \;=\; \operatorname{sgn}\!\bigl\langle P_i \bigr\rangle_{\boldsymbol{\theta}^*}
# \;\in\; \{+1, -1\},
# $$
#
# そして二値割り当ては$x_i = (1 - s_i) / 2$として復元されます。

# %% [markdown]
# ### アンザッツの選択
#
# PCEは特定の回路を規定しません。原論文では**ハードウェア効率の良いブリックワーク型アンザッツ**——単一量子ビット回転と2量子ビットエンタングラを交互に積み重ねた、変数の数に比例してパラメータが増えるアンザッツ——を採用しています。本チュートリアルでは`qamomile.circuit.algorithm.basic`が提供する事前定義のレイヤ（`ry_layer`、`rz_layer`、`cz_entangling_layer`）を`depth`回スタックして使い、合計で$2 \cdot n \cdot \text{depth}$個の変分角度を持たせます。

# %% [markdown]
# ### ゲート規約に関する注意
#
# Qamomileの回転ゲートは標準的な$1/2$係数を持ちます: $\text{RY}(\theta) = e^{-i \theta Y / 2}$、$\text{RZ}(\theta) = e^{-i \theta Z / 2}$。`thetas`ベクトルの各要素は**純粋な変分パラメータ**（オプティマイザが自由にスケールできる量）なので、この定数倍は最適な`thetas`値に吸収されてしまいます。したがって明示的に$2$を掛けず、`thetas[i]`をそのまま渡しています。

# %% [markdown]
# ## Implementation

# %% [markdown]
# ### Step 1: BinaryModelとPCEConverterの構築
#
# §1で導出したIsing形式——$h_i = 0$、各辺で$J_{ij} = 1/2$、定数項$-|E|/2$——を`BinaryModel.from_ising`で直接構築し、得られたSPINモデルと圧縮率$k = 2$を`PCEConverter`に渡します。コンバータは内部で`PCEEncoder`を構築し、必要な量子ビット数を計算します。このスケーリングではスピンモデルのエネルギーが**カット値の符号反転**に等しくなるので、カットが大きいほどエネルギーが低くなります。

# %%
from qamomile.optimization.binary_model import BinaryModel
from qamomile.optimization.pce import PCEConverter

quad = {(i, j): 0.5 for i, j in G.edges()}
ising_model = BinaryModel.from_ising(
    linear={v: 0.0 for v in G.nodes()},
    quad=quad,
    constant=-num_edges / 2,
)
converter = PCEConverter(ising_model, k=2)

spin_model = converter.spin_model
print(f"Number of variables  : {spin_model.num_bits}")
print(f"PCE qubit count      : {converter.num_qubits}")
print(f"Compression rate     : k = {converter.k}")
print(f"Compression factor   : {spin_model.num_bits / converter.num_qubits:.1f}x")

# %% [markdown]
# ### Step 2: 変数ごとのPauliオブザーバブルを確認する
#
# `get_encoded_pauli_list()`は変数ごとに1つのHamiltonianを返します。各Hamiltonianは係数1の$k$体Pauli文字列をちょうど1つ含みます。これらが§3で言及した$P_i$オブザーバブルで、最適化ループはアンザッツカーネル内で`qm.expval`を通してこれらを読み出します。同じ列挙はベースとなる`PCEEncoder` (`converter.encoder`) にも残っており、コンバータを介さずに確認することも可能です。

# %%
observables = converter.get_encoded_pauli_list()

print(f"Total observables : {len(observables)}")
for i, P_i in enumerate(observables):
    print(f"  P_{i:2d}: {P_i}")

# %% [markdown]
# ### Step 3: ハードウェア効率の良いアンザッツの定義
#
# アンザッツは一様重ね合わせから出発し、`ry_layer` + `rz_layer` + `cz_entangling_layer`のブリックワーク層を`depth`回適用します。カーネルは$\langle P \rangle$を返しますが、`P`はbindingsで渡される特定のオブザーバブルで、同じカーネルが$P_i$ごとに1回ずつトランスパイルされます。

# %%
import qamomile.circuit as qmc
from qamomile.circuit.algorithm.basic import (
    cz_entangling_layer,
    ry_layer,
    rz_layer,
)


@qmc.qkernel
def pce_ansatz(
    n: qmc.UInt,
    depth: qmc.UInt,
    thetas: qmc.Vector[qmc.Float],
    P: qmc.Observable,
) -> qmc.Float:
    q = qmc.qubit_array(n, name="q")
    for i in qmc.range(n):
        q[i] = qmc.h(q[i])
    for d in qmc.range(depth):
        offset = d * 2 * n
        q = ry_layer(q, thetas, offset)  # type: ignore[arg-type]
        q = rz_layer(q, thetas, offset + n)  # type: ignore[arg-type,operator]
        q = cz_entangling_layer(q)
    return qmc.expval(q, P)


# %% [markdown]
# ### Step 4: オブザーバブルごとに1つのExecutableをトランスパイルする
#
# 各$P_i$はトランスパイラで異なる期待値計算経路を生み出すため、オブザーバブルごとに1回トランスパイルし、得られた`ExecutableProgram`をキャッシュします。コンパイル時の`bindings`は構造的な入力（`n`、`depth`、`P`）を固定し、`parameters=["thetas"]`は変分角度をオプティマイザが呼び出しのたびに設定するランタイムパラメータとして残します。

# %%
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

n = converter.num_qubits
depth = 3
num_thetas = 2 * n * depth

executables = [
    transpiler.transpile(
        pce_ansatz,
        bindings={"n": n, "depth": depth, "P": P_i},
        parameters=["thetas"],
    )
    for P_i in observables
]

print(f"Executables cached : {len(executables)}")
print(f"Variational params : {num_thetas} (= 2 * n * depth)")

# %% [markdown]
# ### Step 5: 変分パラメータの最適化
#
# 古典側のループは現在の`thetas`に対して各オブザーバブルの$\langle P_i \rangle$を評価し、§3のtanh緩和損失（データ項+正規化項）に値を代入してから、`scipy.optimize.minimize`に更新を依頼します。読者がオプティマイザの収束を確認できるよう損失履歴を記録します。

# %%
import os

from scipy.optimize import minimize

executor = transpiler.executor()
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
maxiter = 10 if docs_test_mode else 100

# https://doi.org/10.48550/arXiv.2401.09421 に従ったハイパーパラメータ:
#   alpha = N^(k/2)（N = ノード数、k = PCEの圧縮率）
#   beta  = 1/2（固定。論文ではランダムグラフ上で一度だけチューニング）
#   nu    = |E| / 2 + (N - 1) / 4（無向MaxCutに対するEdwards-Erdős下界）
N = spin_model.num_bits
k = converter.k
alpha = float(N ** (k / 2))
beta = 0.5
nu = num_edges / 2 + (N - 1) / 4
print(f"alpha = {alpha}, beta = {beta}, nu = {nu}")

cost_history: list[float] = []


def measure_expectations(thetas: list[float]) -> list[float]:
    return [
        exe.run(executor, bindings={"thetas": thetas}).result()
        for exe in executables
    ]


def loss(params: np.ndarray) -> float:
    thetas = list(params)
    expvals = measure_expectations(thetas)
    relaxed = [np.tanh(alpha * e) for e in expvals]

    # データ項: スピン目的関数の滑らかな代理損失
    L_data = 0.0
    for (i, j), J_ij in spin_model.quad.items():
        L_data += J_ij * relaxed[i] * relaxed[j]
    for i, h_i in spin_model.linear.items():
        L_data += h_i * relaxed[i]

    # 正規化項: beta * nu * [(1/N) sum tanh^2(alpha <P_i>)]^2
    mean_sq = sum(r**2 for r in relaxed) / N
    L_reg = beta * nu * mean_sq**2

    L_total = L_data + L_reg
    cost_history.append(L_total)
    return L_total


rng = np.random.default_rng(42)
initial_params = rng.uniform(-np.pi, np.pi, num_thetas)

res = minimize(
    loss, initial_params, method="BFGS", options={"maxiter": maxiter}
)

print(f"Final loss: {res.fun:+.4f}")

# %%
plt.figure(figsize=(8, 4))
plt.plot(cost_history, color="#2696EB")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("PCE Optimization Progress")
plt.show()

# %% [markdown]
# ### Step 6: 最適化された期待値のデコード
#
# `PCEConverter.decode(expectations)`は変数ごとの期待値を受け取り、それぞれを符号丸めしてスピンに変換し、**入力モデルと同じvartype**（今回は`ising_model`を`BinaryModel.from_ising`で構築したのでSPIN）で1サンプルの`BinarySampleSet`を返します。報告されるエネルギーは§1で設定した規約——エネルギー$= -\,\text{cut}$——に従うので、デコード時のエネルギーは符号付きカット値としてもそのまま読めます。

# %%
final_expectations = measure_expectations(list(res.x))
sampleset = converter.decode(final_expectations)

print("Final per-variable expectations:")
for i, e in enumerate(final_expectations):
    print(f"  <P_{i:2d}> = {e:+.4f}")
print()
print(f"Decoded vartype : {sampleset.vartype}")
print(f"Decoded energy  : {sampleset.energy[0]:+.4f}")

# %% [markdown]
# ## Run example

# %% [markdown]
# ### 結果のデコードと分析

# %% [markdown]
# #### 最良のカット
#
# デコードされたスピン割り当てをグラフ分割に変換し、§2で得た全探索による厳密解とカット値を比較します。整合性チェックとして、カット値は上記のスピンエネルギーの符号反転に等しくなるはずです。

# %%
sample = sampleset.samples[0]
spins = [sample[i] for i in range(num_nodes)]
pce_cut = sum(1 for i, j in G.edges() if spins[i] != spins[j])

print(f"PCE spin assignment : {spins}")
print(f"PCE cut value       : {pce_cut}")
print(f"Brute-force optimum : {best_cut}")
print(f"Approximation ratio : {pce_cut / best_cut:.3f}")

# %% [markdown]
# #### 最良解の可視化
#
# 各頂点を分割のどちら側に入ったかで色分けします。色が異なる頂点はカットの反対側に位置します。

# %%
color_map = ["#FF6B6B" if spins[i] == 1 else "#4ECDC4" for i in range(num_nodes)]
plt.figure(figsize=(6, 5))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=color_map,
    node_size=500,
    edgecolors="black",
)
plt.title(f"PCE partition (cut = {pce_cut} / optimum = {best_cut})")
plt.show()

# %% [markdown]
# ## Conclusion
#
# このチュートリアルでは以下を行いました。
#
# 1. MaxCutを最初からスピンIsing問題（$s_i \in \{+1, -1\}$、各辺で$J_{ij} = 1/2$、定数項$-|E|/2$）として書き、スピンエネルギーが$-\text{cut}$と一致するように整えたうえで、ベクトル化したNumPyで$2^{20}$通りすべてのスピン配置を全探索して厳密解を求めました。
# 2. 20個のスピン変数を2体Pauli相関子に符号化し、`PCEConverter(ising_model, k=2)`で**わずか3量子ビット**——およそ7倍の圧縮——に押し込み、`get_encoded_pauli_list()`から変数ごとのオブザーバブルを取り出しました。
# 3. ハードウェア効率の良い`@qkernel`アンザッツを構築して`qm.expval`で$\langle P \rangle$を返すようにし、オブザーバブルごとに1回トランスパイルしてtanh緩和したMaxCut損失と論文の4次正規化項（$\alpha = N^{k/2}$、$\beta = 1/2$、$\nu = |E|/2 + (N-1)/4$）で学習しました。
# 4. 最適化された期待値を`PCEConverter.decode(...)`に与えて離散的なスピン割り当てを復元し、全探索による厳密解と近似比を比較しました。
#
# **限界事項:**
#
# - **ハイパーパラメータとアンザッツの深さは依然として調整が必要です。** 論文の$\alpha$、$\beta$、$\nu$スキームを採用しても、アンザッツの表現力が足りなかったり最適化の予算が短かすぎたりすると、tanh代理損失は停滞することがあります。論文がSIで行っているように、ランダムグラフ上で$\alpha$を掃引するのが最初のステップとして典型的です。
# - **変数ごとに1回ずつトランスパイルが必要です。** $N$個の期待値を推定するには`transpile` + `run`を$N$回別々に行うため、量子ビット数が小さい問題でもこのコストが実時間の大半を占めます。
#
# **次のステップ:**
#
# - 標準的な変数1つあたり1量子ビットの方式と比較するには、固定のコストハミルトニアンとビット列サンプリングで小さなグラフを解く[QAOAでMaxCutを解く](qaoa_maxcut)を参照してください。
# - PCEの組合せ的なPauli列挙ではなく、グラフ彩色を使う部分量子ビット符号化については、`qamomile.optimization.qrao`配下のQamomileのQRAOコンバータを参照してください。
