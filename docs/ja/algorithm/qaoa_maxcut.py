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
# このチュートリアルでは、小さな**MaxCut**問題をQAOAで解きます。グラフの頂点を2つのグループに分け、グループの間をつなぐ辺をできるだけ多くする問題です。QAOA回路はアダマールゲートと回転ゲートから自分で組み立てるので、アルゴリズムのすべての部分がコードに現れます。
#
# このチュートリアルは次の手順で進みます。
#
# 1. 小さなグラフのMaxCutを、スピン変数を使ったイジングモデルとして書き、すべての割り当てを試して厳密解を求めます。
# 2. QAOA回路を`@qkernel`関数として、部品ごとに組み立てます。
# 3. 古典オプティマイザが角度を調整しながら、回路を繰り返し実行します。
# 4. オプティマイザが返した角度で、回路をもう一度測定します。1回の測定は1つのカットを与えます。測定されたカットのうち最大のものを答えとし、その大きさを厳密解と比べます。そのあと、それぞれのカットが何回測定されたかを数えます。
# 5. Qamomileの組み込み関数`qaoa_state`を使い、1回の呼び出しで同じ状態を作ります。
#
# **前提知識**: 量子ビット、量子ゲート、測定。未習の場合は[はじめての量子カーネル](../tutorial/01_your_first_quantum_kernel.ipynb)から始めてください。QAOAの知識は必要ありません。制約のある問題をQamomileのより高水準な`QAOAExperiment`で解く例は、[QAOAによるグラフ分割](qaoa_graph_partition)を参照してください。

# %% [markdown]
# ## インストール

# %%
# 最新のQamomileとNetworkXをpipからインストールします！
# # !pip install qamomile networkx

# %%
# ruff: noqa: F821 -- dimensions are Qamomile dependent shape binders.

from __future__ import annotations

import itertools
import os
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import minimize

import qamomile as qm
from qamomile.optimization import BinaryModel
from qamomile.stdlib import qaoa_state

# %% [markdown]
# ## 問題設定
#
# (qaoa-maxcut-problem-definition)=
# ### 問題の定義
#
# $G = (V, E)$を、頂点の集合$V$と辺の集合$E$からなる無向グラフとし、$n = |V|$とします。辺は、順序のない頂点の組です。$(i, j)$と$(j, i)$は同じ辺で、$E$についての和では1回だけ数えます。グラフは固定されています。選べるのは、各頂点を2つのグループのどちらに入れるかだけです。このように頂点を2つのグループに分けることを*カット*と呼びます。辺の両端が異なるグループにあるとき、その辺はカットを*またぐ*といいます。2つのグループの間に線を引くと、その線が横切る辺が、ちょうどカットをまたぐ辺です。カットをまたぐ辺の数をカットの*大きさ*と呼びます。**MaxCut**問題は、大きさが最大のカットを求める問題です。
#
# この選択をスピン変数で書きます。各頂点$i$に、グループを表すスピン$s_i \in \{+1, -1\}$を割り当てます。辺$(i, j)$がカットをまたぐのは、ちょうど$s_i \ne s_j$のときです。このとき$s_i s_j = -1$で、そうでなければ$s_i s_j = +1$なので、$(1 - s_i s_j)/2$はカットをまたぐ辺では1、それ以外の辺では0になります。したがって、カットの大きさは
#
# $$
# C(\boldsymbol{s}) = \sum_{(i,j) \in E} \frac{1 - s_i s_j}{2},
# $$
#
# です。目標は、$C(\boldsymbol{s})$を最大にするスピンの割り当て$\boldsymbol{s}$を見つけることです。その最大値を、グラフのMaxCut値と呼びます。
#
# MaxCutは、$s_i = 1 - 2x_i$によってバイナリ変数$x_i \in \{0, 1\}$で書き、Quadratic Unconstrained Binary Optimization（QUBO）問題として扱うこともできます。このとき、1つの辺の寄与は$(1 - s_i s_j)/2$の代わりに$x_i + x_j - 2x_i x_j$になります。このチュートリアルでは、一貫してスピン変数を使います。スピン変数では1つの辺が1つの積$s_i s_j$だけを与え、これが以下で使うイジングモデルの形だからです。

# %% [markdown]
# ### 使用するインスタンス
#
# 5頂点、6辺の固定したグラフを使います。頂点0、1、2が三角形を作り、頂点1、2、3も三角形を作ります。頂点4は頂点3だけにつながっています。5頂点では、スピンの割り当ては$2^5 = 32$通りしかないので、以下ではそのすべてを試し、QAOAを厳密解と比べることができます。

# %%
G = nx.Graph()
G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)])
num_nodes = G.number_of_nodes()
# 頂点の数は辺から決まるので、グラフには辺が1本以上必要です。
assert G.number_of_edges() > 0, "The graph must have at least one edge."
# コードは頂点のラベルを位置として使うので、ラベルは0, 1, ..., n-1でなければなりません。
assert sorted(G.nodes) == list(range(num_nodes)), "Label the vertices 0, 1, ..., n-1."
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
plt.title(f"Graph: {num_nodes} vertices, {G.number_of_edges()} edges")
plt.show()

# %% [markdown]
# 別のグラフを試すには、辺のリストを変更してください。頂点のラベルは、$0, 1, \dots, n-1$と欠番なしで付けてください。コードはこのラベルを、スピンの割り当ての中の位置として使うからです。このチュートリアルのassertにある数、たとえばここでの6辺や、厳密解でのカットの大きさ5は、このグラフでのみ成り立つので、あわせて更新してください。厳密解の計算は$2^n$通りのスピンの割り当てをすべて試すので、$n$が大きくなると急速に遅くなります。[結果](#qaoa-maxcut-result)の節では、実行結果をassertで厳密解と照らし合わせます。測定された最大のカットがグラフの最大カットであること、そして最も多く測定されたカットもグラフの最大カットであることの2つです。QAOAはどちらも保証しません。あなたのグラフでどちらかが成り立たないと、ノートブックはそこで`AssertionError`を出して止まります。これは、そのグラフではQAOAの結果が弱かったことを示していて、コードの誤りではありません。

# %% [markdown]
# ### イジング定式化
#
# $\sum_{(i,j) \in E} 1 = |E|$なので、カットの大きさは次のように書き直せます。
#
# $$
# C(\boldsymbol{s}) = \frac{|E|}{2} - \frac{1}{2} H_C(\boldsymbol{s}),
# \qquad
# H_C(\boldsymbol{s}) = \sum_{(i,j) \in E} s_i s_j.
# $$
#
# 第1項はグラフで決まるので、$C$を最大にすることは$H_C$を最小にすることと同じです。大きさ$C$のカットでは$H_C = |E| - 2C$になります。$H_C$は、相互作用するスピンのモデルであるイジングモデルのエネルギーです。イジングモデルの一般形
#
# $$
# H = \sum_i h_i s_i + \sum_{i < j} J_{ij} s_i s_j,
# $$
#
# で見ると、重みなしのMaxCutには線形項がありません（すべての頂点で$h_i = 0$）。結合は、すべての辺$(i, j) \in E$で$J_{ij} = 1$、辺でない組では$J_{ij} = 0$です。正の結合は、2つのスピンが異なるときにエネルギーを下げます。それは、ちょうどその辺がカットをまたぐときです。このようなモデルを反強磁性的と呼びます。
#
# [アルゴリズム](#qaoa-maxcut-algorithm)の節では、$H_C$がQAOAのコストハミルトニアンになります。以下では、`BinaryModel.from_ising`が、バイナリ変数に変換せずに、これらの係数から直接モデルを作ります。そのあと、係数をモデルからモデル自身の順序で読み戻し、回路を作るのに使います。バイナリ変数で書いた問題では、`BinaryModel.from_hubo`が同じ役割を果たします。

# %%
ising_quad: dict[tuple[int, int], float] = {
    tuple(sorted((i, j))): 1.0 for i, j in G.edges()
}
ising_linear: dict[int, float] = {}

spin_model = BinaryModel.from_ising(
    {**ising_quad, **{(i,): value for i, value in ising_linear.items()}},
    variables=range(num_nodes),
)
# 係数をモデルから、モデル自身の順序で読み戻します。
ising_quad = {
    indices: coefficient
    for indices, coefficient in spin_model.coefficients.items()
    if len(indices) == 2
}
ising_linear = {
    indices[0]: coefficient
    for indices, coefficient in spin_model.coefficients.items()
    if len(indices) == 1
}

print(f"Variable type:          {spin_model.vartype}")
print(f"Linear terms (h_i):     {ising_linear}")
print(f"Quadratic terms (J_ij): {ising_quad}")
print(f"Constant:               {spin_model.constant}")
# スピン変数のモデルで、線形項はなく、グラフの辺ごとに1つのJ_ijがあり、定数項はありません。
assert spin_model.vartype.name == "SPIN"
assert ising_linear == {}
assert len(ising_quad) == G.number_of_edges()
assert spin_model.constant == 0.0
# モデルは、バイト列に保存して読み戻しても変わりません。
assert BinaryModel.from_bytes(spin_model.to_bytes()) == spin_model

# %% [markdown]
# ### 厳密解（全探索）
#
# QAOAを実行する前に、$2^n = 32$通りのスピンの割り当てをすべて試して、厳密な答えを求めます。これが、あとでQAOAの結果と比べる参照値になります。

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
print(f"Number of optimal assignments: {len(optimal_partitions)}")
for part in optimal_partitions:
    print(f"  {part}")

# 最適な割り当てのイジングエネルギーは|E| - 2Cです。
optimal_energy = spin_model.energy(optimal_partitions[0])
print(f"Ising energy of the optimal assignments: {optimal_energy}")

# このグラフでは、カットの最大の大きさは5です。2つのスピンの割り当てがこれを達成し、
# 一方は他方のすべてのスピンを反転したものです。
assert best_cut == 5
assert len(optimal_partitions) == 2
for part in optimal_partitions:
    assert tuple(-s for s in part) in optimal_partitions
assert optimal_energy == -4
assert all(
    spin_model.energy(part) == G.number_of_edges() - 2 * best_cut
    for part in optimal_partitions
)

# %% [markdown]
# カットの最大の大きさは5で、辺の数より1つ少なく、対応するイジングエネルギーは$H_C = 6 - 2 \cdot 5 = -4$です。6本の辺すべてがカットをまたぐようなカットはありません。三角形では、3つの頂点のうち少なくとも2つが同じグループに入るので、どちらの三角形でも少なくとも1本の辺はカットをまたぎません。2つの三角形は辺$(1, 2)$を共有しています。そこで、頂点1と2を一方のグループに、頂点0と3をもう一方のグループに入れ、頂点4を頂点3と反対のグループに入れると、残りの辺はすべてカットをまたぎます。2つの最適な割り当ては、この同じカットを表しています。すべてのスピンを反転しても2つのグループの名前が入れ替わるだけなので、どちらも厳密解として数えます。

# %% [markdown]
# (qaoa-maxcut-algorithm)=
# ## アルゴリズム
#
# 量子近似最適化アルゴリズム（Quantum Approximate Optimization Algorithm、QAOA）は、[Farhi, Goldstone and Gutmann, arXiv:1411.4028](https://doi.org/10.48550/arXiv.1411.4028)で提案されました。QAOAは、$H_C$と、1層あたり2個、合わせて$2p$個の角度から作った$p$層の回路で、量子状態を準備します。この状態を測定し、測定で大きなカットが得られやすくなるように、古典オプティマイザが角度を調整します。この節では、この状態、それがゲートになる仕組み、角度の調整の仕方、答えの読み出し方を説明します。
#
# ### QAOA状態
#
# QAOAは、頂点ごとに1個、合わせて$n$個の量子ビットの回路で、$H_C$のエネルギーが最も低い状態を探します。まず、各スピン$s_i$を量子ビット$i$のPauli演算子$Z_i$に置き換えて、$H_C$を演算子にします。
#
# $$
# H_C = \sum_{(i,j) \in E} Z_i Z_j .
# $$
#
# $Z|0\rangle = |0\rangle$、$Z|1\rangle = -|1\rangle$なので、どのビット列$z$も$H_C$の固有状態で、その固有値は、$s_i = 1 - 2z_i$としたスピンの割り当てのエネルギーです。このエネルギーを$E(z)$と書きます。つまり$H_C |z\rangle = E(z)\, |z\rangle$です。ビット0がスピン$+1$、ビット1がスピン$-1$を表します。これは、[問題の定義](#qaoa-maxcut-problem-definition)の$s_i = 1 - 2x_i$と同じ規則です。
#
# 量子ビットは、すべてのビット列を重ね合わせとして同時に保持できます。
#
# $$
# |\psi\rangle = \sum_{z} a_z\, |z\rangle ,
# $$
#
# これを測定すると、ビット列$z$が確率$|a_z|^2$で得られます。目的は、エネルギーの低いビット列、つまり大きなカットの確率が大きい状態を作ることです。そうすれば、それほど多くない測定回数で、よいカットが得られる見込みが高くなります。
#
# **初期状態:** 回路は一様重ね合わせ$|+\rangle^{\otimes n}$から始まります。この状態では、どのビット列も同じ確率を持ちます。各量子ビットにアダマールゲートを適用すると準備できます。
#
# **コストユニタリ:** 角度$\gamma$に対するコストユニタリは$U_C(\gamma) = e^{-i\gamma H_C}$です。各ビット列に、そのエネルギーで決まる位相を与えます。
#
# $$
# U_C(\gamma) |z\rangle = e^{-i\gamma E(z)}\, |z\rangle .
# $$
#
# 位相だけでは測定確率は変わりません。各ビット列のエネルギーを状態に記録するだけです。
#
# **ミキサーユニタリ:** 角度$\beta$に対するミキサーユニタリは次の式です。
#
# $$
# U_M(\beta) = e^{-i\beta \sum_i X_i}
# = \prod_i \bigl(\cos\beta\; I - i \sin\beta\; X_i\bigr) ,
# $$
#
# ここで$I$は恒等演算子で、Pauli演算子$X_i$はビット$i$を反転します。ビット列$y$のビット$i$を反転したものを$y^{(i)}$と書きます。量子ビット$i$の因子は、$|y\rangle$を$|y\rangle$と$|y^{(i)}\rangle$の重ね合わせに変えます。
#
# $$
# \bigl(\cos\beta\; I - i \sin\beta\; X_i\bigr)\, |y\rangle
# = \cos\beta\, |y\rangle - i \sin\beta\, |y^{(i)}\rangle .
# $$
#
# したがって、1つの因子は、1ビットだけ異なる2つのビット列の間で振幅をやり取りします。$\sin\beta$も$\cos\beta$も0でなければ、$n$個の因子の積は、どの2つのビット列の間もつなぎます。同じビット列に届く振幅は、位相が近ければ強め合い、位相が逆なら打ち消し合います。この干渉が確率を変えます。
#
# **層:** コストユニタリに続けてミキサーを適用したものを1つの*層*と呼びます。$p$層で、角度を$\boldsymbol{\gamma} = (\gamma_1, \dots, \gamma_p)$と$\boldsymbol{\beta} = (\beta_1, \dots, \beta_p)$とすると、QAOA状態は次のようになります。
#
# $$
# |\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle
# = U_M(\beta_p)\, U_C(\gamma_p) \cdots U_M(\beta_1)\, U_C(\gamma_1)\,
# |+\rangle^{\otimes n} ,
# $$
#
# この式は右から左へ読みます。回路の構造はグラフが$H_C$を通して決めていて、自由なパラメータは$2p$個の角度だけです。
#
# 適切な角度では、この干渉によってエネルギーの低いビット列の確率を高められます。[補遺](#qaoa-maxcut-small-angle-analysis)では、角度が小さい1層の場合についてこの効果を導き、角度の符号の役割を説明します。
#
# (qaoa-maxcut-unitaries-to-gates)=
# ### ユニタリからゲートへ
#
# 一般のイジングモデルでは、コストハミルトニアンは$H_C = \sum_i h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j$です。どの項も$Z$演算子だけでできているので、項はすべて可換です。したがって、コストユニタリは、項ごとの因子に厳密に分解できます。
#
# $$
# U_C(\gamma) = \prod_i e^{-i\gamma h_i Z_i}
# \prod_{i<j} e^{-i\gamma J_{ij} Z_i Z_j} ,
# $$
#
# ミキサーも、量子ビットごとの因子に分解できます。Qamomileの回転ゲートは、指数に因子$1/2$を含む形で定義されています。
#
# $$
# R_Z(\theta) = e^{-i\theta Z/2}, \qquad
# R_{ZZ}(\theta) = e^{-i\theta Z \otimes Z/2}, \qquad
# R_X(\theta) = e^{-i\theta X/2},
# $$
#
# そのため、それぞれの因子は、角度を2倍にした1つのゲートになります。
#
# $$
# e^{-i\gamma J_{ij} Z_i Z_j} = R_{ZZ}(2\gamma J_{ij}), \qquad
# e^{-i\gamma h_i Z_i} = R_Z(2\gamma h_i), \qquad
# e^{-i\beta X_i} = R_X(2\beta).
# $$
#
# このグラフでは、どの辺でも$J_{ij} = 1$で、$h_i = 0$です。したがって、1つの層には6個の$R_{ZZ}$ゲートと5個の$R_X$ゲートがあります。5個の$R_Z$ゲートは角度が0で、ここでは何もしません。それでも残しているのは、線形項のあるイジングモデルでも同じコードが使えるようにするためです。重み付きのMaxCutでも、変更は要りません。重みは$J_{ij}$として入るからです。
#
# 測定は、上の規則$s_i = 1 - 2z_i$に従って、スピン$+1$にはビット0を、スピン$-1$にはビット1を返します。`BinaryModel.decode`は、測定されたビット列をスピンの割り当てに戻すときに、この規則を使います。
#
# ### 角度の最適化
#
# QAOA状態のエネルギーの期待値が最小になるように、角度を選びます。
#
# $$
# F_p(\boldsymbol{\gamma}, \boldsymbol{\beta})
# = \langle \boldsymbol{\gamma}, \boldsymbol{\beta} |\, H_C \,|
# \boldsymbol{\gamma}, \boldsymbol{\beta} \rangle
# = \sum_z \bigl|\langle z | \boldsymbol{\gamma}, \boldsymbol{\beta}
# \rangle\bigr|^2\, E(z) .
# $$
#
# $C = |E|/2 - H_C/2$なので、これはカットの大きさの期待値$|E|/2 - F_p/2$を最大にすることと同じです。同じ角度で$N$回測定し、$F_p$を推定します。各測定で得られるビット列$z^{(k)}$のエネルギーを計算し、その平均を取ります。
#
# $$
# F_p(\boldsymbol{\gamma}, \boldsymbol{\beta})
# \approx \frac{1}{N} \sum_{k=1}^{N} E\bigl(z^{(k)}\bigr) .
# $$
#
# 古典オプティマイザは角度を選び、量子回路からこの平均値を受け取り、次の角度を選びます。評価のたびに新しい$N$ショットを使うため、全体のショット数は$N$と評価回数の積です。$N$を増やすと、各推定値の統計的な不確かさは小さくなります。
#
# ここでは、勾配を必要としないSciPyのConstrained Optimization BY Linear Approximation（COBYLA）法を使います。サンプルから推定したエネルギーで局所探索するため、大域的な最小値は保証されません。結果は、層の数、初期角度、オプティマイザ、ショット数に依存します。最適化のループは[結果](#qaoa-maxcut-result)の節で示します。
#
# **初期角度:** MaxCutでは、ランダムな初期角度からの最適化が、質の悪い局所解に収束する場合があります（[Sack and Serbyn, Quantum 5, 491 (2021)](https://doi.org/10.22331/q-2021-07-01-491)）。ここでは、量子アニーリングを参考に初期角度を選びます。量子アニーリングでは、ハミルトニアン
#
# $$
# H(s) = -(1 - s) \sum_i X_i + s\, H_C
# $$
#
# を、$s = 0$から$s = 1$へゆっくり変化させます。エネルギーが最も低い状態を基底状態と呼びます。初めの基底状態は一様重ね合わせで、最後の基底状態は最大カットを表します。変化が十分にゆっくりで、関係するエネルギーギャップが開いたままなら、基底状態をたどれるという考え方は断熱定理に基づきます（[Albash and Lidar, Reviews of Modern Physics 90, 015002 (2018)](https://doi.org/10.1103/RevModPhys.90.015002)）。ここでは、すべてのスピンを反転してもハミルトニアンと初期状態が変わらないため、関係するのは同じ対称性を持つほかの状態とのギャップです。この条件は検証せず、アニーリングは初期角度を選ぶ指針としてだけ使います。
#
# 全体の時間を$T$とし、時間発展を長さ$\Delta t = T/p$の$p$ステップに分けます。各ステップ内の$s = t/T$を、その中央での値で近似します。
#
# $$
# s_k = \frac{k - 1/2}{p} ,
# \qquad
# k = 1, \dots, p .
# $$
#
# 時間発展を、コスト層に続けてミキサー層を適用する形に分けると、
#
# $$
# e^{-i \Delta t\, H(s_k)}
# \approx e^{\,i \Delta t\, (1 - s_k) \sum_i X_i}\; e^{-i \Delta t\, s_k H_C}
# = U_M(\beta_k)\, U_C(\gamma_k) ,
# $$
#
# となり、初期角度は次のように得られます。
#
# $$
# \gamma_k = \Delta t\, s_k ,
# \qquad
# \beta_k = -\Delta t\, (1 - s_k) .
# $$
#
# $\beta_k$の負号は、$H(s)$の$\sum_i X_i$の前の負号から来ています。これらの角度は層に沿って線形に変化するため、この選び方を*線形ランプ*と呼びます。$p = 3$ステップだけでは、変化が十分にゆっくりであることも、分割の近似が十分に正確であることも確かめられていません。ランプは初期角度を与えるもので、最大カットを保証するものではありません。
#
# ここでは$\Delta t = 0.75$を使います。これは、各頂点が3つの頂点と隣り合う3-正則グラフのMaxCutについて、Sack and Serbynが見つけた時間ステップです。論文のコストハミルトニアンは同じですが、ミキサーには$-\sum_i X_i$を使うため、論文の$\beta$はここでの$-\beta$に対応します。論文の$s_k = k/p$に対し、ここでは各ステップの中央の値を使います。このグラフは3-正則ではなく、この時間ステップはこのグラフに合わせて調整したものではありません。$p = 3$のとき、初期角度は$\boldsymbol{\gamma} = (0.125, 0.375, 0.625)$と$\boldsymbol{\beta} = (-0.625, -0.375, -0.125)$です。
#
# ### カットの読み出し
#
# COBYLAは、評価した角度の中で、サンプルから推定したエネルギーが最も小さかった角度を返します。その角度で回路をもう一度実行して、カットを得ます。最適化中のサンプルは$F_p$の推定にだけ使い、カットの分析にはこの最後のバッチを使います。
#
# 測定されたビット列$z$は、$s_i = 1 - 2z_i$によって有効なカットを表します。すべてのビットを反転すると2つのグループが入れ替わるだけなので、ビット列とその反転はまとめて数えます。各カットが何回現れたかを数えると、測定された頻度が得られます。
#
# 測定された各カットについて、カットをまたぐ辺を数え、最大のものを答えとします。これは、[Farhi, Goldstone and Gutmann, arXiv:1411.4028](https://doi.org/10.48550/arXiv.1411.4028)の第I節にある、測定とコスト評価を繰り返す手順に沿っています。測定された最大のカットはサンプルの平均以上ですが、グラフの最大カットであるとは限りません。[結果](#qaoa-maxcut-result)の節では厳密解と比較し、最大カットが最も多く測定されたかどうかも確認します。

# %% [markdown]
# ## Qamomileによる実装
#
# ここからは、[アルゴリズム](#qaoa-maxcut-algorithm)の節の回路を、`@qkernel`関数として1ステップずつ書いていきます。一様重ね合わせ、コスト層、ミキサー層、そして$p$個の層と最後の測定からなる回路全体の順です。入力には2種類あります。これは[パラメータ付き量子カーネル](../tutorial/02_parameterized_kernels.ipynb)で説明されています。量子ビットの数やグラフの辺のように`qm.Const`の付いた入力は、カーネルを特殊化するときに固定されます。`qm.Param`の付いた入力、ここでは角度$\boldsymbol{\gamma}$と$\boldsymbol{\beta}$は、自由なまま残り、実行のたびに与えます。

# %% [markdown]
# ### ステップ1: 一様重ね合わせ状態の準備
#
# `qm.qubits(size)`は、$|0\rangle$の量子ビットを`size`個作ります。各量子ビットにアダマールゲートを適用すると、[アルゴリズム](#qaoa-maxcut-algorithm)の節の初期状態である一様重ね合わせ$|{+}\rangle^{\otimes n}$になります。ゲートは、受け取った量子状態を消費し、更新した状態を返します。そのため、ゲートの呼び出しは`q = qm.h(q, target=i)`のように書きます（[はじめての量子カーネル](../tutorial/01_your_first_quantum_kernel.ipynb)を参照）。アノテーション`qm.Q[qm.Bits[size]]`の`size`は、このカーネル自身の入力です。したがって、このアノテーションは、カーネルが`size`個の量子ビットを返すことを表します。ページ冒頭のimportのセルは、`__future__`から`annotations`をimportしています。これによりPythonは、このようなアノテーションを自分では評価せず、Qamomileに任せます。


# %%
@qm.qkernel
def superposition(size: qm.Const[qm.Int]) -> qm.Q[qm.Bits[size]]:
    q = qm.qubits(size)
    for i in range(size):
        q = qm.h(q, target=i)
    return q


# %% [markdown]
# (qaoa-maxcut-step-2)=
# ### ステップ2: コスト層
#
# コスト層は、上の[ユニタリからゲートへ](#qaoa-maxcut-unitaries-to-gates)で導いたように、$U_C(\gamma)$を、辺ごとに1つの$R_{ZZ}(2\gamma J_{ij})$ゲートと、頂点ごとに1つの$R_Z(2\gamma h_i)$ゲートとして適用します。カーネルは、イジングモデルを3つの入力から読み取ります。`edges`は`edge_count` × 2の配列で、その行$k$が辺$k$の2つの頂点を持ちます。`quad[k]`はその辺の結合$J_{ij}$を、`linear[i]`は$h_i$を持ちます。このグラフでは`linear`はすべて0で、$R_Z$ゲートを残している理由は「ユニタリからゲートへ」で説明しています。引数`gamma`は1つの層の角度で、[ステップ4](#qaoa-maxcut-step-4)が`gammas`から取り出して渡します。`gamma`には`qm.Const`も`qm.Param`も付いていません。カーネルは、呼び出し側が渡すどちらの種類の値も受け取るからです。


# %%
@qm.qkernel
def cost_layer(
    q: qm.Q[qm.Bits[size]],
    size: qm.Const[qm.Int],
    edge_count: qm.Const[qm.Int],
    edges: qm.Const[qm.Matrix[qm.Int, edge_count, 2]],
    quad: qm.Const[qm.Vector[qm.Real, edge_count]],
    linear: qm.Const[qm.Vector[qm.Real, size]],
    gamma: qm.Real,
) -> qm.Q[qm.Bits[size]]:
    for edge in range(edge_count):
        q = qm.rzz(
            q, 2.0 * quad[edge] * gamma, left=edges[edge][0], right=edges[edge][1]
        )
    for i in range(size):
        q = qm.rz(q, 2.0 * linear[i] * gamma, target=i)
    return q


# %% [markdown]
# ### ステップ3: ミキサー層
#
# ミキサー層は、上の[ユニタリからゲートへ](#qaoa-maxcut-unitaries-to-gates)で導いたように、$U_M(\beta)$を、量子ビットごとに1つの$R_X(2\beta)$ゲートとして適用します。[ステップ2](#qaoa-maxcut-step-2)の`gamma`と同じく、引数`beta`は1つの層の角度で、[ステップ4](#qaoa-maxcut-step-4)が`betas`から取り出して渡します。


# %%
@qm.qkernel
def mixer_layer(
    q: qm.Q[qm.Bits[size]],
    size: qm.Const[qm.Int],
    beta: qm.Real,
) -> qm.Q[qm.Bits[size]]:
    for i in range(size):
        q = qm.rx(q, 2.0 * beta, target=i)
    return q


# %% [markdown]
# (qaoa-maxcut-step-4)=
# ### ステップ4: QAOAアンザッツを作る
#
# `qaoa_ansatz`は、3つのステップをまとめて、[アルゴリズム](#qaoa-maxcut-algorithm)の節のQAOA状態を作り、最後に測定します。アンザッツとは、自由なパラメータを持つ回路のことで、ここでのパラメータは角度です。このカーネルは、一様重ね合わせを準備し、続いて$p$個の層を適用します。コードでは$p$は`layers`です。ループは`layer` $= 0, \dots, p-1$について回るので、`gammas[layer]`と`betas[layer]`は、$k =$ `layer` $+ 1$とした$\gamma_k$と$\beta_k$です。最後に、すべての量子ビットを測定し、ビット列を返します。
#
# 次に`qm.specialize`が、`qm.Const`の入力を固定します。層の数$p = 3$、量子ビットの数、そして、モデルからモデル自身の順序で読み取ったイジングモデルの辺と係数です。角度は自由なまま残ります。


# %%
@qm.qkernel
def qaoa_ansatz(
    layers: qm.Const[qm.Int],
    size: qm.Const[qm.Int],
    edge_count: qm.Const[qm.Int],
    edges: qm.Const[qm.Matrix[qm.Int, edge_count, 2]],
    quad: qm.Const[qm.Vector[qm.Real, edge_count]],
    linear: qm.Const[qm.Vector[qm.Real, size]],
    gammas: qm.Param[qm.Vector[qm.Real, layers]],
    betas: qm.Param[qm.Vector[qm.Real, layers]],
) -> qm.Shot[qm.Bits[size]]:
    q = superposition(size)
    for layer in range(layers):
        q = cost_layer(q, size, edge_count, edges, quad, linear, gammas[layer])
        q = mixer_layer(q, size, betas[layer])
    return qm.measure(q)


p = 3  # QAOAの層の数
edge_indices = tuple(ising_quad)
quad_values = tuple(ising_quad[edge] for edge in edge_indices)
linear_values = tuple(ising_linear.get(i, 0.0) for i in range(num_nodes))
program = qm.specialize(
    qaoa_ansatz,
    layers=p,
    size=num_nodes,
    edge_count=len(edge_indices),
    edges=edge_indices,
    quad=quad_values,
    linear=linear_values,
)


# %% [markdown]
# ### 回路のコンパイル
#
# `qm.compile`は、特殊化したプログラムを、それを実行するエンジン用の回路に変換します。ここではQamomile内蔵のシミュレータ`qm.Reference()`を使います。コンパイル済みの回路でも、角度`gammas`と`betas`は自由なままです。そのため、オプティマイザは、`qm.compile`をもう一度呼ぶことなく、毎回新しい角度で回路を実行できます。測定結果を再現できるように、毎回の実行で同じ乱数シード（乱数の初期値）を`qm.run`に渡します。

# %%
SEED = 42
executor = qm.Reference()
executable = qm.compile(program, to=executor)
# 同じコンパイル済みオブジェクトがあとでも使われていることを確かめるために残します。
artifact_identity = id(executable)
# コンパイル済みの回路の自由なパラメータは、2つの角度のベクトルです。
assert tuple(slot.name for slot in executable.parameter_abi.slots) == (
    "gammas",
    "betas",
)

# %% [markdown]
# 次のセルは、特殊化したプログラムのゲートを数え、[アルゴリズム](#qaoa-maxcut-algorithm)の節と照らし合わせます。アダマールゲートが5個、そして3つの層のそれぞれに、$R_{ZZ}$ゲートが6個、$R_Z$ゲートが5個、$R_X$ゲートが5個です。

# %%
study = qm.resources.logical(program)
expected_counts = {
    "quantum.h": num_nodes,
    "quantum.rzz": p * len(ising_quad),
    "quantum.rz": p * num_nodes,
    "quantum.rx": p * num_nodes,
}
# numeric_enclosure()は、数の下限と上限を返します。ここでは
# 2つが等しいので、数は厳密です。
for name, expected in expected_counts.items():
    count = study.metric(qm.resources.logical_operation_count(name)).numeric_enclosure()
    print(f"{name}: {count[0]}")
    assert count == (expected, expected)
# リソースの見積もりは、バイト列に保存して読み戻しても変わりません。
assert qm.resources.ResourceStudy.from_bytes(study.to_bytes()) == study

# %% [markdown]
# (qaoa-maxcut-result)=
# ## 結果
#
# ここからは、コンパイル済みの回路で[アルゴリズム](#qaoa-maxcut-algorithm)の節の最適化のループを実行し、そのあとカットを読み出します。
#
# ### 角度を最適化する
#
# `cost_fn`は、COBYLAが評価する関数です。呼び出されるたびに、与えられた角度で、コンパイル済みの回路を`sample_shots`ショット実行します。次に`spin_model.decode`が、測定されたビット列をそれぞれ$s_i = 1 - 2z_i$でスピンの割り当てに直し、そのエネルギーを計算します。`cost_fn`は、ショットについてのこれらのエネルギーの平均を返します。これが$F_p$の推定値です。`cost_fn`はこの値を`cost_history`にも追加するので、このリストには評価1回につき1つの値が入ります。`scipy.optimize.minimize`は、`cost_fn`と初期角度を受け取り、COBYLAが値を必要とするたびに`cost_fn`を呼びます。
#
# 初期角度は、[アルゴリズム](#qaoa-maxcut-algorithm)の節の線形ランプで、$\Delta t = 0.75$です。配列`initial_params`には、まず$\gamma$の$p$個の値が、続いて$\beta$の$p$個の値が入ります。これは、`cost_fn`が配列を分けるときの順序です。
#
# 実行のあと、セルは、評価の回数とそれに使ったショット数、COBYLAが止まった理由、$F_p$の推定値の最小値、その値が得られた角度を表示します。2つ目のセルは、評価ごとの推定値をプロットします。

# %%
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
sample_shots = 256 if docs_test_mode else 2048
maxiter = 20 if docs_test_mode else 500
cost_history: list[float] = []


def cost_fn(params):
    gammas = tuple(float(v) for v in params[:p])
    betas = tuple(float(v) for v in params[p:])
    result = qm.run(
        executable,
        on=executor,
        shots=sample_shots,
        seed=SEED,
        parameters={"gammas": gammas, "betas": betas},
    ).result
    decoded = spin_model.decode(result)
    energy = decoded.mean_energy
    cost_history.append(energy)
    return energy


# 線形ランプ: gamma_k = delta_t * s_k、beta_k = -delta_t * (1 - s_k)。
# ここでs_k = (k - 1/2) / p、k = 1, ..., pです。
delta_t = 0.75
ramp = (np.arange(1, p + 1) - 0.5) / p
initial_params = np.concatenate([delta_t * ramp, -delta_t * (1 - ramp)])
assert initial_params.shape == (2 * p,)

res = minimize(cost_fn, initial_params, method="COBYLA", options={"maxiter": maxiter})

print(f"Evaluations: {res.nfev}, shots used: {res.nfev * sample_shots}")
print(f"Why COBYLA stopped: {res.message}")
print(f"Smallest estimate of F_p: {res.fun:.4f}")
print(f"Returned angles: {[round(float(v), 4) for v in res.x]}")
assert len(cost_history) == res.nfev
assert len(res.x) == 2 * p
# COBYLAは、すべての評価の中で推定値が最も小さかった角度を返します。
assert res.fun == min(cost_history)
if docs_test_mode:
    assert res.nfev == maxiter

# %%
plt.figure(figsize=(8, 4))
plt.plot(cost_history, color="#2696EB")
plt.xlabel("Evaluation")
plt.ylabel("Estimate of $F_p$ (mean energy)")
plt.title("QAOA Optimization Progress")
plt.show()

# %% [markdown]
# プロットの各点は、目的関数を1回評価して得た平均エネルギーの推定値です。出力メッセージが示すとおり、この実行では、探索ステップの大きさを制限する信頼領域の半径が下限に達して停止しました。

# %% [markdown]
# ### サンプリングと評価
#
# 返された角度で、コンパイル済みの回路をもう一度実行し、この最後のバッチをデコードします。最適化中と同じシードを渡すので、このバッチは、推定値が最も小さかった評価のときとまったく同じビット列からなります。そのため、エネルギーの平均はその推定値と一致します。量子デバイスでは、新しいバッチになります。
#
# 2つ目のセルは、カットごとにサンプルを数えます。`decoded.samples`には、異なるスピンの割り当てが1つずつ、その出現回数とともに並んでいます。ある割り当てと、それを反転した割り当ては、同じカットです。これらをまとめて数えるために、それぞれのカットを、2つの割り当てのうち頂点0のスピンが$+1$であるほうで表します。頂点0のスピンが$-1$の割り当ては、先に反転します。そのうえで、出現回数を足し合わせます。測定されたすべてのカットの大きさを計算し、最大のものをQAOAの答えとして、厳密解と比べます。

# %%
final_gammas = tuple(float(v) for v in res.x[:p])
final_betas = tuple(float(v) for v in res.x[p:])

final_result = qm.run(
    executable,
    on=executor,
    shots=sample_shots,
    seed=SEED,
    parameters={"gammas": final_gammas, "betas": final_betas},
).result

decoded = spin_model.decode(final_result)
print(f"Mean energy of the final batch: {decoded.mean_energy:.4f}")
# 同じコンパイル済みオブジェクトが使われていて、固定したシードが、この角度での
# 評価のサンプルを再現します。
assert id(executable) == artifact_identity
assert decoded.mean_energy == res.fun


# %%
def cut_size(spins):
    return sum(1 for i, j in G.edges() if spins[i] != spins[j])


# カットごとにサンプルを数えます。スピンの割り当てと、それを反転した割り当ては
# 同じカットで、ここでは頂点0のスピンが+1であるほうで表します。
cut_counts: Counter[tuple[int, ...]] = Counter()
for sample in decoded.samples:
    assignment = dict(sample.assignment)  # {頂点: スピン}
    spins = tuple(assignment[i] for i in range(num_nodes))
    cut_counts[tuple(s * spins[0] for s in spins)] += sample.occurrences

best_qaoa_sample = max(cut_counts, key=cut_size)
best_qaoa_cut = cut_size(best_qaoa_sample)
# 厳密な最大カットのすべて。頂点0のスピンを+1にして書きます。
exact_cuts = {tuple(s * part[0] for s in part) for part in optimal_partitions}

print(f"Distinct cuts measured: {len(cut_counts)} of {2 ** (num_nodes - 1)}")
print(f"Largest measured cut size: {best_qaoa_cut}  (exact: {best_cut})")
print(f"Its spin assignment: {best_qaoa_sample}")
# すべてのショットが数えられていて、答えは厳密な最大カットです。
assert sum(cut_counts.values()) == sample_shots
assert best_qaoa_cut == best_cut
assert best_qaoa_sample in exact_cuts

# %% [markdown]
# ### カットの分布
#
# 答えだけでは、回路がサンプルをカットの間にどのように分布させているかはわかりません。このグラフには、測定されたカットごとに棒が1本あり、頻度の高い順に並んでいて、それぞれの棒の上にカットの大きさが書かれています。それぞれのカットのラベルは、頂点0のグループに入っていない頂点です。たとえば`{1,2,4}`は、頂点1、2、4を頂点0と3から分けるカットです。ラベル`{}`は、すべての頂点が1つのグループに入る分け方で、その大きさは0です。
#
# セルは、最も頻度の高いカットをそのショット数とともに表示し、それが厳密な最大カットであることを確かめます。つまり最大カットは、サンプルの中に見つかるだけではありません。このバッチで最も多く測定されたカットでもあります。

# %%
# 測定されたカットごとに棒が1本で、頻度の高い順です。カットのラベルは、
# 頂点0のグループに入っていない頂点です。
ranked = cut_counts.most_common()
labels = [
    "{" + ",".join(str(i) for i in range(num_nodes) if cut[i] == -1) + "}"
    for cut, _ in ranked
]
most_frequent_cut, most_frequent_count = ranked[0]
print(
    f"Most frequent cut: {labels[0]}, size {cut_size(most_frequent_cut)}, "
    f"{most_frequent_count} of {sample_shots} shots"
)
assert most_frequent_cut in exact_cuts

plt.figure(figsize=(9, 4))
bars = plt.bar(labels, [count for _, count in ranked], color="#2696EB")
for bar, (cut, _) in zip(bars, ranked):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(cut_size(cut)),
        ha="center",
        va="bottom",
    )
plt.xlabel("Cut, labeled by the vertices outside the group of vertex 0")
plt.ylabel("Number of shots")
plt.title("Measured cuts, with the cut size above each bar")
plt.xticks(rotation=45)
plt.margins(y=0.12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 最良のカットの可視化
#
# 最後に、答えをグラフの上に描きます。頂点の2つの色が、2つのグループを表します。実線の辺はカットをまたぎ、破線の辺は両端が同じグループにあります。タイトルにはカットの大きさが表示されます。これは実線の辺の数です。

# %%
color_map = ["#FF6B6B" if spin == +1 else "#4ECDC4" for spin in best_qaoa_sample]
crossing = [(i, j) for i, j in G.edges() if best_qaoa_sample[i] != best_qaoa_sample[j]]
not_crossing = [edge for edge in G.edges() if edge not in crossing]
# カットの大きさは、カットをまたぐ辺の数です。
assert len(crossing) == best_qaoa_cut

plt.figure(figsize=(5, 4))
# 色は頂点のラベルの順に並んでいるので、頂点もその順に並べます。
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=list(range(num_nodes)),
    node_color=color_map,
    node_size=700,
    edgecolors="black",
)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos, edgelist=crossing, width=2)
nx.draw_networkx_edges(G, pos, edgelist=not_crossing, style="dashed", edge_color="gray")
plt.title(f"Largest measured cut (size {best_qaoa_cut})")
plt.axis("off")
plt.show()

# %% [markdown]
# ## Qamomileの組み込み機能
#
# ### `qaoa_state`を使う
#
# ステップ1から4では、QAOA状態を手で組み立てました。`qamomile.stdlib.qaoa_state`は、これを1回の呼び出しで作ります。量子ビットの数、層の数、コストハミルトニアン、角度を受け取り、QAOA状態にある量子ビットを返します。ハミルトニアンはモデルから`spin_model.hamiltonian`として得られるので、辺や係数を1つずつ渡す必要はなくなります。


# %%
@qm.qkernel
def qaoa_builtin(
    layers: qm.Const[qm.Int],
    size: qm.Const[qm.Int],
    hamiltonian: qm.Const[qm.Observable],
    gammas: qm.Param[qm.Vector[qm.Real, layers]],
    betas: qm.Param[qm.Vector[qm.Real, layers]],
) -> qm.Shot[qm.Bits[size]]:
    q = qaoa_state(size, layers, hamiltonian, gammas, betas)
    return qm.measure(q)


builtin_program = qm.specialize(
    qaoa_builtin,
    layers=p,
    size=num_nodes,
    hamiltonian=spin_model.hamiltonian,
)

# %% [markdown]
# このカーネルをコンパイルし、2つの回路を、[結果](#qaoa-maxcut-result)の節の最後の角度で実行します。手書きと組み込みの両方を、同じ`Reference`で同じシードを渡して実行します。組み込みのプログラムには明示的な$R_Z$ゲートがありません。このハミルトニアンには線形項がないからです。一方、ここで書いた`cost_layer`は、角度0の$R_Z$ゲートを15個残しています。角度0の回転は何もしないので、これらのゲートを省いても状態は変わりません。同じシードのとき、このシミュレータは2つの回路に対してまったく同じカウントを返します。セルはこれを確かめていて、したがって2つのエネルギーの平均も等しくなります。

# %%
exe_builtin = qm.compile(builtin_program, to=executor)

result_manual = qm.run(
    executable,
    on=executor,
    shots=sample_shots,
    seed=SEED,
    parameters={"gammas": final_gammas, "betas": final_betas},
).result
result_builtin = qm.run(
    exe_builtin,
    on=executor,
    shots=sample_shots,
    seed=SEED,
    parameters={"gammas": final_gammas, "betas": final_betas},
).result

decoded_manual = spin_model.decode(result_manual)
decoded_builtin = spin_model.decode(result_builtin)
print(f"Hand-built mean energy: {decoded_manual.mean_energy:.4f}")
print(f"Built-in mean energy:   {decoded_builtin.mean_energy:.4f}")
# 組み込みのプログラムには明示的なR_Zゲートがありません。
builtin_study = qm.resources.logical(builtin_program)
rz_count = qm.resources.logical_operation_count("quantum.rz")
assert rz_count.identifier not in builtin_study.values
# 同じシードのとき、ここでは2つの回路が同じサンプルを与えるので、
# 平均も同じになります。
assert result_manual.counts == result_builtin.counts
assert decoded_manual.mean_energy == decoded_builtin.mean_energy

# %% [markdown]
# ## まとめ
#
# このチュートリアルでは、次のことを行いました。
#
# 1. 5頂点のグラフのMaxCutを、`BinaryModel.from_ising`でスピン変数のイジングモデルとして書き、すべての割り当てを試して、大きさ5の厳密な最大カットを求めました。
# 2. QAOA回路を`@qkernel`関数から組み立てました。一様重ね合わせ、コスト層、ミキサー層、そして3層の回路全体です。
# 3. 線形ランプから始めて、COBYLAで6個の角度を調整しました。評価のたびに、コンパイル済みの回路を新しい角度の組で実行し、そのサンプルのエネルギーの平均を返しました。
# 4. 返された角度で回路をサンプリングし、カットごとにサンプルを数えました。測定された最大のカットは厳密な最大カットで、最も多く測定されたカットでもありました。
# 5. `qamomile.stdlib.qaoa_state`を使い、1回の呼び出しで同じ状態を作り、同じ測定カウントを得ました。
#
# ステップ1から4のカーネルは、係数$h_i$と$J_{ij}$を入力として受け取ります。そのため、重み付きのMaxCutや、線形項のあるイジングモデルの回路も作れます。係数を`BinaryModel.from_ising`に渡し、同じカーネルを使ってください。ランプの時間ステップ$\Delta t$は、3-正則グラフのMaxCutから取ったもので、モデルによっては変える必要があるかもしれません。
#
# 制約のある問題をJijModelingで書き、より高水準な`QAOAExperiment`で解く例は、[QAOAによるグラフ分割](qaoa_graph_partition)を参照してください。

# %% [markdown]
# (qaoa-maxcut-small-angle-analysis)=
# ## 補遺: 小さい角度での1層の解析
#
# 角度が小さい1層の場合を考えます。コストユニタリのあと、どのビット列$z$の振幅も$e^{-i\gamma E(z)}/\sqrt{2^n}$です。$\beta$の1次までで、ミキサーは$I - i\beta \sum_i X_i$なので、$y$の振幅には、$n$個のビット列$y^{(i)}$の振幅が加わります。
#
# $$
# a_y \approx \frac{1}{\sqrt{2^n}}
# \Bigl(e^{-i\gamma E(y)} - i\beta \sum_i e^{-i\gamma E(y^{(i)})}\Bigr) .
# $$
#
# $y$と2ビット以上異なるビット列からも、$y$に振幅が届きます。ただし、反転するビット1つにつき因子$\sin\beta$が1つ掛かるので、これらの寄与は$\beta^2$のオーダーかそれ以下であり、ここでは落とします。
#
# 確率は、絶対値の2乗$|a_y|^2$です。この確率には、$\beta$だけの項も$\gamma$だけの項もありません。ミキサーだけでも、コストユニタリだけでも、一様な確率は変わらないからです。$y$に依存する最初の項は、積$\beta\gamma$に比例します。この項を残し、角度のより高いべきの項を落とすと、次のようになります。
#
# $$
# |a_y|^2 \approx \frac{1}{2^n}
# \Bigl(1 + 2\beta\gamma \sum_i \bigl(E(y) - E(y^{(i)})\bigr)\Bigr) .
# $$
#
# $y$の確率は、$y$のエネルギーがこれら$n$個のビット列のエネルギーと比べてどうであるかで決まります。スピン$i$を反転すると、$s_i$を含む項はすべて符号が変わり、それ以外の項は変わりません。したがって
#
# $$
# E(y) - E(y^{(i)}) = 2\, s_i \sum_{j \ne i} J_{ij}\, s_j ,
# $$
#
# です。ここで$s_i = 1 - 2y_i$は$y$のスピンで、$J_{ij} = J_{ji}$は、$i$と$j$が辺でつながっていれば1、そうでなければ0です。$i$について和を取ると、2重の和になります。
#
# $$
# \sum_i \bigl(E(y) - E(y^{(i)})\bigr)
# = 2 \sum_i \sum_{j \ne i} J_{ij}\, s_i s_j .
# $$
#
# この2重の和には、それぞれの組が$(i, j)$と$(j, i)$として2回現れます。一方、$E(y) = \sum_{i<j} J_{ij}\, s_i s_j$には1回しか現れません。したがって、2重の和は$2E(y)$に等しく、$i$についての和は$4E(y)$です。よって
#
# $$
# |a_y|^2 \approx \frac{1}{2^n}\bigl(1 + 8\beta\gamma\, E(y)\bigr) .
# $$
#
# となります。$\gamma > 0$、$\beta < 0$のとき、因子$\beta\gamma$は負なので、エネルギーの低いビット列、つまり大きなカットは、一様状態のときより得られやすくなります。$\beta > 0$では、同じ式がエネルギーの高いビット列を有利にするので、$\beta$の符号が重要です。
#
# 2つのユニタリはどちらも必要です。一様状態に対して、コストユニタリだけでは位相しか変わらず、ミキサーだけでは測定で見える変化は何も起きません。
