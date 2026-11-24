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
# # QAOAによるグラフ分割
#
# このチュートリアルでは、量子近似最適化アルゴリズム（Quantum Approximate Optimization Algorithm、QAOA）で**グラフ分割**問題を解きます。グラフの頂点を同じ大きさの2つのグループに分け、グループの間をつなぐ辺をできるだけ少なくする問題です。同じ大きさにするという条件は制約であり、標準のQAOA回路にはこれを直接課す方法がありません。そこで、制約をペナルティ項としてコストに加え、サンプルした解はすべて、あとで元の問題に照らして確認します。
#
# グラフ分割はMaxCutとは別の問題です。2つのグループの間の辺の数を、MaxCutは最大化しますが、グラフ分割は*最小化*します。さらに、2つのグループが同じ大きさであることを要求します。
#
# このチュートリアルは次の手順で進みます。
#
# 1. JijModelingで問題を書き、グラフのデータを入れて、具体的な問題インスタンスを作ります。
# 2. インスタンスを`OMMXProblem`に渡します。`OMMXProblem`は制約をペナルティ項に変換します。
# 3. Qamomileの`QAOAExperiment`で、ペナルティを加えた問題からQAOA回路を作り、コンパイルします。
# 4. 古典オプティマイザが角度を調整しながら、回路を繰り返し実行します。
# 5. オプティマイザが返した角度で、回路をもう一度測定し、すべてのサンプルを元の問題で評価します。
#
# **前提知識**: 量子ビット、量子ゲート、測定。未習の場合は[はじめての量子カーネル](../tutorial/01_your_first_quantum_kernel.ipynb)から始めてください。QAOAの知識は必要ありません。JijModelingは問題を書くためだけに使います。構文については[ドキュメント](https://jij-inc-jijmodeling-tutorials-ja.readthedocs-hosted.com/ja/latest/introduction.html)を参照してください。
#

# %% [markdown]
# ## インストール

# %%
# 最新のQamomile、JijModeling、NetworkXをpipからインストールします！
# # !pip install qamomile jijmodeling networkx

# %%
import itertools
import os

import jijmodeling as jm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import minimize

import qamomile as qm
from qamomile.optimization import BinaryModel, OMMXProblem, QAOAExperiment, VarType

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# ## 問題設定
#
# ### 問題の定義
#
# グラフ$G = (V, E)$は、頂点の集合$V$と辺の集合$E$からなります。辺$(u, v)$は2つの頂点$u$と$v$をつなぎます。ここで扱うグラフは無向です。$(u, v)$と$(v, u)$は同じ辺で、$E$についての和では1回だけ数えます。頂点の数$|V|$は偶数です。グラフは固定されていて、辺を足したり取り除いたりはしません。
#
# 選ぶのは、頂点をグループ0とグループ1の2つにどう分けるかだけです。このような分け方を*分割*と呼びます。各頂点$u$について、この選択をバイナリ変数$x_u$で表します。$u$がグループ0なら$x_u = 0$、グループ1なら$x_u = 1$です。すべての変数の値を決めたものを*割り当て*$\boldsymbol{x}$と呼びます。辺の両端が異なるグループにあるとき、その辺は分割を*またぐ*といいます。分割をまたぐ辺の数を、その分割の*カットの大きさ*と呼びます。目標は、次の2つを満たすようにグループを選ぶことです。
#
# - どちらのグループにも、ちょうど$|V|/2$個の頂点が含まれる。
# - カットの大きさができるだけ小さい。
#
# **目的関数:** 小さくしたい量はカットの大きさです。変数$x_u$を使うと、次のように書けます。
#
# $$
# C(\boldsymbol{x})
# = \sum_{(u,v) \in E} \bigl[x_u(1 - x_v) + x_v(1 - x_u)\bigr] .
# $$
#
# 各項は、辺が分割をまたぐときちょうど$1$になり、それ以外では$0$になります。したがって、この和は分割をまたぐ辺の数を数えています。
#
# **制約:** 2つのグループは同じ大きさでなければなりません。
#
# $$
# \sum_{u \in V} x_u = \frac{|V|}{2} .
# $$
#
# この和はグループ1の頂点の数を数えています。したがってこの式は、グループ1がちょうど半分の頂点を含み、残りがグループ0に入ることを表します。
#
# **目標:** 制約を満たす割り当てのうち、目的関数の値が最も小さくなるものを見つけます。
#
# $$
# \min_{\boldsymbol{x}} C(\boldsymbol{x})
# \quad \text{subject to} \quad
# \sum_{u \in V} x_u = \frac{|V|}{2} .
# $$
#
# 制約がなければ、この問題は自明です。すべての頂点を1つのグループに入れれば、分割をまたぐ辺はなく、カットの大きさは0です。
#
# ### 使用するインスタンス
#
# 8頂点、16辺の固定されたグラフを使います。
#

# %%
NUM_NODES = 8
EDGE_LIST = (
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 7),
    (2, 3),
    (2, 6),
    (3, 5),
    (4, 5),
    (4, 6),
    (5, 6),
    (5, 7),
    (6, 7),
)

# 2つのグループは同じ大きさなので、頂点の数は偶数でなければなりません。
assert NUM_NODES % 2 == 0, "NUM_NODES must be even: the two groups have equal size."
# JijModelingは空の辺リストを受け取れないので、グラフには辺が1本以上必要です。
assert len(EDGE_LIST) > 0, "EDGE_LIST must contain at least one edge."

graph = nx.Graph()
graph.add_nodes_from(range(NUM_NODES))
graph.add_edges_from(EDGE_LIST)
positions = nx.spring_layout(graph, seed=1)

assert graph.number_of_nodes() == NUM_NODES
assert graph.number_of_edges() == len(EDGE_LIST) == 16

# %%
plt.figure(figsize=(5, 5))
nx.draw(
    graph,
    positions,
    with_labels=True,
    node_color="white",
    node_size=700,
    edgecolors="black",
)
plt.title(f"Graph: {graph.number_of_nodes()} vertices, {graph.number_of_edges()} edges")
plt.show()

# %% [markdown]
# 別のグラフを試すには、`NUM_NODES`と`EDGE_LIST`を変更してください。頂点の数は偶数のままにし、辺を1本以上残してください。このチュートリアルのassertにある数、たとえばここでの16辺や、以下のハミルトニアンの28項は、このグラフでのみ成り立つので、あわせて更新してください。また、[ペナルティ定式化](#qaoa-graph-partition-penalty-formulation)で説明するように、ペナルティの重みが十分大きいことも確認してください。重みが小さすぎると、[ペナルティモデルの作成](#qaoa-graph-partition-penalty-model)にある全探索の確認で、ノートブックが止まります。
#
# [結果](#qaoa-graph-partition-result)の節では、実行結果を3つのassertで確認します。1つ目は、実行可能なサンプルの割合が、一様ランダムなビット列での割合より大きいことです。2つ目は、最良の実行可能なサンプルのカットの大きさが、厳密解で求めた最小のカットの大きさに等しいことです。3つ目は、そのカットの大きさが、同じショット数の一様ランダムなビット列で期待される回数より多くサンプルされることです。QAOAはこのどれも保証しません。あなたのグラフや設定でどれかが成り立たないと、ノートブックはそこで`AssertionError`を出して止まります。これは、QAOAの結果が弱かったことを示していて、コードの誤りではありません。
#
# 大きなグラフやショット数の多い実行では、`Reference`の作業量の上限を超えることがあります。その場合は`SAMPLE_SHOTS`と`FINAL_SHOTS`を減らしてください。
#
# ### JijModelingによる定式化
#
# 同じ問題を[JijModeling](https://jij-inc-jijmodeling-tutorials-ja.readthedocs-hosted.com/ja/latest/introduction.html)で書きます。コード中のコメントは、どの行が問題のどの部分に対応するかを示しています。

# %%
# "Graph Partitioning"という名前の空の問題を作ります。既定では最小化します。
formulation = jm.Problem("Graph Partitioning")


# `update`はこの関数を1回実行し、関数が問題の中身を定義します。
@formulation.update
def _(problem: jm.DecoratedProblem):
    # グラフのデータのプレースホルダーです。値はあとで`eval`が入れます。
    vertices = problem.Dim()  # 頂点の数|V|
    edges = problem.Natural(ndim=2)  # 辺のリスト。1行が1つの辺[u, v]です

    # 頂点ごとに1つのバイナリ変数x_uを用意します。
    x = problem.BinaryVar(shape=(vertices,))

    # 目的関数C(x)：辺ごとに1つの項があり、辺が
    # 分割をまたぐとき1になります。
    problem += (
        edges.rows()
        .map(lambda edge: x[edge[0]] * (1 - x[edge[1]]) + x[edge[1]] * (1 - x[edge[0]]))
        .sum()
    )

    # 制約：グループ1がちょうど半分の頂点を含みます。
    problem += problem.Constraint("Equal Partition", x.sum() == vertices / 2)


formulation

# %% [markdown]
# グラフのデータを入れると、具体的な問題インスタンスが得られます。

# %%
instance = formulation.eval(
    {
        "vertices": NUM_NODES,
        "edges": [list(edge) for edge in EDGE_LIST],
    }
)

# %% [markdown]
# ### 厳密解（全探索）
#
# 8頂点なら割り当ては$2^8 = 256$通りしかないので、すべて試して厳密な答えを求められます。これを、あとでQAOAの結果を確かめるための参照値として残します。頂点が1つ増えるごとに割り当ての数は2倍になるので、この方法は大きな問題には使えません。

# %%
# 頂点をグループ0と1に分ける割り当てをすべて試します。
cuts = {}
for bits in itertools.product((0, 1), repeat=NUM_NODES):
    # bits[u]は頂点uのグループです。グループの大きさが異なる割り当ては飛ばします。
    if sum(bits) != NUM_NODES // 2:
        continue
    # 分割をまたぐ辺、つまり両端が異なるグループにある辺を数えます。
    cuts[bits] = sum(bits[left] != bits[right] for left, right in EDGE_LIST)

# 最小のカットの大きさと、それを実現するすべての割り当てを求めます。
best_feasible_cut = min(cuts.values())
optimal_partitions = [bits for bits, cut in cuts.items() if cut == best_feasible_cut]

print(f"Equal-size assignments: {len(cuts)}")
print(f"Smallest cut size: {best_feasible_cut}")
for bits in optimal_partitions:
    print(f"Optimal assignment: {bits}")

# 最適な割り当ての2つのグループを入れ替えたものも最適です。
assert all(
    tuple(1 - bit for bit in bits) in optimal_partitions for bits in optimal_partitions
)

# %% [markdown]
# 2つの最適な割り当ては、2つのグループの名前を入れ替えただけの同じ分割です。名前を入れ替えても、どの辺が分割をまたぐかは変わらないので、最適な割り当ては必ずこのようなペアで現れます。

# %% [markdown]
# (qaoa-graph-partition-algorithm)=
# ## アルゴリズム
#
# 量子近似最適化アルゴリズム（QAOA）は、[Farhi, Goldstone and Gutmann, arXiv:1411.4028](https://doi.org/10.48550/arXiv.1411.4028)で提案されました。QAOAへの入力はコスト関数です。コスト関数とは、バイナリ変数のすべての割り当てに、コストという1つの数を与える規則です。QAOAはこの関数から、調整できる角度を$2p$個持つ回路を作ります。$p$は繰り返す層の数で、私たちが選びます。回路を実行して量子ビットを測定すると、割り当てが1つ返ります。測定でコストの低い割り当てが得られやすくなるように、古典オプティマイザが角度を調整します。この節では、まず制約付きの問題を、QAOAが必要とする形である、1つのコスト関数を持つ制約なしの問題に直し、その関数をイジングモデルのエネルギーとして書きます。そのあと、回路とその状態、角度の調整の仕方、サンプルを元の問題に照らして確認する方法を説明します。
#
# (qaoa-graph-partition-penalty-formulation)=
# ### ペナルティ定式化
#
# QAOAは、追加の条件を付けずに、すべての割り当ての上で1つのコストを最小化します。そこで制約を、重み$\lambda > 0$の2次のペナルティとしてコストに組み込みます。こうして得られるのがペナルティ付きコスト$E_\lambda$です。あとでイジングモデルのエネルギーになるので、エネルギーとも呼びます。
#
# $$
# E_\lambda(\boldsymbol{x})
# = C(\boldsymbol{x})
# + \lambda \Bigl(\sum_{u \in V} x_u - \frac{|V|}{2}\Bigr)^2 .
# $$
#
# 制約を満たす割り当てを*実行可能*（feasible）な割り当てと呼びます。ペナルティは、実行可能な割り当てではゼロになり、それ以外の割り当てでは$\lambda$以上になります。実行不可能な割り当てを禁止するのではなく、コストを高くするだけです。
#
# 重みは十分大きくなければなりません。$\lambda$が小さすぎると、$E_\lambda$を最小にする割り当てが実行不可能になることがあります。たとえば$\lambda = 0$では、すべての頂点を1つのグループに入れる割り当てが$E_\lambda$を最小にします。この割り当てでは、分割をまたぐ辺はありません。
#
# 簡単な見積もりとして、頂点を1つ動かす場合を考えます。同じ大きさの分割から、頂点を1つもう一方のグループへ移します。ペナルティは0から$\lambda$に増えます。カットの大きさが減るのは、多くても、その頂点につながっていて分割をまたいでいる辺の数だけです。この数は$\min(\Delta, |V|/2)$以下で、$\Delta$は1つの頂点につながる辺の数の最大値です。したがって、$\lambda \ge \min(\Delta, |V|/2)$であれば、この移動でエネルギーは下がりません。これは[Lucas, Frontiers in Physics 2, 5 (2014)](https://doi.org/10.3389/fphy.2014.00005)（第2.2節）が行っている比較です。このグラフは$\Delta = 5$、$|V| = 8$なので、見積もりは$\lambda \ge 4$を要求します。この見積もりは1回の移動しか考えていないので、目安であって証明ではありません。ここでは$\lambda = 4$を使い、この値でよいことを[ペナルティモデルの作成](#qaoa-graph-partition-penalty-model)で全探索によって確認します。
#
# ### イジング定式化
#
# Pauli演算子$Z$の固有値は、$|0\rangle$で$+1$、$|1\rangle$で$-1$です。この値に合わせて、バイナリ変数$x_u$をスピン$s_u \in \{+1, -1\}$に置き換えます。
#
# $$
# x_u = \frac{1 - s_u}{2} ,
# $$
#
# したがって、$x_u = 0$は$s_u = +1$に、$x_u = 1$は$s_u = -1$に対応します。すると、$C$の辺ごとの項は$(1 - s_u s_v)/2$になり、$\sum_u x_u - |V|/2$は$-\tfrac{1}{2}\sum_u s_u$になります。エネルギーは次のようになります。
#
# $$
# E_\lambda(\boldsymbol{s})
# = \sum_{(u,v) \in E} \frac{1 - s_u s_v}{2}
# + \frac{\lambda}{4}\Bigl(\sum_{u \in V} s_u\Bigr)^2 .
# $$
#
# Lucasはコストをこの形で書いていて、スピンの和の2乗に重み$A$を、カットの大きさに重み$B$を付けています。ここでのコストは、$A = \lambda/4$、$B = 1$とした場合にあたります。Lucasの式(10)、$A/B \ge \min(2\Delta, N)/8$（$N = |V|$）は、前の節の見積もり$\lambda \ge \min(\Delta, |V|/2)$と同じものです。
#
# $s_u^2 = 1$を使って2乗を展開すると、定数と、頂点のすべてのペアについての項が残ります。
#
# $$
# E_\lambda(\boldsymbol{s}) = c_0 + \sum_{u < v} J_{uv}\, s_u s_v ,
# \qquad
# c_0 = \frac{|E|}{2} + \frac{\lambda |V|}{4} ,
# \qquad
# J_{uv} =
# \begin{cases}
# \lambda/2 - 1/2 & (u, v) \in E \\
# \lambda/2 & \text{それ以外}
# \end{cases}
# $$
#
# これは、相互作用するスピンのモデルであるイジングモデルのエネルギーで、$J_{uv}$はスピン$u$とスピン$v$の間の結合です。スピンが1つだけの項はありません。2つのグループを入れ替えて$\boldsymbol{s} \to -\boldsymbol{s}$としても、エネルギーが変わらないからです。定数$c_0$は、$2^{|V|}$通りすべての割り当てについてのエネルギーの平均です。積$s_u s_v$の平均はどれもゼロになるからです。
#
# 各スピン$s_u$を量子ビット$u$に作用する演算子$Z_u$に置き換えると、*コストハミルトニアン*が得られます。
#
# $$
# H_C = c_0 + \sum_{u < v} J_{uv}\, Z_u Z_v .
# $$
#
# 各$Z_u$は計算基底で対角なので、$H_C$も対角です。すべての$x_u$の割り当てであるビット列$z$について、次が成り立ちます。
#
# $$
# H_C |z\rangle = E_\lambda(z)\, |z\rangle .
# $$
#
# つまり、どのビット列も$H_C$の固有状態で、固有値はそのビット列のエネルギーです。エネルギーが最も低い割り当ては、固有値が最も小さい固有状態、すなわち$H_C$の*基底状態*です。グループを入れ替えてもエネルギーは変わらないので、基底状態は少なくとも2つあります。問題を解くことは、そのうちの1つを見つけることに当たります。
#
# このグラフで$\lambda = 4$とすると、$c_0 = 16$、16本の辺では$J_{uv} = 1.5$、残りの12ペアでは$J_{uv} = 2$となります。グラフの辺は16本しかありませんが、ペナルティが頂点のすべてのペアを結合するので、$H_C$には$Z_u Z_v$の形の項が$\binom{8}{2} = 28$個あります。
#
# ### QAOA状態
#
# QAOAは、頂点ごとに1個、合わせて$n = |V|$個の量子ビットの回路で、$H_C$の基底状態の1つを探します。量子ビットは、すべてのビット列を重ね合わせとして同時に保持できます。
#
# $$
# |\psi\rangle = \sum_{z} a_z\, |z\rangle ,
# $$
#
# これを測定すると、ビット列$z$が確率$|a_z|^2$で得られます。目的は、エネルギーの低いビット列の確率が大きい状態を作ることです。そうすれば、それほど多くない測定回数で、よい分割が得られる見込みが高くなります。
#
# **初期状態:** 回路は一様重ね合わせ$|+\rangle^{\otimes n}$から始まります。この状態では、どのビット列も同じ確率を持ちます。各量子ビットにアダマールゲートを適用すると準備できます。
#
# **コストユニタリ:** 角度$\gamma$に対するコストユニタリは$U_C(\gamma) = e^{-i\gamma H_C}$です。$H_C |z\rangle = E_\lambda(z) |z\rangle$なので、各ビット列にエネルギーで決まる位相を与えます。
#
# $$
# U_C(\gamma) |z\rangle = e^{-i\gamma E_\lambda(z)}\, |z\rangle .
# $$
#
# 位相だけでは測定確率は変わりません。各ビット列のエネルギーを状態に記録するだけです。
#
# **ミキサーユニタリ:** 角度$\beta$に対するミキサーユニタリは次の式です。
#
# $$
# U_M(\beta) = e^{-i\beta \sum_u X_u}
# = \prod_u \bigl(\cos\beta\; I - i \sin\beta\; X_u\bigr) ,
# $$
#
# ここで$I$は恒等演算子で、Pauli演算子$X_u$はビット$u$を反転します。ビット列$y$のビット$u$を反転したものを$y^{(u)}$と書きます。量子ビット$u$の因子は、$|y\rangle$を$|y\rangle$と$|y^{(u)}\rangle$の重ね合わせに変えます。
#
# $$
# \bigl(\cos\beta\; I - i \sin\beta\; X_u\bigr)\, |y\rangle
# = \cos\beta\, |y\rangle - i \sin\beta\, |y^{(u)}\rangle .
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
# この式は右から左へ読みます。回路の構造は問題が$H_C$を通して決めていて、自由なパラメータは$2p$個の角度だけです。
#
# **エネルギーの低いビット列が有利になる理由:** 角度が小さい1層の場合を考えます。コストユニタリは、どのビット列の振幅にも、そのエネルギーで決まる位相を与えます。$\beta$の1次までで、ミキサーは$y$の振幅に、$n$個のビット列$y^{(u)}$の振幅を加えます。$y$の確率には、$\beta$だけの項も$\gamma$だけの項もありません。ミキサーだけでも、コストユニタリだけでも、一様な確率は変わらないからです。$y$に依存する最低次の項は、積$\beta\gamma$に比例します。この項を残し、角度のより高いべきの項を落とすと、次のようになります。
#
# $$
# |a_y|^2 \approx \frac{1}{2^n}
# \bigl(1 + 8\beta\gamma\,(E_\lambda(y) - c_0)\bigr) .
# $$
#
# $\gamma > 0$、$\beta < 0$のとき、因子$\beta\gamma$は負なので、エネルギーが平均$c_0$より低いビット列は一様状態のときより得られやすくなり、平均より高いビット列は得られにくくなります。$\beta > 0$では、同じ式がエネルギーの高いビット列を有利にするので、$\beta$の符号が重要です。[QAOAによるMaxCut](qaoa_maxcut)は、定数のないエネルギーについて、この式を「QAOA状態」の節で順を追って導いています。ここでも同じ手順でこの式が得られます。この式は1層で角度が小さい場合のものなので、示しているのは効果の向きであって、以下で使う回路の確率ではありません。
#
# 2つのユニタリはどちらも必要です。一様状態に対して、コストユニタリだけでは位相しか変わらず、ミキサーだけでは測定で見える変化は何も起きません。
#
# **ゲート:** $H_C$の項はどれも$Z$演算子だけでできているので、すべて可換です。したがって、$U_C$は項ごとの因子に厳密に分解でき、ミキサーも量子ビットごとの因子に分解できます。Qamomileの回転ゲートは、$R_{ZZ}(\theta) = e^{-i\theta Z \otimes Z/2}$、$R_X(\theta) = e^{-i\theta X/2}$のように、指数に因子$1/2$を含む形で定義されています。そのため、それぞれの因子は、角度を2倍にした1つのゲートになります。
#
# $$
# e^{-i\gamma J_{uv} Z_u Z_v} = R_{ZZ}(2\gamma J_{uv}) ,
# \qquad
# e^{-i\beta X_u} = R_X(2\beta) .
# $$
#
# 定数$c_0$はグローバル位相を加えるだけで、測定には影響しません。したがって、各層には28個の$R_{ZZ}$ゲートと8個の$R_X$ゲートがあります。
#
# ### 角度の最適化
#
# 角度は、QAOA状態のエネルギーの期待値ができるだけ小さくなるように選びます。
#
# $$
# F_p(\boldsymbol{\gamma}, \boldsymbol{\beta})
# = \langle \boldsymbol{\gamma}, \boldsymbol{\beta} |\, H_C \,|
# \boldsymbol{\gamma}, \boldsymbol{\beta} \rangle
# = \sum_z \bigl|\langle z | \boldsymbol{\gamma}, \boldsymbol{\beta}
# \rangle\bigr|^2\, E_\lambda(z) .
# $$
#
# $E_\lambda$はカットの大きさとペナルティの和なので、$F_p$は、カットの大きさの期待値とペナルティの期待値の和です。$F_p$がその下限である$E_\lambda$の最小値に近づけるのは、実行不可能なサンプルがまれで、実行可能なサンプルのカットの大きさが小さい場合だけです。
#
# $F_p$は$2^n$個すべてのビット列についての和ですが、個々の確率が必要になることはありません。量子コンピュータは、確率を出力しません。回路を1回実行すると、ビット列が1つ返り、ビット列$z$は確率$|\langle z | \boldsymbol{\gamma}, \boldsymbol{\beta} \rangle|^2$で現れます。回路を$N$回実行すると、ビット列$z^{(1)}, \dots, z^{(N)}$が得られます。それぞれのエネルギーは簡単に計算でき、その平均が$F_p$の推定値になります。
#
# $$
# F_p(\boldsymbol{\gamma}, \boldsymbol{\beta})
# \approx \frac{1}{N} \sum_{k=1}^{N} E_\lambda\bigl(z^{(k)}\bigr) .
# $$
#
# この推定値には統計的なノイズがあり、$N$を増やすと小さくなります。古典オプティマイザは、$F_p$をブラックボックスとして扱い、角度を探します。オプティマイザが角度の組を1つ選ぶと、その角度で回路を$N$回実行し、エネルギーの平均が値として返ります。オプティマイザは、それまでに受け取った値から次の角度の組を選び、オプティマイザが止まるまでこれを繰り返します。1回の評価は$N$ショットの新しいバッチなので、全体のコストは評価の回数に$N$を掛けたものです。このためQAOAはハイブリッドアルゴリズムと呼ばれます。量子回路が与えられた角度でのコストを評価し、古典コンピュータが角度を探します。
#
# このチュートリアルでは、SciPyのConstrained Optimization BY Linear Approximation（COBYLA）法を使います。この方法は勾配を必要としません。COBYLAは、$2p + 1$組の角度での値を保持し、それらを通る角度の1次関数を当てはめ、その関数を使って次の組を選びます。1ステップにかかる評価は1回です。どのように始まり、どのように止まるかは、[結果](#qaoa-graph-partition-result)の節で見ます。
#
# 回路のサンプリングには2つの段階があります。最適化の間は、各角度でのサンプルは$F_p$の推定にだけ使い、個々のビット列は使いません。最適化のあとは、返された角度での最後のバッチが、答えの候補になります。
#
# 層を増やすことは、原理的には悪影響を与えません。$M_p$を、すべての角度の中での$F_p$の最小値とします。このとき$M_p \le M_{p-1}$であり、$p \to \infty$で$M_p$は$E_\lambda$の最小値に近づきます（[Farhi, Goldstone and Gutmann, arXiv:1411.4028](https://doi.org/10.48550/arXiv.1411.4028)の式(9)と式(10)。論文では最大化の形で書かれています）。小さな$p$に固定すると、このような保証はありません。結果は、オプティマイザ、初期角度、ショット数、$\lambda$にも依存します。
#
# **初期角度:** ランダムな初期角度から最適化すると、質の悪い局所解に収束しやすいことが、MaxCutについて報告されています（[Sack and Serbyn, Quantum 5, 491 (2021)](https://doi.org/10.22331/q-2021-07-01-491)）。そこで、量子アニーリングから得られる角度を初期角度にします。
#
# 量子アニーリングは、あるハミルトニアンのエネルギーが最も低い状態を準備し、そのハミルトニアンを問題のハミルトニアンへゆっくり変化させます。ここでは、ハミルトニアン
#
# $$
# H(s) = -(1 - s) \sum_u X_u + s\, H_C
# $$
#
# を考え、$s$を0から1まで変化させます。$s = 0$では、エネルギーが最も低い状態は一様状態で、これは回路の出発点です。$s = 1$では、最も低いエネルギーは最適な分割のもので、その2つの割り当てが同じエネルギーを持ちます。エネルギーが最も低い状態から出発した系は、変化が十分にゆっくりで、その状態がエネルギーギャップによってほかの状態から隔てられていれば、ハミルトニアンが変化しても、エネルギーが最も低い状態の近くにとどまります（[Albash and Lidar, Reviews of Modern Physics 90, 015002 (2018)](https://doi.org/10.1103/RevModPhys.90.015002)）。最も低いエネルギーを2つの割り当てが共有していても、この条件は崩れません。2つのグループを入れ替えても、$H(s)$も一様状態も変わりません。そのため、状態は途中でもこの対称性を保ち、問題になるのは、同じ対称性を持つほかの状態とのギャップだけです。このギャップが開いたままなら、十分にゆっくりした変化によって、2つの割り当てを等しい重みで重ね合わせた状態に行き着き、測定すると最適な分割が得られます。[QAOAによるMaxCut](qaoa_maxcut)でも、同じ考え方から初期角度を選んでいます。ここではこの条件を確かめず、この考え方は角度を選ぶための指針としてだけ使います。
#
# これを層に直すために、変化にかける全体の時間を$T$とし、時刻$t$で$s = t/T$とします。そして、変化を長さ$\Delta t = T/p$の等しい$p$個のステップに区切ります。ステップ$k$では、$s$をステップの中央での値
#
# $$
# s_k = \frac{k - 1/2}{p} ,
# \qquad
# k = 1, \dots, p ,
# $$
#
# で置き換え、$H(s_k)$の2つの部分を、それぞれ時間$\Delta t$だけ、1つずつ順に作用させます。
#
# $$
# e^{-i \Delta t\, H(s_k)}
# \approx e^{\,i \Delta t\, (1 - s_k) \sum_u X_u}\; e^{-i \Delta t\, s_k H_C}
# = U_M(\beta_k)\, U_C(\gamma_k) .
# $$
#
# これは、次の角度を持つQAOAの1層です。
#
# $$
# \gamma_k = \Delta t\, s_k ,
# \qquad
# \beta_k = -\Delta t\, (1 - s_k) .
# $$
#
# $\beta_k$の負号は、$\sum_u X_u$の前の負号から来ていて、上の式でエネルギーの低いビット列を有利にする符号です。これらの角度は層ごとに一定の割合で変わるので、この選び方を*線形ランプ*と呼びます。$p = 5$ステップだけでは、変化が十分にゆっくりであることを示すものは何もありません。したがって、ランプは出発点であり、そこからオプティマイザが改善します。
#
# ステップ$\Delta t$は、問題のエネルギーの大きさに合わせる必要があります。Sack and Serbynは、3-正則グラフのMaxCutについて$\Delta t = 0.75$を見つけました。そこでは結合はすべて1で、各スピンは3つの結合を持ちます。ここでは、どのスピンもほかの7つすべてと、強さ1.5または2で結合しています。$\Delta t = 0.75$とすると、ランプだけでのエネルギーの期待値は19.4で、一様ランダムなビット列の16を上回ります。$\Delta t = 0.4$では8.6です。ここでは$\Delta t = 0.4$を使います。これは、このグラフでいくつかの値を試して選んだ値なので、このチュートリアルの設定であって、一般的な規則ではありません。$p = 5$のとき、初期角度は$\boldsymbol{\gamma} = (0.04, 0.12, 0.2, 0.28, 0.36)$と$\boldsymbol{\beta} = (-0.36, -0.28, -0.2, -0.12, -0.04)$です。
#
# ### サンプリング後の実行可能性の確認
#
# 最適化が返すのは角度であって、分割ではありません。返るのは、COBYLAが評価したすべての角度の中で、$F_p$の推定値が最も小さかった角度です。分割を得るために、この角度で回路をもう一度実行し、この最後のバッチで測定されたビット列をすべて残します。
#
# ペナルティは実行不可能な割り当てのコストを高くするだけなので、これらのサンプルの中には制約を満たさないものが含まれることがあります。そこで、すべてのサンプルを元の問題で評価します。実行不可能なサンプルは捨て、実行可能なサンプルはカットの大きさ$C(\boldsymbol{x})$で評価します。実行可能なサンプルではペナルティがゼロなので、$C$と$E_\lambda$は同じ値です。
#
# 答えは、カットの大きさが最も小さい実行可能なサンプルです。QAOAはこれが最適解であることを保証しないので、[結果](#qaoa-graph-partition-result)の節で厳密解と比べます。

# %% [markdown]
# ## Qamomileによる実装
#
# (qaoa-graph-partition-penalty-model)=
# ### ペナルティモデルの作成
#
# 次にペナルティを加えます。`OMMXProblem.from_instance`は、インスタンスと重み$\lambda$（`uniform_penalty_weight`で指定します）を受け取り、次の2つを持つ`problem`オブジェクトを返します。
#
# - `problem.model`：QAOAが最小化するペナルティ付きコスト$E_\lambda$です。そのハミルトニアン$H_C$は`problem.model.hamiltonian`です。
# - 制約を含む元の問題：あとで`problem.evaluate`がサンプルの確認に使います。

# %%
PENALTY_WEIGHT = 4.0

# 変換でインスタンスが変わらないことを確認するため、元の内容を保存します。
original_payload = instance.to_bytes()

problem = OMMXProblem.from_instance(instance, uniform_penalty_weight=PENALTY_WEIGHT)

assert instance.to_bytes() == original_payload
assert problem.model.vartype is VarType.BINARY
assert problem.model.num_variables == NUM_NODES
assert problem.model.order == 2
assert BinaryModel.from_bytes(problem.model.to_bytes()) == problem.model

# ハミルトニアンを、イジング定式化の節の展開式と比べます。
hamiltonian = problem.model.hamiltonian
couplings = [coefficient.real for _, coefficient in hamiltonian.terms]

print(f"Constant c_0: {hamiltonian.constant.real}")
print(f"Terms with J = lambda/2 - 1/2: {couplings.count(PENALTY_WEIGHT / 2 - 0.5)}")
print(f"Terms with J = lambda/2: {couplings.count(PENALTY_WEIGHT / 2)}")
print(f"QAOA Hamiltonian terms: {len(hamiltonian.terms)}")

assert hamiltonian.constant == len(EDGE_LIST) / 2 + PENALTY_WEIGHT * NUM_NODES / 4
assert couplings.count(PENALTY_WEIGHT / 2) == (
    NUM_NODES * (NUM_NODES - 1) // 2 - len(EDGE_LIST)
)
assert len(hamiltonian.terms) == 28

# %% [markdown]
# 次に、$\lambda$の選び方を全探索で確認します。$2^8$通りの割り当ての中で、$E_\lambda$の最小値は厳密解で求めた最小のカットの大きさと一致し、それを実現する割り当てはすべて実行可能であるはずです。そうであれば、$E_\lambda$を最小化すると元の問題が解けます。

# %%
# 実行不可能なものも含めて、すべての割り当てのエネルギーを計算します。
penalty_energies = {}
for bits in itertools.product((0, 1), repeat=NUM_NODES):
    penalty_energies[bits] = problem.model.energy(bits)

# 最小のエネルギーと、それを実現するすべての割り当てを求めます。
min_penalty_energy = min(penalty_energies.values())
minimizers = [
    bits for bits, energy in penalty_energies.items() if energy == min_penalty_energy
]

print(f"Minimum of E_lambda: {min_penalty_energy}")
print(f"Smallest cut size: {best_feasible_cut}")
print(f"Number of minimizers: {len(minimizers)}")

assert min_penalty_energy == best_feasible_cut
assert all(sum(bits) == NUM_NODES // 2 for bits in minimizers)

# %% [markdown]
# ### 実験の構築とコンパイル
#
# `QAOAExperiment`は、`problem.model`から[アルゴリズム](#qaoa-graph-partition-algorithm)の節のQAOA回路を作ります。ここでは$p = 5$層です。このチュートリアルでは、Qamomile組み込みのシミュレータ`qm.Reference()`を量子コンピュータの代わりに使います。回路のコンパイルは1回だけです。角度`gammas`と`betas`はパラメータのまま残し、実行時に値を与えるので、どの評価でも同じコンパイル済み回路を使い回せます。

# %%
LAYERS = 1 if docs_test_mode else 5

experiment = QAOAExperiment(problem.model, layers=LAYERS)

# シミュレータ向けに1回だけコンパイルします。
endpoint = qm.Reference()
compiled = experiment.compile(to=endpoint.target)

# 実行時のパラメータとして残るのは角度だけです。
assert tuple(slot.name for slot in compiled.artifact.parameter_abi.slots) == (
    "gammas",
    "betas",
)

# コンパイル済み回路は、この実験とこの問題に対応します。
assert compiled.identity == experiment.identity
assert compiled.identity.problem_digest == problem.model.semantic_digest

# QAOAは通常の演算の組み合わせで、専用の演算はありません。
assert "quantum.qaoa" not in experiment.program.inspect()

# %% [markdown]
# ### 回路の確認
#
# Qamomileでは、どのエンジン向けにも変換せずに演算の数を数えられます。`qm.resources.logical`はプログラムを直接読み取ります。`LAYERS`層の場合、量子ビットごとに1つのアダマールゲート、コスト層ごとに1つのPauli時間発展（28項のコストユニタリ$U_C$全体）、量子ビットと層ごとに1つの$R_X$回転が報告されます。

# %%
study = qm.resources.logical(experiment.program)

# 量子ビットごとに1つのアダマールゲート。
assert study.metric(
    qm.resources.logical_operation_count("quantum.h")
).numeric_enclosure() == (NUM_NODES, NUM_NODES)

# 量子ビットと層ごとに1つのRX回転。
assert study.metric(
    qm.resources.logical_operation_count("quantum.rx")
).numeric_enclosure() == (NUM_NODES * LAYERS, NUM_NODES * LAYERS)

# コスト層ごとに1つのPauli時間発展。
assert study.metric(
    qm.resources.logical_operation_count("quantum.pauli_evolution")
).numeric_enclosure() == (LAYERS, LAYERS)

# %% [markdown]
# (qaoa-graph-partition-result)=
# ## 結果
#
# ### 角度を最適化する
#
# [アルゴリズム](#qaoa-graph-partition-algorithm)の節の最適化のループを、コンパイル済みの回路で実行します。`cost_function`は、COBYLAが評価する関数です。呼び出されるたびに、与えられた角度で、コンパイル済みの回路を`SAMPLE_SHOTS`ショット実行し、サンプルのエネルギーの平均を返します。これが$F_p$の推定値です。`cost_function`はこの値を`cost_history`にも追加するので、このリストには評価1回につき1つの値が入ります。呼び出しのたびに新しいシードを使うので、どの評価も新しいサンプルのバッチです。
#
# 初期角度は、[アルゴリズム](#qaoa-graph-partition-algorithm)の節の線形ランプで、$\Delta t = 0.4$です。コードでは`RAMP_STEP`です。配列`initial_parameters`には、まず$\gamma$の$p$個の値が、続いて$\beta$の$p$個の値が入ります。これは、`cost_function`が配列を分けるときの順序です。オプション`rhobeg`は、COBYLAの最初のステップの幅を、既定値の1.0ではなく0.1にします。1.0は、どの初期角度よりも大きい値です。評価は最大1,000回まで行えます。256ショットの実行は、`Reference`の作業量の上限に収まります。
#
# 実行のあと、セルは、評価の回数とそれに使ったショット数、COBYLAが止まった理由、初期角度での推定値、$F_p$の推定値の最小値、その値が得られた角度を表示します。2つ目のセルは、評価ごとの推定値をプロットします。

# %%
SAMPLE_SHOTS = 1 if docs_test_mode else 256
MAX_ITERATIONS = 2 * LAYERS + 2 if docs_test_mode else 1_000
FINAL_SHOTS = 1 if docs_test_mode else 256
cost_history = []


def cost_function(parameters) -> float:
    """Run the circuit at the given angles and return the mean sample energy."""

    # オプティマイザはすべての角度を1つの配列で渡します。gammas、betasの順です。
    run = compiled.run(
        on=endpoint,
        gammas=tuple(float(value) for value in parameters[:LAYERS]),
        betas=tuple(float(value) for value in parameters[LAYERS:]),
        shots=SAMPLE_SHOTS,
        seed=901 + len(cost_history),
    )
    cost_history.append(run.samples.mean_energy)
    return run.samples.mean_energy


# アルゴリズムの節の線形ランプから始めます。
RAMP_STEP = 0.4
ramp = (np.arange(LAYERS) + 0.5) / LAYERS
initial_parameters = np.concatenate([RAMP_STEP * ramp, -RAMP_STEP * (1 - ramp)])

optimizer_options = {"maxiter": MAX_ITERATIONS, "rhobeg": 0.1}
if docs_test_mode:
    optimizer_options["f_target"] = float("inf")

optimization = minimize(
    cost_function,
    initial_parameters,
    method="COBYLA",
    options=optimizer_options,
)
if docs_test_mode:
    assert optimization.nfev == 1

print(
    f"Evaluations: {optimization.nfev}, shots used: {optimization.nfev * SAMPLE_SHOTS}"
)
print(f"Why COBYLA stopped: {optimization.message}")
print(f"Estimate at the starting angles: {cost_history[0]:.4f}")
print(f"Smallest estimate of F_p: {optimization.fun:.4f}")
print(f"Returned gammas: {[round(float(v), 4) for v in optimization.x[:LAYERS]]}")
print(f"Returned betas: {[round(float(v), 4) for v in optimization.x[LAYERS:]]}")
assert len(optimization.x) == 2 * LAYERS
assert len(cost_history) == optimization.nfev

# %%
plt.figure(figsize=(8, 4))
plt.plot(cost_history, color="#2696EB")
plt.xlabel("Energy evaluation")
plt.ylabel("Mean energy")
plt.title("QAOA optimization progress")
plt.show()

# %% [markdown]
# このプロットから、COBYLAの進み方がわかります。10個の角度の1次関数には11個の係数があるので、COBYLAはまず、11組の角度での値を必要とします。プロットの最初の11点がそれです。初期角度と、そのあとの10組です。この10組はそれぞれ、それまでで最良の組から、角度を1つ、`rhobeg`だけ変えたものです。この実行では、10個の推定値のうち9個が初期角度での値より上にあり、大きく上回るものもあります。ランプはすでによい角度の組であり、1つの角度を0.1変えるだけで悪くなることがあります。
#
# それ以降、COBYLAは、当てはめた1次関数を手がかりにステップを進めます。ステップの長さはある距離までに制限し、この距離を探索が落ち着くにつれて小さくします。この距離が下限まで小さくなるか、評価回数の上限を使い切ると止まります。表示された理由から、この実行は前者で終わったことがわかります。[QAOAによるMaxCut](qaoa_maxcut)は、「角度を最適化する」の節でこれらのステップをより詳しく説明しています。
#
# 比較のため、一様ランダムなビット列の平均エネルギーは$c_0 = 16$で、最小エネルギーは6です。ランプだけで、推定値はすでに8.56まで下がっています。これがプロットの最初の点です。この実行では、オプティマイザによる改善はわずかです。推定値の最小値は8.13ですが、256ショットから得た1つの推定値には約0.25の不確かさがあるので、この大きさの差はノイズをそれほど上回りません。実際、次の節で、返された同じ角度で256ショットの新しいバッチを測定すると、そのエネルギーの平均は8.70です。

# %% [markdown]
# ### サンプリングと評価
#
# これはサンプリングの第2段階です。返された角度で最後に実行し、分割の候補を得ます。この実行は独自のシードを使うので新しいバッチであり、そのエネルギーの平均は$F_p$の新しい推定値です。この推定値は、最適化での推定値の最小値を上回ることがあり、この実行では実際に上回ります。最小値は、ノイズのある多くの値の中で最も低いものだからです。セルは、この平均の標準誤差も表示します。これは、サンプルのエネルギーの標準偏差を$\sqrt{256}$で割ったもので、「角度を最適化する」で述べた、1つの推定値の不確かさです。続いて`problem.evaluate`が、各サンプルを元の問題で確認し、実行可能かどうかと、カットの大きさを報告します。セルは、実行可能なサンプルの割合を、一様ランダムなビット列での割合と比べます。一様ランダムなビット列は、すべての角度が0のときに回路が返すものなので、この比較から、層が加えた効果がわかります。

# %%
optimized_run = compiled.run(
    on=endpoint,
    gammas=tuple(float(value) for value in optimization.x[:LAYERS]),
    betas=tuple(float(value) for value in optimization.x[LAYERS:]),
    shots=FINAL_SHOTS,
    seed=1901,
)
evaluation = problem.evaluate(optimized_run.samples)

# 制約を満たすサンプルのショット数を数えます。
feasible_shots = sum(
    evaluation.occurrences[sample_id]
    for sample_id, feasible in evaluation.feasible.items()
    if feasible
)

# 厳密解の`cuts`には、すべての実行可能な割り当てが入っています。
uniform_feasible = len(cuts) / 2**NUM_NODES

# エネルギーの平均の標準誤差：サンプルのエネルギーの標準偏差を、
# ショット数の平方根で割ったものです。
sample_energies = np.repeat(
    [sample.energy for sample in optimized_run.samples.samples],
    [sample.occurrences for sample in optimized_run.samples.samples],
)
standard_error = sample_energies.std() / np.sqrt(FINAL_SHOTS)

print(f"Mean energy of the final batch: {optimized_run.samples.mean_energy:.4f}")
print(f"Standard error of this mean: {standard_error:.4f}")
print(
    f"Feasible samples: {feasible_shots} / {FINAL_SHOTS} "
    f"({100.0 * feasible_shots / FINAL_SHOTS:.1f}%)"
)
print(f"Feasible share of uniformly random bitstrings: {100.0 * uniform_feasible:.1f}%")

assert optimized_run.execution.artifact is compiled.artifact
assert optimized_run.samples.total_occurrences == FINAL_SHOTS
assert evaluation.total_occurrences == FINAL_SHOTS
if not docs_test_mode:
    assert feasible_shots / FINAL_SHOTS > uniform_feasible

# %% [markdown]
# 実行可能なサンプルの割合は一様ランダムなビット列よりはるかに高く、最適化した回路が同じ大きさの分割に集中していることが分かります。

# %% [markdown]
# ### 最良の実行可能解
#
# `evaluation.best_feasible`は、カットの大きさが最も小さい実行可能なサンプルを返します。各頂点のグループを読み取り、辺のリストから分割をまたぐ辺を数え直して、厳密解と比べます。

# %%
best = (
    evaluation.get(evaluation.sample_ids[0])
    if docs_test_mode
    else evaluation.best_feasible
)

# 各頂点のグループを読み取ります。
values = best.decision_variables_df["value"]
best_assignment = {
    index: int(round(float(values.loc[index]))) for index in range(NUM_NODES)
}
best_objective = int(round(best.objective))

# 辺のリストから、分割をまたぐ辺を数え直します。
cut_edges = sum(
    best_assignment[left] != best_assignment[right] for left, right in EDGE_LIST
)

print("Best feasible partition:", best_assignment)
print(f"Cut size: {best_objective}")
print(f"Exact smallest cut size: {best_feasible_cut}")

assert best_objective == cut_edges
if not docs_test_mode:
    assert sum(best_assignment.values()) == NUM_NODES // 2
    assert best_objective == best_feasible_cut

# %% [markdown]
# 最良のサンプルのカットの大きさは、厳密解の最小値と一致しています。ただし、これだけでは、回路がよい分割を有利にしていることは示せません。ランダムに選んでも、たいていは最適解が見つかるからです。256個のビット列のうち最適なものは2個だけですが、一様ランダムなビット列を256個引くと、その中に最適なものが少なくとも1つ含まれる確率は$1 - (254/256)^{256} \approx 87\%$です。1回引くごとに、どちらにも当たらない確率が$254/256$だからです。そこで次の節では、それぞれのカットの大きさが何回サンプルされたかを見ます。

# %% [markdown]
# ### カットの大きさの分布
#
# 評価結果は、異なるサンプルを1回ずつ、出現回数とともに保持しています。実行可能なサンプルについて、カットの大きさごとに出現回数を合計し、同じショット数で一様ランダムなビット列から得られる回数と比べます。

# %%
# カットの大きさごとに実行可能なサンプルを数えます。
objective_counts = {}
for sample_id in evaluation.sample_ids:
    if not evaluation.feasible[sample_id]:
        continue
    objective = int(round(evaluation.objectives[sample_id]))
    objective_counts[objective] = (
        objective_counts.get(objective, 0) + evaluation.occurrences[sample_id]
    )

assert sum(objective_counts.values()) == feasible_shots

# 一様ランダムなビット列で期待される回数です。厳密解の`cuts`には、
# すべての実行可能な割り当てのカットの大きさが入っています。
uniform_counts = {}
for cut in cuts.values():
    uniform_counts[cut] = uniform_counts.get(cut, 0) + FINAL_SHOTS / 2**NUM_NODES

cut_values = sorted(uniform_counts)
positions_on_axis = np.arange(len(cut_values))
bar_width = 0.4

plt.figure(figsize=(8, 4))
plt.bar(
    positions_on_axis - bar_width / 2,
    [objective_counts.get(value, 0) for value in cut_values],
    width=bar_width,
    color="#2696EB",
    label="QAOA samples",
)
plt.bar(
    positions_on_axis + bar_width / 2,
    [uniform_counts[value] for value in cut_values],
    width=bar_width,
    color="#eb6834",
    label="Uniformly random bitstrings (expected)",
)
plt.xticks(positions_on_axis, [str(value) for value in cut_values])
plt.xlabel("Cut size")
plt.ylabel("Number of feasible samples")
plt.title("Distribution of feasible solutions")
plt.legend()
plt.show()

print("Cut size: QAOA samples, uniformly random (expected)")
for value in cut_values:
    print(f"{value}: {objective_counts.get(value, 0)}, {uniform_counts[value]:.1f}")

if not docs_test_mode:
    assert (
        objective_counts.get(best_feasible_cut, 0) > uniform_counts[best_feasible_cut]
    )

# %% [markdown]
# QAOAはサンプルを小さなカットの大きさのほうへ移しています。最適なカットの大きさ6は256サンプル中24回現れ、一様ランダムなビット列で期待されるのは2回です。最適化した回路がよい分割を有利にしていることを示すのは、1つの最良のサンプルではなく、この比較です。それでも、最も多いのはカットの大きさ8です。カットの大きさが8になる実行可能な割り当ては16通りあり、6になる割り当ては2通りです。1つの割り当てあたりでは、最適な割り当てが平均12サンプル、カットの大きさ8の割り当てが平均5サンプルで、最適な割り当てのほうが確率は高くなっています。しかし、割り当ての数の差がこれを上回ります。したがって、答えはサンプルの中から選び出す必要があります。

# %% [markdown]
# ### 最良の分割の可視化
#
# 色は最良の実行可能なサンプルの2つのグループを表します。分割をまたぐ6本の辺は、色の異なる頂点を結ぶ辺です。

# %%
# 各頂点をグループごとに色分けします。
colors = [
    "#FF6B6B" if best_assignment[index] == 1 else "#4ECDC4"
    for index in range(NUM_NODES)
]
plt.figure(figsize=(5, 5))
nx.draw(
    graph,
    positions,
    with_labels=True,
    node_color=colors,
    node_size=700,
    edgecolors="black",
)
plt.title(f"Best feasible partition (cut size = {best_objective})")
plt.show()

# %% [markdown]
# ## まとめ
#
# このチュートリアルでは、次のことを行いました。
#
# 1. JijModelingでグラフ分割問題を書き、すべての割り当てを試して、カットの大きさ6の厳密解を求めました。
# 2. 同じ大きさという制約を`OMMXProblem`で2次のペナルティに変え、重み$\lambda = 4$が十分大きいことを全探索で確認しました。
# 3. `QAOAExperiment`でQAOA回路を構築し、コンパイルしました。ペナルティが頂点のすべてのペアを結合するので、グラフの辺は16本だけですが、5つの層のそれぞれに28個の$R_{ZZ}$回転があります。
# 4. 線形ランプから始めて、COBYLAで10個の角度を調整しました。評価のたびに、コンパイル済みの回路を新しい角度の組で実行し、そのサンプルのエネルギーの平均を返しました。
# 5. 返された角度で回路をサンプリングし、すべてのサンプルを元の問題で評価しました。一様ランダムなビット列と比べて、実行可能な割合は27.3%から92.2%に上がり、最適なカットの大きさは2回ではなく256サンプル中24回現れました。
#
# ペナルティの重み$\lambda$とランプの時間ステップ$\Delta t$は、このグラフに合わせて選んだものなので、問題によってはどちらも変える必要があるかもしれません。
#
# QAOA回路をゲートごとに組み立てる方法は、[QAOAによるMaxCut](qaoa_maxcut)を参照してください。
