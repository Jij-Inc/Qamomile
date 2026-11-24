# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# tags: [algorithm, finance, variational]
# ---
#
# # Pauli相関エンコーディングを用いたポートフォリオ最適化
#
# このチュートリアルでは、Pauli相関エンコーディング（PCE）を用いてポートフォリオの銘柄選択問題を解く方法を示します。{cite:t}`10.48550/arXiv.2511.21305`の手順に沿って、PCEで市場グラフを再帰的に分割し、各クラスタから1銘柄を選びます。`PauliCorrelationEncoding`、`@qm.qkernel`で定義するアンザッツ、`Estimator`による測定に基づく期待値推定を使い、アルゴリズムをQamomileで実装して量子回路を`qm.Reference()`でエミュレートします。この例では、2013〜2018年のS&P 500構成銘柄から30銘柄を取り出し、データを時系列に沿って4:1に分割します。

# %%
# pipで最新版のQamomileをインストールします。
# # !pip install "qamomile[visualization]" kagglehub networkx pandas

# %%
from __future__ import annotations

import glob
import os
from collections import deque
from math import comb

import kagglehub
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import qamomile as qm
from qamomile.measurement import Estimator
from qamomile.optimization import BinaryModel, PauliCorrelationEncoding
from qamomile.stdlib import cx_entangling_layer

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# ## 背景
#
# ポートフォリオの銘柄選択では、期待リターンと資産価値の変動のバランスを考えます。ポートフォリオの分散は、各銘柄のリターンの分散と銘柄間の共分散の両方に依存します。そのため、値動きの連動が弱い銘柄を組み合わせることで、リスクを低減できる場合があります。ここでは、関連する銘柄をグループ化し、各グループからリターンが最も高い代表銘柄を選ぶことで分散投資を図ります。これはヒューリスティックであり、平均分散最適化問題を直接解くものではありません。
#
# 銘柄選択に量子アルゴリズムを適用する動機の1つは、組合せ最適化を高速化できる可能性です。標準的なQAOAやVQEのエンコードでは、通常、選択を表す各バイナリ変数に1量子ビットを割り当てるため、小規模なデバイスで扱える銘柄数に制約があります。{cite:t}`10.1038/s41467-024-55346-z`で提案されたPauli相関エンコーディング（PCE）は、$k$を固定したとき、$n = \mathcal{O}(m^{1/k})$量子ビット上の$m$個の$k$体のPauli相関演算子の期待値の符号で$m$個のバイナリ変数を表します。PCEの基礎（エンコード、損失関数、デコード）については、[PCEによるMaxCutのチュートリアル](pce_maxcut.ipynb)を参照してください。ここでは、市場グラフにPCEによる分割を繰り返し適用する方法に焦点を当てます。

# %% [markdown]
# ## 問題設定
#
# このチュートリアルでは、Kaggleの`camnugent/sandp500`データセットのバージョン4を使用します。2013年2月8日から2018年2月7日までのS&P 500構成銘柄505銘柄について、始値・高値・安値・終値・出来高を含む619,040件の日次レコードが収録されています。このうち470銘柄には、データセット内の全取引日の終値が揃っています。
#
# 終値に欠損のない銘柄を残し、アルファベット順で先頭から`M = 30`銘柄を取り出します。日次騰落率をリターンとして計算し、時系列に沿って4:1に分割します。最初の80%をグラフの構築と代表銘柄の選択に、最後の20%を評価に使用します。データセットの終値をそのまま使い、株式分割や配当の追加調整は行いません。そのため、結果は比較の一例であり、トータルリターンのバックテストではありません。

# %%
M = 6 if docs_test_mode else 30

# %% [markdown]
# ## アルゴリズム
#
# 銘柄間の相関グラフを構築し、PCEでMaxCutを解いて目標のクラスタ数に達するまで二分割を繰り返します。その後、各クラスタから代表銘柄を1つ選び、等ウェイトのポートフォリオを評価します。クラスタリングで分散投資を図り、各クラスタで訓練期間のリターンが最も高い銘柄を選ぶことでリターンを追求します。ただし、この選択によって異なるクラスタから相関の強い高リターン銘柄が集まる場合があるため、最終的な代表銘柄間の相関が低くなるとは限りません。
#
# ### 市場グラフと再帰的MaxCut
#
# 各銘柄をノードとします。訓練期間のリターン$r_i$とその標準偏差$\sigma_i$を使って、相関係数を$\rho_{ij} = \mathrm{Cov}(r_i,r_j)/(\sigma_i\sigma_j)$と定義し、その絶対値が$\lambda$を超える場合にのみエッジを残します。
#
# ```{math}
# :label: eq-portfolio-weight
# w_{ij} = 1 - |\rho_{ij}| \quad\text{for } |\rho_{ij}| > \lambda.
# ```
#
# 閾値以下のペアにはエッジを張りません。グラフに用いる統計量は、すべて訓練期間のデータから計算します。ここで$m$は対象とする全銘柄数（`M`）、$m_a$は各分割で孤立ノードを除いたアクティブなノード数（`n_var`）です。$\lambda=0.25$とし、幅優先の順序で6回の分割を成功させて、空でない7つのクラスタを作ります。
#
# 式{eq}`eq-portfolio-weight`では、残したエッジのうち相関が弱いものほど重みが大きくなります。MaxCutは分割をまたぐエッジの重みの合計が大きくなる分割を求めるため、関連の弱い銘柄を分離し、関連の強い銘柄を同じグループに残すことを狙います。閾値を設け、カット全体を目的関数とするため、この方法はヒューリスティックです。
#
# クラスタを表すバイナリラベル$x_i\in\{0,1\}$に対して、スピン$s_i=2x_i-1\in\{-1,1\}$を導入すると、重み付きカットは次のようになります。
#
# ```{math}
# :label: eq-portfolio-cut
# \mathrm{Cut}(\mathcal G,\boldsymbol s)
# = \frac12\sum_{(i,j)\in E}w_{ij}(1-s_i s_j).
# ```
#
# したがって、カットの最大化は$\sum_{(i,j)\in E}w_{ij}s_i s_j$の最小化と等価です。

# %%
LAMBDA = 0.25
n_splits = 1 if docs_test_mode else 6

# %% [markdown]
# ### PCEによる1回の分割
#
# アクティブなノードをPCEでエンコードし、パラメータ付き量子状態を準備します。式{eq}`eq-portfolio-cut`の各スピンを$z_i=\tanh(\alpha\langle\Pi_i\rangle)$で置き換え、ポートフォリオ論文の係数のスケールに合わせて、次の損失関数を最小化します。
#
# ```{math}
# :label: eq-portfolio-loss
# \mathcal L = \sum_{(i,j)\in E}w_{ij}z_i z_j
# + \beta\nu\left(\frac1{m_a}\sum_i z_i^2\right)^2,\qquad
# \nu=\frac W2+\frac{\bar w(m_a-1)}4.
# ```
#
# ここで$W=\sum_{(i,j)\in E}w_{ij}$、$\bar w=W/|E|$です。緩和と正則化項については、[PCEのチュートリアル](pce_maxcut.ipynb)で説明しています。データ項にはIsing係数$w_{ij}/2$ではなく$w_{ij}$を用います。この係数のスケールは、正則化項との相対的な大きさに影響します。Pauli相関演算子の次数を$k=2$、$\beta=0.5$とし、論文の調整規則$\alpha=n^{\lfloor k/2\rfloor}$を使います。この$\alpha$は最適性が証明された値ではありません。`ALPHA = None`を指定すると各分割にこの規則を適用し、数値を指定するとその値を使用します。回路パラメータの最適化後、正の期待値を$+1$、0以下の期待値を$-1$にデコードします。孤立ノードはカットに影響しないため、別途割り当てます。
#
# :::{note}
# Qamomileのエンコードには$X_iY_j$のように異なるPauli演算子が混在するPauli文字列が含まれます。一方、ポートフォリオ論文では、恒等演算子以外の位置がすべて$X$、すべて$Y$、またはすべて$Z$である文字列を使います。そのため、変数と演算子の対応が異なり、必要な量子ビット数も変わる場合があります。ここでは、Qamomileのエンコードを用いて論文のポートフォリオ構築手順を実装します。
# :::

# %%
K = 2
ALPHA = None
BETA = 0.5

# %% [markdown]
# ### 代表銘柄の選択
#
# 最終的な各クラスタ$L_\ell$から、訓練期間の平均リターンが最も高い銘柄を選びます。
#
# ```{math}
# :label: eq-portfolio-selection
# v_\ell^* = \operatorname*{arg\,max}_{v\in L_\ell}
# \overline r_v^{\,\mathrm{train}}.
# ```
#
# クラスタリングで分散投資を図り、代表銘柄の選択でリターンを追求します。各クラスタでリターンが最も高い銘柄を選んでも、異なるクラスタの代表銘柄間に相関が残る場合があります。各クラスタから1銘柄を選ぶだけでは、銘柄間の相関の低さは保証されません。選んだ銘柄を等ウェイトで組み合わせたポートフォリオを、評価用に取り分けた期間で評価します。

# %% [markdown]
# ## Qamomileによる実装
#
# ### データの準備
#
# 繰り返し実行しても同じ入力を使えるよう、データセットのバージョンを固定します。

# %%
if docs_test_mode:
    rng = np.random.default_rng(42)
    market = rng.normal(0.0005, 0.006, (80, 1))
    daily = market + rng.normal(0.0, 0.008, (80, M))
    synthetic_prices = 100.0 * np.cumprod(1.0 + daily, axis=0)
    raw = (
        pd.DataFrame(
            synthetic_prices,
            index=pd.date_range("2020-01-01", periods=80, freq="B", name="date"),
            columns=[f"ASSET{i:02d}" for i in range(M)],
        )
        .reset_index()
        .melt(id_vars="date", var_name="name", value_name="close")
    )
else:
    path = kagglehub.dataset_download("camnugent/sandp500/versions/4")
    csv_candidates = glob.glob(
        os.path.join(path, "**", "all_stocks_5yr.csv"), recursive=True
    )
    assert len(csv_candidates) == 1
    raw = pd.read_csv(csv_candidates[0])

raw.columns = [c.lower() for c in raw.columns]
raw["date"] = pd.to_datetime(raw["date"])
wide = raw.pivot(index="date", columns="name", values="close").sort_index()
wide = wide.loc[:, wide.notna().all()]
assert wide.shape[1] >= M

tickers = sorted(wide.columns)[:M]
prices = wide[tickers]
returns = prices.pct_change().dropna()
split = int(len(returns) * 0.8)
train, test = returns.iloc[:split], returns.iloc[split:]
assert len(tickers) == M and len(train) > 1 and len(test) > 1
assert train.index[-1] < test.index[0]
assert np.isfinite(returns.to_numpy()).all()

print(f"tickers: {tickers}")
print(f"train: {train.shape} ({train.index[0].date()} - {train.index[-1].date()})")
print(f"test : {test.shape} ({test.index[0].date()} - {test.index[-1].date()})")

# %% [markdown]
# ### 市場グラフの構築
#
# いくつかの閾値でグラフの密度を確認します。論文では密度が0.53〜0.87と報告されていますが、$\lambda$は明示されていません。ここで用いる`LAMBDA = 0.25`は、この30銘柄の例ではその範囲内の密度を与えます。`M`を変更する場合は、閾値の見直しが必要になることがあります。

# %%
rho = train.corr().to_numpy()
for lam in np.arange(0.10, 0.65, 0.05):
    adjacency = (np.abs(rho) > lam) & ~np.eye(M, dtype=bool)
    candidate = nx.from_numpy_array(adjacency)
    print(f"lambda={lam:.2f}  density={nx.density(candidate):.3f}")

G = nx.Graph()
G.add_nodes_from(range(M))
nx.set_node_attributes(G, {i: t for i, t in enumerate(tickers)}, "ticker")
for i in range(M):
    for j in range(i + 1, M):
        if abs(rho[i, j]) > LAMBDA:
            G.add_edge(i, j, weight=1.0 - abs(rho[i, j]))

assert G.number_of_nodes() == M and G.number_of_edges() > 0
print(f"nodes={M}, edges={G.number_of_edges()}, density={nx.density(G):.3f}")
print(f"isolated: {[tickers[v] for v in G if G.degree(v) == 0]}")

pos = nx.spring_layout(G, seed=1)
plt.figure(figsize=(6, 5))
nx.draw(
    G,
    pos,
    node_size=250,
    node_color="white",
    edgecolors="black",
    width=[2 * d["weight"] for _, _, d in G.edges(data=True)],
)
plt.title(f"Market graph (lambda={LAMBDA})")
plt.show()

# %% [markdown]
# ### `@qkernel`によるアンザッツの実装
#
# ここで用いるハードウェア効率の良いアンザッツ（hardware-efficient ansatz、HEA）は、単一量子ビットの回転と固定のエンタングリング層で構成します。一様重ね合わせ状態から始め、$R_y$回転と$R_z$回転の後にCXエンタングリング層を適用する操作を繰り返します。`DEPTH = None`を指定すると、各分割で$p=\max(1,\lfloor m_a/n\rfloor)$層を使います。整数を指定すると、層数をその値に固定します。
#
# `@qkernel`では、量子ビット数、層数、パラメータベクトルの長さを定数として宣言し、回転角をパラメータとして扱います。$p$層のそれぞれで$2n$個の回転角を使うため、`count = 2 * nq * layers`となります。

# %%
DEPTH = None


@qm.qkernel
def hea(
    width: qm.Const[qm.Int],
    layers: qm.Const[qm.Int],
    count: qm.Const[qm.Int],
    thetas: qm.Param[qm.Vector[qm.Real, count]],  # noqa: F821
) -> qm.Q[qm.Bits[width]]:  # noqa: F821
    q = qm.qubits(width)
    q = qm.h(q)
    for d in range(layers):
        offset = d * 2 * width
        for i in range(width):
            q = qm.ry(q, thetas[offset + i], target=i)
        for i in range(width):
            q = qm.rz(q, thetas[offset + width + i], target=i)
        q = cx_entangling_layer(q, width)
    return q


executor = qm.Reference()
SHOTS_PER_OBSERVABLE = 2 if docs_test_mode else 1_024
MEASUREMENT_SEED = 100


# %% [markdown]
# ### エンコードと測定に基づく期待値
#
# 孤立ノードを取り除き、残りのノード識別子を連続した添字に対応付けます。`BinaryModel.from_ising`に結合係数$J_{ij}=w_{ij}/2$とオフセット$-W/2$を渡すと、そのエネルギーは式{eq}`eq-portfolio-cut`の符号を反転したものに一致します。`PauliCorrelationEncoding`は、アクティブな変数ごとに1つのオブザーバブルを提供します。
#
# `qm.specialize`で、その分割に用いるHEAの次元を固定します。各オブザーバブルに再利用可能な`Estimator`を用意し、試行するパラメータベクトルを渡して、`qm.Reference()`でそれぞれ1,024回測定して期待値を推定します。オブザーバブルごとに異なる非ゼロのシードを固定し、繰り返し評価したときの再現性を保ちます。この推定にはサンプリングノイズが含まれるため、最適化した分割は厳密な期待値を用いた場合と異なることがあります。


# %%
def encode_subgraph(sub, active):
    """Encode weighted MaxCut using consecutive active-node indices."""
    index = {v: i for i, v in enumerate(active)}
    couplings = {
        (index[u], index[v]): 0.5 * data["weight"]
        for u, v, data in sub.edges(data=True)
    }
    total_weight = sum(data["weight"] for _, _, data in sub.edges(data=True))
    encoding = PauliCorrelationEncoding(
        BinaryModel.from_ising(couplings, constant=-total_weight / 2.0),
        correlator_order=K,
    )
    n_var, nq = len(active), encoding.num_qubits
    assert len(encoding.observables) == n_var
    assert comb(nq, K) * 3**K >= n_var
    assert nq == K or comb(nq - 1, K) * 3**K < n_var
    return encoding, index, total_weight


def sampled_expectations(encoding, n_var):
    """Create a sampled expectation evaluator for a specialized HEA."""
    nq = encoding.num_qubits
    layers = DEPTH if DEPTH is not None else max(1, n_var // nq)
    count = 2 * nq * layers
    program = qm.specialize(hea, width=nq, layers=layers, count=count)
    estimators = tuple(
        Estimator(program, observable) for observable in encoding.observables
    )

    def expectations(theta):
        parameters = {"thetas": tuple(float(v) for v in theta)}
        return [
            estimator.estimate(
                on=executor,
                parameters=parameters,
                shots=SHOTS_PER_OBSERVABLE,
                seed=MEASUREMENT_SEED + i,
            ).value
            for i, estimator in enumerate(estimators)
        ]

    return expectations, count


# %% [markdown]
# ### 損失関数の最適化と結果のデコード
#
# `encoding.model`から、エンコードしたIsingモデルを取得できます。このMaxCutの定式化には1次項がありません。式{eq}`eq-portfolio-loss`の係数のスケールを保つため、グラフの重みからデータ項を計算します。正則化項には、アクティブなノード数と、その時点の部分グラフの重みのスケールを用います。
#
# COBYLAは勾配を使わずに回路パラメータを最適化します。評価回数の上限を`MAXITER = 200`、初期パラメータを生成するシードを`SEED = 42`とします。損失関数を最小化した後、測定から推定した期待値の符号を使って2つのグループに割り当てます。

# %%
MAXITER = 5 if docs_test_mode else 200
SEED = 42


def make_loss(sub, index, expectations, alpha, nu):
    """Build the portfolio paper's relaxed loss with quartic regularization."""

    def loss(theta):
        relaxed = [np.tanh(alpha * value) for value in expectations(theta)]
        data_term = sum(
            data["weight"] * relaxed[index[u]] * relaxed[index[v]]
            for u, v, data in sub.edges(data=True)
        )
        return (
            data_term
            + BETA * nu * (sum(value * value for value in relaxed) / len(relaxed)) ** 2
        )

    return loss


cut_qubits = []


def bipartition(sub):
    """Return a spin label for each node, including isolated nodes."""
    nodes = sorted(sub.nodes())
    active = [v for v in nodes if sub.degree(v) > 0]
    isolated = [v for v in nodes if sub.degree(v) == 0]
    if len(active) < 2:
        return {v: (1 if i < len(nodes) // 2 else -1) for i, v in enumerate(nodes)}

    encoding, index, total_weight = encode_subgraph(sub, active)
    n_var, nq = len(active), encoding.num_qubits
    assert all(len(indices) == 2 for indices in encoding.model.coefficients)
    expectations, count = sampled_expectations(encoding, n_var)
    alpha = float(nq ** (K // 2)) if ALPHA is None else float(ALPHA)
    mean_weight = total_weight / sub.number_of_edges()
    nu = total_weight / 2.0 + mean_weight * (n_var - 1) / 4.0
    loss = make_loss(sub, index, expectations, alpha, nu)
    initial = np.random.default_rng(SEED).uniform(-np.pi, np.pi, count)
    options = {"maxiter": max(MAXITER, count + 2)}
    if docs_test_mode:
        options["f_target"] = float("inf")
    result = minimize(loss, initial, method="COBYLA", options=options)
    values = np.asarray(expectations(result.x))
    assert np.isfinite(result.fun) and np.isfinite(values).all()
    assert np.all(np.abs(values) <= 1.0 + 1e-10)

    labels = {v: (1 if values[index[v]] > 0 else -1) for v in active}
    for i, v in enumerate(isolated):
        labels[v] = 1 if i % 2 == 0 else -1
    cut_qubits.append(nq)
    print(
        f"      variables={n_var:3d} qubits={nq} ({n_var / nq:.1f}x) "
        f"loss={result.fun:+.3f} |<P>|med={np.median(np.abs(values)):.4f}"
    )
    return labels


# %% [markdown]
# ### 再帰的な分割
#
# キューにある次の分割可能なクラスタに分割を適用します。すべての符号が同じになった場合は、成功した分割回数に数えません。

# %%
queue = deque([sorted(G.nodes())])
done = []
remaining = n_splits

while remaining > 0 and queue:
    nodes = queue.popleft()
    if len(nodes) <= 1:
        done.append(nodes)  # 分割できないため、残りの分割回数を他のクラスタに使います。
        continue

    sp = bipartition(G.subgraph(nodes))
    s1 = [v for v in nodes if sp[v] > 0]
    s2 = [v for v in nodes if sp[v] <= 0]

    if not s1 or not s2:
        # 正則化してもすべての符号が一致した場合は、分割の失敗とみなします。
        print(f"  [warn] collapse on {len(nodes)} nodes -- check alpha/beta")
        done.append(nodes)
        continue

    print(f"  split {len(nodes):3d} -> {len(s1):3d} + {len(s2):3d}")
    queue.append(s1)
    queue.append(s2)
    remaining -= 1

clusters = done + list(queue)
print(f"\nclusters: {[len(c) for c in clusters]}")
assert len(clusters) == n_splits + 1
assert all(clusters)
assert sorted(v for cluster in clusters for v in cluster) == list(range(M))
if not docs_test_mode:
    assert max(cut_qubits) == 4
print(f"qubits per optimized cut: {cut_qubits}")

# %% [markdown]
# 分割したグラフを可視化します。

# %%
# 最終的な分割を可視化します。
color = {}
for ci, c in enumerate(clusters):
    for v in c:
        color[v] = ci

plt.figure(figsize=(6, 5))
nx.draw(
    G,
    pos,
    node_size=250,
    edgecolors="black",
    cmap=plt.cm.tab20,
    node_color=[color[v] for v in G.nodes()],
    vmin=0,
    vmax=19,
)
plt.title(f"{len(clusters)} correlation clusters")
plt.show()

# %% [markdown]
# ### 代表銘柄の選択
#
# 訓練期間のリターンだけを使って式{eq}`eq-portfolio-selection`を適用します。選ばれる銘柄はそれぞれ異なるクラスタに属し、評価用のリターンはこの選択に使いません。

# %%
mu = train.mean().to_numpy()
reps = [max(cluster, key=lambda v: mu[v]) for cluster in clusters]
selected_tickers = [tickers[v] for v in reps]
assert len(reps) == len(clusters)
assert len(set(selected_tickers)) == len(reps)
assert all(v in cluster for v, cluster in zip(reps, clusters))
print(f"selected {len(reps)} assets: {selected_tickers}")

# %% [markdown]
# ## 結果
#
# ### 相関の確認
#
# 選んだ銘柄間の相関係数の絶対値の平均を、訓練期間の全対象銘柄における値と比較します。値が低くなれば、分散投資の狙いを裏付ける結果となります。ただし、MaxCutは近似解を用い、リターンが最も高い代表銘柄の選択によって相関が高まる場合もあるため、この指標が改善するとは限りません。


# %%
def mean_absolute_correlation(correlation):
    """Average the upper-triangle off-diagonal absolute correlations."""
    pairs = np.triu_indices(len(correlation), k=1)
    return float(np.abs(correlation[pairs]).mean())


universe_correlation = mean_absolute_correlation(rho)
selected_correlation = mean_absolute_correlation(rho[np.ix_(reps, reps)])
assert np.isfinite([universe_correlation, selected_correlation]).all()
assert 0 <= universe_correlation <= 1 and 0 <= selected_correlation <= 1
print(
    f"mean |rho|: selected={selected_correlation:.3f}, universe={universe_correlation:.3f}"
)

# %% [markdown]
# ### 評価期間におけるポートフォリオの性能
#
# 選んだ銘柄の日次テストリターンを平均し、初期値1,000から複利で資産価値を計算します。これは毎日等ウェイトにリバランスする場合に相当し、取引コストは考慮しません。比較対象では、全`M`銘柄に同じ計算を適用します。リスクフリーレートを0とし、年率換算したシャープレシオを求めます。
#
# ```{math}
# :label: eq-portfolio-sharpe
# S = \sqrt{252}\,\frac{\overline r_p}{\mathrm{std}(r_p)}.
# ```
#
# シャープレシオは、リターンの変動性1単位あたりのリターンを表します。両ポートフォリオで同じ評価日と評価方法を用います。


# %%
def sharpe(r: pd.Series) -> float:
    """Calculate annualized Sharpe with a zero risk-free rate."""
    std = r.std()
    return float(np.sqrt(252) * r.mean() / std) if std > 0 else 0.0


pce_returns = test[selected_tickers].mean(axis=1)
baseline_returns = test.mean(axis=1)
equity = 1000 * (1 + pce_returns).cumprod()
baseline = 1000 * (1 + baseline_returns).cumprod()
assert np.isfinite(sharpe(pce_returns))
assert np.isfinite(sharpe(baseline_returns))
assert np.isfinite(equity).all() and np.isfinite(baseline).all()
assert equity.iloc[-1] > 0 and baseline.iloc[-1] > 0

print(f"PCE      final = {equity.iloc[-1]:8.2f}   sharpe = {sharpe(pce_returns):+.3f}")
print(
    f"baseline final = {baseline.iloc[-1]:8.2f}   sharpe = {sharpe(baseline_returns):+.3f}"
)

plt.figure(figsize=(8, 4))
plt.plot(equity.values, label=f"PCE ({len(reps)} assets)", color="#2696EB")
plt.plot(baseline.values, label=f"baseline ({M} assets)", color="#888", ls="--")
plt.xlabel("test day")
plt.ylabel("portfolio value")
plt.legend()
plt.title("Out-of-sample performance")
plt.show()

# %% [markdown]
# このチュートリアルの設定で実行すると、選んだ銘柄のポートフォリオは全銘柄の比較対象より最終資産価値が高く、シャープレシオも高くなります。一方、訓練期間における相関係数の絶対値の平均は高くなるため、この指標では分散投資の改善を示していません。これらは1つの時系列分割と1つの最適化シードによる結果であり、一般にリターンが向上することや量子優位性の根拠ではありません。古典的なポートフォリオ最適化手法との比較や、論文の250銘柄の実験の再現は行っていません。

# %% [markdown]
# ## まとめ
#
# - `PauliCorrelationEncoding`は、最初の30変数の分割を4量子ビットに対応付けます。この例の後続の分割では、2または3量子ビットを使用します。
# - QamomileのHEAとreferenceによる測定に基づく期待値推定を用いてPCEの損失関数を評価し、グラフの再帰的分割と訓練期間のデータだけによる代表銘柄の選択を行います。
# - この例の7銘柄のポートフォリオは、等ウェイトの比較対象を最終資産価値とシャープレシオで上回りますが、相関の指標では改善を示しません。
# - この実験は、1つのデータ分割とシードによる小規模なシミュレーションです。量子ビット数の削減だけでは、実行時間の優位性や投資価値は示せません。
