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
# tags: [algorithm, optimization, oracle-based]
# ---
#
# # 組合せ多項式二値最適化のためのGrover適応探索
#
# このチュートリアルでは、Grover適応探索（Grover Adaptive Search、GAS）で小さな**ポートフォリオ選択**問題を解きます。9つの資産のそれぞれについて買うかどうかを決め、リスクと期待リターンを組み合わせたコストをできるだけ小さくします。コストは二値変数の多項式で、制約はありません。
#
# GASは、1つのステップを繰り返して、このようなコストの最小値を探します（[Gilliam, Woerner and Gonciulea, Quantum 5, 428 (2021)](https://doi.org/10.22331/q-2021-04-08-428)）。GASは閾値$y$を保持します。これは、それまでに見つかった最小のコストです。Groverの探索アルゴリズムを用いてどの割り当てによるコストが$y$より小さいかという問いを繰り返し投げます。測定でコストが$y$より小さい割り当てが得られると、そのコストが新しい閾値になります。古典的なループが、閾値が改善しなくなるまでこれを繰り返します。
#
# このチュートリアルは次の手順で進みます。
#
# 1. コストを`BinaryModel`として書き、すべての割り当てを試して厳密な最小値を求めます。
# 2. Qamomileの`GASExperiment`で、閾値を固定した1ラウンド分の回路を作ります。
# 3. 1ラウンドを実行し、そのサンプルをGroverのアルゴリズムの予測と比べます。
# 4. 閾値を下げていく古典的なループを書き、その結果を厳密な最小値と比べます。
#
# **前提知識**: Groverのアルゴリズムと[`BinaryModel`の使い方](../usage/binary_model.ipynb)。

# %% [markdown]
# ## インストール

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install qamomile

# %%
import functools
import itertools
import math
import os

import numpy as np

import qamomile as qm
from qamomile.optimization import BinaryModel, GASExperiment

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# ## 問題設定
#
# ### 問題の定義
#
# $n$個の資産のそれぞれについて、買うかどうかを決めます。資産$i$を買うなら$x_i = 1$、買わないなら$x_i = 0$です。*割り当て*$\boldsymbol{x} = (x_0, \dots, x_{n-1})$は、1つのポートフォリオを表します。ベクトル$\boldsymbol{\mu}$は各資産の期待リターンを並べたもので、行列$\Sigma$はリターンの共分散行列です。次の*コスト*が最小になる割り当てを探します。
#
# $$
# f(\boldsymbol{x}) = q\,\boldsymbol{x}^{\mathsf T}\Sigma\,\boldsymbol{x} - \boldsymbol{\mu}^{\mathsf T}\boldsymbol{x}
# $$
#
# 第1項は、ポートフォリオのリターンの分散に*リスク回避度*$q \ge 0$を掛けたものです。分散はリスクの大きさを表し、$q$は期待リターンに比べてリスクにどれだけ強いペナルティを課すかを表します。第2項はポートフォリオの期待リターンです。制約はなく、$2^n$通りの割り当てのすべてが許されます。

# %% [markdown]
# ### 使用するインスタンス
#
# インスタンスは$n = 9$個の資産と$q = 1$からなります。下のコードの配列`expected_returns`と`covariance`が、$\boldsymbol{\mu}$と$\Sigma$です。
#
# `BinaryModel.from_hubo`は、二値変数の多項式を、項ごとに1つのエントリを持つ辞書として受け取ります。キーは、その項で掛け合わされる変数の添字のタプルで、値はその項の係数です。$x_i^2 = x_i$であり$\Sigma$は対称なので、このコストは、資産ごとの項$(q\,\Sigma_{ii} - \mu_i)\,x_i$と、ペア$i < j$ごとの項$2q\,\Sigma_{ij}\,x_i x_j$からなります。たとえば、資産0の項は$(12 - 22)\,x_0 = -10\,x_0$で、エントリは`(0,): -10.0`です。資産0と1の項は$2 \times (-3)\,x_0 x_1 = -6\,x_0 x_1$で、エントリは`(0, 1): -6.0`です。`BinaryModel`では、割り当てのコストを*エネルギー*と呼びます。

# %%
expected_returns = np.array([22, 4, 19, 3, 23, 2, 5, 25, 3], dtype=int)
covariance = np.array(
    [
        [12, -3, 4, 0, -2, 3, 0, 2, -1],
        [-3, 15, 0, 5, 1, -4, 2, 0, 3],
        [4, 0, 10, -6, 3, 2, -1, 4, 0],
        [0, 5, -6, 18, -4, 0, 3, -2, 5],
        [-2, 1, 3, -4, 14, 2, -3, 0, 2],
        [3, -4, 2, 0, 2, 20, 4, -3, 1],
        [0, 2, -1, 3, -3, 4, 11, 2, -2],
        [2, 0, 4, -2, 0, -3, 2, 16, -4],
        [-1, 3, 0, 5, 2, 1, -2, -4, 13],
    ],
    dtype=int,
)
RISK_AVERSION = 1

NUM_ASSETS = 3 if docs_test_mode else 9


def portfolio_model(num_assets: int) -> BinaryModel:
    """Build the cost of the first `num_assets` assets as a `BinaryModel`."""
    terms = {}
    for i in range(num_assets):
        # x_i * x_i = x_iなので、資産iの分散は1次の項に入ります。
        terms[(i,)] = float(RISK_AVERSION * covariance[i, i] - expected_returns[i])
        for j in range(i + 1, num_assets):
            # 行列は対称なので、ペア(i, j)と(j, i)から係数2が出ます。
            coefficient = float(2 * RISK_AVERSION * covariance[i, j])
            if coefficient != 0:
                terms[(i, j)] = coefficient
    return BinaryModel.from_hubo(terms, variables=tuple(range(num_assets)))


model = portfolio_model(NUM_ASSETS)

print(f"Number of assets: {model.num_variables}")
assert model.num_variables == NUM_ASSETS

# %% [markdown]
# ### 厳密解（全探索）
#
# 資産が9個なら割り当ては$2^9 = 512$通りしかないので、すべての割り当てのコストを計算できます。これで厳密な最小値が求まり、GASの結果を比べる基準になります。`model.energy(bits)`は割り当てのコストを返します。

# %%
# すべての割り当てのコストを計算します。
costs = {
    bits: model.energy(bits) for bits in itertools.product((0, 1), repeat=NUM_ASSETS)
}

# 最小のコストと、それを達成するすべての割り当てを求めます。
min_cost = min(costs.values())
optimal_assignments = [bits for bits, cost in costs.items() if cost == min_cost]

print(f"Number of assignments: {len(costs)}")
print(f"Smallest cost: {min_cost}")
for bits in optimal_assignments:
    print(f"Optimal assignment: {bits}")
print(f"Largest cost: {max(costs.values())}")

if not docs_test_mode:
    assert min_cost == -28.0
    assert optimal_assignments == [(1, 0, 0, 0, 1, 0, 0, 1, 0)]

# %% [markdown]
# 最小のコスト$-28$を達成するポートフォリオは、資産0、4、7の組だけです。この組では、分散が$12 + 14 + 16 + 2 \times (-2 + 2 + 0) = 42$、期待リターンが$22 + 23 + 25 = 70$なので、コストは$42 - 70 = -28$です。コストは$-28$から$94$までの値をとります。何も買わない割り当て、つまりすべてが0の割り当てのコストは0です。

# %% [markdown]
# ## アルゴリズム
#
# GASは、目標が動いていくGroverのアルゴリズムです。閾値$y$、つまりそれまでに見つかった最良の割り当てのコストを保持し、Groverのアルゴリズムで$f(\boldsymbol{x}) < y$を満たす割り当てを探します。見つかると、その割り当てのコストが新しい閾値になります。この節は、Groverのアルゴリズムを知っていることを前提に、GASがそこに加えるものを説明します。

# %% [markdown]
# ### GASが使うGroverのアルゴリズムの要素
#
# $N = 2^n$通りの割り当てのうち、$M$個がある条件を満たすとします。Groverのアルゴリズムは、一様な重ね合わせから始めて、1回の*反復*を繰り返します。反復は、条件を満たすすべての割り当ての振幅の符号を変える*オラクル*と、それに続く開始状態についての反射からなります。$r$回の反復のあと、測定で条件を満たす割り当てが得られる確率は次のとおりです（[Boyer, Brassard, Høyer and Tapp, arXiv:quant-ph/9605034](https://doi.org/10.48550/arXiv.quant-ph/9605034)）。
#
# $$
# P_r = \sin^2\big((2r+1)\,\theta\big), \qquad \sin^2\theta = M/N
# $$
#
# $r$を増やすと、この確率は上がったあと再び下がるので、最適な反復回数は$M$によって決まります。

# %% [markdown]
# ### $f(\boldsymbol{x}) < y$のオラクル
#
# 回路には2つのレジスタがあります。割り当てを表す$n$個の*決定量子ビット*と、整数を2の補数で保持する*値レジスタ*です。準備の操作$A_y$は、決定量子ビットを一様な重ね合わせにし、値レジスタに$f(\boldsymbol{x}) - y$を書き込みます（[Gilliam, Woerner and Gonciulea, Quantum 5, 428 (2021)](https://doi.org/10.22331/q-2021-04-08-428)）。
#
# $$
# A_y\,|0\rangle = \frac{1}{\sqrt{N}} \sum_{\boldsymbol{x}} |\boldsymbol{x}\rangle\,|f(\boldsymbol{x}) - y\rangle
# $$
#
# 2の補数では、値レジスタの最上位の量子ビットは、負の数のとき、つまり$f(\boldsymbol{x}) < y$のときに限って1になります。したがって、オラクルはこの量子ビットへの$Z$ゲート1つです。開始状態についての反射の代わりに、回路は$A_y D_0 A_y^\dagger$を適用します。ここで$D_0$は、すべての決定量子ビットが状態$|0\rangle$にあるとき振幅に$-1$を掛け、それ以外では何もしません。この積は因子$-1$を除いて反射に等しく、この因子は確率を変えません。
#
# 値レジスタには、すべての$f(\boldsymbol{x}) - y$を保持できる幅が必要です。幅が足りないと、値が折り返して符号が誤ります。`GASExperiment`は係数から幅を決めます。

# %% [markdown]
# ### 適応ループ
#
# 1回の*ラウンド*では、現在の閾値と反復回数$r$で回路を実行し、決定量子ビットを測定します。古典的なループは次のとおりです。
#
# 1. ある割り当て$\boldsymbol{x}$から始め、$y = f(\boldsymbol{x})$とします。
# 2. $r$を選び、閾値$y$でGrover反復を$r$回行い、最後に決定量子ビットを測定します。
# 3. 測定で得た割り当てのコストが$y$より小さければ、それを新しい$\boldsymbol{x}$とし、そのコストを新しい$y$とします。
# 4. 手順2から繰り返し、コストを更新できないラウンドが所定の回数続いたら終了します。
#
# 閾値より下にある割り当ての数$M$は未知で、$y$とともに変わるので、固定した$r$ではうまくいきません。Gilliamらにならい、ループは$r$を$0, \dots, \lceil k \rceil - 1$から一様に選びます。最初は$k = 1$です。改善のなかったラウンドのあとは$k$に因子$\lambda > 1$を掛け、改善があれば$k$を1に戻します。最小値では条件を満たす割り当てがなく、どのラウンドも改善できません。そのため、改善のないラウンドが決まった回数だけ続いたら、ループを止めます。結果はそれまでに見た最良の割り当てであり、最適であるという証明はありません。

# %% [markdown]
# ## Qamomileによる実装
#
# ### 1ラウンドの構築
#
# `GASExperiment`は、モデル、閾値、反復回数から、1ラウンド分のプログラムを作ります。プログラムは、準備の操作$A_y$、`iterations`回のオラクルと反射、そして測定からなります。ここでは閾値を$y = 0$とし、反復を1回にします。何も買わないときのコストは$f(0, \dots, 0) = 0$なので、このラウンドは、何も買わないよりも良いポートフォリオを探します。

# %%
# 何も買わないときのコストです。
threshold = 0.0
experiment = GASExperiment(model, threshold=threshold, iterations=1)

print(f"Decision qubits: {model.num_variables}")
print(f"Value register qubits: {experiment.resolved_output_bits}")

# プログラム全体の量子ビット数を、実行せずに数えます。
study = qm.resources.logical(experiment.program)
num_qubits, _ = study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure()
print(f"Qubits of one round: {num_qubits}")

# %% [markdown]
# 1ラウンドは18個の量子ビットを使います。内訳は、決定量子ビットが9個、値レジスタの量子ビットが9個です。`qm.resources.logical`は、プログラムを実行せずにこの数を読み取ります（[リソース推定](../tutorial/04_resource_estimation.ipynb)を参照）。反復回数を変えても、この数は変わりません。
#
# :::{note}
# ここでは係数が整数なので、値レジスタはすべての$f(\boldsymbol{x}) - y$を正確に保持します。係数が実数の場合、`GASExperiment`はモデルと閾値をまとめて量子化し、その結果を`effective_model`と`quantization_scale`として公開します。
# :::

# %% [markdown]
# ### コンパイルと実行
#
# ここでは、Qamomile組み込みのシミュレータ`qm.Reference()`を量子コンピュータの代わりに使います。このエンジン向けにプログラムをコンパイルし、1*ショット*だけ実行します。1ショットでは、回路を1回準備して1回測定し、割り当てが1つ得られます。

# %%
endpoint = qm.Reference()

compiled = experiment.compile(to=endpoint.target)
run = compiled.run(on=endpoint, shots=1, seed=7)

sample = run.samples.lowest
bits = tuple(bit for _, bit in sample.assignment)
print(f"Measured assignment: {bits}")
print(f"Cost: {sample.energy}")

# %% [markdown]
# `run.samples`は、測定された割り当てを保持します。1ショットではサンプルは1つで、`run.samples.lowest`がそれを返します。サンプルの`assignment`は各変数とその値の組で、`energy`はその割り当てのコストです。

# %% [markdown]
# ## 結果
#
# ### 閾値を固定した1ラウンド
#
# $y = 0$では、512通りの割り当てのうち104個のコストが閾値より小さくなります。反復回数0、1、2のそれぞれについて、1ショットのラウンドを256回繰り返し、閾値より下になったラウンドの割合を$P_r$と比べます。閾値と反復回数が変わると回路が変わるので、その組み合わせごとにコンパイル済みのプログラムが必要です。`compiled_round`は、それぞれを1回だけ作って使い回します。

# %%
NUM_ROUNDS = 4 if docs_test_mode else 256


@functools.cache
def compiled_round(threshold: float, iterations: int):
    """Compile the round for one threshold and one number of iterations."""
    round_experiment = GASExperiment(model, threshold=threshold, iterations=iterations)
    return round_experiment.compile(to=endpoint.target)


def run_round(threshold: float, iterations: int, seed: int):
    """Run one round with one shot and return the measured assignment and its cost."""
    run = compiled_round(threshold, iterations).run(on=endpoint, shots=1, seed=seed)
    sample = run.samples.lowest
    return tuple(bit for _, bit in sample.assignment), sample.energy


num_marked = sum(cost < threshold for cost in costs.values())
theta = math.asin(math.sqrt(num_marked / len(costs)))

for iterations in range(3):
    measured = [run_round(threshold, iterations, seed)[1] for seed in range(NUM_ROUNDS)]
    below = sum(cost < threshold for cost in measured)
    predicted = math.sin((2 * iterations + 1) * theta) ** 2
    print(
        f"r = {iterations}: {below} of {NUM_ROUNDS} rounds below the threshold, "
        f"share {below / NUM_ROUNDS:.3f}, predicted {predicted:.3f}"
    )
    if not docs_test_mode:
        assert abs(below / NUM_ROUNDS - predicted) < 0.1

# %% [markdown]
# 反復なしのラウンドは割り当てをランダムに選ぶのと同じで、閾値より下になったラウンドの割合は0.215です。反復が1回だと割合は0.965に上がり、2回だと0.559に下がります。これは$P_r$の予測どおりです。最適な反復回数は、閾値より下にある割り当ての数で決まりますが、ループはその数を知りません。

# %% [markdown]
# ### 適応ループの実行
#
# 次に、アルゴリズムの節のループを実行します。各ラウンドは`run_round`の1回の呼び出しです。ループは何も買わない割り当てから始め、反復回数をランダムに選び、改善のないラウンドが`MAX_FAILURES`回続いたら止まります。

# %%
GROWTH = 1.2  # 改善のなかったラウンドのあとに範囲へ掛ける因子λです。
MAX_FAILURES = 2 if docs_test_mode else 20


def adaptive_search(seed: int):
    """Run the adaptive loop and return the best assignment, its cost and the history."""
    rng = np.random.default_rng(seed)
    best_bits = (0,) * NUM_ASSETS  # 何も買わない割り当てから始めます。
    threshold = model.energy(best_bits)
    k, failures, rounds = 1.0, 0, 0
    history = []

    while failures < MAX_FAILURES:
        rounds += 1
        # 反復回数を0, ..., ceil(k) - 1から一様に選びます。
        iterations = int(rng.integers(math.ceil(k)))
        bits, cost = run_round(threshold, iterations, seed=int(rng.integers(2**31)))
        if cost < threshold:
            best_bits, threshold = bits, cost
            history.append((rounds, iterations, threshold))
            k, failures = 1.0, 0
        else:
            k *= GROWTH
            failures += 1

    return best_bits, threshold, history, rounds


best_bits, best_cost, history, num_rounds = adaptive_search(seed=0)

for round_number, iterations, new_threshold in history:
    print(f"Round {round_number}: r = {iterations}, new threshold {new_threshold}")
print(f"Stopped after {num_rounds} rounds")
print(f"Best assignment: {best_bits}, cost {best_cost}")

assert best_cost == model.energy(best_bits)
if not docs_test_mode:
    assert best_cost == min_cost

# %% [markdown]
# 閾値は0から$-4$、$-11$、$-13$、$-16$、$-24$、$-25$と下がり、50ラウンド目に$-28$に達します。続く20ラウンドでは改善がなく、70ラウンド目でループが止まります。結果は、唯一の最適な割り当てです。ループは厳密な最小値を使っておらず、最後に比べているだけです。実行によっては、最小値より上で止まることもあります。`MAX_FAILURES = 20`のとき、$P_r$を使った計算では、このインスタンスの実行1,000回のうち約997回が最小値に達します。

# %% [markdown]
# ## まとめ
#
# このチュートリアルでは、Grover適応探索でポートフォリオ選択問題を解きました。
#
# 1. コストを`BinaryModel`として書き、すべての割り当てを試して厳密な最小値を求めました。
# 2. `GASExperiment`で閾値を固定した1ラウンドを作り、閾値より下になるラウンドの割合が$\sin^2((2r+1)\,\theta)$に従うことを確かめました。
# 3. 適応ループを実行しました。ループは改善のたびに閾値を下げ、反復回数をランダムに選び、厳密な最小値に達しました。
