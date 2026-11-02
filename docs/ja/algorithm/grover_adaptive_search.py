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
# Grover適応探索（GAS）は、二値変数上の多項式目的関数を最小化する手法です。Groverオラクルに「どの $x$ が $f(x) < y$ を満たすか」という1つの問いを繰り返し投げ、より良い解が見つかるたびに閾値 $y$ を下げていきます {cite:p}`10.22331/q-2021-04-08-428`。
#
# このページでは、Qamomileのimmutableな`BinaryModel`と`GASExperiment`を使って制約なしの**ポートフォリオ選択**問題を解きます。
#
# このチュートリアルは次の構成で進めます。
#
# 1. 多項式目的関数を`BinaryModel`として定式化します。
# 2. 具体的なデータから9資産のインスタンスを作成します。
# 3. `GASExperiment`で現在の閾値に対応するGrover回路を構築します。
# 4. サンプリングして最良の候補を保持し、停止条件を満たすまで繰り返します。GASは単一の回路ではなくハイブリッドなループです。

# %%
# 最新のQamomileをpipでインストールしましょう！
# # !pip install "qamomile[qiskit]"

# %%
import itertools
import math
import os

import numpy as np

import qamomile as qm
from qamomile.optimization import BinaryModel, GASExperiment
from qamomile.qiskit import Aer
from qamomile.stdlib import grover_iteration_count

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# ## 背景
#
# Grover探索は、目的の状態の位相を反転させて区別するオラクルを受け取り、振幅増幅によってその測定確率を高めます。これは判定問題を解く手続きです。固定された判定条件はどの状態が条件を満たすかを示すだけで、どれが最良かは示しません。
#
# GASは判定条件を変更することで、これを最小化に応用します。$f(x) < y$ を満たす候補に対応する量子状態の位相を反転させて区別し、候補をサンプリングします。候補の目的関数値が現在の閾値を下回る場合、古典レイヤーが $y$ をその値に更新します。量子回路が一度に1つの固定された問いに答え、古典レイヤーが閾値を更新するという処理を繰り返します。


# %% [markdown]
# ## 問題設定
#
# このチュートリアルでは、GASを適用する問題例としてポートフォリオ選択問題を扱います。
#
# $n$ 個の資産（株式、債券など）があり、それぞれを購入するかどうかという二値の選択をします。リターンを最大化しつつリスクを最小化したいのですが、次のような対立があります。
#
# - $\mu$ は各資産に期待される収益性を表します
# - $\Sigma$ は資産同士の連動の仕方を表します（相関のある資産はリスクを増幅させます）
# - $q$ はリターンに対してリスクをどれだけ重視するかを制御します
#
# 相関リスクを過度に負うことなく、できるだけ大きな収益を得られる資産の部分集合を選ぶ必要があります。
#
# \begin{equation}
# \min_{x\in \lbrace 0 , 1 \rbrace^n} \big( q x^T \Sigma x - \mu^T x \big)
# \end{equation}


# %% [markdown]
# 以下のインスタンスは9個の資産を持ちます。
#
# | 名前     | 期待収益 | 分散       |
# |---------|-----------------|------------|
# | 資産1 | 22              | 12         |
# | 資産2 | 4               | 15         |
# | 資産3 | 19              | 10         |
# | 資産4 | 3               | 18         |
# | 資産5 | 23              | 14         |
# | 資産6 | 2               | 20         |
# | 資産7 | 5               | 11         |
# | 資産8 | 25              | 16         |
# | 資産9 | 3               | 13         |


# %% [markdown]
# 以下の係数表から、同じimmutableなbinary modelを直接構築します。

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
risk_aversion = 1


def portfolio_model(num_assets: int) -> BinaryModel:
    terms: dict[tuple[int, ...], float] = {}
    for left in range(num_assets):
        terms[(left,)] = float(
            risk_aversion * covariance[left, left] - expected_returns[left]
        )
        for right in range(left + 1, num_assets):
            coefficient = float(2 * risk_aversion * covariance[left, right])
            if coefficient:
                terms[(left, right)] = coefficient
    return BinaryModel.from_hubo(terms, variables=tuple(range(num_assets)))


full_model = portfolio_model(9)
MODEL_ASSETS = 3 if docs_test_mode else 9
model = portfolio_model(MODEL_ASSETS)

assert full_model.num_variables == 9
assert model.num_variables == MODEL_ASSETS

energies = {
    bits: model.energy(bits)
    for bits in itertools.product((0, 1), repeat=model.num_variables)
}
optimal_bits, optimal_energy = min(energies.items(), key=lambda item: item[1])

print("exact optimum:", optimal_bits, optimal_energy)
assert optimal_bits == ((1, 0, 1) if docs_test_mode else (1, 0, 0, 0, 1, 0, 0, 1, 0))
assert optimal_energy == (-11.0 if docs_test_mode else -28.0)

full_energies = {
    bits: full_model.energy(bits)
    for bits in itertools.product((0, 1), repeat=full_model.num_variables)
}
full_optimal_bits, full_optimal_energy = min(
    full_energies.items(), key=lambda item: item[1]
)
print("nine-asset exact optimum:", full_optimal_bits, full_optimal_energy)
assert full_optimal_bits == (1, 0, 0, 0, 1, 0, 0, 1, 0)
assert full_optimal_energy == -28.0

# %% [markdown]
# ## アルゴリズム
#
# GASでは、候補解の目的関数値を閾値 $y$ として設定し、それを下回る候補をGrover探索で探します。量子回路から得られた候補を古典計算で評価し、より良い候補が見つかれば閾値を更新するという処理を、停止条件を満たすまで繰り返します。ここでの $y$ は候補解ではなく、目的関数値に対する閾値です。GASは、判定条件 $f(x) < y$ を満たす候補に対応する量子状態の位相を反転させ、ほかの候補と区別します。Groverアンザッツは次の3つの要素から構成されます。
#
# - $A_y$：準備演算子です。量子辞書の状態 $\sum_x \ket{x, f(x) - y}$ を構築することで、各入力に $f(x) - y$ を対応付けます。回路実装は {cite:p}`10.22331/q-2021-04-08-428` により、QFTを用いる方法で与えられています。レジスタは $f(x) - y$ を2の補数で保持するため、判定条件を満たす候補は、最上位ビット（MSB）が $1$ になるものとして識別できます。
# - $O_y$：位相オラクルです。判定条件 $f(x) < y$ を満たす候補の位相を反転させます。符号化された値 $f(x) - y$ が負かどうかは最上位ビット（MSB）で判別できるため、そのMSBに作用する1つの $Z$ ゲートで実装できます。
# - $D$：拡散演算子で、位相を反転させて区別した状態の振幅を増幅します。$X$ 層に挟まれた1つの多重制御 $Z$ ゲートで構成されます。
#
# 1回の反復では $O_y$ を適用し、続いて $A_y^\dagger$、$D$、$A_y$ を適用します。位相反転 $O_y$ だけでは測定確率は変わりません。それを振幅へ変換するのが反射 $A_y D A_y^\dagger$ です。入力レジスタの測定で得られた候補を古典レイヤーで評価し、その目的関数値が現在の閾値を下回る場合にのみ、$y$ をその値に更新します。
#
# 残る問題は、Grover演算子を何回適用するかです。GASはこれに対して、改善が得られなかったラウンドのたびに少しずつ広がる範囲から繰り返し回数をサンプリングすることで答えます。


# %% [markdown]
# ## 1回の固定閾値ラウンドの構築
#
# 現在の古典的な候補のenergyが$-10$だとします。次のオラクルは$f(x) < -10$を満たす割り当てをmarkします。documentation testでは3資産の厳密解だけがこの条件を満たします。通常実行では元の9資産問題に同じ固定閾値ラウンドを適用します。
#
# `GASExperiment`はhost側のexperiment境界であり、2つ目の量子decoratorではありません。modelと閾値をverified `Program`へ閉じます。回路は通常のQamomile合成と可逆整数算術だけで構成され、GAS専用のIR演算やbackend実装はありません。

# %%
threshold = -10.0
experiment = GASExperiment(
    model,
    threshold=threshold,
    iterations=1,
)

print("output bits:", experiment.resolved_output_bits)
print("program digest:", experiment.program.semantic_digest)

inspection = experiment.program.inspect()
assert "quantum.gas" not in inspection
assert "qcore.control_call" in inspection

# %% [markdown]
# ### 振幅増幅scalingの再現
#
# [](https://doi.org/10.48550/arXiv.2310.03011)のSection 14.1は、初期成功振幅
# $a$に対して$m=O(a^{-1})$回のGrover iterationを与えます。$N=2^n$個のうち
# 1個がmarked itemなら$a=N^{-1/2}$なので$m=O(\sqrt N)$です。Qamomileの
# bounded plannerはfloating-point overflowなしに通常の
# $\lfloor(\pi/4)\sqrt{2^n}\rfloor$を返します。

# %%
grover_plans = {width: grover_iteration_count(width) for width in range(2, 11)}
for width, iterations in grover_plans.items():
    assert iterations == math.floor((math.pi / 4.0) * math.sqrt(2**width))

print("Grover plans (n: iterations):", grover_plans)

# %% [markdown]
# 固定したGAS thresholdでは、state preparation、oracle、inverse、reflectionから
# なるbody全体をGrover iterationごとに1回繰り返します。compilerはこのbodyから
# exactなsemantic workを導出します。controlled coherent-arithmetic transformには
# 完全なlogical gate profileがまだないため、gate-familyの上限は意図的にopenです。
# `GASExperiment`は意図的にclosed programを公開するため、2つのendpoint studyから
# affineな切片と傾きを求めます。残りの表示rowは算術であり、compiler解析の繰り返し
# ではありません。

# %%
gas_endpoint_studies = tuple(
    qm.resources.logical(
        GASExperiment(model, threshold=threshold, iterations=iterations).program
    )
    for iterations in (0, 1)
)
gas_endpoint_work = tuple(
    study.metric(qm.resources.LOGICAL_WORK).numeric_enclosure()
    for study in gas_endpoint_studies
)
gas_endpoint_gate_bounds = tuple(
    study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure()
    for study in gas_endpoint_studies
)

assert all(lower == upper for lower, upper in gas_endpoint_work)
assert all(upper is None for _, upper in gas_endpoint_gate_bounds)
work_base = gas_endpoint_work[0][0]
work_step = gas_endpoint_work[1][0] - work_base
gate_base = gas_endpoint_gate_bounds[0][0]
gate_step = gas_endpoint_gate_bounds[1][0] - gate_base
gas_work = tuple(
    (work_base + iterations * work_step, work_base + iterations * work_step)
    for iterations in range(4)
)
gas_gate_bounds = tuple(
    (gate_base + iterations * gate_step, None) for iterations in range(4)
)

print("GAS semantic work for 0..3 iterations:", gas_work)
print("GAS logical-gate bounds for 0..3 iterations:", gas_gate_bounds)

# %% [markdown]
# loop構造により、この具体的Grover bodyのcostは$m$についてaffineであり、2つの
# studyがその有限係数を与えます。$m=\Theta(\sqrt N)$と合成するとサーベイのquery
# scalingを再現します。部分的な
# gate enclosureも重要なevidenceです。Qamomileは欠けているtransform profileを
# name付きで報告し、zeroやexactなtarget gate countとして扱いません。

# %% [markdown]
# 整数係数は厳密にencodeされます。実数係数を使う場合、`GASExperiment`はmodel全体と閾値を同じscaleで量子化し、`effective_model`と`quantization_scale`を公開します。非整数算術を拒否するには`approximate_real_coefficients=False`を指定します。通常、`output_bits`は安全な2の補数範囲から推論されます。明示したregister幅が不足する場合、暗黙にwrapせずエラーになります。

# %% [markdown]
# ## 1度コンパイルしてsampling
#
# このexperimentはほかのQamomileプログラムと同じlifecycleを使います。closed programをendpointのtarget向けにコンパイルし、そのartifactを再利用します。documentation testでは組み込みReference endpointを使い、通常実行では元のtutorialと同じくlocal Qiskit Aerを使います。

# %%
endpoint = qm.Reference() if docs_test_mode else Aer()
compiled = experiment.compile(to=endpoint.target)
shots = 32 if docs_test_mode else 256
run = compiled.run(on=endpoint, shots=shots, seed=7)

for sample in run.samples.samples:
    print(sample.assignment, sample.energy, sample.occurrences)

best = run.samples.lowest
assert best.assignment == tuple(enumerate(optimal_bits))
assert best.energy == optimal_energy
if docs_test_mode:
    assert best.occurrences > shots * 0.75
assert run.execution.artifact is compiled.artifact
assert run.identity == compiled.identity == experiment.identity

# %% [markdown]
# `run.samples`は型付きの`DecodedSamples`値です。元の変数ID、目的関数値、出現回数を保持し、backend adapterが最適化modelを再解釈することはありません。
#
# 振幅増幅の効果を確認するため、同じ閾値でGrover反復を0回にした結果と比較します。このプログラムはdecision registerを一様重ね合わせにしますが、振幅増幅は行いません。

# %%
baseline = GASExperiment(model, threshold=threshold, iterations=0).run(
    on=endpoint,
    shots=shots,
    seed=7,
)

baseline_optimum_count = next(
    (
        sample.occurrences
        for sample in baseline.samples.samples
        if sample.assignment == best.assignment
    ),
    0,
)

print("uniform optimum count:", baseline_optimum_count)
print("amplified optimum count:", best.occurrences)
assert best.occurrences > baseline_optimum_count

# %% [markdown]
# ## 適応的な外側のループ
#
# GASが適応的なのは、samplingした各ラウンドの後でhostが新しい閾値を選ぶためです。一般的なループは次の手順で進みます。
#
# 1. 現在の古典的な候補を評価し、そのenergyを`threshold`に使います。
# 2. boundedなGrover反復回数を選びます。
# 3. 対応するimmutableな`GASExperiment`を作成してコンパイルします。
# 4. samplingし、`run.samples.lowest`を確認します。候補が改善した場合だけ閾値を下げます。
#
# 閾値と反復回数は回路構造を変えるため、launch時の`Param`値ではなく明示的なexperiment設定です。1回のラウンド内では、生成したartifactを再利用でき、厳密なproblem/program identityを保持します。


# %%
def adaptive_search(
    binary_model: BinaryModel,
    *,
    initial_threshold: float,
    max_rounds: int,
    shots_per_round: int,
) -> tuple[tuple[tuple[int, int], ...], float, tuple[float, ...]]:
    threshold = initial_threshold
    best_assignment = tuple((index, 0) for index in range(binary_model.num_variables))
    history = [threshold]

    for round_index in range(max_rounds):
        round_experiment = GASExperiment(
            binary_model,
            threshold=threshold,
            iterations=1,
        )
        round_result = round_experiment.run(
            on=endpoint,
            shots=shots_per_round,
            seed=11 + round_index,
        )
        candidate = round_result.samples.lowest
        if candidate.energy < threshold:
            threshold = candidate.energy
            best_assignment = candidate.assignment
        history.append(threshold)
        if threshold == optimal_energy:
            break

    return best_assignment, threshold, tuple(history)


adaptive_assignment, adaptive_energy, threshold_history = adaptive_search(
    model,
    initial_threshold=0.0,
    max_rounds=4,
    shots_per_round=shots,
)

print("threshold history:", threshold_history)
assert adaptive_assignment == tuple(enumerate(optimal_bits))
assert adaptive_energy == optimal_energy
assert all(
    next_threshold <= current_threshold
    for current_threshold, next_threshold in zip(
        threshold_history,
        threshold_history[1:],
    )
)

# %% [markdown]
# 実行したsearchは厳密解へ到達し、単調非増加の閾値履歴を記録します。通常実行では元の9資産の定式化を全体で使い、documentation testだけがmodel幅と実行budgetを縮小します。

# %% [markdown]
# ## まとめ
#
# このノートブックでは、次のことを行いました。
#
# - 制約なしのポートフォリオ問題をGrover適応探索で解きました。量子回路が「どの $x$ が $f(x) < y$ を満たすか」に答え、古典ループがより良い候補を得たときに閾値 $y$ を更新します。
# - GASをQamomileで実装する例を紹介しました。量子側の処理には`GASExperiment`を使います。`compile()`が現在の閾値とGrover反復回数に対応するGrover回路を構築し、`run()`がsamplingしたビット列を元の変数順へdecodeします。
# - ポートフォリオ選択問題を例にGASを実行し、得られた目的関数値が全探索による最適値と一致することを確認しました。
