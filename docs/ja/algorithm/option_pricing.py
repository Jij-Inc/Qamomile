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
# tags: [algorithm, finance, simulation]
# ---
#
# # 量子振幅推定を用いたオプションプライシング
#
# ゲート型量子コンピュータ上で実行する量子振幅推定は、様々な応用が提案されています。
# その中でも、金融工学のオプションプライシングは、その代表的な応用例の一つです。
# これまでオプションプライシングには古典モンテカルロ法が用いられてきましたが、これは目標精度 $\epsilon$ を達成するための必要なオラクルの呼び出し回数が $\mathcal{O} (\epsilon^{-2})$ でした。
# これに対し、量子振幅推定は $\mathcal{O} (\epsilon^{-1})$ で済み、2次の高速化をもたらすアルゴリズムであることが知られています。
# そこで本記事では、量子振幅推定によるオプションおよびオプションポートフォリオプライシング手法{cite:p}`10.22331/q-2020-07-06-291`の理解と実装についてまとめました。オプションプライシングの例を通して、Qamomileの使い方を学びましょう。
#

# %% [markdown]
# ## インストール
#
# Qamomile distributionをインストールします。
#
# ```bash
# pip install qamomile
# ```

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import math
import os

import numpy as np

import qamomile as qm
from qamomile.stdlib import (
    amplitude_angles,
    amplitude_encoding,
    amplitude_encoding_from_angles,
    uniformly_controlled_angles,
    uniformly_controlled_ry,
    uniformly_controlled_ry_from_angles,
)

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# ## 背景
#
# 最初に、この記事で出てくる用語についてまとめておきましょう。
#
# * アセット (原資産): 株式など、オプションの対象となる金融商品
# * オプション: 将来の決められた時点 (満期) に、決められた価格 (権利行使価格) でアセットを売買する権利
# * ペイオフ: 満期時点でのオプションの価値
#
# ### 問題: オプションプライシング
#
# 金融リスクの計算は、大きな需要があります。
# モンテカルロ (MC) 計算はその中心的な計算手法であり、Value at Risk (VaR) の推定や店頭デリバティブの価格決定など、幅広く用いられています。
# しかしその誤差は $\varepsilon = \mathcal{O} (M^{-1/2})$ で減衰するという欠点があります。
# ここで $M$ はサンプル数です。
# 対象となる確率分布が量子状態として準備される場合、量子振幅推定 (QAE) により、期待値などの統計量を $\mathcal{O} (M^{-1})$ の誤差で推定できることが知られています。
# これは先ほどの古典MCに対し、二次の高速化をもたらすことがわかります。
#
# ### 先行研究
#
# モンテカルロプライシングの量子アルゴリズムの理論的枠組みを示した先行研究として、{cite:p}`10.1103/PhysRevA.98.022321`があります。
# この論文は、プライシング計算において、古典に対し2次高速化が得られることを理論的に示しました。
# そして、特にヨーロピアンオプションとアジアンオプションに対して数値シミュレーションを行い、その有効性を示しました。
# しかし、その成果は理論的な枠組みの提示と数値シミュレーションに留まり、実機での実行は示されていません。
# もう一つの重要な先行研究として、{cite:p}`10.1038/s41534-019-0130-6`があります。
# これはプライシングに対してではなく、VaR や Conditional Value at Risk (CVaR) に QAE を適用した研究です。
# モンテカルロプライシングのペイオフ計算を改良し、必要量子ビット数・ゲート数の大幅な削減に成功しました{cite:p}`10.1103/PhysRevA.98.022321`。
# このペイオフ計算手法をオプションプライシングに拡張した実装も示されています{cite:p}`10.22331/q-2020-07-06-291`。
#
# ## アルゴリズム: QAEによるオプションプライシング計算
#
# ### ペイオフの計算
#
# QAE ではグローバー演算子 $\mathcal{Q}$ を $k$ 回繰り返すことで、測定確率が
# $\sin^2((2k+1)\theta_a)$ という既知の関数形で変化します。
# これにより、少ない測定回数でも $\theta_a$ を高精度に推定することが可能になります。
# 今、アセットの価格を $S_i$とし、それが起こる確率を $p_i$ としましょう。
# 演算子 $\mathcal{A}$ が
#
# $$
# \mathcal{A} \vert 0 \rangle_{n+1}
# = \sum_{i=0}^{2^n - 1} \sqrt{1-f(S_i)} \sqrt{p_i} \vert S_i \rangle \vert 0 \rangle + \sum_{i=0}^{2^n - 1} \sqrt{f(S_i)} \sqrt{p_i} \vert S_i \rangle \vert 1 \rangle \tag{1}
# $$
#
# のような状態を作れるとすると、ペイオフ量子ビットが $\vert 1 \rangle$ の状態で観測される確率は $a = \mathbb{E}[f(S)]$ となり、QAE によりペイオフの期待値を計算できることがわかります。
# 補助レジスタにペイオフをバイナリ表現で格納する手法では、量子ビット数と回路の深さが増大します{cite:p}`10.1103/PhysRevA.98.022321`。
# 軽量な$R_y$手法では、このオーバーヘッドを避けられます{cite:p}`10.22331/q-2020-07-06-291,10.1038/s41534-019-0130-6`。
#
# ### 平均値への適用と $c$ パラメータ
#
# $\sin^2$ の導関数が最大となる $\pi/4$ の周りで展開するために、目標振幅角を
#
# $$
# \tilde{f}(i) = 2 \frac{f(i) - f_\mathrm{min}}{f_\mathrm{max} - f_\mathrm{min}} - 1, \qquad
# \alpha(i) = c\tilde{f}(i) + \frac{\pi}{4} \tag{2}
# $$
#
# と定義します ($c \in [0, 1]$ はスケーリングパラメータ)。
# このときペイオフ量子ビットが $\vert 1 \rangle$ となる確率は
#
# $$
# P_1 \approx \frac{1}{2} + c\,\mathbb{E}[\tilde{f}] + \mathcal{O}(c^3) \tag{3}
# $$
#
# となり、$P_1$ を推定すれば既知の $c, f_\mathrm{min}, f_\mathrm{max}$ から $\mathbb{E}[f(S)]$ を算出できます。
# $P_1$ の推定誤差を $\delta P_1$ とすると、$\mathbb{E}[\tilde{f}] \approx (P_1 - 1/2)/c$ より
# 推定誤差は $\delta P_1/c$ に増幅されます。
# $c$ を小さくするとこの誤差増幅が大きくなり、逆に大きくすると近似バイアス $\mathcal{O}(c^2)$ が増大します。
# このトレードオフから収束率は $\mathcal{O}(M^{-2/3})$ となります。
# 完全な二次高速化には至りませんが、依然として古典 MC の $\mathcal{O}(M^{-1/2})$ を凌駕します。
# 以下の実装では、位相推定を用いない軽量な最尤振幅推定(MLAE)を採用しています。
#
# ### コールオプションへの応用
#
# コールオプションのペイオフ $f_C(S_T) = \max(0, S_T - K)$ は $K$ を境に折れ曲がる区分線形関数です。
# 本実装では均一制御回転(UCR-Y)を用い、各格子点 $S_i$ に対して
#
# $$
# \alpha(S_i) = \frac{\pi}{4} - c + \frac{2c\,\max(S_i - K,\,0)}{S_\mathrm{max} - K} \tag{4}
# $$
#
# という目標振幅角を直接適用します。
# $S < K$ では定数角 $\pi/4 - c$、$S \geq K$ では線形増加という区分線形構造を、正確に符号化することができます。
#

# %% [markdown]
# ## Qamomileによる実装
#
# ### 分布の準備とoptionの離散化
#
# 権利行使価格$K=100$のヨーロピアン・コールを考えます。満期時の価格は、
# リスク中立確率$0.75$で$80$、確率$0.25$で$120$になる2点を残します。
# 現金ペイオフ$\max(S_T-K,0)$を最大値$20$で割ると、正規化ペイオフ表は
# $(0,1)$になります。
#
# 1つのアドレス量子ビットで2つの価格を表し、1つのペイオフ量子ビットで正規化した
# 期待ペイオフを表します。この2量子ビットのwitnessだけで、状態ロード、ペイオフの
# multiplexing、随伴合成、増幅、実行時パラメータbinding、結果decode、資源見積もりを
# 確認できます。gridを増やしても座標vectorが変わるだけで、programming modelは
# 変わりません。

# %%
asset_prices = (80.0, 120.0)
probabilities = (0.75, 0.25)
strike = 100.0
risk_free_rate = 0.05
maturity = 1.0

cash_payoffs = tuple(max(price - strike, 0.0) for price in asset_prices)
payoff_scale = max(cash_payoffs)
normalized_payoffs = tuple(payoff / payoff_scale for payoff in cash_payoffs)
expected_cash_payoff = sum(
    probability * payoff
    for probability, payoff in zip(probabilities, cash_payoffs, strict=True)
)
exact_price = math.exp(-risk_free_rate * maturity) * expected_cash_payoff

assert probabilities == (0.75, 0.25)
assert normalized_payoffs == (0.0, 1.0)
assert expected_cash_payoff == 5.0

# %% [markdown]
# 標準ライブラリの座標変換はRustで実行されます。`amplitude_angles`は確率の平方根を
# とった振幅を正規化し、その位相を保持します。`uniformly_controlled_angles`は、
# 望むbranch回転を、再利用可能なペイオフmultiplexerが使うcyclic Gray-walk座標へ
# 変換します。
#
# 得られたscalarは実行時の`Param`値です。そのため、同じshapeを持つ別の分布や
# ペイオフ表へ型付きbodyを変更せずに、回路を1回compileしてrebindできます。

# %%
distribution = amplitude_angles(
    tuple(math.sqrt(probability) for probability in probabilities)
)
payoff_angles = uniformly_controlled_angles(
    tuple(2.0 * math.asin(math.sqrt(payoff)) for payoff in normalized_payoffs)
)

prepare_distribution = amplitude_encoding_from_angles(1)
encode_payoff = uniformly_controlled_ry_from_angles(1)

parameters = {
    "distribution_ry": distribution.ry,
    "distribution_rz": distribution.rz,
    "distribution_phase": distribution.phase,
    "payoff_angles": payoff_angles,
}

# %% [markdown]
# ### payoffとdistribution loadingの実装
#
# `option_state`は2成分のowned stateを消費します。`split_at`で資産アドレスとペイオフ
# 量子ビットを分け、分布ユニタリでアドレスを準備し、`concat`で成分を結合してから
# ペイオフmultiplexerを適用します。すべての幅と実行時vector長は関数型の一部です。


# %%
@qm.qkernel
def option_state(
    state: qm.Q[qm.Bits[2]],  # type: ignore[valid-type]
    distribution_ry: qm.Param[qm.Vector[qm.Real, 1]],  # type: ignore[valid-type]
    distribution_rz: qm.Param[qm.Vector[qm.Real, 1]],  # type: ignore[valid-type]
    distribution_phase: qm.Param[qm.Real],
    payoff_angles: qm.Param[qm.Vector[qm.Real, 2]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[2]]:  # type: ignore[valid-type]
    """Prepare a two-point call-option payoff amplitude.

    Args:
        state (qm.Q[qm.Bits[2]]): Asset-address and payoff qubits.
        distribution_ry (qm.Param[qm.Vector[qm.Real, 1]]): Runtime magnitude
            coordinate for the risk-neutral distribution.
        distribution_rz (qm.Param[qm.Vector[qm.Real, 1]]): Runtime relative
            phase coordinate for the distribution.
        distribution_phase (qm.Param[qm.Real]): Runtime global phase.
        payoff_angles (qm.Param[qm.Vector[qm.Real, 2]]): Runtime Gray-walk
            coordinates for the normalized payoff.

    Returns:
        qm.Q[qm.Bits[2]]: State whose final component encodes expected payoff.
    """

    asset, payoff = qm.split_at(state, 1)
    asset = prepare_distribution(
        asset,
        distribution_ry,
        distribution_rz,
        distribution_phase,
    )
    return encode_payoff(qm.concat(asset, payoff), payoff_angles)


@qm.qkernel
def option_samples(
    distribution_ry: qm.Param[qm.Vector[qm.Real, 1]],  # type: ignore[valid-type]
    distribution_rz: qm.Param[qm.Vector[qm.Real, 1]],  # type: ignore[valid-type]
    distribution_phase: qm.Param[qm.Real],
    payoff_angles: qm.Param[qm.Vector[qm.Real, 2]],  # type: ignore[valid-type]
    iterations: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Sample the option amplitude after fixed Grover iterations.

    Args:
        distribution_ry (qm.Param[qm.Vector[qm.Real, 1]]): Distribution
            magnitude coordinate.
        distribution_rz (qm.Param[qm.Vector[qm.Real, 1]]): Distribution phase
            coordinate.
        distribution_phase (qm.Param[qm.Real]): Distribution global phase.
        payoff_angles (qm.Param[qm.Vector[qm.Real, 2]]): Payoff multiplexer
            coordinates.
        iterations (qm.Const[qm.Int]): Compile-time Grover iteration count.

    Returns:
        qm.Shot[qm.Bits[2]]: Asset-address and payoff measurement.
    """

    state = option_state(
        qm.qubits(2),
        distribution_ry,
        distribution_rz,
        distribution_phase,
        payoff_angles,
    )
    for _ in range(iterations):
        state = qm.z(state, target=1)
        state = qm.adjoint(
            option_state,
            state,
            distribution_ry,
            distribution_rz,
            distribution_phase,
            payoff_angles,
        )
        state = qm.x(state, target=0)
        state = qm.x(state, target=1)
        state = qm.cz(state, control=0, target=1)
        state = qm.x(state, target=0)
        state = qm.x(state, target=1)
        state = option_state(
            state,
            distribution_ry,
            distribution_rz,
            distribution_phase,
            payoff_angles,
        )
    return qm.measure(state)


# %% [markdown]
# `iterations`は回路構造を変える`Const`であり、specialize時に固定します。分布と
# ペイオフ座標はartifactのinvocation ABIに残る`Param`です。この2段階を混同する、
# `state`を複製する、operationが返すsuccessorを捨てる、といった誤りはbackend
# loweringより前にrejectされます。

# %% [markdown]
# ### 増幅programのinspectionと見積もり
#
# 1反復のprogramは、準備、位相marking、自動導出した随伴、reflection、再度の準備を
# 通常の関数として合成したものです。金融専用のcompiler instructionやbackend pathは
# ありません。

# %%
baseline_program = qm.specialize(option_samples, iterations=0)
amplified_program = qm.specialize(option_samples, iterations=1)
inspection = amplified_program.inspect()
study = qm.resources.logical(amplified_program)

assert "qcore.adjoint_call" in inspection
assert "quantum.option_pricing" not in inspection
assert study.metric(qm.resources.LOGICAL_ROTATION_GATES).numeric_enclosure() == (
    12,
    12,
)
assert study.metric(
    qm.resources.logical_operation_count("quantum.cx")
).numeric_enclosure() == (4, 4)
assert study.metric(
    qm.resources.logical_operation_count("quantum.cz")
).numeric_enclosure() == (1, 1)
assert study.metric(qm.resources.LOGICAL_TWO_QUBIT_GATES).numeric_enclosure() == (
    7,
    7,
)
assert qm.resources.ResourceStudy.from_bytes(study.to_bytes()) == study

# %% [markdown]
# これらのcountは、実際に実行する同じspecialized call graphから得られます。随伴は
# inverse gate listのcopyではなく、estimatorもoption pricing専用の別管理formulaを
# 使いません。

# %% [markdown]
# ### 振幅推定work scalingの再現
#
# [](https://doi.org/10.48550/arXiv.2310.03011)のSection 14.2は、
# $O(1/\epsilon^2)$回のincoherentな古典sampleと$O(1/\epsilon)$回のcoherentな
# walk-operator applicationを比較します。このtutorialは教科書的QPE circuitではなく
# likelihood-based amplitude estimationを使いますが、Grover powerを1つ増やすたびに
# preparation inverse、2個のreflection、1個のpreparationが加わります。resource
# algebraは1度のsymbolic解析からこの有限サイズのaffine costを検証できます。下の
# 各rowではiteration countを代入するだけです。

# %%
amplitude_scaling_study = qm.resources.logical(option_samples)
amplitude_studies = tuple(
    amplitude_scaling_study.evaluate(iterations=iterations) for iterations in range(5)
)
amplitude_scaling = tuple(
    (
        resource_study.metric(qm.resources.LOGICAL_ROTATION_GATES).numeric_enclosure()[
            0
        ],
        resource_study.metric(
            qm.resources.logical_operation_count("quantum.cx")
        ).numeric_enclosure()[0],
        resource_study.metric(
            qm.resources.logical_operation_count("quantum.cz")
        ).numeric_enclosure()[0],
        resource_study.metric(qm.resources.LOGICAL_WORK).numeric_enclosure()[0],
    )
    for resource_study in amplitude_studies
)

for iterations, (rotations, cnot_gates, controlled_z, work) in enumerate(
    amplitude_scaling
):
    assert rotations == 4 + 8 * iterations
    assert cnot_gates == 2 + 2 * iterations
    assert controlled_z == iterations
    assert work == 9 + 20 * iterations

print("amplitude-estimation scaling (k: rotations, CX, CZ, work):", amplitude_scaling)

# %% [markdown]
# 4個の有限metricはすべてGrover powerについてaffineです。したがって、この固定した
# preparationでは$M=\Theta(1/\epsilon)$回のcoherent walkを使うscheduleのlogical
# workは$\Theta(1/\epsilon)$となり、サーベイのquantum側と一致します。このcircuit
# study自体はstatistical success boundを証明せず、classical likelihood optimization
# costも含みません。

# %% [markdown]
# ### quantum amplitude estimationの実装
#
# 同じprogram familyの2つのmemberをReference backend向けにcompileします。baselineは
# $a=\sin^2\theta$を推定し、Groverを1回適用したprogramは$\sin^2(3\theta)$を
# 推定します。厳密値$a=1/4$では後者が1になるため、増幅bodyを明確に検査できます。

# %%
reference = qm.Reference()
baseline_artifact = qm.compile(baseline_program, to=reference)
amplified_artifact = qm.compile(amplified_program, to=reference)

baseline = qm.run(
    baseline_artifact,
    on=reference,
    parameters=parameters,
    shots=128,
    seed=2,
).result
amplified = qm.run(
    amplified_artifact,
    on=reference,
    parameters=parameters,
    shots=4,
    seed=2,
).result

baseline_successes = sum(
    count for bits, count in baseline.counts.items() if bits[1] == 1
)
amplified_successes = sum(
    count for bits, count in amplified.counts.items() if bits[1] == 1
)
observations = (
    (0, baseline_successes, baseline.shots),
    (1, amplified_successes, amplified.shots),
)

assert baseline.shots == 128
assert amplified.counts == {(1, 1): 4}

# %% [markdown]
# ## 結果
#
# ### ホスト側で推定して割り引く
#
# 最尤推定は通常の古典処理です。Pythonに置くことで量子・古典の境界が明確になり、
# 数値optimizerをcompiler semanticsとしてencodeすることも避けられます。観測
# $(h_k,N_k)$に対して次を最大化します。
#
# $$
# \prod_k p_k(\theta)^{h_k}(1-p_k(\theta))^{N_k-h_k},\qquad
# p_k(\theta)=\sin^2((2k+1)\theta).
# $$


# %%
def negative_log_likelihood(theta: float) -> float:
    """Return the binomial negative log likelihood for all Grover powers.

    Args:
        theta (float): Candidate amplitude angle in the open first quadrant.

    Returns:
        float: Negative log likelihood for the measured observations.
    """

    total = 0.0
    for iterations, successes, shots in observations:
        probability = math.sin((2 * iterations + 1) * theta) ** 2
        probability = min(max(probability, 1e-12), 1.0 - 1e-12)
        total -= successes * math.log(probability)
        total -= (shots - successes) * math.log(1.0 - probability)
    return total


theta_candidates = tuple(index * math.pi / (2 * 4096) for index in range(1, 4096))
theta_hat = min(theta_candidates, key=negative_log_likelihood)
amplitude_hat = math.sin(theta_hat) ** 2
payoff_hat = payoff_scale * amplitude_hat
price_hat = math.exp(-risk_free_rate * maturity) * payoff_hat

sampling_error = 6.0 * math.sqrt(0.25 * 0.75 / baseline.shots)
assert abs(amplitude_hat - 0.25) <= sampling_error

print("measured Grover data:", observations)
print(f"estimated normalized payoff: {amplitude_hat:.6f}")
print(f"estimated option price:       {price_hat:.6f}")
print(f"exact discretized price:      {exact_price:.6f}")

# %% [markdown]
# 最後に現金ペイオフscaleを掛け、割引率$e^{-rT}$を掛ける処理は量子programの外側に
# 残します。より細かいgridでは、2つの実行時座標vectorを置き換えてアドレス量子
# ビットを増やしますが、型付きの準備・増幅・推定・割引という分解は変わりません。

# %% [markdown]
# ### 元の16-point log-normal optionを復元する
#
# 上のcompactなwitnessでは各回路componentを容易にinspectできます。ここでは元の読者向け金融modelも実行します。4つのaddress qubitに16-pointへ離散化したlog-normalな満期価格分布を保持し、strikeを1.9、payoff angleを元の$c=0.25$の線形化構成にします。APIを変えずに、同じprepare、Grover amplification、maximum-likelihood、discount workflowを適用します。

# %%
FULL_ADDRESS_QUBITS = 4
FULL_TOTAL_QUBITS = 5
spot = 2.0
full_strike = 1.9
full_rate = 0.05
volatility = 0.4
full_maturity = 40.0 / 365.0
drift = (full_rate - 0.5 * volatility**2) * full_maturity
deviation = volatility * math.sqrt(full_maturity)
minimum_price = spot * math.exp(drift - 3.0 * deviation)
maximum_price = spot * math.exp(drift + 3.0 * deviation)
price_grid = np.linspace(minimum_price, maximum_price, 2**FULL_ADDRESS_QUBITS)
full_probabilities = np.exp(
    -((np.log(price_grid / spot) - drift) ** 2) / (2.0 * deviation**2)
) / (price_grid * deviation * math.sqrt(2.0 * math.pi))
full_probabilities /= full_probabilities.sum()
full_payoffs = np.maximum(price_grid - full_strike, 0.0)
full_payoff_scale = maximum_price - full_strike
scale_c = 0.25
target_amplitudes = (
    math.pi / 4.0 - scale_c + 2.0 * scale_c * full_payoffs / full_payoff_scale
)
full_exact_payoff = float(np.sum(full_probabilities * full_payoffs))
full_exact_price = math.exp(-full_rate * full_maturity) * full_exact_payoff

prepare_full_distribution = amplitude_encoding(
    tuple(math.sqrt(float(value)) for value in full_probabilities)
)
encode_full_payoff = uniformly_controlled_ry(
    tuple(2.0 * float(value) for value in target_amplitudes)
)


@qm.qkernel
def flip_target(target: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:  # type: ignore[valid-type]
    """Flip one target used by the multi-controlled zero reflection."""

    return qm.x(target)


@qm.qkernel
def full_option_state(
    state: qm.Q[qm.Bits[5]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[5]]:  # type: ignore[valid-type]
    """Prepare the original 16-point distribution and scaled payoff."""

    address, payoff = qm.split_at(state, 4)
    address = prepare_full_distribution(address)
    return encode_full_payoff(qm.concat(address, payoff))


@qm.qkernel
def full_option_samples(
    iterations: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[5]]:  # type: ignore[valid-type]
    """Sample the full option model after fixed Grover iterations."""

    state = full_option_state(qm.qubits(5))
    for _ in range(iterations):
        state = qm.z(state, target=4)
        state = qm.adjoint(full_option_state, state)
        for index in range(5):
            state = qm.x(state, target=index)
        controls, target = qm.split_at(state, 4)
        target = qm.h(target)
        controls, target = qm.control(flip_target, controls, target)
        target = qm.h(target)
        state = qm.concat(controls, target)
        for index in range(5):
            state = qm.x(state, target=index)
        state = full_option_state(state)
    return qm.measure(state)


full_shots = 32 if docs_test_mode else 256
full_observations: list[tuple[int, int, int]] = []
for power in (0, 1):
    full_program = qm.specialize(full_option_samples, iterations=power)
    full_artifact = qm.compile(full_program, to=reference)
    full_result = qm.run(
        full_artifact,
        on=reference,
        shots=full_shots,
        seed=20 + power,
    ).result
    successes = sum(
        count
        for bits, count in full_result.counts.items()
        if bits[FULL_TOTAL_QUBITS - 1] == 1
    )
    full_observations.append((power, successes, full_result.shots))


def full_negative_log_likelihood(theta: float) -> float:
    """Return the likelihood objective for the full option observations."""

    total = 0.0
    for power, successes, shots in full_observations:
        probability = math.sin((2 * power + 1) * theta) ** 2
        probability = min(max(probability, 1e-12), 1.0 - 1e-12)
        total -= successes * math.log(probability)
        total -= (shots - successes) * math.log(1.0 - probability)
    return total


full_theta_hat = min(theta_candidates, key=full_negative_log_likelihood)
full_probability_hat = math.sin(full_theta_hat) ** 2
scaled_payoff_hat = (full_probability_hat - 0.5) / scale_c
full_payoff_hat = full_payoff_scale * (scaled_payoff_hat + 1.0) / 2.0
full_price_hat = math.exp(-full_rate * full_maturity) * full_payoff_hat

print("16-point Grover data:", full_observations)
print(f"16-point estimated price: {full_price_hat:.6f}")
print(f"16-point reference price: {full_exact_price:.6f}")
assert abs(full_price_hat - full_exact_price) < (0.5 if docs_test_mode else 0.2)

# %% [markdown]
# ## まとめ
#
# ここでは、オプションプライシングアルゴリズム{cite:p}`10.22331/q-2020-07-06-291`をQamomileで実装する方法をご紹介しました。
# 以下にこのページで紹介した重要な情報をまとめます。
#
# * 対数正規分布を Möttönen の符号化手法により量子状態に符号化、一様な制御回転を用いてコールオプションのペイオフをペイオフ量子ビット振幅に符号化しました。状態の読み込みには、Qamomile の `amplitude_encoding` を用いることができます。
# * Grover 演算子 $\mathcal{Q}$ を $k$ 回適用した回路を様々な $k$ で測定し、最尤推定から振幅角を推定することで、期待ペイオフを求めました。
