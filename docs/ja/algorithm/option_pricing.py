# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# ゲート型量子コンピュータ上で実行する量子振幅推定は、様々な応用が提案されています。その中でも、金融工学のオプションプライシングは、その代表的な応用例の一つです。これまでオプションプライシングには古典モンテカルロ法が用いられてきましたが、これは目標精度 $\epsilon$ を達成するための必要なオラクルの呼び出し回数が $\mathcal{O} (\epsilon^{-2})$ でした。これに対し、量子振幅推定は $\mathcal{O} (\epsilon^{-1})$ で済み、2次の高速化をもたらすアルゴリズムであることが知られています。そこで本記事では、量子振幅推定によるオプションおよびオプションポートフォリオプライシング手法{cite:p}`10.22331/q-2020-07-06-291`の理解と実装についてまとめました。オプションプライシングの例を通して、Qamomileの使い方を学びましょう。

# %%
# Qamomile と Qiskit、SciPy、Matplotlib を PyPI からインストールします。
# #!pip install "qamomile[qiskit]" scipy matplotlib

# %%
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

import qamomile as qm
from qamomile.qiskit import Aer
from qamomile.stdlib import amplitude_encoding, uniformly_controlled_ry

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
# 金融リスクの計算は、大きな需要があります。モンテカルロ (MC) 計算はその中心的な計算手法であり、Value at Risk (VaR) の推定や店頭デリバティブの価格決定など、幅広く用いられています。しかしその誤差は $\varepsilon = \mathcal{O} (M^{-1/2})$ で減衰するという欠点があります。ここで $M$ はサンプル数です。対象となる確率分布が量子状態として準備される場合、量子振幅推定 (QAE) により、期待値などの統計量を $\mathcal{O} (M^{-1})$ の誤差で推定できることが知られています。これは先ほどの古典MCに対し、二次の高速化をもたらすことがわかります。
#
# ### 先行研究
#
# モンテカルロプライシングの量子アルゴリズムの理論的枠組みを示した先行研究として、{cite:p}`10.1103/PhysRevA.98.022321`があります。この論文は、プライシング計算において、古典のモンテカルロ法と比べて2次高速化が得られることを理論的に示しました。特にヨーロピアンオプションとアジアンオプションについて議論し、ヨーロピアンオプションに対しては数値シミュレーションも行い、その有効性を示しました。しかし、その成果は理論的な枠組みの提示と数値シミュレーションに留まり、実機での実行は示されていません。
#
# もう一つの重要な先行研究として、{cite:p}`10.1038/s41534-019-0130-6`があります。これは VaR や Conditional Value at Risk (CVaR) に QAE を適用した研究です。モンテカルロプライシングのペイオフ計算を改良し、必要量子ビット数・ゲート数の大幅な削減に成功しました{cite:p}`10.1103/PhysRevA.98.022321`。このペイオフ計算手法をオプションプライシングに拡張した実装も示されています{cite:p}`10.22331/q-2020-07-06-291`。
#
# ## アルゴリズム: QAEによるオプションプライシング計算
#
# ### ペイオフの計算
#
# QAE ではグローバー演算子 $\mathcal{Q}$ を $k$ 回繰り返すことで、測定確率が $\sin^2((2k+1)\theta_a)$ という既知の関数形で変化します。これにより、少ない測定回数でも $\theta_a$ を高精度に推定することが可能になります。今、アセットの価格を $S_i$とし、それが起こる確率を $p_i$ としましょう。演算子 $\mathcal{A}$ が
#
# $$
# \mathcal{A} \vert 0 \rangle_{n+1}
# = \sum_{i=0}^{2^n - 1} \sqrt{1-f(S_i)} \sqrt{p_i} \vert S_i \rangle \vert 0 \rangle + \sum_{i=0}^{2^n - 1} \sqrt{f(S_i)} \sqrt{p_i} \vert S_i \rangle \vert 1 \rangle \tag{1}
# $$
#
# のような状態を作れるとすると、ペイオフ量子ビットが $\vert 1 \rangle$ の状態で観測される確率は $a = \mathbb{E}[f(S)]$ となり、QAE によりペイオフの期待値を計算できることがわかります。補助レジスタにペイオフをバイナリ表現で格納する手法では、量子ビット数と回路の深さが増大します{cite:p}`10.1103/PhysRevA.98.022321`。軽量な$R_y$手法では、このオーバーヘッドを避けられます{cite:p}`10.22331/q-2020-07-06-291,10.1038/s41534-019-0130-6`。
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
# と定義します。スケーリングパラメータ $c$ は $0 < c \leq 1$ を満たします。このときペイオフ量子ビットが $\vert 1 \rangle$ となる確率は
#
# $$
# P_1 \approx \frac{1}{2} + c\,\mathbb{E}[\tilde{f}] + \mathcal{O}(c^3) \tag{3}
# $$
#
# となり、$P_1$ を推定すれば既知の $c, f_\mathrm{min}, f_\mathrm{max}$ から $\mathbb{E}[f(S)]$ を算出できます。 $P_1$ の推定誤差を $\delta P_1$ とすると、$\mathbb{E}[\tilde{f}] \approx (P_1 - 1/2)/c$ より推定誤差は $\delta P_1/c$ に増幅されます。 $c$ を小さくするとこの誤差増幅が大きくなり、逆に大きくすると近似バイアス $\mathcal{O}(c^2)$ が増大します。このトレードオフから収束率は $\mathcal{O}(M^{-2/3})$ となります。完全な二次高速化には至りませんが、依然として古典 MC の $\mathcal{O}(M^{-1/2})$ を凌駕します。以下の実装では、位相推定を用いない軽量な最尤振幅推定(MLAE)を採用しています。
#
# ### コールオプションへの応用
#
# コールオプションのペイオフ $f_C(S_T) = \max(0, S_T - K)$ は $K$ を境に折れ曲がる区分線形関数です。本実装では均一制御回転(UCR-Y)を用い、各格子点 $S_i$ に対して
#
# $$
# \alpha(S_i) = \frac{\pi}{4} - c + \frac{2c\,\max(S_i - K,\,0)}{S_\mathrm{max} - K} \tag{4}
# $$
#
# という目標振幅角を直接適用します。 $S < K$ では定数角 $\pi/4 - c$、$S \geq K$ では線形増加という区分線形構造を、正確に符号化することができます。

# %% [markdown]
# ## Qamomileによる実装
#
# ### 分布の準備
#
# それでは、オプションプライシングの実装{cite:p}`10.22331/q-2020-07-06-291`を見ていきましょう。ここではまず、幾何ブラウン運動下にある満期 $S_T$ の対数正規分布を離散化したものを準備し、これを株価のデータとします。

# %%
n_qubits = 4
S0, K, r, sigma, T = 2.0, 1.9, 0.05, 0.4, 40 / 365
if not (sigma > 0 and T > 0):
    raise ValueError("この例では sigma と T に正の値を指定してください。")

mu  = (r - 0.5 * sigma**2) * T
std = sigma * np.sqrt(T)
S_min = float(S0 * np.exp(mu - 3 * std))
S_max = float(S0 * np.exp(mu + 3 * std))
S_grid = np.linspace(S_min, S_max, 2**n_qubits)

def lognpdf(S):
    return (np.exp(-(np.log(S / S0) - mu)**2 / (2 * std**2))
            / (S * std * np.sqrt(2 * np.pi)))

probs = lognpdf(S_grid)
probs /= probs.sum()
amps  = np.sqrt(probs)   # Möttönen エンコード用の振幅 √p_i

c     = 0.25             # スケーリングパラメータ
if not 0 < c <= 1:
    raise ValueError("c は 0 < c <= 1 の範囲で指定してください。")

F_max = float(S_max - K)
if F_max <= 0:
    raise ValueError("格子上の受取額はすべて0です。この例では K < S_max を指定してください。")

# ── 目標振幅角 ──
payoff       = np.maximum(S_grid - K, 0.0)
alpha_target = np.pi / 4 - c + 2 * c * payoff / F_max   # shape: (2^n,)

# ── 参照値 ──
# exact_E_f: 数値積分による厳密な期待ペイオフ(離散格子上の積分値)
exact_E_f  = float(np.sum(probs * payoff))
exact_fair = np.exp(-r * T) * exact_E_f

# P1 の厳密値: Σ_i p_i sin²(α_i)  ← 線形近似を使わない
P1_exact  = float(np.sum(probs * np.sin(alpha_target)**2))

# P1 の近似値
E_ftilde  = 2 * exact_E_f / F_max - 1
P1_approx = c * E_ftilde + 0.5

print(f"期待ペイオフ(数値積分):     {exact_E_f:.6f}")
print(f"フェアバリュー(参照値):     {exact_fair:.6f}")
print(f"P1 厳密値:                 {P1_exact:.6f}")
print(f"P1 近似値(線形近似):        {P1_approx:.6f}")
print(f"P1 近似誤差:               {abs(P1_exact - P1_approx):.6f}")


# %% [markdown]
# ### ペイオフ演算子の実装
#
# アルゴリズムの部分で示したように、コールオプションのペイオフ $f(S) = \max(S-K, 0)$ は $K$ で折れ曲がる区分線形関数です。そのため、格子点ごとに異なる目標振幅角
#
# $$
# \alpha(S_i) = \frac{\pi}{4} - c + \frac{2c\,\max(S_i - K,\, 0)}{S_\mathrm{max} - K}
# $$
#
# をペイオフ量子ビットに適用する必要があります。UCR-Y は各基底状態 $\vert i \rangle$ に対して独立な振幅角 $\alpha(S_i)$ を適用できるため、$K$ を境とした区分線形構造を正確に符号化することができます。以下では、価格点が2つの場合を例に、UCR-Y を $R_y$ と CNOT に分解します。格子全体には Qamomile の `uniformly_controlled_ry` を使います。

# %%
# 各価格点の Ry の回転角は、目標振幅角の2倍。
price_rotation_angles = (2 * alpha_target).tolist()

# 最低価格と最高価格の2点だけなら、1個の価格ビットが両方の CNOT を制御する。
# 4個のゲートの標的はすべてペイオフ量子ビット。
low_price_angle, high_price_angle = price_rotation_angles[0], price_rotation_angles[-1]
first_ry = (low_price_angle + high_price_angle) / 2
second_ry = (low_price_angle - high_price_angle) / 2
print(f"Ry({first_ry:.4f}) → CNOT → Ry({second_ry:.4f}) → CNOT")

# 格子全体では、価格ビットが制御ビット、最後のビットが標的のペイオフ量子ビット。
apply_payoff = uniformly_controlled_ry(price_rotation_angles)
print(f"UCR-Y ゲート数: {2 * len(price_rotation_angles)} (Ry + CNOT)")


# %% [markdown]
# ### 分布の読み込みの実装
#
# そして先ほど準備した株価データを読み込む演算と、そのエルミート演算を実装しましょう。Qamomileには、振幅符号化の関数`amplitude_encoding`が標準で備わっています。

# %%
# P_X で分布を読み込み、Ak の中で qm.adjoint を使ってその随伴を適用する。
load_distribution = amplitude_encoding(tuple(amps))


# %% [markdown]
# この分布読み込みと、UCR-Y を施すことで、コールオプションのペイオフをペイオフ量子ビット振幅に変換する回路を構築します。`num_shots` の回数だけ測定を行い、ペイオフ量子ビットが $\vert 1 \rangle$ になる確率 $P_1$ を推定します。そして(3)の近似式 $P_1 \approx \frac{1}{2} + c\mathbb{E}[\tilde{f}]$ から、ペイオフの期待値を算出します。ここで算出された $P_1$ の値が理論値と近い値となることを示すことで、ペイオフ回路が正しく機能していることを確認しています。

# %%
@qm.qkernel
def payoff_call() -> qm.Shot[qm.Bits[1]]:
    q_S = qm.qubits(n_qubits)
    q_p = qm.qubits(1)

    # (1) 分布ロード: P_X
    q_S = load_distribution(q_S)

    # (2) UCR-Y ペイオフ回路: F
    #     |i⟩|0⟩ → |i⟩(cos α(i)|0⟩ + sin α(i)|1⟩)
    #     α(i) = π/4 - c + 2c*max(S_i-K, 0) / (S_max-K)
    #     S < K: 定数角 π/4 - c  /  S >= K: 線形増加
    state = apply_payoff(qm.concat(q_S, q_p))
    q_S, q_p = qm.split_at(state, n_qubits)
    _ = qm.measure(q_S)
    return qm.measure(q_p)


num_shots = 1 if docs_test_mode else 4096
simulator = Aer()
exe_k0 = qm.compile(payoff_call, to=simulator.target)
res_k0 = qm.run(exe_k0, on=simulator, shots=num_shots).result
assert sum(res_k0.counts.values()) == num_shots
ones_k0 = res_k0.counts.get((1,), 0)
P1_k0 = ones_k0 / num_shots

print(f"\n[Qamomile k=0]")
print(f"P1 測定値:              {P1_k0:.4f}")
print(f"P1 厳密値(参照):     {P1_exact:.4f}")
print(f"P1 近似値:          {P1_approx:.4f}")
print("→ 測定値が厳密値に近ければペイオフ回路は正しく動いています")


# %% [markdown]
# ### 量子振幅推定の実装
#
# 振幅推定のためのGrover演算子 $\mathcal{Q} = \mathcal{A} \mathcal{S}_0 \mathcal{A}^\dagger \mathcal{S}_\chi$ を $k$ 回適用する量子回路を構築しましょう。 $k$ 回適用した量子状態において、ペイオフ量子ビットが $\vert 1 \rangle$ になる確率 $P_1^{(k)} = \sin^2 ((2k+1) \theta_\alpha)$ を求めます。

# %%
@qm.qkernel
def flip_target(q_p: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(q_p)


@qm.qkernel
def Ak(k: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    q_S = qm.qubits(n_qubits)
    q_p = qm.qubits(1)

    # ── A = F · P_X ──
    q_S = load_distribution(q_S)
    state = apply_payoff(qm.concat(q_S, q_p))

    # ── Q を k 回適用 ──
    for _ in range(k):

        # S_χ: payoff qubit が |1⟩ のとき位相 -1
        state = qm.z(state, target=n_qubits)

        # A† = P_X† · F†
        state = qm.adjoint(apply_payoff, state)
        q_S, q_p = qm.split_at(state, n_qubits)
        q_S = qm.adjoint(load_distribution, q_S)

        # S₀: |0…0⟩ のとき位相 -1
        # X で全反転 → MCZ(= H · MCX · H)→ X で戻す
        # コントロール = q_S 全体、ターゲット = q_p。アンシラ不要。
        for i in range(n_qubits):
            q_S = qm.x(q_S, target=i)
        q_p = qm.x(q_p)
        q_p = qm.h(q_p)
        q_S, q_p = qm.control(flip_target, q_S, q_p)
        q_p = qm.h(q_p)
        for i in range(n_qubits):
            q_S = qm.x(q_S, target=i)
        q_p = qm.x(q_p)

        # A
        q_S = load_distribution(q_S)
        state = apply_payoff(qm.concat(q_S, q_p))

    q_S, q_p = qm.split_at(state, n_qubits)
    _ = qm.measure(q_S)
    return qm.measure(q_p)

k_list = [1] if docs_test_mode else [0, 1, 2, 4, 8, 16]
shots_per_k = 1 if docs_test_mode else 2048

h_list, N_list = [], []

# P1 の理論値には線形近似値ではなく厳密値を使用
theta_a_theory = np.arcsin(np.sqrt(max(0.0, min(1.0, P1_exact))))
print()
for k in k_list:
    program = qm.specialize(Ak, k=k)
    exe = qm.compile(program, to=simulator.target)
    result = qm.run(exe, on=simulator, shots=shots_per_k).result
    assert sum(result.counts.values()) == shots_per_k
    ones = result.counts.get((1,), 0)
    h_list.append(ones)
    N_list.append(shots_per_k)
    P1_k  = ones / shots_per_k
    P1_th = np.sin((2*k + 1) * theta_a_theory)**2
    print(f"k={k:2d}: P1 測定={P1_k:.4f}  理論値(厳密)={P1_th:.4f}")


# %% [markdown]
# ## 結果
#
# 複数の $k$ での測定データ (二項分布) をベースに、$P_1^{(k)} = \sin^2 ((2k+1) \theta_\alpha)$ というモデルに対する最尤推定を実施しましょう。これにより振幅角 $\theta_\alpha$ を推定し、さらに $P_1 = \sin^2 \theta_\alpha$ を計算します。そしてこの $P_1$ からペイオフの期待値 $\mathbb{E}[f(S)]$ を計算し、無リスク金利で割り引くことで、オプションのフェアバリューを算出します。最後に、厳密計算から求めた参照値との比較を行っています。

# %%
# ============================================================
# 4. 最尤推定(MLE)
# ============================================================
def neg_log_lik(theta: float) -> float:
    """
    負の対数尤度:
      -Σ_k [h_k log sin²((2k+1)θ) + (N_k - h_k) log cos²((2k+1)θ)]
    """
    ll = 0.0
    for k, h, N in zip(k_list, h_list, N_list):
        s2 = float(np.clip(np.sin((2*k + 1) * theta)**2, 1e-12, 1 - 1e-12))
        ll += h * np.log(s2) + (N - h) * np.log(1 - s2)
    return -ll

opt       = minimize_scalar(neg_log_lik, bounds=(0, np.pi / 2), method='bounded')
theta_hat = opt.x
a_hat     = np.sin(theta_hat)**2

# P1 → E[f(S)] の逆算(式(3) の線形近似を使用、近似バイアスは O(c²))
# E[f̃] ≈ (P1 - 1/2) / c, E[f] = F_max/2 * (E[f̃] + 1)
E_ftilde_hat = (a_hat - 0.5) / c
E_f_hat      = (E_ftilde_hat + 1) * F_max / 2
fair_hat     = np.exp(-r * T) * E_f_hat

print(f"\n===== 推定結果 =====")
print(f"推定 P1:          {a_hat:.6f}  "
      f"(厳密値 {P1_exact:.6f}, 近似値 {P1_approx:.6f})")
print(f"推定 E[f(S)]:     {E_f_hat:.6f}  (参照値 {exact_E_f:.6f})")
print(f"推定フェアバリュー: {fair_hat:.6f}  (参照値 {exact_fair:.6f})")
print(f"絶対誤差:          {abs(fair_hat - exact_fair):.6f}")

# %% [markdown]
# 量子回路により推定して得た値 `E[f(S)]` が参照値と異なっているのは
#
# $$
# \mathbb{E}[f] = \frac{F_\mathrm{max}}{2} \left( \frac{P_1 - \frac{1}{2}}{c} + 1 \right)
# $$
#
# のように近似したことによるものです。ここで $F_\mathrm{max} = S_\mathrm{max} - K$ です。実際に、推定された $\mathbb{E}[\hat{f}]$ と参照値との差において、近似誤差がどのくらい寄与しているかをみてみましょう。
# %%
total_error = E_f_hat - exact_E_f
approximation_error = F_max / 2 * (P1_exact - P1_approx) / c
statistical_error = total_error - approximation_error
error_magnitude_sum = abs(approximation_error) + abs(statistical_error)
approximation_share = abs(approximation_error) / error_magnitude_sum

print(f"近似誤差: {approximation_error:.6f}")
print(f"統計誤差: {statistical_error:.6f}")
print(f"誤差の絶対値に占める近似誤差の割合: {approximation_share:.1%}")

# %% [markdown]
# この例では、誤差の大半が近似によるものであることがわかります。残りの部分は統計誤差です。これは量子振幅推定部分の測定回数を増やすことで、減少させることができます。しかし測定回数を増やすと計算実行時間が増大するため、注意が必要です。
#
# 最後に、対数尤度関数 $\log \mathcal{L} (\sin^2 \theta)$と、$P_1^{(k)}$ をプロットしてみましょう。左図は対数尤度関数と、最尤推定値および厳密計算による理論値も示しています。右図は、推定した$\hat{\theta}$を代入した正弦二乗曲線 $P_1^{(k)} = \sin^2 ((2k+1) \hat{\theta})$ と測定値を重ねて描画しています。

# %%
# ============================================================
# 5. 可視化
# ============================================================
theta_grid = np.linspace(1e-4, np.pi / 2 - 1e-4, 400)
ll_grid    = np.array([-neg_log_lik(t) for t in theta_grid])
a_grid     = np.sin(theta_grid)**2

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(a_grid, ll_grid, color='steelblue')
axes[0].axvline(a_hat,     color='red',   ls='--',
                label=f'estimate  {a_hat:.4f}')
axes[0].axvline(P1_exact,  color='navy',  ls=':',
                label=f'exact P1  {P1_exact:.4f}')
axes[0].axvline(P1_approx, color='green', ls='-.',
                label=f'approx P1 {P1_approx:.4f}')
axes[0].set_xlabel('a = sin²(θ)')
axes[0].set_ylabel('log-likelihood')
axes[0].set_title('MLAE likelihood')
axes[0].legend()

k_smooth = np.linspace(0, max(k_list), 500)
P1_sine_curve = np.sin((2 * k_smooth + 1) * theta_hat)**2
axes[1].plot(k_smooth, P1_sine_curve, '-', color='steelblue',
             label=rf'$\sin^2((2k+1)\hat{{\theta}})$ ($\hat{{\theta}}={theta_hat:.3f}$)')
axes[1].plot(k_list, [h / N for h, N in zip(h_list, N_list)],
             'o', color='red', zorder=5, label='measured value')
axes[1].set_xlabel('k  (Grover iterations)')
axes[1].set_ylabel('P₁ estimation')
axes[1].set_title('P₁^(k) = sin²((2k+1)θ̂)')
axes[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# MLAE により、対数尤度が最大となる付近を推定できていることがわかります。

# %% [markdown]
# ## まとめ
#
# ここでは、オプションプライシングアルゴリズム{cite:p}`10.22331/q-2020-07-06-291`をQamomileで実装する方法をご紹介しました。以下にこのページで紹介した重要な情報をまとめます。
#
# * 対数正規分布を Möttönen の符号化手法により量子状態に符号化、一様な制御回転を用いてコールオプションのペイオフをペイオフ量子ビット振幅に符号化しました。状態の読み込みには、Qamomile の `amplitude_encoding` を用いることができます。
# * Grover 演算子 $\mathcal{Q}$ を $k$ 回適用した回路を様々な $k$ で測定し、最尤推定から振幅角を推定することで、期待ペイオフを求めました。
