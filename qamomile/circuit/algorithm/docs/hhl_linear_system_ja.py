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
# # HHL アルゴリズム: 量子コンピュータで線形方程式系を解く
#
# このチュートリアルでは、Qamomile の低レベル回路プリミティブを用いて
# Harrow-Hassidim-Lloyd (HHL) アルゴリズムをステップごとに構築します。
# 高レベルの `hhl()` 関数をそのまま使うのではなく、以下の流れで進めます:
#
# 1. HHL アルゴリズムの理論を確認する。
# 2. 具体的な対角線形方程式系を設定する。
# 3. HHL 回路の各コンポーネント（順方向 QPE、逆回転、逆 QPE）を
#    手動で構築する。
# 4. 状態ベクトルシミュレーションで実行し、古典解と比較して検証する。
# 5. `qamomile.circuit.algorithm.hhl` が同じ回路を単一関数で
#    提供することを確認する。

# %%
# pip から最新の Qamomile をインストールします
# # !pip install qamomile

# %% [markdown]
# ## HHL アルゴリズム
#
# エルミート行列 $A$ とベクトル $|b\rangle$ が与えられたとき、HHL アルゴリズムは
# $A^{-1}|b\rangle$ に比例する量子状態を準備します。
# アルゴリズムは 4 つのステージで構成されます。
#
# ### 1. 量子位相推定 (QPE)
#
# $|b\rangle$ を $A$ の固有基底で展開します:
# $|b\rangle = \sum_j \beta_j |u_j\rangle$、ここで $A|u_j\rangle = \lambda_j |u_j\rangle$。
#
# QPE は固有値をクロックレジスタにエンコードします:
#
# $$
# |0\rangle_c |b\rangle_s |0\rangle_a
# \;\xrightarrow{\text{QPE}}\;
# \sum_j \beta_j |\tilde\lambda_j\rangle_c |u_j\rangle_s |0\rangle_a
# $$
#
# ### 2. 逆回転 (Reciprocal Rotation)
#
# 制御回転により、$C / \tilde\lambda_j$ を補助ビットの振幅に埋め込みます:
#
# $$
# \xrightarrow{\text{Reciprocal}}\;
# \sum_j \beta_j |\tilde\lambda_j\rangle_c |u_j\rangle_s
# \left(
#   \sqrt{1 - \frac{C^2}{\tilde\lambda_j^2}}\,|0\rangle
#   + \frac{C}{\tilde\lambda_j}\,|1\rangle
# \right)_a
# $$
#
# ### 3. 逆 QPE
#
# クロックレジスタを $|0\rangle_c$ に戻します。
#
# ### 4. ポストセレクション
#
# 補助ビットを $|1\rangle$ で測定すると、システムレジスタは以下に射影されます:
#
# $$
# C \sum_j \frac{\beta_j}{\tilde\lambda_j} |u_j\rangle_s
# \;\propto\; A^{-1}|b\rangle
# $$

# %% [markdown]
# ## 問題設定: 対角線形方程式系
#
# 1 量子ビットゲート $U = R_z(\pi)$ をハミルトニアンシミュレーションのユニタリとして
# 使用します。$R_z(\alpha)$ は固有状態 $|0\rangle$ と $|1\rangle$ を持ち、固有値は
# それぞれ $e^{-i\alpha/2}$ と $e^{+i\alpha/2}$ です:
#
# $$
# R_z(\pi) = e^{-i\frac{\pi}{2}Z}
# = \begin{pmatrix} e^{-i\pi/2} & 0 \\ 0 & e^{i\pi/2} \end{pmatrix}
# = \begin{pmatrix} -i & 0 \\ 0 & i \end{pmatrix}
# $$
#
# 2 クロック量子ビット、`phase_scale` $= 2\pi$（符号なしモード）の設定で、QPE は
# これらの固有値を以下のようにクロックレジスタのbinにマッピングします:
#
# | 固有状態 | 固有値 | 固有位相 $\varphi$ | Raw bin | $\hat\lambda = 2\pi \cdot \text{raw}/4$ |
# |:---:|:---:|:---:|:---:|:---:|
# | $\|0\rangle$ | $e^{-i\pi/2}$ | $-\tfrac{1}{4} \bmod 1 = \tfrac{3}{4}$ | 3 | $\tfrac{3\pi}{2}$ |
# | $\|1\rangle$ | $e^{+i\pi/2}$ | $\tfrac{1}{4}$ | 1 | $\tfrac{\pi}{2}$ |
#
# ここで用いている unsigned 位相復号の規約
# (`phase_scale` $= 2\pi$、raw bin を $[0, 2\pi)$ にマッピング）では、
# HHL が反転する実効的な対角行列は
# $A = \mathrm{diag}(3\pi/2,\; \pi/2)$ となります。
# $3\pi/2$ は、負の固有位相 $-1/4$ が modulo 1 で $3/4$ に
# 巻き戻された結果です。
#
# 入力ベクトルとして $|b\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$ を選ぶと、
# 厳密解は:
#
# $$
# A^{-1}|b\rangle \;\propto\;
# \frac{1}{3\pi/2}|0\rangle + \frac{1}{\pi/2}|1\rangle
# \;\propto\; \tfrac{1}{3}|0\rangle + |1\rangle
# $$

# %%
import math

import numpy as np

# Problem parameters
alpha_val = math.pi
C = 0.4
phase_scale = 2.0 * math.pi

# Input vector |b> (will be normalized by amplitude_encoding)
b_amplitudes = [1.0, 1.0]

# Eigenvalues decoded by QPE
lambda_0 = 3 * math.pi / 2  # |0> : raw bin 3
lambda_1 = math.pi / 2  # |1> : raw bin 1

# Classical exact solution
b_norm = np.array(b_amplitudes) / np.linalg.norm(b_amplitudes)
x_exact = np.array([b_norm[0] / lambda_0, b_norm[1] / lambda_1])
x_exact_normalized = x_exact / np.linalg.norm(x_exact)

print(f"Eigenvalues: lambda_0 = {lambda_0:.4f}, lambda_1 = {lambda_1:.4f}")
print(f"|b> (normalized): {b_norm}")
print(f"A^{{-1}}|b> (unnormalized): [{b_norm[0]/lambda_0:.4f}, {b_norm[1]/lambda_1:.4f}]")
print(f"A^{{-1}}|b> (normalized):   {x_exact_normalized}")

# %% [markdown]
# ## HHL をスクラッチで構築する
#
# 組み込みの `hhl()` 関数を使う前に、アルゴリズムの構造を理解するために
# 各コンポーネントを手動で構築しましょう。

# %% [markdown]
# ### ステップ 1: ユニタリカーネルの定義
#
# HHL にはユニタリ $U$ とその随伴 $U^\dagger$ を `@qkernel` 関数として渡す必要が
# あります。`qmc.controlled()` ラッパーに `power`$=2^k$ を指定することで、
# QPE に必要な制御-$U^{2^k}$ 演算を生成します。

# %%
import qamomile.circuit as qmc
from qamomile.circuit.algorithm.mottonen_amplitude_encoding import amplitude_encoding
from qamomile.circuit.stdlib.qft import iqft, qft


# fmt: off
@qmc.qkernel
def rz_unitary(q: qmc.Qubit, alpha: qmc.Float) -> qmc.Qubit:
    """U = Rz(alpha)."""
    return qmc.rz(q, alpha)


@qmc.qkernel
def rz_unitary_inv(q: qmc.Qubit, alpha: qmc.Float) -> qmc.Qubit:
    """U-dagger = Rz(-alpha)."""
    return qmc.rz(q, -1.0 * alpha)


@qmc.qkernel
def ry_gate(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
    """Single RY gate (wrapper for creating multi-controlled version)."""
    return qmc.ry(q, angle)
# fmt: on

# %% [markdown]
# ### ステップ 2: 順方向 QPE
#
# 順方向 QPE は固有値をクロックレジスタにエンコードします:
#
# 1. 全クロック量子ビットにアダマールゲートを適用（等重ね合わせを生成）。
# 2. 制御-$U^{2^k}$ を適用: クロック量子ビット $k$ が $U^{2^k}$ を制御。
# 3. 逆量子フーリエ変換 (IQFT) を適用し、位相情報を計算基底状態に変換。
#
# 2 クロック量子ビットのリトルエンディアン規約（`clock[0]` = LSB）では:
# - `clock[0]` が $U^1$ を制御
# - `clock[1]` が $U^2$ を制御

# %% [markdown]
# ### ステップ 3: 逆回転 (Reciprocal Rotation)
#
# 各固有値binに対して回転角 $\theta = 2 \arcsin(C / \hat\lambda)$ を計算し、
# クロックレジスタが対応する基底状態にあることを条件として、
# 補助ビットにマルチ制御 $R_Y(\theta)$ を適用します。
#
# $R_z(\pi)$、2 クロック量子ビットの場合、bin 1 と 3 のみが占有されます:
#
# - **bin 1** (clock = $|01\rangle$): $\hat\lambda_1 = \pi/2$,
#   $\theta_1 = 2\arcsin(0.4 / (\pi/2)) \approx 0.519$
# - **bin 3** (clock = $|11\rangle$): $\hat\lambda_3 = 3\pi/2$,
#   $\theta_3 = 2\arcsin(0.4 / (3\pi/2)) \approx 0.170$
#
# 特定の基底状態を選択するには、$|0\rangle$ であるべき量子ビットを X ゲートで
# フリップし、全制御ビットが $|1\rangle$ を見るようにします。

# %% [markdown]
# ### ステップ 4: 逆 QPE
#
# 順方向 QPE を逆順にミラーします:
# QFT → 制御-$U^{\dagger 2^k}$（逆順）→ アダマール。

# %% [markdown]
# ### 完全なナイーブ HHL 回路
#
# 3 つのステップを 1 つの `@qkernel` にまとめます:

# %%
# Precompute rotation angles for reciprocal rotation
theta_bin1 = 2.0 * math.asin(C / lambda_1)  # bin 1: lambda = pi/2
theta_bin3 = 2.0 * math.asin(C / lambda_0)  # bin 3: lambda = 3*pi/2

print(f"Reciprocal rotation angles:")
print(f"  Bin 1 (lambda={lambda_1:.4f}): theta = {theta_bin1:.6f}")
print(f"  Bin 3 (lambda={lambda_0:.4f}): theta = {theta_bin3:.6f}")


# %%
@qmc.qkernel
def hhl_naive(alpha: qmc.Float) -> qmc.Bit:
    # --- Allocate registers ---
    sys = qmc.qubit_array(1, name="sys")
    sys = amplitude_encoding(sys, b_amplitudes)
    clock = qmc.qubit_array(2, name="clock")
    anc = qmc.qubit("anc")

    # === Step 1: Forward QPE ===
    # 1a. Hadamard on all clock qubits
    clock[0] = qmc.h(clock[0])
    clock[1] = qmc.h(clock[1])

    # 1b. Controlled-U^(2^k) operations
    # clock[0] (LSB) controls U^1, clock[1] controls U^2
    controlled_u = qmc.controlled(rz_unitary)
    clock[0], sys[0] = controlled_u(clock[0], sys[0], power=1, alpha=alpha)
    clock[1], sys[0] = controlled_u(clock[1], sys[0], power=2, alpha=alpha)

    # 1c. Inverse QFT on clock register
    clock = iqft(clock)

    # === Step 2: Reciprocal Rotation ===
    mc_ry = qmc.controlled(ry_gate, num_controls=2)

    # Bin 1 (clock = |01>): clock[0]=1, clock[1]=0
    # Flip clock[1] so both controls see |1>
    clock[1] = qmc.x(clock[1])
    clock[0], clock[1], anc = mc_ry(
        clock[0], clock[1], anc, angle=theta_bin1
    )
    clock[1] = qmc.x(clock[1])  # undo flip

    # Bin 3 (clock = |11>): clock[0]=1, clock[1]=1
    # No flips needed — both controls are already |1>
    clock[0], clock[1], anc = mc_ry(
        clock[0], clock[1], anc, angle=theta_bin3
    )

    # === Step 3: Inverse QPE (uncompute) ===
    # 3a. QFT on clock (inverse of IQFT)
    clock = qft(clock)

    # 3b. Controlled-U-dagger in reverse qubit order
    controlled_u_inv = qmc.controlled(rz_unitary_inv)
    clock[1], sys[0] = controlled_u_inv(
        clock[1], sys[0], power=2, alpha=alpha
    )
    clock[0], sys[0] = controlled_u_inv(
        clock[0], sys[0], power=1, alpha=alpha
    )

    # 3c. Hadamard on all clock qubits
    clock[0] = qmc.h(clock[0])
    clock[1] = qmc.h(clock[1])

    return qmc.measure(anc)


# %% [markdown]
# ## ナイーブ HHL 回路のシミュレーション

# %%
import matplotlib.pyplot as plt
from qiskit import transpile as qk_transpile
from qiskit_aer import AerSimulator

from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()
simulator = AerSimulator(method="statevector")


def simulate_hhl(kernel, alpha):
    """Transpile, simulate, and extract post-selected system state."""
    exe = transpiler.transpile(kernel, bindings={"alpha": alpha})
    qc = exe.compiled_quantum[0].circuit.copy()
    qc.remove_final_measurements()
    qc_compiled = qk_transpile(qc, simulator)
    qc_compiled.save_statevector()
    result = simulator.run(qc_compiled).result()
    sv = np.array(result.get_statevector())
    return sv


def extract_postselected_system(sv, n_system, n_clock):
    """Extract system amplitudes conditioned on ancilla=|1> and clock=|00...0>.

    Qubit allocation order: sys[0..n_sys-1], clock[0..n_clk-1], anc.
    Qiskit little-endian: statevector index = q0 + q1*2 + q2*4 + ...
    """
    n_total = n_system + n_clock + 1
    anc_pos = n_system + n_clock
    system_amps = np.zeros(2**n_system, dtype=complex)

    for idx in range(len(sv)):
        if not ((idx >> anc_pos) & 1):
            continue
        clock_zero = all(
            not ((idx >> (n_system + c)) & 1) for c in range(n_clock)
        )
        if not clock_zero:
            continue
        sys_val = idx & ((1 << n_system) - 1)
        system_amps[sys_val] = sv[idx]

    return system_amps


# %%
sv_naive = simulate_hhl(hhl_naive, alpha_val)
sys_naive = extract_postselected_system(sv_naive, n_system=1, n_clock=2)
norm_naive = np.linalg.norm(sys_naive)

print("=== Naive HHL Results ===")
print(f"Post-selection probability: {norm_naive**2:.6f}")
print(f"Post-selected system state: {sys_naive / norm_naive}")

fid_naive = float(
    np.abs(np.vdot(x_exact_normalized, sys_naive / norm_naive)) ** 2
)
print(f"Fidelity with exact A^{{-1}}|b>: {fid_naive:.6f}")

# %% [markdown]
# ## 組み込みの `hhl()` を使う
#
# 上記で手動実装した全て（順方向 QPE、逆回転、逆 QPE）は
# `qamomile.circuit.algorithm.hhl` によって提供されています。
# 同じユニタリカーネルを受け取り、固有値のデコード、bin選択、
# マルチ制御回転を自動的に処理します。
#
# 組み込み関数を使って同じ回路を構築し、同一の結果が得られることを
# 確認しましょう。

# %%
from qamomile.circuit.algorithm.hhl import hhl


@qmc.qkernel
def hhl_builtin(alpha: qmc.Float) -> qmc.Bit:
    sys = qmc.qubit_array(1, name="sys")
    sys = amplitude_encoding(sys, b_amplitudes)
    clock = qmc.qubit_array(2, name="clock")
    anc = qmc.qubit("anc")

    sys, clock, anc = hhl(
        sys,
        clock,
        anc,
        unitary=rz_unitary,
        inv_unitary=rz_unitary_inv,
        scaling=C,
        phase_scale=phase_scale,
        supported_raw_bins=(1, 3),
        strict=True,
        alpha=alpha,
    )

    return qmc.measure(anc)


# %%
sv_builtin = simulate_hhl(hhl_builtin, alpha_val)
sys_builtin = extract_postselected_system(sv_builtin, n_system=1, n_clock=2)
norm_builtin = np.linalg.norm(sys_builtin)

print("=== Built-in hhl() Results ===")
print(f"Post-selection probability: {norm_builtin**2:.6f}")
print(f"Post-selected system state: {sys_builtin / norm_builtin}")

fid_builtin = float(
    np.abs(np.vdot(x_exact_normalized, sys_builtin / norm_builtin)) ** 2
)
print(f"Fidelity with exact A^{{-1}}|b>: {fid_builtin:.6f}")

# %% [markdown]
# 両方の回路は同一のフィデリティを示すはずです。組み込みの `hhl()` は
# 手動で構築したものと同じアルゴリズムを実装しているためです。

# %%
# Verify post-selection probability against theory
p_expected = (
    abs(b_norm[0]) ** 2 * (C / lambda_0) ** 2
    + abs(b_norm[1]) ** 2 * (C / lambda_1) ** 2
)
print(f"\nTheoretical post-selection probability: {p_expected:.6f}")
print(f"Naive   post-selection probability:     {norm_naive**2:.6f}")
print(f"Built-in post-selection probability:    {norm_builtin**2:.6f}")

# %% [markdown]
# ## 結果の可視化
#
# 以下の棒グラフは、HHL の出力状態の確率分布と厳密な古典解を比較したものです。

# %%
quantum_probs = np.abs(sys_builtin / norm_builtin) ** 2
classical_probs = np.abs(x_exact_normalized) ** 2

labels = ["|0>", "|1>"]
x_pos = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 4.5))
bars1 = ax.bar(
    x_pos - width / 2,
    quantum_probs,
    width,
    label="HHL (quantum)",
    color="#2696EB",
)
bars2 = ax.bar(
    x_pos + width / 2,
    classical_probs,
    width,
    label=r"Exact $A^{-1}|b\rangle$",
    color="#FF6B6B",
)

ax.set_xlabel("Basis state")
ax.set_ylabel("Probability")
ax.set_title("HHL Result vs Exact Solution")
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylim(0, 1.0)

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 異なる入力ベクトルの試行
#
# HHL 回路は任意の入力 $|b\rangle$ に対して動作します。
# 以下では、組み込みの `hhl()` を使っていくつかの入力に対するフィデリティを
# 確認します。

# %%
test_cases = [
    ([1.0, 1.0], "uniform"),
    ([1.0, 2.0], "asymmetric"),
    ([1.0, 0.0], "basis |0>"),
    ([0.0, 1.0], "basis |1>"),
]

for amps, label in test_cases:

    @qmc.qkernel
    def _circuit(alpha: qmc.Float) -> qmc.Bit:
        sys = qmc.qubit_array(1, name="sys")
        sys = amplitude_encoding(sys, amps)
        clock = qmc.qubit_array(2, name="clock")
        anc = qmc.qubit("anc")
        sys, clock, anc = hhl(
            sys,
            clock,
            anc,
            unitary=rz_unitary,
            inv_unitary=rz_unitary_inv,
            scaling=C,
            phase_scale=phase_scale,
            supported_raw_bins=(1, 3),
            strict=True,
            alpha=alpha,
        )
        return qmc.measure(anc)

    sv_i = simulate_hhl(_circuit, alpha_val)
    sys_i = extract_postselected_system(sv_i, 1, 2)

    b_n = np.array(amps) / np.linalg.norm(amps)
    x_ex = np.array([b_n[0] / lambda_0, b_n[1] / lambda_1])
    x_ex_norm = np.linalg.norm(x_ex)

    if x_ex_norm > 1e-15 and np.linalg.norm(sys_i) > 1e-15:
        f_i = float(
            np.abs(
                np.vdot(x_ex / x_ex_norm, sys_i / np.linalg.norm(sys_i))
            )
            ** 2
        )
    else:
        f_i = (
            1.0
            if x_ex_norm < 1e-15 and np.linalg.norm(sys_i) < 1e-15
            else 0.0
        )

    print(
        f"|b> = {str(amps):>12s}  ({label:>12s}):  fidelity = {f_i:.6f}"
    )
