# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# tags: [algorithm, primitive, simulation]
# ---
#
# # Suzuki–Trotter分解によるハミルトニアンシミュレーション (Rabi振動)
#
# 量子系の時間発展$e^{-iHt}$をシミュレーションすることは、量子コンピュータの代表的な応用の1つです。ハミルトニアンが非可換な部分に分割されるとき、つまり$H = A + B$かつ$[A, B] \neq 0$のとき、素朴な分解$e^{-i(A+B)t} = e^{-iAt}\,e^{-iBt}$は成立しません。そこで、一般には**Suzuki-Trotter積公式**と呼ばれる近似を使用し、各項の短時間発展を交互に並べます。近似の誤差(Trotter誤差)は、近似の次数を上げるほど小さくなっていきます。1次の形はLie-Trotter積公式{cite:p}`10.1090/S0002-9939-1959-0108732-6`に対応し、高次の再帰的構成は文献{cite:p}`10.1007/BF01609348`で与えられています。
# 本記事では、Trotter誤差が測定しやすい1量子ビットのRabi振動のハミルトニアンを題材に、Trotter分解によるハミルトニアンシミュレーションをQamomileで実装する例を示します。実装は、スクラッチ実装と、組み込みの`trotterized_time_evolution`関数を使用する例の2つを紹介します。実装した量子回路を実行し、収束次数($S_k$のフィデリティ誤差が$\Delta t^{2k}$でスケールすること)を、厳密と照合して確認します。

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install "qamomile[qiskit,visualization]"

# %%
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm import trotterized_time_evolution
from qamomile.qiskit import QiskitTranspiler

# %% [markdown]
# ## 問題設定: Rabi振動
#
# 最小の非自明なハミルトニアンシミュレーション問題として、非可換な2つのハミルトニアン項を持つ1量子ビット系を扱います。これにより、回路を小さく保ちながらTrotter誤差を観測できます。
#
# ### ハミルトニアン
#
# 共鳴駆動される2準位系は次のハミルトニアンで記述されます:
#
# $$ H = \underbrace{\tfrac{\omega}{2} Z}_{H_z} + \underbrace{\tfrac{\Omega}{2} X}_{H_x}. $$
#
# $[Z, X] \neq 0$なので、$H$を$H_z$と$H_x$に分割するとTrotter誤差が生じ、1量子ビットでも明確に観測できます。初期状態$|0\rangle$から始めると、励起確率は$P_{|1\rangle}(t) = (\Omega/E)^2 \sin^2(Et/2)$に従って振動します。ただし$E = \sqrt{\omega^2 + \Omega^2}$です。
#
# Qamomileでは、$H_z$と$H_x$を2つの`Observable`として構築し、Pythonのリストに詰めます。`@qkernel`ではそのリストを`Vector[Observable]`として宣言します。トランスパイル時にリストをバインドすると、`Hs.shape[0]`上の反復は項ごとの`pauli_evolve`呼び出しへ展開されます。

# %%
omega = 1.2
Omega = 0.8
T = 1.5

Hz = 0.5 * omega * qm_o.Z(0)
Hx = 0.5 * Omega * qm_o.X(0)
Hs = [Hz, Hx]

# %% [markdown]
# ### $[H_z, H_x] \neq 0$ の確認
#
# Trotter近似が必要になるのは$H_z$と$H_x$が可換でないからです。`qamomile.observable.commutator(a, b)`はハミルトニアン同士の交換子$[a, b] = a b - b a$を直接計算します。内部では各Pauli列ペアを1度だけ走査し、qubitパリティの規則(2つのPauli列は、両方とも非恒等かつ異なるPauliが乗っているqubitの数が奇数のときにだけ反交換)に従って、可換なペアを積を作る前に落とします。`Hz * Hx - Hx * Hz`のように一旦展開してから打ち消す素朴な計算より軽く、結果として完全に簡約された`Hamiltonian`が返ってくるので、そのまま検査したり解析値と比較したりできます。
#
# Rabiハミルトニアンの場合、教科書的な値は
#
# $$ [H_z, H_x] \;=\; \tfrac{\omega \Omega}{4}\,[Z, X] \;=\; i\,\tfrac{\omega \Omega}{2}\, Y $$
#
# であり、`commutator`はこれを厳密に再現します:

# %%
comm_zx = qm_o.commutator(Hz, Hx)
print(comm_zx)

expected = 1j * 0.5 * omega * Omega * qm_o.Y(0)
assert comm_zx == expected

# %% [markdown]
# ### 厳密な参照状態
#
# 2x2の行列指数関数で、厳密な状態$|\psi(T)\rangle = e^{-iHT}|0\rangle$を直接計算します。各Trotter近似は**フィデリティ誤差**$1 - |\langle\psi_\text{exact}|\psi_\text{trotter}\rangle|$によってこの状態と比較します。

# %%
ket0 = np.array([1.0, 0.0], dtype=complex)
X_mat = np.array([[0, 1], [1, 0]], dtype=complex)
Z_mat = np.array([[1, 0], [0, -1]], dtype=complex)
Hz_mat = 0.5 * omega * Z_mat
Hx_mat = 0.5 * Omega * X_mat
H_mat = Hz_mat + Hx_mat


def evolve(hamiltonian: np.ndarray, time: float) -> np.ndarray:
    return expm(-1j * time * hamiltonian)


sv_exact = evolve(H_mat, T) @ ket0


# %% [markdown]
# ## アルゴリズム: Trotterシミュレーション
#
# このアルゴリズムでは、全体の時間発展を各ハミルトニアン項による時間発展の積で近似します。まず1次のLie-Trotter分解{cite:p}`10.1090/S0002-9939-1959-0108732-6`から始め、対称化された2次公式を示します。さらに再帰を用いることで任意の偶数に対する公式である、Suzuki-Trotter分解{cite:p}`10.1007/BF01609348`を得られることを示します。
#
# ### $S_1$: 1次Suzuki–Trotter分解 (Lie–Trotter分解)
#
# もっとも単純な分解は
#
# $$ S_1(\Delta t) = e^{-i H_x \Delta t}\, e^{-i H_z \Delta t} $$
#
# で、これを$N$回適用して全発展時間$T = N \Delta t$をシミュレーションします。1ステップあたりの局所誤差は$O(\Delta t^2)$、$N = T/\Delta t$ステップにわたる大域的な状態ノルム誤差は$O(\Delta t)$です。
#
#
# ### $S_2$: 2次Suzuki–Trotter分解
#
# 中央の項を中心にステップを対称化すると先頭の誤差項が消えます:
#
# $$ S_2(\Delta t) = e^{-i H_z \Delta t/2}\, e^{-i H_x \Delta t}\, e^{-i H_z \Delta t/2}. $$
#
# 局所誤差は$O(\Delta t^3)$、大域的な状態ノルム誤差は$O(\Delta t^2)$になります。このステップは、$H_z$による半ステップ、$H_x$による全ステップ、もう一度$H_z$による半ステップから成ります。
#
# ### 高次のSuzuki–Trotter分解:フラクタル再帰
#
# 文献{cite:p}`10.1007/BF01609348`では、任意の偶数次に対するTrotter近似を$S_2$から**再帰的に**構築する方法が与えられています。各段で5つのリスケーリングされたコピーを入れ子にします:
#
# $$ S_{2k}(\Delta t) = S_{2k-2}(p_k \Delta t)^2 \, S_{2k-2}\bigl((1 - 4 p_k)\Delta t\bigr) \, S_{2k-2}(p_k \Delta t)^2. $$
#
# ここで段ごとの係数は
#
# $$ p_k = \frac{1}{4 - 4^{1/(2k-1)}} $$
#
# です。$p_k$は下位の公式の$(2k-1)$次誤差がキャンセルされるように選ばれているので、1ステップあたりの局所誤差は$O(\Delta t^{2k+1})$になります。**係数は段ごとに必ず計算し直す必要があります。** 具体的には:
#
# - $k=2$ (4次): $p_2 = 1/(4 - 4^{1/3}) \approx 0.4145$
# - $k=3$ (6次): $p_3 = 1/(4 - 4^{1/5}) \approx 0.3731$
# - $k=4$ (8次): $p_4 = 1/(4 - 4^{1/7}) \approx 0.3596$
#
# すべての段で$p_2$を使い回すと$(2k-1)$次の誤差項が残ったままになり、結果として得られる公式は$S_4$とほぼ同等の精度しか持ちません。これはSuzuki-Trotterを手書きで実装するときにハマりがちな罠です。

# %% [markdown]
# ## Qamomileでの実装
#
# Qamomileは、Suzuki-Trotter分解に基づくTrotterシミュレーションを実装した`trotterized_time_evolution`関数を提供しています。
# ここでは、まずTrotter公式をQamomileの量子カーネルとして実装し、上で見たアルゴリズムに従ってどのようにQamomileで実装することができるかを確認します。
# その後に、`trotterized_time_evolution`関数を使用したシンプルな実装例を見ていきます。
#
# ### スクラッチ実装
#
# スクラッチ実装では、上の数式をQamomileの量子カーネルとして直接写します。`rabi_s1`と`rabi_s2`は、量子ビットレジスタを`pauli_evolve`へ渡しながら、時間幅`dt`のスライスを`n_steps`回繰り返します。
#
# $S_1$では、各ループで量子ビットレジスタを$H_z$、$H_x$の順に通します。$S_2$では、同じループの中で対称な半ステップ、全ステップ、半ステップのスケジュールを直接適用します。


# %%
@qmc.qkernel
def rabi_s1(
    Hs: qmc.Vector[qmc.Observable], dt: qmc.Float, n_steps: qmc.UInt
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(1, "q")
    for _ in qmc.range(n_steps):
        q = qmc.pauli_evolve(q, Hs[0], dt)
        q = qmc.pauli_evolve(q, Hs[1], dt)
    return qmc.measure(q)


# %%
@qmc.qkernel
def rabi_s2(
    Hs: qmc.Vector[qmc.Observable], dt: qmc.Float, n_steps: qmc.UInt
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(1, "q")
    for _ in qmc.range(n_steps):
        q = qmc.pauli_evolve(q, Hs[0], 0.5 * dt)
        q = qmc.pauli_evolve(q, Hs[1], dt)
        q = qmc.pauli_evolve(q, Hs[0], 0.5 * dt)
    return qmc.measure(q)


# %% [markdown]
# 数学的な再帰も、対象とする次数を`UInt`パラメータとして受け取り、再帰分岐で`order - 2`を渡して自分自身を呼び出す`@qkernel`にそのまま翻訳できます。`order == 2`の基底ケースでは2次の半ステップ、全ステップ、半ステップのスケジュールを直接適用し、それ以外では5つの入れ子になった呼び出しでSフラクタル再帰を作ります。
#
# Qamomileのトランスパイラは、具体的な`order`バインドの下でinlineとpartial evaluationの固定点ループを走らせることで、自己再帰する量子カーネルを解決します。各反復で`CallBlockOp`を1層展開し、現在の`order`の値を使って基底ケースの`if`を畳み込みます。生成される回路は再帰レベルの数にかかわらずフラットになるため、トランスパイル時に`order=8`をバインドすると、手で公式を生成しなくても具体的な8次Suzuki回路が得られます。
#
# ここで、注意点が2つあります。
#
# - **`order`はトランスパイル時に具体値である必要があります。** バインドがないと基底ケースの`if`が畳み込まれず、unrollループが停止できません。その場合、トランスパイラはIRに自己呼び出しを残し、backendのemitで拒否されます。
# - **停止しない再帰は検出されます。** 本体が`order - 2`ではなく`order + 2`で自分自身を呼ぶ場合や、基底ケースに到達しない場合、unrollループは深さの上限に達した時点で`FrontendTransformError`を送出します。これは古典的な`RecursionError`に相当します。


# %%
@qmc.qkernel
def suzuki_trotter(
    order: qmc.UInt,
    q: qmc.Vector[qmc.Qubit],
    Hs: qmc.Vector[qmc.Observable],
    dt: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    if order == 2:
        q = qmc.pauli_evolve(q, Hs[0], 0.5 * dt)
        q = qmc.pauli_evolve(q, Hs[1], dt)
        q = qmc.pauli_evolve(q, Hs[0], 0.5 * dt)
    else:
        p = 1.0 / (4.0 - 4.0 ** (1.0 / (order - 1)))
        w = 1.0 - 4.0 * p
        q = suzuki_trotter(order - 2, q, Hs, p * dt)
        q = suzuki_trotter(order - 2, q, Hs, p * dt)
        q = suzuki_trotter(order - 2, q, Hs, w * dt)
        q = suzuki_trotter(order - 2, q, Hs, p * dt)
        q = suzuki_trotter(order - 2, q, Hs, p * dt)
    return q


# %%
@qmc.qkernel
def rabi_suzuki(
    order: qmc.UInt,
    Hs: qmc.Vector[qmc.Observable],
    dt: qmc.Float,
    n_steps: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(1, "q")
    for _ in qmc.range(n_steps):
        q = suzuki_trotter(order, q, Hs, dt)
    return qmc.measure(q)


# %% [markdown]
# `rabi_suzuki`は、トランスパイル時の`order`バインドを選ぶだけで$S_2$、$S_4$、$S_6$、$S_8$、…を生成する単一の量子カーネルです。次数ごとに別々の量子カーネルを書く必要はありません。
#
# ### `trotterized_time_evolution`関数
#
# Qamomileには、Suzuki–Trotter分解をまとめて扱う`qamomile.circuit.algorithm`の`trotterized_time_evolution`があります。このヘルパーは、量子ビットレジスタ、ハミルトニアン項のリスト、近似次数`order`、全発展時間`gamma`、Trotterスライス数`step`を受け取り、幅`gamma / step`のスライスを`step`回適用します。
#
# 下の量子カーネルは、このヘルパーを呼び出して最後に測定するラッパーです。`order = 1`ならLie–Trotter分解、正の偶数(`2`, `4`, `6`, …)なら対応する高次公式を選択できます。


# %%
@qmc.qkernel
def rabi_from_algorithm(
    Hs: qmc.Vector[qmc.Observable],
    gamma: qmc.Float,
    order: qmc.UInt,
    step: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(1, name="q")
    q = trotterized_time_evolution(q, Hs, order, gamma, step)
    return qmc.measure(q)


# %% [markdown]
# `Hs`をバインドすると具体的なハミルトニアン項が渡され、`order`と`step`のバインドで公式と分割数が決まります。以降の結果セクションでは、このヘルパーベースの量子カーネルで$S_1$、$S_2$、$S_4$、$S_6$を比較します。

# %% [markdown]
# ## 結果
#
# 各近似を厳密な状態ベクトルと比較し、期待される収束次数を確認します。この1量子ビットのベンチマークでは、同じSuzuki-Trotter積公式を2x2行列として評価できます。上のQamomile量子カーネルは回路の定義に使い、数値誤差の確認にはSDKの状態ベクトルシミュレータを使いません。
#

# %%
FLOAT64_FLOOR = 1e-15


def suzuki_trotter_matrix(order: int, dt: float) -> np.ndarray:
    if order == 1:
        return evolve(Hx_mat, dt) @ evolve(Hz_mat, dt)
    if order == 2:
        return (
            evolve(Hz_mat, 0.5 * dt)
            @ evolve(Hx_mat, dt)
            @ evolve(Hz_mat, 0.5 * dt)
        )

    p = 1.0 / (4.0 - 4.0 ** (1.0 / (order - 1)))
    w = 1.0 - 4.0 * p
    return (
        suzuki_trotter_matrix(order - 2, p * dt)
        @ suzuki_trotter_matrix(order - 2, p * dt)
        @ suzuki_trotter_matrix(order - 2, w * dt)
        @ suzuki_trotter_matrix(order - 2, p * dt)
        @ suzuki_trotter_matrix(order - 2, p * dt)
    )


def trotter_state(order: int, n_steps: int) -> np.ndarray:
    step = suzuki_trotter_matrix(order, T / n_steps)
    return np.linalg.matrix_power(step, n_steps) @ ket0


def fidelity_error(reference: np.ndarray, state: np.ndarray) -> float:
    return max(1.0 - abs(np.vdot(reference, state)), FLOAT64_FLOOR)


# %% [markdown]
#
# ### $N = 8$での簡易チェック
#
# 収束性のスイープを行う前に、ヘルパーベースのQamomile量子カーネルを各公式について一度トランスパイルし、対応する2x2行列積を$N = 8$で計算します。すべての公式で`rabi_from_algorithm`を使い、`order`バインドだけを変えます。

# %%
tr = QiskitTranspiler()
N_demo = 8
trotter_orders = {"S1": 1, "S2": 2, "S4": 4, "S6": 6}

for name, order in trotter_orders.items():
    exe = tr.transpile(
        rabi_from_algorithm,
        bindings={"Hs": Hs, "gamma": T, "order": order, "step": N_demo},
    )
    assert len(exe.compiled_quantum) == 1
    sv = trotter_state(order, N_demo)
    err = fidelity_error(sv_exact, sv)
    print(f"{name} at N={N_demo}: fidelity error = {err:.3e}")

# %% [markdown]
# ### 近似の次数と収束性
#
# Trotterシミュレーションにおける誤差は、ステップ幅が小さいほど、そして近似の次数が高いほど小さくなっていきます。
# Trotterステップ数$N$を掃引し、ステップ幅$\Delta t = T / N$に対してフィデリティ誤差を両対数軸でプロットします。期待される傾きは以下のとおりです:
#
# | 公式 | 局所誤差 | 大域ノルム誤差 | フィデリティ誤差 ($1 -$ 内積) |
# |-----|---------|---------------|------------------------------|
# | $S_1$ | $O(\Delta t^2)$ | $O(\Delta t)$   | $O(\Delta t^2)$  |
# | $S_2$ | $O(\Delta t^3)$ | $O(\Delta t^2)$ | $O(\Delta t^4)$  |
# | $S_4$ | $O(\Delta t^5)$ | $O(\Delta t^4)$ | $O(\Delta t^8)$  |
# | $S_6$ | $O(\Delta t^7)$ | $O(\Delta t^6)$ | $O(\Delta t^{12})$ |
#
# フィデリティ誤差は状態ノルム誤差の**2乗**になります(先頭項)。2つのベクトルが十分近ければ$1 - |\langle a | b \rangle| \approx \tfrac{1}{2}\lVert a - b \rVert^2$なので、以下のプロットに現れる傾きは$1, 2, 4$ではなく$2, 4, 8$です。

# %%
Ns = np.array([2, 4, 8, 16, 32, 64])
all_names = list(trotter_orders)
errors: dict[str, Any] = {name: [] for name in all_names}

for N in Ns:
    for name, order in trotter_orders.items():
        sv = trotter_state(order, int(N))
        errors[name].append(fidelity_error(sv_exact, sv))

errors = {k: np.asarray(v) for k, v in errors.items()}
dts = T / Ns


# %%
def fit_slope(dts, errs, n_points):
    return np.polyfit(np.log(dts[:n_points]), np.log(errs[:n_points]), 1)[0]


# %%
slope_s1 = fit_slope(dts, errors["S1"], len(Ns))
slope_s2 = fit_slope(dts, errors["S2"], len(Ns))
slope_s4 = fit_slope(dts, errors["S4"], 3)
print(f"Fitted slopes:  S1 = {slope_s1:.2f}  S2 = {slope_s2:.2f}  S4 = {slope_s4:.2f}")
print(f"S6 fidelity error at largest dt: {errors['S6'][0]:.3e}")

# 期待される次数を保証し、ベンチマークの意図しない変更をdocs testで検出できるようにします
assert 1.7 < slope_s1 < 2.3, slope_s1
assert 3.7 < slope_s2 < 4.3, slope_s2
assert 7.0 < slope_s4 < 9.0, slope_s4
# S6は1量子ビットのこの問題ではfloat64の精度下限に張り付くので、
# 同じdtにおけるS4の主要誤差と比べて十分小さいことだけ確認します。
assert abs(errors["S6"][0]) < 1e-10, errors["S6"][0]

# %%
fig, ax = plt.subplots(figsize=(6, 4))
markers = {"S1": "o", "S2": "s", "S4": "^", "S6": "D"}
for name in all_names:
    ax.loglog(dts, errors[name], marker=markers[name], label=name)
ax.axhline(1e-15, color="black", linestyle=":", linewidth=1.8, label="float64 floor")
ax.set_xlim(1e-2, 1e0)
ax.set_xlabel(r"step size $\Delta t = T / N$")
ax.set_ylabel(
    r"fidelity error $1 - |\langle \psi_{\rm exact} | \psi_{\rm trotter} \rangle|$"
)
ax.set_title("Trotter convergence on Rabi oscillation")
ax.grid(True, which="both", linewidth=0.3)
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# プロット上の各直線の傾きは$\approx 2, 4, 8$で、上の表のフィデリティ誤差の次数と一致しています。$S_4$は$N = 16$の時点でfloat64の精度下限に到達し、$S_6$はスイープの全域で精度下限に張り付いているため直線が平坦に見えます(期待される$\Delta t^{12}$の傾きは、1量子ビット問題を倍精度で解く限り検出できません)。
# 実際には、近似の次数は要求される精度と計算コストのバランスによって決定されます。次数が高いほど、精度は良くなりますが、1ステップあたりの量子ゲート数は増加します。これは、特にNISQデバイスを使用する際には回路が深いほどハードウェアのノイズの影響を受けるため重要です。

# %% [markdown]
# ## まとめ
#
# このノートブックでは、以下の内容を学びました:
#
# - Suzuki–Trotter分解は、非可換なハミルトニアン項による時間発展を、1次のLie–Trotter公式から再帰的に構成される高い偶数次の公式まで近似します。
# - Qamomileの`trotterized_time_evolution`関数は、ハミルトニアン項のリスト、近似次数、全発展時間、Trotter分割数を指定して、対応する積公式を適用します。
# - 数値実験では、各近似と厳密なRabi振動を比較し、float64の精度下限に達するまで、検証した次数に対応する収束挙動を確認しました。
