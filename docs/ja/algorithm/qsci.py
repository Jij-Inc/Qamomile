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
# tags: [algorithm, sample-based]
# ---
#
# # 量子選択配置間相互作用法（QSCI）
#
# **量子選択配置間相互作用法（Quantum-Selected Configuration Interaction; QSCI）**{cite:p}`10.1103/dmn4-snfx`は、量子状態からサンプリングしたビット列を使って小さな有効ハミルトニアンを構築し、それを古典コンピュータで厳密に対角化するハイブリッド量子古典アルゴリズムです。
# このチュートリアルでは、4量子ビットの横磁場イジング模型に対して、QSCIのワークフローをQamomileで実装します。量子状態の準備とサンプリングはQURI PartsのQulacsシミュレータで実行し、部分空間の構築と対角化は`qamomile.linalg.solve_subspace`を使います。

# %%
# 最新のQamomileをQURI Partsおよび可視化の追加依存関係とともにpipからインストールします。
# # !pip install "qamomile[quri_parts,visualization]"

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.basic import cx_entangling_layer, ry_layer
from qamomile.linalg import solve_subspace
from qamomile.quri_parts import QuriPartsExecutor, QuriPartsTranspiler

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# ## 問題設定: 1次元横磁場イジング模型
#
# 1次元鎖上の4量子ビット横磁場イジング模型を使います:
#
# $$
# H \;=\; -J \sum_{i=0}^{n-2} Z_i Z_{i+1} \;-\; h \sum_{i=0}^{n-1} X_i,
# \quad J = 1,\; h = 0.7.
# $$
#
# $2^4=16$次元のヒルベルト空間は十分小さいので、NumPyで厳密な基底状態エネルギーを直接計算してQSCIの基準値として使えます。

# %%
n_qubits = 4
J = 1.0
h_field = 0.7

H = qm_o.Hamiltonian(num_qubits=n_qubits)
for i in range(n_qubits - 1):
    H += qm_o.Z(i) * qm_o.Z(i + 1) * (-J)
for i in range(n_qubits):
    H += qm_o.X(i) * (-h_field)

exact_eigvals = np.linalg.eigvalsh(H.to_numpy())
E_exact = float(exact_eigvals[0])
print(f"厳密な基底状態エネルギー: {E_exact:.6f}")
# eigvalshは昇順で固有値を返すのでexact_eigvals[0]が基底状態。
# 4量子ビット横磁場イジング模型のヒルベルト空間は2^n_qubits次元。
assert H.num_qubits == n_qubits
assert exact_eigvals.shape == (2**n_qubits,)
assert E_exact == float(exact_eigvals.min())

# %% [markdown]
# ## アルゴリズム
#
# **量子選択配置間相互作用法（QSCI）は**、量子コンピュータ上で準備した量子状態を計算基底で測定し、出現頻度の高いビット列を選んで部分空間を構築し、その部分空間で有効ハミルトニアンを古典的に対角化することで、基底状態エネルギーの推定値を得るハイブリッド量子古典アルゴリズムです。
#
# {cite:t}`10.1103/dmn4-snfx`で提案されたアルゴリズムの手順は次の通りです。
#
# 1. 量子コンピュータ上で入力状態$|\psi_{\mathrm{in}}\rangle$を準備する（典型的には大まかに最適化されたVQEによる量子状態）。
# 2. $|\psi_{\mathrm{in}}\rangle$を計算基底で多数回測定する。
# 3. 出現頻度が高い上位$K$個のビット列を離散部分空間$\{|s_i\rangle\}_{i=1}^{K}$として選ぶ。
# 4. 有効ハミルトニアン$H^{\mathrm{sub}}_{ij} = \langle s_i | H | s_j \rangle$を構築し、古典的に対角化する。
#
# ハミルトニアンの基底状態を求めるハイブリッド量子古典アルゴリズムは、VQE{cite:p}`10.1038/ncomms5213`が広く知られていますが、VQEと比べた利点は結果が厳密な**変分原理の保証**、すなわち次の性質を引き継ぐ点にあります。ノイズのあるハードウェア上でも、
#
# $$
# E_{\mathrm{QSCI}} \;\geq\; E_{\mathrm{exact}}.
# $$
#
# が保証されます。さらに、部分空間のサイズ$K$を増やすと、QSCIエネルギーは単調非増加で、厳密な基底状態エネルギーに収束します。
# QSCIのもう一つの特徴として、VQEのように量子状態のパラメータを完全に最適化する必要がない点が挙げられます。入力状態$|\psi_{\mathrm{in}}\rangle$は、ランダムなパラメータでも、あるいはVQEで大まかに最適化された状態でも構いません。QSCIは、入力状態のサンプリング分布が真の基底状態を支配するビット列に集中している限り、部分空間の情報量を増やすことができます。この利点は、NISQデバイスのようなノイズの多いハードウェア上で、VQEの最適化が困難な場合に特に有効です。

# %% [markdown]
# ## Qamomileによる実装
#
# それでは、Qamomileを使ってQSCIのワークフローを実装していきます。Qamomileは、QSCIのような部分空間法を簡単に実装するためのサブルーチン`solve_subspace`を提供しています。`solve_subspace`は、サンプルビット列のリストとハミルトニアンを受け取り、部分空間での有効ハミルトニアンを構築し、古典的に対角化して固有値と固有ベクトルを返します。
#
# ### 初期状態準備のためのVQEアンザッツ
#
# QSCIの初期状態にはハードウェア効率的なアンザッツを使用します。これは単純な交互レイヤー型アンザッツで、各レイヤーは全量子ビットへの$R_y$と直線状に並べたCNOTゲートを適用し、最後にもう一度$R_y$レイヤーを置きます。以下では、3つの補助量子カーネルを定義します。
#
# - `ansatz_state`は$|\psi(\theta)\rangle$の量子ビットレジスタを構築します。
# - `ansatz_energy`はVQE用に$\langle\psi|H|\psi\rangle$を返します。
# - `ansatz_measure`はQSCIのサンプリング用に状態を計算基底で測定します。


# %%
@qmc.qkernel
def ansatz_state(
    n: qmc.UInt,
    reps: qmc.UInt,
    thetas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(n, name="q")
    for r in qmc.range(reps):
        q = ry_layer(q, thetas, r * n)
        q = cx_entangling_layer(q)
    final_base = reps * n
    q = ry_layer(q, thetas, final_base)
    return q


@qmc.qkernel
def ansatz_energy(
    n: qmc.UInt,
    reps: qmc.UInt,
    thetas: qmc.Vector[qmc.Float],
    H: qmc.Observable,
) -> qmc.Float:
    q = ansatz_state(n, reps, thetas)
    return qmc.expval(q, H)


@qmc.qkernel
def ansatz_measure(
    n: qmc.UInt,
    reps: qmc.UInt,
    thetas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    q = ansatz_state(n, reps, thetas)
    return qmc.measure(q)


# %% [markdown]
# ### 量子カーネルのコンパイル
#
# 作成した量子カーネルをシミュレータで実行するためにコンパイルします。この例では、QURI Partsの`QuriPartsTranspiler`を使ってコンパイルします。コンパイル後の量子カーネルは、`QuriPartsExecutor`を使って実行できます。

# %%
transpiler = QuriPartsTranspiler()
executor = QuriPartsExecutor(seed=42)

reps = 2
n_params = (reps + 1) * n_qubits

energy_exec = transpiler.transpile(
    ansatz_energy,
    bindings={"n": n_qubits, "reps": reps, "H": H},
    parameters=["thetas"],
)
sample_exec = transpiler.transpile(
    ansatz_measure,
    bindings={"n": n_qubits, "reps": reps},
    parameters=["thetas"],
)

# %% [markdown]
# ### QSCIのワークフローの構築
#
# それでは、QSCIのワークフローを構築していきます。まず、短いVQEを実行して入力状態$|\psi_{\mathrm{in}}\rangle$を準備し、その後、Z基底でビット列をサンプリングします。最後に、出現頻度の高いビット列から部分空間を構築し、古典的に対角化してQSCIエネルギーを求めます。
#
# #### ステップ1: 短いVQEで$|\psi_{\mathrm{in}}\rangle$を準備
#
# QSCIは入力状態の最適化が甘くても頑健で、ランダムパラメータでも意味のある部分空間が得られますが、短いVQEを走らせるとサンプリング分布が真の基底状態を支配するビット列に集中するため、同じ$K$でも部分空間の情報量が大きくなります。ここではCOBYLAによる最適化を数回だけ走らせます。


# %%
def cost_fn(params: np.ndarray) -> float:
    return energy_exec.run(executor, bindings={"thetas": list(params)}).result()


rng = np.random.default_rng(0)
init_params = rng.uniform(0, 2 * np.pi, n_params)
assert init_params.shape == (n_params,)

maxiter = max(n_params + 2, 5 if docs_test_mode else 80)
result = minimize(
    cost_fn,
    init_params,
    method="COBYLA",
    options={"maxiter": maxiter, "rhobeg": 0.5},
)
opt_params = result.x
print(f"VQEエネルギー = {result.fun:+.6f}   (E_exactとの差: {result.fun - E_exact:.4e})")
# 変分原理により、COBYLAの予算がいくら短くてもVQEエネルギーはE_exactの上界。
assert result.fun >= E_exact - 1e-9
assert opt_params.shape == (n_params,)

# %% [markdown]
# #### ステップ2: Z基底でビット列をサンプリング
#
# 部分空間を構築するために、VQEで得られたパラメータを使って、量子状態$|\psi_{\mathrm{in}}\rangle$をZ基底で測定し、ビット列をサンプリングします。サンプルの出現頻度を数え、上位$K$個のビット列を選びます。
# 各サンプルはタプル`(b_0, ..., b_{n-1})`で、$q$番目の要素は量子ビット$q$のZ固有値インデックスです。

# %%
shots = 500 if docs_test_mode else 4000
sample_results = (
    sample_exec.sample(executor, bindings={"thetas": list(opt_params)}, shots=shots)
    .result()
    .results
)
sample_results.sort(key=lambda bc: bc[1], reverse=True)
print(f"サンプリングされた異なるビット列数: {len(sample_results)}")
for bits, c in sample_results[:5]:
    print(f"  {bits}  回数={c}")
# 異なるビット列数はヒルベルト空間の次元を超えず、全ショットがカウントされ、
# 各ビット列はn_qubits長。
assert len(sample_results) <= 2**n_qubits
assert sum(c for _, c in sample_results) == shots
assert all(len(bits) == n_qubits for bits, _ in sample_results)


# %% [markdown]
# #### ステップ3,4: QSCI部分空間の構築と対角化
#
# 得られたサンプルビット列の出現頻度を数え、上位$K$個のビット列を選びます。次に、`solve_subspace`を使って有効ハミルトニアン$H^{\mathrm{sub}}_{ij} = \langle s_i|H|s_j\rangle$を構築し、古典的に対角化します。`solve_subspace`はベクトル化された排他的論理和・偶奇計算のルーチンで$H^{\mathrm{sub}}_{ij} = \langle s_i|H|s_j\rangle$を構築し、`numpy.linalg.eigh`を実行します。返ってくる最小固有値がQSCIエネルギーの推定値で、変分原理によって任意の$K$に対し$E_{\mathrm{QSCI}}(K) \geq E_{\mathrm{exact}}$が保証されます。
#
# :::{note}
# `solve_subspace`は内部で`subspace_hamiltonian`という関数を適用します。この関数は行列積を必要としません。各パウリ項は1つの排他的論理和マスクと偶奇符号として寄与し、$K^2$個のサンプルペア全体にわたってベクトル化されます。重複したサンプルビット列は上記の一意なビット列のリストから除外されます。結果として得られる部分空間は条件数が良く、`solve_subspace`は通常のエルミート固有値分解を返します。
# :::

# %%
unique_bitstrings = [bits for bits, _ in sample_results]
K_max = len(unique_bitstrings)
ks = sorted({k for k in (1, 2, 4, 8, 16, K_max) if k <= K_max})

energies = [float(solve_subspace(unique_bitstrings[:K], H)[0][0]) for K in ks]

for K, E in zip(ks, energies):
    print(f"K = {K:3d}   E_QSCI = {E:+.6f}   差 = {E - E_exact:+.3e}")

assert all(E >= E_exact - 1e-9 for E in energies), "変分上界に違反しています"
# 固有値の交互配置則により、部分空間を広げると最小固有値は下がる（または変わらない）ので、
# QSCIエネルギーはKに対して単調非増加。
assert all(energies[i] >= energies[i + 1] - 1e-9 for i in range(len(energies) - 1))
assert len(energies) == len(ks)

# %% [markdown]
# ## 結果
#
# QSCIの収束を可視化します。横軸は部分空間サイズ$K$、縦軸はQSCIエネルギーです。破線は厳密な基底状態エネルギーを示します。

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ks, energies, "-o", label=r"$E_{\mathrm{QSCI}}$")
ax.axhline(E_exact, color="black", linestyle="--", label=r"$E_{\mathrm{exact}}$")
ax.set_xlabel("Subspace size $K$")
ax.set_ylabel("Energy")
ax.set_title("QSCI convergence — 4-qubit TFIM ($J{=}1,\\;h{=}0.7$)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# 結果から、QSCIエネルギーは部分空間サイズ$K$に対して単調非増加であり、厳密な基底状態エネルギーに近づくことが確認できます。

# %% [markdown]
# ## まとめ
#
# このチュートリアルでは、4量子ビットの1次元横磁場イジング模型を題材に、QSCIの原理からQamomileによる実装と精度評価までを学びました。
#
# - **QSCI:** 入力した量子状態を計算基底で測定し、出現頻度の高いビット列から部分空間を定義して、その部分空間に射影したハミルトニアンを古典的に対角化します。得られるエネルギーは変分原理に従うため、入力状態にノイズがある場合や最適化が不十分な場合でも、厳密な基底状態エネルギーの上界になります。
# - **Qamomileによる実装:** Qamomileの量子カーネルを使って入力状態を準備するVQEとZ基底でのサンプリングを実装し、QURI Partsを使って実行しました。Qamomileの`qamomile.linalg.solve_subspace`を使うと、サンプルビット列から有効ハミルトニアンを構築し、古典的な対角化で得られる固有値と固有ベクトルを簡単に取得できます。
# - **部分空間の大きさと精度:** 4量子ビットの1次元横磁場イジング模型を使った実験では、QSCIエネルギーが厳密値を上回る変分上界を保ち、部分空間サイズ$K$を増やすほど単調非増加になることを確認しました。より多くのサンプルビット列を選ぶことで部分空間の表現力が高まり、エネルギー推定値が厳密な基底状態エネルギーに近づきました。
