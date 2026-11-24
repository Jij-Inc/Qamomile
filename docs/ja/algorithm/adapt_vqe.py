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
# tags: [algorithm, chemistry, variational]
# ---
#
# # ADAPT-VQEによる水素分子の基底状態エネルギー計算
#
# このチュートリアルでは、Qamomile を用いて ADAPT-VQE を実装し、水素分子（$\mathrm{H}_2$）の STO-3G 基底での基底状態エネルギーを 8 つの原子間距離で計算します。
# 得られたエネルギー曲線を、同じ基底で求めた完全配置間相互作用（full configuration interaction、FCI）の結果と比較します。\
# {cite:t}`grimsley2019` が提案した ADAPT-VQE は、分子に応じて演算子を1つずつ追加し、アンザッツを構築する手法です。
# アンザッツを事前に固定する VQE に対し、パラメータ数と回路の深さを抑えられる可能性があります。
# Qamomile で状態準備を記述し、Aer の厳密な推定器でエネルギーと演算子選択用の勾配を計算してみましょう。

# %% [markdown]
# ## インストール
#
# QamomileのQiskit対応に加え、分子ハミルトニアンを扱う OpenFermion・量子化学計算を行う PySCF・両者をつなぐ OpenFermion-PySCF が必要です。
# 以下のコマンドで、NumPy・SciPy・Matplotlib・Qiskit Aer を含む依存パッケージをインストールできます。
# Google Colab では、次のセルのインストールコマンドのコメントを外して実行してください。

# %%
# Qamomileと量子化学計算用の依存パッケージをインストールします。
# # !pip install "qamomile[qiskit]" openfermion pyscf openfermionpyscf

# %%
from __future__ import annotations

from itertools import combinations, product
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
from openfermion import (
    FermionOperator,
    commutator,
    hermitian_conjugated,
    jordan_wigner,
    normal_ordered,
)
from openfermion.chem import MolecularData
from openfermion.transforms import get_fermion_operator
from openfermionpyscf import run_pyscf
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2
from scipy.optimize import minimize

import qamomile as qm
import qamomile.observable as qmo
from qamomile.qiskit import Qiskit, to_circuit

# %% [markdown]
# ## 背景
#
# ### 量子化学計算と VQE
#
# 分子の電子状態を表す空間は、軌道数の増加とともに指数的に増大します。
# 量子計算でこの課題に取り組むには、十分な量子ビット数に加え、低いゲート・読み出し誤り率や長いコヒーレンス時間を持つハードウェアと、効率的な量子アルゴリズムが必要となります。\
# 変分量子固有値ソルバー（VQE）は、状態準備と測定を量子コンピュータ上で行い、パラメータを古典計算機で更新します。
# 測定回数は多くなりますが、浅い回路を利用できるため、NISQ デバイスでの実行に適しています。
# 一方、精度はアンザッツの表現能力に依存し、強相関系を高精度で扱うには適切なアンザッツの構成が重要となります。
#
# ### 先行研究: 適応的な結合クラスター法 (`@CC`)
#
# ADAPT-VQE の説明に入る前に、重要な先行研究である {cite:t}`lyakh2010` が提案した古典計算手法について触れておきましょう。
# 分子の電子状態を計算するために、初期条件として Hartree-Fock 法で得られた状態 $\vert \Phi_\mathrm{HF} \rangle$ を用います。
# 結合クラスター (CC) 法では、これに次のような演算子を作用させます。
#
# ```{math}
# :label: eq-adapt-cc
# \vert \Psi_\mathrm{CC} \rangle
# = e^T \vert \Phi_\mathrm{HF} \rangle
# ```
#
# ここで
#
# ```{math}
# :label: eq-adapt-cluster
# T
# = T_1 + T_2 + \cdots
# = \sum_{i, a} t_{ia} a_a^\dagger a_i + \frac{1}{4} \sum_{i, j, a, b} t_{ij}^{ab} a_a^\dagger a_b^\dagger a_j a_i + \cdots
# ```
#
# は励起演算子であり、$T_1$ は単一電子励起、$T_2$ は二電子励起、などに対応します。
# さらに $a_p^\dagger, a_p$ はそれぞれ軌道 $p$ に電子を生成・消滅させる演算子です。
# 結合クラスター法では、指数関数の形で励起演算子を考慮したものと見ることができます。
# 単一電子励起演算子のみを考慮する場合には CCS 法、二電子励起まで考慮する場合には CCSD 法、三電子励起まで考慮する場合には CCSDT 法、などと呼び方が変わります。
# 通常の CC 法では、計算精度を高めるために CCS -> CCSD -> CCSDT のように、独立に扱う励起の次数を増やします。
# しかし、この方法では同じ次数の中でも重要でない項まで全て含めて計算することになり、大量の振幅を扱う必要が出てきます。
# そこで `@CC` 法では、重要と思われる励起のみを追加する方針を取ります。
# 採用する励起の集合を $S$ とすると
#
# ```{math}
# :label: eq-adapt-selected-cluster
# T_S
# = \sum_{\mu \in S} t_\mu \tau_\mu, \quad
# \vert \Psi_S \rangle
# = e^{T_S} \vert \Phi_\mathrm{HF} \rangle
# ```
#
# のようにして、計算対象を設定することができます。
# 例えば計算例として
#
# ```{math}
# :label: eq-adapt-selected-ranks
# T_S
# = T_1 + T_2 + \sum_{\mu \in S_3} t_\mu \tau_\mu + \sum_{\nu \in S_4} t_\nu \tau_\nu
# ```
#
# のようなものが考えられます。
# このとき、どの振幅を採用するかを表す集合 $S$、そして採用した振幅の値 $t_\mu$ を調整する必要があります。
# `@CC` 法では重要度関数を用い、対象となる電子状態の記述に必要なクラスター振幅を選択します。
# これにより、全ての励起演算子を一律に取り込むのではなく、問題に応じて計算モデルを構築することを目指しました。
#
# ## ADAPT-VQEの仕組み
#
# それでは、ADAPT-VQE を詳しくみていきましょう。
# `@CC` 法と同様、出発点は Hartree-Fock 法で得られた $\vert \Phi_\mathrm{HF} \rangle$ です。
# これを用いると、CCSD は $\vert \Psi_\mathrm{CCSD} \rangle = e^{T_1 + T_2} \vert \Phi_\mathrm{HF} \rangle$ のように表されるのでした。
# この状態を量子回路で準備するために、ユニタリな CCSD (UCCSD) を考える必要があります。
# そこで、ユニタリな変換として
#
# ```{math}
# :label: eq-adapt-ucc-generator
# K
# = T_1 - T_1^\dagger + T_2 - T_2^\dagger
# = A_1 + A_2 \quad (A_1 \equiv T_1 - T_1^\dagger, A_2 \equiv T_2 - T_2^\dagger)
# ```
#
# を用い
#
# ```{math}
# :label: eq-adapt-ucc-state
# \lvert \Psi_\mathrm{UCCSD} \rangle
# = e^K \vert \Phi_\mathrm{HF} \rangle
# ```
#
# のようにします。
# 実際このようにすると、$K^\dagger = - K$ となり、$(e^K)^\dagger e^K = e^{-K} e^K = I$ のように、$e^K$ はユニタリとなります。
# 通常の単一電子励起・二電子励起では、Hartree-Fock 状態における占有軌道 $i, j$ と非占有軌道 $a, b$ を区別します。
# しかし、ADAPT-VQE で導入する一般化演算子では、軌道の組合せを Hartree-Fock 状態の占有・非占有で制限は行いません。
# このようにすることで、基準状態 (Hartree-Fock 状態) に直接作用させてもゼロとなる演算子も、候補に含めることができます。
# 先ほどの UCCSD の状態は $\vert \Psi_\mathrm{UCCSD} \rangle = e^{\sum_\mu \theta_\mu A_\mu} \vert \Phi_\mathrm{HF} \rangle$ のように書かれます。
# ここで、$\theta_\mu$ は各演算子の重みを決めるパラメータです。
# これを出発点に、ADAPT-VQE では個々のユニタリ変換を一つずつ選び、その積として状態を構成します。
# すなわち
#
# ```{math}
# :label: eq-adapt-ansatz
# \vert \Psi_\mathrm{ADAPT} \rangle
# = e^{\theta_L A_{\mu_L}} e^{\theta_{L-1} A_{\mu_{L-1}}} \cdots e^{\theta_2 A_{\mu_2}} e^{\theta_1 A_{\mu_1}} \vert \Phi_\mathrm{HF} \rangle
# ```
#
# のように、トロッター分解のようにしてアンザッツを構成します。
# ただし通常のトロッター分解 $e^{A + B} = \lim_{r \rightarrow \infty} (e^{A/r} e^{B/r})^r$ とは異なり、同じ係数を用いないことに注意が必要です。
#
# ### 演算子プールの構築と選択
#
# 先ほど示したとおり、基本となるのは単一電子励起・二電子励起と、それらのエルミート共役 (すなわち脱励起) を組合せた演算子です。
# 単一電子励起の場合
#
# ```{math}
# :label: eq-adapt-single
# A_{pq}
# =  a_p^\dagger a_q - a_q^\dagger a_p
# ```
#
# のようになります。
# そして二電子励起については
#
# ```{math}
# :label: eq-adapt-double
# B_{pqrs}
# = a_p^\dagger a_q^\dagger a_r a_s, \qquad
# A_{pqrs}
# = B_{pqrs} - B_{pqrs}^\dagger
# ```
#
# のような形となります。
# ADAPT-VQE では、式{eq}`eq-adapt-single`と式{eq}`eq-adapt-double`の演算子を用い、回路の構成に使える演算子の候補集合 $\mathcal{P} = \{A_1, A_2, \dots, A_\mu, \dots, A_M\}$ をあらかじめ用意します。
# 例として、この集合の中から最初に $A_3$、次に $A_8$、そして最後に $A_2$ が選ばれたとしましょう。
# すると状態は $\vert \Psi (\boldsymbol{\theta}) \rangle = e^{\theta_3 A_2} e^{\theta_2 A_8} e^{\theta_1 A_3} \vert \Phi_\mathrm{HF} \rangle$ のように計算されます。
# 選ばれた演算子は演算子プールからは取り除かれず、後の反復で再び選ばれることもあります。
# 次に作用させる演算子を決定するために、$k$ 個の演算子を追加し、パラメータ最適化を行うことで得られた状態を $\vert \Psi^{(k)} \rangle$ のように書きましょう。
# 候補となる演算子 $A_\mu$ を追加した状態と、そのエネルギーはそれぞれ
#
# ```{math}
# :label: eq-adapt-candidate
# \vert \Psi (\alpha) \rangle
# = e^{\alpha A_\mu} \vert \Psi^{(k)} \rangle, \qquad
# E_\mu (\alpha)
# = \langle \Psi^{(k)} \vert e^{-\alpha A_\mu} H e^{\alpha A_\mu} \vert \Psi^{(k)} \rangle
# ```
#
# のようになります。
# ここから、まだ何も演算子を作用させていない $\alpha = 0$ における勾配を計算しましょう。
# すると
#
# ```{math}
# :label: eq-adapt-gradient
# g_\mu
# = \left. \frac{dE_\mu (\alpha)}{d\alpha} \right\vert_{\alpha = 0}
# = \left. \langle \Psi^{(k)} \vert (-A_\mu e^{-\alpha A_\mu} H e^{\alpha A_\mu} + e^{-\alpha A_\mu} H A_\mu e^{\alpha A_\mu}) \vert \Psi^{(k)} \rangle \right\vert_{\alpha = 0}
# = \langle \Psi^{(k)} \vert (-A_\mu H + H A_\mu) \vert \Psi^{(k)} \rangle
# = \langle \Psi^{(k)} \vert [H, A_\mu] \vert \Psi^{(k)} \rangle
# ```
#
# このようにして計算された勾配 $g_\mu$ から、勾配の絶対値の一番大きな演算子 $A_{\mu^\ast}$ を選択します。
# 正の勾配を持つものでも、それに対応するパラメータ $\theta_{k+1}$ を負にすれば良いため、符号を考慮せずに絶対値のみで判断します。
# ただし、この選択手法で判断しているのは、現在の状態 $\vert \Psi^{(k)} \rangle$ の近傍での改善のしやすさです。
# 演算子追加後に最適化を行ったとき、最も大きくエネルギーを下げる演算子であるかどうかを厳密に探索しているわけではありません。
# 選ばれた演算子を追加し、アンザッツを $\vert \Psi^{(k+1)} (\boldsymbol{\theta}) \rangle = e^{\theta_{k+1} A_{\mu^\ast}} e^{\theta_{k} A_{\mu_k}} e^{\theta_{k-1} A_{\mu_{k-1}}} \cdots e^{\theta_{1} A_{\mu_{1}}} \vert \Phi_\mathrm{HF} \rangle$ とします。
# そして通常の VQE を実行し
#
# ```{math}
# :label: eq-adapt-optimize
# \min_{\theta_1, \theta_2, \dots, \theta_{k+1}} \langle \Psi^{(k+1)} (\boldsymbol{\theta}) \vert H \vert \Psi^{(k+1)} (\boldsymbol{\theta}) \rangle
# ```
#
# を求めます。
# 最適化するのは、これまで追加されてきた演算子に対する全てのパラメータ $\theta_{k+1}, \theta_k, \dots, \theta_1$ です。
# また $k + 1$ ステップ目の VQE 計算を行う際、パラメータの初期値は $\boldsymbol{\theta}_\mathrm{initial} = (\boldsymbol{\theta}^\ast, 0)$ のように、$k$ ステップ目のパラメータ最適化結果 $\boldsymbol{\theta}^\ast$ を用います。
# 新たに $k+1$ ステップ目で追加された演算子 $A_{\mu^\ast}$ に対するパラメータ $\theta_{k+1}$ の初期値は、ゼロとします。
# $e^{0 A_{\mu^\ast}} = I$ であるため、追加された直後の状態は $k$ ステップ目の最適化結果から得られた状態と同じです。
# したがって、$\theta_{k+1}$ の初期値をゼロにすることで、新しいアンザッツに自然に前回の解を含めることができます。

# %% [markdown]
# ## Qamomile による実装
#
# それでは、実際に Qamomile を用いて ADAPT-VQE を実装してみましょう。
#
# ### 計算する分子の設定
#
# ここでは $\mathrm{H}_2$ 分子の原子間距離を angstrom（Å）単位で指定し、エネルギーを hartree（Ha）単位で計算することにします。\
# 軌道を表す基底には、最小基底 STO-3G を使います。STO は Slater-type orbital（Slater 型軌道）の略で、3G は各 STO を 3 つの Gaussian 型関数の線形結合で近似することを表します。
# $\mathrm{H}_2$ では各水素原子の 1s 軌道に対応する基底関数を 1 つずつ用いるため、2 つの空間軌道が得られます。
# それぞれに 2 つのスピン状態があるため、4 つのスピン軌道を Jordan-Wigner 変換で 4 量子ビットに対応付けます。\
# 比較対象の完全配置間相互作用（full configuration interaction、FCI）は、選んだ軌道基底と対称性セクターで許される全ての電子配置を使い、電子ハミルトニアンを対角化する方法です。
# ここでは電子数 2、スピン一重項の基底状態を求めます。
# FCI との一致は STO-3G モデルでの計算を検証するものであり、完全基底での分子エネルギーとの一致を意味するものではありません。

# %%
# 核間距離 R [Å]。上から順番に実行し、必要に応じてこの配列を変更する。
bond_lengths = np.array([0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.00])
basis = "sto-3g"

gradient_threshold = 1e-3  # 規格化された演算子プールに対する閾値
max_adapt_steps = 6  # 各距離で追加できる演算子の最大数
max_vqe_iterations = 120  # 演算子を追加するたびに行う BFGS の最大反復回数

assert bond_lengths.ndim == 1 and len(bond_lengths) >= 2
assert np.all(bond_lengths > 0)
assert np.all(np.diff(bond_lengths) > 0), "距離は昇順かつ重複なしにしてください"
print("計算する距離 [Å]:", bond_lengths.tolist())
print("計算点数:", len(bond_lengths))

# %% [markdown]
# ### 量子ビット数の計算
#
# STO-3G 基底での $\mathrm{H}_2$ は、2 空間軌道・4 スピン軌道・2 電子からなります。
# 同じ基底を用いる限り、距離を変更しても量子ビット数と電子数は変わりません。
# この例の演算子プールとその生成子は軌道の添字から構成され、原子間距離に依存しないため、一度だけ作成して全ての距離で再利用することができます。\
# 設定した原子間距離のうち、最小距離で分子モデルを作成し、必要な量子ビット数を確認しましょう。

# %%
molecule_dir = TemporaryDirectory()


def calculate_h2(distance):
    """Compute the molecular integrals and reference energies at one distance."""
    geometry = [["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, distance]]]
    molecule = MolecularData(
        geometry,
        basis,
        multiplicity=1,
        charge=0,
        description=str(distance),
        data_directory=molecule_dir.name,
    )
    return run_pyscf(molecule, run_scf=True, run_fci=True)


first_R = float(bond_lengths[0])
first_molecule = calculate_h2(first_R)

n_qubits = first_molecule.n_qubits
n_electrons = first_molecule.n_electrons
n_spatial = n_qubits // 2

assert n_qubits == 4 and n_electrons == 2
print(f"量子ビット数: {n_qubits}, 電子数: {n_electrons}, 空間軌道数: {n_spatial}")
print(
    f"R={first_R:.2f} Å: HF={first_molecule.hf_energy:.10f} Ha, "
    f"FCI={first_molecule.fci_energy:.10f} Ha"
)

# %% [markdown]
# ### 一般化された単一電子励起・二電子励起の列挙
#
# Hartree-Fock の占有軌道から非占有軌道への励起だけでなく、一般化された空間軌道の組合せを候補にします。
# そしてここでは $\mathrm{H}_2$ のために、単純化されたフェルミオン演算子のプールを作成しています。

# %%
raw_pool = []

# スピン相補化した一般化単一電子励起。
for p, q in combinations(range(n_spatial), 2):
    T = FermionOperator(((2 * q, 1), (2 * p, 0))) + FermionOperator(
        ((2 * q + 1, 1), (2 * p + 1, 0))
    )
    raw_pool.append((f"単一電子励起({p}->{q})", T))

# 異なるスピンの二電子励起と、そのスピン入替え項。
for p, q, r, s in product(range(n_spatial), repeat=4):
    T = FermionOperator(
        ((2 * r, 1), (2 * s + 1, 1), (2 * q + 1, 0), (2 * p, 0))
    ) + FermionOperator(((2 * r + 1, 1), (2 * s, 1), (2 * q, 0), (2 * p + 1, 0)))
    raw_pool.append((f"二電子励起_ab({p},{q}->{r},{s})", T))

# 同じスピンの二電子励起と、そのスピン入替え項。
for p, q in combinations(range(n_spatial), 2):
    for r, s in combinations(range(n_spatial), 2):
        T = FermionOperator(
            ((2 * r, 1), (2 * s, 1), (2 * q, 0), (2 * p, 0))
        ) + FermionOperator(
            ((2 * r + 1, 1), (2 * s + 1, 1), (2 * q + 1, 0), (2 * p + 1, 0))
        )
        raw_pool.append((f"二電子励起_same({p},{q}->{r},{s})", T))

print("列挙した候補数（重複除去前）:", len(raw_pool))

# %% [markdown]
# ### 反エルミート化・規格化・重複除去の処理
#
# 先ほど作成した励起演算子 $T_\mu$ から、$A_\mu = T_\mu - T_\mu^\dagger$ を作成します。
# そして、係数ノルムを 1 に規格化します。
# ただし、$A_\mu$ と $- A_\mu$ は、同じパラメータの符号を反転させるだけで実現できるため、この重複を取り除きます。

# %%
pool = []
pool_labels = []
seen = set()

for label, T in raw_pool:
    A = normal_ordered(T - hermitian_conjugated(T))
    A.compress()
    norm = np.sqrt(sum(abs(c) ** 2 for c in A.terms.values()))
    if norm < 1e-12:
        continue

    A = A / norm
    first_coeff = sorted(A.terms.items())[0][1]
    if first_coeff.real < 0:
        A = -A

    key = tuple(
        (term, round(float(coeff.real), 10), round(float(coeff.imag), 10))
        for term, coeff in sorted(A.terms.items())
    )
    if key in seen:
        continue
    seen.add(key)
    pool.append(A)
    pool_labels.append(label)

print("重複除去後の演算子プールの大きさ:", len(pool))
for j, label in enumerate(pool_labels):
    print(f"{j:2d}: {label}")
assert len(pool) > 0

# %% [markdown]
# ### オブザーバブルに変換
#
# 指数の肩に乗せる生成子 $G_\mu = i A_\mu$ を、Jordan-Wigner 変換しましょう。
# このようにして変換された生成子から、$e^{-i\theta G_\mu} \ (=e^{\theta A_\mu})$ を準備します。


# %%
def to_qamomile_observable(op, width):
    """Convert a real-coefficient OpenFermion Pauli sum to an observable."""
    pauli_factories = {"X": qmo.X, "Y": qmo.Y, "Z": qmo.Z}
    obs = qmo.Hamiltonian(num_qubits=width)
    for term, coeff in op.terms.items():
        if abs(coeff.imag) > 1e-8:
            raise ValueError(f"The observable has a complex coefficient: {coeff}")
        value = float(coeff.real)
        if abs(value) < 1e-12:
            continue
        if len(term) == 0:
            obs += qmo.Hamiltonian.identity(value, num_qubits=width)
        else:
            pauli_product = qmo.Hamiltonian.identity(num_qubits=width)
            for qubit_index, symbol in term:
                pauli_product *= pauli_factories[symbol](qubit_index)
            obs += value * pauli_product
    return obs


# H2の一般化励起から作る生成子G_mu = i*A_muをJordan-Wigner変換。
generators = [to_qamomile_observable(jordan_wigner(1j * A), n_qubits) for A in pool]

M = len(pool)
assert len(generators) == M and M > 0
print(f"演算子プールのサイズ: {M}")


# %% [markdown]
# ### Qamomile の量子カーネルの作成
#
# Hartree-Fock 状態 $\vert \Phi_\mathrm{HF} \rangle$ を作成し、選択された $G_\mu$ を順に適用、準備した量子状態を返す量子カーネルを作成しましょう。
# `adapt_vqe_expectation` 関数は Aer の厳密な推定器を用いて、指定されたオブザーバブルの期待値を評価します。
# `observable` に $H$ を与えることで、エネルギーが計算できます。
# さらに $[H, A_\mu]$ を与えることで、演算子選定に用いる勾配を計算することができます。


# %%
@qm.qkernel
def adapt_vqe_kernel(
    width: qm.Const[qm.Int],
    occupied: qm.Const[qm.Int],
    depth: qm.Const[qm.Int],
    count: qm.Const[qm.Int],
    operators: qm.Const[qm.Vector[qm.Observable, count]],  # noqa: F821
    thetas: qm.Param[qm.Vector[qm.Real, count]],  # noqa: F821
) -> qm.Q[qm.Bits[width]]:  # noqa: F821
    q = qm.qubits(width)
    for i in range(occupied):
        q = qm.x(q, target=i)  # Hartree-Fock 状態の占有
    for i in range(depth):
        q = qm.pauli_evolve(q, operators[i], thetas[i])
    return q


def adapt_vqe_expectation(program, thetas, observable):
    circuit = to_circuit(
        program, target=target, parameters={"thetas": tuple(float(x) for x in thetas)}
    )
    width = circuit.num_qubits
    if observable.num_qubits > width:
        raise ValueError(
            f"Observable needs {observable.num_qubits} qubits, circuit has {width}"
        )
    rows = [("I" * width, observable.constant)]
    for word, coefficient in observable.terms:
        label = ["I"] * width
        for factor in word.factors:
            label[width - factor.index - 1] = factor.pauli.value
        rows.append(("".join(label), coefficient))
    operator = SparsePauliOp.from_list(rows)
    return float(executor.run([(circuit, operator)]).result()[0].data.evs)


# %% [markdown]
# ## 実行と結果
#
# いよいよ、各距離で ADAPT-VQE を実行していきましょう。
# 各距離 `R` ごとに、$\mathrm{H}_2$ のハミルトニアンを生成し、Hartree-Fock 状態を計算します。
# 後の比較のために、FCI によるエネルギー計算も行います。
# 次に、演算子プール $\{A_\mu\}$ は共通のまま、$[H(R), A_\mu]$ を距離に応じて作ります。
# 各ステップで $g_\mu = \langle [H(R), A_\mu]\rangle$ を求め、$\vert g_\mu \vert$ が最大となる演算子を追加します。
# そして追加のたびに、全ての変分パラメータを BFGS で最適化し、各距離の最終的なエネルギー・勾配ノルム・演算子数・選ばれた演算子を保存します。
#
# :::{note}
# 本実装では Qamomile のパウリ時間発展を基本ゲートへ展開しています。
# `pauli_evolve` は可換なパウリ項を要求します。
# $\mathrm{LiH}$ や $\mathrm{H}_4$ のような分子で、生成子に非可換なパウリ項が含まれる場合には、明示的な積公式が必要となります。
# そのような場合、分解による近似誤差が生じる可能性があることに注意しましょう。
# :::

# %% [markdown]
# ### 1. ハミルトニアンと勾配のオブザーバブルを作る
#
# 式{eq}`eq-adapt-gradient`に従い、分子ハミルトニアン$H$と各交換子 $[H, A_\mu]$ を作ります。
# 生成子と同じ変換関数を使うことで、恒等項やパウリ積を共通の手順で処理できます。


# %%
def molecular_observables(molecule):
    """Build the energy and commutator observables for one molecular model."""
    assert molecule.n_qubits == n_qubits and molecule.n_electrons == n_electrons
    fermionic_H = get_fermion_operator(molecule.get_molecular_hamiltonian())
    H = to_qamomile_observable(jordan_wigner(fermionic_H), n_qubits)
    gradient_observables = [
        to_qamomile_observable(
            jordan_wigner(normal_ordered(commutator(fermionic_H, A))), n_qubits
        )
        for A in pool
    ]
    assert len(gradient_observables) == M
    return H, gradient_observables


# %% [markdown]
# ### 2. 1つの距離でアンザッツを成長させる
#
# 式{eq}`eq-adapt-ansatz`の状態を準備し、式{eq}`eq-adapt-gradient`の勾配から追加する演算子を選びます。
# その後、式{eq}`eq-adapt-optimize`に従って全パラメータを BFGS で再最適化します。
# 関数を呼ぶたびに Hartree-Fock 状態から始め、勾配ノルムが閾値を下回るか、演算子数の上限に達するまで繰り返します。

# %%
target = Qiskit(use_native_pauli_evolution=False)
executor = EstimatorV2()


def run_adapt_vqe(R, molecule):
    """Optimize one distance and return its results, labels, and energy trace."""
    hf_energy = float(molecule.hf_energy)
    fci_energy = float(molecule.fci_energy)
    H, gradient_observables = molecular_observables(molecule)
    selected = []
    theta = np.array([], dtype=float)
    energy_history = [hf_energy]
    gradient_norm_history = []
    converged = False
    vqe_warning_count = 0

    # 最後の演算子追加後にも勾配ノルムを判定するため +1 回まわす。
    for step in range(max_adapt_steps + 1):
        current_operators = (
            [generators[j] for j in selected] if selected else [qmo.Z(0)]
        )
        program = qm.specialize(
            adapt_vqe_kernel,
            width=n_qubits,
            occupied=n_electrons,
            depth=len(selected),
            count=len(current_operators),
            operators=tuple(current_operators),
        )

        # 全プール演算子のエネルギー勾配を評価する
        gradients = np.zeros(M)
        for j in range(M):
            gradients[j] = adapt_vqe_expectation(
                program, theta if selected else [0.0], gradient_observables[j]
            )

        gradient_norm = float(np.linalg.norm(gradients))
        gradient_norm_history.append(gradient_norm)
        if gradient_norm < gradient_threshold:
            converged = True
            print(f"  収束: ||g||={gradient_norm:.3e}, 演算子数={len(selected)}")
            break

        if step == max_adapt_steps:
            print(f"  最大演算子数に到達: ||g||={gradient_norm:.3e}")
            break

        # 最大勾配の演算子を追加し、全 theta を再最適化
        best = int(np.argmax(np.abs(gradients)))
        selected.append(best)  # 元のプールから削除しないので再選択も可能
        theta = np.append(theta, 0.0)

        energy_program = qm.specialize(
            adapt_vqe_kernel,
            width=n_qubits,
            occupied=n_electrons,
            depth=len(selected),
            count=len(selected),
            operators=tuple(generators[j] for j in selected),
        )
        result = minimize(
            lambda x: adapt_vqe_expectation(energy_program, x, H),
            theta,
            method="BFGS",
            options={"maxiter": max_vqe_iterations, "gtol": 1e-6},
        )
        if not np.isfinite(result.fun):
            raise RuntimeError(f"R={R:.3f} Å: VQE のエネルギーが有限値ではありません")

        theta = np.asarray(result.x, dtype=float)
        energy_history.append(float(result.fun))
        if not result.success:
            vqe_warning_count += 1
            print(f"  BFGS 警告 (step={step + 1}): {result.message}")
        print(f"  追加 {step + 1}: {pool_labels[best]} / E={result.fun:.10f} Ha")

    # この距離の結果を保存する
    adapt_energy = float(energy_history[-1])
    absolute_error_kcal_mol = abs(adapt_energy - fci_energy) * 627.509474
    record = {
        "R": R,
        "HF": hf_energy,
        "FCI": fci_energy,
        "ADAPT": adapt_energy,
        "log10_absolute_error": (
            np.log10(absolute_error_kcal_mol)
            if absolute_error_kcal_mol > 0
            else -np.inf
        ),
        "n_operators": len(selected),
        "gradient_norm": gradient_norm_history[-1],
        "gradient_converged": converged,
        "vqe_warnings": vqe_warning_count,
    }
    return record, [pool_labels[j] for j in selected], energy_history.copy()


# %% [markdown]
# ### 3. 全ての距離を順に計算する
#
# 各距離で分子モデルを作り、先ほどの関数を実行してエネルギー・収束状態・選択した演算子を保存します。
# 最初の距離では、すでに計算した分子モデルを再利用します。

# %%
scan_results = []
selected_labels_by_distance = []
energy_traces_by_distance = []

for distance_index, distance in enumerate(bond_lengths):
    R = float(distance)
    print(f"\n[{distance_index + 1}/{len(bond_lengths)}] R = {R:.3f} Å")

    if distance_index == 0:
        molecule = first_molecule
    else:
        molecule = calculate_h2(R)

    record, labels, energy_history = run_adapt_vqe(R, molecule)
    scan_results.append(record)
    selected_labels_by_distance.append(labels)
    energy_traces_by_distance.append(energy_history)
    print(
        f"  最終: HF={record['HF']:.8f}, ADAPT={record['ADAPT']:.8f}, "
        f"FCI={record['FCI']:.8f} Ha, "
        f"log10(absolute error / (1 kcal/mol))={record['log10_absolute_error']:.4g}"
    )

print(f"\n全 {len(scan_results)} 点の計算が終了しました。")

# %% [markdown]
# ### 4. 計算結果を検証する
#
# 指定した全ての距離を計算し、各距離で勾配ノルムが閾値を下回ったことを確認しましょう。
# さらに、ADAPT-VQE と FCI のエネルギーが絶対許容誤差 $10^{-8}$ Ha以内で一致することを検証します。
# この許容誤差は、この理想シミュレータと STO-3G モデルの数値検証に用います。

# %%
R_values = np.array([row["R"] for row in scan_results])
E_hf = np.array([row["HF"] for row in scan_results])
E_adapt = np.array([row["ADAPT"] for row in scan_results])
E_fci = np.array([row["FCI"] for row in scan_results])

assert len(scan_results) == len(bond_lengths), "Some requested distances are missing"
np.testing.assert_array_equal(R_values, bond_lengths)
assert all(row["gradient_converged"] for row in scan_results), "ADAPT did not converge"
assert all(0 <= row["gradient_norm"] < gradient_threshold for row in scan_results)
assert np.all(np.isfinite(E_adapt)) and np.all(np.isfinite(E_fci))
fci_tolerance = 1e-8  # Absolute energy tolerance in Ha for this exact simulation.
np.testing.assert_allclose(E_adapt, E_fci, rtol=0, atol=fci_tolerance, equal_nan=False)

# %% [markdown]
# ### 各距離の数値の確認と結果の描画
#
# 最後に、横軸を原子間距離（Å）、縦軸をエネルギー（Ha）としてグラフを描画しましょう。

# %%
table_header = (
    f"{'R[Å]':>4}  {'HF[Ha]':>12}  {'ADAPT[Ha]':>12}  {'FCI[Ha]':>12}  "
    f"{'log10(absolute error / (1 kcal/mol))':>36}  {'Nop':>3}  収束"
)
print(table_header)
print("-" * len(table_header))
for result in scan_results:
    print(
        f"{result['R']:4.2f}  {result['HF']:12.9f}  {result['ADAPT']:12.9f}  "
        f"{result['FCI']:12.9f}  {result['log10_absolute_error']:36.5f}  "
        f"{result['n_operators']:3d}  {result['gradient_converged']}"
    )

for record, labels in zip(scan_results, selected_labels_by_distance):
    print(
        f"R={record['R']:.2f} Å: ", " -> ".join(labels) if labels else "演算子追加なし"
    )

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(R_values, E_hf, "s--", label="Hartree-Fock")
ax.plot(R_values, E_adapt, "o-", label="ADAPT-VQE")
ax.plot(R_values, E_fci, "^-", label="FCI")
ax.set_xlabel("H-H distance (Angstrom)")
ax.set_ylabel("Total energy (Hartree)")
ax.set_title("H2 potential energy curve (STO-3G)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## まとめ
#
# ここでは、{cite:t}`grimsley2019` で提案された手法を解説し、Qamomile で実装する手順をご紹介しました。
# 以下に、このページの重要情報をまとめます。
#
# * ADAPT-VQE は、エネルギー勾配に基づいて演算子を選択し、アンザッツを逐次的に成長させる量子・古典ハイブリッドアルゴリズムです。
# * 今回の実装では、演算子プールに一般化された単一電子励起・二電子励起のフェルミオン演算子を利用しています。各候補の勾配は、ハミルトニアンとの交換子の期待値から評価できます。
# * Qamomile を用いることで、パラメータ化された状態準備を量子カーネルとして記述し、Aer の厳密な推定器で期待値を計算できます。同じカーネルを、エネルギー評価と演算子選択の両方に利用可能です。
# * ADAPT-VQE を用いて $\mathrm{H}_2$ のエネルギー曲線を計算したところ、FCI と一致する結果を得ることができました。
# * 今回の結果は理想シミュレータによる小規模な検証です。より複雑な分子や実機での実行では、演算子プールの大きさ・勾配測定のコスト・回路の深さ・ノイズの影響などを考慮する必要があるでしょう。

# %% [markdown]
#
