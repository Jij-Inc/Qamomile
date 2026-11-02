# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#     jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# tags: [algorithm, encoding, resource-estimation]
# ---
#
# # 量子特異値変換
#
# 量子特異値変換（Quantum Singular Value Transformation; QSVT）{cite:p}`10.1145/3313276.3316366`は、より大きなユニタリに埋め込んだ行列の特異値に対して多項式を作用させる手法です。本記事では、非エルミートな$2\times2$行列のブロックエンコーディングを構築し、`qsvt`で2次のチェビシェフ多項式を適用して、変換後の行列と論理リソース推定を検証します。

# %%
# Install the latest Qamomile through pip!
# # !pip install "qamomile[qiskit,visualization]"

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Operator

import qamomile as qm
from qamomile.linalg import decompose_pauli_matrix
from qamomile.qiskit import Aer, Qiskit, to_circuit
from qamomile.stdlib import QSVTPhases, pauli_block_encoding, qsvt

# %% [markdown]
# ## 背景
#
# ### 量子信号処理（QSP）
#
# 量子信号処理（Quantum Signal Processing; QSP）{cite:p}`10.1103/PhysRevLett.118.010501`は、1量子ビットのユニタリに符号化されたスカラー値$x\in[-1,1]$に対して多項式変換を行います。QSPは、信号ユニタリと位相回転の交互積によって構成されます。代表的な規約の一つでは、信号ユニタリを次のように定義します。
#
# $$
# W(x)=e^{i\arccos(x)X}
# =\begin{pmatrix}
# x & i\sqrt{1-x^2}\\
# i\sqrt{1-x^2} & x
# \end{pmatrix}.
# $$
#
# QSPでは、信号ユニタリの間に制御可能な$Z$軸まわりの位相回転を適用します。
#
# $$
# S(\phi)=e^{i\phi Z}
# =\operatorname{diag}(e^{i\phi},e^{-i\phi}).
# $$
#
# 位相列$\Phi=(\phi_0,\phi_1,\ldots,\phi_d)$によって、次の交互積が定まります。
#
# $$
# U_\Phi(x)
# =S(\phi_0)\prod_{j=1}^{d}\left(W(x)S(\phi_j)\right).
# $$
#
# 左上成分$\langle0|U_\Phi(x)|0\rangle=P_\Phi(x)$は$x$の多項式です。実多項式$p(x)$は、次の条件を満たすとき$\operatorname{Re}[P_\Phi(x)]$として表現できます。
#
# - 次数が$d$以下であること。
# - 偶奇性が$d$と一致すること、すなわち$p(-x)=(-1)^d p(x)$であること。
# - 信号区間上で有界であること、すなわち$x\in[-1,1]$に対して$|p(x)|\leq1$であること。
#
# QSPの設計では、まず目標の関数$p(x)$をこの条件を満たす多項式で近似し、対応する位相列を古典計算で合成します。次数$d$の変換は信号ユニタリを$d$回使うため、多項式の次数が問い合わせ計算量を決めます。
#
# QSVTは、このスカラー多項式変換をブロックエンコーディングされた行列の各特異値部分空間へ拡張します。

# %% [markdown]
# ### ブロックエンコーディング
#
# QSPを行列の特異値部分空間へと拡張する前に、量子計算において、ユニタリでない行列を扱う方法について説明します。ブロックエンコーディングは、与えられた（ユニタリでない）行列をより大きなユニタリ行列の一部に埋め込む手法です。
#
# $A$が$n$量子ビットの系レジスタに作用するとします。$(\alpha,a)$ブロックエンコーディングは、追加の$a$量子ビットの信号レジスタに作用するユニタリ$U$で、次を満たします。
#
# $$
# (\langle 0|^{\otimes a}\otimes I)U
# (|0\rangle^{\otimes a}\otimes I)=\frac{A}{\alpha}.
# $$
#
# ここで、$\alpha$は$\alpha\geq\lVert A\rVert_2$を満たす正規化係数です。埋め込まれるブロックは$A/\alpha$であるため、QSVTは区間$[0,1]$の特異値に作用します。$\alpha$の値によって、回路が実現する多項式と、最終的に$A$へ適用される関数との尺度が定まります。
#
# ブロックエンコーディングは、一般には非ユニタリな行列演算をユニタリ発展の一部として表現します。$U$の前後で信号レジスタを$|0\rangle^{\otimes a}$へ射影すると、埋め込まれた行列ブロックが選択されます。この共通の構成により、行列自体がユニタリまたはエルミートでなくても、QSVTでその特異値を操作できます。
#
# ブロックエンコーディングの代表的な構成法には、次のものがあります。
#
# - **スパースアクセスオラクル**：スパース行列の非ゼロ要素について、その位置と値を重ね合わせのまま取得します。これらのオラクルを組み合わせると、正規化されたスパース行列を射影ブロックに持つユニタリを構成できます。
# - **ユニタリの線形結合（LCU）**：$A=\sum_j c_j U_j$を効率的に実装できるユニタリの重み付き和として表し、補助レジスタの振幅に$c_j$を反映して$U_j$を選択します。このレジスタを射影すると、$A/\sum_j|c_j|$が得られます。
#
# このほか、状態準備手続きや量子読み出し専用メモリ（QROM）に格納されたデータからもブロックエンコーディングを構成できます。

# %% [markdown]
# ## アルゴリズム
#
# $U$を行列$A$の$(\alpha,a)$ブロックエンコーディングとし、$A$の特異値分解を
#
# $$
# A=W\Sigma V^\dagger
# $$
#
# とします。信号部分空間への射影を次のように定義します。
#
# $$
# \Pi=|0\rangle\!\langle0|^{\otimes a}\otimes I.
# $$
#
# この部分空間に関する位相付き反射は次のように表せます。
#
# $$
# R_\Pi(\phi)=\exp\!\left(i\phi(2\Pi-I)\right).
# $$
#
# QSVTでは、$U$と$U^\dagger$をこれらの位相回転と交互に適用します。位相列を
#
# $$
# \Phi=(\phi_0,\phi_1,\ldots,\phi_d)
# $$
#
# とすると、得られる積は各特異値$\sigma_j/\alpha$に対応する2次元不変部分空間で独立に作用します。位相を適切に選ぶことで、射影後の応答を多項式$P(\sigma_j/\alpha)$にできます。
#
# $P$の偶奇性によって、左特異ベクトル空間と右特異ベクトル空間の関係が決まります。上記の特異値分解に対する特異値変換は次のようになります。
#
# $$
# P^{(\mathrm{SV})}(A/\alpha)=
# \begin{cases}
# W P(\Sigma/\alpha)V^\dagger, & P\text{が奇関数},\\
# V P(\Sigma/\alpha)V^\dagger, & P\text{が偶関数}.
# \end{cases}
# $$
#
# したがって、QSVTアルゴリズムは次の手順で構成されます。
#
# 1. 入力行列に対して正規化されたブロックエンコーディングを選びます。
# 2. 目標のスカラー関数を、必要な偶奇性を持つ有界な多項式で近似します。
# 3. その多項式に対応する位相列を古典計算で合成します。
# 4. ブロックエンコーディング、その随伴、射影位相回転を交互に適用します。
# 5. 信号部分空間へ射影し、目的の特異値変換を得ます。
#
# QSP/QSVTでは、文献や実装によって位相の定義や演算を適用する順番が異なる場合があります。したがって、ある規約で求めた位相列を使用するときは、回路も同じ規約に従って構成する必要があります。

# %% [markdown]
# ## Qamomileの実装
#
# このセクションでは、LCUを使って非エルミート行列のブロックエンコーディングを構築し、それにQSVTを適用する量子カーネルをQamomileで作成します。その後、量子カーネルを描画し、数値検証に用いるQiskit回路へ変換します。
#
# ### 問題設定
#
# 次の非エルミート行列を考えます。

# %%
matrix = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)

# %% [markdown]
# 計算基底において、この行列は$|1\rangle$を$|0\rangle$へ写し、$|0\rangle$をゼロベクトルへ写します。ユニタリでもエルミートでもないため、量子ゲートとして直接適用することはできません。そこでQSVTでは、射影したブロックが正規化行列$A/\alpha$となるユニタリを利用します。この小さな行列であれば射影結果を容易に確認でき、より大きな行列と同じブロックエンコーディングおよびQSVTの仕組みを使用できます。

# %% [markdown]
# ### ブロックエンコーディングの構築
#
# ブロックエンコーディングの構成法の一つに、ユニタリの線形結合（LCU）があります。LCUでは、対象の行列を複数のユニタリ行列の重み付き和に分解し、補助レジスタで各ユニタリを選択することで、その和を大きなユニタリ回路の一部に埋め込みます。具体的には、行列を次のように表します。
#
# $$
# A=\sum_j c_j U_j,
# \qquad
# \alpha=\sum_j|c_j|.
# $$
#
# ここで各$U_j$はユニタリです。PREPAREは、次の選択レジスタ状態を準備します。
#
# $$
# \operatorname{PREPARE}|0\rangle
# =\sum_j\sqrt{\frac{|c_j|}{\alpha}}|j\rangle.
# $$
#
# SELECTは$|j\rangle$を条件として$e^{i\arg(c_j)}U_j$を適用します。PREPARE--SELECT--UNPREPAREを順に適用すると、選択レジスタの全量子ビットが0であるブロックとして$A/\alpha$が得られます。
#
# ここでは、`decompose_pauli_matrix`でユニタリ$U_j$に対応するパウリ演算子列と複素係数を求め、`pauli_block_encoding`でLCUブロックエンコーディングを構築します。返される記述子には、正規化係数と信号レジスタおよび系レジスタの幅が記録されます。Qamomileは、この情報を使ってブロックエンコーディングを量子カーネルへ組み込みます。

# %%
decomposition = decompose_pauli_matrix(matrix, atol=0.0)
block_encoding = pauli_block_encoding(decomposition.operator)

print("Pauli terms:", decomposition.num_terms)
print("normalization:", block_encoding.normalization)
print("signal qubits:", block_encoding.signal_width)
print("system qubits:", block_encoding.system_width)

assert decomposition.truncation_error_bound == 0.0
assert decomposition.num_terms == 2
assert np.isclose(block_encoding.normalization, 1.0)
assert block_encoding.signal_width == 1
assert block_encoding.system_width == 1

# %% [markdown]
# この行列のパウリ展開は$A=(X+iY)/2$です。したがって、LCUは2つの項からなり、正規化係数は$\alpha=|1/2|+|i/2|=1$となります。1つの信号量子ビットで2つの項を選択し、1つの系量子ビットが$A$を作用させる状態を保持します。上の検証式では、記述子をQSVTへ渡す前にこれらの構造を確認しています。

# %% [markdown]
# ### 量子カーネルの定義
#
# `BlockEncoding`は、信号レジスタと系レジスタを合わせた状態に作用するコヒーレントな関数を保持します。`qsvt`は、この関数と位相列を通常のQamomileのヘルパーに閉じ込めます。量子状態の入力と出力は同じ所有形状を持ちます。測定は、サンプルを必要とする呼び出し側に置きます。
#
# スカラーの`Wx`規約では、3つのゼロ位相が$T_2(x)=2x^2-1$を実装します。`QSVTPhases.from_qsp_wx`で、`qsvt`が使う射影位相の規約へ変換します。この変換だけで近似誤差が証明されるわけではありません。後述の射影行列の比較で、この位相列が実装する変換を検証します。

# %%
qsp_phases = (0.0, 0.0, 0.0)
qsvt_phases = QSVTPhases.from_qsp_wx(qsp_phases, odd=False)
transform = qsvt(block_encoding, phases=qsvt_phases)
total_width = block_encoding.total_width
signal_width = block_encoding.signal_width


@qm.qkernel
def singular_value_transform() -> qm.Q[qm.Bits[total_width]]:
    """Apply the configured QSVT sequence and retain the complete state."""
    return transform(qm.qubits(total_width))


program = singular_value_transform.specialize()
qm.visualization.draw(program, loops="expanded", inline_depth=1)

# %% [markdown]
# ### QSVT回路の構築
#
# コヒーレントな状態準備を特殊化し、`to_circuit`でQiskit回路を確認します。同じ検証済みプログラムを、回路描画、ユニタリの検証、論理リソースの解析に使います。射影位相回転は信号部分空間に直接作用するため、この実装は独立した射影用の補助量子ビットを追加せず、2量子ビットを使います。プログラムはコヒーレントな状態を保持するので、数値検証の前に削除する最終測定はありません。実行するためのコンパイルには古典的な結果が必要です。後述の測定を含む呼び出し側で、この境界を満たします。

# %%
target = Qiskit(optimization_level=0, seed_transpiler=7)
qsvt_circuit = to_circuit(program, target=target)
assert qsvt_circuit.num_qubits == total_width == 2

# %% [markdown]
# ## 実行結果
#
# QSVTユニタリ全体から信号レジスタの全量子ビットが0であるブロックを取り出し、$2(A/\alpha)^\dagger(A/\alpha)-I$と比較します。

# %%
def projected_signal_block(
    unitary: np.ndarray,
    signal_width: int,
    system_width: int,
) -> np.ndarray:
    """Select all-zero low-order signal wires in Qiskit's matrix convention.

    Args:
        unitary (np.ndarray): Complete circuit unitary in SDK bit order.
        signal_width (int): Number of leading logical signal wires.
        system_width (int): Number of remaining logical system wires.

    Returns:
        np.ndarray: Matrix block acting on the system register.
    """
    projected_indices = np.arange(1 << system_width) << signal_width
    return unitary[np.ix_(projected_indices, projected_indices)]


qsvt_unitary = np.asarray(Operator(qsvt_circuit).data)
actual_transform = projected_signal_block(
    qsvt_unitary, block_encoding.signal_width, block_encoding.system_width
)
normalized_matrix = matrix / block_encoding.normalization
expected_transform = (
    2.0 * normalized_matrix.conj().T @ normalized_matrix
    - np.eye(1 << block_encoding.system_width)
)
np.testing.assert_allclose(actual_transform, expected_transform, rtol=0.0, atol=1e-12)
print("QSVT projected block:\n", np.round(actual_transform, decimals=8))

# %% [markdown]
# ### 同じ変換を実行する
#
# この例の射影変換は$\operatorname{diag}(-1,1)$です。系を$|+\rangle$に準備して変換を適用し、2回目のHadamardで相対位相を読み出します。すべてのショットで信号はゼロ、系のビットは1になるはずです。現行のライフサイクルでこのサンプラーをコンパイルして実行し、ローカルのAerと同じソースを使う`Reference`を比較します。

# %%
@qm.qkernel
def interference_sample() -> qm.Shot[qm.Bits[total_width]]:
    """Read the transformed system's relative phase through interference."""
    state = qm.h(qm.qubits(total_width), target=signal_width)
    state = transform(state)
    state = qm.h(state, target=signal_width)
    return qm.measure(state)


endpoint = Aer(optimization_level=0, seed_transpiler=7)
sample_program = interference_sample.specialize()
sample_artifact = qm.compile(sample_program, to=endpoint.target)
record = qm.run(sample_artifact, on=endpoint, shots=32, seed=7)
reference = qm.run(sample_program, on=qm.Reference(), shots=32, seed=7)
assert record.result.counts == reference.result.counts == {(0, 1): 32}
print("signal/system samples:", record.result.counts)

# %% [markdown]
# $A$の2つの特異値は$1$と$0$です。この多項式はそれぞれを$1$と$-1$へ写し、射影したブロックで得られた結果と一致します。

# %%
polynomial_x = np.linspace(-1.0, 1.0, 400)
polynomial_y = 2.0 * polynomial_x**2 - 1.0
singular_values = np.linalg.svd(normalized_matrix, compute_uv=False)
transformed_singular_values = 2.0 * singular_values**2 - 1.0

fig, axis = plt.subplots(figsize=(7, 4))
axis.plot(polynomial_x, polynomial_y, label=r"$T_2(x)=2x^2-1$")
axis.scatter(
    singular_values,
    transformed_singular_values,
    color="tab:red",
    zorder=3,
    label=r"singular values of $A/\alpha$",
)
axis.set_xlabel("x")
axis.set_ylabel("$T_2(x)$")
axis.set_title("Degree-two singular-value transformation")
axis.grid(alpha=0.3)
axis.legend()
plt.show()

# %% [markdown]
# ## リソース推定
#
# ### 問い合わせ計算量
#
# GilyénらによるQSVTの回路構成{cite:p}`10.1145/3313276.3316366`（3.2節、図1）では、$d$次の変換に$U$と$U^\dagger$を合計$d$回、射影位相回転を$d+1$回用います。したがって、ブロックエンコーディングに対する問い合わせ計算量は$\Theta(d)$です。必要な次数$d$は、目標関数、近似区間、許容誤差、正規化係数$\alpha$によって決まります。
#
# $U$と$U^\dagger$の費用を同じ$G_U$とみなし、1回の射影位相回転の費用を$G_\Pi(a)$とすると、測定などの後処理を除くゲート数は次のように整理できます。
#
# $$
# G_{\mathrm{QSVT}}\simeq dG_U+(d+1)G_\Pi(a).
# $$
#
# 第1項はブロックエンコーディングの呼び出し、第2項は位相回転の費用です。これは文献の回路構成に基づく費用の内訳であり、実行先での一意なゲート数ではありません。$G_U$はLCUやスパースオラクルなどの実装方法に、$G_\Pi(a)$は多重制御ゲートの分解方法に依存します。
#
# %% [markdown]
# ### この例の論理リソース
#
# `qm.resources.logical`で、検証済みソースプログラムを解析します。各指標は関係、上下限、根拠、未解決の寄与を持ちます。証明された量子ビット数と、ゲート数や深さの部分的な上下限を区別して読みます。

# %%
study = qm.resources.logical(program)
peak = study.metric(qm.resources.LOGICAL_PEAK_QUBITS)
gates = study.metric(qm.resources.LOGICAL_GATES)
depth = study.metric(qm.resources.LOGICAL_GATE_DEPTH)

print("logical peak qubits:", peak)
print("logical gates:", gates)
print("logical depth:", depth)
for reason in gates.unknown_contributors:
    print("unresolved gate contribution:", reason)

assert peak.numeric_enclosure() == (total_width, total_width) == (2, 2)
assert len(qsvt_phases.phases) - 1 == 2
assert gates.numeric_enclosure()[0] > 0

# %% [markdown]
# 同時に保持する量子ビット数は、信号と系の各1量子ビットで正確に2です。次数は2なので、構成は$U$と$U^\dagger$を各1回、射影位相回転を3回使います。
#
# このソースの論理ゲート数と深さには、制御付き呼び出しの未解決の費用が残ります。有限の下限は全体のゲート数を表すものではなく、上に表示した未解決の寄与をゼロとして扱うことはできません。実行先での分解やスケジューリングは、数学的なブロックエンコーディングの問い合わせ回数とは別に検討できます。

# %% [markdown]
# ## まとめ
#
# 本記事では、次の内容を扱いました。
#
# - 非エルミート行列のパウリLCUブロックエンコーディングを構築しました。
# - 行列の射影とAerおよび`Reference`での決定的な実行により、2次の変換$2(A/\alpha)^\dagger(A/\alpha)-I$を検証しました。
# - 同じ検証済みプログラムについて、数学的な問い合わせ回数、証明された論理量子ビット数、ゲート数や深さの部分的な上下限を比較しました。
#
# `QSVTPhases.from_qsp_wx`は位相の規約を変換します。他の実数目標多項式には、`synthesize_qsvt_polynomial`でRustが検証した証明書を作り、`qsvt(encoding, polynomial=...)`へ渡せます。近似区間や後続処理で必要な誤差条件は、アルゴリズムの設計に含まれます。
