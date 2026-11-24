# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
# tags: [usage, optimization]
# ---
#
# # `BinaryModel`の使い方
#
# `BinaryModel`は、バイナリ変数（0または1）やスピン変数（$-1$または$+1$）を使った、制約のない最適化問題の目的関数を多項式で表します。このノートブックでは、`BinaryModel`の使い方を見ていきます。

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import itertools
import math
import os

import qamomile as qm
from qamomile.optimization import BinaryModel, VarType

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# ## 入力に合わせてモデルの作り方を選ぶ
#
# ### HUBO入力
#
# 0または1を取る変数の最適化問題では、目的関数を整理したとき、異なる3個以上の変数を掛け合わせた項があればHUBO、どの項も2個以下の変数しか含まなければQUBOと呼びます。
#
# どちらの場合も、モデルを作るには`BinaryModel.from_hubo`を使います。引数には辞書を渡し、キーには掛け合わせる変数のIDを、値にはその項の係数を指定します。
#
# 例えば、目的関数$f(x_1,x_3)=x_1+2x_3+3x_1x_3+5$はQUBOです。変数を列ベクトル$\mathbf{x}=(x_1,x_3)^{\mathsf T}$に並べると、次の上三角行列で表せます。
#
# $$
# Q=\begin{bmatrix}1&3\\0&2\end{bmatrix},
# \qquad f(\mathbf{x})=\mathbf{x}^{\mathsf T}Q\mathbf{x}+5.
# $$
#
# バイナリ変数では$x_i^2=x_i$なので、対角成分の1と2がそれぞれ`(1,)`と`(3,)`の係数になります。対角線の右上にある3は`(1, 3)`の係数です。

# %%
coefficients = {(1,): 1.0, (3,): 2.0, (1, 3): 3.0}
constant = 5.0
variables = (1, 3)

model = BinaryModel.from_hubo(coefficients, constant=constant, variables=variables)
print("Variable IDs:", model.variables)
print("Polynomial degree:", model.order)
assert model.vartype is VarType.BINARY
assert model.variables == tuple(sorted(variables))
assert model.order == 2

# %% [markdown]
# ここにキー`(1, 3, 7)`を加えると、異なる3変数の積$0.1x_1x_3x_7$を含むHUBOになります。一方、`(7, 7, 2)`のように同じIDを繰り返すと、$x_7^2x_2=x_7x_2$となります。`(2, 7)`の項もあれば両方の係数が足し合わされます。空のタプル`()`に指定した値は定数項に加わります。

# %%
hubo_model = BinaryModel.from_hubo({**coefficients, (1, 3, 7): 0.1}, constant=constant)
print("HUBO degree:", hubo_model.order)
assert hubo_model.order == 3
assert hubo_model.coefficients[(1, 3, 7)] == 0.1

# %% [markdown]
# ### イジングモデル入力
#
# 変数が$-1$または$+1$を取るときは、`BinaryModel.from_ising`を使います。スピン変数では$s_7^2=1$なので、同じキー`(7, 7, 2)`でも$s_7^2s_2=s_2$となり、その係数は`(2,)`の係数に加わります。次の例では$s_2$の係数は$3-1=2$となり、空のタプル`()`に指定した2が定数項に加わります。

# %%
terms = {(7, 7, 2): 3.0, (2,): -1.0, (): 2.0}
ising_example = BinaryModel.from_ising(terms, variables=(2, 7))

print("Ising terms:", dict(ising_example.coefficients))
assert ising_example.coefficients == {(2,): 2.0}
assert ising_example.constant == 2.0
assert ising_example.variables == (2, 7)

# %% [markdown]
# このイジングモデルの目的関数は$2s_2+2$です。`variables=(2, 7)`を指定すると、式から消えた変数のID`7`もモデルに残ります。

# %% [markdown]
# ## 目的関数の評価
#
# HUBO入力の例で作った2変数のQUBO（`model`）に戻ります。$x_1=1$、$x_3=0$を代入すると、$1+2(0)+3(1)(0)+5=6$です。`energy`を使うと、この目的関数の値を計算できます。辞書で値を渡すときは、元の変数IDをキーにします。タプルで渡すときは、`model.variables`に表示される変数IDの昇順（ここでは`(1, 3)`）に値を並べます。

# %%
assignment = {1: 1, 3: 0}
energy = model.energy(assignment)
print("Objective value:", energy)
assert energy == model.energy((1, 0)) == 6.0

# %% [markdown]
# BINARYモデルで使える値は0と1です。例えば、`model.energy((-1, 1))`を実行すると`ValueError`が発生します。スピンモデルでも`energy`を使えますが、各変数には−1または+1を指定します。

# %% [markdown]
# ## バイナリ変数とスピン変数の変換
#
# 最初に作ったモデルはバイナリ変数を使っています。*同じ目的関数*をスピン変数で表すには、`model.to_ising()`を使います。$s_i=1-2x_i$とおくと、ビット0はスピン$+1$に、ビット1はスピン$-1$に対応します。元の多項式に$x_i=(1-s_i)/2$を代入すると、次の式が得られます。
#
# $$
# f=\tfrac{29}{4}-\tfrac{5}{4}s_1-\tfrac{7}{4}s_3+\tfrac{3}{4}s_1s_3.
# $$
#
# `to_binary()`でバイナリ変数に戻せます。4通りの値の組み合わせについて、スピン変数に変換したときとバイナリ変数に戻したときの目的関数値を、元の値と比べてみます。浮動小数点数の係数には変換時に丸め誤差が生じることがあるため、比較では小さな差を許容します。

# %%
spin_model = model.to_ising()
roundtrip_model = spin_model.to_binary()
print("Spin constant:", spin_model.constant)
print("Spin coefficients:", dict(spin_model.coefficients))
assert spin_model.vartype is VarType.SPIN
assert roundtrip_model.vartype is VarType.BINARY
assert spin_model.variables == roundtrip_model.variables == model.variables

assignments = itertools.product((0, 1), repeat=model.num_variables)
if docs_test_mode:
    assignments = itertools.islice(assignments, 1)
for bits in assignments:
    spins = tuple(1 - 2 * bit for bit in bits)
    expected = model.energy(bits)
    assert math.isclose(
        spin_model.energy(spins), expected, rel_tol=1e-12, abs_tol=1e-12
    )
    assert math.isclose(
        roundtrip_model.energy(bits), expected, rel_tol=1e-12, abs_tol=1e-12
    )

# %% [markdown]
# ## 目的関数の正規化
#
# 次の2つの正規化メソッドは、新しいモデルを返します。目的関数が定数だけなら、その値は変わりません。変数を含む場合は、定数項を含む目的関数全体を1つの正の数で割ります。誤差のない計算では、目的関数を最小にする変数の組み合わせは変わりません。浮動小数点数で計算すると、丸めによって異なる値が同じになったり、同じ値に差が生じたりすることがあります。
#
# - `normalize_by_abs_max()`は、定数項を除く係数の絶対値のうち、最も大きい値で割ります。この例では3で割るので、正規化後の式は$f/3$です。
# - `normalize_by_rms()`は、定数項を除く係数を次数ごとに分け、各グループで係数の二乗の平均を求めます。それらの平均を足した値の平方根が、全体を割る数です。この例では$W=\sqrt{(1^2+2^2)/2+3^2}=\sqrt{11.5}$です。
#
# $f/W$の係数を小数第3位までに丸めると、$0.295x_1+0.590x_3+0.885x_1x_3+1.474$です。先ほどの$x_1=1$、$x_3=0$で評価すると、2つの方法で得られる値はそれぞれ$6/3$と$6/W$です。

# %%
abs_max_model = model.normalize_by_abs_max()
rms_model = model.normalize_by_rms()
print("Original:", energy)
print("Absolute maximum scaling:", abs_max_model.energy(assignment))
print("RMS scaling:", rms_model.energy(assignment))
assert math.isclose(abs_max_model.energy(assignment), energy / 3)
assert math.isclose(rms_model.energy(assignment), energy / math.sqrt(11.5))

# %% [markdown]
# ## 目的関数をハミルトニアンで表示
#
# `model.hamiltonian`を使うと、目的関数を量子ビットに作用する演算子として表せます。パウリ演算子$Z_j$はスピンと同じように、ビット0に対して$+1$、ビット1に対して$-1$の値をとります。`model.variables`の順に、量子ビット0は変数ID`1`に、量子ビット1は変数ID`3`に対応します。スピンで表した目的関数の各スピン変数を、対応する$Z$演算子に置き換えると次の式になります。
#
# $$
# H=7.25I-1.25Z_0-1.75Z_1+0.75Z_0Z_1.
# $$
#
# 定数項は、残りの3項とは別に格納されます。ビットの組`(1, 0)`を当てはめると$7.25+1.25-1.75-0.75=6$となり、先ほど求めた値と一致します。

# %%
hamiltonian = model.hamiltonian
print(f"{hamiltonian.constant.real:g} I")
for word, coefficient in hamiltonian.terms:
    operators = " ".join(
        f"{factor.pauli.value}{factor.index}" for factor in word.factors
    )
    print(f"{coefficient.real:+g} {operators}")
assert hamiltonian.num_qubits == model.num_variables
assert hamiltonian.is_hermitian and hamiltonian.is_pairwise_commuting
assert hamiltonian.constant.real == spin_model.constant

# %% [markdown]
# ## 測定結果と変数の対応付け
#
# 4回の測定で`(1, 0)`が3回、`(0, 1)`が1回得られたとします。ビットはすでにQamomileの論理量子ビットの順に並んでいるので、各結果と回数を直接`SampleResult`に渡せます。`decode`は各ビットを元の変数IDに対応づけ、値の組み合わせごとに目的関数を評価し、それぞれの出現回数も保持します。

# %%
result = qm.SampleResult.from_counts(backend="example", counts={(1, 0): 3, (0, 1): 1})
decoded = model.decode(result)

for sample in decoded.samples:
    print(sample.values, "energy =", sample.energy, "count =", sample.occurrences)
print("Mean energy:", decoded.mean_energy)
assert decoded.total_occurrences == 4
assert decoded.mean_energy == 6.25
assert decoded.lowest.values == assignment
assert decoded.lowest.energy == energy

# %% [markdown]
# 4回の測定で得られた目的関数値の平均は$(3\times6+1\times7)/4=6.25$です。`decoded.lowest`は、*測定で得られた組み合わせの中で*目的関数値が最も小さいものを返します。ここでの値は6です。すべての組み合わせでの最小値は`(0, 0)`のときの5ですが、今回の測定では`(0, 0)`は現れませんでした。

# %% [markdown]
# ## モデルを保存・復元
#
# `to_bytes()`は、ファイルに書き込めるバイト列を返します。`from_bytes()`は、そのバイト列が壊れていないことを確かめ、変数IDを含むモデルを復元します。変数の種類（BINARYまたはSPIN）、変数ID、同類項をまとめた後の係数、定数項が同じモデルは、SHA-256で計算した識別値も同じです。この値は`semantic_digest`で確認できます。

# %%
payload = model.to_bytes()
restored = BinaryModel.from_bytes(payload)
print("Model identifier:", restored.semantic_digest)
assert restored == model
assert restored.to_bytes() == payload
assert restored.semantic_digest == model.semantic_digest

# %% [markdown]
# ## まとめ
#
# - バイナリ変数の式の係数からモデルを作るには`from_hubo`を、スピン変数の式なら`from_ising`を使います。変数を変換したり、目的関数を正規化したりする前に、1通りの組み合わせで`energy`を呼び、目的関数値を確かめます。
# - `model.variables`を見ると、各量子ビットや測定結果の各位置に対応する元の変数IDが分かります。`decode`で測定結果を変数の値に戻すと、組み合わせごとの目的関数値と出現回数が得られます。
#
# ### 関連トピック
#
# - [QAOAでMaxCutを解く](../algorithm/qaoa_maxcut)では、グラフから目的関数の多項式を作ります。
# - [QAOAによるグラフ分割](../algorithm/qaoa_graph_partition)では、測定結果が元のOMMX問題の制約を満たすかを調べます。
# - [パウリ相関符号化でMaxCutを解く](../algorithm/pce_maxcut)では、測定で得られた期待値からスピン変数の値を求めます。
