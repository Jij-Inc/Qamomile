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
# # `BinaryModel`の使い方
#
# 本チュートリアルでは、Qamomileの最適化機能にある`BinaryModel`の使い方について説明します。Qamomileの`BinaryModel`では、二値変数を用いた制約なし最適化問題を定義することができます。特に、変数のタイプとして`binary`と`spin`の両方をサポートしており、`BinaryModel`のメンバ関数`change_vartype`を使うことで、これらのタイプを相互に変換することもできます。

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import numpy as np

from qamomile.optimization.binary_model.expr import VarType, binary
from qamomile.optimization.binary_model.model import BinaryModel

# %% [markdown]
# ## `BinaryExpr`を使ったナイーブな`BinaryModel`の構築
# まずは`BinaryModel`を作る一番ナイーブな方法である`BinaryExpr`を使った方法を紹介します。この方法を実際にユーザーが取ることはそう多くはないと思われますが，以降で紹介する他の方法をとる場合でも内部的には以下の`BinaryExpr`を用いた方法が実質的に行われているため、ここを簡単にでも理解しておくと、`BinaryModel`の理解が深まると思います。
#
# `BinaryExpr`は定数項を含めた二値変数での表現を提供するクラスです。まずは`BinaryExpr`の基本的な使い方を見ていきます。`BinaryExpr`では変数をインデックスで管理します。例えば、一つのbinary(0 or 1)変数を作りたい時には`binary`関数を使って`binary(index)`の形で`BinaryExpr`を作ります。この時、`index`は0-originである必要はありません。$x_1 + 2 x_3 + 3 x_1 x_3 + 5$のような式を作ってみましょう。
#
# > Note: 以降、`BinaryExpr`をbinary変数(0 or 1)を使うことを前提に説明しますが、spin変数(-1 or 1)を使うこともできます。その場合には`qamomile.optimization.binary_model.expr.spin`関数を使ってspin変数を表す`BinaryExpr`を作ることができます。また、簡単のために2次項までで話を進めますが、`BinaryExpr`は3次以上の項もサポートしています。3次項を使う場合には単純に追加で`BinaryExpr`を作成して積に加えれば良いです。

# %%
x_1 = binary(1)  # x_1 を表す BinaryExpr
x_3 = binary(3)  # x_3 を表す BinaryExpr

naive_expr = x_1 + 2 * x_3 + 3 * x_1 * x_3 + 5
naive_expr

# %% [markdown]
# `BinaryExpr`は
# - `vartype`: 変数のタイプ(spin (-1 or 1) or binary (0 or 1))
# - `constant`: 定数項
# - `coefficients`: 変数の係数
#
# 持っており、上の目的関数では以下の通りです。

# %%
print("vartype:", naive_expr.vartype)
assert naive_expr.vartype == VarType.BINARY
print("constant:", naive_expr.constant)
assert naive_expr.constant == 5.0
print("coefficients:", naive_expr.coefficients)
assert naive_expr.coefficients == {(1,): 1.0, (3,): 2.0, (1, 3): 3.0}


# %% [markdown]
# 上の例から分かる通り、`BinaryExpr.coefficients`には辞書形式で変数の係数が格納されており、キーは変数のインデックスのタプル、値はその変数の係数となっています。例えば、上の例では`x_1`と`x_3`の線形項の係数がそれぞれ1であるため、`(1,)`と`(3,)`がキーとなり、値は`x_1`が1.0、`x_3`が2.0になっています。また、二次項である`3 * x_1 * x_3`の係数は3であるため、キーは`(1, 3)`となり、値は3.0になっています。

# %% [markdown]
# では今作った`BinaryExpr`を使って、`BinaryModel`を作ってみましょう。`BinaryModel`は引数に`BinaryExpr`を取ることができます。作成した`BinaryModel`は目的関数の情報を
# - `vartype`: 変数のタイプ
# - `constant`: 定数項
# - `linear`: 変数の線形項の係数
# - `quad`: 変数の二次項の係数
# - `higher`: 変数の三次以上の項の係数
# - `coefficients`: 変数の係数 (`linear`, `quad`, `higher`をまとめたもの)
# - `num_bits`: 変数の数
#
# として保持しています。

# %%
naive_model = BinaryModel(naive_expr)

print("vartype:", naive_model.vartype)
assert naive_model.vartype == VarType.BINARY
print("constant:", naive_model.constant)
assert naive_model.constant == 5.0
print("linear:", naive_model.linear)
assert naive_model.linear == {0: 1.0, 1: 2.0}
print("quad:", naive_model.quad)
assert naive_model.quad == {(0, 1): 3.0}
print("higher:", naive_model.higher)
assert naive_model.higher == {}
print("coefficients:", naive_model.coefficients)
assert naive_model.coefficients == {(0,): 1.0, (1,): 2.0, (0, 1): 3.0}

# %% [markdown]
# ここで重要なこととして、`coefficients`としては、`BinaryExpr.coefficients`と同様に、キーが変数のインデックスのタプル、値がその変数の係数となっていますが、`BinaryModel`に与えた`BinaryExpr`のキーとは異なっていることに注意してください。`BinaryModel`では初期化をする際にキーを0-originの連続した整数に変換しているためです。例えば、上の例では、
# `BinaryExpr`では
# - `x_1`の線形項のキーは`(1,)`
# - `x_3`の線形項のキーは`(3,)`
# - `x_1 * x_3`の二次項のキーは`(1, 3)`
#
# でしたが、それを元にした`BinaryModel`では
# - `x_1`の線形項のキーは`(0,)`
# - `x_3`の線形項のキーは`(1,)`
# - `x_1 * x_3`の二次項のキーは`(0, 1)`
#
# と変換されています。元の`BinaryExpr`のキーを`BinaryModel`のキーとの対応を確認するために
# - `index_new_to_origin`: `BinaryModel`のキーから元の`BinaryExpr`のキーへの対応
# - `index_origin_to_new`: 元の`BinaryExpr`のキーから`BinaryModel`のキーへの対応
#
# が用意されています。

# %%
print("index_new_to_origin:", naive_model.index_new_to_origin)
for new_index, original_index in naive_model.index_new_to_origin.items():
    print("---")
    print(
        f"naive_model.coefficients[(new_index, )] = {naive_model.coefficients[(new_index,)]}"
    )
    print(
        f"naive_expr.coefficients[(original_index, )] = {naive_expr.coefficients[(original_index,)]}"
    )

# %% [markdown]
# 本章の最初でも述べた通り、`BinaryModel`には変数変換のためのメンバ関数`change_vartype`が用意されており、これを使うことで、binary変数とspin変数を相互に変換することができます。例えば、上で作った`BinaryModel`はbinary変数を使っていますが、これをspin変数を使うモデルに変換してみましょう。`change_vartype`は引数に変換したい変数のタイプ(`qamomile.optimization.binary_model.expr.VarType`)を取ります。例えば、binary変数をspin変数に変換したい場合には、`change_vartype(VarType.SPIN)`のようにします。

# %%
spin_naive_model = naive_model.change_vartype(VarType.SPIN)
print("vartype:", spin_naive_model.vartype)
assert spin_naive_model.vartype == VarType.SPIN
print("constant:", spin_naive_model.constant)
assert spin_naive_model.constant == 29.0 / 4.0
print("linear:", spin_naive_model.linear)
assert spin_naive_model.linear == {0: -5.0 / 4.0, 1: -7.0 / 4.0}
print("quad:", spin_naive_model.quad)
assert spin_naive_model.quad == {(0, 1): 3.0 / 4.0}
print("higher:", spin_naive_model.higher)
assert spin_naive_model.higher == {}
print("coefficients:", spin_naive_model.coefficients)
assert spin_naive_model.coefficients == {
    (0,): -5.0 / 4.0,
    (1,): -7.0 / 4.0,
    (0, 1): 3.0 / 4.0,
}

# %% [markdown]
# spin変数とbinary変数の関係は、spin変数を$s$、binary変数を$x$とすると、$s = 1 - 2x$という関係があります。このため、
#
# $$
# \begin{align*}
# x_1 + 2 x_3 + 3 x_1 x_3 + 5
# &= \left( \frac{1 - s_1}{2} \right) + 2 \left( \frac{1 - s_3}{2} \right) + 3 \left( \frac{1 - s_1}{2} \right) \left( \frac{1 - s_3}{2} \right) + 5 \\
# &= \frac{3}{4} s_1 s_3 - \frac{5}{4} s_1 - \frac{7}{4} s_3 + \frac{29}{4} \\
# &= 0.75 s_1 s_3 - 1.25 s_1 - 1.75 s_3 + 7.25
# \end{align*}
# $$
#
# が得られていることがわかります。`BinaryModel.change_vartype`では添え字は元の`BinaryModel`の添え字がそのまま使われることにも注意してください。

# %% [markdown]
# ## QUBO/HUBO/Isingからの`BinaryModel`の構築
# ここまでで、自身で`BinaryExpr`を定義して、それを元に`BinaryModel`を作る方法を紹介しました。しかし実際には、一般にQamomileの`BinaryExpr`でユーザーが数理モデルを保存していることは考えづらく、`BinaryExpr`を毎度構成するのは手間になるためです。そこで、`BinaryModel`には、QUBO/HUBO/Isingの形式から`BinaryModel`を構築するためのクラスメソッドが用意されています。ここでは、これらについて一つずつ見ていきます。なお，これらのクラスメソッドを使った場合にも内部では`BinaryExpr`を経由して`BinaryModel`が構築されています。簡単のためにここから変数の番号を連番で振って説明しますが、実際には変数の番号は連番である必要はなく、ここまでで見たように`BinaryModel`の初期化の際に変数の番号は0-originの連続した整数に変換されます。元の変数を取得するためには、元の変数との対応のためには`index_new_to_origin`や`index_origin_to_new`を利用してください。

# %% [markdown]
# ### QUBOからの`BinaryModel`の構築 (`from_qubo`)
# `from_qubo`関数は`qubo`引数としてQUBOの係数を表す辞書形式のデータを取ります。キーは変数のインデックスのタプル、値はその変数の係数となっています。また、`constant`引数として定数項を取ります。例として以下のようなQUBO行列を考えてみましょう。
#
# $$
# \begin{bmatrix}
# 1 & 0.5 & 0 \\
# 0 & 2 & 1 \\
# 0 & 0 & 3
# \end{bmatrix}
# $$
#
# このQUBO行列は、以下のような目的関数を表しています。
#
# $$
# 1 x_0 + 2 x_1 + 3 x_2 + 0.5 x_0 x_1 + 1 x_1 x_2
# $$
#


# %%
qubo = {
    (0, 0): 1.0,
    (1, 1): 2.0,
    (2, 2): 3.0,
    (0, 1): 0.5,
    (1, 2): 1.0,
}
constant = 0.0
qubo_model = BinaryModel.from_qubo(qubo, constant)
print("vartype:", qubo_model.vartype)
assert qubo_model.vartype == VarType.BINARY
print("constant:", qubo_model.constant)
assert qubo_model.constant == 0.0
print("linear:", qubo_model.linear)
assert qubo_model.linear == {0: 1.0, 1: 2.0, 2: 3.0}
print("quad:", qubo_model.quad)
assert qubo_model.quad == {(0, 1): 0.5, (1, 2): 1.0}
print("higher:", qubo_model.higher)
assert qubo_model.higher == {}
print("coefficients:", qubo_model.coefficients)
assert qubo_model.coefficients == {
    (0,): 1.0,
    (1,): 2.0,
    (2,): 3.0,
    (0, 1): 0.5,
    (1, 2): 1.0,
}

# %% [markdown]
# ### HUBOからの`BinaryModel`の構築 (`from_hubo`)
# `from_hubo`関数は`hubo`引数としてHUBOの係数を表す辞書形式のデータを取ります。キーは変数のインデックスのタプル、値はその変数の係数となっています。また、`constant`引数として定数項を取ります。例として以下のようなHUBOの係数を考えてみましょう。
#
# $$
# \begin{equation*}
# 1 x_0 + 2 x_1 + 3 x_2 + 0.5 x_0 x_1 + 1 x_1 x_2 + 0.1 x_0 x_1 x_2 \\
# \end{equation*}
# $$
#

# %%
hubo = {
    (0,): 1.0,
    (1,): 2.0,
    (2,): 3.0,
    (0, 1): 0.5,
    (1, 2): 1.0,
    (0, 1, 2): 0.1,
}
constant = 0.0
hubo_model = BinaryModel.from_hubo(hubo, constant)
print("vartype:", hubo_model.vartype)
assert hubo_model.vartype == VarType.BINARY
print("constant:", hubo_model.constant)
assert hubo_model.constant == 0.0
print("linear:", hubo_model.linear)
assert hubo_model.linear == {0: 1.0, 1: 2.0, 2: 3.0}
print("quad:", hubo_model.quad)
assert hubo_model.quad == {(0, 1): 0.5, (1, 2): 1.0}
print("higher:", hubo_model.higher)
assert hubo_model.higher == {(0, 1, 2): 0.1}
print("coefficients:", hubo_model.coefficients)
assert hubo_model.coefficients == hubo

# %% [markdown]
# ### Isingからの`BinaryModel`の構築 (`from_ising`)
# `from_ising`関数は`linear`と`quad`という引数を取ります。`linear`引数は線形項の係数を表す辞書形式のデータで、キーは変数のインデックス、値はその変数の係数となっています。`quad`引数は二次項の係数を表す辞書形式のデータで、キーは変数のインデックスのタプル、値はその変数の係数となっています。
# > Note: 高次のIsing項は現在サポートされていませんが、対応予定です。

# %%
ising_linear = {
    0: -1.0,
    1: 2.0,
    2: -3.0,
}
ising_quad = {
    (0, 1): 0.5,
    (1, 2): -1.0,
}
constant = 0.0
ising_model = BinaryModel.from_ising(ising_linear, ising_quad, constant)
print("vartype:", ising_model.vartype)
assert ising_model.vartype == VarType.SPIN
print("constant:", ising_model.constant)
assert ising_model.constant == 0.0
print("linear:", ising_model.linear)
assert ising_model.linear == {0: -1.0, 1: 2.0, 2: -3.0}
print("quad:", ising_model.quad)
assert ising_model.quad == {(0, 1): 0.5, (1, 2): -1.0}
print("higher:", ising_model.higher)
assert ising_model.higher == {}
print("coefficients:", ising_model.coefficients)
assert ising_model.coefficients == {
    (0,): -1.0,
    (1,): 2.0,
    (2,): -3.0,
    (0, 1): 0.5,
    (1, 2): -1.0,
}

# %% [markdown]
# ## `BinaryModel`の正規化とエネルギー計算
# `BinaryModel`には、モデルを正規化するためのメンバ関数`normalize_by_abs_max`と`normalize_by_rms`が用意されています。また、与えられた変数の値に対してエネルギーを計算するためのメンバ関数`calc_energy`が用意されています。ここではこれらの使い方を見ていきます。

# %% [markdown]
# ### 正規化
# `normalize_by_abs_max`関数は、モデルの係数を絶対値の最大値で割ることで正規化します。最初に使ったモデル$x_1 + 2 x_3 + 3 x_1 x_3 + 5$ (`naive_expr`)を正規化してみましょう。係数の最大値は3であるため、正規化されたモデルは
#
# $$
# \frac{1}{3} x_1 + \frac{2}{3} x_3 + 1 x_1 x_3 + \frac{5}{3}
# $$
#
# となります。

# %%
normalized_model = naive_model.normalize_by_abs_max(replace=False)
print("original vartype:", naive_model.vartype)
print("normalized vartype:", normalized_model.vartype)
assert normalized_model.vartype == naive_model.vartype
print("---")
print("original constant:", naive_model.constant)
print("normalized constant:", normalized_model.constant)
assert normalized_model.constant == naive_model.constant / 3.0
print("---")
print("original linear:", naive_model.linear)
print("normalized linear:", normalized_model.linear)
assert normalized_model.linear == {0: 1.0 / 3.0, 1: 2.0 / 3.0}
print("---")
print("original quad:", naive_model.quad)
print("normalized quad:", normalized_model.quad)
assert normalized_model.quad == {(0, 1): 1.0}

# %% [markdown]
# `normalize_by_rms`関数は、モデルの係数をルート平均二乗で割ることで正規化します。ルート平均二乗は以下のように計算されます。
#
# $$
# W = \sqrt{\frac{1}{\lvert E_2 \rvert} \sum_{i, j} (w_{(i, j)})^2 + \frac{1}{\lvert E_1 \rvert} \sum_i (w_i)^2}
# $$
#
# ここで、$w_{(i, j)}$は二次項の係数、$w_i$は線形項の係数、$E_2$は二次項の数、$E_1$は線形項の数です。最初に使ったモデル$x_1 + 2 x_3 + 3 x_1 x_3 + 5$ (`naive_expr`)をルート平均二乗で正規化してみましょう。この時、
# - $E_1$ = 2
# - $E_2$ = 1
# - $\sum_i (w_i)^2$ = $1^2 + 2^2$ = 5
# - $\sum_{i, j} (w_{(i, j)})^2$ = $3^2$ = 9
# であるため、ルート平均二乗は
#
# $$
# \sqrt{5 / 2 + 9 / 1} = \sqrt{2.5 + 9} = \sqrt{11.5} \approx 3.391
# $$
#
# となります。したがって、
#
# $$
# \frac{1}{3.391} x_1 + \frac{2}{3.391} x_3 + \frac{3}{3.391} x_1 x_3 + \frac{5}{3.391}
# \approx 0.295 x_1 + 0.590 x_3 + 0.884 x_1 x_3 + 1.475
# $$
#
# となります。

# %%
normalized_model_rms = naive_model.normalize_by_rms(replace=False)
print("original vartype:", naive_model.vartype)
print("normalized vartype:", normalized_model_rms.vartype)
assert normalized_model_rms.vartype == naive_model.vartype
print("---")
print("original constant:", naive_model.constant)
print("normalized constant:", normalized_model_rms.constant)
assert normalized_model_rms.constant == naive_model.constant / np.sqrt(11.5)
print("---")
print("original linear:", naive_model.linear)
print("normalized linear:", normalized_model_rms.linear)
assert normalized_model_rms.linear == {0: 1.0 / np.sqrt(11.5), 1: 2.0 / np.sqrt(11.5)}
print("---")
print("original quad:", naive_model.quad)
print("normalized quad:", normalized_model_rms.quad)
assert normalized_model_rms.quad == {(0, 1): 3.0 / np.sqrt(11.5)}

# %% [markdown]
# ### 目的関数(エネルギー)計算
# `calc_energy`関数は、与えられた変数の値に対して目的関数の値(エネルギー)を計算します。例えば、モデル$x_1 + 2 x_3 + 3 x_1 x_3 + 5$ (`naive_expr`)に対して、$x_1 = 1$、$x_3 = 0$の時のエネルギーを計算してみましょう。この時、目的関数の値は
#
# $$
# x_1 + 2 x_3 + 3 x_1 x_3 + 5 = 1 + 2 \cdot 0 + 3 \cdot 1 \cdot 0 + 5 = 6
# $$
#
# となります。
# `calc_energy`関数は引数として**`BinaryModel`の変数順序での`list[int]`**を取ります。例えば、上の例では、`BinaryModel`の変数の順序は`x_1`が0番目、`x_3`が1番目であるため、`[1, 0]`のように渡す必要があります。このときに`BinaryModel.index_new_to_origin`を使うことで、機械的に`BinaryModel`の変数順序での`list[int]`を作ることができます。

# %%
# 元の問題での解を作成する。
example_solution = {
    3: 0,  # x_3 = 0
    1: 1,  # x_1 = 1
}
# BinaryModelの変数順序でのlist[int]を作成する。
solution_in_model_order = [
    example_solution[naive_model.index_new_to_origin[new_index]]
    for new_index in range(naive_model.num_bits)
]
# エネルギーを計算する。
energy = naive_model.calc_energy(solution_in_model_order)
print("solution in model order:", solution_in_model_order)
assert solution_in_model_order == [1, 0]
print("energy:", energy)
assert energy == 6.0

# %% [markdown]
# `calc_energy`関数では対象となる`BinaryModel`の`vartype`に応じて、渡された変数の値がspin変数(-1 or 1)なのかbinary変数(0 or 1)なのかの検証をしてから計算を行います。このため、例えばbinary変数である`naive_model`に対して、spin変数の値を渡すとエラーになります。

# %%
# spin変数の値を作成する。
example_spin_solution = [1 - 2 * solution for solution in solution_in_model_order]

try:
    energy = naive_model.calc_energy(example_spin_solution)
except ValueError as e:
    print("エラーが出るのが正常です。")
    print("Error:", e)


# %% [markdown]
# もちろん、`BinaryModel`の`vartype`に合わせた値を渡せば、エネルギーは計算できます。ここでは`spin_naive_model`に対して、spin変数の値を渡してエネルギーを計算してみましょう。これらは変数の種類は異なりますが、同じ問題を表しているためエネルギーとしては同じ値が得られます。

# %%
energy_spin = spin_naive_model.calc_energy(example_spin_solution)
print("solution in model order (spin):", example_spin_solution)
assert example_spin_solution == [-1, 1]
print("energy (spin):", energy_spin)
assert energy_spin == energy

# %% [markdown]
# ## `OMMX`からの`BinaryModel`の構築
# 最後に、`OMMX`から`BinaryModel`を構築する方法を紹介します。[`OMMX`](https://jij-inc.github.io/ommx/ja/introduction.html)（Open Mathematical prograMming eXchange; オミキス）とは、数理最適化を実務に応用する過程で必要となる、ソフトウェア間や人間同士のデータ交換を実現するためのオープンなデータ形式と、それを操作するためのSDKの総称です。Qamomileの最適化機能ではOMMXのデータ形式をサポートしているため、用意されている量子アルゴリズムを使う場合にあえて自身で変換を行う必要はありませんが、自身でカスタムしたアルゴリズムを使う時に便利であるため、ここでも紹介しておきます。ここでは、今までの例で使ってきたモデルをOMMXの形式で持っているとして、それを`BinaryModel`に変換してみましょう。
#
# まずはOMMXの形式でインスタンスを作成します。ここでは簡単なインスタンスを作るためにOMMXそのものの機能でコンポーネントからインスタンスを定義しますが、これは簡単な説明の準備であり、実際に使用する場合にはすでに用意されているパッケージあるいは[JijModeling](https://jij-inc-jijmodeling-tutorials-ja.readthedocs-hosted.com/ja/latest/introduction.html)と呼ばれるPythonコードを使用して数理モデルを記述するための数理最適化モデラーを用いて、インスタンス作成を行います。

# %%
# OMMXの形式でモデルを定義する。
from ommx.v1 import DecisionVariable, Instance

ommx_x_1 = DecisionVariable.binary(1, name="x_1")
ommx_x_3 = DecisionVariable.binary(3, name="x_3")

instance = Instance.from_components(
    decision_variables=[ommx_x_1, ommx_x_3],
    objective=ommx_x_1 + 2 * ommx_x_3 + 3 * ommx_x_1 * ommx_x_3 + 5,
    constraints=[],
    sense=Instance.MINIMIZE,
)

# %% [markdown]
# OMMXインスタンスは`to_qubo`/`to_hubo`関数が用意されており、これを使うことでOMMXインスタンスからQUBO/HUBOの形式で係数を取り出すことができます。これをそのまま`BinaryModel`の`from_qubo`/`from_hubo`関数に渡すことで、OMMXインスタンスから`BinaryModel`を構築することができます。

# %%
qubo_from_ommx, constant = instance.to_qubo()
model_from_ommx = BinaryModel.from_qubo(qubo=qubo_from_ommx, constant=constant)
print("vartype:", model_from_ommx.vartype)
assert model_from_ommx.vartype == naive_model.vartype
print("constant:", model_from_ommx.constant)
assert model_from_ommx.constant == naive_model.constant
print("linear:", model_from_ommx.linear)
assert model_from_ommx.linear == naive_model.linear
print("quad:", model_from_ommx.quad)
assert model_from_ommx.quad == naive_model.quad
print("higher:", model_from_ommx.higher)
assert model_from_ommx.higher == naive_model.higher
print("coefficients:", model_from_ommx.coefficients)
assert model_from_ommx.coefficients == naive_model.coefficients

# %% [markdown]
# ここまでと同じく、変数の番号は`BinaryModel`の初期化時に0-originの連続した整数になっています。このため、正しい対応を取るためには、`index_new_to_origin`や`index_origin_to_new`を利用してください。

# %%
print("index_new_to_origin:", model_from_ommx.index_new_to_origin)
for original_index1, original_index2 in qubo_from_ommx.keys():
    print("---")
    if original_index1 == original_index2:
        new_index = model_from_ommx.index_origin_to_new[original_index1]
        print(
            f"model_from_ommx.coefficients[(new_index, )] = {model_from_ommx.coefficients[(new_index,)]}"
        )
    else:
        new_index1 = model_from_ommx.index_origin_to_new[original_index1]
        new_index2 = model_from_ommx.index_origin_to_new[original_index2]
        print(
            f"model_from_ommx.coefficients[(new_index1, new_index2)] = {model_from_ommx.coefficients[(new_index1, new_index2)]}"
        )
    print(
        f"qubo_from_ommx[(original_index1, original_index2)] = {qubo_from_ommx[(original_index1, original_index2)]}"
    )


# %% [markdown]
# ## まとめ
# ここまでで、`BinaryModel`の使い方について、`BinaryExpr`を使った方法と、QUBO/HUBO/Isingの形式からの方法の両方を紹介しました。また、`BinaryModel`の正規化とエネルギー計算の方法についても紹介しました。`BinaryModel`は、二値変数を用いた制約なし最適化問題を定義するための柔軟なクラスであり、様々な形式から構築することができます。また、OMMXインスタンスから簡単に`BinaryModel`を構築する方法についても紹介しました。多くの場合は`BinaryExpr`を直接使うことはなく、OMMXインスタンスから変換をしたり、qubo/hubo/ising形式の辞書から`from_qubo`/`from_hubo`/`from_ising`を使って構築することになると思いますが、`BinaryExpr`を用いた構築で紹介した流れを理解しておくと、`BinaryModel`の内部構造や動作をより深く理解することができると思います。

# %% [markdown]
# ## 関連トピック
# - [QAOAでMaxCutを解く: 回路をゼロから構築する](../algorithm/qaoa_maxcut): networkxから作ったランダムグラフからQUBO辞書を作成し、直接`BinaryModel`を定義した上でQAOAを適用する例
# - [QAOAによるグラフ分割](../algorithm/qaoa_graph_partition): OMMXインスタンスからQamomileの`QAOAConverter`を用いてQAOAを適用する例 (直接`BinaryModel`を使うのは正規化の部分だけですが、実際にOMMXインスタンスを使ったend-to-endな例です。)
