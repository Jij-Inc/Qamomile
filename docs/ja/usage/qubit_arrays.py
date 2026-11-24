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
# tags: [usage]
# ---
#
# # 量子ビット列の操作
#
# Qamomileでは、順序を持つ量子ビットのまとまりを**量子ビット列**として扱います。特定の量子ビットにゲートを適用したり、列を分割して一部を量子カーネルに渡し、処理後に結合したりできます。これにより、同じ量子カーネルを大きな量子ビット列のさまざまな部分に適用して再利用できます。
#
# このガイドでは、インデックスを指定した量子ビットの選択、量子ビット列の分割・再結合、論理的な並び順の変更を扱います。また、インデックスが連続していない量子ビットに量子カーネルを適用する方法も紹介します。ここでいう連続性は、量子ビット列の論理インデックスについてのもので、量子ビットの物理的な隣接を意味しません。
#
# 分割と結合は、既存の量子ビットをプログラム上でどうまとめ、どの順番で扱うかを変える操作です。量子ゲートは追加されず、物理的な量子状態や、部分間のエンタングルメントも変わりません。

# %% [markdown]
# ## セットアップ
#
# このガイドの例は、Qamomileに組み込まれたReferenceシミュレーターを使って手元の環境で実行します。Qamomileをインストールしてください。

# %%
# # !pip install qamomile

# %%
from __future__ import annotations

import qamomile as qm
from qamomile.stdlib import on_qubits

endpoint = qm.Reference()

# %% [markdown]
# ## 選んだ量子ビットにゲートを適用する
#
# `qm.x(...)`のような単一量子ビットゲートでは、`target=`引数にインデックスを指定して、操作する量子ビットを選びます。インデックスは0から始まるため、6量子ビットの列では0から5までになります。
#
# ### 1個の量子ビットを選ぶ
#
# `qm.qubits(6)`は、6個の量子ビットをそれぞれ状態$|0\rangle$に準備します。Xゲートは$|0\rangle$と$|1\rangle$を入れ替えるため、`target=5`を指定すると、最後の量子ビットが$|1\rangle$になります。


# %%
@qm.qkernel
def flip_last() -> qm.Shot[qm.Bits[6]]:
    state = qm.qubits(6)
    state = qm.x(state, target=5)
    return qm.measure(state)


last_result = qm.run(flip_last, on=endpoint, shots=1, seed=7).result
print(last_result.values)
assert last_result.values == {(0, 0, 0, 0, 0, 1): 1}

# %% [markdown]
# `last_result.values`は、測定結果を表すビットのタプルをキー、その出現回数を値とする読み取り専用の対応表を返します。ビットは量子ビット列のインデックス順に並びます。この例では、1回のショットでインデックス5だけが1になります。
#
# ### 複数の量子ビットを選ぶ
#
# 複数の量子ビットに同じゲートを適用するには、ループを使います。量子カーネルの中ではPythonの`range(...)`を使うことができます。


# %%
@qm.qkernel
def flip_even() -> qm.Shot[qm.Bits[6]]:
    state = qm.qubits(6)
    for index in range(0, 6, 2):
        state = qm.x(state, target=index)
    return qm.measure(state)


even_result = qm.run(flip_even, on=endpoint, shots=1, seed=7).result
print(even_result.values)
assert even_result.values == {(1, 0, 1, 0, 1, 0): 1}

# %% [markdown]
# ### すべての量子ビットにゲートを適用する
#
# `qm.x(...)`のような単一量子ビットゲートは、ブロードキャストに対応しています。`target=`を省略すると、列のすべての量子ビットにゲートを適用します。


# %%
# 戻り値の型に含まれるwidthはQamomileが解決します。
@qm.qkernel
def flip_every_qubit(width: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[width]]:  # noqa: F821
    state = qm.qubits(width)
    state = qm.x(state)
    return qm.measure(state)


# 量子ビット列の長さを4に固定します。
all_program = qm.specialize(flip_every_qubit, width=4)
all_result = qm.run(all_program, on=endpoint, shots=1, seed=7).result
print(all_result.values)
assert all_result.values == {(1, 1, 1, 1): 1}

# %% [markdown]
# これらの例では、量子ビット列を分割せずに操作する位置を選びました。次の節では、列を分割して、その一部を量子カーネルで処理する方法を紹介します。

# %% [markdown]
# ## 量子ビット列の一部に量子カーネルを適用する
#
# 量子ビット列を2つに分割するには、`qm.split_at(...)`を使います。また、複数の量子ビット列を結合して大きな1つの量子ビット列にするには、`qm.concat(...)`を使います。ライブラリで提供される量子カーネルも、同じ方法で量子ビット列の一部に適用できます。
#
# なお、以下の例では量子カーネルを別の量子カーネルから呼び出しています。


# %%
@qm.qkernel
def flip_three(state: qm.Q[qm.Bits[3]]) -> qm.Q[qm.Bits[3]]:
    return qm.x(state)


@qm.qkernel
def rejoin_halves() -> qm.Shot[qm.Bits[6]]:
    state = qm.qubits(6)
    # 6量子ビットの列を先頭の3量子ビットと後ろの3量子ビットに分けます。
    # front: original[0:3], back: original[3:6]
    front, back = qm.split_at(state, 3)

    front = flip_three(front)

    # 渡した順に結合して元のインデックス順に戻します。
    state = qm.concat(front, back)
    return qm.measure(state)


# %%
rejoined_result = qm.run(rejoin_halves, on=endpoint, shots=1, seed=7).result
print(rejoined_result.values)
assert rejoined_result.values == {(1, 1, 1, 0, 0, 0): 1}

# %% [markdown]
# 分割後の各部分では、インデックスがそれぞれ0から始まります。例えば、`back`のインデックス0は、元の列のインデックス3に対応します。
#
# `qm.split_at(...)`に渡した量子ビット列の値は消費されるため、その後の操作には`qm.split_at(...)`の戻り値（ここでは`front`と`back`）を使います。同様に、`qm.concat(...)`は渡された値を消費し、結合した量子ビット列を返します。ここでは、その戻り値を再び`state`という名前で受け取っています。
#
# 分割後の両方の部分に、少なくとも1個の量子ビットが必要です。`n`量子ビットの列では、分割位置は`0 < boundary < n`を満たす必要があります。`qm.concat(...)`には、2つ以上の量子ビット列を渡せます。また、`qm.concat(...)`に渡される量子ビット列は元々別々に`qm.qubits(...)`で用意されたものでも構いません。

# %% [markdown]
# :::{note}
# **現在の制約：** これらの分割・結合操作では、Qamomileが量子カーネルのソースコードを読み取る時点で、列の長さが具体的に決まっている必要があります。`qm.split_at(...)`では分割位置も必要です。現在は、`Const`や`Param`で型注釈を付けた量子カーネル引数を使って、これらの長さや分割位置を指定することには対応していません。上の例のように、具体的な整数を使ってください。
# :::

# %% [markdown]
# ## 分割を繰り返し、順番を変えて結合する
#
# 分割した部分をさらに分割すると、より小さな範囲を選べます。分割位置は、そのとき分割する列の先頭から数えます。
#
# 次の例では、3回の分割で元のインデックス2と3の量子ビットを取り出します。それぞれにHadamard（H）ゲートを適用してから、分割した量子ビット列を元とは異なる順番で結合します。


# %%
@qm.qkernel
def nested_split() -> qm.Shot[qm.Bits[6]]:
    state = qm.qubits(6)

    # 以下のスライス表記は、元の6量子ビットの列を基準としています。
    prefix, remainder = qm.split_at(state, 1)
    # prefix: original[0:1], remainder: original[1:6]

    window, suffix = qm.split_at(remainder, 3)
    # window: original[1:4], suffix: original[4:6]

    first, inner = qm.split_at(window, 1)
    # first: original[1:2], inner: original[2:4]

    inner = qm.h(inner)

    window = qm.concat(first, inner)
    state = qm.concat(suffix, window, prefix)
    # 結合後の順番を元のインデックスで表すと、[4, 5, 1, 2, 3, 0]になります。

    return qm.measure(state)


# %%
nested_result = qm.run(nested_split, on=endpoint, shots=32, seed=7).result
print(nested_result.values)

assert sum(nested_result.values.values()) == 32
# Hゲートを適用した量子ビットは、各測定結果のタプルのインデックス3と4に対応します。
# 他のビットはすべて0です。
assert all(
    len(bits) == 6 and bits[0] == bits[1] == bits[2] == bits[5] == 0
    for bits in nested_result.values
)

# %% [markdown]
# 例えば、`qm.split_at(remainder, 3)`は、`remainder`を先頭3量子ビットの`window`と残りの2量子ビットの`suffix`に分けます。元々`remainder`が元の量子ビット列の中の後ろの5量子ビットであるため、`window`は元の量子ビット列のインデックス1、2、3に対応します。さらに、その`window`を位置1で分割すると、元のインデックス1と、インデックス2、3の部分に分かれます。
#
# このため、`qm.concat(suffix, window, prefix)`で結合した結果の量子ビット列は元の量子ビット列のインデックスで表すと`[4, 5, 1, 2, 3, 0]`となります。その後のゲートのインデックス指定と測定結果は、この新しい順番に従います。例えば、`target=0`は元の量子ビット列のインデックス4を選びます。
#
# この並べ替えで変わるのは、配列のインデックスと量子ビットの対応だけです。このためSWAPゲートはもちろん追加されず、物理的な量子状態やエンタングルメントを変えることはありません。また、この例では最終的に再度`state`として元の量子ビット列と同じ長さの量子ビット列を構成していますが、必ず元の長さに戻す必要はありません。

# %% [markdown]
# ## 連続していない量子ビットに量子カーネルを適用する
#
# `on_qubits(...)`を使うと、分割と結合を自分で記述せずに、選んだ量子ビットに量子カーネルを適用できます。選ぶインデックスは連続していなくても構いません。
#
# ここでは先に定義した`flip_three`を6量子ビットの列のインデックス0、2、5に適用します。`width=6`は列全体の量子ビット数、`indices=(0, 2, 5)`は`flip_three`に渡す3量子ビットのインデックスを指定しています。

# %%
flip_selected = on_qubits(flip_three, width=6, indices=(0, 2, 5))


@qm.qkernel
def nonconsecutive_demo() -> qm.Shot[qm.Bits[6]]:
    state = qm.qubits(6)
    state = flip_selected(state)
    return qm.measure(state)


selected_result = qm.run(nonconsecutive_demo, on=endpoint, shots=1, seed=7).result
print(selected_result.values)
assert selected_result.values == {(1, 0, 1, 0, 0, 1): 1}

# %% [markdown]
# `on_qubits(...)`は新しい量子カーネルを返します。ここでは、その新たな量子カーネルを`flip_selected`で受け取っています。
#
# `flip_three`内のインデックス0、1、2は、それぞれ元の量子ビット列のインデックス0、2、5に対応します。この対応は、`indices`に並べた順番で決まります。
#
# `flip_selected`の戻り値の量子ビット列には、元のインデックス順で6量子ビットすべてが含まれるため、測定結果では位置0、2、5だけが1になります。

# %% [markdown]
# :::{note}
# **現在の制約：** 現在の実装では、`on_qubits(...)`を呼び出す時点で、`width`と`indices`に具体的なPythonの値を渡す必要があります。これらは量子カーネルの生成に使われるため、後から`qm.specialize(...)`や`qm.run(...)`で指定することはできません。
#
# インデックスは列の範囲内で、互いに異なる必要があります。選ぶインデックスの個数は、量子カーネルが受け取る量子ビット数と一致させます。`on_qubits(...)`に渡す量子カーネルはユニタリ演算である必要があり、量子ビット列1つだけを引数に取り、同じ量子ビット数の量子ビット列を返す必要があります。量子カーネル内で測定を行うものは使えません。また、`on_qubits(...)`には、`Const`引数や`Param`引数を取る量子カーネルは渡せません。そのようなカーネルを使う場合は、量子ビット列だけを引数に取り、内部で固定値を渡して元のカーネルを呼び出す量子カーネルで包んでください。
# :::

# %% [markdown]
# ## まとめ
#
# - ゲートの`target=`引数で1量子ビットを選べます。複数のインデックスにゲートを適用するには、ループを使います。
# - `qm.split_at(...)`で、インデックスが連続する部分を取り出し、量子カーネルに渡せます。
# - `qm.concat(...)`で、必要なインデックス順に量子ビット列を結合できます。別々に準備した量子ビット列も結合できます。
# - `on_qubits(...)`で、選んだインデックスに量子カーネルを適用する量子カーネルを生成できます。インデックスは連続していなくても構いません。生成時に、`width`と`indices`に具体的な値を指定します。
#
# 分割と結合は、量子ビットのグループ分けやインデックスの対応を変える操作です。これらの操作では量子ゲートが追加されず、部分間のエンタングルメントを含む物理的な量子状態を保ちます。
