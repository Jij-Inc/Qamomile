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
# tags: [usage]
# ---
#
# # 量子ビット列の操作
#
# 量子プログラムを書いていると、量子ビットごとに異なる操作をしたいことがあります。例えば、6個のうち最後の量子ビットだけを反転する、先頭の3個を別の関数で処理する、別々に準備した2つのまとまりを次の処理のために結合する、といった場合です。Qamomileでは、順序を持った量子ビットのまとまりを**量子ビット列**として扱います。
#
# このガイドでは、一部へのゲート操作、分割、結合の3つを扱います。それぞれ、使いどころを示してから小さなコード例を紹介します。

# %%
# この後の例を手元で模擬実行できるよう、Qiskit対応のQamomileをインストールします。
# # !pip install "qamomile[qiskit]"

# %%
from __future__ import annotations

import qamomile as qm
from qamomile.qiskit import Aer
from qamomile.stdlib import qft

# %% [markdown]
# ## 量子ビット列の一部へのゲート操作
#
# 量子ビット列をひとまとまりのまま扱いながら、操作する位置を選びたいときは、量子ゲートの`target=`引数を使います。位置番号はPythonの配列と同じく0から始まります。
#
# ### 1個の量子ビットを指定する
#
# 最後の量子ビットだけを1の状態にしてみましょう。`qm.qubits(6)`は、0の状態の量子ビットを6個用意します。Xゲートは0と1を反転する操作なので、`target=5`を指定すると最後の位置だけを1にできます。


# %%
@qm.qkernel
def flip_last() -> qm.Bits[6]:
    state = qm.qubits(6)
    state = qm.x(state, target=5)
    return qm.measure(state)


# %% [markdown]
# 最後の量子ビットだけにXゲートが適用されることを確認するため、`Aer`でこの量子プログラムをシミュレーションしてみましょう。

# %%
endpoint = Aer()
last_result = qm.run(flip_last, on=endpoint, shots=1, seed=7).result
print(last_result.values)
assert last_result.values == {(0, 0, 0, 0, 0, 1): 1}

# %% [markdown]
# 列の長さや操作する位置を変えて使いたい場合は、シンボリックな指定にできます。次の例では、長さを`width`、位置を`position`として受け取ります。`qm.Const[qm.Int]`は、量子プログラムの構造を決める整数の引数です。実行前に`qm.specialize`で値を固定します。
#
# ここでは4量子ビットの列の位置1を反転します。位置は`0 <= position < width`を満たす必要があります。


# %%
@qm.qkernel
def flip_at(width: qm.Const[qm.Int], position: qm.Const[qm.Int]) -> qm.Bits[width]:
    state = qm.qubits(width)
    state = qm.x(state, target=position)
    return qm.measure(state)


# %%
selected_program = qm.specialize(flip_at, width=4, position=1)
selected_result = qm.run(selected_program, on=endpoint, shots=1, seed=7).result
print(selected_result.values)
assert selected_result.values == {(0, 1, 0, 0): 1}

# %% [markdown]
# ### 複数の量子ビットを選んで操作する
#
# 複数の位置に同じゲートを適用したいときは、ループを使えます。例えば、`range(0, 6, 2)`は位置0・2・4を順に選ぶので、次の関数は位置番号が偶数の量子ビットだけを反転します。


# %%
@qm.qkernel
def flip_even() -> qm.Bits[6]:
    state = qm.qubits(6)
    for target in range(0, 6, 2):
        state = qm.x(state, target=target)
    return qm.measure(state)


# %%
even_result = qm.run(flip_even, on=endpoint, shots=1, seed=7).result
print(even_result.values)
assert even_result.values == {(1, 0, 1, 0, 1, 0): 1}

# %% [markdown]
# 連続した範囲を操作する場合は、連続した位置番号を使います。ここでは`range(3)`が位置0・1・2を順に選び、先頭の3個を反転します。


# %%
@qm.qkernel
def flip_first_three() -> qm.Bits[6]:
    state = qm.qubits(6)
    for target in range(3):
        state = qm.x(state, target=target)
    return qm.measure(state)


# %%
first_three_result = qm.run(flip_first_three, on=endpoint, shots=1, seed=7).result
print(first_three_result.values)
assert first_three_result.values == {(1, 1, 1, 0, 0, 0): 1}

# %% [markdown]
# 列の長さと操作する範囲も、引数で指定できます。`start`は開始位置、`stop`は終了位置（この位置は含みません）、`step`は位置番号の間隔です。ここでは正の`step`を使い、`0 <= start < stop <= width`となる範囲を指定します。


# %%
@qm.qkernel
def flip_range(
    width: qm.Const[qm.Int],
    start: qm.Const[qm.Int],
    stop: qm.Const[qm.Int],
    step: qm.Const[qm.Int],
) -> qm.Bits[width]:
    state = qm.qubits(width)
    for target in range(start, stop, step):
        state = qm.x(state, target=target)
    return qm.measure(state)


# %% [markdown]
# `step=2`にすると、1個おきに操作できます。8量子ビットの列で位置1・3・5・7を反転してみましょう。

# %%
spaced_program = qm.specialize(flip_range, width=8, start=1, stop=8, step=2)
spaced_result = qm.run(spaced_program, on=endpoint, shots=1, seed=7).result
print(spaced_result.values)
assert spaced_result.values == {(0, 1, 0, 1, 0, 1, 0, 1): 1}

# %% [markdown]
# 同じ関数でも、`step=1`にすると連続した範囲を操作できます。今度は5量子ビットの列の先頭2個を反転します。

# %%
contiguous_program = qm.specialize(flip_range, width=5, start=0, stop=2, step=1)
contiguous_result = qm.run(contiguous_program, on=endpoint, shots=1, seed=7).result
print(contiguous_result.values)
assert contiguous_result.values == {(1, 1, 0, 0, 0): 1}

# %% [markdown]
# 列全体に同じ操作を適用する場合は、ループを書く必要はありません。`qm.x`のような単一量子ビットゲートでは、`target=`を省略すると列の全量子ビットに個別にゲートを適用します。例えば、`qm.x(state)`は列のすべての量子ビットを反転します。
#
# この書き方も、長さを引数にした関数で使えます。次の例では4個すべてを反転します。


# %%
@qm.qkernel
def flip_every_qubit(width: qm.Const[qm.Int]) -> qm.Bits[width]:
    state = qm.qubits(width)
    state = qm.x(state)
    return qm.measure(state)


# %%
all_program = qm.specialize(flip_every_qubit, width=4)
all_result = qm.run(all_program, on=endpoint, shots=1, seed=7).result
print(all_result.values)
assert all_result.values == {(1, 1, 1, 1): 1}

# %% [markdown]
# ## 量子ビット列の分割
#
# 部分ごとに別の測定結果を得たいときや、一部だけを別の関数に渡したいときは、量子ビット列を分割します。`qm.split_at`を使うと、分けた部分をそれぞれ別の値として扱えます。
#
# ### 指定した位置で2つに分ける
#
# 先頭の3個と後ろの3個を別々に扱うには、6量子ビットの列を位置3で分割します。`qm.split_at(state, 3)`の最初の戻り値には位置0・1・2が、2番目には位置3・4・5が入ります。
#
# 分割後は、それぞれの列の位置番号が0から始まります。次の例では、分割前に元の位置2を反転し、分割後に`back`の位置0を反転します。この位置は、元の列では位置3に当たります。最後に、両方を測定して返します。


# %%
@qm.qkernel
def split_halves() -> tuple[qm.Bits[3], qm.Bits[3]]:
    state = qm.qubits(6)
    state = qm.x(state, target=2)
    front, back = qm.split_at(state, 3)
    back = qm.x(back, target=0)
    return qm.measure(front), qm.measure(back)


# %% [markdown]
# `result.values`には、戻り値を2つに分けた形が保たれます。最初のタプルが`front`の測定結果、2番目が`back`の測定結果です。

# %%
halves_result = qm.run(split_halves, on=endpoint, shots=1, seed=7).result
print(halves_result.values)
assert halves_result.values == {((0, 0, 1), (1, 0, 0)): 1}

# %% [markdown]
# :::{note}
# 分割した後は、返された量子ビット列、この例では`front`と`back`を使って処理を続けます。分割前の`state`は再利用できません。
#
# 量子ビット列は線形な値です。`qm.split_at`も入力の`state`を消費し、代わりに2つの列を返します。ゲート操作も同様に、`back = qm.x(back, target=0)`と書くと、操作後の新しい値が`back`に代入されます。この新しい値には、さらに操作を適用できます。分割した列は、それぞれ測定したり、関数から返したり、次の節のように結合したりできます。
# :::
#
# 分割した両方の部分には、少なくとも1個の量子ビットが必要です。`n`個の量子ビットを分ける位置は`0 < boundary < n`を満たす必要があり、0や`n`を指定するとエラーになります。分割は既存の量子ビットのまとめ方を変えるだけで、量子プログラムに新たな量子ゲートを追加しません。

# %% [markdown]
# :::{note}
# 現在、`qm.split_at`は、列の長さや分割位置を`qm.Const`引数で表すシンボリックな指定に対応していません。分割する列の長さと位置には、上の例のように具体的な整数を使ってください。
# :::

# %% [markdown]
# ### 途中の範囲を取り出す
#
# 3量子ビットを処理する関数に、大きな列の位置1・2・3だけを渡したい場合を考えます。まず位置0を分け、残りの列を位置3で分割すると、目的の範囲と末尾の2個に分けられます。
#
# 次の補助関数`flip_three`は、`qm.Q[qm.Bits[3]]`を受け取って返します。引数と戻り値に書かれたこの型は、3量子ビットの列を表します。この関数は3個すべてにXゲートを適用します。切り出した`window`を渡せば、目的の部分だけにこの処理を適用できます。


# %%
@qm.qkernel
def flip_three(state: qm.Q[qm.Bits[3]]) -> qm.Q[qm.Bits[3]]:
    return qm.x(state)


@qm.qkernel
def split_window() -> tuple[qm.Bits[1], qm.Bits[3], qm.Bits[2]]:
    state = qm.qubits(6)
    prefix, remainder = qm.split_at(state, 1)
    window, suffix = qm.split_at(remainder, 3)
    window = flip_three(window)
    return qm.measure(prefix), qm.measure(window), qm.measure(suffix)


# %% [markdown]
# 3つの結果は、それぞれ元の位置0、位置1〜3、位置4〜5に対応します。変わるのは中央の部分だけです。2回目の分割位置は`remainder`の中で数えるため、指定した3は、その部分の先頭から数えた位置になります。

# %%
window_result = qm.run(split_window, on=endpoint, shots=1, seed=7).result
print(window_result.values)
assert window_result.values == {((0,), (1, 1, 1), (0, 0)): 1}

# %% [markdown]
# ## 量子ビット列の結合
#
# 全体をまとめて測定したいときや、1つの列を受け取る関数に渡したいときは、量子ビット列を結合します。`qm.concat`は2つ以上の列を受け取り、渡した順に結合します。
#
# :::{note}
# 現在、`qm.concat`は、入力する列の長さを`qm.Const`引数で表すシンボリックな指定に対応していません。この節では、長さを具体的な整数で指定した列を使います。
# :::
#
# ### 分割して処理した部分をまとめる
#
# 一部を処理するために分割した列は、処理後に再び結合できます。次の例では、前半を前の節の補助関数`flip_three`に渡し、操作していない後半と結合します。


# %%
@qm.qkernel
def rejoin_halves() -> qm.Bits[6]:
    state = qm.qubits(6)
    front, back = qm.split_at(state, 3)
    front = flip_three(front)
    state = qm.concat(front, back)
    return qm.measure(state)


# %% [markdown]
# `qm.concat(front, back)`では、`front`が前に、`back`が後ろに並びます。結合した列を測定すると、3ビットずつのタプルが2つではなく、6ビットのタプルが1つ得られます。

# %%
rejoined_result = qm.run(rejoin_halves, on=endpoint, shots=1, seed=7).result
print(rejoined_result.values)
assert rejoined_result.values == {(1, 1, 1, 0, 0, 0): 1}

# %% [markdown]
# 同じ手順は、Qamomileが用意している関数にも使えます。例えば、`qft`は指定した個数の量子ビット列に量子フーリエ変換（QFT）を適用します。ここでは3量子ビットの範囲を取り出して`qft`に渡し、その前後の部分と元の順に結合します。


# %%
@qm.qkernel
def qft_on_window() -> qm.Bits[6]:
    state = qm.qubits(6)
    prefix, remainder = qm.split_at(state, 1)
    window, suffix = qm.split_at(remainder, 3)
    window = qft(window, 3)
    state = qm.concat(prefix, window, suffix)
    return qm.measure(state)


# %% [markdown]
# この例では、QFTによって中央の3個が重ね合わせの状態になるため、測定結果は実行ごとに変わる可能性があります。32回測定して、周囲の量子ビットが0のままであることを確認します。変わる可能性があるのは、元の位置1・2・3だけです。

# %%
qft_result = qm.run(qft_on_window, on=endpoint, shots=32, seed=7).result
print(qft_result.values)
assert sum(qft_result.values.values()) == 32
assert all(
    len(bits) == 6 and bits[0] == bits[4] == bits[5] == 0 for bits in qft_result.values
)

# %% [markdown]
# :::{note}
# 結合すると、入力した量子ビット列はすべて消費されます。`state = qm.concat(front, back)`の後では、次の操作に`state`を使います。それ以前の`front`と`back`は再利用できず、もう一度`qm.concat`に渡すこともできません。なお、結合は量子ゲートを追加せず、量子ビットを複製することもありません。
# :::

# %% [markdown]
# ### 別々に用意した量子ビット列をまとめる
#
# `qm.concat`には、別々に準備した量子ビット列を渡すこともできます。計算の異なる部分でそれぞれ量子ビットを用意し、後の関数には1つの列として渡したい場合に使えます。
#
# ここでは、値が`(1, 0)`の2量子ビットの列と、値が`(1,)`の1量子ビットの列を用意して結合します。


# %%
@qm.qkernel
def combine_prepared() -> qm.Bits[3]:
    first = qm.qubits(2)
    first = qm.x(first, target=0)
    second = qm.qubits(1)
    second = qm.x(second, target=0)
    state = qm.concat(first, second)
    return qm.measure(state)


# %%
prepared_result = qm.run(combine_prepared, on=endpoint, shots=1, seed=7).result
print(prepared_result.values)
assert prepared_result.values == {(1, 0, 1): 1}

# %% [markdown]
# `qm.concat(first, second)`では、`first`の2個が位置0・1に、`second`の1個が位置2に並びます。代わりに`qm.concat(second, first)`と書くと結果は`(1, 1, 0)`になります。このように、引数の順番で結合後の量子ビットの並びが決まります。

# %% [markdown]
# `qm.Const`と`qm.specialize`の詳しい使い方は、[パラメータと特殊化](../tutorial/02_parameterized_kernels.ipynb)を参照してください。量子関数を組み合わせるほかの方法は、[量子関数の再利用](../tutorial/07_reuse_patterns.ipynb)で説明しています。
