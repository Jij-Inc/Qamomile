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
# tags: [tutorial]
# ---
#
# # Python の制御構文を使う
#
# 量子回路を用いて演算を設計していると、同じゲートを複数の量子ビットに適用することや、条件に応じて演算を切り替えるなどが必要となる場面があります。
# Qamomile の `@qm.qkernel` では、Python と同じ `for` や `if` を用い、このような処理を記述することが可能です。
# この記事では、次の 3 つを中心に見ていきます。
#
# * `for` と `range()` により量子ゲートを繰り返し適用します。
# * `if` と実行時のパラメータにより適用する演算を切り替えます。
# * 量子ビットの測定結果に依存して、次に行う演算を選択します。
#
# 最後に応用として、`break` と `continue` も短くご紹介します。

# %%
# 最新のQamomileをpipからインストールします。
# # !pip install qamomile
# %%
from __future__ import annotations

import qamomile as qm

# %% [markdown]
# ## `for` を用いた量子ゲートの反復
#
# まず、指定した量子ビット数に対し、ゲートを順番に適用するプログラムを書いてみましょう。
# ここでは例として、偶数番目の量子ビットにアダマールゲート $H$ を適用し、次に隣接する番号の量子ビットの組に CX ゲートを適用する回路を作成します。
# そのために、次のような `qm.qkernel` を作成します。


# %%
@qm.qkernel
def hadamard_chain(
    width: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[width]]:  # noqa: F821
    state = qm.qubits(width)
    for target in range(0, width, 2):
        state = qm.h(state, target=target)
    for control in range(width - 1):
        state = qm.cx(state, control=control, target=control + 1)
    return qm.measure(state)


# %% [markdown]
# `for target in range(0, width, 2)` により、0, 2, 4, ... のように偶数番目の量子ビットを選択します。
# 次の `for control in range(width-1)` により、制御量子ビットを 0, 1, ..., width-2 番目の量子ビットまで順次選んでいきます。
# `state = qm.h(state, ...)` のように、操作の結果を同じ変数へ代入し、次のゲートへ渡します。
# これは Qamomile で量子状態を扱うときの基本的な書き方です。
# そして、`width: qm.Const[qm.Int]` の `Const` は、量子ビット数のように回路の構造を決める値を指定するときに用います。
# 次の例では `qm.specialize` を用い、`width=5` としてプログラムを具体化します。


# %%
chain_program = qm.specialize(hadamard_chain, width=5)
chain_result = qm.run(
    chain_program,
    on=qm.Reference(),
    shots=32,
    seed=5,
).result

print("Number of shots:", chain_result.shots)
print("Measurement counts:", chain_result.counts)

assert chain_result.width == 5
assert chain_result.shots == 32


# %% [markdown]
# `width=5` の場合、$H$ ゲートは量子ビット `0, 2, 4` に、CX ゲートは `0→1`、`1→2`、`2→3`、`3→4` に適用されます。
# 測定結果は `counts` に格納され、ビット列ごとの測定回数として表示されます。
# 複数回の測定を行うため、結果の各回数は乱数や実行環境によって変化します。
#
# ## `if` を用いたゲートの切り替え
#
# 次に、`if` を用い、X ゲートを適用するかどうかの切り替えを行いましょう。
# `enabled` は `qm.Param[qm.Bool]` として宣言されているため、実行前に真偽値を指定する必要があります。
# 同じプログラムを使い、実行ごとに異なる値を指定できます。


# %%
@qm.qkernel
def choose(enabled: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    state = qm.qubits(1)
    if enabled:
        state = qm.x(state)
    return qm.measure(state)


# %% [markdown]
# ここでは、`qm.evaluate(program, ...)` を用いることにします。
# Qamomile に組み込まれた Reference シミュレーターで 1 ショットを実行し、その結果を Python の値として直接返します。
# `enabled=False`、`enabled=True` の順に `choice_program` を実行してみましょう。

# %%
choice_program = qm.specialize(choose)

disabled_result = qm.evaluate(choice_program, enabled=False)
enabled_result = qm.evaluate(choice_program, enabled=True)

print("enabled=False:", disabled_result)
print("enabled=True: ", enabled_result)

assert disabled_result == (0,)
assert enabled_result == (1,)


# %% [markdown]
# 初期に準備される量子ビットは $\vert 0 \rangle$ であるため、X ゲートを適用しない場合 (`False`) には $\vert 0 \rangle$ のまま、適用した場合 (`True`) には $\vert 1 \rangle$ となります。
# よって `qm.evaluate` により出力されるビット列の結果は、それぞれ `(0,)` と `(1,)` となります。
#
# ここまでに登場した `Const` と `Param` は、Qamomile では次のように使い分けます。
#
# | 型 | 値を指定するタイミング | この Notebook の例 |
# | --- | --- | --- |
# | `Const` | 量子カーネルを特殊化する時点まで | `width`（量子ビット数） |
# | `Param` | 実行前 | `enabled`（X ゲートを使うか） |
#
#
# ## 測定結果から次の演算を決定する
#
# 量子カーネルの中では、量子ビットの測定結果を `if` の条件として使うこともできます。
# この例では、`qm.prepare(True)`で最初の量子ビットを $\lvert 1\rangle$ に準備するため、その測定結果は `True` になります。
# `qm.prepare(False)`は、操作対象の量子ビットを $\lvert 0 \rangle$ に準備します。
# 最初の測定結果が `True` なら、操作対象の量子ビットに X ゲートを適用します。


# %%
@qm.qkernel
def conditional_flip() -> qm.Shot[qm.Bool]:
    condition = qm.measure(qm.prepare(True))
    target = qm.prepare(False)
    if condition:
        target = qm.x(target)
    return qm.measure(target)


# %%
feedback_program = qm.specialize(conditional_flip)
feedback_result = qm.evaluate(feedback_program)

print("Measurement feedback result:", feedback_result)

assert feedback_result is True

# %% [markdown]
# 最初の測定結果が True になるため、出力も `True` になります。
# 測定を行い、その結果を確認して、条件を満たす場合にゲートを適用します。
# `qm.measure(...)` が返す値は、実行中に得られます。
# ここでは、その古典的な値を `if` の条件に使って演算を選びます。
# 一方、`qm.control(...)` は制御量子ビットを測定せずに制御演算を適用します。
#
# :::{note}
# 測定結果に応じた条件分岐の実行に対応しているかは、バックエンドにより異なります。
# この例では、Reference シミュレーターを使う `qm.evaluate(...)` で動作を確認しています。
# :::
#
# ## 補足：`continue` と `break`
#
# `for` の途中で特定の繰り返し部分を飛ばしたいときは `continue`、繰り返しを途中で終了したいときは `break` を使うことができます。
# 次の例は量子ゲートを使わず、制御構文の動きを簡単に確認するためのものです。
# 通常ならば `for index in range(limit)`　の反復を行いますが、`continue` により `index == 2` となる反復部分を飛ばします。
# そして `index` の総和が 5 以上になったところで、`break` により反復を終了します。


# %%
@qm.qkernel
def sum_until(limit: qm.Const[qm.Int]) -> qm.Int:
    total = 0
    for index in range(limit):
        if index == 2:
            continue
        total += index
        if total >= 5:
            break
    return total


# %% [markdown]
# `qm.specialize` により `limit = 6` に固定し、先ほどの `@qm.qkernel` を実行しましょう。

# %%
summary_program = qm.specialize(sum_until, limit=6)
summary_result = qm.evaluate(summary_program)

print("Sum until threshold:", summary_result)

assert summary_result == 8

# %% [markdown]
# 0 + 1 + 3 + 4 = 8 となったところで `break` 処理が行われるため、結果は 8 になります。
# このように、上限があらかじめ設定されている `for` の中でも条件によって処理を飛ばしたり、早めに終了することができます。
#
# ## まとめ
#
# - `for` と `range()` を用いることで、複数の量子ビットにゲートを繰り返し適用できます。
# - `Const` の入力値は、量子カーネルを特殊化する時点までに指定する必要があります。`Param` の入力値は実行前に指定し、実行ごとに異なる値を指定できます。
# - `if` を用いると、実行時パラメータや測定結果に応じて量子演算を切り替えることができます。
# - さらに `continue` と `break` を使えば、上限が決まった繰り返しの途中で処理を飛ばしたり終了することも可能です。
#
# **次へ**：[再利用パターン](07_reuse_patterns.ipynb)では、よく使う量子回路を別の `QKernel` に切り出し、複数のプログラムから使う方法を学びます。

# %% [markdown]
#
