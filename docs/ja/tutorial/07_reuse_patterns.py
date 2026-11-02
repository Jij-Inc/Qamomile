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
# # 量子カーネルの再利用
#
# 複雑な量子回路を作る場合、同じゲート操作を何度も書くとコードが長くなり、修正も困難になります。Qamomileでは、繰り返し使いたい操作を`@qm.qkernel`で関数として定義し、別のQKernelから呼び出すことが可能です。別のカーネルからヘルパーQKernelを呼び出し、4量子ビットのGHZ状態を作ります。ヘルパーQKernelは、別のカーネルから呼び出して処理を再利用するための量子関数です。補足では、カーネルを引数として渡す例も紹介します。

# %%
# 最新のQamomileをpipからインストールします。
# # !pip install qamomile

# %%
import qamomile as qm

# %% [markdown]
# ## 1. ヘルパーQKernelを定義する
#
# 最初に、指定した2量子ビットにCXゲートを適用する処理を`entangle_once`として定義しましょう。`state`は4量子ビットの量子レジスタです。`control`と`target`には、それぞれ量子ビットのインデックスを整数1つで指定します。`control`は制御量子ビットを、`target`は制御量子ビットが1のときにXゲートを適用する量子ビットを指定します。インデックスは0から始まります。


# %%
@qm.qkernel
def entangle_once(
    state: qm.Q[qm.Bits[4]],
    control: qm.Const[qm.Int],
    target: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[4]]:
    return qm.cx(state, control=control, target=target)


# %% [markdown]
# 量子演算の結果は、更新された量子レジスタとして返されます。呼び出し側では`state = entangle_once(...)`のように受け取り、次の操作には更新後の`state`を渡す必要があります。この書き方により、QKernelを通常の関数のように組合せることが可能です。
#
# ## 2. ヘルパーを使ってGHZ状態を作る
#
# 4量子ビットのGHZ状態は、次のように表されます。
#
# $$
# |\mathrm{GHZ}_4\rangle = \frac{|0000\rangle + |1111\rangle}{\sqrt{2}}
# $$
#
# 最初の量子ビットにHゲートを適用し、隣り合う量子ビットへCXゲートを順番に適用することで、GHZ状態を準備できます。下の例では、CXの部分に先ほど定義した`entangle_once()`を用いています。


# %%
@qm.qkernel
def ghz_with_helper() -> qm.Shot[qm.Bits[4]]:
    state = qm.qubits(4)
    state = qm.h(state, target=0)
    for control in range(3):
        state = entangle_once(state, control, control + 1)
    return qm.measure(state)


# %% [markdown]
# `for control in range(3)`によって、`(control, target) = (0, 1), (1, 2), (2, 3)`の順にCXゲートを適用しています。ここでは`0`や`1`のような通常のPythonの整数を、QKernelの引数として渡すことができます。量子レジスタ更新後に測定を行うため、最後に`qm.measure(state)`を呼び出しています。それでは、これを実行してみましょう。

# %%
helper_program = qm.specialize(ghz_with_helper)
helper_result = qm.run(
    helper_program,
    on=qm.Reference(),
    shots=64,
    seed=7,
).result

expected_outcomes = {(0, 0, 0, 0), (1, 1, 1, 1)}

assert helper_result.shots == 64
assert sum(helper_result.counts.values()) == 64
assert set(helper_result.counts) <= expected_outcomes

print("Helper QKernel counts:", helper_result.counts)


# %% [markdown]
# `helper_result.counts`には、観測されたビット列のタプルと、その出現回数が対応付けられています。辞書と同じように、例えば`helper_result.counts.get((0, 0, 0, 0), 0)`で回数を参照できます。ただし、内容を変更することはできません。理想的なGHZ状態を計算基底で測定すると、`(0, 0, 0, 0)`と`(1, 1, 1, 1)`がそれぞれ確率1/2で得られます。実際の出現回数にはばらつきがあるため、64ショットで必ず32回ずつになるわけではありません。`entangle_once(...)`を1度定義するだけで、繰り返し使う処理を簡潔に記述できていることがわかります。
#
# ## 3. 補足：カーネルを引数として渡す
#
# 量子カーネルには、別の量子カーネルを引数として渡すこともできます。`qm.QKernel[A, B]`は、型`A`の値を受け取り、型`B`の値を返すカーネルを表します。次の`apply_twice(...)`は、引数で渡された1量子ビット用のカーネルを2回適用します。同じ入出力の型を持つ別のカーネルを渡せば、このヘルパーを書き換えずに演算を変更できます。


# %%
@qm.qkernel
def apply_twice(
    operation: qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]],
    state: qm.Q[qm.Bits[1]],
) -> qm.Q[qm.Bits[1]]:
    state = operation(state)
    return operation(state)


# %% [markdown]
# このヘルパーを試すため、Xを適用する1量子ビット用のカーネルを定義し、`apply_twice`に渡します。量子ビットは`0`から始まります。Xは`0`を`1`に、`1`を`0`に変えるため、2回適用すると`0`に戻ります。


# %%
@qm.qkernel
def flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)


@qm.qkernel
def flip_twice() -> qm.Shot[qm.Bits[1]]:
    state = qm.qubits(1)
    state = apply_twice(flip, state)
    return qm.measure(state)


# %%
twice_result = qm.run(flip_twice, on=qm.Reference(), shots=16, seed=7).result

assert twice_result.shots == 16
assert twice_result.counts == {(0,): 16}

print("Two applications of X:", twice_result.counts)

# %% [markdown]
# カーネルとして`flip`を渡すと、`apply_twice`を使った測定結果は毎回`(0,)`になります。渡すカーネルの引数と戻り値は、`operation`で宣言した型に一致させる必要があります。
#
# 実装がまだない量子カーネルのリソース推定については、「[リソース推定](04_resource_estimation.ipynb)」を参照してください。
#
# ## 4. 補足：同じカーネルをQiskitで実行する
#
# 第1章で使ったQiskit Aerでも、この章の2つのカーネルを実行できます。この補足には`qamomile[qiskit]`が必要です。`pip install "qamomile[qiskit]"`で追加できます。カーネルの定義はそのままで、実行先を表すエンドポイントにシミュレーターの`Aer()`を指定します。
#

# %%
from qamomile.qiskit import Aer  # noqa: E402 -- Optional SDK import.

qiskit_result = qm.run(helper_program, on=Aer(), shots=64, seed=7).result
qiskit_twice_result = qm.run(flip_twice, on=Aer(), shots=16, seed=7).result

assert qiskit_result.shots == 64
assert sum(qiskit_result.counts.values()) == 64
assert set(qiskit_result.counts) <= expected_outcomes
assert qiskit_twice_result.shots == 16
assert qiskit_twice_result.counts == {(0,): 16}

print("Qiskit GHZ counts:", qiskit_result.counts)
print("Qiskit X twice counts:", qiskit_twice_result.counts)

# %% [markdown]
# カーネルの定義、ヘルパーの呼び出し、`qm.QKernel[...]`による引数の指定は、Qiskitに依存しない機能です。ほかの対応するシミュレーターでも、それぞれのエンドポイントを選ぶことで同じカーネルを実行できます。GHZ状態の測定結果にはばらつきがあるため、シミュレーター間で出現回数が一致するとは限りません。出現回数の完全一致ではなく、得られるビット列と測定回数の合計を確認します。
#
# ## まとめ
#
# - `@qm.qkernel`で定義した処理は、別のQKernelから繰り返し呼び出すことが可能です。
# - 量子レジスタに演算を施した後は、更新された`state`を受け取って次の演算へ渡します。
# - `qm.QKernel[...]`でカーネルを引数として受け取り、同じ入出力の型を持つ別の演算にヘルパーを再利用できます。
#
# **次へ**：[エルミート行列から量子回路へ](08_hermitian_decomposition.ipynb)では、NumPy配列で与えたエルミート行列をパウリ演算子の項に分解し、それらを使って時間発展を表す量子回路を作る方法を学びます。
