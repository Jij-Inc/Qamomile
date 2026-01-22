# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: qamomile
#     language: python
#     name: qamomile
# ---

# %% [markdown]
# # 複数量子ビットとエンタングルメント
#
# このチュートリアルでは、量子コンピューティングの最も不思議で強力な概念である
# **エンタングルメント（量子もつれ）**について学びます。
#
# ## このチュートリアルで学ぶこと
# - 複数の量子ビットを扱う方法
# - CNOTゲート（制御NOTゲート）
# - ベル状態：最も基本的なエンタングル状態
# - `qubit_array()` と `Vector[Qubit]` で配列を扱う
# - `qm.range()` でループを回す
# - GHZ状態：多量子ビットのエンタングルメント

# %%
import math
import qamomile.circuit as qm
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## 1. 複数の量子ビットを使う
#
# これまでは1つの量子ビットだけを扱ってきました。
# 複数の量子ビットを使うには、単純に `qm.qubit()` を複数回呼び出します。

# %%
@qm.qkernel
def two_qubits_independent() -> tuple[qm.Bit, qm.Bit]:
    """2つの独立した量子ビット"""
    q0 = qm.qubit(name="q0")
    q1 = qm.qubit(name="q1")

    # それぞれに独立した操作
    q0 = qm.h(q0)  # q0 を重ね合わせ
    q1 = qm.x(q1)  # q1 を反転

    # 別々に測定して返す
    return qm.measure(q0), qm.measure(q1)


# %%
exec_two = transpiler.transpile(two_qubits_independent)
result_two = exec_two.sample(transpiler.executor(), shots=1000).result()

print("=== 2つの独立した量子ビット ===")
for value, count in result_two.results:
    print(f"  結果: {value}, 回数: {count}")

# %% [markdown]
# ### 結果の解釈
#
# 結果は `(bit0, bit1)` のタプルです。
# - `q0` は H ゲートで重ね合わせ → 0 か 1 が約50%ずつ
# - `q1` は X ゲートで反転 → 常に 1
#
# したがって、`(0, 1)` と `(1, 1)` が約50%ずつ出るはずです。

# %% [markdown]
# ## 2. CNOTゲート（制御NOTゲート）
#
# **CNOTゲート**（Controlled-NOT、CXゲートとも呼ばれる）は、2量子ビットゲートの中で最も基本的なものです。
#
# ### 動作
# - **制御ビット（control）**: この量子ビットが `|1⟩` のとき、ターゲットに作用
# - **ターゲットビット（target）**: 制御ビットが `|1⟩` なら反転
#
# ```
# |00⟩ → |00⟩  (制御が0なのでターゲット変化なし)
# |01⟩ → |01⟩  (制御が0なのでターゲット変化なし)
# |10⟩ → |11⟩  (制御が1なのでターゲット反転)
# |11⟩ → |10⟩  (制御が1なのでターゲット反転)
# ```

# %%
@qm.qkernel
def cnot_example() -> tuple[qm.Bit, qm.Bit]:
    """CNOTゲートの基本的な使い方"""
    q0 = qm.qubit(name="control")
    q1 = qm.qubit(name="target")

    # 制御ビットを |1⟩ にする
    q0 = qm.x(q0)

    # CNOTゲートを適用
    # 重要: 両方の量子ビットを返り値として受け取る！
    q0, q1 = qm.cx(q0, q1)

    return qm.measure(q0), qm.measure(q1)


# %%
exec_cnot = transpiler.transpile(cnot_example)
result_cnot = exec_cnot.sample(transpiler.executor(), shots=1000).result()

print("=== CNOTゲートの例（制御ビット=1） ===")
for value, count in result_cnot.results:
    print(f"  結果: {value}, 回数: {count}")

print("\n制御ビットが1なので、ターゲットも反転して (1, 1) になります")

# %% [markdown]
# ### 重要：2量子ビットゲートの戻り値
#
# Qamomileの線形型システムでは、2量子ビットゲートは**両方の量子ビットを返します**。
#
# ```python
# # 正しい書き方
# q0, q1 = qm.cx(q0, q1)  # 両方受け取る
#
# # 間違った書き方
# qm.cx(q0, q1)           # 戻り値を無視するとエラー
# q0 = qm.cx(q0, q1)      # 片方だけ受け取ると q1 が使えなくなる
# ```

# %% [markdown]
# ## 3. ベル状態：エンタングルメントの基本
#
# **エンタングルメント（量子もつれ）**とは、複数の量子ビットが強い相関を持ち、
# 1つを測定すると他の量子ビットの状態も瞬時に決まる現象です。
#
# 最も基本的なエンタングル状態が**ベル状態**です。

# %%
@qm.qkernel
def bell_state() -> tuple[qm.Bit, qm.Bit]:
    """ベル状態 |Φ+⟩ = (|00⟩ + |11⟩)/√2 を作成"""
    q0 = qm.qubit(name="q0")
    q1 = qm.qubit(name="q1")

    # Step 1: 最初の量子ビットを重ね合わせにする
    q0 = qm.h(q0)

    # Step 2: CNOTで「もつれ」を作る
    q0, q1 = qm.cx(q0, q1)

    return qm.measure(q0), qm.measure(q1)


# %%
exec_bell = transpiler.transpile(bell_state)
result_bell = exec_bell.sample(transpiler.executor(), shots=1000).result()

print("=== ベル状態の測定結果 ===")
for value, count in result_bell.results:
    percentage = count / 1000 * 100
    print(f"  結果: {value}, 回数: {count} ({percentage:.1f}%)")

# %% [markdown]
# ### ベル状態の特徴
#
# 結果を見ると、`(0, 0)` と `(1, 1)` だけが出て、`(0, 1)` や `(1, 0)` は出ません。
#
# これがエンタングルメントの証拠です：
# - 2つの量子ビットは**完全に相関**している
# - 一方を測定して 0 が出たら、もう一方も必ず 0
# - 一方を測定して 1 が出たら、もう一方も必ず 1
# - しかし、どちらが出るかは測定するまで分からない（50/50）
#
# この相関は、古典的な確率では説明できない量子力学特有の現象です。

# %% [markdown]
# ### 回路の可視化

# %%
qiskit_bell = transpiler.to_circuit(bell_state)
print("=== ベル状態を作る回路 ===")
print(qiskit_bell.draw(output="text"))

# %% [markdown]
# ## 4. 4つのベル状態
#
# ベル状態は4種類あり、量子情報の基本となる状態です。

# %%
@qm.qkernel
def bell_phi_plus() -> tuple[qm.Bit, qm.Bit]:
    """|Φ+⟩ = (|00⟩ + |11⟩)/√2"""
    q0, q1 = qm.qubit(name="q0"), qm.qubit(name="q1")
    q0 = qm.h(q0)
    q0, q1 = qm.cx(q0, q1)
    return qm.measure(q0), qm.measure(q1)


@qm.qkernel
def bell_phi_minus() -> tuple[qm.Bit, qm.Bit]:
    """|Φ−⟩ = (|00⟩ − |11⟩)/√2"""
    q0, q1 = qm.qubit(name="q0"), qm.qubit(name="q1")
    q0 = qm.h(q0)
    q0, q1 = qm.cx(q0, q1)
    q0 = qm.rz(q0, math.pi)  # 位相反転
    return qm.measure(q0), qm.measure(q1)


@qm.qkernel
def bell_psi_plus() -> tuple[qm.Bit, qm.Bit]:
    """|Ψ+⟩ = (|01⟩ + |10⟩)/√2"""
    q0, q1 = qm.qubit(name="q0"), qm.qubit(name="q1")
    q0 = qm.h(q0)
    q0, q1 = qm.cx(q0, q1)
    q1 = qm.x(q1)  # ターゲットを反転
    return qm.measure(q0), qm.measure(q1)


@qm.qkernel
def bell_psi_minus() -> tuple[qm.Bit, qm.Bit]:
    """|Ψ−⟩ = (|01⟩ − |10⟩)/√2"""
    q0, q1 = qm.qubit(name="q0"), qm.qubit(name="q1")
    q0 = qm.h(q0)
    q0, q1 = qm.cx(q0, q1)
    q1 = qm.x(q1)
    q0 = qm.rz(q0, math.pi)
    return qm.measure(q0), qm.measure(q1)


# %%
bell_states = [
    ("Φ+", bell_phi_plus),
    ("Φ−", bell_phi_minus),
    ("Ψ+", bell_psi_plus),
    ("Ψ−", bell_psi_minus),
]

print("=== 4つのベル状態 ===\n")

for name, circuit in bell_states:
    exec_b = transpiler.transpile(circuit)
    result_b = exec_b.sample(transpiler.executor(), shots=1000).result()

    print(f"|{name}⟩:")
    for value, count in result_b.results:
        print(f"  {value}: {count}")
    print()

# %% [markdown]
# ## 5. 量子ビット配列：`qubit_array()` と `Vector[Qubit]`
#
# 多くの量子ビットを扱う場合、`qubit_array()` で配列として作成すると便利です。

# %%
@qm.qkernel
def array_example() -> qm.Vector[qm.Bit]:
    """量子ビット配列の基本的な使い方"""
    # 3量子ビットの配列を作成
    qubits = qm.qubit_array(3, name="q")

    # インデックスでアクセス
    qubits[0] = qm.h(qubits[0])
    qubits[1] = qm.x(qubits[1])
    # qubits[2] は何もしない（|0⟩のまま）

    # 配列全体を測定
    return qm.measure(qubits)


# %%
exec_arr = transpiler.transpile(array_example)
result_arr = exec_arr.sample(transpiler.executor(), shots=1000).result()

print("=== 量子ビット配列の例 ===")
print("q[0]: H（重ね合わせ）, q[1]: X（反転）, q[2]: なし")
print()
for value, count in result_arr.results:
    print(f"  結果: {value}, 回数: {count}")

# %% [markdown]
# ### 結果の形式
#
# `Vector[Bit]` として返される結果は、タプル形式で表示されます。
# 例: `(1, 1, 0)` は q[0]=1, q[1]=1, q[2]=0 を意味します。

# %% [markdown]
# ## 6. ループ処理：`qm.range()`
#
# 配列の各要素に操作を適用するには、`qm.range()` を使ってループを書きます。
#
# **注意**: 通常の Python の `range()` ではなく `qm.range()` を使います。

# %%
@qm.qkernel
def loop_example(n: int) -> qm.Vector[qm.Bit]:
    """ループですべての量子ビットにHゲートを適用"""
    qubits = qm.qubit_array(n, name="q")

    # すべての量子ビットに H ゲートを適用
    for i in qm.range(n):
        qubits[i] = qm.h(qubits[i])

    return qm.measure(qubits)


# %%
exec_loop = transpiler.transpile(loop_example, bindings={"n": 4})
result_loop = exec_loop.sample(transpiler.executor(), shots=1000).result()

print("=== ループで全量子ビットを重ね合わせ（n=4）===")
print("すべての2^4=16パターンが均等に出現するはず\n")

# 結果をソートして表示
sorted_results = sorted(result_loop.results, key=lambda x: str(x[0]))
for value, count in sorted_results:
    percentage = count / 1000 * 100
    print(f"  {value}: {count} ({percentage:.1f}%)")

# %% [markdown]
# ## 7. GHZ状態：多量子ビットのエンタングルメント
#
# **GHZ状態**（Greenberger–Horne–Zeilinger状態）は、3量子ビット以上のエンタングルメントです。
#
# $$|GHZ\rangle = \frac{|00...0\rangle + |11...1\rangle}{\sqrt{2}}$$
#
# ベル状態の拡張版で、すべての量子ビットが「全部0」か「全部1」のどちらかになります。

# %%
@qm.qkernel
def ghz_state(n: int) -> qm.Vector[qm.Bit]:
    """N量子ビットのGHZ状態を作成"""
    qubits = qm.qubit_array(n, name="q")

    # 最初の量子ビットを重ね合わせにする
    qubits[0] = qm.h(qubits[0])

    # 連鎖的にCNOTを適用してエンタングルメントを広げる
    for i in qm.range(n - 1):
        qubits[i], qubits[i + 1] = qm.cx(qubits[i], qubits[i + 1])

    return qm.measure(qubits)


# %%
# 3量子ビットGHZ状態
exec_ghz3 = transpiler.transpile(ghz_state, bindings={"n": 3})
result_ghz3 = exec_ghz3.sample(transpiler.executor(), shots=1000).result()

print("=== 3量子ビットGHZ状態 ===")
print("|GHZ⟩ = (|000⟩ + |111⟩)/√2\n")
for value, count in result_ghz3.results:
    percentage = count / 1000 * 100
    print(f"  {value}: {count} ({percentage:.1f}%)")

# %%
# 5量子ビットGHZ状態
exec_ghz5 = transpiler.transpile(ghz_state, bindings={"n": 5})
result_ghz5 = exec_ghz5.sample(transpiler.executor(), shots=1000).result()

print("\n=== 5量子ビットGHZ状態 ===")
print("|GHZ⟩ = (|00000⟩ + |11111⟩)/√2\n")
for value, count in result_ghz5.results:
    percentage = count / 1000 * 100
    print(f"  {value}: {count} ({percentage:.1f}%)")

# %% [markdown]
# ### GHZ状態の特徴
#
# - N個の量子ビットすべてが完全に相関している
# - 1つでも測定すると、残り全ての状態が決まる
# - 出力は「全部0」か「全部1」のどちらかだけ

# %% [markdown]
# ### GHZ回路の可視化

# %%
qiskit_ghz = transpiler.to_circuit(ghz_state, bindings={"n": 4})
print("=== 4量子ビットGHZ回路 ===")
print(qiskit_ghz.draw(output="text"))

# %% [markdown]
# ## 8. その他の2量子ビットゲート
#
# Qamomileで使える主な2量子ビットゲートを紹介します。

# %% [markdown]
# ### SWAPゲート
#
# 2つの量子ビットの状態を交換します。

# %%
@qm.qkernel
def swap_example() -> tuple[qm.Bit, qm.Bit]:
    """SWAPゲートで状態を交換"""
    q0 = qm.qubit(name="q0")
    q1 = qm.qubit(name="q1")

    # q0 を |1⟩ に、q1 は |0⟩ のまま
    q0 = qm.x(q0)

    # SWAP で交換
    q0, q1 = qm.swap(q0, q1)

    # 交換後: q0=|0⟩, q1=|1⟩
    return qm.measure(q0), qm.measure(q1)


# %%
exec_swap = transpiler.transpile(swap_example)
result_swap = exec_swap.sample(transpiler.executor(), shots=1000).result()

print("=== SWAPゲートの例 ===")
print("SWAP前: q0=|1⟩, q1=|0⟩")
print("SWAP後: q0=|0⟩, q1=|1⟩\n")
for value, count in result_swap.results:
    print(f"  結果: {value}, 回数: {count}")

# %% [markdown]
# ### RZZゲート
#
# 両方の量子ビットに同時に Z 回転を適用します。
# 量子最適化アルゴリズム（QAOA）で重要なゲートです。

# %%
@qm.qkernel
def rzz_example(theta: qm.Float) -> tuple[qm.Bit, qm.Bit]:
    """RZZゲートの例"""
    q0 = qm.qubit(name="q0")
    q1 = qm.qubit(name="q1")

    # 重ね合わせを作る
    q0 = qm.h(q0)
    q1 = qm.h(q1)

    # RZZ ゲートを適用
    q0, q1 = qm.rzz(q0, q1, angle=theta)

    # 干渉のために H を適用
    q0 = qm.h(q0)
    q1 = qm.h(q1)

    return qm.measure(q0), qm.measure(q1)


# %%
exec_rzz = transpiler.transpile(rzz_example, bindings={"theta": math.pi / 2})
result_rzz = exec_rzz.sample(transpiler.executor(), shots=1000).result()

print("=== RZZゲートの例（θ=π/2）===")
for value, count in result_rzz.results:
    print(f"  結果: {value}, 回数: {count}")

# %% [markdown]
# ### CPゲート（制御位相ゲート）
#
# 制御ビットが |1⟩ のとき、ターゲットに位相を付加します。
# 量子フーリエ変換（QFT）で使われます。

# %%
@qm.qkernel
def cp_example() -> tuple[qm.Bit, qm.Bit]:
    """CPゲートの例"""
    q0 = qm.qubit(name="q0")
    q1 = qm.qubit(name="q1")

    # 両方を重ね合わせにする
    q0 = qm.h(q0)
    q1 = qm.h(q1)

    # CP ゲートを適用（π/2 の位相）
    q0, q1 = qm.cp(q0, q1, math.pi / 2)

    # 干渉のために H を適用
    q0 = qm.h(q0)
    q1 = qm.h(q1)

    return qm.measure(q0), qm.measure(q1)


# %%
exec_cp = transpiler.transpile(cp_example)
result_cp = exec_cp.sample(transpiler.executor(), shots=1000).result()

print("=== CPゲートの例（θ=π/2）===")
for value, count in result_cp.results:
    print(f"  結果: {value}, 回数: {count}")

# %% [markdown]
# ## 9. まとめ
#
# このチュートリアルでは、以下のことを学びました：
#
# ### 複数量子ビットの扱い
# ```python
# # 個別に作成
# q0 = qm.qubit(name="q0")
# q1 = qm.qubit(name="q1")
#
# # 配列として作成
# qubits = qm.qubit_array(n, name="q")
# ```
#
# ### 2量子ビットゲート
# | ゲート | Qamomile | 効果 |
# |--------|----------|------|
# | CNOT | `qm.cx(ctrl, tgt)` | 制御NOTゲート |
# | SWAP | `qm.swap(q0, q1)` | 状態を交換 |
# | RZZ | `qm.rzz(q0, q1, θ)` | ZZ相互作用 |
# | CP | `qm.cp(ctrl, tgt, θ)` | 制御位相ゲート |
#
# ### 重要：2量子ビットゲートの戻り値
# ```python
# # 必ず両方受け取る！
# q0, q1 = qm.cx(q0, q1)
# ```
#
# ### ループ処理
# ```python
# for i in qm.range(n):
#     qubits[i] = qm.h(qubits[i])
# ```
#
# ### 重要な状態
# - **ベル状態**: 2量子ビットの最大エンタングルメント
# - **GHZ状態**: N量子ビットのエンタングルメント
#
# 次のチュートリアル（`04_algorithms.py`）では、これらの知識を使って
# 最初の量子アルゴリズム「Deutsch-Jozsa」を実装します。
