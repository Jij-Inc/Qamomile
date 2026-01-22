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
# # 単一量子ビットゲートと重ね合わせ
#
# このチュートリアルでは、量子コンピューティングの核心概念である**重ね合わせ（superposition）**と、
# それを実現する様々な量子ゲートについて学びます。
#
# ## このチュートリアルで学ぶこと
# - アダマールゲート（Hゲート）と重ね合わせ状態
# - 確率的な測定結果の解釈
# - 回転ゲート（RX, RY, RZ）とパラメータ
# - 位相ゲート（Pゲート）
# - パラメータ化された量子回路

# %%
import math
import qamomile.circuit as qm
from qamomile.qiskit import QiskitTranspiler

# トランスパイラを準備
transpiler = QiskitTranspiler()

# %% [markdown]
# ## 1. アダマールゲート（Hゲート）
#
# **アダマールゲート**は、量子コンピューティングで最も重要なゲートの1つです。
# このゲートは、量子ビットを**重ね合わせ状態**にします。
#
# ### 数学的な定義
#
# アダマールゲートは以下の変換を行います：
#
# $$H|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$$
#
# $$H|1\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}$$
#
# つまり、`|0⟩` に H ゲートを適用すると、`|0⟩` と `|1⟩` が等しい確率で出る状態になります。

# %%
@qm.qkernel
def hadamard_circuit() -> qm.Bit:
    """アダマールゲートを適用して重ね合わせ状態を作る"""
    q = qm.qubit(name="q")

    # Hゲートで重ね合わせ状態を作成
    q = qm.h(q)

    return qm.measure(q)


# %%
# 実行してみましょう
executable = transpiler.transpile(hadamard_circuit)
job = executable.sample(transpiler.executor(), shots=1000)
result = job.result()

print("=== アダマールゲートの実行結果 ===")
for value, count in result.results:
    percentage = count / 1000 * 100
    print(f"  測定結果: {value}, 回数: {count} ({percentage:.1f}%)")

# %% [markdown]
# ### 結果の解釈
#
# 結果を見ると、`0` と `1` がそれぞれ約50%の確率で出ていることがわかります。
# これが**重ね合わせ**の本質です。
#
# - 量子ビットは測定するまで、0と1の両方の状態を「同時に」持っています
# - 測定した瞬間に、どちらか一方に「崩壊」します
# - どちらに崩壊するかは確率的に決まります
#
# これは、量子コイン投げのようなものです！

# %% [markdown]
# ### 回路の可視化

# %%
qiskit_circuit = transpiler.to_circuit(hadamard_circuit)
print("=== アダマールゲート回路 ===")
print(qiskit_circuit.draw(output="text"))

# %% [markdown]
# ## 2. 複数のゲートを組み合わせる
#
# 量子ゲートは組み合わせることで、様々な状態を作り出せます。
# Hゲートを2回適用するとどうなるでしょうか？

# %%
@qm.qkernel
def double_hadamard() -> qm.Bit:
    """Hゲートを2回適用する"""
    q = qm.qubit(name="q")

    q = qm.h(q)  # 1回目: |0⟩ → (|0⟩+|1⟩)/√2
    q = qm.h(q)  # 2回目: (|0⟩+|1⟩)/√2 → |0⟩

    return qm.measure(q)


# %%
executable2 = transpiler.transpile(double_hadamard)
job2 = executable2.sample(transpiler.executor(), shots=1000)
result2 = job2.result()

print("=== Hゲート2回の実行結果 ===")
for value, count in result2.results:
    print(f"  測定結果: {value}, 回数: {count}")

# %% [markdown]
# ### なぜ0だけになるのか？
#
# Hゲートを2回適用すると、元の状態に戻ります！
#
# $$H \cdot H = I$$
#
# これは、Hゲートが「自己逆元」（自分自身の逆操作）であるためです。
# 数学的には：
#
# $$H \cdot H |0\rangle = H \cdot \frac{|0\rangle + |1\rangle}{\sqrt{2}} = |0\rangle$$

# %% [markdown]
# ## 3. |+⟩ 状態と |−⟩ 状態
#
# 重ね合わせ状態には名前がついています：
#
# - **|+⟩ 状態**: $\frac{|0\rangle + |1\rangle}{\sqrt{2}}$ （H|0⟩）
# - **|−⟩ 状態**: $\frac{|0\rangle - |1\rangle}{\sqrt{2}}$ （H|1⟩）
#
# これらは測定すると同じ確率分布（50/50）になりますが、量子的には異なる状態です。

# %%
@qm.qkernel
def plus_state() -> qm.Bit:
    """|+⟩ 状態を作成"""
    q = qm.qubit(name="q")
    q = qm.h(q)  # |0⟩ → |+⟩
    return qm.measure(q)


@qm.qkernel
def minus_state() -> qm.Bit:
    """|−⟩ 状態を作成"""
    q = qm.qubit(name="q")
    q = qm.x(q)  # |0⟩ → |1⟩
    q = qm.h(q)  # |1⟩ → |−⟩
    return qm.measure(q)


# %%
# 両方実行して比較
exec_plus = transpiler.transpile(plus_state)
exec_minus = transpiler.transpile(minus_state)

result_plus = exec_plus.sample(transpiler.executor(), shots=1000).result()
result_minus = exec_minus.sample(transpiler.executor(), shots=1000).result()

print("=== |+⟩ 状態の測定結果 ===")
for value, count in result_plus.results:
    print(f"  測定結果: {value}, 回数: {count}")

print("\n=== |−⟩ 状態の測定結果 ===")
for value, count in result_minus.results:
    print(f"  測定結果: {value}, 回数: {count}")

print("\n両方とも約50/50になりますが、量子的には異なる状態です！")

# %% [markdown]
# ## 4. 回転ゲート（RX, RY, RZ）
#
# より細かい制御のために、**回転ゲート**を使います。
# これらは回転角度をパラメータとして受け取ります。
#
# ### 回転ゲートの種類
#
# - **RX(θ)**: X軸周りにθラジアン回転
# - **RY(θ)**: Y軸周りにθラジアン回転
# - **RZ(θ)**: Z軸周りにθラジアン回転
#
# ### ブロッホ球のイメージ
#
# 量子ビットの状態は「ブロッホ球」という球面上の点として表現できます。
# 回転ゲートは、この球面上で状態を回転させます。

# %%
@qm.qkernel
def rx_circuit(theta: qm.Float) -> qm.Bit:
    """RX回転ゲートを適用"""
    q = qm.qubit(name="q")
    q = qm.rx(q, theta)  # X軸周りにtheta回転
    return qm.measure(q)


# %%
# 角度を変えて実行してみましょう
angles = [0, math.pi / 4, math.pi / 2, math.pi]
angle_names = ["0", "π/4", "π/2", "π"]

print("=== RXゲートの角度による変化 ===\n")

for angle, name in zip(angles, angle_names):
    executable = transpiler.transpile(rx_circuit, bindings={"theta": angle})
    result = executable.sample(transpiler.executor(), shots=1000).result()

    print(f"RX({name}):")
    for value, count in result.results:
        percentage = count / 1000 * 100
        print(f"  {value}: {count} ({percentage:.1f}%)")
    print()

# %% [markdown]
# ### 結果の解釈
#
# - **RX(0)**: 何も回転しない → 常に0
# - **RX(π/4)**: 少し回転 → 1が少し出始める
# - **RX(π/2)**: 45度回転 → ほぼ50/50（Hゲートと似た効果）
# - **RX(π)**: 180度回転 → 完全に反転して常に1（Xゲートと同じ効果）

# %% [markdown]
# ## 5. パラメータ化された回路
#
# Qamomileでは、パラメータを変数として扱い、実行時に値を指定できます。
# これは変分量子アルゴリズム（VQA）で重要な機能です。

# %%
@qm.qkernel
def parameterized_circuit(theta: qm.Float, phi: qm.Float) -> qm.Bit:
    """複数のパラメータを持つ回路"""
    q = qm.qubit(name="q")

    q = qm.ry(q, theta)  # Y軸回転
    q = qm.rz(q, phi)    # Z軸回転

    return qm.measure(q)


# %%
# パラメータを指定してコンパイル
# parameters: 実行時に変更可能なパラメータ名のリスト
executable_param = transpiler.transpile(
    parameterized_circuit,
    parameters=["theta", "phi"]  # これらは後で値を指定
)

# 異なるパラメータで実行
params_list = [
    {"theta": 0, "phi": 0},
    {"theta": math.pi / 2, "phi": 0},
    {"theta": math.pi / 2, "phi": math.pi},
]

print("=== パラメータ化回路の実行 ===\n")

for params in params_list:
    result = executable_param.sample(
        transpiler.executor(),
        bindings=params,
        shots=1000
    ).result()

    print(f"theta={params['theta']:.2f}, phi={params['phi']:.2f}:")
    for value, count in result.results:
        print(f"  {value}: {count}")
    print()

# %% [markdown]
# ## 6. 位相ゲート（Pゲート）
#
# **位相ゲート** P(θ) は、`|1⟩` 状態に位相 $e^{i\theta}$ を付加します。
#
# $$P(\theta)|0\rangle = |0\rangle$$
# $$P(\theta)|1\rangle = e^{i\theta}|1\rangle$$
#
# 位相は測定結果には直接影響しませんが、量子干渉で重要な役割を果たします。

# %%
@qm.qkernel
def phase_example() -> qm.Bit:
    """位相ゲートの例"""
    q = qm.qubit(name="q")

    q = qm.h(q)           # 重ね合わせを作る
    q = qm.p(q, math.pi)  # |1⟩ に位相 π を付加（符号反転）
    q = qm.h(q)           # 干渉させる

    return qm.measure(q)


# %%
exec_phase = transpiler.transpile(phase_example)
result_phase = exec_phase.sample(transpiler.executor(), shots=1000).result()

print("=== 位相ゲートの例（H-P(π)-H = X） ===")
for value, count in result_phase.results:
    print(f"  測定結果: {value}, 回数: {count}")

# %% [markdown]
# ### 結果の説明
#
# H → P(π) → H の組み合わせは、Xゲートと同じ効果があります！
#
# これは量子干渉の一例です：
# 1. Hゲートで重ね合わせを作る
# 2. P(π)で|1⟩成分の符号を反転
# 3. 2回目のHゲートで干渉させると、|1⟩だけが残る

# %% [markdown]
# ## 7. 回路の可視化：回転ゲート

# %%
qiskit_param = transpiler.to_circuit(
    parameterized_circuit,
    bindings={"theta": math.pi / 4, "phi": math.pi / 2}
)
print("=== パラメータ化回路の構造 ===")
print(qiskit_param.draw(output="text"))

# %% [markdown]
# ## 8. まとめ
#
# このチュートリアルでは、以下のことを学びました：
#
# ### 量子ゲート
# | ゲート | Qamomile | 効果 |
# |--------|----------|------|
# | アダマール | `qm.h(q)` | 重ね合わせを作成 |
# | X回転 | `qm.rx(q, θ)` | X軸周りに回転 |
# | Y回転 | `qm.ry(q, θ)` | Y軸周りに回転 |
# | Z回転 | `qm.rz(q, θ)` | Z軸周りに回転 |
# | 位相 | `qm.p(q, θ)` | |1⟩に位相を付加 |
#
# ### 重要な概念
# - **重ね合わせ**: 0と1を同時に持つ状態。測定すると確率的に一方に崩壊
# - **量子干渉**: ゲートの組み合わせで確率振幅が強め合ったり打ち消し合ったりする
# - **パラメータ化**: `parameters` 引数で実行時に値を変更可能
#
# ### パラメータ化回路のパターン
#
# ```python
# # 1. パラメータ付きで回路を定義
# @qm.qkernel
# def circuit(theta: qm.Float) -> qm.Bit:
#     q = qm.qubit(name="q")
#     q = qm.rx(q, theta)
#     return qm.measure(q)
#
# # 2. パラメータ名を指定してコンパイル
# executable = transpiler.transpile(circuit, parameters=["theta"])
#
# # 3. 実行時に値を指定
# result = executable.sample(executor, bindings={"theta": 0.5}, shots=1000)
# ```
#
# 次のチュートリアル（`03_entanglement.py`）では、複数の量子ビットを扱い、
# 量子力学の最も不思議な現象である**エンタングルメント（量子もつれ）**について学びます。
