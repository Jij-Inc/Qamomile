# ---
# jupyter:
#   jupytext:
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
# # Qamomileの量子ゲート一覧
#
# Qamomileで利用できる基本的な量子ゲートのリファレンスです。
#
# ## このチュートリアルで学ぶこと
# - Qamomileで利用可能な全ての1量子ビットゲート
# - Qamomileで利用可能な全ての多量子ビットゲート
# - ゲートの戻り値パターン（線形型の実践）

# %%
import math

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## 1. 前回の復習
#
# 前のチュートリアル（`02_type_system`）では、`qmc.qubit()` と `qmc.qubit_array()` で
# 量子ビットを作成する方法を学びました。ここではそれらを説明なしに使います。
# 全ての量子ビットの初期状態は $|0\rangle$ です。

# %% [markdown]
# ---
# ## 2. 1量子ビットゲート
#
# Qamomileには6つの1量子ビットゲートがあります。以下が**全て**のリストです：
#
# | ゲート | 構文 | パラメータ |
# |------|--------|------------|
# | H | `qmc.h(q)` | なし |
# | X | `qmc.x(q)` | なし |
# | P | `qmc.p(q, theta)` | 角度 |
# | RX | `qmc.rx(q, angle)` | 角度 |
# | RY | `qmc.ry(q, angle)` | 角度 |
# | RZ | `qmc.rz(q, angle)` | 角度 |

# %% [markdown]
# ### 2.1 Hゲート（アダマール）
#
# $$H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix},
# \quad H|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$$


# %%
@qmc.qkernel
def h_gate_demo() -> qmc.Bit:
    q = qmc.qubit(name="q")
    q = qmc.h(q)
    return qmc.measure(q)


h_gate_demo.draw()

# %%
result_h = (
    transpiler.transpile(h_gate_demo).sample(transpiler.executor(), shots=1000).result()
)
print("=== H Gate ===")
for value, count in result_h.results:
    print(f"  {value}: {count} ({count / 10:.1f}%)")

# %% [markdown]
# およそ50/50の結果となり、重ね合わせ状態であることが確認できます。

# %% [markdown]
# ### 2.2 Xゲート（NOT）
#
# $$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix},
# \quad X|0\rangle = |1\rangle$$


# %%
@qmc.qkernel
def x_gate_demo() -> qmc.Bit:
    q = qmc.qubit(name="q")
    q = qmc.x(q)
    return qmc.measure(q)


x_gate_demo.draw()

# %%
result_x = (
    transpiler.transpile(x_gate_demo).sample(transpiler.executor(), shots=1000).result()
)
print("=== X Gate ===")
for value, count in result_x.results:
    print(f"  {value}: {count}")

# %% [markdown]
# $|0\rangle$ から常に $|1\rangle$ が得られます。

# %% [markdown]
# ### 2.3 Pゲート（位相）
#
# $$P(\theta) = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\theta} \end{pmatrix}$$
#
# 位相だけでは測定確率は変わりません。その効果を確認するには、
# Hゲートで挟みます：H-P($\pi$)-H はXゲートと等価です。


# %%
@qmc.qkernel
def p_gate_demo() -> qmc.Bit:
    q = qmc.qubit(name="q")
    q = qmc.h(q)
    q = qmc.p(q, math.pi)
    q = qmc.h(q)
    return qmc.measure(q)


p_gate_demo.draw()

# %%
result_p = (
    transpiler.transpile(p_gate_demo).sample(transpiler.executor(), shots=1000).result()
)
print("=== P Gate: H-P(pi)-H = X ===")
for value, count in result_p.results:
    print(f"  {value}: {count}")

# %% [markdown]
# 常に1になります。位相が $|0\rangle$ の破壊的干渉を引き起こすためです。

# %% [markdown]
# ### 2.4 RXゲート（X軸回転）
#
# $$RX(\theta) = \exp\!\bigl(-i\,\tfrac{\theta}{2}\,X\bigr)
#   = \begin{pmatrix} \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
#     -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$


# %%
@qmc.qkernel
def rx_demo(theta: qmc.Float) -> qmc.Bit:
    q = qmc.qubit(name="q")
    q = qmc.rx(q, theta)
    return qmc.measure(q)


rx_demo.draw()

# %%
print("=== RX Gate at Different Angles ===\n")
for angle, name in zip(
    [0, math.pi / 4, math.pi / 2, math.pi], ["0", "pi/4", "pi/2", "pi"]
):
    result = (
        transpiler.transpile(rx_demo, bindings={"theta": angle})
        .sample(transpiler.executor(), shots=1000)
        .result()
    )
    print(f"RX({name}):")
    for value, count in result.results:
        print(f"  {value}: {count} ({count / 10:.1f}%)")
    print()

# %% [markdown]
# - **RX(0)**: 回転なし -- 常に0。
# - **RX(pi/4)**: わずかな回転 -- 1が出始めます。
# - **RX(pi/2)**: およそ50/50で、Hゲートと同様です。
# - **RX(pi)**: 常に1で、Xゲートと等価です（グローバル位相を除く）。

# %% [markdown]
# ### 2.5 RYゲート（Y軸回転）
#
# $$RY(\theta) = \exp\!\bigl(-i\,\tfrac{\theta}{2}\,Y\bigr)
#   = \begin{pmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
#     \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$
#
# RXと異なり、RYは純粋に実数の振幅を生成します。


# %%
@qmc.qkernel
def ry_demo(theta: qmc.Float) -> qmc.Bit:
    q = qmc.qubit(name="q")
    q = qmc.ry(q, theta)
    return qmc.measure(q)


ry_demo.draw()

# %%
result_ry = (
    transpiler.transpile(ry_demo, bindings={"theta": math.pi / 2})
    .sample(transpiler.executor(), shots=1000)
    .result()
)
print("=== RY(pi/2) ===")
for value, count in result_ry.results:
    print(f"  {value}: {count} ({count / 10:.1f}%)")

# %% [markdown]
# およそ50/50で、Hゲートと同様の結果になります。

# %% [markdown]
# ### 2.6 RZゲート（Z軸回転）
#
# $$RZ(\theta) = \exp\!\bigl(-i\,\tfrac{\theta}{2}\,Z\bigr)
#   = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$
#
# Pゲートと同様に、RZは位相にのみ影響します。$|0\rangle$ だけでは効果が
# 見えないため、Hゲートで挟みます。


# %%
@qmc.qkernel
def rz_demo() -> qmc.Bit:
    q = qmc.qubit(name="q")
    q = qmc.h(q)
    q = qmc.rz(q, math.pi)
    q = qmc.h(q)
    return qmc.measure(q)


rz_demo.draw()

# %%
result_rz = (
    transpiler.transpile(rz_demo).sample(transpiler.executor(), shots=1000).result()
)
print("=== RZ(pi) sandwiched between H gates ===")
for value, count in result_rz.results:
    print(f"  {value}: {count}")

# %% [markdown]
# 常に1になります。Z回転は重ね合わせ内の相対位相を反転させ、
# 2番目のHゲートがその位相差をビット反転に変換します。

# %% [markdown]
# ---
# ## 3. 多量子ビットゲート
#
# Qamomileには5つの2量子ビットゲートがあります。以下が**全て**のリストです：
#
# | ゲート | 構文 | パラメータ |
# |------|--------|------------|
# | CX (CNOT) | `qmc.cx(control, target)` | なし |
# | CZ | `qmc.cz(control, target)` | なし |
# | SWAP | `qmc.swap(q0, q1)` | なし |
# | CP | `qmc.cp(control, target, theta)` | 角度 |
# | RZZ | `qmc.rzz(q0, q1, angle)` | 角度 |
#
# 全ての2量子ビットゲートは**両方の**量子ビットを返します（セクション4参照）。
#
# > **量子ビットの順序の復習**（[01_introduction](01_introduction.ipynb)参照）：
# > タプルの結果は配列の順序 `(q[0], q[1], ...)` に従います。
# > ケット表記はビッグエンディアンです：$|q_n \cdots q_1 q_0\rangle$。
# > 例えば、`(q0, q1)` に対する結果 `(1, 0)` は $|01\rangle$ に対応します。

# %% [markdown]
# ### 3.1 CXゲート（CNOT）
#
# 制御量子ビットが $|1\rangle$ のとき、ターゲットを反転させます。
#
# $|10\rangle \to |11\rangle,\quad |11\rangle \to |10\rangle$
# （$|00\rangle$ と $|01\rangle$ は変化しません。）


# %%
@qmc.qkernel
def cx_demo() -> tuple[qmc.Bit, qmc.Bit]:
    ctrl = qmc.qubit(name="ctrl")
    tgt = qmc.qubit(name="tgt")
    ctrl = qmc.x(ctrl)
    ctrl, tgt = qmc.cx(ctrl, tgt)
    return qmc.measure(ctrl), qmc.measure(tgt)


cx_demo.draw()

# %%
result_cx = (
    transpiler.transpile(cx_demo).sample(transpiler.executor(), shots=1000).result()
)
print("=== CX Gate (control=1) ===")
for value, count in result_cx.results:
    print(f"  {value}: {count}")

# %% [markdown]
# 制御が $|1\rangle$ なので、ターゲットが反転します。
# 結果 $(1, 1)$ は `(ctrl, tgt) = (1, 1)` を意味し、ケット表記では $|11\rangle$ です。

# %% [markdown]
# ### 3.2 CZゲート（制御Z）
#
# $CZ|11\rangle = -|11\rangle$；他の全ての基底状態は変化しません。
# このゲートは2つの量子ビットに対して対称です。
#
# CZは位相のみを変化させるため、測定確率には直接見えません。
# よく知られた恒等式として、**ターゲットのみに**Hゲートを挟むとCZをCXに変換できます：
# $(I \otimes H) \cdot CZ \cdot (I \otimes H) = CX$。
# 以下でこれを実演します。


# %%
@qmc.qkernel
def cz_demo() -> tuple[qmc.Bit, qmc.Bit]:
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")
    q0 = qmc.x(q0)  # control = |1>
    q1 = qmc.h(q1)  # H on target only
    q0, q1 = qmc.cz(q0, q1)
    q1 = qmc.h(q1)  # H on target only
    return qmc.measure(q0), qmc.measure(q1)


cz_demo.draw()

# %%
result_cz = (
    transpiler.transpile(cz_demo).sample(transpiler.executor(), shots=1000).result()
)
print("=== CZ Gate: (I⊗H)·CZ·(I⊗H) = CX ===")
for value, count in result_cz.results:
    print(f"  {value}: {count}")

# %% [markdown]
# ターゲット量子ビットのみにHゲートを挟むことで、CZはCXと全く同じ動作をします。
# 制御が $|1\rangle$ なのでターゲットが反転し、結果 $(1, 1)$ は
# `(q0, q1) = (1, 1)`（ケット $|11\rangle$）となり、上のCXの例と一致します。

# %% [markdown]
# ### 3.3 SWAPゲート
#
# 量子状態を交換します：$\text{SWAP}|a,b\rangle = |b,a\rangle$。


# %%
@qmc.qkernel
def swap_demo() -> tuple[qmc.Bit, qmc.Bit]:
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")
    q0 = qmc.x(q0)  # q0=|1>, q1=|0>
    q0, q1 = qmc.swap(q0, q1)
    return qmc.measure(q0), qmc.measure(q1)


swap_demo.draw()

# %%
result_swap = (
    transpiler.transpile(swap_demo).sample(transpiler.executor(), shots=1000).result()
)
print("=== SWAP Gate (q0=|1>, q1=|0> before) ===")
for value, count in result_swap.results:
    print(f"  {value}: {count}")

# %% [markdown]
# SWAP前：`q0=|1>, q1=|0>`。SWAP後：`q0=|0>, q1=|1>`。
# 結果 $(0, 1)$ は `(q0, q1) = (0, 1)` を意味し、ケット表記では $|10\rangle$ です。

# %% [markdown]
# ### 3.4 CPゲート（制御位相）
#
# $CP(\theta)|11\rangle = e^{i\theta}|11\rangle$；他の基底状態は変化しません。
# 量子フーリエ変換（QFT）の中核となるゲートです。


# %%
@qmc.qkernel
def cp_demo() -> tuple[qmc.Bit, qmc.Bit]:
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")
    q0 = qmc.x(q0)  # control = |1>
    q1 = qmc.h(q1)  # H on target only
    q0, q1 = qmc.cp(q0, q1, math.pi)
    q1 = qmc.h(q1)  # H on target only
    return qmc.measure(q0), qmc.measure(q1)


cp_demo.draw()

# %%
result_cp = (
    transpiler.transpile(cp_demo).sample(transpiler.executor(), shots=1000).result()
)
print("=== CP(pi): same pattern as CZ ===")
for value, count in result_cp.results:
    print(f"  {value}: {count}")

# %% [markdown]
# CP($\pi$) はCZと同一なので、CZの例と同じ結果になります：
# `(q0, q1) = (1, 1)`（ケット $|11\rangle$）が常に得られます。より小さな角度
# （例えば $\pi/4$）では部分的な位相回転となり、QFTの構成要素になります。

# %% [markdown]
# ### 3.5 RZZゲート
#
# $$RZZ(\theta) = \exp\!\bigl(-i\,\tfrac{\theta}{2}\,Z \otimes Z\bigr)$$
#
# QAOAにおいて、イジング相互作用項をエンコードする際に重要なゲートです。


# %%
@qmc.qkernel
def rzz_demo(theta: qmc.Float) -> tuple[qmc.Bit, qmc.Bit]:
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")
    q0 = qmc.h(q0)
    q1 = qmc.h(q1)
    q0, q1 = qmc.rzz(q0, q1, theta)
    q0 = qmc.h(q0)
    q1 = qmc.h(q1)
    return qmc.measure(q0), qmc.measure(q1)


rzz_demo.draw()

# %%
result_rzz = (
    transpiler.transpile(rzz_demo, bindings={"theta": math.pi / 2})
    .sample(transpiler.executor(), shots=1000)
    .result()
)
print("=== RZZ(pi/2) with H-RZZ-H ===")
for value, count in result_rzz.results:
    print(f"  {value}: {count} ({count / 10:.1f}%)")

# %% [markdown]
# RZZは量子ビット間に相関を生み出します。支配的な結果は
# $(0, 0)$ と $(1, 1)$、すなわち `(q0, q1)` が一致する状態です（ケット表記では
# $|00\rangle$ と $|11\rangle$）。このゲートはQAOAを扱う際に再び登場します。

# %% [markdown]
# ---
# ## 4. ゲートの戻り値パターン
#
# 全てのゲートは、消費した量子ビットを返します。これは**線形型システム**の帰結です：
# 量子ビットがゲートに渡されると、古いハンドルは無効化され、新しいハンドルが返されます。
#
# ### 1量子ビットゲート：1つの量子ビットを返す
# ```python
# q = qmc.h(q)
# q = qmc.rx(q, theta)
# ```
#
# ### 2量子ビットゲート：両方の量子ビットを返す
# ```python
# q0, q1 = qmc.cx(q0, q1)
# q0, q1 = qmc.swap(q0, q1)
# ```
#
# ### 量子ビット配列の場合：インデックスに再代入する
# ```python
# qubits[i] = qmc.h(qubits[i])
# qubits[i], qubits[j] = qmc.cx(qubits[i], qubits[j])
# ```
#
# ### 角度パラメータは返されない
# ```python
# q = qmc.rx(q, theta)              # returns Qubit, not (Qubit, Float)
# q0, q1 = qmc.rzz(q0, q1, theta)  # returns (Qubit, Qubit)
# ```
#
# 戻り値を無視したり、2量子ビットゲートから片方の量子ビットだけを取得したりすると、
# ビルド時にエラーが発生します。

# %% [markdown]
# 以下は、複数のゲート種類を1つの回路で組み合わせた具体例です。


# %%
@qmc.qkernel
def return_value_demo() -> qmc.Vector[qmc.Bit]:
    qubits = qmc.qubit_array(3, name="q")

    # Single-qubit: assign back to the same slot
    qubits[0] = qmc.h(qubits[0])
    qubits[1] = qmc.rx(qubits[1], math.pi / 4)

    # Two-qubit: unpack both return values
    qubits[0], qubits[1] = qmc.cx(qubits[0], qubits[1])
    qubits[1], qubits[2] = qmc.swap(qubits[1], qubits[2])

    return qmc.measure(qubits)


return_value_demo.draw()

# %%
result_rv = (
    transpiler.transpile(return_value_demo)
    .sample(transpiler.executor(), shots=1000)
    .result()
)
print("=== Return Value Pattern Demo ===")
for value, count in result_rv.results:
    print(f"  {value}: {count}")

# %% [markdown]
# **結果の解釈：**
#
# 1. Hゲートが `q[0]` を重ね合わせ状態にします：$\frac{|0\rangle + |1\rangle}{\sqrt{2}}$
# 2. RX($\pi/4$) が `q[1]` を $|0\rangle$ からわずかに回転させます
# 3. CXが `q[0]` と `q[1]` をエンタングルさせます -- `q[0]` が $|1\rangle$ のとき
#    `q[1]` が反転します
# 4. SWAPが `q[1]` と `q[2]` を交換し、エンタングルした状態を `q[2]` に移動させます
#
# 支配的な結果は `(0, 0, 0)` と `(1, 0, 1)`（タプルの順序：
# `(q[0], q[1], q[2])`；ケット表記では $|000\rangle$ と $|101\rangle$）になるはずで、
# これは `q[0]` と `q[2]` の位置にスワップされた量子ビットとの間のエンタングルメントを
# 反映しています。`q[1]` に対する小さなRX回転により、他のビット列からの
# わずかな寄与も見られます。

# %% [markdown]
# ---
# ## 5. まとめ表
#
# ### 全ての1量子ビットゲート
#
# | ゲート | Qamomileの構文 | 数学的定義 |
# |------|-----------------|------------------------|
# | H（アダマール） | `q = qmc.h(q)` | $\frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix}$ |
# | X（NOT） | `q = qmc.x(q)` | $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ |
# | P（位相） | `q = qmc.p(q, theta)` | $\begin{pmatrix}1&0\\0&e^{i\theta}\end{pmatrix}$ |
# | RX | `q = qmc.rx(q, angle)` | $\exp(-i\,\frac{\text{angle}}{2}\,X)$ |
# | RY | `q = qmc.ry(q, angle)` | $\exp(-i\,\frac{\text{angle}}{2}\,Y)$ |
# | RZ | `q = qmc.rz(q, angle)` | $\exp(-i\,\frac{\text{angle}}{2}\,Z)$ |
#
# ### 全ての多量子ビットゲート
#
# | ゲート | Qamomileの構文 | 説明 |
# |------|-----------------|-------------|
# | CX（CNOT） | `q0, q1 = qmc.cx(q0, q1)` | 制御が $\|1\rangle$ のときターゲットを反転 |
# | CZ | `q0, q1 = qmc.cz(q0, q1)` | 制御が $\|1\rangle$ のときターゲットにZを適用 |
# | SWAP | `q0, q1 = qmc.swap(q0, q1)` | 2つの量子ビットの状態を交換 |
# | CP | `q0, q1 = qmc.cp(q0, q1, theta)` | $\theta$ による制御位相回転 |
# | RZZ | `q0, q1 = qmc.rzz(q0, q1, angle)` | $\exp(-i\,\frac{\text{angle}}{2}\,Z \otimes Z)$ |
#
# ### 重要なルール
#
# 全てのゲートは消費した量子ビットを返します。常に戻り値を受け取ってください。
#
# 次のチュートリアルでは、これらのゲートを使って最初の量子アルゴリズムを構築します。

# %% [markdown]
# ## このチュートリアルで学んだこと
#
# - **Qamomileで利用可能な全ての1量子ビットゲート** -- H、X、P、RX、RY、RZの各ゲートは、量子ビット（とオプションで角度）を受け取り、変換された量子ビットを返します。
# - **Qamomileで利用可能な全ての多量子ビットゲート** -- CX（CNOT）、CZ、SWAP、CP、RZZの各ゲートは、関与する全ての量子ビットをタプルとして消費・返却します。
# - **ゲートの戻り値パターン（線形型の実践）** -- 全てのゲートは消費した量子ビットを返します。線形型システムを満たすために、常に戻り値を受け取る必要があります。
