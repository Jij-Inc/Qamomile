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
# ---
# title: Steane [[7,1,3]] 符号
# tags: [error-correction]
# ---
#
# # 量子誤り訂正(2): Steane [[7,1,3]] 符号
#
# <!-- BEGIN auto-tags -->
# **タグ:** <a class="tag-chip" href="../tags/error-correction.md">error-correction</a>
# <!-- END auto-tags -->
#
# [前章](quantum_error_correction.ipynb)では、3量子ビット符号と Shor の9量子ビット符号を実装しました。本章では、より構造がきれいな **Steane [[7,1,3]] 符号** を扱います。
#
# Steane 符号は、古典の Hamming [7,4,3] 符号から作る CSS 符号です。7つの物理量子ビットで1つの論理量子ビットを守り、任意の単一量子ビット Pauli エラー($X$, $Y$, $Z$)を訂正できます。
#
# このチュートリアルで実装することは3つです。
#
# 1. Hamming 符号の構造から Steane 符号のスタビライザーを作る。
# 2. 6つのスタビライザーを測り、シンドロームから単一エラーを訂正する。
# 3. 物理 $H$ を7つ並べるだけで論理 Hadamard $\bar{H}$ になることを確認する。

# %%
# 最新のQamomileをpipからインストールします。
# # !pip install qamomile
# # or
# # !uv add qamomile

# %%
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# Create a seeded backend for reproducible documentation output
from qiskit_aer import AerSimulator

_seeded_backend = AerSimulator(seed_simulator=42, max_parallel_threads=1)
_seeded_executor = transpiler.executor(backend=_seeded_backend)


def _bits7(outcome) -> list[int]:
    """Return seven measured bits in qubit-index order."""
    if isinstance(outcome, (list, tuple)):
        return list(outcome)
    return [(outcome >> i) & 1 for i in range(7)]


def _is_steane_zero_word(outcome) -> bool:
    """Return True when the outcome is an even Hamming codeword."""
    bits = _bits7(outcome)
    h_checks = [
        bits[3] ^ bits[4] ^ bits[5] ^ bits[6],
        bits[1] ^ bits[2] ^ bits[5] ^ bits[6],
        bits[0] ^ bits[2] ^ bits[4] ^ bits[6],
    ]
    return all(check == 0 for check in h_checks) and sum(bits) % 2 == 0


# %% [markdown]
# ## 1. Hamming 符号から CSS 符号へ
#
# 古典 Hamming [7,4,3] 符号のパリティ検査行列を次のように取ります。
#
# $$
# H =
# \begin{pmatrix}
# 0 & 0 & 0 & 1 & 1 & 1 & 1 \\
# 0 & 1 & 1 & 0 & 0 & 1 & 1 \\
# 1 & 0 & 1 & 0 & 1 & 0 & 1
# \end{pmatrix}
# $$
#
# 列 $j$ は $j+1$ の2進表現です。そのため、3ビットのシンドロームを読むだけで、どのビットに誤りが起きたかが一意に分かります。
#
# CSS 符号では、同じパリティ検査行列から2種類のスタビライザーを作ります。
#
# - $Z$ 型スタビライザー: $X$ エラーを検出する。
# - $X$ 型スタビライザー: $Z$ エラーを検出する。
#
# Steane 符号の生成子は次の6つです。
#
# | 種類 | スタビライザー | 検出するエラー |
# | --- | --- | --- |
# | $X$ 型 | $X_3X_4X_5X_6$ | $Z$ |
# | $X$ 型 | $X_1X_2X_5X_6$ | $Z$ |
# | $X$ 型 | $X_0X_2X_4X_6$ | $Z$ |
# | $Z$ 型 | $Z_3Z_4Z_5Z_6$ | $X$ |
# | $Z$ 型 | $Z_1Z_2Z_5Z_6$ | $X$ |
# | $Z$ 型 | $Z_0Z_2Z_4Z_6$ | $X$ |

# %% [markdown]
# ## 2. $\lvert0_L\rangle$ のエンコード
#
# Steane 符号の $\lvert0_L\rangle$ は、偶重みの Hamming 符号語8個の重ね合わせです。
#
# $$
# \lvert0_L\rangle =
# \frac{1}{2\sqrt{2}}
# \sum_{c \in C,\; w(c)\ {\rm even}} \lvert c\rangle
# $$
#
# 次の回路では、3つの $X$ 型スタビライザーに対応するパターンを順に作り、$\lvert0\rangle^{\otimes 7}$ から $\lvert0_L\rangle$ を準備します。


# %%
@qmc.qkernel
def encode_steane_zero(data: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    data[3] = qmc.h(data[3])
    data[3], data[4] = qmc.cx(data[3], data[4])
    data[3], data[5] = qmc.cx(data[3], data[5])
    data[3], data[6] = qmc.cx(data[3], data[6])

    data[1] = qmc.h(data[1])
    data[1], data[2] = qmc.cx(data[1], data[2])
    data[1], data[5] = qmc.cx(data[1], data[5])
    data[1], data[6] = qmc.cx(data[1], data[6])

    data[0] = qmc.h(data[0])
    data[0], data[2] = qmc.cx(data[0], data[2])
    data[0], data[4] = qmc.cx(data[0], data[4])
    data[0], data[6] = qmc.cx(data[0], data[6])

    return data


# %%
@qmc.qkernel
def encode_zero_and_measure() -> qmc.Vector[qmc.Bit]:
    data = qmc.qubit_array(7, name="data")
    data = encode_steane_zero(data)
    return qmc.measure(data)


# %% [markdown]
# エンコーダを測定して、出てくるビット列が偶重み Hamming 符号語だけになっているか確認します。

# %%
print("|0_L⟩ をエンコードして測定")
exe = transpiler.transpile(encode_zero_and_measure)
result = exe.sample(_seeded_executor, shots=1024).result()
valid = sum(count for outcome, count in result.results if _is_steane_zero_word(outcome))
total = sum(count for _, count in result.results)
print(f"  偶重み Hamming 符号語の割合: {valid / total:.3f}")
print(f"  観測された符号語数: {len(result.results)}")

# %% [markdown]
# ## 3. シンドローム測定と訂正
#
# Steane 符号では $X$ エラーと $Z$ エラーを別々に扱えます。
#
# - $Z$ 型スタビライザーを測ると、$X$ 成分のシンドロームが得られる。
# - $X$ 型スタビライザーを測ると、$Z$ 成分のシンドロームが得られる。
#
# シンドローム表は Hamming 符号の列そのものです。
#
# | 誤り位置 | シンドローム $(s_2,s_1,s_0)$ |
# | --- | --- |
# | なし | $(0,0,0)$ |
# | $q_0$ | $(0,0,1)$ |
# | $q_1$ | $(0,1,0)$ |
# | $q_2$ | $(0,1,1)$ |
# | $q_3$ | $(1,0,0)$ |
# | $q_4$ | $(1,0,1)$ |
# | $q_5$ | $(1,1,0)$ |
# | $q_6$ | $(1,1,1)$ |
#
# 実装では `error_type` を `1=X`, `2=Y`, `3=Z` として渡します。`error_pos` は `0..6` の物理量子ビット位置です。


# %%
@qmc.qkernel
def steane_run(
    error_type: qmc.UInt,
    error_pos: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    data = qmc.qubit_array(7, name="data")
    anc = qmc.qubit_array(6, name="anc")

    data = encode_steane_zero(data)

    for i in qmc.range(7):
        if (error_type == 1) & (error_pos == i):  # 1 means X error.
            data[i] = qmc.x(data[i])
        if (error_type == 2) & (error_pos == i):  # 2 means Y error.
            data[i] = qmc.y(data[i])
        if (error_type == 3) & (error_pos == i):  # 3 means Z error.
            data[i] = qmc.z(data[i])

    # Z-type stabilizers: detect the X component.
    data[3], anc[0] = qmc.cx(data[3], anc[0])
    data[4], anc[0] = qmc.cx(data[4], anc[0])
    data[5], anc[0] = qmc.cx(data[5], anc[0])
    data[6], anc[0] = qmc.cx(data[6], anc[0])
    sx_2 = qmc.measure(anc[0])

    data[1], anc[1] = qmc.cx(data[1], anc[1])
    data[2], anc[1] = qmc.cx(data[2], anc[1])
    data[5], anc[1] = qmc.cx(data[5], anc[1])
    data[6], anc[1] = qmc.cx(data[6], anc[1])
    sx_1 = qmc.measure(anc[1])

    data[0], anc[2] = qmc.cx(data[0], anc[2])
    data[2], anc[2] = qmc.cx(data[2], anc[2])
    data[4], anc[2] = qmc.cx(data[4], anc[2])
    data[6], anc[2] = qmc.cx(data[6], anc[2])
    sx_0 = qmc.measure(anc[2])

    # X-type stabilizers: detect the Z component.
    anc[3] = qmc.h(anc[3])
    anc[3], data[3] = qmc.cx(anc[3], data[3])
    anc[3], data[4] = qmc.cx(anc[3], data[4])
    anc[3], data[5] = qmc.cx(anc[3], data[5])
    anc[3], data[6] = qmc.cx(anc[3], data[6])
    anc[3] = qmc.h(anc[3])
    sz_2 = qmc.measure(anc[3])

    anc[4] = qmc.h(anc[4])
    anc[4], data[1] = qmc.cx(anc[4], data[1])
    anc[4], data[2] = qmc.cx(anc[4], data[2])
    anc[4], data[5] = qmc.cx(anc[4], data[5])
    anc[4], data[6] = qmc.cx(anc[4], data[6])
    anc[4] = qmc.h(anc[4])
    sz_1 = qmc.measure(anc[4])

    anc[5] = qmc.h(anc[5])
    anc[5], data[0] = qmc.cx(anc[5], data[0])
    anc[5], data[2] = qmc.cx(anc[5], data[2])
    anc[5], data[4] = qmc.cx(anc[5], data[4])
    anc[5], data[6] = qmc.cx(anc[5], data[6])
    anc[5] = qmc.h(anc[5])
    sz_0 = qmc.measure(anc[5])

    if (~sx_2) & (~sx_1) & sx_0:
        data[0] = qmc.x(data[0])
    if (~sx_2) & sx_1 & (~sx_0):
        data[1] = qmc.x(data[1])
    if (~sx_2) & sx_1 & sx_0:
        data[2] = qmc.x(data[2])
    if sx_2 & (~sx_1) & (~sx_0):
        data[3] = qmc.x(data[3])
    if sx_2 & (~sx_1) & sx_0:
        data[4] = qmc.x(data[4])
    if sx_2 & sx_1 & (~sx_0):
        data[5] = qmc.x(data[5])
    if sx_2 & sx_1 & sx_0:
        data[6] = qmc.x(data[6])

    if (~sz_2) & (~sz_1) & sz_0:
        data[0] = qmc.z(data[0])
    if (~sz_2) & sz_1 & (~sz_0):
        data[1] = qmc.z(data[1])
    if (~sz_2) & sz_1 & sz_0:
        data[2] = qmc.z(data[2])
    if sz_2 & (~sz_1) & (~sz_0):
        data[3] = qmc.z(data[3])
    if sz_2 & (~sz_1) & sz_0:
        data[4] = qmc.z(data[4])
    if sz_2 & sz_1 & (~sz_0):
        data[5] = qmc.z(data[5])
    if sz_2 & sz_1 & sz_0:
        data[6] = qmc.z(data[6])

    return qmc.measure(data)


# %% [markdown]
# $X$, $Y$, $Z$ の全単一エラー、つまり 21 通りを実行します。訂正後に測定したビット列がすべて $\lvert0_L\rangle$ の符号語であれば成功です。

# %%
print("Steane 符号: X/Y/Z × 7 位置の単一エラーを訂正")
print(f"  {'誤り':4s} | {'位置':5s} | |0_L⟩ 符号語")
print(f"  {'-' * 4}-+-{'-' * 5}-+-{'-' * 12}")

for name, error_type in [("X", 1), ("Y", 2), ("Z", 3)]:
    for pos in range(7):
        exe = transpiler.transpile(
            steane_run,
            bindings={"error_type": error_type, "error_pos": pos},
        )
        result = exe.sample(_seeded_executor, shots=128).result()
        valid = sum(
            count for outcome, count in result.results if _is_steane_zero_word(outcome)
        )
        total = sum(count for _, count in result.results)
        print(f"  {name:4s} | q[{pos}]  | {valid / total:.3f}")

# %% [markdown]
# すべて 1.000 になれば、どの位置の $X$, $Y$, $Z$ エラーでも $\lvert0_L\rangle$ の符号空間へ戻せています。$Y=iXZ$ なので、$Y$ エラーでは $X$ 成分と $Z$ 成分の両方が検出され、両方の訂正が入ります。

# %% [markdown]
# ## 4. 横断的 Hadamard
#
# Steane 符号の重要な性質として、論理 Hadamard $\bar{H}$ を物理 Hadamard 7個で実装できます。
#
# $$
# \bar{H} = H^{\otimes 7}
# $$
#
# これは $X$ 型と $Z$ 型のスタビライザーが同じ Hamming パターンを持つためです。各物理量子ビットに独立にゲートを当てるだけなので、1つの物理ゲートの失敗が複数量子ビットへ広がりにくく、フォールトトレラント計算で重要です。


# %%
@qmc.qkernel
def transversal_hadamard_to_plus_l() -> qmc.Vector[qmc.Bit]:
    data = qmc.qubit_array(7, name="data")
    data = encode_steane_zero(data)

    for i in qmc.range(7):
        data[i] = qmc.h(data[i])

    for i in qmc.range(7):
        data[i] = qmc.h(data[i])

    return qmc.measure(data)


@qmc.qkernel
def transversal_hadamard_round_trip() -> qmc.Vector[qmc.Bit]:
    data = qmc.qubit_array(7, name="data")
    data = encode_steane_zero(data)

    for i in qmc.range(7):
        data[i] = qmc.h(data[i])
    for i in qmc.range(7):
        data[i] = qmc.h(data[i])

    return qmc.measure(data)


def _logical_x_parity(outcome) -> int:
    """Return the measured parity for logical X = X0 X1 X2."""
    bits = _bits7(outcome)
    return (bits[0] + bits[1] + bits[2]) % 2


# %% [markdown]
# まず、$\bar{H}\lvert0_L\rangle=\lvert+_L\rangle$ になっていることを確認します。$\lvert+_L\rangle$ は論理 $X$ の +1 固有状態なので、$X$ 基底で測ると $q_0 \oplus q_1 \oplus q_2 = 0$ になります。

# %%
print("横断的 H: |0_L⟩ -> |+_L⟩")
exe_plus = transpiler.transpile(transversal_hadamard_to_plus_l)
result_plus = exe_plus.sample(_seeded_executor, shots=1024).result()
parity_zero = sum(
    count for outcome, count in result_plus.results if _logical_x_parity(outcome) == 0
)
total_plus = sum(count for _, count in result_plus.results)
print(f"  q[0]⊕q[1]⊕q[2] = 0 の割合: {parity_zero / total_plus:.3f}")

# %% [markdown]
# 次に、横断的 $H$ を2回当てると $\bar{H}^2=I$ なので $\lvert0_L\rangle$ に戻ることを確認します。

# %%
print("横断的 H の round trip: |0_L⟩ -> H -> H -> |0_L⟩")
exe_round_trip = transpiler.transpile(transversal_hadamard_round_trip)
result_round_trip = exe_round_trip.sample(_seeded_executor, shots=1024).result()
valid = sum(
    count
    for outcome, count in result_round_trip.results
    if _is_steane_zero_word(outcome)
)
total = sum(count for _, count in result_round_trip.results)
print(f"  |0_L⟩ 符号語の割合: {valid / total:.3f}")

# %% [markdown]
# ## 5. まとめ
#
# 本章では Steane [[7,1,3]] 符号を実装しました。
#
# - Hamming [7,4,3] 符号から、3つの $X$ 型スタビライザーと3つの $Z$ 型スタビライザーを作った。
# - $Z$ 型スタビライザーで $X$ 成分を、$X$ 型スタビライザーで $Z$ 成分を検出した。
# - 21 通りの単一 Pauli エラー($X/Y/Z \times 7$ 位置)を訂正できることを確認した。
# - 物理 $H$ 7個が論理 Hadamard $\bar{H}$ として働くことを確認した。
#
# Steane 符号は Shor 符号より少ない物理量子ビットで同じ距離 $d=3$ を持ち、CSS 符号と横断的 Clifford ゲートの基本例になっています。
#
# ### 次へ
#
# - [量子誤り訂正(1)](quantum_error_correction.ipynb) — 3量子ビット bit-flip / phase-flip / Shor 符号
# - 表面符号 — 2D 格子上のローカルなスタビライザーと繰り返しシンドローム測定
