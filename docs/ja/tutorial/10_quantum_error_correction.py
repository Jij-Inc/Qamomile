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
# title: 量子誤り訂正入門
# tags: [qec, tutorial]
# ---
#
# # 量子誤り訂正入門
#
# <!-- BEGIN auto-tags -->
# **タグ:** <a class="tag-chip" href="../tags/qec.md">qec</a> <a class="tag-chip" href="../tags/tutorial.md">tutorial</a>
# <!-- END auto-tags -->
#
# 量子誤り訂正(Quantum Error Correction; QEC)は、壊れやすい量子状態を複数の物理量子ビットへ分散し、状態そのものを測らずにエラーだけを検出して戻す技術です。
#
# このチュートリアルでは、次の流れを Qamomile の `@qkernel` で実装します。
#
# 1. 3量子ビット bit-flip 符号で、単一の $X$ エラーを訂正する。
# 2. Hadamard 変換で phase-flip 符号を作り、単一の $Z$ エラーを訂正する。
# 3. 両者を組み合わせた Shor の9量子ビット符号で、単一の $X$, $Y$, $Z$ エラーを訂正する。
# 4. 最後にスタビライザー形式で、ここまでの回路を統一的に整理する。
#
# 主役は一貫して **シンドローム測定** です。論理状態を直接測るのではなく、補助量子ビット(ancilla)を使って「どこに、どの種類のエラーが起きたか」だけを取り出します。

# %%
# 最新のQamomileをpipからインストールします。
# # !pip install qamomile
# # or
# # !uv add qamomile

# %%
import math

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# Create a seeded backend for reproducible documentation output
from qiskit_aer import AerSimulator

_seeded_backend = AerSimulator(seed_simulator=42, max_parallel_threads=1)
_seeded_executor = transpiler.executor(backend=_seeded_backend)


def _first_bit_distribution(result) -> dict[int, int]:
    """Return counts for the first measured bit."""
    counts = {0: 0, 1: 0}
    for outcome, count in result.results:
        bit = outcome[0] if isinstance(outcome, (list, tuple)) else outcome & 1
        counts[bit] += count
    return counts


def _sample_first_bit(
    kernel,
    *,
    bindings: dict[str, object] | None = None,
    parameters: list[str] | None = None,
    runtime_bindings: dict[str, object] | None = None,
    shots: int = 256,
) -> dict[int, int]:
    """Compile a tutorial kernel and return the first-bit count distribution."""
    executable = transpiler.transpile(
        kernel,
        bindings=bindings or {},
        parameters=parameters or [],
    )
    job = executable.sample(
        _seeded_executor,
        shots=shots,
        bindings=runtime_bindings or {},
    )
    return _first_bit_distribution(job.result())


# %% [markdown]
# ## 1. 何を訂正したいのか
#
# 古典の繰り返し符号では、ビット $b$ を `bbb` と3回送れば、1ビット壊れても多数決で復元できます。量子でも似た発想を使いたくなりますが、そのままでは動きません。
#
# - 未知の量子状態 $\lvert\psi\rangle$ はコピーできない(複製禁止定理)。
# - 論理状態を直接測ると、重ね合わせが壊れる。
#
# 量子誤り訂正は、コピーの代わりに **エンタングルメント** で情報を分散し、直接測定の代わりに **シンドローム測定** でエラー情報だけを測ります。
#
# 標準的な手順は次の5段階です。
#
# ```text
# encode -> error -> syndrome measurement -> correction -> encoded state
# ```

# %% [markdown]
# ## 2. 3量子ビット bit-flip 符号
#
# まず、bit-flip エラー $X$ だけを訂正する符号から始めます。論理状態
#
# $$\alpha\lvert 0\rangle + \beta\lvert 1\rangle$$
#
# を3つの物理量子ビットへ
#
# $$\alpha\lvert 000\rangle + \beta\lvert 111\rangle$$
#
# として埋め込みます。CNOTで相関を作っているだけなので、未知状態のコピーではありません。


# %%
@qmc.qkernel
def encode_3qubit_bitflip(
    q0: qmc.Qubit, q1: qmc.Qubit, q2: qmc.Qubit
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    q0, q1 = qmc.cx(q0, q1)
    q0, q2 = qmc.cx(q0, q2)
    return q0, q1, q2


# %% [markdown]
# ### シンドローム
#
# bit-flip 符号では、次の2つの $Z$ パリティを測ります。
#
# - $S_0 = Z_0Z_1$
# - $S_1 = Z_0Z_2$
#
# 符号空間 $\{\lvert000\rangle, \lvert111\rangle\}$ では、どちらも「同じ値か」を見るだけです。単一の $X$ エラーが入ると、どのパリティが反転したかでエラー位置が分かります。
#
# | エラー | $(s_0, s_1)$ | 訂正 |
# | --- | --- | --- |
# | なし | $(0, 0)$ | なし |
# | $X_0$ | $(1, 1)$ | $X_0$ |
# | $X_1$ | $(1, 0)$ | $X_1$ |
# | $X_2$ | $(0, 1)$ | $X_2$ |
#
# $Z_iZ_j$ の測定は、ancilla を用意して `CX(data[i], anc); CX(data[j], anc); measure(anc)` と書けます。ancilla だけを測るので、論理状態そのものは直接読みません。


# %%
@qmc.qkernel
def bitflip_syndrome_run(
    error_pos: qmc.UInt,
    theta: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    data = qmc.qubit_array(3, name="data")
    anc = qmc.qubit_array(2, name="anc")

    data[0] = qmc.ry(data[0], theta)
    data[0], data[1], data[2] = encode_3qubit_bitflip(data[0], data[1], data[2])

    for i in qmc.range(3):
        if error_pos == i:
            data[i] = qmc.x(data[i])

    data[0], anc[0] = qmc.cx(data[0], anc[0])
    data[1], anc[0] = qmc.cx(data[1], anc[0])
    s0 = qmc.measure(anc[0])

    data[0], anc[1] = qmc.cx(data[0], anc[1])
    data[2], anc[1] = qmc.cx(data[2], anc[1])
    s1 = qmc.measure(anc[1])

    if s0 & s1:
        data[0] = qmc.x(data[0])
    if s0 & ~s1:
        data[1] = qmc.x(data[1])
    if ~s0 & s1:
        data[2] = qmc.x(data[2])

    return qmc.measure(data)


# %% [markdown]
# `error_pos` はコンパイル時パラメータです。`0`, `1`, `2` のいずれかならその位置に $X$ を注入し、`3` ならどの分岐にも入らないのでエラーなしとして扱います。
#
# まず論理 $\lvert1\rangle$ を準備し、どの位置に $X$ が入っても `data[0]` が1に戻ることを確認します。

# %%
bitflip_cases = [
    ("エラーなし", 3),
    ("X on data[0]", 0),
    ("X on data[1]", 1),
    ("X on data[2]", 2),
]

print("3量子ビット bit-flip 符号: 論理 |1⟩")
for label, error_pos in bitflip_cases:
    counts = _sample_first_bit(
        bitflip_syndrome_run,
        bindings={"error_pos": error_pos},
        parameters=["theta"],
        runtime_bindings={"theta": math.pi},
    )
    print(
        f"  {label:14s}: data[0]=0 が {counts[0]:3d} 回, data[0]=1 が {counts[1]:3d} 回"
    )

# %% [markdown]
# 重ね合わせでも同じです。$\theta=\pi/3$ で
#
# $$\cos(\pi/6)\lvert0\rangle + \sin(\pi/6)\lvert1\rangle$$
#
# を準備すると、理論上 $P(data[0]=1)=\sin^2(\pi/6)=0.25$ です。エラー位置によらずこの確率が保たれれば、振幅情報も壊れていないことが分かります。

# %%
print("3量子ビット bit-flip 符号: 重ね合わせ状態")
for label, error_pos in bitflip_cases:
    counts = _sample_first_bit(
        bitflip_syndrome_run,
        bindings={"error_pos": error_pos},
        parameters=["theta"],
        runtime_bindings={"theta": math.pi / 3},
        shots=2000,
    )
    total = counts[0] + counts[1]
    print(f"  {label:14s}: P(data[0]=1) = {counts[1] / total:.3f}")

# %% [markdown]
# ## 3. 3量子ビット phase-flip 符号
#
# phase-flip エラーは $Z$ で表されます。Hadamard 変換を挟むと
#
# $$H Z H = X,\qquad H X H = Z$$
#
# なので、$H$ 基底では phase-flip は bit-flip として見えます。したがって phase-flip 符号は、bit-flip 符号をエンコードした後に各量子ビットへ $H$ を当てれば作れます。
#
# 論理基底は
#
# - $\lvert0_L\rangle=\lvert+++\rangle$
# - $\lvert1_L\rangle=\lvert---\rangle$
#
# です。測るスタビライザーも $Z$ パリティから $X$ パリティへ入れ替わります。


# %%
@qmc.qkernel
def encode_3qubit_phaseflip(
    q0: qmc.Qubit, q1: qmc.Qubit, q2: qmc.Qubit
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    q0, q1, q2 = encode_3qubit_bitflip(q0, q1, q2)
    q0 = qmc.h(q0)
    q1 = qmc.h(q1)
    q2 = qmc.h(q2)
    return q0, q1, q2


# %% [markdown]
# $X_iX_j$ の測定では、ancilla を $\lvert+\rangle$ にしてから、ancilla を制御としてデータ量子ビットへ CNOT を当てます。最後に ancilla へもう一度 $H$ を当て、$Z$ 基底で測ります。
#
# | エラー | $(s_0, s_1)$ | 訂正 |
# | --- | --- | --- |
# | なし | $(0, 0)$ | なし |
# | $Z_0$ | $(1, 1)$ | $Z_0$ |
# | $Z_1$ | $(1, 0)$ | $Z_1$ |
# | $Z_2$ | $(0, 1)$ | $Z_2$ |


# %%
@qmc.qkernel
def phaseflip_syndrome_run(error_pos: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    data = qmc.qubit_array(3, name="data")
    anc = qmc.qubit_array(2, name="anc")

    data[0], data[1], data[2] = encode_3qubit_phaseflip(data[0], data[1], data[2])

    for i in qmc.range(3):
        if error_pos == i:
            data[i] = qmc.z(data[i])

    anc[0] = qmc.h(anc[0])
    anc[0], data[0] = qmc.cx(anc[0], data[0])
    anc[0], data[1] = qmc.cx(anc[0], data[1])
    anc[0] = qmc.h(anc[0])
    s0 = qmc.measure(anc[0])

    anc[1] = qmc.h(anc[1])
    anc[1], data[0] = qmc.cx(anc[1], data[0])
    anc[1], data[2] = qmc.cx(anc[1], data[2])
    anc[1] = qmc.h(anc[1])
    s1 = qmc.measure(anc[1])

    if s0 & s1:
        data[0] = qmc.z(data[0])
    if s0 & ~s1:
        data[1] = qmc.z(data[1])
    if ~s0 & s1:
        data[2] = qmc.z(data[2])

    data[0] = qmc.h(data[0])
    return qmc.measure(data)


# %% [markdown]
# ここでは論理 $\lvert0_L\rangle=\lvert+++\rangle$ を使います。訂正後の `data[0]` は $\lvert+\rangle$ に戻るので、最後に $H$ を当ててから測れば常に0になります。

# %%
phaseflip_cases = [
    ("エラーなし", 3),
    ("Z on data[0]", 0),
    ("Z on data[1]", 1),
    ("Z on data[2]", 2),
]

print("3量子ビット phase-flip 符号: 論理 |0_L⟩ = |+++⟩")
for label, error_pos in phaseflip_cases:
    counts = _sample_first_bit(
        phaseflip_syndrome_run,
        bindings={"error_pos": error_pos},
    )
    print(
        f"  {label:14s}: data[0]=0 が {counts[0]:3d} 回, data[0]=1 が {counts[1]:3d} 回"
    )

# %% [markdown]
# ## 4. Shor の9量子ビット符号
#
# bit-flip 符号は $X$ エラーだけ、phase-flip 符号は $Z$ エラーだけを訂正します。任意の単一量子ビット Pauli エラーを直すには、両方を組み合わせます。
#
# Shor 符号は次の連結符号です。
#
# 1. 1量子ビットを3量子ビット phase-flip 符号でエンコードする。
# 2. その3つの量子ビットを、それぞれ3量子ビット bit-flip 符号でエンコードする。
#
# 9量子ビットを3つのブロック
#
# ```text
# (q0, q1, q2), (q3, q4, q5), (q6, q7, q8)
# ```
#
# と見ます。各ブロック内の $X$ エラーは $Z$ パリティで検出し、ブロック間の $Z$ エラーは $X$ パリティで検出します。$Y=iXZ$ は両方のシンドロームが立つので、$X$ 訂正と $Z$ 訂正を独立に行えば直せます。


# %%
@qmc.qkernel
def encode_shor(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    q[0], q[3], q[6] = encode_3qubit_phaseflip(q[0], q[3], q[6])

    q[0], q[1], q[2] = encode_3qubit_bitflip(q[0], q[1], q[2])
    q[3], q[4], q[5] = encode_3qubit_bitflip(q[3], q[4], q[5])
    q[6], q[7], q[8] = encode_3qubit_bitflip(q[6], q[7], q[8])
    return q


# %% [markdown]
# Shor 符号のシンドロームは8ビットです。
#
# - `anc[0..5]`: 各ブロック内の $Z$ パリティ。どの物理量子ビットに $X$ 成分があるかを検出する。
# - `anc[6..7]`: ブロック間の $X$ パリティ。どのブロックに $Z$ 成分があるかを検出する。
#
# 訂正後の状態はまだ9量子ビットにエンコードされています。動作確認では読みやすさのため、最後にエンコーダの逆を当てて `q[0]` に論理ビットを戻してから測定します。これは検証用の後処理であり、誤り訂正そのものはシンドローム測定と古典フィードバックで完結しています。


# %%
@qmc.qkernel
def shor_syndrome_run(
    error_type: qmc.UInt,
    error_pos: qmc.UInt,
    theta: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(9, name="q")
    anc = qmc.qubit_array(8, name="anc")

    q[0] = qmc.ry(q[0], theta)
    q = encode_shor(q)

    for i in qmc.range(9):
        if (error_type == 1) & (error_pos == i):  # 1 means X error.
            q[i] = qmc.x(q[i])
        if (error_type == 2) & (error_pos == i):  # 2 means Y error.
            q[i] = qmc.y(q[i])
        if (error_type == 3) & (error_pos == i):  # 3 means Z error.
            q[i] = qmc.z(q[i])

    # Block 0: Z0Z1, Z0Z2
    q[0], anc[0] = qmc.cx(q[0], anc[0])
    q[1], anc[0] = qmc.cx(q[1], anc[0])
    b0_s0 = qmc.measure(anc[0])
    q[0], anc[1] = qmc.cx(q[0], anc[1])
    q[2], anc[1] = qmc.cx(q[2], anc[1])
    b0_s1 = qmc.measure(anc[1])

    # Block 1: Z3Z4, Z3Z5
    q[3], anc[2] = qmc.cx(q[3], anc[2])
    q[4], anc[2] = qmc.cx(q[4], anc[2])
    b1_s0 = qmc.measure(anc[2])
    q[3], anc[3] = qmc.cx(q[3], anc[3])
    q[5], anc[3] = qmc.cx(q[5], anc[3])
    b1_s1 = qmc.measure(anc[3])

    # Block 2: Z6Z7, Z6Z8
    q[6], anc[4] = qmc.cx(q[6], anc[4])
    q[7], anc[4] = qmc.cx(q[7], anc[4])
    b2_s0 = qmc.measure(anc[4])
    q[6], anc[5] = qmc.cx(q[6], anc[5])
    q[8], anc[5] = qmc.cx(q[8], anc[5])
    b2_s1 = qmc.measure(anc[5])

    # X0X1X2X3X4X5
    anc[6] = qmc.h(anc[6])
    anc[6], q[0] = qmc.cx(anc[6], q[0])
    anc[6], q[1] = qmc.cx(anc[6], q[1])
    anc[6], q[2] = qmc.cx(anc[6], q[2])
    anc[6], q[3] = qmc.cx(anc[6], q[3])
    anc[6], q[4] = qmc.cx(anc[6], q[4])
    anc[6], q[5] = qmc.cx(anc[6], q[5])
    anc[6] = qmc.h(anc[6])
    phase_s0 = qmc.measure(anc[6])

    # X3X4X5X6X7X8
    anc[7] = qmc.h(anc[7])
    anc[7], q[3] = qmc.cx(anc[7], q[3])
    anc[7], q[4] = qmc.cx(anc[7], q[4])
    anc[7], q[5] = qmc.cx(anc[7], q[5])
    anc[7], q[6] = qmc.cx(anc[7], q[6])
    anc[7], q[7] = qmc.cx(anc[7], q[7])
    anc[7], q[8] = qmc.cx(anc[7], q[8])
    anc[7] = qmc.h(anc[7])
    phase_s1 = qmc.measure(anc[7])

    if b0_s0 & b0_s1:
        q[0] = qmc.x(q[0])
    if b0_s0 & ~b0_s1:
        q[1] = qmc.x(q[1])
    if ~b0_s0 & b0_s1:
        q[2] = qmc.x(q[2])

    if b1_s0 & b1_s1:
        q[3] = qmc.x(q[3])
    if b1_s0 & ~b1_s1:
        q[4] = qmc.x(q[4])
    if ~b1_s0 & b1_s1:
        q[5] = qmc.x(q[5])

    if b2_s0 & b2_s1:
        q[6] = qmc.x(q[6])
    if b2_s0 & ~b2_s1:
        q[7] = qmc.x(q[7])
    if ~b2_s0 & b2_s1:
        q[8] = qmc.x(q[8])

    if phase_s0 & ~phase_s1:
        q[0] = qmc.z(q[0])
    if phase_s0 & phase_s1:
        q[3] = qmc.z(q[3])
    if ~phase_s0 & phase_s1:
        q[6] = qmc.z(q[6])

    q[0], q[1] = qmc.cx(q[0], q[1])
    q[0], q[2] = qmc.cx(q[0], q[2])
    q[3], q[4] = qmc.cx(q[3], q[4])
    q[3], q[5] = qmc.cx(q[3], q[5])
    q[6], q[7] = qmc.cx(q[6], q[7])
    q[6], q[8] = qmc.cx(q[6], q[8])

    q[0] = qmc.h(q[0])
    q[3] = qmc.h(q[3])
    q[6] = qmc.h(q[6])
    q[0], q[3] = qmc.cx(q[0], q[3])
    q[0], q[6] = qmc.cx(q[0], q[6])

    return qmc.measure(q)


# %% [markdown]
# 代表として、3つのブロックから1つずつ位置を選び、$X$, $Y$, $Z$ エラーを注入します。論理 $\lvert1\rangle$ が保たれていれば、すべて `q[0]=1` になります。

# %%
shor_cases = [
    ("X", 1, 0),
    ("Y", 2, 4),
    ("Z", 3, 8),
]

print("Shor 9量子ビット符号: 論理 |1⟩")
print(f"  {'エラー':6s} | {'位置':5s} | P(q[0]=1)")
print(f"  {'-' * 6}-+-{'-' * 5}-+-{'-' * 9}")
for name, error_type, error_pos in shor_cases:
    counts = _sample_first_bit(
        shor_syndrome_run,
        bindings={"error_type": error_type, "error_pos": error_pos},
        parameters=["theta"],
        runtime_bindings={"theta": math.pi},
    )
    total = counts[0] + counts[1]
    print(f"  {name:6s} | q[{error_pos}]  | {counts[1] / total:.3f}")

# %% [markdown]
# Shor 符号は、単一量子ビットの任意の Pauli エラーを訂正できます。一般の小さなノイズも Pauli エラーの線形結合として扱えるため、まず Pauli エラーを直せることが量子誤り訂正の出発点になります。

# %% [markdown]
# ## 5. スタビライザー形式で見る
#
# ここまでの符号は、どれもスタビライザーで表せます。スタビライザーとは、符号空間の状態を変えない Pauli 演算子の集合です。測定結果が $+1$ なら符号空間内、$-1$ ならエラーによって外へ出たことを意味します。
#
# | 符号 | スタビライザー生成子 | 訂正できる単一エラー |
# | --- | --- | --- |
# | 3量子ビット bit-flip | $Z_0Z_1$, $Z_0Z_2$ | $X$ |
# | 3量子ビット phase-flip | $X_0X_1$, $X_0X_2$ | $Z$ |
# | Shor 9量子ビット | $Z_0Z_1$, $Z_0Z_2$, $Z_3Z_4$, $Z_3Z_5$, $Z_6Z_7$, $Z_6Z_8$, $X_0X_1X_2X_3X_4X_5$, $X_3X_4X_5X_6X_7X_8$ | $X$, $Y$, $Z$ |
#
# 表面符号や Steane 符号でも基本は同じです。スタビライザーごとに ancilla を用意し、パリティを測り、シンドロームから訂正 Pauli を決めます。

# %% [markdown]
# ## 6. まとめ
#
# このチュートリアルでは、量子誤り訂正を次の順に実装しました。
#
# - 3量子ビット bit-flip 符号で $Z$ パリティを測り、単一 $X$ エラーを訂正した。
# - Hadamard 変換で phase-flip 符号を作り、$X$ パリティを測って単一 $Z$ エラーを訂正した。
# - Shor 符号でブロック内の $X$ 成分とブロック間の $Z$ 成分を別々に検出し、単一 $X$, $Y$, $Z$ エラーを訂正した。
# - それらをスタビライザー形式で統一的に見直した。
#
# 次のステップとしては、[Steane [[7,1,3]] 符号と CSS 構成](11_steane_code.ipynb) が自然です。CSS 符号では、$X$ 型と $Z$ 型のスタビライザーをより体系的に組み合わせます。
