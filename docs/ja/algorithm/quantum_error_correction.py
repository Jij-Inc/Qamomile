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
# tags: [algorithm, error-correction]
# ---
#
# # 量子誤り訂正入門
#
# 量子ビットはノイズで簡単に状態が乱れます。**量子誤り訂正(Quantum Error Correction; QEC)** は、1つの論理量子ビットを複数の物理量子ビットに分散させて情報を守る技術です。
#
# この記事は QEC 入門の前編です。次の3つの符号を Qamomile の `@qkernel` で実装します。
#
# 1. **3量子ビット bit-flip 符号** ― 単一のビット反転($X$)エラーを訂正する。
# 2. **3量子ビット phase-flip 符号** ― 単一の位相反転($Z$)エラーを訂正する。
# 3. **Shor の9量子ビット符号** ― 任意の単一量子ビットエラー($X$, $Y$, $Z$)を訂正する。
#
# 後編の [スタビライザ形式論と Steane 符号](steane_code.ipynb) では、これらを統一的に扱う枠組みを説明します。
#
# **前提知識**: `@qkernel`、CNOT ゲート、測定。未習の場合は [最初の量子カーネル](../tutorial/01_your_first_quantum_kernel.ipynb) を先にご覧ください。

# %%
# 最新のQamomileをpipからインストールします。
# # !pip install qamomile
# # or
# # !uv add qamomile

# %% [markdown]
# ## 1. なぜ量子の誤り訂正は難しいのか
#
# 実際の量子誤り訂正に入る前に、量子の場合は古典と何が違って難しいのかを説明します。

# %% [markdown]
# ### 1.1 古典の反復符号
#
# 古典コンピュータでビット $b$ をノイズから守る、いちばんシンプルな方法が **反復符号** です。
#
# ```text
# 符号化:  0 → 000     1 → 111
# 計算:    符号化したまま処理する
# 復元:    最後に3ビットを読み、多数決をとる
# ```
#
# 途中で1ビットが反転しても(`000 → 010` など)、多数決で元の値に戻せます。
#

# %% [markdown]
# ### 1.2 符号化は量子でもそのまま作れる
#
# 同じ発想を量子に持ち込みます。論理状態
#
# $$\alpha\lvert0\rangle + \beta\lvert1\rangle$$
#
# を3量子ビットに広げます。CNOT ゲートを2つ使うと、次の状態が作れます。
#
# $$\alpha\lvert000\rangle + \beta\lvert111\rangle$$
#
# これは状態を3つ複製したもの($(\alpha\lvert0\rangle+\beta\lvert1\rangle)^{\otimes 3}$)ではなく、3量子ビットにまたがる **エンタングルメント(量子もつれ)** です。符号化はこのように問題なく作れます。

# %% [markdown]
# ### 1.3 本当の壁は「訂正のしかた」
#
# 詰まるのは符号化ではなく **訂正** です。
#
# 古典の反復符号は、訂正のときに3ビットを **読んで** 多数決をとります。古典のビットは読んでも壊れません。
#
# 量子で「読む」ことは **測定** であり、測定は重ね合わせを壊します。$\alpha\lvert000\rangle+\beta\lvert111\rangle$ をそのまま測れば $\lvert000\rangle$ か $\lvert111\rangle$ に確定し、$\alpha$ と $\beta$ は失われます。
#
# 計算の最後の読み出しなら、古典と同じく「全部測って多数決」で構いません。困るのは計算の途中で訂正したいとき ― 続きの計算のために状態を保ったまま、エラーだけを直したい場面です。
#
# そこで量子誤り訂正は、データそのものを測らず、**「どこに、どんなエラーが起きたか」だけ** を測ります。このエラーの位置と種類を表す情報を **シンドローム(syndrome)** と呼びます。
# シンドロームはもともと医学用語で「症候群」、つまり複数の症状の組み合わせを指す言葉です。医師が病気そのものを直接見ずに症状から診断するように、エラー自体を測らず、その「症状」だけからエラーの正体を突き止める ― この呼び名はそこにちなんでいます。
#
# シンドロームは補助量子ビット(ancilla)を経由して取り出すので、論理状態($\alpha,\beta$)には触れません。この操作が **シンドローム測定** です。

# %% [markdown]
# ### 1.4 もう1つの違い:位相エラー
#
# 古典と量子には、もう1つ違いがあります。
#
# 古典のエラーはビット反転($0 \leftrightarrow 1$)だけです。量子ビットには **位相エラー** があります。$\lvert1\rangle$ の符号だけが反転する
#
# $$\alpha\lvert0\rangle + \beta\lvert1\rangle \;\longrightarrow\; \alpha\lvert0\rangle - \beta\lvert1\rangle$$
#
# というエラーは、計算基底で測っても確率が変わらず、多数決では検出できません。
#
# さらに量子のエラーは **連続的** で、ビットがわずかに回転するような中間的なエラーも無数にありえます。これを1つずつ訂正するのは無理に見えますが、実は離散的なエラーだけ考えれば十分です。
#
# ここでは簡単に2段階の説明をしておきます。
# まず、1量子ビットへのどんなエラーも、行列としては $I$(何もしない)・$X$・$Y$・$Z$ の線形結合で書けます。
# たとえば微小回転は $I$ と $X$ が少しずつ混ざったものです。このエラーを符号化された状態に作用させると、状態は「エラーなし」「$X$ エラー」「$Z$ エラー」… が重ね合わさったものになります。
#
# 次に、この状態のシンドロームを測ると、重ね合わさっていたケースのうち1つに **収縮** します。測定後に残るのは「量子ビット $i$ に $X$ エラー」のような離散的なエラーが1つだけです。連続的なエラーが、シンドローム測定によって離散的な **Pauli エラー** に変わるのです(この収縮は次節以降で実際に確認します)。
#
# そのため訂正すべきは完全な $X$・$Y$・$Z$ だけです。しかも $Y=iXZ$ なので、$X$ と $Z$ さえ訂正できれば $Y$ も自動的にカバーされます。

# %% [markdown]
# ### 1.5 量子誤り訂正の流れ
#
# 量子誤り訂正は次の流れで進みます。
#
# ```text
# 符号化  →  エラー発生  →  シンドローム測定  →  訂正
# ```
#
# - **符号化**: 1つの論理量子ビットを、もつれを使って複数の物理量子ビットに広げる。
# - **シンドローム測定**: 論理状態を測らず、エラーの位置と種類だけを補助量子ビットに取り出す。
# - **訂正**: シンドロームに応じて Pauli ゲートを当て、エラーを打ち消す。
#
# 次節からは、いちばんシンプルな3量子ビット bit-flip 符号で、この流れを実装していきます。

# %% [markdown]
# 実装に入る前に、Qamomile と Qiskit バックエンドを読み込み、補助関数を2つ用意します。`_first_bit_distribution` と `_sample_first_bit` は、カーネルをコンパイル・実行して先頭ビットの 0/1 集計を返すだけのユーティリティです。QEC の本筋ではないので、読み飛ばして構いません。

# %%
import math
import os

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.job import SampleResult
from qamomile.qiskit import QiskitTranspiler

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
default_shots = 64 if docs_test_mode else 256
superposition_shots = 512 if docs_test_mode else 2000

transpiler = QiskitTranspiler()

# ドキュメントの出力を再現可能にするため、シード付きのバックエンドを用意します。
from qiskit_aer import AerSimulator

_seeded_backend = AerSimulator(seed_simulator=42, max_parallel_threads=1)
_seeded_executor = transpiler.executor(backend=_seeded_backend)


def _first_bit_distribution(result: SampleResult) -> dict[int, int]:
    """先頭の測定ビットについて、0 と 1 の出現回数を返す。"""
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
    shots: int = default_shots,
) -> dict[int, int]:
    """カーネルをコンパイル・実行し、先頭ビットの 0/1 出現回数を返す。"""
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
# ## 2. 3量子ビット bit-flip 符号
#
# 最初に作るのは **bit-flip 符号** です。ビット反転($X$)エラー1つだけを訂正する、いちばんシンプルな量子誤り訂正符号です。1.5 で見た流れ ― 符号化 → エラー → シンドローム測定 → 訂正 ― を、この符号で一通り実装します。

# %% [markdown]
# ### 2.1 狙うエラー:ビット反転
#
# $X$ エラーは量子ビットを $\lvert0\rangle\leftrightarrow\lvert1\rangle$ と反転させます。古典のビット反転にあたるエラーです。bit-flip 符号は、この単一の $X$ エラーを訂正することを目標にします。

# %% [markdown]
# ### 2.2 符号空間
#
# 論理状態 $\alpha\lvert0\rangle+\beta\lvert1\rangle$ を、3量子ビットの
#
# $$\alpha\lvert000\rangle + \beta\lvert111\rangle$$
#
# に符号化します。論理 $\lvert0\rangle$ は $\lvert000\rangle$、論理 $\lvert1\rangle$ は $\lvert111\rangle$ に対応します。この2つが張る空間を **符号空間(code space)** と呼び、正しい符号語は $\lvert000\rangle$ と $\lvert111\rangle$ の2つだけです。

# %% [markdown]
# ### 2.3 符号化回路
#
# 符号化は、データ量子ビット $q_0$ から $q_1$, $q_2$ へ CNOT を1つずつかけるだけです。


# %%
@qmc.qkernel
def encode_3qubit_bitflip(
    q0: qmc.Qubit, q1: qmc.Qubit, q2: qmc.Qubit
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    q0, q1 = qmc.cx(q0, q1)
    q0, q2 = qmc.cx(q0, q2)
    return q0, q1, q2


# %% [markdown]
# $q_0$ が $\lvert1\rangle$ なら、2つの CNOT が $q_1$, $q_2$ も反転させて $\lvert111\rangle$ になります。$q_0$ が $\lvert0\rangle$ なら何も起こらず $\lvert000\rangle$ のままです。$q_0$ が重ね合わせ $\alpha\lvert0\rangle+\beta\lvert1\rangle$ なら、結果は $\alpha\lvert000\rangle+\beta\lvert111\rangle$ になります。

# %% [markdown]
# ### 2.4 シンドローム測定:$Z$ パリティ
#
# エラーの位置を知るために、2つの **パリティ**(2量子ビットが同じ値かどうか)を測ります。
#
# - $S_0 = Z_0Z_1$ ― $q_0$ と $q_1$ が同じ値か
# - $S_1 = Z_0Z_2$ ― $q_0$ と $q_2$ が同じ値か
#
# 符号空間($\lvert000\rangle$ と $\lvert111\rangle$)では3量子ビットはすべて同じ値なので、どちらのパリティも「等しい」を返します。$X$ エラーが1つ入ると、その位置に応じてパリティのパターンが変わります。
#
# | エラー | シンドローム $(s_0, s_1)$ | 訂正 |
# | --- | --- | --- |
# | なし | $(0, 0)$ | なし |
# | $X_0$ | $(1, 1)$ | $X_0$ |
# | $X_1$ | $(1, 0)$ | $X_1$ |
# | $X_2$ | $(0, 1)$ | $X_2$ |
#
# 3つの $X$ エラーはそれぞれ異なるシンドロームを示すので、シンドロームからエラーの位置が一意に決まります。
#
# $Z_iZ_j$ を測るには、補助量子ビットを1つ用意し、`CX(data[i], anc)` と `CX(data[j], anc)` をかけてから `anc` を測定します。測るのは補助量子ビットだけで、データ量子ビットには触れません。
#
# 具体例で追ってみましょう。符号化された状態 $a\lvert000\rangle+b\lvert111\rangle$ の2番目の量子ビット $q_1$ に $X$ エラーが入ったとします。$X$ は $q_1$ を反転させるので、状態は
#
# $$a\lvert010\rangle + b\lvert101\rangle$$
#
# になります。ここに $\lvert00\rangle$ で初期化した補助量子ビットを2つ加え、シンドローム測定を行います。`CX` は制御量子ビットの値を補助量子ビットへ XOR で足し込むので、
#
# - $S_0=Z_0Z_1$:補助量子ビット1に $q_0\oplus q_1$ が入る。$\lvert010\rangle$ では $0\oplus1=1$、$\lvert101\rangle$ では $1\oplus0=1$。
# - $S_1=Z_0Z_2$:補助量子ビット2に $q_0\oplus q_2$ が入る。$\lvert010\rangle$ では $0\oplus0=0$、$\lvert101\rangle$ では $1\oplus1=0$。
#
# どちらの項でも補助量子ビットの値は同じ $(1,0)$ になり、状態は次のように変化します。
#
# $$(a\lvert010\rangle + b\lvert101\rangle)\lvert00\rangle \;\longrightarrow\; (a\lvert010\rangle + b\lvert101\rangle)\lvert10\rangle$$
#
# 重ね合わせの2つの項が **同じ** 補助量子ビットの値を与える点が重要です。そのため補助量子ビットを測定しても、$a$ と $b$ の重ね合わせは壊れません。
# 測定で得られるのはシンドローム $(s_0,s_1)=(1,0)$ だけで、表を見るとこれは $q_1$ の $X$ エラーを指しています。

# %% [markdown]
# ### 2.5 訂正
#
# シンドロームが分かれば、訂正は検出したエラーと同じ $X$ を同じ位置にもう一度かけるだけです。$X$ は2回かけると元に戻る($X^2=I$)ので、エラーの $X$ と訂正の $X$ が打ち消し合います。
#
# さきほどの例を続けます。シンドローム $(1,0)$ は $q_1$ の $X$ エラーを表すので、$q_1$ に $X$ をかけると、
#
# $$a\lvert010\rangle + b\lvert101\rangle \;\longrightarrow\; a\lvert000\rangle + b\lvert111\rangle$$
#
# と、符号化された状態がそのまま復元されます。上の表の「訂正」列が、各シンドロームに対応する訂正です。

# %% [markdown]
# ### 2.6 実装と実行:論理 $\lvert1\rangle$
#
# ここまでを1つの `@qkernel` にまとめます。`error_pos` でエラーを入れる位置を指定し、符号化・エラー注入・シンドローム測定・訂正までを行います。


# %%
@qmc.qkernel
def bitflip_syndrome_run(
    error_pos: qmc.UInt,
    theta: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    # データ用に3量子ビット、シンドローム測定用に補助量子ビットを2つ確保する。
    data = qmc.qubit_array(3, name="data")
    anc = qmc.qubit_array(2, name="anc")

    # 論理状態を ry(theta) で用意し、3量子ビットに符号化する。
    data[0] = qmc.ry(data[0], theta)
    data[0], data[1], data[2] = encode_3qubit_bitflip(data[0], data[1], data[2])

    # error_pos で指定した位置に X エラーを注入する(error_pos=3 ならエラーなし)。
    for i in qmc.range(3):
        if error_pos == i:
            data[i] = qmc.x(data[i])

    # シンドローム測定 1: Z0 Z1 パリティを anc[0] に取り出す。
    data[0], anc[0] = qmc.cx(data[0], anc[0])
    data[1], anc[0] = qmc.cx(data[1], anc[0])
    s0 = qmc.measure(anc[0])

    # シンドローム測定 2: Z0 Z2 パリティを anc[1] に取り出す。
    data[0], anc[1] = qmc.cx(data[0], anc[1])
    data[2], anc[1] = qmc.cx(data[2], anc[1])
    s1 = qmc.measure(anc[1])

    # シンドローム (s0, s1) からエラー位置を特定し、同じ位置に X をかけて訂正する。
    if s0 & s1:  # (1, 1) -> data[0]
        data[0] = qmc.x(data[0])
    if s0 & ~s1:  # (1, 0) -> data[1]
        data[1] = qmc.x(data[1])
    if ~s0 & s1:  # (0, 1) -> data[2]
        data[2] = qmc.x(data[2])

    return qmc.measure(data)


# %% [markdown]
# `error_pos` はコンパイル時に決まるパラメータです。値 `0`, `1`, `2` はその位置に $X$ エラーを注入します。値 `3` はどの分岐にも一致しないので、「エラーなし」を意味します。
#
# まず論理 $\lvert1\rangle$ を用意します(`theta` $=\pi$ の `ry` ゲート)。訂正が正しく働けば、`data[0]` は常に 1 になるはずです。

# %%
bitflip_cases = [
    ("no error", 3),
    ("X on data[0]", 0),
    ("X on data[1]", 1),
    ("X on data[2]", 2),
]
if docs_test_mode:
    bitflip_cases = [
        ("no error", 3),
        ("X on data[1]", 1),
    ]

print("3-qubit bit-flip code: logical |1>")
for label, error_pos in bitflip_cases:
    counts = _sample_first_bit(
        bitflip_syndrome_run,
        bindings={"error_pos": error_pos},
        parameters=["theta"],
        runtime_bindings={"theta": math.pi},
    )
    print(f"  {label:14s}: data[0]=0 -> {counts[0]:3d}, data[0]=1 -> {counts[1]:3d}")
    # 純粋な |1> 入力に対する単一 X 訂正は完全に決定的で、各ショットが
    # data[0] = 1 を返す。
    assert counts[0] == 0
    assert counts[1] == counts[0] + counts[1]

# %% [markdown]
# ### 2.7 重ね合わせ入力
#
# 符号は振幅も保ちます。`theta` $=\pi/3$ で用意した状態は
#
# $$P(\text{data}[0]=1)=\sin^2(\pi/6)=0.25$$
#
# の確率を持ちます。注入したエラーによらず、この確率が保たれるはずです。

# %%
print("3-qubit bit-flip code: superposition input")
for label, error_pos in bitflip_cases:
    counts = _sample_first_bit(
        bitflip_syndrome_run,
        bindings={"error_pos": error_pos},
        parameters=["theta"],
        runtime_bindings={"theta": math.pi / 3},
        shots=superposition_shots,
    )
    total = counts[0] + counts[1]
    print(f"  {label:14s}: P(data[0]=1) = {counts[1] / total:.3f}")
    assert abs(counts[1] / total - 0.25) < (0.08 if docs_test_mode else 0.05)
    assert total == superposition_shots

# %% [markdown]
# ### 2.8 限界:位相エラーには無力
#
# bit-flip 符号が訂正できるのは $X$ エラーだけです。位相エラー $Z$ には無力です。
#
# 理由は $Z$ パリティの測り方にあります。$Z$ エラーは $\lvert000\rangle$ や $\lvert111\rangle$ の符号を変えるだけで、ビットの値は変えません。$Z_iZ_j$ パリティはビットが等しいかどうかしか見ないので、$Z$ エラーが入ってもシンドロームは $(0,0)$ のまま ― エラーを検出できないのです。
#
# $Z$ エラーを訂正するには、別の符号が必要です。次節では、Hadamard ゲートを使って bit-flip 符号を「位相の世界」に移し替えた **phase-flip 符号** を作ります。

# %% [markdown]
# ## 3. 3量子ビット phase-flip 符号
#
# bit-flip 符号は $X$ エラーしか訂正できませんでした。次は位相エラー $Z$ を訂正する符号を作ります。一から設計し直すのではなく、bit-flip 符号を「基底を変えて」流用します。

# %% [markdown]
# ### 3.1 狙うエラー:位相反転
#
# $Z$ エラーは $\lvert1\rangle$ の符号だけを反転させます($\lvert0\rangle\to\lvert0\rangle$、$\lvert1\rangle\to-\lvert1\rangle$)。1.4 で見たとおり、計算基底で測っても見えないエラーです。phase-flip 符号は、この単一の $Z$ エラーを訂正することを目標にします。

# %% [markdown]
# ### 3.2 鍵となる等式:$H$ が $X$ と $Z$ を入れ替える
#
# Hadamard ゲート $H$ は、次の関係を満たします。
#
# $$HZH = X, \qquad HXH = Z$$
#
# つまり $H$ で挟むと、$X$ エラーと $Z$ エラーが入れ替わります。bit-flip 符号は $X$ エラーを訂正できました。各量子ビットを $H$ で基底変換すれば、その符号はそのまま $Z$ エラーを訂正する符号になります。これが phase-flip 符号です。

# %% [markdown]
# ### 3.3 符号空間
#
# $H\lvert0\rangle=\lvert+\rangle$、$H\lvert1\rangle=\lvert-\rangle$ なので、bit-flip 符号の符号語 $\lvert000\rangle$、$\lvert111\rangle$ を3量子ビットすべての $H$ で変換すると、phase-flip 符号の論理状態が得られます。
#
# - $\lvert0_L\rangle = \lvert+++\rangle$
# - $\lvert1_L\rangle = \lvert---\rangle$
#
# bit-flip 符号で $X$ エラーが $\lvert0\rangle\leftrightarrow\lvert1\rangle$ を入れ替えたのと同じように、phase-flip 符号では $Z$ エラーが $\lvert+\rangle\leftrightarrow\lvert-\rangle$ を入れ替えます($Z\lvert+\rangle=\lvert-\rangle$)。

# %% [markdown]
# ### 3.4 符号化回路
#
# 符号化は、bit-flip 符号の符号化に続けて、3量子ビットすべてに $H$ をかけるだけです。


# %%
@qmc.qkernel
def encode_3qubit_phaseflip(
    q0: qmc.Qubit, q1: qmc.Qubit, q2: qmc.Qubit
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    # bit-flip 符号で符号化したあと、3量子ビットすべてを H で X 基底に移す。
    q0, q1, q2 = encode_3qubit_bitflip(q0, q1, q2)
    q0 = qmc.h(q0)
    q1 = qmc.h(q1)
    q2 = qmc.h(q2)
    return q0, q1, q2


# %% [markdown]
# ### 3.5 シンドローム測定:$X$ パリティ
#
# bit-flip 符号では $Z$ パリティ $Z_iZ_j$ を測りました。phase-flip 符号では $X$ と $Z$ が入れ替わっているので、$X$ パリティ $X_iX_j$ を測ります。
#
# $X_iX_j$ を測るには、補助量子ビットを $\lvert+\rangle$ に用意し($H$ をかける)、それを制御として2つのデータ量子ビットへ CNOT し、再び $H$ をかけてから測定します。シンドロームとエラーの対応は bit-flip 符号と同じ形で、訂正が $X$ から $Z$ に変わるだけです。
#
# | エラー | シンドローム $(s_0, s_1)$ | 訂正 |
# | --- | --- | --- |
# | なし | $(0, 0)$ | なし |
# | $Z_0$ | $(1, 1)$ | $Z_0$ |
# | $Z_1$ | $(1, 0)$ | $Z_1$ |
# | $Z_2$ | $(0, 1)$ | $Z_2$ |

# %% [markdown]
# ### 3.6 実装と実行
#
# ここまでを1つの `@qkernel` にまとめます。今回は論理 $\lvert0_L\rangle=\lvert+++\rangle$ を用意します。


# %%
@qmc.qkernel
def phaseflip_syndrome_run(error_pos: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    # データ用に3量子ビット、シンドローム測定用に補助量子ビットを2つ確保する。
    data = qmc.qubit_array(3, name="data")
    anc = qmc.qubit_array(2, name="anc")

    # 論理 |0_L> = |+++> に符号化する。
    data[0], data[1], data[2] = encode_3qubit_phaseflip(data[0], data[1], data[2])

    # error_pos で指定した位置に Z エラーを注入する(error_pos=3 ならエラーなし)。
    for i in qmc.range(3):
        if error_pos == i:
            data[i] = qmc.z(data[i])

    # シンドローム測定 1: X0 X1 パリティを anc[0] に取り出す(H で X 基底に変換)。
    anc[0] = qmc.h(anc[0])
    anc[0], data[0] = qmc.cx(anc[0], data[0])
    anc[0], data[1] = qmc.cx(anc[0], data[1])
    anc[0] = qmc.h(anc[0])
    s0 = qmc.measure(anc[0])

    # シンドローム測定 2: X0 X2 パリティを anc[1] に取り出す。
    anc[1] = qmc.h(anc[1])
    anc[1], data[0] = qmc.cx(anc[1], data[0])
    anc[1], data[2] = qmc.cx(anc[1], data[2])
    anc[1] = qmc.h(anc[1])
    s1 = qmc.measure(anc[1])

    # シンドローム (s0, s1) からエラー位置を特定し、同じ位置に Z をかけて訂正する。
    if s0 & s1:  # (1, 1) -> data[0]
        data[0] = qmc.z(data[0])
    if s0 & ~s1:  # (1, 0) -> data[1]
        data[1] = qmc.z(data[1])
    if ~s0 & s1:  # (0, 1) -> data[2]
        data[2] = qmc.z(data[2])

    # data[0] は |+> に戻っているので、H で |0> に直してから測定する。
    data[0] = qmc.h(data[0])
    return qmc.measure(data)


# %% [markdown]
# 訂正が終わると `data[0]` は $\lvert+\rangle$ に戻ります。$\lvert+\rangle$ はそのまま測ると 0 と 1 が半々になるので、最後に `data[0]` へ $H$ を1つかけて $\lvert0\rangle$ に直してから測定します。訂正が正しく働けば、`data[0]` は常に 0 になるはずです。

# %%
phaseflip_cases = [
    ("no error", 3),
    ("Z on data[0]", 0),
    ("Z on data[1]", 1),
    ("Z on data[2]", 2),
]
if docs_test_mode:
    phaseflip_cases = [
        ("no error", 3),
        ("Z on data[1]", 1),
    ]

print("3-qubit phase-flip code: logical |0_L> = |+++>")
for label, error_pos in phaseflip_cases:
    counts = _sample_first_bit(
        phaseflip_syndrome_run,
        bindings={"error_pos": error_pos},
    )
    print(f"  {label:14s}: data[0]=0 -> {counts[0]:3d}, data[0]=1 -> {counts[1]:3d}")
    # 単一 Z 訂正が完全に効いた後、最後の H が data[0] を |0> に戻すので
    # 各ショットで 0 が読まれる。
    assert counts[1] == 0
    assert counts[0] == counts[0] + counts[1]

# %% [markdown]
# ### 3.7 限界:片方のエラーしか直せない
#
# phase-flip 符号は $Z$ エラーを訂正できますが、今度は $X$ エラーを訂正できません。bit-flip 符号を基底変換したので、訂正できるエラーも入れ替わっただけです。
#
# bit-flip 符号は $X$ だけ、phase-flip 符号は $Z$ だけ ― どちらも片方のエラーしか訂正できません。しかし現実のノイズは $X$ も $Z$ も、さらに両方が同時に起きる $Y$ も引き起こします。次節では、この2つの符号を組み合わせて、任意の単一量子ビットエラーを訂正する **Shor の9量子ビット符号** を作ります。

# %% [markdown]
# ## 4. Shor の9量子ビット符号
#
# bit-flip 符号と phase-flip 符号を組み合わせれば、$X$ と $Z$ の両方を訂正できそうです。それを実現するのが **Shor の9量子ビット符号** です。

# %% [markdown]
# ### 4.1 アイデア:2つの符号を入れ子にする
#
# Shor 符号は、符号化を2段階で行います。
#
# 1. 1量子ビットを phase-flip 符号で3量子ビットに符号化する。
# 2. その3量子ビットを、それぞれさらに bit-flip 符号で3量子ビットに符号化する。
#
# こうして 1 → 3 → 9 量子ビットへと広がります。「符号の中にもう一段符号を入れる」この構成を **連接符号(concatenated code)** と呼びます。外側の phase-flip 層が $Z$ エラーを、内側の bit-flip 層が $X$ エラーを担当します。

# %% [markdown]
# ### 4.2 9量子ビットを3ブロックで見る
#
# 9量子ビットを、3つの **ブロック** に分けて見ます。
#
# ```text
# (q0, q1, q2)   (q3, q4, q5)   (q6, q7, q8)
# ```
#
# 各ブロックが内側の bit-flip 符号です。そして3ブロックの代表である $q_0, q_3, q_6$ が、外側の phase-flip 符号を構成します。

# %% [markdown]
# ### 4.3 符号化回路
#
# 符号化は、外側の phase-flip 符号化を $q_0, q_3, q_6$ にかけてから、各ブロックを bit-flip 符号化するだけです。


# %%
@qmc.qkernel
def encode_shor(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    # 外側: q[0], q[3], q[6] を phase-flip 符号で符号化する。
    q[0], q[3], q[6] = encode_3qubit_phaseflip(q[0], q[3], q[6])

    # 内側: 3つのブロックをそれぞれ bit-flip 符号で符号化する。
    q[0], q[1], q[2] = encode_3qubit_bitflip(q[0], q[1], q[2])
    q[3], q[4], q[5] = encode_3qubit_bitflip(q[3], q[4], q[5])
    q[6], q[7], q[8] = encode_3qubit_bitflip(q[6], q[7], q[8])
    return q


# %% [markdown]
# ### 4.4 シンドローム測定
#
# Shor 符号のシンドロームは8ビットあり、2種類に分かれます。
#
# - **ブロック内の $Z$ パリティ**(`anc[0]`〜`anc[5]`、ブロックあたり2個):これは bit-flip 符号のシンドローム測定そのもので、各ブロックの中で $X$ エラーの位置を特定します。
# - **ブロックをまたぐ $X$ パリティ**(`anc[6]`, `anc[7]`):$X_0X_1X_2X_3X_4X_5$ と $X_3X_4X_5X_6X_7X_8$ の2つです。これは phase-flip 符号のシンドローム測定にあたり、$Z$ エラーを含むブロックを特定します。

# %% [markdown]
# ### 4.5 $Y$ エラーはなぜ直るのか
#
# Shor 符号は $X$ と $Z$ だけでなく $Y$ エラーも訂正できます。$Y=iXZ$ なので、$Y$ エラーは $X$ 成分と $Z$ 成分の両方を含んでいます。
#
# ブロック内の $Z$ パリティが $X$ 成分を、ブロック間の $X$ パリティが $Z$ 成分を、それぞれ独立に検出します。$X$ 訂正と $Z$ 訂正が両方とも当たることで、$Y$ エラーが打ち消されます。

# %% [markdown]
# ### 4.6 実装と実行
#
# ここまでを1つの `@qkernel` にまとめます。`error_type` は `1=X`、`2=Y`、`3=Z` を表し、`error_pos` はエラーを入れる量子ビットの番号です。
#
# 訂正が終わっても、論理状態はまだ9量子ビットに符号化されたままです。この実演では論理ビットを `q[0]` から直接読めるように、最後に符号化の逆回路をかけます。この逆符号化は結果を確認するための手順で、訂正そのものはシンドローム測定とフィードバックで完了しています。


# %%
@qmc.qkernel
def shor_syndrome_run(
    error_type: qmc.UInt,
    error_pos: qmc.UInt,
    theta: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    # データ用に9量子ビット、シンドローム測定用に補助量子ビットを8つ確保する。
    q = qmc.qubit_array(9, name="q")
    anc = qmc.qubit_array(8, name="anc")

    # 論理状態を ry(theta) で用意し、9量子ビットに符号化する。
    q[0] = qmc.ry(q[0], theta)
    q = encode_shor(q)

    # error_type / error_pos で指定した X / Y / Z エラーを注入する。
    for i in qmc.range(9):
        if (error_type == 1) & (error_pos == i):  # 1: X エラー
            q[i] = qmc.x(q[i])
        if (error_type == 2) & (error_pos == i):  # 2: Y エラー
            q[i] = qmc.y(q[i])
        if (error_type == 3) & (error_pos == i):  # 3: Z エラー
            q[i] = qmc.z(q[i])

    # ブロック内 Z パリティ: ブロック b では anc[2b]=Z(3b,3b+1)、anc[2b+1]=Z(3b,3b+2)。
    for b in qmc.range(3):
        q[3 * b], anc[2 * b] = qmc.cx(q[3 * b], anc[2 * b])
        q[3 * b + 1], anc[2 * b] = qmc.cx(q[3 * b + 1], anc[2 * b])
        q[3 * b], anc[2 * b + 1] = qmc.cx(q[3 * b], anc[2 * b + 1])
        q[3 * b + 2], anc[2 * b + 1] = qmc.cx(q[3 * b + 2], anc[2 * b + 1])

    # ブロック間 X パリティ: anc[6] は q[0..5]、anc[7] は q[3..8] にかかる。
    for p in qmc.range(2):
        anc[6 + p] = qmc.h(anc[6 + p])
        for i in qmc.range(6):
            anc[6 + p], q[3 * p + i] = qmc.cx(anc[6 + p], q[3 * p + i])
        anc[6 + p] = qmc.h(anc[6 + p])

    # X 成分の訂正: ブロックごとに、シンドロームを測定して X エラー位置を特定し訂正する。
    for b in qmc.range(3):
        s0 = qmc.measure(anc[2 * b])
        s1 = qmc.measure(anc[2 * b + 1])
        if s0 & s1:  # (1, 1) -> ブロック内 0 番目
            q[3 * b] = qmc.x(q[3 * b])
        if s0 & ~s1:  # (1, 0) -> ブロック内 1 番目
            q[3 * b + 1] = qmc.x(q[3 * b + 1])
        if ~s0 & s1:  # (0, 1) -> ブロック内 2 番目
            q[3 * b + 2] = qmc.x(q[3 * b + 2])

    # Z 成分の訂正: Z エラーを含むブロックを特定して代表量子ビットに Z をかける。
    phase_s0 = qmc.measure(anc[6])
    phase_s1 = qmc.measure(anc[7])
    if phase_s0 & ~phase_s1:
        q[0] = qmc.z(q[0])
    if phase_s0 & phase_s1:
        q[3] = qmc.z(q[3])
    if ~phase_s0 & phase_s1:
        q[6] = qmc.z(q[6])

    # 検証用: 符号化の逆回路をかけ、論理ビットを q[0] に集める。
    for b in qmc.range(3):
        q[3 * b], q[3 * b + 1] = qmc.cx(q[3 * b], q[3 * b + 1])
        q[3 * b], q[3 * b + 2] = qmc.cx(q[3 * b], q[3 * b + 2])
        q[3 * b] = qmc.h(q[3 * b])
    q[0], q[3] = qmc.cx(q[0], q[3])
    q[0], q[6] = qmc.cx(q[0], q[6])

    return qmc.measure(q)


# %% [markdown]
# 各ブロックから代表として1つずつ($X$ をブロック0、$Y$ をブロック1、$Z$ をブロック2)エラーを入れて試します。論理 $\lvert1\rangle$ が保たれていれば、`q[0]` は常に 1 になるはずです。

# %%
shor_cases = [
    ("X", 1, 0),
    ("Y", 2, 4),
    ("Z", 3, 8),
]
if docs_test_mode:
    shor_cases = [
        ("Y", 2, 4),
    ]

print("Shor 9-qubit code: logical |1>")
print(f"  {'error':6s} | {'pos':5s} | P(q[0]=1)")
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
    # 純粋な |1> 入力は Shor 符号の下で単一 Pauli エラーを完全に乗り越え、
    # q[0] は各ショットで 1 を返す。
    assert counts[0] == 0
    assert counts[1] == total

# %% [markdown]
# ### 4.7 なぜ「任意の単一エラー」を訂正できるのか
#
# Shor 符号は $X$, $Y$, $Z$ の3つを訂正できます。これがそのまま「任意の単一量子ビットエラーを訂正できる」ことを意味します。
#
# 1.4 で見たように、どんなエラーも行列としては $I, X, Y, Z$ の線形結合で書け、シンドローム測定がその重ね合わせを1つの離散的な Pauli エラーに収縮させます。収縮後に残るのは $X$, $Y$, $Z$ のいずれか(またはエラーなし)だけで、Shor 符号はそのすべてを訂正できます。だから連続的なものも含めた任意の単一量子ビットエラーに対応できるのです。
#
# 形式的な扱いは後編の [スタビライザ形式論と Steane 符号](steane_code.ipynb) で改めて整理します。

# %% [markdown]
# ## 5. ここまでの共通パターン
#
# 3つの符号 ― bit-flip、phase-flip、Shor ― を作ってきました。実はどれも、同じ4段階の型に従っています。
#
# 1. **符号空間への符号化**: 1つの論理量子ビットを、もつれを使って複数の物理量子ビットに広げる。
# 2. **パリティ測定**: データを直接測らず、複数量子ビットのパリティ($Z_iZ_j$ や $X_iX_j$)を補助量子ビットに取り出す。
# 3. **シンドローム**: パリティの測定結果から、エラーの位置と種類を読み取る。
# 4. **フィードバック訂正**: シンドロームに応じて Pauli ゲートを当て、エラーを打ち消す。
#
# 符号が変わっても、この骨組みは変わりません。違うのは「どのパリティを測るか」だけです。

# %% [markdown]
# ### 5.1 パリティ演算子の名前:スタビライザ
#
# ここまで測ってきたパリティ演算子 ― $Z_0Z_1$ や $X_0X_1$ など ― には名前があります。**スタビライザ(stabilizer)** です。
#
# スタビライザは、符号空間のすべての状態を変えない Pauli 演算子です。正しい符号語に作用させても状態はそのまま(固有値 $+1$)。エラーが入ると、そのエラーと反交換するスタビライザの測定値が $-1$ に変わり、それがシンドロームのビットになります。
#
# bit-flip 符号の $Z_0Z_1$ も、phase-flip 符号の $X_0X_1$ も、Shor 符号の8つのパリティも、すべてスタビライザです。3つの符号は「スタビライザを測ってシンドロームを得る」という1つの考え方の、異なる現れ方だったわけです。この見方を **スタビライザ形式論** と呼び、後編で本格的に扱います。

# %% [markdown]
# ### 5.2 3つの符号のまとめ
#
# 3つの符号を、スタビライザと合わせてまとめます。
#
# | 符号 | $[[n,k,d]]$ | スタビライザ生成子 | 訂正できるエラー |
# | --- | --- | --- | --- |
# | 3量子ビット bit-flip | $[[3,1,1]]$ | $Z_0Z_1,\ Z_0Z_2$ | 単一の $X$ |
# | 3量子ビット phase-flip | $[[3,1,1]]$ | $X_0X_1,\ X_0X_2$ | 単一の $Z$ |
# | Shor 9量子ビット | $[[9,1,3]]$ | $Z_0Z_1,\ Z_0Z_2,\ Z_3Z_4,\ Z_3Z_5,\ Z_6Z_7,\ Z_6Z_8,$ $X_0X_1X_2X_3X_4X_5,\ X_3X_4X_5X_6X_7X_8$ | 単一の $X,\ Y,\ Z$ |
#
# $[[n,k,d]]$ は符号を表す記法です。$n$ は物理量子ビット数、$k$ は守る論理量子ビット数、$d$ は **符号距離** を表します。$d$ は「任意の単一量子ビットエラーを訂正するには $d\ge3$ が必要」という指標で、3量子ビット符号は $d=1$ ― 特定の種類($X$ または $Z$)のエラーしか直せません。Shor 符号は $d=3$ で、任意の単一エラーを直せます。距離の正確な定義は後編で扱います。

# %% [markdown]
# ## 6. やってみよう
#
# 理解を試すために、コードを少し変えて動かしてみましょう。
#
# - **エラー位置を変える**: 各 `*_syndrome_run` の `error_pos` をいろいろな値にして、訂正が効くことを確かめる。
# - **わざと失敗させる**: 以下で、bit-flip 符号の限界を体感します。

# %% [markdown]
# ### わざと失敗させる
#
# bit-flip 符号が訂正できるのは、単一の $X$ エラーまでです。2か所に $X$ エラーが入るとどうなるでしょうか。
#
# 次のカーネルは、論理 $\lvert1\rangle$($\lvert111\rangle$)に `data[0]` と `data[1]` の2か所の $X$ エラーを入れ、いつもどおりシンドローム測定と訂正を行います。

# %%
@qmc.qkernel
def bitflip_two_errors(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    data = qmc.qubit_array(3, name="data")
    anc = qmc.qubit_array(2, name="anc")

    data[0] = qmc.ry(data[0], theta)
    data[0], data[1], data[2] = encode_3qubit_bitflip(data[0], data[1], data[2])

    # 2か所に X エラーを注入する(単一エラー訂正の能力を超える)。
    data[0] = qmc.x(data[0])
    data[1] = qmc.x(data[1])

    # シンドローム測定と訂正は bitflip_syndrome_run と同じ。
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


# %%
print("bit-flip code with TWO X errors (logical |1>)")
counts = _sample_first_bit(
    bitflip_two_errors,
    parameters=["theta"],
    runtime_bindings={"theta": math.pi},
)
print(f"  data[0]=0 -> {counts[0]:3d}, data[0]=1 -> {counts[1]:3d}")
# 失敗モード: 2 か所の X エラーはシンドロームを単一 X エラーと取り違えさせ、
# 「訂正」が論理 |1> を論理 |0> に裏返してしまう。各ショットで data[0] = 0 が
# 決定的に読まれる — これが d=1 の意味。
assert counts[1] == 0
assert counts[0] == counts[0] + counts[1]

# %% [markdown]
# `data[0]` は 1 ではなく 0 になります。訂正が状態を直すどころか、論理 $\lvert1\rangle$ を論理 $\lvert0\rangle$ に変えてしまいました。
#
# 2か所に $X$ が入った $\lvert001\rangle$ では、シンドロームは $\lvert111\rangle$ から「1か所だけ違う」ように見え、訂正が残り1か所にも $X$ をかけて $\lvert000\rangle$ にしてしまうためです。これが「単一エラーまで」という限界 ― 5.2 の表で bit-flip 符号が $d=1$ だったことの実際の意味です。

# %% [markdown]
# ## 7. まとめ
#
# この記事では、3つの量子誤り訂正符号を実装しました。
#
# - **3量子ビット bit-flip 符号** ― $Z$ パリティで単一の $X$ エラーを訂正する。
# - **3量子ビット phase-flip 符号** ― $X$ パリティで単一の $Z$ エラーを訂正する。
# - **Shor の9量子ビット符号** ― 2つを連接し、任意の単一量子ビットエラー($X,\ Y,\ Z$)を訂正する。
#
# 共通する骨組みは「符号空間への符号化 → パリティ(スタビライザ)測定 → シンドローム → フィードバック訂正」でした。
#
# ### 次へ
#
# 後編 [スタビライザ形式論と Steane 符号](steane_code.ipynb) では、ここで名前だけ与えたスタビライザを形式的に扱います。古典符号から系統的に量子符号を作る **CSS 構成** と、$d=3$ を Shor 符号の9量子ビットより少ない7量子ビットで実現する **Steane 符号** が主役です。
