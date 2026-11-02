# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
# 実装に入る前に、QamomileとQiskit連携を読み込み、補助関数を2つ用意します。`_first_bit_distribution`と`_sample_first_bit`は、カーネルをコンパイル・実行して先頭ビットの0/1集計を返すだけのユーティリティです。QECの本筋ではないので、読み飛ばして構いません。

# %% [markdown]
# ## インストール
#
# ```bash
# pip install qamomile
# ```

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import math

import qamomile as qm

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


# %% [markdown]
# `error_pos` はコンパイル時に決まるパラメータです。値 `0`, `1`, `2` はその位置に $X$ エラーを注入します。値 `3` はどの分岐にも一致しないので、「エラーなし」を意味します。
#
# まず論理 $\lvert1\rangle$ を用意します(`theta` $=\pi$ の `ry` ゲート)。訂正が正しく働けば、`data[0]` は常に 1 になるはずです。


# %%
@qm.qkernel
def bit_flip_code_state(
    state: qm.Q[qm.Bits[5]],  # type: ignore[valid-type]
    theta: qm.Real,
    error_position: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[5]]:  # type: ignore[valid-type]
    """Encode, diagnose, and coherently correct one bit-flip error.

    Args:
        state (qm.Q[qm.Bits[5]]): Three data and two clean syndrome qubits.
        theta (qm.Real): Logical-state Y rotation angle in radians.
        error_position (qm.Const[qm.Int]): Erroneous data-qubit index.

    Returns:
        qm.Q[qm.Bits[5]]: Decoded logical qubit, clean data, and syndrome bits.
    """

    state = qm.ry(state, theta, target=0)
    state = qm.cx(state, control=0, target=1)
    state = qm.cx(state, control=0, target=2)
    state = qm.x(state, target=error_position)

    state = qm.cx(state, control=0, target=3)
    state = qm.cx(state, control=1, target=3)
    state = qm.cx(state, control=0, target=4)
    state = qm.cx(state, control=2, target=4)

    state = qm.ccx(state, control1=3, control2=4, target=0)
    state = qm.x(state, target=4)
    state = qm.ccx(state, control1=3, control2=4, target=1)
    state = qm.x(state, target=4)
    state = qm.x(state, target=3)
    state = qm.ccx(state, control1=3, control2=4, target=2)
    state = qm.x(state, target=3)

    state = qm.cx(state, control=0, target=2)
    return qm.cx(state, control=0, target=1)


@qm.qkernel
def bit_flip_code_samples(
    theta: qm.Param[qm.Real],
    error_position: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[5]]:  # type: ignore[valid-type]
    """Sample the runtime-parametric three-qubit repetition code.

    Args:
        theta (qm.Param[qm.Real]): Runtime logical-state angle in radians.
        error_position (qm.Const[qm.Int]): Erroneous data-qubit index.

    Returns:
        qm.Shot[qm.Bits[5]]: Logical, clean-data, and syndrome bits.
    """

    return qm.measure(bit_flip_code_state(qm.qubits(5), theta, error_position))


# %% [markdown]
# `error_position`は回路topologyを変えるため`Const`です。`theta`は`Param`なので、1つのコンパイル済みartifactへ異なる論理振幅を渡せます。2つの名前がcompile-time valueとlaunch-time valueとして誤って重複することはありません。

# %% [markdown]
# ## 訂正bodyをinspectしてestimateする
#
# data量子ビット1の代表的な誤りへspecializeします。QEC固有のoperationは導入しません。Rustは通常のcallをたどり、そのbodyから厳密なgate familyと5量子ビットのpeak widthを導出します。

# %%
ERROR_POSITION = 1
program = bit_flip_code_samples.specialize(
    error_position=ERROR_POSITION,
)
inspection = program.inspect()
study = qm.resources.logical(program)

assert "qcore.call" in inspection
assert "quantum.error_correction" not in inspection
assert study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure() == (5, 5)
assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (17, 17)
assert study.metric(
    qm.resources.logical_operation_count("quantum.cx")
).numeric_enclosure() == (8, 8)
assert study.metric(
    qm.resources.logical_operation_count("quantum.ccx")
).numeric_enclosure() == (3, 3)
assert study.metric(qm.resources.LOGICAL_ROTATION_GATES).numeric_enclosure() == (1, 1)
assert qm.resources.ResourceStudy.from_bytes(study.to_bytes()) == study

# %% [markdown]
# ## 1回compileして振幅の保持を検証する
#
# $\theta=\pi/3$の場合、元の論理状態は$P(1)=\sin^2(\theta/2)=1/4$です。64 shotsのsampleで、6標準誤差以内にあることを検証します。decode後の2つのclean data bitは常に0であり、保持したsyndromeは常にdata量子ビット1を示す必要があります。

# %%
THETA = math.pi / 3.0
SHOTS = 64
EXPECTED_SYNDROME = (1, 0)

endpoint = qm.Reference()
artifact = qm.compile(program, to=endpoint.target)
record = qm.run(
    artifact,
    on=endpoint,
    parameters={"theta": THETA},
    shots=SHOTS,
    seed=2,
)
result = record.result
expected_one_probability = math.sin(THETA / 2.0) ** 2
observed_one_probability = (
    sum(count for bits, count in result.counts.items() if bits[0] == 1) / result.shots
)
sampling_error = 6.0 * math.sqrt(
    expected_one_probability * (1.0 - expected_one_probability) / SHOTS
)

print("corrected counts:", result.counts)
print("P(logical=1):", observed_one_probability)

assert tuple(slot.name for slot in artifact.parameter_abi.slots) == ("theta",)
assert record.artifact is artifact
assert all(bits[1:3] == (0, 0) for bits in result.counts)
assert all(bits[3:5] == EXPECTED_SYNDROME for bits in result.counts)
assert abs(observed_one_probability - expected_one_probability) <= sampling_error

# %% [markdown]
# ### 2.7 Superposition input
#
# 測定確率により、correctionが古典値0と1だけでなく、$\cos(\theta/2)|0\rangle+\sin(\theta/2)|1\rangle$のamplitudeも保存することを確認できます。
#
# ### 2.8 制限: phase errorには無力
#
# $Z_0Z_1$と$Z_0Z_2$ parityは符号の変化を検出できません。$Z$を訂正するには、同じrepetition codeを$X$ basisへ移します。

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


# %% [markdown]
# 訂正が終わると `data[0]` は $\lvert+\rangle$ に戻ります。$\lvert+\rangle$ はそのまま測ると 0 と 1 が半々になるので、最後に `data[0]` へ $H$ を1つかけて $\lvert0\rangle$ に直してから測定します。訂正が正しく働けば、`data[0]` は常に 0 になるはずです。


# %% [markdown]
# ### 3.7 限界:片方のエラーしか直せない
#
# phase-flip 符号は $Z$ エラーを訂正できますが、今度は $X$ エラーを訂正できません。bit-flip 符号を基底変換したので、訂正できるエラーも入れ替わっただけです。
#
# bit-flip 符号は $X$ だけ、phase-flip 符号は $Z$ だけ ― どちらも片方のエラーしか訂正できません。しかし現実のノイズは $X$ も $Z$ も、さらに両方が同時に起きる $Y$ も引き起こします。次節では、この2つの符号を組み合わせて、任意の単一量子ビットエラーを訂正する **Shor の9量子ビット符号** を作ります。


# %%
@qm.qkernel
def phase_flip_code_state(
    state: qm.Q[qm.Bits[5]],  # type: ignore[valid-type]
    theta: qm.Real,
    error_position: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[5]]:  # type: ignore[valid-type]
    """Encode, diagnose, and coherently correct one phase-flip error.

    Args:
        state (qm.Q[qm.Bits[5]]): Three data and two clean syndrome qubits.
        theta (qm.Real): Logical-state Y rotation angle in radians.
        error_position (qm.Const[qm.Int]): Erroneous data-qubit index.

    Returns:
        qm.Q[qm.Bits[5]]: Decoded logical qubit, clean data, and syndrome bits.
    """

    state = qm.ry(state, theta, target=0)
    state = qm.cx(state, control=0, target=1)
    state = qm.cx(state, control=0, target=2)
    for index in range(3):
        state = qm.h(state, target=index)
    state = qm.z(state, target=error_position)
    for index in range(3):
        state = qm.h(state, target=index)

    state = qm.cx(state, control=0, target=3)
    state = qm.cx(state, control=1, target=3)
    state = qm.cx(state, control=0, target=4)
    state = qm.cx(state, control=2, target=4)
    state = qm.ccx(state, control1=3, control2=4, target=0)
    state = qm.x(state, target=4)
    state = qm.ccx(state, control1=3, control2=4, target=1)
    state = qm.x(state, target=4)
    state = qm.x(state, target=3)
    state = qm.ccx(state, control1=3, control2=4, target=2)
    state = qm.x(state, target=3)
    state = qm.cx(state, control=0, target=2)
    return qm.cx(state, control=0, target=1)


@qm.qkernel
def phase_flip_code_samples(
    theta: qm.Param[qm.Real],
    error_position: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[5]]:  # type: ignore[valid-type]
    """Sample the runtime-parametric phase-flip repetition code.

    Args:
        theta (qm.Param[qm.Real]): Runtime logical-state angle in radians.
        error_position (qm.Const[qm.Int]): Erroneous data-qubit index.

    Returns:
        qm.Shot[qm.Bits[5]]: Logical, clean-data, and syndrome bits.
    """

    return qm.measure(phase_flip_code_state(qm.qubits(5), theta, error_position))


phase_result = qm.run(
    phase_flip_code_samples,
    on=endpoint,
    error_position=1,
    parameters={"theta": THETA},
    shots=SHOTS,
    seed=3,
).result
phase_one_probability = (
    sum(count for bits, count in phase_result.counts.items() if bits[0] == 1)
    / phase_result.shots
)
assert all(bits[1:3] == (0, 0) for bits in phase_result.counts)
assert all(bits[3:5] == EXPECTED_SYNDROME for bits in phase_result.counts)
assert abs(phase_one_probability - expected_one_probability) <= sampling_error

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
# `error_type`は`1=X`、`2=Y`、`3=Z`です。このbounded witnessではerror typeと
# locationがcircuit structureを選ぶため、どちらも`Const`です。`theta`は小さい
# codeと同様にruntime parameterのままです。


# %%
@qm.qkernel
def flip_shor_data(state: qm.Q[qm.Bits[9]], position: qm.Int) -> qm.Q[qm.Bits[9]]:
    """Flip one component of a Shor data block.

    Args:
        state (qm.Q[qm.Bits[9]]): Nine encoded data qubits.
        position (qm.Int): Component to flip.

    Returns:
        qm.Q[qm.Bits[9]]: Updated data block.
    """

    return qm.x(state, target=position)


@qm.qkernel
def phase_shor_data(state: qm.Q[qm.Bits[9]], position: qm.Int) -> qm.Q[qm.Bits[9]]:
    """Apply one phase correction to a Shor data block.

    Args:
        state (qm.Q[qm.Bits[9]]): Nine encoded data qubits.
        position (qm.Int): Component receiving the phase correction.

    Returns:
        qm.Q[qm.Bits[9]]: Updated data block.
    """

    return qm.z(state, target=position)


@qm.qkernel
def shor_code_samples(
    theta: qm.Param[qm.Real],
    error_type: qm.Const[qm.Int],
    error_position: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[17]]:  # type: ignore[valid-type]
    """Encode, correct, decode, and sample one Shor-code experiment.

    Args:
        theta (qm.Param[qm.Real]): Runtime logical-state angle in radians.
        error_type (qm.Const[qm.Int]): Pauli error selector: 1, 2, or 3.
        error_position (qm.Const[qm.Int]): Erroneous data-qubit index.

    Returns:
        qm.Shot[qm.Bits[17]]: Decoded data followed by measured syndromes.
    """

    state = qm.qubits(17)
    state = qm.ry(state, theta, target=0)

    # Outer phase-flip encoding on representatives 0, 3, and 6.
    state = qm.cx(state, control=0, target=3)
    state = qm.cx(state, control=0, target=6)
    for target in range(0, 9, 3):
        state = qm.h(state, target=target)

    # Inner bit-flip encoding in each three-qubit block.
    for block in range(3):
        representative = 3 * block
        state = qm.cx(state, control=representative, target=representative + 1)
        state = qm.cx(state, control=representative, target=representative + 2)

    if error_type == 1:
        state = qm.x(state, target=error_position)
    elif error_type == 2:
        state = qm.y(state, target=error_position)
    elif error_type == 3:
        state = qm.z(state, target=error_position)

    # Six intra-block Z-parity syndromes occupy components 9 through 14.
    for block in range(3):
        representative = 3 * block
        first_syndrome = 9 + 2 * block
        second_syndrome = first_syndrome + 1
        state = qm.cx(state, control=representative, target=first_syndrome)
        state = qm.cx(state, control=representative + 1, target=first_syndrome)
        state = qm.cx(state, control=representative, target=second_syndrome)
        state = qm.cx(state, control=representative + 2, target=second_syndrome)

    # Inter-block X parities occupy components 15 and 16.
    for parity in range(2):
        syndrome = 15 + parity
        state = qm.h(state, target=syndrome)
        for target in range(3 * parity, 3 * parity + 6):
            state = qm.cx(state, control=syndrome, target=target)
        state = qm.h(state, target=syndrome)

    # Keep syndrome extraction coherent and realize each lookup table with a
    # two-control call. Control component zero is the least-significant bit.
    data, syndromes = qm.split_at(state, 9)
    x0, syndromes = qm.split_at(syndromes, 2)
    x1, syndromes = qm.split_at(syndromes, 2)
    x2, phase = qm.split_at(syndromes, 2)

    x0, data = qm.control(flip_shor_data, x0, data, 0, control_state=3)
    x0, data = qm.control(flip_shor_data, x0, data, 1, control_state=1)
    x0, data = qm.control(flip_shor_data, x0, data, 2, control_state=2)
    x1, data = qm.control(flip_shor_data, x1, data, 3, control_state=3)
    x1, data = qm.control(flip_shor_data, x1, data, 4, control_state=1)
    x1, data = qm.control(flip_shor_data, x1, data, 5, control_state=2)
    x2, data = qm.control(flip_shor_data, x2, data, 6, control_state=3)
    x2, data = qm.control(flip_shor_data, x2, data, 7, control_state=1)
    x2, data = qm.control(flip_shor_data, x2, data, 8, control_state=2)

    phase, data = qm.control(phase_shor_data, phase, data, 0, control_state=1)
    phase, data = qm.control(phase_shor_data, phase, data, 3, control_state=3)
    phase, data = qm.control(phase_shor_data, phase, data, 6, control_state=2)

    # Invert the inner and outer encoders to expose the logical qubit at q0.
    for block in range(3):
        representative = 3 * block
        data = qm.cx(data, control=representative, target=representative + 2)
        data = qm.cx(data, control=representative, target=representative + 1)
        data = qm.h(data, target=representative)
    data = qm.cx(data, control=0, target=6)
    data = qm.cx(data, control=0, target=3)
    syndromes = qm.concat(x0, x1, x2, phase)
    return qm.measure(qm.concat(data, syndromes))


# %%
shor_cases = (("X", 1, 0), ("Y", 2, 4), ("Z", 3, 8))
shor_results = tuple(
    qm.run(
        shor_code_samples,
        on=endpoint,
        error_type=error_type,
        error_position=error_position,
        parameters={"theta": math.pi},
        shots=1,
        seed=10 + error_position,
    ).result
    for _, error_type, error_position in shor_cases
)

for (name, _, position), shor_result in zip(
    shor_cases,
    shor_results,
    strict=True,
):
    print(f"Shor code: {name} error on q[{position}] -> {shor_result.counts}")
    assert all(bits[0] == 1 for bits in shor_result.counts)
#
# ### 4.7 「任意の単一error」を訂正できる理由
#
# syndrome抽出は任意の1-qubit errorをPauli component $I,X,Y,Z$のいずれかへprojectします。Shor codeは各componentを訂正するため、線形性により任意の単一errorを訂正します。

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


# %% [markdown]
# `data[0]` は 1 ではなく 0 になります。訂正が状態を直すどころか、論理 $\lvert1\rangle$ を論理 $\lvert0\rangle$ に変えてしまいました。
#
# 2か所に $X$ が入った $\lvert001\rangle$ では、シンドロームは $\lvert111\rangle$ から「1か所だけ違う」ように見え、訂正が残り1か所にも $X$ をかけて $\lvert000\rangle$ にしてしまうためです。これが「単一エラーまで」という限界 ― 5.2 の表で bit-flip 符号が $d=1$ だったことの実際の意味です。

# %% [markdown]
# ## bit flipから一般のPauli誤りへ
#
# 3量子ビットphase-flip codeは、同じrepetition構成をHadamardで共役したものです。$HZH=X$により、1つの$Z$誤りは上のbit-flip問題へ変換されます。$Y=iXZ$であるため、任意の1量子ビット誤りを訂正するにはbitとphaseの独立したsyndromeが必要です。次の記事[スタビライザー形式とSteane符号](steane_code.ipynb)では、この完全な合成を10量子ビットに限定した1つのQamomile programとして実装します。実行witnessをそちらへ集約することで、この入門ページでは深いstatevector testを重複させません。

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
