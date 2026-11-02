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
# # スタビライザ形式論と Steane 符号
#
# [量子誤り訂正入門](quantum_error_correction.ipynb)(前編)では、3量子ビット bit-flip / phase-flip 符号と Shor の9量子ビット符号を作り、最後に「パリティ演算子=スタビライザ」という名前を与えました。
#
# 後編では、このスタビライザを形式的に扱います。そして古典の Hamming 符号から系統的に量子符号を作る **CSS 構成** と、その代表例 **Steane 符号** を実装します。Steane 符号は、Shor 符号と同じ $d=3$ を9量子ビットより少ない7量子ビットで実現します。
#
# この記事で扱うこと:
#
# 1. **スタビライザ形式論** ― スタビライザ、生成子、シンドロームを正式に整理する。
# 2. **CSS 構成** ― 古典 Hamming 符号から Steane 符号のスタビライザを作る。
# 3. Steane 符号の符号化・シンドローム測定・訂正を実装する。
# 4. 7つの物理 Hadamard が論理 Hadamard になることを確かめる。
#
# **前提知識**: [前編](quantum_error_correction.ipynb)の内容(シンドローム測定、Pauli エラー、スタビライザという用語)。

# %% [markdown]
# ## 1. スタビライザ形式論
#
# 前編では、$Z_0Z_1$ や $X_0X_1$ のようなパリティ演算子を **スタビライザ** と呼びました。まずこの言葉を正式に整理します。


# %% [markdown]
# ### 1.1 スタビライザとは
#
# **スタビライザ** とは、符号空間のすべての状態を変えない Pauli 演算子 $S$ のことです。任意の符号語 $\lvert\psi\rangle$ に対して
#
# $$S\lvert\psi\rangle = \lvert\psi\rangle$$
#
# が成り立ちます。「符号空間とは、スタビライザの固有値 $+1$ の固有空間である」と言い換えられます。
#
# たとえば bit-flip 符号の符号語は $\lvert000\rangle$ と $\lvert111\rangle$ です。$Z_0Z_1$ をかけると、$Z_0Z_1\lvert000\rangle=\lvert000\rangle$、$Z_0Z_1\lvert111\rangle=(-1)(-1)\lvert111\rangle=\lvert111\rangle$ となり、どちらも変わりません。だから $Z_0Z_1$ は bit-flip 符号のスタビライザです。


# %% [markdown]
# ### 1.2 生成子とシンドローム
#
# 1つの符号にはたくさんのスタビライザがありますが、すべてを並べる必要はありません。いくつかの **生成子(generator)** を選べば、残りはその積で得られます。bit-flip 符号なら $Z_0Z_1$ と $Z_0Z_2$ の2つが生成子です。
#
# スタビライザ生成子を測定すると、結果は固有値 $+1$ か $-1$ になります。
#
# - エラーがなければ、状態は符号空間にあり、すべての生成子が $+1$ を返します。
# - エラー $E$ が入ると、$E$ と **反交換する** 生成子の測定値が $-1$ に変わります。
#
# この $\pm1$(ビットで表せば $0/1$)のパターンが **シンドローム** です。前編で補助量子ビットに取り出していた値は、まさにスタビライザ生成子の測定結果でした。どの生成子が $-1$ になったかを見れば、エラーの位置と種類が分かります。


# %% [markdown]
# ### 1.3 前編の符号をスタビライザで見る
#
# 前編の3つの符号は、すべてスタビライザ生成子で記述できます。
#
# | 符号 | スタビライザ生成子 |
# | --- | --- |
# | 3量子ビット bit-flip | $Z_0Z_1,\ Z_0Z_2$ |
# | 3量子ビット phase-flip | $X_0X_1,\ X_0X_2$ |
# | Shor 9量子ビット | $Z_0Z_1,\ Z_0Z_2,\ Z_3Z_4,\ Z_3Z_5,\ Z_6Z_7,\ Z_6Z_8,\ X_0X_1X_2X_3X_4X_5,\ X_3X_4X_5X_6X_7X_8$ |
#
# bit-flip 符号は $Z$ 型の生成子だけ、phase-flip 符号は $X$ 型の生成子だけを持ちます。Shor 符号は両方を持ち、$Z$ 型が $X$ エラーを、$X$ 型が $Z$ エラーを検出します。この「$X$ 型と $Z$ 型を別々に持つ」構造が、次節の CSS 構成につながります。


# %% [markdown]
# ### 1.4 $[[n,k,d]]$ と符号距離
#
# スタビライザ符号は $[[n,k,d]]$ という3つの数で特徴づけられます。
#
# - $n$:物理量子ビット数。
# - $k$:守る論理量子ビット数。生成子1つにつき自由度が1つ減るので、$k = n - (\text{生成子の数})$ です。
# - $d$:**符号距離**。スタビライザすべてと交換するが、それ自身はスタビライザでない Pauli 演算子 ― これを **論理演算子** と呼びます ― のうち、最小の重み(かかる量子ビット数)です。
#
# 距離 $d$ の符号は $\lfloor(d-1)/2\rfloor$ 個までのエラーを訂正できます。任意の単一量子ビットエラーを訂正するには $d\ge3$ が必要です。前編の3量子ビット符号は $d=1$、Shor 符号は $d=3$ でした。これから作る Steane 符号は $[[7,1,3]]$ です。

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
import qamomile as qm

# %% [markdown]
# ## 2. 古典 Hamming 符号から CSS 符号へ
#
# **CSS 符号(Calderbank-Shor-Steane 符号)** は、古典の誤り訂正符号から量子符号を系統的に作る方法です。Steane 符号はその代表例で、古典の Hamming 符号をもとにします。


# %% [markdown]
# ### 2.1 古典 Hamming [7,4,3] 符号
#
# 古典の Hamming [7,4,3] 符号は、7ビットで4ビットの情報を守り、1ビットの誤りを訂正できる古典符号です。この符号は **パリティ検査行列** $H$ で定義されます。
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
# 7ビットの語 $c$ が符号語であることは、$Hc=0$(各行とのパリティがすべて偶数)と同値です。


# %% [markdown]
# ### 2.2 列が誤り位置を指す仕掛け
#
# $H$ の列をよく見ると、第 $j$ 列は数 $j+1$ の2進表現になっています(第0列は $001$、第1列は $010$、…、第6列は $111$)。
#
# 1ビットの誤りが位置 $j$ に入ると、$Hc$ はちょうど第 $j$ 列に等しくなります。つまり $Hc$ の3ビットを2進数として読めば、それがそのまま誤りの位置を指します。古典 Hamming 符号は、この仕掛けで誤り位置を一発で特定します。


# %% [markdown]
# ### 2.3 CSS 構成:$X$ 型と $Z$ 型のスタビライザ
#
# CSS 構成は、このパリティ検査行列 $H$ から2種類のスタビライザを作ります。
#
# - **$Z$ 型スタビライザ**:$H$ の各行を、$1$ の立つ位置に $Z$ を置いた演算子に読み替える。$X$ エラーを検出する。
# - **$X$ 型スタビライザ**:同じ各行を、$X$ を置いた演算子に読み替える。$Z$ エラーを検出する。
#
# $X$ エラーは $Z$ 型スタビライザと反交換し、$Z$ エラーは $X$ 型スタビライザと反交換します。だから $Z$ 型で $X$ エラーの位置を、$X$ 型で $Z$ エラーの位置を、それぞれ古典 Hamming 符号と同じやり方で特定できます。前編の Shor 符号で見た「$X$ と $Z$ を独立に直す」が、ここでは古典符号から自動的に出てきます。


# %% [markdown]
# ### 2.4 Steane 符号の6つの生成子
#
# Hamming 行列 $H$ の3つの行から、$Z$ 型・$X$ 型それぞれ3つ、合わせて6つのスタビライザ生成子が得られます。
#
# | 型 | スタビライザ | 検出するエラー |
# | --- | --- | --- |
# | $X$ 型 | $X_3X_4X_5X_6$ | $Z$ |
# | $X$ 型 | $X_1X_2X_5X_6$ | $Z$ |
# | $X$ 型 | $X_0X_2X_4X_6$ | $Z$ |
# | $Z$ 型 | $Z_3Z_4Z_5Z_6$ | $X$ |
# | $Z$ 型 | $Z_1Z_2Z_5Z_6$ | $X$ |
# | $Z$ 型 | $Z_0Z_2Z_4Z_6$ | $X$ |
#
# 物理量子ビット7個、生成子6個なので、守れる論理量子ビットは $7-6=1$ 個。これが Steane $[[7,1,3]]$ 符号です。


# %% [markdown]
# 実装に入る前に、QamomileとQiskit連携を読み込み、補助関数を用意します。`_bits7` / `_passes_hamming_checks` / `_is_steane_zero_word`は、測定結果がHamming符号語や$\lvert0_L\rangle$の符号語かを判定するユーティリティです。QECの本筋ではないので、読み飛ばして構いません。

# %% [markdown]
# ## 3. 論理 $\lvert0_L\rangle$ の符号化
#
# ### 3.1 論理ゼロは偶数重み Hamming 符号語の重ね合わせ
#
# Steane 符号の論理 $\lvert0_L\rangle$ は、重みが偶数の Hamming 符号語をすべて足し合わせた重ね合わせです。
#
# $$\lvert0_L\rangle = \frac{1}{2\sqrt2}\sum_{c\in C,\ w(c)\,\text{は偶数}} \lvert c\rangle$$
#
# ここで $C$ は Hamming 符号語の集合、$w(c)$ はその重みです。偶数重みの Hamming 符号語は8個あるので、8項の重ね合わせになります。


# %% [markdown]
# ### 3.2 符号化回路
#
# 次の回路は、$\lvert0\rangle^{\otimes7}$ から $\lvert0_L\rangle$ を作ります。3つの $X$ 型スタビライザのパターン($X_3X_4X_5X_6$、$X_1X_2X_5X_6$、$X_0X_2X_4X_6$)を、Hadamard と CNOT で順に書き込みます。
#
#

# %% [markdown]
# ### 3.3 符号化の確認
#
# 符号化器の出力を測定し、観測されるビット列がすべて偶数重みの Hamming 符号語($\lvert0_L\rangle$ の符号語)になっているかを確かめます。

# %%
@qm.qkernel
def steane_encode_zero(
    state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
    """Encode the final seven components as the Steane logical zero.

    Args:
        state (qm.Q[qm.Bits[10]]): Three syndrome and seven data qubits.

    Returns:
        qm.Q[qm.Bits[10]]: State with components 3 through 9 encoded as logical zero.
    """

    state = qm.h(state, target=6)
    for target in (7, 8, 9):
        state = qm.cx(state, control=6, target=target)
    state = qm.h(state, target=4)
    for target in (5, 8, 9):
        state = qm.cx(state, control=4, target=target)
    state = qm.h(state, target=3)
    for target in (5, 7, 9):
        state = qm.cx(state, control=3, target=target)
    return state


@qm.qkernel
def steane_extract_bit_syndrome(
    state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
    """Extract one Steane bit-error syndrome coherently.

    Args:
        state (qm.Q[qm.Bits[10]]): Encoded data and three clean syndrome qubits.

    Returns:
        qm.Q[qm.Bits[10]]: Data with the syndrome in components 0 through 2.
    """

    for data in (3, 4, 5, 6):
        state = qm.cx(state, control=data + 3, target=0)
    for data in (1, 2, 5, 6):
        state = qm.cx(state, control=data + 3, target=1)
    for data in (0, 2, 4, 6):
        state = qm.cx(state, control=data + 3, target=2)
    return state


@qm.qkernel
def steane_extract_phase_syndrome(
    state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
    """Extract one Steane phase-error syndrome coherently.

    Args:
        state (qm.Q[qm.Bits[10]]): Encoded data and three clean syndrome qubits.

    Returns:
        qm.Q[qm.Bits[10]]: Data with the syndrome in components 0 through 2.
    """

    state = qm.h(state, target=0)
    for data in (3, 4, 5, 6):
        state = qm.cx(state, control=0, target=data + 3)
    state = qm.h(state, target=0)
    state = qm.h(state, target=1)
    for data in (1, 2, 5, 6):
        state = qm.cx(state, control=1, target=data + 3)
    state = qm.h(state, target=1)
    state = qm.h(state, target=2)
    for data in (0, 2, 4, 6):
        state = qm.cx(state, control=2, target=data + 3)
    state = qm.h(state, target=2)
    return state

# %% [markdown]
# ## 4. シンドローム測定と訂正
#
# ### 4.1 $X$ と $Z$ を独立にデコードする
#
# CSS 符号の利点は、$X$ エラーと $Z$ エラーを完全に独立に扱えることです。
#
# - $Z$ 型スタビライザを測ると、$X$ エラーのシンドロームが得られる。
# - $X$ 型スタビライザを測ると、$Z$ エラーのシンドロームが得られる。
#
# それぞれ3ビットのシンドロームで、2.2 で見た Hamming 行列の列の仕掛けがそのまま使えます。


# %% [markdown]
# ### 4.2 シンドローム表
#
# 3ビットのシンドローム $(s_2,s_1,s_0)$ は、エラーの位置をそのまま2進数で表します。
#
# | エラー位置 | シンドローム $(s_2,s_1,s_0)$ |
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
# $Z$ 型スタビライザから得た $(s_2,s_1,s_0)$ は $X$ エラーの位置を、$X$ 型から得たものは $Z$ エラーの位置を指します。


# %% [markdown]
# ### 4.3 実装
#
# `error_type` は `1=X`、`2=Y`、`3=Z`、`error_pos` はエラーを入れる量子ビット番号 `0..6` です。


# %%
def _steane_syndrome_correction(
    target: int,
    *,
    pauli: int,
) -> qm.QKernel:
    """Construct one coherent three-bit syndrome correction.

    Args:
        target: Data component corrected by this syndrome value.
        pauli: Pauli selector: one X, two Y, three Z.

    Returns:
        Unary correction with its syndrome pattern closed in.
    """

    syndrome_value = target + 1
    zero_controls = tuple(
        offset for offset, mask in enumerate((4, 2, 1)) if syndrome_value & mask == 0
    )

    if pauli == 1:

        @qm.qkernel
        def correct_data(
            data: qm.Q[qm.Bits[7]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[7]]:  # type: ignore[valid-type]
            return qm.x(data, target=target)

    elif pauli == 2:

        @qm.qkernel
        def correct_data(
            data: qm.Q[qm.Bits[7]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[7]]:  # type: ignore[valid-type]
            return qm.y(data, target=target)

    else:

        @qm.qkernel
        def correct_data(
            data: qm.Q[qm.Bits[7]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[7]]:  # type: ignore[valid-type]
            return qm.z(data, target=target)

    if not zero_controls:

        @qm.qkernel
        def active_correction(
            state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
            syndrome, data = qm.split_at(state, 3)
            syndrome, data = qm.control(correct_data, syndrome, data)
            return qm.concat(syndrome, data)

        return active_correction

    @qm.qkernel
    def correct(
        state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
        syndrome, data = qm.split_at(state, 3)
        for control in zero_controls:
            syndrome = qm.x(syndrome, target=control)
        syndrome, data = qm.control(correct_data, syndrome, data)
        for control in zero_controls:
            syndrome = qm.x(syndrome, target=control)
        return qm.concat(syndrome, data)

    return correct


def _steane_correction(pauli: int) -> qm.QKernel:
    """Compose all seven nonzero syndrome corrections.

    Args:
        pauli: Pauli selector: one X, two Y, three Z.

    Returns:
        Total coherent correction for one Pauli component.
    """

    @qm.qkernel
    def identity(
        state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
        return state

    current = identity
    for target in range(7):
        previous = current
        correction = _steane_syndrome_correction(target, pauli=pauli)

        @qm.qkernel
        def append_correction(
            state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
            return correction(previous(state))

        current = append_correction
    return current


_STEANE_X_CORRECTION = _steane_correction(1)
_STEANE_Y_CORRECTION = _steane_correction(2)
_STEANE_Z_CORRECTION = _steane_correction(3)

# %% [markdown]
# ## 静的に選んだ誤りexperimentを合成する
#
# $Y=iXZ$誤りにはbitとphaseの両成分があります。以下の代表例ではbit syndromeを取り出し、対応するcoherentな$Y$ correctionを適用します。ほかのstatic branchは対応する$X$または$Z$ extractionと、同じ生成済みcorrection構造を使います。

# %%
@qm.qkernel
def steane_code_samples(
    error_kind: qm.Const[qm.Int],
    error_position: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[10]]:  # type: ignore[valid-type]
    """Correct one statically selected Pauli error in the Steane code.

    Args:
        error_kind (qm.Const[qm.Int]): Pauli selector: one X, two Y, three Z.
        error_position (qm.Const[qm.Int]): Erroneous data-qubit index.

    Returns:
        qm.Shot[qm.Bits[10]]: Corrected codeword and retained syndrome.
    """

    state = steane_encode_zero(qm.qubits(10))
    if error_kind == 1:
        state = qm.x(state, target=error_position + 3)
        state = steane_extract_bit_syndrome(state)
        state = _STEANE_X_CORRECTION(state)
    if error_kind == 2:
        state = qm.y(state, target=error_position + 3)
        state = steane_extract_bit_syndrome(state)
        state = _STEANE_Y_CORRECTION(state)
    if error_kind == 3:
        state = qm.z(state, target=error_position + 3)
        state = steane_extract_phase_syndrome(state)
        state = _STEANE_Z_CORRECTION(state)
    return qm.measure(state)


@qm.qkernel
def steane_transversal_h(
    state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
    """Apply the logical Hadamard transversally to the Steane data.

    Args:
        state (qm.Q[qm.Bits[10]]): Three syndrome and seven encoded data qubits.

    Returns:
        qm.Q[qm.Bits[10]]: State after one Hadamard on every data component.
    """

    for target in range(3, 10):
        state = qm.h(state, target=target)
    return state


@qm.qkernel
def steane_logical_plus_state() -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
    """Prepare logical plus by a transversal Hadamard on logical zero.

    Returns:
        qm.Q[qm.Bits[10]]: Three clean syndrome qubits and logical plus data.
    """

    return steane_transversal_h(steane_encode_zero(qm.qubits(10)))

# %% [markdown]
# ## inspect、estimate、compile、executeする
#
# data位置3の$Y$誤りを選びます。この1件で両方のPauli成分を扱い、deterministicな4 shots以内に収めます。Rustは同じspecialize済みbodyからpeak width 10、21個の`CX` operation、7個のcoherent callを証明します。論理Hadamardは別の7 gate resource witnessとし、2つ目のsimulatorを起動しません。

# %%
ERROR_KIND = 2
ERROR_POSITION = 3
SHOTS = 4

program = steane_code_samples.specialize(
    **{
        "error_kind": ERROR_KIND,
        "error_position": ERROR_POSITION,
    }
)
inspection = program.inspect()
study = qm.resources.logical(program)
transversal_study = qm.resources.logical(steane_transversal_h)

assert "quantum.steane" not in inspection
assert "quantum.error_correction" not in inspection
assert study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure() == (10, 10)
assert study.metric(
    qm.resources.logical_operation_count("quantum.cx")
).numeric_enclosure() == (21, 21)
assert study.metric(
    qm.resources.logical_operation_count("qcore.control_call")
).numeric_enclosure() == (7, 7)
assert transversal_study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (
    7,
    7,
)
assert qm.resources.ResourceStudy.from_bytes(study.to_bytes()) == study

endpoint = qm.Reference()
artifact = qm.compile(program, to=endpoint.target)
record = qm.run(artifact, on=endpoint, shots=SHOTS, seed=2)

logical_x = qm.X(3) * qm.X(4) * qm.X(5) * qm.X(6) * qm.X(7) * qm.X(8) * qm.X(9)
logical_plus_x = qm.expectation(
    steane_logical_plus_state,
    logical_x,
    on=endpoint,
    shots=4,
    seed=3,
)
assert logical_plus_x.value == 1.0

# %% [markdown]
# 保持したsyndromeは`error_position + 1`のbig-endian bit patternです。すべてのdata outcomeはeven-weight Hamming codewordである必要があり、これはSteane論理ゼロの計算基底supportです。

# %%
def is_steane_zero_word(bits: tuple[int, ...]) -> bool:
    """Return whether seven bits form a Steane logical-zero codeword.

    Args:
        bits: Seven data bits in source order.

    Returns:
        True for an even-weight word satisfying all Hamming checks.
    """

    return (
        len(bits) == 7
        and (bits[3] ^ bits[4] ^ bits[5] ^ bits[6]) == 0
        and (bits[1] ^ bits[2] ^ bits[5] ^ bits[6]) == 0
        and (bits[0] ^ bits[2] ^ bits[4] ^ bits[6]) == 0
        and sum(bits) % 2 == 0
    )


syndrome_value = ERROR_POSITION + 1
expected_syndrome = tuple(int(bool(syndrome_value & mask)) for mask in (4, 2, 1))

print("corrected counts:", record.result.counts)
print("retained syndrome:", expected_syndrome)

assert artifact.parameter_abi.slots == ()
assert record.artifact is artifact
assert record.result.shots == SHOTS
assert all(bits[:3] == expected_syndrome for bits in record.result.counts)
assert all(is_steane_zero_word(bits[3:10]) for bits in record.result.counts)

# %% [markdown]
# ### 4.4 検証:21通りの単一エラー
#
# $X$、$Y$、$Z$ を7つの量子ビットそれぞれに入れた、計21通りの単一エラーを試します。訂正後、測定されるビット列はすべて $\lvert0_L\rangle$ の符号語に戻るはずです。


# %% [markdown]
# 比率が 1.000 なら、その単一 Pauli エラーに対して状態が $\lvert0_L\rangle$ の符号空間に戻ったことを意味します。$Y=iXZ$ のエラーは $X$ 成分と $Z$ 成分の訂正の両方を引き起こしますが、CSS 符号ではこの2つが独立なので、そのまま訂正できます。

# %% [markdown]
# ## 5. Transversal な Hadamard ゲート
#
# ### 5.1 transversal ゲートとは
#
# 論理量子ビットに論理ゲートをかけるとき、各物理量子ビットに **それぞれ独立に** 物理ゲートをかけるだけで済む場合、そのゲートを **transversal(横断的)** と言います。
#
# transversal なゲートはフォールトトレラント量子計算で重要です。物理ゲートが互いに独立なので、1つの物理ゲートが故障しても、その誤りが1ブロック内の複数の量子ビットに広がりません。
#
# Steane 符号の大きな特徴は、論理 Hadamard $\bar H$ が transversal なことです。7つの物理量子ビットそれぞれに $H$ をかけるだけで、論理 Hadamard になります。
#
# $$\bar H = H^{\otimes 7}$$
#
# これは、Steane 符号の $X$ 型と $Z$ 型のスタビライザが同じ Hamming パターンを持つ ― CSS 構成で同じ行列 $H$ を使った ― ことの帰結です。


# %% [markdown]
# ### 5.2 論理 Hadamard を確かめる
#
# $\bar H = H^{\otimes7}$ であることを、2つの性質で確かめます。
#
# **性質1: $\bar H$ は $\lvert0_L\rangle$ を $\lvert+_L\rangle$ に移す。**
# $\lvert+_L\rangle$ は $\lvert0_L\rangle$ と $\lvert1_L\rangle$ の重ね合わせです。$\lvert0_L\rangle$ が偶数重みの Hamming 符号語からなるのに対し、$\lvert1_L\rangle$ は奇数重みの Hamming 符号語からなります。したがって $\lvert+_L\rangle$ を測ると、偶数・奇数いずれの重みの Hamming 符号語も現れます($\lvert0_L\rangle$ なら偶数重みのみ)。
#
#

# %% [markdown]
# Hamming 符号語の比率は 1.000 で、奇数重みの符号語も約半分現れます。偶数重みしか出ない $\lvert0_L\rangle$ とは異なる ― 状態が $\lvert+_L\rangle$ に移っていることが確認できます。
#
# **性質2: $\bar H$ を2回かけると恒等変換に戻る($\bar H^2 = I$)。**
# $\lvert0_L\rangle$ に $H^{\otimes7}$ を2回かけると、$\lvert0_L\rangle$ に戻るはずです。
#
#

# %% [markdown]
# 比率は 1.000 ― 2回の transversal Hadamard で $\lvert0_L\rangle$ に戻りました。7つの物理 $H$ が、確かに論理 $\bar H$ として働いています。

# %% [markdown]
# ## 6. まとめ
#
# この記事では、スタビライザ形式論と Steane $[[7,1,3]]$ 符号を扱いました。
#
# - **スタビライザ形式論** ― 符号空間を Pauli 演算子(スタビライザ)の $+1$ 固有空間として捉え、生成子の測定値としてシンドロームを定義した。
# - **CSS 構成** ― 古典 Hamming [7,4,3] 符号のパリティ検査行列から、$X$ 型と $Z$ 型のスタビライザを作った。
# - **Steane 符号** ― $\lvert0_L\rangle$ の符号化、6つのスタビライザによるシンドローム測定、21通りの単一 Pauli エラーの訂正を実装した。
# - **transversal な Hadamard** ― 7つの物理 $H$ が論理 $\bar H$ になることを確かめた。
#
# Steane 符号は、Shor 符号と同じ $d=3$ を9量子ビットより少ない7量子ビットで実現し、CSS 構成と transversal な Clifford ゲートのきれいな例になっています。
#
# ### その先
#
# - **表面符号(surface code)** ― スタビライザを2次元格子上の局所的な演算子にした符号。現在の超伝導量子コンピュータでの誤り訂正の主役で、シンドローム測定を繰り返し行う点が新しい。
# - **フォールトトレラント量子計算** ― 符号化したまま論理ゲートを実行し、誤りを増幅させずに計算を進める枠組み。transversal なゲートはその出発点です。
