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
# # スタビライザ形式論とSteane符号
#
# [量子誤り訂正入門](quantum_error_correction.ipynb)のシンドロームをスタビライザで表し、Steane符号の符号化・訂正・論理HadamardをQamomileで実装します。訂正回路は誤りの種類や位置を受け取らず、ビット反転と位相反転の両シンドロームから単一量子ビット誤りを訂正します。データ7個と、測定・初期化して再利用する補助3個を使い、理想ゲートでの1回の訂正を扱います。

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import os

import qamomile as qm

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"


# %% [markdown]
# ## スタビライザと符号空間
#
# スタビライザ生成子は互いに可換なPauli演算子であり、符号空間の状態$|\psi_L\rangle$に対して$S|\psi_L\rangle=|\psi_L\rangle$を満たします。誤り$E$が$S$と反可換なら、$SE|\psi_L\rangle=-E|\psi_L\rangle$となり、検査の固有値が反転します。これをシンドロームとして読み取ります。
#
# Steane符号は$[[7,1,3]]$符号です。7個の物理量子ビットで1個の論理量子ビットを表し、距離3なので任意の単一量子ビット誤りを訂正できます。スタビライザ形式論の詳細は {cite:t}`10.7907/rzr7-dt72` を参照してください。

# %% [markdown]
# ## Hamming行列から検査を作る
#
# Calderbank–Shor–Steane（CSS）構成では、古典符号のパリティ検査から $X$ 型と $Z$ 型の検査を別々に定めます。Steane符号では、データ量子ビットを$q_0,\ldots,q_6$とし、どちらの検査にも次のHamming符号のパリティ検査行列$H_{\mathrm{cl}}$を使います。
#
# $$
# H_{\mathrm{cl}}=
# \begin{pmatrix}
# 0&0&0&1&1&1&1\\
# 0&1&1&0&0&1&1\\
# 1&0&1&0&1&0&1
# \end{pmatrix}.
# $$
#
# 各行の1の位置に$Z$を置く3個の生成子と、$X$を置く3個の生成子を使います。各行の重みは4で、異なる行は2か所で重なるため、6個の検査はすべて可換です。$Z$型の検査でビット反転成分を、$X$型の検査で位相反転成分を検出します。
#
# 第$j$列は$j+1$の2進表現です。単一誤りの位置は3ビットのシンドロームから求まります。各シンドロームの3ビットは、順に重み4、2、1に対応します。全体の並びは「補助3個＋データ7個」の10量子ビットとし、データ$q_j$を位置`3+j`に置きます。2組のシンドロームは同じ補助3個で順に測り、古典ビットに保存します。

# %% [markdown]
# ## 論理ゼロとシンドロームの回路
#
# 論理ゼロ$|0_L\rangle$は、$H_{\mathrm{cl}}\mathbf{x}=0\pmod 2$を満たす偶数重みの7ビット列8個の等振幅重ね合わせです。以下で符号化器と、ビット反転・位相反転それぞれのシンドローム抽出器を定義します。$Z$ 検査はデータから補助へのCXでパリティを記録し、$X$ 検査は補助の $H$ で挟んだ補助からデータへのCXを使います。どちらも$|0\rangle$に初期化した補助に、固有値$+1,-1$をビット0,1として記録します。


# %%
@qm.qkernel
def steane_encode_zero(
    state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
    """Encode the final seven components as the Steane logical zero."""

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
    """Extract one Steane bit-error syndrome coherently."""

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
    """Extract one Steane phase-error syndrome coherently."""

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
# ## 測定したシンドロームで訂正する
#
# 測定した3ビットが$j+1$の2進表現と一致するとき、データ$q_j$へ訂正をかけます。以下では重み4、2、1の順に`if`で判定し、000には何もしません。判定は回路の実行中に得た測定値を使います。


# %%
@qm.qkernel
def steane_correct_x(
    data: qm.Q[qm.Bits[7]], syndrome: qm.Shot[qm.Bits[3]]
) -> qm.Q[qm.Bits[7]]:
    """Apply X at the position selected by a measured syndrome."""

    if not syndrome[0]:
        if not syndrome[1]:
            if syndrome[2]:
                data = qm.x(data, target=0)
    if not syndrome[0]:
        if syndrome[1]:
            if not syndrome[2]:
                data = qm.x(data, target=1)
    if not syndrome[0]:
        if syndrome[1]:
            if syndrome[2]:
                data = qm.x(data, target=2)
    if syndrome[0]:
        if not syndrome[1]:
            if not syndrome[2]:
                data = qm.x(data, target=3)
    if syndrome[0]:
        if not syndrome[1]:
            if syndrome[2]:
                data = qm.x(data, target=4)
    if syndrome[0]:
        if syndrome[1]:
            if not syndrome[2]:
                data = qm.x(data, target=5)
    if syndrome[0]:
        if syndrome[1]:
            if syndrome[2]:
                data = qm.x(data, target=6)
    return data


@qm.qkernel
def steane_correct_z(
    data: qm.Q[qm.Bits[7]], syndrome: qm.Shot[qm.Bits[3]]
) -> qm.Q[qm.Bits[7]]:
    """Apply Z at the position selected by a measured syndrome."""

    if not syndrome[0]:
        if not syndrome[1]:
            if syndrome[2]:
                data = qm.z(data, target=0)
    if not syndrome[0]:
        if syndrome[1]:
            if not syndrome[2]:
                data = qm.z(data, target=1)
    if not syndrome[0]:
        if syndrome[1]:
            if syndrome[2]:
                data = qm.z(data, target=2)
    if syndrome[0]:
        if not syndrome[1]:
            if not syndrome[2]:
                data = qm.z(data, target=3)
    if syndrome[0]:
        if not syndrome[1]:
            if syndrome[2]:
                data = qm.z(data, target=4)
    if syndrome[0]:
        if syndrome[1]:
            if not syndrome[2]:
                data = qm.z(data, target=5)
    if syndrome[0]:
        if syndrome[1]:
            if syndrome[2]:
                data = qm.z(data, target=6)
    return data


# %% [markdown]
# ## 両方のシンドロームから訂正する
#
# 位置$j$に対応するHamming行列の列を$h_j$とすると、単一Pauli誤りのシンドロームは次の組になります。
#
# | 誤り | ビット反転シンドローム | 位相反転シンドローム |
# | --- | --- | --- |
# | なし | $0$ | $0$ |
# | $X_j$ | $h_j$ | $0$ |
# | $Z_j$ | $0$ | $h_j$ |
# | $Y_j$ | $h_j$ | $h_j$ |
#
# `steane_recover`は量子状態だけを受け取ります。位相反転シンドロームを抽出した後、`qm.measure_reset`で測定値を古典ビットへ保存し、補助3個を$|000\rangle$へ戻します。同じ補助3個でビット反転シンドロームを抽出し、再び測定・初期化します。最後に、ビット反転成分には $X$、位相反転成分には $Z$ を適用します。$Y$ 誤りでは両方が作動し、データ7個は保持されます。
#
# 同じ訂正可能なPauli誤りでは論理0と論理1が同じシンドロームを返すため、理想的なシンドローム測定は論理0と論理1を測り分けず、両者の重ね合わせを保ちます。`qm.measure_reset`が初期化するのは測定した補助だけで、データの論理情報を保ったまま、保存したシンドロームに基づいて訂正を続けられます。
#
# :::{note}
# この回路には途中測定・初期化・測定値による条件分岐が必要です。以下はReferenceで実行します。Qiskit Aer、Selene、Braket Local、CUDA-Qでも確認済みですが、現在のQamomileのCirq・QURI Parts/Qulacs接続はこの構成に未対応です。
# :::


# %%
@qm.qkernel
def steane_recover(
    state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
) -> tuple[qm.Shot[qm.Bits[3]], qm.Shot[qm.Bits[3]], qm.Q[qm.Bits[10]]]:  # type: ignore[valid-type]
    """Measure both syndromes with three reusable ancillas and correct the data."""

    state = steane_extract_phase_syndrome(state)
    ancilla, data = qm.split_at(state, 3)
    phase, ancilla = qm.measure_reset(ancilla)
    state = steane_extract_bit_syndrome(qm.concat(ancilla, data))
    ancilla, data = qm.split_at(state, 3)
    bit, ancilla = qm.measure_reset(ancilla)
    data = steane_correct_x(data, bit)
    data = steane_correct_z(data, phase)
    return bit, phase, qm.concat(ancilla, data)


# %% [markdown]
# ## 誤りを入れて訂正を実行する
#
# `error_kind`と`error_position`は、実験で入れる誤りを選ぶための引数です。0は誤りなし、1, 2, 3は $X, Y, Z$ を表します。訂正関数にはこれらの値を渡しません。`logical_plus=False`では論理ゼロを用い、後の位相の確認では論理プラスを用います。


# %%
@qm.qkernel
def steane_transversal_h(
    state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
    """Apply the logical Hadamard transversally to the Steane data."""

    for target in range(3, 10):
        state = qm.h(state, target=target)
    return state


@qm.qkernel
def steane_code_state(
    error_kind: qm.Const[qm.Int],
    error_position: qm.Const[qm.Int],
    logical_plus: qm.Const[qm.Bool],
) -> tuple[qm.Shot[qm.Bits[3]], qm.Shot[qm.Bits[3]], qm.Q[qm.Bits[10]]]:  # type: ignore[valid-type]
    """Prepare a logical state, inject a test error, and recover without its label."""

    state = steane_encode_zero(qm.qubits(10))
    if logical_plus:
        state = steane_transversal_h(state)
    if error_kind == 1:
        state = qm.x(state, target=error_position + 3)
    if error_kind == 2:
        state = qm.y(state, target=error_position + 3)
    if error_kind == 3:
        state = qm.z(state, target=error_position + 3)
    return steane_recover(state)


@qm.qkernel
def steane_code_samples(
    error_kind: qm.Const[qm.Int],
    error_position: qm.Const[qm.Int],
    logical_plus: qm.Const[qm.Bool],
) -> tuple[qm.Shot[qm.Bits[3]], qm.Shot[qm.Bits[3]], qm.Shot[qm.Bits[10]]]:  # type: ignore[valid-type]
    """Return both classical syndromes and data read in the preparation basis."""

    bit, phase, state = steane_code_state(error_kind, error_position, logical_plus)
    if logical_plus:
        state = steane_transversal_h(state)
    return bit, phase, qm.measure(state)


# %%
ERROR_KIND = 2
ERROR_POSITION = 3
SHOTS = 1 if docs_test_mode else 4

program = steane_code_samples.specialize(
    error_kind=ERROR_KIND,
    error_position=ERROR_POSITION,
    logical_plus=False,
)
endpoint = qm.Reference()
artifact = qm.compile(program, to=endpoint.target)
record = qm.run(artifact, on=endpoint, shots=SHOTS, seed=2)


# %% [markdown]
# `record.result.values`のキーは`(bit, phase, bits)`です。`bit`と`phase`は保存した各3ビットのシンドローム、`bits`は最後に測定した10量子ビットで、`bits[:3]`が初期化済みの補助、`bits[3:]`がデータです。以下のホスト側関数は検査用の期待値を作り、訂正回路の入力には使いません。


# %%
def expected_syndromes(error_kind: int, position: int) -> tuple[int, ...]:
    """Return the two expected syndromes for checking an injected test error."""

    location = tuple(int(bool((position + 1) & mask)) for mask in (4, 2, 1))
    bit = location if error_kind in (1, 2) else (0, 0, 0)
    phase = location if error_kind in (2, 3) else (0, 0, 0)
    return bit + phase


def is_steane_zero_word(bits: tuple[int, ...]) -> bool:
    """Return whether seven bits form a Steane logical-zero codeword."""

    return (
        len(bits) == 7
        and (bits[3] ^ bits[4] ^ bits[5] ^ bits[6]) == 0
        and (bits[1] ^ bits[2] ^ bits[5] ^ bits[6]) == 0
        and (bits[0] ^ bits[2] ^ bits[4] ^ bits[6]) == 0
        and sum(bits) % 2 == 0
    )


expected = expected_syndromes(ERROR_KIND, ERROR_POSITION)
print("corrected values:", record.result.values)
print("bit syndromes:", {bit for bit, phase, bits in record.result.values})
print("phase syndromes:", {phase for bit, phase, bits in record.result.values})

assert record.result.shots == SHOTS
assert all(bit + phase == expected for bit, phase, bits in record.result.values)
assert all(bits[:3] == (0, 0, 0) for bit, phase, bits in record.result.values)
assert all(is_steane_zero_word(bits[3:]) for bit, phase, bits in record.result.values)

# %% [markdown]
# ## 全位置の単一誤りを試す
#
# 通常の実行では、誤りなしと、$X, Y, Z$ と7か所の組合せ21通りを確認します。すべてに同じ訂正関数を使います。計算基底の測定で検査するのはシンドロームと符号語のビット列です。これだけでは相対位相の回復を確認できないため、次に別の基底での確認を行います。

# %%
cases = (
    ((ERROR_KIND, ERROR_POSITION),)
    if docs_test_mode
    else ((0, 0),)
    + tuple((kind, position) for kind in (1, 2, 3) for position in range(7))
)
for kind, position in cases:
    result = qm.run(
        steane_code_samples,
        on=endpoint,
        error_kind=kind,
        error_position=position,
        logical_plus=False,
        shots=SHOTS,
        seed=10 + position,
    ).result
    assert all(
        bit + phase == expected_syndromes(kind, position)
        for bit, phase, bits in result.values
    )
    assert all(bits[:3] == (0, 0, 0) for bit, phase, bits in result.values)
    assert all(is_steane_zero_word(bits[3:]) for bit, phase, bits in result.values)
print(f"Checked {len(cases)} patterns (no error and single-qubit errors)")

# %% [markdown]
# ## 論理Hadamardと位相の確認
#
# データ7個へそれぞれHをかける操作を、横断的（transversal）なHadamardと呼びます。$HXH=Z$、$HZH=X$なので、Steane 符号の $X$ 型と $Z$ 型の生成子を入れ替えます。論理演算子$\overline Z=Z_0\cdots Z_6$と$\overline X=X_0\cdots X_6$も入れ替えるため、論理Zの固有値$+1$の状態$|0_L\rangle$を、論理Xの固有値$+1$の状態$|+_L\rangle$へ移します。
#
# ここでは論理プラスへ $Z$ 誤りを入れて訂正し、データ 7 個へ $H$ をかけてから測定します。測定ビット列を$\mathbf b$とすると、$(-1)^{\sum_j b_j}$の平均が論理演算子$\overline X=X_0\cdots X_6$の期待値です。これが$+1$へ戻ることを確かめます。この量は相対位相に敏感です。物理的な $Z$ 誤りを残したまま測ると符号が反転するため、計算基底のビット列だけでは見えない訂正の効果を確認できます。これは1例の検査であり、任意状態や耐故障性の完全な検証ではありません。

# %%
plus_result = qm.run(
    steane_code_samples,
    on=endpoint,
    error_kind=3,
    error_position=ERROR_POSITION,
    logical_plus=True,
    shots=SHOTS,
    seed=3,
).result
assert all(
    bit + phase == expected_syndromes(3, ERROR_POSITION)
    for bit, phase, bits in plus_result.values
)
assert all(bits[:3] == (0, 0, 0) for bit, phase, bits in plus_result.values)
corrected_plus_x = (
    sum(
        (-1) ** sum(bits[3:]) * count
        for (bit, phase, bits), count in plus_result.values.items()
    )
    / plus_result.shots
)
assert corrected_plus_x == 1.0
print("Logical X after phase-error correction:", corrected_plus_x)

# %% [markdown]
# ## まとめ
#
# Steane符号の6個の生成子から2組のシンドロームを取り出し、その値だけでXとZの訂正を制御しました。誤りの種類と位置を指定するのはテスト用の注入処理だけです。補助3個を測定・初期化して再利用し、両シンドロームは古典ビットに保持するため、全体は10量子ビットです。
#
# 通常の実行では、全22例で期待どおりのシンドロームが得られ、訂正後に測定したビット列はすべて論理ゼロの符号語でした。論理プラスに入れた $Z$ 誤りを訂正すると、論理 $X$ の期待値は $+1$ になりました。ゲート・測定・初期化自体の故障や、繰り返し訂正を含む耐故障性は扱っていません。
