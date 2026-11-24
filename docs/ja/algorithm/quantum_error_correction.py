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
# 3量子ビットの反復符号と9量子ビットのShor符号を、Qamomileのゲートと制御操作で実装します。1個の論理量子ビットを符号化し、誤りを入れ、シンドロームを取り出して訂正した後に復号します。Pauliゲートと制御ゲートの基本を前提とします。

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import math
import os

import qamomile as qm

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"


# %% [markdown]
# ## 3量子ビットのビット反転符号
#
# 入力状態を$|\psi\rangle=\alpha|0\rangle+\beta|1\rangle$とします。$|00\rangle$に初期化した2個のデータ量子ビットを加えて$\alpha|000\rangle+\beta|111\rangle$に符号化すると、データのどれか1個に生じた$X$誤りを訂正できます。
#
# データを$q_0,q_1,q_2$とし、$Z_0Z_1$と$Z_0Z_2$の固有値を2個の補助量子ビットに記録します。これらのPauli検査は、すべての符号化状態で固有値$+1$を持つスタビライザです。固有値$+1$を0、$-1$を1とした組$(s_0,s_1)$がシンドロームです。論理状態の$\alpha,\beta$を測らずに、誤り位置を区別できます。
#
# | 誤り位置 | $(s_0,s_1)$ | 訂正 |
# | --- | --- | --- |
# | なし | $(0,0)$ | なし |
# | $q_0$ | $(1,1)$ | $X_0$ |
# | $q_1$ | $(1,0)$ | $X_1$ |
# | $q_2$ | $(0,1)$ | $X_2$ |
#
# この例では、`qm.qubits`でデータと補助をすべて$|0\rangle$に初期化します。以下では、RYで入力を準備し、`error_position`の位置に$X$を入れます。シンドロームを途中で測定する代わりに、補助量子ビットで訂正ゲートを制御します。復号後もシンドロームは残ります。


# %%
@qm.qkernel
def bit_flip_code_state(
    state: qm.Q[qm.Bits[5]],  # type: ignore[valid-type]
    theta: qm.Real,
    error_position: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[5]]:  # type: ignore[valid-type]
    """Encode, diagnose, and coherently correct one bit-flip error."""

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
    """Sample the runtime-parametric three-qubit repetition code."""

    return qm.measure(bit_flip_code_state(qm.qubits(5), theta, error_position))


# %% [markdown]
# ## 訂正して復号する
#
# `specialize`で誤り位置を固定し、Qamomileに組み込まれたシミュレーター`Reference`で実行します。角度`theta`は実行時に与えます。測定結果は、論理量子ビット、0に戻った2個のデータ量子ビット、2個のシンドロームの順です。

# %%
ERROR_POSITION = 1
program = bit_flip_code_samples.specialize(
    error_position=ERROR_POSITION,
)

# %%
THETA = math.pi / 3.0
SHOTS = 1 if docs_test_mode else 256
EXPECTED_SYNDROME = ((1, 1), (1, 0), (0, 1))[ERROR_POSITION]

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
sampling_error = math.sqrt(math.log(4.0e6) / (2 * SHOTS))

print("corrected counts:", result.counts)
print("P(logical=1):", observed_one_probability)

assert result.shots == SHOTS
assert sum(result.counts.values()) == SHOTS
assert all(bits[1:3] == (0, 0) for bits in result.counts)
assert all(bits[3:5] == EXPECTED_SYNDROME for bits in result.counts)
if not docs_test_mode:
    assert abs(observed_one_probability - expected_one_probability) <= sampling_error


# %% [markdown]
# 入力は$\alpha=\cos(\theta/2)$、$\beta=\sin(\theta/2)$なので、復号後の$P(1)$を$\sin^2(\theta/2)$と比較します。`sampling_error`は、独立な理想ショットを仮定し、各比較で両側の逸脱確率を$5\times10^{-7}$以下とするHoeffdingの許容幅です{cite:p}`10.1080/01621459.1963.10500830`。計算基底での測定だけでは相対位相の保存は判定できません。位相を含む状態の検証には、別の基底での測定や状態全体の比較が必要です。

# %% [markdown]
# ## 位相反転符号
#
# 符号化した3個のデータ量子ビットに $H$ を適用し、$Z$ 誤りの後に再び $H$ を適用します。$HZH=X$ なので、以後は同じシンドローム表と訂正回路を使えます。この符号が訂正するのは、1個の $Z$ 誤りです。カーネルを`qm.run`へ直接渡す場合、`Const`の誤り位置はキーワード引数、`Param`の角度は`parameters`で指定します。


# %%
@qm.qkernel
def phase_flip_code_state(
    state: qm.Q[qm.Bits[5]],  # type: ignore[valid-type]
    theta: qm.Real,
    error_position: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[5]]:  # type: ignore[valid-type]
    """Encode, diagnose, and coherently correct one phase-flip error."""

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
    """Sample the runtime-parametric phase-flip repetition code."""

    return qm.measure(phase_flip_code_state(qm.qubits(5), theta, error_position))


phase_result = qm.run(
    phase_flip_code_samples,
    on=endpoint,
    error_position=ERROR_POSITION,
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
assert phase_result.shots == SHOTS
assert sum(phase_result.counts.values()) == SHOTS
if not docs_test_mode:
    assert abs(phase_one_probability - expected_one_probability) <= sampling_error


# %% [markdown]
# ## Shor符号で両方の成分を訂正する
#
# 単一量子ビットの誤り演算子は$I,X,Y,Z$の線形結合で表せるため、Pauli誤りを誤りの基底として使います。符号空間全体で同じ回復操作によりこの基底を訂正できれば、その線形結合も扱えますが、以下の有限ショットの例だけでこの一般的な性質を証明するわけではありません。
#
# $Y=iXZ$を含む単一量子ビットのPauli誤りを扱うため、Shor符号では位相反転の符号化とビット反転の符号化を組み合わせます。9個のデータ量子ビットを3個ずつのブロックに分け、ブロック内の6個の$Z$パリティと、ブロック間の2個の$X$パリティを取り出します。この例は8個の補助量子ビットを加えた17量子ビットの回路です。2個のX検査は$X_0X_1X_2X_3X_4X_5$と$X_3X_4X_5X_6X_7X_8$です。ブロック0、1、2の位相誤りは順に$(1,0)$、$(1,1)$、$(0,1)$となり、`control_state`の1、3、2に対応します。符号化したブロック内の単一Z誤りは同じ作用をするため、位相の訂正には代表の$q_0,q_3,q_6$を使えます。
#
# `error_type`は$X,Y,Z$をそれぞれ1、2、3で指定します。訂正は、取り出したシンドロームで`qm.control`を制御して行います。最後に符号化を逆順で解き、論理状態を$q_0$へ戻します。符号の背景はGottesmanの解説{cite:p}`10.48550/arXiv.quant-ph/9705052`を参照してください。


# %%
@qm.qkernel
def flip_shor_data(state: qm.Q[qm.Bits[9]], position: qm.Int) -> qm.Q[qm.Bits[9]]:
    """Flip one component of a Shor data block."""

    return qm.x(state, target=position)


@qm.qkernel
def phase_shor_data(state: qm.Q[qm.Bits[9]], position: qm.Int) -> qm.Q[qm.Bits[9]]:
    """Apply one phase correction to a Shor data block."""

    return qm.z(state, target=position)


@qm.qkernel
def shor_code_samples(
    theta: qm.Param[qm.Real],
    error_type: qm.Const[qm.Int],
    error_position: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[17]]:  # type: ignore[valid-type]
    """Encode, correct, decode, and sample one Shor-code experiment."""

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


# %% [markdown]
# ## 単一Pauli誤りを確認する
#
# 通常の実行では、9か所それぞれの$X,Y,Z$誤り、計27通りを試します。ここでは入力を$|1\rangle$に固定し、復号後の論理ビットと残りのデータ量子ビットの値を確認します。この確認は任意状態の完全な検証ではなく、訂正回路を動かす最小の例です。

# %%
shor_cases = (
    (("Y", 2, 4),)
    if docs_test_mode
    else tuple(
        (name, kind, position)
        for kind, name in enumerate(("X", "Y", "Z"), 1)
        for position in range(9)
    )
)
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
    assert all(bits[0] == 1 and bits[1:9] == (0,) * 8 for bits in shor_result.counts)

# %% [markdown]
# ## まとめと訂正できる範囲
#
# 反復符号では、シンドロームに従って $X$ 誤りを訂正しました。$H$ で基底を変えると、同じ回路で $Z$ 誤りも訂正できました。Shor符号の例では、入力を $|1\rangle$ とし、誤りの種類と位置を変えた全27通りで復号後の論理ビットが1に戻ることを確認しました。
#
# 3量子ビットの反復符号は、想定する種類の単一誤りを訂正します。Shor符号は任意の単一量子ビット誤りを訂正できますが、2個以上への誤りは一般に保証されません。たとえばビット反転符号の$q_0,q_1$を同時に反転すると、$q_2$だけの反転と同じシンドロームになり、論理ビットを誤って反転します。
#
# ここで使った符号化・検査・訂正ゲート自体は理想的です。回路内の故障や繰り返し訂正を含む耐故障性の実証ではありません。次の[スタビライザ形式論とSteane符号](steane_code.ipynb)では、シンドロームをPauli演算子から整理します。
