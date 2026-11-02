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
# tags: [algorithm, encoding, primitive]
# ---
#
# ---
# tags: [algorithm, encoding, primitive]
# ---
#
# # Möttönen振幅エンコーディング
#
# このノートブックでは、Qamomileの明示的なMöttönen構成を使って実数および複素数の振幅ベクトルを準備し、得られる状態とゲート数を検証します。また、汎用の振幅エンコーディングではなく、手法を指定するAPIを選ぶべき場合について説明します。
#
# ## 背景
#
# **振幅エンコーディング**は、単位ノルムの複素ベクトル$a \in \mathbb{C}^{2^n}$から$n$量子ビットの状態を準備します。
#
# $$
# |0\rangle^{\otimes n}
# \longmapsto
# |\psi\rangle
# = \sum_{i=0}^{2^n - 1} a_i |i\rangle.
# $$
#
# Möttönenの振幅encodingは、このstate preparationを実現するalgorithmの一つです。
# 現在のstdlibは、portableな`amplitude_encoding(...)`を1つ公開しており、
# この名前はここで説明する明示的なMöttönen/Gray-walk構成を意味します。
# target adapterが無関係なnative state-preparation methodへ暗黙に置き換えることはありません。
#
# 明示的な実装は、Möttönen、Vartiainen、Bergholm、Salomaaによる一様制御回転の構成{cite:p}`10.48550/arXiv.quant-ph/0407010`に従います。論文では、より一般的な任意状態変換$|a\rangle \to |b\rangle$を扱います。Qamomileでは入力を$|0\rangle^{\otimes n}$に固定し、状態準備に相当する片側を実装しています。
#
# :::{note}
# 入力量子ビットが$|0\rangle^{\otimes n}$でない場合、これらのAPIは一般に出力状態を保証しません。
# :::

# %% [markdown]
# ## インストール
#
# Qamomile distributionをインストールします。
#
# ```bash
# pip install qamomile
# ```

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install qamomile

# %%
import math

import qamomile as qm
from qamomile.stdlib import amplitude_angles, amplitude_encoding_from_angles

# %% [markdown]
# ## 問題設定
#
# 2量子ビットを使い、準備した状態ベクトルを次の3つの正規化された目標と比較します。
#
# - 正の実数ベクトル$(1, 2, 3, 4)$
# - 符号付き実数ベクトル$(1, -1, 1, -1)$
# - 複素数ベクトル$(1, 1+i, 1-i, 2i)$
#
# 状態忠実度は位相に依存しないため、エンジンが異なるグローバル位相を選んだ場合でも、意図した物理状態を検証できます。

# %%
positive_amplitudes = (1.0, 2.0, 3.0, 4.0)
signed_amplitudes = (1.0, -1.0, 1.0, -1.0)

# %% [markdown]
# ## アルゴリズム
#
# Möttönenの状態準備は、2つのカスケードで構成されます。
#
# 1. 一様制御$R_y$回転によって、計算基底状態に大きさ$|a_i|$を分配します。実数ベクトルでは、符号付きの回転角によって負の振幅もエンコードします。
# 2. 一様制御$R_z$回転によって、相対位相を復元します。すべての振幅の虚部がゼロの場合、Qamomileはこのカスケードを省略します。
#
# 各一様制御回転は、Gray符号構成によって基本回転と`CNOT`ゲートに分解されます。Qamomileが実装する$|0\rangle^{\otimes n} \to |\psi\rangle$の場合、ゲート数は次のようになります。
#
# | 入力 | 回転 | `CNOT`ゲート |
# |---|---:|---:|
# | 実数 | $2^n - 1$ | $2^n - 2$ |
# | 複素数 | $2(2^n - 1)$ | $2(2^n - 2)$ |
#
# $k$個の制御を持つ一様制御回転には、$2^k$個の基本回転と$2^k$個の`CNOT`ゲートが必要です。$k = 0, 1, \ldots, n-1$の各段を合計すると、上表の式が得られます。制御のない段には`CNOT`ゲートがありません。論文の要旨に示された、より大きなゲート数は、ここで実装する状態準備の片側ではなく、任意入力の完全な変換$|a\rangle \to |b\rangle$に対するものです。また、Qamomileはカスケード間の`CNOT`相殺を適用せず、各段をそのまま分解します。
#
# ### 再利用可能な座標の導出
#
# 次の2量子ビット複素vectorを準備します。
#
# $$
# (1,\;1+i,\;1-i,\;2i)
# $$
#
# 二乗normは9なので、計算基底での確率は$(1,2,2,4)/9$です。`amplitude_angles`は正規化前のvectorを受け取り、3つの$R_y$角、3つの$R_z$角、保持された1つのglobal phaseを含むimmutableな値を返します。

# %%
amplitudes = (
    1.0 + 0.0j,
    1.0 + 1.0j,
    1.0 - 1.0j,
    0.0 + 2.0j,
)
coordinates = amplitude_angles(amplitudes)

assert coordinates.width == 2
assert len(coordinates.ry) == 3
assert len(coordinates.rz) == 3
assert all(math.isfinite(value) for value in coordinates.ry)
assert all(math.isfinite(value) for value in coordinates.rz)
assert math.isfinite(coordinates.phase)
print(coordinates)

# %% [markdown]
# 座標合成は有限の古典dataに対するhost-side preprocessingです。必須のRust boundaryを1回通り、Pythonによる数値fallbackはありません。不正な長さ、非有限値、ゼロvector、過大な幅はQCore構築前に失敗します。

# %% [markdown]
# ## 実装
#
# ### runtimeで角度を再指定できるよう一度compileする
#
# `amplitude_encoding_from_angles(2)`は2量子ビットのtopologyだけを固定します。座標はaggregate runtime parameterのままなので、再コンパイルせずに1つのartifactで多数のvectorを準備できます。shapeは関数型と生成されるlaunch ABIの一部です。

# %%
encode = amplitude_encoding_from_angles(2)


@qm.qkernel
def prepared_state(
    ry_angles: qm.Param[qm.Vector[qm.Real, 3]],
    rz_angles: qm.Param[qm.Vector[qm.Real, 3]],
    phase: qm.Param[qm.Real],
) -> qm.Q[qm.Bits[2]]:
    """Prepare one two-qubit state from Möttönen coordinates.

    Args:
        ry_angles (qm.Param[qm.Vector[qm.Real, 3]]): Magnitude-cascade angles.
        rz_angles (qm.Param[qm.Vector[qm.Real, 3]]): Relative-phase angles.
        phase (qm.Param[qm.Real]): Retained global phase.

    Returns:
        qm.Q[qm.Bits[2]]: Prepared two-qubit state.
    """

    return encode(qm.qubits(2), ry_angles, rz_angles, phase)


@qm.qkernel
def prepared_samples(
    ry_angles: qm.Param[qm.Vector[qm.Real, 3]],
    rz_angles: qm.Param[qm.Vector[qm.Real, 3]],
    phase: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[2]]:
    """Measure the reusable two-qubit preparation.

    Args:
        ry_angles (qm.Param[qm.Vector[qm.Real, 3]]): Magnitude-cascade angles.
        rz_angles (qm.Param[qm.Vector[qm.Real, 3]]): Relative-phase angles.
        phase (qm.Param[qm.Real]): Retained global phase.

    Returns:
        qm.Shot[qm.Bits[2]]: Sampled computational-basis state.
    """

    return qm.measure(prepared_state(ry_angles, rz_angles, phase))


# %% [markdown]
# compile時にvectorを固定する場合は、`amplitude_encoding(amplitudes)`が閉じた単項transformを返します。どちらも同じRust座標と通常のgate語彙を使います。artifactの再利用が必要な場合だけruntime形式を選びます。

# %% [markdown]
# ### resource確認用kernelの構築
#
# 検証済みprogramにはamplitude-preparation opcodeがありません。resource analysisは通常のrotationとcontrolled operationを直接合成します。

# %%
program = qm.specialize(prepared_samples)
study = qm.resources.logical(program)
inspection = program.inspect()

assert "quantum.amplitude" not in inspection
assert study.metric(qm.resources.LOGICAL_ROTATION_GATES).numeric_enclosure() == (
    6,
    6,
)
assert study.metric(
    qm.resources.logical_operation_count("quantum.cx")
).numeric_enclosure() == (4, 4)

# %% [markdown]
# 一般の$n$量子ビット複素状態準備では、この分解に$2(2^n-1)$個のrotationと$2(2^n-2)$個の`CX` gateが必要です。指数関数的なcostはbackend-native instructionに隠れず、推定値に現れます。

# %% [markdown]
# ## 結果
#
# ### stateの振幅
#
# コンパイルにより3つのaggregate parameter slotが封印されます。`bindings()`はglobal phaseを含む、launch ABIに正確に一致したmappingを生成します。

# %%
reference = qm.Reference()
artifact = qm.compile(program, to=reference)
parameters = coordinates.bindings()
record = qm.run(
    artifact,
    on=reference,
    parameters=parameters,
    shots=128,
    seed=2,
)

assert tuple(slot.name for slot in artifact.parameter_abi.slots) == (
    "ry_angles",
    "rz_angles",
    "phase",
)
assert record.result.shots == 128
assert sum(record.result.counts.values()) == 128

expected_probabilities = {
    (0, 0): 1.0 / 9.0,
    (1, 0): 2.0 / 9.0,
    (0, 1): 2.0 / 9.0,
    (1, 1): 4.0 / 9.0,
}
for bits, expected in expected_probabilities.items():
    sampling_bound = 5.0 * math.sqrt(expected * (1.0 - expected) / record.result.shots)
    assert abs(record.result.probability(bits) - expected) <= sampling_bound

observed_probabilities = {
    bits: record.result.probability(bits) for bits in expected_probabilities
}
print("sampled probabilities:", observed_probabilities)

# %% [markdown]
# ### runtime rebinding
#
# 6個のrotation angleとglobal phaseだけが変わるため、同じartifactで3つのtargetすべてをprepareできます。structure、result ABI、target payloadは固定されたままです。

# %%
for trial_amplitudes in (
    positive_amplitudes,
    signed_amplitudes,
    amplitudes,
):
    trial_coordinates = amplitude_angles(trial_amplitudes)
    trial_record = qm.run(
        artifact,
        on=reference,
        parameters=trial_coordinates.bindings(),
        shots=128,
        seed=3,
    )
    norm = sum(abs(value) ** 2 for value in trial_amplitudes)
    for index, value in enumerate(trial_amplitudes):
        bits = (index & 1, (index >> 1) & 1)
        expected = abs(value) ** 2 / norm
        bound = 5.0 * math.sqrt(expected * (1.0 - expected) / trial_record.result.shots)
        assert abs(trial_record.result.probability(bits) - expected) <= bound

# %% [markdown]
# samplingはmagnitudeを検査します。signとcomplex phaseにはinterference witnessが必要なので、次のsectionではobservableも評価します。

# %% [markdown]
# ## 相対位相の検証
#
# 計算基底での確率だけでは複素位相を区別できません。このvectorに対するobservable $Y_0X_1$の厳密な期待値は$8/9$なので、同じ`prepared_state`実装を使った位相感度のあるwitnessになります。

# %%
phase_witness = qm.expectation(
    prepared_state,
    qm.Y(0) * qm.X(1),
    on=reference,
    parameters=parameters,
    shots=64,
    seed=2,
)
expected_phase_witness = 8.0 / 9.0
phase_sampling_bound = 5.0 * math.sqrt(
    (1.0 - expected_phase_witness**2) / phase_witness.shots
)

assert abs(phase_witness.value - expected_phase_witness) <= phase_sampling_bound
print(f"<Y0 X1>: {phase_witness.value:.3f}")

# %%
positive_coordinates = amplitude_angles(positive_amplitudes)
signed_coordinates = amplitude_angles(signed_amplitudes)
positive_witness = qm.expectation(
    prepared_state,
    qm.Z(0),
    on=reference,
    parameters=positive_coordinates.bindings(),
    shots=64,
    seed=4,
)
signed_witness = qm.expectation(
    prepared_state,
    qm.X(0),
    on=reference,
    parameters=signed_coordinates.bindings(),
    shots=64,
    seed=4,
)

assert abs(positive_witness.value + 1.0 / 3.0) <= 5.0 * math.sqrt(
    (1.0 - (1.0 / 3.0) ** 2) / positive_witness.shots
)
assert abs(signed_witness.value + 1.0) <= 0.1

# %% [markdown]
# $(1,2,3,4)$では$\langle Z_0\rangle=-1/3$です。$(1,-1,1,-1)$では$\langle X_0\rangle=-1$となり、computational-basis probabilityからは見えないnegative signが保持されたことを証明します。

# %% [markdown]
# ## fixed amplitudeとreusable angle
#
# `amplitude_encoding(amplitudes)`はPython factory timeにcoordinateを計算して閉じます。target vectorがprogram structureの一部ならこちらを使います。`amplitude_encoding_from_angles(width)`はtopologyだけを固定し、このページのように型付きangle vectorをruntime ABIへ残します。どちらも同じMöttönen synthesisなので、gate-count formulaも同じです。

# %% [markdown]
# ## まとめ
#
# - 正規化、exact-phase座標、Gray-walk topology、幅の上限はRustが所有します。
# - aggregate `Param[Vector[Real, 3]]` annotationにより、再利用可能なABIを型付けします。
# - programは通常の`RY`、`RZ`、`CX`、global-phase compositionです。
# - 同じsourceで厳密なlogical resource、compilation、sampling、位相感度のあるexpectation valueを扱えます。
# - 1つのartifactをpositive、signed-real、complex targetへrebindします。
# - 実行する証拠はすべて2量子ビットに限定し、より広い幅のscalingはsymbolicに扱います。
#
# 次の[多次元QFT](multidimensional_qft.ipynb)では、状態準備と2つの論理軸に沿った独立なFourier transformを合成します。
