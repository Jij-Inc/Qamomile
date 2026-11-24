# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
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
# # Möttönen 法による振幅符号化
#
# 正の実数、符号付き実数、複素数の3種類の振幅ベクトルから、Möttönen 法で2量子ビットの状態を準備します。振幅から計算した角度を渡す方法と、振幅を直接渡す方法を実装し、測定確率とゲート数を比較します。

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install qamomile

# %%
import os

import qamomile as qm
from qamomile.stdlib import amplitude_angles, amplitude_encoding, amplitude_encoding_from_angles

SAMPLE_SHOTS = 1 if os.environ.get("QAMOMILE_DOCS_TEST") == "1" else 128

# %% [markdown]
# ## 背景
#
# 振幅符号化（amplitude encoding）は、長さ $2^n$ のゼロでないベクトル $a\in\mathbb{C}^{2^n}$ を規格化し、$n$ 量子ビットの量子状態に対応付ける方法です。$\tilde a_i=a_i/\|a\|_2$ とすると、
#
# $$
# |0\rangle^{\otimes n}\longmapsto
# |\psi(a)\rangle=\sum_{i=0}^{2^n-1}\tilde a_i|i\rangle,
# \qquad \sum_i|\tilde a_i|^2=1.
# $$
#
# 原論文では、一様制御回転を用いて任意の初期状態を目的の状態に変換する方法が示されています{cite:p}`10.48550/arXiv.quant-ph/0407010`。この例では、すべての量子ビットが $|0\rangle$ の状態から目的の状態を準備するように回転角を求めます。初期状態が異なると、同じ回路を適用しても、一般には目的の状態は得られません。
#
# ## 問題設定
#
# 2 量子ビット（振幅 4 成分）を用い、次の 3 種類の規格化前の入力ベクトルから状態を準備します。
#
# - 正の実数ベクトル：$(1,2,3,4)$（ノルム $\sqrt{30}$）
# - 符号付き実数ベクトル：$(1,-1,1,-1)$（ノルム $2$）
# - 複素数ベクトル：$(1,1+i,1-i,2i)$（ノルム $3$）
#
# いずれも実際に準備する状態の振幅は、入力をそのノルムで割った値です。
#
# 以下では、量子ビット番号 $0$ を下位ビットとし、インデックス $i=b_0+2b_1$ に対応する状態を $|b_1b_0\rangle$ と表記することにします。Qamomile の測定ビット列は $(b_0,b_1)$ の順で扱います。例えば、$(1,0)$ はインデックス1（$|01\rangle$）に対応します。

# %%
input_vectors = {
    "positive": (1.0, 2.0, 3.0, 4.0),
    "signed": (1.0, -1.0, 1.0, -1.0),
    "complex": (1.0, 1.0 + 1.0j, 1.0 - 1.0j, 2.0j),
}

# %% [markdown]
# ## アルゴリズム
#
# Möttönen 法による状態準備は、次の2種類の回転から構成されます。
#
# 1. 一様制御$R_y$回転：各計算基底状態に必要な振幅の大きさを分配します。
# 2. 一様制御$R_z$回転：相対位相を付与します。正と負の実数振幅の間にある$\pi$の位相差も、この回転で表します。Qamomileの固定入力用`amplitude_encoding`は、計算した$R_z$の角度がすべてゼロの場合に、この位相回転列を省略します。
#
# 一様制御回転は Gray 符号に基づくゲート列に分解されます。固定入力用の`amplitude_encoding`で作る状態準備回路の論理ゲート数は、計算した$R_z$の角度によって次のように変わります。
#
# | 回路の種類 | $R_y$ / $R_z$ 回転の合計 | `CX` ゲート |
# |---|---:|---:|
# | $R_z$の角度がすべてゼロ（$R_y$のみ） | $2^n-1$ | $2^n-2$ |
# | $R_z$の角度にゼロでないものがある（$R_y$と$R_z$） | $2(2^n-1)$ | $2(2^n-2)$ |
#
# ゼロでない振幅すべてに共通する位相は大域位相として扱います。負の振幅があっても、計算した $R_z$ の角度がすべてゼロであれば、固定入力の回路では $R_z$ の回転列を省略できます。
#
# 2量子ビットの $R_y$ 回転列では、最初に量子ビット1を1回だけ回します。次に、量子ビット1が $0$ か $1$ かに応じて量子ビット0の回転角を変えます。この段に回転2個と `CX` 2個を使うので、合計は回転3個・`CX` 2個です。
#
# 一般に、制御量子ビットが $k\ge1$ 個の段では、制御ビットの組み合わせが $2^k$ 通りあり、この構成では回転と `CX` を $2^k$ 個ずつ使います。制御のない最初の段も含めて $n$ 段を足すと、回転は $2^n-1$ 個、`CX` は $2^n-2$ 個になります。

# %% [markdown]
# ## 実装
#
# ### `amplitude_encoding_from_angles` で角度を渡す
#
# この回路は2量子ビットをすべて $|0\rangle$ で用意し、$R_y$ と $R_z$ の回転角を3個ずつ、さらに大域位相を実行時に受け取ります。回路は一度だけコンパイルします。実行時に、その入力の振幅から `amplitude_angles` で角度を計算して渡します。

# %%
encode_angles = amplitude_encoding_from_angles(2)


@qm.qkernel
def samples_from_angles(
    ry_angles: qm.Param[qm.Vector[qm.Real, 3]],
    rz_angles: qm.Param[qm.Vector[qm.Real, 3]],
    phase: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[2]]:
    """Prepare and measure two qubits using the supplied rotation angles."""
    state = encode_angles(qm.qubits(2), ry_angles, rz_angles, phase)
    return qm.measure(state)


reference = qm.Reference()
angle_program = qm.specialize(samples_from_angles)
angle_artifact = qm.compile(angle_program, to=reference)

# %% [markdown]
# ### `amplitude_encoding` に振幅を直接渡す
#
# こちらは振幅ベクトルを渡すと、`amplitude_encoding` が Möttönen 法の回転角を計算します。正の実数、負の値を含む実数、複素数の3種類の入力について、それぞれの振幅を組み込んだ測定回路を定義します。

# %%
prepare_positive = amplitude_encoding(input_vectors["positive"])


@qm.qkernel
def sample_positive() -> qm.Shot[qm.Bits[2]]:
    qubits = qm.qubits(2)
    state = prepare_positive(qubits)
    return qm.measure(state)


# %%
prepare_signed = amplitude_encoding(input_vectors["signed"])


@qm.qkernel
def sample_signed() -> qm.Shot[qm.Bits[2]]:
    qubits = qm.qubits(2)
    state = prepare_signed(qubits)
    return qm.measure(state)


# %%
prepare_complex = amplitude_encoding(input_vectors["complex"])


@qm.qkernel
def sample_complex() -> qm.Shot[qm.Bits[2]]:
    qubits = qm.qubits(2)
    state = prepare_complex(qubits)
    return qm.measure(state)


# %%
fixed_programs = {
    "positive": qm.specialize(sample_positive),
    "signed": qm.specialize(sample_signed),
    "complex": qm.specialize(sample_complex),
}

# %% [markdown]
# ## 結果
#
# 3種類の入力を両方の回路で測定し、計算基底の各状態 $|b_1b_0\rangle$ が現れる確率を比べます。表の `ideal` は理論値 $|a_i|^2 / \sum_j |a_j|^2$、`from_angles` は角度を渡した回路、`fixed` は振幅を直接渡した回路の測定頻度です。

# %%
bitstrings = ((0, 0), (1, 0), (0, 1), (1, 1))


def ideal_probabilities(amplitudes):
    """Return the normalized squared magnitudes in basis-index order."""
    scale = max(max(abs(value.real), abs(value.imag)) for value in amplitudes)
    weights = [abs(value / scale) ** 2 for value in amplitudes]
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def sample_probabilities(program, *, seed, parameters=None):
    """Run the sampling circuit and return probabilities in the same order."""
    result = qm.run(
        program, on=reference, parameters=parameters, shots=SAMPLE_SHOTS, seed=seed
    ).result
    assert sum(result.counts.values()) == SAMPLE_SHOTS
    return tuple(result.probability(bits) for bits in bitstrings)


for name, amplitudes in input_vectors.items():
    ideal = ideal_probabilities(amplitudes)
    from_angles = sample_probabilities(
        angle_artifact, parameters=amplitude_angles(amplitudes).bindings(), seed=2
    )
    fixed = sample_probabilities(fixed_programs[name], seed=3)

    print(f"\n{name}")
    print("basis   ideal   from_angles   fixed")
    for i in range(len(ideal)):
        print(f"|{i:02b}>   {ideal[i]:.3f}   {from_angles[i]:11.3f}   {fixed[i]:.3f}")

# %% [markdown]
# 理論確率は、正の実数例では $(1,4,9,16)/30$、符号付き実数例では $(1,1,1,1)/4$、複素数例では $(1,2,2,4)/9$ です。2方式が準備する状態の理論確率は同じですが、有限回の測定で得られる頻度にはばらつきがあります。この測定で比較できるのは振幅の大きさです。
#
# 次に、同じ3入力について回転ゲートと `CX` ゲートの個数を比較します。

# %%
def gate_counts(program):
    """Return the rotation and CX counts for this two-qubit circuit."""
    study = qm.resources.logical(program)
    rotations = study.metric(qm.resources.LOGICAL_ROTATION_GATES).numeric_enclosure()
    cx_metric = qm.resources.logical_operation_count("quantum.cx")
    cx = study.metric(cx_metric).numeric_enclosure()
    assert rotations[0] == rotations[1] and cx[0] == cx[1]
    return rotations[0], cx[0]


reusable_counts = gate_counts(angle_program)
print("input      from_angles (rot, CX)   fixed (rot, CX)")
for name, program in fixed_programs.items():
    print(f"{name:10} {str(reusable_counts):23} {gate_counts(program)}")

# %% [markdown]
# 角度を渡す回路は3入力とも回転6個・`CX` 4個です。振幅を直接渡す回路では、正の実数例だけ $R_z$ の回転列を省略できるため、回転3個・`CX` 2個になります。
#
# ## まとめ
#
# 入力を切り替えながら同じ回路を使うには、`amplitude_angles` で計算した角度を `amplitude_encoding_from_angles` の回路に渡します。この量子カーネルは回転角を実行時のパラメータとして受け取るため、$R_z$ の角度がすべてゼロの入力でも、回路には $R_z$ ゲートが含まれます。
#
# 一方、入力が決まっている場合は、`amplitude_encoding` に振幅を直接渡せます。計算した $R_z$ の角度がすべてゼロなら、その回転列が省かれ、ゲート数が少なくなります。
