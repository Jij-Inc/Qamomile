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
# # Möttönen振幅エンコーディング
#
# このノートブックでは、Qamomileの明示的なMöttönen構成を使って実数および複素数の振幅ベクトルを準備し、得られる状態とゲート数を検証します。また、汎用の振幅エンコーディングではなく、手法を指定するAPIを選ぶべき場合について説明します。

# %%
# pipで最新のQamomileをインストールします！
# # !pip install qamomile

# %%
import numpy as np
from qiskit.quantum_info import Statevector

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.stdlib import (
    amplitude_encoding,
    mottonen_amplitude_encoding,
    mottonen_amplitude_encoding_from_angles,
)
from qamomile.linalg import (
    compute_mottonen_amplitude_encoding_ry_angles,
    compute_mottonen_amplitude_encoding_rz_angles,
)
from qamomile.qiskit import QiskitTranspiler

# %% [markdown]
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
# Möttönenの振幅エンコーディングはこの状態準備を実現する一つのアルゴリズムです。一方で、このような状態準備を実現するアルゴリズムは他にも存在するため、Qamomileでは明示的にMöttönenの手法を指定するAPIと、方式は問わないが状態準備を行うAPIの二つを用意しています。
#
# - `mottonen_amplitude_encoding(...)`はQamomileのMöttönen構成を明示的に要求します。分解方法、リソース数、または合成方法がプログラムの意図に含まれる場合に使用します。
# - `amplitude_encoding(...)`は目標状態だけを表します。バックエンドは固有の状態準備操作を利用できます。バックエンド固有の状態準備が使われない場合、Qamomileは現在Möttönen構成を移植可能な実装として使います。ただし、この実装方式は汎用APIの保証ではありません。
#
# 明示的な実装は、Möttönen、Vartiainen、Bergholm、Salomaaによる一様制御回転の構成{cite:p}`10.48550/arXiv.quant-ph/0407010`に従います。論文では、より一般的な任意状態変換$|a\rangle \to |b\rangle$を扱います。Qamomileでは入力を$|0\rangle^{\otimes n}$に固定し、状態準備に相当する片側を実装しています。
#
# :::{note}
# 入力量子ビットが$|0\rangle^{\otimes n}$でない場合、これらのAPIは一般に出力状態を保証しません。
# :::

# %% [markdown]
# ## 問題設定
#
# 2量子ビットを使い、準備した状態ベクトルを次の3つの正規化された目標と比較します。
#
# - 正の実数ベクトル$(1, 2, 3, 4)$
# - 符号付き実数ベクトル$(1, -1, 1, -1)$
# - 複素数ベクトル$(1, 1+i, 1-i, 2i)$
#
# 状態忠実度は位相に依存しないため、バックエンドが異なるグローバル位相を選んだ場合でも、意図した物理状態を検証できます。

# %%
transpiler = QiskitTranspiler()
executor = transpiler.executor()

ATOL_STATEVECTOR = 1e-8
ATOL_SHOT = 0.05


def fidelity(prepared: np.ndarray, target: np.ndarray) -> float:
    """2つの状態ベクトル間の位相不変な忠実度を返します。"""
    return float(np.abs(np.vdot(prepared, target)) ** 2)


def normalize(amplitudes: list[float] | list[complex]) -> np.ndarray:
    """振幅ベクトルを単位ノルムに正規化したコピーを返します。"""
    if any(isinstance(value, complex) for value in amplitudes):
        array = np.asarray(amplitudes, dtype=complex)
    else:
        array = np.asarray(amplitudes, dtype=float)
    return array / np.linalg.norm(array)


def statevector_of(kernel: qmc.QKernel, **bindings) -> np.ndarray:
    """量子カーネルをトランスパイルし、最終状態ベクトルを返します。"""
    circuit = transpiler.to_circuit(kernel, bindings=bindings or None)
    stripped = circuit.remove_final_measurements(inplace=False)
    assert stripped is not None
    return Statevector.from_instruction(stripped).data


amps_real = [1.0, 2.0, 3.0, 4.0]
amps_signed = [1.0, -1.0, 1.0, -1.0]
amps_complex = [1 + 0j, 1 + 1j, 1 - 1j, 0 + 2j]

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

# %% [markdown]
# ## 実装
#
# ### トランスパイル時に振幅をバインドする
#
# 実数または符号付き実数の振幅ベクトルは、量子カーネルの`Vector[Float]`引数として宣言できます。トランスパイル時に`bindings`で具体値を渡すと、Qamomileは振幅を自動的に正規化し、Möttönen角度とゲート列を生成します。


# %%
@qmc.qkernel
def prepare_from_amplitudes(
    amplitudes: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q = mottonen_amplitude_encoding(q, amplitudes)
    return qmc.measure(q)


prepare_from_amplitudes.draw(fold_loops=False, amplitudes=amps_real)

# %% [markdown]
# 展開前の回路では、手法を指定する操作を1つのコンポジットゲートとして保持します。`expand_composite=True`を渡すと、基本的な$R_y$、$R_z$、`CNOT`の列を確認できます。実数回路には$R_z$位相カスケードがありません。

# %%
prepare_from_amplitudes.draw(
    fold_loops=False,
    expand_composite=True,
    amplitudes=amps_real,
)

# %% [markdown]
# `Vector[Float]`は実数のみを保持するため、複素数の振幅ベクトルは直接バインドできません。複素数の入力では、Möttönen角度を量子カーネルの外で計算し、以下の角度APIへ2本の`Vector[Float]`として渡します。
#
# また、`amplitudes`を`parameters`に残すことはできません。`atan2(|a_1|, |a_0|)`などの演算では、トレース時に具体的な値が必要になるためです。実行時に再バインドできる回路を作るには、Möttönen角度を事前に計算し、以下の角度APIを使用します。

# %%
try:
    transpiler.transpile(prepare_from_amplitudes, parameters=["amplitudes"])
except ValueError as exc:
    print(f"ValueError: {exc}")
    raised = True
else:
    raised = False
assert raised, "expected ValueError when amplitudes is a runtime parameter"

# %% [markdown]
# ### 実行時に再バインド可能な角度で一度だけコンパイルする
#
# `mottonen_amplitude_encoding_from_angles(...)`は、呼び出し側で計算したMöttönen角度を受け取ります。この角度は実行時パラメータとして保持できるため、1つのコンパイル済み回路で複数の振幅ベクトルを準備できます。この関数はコンポジット操作でラップせず、基本回転と`CNOT`ゲートを直接出力します。
#
# 以前の名前`amplitude_encoding_from_angles(...)`も互換ラッパーとして利用できます。渡す角度と出力されるゲート列はMöttönen固有であるため、新しいコードでは手法を明示する名前を使用してください。


# %%
@qmc.qkernel
def prepare_from_angles(
    ry_angles: qmc.Vector[qmc.Float], rz_angles: qmc.Vector[qmc.Float]
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q = mottonen_amplitude_encoding_from_angles(q, ry_angles, rz_angles)
    return qmc.measure(q)


complex_ry_angles = compute_mottonen_amplitude_encoding_ry_angles(amps_complex).tolist()
complex_rz_angles = compute_mottonen_amplitude_encoding_rz_angles(amps_complex).tolist()

prepare_from_angles.draw(
    fold_loops=False,
    ry_angles=complex_ry_angles,
    rz_angles=complex_rz_angles,
)

# %% [markdown]
# ### リソース確認用の量子カーネルを構築する
#
# 次のヘルパーは、複数のレジスタサイズに対して実数および複素数のMöttönen回路を作成します。明示的なAPIを使うことで、バックエンド固有の状態準備によって測定対象の構成が変わらないようにします。


# %%
def make_real_kernel(n_qubits: int) -> qmc.QKernel:
    """実数入力用のMöttönen状態準備量子カーネルを構築します。"""

    @qmc.qkernel
    def kernel(amplitudes: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(n_qubits, "q")
        q = mottonen_amplitude_encoding(q, amplitudes)
        return qmc.measure(q)

    return kernel


def make_complex_kernel(n_qubits: int) -> qmc.QKernel:
    """複素数入力用のMöttönen状態準備量子カーネルを構築します。"""

    @qmc.qkernel
    def kernel(
        ry_angles: qmc.Vector[qmc.Float],
        rz_angles: qmc.Vector[qmc.Float],
    ) -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(n_qubits, "q")
        q = mottonen_amplitude_encoding_from_angles(q, ry_angles, rz_angles)
        return qmc.measure(q)

    return kernel


# %% [markdown]
# ## 結果
#
# ### 状態忠実度
#
# 実数と符号付き実数は振幅の`bindings`を使い、複素数は角度の`bindings`を使って、グローバル位相を除いた正規化後の目標状態を再現できます。

# %%
for label, target_amplitudes in (
    ("実数", amps_real),
    ("符号付き実数", amps_signed),
):
    prepared = statevector_of(
        prepare_from_amplitudes,
        amplitudes=target_amplitudes,
    )
    target = normalize(target_amplitudes)
    value = fidelity(prepared, target)
    print(f"{label:>11s} 忠実度 = {value:.8f}")
    assert np.isclose(value, 1.0, atol=ATOL_STATEVECTOR)

prepared_complex = statevector_of(
    prepare_from_angles,
    ry_angles=complex_ry_angles,
    rz_angles=complex_rz_angles,
)
complex_fidelity = fidelity(prepared_complex, normalize(amps_complex))
print(f"{'複素数':>11s} 忠実度 = {complex_fidelity:.8f}")
assert np.isclose(complex_fidelity, 1.0, atol=ATOL_STATEVECTOR)

# %% [markdown]
# ### 実行時の再バインド
#
# コンパイル済みの2量子ビット複素数回路には、3つの$R_y$角度と3つの$R_z$角度からなる6つの実行時回転パラメータがあります。この回路を3つの目標ベクトルに再利用し、サンプリングした確率を$|a_i|^2$と比較します。

# %%
executable = transpiler.transpile(
    prepare_from_angles, parameters=["ry_angles", "rz_angles"]
)
n_runtime_params = len(executable.compiled_quantum[0].circuit.parameters)
print(f"コンパイル済み回路の実行時パラメータ数: {n_runtime_params}")
assert n_runtime_params == 2 * (2**2 - 1)

shots = 8192
for trial_amplitudes in (
    [1.0, 0.0, 0.0, 1.0],
    [3.0, 4.0, 0.0, 0.0],
    [1 + 0j, 1j, -1 + 0j, -1j],
):
    ry_values = compute_mottonen_amplitude_encoding_ry_angles(trial_amplitudes).tolist()
    rz_values = compute_mottonen_amplitude_encoding_rz_angles(trial_amplitudes).tolist()
    counts = (
        executable.sample(
            executor,
            shots=shots,
            bindings={"ry_angles": ry_values, "rz_angles": rz_values},
        )
        .result()
        .results
    )
    observed = np.zeros(4)
    for bits, count in counts:
        index = sum(int(bit) << i for i, bit in enumerate(bits))
        observed[index] = count / shots
    expected_probabilities = np.abs(normalize(trial_amplitudes)) ** 2
    max_deviation = float(np.max(np.abs(observed - expected_probabilities)))
    print(f"振幅={str(trial_amplitudes):<48s} max|p_obs - p_exp| = {max_deviation:.4f}")
    assert max_deviation < ATOL_SHOT

# %% [markdown]
# ### リソース式
#
# `estimate_resources()`はコンポジットのメタデータを展開し、テストしたすべてのレジスタサイズでMöttönenの式と一致します。

# %%
print(f"{'n':>3s} | {'実数(回転/CNOT)':>16s} | {'複素数(回転/CNOT)':>20s}")
print(f"{'---':>3s} | {'---':>16s} | {'---':>20s}")
for n_qubits in (2, 3, 4, 5):
    real_amplitudes = np.ones(2**n_qubits).tolist()
    complex_amplitudes = (np.ones(2**n_qubits) + 1j * np.arange(2**n_qubits)).tolist()
    real_resources = qmc.estimate_resources(
        make_real_kernel(n_qubits).build(amplitudes=real_amplitudes)
    )
    complex_resources = qmc.estimate_resources(
        make_complex_kernel(n_qubits).build(
            ry_angles=compute_mottonen_amplitude_encoding_ry_angles(
                complex_amplitudes
            ).tolist(),
            rz_angles=compute_mottonen_amplitude_encoding_rz_angles(
                complex_amplitudes
            ).tolist(),
        )
    )
    real_rotations = int(real_resources.gates.rotation_gates)
    real_cnots = int(real_resources.gates.two_qubit)
    complex_rotations = int(complex_resources.gates.rotation_gates)
    complex_cnots = int(complex_resources.gates.two_qubit)
    print(
        f"{n_qubits:>3d} | "
        f"{f'{real_rotations} / {real_cnots}':>16s} | "
        f"{f'{complex_rotations} / {complex_cnots}':>20s}"
    )

    assert real_rotations == 2**n_qubits - 1
    assert real_cnots == 2**n_qubits - 2
    assert complex_rotations == 2 * (2**n_qubits - 1)
    assert complex_cnots == 2 * (2**n_qubits - 2)

# %% [markdown]
# 回転と`CNOT`の数はどちらも量子ビット数$n$に対して$O(2^n)$で増加します。古典ベクトルが大きい場合、この回路コストの指数関数的増加は、振幅エンコーディングを選ぶ際の重要な制約です。

# %% [markdown]
# ### Observableとの組み合わせ
#
# 明示的な状態準備操作は、量子カーネル内の他の処理と組み合わせられます。Qamomileのlittle-endian順序で$(1, 2, 3, 4)$を用いると、次のようになります。
#
# $$
# \langle Z_0 \rangle
# = \frac{1 + 9 - 4 - 16}{30}
# = -\frac{1}{3}.
# $$


# %%
@qmc.qkernel
def expval_kernel(
    amplitudes: qmc.Vector[qmc.Float],
    observable: qmc.Observable,
) -> qmc.Float:
    q = qmc.qubit_array(2, "q")
    q = mottonen_amplitude_encoding(q, amplitudes)
    return qmc.expval(q, observable)


hamiltonian = qm_o.Z(0) + 0.0 * qm_o.Z(1)
expval_executable = transpiler.transpile(
    expval_kernel,
    bindings={"amplitudes": amps_real, "observable": hamiltonian},
)
expval_result = expval_executable.run(executor).result()
print(f"<Z_0> = {float(expval_result):+.6f}")
assert np.isclose(float(expval_result), -1.0 / 3.0, atol=ATOL_STATEVECTOR)

# %% [markdown]
# ## Qamomileの組み込み機能
#
# Qamomileは、汎用と手法指定の両方の状態準備ヘルパーを提供します。準備した状態だけがプログラムの規約に含まれるのか、合成方法も含まれるのかに応じて選択してください。
#
# | 目的 | API |
# |---|---|
# | 目標状態を準備し、バックエンド固有の合成を許可する | `amplitude_encoding(q, amplitudes)` |
# | QamomileのMöttönen構成を要求する | `mottonen_amplitude_encoding(q, amplitudes)` |
# | Möttönen方式を維持しながら、トランスパイル時に実数振幅をバインドする | `mottonen_amplitude_encoding(q, amps)`と`bindings={"amps": [...]}` |
# | 実行時の角度バインドにより1つのMöttönen回路を再利用する | `mottonen_amplitude_encoding_from_angles(q, ry, rz)`と`parameters=[...]` |
# | 移行中も既存コードを動作させる | `amplitude_encoding_from_angles(...)`（互換ラッパー） |
#
# 汎用と明示的な振幅APIは同じ目標状態を準備しますが、バックエンドは汎用操作だけを別の状態準備方法で実現できます。


# %%
@qmc.qkernel
def prepare_generic(
    amplitudes: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q = amplitude_encoding(q, amplitudes)
    return qmc.measure(q)


generic_state = statevector_of(prepare_generic, amplitudes=amps_real)
assert np.isclose(
    fidelity(generic_state, normalize(amps_real)), 1.0, atol=ATOL_STATEVECTOR
)

# %% [markdown]
# ## まとめ
#
# このノートブックでは、次のことを行いました。
#
# - 実数振幅とMöttönen角度を量子カーネル引数として渡し、実数状態と複素数状態の忠実度が1であることを確認しました。
# - Qamomileの構成における回転と`CNOT`の数が$O(2^n)$で増加することを検証しました。
# - 正式な`mottonen_amplitude_encoding_from_angles(...)` APIを使い、1つのコンパイル済み回路を再利用しました。
# - 手法に依存しない`amplitude_encoding(...)`と、Möttönen合成を意図に含める場合に使う手法固有のAPIを区別しました。
