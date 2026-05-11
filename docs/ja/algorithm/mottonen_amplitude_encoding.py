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
# **振幅エンコーディング**とは、与えられた単位ノルムの複素ベクトル$a \in \mathbb{C}^{2^n}$に対して、$|0\rangle^{\otimes n}$から$n$量子ビットの状態
#
# $$
# |\psi\rangle \;=\; \sum_{i=0}^{2^n - 1} a_i \, |i\rangle
# $$
#
# を準備する操作です。古典データを量子状態として読み込むあらゆるアルゴリズム — HHL系の線形方程式ソルバー、カーネル法、多くの量子シミュレーションプロトコルなど — の入口にあたります。Qamomileは`qamomile.circuit.algorithm.state_preparation`の下に、Möttönen, Vartiainen, Bergholm, Salomaaの一様制御回転構成（arXiv:quant-ph/0407010）に基づいたSDK移植可能な実装を提供しています。
#
# この構成は2段階からなります:
#
# 1. $n$個の**一様制御$R_y$**ゲートのカスケード。基底状態に振幅$|a_i|$を分配します。実数（符号付き）振幅ベクトルにはこの段だけで十分です。
# 2. **一様制御$R_z$**ゲートの2段目のカスケード。相対位相を復元します。入力に非ゼロの虚部が含まれる場合のみ発行されます。
#
# 各一様制御回転は、Möttönen-VartiainenのGray符号レシピを使って基本的な`RY` / `RZ`と`CNOT`ゲートに分解されます。総コストは次のとおりです:
#
# | 段 | 実数入力 | 複素入力 |
# |---|---:|---:|
# | $R_y$回転 | $2^n - 1$ | $2^n - 1$ |
# | $R_z$回転 | $0$ | $2^n - 1$ |
# | `CNOT` | $2^n - 2$ | $2 (2^n - 2)$ |
#
# このチュートリアルでは公開API表面を一通り見て、3つの入力モード（具体的なシーケンス、コンパイル時バインドの`Vector[Float]`、ランタイムパラメトリックな角度）を実演し、各モードがどこで活きるかを示します。

# %%
import numpy as np
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info import Statevector

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm import (
    MottonenAmplitudeEncoding,
    amplitude_encoding,
    amplitude_encoding_from_angles,
    compute_mottonen_amplitude_encoding_ry_angles,
    compute_mottonen_amplitude_encoding_rz_angles,
)
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()
executor = transpiler.executor()


def fidelity(prepared: np.ndarray, target: np.ndarray) -> float:
    """位相不変なフィデリティ ``|<prepared|target>|^2``。"""
    return float(np.abs(np.vdot(prepared, target)) ** 2)


def normalize(amps: list[float] | list[complex]) -> np.ndarray:
    """単位ノルム化した *amps* のコピー（任意の要素が複素数なら複素dtype）。"""
    if any(isinstance(x, complex) for x in amps):
        arr = np.asarray(amps, dtype=complex)
    else:
        arr = np.asarray(amps, dtype=float)
    return arr / np.linalg.norm(arr)


# %% [markdown]
# ## 1. もっとも単純な呼び出し — 具体的な実数振幅
#
# `amplitude_encoding(qubits, amplitudes)`が日常的な入口です。Pythonのシーケンスやnumpy配列を受け取り、自動的に正規化して対応する状態を準備します。
#
# 最初の確認として、未正規化のベクトル$a = (1, 2, 3, 4)$を2量子ビットレジスタにエンコードし、シミュレータの状態ベクトルを読み戻します。

# %%
amps_real = [1.0, 2.0, 3.0, 4.0]


@qmc.qkernel
def prepare_real() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q = amplitude_encoding(q, amps_real)
    return qmc.measure(q)


qc = transpiler.to_circuit(prepare_real)
sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False)).data
expected = normalize(amps_real)
print(f"prepared      = {np.round(sv, 4)}")
print(f"target (norm) = {np.round(expected, 4)}")
print(f"fidelity      = {fidelity(sv, expected):.6f}")

# %% [markdown]
# 負の実数振幅は magnitude 段を自然に通過します — 葉レベルの$R_y$角は符号付きの`arctan2`として取られるため、追加の位相段なしで符号が捕捉されます。したがって状態$a = (1, -1, 1, -1)$は`RY`と`CNOT`のみで準備されます。

# %%
amps_signed = [1.0, -1.0, 1.0, -1.0]


@qmc.qkernel
def prepare_signed() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q = amplitude_encoding(q, amps_signed)
    return qmc.measure(q)


qc = transpiler.to_circuit(prepare_signed)
sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False)).data
expected = normalize(amps_signed)
print(f"fidelity (signed) = {fidelity(sv, expected):.6f}")

# %% [markdown]
# ## 2. 複素振幅
#
# 同じAPIが複素入力を受け取ります。少なくとも1つの要素が非ゼロの虚部を持つ場合、実装は2段（Ry + Rz）構成に自動的に切り替わります。虚部が恒等的にゼロの複素ベクトルは、より安価な実数経路に静かに変換されます。
#
# $a = (1, 1+i, 1-i, 2i)$ — 一般的な複素2量子ビット状態 — をエンコードし、結果の振幅が（大域位相を除いて）一致することを確認します。

# %%
amps_complex = [1 + 0j, 1 + 1j, 1 - 1j, 0 + 2j]


@qmc.qkernel
def prepare_complex() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q = amplitude_encoding(q, amps_complex)
    return qmc.measure(q)


qc = transpiler.to_circuit(prepare_complex)
sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False)).data
expected = normalize(amps_complex)
print(f"fidelity (complex) = {fidelity(sv, expected):.6f}")

# %% [markdown]
# ## 3. ゲートバジェットの確認
#
# `MottonenAmplitudeEncoding`は`_resources()`メソッドを公開しており、Gray-walk分解で予測される各ゲート数を返します。単段経路（実数入力）と2段経路（複素入力）は別々に報告されます。

# %%
gate_real = MottonenAmplitudeEncoding(amps_real)
gate_complex = MottonenAmplitudeEncoding(amps_complex)

for label, gate in (("real", gate_real), ("complex", gate_complex)):
    meta = gate._resources().custom_metadata
    print(
        f"{label:7s}: RY={meta['num_ry_gates']:>3d}  RZ={meta['num_rz_gates']:>3d}"
        f"  CNOT={meta['num_cnot_gates']:>3d}  complex_input={meta['complex_input']}"
    )

# %% [markdown]
# どちらの数も$O(2^n)$で増加します。振幅エンコーディングは多量子ビットで本質的に高価です。実用上、この構成は大きいアルゴリズム内のビルディングブロックとして遭遇する小さなレジスタサイズ（HHLで4〜8論理量子ビットの入力レジスタ、サンプリングされた部分空間を持つQSCI、誤り訂正のwarm-startsなど）で最も有用であり、数百量子ビットのスタンドアロンの状態準備としてではありません。

# %% [markdown]
# ## 4. 3つの入力モード
#
# `amplitude_encoding`は3つの異なる形式で振幅を受け取ります。`amplitude_encoding_from_angles`は事前計算済みの角度を公開する4つ目のモードを追加し、同じコンパイル済み回路をランタイムで再バインドできるようにします。どれを選ぶかは、値が**いつ**判明するかと、それをカーネルに**どう**渡したいかに依存します。
#
# | モード | 入力の所在 | 値が必要なタイミング | 再バインド |
# |---|---|---|---|
# | A. クロージャ | Python（外側スコープ） | trace時 | 振幅ごとに再コンパイル |
# | B. `bindings={"amps": ...}`で`Vector[Float]` | カーネルパラメータ | trace時（バインドメタデータから抽出） | 振幅ごとに再コンパイル |
# | C. `amplitude_encoding_from_angles` + `bindings` | カーネルパラメータ | trace時 | 角度ベクトルごとに再コンパイル |
# | D. `amplitude_encoding_from_angles` + `parameters` | カーネルパラメータ | **ランタイム** | 再コンパイルなしで再バインド |
#
# モードA〜Cは「振幅がコンパイル時に判明している」をいろいろな形で表現したものです。Dは内部ループの中で異なる振幅をまたいでコンパイル済み回路を再利用できる唯一のモードです。
#
# 以下で各モードを実演します。

# %% [markdown]
# ### モードA — クロージャ（既出の呼び出し）
#
# 上のすべての例で使ったのがこれです。振幅は外側スコープにPythonリストとして存在し、カーネルがそれをクロージャで取り込みます。カーネルと振幅がコード上で近くにある場合に最適です。

# %% [markdown]
# ### モードB — `Vector[Float]`パラメータ、コンパイル時にバインド
#
# 振幅をカーネルパラメータとして公開したい場合（ドキュメント目的、異なるベクトルを掃引する目的、あるいはカーネル定義をマジックナンバーから解放するため）は、パラメータを`Vector[Float]`として宣言し、`bindings={...}`で値を渡します。実装はtrace時にハンドルの`array_runtime_metadata`からバインド済みの具体データを読み出すため、角度計算は依然として古典的に走り、IRには単一の`MottonenAmplitudeEncoding`コンポジットゲートが残ります。

# %%
@qmc.qkernel
def prepare_via_binding(amps: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q = amplitude_encoding(q, amps)
    return qmc.measure(q)


qc = transpiler.to_circuit(
    prepare_via_binding, bindings={"amps": [1.0, 2.0, 3.0, 4.0]}
)
sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False)).data
print(f"fidelity (mode B) = {fidelity(sv, normalize([1.0, 2.0, 3.0, 4.0])):.6f}")

# %% [markdown]
# `parameters=["amps"]`でパラメータをsymbolicに残そうとすると、方向付きエラーで拒否されます — 角度計算（`atan2(|a_1|, |a_0|)`など）には具体的な数値が必要なので、ランタイムへ遅延させることは本質的にできません。エラーメッセージはランタイムケース用に`amplitude_encoding_from_angles`を指し示します。

# %%
try:
    transpiler.transpile(prepare_via_binding, parameters=["amps"])
except ValueError as exc:
    print(f"ValueError: {exc}")

# %% [markdown]
# ### モードC — 事前計算済みの角度、コンパイル時にバインド
#
# Möttönen構成の古典部分は2つのヘルパーです: `compute_mottonen_amplitude_encoding_ry_angles(amps)`と`compute_mottonen_amplitude_encoding_rz_angles(amps)`。それぞれ長さ$2^n - 1$のGray-walk順$R_y$および$R_z$角度を返します（$R_z$配列は実数入力に対して恒等的にゼロ）。
#
# `amplitude_encoding_from_angles`は振幅ではなくこれらの角度を受け取るコンパニオン関数です。`bindings={...}`と一緒に使うとモードBによく似た振る舞いをします — ただし`MottonenAmplitudeEncoding`コンポジットゲートで**ラップしない**点が異なります。IRには基本的な`RY` / `RZ` / `CNOT`ゲートが直接乗り、これが下のランタイムモードを可能にしています。

# %%
ry_angles = compute_mottonen_amplitude_encoding_ry_angles(amps_complex).tolist()
rz_angles = compute_mottonen_amplitude_encoding_rz_angles(amps_complex).tolist()


@qmc.qkernel
def prepare_from_angles(
    ry_a: qmc.Vector[qmc.Float], rz_a: qmc.Vector[qmc.Float]
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q = amplitude_encoding_from_angles(q, ry_a, rz_a)
    return qmc.measure(q)


qc = transpiler.to_circuit(
    prepare_from_angles, bindings={"ry_a": ry_angles, "rz_a": rz_angles}
)
sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False)).data
print(f"fidelity (mode C) = {fidelity(sv, normalize(amps_complex)):.6f}")

# %% [markdown]
# ### モードD — ランタイムパラメトリックな角度
#
# `amplitude_encoding_from_angles`が存在する理由はこのモードを実現するためです。`parameters=[...]`でtranspileすることで、出力された回路上で角度ベクトルがsymbolicに保たれ、**同じコンパイル済み回路**を`executable.sample(bindings={...})`経由で多くの異なる振幅ベクトルに再バインドできます。これがハイブリッドループ（例: 振幅に対する古典最適化）の正しいパターンであり、ここで反復ごとの再コンパイルが支配的になるのを避けられます。

# %%
exe = transpiler.transpile(prepare_from_angles, parameters=["ry_a", "rz_a"])
n_runtime_params = len(exe.compiled_quantum[0].circuit.parameters)
print(f"runtime parameters in compiled circuit: {n_runtime_params}")

shots = 8192
for trial_amps in (
    [1.0, 0.0, 0.0, 1.0],
    [3.0, 4.0, 0.0, 0.0],
    [1 + 0j, 1j, -1 + 0j, -1j],
):
    ry = compute_mottonen_amplitude_encoding_ry_angles(trial_amps).tolist()
    rz = compute_mottonen_amplitude_encoding_rz_angles(trial_amps).tolist()
    counts = (
        exe.sample(executor, shots=shots, bindings={"ry_a": ry, "rz_a": rz})
        .result()
        .results
    )
    observed = np.zeros(4)
    for bits, c in counts:
        idx = sum(int(b) << i for i, b in enumerate(bits))
        observed[idx] = c / shots
    expected_probs = np.abs(normalize(trial_amps)) ** 2
    print(
        f"amps={str(trial_amps):<48s}  "
        f"max|p_obs - p_exp| = {np.max(np.abs(observed - expected_probs)):.4f}"
    )

# %% [markdown]
# 3回の反復はすべて同じコンパイル済み回路からサンプリングしています。変化するのはランタイムバインディングだけです。ビンごとの最大偏差は多項分布のショットノイズの数標準偏差以内に収まります。

# %% [markdown]
# ## 5. エンコード状態に対する観測量推定
#
# `amplitude_encoding`はビルディングブロックです — ほとんどのユーザはこれを大きいカーネルに組み込みます。最も単純な使い方は、準備された状態に対するハミルトニアン$H$の期待値$\langle \psi | H | \psi \rangle$の計算です。カーネルは1回の`expval`になり、観測量はランタイムバインディングとして渡せます。
#
# 小さな解析的チェックとして、$a = (1, 2, 3, 4)$（リトルエンディアン、qubit $0$ = LSB）に対するエンコード状態は
#
# $$
#   \langle Z_0 \rangle
#   = (p_{00} + p_{10}) - (p_{01} + p_{11})
#   = \frac{1 + 9 - 4 - 16}{30}
#   = -\tfrac{1}{3},
# $$
#
# となり、estimator経路でこれを再現します。

# %%
@qmc.qkernel
def expval_kernel(H: qmc.Observable) -> qmc.Float:
    q = qmc.qubit_array(2, "q")
    q = amplitude_encoding(q, [1.0, 2.0, 3.0, 4.0])
    return qmc.expval(q, H)


H = qm_o.Z(0) + 0.0 * qm_o.Z(1)  # 2量子ビット幅にパディング
exe_expval = transpiler.transpile(expval_kernel, bindings={"H": H})
result = exe_expval.run(executor).result()
print(f"<Z_0> = {float(result):+.6f}   (analytic: {-1/3:+.6f})")

# %% [markdown]
# ## どれをいつ使うか
#
# - **Pythonで具体的な振幅が判明している** → `amplitude_encoding`をlist/arrayで直接呼ぶ。モードA。
# - **明確化のために振幅をカーネルパラメータにしたい** → `Vector[Float]`として宣言し、`bindings={...}`でバインドする。モードB。
# - **ハイブリッド最適化ループ、再コンパイルコストが効いてくる** → `amplitude_encoding_from_angles`を`parameters=[...]`で呼び、ループ内で角度を事前計算する。モードD。
# - **リソース推定** → transpile前に`MottonenAmplitudeEncoding._resources()`で閉形式のゲート数を取得する。
#
# $O(2^n)$回転およびCNOTという指数的なゲート増加はMöttönen構成の本質的なコストです。より大きいレジスタには専用構成（例: 入力の低ランク近似や、Qiskitの`StatePreparation`のようなSDKネイティブのプリミティブ）が通常好まれます — Qamomileのコンポジットゲート機構は、将来`CompositeGateEmitter`を介してそのようなネイティブ経路へディスパッチする余地を残していますが、上の基本分解は移植可能なフォールバックとして常に利用できます。
