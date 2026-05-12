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
# を準備する操作です。古典データを量子状態として読み込むあらゆるアルゴリズム — HHL系の線形方程式ソルバー、カーネル法、多くの量子シミュレーションプロトコルなど — の入口にあたります。Qamomileは`qamomile.circuit.algorithm.state_preparation`の下に、Möttönen, Vartiainen, Bergholm, Salomaaの一様制御回転構成 {cite:p}`10.48550/arXiv.quant-ph/0407010` に基づいたSDK移植可能な実装を提供しています。
#
# この構成は2段階からなります:
#
# 1. $n$個の**一様制御$R_y$ゲート**のカスケード。基底状態に振幅$|a_i|$を分配します。実数（符号付き）振幅ベクトルにはこの段だけで十分です。
# 2. **一様制御$R_z$ゲート**の2段目のカスケード。相対位相を復元します。入力に非ゼロの虚部が含まれる場合のみ発行されます。
#
# 各一様制御回転は、Möttönen-VartiainenのGray符号レシピを使って基本的な`RY` / `RZ`と`CNOT`ゲートに分解されます。総コストは次のとおりです:
#
# | 段 | 実数入力 | 複素入力 |
# |---|---:|---:|
# | $R_y$回転 | $2^n - 1$ | $2^n - 1$ |
# | $R_z$回転 | $0$ | $2^n - 1$ |
# | `CNOT` | $2^n - 2$ | $2 (2^n - 2)$ |
#
# このチュートリアルでは公開API表面を一通り見て、各エントリポイントを動作確認しながら、最後に「どの場面でどれを選ぶか」を一覧できる比較表を示します。

# %%
import numpy as np
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

# 以下のすべての fidelity / 確率チェックで使う許容値。
# docs build 時にサイレントなリグレッションを検出できるくらいタイトに、
# sampler 比較ではショットノイズを吸収できるくらいゆるく取っています。
ATOL_STATEVECTOR = 1e-8
ATOL_SHOT = 0.05  # 8192 shots, p(1-p)/N に対しおよそ5σ


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


def statevector_of(kernel: qmc.QKernel, **bindings) -> np.ndarray:
    """*kernel*をQiskitのstatevectorシミュレータで実行してデータを返す。"""
    qc = transpiler.to_circuit(kernel, bindings=bindings or None)
    return Statevector.from_instruction(
        qc.remove_final_measurements(inplace=False)
    ).data


# %% [markdown]
# ## 1. もっとも単純な呼び出し — 具体的な実数振幅
#
# `amplitude_encoding(qubits, amplitudes)`が日常的な入口です。Pythonのシーケンスやnumpy配列を受け取り、自動的に正規化して対応する状態を準備します。
#
# 最初の確認として、未正規化のベクトル$a = (1, 2, 3, 4)$を2量子ビットレジスタにエンコードし、シミュレータの状態ベクトルを読み戻して、正規化されたターゲットと（位相を除いて）一致することを assert します。

# %%
amps_real = [1.0, 2.0, 3.0, 4.0]


@qmc.qkernel
def prepare_real() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q = amplitude_encoding(q, amps_real)
    return qmc.measure(q)


sv = statevector_of(prepare_real)
expected = normalize(amps_real)
print(f"prepared      = {np.round(sv, 4)}")
print(f"target (norm) = {np.round(expected, 4)}")
print(f"fidelity      = {fidelity(sv, expected):.6f}")
assert np.isclose(fidelity(sv, expected), 1.0, atol=ATOL_STATEVECTOR), (
    "実振幅エンコーディングのfidelityが落ちています"
)

# %% [markdown]
# 負の実数振幅は magnitude 段を自然に通過します — 葉レベルの$R_y$角は符号付きの`arctan2`として取られるため、追加の位相段なしで符号が捕捉されます。したがって状態$a = (1, -1, 1, -1)$は`RY`と`CNOT`のみで準備されます。

# %%
amps_signed = [1.0, -1.0, 1.0, -1.0]


@qmc.qkernel
def prepare_signed() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q = amplitude_encoding(q, amps_signed)
    return qmc.measure(q)


sv = statevector_of(prepare_signed)
expected = normalize(amps_signed)
print(f"fidelity (signed) = {fidelity(sv, expected):.6f}")
assert np.isclose(fidelity(sv, expected), 1.0, atol=ATOL_STATEVECTOR), (
    "符号付き実数エンコーディングのfidelityが落ちています"
)

# %% [markdown]
# ## 2. 複素振幅
#
# 同じAPIが複素入力を受け取ります。少なくとも1つの要素が非ゼロの虚部を持つ場合、実装は2段（Ry + Rz）構成に自動的に切り替わります。虚部が恒等的にゼロの複素ベクトルは、より安価な実数経路に静かに変換されます。
#
# $a = (1, 1+i, 1-i, 2i)$ — 一般的な複素2量子ビット状態 — をエンコードし、結果の状態ベクトルが（大域位相を除いて）一致することをassertします。

# %%
amps_complex = [1 + 0j, 1 + 1j, 1 - 1j, 0 + 2j]


@qmc.qkernel
def prepare_complex() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q = amplitude_encoding(q, amps_complex)
    return qmc.measure(q)


sv = statevector_of(prepare_complex)
expected = normalize(amps_complex)
print(f"fidelity (complex) = {fidelity(sv, expected):.6f}")
assert np.isclose(fidelity(sv, expected), 1.0, atol=ATOL_STATEVECTOR), (
    "複素エンコーディングのfidelityが落ちています"
)

# %% [markdown]
# ## 3. 可視化とリソース推定
#
# カーネルが`amplitude_encoding`を使うと、IR全体ではエンコード処理が単一の`MottonenAmplitudeEncoding`コンポジットゲートとして残ります。Qamomile側の検査APIが2つそのまま機能します:
#
# * `kernel.draw(fold_loops=False)`でIRを描画。`fold_loops=False`を渡すことで、ゲート内部の`qmc.range`がアンロールされた図が得られます（デフォルトの`True`では各ループが1ブロックに折り畳まれます）。
# * `kernel.estimate_resources()`はIRを歩き、量子ビット数とゲート数の内訳を持つ`ResourceEstimate`を返します。コンポジットゲート展開後の基本RY / RZ / CNOTのカウントを含みます。これがゲートコスト確認のサポートされた方法であり、コンポジットゲート機構を抽象化してくれるので、ユーザコードが`_resources().custom_metadata`を直接見る必要はありません。

# %%
prepare_real.draw(fold_loops=False)

# %%
prepare_complex.draw(fold_loops=False)

# %%
est_real = prepare_real.estimate_resources()
est_complex = prepare_complex.estimate_resources()

print(
    f"real    : qubits={est_real.qubits}  total={est_real.gates.total}  "
    f"single={est_real.gates.single_qubit}  two={est_real.gates.two_qubit}  "
    f"rotations={est_real.gates.rotation_gates}  clifford={est_real.gates.clifford_gates}"
)
print(
    f"complex : qubits={est_complex.qubits}  total={est_complex.gates.total}  "
    f"single={est_complex.gates.single_qubit}  two={est_complex.gates.two_qubit}  "
    f"rotations={est_complex.gates.rotation_gates}  clifford={est_complex.gates.clifford_gates}"
)

# 閉形式の参照値 (n = 2 qubits):
#   real    Ry: 2^n - 1 = 3,  CNOT: 2^n - 2 = 2,  total = 5
#   complex Ry: 3,  Rz: 3,  CNOT: 2 * (2^n - 2) = 4,  total = 10
assert int(est_real.qubits) == 2
assert int(est_real.gates.total) == 5
assert int(est_real.gates.rotation_gates) == 3
assert int(est_real.gates.two_qubit) == 2
assert int(est_complex.qubits) == 2
assert int(est_complex.gates.total) == 10
assert int(est_complex.gates.rotation_gates) == 6
assert int(est_complex.gates.two_qubit) == 4

# %% [markdown]
# どちらの数も$O(2^n)$で増加します。振幅エンコーディングは多量子ビットで本質的に高価です。実用上、この構成は大きいアルゴリズム内のビルディングブロックとして遭遇する小さなレジスタサイズ（HHLで4〜8論理量子ビットの入力レジスタ、サンプリングされた部分空間を持つQSCI、誤り訂正のwarm-startsなど）で最も有用であり、数百量子ビットのスタンドアロンの状態準備としてではありません。

# %% [markdown]
# ### 論文の公式値とのゲート数照合
#
# Möttönen, Vartiainen, Bergholm, Salomaa {cite:p}`10.48550/arXiv.quant-ph/0407010` はGray符号分解の閉形式を明示しています (Lemma 5, Section 3): $k$個のcontrolを持つuniformly controlled rotation は $2^k$個の基本回転と $2^k$個のCNOTに分解される。振幅エンコーディングのカスケードを構成する $n$ ステージ — $k = 0, 1, \ldots, n-1$ で stage $0$ は uncontrolled (したがって CNOT なし) — について和を取ると:
#
# | 入力     | 回転数               | CNOT数                  |
# |----------|---------------------:|-----------------------:|
# | 実数     | $2^n - 1$            | $2^n - 2$              |
# | 複素数   | $2 \cdot (2^n - 1)$  | $2 \cdot (2^n - 2)$    |
#
# `kernel.estimate_resources()`がこの値を正確に報告することを、複数のレジスタサイズに対して確認します。これは `MottonenAmplitudeEncoding._resources()` のメタデータを直接見るのではなく、IRを歩いてコンポジットゲートを解決するフルの推定経路を通るため、テスト的にもより強い保証になります。


# %%
def make_real_kernel(n: int) -> qmc.QKernel:
    """``n``量子ビットで実振幅Möttönen経路を走らせるカーネル。"""
    real_amps = np.ones(2**n).tolist()

    @qmc.qkernel
    def kernel() -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(n, "q")
        q = amplitude_encoding(q, real_amps)
        return qmc.measure(q)

    return kernel


def make_complex_kernel(n: int) -> qmc.QKernel:
    """複素 (Ry+Rz) Möttönen経路を走らせるカーネル。"""
    cplx_amps = (np.ones(2**n) + 1j * np.arange(2**n)).tolist()

    @qmc.qkernel
    def kernel() -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(n, "q")
        q = amplitude_encoding(q, cplx_amps)
        return qmc.measure(q)

    return kernel


print(f"{'n':>3s} | {'real(rot/CNOT)':>16s} | {'complex(rot/CNOT)':>20s}")
print(f"{'---':>3s} | {'---':>16s} | {'---':>20s}")
for n in (2, 3, 4, 5):
    er = make_real_kernel(n).estimate_resources()
    ec = make_complex_kernel(n).estimate_resources()
    rot_real, cnot_real = int(er.gates.rotation_gates), int(er.gates.two_qubit)
    rot_cplx, cnot_cplx = int(ec.gates.rotation_gates), int(ec.gates.two_qubit)
    print(
        f"{n:>3d} | {f'{rot_real} / {cnot_real}':>16s} | {f'{rot_cplx} / {cnot_cplx}':>20s}"
    )

    # Möttönen-Vartiainen の閉形式を直接 assert:
    assert rot_real == 2**n - 1, f"実数の回転数が公式値と不一致 (n={n})"
    assert cnot_real == 2**n - 2, f"実数のCNOT数が公式値と不一致 (n={n})"
    assert rot_cplx == 2 * (2**n - 1), f"複素の回転数が公式値と不一致 (n={n})"
    assert cnot_cplx == 2 * (2**n - 2), f"複素のCNOT数が公式値と不一致 (n={n})"

# %% [markdown]
# 上記は **基本** のMöttönen-Vartiainen Gray符号カウントです。同じ論文の後半および後続研究では、ステージ間のCNOTキャンセルにより複素の場合の最適漸近コストを $2^{n+1} - 2n$ まで下げる方法が記述されています。Qamomileの実装は明確さと移植性のため、これらのキャンセル最適化を適用していません — 各ステージごとの素直な分解で留めています。したがって上のassertが正しい参照値です。

# %% [markdown]
# ## 4. 公開API表面 — どの場面でどれを使うか
#
# state_preparationパッケージは5つの公開名を提供します。下表は「どの仕事にどれを使うか」と「それぞれのトレードオフ」をまとめたものです。
#
# | API | こういう時に使う | メリット | デメリット |
# |---|---|---|---|
# | `amplitude_encoding(q, amplitudes)` (具体シーケンス) | 振幅がカーネル定義近傍のPythonリスト・numpy配列として手元にある場合。 | 呼び出しがもっとも単純。IRには単一の`MottonenAmplitudeEncoding`コンポジットゲートが残るので、リソース推定や将来のbackend native dispatch (例: `CompositeGateEmitter`経由のQiskit `StatePreparation`) の余地が保たれる。 | 振幅を変えるたびに`transpile()`が必要。 |
# | `amplitude_encoding(q, amps)` + `bindings={"amps": [...]}` | 振幅をカーネルパラメータ`amps: Vector[Float]`として宣言したい（ドキュメント目的、掃引、マジックナンバー回避）。値はコンパイル時に解決する。 | IRの形と利点は具体シーケンス版と同じ。Möttönen角度を事前計算する必要なし。実装は`amps.value.get_const_array()`でtrace時にバインド済み具体データを読み出す。 | 実数のみ（`Vector[Float]`は複素数を運べない）。`parameters=["amps"]`は意図的に拒否される — 以下の角度ベースのrun-time pathを参照。 |
# | `amplitude_encoding_from_angles(q, ry_angles, rz_angles=None)` + `bindings={...}` | Möttönen角度を既に事前計算済みで（複数カーネルで共有など）、コンパイル時にバインドしたい場合。 | コンパイル時バインド、再コンパイルコストは上の経路と同じ。実数 (`rz_angles=None`) でも複素入力でも動く。 | `MottonenAmplitudeEncoding`コンポジットゲートのラッピングをスキップし、基本`RY` / `RZ` / `CNOT`ゲートを直接IRに発行する — リソース推定は高レベルopではなく基本ゲート列を見ることになる。 |
# | `amplitude_encoding_from_angles(q, ry_a, rz_a)` + `parameters=[...]` | **同じコンパイル済み回路** をrun-timeで多くの異なる振幅ベクトルに再バインドしたい場合（ハイブリッド最適化ループ、パラメータ掃引）。 | 1度コンパイル、`executable.sample(bindings={...})`で何度でもサンプル — 支配的なコストが再コンパイルでなくなる。runtime symbolic 角度をサポートする唯一の経路。 | 呼び出し側 (ユーザコード) が反復ごとに`compute_mottonen_amplitude_encoding_*_angles(...)`を呼んで振幅 → 角度の変換をする必要がある。同じflat-IRの caveat も該当。 |
# | `MottonenAmplitudeEncoding(amplitudes)` (クラス) | コンポジットゲートをファーストクラスのオブジェクトとして欲しい場合 — 典型的にはカーネル外でのリソース推定や、カスタム分解戦略への組み込み。 | `gate.num_target_qubits`やIR側のコンポジットゲートオブジェクトに直接アクセスできる。 | ほとんどのユーザコードは構築をラップする`amplitude_encoding`の方が使いやすい。 |
# | `compute_mottonen_amplitude_encoding_ry_angles(amps)` / `compute_mottonen_amplitude_encoding_rz_angles(amps)` | Gray-walk Ry / Rz角度を古典的に得たい場合 — 上のrun-time parametric pathに食わせる、複数カーネル間でキャッシュする、角度値そのものを推論したいなど。 | 純粋なPython / NumPyの前処理として高速。角度計算をカーネルbuildの外側に保てる。実数入力ではRzヘルパは常にゼロ配列を返す。 | 返ってくる配列が既にbit-reverse + $M^{(k)}$ Gray-walk変換済みであることを理解しておく必要あり — 生のper-control-state角度ではない。 |
#
# 次の2セルでは、上記表のうち非自明な`bindings` / `parameters`の入口を実際に動かして見せます。

# %% [markdown]
# ### `amplitude_encoding`にbound `Vector[Float]`パラメータを渡す


# %%
@qmc.qkernel
def prepare_via_binding(amps: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q = amplitude_encoding(q, amps)
    return qmc.measure(q)


prepare_via_binding.draw(fold_loops=False, amps=[1.0, 2.0, 3.0, 4.0])

# %%
sv = statevector_of(prepare_via_binding, amps=[1.0, 2.0, 3.0, 4.0])
print(
    f"fidelity (bound Vector[Float]) = {fidelity(sv, normalize([1.0, 2.0, 3.0, 4.0])):.6f}"
)
assert np.isclose(
    fidelity(sv, normalize([1.0, 2.0, 3.0, 4.0])), 1.0, atol=ATOL_STATEVECTOR
)

# %% [markdown]
# `parameters=["amps"]`でパラメータをsymbolicに残そうとすると、方向付きエラーで拒否されます — 角度計算（`atan2(|a_1|, |a_0|)`など）には具体的な数値が必要なので、ランタイムへ遅延させることは本質的にできません。エラーメッセージはランタイムケース用に`amplitude_encoding_from_angles`を指し示します。

# %%
try:
    transpiler.transpile(prepare_via_binding, parameters=["amps"])
except ValueError as exc:
    print(f"ValueError: {exc}")
    raised = True
else:
    raised = False
assert raised, "ampsをruntime parameterにすると ValueError が出るはず"

# %% [markdown]
# ### Runtime parametric な角度 — 1度コンパイルして何度も再バインド
#
# `amplitude_encoding_from_angles`は、1つのコンパイル済み回路を異なる振幅ベクトルにrun-timeで再バインドできる唯一のパスです。`compute_mottonen_amplitude_encoding_*_angles`ヘルパーで角度を古典的に事前計算し、`parameters=[...]`で1度transpileしておき、反復ごとに新しいバインディングでサンプリングします。


# %%
@qmc.qkernel
def prepare_from_angles(
    ry_a: qmc.Vector[qmc.Float], rz_a: qmc.Vector[qmc.Float]
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, "q")
    q = amplitude_encoding_from_angles(q, ry_a, rz_a)
    return qmc.measure(q)


prepare_from_angles.draw(
    fold_loops=False,
    ry_a=compute_mottonen_amplitude_encoding_ry_angles(amps_complex).tolist(),
    rz_a=compute_mottonen_amplitude_encoding_rz_angles(amps_complex).tolist(),
)

# %%
exe = transpiler.transpile(prepare_from_angles, parameters=["ry_a", "rz_a"])
n_runtime_params = len(exe.compiled_quantum[0].circuit.parameters)
print(f"runtime parameters in compiled circuit: {n_runtime_params}")
assert n_runtime_params == 2 * (2**2 - 1), (
    "n=2 複素ケースでは 2 * (2^n - 1) 個のパラメトリック回転を期待"
)

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
    max_dev = float(np.max(np.abs(observed - expected_probs)))
    print(f"amps={str(trial_amps):<48s}  max|p_obs - p_exp| = {max_dev:.4f}")
    assert max_dev < ATOL_SHOT, (
        f"runtime-parametric サンプリングが乖離 (amps={trial_amps})"
    )

# %% [markdown]
# 3回の反復はすべて同じコンパイル済み回路からサンプリングしています。変化するのはランタイムバインディングだけです。ビンごとの最大偏差はショットノイズ許容内に収まります。

# %% [markdown]
# ## 5. 大きいカーネルへの組み込み — 観測量推定
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
print(f"<Z_0> = {float(result):+.6f}   (analytic: {-1 / 3:+.6f})")
assert np.isclose(float(result), -1.0 / 3.0, atol=1e-8), (
    "<Z_0> estimator が解析値から乖離"
)
