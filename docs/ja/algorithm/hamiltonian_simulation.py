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
# tags: [algorithm, primitive, simulation]
# ---
#
# # Suzuki–Trotter分解によるハミルトニアンシミュレーション (Rabi振動)
#
# 量子系の時間発展$e^{-iHt}$をシミュレーションすることは、量子コンピュータの代表的な応用の1つです。ハミルトニアンが非可換な部分に分割されるとき、つまり$H = A + B$かつ$[A, B] \neq 0$のとき、素朴な分解$e^{-i(A+B)t} = e^{-iAt}\,e^{-iBt}$は成立しません。標準的な対処法が**Trotter–Suzuki積公式**で、各項の短時間発展を交互に並べます。誤差は、ステップ幅を小さくすれば減り(Lie–Trotter、1次)、ステップを対称化すれば減り(Strang、2次)、対称ステップをSuzukiの構成で再帰的に入れ子にすればさらに減ります(任意の偶数次)。
#
# 本記事では、1量子ビットのRabiハミルトニアンでTrotter近似を確認します。まず$S_1$と$S_2$を扱い、高次の公式には標準ライブラリの`suzuki_trotter`を使います。Rustが静的な指定から上限を検査したスケジュールを導出し、生成されるqkernelは構造化されたループを保持します。最後に、フィデリティ誤差の収束次数を2×2の厳密なプロパゲータと比較します。

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

import numpy as np

import qamomile as qm
from qamomile.stdlib import suzuki_trotter

# %% [markdown]
# ## Rabiハミルトニアン
#
# 共鳴駆動される2準位系は次のハミルトニアンで記述されます:
#
# $$ H = \underbrace{\tfrac{\omega}{2} Z}_{H_z} + \underbrace{\tfrac{\Omega}{2} X}_{H_x}. $$
#
# $[Z, X] \neq 0$なので、$H$を$H_z$と$H_x$に分割するとTrotter誤差が生じ、1量子ビットでも明確に観測できます。初期状態$|0\rangle$から始めると、励起確率は$P_{|1\rangle}(t) = (\Omega/E)^2 \sin^2(Et/2)$に従って振動します。ただし$E = \sqrt{\omega^2 + \Omega^2}$です。
#
# Qamomileでは、$H_z$と$H_x$を2つの`Observable`としてリストにまとめます。生成されるqkernelは固定長の`Const[Vector[Observable, ...]]`を受け取り、特殊化時に値を束縛します。QCoreではPauli時間発展とループを抽象的に保持し、ターゲットへのloweringでSDKの操作に変換します。


# %% [markdown]
# ### $[H_z, H_x] \neq 0$ の確認
#
# Trotter近似が必要になるのは$H_z$と$H_x$が可換でないからです。`qamomile.observable.commutator(a, b)`はハミルトニアン同士の交換子$[a, b] = a b - b a$を直接計算します。内部では各Pauli列ペアを1度だけ走査し、qubitパリティの規則(2つのPauli列は、両方とも非恒等かつ異なるPauliが乗っているqubitの数が奇数のときにだけ反交換)に従って、可換なペアを積を作る前に落とします。`Hz * Hx - Hx * Hz`のように一旦展開してから打ち消す素朴な計算より軽く、結果として完全に簡約された`Hamiltonian`が返ってくるので、そのまま検査したり解析値と比較したりできます。
#
# Rabiハミルトニアンの場合、教科書的な値は
#
# $$ [H_z, H_x] \;=\; \tfrac{\omega \Omega}{4}\,[Z, X] \;=\; i\,\tfrac{\omega \Omega}{2}\, Y $$
#
# であり、`commutator`はこれを厳密に再現します:

# %%
omega = 1.2
drive = 0.8
evolution_time = 1.5

drift_hamiltonian = 0.5 * omega * qm.Z(0)
drive_hamiltonian = 0.5 * drive * qm.X(0)
hamiltonians = (drift_hamiltonian, drive_hamiltonian)

commutator = qm.commutator(*hamiltonians)
expected_commutator = 0.5j * omega * drive * qm.Y(0)

assert commutator == expected_commutator
print("[H_z, H_x] =", commutator)

# %% [markdown]
# `Hamiltonian`はSDKに依存せず、論理量子ビットの規約とPauli係数を保持します。そのため、targetを選択する前に同じ値をQamomile programへ渡せます。

# %% [markdown]
# ## 厳密な参照状態
#
# 2x2行列の指数関数で厳密な状態$|\psi(T)\rangle = e^{-iHT}|0\rangle$が得られます。各Trotter近似は**フィデリティ誤差**$1 - |\langle\psi_\text{exact}|\psi_\text{trotter}\rangle|$によってこの状態と比較します。

# %%
pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
matrix = 0.5 * omega * pauli_z + 0.5 * drive * pauli_x
initial_state = np.array([1.0, 0.0], dtype=np.complex128)


def matrix_evolution(hamiltonian: np.ndarray, time: float) -> np.ndarray:
    """Return the exact dense exponential of one Hermitian matrix."""

    values, vectors = np.linalg.eigh(hamiltonian)
    return vectors @ np.diag(np.exp(-1j * time * values)) @ vectors.conj().T


exact_state = matrix_evolution(matrix, evolution_time) @ initial_state

# %% [markdown]
# ## $S_1$: 1次Suzuki–Trotter分解 (Lie–Trotter)
#
# もっとも単純な分解は
#
# $$ S_1(\Delta t) = e^{-i H_x \Delta t}\, e^{-i H_z \Delta t} $$
#
# で、これを$N$回適用して全発展時間$T = N \Delta t$をシミュレーションします。1ステップあたりの局所誤差は$O(\Delta t^2)$、$N = T/\Delta t$ステップにわたる大域的な状態ノルム誤差は$O(\Delta t)$です。
#
# 以下のfactoryで、時間を4分割する1次と2次のカーネルを生成します。`steps`は静的なfactory引数ですが、全発展時間は`Param`です。時間を変えて同じ公式を再利用できます。分割数を変える場合は別の公式を生成します。


# %% [markdown]
# ## $S_2$: 2次Suzuki–Trotter分解 (Strang分解)
#
# 中央の項を中心にステップを対称化すると先頭の誤差項が消えます:
#
# $$ S_2(\Delta t) = e^{-i H_z \Delta t/2}\, e^{-i H_x \Delta t}\, e^{-i H_z \Delta t/2}. $$
#
# 局所誤差は$O(\Delta t^3)$、大域的な状態ノルム誤差は$O(\Delta t^2)$になります。ステップカーネルは`pauli_evolve`を3回呼ぶだけです。


# %%
first_order = suzuki_trotter(term_count=2, order=1, steps=4)
second_order = suzuki_trotter(term_count=2, order=2, steps=4)


@qm.qkernel
def compare_orders(
    hamiltonians: qm.Const[qm.Vector[qm.Observable, 2]],
    time: qm.Param[qm.Real],
) -> tuple[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]]:
    first = first_order(qm.qubits(1), hamiltonians, time)
    second = second_order(qm.qubits(1), hamiltonians, time)
    return first, second


# %%
order_program = qm.specialize(compare_orders, hamiltonians=hamiltonians)
order_study = qm.resources.logical(order_program)
order_metric = qm.resources.logical_operation_count("quantum.pauli_evolution")

assert order_study.metric(order_metric).numeric_enclosure() == (20, 20)

# %% [markdown]
# 4個のfirst-order sliceには8回、4個のsecond-order sliceには12回のterm evolutionがあります。各allocationに独立したownerが1つあるため、2つのowned stateを返せます。callerはそれぞれを別にcomposeまたはmeasureできます。

# %% [markdown]
# ## 高次のSuzuki–Trotter分解:フラクタル再帰
#
# 鈴木増雄氏は、任意の偶数次Trotter近似を$S_2$から**再帰的に**構築できることを示しました。各段で5つのリスケーリングされたコピーを入れ子にします:
#
# $$ S_{2k}(\Delta t) = S_{2k-2}(p_k \Delta t)^2 \, S_{2k-2}\bigl((1 - 4 p_k)\Delta t\bigr) \, S_{2k-2}(p_k \Delta t)^2. $$
#
# ここで段ごとの係数は
#
# $$ p_k = \frac{1}{4 - 4^{1/(2k-1)}} $$
#
# です。$p_k$は下位の公式の$(2k-1)$次誤差がキャンセルされるように選ばれているので、1ステップあたりの局所誤差は$O(\Delta t^{2k+1})$になります。**係数は段ごとに必ず計算し直す必要があります。** 具体的には:
#
# - $k=2$ (4次): $p_2 = 1/(4 - 4^{1/3}) \approx 0.4145$
# - $k=3$ (6次): $p_3 = 1/(4 - 4^{1/5}) \approx 0.3731$
# - $k=4$ (8次): $p_4 = 1/(4 - 4^{1/7}) \approx 0.3596$
#
# すべての段で$p_2$を使い回すと$(2k-1)$次の誤差項が残ったままになり、結果として得られる公式は$S_4$とほぼ同等の精度しか持ちません。これはSuzuki-Trotterを手書きで実装するときにハマりがちな罠です。

# %% [markdown]
# ## 上限付き4次公式の選択
#
# `suzuki_trotter`には、通常のPython factoryを呼ぶ段階で静的なtopology optionを渡します。ここではHamiltonian 2項、4次、4つの等しい時間sliceを選びます。Rustが11行の対称scheduleを導出して上限を検査し、通常の型付きQamomile関数を返します。
#
# scheduleは合成されたprogramであり、ハミルトニアンシミュレーション専用opcodeではありません。QCoreは1つの抽象的なPauli時間発展operationを固定loop内に保持し、targetはlowering時にだけその実現方法を決定します。

# %%
evolve = suzuki_trotter(term_count=2, order=4, steps=4)


@qm.qkernel
def rabi_samples(
    hamiltonians: qm.Const[qm.Vector[qm.Observable, 2]],
    time: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[1]]:
    """Evolve and measure one driven two-level system.

    Args:
        hamiltonians (qm.Const[qm.Vector[qm.Observable, 2]]): Ordered drift and
            drive terms.
        time (qm.Param[qm.Real]): Total evolution time.

    Returns:
        qm.Shot[qm.Bits[1]]: One sampled excitation bit.
    """

    state = evolve(qm.qubits(1), hamiltonians, time)
    return qm.measure(state)


# %% [markdown]
# Hamiltonianの分割は静的なprogram dataを選ぶため`Const`です。時間発展の長さは`Param`なので、構造を再構築せずに1つのコンパイル済みartifactを異なる時刻でlaunchできます。

# %% [markdown]
# ## hardwareを選ぶ前のresource estimation
#
# specializationにより、検証済みでtarget-independentな`Program`が得られます。logical resource analysisは同じimageを読み、loopの反復回数とRustが所有する積公式scheduleを合成します。

# %%
program = qm.specialize(rabi_samples, hamiltonians=hamiltonians)
study = qm.resources.logical(program)
pauli_evolutions = qm.resources.logical_operation_count("quantum.pauli_evolution")

assert "quantum.pauli_evolution" in program.inspect()
assert "quantum.hamiltonian_simulation" not in program.inspect()
assert study.metric(pauli_evolutions).numeric_enclosure() == (44, 44)
print("logical Pauli evolutions:", study.metric(pauli_evolutions))

# %% [markdown]
# 推定値は厳密です。4つのsliceで11個のschedule entryがそれぞれ実行されます。resource analysisではstatevectorやSDK回路を展開せず、利用できない分解をwork 0として扱うこともありません。

# %% [markdown]
# ### 積公式resource scalingの再現
#
# [](https://doi.org/10.48550/arXiv.2310.03011)のSection 11.1は、$r$個の
# short-time sliceを繰り返して積公式を構成し、sliceあたりのworkに$r$を掛けて
# gate costを得ます。Pauli-access modelでは追加ancilla qubitを使いません。この
# 2項second-order formulaでは、各sliceがexactに3個のabstract Pauli evolutionを
# 含みます。repeat countを`Const`引数として公開するため、本体の解析は1度だけで、
# reportする各rowはalgebraへの代入になります。


# %%
second_order_slice = suzuki_trotter(term_count=2, order=2, steps=1)


@qm.qkernel
def surveyed_product_formula(
    terms: qm.Const[qm.Vector[qm.Observable, 2]],
    time: qm.Param[qm.Real],
    steps: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[1]]:
    """Second-order Suzuki sliceをsymbolicな回数だけ繰り返す。"""

    state = qm.qubits(1)
    for _ in range(steps):
        state = second_order_slice(state, terms, time / steps)
    return state


product_formula_study = qm.resources.logical(surveyed_product_formula)


def product_formula_resource_row(steps: int) -> tuple[int, int]:
    """1つのstep countについてPauli-evolution数と割当qubit数を返す。"""

    resource_study = product_formula_study.evaluate(steps=steps)
    evolutions = resource_study.metric(pauli_evolutions).numeric_enclosure()
    allocated = resource_study.metric(
        qm.resources.LOGICAL_ALLOCATED_QUBITS
    ).numeric_enclosure()
    assert evolutions[0] == evolutions[1]
    assert allocated[0] == allocated[1]
    return evolutions[0], allocated[0]


product_formula_scaling = {
    steps: product_formula_resource_row(steps) for steps in (1, 2, 4, 8, 16)
}
for steps, (evolutions, allocated) in product_formula_scaling.items():
    assert evolutions == 3 * steps
    assert allocated == 1

print("second-order product formula (r: evolutions, qubits):", product_formula_scaling)

# %% [markdown]
# 有限サイズのexact countは$3r$で、registerは1 qubitのままです。これはサーベイの
# 「sliceあたりのcostを$r$倍する」というalgebraとno-ancillaの記述を再現します。
# 1個のabstract `pauli_evolution`がfault-tolerant elementary gate 1個に等しいとは
# 主張しません。その後のrealizationはtarget依存です。

# %% [markdown]
# ## 収束性のスイープ
#
# Trotterステップ数$N$を掃引し、ステップ幅$\Delta t = T / N$に対してフィデリティ誤差を両対数軸でプロットします。期待される傾きは以下のとおりです:
#
# | 公式 | 局所誤差 | 大域ノルム誤差 | フィデリティ誤差 ($1 -$ 内積) |
# |-----|---------|---------------|------------------------------|
# | $S_1$ | $O(\Delta t^2)$ | $O(\Delta t)$   | $O(\Delta t^2)$  |
# | $S_2$ | $O(\Delta t^3)$ | $O(\Delta t^2)$ | $O(\Delta t^4)$  |
# | $S_4$ | $O(\Delta t^5)$ | $O(\Delta t^4)$ | $O(\Delta t^8)$  |
# | $S_6$ | $O(\Delta t^7)$ | $O(\Delta t^6)$ | $O(\Delta t^{12})$ |
#
# フィデリティ誤差は状態ノルム誤差の**2乗**になります(先頭項)。2つのベクトルが十分近ければ$1 - |\langle a | b \rangle| \approx \tfrac{1}{2}\lVert a - b \rVert^2$なので、以下のプロットに現れる傾きは$1, 2, 4$ではなく$2, 4, 8$です。


# %% [markdown]
# プロット上の各直線の傾きは$\approx 2, 4, 8$で、上の表のフィデリティ誤差の次数と一致しています。$S_4$は$N = 16$の時点でfloat64の精度下限に到達し、$S_6$はスイープの全域で精度下限に張り付いているため直線が平坦に見えます(期待される$\Delta t^{12}$の傾きは、1量子ビット問題を倍精度で解く限り検出できません)。

# %%
drift_matrix = 0.5 * omega * pauli_z
drive_matrix = 0.5 * drive * pauli_x


def suzuki_step(order: int, duration: float) -> np.ndarray:
    """Return one dense Suzuki step for the two-term tutorial model."""

    if order == 1:
        return matrix_evolution(drive_matrix, duration) @ matrix_evolution(
            drift_matrix, duration
        )
    if order == 2:
        half_drift = matrix_evolution(drift_matrix, duration / 2.0)
        return half_drift @ matrix_evolution(drive_matrix, duration) @ half_drift
    coefficient = 1.0 / (4.0 - 4.0 ** (1.0 / (order - 1)))
    outer = suzuki_step(order - 2, coefficient * duration)
    middle = suzuki_step(order - 2, (1.0 - 4.0 * coefficient) * duration)
    return outer @ outer @ middle @ outer @ outer


step_counts = np.array([2, 4, 8, 16], dtype=np.int64)
step_sizes = evolution_time / step_counts
errors: dict[int, list[float]] = {1: [], 2: [], 4: []}

for steps in step_counts:
    for order in errors:
        step = suzuki_step(order, evolution_time / int(steps))
        approximation = np.linalg.matrix_power(step, int(steps)) @ initial_state
        errors[order].append(1.0 - abs(np.vdot(exact_state, approximation)))

slopes = {
    order: np.polyfit(
        np.log(step_sizes[:3] if order == 4 else step_sizes),
        np.log(values[:3] if order == 4 else values),
        1,
    )[0]
    for order, values in errors.items()
}

assert 1.7 < slopes[1] < 2.3
assert 3.7 < slopes[2] < 4.3
assert 7.0 < slopes[4] < 9.0
print("fidelity-error slopes:", slopes)

# %% [markdown]
# 測定したslopeは期待される$\Delta t^2$、$\Delta t^4$、$\Delta t^8$のfidelity-error behaviorを再現します。applicationで`steps`を変えるとprogram構造も変わるため、目的のbounded factoryを明示的にbuildします。runtime `time`値は生成されたartifactを引き続き再利用できます。

# %% [markdown]
# ## 同じprogramのcompileと実行
#
# `Reference`はtargetとローカルendpointの両方を提供します。コンパイルすると、program digest、runtime parameter ABI、result ABI、resource evidenceがimmutableなartifactに封印されます。実行時はPython関数を再度traceせず、このartifactを使います。

# %%
reference = qm.Reference()
artifact = qm.compile(program, to=reference)
record = qm.run(
    artifact,
    on=reference,
    parameters={"time": evolution_time},
    shots=128,
    seed=2,
)

assert artifact.source_digest.hex() == program.semantic_digest
assert tuple(slot.name for slot in artifact.parameter_abi.slots) == ("time",)
assert record.result.shots == 128
assert sum(record.result.counts.values()) == 128

# %% [markdown]
# この1量子ビットmodelの厳密な励起確率は
#
# $$
# P_1(t)=\left(\frac{\Omega}{E}\right)^2
# \sin^2\left(\frac{Et}{2}\right),
# \qquad E=\sqrt{\omega^2+\Omega^2}
# $$
#
# です。この規模では積公式の誤差がsampling uncertaintyより十分小さいため、sampling結果は保守的な4 standard deviationと小さな離散化許容量を合わせた範囲に入るはずです。

# %%
energy = math.hypot(omega, drive)
expected_probability = (drive / energy) ** 2 * math.sin(
    energy * evolution_time / 2.0
) ** 2
observed_probability = record.result.probability((1,))
sampling_bound = (
    4.0
    * math.sqrt(
        expected_probability * (1.0 - expected_probability) / record.result.shots
    )
    + 0.02
)

assert abs(observed_probability - expected_probability) <= sampling_bound
print(f"expected P(1): {expected_probability:.3f}")
print(f"sampled  P(1): {observed_probability:.3f}")

# %% [markdown]
# ## まとめ
#
# - **モデル**: 1量子ビットのRabiハミルトニアン$H = H_z + H_x$。非可換な2項の和でTrotter誤差が測定可能です。
# - **`Const[Vector[Observable, ...]]` + `pauli_evolve`**: ハミルトニアンを特殊化し、QCoreでは時間発展とループ構造を抽象的に保持します。
# - **Suzuki–Trotterフラクタル**: $S_{2k}$は$S_{2k-2}$のリスケーリングされた5つのコピーを段ごとの係数$p_k = 1/(4 - 4^{1/(2k-1)})$で入れ子にして構築します。段をまたいで同じ定数を使い回すのはよくある罠で、フラクタルの構成にはなりません。
# - **上限を検査した構築**: `suzuki_trotter`に項数、次数、時間分割数を静的に渡し、Rustが生成qkernelのスケジュールを導出します。
# - **収束**: ホスト側の行列計算でフィデリティ誤差の傾き$2, 4, 8$を確認します。この数値的な基準計算は、サンプリング実行とは区別します。
