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
# tags: [algorithm, chemistry, variational]
# ---
#
# # 水素分子のVQE：エネルギーの推定と変分最適化
#
# 変分量子固有値ソルバー（Variational Quantum Eigensolver: VQE）は、パラメータを持つ量子回路と古典的な最適化を組み合わせて、ハミルトニアンの基底状態エネルギーを近似する方法です。ここでは、水素分子$(\mathrm{H}_2)$を例に、OpenFermionとPySCFでハミルトニアンを作成し、Qamomileの状態準備カーネルと`Estimator`を使ってエネルギーを推定します。最初に一つの原子間距離で最適化し、その後、原子間距離を変えたエネルギー曲線を調べてみましょう。

# %%
# Install the latest Qamomile and chemistry extras through pip!
# # !pip install "qamomile[visualization]" openfermion pyscf openfermionpyscf

# %%
from __future__ import annotations

import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import openfermion.chem as of_chem
import openfermion.transforms as of_trans
import openfermionpyscf as of_pyscf
from scipy.optimize import minimize

import qamomile as qm
from qamomile.measurement import Estimator

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"


# %% [markdown]
# ## 問題設定
#
# ### 水素分子のハミルトニアンを作る
#
# 原子核の位置を固定し、STO-3G基底を用いて電子状態を計算しましょう。$\mathrm{H}_2$のSTO-3G計算では、2つの空間軌道にそれぞれ2種類のスピンを対応させるため、4つのスピン軌道を扱う必要があります。ジョルダン・ウィグナー変換では、各スピン軌道の占有を1量子ビットで表すので、量子ビット削減を行わない本例では4量子ビットが必要です。
#
# 処理は、分子積分の計算・フェルミオン演算子の構築・ジョルダン・ウィグナー変換の順に行います。`MolecularData`の`multiplicity=1`はスピン一重項を指定する設定、`charge=0`は電荷がゼロであることを表します。$\mathrm{H}_2$分子は2電子系です。まずは例として、原子間距離を$0.977\,\mathrm{\AA}$に固定しましょう。
#
# PySCFを用いて完全配置間相互作用（Full Configuration Interaction: FCI）も計算し、比較用のエネルギーを保存しておきます。FCIは、ここで選んだ基底・電子数などの条件の下での参照解です。有限のSTO-3G基底による誤差までなくなるわけではないため、注意が必要です。
#
# 以降、距離の単位はオングストローム（Å）、エネルギーの単位はHartreeとします。

# %%
def make_hydrogen_molecule(bond_length: float):
    """Generate one STO-3G hydrogen molecule and its qubit Hamiltonian.

    Args:
        bond_length: Distance between the hydrogen atoms in angstroms.

    Returns:
        A pair containing the Jordan–Wigner operator and molecular data.
    """

    geometry = [["H", [0, 0, 0]], ["H", [0, 0, bond_length]]]
    molecule = of_chem.MolecularData(geometry, "sto-3g", 1, 0)
    molecule = of_pyscf.run_pyscf(molecule, run_scf=True, run_fci=True)
    fermionic = of_trans.get_fermion_operator(molecule.get_molecular_hamiltonian())
    return of_trans.jordan_wigner(fermionic), molecule


jordan_wigner_hamiltonian, molecule = make_hydrogen_molecule(0.977)
N_QUBITS = molecule.n_qubits
N_ELECTRONS = molecule.n_electrons

assert N_QUBITS == 4
assert N_ELECTRONS == 2
assert molecule.fci_energy is not None
print(f"FCI reference energy: {molecule.fci_energy:+.12f}")

# %% [markdown]
# ## アルゴリズム
#
# ### VQEで最小化する量
#
# 量子回路が準備する試行状態を$|\psi(\boldsymbol{\theta})\rangle$、量子ビットのハミルトニアンを$H$とすると、最小化したい量は次のとおりです。
#
# $$
# E(\boldsymbol{\theta})
# =
# \langle\psi(\boldsymbol{\theta})|H|\psi(\boldsymbol{\theta})\rangle
# $$
#
# ここで$\boldsymbol{\theta}$は、回転ゲートに与える角度の集まりです。量子コンピュータ側で状態を準備して測定し、古典側でエネルギーの推定値を計算して角度を更新します。この操作を繰り返すことで、より低いエネルギーを与えるパラメータを探索します。
#
# 本例では、Qamomileに組み込まれたローカルシミュレーターである`qm.Reference()`を実行先に用います。
#
# 規格化された状態の真の期待値は、その状態が属する探索空間におけるハミルトニアンの最低固有値以上の値になります。ただし、有限回の測定から得た推定値には統計的なばらつきがあります。また、分子のFCI基準値と比較する際は、電子数などの条件も確認する必要があります。

# %% [markdown]
# ## 実装
#
# ### Qamomileのハミルトニアンへ変換する
#
# 得られたOpenFermionの`QubitOperator`を、同じ演算子を表すQamomileの`Hamiltonian`に変換しましょう。量子ビットのハミルトニアンは、一般に次のように表せます。
#
# $$
# H=c_0 I+\sum_{\ell}c_\ell P_\ell
# $$
#
# $P_\ell$はパウリ演算子の積、$c_\ell$はその係数、$c_0 I$は恒等演算子に比例する定数項です。
#
# 次の関数`openfermion_to_qamomile`を用い、各項の量子ビット番号と係数をそのまま対応させましょう。エネルギーをFCIと比較するため、定数項も保持します。
#
# `len(hamiltonian.terms)`は非恒等パウリ項の数を確認するために表示します。定数項は別に保持されるため、OpenFermion側で恒等項も含めて数えた項数とは区別するようにします。


# %%
def openfermion_to_qamomile(operator, num_qubits: int) -> qm.Hamiltonian:
    """Convert an OpenFermion QubitOperator into a Qamomile Hamiltonian.

    Args:
        operator: Jordan–Wigner-transformed OpenFermion operator.
        num_qubits: Width retained even when the highest mode is identity.

    Returns:
        The equivalent immutable Qamomile Hamiltonian.
    """

    paulis = {"X": qm.X, "Y": qm.Y, "Z": qm.Z}
    result = qm.Hamiltonian.zero(num_qubits=num_qubits)
    for factors, coefficient in operator.terms.items():
        assert abs(coefficient.imag) < 1e-12
        if not factors:
            result = result + coefficient.real
            continue
        word = qm.Hamiltonian.identity(num_qubits=num_qubits)
        for index, label in factors:
            word = word * paulis[label](index)
        result = result + coefficient.real * word
    return result


hamiltonian = openfermion_to_qamomile(
    jordan_wigner_hamiltonian,
    N_QUBITS,
)

assert hamiltonian.num_qubits == N_QUBITS
assert hamiltonian.is_hermitian
assert qm.Hamiltonian.from_portable(hamiltonian.to_portable()) == hamiltonian
print(f"Hamiltonian terms: {len(hamiltonian.terms)}")

# %% [markdown]
# ### VQE ansatzを作る
#
# アンザッツとは、パラメータを変化させながら試行状態を準備する回路のことです。ここでは、RYゲート・RZゲート・隣接量子ビット間のCXゲートを組合せ、ハードウェア効率の良いアンザッツ回路を準備しましょう。
#
# 「全量子ビットへのRY→全量子ビットへのRZ→CX鎖」を4回繰り返し、最後にRYとRZの層を加えます。各回転ゲートが独立した角度を持つため、パラメータ数は次のように求められます。
#
# $$
# (4+1)\times2\times4=40
# $$
#
# `thetas`を`Param`として宣言し、回路構造を保ったまま実行ごとに角度を変えます。次の`hydrogen_vqe_state`は測定せずに量子レジスタを返す、状態準備用のカーネルです。後でこの定義を`Estimator`に渡し、必要な測定と組合せます。

# %%
REPETITIONS = 1 if docs_test_mode else 4
NUM_PARAMETERS = (REPETITIONS + 1) * 2 * N_QUBITS


@qm.qkernel
def hydrogen_vqe_state(
    thetas: qm.Param[qm.Vector[qm.Real, NUM_PARAMETERS]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[4]]:  # type: ignore[valid-type]
    """Prepare the four-qubit hardware-efficient hydrogen trial state.

    Args:
        thetas (qm.Param[qm.Vector[qm.Real, NUM_PARAMETERS]]): Runtime SU(2) angles.

    Returns:
        qm.Q[qm.Bits[4]]: Prepared variational state.
    """

    state = qm.qubits(4)
    for repetition in range(REPETITIONS):
        base = repetition * 8
        for target in range(4):
            state = qm.ry(state, thetas[base + target], target=target)
        for target in range(4):
            state = qm.rz(state, thetas[base + 4 + target], target=target)
        for control in range(3):
            state = qm.cx(state, control=control, target=control + 1)
    for target in range(4):
        state = qm.ry(state, thetas[8 * REPETITIONS + target], target=target)
    for target in range(4):
        state = qm.rz(state, thetas[8 * REPETITIONS + 4 + target], target=target)
    return state


# %% [markdown]
# :::{note} この回路は電子数を保存しない
#
# この量子回路では$|0000\rangle$から始まり、RYとCXによって占有数の異なる状態も準備されます。`N_ELECTRONS = 2`を読み取ったり、分子計算で`charge=0`を指定したりするだけでは、変分回路が2電子状態に制限されるわけではないことに注意しましょう。
#
# この例のFCIは「対象とする2電子分子の参照値」として読みます。任意のパラメータに対する真の期待値が、そのFCI値以上になるという主張には、探索する電子数・スピンの条件の一致、または全探索空間の最低エネルギーとの一致の確認が必要です。
#
# 分子の基底状態をより厳密に探索する実装に進む場合は、電子数などの対称性を保つアンザッツや制約を考慮しなければなりません。ハートリー・フォック状態から始めるだけでは、その後の回路による電子数の変化は防げません。
# :::
#
# ### `Estimator`でエネルギーを推定する
#
# `Estimator`に、状態準備用カーネルとハミルトニアンを渡しましょう。各パウリ項の期待値を測定し、その係数を掛けて足し合わせることで、$E(\boldsymbol{\theta})$を推定することができます。
#
# `grouping="qwc"`は、各量子ビットについて同じ測定基底を使える項をまとめる指定です。例えば$Z_0$と$Z_0Z_1$は一緒に測定できますが、$Z_0$と$X_0$は異なる基底で測定します。可換な項であっても、各量子ビットの測定基底が異なる場合、この方法では別々に測定します。
#
# 状態準備の戻り値の量子ビット番号と、`qm.X(i)`、`qm.Z(i)`などの添字を対応させておきます。測定基底を変える場合も、新しく状態を準備して測定します。
# %%
endpoint = qm.Reference()
estimator = Estimator(hydrogen_vqe_state, hamiltonian, grouping="qwc")
assert NUM_PARAMETERS == (16 if docs_test_mode else 40)


# %% [markdown]
# #### 総ショット数と標準誤差
#
# 次に定義する関数は、エネルギー推定結果を返す関数です。`shots`は、1回の`estimate`呼び出しで使用する測定グループ全体のショット数です。既定の`allocation="uniform"`では、総ショット数を測定グループ間で均等に分け、割り切れない場合は余りを先頭のグループから1回ずつ配ります。恒等項の寄与は定数なので、測定せずに加算します。
#
# 同じグループ内の各パウリ項は、そのグループに割り当てられた測定結果を共有します。例えば$Z_0$と$Z_0Z_1$を同じグループで1,024回測定した場合、両方の項の期待値を同じ1,024回の結果から推定します。グループ内の項数で測定回数をさらに分けることはありません。
#
# 本例では5つの測定グループを用います。目的関数の評価では総ショット数5,120を各グループ1,024回に、最適化後の再測定では総ショット数40,960を各グループ8,192回に分けます。
#
# 同じ環境で同じ順序で実行したときの再現性を確保するため、各呼び出しの乱数シードを順に変え、最適化中と最終測定で異なる疑似乱数系列を使うことにします。
#
# `estimate.value`がエネルギー推定値、`estimate.standard_error`がその標準誤差を表します。標準誤差は、独立なショットを仮定した測定のばらつきに対応します。最適化の未収束・回路の表現力の不足・基底の近似・実機の系統誤差は含みません。
#
# グループごとの測定回数が足りなければ、標準誤差を推定できない場合もあることにご留意ください。`SHOTS`を大幅に減らす場合は、測定グループ数に対して十分な標本数があるかも確認すると良いでしょう。

# %%
SHOTS = 10 if docs_test_mode else 5120
FINAL_SHOTS = 10 if docs_test_mode else 40960
measurement_seeds = itertools.count(42)


def estimate_energy(values, estimator, *, shots=SHOTS):
    """Evaluate the prepared estimator with fresh random streams."""
    estimate = estimator.estimate(
        on=endpoint,
        parameters={"thetas": tuple(float(value) for value in values)},
        shots=shots,
        seed=next(measurement_seeds),
    )
    assert estimate.shots == shots
    assert np.isfinite(estimate.value)
    assert estimate.standard_error is not None
    assert np.isfinite(estimate.standard_error)
    return estimate


# %% [markdown]
# ## 実行結果
#
# ### 古典最適化を実行し、最終パラメータを再評価する
#
# SciPyの`minimize`とCOBYLAを用い、測定から得たエネルギー推定値を小さくする角度を探索しましょう。COBYLAは勾配を要求しない最適化手法であり、有限差分の勾配計算を必要としません。ただし、勾配不要であるからといって、ショットノイズのある目的関数で大域最適解へ収束する保証があるわけではありません。
#
# この手法の`maxiter`は最大反復回数ではなく、**目的関数の最大評価回数**を指定する変数です。各最適化の評価上限を500回とします。初期値は、乱数シードを固定した一様乱数で選びます。
#
# 最適化中に観測された最も低いエネルギーには、偶然低く出た測定値を選ぶ偏りが含まれる可能性があります。そのため、最終結果には`optimization.fun`をそのまま用いず、返された`optimization.x`を新しいショットで再測定した値を用いることにします。
#
# `success`と`message`は最適化の終了条件の確認に使います。`success=True`であっても、FCIと一致したことや、大域最適解が見つかったことを意味するわけではないことに注意が必要です。

# %%
cost_history = []


def cost_function(values) -> float:
    """Record the sampled energy used by the classical optimizer."""
    estimate = estimate_energy(values, estimator)
    cost_history.append(estimate.value)
    return estimate.value


rng = np.random.default_rng(42)
initial_parameters = rng.uniform(0.0, np.pi, NUM_PARAMETERS)
max_evaluations = NUM_PARAMETERS + 2 if docs_test_mode else 500
optimizer_options = {"maxiter": max_evaluations, "rhobeg": 0.5}
if docs_test_mode:
    optimizer_options["f_target"] = float("inf")

optimization = minimize(
    cost_function,
    initial_parameters,
    method="COBYLA",
    options=optimizer_options,
)
if docs_test_mode:
    assert optimization.nfev == 1
final_estimate = estimate_energy(optimization.x, estimator, shots=FINAL_SHOTS)
fci_reference = float(molecule.fci_energy)
print(
    f"VQE energy: {final_estimate.value:+.6f} +/- {final_estimate.standard_error:.6f} (SE)"
)
print(f"FCI energy: {fci_reference:+.6f}")
print(f"Final measurement shots: {final_estimate.shots}")
assert len(optimization.x) == NUM_PARAMETERS
assert len(cost_history) == optimization.nfev
print("Optimizer success:", optimization.success)
print("Optimizer message:", optimization.message)
print("Objective evaluations:", optimization.nfev)
print(f"Difference from FCI: {final_estimate.value - fci_reference:+.6f} Hartree")

# %% [markdown]
# ### 最適化履歴の確認
#
# 横軸に目的関数を評価した順番として、エネルギー探索の推移を見てみましょう。最後に表示する点は、選ばれたパラメータを独立に再測定した結果であり、追加の最適化ステップではありません。
#
# 誤差棒は標準誤差1個分（1 SE）です。分子の真の基底状態エネルギーを必ず含む区間ではないことに注意しましょう。

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
evaluation_numbers = np.arange(1, len(cost_history) + 1)
ax.plot(evaluation_numbers, cost_history, label="Sampled VQE objective")
ax.errorbar(
    [len(cost_history) + 1],
    [final_estimate.value],
    yerr=[final_estimate.standard_error],
    fmt="o",
    capsize=4,
    label="Fresh final estimate (1 SE)",
)
ax.axhline(fci_reference, linestyle="dashed", label="FCI reference (2 electrons)")
ax.set_xlabel("Objective evaluation / final remeasurement")
ax.set_ylabel("Energy (Hartree)")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ### FCIとの差が残った場合
#
# 最終推定値とFCIの差が標準誤差より十分大きい場合、測定のばらつきだけでは説明できません。このような場合、評価回数・初期値・回路の構造・電子数条件・スピンの条件などを確認する必要があるでしょう。1回の実行結果だけでは、どの要因が支配的かを特定するのは、至難の業です。
#
# 最終測定のショット数を増やすと、同じパラメータでのエネルギーをより精密に推定することができます。しかし、すでに選ばれた状態そのものを、より低いエネルギーの状態へ改善するわけではありません。
#
# 変分原理が適用される真の期待値と、有限ショットの推定値も区別する必要があります。同じ探索空間の基底エネルギーに対しても、推定値は統計的な揺らぎによって下回る場合があります。本例ではさらに、FCIと変分回路の電子数条件が自動では一致しない点に注意しましょう。
#
# ### 原子間距離を変えてエネルギー曲線を調べる
#
# 各原子間距離$R$についてハミルトニアンとFCIの参照値を再計算し、そのハミルトニアン用の`Estimator`で最適化しましょう。$0.2$ Åから$1.5$ Åまでの15点を計算します。
#
# 最初の計算点には前節で得たパラメータを使い、それ以降は一つ前の距離で得たパラメータを初期値として渡すことにします。この方法はwarm startと呼ばれますが、前の点での最適化不足を引き継ぐ場合もあり、注意が必要です。そして距離を調べる順序によって、同じ結果にならない可能性もあります。
#
# ここでも`grouping="qwc"`を明示し、前節と同じグループ化を用いることにしましょう。各点の最終エネルギーは、新しいショットで再評価することにします。
#
# ここで行っているのは、固定した距離ごとのエネルギーを比較する計算です。原子間距離を連続変数として最適化する構造最適化そのものではありません。有限基底と限られた評価予算の下で得られる曲線として計算します。

# %%
n_points = 1 if docs_test_mode else 15
bond_lengths = np.linspace(0.2, 1.5, n_points)
vqe_energies = []
vqe_errors = []
fci_energies = []
optimization_success = []
objective_evaluations = []
starting_parameters = optimization.x.copy()

for bond_length in bond_lengths:
    transformed, bond_molecule = make_hydrogen_molecule(float(bond_length))
    bond_hamiltonian = openfermion_to_qamomile(transformed, bond_molecule.n_qubits)
    bond_estimator = Estimator(hydrogen_vqe_state, bond_hamiltonian, grouping="qwc")
    bond_fci = bond_molecule.fci_energy
    assert bond_fci is not None
    bond_result = minimize(
        lambda values: estimate_energy(values, bond_estimator).value,
        starting_parameters,
        method="COBYLA",
        options=optimizer_options,
    )
    starting_parameters = bond_result.x.copy()
    optimization_success.append(bool(bond_result.success))
    objective_evaluations.append(int(bond_result.nfev))
    measured = estimate_energy(
        starting_parameters,
        bond_estimator,
        shots=FINAL_SHOTS,
    )
    vqe_energies.append(measured.value)
    vqe_errors.append(measured.standard_error)
    fci_energies.append(float(bond_fci))
    print(
        f"distance={bond_length:.3f}, VQE={measured.value:+.6f} "
        f"+/- {measured.standard_error:.6f} (SE), FCI={bond_fci:+.6f}, "
        f"success={bond_result.success}, nfev={bond_result.nfev}"
    )

assert len(vqe_energies) == len(vqe_errors) == len(fci_energies) == n_points
print("Optimizer successes:", sum(optimization_success), "/", n_points)

# %% [markdown]
# 得られた結果を、可視化してみましょう。

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.errorbar(
    bond_lengths,
    vqe_energies,
    yerr=vqe_errors,
    fmt="-o",
    capsize=3,
    label="Sampled VQE (1 SE)",
)
ax.plot(bond_lengths, fci_energies, "--", label="FCI reference (2 electrons)")
ax.set_xlabel("H-H distance (angstrom)")
ax.set_ylabel("Energy (Hartree)")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## まとめ
#
# - OpenFermionとPySCFでハミルトニアンを作成し、ジョルダン・ウィグナー変換後の4量子ビット演算子をQamomileに渡しました。
# - パラメータ化された状態準備用カーネルに`Estimator`を組合せ、有限ショットのエネルギー推定値をCOBYLAで最適化しました。
# - 結果の評価には、FCIと比較している基底・電子数条件・最適化の終了要件・最終再測定時の標準誤差を分けて考える必要があります。
# - 最終結果は新たな測定によるエネルギー推定値と誤差棒で示しました。ショットをまたぐ集計はqkernelの外で行うようにしています。
