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
# ---
# tags: [algorithm, chemistry, variational]
# ---
#
# # 水素分子のための変分量子固有値ソルバー（VQE）
#
# このチュートリアルでは、水素分子（H₂）の基底状態エネルギーを求めるための変分量子固有値ソルバー（VQE）アルゴリズムの実装について解説します。分子ハミルトニアンの生成には [OpenFermion](https://quantumai.google/openfermion) を使用します。
#
# ワークフローは以下の通りです：
# 1. 分子ハミルトニアンを量子ビット演算子へ変換
# 2. パラメータ化された量子回路（アンザッツ）の作成
# 3. VQEによる最適化の実装
# 4. 原子間距離ごとのエネルギー地形の解析
#
# 量子コンピューティングを用いた量子化学問題の解法を紹介し、特にH₂分子の最小エネルギー構造の探索に焦点を当てます。

# %% [markdown]
# ## インストール
#
# ```bash
# pip install "qamomile[qiskit,visualization]" openfermion pyscf openfermionpyscf
# ```

# %%
# Install the latest Qamomile and chemistry extras through pip!
# # !pip install "qamomile[qiskit,visualization]" openfermion pyscf openfermionpyscf

# %%
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
from qamomile.qiskit import Aer

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# ## 水素分子のHamiltonianを作る
#
# 最小基底STO-3GのH₂には2つの空間軌道があり、Jordan–Wigner符号化後は4つの
# spin orbital、すなわち4量子ビットになります。PySCFから分子積分に加えて、
# 変分結果の検証に用いるfull-configuration-interaction（FCI）energyを得ます。


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
# ## Qamomile ハミルトニアンへの変換
#
# このセクションでは、OpenFermionのハミルトニアンをQamomileフォーマットに変換します。Jordan-Wigner変換を適用してフェルミ粒子演算子を量子ビット演算子へ変換し、その後、カスタム変換関数を用いてQamomileに適したハミルトニアン表現を作成します。


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
# ## VQE ansatzを作る
#
# hardware-efficient SU(2) ansatzは従来のチュートリアルと同じです。RY layer、
# RZ layer、線形CX chainを4回繰り返し、最後にRY/RZの組を追加します。
# 4量子ビットでは$(4 + 1)\times 2\times 4=40$個のruntime parameterになります。
# この状態準備を、測定基底を明示するプログラムから呼び出します。
# 各基底で新しい状態を準備して測定し、その結果からエネルギーを推定します。

# %%
REPETITIONS = 4
NUM_PARAMETERS = (REPETITIONS + 1) * 2 * N_QUBITS


@qm.qkernel
def hydrogen_vqe_state(
    thetas: qm.Param[qm.Vector[qm.Real, 40]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[4]]:  # type: ignore[valid-type]
    """Prepare the four-qubit hardware-efficient hydrogen trial state.

    Args:
        thetas (qm.Param[qm.Vector[qm.Real, 40]]): Runtime SU(2) angles.

    Returns:
        qm.Q[qm.Bits[4]]: Prepared variational state.
    """

    state = qm.qubits(4)
    for repetition in range(4):
        base = repetition * 8
        for target in range(4):
            state = qm.ry(state, thetas[base + target], target=target)
        for target in range(4):
            state = qm.rz(state, thetas[base + 4 + target], target=target)
        for control in range(3):
            state = qm.cx(state, control=control, target=control + 1)
    for target in range(4):
        state = qm.ry(state, thetas[32 + target], target=target)
    for target in range(4):
        state = qm.rz(state, thetas[36 + target], target=target)
    return state


# %% [markdown]
# ## エネルギー推定器の準備
#
# `Estimator`は状態準備の定義とHamiltonianを受け取り、コンストラクタで検証します。
# `estimate`は実行先を確認し、契約を満たすバックエンドの推定機能か、測定回路を生成する共通経路を選びます。
# QWCの項は測定基底を共有します。返された量子ビットの位置がHamiltonianの添字に対応します。
# 生成する実行入口は古典の測定結果を返し、量子状態が実行の外へ出ることはありません。
# 共通経路では、実行時の角度を変えてもコンパイル済みの回路を再利用します。
#
# %%
endpoint = Aer()
estimator = Estimator(hydrogen_vqe_state, hamiltonian, grouping="qwc")
assert NUM_PARAMETERS == 40


# %% [markdown]
# ## 有限ショットによるエネルギー推定
#
# `estimate`が測定をスケジュールし、統計量を集計します。`shots`は**総予算**で、測定グループへ均等に配分されます。同じグループ内の共分散を保持し、異なるグループには別々の実行と乱数系列を使います。恒等項にはデバイスのショットを使いません。
#
# 標準誤差は独立同分布のショットを仮定したサンプリングの不確かさを表し、ハードウェアの偏りや最適化誤差を含みません。1ショットでは標本分散を求められず、標準誤差は`None`になります。

# %%
SHOTS = 640 if docs_test_mode else 5120
FINAL_SHOTS = 2560 if docs_test_mode else 40960
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
# ## 最適化後のパラメータを改めて測定する
#
# COBYLAにはサンプリングで推定したエネルギーを渡すため、有限差分の勾配は不要です。目的関数にはショットノイズが含まれます。最終結果には最適化中の最小の推定値を流用せず、より多いショット数で新たに測定した値を使います。
#
# 読者向けの評価予算は結合距離ごとに目的関数200回分です。テスト用の予算は実行経路の確認のみを目的にしています。誤差棒は選ばれたパラメータでのサンプリングの不確かさを示し、最適化の未収束やansatzの表現力による誤差は含みません。そのため、FCIの値を含むとは限りません。
#
# 変分状態の真の期待値は基底状態エネルギー以上ですが、有限ショットの推定値は基準値を下回る場合があります。そのため、各標本平均に厳密な変分上界を要求せず、推定した標準誤差を表示します。誤差棒は標準誤差1個分であり、保証された信頼区間ではありません。

# %%
cost_history = []


def cost_function(values) -> float:
    """Record the sampled energy used by the classical optimizer."""
    estimate = estimate_energy(values, estimator)
    cost_history.append(estimate.value)
    return estimate.value


rng = np.random.default_rng(42)
initial_parameters = rng.uniform(0.0, np.pi, NUM_PARAMETERS)
max_iterations = 42 if docs_test_mode else 200
optimization = minimize(
    cost_function,
    initial_parameters,
    method="COBYLA",
    options={"maxiter": max_iterations, "rhobeg": 0.5},
)
final_estimate = estimate_energy(optimization.x, estimator, shots=FINAL_SHOTS)
fci_reference = float(molecule.fci_energy)
print(
    f"VQE energy: {final_estimate.value:+.6f} +/- {final_estimate.standard_error:.6f} (SE)"
)
print(f"FCI energy: {fci_reference:+.6f}")
print(f"Final measurement shots: {final_estimate.shots}")
assert len(optimization.x) == NUM_PARAMETERS
assert len(cost_history) == optimization.nfev

# %%
plt.plot(cost_history, label="Sampled VQE objective")
plt.errorbar(
    [len(cost_history) - 1],
    [final_estimate.value],
    yerr=[final_estimate.standard_error],
    fmt="o",
    capsize=4,
    label="Fresh final estimate (1 SE)",
)
plt.axhline(fci_reference, linestyle="dashed", color="black", label="FCI reference")
plt.xlabel("Energy evaluation")
plt.ylabel("Energy (Hartree)")
plt.legend()
plt.show()

# %% [markdown]
# ## 原子間距離を変える
#
# 各結合長でハミルトニアンを再計算し、推定器を作成します。各推定器は最適化を通じてコンパイル済み測定プログラムを再利用します。最適化した各点は、新しいショットで再評価します。ドキュメントテストでは1点、読者向けのノートブックでは15点の結合長を使います。

# %%
n_points = 1 if docs_test_mode else 15
bond_lengths = np.linspace(0.2, 1.5, n_points)
vqe_energies = []
vqe_errors = []
fci_energies = []
starting_parameters = optimization.x

for bond_length in bond_lengths:
    transformed, bond_molecule = make_hydrogen_molecule(float(bond_length))
    bond_hamiltonian = openfermion_to_qamomile(transformed, bond_molecule.n_qubits)
    bond_estimator = Estimator(hydrogen_vqe_state, bond_hamiltonian)
    bond_fci = bond_molecule.fci_energy
    assert bond_fci is not None
    bond_result = minimize(
        lambda values: estimate_energy(values, bond_estimator).value,
        starting_parameters,
        method="COBYLA",
        options={"maxiter": max_iterations, "rhobeg": 0.5},
    )
    starting_parameters = bond_result.x
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
        f"+/- {measured.standard_error:.6f} (SE), FCI={bond_fci:+.6f}"
    )

assert len(vqe_energies) == len(vqe_errors) == len(fci_energies) == n_points

# %%
plt.errorbar(
    bond_lengths,
    vqe_energies,
    yerr=vqe_errors,
    fmt="-o",
    capsize=3,
    label="Sampled VQE (1 SE)",
)
plt.plot(bond_lengths, fci_energies, "--", label="FCI reference")
plt.xlabel("H–H distance (Å)")
plt.ylabel("Energy (Hartree)")
plt.legend()
plt.show()

# %% [markdown]
# ## まとめ
#
# - OpenFermion/PySCFで4量子ビットのHamiltonianとFCI基準値を得ます。
# - 状態準備qkernelに、`Estimator`が測定回路を組み合わせます。
# - パラメータ更新には通常のコンパイル済みサンプリングプログラムを再利用します。
# - 古典的なパリティの集計、グループ内の共分散、独立グループ間の誤差の合成は`qamomile.statistics`が担当します。
# - 最終結果は新たな測定によるエネルギー推定値と誤差棒で示します。ショットをまたぐ集計はqkernelの外で行います。
