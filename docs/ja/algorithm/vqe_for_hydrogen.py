# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: qamomile
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# title: 水素分子のためのVQE
# tags: [chemistry, variational]
# ---
#
# # 水素分子のための変分量子固有値ソルバー（VQE）
#
# <!-- BEGIN auto-tags -->
# **タグ:** <a class="tag-chip" href="../tags/chemistry.md">chemistry</a> <a class="tag-chip" href="../tags/variational.md">variational</a>
# <!-- END auto-tags -->
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

# %%
# 必要なパッケージは以下のコマンドでインストールできます
# # !pip install openfermion pyscf openfermionpyscf

# %%
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import openfermion.chem as of_chem
import openfermion.transforms as of_trans
import openfermionpyscf as of_pyscf
from qiskit_aer.primitives import EstimatorV2
from scipy.optimize import minimize

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.basic import cx_entangling_layer, ry_layer, rz_layer
from qamomile.qiskit import QiskitTranspiler
from qamomile.qiskit.transpiler import QiskitExecutor

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# ## 水素分子のハミルトニアンの作成

# %%
basis = "sto-3g"
multiplicity = 1
charge = 0
distance = 0.977
geometry = [["H", [0, 0, 0]], ["H", [0, 0, distance]]]
description = "tmp"
molecule = of_chem.MolecularData(geometry, basis, multiplicity, charge, description)
molecule = of_pyscf.run_pyscf(molecule, run_scf=True, run_fci=True)
n_qubit = molecule.n_qubits
n_electron = molecule.n_electrons
fermionic_hamiltonian = of_trans.get_fermion_operator(molecule.get_molecular_hamiltonian())
jw_hamiltonian = of_trans.jordan_wigner(fermionic_hamiltonian)


# %% [markdown]
# ## Qamomile ハミルトニアンへの変換
#
# このセクションでは、OpenFermionのハミルトニアンをQamomileフォーマットに変換します。Jordan-Wigner変換を適用してフェルミ粒子演算子を量子ビット演算子へ変換し、その後、カスタム変換関数を用いてQamomileに適したハミルトニアン表現を作成します。

# %%
def operator_to_qamomile(operators: tuple[tuple[int, str], ...]) -> qm_o.Hamiltonian:
    pauli = {"X": qm_o.X, "Y": qm_o.Y, "Z": qm_o.Z}
    H = qm_o.Hamiltonian()
    H.constant = 1.0
    for ope in operators:
        H *= pauli[ope[1]](ope[0])
    return H

def openfermion_to_qamomile(of_h) -> qm_o.Hamiltonian:
    H = qm_o.Hamiltonian()
    for k, v in of_h.terms.items():
        if len(k) == 0:
            H.constant += v
        else:
            H += operator_to_qamomile(k) * v
    return H

hamiltonian = openfermion_to_qamomile(jw_hamiltonian)


# %% [markdown]
# ## VQE アンザッツの作成
#
# このセクションでは、VQEアルゴリズムのための EfficientSU2 アンザッツを `@qkernel` デコレータを用いて作成します。アンザッツとは、試行波動関数を準備するパラメータ付き量子回路です。`ry_layer`、`rz_layer` および線形 CX エンタングル層を組み合わせて構築し、最後に `expval` でハミルトニアンの期待値を計算します。

# %%
@qmc.qkernel
def vqe_ansatz(
    n: qmc.UInt,
    reps: qmc.UInt,
    thetas: qmc.Vector[qmc.Float],
    H: qmc.Observable,
) -> qmc.Float:
    q = qmc.qubit_array(n, name="q")
    for r in qmc.range(reps):
        base = r * 2 * n
        q = ry_layer(q, thetas, base)
        q = rz_layer(q, thetas, base + n)
        q = cx_entangling_layer(q)
    # Final rotation layer
    final_base = reps * 2 * n
    q = ry_layer(q, thetas, final_base)
    q = rz_layer(q, thetas, final_base + n)
    return qmc.expval(q, H)


# %% [markdown]
# ## Qiskitを用いたVQEの実行
#
# このセクションでは、`QiskitTranspiler` を使って VQE カーネルを実行可能オブジェクトにトランスパイルします。デフォルトの executor がこのオブジェクトを実行し、qkernel で定義した `expval` による期待値を返します。そのため、ユーザーは最適化ループのみ実装すれば問題ありません。

# %%
transpiler = QiskitTranspiler()
reps = 4

executable = transpiler.transpile(
    vqe_ansatz,
    bindings={"n": n_qubit, "reps": reps, "H": hamiltonian},
    parameters=["thetas"],
)

# Transpiled quantum circuit
executable.quantum_circuit.draw("mpl")

# %%
cost_history = []
executor = QiskitExecutor(estimator=EstimatorV2())


def cost_fn(param_values):
    job = executable.run(executor, bindings={"thetas": list(param_values)})
    return job.result()


def cost_callback(param_values):
    cost_history.append(cost_fn(param_values))


num_params = len(executable.parameter_names)
rng = np.random.default_rng(42)
initial_params = rng.uniform(0, np.pi, num_params)

# VQE 最適化を実行します
maxiter = 1 if docs_test_mode else 50
warnings.filterwarnings("ignore", message="Maximum number of iterations")
result = minimize(
    cost_fn,
    initial_params,
    method="BFGS",
    options={"disp": True, "maxiter": maxiter, "gtol": 1e-6},
    callback=cost_callback,
)
print(result)

# %%
plt.plot(cost_history)
plt.plot(
    range(len(cost_history)),
    [molecule.fci_energy] * len(cost_history),
    linestyle="dashed",
    color="black",
    label="Exact Solution",
)
plt.legend()
plt.show()


# %% [markdown]
# ## 原子同士の距離を変更する

# %%
def hydrogen_molecule(bond_length):
    basis = "sto-3g"
    multiplicity = 1
    charge = 0
    geometry = [["H", [0, 0, 0]], ["H", [0, 0, bond_length]]]
    description = "tmp"
    molecule = of_chem.MolecularData(geometry, basis, multiplicity, charge, description)
    molecule = of_pyscf.run_pyscf(molecule, run_scf=True, run_fci=True)
    fermionic_hamiltonian = of_trans.get_fermion_operator(
        molecule.get_molecular_hamiltonian()
    )
    jw_hamiltonian = of_trans.jordan_wigner(fermionic_hamiltonian)
    return openfermion_to_qamomile(jw_hamiltonian), molecule.fci_energy

n_points = 3 if docs_test_mode else 15
bond_lengths = np.linspace(0.2, 1.5, n_points)
energies = []
for bond_length in bond_lengths:
    hamiltonian, fci_energy = hydrogen_molecule(bond_length)

    executable = transpiler.transpile(
        vqe_ansatz,
        bindings={"n": hamiltonian.num_qubits, "reps": reps, "H": hamiltonian},
        parameters=["thetas"],
    )

    num_params = len(executable.parameter_names)
    initial_params = rng.uniform(0, np.pi, num_params)
    result = minimize(
        cost_fn,
        initial_params,
        method="BFGS",
        options={"maxiter": maxiter, "gtol": 1e-6},
    )

    energies.append(result.fun)

    print("distance: ", bond_length, "energy: ", result.fun, "fci_energy: ", fci_energy)

# %%
plt.plot(bond_lengths, energies, "-o")
plt.xlabel("Distance")
plt.ylabel("Energy")
plt.show()
