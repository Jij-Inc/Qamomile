# ---
# jupyter:
#   jupytext:
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

# %%
# 最新のQamomileをpipからインストールします！
# Colabで開いている場合は、下のタブで選んだTranspilerに合う行を1つ選び、行頭のコメントを外して実行してください:
# # !pip install qamomile openfermion pyscf openfermionpyscf                  # Qiskit（デフォルト）
# # !pip install "qamomile[quri_parts]" openfermion pyscf openfermionpyscf    # QURI Parts
# # !pip install "qamomile[cudaq-cu12]" openfermion pyscf openfermionpyscf    # CUDA-Q (CUDA 12.x toolchain。CUDA 13.xならcudaq-cu13)。Linux / macOS-arm64 / WSL2のみ。

# %% [markdown]
# この記事はデフォルトでQiskitを使います。Qamomileは同じ`@qkernel`を複数の量子SDKへトランスパイルできるので、下のimportを差し替えるだけで他のSDKでも同じ流れで進められます。記事本体のコードはどのSDKを選んでも同一です。Colabの場合は上のpipセルで対応する行のコメントを先に外しておいてください。
#
# ::::{tab-set}
# :::{tab-item} Qiskit
# :sync: qiskit
#
# ```python
# from qamomile.qiskit import QiskitTranspiler
#
# transpiler = QiskitTranspiler()
# ```
# :::
#
# :::{tab-item} QURI Parts
# :sync: quri_parts
#
# ```python
# from qamomile.quri_parts import QuriPartsTranspiler
#
# transpiler = QuriPartsTranspiler()
# ```
# :::
#
# :::{tab-item} CUDA-Q
# :sync: cudaq
#
# CUDA 12.x環境では`qamomile[cudaq-cu12]`、CUDA 13.x環境では`qamomile[cudaq-cu13]`を使ってください（インストール済みのCUDA Toolkitに合わせて選択）。CUDA-QはLinux、macOS arm64、Windows（WSL2経由）のみ対応です。
#
# ```python
# from qamomile.cudaq import CudaqTranspiler
#
# transpiler = CudaqTranspiler()
# ```
# :::
# ::::

# %%
# Transpiler — この記事はデフォルトでQiskitを使います。
# 上のタブでQURI PartsまたはCUDA-Qを選んだ場合は、そのタブに書かれた
# 2行（importとtranspiler = ...）を以下の2行と入れ替えてください。
# あわせて、上のpipセルで対応する行のコメントも外しておくこと。
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %%
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import openfermion.chem as of_chem
import openfermion.transforms as of_trans
import openfermionpyscf as of_pyscf
from scipy.optimize import minimize

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.basic import cx_entangling_layer, ry_layer, rz_layer

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
reps = 4

executable = transpiler.transpile(
    vqe_ansatz,
    bindings={"n": n_qubit, "reps": reps, "H": hamiltonian},
    parameters=["thetas"],
)

# トランスパイル後の量子回路（SDK固有のオブジェクト。`.draw("mpl")` は
# Qiskit固有なので、SDK間で動くように型名だけprintします。実際の図が見たい時は
# 各SDKの描画APIを使ってください）。
print(type(executable.quantum_circuit).__name__)

# %% [markdown]
# 期待値を計算できるexecutorが必要です（`vqe_ansatz`内の`expval`はパラメトリックな`Float`を返し、最適化ループはそれを最小化します）。配線のしかたは選んだSDKによって違うので、上のタブで別のSDKを選んだ場合は下のタブブロックから対応するスニペットをコピペしてください。
#
# ::::{tab-set}
# :::{tab-item} Qiskit
# :sync: qiskit
#
# ```python
# from qiskit_aer.primitives import EstimatorV2
# from qamomile.qiskit.transpiler import QiskitExecutor
#
# executor = QiskitExecutor(estimator=EstimatorV2())
# ```
#
# `EstimatorV2`はQiskitの現行世代のprimitiveで、サンプリングを介さず直接期待値を計算します。
# :::
#
# :::{tab-item} QURI Parts
# :sync: quri_parts
#
# ```python
# from qamomile.quri_parts import QuriPartsExecutor
#
# # QuriPartsExecutor は estimator が指定されない場合、qulacs バックエンドの
# # パラメトリック estimator を遅延構築します。VQE ループにはそのデフォルトで十分です。
# executor = QuriPartsExecutor()
# ```
# :::
#
# :::{tab-item} CUDA-Q
# :sync: cudaq
#
# ```python
# from qamomile.cudaq import CudaqExecutor
#
# # CudaqExecutor は cudaq.observe を内部で呼んで期待値を計算するため、
# # 明示的な estimator の配線は不要です。
# executor = CudaqExecutor()
# ```
# :::
# ::::

# %%
# Executor — この記事はデフォルトで Qiskit の EstimatorV2 を使って期待値を計算します。
# 上のタブで別のSDKを選んだ場合は、対応するタブのスニペットで以下を上書きしてください
# （あわせて記事冒頭のpipセルで対応する行のコメントも外しておくこと）。
from qiskit_aer.primitives import EstimatorV2

from qamomile.qiskit.transpiler import QiskitExecutor

executor = QiskitExecutor(estimator=EstimatorV2())

# %%
cost_history = []


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
