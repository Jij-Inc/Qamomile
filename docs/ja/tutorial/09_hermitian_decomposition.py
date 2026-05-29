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
# tags: [tutorial, primitive, encoding, simulation]
# ---
#
# # エルミート行列から量子回路へ
#
# 量子アルゴリズムの多くは、密な$2^n \times 2^n$のnumpy配列として与えられた**エルミート行列**（ハミルトニアン）から出発し、その時間発展$e^{-iHt}$を量子コンピュータ上でシミュレーションしたいという状況からはじまります。定石は以下の2ステップです。
#
# 1. $H$をPauli文字列の重み付き和に分解する。
# 2. その和を`pauli_evolve`に渡し、対応する量子回路を生成する。
#
# Qamomileではこれを小さな型安全パイプラインとして提供しています。
#
# ```
# np.ndarray  →  HermitianMatrix  →  Hamiltonian  →  pauli_evolve  →  circuit
# ```
#
# このチュートリアルでは以下を行います。
#
# - numpyで小さなエルミート行列を作る
# - `qamomile.linalg.HermitianMatrix`でラップして`to_hamiltonian()`を呼ぶ
# - 得られた`Hamiltonian`を`@qkernel`内の`pauli_evolve`で使う
# - 最終statevectorを厳密な行列指数関数と比較して検証する
#
# 分解の内部では**Fast Walsh-Hadamard Transform**を使い、NumPyだけで$O(n \cdot 4^n)$時間で走ります。Qiskit依存はありません。

# %%
# 最新のQamomileをpipからインストールします！
# Colabで開いている場合は、下のタブで選んだTranspilerに合う行を1つ選び、行頭のコメントを外して実行してください:
# # !pip install qamomile                  # Qiskit（デフォルト）
# # !pip install "qamomile[quri_parts]"    # QURI Parts
# # !pip install "qamomile[cudaq-cu12]"    # CUDA-Q (CUDA 12.x toolchain。CUDA 13.xなら`qamomile[cudaq-cu13]`)。Linux / macOS-arm64 / WSL2のみ。

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
import numpy as np

import qamomile.circuit as qmc
import qamomile.observable as qmo
from qamomile.linalg import HermitianMatrix

# %% [markdown]
# ## 小さなエルミート行列
#
# 例として2サイトの横磁場Isingモデルを使います。
#
# $$
# M \;=\; -Z_0 Z_1 \;-\; h \, (X_0 + X_1),
# $$
#
# 横磁場は$h = 0.7$とします。ここはQamomile固有の要素は何もなく、`np.kron`で普通の$4 \times 4$のnumpy配列を組むだけです。

# %%
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

h_field = 0.7
M = -np.kron(Z, Z) - h_field * (np.kron(I2, X) + np.kron(X, I2))
print("shape:", M.shape)
assert M.shape == (4, 4)
print("Hermitian:", np.allclose(M, M.conj().T))
assert np.allclose(M, M.conj().T)

# %% [markdown]
# ## ラップして分解する
#
# `HermitianMatrix`は、構築時に配列を検証（2D、正方、2冪の次元、エルミート）し、`to_hamiltonian()`を公開する薄いラッパです。返される`Hamiltonian`は`qamomile.observable`のもので、`pauli_evolve`や`expval`、最適化ヘルパが使うものと同じ型です。
#
# Qamomileの分解では、**qubit 0が行列の計算基底インデックスの最下位ビット**に対応します。これはQiskitの内部順序と一致しています。

# %%
H_mat = HermitianMatrix(M)
print("num_qubits:", H_mat.num_qubits)
# 4x4 行列 → log2(4) = 2 量子ビット。
assert H_mat.num_qubits == 2

H_op = H_mat.to_hamiltonian()
print("constant:", H_op.constant)
for ops, coeff in H_op.terms.items():
    print(f"  {ops}: {coeff:+.3f}")

# %% [markdown]
# 2サイト横磁場Isingモデルに対しては、ちょうど3つの非零項が期待されます。係数$-1$の$Z_0 Z_1$と、係数$-h$の単一qubit $X$が2つです。

# %%
expected_zz = (
    qmo.PauliOperator(qmo.Pauli.Z, 0),
    qmo.PauliOperator(qmo.Pauli.Z, 1),
)
expected_x0 = (qmo.PauliOperator(qmo.Pauli.X, 0),)
expected_x1 = (qmo.PauliOperator(qmo.Pauli.X, 1),)

assert set(H_op.terms.keys()) == {expected_zz, expected_x0, expected_x1}
assert abs(H_op.terms[expected_zz] - (-1.0)) < 1e-12
assert abs(H_op.terms[expected_x0] - (-h_field)) < 1e-12
assert abs(H_op.terms[expected_x1] - (-h_field)) < 1e-12
assert H_op.constant == 0.0

# %% [markdown]
# ## `@qkernel`での時間発展
#
# `pauli_evolve(q, H, t)`はqubitレジスタに$e^{-iHt}$を作用させます。`H`は`qmc.Observable`ハンドル型で宣言され、トランスパイル時にbindings経由で渡されます。`t`はbindingでもよいですし、スイープ可能なパラメータのまま残しても構いません。
#
# ここでは$\lvert 00\rangle$から出発し、qubit 0にHadamardで重ね合わせを作ってから時間発展のステップを適用します。最後に測定をつけているのは、`transpile()` / `to_circuit()`に渡すエントリポイントのカーネルが古典出力を持つ必要があるためです。


# %%
@qmc.qkernel
def time_evolution(
    n: qmc.UInt,
    hamiltonian: qmc.Observable,
    t: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")
    q[0] = qmc.h(q[0])
    q = qmc.pauli_evolve(q, hamiltonian, t)
    return qmc.measure(q)


# %% [markdown]
# 3つのbindingをすべて固定してトランスパイルします。`transpiler.to_circuit()`はSDK固有の回路オブジェクト（例: Qiskitなら`QuantumCircuit`）を返します。返ってくる型は上で選んだTranspilerに応じて変わります。

# %%
t_value = 0.8

circuit = transpiler.to_circuit(
    time_evolution,
    bindings={
        "n": H_mat.num_qubits,
        "hamiltonian": H_op,
        "t": t_value,
    },
)
print(type(circuit).__name__)

# %% [markdown]
# ## 厳密な指数関数との比較による検証
#
# 小さなエルミート行列であれば、numpyの固有値分解から直接$U = e^{-iMt}$を組み、同じ初期状態に作用させたうえで、Qamomileが生成した回路が同じ最終状態を（グローバル位相を除いて）与えることを確認できます。

# %%
# 解析的なリファレンス — pure numpy、SDK非依存です。
psi0 = np.zeros(4, dtype=complex)
psi0[0] = 1.0 / np.sqrt(2.0)
psi0[1] = 1.0 / np.sqrt(2.0)

eigvals, eigvecs = np.linalg.eigh(M)
U = eigvecs @ np.diag(np.exp(-1j * t_value * eigvals)) @ eigvecs.conj().T
psi_exact = U @ psi0

# %% [markdown]
# トランスパイル後の回路から状態ベクトルを取り出す方法、そして`psi_exact`に対してどれだけ厳しく assert できるかは SDK ごとに違います。Qamomile の Qiskit emit pass は `pauli_evolve` を Qiskit 標準の `PauliEvolutionGate`（解析的な行列指数関数評価）に流すので、生成されるユニタリは厳密に $e^{-iHt}$、fidelity は 1 に丸まります。QURI Parts と CUDA-Q の emit pass にはまだ対応する native 経路がなく、現状は 1 次 Trotter にフォールバックしている — そのため項が非可換だと近似ユニタリ（この2量子ビット例 $t=0.8$ では fidelity 約 0.9）になります。Qiskit タブは厳格 assertion、他 2 つは fidelity の下限テストにしているのはこのためです。
#
# ::::{tab-set}
# :::{tab-item} Qiskit
# :sync: qiskit
#
# ```python
# import warnings
#
# from qiskit.quantum_info import Statevector
# from scipy.sparse import SparseEfficiencyWarning
#
# # Qiskit の PauliEvolutionGate.to_matrix() — pauli_evolve が生成するゲートを
# # 厳密なユニタリに変換するために Statevector から呼ばれます — は、CSC 形式で
# # ない Hamiltonian に対して scipy.sparse.linalg.expm を使うため、ノイジーな
# # SparseEfficiencyWarning を発生させます。実行済み notebook に絶対パスが
# # 残らないよう、ここだけ抑制しています。
# unitary_circuit = circuit.remove_final_measurements(inplace=False)
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore", category=SparseEfficiencyWarning)
#     psi_qm = np.array(Statevector.from_instruction(unitary_circuit).data)
#
# fidelity = abs(np.vdot(psi_exact, psi_qm))
# print(f"fidelity (|<exact|qamomile>|): {fidelity:.12f}")
# # Qiskit emit は解析的な PauliEvolutionGate を使うため、ユニタリは厳密に一致。
# assert abs(fidelity - 1.0) < 1e-8
# ```
# :::
#
# :::{tab-item} QURI Parts
# :sync: quri_parts
#
# ```python
# from quri_parts.core.state import GeneralCircuitQuantumState
# from quri_parts.qulacs.simulator import evaluate_state_to_vector
#
# state = GeneralCircuitQuantumState(circuit.qubit_count, circuit)
# psi_qm = np.asarray(evaluate_state_to_vector(state).vector)
#
# fidelity = abs(np.vdot(psi_exact, psi_qm))
# print(f"fidelity (|<exact|qamomile>|): {fidelity:.12f}")
# # Qamomile の QURI Parts emit は現状 `pauli_evolve` をデフォルトの phase-gadget
# # 分解（実質1次 Trotter）に落とし込みます。Hamiltonian の項が非可換だと
# # ユニタリは近似（この2量子ビット例 t=0.8 で fidelity 約 0.9）になります。
# # よってここでは厳格な等号ではなく下限テストにします。QURI Parts emit pass に
# # ネイティブ exponentiation 経路（UnitaryMatrix ゲートに `expm(-i H t)` を
# # 注入する等）が入れば、Qiskit と同じ厳格 assertion に戻せます。
# assert fidelity > 0.85
# ```
# :::
#
# :::{tab-item} CUDA-Q
# :sync: cudaq
#
# ```python
# import cudaq
#
# state = cudaq.get_state(circuit.kernel_func)
# psi_qm = np.asarray(state)
#
# fidelity = abs(np.vdot(psi_exact, psi_qm))
# print(f"fidelity (|<exact|qamomile>|): {fidelity:.12f}")
# # QURI Parts と同じ事情で、Qamomile の CUDA-Q emit は現状デフォルトの
# # phase-gadget 分解（1次 Trotter）を使うため、ユニタリは近似（fidelity 約 0.9）。
# # CUDA-Q カーネル面には QURI Parts の UnitaryMatrix に相当する公開 API が
# # ないため、解析的に厳密な経路には別途設計が要る — その間は下限テスト。
# assert fidelity > 0.85
# ```
# :::
# ::::

# %%
# Statevector extraction + fidelity check（デフォルトは Qiskit）。
# 上のタブで別のSDKを選んだ場合は、対応するタブの**全部**を以下の行で
# 上書きしてください — assert の許容値もSDKによって違うので、必ず assert
# 行も含めてコピーすること。あわせて記事冒頭のpipセルで対応する行のコメントも外しておくこと。
import warnings

from qiskit.quantum_info import Statevector
from scipy.sparse import SparseEfficiencyWarning

unitary_circuit = circuit.remove_final_measurements(inplace=False)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=SparseEfficiencyWarning)
    psi_qm = np.array(Statevector.from_instruction(unitary_circuit).data)

fidelity = abs(np.vdot(psi_exact, psi_qm))
print(f"fidelity (|<exact|qamomile>|): {fidelity:.12f}")
assert abs(fidelity - 1.0) < 1e-8

# %% [markdown]
# Qiskit では厳格な $1\\mathrm{e}{-8}$ 許容差で assertion が通ります — Qamomile の Qiskit emit pass が `pauli_evolve` を `PauliEvolutionGate` に流すので、生成される ユニタリは厳密に $e^{-iHt}$ です。QURI Parts と CUDA-Q では同じ分解が現状 1 次 Trotter で emit されるため、この例の fidelity は 0.9 付近に留まります（QURI Parts と CUDA-Q の emit pass に native 解析経路が入るまで）。
#
# ## まとめ
#
# - `HermitianMatrix`は密なエルミートnumpy配列を検証し、FWHTベースの分解でPauli和への変換を担います。
# - `to_hamiltonian()`は`qamomile.observable.Hamiltonian`を返します。これは`pauli_evolve`や`expval`など、量子アルゴリズム一式が使うのと同じ型です。
# - 両者が揃うことで、エルミート行列から出発するアルゴリズム（ハミルトニアンシミュレーション、VQE、block-encoding / LCUのエルミート側など）に対し、`np.ndarray → Hamiltonian → circuit`という直接経路が得られます。
#
# **非自己共役な作用素**（たとえば数値流体シミュレーションに現れる移流stencilなど）のサポートは自然な次のステップで、非エルミート行列の適切な統合ポイントが決まり次第、改めて扱う予定です。
