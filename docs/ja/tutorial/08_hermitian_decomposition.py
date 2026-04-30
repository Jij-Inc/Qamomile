# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: qamomile
#     language: python
#     name: qamomile
# ---

# %% [markdown]
# ---
# title: エルミート行列から量子回路へ
# tags: [hamiltonian-simulation, pauli-decomposition, algorithm, tutorial]
# ---
#
# # エルミート行列から量子回路へ
#
# <!-- BEGIN auto-tags -->
# **タグ:** <a class="tag-chip" href="../tags/hamiltonian-simulation.md">hamiltonian-simulation</a> <a class="tag-chip" href="../tags/pauli-decomposition.md">pauli-decomposition</a> <a class="tag-chip" href="../tags/algorithm.md">algorithm</a> <a class="tag-chip" href="../tags/tutorial.md">tutorial</a>
# <!-- END auto-tags -->
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
import numpy as np

import qamomile.circuit as qmc
import qamomile.observable as qmo
from qamomile.linalg import HermitianMatrix
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

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
print("Hermitian:", np.allclose(M, M.conj().T))

# %% [markdown]
# ## ラップして分解する
#
# `HermitianMatrix`は、構築時に配列を検証（2D、正方、2冪の次元、エルミート）し、`to_hamiltonian()`を公開する薄いラッパです。返される`Hamiltonian`は`qamomile.observable`のもので、`pauli_evolve`や`expval`、最適化ヘルパが使うものと同じ型です。
#
# Qamomileの分解では、**qubit 0が行列の計算基底インデックスの最下位ビット**に対応します。これはQiskitの内部順序と一致しています。

# %%
H_mat = HermitianMatrix(M)
print("num_qubits:", H_mat.num_qubits)

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
# 3つのbindingをすべて固定してトランスパイルします。Qiskit回路は`transpiler.to_circuit()`から素のQuantumCircuitとして返ってきます。

# %%
t_value = 0.8

qiskit_circuit = transpiler.to_circuit(
    time_evolution,
    bindings={
        "n": H_mat.num_qubits,
        "hamiltonian": H_op,
        "t": t_value,
    },
)
print(qiskit_circuit)

# %% [markdown]
# ## 厳密な指数関数との比較による検証
#
# 小さなエルミート行列であれば、numpyの固有値分解から直接$U = e^{-iMt}$を組み、同じ初期状態に作用させたうえで、Qamomileが生成した回路が同じ最終状態を（グローバル位相を除いて）与えることを確認できます。

# %%
import warnings

from qiskit.quantum_info import Statevector
from scipy.sparse import SparseEfficiencyWarning

# Qiskit の `PauliEvolutionGate.to_matrix()` — `pauli_evolve` が生成するゲートを
# 厳密なユニタリに変換するために `Statevector` から呼ばれます — は、CSC 形式で
# ない Hamiltonian に対して `scipy.sparse.linalg.expm` を使うため、ノイジーな
# `SparseEfficiencyWarning` を発生させます。実行済み notebook に絶対パスが
# 残らないよう、ここだけ抑制しています。(回路を decompose しても警告は消えます
# が、厳密な発展が Trotter 近似に置き換わり下のフィデリティチェックが破綻します。)
qiskit_unitary_circuit = qiskit_circuit.remove_final_measurements(inplace=False)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=SparseEfficiencyWarning)
    psi_qm = np.array(Statevector.from_instruction(qiskit_unitary_circuit).data)

psi0 = np.zeros(4, dtype=complex)
psi0[0] = 1.0 / np.sqrt(2.0)
psi0[1] = 1.0 / np.sqrt(2.0)

eigvals, eigvecs = np.linalg.eigh(M)
U = eigvecs @ np.diag(np.exp(-1j * t_value * eigvals)) @ eigvecs.conj().T
psi_exact = U @ psi0

fidelity = abs(np.vdot(psi_exact, psi_qm))
print(f"fidelity (|<exact|qamomile>|): {fidelity:.12f}")
assert abs(fidelity - 1.0) < 1e-8

# %% [markdown]
# フィデリティは数値的に$1$と区別できません。Pauli分解からQamomileが組み立てた回路は、直接の行列指数関数と同じユニタリを実現しています。
#
# ## まとめ
#
# - `HermitianMatrix`は密なエルミートnumpy配列を検証し、FWHTベースの分解でPauli和への変換を担います。
# - `to_hamiltonian()`は`qamomile.observable.Hamiltonian`を返します。これは`pauli_evolve`や`expval`など、量子アルゴリズム一式が使うのと同じ型です。
# - 両者が揃うことで、エルミート行列から出発するアルゴリズム（ハミルトニアンシミュレーション、VQE、block-encoding / LCUのエルミート側など）に対し、`np.ndarray → Hamiltonian → circuit`という直接経路が得られます。
#
# **非自己共役な作用素**（たとえば数値流体シミュレーションに現れる移流stencilなど）のサポートは自然な次のステップで、非エルミート行列の適切な統合ポイントが決まり次第、改めて扱う予定です。
