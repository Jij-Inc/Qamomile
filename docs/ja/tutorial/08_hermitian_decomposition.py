# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# tags: [tutorial, primitive, encoding, simulation]
# ---
#
# # Hermitian行列から量子回路へ
#
# 量子アルゴリズムでは、物理系や最適化問題をハミルトニアンで表します。
# ハミルトニアンはエルミート行列 ($H^\dagger = H$) として表され、量子状態の時間発展 $e^{-iHt}$ などに用いられます。
# Qamomile では、NumPy で作成された行列から、量子ビット上の演算子として用いるための `Hamiltonian` に変換することができます。
#
# ```text
# NumPy でエルミート行列を作成
#     ↓
# パウリ演算子の和で分解
#     ↓
# qkernel で利用
# ```
#
# 具体的に、この章では次の手順を行います。
#
# 1. 2量子ビットのエルミート行列を NumPy で作成
# 2. `HermitianMatrix.to_hamiltonian()` によりパウリ演算子の和に変換
# 3. 変換したハミルトニアンを `@qm.qkernel` の時間発展で用いる
# 4. シミュレータで測定結果を確認
#
# 最後に、興味がある方向けの発展例として、時間発展の近似精度も確認します。

# %%
import numpy as np
from qiskit.quantum_info import Statevector

import qamomile as qm
from qamomile.qiskit import to_circuit
from qamomile.stdlib import suzuki_trotter

# %% [markdown]
# ## 1. NumPy でエルミート行列を作成
#
# 例として、2 サイトの横磁場イジング模型を考えましょう。
# このハミルトニアンは、次のように与えられます。
#
# $$
# \label{tutorial08-ising-hamiltonian}
# H = -Z_0 Z_1 - h(X_0 + X_1), \qquad h = 0.7
# $$
#
# ここで $X_i, Z_i$ は、$i$ 番目のサイトに作用するパウリ演算子、そして $h$ は横磁場の強度です。\
# この2サイトのハミルトニアンは、$4\times4$ の行列で表されます。
# この行列の構築には `np.kron(A, B)` を使います。
# この関数は、テンソル積 $A \otimes B$ の行列表現であるクロネッカー積を計算します。\
# Qamomileでは、計算基底を $\lvert q_1 q_0 \rangle$ の順で表し、$\lvert 00\rangle, \lvert 01\rangle, \lvert 10\rangle, \lvert 11\rangle$ の順に並べます。
# ここで $q_i$ は量子ビット $i$ に対応するビットです。
# そのため、`np.kron(identity, pauli_x)` は量子ビット 0 への X 演算に、`np.kron(pauli_x, identity)` は量子ビット 1 への X 演算に対応します。

# %%
pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
pauli_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
identity = np.eye(2, dtype=np.complex128)

field_strength = 0.7
matrix = -np.kron(pauli_z, pauli_z) - field_strength * (
    np.kron(identity, pauli_x) + np.kron(pauli_x, identity)
)
operator = qm.linalg.HermitianMatrix(matrix)

print("Matrix shape:", matrix.shape)
print("Is Hermitian:", np.allclose(matrix, matrix.conj().T))
print(matrix)

# %% [markdown]
# ## 2. Qamomile の `Hamiltonian` へ変換
#
# 行列を `qm.linalg.HermitianMatrix(...)` に渡すことで、行列の形とエルミート性を検査することができます。
# $n$ 量子ビットには $2^n$ 通りの計算基底状態があるため、そのハミルトニアンは $2^n \times 2^n$ の行列で表されます。\
# 続いて `to_hamiltonian(...)` を呼び出し、行列をパウリ演算子の和で表しましょう。
# この例では、$Z_0Z_1$、$X_0$、$X_1$ の 3 項が得られます。
# `tol` は係数の小さいパウリ項を省くための閾値で、係数の絶対値が `tol` 以下の項を省きます。
# `tol=0.0` では、計算された係数がゼロでない項をすべて残します。
# この閾値は、後で扱う時間発展の近似とは別の設定です。

# %%
operator = qm.linalg.HermitianMatrix(matrix)
hamiltonian = operator.to_hamiltonian(tol=0.0)

print("Number of qubits:", operator.num_qubits)
print("Number of Pauli terms:", len(hamiltonian.terms))
print("Pauli decomposition:", hamiltonian)

expected = -qm.Z(0) * qm.Z(1) - field_strength * (qm.X(0) + qm.X(1))

assert operator.num_qubits == 2
assert hamiltonian == expected

# %% [markdown]
# 得られた 3 項の係数は
#
# * $X_0$ の係数: -0.7
# * $Z_0 Z_1$ の係数: -1.0
# * $X_1$ の係数: -0.7
#
# のように、[](#tutorial08-ising-hamiltonian) の係数と一致しています。
#
# ## 3. `Hamiltonian` で時間発展
#
# ここからは、先ほど得られた `hamiltonian` を用いて、時間発展の回路を作りましょう。
# ハミルトニアンを、相互作用 $A = - Z_0 Z_1$ と横磁場 $B = - h (X_0 + X_1)$ に分け、$H = A + B$ と書きましょう。
# $A$ と $B$ は可換ではないため、それぞれによる時間発展を順に適用しても、一般には $e^{-iHt}$ を正確に再現することができません。
# 鈴木–トロッター公式では、短い時間区間でのそれぞれの時間発展を組合せ、全体の時間発展を近似します。\
# 一方、横磁場部分$B$に含まれる$X_0$と$X_1$は異なる量子ビットに作用するため可換であり、$B$の時間発展は次のように厳密に分解できます。
#
# $$
# e^{-iB \Delta t}
# = e^{ihX_0 \Delta t} e^{ihX_1 \Delta t}
# $$
#
# 相互作用部分$A$はパウリ項が1つなので、$A$単独の時間発展も厳密に実現できます。
# この例で近似が入るのは、$A$と$B$の時間発展を組み合わせて、$H$全体の時間発展を再現する段階です。
# 相互作用の項を定義し、それを `hamiltonian` から引くことで、横磁場の部分を取り出すことができます。

# %%
interaction = -qm.Z(0) * qm.Z(1)
transverse_field = hamiltonian - interaction
evolve = suzuki_trotter(term_count=2, order=2, steps=2)


# %% [markdown]
# `suzuki_trotter(...)` は量子カーネルを返し、ここではそれを `evolve` に格納します。
# このカーネルは、量子状態・分割したハミルトニアン・時間発展の時間を受け取ります。
# この例では、2つのグループ（`term_count=2`）に対して 2 次の公式（`order=2`）を使い、全体の時間を2つの等しい区間（`steps=2`）に分けます。
#
# :::{note}
# `evolve(...)`に渡す各ハミルトニアングループでは、含まれるパウリ項が互いに可換である必要があります。異なるグループ同士は可換でなくても構いません。
# :::
#
# 次の `ising_state(...)` カーネルは 2 量子ビットの状態を準備し、`evolve(...)` を呼び出します。
# そして `simulate_ising(...)` カーネルは、そこに最後の測定を加えます。
# 測定を分けておくことで、後の節での量子状態の比較が可能になります。\
# 型注釈 `qm.Const[qm.Vector[qm.Observable, 2]]` は、演算子を表すデータ 2 個を特殊化時に固定するための指定です。
# カーネルの定義後、`qm.specialize(...)` で (`interaction`, `transverse_field`) を渡し、`program` を作成します。
# 時間発展の時間は `Param` として残し、実行ごとに指定します。


# %%
@qm.qkernel
def ising_state(
    hamiltonians: qm.Const[qm.Vector[qm.Observable, 2]],
    time: qm.Param[qm.Real],
) -> qm.Q[qm.Bits[2]]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = evolve(state, hamiltonians, time)
    return state


@qm.qkernel
def simulate_ising(
    hamiltonians: qm.Const[qm.Vector[qm.Observable, 2]],
    time: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[2]]:
    return qm.measure(ising_state(hamiltonians, time))


program = qm.specialize(
    simulate_ising,
    hamiltonians=(interaction, transverse_field),
)

# %% [markdown]
# ## 4. 測定結果の確認
#
# `parameters` 辞書に時間発展の時間として 0.8 を指定し、`qm.run(...)` で `program` を実行しましょう。\
# `on=qm.Reference()` とすることで、ローカルで実行される Qamomile の組み込みシミュレーターを指定しています。
# `shots=128` により、状態の準備・時間発展・測定を 128 回繰り返します。
# また、測定結果を再現できるように乱数の `seed` を固定しています。

# %%
result = qm.run(
    program,
    on=qm.Reference(),
    parameters={"time": 0.8},
    shots=128,
    seed=7,
).result

print("Number of shots:", result.shots)
print("Measurement counts:", result.counts)

assert result.shots == 128
assert sum(result.counts.values()) == 128

# %% [markdown]
# `result.counts` には、観測された各ビット列と、その出現回数が格納されています。
# このようにして、NumPy 行列から変換したハミルトニアンを用いて、量子回路の測定結果を得ることができました。
#
# ## 5. 時間発展の近似精度を調べる
#
# 同じ初期状態から得られる 2 つの最終状態を比較し、近似精度を調べてみましょう。
# 一方は $e^{-iHt}$ から計算した状態、もう一方は鈴木–トロッター公式による回路から得られる状態です。\
# フィデリティを計算するには、両方の状態ベクトルが必要です。
# 前の節で集めた測定結果の出現回数だけでは、これらの状態ベクトルは求められません。\
# そこで近似した回路の最終状態を、Qiskit で計算しましょう。
# `to_circuit(...)` で `ising_state` を Qiskit の回路に変換し、`Statevector.from_instruction(...)` で初期状態の準備を含めてシミュレートします。
# この小さな例では、比較の基準となる状態を、鈴木–トロッター近似を行わずに NumPy でハミルトニアンから直接計算することができます。\
# 2つの状態を、次の状態フィデリティで比較します。
#
# $$
# F = |\langle\psi_{\mathrm{exact}}\mid\psi_{\mathrm{approx}}\rangle|^2
# $$
#
# すなわち、$F$ の値が 1 に近いほど、2 つの状態がよく一致していることになります。

# %%
unitary_circuit = to_circuit(
    ising_state,
    hamiltonians=(interaction, transverse_field),
    parameters={"time": 0.8},
)
approximate_state = np.asarray(Statevector.from_instruction(unitary_circuit).data)

# |q1 q0> の順で表すと、初期状態は |0+> = (|00> + |01>) / sqrt(2) です。
initial_state = np.zeros(4, dtype=np.complex128)
initial_state[0] = 1.0 / np.sqrt(2.0)
initial_state[1] = 1.0 / np.sqrt(2.0)

eigenvalues, eigenvectors = np.linalg.eigh(matrix)
exact_unitary = (
    eigenvectors @ np.diag(np.exp(-1j * 0.8 * eigenvalues)) @ eigenvectors.conj().T
)
exact_state = exact_unitary @ initial_state

fidelity = float(abs(np.vdot(exact_state, approximate_state)) ** 2)
print(f"State fidelity: {fidelity:.6f}")
assert fidelity > 0.99

# %% [markdown]
# ## 6. まとめ
#
# このチュートリアルでは、次のやり方を学びました。
#
# * `qm.linalg.HermitianMatrix()` で、NumPy で準備したエルミート行列を読み込みます。
# * `to_hamiltonian()` により、パウリ演算子の和に分解します。
# * 得られたハミルトニアンを、`@qm.qkernel` 内の時間発展に利用できます。
# * `qm.specialize()` と `qm.run()` を用いて、測定結果を得ることができます。
