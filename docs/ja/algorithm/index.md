---
slug: algorithm
title: アルゴリズム
---

# アルゴリズム

Qamomile で実装した具体的な量子アルゴリズム例です。下のタグをクリックして絞り込むか、全アルゴリズムから選んでください。

## タグで探す

[`advanced`](tags/advanced.md) (1) · [`built-in`](tags/built-in.md) (1) · [`chemistry`](tags/chemistry.md) (1) · [`decomposition`](tags/decomposition.md) (1) · [`from-scratch`](tags/from-scratch.md) (1) · [`graph`](tags/graph.md) (2) · [`graph-partition`](tags/graph-partition.md) (1) · [`ground-state`](tags/ground-state.md) (1) · [`hamiltonian-simulation`](tags/hamiltonian-simulation.md) (2) · [`intermediate`](tags/intermediate.md) (4) · [`jijmodeling`](tags/jijmodeling.md) (1) · [`linalg`](tags/linalg.md) (1) · [`maxcut`](tags/maxcut.md) (1) · [`openfermion`](tags/openfermion.md) (1) · [`optimization`](tags/optimization.md) (2) · [`pauli`](tags/pauli.md) (2) · [`qaoa`](tags/qaoa.md) (2) · [`simulation`](tags/simulation.md) (1) · [`suzuki`](tags/suzuki.md) (1) · [`time-evolution`](tags/time-evolution.md) (2) · [`trotter`](tags/trotter.md) (1) · [`variational`](tags/variational.md) (3) · [`vqe`](tags/vqe.md) (1)

## 全アルゴリズム

### [QAOA によるグラフ分割](qaoa_graph_partition.ipynb)

**タグ:** [`qaoa`](tags/qaoa.md) [`optimization`](tags/optimization.md) [`variational`](tags/variational.md) [`graph`](tags/graph.md) [`graph-partition`](tags/graph-partition.md) [`jijmodeling`](tags/jijmodeling.md) [`built-in`](tags/built-in.md) [`intermediate`](tags/intermediate.md)

本チュートリアルでは、Quantum Approximate Optimization Algorithm（QAOA）を用いて**グラフ分割問題**を解く方法を紹介します。

### [QAOAでMaxCutを解く](qaoa_maxcut.ipynb)

**タグ:** [`qaoa`](tags/qaoa.md) [`optimization`](tags/optimization.md) [`variational`](tags/variational.md) [`graph`](tags/graph.md) [`maxcut`](tags/maxcut.md) [`from-scratch`](tags/from-scratch.md) [`intermediate`](tags/intermediate.md)

このチュートリアルでは、Qamomileの低レベル回路プリミティブを使って、QAOA (Quantum Approximate Optimization Algorithm) のパイプラインをステップごとに構築します。高レベルな`QAOAConverter`は使わずに、以下の手順で進めます:

### [Suzuki–Trotter によるハミルトニアンシミュレーション](hamiltonian_simulation.ipynb)

**タグ:** [`hamiltonian-simulation`](tags/hamiltonian-simulation.md) [`trotter`](tags/trotter.md) [`suzuki`](tags/suzuki.md) [`simulation`](tags/simulation.md) [`time-evolution`](tags/time-evolution.md) [`pauli`](tags/pauli.md) [`advanced`](tags/advanced.md)

量子系の時間発展$e^{-iHt}$をシミュレーションすることは、量子コンピュータの代表的な応用の1つです。ハミルトニアンが非可換な部分に分割されるとき、つまり$H = A + B$かつ$[A, B] \neq 0$のとき、素朴な分解$e^{-i(A+B)t} = e^{-iAt}\,e^{-iBt}$は成立しません。標準的な対処法が**Trotter–Suzuki積公式**で、各項の短時間発展を交互に並べます。誤差は、ステップ幅を小さくすれば減り(Lie–Trotter、1次)、ステップを対称化すれば減り(Strang、2次)、対称ステップをSuzukiの構成で再帰的に入れ子にすればさらに減ります(任意の偶数次)。

### [エルミート行列から量子回路へ](hermitian_decomposition.ipynb)

**タグ:** [`hamiltonian-simulation`](tags/hamiltonian-simulation.md) [`pauli`](tags/pauli.md) [`decomposition`](tags/decomposition.md) [`time-evolution`](tags/time-evolution.md) [`linalg`](tags/linalg.md) [`intermediate`](tags/intermediate.md)

量子アルゴリズムの多くは、密な$2^n \times 2^n$のnumpy配列として与えられた**エルミート行列**（ハミルトニアン）から出発し、その時間発展$e^{-iHt}$を量子コンピュータ上でシミュレーションしたいという状況からはじまります。定石は以下の2ステップです。

### [水素分子のためのVQE](vqe_for_hydrogen.ipynb)

**タグ:** [`vqe`](tags/vqe.md) [`variational`](tags/variational.md) [`chemistry`](tags/chemistry.md) [`ground-state`](tags/ground-state.md) [`openfermion`](tags/openfermion.md) [`intermediate`](tags/intermediate.md)

このチュートリアルでは、水素分子（H₂）の基底状態エネルギーを求めるための変分量子固有値ソルバー（VQE）アルゴリズムの実装について解説します。分子ハミルトニアンの生成には [OpenFermion](https://quantumai.google/openfermion) を使用します。
