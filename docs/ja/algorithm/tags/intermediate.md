---
title: "タグ: intermediate"
tags: [intermediate]
---

# `intermediate`

**`intermediate`** タグが付いたアルゴリズム例 (4 件)。

[← 全アルゴリズムへ戻る](../index.md)

---

**タグで探す:** 
[`advanced`](./advanced.md) · [`built-in`](./built-in.md) · [`chemistry`](./chemistry.md) · [`decomposition`](./decomposition.md) · [`from-scratch`](./from-scratch.md) · [`graph`](./graph.md) · [`graph-partition`](./graph-partition.md) · [`ground-state`](./ground-state.md) · [`hamiltonian-simulation`](./hamiltonian-simulation.md) · [`jijmodeling`](./jijmodeling.md) · [`linalg`](./linalg.md) · [`maxcut`](./maxcut.md) · [`openfermion`](./openfermion.md) · [`optimization`](./optimization.md) · [`pauli`](./pauli.md) · [`qaoa`](./qaoa.md) · [`simulation`](./simulation.md) · [`suzuki`](./suzuki.md) · [`time-evolution`](./time-evolution.md) · [`trotter`](./trotter.md) · [`variational`](./variational.md) · [`vqe`](./vqe.md)

---

### [QAOA によるグラフ分割](../qaoa_graph_partition.ipynb)

**タグ:** [`qaoa`](./qaoa.md) [`optimization`](./optimization.md) [`variational`](./variational.md) [`graph`](./graph.md) [`graph-partition`](./graph-partition.md) [`jijmodeling`](./jijmodeling.md) [`built-in`](./built-in.md)

本チュートリアルでは、Quantum Approximate Optimization Algorithm（QAOA）を用いて**グラフ分割問題**を解く方法を紹介します。

### [QAOAでMaxCutを解く](../qaoa_maxcut.ipynb)

**タグ:** [`qaoa`](./qaoa.md) [`optimization`](./optimization.md) [`variational`](./variational.md) [`graph`](./graph.md) [`maxcut`](./maxcut.md) [`from-scratch`](./from-scratch.md)

このチュートリアルでは、Qamomileの低レベル回路プリミティブを使って、QAOA (Quantum Approximate Optimization Algorithm) のパイプラインをステップごとに構築します。高レベルな`QAOAConverter`は使わずに、以下の手順で進めます:

### [エルミート行列から量子回路へ](../hermitian_decomposition.ipynb)

**タグ:** [`hamiltonian-simulation`](./hamiltonian-simulation.md) [`pauli`](./pauli.md) [`decomposition`](./decomposition.md) [`time-evolution`](./time-evolution.md) [`linalg`](./linalg.md)

量子アルゴリズムの多くは、密な$2^n \times 2^n$のnumpy配列として与えられた**エルミート行列**（ハミルトニアン）から出発し、その時間発展$e^{-iHt}$を量子コンピュータ上でシミュレーションしたいという状況からはじまります。定石は以下の2ステップです。

### [水素分子のためのVQE](../vqe_for_hydrogen.ipynb)

**タグ:** [`vqe`](./vqe.md) [`variational`](./variational.md) [`chemistry`](./chemistry.md) [`ground-state`](./ground-state.md) [`openfermion`](./openfermion.md)

このチュートリアルでは、水素分子（H₂）の基底状態エネルギーを求めるための変分量子固有値ソルバー（VQE）アルゴリズムの実装について解説します。分子ハミルトニアンの生成には [OpenFermion](https://quantumai.google/openfermion) を使用します。
