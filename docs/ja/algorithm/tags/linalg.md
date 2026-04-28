---
title: "タグ: linalg"
tags: [linalg]
---

# `linalg`

**`linalg`** タグが付いたアルゴリズム例 (1 件)。

[← 全アルゴリズムへ戻る](../index.md)

---

**タグで探す:** 
[`advanced`](./advanced.md) · [`built-in`](./built-in.md) · [`chemistry`](./chemistry.md) · [`decomposition`](./decomposition.md) · [`from-scratch`](./from-scratch.md) · [`graph`](./graph.md) · [`graph-partition`](./graph-partition.md) · [`ground-state`](./ground-state.md) · [`hamiltonian-simulation`](./hamiltonian-simulation.md) · [`intermediate`](./intermediate.md) · [`jijmodeling`](./jijmodeling.md) · [`maxcut`](./maxcut.md) · [`openfermion`](./openfermion.md) · [`optimization`](./optimization.md) · [`pauli`](./pauli.md) · [`qaoa`](./qaoa.md) · [`simulation`](./simulation.md) · [`suzuki`](./suzuki.md) · [`time-evolution`](./time-evolution.md) · [`trotter`](./trotter.md) · [`variational`](./variational.md) · [`vqe`](./vqe.md)

---

### [エルミート行列から量子回路へ](../hermitian_decomposition.ipynb)

**タグ:** [`hamiltonian-simulation`](./hamiltonian-simulation.md) [`pauli`](./pauli.md) [`decomposition`](./decomposition.md) [`time-evolution`](./time-evolution.md) [`intermediate`](./intermediate.md)

量子アルゴリズムの多くは、密な$2^n \times 2^n$のnumpy配列として与えられた**エルミート行列**（ハミルトニアン）から出発し、その時間発展$e^{-iHt}$を量子コンピュータ上でシミュレーションしたいという状況からはじまります。定石は以下の2ステップです。
