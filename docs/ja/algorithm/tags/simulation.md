---
title: "タグ: simulation"
tags: [simulation]
---

# `simulation`

**`simulation`** タグが付いたアルゴリズム例 (1 件)。

[← 全アルゴリズムへ戻る](../index.md)

---

**タグで探す:** 
[`advanced`](./advanced.md) · [`built-in`](./built-in.md) · [`chemistry`](./chemistry.md) · [`decomposition`](./decomposition.md) · [`from-scratch`](./from-scratch.md) · [`graph`](./graph.md) · [`graph-partition`](./graph-partition.md) · [`ground-state`](./ground-state.md) · [`hamiltonian-simulation`](./hamiltonian-simulation.md) · [`intermediate`](./intermediate.md) · [`jijmodeling`](./jijmodeling.md) · [`linalg`](./linalg.md) · [`maxcut`](./maxcut.md) · [`openfermion`](./openfermion.md) · [`optimization`](./optimization.md) · [`pauli`](./pauli.md) · [`qaoa`](./qaoa.md) · [`suzuki`](./suzuki.md) · [`time-evolution`](./time-evolution.md) · [`trotter`](./trotter.md) · [`variational`](./variational.md) · [`vqe`](./vqe.md)

---

### [Suzuki–Trotter によるハミルトニアンシミュレーション](../hamiltonian_simulation.ipynb)

**タグ:** [`hamiltonian-simulation`](./hamiltonian-simulation.md) [`trotter`](./trotter.md) [`suzuki`](./suzuki.md) [`time-evolution`](./time-evolution.md) [`pauli`](./pauli.md) [`advanced`](./advanced.md)

量子系の時間発展$e^{-iHt}$をシミュレーションすることは、量子コンピュータの代表的な応用の1つです。ハミルトニアンが非可換な部分に分割されるとき、つまり$H = A + B$かつ$[A, B] \neq 0$のとき、素朴な分解$e^{-i(A+B)t} = e^{-iAt}\,e^{-iBt}$は成立しません。標準的な対処法が**Trotter–Suzuki積公式**で、各項の短時間発展を交互に並べます。誤差は、ステップ幅を小さくすれば減り(Lie–Trotter、1次)、ステップを対称化すれば減り(Strang、2次)、対称ステップをSuzukiの構成で再帰的に入れ子にすればさらに減ります(任意の偶数次)。
