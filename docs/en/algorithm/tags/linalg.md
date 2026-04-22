---
title: "Tag: linalg"
tags: [linalg]
---

# `linalg`

Algorithm examples tagged **`linalg`** (1).

[← Back to all algorithms](../index.md)

---

**Browse by tag:** 
[`advanced`](./advanced.md) · [`built-in`](./built-in.md) · [`decomposition`](./decomposition.md) · [`from-scratch`](./from-scratch.md) · [`graph`](./graph.md) · [`graph-partition`](./graph-partition.md) · [`hamiltonian-simulation`](./hamiltonian-simulation.md) · [`intermediate`](./intermediate.md) · [`jijmodeling`](./jijmodeling.md) · [`maxcut`](./maxcut.md) · [`optimization`](./optimization.md) · [`pauli`](./pauli.md) · [`qaoa`](./qaoa.md) · [`simulation`](./simulation.md) · [`suzuki`](./suzuki.md) · [`time-evolution`](./time-evolution.md) · [`trotter`](./trotter.md) · [`variational`](./variational.md)

---

### [From a Hermitian Matrix to a Quantum Circuit](../hermitian_decomposition.ipynb)

**Tags:** [`hamiltonian-simulation`](./hamiltonian-simulation.md) [`pauli`](./pauli.md) [`decomposition`](./decomposition.md) [`time-evolution`](./time-evolution.md) [`intermediate`](./intermediate.md)

In many quantum algorithms you start from a **Hermitian matrix** — a Hamiltonian given as a dense $2^n \times 2^n$ numpy array — and you want to simulate its time evolution $e^{-iHt}$ on a quantum computer. The standard path is:
