---
title: "Tag: hamiltonian-simulation"
tags: [hamiltonian-simulation]
---

# `hamiltonian-simulation`

Algorithm examples tagged **`hamiltonian-simulation`** (2).

[← Back to all algorithms](../index.md)

---

**Browse by tag:** 
[`advanced`](./advanced.md) · [`built-in`](./built-in.md) · [`chemistry`](./chemistry.md) · [`decomposition`](./decomposition.md) · [`from-scratch`](./from-scratch.md) · [`graph`](./graph.md) · [`graph-partition`](./graph-partition.md) · [`ground-state`](./ground-state.md) · [`intermediate`](./intermediate.md) · [`jijmodeling`](./jijmodeling.md) · [`linalg`](./linalg.md) · [`maxcut`](./maxcut.md) · [`openfermion`](./openfermion.md) · [`optimization`](./optimization.md) · [`pauli`](./pauli.md) · [`qaoa`](./qaoa.md) · [`simulation`](./simulation.md) · [`suzuki`](./suzuki.md) · [`time-evolution`](./time-evolution.md) · [`trotter`](./trotter.md) · [`variational`](./variational.md) · [`vqe`](./vqe.md)

---

### [From a Hermitian Matrix to a Quantum Circuit](../hermitian_decomposition.ipynb)

**Tags:** [`pauli`](./pauli.md) [`decomposition`](./decomposition.md) [`time-evolution`](./time-evolution.md) [`linalg`](./linalg.md) [`intermediate`](./intermediate.md)

In many quantum algorithms you start from a **Hermitian matrix** — a Hamiltonian given as a dense $2^n \times 2^n$ numpy array — and you want to simulate its time evolution $e^{-iHt}$ on a quantum computer. The standard path is:

### [Hamiltonian Simulation with Suzuki–Trotter](../hamiltonian_simulation.ipynb)

**Tags:** [`trotter`](./trotter.md) [`suzuki`](./suzuki.md) [`simulation`](./simulation.md) [`time-evolution`](./time-evolution.md) [`pauli`](./pauli.md) [`advanced`](./advanced.md)

Simulating the time evolution $e^{-iHt}$ of a quantum system is one of the canonical applications of a quantum computer. When the Hamiltonian splits into non-commuting pieces $H = A + B$ with $[A, B] \neq 0$, the naive factorisation $e^{-i(A+B)t} = e^{-iAt}\,e^{-iBt}$ is wrong; the standard fix is **Trotter–Suzuki product formulas**, which interleave short evolutions of each piece. The error decreases as we take smaller steps (Lie–Trotter, first order), symmetrise the step (Strang, second order), or nest the symmetric step recursively via Suzuki's construction (any even order).
